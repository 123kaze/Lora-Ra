"""Combine completed audited shards and apply the predeclared paired A/B rule."""
import argparse
import csv
import hashlib
import json
import shutil
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

from carryrank.paired_stats import all_positive_bounds, paired_percentile_bootstrap

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def rows(path):
    with path.open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def csv_write(path, values):
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(values[0]))
        writer.writeheader(); writer.writerows(values)


def analyze(shards, cfg, output):
    if output.exists() or output.parent != ROOT/'runs':
        raise ValueError('Choose a fresh direct runs/<analysis-id>')
    raw, conditions, source_info = [], [], {}
    parent_hash = hashlib.sha256((ROOT/'configs/decisive_stage0.json').read_bytes()).hexdigest()
    for shard in shards:
        completion = read(shard/'completion.json')
        audit = read(shard/'checkpoint_audit.json')
        config = read(shard/'config.json')
        identity = read(shard/'identity.json')
        assert identity['protocol_sha256'] == hashlib.sha256((ROOT/'plan/decisive-stage0-protocol-v1.md').read_bytes()).hexdigest()
        assert completion['complete'] and not completion['smoke'] and completion['phase'] == 'main'
        assert audit['complete_run'] and audit['pass'] and audit['max_validation_reload_error'] == 0
        assert audit['actual_checkpoints_reloaded'] == completion['actual_optimizer_trajectories']
        assert audit['logical_conditions_verified'] == completion['logical_conditions']
        assert config['parent_config_sha256'] == parent_hash
        assert config['main_base_seeds'] == cfg['main_base_seeds'] and config['adapter_seeds'] == cfg['adapter_seeds']
        assert config['budgets'] == cfg['budgets'] and config['scope'] == 'matched_final'
        assert len(config['widths']) == 1
        profile = next(iter(config['widths']))
        assert profile not in source_info
        source_info[profile] = {'source_run': shard.relative_to(ROOT).as_posix(),
                                'packaged_path': f'shards/{profile}', 'completion': completion,
                                'audit_summary': {k: v for k, v in audit.items() if k != 'records'}}
        raw.extend(rows(shard/'main.csv'))
        conditions.extend(rows(shard/'conditions.csv'))
    assert set(source_info) == set(cfg['widths'])
    assert len(raw) == 1440 and len(conditions) == 720
    keys = [(r['profile'], r['base_seed'], r['adapter_seed'], r['budget'], r['method'], r['step']) for r in raw]
    assert len(set(keys)) == len(raw)
    assert {int(r['base_seed']) for r in raw} == set(cfg['main_base_seeds'])
    assert all(int(r['parameters']) == int(r['budget']) and int(r['val_bytes']) == 65536 for r in raw)
    traces = defaultdict(set)
    grouped = defaultdict(list)
    for row in raw:
        traces[(row['adapter_seed'], row['step'])].add(row['data_trace_sha256'])
        grouped[(row['profile'], int(row['base_seed']), int(row['budget']), row['method'], int(row['step']))].append(float(row['val_loss']))
    assert all(len(v) == 1 for v in traces.values())
    means = {}
    by_base = []
    for (profile, seed, budget, method, step), values in sorted(grouped.items()):
        assert len(values) == 2
        means[(profile, seed, budget, method, step)] = statistics.mean(values)
        by_base.append({'profile': profile, 'base_seed': seed, 'budget': budget, 'method': method, 'step': step,
                        'val_loss_mean': statistics.mean(values), 'adapter_sample_std': statistics.stdev(values)})
    methods = ['FullBudgetBalanced', 'GroupedBudgetBalanced', 'TwoSidedTrue'] + [f'TwoSidedShuffle{s}' for s in cfg['shuffle_seeds']] + ['WidthRankProportional']
    summary, differences = [], []
    for profile in cfg['widths']:
        for budget in cfg['budgets']:
            for step in cfg['eval_steps']:
                for method in methods + ['ShuffledMean']:
                    values = []
                    for seed in cfg['main_base_seeds']:
                        if method == 'ShuffledMean':
                            value = statistics.mean(means[(profile, seed, budget, f'TwoSidedShuffle{s}', step)] for s in cfg['shuffle_seeds'])
                        else:
                            value = means[(profile, seed, budget, method, step)]
                        values.append(value)
                    summary.append({'profile': profile, 'budget': budget, 'method': method, 'step': step,
                                    'n_base_seeds': len(values), 'val_loss_mean': statistics.mean(values),
                                    'base_mean_sample_std': statistics.stdev(values)})
                for seed in cfg['main_base_seeds']:
                    actual = means[(profile, seed, budget, 'TwoSidedTrue', step)]
                    full = means[(profile, seed, budget, 'FullBudgetBalanced', step)]
                    random = [means[(profile, seed, budget, f'TwoSidedShuffle{s}', step)] for s in cfg['shuffle_seeds']]
                    rank = 1 + sum(v < actual-1e-12 for v in random) + .5*sum(abs(v-actual) <= 1e-12 for v in random)
                    differences.append({'profile': profile, 'base_seed': seed, 'budget': budget, 'step': step,
                        'true_loss': actual, 'full_loss': full, 'shuffled_mean_loss': statistics.mean(random),
                        'delta_full': full-actual, 'delta_shuffle': statistics.mean(random)-actual,
                        'true_rank_among_6': rank})
    lookup = {(r['profile'], r['base_seed'], r['budget'], r['step']): r for r in differences}
    column_names, matrix, required = [], [], []
    for budget in cfg['budgets']:
        offset = len(column_names)
        column_names += [f'B{budget}_{name}' for name in ('long_full', 'long_shuffle', 'short_full', 'short_shuffle', 'age_full', 'age_shuffle')]
        required += [offset, offset+1, offset+4, offset+5]
    for seed in cfg['main_base_seeds']:
        vector = []
        for budget in cfg['budgets']:
            long = lookup[('long', seed, budget, cfg['steps'])]
            short = lookup[('short', seed, budget, cfg['steps'])]
            vector += [long['delta_full'], long['delta_shuffle'], short['delta_full'], short['delta_shuffle'],
                       long['delta_full']-short['delta_full'], long['delta_shuffle']-short['delta_shuffle']]
        matrix.append(vector)
    bootstrap = paired_percentile_bootstrap(np.array(matrix), draws=cfg['bootstrap']['draws'],
        seed=cfg['bootstrap']['seed'], confidence=cfg['bootstrap']['confidence'])
    bootstrap['column_names'] = column_names
    bootstrap['required_columns'] = required
    route_a = all_positive_bounds(bootstrap, required)
    output.mkdir(parents=True)
    for profile, source in source_info.items():
        shard = ROOT/source['source_run']
        shutil.copytree(shard, output/source['packaged_path'], ignore=shutil.ignore_patterns('*.pt', '__pycache__', '*.pyc'))
    csv_write(output/'main.csv', raw)
    csv_write(output/'conditions.csv', conditions)
    csv_write(output/'by_base_summary.csv', by_base)
    csv_write(output/'summary.csv', summary)
    csv_write(output/'paired_differences.csv', differences)
    csv_write(output/'bootstrap_input.csv', [{'base_seed': seed, **dict(zip(column_names, vector))}
        for seed, vector in zip(cfg['main_base_seeds'], matrix)])
    csv_write(output/'confidence_intervals.csv', [{'contrast': name, 'mean': bootstrap['mean'][i],
        'low': bootstrap['low'][i], 'high': bootstrap['high'][i], 'required_for_A': i in required,
        'positive_lower_bound': bootstrap['low'][i] > 0} for i, name in enumerate(column_names)])
    write(output/'bootstrap.json', bootstrap)
    write(output/'config.json', cfg)
    write(output/'worker_links.json', source_info)
    shutil.copy2(ROOT/'plan/decisive-stage0-protocol-v1.md', output/'protocol.md')
    verdict = {'route': 'A' if route_a else 'B', 'route_A_gate_pass': route_a,
        'rule': cfg['route_A_rule'], 'required_contrasts': [column_names[i] for i in required],
        'all_required_lower_bounds_positive': route_a, 'independent_bases_per_width': 10,
        'logical_conditions': len(conditions), 'actual_optimizer_trajectories': sum(not r['reused_from'] for r in conditions),
        'validation_rows': len(raw), 'test_split_used': False, 'paired_traces_match': True,
        'all_shard_checkpoint_audits_pass': True, 'not_an_equivalence_test': True,
        'B_next_action_if_applicable': 'Literature/code audit before choosing invariance or negative-results direction'}
    write(output/'verdict.json', verdict)
    print(json.dumps({'output': str(output), 'verdict': verdict,
                      'intervals': read_csv_for_print(output/'confidence_intervals.csv')}, indent=2))


def read_csv_for_print(path):
    return rows(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shards', type=Path, nargs=2, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/decisive_stage0.json')
    args = parser.parse_args()
    analyze([p.resolve() for p in args.shards], read(args.config), args.output.resolve())
