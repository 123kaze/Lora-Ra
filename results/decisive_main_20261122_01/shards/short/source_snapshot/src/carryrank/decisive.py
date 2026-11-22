"""Stage-0 discovery capacity, prospective labels and single-group probes."""
from __future__ import annotations

import argparse
import copy
import json
import shutil
import statistics
import sys

import numpy as np
import torch

from .activation_stats import collect_diagonals
from .allocation import allocate
from .capacity import dataset
from .curvature import spectral_utility, whiten
from .decisive_training import create_experiment, train_adapter
from .experiment import ROOT, base_digest, digest, read_json, write_csv, write_json
from .preflight import full_groups, grouped, integer_rank_balanced, target_specifications
from .prefix_cache import build_training_cache, build_validation_cache


def prepare_base(run, cfg, profile, seed, splits):
    child = run / f'{profile}_base_{seed}'
    child.mkdir(exist_ok=True)
    base_cfg = read_json(ROOT / 'configs/pilot.json')
    base_cfg['name'] = f'decisive_{profile}_{seed}'
    base_cfg['model']['widths'] = cfg['widths'][profile]
    base_cfg['pretrain']['seed'] = seed
    base_cfg['runtime']['log_every'] = 120
    base_cfg['evaluation']['max_blocks'] = 4 if cfg['smoke'] else cfg['evaluation_max_blocks']
    write_json(child / 'config.json', base_cfg)
    if seed in cfg['discovery_base_seeds'] and not (child / 'base.pt').exists():
        source = ROOT / cfg['discovery_parent'] / f'{profile}_base_{seed}'
        for name in ('base.pt', 'pretrain.json'):
            shutil.copy2(source / name, child / name)
        write_json(child / 'inherited_base.json', {'source': source.relative_to(ROOT).as_posix(),
                   'base_file_sha256': digest((child / 'base.pt').read_bytes())})
    experiment = create_experiment(base_cfg, child, splits)
    experiment.pretrain()
    model, base = experiment.load_base()
    assert base['summary']['pretraining_gate']
    model.requires_grad_(False)
    return experiment, model, base


def assignments_for_base(experiment, model, cfg, phase, resume):
    destination = experiment.output / 'assignments.json'
    if destination.exists() and resume:
        return read_json(destination)
    targets = target_specifications(model, 'matched_final')
    true, full = grouped(targets), full_groups(targets)
    result = {}
    if phase == 'capacity':
        for budget in cfg['capacity_budgets']:
            ranks, spent, _ = integer_rank_balanced(full, budget, cfg['max_rank'])
            assert spent == budget
            result[str(budget)] = {'FullBudgetBalanced': {'groups': full, 'ranks': ranks}}
    else:
        stats, audit = collect_diagonals(model, targets, experiment.splits['target_train'],
            experiment.model_config.seq_len, cfg['calibration']['batch_size'], cfg['calibration']['batches'],
            cfg['calibration']['seed'], experiment.device)
        matrices = {name: whiten(v['G'], v['Hin'], v['Hout'], cfg['damping_in'], cfg['damping_out'])
                    for name, v in stats.items()}
        portable, mapping = {}, {}
        for i, target in enumerate(targets):
            name = target['module']
            mapping[name] = {}
            for key in ('G', 'Hin', 'Hout'):
                field = f'm{i}_{key}'
                portable[field] = stats[name][key].numpy()
                mapping[name][key] = field
        np.savez_compressed(experiment.output/'calibration_arrays.npz', **portable)
        write_json(experiment.output/'calibration_arrays.json', mapping)
        write_json(experiment.output/'calibration_audit.json', audit)
        schemes = {'TwoSidedTrue': true, **{f'TwoSidedShuffle{s}': grouped(targets, s) for s in cfg['shuffle_seeds']}}
        utilities, spectral = {}, []
        for scheme, groups in schemes.items():
            utilities[scheme] = {}
            for group in groups:
                matrix = matrices[group['module']]
                matrix = matrix[:, group['indices']] if group['axis'] == 'column' else matrix[group['indices'], :]
                sigma, utility = spectral_utility(matrix, cfg['max_rank'])
                utilities[scheme][group['id']] = utility
                for rank, singular in enumerate(sigma.tolist(), 1):
                    spectral.append({'scheme': scheme, 'group': group['id'], 'axis': group['axis'],
                                     'cost': group['cost'], 'rank': rank, 'singular_value': singular,
                                     'prefix_utility': utility[rank]})
        write_csv(experiment.output/'spectra.csv', spectral)
        write_json(experiment.output/'groups.json', schemes)
        if phase == 'mechanism':
            for group in true:
                budget = group['cost']
                name = 'probe_' + group['id'].replace('.', '_').replace(':', '_')
                result.setdefault(str(budget), {})[name] = {'groups': [group], 'ranks': {group['id']: 1},
                    'group_id': group['id'], 'predicted_utility': utilities['TwoSidedTrue'][group['id']][1]}
        else:
            for budget in cfg['budgets']:
                methods = {}
                for name, groups in [('FullBudgetBalanced', full), ('GroupedBudgetBalanced', true)]:
                    ranks, spent, _ = integer_rank_balanced(groups, budget, cfg['max_rank'])
                    assert spent == budget
                    methods[name] = {'groups': groups, 'ranks': ranks}
                for name, groups in schemes.items():
                    ranks, spent, score = allocate(groups, utilities[name], budget, cfg['max_rank'])
                    assert spent == budget
                    methods[name] = {'groups': groups, 'ranks': ranks, 'predicted_utility': score}
                ranks, spent, _ = integer_rank_balanced(true, budget, cfg['max_rank'], width_proportional=True)
                assert ranks == methods['GroupedBudgetBalanced']['ranks'] and spent == budget
                methods['WidthRankProportional'] = {'groups': true, 'ranks': ranks, 'definitionally_equivalent_to': 'GroupedBudgetBalanced'}
                result[str(budget)] = methods
    write_json(destination, result)
    return result


def capacity_gate(rows, cfg):
    if cfg['smoke']:
        return {'pass': False, 'smoke_ineligible': True}
    outcomes = []
    for profile in cfg['widths']:
        for low, high in zip(cfg['capacity_budgets'][:-1], cfg['capacity_budgets'][1:]):
            differences = []
            for seed in cfg['discovery_base_seeds']:
                means = {}
                for budget in (low, high):
                    values = [r['val_loss'] for r in rows if r['profile'] == profile and r['base_seed'] == seed
                              and r['budget'] == budget and r['step'] == cfg['steps']]
                    assert len(values) == len(cfg['adapter_seeds'])
                    means[budget] = statistics.mean(values)
                differences.append(means[low]-means[high])
            outcomes.append({'profile': profile, 'low_budget': low, 'high_budget': high,
                'paired_base_mean_gains': differences, 'mean': statistics.mean(differences),
                'pass': all(v > 0 for v in differences) and statistics.mean(differences) >= cfg['capacity_gate']['minimum_mean_gain_each_interval']})
    return {'pass': all(r['pass'] for r in outcomes), 'outcomes': outcomes,
            'scope': 'matched_final_seven_types', 'rule': cfg['capacity_gate']}


def run(output, config, phase, resume=False):
    if output.parent != ROOT/'runs':
        raise ValueError('Use direct runs/<run-id>')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    splits, data = dataset(read_json(ROOT/'configs/pilot.json'))
    proof = None
    if not config['smoke']:
        proof = read_json(ROOT/'reports/decisive_prefix_equivalence.json')
        assert proof['pass'] and proof['torch'] == torch.__version__
        assert proof['prefix_cache_source_sha256'] == digest((ROOT/'src/carryrank/prefix_cache.py').read_bytes())
        assert proof['axis_lora_source_sha256'] == digest((ROOT/'src/carryrank/axis_lora.py').read_bytes())
    if phase == 'main' and not config['smoke']:
        if not read_json(ROOT/config['capacity_run']/'capacity_gate.json')['pass']:
            raise ValueError('Main run rejected: expanded-target capacity gate did not pass')
    paths = sorted((ROOT/'src').rglob('*.py'))
    source = {p.relative_to(ROOT).as_posix(): digest(p.read_bytes()) for p in paths}
    identity = {'config': config, 'phase': phase, 'source': source, 'data': data,
                'protocol_sha256': digest((ROOT/'plan/decisive-stage0-protocol-v1.md').read_bytes()),
                'prefix_equivalence_proof': proof,
                'python': sys.version, 'torch': torch.__version__, 'numpy': np.__version__,
                'device': torch.cuda.get_device_name(0)}
    if output.exists():
        if not resume:
            raise FileExistsError(output)
        previous = read_json(output/'identity.json')
        for name, expected in previous['source'].items():
            assert digest((ROOT/name).read_bytes()) == expected, name
        assert {k: v for k, v in identity.items() if k != 'source'} == {k: v for k, v in previous.items() if k != 'source'}
        if (output/'completion.json').exists():
            print(read_json(output/'completion.json'), flush=True)
            return
    else:
        if resume:
            raise FileNotFoundError(output)
        output.mkdir(parents=True)
        write_json(output/'identity.json', identity)
        write_json(output/'config.json', config)
        write_json(output/'data_manifest.json', data)
        shutil.copy2(ROOT/'plan/decisive-stage0-protocol-v1.md', output/'protocol.md')
        for name in source:
            destination = output/'source_snapshot'/name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT/name, destination)
        prior = [ROOT/'研究方案.md'] + [p for p in (ROOT/'results').rglob('*') if p.is_file()]
        write_json(output/'prior_artifacts.json', {p.relative_to(ROOT).as_posix(): digest(p.read_bytes()) for p in prior})
    seeds = config['main_base_seeds'] if phase == 'main' else config['discovery_base_seeds']
    adapter_seeds = [11] if phase == 'mechanism' else config['adapter_seeds']
    rows, units = [], []
    for profile in config['widths']:
        for seed in seeds:
            experiment, base_model, base = prepare_base(output, config, profile, seed, splits)
            assignments = assignments_for_base(experiment, base_model, config, phase, resume)
            base_model.requires_grad_(False)
            validation_cache = build_validation_cache(experiment, base_model)
            for adapter_seed in adapter_seeds:
                all_done = resume and all((experiment.output/'adapters'/f'B{budget}_{method}_seed{adapter_seed}'/'result.json').exists()
                                         for budget, methods in assignments.items() for method in methods)
                train_cache = None
                if not all_done:
                    train_cache = build_training_cache(experiment, base_model, adapter_seed, config['steps'],
                                                        config['batch_size'], config['eval_steps'])
                    write_json(experiment.output/f'cache_seed{adapter_seed}.json',
                               {k: v for k, v in train_cache.items() if k not in ('hidden', 'targets', 'first_tokens')})
                aliases = {}
                for raw_budget, methods in assignments.items():
                    budget = int(raw_budget)
                    for method, assignment in methods.items():
                        result = train_adapter(experiment, config, profile, seed, adapter_seed, budget, method,
                            assignment['groups'], assignment['ranks'], train_cache, validation_cache, aliases, resume)
                        for row in result['evaluations']:
                            extra = {'group_id': assignment.get('group_id', ''),
                                     'predicted_utility': assignment.get('predicted_utility', '')}
                            rows.append({**row, **extra})
                        units.append({'profile': profile, 'base_seed': seed, 'adapter_seed': adapter_seed,
                            'budget': budget, 'method': method, 'reused_from': result['reused_from'],
                            'time_s': result['time_s'], 'zero_initial_error': result['zero_initial_error'],
                            'merge_max_error': result['merge_max_error'], 'cache_full_max_error': result['cache_full_max_error'],
                            'base_frozen': result['base_frozen']})
                        write_csv(output/'main.csv', rows)
                        write_csv(output/'conditions.csv', units)
                        write_json(output/'progress.json', {'phase': phase, 'completed_logical_conditions': len(units),
                            'actual_optimizer_trajectories': sum(not u['reused_from'] for u in units),
                            'current_profile': profile, 'current_base': seed, 'current_adapter': adapter_seed,
                            'current_method': method, 'current_budget': budget})
                del train_cache
                torch.cuda.empty_cache()
            del validation_cache, base_model
            torch.cuda.empty_cache()
    if phase == 'capacity':
        write_json(output/'capacity_gate.json', capacity_gate(rows, config))
    prior = read_json(output/'prior_artifacts.json')
    assert all(digest((ROOT/p).read_bytes()) == expected for p, expected in prior.items())
    assert all(u['base_frozen'] and u['zero_initial_error'] == 0 and u['cache_full_max_error'] == 0 and u['merge_max_error'] < 1e-4 for u in units)
    expected = len(config['widths'])*len(seeds)*len(adapter_seeds)*(18 if phase == 'main' else 14 if phase == 'mechanism' else 3)
    assert len(units) == expected and len(rows) == expected*len(config['eval_steps'])
    summary = {'phase': phase, 'smoke': config['smoke'], 'logical_conditions': len(units),
        'actual_optimizer_trajectories': sum(not u['reused_from'] for u in units), 'validation_rows': len(rows),
        'old_artifacts_verified': len(prior), 'old_artifacts_unchanged': True,
        'max_merge_error': max(u['merge_max_error'] for u in units), 'max_cache_full_error': 0,
        'complete': True, 'test_split_used': False}
    write_json(output/'completion.json', summary)
    print(json.dumps(summary), flush=True)


def main():
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/decisive_stage0.json')
    parser.add_argument('--phase', choices=['capacity', 'main', 'mechanism'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    cfg = read_json(args.config)
    if args.smoke:
        cfg.update(smoke=True, main_base_seeds=[2026], discovery_base_seeds=[2026], adapter_seeds=[11],
                   steps=3, eval_steps=[3])
    run(args.output.resolve(), cfg, args.phase, args.resume)


if __name__ == '__main__':
    main()
