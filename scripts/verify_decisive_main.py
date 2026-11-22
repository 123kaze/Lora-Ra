"""Independent NumPy/stdlib reconstruction of stage-0 statistics, spectra and logs."""
import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

from verify_output_diagonal import solve

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def rows(path):
    with path.open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def close(a, b):
    assert math.isclose(float(a), float(b), abs_tol=1e-12, rel_tol=1e-12), (a, b)


def verify(bundle):
    manifest = read(bundle/'artifact-manifest.json')
    for item in manifest['files']:
        path = (ROOT/item['path']).resolve()
        assert path.is_relative_to(ROOT)
        value = path.read_bytes()
        assert len(value) == item['bytes'] and hashlib.sha256(value).hexdigest() == item['sha256'], item['path']
    cfg, links = read(bundle/'config.json'), read(bundle/'worker_links.json')
    raw = rows(bundle/'main.csv')
    assert len(raw) == 1440
    grouped, traces = defaultdict(list), defaultdict(set)
    row_lookup = {}
    for row in raw:
        key = (row['profile'], int(row['base_seed']), int(row['budget']), row['method'], int(row['step']))
        grouped[key].append(float(row['val_loss']))
        row_lookup[(*key, int(row['adapter_seed']))] = row
        assert int(row['parameters']) == int(row['budget'])
        traces[(row['adapter_seed'], row['step'])].add(row['data_trace_sha256'])
    assert len(row_lookup) == len(raw) and all(len(v) == 1 for v in traces.values())
    means = {k: statistics.mean(v) for k, v in grouped.items()}
    assert all(len(v) == 2 for v in grouped.values())
    for row in rows(bundle/'by_base_summary.csv'):
        key = (row['profile'], int(row['base_seed']), int(row['budget']), row['method'], int(row['step']))
        close(row['val_loss_mean'], means[key])
        close(row['adapter_sample_std'], statistics.stdev(grouped[key]))
    for row in rows(bundle/'summary.csv'):
        values = []
        for seed in cfg['main_base_seeds']:
            key = row['profile'], seed, int(row['budget'])
            if row['method'] == 'ShuffledMean':
                values.append(statistics.mean(means[(*key, f'TwoSidedShuffle{s}', int(row['step']))] for s in cfg['shuffle_seeds']))
            else:
                values.append(means[(*key, row['method'], int(row['step']))])
        close(row['val_loss_mean'], statistics.mean(values))
        close(row['base_mean_sample_std'], statistics.stdev(values))
    observed = {}
    for row in rows(bundle/'paired_differences.csv'):
        profile, seed, budget, step = row['profile'], int(row['base_seed']), int(row['budget']), int(row['step'])
        actual = means[(profile, seed, budget, 'TwoSidedTrue', step)]
        full = means[(profile, seed, budget, 'FullBudgetBalanced', step)]
        random = [means[(profile, seed, budget, f'TwoSidedShuffle{s}', step)] for s in cfg['shuffle_seeds']]
        close(row['delta_full'], full-actual)
        close(row['delta_shuffle'], statistics.mean(random)-actual)
        observed[(profile, seed, budget, step)] = (full-actual, statistics.mean(random)-actual)
    columns, matrix, required = [], [], []
    for budget in cfg['budgets']:
        offset = len(columns)
        columns += [f'B{budget}_{x}' for x in ('long_full', 'long_shuffle', 'short_full', 'short_shuffle', 'age_full', 'age_shuffle')]
        required += [offset, offset+1, offset+4, offset+5]
    for seed in cfg['main_base_seeds']:
        values = []
        for budget in cfg['budgets']:
            long, short = observed[('long', seed, budget, 600)], observed[('short', seed, budget, 600)]
            values += [long[0], long[1], short[0], short[1], long[0]-short[0], long[1]-short[1]]
        matrix.append(values)
    matrix = np.asarray(matrix, dtype=np.float64)
    cfg_boot = cfg['bootstrap']
    generator = np.random.default_rng(cfg_boot['seed'])
    bootstrap = np.empty((cfg_boot['draws'], len(columns)))
    trace = hashlib.sha256()
    for start in range(0, cfg_boot['draws'], 10000):
        indices = generator.integers(0, len(matrix), size=(min(10000, cfg_boot['draws']-start), len(matrix)), dtype=np.int64)
        trace.update(indices.astype('<i8', copy=False).tobytes())
        bootstrap[start:start+len(indices)] = matrix[indices].mean(axis=1)
    alpha = (1-cfg_boot['confidence'])/2
    lower, upper = np.quantile(bootstrap, [alpha, 1-alpha], axis=0, method='linear')
    recorded = read(bundle/'bootstrap.json')
    assert recorded['column_names'] == columns and recorded['required_columns'] == required
    assert recorded['resample_index_sha256'] == trace.hexdigest()
    assert np.max(np.abs(np.asarray(recorded['mean'])-matrix.mean(axis=0))) < 1e-12
    assert np.max(np.abs(np.asarray(recorded['low'])-lower)) < 1e-12
    assert np.max(np.abs(np.asarray(recorded['high'])-upper)) < 1e-12
    verdict = read(bundle/'verdict.json')
    assert verdict['route_A_gate_pass'] == bool(all(lower[i] > 0 for i in required))
    assert verdict['route'] == ('A' if verdict['route_A_gate_pass'] else 'B')
    spectra_count, allocation_count, baseline_count, physical, alias, logs = 0, 0, 0, 0, 0, 0
    max_error = 0.
    coverage_rows = {(r['profile'], int(r['base_seed']), int(r['budget']), r['method']): r for r in rows(bundle/'allocation_summary.csv')}
    distance_rows = {(r['profile'], int(r['base_seed']), int(r['budget']), r['baseline']): r for r in rows(bundle/'allocation_distances.csv')}
    coverage_checked, tv_values = 0, []
    for profile, link in links.items():
        parent = bundle/link['packaged_path']
        audit = read(parent/'checkpoint_audit.json')
        assert audit['complete_run'] and audit['pass'] and audit['max_validation_reload_error'] == 0
        for child in sorted(parent.glob('*_base_*')):
            seed = int(child.name.rsplit('_', 1)[1])
            cfg_model = read(child/'config.json')['model']
            schemes = read(child/'groups.json')
            mapping = read(child/'calibration_arrays.json')
            arrays = np.load(child/'calibration_arrays.npz', allow_pickle=False)
            expected = {(r['scheme'], r['group'], int(r['rank'])): float(r['prefix_utility']) for r in rows(child/'spectra.csv')}
            assignments = read(child/'assignments.json')
            last_layer = len(cfg_model['widths'])-1
            for budget in cfg['budgets']:
                distributions = {}
                for name, item in assignments[str(budget)].items():
                    distribution = defaultdict(int)
                    for g in item['groups']:
                        distribution[g['module']] += g['cost']*item['ranks'][g['id']]
                    distributions[name] = distribution
                    row = coverage_rows[(profile, seed, budget, name)]
                    assert int(row['actual_parameters']) == sum(distribution.values()) == budget
                    assert int(row['total_rank']) == sum(item['ranks'].values())
                    assert int(row['nonzero_groups']) == sum(r > 0 for r in item['ranks'].values())
                    assert int(row['candidate_groups']) == len(item['groups'])
                    assert int(row['nonzero_modules']) == sum(v > 0 for v in distribution.values())
                    assert int(row['candidate_modules']) == len(distribution)
                    coverage_checked += 1
                for name in ('FullBudgetBalanced', 'GroupedBudgetBalanced'):
                    a, b = distributions['TwoSidedTrue'], distributions[name]
                    tv = sum(abs(a[k]-b[k]) for k in set(a)|set(b))/(2*budget)
                    close(distance_rows[(profile, seed, budget, name)]['module_parameter_tv'], tv)
                    tv_values.append(tv)
                for name in ('FullBudgetBalanced', 'GroupedBudgetBalanced', 'WidthRankProportional'):
                    item = assignments[str(budget)][name]
                    candidates = item['groups']
                    target = budget/sum(g['cost'] for g in candidates)
                    utility_b = {g['id']: [-(r-target)**2 for r in range(cfg['max_rank']+1)] for g in candidates}
                    assert solve(candidates, utility_b, budget, cfg['max_rank']) == item['ranks']
                    assert sum(g['cost']*item['ranks'][g['id']] for g in candidates) == budget
                    if name == 'FullBudgetBalanced' and budget == 2464:
                        assert len(candidates) == 7 and set(item['ranks'].values()) == {1}
                    baseline_count += 1
                assert assignments[str(budget)]['GroupedBudgetBalanced']['ranks'] == assignments[str(budget)]['WidthRankProportional']['ranks']
            for scheme, groups in schemes.items():
                utility = {}
                assert len(groups) == 14 and all(g['module'].startswith(f'blocks.{last_layer}.') for g in groups)
                rng = None if scheme == 'TwoSidedTrue' else np.random.default_rng(int(scheme.removeprefix('TwoSidedShuffle')))
                for start in range(0, len(groups), 2):
                    a, b = groups[start:start+2]
                    assert a['module'] == b['module'] and sorted(a['indices']+b['indices']) == list(range(128))
                    expected_indices = list(range(128)) if rng is None else rng.permutation(128).tolist()
                    assert a['indices'] == sorted(expected_indices[:64]) and b['indices'] == sorted(expected_indices[64:])
                for group in groups:
                    keys = mapping[group['module']]
                    G, hi, ho = [arrays[keys[k]] for k in ('G', 'Hin', 'Hout')]
                    assert group['axis'] == ('row' if group['module'].endswith(('o_proj', '.down')) else 'column')
                    shape = (len(group['indices']), G.shape[1]) if group['axis'] == 'row' else (G.shape[0], len(group['indices']))
                    assert shape == (group['m'], group['n']) and group['cost'] == sum(shape)
                    matrix_g = G/np.sqrt(ho+cfg['damping_out']*ho.mean())[:, None]/np.sqrt(hi+cfg['damping_in']*hi.mean())[None, :]
                    block = matrix_g[:, group['indices']] if group['axis'] == 'column' else matrix_g[group['indices'], :]
                    singular = np.linalg.svd(block, compute_uv=False)[:cfg['max_rank']]
                    prefix = np.r_[0., np.cumsum(.5*singular**2)]
                    utility[group['id']] = prefix
                    for rank in range(1, len(prefix)):
                        max_error = max(max_error, abs(prefix[rank]-expected[(scheme, group['id'], rank)]))
                        spectra_count += 1
                for budget in cfg['budgets']:
                    answer = solve(groups, utility, budget, cfg['max_rank'])
                    assert answer == assignments[str(budget)][scheme]['ranks']
                    allocation_count += 1
            arrays.close()
            for result_path in sorted(child.glob('adapters/*/result.json')):
                result = read(result_path)
                assert result['base_frozen'] and result['zero_initial_error'] == 0
                assert result['cache_full_max_error'] == 0 and result['merge_max_error'] < 1e-4
                if result['reused_from']:
                    source_name = Path(result['reused_from']).relative_to(Path(link['source_run']))
                    source = read(parent/source_name)
                    assert not source['reused_from'] and source['configuration_sha256'] == result['configuration_sha256']
                    for a, b in zip(source['evaluations'], result['evaluations']):
                        for field in ('profile', 'base_seed', 'adapter_seed', 'budget', 'step', 'val_loss', 'val_accuracy', 'data_trace_sha256'):
                            assert a[field] == b[field]
                    alias += 1
                else:
                    normalized = result['log_path'].replace('\\', '/')
                    relative_log = normalized.split(link['source_run']+'/', 1)[1]
                    events = [json.loads(line) for line in (parent/relative_log).read_text(encoding='utf-8').splitlines()]
                    for event in events:
                        if event['stage'] != 'decisive_validation':
                            continue
                        key = (profile, seed, event['budget'], event['method'], event['step'], event['adapter_seed'])
                        cell = row_lookup[key]
                        close(cell['val_loss'], event['val_loss'])
                        assert cell['data_trace_sha256'] == event['data_trace_sha256']
                        logs += 1
                    physical += 1
                for event in result['evaluations']:
                    key = (profile, seed, event['budget'], event['method'], event['step'], event['adapter_seed'])
                    close(row_lookup[key]['val_loss'], event['val_loss'])
                    assert event['parameters'] == event['budget']
        assert audit['actual_checkpoints_reloaded'] == sum(not read(p)['reused_from'] for p in parent.glob('*_base_*/adapters/*/result.json'))
    assert max_error < 1e-10
    assert physical == verdict['actual_optimizer_trajectories'] and physical+alias == 720 and logs == physical*2
    assert coverage_checked == len(coverage_rows) == 360 and len(tv_values) == len(distance_rows) == 80
    report = {'files_verified': len(manifest['files']), 'logical_conditions': 720,
              'actual_optimizer_trajectories': physical, 'explicit_alias_conditions': alias,
              'logical_validation_rows_recomputed': len(raw), 'physical_validation_logs_matched': logs,
              'bootstrap_draws_recomputed': cfg_boot['draws'], 'bootstrap_indices_sha256': trace.hexdigest(),
              'prefix_utilities_recomputed': spectra_count, 'max_numpy_prefix_error': max_error,
              'adaptive_allocations_recomputed': allocation_count, 'baseline_allocations_recomputed': baseline_count,
              'random_partitions_reconstructed': True, 'route': verdict['route'],
              'allocation_coverage_rows_recomputed': coverage_checked,
              'allocation_distance_rows_recomputed': len(tv_values),
              'minimum_true_vs_balanced_module_parameter_tv': min(tv_values),
              'requires_torch': False, 'pass': True}
    print(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle', type=Path)
    verify(parser.parse_args().bundle.resolve())
