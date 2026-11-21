"""Independent NumPy/stdlib audit of the exported label experiment, no torch."""
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


def verify(bundle):
    manifest = read(bundle / 'artifact-manifest.json')
    for item in manifest['files']:
        path = (ROOT / item['path']).resolve()
        assert path.is_relative_to(ROOT)
        value = path.read_bytes()
        assert len(value) == item['bytes'] and hashlib.sha256(value).hexdigest() == item['sha256'], item['path']
    cfg = read(bundle / 'config.json')
    data = rows(bundle / 'main.csv')
    expected_units = len(cfg['widths']) * len(cfg['base_seeds']) * len(cfg['adapter_seeds']) * 7
    assert len(data) == expected_units * len(cfg['eval_steps']) == 168
    means, grouped = {}, defaultdict(list)
    traces = defaultdict(set)
    for row in data:
        grouped[(row['profile'], int(row['base_seed']), row['method'], int(row['step']))].append(float(row['val_loss']))
        traces[(int(row['adapter_seed']), int(row['step']))].add(row['data_trace_sha256'])
        assert int(row['budget']) == cfg['budget'] == 384
        assert int(row['val_bytes']) == 65536
    assert all(len(t) == 1 for t in traces.values())
    for key, values in grouped.items():
        assert len(values) == len(cfg['adapter_seeds'])
        means[key] = statistics.mean(values)
    for row in rows(bundle / 'by_base_summary.csv'):
        key = (row['profile'], int(row['base_seed']), row['method'], int(row['step']))
        assert math.isclose(means[key], float(row['val_loss_mean']), abs_tol=1e-12)
        assert math.isclose(statistics.stdev(grouped[key]), float(row['val_loss_adapter_sample_std']), abs_tol=1e-12)
    for row in rows(bundle / 'summary.csv'):
        profile, method, step = row['profile'], row['method'], int(row['step'])
        values = []
        for seed in cfg['base_seeds']:
            if method == 'ShuffledMean':
                values.append(statistics.mean(means[(profile, seed, f'shuffle_{s}', step)] for s in cfg['shuffle_seeds']))
            else:
                values.append(means[(profile, seed, method, step)])
        assert math.isclose(statistics.mean(values), float(row['val_loss_mean']), abs_tol=1e-12)
        assert math.isclose(statistics.stdev(values), float(row['val_loss_base_sample_std']), abs_tol=1e-12)
    verdict = read(bundle / 'verdict.json')
    for profile in cfg['widths']:
        real = [means[(profile, seed, 'true', cfg['primary_step'])] for seed in cfg['base_seeds']]
        shuffled = [statistics.mean(means[(profile, seed, f'shuffle_{s}', cfg['primary_step'])]
                                    for s in cfg['shuffle_seeds']) for seed in cfg['base_seeds']]
        fixed = [means[(profile, seed, 'VBalanced', cfg['primary_step'])] for seed in cfg['base_seeds']]
        for contrast, other in [('shuffled_minus_true', shuffled), ('VBalanced_minus_true', fixed)]:
            difference = [a - b for a, b in zip(other, real)]
            expected = verdict['by_profile'][profile][contrast]
            assert expected['paired_base_differences'] == difference
            assert expected['mean'] == statistics.mean(difference)
            assert expected['sample_std'] == statistics.stdev(difference)
            assert expected['pass'] == (all(x > 0 for x in difference) and statistics.mean(difference) >= cfg['minimum_label_advantage'])
    max_error, spectra_count, dp_count, log_count = 0., 0, 0, 0
    for profile in cfg['widths']:
        for seed in cfg['base_seeds']:
            child = bundle / f'{profile}_base_{seed}'
            groups = read(child / 'groups.json')
            mapping = read(child / 'calibration_arrays.json')
            arrays = np.load(child / 'calibration_arrays.npz', allow_pickle=False)
            expected = {(r['scheme'], r['group'], int(r['rank'])): float(r['prefix_utility']) for r in rows(child / 'spectra.csv')}
            assignments = read(child / 'allocations.json')
            for scheme, candidates in groups.items():
                utilities = {}
                for module in mapping:
                    selected = [g for g in candidates if g['module'] == module]
                    assert len(selected) == 2
                    assert sorted(selected[0]['indices'] + selected[1]['indices']) == list(range(128))
                for group in candidates:
                    keys = mapping[group['module']]
                    hi, ho = arrays[keys['Hin']], arrays[keys['Hout']]
                    matrix = arrays[keys['G']][:, group['indices']]
                    matrix = matrix / np.sqrt(ho + cfg['damping_out'] * ho.mean())[:, None]
                    matrix = matrix / np.sqrt((hi + cfg['damping_in'] * hi.mean())[group['indices']])
                    sigma = np.linalg.svd(matrix, compute_uv=False)[:cfg['max_rank']]
                    utility = np.r_[0., np.cumsum(.5 * sigma**2)]
                    utilities[group['id']] = utility
                    for rank in range(1, len(utility)):
                        max_error = max(max_error, abs(utility[rank] - expected[(scheme, group['id'], rank)]))
                        spectra_count += 1
                answer = solve(candidates, utilities, cfg['budget'], cfg['max_rank'])
                assert assignments[scheme]['ranks'] == answer
                dp_count += 1
            arrays.close()
            for path in child.glob('logs/*/attempt_*.jsonl'):
                for line in path.read_text(encoding='utf-8').splitlines():
                    item = json.loads(line)
                    if item['stage'] != 'diagnostic_validation':
                        continue
                    matching = [r for r in data if r['profile'] == profile and all(str(r[k]) == str(item[k])
                                for k in ('base_seed', 'adapter_seed', 'method', 'step'))]
                    assert len(matching) == 1 and float(matching[0]['val_loss']) == item['val_loss']
                    assert matching[0]['data_trace_sha256'] == item['data_trace_sha256']
                    log_count += 1
    audit = read(bundle / 'checkpoint_audit.json')
    assert audit['pass'] and audit['checkpoints_reloaded'] == expected_units
    assert max_error < 1e-10 and log_count == len(data)
    result = {'files_verified': len(manifest['files']), 'trajectories': expected_units,
              'validation_rows_and_raw_logs_matched': log_count, 'prefix_utilities_recomputed': spectra_count,
              'max_numpy_spectrum_error': max_error, 'exact_allocations_recomputed': dp_count,
              'base_level_summaries_and_gates_recomputed': True, 'requires_torch': False, 'pass': True}
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle', type=Path)
    verify(parser.parse_args().bundle.resolve())
