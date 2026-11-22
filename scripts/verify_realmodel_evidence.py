"""Independently audit real-model results from predictions, calibration and hashes.

Uses only NumPy/stdlib. It never imports the analyzer and cannot reload model
checkpoints absent from the lightweight package; GPU audit records are evidence
to inspect, not a claim that this verifier executed model inference.
"""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
from itertools import combinations

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
USED = set()
METHODS = ('UniformLoRA', 'AdaLoRA', 'TwoSidedSpectral', 'CoverageFloor')
COMPARISONS = (
    ('uniform_minus_spectral', 'UniformLoRA', 'TwoSidedSpectral', True),
    ('spectral_minus_floor', 'TwoSidedSpectral', 'CoverageFloor', False),
    ('uniform_minus_adalora', 'UniformLoRA', 'AdaLoRA', False),
)


def read(path):
    USED.add(path.resolve())
    return json.loads(path.read_text(encoding='utf-8'))


def records(path):
    USED.add(path.resolve())
    with path.open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def assert_close(a, b, tolerance=1e-12):
    assert math.isclose(float(a), float(b), rel_tol=tolerance, abs_tol=tolerance), (a, b)


def resolve_pattern(raw, modules):
    ranks, seen = {module: 0 for module in modules}, set()
    for key, value in raw.items():
        matches = [module for module in modules if key == module or key.endswith('.'+module)]
        assert len(matches) == 1 and matches[0] not in seen
        assert not isinstance(value, bool) and int(value) == value
        ranks[matches[0]] = int(value); seen.add(matches[0])
    return ranks


def verify_predictions(path, size):
    data = records(path)
    assert len(data) == size
    identifiers = [row['example_id'] for row in data]
    assert len(set(identifiers)) == size
    labels = np.asarray([int(row['label']) for row in data])
    predictions = np.asarray([int(row['prediction']) for row in data])
    logits = np.asarray([[float(row['label0_logit']), float(row['label1_logit'])] for row in data])
    lp = np.asarray([float(row['target_logprob']) for row in data], dtype=np.float64)
    assert np.isin(labels, [0, 1]).all() and np.isin(predictions, [0, 1]).all()
    assert np.isfinite(logits).all() and np.isfinite(lp).all() and np.max(lp) <= 1e-6
    assert np.array_equal(predictions, np.argmax(logits, axis=1))
    true_positive = int(np.count_nonzero((predictions == 1) & (labels == 1)))
    denominator = int(np.count_nonzero(predictions == 1)+np.count_nonzero(labels == 1))
    metrics = {'nll': -float(lp.mean()), 'accuracy': float((predictions == labels).mean()),
               'f1': 2*true_positive/denominator if denominator else 0., 'examples': size}
    return metrics, list(zip(identifiers, labels.tolist()))


def verify_selection_lock(bundle):
    lock = read(bundle/'selection_lock.json')
    path = bundle/'selection.json'
    USED.add(path.resolve())
    assert lock['selection_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert lock['tuning_units'] == 24 and lock['official_validation_used_for_selection'] is False


def verify_checkpoint_records(audit, expected):
    checked = {row['result_path']: row for row in audit['records']}
    assert len(checked) == len(audit['records']) == len(expected)
    assert set(checked) == set(expected)
    for path, sha in expected.items():
        row = checked[path]
        assert row['checkpoint_sha256'] == sha and row['max_metric_reload_error'] == 0
        assert row['pass'] is True


def independent_dp(modules, utility, total, maximum, floor):
    # States contain a value and rank tuple. Enumerate natural layer order,
    # ascending previous spend and ascending new rank, retaining the first tie.
    best = {0: (0., ())}
    for name in modules:
        updated = {}
        for spent, (objective, previous) in best.items():
            for rank in range(floor, maximum+1):
                next_spend = spent+rank
                if next_spend > total:
                    break
                score = objective+utility[name][rank]
                if next_spend not in updated or score > updated[next_spend][0]:
                    updated[next_spend] = score, previous+(rank,)
        best = updated
    value, ranks = best[total]
    return dict(zip(modules, ranks)), float(value)


def verify_calibration(bundle, cfg):
    output, count, max_error, ties = {}, 0, 0., 0
    for task in cfg['tasks']:
        folder = bundle/'calibration'/task
        metadata = read(folder/'modules.json')
        names = sorted(metadata, key=lambda key: metadata[key]['index'])
        assert len(names) == 12 and {metadata[name]['index'] for name in names} == set(range(12))
        listed = records(folder/'spectra.csv')
        measured = {(r['module'], int(r['rank'])): r for r in listed}
        assert len(measured) == len(listed) == 12*cfg['rank_cap']
        utility = {}
        for name in names:
            spec = metadata[name]
            assert spec['shape'] == [2304, 768] and spec['cost'] == 3072
            path = (folder/spec['file']).resolve()
            assert path.is_relative_to(folder.resolve())
            USED.add(path)
            with np.load(path, allow_pickle=False) as arrays:
                assert set(arrays.files) >= {'G', 'Hin', 'Hout'}
                assert all(arrays[key].dtype == np.float32 for key in ('G', 'Hin', 'Hout'))
                gradient, hin, hout = [np.asarray(arrays[key], dtype=np.float64) for key in ('G', 'Hin', 'Hout')]
            assert gradient.shape == (2304, 768) and hin.shape == (768,) and hout.shape == (2304,)
            assert all(np.isfinite(a).all() for a in (gradient, hin, hout))
            assert hin.min() >= 0 and hout.min() >= 0 and hin.mean() > 0 and hout.mean() > 0
            matrix = gradient/np.sqrt(hout+cfg['damping']*hout.mean())[:, None]
            matrix = matrix/np.sqrt(hin+cfg['damping']*hin.mean())[None, :]
            spectrum = np.linalg.svd(matrix, compute_uv=False)[:cfg['rank_cap']]
            prefixes = np.r_[0., np.cumsum(spectrum**2)/2]
            utility[name] = prefixes
            for rank, sigma in enumerate(spectrum, 1):
                source = measured[(name, rank)]
                assert_close(source['sigma'], sigma, 1e-7)
                assert_close(source['prefix_utility'], prefixes[rank], 1e-7)
                max_error = max(max_error, abs(float(source['prefix_utility'])-float(prefixes[rank])))
                count += 1
        allocated = read(folder/'allocations.json')['methods']
        output[task] = {}
        for method, floor in (('UniformLoRA', 0), ('TwoSidedSpectral', 0), ('CoverageFloor', cfg['coverage_floor'])):
            ranks = resolve_pattern(allocated[method]['ranks'], names)
            assert sum(ranks.values()) == 48
            assert all(floor <= rank <= cfg['rank_cap'] for rank in ranks.values())
            if method == 'UniformLoRA':
                assert set(ranks.values()) == {cfg['uniform_rank']}
            else:
                answer, optimal = independent_dp(names, utility, 48, cfg['rank_cap'], floor)
                score = sum(utility[name][rank] for name, rank in ranks.items())
                assert_close(score, optimal, 1e-7)
                if answer != ranks:
                    ties += 1
            if 'predicted_utility' in allocated[method]:
                value = sum(utility[name][rank] for name, rank in ranks.items())
                assert_close(allocated[method]['predicted_utility'], value, 1e-7)
            output[task][method] = ranks
    return output, {'spectral_prefixes_recomputed': count, 'max_prefix_absolute_error': max_error,
                    'adaptive_DP_objectives_recomputed': 4,
                    'numerically_equivalent_optima_with_different_rank_tuple': ties}


def verify(bundle):
    USED.clear()
    assert bundle.is_relative_to(ROOT)
    manifest_path = bundle/'artifact-manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    hashed = set()
    for item in manifest['files']:
        path = (ROOT/item['path']).resolve()
        assert path.is_relative_to(ROOT) and path not in hashed
        value = path.read_bytes()
        assert len(value) == item['bytes'] and hashlib.sha256(value).hexdigest() == item['sha256'], item['path']
        hashed.add(path)
    cfg, complete = read(bundle/'config.json'), read(bundle/'completion.json')
    assert complete['complete'] and not complete.get('smoke', False)
    assert complete['test_used'] is False
    USED.add((bundle/'protocol.md').resolve())
    assert (bundle/'protocol.md').read_text(encoding='utf-8').strip()
    assert complete['main'] == 40 and complete['tuning'] == 24
    assert set(cfg['tasks']) == {'sst2', 'mrpc'} and set(cfg['methods']) == set(METHODS)
    assert len(cfg['main_seeds']) == 5 and cfg['tuning_seed'] not in cfg['main_seeds']
    assert cfg['bootstrap_draws'] == 200000 and cfg['bootstrap_seed'] == 95531
    verify_selection_lock(bundle)
    audit = read(bundle/'checkpoint_audit.json')
    assert audit['pass'] and audit['complete'] and audit['main_checkpoints_reloaded'] == 40
    assert len(audit['records']) == 40
    for record in audit['records']:
        if 'pass' in record:
            assert record['pass']
    assignments, calibration_report = verify_calibration(bundle, cfg)
    equivalent = []
    for task in cfg['tasks']:
        for first, second in combinations(('UniformLoRA', 'TwoSidedSpectral', 'CoverageFloor'), 2):
            if assignments[task][first] == assignments[task][second]:
                equivalent.append({'task': task, 'first_method': first, 'second_method': second,
                    'independently_retrained': True, 'additional_independent_mechanism_evidence': False})
    main, tuning = records(bundle/'main.csv'), records(bundle/'tuning.csv')
    assert len(main) == 40 and len(tuning) == 24
    published = {r['unit_id']: r for r in main+tuning}
    assert len(published) == 64
    values, traces, identities, bases = {}, defaultdict(set), {}, set()
    prediction_rows, largest_metric_error, max_compact_logit, max_compact_loss = 0, 0., 0., 0.
    history_steps, peak_effective_factor_parameters = 0, 0
    expected_audit = {}
    for path in sorted((bundle/'units').glob('*/result.json')):
        result = read(path)
        identifier = path.parent.name
        saved = published[identifier]
        phase, task, method, seed = result['phase'], result['task'], result['method'], int(result['seed'])
        assert phase in ('main', 'tuning') and task in cfg['tasks'] and method in METHODS
        assert result['steps'] == cfg['steps']
        assert seed in (cfg['main_seeds'] if phase == 'main' else [cfg['tuning_seed']])
        split = 'validation' if phase == 'main' else 'tune'
        assert result['evaluated_split'] == split
        number = {'sst2': 872, 'mrpc': 408}[task] if phase == 'main' else 256
        measured, identity = verify_predictions(path.parent/'predictions.csv', number)
        prediction_rows += number
        if (task, split) in identities:
            assert identities[(task, split)] == identity
        identities[(task, split)] = identity
        assert result['metrics']['examples'] == number and int(saved['examples']) == number
        for metric in ('nll', 'accuracy', 'f1'):
            error = abs(measured[metric]-float(result['metrics'][metric]))
            largest_metric_error = max(largest_metric_error, error)
            assert_close(measured[metric], result['metrics'][metric], 2e-6 if metric == 'nll' else 1e-12)
            assert_close(measured[metric], saved[metric])
        for key, expected in [('phase', phase), ('task', task), ('method', method), ('evaluated_split', split)]:
            assert saved[key] == expected
        assert int(saved['seed']) == seed and int(saved['steps']) == result['steps']
        assert_close(saved['lr'], result['lr'])
        assert result['zero_init_error'] == 0 and result['base_frozen']
        assert result['independently_retrained'] is True and saved['independently_retrained'] == 'True'
        assert result['initial_base_sha256'] == result['final_base_sha256']
        bases.add(result['initial_base_sha256'])
        for key in ('initial_base_sha256', 'data_trace_sha256', 'checkpoint_sha256'):
            assert len(result[key]) == 64 and all(character in '0123456789abcdef' for character in result[key])
        assert saved['data_trace_sha256'] == result['data_trace_sha256']
        assert saved['checkpoint_sha256'] == result['checkpoint_sha256']
        if phase == 'main':
            expected_audit[path.relative_to(bundle).as_posix()] = result['checkpoint_sha256']
        traces[(phase, task, seed)].add(result['data_trace_sha256'])
        names = list(assignments[task]['UniformLoRA'])
        ranks = resolve_pattern(result['rank_pattern'], names)
        assert ranks == json.loads(saved['rank_pattern'])
        assert sum(ranks.values()) == result['final_rank_total'] == int(saved['final_rank_total']) == 48
        assert all(0 <= rank <= cfg['rank_cap'] for rank in ranks.values())
        if method != 'AdaLoRA':
            assert ranks == assignments[task][method]
        reference = None if method == 'AdaLoRA' else next((name for name in ('UniformLoRA', 'TwoSidedSpectral')
            if name != method and assignments[task][name] == ranks), None)
        assert result['configuration_equivalent_to'] == reference
        assert saved['configuration_equivalent_to'] == (reference or '')
        physical, effective = (295008, 147504) if method == 'AdaLoRA' else (147456, 147456)
        assert result['physical_trainable'] == int(saved['physical_trainable']) == physical
        assert result['final_effective_parameters'] == int(saved['final_effective_parameters']) == effective
        assert result['final_factor_parameters'] == int(saved['final_factor_parameters']) == 147456
        hhi = float(np.square(np.asarray(list(ranks.values()), dtype=np.float64)/48).sum())
        coverage = sum(rank > 0 for rank in ranks.values())
        tv = float(np.abs(np.asarray(list(ranks.values()))-4).sum()/96)
        assert_close(saved['rank_hhi'], hhi); assert_close(saved['rank_tv_uniform'], tv)
        assert int(saved['nonzero_modules']) == coverage
        for key in ('training_seconds', 'peak_cuda_allocated_bytes'):
            assert float(result[key]) > 0 and math.isfinite(float(result[key]))
            assert_close(saved[key], result[key])
        for key in ('full_vs_compact_max_logit_error', 'full_vs_compact_loss_error', 'fp32_module_merge_error'):
            assert math.isfinite(result[key]) and result[key] >= 0
        assert result['full_vs_compact_max_logit_error'] < 5e-4
        assert result['full_vs_compact_loss_error'] <= 2*result['full_vs_compact_max_logit_error']+1e-6
        assert result['fp32_module_merge_error'] < 5e-4
        max_compact_logit = max(max_compact_logit, result['full_vs_compact_max_logit_error'])
        max_compact_loss = max(max_compact_loss, result['full_vs_compact_loss_error'])
        curve_path = path.parent/'training.jsonl'
        USED.add(curve_path.resolve())
        curve = [json.loads(line) for line in curve_path.read_text(encoding='utf-8').splitlines()]
        assert len(curve) == cfg['steps']
        ranks_over_time = []
        for index, event in enumerate(curve, 1):
            assert event['step'] == index
            current = resolve_pattern(event['rank_pattern'], names)
            total = sum(current.values()); ranks_over_time.append(total)
            assert total == event['effective_rank']
            assert event['effective_factor_parameters'] == total*3072
            assert all(0 <= rank <= cfg['rank_cap'] for rank in current.values())
            if method != 'AdaLoRA':
                assert current == ranks
            assert all(math.isfinite(event[key]) for key in ('nll', 'total_loss', 'regularization', 'lr'))
            assert_close(event['regularization'], event['total_loss']-event['nll'])
            assert event['lr'] >= 0
        assert curve[-1]['data_trace_sha256'] == result['data_trace_sha256']
        assert curve[-1]['rank_pattern'] == result['rank_pattern']
        assert result['effective_factor_parameter_peak'] == max(ranks_over_time)*3072
        assert_close(result['effective_factor_parameter_average'], np.mean(ranks_over_time)*3072)
        for key in ('effective_factor_parameter_peak', 'effective_factor_parameter_average'):
            assert_close(saved[key], result[key])
        history_steps += len(curve)
        peak_effective_factor_parameters = max(peak_effective_factor_parameters, result['effective_factor_parameter_peak'])
        key = (phase, task, method, seed, float(result['lr']))
        assert key not in values
        values[key] = {**measured, 'unit_id': identifier, 'lr': float(result['lr']), 'rank_hhi': hhi,
            'nonzero_modules': coverage, 'rank_tv_uniform': tv,
            'training_seconds': float(result['training_seconds']),
            'peak_cuda_allocated_bytes': int(result['peak_cuda_allocated_bytes']),
            'effective_factor_parameter_peak': int(result['effective_factor_parameter_peak']),
            'effective_factor_parameter_average': float(result['effective_factor_parameter_average'])}
    assert len(values) == len(published) == 64 and len(bases) == 1
    assert len(expected_audit) == 40
    verify_checkpoint_records(audit, expected_audit)
    assert all(len(trace) == 1 for trace in traces.values())
    selection = read(bundle/'selection.json')
    selected_values = {}
    for task in cfg['tasks']:
        for method in METHODS:
            candidates = [value for (phase, name, algorithm, _, _), value in values.items()
                          if (phase, name, algorithm) == ('tuning', task, method)]
            assert len(candidates) == 3 and {v['lr'] for v in candidates} == set(cfg['learning_rates'])
            best = sorted(candidates, key=lambda v: (v['nll'], v['lr']))[0]
            item = selection[task][method]
            assert item['unit_id'] == best['unit_id']
            assert_close(item['lr'], best['lr']); assert_close(item['tuning_nll'], best['nll'], 2e-6)
            listed = {v['unit_id']: v for v in item['candidates']}
            assert set(listed) == {v['unit_id'] for v in candidates}
            for value in candidates:
                assert_close(listed[value['unit_id']]['lr'], value['lr'])
                assert_close(listed[value['unit_id']]['tuning_nll'], value['nll'], 2e-6)
            for seed in cfg['main_seeds']:
                selected_values[(task, method, seed)] = values[('main', task, method, seed, best['lr'])]
    summary = records(bundle/'summary.csv')
    assert len(summary) == 8 and len({(r['task'], r['method']) for r in summary}) == 8
    for row in summary:
        selected = [selected_values[(row['task'], row['method'], seed)] for seed in cfg['main_seeds']]
        assert int(row['main_seed_count']) == 5 and int(row['pretrained_checkpoint_count']) == 1
        for metric in ('nll', 'accuracy', 'f1', 'rank_hhi', 'nonzero_modules', 'rank_tv_uniform',
                       'training_seconds', 'peak_cuda_allocated_bytes',
                       'effective_factor_parameter_peak', 'effective_factor_parameter_average'):
            numbers = np.asarray([value[metric] for value in selected], dtype=np.float64)
            assert_close(row[metric+'_mean'], numbers.mean())
            assert_close(row[metric+'_sample_std'], numbers.std(ddof=1))
        physical, effective = (295008, 147504) if row['method'] == 'AdaLoRA' else (147456, 147456)
        assert int(row['physical_trainable']) == physical and int(row['final_effective_parameters']) == effective
        assert int(row['final_factor_parameters']) == 147456
    pair_rows = records(bundle/'paired_differences.csv')
    pairs = {(r['task'], int(r['seed']), r['contrast']): r for r in pair_rows}
    assert len(pairs) == len(pair_rows) == 30
    matrix, columns = [], [task+'_'+name for task in cfg['tasks'] for name, _, _, _ in COMPARISONS]
    for seed in cfg['main_seeds']:
        vector = []
        for task in cfg['tasks']:
            for name, first, second, primary in COMPARISONS:
                a, b = selected_values[(task, first, seed)], selected_values[(task, second, seed)]
                row = pairs[(task, seed, name)]
                assert row['first_method'] == first and row['second_method'] == second
                assert (row['primary'] == 'True') == primary
                for metric in ('nll', 'accuracy', 'f1'):
                    assert_close(row['delta_'+metric], a[metric]-b[metric])
                vector.append(a['nll']-b['nll'])
        matrix.append(vector)
    source_inputs = records(bundle/'bootstrap_input.csv')
    assert [int(r['seed']) for r in source_inputs] == cfg['main_seeds']
    for saved, vector in zip(source_inputs, matrix):
        assert all(math.isclose(float(saved[column]), value, abs_tol=1e-12, rel_tol=1e-12)
                   for column, value in zip(columns, vector))
    matrix = np.asarray(matrix, dtype=np.float64)
    generator = np.random.default_rng(cfg['bootstrap_seed'])
    samples = np.empty((cfg['bootstrap_draws'], len(columns)))
    digest = hashlib.sha256()
    for start in range(0, len(samples), 10000):
        indices = generator.integers(0, 5, size=(min(10000, len(samples)-start), 5), dtype=np.int64)
        digest.update(indices.astype('<i8', copy=False).tobytes())
        frequencies = (indices[:, :, None] == np.arange(5)[None, None, :]).sum(axis=1)
        samples[start:start+len(indices)] = frequencies@matrix/5
    bounds = np.quantile(samples, [.025, .975], axis=0, method='linear')
    source = read(bundle/'bootstrap.json')
    assert source['columns'] == columns and source['seed_pairs'] == 5
    assert source['draws'] == cfg['bootstrap_draws'] and source['seed'] == cfg['bootstrap_seed']
    assert source['confidence'] == .95 and source['resample_index_sha256'] == digest.hexdigest()
    assert np.max(np.abs(np.asarray(source['mean'])-matrix.mean(axis=0))) < 1e-12
    assert np.max(np.abs(np.asarray(source['low'])-bounds[0])) < 1e-12
    assert np.max(np.abs(np.asarray(source['high'])-bounds[1])) < 1e-12
    intervals = records(bundle/'confidence_intervals.csv')
    assert len(intervals) == 6 and {r['column'] for r in intervals} == set(columns)
    for row in intervals:
        index = columns.index(row['column'])
        assert_close(row['mean'], matrix[:, index].mean())
        assert_close(row['low'], bounds[0, index]); assert_close(row['high'], bounds[1, index])
        assert int(row['seed_pairs']) == 5
        expected = next(c for c in COMPARISONS if c[0] == row['contrast'])
        assert row['positive_nll_delta_favors'] == expected[2]
        assert (row['primary'] == 'True') == expected[3]
    interpretation = read(bundle/'interpretation.json')
    primary = [index for index, column in enumerate(columns) if column.endswith('_uniform_minus_spectral')]
    decision = all(bounds[1, index] < 0 for index in primary)
    assert len(primary) == 2 and interpretation['two_task_spectral_nll_disadvantage'] == bool(decision)
    assert interpretation['complete'] and interpretation['pretrained_checkpoint_count'] == 1
    assert interpretation['main_units'] == 40 and interpretation['tuning_units'] == 24
    assert interpretation['main_seed_pairs_per_task'] == 5
    assert not interpretation['concentration_causality_established']
    assert not interpretation['universal_disadvantage_established']
    assert not interpretation['pretraining_seed_uncertainty_estimated']
    assert not interpretation['adalora_peak_physical_parameters_match_static']
    assert interpretation['adalora_final_effective_extra_scalar_parameters'] == 48
    assert interpretation['selection_lock_verified'] and interpretation['all_64_units_independently_retrained']
    assert interpretation['rank_equivalent_static_method_pairs'] == equivalent
    assert interpretation['coverage_floor_rank_degenerate_tasks'] == [task for task in cfg['tasks']
        if assignments[task]['TwoSidedSpectral'] == assignments[task]['CoverageFloor']]
    assert interpretation['paired_batch_traces_match'] and not interpretation['test_split_used']
    opposite = []
    for index, column in enumerate(columns):
        task = next(name for name in cfg['tasks'] if column.startswith(name+'_'))
        name = column[len(task)+1:]
        for metric in ('accuracy', 'f1'):
            delta = np.mean([float(pairs[(task, seed, name)]['delta_'+metric]) for seed in cfg['main_seeds']])
            if delta*matrix[:, index].mean() > 0:
                opposite.append((task, name, metric, float(delta), float(matrix[:, index].mean())))
    declared = interpretation['classification_directions_opposite_to_nll']
    assert len(opposite) == len(declared)
    for expected, actual in zip(opposite, declared):
        assert (actual['task'], actual['contrast'], actual['classification_metric']) == expected[:3]
        assert_close(actual['mean_classification_delta'], expected[3])
        assert_close(actual['mean_nll_delta'], expected[4])
    near = [r['task'] for r in summary if r['method'] == 'TwoSidedSpectral' and float(r['rank_tv_uniform_mean']) < .10]
    assert interpretation['near_degenerate_spectral_tasks_TV_below_0p10'] == near
    assert USED <= hashed, 'A consumed evidence file is not listed in the artifact manifest'
    result = {'pass': True, 'requires_torch': False, 'files_verified': len(hashed),
        'main_units_recomputed': 40, 'tuning_units_recomputed': 24,
        'prediction_rows_recomputed': prediction_rows, 'paired_metric_contrasts_recomputed': 30,
        'marginal_NLL_intervals_recomputed': 6, 'bootstrap_draws_recomputed': cfg['bootstrap_draws'],
        'bootstrap_indices_sha256': digest.hexdigest(), 'paired_batch_traces_match': True,
        'stored_GPU_checkpoint_audit_records_inspected': 40,
        'checkpoint_tensors_reloaded_by_this_verifier': False,
        'one_pretrained_checkpoint_only': True, 'two_task_spectral_NLL_disadvantage': bool(decision),
        'maximum_saved_vs_recomputed_metric_error': largest_metric_error,
        'maximum_recorded_full_vs_compact_logit_error': max_compact_logit,
        'maximum_recorded_full_vs_compact_loss_error': max_compact_loss,
        'training_curve_steps_recomputed': history_steps,
        'maximum_effective_factor_parameters_during_training': peak_effective_factor_parameters,
        'selection_lock_hash_verified': True,
        'main_checkpoint_audit_paths_and_hashes_cross_matched': 40,
        'rank_equivalent_static_method_pairs': equivalent,
        **calibration_report}
    print(json.dumps(result, indent=2, allow_nan=False))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle', type=Path)
    verify(parser.parse_args().bundle.resolve())
