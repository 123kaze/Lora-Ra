"""Analyze completed real GPT-2 runs from predictions; NumPy/stdlib, no torch."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
from itertools import combinations

import numpy as np

METHODS = ('UniformLoRA', 'AdaLoRA', 'TwoSidedSpectral', 'CoverageFloor')
CONTRASTS = (
    ('uniform_minus_spectral', 'UniformLoRA', 'TwoSidedSpectral', True),
    ('spectral_minus_floor', 'TwoSidedSpectral', 'CoverageFloor', False),
    ('uniform_minus_adalora', 'UniformLoRA', 'AdaLoRA', False),
)
VALIDATION_COUNTS = {'sst2': 872, 'mrpc': 408}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    temp = path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def rows(path):
    with path.open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def csv_write(path, values):
    if not values:
        raise ValueError('No actual observations to write')
    temp = path.with_name(path.name+'.tmp')
    with temp.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(values[0]))
        writer.writeheader(); writer.writerows(values)
    temp.replace(path)


def close(a, b, tolerance=1e-12):
    if not math.isclose(float(a), float(b), rel_tol=tolerance, abs_tol=tolerance):
        raise AssertionError((a, b))


def prediction_metrics(path, count):
    values = rows(path)
    if len(values) != count or not values:
        raise ValueError(('Prediction row count', path, len(values), count))
    identities, logprob, correct, tp, fp, fn = [], [], 0, 0, 0, 0
    for row in values:
        label, predicted = int(row['label']), int(row['prediction'])
        logits = [float(row['label0_logit']), float(row['label1_logit'])]
        target = float(row['target_logprob'])
        if label not in (0, 1) or predicted not in (0, 1):
            raise ValueError('Only the predefined binary labels are valid')
        if not all(math.isfinite(v) for v in [*logits, target]) or target > 1e-6:
            raise ValueError('Invalid finite full-vocabulary target log probability')
        if predicted != int(logits[1] > logits[0]):
            raise ValueError('Prediction must be argmax over the two fixed verbalizers')
        identities.append((row['example_id'], label)); logprob.append(target)
        correct += label == predicted
        tp += label == predicted == 1
        fp += predicted == 1 and label == 0
        fn += predicted == 0 and label == 1
    if len({identity for identity, _ in identities}) != len(values):
        raise ValueError('Duplicate prediction example IDs')
    denominator = 2*tp+fp+fn
    return {'nll': -math.fsum(logprob)/len(values), 'accuracy': correct/len(values),
            'f1': 2*tp/denominator if denominator else 0., 'examples': len(values)}, identities


def normalized_ranks(pattern, modules):
    output = {name: 0 for name in modules}
    seen = set()
    for key, rank in pattern.items():
        matches = [name for name in modules if key == name or key.endswith('.'+name)]
        if len(matches) != 1 or isinstance(rank, bool) or int(rank) != rank:
            raise ValueError(('Unknown/ambiguous module or noninteger rank', key, rank))
        name = matches[0]
        if name in seen:
            raise ValueError(('Duplicate rank key', key))
        seen.add(name)
        output[name] = int(rank)
    return output


def rank_metrics(pattern, reference):
    total = sum(pattern.values())
    if total <= 0:
        raise ValueError('No active adaptation rank')
    return {'rank_hhi': math.fsum((r/total)**2 for r in pattern.values()),
            'nonzero_modules': sum(r > 0 for r in pattern.values()),
            'rank_tv_uniform': math.fsum(abs(pattern[k]-reference) for k in pattern)/(2*total)}


def paired_bootstrap(matrix, draws=200000, seed=95531):
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or len(matrix) < 2 or not np.isfinite(matrix).all():
        raise ValueError('Expected complete finite paired-seed columns')
    rng = np.random.default_rng(seed)
    samples = np.empty((draws, matrix.shape[1]), dtype=np.float64)
    digest = hashlib.sha256()
    for start in range(0, draws, 10000):
        indices = rng.integers(0, len(matrix), size=(min(10000, draws-start), len(matrix)), dtype=np.int64)
        digest.update(indices.astype('<i8', copy=False).tobytes())
        samples[start:start+len(indices)] = matrix[indices].mean(axis=1)
    low, high = np.quantile(samples, [.025, .975], axis=0, method='linear')
    return {'seed_pairs': len(matrix), 'draws': draws, 'seed': seed, 'confidence': .95,
            'method': 'paired adapter/data-seed percentile; NumPy linear quantiles',
            'scope': 'conditional on one pretrained checkpoint; marginal intervals',
            'resample_index_sha256': digest.hexdigest(), 'mean': matrix.mean(axis=0).tolist(),
            'low': low.tolist(), 'high': high.tolist()}


def analyze(run):
    cfg, completion = read(run/'config.json'), read(run/'completion.json')
    if not completion['complete'] or completion.get('smoke', False):
        raise ValueError('Only a completed nonsmoke run can be analyzed')
    assert completion['test_used'] is False
    tasks, seeds = cfg['tasks'], cfg['main_seeds']
    assert set(tasks) == {'sst2', 'mrpc'} and set(cfg['methods']) == set(METHODS)
    assert len(seeds) == 5 and cfg['tuning_seed'] not in seeds
    assert cfg['bootstrap_draws'] == 200000 and cfg['bootstrap_seed'] == 95531
    audit = read(run/'checkpoint_audit.json')
    assert audit['pass'] and audit['complete'] and audit['main_checkpoints_reloaded'] == 40
    assert len(audit['records']) == 40
    selection = read(run/'selection.json')
    lock = read(run/'selection_lock.json')
    assert lock['selection_sha256'] == hashlib.sha256((run/'selection.json').read_bytes()).hexdigest()
    assert lock['tuning_units'] == 24 and lock['official_validation_used_for_selection'] is False
    audit_records = {item['result_path']: item for item in audit['records']}
    assert len(audit_records) == 40
    expected_audit = set()
    modules, assignments, equivalent = {}, {}, []
    for task in tasks:
        metadata = read(run/'calibration'/task/'modules.json')
        modules[task] = sorted(metadata, key=lambda name: metadata[name]['index'])
        assert len(modules[task]) == 12
        choices = read(run/'calibration'/task/'allocations.json')['methods']
        assignments[task] = {name: normalized_ranks(value['ranks'], modules[task]) for name, value in choices.items()}
        for first, second in combinations(('UniformLoRA', 'TwoSidedSpectral', 'CoverageFloor'), 2):
            if assignments[task][first] == assignments[task][second]:
                equivalent.append({'task': task, 'first_method': first, 'second_method': second,
                    'independently_retrained': True, 'additional_independent_mechanism_evidence': False})
    data, identities, traces, base_hashes = [], {}, defaultdict(set), set()
    metric_error = 0.
    for path in sorted((run/'units').glob('*/result.json')):
        result = read(path)
        phase, task, method = result['phase'], result['task'], result['method']
        assert phase in ('tuning', 'main') and task in tasks and method in METHODS
        assert int(result['steps']) == cfg['steps']
        split = 'tune' if phase == 'tuning' else 'validation'
        assert result['evaluated_split'] == split
        assert int(result['seed']) in ([cfg['tuning_seed']] if phase == 'tuning' else seeds)
        expected = cfg['tune_examples'] if phase == 'tuning' else VALIDATION_COUNTS[task]
        actual, identity = prediction_metrics(path.parent/'predictions.csv', expected)
        for metric in ('nll', 'accuracy', 'f1'):
            error = abs(actual[metric]-float(result['metrics'][metric]))
            metric_error = max(metric_error, error)
            close(actual[metric], result['metrics'][metric], 2e-6 if metric == 'nll' else 1e-12)
        assert result['metrics']['examples'] == expected
        if (task, split) in identities:
            assert identities[(task, split)] == identity
        identities[(task, split)] = identity
        assert result['base_frozen'] and result['zero_init_error'] == 0
        assert result['independently_retrained'] is True
        assert result['initial_base_sha256'] == result['final_base_sha256']
        base_hashes.add(result['initial_base_sha256'])
        assert result['full_vs_compact_max_logit_error'] < 5e-4
        assert result['full_vs_compact_loss_error'] <= 2*result['full_vs_compact_max_logit_error']+1e-6
        assert result['fp32_module_merge_error'] < 5e-4
        for name in ('data_trace_sha256', 'initial_base_sha256', 'checkpoint_sha256'):
            assert len(result[name]) == 64 and all(c in '0123456789abcdef' for c in result[name])
        traces[(phase, task, result['seed'])].add(result['data_trace_sha256'])
        pattern = normalized_ranks(result['rank_pattern'], modules[task])
        reference = None
        if method != 'AdaLoRA':
            assert pattern == assignments[task][method]
            reference = next((name for name in ('UniformLoRA', 'TwoSidedSpectral')
                              if name != method and assignments[task][name] == pattern), None)
        assert result['configuration_equivalent_to'] == reference
        if phase == 'main':
            result_path = path.relative_to(run).as_posix()
            expected_audit.add(result_path)
            evidence = audit_records[result_path]
            assert evidence['checkpoint_sha256'] == result['checkpoint_sha256']
            assert evidence['pass'] and evidence['max_metric_reload_error'] == 0
        assert all(0 <= rank <= cfg['rank_cap'] for rank in pattern.values())
        assert sum(pattern.values()) == result['final_rank_total'] == 48
        assert result['final_factor_parameters'] == 147456
        physical, effective = (295008, 147504) if method == 'AdaLoRA' else (147456, 147456)
        assert result['physical_trainable'] == physical and result['final_effective_parameters'] == effective
        if method == 'UniformLoRA':
            assert set(pattern.values()) == {cfg['uniform_rank']}
        if method == 'CoverageFloor':
            assert min(pattern.values()) >= cfg['coverage_floor']
        record = {'unit_id': path.parent.name, 'phase': phase, 'task': task, 'method': method,
            'seed': int(result['seed']), 'lr': float(result['lr']), 'steps': int(result['steps']),
            'evaluated_split': split, **actual,
            'physical_trainable': physical, 'final_effective_parameters': effective,
            'final_factor_parameters': 147456, 'final_rank_total': 48,
            **rank_metrics(pattern, cfg['uniform_rank']), 'rank_pattern': json.dumps(pattern, sort_keys=True),
            'training_seconds': float(result['training_seconds']),
            'peak_cuda_allocated_bytes': int(result['peak_cuda_allocated_bytes']),
            'effective_factor_parameter_peak': int(result['effective_factor_parameter_peak']),
            'effective_factor_parameter_average': float(result['effective_factor_parameter_average']),
            'data_trace_sha256': result['data_trace_sha256'], 'checkpoint_sha256': result['checkpoint_sha256']}
        record.update(configuration_equivalent_to=reference, independently_retrained=True)
        assert math.isfinite(record['training_seconds']) and record['training_seconds'] > 0
        assert record['peak_cuda_allocated_bytes'] > 0
        data.append(record)
    main = [row for row in data if row['phase'] == 'main']
    tuning = [row for row in data if row['phase'] == 'tuning']
    assert len(main) == completion['main'] == 40 and len(tuning) == completion['tuning'] == 24
    assert expected_audit == set(audit_records)
    assert len(base_hashes) == 1
    assert all(len(values) == 1 for values in traces.values())
    lookup = {(r['task'], r['method'], r['seed']): r for r in main}
    assert len(lookup) == 40
    for task in tasks:
        for method in METHODS:
            candidates = [row for row in tuning if (row['task'], row['method']) == (task, method)]
            assert len(candidates) == len(cfg['learning_rates']) == 3
            assert {r['lr'] for r in candidates} == set(cfg['learning_rates'])
            best = min(candidates, key=lambda row: (row['nll'], row['lr']))
            selected = selection[task][method]
            assert selected['unit_id'] == best['unit_id']
            close(selected['lr'], best['lr']); close(selected['tuning_nll'], best['nll'], 2e-6)
            choices = {item['unit_id']: item for item in selected['candidates']}
            assert set(choices) == {row['unit_id'] for row in candidates}
            for row in candidates:
                close(choices[row['unit_id']]['lr'], row['lr'])
                close(choices[row['unit_id']]['tuning_nll'], row['nll'], 2e-6)
            for seed in seeds:
                close(lookup[(task, method, seed)]['lr'], best['lr'])
    summary = []
    for task in tasks:
        for method in METHODS:
            values = [lookup[(task, method, seed)] for seed in seeds]
            row = {'task': task, 'method': method, 'main_seed_count': len(values),
                   'pretrained_checkpoint_count': 1}
            for metric in ('nll', 'accuracy', 'f1', 'rank_hhi', 'nonzero_modules', 'rank_tv_uniform',
                           'training_seconds', 'peak_cuda_allocated_bytes',
                           'effective_factor_parameter_peak', 'effective_factor_parameter_average'):
                measured = [item[metric] for item in values]
                row[metric+'_mean'] = statistics.mean(measured)
                row[metric+'_sample_std'] = statistics.stdev(measured)
            row.update({name: values[0][name] for name in ('physical_trainable', 'final_effective_parameters', 'final_factor_parameters')})
            summary.append(row)
    paired, bootstrap_input, columns = [], [], []
    for task in tasks:
        columns.extend(task+'_'+contrast[0] for contrast in CONTRASTS)
        for seed in seeds:
            for name, first, second, primary in CONTRASTS:
                a, b = lookup[(task, first, seed)], lookup[(task, second, seed)]
                paired.append({'task': task, 'seed': seed, 'contrast': name, 'primary': primary,
                    'first_method': first, 'second_method': second,
                    **{'delta_'+metric: a[metric]-b[metric] for metric in ('nll', 'accuracy', 'f1')}})
    differences = {(r['task'], r['seed'], r['contrast']): r for r in paired}
    for seed in seeds:
        bootstrap_input.append({'seed': seed, **{task+'_'+name: differences[(task, seed, name)]['delta_nll']
            for task in tasks for name, _, _, _ in CONTRASTS}})
    matrix = np.array([[row[column] for column in columns] for row in bootstrap_input])
    boot = paired_bootstrap(matrix, cfg['bootstrap_draws'], cfg['bootstrap_seed'])
    boot['columns'] = columns
    intervals, opposite = [], []
    for index, column in enumerate(columns):
        task = next(task for task in tasks if column.startswith(task+'_'))
        name = column[len(task)+1:]
        contrast = next(value for value in CONTRASTS if value[0] == name)
        intervals.append({'task': task, 'contrast': name, 'column': column, 'primary': contrast[3],
            'mean': boot['mean'][index], 'low': boot['low'][index], 'high': boot['high'][index],
            'seed_pairs': len(seeds), 'positive_nll_delta_favors': contrast[2]})
        for metric in ('accuracy', 'f1'):
            delta = statistics.mean(differences[(task, seed, name)]['delta_'+metric] for seed in seeds)
            if delta*boot['mean'][index] > 0:
                opposite.append({'task': task, 'contrast': name, 'classification_metric': metric,
                                 'mean_classification_delta': delta, 'mean_nll_delta': boot['mean'][index]})
    primary = [r for r in intervals if r['primary']]
    assert len(primary) == 2
    near = [r['task'] for r in summary if r['method'] == 'TwoSidedSpectral' and r['rank_tv_uniform_mean'] < .10]
    interpretation = {'complete': True, 'analysis_kind': 'real_model_conditional_adapter_data_seed_analysis',
        'pretrained_checkpoint_count': 1, 'main_seed_pairs_per_task': 5, 'main_units': 40, 'tuning_units': 24,
        'two_task_spectral_nll_disadvantage': all(row['high'] < 0 for row in primary),
        'decision_rule': 'Both Uniform-minus-Spectral marginal 95% NLL CI upper bounds must be negative',
        'primary_nll_is_not_a_task_accuracy_verdict': True,
        'classification_directions_opposite_to_nll': opposite,
        'near_degenerate_spectral_tasks_TV_below_0p10': near,
        'rank_equivalent_static_method_pairs': equivalent,
        'coverage_floor_rank_degenerate_tasks': [task for task in tasks
            if assignments[task]['TwoSidedSpectral'] == assignments[task]['CoverageFloor']],
        'all_64_units_independently_retrained': True, 'selection_lock_verified': True,
        'concentration_causality_established': False, 'universal_disadvantage_established': False,
        'pretraining_seed_uncertainty_estimated': False,
        'adalora_final_effective_extra_scalar_parameters': 48,
        'adalora_peak_physical_parameters_match_static': False,
        'maximum_saved_vs_recomputed_metric_error': metric_error,
        'test_split_used': False, 'paired_batch_traces_match': True}
    for name, values in [('main.csv', main), ('tuning.csv', tuning), ('summary.csv', summary),
                         ('paired_differences.csv', paired), ('confidence_intervals.csv', intervals),
                         ('bootstrap_input.csv', bootstrap_input)]:
        csv_write(run/name, values)
    write(run/'bootstrap.json', boot); write(run/'interpretation.json', interpretation)
    print(json.dumps(interpretation, indent=2, allow_nan=False))
    return interpretation


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    analyze(parser.parse_args().run.resolve())
