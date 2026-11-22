"""Descriptive cross-group rank-one prediction check, stratified by exact cost."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def rows(path):
    with path.open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def csv_write(path, values):
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(values[0]))
        writer.writeheader(); writer.writerows(values)


def average_ranks(values):
    value = np.asarray(values, dtype=np.float64)
    order = np.argsort(value, kind='stable')
    ranks = np.empty(len(value))
    start = 0
    while start < len(value):
        end = start+1
        while end < len(value) and value[order[end]] == value[order[start]]:
            end += 1
        ranks[order[start:end]] = (start+1+end)/2
        start = end
    return ranks


def spearman(x, y):
    a, b = average_ranks(x), average_ranks(y)
    if np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def analyze(run):
    cfg, completion = read(run/'config.json'), read(run/'completion.json')
    assert completion['phase'] == 'mechanism' and completion['complete'] and not completion['smoke']
    data = [r for r in rows(run/'main.csv') if int(r['step']) == cfg['steps']]
    assert len(data) == 84
    correlations = []
    for profile in cfg['widths']:
        for seed in cfg['discovery_base_seeds']:
            selected = [r for r in data if r['profile'] == profile and int(r['base_seed']) == seed]
            assert len(selected) == 14
            for cost in [None, 192, 416]:
                subset = [r for r in selected if cost is None or int(r['budget']) == cost]
                x, y = [float(r['predicted_utility']) for r in subset], [float(r['improvement']) for r in subset]
                correlations.append({'profile': profile, 'base_seed': seed,
                    'cost_stratum': 'mixed_uncontrolled' if cost is None else str(cost), 'groups': len(subset),
                    'spearman': spearman(x, y), 'status': 'descriptive_not_independent_group_samples'})
                if cost is not None:
                    rank_x, rank_y = average_ranks(x), average_ranks(y)
                    for row, px, py in zip(subset, rank_x, rank_y):
                        row['proxy_rank_within_cost'] = float(px)
                        row['actual_rank_within_cost'] = float(py)
    csv_write(run/'single_group_results.csv', data)
    csv_write(run/'spearman_by_base.csv', correlations)
    summary = {'groups_measured': 84, 'optimizer_steps_each': cfg['steps'],
        'independent_discovery_bases_per_width': 3, 'adapter_seed': 11, 'rank_each': 1,
        'cost_strata': [192, 416], 'global_method_loss_decomposed_into_groups': False,
        'main_A_B_gate_uses_this_analysis': False, 'calibration_prediction_scale_fitted': False,
        'interpretation': 'Single-group finite-step ranking diagnostic, not optimal-rank or causal mechanism proof',
        'correlations': correlations}
    (run/'mechanism_summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    analyze(parser.parse_args().run.resolve())
