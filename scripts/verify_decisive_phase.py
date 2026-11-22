"""No-torch verification of discovery capacity or single-group phase artifacts."""
import argparse
import csv
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def rows(path):
    with path.open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def verify(bundle):
    files = read(bundle/'artifact-manifest.json')['files']
    for item in files:
        path = (ROOT/item['path']).resolve()
        assert path.is_relative_to(ROOT)
        value = path.read_bytes()
        assert len(value) == item['bytes'] and hashlib.sha256(value).hexdigest() == item['sha256'], item['path']
    cfg, completion = read(bundle/'config.json'), read(bundle/'completion.json')
    assert completion['complete'] and not completion['smoke']
    data = rows(bundle/'main.csv')
    assert len(data) == completion['validation_rows']
    grouped, traces = defaultdict(list), defaultdict(set)
    for row in data:
        grouped[(row['profile'], int(row['base_seed']), int(row['budget']), row['method'], int(row['step']))].append(float(row['val_loss']))
        assert int(row['parameters']) == int(row['budget'])
        traces[(row['adapter_seed'], row['step'])].add(row['data_trace_sha256'])
    assert all(len(v) == 1 for v in traces.values())
    means = {}
    for row in rows(bundle/'by_base_summary.csv'):
        key = (row['profile'], int(row['base_seed']), int(row['budget']), row['method'], int(row['step']))
        mean = statistics.mean(grouped[key]); means[key] = mean
        assert abs(mean-float(row['val_loss_mean'])) < 1e-12
    for row in rows(bundle/'summary.csv'):
        values = [v for (p, _, b, m, t), v in means.items() if (p, b, m, t) == (row['profile'], int(row['budget']), row['method'], int(row['step']))]
        assert abs(statistics.mean(values)-float(row['val_loss_mean'])) < 1e-12
        expected = statistics.stdev(values) if len(values) > 1 else 0.
        assert abs(expected-float(row['base_mean_sample_std'])) < 1e-12
    if completion['phase'] == 'capacity':
        gate = read(bundle/'capacity_gate.json')
        for outcome in gate['outcomes']:
            differences = []
            for seed in cfg['discovery_base_seeds']:
                a = means[(outcome['profile'], seed, outcome['low_budget'], 'FullBudgetBalanced', 600)]
                b = means[(outcome['profile'], seed, outcome['high_budget'], 'FullBudgetBalanced', 600)]
                differences.append(a-b)
            assert differences == outcome['paired_base_mean_gains']
            assert statistics.mean(differences) == outcome['mean']
            assert outcome['pass'] == (all(x > 0 for x in differences) and statistics.mean(differences) >= .01)
        assert gate['pass'] == all(r['pass'] for r in gate['outcomes'])
    audit = read(bundle/'checkpoint_audit.json')
    assert audit['pass'] and audit['complete_run']
    assert audit['logical_conditions_verified'] == completion['logical_conditions']
    assert audit['actual_checkpoints_reloaded'] == completion['actual_optimizer_trajectories']
    print(json.dumps({'files_verified': len(files), 'validation_rows_recomputed': len(data),
        'phase': completion['phase'], 'independent_checkpoint_audit_pass': True,
        'requires_torch': False, 'pass': True}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle', type=Path)
    verify(parser.parse_args().bundle.resolve())
