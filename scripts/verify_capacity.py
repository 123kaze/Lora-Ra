"""Recompute capacity summaries and locked paired gates using only the stdlib."""
import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def csv_rows(path):
    with path.open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def verify(bundle):
    files = read(bundle / 'artifact-manifest.json')['files']
    for item in files:
        path = (ROOT / item['path']).resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError(item['path'])
        content = path.read_bytes()
        if len(content) != item['bytes'] or hashlib.sha256(content).hexdigest() != item['sha256']:
            raise AssertionError('Artifact hash mismatch: ' + item['path'])
    cfg = read(bundle / 'config.json')
    rows = csv_rows(bundle / 'main.csv')
    groups = defaultdict(list)
    traces = defaultdict(set)
    for row in rows:
        key = (row['profile'], int(row['base_seed']), int(row['rank']), int(row['step']))
        groups[key].append(row)
        traces[(int(row['adapter_seed']), int(row['step']))].add(row['data_trace_sha256'])
        assert int(row['budget']) == 384 * int(row['rank'])
        assert int(row['val_bytes']) == cfg['evaluation_max_blocks'] * 128
    for row in csv_rows(bundle / 'summary.csv'):
        key = (row['profile'], int(row['base_seed']), int(row['rank']), int(row['step']))
        values = groups[key]
        assert len(values) == int(row['n_adapter_seeds']) == len(cfg['adapter_seeds'])
        for metric in ('val_loss', 'train_eval_loss', 'val_improvement'):
            data = [float(r[metric]) for r in values]
            assert math.isclose(statistics.mean(data), float(row[metric + '_mean']), abs_tol=1e-12)
            std = statistics.stdev(data) if len(data) > 1 else 0.
            assert math.isclose(std, float(row[metric + '_sample_std']), abs_tol=1e-12)
    gate = read(bundle / 'capacity_gate.json')
    outcomes = []
    if not cfg['smoke']:
        specification = cfg['gate']
        assert gate['specification'] == specification
        for outcome in gate['outcomes']:
            differences = []
            for seed in cfg['adapter_seeds']:
                pair = {int(r['rank']): float(r['val_loss']) for r in rows
                        if r['profile'] == outcome['profile'] and int(r['base_seed']) == outcome['base_seed']
                        and int(r['adapter_seed']) == seed and int(r['step']) == specification['step']}
                differences.append(pair[specification['low_rank']] - pair[specification['high_rank']])
            assert differences == outcome['paired_rank1_minus_rank4']
            assert statistics.mean(differences) == outcome['mean']
            passed = statistics.mean(differences) >= specification['minimum_mean_loss_reduction'] and all(x > 0 for x in differences)
            assert passed == outcome['pass']
            outcomes.append(passed)
        assert gate['pass'] == all(outcomes)
    else:
        assert gate['pass'] is False
    expected_units = len(cfg['widths']) * len(cfg['base_seeds']) * len(cfg['adapter_seeds']) * len(cfg['ranks'])
    assert len(rows) == expected_units * len(cfg['eval_steps'])
    assert all(len(value) == 1 for value in traces.values())
    audit = read(bundle / 'checkpoint_audit.json')
    assert audit['pass'] and audit['checkpoints_reloaded'] == expected_units
    validation_records = 0
    for profile in cfg['widths']:
        for seed in cfg['base_seeds']:
            child = bundle / f'{profile}_base_{seed}'
            for path in child.glob('logs/*/attempt_*.jsonl'):
                for line in path.read_text(encoding='utf-8').splitlines():
                    item = json.loads(line)
                    if item['stage'] != 'capacity_validation':
                        continue
                    matches = [r for r in rows if all(str(r[k]) == str(item[k]) for k in
                               ('profile', 'base_seed', 'adapter_seed', 'rank', 'step'))]
                    assert len(matches) == 1
                    assert float(matches[0]['val_loss']) == item['val_loss']
                    assert matches[0]['data_trace_sha256'] == item['data_trace_sha256']
                    validation_records += 1
    assert validation_records == len(rows)
    result = {'files_verified': len(files), 'trajectories': expected_units,
              'validation_rows_recomputed': len(rows), 'raw_validation_logs_matched': validation_records,
              'paired_traces_match': True, 'capacity_gate': gate['pass'],
              'independent_checkpoint_audit_pass': audit['pass'], 'requires_torch': False, 'pass': True}
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle', type=Path)
    verify(parser.parse_args().bundle.resolve())
