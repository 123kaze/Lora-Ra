"""Independently reconstruct calibration spectra, allocations and screen distances."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from verify_output_diagonal import solve

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def rows(path):
    with path.open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def distribution(groups, ranks, modules):
    return {name: sum(g['cost']*ranks[g['id']] for g in groups if g['module'] == name) for name in modules}


def balanced(groups, budget, cap):
    average = budget/sum(g['cost'] for g in groups)
    values = {g['id']: [-(r-average)**2 for r in range(cap+1)] for g in groups}
    try:
        return solve(groups, values, budget, cap)
    except KeyError:
        return None


def verify(bundle):
    manifest = read(bundle/'artifact-manifest.json')
    for item in manifest['files']:
        path = (ROOT/item['path']).resolve()
        assert path.is_relative_to(ROOT)
        value = path.read_bytes()
        assert len(value) == item['bytes'] and hashlib.sha256(value).hexdigest() == item['sha256'], item['path']
    cfg = read(bundle/'config.json')
    expected = {(r['profile'], int(r['base_seed']), r['scope'], r['scheme'], r['group'], int(r['rank'])):
                float(r['prefix_utility']) for r in rows(bundle/'spectra.csv')}
    scan = {(r['profile'], int(r['base_seed']), r['scope'], r['scheme'], int(r['budget'])): r
            for r in rows(bundle/'allocation_scan.csv')}
    assignments = read(bundle/'allocations.json')
    max_error, spectra_count, allocation_count = 0., 0, 0
    baseline_cache = {}
    for profile in cfg['profiles']:
        for seed in cfg['discovery_base_seeds']:
            child = bundle/f'{profile}_base_{seed}'
            audit = read(child/'calibration_audit.json')
            assert audit['optimizer_steps'] == 0 and audit['base_unchanged'] and not audit['validation_accessed']
            mapping = read(child/'calibration_arrays.json')
            arrays = np.load(child/'calibration_arrays.npz', allow_pickle=False)
            white = {}
            for name, keys in mapping.items():
                gradient, hi, ho = [arrays[keys[k]] for k in ('G', 'Hin', 'Hout')]
                white[name] = gradient/np.sqrt(ho+.01*ho.mean())[:, None]/np.sqrt(hi+.01*hi.mean())[None, :]
            for scope in cfg['scopes']:
                schemes = read(child/f'groups_{scope}.json')
                targets = read(child/f'targets_{scope}.json')
                full = [{'id': t['module'], 'module': t['module'], 'm': t['m'], 'n': t['n'], 'cost': t['m']+t['n']} for t in targets]
                true = schemes['true']
                modules = [t['module'] for t in targets]
                all_budgets = assignments[f'{profile}:{seed}:{scope}']
                for scheme, groups in schemes.items():
                    utility = {}
                    for group in groups:
                        matrix = white[group['module']]
                        matrix = matrix[:, group['indices']] if group['axis'] == 'column' else matrix[group['indices'], :]
                        singular = np.linalg.svd(matrix, compute_uv=False)[:cfg['max_rank']]
                        value = np.r_[0., np.cumsum(.5*singular**2)]
                        utility[group['id']] = value
                        for rank in range(1, len(value)):
                            max_error = max(max_error, abs(value[rank]-expected[(profile, seed, scope, scheme, group['id'], rank)]))
                            spectra_count += 1
                    for raw_budget, candidates in all_budgets.items():
                        if scheme not in candidates:
                            continue
                        budget = int(raw_budget)
                        answer = solve(groups, utility, budget, cfg['max_rank'])
                        recorded = candidates[scheme]
                        assert answer == recorded['ranks']
                        values = distribution(groups, answer, modules)
                        assert values == recorded['module_parameters'] and sum(values.values()) == budget
                        row = scan[(profile, seed, scope, scheme, budget)]
                        for field, baseline_groups in [('TV_vs_full_integer_balanced', full), ('TV_vs_grouped_integer_balanced', true)]:
                            cache_key = (tuple((g['id'], g['m'], g['n'], g['cost']) for g in baseline_groups), budget)
                            if cache_key not in baseline_cache:
                                baseline_cache[cache_key] = balanced(baseline_groups, budget, cfg['max_rank'])
                            ranks = baseline_cache[cache_key]
                            if ranks is None:
                                assert row[field] == ''
                            else:
                                base_values = distribution(baseline_groups, ranks, modules)
                                tv = sum(abs(values[m]-base_values[m]) for m in modules)/(2*budget)
                                assert abs(tv-float(row[field])) < 1e-12
                        final_v = f"blocks.{max(t['layer'] for t in targets)}.attn.v_proj"
                        reference = {m: budget if m == final_v else 0 for m in modules}
                        tv = sum(abs(values[m]-reference[m]) for m in modules)/(2*budget)
                        assert abs(tv-float(row['TV_vs_V_budget_reference'])) < 1e-12
                        allocation_count += 1
            arrays.close()
    assert max_error < 1e-10 and spectra_count == len(expected) and allocation_count == len(scan)
    selection = read(bundle/'selection.json')
    assert selection['formal_budgets_locked'] == [] and selection['selection_uses_validation_loss'] is False
    # Reproduce the signed old-case differences directly from embedded old measurements.
    legacy = rows(bundle/'legacy_inputs/main.csv')
    import statistics
    cases = rows(bundle/'previous_reversal_diagnostic/case_summary.csv')
    for case in cases:
        values = [r for r in legacy if r['profile'] == case['profile'] and r['base_seed'] == case['base_seed'] and r['step'] == '600']
        actual = statistics.mean(float(r['val_loss']) for r in values if r['method'] == 'true')
        shuffled = statistics.mean(statistics.mean(float(r['val_loss']) for r in values if r['method'] == f'shuffle_{s}')
                                   for s in cfg['shuffle_seeds'])
        assert abs(shuffled-actual-float(case['shuffled_minus_true'])) < 1e-12
    report = {'files_verified': len(manifest['files']), 'prefix_utilities_recomputed': spectra_count,
              'max_numpy_prefix_error': max_error, 'exact_allocations_and_distances_recomputed': allocation_count,
              'old_case_paired_differences_recomputed': len(cases), 'requires_torch': False,
              'new_adapter_training_steps': 0, 'pass': True}
    print(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle', type=Path)
    verify(parser.parse_args().bundle.resolve())
