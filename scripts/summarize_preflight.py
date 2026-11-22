"""Summarize calibration-only feasibility; never authorize a training run."""
import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DISTANCES = ['TV_vs_full_integer_balanced', 'TV_vs_grouped_integer_balanced', 'TV_vs_V_budget_reference']


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def rows(path):
    with path.open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    config = read(run / 'config.json')
    scans, feasibility = rows(run / 'allocation_scan.csv'), rows(run / 'budget_feasibility.csv')
    output = []
    candidates = {}
    for scope in config['scopes']:
        candidates[scope] = []
        for budget in sorted({int(r['budget']) for r in scans}):
            values = [r for r in scans if r['scope'] == scope and r['scheme'] == 'true' and int(r['budget']) == budget]
            assert len(values) == 6
            checks = [r for r in feasibility if r['scope'] == scope and int(r['budget']) == budget]
            row = {'scope': scope, 'budget': budget, 'discovery_bases': len(values),
                   'requested': budget in config['requested_budgets'],
                   'strict_requested_control_pair_feasible': all(r['strict_uniform_feasible'] == 'True' and r['strict_grouped_uniform_feasible'] == 'True' for r in checks),
                   'all_module_costs_eligible': all(r['all_cost_types_eligible_both'] == 'True' for r in checks),
                   'V_reference_is_realizable_equal_rank': all(r['equal_rank_final_V_feasible'] == 'True' for r in checks)}
            for distance in DISTANCES:
                data = [float(r[distance]) for r in values if r[distance] != '']
                row[distance + '_minimum'] = min(data) if len(data) == 6 else ''
                row[distance + '_maximum'] = max(data) if len(data) == 6 else ''
            row['integer_balanced_variant_nondegenerate'] = (row['all_module_costs_eligible'] and
                all(row[k+'_minimum'] != '' and row[k+'_minimum'] > config['nondegeneracy_parameter_tv_threshold'] for k in DISTANCES))
            if row['integer_balanced_variant_nondegenerate']:
                candidates[scope].append(budget)
            output.append(row)
    with (run / 'nondegeneracy_summary.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    selection = {'formal_budgets_locked': [], 'formal_protocol_locked': False,
        'requested_strict_specification_feasible': False, 'route_A_or_B_decided': False,
        'provisional_integer_balanced_candidates': candidates,
        'requires_baseline_definition_and_scope_resolution': True,
        'V_budget_reference_at_provisional_candidates_is_not_a_trainable_VBalanced_control': True,
        'selection_uses_validation_loss': False, 'training_performed': False}
    (run / 'selection.json').write_text(json.dumps(selection, indent=2), encoding='utf-8')
    print(json.dumps(selection, indent=2))


if __name__ == '__main__':
    main()
