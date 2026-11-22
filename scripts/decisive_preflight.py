"""Run stage-0 shape/coin-budget audit and calibration-only allocation scans."""
import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np
import torch

from carryrank.activation_stats import collect_diagonals
from carryrank.allocation import allocate
from carryrank.curvature import spectral_utility, whiten
from carryrank.experiment import ROOT, base_digest, digest, read_json, write_csv, write_json
from carryrank.model import ModelConfig, VariableWidthLM
from carryrank.preflight import (common_low_budgets, eligibility_by_cost, exact_feasible, full_groups,
                                grouped, integer_rank_balanced, module_parameters, parameter_tv,
                                strict_uniform, target_specifications)


def run(output, config):
    if output.exists() or output.parent != ROOT / 'runs':
        raise ValueError('Choose a fresh direct runs/<run-id> directory')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for this calibration-only job')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    parent = ROOT / config['parent_run']
    output.mkdir(parents=True)
    write_json(output / 'config.json', config)
    sources = sorted((ROOT / 'src').rglob('*.py')) + [Path(__file__).resolve()]
    identity = {'source': {p.relative_to(ROOT).as_posix(): digest(p.read_bytes()) for p in sources},
                'training_performed': False, 'torch': torch.__version__, 'numpy': np.__version__,
                'gpu': torch.cuda.get_device_name(0), 'base_files': {}}
    shutil.copy2(ROOT / 'plan/decisive-stage0-roadmap.md', output / 'protocol-draft.md')
    identity['protocol_draft_sha256'] = digest((output / 'protocol-draft.md').read_bytes())
    for relative in identity['source']:
        destination = output / 'source_snapshot' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    previous = [ROOT / '研究方案.md'] + [p for p in (ROOT / 'results').rglob('*') if p.is_file()]
    prior = {p.relative_to(ROOT).as_posix(): digest(p.read_bytes()) for p in previous}
    write_json(output / 'prior_artifacts.json', prior)
    target_manifest = read_json(ROOT / 'data/processed/wikitext-2-raw-v1/manifest.json')
    train_record = next(r for r in target_manifest['splits'] if r['split'] == 'train')
    value = (ROOT / train_record['bin_path']).read_bytes()
    expected = 'fc1dd2874bf42e945db9bd38e856ae18010f5c088e7446c706c5ba2f39310ee3'
    assert digest(value) == expected == train_record['bin_sha256']
    train = torch.from_numpy(np.frombuffer(value, dtype=np.uint8).copy()).long()
    write_json(output / 'data_manifest.json', {'used_split': train_record, 'revision': target_manifest['revision'],
               'validation_read_by_allocator': False, 'test_read': False})
    shape_rows, feasibility, scans, spectral_rows = [], [], [], []
    all_allocations, alternatives = {}, {}
    for profile in config['profiles']:
        for seed in config['discovery_base_seeds']:
            child = parent / f'{profile}_base_{seed}'
            cfg = read_json(child / 'config.json')
            base_path = child / 'base.pt'
            identity['base_files'][base_path.relative_to(ROOT).as_posix()] = digest(base_path.read_bytes())
            base = torch.load(base_path, map_location='cpu', weights_only=True)
            model = VariableWidthLM(ModelConfig(**cfg['model'])).cuda()
            model.load_state_dict(base['model'])
            assert base_digest(model) == base['sha256']
            full_targets = target_specifications(model, 'all_expansions')
            stats, audit = collect_diagonals(model, full_targets, train, cfg['model']['seq_len'],
                config['calibration']['batch_size'], config['calibration']['batches'],
                config['calibration']['seed'], 'cuda')
            destination = output / f'{profile}_base_{seed}'
            destination.mkdir()
            arrays, metadata = {}, {}
            for i, target in enumerate(full_targets):
                name = target['module']
                metadata[name] = {}
                for key in ('G', 'Hin', 'Hout'):
                    field = f'm{i}_{key}'
                    arrays[field] = stats[name][key].numpy()
                    metadata[name][key] = field
            np.savez_compressed(destination / 'calibration_arrays.npz', **arrays)
            write_json(destination / 'calibration_arrays.json', metadata)
            write_json(destination / 'calibration_audit.json', audit)
            matrices = {name: whiten(v['G'], v['Hin'], v['Hout'], config['input_damping'], config['output_damping'])
                        for name, v in stats.items()}
            for scope in config['scopes']:
                targets = target_specifications(model, scope)
                true = grouped(targets)
                full = full_groups(targets)
                modules = [t['module'] for t in targets]
                schemes = {'true': true, **{f'shuffle_{s}': grouped(targets, s) for s in config['shuffle_seeds']}}
                write_json(destination / f'groups_{scope}.json', schemes)
                write_json(destination / f'targets_{scope}.json', targets)
                static_key = profile + ':' + scope
                if static_key not in alternatives:
                    alternatives[static_key] = common_low_budgets(full, true, min(config['requested_budgets']),
                        config['max_alternative_budget'], config['alternative_budget_quantum'], config['max_rank'])
                    full_cost, grouped_cost = sum(g['cost'] for g in full), sum(g['cost'] for g in true)
                    shape_rows.append({'profile': profile, 'scope': scope, 'modules': len(targets),
                        'groups': len(true), 'expansion_layers': json.dumps(sorted(set(t['layer'] for t in targets))),
                        'strict_uniform_rank1_cost': full_cost, 'strict_grouped_uniform_rank1_cost': grouped_cost,
                        'first_common_strict_budget_ignoring_cap': math.lcm(full_cost, grouped_cost),
                        'width_proportional_rank_equals_uniform_target': len(set(t['residual_width'] for t in targets)) == 1})
                budgets = sorted(set(config['requested_budgets'] + alternatives[static_key]))
                utilities = {}
                for scheme, groups in schemes.items():
                    utilities[scheme] = {}
                    for group in groups:
                        full_matrix = matrices[group['module']]
                        matrix = (full_matrix[:, group['indices']] if group['axis'] == 'column'
                                  else full_matrix[group['indices'], :])
                        sigma, utility = spectral_utility(matrix, config['max_rank'])
                        utilities[scheme][group['id']] = utility
                        for rank, singular in enumerate(sigma.tolist(), 1):
                            spectral_rows.append({'profile': profile, 'base_seed': seed, 'scope': scope,
                                'scheme': scheme, 'group': group['id'], 'axis': group['axis'], 'rank': rank,
                                'cost': group['cost'], 'singular_value': singular, 'prefix_utility': utility[rank]})
                all_allocations.setdefault(f'{profile}:{seed}:{scope}', {})
                final_v = f"blocks.{len(cfg['model']['widths'])-1}.attn.v_proj"
                for budget in budgets:
                    baselines = {}
                    for name, groups in [('FullBudgetBalanced', full), ('GroupedBudgetBalanced', true),
                                         ('WidthRankProportional', true)]:
                        try:
                            ranks, spent, _ = integer_rank_balanced(groups, budget, config['max_rank'], name == 'WidthRankProportional')
                            baselines[name] = module_parameters(groups, ranks, modules)
                            assert spent == budget
                        except ValueError:
                            baselines[name] = None
                    v_reference = {m: budget if m == final_v else 0 for m in modules}
                    # This is a reference allocation, not necessarily a realizable equal-rank V adapter.
                    if seed == config['discovery_base_seeds'][0]:
                        full_eligible = eligibility_by_cost(full, budget, config['max_rank'])
                        grouped_eligible = eligibility_by_cost(true, budget, config['max_rank'])
                        feasibility.append({'profile': profile, 'scope': scope, 'budget': budget,
                            'requested': budget in config['requested_budgets'],
                            'strict_uniform_feasible': strict_uniform(full, budget, config['max_rank']) is not None,
                            'strict_grouped_uniform_feasible': strict_uniform(true, budget, config['max_rank']) is not None,
                            'full_integer_feasible': baselines['FullBudgetBalanced'] is not None,
                            'full_cost_eligibility': json.dumps(full_eligible, sort_keys=True),
                            'grouped_cost_eligibility': json.dumps(grouped_eligible, sort_keys=True),
                            'all_cost_types_eligible_both': all(full_eligible.values()) and all(grouped_eligible.values()),
                            'equal_rank_final_V_feasible': budget % 384 == 0 and 1 <= budget // 384 <= config['max_rank']})
                    per_budget = {}
                    for scheme, groups in schemes.items():
                        try:
                            ranks, spent, score = allocate(groups, utilities[scheme], budget, config['max_rank'])
                        except ValueError:
                            continue
                        distribution = module_parameters(groups, ranks, modules)
                        per_budget[scheme] = {'ranks': ranks, 'spent': spent, 'score': score,
                                              'module_parameters': distribution}
                        scans.append({'profile': profile, 'base_seed': seed, 'scope': scope, 'scheme': scheme,
                            'budget': budget, 'requested': budget in config['requested_budgets'],
                            'predicted_utility': score, 'active_groups': sum(r > 0 for r in ranks.values()),
                            'Q_parameters': sum(v for k, v in distribution.items() if k.endswith('q_proj')),
                            'MLP_parameters': sum(v for k, v in distribution.items() if k.endswith(('.up', '.gate', '.down'))),
                            'TV_vs_full_integer_balanced': '' if baselines['FullBudgetBalanced'] is None else parameter_tv(distribution, baselines['FullBudgetBalanced'], budget),
                            'TV_vs_grouped_integer_balanced': '' if baselines['GroupedBudgetBalanced'] is None else parameter_tv(distribution, baselines['GroupedBudgetBalanced'], budget),
                            'TV_vs_V_budget_reference': parameter_tv(distribution, v_reference, budget)})
                    all_allocations[f'{profile}:{seed}:{scope}'][str(budget)] = per_budget
            del model
            print(json.dumps({'stage': 'preflight_base_complete', 'profile': profile, 'base_seed': seed,
                              'modules_calibrated': len(full_targets), 'optimizer_steps': 0}), flush=True)
    write_csv(output / 'shapes.csv', shape_rows)
    write_csv(output / 'budget_feasibility.csv', feasibility)
    write_csv(output / 'allocation_scan.csv', scans)
    write_csv(output / 'spectra.csv', spectral_rows)
    write_json(output / 'allocations.json', all_allocations)
    write_json(output / 'alternative_budget_candidates.json', alternatives)
    write_json(output / 'identity.json', identity)
    unchanged = all(digest((ROOT / p).read_bytes()) == expected for p, expected in prior.items())
    base_unchanged = all(digest((ROOT / p).read_bytes()) == expected for p, expected in identity['base_files'].items())
    assert unchanged and base_unchanged
    check = {'calibrated_bases': 6, 'optimizer_steps': 0, 'training_performed': False,
             'calibration_forward_backward_batches': 6 * config['calibration']['batches'],
             'source_weights_unchanged': base_unchanged, 'prior_artifact_count': len(prior),
             'prior_artifacts_unchanged': unchanged, 'allocation_scan_rows': len(scans),
             'spectra_rows': len(spectral_rows), 'formal_training_authorized_by_locked_protocol': False,
             'blocker': 'Strict uniform/grouped uniform equal-budget infeasibility; scope decision pending', 'pass': True}
    write_json(output / 'verification.json', check)
    print(json.dumps(check), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/decisive_preflight.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.output.resolve(), read_json(args.config))
