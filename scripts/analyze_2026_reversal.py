"""Post-hoc descriptive analysis of already published runs, without torch."""
import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def csv_rows(path):
    with path.open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def write_rows(path, rows):
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def decay(matrix):
    u, sigma, _ = np.linalg.svd(matrix, full_matrices=False)
    energy = sigma**2
    p = energy / energy.sum()
    cdf = np.cumsum(p)
    return {'stable_rank': float(energy.sum() / energy[0]),
            'energy_effective_rank': float(np.exp(-np.sum(p[p > 0] * np.log(p[p > 0])))),
            'rank1_energy_fraction': float(cdf[0]), 'rank2_energy_fraction': float(cdf[1]),
            'rank4_energy_fraction': float(cdf[3]), 'rank90': int(np.searchsorted(cdf, .90) + 1)}, sigma, u[:, 0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--input', type=Path, default=ROOT / 'results/crossdomain_labels_20261122_01')
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    parent = args.input.resolve()
    raw = csv_rows(parent / 'main.csv')
    moments, spectra, cases, paired = [], [], [], []
    input_files = [parent / 'main.csv']
    for profile in ('long', 'short'):
        for seed in (2026, 2027, 2028):
            child = parent / f'{profile}_base_{seed}'
            mapping = read(child / 'calibration_arrays.json')
            schemes = read(child / 'groups.json')
            assignments = read(child / 'allocations.json')
            input_files += [child / x for x in ('calibration_arrays.json', 'calibration_arrays.npz', 'groups.json', 'allocations.json')]
            arrays = np.load(child / 'calibration_arrays.npz', allow_pickle=False)
            true_losses = []
            shuffle_losses = {s: [] for s in (11, 22, 33, 44, 55)}
            for adapter in (11, 22):
                values = {r['method']: float(r['val_loss']) for r in raw if r['profile'] == profile
                          and int(r['base_seed']) == seed and int(r['adapter_seed']) == adapter and int(r['step']) == 600}
                true_losses.append(values['true'])
                for s in shuffle_losses:
                    shuffle_losses[s].append(values[f'shuffle_{s}'])
                random_values = [values[f'shuffle_{s}'] for s in shuffle_losses]
                paired.append({'profile': profile, 'base_seed': seed, 'adapter_seed': adapter,
                               'true_loss': values['true'], 'shuffled_mean': statistics.mean(random_values),
                               'shuffled_minus_true': statistics.mean(random_values) - values['true'],
                               'true_rank_of_6': 1 + sum(x < values['true'] for x in random_values)})
            actual = statistics.mean(true_losses)
            random_values = [statistics.mean(v) for v in shuffle_losses.values()]
            predicted_true = assignments['true']['predicted_utility']
            predicted_random = statistics.mean(assignments[f'shuffle_{s}']['predicted_utility'] for s in shuffle_losses)
            case = {'profile': profile, 'base_seed': seed, 'true_loss': actual,
                    'shuffled_mean_loss': statistics.mean(random_values),
                    'shuffled_minus_true': statistics.mean(random_values) - actual,
                    'true_rank_of_6_base_means': 1 + sum(x < actual for x in random_values),
                    'proxy_true_minus_shuffled': predicted_true - predicted_random,
                    'proxy_relative_margin': (predicted_true - predicted_random) / abs(predicted_true),
                    'true_ranks': json.dumps(assignments['true']['ranks'], sort_keys=True)}
            for name, keys in mapping.items():
                G, hi, ho = [arrays[keys[k]] for k in ('G', 'Hin', 'Hout')]
                group_lookup = {g['label']: g for g in schemes['true'] if g['module'] == name}
                C, R = group_lookup['C']['indices'], group_lookup['R']['indices']
                row = {'profile': profile, 'base_seed': seed, 'module': name,
                       'C_input_second_moment': float(hi[C].mean()), 'R_input_second_moment': float(hi[R].mean()),
                       'C_over_R_input_second_moment': float(hi[C].mean() / hi[R].mean())}
                metrics = {'raw': G, 'diag_input': G / np.sqrt(hi + .01 * hi.mean())[None, :],
                           'diag_both': G / np.sqrt(hi + .01 * hi.mean())[None, :] / np.sqrt(ho + .01 * ho.mean())[:, None]}
                for metric, matrix in metrics.items():
                    directions = {}
                    for label, indices in [('C', C), ('R', R)]:
                        feature, singular, left = decay(matrix[:, indices])
                        directions[label] = left
                        row.update({f'{label}_{metric}_{k}': v for k, v in feature.items()})
                        energy = np.cumsum(singular**2) / np.sum(singular**2)
                        for rank, sigma in enumerate(singular, 1):
                            spectra.append({'profile': profile, 'base_seed': seed, 'module': name, 'metric': metric,
                                            'label': label, 'rank': rank, 'singular_value': float(sigma),
                                            'cumulative_energy_fraction': float(energy[rank-1])})
                    row[f'{metric}_CR_top_left_cosine_squared'] = float(np.dot(directions['C'], directions['R'])**2)
                moments.append(row)
                if name.endswith('v_proj'):
                    case.update({'V_C_over_R_input_second_moment': row['C_over_R_input_second_moment'],
                                 'V_R_both_rank1_energy_fraction': row['R_diag_both_rank1_energy_fraction'],
                                 'V_R_both_stable_rank': row['R_diag_both_stable_rank'],
                                 'V_CR_top_left_cosine_squared': row['diag_both_CR_top_left_cosine_squared']})
            arrays.close()
            cases.append(case)
    write_rows(output / 'moments_and_decay.csv', moments)
    write_rows(output / 'full_spectra.csv', spectra)
    write_rows(output / 'case_summary.csv', cases)
    write_rows(output / 'adapter_paired_differences.csv', paired)
    manifest = {'files': [{'path': p.relative_to(ROOT).as_posix(), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
                          for p in sorted(set(input_files))], 'new_training_performed': False,
                'analysis_type': 'post_hoc_descriptive; no causal or predictive rule established',
                'primary_future_training_budget_selection_uses_this_analysis': False}
    (output / 'input_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(cases, indent=2))


if __name__ == '__main__':
    main()
