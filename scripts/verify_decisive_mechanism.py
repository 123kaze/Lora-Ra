"""Recompute rank-one proxies and tied Spearman by a second implementation."""
import argparse
import math
from pathlib import Path

import numpy as np

from verify_decisive_phase import read, rows, verify


def rank(values):
    return np.asarray([1+sum(x < v for x in values)+.5*(sum(x == v for x in values)-1) for v in values])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle', type=Path)
    bundle = parser.parse_args().bundle.resolve()
    verify(bundle)
    cfg = read(bundle/'config.json')
    data = rows(bundle/'single_group_results.csv')
    correlations = rows(bundle/'spearman_by_base.csv')
    checks, max_error = 0, 0.
    for profile in cfg['widths']:
        for seed in cfg['discovery_base_seeds']:
            child = bundle/f'{profile}_base_{seed}'
            arrays = np.load(child/'calibration_arrays.npz', allow_pickle=False)
            mapping = read(child/'calibration_arrays.json')
            groups = {g['id']: g for g in read(child/'groups.json')['TwoSidedTrue']}
            selected = [r for r in data if r['profile'] == profile and int(r['base_seed']) == seed]
            for row in selected:
                group = groups[row['group_id']]
                keys = mapping[group['module']]
                G, hi, ho = [arrays[keys[k]] for k in ('G', 'Hin', 'Hout')]
                matrix = G/np.sqrt(ho+.01*ho.mean())[:, None]/np.sqrt(hi+.01*hi.mean())[None, :]
                block = matrix[:, group['indices']] if group['axis'] == 'column' else matrix[group['indices'], :]
                prediction = .5*np.linalg.svd(block, compute_uv=False)[0]**2
                max_error = max(max_error, abs(prediction-float(row['predicted_utility'])))
                assert int(row['budget']) == group['cost']
                checks += 1
            for stratum in ['mixed_uncontrolled', '192', '416']:
                subset = [r for r in selected if stratum == 'mixed_uncontrolled' or r['budget'] == stratum]
                a = rank([float(r['predicted_utility']) for r in subset])
                b = rank([float(r['improvement']) for r in subset])
                a -= a.mean(); b -= b.mean()
                denominator = math.sqrt(float(np.dot(a, a)*np.dot(b, b)))
                recorded = next(r for r in correlations if r['profile'] == profile and int(r['base_seed']) == seed and r['cost_stratum'] == stratum)
                if denominator == 0:
                    assert recorded['spearman'] == ''
                else:
                    assert abs(float(np.dot(a, b)/denominator)-float(recorded['spearman'])) < 1e-12
            arrays.close()
    assert checks == 84 and max_error < 1e-10
    print({'single_group_proxies_recomputed': checks, 'max_numpy_prediction_error': max_error,
           'stratified_spearman_recomputed': len(correlations), 'requires_torch': False, 'pass': True})


if __name__ == '__main__':
    main()
