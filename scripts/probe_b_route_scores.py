"""Algebra-only checks; not external-algorithm training or language-model evidence."""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def score(g):
    return float(np.linalg.svd(g, compute_uv=False)[0]**2 / sum(g.shape))


def white(g, hi, ho):
    return g / np.sqrt(ho[:, None]+.01*ho.mean()) / np.sqrt(hi[None, :]+.01*hi.mean())


def main():
    rng = np.random.default_rng(47291)
    p, grad = rng.normal(size=(5, 4)), rng.normal(size=(5, 4))
    ad_error = float(np.max(np.abs(np.abs(p*grad)-np.abs((p/8)*(grad*8)))))
    q, k = np.diag([0., 1., 0.]), np.diag([0., 2., 0.])
    # Adaptation X = diag([3,0,0]) has both singular subspace projectors e1 e1^T.
    # The two gradients above already lie in its normal space. Scaling X does
    # not change either projector. Eq. 5 of GEAR therefore changes as below.
    values = []
    hi = np.array([1., 2., 3.]); ho = np.array([2., 4., 6.])
    ref_q, ref_k = white(q, hi, ho), white(k, hi, ho)
    for c in (.125, 1., 8.):
        gq, gk = q/c, k*c
        sq, sk = score(gq), score(gk)
        err = max(float(np.max(np.abs(white(gq, hi, ho/c**2)-ref_q))),
                  float(np.max(np.abs(white(gk, hi, ho*c**2)-ref_k))))
        values.append({'c': c, 'GEAR_eq5_Q_normal_score': sq,
                       'GEAR_eq5_K_normal_score': sk,
                       'top_equal_cost_candidate': 'Q' if sq > sk else 'K',
                       'relative_damped_two_sided_max_abs_error': err})
    g = rng.normal(size=(7, 5))
    u, _, vt = np.linalg.svd(g, full_matrices=False)
    uc, _, vtc = np.linalg.svd(g*8, full_matrices=False)
    svd_error = max(float(np.max(np.abs(u[:, :2]@u[:, :2].T-uc[:, :2]@uc[:, :2].T))),
                    float(np.max(np.abs(vt[:2].T@vt[:2]-vtc[:2].T@vtc[:2]))))
    assert ad_error < 1e-12 and svd_error < 1e-12
    assert [v['top_equal_cost_candidate'] for v in values] == ['Q', 'K', 'K']
    assert all(v['relative_damped_two_sided_max_abs_error'] < 1e-12 for v in values)
    output = {'kind': 'synthetic_algebra_only', 'language_model_trained': False,
              'external_implementation_executed': False,
              'AdaLoRA_elementwise_product_max_error': ad_error,
              'positive_scalar_gradient_SVD_subspace_error': svd_error,
              'GEAR_candidate_score_cases': values,
              'GEAR_final_accepted_rank_or_loss_tested': False,
              'general_coordinate_invariance_proved': False, 'pass': True}
    path = ROOT/'reports/b_route_score_probe_20261122.json'
    path.write_text(json.dumps(output, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
