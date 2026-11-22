"""Predeclared base-level paired percentile bootstrap for stage-0 contrasts."""
from __future__ import annotations

import hashlib

import numpy as np


def paired_percentile_bootstrap(matrix, draws=200000, seed=94721, confidence=.95, batch=10000):
    """Each row is one independent base; all columns share resampling indices.

    Columns may include both profiles, budgets and their paired interactions.
    This returns marginal intervals, not simultaneous family coverage.
    """
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or values.shape[0] < 2 or not np.isfinite(values).all():
        raise ValueError('Expected >=2 complete finite base pairs')
    if not isinstance(draws, int) or draws < 100 or not 0 < confidence < 1 or batch < 1:
        raise ValueError('Invalid bootstrap configuration')
    generator = np.random.default_rng(seed)
    bootstrap = np.empty((draws, values.shape[1]), dtype=np.float64)
    trace = hashlib.sha256()
    for start in range(0, draws, batch):
        length = min(batch, draws-start)
        indices = generator.integers(0, values.shape[0], size=(length, values.shape[0]), dtype=np.int64)
        trace.update(indices.astype('<i8', copy=False).tobytes())
        bootstrap[start:start+length] = values[indices].mean(axis=1)
    alpha = (1-confidence) / 2
    bounds = np.quantile(bootstrap, [alpha, 1-alpha], axis=0, method='linear')
    return {'base_pairs': values.shape[0], 'draws': draws, 'seed': seed, 'confidence': confidence,
            'method': 'paired base percentile; numpy linear quantiles', 'resample_index_sha256': trace.hexdigest(),
            'mean': values.mean(axis=0).tolist(), 'low': bounds[0].tolist(), 'high': bounds[1].tolist()}


def all_positive_bounds(result, required_columns):
    if not required_columns or len(set(required_columns)) != len(required_columns):
        raise ValueError('Predeclare a nonempty unique contrast set')
    if any(not isinstance(column, int) or column < 0 or column >= len(result['low']) for column in required_columns):
        raise ValueError('Invalid predeclared contrast index')
    return all(result['low'][column] > 0 for column in required_columns)
