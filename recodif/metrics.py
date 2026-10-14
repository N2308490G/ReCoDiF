"""Empirical scores used in the manuscript, retaining V-statistic diagonals."""
import numpy as np


def _finite_samples(values):
    x = np.asarray(values, dtype=np.float64)
    if not x.size or not np.isfinite(x).all():
        raise ValueError('Samples must be nonempty and finite.')
    return x


def empirical_crps(draws, observed):
    """Pointwise CRPS; draw axis first, observation axes after it.

    Computes mean |Z-y| - 0.5 mean |Z-Z'| using all empirical pairs.
    """
    x = _finite_samples(draws)
    y = _finite_samples(observed)
    if x.ndim < 1 or x.shape[1:] != y.shape:
        raise ValueError('draws.shape[1:] must equal observed.shape.')
    n = len(x)
    weights = (2 * np.arange(n) - n + 1).reshape((n,) + (1,) * (x.ndim - 1))
    return np.abs(x - y).mean(axis=0) - (weights * np.sort(x, axis=0)).sum(axis=0) / n**2


def central_interval(draws, probability=0.9):
    """Pointwise central interval over the first (draw) axis."""
    x = _finite_samples(draws)
    if x.ndim < 1 or not 0 < probability < 1:
        raise ValueError('Draws need a sample axis and 0 < probability < 1.')
    tail = (1 - probability) / 2
    return np.quantile(x, [tail, 1 - tail], axis=0)


def train_mean_rmse(predicted, observed, train_ids, scale=1.0):
    """Normalize residuals, take RMSE per recovery train, then average trains."""
    predicted, observed = _finite_samples(predicted), _finite_samples(observed)
    ids = np.asarray(train_ids)
    if observed.ndim != 1 or predicted.shape != observed.shape or ids.shape != observed.shape:
        raise ValueError('Predictions, observations and recovery-train IDs must be matching vectors.')
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Normalization scale must be finite and positive.')
    if ids.dtype.kind not in 'USiu' or any(not str(v) for v in ids):
        raise ValueError('Recovery-train IDs must be nonempty strings or integers.')
    residual = (predicted-observed)/scale
    return float(np.mean([np.sqrt(np.mean(residual[ids == train]**2)) for train in np.unique(ids)]))


def site_macro_mean(rows):
    """Average masks -> depth records -> wells/specimens -> sites equally.

    Rows are (site_id, well_id, record_id, mask_id, scalar_score).
    """
    records, seen = {}, set()
    for site, well, record, mask, score in rows:
        key = (site, well, record, mask)
        if any(value is None or not str(value) for value in key) or key in seen or not np.isfinite(score):
            raise ValueError('Require unique site/well/record/mask IDs and finite scores.')
        seen.add(key)
        records.setdefault(key[:3], []).append(float(score))
    if not records:
        raise ValueError('No scores to aggregate.')
    wells, sites = {}, {}
    for (site, well, _), scores in records.items():
        wells.setdefault((site, well), []).append(np.mean(scores))
    for (site, _), scores in wells.items():
        sites.setdefault(site, []).append(np.mean(scores))
    return float(np.mean([np.mean(scores) for scores in sites.values()]))


def energy_discrepancy_v(first, second, block_size=128):
    """Energy discrepancy between two (draws, regions) arrays.

    Includes within-sample diagonal terms, as in the manuscript. Blocking
    changes memory use only; every sample pair contributes.
    """
    x, y = _finite_samples(first), _finite_samples(second)
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        raise ValueError('Both samples must have shape (draws, matching regions).')
    if not isinstance(block_size, int) or block_size < 1:
        raise ValueError('block_size must be a positive integer.')

    def mean_distance(a, b):
        total = 0.0
        for i in range(0, len(a), block_size):
            for j in range(0, len(b), block_size):
                delta = a[i:i + block_size, None, :] - b[None, j:j + block_size, :]
                total += np.linalg.norm(delta, axis=-1).sum()
        return total / (len(a) * len(b))

    return 2 * mean_distance(x, y) - mean_distance(x, x) - mean_distance(y, y)
