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
