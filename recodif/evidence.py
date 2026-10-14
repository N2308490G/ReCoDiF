"""One fixed finite likelihood, retained-row whitening, and prediction."""
from dataclasses import dataclass
import numpy as np
from scipy.linalg import solve_triangular
from numpy.polynomial.legendre import leggauss


@dataclass
class Evidence:
    y: np.ndarray
    kernel: np.ndarray
    covariance: np.ndarray

    def __post_init__(self):
        self.y = np.asarray(self.y, dtype=np.float64)
        self.kernel = np.asarray(self.kernel, dtype=np.float64)
        self.covariance = np.asarray(self.covariance, dtype=np.float64)
        n = self.y.size
        if self.y.shape != (n,) or n == 0 or self.kernel.ndim != 2 or self.kernel.shape[0] != n or self.covariance.shape != (n, n):
            raise ValueError('Expected y:[observations], K:[observations,cells], covariance:[observations,observations].')
        if not all(np.all(np.isfinite(v)) for v in (self.y, self.kernel, self.covariance)):
            raise ValueError('Evidence must be finite.')
        if not np.allclose(self.covariance, self.covariance.T, rtol=1e-10, atol=1e-12):
            raise ValueError('Noise covariance must be symmetric.')
        self.cholesky = np.linalg.cholesky(self.covariance)
        self.white_y = solve_triangular(self.cholesky, self.y, lower=True)
        self.white_kernel = solve_triangular(self.cholesky, self.kernel, lower=True)

    @classmethod
    def retained(cls, y, kernel, covariance, mask=None, observation_ids=None):
        """Remove masked rows and exact *physical copies*, before whitening.

        Numerically equal independent repeats must have different identifiers.
        Covariance rows must also agree for copies of the same observation.
        """
        y, k, cov = map(np.asarray, (y, kernel, covariance))
        if y.ndim != 1 or k.ndim != 2 or k.shape[0] != len(y) or cov.shape != (len(y), len(y)):
            raise ValueError('Inconsistent evidence shapes.')
        mask = np.ones(len(y), dtype=bool) if mask is None else np.asarray(mask)
        if mask.shape != y.shape or mask.dtype != bool:
            raise ValueError('Retained mask must be a Boolean observation vector.')
        indices = np.flatnonzero(mask)
        if observation_ids is not None:
            ids = np.asarray(observation_ids)
            if ids.shape != y.shape:
                raise ValueError('Observation IDs must match y.')
            seen, kept = {}, []
            for i in indices:
                key = str(ids[i])
                if key in seen:
                    j = seen[key]
                    if y[i] != y[j] or not np.array_equal(k[i], k[j]) or not np.array_equal(cov[i, indices], cov[j, indices]):
                        raise ValueError(f'Conflicting copies for observation ID {key!r}.')
                else:
                    seen[key] = i
                    kept.append(i)
            indices = np.asarray(kept, dtype=int)
        return cls(y[indices], k[indices], cov[np.ix_(indices, indices)]), indices


def nmr_kernel(grid, recovery_s, echo_s, contrast=2.0, quadrature_order=4):
    """Cell-average IR/SR response; coordinates and acquisition times are seconds.

    Integrated cell masses multiply this kernel. Log-cell averaging is a
    discretization choice used only when constructing the fixed likelihood.
    """
    recovery, echo = np.broadcast_arrays(np.asarray(recovery_s, float), np.asarray(echo_s, float))
    if np.any(recovery < 0) or np.any(echo < 0) or not np.all(np.isfinite(recovery+echo)) or contrast not in (1, 2):
        raise ValueError('Use finite nonnegative times and contrast 1 (SR) or 2 (IR).')
    if quadrature_order < 1:
        raise ValueError('Quadrature order must be positive.')
    nodes, weights = leggauss(quadrature_order)
    def averages(edges, times):
        log_points = (edges[:-1, None]+edges[1:, None])/2 + np.diff(edges)[:, None]*nodes/2
        values = np.exp(-times.ravel()[:, None, None] / 10.0**log_points[None])
        return values @ (weights/2)
    first = 1-contrast*averages(grid.t1, recovery)
    second = averages(grid.t2, echo)
    return (first[:, :, None]*second[:, None, :]).reshape(recovery.size, grid.size)


def predictive_distribution(evidence, draws, kernel_h, covariance_hh, covariance_ho=None):
    """Conditional held-out Gaussian means/covariance, including correlated noise."""
    x, kh, hh = map(lambda a: np.asarray(a, dtype=float), (draws, kernel_h, covariance_hh))
    if x.ndim != 2 or kh.ndim != 2 or kh.shape[1] != x.shape[1] or hh.shape != (len(kh), len(kh)):
        raise ValueError('Invalid predictive shapes.')
    ho = np.zeros((len(kh), len(evidence.y))) if covariance_ho is None else np.asarray(covariance_ho, float)
    if ho.shape != (len(kh), len(evidence.y)):
        raise ValueError('Cross-covariance must index held-out by retained observations.')
    gain = np.linalg.solve(evidence.covariance, ho.T).T
    mean = x @ kh.T + (evidence.y - x @ evidence.kernel.T) @ gain.T
    variance = hh - gain @ ho.T
    if not np.all(np.isfinite(mean)) or not np.allclose(variance, variance.T, atol=1e-10):
        raise ValueError('Invalid predictive moments.')
    np.linalg.cholesky(variance)
    return mean, variance
