"""Continuous conditional allocation with beta-weighted integration."""
import numpy as np
from scipy.integrate import quad
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq
from scipy.special import betaincinv


class AllocationCDF:
    """Integrate the bounded residual potential separately from beta endpoints.

    The neural potential is interpolated on successively doubled Chebyshev
    grids. potential_tolerance controls this approximation; quadrature_tolerance
    controls the subsequent weighted integral, not the neural approximation.
    """
    def __init__(self, q1, q2, derivative, quadrature_tolerance=1e-10,
                 potential_tolerance=5e-5, max_nodes=513):
        if q1 <= 0 or q2 <= 0 or not np.isfinite(q1+q2):
            raise ValueError('Beta shapes must be finite and positive.')
        if not 0 < quadrature_tolerance < 1 or potential_tolerance <= 0 or max_nodes < 33:
            raise ValueError('Invalid allocation tolerances or node budget.')
        self.q1, self.q2, self.tolerance = q1, q2, quadrature_tolerance
        previous = None
        nodes = 17
        while nodes <= max_nodes:
            s = (1-np.cos(np.linspace(0, np.pi, nodes)))/2
            d = np.asarray(derivative(s), dtype=float)
            if d.shape != s.shape or not np.all(np.isfinite(d)):
                raise FloatingPointError('Invalid conditional residual derivative.')
            potential = CubicSpline(s, d).antiderivative()
            offset = float(potential(0.5))
            probe = np.linspace(0, 1, 4*nodes)
            values = potential(probe)-offset
            if previous is not None:
                error = np.max(np.abs(values-(previous(probe)-float(previous(0.5)))))
                if error <= potential_tolerance:
                    break
            previous = potential
            nodes = 2*nodes-1
        else:
            raise RuntimeError('Conditional potential failed to converge; increase max_nodes or inspect the field.')
        self.potential, self.offset = potential, offset
        self.nodes, self.potential_error = nodes, float(error)
        # Include spline extrema when shifting the exponential.
        stationary = potential.derivative().roots(extrapolate=False)
        stationary = stationary[(stationary >= 0) & (stationary <= 1)]
        points = np.r_[0., 1., stationary]
        self.shift = float(np.max(potential(points)-offset))
        self.normalizer = self._quad(self.weight, 0., 1., q1-1, q2-1)
        if self.normalizer <= 0 or not np.isfinite(self.normalizer):
            raise FloatingPointError('Allocation normalizer is not finite and positive.')

    def weight(self, s):
        return np.exp(self.potential(s)-self.offset-self.shift)

    def _quad(self, f, left, right, alpha, beta):
        answer = quad(f, left, right, weight='alg', wvar=(alpha, beta),
                      epsabs=self.tolerance, epsrel=self.tolerance, limit=200, full_output=1)
        if len(answer) != 3:
            raise RuntimeError('Conditional weighted quadrature failed: '+str(answer[3]))
        return answer[0]

    def cdf(self, s):
        if s <= 0:
            return 0.0
        if s >= 1:
            return 1.0
        if s <= 0.5:
            value = self._quad(lambda u: self.weight(u)*(1-u)**(self.q2-1), 0., s, self.q1-1, 0.)
            return float(value/self.normalizer)
        tail = self._quad(lambda u: self.weight(1-u)*(1-u)**(self.q1-1), 0., 1-s, self.q2-1, 0.)
        return float(1-tail/self.normalizer)

    def quantile(self, probability):
        p = float(probability)
        if not 0 < p < 1:
            raise ValueError('Use a probability strictly between zero and one.')
        # Invert in beta-probability coordinates: resolves mass close to 0/1
        # that a uniformly spaced allocation grid would miss.
        t = brentq(lambda u: self.cdf(float(betaincinv(self.q1, self.q2, u)))-p,
                   0., 1., xtol=self.tolerance, rtol=max(4*np.finfo(float).eps, self.tolerance))
        return float(betaincinv(self.q1, self.q2, t))


def split_mass(mass, before, after, index, time, reference, kappa, sigma,
               residual_evaluator, rng, **options):
    """Evaluate candidates on the complete *refined* state, then replace parent."""
    state = np.asarray(mass, dtype=float)
    if state.shape != (before.size,) or not np.all(np.isfinite(state)) or np.any(state < 0):
        raise ValueError('Expected one nonnegative complete current state.')
    parent = float(state[index])
    template = np.insert(state, index+1, 0.)
    if parent == 0:
        template[index:index+2] = 0.
        return template
    a = after.aggregate(reference)
    def derivative(s):
        candidates = np.broadcast_to(template, (len(s), after.size)).copy()
        candidates[:, index] = parent*s
        candidates[:, index+1] = parent-candidates[:, index]
        h = residual_evaluator(candidates, after, time)
        return parent*(h[:, index]-h[:, index+1])
    distribution = AllocationCDF(2*kappa*a[index]/sigma**2, 2*kappa*a[index+1]/sigma**2,
                                 derivative, **options)
    # random() can be exactly zero in finite precision; use midpoint 53-bit bins.
    probability = (int(rng.integers(0, 2**53)) + 0.5)/2**53
    probability = min(probability, np.nextafter(1., 0.))
    fraction = distribution.quantile(probability)
    template[index] = parent*fraction
    template[index+1] = parent-template[index]
    return template
