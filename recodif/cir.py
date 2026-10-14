"""Exact CIR corruption, its mass-weighted score, and positive reverse steps."""
import numpy as np
from scipy.special import ive


def _parameters(mass, reference, kappa, sigma):
    x, a = np.broadcast_arrays(np.asarray(mass, float), np.asarray(reference, float))
    if not (np.isfinite(kappa) and kappa > 0 and np.isfinite(sigma) and sigma > 0):
        raise ValueError('kappa and sigma must be finite and positive.')
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(a)) or np.any(x < 0) or np.any(a <= 0):
        raise ValueError('Masses must be finite/nonnegative and reference masses positive.')
    return x, a


def corrupt(x0, reference, time, kappa, sigma, rng):
    x, a = _parameters(x0, reference, kappa, sigma)
    t = np.asarray(time, float)
    if np.any(t <= 0) or not np.all(np.isfinite(t)):
        raise ValueError('Corruption time must be strictly positive.')
    rho = np.exp(-kappa*t)
    c = sigma*sigma*(-np.expm1(-kappa*t))/(4*kappa)
    return c*rng.noncentral_chisquare(4*kappa*a/sigma**2, rho*x/c)


def _bessel_ratio(q, w):
    """I_(q+1)(w)/I_q(w), using scaled Bessel functions and an underflow fallback."""
    q, w = np.broadcast_arrays(q, w)
    result = np.zeros_like(w)
    small = w < 1e-5
    result[small] = w[small]/(2*(q[small]+1))
    active = ~small
    with np.errstate(invalid='ignore', divide='ignore'):
        result[active] = ive(q[active]+1, w[active])/ive(q[active], w[active])
    bad = active & ~np.isfinite(result)
    for index in np.flatnonzero(bad):
        order, value = q.flat[index], w.flat[index]
        previous = None
        for depth in (64, 128, 256, 512, 1024, 2048, 4096):
            ratio = 0.0
            for j in range(depth, 0, -1):
                ratio = value/(2*(order+j) + value*ratio)
            if previous is not None and abs(ratio-previous) <= 1e-13*max(1, abs(ratio)):
                result.flat[index] = ratio
                break
            previous = ratio
        else:
            raise FloatingPointError('Bessel ratio did not converge.')
    return result


def denoising_target(xt, x0, reference, time, kappa, sigma):
    """Eq. (denoising-target); analytic derivative of the noncentral chi-square.

    At numerical zero this returns the conservative boundary limit q. The
    recurrence uses nonnegative Bessel orders even when q < 1.
    """
    x, a = _parameters(xt, reference, kappa, sigma)
    initial, _ = _parameters(x0, reference, kappa, sigma)
    t = np.asarray(time, float)
    if np.any(t <= 0) or not np.all(np.isfinite(t)):
        raise ValueError('Denoising time must be strictly positive.')
    c = sigma**2*(-np.expm1(-kappa*t))/(4*kappa)
    z, nc = x/c, np.exp(-kappa*t)*initial/c
    q, w = np.broadcast_arrays(2*kappa*a/sigma**2, np.sqrt(z*nc))
    ratio = _bessel_ratio(q, w)
    target = q-z/2 + w*w/(2*(2*q+w*ratio))
    if not np.all(np.isfinite(target)):
        raise FloatingPointError('Nonfinite CIR denoising target.')
    return target


def positive_reverse_step(mass, reference, residual, dt, kappa, sigma, rng):
    """Frozen-coefficient noncentral-chi-square update; no Euler clipping."""
    x, a = _parameters(mass, reference, kappa, sigma)
    h = np.asarray(residual, float)
    if dt <= 0 or not np.isfinite(dt) or not np.all(np.isfinite(h)):
        raise ValueError('A positive step and finite residuals are required.')
    z = (kappa+sigma**2*h)*dt
    with np.errstate(over='raise', invalid='raise'):
        # expm1(z)/z is stable away from exactly zero; analytic series at zero.
        ratio = np.empty_like(z)
        near = abs(z) < 1e-6
        ratio[near] = 1+z[near]/2+z[near]**2/6+z[near]**3/24
        ratio[~near] = np.expm1(z[~near])/z[~near]
        c = sigma**2*dt*ratio/4
        result = c*rng.noncentral_chisquare(4*kappa*a/sigma**2, np.exp(z)*x/c)
    if not np.all(np.isfinite(result)):
        raise FloatingPointError('Reverse trajectory became nonfinite; reduce step size or inspect the learned field.')
    return result
