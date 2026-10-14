"""Lognormal-prior reference posterior via elliptical slice sampling.

This sampler is an explicit implementation choice. Its Gaussian prior and
fixed retained-data likelihood follow the manuscript; it is not the CIR
stationary gamma initializer used by the reverse diffusion.
"""
import numpy as np


def lognormal_prior(grid, settings):
    centers = grid.centers.copy()
    centers[:, 0] = (centers[:, 0]-grid.t1[0])/(grid.t1[-1]-grid.t1[0])
    centers[:, 1] = (centers[:, 1]-grid.t2[0])/(grid.t2[-1]-grid.t2[0])
    az, nugget = float(settings['a_z']), float(settings['delta_z'])
    lengths = np.array([settings['L_1'], settings['L_2']], float)
    total = float(settings.get('expected_total_mass', 1.))
    if az <= 0 or nugget <= 0 or np.any(lengths <= 0) or total <= 0:
        raise ValueError('Prior scales must be positive.')
    distance = (centers[:, None]-centers[None, :])/lengths
    covariance = az**2*np.exp(-.5*(distance**2).sum(-1)) + nugget**2*np.eye(grid.size)
    mean = np.full(grid.size, np.log(total/grid.areas.sum())-(az**2+nugget**2)/2)
    return mean, np.linalg.cholesky(covariance)


def posterior_references(evidence, grid, settings, draws_per_chain=512, burnin=1000,
                         chains=4, thinning=1, seed=1001):
    if chains < 4 or draws_per_chain < 4 or burnin < 0 or thinning < 1:
        raise ValueError('Use at least four chains/four draws, nonnegative burn-in and positive thinning.')
    mean, chol = lognormal_prior(grid, settings)
    log_area = np.log(grid.areas)
    def likelihood(centered):
        with np.errstate(over='ignore', invalid='ignore'):
            mass = np.exp(log_area+mean+centered)
            residual = evidence.white_y-evidence.white_kernel @ mass
            result = -.5*(residual @ residual)
        return float(result) if np.isfinite(result) else -np.inf
    result = np.empty((chains, draws_per_chain, grid.size))
    for chain, sequence in enumerate(np.random.SeedSequence(seed).spawn(chains)):
        rng = np.random.default_rng(sequence)
        centered = chol @ rng.normal(size=grid.size)
        log_likelihood = likelihood(centered)
        if not np.isfinite(log_likelihood):
            raise FloatingPointError('Reference-chain initialization has a nonfinite likelihood.')
        saved = 0
        for iteration in range(burnin+draws_per_chain*thinning):
            direction = chol @ rng.normal(size=grid.size)
            threshold = log_likelihood+np.log(max(rng.random(), np.finfo(float).tiny))
            theta = rng.uniform(0, 2*np.pi)
            lower, upper = theta-2*np.pi, theta
            for attempt in range(10000):
                proposal = centered*np.cos(theta)+direction*np.sin(theta)
                proposal_likelihood = likelihood(proposal)
                if proposal_likelihood >= threshold:
                    centered, log_likelihood = proposal, proposal_likelihood
                    break
                if theta < 0:
                    lower = theta
                else:
                    upper = theta
                theta = rng.uniform(lower, upper)
            else:
                raise RuntimeError('Elliptical slice bracket exhausted.')
            if iteration >= burnin and (iteration-burnin) % thinning == 0:
                result[chain, saved] = np.exp(log_area+mean+centered)
                saved += 1
    return result


def reference_diagnostics(draws, grid, regional_edges=(-4., -2., -1., 0., 1.)):
    """Rank R-hat and bulk/tail ESS on total and exactly representable regions."""
    import arviz as az
    samples = np.asarray(draws, float)
    if samples.ndim != 3 or samples.shape[0] < 4 or samples.shape[1] < 4 or samples.shape[2] != grid.size:
        raise ValueError('Diagnostics require [chain,draw,reference cell] samples.')
    values = {'total': samples.sum(-1)}
    edges = np.asarray(regional_edges, float)
    if edges.ndim != 1 or len(edges) < 2 or np.any(np.diff(edges) <= 0):
        raise ValueError('Regional boundaries must be ordered.')
    # A region may combine whole reference cells; do not split their mass by area.
    for coordinate in (grid.t1, grid.t2):
        inside = edges[(edges > coordinate[0]) & (edges < coordinate[-1])]
        if any(not np.any(np.isclose(coordinate, point, atol=1e-12, rtol=0)) for point in inside):
            raise ValueError('Diagnostic regional boundaries must coincide with reference-grid boundaries.')
    centers = grid.centers
    for i, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        for j, (bottom, top) in enumerate(zip(edges[:-1], edges[1:])):
            mask = (centers[:, 0] >= left) & (centers[:, 0] < right) & (centers[:, 1] >= bottom) & (centers[:, 1] < top)
            if np.any(mask):
                values[f'region_{i}_{j}'] = samples[..., mask].sum(-1)
    report = {}
    for name, series in values.items():
        rhat = float(az.rhat(series, method='rank'))
        bulk = float(az.ess(series, method='bulk'))
        tail = float(az.ess(series, method='tail'))
        report[name] = {'rhat': rhat, 'bulk_ess': bulk, 'tail_ess': tail}
    report['criteria_met'] = bool(all(np.isfinite(v['rhat']+v['bulk_ess']+v['tail_ess']) and
                                  v['rhat'] < 1.01 and min(v['bulk_ess'], v['tail_ess']) > 400
                                  for v in report.values()))
    return report
