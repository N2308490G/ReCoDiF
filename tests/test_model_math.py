"""Known distributions and asymmetric cases protect the manuscript mechanisms."""
import unittest
import numpy as np
from numpy.testing import assert_allclose
from scipy.stats import ncx2, beta
from recodif.partitions import Grid, hierarchy, aggregation_matrix
from recodif.evidence import Evidence, nmr_kernel, predictive_distribution
from recodif.cir import corrupt, denoising_target, positive_reverse_step
from recodif.refinement import AllocationCDF, split_mass
from recodif.sampling import summarize


class FiniteLikelihoodTests(unittest.TestCase):
    def setUp(self):
        self.grid = Grid(np.array([-3., -2., 0.]), np.array([-4., -1., 1.]))

    def test_hierarchy_mixed_states_and_order(self):
        grid = Grid(np.linspace(-4, 1, 5), np.linspace(-4, 1, 5))
        start, stages = hierarchy(grid, [[1, 1], [2, 2], [4, 4]])
        reference = np.arange(1., 17.)
        for before, after, _ in sum(stages, []):
            p = aggregation_matrix(after, before)
            assert_allclose(p @ after.aggregate(reference), before.aggregate(reference))
        final = stages[-1][-1][1]
        assert_allclose(final.to_reference(final.aggregate(reference)), reference)
        self.assertEqual([s[1].size for s in stages[0]], [2, 3, 4])

    def test_deduplicate_then_whiten(self):
        base_y = np.array([-2., 3.])
        base_k = np.array([[1., 2.], [3., 5.]])
        base_cov = np.array([[4., 1.], [1., 2.]])
        ix = np.array([0, 1, 0])
        evidence, keep = Evidence.retained(base_y[ix], base_k[ix], base_cov[np.ix_(ix, ix)],
                                           observation_ids=['A', 'B', 'A'])
        assert_allclose(keep, [0, 1])
        residual = base_y-base_k @ np.array([0.2, 0.3])
        whitened = evidence.white_y-evidence.white_kernel @ np.array([0.2, 0.3])
        assert_allclose(whitened @ whitened, residual @ np.linalg.solve(base_cov, residual))
        repeat, _ = Evidence.retained([1., 1.], [[1.], [1.]], np.eye(2), observation_ids=['A', 'B'])
        self.assertEqual(len(repeat.y), 2)

    def test_correlated_prediction_and_signed_kernel(self):
        evidence = Evidence(np.array([2.]), np.array([[3.]]), np.array([[4.]]))
        mean, cov = predictive_distribution(evidence, [[0.5], [1.]], [[5.]], [[2.]], [[1.]])
        assert_allclose(mean[:, 0], [2.625, 4.75])
        assert_allclose(cov, [[1.75]])
        kernel = nmr_kernel(self.grid, [0., 100.], [0., 0.], contrast=2)
        assert_allclose(kernel[0], -1.)
        assert_allclose(kernel[1], 1., atol=1e-10)

    def test_mixed_ir_sr_protocols_share_one_kernel(self):
        kernel = nmr_kernel(self.grid, [0., 0.], [0., 0.], contrast=[1., 2.])
        assert_allclose(kernel[0], 0.)
        assert_allclose(kernel[1], -1.)

    def test_marginal_quantiles_are_draw_level(self):
        x = np.array([[1., 0., 0., 3.], [0., 3., 1., 0.], [5., 1., 1., 2.]])
        summary = summarize(x, self.grid)
        marginal = x.reshape(3, 2, 2).sum(1)/np.diff(self.grid.t2)
        assert_allclose(summary['t2_marginal_lower90'], np.quantile(marginal, .05, axis=0))
        assert_allclose(summary['mean_spectrum']*self.grid.areas.reshape(2, 2), x.mean(0).reshape(2, 2))


class CIRTests(unittest.TestCase):
    def test_scalar_target_and_bessel_underflow_fallback(self):
        assert_allclose(denoising_target(0., .1, .02, .5, 1., 1.), .04)
        # Large q and small argument underflow even scaled Bessel functions.
        value = denoising_target(1e-4, 1e-4, 500., .5, 1., 1.)
        self.assertTrue(np.isfinite(value))

    def test_target_against_log_density_derivative(self):
        k, sig, t = 1.3, .9, .41
        for a in (0.02, 0.3, 2., 30.):
            for initial in (0., .2, 2.):
                x = np.array([.04, .31, 2.])
                c = sig**2*(-np.expm1(-k*t))/(4*k)
                df, nc = 4*k*a/sig**2, np.exp(-k*t)*initial/c
                epsilon = x*1e-5
                derivative = (ncx2.logpdf((x+epsilon)/c, df, nc)-ncx2.logpdf((x-epsilon)/c, df, nc))/(2*epsilon)
                expected = 1+x*derivative
                assert_allclose(denoising_target(x, initial, a, t, k, sig), expected, rtol=2e-6, atol=3e-7)

    def test_corruption_mean_variance_and_additivity(self):
        rng = np.random.default_rng(93)
        n, k, sig, t = 70000, 1.1, .7, .8
        a, initial = np.array([.1, .8]), np.array([.4, 1.3])
        samples = corrupt(np.broadcast_to(initial, (n, 2)), a, t, k, sig, rng)
        rho = np.exp(-k*t)
        mean = a+(initial-a)*rho
        var = initial*sig**2*rho*(1-rho)/k + a*sig**2*(1-rho)**2/(2*k)
        assert_allclose(samples.mean(0), mean, atol=.006)
        assert_allclose(samples.var(0), var, rtol=.03)
        coarse = corrupt(np.full(n, initial.sum()), a.sum(), t, k, sig, rng)
        assert_allclose(np.quantile(samples.sum(1), [.1, .5, .9]), np.quantile(coarse, [.1, .5, .9]), atol=.02)

    def test_positive_reverse_beta_zero_and_both_signs(self):
        rng = np.random.default_rng(13)
        for b in (-2., 0., 1.2):
            k, sig, a, x, dt = 1., .8, .07, .2, .12
            sample = positive_reverse_step(np.full(80000, x), a, (b-k)/sig**2, dt, k, sig, rng)
            expected = x*np.exp(b*dt)+k*a*(np.expm1(b*dt)/b if b else dt)
            self.assertTrue(np.all(sample >= 0))
            assert_allclose(sample.mean(), expected, atol=.002)


class AllocationTests(unittest.TestCase):
    def test_beta_density_including_singular_endpoints(self):
        for a, b in ((.08, .3), (.7, 1.8), (3., 2.)):
            distribution = AllocationCDF(a, b, lambda s: np.zeros_like(s), potential_tolerance=1e-10)
            for p in (.05, .2, .5, .9):
                s = distribution.quantile(p)
                assert_allclose(s, beta.ppf(p, a, b), atol=2e-8, rtol=1e-5)
                assert_allclose(distribution.cdf(s), p, atol=3e-7)

    def test_tilted_uniform_has_continuous_inverse(self):
        distribution = AllocationCDF(1., 1., lambda s: np.full_like(s, 2.), potential_tolerance=1e-10)
        for p in (.1, .4, .8):
            assert_allclose(distribution.quantile(p), np.log1p(p*np.expm1(2.))/2, atol=1e-9)

    def test_full_conditioning_and_mass_conservation(self):
        grid = Grid(np.linspace(-2, 0, 3), np.linspace(-3, 0, 3))
        before = grid.uniform_partition((1, 2))
        after = before.split(0, 0)
        calls = []
        def evaluate(states, partition, time):
            self.assertEqual(partition.cells, after.cells)
            assert_allclose(states[:, 2], 2.3)
            assert_allclose(states[:, 0]+states[:, 1], .7)
            self.assertEqual(time, .5)
            calls.append(len(states))
            return np.zeros_like(states)
        result = split_mass([.7, 2.3], before, after, 0, .5, np.ones(4)*.3, 1., 1., evaluate, np.random.default_rng(7))
        assert_allclose(result[:2].sum(), .7, atol=1e-15)
        self.assertEqual(result[2], 2.3)
        self.assertTrue(calls)
        calls.clear()
        assert_allclose(split_mass([0., 2.3], before, after, 0, .5, np.ones(4), 1., 1., evaluate, np.random.default_rng(7)), [0, 0, 2.3])
        self.assertFalse(calls)


if __name__ == '__main__':
    unittest.main()
