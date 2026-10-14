import unittest
import numpy as np
from recodif.metrics import empirical_crps, central_interval, energy_discrepancy_v


class ScoreTests(unittest.TestCase):
    def test_crps_matches_pairwise_definition(self):
        x = np.array([[0., 4.], [1., -2.], [3., 5.], [7., 0.]])
        y = np.array([2., 1.])
        expected = np.abs(x-y).mean(axis=0) - 0.5*np.abs(x[:, None]-x[None, :]).mean(axis=(0, 1))
        np.testing.assert_allclose(empirical_crps(x, y), expected)

    def test_single_draw_crps_is_absolute_error(self):
        np.testing.assert_allclose(empirical_crps([[2., 7.]], [1., 3.]), [1., 4.])

    def test_interval_uses_draw_axis(self):
        x = np.array([[0., 10.], [2., 20.], [8., 30.]])
        np.testing.assert_allclose(central_interval(x, 0.5), [[1., 15.], [5., 25.]])

    def test_energy_known_singletons(self):
        self.assertEqual(energy_discrepancy_v([[0., 0.]], [[3., 4.]]), 10.)

    def test_energy_is_zero_for_identical_empirical_laws(self):
        x = np.array([[1., 2.], [4., -1.], [0., 5.]])
        self.assertAlmostEqual(energy_discrepancy_v(x, x, block_size=2), 0.)

    def test_energy_blocks_preserve_full_pairwise_result(self):
        x = np.array([[0., 1.], [3., 2.], [5., 8.]])
        y = np.array([[2., 0.], [4., 7.]])
        distance = lambda a,b: np.linalg.norm(a[:,None,:]-b[None,:,:],axis=-1).mean()
        expected = 2*distance(x,y)-distance(x,x)-distance(y,y)
        self.assertAlmostEqual(energy_discrepancy_v(x,y,block_size=1),expected)

    def test_invalid_data_is_rejected(self):
        with self.assertRaises(ValueError):
            empirical_crps([[float('nan')]], [1.])
        with self.assertRaises(ValueError):
            central_interval([[1.]], 1.1)
        with self.assertRaises(ValueError):
            energy_discrepancy_v([[1.]], [[1.,2.]])


if __name__ == '__main__':
    unittest.main()
