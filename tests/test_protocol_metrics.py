import unittest
import numpy as np
from numpy.testing import assert_allclose
from recodif.metrics import train_mean_rmse, site_macro_mean


class ProtocolMetricTests(unittest.TestCase):
    def test_root_precedes_recovery_train_average(self):
        observed = np.array([0., 0., 6.])
        result = train_mean_rmse(np.zeros(3), observed, ['A', 'A', 'B'], scale=2.)
        self.assertEqual(result, 1.5)
        self.assertNotAlmostEqual(result, np.sqrt(np.mean(observed**2))/2)

    def test_masks_depths_wells_sites_have_equal_weights_at_each_level(self):
        rows = [('A', 'well1', 'depth1', '0', 0.), ('A', 'well1', 'depth1', '1', 2.),
                ('A', 'well1', 'depth2', '0', 3.), ('A', 'well2', 'depth1', '0', 10.),
                ('B', 'well3', 'depth1', '0', 4.)]
        # depth1=1; well1=(1+3)/2=2; site A=(2+10)/2=6; macro=(6+4)/2=5.
        assert_allclose(site_macro_mean(rows), 5.)
        with self.assertRaises(ValueError):
            site_macro_mean(rows+[rows[0]])


if __name__ == '__main__':
    unittest.main()
