"""Reject wrong-direction significance and preserve paired profile interactions."""
import unittest

import numpy as np

from carryrank.paired_stats import all_positive_bounds, paired_percentile_bootstrap


class PairedStatsTests(unittest.TestCase):
    def test_negative_or_zero_excluding_positive_advantage(self):
        negative = paired_percentile_bootstrap([-3., -2., -1.], draws=1000)
        self.assertLess(negative['high'][0], 0)
        self.assertFalse(all_positive_bounds(negative, [0]))
        zero = paired_percentile_bootstrap(np.zeros(10), draws=1000)
        self.assertFalse(all_positive_bounds(zero, [0]))

    def test_shared_pair_resampling_preserves_interaction_identity(self):
        matrix = np.array([[1., .5, .5], [2., 1., 1.], [3., 1.5, 1.5]])
        result = paired_percentile_bootstrap(matrix, draws=2000)
        self.assertEqual(result['low'][0], 2*result['low'][1])
        self.assertEqual(result['high'][1], result['high'][2])
        self.assertTrue(all_positive_bounds(result, [0, 2]))
        other = paired_percentile_bootstrap(matrix, draws=2000, batch=73)
        self.assertEqual(result, other)

    def test_token_or_missing_pair_shapes_do_not_silently_pass(self):
        for values in ([1.], [1., float('nan')], np.ones((2, 3, 4))):
            with self.assertRaises(ValueError):
                paired_percentile_bootstrap(values)


if __name__ == '__main__':
    unittest.main()
