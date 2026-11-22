"""The joint rule must not pick one favorable budget or significant negative effect."""
import unittest

import numpy as np

from carryrank.paired_stats import all_positive_bounds, paired_percentile_bootstrap


class JointRouteTests(unittest.TestCase):
    def test_one_failed_required_contrast_prevents_A(self):
        matrix = np.ones((10, 12)) * .02
        required = [0, 1, 4, 5, 6, 7, 10, 11]
        result = paired_percentile_bootstrap(matrix, draws=1000)
        self.assertTrue(all_positive_bounds(result, required))
        matrix[:, 11] = -.01
        result = paired_percentile_bootstrap(matrix, draws=1000)
        self.assertFalse(all_positive_bounds(result, required))
        self.assertLess(result['high'][11], 0)

    def test_short_effect_need_not_be_positive_for_long_interaction(self):
        matrix = np.tile([.02, .02, -.01, -.01, .03, .03]*2, (10, 1))
        result = paired_percentile_bootstrap(matrix, draws=1000)
        self.assertTrue(all_positive_bounds(result, [0, 1, 4, 5, 6, 7, 10, 11]))


if __name__ == '__main__':
    unittest.main()
