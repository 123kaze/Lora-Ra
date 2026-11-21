import unittest
import torch
from carryrank.diagnostics import input_utility, scaling_for


class Diagnostics(unittest.TestCase):
    def test_input_normalization_removes_positive_column_rescaling_without_damping(self):
        torch.manual_seed(9)
        gradient = torch.randn(8, 6, dtype=torch.float64)
        moment = torch.rand(6, dtype=torch.float64) + .2
        scale = torch.tensor([.1, .5, 1, 2, 5, 10], dtype=torch.float64)
        a, _ = input_utility(gradient, moment, 6, damping=0)
        b, _ = input_utility(gradient * scale, moment * scale.square(), 6, damping=0)
        self.assertTrue(torch.allclose(torch.tensor(a), torch.tensor(b), atol=1e-6))

    def test_scaling_reference_and_invalid_mode(self):
        self.assertEqual(scaling_for(3, 'rs_ref3'), 1)
        self.assertEqual(scaling_for(7, 'constant'), 1)
        self.assertAlmostEqual(scaling_for(12, 'rs_ref3'), .5)
        with self.assertRaises(ValueError):
            scaling_for(3, 'unknown')

    def test_spectrum_is_prefix_monotone(self):
        values, sigma = input_utility(torch.eye(4, dtype=torch.float64), torch.ones(4, dtype=torch.float64), 4)
        self.assertEqual(len(values), 5)
        self.assertTrue(all(a <= b for a, b in zip(values, values[1:])))
        self.assertAlmostEqual(values[-1], 2 / 1.01)
