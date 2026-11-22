"""Residual row masking is distinct from hidden/input-column masking."""
import unittest

import torch
from torch import nn

from carryrank.axis_lora import AxisGroupedLinear, attach_axes, merge_axes
from carryrank.experiment import base_digest


class AxisLoRATests(unittest.TestCase):
    def test_row_mask_zero_initialization_gradient_and_merge(self):
        torch.manual_seed(41)
        model = nn.Sequential(nn.Linear(9, 6, bias=True))
        # A named container permits the same module paths used by the model.
        wrapper = nn.Module()
        wrapper.block = model
        group = {'id': 'r', 'module': 'block.0', 'axis': 'row', 'indices': [1, 4], 'm': 2, 'n': 9, 'cost': 11}
        x = torch.randn(3, 5, 9)
        original = wrapper.block(x).detach()
        frozen = base_digest(wrapper)
        self.assertEqual(attach_axes(wrapper, [group], {'r': 2}), 22)
        self.assertTrue(torch.equal(wrapper.block(x), original))
        factor = wrapper.block[0].factors[0]
        with torch.no_grad():
            factor.B.normal_()
        output = wrapper.block(x)
        self.assertTrue(torch.equal(output[..., [0, 2, 3, 5]], original[..., [0, 2, 3, 5]]))
        output.square().mean().backward()
        self.assertIsNotNone(factor.A.grad)
        self.assertIsNone(wrapper.block[0].base.weight.grad)
        self.assertEqual(base_digest(wrapper), frozen)
        merge_axes(wrapper)
        self.assertTrue(torch.allclose(wrapper.block(x), output.detach(), atol=1e-6, rtol=1e-6))

    def test_column_and_full_matrix_adapters(self):
        for indices in ([0, 2, 4], list(range(5))):
            linear = nn.Linear(5, 7, bias=False)
            grouped = AxisGroupedLinear(linear, [('column', indices, 2)], 1.)
            with torch.no_grad():
                grouped.factors[0].B.normal_()
            merged = grouped.merged()
            x = torch.randn(2, 3, 5)
            self.assertTrue(torch.allclose(grouped(x), merged(x), atol=1e-6, rtol=1e-6))
            delta = merged.weight - linear.weight
            unused = [i for i in range(5) if i not in indices]
            if unused:
                self.assertEqual(delta[:, unused].abs().max().item(), 0)

    def test_rejects_silent_parameter_padding(self):
        wrapper = nn.Module()
        wrapper.block = nn.Sequential(nn.Linear(5, 7))
        with self.assertRaises(ValueError):
            attach_axes(wrapper, [{'id': 'bad', 'module': 'block.0', 'm': 7, 'n': 5, 'cost': 99}], {'bad': 1})


if __name__ == '__main__':
    unittest.main()
