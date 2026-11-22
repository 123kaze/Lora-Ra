"""Cache must preserve full-model values, gradients and optimizer updates."""
import contextlib
import copy
import unittest

import torch
from torch.nn import functional as F

from carryrank.axis_lora import attach_axes
from carryrank.model import ModelConfig, VariableWidthLM
from carryrank.preflight import grouped, target_specifications
from carryrank.prefix_cache import assert_only_final_block_trainable, prefix_forward, suffix_forward


class PrefixCacheTests(unittest.TestCase):
    def test_full_and_cached_paths_have_equal_gradients_and_updates(self):
        torch.manual_seed(32)
        model = VariableWidthLM(ModelConfig([32, 16, 32], head_dim=16, seq_len=8))
        groups = grouped(target_specifications(model, 'matched_final'))
        attach_axes(model, groups, {g['id']: 1 for g in groups})
        other = copy.deepcopy(model)
        x, y = torch.randint(0, 256, (2, 8)), torch.randint(0, 256, (2, 8))
        hidden = prefix_forward(model, x)
        self.assertFalse(hidden.requires_grad)
        self.assertTrue(torch.equal(model(x), suffix_forward(other, hidden)))
        a = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001, weight_decay=0.)
        b = torch.optim.AdamW([p for p in other.parameters() if p.requires_grad], lr=.001, weight_decay=0.)
        for _ in range(3):
            a.zero_grad(); b.zero_grad()
            F.cross_entropy(model(x).flatten(0, 1), y.flatten()).backward()
            F.cross_entropy(suffix_forward(other, hidden).flatten(0, 1), y.flatten()).backward()
            for (name, p), (name2, q) in zip(model.named_parameters(), other.named_parameters()):
                self.assertEqual(name, name2)
                if p.requires_grad:
                    self.assertTrue(torch.equal(p.grad, q.grad), name)
            a.step(); b.step()
        for p, q in zip(model.parameters(), other.parameters()):
            self.assertTrue(torch.equal(p, q))

    def test_rejects_trainable_prefix(self):
        model = VariableWidthLM(ModelConfig([32, 16, 32], head_dim=16))
        with self.assertRaises(ValueError):
            assert_only_final_block_trainable(model)


if __name__ == '__main__':
    unittest.main()
