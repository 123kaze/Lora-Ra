"""Catch invalid residual axes, impossible baselines and loss-averaging scale errors."""
import unittest

import torch
from torch import nn

from carryrank.activation_stats import collect_diagonals
from carryrank.model import ModelConfig, VariableWidthLM
from carryrank.preflight import (eligibility_by_cost, exact_feasible, full_groups, grouped,
                                parameter_tv, strict_uniform, target_specifications)


class PreflightTests(unittest.TestCase):
    def setUp(self):
        self.model = VariableWidthLM(ModelConfig([128, 128, 64, 64, 64, 64, 64, 128]))
        self.targets = target_specifications(self.model)
        self.full, self.groups = full_groups(self.targets), grouped(self.targets)

    def test_residual_rows_and_columns_have_correct_shapes(self):
        self.assertEqual(len(self.targets), 7)
        self.assertEqual(sum(g['cost'] for g in self.full), 2464)
        self.assertEqual(sum(g['cost'] for g in self.groups), 4032)
        lookup = {g['id']: g for g in self.groups}
        down = lookup['blocks.7.down:R']
        up = lookup['blocks.7.up:R']
        self.assertEqual((down['axis'], down['m'], down['n']), ('row', 64, 352))
        self.assertEqual((up['axis'], up['m'], up['n']), ('column', 352, 64))
        self.assertEqual(lookup['blocks.7.attn.o_proj:R']['axis'], 'row')

    def test_exact_budget_can_exclude_an_entire_module_family(self):
        self.assertFalse(exact_feasible(self.full, 384, 16))
        self.assertTrue(exact_feasible(self.full, 768, 16))
        self.assertFalse(eligibility_by_cost(self.full, 3072, 16)['480'])
        self.assertTrue(all(eligibility_by_cost(self.full, 992, 16).values()))
        self.assertTrue(all(eligibility_by_cost(self.groups, 992, 16).values()))
        for b in [384, 768, 1152, 1536, 2304, 3072]:
            self.assertFalse(strict_uniform(self.full, b, 16) and strict_uniform(self.groups, b, 16))
        self.assertIsNone(strict_uniform(self.full, 44352, 16))
        self.assertEqual(set(strict_uniform(self.full, 44352, 32).values()), {18})
        self.assertEqual(set(strict_uniform(self.groups, 44352, 32).values()), {11})

    def test_parameter_movement_requires_equal_actual_budgets(self):
        self.assertEqual(parameter_tv({'a': 300, 'b': 100}, {'a': 100, 'b': 300}, 400), .5)
        with self.assertRaises(ValueError):
            parameter_tv({'a': 300}, {'a': 400}, 400)

    def test_output_moment_undoes_mean_loss_scaling(self):
        class Tiny(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(3, 4)
                self.probe = nn.Linear(4, 3, bias=False)
            def forward(self, x):
                return self.probe(self.embedding(x))
        torch.manual_seed(17)
        model = Tiny()
        x = model.embedding.weight[0].detach().double()
        with torch.no_grad():
            logits = model(torch.zeros((1, 1), dtype=torch.long))[0, 0]
            delta = logits.softmax(-1).double()
            delta[0] -= 1
        stats, audit = collect_diagonals(model, [{'module': 'probe'}], torch.zeros(40, dtype=torch.long),
                                          8, 2, 1, 3, 'cpu')
        self.assertTrue(torch.allclose(stats['probe']['Hin'], x.square(), rtol=1e-6, atol=1e-8))
        self.assertTrue(torch.allclose(stats['probe']['Hout'], delta.square(), rtol=1e-6, atol=1e-8))
        self.assertTrue(torch.allclose(stats['probe']['G'], delta[:, None] * x[None, :], rtol=1e-6, atol=1e-8))
        self.assertEqual(audit['optimizer_steps'], 0)
        self.assertFalse(audit['validation_accessed'])
        self.assertFalse(model.probe._forward_hooks)


if __name__ == '__main__':
    unittest.main()
