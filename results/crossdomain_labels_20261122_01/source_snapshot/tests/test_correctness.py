"""Correctness invariants that can invalidate the scientific comparison."""
import itertools
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

from carryrank.allocation import allocate, equal_parameters
from carryrank.data import evaluation_batches, prepare, random_batch
from carryrank.experiment import base_digest
from carryrank.lora import GroupedLinear, attach, merge_all
from carryrank.model import Block, ModelConfig, VariableWidthLM, structure
from carryrank.report import ranks, spearman


class Correctness(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)

    def test_history_distinguishes_new_and_reactivated(self):
        records = structure([4, 2, 6, 2, 6])
        self.assertEqual(records[2]["C"], [0, 1])
        self.assertEqual(records[2]["R"], [2, 3])
        self.assertEqual(records[2]["new"], [4, 5])
        self.assertEqual(records[2]["R_ages"], [1, 1])
        self.assertEqual(records[4]["R"], [2, 3, 4, 5])

    def test_pilot_bypass_ages(self):
        records = structure([192, 128, 64, 128, 192])
        self.assertEqual(set(records[3]["R_ages"]), {1})
        self.assertEqual(set(records[4]["R_ages"]), {3})
        self.assertEqual(len(records[4]["C"]), 128)

    def test_inactive_coordinates_and_gradients_are_carried(self):
        block = Block(16, 8)
        h = torch.randn(2, 7, 32, requires_grad=True)
        out = block(h)
        self.assertTrue(torch.equal(out[..., 16:], h[..., 16:]))
        out[..., 16:].sum().backward()
        self.assertTrue(torch.equal(h.grad[..., 16:], torch.ones_like(h.grad[..., 16:])))

    def test_causality(self):
        model = VariableWidthLM(ModelConfig([32, 16, 32], head_dim=8, seq_len=8)).eval()
        x = torch.randint(256, (2, 8))
        y = x.clone()
        y[:, 5:] = torch.randint(256, (2, 3))
        with torch.no_grad():
            self.assertTrue(torch.allclose(model(x)[:, :5], model(y)[:, :5], atol=1e-6))

    def test_column_mask_zero_initialization_and_merge(self):
        base = nn.Linear(12, 8, bias=False)
        module = GroupedLinear(base, [([0, 2, 7], 2), ([3, 5], 1)])
        x = torch.randn(2, 4, 12)
        self.assertTrue(torch.equal(module(x), base(x)))
        with torch.no_grad():
            for factor in module.factors:
                factor.B.normal_()
        merged = module.merged()
        delta = merged.weight - base.weight
        self.assertTrue(torch.equal(delta[:, [1, 4, 6, 8, 9, 10, 11]], torch.zeros(8, 7)))
        self.assertTrue(torch.allclose(module(x), merged(x), atol=1e-6))

    def test_training_leaves_base_frozen_and_merge_equivalent(self):
        model = VariableWidthLM(ModelConfig([32, 16, 32], head_dim=8, seq_len=8))
        before = base_digest(model)
        name, _, rec = next(model.targets())
        group = {"id": "test", "module": name, "indices": rec["R"], "cost": 48}
        count = attach(model, [group], {"test": 2})
        self.assertEqual(count, 96)
        self.assertEqual(base_digest(model), before)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
        x = torch.randint(256, (2, 8))
        loss = model(x).square().mean()
        loss.backward()
        optimizer.step()
        self.assertEqual(base_digest(model), before)
        with torch.no_grad():
            expected = model(x)
            merge_all(model)
            self.assertTrue(torch.allclose(model(x), expected, atol=1e-6))

    def test_knapsack_matches_independent_bruteforce(self):
        groups = [{"id": "a", "cost": 3, "m": 4, "n": 4},
                  {"id": "b", "cost": 4, "m": 4, "n": 4},
                  {"id": "c", "cost": 7, "m": 4, "n": 4}]
        utility = {"a": [0, 5, 8, 10], "b": [0, 6, 11, 12], "c": [0, 14, 16, 17]}
        for budget in range(1, 30):
            candidates = [(sum(utility[g["id"]][k] for g, k in zip(groups, combo)), combo)
                          for combo in itertools.product(range(4), repeat=3)
                          if sum(g["cost"] * k for g, k in zip(groups, combo)) == budget]
            if not candidates:
                with self.assertRaises(ValueError):
                    allocate(groups, utility, budget, 3)
            else:
                allocation, spent, value = allocate(groups, utility, budget, 3)
                self.assertEqual(spent, budget)
                self.assertAlmostEqual(value, max(candidates)[0])

    def test_budget_matching_all_main_shapes(self):
        groups = []
        for layer, width, c in [(3, 128, 64), (4, 192, 128)]:
            for projection in ("q", "v"):
                for label, n in [("C", c), ("R", width - c)]:
                    groups.append({"id": f"{layer}/{projection}/{label}", "label": label,
                                   "cost": width + n, "m": width, "n": n})
        for label in ("C", "R"):
            restricted = [g for g in groups if g["label"] == label]
            for budget in (1920, 3840):
                allocation, spent, _ = equal_parameters(restricted, budget, 16)
                self.assertEqual(spent, budget)
                self.assertEqual(sum(g["cost"] * allocation[g["id"]] for g in restricted), budget)

    def test_data_split_and_batch_boundaries(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "data/raw").mkdir(parents=True)
            content = bytes(range(256)) * 20
            (root / "data/raw/tinyshakespeare.txt").write_bytes(content)
            splits, manifest = prepare(root, {"url": "unused", "fractions": [0, .2, .4, .6, .8, 1]})
            records = manifest["splits"]
            self.assertEqual(sum(r["bytes"] for r in records), len(content))
            for left, right in zip(records, records[1:]):
                self.assertEqual(left["end"], right["start"])
            gen = torch.Generator().manual_seed(7)
            x, y = random_batch(splits["target_train"], 3, 16, gen, "cpu")
            self.assertTrue(torch.equal(x[:, 1:], y[:, :-1]))
            total = sum(y.numel() for _, y in evaluation_batches(splits["target_val"], 3, 16, 20, "cpu"))
            self.assertEqual(total, 320)

    def test_rank_statistics_handle_ties(self):
        self.assertTrue(np.allclose(ranks([3, 1, 1, 2]), [4, 1.5, 1.5, 3]))
        self.assertAlmostEqual(spearman([1, 2, 3], [3, 2, 1]), -1.0)
        self.assertIsNone(spearman([1, 1], [2, 3]))


if __name__ == "__main__":
    unittest.main()
