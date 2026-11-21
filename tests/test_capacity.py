"""Prevent post-hoc rank selection or seed averaging from bypassing the gate."""
import copy
import unittest

from carryrank.capacity import capacity_gate, final_v_groups
from carryrank.experiment import ROOT, read_json
from carryrank.model import structure


class CapacityProtocolTests(unittest.TestCase):
    def setUp(self):
        self.config = read_json(ROOT / 'configs/crossdomain_capacity.json')
        self.rows = []
        for profile in self.config['widths']:
            for seed in self.config['adapter_seeds']:
                for rank, loss in [(1, 2.0), (4, 1.98), (8, 1.0)]:
                    self.rows.append(dict(profile=profile, base_seed=2026, adapter_seed=seed,
                                          rank=rank, step=600, val_loss=loss))

    def test_one_negative_pair_cannot_hide_in_mean(self):
        self.assertTrue(capacity_gate(self.rows, self.config)['pass'])
        rows = copy.deepcopy(self.rows)
        next(r for r in rows if r['profile'] == 'long' and r['adapter_seed'] == 11 and r['rank'] == 4)['val_loss'] = 2.001
        self.assertFalse(capacity_gate(rows, self.config)['pass'])

    def test_better_diagnostic_rank_cannot_replace_primary_contrast(self):
        for row in self.rows:
            if row['profile'] == 'short' and row['rank'] == 4:
                row['val_loss'] = 1.999
        self.assertFalse(capacity_gate(self.rows, self.config)['pass'])
        self.assertFalse(capacity_gate(self.rows, {**self.config, 'smoke': True})['pass'])

    def test_candidate_costs_and_actual_bypass_ages(self):
        for profile, age in [('long', 5), ('short', 1)]:
            widths = self.config['widths'][profile]
            self.assertEqual(set(structure(widths)[-1]['R_ages']), {age})
            groups = final_v_groups(widths)
            self.assertEqual([g['cost'] for g in groups], [192, 192])
            self.assertFalse(set(groups[0]['indices']) & set(groups[1]['indices']))
            self.assertEqual(sorted(groups[0]['indices'] + groups[1]['indices']), list(range(128)))


if __name__ == '__main__':
    unittest.main()
