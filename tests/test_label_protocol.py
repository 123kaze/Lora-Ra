"""Check that random controls remain equal-size, distinct and paired across profiles."""
import unittest

from carryrank.labels import group_schemes


class LabelControlTests(unittest.TestCase):
    def test_random_partitions_are_distinct_complete_and_paired(self):
        seeds = [11, 22, 33, 44, 55]
        long = group_schemes([128, 128, 64, 64, 64, 64, 64, 128], seeds)
        short = group_schemes([128, 64, 64, 64, 64, 128, 64, 128], seeds)
        self.assertEqual(long, short)
        for groups in long.values():
            for projection in ('q_proj', 'v_proj'):
                a, b = [g for g in groups if g['module'].endswith(projection)]
                self.assertEqual((a['cost'], b['cost']), (192, 192))
                self.assertFalse(set(a['indices']) & set(b['indices']))
                self.assertEqual(sorted(a['indices'] + b['indices']), list(range(128)))


if __name__ == '__main__':
    unittest.main()
