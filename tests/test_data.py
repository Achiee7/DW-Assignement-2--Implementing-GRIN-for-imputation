"""Tests for protocol details that can otherwise create optimistic test scores."""
import unittest

import numpy as np
import pandas as pd

from grin_reproduction.data import _covered, split_aqi36


class SplitProtocolTests(unittest.TestCase):
    def setUp(self):
        self.timestamps = pd.date_range("2014-05-01 01:00", periods=8759, freq="h")
        self.window = 36
        self.train, self.val, self.test = split_aqi36(self.timestamps, self.window)

    def test_observations_are_disjoint_across_splits(self):
        coverage = [_covered(starts, self.window, len(self.timestamps))
                    for starts in (self.train, self.val, self.test)]
        self.assertFalse(np.any(coverage[0] & coverage[1]))
        self.assertFalse(np.any(coverage[0] & coverage[2]))
        self.assertFalse(np.any(coverage[1] & coverage[2]))

    def test_test_windows_stay_entirely_in_published_test_months(self):
        times = self.test[:, None] + np.arange(self.window)
        self.assertTrue(np.isin(self.timestamps.month[times.ravel()], [3, 6, 9, 12]).all())

    def test_split_counts_match_released_calendar_protocol(self):
        self.assertEqual((len(self.train), len(self.val), len(self.test)), (4948, 564, 2788))


if __name__ == "__main__":
    unittest.main()
