import unittest

import numpy as np

from route_a.run_ct13_real import _stage_metrics


class StageMetricsTest(unittest.TestCase):
    def test_confusion_metrics_and_task1_na(self):
        scores = np.array([[1., 0.], [1., 0.], [1., 0.], [1., 0.]])
        stage, rows, _ = _stage_metrics(scores, np.array([0, 0, 0, 1]), [0, 1], [0, 1], {1})
        self.assertEqual(stage["ba"], 0.5)
        self.assertAlmostEqual(stage["macro_f1"], 3 / 7)
        self.assertTrue(np.isnan(stage["old_ba"]))
        self.assertTrue(np.isnan(stage["hm"]))
        self.assertEqual([(r["tp"], r["fp"], r["fn"]) for r in rows], [(3, 1, 0), (0, 0, 1)])

    def test_ties_follow_original_label_across_column_orders(self):
        labels = np.array([2, 5])
        for seen in ([5, 2], [2, 5]):
            _, _, pred = _stage_metrics(np.ones((2, 2)), labels, seen, seen, set())
            np.testing.assert_array_equal(pred, [2, 2])

    def test_missing_support_blocks(self):
        with self.assertRaisesRegex(ValueError, "BLOCKED_VAL_MISSING_SUPPORT:1"):
            _stage_metrics(np.array([[1., 0.]]), np.array([0]), [0, 1], [0, 1], set())


if __name__ == "__main__":
    unittest.main()
