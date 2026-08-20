"""Small real-PyTorch checks for the disposable linear count probe."""

import unittest

try:
    import torch
except ImportError:
    torch = None

from prototype.graph_encoder.memory_count_probe import (
    MEMORY_COUNT_CLASSES,
    _count_metrics,
    fit_count_probe,
)


def _separable_rows():
    totals = (100, 100, 100, 107)
    rows = []
    family_index = 0
    for class_index, (node_count, total) in enumerate(zip(
        MEMORY_COUNT_CLASSES, totals
    )):
        for replica in range(total):
            one_hot = [0.0] * 4
            one_hot[class_index] = 10.0
            jitter = (replica % 7) * 1e-4
            feature = tuple(one_hot) + (jitter,)
            rows.append({
                "family_id": "family-{:04d}".format(family_index),
                "node_count": node_count,
                "memory": feature,
                "prequant": feature,
            })
            family_index += 1
    return tuple(rows)


@unittest.skipIf(torch is None, "PyTorch unavailable")
class MemoryCountProbeRuntimeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_separable_count_is_recovered_above_both_controls(self):
        result = fit_count_probe(_separable_rows(), "memory")
        self.assertTrue(result["linearly_recoverable"])
        self.assertEqual(result["test_metrics"]["balanced_accuracy"], 1.0)
        self.assertGreaterEqual(
            result["test_metrics"]["balanced_accuracy"]
            - result["majority_class_baseline"]["metrics"]["balanced_accuracy"],
            0.10,
        )
        self.assertGreaterEqual(
            result["test_metrics"]["balanced_accuracy"]
            - result["shuffled_label_control"]["metrics"]["balanced_accuracy"],
            0.10,
        )

    def test_metrics_are_four_way_balanced(self):
        metrics = _count_metrics((0, 1, 2, 3), (0, 1, 1, 3))
        self.assertEqual(metrics["accuracy"], 0.75)
        self.assertEqual(metrics["balanced_accuracy"], 0.75)
        self.assertEqual(metrics["confusion_matrix"][2][1], 1)


if __name__ == "__main__":
    unittest.main()
