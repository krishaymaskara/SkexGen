"""Pure contract coverage for the ADR-0016 prerequisite probe."""

from pathlib import Path
import unittest

from prototype.graph_encoder.memory_count_probe import (
    MEMORY_COUNT_ACCURACY_MIN,
    MEMORY_COUNT_CLASSES,
    MEMORY_COUNT_CONTROL_MARGIN_MIN,
    MEMORY_COUNT_FEATURES,
    MEMORY_COUNT_PROBE_VERSION,
    deterministic_count_split,
)
from prototype.graph_encoder.memory_count_probe_audit import audit


ROOT = Path(__file__).resolve().parents[3]
RUNNER = ROOT / "prototype/graph_encoder/adroit/ge1_memory_count_probe_cpu.slurm"


def _rows():
    counts = (100, 100, 100, 107)
    result = []
    index = 0
    for node_count, total in zip(MEMORY_COUNT_CLASSES, counts):
        for unused in range(total):
            del unused
            result.append({
                "family_id": "family-{:04d}".format(index),
                "node_count": node_count,
            })
            index += 1
    return tuple(result)


class MemoryCountProbeContractTests(unittest.TestCase):
    def test_frozen_probe_identity_and_rule(self):
        self.assertEqual(MEMORY_COUNT_PROBE_VERSION,
                         "GE1-MEMORY-COUNT-LINEAR-PROBE-v1")
        self.assertEqual(MEMORY_COUNT_CLASSES, (4, 5, 7, 8))
        self.assertEqual(MEMORY_COUNT_FEATURES, ("memory", "prequant"))
        self.assertEqual(MEMORY_COUNT_ACCURACY_MIN, 0.90)
        self.assertEqual(MEMORY_COUNT_CONTROL_MARGIN_MIN, 0.10)

    def test_split_is_family_disjoint_stratified_and_order_invariant(self):
        rows = _rows()
        train, test, evidence = deterministic_count_split(rows)
        reversed_result = deterministic_count_split(tuple(reversed(rows)))
        self.assertEqual((train, test, evidence), reversed_result)
        train_ids = {item["family_id"] for item in train}
        test_ids = {item["family_id"] for item in test}
        self.assertFalse(train_ids & test_ids)
        self.assertEqual(len(train_ids | test_ids), 407)
        self.assertEqual(set(evidence["per_class"]), {"4", "5", "7", "8"})
        self.assertTrue(all(
            row["train"] > 0 and row["test"] > 0
            for row in evidence["per_class"].values()
        ))

    def test_runner_and_source_audit(self):
        result = audit(ROOT / "prototype/graph_encoder", RUNNER)
        self.assertEqual(result["version"], "GE1-MEMORY-COUNT-PROBE-AUDIT-v1")
        self.assertEqual(result["train_loader_call_count"], 1)
        self.assertEqual(result["exact_checkpoint_file_bind_count"], 6)
        self.assertEqual(result["protected_bind_count"], 0)
        self.assertFalse(result["submission_command_invoked"])

    def test_frozen_flat_source_is_unchanged_at_adr_parent(self):
        source = ROOT / "prototype/flat_baseline/constrained_v6_autonomous.py"
        self.assertTrue(source.is_file())
        text = source.read_text(encoding="utf-8")
        self.assertIn("def greedy_decode_v6_from_memory(", text)
        self.assertNotIn("memory_count_probe", text)


if __name__ == "__main__":
    unittest.main()
