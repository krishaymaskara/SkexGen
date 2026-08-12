"""PyTorch-independent tests for accepted ADR-0011."""

from __future__ import annotations

import ast
from pathlib import Path
import unittest

from prototype.graph_encoder.errors import GraphEncoderError
from prototype.graph_encoder import representation_probe as probe


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "prototype/graph_encoder/representation_probe.py"
RUNNER = ROOT / "prototype/graph_encoder/adroit/ge1_representation_probe_cpu.slurm"


def _threshold_rows():
    rows = []
    for family_index in range(10):
        label = family_index % 5
        rows.append({
            "key": "f{}:0".format(family_index),
            "family_id": "f{}".format(family_index),
            "class_index": label,
            "value": float(label),
        })
    return tuple(rows)


class RepresentationProbeContractTests(unittest.TestCase):
    def test_exact_protocol_source_and_hashes_are_frozen(self):
        contract = probe.probe_contract()
        self.assertEqual(
            contract["protocol_version"],
            "GE1-C7-REPAIRED-REPRESENTATION-PROBE-v1",
        )
        self.assertEqual(
            contract["source"]["commit"],
            "705eb820f7a15d64fe650df7350b1553b2bc8172",
        )
        self.assertEqual(contract["source"]["slurm_job_id"], "3345280")
        self.assertEqual(len(contract["source"]["direct_sha256"]), 4)
        self.assertEqual(
            contract["cohort"]["family_ids_sha256"],
            "0888d517561ca24c99065465f8a2378e0e8e457f4bd20024e7f983956456f9ce",
        )

    def test_feature_and_probe_dimensions_are_exact(self):
        self.assertEqual(probe.FEATURE_DIMENSIONS, {"A": 1, "B": 1, "C": 32, "D": 36})
        self.assertEqual(probe.LINEAR_PARAMETER_COUNTS, {"C": 165, "D": 185})
        self.assertEqual(probe.MLP_PARAMETER_COUNTS, {"C": 309, "D": 341})
        self.assertEqual(probe.REGULARIZATION_GRID, (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0))
        self.assertEqual(probe.PERMUTATION_COUNT, 100)

    def test_D_uses_prequant_while_decoder_receives_memory(self):
        contract = probe.probe_contract()["features"]
        self.assertIn("prequant", contract["D"])
        self.assertEqual(
            contract["decoder_facing_memory"],
            "2x32_from_codebook_prequant_used_only_by_shared_decoder",
        )
        source = SOURCE.read_text(encoding="utf-8")
        self.assertIn("decoder_memory = encoded.memory.detach()", source)
        self.assertIn("continuous_bottleneck = encoded.prequant.detach()", source)
        self.assertIn("model.decoder(\n                    memory,", source)
        self.assertIn(
            "d_feature = prequant_values + type_one_hot + slot_one_hot", source
        )

    def test_nearest_grid_tie_chooses_smaller_magnitude(self):
        self.assertEqual(probe.nearest_grid_predictions((0.75,), "extrude"), (0,))
        self.assertEqual(probe.nearest_grid_predictions((67.5,), "revolve"), (0,))

    def test_dynamic_thresholds_recover_ordered_classes(self):
        values = (0.0, 0.1, 1.0, 1.1, 2.0, 2.1, 3.0, 3.1, 4.0, 4.1)
        labels = (0, 0, 1, 1, 2, 2, 3, 3, 4, 4)
        fitted = probe.fit_monotonic_thresholds(values, labels)
        self.assertEqual(fitted["metrics"]["accuracy"], 1.0)
        self.assertEqual(fitted["algorithm"], "dynamic_programming_with_empty_intervals")
        self.assertEqual(fitted["thresholds"], [0.55, 1.55, 2.55, 3.55])

    def test_dynamic_thresholds_allow_empty_intervals_and_tie_lexicographically(self):
        fitted = probe.fit_monotonic_thresholds((1.0, 2.0), (0, 4))
        self.assertEqual(fitted["metrics"]["accuracy"], 1.0)
        self.assertEqual(fitted["thresholds"], [1.5, 1.5, 1.5, 1.5])

    def test_grouped_thresholds_hold_complete_families_out(self):
        result = probe.grouped_threshold_predictions(_threshold_rows(), value_key="value")
        self.assertEqual(len(result["folds"]), 10)
        for fold in result["folds"]:
            self.assertNotIn(fold["held_out_family_id"], fold["train_family_ids"])

    def test_classification_metrics_always_use_five_classes(self):
        result = probe.classification_metrics((0, 4), (0, 0))
        self.assertEqual(len(result["confusion_matrix"]), 5)
        self.assertEqual(result["class_support"], [1, 0, 0, 0, 1])
        self.assertEqual(result["balanced_accuracy"], 0.2)

    def test_permutation_summary_uses_nearest_rank_and_plus_one_p_value(self):
        values = tuple(index / 100.0 for index in range(100))
        result = probe.permutation_summary(0.95, values)
        self.assertEqual(result["nearest_rank_95th_percentile"], 0.94)
        self.assertEqual(result["empirical_one_sided_p_value"], 6.0 / 101.0)

    def test_family_label_permutation_preserves_ee_ordered_blocks(self):
        rows = []
        for family_index in range(8):
            for operation_index in range(2):
                rows.append({
                    "key": "f{}:{}".format(family_index, operation_index),
                    "family_id": "f{}".format(family_index),
                    "operation_index": operation_index,
                    "operation_type": "extrude",
                    "operation_template": "EE",
                    "class_index": (family_index + operation_index) % 5,
                })
        permuted = probe.permute_family_label_blocks(
            rows,
            arm="flat",
            operation_type="extrude",
            feature_level="C",
            probe_kind="linear",
            permutation_index=0,
        )
        source_blocks = {
            tuple((family_index + operation_index) % 5 for operation_index in range(2))
            for family_index in range(8)
        }
        for family_index in range(8):
            observed = tuple(
                permuted["f{}:{}".format(family_index, operation_index)]
                for operation_index in range(2)
            )
            self.assertIn(observed, source_blocks)

    def test_permutation_is_deterministic_and_support_preserving(self):
        rows = []
        for family_index in range(8):
            rows.append({
                "key": "f{}:0".format(family_index),
                "family_id": "f{}".format(family_index),
                "operation_index": 0,
                "operation_type": "revolve",
                "operation_template": "R",
                "class_index": family_index % 5,
            })
        first = probe.permute_family_label_blocks(
            rows, arm="typed_graph", operation_type="revolve",
            feature_level="D", probe_kind="mlp", permutation_index=17,
        )
        second = probe.permute_family_label_blocks(
            rows, arm="typed_graph", operation_type="revolve",
            feature_level="D", probe_kind="mlp", permutation_index=17,
        )
        self.assertEqual(first, second)
        self.assertEqual(sorted(first.values()), sorted(item["class_index"] for item in rows))

    def test_geometry_loss_audit_matches_frozen_coefficients(self):
        audit = probe.geometry_loss_audit()
        self.assertEqual(audit["smooth_l1_beta"], 1.0)
        self.assertEqual(audit["cohort_average_coefficients"]["all_operation_channels"], "19/30")
        self.assertEqual(audit["cohort_average_coefficients"]["revolve"], "11/120")
        self.assertEqual(audit["per_operation_mean_coefficients"]["all_48"], "19/45")
        self.assertFalse(audit["strictly_linear_above_beta_region_reached"])
        self.assertFalse(audit["source_contract"]["beta_keyword_present"])

    def test_geometry_loss_source_audit_rejects_a_beta_override(self):
        source = """
def _per_example_smooth_l1(a, b):
    return torch.nn.functional.smooth_l1_loss(a, b, beta=0.5)
"""
        with self.assertRaises(GraphEncoderError) as context:
            probe.validate_geometry_loss_source_contract(source)
        self.assertEqual(context.exception.code, "geometry_loss_source_mismatch")

    def test_access_tracker_keeps_every_later_stage_and_partition_closed(self):
        access = probe.RepresentationProbeAccessTracker().declarations()
        for name in probe.PROTECTED_ACCESS_FIELDS:
            self.assertFalse(access[name])
        self.assertFalse(access["ge1_training_performed"])
        self.assertFalse(access["ge1_optimizer_constructed"])
        self.assertFalse(access["stage6_performed"])
        self.assertFalse(access["c8_or_later_performed"])

    def test_target_free_feature_validator_rejects_label_keys(self):
        with self.assertRaises(GraphEncoderError) as context:
            probe.validate_feature_rows(({"label": 1},))
        self.assertEqual(context.exception.code, "target_leakage_in_features")

    def test_extractor_signature_has_no_target_parameter(self):
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        function = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "extract_target_free_features"
        )
        arguments = [item.arg for item in function.args.args + function.args.kwonlyargs]
        self.assertNotIn("target", arguments)
        self.assertNotIn("targets", arguments)

    def test_no_protected_loader_or_training_call_enters_probe_source(self):
        source = SOURCE.read_text(encoding="utf-8")
        self.assertNotIn("load_development", source)
        self.assertNotIn("load_partition_physical_examples", source)
        self.assertNotIn("run_ge1_training(", source)
        self.assertNotIn("save_ge1_checkpoint(", source)
        self.assertNotIn("save_training_checkpoint(", source)
        self.assertNotIn("secondary_systematic_validation", source)

    def test_runner_is_prepared_but_contains_no_submission_command(self):
        if not RUNNER.exists():
            self.skipTest("runner is added with the implementation patch")
        source = RUNNER.read_text(encoding="utf-8")
        self.assertNotIn("sbatch ", source)
        self.assertIn("ge1_representation_probe_cpu.slurm", str(RUNNER))
        self.assertIn("GE1-C7-REPAIRED-REPRESENTATION-PROBE-v1", source)
        self.assertIn('"PROBE_FEATURE_VERSION" in package.__all__', source)
        self.assertNotIn("probe.PROBE_FEATURE_VERSION in package.__all__", source)


if __name__ == "__main__":
    unittest.main()
