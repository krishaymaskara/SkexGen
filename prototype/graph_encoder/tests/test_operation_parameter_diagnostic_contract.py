"""Pure contract tests for the C7-v2 operation-parameter diagnostic."""

from __future__ import annotations

import ast
import copy
import hashlib
from pathlib import Path
import tempfile
import unittest

from prototype.graph_encoder.errors import GraphEncoderError
from prototype.graph_encoder import operation_parameter_diagnostic as diagnostic


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "prototype/graph_encoder/operation_parameter_diagnostic.py"
SLURM = (
    ROOT
    / "prototype/graph_encoder/adroit/"
    "ge1_operation_parameter_diagnostic_cpu.slurm"
)


def _checkpoint_fixture(arm):
    return {
        "encoder_type": arm,
        "completed_epoch": 200,
        "selected_checkpoint_epoch": 200,
        "selected_experimental_checkpoint": True,
        "seed": 2026,
        "source_digest": "tree-digest",
        "provenance": {
            "git_commit": diagnostic.SOURCE_C7_V2_COMMIT,
            "slurm_job_id": diagnostic.SOURCE_C7_V2_JOB_ID,
            "encoder_arm": arm,
            "device": "cpu",
            "python_version": "3.8.13",
            "pytorch_version": "1.11.0",
            "source_tree_sha256": "tree-digest",
        },
    }


def _arm_result(arm):
    return {
        "arm": arm,
        "executor_result": copy.deepcopy(diagnostic.EXECUTOR_UNAVAILABLE),
        "legal_parameter_geometric_incompatibility": copy.deepcopy(
            diagnostic.GEOMETRIC_INCOMPATIBILITY_UNASSESSABLE
        ),
        "operations": [],
    }


def _minimal_record():
    return {
        "schema_version": diagnostic.DIAGNOSTIC_RECORD_VERSION,
        "family_diagnostics": [
            {
                "family_id": family_id,
                "arm_results": [
                    _arm_result("flat"),
                    _arm_result("typed_graph"),
                ],
            }
            for family_id in diagnostic.FAILED_FAMILY_IDS
        ],
    }


class CorrectedValidityContractTests(unittest.TestCase):
    def test_analytic_domain_and_boundaries_are_explicit(self):
        negative = diagnostic.classify_operation_parameter(-0.25, 37, True)
        zero = diagnostic.classify_operation_parameter(0.0, 38, True)
        positive = diagnostic.classify_operation_parameter(0.25, 38, True)
        inactive = diagnostic.classify_operation_parameter(0.25, 37, False)
        self.assertEqual(negative["classification"], "negative_value")
        self.assertEqual(negative["physical_value"], -1.0)
        self.assertFalse(negative["analytic_legal"])
        self.assertEqual(zero["classification"], "zero_boundary_value")
        self.assertFalse(zero["analytic_legal"])
        self.assertEqual(positive["physical_value"], 90.0)
        self.assertTrue(positive["analytic_legal"])
        self.assertTrue(positive["normalized_within_declared_network_range"])
        self.assertEqual(inactive["classification"], "inactive_channel")
        self.assertFalse(inactive["analytic_legal"])

    def test_no_kernel_results_are_structurally_unavailable(self):
        contract = diagnostic.corrected_validity_contract()
        self.assertEqual(
            contract["executor_result"], diagnostic.EXECUTOR_UNAVAILABLE
        )
        self.assertEqual(
            contract["legal_parameter_geometric_incompatibility"],
            diagnostic.GEOMETRIC_INCOMPATIBILITY_UNASSESSABLE,
        )
        self.assertIn("reconstruction_target.valid", contract["strict_conversion"])
        self.assertIn(
            "controlled_domain.valid",
            contract["analytic_controlled_domain_validity"],
        )

    def test_record_requires_both_arms_and_corrected_unavailability(self):
        record = _minimal_record()
        self.assertIs(diagnostic.validate_minimal_diagnostic_record(record), record)
        record["family_diagnostics"][0]["arm_results"][0][
            "executor_result"
        ] = {"available": True, "result": "invented"}
        with self.assertRaisesRegex(GraphEncoderError, "no-kernel"):
            diagnostic.validate_minimal_diagnostic_record(record)

    def test_raw_payload_keys_are_rejected_recursively(self):
        record = _minimal_record()
        record["family_diagnostics"][0]["arm_results"][0][
            "raw_cad_history"
        ] = {"forbidden": True}
        with self.assertRaisesRegex(GraphEncoderError, "forbidden keys"):
            diagnostic.validate_minimal_diagnostic_record(record)

    def test_case_classification_supports_scalar_mechanism_only(self):
        failed_operation = {
            "operation_index": 0,
            "operation_type": "extrude",
            "normalized_predicted_value_post_tanh": -0.25,
            "physical_predicted_value": -1.0,
            "raw_head_output_pre_tanh": -0.255,
            "analytic_parameter_contract": {
                "analytic_legal": False,
                "classification": "negative_value",
                "normalized_within_declared_network_range": True,
            },
            "normalization_consistency": True,
            "geometry_mask_active": True,
            "target_geometry_mask_active": True,
            "channel_selection_consistency": True,
        }
        family = {
            "failing_arm": "flat",
            "arm_results": [
                {
                    "arm": "flat",
                    "failed_operation_index": 0,
                    "first_failure_code": "invalid_operation_parameter",
                    "operations": [failed_operation],
                },
                {"arm": "typed_graph", "operations": []},
            ],
        }
        result = diagnostic._classify_failed_family(family)
        hypotheses = result["hypothesis_classification"]
        self.assertEqual(
            hypotheses["zero_negative_or_boundary_value"]["status"],
            "supported",
        )
        self.assertEqual(
            hypotheses["legal_but_geometrically_incompatible_parameter"][
                "status"
            ],
            "unassessable",
        )
        self.assertEqual(
            hypotheses["encoder_memory_error"]["status"], "unresolved"
        )
        self.assertFalse(result["root_cause_attribution"]["resolved"])


class FrozenIdentityAndReproductionTests(unittest.TestCase):
    def test_checkpoint_commit_job_arm_epoch_and_runtime_are_enforced(self):
        fixture = _checkpoint_fixture("flat")
        self.assertIs(
            diagnostic.validate_checkpoint_identity_fields(
                fixture, expected_arm="flat"
            ),
            fixture,
        )
        mutations = (
            ("completed_epoch", 199),
            ("encoder_type", "typed_graph"),
            ("seed", 7),
        )
        for field, value in mutations:
            changed = copy.deepcopy(fixture)
            changed[field] = value
            with self.assertRaises(GraphEncoderError, msg=field):
                diagnostic.validate_checkpoint_identity_fields(
                    changed, expected_arm="flat"
                )
        changed = copy.deepcopy(fixture)
        changed["provenance"]["slurm_job_id"] = "999"
        with self.assertRaises(GraphEncoderError):
            diagnostic.validate_checkpoint_identity_fields(
                changed, expected_arm="flat"
            )

    def test_checkpoint_file_hash_is_enforced_before_loading(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "checkpoint.pt"
            payload = b"immutable-checkpoint-fixture\n"
            path.write_bytes(payload)
            expected = hashlib.sha256(payload).hexdigest()
            self.assertEqual(
                diagnostic.validate_checkpoint_file_hash(path, expected),
                expected,
            )
            with self.assertRaisesRegex(GraphEncoderError, "SHA-256 differs"):
                diagnostic.validate_checkpoint_file_hash(path, "0" * 64)

    def test_original_stable_metrics_must_match_before_acceptance(self):
        templates = dict(diagnostic.EXPECTED_TEMPLATES)
        values = {}
        for family_id in templates:
            values[family_id] = {
                "exact_node_sequence": 1.0,
                "exact_graph": 1.0,
                "strict_conversion": 1.0,
                "depends_on_exactness": 1.0,
                "complete_executable_validity": (
                    0.0
                    if family_id in diagnostic.FAILED_FAMILIES_BY_ARM["flat"]
                    else 1.0
                ),
            }
        metrics = {
            "condition": "P_true",
            "available": True,
            "family_values": values,
            "metric_computation_seconds": 1.25,
        }
        reproduced = copy.deepcopy(metrics)
        reproduced["metric_computation_seconds"] = 99.0
        record = diagnostic.validate_reproduced_c7_metrics(
            metrics,
            reproduced,
            arm="flat",
            templates_by_family=templates,
        )
        self.assertEqual(record["status"], "exact_stable_reproduction_passed")
        reproduced["family_values"][next(iter(values))]["exact_graph"] = 0.0
        with self.assertRaisesRegex(GraphEncoderError, "stable P_true"):
            diagnostic.validate_reproduced_c7_metrics(
                metrics,
                reproduced,
                arm="flat",
                templates_by_family=templates,
            )

    def test_minimal_artifact_is_checksummed_and_has_no_checkpoints(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            staging = root / "diagnostic.incomplete-123456"
            final = root / "diagnostic"
            staging.mkdir()
            access = diagnostic.DiagnosticAccessTracker(
                operation_template_manifest_accessed=True,
                operation_template_train_payload_accessed=True,
                scaled_train_payload_accessed=True,
            ).declarations(completed=True)
            diagnostic._atomic_write_json(
                staging / "diagnosis.json", _minimal_record()
            )
            diagnostic._atomic_write_json(staging / "resolved_config.json", {
                "protocol_version": diagnostic.DIAGNOSTIC_PROTOCOL_VERSION,
                "artifact_version": diagnostic.DIAGNOSTIC_ARTIFACT_VERSION,
                "diagnostic_source": {"git_commit": "a" * 40},
                "source_c7_v2": {
                    "commit": diagnostic.SOURCE_C7_V2_COMMIT,
                    "job_id": diagnostic.SOURCE_C7_V2_JOB_ID,
                },
                "source_c7_v2_immutable": True,
                "runtime": {"slurm_job_id": "123456"},
                "access": access,
            })
            diagnostic._atomic_write_jsonl(staging / "metrics.jsonl", ({
                "event": "diagnostic_completed",
                "diagnostic_completed": True,
            },))
            diagnostic.finalize_diagnostic_artifact(staging, final)
            verified = diagnostic.verify_diagnostic_artifact(
                final,
                expected_commit="a" * 40,
                expected_slurm_job_id="123456",
            )
            self.assertTrue(verified["diagnostic_completed"])
            self.assertEqual(
                sorted(path.name for path in final.iterdir()),
                [
                    "SHA256SUMS",
                    "artifact_manifest.json",
                    "diagnosis.json",
                    "metrics.jsonl",
                    "resolved_config.json",
                ],
            )
            self.assertFalse(tuple(final.rglob("*.pt")))


class StaticReachabilityTests(unittest.TestCase):
    def test_source_has_no_training_or_repair_call_path(self):
        source = SOURCE.read_text(encoding="utf-8")
        tree = ast.parse(source)
        calls = set()
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                imports.update(alias.name for alias in node.names)
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    calls.add(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    calls.add(node.func.attr)
        self.assertNotIn("load_development", imports)
        self.assertNotIn("load_partition_physical_examples", imports)
        self.assertIn("load_train", imports)
        for forbidden in (
            "backward",
            "optimizer",
            "run_ge1_training",
            "save_training_checkpoint",
            "step",
            "train",
            "repair_decoder",
        ):
            self.assertNotIn(forbidden, calls)
        self.assertNotIn("secondary_systematic_validation", source)
        self.assertNotIn("load_development", source)

    def test_runner_pins_runtime_scope_and_no_training(self):
        source = SLURM.read_text(encoding="utf-8")
        self.assertIn("3.8.13", source)
        self.assertIn("1.11.0", source)
        self.assertIn("CUDA_VISIBLE_DEVICES", source)
        self.assertIn("operation_parameter_diagnostic", source)
        self.assertIn("SOURCE_C7_V2_ARTIFACT", source)
        self.assertIn("sha256sum -c SHA256SUMS", source)
        self.assertNotIn("-m prototype.graph_encoder.training", source)
        self.assertNotIn("-m prototype.graph_encoder.c7_v2 ", source)
        self.assertEqual(
            source.count("-m prototype.graph_encoder.operation_parameter_diagnostic run"),
            1,
        )
        self.assertNotIn("sbatch ", source)

    def test_python_38_grammar(self):
        ast.parse(
            SOURCE.read_text(encoding="utf-8"),
            filename=str(SOURCE),
            feature_version=(3, 8),
        )


if __name__ == "__main__":
    unittest.main()
