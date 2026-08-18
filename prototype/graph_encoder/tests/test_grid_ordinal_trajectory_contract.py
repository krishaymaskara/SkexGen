"""Pure contracts for the corpus-free grid-ordinal trajectory diagnostic."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import tempfile
import unittest

from prototype.graph_encoder import grid_ordinal_trajectory as diagnostic
from prototype.graph_encoder.errors import GraphEncoderError
from prototype.graph_encoder.grid_magnitude import (
    NORMALIZED_GRIDS,
    OPERATION_NODE_TYPE_IDS,
    OPERATION_TYPES,
    SERIALIZED_CHANNELS,
)
from prototype.graph_encoder.grid_ordinal_trajectory_audit import (
    audit_grid_ordinal_trajectory,
)


COMMIT = "a" * 40
JOB_ID = "123456"


def _fake_rows():
    rows = []
    for condition in diagnostic.condition_matrix():
        for step in range(diagnostic.OPTIMIZER_UPDATES + 1):
            rows.append({
                "event": "grid_ordinal_trajectory_step",
                "version": diagnostic.TRAJECTORY_VERSION,
                "condition_id": condition["condition_id"],
                "step": step,
                "scientific_training": False,
            })
    return rows


def _fake_summary():
    return {
        "diagnostic_version": diagnostic.DIAGNOSTIC_VERSION,
        "trajectory_version": diagnostic.TRAJECTORY_VERSION,
        "artifact_version": diagnostic.ARTIFACT_VERSION,
        "source_commit": COMMIT,
        "slurm_job_id": JOB_ID,
        "condition_count": diagnostic.EXPECTED_CONDITION_COUNT,
        "trajectory_record_count": diagnostic.EXPECTED_TRAJECTORY_RECORD_COUNT,
        "conditions": [dict(row) for row in diagnostic.condition_matrix()],
        "hypothesis_evaluation": {
            "interpretation_category": "fixture",
        },
        "diagnostic_completed": True,
        **dict(diagnostic.ACCESS_RECORD),
    }


def _fake_resolved():
    return {
        "diagnostic_version": diagnostic.DIAGNOSTIC_VERSION,
        "trajectory_version": diagnostic.TRAJECTORY_VERSION,
        "artifact_version": diagnostic.ARTIFACT_VERSION,
        "source": {
            "git_commit": COMMIT,
            "git_branch": None,
            "detached_head": True,
            "git_dirty": False,
            "git_status_porcelain": [],
        },
        "runtime": {
            "python": "3.8.13",
            "pytorch": "1.11.0",
            "device": "cpu",
            "cuda_available": False,
            "cpu_threads": 1,
            "slurm_job_id": JOB_ID,
        },
        "arithmetic": diagnostic.diagnostic_arithmetic(),
        "condition_matrix": list(diagnostic.condition_matrix()),
        "access": dict(diagnostic.ACCESS_RECORD),
    }


def _finalized_fixture(root):
    parent = Path(root)
    staging = parent / "trajectory.incomplete-{}".format(JOB_ID)
    final = parent / "trajectory"
    staging.mkdir()
    diagnostic._atomic_write_json(staging / "resolved_config.json", _fake_resolved())
    diagnostic._atomic_write_jsonl(staging / "trajectory.jsonl", _fake_rows())
    diagnostic._atomic_write_json(staging / "summary.json", _fake_summary())
    diagnostic.finalize_artifact(staging, final)
    return final


class IdentityTests(unittest.TestCase):
    def test_separate_versions_are_frozen(self):
        self.assertEqual(
            diagnostic.DIAGNOSTIC_VERSION,
            "GE1-GRID-ORDINAL-TRAJECTORY-DIAGNOSTIC-v1",
        )
        self.assertEqual(
            diagnostic.ARTIFACT_VERSION,
            "GE1-GRID-ORDINAL-TRAJECTORY-ARTIFACT-v1",
        )

    def test_frozen_arithmetic(self):
        value = diagnostic.diagnostic_arithmetic()
        self.assertEqual(value["optimizer_updates_per_condition"], 200)
        self.assertEqual(value["recorded_steps_per_condition"], 201)
        self.assertEqual(value["trajectory_record_count"], 4020)
        self.assertEqual((value["batch_size"], value["model_width"]), (8, 32))
        self.assertEqual((value["learning_rate"], value["weight_decay"]), (0.001, 0.0))

    def test_complete_condition_matrix(self):
        rows = diagnostic.condition_matrix()
        self.assertEqual(len(rows), 20)
        self.assertEqual(len({row["condition_id"] for row in rows}), 20)
        observed = {
            (row["operation_type"], row["raw_gap_initialization"], row["target_class"])
            for row in rows
        }
        expected = {
            (operation, gap, target)
            for operation in OPERATION_TYPES
            for gap in (0.0, 0.5)
            for target in range(5)
        }
        self.assertEqual(observed, expected)

    def test_primary_and_control_roles(self):
        for row in diagnostic.condition_matrix():
            expected = "primary" if row["target_class"] in (1, 3) else "control"
            self.assertEqual(row["role"], expected)

    def test_real_frozen_grid_and_channel_identities(self):
        self.assertEqual(NORMALIZED_GRIDS["extrude"], (0.125, 0.25, 0.375, 0.5, 0.75))
        self.assertEqual(NORMALIZED_GRIDS["revolve"], (0.125, 0.25, 0.5, 0.75, 1.0))
        self.assertEqual(SERIALIZED_CHANNELS, {"extrude": 37, "revolve": 38})
        self.assertEqual(len(OPERATION_NODE_TYPE_IDS), 2)

    def test_authority_boundary_is_uniformly_false(self):
        self.assertTrue(diagnostic.ACCESS_RECORD["engineering_only"])
        for name, value in diagnostic.ACCESS_RECORD.items():
            if name != "engineering_only":
                self.assertFalse(value, name)


class SummaryTests(unittest.TestCase):
    def _records(self, target, classes, gap=0.0, operation="extrude"):
        result = []
        for step, decoded in enumerate(classes):
            result.append({
                "condition_id": diagnostic.condition_id(operation, gap, target),
                "condition_role": "primary" if target in (1, 3) else "control",
                "operation_type": operation,
                "target_class": target,
                "target_normalized_value": NORMALIZED_GRIDS[operation][target],
                "target_physical_value": diagnostic.PHYSICAL_GRIDS[operation][target],
                "raw_gap_initialization": gap,
                "step": step,
                "decoded_class": decoded,
                "decoded_class_equals_target": decoded == target,
                "loss": 1.0 / (step + 1),
                "projection_score": 0.0,
                "raw_gap_values": [gap] * 3,
                "ordered_biases": [0.0, -1.0, -2.0, -3.0],
                "logits": [1.0, -1.0, -2.0, -3.0],
                "sigmoid_probabilities": [0.7, 0.3, 0.2, 0.1],
                "minimum_decision_margin": 1.0,
            })
        return result

    def test_first_correct_step_and_persistence(self):
        values = [0] * 10 + [1] * 191
        summary = diagnostic.summarize_condition(self._records(1, values))
        self.assertEqual(summary["first_correct_step"], 10)
        self.assertTrue(summary["correctness_persisted_after_first_correct"])
        self.assertEqual(summary["class_transitions"], [{
            "step": 10, "from_class": 0, "to_class": 1,
        }])

    def test_never_correct_is_explicit(self):
        summary = diagnostic.summarize_condition(self._records(1, [0] * 201))
        self.assertIsNone(summary["first_correct_step"])
        self.assertIsNone(summary["correctness_persisted_after_first_correct"])
        self.assertTrue(summary["class_1_remained_class_0_through_step_200"])

    def test_gap_intervention_summary_is_not_a_gate(self):
        summary = diagnostic.summarize_condition(
            self._records(3, [4] * 200 + [3], gap=0.5, operation="revolve")
        )
        self.assertTrue(summary["diagnostic_gap_0p5_corrected_by_step_200"])

    def test_hypothesis_category_records_observation(self):
        summaries = []
        for row in diagnostic.condition_matrix():
            target = row["target_class"]
            if row["raw_gap_initialization"] == 0.0 and target == 1:
                classes = [2] + [0] * 200
            elif row["raw_gap_initialization"] == 0.0 and target == 3:
                classes = [0] + [4] * 200
            else:
                classes = [target] * 201
            summaries.append(diagnostic.summarize_condition(self._records(
                target, classes, gap=row["raw_gap_initialization"],
                operation=row["operation_type"],
            )))
        value = diagnostic.hypothesis_summary(summaries)
        self.assertEqual(
            value["interpretation_category"],
            "predicted_trap_and_gap_intervention_pattern_observed",
        )
        self.assertFalse(all(
            value["zero_gap_class_3_remained_class_4_all_steps"].values()
        ))
        self.assertTrue(
            value[
                "predicted_zero_gap_trap_observed_at_step_200_for_both_operations"
            ]
        )
        self.assertFalse(value["outcome_is_an_artifact_validity_gate"])


class ArtifactTests(unittest.TestCase):
    def test_exact_atomic_artifact_verifies(self):
        with tempfile.TemporaryDirectory(prefix="grid-trajectory-") as root:
            final = _finalized_fixture(root)
            verified = diagnostic.verify_artifact(
                final, expected_commit=COMMIT, expected_slurm_job_id=JOB_ID
            )
            self.assertEqual(verified["regular_file_count"], 5)
            self.assertEqual(verified["trajectory_record_count"], 4020)
            self.assertEqual(
                sorted(path.name for path in final.iterdir()),
                ["SHA256SUMS", "artifact_manifest.json", "resolved_config.json", "summary.json", "trajectory.jsonl"],
            )

    def test_checksum_tampering_fails(self):
        with tempfile.TemporaryDirectory(prefix="grid-trajectory-") as root:
            final = _finalized_fixture(root)
            with (final / "trajectory.jsonl").open("ab") as handle:
                handle.write(b"{}\n")
            with self.assertRaises(GraphEncoderError):
                diagnostic.verify_artifact(
                    final, expected_commit=COMMIT, expected_slurm_job_id=JOB_ID
                )

    def test_wrong_job_identity_fails(self):
        with tempfile.TemporaryDirectory(prefix="grid-trajectory-") as root:
            final = _finalized_fixture(root)
            with self.assertRaises(GraphEncoderError):
                diagnostic.verify_artifact(
                    final, expected_commit=COMMIT, expected_slurm_job_id="999"
                )

    def test_slurm_identity_validator_rejects_malformed_values(self):
        for value in (None, "", "12a", "-1", 123):
            with self.subTest(value=value), self.assertRaises(GraphEncoderError):
                diagnostic.validate_slurm_job_id(value)


class SourceBoundaryTests(unittest.TestCase):
    def test_python_38_grammar(self):
        root = Path(__file__).resolve().parents[1]
        for name in ("grid_ordinal_trajectory.py", "grid_ordinal_trajectory_audit.py"):
            path = root / name
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path), feature_version=(3, 8))

    def test_no_scientific_or_data_loader_imports(self):
        path = Path(__file__).resolve().parents[1] / "grid_ordinal_trajectory.py"
        source = path.read_text(encoding="utf-8")
        for value in ("load_train", "load_development", "run_ge1_training", "load_ge1_checkpoint"):
            self.assertNotIn(value, source)

    def test_runner_and_source_pass_structural_audit(self):
        package = Path(__file__).resolve().parents[1]
        runner = package / "adroit" / "ge1_grid_ordinal_trajectory_cpu.slurm"
        result = audit_grid_ordinal_trajectory(package, runner)
        self.assertEqual(result["bind_count"], 2)
        self.assertEqual(result["scientific_entry_point_count"], 0)


if __name__ == "__main__":
    unittest.main()
