"""Pure contracts for the extended grid-ordinal horizon diagnostic."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from prototype.graph_encoder import grid_ordinal_horizon as diagnostic
from prototype.graph_encoder.errors import GraphEncoderError
from prototype.graph_encoder.grid_magnitude import (
    NORMALIZED_GRIDS,
    OPERATION_TYPES,
    PHYSICAL_GRIDS,
    SERIALIZED_CHANNELS,
)
from prototype.graph_encoder.grid_ordinal_horizon_audit import (
    CPU_DISCOVERY_CUDA_SKIP_IDS,
    audit_grid_ordinal_horizon,
    validate_cpu_horizon_complete_discovery,
)


COMMIT = "b" * 40
JOB_ID = "654321"


def _fake_record(condition, step):
    target = condition["target_class"]
    decoded = 1 if step < 10 else target
    logits = [1.0 if cut < decoded else -1.0 for cut in range(4)]
    gradients = {
        name: {
            "before_clipping": [0.1] * size,
            "after_clipping": [0.1] * size,
            "before_clipping_norm": 0.1 * (size ** 0.5),
            "after_clipping_norm": 0.1 * (size ** 0.5),
        }
        for name, size in (("projection", 32), ("first_bias", 1), ("raw_gaps", 3))
    }
    gradients.update({
        "global_before_clipping": 0.6,
        "global_after_clipping": 0.6,
        "clip_norm": 1.0,
        "clipping_occurred": False,
    })
    return {
        "event": "grid_ordinal_horizon_step",
        "version": diagnostic.TRAJECTORY_VERSION,
        "condition_id": condition["condition_id"],
        "operation_type": condition["operation_type"],
        "target_class": target,
        "target_normalized_value": NORMALIZED_GRIDS[condition["operation_type"]][target],
        "target_physical_value": PHYSICAL_GRIDS[condition["operation_type"]][target],
        "raw_gap_initialization": 0.0,
        "step": step,
        "state_coordinate": "initialization" if step == 0 else "post_optimizer_update",
        "optimizer_updates_completed": step,
        "gradient_update_follows_record": step < diagnostic.OPTIMIZER_UPDATES,
        "loss": 1.0 / (step + 1),
        "projection_score": 0.0,
        "first_bias": 0.0,
        "raw_gap_values": [0.0, 0.0, 0.0],
        "ordered_biases": [0.0, -1.0, -2.0, -3.0],
        "logits": logits,
        "sigmoid_probabilities": [0.7 if value > 0 else 0.3 for value in logits],
        "decoded_class": decoded,
        "decoded_normalized_value": NORMALIZED_GRIDS[condition["operation_type"]][decoded],
        "decoded_physical_value": PHYSICAL_GRIDS[condition["operation_type"]][decoded],
        "minimum_decision_margin": 1.0,
        "gradients": gradients,
        "adamw_state": {
            "step": step,
            "projection": {"exp_avg": [0.0] * 32, "exp_avg_sq": [0.0] * 32},
            "first_bias": {"exp_avg": 0.0, "exp_avg_sq": 0.0},
            "raw_gaps": {"exp_avg": [0.0] * 3, "exp_avg_sq": [0.0] * 3},
        },
        "decoded_class_equals_target": decoded == target,
        "engineering_only": True,
        "scientific_training": False,
    }


def _fake_rows():
    return [
        _fake_record(condition, step)
        for condition in diagnostic.condition_matrix()
        for step in range(diagnostic.OPTIMIZER_UPDATES + 1)
    ]


def _grouped_summaries(rows):
    return [
        diagnostic.summarize_condition([
            row for row in rows if row["condition_id"] == condition["condition_id"]
        ])
        for condition in diagnostic.condition_matrix()
    ]


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
    rows = _fake_rows()
    summaries = _grouped_summaries(rows)
    summary = {
        "diagnostic_version": diagnostic.DIAGNOSTIC_VERSION,
        "trajectory_version": diagnostic.TRAJECTORY_VERSION,
        "artifact_version": diagnostic.ARTIFACT_VERSION,
        "source_commit": COMMIT,
        "slurm_job_id": JOB_ID,
        "condition_count": diagnostic.EXPECTED_CONDITION_COUNT,
        "trajectory_record_count": diagnostic.EXPECTED_TRAJECTORY_RECORD_COUNT,
        "conditions": summaries,
        "horizon_evaluation": diagnostic.horizon_summary(summaries),
        "diagnostic_completed": True,
        **dict(diagnostic.ACCESS_RECORD),
    }
    parent = Path(root)
    staging = parent / "horizon.incomplete-{}".format(JOB_ID)
    final = parent / "horizon"
    staging.mkdir()
    diagnostic._atomic_write_json(staging / "resolved_config.json", _fake_resolved())
    diagnostic._atomic_write_jsonl(staging / "trajectory.jsonl", rows)
    diagnostic._atomic_write_json(staging / "summary.json", summary)
    diagnostic.finalize_artifact(staging, final)
    return final


class IdentityTests(unittest.TestCase):
    def test_separate_versions_are_frozen(self):
        self.assertEqual(
            diagnostic.DIAGNOSTIC_VERSION,
            "GE1-GRID-ORDINAL-HORIZON-DIAGNOSTIC-v1",
        )
        self.assertEqual(
            diagnostic.ARTIFACT_VERSION,
            "GE1-GRID-ORDINAL-HORIZON-ARTIFACT-v1",
        )

    def test_frozen_arithmetic(self):
        value = diagnostic.diagnostic_arithmetic()
        self.assertEqual(value["optimizer_updates_per_condition"], 2000)
        self.assertEqual(value["recorded_steps_per_condition"], 2001)
        self.assertEqual(value["trajectory_record_count"], 12006)
        self.assertEqual(value["target_classes"], [2, 3, 4])
        self.assertEqual(value["checkpoint_steps"], [200, 500, 1000, 2000])

    def test_six_fresh_conditions(self):
        rows = diagnostic.condition_matrix()
        self.assertEqual(len(rows), 6)
        self.assertEqual(
            {(row["operation_type"], row["target_class"]) for row in rows},
            {(operation, target) for operation in OPERATION_TYPES for target in (2, 3, 4)},
        )
        self.assertTrue(all(row["fresh_seed"] == 2026 for row in rows))

    def test_real_grid_and_channel_identities(self):
        self.assertEqual(SERIALIZED_CHANNELS, {"extrude": 37, "revolve": 38})
        self.assertEqual(NORMALIZED_GRIDS["extrude"][2:], (0.375, 0.5, 0.75))
        self.assertEqual(NORMALIZED_GRIDS["revolve"][2:], (0.5, 0.75, 1.0))

    def test_authority_boundary(self):
        self.assertTrue(diagnostic.ACCESS_RECORD["engineering_only"])
        for name, value in diagnostic.ACCESS_RECORD.items():
            if name != "engineering_only":
                self.assertFalse(value, name)


class SummaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        condition = diagnostic.condition_matrix()[0]
        cls.records = [_fake_record(condition, step) for step in range(2001)]
        cls.summary = diagnostic.summarize_condition(cls.records)

    def test_first_positive_steps_and_first_correct(self):
        self.assertEqual(self.summary["first_positive_step_by_cut"], {
            "0": 0, "1": 10, "2": None, "3": None,
        })
        self.assertEqual(self.summary["first_correct_step"], 10)
        self.assertFalse(self.summary["cut_crossed_zero"]["0"])
        self.assertTrue(self.summary["cut_crossed_zero"]["1"])

    def test_required_checkpoint_values(self):
        self.assertEqual(
            set(self.summary["checkpoint_values"]), {"200", "500", "1000", "2000"}
        )
        self.assertEqual(self.summary["checkpoint_values"]["2000"]["adamw_step"], 2000)

    def test_required_interval_changes(self):
        self.assertEqual(
            set(self.summary["interval_changes"]),
            {"200-500", "500-1000", "1000-2000"},
        )

    def test_transitions_and_final_correctness(self):
        self.assertEqual(self.summary["class_transitions"], [
            {"step": 10, "from_class": 1, "to_class": 2},
        ])
        self.assertTrue(self.summary["final_class_correct"])

    def test_outcome_never_controls_validity(self):
        values = _grouped_summaries(_fake_rows())
        result = diagnostic.horizon_summary(values)
        self.assertFalse(result["outcome_is_an_artifact_validity_gate"])
        self.assertEqual(result["final_correct_condition_count"], 6)


class ArtifactTests(unittest.TestCase):
    def test_exact_atomic_artifact_verifies(self):
        with tempfile.TemporaryDirectory(prefix="grid-horizon-") as root:
            final = _finalized_fixture(root)
            result = diagnostic.verify_artifact(
                final, expected_commit=COMMIT, expected_slurm_job_id=JOB_ID
            )
            self.assertEqual(result["trajectory_record_count"], 12006)
            self.assertEqual(result["regular_file_count"], 5)

    def test_checksum_tampering_fails(self):
        with tempfile.TemporaryDirectory(prefix="grid-horizon-") as root:
            final = _finalized_fixture(root)
            with (final / "summary.json").open("ab") as handle:
                handle.write(b"{}\n")
            with self.assertRaises(GraphEncoderError):
                diagnostic.verify_artifact(
                    final, expected_commit=COMMIT, expected_slurm_job_id=JOB_ID
                )

    def test_wrong_job_identity_fails(self):
        with tempfile.TemporaryDirectory(prefix="grid-horizon-") as root:
            final = _finalized_fixture(root)
            with self.assertRaises(GraphEncoderError):
                diagnostic.verify_artifact(
                    final, expected_commit=COMMIT, expected_slurm_job_id="1"
                )

    def test_slurm_identity_rejects_malformed_values(self):
        for value in (None, "", "abc", "-1", 123):
            with self.subTest(value=value), self.assertRaises(GraphEncoderError):
                diagnostic.validate_slurm_job_id(value)


class SourceBoundaryTests(unittest.TestCase):
    def test_python_38_grammar(self):
        package = Path(__file__).resolve().parents[1]
        for name in ("grid_ordinal_horizon.py", "grid_ordinal_horizon_audit.py"):
            path = package / name
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path), feature_version=(3, 8))

    def test_no_scientific_or_data_loader_imports(self):
        path = Path(__file__).resolve().parents[1] / "grid_ordinal_horizon.py"
        source = path.read_text(encoding="utf-8")
        for value in ("load_train", "load_development", "run_ge1_training", "load_ge1_checkpoint"):
            self.assertNotIn(value, source)

    def test_runner_and_source_pass_structural_audit(self):
        package = Path(__file__).resolve().parents[1]
        runner = package / "adroit" / "ge1_grid_ordinal_horizon_cpu.slurm"
        result = audit_grid_ordinal_horizon(package, runner)
        self.assertEqual(result["bind_count"], 2)
        self.assertEqual(result["optimizer_state_audit"], "pass")
        source = runner.read_text(encoding="utf-8")
        self.assertNotIn("result.testsRun != 540", source)
        self.assertNotIn("all 540 graph-encoder tests", source)

        def result_for(skip_ids, tests_run=700, failures=(), errors=()):
            skipped = [
                (SimpleNamespace(id=lambda value=value: value), "CUDA unavailable")
                for value in skip_ids
            ]
            return SimpleNamespace(
                testsRun=tests_run,
                failures=list(failures),
                errors=list(errors),
                skipped=skipped,
                wasSuccessful=lambda: not failures and not errors,
            )

        telemetry = validate_cpu_horizon_complete_discovery(
            700, result_for(CPU_DISCOVERY_CUDA_SKIP_IDS)
        )
        self.assertEqual(telemetry["declared_tests"], 700)
        self.assertEqual(telemetry["tests_run"], 700)
        self.assertEqual(telemetry["failures"], 0)
        self.assertEqual(telemetry["errors"], 0)
        self.assertEqual(telemetry["skipped"], 6)
        self.assertEqual(telemetry["skip_ids"], list(CPU_DISCOVERY_CUDA_SKIP_IDS))

        arbitrary = "arbitrary.module.ArbitraryTests.test_unrelated_skip"
        for skip_ids in (
            CPU_DISCOVERY_CUDA_SKIP_IDS[:-1],
            CPU_DISCOVERY_CUDA_SKIP_IDS + (CPU_DISCOVERY_CUDA_SKIP_IDS[-1],),
            CPU_DISCOVERY_CUDA_SKIP_IDS[:-1] + (arbitrary,),
            CPU_DISCOVERY_CUDA_SKIP_IDS + (arbitrary,),
        ):
            with self.subTest(skip_ids=skip_ids), self.assertRaises(AssertionError):
                validate_cpu_horizon_complete_discovery(700, result_for(skip_ids))
        for discovery_result in (
            result_for(CPU_DISCOVERY_CUDA_SKIP_IDS, tests_run=699),
            result_for(CPU_DISCOVERY_CUDA_SKIP_IDS, failures=((object(), "failure"),)),
            result_for(CPU_DISCOVERY_CUDA_SKIP_IDS, errors=((object(), "error"),)),
        ):
            with self.assertRaises(AssertionError):
                validate_cpu_horizon_complete_discovery(700, discovery_result)


if __name__ == "__main__":
    unittest.main()
