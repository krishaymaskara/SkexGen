"""Promotion and usage-policy tests for full train-kmeans retraining."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

try:
    import torch
except (ImportError, OSError):
    torch = None

from prototype.flat_baseline.training_config import TrainingConfig
from prototype.flat_baseline.vq_full_policy import (
    code_usage_trend,
    empty_collapse_state,
    final_training_decision,
    training_epoch_monitor,
)

if torch is not None:
    from prototype.flat_baseline.vq_full_retrain import (
        FullRetrainError,
        _validate_pilot,
        authoritative_epoch_budget,
    )


class FullRetrainSlurmContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.script = (
            Path(__file__).parents[1]
            / "adroit"
            / "train_kmeans_full_cpu.slurm"
        ).read_text()
        cls.implementation = (
            Path(__file__).parents[1] / "vq_full_retrain.py"
        ).read_text()

    def test_cpu_is_explicit_for_final_training_invocation(self):
        invocation = self.script.split("stage full_training", 1)[1].split(
            "stage artifact_validation", 1
        )[0]
        self.assertIn("--device cpu", invocation)
        self.assertIn('assert not torch.cuda.is_available()', self.script)
        self.assertIn(
            'configuration["training_config"]["device"] == "cpu"',
            self.script,
        )
        self.assertIn(
            'last["training_config"]["device"] == "cpu"', self.script
        )

    def test_resume_is_exact_and_restart_is_not_selected(self):
        self.assertIn(
            '"resume_strategy": "exact_epoch_boundary_resume"',
            self.implementation,
        )
        self.assertIn("restore_rng=True", self.implementation)
        self.assertIn('restored["epoch"] + 1', self.implementation)
        self.assertIn('restored["global_step"]', self.implementation)
        self.assertIn('restored["data_state"]', self.implementation)
        self.assertIn('"scheduler_present": False', self.implementation)

    def test_no_kmeans_reinitialization_occurs_on_resume(self):
        self.assertNotIn(
            "initialize_train_codebook", self.implementation
        )
        invocation = self.script.split("stage full_training", 1)[1].split(
            "stage artifact_validation", 1
        )[0]
        self.assertIn("--pilot-run", invocation)
        self.assertIn("--vq-init train-kmeans", invocation)

    def test_successful_pilot_provenance_is_required(self):
        self.assertIn('PILOT_JOB="3326278"', self.script)
        self.assertIn(
            'decision["decision"] == "PASS_FOR_FULL_RETRAIN"',
            self.script,
        )
        self.assertIn(
            'checkpoint["data_state"]["initialization_report_sha256"]',
            self.script,
        )

    def test_test_partition_remains_inaccessible(self):
        call = self.implementation.split(
            "data = load_training_data(", 1
        )[1].split(")", 1)[0]
        self.assertIn('"train"', call)
        self.assertIn('"validation"', call)
        self.assertNotIn('"test"', call)
        self.assertIn('"test_partition_evaluated": False', self.implementation)
        self.assertNotIn("--test-partition", self.script)

    def test_output_namespace_cannot_overwrite_pilot(self):
        self.assertIn(
            'OUTPUT="${RUN_ROOT}/train-kmeans-full-${REVIEWED_COMMIT}-'
            '${SLURM_JOB_ID}"',
            self.script,
        )
        self.assertIn("destination.resolve() == pilot_root", self.implementation)
        self.assertIn("full output cannot overwrite the pilot", self.implementation)


def _usage_record(mode, epoch, active, perplexity, finite=True):
    return {
        "mode": mode,
        "epoch": epoch,
        "metrics": {"total": 1.0},
        "diagnostics": {
            "active_code_count": active,
            "codebook_perplexity": perplexity,
            "finite": finite,
            "ema_state_consistent": finite,
        },
    }


class FullRetrainUsagePolicyTests(unittest.TestCase):
    def test_one_violation_warns_but_does_not_stop(self):
        state, monitor = training_epoch_monitor(
            empty_collapse_state(),
            _usage_record("train", 6, 2, 1.9),
        )
        self.assertTrue(monitor["warning"])
        self.assertFalse(monitor["stop"])
        self.assertEqual(state["consecutive_violations"], 1)

    def test_two_consecutive_low_perplexity_epochs_stop(self):
        state = empty_collapse_state()
        state, first = training_epoch_monitor(
            state, _usage_record("train", 6, 3, 1.9)
        )
        state, second = training_epoch_monitor(
            state, _usage_record("train", 7, 4, 1.8)
        )
        self.assertFalse(first["stop"])
        self.assertTrue(second["stop"])
        self.assertEqual(state["maximum_consecutive_violations"], 2)

    def test_two_consecutive_low_active_code_epochs_stop(self):
        state = empty_collapse_state()
        state, first = training_epoch_monitor(
            state, _usage_record("train", 6, 1, 3.0)
        )
        state, second = training_epoch_monitor(
            state, _usage_record("train", 7, 0, 2.5)
        )
        self.assertFalse(first["stop"])
        self.assertTrue(second["stop"])

    def test_recovered_epoch_resets_consecutive_counter(self):
        state = empty_collapse_state()
        state, unused = training_epoch_monitor(
            state, _usage_record("train", 6, 3, 1.5)
        )
        del unused
        state, recovered = training_epoch_monitor(
            state, _usage_record("train", 7, 2, 2.0)
        )
        state, later = training_epoch_monitor(
            state, _usage_record("train", 8, 1, 1.0)
        )
        self.assertFalse(recovered["warning"])
        self.assertEqual(recovered["consecutive_training_violations"], 0)
        self.assertFalse(later["stop"])
        self.assertEqual(state["consecutive_violations"], 1)

    def test_exactly_one_active_code_stops_under_general_rule(self):
        state = empty_collapse_state()
        for epoch in (6, 7):
            state, monitor = training_epoch_monitor(
                state, _usage_record("train", epoch, 1, 1.0)
            )
        self.assertTrue(monitor["stop"])
        self.assertEqual(monitor["consecutive_training_violations"], 2)

    def test_epoch_50_with_failed_validation_gate_does_not_pass(self):
        trend = code_usage_trend(self._trend_records())
        failed = _usage_record("validation", 50, 2, 1.9)
        result = final_training_decision(
            50, 50, None, True, True, failed, False, trend
        )
        self.assertEqual(
            result["decision"], "FAIL_ASSIGNMENT_COLLAPSE"
        )

    def test_epoch_50_with_valid_gates_passes_for_evaluation(self):
        trend = code_usage_trend(self._trend_records())
        selected = _usage_record("validation", 50, 3, 2.5)
        result = final_training_decision(
            50, 50, None, True, True, selected, False, trend
        )
        self.assertEqual(result["decision"], "PASS_FOR_EVALUATION")
        self.assertFalse(result["test_partition_evaluated"])

    def test_nonfinite_selected_usage_fails_numerically(self):
        trend = code_usage_trend(self._trend_records())
        selected = _usage_record(
            "validation", 50, 3, float("nan"), finite=False
        )
        result = final_training_decision(
            50, 50, None, True, True, selected, False, trend
        )
        self.assertEqual(result["decision"], "FAIL_NUMERICAL")

    def test_trend_summary_is_deterministic(self):
        records = self._trend_records()
        first = code_usage_trend(records)
        second = code_usage_trend(tuple(reversed(records)))
        self.assertEqual(first, second)
        self.assertEqual(first["trend"], "increasing")
        self.assertEqual(len(first["final_five_train_epochs"]), 5)
        self.assertEqual(len(first["final_five_validation_epochs"]), 5)
        self.assertEqual(
            first["pilot_epoch_5"]["train"]["active_code_count"], 5
        )

    @staticmethod
    def _trend_records():
        records = [
            _usage_record("train", 5, 5, 3.3),
            _usage_record("validation", 5, 4, 3.6),
        ]
        for epoch, active, perplexity in (
            (46, 2, 2.1),
            (47, 2, 2.2),
            (48, 3, 2.2),
            (49, 3, 2.4),
            (50, 4, 2.6),
        ):
            records.append(
                _usage_record("train", epoch, active, perplexity)
            )
            records.append(
                _usage_record(
                    "validation", epoch, active, perplexity + 0.1
                )
            )
        return records


@unittest.skipUnless(torch is not None, "real PyTorch is required")
class FullRetrainPromotionTests(unittest.TestCase):
    def test_authoritative_full_epoch_budget_is_inherited_and_reconciled(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            training = TrainingConfig(epochs=50)
            torch.save(
                {"training_config": training.to_dict()},
                str(root / "best.pt"),
            )
            metadata = {
                "event": "run_metadata",
                "training_config": training.to_dict(),
            }
            (root / "metrics.jsonl").write_text(
                json.dumps(metadata) + "\n"
            )
            self.assertEqual(authoritative_epoch_budget(root, torch), 50)

            metadata["training_config"]["epochs"] = 49
            (root / "metrics.jsonl").write_text(
                json.dumps(metadata) + "\n"
            )
            with self.assertRaises(FullRetrainError) as captured:
                authoritative_epoch_budget(root, torch)
            self.assertEqual(
                captured.exception.code, "baseline_provenance"
            )

    def test_successful_pilot_artifacts_are_required(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FullRetrainError) as captured:
                _validate_pilot(directory, torch)
            self.assertEqual(captured.exception.code, "pilot_provenance")


if __name__ == "__main__":
    unittest.main()
