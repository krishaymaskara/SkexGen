"""Decision, instrumentation, provenance, and resume tests for the pilot."""

from __future__ import annotations

from dataclasses import replace
import copy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

try:
    import torch
except (ImportError, OSError):
    torch = None

from prototype.flat_baseline.training_config import TrainingConfig

if torch is not None:
    from prototype.flat_baseline.config import FlatBaselineConfig
    from prototype.flat_baseline.training_config import LOSS_METRICS
    from prototype.flat_baseline.vq_initialization import (
        initialization_report_sha256,
    )
    from prototype.flat_baseline.vq_pilot import (
        PilotError,
        PilotInstrumentation,
        _decision_for_exception,
        pilot_decision,
        run_pilot,
    )


def _record(mode, epoch, active, perplexity, distinct=3, finite=True):
    return {
        "mode": mode,
        "epoch": epoch,
        "diagnostics": {
            "active_code_count": active,
            "codebook_perplexity": perplexity,
            "distinct_prequant_vector_count": distinct,
            "finite": finite,
            "ema_state_consistent": finite,
        },
    }


@unittest.skipUnless(torch is not None, "real PyTorch is required")
class PilotDecisionTests(unittest.TestCase):
    def test_epoch_two_collapse_gate_fails(self):
        result = pilot_decision((
            _record("train", 1, 4, 3.0),
            _record("validation", 1, 2, 2.0),
            _record("train", 2, 1, 1.0),
        ))
        self.assertEqual(result["decision"], "FAIL_ASSIGNMENT_COLLAPSE")
        self.assertEqual(result["reason"], "epoch_2_assignment_gate")

    def test_epoch_two_gates_are_independent_strict_or_conditions(self):
        cases = (
            _record("train", 2, 1, 3.0),
            _record("train", 2, 3, 1.999),
        )
        for record in cases:
            with self.subTest(record=record):
                result = pilot_decision((record,))
                self.assertEqual(
                    result["decision"], "FAIL_ASSIGNMENT_COLLAPSE"
                )

    def test_qualifying_synthetic_trajectory_passes(self):
        records = [
            _record("train", 2, 3, 2.5),
            _record("train", 5, 4, 3.0),
            _record("validation", 5, 2, 2.1),
        ]
        result = pilot_decision(records)
        self.assertEqual(result["decision"], "PASS_FOR_FULL_RETRAIN")

    def test_nonfinite_trajectory_fails_numerically(self):
        result = pilot_decision((
            _record("train", 2, 3, 2.5, finite=False),
        ))
        self.assertEqual(result["decision"], "FAIL_NUMERICAL")

    def test_provenance_exception_maps_to_provenance_failure(self):
        result = _decision_for_exception(
            PilotError("source_provenance", "dirty")
        )
        self.assertEqual(result["decision"], "FAIL_PROVENANCE")

    def test_initialization_mode_and_seed_are_resume_provenance(self):
        treatment = replace(
            TrainingConfig(), seed=2026, vq_init="train-kmeans"
        )
        changed_mode = replace(treatment, vq_init="normal")
        changed_seed = replace(treatment, seed=7)
        self.assertNotEqual(
            treatment.resume_signature(), changed_mode.resume_signature()
        )
        self.assertNotEqual(
            treatment.resume_signature(), changed_seed.resume_signature()
        )
        payload = treatment.to_dict()
        self.assertEqual(payload["vq_init"], "train-kmeans")
        self.assertEqual(payload["seed"], 2026)


@unittest.skipUnless(torch is not None, "real PyTorch is required")
class PilotInstrumentationTests(unittest.TestCase):
    def test_early_step_logging_does_not_mutate_training_state(self):
        model = SimpleNamespace(
            to_codebook=torch.nn.Linear(4, 4),
            vq=SimpleNamespace(
                embedding=torch.arange(16, dtype=torch.float32).reshape(4, 4),
                ema_cluster_size=torch.ones(4),
                ema_weight=torch.arange(
                    16, dtype=torch.float32
                ).reshape(4, 4),
                embedding_dim=4,
                epsilon=1e-5,
            ),
        )
        before = {
            name: value.detach().clone()
            for name, value in model.to_codebook.state_dict().items()
        }
        records = []
        logger = SimpleNamespace(write=records.append)
        optimizer = torch.optim.AdamW(model.to_codebook.parameters(), lr=1e-3)
        optimizer_before = copy.deepcopy(optimizer.state_dict())
        for parameter in model.to_codebook.parameters():
            parameter.grad = torch.ones_like(parameter)
        gradients_before = {
            name: parameter.grad.clone()
            for name, parameter in model.to_codebook.named_parameters()
        }
        instrumentation = PilotInstrumentation(logger, torch)
        instrumentation.begin(model, "train", 1)
        projected = model.to_codebook(torch.ones(2, 4)).reshape(1, 2, 4)
        output = SimpleNamespace(
            assignment_counts=torch.tensor([1, 1, 0, 0])
        )
        per_example = {
            name: torch.tensor([1.0])
            for name in (
                "total",
                "node_type",
                "categorical_attributes",
                "geometry",
                "edge_presence",
                "edge_type",
                "operation_pointer",
                "vq_commitment",
            )
        }
        losses = SimpleNamespace(per_example=per_example)
        del projected
        torch.manual_seed(789)
        rng_before = torch.get_rng_state().clone()
        instrumentation.observe_batch(model, output, losses, 0)
        instrumentation.finish(model)
        self.assertEqual(len(records), 1)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng_before))
        self.assertEqual(optimizer.state_dict(), optimizer_before)
        for name, parameter in model.to_codebook.named_parameters():
            self.assertTrue(torch.equal(
                parameter.grad, gradients_before[name]
            ))
        for name, value in model.to_codebook.state_dict().items():
            self.assertTrue(torch.equal(value, before[name]))


@unittest.skipUnless(torch is not None, "real PyTorch is required")
class PilotPartitionExecutionTests(unittest.TestCase):
    def test_run_pilot_never_passes_test_examples_to_model_paths(self):
        train_example = object()
        validation_example = object()
        test_example = object()
        data = SimpleNamespace(
            train_examples=(train_example,),
            validation_examples=(validation_example,),
            train_family_ids=("train-family",),
            validation_family_ids=("validation-family",),
        )
        physical = (
            SimpleNamespace(
                partition="train", physical_family_id="train-family"
            ),
            SimpleNamespace(
                partition="validation",
                physical_family_id="validation-family",
            ),
            SimpleNamespace(
                partition="test", physical_family_id="test-family"
            ),
        )
        model_config = FlatBaselineConfig()
        training_config = TrainingConfig(
            seed=2026, epochs=1, device="cpu"
        )
        checkpoint = {
            "model_config": model_config.to_dict(),
            "training_config": training_config.to_dict(),
            "data_state": {
                "train_family_ids": ["train-family"],
                "validation_family_ids": ["validation-family"],
            },
        }
        initialization = {
            "mode": "train-kmeans",
            "method": "seeded-kmeans++-fixed-lloyd",
            "seed": 2026,
            "pseudo_count_policy": (
                "cluster-proportions-total-codebook-size"
            ),
        }
        initialization["report_sha256"] = initialization_report_sha256(
            initialization
        )

        class DummyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.parameter = torch.nn.Parameter(torch.zeros(()))

        def summary():
            return {
                "number_of_examples": 1,
                "metrics": {name: 1.0 for name in LOSS_METRICS},
                "codebook": {
                    "active_code_count": 3,
                    "codebook_perplexity": 2.5,
                    "codebook_utilization": 0.1,
                },
                "diagnostics": _record(
                    "train", 5, 3, 2.5
                )["diagnostics"],
            }

        train_seen = []
        validation_seen = []

        def train(
            model,
            optimizer,
            examples,
            model_config,
            training_config,
            device,
            epoch,
            global_step,
            torch_module,
            instrumentation,
        ):
            del (
                model,
                optimizer,
                model_config,
                training_config,
                device,
                epoch,
                torch_module,
                instrumentation,
            )
            train_seen.extend(examples)
            return summary(), global_step + 1

        def evaluate(
            model,
            examples,
            model_config,
            training_config,
            device,
            torch_module,
            instrumentation,
            epoch,
        ):
            del (
                model,
                model_config,
                training_config,
                device,
                torch_module,
                instrumentation,
                epoch,
            )
            validation_seen.extend(examples)
            return summary()

        with tempfile.TemporaryDirectory() as temporary:
            baseline = Path(temporary) / "baseline.pt"
            baseline.write_bytes(b"baseline")
            output = Path(temporary) / "pilot"
            with mock.patch(
                "prototype.flat_baseline.vq_pilot.source_state",
                return_value={
                    "git_commit": "reviewed",
                    "git_dirty": False,
                    "git_status_porcelain": [],
                    "source_tree_sha256": "c" * 64,
                },
            ), mock.patch(
                "prototype.flat_baseline.vq_pilot.load_physical_examples",
                return_value=physical,
            ), mock.patch(
                "prototype.flat_baseline.vq_pilot.load_training_data",
                return_value=data,
            ), mock.patch.object(
                torch, "load", return_value=checkpoint
            ), mock.patch(
                "prototype.flat_baseline.vq_pilot.FlatMixedVQModel",
                return_value=DummyModel(),
            ), mock.patch(
                "prototype.flat_baseline.vq_pilot.initialize_train_codebook",
                return_value=initialization,
            ) as initialize, mock.patch(
                "prototype.flat_baseline.vq_pilot.train_epoch",
                side_effect=train,
            ), mock.patch(
                "prototype.flat_baseline.vq_pilot.evaluate_epoch",
                side_effect=evaluate,
            ):
                result = run_pilot(
                    temporary,
                    baseline,
                    output,
                    "reviewed",
                    {"train": 1, "validation": 1, "test": 1},
                    torch,
                )
        self.assertEqual(result["decision"], "PASS_FOR_FULL_RETRAIN")
        self.assertEqual(initialize.call_args.args[1], data)
        self.assertEqual(train_seen, [train_example] * 5)
        self.assertEqual(validation_seen, [validation_example] * 5)
        self.assertNotIn(test_example, train_seen + validation_seen)


if __name__ == "__main__":
    unittest.main()
