"""Frozen V3 canonical-plane pilot contract tests."""

from __future__ import annotations

from dataclasses import replace
import ast
import json
from pathlib import Path
import unittest

try:
    import torch
except ImportError:
    torch = None

from prototype.flat_baseline.constrained_v3_pilot_config import (
    PILOT_EXPECTED_EXAMPLES_PROCESSED,
    PILOT_EXPECTED_OPTIMIZER_STEPS,
    PILOT_EXPECTED_TRAIN_EXAMPLES,
    PILOT_EXPECTED_VALIDATION_EXAMPLES,
    PILOT_IDENTITY,
    ConstrainedV3PilotConfig,
    ConstrainedV3PilotError,
    validate_pilot_partition_authorization,
)
from prototype.flat_baseline.constrained_v3_training_config import ConstrainedV3TrainingConfig

TORCH_REASON = "real PyTorch execution is deferred to the authoritative environment"


class ConstrainedV3PilotStaticTests(unittest.TestCase):
    def test_frozen_budget_partition_and_identity(self):
        config = ConstrainedV3PilotConfig()
        config.validate()
        self.assertIn("CANONICAL-PLANE-v3", PILOT_IDENTITY)
        self.assertEqual((config.seed, config.epochs, config.batch_size), (2026, 2, 8))
        self.assertEqual(
            (
                PILOT_EXPECTED_TRAIN_EXAMPLES,
                PILOT_EXPECTED_VALIDATION_EXAMPLES,
                PILOT_EXPECTED_OPTIMIZER_STEPS,
                PILOT_EXPECTED_EXAMPLES_PROCESSED,
            ),
            (544, 68, 136, 1088),
        )
        self.assertEqual(config.training_partition, "train")
        self.assertEqual(config.validation_partition_identity, "iid_validation")
        self.assertEqual(json.loads(config.to_json()), config.to_dict())
        self.assertFalse(config.systematic_partition_accessed)
        self.assertFalse(config.test_partition_accessed)

    def test_protected_access_and_non_boolean_facts_are_rejected(self):
        for change in (
            {"systematic_partition_accessed": True},
            {"test_partition_accessed": True},
            {"systematic_partition_accessed": 0},
            {"test_partition_accessed": None},
            {"validation_partition": "systematic"},
        ):
            with self.assertRaises(ConstrainedV3PilotError):
                validate_pilot_partition_authorization(replace(ConstrainedV3PilotConfig(), **change))

    def test_pilot_is_distinct_from_v3_tiny_training(self):
        pilot = ConstrainedV3PilotConfig()
        tiny = ConstrainedV3TrainingConfig()
        self.assertNotEqual(pilot.pilot_identity, tiny.tiny_overfit_selection_identity)
        self.assertEqual(tiny.validation_partition, "none")

    def test_python_38_and_slurm_syntax_surface(self):
        root = Path(__file__).resolve().parents[1]
        for name in ("constrained_v3_pilot.py", "constrained_v3_pilot_config.py", "run_constrained_v3_pilot.py"):
            path = root / name
            ast.parse(path.read_text(encoding="utf-8"), str(path), feature_version=(3, 8))
        slurm = root / "adroit" / "constrained_v3_pilot_cpu.slurm"
        text = slurm.read_text(encoding="utf-8")
        self.assertIn('torch.version.cuda == "11.3"', text)
        self.assertIn("focused_v3_tests_skipped", text)
        self.assertIn("constrained_v3_runs", text)


@unittest.skipIf(torch is None, TORCH_REASON)
class ConstrainedV3PilotTensorTests(unittest.TestCase):
    def test_positive_acceptance_predicates(self):
        from prototype.flat_baseline.constrained_v3_pilot import _require_pilot_acceptance
        scientific = {"configured_budget_completed": True, "autonomous_completed": True}
        result = _require_pilot_acceptance(scientific, {
            "systematic_partition_accessed": False,
            "test_partition_accessed": False,
        })
        self.assertTrue(result["systematic_partition_not_accessed"])
        self.assertTrue(result["test_partition_not_accessed"])
        for state in (
            {"systematic_partition_accessed": True, "test_partition_accessed": False},
            {"systematic_partition_accessed": False, "test_partition_accessed": True},
            {"systematic_partition_accessed": None, "test_partition_accessed": False},
            {"test_partition_accessed": False},
        ):
            with self.assertRaises(ConstrainedV3PilotError):
                _require_pilot_acceptance(scientific, state)

    def test_checkpoint_field_contract_names_v3_geometry(self):
        from prototype.flat_baseline.constrained_v3_pilot import PILOT_CHECKPOINT_FIELDS
        self.assertIn("learned_geometry_channel_indices", PILOT_CHECKPOINT_FIELDS)
        self.assertIn("canonical_plane_contract_id", PILOT_CHECKPOINT_FIELDS)
