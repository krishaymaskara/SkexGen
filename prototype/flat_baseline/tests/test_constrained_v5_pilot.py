"""V5 fixed-pilot identity and no-training smoke tests."""

from __future__ import annotations

from dataclasses import replace
import ast
import json
from pathlib import Path
import tempfile
import unittest

try:
    import torch
except ImportError:
    torch = None

from prototype.flat_baseline.constrained_v5_pilot_config import (
    PILOT_EXPECTED_EXAMPLES_PROCESSED,
    PILOT_EXPECTED_OPTIMIZER_STEPS,
    PILOT_EXPECTED_TRAIN_EXAMPLES,
    PILOT_EXPECTED_VALIDATION_EXAMPLES,
    PILOT_IDENTITY,
    ConstrainedV5PilotConfig,
    ConstrainedV5PilotError,
    validate_pilot_partition_authorization,
)


TORCH_REASON = "real PyTorch execution is deferred to the authoritative environment"


class V5PilotStaticTests(unittest.TestCase):
    def test_exact_protocol_and_protected_boundaries(self):
        config = ConstrainedV5PilotConfig()
        config.validate()
        self.assertIn("PREFIX-GRAMMAR-v5", PILOT_IDENTITY)
        self.assertEqual((config.seed, config.epochs, config.batch_size), (2026, 2, 8))
        self.assertEqual(
            (PILOT_EXPECTED_TRAIN_EXAMPLES, PILOT_EXPECTED_VALIDATION_EXAMPLES,
             PILOT_EXPECTED_OPTIMIZER_STEPS, PILOT_EXPECTED_EXAMPLES_PROCESSED),
            (544, 68, 136, 1088),
        )
        self.assertFalse(config.systematic_partition_accessed)
        self.assertFalse(config.test_partition_accessed)
        for changes in (
            {"systematic_partition_accessed": True},
            {"test_partition_accessed": True},
            {"validation_partition": "systematic"},
            {"epochs": 3},
            {"batch_size": 4},
        ):
            with self.assertRaises(ConstrainedV5PilotError):
                validate_pilot_partition_authorization(replace(config, **changes))

    def test_python38_and_slurm_contract(self):
        root = Path(__file__).resolve().parents[1]
        for name in (
            "constrained_v5_training.py",
            "constrained_v5_pilot.py",
            "constrained_v5_pilot_config.py",
            "run_constrained_v5_pilot.py",
        ):
            path = root / name
            ast.parse(path.read_text(), str(path), feature_version=(3, 8))
        slurm = (root / "adroit" / "constrained_v5_pilot_cpu.slurm").read_text()
        self.assertIn('torch.__version__.split("+")[0] == "1.11.0"', slurm)
        self.assertIn('torch.version.cuda == "11.3"', slurm)
        self.assertIn("focused_v5_tests_skipped", slurm)
        self.assertIn("constrained_v5_runs", slurm)


@unittest.skipIf(torch is None, TORCH_REASON)
class V5PilotTensorTests(unittest.TestCase):
    def test_acceptance_includes_positive_node_grammar_predicate(self):
        from prototype.flat_baseline.constrained_v5_pilot import _require_pilot_acceptance
        scientific = {
            "configured_budget_completed": True,
            "constrained_categorical_contract_satisfied": True,
            "constrained_node_grammar_contract_satisfied": True,
        }
        result = _require_pilot_acceptance(scientific, {
            "systematic_partition_accessed": False,
            "test_partition_accessed": False,
        })
        self.assertTrue(all(result.values()))
        scientific["constrained_node_grammar_contract_satisfied"] = False
        with self.assertRaises(ConstrainedV5PilotError):
            _require_pilot_acceptance(scientific, {
                "systematic_partition_accessed": False,
                "test_partition_accessed": False,
            })

    def test_no_training_production_shaped_v5_pilot_smoke(self):
        from prototype.controlled_data.builders import build_history
        from prototype.controlled_data.factors import PrimitiveFamily
        from prototype.controlled_data.identity import source_family_id
        from prototype.flat_baseline.constrained_v5 import ConstrainedProfileV5Model
        from prototype.flat_baseline.constrained_v5_config import ConstrainedProfileV5Config
        from prototype.flat_baseline.constrained_v5_pilot import (
            _pilot_partition,
            _require_finite_json,
            autonomous_validation,
            teacher_forced_validation,
        )
        from prototype.model_data.loader import load_partition_physical_examples
        from prototype.model_data.tests.fixtures import source, write_physical_corpus
        from prototype.model_data.vocab import NODE_TYPES
        from prototype.representation.model import GeometryEncoding

        validate_pilot_partition_authorization(ConstrainedV5PilotConfig())
        sources = (
            source("E", PrimitiveFamily.CIRCLE, extents=(1.0,)),
            source("R", PrimitiveFamily.RECTANGLE_LINES, extents=(1.0,)),
            source("EE", PrimitiveFamily.CAPSULE_LINE_ARC, extents=(1.0, 2.0)),
            source("ER", PrimitiveFamily.CIRCLE, extents=(2.0, 3.0)),
            source("RE", PrimitiveFamily.RECTANGLE_LINES, extents=(3.0, 1.0)),
            source("RR", PrimitiveFamily.CAPSULE_LINE_ARC, extents=(2.0, 1.0)),
            source("E", PrimitiveFamily.RECTANGLE_LINES, extents=(2.0,)),
            source("R", PrimitiveFamily.CIRCLE, extents=(2.0,)),
        )
        family_ids = tuple(
            source_family_id(build_history(item, GeometryEncoding.CONTINUOUS))
            for item in sources
        )
        with tempfile.TemporaryDirectory() as temporary:
            write_physical_corpus(
                temporary,
                sources,
                partitions={item: "validation" for item in family_ids},
            )
            physical = load_partition_physical_examples(
                temporary, "iid", "validation"
            )
            partition = _pilot_partition(
                physical,
                tuple(item.physical_family_id for item in physical),
                "validation",
                "iid_validation",
            )
            model_config = ConstrainedProfileV5Config()
            model = ConstrainedProfileV5Model(model_config)
            with torch.no_grad():
                model.node_type_head.weight.zero_()
                model.node_type_head.bias.zero_()
                model.node_type_head.bias[NODE_TYPES.id(None)] = 10.0
            teacher = teacher_forced_validation(
                model, partition, model_config, 8, torch.device("cpu")
            )
            autonomous = autonomous_validation(
                model, partition, model_config, 8, torch.device("cpu")
            )
        self.assertEqual(teacher["example_count"], 8)
        self.assertEqual(autonomous["example_count"], 8)
        grammar = autonomous["node_grammar_selection_metrics"]
        self.assertEqual(grammar["constrained_grammar_violation_count"], 0)
        self.assertEqual(autonomous["constrained_grammar_valid_sequence_count"], 8)
        self.assertEqual(len(autonomous["outcomes"]), 8)
        _require_finite_json({"teacher": teacher, "autonomous": autonomous})
        json.dumps(
            {"teacher": teacher, "autonomous": autonomous},
            sort_keys=True,
            allow_nan=False,
        )
