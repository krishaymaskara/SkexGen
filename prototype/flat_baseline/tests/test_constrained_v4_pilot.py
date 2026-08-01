"""V4 fixed-pilot identity and acceptance tests."""

from __future__ import annotations

from dataclasses import replace
import ast
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

try:
    import torch
except ImportError:
    torch = None

from prototype.flat_baseline.constrained_v4_pilot_config import (
    PILOT_EXPECTED_EXAMPLES_PROCESSED,
    PILOT_EXPECTED_OPTIMIZER_STEPS,
    PILOT_EXPECTED_TRAIN_EXAMPLES,
    PILOT_EXPECTED_VALIDATION_EXAMPLES,
    PILOT_IDENTITY,
    ConstrainedV4PilotConfig,
    ConstrainedV4PilotError,
    validate_pilot_partition_authorization,
)

TORCH_REASON = "real PyTorch execution is deferred to the authoritative environment"


class V4PilotStaticTests(unittest.TestCase):
    def test_exact_protocol_identity_and_protected_boundaries(self):
        config = ConstrainedV4PilotConfig()
        config.validate()
        self.assertIn("NODE-CONDITIONED-CATEGORIES-v4", PILOT_IDENTITY)
        self.assertEqual((config.seed, config.epochs, config.batch_size), (2026, 2, 8))
        self.assertEqual(
            (PILOT_EXPECTED_TRAIN_EXAMPLES, PILOT_EXPECTED_VALIDATION_EXAMPLES,
             PILOT_EXPECTED_OPTIMIZER_STEPS, PILOT_EXPECTED_EXAMPLES_PROCESSED),
            (544, 68, 136, 1088),
        )
        self.assertFalse(config.systematic_partition_accessed)
        self.assertFalse(config.test_partition_accessed)

    def test_partition_changes_and_non_boolean_facts_fail(self):
        for changes in (
            {"systematic_partition_accessed": True},
            {"test_partition_accessed": True},
            {"systematic_partition_accessed": 0},
            {"validation_partition": "systematic"},
        ):
            with self.assertRaises(ConstrainedV4PilotError):
                validate_pilot_partition_authorization(
                    replace(ConstrainedV4PilotConfig(), **changes)
                )

    def test_python38_and_slurm_contract(self):
        root = Path(__file__).resolve().parents[1]
        for name in ("constrained_v4_pilot.py", "constrained_v4_pilot_config.py", "run_constrained_v4_pilot.py"):
            path = root / name
            ast.parse(path.read_text(), str(path), feature_version=(3, 8))
        slurm = (root / "adroit" / "constrained_v4_pilot_cpu.slurm").read_text()
        self.assertIn('torch.version.cuda == "11.3"', slurm)
        self.assertIn("focused_v4_tests_skipped", slurm)
        self.assertIn("constrained_v4_runs", slurm)


@unittest.skipIf(torch is None, TORCH_REASON)
class V4PilotTensorTests(unittest.TestCase):
    def test_positive_acceptance_requires_contract_and_false_access(self):
        from prototype.flat_baseline.constrained_v4_pilot import _require_pilot_acceptance
        scientific = {
            "configured_budget_completed": True,
            "constrained_categorical_contract_satisfied": True,
        }
        result = _require_pilot_acceptance(scientific, {
            "systematic_partition_accessed": False,
            "test_partition_accessed": False,
        })
        self.assertTrue(all(result.values()))
        for access in ({}, {"systematic_partition_accessed": True, "test_partition_accessed": False}):
            with self.assertRaises(ConstrainedV4PilotError):
                _require_pilot_acceptance(scientific, access)

    def test_checkpoint_metadata_requires_both_contracts(self):
        from prototype.flat_baseline.constrained_v4_pilot import PILOT_CHECKPOINT_FIELDS
        self.assertIn("canonical_plane_contract_id", PILOT_CHECKPOINT_FIELDS)
        self.assertIn("base_geometry_contract_id", PILOT_CHECKPOINT_FIELDS)
        self.assertIn("categorical_selection_contract_id", PILOT_CHECKPOINT_FIELDS)
        self.assertIn("categorical_selection_contract", PILOT_CHECKPOINT_FIELDS)

    def test_job_3334551_no_training_pilot_selector_smoke(self):
        from prototype.controlled_data.builders import build_history
        from prototype.controlled_data.factors import PrimitiveFamily
        from prototype.flat_baseline.constrained_v4 import ConstrainedProfileV4Model
        from prototype.flat_baseline.constrained_v4_config import ConstrainedProfileV4Config
        from prototype.flat_baseline.constrained_v4_pilot import teacher_forced_validation
        from prototype.model_data.adapters import adapt_flat_mixed
        from prototype.model_data.canonical import canonical_nodes_and_edges, reconstruction_target
        from prototype.model_data.tests.fixtures import source
        from prototype.representation.model import GeometryEncoding
        history = build_history(
            source("E", PrimitiveFamily.CIRCLE, extents=(1.0,)),
            GeometryEncoding.CONTINUOUS,
        )
        nodes, edges = canonical_nodes_and_edges(history)
        target = reconstruction_target(nodes, edges, history.structure.operation_sequence)
        example = adapt_flat_mixed(SimpleNamespace(
            physical_family_id="job-3334551-smoke", nodes=nodes, target=target
        ))
        config = ConstrainedProfileV4Config()
        model = ConstrainedProfileV4Model(config)
        with torch.no_grad():
            model.node_type_head.weight.zero_()
            model.node_type_head.bias.fill_(-10.0)
            model.node_type_head.bias[1] = 10.0
        summary = teacher_forced_validation(
            model,
            SimpleNamespace(flat_examples=(example,)),
            config,
            1,
            torch.device("cpu"),
        )
        self.assertEqual(summary["example_count"], 1)
        self.assertGreaterEqual(summary["examples_with_categorical_correction"], 1)

    def test_job_3336430_no_training_autonomous_pilot_smoke(self):
        from prototype.controlled_data.builders import build_history
        from prototype.controlled_data.factors import PrimitiveFamily
        from prototype.controlled_data.identity import source_family_id
        from prototype.flat_baseline.constrained_v4 import ConstrainedProfileV4Model
        from prototype.flat_baseline.constrained_v4_autonomous import greedy_decode_v4
        from prototype.flat_baseline.constrained_v4_config import ConstrainedProfileV4Config
        from prototype.flat_baseline.constrained_v4_pilot import (
            _pilot_partition,
            _require_finite_json,
            autonomous_validation,
            teacher_forced_validation,
        )
        from prototype.model_data.loader import load_partition_physical_examples
        from prototype.model_data.tests.fixtures import source, write_physical_corpus
        from prototype.model_data.vocab import NODE_TYPES
        from prototype.representation.model import GeometryEncoding

        pilot_config = ConstrainedV4PilotConfig()
        validate_pilot_partition_authorization(pilot_config)
        sources = (
            source("E", PrimitiveFamily.CIRCLE, extents=(1.0,)),
            source("R", PrimitiveFamily.RECTANGLE_LINES, extents=(1.0,)),
            source("ER", PrimitiveFamily.CAPSULE_LINE_ARC, extents=(1.0, 2.0)),
            source("EE", PrimitiveFamily.CIRCLE, extents=(2.0, 3.0)),
            source("R", PrimitiveFamily.RECTANGLE_LINES, extents=(3.0,)),
            source("E", PrimitiveFamily.CAPSULE_LINE_ARC, extents=(2.0,)),
            source("R", PrimitiveFamily.CIRCLE, extents=(2.0,)),
            source("E", PrimitiveFamily.RECTANGLE_LINES, extents=(1.0,)),
        )
        family_ids = tuple(
            source_family_id(build_history(item, GeometryEncoding.CONTINUOUS))
            for item in sources
        )
        with tempfile.TemporaryDirectory() as temporary:
            write_physical_corpus(
                temporary,
                sources,
                partitions={family_id: "validation" for family_id in family_ids},
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

            model_config = ConstrainedProfileV4Config()
            model = ConstrainedProfileV4Model(model_config)
            with torch.no_grad():
                model.node_type_head.weight.zero_()
                model.node_type_head.bias.fill_(-10.0)
                model.node_type_head.bias[NODE_TYPES.id(None)] = 10.0
            self.assertNotIn("target", inspect.signature(greedy_decode_v4).parameters)
            teacher = teacher_forced_validation(
                model, partition, model_config, 8, torch.device("cpu")
            )
            autonomous = autonomous_validation(
                model, partition, model_config, 8, torch.device("cpu")
            )

        self.assertEqual(teacher["example_count"], 8)
        self.assertEqual(autonomous["example_count"], 8)
        self.assertEqual(len(autonomous["outcomes"]), 8)
        for outcome in autonomous["outcomes"]:
            self.assertEqual(
                outcome["requested_node_count"], outcome["generated_node_count"]
            )
            self.assertEqual(
                outcome["node_count_source"], "authorized_validation_length"
            )
        self.assertIn("categorical_selection_metrics", autonomous)
        self.assertIn("raw_failure_reason_histogram", autonomous)
        _require_finite_json({"teacher": teacher, "autonomous": autonomous})
        json.dumps(
            {"teacher": teacher, "autonomous": autonomous},
            sort_keys=True,
            allow_nan=False,
        )
