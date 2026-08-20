"""PyTorch-independent contracts for ADR-0016 autonomous stopping."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import unittest

from prototype.graph_encoder.batching import build_paired_batch
from prototype.graph_encoder.config import (
    autonomous_stop_frozen_encoder_config,
    frozen_encoder_config,
)
from prototype.graph_encoder.decoder_contract import (
    AUTONOMOUS_STOP_CHECKPOINT_SCHEMA,
    AUTONOMOUS_STOP_NODE_GENERATION_IDENTITY,
    AUTONOMOUS_STOP_OUTPUT_POSITION_CONTRACT_VERSION,
    AUTONOMOUS_STOP_SHARED_DECODER_VERSION,
    GRID_SOFTMAX_OPERATION_MAGNITUDE_PARAMETERIZATION,
    LEGACY_NODE_GENERATION_IDENTITY,
    checkpoint_schema_for,
    output_position_contract_metadata,
    output_position_contract_version_for,
    shared_decoder_version_for,
)
from prototype.graph_encoder.errors import GraphEncoderError
from prototype.graph_encoder.stage6_structure_only_producer import (
    _grammar_valid,
    optimization_reliability,
    producer_config,
)
from prototype.graph_encoder.tests.fixtures import procedural_fixture
from prototype.model_data.vocab import NODE_TYPES
from prototype.node_grammar import legal_next_node_ids


ROOT = Path(__file__).parents[3]
STOP = AUTONOMOUS_STOP_NODE_GENERATION_IDENTITY
SOFTMAX = GRID_SOFTMAX_OPERATION_MAGNITUDE_PARAMETERIZATION


class AutonomousStopIdentityTests(unittest.TestCase):
    def test_identity_versions_are_checkpoint_incompatible(self):
        self.assertEqual(
            STOP, "GE1-PAD-TERMINATED-UNCONSTRAINED-NODES-v1"
        )
        self.assertEqual(
            shared_decoder_version_for(SOFTMAX, STOP),
            AUTONOMOUS_STOP_SHARED_DECODER_VERSION,
        )
        self.assertEqual(
            output_position_contract_version_for(SOFTMAX, STOP),
            AUTONOMOUS_STOP_OUTPUT_POSITION_CONTRACT_VERSION,
        )
        self.assertEqual(
            checkpoint_schema_for(SOFTMAX, STOP),
            AUTONOMOUS_STOP_CHECKPOINT_SCHEMA,
        )

    def test_new_configuration_is_explicit_and_historical_default_is_omitted(self):
        config = autonomous_stop_frozen_encoder_config("flat")
        payload = json.loads(config.to_json())
        self.assertEqual(payload["node_generation_identity"], STOP)
        self.assertEqual(
            payload["operation_magnitude_parameterization"], SOFTMAX
        )
        self.assertEqual(payload["checkpoint_schema"], AUTONOMOUS_STOP_CHECKPOINT_SCHEMA)
        self.assertNotIn(
            "node_generation_identity",
            json.loads(frozen_encoder_config("flat").to_json()),
        )

    def test_stop_identity_cannot_be_combined_with_an_old_magnitude_head(self):
        values = autonomous_stop_frozen_encoder_config("flat").to_dict()
        values.pop("arm_identity")
        values["operation_magnitude_parameterization"] = (
            "GE1-OPERATION-MAGNITUDE-POSITIVE-v1"
        )
        with self.assertRaises(GraphEncoderError):
            type(autonomous_stop_frozen_encoder_config("flat"))(**values).validate()

    def test_output_metadata_adds_one_non_graph_terminator_position(self):
        metadata = output_position_contract_metadata(STOP)
        self.assertEqual(
            metadata["version"],
            AUTONOMOUS_STOP_OUTPUT_POSITION_CONTRACT_VERSION,
        )
        self.assertEqual(metadata["learned_terminator"]["node_type_id"], 0)
        self.assertFalse(
            metadata["learned_terminator"]["enters_graph_pair_enumeration"]
        )


class AutonomousStopTargetTests(unittest.TestCase):
    def test_each_row_gets_exactly_one_active_trailing_pad(self):
        examples = tuple(
            procedural_fixture(name).physical for name in ("E", "R")
        )
        legacy = build_paired_batch(examples)
        stopped = build_paired_batch(examples, supervise_terminator=True)
        self.assertEqual(len(stopped.target.node_mask[0]), 6)
        for old_mask, new_mask, ids, geometry_mask in zip(
            legacy.target.node_mask,
            stopped.target.node_mask,
            stopped.target.node_type_ids,
            stopped.target.geometry_mask,
        ):
            count = sum(old_mask)
            self.assertEqual(sum(new_mask), count + 1)
            self.assertEqual(ids[count], NODE_TYPES.pad_id)
            self.assertTrue(new_mask[count])
            self.assertFalse(any(new_mask[count + 1:]))
            self.assertFalse(any(geometry_mask[count]))
        self.assertEqual(stopped.target.edge_index, legacy.target.edge_index)
        self.assertEqual(stopped.target.edge_offsets, legacy.target.edge_offsets)

    def test_transition_only_grammar_does_not_change_legacy_exact_lengths(self):
        plane = NODE_TYPES.id("reference_plane")
        legacy = legal_next_node_ids((), 4)
        transition_only = legal_next_node_ids((), None)
        self.assertEqual(legacy, (plane,))
        self.assertEqual(transition_only, (plane,))
        prefix = (
            plane,
            NODE_TYPES.id("sketch"),
            NODE_TYPES.id("profile"),
        )
        self.assertEqual(legal_next_node_ids(prefix, 4), (NODE_TYPES.id("extrude"),))
        self.assertEqual(
            set(legal_next_node_ids(prefix, None)),
            {NODE_TYPES.id("axis"), NODE_TYPES.id("extrude")},
        )


class AutonomousStopGovernanceTests(unittest.TestCase):
    def test_cross_arm_train_difference_is_diagnostic_and_nonbinding(self):
        runs = [{
            "arm": arm,
            "seed": seed,
            "epoch_losses": [1.0] * 200,
            "epoch_gradient_norms": [0.5] * 200,
            "completed_epoch": 200,
            "checkpoint_epoch": 200,
        } for arm in ("flat", "typed_graph") for seed in (2026, 2027, 2028)]
        scores = {
            (arm, seed): (0.99 if arm == "flat" else 0.20)
            for arm in ("flat", "typed_graph") for seed in (2026, 2027, 2028)
        }
        result = optimization_reliability(runs, scores, (2026, 2027, 2028))
        self.assertTrue(result["pass"])
        self.assertTrue(all(not row["binding"] for row in result["train_ceiling_comparisons"]))
        self.assertTrue(all("pass" not in row for row in result["train_ceiling_comparisons"]))
        self.assertFalse(producer_config()["train_ceiling_cross_arm_gate_applied"])

    def test_grammar_invalid_generated_sequence_is_scoreable_as_false(self):
        invalid = (NODE_TYPES.id("extrude"),) * 16
        self.assertFalse(_grammar_valid(invalid, len(invalid)))

    def test_frozen_generator_source_is_unchanged(self):
        source = ROOT / "prototype/flat_baseline/constrained_v6_autonomous.py"
        self.assertEqual(
            hashlib.sha256(source.read_bytes()).hexdigest(),
            "6391352ae11a7ee2a7dc3623e57bc14a135cdfac5b3453e379d13d696a7d5fb2",
        )
        copied = (ROOT / "prototype/graph_encoder/autonomous_stop.py").read_text()
        self.assertIn("Derived from", copied)
        self.assertIn("generation_cap_reached", copied)

    def test_shared_dispatch_has_one_decoder_call_and_no_count_on_stop(self):
        source = (ROOT / "prototype/graph_encoder/autonomous.py").read_text()
        tree = ast.parse(source)
        helper = next(
            item for item in tree.body
            if isinstance(item, ast.FunctionDef)
            and item.name == "_decode_condition"
        )
        text = ast.get_source_segment(source, helper)
        self.assertEqual(text.count("model.decoder("), 1)
        self.assertIn("uses_autonomous_stop", text)
        self.assertIn("decoder_kwargs = {}", text)

    def test_legacy_identity_literal_is_still_the_default(self):
        self.assertEqual(
            frozen_encoder_config("flat").node_generation_identity,
            LEGACY_NODE_GENERATION_IDENTITY,
        )
