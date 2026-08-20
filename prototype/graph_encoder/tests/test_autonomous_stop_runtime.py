"""Corpus-free PyTorch runtime tests for ADR-0016 autonomous stopping."""

from __future__ import annotations

import tempfile
import unittest

try:
    import torch
except ImportError:  # pragma: no cover - authoritative runtime supplies torch
    torch = None

from prototype.graph_encoder.batching import build_paired_batch
from prototype.graph_encoder.config import autonomous_stop_frozen_encoder_config
from prototype.graph_encoder.decoder_contract import (
    AUTONOMOUS_STOP_CHECKPOINT_SCHEMA,
    AUTONOMOUS_STOP_NODE_GENERATION_IDENTITY,
    GRID_SOFTMAX_OPERATION_MAGNITUDE_PARAMETERIZATION,
    LEGACY_NODE_GENERATION_IDENTITY,
)
from prototype.graph_encoder.tests.fixtures import procedural_fixture
from prototype.model_data.vocab import NODE_TYPES


REASON = "autonomous-stop runtime tests require the PyTorch runtime"
STOP = AUTONOMOUS_STOP_NODE_GENERATION_IDENTITY
SOFTMAX = GRID_SOFTMAX_OPERATION_MAGNITUDE_PARAMETERIZATION


@unittest.skipIf(torch is None, REASON)
class AutonomousStopRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def _decoder(self, identity=STOP):
        from prototype.graph_encoder.shared_decoder import SharedGE1Decoder

        return SharedGE1Decoder(
            operation_magnitude_parameterization=SOFTMAX,
            node_generation_identity=identity,
        )

    def _memory(self, decoder):
        return torch.zeros(
            1, decoder.config.latent_tokens, decoder.config.model_dim
        )

    def test_stop_selector_is_plain_eight_class_argmax(self):
        from prototype.node_grammar_torch import select_prefix_conditioned_node_types

        logits = torch.full((1, len(NODE_TYPES.tokens)), -10.0)
        logits[0, NODE_TYPES.pad_id] = 10.0
        result = select_prefix_conditioned_node_types(
            logits,
            ((),),
            None,
            current_positions=torch.tensor((0,), dtype=torch.long),
            node_generation_identity=STOP,
        )
        self.assertEqual(int(result.grammar_constrained_node_type_ids[0]), 0)
        self.assertFalse(bool(result.node_type_correction_mask[0]))
        self.assertTrue(bool(result.legal_node_type_mask.all()))

    def test_legacy_selector_still_masks_and_corrects(self):
        from prototype.node_grammar_torch import select_prefix_conditioned_node_types

        logits = torch.full((1, len(NODE_TYPES.tokens)), -10.0)
        logits[0, NODE_TYPES.id("extrude")] = 10.0
        result = select_prefix_conditioned_node_types(
            logits,
            ((),),
            torch.tensor((4,), dtype=torch.long),
            current_positions=torch.tensor((0,), dtype=torch.long),
            node_generation_identity=LEGACY_NODE_GENERATION_IDENTITY,
        )
        self.assertEqual(
            int(result.grammar_constrained_node_type_ids[0]),
            NODE_TYPES.id("reference_plane"),
        )
        self.assertTrue(bool(result.node_type_correction_mask[0]))

    def test_immediate_pad_is_a_learned_zero_node_stop_not_an_exception(self):
        decoder = self._decoder()
        with torch.no_grad():
            decoder.node_type_head.weight.zero_()
            decoder.node_type_head.bias.fill_(-10.0)
            decoder.node_type_head.bias[NODE_TYPES.pad_id] = 10.0
        output = decoder(self._memory(decoder))
        raw = output.raw_prediction[0]
        self.assertEqual(raw.node_count, 0)
        self.assertEqual(raw.terminator_position, 0)
        self.assertFalse(raw.generation_cap_reached)
        self.assertTrue(output.converted_prediction[0].raised_failure)

    def test_no_pad_reaches_cap_and_preserves_invalid_sequence_for_scoring(self):
        from prototype.graph_encoder.stage6_structure_only_producer import _grammar_valid

        decoder = self._decoder()
        with torch.no_grad():
            decoder.node_type_head.weight.zero_()
            decoder.node_type_head.bias.fill_(-10.0)
            decoder.node_type_head.bias[NODE_TYPES.id("extrude")] = 10.0
        output = decoder(self._memory(decoder))
        raw = output.raw_prediction[0]
        self.assertEqual(raw.node_count, decoder.config.max_nodes)
        self.assertTrue(raw.generation_cap_reached)
        self.assertTrue(output.converted_prediction[0].raised_failure)
        self.assertFalse(
            _grammar_valid(raw.grammar_constrained_node_type_ids, raw.node_count)
        )

    def test_stop_forward_forbids_counts_and_legacy_requires_them(self):
        stop = self._decoder()
        with self.assertRaises(ValueError):
            stop(
                self._memory(stop),
                node_counts=torch.tensor((4,), dtype=torch.long),
                node_count_source="forbidden",
            )
        legacy = self._decoder(LEGACY_NODE_GENERATION_IDENTITY)
        with self.assertRaises(ValueError):
            legacy(self._memory(legacy))

    def test_teacher_forced_loss_supervises_one_stop_and_excludes_its_edges(self):
        from prototype.constrained_profile_decoder import profile_targets_for_loss
        from prototype.graph_encoder.losses import common_ge1_loss

        paired = build_paired_batch(
            (procedural_fixture("E").physical,), supervise_terminator=True
        )
        target = paired.target.to_torch(torch)
        flat = paired.flat_input.to_torch(torch)
        profiles = profile_targets_for_loss(paired.target, flat["geometry"])
        decoder = self._decoder()
        output = decoder.teacher_forced(
            self._memory(decoder), target, profiles
        )
        output.graph_edge_logits.retain_grad()
        grid_logits = decoder.grid_magnitude_logits(output.decoded_states)
        config = autonomous_stop_frozen_encoder_config("flat")
        loss = common_ge1_loss(
            output,
            target,
            profiles,
            config,
            grid_magnitude_logits=grid_logits,
        )
        self.assertTrue(bool(torch.isfinite(loss.total).item()))
        loss.total.backward()
        terminator = int(target["node_mask"][0].long().sum().item()) - 1
        self.assertEqual(
            float(output.graph_edge_logits.grad[0, terminator].abs().sum()), 0.0
        )
        self.assertEqual(
            float(output.graph_edge_logits.grad[0, :, terminator].abs().sum()), 0.0
        )

    def test_loss_rejects_more_than_one_active_pad(self):
        from prototype.constrained_profile_decoder import profile_targets_for_loss
        from prototype.graph_encoder.errors import GraphEncoderError
        from prototype.graph_encoder.losses import common_ge1_loss

        paired = build_paired_batch(
            tuple(procedural_fixture(name).physical for name in ("E", "R")),
            supervise_terminator=True,
        )
        target = paired.target.to_torch(torch)
        flat = paired.flat_input.to_torch(torch)
        profiles = profile_targets_for_loss(paired.target, flat["geometry"])
        decoder = self._decoder()
        output = decoder.teacher_forced(self._memory(decoder).expand(2, -1, -1).contiguous(), target, profiles)
        grid_logits = decoder.grid_magnitude_logits(output.decoded_states)
        target["node_mask"][0, -1] = True
        with self.assertRaises(GraphEncoderError):
            common_ge1_loss(
                output,
                target,
                profiles,
                autonomous_stop_frozen_encoder_config("flat"),
                grid_magnitude_logits=grid_logits,
            )

    def test_v4_checkpoint_round_trip_and_cross_identity_rejection(self):
        from prototype.graph_encoder.checkpoint import (
            GE1CheckpointError,
            ge1_checkpoint_payload,
            load_ge1_checkpoint,
            save_ge1_checkpoint,
        )
        from prototype.graph_encoder.model import build_ge1_model

        model = build_ge1_model(autonomous_stop_frozen_encoder_config("flat"))
        payload = ge1_checkpoint_payload(model, code_revision="a" * 40)
        self.assertEqual(payload["checkpoint_schema"], AUTONOMOUS_STOP_CHECKPOINT_SCHEMA)
        self.assertEqual(payload["node_generation_identity"], STOP)
        with tempfile.TemporaryDirectory() as temporary:
            path = temporary + "/model.pt"
            save_ge1_checkpoint(path, model, code_revision="a" * 40)
            loaded, unused = load_ge1_checkpoint(
                path,
                expected_code_revision="a" * 40,
                expected_operation_magnitude_parameterization=SOFTMAX,
                expected_node_generation_identity=STOP,
            )
            self.assertEqual(loaded.config, model.config)
            with self.assertRaises(GE1CheckpointError):
                load_ge1_checkpoint(
                    path,
                    expected_operation_magnitude_parameterization=SOFTMAX,
                    expected_node_generation_identity=(
                        LEGACY_NODE_GENERATION_IDENTITY
                    ),
                )
