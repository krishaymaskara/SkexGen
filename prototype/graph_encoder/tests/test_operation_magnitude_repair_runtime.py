"""Real-PyTorch tests for positive operation magnitudes and strict identity."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from prototype.graph_encoder.batching import build_paired_batch
from prototype.graph_encoder.tests.fixtures import procedural_fixture


try:
    import torch
except ImportError:
    torch = None


TORCH_REASON = "operation-magnitude repair tests require real PyTorch"


@unittest.skipUnless(torch is not None, TORCH_REASON)
class OperationMagnitudeRuntimeTests(unittest.TestCase):
    def _decoders(self):
        from prototype.graph_encoder.decoder_contract import (
            LEGACY_OPERATION_MAGNITUDE_PARAMETERIZATION,
            POSITIVE_OPERATION_MAGNITUDE_PARAMETERIZATION,
        )
        from prototype.graph_encoder.shared_decoder import SharedGE1Decoder

        torch.manual_seed(9101)
        legacy = SharedGE1Decoder(
            operation_magnitude_parameterization=(
                LEGACY_OPERATION_MAGNITUDE_PARAMETERIZATION
            )
        ).eval()
        repaired = SharedGE1Decoder(
            operation_magnitude_parameterization=(
                POSITIVE_OPERATION_MAGNITUDE_PARAMETERIZATION
            )
        ).eval()
        repaired.load_state_dict(legacy.state_dict(), strict=True)
        return legacy, repaired

    def _teacher_fixture(self):
        from prototype.constrained_profile_decoder import profile_targets_for_loss

        paired = build_paired_batch(tuple(
            procedural_fixture(name).physical for name in ("E", "R")
        ))
        flat = paired.flat_input.to_torch(torch)
        target = paired.target.to_torch(torch)
        profiles = profile_targets_for_loss(paired.target, flat["geometry"])
        memory = torch.randn(2, 2, 32)
        return target, profiles, memory

    def test_legacy_is_exact_tanh_and_raw_values_remain_available(self):
        legacy, unused = self._decoders()
        raw = torch.tensor((
            (-3.0, -2.0, -1.0, 0.0, 1.0, 2.0),
            (3.0, 2.0, 1.0, 0.0, -1.0, -2.0),
        ))
        observed = legacy.parameterize_remaining_geometry(raw)
        self.assertTrue(torch.equal(observed, torch.tanh(raw)))
        states = torch.randn(2, 3, legacy.config.model_dim)
        self.assertTrue(torch.equal(
            legacy.raw_remaining_geometry(states),
            legacy.remaining_geometry_head(states),
        ))

    def test_repaired_extremes_are_finite_strictly_positive_and_bounded(self):
        unused, repaired = self._decoders()
        maximum = torch.finfo(torch.float32).max
        values = torch.tensor((-maximum, -10.0, 0.0, 10.0, maximum))
        raw = values.view(-1, 1).expand(-1, 6).contiguous()
        observed = repaired.parameterize_remaining_geometry(raw)
        operation = observed[:, 4:6]
        self.assertTrue(torch.isfinite(operation).all())
        self.assertTrue((operation > 0.0).all())
        self.assertTrue((operation <= 1.0).all())
        self.assertEqual(
            operation[0, 0].item(), torch.finfo(torch.float32).tiny
        )
        self.assertEqual(operation[-1, 0].item(), 1.0)

    def test_only_extrude_and_revolve_channels_change(self):
        legacy, repaired = self._decoders()
        raw = torch.tensor(((-0.8, -0.4, 0.0, 0.4, -0.8, 0.8),))
        historical = legacy.parameterize_remaining_geometry(raw)
        positive = repaired.parameterize_remaining_geometry(raw)
        self.assertTrue(torch.equal(historical[..., :4], positive[..., :4]))
        self.assertTrue(torch.equal(historical[..., :4], torch.tanh(raw[..., :4])))
        self.assertGreater(positive[..., 4].item(), 0.0)
        self.assertGreater(positive[..., 5].item(), 0.0)
        self.assertNotEqual(historical[..., 4].item(), positive[..., 4].item())

    def test_masks_direction_boolean_and_nonoperation_outputs_are_unchanged(self):
        legacy, repaired = self._decoders()
        target, profiles, memory = self._teacher_fixture()
        with torch.no_grad():
            historical = legacy.teacher_forced(memory, target, profiles)
            positive = repaired.teacher_forced(memory, target, profiles)
        self.assertTrue(torch.equal(
            historical.remaining_geometry[..., :4],
            positive.remaining_geometry[..., :4],
        ))
        self.assertTrue(torch.equal(
            historical.remaining_geometry_mask,
            positive.remaining_geometry_mask,
        ))
        for historical_logits, positive_logits in zip(
            historical.categorical_logits, positive.categorical_logits
        ):
            self.assertTrue(torch.equal(historical_logits, positive_logits))
        self.assertTrue(torch.equal(
            historical.training_geometry_mask,
            positive.training_geometry_mask,
        ))

    def test_five_failure_shaped_scalars_pass_the_analytic_parameter_contract(self):
        from prototype.flat_baseline.tests.test_constrained_v6 import (
            V6GrammarTensorTests,
        )
        from prototype.graph_baseline.conversion import (
            graph_prediction_from_evidence,
            validate_and_convert_graph_prediction,
        )
        from prototype.graph_baseline.graph_contract import (
            graph_from_reconstruction_target,
        )
        from prototype.model_data.vocab import NODE_TYPES

        unused, repaired = self._decoders()
        cases = (
            ("RE", 0, -0.08333395421504974),
            ("EE", 1, -0.015737812966108322),
            ("RE", 0, -0.030291300266981125),
            ("E", 0, -0.011095978319644928),
            ("EE", 1, -0.011643193662166595),
        )
        helper = V6GrammarTensorTests()
        for template, operation_index, raw_value in cases:
            with self.subTest(template=template, operation_index=operation_index):
                prediction = helper._authoritative_v6_prediction(template)
                operation_node = prediction.predicted_operation_node_indices[
                    operation_index
                ]
                operation_type = NODE_TYPES.tokens[
                    prediction.raw_nodes[operation_node].node_type_id
                ]
                compact_channel = 4 if operation_type == "extrude" else 5
                serialized_channel = 37 if operation_type == "extrude" else 38
                raw = torch.zeros(1, 6)
                raw[0, compact_channel] = raw_value
                normalized = float(
                    repaired.parameterize_remaining_geometry(raw)[
                        0, compact_channel
                    ].item()
                )
                nodes = list(prediction.raw_nodes)
                geometry = list(nodes[operation_node].normalized_geometry)
                geometry[serialized_channel] = normalized
                nodes[operation_node] = replace(
                    nodes[operation_node], normalized_geometry=tuple(geometry)
                )
                prediction = replace(prediction, raw_nodes=tuple(nodes))
                target = procedural_fixture(template).target
                graph = graph_from_reconstruction_target(target)
                classes = [[0] * graph.node_count for unused in range(graph.node_count)]
                for edge in graph.directed_typed_edges:
                    classes[edge.source][edge.destination] = edge.edge_type_id
                graph_prediction = graph_prediction_from_evidence(
                    prediction,
                    classes,
                    classes,
                    [[False] * graph.node_count for unused in range(graph.node_count)],
                )
                result = validate_and_convert_graph_prediction(graph_prediction)
                self.assertTrue(result.reconstruction_target.valid)
                self.assertTrue(result.controlled_domain.valid)
                self.assertNotIn(
                    "invalid_operation_parameter",
                    result.controlled_domain.failure_codes,
                )
                self.assertFalse(hasattr(result, "cad_kernel_result"))

    def test_checkpoint_identity_rejects_legacy_repaired_interchange(self):
        from prototype.graph_encoder.checkpoint import (
            GE1CheckpointError,
            load_ge1_checkpoint,
            save_ge1_checkpoint,
        )
        from prototype.graph_encoder.config import (
            frozen_encoder_config,
            legacy_frozen_encoder_config,
        )
        from prototype.graph_encoder.decoder_contract import (
            LEGACY_OPERATION_MAGNITUDE_PARAMETERIZATION,
            POSITIVE_OPERATION_MAGNITUDE_PARAMETERIZATION,
        )
        from prototype.graph_encoder.model import build_ge1_model

        legacy = build_ge1_model(legacy_frozen_encoder_config("flat"))
        repaired = build_ge1_model(frozen_encoder_config("flat"))
        with tempfile.TemporaryDirectory() as temporary:
            legacy_path = Path(temporary) / "legacy.pt"
            repaired_path = Path(temporary) / "repaired.pt"
            save_ge1_checkpoint(legacy_path, legacy, code_revision="a" * 40)
            save_ge1_checkpoint(repaired_path, repaired, code_revision="a" * 40)
            with self.assertRaises(GE1CheckpointError) as caught:
                load_ge1_checkpoint(legacy_path)
            self.assertEqual(caught.exception.code, "invalid_ge1_checkpoint")
            loaded_legacy, legacy_payload = load_ge1_checkpoint(
                legacy_path,
                expected_operation_magnitude_parameterization=(
                    LEGACY_OPERATION_MAGNITUDE_PARAMETERIZATION
                ),
            )
            loaded_repaired, repaired_payload = load_ge1_checkpoint(
                repaired_path,
                expected_operation_magnitude_parameterization=(
                    POSITIVE_OPERATION_MAGNITUDE_PARAMETERIZATION
                ),
            )
        self.assertEqual(
            loaded_legacy.config.operation_magnitude_parameterization,
            LEGACY_OPERATION_MAGNITUDE_PARAMETERIZATION,
        )
        self.assertEqual(
            loaded_repaired.config.operation_magnitude_parameterization,
            POSITIVE_OPERATION_MAGNITUDE_PARAMETERIZATION,
        )
        self.assertEqual(
            legacy_payload["operation_magnitude_parameterization"],
            LEGACY_OPERATION_MAGNITUDE_PARAMETERIZATION,
        )
        self.assertEqual(
            repaired_payload["operation_magnitude_parameterization"],
            POSITIVE_OPERATION_MAGNITUDE_PARAMETERIZATION,
        )

    def test_recovery_checkpoint_rejects_legacy_repaired_interchange(self):
        from types import SimpleNamespace

        from prototype.graph_encoder.config import (
            GE1TrainingConfig,
            frozen_encoder_config,
            legacy_frozen_encoder_config,
        )
        from prototype.graph_encoder.errors import GraphEncoderError
        from prototype.graph_encoder.model import build_ge1_model
        from prototype.graph_encoder.provenance import (
            training_partition_identity,
        )
        from prototype.graph_encoder.training import (
            load_training_checkpoint,
            plateau_state,
            training_checkpoint_payload,
        )

        commit = "a" * 40
        legacy = build_ge1_model(legacy_frozen_encoder_config("flat"))
        legacy_optimizer = torch.optim.AdamW(legacy.parameters(), lr=1e-3)
        partition = training_partition_identity(("a", "b", "c", "d"))
        provenance = SimpleNamespace(
            source_tree_sha256="b" * 64,
            to_dict=lambda: {
                "git_commit": commit,
                "source_tree_sha256": "b" * 64,
                "operation_magnitude_parameterization": (
                    legacy.config.operation_magnitude_parameterization
                ),
            },
        )
        payload = training_checkpoint_payload(
            model=legacy,
            optimizer=legacy_optimizer,
            training_config=GE1TrainingConfig(),
            completed_epoch=0,
            optimizer_step_count=0,
            training_example_presentations=0,
            plateau=plateau_state(()),
            epoch_records=(),
            family_order_history=(),
            data_order_random_state=None,
            provenance=provenance,
            partition_identity=partition,
            expected_code_revision=commit,
        )
        self.assertEqual(
            payload["model_config"]["operation_magnitude_parameterization"],
            legacy.config.operation_magnitude_parameterization,
        )
        self.assertEqual(
            payload["provenance"]["operation_magnitude_parameterization"],
            legacy.config.operation_magnitude_parameterization,
        )
        repaired = build_ge1_model(frozen_encoder_config("flat"))
        repaired_optimizer = torch.optim.AdamW(
            repaired.parameters(), lr=1e-3
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "legacy-recovery.pt"
            torch.save(payload, path)
            with self.assertRaises(GraphEncoderError) as caught:
                load_training_checkpoint(
                    path,
                    model=repaired,
                    optimizer=repaired_optimizer,
                    training_config=GE1TrainingConfig(),
                    partition_identity=partition,
                    expected_code_revision=commit,
                    restore_rng=False,
                )
        self.assertEqual(caught.exception.code, "invalid_training_checkpoint")

    def test_both_repaired_arms_retain_matched_shared_initialization(self):
        from prototype.graph_encoder.decoder_contract import (
            POSITIVE_OPERATION_MAGNITUDE_PARAMETERIZATION,
        )
        from prototype.graph_encoder.model import build_matched_ge1_models

        flat, graph = build_matched_ge1_models(seed=2026)
        self.assertEqual(
            flat.decoder.operation_magnitude_parameterization,
            POSITIVE_OPERATION_MAGNITUDE_PARAMETERIZATION,
        )
        self.assertEqual(
            graph.decoder.operation_magnitude_parameterization,
            POSITIVE_OPERATION_MAGNITUDE_PARAMETERIZATION,
        )
        self.assertEqual(tuple(flat.decoder.state_dict()), tuple(graph.decoder.state_dict()))
        self.assertTrue(all(
            torch.equal(flat.decoder.state_dict()[name], graph.decoder.state_dict()[name])
            for name in flat.decoder.state_dict()
        ))


if __name__ == "__main__":
    unittest.main()
