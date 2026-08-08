"""Real-PyTorch C5 decoder parity, identity, loss, and checkpoint tests."""

from __future__ import annotations

from dataclasses import fields, replace
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from prototype.graph_encoder.batching import build_paired_batch
from prototype.graph_encoder.tests.fixtures import procedural_fixture


try:
    import torch
except ImportError:
    torch = None


TORCH_REASON = "C5 real-tensor tests require Python 3.8/PyTorch 1.11 CPU"


@unittest.skipUnless(torch is not None, TORCH_REASON)
class C5TensorTestCase(unittest.TestCase):
    def _paired(self, templates=("E", "R")):
        return build_paired_batch(tuple(
            procedural_fixture(name).physical for name in templates
        ))

    def _fixture(self, templates=("E", "R")):
        from prototype.constrained_profile_decoder import profile_targets_for_loss

        paired = self._paired(templates)
        flat = paired.flat_input.to_torch(torch)
        target = paired.target.to_torch(torch)
        profiles = profile_targets_for_loss(paired.target, flat["geometry"])
        return paired, flat, target, profiles

    def _reference_and_decoder(self):
        from prototype.graph_baseline.config import GraphV1Config
        from prototype.graph_baseline.model import GraphV1Model
        from prototype.graph_encoder.shared_decoder import SharedGE1Decoder

        torch.manual_seed(1201)
        reference = GraphV1Model(GraphV1Config()).eval()
        decoder = SharedGE1Decoder.from_graph_v1(reference).eval()
        return reference, decoder

    def _outputs(self, templates=("E", "R")):
        reference, decoder = self._reference_and_decoder()
        paired, flat, target, profiles = self._fixture(templates)
        torch.manual_seed(1203)
        memory = torch.randn(
            len(templates),
            decoder.config.latent_tokens,
            decoder.config.model_dim,
        )
        zero = memory.sum() * 0.0
        encoded = SimpleNamespace(
            memory=memory,
            vq=SimpleNamespace(
                loss=zero,
                per_example_loss=memory.new_zeros(len(templates)),
                indices=torch.zeros(
                    len(templates), decoder.config.latent_tokens,
                    dtype=torch.long,
                ),
                assignment_counts=torch.zeros(
                    decoder.config.codebook_size, dtype=torch.long
                ),
                active_code_count=torch.zeros((), dtype=torch.long),
                utilization=zero,
                perplexity=zero,
            ),
        )
        with torch.no_grad():
            with patch.object(
                reference, "encode_to_memory", return_value=encoded
            ):
                retained = reference(
                    target=target, profile_targets=profiles, **flat
                )
            shared = decoder.teacher_forced(memory, target, profiles)
        return (
            reference, decoder, paired, flat, target, profiles,
            memory, retained, shared,
        )


class DecoderComponentParityTests(C5TensorTestCase):
    def test_exact_constrained_v6_node_and_geometry_surface(self):
        from prototype.graph_encoder.shared_decoder import assert_exact_tensor_parity

        *unused, retained, shared = self._outputs()
        tensor_names = (
            "decoded_states",
            "node_type_logits",
            "profile_family_logits",
            "raw_profile_parameters",
            "constrained_profile_parameters",
            "remaining_geometry",
            "remaining_geometry_scattered",
            "remaining_geometry_mask",
            "training_reference_plane_geometry",
            "training_reference_plane_geometry_mask",
            "training_primitive_type_ids",
            "training_profile_geometry",
            "training_profile_geometry_mask",
            "training_geometry",
            "training_geometry_mask",
            "graph_raw_node_type_argmax_ids",
            "authoritative_graph_node_type_ids",
            "graph_node_type_correction_mask",
            "graph_legal_node_type_mask",
        )
        for name in tensor_names:
            assert_exact_tensor_parity(
                getattr(retained, name),
                getattr(shared, name),
                stage="node_geometry",
                name=name,
            )
        self.assertEqual(
            retained.graph_grammar_state_evidence,
            shared.graph_grammar_state_evidence,
        )
        for index, (expected, observed) in enumerate(zip(
            retained.categorical_logits, shared.categorical_logits
        )):
            assert_exact_tensor_parity(
                expected, observed, stage="categorical", name=str(index)
            )

    def test_exact_graph_v1_main_pair_logits_and_excluded_bias(self):
        *prefix, retained, shared = self._outputs()
        decoder = prefix[1]
        self.assertFalse(hasattr(decoder, "source_position_factor"))
        self.assertFalse(hasattr(decoder, "destination_position_factor"))
        self.assertFalse(hasattr(decoder, "position_class_projection"))
        torch.testing.assert_close(
            retained.graph_main_pair_logits,
            shared.graph_main_pair_logits,
            rtol=0.0,
            atol=0.0,
        )
        self.assertTrue(torch.equal(
            shared.graph_edge_logits, shared.graph_main_pair_logits
        ))
        self.assertTrue(torch.equal(
            shared.graph_position_bias_logits,
            torch.zeros_like(shared.graph_position_bias_logits),
        ))

    def test_exact_constrained_graph_contract_and_conversion_for_main_path(self):
        from prototype.graph_baseline.conversion import (
            graph_v1_teacher_forced_predictions,
            validate_and_convert_graph_prediction,
        )

        *prefix, retained, shared = self._outputs()
        target = prefix[4]
        retained_main_only = replace(
            retained,
            # Latent indices are encoder/VQ provenance, not a decoder or graph
            # contract output.  Put the retained decoder evidence under the
            # same continuous-memory provenance as C5 before comparing every
            # constrained graph field exactly.
            code_indices=shared.code_indices,
            graph_edge_logits=retained.graph_main_pair_logits,
            graph_position_bias_logits=torch.zeros_like(
                retained.graph_position_bias_logits
            ),
        )
        expected = graph_v1_teacher_forced_predictions(
            retained_main_only,
            node_mask=target["node_mask"],
            node_count_source="procedural_test",
        )
        observed = graph_v1_teacher_forced_predictions(
            shared,
            node_mask=target["node_mask"],
            node_count_source="procedural_test",
        )
        self.assertEqual(expected, observed)
        self.assertEqual(
            tuple(validate_and_convert_graph_prediction(item) for item in expected),
            tuple(validate_and_convert_graph_prediction(item) for item in observed),
        )

    def test_parity_error_reports_earliest_tensor_and_differences(self):
        from prototype.graph_encoder.shared_decoder import (
            DecoderParityError,
            assert_exact_tensor_parity,
        )

        expected = torch.zeros(2, 3)
        observed = expected.clone()
        observed[1, 2] = 0.5
        with self.assertRaises(DecoderParityError) as caught:
            assert_exact_tensor_parity(
                expected, observed, stage="edge", name="main_pair_logits"
            )
        self.assertEqual(caught.exception.stage, "edge")
        self.assertEqual(caught.exception.name, "main_pair_logits")
        self.assertEqual(caught.exception.maximum_absolute_difference, 0.5)

    def test_original_encoders_are_bypassed_by_component_parity(self):
        reference, decoder, unused_paired, flat, target, profiles = (
            self._outputs()[:6]
        )
        memory = torch.zeros(2, 2, 32)
        with patch.object(
            reference, "encode_to_memory", side_effect=AssertionError
        ) as retained_encoder, patch(
            "prototype.graph_encoder.encoders.FlatProgramEncoder.forward",
            side_effect=AssertionError,
        ) as ge1_encoder:
            decoder.teacher_forced(memory, target, profiles)
        retained_encoder.assert_not_called()
        ge1_encoder.assert_not_called()


class SharedDecoderIdentityTests(C5TensorTestCase):
    def test_matched_decoders_have_equal_state_and_disjoint_objects(self):
        from prototype.graph_encoder.model import build_matched_ge1_models
        from prototype.graph_encoder.shared_decoder import SharedGE1Decoder

        flat, graph = build_matched_ge1_models(seed=2026)
        self.assertIs(type(flat.decoder), SharedGE1Decoder)
        self.assertIs(type(graph.decoder), SharedGE1Decoder)
        self.assertIs(flat.decoder.forward.__func__, graph.decoder.forward.__func__)
        flat_state = flat.decoder.state_dict()
        graph_state = graph.decoder.state_dict()
        self.assertEqual(tuple(flat_state), tuple(graph_state))
        self.assertTrue(all(
            torch.equal(flat_state[name], graph_state[name]) for name in flat_state
        ))
        flat_parameters = dict(flat.decoder.named_parameters())
        graph_parameters = dict(graph.decoder.named_parameters())
        self.assertEqual(
            {name: tuple(value.shape) for name, value in flat_parameters.items()},
            {name: tuple(value.shape) for name, value in graph_parameters.items()},
        )
        self.assertTrue(all(
            flat_parameters[name] is not graph_parameters[name]
            for name in flat_parameters
        ))
        flat_buffers = dict(flat.decoder.named_buffers())
        graph_buffers = dict(graph.decoder.named_buffers())
        self.assertEqual(set(flat_buffers), set(graph_buffers))
        self.assertTrue(all(
            flat_buffers[name] is not graph_buffers[name] for name in flat_buffers
        ))

    def test_mutation_is_independent_and_construction_order_is_irrelevant(self):
        from prototype.graph_encoder.model import build_matched_ge1_models

        flat, graph = build_matched_ge1_models(
            seed=2026, encoder_order=("flat", "typed_graph")
        )
        reversed_flat, reversed_graph = build_matched_ge1_models(
            seed=2026, encoder_order=("typed_graph", "flat")
        )
        before = {
            name: value.clone() for name, value in graph.decoder.state_dict().items()
        }
        with torch.no_grad():
            next(flat.decoder.parameters()).add_(1.0)
            flat.decoder.relative_position_denominator.add_(1.0)
        self.assertTrue(all(
            torch.equal(before[name], graph.decoder.state_dict()[name])
            for name in before
        ))
        for expected, observed in (
            (reversed_flat.decoder.state_dict(), graph.decoder.state_dict()),
            (reversed_graph.decoder.state_dict(), graph.decoder.state_dict()),
        ):
            self.assertEqual(tuple(expected), tuple(observed))
            self.assertTrue(all(
                torch.equal(expected[name], observed[name]) for name in expected
            ))

    def test_both_arm_wrappers_pass_only_common_memory_to_decoder(self):
        from prototype.graph_encoder.encoders import EncodedMemory
        from prototype.graph_encoder.model import build_matched_ge1_models

        flat, graph = build_matched_ge1_models(seed=2026)
        for model in (flat, graph):
            memory = torch.randn(2, 2, 32)
            encoded = EncodedMemory(
                memory=memory,
                prequant=torch.randn(2, 2, 16),
            )
            sentinel = object()
            with patch.object(
                model.encoder, "forward", return_value=encoded
            ) as encoder_spy, patch.object(
                model.decoder, "forward", return_value=sentinel
            ) as decoder_spy:
                observed = model(
                    {},
                    node_counts=torch.tensor((4, 5), dtype=torch.long),
                    node_count_source="procedural_test",
                )
            self.assertIs(observed, sentinel)
            encoder_spy.assert_called_once_with()
            decoder_spy.assert_called_once()
            positional, keywords = decoder_spy.call_args
            self.assertEqual(len(positional), 1)
            self.assertIs(positional[0], memory)
            self.assertNotIn("prequant", keywords)
            self.assertNotIn("encoder", keywords)

    def test_complete_models_share_no_parameter_objects(self):
        from prototype.graph_encoder.model import build_matched_ge1_models

        flat, graph = build_matched_ge1_models(seed=2026)
        flat_ids = {id(item) for item in flat.parameters()}
        graph_ids = {id(item) for item in graph.parameters()}
        self.assertTrue(flat_ids.isdisjoint(graph_ids))


class PredictionLossAndCheckpointTests(C5TensorTestCase):
    def test_returned_records_carry_ge1_provenance_not_the_v6_shim(self):
        """F1: the compatibility literal must never escape the decoder."""

        from prototype.flat_baseline.constrained_v6_autonomous import (
            V6_ENCODED_MEMORY_SOURCE,
            validate_and_convert_v6_autonomous_prediction,
        )
        from prototype.graph_encoder.shared_decoder import (
            GE1_CONTINUOUS_MEMORY_SOURCE,
            V6_ENTRY_POINT_COMPATIBILITY_SOURCE,
        )
        from prototype.node_grammar import NodeGrammarError

        decoder = self._reference_and_decoder()[1]
        prediction = decoder(
            torch.zeros(2, 2, 32),
            node_counts=torch.tensor((4, 5), dtype=torch.long),
            node_count_source="procedural_test",
        )
        constrained = prediction.constrained_prediction
        self.assertTrue(constrained)
        for index, item in enumerate(constrained):
            node_prediction = item.node_prediction
            with self.subTest(row=index):
                # Corrected to GE1's true, continuous provenance.
                self.assertEqual(
                    node_prediction.encoded_memory_source,
                    GE1_CONTINUOUS_MEMORY_SOURCE,
                )
                # The inherited literal must not survive into any record.
                self.assertNotEqual(
                    node_prediction.encoded_memory_source,
                    V6_ENTRY_POINT_COMPATIBILITY_SOURCE,
                )
                self.assertNotEqual(
                    node_prediction.encoded_memory_source,
                    V6_ENCODED_MEMORY_SOURCE,
                )
                # No nearest-code assignment occurred, so no indices exist. The
                # inert zeros handed to the inherited validator are discarded
                # rather than reported as a collapsed codebook.
                self.assertEqual(node_prediction.latent_indices, ())
                with self.assertRaises(NodeGrammarError) as caught:
                    validate_and_convert_v6_autonomous_prediction(
                        node_prediction,
                        max_operations=2,
                    )
                self.assertEqual(
                    getattr(caught.exception, "code", None),
                    "invalid_v6_node_selection",
                )
                self.assertEqual(
                    getattr(caught.exception, "detail", None),
                    "encoded memory provenance is wrong",
                )

    def test_autonomous_levels_are_separate_target_free_and_deterministic(self):
        decoder = self._reference_and_decoder()[1]
        memory = torch.zeros(2, 2, 32)
        counts = torch.tensor((4, 5), dtype=torch.long)
        first = decoder(
            memory, node_counts=counts, node_count_source="procedural_test"
        )
        second = decoder(
            memory, node_counts=counts, node_count_source="procedural_test"
        )
        self.assertEqual(first.version, second.version)
        self.assertEqual(first.constrained_prediction, second.constrained_prediction)
        self.assertEqual(first.converted_prediction, second.converted_prediction)
        self.assertIsNot(first.raw_prediction, first.constrained_prediction)
        self.assertIsNot(first.constrained_prediction, first.converted_prediction)
        self.assertTrue(all(
            not item.raised_failure for item in first.converted_prediction
        ))
        self.assertTrue(all(
            hasattr(item.result, "primary_failure")
            and hasattr(item.result, "secondary_failures")
            for item in first.converted_prediction
        ))
        self.assertTrue(all(
            torch.isfinite(item.graph_edge_components.edge_logits).all()
            for item in first.raw_prediction
        ))
        raw_before = tuple(
            item.graph_edge_components.edge_logits.clone()
            for item in first.raw_prediction
        )
        constrained_before = first.constrained_prediction
        from prototype.graph_baseline.conversion import (
            validate_and_convert_graph_prediction,
        )
        tuple(
            validate_and_convert_graph_prediction(item)
            for item in first.constrained_prediction
        )
        self.assertTrue(all(
            torch.equal(before, item.graph_edge_components.edge_logits)
            for before, item in zip(raw_before, first.raw_prediction)
        ))
        self.assertEqual(constrained_before, first.constrained_prediction)

    def test_common_loss_is_exact_graph_v1_per_example_assembly(self):
        from prototype.graph_baseline.config import GraphV1Config
        from prototype.graph_baseline.losses import graph_v1_loss
        from prototype.graph_encoder.config import frozen_encoder_config
        from prototype.graph_encoder.losses import common_ge1_loss

        *prefix, shared = self._outputs()
        target = prefix[4]
        profiles = prefix[5]
        observed = common_ge1_loss(
            shared, target, profiles, frozen_encoder_config("flat")
        )
        expected = graph_v1_loss(shared, target, profiles, GraphV1Config())
        self.assertEqual(
            tuple(field.name for field in fields(observed)),
            tuple(field.name for field in fields(expected)),
        )
        for name in observed.per_example:
            torch.testing.assert_close(
                observed.per_example[name],
                expected.per_example[name],
                rtol=0.0,
                atol=0.0,
            )
        torch.testing.assert_close(observed.total, expected.total, rtol=0.0, atol=0.0)
        self.assertTrue(torch.equal(
            observed.vq_commitment, torch.zeros_like(observed.vq_commitment)
        ))

    def test_padding_and_masked_pairs_contribute_zero(self):
        from prototype.graph_encoder.config import frozen_encoder_config
        from prototype.graph_encoder.losses import common_ge1_loss

        *prefix, shared = self._outputs(("E", "R"))
        target = prefix[4]
        profiles = prefix[5]
        baseline = common_ge1_loss(
            shared, target, profiles, frozen_encoder_config("flat")
        )
        node = shared.node_type_logits.clone()
        edge = shared.graph_edge_logits.clone()
        remaining = shared.remaining_geometry.clone()
        categorical = tuple(item.clone() for item in shared.categorical_logits)
        node_counts = target["node_mask"].long().sum(dim=1)
        padded_row = next(
            index for index, count in enumerate(node_counts.tolist())
            if count < target["node_mask"].size(1)
        )
        padded_position = int(node_counts[padded_row].item())
        self.assertFalse(bool(
            target["node_mask"][padded_row, padded_position].item()
        ))
        node[padded_row, padded_position] = 1000.0
        remaining[padded_row, padded_position] = 1.0
        edge[padded_row, padded_position, :, :] = 1000.0
        edge[padded_row, :, padded_position, :] = -1000.0
        for item in categorical:
            item[padded_row, padded_position] = 1000.0
        changed = replace(
            shared,
            node_type_logits=node,
            graph_edge_logits=edge,
            remaining_geometry=remaining,
            categorical_logits=categorical,
        )
        observed = common_ge1_loss(
            changed, target, profiles, frozen_encoder_config("flat")
        )
        torch.testing.assert_close(
            observed.per_example["total"][padded_row],
            baseline.per_example["total"][padded_row],
            rtol=0.0,
            atol=0.0,
        )

    def test_forward_loss_and_decoder_gradients_are_finite(self):
        from prototype.graph_encoder.config import frozen_encoder_config
        from prototype.graph_encoder.losses import common_ge1_loss

        reference, decoder, paired, flat, target, profiles = self._outputs()[:6]
        del reference, paired, flat
        decoder.train()
        memory = torch.randn(2, 2, 32, requires_grad=True)
        output = decoder.teacher_forced(memory, target, profiles)
        loss = common_ge1_loss(
            output, target, profiles, frozen_encoder_config("typed_graph")
        )
        self.assertTrue(torch.isfinite(loss.total))
        loss.total.backward()
        gradients = tuple(
            item.grad for item in decoder.parameters() if item.requires_grad
        )
        self.assertTrue(gradients)
        self.assertTrue(all(item is not None for item in gradients))
        self.assertTrue(all(torch.isfinite(item).all() for item in gradients))
        self.assertIsNotNone(memory.grad)
        self.assertTrue(torch.isfinite(memory.grad).all())

    def test_strict_checkpoint_round_trip_and_rejections(self):
        from prototype.graph_encoder.checkpoint import (
            GE1CheckpointError,
            ge1_checkpoint_payload,
            load_ge1_checkpoint,
            save_ge1_checkpoint,
        )
        from prototype.graph_encoder.model import build_matched_ge1_models

        model = build_matched_ge1_models(seed=2026)[0].eval()
        revision = "1" * 40
        paired, flat, target, profiles = self._fixture()
        with torch.no_grad():
            before = model.teacher_forced(flat, target, profiles).decoder_output
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "ge1.pt"
            save_ge1_checkpoint(path, model, code_revision=revision)
            loaded, payload = load_ge1_checkpoint(
                path, expected_code_revision=revision
            )
            loaded.eval()
            with torch.no_grad():
                after = loaded.teacher_forced(
                    flat, target, profiles
                ).decoder_output
            torch.testing.assert_close(
                before.graph_edge_logits,
                after.graph_edge_logits,
                rtol=0.0,
                atol=0.0,
            )
            self.assertEqual(
                tuple(model.decoder.state_dict()),
                tuple(loaded.decoder.state_dict()),
            )
            bad = dict(payload)
            bad["shared_decoder_version"] = "wrong"
            torch.save(bad, str(path))
            with self.assertRaises(GE1CheckpointError):
                load_ge1_checkpoint(path)
            bad = dict(payload)
            bad["model_state"] = dict(payload["model_state"])
            bad["model_state"].pop(next(iter(bad["model_state"])))
            torch.save(bad, str(path))
            with self.assertRaises(GE1CheckpointError):
                load_ge1_checkpoint(path)
            bad = dict(payload)
            bad["model_state"] = dict(payload["model_state"])
            tensor_name = next(
                name for name, value in bad["model_state"].items()
                if value.dtype.is_floating_point
            )
            bad["model_state"][tensor_name] = bad["model_state"][
                tensor_name
            ].to(dtype=torch.float64)
            torch.save(bad, str(path))
            with self.assertRaises(GE1CheckpointError):
                load_ge1_checkpoint(path)
        metadata = ge1_checkpoint_payload(model, code_revision=revision)
        self.assertEqual(metadata["encoder_feedforward_width"], 192)
        self.assertEqual(metadata["relation_basis_count"], 2)


if __name__ == "__main__":
    unittest.main()
