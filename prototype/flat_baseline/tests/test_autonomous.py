"""Length-conditioned autoregressive decoding regression tests."""

from __future__ import annotations

import inspect
from unittest import mock
import tempfile
import unittest

try:
    import torch
except ImportError:
    torch = None

from prototype.model_data.adapters import adapt_flat_mixed
from prototype.model_data.batching import collate_flat
from prototype.model_data.loader import load_physical_examples
from prototype.model_data.tests.fixtures import source, write_physical_corpus

if torch is not None:
    from prototype.flat_baseline.autonomous import (
        AutonomousDecodingError,
        RAW_PREFIX_FEEDBACK,
        decode_length_conditioned,
        derive_geometry_mask,
    )
    from prototype.flat_baseline.config import FlatBaselineConfig
    from prototype.flat_baseline.model import (
        FlatMixedVQModel,
        FlatMixedVQOutput,
    )
    from prototype.model_data.geometry import GEOMETRY_WIDTH
    from prototype.model_data.vocab import (
        NODE_TYPES,
        PRIMITIVE_TYPES,
    )
    # VQ returns inputs + (looked_up - inputs), which is mathematically the
    # lookup but adds float32 subtraction/addition rounding before projection.
    FLOAT32_STRAIGHT_THROUGH_RTOL = 8.0 * torch.finfo(torch.float32).eps
    FLOAT32_STRAIGHT_THROUGH_ATOL = torch.finfo(torch.float32).eps


TORCH_REASON = "real PyTorch execution is deferred to the Adroit environment"


def _batch(templates=("E", "R")):
    temporary = tempfile.TemporaryDirectory()
    write_physical_corpus(
        temporary.name, tuple(source(template) for template in templates)
    )
    examples = load_physical_examples(temporary.name)
    batch = collate_flat(tuple(adapt_flat_mixed(item) for item in examples))
    return (
        temporary,
        batch.to_torch(torch),
        batch.target.to_torch(torch),
    )


def _config(**changes):
    values = {
        "model_dim": 8,
        "num_heads": 2,
        "feedforward_dim": 12,
        "encoder_layers": 1,
        "decoder_layers": 1,
        "dropout": 0.25,
        "max_nodes": 5,
        "max_operations": 2,
        "latent_tokens": 2,
        "codebook_size": 4,
        "codebook_dim": 4,
        "edge_pair_dim": 6,
    }
    values.update(changes)
    return FlatBaselineConfig(**values)


def _indices(batch_size=1):
    return torch.tensor([[0, 1]] * batch_size, dtype=torch.long)


def _force_head_choice(head, choice):
    with torch.no_grad():
        head.weight.zero_()
        head.bias.fill_(-10.0)
        head.bias[choice] = 10.0


def _committed_forward_reference(model, inputs, target):
    """Test-local copy of the committed pre-extraction operation order."""

    categorical_ids = inputs["categorical_ids"]
    geometry = inputs["geometry"]
    geometry_mask = inputs["geometry_mask"]
    padding_mask = inputs["padding_mask"]
    model._validate_inputs(
        categorical_ids, geometry, geometry_mask, padding_mask, target
    )
    batch_size, node_count = categorical_ids.shape[:2]
    positions = torch.arange(
        node_count, device=categorical_ids.device
    ).unsqueeze(0)
    content = model._record_content(
        categorical_ids, geometry, geometry_mask
    )
    nodes = model.input_norm(
        content + model.node_position_embedding(positions)
    )
    queries = model.latent_queries.unsqueeze(0).expand(
        batch_size, -1, -1
    )
    encoder_input = torch.cat((queries, nodes), dim=1)
    query_mask = torch.zeros(
        batch_size,
        model.config.latent_tokens,
        dtype=torch.bool,
        device=padding_mask.device,
    )
    encoder_padding = torch.cat((query_mask, ~padding_mask), dim=1)
    encoded = model.encoder(
        encoder_input, src_key_padding_mask=encoder_padding
    )
    latent = model.to_codebook(
        encoded[:, : model.config.latent_tokens]
    )
    vq = model.vq(latent)
    memory = model.from_codebook(vq.quantized)

    target_categories = torch.cat(
        (
            target["node_type_ids"].unsqueeze(-1),
            target["categorical_attributes"],
        ),
        dim=-1,
    )
    target_content = model._record_content(
        target_categories,
        target["geometry"],
        target["geometry_mask"],
    )
    bos = model.bos.view(1, 1, -1).expand(batch_size, 1, -1)
    shifted = torch.cat((bos, target_content[:, :-1]), dim=1)
    decoder_input = model.decoder_input_norm(
        shifted + model.decoder_position_embedding(positions)
    )
    causal_mask = torch.triu(
        torch.ones(
            node_count,
            node_count,
            dtype=torch.bool,
            device=categorical_ids.device,
        ),
        diagonal=1,
    )
    decoder_valid = torch.cat(
        (
            torch.ones(
                batch_size,
                1,
                dtype=torch.bool,
                device=padding_mask.device,
            ),
            target["node_mask"][:, :-1],
        ),
        dim=1,
    )
    decoded = model.decoder(
        decoder_input,
        memory,
        tgt_mask=causal_mask,
        tgt_key_padding_mask=~decoder_valid,
    )

    source_states = model.edge_source(decoded).unsqueeze(2)
    target_states = model.edge_target(decoded).unsqueeze(1)
    pairs = torch.tanh(source_states + target_states)
    edge_presence = model.edge_presence_head(pairs).squeeze(-1)
    edge_types = model.edge_type_head(pairs)
    operation_logits = torch.einsum(
        "od,bnd->bon",
        model.operation_queries,
        model.operation_keys(decoded),
    )
    minimum = torch.finfo(operation_logits.dtype).min
    operation_logits = operation_logits.masked_fill(
        ~target["node_mask"].unsqueeze(1), minimum
    )
    return FlatMixedVQOutput(
        decoded,
        model.node_type_head(decoded),
        tuple(head(decoded) for head in model.categorical_heads),
        torch.tanh(model.geometry_head(decoded)),
        edge_presence,
        edge_types,
        operation_logits,
        memory,
        vq.loss,
        vq.per_example_loss,
        vq.indices,
        vq.assignment_counts,
        vq.active_code_count,
        vq.utilization,
        vq.perplexity,
    )


def _helper_forward_composition(model, inputs, target):
    """Compose extracted helpers while retaining the committed target shift."""

    encoded = model.encode_to_memory(
        inputs["categorical_ids"],
        inputs["geometry"],
        inputs["geometry_mask"],
        inputs["padding_mask"],
    )
    target_categories = torch.cat(
        (
            target["node_type_ids"].unsqueeze(-1),
            target["categorical_attributes"],
        ),
        dim=-1,
    )
    target_content = model._record_content(
        target_categories,
        target["geometry"],
        target["geometry_mask"],
    )
    batch_size = target_categories.size(0)
    bos = model.bos.view(1, 1, -1).expand(batch_size, 1, -1)
    shifted = torch.cat((bos, target_content[:, :-1]), dim=1)
    decoder_valid = torch.cat(
        (
            torch.ones(
                batch_size,
                1,
                dtype=torch.bool,
                device=inputs["padding_mask"].device,
            ),
            target["node_mask"][:, :-1],
        ),
        dim=1,
    )
    decoded = model._decode_embedded_prefix(
        shifted, encoded.memory, decoder_valid
    )
    relations = model.decode_relations(decoded, target["node_mask"])
    return FlatMixedVQOutput(
        decoded,
        model.node_type_head(decoded),
        tuple(head(decoded) for head in model.categorical_heads),
        torch.tanh(model.geometry_head(decoded)),
        relations.edge_presence_logits,
        relations.edge_type_logits,
        relations.operation_pointer_logits,
        encoded.memory,
        encoded.vq.loss,
        encoded.vq.per_example_loss,
        encoded.vq.indices,
        encoded.vq.assignment_counts,
        encoded.vq.active_code_count,
        encoded.vq.utilization,
        encoded.vq.perplexity,
    )


def _state_dict_contract(model):
    return tuple(
        (name, tuple(value.shape), str(value.dtype))
        for name, value in model.state_dict().items()
    )


@unittest.skipUnless(torch is not None, TORCH_REASON)
class InterfaceEquivalenceTests(unittest.TestCase):
    def test_extracted_interfaces_are_exactly_forward_equivalent(self):
        torch.manual_seed(41)
        temporary, inputs, target = _batch()
        self.addCleanup(temporary.cleanup)
        model = FlatMixedVQModel(_config()).eval()
        record_widths = []
        original_record_content = model._record_content

        def record_spy(categories, geometry, geometry_mask):
            record_widths.append(categories.size(1))
            return original_record_content(
                categories, geometry, geometry_mask
            )

        with torch.no_grad():
            reference = _committed_forward_reference(
                model, inputs, target
            )
            with mock.patch.object(
                model, "_record_content", side_effect=record_spy
            ):
                public = model(target=target, **inputs)
            helpers = _helper_forward_composition(
                model, inputs, target
            )

        node_count = inputs["categorical_ids"].size(1)
        self.assertEqual(record_widths, [node_count, node_count])
        self.assertTrue(
            torch.equal(reference.code_indices, public.code_indices)
        )
        self.assertTrue(
            torch.equal(reference.code_indices, helpers.code_indices)
        )
        comparisons = (
            ("quantized_memory",),
            ("node_type_logits",),
            ("geometry",),
            ("edge_presence_logits",),
            ("edge_type_logits",),
            ("operation_pointer_logits",),
        )
        for (name,) in comparisons:
            expected = getattr(reference, name)
            for actual in (getattr(public, name), getattr(helpers, name)):
                self.assertEqual(expected.shape, actual.shape)
                torch.testing.assert_close(
                    expected, actual, rtol=0.0, atol=0.0
                )
        for actual_output in (public, helpers):
            self.assertEqual(
                len(reference.categorical_logits),
                len(actual_output.categorical_logits),
            )
            for expected, actual in zip(
                reference.categorical_logits,
                actual_output.categorical_logits,
            ):
                self.assertEqual(expected.shape, actual.shape)
                torch.testing.assert_close(
                    expected, actual, rtol=0.0, atol=0.0
                )

    def test_evaluation_encoding_does_not_modify_ema_buffers(self):
        temporary, inputs, _ = _batch(("E",))
        self.addCleanup(temporary.cleanup)
        model = FlatMixedVQModel(_config()).eval()
        before = {
            name: value.clone()
            for name, value in model.named_buffers()
            if name.startswith("vq.")
        }
        with torch.no_grad():
            model.encode_to_memory(
                inputs["categorical_ids"],
                inputs["geometry"],
                inputs["geometry_mask"],
                inputs["padding_mask"],
            )
        after = dict(model.named_buffers())
        for name, expected in before.items():
            torch.testing.assert_close(
                expected, after[name], rtol=0.0, atol=0.0
            )

    def test_supplied_latent_indices_validate_shape_dtype_and_bounds(self):
        model = FlatMixedVQModel(_config()).eval()
        with self.assertRaises(TypeError):
            model.memory_from_indices(torch.zeros((1, 2), dtype=torch.bool))
        with self.assertRaises(TypeError):
            model.memory_from_indices(torch.zeros((1, 2)))
        with self.assertRaises(ValueError):
            model.memory_from_indices(torch.zeros((1, 1), dtype=torch.long))
        with self.assertRaisesRegex(ValueError, "positive batch size"):
            model.memory_from_indices(
                torch.empty((0, 2), dtype=torch.long)
            )
        with self.assertRaises(ValueError):
            model.memory_from_indices(torch.tensor([[-1, 0]]))
        with self.assertRaises(ValueError):
            model.memory_from_indices(torch.tensor([[0, 4]]))
        memory = model.memory_from_indices(_indices())
        self.assertEqual(memory.shape, (1, 2, 8))

    def test_encoded_indices_reproduce_evaluation_memory_numerically(self):
        temporary, inputs, _ = _batch(("E",))
        self.addCleanup(temporary.cleanup)
        model = FlatMixedVQModel(_config()).eval()
        state_contract = _state_dict_contract(model)
        ema_before = {
            name: value.clone()
            for name, value in model.named_buffers()
            if name.startswith("vq.")
        }
        with torch.no_grad():
            encoded = model.encode_to_memory(
                inputs["categorical_ids"],
                inputs["geometry"],
                inputs["geometry_mask"],
                inputs["padding_mask"],
            )
            with mock.patch.object(
                model.vq,
                "forward",
                side_effect=AssertionError(
                    "memory_from_indices invoked the quantizer"
                ),
            ):
                looked_up = model.memory_from_indices(encoded.vq.indices)
                looked_up_again = model.memory_from_indices(
                    encoded.vq.indices
                )
            direct_code_vectors = torch.nn.functional.embedding(
                encoded.vq.indices, model.vq.embedding
            )
            directly_projected = model.from_codebook(
                direct_code_vectors
            )

        self.assertEqual(
            encoded.vq.quantized.shape, direct_code_vectors.shape
        )
        self.assertEqual(
            encoded.vq.quantized.dtype, direct_code_vectors.dtype
        )
        self.assertEqual(
            encoded.vq.quantized.device, direct_code_vectors.device
        )
        self.assertEqual(encoded.vq.quantized.dtype, torch.float32)
        torch.testing.assert_close(
            encoded.vq.quantized,
            direct_code_vectors,
            rtol=FLOAT32_STRAIGHT_THROUGH_RTOL,
            atol=FLOAT32_STRAIGHT_THROUGH_ATOL,
        )
        self.assertEqual(encoded.memory.shape, looked_up.shape)
        self.assertEqual(encoded.memory.dtype, looked_up.dtype)
        self.assertEqual(encoded.memory.device, looked_up.device)
        torch.testing.assert_close(
            encoded.memory,
            looked_up,
            rtol=FLOAT32_STRAIGHT_THROUGH_RTOL,
            atol=FLOAT32_STRAIGHT_THROUGH_ATOL,
        )
        torch.testing.assert_close(
            looked_up, directly_projected, rtol=0.0, atol=0.0
        )
        torch.testing.assert_close(
            looked_up, looked_up_again, rtol=0.0, atol=0.0
        )
        self.assertEqual(state_contract, _state_dict_contract(model))
        buffers_after = dict(model.named_buffers())
        for name, expected in ema_before.items():
            torch.testing.assert_close(
                expected, buffers_after[name], rtol=0.0, atol=0.0
            )

    def test_state_dict_contract_is_ordered_and_repeatable(self):
        first = FlatMixedVQModel(_config())
        second = FlatMixedVQModel(_config())
        first_contract = _state_dict_contract(first)
        self.assertTrue(first_contract)
        self.assertEqual(first_contract, _state_dict_contract(second))
        self.assertEqual(
            tuple(name for name, _, _ in first_contract),
            tuple(first.state_dict().keys()),
        )


@unittest.skipUnless(torch is not None, TORCH_REASON)
class GreedyDecodingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(73)
        self.model = FlatMixedVQModel(_config())

    def test_first_decoder_input_is_learned_bos_only(self):
        captured = []

        def capture(module, args):
            del module
            captured.append(args[0].detach().clone())

        handle = self.model.decoder.register_forward_pre_hook(capture)
        self.addCleanup(handle.remove)
        decode_length_conditioned(
            self.model, _indices(), node_counts=(1,)
        )
        expected = self.model.decoder_input_norm(
            self.model.bos.view(1, 1, -1)
            + self.model.decoder_position_embedding(
                torch.tensor([[0]], dtype=torch.long)
            )
        )
        self.assertEqual(captured[0].shape, (1, 1, 8))
        torch.testing.assert_close(
            captured[0], expected, rtol=0.0, atol=0.0
        )

    def test_prefix_grows_one_record_and_final_pass_is_aligned(self):
        prefix_lengths = []
        relation_shapes = []
        original_prefix = self.model.decode_prefix
        original_relations = self.model.decode_relations

        def prefix_spy(memory, categories, geometry, geometry_mask, **kwargs):
            prefix_lengths.append(categories.size(1))
            return original_prefix(
                memory,
                categories,
                geometry,
                geometry_mask,
                **kwargs
            )

        def relation_spy(states, mask):
            relation_shapes.append((tuple(states.shape), tuple(mask.shape)))
            return original_relations(states, mask)

        with mock.patch.object(
            self.model, "decode_prefix", side_effect=prefix_spy
        ), mock.patch.object(
            self.model, "decode_relations", side_effect=relation_spy
        ):
            decode_length_conditioned(
                self.model, _indices(), node_counts=(4,)
            )
        self.assertEqual(prefix_lengths, [0, 1, 2, 3, 3])
        self.assertEqual(relation_shapes, [((1, 4, 8), (1, 4))])

    def test_decoding_cannot_access_teacher_forced_forward_or_target(self):
        class TargetSentinel:
            def __call__(self, *args, **kwargs):
                del args, kwargs
                raise AssertionError(
                    "teacher-forced target path was accessed"
                )

        sentinel = TargetSentinel()
        with mock.patch.object(
            self.model,
            "forward",
            new=sentinel,
        ):
            result = decode_length_conditioned(
                self.model, _indices(), node_counts=(2,)
            )
        self.assertEqual(len(result[0].raw_nodes), 2)
        self.assertNotIn(
            "target", inspect.signature(decode_length_conditioned).parameters
        )
        self.assertNotIn(
            "operation_counts",
            inspect.signature(decode_length_conditioned).parameters,
        )

    def test_exact_length_and_checkpoint_maximum(self):
        result = decode_length_conditioned(
            self.model,
            _indices(),
            node_counts=(self.model.config.max_nodes,),
            node_count_source="target_canonical_metadata",
        )[0]
        self.assertEqual(len(result.raw_nodes), self.model.config.max_nodes)
        self.assertEqual(result.node_count, self.model.config.max_nodes)
        self.assertEqual(
            result.node_count_source, "target_canonical_metadata"
        )
        self.assertEqual(
            result.termination_reason, "requested_node_count_reached"
        )
        self.assertFalse(result.termination_is_learned)

    def test_invalid_node_counts_are_rejected(self):
        invalid = (0, -1, True, 1.0, self.model.config.max_nodes + 1)
        for value in invalid:
            with self.subTest(value=value):
                with self.assertRaises(AutonomousDecodingError):
                    decode_length_conditioned(
                        self.model, _indices(), node_counts=(value,)
                    )
        with self.assertRaises(AutonomousDecodingError):
            decode_length_conditioned(
                self.model, _indices(2), node_counts=(1,)
            )
        with self.assertRaisesRegex(ValueError, "positive batch size"):
            decode_length_conditioned(
                self.model,
                torch.empty((0, 2), dtype=torch.long),
                node_counts=(),
            )

    def test_greedy_output_is_deterministic_and_restores_mode(self):
        self.model.train()
        first = decode_length_conditioned(
            self.model, _indices(), node_counts=(3,)
        )
        self.assertTrue(self.model.training)
        second = decode_length_conditioned(
            self.model, _indices(), node_counts=(3,)
        )
        self.assertEqual(first, second)

    def test_training_mode_is_restored_after_decoding_exception(self):
        self.model.train()
        with mock.patch.object(
            self.model,
            "decode_prefix",
            side_effect=RuntimeError("forced prefix failure"),
        ):
            with self.assertRaisesRegex(
                RuntimeError, "forced prefix failure"
            ):
                decode_length_conditioned(
                    self.model, _indices(), node_counts=(2,)
                )
        self.assertTrue(self.model.training)

    def test_batch_size_one_and_greater_than_one(self):
        single = decode_length_conditioned(
            self.model, _indices(), node_counts=(2,)
        )
        indices = torch.tensor(
            ((0, 1), (2, 3)), dtype=torch.long
        )
        multiple = decode_length_conditioned(
            self.model, indices, node_counts=(2, 4)
        )
        self.assertEqual(len(single), 1)
        self.assertEqual(tuple(len(item.raw_nodes) for item in multiple), (2, 4))
        self.assertEqual(
            tuple(item.latent_indices for item in multiple),
            ((0, 1), (2, 3)),
        )
        separately = tuple(
            decode_length_conditioned(
                self.model,
                indices[index : index + 1],
                node_counts=(node_count,),
            )[0]
            for index, node_count in enumerate((2, 4))
        )
        self.assertEqual(multiple, separately)

    def test_raw_argmax_preserves_sentinel_winners(self):
        _force_head_choice(self.model.node_type_head, 0)
        for head in self.model.categorical_heads:
            _force_head_choice(head, 1)
        result = decode_length_conditioned(
            self.model, _indices(), node_counts=(1,)
        )[0]
        self.assertEqual(result.prefix_feedback, RAW_PREFIX_FEEDBACK)
        self.assertEqual(result.raw_nodes[0].node_type_id, 0)
        self.assertEqual(result.raw_nodes[0].categorical_ids, (1,) * 9)
        self.assertFalse(any(result.raw_nodes[0].derived_geometry_mask))

    def test_edge_presence_threshold_includes_exactly_zero(self):
        with torch.no_grad():
            self.model.edge_presence_head.weight.zero_()
            self.model.edge_presence_head.bias.zero_()
        result = decode_length_conditioned(
            self.model, _indices(), node_counts=(2,)
        )[0]
        self.assertEqual(len(result.raw_edges), 4)
        self.assertTrue(all(item.presence_logit == 0.0 for item in result.raw_edges))
        self.assertTrue(all(item.present for item in result.raw_edges))

    def test_operation_count_comes_only_from_generated_node_types(self):
        _force_head_choice(
            self.model.node_type_head, NODE_TYPES.id("extrude")
        )
        result = decode_length_conditioned(
            self.model, _indices(), node_counts=(4,)
        )[0]
        self.assertEqual(
            result.predicted_operation_node_indices, (0, 1, 2, 3)
        )
        self.assertEqual(result.predicted_operation_count, 4)
        self.assertTrue(result.operation_count_exceeds_limit)
        self.assertEqual(
            len(result.raw_operation_pointers),
            self.model.config.max_operations,
        )

    def test_zero_generated_operations_produces_no_pointer_records(self):
        _force_head_choice(
            self.model.node_type_head, NODE_TYPES.id("profile")
        )
        result = decode_length_conditioned(
            self.model, _indices(), node_counts=(3,)
        )[0]
        self.assertEqual(result.predicted_operation_count, 0)
        self.assertEqual(result.predicted_operation_node_indices, ())
        self.assertEqual(result.raw_operation_pointers, ())

    def test_sentinel_primitive_slots_do_not_activate_geometry(self):
        _force_head_choice(
            self.model.node_type_head, NODE_TYPES.id("sketch")
        )
        primitive_choices = (
            PRIMITIVE_TYPES.id("<pad>"),
            PRIMITIVE_TYPES.id(None),
            PRIMITIVE_TYPES.id("line"),
            PRIMITIVE_TYPES.id("<none>"),
        )
        for head, choice in zip(
            self.model.categorical_heads[4:8], primitive_choices
        ):
            _force_head_choice(head, choice)
        result = decode_length_conditioned(
            self.model, _indices(), node_counts=(1,)
        )[0]
        node = result.raw_nodes[0]
        self.assertEqual(node.categorical_ids[4:8], primitive_choices)
        active = {
            index
            for index, applicable in enumerate(
                node.derived_geometry_mask
            )
            if applicable
        }
        self.assertEqual(active, set(range(21, 25)))


@unittest.skipUnless(torch is not None, TORCH_REASON)
class GeometryMaskTests(unittest.TestCase):
    def _mask(self, node_type, primitives=()):
        attributes = [1] * 9
        for index, primitive in enumerate(primitives):
            attributes[4 + index] = PRIMITIVE_TYPES.id(primitive)
        return derive_geometry_mask(
            NODE_TYPES.id(node_type), tuple(attributes)
        )

    def test_node_geometry_masks(self):
        cases = {
            "reference_plane": set(range(0, 9)),
            "profile": set(),
            "axis": set(range(33, 37)),
            "extrude": {37},
            "revolve": {38},
        }
        for node_type, expected in cases.items():
            with self.subTest(node_type=node_type):
                mask = self._mask(node_type)
                self.assertEqual(
                    {index for index, value in enumerate(mask) if value},
                    expected,
                )

    def test_all_supported_primitive_masks(self):
        mask = self._mask(
            "sketch", ("line", "arc", "circle")
        )
        expected = (
            set(range(9, 13))
            | set(range(15, 21))
            | set(range(21, 24))
        )
        self.assertEqual(
            {index for index, value in enumerate(mask) if value},
            expected,
        )
        self.assertEqual(len(mask), GEOMETRY_WIDTH)
