"""Phase B Stage 3 shared-memory decoding tests."""

from __future__ import annotations

from dataclasses import fields, replace
import inspect
import math
import re
from types import SimpleNamespace
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
    import prototype.flat_baseline.autonomous as autonomous
    from prototype.flat_baseline.autonomous import (
        AutonomousDecodingError,
        RAW_PREFIX_FEEDBACK,
        REQUESTED_LENGTH_TERMINATION,
        TEACHER_FORCED_PREFIX_FEEDBACK,
        RawDecodedEdge,
        RawDecodedNode,
        RawDecodedPointer,
        RawDecodedPrediction,
        decode_length_conditioned,
        decode_length_conditioned_from_memory,
        decode_paired_from_batch,
        decode_teacher_forced_from_memory,
    )
    from prototype.flat_baseline.config import FlatBaselineConfig
    from prototype.flat_baseline.losses import flat_mixed_vq_loss
    from prototype.flat_baseline.model import FlatMixedVQModel
    from prototype.model_data.geometry import GEOMETRY_WIDTH
    from prototype.model_data.vocab import (
        BOOLEAN_MODES,
        DIRECTIONS,
        EDGE_TYPES,
        LOOP_ROLES,
        NODE_TYPES,
        OPERATION_TYPES,
        PRIMITIVE_TYPES,
        REFERENCE_PLANES,
    )


TORCH_REASON = "real PyTorch execution is deferred to the Adroit environment"
RAW_FLOAT_REL_TOL = 1e-6
RAW_FLOAT_ABS_TOL = 2e-6


def _config(**changes):
    values = dict(
        model_dim=8,
        num_heads=2,
        feedforward_dim=12,
        encoder_layers=1,
        decoder_layers=1,
        dropout=0.25,
        max_nodes=10,
        max_operations=2,
        latent_tokens=2,
        codebook_size=4,
        codebook_dim=4,
        edge_pair_dim=6,
    )
    values.update(changes)
    return FlatBaselineConfig(**values)


def _batch(templates=("E", "R", "ER")):
    temporary, inputs, target, _ = _batch_with_ids(templates)
    return temporary, inputs, target


def _batch_with_ids(templates):
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
        batch.family_ids,
    )


def _counts(target):
    return tuple(
        int(row.sum().item()) for row in target["node_mask"]
    )


def _encode_lookup(model, inputs):
    with torch.no_grad():
        encoded = model.encode_to_memory(
            inputs["categorical_ids"],
            inputs["geometry"],
            inputs["geometry_mask"],
            inputs["padding_mask"],
        )
        memory = model.memory_from_indices(encoded.vq.indices)
    return encoded.vq.indices, memory


def _rows(mapping, order, batch_size):
    result = {}
    for name, value in mapping.items():
        if (
            torch.is_tensor(value)
            and value.dim() > 0
            and value.size(0) == batch_size
        ):
            result[name] = value[order]
        else:
            result[name] = value
    return result


def _state_snapshot(model):
    return tuple(
        (
            name,
            tuple(value.shape),
            value.dtype,
            value.device,
            value.detach().clone(),
        )
        for name, value in model.state_dict().items()
    )


def _requires_grad_snapshot(model):
    return tuple(
        (name, parameter.requires_grad)
        for name, parameter in model.named_parameters()
    )


def _vq_attribute_snapshot(model):
    return tuple(
        (name, getattr(model.vq, name))
        for name in (
            "num_embeddings",
            "embedding_dim",
            "commitment_cost",
            "decay",
            "epsilon",
        )
    )


def _cuda_rng_snapshot():
    if not torch.cuda.is_available():
        return None
    return tuple(value.clone() for value in torch.cuda.get_rng_state_all())


def _assert_cuda_rng_equal(test, expected):
    if expected is None:
        return
    actual = torch.cuda.get_rng_state_all()
    test.assertEqual(len(actual), len(expected))
    for first, second in zip(expected, actual):
        test.assertTrue(torch.equal(first, second))


def _assert_state_equal(test, before, model):
    after = tuple(model.state_dict().items())
    test.assertEqual(tuple(item[0] for item in before), tuple(
        name for name, _ in after
    ))
    for expected, (name, value) in zip(before, after):
        expected_name, shape, dtype, device, contents = expected
        test.assertEqual(name, expected_name)
        test.assertEqual(tuple(value.shape), shape)
        test.assertEqual(value.dtype, dtype)
        test.assertEqual(value.device, device)
        test.assertTrue(torch.equal(value, contents))


def _assert_output_equal(test, expected, actual):
    test.assertEqual(
        tuple(field.name for field in fields(expected)),
        tuple(field.name for field in fields(actual)),
    )
    for field in fields(expected):
        first = getattr(expected, field.name)
        second = getattr(actual, field.name)
        if torch.is_tensor(first):
            test.assertTrue(torch.equal(first, second), field.name)
        elif isinstance(first, tuple) and first and torch.is_tensor(first[0]):
            test.assertEqual(len(first), len(second))
            for left, right in zip(first, second):
                test.assertTrue(torch.equal(left, right), field.name)
        else:
            test.assertEqual(first, second)


def _assert_raw_prediction_tuples_close(
    test,
    expected,
    actual,
    path="prediction",
):
    test.assertEqual(len(expected), len(actual), path)
    maxima = {
        "normalized_geometry": (0.0, 0.0),
        "edge_presence_logit": (0.0, 0.0),
        "pointer_logit": (0.0, 0.0),
    }
    for index, (first, second) in enumerate(zip(expected, actual)):
        current = _assert_raw_prediction_close(
            test,
            first,
            second,
            "{}[{}]".format(path, index),
        )
        for family, differences in current.items():
            prior = maxima[family]
            maxima[family] = (
                max(prior[0], differences[0]),
                max(prior[1], differences[1]),
            )
    return maxima


def _assert_raw_prediction_close(test, expected, actual, path):
    for name in (
        "latent_indices",
        "node_count",
        "node_count_source",
        "termination_reason",
        "termination_is_learned",
        "prefix_feedback",
        "predicted_operation_node_indices",
        "predicted_operation_count",
        "operation_count_exceeds_limit",
    ):
        test.assertEqual(
            getattr(expected, name),
            getattr(actual, name),
            "{}.{}".format(path, name),
        )
    maxima = {
        "normalized_geometry": (0.0, 0.0),
        "edge_presence_logit": (0.0, 0.0),
        "pointer_logit": (0.0, 0.0),
    }
    test.assertEqual(
        len(expected.raw_nodes), len(actual.raw_nodes),
        "{}.raw_nodes".format(path),
    )
    for index, (first, second) in enumerate(
        zip(expected.raw_nodes, actual.raw_nodes)
    ):
        node_path = "{}.raw_nodes[{}]".format(path, index)
        for name in (
            "position",
            "node_type_id",
            "categorical_ids",
            "derived_geometry_mask",
        ):
            test.assertEqual(
                getattr(first, name),
                getattr(second, name),
                "{}.{}".format(node_path, name),
            )
        test.assertEqual(
            len(first.normalized_geometry),
            len(second.normalized_geometry),
            "{}.normalized_geometry".format(node_path),
        )
        for channel, (left, right) in enumerate(zip(
            first.normalized_geometry, second.normalized_geometry
        )):
            differences = _assert_close_finite_float(
                test,
                left,
                right,
                "{}.normalized_geometry[{}]".format(
                    node_path, channel
                ),
            )
            maxima["normalized_geometry"] = _updated_maximum(
                maxima["normalized_geometry"], differences
            )
    test.assertEqual(
        len(expected.raw_edges), len(actual.raw_edges),
        "{}.raw_edges".format(path),
    )
    for index, (first, second) in enumerate(
        zip(expected.raw_edges, actual.raw_edges)
    ):
        edge_path = "{}.raw_edges[{}]".format(path, index)
        for name in (
            "source_index",
            "target_index",
            "present",
            "edge_type_id",
        ):
            test.assertEqual(
                getattr(first, name),
                getattr(second, name),
                "{}.{}".format(edge_path, name),
            )
        differences = _assert_close_finite_float(
            test,
            first.presence_logit,
            second.presence_logit,
            "{}.presence_logit".format(edge_path),
        )
        maxima["edge_presence_logit"] = _updated_maximum(
            maxima["edge_presence_logit"], differences
        )
    test.assertEqual(
        len(expected.raw_operation_pointers),
        len(actual.raw_operation_pointers),
        "{}.raw_operation_pointers".format(path),
    )
    for index, (first, second) in enumerate(zip(
        expected.raw_operation_pointers,
        actual.raw_operation_pointers,
    )):
        pointer_path = "{}.raw_operation_pointers[{}]".format(
            path, index
        )
        for name in ("query_index", "selected_node_index"):
            test.assertEqual(
                getattr(first, name),
                getattr(second, name),
                "{}.{}".format(pointer_path, name),
            )
        differences = _assert_close_finite_float(
            test,
            first.selected_logit,
            second.selected_logit,
            "{}.selected_logit".format(pointer_path),
        )
        maxima["pointer_logit"] = _updated_maximum(
            maxima["pointer_logit"], differences
        )
    return maxima


def _assert_close_finite_float(test, expected, actual, path):
    test.assertIs(type(expected), float, path)
    test.assertIs(type(actual), float, path)
    test.assertTrue(math.isfinite(expected), path)
    test.assertTrue(math.isfinite(actual), path)
    absolute = abs(expected - actual)
    scale = max(abs(expected), abs(actual))
    relative = 0.0 if absolute == 0.0 else (
        math.inf if scale == 0.0 else absolute / scale
    )
    if not math.isclose(
        expected,
        actual,
        rel_tol=RAW_FLOAT_REL_TOL,
        abs_tol=RAW_FLOAT_ABS_TOL,
    ):
        test.fail(
            "{} differs: {!r} != {!r}; abs={!r}, rel={!r}".format(
                path, expected, actual, absolute, relative
            )
        )
    return absolute, relative


def _updated_maximum(previous, current):
    return max(previous[0], current[0]), max(previous[1], current[1])


def _frozen_base_predictions_from_memory(
    model,
    memory,
    latent_indices,
    node_counts,
    node_count_source="target_canonical_metadata",
):
    """Frozen commit-673ae7b Phase A decoder, independent of Stage 3."""

    return tuple(
        _frozen_base_decode_one(
            model,
            memory[index : index + 1],
            latent_indices[index],
            count,
            node_count_source,
        )
        for index, count in enumerate(node_counts)
    )


def _frozen_geometry_mask(node_type_id, categorical_ids):
    mask = [False] * GEOMETRY_WIDTH
    node_type = NODE_TYPES.tokens[node_type_id]
    if node_type == "reference_plane":
        mask[:9] = [True] * 9
    elif node_type == "sketch":
        widths = {"line": 4, "arc": 6, "circle": 3}
        for slot in range(4):
            primitive = PRIMITIVE_TYPES.tokens[
                categorical_ids[4 + slot]
            ]
            start = 9 + slot * 6
            width = widths.get(primitive, 0)
            mask[start : start + width] = [True] * width
    elif node_type == "axis":
        mask[33:37] = [True] * 4
    elif node_type == "extrude":
        mask[37] = True
    elif node_type == "revolve":
        mask[38] = True
    return tuple(mask)


def _frozen_base_decode_one(
    model,
    memory,
    latent_indices,
    node_count,
    node_count_source,
):
    device = memory.device
    categories = torch.empty((1, 0, 10), dtype=torch.long, device=device)
    geometry = torch.empty(
        (1, 0, GEOMETRY_WIDTH), dtype=memory.dtype, device=device
    )
    geometry_mask = torch.empty(
        (1, 0, GEOMETRY_WIDTH), dtype=torch.bool, device=device
    )
    raw_nodes = []
    for position in range(node_count):
        output = model.decode_prefix(
            memory, categories, geometry, geometry_mask
        )
        node_type = output.node_type_logits[:, -1].argmax(dim=-1)
        attributes = torch.stack(
            tuple(
                logits[:, -1].argmax(dim=-1)
                for logits in output.categorical_logits
            ),
            dim=-1,
        )
        predicted_geometry = output.geometry[:, -1]
        node_type_value = int(node_type.item())
        attribute_values = tuple(
            int(value)
            for value in attributes[0].detach().cpu().tolist()
        )
        mask_values = _frozen_geometry_mask(
            node_type_value, attribute_values
        )
        current_mask = torch.tensor(
            mask_values, dtype=torch.bool, device=device
        ).unsqueeze(0)
        current_categories = torch.cat(
            (node_type.unsqueeze(-1), attributes), dim=-1
        ).unsqueeze(1)
        categories = torch.cat(
            (categories, current_categories), dim=1
        )
        geometry = torch.cat(
            (geometry, predicted_geometry.unsqueeze(1)), dim=1
        )
        geometry_mask = torch.cat(
            (geometry_mask, current_mask.unsqueeze(1)), dim=1
        )
        raw_nodes.append(RawDecodedNode(
            position,
            node_type_value,
            attribute_values,
            tuple(
                float(value)
                for value in predicted_geometry[
                    0
                ].detach().cpu().tolist()
            ),
            mask_values,
        ))
    aligned = model.decode_prefix(
        memory,
        categories[:, :-1],
        geometry[:, :-1],
        geometry_mask[:, :-1],
    )
    return _frozen_prediction_from_aligned(
        model,
        aligned,
        tuple(raw_nodes),
        latent_indices,
        node_count,
        node_count_source,
        RAW_PREFIX_FEEDBACK,
    )


def _frozen_teacher_predictions(
    model,
    memory,
    latent_indices,
    target,
    node_counts,
    node_count_source="target_canonical_metadata",
):
    """Independent committed-internal shifted-prefix prediction oracle."""

    predictions = []
    for index, count in enumerate(node_counts):
        categories = torch.cat(
            (
                target["node_type_ids"][
                    index : index + 1, :count
                ].unsqueeze(-1),
                target["categorical_attributes"][
                    index : index + 1, :count
                ],
            ),
            dim=-1,
        )
        aligned = model.decode_prefix(
            memory[index : index + 1],
            categories[:, :-1],
            target["geometry"][index : index + 1, :count - 1],
            target["geometry_mask"][index : index + 1, :count - 1],
        )
        raw_nodes = []
        for position in range(count):
            node_type = int(
                aligned.node_type_logits[
                    0, position
                ].argmax(dim=-1).item()
            )
            attributes = tuple(
                int(logits[0, position].argmax(dim=-1).item())
                for logits in aligned.categorical_logits
            )
            raw_nodes.append(RawDecodedNode(
                position,
                node_type,
                attributes,
                tuple(
                    float(value)
                    for value in aligned.geometry[
                        0, position
                    ].detach().cpu().tolist()
                ),
                _frozen_geometry_mask(node_type, attributes),
            ))
        predictions.append(_frozen_prediction_from_aligned(
            model,
            aligned,
            tuple(raw_nodes),
            latent_indices[index],
            count,
            node_count_source,
            TEACHER_FORCED_PREFIX_FEEDBACK,
        ))
    return tuple(predictions)


def _frozen_prediction_from_aligned(
    model,
    aligned,
    raw_nodes,
    latent_indices,
    node_count,
    node_count_source,
    prefix_feedback,
):
    node_mask = torch.ones(
        (1, node_count),
        dtype=torch.bool,
        device=aligned.decoded_states.device,
    )
    relations = model.decode_relations(aligned.decoded_states, node_mask)
    raw_edges = []
    for source_index in range(node_count):
        for target_index in range(node_count):
            presence_logit = float(
                relations.edge_presence_logits[
                    0, source_index, target_index
                ].item()
            )
            edge_type = int(
                relations.edge_type_logits[
                    0, source_index, target_index
                ].argmax(dim=-1).item()
            )
            raw_edges.append(RawDecodedEdge(
                source_index,
                target_index,
                presence_logit,
                presence_logit >= 0.0,
                edge_type,
            ))
    operation_ids = {
        NODE_TYPES.id("extrude"),
        NODE_TYPES.id("revolve"),
    }
    operation_nodes = tuple(
        node.position
        for node in raw_nodes
        if node.node_type_id in operation_ids
    )
    pointer_count = min(
        len(operation_nodes), model.config.max_operations
    )
    pointers = []
    for query_index in range(pointer_count):
        logits = relations.operation_pointer_logits[0, query_index]
        selected = int(logits.argmax(dim=-1).item())
        pointers.append(RawDecodedPointer(
            query_index,
            selected,
            float(logits[selected].item()),
        ))
    return RawDecodedPrediction(
        tuple(
            int(value)
            for value in latent_indices.detach().cpu().tolist()
        ),
        node_count,
        node_count_source,
        REQUESTED_LENGTH_TERMINATION,
        False,
        prefix_feedback,
        raw_nodes,
        tuple(raw_edges),
        operation_nodes,
        len(operation_nodes),
        len(operation_nodes) > model.config.max_operations,
        tuple(pointers),
    )


@unittest.skipUnless(torch is not None, TORCH_REASON)
class SharedMemoryDecodingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(913)
        self.model = FlatMixedVQModel(_config())

    def test_public_signatures_and_teacher_latent_provenance(self):
        self.assertEqual(
            tuple(inspect.signature(
                decode_length_conditioned_from_memory
            ).parameters),
            (
                "model",
                "memory",
                "latent_indices",
                "node_counts",
                "node_count_source",
                "prefix_feedback",
            ),
        )
        self.assertIn(
            "latent_indices",
            inspect.signature(
                decode_teacher_forced_from_memory
            ).parameters,
        )
        self.assertNotIn(
            "target",
            inspect.signature(
                decode_length_conditioned_from_memory
            ).parameters,
        )

    def test_predicted_history_is_exactly_phase_a_equivalent(self):
        all_templates = ("E", "R", "EE", "ER", "RE", "RR")
        cases = tuple((template,) for template in all_templates) + (
            all_templates,
        )
        for templates in cases:
            temporary, inputs, target = _batch(templates)
            self.addCleanup(temporary.cleanup)
            self.model.eval()
            indices, memory = _encode_lookup(self.model, inputs)
            counts = _counts(target)
            with torch.no_grad():
                expected = _frozen_base_predictions_from_memory(
                    self.model, memory, indices, counts
                )
            public = decode_length_conditioned(
                self.model,
                indices,
                node_counts=counts,
                node_count_source="target_canonical_metadata",
            )
            from_memory = decode_length_conditioned_from_memory(
                self.model,
                memory,
                indices,
                node_counts=counts,
                node_count_source="target_canonical_metadata",
                prefix_feedback=RAW_PREFIX_FEEDBACK,
            )
            self.assertEqual(public, expected)
            self.assertEqual(from_memory, expected)

    def test_teacher_forced_matches_committed_shifted_prefix_internals(self):
        temporary, inputs, target = _batch(("E", "R", "ER"))
        self.addCleanup(temporary.cleanup)
        self.model.eval()
        indices, memory = _encode_lookup(self.model, inputs)
        actual = decode_teacher_forced_from_memory(
            self.model,
            memory,
            indices,
            target,
            node_count_source="target_canonical_metadata",
        )
        with torch.no_grad():
            expected = _frozen_teacher_predictions(
                self.model,
                memory,
                indices,
                target,
                _counts(target),
            )
        self.assertEqual(actual, expected)

    def test_paired_path_uses_one_encode_one_lookup_and_same_memory(self):
        temporary, inputs, target = _batch(("E", "R"))
        self.addCleanup(temporary.cleanup)
        original_encode = self.model.encode_to_memory
        original_lookup = self.model.memory_from_indices
        original_encoder = self.model.encoder.forward
        original_vq = self.model.vq.forward
        original_prefix = self.model.decode_prefix
        original_relations = self.model.decode_relations
        original_embedding = torch.nn.functional.embedding
        original_teacher = autonomous.decode_teacher_forced_from_memory
        original_predicted = autonomous.decode_length_conditioned_from_memory
        inside_encode = [False]
        inside_lookup = [False]
        encoded_indices = []
        lookup_memory = []
        lookup_before = []
        poison_memory = []
        helper_entries = []
        decoded_state_ids = set()
        prefix_storage_pointers = []

        def encode_spy(*args):
            self.assertEqual(len(args), 4)
            for actual, name in zip(args, (
                "categorical_ids",
                "geometry",
                "geometry_mask",
                "padding_mask",
            )):
                self.assertIs(actual, inputs[name])
            self.assertFalse(inside_encode[0])
            inside_encode[0] = True
            try:
                encoded = original_encode(*args)
            finally:
                inside_encode[0] = False
            encoded_indices.append(encoded.vq.indices)
            poison = torch.full_like(encoded.memory, 12345.0)
            poison_memory.append(poison)
            encoded.memory = poison
            return encoded

        def lookup_spy(indices):
            self.assertEqual(len(encoded_indices), 1)
            self.assertIs(indices, encoded_indices[0])
            inside_lookup[0] = True
            try:
                lookup = original_lookup(indices)
            finally:
                inside_lookup[0] = False
            lookup_memory.append(lookup)
            lookup_before.append(lookup.clone())
            return lookup

        def encoder_spy(*args, **kwargs):
            if not inside_encode[0]:
                raise AssertionError("decoder re-entered model.encoder")
            return original_encoder(*args, **kwargs)

        def vq_spy(*args, **kwargs):
            if not inside_encode[0]:
                raise AssertionError("decoder re-entered model.vq")
            return original_vq(*args, **kwargs)

        def embedding_spy(indices, weight, *args, **kwargs):
            codebook_lookup = (
                weight.data_ptr() == self.model.vq.embedding.data_ptr()
            )
            if (
                codebook_lookup
                and not inside_encode[0]
                and not inside_lookup[0]
            ):
                raise AssertionError(
                    "decoder performed a direct codebook lookup"
                )
            return original_embedding(indices, weight, *args, **kwargs)

        def prefix_spy(memory, *args, **kwargs):
            self.assertEqual(len(lookup_memory), 1)
            lookup = lookup_memory[0]
            storage_pointer = memory.storage().data_ptr()
            self.assertEqual(
                storage_pointer, lookup.storage().data_ptr()
            )
            prefix_storage_pointers.append(storage_pointer)
            output = original_prefix(memory, *args, **kwargs)
            decoded_state_ids.add(id(output.decoded_states))
            return output

        def relation_spy(states, *args, **kwargs):
            self.assertIn(id(states), decoded_state_ids)
            return original_relations(states, *args, **kwargs)

        def teacher_spy(model, memory, *args, **kwargs):
            self.assertIs(memory, lookup_memory[0])
            helper_entries.append((
                "teacher",
                id(memory),
                memory.storage().data_ptr(),
            ))
            self.assertFalse(model.training)
            self.assertFalse(torch.is_grad_enabled())
            return original_teacher(model, memory, *args, **kwargs)

        def predicted_spy(model, memory, *args, **kwargs):
            self.assertIs(memory, lookup_memory[0])
            helper_entries.append((
                "predicted",
                id(memory),
                memory.storage().data_ptr(),
            ))
            self.assertFalse(model.training)
            self.assertFalse(torch.is_grad_enabled())
            return original_predicted(model, memory, *args, **kwargs)

        self.model.train()
        with mock.patch.object(
            self.model, "encode_to_memory", side_effect=encode_spy
        ) as encode_call, mock.patch.object(
            self.model, "memory_from_indices", side_effect=lookup_spy
        ) as lookup_call, mock.patch.object(
            self.model.encoder, "forward", side_effect=encoder_spy
        ), mock.patch.object(
            self.model.vq, "forward", side_effect=vq_spy
        ), mock.patch.object(
            self.model, "decode_prefix", side_effect=prefix_spy
        ), mock.patch.object(
            self.model, "decode_relations", side_effect=relation_spy
        ), mock.patch.object(
            torch.nn.functional, "embedding", side_effect=embedding_spy
        ), mock.patch.object(
            autonomous,
            "decode_teacher_forced_from_memory",
            side_effect=teacher_spy,
        ), mock.patch.object(
            autonomous,
            "decode_length_conditioned_from_memory",
            side_effect=predicted_spy,
        ):
            paired = decode_paired_from_batch(
                self.model,
                (inputs, target),
                node_counts=_counts(target),
                node_count_source="target_canonical_metadata",
            )
        self.assertTrue(self.model.training)
        self.assertEqual(encode_call.call_count, 1)
        self.assertEqual(lookup_call.call_count, 1)
        self.assertEqual(len(encoded_indices), 1)
        self.assertEqual(len(lookup_memory), 1)
        lookup = lookup_memory[0]
        self.assertEqual(
            tuple(helper_entries),
            (
                ("teacher", id(lookup), lookup.storage().data_ptr()),
                ("predicted", id(lookup), lookup.storage().data_ptr()),
            ),
        )
        self.assertTrue(prefix_storage_pointers)
        self.assertEqual(
            set(prefix_storage_pointers),
            {lookup.storage().data_ptr()},
        )
        self.assertTrue(torch.equal(lookup_before[0], lookup))
        self.assertFalse(torch.equal(poison_memory[0], lookup))
        self.assertEqual(
            paired.latent_indices,
            tuple(item.latent_indices for item in paired.teacher_forced),
        )
        self.assertEqual(
            paired.latent_indices,
            tuple(item.latent_indices for item in paired.predicted_history),
        )

    def test_authoritative_input_and_target_masks_must_match_exactly(self):
        temporary, inputs, target = _batch(("E", "R", "ER"))
        self.addCleanup(temporary.cleanup)
        counts = _counts(target)
        valid = decode_paired_from_batch(
            self.model,
            (inputs, target),
            node_counts=counts,
            node_count_source="target_canonical_metadata",
        )
        self.assertEqual(
            tuple(item.node_count for item in valid.predicted_history),
            counts,
        )
        active = dict(target)
        active_mask = target["node_mask"].clone()
        active_mask[0, 0] = False
        active["node_mask"] = active_mask
        with self.assertRaisesRegex(ValueError, "node_mask disagree"):
            decode_paired_from_batch(
                self.model,
                (inputs, active),
                node_counts=counts,
                node_count_source="source",
            )

        padding_position = (
            (~target["node_mask"]).nonzero(as_tuple=False)[0]
        )
        padded = dict(target)
        padded_mask = target["node_mask"].clone()
        padded_mask[
            int(padding_position[0]), int(padding_position[1])
        ] = True
        padded["node_mask"] = padded_mask
        with self.assertRaisesRegex(ValueError, "node_mask disagree"):
            decode_paired_from_batch(
                self.model,
                (inputs, padded),
                node_counts=counts,
                node_count_source="source",
            )

        shaped = dict(target)
        shaped["node_mask"] = target["node_mask"][:, :-1]
        with self.assertRaisesRegex(ValueError, "shapes disagree"):
            decode_paired_from_batch(
                self.model,
                (inputs, shaped),
                node_counts=counts,
                node_count_source="source",
            )
        typed = dict(target)
        typed["node_mask"] = target["node_mask"].to(torch.uint8)
        with self.assertRaisesRegex(TypeError, "torch.bool"):
            decode_paired_from_batch(
                self.model,
                (inputs, typed),
                node_counts=counts,
                node_count_source="source",
            )
        if torch.cuda.is_available():
            placed = dict(target)
            placed["node_mask"] = target["node_mask"].to(
                device=torch.device("cuda")
            )
            with self.assertRaisesRegex(ValueError, "devices disagree"):
                decode_paired_from_batch(
                    self.model,
                    (inputs, placed),
                    node_counts=counts,
                    node_count_source="source",
                )

    def test_paired_preserves_full_state_mode_rng_and_is_deterministic(self):
        temporary, inputs, target = _batch(("E", "R", "ER"))
        self.addCleanup(temporary.cleanup)
        for initial_training in (False, True):
            self.model.train(initial_training)
            state = _state_snapshot(self.model)
            requires_grad = _requires_grad_snapshot(self.model)
            vq_attributes = _vq_attribute_snapshot(self.model)
            rng = torch.get_rng_state().clone()
            cuda_rng = _cuda_rng_snapshot()
            first = decode_paired_from_batch(
                self.model,
                (inputs, target),
                node_counts=_counts(target),
                node_count_source="target_canonical_metadata",
            )
            self.assertEqual(self.model.training, initial_training)
            _assert_state_equal(self, state, self.model)
            self.assertEqual(
                _requires_grad_snapshot(self.model), requires_grad
            )
            self.assertEqual(
                _vq_attribute_snapshot(self.model), vq_attributes
            )
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            _assert_cuda_rng_equal(self, cuda_rng)
            second = decode_paired_from_batch(
                self.model,
                (inputs, target),
                node_counts=_counts(target),
                node_count_source="target_canonical_metadata",
            )
            self.assertEqual(first, second)
            _assert_state_equal(self, state, self.model)
            self.assertEqual(
                _requires_grad_snapshot(self.model), requires_grad
            )
            self.assertEqual(
                _vq_attribute_snapshot(self.model), vq_attributes
            )
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            _assert_cuda_rng_equal(self, cuda_rng)

    def test_mode_grad_and_state_restore_at_every_failure_boundary(self):
        temporary, inputs, target = _batch(("E", "R"))
        self.addCleanup(temporary.cleanup)
        boundaries = (
            ("encode", self.model, "encode_to_memory"),
            ("lookup", self.model, "memory_from_indices"),
            (
                "teacher",
                autonomous,
                "decode_teacher_forced_from_memory",
            ),
            (
                "predicted",
                autonomous,
                "decode_length_conditioned_from_memory",
            ),
        )
        for initial_training in (False, True):
            for gradient_enabled in (False, True):
                gradient_context = (
                    torch.enable_grad()
                    if gradient_enabled else torch.no_grad()
                )
                with gradient_context:
                    for label, owner, attribute in boundaries:
                        with self.subTest(
                            initial_training=initial_training,
                            gradient_enabled=gradient_enabled,
                            boundary=label,
                        ):
                            self.model.train(initial_training)
                            state = _state_snapshot(self.model)
                            flags = _requires_grad_snapshot(self.model)
                            cpu_rng = torch.get_rng_state().clone()
                            cuda_rng = _cuda_rng_snapshot()
                            error = RuntimeError(
                                "forced {} failure".format(label)
                            )
                            with mock.patch.object(
                                owner, attribute, side_effect=error
                            ):
                                with self.assertRaises(RuntimeError) as raised:
                                    decode_paired_from_batch(
                                        self.model,
                                        (inputs, target),
                                        node_counts=_counts(target),
                                        node_count_source="source",
                                    )
                            self.assertIs(raised.exception, error)
                            self.assertEqual(
                                self.model.training, initial_training
                            )
                            self.assertEqual(
                                torch.is_grad_enabled(), gradient_enabled
                            )
                            self.assertEqual(
                                _requires_grad_snapshot(self.model), flags
                            )
                            _assert_state_equal(self, state, self.model)
                            self.assertTrue(torch.equal(
                                cpu_rng, torch.get_rng_state()
                            ))
                            _assert_cuda_rng_equal(self, cuda_rng)

    def test_training_forward_and_strict_state_dict_are_unchanged(self):
        temporary, inputs, target = _batch(("E", "R"))
        self.addCleanup(temporary.cleanup)
        config = _config(dropout=0.0)
        torch.manual_seed(177)
        reference = FlatMixedVQModel(config)
        initial = {
            name: value.detach().clone()
            for name, value in reference.state_dict().items()
        }
        staged = FlatMixedVQModel(config)
        load_result = staged.load_state_dict(initial, strict=True)
        self.assertEqual(load_result.missing_keys, [])
        self.assertEqual(load_result.unexpected_keys, [])
        self.assertEqual(
            tuple(initial), tuple(staged.state_dict())
        )
        self.assertEqual(
            tuple(
                (name, tuple(value.shape))
                for name, value in initial.items()
            ),
            tuple(
                (name, tuple(value.shape))
                for name, value in staged.state_dict().items()
            ),
        )
        reference.train()
        staged.train()
        rng = torch.get_rng_state().clone()
        torch.set_rng_state(rng)
        reference_output = reference(target=target, **inputs)
        decode_paired_from_batch(
            staged,
            (inputs, target),
            node_counts=_counts(target),
            node_count_source="target_canonical_metadata",
        )
        self.assertTrue(staged.training)
        torch.set_rng_state(rng)
        staged_output = staged(target=target, **inputs)
        _assert_output_equal(self, reference_output, staged_output)
        reference_loss = flat_mixed_vq_loss(
            reference_output, target, config
        )
        staged_loss = flat_mixed_vq_loss(
            staged_output, target, config
        )
        for name, expected in reference_loss.as_dict().items():
            self.assertTrue(torch.equal(
                expected, staged_loss.as_dict()[name]
            ), name)
        self.assertEqual(
            tuple(reference_loss.per_example),
            tuple(staged_loss.per_example),
        )
        for name, expected in reference_loss.per_example.items():
            self.assertTrue(torch.equal(
                expected, staged_loss.per_example[name]
            ), name)
        reference_loss.total.backward()
        staged_loss.total.backward()
        reference_gradients = dict(reference.named_parameters())
        staged_gradients = dict(staged.named_parameters())
        self.assertEqual(
            tuple(reference_gradients), tuple(staged_gradients)
        )
        for name, parameter in reference_gradients.items():
            actual = staged_gradients[name]
            if parameter.grad is None:
                self.assertIsNone(actual.grad, name)
            else:
                self.assertTrue(torch.equal(
                    parameter.grad, actual.grad
                ), name)
        _assert_state_equal(self, _state_snapshot(reference), staged)
        self.assertEqual(
            _requires_grad_snapshot(reference),
            _requires_grad_snapshot(staged),
        )

    def test_batch_composition_split_and_permutation_invariance(self):
        templates = ("E", "R", "EE", "ER", "RE", "RR")
        temporary, inputs, target, family_ids = _batch_with_ids(templates)
        self.addCleanup(temporary.cleanup)
        self.model.eval()
        batch_size = inputs["categorical_ids"].size(0)
        full = decode_paired_from_batch(
            self.model,
            (inputs, target),
            node_counts=_counts(target),
            node_count_source="target_canonical_metadata",
        )
        observed = {
            "normalized_geometry": (0.0, 0.0),
            "edge_presence_logit": (0.0, 0.0),
            "pointer_logit": (0.0, 0.0),
        }

        def compare(expected, actual, path):
            current = _assert_raw_prediction_tuples_close(
                self, expected, actual, path
            )
            for family, differences in current.items():
                observed[family] = _updated_maximum(
                    observed[family], differences
                )

        singles = []
        for index in range(batch_size):
            order = torch.tensor((index,), dtype=torch.long)
            single_inputs = _rows(inputs, order, batch_size)
            single_target = _rows(target, order, batch_size)
            singles.append(decode_paired_from_batch(
                self.model,
                (single_inputs, single_target),
                node_counts=_counts(single_target),
                node_count_source="target_canonical_metadata",
            ))
        compare(
            full.predicted_history,
            tuple(item.predicted_history[0] for item in singles),
            "singleton.predicted",
        )
        compare(
            full.teacher_forced,
            tuple(item.teacher_forced[0] for item in singles),
            "singleton.teacher",
        )
        order = torch.tensor((5, 2, 0, 4, 1, 3), dtype=torch.long)
        permuted_inputs = _rows(inputs, order, batch_size)
        permuted_target = _rows(target, order, batch_size)
        permuted = decode_paired_from_batch(
            self.model,
            (permuted_inputs, permuted_target),
            node_counts=_counts(permuted_target),
            node_count_source="target_canonical_metadata",
        )
        inverse = torch.argsort(order).tolist()
        compare(
            full.predicted_history,
            tuple(permuted.predicted_history[index] for index in inverse),
            "permutation.predicted",
        )
        compare(
            full.teacher_forced,
            tuple(permuted.teacher_forced[index] for index in inverse),
            "permutation.teacher",
        )
        self.assertEqual(
            tuple(permuted.latent_indices[index] for index in inverse),
            full.latent_indices,
        )
        split_predictions = []
        split_teacher = []
        for values in ((0, 1, 2), (3, 4, 5)):
            split_order = torch.tensor(values, dtype=torch.long)
            split_inputs = _rows(inputs, split_order, batch_size)
            split_target = _rows(target, split_order, batch_size)
            split = decode_paired_from_batch(
                self.model,
                (split_inputs, split_target),
                node_counts=_counts(split_target),
                node_count_source="target_canonical_metadata",
            )
            split_predictions.extend(split.predicted_history)
            split_teacher.extend(split.teacher_forced)
        compare(
            full.predicted_history,
            tuple(split_predictions),
            "split.predicted",
        )
        compare(
            full.teacher_forced,
            tuple(split_teacher),
            "split.teacher",
        )

        equal_length_indices = tuple(
            index for index, count in enumerate(_counts(target))
            if count == 8
        )
        self.assertEqual(len(equal_length_indices), 2)
        equal_order = torch.tensor(
            equal_length_indices, dtype=torch.long
        )
        equal_inputs = _rows(inputs, equal_order, batch_size)
        equal_target = _rows(target, equal_order, batch_size)
        equal = decode_paired_from_batch(
            self.model,
            (equal_inputs, equal_target),
            node_counts=_counts(equal_target),
            node_count_source="target_canonical_metadata",
        )
        compare(
            tuple(
                full.predicted_history[index]
                for index in equal_length_indices
            ),
            equal.predicted_history,
            "equal_length.predicted",
        )
        compare(
            tuple(
                full.teacher_forced[index]
                for index in equal_length_indices
            ),
            equal.teacher_forced,
            "equal_length.teacher",
        )
        self.assertEqual(len(set(family_ids)), len(templates))
        for absolute, relative in observed.values():
            self.assertTrue(math.isfinite(absolute))
            self.assertTrue(math.isfinite(relative))

    def test_focal_sample_is_invariant_to_shorter_and_longer_neighbors(self):
        def decoded_by_family(templates):
            temporary, inputs, target, family_ids = _batch_with_ids(
                templates
            )
            self.addCleanup(temporary.cleanup)
            decoded = decode_paired_from_batch(
                self.model,
                (inputs, target),
                node_counts=_counts(target),
                node_count_source="target_canonical_metadata",
            )
            return {
                family_id: (
                    decoded.predicted_history[index],
                    decoded.teacher_forced[index],
                )
                for index, family_id in enumerate(family_ids)
            }

        self.model.eval()
        focal = decoded_by_family(("ER",))
        focal_id = next(iter(focal))
        with_shorter = decoded_by_family(("ER", "E"))
        with_changed_shorter = decoded_by_family(("ER", "R"))
        with_longer = decoded_by_family(("ER", "RR"))
        for label, neighbor in (
            ("shorter", with_shorter),
            ("changed_shorter", with_changed_shorter),
            ("longer", with_longer),
        ):
            _assert_raw_prediction_tuples_close(
                self,
                (focal[focal_id][0],),
                (neighbor[focal_id][0],),
                "neighbor.{}.predicted".format(label),
            )
            _assert_raw_prediction_tuples_close(
                self,
                (focal[focal_id][1],),
                (neighbor[focal_id][1],),
                "neighbor.{}.teacher".format(label),
            )

    def test_predicted_history_has_no_target_leakage(self):
        temporary, inputs, target = _batch(("E", "R"))
        self.addCleanup(temporary.cleanup)
        self.model.eval()
        indices, memory = _encode_lookup(self.model, inputs)
        counts = _counts(target)
        direct = decode_length_conditioned_from_memory(
            self.model,
            memory,
            indices,
            node_counts=counts,
            node_count_source="target_canonical_metadata",
            prefix_feedback=RAW_PREFIX_FEEDBACK,
        )
        teacher = decode_teacher_forced_from_memory(
            self.model,
            memory,
            indices,
            target,
            node_count_source="target_canonical_metadata",
        )
        mutations = []

        node_types = target["node_type_ids"].clone()
        node_types[0, 0] = (
            int(node_types[0, 0].item()) + 1
        ) % len(NODE_TYPES.tokens)
        mutations.append(("node_types", "node_type_ids", node_types, True))

        attribute_vocabularies = (
            OPERATION_TYPES,
            BOOLEAN_MODES,
            DIRECTIONS,
            REFERENCE_PLANES,
            PRIMITIVE_TYPES,
            PRIMITIVE_TYPES,
            PRIMITIVE_TYPES,
            PRIMITIVE_TYPES,
            LOOP_ROLES,
        )
        for column, vocabulary in enumerate(attribute_vocabularies):
            attributes = target["categorical_attributes"].clone()
            attributes[0, 0, column] = (
                int(attributes[0, 0, column].item()) + 1
            ) % len(vocabulary.tokens)
            mutations.append((
                "categorical_{}".format(column),
                "categorical_attributes",
                attributes,
                True,
            ))

        geometry = target["geometry"].clone()
        applicable = target["geometry_mask"][0, 0]
        geometry[0, 0, applicable] = (
            torch.where(
                geometry[0, 0, applicable] == 0.0,
                torch.full_like(geometry[0, 0, applicable], 0.5),
                -geometry[0, 0, applicable],
            )
        )
        mutations.append(("geometry", "geometry", geometry, True))

        geometry_mask = target["geometry_mask"].clone()
        geometry_mask[0, 0, 0] = ~geometry_mask[0, 0, 0]
        mutations.append((
            "geometry_mask",
            "geometry_mask",
            geometry_mask,
            True,
        ))

        booleans = target["boolean_mode_targets"].clone()
        booleans[0, 0] = (int(booleans[0, 0].item()) + 1) % len(
            BOOLEAN_MODES.tokens
        )
        mutations.append((
            "boolean_modes",
            "boolean_mode_targets",
            booleans,
            False,
        ))

        edge_index = target["edge_index"].clone()
        edge_index = edge_index.flip(1)
        mutations.append(("edges", "edge_index", edge_index, False))
        edge_types = target["edge_type_ids"].clone()
        if edge_types.numel():
            edge_types[0] = (
                int(edge_types[0].item()) + 1
            ) % len(EDGE_TYPES.tokens)
        mutations.append(("edge_types", "edge_type_ids", edge_types, False))

        operations = target["operation_sequence"].clone()
        operations[0, 0] = (
            int(operations[0, 0].item()) + 1
        ) % counts[0]
        mutations.append((
            "operation_sequence",
            "operation_sequence",
            operations,
            False,
        ))
        operation_mask = target["operation_mask"].clone()
        operation_mask[0, 0] = ~operation_mask[0, 0]
        mutations.append((
            "operation_mask",
            "operation_mask",
            operation_mask,
            False,
        ))

        inactive = (~target["node_mask"]).nonzero(as_tuple=False)[0]
        inactive_row = int(inactive[0])
        inactive_column = int(inactive[1])
        padded_nodes = target["node_type_ids"].clone()
        padded_nodes[inactive_row, inactive_column] = NODE_TYPES.id(
            "revolve"
        )
        mutations.append((
            "padded_node",
            "node_type_ids",
            padded_nodes,
            False,
        ))
        padded_attributes = target["categorical_attributes"].clone()
        padded_attributes[inactive_row, inactive_column] = 1
        mutations.append((
            "padded_categories",
            "categorical_attributes",
            padded_attributes,
            False,
        ))
        padded_geometry = target["geometry"].clone()
        padded_geometry[inactive_row, inactive_column] = 0.75
        mutations.append((
            "padded_geometry",
            "geometry",
            padded_geometry,
            False,
        ))
        padded_geometry_mask = target["geometry_mask"].clone()
        padded_geometry_mask[inactive_row, inactive_column] = True
        mutations.append((
            "padded_geometry_mask",
            "geometry_mask",
            padded_geometry_mask,
            False,
        ))

        for name, field, value, teacher_must_change in mutations:
            with self.subTest(target_family=name):
                poisoned = dict(target)
                poisoned[field] = value
                predicted = decode_length_conditioned_from_memory(
                    self.model,
                    memory,
                    indices,
                    node_counts=counts,
                    node_count_source="target_canonical_metadata",
                    prefix_feedback=RAW_PREFIX_FEEDBACK,
                )
                self.assertEqual(predicted, direct)
                poisoned_teacher = decode_teacher_forced_from_memory(
                    self.model,
                    memory,
                    indices,
                    poisoned,
                    node_count_source="target_canonical_metadata",
                )
                if teacher_must_change:
                    self.assertNotEqual(poisoned_teacher, teacher)
                else:
                    self.assertEqual(poisoned_teacher, teacher)
        self.assertNotIn(
            "target",
            inspect.signature(
                decode_length_conditioned_from_memory
            ).parameters,
        )

    def test_structured_batch_comparator_rejects_semantic_changes(self):
        attributes = (1,) * 9
        geometry = (0.0,) * GEOMETRY_WIDTH
        mask = (False,) * GEOMETRY_WIDTH
        nodes = (
            RawDecodedNode(0, 2, attributes, geometry, mask),
            RawDecodedNode(1, 3, attributes, geometry, mask),
        )
        edges = (
            RawDecodedEdge(0, 0, 0.25, True, 2),
            RawDecodedEdge(0, 1, -0.25, False, 1),
        )
        pointers = (
            RawDecodedPointer(0, 0, 0.5),
            RawDecodedPointer(1, 1, 0.25),
        )
        baseline = RawDecodedPrediction(
            (0, 1),
            2,
            "source",
            REQUESTED_LENGTH_TERMINATION,
            False,
            RAW_PREFIX_FEEDBACK,
            nodes,
            edges,
            (0, 1),
            2,
            False,
            pointers,
        )

        changed_attributes = list(attributes)
        changed_attributes[3] = 2
        categorical_node = replace(
            nodes[0], categorical_ids=tuple(changed_attributes)
        )
        changed_mask = list(mask)
        changed_mask[17] = True
        mask_node = replace(
            nodes[0], derived_geometry_mask=tuple(changed_mask)
        )
        changed_geometry = list(geometry)
        changed_geometry[17] = RAW_FLOAT_ABS_TOL * 2.0
        geometry_node = replace(
            nodes[0], normalized_geometry=tuple(changed_geometry)
        )
        mutations = (
            (
                replace(
                    baseline,
                    raw_nodes=(categorical_node, nodes[1]),
                ),
                "prediction[0].raw_nodes[0].categorical_ids",
            ),
            (
                replace(baseline, raw_nodes=(mask_node, nodes[1])),
                "prediction[0].raw_nodes[0].derived_geometry_mask",
            ),
            (
                replace(
                    baseline,
                    raw_edges=(
                        replace(edges[0], present=False),
                        edges[1],
                    ),
                ),
                "prediction[0].raw_edges[0].present",
            ),
            (
                replace(
                    baseline,
                    raw_operation_pointers=(
                        replace(pointers[0], selected_node_index=1),
                        pointers[1],
                    ),
                ),
                "prediction[0].raw_operation_pointers[0]"
                ".selected_node_index",
            ),
            (
                replace(baseline, raw_edges=tuple(reversed(edges))),
                "prediction[0].raw_edges[0].target_index",
            ),
            (
                replace(
                    baseline,
                    raw_operation_pointers=tuple(reversed(pointers)),
                ),
                "prediction[0].raw_operation_pointers[0].query_index",
            ),
            (
                replace(
                    baseline,
                    raw_nodes=(geometry_node, nodes[1]),
                ),
                "prediction[0].raw_nodes[0]"
                ".normalized_geometry[17]",
            ),
        )
        for changed, expected_path in mutations:
            with self.subTest(expected_path=expected_path):
                with self.assertRaisesRegex(
                    AssertionError, re.escape(expected_path)
                ):
                    _assert_raw_prediction_tuples_close(
                        self, (baseline,), (changed,)
                    )

    def test_input_validation_is_deterministic_and_does_not_repair(self):
        temporary, inputs, target = _batch(("E", "R"))
        self.addCleanup(temporary.cleanup)
        self.model.eval()
        indices, memory = _encode_lookup(self.model, inputs)
        counts = _counts(target)
        invalid_calls = (
            (
                TypeError,
                lambda: decode_length_conditioned_from_memory(
                    self.model,
                    memory.to(torch.int64),
                    indices,
                    node_counts=counts,
                    node_count_source="source",
                    prefix_feedback=RAW_PREFIX_FEEDBACK,
                ),
            ),
            (
                ValueError,
                lambda: decode_length_conditioned_from_memory(
                    self.model,
                    memory[:, :, :-1],
                    indices,
                    node_counts=counts,
                    node_count_source="source",
                    prefix_feedback=RAW_PREFIX_FEEDBACK,
                ),
            ),
            (
                TypeError,
                lambda: decode_length_conditioned_from_memory(
                    self.model,
                    memory,
                    indices.to(torch.bool),
                    node_counts=counts,
                    node_count_source="source",
                    prefix_feedback=RAW_PREFIX_FEEDBACK,
                ),
            ),
            (
                ValueError,
                lambda: decode_length_conditioned_from_memory(
                    self.model,
                    memory,
                    indices[:1],
                    node_counts=counts,
                    node_count_source="source",
                    prefix_feedback=RAW_PREFIX_FEEDBACK,
                ),
            ),
            (
                AutonomousDecodingError,
                lambda: decode_length_conditioned_from_memory(
                    self.model,
                    memory,
                    indices,
                    node_counts=(True, counts[1]),
                    node_count_source="source",
                    prefix_feedback=RAW_PREFIX_FEEDBACK,
                ),
            ),
            (
                AutonomousDecodingError,
                lambda: decode_length_conditioned_from_memory(
                    self.model,
                    memory,
                    indices,
                    node_counts=counts,
                    node_count_source="",
                    prefix_feedback=RAW_PREFIX_FEEDBACK,
                ),
            ),
            (
                AutonomousDecodingError,
                lambda: decode_length_conditioned_from_memory(
                    self.model,
                    memory,
                    indices,
                    node_counts=counts,
                    node_count_source="source",
                    prefix_feedback="other",
                ),
            ),
        )
        for error, call in invalid_calls:
            with self.subTest(error=error):
                with self.assertRaises(error):
                    call()

        shortened = dict(target)
        shortened["categorical_attributes"] = target[
            "categorical_attributes"
        ][:, :-1]
        with self.assertRaisesRegex(ValueError, "misaligned"):
            decode_teacher_forced_from_memory(
                self.model,
                memory,
                indices,
                shortened,
                node_count_source="source",
            )
        with self.assertRaisesRegex(
            AutonomousDecodingError, "target_node_count_mismatch"
        ):
            decode_paired_from_batch(
                self.model,
                (inputs, target),
                node_counts=(counts[0] - 1, counts[1]),
                node_count_source="source",
            )


class ComparatorContractTests(unittest.TestCase):
    def test_comparator_rejects_discrete_order_and_large_float_changes(self):
        def changed(record, **updates):
            values = dict(vars(record))
            values.update(updates)
            return SimpleNamespace(**values)

        attributes = (1,) * 9
        geometry = (0.0,) * 39
        mask = (False,) * 39
        nodes = (
            SimpleNamespace(
                position=0,
                node_type_id=2,
                categorical_ids=attributes,
                normalized_geometry=geometry,
                derived_geometry_mask=mask,
            ),
            SimpleNamespace(
                position=1,
                node_type_id=3,
                categorical_ids=attributes,
                normalized_geometry=geometry,
                derived_geometry_mask=mask,
            ),
        )
        edges = (
            SimpleNamespace(
                source_index=0,
                target_index=0,
                presence_logit=0.25,
                present=True,
                edge_type_id=2,
            ),
            SimpleNamespace(
                source_index=0,
                target_index=1,
                presence_logit=-0.25,
                present=False,
                edge_type_id=1,
            ),
        )
        pointers = (
            SimpleNamespace(
                query_index=0,
                selected_node_index=0,
                selected_logit=0.5,
            ),
            SimpleNamespace(
                query_index=1,
                selected_node_index=1,
                selected_logit=0.25,
            ),
        )
        baseline = SimpleNamespace(
            latent_indices=(0, 1),
            node_count=2,
            node_count_source="source",
            termination_reason="requested_node_count_reached",
            termination_is_learned=False,
            prefix_feedback="raw_argmax_with_derived_geometry_mask",
            raw_nodes=nodes,
            raw_edges=edges,
            predicted_operation_node_indices=(0, 1),
            predicted_operation_count=2,
            operation_count_exceeds_limit=False,
            raw_operation_pointers=pointers,
        )
        categories = list(attributes)
        categories[3] = 2
        changed_mask = list(mask)
        changed_mask[17] = True
        changed_geometry = list(geometry)
        changed_geometry[17] = RAW_FLOAT_ABS_TOL * 2.0
        cases = (
            (
                changed(
                    baseline,
                    raw_nodes=(
                        changed(nodes[0], categorical_ids=tuple(categories)),
                        nodes[1],
                    ),
                ),
                "prediction[0].raw_nodes[0].categorical_ids",
            ),
            (
                changed(
                    baseline,
                    raw_nodes=(
                        changed(
                            nodes[0],
                            derived_geometry_mask=tuple(changed_mask),
                        ),
                        nodes[1],
                    ),
                ),
                "prediction[0].raw_nodes[0].derived_geometry_mask",
            ),
            (
                changed(
                    baseline,
                    raw_edges=(
                        changed(edges[0], present=False),
                        edges[1],
                    ),
                ),
                "prediction[0].raw_edges[0].present",
            ),
            (
                changed(
                    baseline,
                    raw_operation_pointers=(
                        changed(pointers[0], selected_node_index=1),
                        pointers[1],
                    ),
                ),
                "prediction[0].raw_operation_pointers[0]"
                ".selected_node_index",
            ),
            (
                changed(baseline, raw_edges=tuple(reversed(edges))),
                "prediction[0].raw_edges[0].target_index",
            ),
            (
                changed(
                    baseline,
                    raw_operation_pointers=tuple(reversed(pointers)),
                ),
                "prediction[0].raw_operation_pointers[0].query_index",
            ),
            (
                changed(
                    baseline,
                    raw_nodes=(
                        changed(
                            nodes[0],
                            normalized_geometry=tuple(changed_geometry),
                        ),
                        nodes[1],
                    ),
                ),
                "prediction[0].raw_nodes[0]"
                ".normalized_geometry[17]",
            ),
        )
        for candidate, path in cases:
            with self.subTest(path=path):
                with self.assertRaisesRegex(
                    AssertionError, re.escape(path)
                ):
                    _assert_raw_prediction_tuples_close(
                        self, (baseline,), (candidate,)
                    )


if __name__ == "__main__":
    unittest.main()
