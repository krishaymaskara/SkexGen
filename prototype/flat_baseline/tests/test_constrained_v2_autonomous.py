"""Autonomous predicted-history tests for the constrained V2 flat model."""

from __future__ import annotations

from dataclasses import replace
import ast
import inspect
import math
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import unittest

try:
    import torch
except ImportError:
    torch = None

from prototype.controlled_data.builders import build_history
from prototype.model_data.adapters import adapt_flat_mixed
from prototype.model_data.batching import collate_flat
from prototype.model_data.canonical import (
    canonical_nodes_and_edges,
    reconstruction_target,
)
from prototype.model_data.tests.fixtures import source
from prototype.profile_geometry import PROFILE_FAMILIES
from prototype.representation.model import GeometryEncoding

if torch is not None:
    from prototype.constrained_profile_decoder import (
        profile_targets_for_loss,
    )
    from prototype.flat_baseline.constrained_v2 import (
        NON_PROFILE_GEOMETRY_INDICES,
        ConstrainedProfileV2Model,
    )
    from prototype.flat_baseline.constrained_v2_autonomous import (
        V2_AUTONOMOUS_PREFIX_FEEDBACK,
        V2_ENCODED_MEMORY_SOURCE,
        V2AutonomousRawPrediction,
        V2EncodedMemory,
        encode_v2_to_memory,
        greedy_decode_v2,
        greedy_decode_v2_from_memory,
        validate_and_convert_v2_autonomous_prediction,
    )
    from prototype.flat_baseline.constrained_v2_config import (
        ConstrainedProfileV2Config,
    )
    from prototype.flat_baseline.constrained_v2_conversion import (
        construct_v2_predicted_node_tensors,
        v2_teacher_forced_predictions,
    )
    from prototype.flat_baseline.conversion import ConversionResult
    from prototype.model_data.geometry import GEOMETRY_WIDTH
    from prototype.model_data.vocab import (
        BOOLEAN_MODES,
        DIRECTIONS,
        LOOP_ROLES,
        NODE_TYPES,
        OPERATION_TYPES,
        PRIMITIVE_TYPES,
        REFERENCE_PLANES,
    )


TORCH_REASON = "real PyTorch execution is deferred to the Adroit environment"
_RETAINED_VOCABULARIES = (
    OPERATION_TYPES if torch is not None else None,
    BOOLEAN_MODES if torch is not None else None,
    DIRECTIONS if torch is not None else None,
    REFERENCE_PLANES if torch is not None else None,
    LOOP_ROLES if torch is not None else None,
)


def _controlled_batch():
    examples = []
    for index, (family, template) in enumerate(
        zip(PROFILE_FAMILIES, ("E", "R", "ER"))
    ):
        history = build_history(
            source(
                template,
                family,
                extents=(1.0 + index,) * len(template),
            ),
            GeometryEncoding.CONTINUOUS,
        )
        nodes, edges = canonical_nodes_and_edges(history)
        target = reconstruction_target(
            nodes, edges, history.structure.operation_sequence
        )
        examples.append(
            adapt_flat_mixed(
                SimpleNamespace(
                    physical_family_id="{:02d}-{}".format(
                        index, family.value
                    ),
                    nodes=nodes,
                    target=target,
                )
            )
        )
    return collate_flat(tuple(examples))


class ConstrainedV2AutonomousSourceTests(unittest.TestCase):
    def test_python_38_grammar_api_and_pytorch_111_operations(self):
        root = Path(__file__).resolve().parents[1]
        paths = (
            root / "constrained_v2.py",
            root / "constrained_v2_conversion.py",
            root / "constrained_v2_autonomous.py",
        )
        forbidden = (
            "torch.compile",
            "torch.asarray",
            "torch.func",
            "torch.vmap",
            "Tensor.scatter_reduce",
        )
        for path in paths:
            source_text = path.read_text(encoding="utf-8")
            ast.parse(
                source_text,
                filename=str(path),
                feature_version=(3, 8),
            )
            for token in forbidden:
                self.assertNotIn(token, source_text)

    @unittest.skipUnless(torch is not None, TORCH_REASON)
    def test_autonomous_interfaces_do_not_accept_targets(self):
        for function in (
            encode_v2_to_memory,
            greedy_decode_v2,
            greedy_decode_v2_from_memory,
        ):
            parameters = inspect.signature(function).parameters
            self.assertNotIn("target", parameters)
            self.assertNotIn("authoritative_batch", parameters)


@unittest.skipUnless(torch is not None, TORCH_REASON)
class ConstrainedV2AutonomousTensorTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(211)
        self.batch = _controlled_batch()
        self.inputs = self.batch.to_torch(torch)
        self.target = self.batch.target.to_torch(torch)
        self.config = ConstrainedProfileV2Config(
            model_dim=16,
            num_heads=2,
            feedforward_dim=24,
            encoder_layers=1,
            decoder_layers=1,
            dropout=0.25,
            max_nodes=10,
            max_operations=2,
            latent_tokens=1,
            codebook_size=8,
            codebook_dim=8,
            edge_pair_dim=12,
        )
        self.model = ConstrainedProfileV2Model(self.config)
        self.model.eval()
        self.memory = encode_v2_to_memory(self.model, self.inputs)
        self.counts = torch.tensor(
            tuple(
                int(row.sum().item())
                for row in self.target["node_mask"]
            ),
            dtype=torch.long,
        )

    def _decode(self, memory=None, counts=None):
        return greedy_decode_v2_from_memory(
            self.model,
            self.memory if memory is None else memory,
            node_counts=self.counts if counts is None else counts,
            node_count_source="authorized_controlled_length",
        )

    def _single_memory(self, index=0):
        return V2EncodedMemory(
            self.memory.memory[index : index + 1],
            self.memory.code_indices[index : index + 1],
            self.memory.encoding_source,
            self.memory.model_name,
        )

    def _force_node(self, node_type, family_id=0, raw=(0.0, 0.0, 0.0)):
        with torch.no_grad():
            self.model.node_type_head.weight.zero_()
            self.model.node_type_head.bias.fill_(-10.0)
            self.model.node_type_head.bias[NODE_TYPES.id(node_type)] = 10.0
            for head, vocabulary in zip(
                self.model.remaining_categorical_heads,
                _RETAINED_VOCABULARIES,
            ):
                head.weight.zero_()
                head.bias.fill_(-10.0)
                head.bias[vocabulary.id(None)] = 10.0
            self.model.profile_heads.family_head.weight.zero_()
            self.model.profile_heads.family_head.bias.fill_(-10.0)
            self.model.profile_heads.family_head.bias[family_id] = 10.0
            self.model.profile_heads.parameter_head.weight.zero_()
            self.model.profile_heads.parameter_head.bias.copy_(
                self.model.profile_heads.parameter_head.bias.new_tensor(raw)
            )
            self.model.non_profile_geometry_head.weight.zero_()
            self.model.non_profile_geometry_head.bias.zero_()

    def test_encoding_is_source_only_and_preserves_mode_state_and_ema(self):
        self.model.train()
        before = {
            name: value.detach().clone()
            for name, value in self.model.state_dict().items()
        }
        encoded = encode_v2_to_memory(self.model, self.inputs)
        self.assertTrue(self.model.training)
        self.assertEqual(
            encoded.memory.shape,
            (
                len(self.batch.family_ids),
                self.config.latent_tokens,
                self.config.model_dim,
            ),
        )
        self.assertEqual(
            encoded.code_indices.shape,
            (len(self.batch.family_ids), self.config.latent_tokens),
        )
        self.assertEqual(
            encoded.encoding_source, V2_ENCODED_MEMORY_SOURCE
        )
        self.assertEqual(encoded.model_name, self.config.model_name)
        self.assertEqual(encoded.memory.dtype, self.model.bos.dtype)
        self.assertEqual(encoded.memory.device, self.model.bos.device)
        self.assertEqual(encoded.code_indices.dtype, torch.long)
        self.assertEqual(encoded.code_indices.device, self.model.bos.device)
        self.assertTrue(encoded.memory.is_contiguous())
        self.assertTrue(encoded.code_indices.is_contiguous())
        self.assertFalse(encoded.memory.requires_grad)
        self.assertFalse(encoded.code_indices.requires_grad)
        for name, value in self.model.state_dict().items():
            torch.testing.assert_close(
                value, before[name], rtol=0.0, atol=0.0
            )
        with self.assertRaisesRegex(ValueError, "only the four"):
            encode_v2_to_memory(
                self.model,
                dict(self.inputs, target=self.target),
            )

    def test_encoding_restores_mode_after_exception(self):
        self.model.train()
        with mock.patch.object(
            self.model,
            "encode_to_memory",
            side_effect=RuntimeError("injected encoding failure"),
        ):
            with self.assertRaisesRegex(RuntimeError, "injected"):
                encode_v2_to_memory(self.model, self.inputs)
        self.assertTrue(self.model.training)

    def test_mixed_lengths_batch_one_permutation_and_composition_invariance(self):
        counts = torch.tensor((1, 3, 6), dtype=torch.long)
        predictions = self._decode(counts=counts)
        self.assertEqual(
            tuple(item.node_count for item in predictions), (1, 3, 6)
        )
        self.assertEqual(
            tuple(len(item.raw_nodes) for item in predictions), (1, 3, 6)
        )
        self.assertEqual(
            tuple(len(item.raw_edges) for item in predictions), (1, 9, 36)
        )
        one = self._decode(
            self._single_memory(1), torch.tensor((3,), dtype=torch.long)
        )
        self.assertEqual(one, (predictions[1],))

        order = torch.tensor((2, 0, 1), dtype=torch.long)
        permuted_memory = V2EncodedMemory(
            self.memory.memory[order],
            self.memory.code_indices[order],
            self.memory.encoding_source,
            self.memory.model_name,
        )
        permuted = self._decode(
            permuted_memory, counts[order]
        )
        self.assertEqual(
            permuted,
            (predictions[2], predictions[0], predictions[1]),
        )

    def test_bos_only_then_complete_generated_records_grow_prefix(self):
        self._force_node("sketch", family_id=2)
        captured = []
        original = self.model.decode_prefix

        def capture(memory, categories, geometry, geometry_mask):
            captured.append(
                (
                    categories.detach().clone(),
                    geometry.detach().clone(),
                    geometry_mask.detach().clone(),
                )
            )
            return original(
                memory, categories, geometry, geometry_mask
            )

        with mock.patch.object(
            self.model, "decode_prefix", side_effect=capture
        ):
            prediction = self._decode(
                self._single_memory(), torch.tensor((3,), dtype=torch.long)
            )[0]
        self.assertEqual(
            tuple(item[0].size(1) for item in captured), (0, 1, 2, 2)
        )
        self.assertEqual(captured[0][0].numel(), 0)
        first = prediction.raw_nodes[0]
        expected_categories = torch.tensor(
            ((first.node_type_id, *first.categorical_ids),),
            dtype=torch.long,
        )
        expected_geometry = torch.tensor(
            (first.normalized_geometry,), dtype=self.memory.memory.dtype
        )
        expected_mask = torch.tensor(
            (first.derived_geometry_mask,), dtype=torch.bool
        )
        self.assertTrue(
            torch.equal(captured[1][0][0], expected_categories)
        )
        torch.testing.assert_close(
            captured[1][1][0], expected_geometry, rtol=0.0, atol=0.0
        )
        self.assertTrue(
            torch.equal(captured[1][2][0], expected_mask)
        )

    def test_all_profile_families_have_canonical_feedback(self):
        patterns = (
            (
                PRIMITIVE_TYPES.id("circle"),
                PRIMITIVE_TYPES.id(None),
                PRIMITIVE_TYPES.id(None),
                PRIMITIVE_TYPES.id(None),
            ),
            (PRIMITIVE_TYPES.id("line"),) * 4,
            (
                PRIMITIVE_TYPES.id("arc"),
                PRIMITIVE_TYPES.id("arc"),
                PRIMITIVE_TYPES.id("line"),
                PRIMITIVE_TYPES.id("line"),
            ),
        )
        for family_id, pattern in enumerate(patterns):
            with self.subTest(family_id=family_id):
                self._force_node("sketch", family_id=family_id)
                prediction = self._decode(
                    self._single_memory(),
                    torch.tensor((1,), dtype=torch.long),
                )[0]
                node = prediction.raw_nodes[0]
                self.assertEqual(node.categorical_ids[4:8], pattern)
                self.assertEqual(
                    prediction.predicted_profile_family_ids, (family_id,)
                )
                self.assertTrue(all(
                    value == 0.0
                    for value, active in zip(
                        node.normalized_geometry,
                        node.derived_geometry_mask,
                    )
                    if not active
                ))
                self.assertTrue(all(
                    -1.0 <= value <= 1.0
                    for value in node.normalized_geometry
                ))
                if family_id == 1:
                    self._assert_rectangle_closed(node.normalized_geometry)
                elif family_id == 2:
                    self._assert_capsule_closed(node.normalized_geometry)
                else:
                    self.assertEqual(
                        node.normalized_geometry[11], 0.25
                    )

    def _assert_rectangle_closed(self, geometry):
        lines = tuple(
            geometry[start : start + 4]
            for start in (9, 15, 21, 27)
        )
        for current, following in zip(
            lines, lines[1:] + lines[:1]
        ):
            self.assertEqual(current[2:4], following[0:2])

    def _assert_capsule_closed(self, geometry):
        arc_right = geometry[9:15]
        arc_left = geometry[15:21]
        line_bottom = geometry[21:25]
        line_top = geometry[27:31]
        self.assertEqual(line_bottom[2:4], arc_right[0:2])
        self.assertEqual(arc_right[4:6], line_top[0:2])
        self.assertEqual(line_top[2:4], arc_left[0:2])
        self.assertEqual(arc_left[4:6], line_bottom[0:2])

    def test_non_sketch_ignores_profile_outputs_and_profile_channels(self):
        self._force_node(
            "reference_plane",
            family_id=2,
            raw=(1000.0, -1000.0, 1000.0),
        )
        prediction = self._decode(
            self._single_memory(), torch.tensor((1,), dtype=torch.long)
        )[0]
        node = prediction.raw_nodes[0]
        self.assertEqual(
            node.categorical_ids[4:8],
            (PRIMITIVE_TYPES.id(None),) * 4,
        )
        self.assertEqual(prediction.predicted_profile_family_ids, (-1,))
        self.assertEqual(
            prediction.constrained_profile_parameters,
            ((0.0, 0.0, 0.0),),
        )
        self.assertFalse(any(node.derived_geometry_mask[9:33]))
        self.assertFalse(any(node.normalized_geometry[9:33]))
        self.assertTrue(all(node.derived_geometry_mask[:9]))
        self.assertFalse(any(node.derived_geometry_mask[33:]))

    def test_profile_and_non_profile_channels_are_disjoint(self):
        self._force_node("sketch", family_id=1)
        sketch = self._decode(
            self._single_memory(), torch.tensor((1,), dtype=torch.long)
        )[0].raw_nodes[0]
        self.assertFalse(any(
            sketch.derived_geometry_mask[index]
            for index in NON_PROFILE_GEOMETRY_INDICES
        ))
        self._force_node("extrude", family_id=1)
        operation = self._decode(
            self._single_memory(), torch.tensor((1,), dtype=torch.long)
        )[0].raw_nodes[0]
        self.assertTrue(operation.derived_geometry_mask[37])
        self.assertFalse(any(operation.derived_geometry_mask[9:33]))

    def test_operation_counts_and_pointers_come_from_generated_types(self):
        self._force_node("extrude")
        prediction = self._decode(
            self._single_memory(), torch.tensor((3,), dtype=torch.long)
        )[0]
        self.assertEqual(
            prediction.predicted_operation_node_indices, (0, 1, 2)
        )
        self.assertEqual(prediction.predicted_operation_count, 3)
        self.assertTrue(prediction.operation_count_exceeds_limit)
        self.assertEqual(len(prediction.raw_operation_pointers), 2)
        self.assertTrue(all(
            0 <= item.selected_node_index < 3
            for item in prediction.raw_operation_pointers
        ))

    def test_determinism_no_grad_and_model_state_preservation(self):
        self.model.train()
        before = {
            name: value.detach().clone()
            for name, value in self.model.state_dict().items()
        }
        first = self._decode()
        second = self._decode()
        self.assertEqual(first, second)
        self.assertTrue(self.model.training)
        for name, value in self.model.state_dict().items():
            torch.testing.assert_close(
                value, before[name], rtol=0.0, atol=0.0
            )
        self.assertTrue(all(
            type(value) is float and math.isfinite(value)
            for prediction in first
            for node in prediction.raw_nodes
            for value in node.normalized_geometry
        ))

    def test_nonfinite_output_is_rejected_and_mode_restored_after_exception(self):
        self.model.train()
        original = self.model.decode_prefix

        def nonfinite(*args, **kwargs):
            output = original(*args, **kwargs)
            values = output.non_profile_geometry.clone()
            values.reshape(-1)[0] = math.nan
            return replace(output, non_profile_geometry=values)

        with mock.patch.object(
            self.model, "decode_prefix", side_effect=nonfinite
        ):
            with self.assertRaisesRegex(ValueError, "finite"):
                self._decode(
                    self._single_memory(),
                    torch.tensor((1,), dtype=torch.long),
                )
        self.assertTrue(self.model.training)

    def test_nonfinite_relation_output_is_rejected(self):
        original = self.model.decode_relations

        def nonfinite(*args, **kwargs):
            output = original(*args, **kwargs)
            values = output.edge_presence_logits.clone()
            values.reshape(-1)[0] = math.inf
            return replace(output, edge_presence_logits=values)

        with mock.patch.object(
            self.model, "decode_relations", side_effect=nonfinite
        ):
            with self.assertRaisesRegex(ValueError, "finite"):
                self._decode(
                    self._single_memory(),
                    torch.tensor((1,), dtype=torch.long),
                )

    def test_encoded_memory_shape_dtype_finiteness_and_provenance(self):
        invalid = (
            replace(
                self.memory,
                memory=self.memory.memory[:, :, :-1],
            ),
            replace(
                self.memory,
                memory=self.memory.memory.to(torch.float64),
            ),
            replace(
                self.memory,
                code_indices=self.memory.code_indices.to(torch.int32),
            ),
            replace(self.memory, encoding_source="unknown"),
            replace(self.memory, model_name="another-model"),
        )
        for value in invalid:
            with self.subTest(
                memory_shape=tuple(value.memory.shape),
                memory_dtype=value.memory.dtype,
                index_dtype=value.code_indices.dtype,
            ):
                with self.assertRaises((TypeError, ValueError)):
                    self._decode(memory=value)
        nonfinite = self.memory.memory.clone()
        nonfinite.reshape(-1)[0] = math.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            self._decode(memory=replace(self.memory, memory=nonfinite))

    def test_node_count_shape_dtype_bounds_device_and_metadata(self):
        cases = (
            torch.tensor((1, 2), dtype=torch.long),
            torch.tensor(((1, 2, 3),), dtype=torch.long),
            torch.tensor((1, 2, 3), dtype=torch.int32),
            torch.tensor((0, 2, 3), dtype=torch.long),
            torch.tensor((1, 2, 11), dtype=torch.long),
        )
        for counts in cases:
            with self.subTest(shape=tuple(counts.shape), dtype=counts.dtype):
                with self.assertRaises((TypeError, ValueError)):
                    self._decode(counts=counts)
        with self.assertRaisesRegex(ValueError, "nonempty"):
            greedy_decode_v2_from_memory(
                self.model,
                self.memory,
                node_counts=self.counts,
                node_count_source="",
            )

    def test_current_node_constructor_rejects_malformed_outputs(self):
        node_ids = torch.tensor((NODE_TYPES.id("sketch"),))
        retained = torch.tensor(
            (
                tuple(
                    vocabulary.id(None)
                    for vocabulary in _RETAINED_VOCABULARIES
                ),
            ),
            dtype=torch.long,
        )
        family_logits = torch.zeros((1, len(PROFILE_FAMILIES)))
        raw = torch.zeros((1, 3))
        non_profile = torch.zeros((1, 15))
        valid = construct_v2_predicted_node_tensors(
            node_ids, retained, family_logits, raw, non_profile
        )
        self.assertEqual(valid.categorical_ids.shape, (1, 9))
        self.assertEqual(valid.geometry.shape, (1, GEOMETRY_WIDTH))
        self.assertEqual(valid.geometry_mask.dtype, torch.bool)
        self.assertTrue(valid.geometry.is_contiguous())
        self.assertTrue(valid.geometry_mask.is_contiguous())

        invalid_node_ids = node_ids.clone()
        invalid_node_ids[0] = len(NODE_TYPES.tokens)
        bad_non_profile = non_profile.clone()
        bad_non_profile[0, 0] = math.nan
        cases = (
            (invalid_node_ids, retained, family_logits, raw, non_profile),
            (node_ids, retained[:, :-1], family_logits, raw, non_profile),
            (node_ids, retained, family_logits[:, :-1], raw, non_profile),
            (node_ids, retained, family_logits, raw[:, :-1], non_profile),
            (node_ids, retained, family_logits, raw, non_profile[:, :-1]),
            (node_ids, retained, family_logits, raw, bad_non_profile),
        )
        for values in cases:
            with self.subTest(shapes=tuple(
                tuple(value.shape) for value in values
            )):
                with self.assertRaises((TypeError, ValueError)):
                    construct_v2_predicted_node_tensors(*values)

    def test_target_mutation_cannot_change_autonomous_predictions(self):
        baseline = self._decode()
        changed_target = {}
        for name, value in self.target.items():
            changed = value.clone()
            if changed.dtype == torch.bool:
                changed = ~changed
            elif changed.numel():
                changed.reshape(-1)[0] += 1
            changed_target[name] = changed
        self.assertTrue(any(
            not torch.equal(changed_target[name], self.target[name])
            for name in self.target
        ))
        self.assertEqual(self._decode(), baseline)
        with self.assertRaisesRegex(ValueError, "only the four"):
            greedy_decode_v2(
                self.model,
                dict(self.inputs, target=changed_target),
                node_counts=self.counts,
                node_count_source="authorized_controlled_length",
            )

    def test_generated_family_feedback_changes_later_decoder_state(self):
        captured = []
        original = self.model.decode_prefix

        def capture(*args, **kwargs):
            output = original(*args, **kwargs)
            captured.append(output.decoded_states.detach().clone())
            return output

        self._force_node("sketch", family_id=0)
        with mock.patch.object(
            self.model, "decode_prefix", side_effect=capture
        ):
            circle = self._decode(
                self._single_memory(), torch.tensor((2,), dtype=torch.long)
            )[0]
        circle_second_state = captured[1][0, -1].clone()
        captured[:] = []
        self._force_node("sketch", family_id=1)
        with mock.patch.object(
            self.model, "decode_prefix", side_effect=capture
        ):
            rectangle = self._decode(
                self._single_memory(), torch.tensor((2,), dtype=torch.long)
            )[0]
        rectangle_second_state = captured[1][0, -1]
        self.assertNotEqual(
            circle.raw_nodes[0].categorical_ids[4:8],
            rectangle.raw_nodes[0].categorical_ids[4:8],
        )
        self.assertFalse(torch.equal(
            circle_second_state, rectangle_second_state
        ))

    def test_teacher_forced_and_autonomous_share_current_node_construction(self):
        profile_targets = profile_targets_for_loss(
            self.batch.target, self.inputs["geometry"]
        )
        self.model.eval()
        with torch.no_grad():
            output = self.model(
                target=self.target,
                profile_targets=profile_targets,
                **self.inputs
            )
        predictions = v2_teacher_forced_predictions(
            output,
            node_mask=self.target["node_mask"],
            node_count_source="authoritative_node_mask_length_only",
        )
        node_ids = output.node_type_logits.argmax(dim=-1)
        retained = torch.stack(
            tuple(
                logits.argmax(dim=-1)
                for logits in output.categorical_logits
            ),
            dim=-1,
        )
        records = construct_v2_predicted_node_tensors(
            node_ids,
            retained,
            output.profile_family_logits,
            output.raw_profile_parameters,
            output.non_profile_geometry,
        )
        for batch_index, prediction in enumerate(predictions):
            for position, node in enumerate(prediction.raw_nodes):
                self.assertEqual(
                    node.categorical_ids,
                    tuple(
                        int(value)
                        for value in records.categorical_ids[
                            batch_index, position
                        ].tolist()
                    ),
                )
                self.assertEqual(
                    node.derived_geometry_mask,
                    tuple(
                        bool(value)
                        for value in records.geometry_mask[
                            batch_index, position
                        ].tolist()
                    ),
                )
                self.assertEqual(
                    node.normalized_geometry,
                    tuple(
                        float(value)
                        for value in records.geometry[
                            batch_index, position
                        ].tolist()
                    ),
                )

    def test_raw_evidence_and_strict_conversion_adapter(self):
        prediction = self._decode(
            self._single_memory(), torch.tensor((2,), dtype=torch.long)
        )[0]
        self.assertIsInstance(prediction, V2AutonomousRawPrediction)
        self.assertEqual(
            prediction.prefix_feedback, V2_AUTONOMOUS_PREFIX_FEEDBACK
        )
        self.assertEqual(
            prediction.encoded_memory_source, V2_ENCODED_MEMORY_SOURCE
        )
        self.assertEqual(
            prediction.node_count_source, "authorized_controlled_length"
        )
        self.assertEqual(prediction.node_count, 2)
        self.assertEqual(len(prediction.profile_family_logits), 2)
        self.assertEqual(len(prediction.raw_profile_parameters), 2)
        self.assertEqual(len(prediction.constrained_profile_parameters), 2)
        result = validate_and_convert_v2_autonomous_prediction(
            prediction, max_operations=self.config.max_operations
        )
        self.assertIsInstance(result, ConversionResult)


if __name__ == "__main__":
    unittest.main()
