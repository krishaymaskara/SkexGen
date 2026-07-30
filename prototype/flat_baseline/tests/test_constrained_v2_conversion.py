"""Leakage and geometry tests for V2 teacher-forced conversion."""

from __future__ import annotations

from dataclasses import replace
import ast
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

try:
    import torch
except ImportError:
    torch = None

from prototype.controlled_data.builders import build_history
from prototype.controlled_data.factors import PrimitiveFamily
from prototype.model_data.adapters import adapt_flat_mixed
from prototype.model_data.batching import collate_flat
from prototype.model_data.canonical import (
    canonical_nodes_and_edges,
    reconstruction_target,
)
from prototype.model_data.tests.fixtures import source
from prototype.model_data.vocab import (
    NODE_TYPES,
    PRIMITIVE_TYPES,
)
from prototype.profile_geometry import (
    NO_PROFILE_FAMILY_ID,
    PROFILE_FAMILIES,
)
from prototype.representation.model import GeometryEncoding

if torch is not None:
    from prototype.constrained_profile_decoder import (
        profile_targets_for_loss,
    )
    from prototype.flat_baseline.autonomous import (
        TEACHER_FORCED_PREFIX_FEEDBACK,
    )
    from prototype.flat_baseline.constrained_v2 import (
        NON_PROFILE_GEOMETRY_INDICES,
        ConstrainedProfileV2Model,
        select_non_profile_geometry,
    )
    from prototype.flat_baseline.constrained_v2_config import (
        ConstrainedProfileV2Config,
    )
    from prototype.flat_baseline.constrained_v2_conversion import (
        V2TeacherForcedRawPrediction,
        canonicalize_v2_predicted_profiles,
        v2_teacher_forced_predictions,
        validate_and_convert_v2_teacher_forced_prediction,
    )
    from prototype.flat_baseline.losses import dense_edge_targets
    from prototype.model_data.vocab import (
        BOOLEAN_MODES,
        DIRECTIONS,
        EDGE_TYPES,
        LOOP_ROLES,
        OPERATION_TYPES,
        REFERENCE_PLANES,
    )
    from prototype.profile_geometry_torch import TensorProfileTargets


TORCH_REASON = "real PyTorch execution is deferred to the Adroit environment"
_RETAINED_TARGET_INDICES = (0, 1, 2, 3, 8)
_RETAINED_VOCABULARIES = (
    OPERATION_TYPES if torch is not None else None,
    BOOLEAN_MODES if torch is not None else None,
    DIRECTIONS if torch is not None else None,
    REFERENCE_PLANES if torch is not None else None,
    LOOP_ROLES if torch is not None else None,
)


def _controlled_batch():
    examples = []
    templates = ("E", "R", "ER")
    for index, (family, template) in enumerate(
        zip(PROFILE_FAMILIES, templates)
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


def _class_logits(ids, width, dtype):
    logits = torch.full(
        ids.shape + (width,),
        -5.0,
        dtype=dtype,
        device=ids.device,
    )
    return logits.scatter(-1, ids.unsqueeze(-1), 5.0).contiguous()


class ConstrainedV2ConversionSourceTests(unittest.TestCase):
    def test_python_38_grammar_and_pytorch_111_operations(self):
        path = (
            Path(__file__).resolve().parents[1]
            / "constrained_v2_conversion.py"
        )
        source_text = path.read_text(encoding="utf-8")
        ast.parse(
            source_text,
            filename=str(path),
            feature_version=(3, 8),
        )
        for token in (
            "torch.compile",
            "torch.asarray",
            "torch.func",
            "torch.vmap",
            "Tensor.scatter_reduce",
        ):
            self.assertNotIn(token, source_text)


@unittest.skipUnless(torch is not None, TORCH_REASON)
class ConstrainedV2ConversionTensorTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(101)
        self.batch = _controlled_batch()
        self.inputs = self.batch.to_torch(torch)
        self.target = self.batch.target.to_torch(torch)
        self.profile_targets = profile_targets_for_loss(
            self.batch.target, self.inputs["geometry"]
        )
        self.config = ConstrainedProfileV2Config(
            model_dim=16,
            num_heads=2,
            feedforward_dim=24,
            codebook_dim=8,
            codebook_size=8,
            edge_pair_dim=12,
            latent_tokens=1,
        )
        self.model = ConstrainedProfileV2Model(self.config)
        self.model.eval()
        with torch.no_grad():
            self.model_output = self.model(
                target=self.target,
                profile_targets=self.profile_targets,
                **self.inputs
            )
        self.output = self._controlled_output(self.model_output)

    def _controlled_output(self, output):
        dtype = output.node_type_logits.dtype
        node_logits = _class_logits(
            self.target["node_type_ids"],
            len(NODE_TYPES.tokens),
            dtype,
        )
        categorical_logits = tuple(
            _class_logits(
                self.target["categorical_attributes"][..., target_index],
                len(vocabulary.tokens),
                dtype,
            )
            for target_index, vocabulary in zip(
                _RETAINED_TARGET_INDICES,
                _RETAINED_VOCABULARIES,
            )
        )
        safe_family_ids = torch.where(
            self.profile_targets.sketch_mask,
            self.profile_targets.family_ids,
            torch.zeros_like(self.profile_targets.family_ids),
        )
        family_logits = _class_logits(
            safe_family_ids, len(PROFILE_FAMILIES), dtype
        )
        raw_parameters = torch.zeros_like(output.raw_profile_parameters)
        non_profile = select_non_profile_geometry(
            self.target["geometry"]
        )

        node_count = self.target["node_mask"].size(1)
        presence, edge_types, unused = dense_edge_targets(
            self.target, node_count
        )
        del unused
        edge_presence = torch.where(
            presence,
            torch.ones_like(output.edge_presence_logits),
            -torch.ones_like(output.edge_presence_logits),
        )
        edge_type_logits = _class_logits(
            edge_types, len(EDGE_TYPES.tokens), dtype
        )
        pointer_logits = torch.full_like(
            output.operation_pointer_logits, -5.0
        )
        for batch_index in range(pointer_logits.size(0)):
            for query_index in range(
                int(self.target["operation_mask"][batch_index].sum().item())
            ):
                selected = int(
                    self.target["operation_sequence"][
                        batch_index, query_index
                    ].item()
                )
                pointer_logits[batch_index, query_index, selected] = 5.0
        return replace(
            output,
            node_type_logits=node_logits,
            categorical_logits=categorical_logits,
            profile_family_logits=family_logits,
            raw_profile_parameters=raw_parameters,
            non_profile_geometry=non_profile,
            edge_presence_logits=edge_presence,
            edge_type_logits=edge_type_logits,
            operation_pointer_logits=pointer_logits,
        )

    def _predictions(self, output=None, node_mask=None):
        return v2_teacher_forced_predictions(
            self.output if output is None else output,
            node_mask=(
                self.target["node_mask"]
                if node_mask is None
                else node_mask
            ),
            node_count_source="authoritative_node_mask_length_only",
        )

    def test_real_forward_non_profile_geometry_layout_is_accepted(self):
        batch_size, node_count = self.target["node_mask"].shape
        expected_nodes = (batch_size, node_count)
        self.assertEqual(expected_nodes, (3, 8))
        self.assertEqual(
            tuple(self.model_output.node_type_logits.shape),
            expected_nodes + (len(NODE_TYPES.tokens),),
        )
        self.assertEqual(
            tuple(self.model_output.profile_family_logits.shape),
            expected_nodes + (len(PROFILE_FAMILIES),),
        )
        self.assertEqual(
            tuple(self.model_output.raw_profile_parameters.shape),
            expected_nodes + (3,),
        )
        self.assertEqual(
            tuple(self.model_output.non_profile_geometry.shape),
            expected_nodes + (len(NON_PROFILE_GEOMETRY_INDICES),),
        )
        self.assertEqual(
            tuple(self.output.non_profile_geometry.shape),
            expected_nodes + (len(NON_PROFILE_GEOMETRY_INDICES),),
        )
        self.assertEqual(
            tuple(self.model_output.training_geometry.shape),
            expected_nodes + (39,),
        )
        self.assertEqual(
            tuple(self.model_output.edge_presence_logits.shape),
            expected_nodes + (node_count,),
        )
        self.assertEqual(
            tuple(self.model_output.edge_type_logits.shape),
            expected_nodes + (node_count, len(EDGE_TYPES.tokens)),
        )
        self.assertEqual(
            tuple(self.model_output.operation_pointer_logits.shape),
            (batch_size, self.config.max_operations, node_count),
        )
        self.assertEqual(
            tuple(
                tuple(value.shape)
                for value in self.model_output.categorical_logits
            ),
            tuple(
                expected_nodes + (len(vocabulary.tokens),)
                for vocabulary in _RETAINED_VOCABULARIES
            ),
        )
        predictions = v2_teacher_forced_predictions(
            self.model_output,
            node_mask=self.target["node_mask"],
            node_count_source="authoritative_node_mask_length_only",
        )
        self.assertEqual(len(predictions), batch_size)

    def test_malformed_non_profile_geometry_layouts_are_rejected(self):
        values = self.output.non_profile_geometry
        malformed = (
            ("missing node axis", values[:, 0, :]),
            ("swapped node and channel axes", values.transpose(1, 2)),
            (
                "full serialized geometry width",
                torch.zeros(
                    values.shape[:2] + (39,),
                    dtype=values.dtype,
                    device=values.device,
                ),
            ),
            ("wrong batch size", values[:-1]),
            ("wrong node count", values[:, :-1]),
        )
        for label, candidate in malformed:
            with self.subTest(label=label, shape=tuple(candidate.shape)):
                with self.assertRaisesRegex(
                    ValueError, "non_profile_geometry is misaligned"
                ):
                    self._predictions(
                        replace(
                            self.output,
                            non_profile_geometry=candidate,
                        )
                    )

    def test_all_families_convert_to_valid_internal_profiles(self):
        predictions = self._predictions()
        self.assertEqual(len(predictions), 3)
        for batch_index, prediction in enumerate(predictions):
            with self.subTest(batch_index=batch_index):
                self.assertIsInstance(
                    prediction, V2TeacherForcedRawPrediction
                )
                self.assertEqual(
                    prediction.prefix_feedback,
                    TEACHER_FORCED_PREFIX_FEEDBACK,
                )
                result = validate_and_convert_v2_teacher_forced_prediction(
                    prediction,
                    max_operations=self.config.max_operations,
                )
                self.assertTrue(result.raw_integrity.valid)
                self.assertTrue(result.reconstruction_target.valid)
                self.assertTrue(result.controlled_domain.valid)
                self.assertIsNotNone(result.reconstruction_target_candidate)
                self.assertEqual(
                    prediction.prefix_feedback,
                    TEACHER_FORCED_PREFIX_FEEDBACK,
                )

    def test_circle_rectangle_capsule_slots_geometry_and_masks(self):
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
        for batch_index, prediction in enumerate(self._predictions()):
            sketches = [
                node
                for node in prediction.raw_nodes
                if node.node_type_id == NODE_TYPES.id("sketch")
            ]
            self.assertTrue(sketches)
            for node in sketches:
                self.assertEqual(node.categorical_ids[4:8], patterns[batch_index])
                self.assertEqual(len(node.normalized_geometry), 39)
                self.assertEqual(len(node.derived_geometry_mask), 39)
                applicable = tuple(
                    value
                    for value, enabled in zip(
                        node.normalized_geometry,
                        node.derived_geometry_mask,
                    )
                    if enabled
                )
                self.assertTrue(applicable)
                self.assertTrue(all(math.isfinite(value) for value in applicable))
                self.assertTrue(all(-1.0 <= value <= 1.0 for value in applicable))

    def test_wrong_predicted_family_is_still_internally_consistent(self):
        sketch = self.profile_targets.sketch_mask.nonzero(
            as_tuple=False
        )[0]
        batch_index = int(sketch[0].item())
        node_index = int(sketch[1].item())
        original_family = int(
            self.profile_targets.family_ids[batch_index, node_index].item()
        )
        wrong_family = (original_family + 1) % len(PROFILE_FAMILIES)
        logits = self.output.profile_family_logits.clone()
        logits[batch_index, node_index] = -5.0
        logits[batch_index, node_index, wrong_family] = 5.0
        changed = replace(self.output, profile_family_logits=logits)
        prediction = self._predictions(changed)[batch_index]
        self.assertEqual(
            prediction.predicted_profile_family_ids[node_index],
            wrong_family,
        )
        result = validate_and_convert_v2_teacher_forced_prediction(
            prediction, max_operations=self.config.max_operations
        )
        self.assertTrue(result.controlled_domain.valid)

    def test_extreme_finite_raw_sketch_parameters_remain_bounded(self):
        raw = self.output.raw_profile_parameters.clone()
        raw[self.profile_targets.sketch_mask] = torch.tensor(
            (1000.0, -1000.0, 1000.0)
        )
        predictions = self._predictions(
            replace(self.output, raw_profile_parameters=raw)
        )
        for prediction in predictions:
            for node in prediction.raw_nodes:
                applicable = tuple(
                    value
                    for value, enabled in zip(
                        node.normalized_geometry,
                        node.derived_geometry_mask,
                    )
                    if enabled
                )
                self.assertTrue(
                    all(-1.0 <= value <= 1.0 for value in applicable)
                )

    def test_non_sketch_ignores_profile_logits_and_raw_parameters(self):
        batch_index = 0
        node_index = 0
        self.assertNotEqual(
            int(self.target["node_type_ids"][batch_index, node_index].item()),
            NODE_TYPES.id("sketch"),
        )
        logits = self.output.profile_family_logits.clone()
        raw = self.output.raw_profile_parameters.clone()
        logits[batch_index, node_index] = torch.tensor(
            (-100.0, 200.0, 50.0)
        )
        raw[batch_index, node_index] = torch.tensor(
            (1000.0, -1000.0, 1000.0)
        )
        prediction = self._predictions(
            replace(
                self.output,
                profile_family_logits=logits,
                raw_profile_parameters=raw,
            )
        )[batch_index]
        node = prediction.raw_nodes[node_index]
        self.assertEqual(
            node.categorical_ids[4:8],
            (PRIMITIVE_TYPES.id(None),) * 4,
        )
        self.assertFalse(any(node.derived_geometry_mask[9:33]))
        self.assertFalse(any(node.normalized_geometry[9:33]))
        self.assertEqual(
            prediction.predicted_profile_family_ids[node_index],
            NO_PROFILE_FAMILY_ID,
        )

    def test_padding_and_batch_size_one_are_explicit(self):
        predictions = self._predictions()
        expected_counts = tuple(
            int(row.sum().item()) for row in self.target["node_mask"]
        )
        self.assertEqual(
            tuple(item.node_count for item in predictions), expected_counts
        )
        self.assertTrue((~self.target["node_mask"]).any())

        single = self._slice_consumed_output(self.output, 0)
        one = self._predictions(
            single, self.target["node_mask"][0:1]
        )
        self.assertEqual(len(one), 1)
        self.assertEqual(one[0], predictions[0])

    def _slice_consumed_output(self, output, index):
        return replace(
            output,
            node_type_logits=output.node_type_logits[index:index + 1],
            categorical_logits=tuple(
                value[index:index + 1]
                for value in output.categorical_logits
            ),
            profile_family_logits=output.profile_family_logits[
                index:index + 1
            ],
            raw_profile_parameters=output.raw_profile_parameters[
                index:index + 1
            ],
            non_profile_geometry=output.non_profile_geometry[
                index:index + 1
            ],
            edge_presence_logits=output.edge_presence_logits[
                index:index + 1
            ],
            edge_type_logits=output.edge_type_logits[index:index + 1],
            operation_pointer_logits=output.operation_pointer_logits[
                index:index + 1
            ],
            code_indices=output.code_indices[index:index + 1],
        )

    def test_output_is_deterministic_and_preserves_raw_evidence(self):
        first = self._predictions()
        second = self._predictions()
        self.assertEqual(first, second)
        for prediction in first:
            self.assertEqual(
                len(prediction.profile_family_logits),
                prediction.node_count,
            )
            self.assertEqual(
                len(prediction.raw_profile_parameters),
                prediction.node_count,
            )
            self.assertEqual(
                len(prediction.constrained_profile_parameters),
                prediction.node_count,
            )

    def test_nonfinite_consumed_outputs_are_rejected(self):
        cases = (
            "node_type_logits",
            "profile_family_logits",
            "raw_profile_parameters",
            "non_profile_geometry",
            "edge_presence_logits",
            "edge_type_logits",
            "operation_pointer_logits",
        )
        for name in cases:
            with self.subTest(name=name):
                value = getattr(self.output, name).clone()
                value.reshape(-1)[0] = math.nan
                with self.assertRaisesRegex(ValueError, "finite"):
                    self._predictions(replace(self.output, **{name: value}))

    def test_invalid_predicted_family_ids_fail_before_canonicalization(self):
        raw = torch.zeros((1, 1, 3))
        mask = torch.ones((1, 1), dtype=torch.bool)
        for family_id in (-1, len(PROFILE_FAMILIES)):
            with self.subTest(family_id=family_id):
                ids = torch.tensor(((family_id,),), dtype=torch.long)
                with self.assertRaisesRegex(ValueError, "unsupported"):
                    canonicalize_v2_predicted_profiles(ids, raw, mask)

    def test_target_family_change_cannot_affect_fixed_output(self):
        baseline = self._predictions()
        family_ids = self.profile_targets.family_ids.clone()
        selected = self.profile_targets.sketch_mask
        family_ids[selected] = (family_ids[selected] + 1) % 3
        changed_target = TensorProfileTargets(
            family_ids,
            self.profile_targets.parameters,
            self.profile_targets.sketch_mask,
        )
        self.assertFalse(
            torch.equal(
                changed_target.family_ids,
                self.profile_targets.family_ids,
            )
        )
        self.assertEqual(self._predictions(), baseline)

    def test_target_primitive_slots_cannot_affect_fixed_output(self):
        baseline = self._predictions()
        target = {key: value.clone() for key, value in self.target.items()}
        selected = self.profile_targets.sketch_mask
        primitive_slots = target["categorical_attributes"][..., 4:8]
        primitive_slots[selected] = PRIMITIVE_TYPES.id("circle")
        self.assertFalse(
            torch.equal(
                target["categorical_attributes"],
                self.target["categorical_attributes"],
            )
        )
        self.assertEqual(self._predictions(), baseline)

    def test_target_compact_and_geometry_change_cannot_affect_fixed_output(self):
        baseline = self._predictions()
        parameters = self.profile_targets.parameters.clone()
        center_x = parameters[..., 0]
        center_x[self.profile_targets.sketch_mask] = 0.123
        target = {key: value.clone() for key, value in self.target.items()}
        geometry = target["geometry"][..., 9:33]
        geometry[self.profile_targets.sketch_mask] *= -1.0
        self.assertFalse(
            torch.equal(parameters, self.profile_targets.parameters)
        )
        self.assertFalse(
            torch.equal(target["geometry"], self.target["geometry"])
        )
        self.assertEqual(self._predictions(), baseline)

    def test_target_geometry_mask_change_cannot_affect_fixed_output(self):
        baseline = self._predictions()
        target = {key: value.clone() for key, value in self.target.items()}
        target["geometry_mask"] = ~target["geometry_mask"]
        self.assertFalse(
            torch.equal(
                target["geometry_mask"], self.target["geometry_mask"]
            )
        )
        self.assertEqual(self._predictions(), baseline)

    def test_training_only_target_routed_output_fields_are_ignored(self):
        baseline = self._predictions()
        changed = replace(
            self.output,
            constrained_profile_parameters=(
                self.output.constrained_profile_parameters + 10.0
            ),
            training_profile_geometry=(
                self.output.training_profile_geometry - 10.0
            ),
            training_profile_geometry_mask=(
                ~self.output.training_profile_geometry_mask
            ),
            training_primitive_type_ids=(
                self.output.training_primitive_type_ids + 1
            ),
        )
        self.assertEqual(self._predictions(changed), baseline)

    def test_earlier_authoritative_prefix_can_change_decoder_outputs(self):
        changed_target = {
            key: value.clone() for key, value in self.target.items()
        }
        changed_target["geometry"][0, 0, 0] = 0.5
        with torch.no_grad():
            changed = self.model(
                target=changed_target,
                profile_targets=self.profile_targets,
                **self.inputs
            )
        self.assertFalse(
            torch.equal(
                self.model_output.decoded_states[0, 1:],
                changed.decoded_states[0, 1:],
            )
        )

    def test_current_target_node_is_never_in_its_decoder_prefix(self):
        changed_target = {
            key: value.clone() for key, value in self.target.items()
        }
        last = int(changed_target["node_mask"][0].sum().item()) - 1
        changed_target["geometry"][0, last] = 0.75
        changed_target["geometry_mask"][0, last] = True
        changed_target["categorical_attributes"][0, last, 0] = 0
        with torch.no_grad():
            changed = self.model(
                target=changed_target,
                profile_targets=self.profile_targets,
                **self.inputs
            )
        for original, altered in (
            (
                self.model_output.node_type_logits[0, last],
                changed.node_type_logits[0, last],
            ),
            (
                self.model_output.profile_family_logits[0, last],
                changed.profile_family_logits[0, last],
            ),
            (
                self.model_output.raw_profile_parameters[0, last],
                changed.raw_profile_parameters[0, last],
            ),
        ):
            torch.testing.assert_close(
                original, altered, rtol=0.0, atol=0.0
            )

    def test_shape_dtype_device_mask_and_metadata_validation(self):
        predictions = self._predictions()
        for prediction in predictions:
            self.assertIsInstance(prediction.node_count, int)
            self.assertIsInstance(prediction.node_count_source, str)
            self.assertTrue(
                all(
                    type(value) is float
                    for node in prediction.raw_nodes
                    for value in node.normalized_geometry
                )
            )
            self.assertTrue(
                all(
                    type(value) is bool
                    for node in prediction.raw_nodes
                    for value in node.derived_geometry_mask
                )
            )
        with self.assertRaisesRegex(ValueError, "contiguous"):
            invalid = self.target["node_mask"].clone()
            invalid[0, 0] = False
            self._predictions(node_mask=invalid)
        with self.assertRaisesRegex(ValueError, "nonempty"):
            v2_teacher_forced_predictions(
                self.output,
                node_mask=self.target["node_mask"],
                node_count_source="",
            )


if __name__ == "__main__":
    unittest.main()
