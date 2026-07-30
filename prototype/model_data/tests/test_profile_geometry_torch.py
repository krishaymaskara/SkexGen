"""Differentiable shared-profile tensor contract tests."""

from __future__ import annotations

from dataclasses import replace
import ast
import math
from pathlib import Path
import unittest

try:
    import torch
except ImportError:
    torch = None

from prototype.controlled_data.builders import build_history
from prototype.controlled_data.factors import PrimitiveFamily
from prototype.model_data.batching import _collate_targets
from prototype.model_data.canonical import (
    canonical_nodes_and_edges,
    reconstruction_target,
)
from prototype.model_data.geometry import GEOMETRY_WIDTH
from prototype.model_data.tests.fixtures import source
from prototype.model_data.vocab import NODE_TYPES, PRIMITIVE_TYPES
from prototype.profile_geometry import (
    NO_PROFILE_FAMILY_ID,
    PROFILE_EXTENT_MAX,
    PROFILE_EXTENT_MIN,
    PROFILE_FAMILIES,
    canonical_primitive_type_ids,
    construct_profile_geometry,
    profile_family_id,
)
from prototype.profile_geometry_torch import (
    canonicalize_profile_tensors,
    constrain_profile_parameters,
    extract_profile_target_tensors,
)
from prototype.representation.model import GeometryEncoding


def _target(template, family, extent, encoding):
    history = build_history(
        source(template, family, extents=(extent,) * len(template)),
        encoding,
    )
    nodes, edges = canonical_nodes_and_edges(history)
    return reconstruction_target(
        nodes, edges, history.structure.operation_sequence
    )


@unittest.skipIf(torch is None, "PyTorch is unavailable")
class TensorProfileGeometryTests(unittest.TestCase):
    def test_all_generator_cases_match_python_and_tensor_construction(self):
        for family in PROFILE_FAMILIES:
            for physical_extent in (1.0, 1.5, 2.0, 2.5, 3.0):
                for encoding in GeometryEncoding:
                    with self.subTest(
                        family=family.value,
                        extent=physical_extent,
                        encoding=encoding.value,
                    ):
                        target = _target(
                            "E", family, physical_extent, encoding
                        )
                        extracted = extract_profile_target_tensors(target)
                        result = canonicalize_profile_tensors(
                            extracted.family_ids,
                            extracted.parameters,
                            extracted.sketch_mask,
                        )
                        for index, is_sketch in enumerate(
                            extracted.sketch_mask.tolist()
                        ):
                            if not is_sketch:
                                continue
                            parameters = extracted.parameters[index].tolist()
                            expected = construct_profile_geometry(
                                family, *parameters
                            )
                            self.assertEqual(
                                result.primitive_type_ids[index].tolist(),
                                list(canonical_primitive_type_ids(family)),
                            )
                            self.assertEqual(
                                result.geometry[index].tolist(),
                                list(expected[0]),
                            )
                            self.assertEqual(
                                result.geometry_mask[index].tolist(),
                                list(expected[1]),
                            )

    def test_parameter_transform_is_bounded_and_differentiable(self):
        raw = torch.zeros((1, 3, 3), dtype=torch.float64, requires_grad=True)
        family_ids = torch.tensor(
            [[0, 1, 2]], dtype=torch.long
        )
        sketch_mask = torch.ones((1, 3), dtype=torch.bool)
        parameters = constrain_profile_parameters(
            raw, family_ids, sketch_mask
        )
        self.assertEqual(parameters.shape, (1, 3, 3))
        self.assertTrue(
            torch.all(parameters[..., 2] > PROFILE_EXTENT_MIN)
        )
        self.assertTrue(
            torch.all(parameters[..., 2] < PROFILE_EXTENT_MAX)
        )
        parameters.sum().backward()
        self.assertIsNotNone(raw.grad)
        self.assertTrue(torch.isfinite(raw.grad).all())
        self.assertTrue((raw.grad != 0.0).all())

    def test_extreme_finite_raw_values_reach_no_invalid_geometry(self):
        raw = torch.tensor(
            (
                (-1000.0, 1000.0, -1000.0),
                (1000.0, -1000.0, 1000.0),
                (-1000.0, -1000.0, 1000.0),
            ),
            dtype=torch.float32,
        )
        family_ids = torch.tensor((0, 1, 2), dtype=torch.long)
        sketch_mask = torch.ones(3, dtype=torch.bool)
        parameters = constrain_profile_parameters(
            raw, family_ids, sketch_mask
        )
        self.assertTrue(
            torch.all(parameters[:, 2] >= PROFILE_EXTENT_MIN)
        )
        self.assertTrue(
            torch.all(parameters[:, 2] <= PROFILE_EXTENT_MAX)
        )
        result = canonicalize_profile_tensors(
            family_ids, parameters, sketch_mask
        )
        self.assertTrue(torch.isfinite(result.geometry).all())
        self.assertTrue(torch.all(result.geometry >= -1.0))
        self.assertTrue(torch.all(result.geometry <= 1.0))

    def test_nonfinite_raw_parameters_are_rejected(self):
        family_ids = torch.tensor((0,), dtype=torch.long)
        mask = torch.tensor((True,), dtype=torch.bool)
        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(value=value):
                raw = torch.tensor(((0.0, 0.0, value),))
                with self.assertRaisesRegex(ValueError, "finite"):
                    constrain_profile_parameters(raw, family_ids, mask)

    def test_circle_uses_stored_channel_bounds_without_boundary_margin(self):
        raw = torch.tensor(((1000.0, -1000.0, 0.0),))
        family_ids = torch.tensor(
            (profile_family_id(PrimitiveFamily.CIRCLE),), dtype=torch.long
        )
        mask = torch.tensor((True,), dtype=torch.bool)
        parameters = constrain_profile_parameters(raw, family_ids, mask)
        self.assertEqual(parameters[0, 0].item(), 1.0)
        self.assertEqual(parameters[0, 1].item(), -1.0)
        result = canonicalize_profile_tensors(
            family_ids, parameters, mask
        )
        self.assertEqual(result.geometry[0, 9].item(), 1.0)
        self.assertEqual(result.geometry[0, 10].item(), -1.0)

    def test_rectangle_and_capsule_are_closed(self):
        raw = torch.zeros((2, 3))
        family_ids = torch.tensor(
            (
                profile_family_id(PrimitiveFamily.RECTANGLE_LINES),
                profile_family_id(PrimitiveFamily.CAPSULE_LINE_ARC),
            ),
            dtype=torch.long,
        )
        mask = torch.ones(2, dtype=torch.bool)
        parameters = constrain_profile_parameters(raw, family_ids, mask)
        result = canonicalize_profile_tensors(
            family_ids, parameters, mask
        )
        rectangle = result.geometry[0]
        lines = tuple(
            rectangle[9 + slot * 6:13 + slot * 6]
            for slot in range(4)
        )
        for index, line in enumerate(lines):
            self.assertTrue(
                torch.equal(line[2:4], lines[(index + 1) % 4][0:2])
            )

        capsule = result.geometry[1]
        right_arc = capsule[9:15]
        left_arc = capsule[15:21]
        lower_line = capsule[21:25]
        upper_line = capsule[27:31]
        traversal = (lower_line, right_arc, upper_line, left_arc)
        for index, primitive in enumerate(traversal):
            self.assertTrue(
                torch.equal(
                    primitive[-2:],
                    traversal[(index + 1) % 4][0:2],
                )
            )

    def test_non_sketch_and_padding_outputs_are_explicit(self):
        first = _target(
            "E", PrimitiveFamily.CIRCLE, 1.0, GeometryEncoding.CONTINUOUS
        )
        second = _target(
            "EE",
            PrimitiveFamily.CAPSULE_LINE_ARC,
            3.0,
            GeometryEncoding.CONTINUOUS,
        )
        batch = _collate_targets((first, second))
        extracted = extract_profile_target_tensors(batch)
        result = canonicalize_profile_tensors(
            extracted.family_ids,
            extracted.parameters,
            extracted.sketch_mask,
        )
        none_id = PRIMITIVE_TYPES.id(None)
        for batch_index in range(len(batch.node_mask)):
            for node_index in range(len(batch.node_mask[batch_index])):
                is_sketch = extracted.sketch_mask[
                    batch_index, node_index
                ].item()
                if is_sketch:
                    continue
                self.assertEqual(
                    extracted.family_ids[batch_index, node_index].item(),
                    NO_PROFILE_FAMILY_ID,
                )
                self.assertEqual(
                    result.primitive_type_ids[
                        batch_index, node_index
                    ].tolist(),
                    [none_id] * 4,
                )
                self.assertFalse(
                    result.geometry_mask[batch_index, node_index].any()
                )
                self.assertFalse(
                    result.geometry[batch_index, node_index].any()
                )

    def test_supported_wrong_family_remains_internally_consistent(self):
        family_ids = torch.tensor(
            (profile_family_id(PrimitiveFamily.CAPSULE_LINE_ARC),),
            dtype=torch.long,
        )
        raw = torch.zeros((1, 3))
        mask = torch.tensor((True,), dtype=torch.bool)
        parameters = constrain_profile_parameters(raw, family_ids, mask)
        result = canonicalize_profile_tensors(
            family_ids, parameters, mask
        )
        self.assertEqual(
            result.primitive_type_ids[0].tolist(),
            list(
                canonical_primitive_type_ids(
                    PrimitiveFamily.CAPSULE_LINE_ARC
                )
            ),
        )
        self.assertEqual(result.geometry_mask[0].sum().item(), 20)

    def test_unsupported_family_ids_fail_explicitly(self):
        raw = torch.zeros((1, 3))
        mask = torch.tensor((True,), dtype=torch.bool)
        for family_id in (-1, len(PROFILE_FAMILIES)):
            with self.subTest(family_id=family_id):
                ids = torch.tensor((family_id,), dtype=torch.long)
                with self.assertRaisesRegex(ValueError, "unsupported"):
                    constrain_profile_parameters(raw, ids, mask)
                with self.assertRaisesRegex(ValueError, "unsupported"):
                    canonicalize_profile_tensors(ids, raw, mask)

    def test_target_extraction_rejects_out_of_domain_and_noncanonical_rows(self):
        target = _target(
            "E", PrimitiveFamily.CIRCLE, 1.0, GeometryEncoding.CONTINUOUS
        )
        index = target.node_type_ids.index(NODE_TYPES.id("sketch"))
        rows = list(target.geometry)
        row = list(rows[index])
        row[11] = 0.01
        rows[index] = tuple(row)
        with self.assertRaises(ValueError):
            extract_profile_target_tensors(
                replace(target, geometry=tuple(rows))
            )

        masks = list(target.geometry_mask)
        row_mask = list(masks[index])
        row_mask[9] = False
        masks[index] = tuple(row_mask)
        with self.assertRaisesRegex(ValueError, "mask"):
            extract_profile_target_tensors(
                replace(target, geometry_mask=tuple(masks))
            )

        rows = list(target.geometry)
        row = list(rows[index])
        row[11] = math.nan
        rows[index] = tuple(row)
        with self.assertRaisesRegex(ValueError, "finite"):
            extract_profile_target_tensors(
                replace(target, geometry=tuple(rows))
            )

        rows = list(target.geometry)
        row = list(rows[index])
        row[11] = 0.0
        rows[index] = tuple(row)
        with self.assertRaisesRegex(ValueError, "positive"):
            extract_profile_target_tensors(
                replace(target, geometry=tuple(rows))
            )

        attributes = list(target.categorical_attributes)
        row_attributes = list(attributes[index])
        row_attributes[4:8] = (
            PRIMITIVE_TYPES.id("line"),
            PRIMITIVE_TYPES.id("arc"),
            PRIMITIVE_TYPES.id("line"),
            PRIMITIVE_TYPES.id("arc"),
        )
        attributes[index] = tuple(row_attributes)
        with self.assertRaisesRegex(ValueError, "unsupported"):
            extract_profile_target_tensors(
                replace(target, categorical_attributes=tuple(attributes))
            )

    def test_output_shapes_dtypes_and_masks_are_exact(self):
        raw = torch.zeros((2, 3, 3), dtype=torch.float32)
        family_ids = torch.tensor(
            ((0, 1, 2), (2, 1, 0)), dtype=torch.long
        )
        mask = torch.ones((2, 3), dtype=torch.bool)
        parameters = constrain_profile_parameters(raw, family_ids, mask)
        result = canonicalize_profile_tensors(
            family_ids, parameters, mask
        )
        self.assertEqual(result.primitive_type_ids.shape, (2, 3, 4))
        self.assertEqual(result.geometry.shape, (2, 3, GEOMETRY_WIDTH))
        self.assertEqual(
            result.geometry_mask.shape, (2, 3, GEOMETRY_WIDTH)
        )
        self.assertEqual(result.primitive_type_ids.dtype, torch.long)
        self.assertEqual(result.geometry.dtype, torch.float32)
        self.assertEqual(result.geometry_mask.dtype, torch.bool)
        expected_counts = torch.tensor((3, 16, 20))
        self.assertTrue(
            torch.equal(result.geometry_mask[0].sum(dim=-1), expected_counts)
        )


class TensorSourceCompatibilityTests(unittest.TestCase):
    def test_python_38_grammar_and_pytorch_111_operations(self):
        root = Path(__file__).resolve().parents[2]
        paths = (
            root / "profile_geometry.py",
            root / "profile_geometry_torch.py",
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


if __name__ == "__main__":
    unittest.main()
