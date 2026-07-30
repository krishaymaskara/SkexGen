"""Focused tests for the shared controlled-profile geometry contract."""

from __future__ import annotations

from dataclasses import replace
import math
import unittest

from prototype.controlled_data.builders import build_history
from prototype.controlled_data.config import GeneratorConfig
from prototype.controlled_data.factors import PrimitiveFamily
from prototype.model_data.canonical import (
    canonical_nodes_and_edges,
    reconstruction_target,
)
from prototype.model_data.geometry import (
    GEOMETRY_WIDTH,
    LENGTH_SCALE,
    NORMALIZED_MAX,
)
from prototype.model_data.tests.fixtures import source
from prototype.model_data.vocab import NODE_TYPES, PRIMITIVE_TYPES
from prototype.profile_geometry import (
    PROFILE_EXTENT_MAX,
    PROFILE_EXTENT_MIN,
    PROFILE_FAMILIES,
    PROFILE_TEMPLATES,
    canonical_primitive_type_ids,
    canonical_primitive_type_ids_from_family_id,
    canonical_profile_template,
    construct_profile_geometry,
    extract_profile_parameters,
    profile_family_from_id,
    profile_family_from_primitive_type_ids,
    profile_family_id,
    profile_family_id_from_primitive_type_ids,
)
from prototype.representation.model import GeometryEncoding


def _target(family, extent=2.0, encoding=GeometryEncoding.CONTINUOUS):
    history = build_history(
        source("E", family, extents=(extent,)),
        encoding,
    )
    nodes, edges = canonical_nodes_and_edges(history)
    return reconstruction_target(
        nodes, edges, history.structure.operation_sequence
    )


def _sketch_index(target):
    return target.node_type_ids.index(NODE_TYPES.id("sketch"))


def _slot(values, slot, width):
    start = 9 + slot * 6
    return values[start:start + width]


class ProfileGeometryTests(unittest.TestCase):
    def test_profile_family_class_order_and_extent_bounds_are_frozen(self):
        self.assertEqual(
            PROFILE_FAMILIES,
            (
                PrimitiveFamily.CIRCLE,
                PrimitiveFamily.RECTANGLE_LINES,
                PrimitiveFamily.CAPSULE_LINE_ARC,
            ),
        )
        self.assertEqual(PROFILE_EXTENT_MIN, 0.25)
        self.assertEqual(PROFILE_EXTENT_MAX, 0.75)
        physical_extents = (
            GeneratorConfig.__dataclass_fields__["sketch_extents"].default
        )
        self.assertEqual(
            PROFILE_EXTENT_MIN, min(physical_extents) / LENGTH_SCALE
        )
        self.assertEqual(
            PROFILE_EXTENT_MAX, max(physical_extents) / LENGTH_SCALE
        )
        self.assertEqual(
            tuple(template.family for template in PROFILE_TEMPLATES),
            PROFILE_FAMILIES,
        )

    def test_exact_primitive_family_mappings(self):
        none = PRIMITIVE_TYPES.id(None)
        arc = PRIMITIVE_TYPES.id("arc")
        circle = PRIMITIVE_TYPES.id("circle")
        line = PRIMITIVE_TYPES.id("line")
        expected = {
            PrimitiveFamily.CIRCLE: (circle, none, none, none),
            PrimitiveFamily.RECTANGLE_LINES: (line, line, line, line),
            PrimitiveFamily.CAPSULE_LINE_ARC: (arc, arc, line, line),
        }
        for family, pattern in expected.items():
            with self.subTest(family=family.value):
                self.assertEqual(canonical_primitive_type_ids(family), pattern)
                self.assertIs(
                    profile_family_from_primitive_type_ids(pattern), family
                )
                class_id = profile_family_id(family)
                self.assertIs(profile_family_from_id(class_id), family)
                self.assertEqual(
                    profile_family_id_from_primitive_type_ids(pattern),
                    class_id,
                )
                self.assertEqual(
                    canonical_primitive_type_ids_from_family_id(class_id),
                    pattern,
                )

    def test_templates_are_complete_and_drive_python_construction(self):
        for family in PROFILE_FAMILIES:
            template = canonical_profile_template(family)
            with self.subTest(family=family.value):
                self.assertEqual(
                    len(template.geometry_coefficients), GEOMETRY_WIDTH
                )
                self.assertEqual(len(template.geometry_bias), GEOMETRY_WIDTH)
                self.assertEqual(len(template.geometry_mask), GEOMETRY_WIDTH)
                self.assertEqual(
                    template.primitive_type_ids,
                    canonical_primitive_type_ids(family),
                )
                values, mask = construct_profile_geometry(
                    family, 0.3, -0.1, 0.5
                )
                expected = tuple(
                    (
                        coefficients[0] * 0.3
                        + coefficients[1] * -0.1
                        + coefficients[2] * 0.5
                        + bias
                    )
                    if applicable else 0.0
                    for coefficients, bias, applicable in zip(
                        template.geometry_coefficients,
                        template.geometry_bias,
                        template.geometry_mask,
                    )
                )
                self.assertEqual(values, expected)
                self.assertEqual(mask, template.geometry_mask)

    def test_circle_center_radius_and_mask(self):
        values, mask = construct_profile_geometry(
            PrimitiveFamily.CIRCLE, 0.4, -0.2, 0.6
        )
        self.assertEqual(_slot(values, 0, 3), (0.4, -0.2, 0.3))
        self.assertEqual(sum(mask), 3)
        self.assertEqual(mask[9:12], (True, True, True))
        self.assertTrue(all(value == 0.0 for value in values[12:]))

    def test_rectangle_is_connected_and_closed(self):
        values, mask = construct_profile_geometry(
            PrimitiveFamily.RECTANGLE_LINES, 0.25, 0.0, 0.5
        )
        lines = tuple(_slot(values, slot, 4) for slot in range(4))
        for index, line in enumerate(lines):
            self.assertEqual(line[2:4], lines[(index + 1) % 4][0:2])
        self.assertEqual(sum(mask), 16)
        self.assertEqual(
            lines,
            (
                (0.0, -0.125, 0.5, -0.125),
                (0.5, -0.125, 0.5, 0.125),
                (0.5, 0.125, 0.0, 0.125),
                (0.0, 0.125, 0.0, -0.125),
            ),
        )

    def test_capsule_is_connected_and_closed_in_builder_loop_order(self):
        values, mask = construct_profile_geometry(
            PrimitiveFamily.CAPSULE_LINE_ARC, 0.375, 0.0, 0.5
        )
        right_arc = _slot(values, 0, 6)
        left_arc = _slot(values, 1, 6)
        lower_line = _slot(values, 2, 4)
        upper_line = _slot(values, 3, 4)
        traversal = (lower_line, right_arc, upper_line, left_arc)
        for index, primitive in enumerate(traversal):
            self.assertEqual(
                primitive[-2:], traversal[(index + 1) % 4][0:2]
            )
        self.assertEqual(sum(mask), 20)
        self.assertEqual(right_arc, (0.5, -0.125, 0.625, 0.0, 0.5, 0.125))
        self.assertEqual(left_arc, (0.25, 0.125, 0.125, 0.0, 0.25, -0.125))

    def test_all_generator_targets_extract_and_reconstruct_exactly(self):
        physical_extents = (
            GeneratorConfig.__dataclass_fields__["sketch_extents"].default
        )
        for encoding in GeometryEncoding:
            for family in PROFILE_FAMILIES:
                for physical_extent in physical_extents:
                    with self.subTest(
                        encoding=encoding.value,
                        family=family.value,
                        extent=physical_extent,
                    ):
                        target = _target(family, physical_extent, encoding)
                        index = _sketch_index(target)
                        parameters = extract_profile_parameters(target, index)
                        self.assertIs(parameters.family, family)
                        self.assertEqual(
                            parameters.extent, physical_extent / LENGTH_SCALE
                        )
                        self.assertEqual(
                            parameters.center_x,
                            (0.5 + physical_extent / 2.0) / LENGTH_SCALE,
                        )
                        self.assertEqual(parameters.center_y, 0.0)
                        self.assertEqual(
                            construct_profile_geometry(
                                parameters.family,
                                parameters.center_x,
                                parameters.center_y,
                                parameters.extent,
                            ),
                            (
                                target.geometry[index],
                                target.geometry_mask[index],
                            ),
                        )

    def test_positive_extent_is_enforced(self):
        for extent in (0.0, -0.1):
            with self.subTest(extent=extent):
                with self.assertRaisesRegex(ValueError, "strictly positive"):
                    construct_profile_geometry(
                        PrimitiveFamily.CIRCLE, 0.0, 0.0, extent
                    )

    def test_nonfinite_parameters_are_rejected(self):
        for name, values in (
            ("center_x", (math.nan, 0.0, 0.5)),
            ("center_y", (0.0, math.inf, 0.5)),
            ("extent", (0.0, 0.0, -math.inf)),
        ):
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, "finite"):
                    construct_profile_geometry(
                        PrimitiveFamily.RECTANGLE_LINES, *values
                    )

    def test_malformed_and_unsupported_primitive_patterns_are_rejected(self):
        line = PRIMITIVE_TYPES.id("line")
        arc = PRIMITIVE_TYPES.id("arc")
        cases = (
            (),
            (line, line, line),
            (line, arc, line, arc),
            (line, line, line, PRIMITIVE_TYPES.id(None)),
            (line, line, line, True),
        )
        for pattern in cases:
            with self.subTest(pattern=pattern):
                with self.assertRaises(ValueError):
                    profile_family_from_primitive_type_ids(pattern)

    def test_geometry_bounds_are_enforced_without_clipping(self):
        with self.assertRaisesRegex(ValueError, "outside normalized bounds"):
            construct_profile_geometry(
                PrimitiveFamily.CIRCLE, NORMALIZED_MAX + 0.01, 0.0, 0.1
            )
        with self.assertRaisesRegex(ValueError, "outside normalized bounds"):
            construct_profile_geometry(
                PrimitiveFamily.CIRCLE, 0.0, 0.0, NORMALIZED_MAX + 0.01
            )
        values, _ = construct_profile_geometry(
            PrimitiveFamily.CIRCLE, 0.95, 0.0, 0.1
        )
        self.assertEqual(values[9], 0.95)
        self.assertEqual(values[11], 0.05)

    def test_extraction_rejects_wrong_mask_geometry_and_nonfinite_values(self):
        target = _target(PrimitiveFamily.RECTANGLE_LINES)
        index = _sketch_index(target)

        masks = list(target.geometry_mask)
        row_mask = list(masks[index])
        row_mask[9] = False
        masks[index] = tuple(row_mask)
        with self.assertRaisesRegex(ValueError, "mask is not canonical"):
            extract_profile_parameters(
                replace(target, geometry_mask=tuple(masks)), index
            )

        rows = list(target.geometry)
        row = list(rows[index])
        row[11] += 0.01
        rows[index] = tuple(row)
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            extract_profile_parameters(
                replace(target, geometry=tuple(rows)), index
            )

        row[11] = math.nan
        rows[index] = tuple(row)
        with self.assertRaisesRegex(ValueError, "finite"):
            extract_profile_parameters(
                replace(target, geometry=tuple(rows)), index
            )

    def test_all_outputs_preserve_full_geometry_layout_and_exact_masks(self):
        expected_counts = {
            PrimitiveFamily.CIRCLE: 3,
            PrimitiveFamily.RECTANGLE_LINES: 16,
            PrimitiveFamily.CAPSULE_LINE_ARC: 20,
        }
        for family, count in expected_counts.items():
            values, mask = construct_profile_geometry(
                family, 0.25, 0.0, 0.5
            )
            with self.subTest(family=family.value):
                self.assertEqual(len(values), GEOMETRY_WIDTH)
                self.assertEqual(len(mask), GEOMETRY_WIDTH)
                self.assertEqual(sum(mask), count)
                self.assertTrue(
                    all(
                        value == 0.0
                        for value, present in zip(values, mask)
                        if not present
                    )
                )


if __name__ == "__main__":
    unittest.main()
