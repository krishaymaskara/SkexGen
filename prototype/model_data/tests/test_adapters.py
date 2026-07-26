"""Shared-target, geometry-mask, and tensor adapter tests."""

from __future__ import annotations

import math
import tempfile
import unittest

from prototype.controlled_data.factors import PrimitiveFamily, ReferencePlane
from prototype.model_data.adapters import adapt_flat_mixed, adapt_typed_graph
from prototype.model_data.errors import ModelDataError
from prototype.model_data.geometry import (
    ANGLE_SCALE,
    GEOMETRY_CHANNEL_SCALES,
    GEOMETRY_WIDTH,
    LENGTH_SCALE,
    denormalize_applicable_geometry,
    place_node_geometry,
    place_primitive_geometry,
)
from prototype.model_data.loader import load_physical_examples
from prototype.model_data.serialization import canonical_record_json
from prototype.model_data.tests.fixtures import source, write_physical_corpus
from prototype.model_data.vocab import BOOLEAN_MODES, DIRECTIONS
from prototype.representation.model import (
    BooleanMode,
    Direction,
    ExtrudeGeometry,
    LineGeometry,
    NumericValue,
    RevolveGeometry,
)


class _FakeTensor:
    def __init__(self, value, dtype):
        self.value = value
        self.dtype = dtype
        self.shape = _shape(value)
        self.contiguous_calls = 0

    def contiguous(self):
        self.contiguous_calls += 1
        return self

    def __eq__(self, other):
        return (
            isinstance(other, _FakeTensor)
            and self.value == other.value
            and self.dtype == other.dtype
            and self.shape == other.shape
            and self.contiguous_calls == other.contiguous_calls
        )


class _FakeTorch:
    long = "long"
    float32 = "float32"
    bool = "bool"

    @staticmethod
    def tensor(value, dtype):
        return _FakeTensor(value, dtype)


def _shape(value):
    result = []
    current = value
    while isinstance(current, (tuple, list)):
        result.append(len(current))
        if not current:
            break
        current = current[0]
    return tuple(result)


class AdapterTests(unittest.TestCase):
    def _examples(self):
        temporary = tempfile.TemporaryDirectory()
        write_physical_corpus(
            temporary.name,
            (
                source("E", PrimitiveFamily.RECTANGLE_LINES, ReferencePlane.XY),
                source("R", PrimitiveFamily.CIRCLE, ReferencePlane.XZ),
                source("ER", PrimitiveFamily.CAPSULE_LINE_ARC, ReferencePlane.YZ),
            ),
        )
        return temporary, load_physical_examples(temporary.name)

    def test_flat_and_graph_share_exact_reconstruction_target(self):
        temporary, examples = self._examples()
        self.addCleanup(temporary.cleanup)
        for physical in examples:
            flat = adapt_flat_mixed(physical)
            graph = adapt_typed_graph(physical)
            self.assertIs(flat.target, physical.target)
            self.assertIs(graph.target, physical.target)
            self.assertEqual(flat.geometry, graph.geometry)
            self.assertEqual(flat.geometry_mask, graph.geometry_mask)
            self.assertEqual(
                tuple(row[0] for row in flat.categorical_ids), graph.node_type_ids
            )
            self.assertFalse(hasattr(flat, "edge_index"))

    def test_geometry_masks_cover_every_supported_primitive_and_operation(self):
        temporary, examples = self._examples()
        self.addCleanup(temporary.cleanup)
        expected_sketch_counts = {
            "rectangle_lines": 16,
            "circle": 3,
            "capsule_line_arc": 20,
        }
        seen_operations = set()
        for example in examples:
            for node in example.nodes:
                count = sum(node.geometry_mask)
                if node.node_type == "reference_plane":
                    self.assertEqual(count, 9)
                elif node.node_type == "sketch":
                    self.assertEqual(
                        count, expected_sketch_counts[example.metadata.primitive_family]
                    )
                elif node.node_type in ("profile",):
                    self.assertEqual(count, 0)
                elif node.node_type == "axis":
                    self.assertEqual(count, 4)
                elif node.node_type == "extrude":
                    seen_operations.add("extrude")
                    self.assertEqual(count, 1)
                elif node.node_type == "revolve":
                    seen_operations.add("revolve")
                    self.assertEqual(count, 1)
                else:
                    self.fail("unexpected node type %s" % node.node_type)
                self.assertEqual(len(node.geometry), GEOMETRY_WIDTH)
        self.assertEqual(seen_operations, {"extrude", "revolve"})

    def test_tensorization_is_deterministic_and_uses_plain_tensor_calls(self):
        temporary, examples = self._examples()
        self.addCleanup(temporary.cleanup)
        flat = adapt_flat_mixed(examples[0])
        graph = adapt_typed_graph(examples[0])
        flat_tensors = flat.to_torch(_FakeTorch)
        first = graph.to_torch(_FakeTorch)
        second = graph.to_torch(_FakeTorch)
        target = graph.target.to_torch(_FakeTorch)
        self.assertEqual(first, second)
        for name in (
            "node_type_ids",
            "edge_index",
            "edge_type_ids",
            "categorical_attributes",
            "operation_sequence",
        ):
            self.assertEqual(first[name].dtype, "long")
        for name in ("geometry",):
            self.assertEqual(first[name].dtype, "float32")
        for name in ("geometry_mask",):
            self.assertEqual(first[name].dtype, "bool")
        self.assertEqual(flat_tensors["categorical_ids"].dtype, "long")
        self.assertEqual(flat_tensors["geometry"].dtype, "float32")
        self.assertEqual(flat_tensors["geometry_mask"].dtype, "bool")
        self.assertEqual(target["boolean_mode_targets"].dtype, "long")
        self.assertEqual(target["geometry_mask"].dtype, "bool")
        self.assertEqual(first["edge_index"].shape[0], 2)
        self.assertEqual(first["geometry"].shape[1], GEOMETRY_WIDTH)
        self.assertTrue(
            all(
                item.contiguous_calls == 1
                for collection in (flat_tensors, first, target)
                for item in collection.values()
            )
        )

    def test_normalization_boundaries_zero_masks_and_signed_categories(self):
        values = [0.0] * GEOMETRY_WIDTH
        mask = [False] * GEOMETRY_WIDTH
        place_primitive_geometry(
            values,
            mask,
            0,
            LineGeometry(
                NumericValue.continuous(-4.0, 0.0),
                NumericValue.continuous(4.0, 0.0),
            ),
        )
        self.assertEqual(tuple(values[9:13]), (-1.0, 0.0, 1.0, 0.0))
        self.assertEqual(tuple(mask[9:13]), (True, True, True, True))
        self.assertFalse(mask[13])
        values = [0.0] * GEOMETRY_WIDTH
        mask = [False] * GEOMETRY_WIDTH
        place_node_geometry(
            values,
            mask,
            ExtrudeGeometry(NumericValue.continuous(4.0)),
        )
        place_node_geometry(
            values,
            mask,
            RevolveGeometry(NumericValue.continuous(360.0)),
        )
        self.assertEqual(values[37:39], [1.0, 1.0])
        self.assertEqual(mask[37:39], [True, True])

        with tempfile.TemporaryDirectory() as temporary:
            write_physical_corpus(
                temporary,
                (
                    source(
                        "EE",
                        directions=(Direction.NEGATIVE, Direction.POSITIVE),
                        mode=BooleanMode.CUT,
                    ),
                ),
            )
            example = load_physical_examples(temporary)[0]
        operation_nodes = [
            item for item in example.nodes if item.operation_type is not None
        ]
        self.assertEqual(
            tuple(DIRECTIONS.id(item.direction) for item in operation_nodes),
            (
                DIRECTIONS.id(Direction.NEGATIVE.value),
                DIRECTIONS.id(Direction.POSITIVE.value),
            ),
        )
        self.assertEqual(
            tuple(BOOLEAN_MODES.id(item.boolean_mode) for item in operation_nodes),
            (
                BOOLEAN_MODES.id(BooleanMode.NEW_BODY.value),
                BOOLEAN_MODES.id(BooleanMode.CUT.value),
            ),
        )

    def test_out_of_range_geometry_is_rejected_without_clipping(self):
        values = [0.0] * GEOMETRY_WIDTH
        mask = [False] * GEOMETRY_WIDTH
        with self.assertRaises(ModelDataError) as caught:
            place_primitive_geometry(
                values,
                mask,
                0,
                LineGeometry(
                    NumericValue.continuous(-4.000001, 0.0),
                    NumericValue.continuous(4.0, 0.0),
                ),
            )
        self.assertEqual(caught.exception.code, "geometry_out_of_range")
        self.assertNotIn(-1.00000025, values)

        with tempfile.TemporaryDirectory() as temporary:
            write_physical_corpus(
                temporary, (source("E", extents=(4.0,)),)
            )
            with self.assertRaises(ModelDataError) as caught:
                load_physical_examples(temporary)
        self.assertEqual(caught.exception.code, "geometry_out_of_range")

    def test_inverse_normalization_boundaries_scales_and_signed_zero(self):
        values = [-1.0, 0.0, 1.0] * 13
        mask = [True] * GEOMETRY_WIDTH
        physical = denormalize_applicable_geometry(values, mask)
        for index, (normalized, scale) in enumerate(
            zip(values, GEOMETRY_CHANNEL_SCALES)
        ):
            self.assertEqual(physical[index], normalized * scale)
        self.assertIn(LENGTH_SCALE, GEOMETRY_CHANNEL_SCALES)
        self.assertIn(ANGLE_SCALE, GEOMETRY_CHANNEL_SCALES)
        self.assertIn(1.0, GEOMETRY_CHANNEL_SCALES)

        signed = [0.0] * GEOMETRY_WIDTH
        signed[0] = -0.0
        result = denormalize_applicable_geometry(signed, mask)
        self.assertEqual(result[0], 0.0)
        self.assertEqual(math.copysign(1.0, result[0]), 1.0)

    def test_inverse_normalization_round_trip_and_masked_absence(self):
        values = [
            ((index % 9) - 4) / 4.0 for index in range(GEOMETRY_WIDTH)
        ]
        mask = [(index % 3) != 0 for index in range(GEOMETRY_WIDTH)]
        physical = denormalize_applicable_geometry(values, mask)
        reconstructed = tuple(
            None if item is None else item / GEOMETRY_CHANNEL_SCALES[index]
            for index, item in enumerate(physical)
        )
        self.assertEqual(
            reconstructed,
            tuple(value if present else None for value, present in zip(values, mask)),
        )
        values[0] = float("nan")
        values[4] = 0.0
        physical = denormalize_applicable_geometry(values, mask)
        self.assertIsNone(physical[0])
        self.assertEqual(physical[4], 0.0)

    def test_inverse_normalization_rejects_invalid_inputs_deterministically(self):
        valid_values = [0.0] * GEOMETRY_WIDTH
        valid_mask = [True] * GEOMETRY_WIDTH
        cases = (
            (valid_values[:-1], valid_mask, "invalid_geometry_width"),
            (valid_values, valid_mask[:-1], "invalid_geometry_mask_width"),
            (valid_values, [1] + valid_mask[1:], "invalid_geometry_mask_type"),
            (
                [float("nan")] + valid_values[1:],
                valid_mask,
                "nonfinite_geometry",
            ),
            (
                [float("inf")] + valid_values[1:],
                valid_mask,
                "nonfinite_geometry",
            ),
            ([1.000001] + valid_values[1:], valid_mask, "geometry_out_of_range"),
            ([-1.000001] + valid_values[1:], valid_mask, "geometry_out_of_range"),
        )
        for values, mask, code in cases:
            details = []
            for _ in range(2):
                with self.assertRaises(ModelDataError) as caught:
                    denormalize_applicable_geometry(values, mask)
                self.assertEqual(caught.exception.code, code)
                details.append(caught.exception.detail)
            self.assertEqual(details[0], details[1])

    def test_inverse_normalization_repeatability(self):
        values = tuple((index % 5) / 4.0 for index in range(GEOMETRY_WIDTH))
        mask = tuple(index % 2 == 0 for index in range(GEOMETRY_WIDTH))
        self.assertEqual(
            denormalize_applicable_geometry(values, mask),
            denormalize_applicable_geometry(values, mask),
        )

    def test_current_controlled_grid_extrema_fit_normalization_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            write_physical_corpus(
                temporary,
                (
                    source(
                        "ER",
                        PrimitiveFamily.RECTANGLE_LINES,
                        extents=(3.0, 3.0),
                        parameters=(3.0, 360.0),
                    ),
                    source(
                        "RE",
                        PrimitiveFamily.CAPSULE_LINE_ARC,
                        extents=(3.0, 3.0),
                        parameters=(360.0, 3.0),
                    ),
                    source(
                        "R",
                        PrimitiveFamily.CIRCLE,
                        extents=(3.0,),
                        parameters=(360.0,),
                    ),
                ),
            )
            examples = load_physical_examples(temporary)
        applicable = [
            value
            for example in examples
            for node in example.nodes
            for value, present in zip(node.geometry, node.geometry_mask)
            if present
        ]
        self.assertTrue(applicable)
        self.assertGreaterEqual(min(applicable), -1.0)
        self.assertLessEqual(max(applicable), 1.0)

    def test_adapter_records_are_deterministically_serializable(self):
        temporary, examples = self._examples()
        self.addCleanup(temporary.cleanup)
        for example in examples:
            self.assertEqual(
                canonical_record_json(adapt_flat_mixed(example)),
                canonical_record_json(adapt_flat_mixed(example)),
            )
            self.assertEqual(
                canonical_record_json(adapt_typed_graph(example)),
                canonical_record_json(adapt_typed_graph(example)),
            )


if __name__ == "__main__":
    unittest.main()
