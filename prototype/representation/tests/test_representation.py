from __future__ import annotations

from dataclasses import replace
import json
import math
import unittest

from prototype.representation.examples import (
    sketch_extrude,
    sketch_extrude_sketch_extrude,
    sketch_extrude_sketch_revolve,
    sketch_revolve,
)
from prototype.representation.model import (
    ArcGeometry,
    BooleanMode,
    CircleGeometry,
    Edge,
    EdgeType,
    ExtrudeGeometry,
    GeometryEncoding,
    LineGeometry,
    LoopRole,
    NumericValue,
    PrimitiveType,
    ProfileNode,
    RevolveGeometry,
    SketchElementGeometryRecord,
    SketchLoop,
    SketchPrimitive,
)
from prototype.representation.serialization import (
    SerializationError,
    history_from_json,
    history_to_json,
)
from prototype.representation.validation import ValidationError, validate_history


class RepresentationTestCase(unittest.TestCase):
    def assert_invalid(self, history, code: str) -> ValidationError:
        with self.assertRaises(ValidationError) as context:
            validate_history(history)
        self.assertIn(code, {issue.code for issue in context.exception.issues})
        return context.exception

    @staticmethod
    def replace_node_geometry(history, node_id: str, geometry):
        records = tuple(
            replace(record, geometry=geometry) if record.node_id == node_id else record
            for record in history.geometry.node_geometry
        )
        return replace(history, geometry=replace(history.geometry, node_geometry=records))

    @staticmethod
    def replace_element_geometry(history, sketch_id: str, element_id: str, geometry):
        records = tuple(
            replace(record, geometry=geometry)
            if (record.sketch_id, record.element_id) == (sketch_id, element_id)
            else record
            for record in history.geometry.sketch_element_geometry
        )
        return replace(history, geometry=replace(history.geometry, sketch_element_geometry=records))

    @staticmethod
    def replace_sketch_elements(history, sketch_id: str, primitives, loops, geometry_records):
        nodes = tuple(
            replace(node, primitives=primitives, loops=loops) if node.node_id == sketch_id else node
            for node in history.structure.nodes
        )
        retained = tuple(
            record
            for record in history.geometry.sketch_element_geometry
            if record.sketch_id != sketch_id
        )
        return replace(
            history,
            structure=replace(history.structure, nodes=nodes),
            geometry=replace(
                history.geometry,
                sketch_element_geometry=retained + tuple(geometry_records),
            ),
        )


class ValidHistoryTests(RepresentationTestCase):
    def test_all_required_example_histories_validate(self):
        for factory in (
            sketch_extrude,
            sketch_revolve,
            sketch_extrude_sketch_extrude,
            sketch_extrude_sketch_revolve,
        ):
            with self.subTest(factory=factory.__name__):
                validate_history(factory())

    def test_quantized_geometry_is_decoded_before_validation(self):
        history = sketch_extrude()
        distance = NumericValue.quantized((-5,), scale=0.5, offset=3.0)
        history = self.replace_node_geometry(history, "extrude_1", ExtrudeGeometry(distance))
        validate_history(history)
        operation_geometry = next(
            record.geometry for record in history.geometry.node_geometry if record.node_id == "extrude_1"
        )
        self.assertEqual(operation_geometry.distance.encoding, GeometryEncoding.QUANTIZED)
        self.assertEqual(operation_geometry.distance.values, (-5,))

    def test_single_circle_loop_and_profile_inner_loop_validate(self):
        history = sketch_extrude()
        sketch = next(node for node in history.structure.nodes if node.node_id == "sketch_1")
        circle = SketchPrimitive("hole_circle", PrimitiveType.CIRCLE)
        inner = SketchLoop("inner_loop", LoopRole.INNER, ("hole_circle",))
        nodes = tuple(
            replace(node, inner_loop_ids=("inner_loop",))
            if isinstance(node, ProfileNode)
            else replace(node, primitives=node.primitives + (circle,), loops=node.loops + (inner,))
            if node.node_id == "sketch_1"
            else node
            for node in history.structure.nodes
        )
        geometry = history.geometry.sketch_element_geometry + (
            SketchElementGeometryRecord(
                "sketch_1",
                "hole_circle",
                CircleGeometry(NumericValue.continuous(1.0, 0.5), NumericValue.continuous(0.1)),
            ),
        )
        validate_history(
            replace(
                history,
                structure=replace(history.structure, nodes=nodes),
                geometry=replace(history.geometry, sketch_element_geometry=geometry),
            )
        )

    def test_primitive_ids_may_repeat_in_different_sketches(self):
        history = sketch_extrude_sketch_extrude()
        sketches = [node for node in history.structure.nodes if node.node_type.value == "sketch"]
        second_primitives = tuple(replace(p, primitive_id=p.primitive_id.replace("s2_", "s1_")) for p in sketches[1].primitives)
        second_loop = replace(
            sketches[1].loops[0],
            primitive_ids=tuple(item.replace("s2_", "s1_") for item in sketches[1].loops[0].primitive_ids),
        )
        new_sketch = replace(sketches[1], primitives=second_primitives, loops=(second_loop,))
        nodes = tuple(new_sketch if node.node_id == new_sketch.node_id else node for node in history.structure.nodes)
        element_geometry = tuple(
            replace(record, element_id=record.element_id.replace("s2_", "s1_")) if record.sketch_id == "sketch_2" else record
            for record in history.geometry.sketch_element_geometry
        )
        validate_history(
            replace(
                history,
                structure=replace(history.structure, nodes=nodes),
                geometry=replace(history.geometry, sketch_element_geometry=element_geometry),
            )
        )


class SerializationTests(RepresentationTestCase):
    def test_canonical_json_round_trip_is_exact(self):
        for history in (sketch_extrude(), sketch_revolve(), sketch_extrude(quantized_distance=True)):
            payload = history_to_json(history)
            restored = history_from_json(payload)
            self.assertEqual(history_to_json(restored), payload)

    def test_in_memory_order_does_not_change_serialized_output(self):
        history = sketch_extrude()
        reordered = replace(
            history,
            structure=replace(
                history.structure,
                nodes=tuple(reversed(history.structure.nodes)),
                edges=tuple(reversed(history.structure.edges)),
            ),
            geometry=replace(
                history.geometry,
                node_geometry=tuple(reversed(history.geometry.node_geometry)),
                sketch_element_geometry=tuple(reversed(history.geometry.sketch_element_geometry)),
            ),
        )
        self.assertEqual(history_to_json(reordered), history_to_json(history))

    def test_equivalent_numeric_types_have_same_canonical_json(self):
        history = sketch_extrude()
        integer_distance = self.replace_node_geometry(
            history,
            "extrude_1",
            ExtrudeGeometry(NumericValue.continuous(2)),
        )
        self.assertEqual(history_to_json(integer_distance), history_to_json(history))

    def test_unknown_json_field_is_rejected(self):
        data = json.loads(history_to_json(sketch_extrude()))
        data["unknown"] = 1
        with self.assertRaises(SerializationError):
            history_from_json(json.dumps(data))

    def test_unknown_enum_is_rejected(self):
        data = json.loads(history_to_json(sketch_extrude()))
        data["structure"]["nodes"][0]["node_type"] = "unsupported"
        with self.assertRaises(SerializationError):
            history_from_json(json.dumps(data))

    def test_boolean_quantized_code_is_rejected_during_parsing(self):
        data = json.loads(history_to_json(sketch_extrude(quantized_distance=True)))
        extrude = next(item for item in data["geometry"]["node_geometry"] if item["node_id"] == "extrude_1")
        extrude["geometry"]["distance"]["values"] = [True]
        with self.assertRaises(SerializationError):
            history_from_json(json.dumps(data))

    def test_duplicate_json_fields_are_rejected(self):
        with self.assertRaises(SerializationError):
            history_from_json('{"schema_version":1,"schema_version":1,"structure":{},"geometry":{}}')

    def test_nonfinite_json_constants_are_rejected(self):
        payload = history_to_json(sketch_extrude()).replace('"values":[2.0]', '"values":[NaN]', 1)
        with self.assertRaises(SerializationError):
            history_from_json(payload)

    def test_missing_json_field_is_rejected(self):
        data = json.loads(history_to_json(sketch_extrude()))
        del data["structure"]["operation_sequence"]
        with self.assertRaises(SerializationError):
            history_from_json(json.dumps(data))


class DependencyAndStructureTests(RepresentationTestCase):
    def test_every_operation_must_appear_exactly_once(self):
        history = sketch_extrude_sketch_extrude()
        duplicate = replace(history.structure, operation_sequence=("extrude_1", "extrude_1"))
        self.assert_invalid(replace(history, structure=duplicate), "duplicate_operation")

    def test_missing_operation_is_rejected(self):
        history = sketch_extrude_sketch_extrude()
        structure = replace(history.structure, operation_sequence=("extrude_1",))
        self.assert_invalid(replace(history, structure=structure), "missing_operation")

    def test_empty_operation_history_is_rejected(self):
        history = sketch_extrude()
        nodes = tuple(node for node in history.structure.nodes if node.node_id != "extrude_1")
        edges = tuple(edge for edge in history.structure.edges if edge.source_id != "extrude_1")
        node_geometry = tuple(record for record in history.geometry.node_geometry if record.node_id != "extrude_1")
        invalid = replace(
            history,
            structure=replace(history.structure, nodes=nodes, edges=edges, operation_sequence=()),
            geometry=replace(history.geometry, node_geometry=node_geometry),
        )
        self.assert_invalid(invalid, "empty_operation_sequence")

    def test_non_operation_in_sequence_is_rejected(self):
        history = sketch_extrude()
        structure = replace(history.structure, operation_sequence=("sketch_1",))
        self.assert_invalid(replace(history, structure=structure), "non_operation")

    def test_dependency_must_point_to_earlier_operation(self):
        history = sketch_extrude_sketch_extrude()
        sequence = tuple(reversed(history.structure.operation_sequence))
        self.assert_invalid(replace(history, structure=replace(history.structure, operation_sequence=sequence)), "dependency_order")

    def test_dependency_cycle_is_rejected(self):
        history = sketch_extrude_sketch_extrude()
        edges = history.structure.edges + (Edge("extrude_1", EdgeType.DEPENDS_ON, "extrude_2"),)
        self.assert_invalid(replace(history, structure=replace(history.structure, edges=edges)), "dependency_cycle")

    def test_nonexistent_edge_endpoint_is_rejected(self):
        history = sketch_extrude()
        edges = history.structure.edges + (Edge("extrude_1", EdgeType.DEPENDS_ON, "missing"),)
        self.assert_invalid(replace(history, structure=replace(history.structure, edges=edges)), "missing_node")

    def test_later_new_body_is_rejected(self):
        history = sketch_extrude_sketch_extrude()
        nodes = tuple(
            replace(node, boolean_mode=BooleanMode.NEW_BODY) if node.node_id == "extrude_2" else node
            for node in history.structure.nodes
        )
        self.assert_invalid(replace(history, structure=replace(history.structure, nodes=nodes)), "later_operation_mode")

    def test_primitive_and_loop_cannot_share_id_within_sketch(self):
        history = sketch_extrude()
        sketch = next(node for node in history.structure.nodes if node.node_id == "sketch_1")
        loop = replace(sketch.loops[0], loop_id=sketch.primitives[0].primitive_id)
        new_sketch = replace(sketch, loops=(loop,))
        nodes = tuple(new_sketch if node.node_id == "sketch_1" else node for node in history.structure.nodes)
        self.assert_invalid(replace(history, structure=replace(history.structure, nodes=nodes)), "duplicate_element")

    def test_profile_loop_references_are_explicit_and_distinct(self):
        history = sketch_extrude()
        nodes = tuple(
            replace(node, inner_loop_ids=(node.outer_loop_id, node.outer_loop_id)) if isinstance(node, ProfileNode) else node
            for node in history.structure.nodes
        )
        invalid = replace(history, structure=replace(history.structure, nodes=nodes))
        error = self.assert_invalid(invalid, "duplicate_inner_loop")
        self.assertIn("overlapping_loop_reference", {issue.code for issue in error.issues})

    def test_revolve_profile_and_axis_must_share_sketch(self):
        history = sketch_extrude_sketch_revolve()
        edges = tuple(
            replace(edge, target_id="sketch_1")
            if edge.source_id == "axis_2" and edge.edge_type is EdgeType.DEFINED_IN
            else edge
            for edge in history.structure.edges
        )
        self.assert_invalid(replace(history, structure=replace(history.structure, edges=edges)), "revolve_sketch_compatibility")

    def test_invalid_direct_enum_value_is_structured_error(self):
        history = sketch_extrude()
        nodes = tuple(
            replace(node, direction=True) if node.node_id == "extrude_1" else node
            for node in history.structure.nodes
        )
        self.assert_invalid(replace(history, structure=replace(history.structure, nodes=nodes)), "enum_type")

    def test_mutable_collection_is_rejected(self):
        history = sketch_extrude()
        structure = replace(history.structure, operation_sequence=["extrude_1"])
        self.assert_invalid(replace(history, structure=structure), "immutable_collection")

    def test_invalid_direct_primitive_enum_does_not_crash(self):
        history = sketch_extrude()
        sketch = next(node for node in history.structure.nodes if node.node_id == "sketch_1")
        primitive = replace(sketch.primitives[0], primitive_type=[])
        nodes = tuple(
            replace(node, primitives=(primitive,) + node.primitives[1:])
            if node.node_id == "sketch_1"
            else node
            for node in history.structure.nodes
        )
        self.assert_invalid(replace(history, structure=replace(history.structure, nodes=nodes)), "enum_type")


class GeometryValidationTests(RepresentationTestCase):
    def test_open_loop_is_rejected(self):
        history = sketch_extrude()
        geometry = next(
            record.geometry
            for record in history.geometry.sketch_element_geometry
            if record.element_id == "s1_line_0"
        )
        invalid_line = replace(geometry, end=NumericValue.continuous(2.0, 0.25))
        self.assert_invalid(
            self.replace_element_geometry(history, "sketch_1", "s1_line_0", invalid_line),
            "open_loop",
        )

    def test_quantized_loop_coordinates_use_decoded_values(self):
        history = sketch_extrude()
        geometry = next(
            record.geometry
            for record in history.geometry.sketch_element_geometry
            if record.element_id == "s1_line_0"
        )
        closed = replace(geometry, end=NumericValue.quantized((4, 0), scale=0.5))
        validate_history(self.replace_element_geometry(history, "sketch_1", "s1_line_0", closed))
        opened = replace(geometry, end=NumericValue.quantized((4, 1), scale=0.5))
        self.assert_invalid(
            self.replace_element_geometry(history, "sketch_1", "s1_line_0", opened),
            "open_loop",
        )

    def test_two_arc_loop_checks_final_closure(self):
        history = sketch_extrude()
        primitives = (
            SketchPrimitive("arc_1", PrimitiveType.ARC),
            SketchPrimitive("arc_2", PrimitiveType.ARC),
        )
        loops = (SketchLoop("s1_outer", LoopRole.OUTER, ("arc_1", "arc_2")),)
        records = (
            SketchElementGeometryRecord(
                "sketch_1",
                "arc_1",
                ArcGeometry(
                    NumericValue.continuous(0.0, 0.0),
                    NumericValue.continuous(0.5, 0.5),
                    NumericValue.continuous(1.0, 0.0),
                ),
            ),
            SketchElementGeometryRecord(
                "sketch_1",
                "arc_2",
                ArcGeometry(
                    NumericValue.continuous(1.0, 0.0),
                    NumericValue.continuous(0.5, -0.5),
                    NumericValue.continuous(0.0, 0.0),
                ),
            ),
        )
        valid = self.replace_sketch_elements(history, "sketch_1", primitives, loops, records)
        validate_history(valid)
        open_arc = replace(records[1].geometry, end=NumericValue.continuous(0.0, 0.25))
        self.assert_invalid(
            self.replace_element_geometry(valid, "sketch_1", "arc_2", open_arc),
            "open_loop",
        )

    def test_circle_cannot_be_mixed_with_other_loop_primitives(self):
        history = sketch_extrude()
        circle = SketchPrimitive("circle", PrimitiveType.CIRCLE)
        line = SketchPrimitive("line", PrimitiveType.LINE)
        primitives = (circle, line)
        loops = (SketchLoop("s1_outer", LoopRole.OUTER, ("circle", "line")),)
        records = (
            SketchElementGeometryRecord(
                "sketch_1",
                "circle",
                CircleGeometry(NumericValue.continuous(0.0, 0.0), NumericValue.continuous(1.0)),
            ),
            SketchElementGeometryRecord(
                "sketch_1",
                "line",
                LineGeometry(NumericValue.continuous(1.0, 0.0), NumericValue.continuous(-1.0, 0.0)),
            ),
        )
        invalid = self.replace_sketch_elements(history, "sketch_1", primitives, loops, records)
        self.assert_invalid(invalid, "unsupported_loop")

    def test_nonpositive_decoded_quantized_distance_is_rejected(self):
        history = sketch_extrude(quantized_distance=True)
        invalid = ExtrudeGeometry(NumericValue.quantized((0,), scale=0.2))
        self.assert_invalid(self.replace_node_geometry(history, "extrude_1", invalid), "invalid_extrude_distance")

    def test_noninteger_quantized_code_is_rejected(self):
        history = sketch_extrude()
        invalid = ExtrudeGeometry(NumericValue(GeometryEncoding.QUANTIZED, (True,), 1.0, 0.0))
        self.assert_invalid(self.replace_node_geometry(history, "extrude_1", invalid), "quantized_code")

    def test_invalid_quantization_metadata_is_rejected(self):
        history = sketch_extrude()
        invalid = ExtrudeGeometry(NumericValue(GeometryEncoding.QUANTIZED, (1,), 0.0, 0.0))
        self.assert_invalid(self.replace_node_geometry(history, "extrude_1", invalid), "quantized_scale")

    def test_quantized_decode_overflow_is_structured_error(self):
        history = sketch_extrude()
        value = NumericValue.quantized((10**10000,), scale=1.0)
        invalid = ExtrudeGeometry(value)
        self.assert_invalid(self.replace_node_geometry(history, "extrude_1", invalid), "decoded_nonfinite")

    def test_nonfinite_values_are_rejected(self):
        history = sketch_extrude()
        invalid = ExtrudeGeometry(NumericValue.continuous(math.inf))
        self.assert_invalid(self.replace_node_geometry(history, "extrude_1", invalid), "nonfinite_numeric")

    def test_plane_frame_must_be_orthonormal(self):
        history = sketch_extrude()
        geometry = next(record.geometry for record in history.geometry.node_geometry if record.node_id == "plane_1")
        invalid = replace(geometry, y_axis=NumericValue.continuous(1.0, 0.0, 0.0))
        self.assert_invalid(self.replace_node_geometry(history, "plane_1", invalid), "invalid_plane_frame")

    def test_axis_direction_must_be_unit_length(self):
        history = sketch_revolve()
        geometry = next(record.geometry for record in history.geometry.node_geometry if record.node_id == "axis_1")
        invalid = replace(geometry, direction=NumericValue.continuous(0.0, 0.0))
        self.assert_invalid(self.replace_node_geometry(history, "axis_1", invalid), "invalid_axis_direction")

    def test_revolve_angle_range_is_enforced(self):
        history = sketch_revolve()
        invalid = self.replace_node_geometry(
            history,
            "revolve_1",
            RevolveGeometry(NumericValue.continuous(361.0)),
        )
        self.assert_invalid(invalid, "invalid_revolve_angle")

    def test_missing_required_geometry_is_rejected(self):
        history = sketch_extrude()
        records = tuple(record for record in history.geometry.node_geometry if record.node_id != "extrude_1")
        self.assert_invalid(
            replace(history, geometry=replace(history.geometry, node_geometry=records)),
            "missing_geometry",
        )


if __name__ == "__main__":
    unittest.main()
