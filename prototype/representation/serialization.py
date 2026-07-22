"""Strict deterministic JSON serialization for schema version 1."""

from __future__ import annotations

import json
from typing import Any

from .model import (
    ArcGeometry,
    AxisGeometry,
    AxisNode,
    BooleanMode,
    CADHistory,
    CircleGeometry,
    Direction,
    Edge,
    EdgeType,
    ExtrudeGeometry,
    ExtrudeNode,
    GeometryEncoding,
    GeometryStore,
    LineGeometry,
    LoopRole,
    NodeGeometryRecord,
    NodeType,
    NumericValue,
    PlaneGeometry,
    PrimitiveType,
    ProfileNode,
    ReferencePlaneNode,
    RevolveGeometry,
    RevolveNode,
    SketchElementGeometryRecord,
    SketchLoop,
    SketchNode,
    SketchPrimitive,
    StructureGraph,
)
from .validation import validate_history


class SerializationError(ValueError):
    """A strict JSON field, enum, or type violation."""

    def __init__(self, path: str, message: str):
        self.path = path
        self.message = message
        super().__init__(f"{path}: {message}")


def history_to_json(history: CADHistory) -> str:
    """Return canonical JSON after validating the complete history."""

    validate_history(history)
    return json.dumps(
        _history_to_dict(history),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def history_from_json(payload: str) -> CADHistory:
    """Parse strict version 1 JSON and validate the resulting history."""

    if not isinstance(payload, str):
        raise SerializationError("$", "JSON payload must be a string")
    try:
        data = json.loads(
            payload,
            object_pairs_hook=_reject_duplicate_fields,
            parse_constant=_reject_nonfinite_constant,
        )
    except json.JSONDecodeError as exc:
        raise SerializationError("$", f"malformed JSON: {exc.msg}") from exc
    root = _object(data, "$", {"schema_version", "structure", "geometry"})
    version = _integer(root["schema_version"], "$.schema_version")
    structure_data = _object(root["structure"], "$.structure", {"nodes", "edges", "operation_sequence"})
    geometry_data = _object(root["geometry"], "$.geometry", {"node_geometry", "sketch_element_geometry"})

    nodes = tuple(_node_from_dict(item, f"$.structure.nodes[{i}]") for i, item in enumerate(_list(structure_data["nodes"], "$.structure.nodes")))
    edges = tuple(_edge_from_dict(item, f"$.structure.edges[{i}]") for i, item in enumerate(_list(structure_data["edges"], "$.structure.edges")))
    operation_sequence = tuple(
        _string(item, f"$.structure.operation_sequence[{i}]")
        for i, item in enumerate(_list(structure_data["operation_sequence"], "$.structure.operation_sequence"))
    )
    node_geometry = tuple(
        _node_geometry_from_dict(item, f"$.geometry.node_geometry[{i}]")
        for i, item in enumerate(_list(geometry_data["node_geometry"], "$.geometry.node_geometry"))
    )
    element_geometry = tuple(
        _element_geometry_from_dict(item, f"$.geometry.sketch_element_geometry[{i}]")
        for i, item in enumerate(_list(geometry_data["sketch_element_geometry"], "$.geometry.sketch_element_geometry"))
    )
    history = CADHistory(
        version,
        StructureGraph(nodes, edges, operation_sequence),
        GeometryStore(node_geometry, element_geometry),
    )
    validate_history(history)
    return history


def _history_to_dict(history: CADHistory) -> dict[str, Any]:
    nodes = sorted(history.structure.nodes, key=lambda node: node.node_id)
    edges = sorted(
        history.structure.edges,
        key=lambda edge: (edge.source_id, edge.edge_type.value, edge.target_id),
    )
    node_geometry = sorted(history.geometry.node_geometry, key=lambda record: record.node_id)
    element_geometry = sorted(
        history.geometry.sketch_element_geometry,
        key=lambda record: (record.sketch_id, record.element_id),
    )
    return {
        "schema_version": history.schema_version,
        "structure": {
            "nodes": [_node_to_dict(node) for node in nodes],
            "edges": [
                {
                    "source_id": edge.source_id,
                    "edge_type": edge.edge_type.value,
                    "target_id": edge.target_id,
                }
                for edge in edges
            ],
            "operation_sequence": list(history.structure.operation_sequence),
        },
        "geometry": {
            "node_geometry": [
                {
                    "node_id": record.node_id,
                    "geometry": _geometry_to_dict(record.geometry),
                }
                for record in node_geometry
            ],
            "sketch_element_geometry": [
                {
                    "sketch_id": record.sketch_id,
                    "element_id": record.element_id,
                    "geometry": _geometry_to_dict(record.geometry),
                }
                for record in element_geometry
            ],
        },
    }


def _node_to_dict(node) -> dict[str, Any]:
    result: dict[str, Any] = {"node_id": node.node_id, "node_type": node.node_type.value}
    if isinstance(node, SketchNode):
        result["primitives"] = [
            {"primitive_id": item.primitive_id, "primitive_type": item.primitive_type.value}
            for item in sorted(node.primitives, key=lambda item: item.primitive_id)
        ]
        result["loops"] = [
            {"loop_id": item.loop_id, "role": item.role.value, "primitive_ids": list(item.primitive_ids)}
            for item in sorted(node.loops, key=lambda item: item.loop_id)
        ]
    elif isinstance(node, ProfileNode):
        result["outer_loop_id"] = node.outer_loop_id
        result["inner_loop_ids"] = list(node.inner_loop_ids)
    elif isinstance(node, (ExtrudeNode, RevolveNode)):
        result["boolean_mode"] = node.boolean_mode.value
        result["direction"] = node.direction.value
    return result


def _node_from_dict(data: Any, path: str):
    base = _object_at_least(data, path, {"node_id", "node_type"})
    node_id = _string(base["node_id"], f"{path}.node_id")
    node_type = _enum(NodeType, base["node_type"], f"{path}.node_type")
    common = {"node_id", "node_type"}
    if node_type is NodeType.REFERENCE_PLANE:
        _exact_fields(base, path, common)
        return ReferencePlaneNode(node_id)
    if node_type is NodeType.AXIS:
        _exact_fields(base, path, common)
        return AxisNode(node_id)
    if node_type is NodeType.SKETCH:
        _exact_fields(base, path, common | {"primitives", "loops"})
        primitives = tuple(
            _primitive_from_dict(item, f"{path}.primitives[{i}]")
            for i, item in enumerate(_list(base["primitives"], f"{path}.primitives"))
        )
        loops = tuple(
            _loop_from_dict(item, f"{path}.loops[{i}]")
            for i, item in enumerate(_list(base["loops"], f"{path}.loops"))
        )
        return SketchNode(node_id, primitives, loops)
    if node_type is NodeType.PROFILE:
        _exact_fields(base, path, common | {"outer_loop_id", "inner_loop_ids"})
        inner = tuple(
            _string(item, f"{path}.inner_loop_ids[{i}]")
            for i, item in enumerate(_list(base["inner_loop_ids"], f"{path}.inner_loop_ids"))
        )
        return ProfileNode(node_id, _string(base["outer_loop_id"], f"{path}.outer_loop_id"), inner)
    if node_type in (NodeType.EXTRUDE, NodeType.REVOLVE):
        _exact_fields(base, path, common | {"boolean_mode", "direction"})
        mode = _enum(BooleanMode, base["boolean_mode"], f"{path}.boolean_mode")
        direction = _enum(Direction, base["direction"], f"{path}.direction")
        return ExtrudeNode(node_id, mode, direction) if node_type is NodeType.EXTRUDE else RevolveNode(node_id, mode, direction)
    raise SerializationError(f"{path}.node_type", "unsupported node type")


def _primitive_from_dict(data: Any, path: str) -> SketchPrimitive:
    obj = _object(data, path, {"primitive_id", "primitive_type"})
    return SketchPrimitive(
        _string(obj["primitive_id"], f"{path}.primitive_id"),
        _enum(PrimitiveType, obj["primitive_type"], f"{path}.primitive_type"),
    )


def _loop_from_dict(data: Any, path: str) -> SketchLoop:
    obj = _object(data, path, {"loop_id", "role", "primitive_ids"})
    primitive_ids = tuple(
        _string(item, f"{path}.primitive_ids[{i}]")
        for i, item in enumerate(_list(obj["primitive_ids"], f"{path}.primitive_ids"))
    )
    return SketchLoop(
        _string(obj["loop_id"], f"{path}.loop_id"),
        _enum(LoopRole, obj["role"], f"{path}.role"),
        primitive_ids,
    )


def _edge_from_dict(data: Any, path: str) -> Edge:
    obj = _object(data, path, {"source_id", "edge_type", "target_id"})
    return Edge(
        _string(obj["source_id"], f"{path}.source_id"),
        _enum(EdgeType, obj["edge_type"], f"{path}.edge_type"),
        _string(obj["target_id"], f"{path}.target_id"),
    )


def _node_geometry_from_dict(data: Any, path: str) -> NodeGeometryRecord:
    obj = _object(data, path, {"node_id", "geometry"})
    return NodeGeometryRecord(
        _string(obj["node_id"], f"{path}.node_id"),
        _geometry_from_dict(obj["geometry"], f"{path}.geometry"),
    )


def _element_geometry_from_dict(data: Any, path: str) -> SketchElementGeometryRecord:
    obj = _object(data, path, {"sketch_id", "element_id", "geometry"})
    return SketchElementGeometryRecord(
        _string(obj["sketch_id"], f"{path}.sketch_id"),
        _string(obj["element_id"], f"{path}.element_id"),
        _geometry_from_dict(obj["geometry"], f"{path}.geometry"),
    )


def _geometry_to_dict(geometry) -> dict[str, Any]:
    fields_by_type = {
        PlaneGeometry: ("plane", ("origin", "x_axis", "y_axis")),
        LineGeometry: ("line", ("start", "end")),
        ArcGeometry: ("arc", ("start", "midpoint", "end")),
        CircleGeometry: ("circle", ("center", "radius")),
        AxisGeometry: ("axis", ("point", "direction")),
        ExtrudeGeometry: ("extrude", ("distance",)),
        RevolveGeometry: ("revolve", ("angle_degrees",)),
    }
    geometry_type, fields = fields_by_type[type(geometry)]
    result = {"geometry_type": geometry_type}
    result.update({name: _numeric_to_dict(getattr(geometry, name)) for name in fields})
    return result


def _geometry_from_dict(data: Any, path: str):
    obj = _object_at_least(data, path, {"geometry_type"})
    geometry_type = _string(obj["geometry_type"], f"{path}.geometry_type")
    definitions = {
        "plane": (PlaneGeometry, ("origin", "x_axis", "y_axis")),
        "line": (LineGeometry, ("start", "end")),
        "arc": (ArcGeometry, ("start", "midpoint", "end")),
        "circle": (CircleGeometry, ("center", "radius")),
        "axis": (AxisGeometry, ("point", "direction")),
        "extrude": (ExtrudeGeometry, ("distance",)),
        "revolve": (RevolveGeometry, ("angle_degrees",)),
    }
    if geometry_type not in definitions:
        raise SerializationError(f"{path}.geometry_type", f"unknown geometry type {geometry_type!r}")
    cls, fields = definitions[geometry_type]
    _exact_fields(obj, path, {"geometry_type", *fields})
    values = [_numeric_from_dict(obj[name], f"{path}.{name}") for name in fields]
    return cls(*values)


def _numeric_to_dict(value: NumericValue) -> dict[str, Any]:
    values = (
        [_canonical_float(item) for item in value.values]
        if value.encoding is GeometryEncoding.CONTINUOUS
        else list(value.values)
    )
    result: dict[str, Any] = {"encoding": value.encoding.value, "values": values}
    if value.encoding is GeometryEncoding.QUANTIZED:
        result["scale"] = _canonical_float(value.scale)
        result["offset"] = _canonical_float(value.offset)
    return result


def _numeric_from_dict(data: Any, path: str) -> NumericValue:
    obj = _object_at_least(data, path, {"encoding", "values"})
    encoding = _enum(GeometryEncoding, obj["encoding"], f"{path}.encoding")
    raw_values = _list(obj["values"], f"{path}.values")
    if encoding is GeometryEncoding.CONTINUOUS:
        _exact_fields(obj, path, {"encoding", "values"})
        values = tuple(_number(item, f"{path}.values[{i}]") for i, item in enumerate(raw_values))
        return NumericValue(encoding, values)
    _exact_fields(obj, path, {"encoding", "values", "scale", "offset"})
    codes = tuple(_integer(item, f"{path}.values[{i}]") for i, item in enumerate(raw_values))
    return NumericValue(
        encoding,
        codes,
        _number(obj["scale"], f"{path}.scale"),
        _number(obj["offset"], f"{path}.offset"),
    )


def _object(data: Any, path: str, fields: set[str]) -> dict[str, Any]:
    obj = _object_at_least(data, path, fields)
    _exact_fields(obj, path, fields)
    return obj


def _object_at_least(data: Any, path: str, required: set[str]) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise SerializationError(path, "expected an object")
    missing = required - set(data)
    if missing:
        raise SerializationError(path, f"missing required fields {sorted(missing)!r}")
    if not all(isinstance(key, str) for key in data):
        raise SerializationError(path, "object keys must be strings")
    return data


def _exact_fields(data: dict[str, Any], path: str, fields: set[str]) -> None:
    unknown = set(data) - fields
    missing = fields - set(data)
    if missing:
        raise SerializationError(path, f"missing required fields {sorted(missing)!r}")
    if unknown:
        raise SerializationError(path, f"unknown fields {sorted(unknown)!r}")


def _list(data: Any, path: str) -> list[Any]:
    if not isinstance(data, list):
        raise SerializationError(path, "expected a list")
    return data


def _string(data: Any, path: str) -> str:
    if not isinstance(data, str):
        raise SerializationError(path, "expected a string")
    return data


def _integer(data: Any, path: str) -> int:
    if isinstance(data, bool) or not isinstance(data, int):
        raise SerializationError(path, "expected an integer (booleans are not integers)")
    return data


def _number(data: Any, path: str) -> int | float:
    if isinstance(data, bool) or not isinstance(data, (int, float)):
        raise SerializationError(path, "expected a number (booleans are not numbers)")
    return data


def _enum(enum_type, data: Any, path: str):
    value = _string(data, path)
    try:
        return enum_type(value)
    except ValueError as exc:
        raise SerializationError(path, f"unknown {enum_type.__name__} value {value!r}") from exc


def _reject_duplicate_fields(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SerializationError("$", f"duplicate JSON field {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str):
    raise SerializationError("$", f"non-finite JSON constant {value!r} is not allowed")


def _canonical_float(value: int | float) -> float:
    numeric = float(value)
    return 0.0 if numeric == 0.0 else numeric
