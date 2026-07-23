"""Build representation histories from controlled physical factors."""

from __future__ import annotations

import math

from prototype.representation.model import (
    ArcGeometry,
    AxisGeometry,
    AxisNode,
    BooleanMode,
    CADHistory,
    CircleGeometry,
    Edge,
    EdgeType,
    ExtrudeGeometry,
    ExtrudeNode,
    GeometryEncoding,
    GeometryStore,
    LineGeometry,
    LoopRole,
    NodeGeometryRecord,
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

from .config import ANGLE_QUANTIZATION_SCALE, LENGTH_QUANTIZATION_SCALE
from .factors import PhysicalSource, PrimitiveFamily, ReferencePlane


def build_history(source: PhysicalSource, encoding: GeometryEncoding) -> CADHistory:
    """Build one schema-v1 variant; representation validation remains authoritative."""

    plane = ReferencePlaneNode("plane_1")
    nodes = [plane]
    edges: list[Edge] = []
    node_geometry = [NodeGeometryRecord("plane_1", _plane_geometry(source.reference_plane, encoding))]
    element_geometry: list[SketchElementGeometryRecord] = []
    operation_ids: list[str] = []

    for index, (operation_kind, extent, direction, parameter) in enumerate(
        zip(
            source.operation_template.operations,
            source.sketch_extents,
            source.directions,
            source.operation_parameters,
        ),
        start=1,
    ):
        sketch_id = f"sketch_{index}"
        profile_id = f"profile_{index}"
        operation_id = f"{operation_kind}_{index}"
        sketch, sketch_geometry = _sketch(sketch_id, source.primitive_family, extent, encoding)
        profile = ProfileNode(profile_id, "outer_loop")
        boolean_mode = BooleanMode.NEW_BODY if index == 1 else source.later_boolean_mode
        if boolean_mode is None:
            raise ValueError("depth-two histories require a later Boolean mode")

        nodes.extend((sketch, profile))
        element_geometry.extend(sketch_geometry)
        edges.extend(
            (
                Edge(sketch_id, EdgeType.PLACED_ON, "plane_1"),
                Edge(profile_id, EdgeType.DEFINED_IN, sketch_id),
            )
        )
        if operation_kind == "extrude":
            operation = ExtrudeNode(operation_id, boolean_mode, direction)
            node_geometry.append(
                NodeGeometryRecord(
                    operation_id,
                    ExtrudeGeometry(_numeric((parameter,), encoding, LENGTH_QUANTIZATION_SCALE)),
                )
            )
        else:
            axis_id = f"axis_{index}"
            axis = AxisNode(axis_id)
            nodes.append(axis)
            edges.extend(
                (
                    Edge(axis_id, EdgeType.DEFINED_IN, sketch_id),
                    Edge(operation_id, EdgeType.USES_AXIS, axis_id),
                )
            )
            node_geometry.append(
                NodeGeometryRecord(
                    axis_id,
                    AxisGeometry(
                        _numeric((0.0, 0.0), encoding, LENGTH_QUANTIZATION_SCALE),
                        _numeric((0.0, 1.0), encoding, LENGTH_QUANTIZATION_SCALE),
                    ),
                )
            )
            operation = RevolveNode(operation_id, boolean_mode, direction)
            node_geometry.append(
                NodeGeometryRecord(
                    operation_id,
                    RevolveGeometry(_numeric((parameter,), encoding, ANGLE_QUANTIZATION_SCALE)),
                )
            )
        nodes.append(operation)
        edges.append(Edge(operation_id, EdgeType.USES_PROFILE, profile_id))
        if operation_ids:
            edges.append(Edge(operation_id, EdgeType.DEPENDS_ON, operation_ids[-1]))
        operation_ids.append(operation_id)

    return CADHistory(
        1,
        StructureGraph(tuple(nodes), tuple(edges), tuple(operation_ids)),
        GeometryStore(tuple(node_geometry), tuple(element_geometry)),
    )


def _plane_geometry(plane: ReferencePlane, encoding: GeometryEncoding) -> PlaneGeometry:
    frames = {
        ReferencePlane.XY: ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        ReferencePlane.XZ: ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        ReferencePlane.YZ: ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    }
    x_axis, y_axis = frames[plane]
    return PlaneGeometry(
        _numeric((0.0, 0.0, 0.0), encoding, LENGTH_QUANTIZATION_SCALE),
        _numeric(x_axis, encoding, LENGTH_QUANTIZATION_SCALE),
        _numeric(y_axis, encoding, LENGTH_QUANTIZATION_SCALE),
    )


def _sketch(
    sketch_id: str,
    family: PrimitiveFamily,
    extent: float,
    encoding: GeometryEncoding,
) -> tuple[SketchNode, tuple[SketchElementGeometryRecord, ...]]:
    x0 = 0.5
    if family is PrimitiveFamily.RECTANGLE_LINES:
        points = (
            (x0, -extent / 4),
            (x0 + extent, -extent / 4),
            (x0 + extent, extent / 4),
            (x0, extent / 4),
        )
        ids = tuple(f"line_{index}" for index in range(4))
        primitives = tuple(SketchPrimitive(item, PrimitiveType.LINE) for item in ids)
        records = tuple(
            SketchElementGeometryRecord(
                sketch_id,
                ids[index],
                LineGeometry(
                    _numeric(points[index], encoding, LENGTH_QUANTIZATION_SCALE),
                    _numeric(points[(index + 1) % 4], encoding, LENGTH_QUANTIZATION_SCALE),
                ),
            )
            for index in range(4)
        )
    elif family is PrimitiveFamily.CIRCLE:
        ids = ("circle_0",)
        primitives = (SketchPrimitive(ids[0], PrimitiveType.CIRCLE),)
        records = (
            SketchElementGeometryRecord(
                sketch_id,
                ids[0],
                CircleGeometry(
                    _numeric((x0 + extent / 2, 0.0), encoding, LENGTH_QUANTIZATION_SCALE),
                    _numeric((extent / 2,), encoding, LENGTH_QUANTIZATION_SCALE),
                ),
            ),
        )
    else:
        radius = extent / 4
        left_center = x0 + radius
        right_center = x0 + extent - radius
        ids = ("line_0", "arc_1", "line_2", "arc_3")
        primitives = (
            SketchPrimitive(ids[0], PrimitiveType.LINE),
            SketchPrimitive(ids[1], PrimitiveType.ARC),
            SketchPrimitive(ids[2], PrimitiveType.LINE),
            SketchPrimitive(ids[3], PrimitiveType.ARC),
        )
        records = (
            SketchElementGeometryRecord(
                sketch_id,
                ids[0],
                LineGeometry(
                    _numeric((left_center, -radius), encoding, LENGTH_QUANTIZATION_SCALE),
                    _numeric((right_center, -radius), encoding, LENGTH_QUANTIZATION_SCALE),
                ),
            ),
            SketchElementGeometryRecord(
                sketch_id,
                ids[1],
                ArcGeometry(
                    _numeric((right_center, -radius), encoding, LENGTH_QUANTIZATION_SCALE),
                    _numeric((right_center + radius, 0.0), encoding, LENGTH_QUANTIZATION_SCALE),
                    _numeric((right_center, radius), encoding, LENGTH_QUANTIZATION_SCALE),
                ),
            ),
            SketchElementGeometryRecord(
                sketch_id,
                ids[2],
                LineGeometry(
                    _numeric((right_center, radius), encoding, LENGTH_QUANTIZATION_SCALE),
                    _numeric((left_center, radius), encoding, LENGTH_QUANTIZATION_SCALE),
                ),
            ),
            SketchElementGeometryRecord(
                sketch_id,
                ids[3],
                ArcGeometry(
                    _numeric((left_center, radius), encoding, LENGTH_QUANTIZATION_SCALE),
                    _numeric((left_center - radius, 0.0), encoding, LENGTH_QUANTIZATION_SCALE),
                    _numeric((left_center, -radius), encoding, LENGTH_QUANTIZATION_SCALE),
                ),
            ),
        )
    return (
        SketchNode(
            sketch_id,
            primitives,
            (SketchLoop("outer_loop", LoopRole.OUTER, ids),),
        ),
        records,
    )


def _numeric(
    values: tuple[float, ...], encoding: GeometryEncoding, scale: float
) -> NumericValue:
    normalized = tuple(0.0 if float(value) == 0.0 else float(value) for value in values)
    if encoding is GeometryEncoding.CONTINUOUS:
        return NumericValue.continuous(*normalized)
    if encoding is not GeometryEncoding.QUANTIZED:
        raise ValueError(f"unsupported geometry encoding {encoding!r}")
    codes = tuple(round(value / scale) for value in normalized)
    for value, code in zip(normalized, codes):
        if not math.isclose(code * scale, value, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"physical value {value} is not representable with scale {scale}")
    return NumericValue.quantized(codes, scale=scale, offset=0.0)
