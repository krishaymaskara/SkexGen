"""Small valid histories used by documentation and tests."""

from __future__ import annotations

from .model import (
    AxisGeometry,
    AxisNode,
    BooleanMode,
    CADHistory,
    Direction,
    Edge,
    EdgeType,
    ExtrudeGeometry,
    ExtrudeNode,
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


def _rectangle(sketch_id: str, prefix: str) -> tuple[SketchNode, tuple[SketchElementGeometryRecord, ...]]:
    ids = tuple(f"{prefix}_line_{i}" for i in range(4))
    sketch = SketchNode(
        sketch_id,
        tuple(SketchPrimitive(item, PrimitiveType.LINE) for item in ids),
        (SketchLoop(f"{prefix}_outer", LoopRole.OUTER, ids),),
    )
    points = ((0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 1.0))
    geometry = tuple(
        SketchElementGeometryRecord(
            sketch_id,
            ids[i],
            LineGeometry(NumericValue.continuous(*points[i]), NumericValue.continuous(*points[(i + 1) % 4])),
        )
        for i in range(4)
    )
    return sketch, geometry


def _plane(node_id: str) -> tuple[ReferencePlaneNode, NodeGeometryRecord]:
    return (
        ReferencePlaneNode(node_id),
        NodeGeometryRecord(
            node_id,
            PlaneGeometry(
                NumericValue.continuous(0.0, 0.0, 0.0),
                NumericValue.continuous(1.0, 0.0, 0.0),
                NumericValue.continuous(0.0, 1.0, 0.0),
            ),
        ),
    )


def sketch_extrude(*, quantized_distance: bool = False) -> CADHistory:
    plane, plane_geometry = _plane("plane_1")
    sketch, element_geometry = _rectangle("sketch_1", "s1")
    profile = ProfileNode("profile_1", "s1_outer")
    operation = ExtrudeNode("extrude_1", BooleanMode.NEW_BODY, Direction.POSITIVE)
    distance = NumericValue.quantized((10,), scale=0.2) if quantized_distance else NumericValue.continuous(2.0)
    return CADHistory(
        1,
        StructureGraph(
            (plane, sketch, profile, operation),
            (
                Edge("sketch_1", EdgeType.PLACED_ON, "plane_1"),
                Edge("profile_1", EdgeType.DEFINED_IN, "sketch_1"),
                Edge("extrude_1", EdgeType.USES_PROFILE, "profile_1"),
            ),
            ("extrude_1",),
        ),
        GeometryStore(
            (plane_geometry, NodeGeometryRecord("extrude_1", ExtrudeGeometry(distance))),
            element_geometry,
        ),
    )


def sketch_revolve() -> CADHistory:
    plane, plane_geometry = _plane("plane_1")
    sketch, element_geometry = _rectangle("sketch_1", "s1")
    profile = ProfileNode("profile_1", "s1_outer")
    axis = AxisNode("axis_1")
    operation = RevolveNode("revolve_1", BooleanMode.NEW_BODY, Direction.POSITIVE)
    return CADHistory(
        1,
        StructureGraph(
            (plane, sketch, profile, axis, operation),
            (
                Edge("sketch_1", EdgeType.PLACED_ON, "plane_1"),
                Edge("profile_1", EdgeType.DEFINED_IN, "sketch_1"),
                Edge("axis_1", EdgeType.DEFINED_IN, "sketch_1"),
                Edge("revolve_1", EdgeType.USES_PROFILE, "profile_1"),
                Edge("revolve_1", EdgeType.USES_AXIS, "axis_1"),
            ),
            ("revolve_1",),
        ),
        GeometryStore(
            (
                plane_geometry,
                NodeGeometryRecord("axis_1", AxisGeometry(NumericValue.continuous(0.0, 0.0), NumericValue.continuous(0.0, 1.0))),
                NodeGeometryRecord("revolve_1", RevolveGeometry(NumericValue.continuous(180.0))),
            ),
            element_geometry,
        ),
    )


def sketch_extrude_sketch_extrude() -> CADHistory:
    return _two_operation_history(revolve_second=False)


def sketch_extrude_sketch_revolve() -> CADHistory:
    return _two_operation_history(revolve_second=True)


def _two_operation_history(*, revolve_second: bool) -> CADHistory:
    plane, plane_geometry = _plane("plane_1")
    sketch1, geometry1 = _rectangle("sketch_1", "s1")
    sketch2, geometry2 = _rectangle("sketch_2", "s2")
    profile1 = ProfileNode("profile_1", "s1_outer")
    profile2 = ProfileNode("profile_2", "s2_outer")
    first = ExtrudeNode("extrude_1", BooleanMode.NEW_BODY, Direction.POSITIVE)
    second = (
        RevolveNode("revolve_2", BooleanMode.JOIN, Direction.POSITIVE)
        if revolve_second
        else ExtrudeNode("extrude_2", BooleanMode.JOIN, Direction.NEGATIVE)
    )
    nodes = [plane, sketch1, profile1, first, sketch2, profile2, second]
    edges = [
        Edge("sketch_1", EdgeType.PLACED_ON, "plane_1"),
        Edge("sketch_2", EdgeType.PLACED_ON, "plane_1"),
        Edge("profile_1", EdgeType.DEFINED_IN, "sketch_1"),
        Edge("profile_2", EdgeType.DEFINED_IN, "sketch_2"),
        Edge("extrude_1", EdgeType.USES_PROFILE, "profile_1"),
        Edge(second.node_id, EdgeType.USES_PROFILE, "profile_2"),
        Edge(second.node_id, EdgeType.DEPENDS_ON, "extrude_1"),
    ]
    node_geometry = [
        plane_geometry,
        NodeGeometryRecord("extrude_1", ExtrudeGeometry(NumericValue.continuous(2.0))),
    ]
    if revolve_second:
        axis = AxisNode("axis_2")
        nodes.append(axis)
        edges.extend(
            [
                Edge("axis_2", EdgeType.DEFINED_IN, "sketch_2"),
                Edge("revolve_2", EdgeType.USES_AXIS, "axis_2"),
            ]
        )
        node_geometry.extend(
            [
                NodeGeometryRecord("axis_2", AxisGeometry(NumericValue.continuous(0.0, 0.0), NumericValue.continuous(0.0, 1.0))),
                NodeGeometryRecord("revolve_2", RevolveGeometry(NumericValue.continuous(90.0))),
            ]
        )
    else:
        node_geometry.append(NodeGeometryRecord("extrude_2", ExtrudeGeometry(NumericValue.continuous(1.0))))
    return CADHistory(
        1,
        StructureGraph(tuple(nodes), tuple(edges), ("extrude_1", second.node_id)),
        GeometryStore(tuple(node_geometry), geometry1 + geometry2),
    )

