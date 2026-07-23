"""Decode representation geometry and transform sketch-local data to world space."""

from __future__ import annotations

from dataclasses import dataclass
import math

from prototype.representation.model import (
    ArcGeometry,
    AxisGeometry,
    CADHistory,
    CircleGeometry,
    EdgeType,
    LineGeometry,
    NumericValue,
    PlaneGeometry,
    ProfileNode,
    SketchNode,
)

from .adapter import Point3, Vector3, WorldArc, WorldCircle, WorldLine, WorldPrimitive
from .config import VECTOR_NORM_TOLERANCE
from .model import ExecutionFailure, FailureCategory, FailureStage


@dataclass(frozen=True)
class PlaneFrame:
    origin: Point3
    x_axis: Vector3
    y_axis: Vector3
    normal: Vector3

    def point(self, local: tuple[float, float]) -> Point3:
        return tuple(
            self.origin[i] + local[0] * self.x_axis[i] + local[1] * self.y_axis[i]
            for i in range(3)
        )

    def vector(self, local: tuple[float, float]) -> Vector3:
        return _normalize(
            tuple(local[0] * self.x_axis[i] + local[1] * self.y_axis[i] for i in range(3))
        )


class HistoryIndex:
    def __init__(self, history: CADHistory):
        self.history = history
        self.nodes = {node.node_id: node for node in history.structure.nodes}
        self.node_geometry = {item.node_id: item.geometry for item in history.geometry.node_geometry}
        self.element_geometry = {
            (item.sketch_id, item.element_id): item.geometry
            for item in history.geometry.sketch_element_geometry
        }
        self.outgoing = {
            (edge.source_id, edge.edge_type): edge.target_id
            for edge in history.structure.edges
        }

    def target(self, source_id: str, edge_type: EdgeType, operation_index: int) -> str:
        try:
            return self.outgoing[(source_id, edge_type)]
        except KeyError as exc:
            raise ExecutionFailure(
                FailureCategory.SCHEMA_OR_PARSE_FAILURE,
                FailureStage.REFERENCE_RESOLUTION,
                f"missing {edge_type.value} reference for {source_id}",
                operation_index,
            ) from exc

    def profile_and_sketch(self, operation_id: str, operation_index: int) -> tuple[ProfileNode, SketchNode]:
        profile_id = self.target(operation_id, EdgeType.USES_PROFILE, operation_index)
        profile = self.nodes.get(profile_id)
        if not isinstance(profile, ProfileNode):
            raise ExecutionFailure(
                FailureCategory.SCHEMA_OR_PARSE_FAILURE,
                FailureStage.REFERENCE_RESOLUTION,
                f"profile reference {profile_id} has the wrong node type",
                operation_index,
            )
        if profile.inner_loop_ids:
            raise ExecutionFailure(
                FailureCategory.UNSUPPORTED_FEATURE,
                FailureStage.REFERENCE_RESOLUTION,
                "profile holes are unsupported by kernel-validation version 1",
                operation_index,
            )
        sketch_id = self.target(profile.node_id, EdgeType.DEFINED_IN, operation_index)
        sketch = self.nodes.get(sketch_id)
        if not isinstance(sketch, SketchNode):
            raise ExecutionFailure(
                FailureCategory.SCHEMA_OR_PARSE_FAILURE,
                FailureStage.REFERENCE_RESOLUTION,
                f"profile sketch reference {sketch_id} has the wrong node type",
                operation_index,
            )
        return profile, sketch

    def frame_for_sketch(self, sketch: SketchNode, operation_index: int) -> PlaneFrame:
        plane_id = self.target(sketch.node_id, EdgeType.PLACED_ON, operation_index)
        geometry = self.node_geometry.get(plane_id)
        if not isinstance(geometry, PlaneGeometry):
            raise ExecutionFailure(
                FailureCategory.INVALID_REFERENCE_PLANE,
                FailureStage.WORLD_TRANSFORM,
                f"reference plane {plane_id} has no plane geometry",
                operation_index,
            )
        try:
            origin = decode_numeric(geometry.origin, 3)
            x_axis = _normalize(decode_numeric(geometry.x_axis, 3))
            y_axis = _normalize(decode_numeric(geometry.y_axis, 3))
            normal = _normalize(_cross(x_axis, y_axis))
        except ValueError as exc:
            raise ExecutionFailure(
                FailureCategory.INVALID_REFERENCE_PLANE,
                FailureStage.WORLD_TRANSFORM,
                str(exc),
                operation_index,
            ) from exc
        return PlaneFrame(origin, x_axis, y_axis, normal)

    def profile_primitives(
        self, profile: ProfileNode, sketch: SketchNode, frame: PlaneFrame, operation_index: int
    ) -> tuple[WorldPrimitive, ...]:
        loops = {loop.loop_id: loop for loop in sketch.loops}
        loop = loops.get(profile.outer_loop_id)
        if loop is None:
            raise ExecutionFailure(
                FailureCategory.SCHEMA_OR_PARSE_FAILURE,
                FailureStage.REFERENCE_RESOLUTION,
                f"outer loop {profile.outer_loop_id} is missing",
                operation_index,
            )
        result: list[WorldPrimitive] = []
        for primitive_id in loop.primitive_ids:
            geometry = self.element_geometry.get((sketch.node_id, primitive_id))
            try:
                if isinstance(geometry, LineGeometry):
                    result.append(WorldLine(frame.point(decode_numeric(geometry.start, 2)), frame.point(decode_numeric(geometry.end, 2))))
                elif isinstance(geometry, ArcGeometry):
                    result.append(WorldArc(frame.point(decode_numeric(geometry.start, 2)), frame.point(decode_numeric(geometry.midpoint, 2)), frame.point(decode_numeric(geometry.end, 2))))
                elif isinstance(geometry, CircleGeometry):
                    result.append(WorldCircle(frame.point(decode_numeric(geometry.center, 2)), frame.normal, frame.x_axis, decode_numeric(geometry.radius, 1)[0]))
                else:
                    raise ValueError(f"unsupported or missing primitive geometry for {primitive_id}")
            except ValueError as exc:
                raise ExecutionFailure(
                    FailureCategory.SKETCH_CONSTRUCTION_FAILURE,
                    FailureStage.WORLD_TRANSFORM,
                    str(exc),
                    operation_index,
                ) from exc
        return tuple(result)

    def axis_world(self, operation_id: str, frame: PlaneFrame, operation_index: int) -> tuple[Point3, Vector3]:
        axis_id = self.target(operation_id, EdgeType.USES_AXIS, operation_index)
        geometry = self.node_geometry.get(axis_id)
        if not isinstance(geometry, AxisGeometry):
            raise ExecutionFailure(
                FailureCategory.INVALID_REVOLVE_AXIS,
                FailureStage.REFERENCE_RESOLUTION,
                f"axis {axis_id} has no axis geometry",
                operation_index,
            )
        try:
            return frame.point(decode_numeric(geometry.point, 2)), frame.vector(decode_numeric(geometry.direction, 2))
        except ValueError as exc:
            raise ExecutionFailure(
                FailureCategory.INVALID_REVOLVE_AXIS,
                FailureStage.WORLD_TRANSFORM,
                str(exc),
                operation_index,
            ) from exc


def decode_numeric(value: NumericValue, expected_length: int) -> tuple[float, ...]:
    if len(value.values) != expected_length:
        raise ValueError(f"expected {expected_length} numeric values")
    if value.encoding.value == "continuous":
        decoded = tuple(float(item) for item in value.values)
    elif value.encoding.value == "quantized":
        if value.scale is None or value.offset is None:
            raise ValueError("quantized value is missing scale or offset")
        decoded = tuple(int(item) * float(value.scale) + float(value.offset) for item in value.values)
    else:
        raise ValueError("unsupported geometry encoding")
    if not all(math.isfinite(item) for item in decoded):
        raise ValueError("decoded geometry must be finite")
    return decoded


def _cross(left: Vector3, right: Vector3) -> Vector3:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _normalize(vector: tuple[float, ...]) -> tuple[float, ...]:
    norm = math.sqrt(sum(item * item for item in vector))
    if not math.isfinite(norm) or norm <= VECTOR_NORM_TOLERANCE:
        raise ValueError("vector is nonfinite or has near-zero magnitude")
    return tuple(item / norm for item in vector)
