"""Typed structural and geometric records for the representation prototype."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TypeAlias


class NodeType(str, Enum):
    REFERENCE_PLANE = "reference_plane"
    SKETCH = "sketch"
    PROFILE = "profile"
    AXIS = "axis"
    EXTRUDE = "extrude"
    REVOLVE = "revolve"


class EdgeType(str, Enum):
    PLACED_ON = "placed_on"
    DEFINED_IN = "defined_in"
    USES_PROFILE = "uses_profile"
    USES_AXIS = "uses_axis"
    DEPENDS_ON = "depends_on"


class PrimitiveType(str, Enum):
    LINE = "line"
    ARC = "arc"
    CIRCLE = "circle"


class LoopRole(str, Enum):
    OUTER = "outer"
    INNER = "inner"


class BooleanMode(str, Enum):
    NEW_BODY = "new_body"
    JOIN = "join"
    CUT = "cut"


class Direction(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"


class GeometryEncoding(str, Enum):
    CONTINUOUS = "continuous"
    QUANTIZED = "quantized"


@dataclass(frozen=True)
class NumericValue:
    """A scalar/vector stored continuously or as affine-decoded integer codes."""

    encoding: GeometryEncoding
    values: tuple[int | float, ...]
    scale: float | None = None
    offset: float | None = None

    @classmethod
    def continuous(cls, *values: int | float) -> NumericValue:
        return cls(GeometryEncoding.CONTINUOUS, tuple(values))

    @classmethod
    def quantized(
        cls, codes: tuple[int, ...], *, scale: float, offset: float = 0.0
    ) -> NumericValue:
        return cls(GeometryEncoding.QUANTIZED, codes, scale, offset)


@dataclass(frozen=True)
class SketchPrimitive:
    primitive_id: str
    primitive_type: PrimitiveType


@dataclass(frozen=True)
class SketchLoop:
    loop_id: str
    role: LoopRole
    primitive_ids: tuple[str, ...]


@dataclass(frozen=True)
class ReferencePlaneNode:
    node_id: str
    node_type: NodeType = field(init=False, default=NodeType.REFERENCE_PLANE)


@dataclass(frozen=True)
class SketchNode:
    node_id: str
    primitives: tuple[SketchPrimitive, ...]
    loops: tuple[SketchLoop, ...]
    node_type: NodeType = field(init=False, default=NodeType.SKETCH)


@dataclass(frozen=True)
class ProfileNode:
    node_id: str
    outer_loop_id: str
    inner_loop_ids: tuple[str, ...] = ()
    node_type: NodeType = field(init=False, default=NodeType.PROFILE)


@dataclass(frozen=True)
class AxisNode:
    node_id: str
    node_type: NodeType = field(init=False, default=NodeType.AXIS)


@dataclass(frozen=True)
class ExtrudeNode:
    node_id: str
    boolean_mode: BooleanMode
    direction: Direction
    node_type: NodeType = field(init=False, default=NodeType.EXTRUDE)


@dataclass(frozen=True)
class RevolveNode:
    node_id: str
    boolean_mode: BooleanMode
    direction: Direction
    node_type: NodeType = field(init=False, default=NodeType.REVOLVE)


GraphNode: TypeAlias = (
    ReferencePlaneNode | SketchNode | ProfileNode | AxisNode | ExtrudeNode | RevolveNode
)


@dataclass(frozen=True)
class Edge:
    source_id: str
    edge_type: EdgeType
    target_id: str


@dataclass(frozen=True)
class StructureGraph:
    nodes: tuple[GraphNode, ...]
    edges: tuple[Edge, ...]
    operation_sequence: tuple[str, ...]


@dataclass(frozen=True)
class PlaneGeometry:
    origin: NumericValue
    x_axis: NumericValue
    y_axis: NumericValue


@dataclass(frozen=True)
class LineGeometry:
    start: NumericValue
    end: NumericValue


@dataclass(frozen=True)
class ArcGeometry:
    start: NumericValue
    midpoint: NumericValue
    end: NumericValue


@dataclass(frozen=True)
class CircleGeometry:
    center: NumericValue
    radius: NumericValue


@dataclass(frozen=True)
class AxisGeometry:
    point: NumericValue
    direction: NumericValue


@dataclass(frozen=True)
class ExtrudeGeometry:
    distance: NumericValue


@dataclass(frozen=True)
class RevolveGeometry:
    angle_degrees: NumericValue


GeometryPayload: TypeAlias = (
    PlaneGeometry
    | LineGeometry
    | ArcGeometry
    | CircleGeometry
    | AxisGeometry
    | ExtrudeGeometry
    | RevolveGeometry
)


@dataclass(frozen=True)
class NodeGeometryRecord:
    node_id: str
    geometry: GeometryPayload


@dataclass(frozen=True)
class SketchElementGeometryRecord:
    """Geometry addressed unambiguously within one sketch's element namespace."""

    sketch_id: str
    element_id: str
    geometry: GeometryPayload


@dataclass(frozen=True)
class GeometryStore:
    node_geometry: tuple[NodeGeometryRecord, ...]
    sketch_element_geometry: tuple[SketchElementGeometryRecord, ...]


@dataclass(frozen=True)
class CADHistory:
    schema_version: int
    structure: StructureGraph
    geometry: GeometryStore

