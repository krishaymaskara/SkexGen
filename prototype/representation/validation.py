"""Schema and geometric-sanity validation for version 1 CAD histories."""

from __future__ import annotations

from dataclasses import dataclass
import math

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

SCHEMA_VERSION = 1
FRAME_ORTHONORMAL_TOLERANCE = 1e-6
AXIS_DIRECTION_TOLERANCE = 1e-6
LOOP_CONTINUITY_TOLERANCE = 1e-6

SUPPORTED_NODE_TYPES = (
    ReferencePlaneNode,
    SketchNode,
    ProfileNode,
    AxisNode,
    ExtrudeNode,
    RevolveNode,
)


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    path: str
    message: str


class ValidationError(ValueError):
    """One or more structured violations of the versioned representation contract."""

    def __init__(self, issues: list[ValidationIssue] | tuple[ValidationIssue, ...]):
        self.issues = tuple(issues)
        super().__init__("; ".join(f"{i.code} at {i.path}: {i.message}" for i in self.issues))


class _Validator:
    def __init__(self, history: CADHistory):
        self.history = history
        self.issues: list[ValidationIssue] = []
        self.nodes = {}
        self.sketches: dict[str, SketchNode] = {}
        self.node_geometry = {}
        self.element_geometry = {}

    def issue(self, code: str, path: str, message: str) -> None:
        self.issues.append(ValidationIssue(code, path, message))

    def run(self) -> None:
        self._validate_version_and_nodes()
        self._validate_sketch_elements()
        self._validate_edges()
        self._validate_operation_sequence()
        self._validate_geometry_store()
        self._validate_profiles_and_revolves()
        self._validate_loops()
        if self.issues:
            raise ValidationError(self.issues)

    def _validate_version_and_nodes(self) -> None:
        version = self.history.schema_version
        if isinstance(version, bool) or not isinstance(version, int) or version != SCHEMA_VERSION:
            self.issue("schema_version", "schema_version", "expected integer schema version 1")
        for index, node in enumerate(self.history.structure.nodes):
            path = f"structure.nodes[{index}]"
            if not isinstance(node.node_id, str) or not node.node_id:
                self.issue("invalid_id", f"{path}.node_id", "node ID must be a nonempty string")
                continue
            if node.node_id in self.nodes:
                self.issue("duplicate_node", f"{path}.node_id", f"duplicate node ID {node.node_id!r}")
                continue
            self.nodes[node.node_id] = node
            if isinstance(node, SketchNode):
                self.sketches[node.node_id] = node
            elif isinstance(node, ProfileNode):
                if not isinstance(node.outer_loop_id, str) or not node.outer_loop_id:
                    self.issue("invalid_id", f"{path}.outer_loop_id", "outer-loop ID must be a nonempty string")
                for inner_index, loop_id in enumerate(node.inner_loop_ids):
                    if not isinstance(loop_id, str) or not loop_id:
                        self.issue("invalid_id", f"{path}.inner_loop_ids[{inner_index}]", "inner-loop ID must be a nonempty string")
            elif isinstance(node, (ExtrudeNode, RevolveNode)):
                if not isinstance(node.boolean_mode, BooleanMode):
                    self.issue("enum_type", f"{path}.boolean_mode", "boolean mode must be a BooleanMode enum")
                if not isinstance(node.direction, Direction):
                    self.issue("enum_type", f"{path}.direction", "direction must be a Direction enum")

    def _validate_sketch_elements(self) -> None:
        for sketch_id, sketch in self.sketches.items():
            if not sketch.primitives:
                self.issue("empty_sketch", f"sketch[{sketch_id}].primitives", "a sketch must declare at least one primitive")
            if not sketch.loops:
                self.issue("empty_sketch", f"sketch[{sketch_id}].loops", "a sketch must declare at least one loop")
            seen: set[str] = set()
            primitive_ids: set[str] = set()
            for index, primitive in enumerate(sketch.primitives):
                path = f"sketch[{sketch_id}].primitives[{index}].primitive_id"
                if not isinstance(primitive.primitive_id, str) or not primitive.primitive_id:
                    self.issue("invalid_id", path, "primitive ID must be a nonempty string")
                elif primitive.primitive_id in seen:
                    self.issue("duplicate_element", path, "element ID is already used in this sketch")
                else:
                    seen.add(primitive.primitive_id)
                    primitive_ids.add(primitive.primitive_id)
                if not isinstance(primitive.primitive_type, PrimitiveType):
                    self.issue("enum_type", f"sketch[{sketch_id}].primitives[{index}].primitive_type", "primitive type must be a PrimitiveType enum")
            for index, loop in enumerate(sketch.loops):
                path = f"sketch[{sketch_id}].loops[{index}]"
                if not isinstance(loop.loop_id, str) or not loop.loop_id:
                    self.issue("invalid_id", f"{path}.loop_id", "loop ID must be a nonempty string")
                elif loop.loop_id in seen:
                    self.issue("duplicate_element", f"{path}.loop_id", "element ID is already used in this sketch")
                else:
                    seen.add(loop.loop_id)
                if not isinstance(loop.role, LoopRole):
                    self.issue("enum_type", f"{path}.role", "loop role must be a LoopRole enum")
                if not loop.primitive_ids:
                    self.issue("empty_loop", f"{path}.primitive_ids", "a loop must reference primitives")
                if len(loop.primitive_ids) != len(set(loop.primitive_ids)):
                    self.issue("duplicate_loop_primitive", f"{path}.primitive_ids", "loop primitive references must be unique")
                for primitive_id in loop.primitive_ids:
                    if primitive_id not in primitive_ids:
                        self.issue("missing_primitive", f"{path}.primitive_ids", f"unknown primitive {primitive_id!r}")

    def _validate_edges(self) -> None:
        allowed = {
            EdgeType.PLACED_ON: ((SketchNode,), (ReferencePlaneNode,)),
            EdgeType.DEFINED_IN: ((ProfileNode, AxisNode), (SketchNode,)),
            EdgeType.USES_PROFILE: ((ExtrudeNode, RevolveNode), (ProfileNode,)),
            EdgeType.USES_AXIS: ((RevolveNode,), (AxisNode,)),
            EdgeType.DEPENDS_ON: ((ExtrudeNode, RevolveNode), (ExtrudeNode, RevolveNode)),
        }
        seen = set()
        outgoing: dict[tuple[str, EdgeType], list[str]] = {}
        for index, edge in enumerate(self.history.structure.edges):
            path = f"structure.edges[{index}]"
            if not isinstance(edge.edge_type, EdgeType):
                self.issue("invalid_edge_type", f"{path}.edge_type", "edge type must be an EdgeType enum")
                continue
            if not isinstance(edge.source_id, str) or not edge.source_id:
                self.issue("invalid_id", f"{path}.source_id", "edge source ID must be a nonempty string")
                continue
            if not isinstance(edge.target_id, str) or not edge.target_id:
                self.issue("invalid_id", f"{path}.target_id", "edge target ID must be a nonempty string")
                continue
            key = (edge.source_id, edge.edge_type, edge.target_id)
            if key in seen:
                self.issue("duplicate_edge", path, "duplicate edge")
            seen.add(key)
            source = self.nodes.get(edge.source_id)
            target = self.nodes.get(edge.target_id)
            if source is None:
                self.issue("missing_node", f"{path}.source_id", f"unknown node {edge.source_id!r}")
            if target is None:
                self.issue("missing_node", f"{path}.target_id", f"unknown node {edge.target_id!r}")
            if source is None or target is None:
                continue
            source_types, target_types = allowed[edge.edge_type]
            if not isinstance(source, source_types) or not isinstance(target, target_types):
                self.issue("invalid_edge", path, f"{edge.edge_type.value} is not allowed between these node types")
            outgoing.setdefault((edge.source_id, edge.edge_type), []).append(edge.target_id)

        for node_id, node in self.nodes.items():
            requirements = []
            if isinstance(node, SketchNode):
                requirements = [(EdgeType.PLACED_ON, 1)]
            elif isinstance(node, (ProfileNode, AxisNode)):
                requirements = [(EdgeType.DEFINED_IN, 1)]
            elif isinstance(node, ExtrudeNode):
                requirements = [(EdgeType.USES_PROFILE, 1), (EdgeType.USES_AXIS, 0)]
            elif isinstance(node, RevolveNode):
                requirements = [(EdgeType.USES_PROFILE, 1), (EdgeType.USES_AXIS, 1)]
            for edge_type, expected in requirements:
                actual = len(outgoing.get((node_id, edge_type), []))
                if actual != expected:
                    self.issue(
                        "edge_cardinality",
                        f"node[{node_id}].{edge_type.value}",
                        f"expected {expected} outgoing edge(s), found {actual}",
                    )
        self.outgoing = outgoing

    def _validate_operation_sequence(self) -> None:
        sequence = self.history.structure.operation_sequence
        operations = {node_id for node_id, node in self.nodes.items() if isinstance(node, (ExtrudeNode, RevolveNode))}
        if not sequence:
            self.issue("empty_operation_sequence", "structure.operation_sequence", "a version 1 history must produce one solid")
        if len(sequence) != len(set(sequence)):
            self.issue("duplicate_operation", "structure.operation_sequence", "operations must appear exactly once")
        for node_id in sequence:
            if node_id not in self.nodes:
                self.issue("missing_node", "structure.operation_sequence", f"unknown node {node_id!r}")
            elif node_id not in operations:
                self.issue("non_operation", "structure.operation_sequence", f"{node_id!r} is not an operation")
        missing = operations - set(sequence)
        if missing:
            self.issue("missing_operation", "structure.operation_sequence", f"missing operations {sorted(missing)!r}")

        positions = {node_id: index for index, node_id in enumerate(sequence)}
        dependency_targets: dict[str, list[str]] = {}
        for edge in self.history.structure.edges:
            if edge.edge_type is not EdgeType.DEPENDS_ON:
                continue
            dependency_targets.setdefault(edge.source_id, []).append(edge.target_id)
            if edge.source_id in positions and edge.target_id in positions:
                if positions[edge.target_id] >= positions[edge.source_id]:
                    self.issue("dependency_order", f"edge[{edge.source_id}->{edge.target_id}]", "dependency must point to an earlier operation")

        # Version 1 has one current solid, so the semantic dependency chain is linear.
        for index, node_id in enumerate(sequence):
            targets = dependency_targets.get(node_id, [])
            expected = [] if index == 0 else [sequence[index - 1]]
            if targets != expected:
                self.issue("single_body_dependency", f"node[{node_id}].depends_on", f"expected dependency targets {expected!r}, found {targets!r}")
            node = self.nodes.get(node_id)
            if isinstance(node, (ExtrudeNode, RevolveNode)):
                expected_mode = BooleanMode.NEW_BODY if index == 0 else None
                if expected_mode and node.boolean_mode is not expected_mode:
                    self.issue("first_operation_mode", f"node[{node_id}].boolean_mode", "first operation must use NEW_BODY")
                if index > 0 and node.boolean_mode not in (BooleanMode.JOIN, BooleanMode.CUT):
                    self.issue("later_operation_mode", f"node[{node_id}].boolean_mode", "later operations must use JOIN or CUT")
        self._validate_dependency_cycles(dependency_targets)

    def _validate_dependency_cycles(self, dependencies: dict[str, list[str]]) -> None:
        state: dict[str, int] = {}

        def visit(node_id: str) -> None:
            if state.get(node_id) == 1:
                self.issue("dependency_cycle", "structure.edges", f"cycle includes operation {node_id!r}")
                return
            if state.get(node_id) == 2:
                return
            state[node_id] = 1
            for target in dependencies.get(node_id, []):
                visit(target)
            state[node_id] = 2

        for node_id in dependencies:
            visit(node_id)

    def _validate_geometry_store(self) -> None:
        for index, record in enumerate(self.history.geometry.node_geometry):
            path = f"geometry.node_geometry[{index}]"
            if not isinstance(record.node_id, str) or not record.node_id:
                self.issue("invalid_id", f"{path}.node_id", "geometry node ID must be a nonempty string")
                continue
            if record.node_id in self.node_geometry:
                self.issue("duplicate_geometry", path, f"duplicate geometry for node {record.node_id!r}")
            self.node_geometry[record.node_id] = record.geometry
            node = self.nodes.get(record.node_id)
            if node is None:
                self.issue("missing_node", f"{path}.node_id", f"unknown node {record.node_id!r}")
                continue
            expected = {
                ReferencePlaneNode: PlaneGeometry,
                AxisNode: AxisGeometry,
                ExtrudeNode: ExtrudeGeometry,
                RevolveNode: RevolveGeometry,
            }
            expected_type = next((g for n, g in expected.items() if isinstance(node, n)), None)
            if expected_type is None or not isinstance(record.geometry, expected_type):
                self.issue("geometry_type", path, "geometry payload is not valid for this node type")
            self._validate_geometry_payload(record.geometry, path)

        for index, record in enumerate(self.history.geometry.sketch_element_geometry):
            path = f"geometry.sketch_element_geometry[{index}]"
            if not isinstance(record.sketch_id, str) or not record.sketch_id:
                self.issue("invalid_id", f"{path}.sketch_id", "sketch ID must be a nonempty string")
                continue
            if not isinstance(record.element_id, str) or not record.element_id:
                self.issue("invalid_id", f"{path}.element_id", "element ID must be a nonempty string")
                continue
            key = (record.sketch_id, record.element_id)
            if key in self.element_geometry:
                self.issue("duplicate_geometry", path, f"duplicate geometry for sketch element {key!r}")
            self.element_geometry[key] = record.geometry
            sketch = self.sketches.get(record.sketch_id)
            if sketch is None:
                self.issue("missing_sketch", f"{path}.sketch_id", f"unknown sketch {record.sketch_id!r}")
                continue
            primitives = {p.primitive_id: p for p in sketch.primitives}
            loops = {loop.loop_id for loop in sketch.loops}
            primitive = primitives.get(record.element_id)
            if primitive is None:
                code = "unsupported_loop_geometry" if record.element_id in loops else "missing_element"
                self.issue(code, f"{path}.element_id", "version 1 geometry records must address an existing primitive")
                continue
            expected_geometry = {
                PrimitiveType.LINE: LineGeometry,
                PrimitiveType.ARC: ArcGeometry,
                PrimitiveType.CIRCLE: CircleGeometry,
            }.get(primitive.primitive_type)
            if expected_geometry is None or not isinstance(record.geometry, expected_geometry):
                self.issue("geometry_type", path, "geometry payload does not match primitive type")
            self._validate_geometry_payload(record.geometry, path)

        for node_id, node in self.nodes.items():
            if isinstance(node, (ReferencePlaneNode, AxisNode, ExtrudeNode, RevolveNode)) and node_id not in self.node_geometry:
                self.issue("missing_geometry", f"geometry.node[{node_id}]", "required node geometry is missing")
        for sketch_id, sketch in self.sketches.items():
            for primitive in sketch.primitives:
                if (sketch_id, primitive.primitive_id) not in self.element_geometry:
                    self.issue("missing_geometry", f"geometry.element[{sketch_id},{primitive.primitive_id}]", "required primitive geometry is missing")

    def _decoded(self, value: NumericValue, path: str, length: int) -> tuple[float, ...] | None:
        if not isinstance(value, NumericValue):
            self.issue("numeric_type", path, "expected NumericValue")
            return None
        if not isinstance(value.values, tuple):
            self.issue("immutable_collection", f"{path}.values", "numeric values must be stored as a tuple")
            return None
        if len(value.values) != length:
            self.issue("numeric_shape", path, f"expected {length} value(s), found {len(value.values)}")
            return None
        decoded: list[float] = []
        if value.encoding is GeometryEncoding.CONTINUOUS:
            if value.scale is not None or value.offset is not None:
                self.issue("continuous_metadata", path, "continuous values may not define scale or offset")
            for raw in value.values:
                if not _is_finite_number(raw):
                    self.issue("nonfinite_numeric", path, "continuous values must be finite numbers")
                    return None
                try:
                    decoded.append(float(raw))
                except OverflowError:
                    self.issue("nonfinite_numeric", path, "continuous values must be representable as finite floats")
                    return None
        elif value.encoding is GeometryEncoding.QUANTIZED:
            if not _is_finite_number(value.scale) or value.scale <= 0:
                self.issue("quantized_scale", path, "quantized scale must be finite and strictly positive")
                return None
            if not _is_finite_number(value.offset):
                self.issue("quantized_offset", path, "quantized offset must be finite")
                return None
            for code in value.values:
                if isinstance(code, bool) or not isinstance(code, int):
                    self.issue("quantized_code", path, "quantized codes must be actual integers")
                    return None
                try:
                    physical = code * float(value.scale) + float(value.offset)
                except OverflowError:
                    self.issue("decoded_nonfinite", path, "decoded physical values must be finite")
                    return None
                if not _is_finite_number(physical):
                    self.issue("decoded_nonfinite", path, "decoded physical values must be finite")
                    return None
                decoded.append(physical)
        else:
            self.issue("geometry_encoding", path, "unsupported geometry encoding")
            return None
        return tuple(decoded)

    def _validate_geometry_payload(self, geometry, path: str) -> None:
        if isinstance(geometry, PlaneGeometry):
            origin = self._decoded(geometry.origin, f"{path}.origin", 3)
            x_axis = self._decoded(geometry.x_axis, f"{path}.x_axis", 3)
            y_axis = self._decoded(geometry.y_axis, f"{path}.y_axis", 3)
            if origin and x_axis and y_axis:
                if abs(_norm(x_axis) - 1.0) > FRAME_ORTHONORMAL_TOLERANCE or abs(_norm(y_axis) - 1.0) > FRAME_ORTHONORMAL_TOLERANCE or abs(_dot(x_axis, y_axis)) > FRAME_ORTHONORMAL_TOLERANCE:
                    self.issue("invalid_plane_frame", path, "plane axes must be orthonormal within FRAME_ORTHONORMAL_TOLERANCE")
        elif isinstance(geometry, LineGeometry):
            start = self._decoded(geometry.start, f"{path}.start", 2)
            end = self._decoded(geometry.end, f"{path}.end", 2)
            if start and end and _distance(start, end) <= LOOP_CONTINUITY_TOLERANCE:
                self.issue("degenerate_line", path, "line endpoints must be distinct")
        elif isinstance(geometry, ArcGeometry):
            start = self._decoded(geometry.start, f"{path}.start", 2)
            midpoint = self._decoded(geometry.midpoint, f"{path}.midpoint", 2)
            end = self._decoded(geometry.end, f"{path}.end", 2)
            if start and midpoint and end:
                area2 = (midpoint[0] - start[0]) * (end[1] - start[1]) - (midpoint[1] - start[1]) * (end[0] - start[0])
                if abs(area2) <= LOOP_CONTINUITY_TOLERANCE:
                    self.issue("degenerate_arc", path, "arc points must be non-collinear")
        elif isinstance(geometry, CircleGeometry):
            self._decoded(geometry.center, f"{path}.center", 2)
            radius = self._decoded(geometry.radius, f"{path}.radius", 1)
            if radius and radius[0] <= 0:
                self.issue("invalid_radius", f"{path}.radius", "circle radius must be positive")
        elif isinstance(geometry, AxisGeometry):
            self._decoded(geometry.point, f"{path}.point", 2)
            direction = self._decoded(geometry.direction, f"{path}.direction", 2)
            if direction and (not math.isfinite(_norm(direction)) or abs(_norm(direction) - 1.0) > AXIS_DIRECTION_TOLERANCE):
                self.issue("invalid_axis_direction", f"{path}.direction", "axis direction must be nonzero and unit length within AXIS_DIRECTION_TOLERANCE")
        elif isinstance(geometry, ExtrudeGeometry):
            distance = self._decoded(geometry.distance, f"{path}.distance", 1)
            if distance and distance[0] <= 0:
                self.issue("invalid_extrude_distance", f"{path}.distance", "one-sided extrusion distance must be a positive magnitude")
        elif isinstance(geometry, RevolveGeometry):
            angle = self._decoded(geometry.angle_degrees, f"{path}.angle_degrees", 1)
            if angle and not (0 < angle[0] <= 360):
                self.issue("invalid_revolve_angle", f"{path}.angle_degrees", "revolve angle in degrees must lie in (0, 360]")

    def _validate_profiles_and_revolves(self) -> None:
        for node_id, node in self.nodes.items():
            if isinstance(node, ProfileNode):
                sketches = self.outgoing.get((node_id, EdgeType.DEFINED_IN), [])
                if len(sketches) != 1 or sketches[0] not in self.sketches:
                    continue
                sketch = self.sketches[sketches[0]]
                loops = {loop.loop_id: loop for loop in sketch.loops}
                if node.outer_loop_id not in loops:
                    self.issue("missing_loop", f"node[{node_id}].outer_loop_id", "outer loop is not defined in the profile sketch")
                elif loops[node.outer_loop_id].role is not LoopRole.OUTER:
                    self.issue("loop_role", f"node[{node_id}].outer_loop_id", "outer loop must have OUTER role")
                if len(node.inner_loop_ids) != len(set(node.inner_loop_ids)):
                    self.issue("duplicate_inner_loop", f"node[{node_id}].inner_loop_ids", "inner-loop references must be unique")
                if node.outer_loop_id in node.inner_loop_ids:
                    self.issue("overlapping_loop_reference", f"node[{node_id}]", "outer and inner loops must be distinct")
                for loop_id in node.inner_loop_ids:
                    if loop_id not in loops:
                        self.issue("missing_loop", f"node[{node_id}].inner_loop_ids", f"loop {loop_id!r} is not defined in the profile sketch")
                    elif loops[loop_id].role is not LoopRole.INNER:
                        self.issue("loop_role", f"node[{node_id}].inner_loop_ids", f"loop {loop_id!r} must have INNER role")

            if isinstance(node, RevolveNode):
                profiles = self.outgoing.get((node_id, EdgeType.USES_PROFILE), [])
                axes = self.outgoing.get((node_id, EdgeType.USES_AXIS), [])
                if len(profiles) == 1 and len(axes) == 1:
                    profile_sketch = self.outgoing.get((profiles[0], EdgeType.DEFINED_IN), [])
                    axis_sketch = self.outgoing.get((axes[0], EdgeType.DEFINED_IN), [])
                    # Same-sketch profile/axis is a version 1 simplification, not a general CAD rule.
                    if len(profile_sketch) == 1 and len(axis_sketch) == 1 and profile_sketch[0] != axis_sketch[0]:
                        self.issue("revolve_sketch_compatibility", f"node[{node_id}]", "version 1 requires profile and axis to be defined in the same sketch")

    def _validate_loops(self) -> None:
        for sketch_id, sketch in self.sketches.items():
            primitives = {p.primitive_id: p for p in sketch.primitives}
            for loop in sketch.loops:
                entries = []
                missing_geometry = False
                for primitive_id in loop.primitive_ids:
                    primitive = primitives.get(primitive_id)
                    geometry = self.element_geometry.get((sketch_id, primitive_id))
                    if primitive is None or geometry is None:
                        missing_geometry = True
                        continue
                    entries.append((primitive, geometry))
                if missing_geometry or len(entries) != len(loop.primitive_ids):
                    continue
                path = f"sketch[{sketch_id}].loop[{loop.loop_id}]"
                if len(entries) == 1 and isinstance(entries[0][1], CircleGeometry):
                    continue
                if any(isinstance(geometry, CircleGeometry) for _, geometry in entries):
                    self.issue("unsupported_loop", path, "a circle may only form a loop by itself")
                    continue
                if not all(isinstance(geometry, (LineGeometry, ArcGeometry)) for _, geometry in entries):
                    self.issue("unsupported_loop", path, "only line/arc chains or one circle are supported")
                    continue
                endpoints = []
                for index, (_, geometry) in enumerate(entries):
                    start = self._decoded(geometry.start, f"{path}.primitive[{index}].start", 2)
                    end = self._decoded(geometry.end, f"{path}.primitive[{index}].end", 2)
                    if start is None or end is None:
                        endpoints = []
                        break
                    endpoints.append((start, end))
                for index, (_, end) in enumerate(endpoints):
                    next_start = endpoints[(index + 1) % len(endpoints)][0]
                    if _distance(end, next_start) > LOOP_CONTINUITY_TOLERANCE:
                        self.issue("open_loop", path, f"primitive {index} endpoint does not meet the next start within LOOP_CONTINUITY_TOLERANCE")


def _dot(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _norm(vector: tuple[float, ...]) -> float:
    return math.sqrt(_dot(vector, vector))


def _distance(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    return _norm(tuple(x - y for x, y in zip(a, b)))


def _is_finite_number(value) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _preflight_issues(history) -> list[ValidationIssue]:
    """Reject malformed direct dataclass construction before deeper validation."""

    issues: list[ValidationIssue] = []

    def require_tuple(value, path: str) -> bool:
        if not isinstance(value, tuple):
            issues.append(ValidationIssue("immutable_collection", path, "version 1 collections must be tuples"))
            return False
        return True

    if not isinstance(history, CADHistory):
        return [ValidationIssue("history_type", "$", "expected a CADHistory")]
    if not isinstance(history.structure, StructureGraph):
        issues.append(ValidationIssue("structure_type", "structure", "expected a StructureGraph"))
    if not isinstance(history.geometry, GeometryStore):
        issues.append(ValidationIssue("geometry_store_type", "geometry", "expected a GeometryStore"))
    if issues:
        return issues

    if require_tuple(history.structure.nodes, "structure.nodes"):
        for index, node in enumerate(history.structure.nodes):
            path = f"structure.nodes[{index}]"
            if not isinstance(node, SUPPORTED_NODE_TYPES):
                issues.append(ValidationIssue("node_type", path, "unsupported graph-node record"))
                continue
            if isinstance(node, SketchNode):
                if require_tuple(node.primitives, f"{path}.primitives"):
                    for primitive_index, primitive in enumerate(node.primitives):
                        if not isinstance(primitive, SketchPrimitive):
                            issues.append(ValidationIssue("primitive_type", f"{path}.primitives[{primitive_index}]", "expected a SketchPrimitive"))
                            continue
                        if not isinstance(primitive.primitive_id, str) or not primitive.primitive_id:
                            issues.append(ValidationIssue("invalid_id", f"{path}.primitives[{primitive_index}].primitive_id", "primitive ID must be a nonempty string"))
                        if not isinstance(primitive.primitive_type, PrimitiveType):
                            issues.append(ValidationIssue("enum_type", f"{path}.primitives[{primitive_index}].primitive_type", "primitive type must be a PrimitiveType enum"))
                if require_tuple(node.loops, f"{path}.loops"):
                    for loop_index, loop in enumerate(node.loops):
                        if not isinstance(loop, SketchLoop):
                            issues.append(ValidationIssue("loop_type", f"{path}.loops[{loop_index}]", "expected a SketchLoop"))
                            continue
                        if not isinstance(loop.loop_id, str) or not loop.loop_id:
                            issues.append(ValidationIssue("invalid_id", f"{path}.loops[{loop_index}].loop_id", "loop ID must be a nonempty string"))
                        if not isinstance(loop.role, LoopRole):
                            issues.append(ValidationIssue("enum_type", f"{path}.loops[{loop_index}].role", "loop role must be a LoopRole enum"))
                        if require_tuple(loop.primitive_ids, f"{path}.loops[{loop_index}].primitive_ids"):
                            for ref_index, primitive_id in enumerate(loop.primitive_ids):
                                if not isinstance(primitive_id, str) or not primitive_id:
                                    issues.append(ValidationIssue("invalid_id", f"{path}.loops[{loop_index}].primitive_ids[{ref_index}]", "primitive ID must be a nonempty string"))
            elif isinstance(node, ProfileNode):
                if not isinstance(node.outer_loop_id, str) or not node.outer_loop_id:
                    issues.append(ValidationIssue("invalid_id", f"{path}.outer_loop_id", "outer-loop ID must be a nonempty string"))
                if require_tuple(node.inner_loop_ids, f"{path}.inner_loop_ids"):
                    for loop_index, loop_id in enumerate(node.inner_loop_ids):
                        if not isinstance(loop_id, str) or not loop_id:
                            issues.append(ValidationIssue("invalid_id", f"{path}.inner_loop_ids[{loop_index}]", "inner-loop ID must be a nonempty string"))

    if require_tuple(history.structure.edges, "structure.edges"):
        for index, edge in enumerate(history.structure.edges):
            if not isinstance(edge, Edge):
                issues.append(ValidationIssue("edge_type", f"structure.edges[{index}]", "expected an Edge record"))
                continue
            if not isinstance(edge.source_id, str) or not edge.source_id:
                issues.append(ValidationIssue("invalid_id", f"structure.edges[{index}].source_id", "edge source ID must be a nonempty string"))
            if not isinstance(edge.target_id, str) or not edge.target_id:
                issues.append(ValidationIssue("invalid_id", f"structure.edges[{index}].target_id", "edge target ID must be a nonempty string"))
            if not isinstance(edge.edge_type, EdgeType):
                issues.append(ValidationIssue("invalid_edge_type", f"structure.edges[{index}].edge_type", "edge type must be an EdgeType enum"))
    if require_tuple(history.structure.operation_sequence, "structure.operation_sequence"):
        for index, node_id in enumerate(history.structure.operation_sequence):
            if not isinstance(node_id, str) or not node_id:
                issues.append(ValidationIssue("invalid_id", f"structure.operation_sequence[{index}]", "operation ID must be a nonempty string"))

    if require_tuple(history.geometry.node_geometry, "geometry.node_geometry"):
        for index, record in enumerate(history.geometry.node_geometry):
            if not isinstance(record, NodeGeometryRecord):
                issues.append(ValidationIssue("geometry_record_type", f"geometry.node_geometry[{index}]", "expected a NodeGeometryRecord"))
            elif not isinstance(record.node_id, str) or not record.node_id:
                issues.append(ValidationIssue("invalid_id", f"geometry.node_geometry[{index}].node_id", "geometry node ID must be a nonempty string"))
    if require_tuple(history.geometry.sketch_element_geometry, "geometry.sketch_element_geometry"):
        for index, record in enumerate(history.geometry.sketch_element_geometry):
            if not isinstance(record, SketchElementGeometryRecord):
                issues.append(ValidationIssue("geometry_record_type", f"geometry.sketch_element_geometry[{index}]", "expected a SketchElementGeometryRecord"))
                continue
            if not isinstance(record.sketch_id, str) or not record.sketch_id:
                issues.append(ValidationIssue("invalid_id", f"geometry.sketch_element_geometry[{index}].sketch_id", "sketch ID must be a nonempty string"))
            if not isinstance(record.element_id, str) or not record.element_id:
                issues.append(ValidationIssue("invalid_id", f"geometry.sketch_element_geometry[{index}].element_id", "element ID must be a nonempty string"))
    return issues


def validate_history(history: CADHistory) -> None:
    """Raise ValidationError with all detected representation violations."""

    issues = _preflight_issues(history)
    if issues:
        raise ValidationError(issues)
    _Validator(history).run()
