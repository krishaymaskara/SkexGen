"""Position-free canonicalization for one controlled GE1 graph."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math

from prototype.graph_baseline.graph_contract import (
    edge_type_is_compatible,
    graph_class_from_model_data_edge_id,
)
from prototype.model_data.vocab import EDGE_TYPES, NODE_TYPES

from .errors import GraphEncoderError


MALFORMED_NODE_FIELDS = "malformed_or_misaligned_node_fields"
INVALID_NODE_TYPE_ID = "invalid_node_type_id"
INVALID_EDGE_TYPE_ID = "invalid_edge_type_id"
MALFORMED_EDGE_INDEX = "malformed_edge_index"
OUT_OF_RANGE_EDGE_ENDPOINT = "out_of_range_edge_endpoint"
INCOMPATIBLE_TYPED_EDGE = "incompatible_typed_edge"
DUPLICATE_EDGE = "duplicate_edge"
SELF_EDGE = "self_edge"
UNSUPPORTED_PLANE_COUNT = "unsupported_plane_count_for_controlled_scope"
UNSUPPORTED_OPERATION_COUNT = "unsupported_operation_count_for_controlled_scope"
OPERATION_CHAIN_BRANCHING = "operation_chain_branching"
OPERATION_CHAIN_CYCLE = "operation_chain_cycle"
DISCONNECTED_OPERATION_CHAIN = "disconnected_operation_chain"
AMBIGUOUS_OPERATION_CHAIN = "ambiguous_operation_chain"
MISSING_OR_MULTIPLE_OPERATION_PROFILE = "missing_or_multiple_operation_profile"
INVALID_PROFILE_TARGET_TYPE = "invalid_profile_target_type"
MISSING_OR_MULTIPLE_PROFILE_DEFINED_IN = "missing_or_multiple_profile_defined_in"
INVALID_PROFILE_SKETCH_TARGET_TYPE = "invalid_profile_sketch_target_type"
MISSING_FORBIDDEN_OR_MULTIPLE_OPERATION_AXIS = (
    "missing_forbidden_or_multiple_operation_axis"
)
INVALID_AXIS_TARGET_TYPE = "invalid_axis_target_type"
MISSING_OR_MULTIPLE_AXIS_DEFINED_IN = "missing_or_multiple_axis_defined_in"
INVALID_AXIS_SKETCH_TARGET_TYPE = "invalid_axis_sketch_target_type"
AXIS_PROFILE_SKETCH_MISMATCH = "axis_profile_sketch_mismatch"
MISSING_OR_MULTIPLE_SKETCH_PLACED_ON = "missing_or_multiple_sketch_placed_on"
PLACEMENT_ON_NON_PLANE_NODE = "placement_on_non_plane_node"
PLACEMENT_ON_WRONG_SHARED_PLANE = "placement_on_wrong_shared_plane"
MULTIPLY_ASSIGNED_NODE = "multiply_assigned_node"
UNASSIGNED_NODE = "unassigned_node"

FAILURE_CODES = (
    MALFORMED_NODE_FIELDS,
    INVALID_NODE_TYPE_ID,
    INVALID_EDGE_TYPE_ID,
    MALFORMED_EDGE_INDEX,
    OUT_OF_RANGE_EDGE_ENDPOINT,
    INCOMPATIBLE_TYPED_EDGE,
    DUPLICATE_EDGE,
    SELF_EDGE,
    UNSUPPORTED_PLANE_COUNT,
    UNSUPPORTED_OPERATION_COUNT,
    OPERATION_CHAIN_BRANCHING,
    OPERATION_CHAIN_CYCLE,
    DISCONNECTED_OPERATION_CHAIN,
    AMBIGUOUS_OPERATION_CHAIN,
    MISSING_OR_MULTIPLE_OPERATION_PROFILE,
    INVALID_PROFILE_TARGET_TYPE,
    MISSING_OR_MULTIPLE_PROFILE_DEFINED_IN,
    INVALID_PROFILE_SKETCH_TARGET_TYPE,
    MISSING_FORBIDDEN_OR_MULTIPLE_OPERATION_AXIS,
    INVALID_AXIS_TARGET_TYPE,
    MISSING_OR_MULTIPLE_AXIS_DEFINED_IN,
    INVALID_AXIS_SKETCH_TARGET_TYPE,
    AXIS_PROFILE_SKETCH_MISMATCH,
    MISSING_OR_MULTIPLE_SKETCH_PLACED_ON,
    PLACEMENT_ON_NON_PLANE_NODE,
    PLACEMENT_ON_WRONG_SHARED_PLANE,
    MULTIPLY_ASSIGNED_NODE,
    UNASSIGNED_NODE,
)

_PLANE = NODE_TYPES.id("reference_plane")
_SKETCH = NODE_TYPES.id("sketch")
_PROFILE = NODE_TYPES.id("profile")
_AXIS = NODE_TYPES.id("axis")
_EXTRUDE = NODE_TYPES.id("extrude")
_REVOLVE = NODE_TYPES.id("revolve")
_OPERATION_TYPES = (_EXTRUDE, _REVOLVE)

_PLACED_ON = EDGE_TYPES.id("placed_on")
_DEFINED_IN = EDGE_TYPES.id("defined_in")
_USES_PROFILE = EDGE_TYPES.id("uses_profile")
_USES_AXIS = EDGE_TYPES.id("uses_axis")
_DEPENDS_ON = EDGE_TYPES.id("depends_on")


@dataclass(frozen=True)
class GraphCanonicalizationInput:
    """Narrow single-graph content with no target or chronology fields."""

    node_type_ids: tuple
    edge_index: tuple
    edge_type_ids: tuple
    categorical_attributes: tuple
    geometry: tuple
    geometry_mask: tuple


@dataclass(frozen=True)
class CanonicalizedGraph:
    """Immutable canonical graph and incoming-index correspondence."""

    canonical_to_old: tuple
    old_to_canonical: tuple
    node_type_ids: tuple
    categorical_attributes: tuple
    geometry: tuple
    geometry_mask: tuple
    edge_index: tuple
    edge_type_ids: tuple
    operation_sequence: tuple


def canonicalize_graph(graph):
    """Recover one controlled executable order using typed relations only."""

    _validate_node_fields(graph)
    count = len(graph.node_type_ids)
    edges = _validate_edges(graph, count)
    by_source_type = _edges_by_source_type(edges)

    planes = tuple(
        index for index, value in enumerate(graph.node_type_ids) if value == _PLANE
    )
    operations = tuple(
        index
        for index, value in enumerate(graph.node_type_ids)
        if value in _OPERATION_TYPES
    )
    if not planes:
        _fail(UNSUPPORTED_PLANE_COUNT, "controlled GE1 requires one plane")
    if not operations:
        _fail(UNSUPPORTED_OPERATION_COUNT, "controlled GE1 requires one operation")

    operation_order = _operation_order(operations, by_source_type)
    if len(operations) not in (1, 2):
        _fail(UNSUPPORTED_OPERATION_COUNT, "controlled GE1 supports one or two operations")

    groups = []
    placed_plane_targets = []
    for operation in operation_order:
        group = _recover_group(
            operation,
            graph.node_type_ids,
            by_source_type,
        )
        groups.append(group)
        sketch = group[0]
        targets = _targets(by_source_type, sketch, _PLACED_ON)
        if len(targets) != 1:
            _fail(
                MISSING_OR_MULTIPLE_SKETCH_PLACED_ON,
                "each recovered sketch requires one placed_on edge",
            )
        target = targets[0]
        if graph.node_type_ids[target] != _PLANE:
            _fail(PLACEMENT_ON_NON_PLANE_NODE, "placed_on must target a plane")
        placed_plane_targets.append(target)

    distinct_placed_planes = tuple(sorted(set(placed_plane_targets)))
    if len(distinct_placed_planes) > 1:
        _fail(
            PLACEMENT_ON_WRONG_SHARED_PLANE,
            "controlled sketches do not share one reference plane",
        )
    if len(planes) != 1:
        _fail(UNSUPPORTED_PLANE_COUNT, "controlled GE1 requires exactly one plane")
    shared_plane = planes[0]
    if any(target != shared_plane for target in placed_plane_targets):
        _fail(
            PLACEMENT_ON_WRONG_SHARED_PLANE,
            "a sketch is not placed on the shared plane",
        )

    assigned = {shared_plane}
    canonical_to_old = [shared_plane]
    operation_indices = []
    for group in groups:
        for node in group:
            if node in assigned:
                _fail(MULTIPLY_ASSIGNED_NODE, "a node belongs to multiple groups")
            assigned.add(node)
            canonical_to_old.append(node)
        operation_indices.append(len(canonical_to_old) - 1)
    if len(assigned) != count:
        _fail(UNASSIGNED_NODE, "controlled graph contains an unassigned node")

    canonical_to_old_tuple = tuple(canonical_to_old)
    old_to_canonical = [None] * count
    for canonical, old in enumerate(canonical_to_old_tuple):
        old_to_canonical[old] = canonical
    old_to_canonical_tuple = tuple(old_to_canonical)

    canonical_edges = tuple(sorted(
        (
            old_to_canonical_tuple[source],
            edge_type,
            old_to_canonical_tuple[destination],
        )
        for source, destination, edge_type in edges
    ))
    return CanonicalizedGraph(
        canonical_to_old_tuple,
        old_to_canonical_tuple,
        tuple(graph.node_type_ids[index] for index in canonical_to_old_tuple),
        tuple(graph.categorical_attributes[index] for index in canonical_to_old_tuple),
        tuple(graph.geometry[index] for index in canonical_to_old_tuple),
        tuple(graph.geometry_mask[index] for index in canonical_to_old_tuple),
        (
            tuple(item[0] for item in canonical_edges),
            tuple(item[2] for item in canonical_edges),
        ),
        tuple(item[1] for item in canonical_edges),
        tuple(operation_indices),
    )


def _validate_node_fields(graph):
    if not isinstance(graph, GraphCanonicalizationInput):
        _fail(MALFORMED_NODE_FIELDS, "input must be GraphCanonicalizationInput")
    fields = (
        graph.node_type_ids,
        graph.categorical_attributes,
        graph.geometry,
        graph.geometry_mask,
    )
    if any(not isinstance(value, tuple) for value in fields):
        _fail(MALFORMED_NODE_FIELDS, "node-aligned fields must be tuples")
    count = len(graph.node_type_ids)
    if count <= 0 or any(len(value) != count for value in fields[1:]):
        _fail(MALFORMED_NODE_FIELDS, "node-aligned fields are misaligned")
    for node_type in graph.node_type_ids:
        if (
            isinstance(node_type, bool)
            or not isinstance(node_type, int)
            or not 0 <= node_type < len(NODE_TYPES.tokens)
            or node_type in (NODE_TYPES.pad_id, NODE_TYPES.id(None))
        ):
            _fail(INVALID_NODE_TYPE_ID, "node type ID is unsupported")
    for row in graph.categorical_attributes:
        if (
            not isinstance(row, tuple)
            or len(row) != 9
            or any(isinstance(value, bool) or not isinstance(value, int) for value in row)
        ):
            _fail(MALFORMED_NODE_FIELDS, "categorical rows must contain nine integers")
    for row in graph.geometry:
        if (
            not isinstance(row, tuple)
            or len(row) != 39
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in row
            )
        ):
            _fail(MALFORMED_NODE_FIELDS, "geometry rows must contain 39 finite numbers")
    for row in graph.geometry_mask:
        if (
            not isinstance(row, tuple)
            or len(row) != 39
            or any(not isinstance(value, bool) for value in row)
        ):
            _fail(MALFORMED_NODE_FIELDS, "geometry masks must contain 39 Booleans")


def _validate_edges(graph, count):
    if (
        not isinstance(graph.edge_index, tuple)
        or len(graph.edge_index) != 2
        or any(not isinstance(row, tuple) for row in graph.edge_index)
        or len(graph.edge_index[0]) != len(graph.edge_index[1])
        or not isinstance(graph.edge_type_ids, tuple)
        or len(graph.edge_type_ids) != len(graph.edge_index[0])
    ):
        _fail(MALFORMED_EDGE_INDEX, "edge fields are misaligned")
    edges = []
    seen_pairs = set()
    for source, destination, edge_type in zip(
        graph.edge_index[0], graph.edge_index[1], graph.edge_type_ids
    ):
        if (
            isinstance(source, bool)
            or not isinstance(source, int)
            or isinstance(destination, bool)
            or not isinstance(destination, int)
        ):
            _fail(MALFORMED_EDGE_INDEX, "edge endpoints must be integers")
        if not 0 <= source < count or not 0 <= destination < count:
            _fail(OUT_OF_RANGE_EDGE_ENDPOINT, "edge endpoint is outside the graph")
        if source == destination:
            _fail(SELF_EDGE, "self edges are forbidden")
        if (
            isinstance(edge_type, bool)
            or not isinstance(edge_type, int)
            or not 0 <= edge_type < len(EDGE_TYPES.tokens)
            or edge_type in (EDGE_TYPES.pad_id, EDGE_TYPES.id(None))
        ):
            _fail(INVALID_EDGE_TYPE_ID, "edge type ID is unsupported")
        pair = (source, destination)
        if pair in seen_pairs:
            _fail(DUPLICATE_EDGE, "one ordered pair may carry only one edge")
        seen_pairs.add(pair)
        _validate_edge_compatibility(
            graph.node_type_ids[source],
            graph.node_type_ids[destination],
            edge_type,
        )
        edges.append((source, destination, edge_type))
    return tuple(edges)


def _validate_edge_compatibility(source_type, destination_type, edge_type):
    class_id = graph_class_from_model_data_edge_id(edge_type)
    if edge_type_is_compatible(source_type, destination_type, class_id):
        return
    if edge_type == _USES_PROFILE and source_type in _OPERATION_TYPES:
        _fail(INVALID_PROFILE_TARGET_TYPE, "uses_profile must target a profile")
    if edge_type == _DEFINED_IN and source_type == _PROFILE:
        _fail(INVALID_PROFILE_SKETCH_TARGET_TYPE, "profile defined_in must target a sketch")
    if edge_type == _USES_AXIS and source_type == _REVOLVE:
        _fail(INVALID_AXIS_TARGET_TYPE, "uses_axis must target an axis")
    if edge_type == _DEFINED_IN and source_type == _AXIS:
        _fail(INVALID_AXIS_SKETCH_TARGET_TYPE, "axis defined_in must target a sketch")
    if edge_type == _PLACED_ON and source_type == _SKETCH:
        _fail(PLACEMENT_ON_NON_PLANE_NODE, "placed_on must target a plane")
    _fail(INCOMPATIBLE_TYPED_EDGE, "typed edge violates the graph contract")


def _edges_by_source_type(edges):
    result = {}
    for source, destination, edge_type in edges:
        result.setdefault((source, edge_type), []).append(destination)
    return {
        key: tuple(sorted(values))
        for key, values in sorted(result.items())
    }


def _targets(by_source_type, source, edge_type):
    return by_source_type.get((source, edge_type), ())


def _operation_order(operations, by_source_type):
    operation_set = set(operations)
    outgoing = {
        operation: tuple(
            target
            for target in _targets(by_source_type, operation, _DEPENDS_ON)
            if target in operation_set
        )
        for operation in operations
    }
    incoming_counts = Counter(
        target for targets in outgoing.values() for target in targets
    )
    if any(len(targets) > 1 for targets in outgoing.values()) or any(
        value > 1 for value in incoming_counts.values()
    ):
        _fail(OPERATION_CHAIN_BRANCHING, "operation dependency is not linear")
    if _has_operation_cycle(operations, outgoing):
        _fail(OPERATION_CHAIN_CYCLE, "operation dependency contains a cycle")
    edge_count = sum(len(values) for values in outgoing.values())
    if len(operations) == 1:
        if edge_count:
            _fail(DISCONNECTED_OPERATION_CHAIN, "single operation has a dependency")
        return operations
    if edge_count == 0:
        _fail(AMBIGUOUS_OPERATION_CHAIN, "operation chronology is not represented")
    if edge_count != len(operations) - 1:
        _fail(DISCONNECTED_OPERATION_CHAIN, "operation dependency is disconnected")
    roots = tuple(operation for operation in operations if not outgoing[operation])
    finals = tuple(
        operation for operation in operations if incoming_counts[operation] == 0
    )
    if len(roots) != 1 or len(finals) != 1:
        _fail(AMBIGUOUS_OPERATION_CHAIN, "operation chain endpoints are ambiguous")
    ordered = [roots[0]]
    current = roots[0]
    while len(ordered) < len(operations):
        successors = tuple(
            operation
            for operation in operations
            if outgoing[operation] == (current,)
        )
        if len(successors) != 1:
            _fail(DISCONNECTED_OPERATION_CHAIN, "operation chain cannot be traversed")
        current = successors[0]
        if current in ordered:
            _fail(OPERATION_CHAIN_CYCLE, "operation dependency contains a cycle")
        ordered.append(current)
    return tuple(ordered)


def _has_operation_cycle(operations, outgoing):
    for start in operations:
        seen = set()
        current = start
        while outgoing[current]:
            if current in seen:
                return True
            seen.add(current)
            current = outgoing[current][0]
        if current in seen:
            return True
    return False


def _recover_group(operation, node_types, by_source_type):
    profiles = _targets(by_source_type, operation, _USES_PROFILE)
    if len(profiles) != 1:
        _fail(
            MISSING_OR_MULTIPLE_OPERATION_PROFILE,
            "each operation requires one uses_profile edge",
        )
    profile = profiles[0]
    if node_types[profile] != _PROFILE:
        _fail(INVALID_PROFILE_TARGET_TYPE, "uses_profile must target a profile")
    sketches = _targets(by_source_type, profile, _DEFINED_IN)
    if len(sketches) != 1:
        _fail(
            MISSING_OR_MULTIPLE_PROFILE_DEFINED_IN,
            "each profile requires one defined_in edge",
        )
    sketch = sketches[0]
    if node_types[sketch] != _SKETCH:
        _fail(
            INVALID_PROFILE_SKETCH_TARGET_TYPE,
            "profile defined_in must target a sketch",
        )

    axes = _targets(by_source_type, operation, _USES_AXIS)
    if node_types[operation] == _EXTRUDE:
        if axes:
            _fail(
                MISSING_FORBIDDEN_OR_MULTIPLE_OPERATION_AXIS,
                "extrude must not use an axis",
            )
        return (sketch, profile, operation)
    if len(axes) != 1:
        _fail(
            MISSING_FORBIDDEN_OR_MULTIPLE_OPERATION_AXIS,
            "revolve requires exactly one axis",
        )
    axis = axes[0]
    if node_types[axis] != _AXIS:
        _fail(INVALID_AXIS_TARGET_TYPE, "uses_axis must target an axis")
    axis_sketches = _targets(by_source_type, axis, _DEFINED_IN)
    if len(axis_sketches) != 1:
        _fail(
            MISSING_OR_MULTIPLE_AXIS_DEFINED_IN,
            "each axis requires one defined_in edge",
        )
    axis_sketch = axis_sketches[0]
    if node_types[axis_sketch] != _SKETCH:
        _fail(INVALID_AXIS_SKETCH_TARGET_TYPE, "axis defined_in must target a sketch")
    if axis_sketch != sketch:
        _fail(AXIS_PROFILE_SKETCH_MISMATCH, "axis and profile use different sketches")
    return (sketch, profile, axis, operation)


def _fail(code, detail):
    raise GraphEncoderError(code=code, detail=detail)
