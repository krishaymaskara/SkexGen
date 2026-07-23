"""Canonical physical history ordering and normalized model fields."""

from __future__ import annotations

from prototype.representation.model import (
    AxisNode,
    ExtrudeNode,
    ReferencePlaneNode,
    RevolveNode,
    SketchNode,
)

from .errors import ModelDataError
from .geometry import (
    decode,
    empty_geometry,
    place_node_geometry,
    place_primitive_geometry,
)
from .records import CanonicalEdge, CanonicalNode, ReconstructionTarget
from .vocab import (
    BOOLEAN_MODES,
    DIRECTIONS,
    EDGE_TYPES,
    LOOP_ROLES,
    MAX_PRIMITIVES,
    NODE_TYPES,
    OPERATION_TYPES,
    PRIMITIVE_TYPES,
    REFERENCE_PLANES,
)


def canonical_nodes_and_edges(history):
    nodes_by_id = {node.node_id: node for node in history.structure.nodes}
    ordered_ids = []
    ordered_ids.extend(
        node.node_id
        for node in sorted(
            (
                item
                for item in history.structure.nodes
                if isinstance(item, ReferencePlaneNode)
            ),
            key=lambda item: item.node_id,
        )
    )
    for operation_id in history.structure.operation_sequence:
        operation = nodes_by_id[operation_id]
        profile_id = _edge_target(history, operation_id, "uses_profile")
        sketch_id = _edge_target(history, profile_id, "defined_in")
        ordered_ids.extend((sketch_id, profile_id))
        if isinstance(operation, RevolveNode):
            ordered_ids.append(_edge_target(history, operation_id, "uses_axis"))
        ordered_ids.append(operation_id)
    remaining = sorted(set(nodes_by_id) - set(ordered_ids))
    ordered_ids.extend(remaining)
    if len(ordered_ids) != len(set(ordered_ids)) or set(ordered_ids) != set(nodes_by_id):
        raise ModelDataError("canonical_node_order", "node ordering is not one-to-one")

    node_geometry = {
        record.node_id: record.geometry for record in history.geometry.node_geometry
    }
    element_geometry = {
        (record.sketch_id, record.element_id): record.geometry
        for record in history.geometry.sketch_element_geometry
    }
    canonical_nodes = tuple(
        _canonical_node(
            nodes_by_id[node_id],
            node_geometry,
            element_geometry,
        )
        for node_id in ordered_ids
    )
    positions = {node_id: index for index, node_id in enumerate(ordered_ids)}
    canonical_edges = tuple(
        CanonicalEdge(edge.source_id, edge.target_id, edge.edge_type.value)
        for edge in sorted(
            history.structure.edges,
            key=lambda item: (
                positions[item.source_id],
                EDGE_TYPES.id(item.edge_type.value),
                positions[item.target_id],
            ),
        )
    )
    return canonical_nodes, canonical_edges


def reconstruction_target(nodes, edges, operation_sequence):
    positions = {node.node_id: index for index, node in enumerate(nodes)}
    edge_sources = tuple(positions[item.source_id] for item in edges)
    edge_targets = tuple(positions[item.target_id] for item in edges)
    attributes = tuple(_categorical_attributes(item) for item in nodes)
    return ReconstructionTarget(
        node_type_ids=tuple(NODE_TYPES.id(item.node_type) for item in nodes),
        categorical_attributes=attributes,
        edge_index=(edge_sources, edge_targets),
        edge_type_ids=tuple(EDGE_TYPES.id(item.edge_type) for item in edges),
        boolean_mode_targets=tuple(
            BOOLEAN_MODES.id(item.boolean_mode) for item in nodes
        ),
        operation_sequence=tuple(positions[item] for item in operation_sequence),
        geometry=tuple(item.geometry for item in nodes),
        geometry_mask=tuple(item.geometry_mask for item in nodes),
    )


def infer_metadata(history):
    nodes = {item.node_id: item for item in history.structure.nodes}
    operation_template = "".join(
        "E" if isinstance(nodes[item], ExtrudeNode) else "R"
        for item in history.structure.operation_sequence
    )
    sketches = [item for item in history.structure.nodes if isinstance(item, SketchNode)]
    families = {_primitive_family(item) for item in sketches}
    if len(families) != 1:
        raise ModelDataError(
            "mixed_primitive_family", "one physical history has mixed sketch families"
        )
    planes = [
        item for item in history.structure.nodes if isinstance(item, ReferencePlaneNode)
    ]
    if len(planes) != 1:
        raise ModelDataError(
            "reference_plane_count", "version 1 requires one reference plane"
        )
    geometry = {
        item.node_id: item.geometry for item in history.geometry.node_geometry
    }
    return operation_template, next(iter(families)), _reference_plane(
        geometry[planes[0].node_id]
    )


def _canonical_node(node, node_geometry, element_geometry):
    values, mask = empty_geometry()
    primitive_types = ()
    loop_role = None
    operation_type = None
    boolean_mode = None
    direction = None
    reference_plane = None
    if node.node_id in node_geometry:
        place_node_geometry(values, mask, node_geometry[node.node_id])
    if isinstance(node, SketchNode):
        primitives = sorted(node.primitives, key=lambda item: item.primitive_id)
        if len(primitives) > MAX_PRIMITIVES:
            raise ModelDataError(
                "too_many_primitives",
                f"{node.node_id} has {len(primitives)} primitives",
            )
        primitive_types = tuple(item.primitive_type.value for item in primitives)
        for slot, primitive in enumerate(primitives):
            key = (node.node_id, primitive.primitive_id)
            if key not in element_geometry:
                raise ModelDataError(
                    "missing_element_geometry", f"missing geometry for {key!r}"
                )
            place_primitive_geometry(values, mask, slot, element_geometry[key])
        roles = {item.role.value for item in node.loops}
        if len(roles) == 1:
            loop_role = next(iter(roles))
    elif isinstance(node, (ExtrudeNode, RevolveNode)):
        operation_type = node.node_type.value
        boolean_mode = node.boolean_mode.value
        direction = node.direction.value
    elif isinstance(node, ReferencePlaneNode):
        reference_plane = _reference_plane(node_geometry[node.node_id])
    return CanonicalNode(
        node.node_id,
        node.node_type.value,
        operation_type,
        boolean_mode,
        direction,
        reference_plane,
        primitive_types,
        loop_role,
        tuple(values),
        tuple(mask),
    )


def _categorical_attributes(node):
    primitive_ids = [
        PRIMITIVE_TYPES.id(item) for item in node.primitive_types
    ]
    primitive_ids.extend(
        [PRIMITIVE_TYPES.id(None)] * (MAX_PRIMITIVES - len(primitive_ids))
    )
    return (
        OPERATION_TYPES.id(node.operation_type),
        BOOLEAN_MODES.id(node.boolean_mode),
        DIRECTIONS.id(node.direction),
        REFERENCE_PLANES.id(node.reference_plane),
        *primitive_ids,
        LOOP_ROLES.id(node.loop_role),
    )


def _edge_target(history, source_id, edge_type):
    targets = [
        item.target_id
        for item in history.structure.edges
        if item.source_id == source_id and item.edge_type.value == edge_type
    ]
    if len(targets) != 1:
        raise ModelDataError(
            "reference_resolution",
            f"{source_id} has {len(targets)} {edge_type} targets",
        )
    return targets[0]


def _primitive_family(sketch):
    values = tuple(item.primitive_type.value for item in sketch.primitives)
    if values == ("circle",):
        return "circle"
    if len(values) == 4 and set(values) == {"line"}:
        return "rectangle_lines"
    if len(values) == 4 and values.count("line") == 2 and values.count("arc") == 2:
        return "capsule_line_arc"
    raise ModelDataError(
        "unsupported_primitive_family", f"{sketch.node_id}: {values!r}"
    )


def _reference_plane(geometry):
    x_axis = decode(geometry.x_axis)
    y_axis = decode(geometry.y_axis)
    frames = {
        "XY": ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        "XZ": ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        "YZ": ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    }
    for name, frame in frames.items():
        if x_axis == frame[0] and y_axis == frame[1]:
            return name
    raise ModelDataError("unsupported_reference_plane", repr((x_axis, y_axis)))
