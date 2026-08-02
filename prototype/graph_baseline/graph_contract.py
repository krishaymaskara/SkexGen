"""Canonical directed typed-graph contract for controlled CAD histories."""

from __future__ import annotations

from dataclasses import dataclass, replace

from prototype.model_data.records import ReconstructionTarget
from prototype.model_data.vocab import EDGE_TYPES, NODE_TYPES


GRAPH_REPRESENTATION_NAME = "B0-CONTROLLED-CAD-TYPED-GRAPH-v1"
GRAPH_CONTRACT_VERSION = 1
GRAPH_TENSOR_VERSION = 1
GRAPH_EDGE_CLASS_ORDER = (
    "<none>",
    "defined_in",
    "depends_on",
    "placed_on",
    "uses_axis",
    "uses_profile",
)
GRAPH_DIRECTION_CONVENTION = "directed_source_to_destination"
GRAPH_ACTIVE_PAIR_CONTRACT = "active-nodes-excluding-self-pairs-v1"
GRAPH_MINIMAL_MASK_CONTRACT = "active-nonself-and-node-type-compatible-v1"


class GraphContractError(ValueError):
    """An externally supplied graph violates the graph-v1 representation."""

    def __init__(self, code, detail):
        self.code = code
        self.detail = detail
        super().__init__("{}: {}".format(code, detail))


@dataclass(frozen=True)
class DirectedTypedEdge:
    source: int
    destination: int
    edge_type_id: int


@dataclass(frozen=True)
class CanonicalGraphRecord:
    representation_name: str
    contract_version: int
    tensor_version: int
    node_type_ids: tuple
    directed_typed_edges: tuple
    active_node_mask: tuple
    active_pair_mask: tuple
    node_count: int


def graph_edge_class_id(name):
    try:
        return GRAPH_EDGE_CLASS_ORDER.index(name)
    except ValueError as exc:
        raise GraphContractError(
            "unsupported_graph_edge_type", repr(name)
        ) from exc


def graph_edge_class_name(class_id):
    if (
        isinstance(class_id, bool)
        or not isinstance(class_id, int)
        or not 0 <= class_id < len(GRAPH_EDGE_CLASS_ORDER)
    ):
        raise GraphContractError(
            "unsupported_graph_edge_type", repr(class_id)
        )
    return GRAPH_EDGE_CLASS_ORDER[class_id]


def graph_class_from_model_data_edge_id(edge_type_id):
    if isinstance(edge_type_id, bool) or not isinstance(edge_type_id, int):
        raise GraphContractError("unsupported_graph_edge_type", repr(edge_type_id))
    if not 0 <= edge_type_id < len(EDGE_TYPES.tokens):
        raise GraphContractError("unsupported_graph_edge_type", repr(edge_type_id))
    token = EDGE_TYPES.tokens[edge_type_id]
    if token in ("<pad>", "<none>"):
        raise GraphContractError("unsupported_graph_edge_type", repr(token))
    return graph_edge_class_id(token)


def model_data_edge_id_from_graph_class(class_id):
    name = graph_edge_class_name(class_id)
    if name == "<none>":
        raise GraphContractError(
            "unsupported_graph_edge_type", "<none> is not a present edge"
        )
    return EDGE_TYPES.id(name)


def graph_from_reconstruction_target(target):
    """Adapt an existing canonical target into deterministic graph-v1 order."""

    if not isinstance(target, ReconstructionTarget):
        raise TypeError("target must be ReconstructionTarget")
    count = len(target.node_type_ids)
    if len(target.edge_index) != 2 or len(target.edge_type_ids) != len(
        target.edge_index[0]
    ) or len(target.edge_index[0]) != len(target.edge_index[1]):
        raise GraphContractError("malformed_graph_edges", "target edges are misaligned")
    edges = tuple(sorted(
        (
            DirectedTypedEdge(
                int(source),
                int(destination),
                graph_class_from_model_data_edge_id(int(edge_type)),
            )
            for source, destination, edge_type in zip(
                target.edge_index[0], target.edge_index[1], target.edge_type_ids
            )
        ),
        key=lambda item: (item.source, item.edge_type_id, item.destination),
    ))
    active_nodes = (True,) * count
    active_pairs = tuple(
        tuple(source != destination for destination in range(count))
        for source in range(count)
    )
    graph = CanonicalGraphRecord(
        GRAPH_REPRESENTATION_NAME,
        GRAPH_CONTRACT_VERSION,
        GRAPH_TENSOR_VERSION,
        tuple(target.node_type_ids),
        edges,
        active_nodes,
        active_pairs,
        count,
    )
    return validate_graph_record(graph)


def reconstruction_target_from_graph(graph, base_target):
    """Replace only structural fields of a canonical target from graph edges."""

    validate_graph_record(graph)
    if not isinstance(base_target, ReconstructionTarget):
        raise TypeError("base_target must be ReconstructionTarget")
    if tuple(base_target.node_type_ids) != graph.node_type_ids:
        raise GraphContractError(
            "graph_node_mismatch", "base target nodes differ from graph nodes"
        )
    edges = graph.directed_typed_edges
    return replace(
        base_target,
        edge_index=(
            tuple(item.source for item in edges),
            tuple(item.destination for item in edges),
        ),
        edge_type_ids=tuple(
            model_data_edge_id_from_graph_class(item.edge_type_id)
            for item in edges
        ),
        operation_sequence=tuple(
            index
            for index, node_id in enumerate(graph.node_type_ids)
            if NODE_TYPES.tokens[node_id] in ("extrude", "revolve")
        ),
    )


def validate_graph_record(graph):
    if not isinstance(graph, CanonicalGraphRecord):
        raise TypeError("graph must be CanonicalGraphRecord")
    if (
        graph.representation_name != GRAPH_REPRESENTATION_NAME
        or graph.contract_version != GRAPH_CONTRACT_VERSION
        or graph.tensor_version != GRAPH_TENSOR_VERSION
    ):
        raise GraphContractError("graph_contract_mismatch", "graph identity differs")
    count = graph.node_count
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise GraphContractError("invalid_graph_node_count", repr(count))
    if len(graph.node_type_ids) != count or graph.active_node_mask != (True,) * count:
        raise GraphContractError("graph_node_mismatch", "active nodes are misaligned")
    if any(
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value < len(NODE_TYPES.tokens)
        or value in (NODE_TYPES.pad_id, NODE_TYPES.id(None))
        for value in graph.node_type_ids
    ):
        raise GraphContractError("unsupported_graph_node_type", repr(graph.node_type_ids))
    expected_pairs = tuple(
        tuple(source != destination for destination in range(count))
        for source in range(count)
    )
    if graph.active_pair_mask != expected_pairs:
        raise GraphContractError("invalid_active_pair_mask", "self pairs must be inactive")
    seen_pairs = set()
    previous = None
    for edge in graph.directed_typed_edges:
        if not isinstance(edge, DirectedTypedEdge):
            raise GraphContractError("malformed_graph_edges", "edge has wrong type")
        key = (edge.source, edge.edge_type_id, edge.destination)
        if previous is not None and key < previous:
            raise GraphContractError("noncanonical_graph_edge_order", repr(key))
        previous = key
        if (
            isinstance(edge.source, bool)
            or not isinstance(edge.source, int)
            or isinstance(edge.destination, bool)
            or not isinstance(edge.destination, int)
            or not 0 <= edge.source < count
            or not 0 <= edge.destination < count
        ):
            raise GraphContractError("inactive_graph_edge", repr(edge))
        if edge.source == edge.destination:
            raise GraphContractError("forbidden_graph_self_edge", repr(edge))
        graph_edge_class_name(edge.edge_type_id)
        if edge.edge_type_id == 0:
            raise GraphContractError("unsupported_graph_edge_type", "present edge is <none>")
        pair = (edge.source, edge.destination)
        if pair in seen_pairs:
            raise GraphContractError("duplicate_graph_edge", repr(pair))
        seen_pairs.add(pair)
        if not edge_type_is_compatible(
            graph.node_type_ids[edge.source],
            graph.node_type_ids[edge.destination],
            edge.edge_type_id,
        ):
            raise GraphContractError("graph_edge_type_mismatch", repr(edge))
    return graph


def edge_type_is_compatible(source_node_id, destination_node_id, class_id):
    if class_id == 0:
        return True
    source = NODE_TYPES.tokens[source_node_id]
    destination = NODE_TYPES.tokens[destination_node_id]
    name = graph_edge_class_name(class_id)
    allowed = {
        "placed_on": (("sketch",), ("reference_plane",)),
        "defined_in": (("profile", "axis"), ("sketch",)),
        "uses_profile": (("extrude", "revolve"), ("profile",)),
        "uses_axis": (("revolve",), ("axis",)),
        "depends_on": (("extrude", "revolve"), ("extrude", "revolve")),
    }
    source_types, destination_types = allowed[name]
    return source in source_types and destination in destination_types


def position_only_canonical_edges(node_type_ids):
    """Scoring-only topology implied by canonical controlled serialization."""

    tokens = tuple(NODE_TYPES.tokens[item] for item in node_type_ids)
    if not tokens or tokens[0] != "reference_plane":
        raise GraphContractError("graph_node_mismatch", "sequence must start with plane")
    result = []
    operations = []
    position = 1
    while position < len(tokens):
        sketch = position
        profile = position + 1
        if profile >= len(tokens) or tokens[sketch:profile + 1] != ("sketch", "profile"):
            raise GraphContractError("graph_node_mismatch", repr(tokens))
        result.extend((
            DirectedTypedEdge(sketch, 0, graph_edge_class_id("placed_on")),
            DirectedTypedEdge(profile, sketch, graph_edge_class_id("defined_in")),
        ))
        next_position = profile + 1
        axis = None
        if next_position < len(tokens) and tokens[next_position] == "axis":
            axis = next_position
            next_position += 1
            result.append(DirectedTypedEdge(
                axis, sketch, graph_edge_class_id("defined_in")
            ))
        if next_position >= len(tokens) or tokens[next_position] not in ("extrude", "revolve"):
            raise GraphContractError("graph_node_mismatch", repr(tokens))
        operation = next_position
        result.append(DirectedTypedEdge(
            operation, profile, graph_edge_class_id("uses_profile")
        ))
        if tokens[operation] == "revolve":
            if axis is None:
                raise GraphContractError("graph_node_mismatch", "revolve lacks axis")
            result.append(DirectedTypedEdge(
                operation, axis, graph_edge_class_id("uses_axis")
            ))
        elif axis is not None:
            raise GraphContractError("graph_node_mismatch", "extrude has axis")
        if operations:
            result.append(DirectedTypedEdge(
                operation, operations[-1], graph_edge_class_id("depends_on")
            ))
        operations.append(operation)
        position = operation + 1
    return tuple(sorted(
        result, key=lambda item: (item.source, item.edge_type_id, item.destination)
    ))


def graph_contract_metadata():
    return {
        "representation_name": GRAPH_REPRESENTATION_NAME,
        "graph_contract_version": GRAPH_CONTRACT_VERSION,
        "graph_tensor_version": GRAPH_TENSOR_VERSION,
        "edge_vocabulary": list(GRAPH_EDGE_CLASS_ORDER),
        "direction_convention": GRAPH_DIRECTION_CONVENTION,
        "active_pair_construction": GRAPH_ACTIVE_PAIR_CONTRACT,
        "minimal_mask_contract": GRAPH_MINIMAL_MASK_CONTRACT,
        "duplicates_allowed": False,
        "self_edges_allowed": False,
        "multiple_types_per_ordered_pair": False,
    }
