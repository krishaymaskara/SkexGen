"""Target-separated paired batching for the two GE1 encoder views."""

from __future__ import annotations

from dataclasses import dataclass

from prototype.model_data.adapters import adapt_flat_mixed, adapt_typed_graph
from prototype.model_data.batching import collate_flat, collate_graph
from prototype.model_data.records import (
    GraphBatch,
    PhysicalExample,
    ReconstructionBatch,
    TypedGraphExample,
    with_autonomous_stop_supervision,
)
from prototype.model_data.serialization import canonical_record_json

from .canonicalization import GraphCanonicalizationInput, canonicalize_graph
from .errors import GraphEncoderError


EMPTY_PAIRED_BATCH = "empty_paired_batch"
INVALID_PHYSICAL_EXAMPLE = "invalid_physical_example"
DUPLICATE_FAMILY_ID = "duplicate_family_id"
FAMILY_ALIGNMENT_MISMATCH = "family_alignment_mismatch"
CANONICAL_TARGET_MISMATCH = "canonical_target_mismatch"
FLAT_GRAPH_TARGET_MISMATCH = "flat_graph_target_mismatch"
COLLATED_TARGET_MISMATCH = "collated_target_mismatch"
INVALID_PERMUTATION = "invalid_permutation"
INVALID_FLAT_PADDING = "invalid_flat_padding"
INVALID_GRAPH_OFFSETS = "invalid_graph_offsets"
INVALID_EDGE_OFFSETS = "invalid_edge_offsets"
INVALID_NODE_GRAPH_IDS = "invalid_node_graph_ids"
CROSS_GRAPH_EDGE = "cross_graph_edge"
INVALID_NODE_MASK = "invalid_node_mask"

C3_FAILURE_CODES = (
    EMPTY_PAIRED_BATCH,
    INVALID_PHYSICAL_EXAMPLE,
    DUPLICATE_FAMILY_ID,
    FAMILY_ALIGNMENT_MISMATCH,
    CANONICAL_TARGET_MISMATCH,
    FLAT_GRAPH_TARGET_MISMATCH,
    COLLATED_TARGET_MISMATCH,
    INVALID_PERMUTATION,
    INVALID_FLAT_PADDING,
    INVALID_GRAPH_OFFSETS,
    INVALID_EDGE_OFFSETS,
    INVALID_NODE_GRAPH_IDS,
    CROSS_GRAPH_EDGE,
    INVALID_NODE_MASK,
)


@dataclass(frozen=True)
class FlatEncoderInput:
    """Padded flat content with no target or provenance fields."""

    categorical_ids: tuple
    geometry: tuple
    geometry_mask: tuple
    padding_mask: tuple

    def to_torch(self, torch_module=None):
        return _to_torch(
            torch_module,
            categorical_ids=(self.categorical_ids, "long"),
            geometry=(self.geometry, "float32"),
            geometry_mask=(self.geometry_mask, "bool"),
            padding_mask=(self.padding_mask, "bool"),
        )


@dataclass(frozen=True)
class GraphNodeContent:
    """Only semantic node values accepted by the future node-feature path."""

    node_type_ids: tuple
    categorical_attributes: tuple
    geometry: tuple
    geometry_mask: tuple

    def to_torch(self, torch_module=None):
        return _to_torch(
            torch_module,
            node_type_ids=(self.node_type_ids, "long"),
            categorical_attributes=(self.categorical_attributes, "long"),
            geometry=(self.geometry, "float32"),
            geometry_mask=(self.geometry_mask, "bool"),
        )


@dataclass(frozen=True)
class GraphSemanticInput:
    """Concatenated target-free graph semantics, excluding bookkeeping."""

    node_content: GraphNodeContent
    edge_index: tuple
    edge_type_ids: tuple

    @property
    def node_type_ids(self):
        return self.node_content.node_type_ids

    @property
    def categorical_attributes(self):
        return self.node_content.categorical_attributes

    @property
    def geometry(self):
        return self.node_content.geometry

    @property
    def geometry_mask(self):
        return self.node_content.geometry_mask

    def to_torch(self, torch_module=None):
        result = self.node_content.to_torch(torch_module)
        result.update(
            _to_torch(
                torch_module,
                edge_index=(self.edge_index, "long"),
                edge_type_ids=(self.edge_type_ids, "long"),
            )
        )
        return result


@dataclass(frozen=True)
class GraphBookkeeping:
    """Graph addressing and membership values, never semantic features."""

    graph_offsets: tuple
    edge_offsets: tuple
    node_graph_ids: tuple
    node_mask: tuple

    def to_torch(self, torch_module=None):
        return _to_torch(
            torch_module,
            graph_offsets=(self.graph_offsets, "long"),
            edge_offsets=(self.edge_offsets, "long"),
            node_graph_ids=(self.node_graph_ids, "long"),
            node_mask=(self.node_mask, "bool"),
        )


@dataclass(frozen=True)
class PairedBatch:
    """Aligned target-free encoder inputs and one separate shared target."""

    family_ids: tuple
    flat_input: FlatEncoderInput
    graph_input: GraphSemanticInput
    graph_bookkeeping: GraphBookkeeping
    target: ReconstructionBatch


def build_paired_batch(examples, *, supervise_terminator=False):
    """Build deterministic flat/graph views over the same physical families."""

    ordered = _ordered_physical_examples(examples)
    flat_examples = []
    graph_examples = []
    node_counts = []
    edge_counts = []

    for example in ordered:
        flat = adapt_flat_mixed(example)
        graph = adapt_typed_graph(example)
        _assert_family_alignment(example, flat, graph)
        if _record_bytes(flat.target) != _record_bytes(graph.target):
            _fail(
                FLAT_GRAPH_TARGET_MISMATCH,
                "adapted flat and graph targets differ for one family",
            )

        narrow = GraphCanonicalizationInput(
            graph.node_type_ids,
            graph.edge_index,
            graph.edge_type_ids,
            graph.categorical_attributes,
            graph.geometry,
            graph.geometry_mask,
        )
        canonical = canonicalize_graph(narrow)
        _assert_canonical_target(canonical, example.target)
        canonical_graph = TypedGraphExample(
            example.physical_family_id,
            canonical.node_type_ids,
            canonical.edge_index,
            canonical.edge_type_ids,
            canonical.categorical_attributes,
            canonical.geometry,
            canonical.geometry_mask,
            canonical.operation_sequence,
            example.target,
        )
        if _record_bytes(flat.target) != _record_bytes(canonical_graph.target):
            _fail(
                FLAT_GRAPH_TARGET_MISMATCH,
                "canonical flat and graph targets differ for one family",
            )
        flat_examples.append(flat)
        graph_examples.append(canonical_graph)
        node_counts.append(len(canonical.node_type_ids))
        edge_counts.append(len(canonical.edge_type_ids))

    flat_batch = collate_flat(tuple(flat_examples))
    graph_batch = collate_graph(tuple(graph_examples))
    family_ids = tuple(item.physical_family_id for item in ordered)
    if (
        flat_batch.family_ids != family_ids
        or graph_batch.family_ids != family_ids
        or flat_batch.family_ids != graph_batch.family_ids
    ):
        _fail(
            FAMILY_ALIGNMENT_MISMATCH,
            "collated family orders do not match the sorted physical examples",
        )
    if _record_bytes(flat_batch.target) != _record_bytes(graph_batch.target):
        _fail(
            COLLATED_TARGET_MISMATCH,
            "complete flat and graph ReconstructionBatch targets differ",
        )

    flat_input = _flat_input_from_batch(flat_batch, tuple(node_counts))
    graph_input, graph_bookkeeping = _split_graph_batch(
        graph_batch,
        expected_family_ids=family_ids,
        expected_node_counts=tuple(node_counts),
        expected_edge_counts=tuple(edge_counts),
    )
    target = flat_batch.target
    if supervise_terminator:
        target = with_autonomous_stop_supervision(target)
    return PairedBatch(
        family_ids,
        flat_input,
        graph_input,
        graph_bookkeeping,
        target,
    )


def permute_graph(graph, permutation):
    """Relabel one narrow graph under permutation[new] = old, without targets."""

    if not isinstance(graph, GraphCanonicalizationInput):
        _fail(INVALID_PERMUTATION, "graph must be GraphCanonicalizationInput")
    count = len(graph.node_type_ids)
    try:
        order = tuple(permutation)
    except TypeError as exc:
        raise GraphEncoderError(
            INVALID_PERMUTATION, "permutation must be iterable"
        ) from exc
    if len(order) != count:
        _fail(INVALID_PERMUTATION, "permutation length must equal node count")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in order):
        _fail(INVALID_PERMUTATION, "permutation entries must be integers")
    if any(value < 0 or value >= count for value in order):
        _fail(INVALID_PERMUTATION, "permutation entry is outside the graph")
    if len(set(order)) != count:
        _fail(INVALID_PERMUTATION, "permutation must be bijective")
    if any(
        len(field) != count
        for field in (
            graph.categorical_attributes,
            graph.geometry,
            graph.geometry_mask,
        )
    ):
        _fail(INVALID_PERMUTATION, "graph node-aligned fields are misaligned")
    if (
        not isinstance(graph.edge_index, tuple)
        or len(graph.edge_index) != 2
        or any(not isinstance(row, tuple) for row in graph.edge_index)
        or len(graph.edge_index[0]) != len(graph.edge_index[1])
        or len(graph.edge_index[0]) != len(graph.edge_type_ids)
    ):
        _fail(INVALID_PERMUTATION, "graph edge fields are misaligned")

    endpoints = graph.edge_index[0] + graph.edge_index[1]
    if any(
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        or value >= count
        for value in endpoints
    ):
        _fail(INVALID_PERMUTATION, "graph edge endpoint is not a valid local index")

    old_to_new = [None] * count
    for new, old in enumerate(order):
        old_to_new[old] = new
    sources = tuple(old_to_new[value] for value in graph.edge_index[0])
    destinations = tuple(old_to_new[value] for value in graph.edge_index[1])
    return GraphCanonicalizationInput(
        tuple(graph.node_type_ids[old] for old in order),
        (sources, destinations),
        graph.edge_type_ids,
        tuple(graph.categorical_attributes[old] for old in order),
        tuple(graph.geometry[old] for old in order),
        tuple(graph.geometry_mask[old] for old in order),
    )


def _ordered_physical_examples(examples):
    try:
        values = tuple(examples)
    except TypeError as exc:
        raise GraphEncoderError(
            INVALID_PHYSICAL_EXAMPLE, "examples must be iterable"
        ) from exc
    if not values:
        _fail(EMPTY_PAIRED_BATCH, "at least one PhysicalExample is required")
    if any(not isinstance(item, PhysicalExample) for item in values):
        _fail(INVALID_PHYSICAL_EXAMPLE, "every item must be a PhysicalExample")
    if any(
        not isinstance(item.physical_family_id, str)
        or not item.physical_family_id
        for item in values
    ):
        _fail(INVALID_PHYSICAL_EXAMPLE, "family IDs must be nonempty strings")
    ids = tuple(item.physical_family_id for item in values)
    if len(ids) != len(set(ids)):
        _fail(DUPLICATE_FAMILY_ID, "physical family IDs must be unique")
    return tuple(sorted(values, key=lambda item: item.physical_family_id))


def _assert_family_alignment(example, flat, graph):
    family_id = example.physical_family_id
    if flat.physical_family_id != family_id or graph.physical_family_id != family_id:
        _fail(
            FAMILY_ALIGNMENT_MISMATCH,
            "adapted views disagree with the physical-family identity",
        )


def _assert_canonical_target(canonical, target):
    canonical_fields = (
        canonical.node_type_ids,
        canonical.categorical_attributes,
        canonical.geometry,
        canonical.geometry_mask,
        canonical.edge_index,
        canonical.edge_type_ids,
        canonical.operation_sequence,
    )
    target_fields = (
        target.node_type_ids,
        target.categorical_attributes,
        target.geometry,
        target.geometry_mask,
        target.edge_index,
        target.edge_type_ids,
        target.operation_sequence,
    )
    if canonical_fields != target_fields:
        _fail(
            CANONICAL_TARGET_MISMATCH,
            "C2 canonical semantics differ from the authoritative target",
        )


def _flat_input_from_batch(batch, node_counts):
    batch_size = len(batch.family_ids)
    maximum = max(node_counts)
    fields = (
        batch.categorical_ids,
        batch.geometry,
        batch.geometry_mask,
        batch.padding_mask,
    )
    if any(len(field) != batch_size for field in fields) or any(
        len(row) != maximum for field in fields for row in field
    ):
        _fail(INVALID_FLAT_PADDING, "flat padded fields have inconsistent shapes")
    expected_mask = tuple(
        (True,) * count + (False,) * (maximum - count)
        for count in node_counts
    )
    if batch.padding_mask != expected_mask or batch.target.node_mask != expected_mask:
        _fail(INVALID_FLAT_PADDING, "flat padding does not match true node counts")
    if any(not isinstance(value, bool) for row in batch.padding_mask for value in row):
        _fail(INVALID_FLAT_PADDING, "flat padding masks must contain Booleans")
    return FlatEncoderInput(
        batch.categorical_ids,
        batch.geometry,
        batch.geometry_mask,
        batch.padding_mask,
    )


def _split_graph_batch(
    batch,
    expected_family_ids=None,
    expected_node_counts=None,
    expected_edge_counts=None,
):
    """Validate and separate one inherited GraphBatch into C3 boundaries."""

    if not isinstance(batch, GraphBatch):
        _fail(INVALID_GRAPH_OFFSETS, "graph batch must be a GraphBatch")
    family_ids = batch.family_ids
    batch_size = len(family_ids)
    if expected_family_ids is not None and family_ids != expected_family_ids:
        _fail(FAMILY_ALIGNMENT_MISMATCH, "graph batch family order differs")

    node_total = len(batch.node_type_ids)
    node_fields = (
        batch.categorical_attributes,
        batch.geometry,
        batch.geometry_mask,
    )
    if any(len(field) != node_total for field in node_fields):
        _fail(INVALID_GRAPH_OFFSETS, "graph node-aligned fields are misaligned")
    graph_offsets = batch.graph_offsets
    if not _valid_offsets(graph_offsets, batch_size, node_total):
        _fail(INVALID_GRAPH_OFFSETS, "graph offsets must span all nodes monotonically")
    node_counts = tuple(
        graph_offsets[index + 1] - graph_offsets[index]
        for index in range(batch_size)
    )
    if expected_node_counts is not None and node_counts != expected_node_counts:
        _fail(INVALID_GRAPH_OFFSETS, "graph intervals differ from source node counts")

    edge_total = len(batch.edge_type_ids)
    if (
        not isinstance(batch.edge_index, tuple)
        or len(batch.edge_index) != 2
        or len(batch.edge_index[0]) != edge_total
        or len(batch.edge_index[1]) != edge_total
    ):
        _fail(INVALID_EDGE_OFFSETS, "graph edge fields are misaligned")
    edge_offsets = batch.edge_offsets
    if not _valid_offsets(edge_offsets, batch_size, edge_total):
        _fail(INVALID_EDGE_OFFSETS, "edge offsets must span all edges monotonically")
    edge_counts = tuple(
        edge_offsets[index + 1] - edge_offsets[index]
        for index in range(batch_size)
    )
    if expected_edge_counts is not None and edge_counts != expected_edge_counts:
        _fail(INVALID_EDGE_OFFSETS, "edge intervals differ from source edge counts")

    expected_graph_ids = tuple(
        graph_index
        for graph_index, count in enumerate(node_counts)
        for _ in range(count)
    )
    if batch.node_graph_ids != expected_graph_ids or any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in batch.node_graph_ids
    ):
        _fail(INVALID_NODE_GRAPH_IDS, "node graph IDs disagree with graph intervals")

    maximum = max(node_counts, default=0)
    expected_node_mask = tuple(
        (True,) * count + (False,) * (maximum - count)
        for count in node_counts
    )
    if batch.node_mask != expected_node_mask or any(
        not isinstance(value, bool) for row in batch.node_mask for value in row
    ):
        _fail(INVALID_NODE_MASK, "node mask disagrees with graph-local lengths")

    for graph_index in range(batch_size):
        node_start = graph_offsets[graph_index]
        node_stop = graph_offsets[graph_index + 1]
        edge_start = edge_offsets[graph_index]
        edge_stop = edge_offsets[graph_index + 1]
        for source, destination in zip(
            batch.edge_index[0][edge_start:edge_stop],
            batch.edge_index[1][edge_start:edge_stop],
        ):
            if (
                isinstance(source, bool)
                or not isinstance(source, int)
                or isinstance(destination, bool)
                or not isinstance(destination, int)
                or not node_start <= source < node_stop
                or not node_start <= destination < node_stop
            ):
                _fail(CROSS_GRAPH_EDGE, "an edge crosses a graph-family boundary")

    semantic = GraphSemanticInput(
        GraphNodeContent(
            batch.node_type_ids,
            batch.categorical_attributes,
            batch.geometry,
            batch.geometry_mask,
        ),
        batch.edge_index,
        batch.edge_type_ids,
    )
    bookkeeping = GraphBookkeeping(
        graph_offsets,
        edge_offsets,
        batch.node_graph_ids,
        batch.node_mask,
    )
    return semantic, bookkeeping


def _valid_offsets(offsets, batch_size, total):
    return (
        isinstance(offsets, tuple)
        and len(offsets) == batch_size + 1
        and all(isinstance(value, int) and not isinstance(value, bool) for value in offsets)
        and offsets[0] == 0
        and offsets[-1] == total
        and all(left <= right for left, right in zip(offsets, offsets[1:]))
    )


def _record_bytes(value):
    return canonical_record_json(value).encode("utf-8")


def _to_torch(torch_module, **fields):
    torch = torch_module
    if torch is None:
        try:
            import torch as imported_torch
        except ImportError as exc:
            raise RuntimeError(
                "PyTorch is required only for to_torch(); tuple records remain available"
            ) from exc
        torch = imported_torch
    return {
        name: torch.tensor(value, dtype=getattr(torch, dtype_name)).contiguous()
        for name, (value, dtype_name) in fields.items()
    }


def _fail(code, detail):
    raise GraphEncoderError(code=code, detail=detail)
