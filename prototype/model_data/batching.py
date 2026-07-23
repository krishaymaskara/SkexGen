"""Stable padding and graph-offset collation."""

from __future__ import annotations

from .errors import ModelDataError
from .geometry import GEOMETRY_WIDTH
from .records import FlatBatch, GraphBatch, ReconstructionBatch
from .vocab import FLAT_CATEGORICAL_FIELDS, CATEGORICAL_ATTRIBUTE_FIELDS


def collate_flat(examples):
    ordered = _ordered(examples)
    maximum = max((len(item.categorical_ids) for item in ordered), default=0)
    categorical = []
    geometry = []
    geometry_mask = []
    padding_mask = []
    for item in ordered:
        count = len(item.categorical_ids)
        categorical.append(
            item.categorical_ids
            + ((0,) * len(FLAT_CATEGORICAL_FIELDS),) * (maximum - count)
        )
        geometry.append(
            item.geometry + ((0.0,) * GEOMETRY_WIDTH,) * (maximum - count)
        )
        geometry_mask.append(
            item.geometry_mask + ((False,) * GEOMETRY_WIDTH,) * (maximum - count)
        )
        padding_mask.append((True,) * count + (False,) * (maximum - count))
    return FlatBatch(
        tuple(item.physical_family_id for item in ordered),
        tuple(categorical),
        tuple(geometry),
        tuple(geometry_mask),
        tuple(padding_mask),
        _collate_targets(tuple(item.target for item in ordered)),
    )


def collate_graph(examples):
    ordered = _ordered(examples)
    node_type_ids = []
    edge_sources = []
    edge_targets = []
    edge_type_ids = []
    attributes = []
    geometry = []
    geometry_mask = []
    graph_offsets = [0]
    edge_offsets = [0]
    graph_ids = []
    maximum = max((len(item.node_type_ids) for item in ordered), default=0)
    node_mask = []
    for graph_index, item in enumerate(ordered):
        offset = len(node_type_ids)
        node_type_ids.extend(item.node_type_ids)
        attributes.extend(item.categorical_attributes)
        geometry.extend(item.geometry)
        geometry_mask.extend(item.geometry_mask)
        edge_sources.extend(offset + value for value in item.edge_index[0])
        edge_targets.extend(offset + value for value in item.edge_index[1])
        edge_type_ids.extend(item.edge_type_ids)
        graph_ids.extend([graph_index] * len(item.node_type_ids))
        graph_offsets.append(len(node_type_ids))
        edge_offsets.append(len(edge_type_ids))
        node_mask.append(
            (True,) * len(item.node_type_ids)
            + (False,) * (maximum - len(item.node_type_ids))
        )
    return GraphBatch(
        tuple(item.physical_family_id for item in ordered),
        tuple(node_type_ids),
        (tuple(edge_sources), tuple(edge_targets)),
        tuple(edge_type_ids),
        tuple(attributes),
        tuple(geometry),
        tuple(geometry_mask),
        tuple(graph_offsets),
        tuple(edge_offsets),
        tuple(graph_ids),
        tuple(node_mask),
        _collate_targets(tuple(item.target for item in ordered)),
    )


def _collate_targets(targets):
    maximum_nodes = max((len(item.node_type_ids) for item in targets), default=0)
    maximum_operations = max(
        (len(item.operation_sequence) for item in targets), default=0
    )
    node_types = []
    attributes = []
    booleans = []
    geometry = []
    geometry_mask = []
    node_mask = []
    operations = []
    operation_mask = []
    edge_sources = []
    edge_targets = []
    edge_types = []
    edge_offsets = [0]
    node_offset = 0
    for item in targets:
        node_count = len(item.node_type_ids)
        node_types.append(item.node_type_ids + (0,) * (maximum_nodes - node_count))
        attributes.append(
            item.categorical_attributes
            + ((0,) * len(CATEGORICAL_ATTRIBUTE_FIELDS),)
            * (maximum_nodes - node_count)
        )
        booleans.append(
            item.boolean_mode_targets + (0,) * (maximum_nodes - node_count)
        )
        geometry.append(
            item.geometry
            + ((0.0,) * GEOMETRY_WIDTH,) * (maximum_nodes - node_count)
        )
        geometry_mask.append(
            item.geometry_mask
            + ((False,) * GEOMETRY_WIDTH,) * (maximum_nodes - node_count)
        )
        node_mask.append((True,) * node_count + (False,) * (maximum_nodes - node_count))
        operations.append(
            item.operation_sequence
            + (-1,) * (maximum_operations - len(item.operation_sequence))
        )
        operation_mask.append(
            (True,) * len(item.operation_sequence)
            + (False,) * (maximum_operations - len(item.operation_sequence))
        )
        edge_sources.extend(node_offset + value for value in item.edge_index[0])
        edge_targets.extend(node_offset + value for value in item.edge_index[1])
        edge_types.extend(item.edge_type_ids)
        edge_offsets.append(len(edge_types))
        node_offset += node_count
    return ReconstructionBatch(
        tuple(node_types),
        tuple(attributes),
        tuple(booleans),
        tuple(geometry),
        tuple(geometry_mask),
        tuple(node_mask),
        tuple(operations),
        tuple(operation_mask),
        (tuple(edge_sources), tuple(edge_targets)),
        tuple(edge_types),
        tuple(edge_offsets),
    )


def _ordered(examples):
    ordered = tuple(sorted(examples, key=lambda item: item.physical_family_id))
    if not ordered:
        raise ModelDataError("empty_batch", "at least one example is required")
    ids = [item.physical_family_id for item in ordered]
    if len(ids) != len(set(ids)):
        raise ModelDataError("duplicate_batch_family", "family IDs must be unique")
    return ordered
