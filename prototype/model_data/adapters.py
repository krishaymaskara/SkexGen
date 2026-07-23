"""Flat/mixed and typed-graph adapters over one canonical physical example."""

from __future__ import annotations

from .records import FlatMixedExample, TypedGraphExample
from .vocab import NODE_TYPES


def adapt_flat_mixed(example):
    """Return chronological rows without explicit dependency-edge inputs."""

    target = example.target
    categorical = tuple(
        (NODE_TYPES.id(node.node_type), *attributes)
        for node, attributes in zip(example.nodes, target.categorical_attributes)
    )
    return FlatMixedExample(
        example.physical_family_id,
        categorical,
        target.geometry,
        target.geometry_mask,
        target,
    )


def adapt_typed_graph(example):
    """Return the shared graph input used by both future graph architectures."""

    target = example.target
    return TypedGraphExample(
        example.physical_family_id,
        target.node_type_ids,
        target.edge_index,
        target.edge_type_ids,
        target.categorical_attributes,
        target.geometry,
        target.geometry_mask,
        target.operation_sequence,
        target,
    )
