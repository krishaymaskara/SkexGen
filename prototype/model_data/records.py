"""Immutable physical examples, adapter arrays, and batches."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class FamilyMetadata:
    operation_template: str
    primitive_family: str
    reference_plane: str
    history_depth: int
    geometry_encodings: tuple[str, ...]
    sample_ids: tuple[str, ...]


@dataclass(frozen=True)
class CanonicalNode:
    node_id: str
    node_type: str
    operation_type: str | None
    boolean_mode: str | None
    direction: str | None
    reference_plane: str | None
    primitive_types: tuple[str, ...]
    loop_role: str | None
    geometry: tuple[float, ...]
    geometry_mask: tuple[bool, ...]


@dataclass(frozen=True)
class CanonicalEdge:
    source_id: str
    target_id: str
    edge_type: str


@dataclass(frozen=True)
class ReconstructionTarget:
    node_type_ids: tuple[int, ...]
    categorical_attributes: tuple[tuple[int, ...], ...]
    edge_index: tuple[tuple[int, ...], tuple[int, ...]]
    edge_type_ids: tuple[int, ...]
    boolean_mode_targets: tuple[int, ...]
    operation_sequence: tuple[int, ...]
    geometry: tuple[tuple[float, ...], ...]
    geometry_mask: tuple[tuple[bool, ...], ...]

    def to_torch(self, torch_module=None) -> dict[str, Any]:
        return _example_to_torch(
            torch_module,
            node_type_ids=(self.node_type_ids, "long"),
            categorical_attributes=(self.categorical_attributes, "long"),
            edge_index=(self.edge_index, "long"),
            edge_type_ids=(self.edge_type_ids, "long"),
            boolean_mode_targets=(self.boolean_mode_targets, "long"),
            operation_sequence=(self.operation_sequence, "long"),
            geometry=(self.geometry, "float32"),
            geometry_mask=(self.geometry_mask, "bool"),
        )


@dataclass(frozen=True)
class PhysicalExample:
    physical_family_id: str
    metadata: FamilyMetadata
    split_name: str
    partition: str
    operation_sequence: tuple[str, ...]
    nodes: tuple[CanonicalNode, ...]
    edges: tuple[CanonicalEdge, ...]
    target: ReconstructionTarget
    canonical_reconstruction_json: str


@dataclass(frozen=True)
class FlatMixedExample:
    physical_family_id: str
    categorical_ids: tuple[tuple[int, ...], ...]
    geometry: tuple[tuple[float, ...], ...]
    geometry_mask: tuple[tuple[bool, ...], ...]
    target: ReconstructionTarget

    def to_torch(self, torch_module=None) -> dict[str, Any]:
        return _example_to_torch(
            torch_module,
            categorical_ids=(self.categorical_ids, "long"),
            geometry=(self.geometry, "float32"),
            geometry_mask=(self.geometry_mask, "bool"),
        )


@dataclass(frozen=True)
class TypedGraphExample:
    physical_family_id: str
    node_type_ids: tuple[int, ...]
    edge_index: tuple[tuple[int, ...], tuple[int, ...]]
    edge_type_ids: tuple[int, ...]
    categorical_attributes: tuple[tuple[int, ...], ...]
    geometry: tuple[tuple[float, ...], ...]
    geometry_mask: tuple[tuple[bool, ...], ...]
    operation_sequence: tuple[int, ...]
    target: ReconstructionTarget

    def to_torch(self, torch_module=None) -> dict[str, Any]:
        return _example_to_torch(
            torch_module,
            node_type_ids=(self.node_type_ids, "long"),
            edge_index=(self.edge_index, "long"),
            edge_type_ids=(self.edge_type_ids, "long"),
            categorical_attributes=(self.categorical_attributes, "long"),
            geometry=(self.geometry, "float32"),
            geometry_mask=(self.geometry_mask, "bool"),
            operation_sequence=(self.operation_sequence, "long"),
        )


@dataclass(frozen=True)
class FlatBatch:
    family_ids: tuple[str, ...]
    categorical_ids: tuple[tuple[tuple[int, ...], ...], ...]
    geometry: tuple[tuple[tuple[float, ...], ...], ...]
    geometry_mask: tuple[tuple[tuple[bool, ...], ...], ...]
    padding_mask: tuple[tuple[bool, ...], ...]
    target: ReconstructionBatch

    def to_torch(self, torch_module=None) -> dict[str, Any]:
        return _example_to_torch(
            torch_module,
            categorical_ids=(self.categorical_ids, "long"),
            geometry=(self.geometry, "float32"),
            geometry_mask=(self.geometry_mask, "bool"),
            padding_mask=(self.padding_mask, "bool"),
        )


@dataclass(frozen=True)
class GraphBatch:
    family_ids: tuple[str, ...]
    node_type_ids: tuple[int, ...]
    edge_index: tuple[tuple[int, ...], tuple[int, ...]]
    edge_type_ids: tuple[int, ...]
    categorical_attributes: tuple[tuple[int, ...], ...]
    geometry: tuple[tuple[float, ...], ...]
    geometry_mask: tuple[tuple[bool, ...], ...]
    graph_offsets: tuple[int, ...]
    edge_offsets: tuple[int, ...]
    node_graph_ids: tuple[int, ...]
    node_mask: tuple[tuple[bool, ...], ...]
    target: ReconstructionBatch

    def to_torch(self, torch_module=None) -> dict[str, Any]:
        return _example_to_torch(
            torch_module,
            node_type_ids=(self.node_type_ids, "long"),
            edge_index=(self.edge_index, "long"),
            edge_type_ids=(self.edge_type_ids, "long"),
            categorical_attributes=(self.categorical_attributes, "long"),
            geometry=(self.geometry, "float32"),
            geometry_mask=(self.geometry_mask, "bool"),
            graph_offsets=(self.graph_offsets, "long"),
            edge_offsets=(self.edge_offsets, "long"),
            node_graph_ids=(self.node_graph_ids, "long"),
            node_mask=(self.node_mask, "bool"),
        )


@dataclass(frozen=True)
class CounterfactualExample:
    source: PhysicalExample
    target: PhysicalExample
    edit_family_id: str
    edit_sample_id: str
    edit_type: str
    edited_attribute_paths: tuple[AttributePath, ...]
    unaffected_attribute_paths: tuple[AttributePath, ...]
    split_name: str
    partition: str
    # Counterfactual records cannot be converted into training records by
    # passing a constructor flag accidentally.
    evaluation_only: bool = field(default=True, init=False)


@dataclass(frozen=True)
class AttributePath:
    kind: str
    owner_id: str
    field_path: str
    element_id: str | None = None


@dataclass(frozen=True)
class ReconstructionBatch:
    node_type_ids: tuple[tuple[int, ...], ...]
    categorical_attributes: tuple[tuple[tuple[int, ...], ...], ...]
    boolean_mode_targets: tuple[tuple[int, ...], ...]
    geometry: tuple[tuple[tuple[float, ...], ...], ...]
    geometry_mask: tuple[tuple[tuple[bool, ...], ...], ...]
    node_mask: tuple[tuple[bool, ...], ...]
    operation_sequence: tuple[tuple[int, ...], ...]
    operation_mask: tuple[tuple[bool, ...], ...]
    edge_index: tuple[tuple[int, ...], tuple[int, ...]]
    edge_type_ids: tuple[int, ...]
    edge_offsets: tuple[int, ...]

    def to_torch(self, torch_module=None) -> dict[str, Any]:
        return _example_to_torch(
            torch_module,
            node_type_ids=(self.node_type_ids, "long"),
            categorical_attributes=(self.categorical_attributes, "long"),
            boolean_mode_targets=(self.boolean_mode_targets, "long"),
            geometry=(self.geometry, "float32"),
            geometry_mask=(self.geometry_mask, "bool"),
            node_mask=(self.node_mask, "bool"),
            operation_sequence=(self.operation_sequence, "long"),
            operation_mask=(self.operation_mask, "bool"),
            edge_index=(self.edge_index, "long"),
            edge_type_ids=(self.edge_type_ids, "long"),
            edge_offsets=(self.edge_offsets, "long"),
        )


def _example_to_torch(torch_module, **fields):
    torch = torch_module
    if torch is None:
        try:
            import torch as imported_torch
        except ImportError as exc:
            raise RuntimeError(
                "PyTorch is required only for to_torch(); canonical arrays remain available"
            ) from exc
        torch = imported_torch
    result = {}
    for name, (value, dtype_name) in fields.items():
        result[name] = torch.tensor(
            value, dtype=getattr(torch, dtype_name)
        ).contiguous()
    return result
