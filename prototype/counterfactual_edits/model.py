"""Typed records for physical edit families and encoded edit samples."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from prototype.controlled_data.factors import PhysicalSource


class EditType(str, Enum):
    PROFILE_EXTENT = "profile_extent"
    EXTRUSION_DISTANCE = "extrusion_distance"
    REVOLVE_ANGLE = "revolve_angle"
    OPERATION_DIRECTION = "operation_direction"
    BOOLEAN_MODE = "boolean_mode"


class AddressKind(str, Enum):
    NODE_FIELD = "node_field"
    NODE_GEOMETRY = "node_geometry"
    SKETCH_ELEMENT_GEOMETRY = "sketch_element_geometry"


@dataclass(frozen=True)
class RepresentationAddress:
    kind: AddressKind
    owner_id: str
    field_path: str
    element_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result = {
            "kind": self.kind.value,
            "owner_id": self.owner_id,
            "field_path": self.field_path,
        }
        if self.element_id is not None:
            result["element_id"] = self.element_id
        return result


@dataclass(frozen=True)
class EditTarget:
    operation_index: int
    node_id: str
    changed_field: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation_index": self.operation_index,
            "node_id": self.node_id,
            "changed_field": self.changed_field,
        }


@dataclass(frozen=True)
class EditCandidate:
    edit_type: EditType
    target: EditTarget
    endpoint_a: PhysicalSource
    endpoint_b: PhysicalSource


@dataclass(frozen=True)
class IdentifiedCandidate:
    candidate: EditCandidate
    lower_source_family_id: str
    higher_source_family_id: str


@dataclass(frozen=True)
class OrientedCandidate:
    identified: IdentifiedCandidate
    base_source: PhysicalSource
    edited_source: PhysicalSource
    is_coverage_anchor: bool


@dataclass(frozen=True)
class LocalityGroundTruth:
    allowed_changed_paths: tuple[RepresentationAddress, ...]
    expected_unchanged_paths: tuple[RepresentationAddress, ...]
    expected_unchanged_edges: tuple[tuple[str, str, str], ...]
    expected_operation_sequence: tuple[str, ...]
    causal_downstream_operation_ids: tuple[str, ...]
    structural_graph_edit_distance: int = 0
    semantic_parameter_edit_distance: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed_changed_paths": [item.to_dict() for item in self.allowed_changed_paths],
            "expected_unchanged_paths": [
                item.to_dict() for item in self.expected_unchanged_paths
            ],
            "expected_unchanged_edges": [list(item) for item in self.expected_unchanged_edges],
            "expected_operation_sequence": list(self.expected_operation_sequence),
            "causal_downstream_operation_ids": list(self.causal_downstream_operation_ids),
            "structural_graph_edit_distance": self.structural_graph_edit_distance,
            "semantic_parameter_edit_distance": self.semantic_parameter_edit_distance,
        }


@dataclass(frozen=True)
class EditFamily:
    base_source: PhysicalSource
    edited_source: PhysicalSource
    base_source_family_id: str
    edited_source_family_id: str
    edit_type: EditType
    target: EditTarget
    before: Any
    after: Any
    locality: LocalityGroundTruth
    edit_family_id: str


@dataclass(frozen=True)
class EditSample:
    edit_sample_id: str
    edit_family_id: str
    geometry_encoding: str
    base_source_family_id: str
    edited_source_family_id: str
    base_sample_id: str
    edited_sample_id: str
    edit_type: EditType
    target: EditTarget
    before: Any
    after: Any
    locality: LocalityGroundTruth
