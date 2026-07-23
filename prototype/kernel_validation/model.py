"""Pure-Python execution requests, results, and failure records."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Any


class FailureCategory(str, Enum):
    SCHEMA_OR_PARSE_FAILURE = "schema_or_parse_failure"
    UNSUPPORTED_FEATURE = "unsupported_feature"
    SKETCH_CONSTRUCTION_FAILURE = "sketch_construction_failure"
    PROFILE_FACE_FAILURE = "profile_face_failure"
    INVALID_REFERENCE_PLANE = "invalid_reference_plane"
    INVALID_REVOLVE_AXIS = "invalid_revolve_axis"
    EXTRUDE_FAILURE = "extrude_failure"
    REVOLVE_FAILURE = "revolve_failure"
    BOOLEAN_JOIN_FAILURE = "boolean_join_failure"
    BOOLEAN_CUT_FAILURE = "boolean_cut_failure"
    BOOLEAN_NO_EFFECT = "boolean_no_effect"
    EMPTY_RESULT = "empty_result"
    INVALID_FINAL_SHAPE = "invalid_final_shape"
    UNEXPECTED_KERNEL_EXCEPTION = "unexpected_kernel_exception"


class FailureStage(str, Enum):
    CORPUS_LOADING = "corpus_loading"
    DESERIALIZATION = "deserialization"
    REFERENCE_RESOLUTION = "reference_resolution"
    WORLD_TRANSFORM = "world_transform"
    WIRE_CONSTRUCTION = "wire_construction"
    FACE_CONSTRUCTION = "face_construction"
    FEATURE_CONSTRUCTION = "feature_construction"
    BOOLEAN_OPERATION = "boolean_operation"
    SOLID_VALIDATION = "solid_validation"
    SEMANTIC_EFFECT = "semantic_effect"
    REPORTING = "reporting"


@dataclass(frozen=True)
class ShapeMetrics:
    is_null: bool
    is_valid: bool
    shape_type: str | None
    volume: float | None
    bounding_box: tuple[float, float, float, float, float, float] | None
    solid_count: int
    face_count: int
    edge_count: int
    vertex_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_null": self.is_null,
            "is_valid": self.is_valid,
            "shape_type": self.shape_type,
            "volume": self.volume,
            "bounding_box": list(self.bounding_box) if self.bounding_box is not None else None,
            "solid_count": self.solid_count,
            "face_count": self.face_count,
            "edge_count": self.edge_count,
            "vertex_count": self.vertex_count,
        }


@dataclass(frozen=True)
class InspectedShape:
    shape: object
    metrics: ShapeMetrics


@dataclass(frozen=True)
class OperationResult:
    operation_index: int
    node_id: str
    operation_type: str
    boolean_mode: str
    scheduled: bool = True
    reached: bool = False
    kernel_completed: bool = False
    semantically_effective: bool = False
    failed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation_index": self.operation_index,
            "node_id": self.node_id,
            "operation_type": self.operation_type,
            "boolean_mode": self.boolean_mode,
            "scheduled": self.scheduled,
            "reached": self.reached,
            "kernel_completed": self.kernel_completed,
            "semantically_effective": self.semantically_effective,
            "failed": self.failed,
        }


@dataclass(frozen=True)
class SampleInput:
    source_family_id: str
    sample_id: str
    geometry_encoding: str
    operation_template: str
    primitive_family: str
    relative_json_path: str
    payload: str | None
    loading_error: str | None = None


@dataclass(frozen=True)
class SampleExecutionResult:
    source_family_id: str
    sample_id: str
    geometry_encoding: str
    operation_template: str
    primitive_family: str
    kernel_status: str
    failed_operation_index: int | None
    failure_category: str | None
    failure_stage: str | None
    error_detail: str | None
    final_shape_type: str | None
    final_shape_valid: bool
    final_metrics: ShapeMetrics | None
    operation_results: tuple[OperationResult, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_family_id": self.source_family_id,
            "sample_id": self.sample_id,
            "geometry_encoding": self.geometry_encoding,
            "operation_template": self.operation_template,
            "primitive_family": self.primitive_family,
            "kernel_status": self.kernel_status,
            "failed_operation_index": self.failed_operation_index,
            "failure_category": self.failure_category,
            "failure_stage": self.failure_stage,
            "error_detail": self.error_detail,
            "final_shape_type": self.final_shape_type,
            "final_shape_valid": self.final_shape_valid,
            "final_metrics": self.final_metrics.to_dict() if self.final_metrics else None,
            "operation_results": [item.to_dict() for item in self.operation_results],
        }


@dataclass(frozen=True)
class ExecutionFailure(Exception):
    category: FailureCategory
    stage: FailureStage
    detail: str
    operation_index: int | None = None
    final_metrics: ShapeMetrics | None = None

    def __str__(self) -> str:
        return self.detail


def update_operation(
    operations: tuple[OperationResult, ...], index: int, **changes: Any
) -> tuple[OperationResult, ...]:
    return tuple(
        replace(item, **changes) if item.operation_index == index else item
        for item in operations
    )
