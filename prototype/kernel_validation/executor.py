"""Execute one canonical history through an injected CAD-kernel adapter."""

from __future__ import annotations

import math
import re

from prototype.controlled_data.identity import sample_id, source_family_id
from prototype.representation.model import (
    BooleanMode,
    Direction,
    ExtrudeGeometry,
    ExtrudeNode,
    PrimitiveType,
    RevolveGeometry,
    RevolveNode,
    SketchNode,
)
from prototype.representation.serialization import history_from_json
from prototype.representation.validation import validate_history

from .adapter import KernelAdapter, KernelAdapterError
from .config import (
    BOOLEAN_VOLUME_ABSOLUTE_TOLERANCE,
    BOOLEAN_VOLUME_RELATIVE_TOLERANCE,
    ERROR_DETAIL_LIMIT,
    MIN_SOLID_VOLUME,
)
from .geometry import HistoryIndex, decode_numeric
from .model import (
    ExecutionFailure,
    FailureCategory,
    FailureStage,
    InspectedShape,
    OperationResult,
    SampleExecutionResult,
    SampleInput,
    ShapeMetrics,
    update_operation,
)


def execute_sample(sample: SampleInput, adapter: KernelAdapter) -> SampleExecutionResult:
    """Execute a sample without importing PythonOCC in the orchestration layer."""

    operations: tuple[OperationResult, ...] = ()
    current: InspectedShape | None = None
    active_index: int | None = None
    try:
        if sample.loading_error is not None or sample.payload is None:
            raise ExecutionFailure(
                FailureCategory.SCHEMA_OR_PARSE_FAILURE,
                FailureStage.CORPUS_LOADING,
                sample.loading_error or "sample payload is missing",
            )
        try:
            history = history_from_json(sample.payload)
            validate_history(history)
        except Exception as exc:
            raise ExecutionFailure(
                FailureCategory.SCHEMA_OR_PARSE_FAILURE,
                FailureStage.DESERIALIZATION,
                _exception_detail(exc),
            ) from exc

        index = HistoryIndex(history)
        operation_nodes = [index.nodes[node_id] for node_id in history.structure.operation_sequence]
        operations = tuple(
            OperationResult(
                operation_index=position,
                node_id=node.node_id,
                operation_type=node.node_type.value,
                boolean_mode=node.boolean_mode.value,
            )
            for position, node in enumerate(operation_nodes)
        )
        _verify_identity_and_metadata(sample, history)

        for active_index, node in enumerate(operation_nodes):
            operations = update_operation(operations, active_index, reached=True)
            profile, sketch = index.profile_and_sketch(node.node_id, active_index)
            frame = index.frame_for_sketch(sketch, active_index)
            primitives = index.profile_primitives(profile, sketch, frame, active_index)
            wire = _kernel_call(
                adapter.make_wire,
                primitives,
                category=FailureCategory.SKETCH_CONSTRUCTION_FAILURE,
                stage=FailureStage.WIRE_CONSTRUCTION,
                operation_index=active_index,
            )
            try:
                wire_closed = adapter.wire_is_closed(wire)
                wire_valid = adapter.shape_is_valid(wire)
            except KernelAdapterError as exc:
                raise ExecutionFailure(
                    FailureCategory.SKETCH_CONSTRUCTION_FAILURE,
                    FailureStage.WIRE_CONSTRUCTION,
                    _exception_detail(exc),
                    active_index,
                ) from exc
            if not wire_closed or not wire_valid:
                raise ExecutionFailure(
                    FailureCategory.SKETCH_CONSTRUCTION_FAILURE,
                    FailureStage.WIRE_CONSTRUCTION,
                    "constructed sketch wire is not closed and valid",
                    active_index,
                )
            face = _kernel_call(
                adapter.make_face,
                wire,
                category=FailureCategory.PROFILE_FACE_FAILURE,
                stage=FailureStage.FACE_CONSTRUCTION,
                operation_index=active_index,
            )
            try:
                face_valid = adapter.shape_is_valid(face)
            except KernelAdapterError as exc:
                raise ExecutionFailure(
                    FailureCategory.PROFILE_FACE_FAILURE,
                    FailureStage.FACE_CONSTRUCTION,
                    _exception_detail(exc),
                    active_index,
                ) from exc
            if not face_valid:
                raise ExecutionFailure(
                    FailureCategory.PROFILE_FACE_FAILURE,
                    FailureStage.FACE_CONSTRUCTION,
                    "constructed profile face is invalid",
                    active_index,
                )

            feature = _construct_feature(adapter, index, node, face, frame, active_index)
            if node.boolean_mode is BooleanMode.NEW_BODY:
                operations = update_operation(operations, active_index, kernel_completed=True)
            feature_inspected = _inspect_and_require_solid(adapter, feature, active_index)

            if node.boolean_mode is BooleanMode.NEW_BODY:
                current = feature_inspected
            else:
                if current is None:
                    raise ExecutionFailure(
                        FailureCategory.SCHEMA_OR_PARSE_FAILURE,
                        FailureStage.REFERENCE_RESOLUTION,
                        "Boolean operation has no current body",
                        active_index,
                    )
                category = (
                    FailureCategory.BOOLEAN_JOIN_FAILURE
                    if node.boolean_mode is BooleanMode.JOIN
                    else FailureCategory.BOOLEAN_CUT_FAILURE
                )
                method = adapter.join if node.boolean_mode is BooleanMode.JOIN else adapter.cut
                boolean_shape = _kernel_call(
                    method,
                    current.shape,
                    feature_inspected.shape,
                    category=category,
                    stage=FailureStage.BOOLEAN_OPERATION,
                    operation_index=active_index,
                )
                operations = update_operation(operations, active_index, kernel_completed=True)
                result = _inspect_and_require_solid(adapter, boolean_shape, active_index)
                _require_boolean_effect(current.metrics, result.metrics, node.boolean_mode, active_index)
                current = result

            operations = update_operation(operations, active_index, semantically_effective=True)

        if current is None:
            raise ExecutionFailure(
                FailureCategory.EMPTY_RESULT,
                FailureStage.SOLID_VALIDATION,
                "history produced no final shape",
            )
        return _result(sample, "success", operations, metrics=current.metrics)
    except ExecutionFailure as exc:
        if exc.operation_index is not None and operations:
            operations = update_operation(operations, exc.operation_index, failed=True)
        # A previously valid partial body is not the failed history's final result.
        # Only expose metrics when the failing validation inspected the attempted
        # result itself (including a semantically ineffective Boolean result).
        metrics = exc.final_metrics
        return _result(
            sample,
            "failed",
            operations,
            failure=exc,
            metrics=metrics,
        )
    except Exception as exc:
        if active_index is not None and operations:
            operations = update_operation(operations, active_index, failed=True)
        failure = ExecutionFailure(
            FailureCategory.UNEXPECTED_KERNEL_EXCEPTION,
            FailureStage.REPORTING if active_index is None else FailureStage.FEATURE_CONSTRUCTION,
            _exception_detail(exc),
            active_index,
        )
        return _result(
            sample,
            "failed",
            operations,
            failure=failure,
            metrics=None,
        )


def _verify_identity_and_metadata(sample: SampleInput, history: object) -> None:
    actual_family = source_family_id(history)
    actual_sample = sample_id(history)
    if actual_family != sample.source_family_id or actual_sample != sample.sample_id:
        raise ExecutionFailure(
            FailureCategory.SCHEMA_OR_PARSE_FAILURE,
            FailureStage.CORPUS_LOADING,
            "sample or source-family identity does not match canonical history",
        )
    encodings = {
        value.encoding.value
        for record in history.geometry.node_geometry
        for value in _numeric_values(record.geometry)
    } | {
        value.encoding.value
        for record in history.geometry.sketch_element_geometry
        for value in _numeric_values(record.geometry)
    }
    if encodings != {sample.geometry_encoding}:
        raise ExecutionFailure(
            FailureCategory.SCHEMA_OR_PARSE_FAILURE,
            FailureStage.CORPUS_LOADING,
            "manifest geometry encoding does not match the history",
        )
    nodes = {node.node_id: node for node in history.structure.nodes}
    template = "".join(
        "E" if isinstance(nodes[node_id], ExtrudeNode) else "R"
        for node_id in history.structure.operation_sequence
    )
    if template != sample.operation_template:
        raise ExecutionFailure(
            FailureCategory.SCHEMA_OR_PARSE_FAILURE,
            FailureStage.CORPUS_LOADING,
            "manifest operation template does not match the history",
        )
    sketch_families = set()
    for node in history.structure.nodes:
        if not isinstance(node, SketchNode):
            continue
        primitive_types = tuple(item.primitive_type for item in node.primitives)
        if primitive_types == (PrimitiveType.CIRCLE,):
            sketch_families.add("circle")
        elif primitive_types == (PrimitiveType.LINE,) * 4:
            sketch_families.add("rectangle_lines")
        elif (
            len(primitive_types) == 4
            and primitive_types.count(PrimitiveType.LINE) == 2
            and primitive_types.count(PrimitiveType.ARC) == 2
        ):
            sketch_families.add("capsule_line_arc")
        else:
            sketch_families.add("unsupported")
    if sketch_families != {sample.primitive_family}:
        raise ExecutionFailure(
            FailureCategory.SCHEMA_OR_PARSE_FAILURE,
            FailureStage.CORPUS_LOADING,
            "manifest primitive family does not match the history",
        )


def _numeric_values(geometry: object) -> tuple[object, ...]:
    return tuple(
        value
        for value in vars(geometry).values()
        if hasattr(value, "encoding") and hasattr(value, "values")
    )


def _construct_feature(adapter, index, node, face, frame, operation_index):
    if isinstance(node, ExtrudeNode):
        geometry = index.node_geometry.get(node.node_id)
        if not isinstance(geometry, ExtrudeGeometry):
            raise ExecutionFailure(
                FailureCategory.EXTRUDE_FAILURE,
                FailureStage.REFERENCE_RESOLUTION,
                "extrude operation has no extrusion geometry",
                operation_index,
            )
        distance = decode_numeric(geometry.distance, 1)[0]
        sign = 1.0 if node.direction is Direction.POSITIVE else -1.0
        vector = tuple(sign * distance * item for item in frame.normal)
        return _kernel_call(
            adapter.extrude,
            face,
            vector,
            category=FailureCategory.EXTRUDE_FAILURE,
            stage=FailureStage.FEATURE_CONSTRUCTION,
            operation_index=operation_index,
        )
    if isinstance(node, RevolveNode):
        geometry = index.node_geometry.get(node.node_id)
        if not isinstance(geometry, RevolveGeometry):
            raise ExecutionFailure(
                FailureCategory.REVOLVE_FAILURE,
                FailureStage.REFERENCE_RESOLUTION,
                "revolve operation has no revolution geometry",
                operation_index,
            )
        point, direction = index.axis_world(node.node_id, frame, operation_index)
        axis = _kernel_call(
            adapter.make_axis,
            point,
            direction,
            category=FailureCategory.INVALID_REVOLVE_AXIS,
            stage=FailureStage.FEATURE_CONSTRUCTION,
            operation_index=operation_index,
        )
        angle = math.radians(decode_numeric(geometry.angle_degrees, 1)[0])
        if node.direction is Direction.NEGATIVE:
            angle = -angle
        return _kernel_call(
            adapter.revolve,
            face,
            axis,
            angle,
            category=FailureCategory.REVOLVE_FAILURE,
            stage=FailureStage.FEATURE_CONSTRUCTION,
            operation_index=operation_index,
        )
    raise ExecutionFailure(
        FailureCategory.UNSUPPORTED_FEATURE,
        FailureStage.FEATURE_CONSTRUCTION,
        f"unsupported operation type {type(node).__name__}",
        operation_index,
    )


def _inspect_and_require_solid(adapter, shape, operation_index: int) -> InspectedShape:
    inspected = _kernel_call(
        adapter.inspect_solid,
        shape,
        category=FailureCategory.INVALID_FINAL_SHAPE,
        stage=FailureStage.SOLID_VALIDATION,
        operation_index=operation_index,
    )
    metrics = inspected.metrics
    if metrics.is_null:
        raise ExecutionFailure(
            FailureCategory.EMPTY_RESULT,
            FailureStage.SOLID_VALIDATION,
            "kernel result is null",
            operation_index,
            metrics,
        )
    if metrics.solid_count != 1:
        raise ExecutionFailure(
            FailureCategory.INVALID_FINAL_SHAPE,
            FailureStage.SOLID_VALIDATION,
            f"kernel result contains {metrics.solid_count} solids; exactly one is required",
            operation_index,
            metrics,
        )
    if not metrics.is_valid:
        raise ExecutionFailure(
            FailureCategory.INVALID_FINAL_SHAPE,
            FailureStage.SOLID_VALIDATION,
            "kernel result is invalid",
            operation_index,
            metrics,
        )
    if metrics.volume is None or not math.isfinite(metrics.volume) or metrics.volume <= MIN_SOLID_VOLUME:
        raise ExecutionFailure(
            FailureCategory.INVALID_FINAL_SHAPE,
            FailureStage.SOLID_VALIDATION,
            f"solid volume must be finite and greater than {MIN_SOLID_VOLUME:g}",
            operation_index,
            metrics,
        )
    return inspected


def _require_boolean_effect(before, after, mode: BooleanMode, operation_index: int) -> None:
    assert before.volume is not None and after.volume is not None
    tolerance = max(
        BOOLEAN_VOLUME_ABSOLUTE_TOLERANCE,
        BOOLEAN_VOLUME_RELATIVE_TOLERANCE * max(abs(before.volume), abs(after.volume)),
    )
    change = after.volume - before.volume
    effective = change > tolerance if mode is BooleanMode.JOIN else -change > tolerance
    if not effective:
        raise ExecutionFailure(
            FailureCategory.BOOLEAN_NO_EFFECT,
            FailureStage.SEMANTIC_EFFECT,
            f"{mode.value} changed volume by {change:.12g}, not beyond tolerance {tolerance:.12g}",
            operation_index,
            after,
        )


def _kernel_call(method, *args, category, stage, operation_index):
    try:
        return method(*args)
    except KernelAdapterError as exc:
        raise ExecutionFailure(category, stage, _exception_detail(exc), operation_index) from exc
    except ExecutionFailure:
        raise
    except Exception as exc:
        raise ExecutionFailure(
            FailureCategory.UNEXPECTED_KERNEL_EXCEPTION,
            stage,
            _exception_detail(exc),
            operation_index,
        ) from exc


def _exception_detail(exc: BaseException) -> str:
    detail = re.sub(r"\s+", " ", str(exc)).strip()
    detail = re.sub(r"(?:[A-Za-z]:)?/(?:[^\s/]+/)+[^\s]+", "<path>", detail)
    detail = re.sub(r"0x[0-9A-Fa-f]+", "<address>", detail)
    if not detail:
        detail = type(exc).__name__
    return detail[:ERROR_DETAIL_LIMIT]


def _result(sample, status, operations, *, failure=None, metrics=None):
    return SampleExecutionResult(
        source_family_id=sample.source_family_id,
        sample_id=sample.sample_id,
        geometry_encoding=sample.geometry_encoding,
        operation_template=sample.operation_template,
        primitive_family=sample.primitive_family,
        kernel_status=status,
        failed_operation_index=failure.operation_index if failure else None,
        failure_category=failure.category.value if failure else None,
        failure_stage=failure.stage.value if failure else None,
        error_detail=_exception_detail(failure) if failure else None,
        final_shape_type=metrics.shape_type if metrics else None,
        final_shape_valid=bool(metrics and metrics.is_valid),
        final_metrics=metrics,
        operation_results=operations,
    )
