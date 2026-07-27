"""Immutable Phase B metrics over raw flat-baseline predictions."""

from __future__ import annotations

from dataclasses import dataclass
import math

from prototype.controlled_data.factors import (
    OperationTemplate,
    PrimitiveFamily,
)
from prototype.model_data.geometry import (
    GEOMETRY_CHANNEL_SCALES,
    GEOMETRY_WIDTH,
)
from prototype.model_data.records import PhysicalExample, ReconstructionTarget
from prototype.model_data.vocab import (
    CATEGORICAL_ATTRIBUTE_FIELDS,
    EDGE_TYPES,
    NODE_TYPES,
)

from .conversion import (
    FAILURE_CODE_ORDER,
    STAGE_ORDER,
    ConversionFailure,
    LayerValidity,
    validate_and_convert_raw_prediction,
)


@dataclass(frozen=True)
class CategoricalFieldMetric:
    field_name: str
    correct_positions: int
    target_positions: int
    accuracy: float | None


@dataclass(frozen=True)
class NodeCategoricalMetricRecord:
    node_records_available: bool
    node_type_correct_positions: int
    node_type_target_positions: int
    node_type_token_accuracy: float | None
    exact_node_type_sequence: bool
    categorical_fields: tuple[CategoricalFieldMetric, ...]
    exact_nine_attribute_match: bool
    exact_complete_ten_field_match: bool
    geometry_mask_matching_cells: int
    geometry_mask_target_cells: int
    geometry_mask_cell_accuracy: float | None
    exact_geometry_mask_match: bool


@dataclass(frozen=True)
class GeometryMetricRecord:
    target_applicable_geometry_channel_count: int
    finite_predicted_applicable_geometry_channel_count: int
    nonfinite_predicted_applicable_geometry_channel_count: int
    unavailable_predicted_applicable_geometry_channel_count: int
    finite_applicable_geometry_absolute_error_sum: float
    finite_applicable_geometry_squared_error_sum: float
    finite_applicable_geometry_mae: float | None
    finite_applicable_geometry_rmse: float | None
    complete_finite_geometry: bool


@dataclass(frozen=True)
class OperationMetricRecord:
    available: bool
    target_operation_count: int
    predicted_operation_count: int | None
    predicted_zero_operations: bool | None
    predicted_exceeds_target: bool | None
    predicted_exceeds_configured_limit: bool | None
    target_operation_type_sequence: tuple[str, ...]
    predicted_operation_type_sequence: tuple[str, ...] | None
    exact_operation_type_sequence: bool


@dataclass(frozen=True)
class PointerMetricRecord:
    available: bool
    target_operation_slots: int
    correct_target_operation_slots: int
    comparable_pointer_queries: int
    correct_comparable_queries: int
    missing_target_slot_predictions: int
    extra_predicted_pointer_records: int
    overall_pointer_accuracy: float | None
    conditional_pointer_accuracy: float | None
    exact_operation_sequence_pointer_match: bool


@dataclass(frozen=True)
class EdgeMetricRecord:
    available: bool
    predicted_present_pair_count: int | None
    target_present_pair_count: int
    true_positive_pairs: int
    false_positive_pairs: int
    false_negative_pairs: int
    precision: float | None
    recall: float | None
    f1: float | None
    shared_present_pair_count: int
    correctly_typed_shared_pair_count: int
    edge_type_accuracy: float | None
    exact_untyped_presence_set_match: bool
    exact_typed_edge_set_match: bool


@dataclass(frozen=True)
class PredictionMetrics:
    family_id: str
    raw_completion: bool
    raw_integrity: LayerValidity
    reconstruction_target: LayerValidity
    controlled_domain: LayerValidity
    primary_failure: ConversionFailure | None
    secondary_failures: tuple[ConversionFailure, ...]
    nodes: NodeCategoricalMetricRecord
    geometry: GeometryMetricRecord
    operations: OperationMetricRecord
    pointers: PointerMetricRecord
    edges: EdgeMetricRecord


@dataclass(frozen=True)
class AggregateCategoricalField:
    field_name: str
    correct_positions: int
    target_positions: int
    accuracy: float | None


@dataclass(frozen=True)
class AggregateNodeCategoricalMetrics:
    available_record_count: int
    unavailable_record_count: int
    node_type_correct_positions: int
    node_type_target_positions: int
    node_type_token_accuracy: float | None
    exact_node_type_sequence_count: int
    exact_node_type_sequence_rate: float | None
    categorical_fields: tuple[AggregateCategoricalField, ...]
    exact_nine_attribute_match_count: int
    exact_nine_attribute_match_rate: float | None
    exact_complete_ten_field_match_count: int
    exact_complete_ten_field_match_rate: float | None
    geometry_mask_matching_cells: int
    geometry_mask_target_cells: int
    geometry_mask_cell_accuracy: float | None
    exact_geometry_mask_match_count: int
    exact_geometry_mask_match_rate: float | None


@dataclass(frozen=True)
class AggregateGeometryMetrics:
    target_applicable_geometry_channel_count: int
    finite_predicted_applicable_geometry_channel_count: int
    nonfinite_predicted_applicable_geometry_channel_count: int
    unavailable_predicted_applicable_geometry_channel_count: int
    finite_applicable_geometry_absolute_error_sum: float | None
    finite_applicable_geometry_squared_error_sum: float | None
    absolute_error_aggregation_overflow: bool
    squared_error_aggregation_overflow: bool
    finite_applicable_geometry_mae: float | None
    finite_applicable_geometry_rmse: float | None
    complete_finite_example_count: int
    complete_finite_example_rate: float | None


@dataclass(frozen=True)
class AggregateOperationMetrics:
    available_record_count: int
    unavailable_record_count: int
    target_operation_count_sum: int
    predicted_operation_count_sum: int
    predicted_operation_count_defined_record_count: int
    mean_predicted_operation_count: float | None
    predicted_zero_operation_count: int
    predicted_zero_operation_rate: float | None
    predicted_target_excess_count: int
    predicted_target_excess_rate: float | None
    predicted_configured_limit_excess_count: int
    predicted_configured_limit_excess_rate: float | None
    exact_operation_type_sequence_count: int
    exact_operation_type_sequence_rate: float | None


@dataclass(frozen=True)
class AggregatePointerMetrics:
    available_record_count: int
    unavailable_record_count: int
    target_operation_slots: int
    correct_target_operation_slots: int
    comparable_pointer_queries: int
    correct_comparable_queries: int
    missing_target_slot_predictions: int
    extra_predicted_pointer_records: int
    overall_pointer_accuracy: float | None
    conditional_pointer_accuracy: float | None
    exact_operation_sequence_pointer_match_count: int
    exact_operation_sequence_pointer_match_rate: float | None


@dataclass(frozen=True)
class AggregateEdgeMetrics:
    available_record_count: int
    unavailable_record_count: int
    micro_true_positive_pairs: int
    micro_false_positive_pairs: int
    micro_false_negative_pairs: int
    micro_precision: float | None
    micro_recall: float | None
    micro_f1: float | None
    micro_correctly_typed_shared_pairs: int
    micro_shared_present_pairs: int
    micro_edge_type_accuracy: float | None
    macro_precision: float | None
    macro_precision_defined_record_count: int
    macro_recall: float | None
    macro_recall_defined_record_count: int
    macro_f1: float | None
    macro_f1_defined_record_count: int
    macro_edge_type_accuracy: float | None
    macro_edge_type_accuracy_defined_record_count: int
    exact_untyped_presence_set_match_count: int
    exact_untyped_presence_set_match_rate: float | None
    exact_typed_edge_set_match_count: int
    exact_typed_edge_set_match_rate: float | None


@dataclass(frozen=True)
class FailureAggregate:
    code: str
    count: int
    rate: float | None


@dataclass(frozen=True)
class ValidityFailureMetrics:
    attempted_example_count: int
    raw_completion_count: int
    raw_completion_rate: float | None
    raw_integrity_valid_count: int
    raw_integrity_valid_rate: float | None
    reconstruction_target_valid_count: int
    reconstruction_target_valid_rate: float | None
    controlled_domain_valid_count: int
    controlled_domain_valid_rate: float | None
    primary_failure_counts: tuple[FailureAggregate, ...]
    any_failure_counts: tuple[FailureAggregate, ...]


@dataclass(frozen=True)
class AggregateMetrics:
    attempted_example_count: int
    nodes: AggregateNodeCategoricalMetrics
    geometry: AggregateGeometryMetrics
    operations: AggregateOperationMetrics
    pointers: AggregatePointerMetrics
    edges: AggregateEdgeMetrics
    validity: ValidityFailureMetrics


@dataclass(frozen=True)
class AggregateEvaluation:
    overall: AggregateMetrics
    by_operation_template: tuple[tuple[str, AggregateMetrics], ...]
    by_primitive_family: tuple[tuple[str, AggregateMetrics], ...]
    by_target_operation_type_sequence: tuple[
        tuple[str, AggregateMetrics], ...
    ]


def evaluate_prediction(
    raw_prediction,
    target,
    *,
    family_id,
    max_operations,
) -> PredictionMetrics:
    """Evaluate one raw prediction without using stratum metadata."""

    _validate_family_id(family_id)
    if not isinstance(target, ReconstructionTarget):
        raise TypeError("target must be a ReconstructionTarget")
    _validate_max_operations(max_operations)
    conversion_result = validate_and_convert_raw_prediction(
        raw_prediction, max_operations=max_operations
    )
    return PredictionMetrics(
        family_id=family_id,
        raw_completion=raw_prediction is not None,
        raw_integrity=conversion_result.raw_integrity,
        reconstruction_target=conversion_result.reconstruction_target,
        controlled_domain=conversion_result.controlled_domain,
        primary_failure=conversion_result.primary_failure,
        secondary_failures=conversion_result.secondary_failures,
        nodes=_node_metrics(raw_prediction, target),
        geometry=_geometry_metrics(raw_prediction, target),
        operations=_operation_metrics(
            raw_prediction, target, max_operations
        ),
        pointers=_pointer_metrics(raw_prediction, target),
        edges=_edge_metrics(raw_prediction, target),
    )


def aggregate_prediction_metrics(
    records,
    *,
    metadata_by_family,
) -> AggregateEvaluation:
    """Aggregate sufficient statistics and apply authoritative strata."""

    records = tuple(records)
    for record in records:
        if not isinstance(record, PredictionMetrics):
            raise TypeError("records must contain PredictionMetrics")
    family_ids = tuple(record.family_id for record in records)
    if len(family_ids) != len(set(family_ids)):
        raise ValueError("duplicate metric family IDs")
    try:
        metadata_items = tuple(metadata_by_family.items())
    except AttributeError as exc:
        raise TypeError("metadata_by_family must be a mapping") from exc
    metadata_ids = tuple(key for key, _ in metadata_items)
    if len(metadata_ids) != len(set(metadata_ids)):
        raise ValueError("duplicate metadata family IDs")
    for family_id in (*family_ids, *metadata_ids):
        _validate_family_id(family_id)
    record_set = set(family_ids)
    metadata_set = set(metadata_ids)
    if record_set - metadata_set:
        raise ValueError("missing metadata for metric families")
    if metadata_set - record_set:
        raise ValueError("metadata contains unknown or missing metric families")
    normalized = {}
    for family_id, value in metadata_items:
        if not isinstance(value, PhysicalExample):
            raise TypeError(
                "metadata values must be authoritative PhysicalExample records"
            )
        if value.physical_family_id != family_id:
            raise ValueError("inconsistent metadata family ID")
        metadata = value.metadata
        try:
            operation_template = OperationTemplate(
                metadata.operation_template
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("unsupported operation template") from exc
        try:
            primitive_family = PrimitiveFamily(metadata.primitive_family)
        except (TypeError, ValueError) as exc:
            raise ValueError("unsupported primitive family") from exc
        operation_types = operation_template.operations
        if metadata.history_depth != len(operation_types):
            raise ValueError(
                "metadata history depth disagrees with operation template"
            )
        nodes_by_id = {
            node.node_id: node for node in value.nodes
            if hasattr(node, "node_id")
        }
        try:
            physical_operation_types = tuple(
                nodes_by_id[node_id].operation_type
                for node_id in value.operation_sequence
            )
        except (KeyError, TypeError) as exc:
            raise ValueError(
                "unsupported operation type sequence in PhysicalExample"
            ) from exc
        if physical_operation_types != operation_types:
            raise ValueError(
                "unsupported operation type sequence in PhysicalExample"
            )
        normalized[family_id] = (
            metadata,
            operation_template,
            primitive_family,
        )
    ordered_records = tuple(sorted(records, key=lambda item: item.family_id))
    return AggregateEvaluation(
        overall=_aggregate(ordered_records),
        by_operation_template=_strata(
            ordered_records,
            normalized,
            lambda item: item[1].value,
        ),
        by_primitive_family=_strata(
            ordered_records,
            normalized,
            lambda item: item[2].value,
        ),
        by_target_operation_type_sequence=_strata(
            ordered_records,
            normalized,
            lambda item: ",".join(item[1].operations),
        ),
    )


def _node_metrics(raw, target):
    target_count = len(target.node_type_ids)
    nodes = _tuple_field(raw, "raw_nodes")
    available = _usable_node_records(nodes, target_count)
    node_correct = 0
    field_correct = [0] * len(CATEGORICAL_ATTRIBUTE_FIELDS)
    mask_correct = 0
    for index in range(target_count):
        node = _aligned_node(nodes, index)
        node_type = getattr(node, "node_type_id", None)
        if _integer(node_type) and node_type == target.node_type_ids[index]:
            node_correct += 1
        attributes = getattr(node, "categorical_ids", None)
        if isinstance(attributes, tuple):
            for field in range(len(CATEGORICAL_ATTRIBUTE_FIELDS)):
                if (
                    field < len(attributes)
                    and _integer(attributes[field])
                    and attributes[field]
                    == target.categorical_attributes[index][field]
                ):
                    field_correct[field] += 1
        predicted_mask = getattr(node, "derived_geometry_mask", None)
        if isinstance(predicted_mask, tuple):
            for channel in range(GEOMETRY_WIDTH):
                if (
                    channel < len(predicted_mask)
                    and isinstance(predicted_mask[channel], bool)
                    and predicted_mask[channel]
                    is target.geometry_mask[index][channel]
                ):
                    mask_correct += 1
    node_exact = available and node_correct == target_count
    attribute_exact = (
        available
        and all(value == target_count for value in field_correct)
    )
    mask_denominator = target_count * GEOMETRY_WIDTH
    mask_exact = (
        available
        and mask_correct == mask_denominator
    )
    return NodeCategoricalMetricRecord(
        available,
        node_correct,
        target_count,
        _ratio(node_correct, target_count),
        node_exact,
        tuple(
            CategoricalFieldMetric(
                name,
                field_correct[index],
                target_count,
                _ratio(field_correct[index], target_count),
            )
            for index, name in enumerate(CATEGORICAL_ATTRIBUTE_FIELDS)
        ),
        attribute_exact,
        node_exact and attribute_exact,
        mask_correct,
        mask_denominator,
        _ratio(mask_correct, mask_denominator),
        mask_exact,
    )


def _geometry_metrics(raw, target):
    nodes = _tuple_field(raw, "raw_nodes")
    target_applicable = 0
    finite = 0
    nonfinite = 0
    unavailable = 0
    absolute_sum = 0.0
    squared_sum = 0.0
    for node_index, (target_values, target_mask) in enumerate(
        zip(target.geometry, target.geometry_mask)
    ):
        predicted_values = None
        node = _aligned_node(nodes, node_index)
        if node is not None:
            candidate = getattr(node, "normalized_geometry", None)
            if isinstance(candidate, tuple):
                predicted_values = candidate
        for channel, applicable in enumerate(target_mask):
            if not applicable:
                continue
            target_applicable += 1
            if (
                predicted_values is None
                or channel >= len(predicted_values)
                or type(predicted_values[channel]) is not float
            ):
                unavailable += 1
                continue
            predicted = predicted_values[channel]
            if not math.isfinite(predicted):
                nonfinite += 1
                continue
            errors = _checked_geometry_errors(
                predicted,
                target_values[channel],
                GEOMETRY_CHANNEL_SCALES[channel],
            )
            if errors is None:
                unavailable += 1
                continue
            absolute_error, squared_error = errors
            next_absolute = absolute_sum + absolute_error
            next_squared = squared_sum + squared_error
            if (
                not math.isfinite(next_absolute)
                or not math.isfinite(next_squared)
            ):
                unavailable += 1
                continue
            finite += 1
            absolute_sum = next_absolute
            squared_sum = next_squared
    absolute_sum = _zero(absolute_sum)
    squared_sum = _zero(squared_sum)
    return GeometryMetricRecord(
        target_applicable,
        finite,
        nonfinite,
        unavailable,
        absolute_sum,
        squared_sum,
        _ratio(absolute_sum, finite),
        _root_mean(squared_sum, finite),
        finite == target_applicable,
    )


def _operation_metrics(raw, target, max_operations):
    target_types = tuple(
        NODE_TYPES.tokens[target.node_type_ids[index]]
        for index in target.operation_sequence
    )
    nodes = _tuple_field(raw, "raw_nodes")
    available = _usable_operation_nodes(
        nodes, len(target.node_type_ids)
    )
    if not available:
        return OperationMetricRecord(
            False,
            len(target_types),
            None,
            None,
            None,
            None,
            target_types,
            None,
            False,
        )
    predicted_types = tuple(
        NODE_TYPES.tokens[node.node_type_id]
        for node in nodes
        if NODE_TYPES.tokens[node.node_type_id] in ("extrude", "revolve")
    )
    predicted_count = len(predicted_types)
    return OperationMetricRecord(
        True,
        len(target_types),
        predicted_count,
        predicted_count == 0,
        predicted_count > len(target_types),
        predicted_count > max_operations,
        target_types,
        predicted_types,
        predicted_types == target_types,
    )


def _pointer_metrics(raw, target):
    target_sequence = target.operation_sequence
    pointers = _tuple_field(raw, "raw_operation_pointers")
    available = pointers is not None and all(
        _usable_pointer(pointer, index)
        for index, pointer in enumerate(pointers)
    )
    if not available:
        target_count = len(target_sequence)
        return PointerMetricRecord(
            False,
            target_count,
            0,
            0,
            0,
            target_count,
            0,
            _ratio(0, target_count),
            None,
            False,
        )
    comparable = min(len(pointers), len(target_sequence))
    correct = sum(
        pointers[index].selected_node_index == target_sequence[index]
        for index in range(comparable)
    )
    target_count = len(target_sequence)
    missing = max(target_count - len(pointers), 0)
    extra = max(len(pointers) - target_count, 0)
    return PointerMetricRecord(
        True,
        target_count,
        correct,
        comparable,
        correct,
        missing,
        extra,
        _ratio(correct, target_count),
        _ratio(correct, comparable),
        correct == target_count and not missing and not extra,
    )


def _edge_metrics(raw, target):
    target_typed = {
        (source, edge_type, destination)
        for source, destination, edge_type in zip(
            target.edge_index[0],
            target.edge_index[1],
            target.edge_type_ids,
        )
    }
    target_pairs = {
        (source, destination)
        for source, _, destination in target_typed
    }
    edges = _usable_edge_matrix(raw, len(target.node_type_ids))
    if edges is None:
        return EdgeMetricRecord(
            False,
            None,
            len(target_pairs),
            0,
            0,
            len(target_pairs),
            None,
            None,
            None,
            0,
            0,
            None,
            False,
            False,
        )
    predicted_typed = {
        (edge.source_index, edge.edge_type_id, edge.target_index)
        for edge in edges
        if edge.present
    }
    predicted_pairs = {
        (source, destination)
        for source, _, destination in predicted_typed
    }
    shared = predicted_pairs & target_pairs
    true_positive = len(shared)
    false_positive = len(predicted_pairs - target_pairs)
    false_negative = len(target_pairs - predicted_pairs)
    precision, recall, f1 = _edge_ratios(
        true_positive, false_positive, false_negative
    )
    predicted_types = {
        (source, destination): edge_type
        for source, edge_type, destination in predicted_typed
    }
    target_types = {
        (source, destination): edge_type
        for source, edge_type, destination in target_typed
    }
    typed_correct = sum(
        predicted_types[pair] == target_types[pair] for pair in shared
    )
    return EdgeMetricRecord(
        True,
        len(predicted_pairs),
        len(target_pairs),
        true_positive,
        false_positive,
        false_negative,
        precision,
        recall,
        f1,
        len(shared),
        typed_correct,
        _ratio(typed_correct, len(shared)),
        predicted_pairs == target_pairs,
        predicted_typed == target_typed,
    )


def _aggregate(records):
    attempted = len(records)
    return AggregateMetrics(
        attempted,
        _aggregate_nodes(records, attempted),
        _aggregate_geometry(records, attempted),
        _aggregate_operations(records, attempted),
        _aggregate_pointers(records, attempted),
        _aggregate_edges(records, attempted),
        _aggregate_validity(records, attempted),
    )


def _aggregate_nodes(records, attempted):
    correct = sum(item.nodes.node_type_correct_positions for item in records)
    target = sum(item.nodes.node_type_target_positions for item in records)
    fields = tuple(
        AggregateCategoricalField(
            name,
            sum(item.nodes.categorical_fields[index].correct_positions
                for item in records),
            sum(item.nodes.categorical_fields[index].target_positions
                for item in records),
            _ratio(
                sum(item.nodes.categorical_fields[index].correct_positions
                    for item in records),
                sum(item.nodes.categorical_fields[index].target_positions
                    for item in records),
            ),
        )
        for index, name in enumerate(CATEGORICAL_ATTRIBUTE_FIELDS)
    )
    mask_correct = sum(
        item.nodes.geometry_mask_matching_cells for item in records
    )
    mask_target = sum(
        item.nodes.geometry_mask_target_cells for item in records
    )
    available = sum(item.nodes.node_records_available for item in records)
    exact_node = sum(item.nodes.exact_node_type_sequence for item in records)
    exact_attributes = sum(
        item.nodes.exact_nine_attribute_match for item in records
    )
    exact_complete = sum(
        item.nodes.exact_complete_ten_field_match for item in records
    )
    exact_mask = sum(
        item.nodes.exact_geometry_mask_match for item in records
    )
    return AggregateNodeCategoricalMetrics(
        available,
        attempted - available,
        correct,
        target,
        _ratio(correct, target),
        exact_node,
        _ratio(exact_node, attempted),
        fields,
        exact_attributes,
        _ratio(exact_attributes, attempted),
        exact_complete,
        _ratio(exact_complete, attempted),
        mask_correct,
        mask_target,
        _ratio(mask_correct, mask_target),
        exact_mask,
        _ratio(exact_mask, attempted),
    )


def _aggregate_geometry(records, attempted):
    target = sum(
        item.geometry.target_applicable_geometry_channel_count
        for item in records
    )
    finite = sum(
        item.geometry.finite_predicted_applicable_geometry_channel_count
        for item in records
    )
    nonfinite = sum(
        item.geometry.nonfinite_predicted_applicable_geometry_channel_count
        for item in records
    )
    unavailable = sum(
        item.geometry.unavailable_predicted_applicable_geometry_channel_count
        for item in records
    )
    absolute = _checked_sum(tuple(
        item.geometry.finite_applicable_geometry_absolute_error_sum
        for item in records
    ))
    squared = _checked_sum(tuple(
        item.geometry.finite_applicable_geometry_squared_error_sum
        for item in records
    ))
    complete = sum(
        item.geometry.complete_finite_geometry for item in records
    )
    return AggregateGeometryMetrics(
        target,
        finite,
        nonfinite,
        unavailable,
        absolute,
        squared,
        absolute is None,
        squared is None,
        None if absolute is None else _ratio(absolute, finite),
        None if squared is None else _root_mean(squared, finite),
        complete,
        _ratio(complete, attempted),
    )


def _aggregate_operations(records, attempted):
    available_records = tuple(
        item.operations for item in records if item.operations.available
    )
    available = len(available_records)
    predicted_sum = sum(
        item.predicted_operation_count for item in available_records
    )
    zero = sum(item.predicted_zero_operations for item in available_records)
    target_excess = sum(
        item.predicted_exceeds_target for item in available_records
    )
    limit_excess = sum(
        item.predicted_exceeds_configured_limit
        for item in available_records
    )
    exact = sum(
        item.operations.exact_operation_type_sequence for item in records
    )
    return AggregateOperationMetrics(
        available,
        attempted - available,
        sum(item.operations.target_operation_count for item in records),
        predicted_sum,
        available,
        _ratio(predicted_sum, available),
        zero,
        _ratio(zero, attempted),
        target_excess,
        _ratio(target_excess, attempted),
        limit_excess,
        _ratio(limit_excess, attempted),
        exact,
        _ratio(exact, attempted),
    )


def _aggregate_pointers(records, attempted):
    target = sum(item.pointers.target_operation_slots for item in records)
    correct = sum(
        item.pointers.correct_target_operation_slots for item in records
    )
    comparable = sum(
        item.pointers.comparable_pointer_queries for item in records
    )
    comparable_correct = sum(
        item.pointers.correct_comparable_queries for item in records
    )
    available = sum(item.pointers.available for item in records)
    exact = sum(
        item.pointers.exact_operation_sequence_pointer_match
        for item in records
    )
    return AggregatePointerMetrics(
        available,
        attempted - available,
        target,
        correct,
        comparable,
        comparable_correct,
        sum(item.pointers.missing_target_slot_predictions for item in records),
        sum(item.pointers.extra_predicted_pointer_records for item in records),
        _ratio(correct, target),
        _ratio(comparable_correct, comparable),
        exact,
        _ratio(exact, attempted),
    )


def _aggregate_edges(records, attempted):
    available_records = tuple(
        item.edges for item in records if item.edges.available
    )
    available = len(available_records)
    true_positive = sum(item.true_positive_pairs for item in available_records)
    false_positive = sum(item.false_positive_pairs for item in available_records)
    false_negative = sum(item.false_negative_pairs for item in available_records)
    if available:
        precision, recall, f1 = _edge_ratios(
            true_positive, false_positive, false_negative
        )
    else:
        precision, recall, f1 = None, None, None
    shared = sum(
        item.shared_present_pair_count for item in available_records
    )
    typed = sum(
        item.correctly_typed_shared_pair_count for item in available_records
    )
    macro_precision = tuple(
        item.precision for item in available_records
        if item.precision is not None
    )
    macro_recall = tuple(
        item.recall for item in available_records if item.recall is not None
    )
    macro_f1 = tuple(
        item.f1 for item in available_records if item.f1 is not None
    )
    macro_types = tuple(
        item.edge_type_accuracy for item in available_records
        if item.edge_type_accuracy is not None
    )
    exact_untyped = sum(
        item.edges.exact_untyped_presence_set_match for item in records
    )
    exact_typed = sum(
        item.edges.exact_typed_edge_set_match for item in records
    )
    return AggregateEdgeMetrics(
        available,
        attempted - available,
        true_positive,
        false_positive,
        false_negative,
        precision,
        recall,
        f1,
        typed,
        shared,
        _ratio(typed, shared),
        _mean(macro_precision),
        len(macro_precision),
        _mean(macro_recall),
        len(macro_recall),
        _mean(macro_f1),
        len(macro_f1),
        _mean(macro_types),
        len(macro_types),
        exact_untyped,
        _ratio(exact_untyped, attempted),
        exact_typed,
        _ratio(exact_typed, attempted),
    )


def _aggregate_validity(records, attempted):
    completion = sum(item.raw_completion for item in records)
    raw_valid = sum(item.raw_integrity.valid for item in records)
    target_valid = sum(
        item.reconstruction_target.valid for item in records
    )
    domain_valid = sum(item.controlled_domain.valid for item in records)
    ordered_codes = tuple(
        code for stage in STAGE_ORDER for code in FAILURE_CODE_ORDER[stage]
    )
    primary = {
        code: sum(
            item.primary_failure is not None
            and item.primary_failure.code == code
            for item in records
        )
        for code in ordered_codes
    }
    any_counts = {}
    for code in ordered_codes:
        any_counts[code] = sum(
            code in {
                failure.code
                for failure in (
                    ((item.primary_failure,) if item.primary_failure else ())
                    + item.secondary_failures
                )
            }
            for item in records
        )
    return ValidityFailureMetrics(
        attempted,
        completion,
        _ratio(completion, attempted),
        raw_valid,
        _ratio(raw_valid, attempted),
        target_valid,
        _ratio(target_valid, attempted),
        domain_valid,
        _ratio(domain_valid, attempted),
        tuple(
            FailureAggregate(code, primary[code], _ratio(primary[code], attempted))
            for code in ordered_codes
        ),
        tuple(
            FailureAggregate(
                code, any_counts[code], _ratio(any_counts[code], attempted)
            )
            for code in ordered_codes
        ),
    )


def _strata(records, metadata, key_function):
    grouped = {}
    for record in records:
        key = key_function(metadata[record.family_id])
        if not isinstance(key, str) or not key:
            raise ValueError("stratum metadata must be a nonempty string")
        grouped.setdefault(key, []).append(record)
    return tuple(
        (key, _aggregate(tuple(grouped[key]))) for key in sorted(grouped)
    )


def _usable_edge_matrix(raw, expected_node_count):
    edges = _tuple_field(raw, "raw_edges")
    if (
        edges is None
        or len(edges) != expected_node_count ** 2
    ):
        return None
    for index, edge in enumerate(edges):
        source, target = divmod(index, expected_node_count)
        if (
            not hasattr(edge, "source_index")
            or not hasattr(edge, "target_index")
            or not hasattr(edge, "presence_logit")
            or not hasattr(edge, "present")
            or not hasattr(edge, "edge_type_id")
            or not _integer(edge.source_index)
            or not _integer(edge.target_index)
            or edge.source_index != source
            or edge.target_index != target
            or type(edge.presence_logit) is not float
            or not isinstance(edge.present, bool)
            or (
                math.isfinite(edge.presence_logit)
                and edge.present != (edge.presence_logit >= 0.0)
            )
            or (
                edge.present
                and not _valid_vocabulary_id(edge.edge_type_id, EDGE_TYPES)
            )
        ):
            return None
    return edges


def _usable_node_records(nodes, expected_node_count):
    if nodes is None or len(nodes) != expected_node_count:
        return False
    for index, node in enumerate(nodes):
        if (
            not hasattr(node, "position")
            or not hasattr(node, "node_type_id")
            or not hasattr(node, "categorical_ids")
            or not hasattr(node, "normalized_geometry")
            or not hasattr(node, "derived_geometry_mask")
            or not _integer(node.position)
            or node.position != index
            or not _integer(node.node_type_id)
            or not isinstance(node.categorical_ids, tuple)
            or len(node.categorical_ids)
            != len(CATEGORICAL_ATTRIBUTE_FIELDS)
            or not all(_integer(value) for value in node.categorical_ids)
            or not isinstance(node.derived_geometry_mask, tuple)
            or len(node.derived_geometry_mask) != GEOMETRY_WIDTH
            or not all(
                type(value) is bool
                for value in node.derived_geometry_mask
            )
        ):
            return False
    return True


def _usable_operation_nodes(nodes, expected_node_count):
    if nodes is None or len(nodes) != expected_node_count:
        return False
    return all(
        node is not None
        and _valid_vocabulary_id(
            getattr(node, "node_type_id", None), NODE_TYPES
        )
        for node in (
            _aligned_node(nodes, index)
            for index in range(expected_node_count)
        )
    )


def _usable_pointer(pointer, query_index):
    return (
        hasattr(pointer, "query_index")
        and hasattr(pointer, "selected_node_index")
        and hasattr(pointer, "selected_logit")
        and _integer(pointer.query_index)
        and pointer.query_index == query_index
        and _integer(pointer.selected_node_index)
        and type(pointer.selected_logit) is float
    )


def _tuple_field(raw, name):
    if raw is None or not hasattr(raw, name):
        return None
    value = getattr(raw, name)
    return value if isinstance(value, tuple) else None


def _aligned_node(nodes, index):
    if nodes is None or index >= len(nodes):
        return None
    node = nodes[index]
    position = getattr(node, "position", None)
    if not _integer(position) or position != index:
        return None
    return node


def _valid_vocabulary_id(value, vocabulary):
    return _integer(value) and 0 <= value < len(vocabulary.tokens)


def _checked_geometry_errors(predicted, target, scale):
    """Return errors only when every required arithmetic result is finite.

    Extreme finite raw values are metric-unavailable when scaling,
    subtraction, absolute value, or squaring overflows. They are not counted
    as raw nonfinite predictions, and no clipping or partial accumulation is
    performed.
    """

    try:
        physical_prediction = predicted * scale
        physical_target = target * scale
        error = physical_prediction - physical_target
        absolute_error = abs(error)
        squared_error = error * error
    except OverflowError:
        return None
    if not all(math.isfinite(value) for value in (
        physical_prediction,
        physical_target,
        error,
        absolute_error,
        squared_error,
    )):
        return None
    return absolute_error, squared_error


def _checked_sum(values):
    total = 0.0
    for value in values:
        candidate = total + value
        if not math.isfinite(candidate):
            return None
        total = candidate
    return _zero(total)


def _integer(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _edge_ratios(true_positive, false_positive, false_negative):
    predicted = true_positive + false_positive
    target = true_positive + false_negative
    if not predicted and not target:
        return 1.0, 1.0, 1.0
    if not predicted:
        return 0.0, 0.0, 0.0
    if not target:
        return 0.0, 1.0, 0.0
    precision = true_positive / predicted
    recall = true_positive / target
    f1 = (
        0.0 if precision + recall == 0.0
        else 2.0 * precision * recall / (precision + recall)
    )
    return _zero(precision), _zero(recall), _zero(f1)


def _validate_family_id(family_id):
    if not isinstance(family_id, str) or not family_id:
        raise ValueError("family_id must be a nonempty string")


def _validate_max_operations(max_operations):
    if (
        isinstance(max_operations, bool)
        or not isinstance(max_operations, int)
        or max_operations <= 0
    ):
        raise ValueError("max_operations must be a positive integer")


def _ratio(numerator, denominator):
    if denominator == 0:
        return None
    return _zero(float(numerator) / float(denominator))


def _root_mean(squared_sum, count):
    if count == 0:
        return None
    return _zero(math.sqrt(float(squared_sum) / float(count)))


def _mean(values):
    if not values:
        return None
    return _zero(sum(values) / len(values))


def _zero(value):
    return 0.0 if value == 0.0 else float(value)
