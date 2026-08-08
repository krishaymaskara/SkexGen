"""Frozen C6 family-macro metrics and executable-prefix evaluation."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, replace
import copy
import math
import platform
import resource
import time

from prototype.model_data.geometry import (
    GEOMETRY_CHANNEL_SCALES,
    GEOMETRY_CHANNELS,
)
from prototype.model_data.vocab import EDGE_TYPES, NODE_TYPES

from .errors import GraphEncoderError


METRICS_SCHEMA_VERSION = "GE1-C6-METRICS-v2"
DONOR_TEMPLATE_AGREEMENT_VERSION = "GE1-C6-DONOR-TEMPLATE-AGREEMENT-v1"
PRIMARY_REPORTING_CONTRACT_VERSION = "GE1-PRIMARY-REPORT-v1"

# Every primary result must publish the memory-intervention evidence beside it.
# The shared decoder receives four output-side serialized-position signals, and
# the frozen Graph V1 position-only prior reached 59/68 exact graphs, so a
# primary number is uninterpretable without the accompanying evidence that the
# decoder used encoder memory at all. These five values are therefore a
# reporting requirement, not an optional diagnostic.
PRIMARY_REPORT_REQUIRED_INTERVENTION_KEYS = (
    "P_true",
    "P_shuffle",
    "P_mean",
    "R_shuffle",
    "R_mean",
)
PREFIX_SCORER_VERSION = "GE1-C6-EXECUTABLE-PREFIX-v1"
AGGREGATION_VERSION = "GE1-C6-PHYSICAL-FAMILY-MACRO-v1"
FIRST_FAILURE_STAGES = (
    "encoder/input preparation",
    "node rollout",
    "geometry prediction/reconstruction",
    "edge prediction",
    "canonicalization",
    "prefix construction",
    "strict conversion",
    "analytic validity",
    "no failure",
)
GEOMETRY_CHANNEL_FAMILIES = (
    ("reference_plane", tuple(range(0, 9))),
    ("profile_primitives", tuple(range(9, 33))),
    ("axis", tuple(range(33, 37))),
    ("operation_parameter", tuple(range(37, 39))),
)
ATTACHMENT_EDGE_NAMES = (
    "defined_in", "placed_on", "uses_axis", "uses_profile"
)
EXPECTED_FLAT_ENCODER_PARAMETERS = 22800
EXPECTED_GRAPH_ENCODER_PARAMETERS = 23468
EXPECTED_ENCODER_DIFFERENCE = 668
EXPECTED_RELATIVE_DIFFERENCE_PERCENT = 2.9298245614035086
MAX_CONTROLLED_TEMPLATE_DIAMETER = 3
RELATIONAL_MESSAGE_PASSING_LAYERS = 3


@dataclass(frozen=True)
class UndefinedMetric:
    value: object
    reason: str

    def to_dict(self):
        return {"value": self.value, "reason": self.reason}


@dataclass(frozen=True)
class PrefixAttempt:
    k: int
    strict_conversion: bool
    analytic_validity: bool
    failure_stage: str
    failure_code: str

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class PrefixScore:
    version: str
    target_operation_count: int
    canonicalization_status: str
    canonicalization_failure_code: object
    ordered_generated_operation_group_count: object
    evaluated_prefixes: tuple
    selected_k_max: int
    normalized_longest_executable_operation_prefix: float
    unnormalized_longest_executable_prefix: int
    first_failed_stage: str
    first_failure_code: str

    def to_dict(self):
        return {
            "version": self.version,
            "target_operation_count": self.target_operation_count,
            "canonicalization_status": self.canonicalization_status,
            "canonicalization_failure_code": self.canonicalization_failure_code,
            "ordered_generated_operation_group_count": (
                self.ordered_generated_operation_group_count
            ),
            "evaluated_prefixes": [item.to_dict() for item in self.evaluated_prefixes],
            "selected_k_max": self.selected_k_max,
            "normalized_longest_executable_operation_prefix": (
                self.normalized_longest_executable_operation_prefix
            ),
            "unnormalized_longest_executable_prefix": (
                self.unnormalized_longest_executable_prefix
            ),
            "first_failed_stage": self.first_failed_stage,
            "first_failure_code": self.first_failure_code,
        }


def undefined(reason):
    return UndefinedMetric(None, str(reason)).to_dict()


def prefix_score_from_outcomes(
    target_operation_count,
    *,
    canonicalization_status,
    canonicalization_failure_code=None,
    generated_operation_group_count=None,
    outcomes=(),
):
    """Pure, immutable scoring core used by fixtures and production graphs."""

    if (
        isinstance(target_operation_count, bool)
        or not isinstance(target_operation_count, int)
        or target_operation_count not in (1, 2)
    ):
        raise GraphEncoderError(
            "invalid_prefix_denominator",
            "controlled target operation count must be one or two",
        )
    if canonicalization_status != "success":
        attempt = PrefixAttempt(
            0, True, True, "canonicalization",
            str(canonicalization_failure_code or "canonicalization_failed")
        )
        return PrefixScore(
            PREFIX_SCORER_VERSION,
            target_operation_count,
            "failed",
            canonicalization_failure_code or "canonicalization_failed",
            None,
            (attempt,),
            0,
            0.0,
            0,
            "canonicalization",
            str(canonicalization_failure_code or "canonicalization_failed"),
        )
    attempts = [PrefixAttempt(0, True, True, "no failure", "valid_zero_prefix")]
    by_k = {int(item.k): item for item in outcomes}
    k_max = 0
    first_stage = "no failure"
    first_code = "valid"
    for k in range(1, target_operation_count + 1):
        if k > int(generated_operation_group_count):
            attempt = PrefixAttempt(
                k, False, False, "prefix construction",
                "missing_complete_generated_operation_group",
            )
        else:
            attempt = by_k.get(k)
            if attempt is None:
                raise GraphEncoderError(
                    "incomplete_prefix_record",
                    "every constructible positive k requires one outcome",
                )
        attempts.append(attempt)
        if attempt.strict_conversion and attempt.analytic_validity:
            k_max = k
        elif first_stage == "no failure":
            first_stage = attempt.failure_stage
            first_code = attempt.failure_code
    return PrefixScore(
        PREFIX_SCORER_VERSION,
        target_operation_count,
        "success",
        None,
        int(generated_operation_group_count),
        tuple(attempts),
        k_max,
        k_max / float(target_operation_count),
        k_max,
        first_stage,
        first_code,
    )


def score_prediction_prefix(prediction, target_operation_count):
    """C2-canonicalize a generated graph and strictly score its prefixes."""

    from prototype.graph_baseline.graph_contract import (
        model_data_edge_id_from_graph_class,
    )
    from prototype.graph_baseline.conversion import (
        validate_and_convert_graph_prediction,
    )
    from .canonicalization import GraphCanonicalizationInput, canonicalize_graph

    original = copy.deepcopy(prediction)
    node_prediction = prediction.node_prediction
    graph = prediction.graph
    narrow = GraphCanonicalizationInput(
        graph.node_type_ids,
        (
            tuple(item.source for item in graph.directed_typed_edges),
            tuple(item.destination for item in graph.directed_typed_edges),
        ),
        tuple(
            model_data_edge_id_from_graph_class(item.edge_type_id)
            for item in graph.directed_typed_edges
        ),
        tuple(node.categorical_ids for node in node_prediction.raw_nodes),
        tuple(node.normalized_geometry for node in node_prediction.raw_nodes),
        tuple(node.derived_geometry_mask for node in node_prediction.raw_nodes),
    )
    try:
        canonical = canonicalize_graph(narrow)
    except Exception as exc:
        code = getattr(exc, "code", type(exc).__name__)
        return prefix_score_from_outcomes(
            target_operation_count,
            canonicalization_status="failed",
            canonicalization_failure_code=code,
        )

    outcomes = []
    operation_count = len(canonical.operation_sequence)
    for k in range(1, min(operation_count, target_operation_count) + 1):
        stop = canonical.operation_sequence[k - 1] + 1
        try:
            prefix = _canonical_prefix_prediction(
                prediction, canonical.canonical_to_old[:stop]
            )
        except Exception as exc:
            outcomes.append(PrefixAttempt(
                k, False, False, "prefix construction",
                getattr(exc, "code", type(exc).__name__),
            ))
            continue
        try:
            converted = validate_and_convert_graph_prediction(prefix)
        except Exception as exc:
            outcomes.append(PrefixAttempt(
                k, False, False, "strict conversion",
                getattr(exc, "code", type(exc).__name__),
            ))
            continue
        strict = bool(converted.reconstruction_target.valid)
        analytic = bool(converted.controlled_domain.valid)
        failure = converted.primary_failure
        if not strict:
            stage = "strict conversion"
            code = failure.code if failure is not None else "strict_conversion_failed"
        elif not analytic:
            stage = "analytic validity"
            code = failure.code if failure is not None else "analytic_validity_failed"
        else:
            stage = "no failure"
            code = "valid"
        outcomes.append(PrefixAttempt(k, strict, analytic, stage, code))
    result = prefix_score_from_outcomes(
        target_operation_count,
        canonicalization_status="success",
        generated_operation_group_count=operation_count,
        outcomes=tuple(outcomes),
    )
    if prediction != original:
        raise AssertionError("prefix scoring mutated the autonomous prediction")
    return result


def _canonical_prefix_prediction(prediction, canonical_old_indices):
    """Retain only complete canonical groups; never repair or synthesize data."""

    from prototype.flat_baseline.autonomous import (
        RawDecodedEdge,
        RawDecodedNode,
        RawDecodedPointer,
    )
    from prototype.graph_baseline.conversion import graph_prediction_from_evidence
    from prototype.graph_baseline.graph_contract import (
        model_data_edge_id_from_graph_class,
    )
    from prototype.node_grammar import (
        V5_NODE_GRAMMAR,
        grammar_state_evidence,
        legal_next_node_ids,
    )

    indices = tuple(canonical_old_indices)
    count = len(indices)
    if not indices or len(indices) != len(set(indices)):
        raise GraphEncoderError("invalid_prefix_nodes", "prefix indices differ")
    old_to_new = {old: new for new, old in enumerate(indices)}
    node = prediction.node_prediction

    def reordered(name):
        values = getattr(node, name)
        return tuple(values[old] for old in indices)

    raw_nodes = tuple(
        replace(reordered("raw_nodes")[new], position=new)
        for new in range(count)
    )
    shadow_nodes = tuple(
        replace(reordered("same_history_raw_shadow_nodes")[new], position=new)
        for new in range(count)
    )
    node_ids = tuple(item.node_type_id for item in raw_nodes)
    shadow_ids = tuple(item.node_type_id for item in shadow_nodes)
    operations = tuple(
        index for index, value in enumerate(node_ids)
        if NODE_TYPES.tokens[value] in ("extrude", "revolve")
    )
    shadow_operations = tuple(
        index for index, value in enumerate(shadow_ids)
        if NODE_TYPES.tokens[value] in ("extrude", "revolve")
    )
    legal_masks = []
    evidence = []
    prefix_ids = ()
    for position in range(count):
        legal = legal_next_node_ids(prefix_ids, count, V5_NODE_GRAMMAR)
        legal_masks.append(tuple(
            candidate in legal for candidate in range(len(NODE_TYPES.tokens))
        ))
        evidence.append(grammar_state_evidence(prefix_ids, count, legal))
        prefix_ids = prefix_ids + (node_ids[position],)

    retained_edges = tuple(
        item for item in prediction.graph.directed_typed_edges
        if item.source in old_to_new and item.destination in old_to_new
    )
    edge_by_pair = {
        (old_to_new[item.source], old_to_new[item.destination]): item
        for item in retained_edges
    }
    raw_edges = []
    raw_matrix = []
    masked_matrix = []
    correction_matrix = []
    main_logits = [] if prediction.main_pair_logits is not None else None
    position_logits = [] if prediction.position_bias_logits is not None else None
    for source_new, source_old in enumerate(indices):
        raw_row = []
        masked_row = []
        correction_row = []
        main_row = []
        position_row = []
        for destination_new, destination_old in enumerate(indices):
            edge = edge_by_pair.get((source_new, destination_new))
            raw_class = prediction.raw_graph_edge_predictions[
                source_old
            ][destination_old]
            masked_class = 0 if edge is None else edge.edge_type_id
            raw_row.append(raw_class)
            masked_row.append(masked_class)
            correction_row.append(bool(raw_class != masked_class))
            raw_edges.append(RawDecodedEdge(
                source_new,
                destination_new,
                1.0 if edge is not None else -1.0,
                edge is not None,
                (
                    model_data_edge_id_from_graph_class(edge.edge_type_id)
                    if edge is not None else EDGE_TYPES.id(None)
                ),
            ))
            if main_logits is not None:
                main_row.append(
                    prediction.main_pair_logits[source_old][destination_old]
                )
                position_row.append(
                    prediction.position_bias_logits[source_old][destination_old]
                )
        raw_matrix.append(tuple(raw_row))
        masked_matrix.append(tuple(masked_row))
        correction_matrix.append(tuple(correction_row))
        if main_logits is not None:
            main_logits.append(tuple(main_row))
            position_logits.append(tuple(position_row))

    aligned_fields = (
        "profile_family_logits",
        "predicted_profile_family_ids",
        "raw_profile_parameters",
        "constrained_profile_parameters",
        "raw_categorical_argmax_ids",
        "node_conditioned_categorical_ids",
        "categorical_correction_mask",
        "raw_node_type_argmax_ids",
        "grammar_constrained_node_type_ids",
        "node_type_correction_mask",
        "raw_axis_geometry",
        "constrained_axis_geometry",
        "axis_geometry_correction_mask",
    )
    replacements = {name: reordered(name) for name in aligned_fields}
    replacements.update({
        "node_count": count,
        "raw_nodes": raw_nodes,
        "raw_edges": tuple(raw_edges),
        "predicted_operation_node_indices": operations,
        "predicted_operation_count": len(operations),
        "operation_count_exceeds_limit": False,
        "raw_operation_pointers": tuple(
            RawDecodedPointer(index, value, 1.0)
            for index, value in enumerate(operations)
        ),
        "legal_node_type_masks": tuple(legal_masks),
        "grammar_state_evidence": tuple(evidence),
        "same_history_raw_shadow_nodes": shadow_nodes,
        "same_history_raw_shadow_operation_node_indices": shadow_operations,
        "same_history_raw_shadow_operation_count": len(shadow_operations),
        "same_history_raw_shadow_operation_count_exceeds_limit": False,
        "same_history_raw_shadow_operation_pointers": tuple(
            RawDecodedPointer(index, value, 1.0)
            for index, value in enumerate(shadow_operations)
        ),
    })
    prefix_node = replace(node, **replacements)
    return graph_prediction_from_evidence(
        prefix_node,
        tuple(raw_matrix),
        tuple(masked_matrix),
        tuple(correction_matrix),
        main_pair_logits=(None if main_logits is None else tuple(main_logits)),
        position_bias_logits=(
            None if position_logits is None else tuple(position_logits)
        ),
    )


def score_condition(condition_result, targets_by_family):
    """Introduce targets only here, after complete autonomous generation."""

    start = time.perf_counter()
    if not condition_result.available:
        return {
            "schema_version": METRICS_SCHEMA_VERSION,
            "condition": condition_result.condition,
            "available": False,
            "unavailable_reason": condition_result.unavailable_reason,
            "family_values": {},
            "aggregates": {},
            "metric_computation_seconds": time.perf_counter() - start,
        }
    predictions = tuple(condition_result.predictions)
    families = tuple(item.family_id for item in predictions)
    if set(families) != set(targets_by_family):
        raise GraphEncoderError(
            "metric_target_alignment_failure", "prediction/target families differ"
        )
    grouped = {}
    for item in predictions:
        grouped.setdefault(item.family_id, []).append(
            _score_one(item, targets_by_family[item.family_id])
        )
    family_values = {
        family_id: _within_family(tuple(grouped[family_id]))
        for family_id in sorted(grouped)
    }
    aggregates = _macro_aggregates(family_values)
    return {
        "schema_version": METRICS_SCHEMA_VERSION,
        "aggregation_version": AGGREGATION_VERSION,
        "condition": condition_result.condition,
        "available": True,
        "unavailable_reason": None,
        "family_values": family_values,
        "aggregates": aggregates,
        "physical_family_denominator": len(family_values),
        "sample_record_denominator": len(predictions),
        "metric_computation_seconds": time.perf_counter() - start,
    }


def _score_one(item, target):
    from prototype.graph_baseline.graph_contract import (
        graph_edge_class_id,
        graph_from_reconstruction_target,
    )

    prediction = item.constrained_prediction
    expected = graph_from_reconstruction_target(target)
    observed_nodes = tuple(prediction.graph.node_type_ids)
    expected_nodes = tuple(expected.node_type_ids)
    observed_edges = {
        (edge.source, edge.destination, edge.edge_type_id)
        for edge in prediction.graph.directed_typed_edges
    }
    expected_edges = {
        (edge.source, edge.destination, edge.edge_type_id)
        for edge in expected.directed_typed_edges
    }
    operation_count = len(target.operation_sequence)
    prefix = score_prediction_prefix(prediction, operation_count)
    outcome = item.converted_prediction
    result = outcome.result if not outcome.raised_failure else None
    strict = bool(result is not None and result.reconstruction_target.valid)
    analytic = bool(result is not None and result.controlled_domain.valid)
    if outcome.raised_failure:
        failure_code = outcome.failure_type
        failure_stage = "strict conversion"
    elif result.primary_failure is None:
        failure_code = "valid"
        failure_stage = "no failure"
    else:
        failure_code = result.primary_failure.code
        failure_stage = _broad_failure_stage(result.primary_failure.stage)

    depends = graph_edge_class_id("depends_on")
    predicted_depends = {edge for edge in observed_edges if edge[2] == depends}
    expected_depends = {edge for edge in expected_edges if edge[2] == depends}
    depends_tp = len(predicted_depends & expected_depends)
    precision_denominator = len(predicted_depends)
    recall_denominator = len(expected_depends)
    precision = (
        depends_tp / float(precision_denominator)
        if precision_denominator else undefined("zero_predicted_depends_on_denominator")
    )
    recall = (
        depends_tp / float(recall_denominator)
        if recall_denominator else undefined("zero_target_depends_on_denominator")
    )
    attachment_ids = {
        graph_edge_class_id(name) for name in ATTACHMENT_EDGE_NAMES
    }
    predicted_attachments = {edge for edge in observed_edges if edge[2] in attachment_ids}
    expected_attachments = {edge for edge in expected_edges if edge[2] in attachment_ids}
    attachment_denominator = len(expected_attachments)
    attachment_accuracy = (
        len(predicted_attachments & expected_attachments) /
        float(attachment_denominator)
        if attachment_denominator else undefined("zero_target_attachment_denominator")
    )
    geometry = _geometry_errors(prediction, target)
    return {
        "family_id": item.family_id,
        "primary": prefix.normalized_longest_executable_operation_prefix,
        "prefix": prefix.to_dict(),
        "complete_executable_validity": float(analytic),
        "valid_single_solid": float(analytic),
        "strict_conversion": float(strict),
        "exact_graph": float(
            observed_nodes == expected_nodes and observed_edges == expected_edges
        ),
        "exact_node_sequence": float(observed_nodes == expected_nodes),
        "depends_on_precision": precision,
        "depends_on_precision_denominator": precision_denominator,
        "depends_on_recall": recall,
        "depends_on_recall_denominator": recall_denominator,
        "depends_on_exactness": float(predicted_depends == expected_depends),
        "attachment_accuracy": attachment_accuracy,
        "attachment_denominator": attachment_denominator,
        "geometry_error_by_channel_family": geometry,
        "first_failure_code": failure_code,
        "stage_of_first_failure": failure_stage,
        "unnormalized_longest_executable_prefix": (
            prefix.unnormalized_longest_executable_prefix
        ),
    }


def _geometry_errors(prediction, target):
    nodes = prediction.node_prediction.raw_nodes
    result = {}
    for family, indices in GEOMETRY_CHANNEL_FAMILIES:
        absolute = []
        for node_index in range(min(len(nodes), len(target.geometry))):
            for channel in indices:
                if target.geometry_mask[node_index][channel]:
                    predicted = nodes[node_index].normalized_geometry[channel]
                    expected = target.geometry[node_index][channel]
                    absolute.append(
                        abs(float(predicted) - float(expected)) *
                        GEOMETRY_CHANNEL_SCALES[channel]
                    )
        result[family] = {
            "physical_mae": (
                sum(absolute) / float(len(absolute)) if absolute
                else undefined("zero_applicable_target_channels")
            ),
            "applicable_channel_denominator": len(absolute),
            "channel_indices": list(indices),
            "channel_names": [GEOMETRY_CHANNELS[index] for index in indices],
            "success_tolerance": undefined("diagnostic_error_has_no_success_tolerance"),
        }
    return result


def _within_family(rows):
    if len(rows) == 1:
        return rows[0]
    result = {"family_id": rows[0]["family_id"], "sample_record_count": len(rows)}
    numeric = (
        "primary", "complete_executable_validity", "valid_single_solid",
        "strict_conversion", "exact_graph", "exact_node_sequence",
        "depends_on_exactness", "unnormalized_longest_executable_prefix",
    )
    for name in numeric:
        result[name] = sum(row[name] for row in rows) / float(len(rows))
    for name in ("depends_on_precision", "depends_on_recall", "attachment_accuracy"):
        values = [row[name] for row in rows if isinstance(row[name], (int, float))]
        result[name] = (
            sum(values) / float(len(values)) if values
            else undefined("all_within_family_values_undefined")
        )
    result["prefix"] = [row["prefix"] for row in rows]
    result["geometry_error_by_channel_family"] = {}
    for family, unused_indices in GEOMETRY_CHANNEL_FAMILIES:
        del unused_indices
        values = [
            row["geometry_error_by_channel_family"][family]["physical_mae"]
            for row in rows
        ]
        defined = [item for item in values if isinstance(item, (int, float))]
        result["geometry_error_by_channel_family"][family] = {
            "physical_mae": (
                sum(defined) / float(len(defined)) if defined
                else undefined("all_within_family_values_undefined")
            ),
            "defined_sample_denominator": len(defined),
            "sample_record_denominator": len(rows),
        }
    failure_codes = tuple(row["first_failure_code"] for row in rows)
    failure_stages = tuple(row["stage_of_first_failure"] for row in rows)
    result["first_failure_code"] = (
        failure_codes[0] if len(set(failure_codes)) == 1 else "mixed"
    )
    result["stage_of_first_failure"] = (
        failure_stages[0] if len(set(failure_stages)) == 1 else "mixed"
    )
    result["sample_first_failure_codes"] = list(failure_codes)
    result["sample_first_failure_stages"] = list(failure_stages)
    return result


def _macro_aggregates(family_values):
    families = tuple(family_values.values())
    result = {"physical_family_denominator": len(families)}
    numeric = (
        "primary", "complete_executable_validity", "valid_single_solid",
        "strict_conversion", "exact_graph", "exact_node_sequence",
        "depends_on_precision", "depends_on_recall", "depends_on_exactness",
        "attachment_accuracy", "unnormalized_longest_executable_prefix",
    )
    for name in numeric:
        values = [row[name] for row in families if isinstance(row[name], (int, float))]
        result[name] = {
            "value": sum(values) / float(len(values)) if values else None,
            "defined_family_denominator": len(values),
            "physical_family_denominator": len(families),
            "undefined_reason": None if values else "all_family_values_undefined",
        }
    failures = Counter(row["first_failure_code"] for row in families)
    stages = Counter(row["stage_of_first_failure"] for row in families)
    result["first_failure_histogram"] = dict(sorted(failures.items()))
    result["stage_of_first_failure_histogram"] = dict(sorted(stages.items()))
    result["geometry_error_by_channel_family"] = {}
    for family, unused_indices in GEOMETRY_CHANNEL_FAMILIES:
        del unused_indices
        values = []
        for row in families:
            value = row["geometry_error_by_channel_family"][family]["physical_mae"]
            if isinstance(value, (int, float)):
                values.append(value)
        result["geometry_error_by_channel_family"][family] = {
            "value": sum(values) / float(len(values)) if values else None,
            "defined_family_denominator": len(values),
            "physical_family_denominator": len(families),
            "undefined_reason": None if values else "all_family_values_undefined",
        }
    return result


def intervention_ratios(condition_metrics):
    by_name = {item["condition"]: item for item in condition_metrics}
    true = _aggregate_primary(by_name.get("P_true"))
    shuffled = _aggregate_primary(by_name.get("P_shuffle"))
    mean = _aggregate_primary(by_name.get("P_mean"))
    return {
        "P_true": true,
        "P_shuffle": shuffled,
        "P_mean": mean,
        "R_shuffle": _ratio(shuffled, true),
        "R_mean": _ratio(mean, true),
    }


def validate_primary_report(record):
    """Reject any primary result that omits the memory-intervention evidence.

    Every value must be finite numeric evidence or a structured undefined value
    with a nonempty reason. Bare ``None`` is ambiguous and is rejected.
    """

    ratios = record.get("intervention_ratios")
    if not isinstance(ratios, dict):
        raise GraphEncoderError(
            "incomplete_primary_report",
            "a primary result must carry an intervention_ratios block",
        )
    missing = tuple(
        name
        for name in PRIMARY_REPORT_REQUIRED_INTERVENTION_KEYS
        if name not in ratios
    )
    if missing:
        raise GraphEncoderError(
            "incomplete_primary_report",
            "primary report omits {}".format(", ".join(missing)),
        )
    malformed = tuple(
        name
        for name in PRIMARY_REPORT_REQUIRED_INTERVENTION_KEYS
        if not _valid_reported_metric(ratios[name])
    )
    if malformed:
        raise GraphEncoderError(
            "incomplete_primary_report",
            "primary report has unreasoned or nonfinite values for {}".format(
                ", ".join(malformed)
            ),
        )
    return record


def _valid_reported_metric(value):
    if (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    ):
        return True
    return (
        isinstance(value, dict)
        and value.get("value") is None
        and isinstance(value.get("reason"), str)
        and bool(value["reason"].strip())
    )


def _unavailable_donor_agreement(condition, reason, assignment_count):
    return {
        "version": DONOR_TEMPLATE_AGREEMENT_VERSION,
        "condition": condition,
        "available": False,
        "unavailable_reason": reason,
        "assignment_count": int(assignment_count),
        "same_template_donor_count": None,
        "observed_agreement": None,
        "chance_agreement": None,
        "agreement_excess_over_chance": None,
        "recipient_template_counts": {},
        "chance_baseline": "random_distinct_family_donor",
    }


def donor_template_agreement(memory_assignments, templates_by_family, *, condition):
    """Report how often a shuffled family donor shares its recipient's template.

    The frozen intervention is a deterministic cyclic derangement over sorted
    family IDs, so donors sit at a fixed offset rather than being drawn
    uniformly. If sorted IDs correlate with operation template, donors may
    systematically share the recipient's template, which makes `P_shuffle`
    easier and pushes `R_shuffle` upward. That direction is conservative for the
    frozen ratio gate, but a borderline value cannot be interpreted without
    knowing the agreement rate, so it is reported rather than assumed. The
    chance baseline conditions on the frozen no-self-donor constraint.

    P_true has no distinct donor and P_mean has a synthetic batch-mean source;
    both are reported as structurally unavailable instead of being resolved
    through the physical-family template map.

    This is a diagnostic only. It does not alter the derangement, the ratios, or
    any gate.
    """

    raw_assignments = tuple(memory_assignments)
    if condition == "P_true":
        return _unavailable_donor_agreement(
            condition,
            "recipient memory has no distinct donor family",
            len(raw_assignments),
        )
    if condition == "P_mean":
        return _unavailable_donor_agreement(
            condition,
            "batch-mean memory is synthetic and has no donor family",
            len(raw_assignments),
        )
    if condition != "P_shuffle":
        raise GraphEncoderError(
            "metric_target_alignment_failure",
            "donor agreement received an unknown memory condition",
        )
    if not raw_assignments:
        return _unavailable_donor_agreement(
            condition, "shuffle condition is unavailable", 0
        )
    if any(
        donor is None or donor == recipient
        for recipient, donor in raw_assignments
    ):
        raise GraphEncoderError(
            "metric_target_alignment_failure",
            "P_shuffle donor agreement requires one distinct family donor per row",
        )
    assignments = raw_assignments
    unknown = tuple(
        name
        for pair in assignments
        for name in pair
        if name not in templates_by_family
    )
    if unknown:
        raise GraphEncoderError(
            "metric_target_alignment_failure",
            "donor agreement requires a template for every assigned family",
        )
    same = sum(
        1
        for recipient, donor in assignments
        if templates_by_family[recipient] == templates_by_family[donor]
    )
    counts = Counter(templates_by_family[recipient] for recipient, _ in assignments)
    assignment_count = len(assignments)
    if assignment_count < 2:
        raise GraphEncoderError(
            "metric_target_alignment_failure",
            "P_shuffle donor agreement requires at least two assignments",
        )
    total = float(assignment_count)
    chance = sum(
        value * (value - 1)
        for value in counts.values()
    ) / (total * (total - 1.0))
    return {
        "version": DONOR_TEMPLATE_AGREEMENT_VERSION,
        "condition": condition,
        "available": True,
        "unavailable_reason": None,
        "assignment_count": assignment_count,
        "same_template_donor_count": same,
        "observed_agreement": same / total,
        "chance_agreement": chance,
        "agreement_excess_over_chance": (same / total) - chance,
        "recipient_template_counts": dict(sorted(counts.items())),
        "chance_baseline": "random_distinct_family_donor",
    }


def _aggregate_primary(record):
    if not record or not record.get("available"):
        return undefined("condition_unavailable")
    return record["aggregates"]["primary"]["value"]


def _ratio(numerator, denominator):
    if not isinstance(numerator, (int, float)):
        return undefined("numerator_unavailable")
    if not isinstance(denominator, (int, float)):
        return undefined("true_condition_unavailable")
    if denominator == 0:
        return undefined("zero_true_denominator")
    return numerator / denominator


def parameter_count_record(flat_model, graph_model):
    """Report actual instantiated trainable counts and enforce C4 capacity."""

    from .encoders import encoder_parameter_report

    flat_encoder = encoder_parameter_report(flat_model.encoder)
    graph_encoder = encoder_parameter_report(graph_model.encoder)
    flat_count = sum(
        parameter.numel() for parameter in flat_model.encoder.parameters()
        if parameter.requires_grad
    )
    graph_count = sum(
        parameter.numel() for parameter in graph_model.encoder.parameters()
        if parameter.requires_grad
    )
    if (flat_count, graph_count) != (
        EXPECTED_FLAT_ENCODER_PARAMETERS, EXPECTED_GRAPH_ENCODER_PARAMETERS
    ):
        raise GraphEncoderError(
            "capacity_record_mismatch",
            "actual encoder counts are {}/{}".format(flat_count, graph_count),
        )
    decoder = _module_parameter_groups(flat_model.decoder)
    graph_decoder = _module_parameter_groups(graph_model.decoder)
    if decoder != graph_decoder:
        raise GraphEncoderError(
            "shared_decoder_capacity_mismatch", "decoder counts differ by arm"
        )
    decoder_total = sum(
        parameter.numel() for parameter in flat_model.decoder.parameters()
        if parameter.requires_grad
    )
    difference = graph_count - flat_count
    relative = difference * 100.0 / float(flat_count)
    if difference != EXPECTED_ENCODER_DIFFERENCE or not math.isclose(
        relative, EXPECTED_RELATIVE_DIFFERENCE_PERCENT, rel_tol=0.0, abs_tol=1e-12
    ):
        raise GraphEncoderError("capacity_record_mismatch", "C4 delta differs")
    return {
        "flat_encoder_by_component": flat_encoder,
        "graph_encoder_by_component": graph_encoder,
        "shared_decoder_by_top_level_component": decoder,
        "flat_total_encoder": flat_count,
        "graph_total_encoder": graph_count,
        "total_decoder": decoder_total,
        "flat_total_model": flat_count + decoder_total,
        "graph_total_model": graph_count + decoder_total,
        "encoder_difference": difference,
        "encoder_relative_difference_percent": relative,
    }


def _module_parameter_groups(module):
    groups = {}
    for name, parameter in module.named_parameters():
        if parameter.requires_grad:
            top = name.split(".", 1)[0]
            groups[top] = groups.get(top, 0) + parameter.numel()
    return dict(sorted(groups.items()))


def receptive_field_record():
    return {
        "relational_message_passing_layers": RELATIONAL_MESSAGE_PASSING_LAYERS,
        "verified_maximum_undirected_controlled_template_diameter": (
            MAX_CONTROLLED_TEMPLATE_DIAMETER
        ),
        "maximum_distance_covered": True,
        "interpretation": (
            "architectural coverage only; not a claim that information is learned"
        ),
    }


def peak_memory_record():
    """Return the cumulative process high-water mark without claiming a phase peak."""

    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == "Darwin":
        bytes_value = int(usage)
        native_units = "bytes"
    else:
        bytes_value = int(usage) * 1024
        native_units = "KiB"
    return {
        "method": "resource.getrusage(resource.RUSAGE_SELF).ru_maxrss",
        "native_value": usage,
        "native_units": native_units,
        "bytes": bytes_value,
        "scope": "current_process_only",
        "includes_children": False,
        "resettable_within_process": False,
        "interpretation": "cumulative process high-water mark at observation time",
    }


def complete_metrics_record(
    *, arm, seed, epoch, checkpoint_identity, training_result,
    autonomous_result, condition_metrics, parameter_counts,
    strict_checkpoint_reload, total_run_seconds, templates_by_family
):
    """Build the complete versioned C6 smoke artifact.

    `templates_by_family` is required so that donor/recipient template
    agreement is reported alongside the frozen intervention ratios.
    """

    timings = {
        "monotonic_clock": "time.perf_counter",
        "data_preparation_seconds": dict(training_result.timing)[
            "data_preparation_seconds"
        ],
        "training_seconds": dict(training_result.timing)["training_seconds"],
        "checkpoint_and_provenance_seconds": dict(training_result.timing)[
            "checkpoint_and_provenance_seconds"
        ],
        "autonomous_encoding_seconds": autonomous_result.encoding_seconds,
        "autonomous_evaluation_seconds_by_condition": {
            item.condition: item.elapsed_seconds for item in autonomous_result.conditions
        },
        "metric_computation_seconds_by_condition": {
            item["condition"]: item["metric_computation_seconds"]
            for item in condition_metrics
        },
        "memory_intervention_seconds": sum(
            item.intervention_seconds for item in autonomous_result.conditions
            if item.condition != "P_true"
        ),
        "total_run_seconds": float(total_run_seconds),
        "queue_wait_included": False,
    }
    templates = dict(templates_by_family)
    record = {
        "schema_version": METRICS_SCHEMA_VERSION,
        "primary_reporting_contract": PRIMARY_REPORTING_CONTRACT_VERSION,
        "run_identity": training_result.run_identity,
        "arm": arm,
        "seed": int(seed),
        "epoch": int(epoch),
        "checkpoint_identity": checkpoint_identity,
        "checkpoint_schema": training_result.provenance.checkpoint_schema,
        "engineering_smoke_only": True,
        "scientific_result": False,
        "training": training_result.to_dict(),
        "parameter_counts": parameter_counts,
        "receptive_field": receptive_field_record(),
        "wall_time": timings,
        "peak_memory": peak_memory_record(),
        "conditions": list(condition_metrics),
        "intervention_ratios": intervention_ratios(condition_metrics),
        "donor_template_agreement": [
            donor_template_agreement(
                item.memory_assignments, templates, condition=item.condition
            )
            for item in autonomous_result.conditions
        ],
        "strict_checkpoint_reload": bool(strict_checkpoint_reload),
        "protected_partition_access": False,
        "development_access": False,
        "c7_or_later_performed": False,
    }
    return validate_primary_report(record)


def _broad_failure_stage(conversion_stage):
    if conversion_stage in ("controlled_geometry",):
        return "geometry prediction/reconstruction"
    if conversion_stage in ("controlled_edges", "cross_record_consistency"):
        return "edge prediction"
    if conversion_stage in ("controlled_node_grammar", "controlled_operations"):
        return "node rollout"
    return "strict conversion"
