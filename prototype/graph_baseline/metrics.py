"""Pair- and graph-level metrics for graph-v1 predictions."""

from __future__ import annotations

from prototype.model_data.vocab import NODE_TYPES

from .graph_contract import (
    GRAPH_EDGE_CLASS_ORDER,
    GraphContractError,
    edge_type_is_compatible,
    graph_from_reconstruction_target,
    position_only_canonical_edges,
)


FROZEN_V6_REFERENCE = {
    "pilot_job": "3338639",
    "commit": "ac6ef718ae9bab7fa5a80d9f48d0976adf5cafad",
    "iid_examples": 68,
    "conversion_success": 68,
    "complete_validity": 0,
    "unexpected_edge_failures": 68,
}


def new_graph_metrics():
    return {
        "active_ordered_pair_count": 0,
        "positive_target_edge_count": 0,
        "negative_target_pair_count": 0,
        "raw_correct": 0,
        "masked_correct": 0,
        "raw_true_positive": 0,
        "raw_false_positive": 0,
        "raw_false_negative": 0,
        "masked_true_positive": 0,
        "masked_false_positive": 0,
        "masked_false_negative": 0,
        "masked_wrong_type_edge_count": 0,
        "masked_typed_true_positive": 0,
        "masked_typed_false_positive": 0,
        "masked_typed_false_negative": 0,
        "self_edge_prediction_count": 0,
        "inactive_node_edge_prediction_count": 0,
        "nonfinite_logit_count": 0,
        "graph_edge_correction_count": 0,
        "exact_graph_match_count": 0,
        "raw_exact_graph_match_count": 0,
        "raw_graph_valid_count": 0,
        "raw_conversion_success_count": 0,
        "raw_first_failure_histogram": {},
        "exact_edge_set_match_count": 0,
        "position_only_exact_graph_match_count": 0,
        "node_type_pair_exact_graph_match_count": 0,
        "edge_type_prior_exact_graph_match_count": 0,
        "graph_valid_count": 0,
        "conversion_success_count": 0,
        "complete_cad_validity_count": 0,
        "first_failure_histogram": {},
        "structural_violation_histogram": {},
        "per_edge_type": {
            name: {"true_positive": 0, "false_positive": 0, "false_negative": 0, "support": 0}
            for name in GRAPH_EDGE_CLASS_ORDER[1:]
        },
        "example_count": 0,
        "outcomes": [],
    }


def update_pair_metrics(stats, prediction, target):
    graph = graph_from_reconstruction_target(target)
    count = graph.node_count
    targets = [[0 for _ in range(count)] for _ in range(count)]
    for edge in graph.directed_typed_edges:
        targets[edge.source][edge.destination] = edge.edge_type_id
    stats["self_edge_prediction_count"] += prediction.raw_self_edge_prediction_count
    stats["inactive_node_edge_prediction_count"] += (
        prediction.raw_inactive_node_edge_prediction_count
    )
    stats["graph_edge_correction_count"] += prediction.graph_edge_correction_count
    for source in range(count):
        for destination in range(count):
            if source == destination:
                continue
            expected = targets[source][destination]
            raw = prediction.raw_graph_edge_predictions[source][destination]
            masked = prediction.masked_graph_edge_predictions[source][destination]
            stats["active_ordered_pair_count"] += 1
            stats["positive_target_edge_count"] += int(expected != 0)
            stats["negative_target_pair_count"] += int(expected == 0)
            stats["raw_correct"] += int(raw == expected)
            stats["masked_correct"] += int(masked == expected)
            _binary(stats, "raw", raw, expected)
            _binary(stats, "masked", masked, expected)
            if masked != 0 and expected != 0 and masked != expected:
                stats["masked_wrong_type_edge_count"] += 1
            stats["masked_typed_true_positive"] += int(
                masked != 0 and masked == expected
            )
            stats["masked_typed_false_positive"] += int(
                masked != 0 and masked != expected
            )
            stats["masked_typed_false_negative"] += int(
                expected != 0 and masked != expected
            )
            for class_id, name in enumerate(GRAPH_EDGE_CLASS_ORDER[1:], start=1):
                row = stats["per_edge_type"][name]
                row["support"] += int(expected == class_id)
                row["true_positive"] += int(masked == class_id and expected == class_id)
                row["false_positive"] += int(masked == class_id and expected != class_id)
                row["false_negative"] += int(masked != class_id and expected == class_id)


def update_graph_outcome(stats, prediction, target, result, metadata):
    expected = graph_from_reconstruction_target(target)
    predicted_typed = {
        (item.source, item.destination, item.edge_type_id)
        for item in prediction.graph.directed_typed_edges
    }
    expected_typed = {
        (item.source, item.destination, item.edge_type_id)
        for item in expected.directed_typed_edges
    }
    predicted_pairs = {(a, b) for a, b, unused in predicted_typed}
    expected_pairs = {(a, b) for a, b, unused in expected_typed}
    exact = predicted_typed == expected_typed
    position_only = {
        (item.source, item.destination, item.edge_type_id)
        for item in position_only_canonical_edges(prediction.graph.node_type_ids)
    }
    node_type_pair = set()
    for source, source_id in enumerate(prediction.graph.node_type_ids):
        for destination, destination_id in enumerate(prediction.graph.node_type_ids):
            if source == destination:
                continue
            allowed = [
                class_id for class_id in range(1, len(GRAPH_EDGE_CLASS_ORDER))
                if edge_type_is_compatible(source_id, destination_id, class_id)
            ]
            if len(allowed) == 1:
                node_type_pair.add((source, destination, allowed[0]))
    code = "valid" if result.primary_failure is None else result.primary_failure.code
    stats["example_count"] += 1
    stats["exact_graph_match_count"] += int(
        exact and prediction.graph.node_type_ids == expected.node_type_ids
    )
    stats["exact_edge_set_match_count"] += int(predicted_pairs == expected_pairs)
    stats["position_only_exact_graph_match_count"] += int(position_only == expected_typed)
    stats["node_type_pair_exact_graph_match_count"] += int(node_type_pair == expected_typed)
    # Compatibility permits at most one non-none class per node-type pair, so
    # the source/destination edge-type prior is exactly this scoring-only arm.
    stats["edge_type_prior_exact_graph_match_count"] += int(
        node_type_pair == expected_typed
    )
    stats["graph_valid_count"] += int(result.controlled_domain.valid)
    stats["conversion_success_count"] += int(result.reconstruction_target.valid)
    stats["complete_cad_validity_count"] += int(result.controlled_domain.valid)
    stats["first_failure_histogram"][code] = stats["first_failure_histogram"].get(code, 0) + 1
    if result.primary_failure is not None and result.primary_failure.stage in (
        "controlled_edges", "cross_record_consistency"
    ):
        stats["structural_violation_histogram"][code] = (
            stats["structural_violation_histogram"].get(code, 0) + 1
        )
    stats["outcomes"].append({
        "valid": bool(result.controlled_domain.valid),
        "conversion_success": bool(result.reconstruction_target.valid),
        "failure_code": code,
        "node_count": prediction.graph.node_count,
        "edge_count": len(prediction.graph.directed_typed_edges),
        "operation_family": metadata.operation_template,
        "profile_family": metadata.primitive_family,
        "reference_plane": metadata.reference_plane,
    })


def update_raw_graph_outcome(stats, prediction, target):
    from .conversion import (
        graph_prediction_from_evidence,
        validate_and_convert_graph_prediction,
    )
    expected = graph_from_reconstruction_target(target)
    count = prediction.graph.node_count
    raw_typed = {
        (source, destination, prediction.raw_graph_edge_predictions[source][destination])
        for source in range(count)
        for destination in range(count)
        if prediction.raw_graph_edge_predictions[source][destination] != 0
    }
    expected_typed = {
        (item.source, item.destination, item.edge_type_id)
        for item in expected.directed_typed_edges
    }
    stats["raw_exact_graph_match_count"] += int(raw_typed == expected_typed)
    try:
        raw_prediction = graph_prediction_from_evidence(
            prediction.node_prediction,
            prediction.raw_graph_edge_predictions,
            prediction.raw_graph_edge_predictions,
            tuple(tuple(False for _ in range(count)) for _ in range(count)),
        )
        result = validate_and_convert_graph_prediction(raw_prediction)
    except GraphContractError as exc:
        code = getattr(exc, "code", "invalid_raw_graph_representation")
    else:
        stats["raw_graph_valid_count"] += int(result.controlled_domain.valid)
        stats["raw_conversion_success_count"] += int(result.reconstruction_target.valid)
        code = "valid" if result.primary_failure is None else result.primary_failure.code
    histogram = stats["raw_first_failure_histogram"]
    histogram[code] = histogram.get(code, 0) + 1


def finish_graph_metrics(stats):
    result = dict(stats)
    pairs = stats["active_ordered_pair_count"]
    result["raw_edge_class_accuracy"] = stats["raw_correct"] / float(pairs) if pairs else 0.0
    result["masked_edge_class_accuracy"] = stats["masked_correct"] / float(pairs) if pairs else 0.0
    for prefix in ("raw", "masked"):
        tp = stats[prefix + "_true_positive"]
        fp = stats[prefix + "_false_positive"]
        fn = stats[prefix + "_false_negative"]
        precision = tp / float(tp + fp) if tp + fp else 0.0
        recall = tp / float(tp + fn) if tp + fn else 0.0
        result[prefix + "_positive_edge_precision"] = precision
        result[prefix + "_positive_edge_recall"] = recall
        result[prefix + "_positive_edge_f1"] = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )
    for name, row in result["per_edge_type"].items():
        tp, fp, fn = row["true_positive"], row["false_positive"], row["false_negative"]
        precision = tp / float(tp + fp) if tp + fp else 0.0
        recall = tp / float(tp + fn) if tp + fn else 0.0
        row["precision"] = precision
        row["recall"] = recall
        row["f1"] = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    typed_tp = stats["masked_typed_true_positive"]
    typed_fp = stats["masked_typed_false_positive"]
    typed_fn = stats["masked_typed_false_negative"]
    typed_precision = typed_tp / float(typed_tp + typed_fp) if typed_tp + typed_fp else 0.0
    typed_recall = typed_tp / float(typed_tp + typed_fn) if typed_tp + typed_fn else 0.0
    result["exact_typed_edge_precision"] = typed_precision
    result["exact_typed_edge_recall"] = typed_recall
    result["exact_typed_edge_f1"] = (
        2 * typed_precision * typed_recall / (typed_precision + typed_recall)
        if typed_precision + typed_recall else 0.0
    )
    examples = stats["example_count"]
    for name in (
        "exact_graph_match_count", "exact_edge_set_match_count", "graph_valid_count",
        "conversion_success_count", "complete_cad_validity_count",
        "position_only_exact_graph_match_count", "node_type_pair_exact_graph_match_count",
        "edge_type_prior_exact_graph_match_count",
        "raw_exact_graph_match_count", "raw_graph_valid_count",
        "raw_conversion_success_count",
    ):
        result[name.replace("_count", "_rate")] = (
            stats[name] / float(examples) if examples else 0.0
        )
    result["frozen_v6_reference"] = dict(FROZEN_V6_REFERENCE)
    result["examples_first_failure_differs_from_v6"] = sum(
        item["failure_code"] != "unexpected_edge" for item in stats["outcomes"]
    )
    result["examples_progressing_beyond_controlled_edges"] = sum(
        item["failure_code"] not in ("missing_required_edge", "unexpected_edge")
        for item in stats["outcomes"]
    )
    result["examples_becoming_completely_valid"] = stats["complete_cad_validity_count"]
    result["validity_by_node_count"] = _outcome_rates(stats["outcomes"], "node_count")
    result["validity_by_operation_count"] = {
        key: value for key, value in _derived_rates(
            stats["outcomes"], lambda item: len(item["operation_family"])
        ).items()
    }
    result["validity_by_operation_family"] = _outcome_rates(
        stats["outcomes"], "operation_family"
    )
    result["validity_by_profile_family"] = _outcome_rates(
        stats["outcomes"], "profile_family"
    )
    result["validity_by_reference_plane"] = _outcome_rates(
        stats["outcomes"], "reference_plane"
    )
    result["validity_by_edge_count"] = _outcome_rates(
        stats["outcomes"], "edge_count"
    )
    return result


def _binary(stats, prefix, predicted, expected):
    stats[prefix + "_true_positive"] += int(predicted != 0 and expected != 0)
    stats[prefix + "_false_positive"] += int(predicted != 0 and expected == 0)
    stats[prefix + "_false_negative"] += int(predicted == 0 and expected != 0)


def _outcome_rates(outcomes, field):
    return _derived_rates(outcomes, lambda item: item[field])


def _derived_rates(outcomes, key_function):
    groups = {}
    for item in outcomes:
        key = str(key_function(item))
        row = groups.setdefault(key, {"count": 0, "valid_count": 0})
        row["count"] += 1
        row["valid_count"] += int(item["valid"])
    for row in groups.values():
        row["validity"] = row["valid_count"] / float(row["count"])
    return {key: groups[key] for key in sorted(groups)}
