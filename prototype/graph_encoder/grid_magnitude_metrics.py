"""GE1-owned grid-magnitude diagnostics for engineering validation.

The frozen Graph V1 metric schema is not extended.  These diagnostics are
versioned separately and consume targets only after teacher-forced loss
construction or complete autonomous generation.
"""

from __future__ import annotations

import math

from prototype.model_data.vocab import NODE_TYPES

from .errors import GraphEncoderError
from .grid_magnitude import (
    GRID_CLASS_COUNT,
    GRID_MAGNITUDE_DIAGNOSTICS_VERSION,
    GRID_MAGNITUDE_GRADIENT_VERSION,
    NORMALIZED_GRIDS,
    OPERATION_TYPES,
    PHYSICAL_GRIDS,
    SERIALIZED_CHANNELS,
    class_index_from_normalized_target,
)


GRID_MAGNITUDE_AUTONOMOUS_METRICS_VERSION = (
    "GE1-GRID-MAGNITUDE-AUTONOMOUS-METRICS-v1"
)
GRID_MAGNITUDE_VALUE_AGREEMENT_VERSION = (
    "GE1-GRID-MAGNITUDE-VALUE-AGREEMENT-v1"
)


def engineering_grid_magnitude_diagnostic(
    loss_terms,
    *,
    teacher_values=None,
    autonomous_values=None,
    active_masks=None,
    model=None
):
    """Assemble optional engineering evidence around structured loss terms."""

    record = loss_terms.diagnostic_record()
    supplied = (
        teacher_values is not None,
        autonomous_values is not None,
        active_masks is not None,
    )
    if any(supplied) and not all(supplied):
        raise GraphEncoderError(
            "incomplete_grid_value_agreement",
            "teacher values, autonomous values, and masks are all required",
        )
    if all(supplied):
        record["teacher_forced_autonomous_grid_value_agreement"] = (
            grid_value_agreement(
                teacher_values, autonomous_values, active_masks
            )
        )
    if model is not None:
        record["gradient_norms"] = engineering_gradient_norms(model)
    return record


def grid_value_agreement(teacher_values, autonomous_values, active_masks):
    """Compare two target-free decode surfaces on the same active positions."""

    by_type = {}
    total = 0
    equal = 0
    for operation_type in OPERATION_TYPES:
        expected = teacher_values[operation_type]
        observed = autonomous_values[operation_type]
        active = active_masks[operation_type]
        if expected.shape != observed.shape or active.shape != expected.shape:
            raise GraphEncoderError(
                "grid_metric_alignment_failure",
                "teacher/autonomous grid tensors are misaligned",
            )
        selected_expected = expected[active]
        selected_observed = observed[active]
        count = int(selected_expected.numel())
        matches = int((selected_expected == selected_observed).sum().item())
        total += count
        equal += matches
        by_type[operation_type] = {
            "comparison_count": count,
            "exact_agreement_count": matches,
            "exact_agreement": matches == count,
        }
    return {
        "version": GRID_MAGNITUDE_VALUE_AGREEMENT_VERSION,
        "available": True,
        "comparison_count": total,
        "exact_agreement_count": equal,
        "exact_agreement": equal == total,
        "operation_types": by_type,
        "comparison": "bit_exact_normalized_grid_value",
    }


def engineering_gradient_norms(model):
    """Report head and shared-decoder-trunk norms after a synthetic backward."""

    groups = {"grid_head": [], "shared_decoder_trunk": []}
    for name, parameter in model.named_parameters():
        if name.startswith("decoder.grid_magnitude_head."):
            groups["grid_head"].append((name, parameter))
        elif (
            name == "decoder.bos"
            or name.startswith("decoder.decoder.")
            or name.startswith("decoder.decoder_input_norm.")
            or name.startswith("decoder.decoder_position_embedding.")
        ):
            groups["shared_decoder_trunk"].append((name, parameter))
    result = {}
    for group, parameters in groups.items():
        squared = 0.0
        names = []
        missing = []
        for name, parameter in parameters:
            names.append(name)
            if parameter.grad is None:
                missing.append(name)
                continue
            values = parameter.grad.detach()
            if not bool(values.isfinite().all().item()):
                raise GraphEncoderError(
                    "nonfinite_grid_engineering_gradient",
                    "{} contains a nonfinite gradient".format(name),
                )
            squared += float((values.double() ** 2).sum().item())
        result[group] = {
            "parameter_names": names,
            "missing_gradient_names": missing,
            "l2_norm": math.sqrt(squared),
            "finite": True,
            "nonzero": squared > 0.0,
        }
    return {
        "version": GRID_MAGNITUDE_GRADIENT_VERSION,
        "available": True,
        "engineering_only": True,
        "scientific_training": False,
        "groups": result,
    }


def autonomous_grid_magnitude_diagnostics(prediction, target):
    """Score exact grid classes after autonomous generation is complete."""

    nodes = prediction.node_prediction.raw_nodes
    by_type_records = {name: [] for name in OPERATION_TYPES}
    for node_index, node_type_id in enumerate(target.node_type_ids):
        operation_type = NODE_TYPES.tokens[node_type_id]
        if operation_type not in OPERATION_TYPES:
            continue
        channel = SERIALIZED_CHANNELS[operation_type]
        normalized_target = float(target.geometry[node_index][channel])
        target_class = class_index_from_normalized_target(
            operation_type, normalized_target
        )
        record = {
            "node_index": node_index,
            "target_class": target_class,
            "target_normalized_value": normalized_target,
            "target_physical_value": PHYSICAL_GRIDS[operation_type][target_class],
            "predicted_class": None,
            "predicted_normalized_value": None,
            "predicted_physical_value": None,
            "prediction_available": False,
            "unavailable_reason": None,
            "ordinal_cut_probabilities": {
                "available": False,
                "reason": "autonomous_serialized_prediction_has_no_cut_logits",
            },
            "decision_margin": {
                "available": False,
                "reason": "autonomous_serialized_prediction_has_no_cut_logits",
            },
        }
        if node_index >= len(nodes):
            record["unavailable_reason"] = "predicted_node_missing"
        else:
            predicted_node = nodes[node_index]
            predicted_type = NODE_TYPES.tokens[predicted_node.node_type_id]
            if predicted_type != operation_type:
                record["unavailable_reason"] = "predicted_operation_type_differs"
            else:
                normalized = float(
                    predicted_node.normalized_geometry[channel]
                )
                try:
                    predicted_class = class_index_from_normalized_target(
                        operation_type, normalized
                    )
                except GraphEncoderError:
                    record["unavailable_reason"] = "predicted_value_is_off_grid"
                else:
                    record.update({
                        "predicted_class": predicted_class,
                        "predicted_normalized_value": normalized,
                        "predicted_physical_value": (
                            PHYSICAL_GRIDS[operation_type][predicted_class]
                        ),
                        "prediction_available": True,
                    })
        by_type_records[operation_type].append(record)
    return {
        "schema_version": GRID_MAGNITUDE_AUTONOMOUS_METRICS_VERSION,
        "diagnostics_contract": GRID_MAGNITUDE_DIAGNOSTICS_VERSION,
        "target_introduced_after_autonomous_generation": True,
        "operation_types": {
            name: _classification_summary(name, by_type_records[name])
            for name in OPERATION_TYPES
        },
    }


def combine_autonomous_grid_magnitude_diagnostics(diagnostics):
    """Combine row/family records without changing their scoring semantics."""

    values = tuple(diagnostics)
    by_type_records = {name: [] for name in OPERATION_TYPES}
    for diagnostic in values:
        if (
            diagnostic.get("schema_version")
            != GRID_MAGNITUDE_AUTONOMOUS_METRICS_VERSION
        ):
            raise GraphEncoderError(
                "grid_metric_schema_mismatch",
                "autonomous grid diagnostics use different schemas",
            )
        for operation_type in OPERATION_TYPES:
            by_type_records[operation_type].extend(
                diagnostic["operation_types"][operation_type][
                    "operation_records"
                ]
            )
    return {
        "schema_version": GRID_MAGNITUDE_AUTONOMOUS_METRICS_VERSION,
        "diagnostics_contract": GRID_MAGNITUDE_DIAGNOSTICS_VERSION,
        "target_introduced_after_autonomous_generation": True,
        "combined_record_count": len(values),
        "operation_types": {
            name: _classification_summary(name, by_type_records[name])
            for name in OPERATION_TYPES
        },
    }


def _classification_summary(operation_type, records):
    confusion = [
        [0] * GRID_CLASS_COUNT for _ in range(GRID_CLASS_COUNT)
    ]
    true_counts = [0] * GRID_CLASS_COUNT
    predicted_counts = [0] * GRID_CLASS_COUNT
    unavailable = 0
    for record in records:
        target = record["target_class"]
        true_counts[target] += 1
        predicted = record["predicted_class"]
        if predicted is None:
            unavailable += 1
            continue
        predicted_counts[predicted] += 1
        confusion[target][predicted] += 1
    recall = [
        (
            confusion[index][index] / float(true_counts[index])
            if true_counts[index]
            else None
        )
        for index in range(GRID_CLASS_COUNT)
    ]
    extremes = [recall[index] for index in (0, 4) if recall[index] is not None]
    return {
        "operation_records": records,
        "target_count": len(records),
        "unavailable_prediction_count": unavailable,
        "confusion_matrix": confusion,
        "true_class_counts": true_counts,
        "predicted_class_counts": predicted_counts,
        "per_class_recall": recall,
        "extreme_class_recall": {
            "class_0": recall[0],
            "class_4": recall[4],
            "mean_over_supported_extremes": (
                sum(extremes) / float(len(extremes)) if extremes else None
            ),
        },
        "zero_support_flags": [value == 0 for value in true_counts],
        "class_masking_flags": [
            true_counts[index] > 0 and predicted_counts[index] == 0
            for index in range(GRID_CLASS_COUNT)
        ],
        "normalized_grid": list(NORMALIZED_GRIDS[operation_type]),
        "physical_grid": list(PHYSICAL_GRIDS[operation_type]),
    }
