"""Prospective, target-isolated operation-geometry fidelity metrics."""

from __future__ import annotations

import math

from prototype.model_data.geometry import GEOMETRY_CHANNEL_SCALES
from prototype.model_data.vocab import NODE_TYPES

from .errors import GraphEncoderError


OPERATION_GEOMETRY_FIDELITY_VERSION = (
    "GE1-OPERATION-GEOMETRY-FIDELITY-v1"
)
EXTRUSION_GRID = (0.5, 1.0, 1.5, 2.0, 3.0)
REVOLVE_GRID = (45.0, 90.0, 180.0, 270.0, 360.0)
EXTRUSION_CHANNEL = 37
REVOLVE_CHANNEL = 38
EXTRUSION_SCALE = 4.0
REVOLVE_SCALE = 360.0
EXTRUSION_ABSOLUTE_ERROR_MAX_EXCLUSIVE = 0.25
REVOLVE_ABSOLUTE_ERROR_MAX_EXCLUSIVE = 22.5
OPERATION_TYPES = ("extrude", "revolve")


def operation_fidelity_contract():
    """Return the frozen target-grid, channel, scale, and threshold contract."""

    _validate_repository_scales()
    return {
        "version": OPERATION_GEOMETRY_FIDELITY_VERSION,
        "gate_name": "operation_geometry_fidelity",
        "generation_condition": "P_true",
        "comparison": "strictly_less_than_unrounded",
        "aggregation": (
            "all_operations_all_representation_variants_per_physical_family"
        ),
        "extrude": {
            "serialized_channel": EXTRUSION_CHANNEL,
            "physical_grid": list(EXTRUSION_GRID),
            "normalization_scale": EXTRUSION_SCALE,
            "absolute_physical_error_max_exclusive": (
                EXTRUSION_ABSOLUTE_ERROR_MAX_EXCLUSIVE
            ),
        },
        "revolve": {
            "serialized_channel": REVOLVE_CHANNEL,
            "physical_grid": list(REVOLVE_GRID),
            "normalization_scale": REVOLVE_SCALE,
            "absolute_physical_error_max_exclusive": (
                REVOLVE_ABSOLUTE_ERROR_MAX_EXCLUSIVE
            ),
        },
        "non_operation_geometry_thresholds_added": False,
        "cad_kernel_required_or_used": False,
    }


def operation_fidelity_record(
    *,
    family_id,
    representation_variant,
    operation_index,
    target_operation_type,
    predicted_operation_type,
    predicted_channel,
    predicted_mask_active,
    predicted_normalized_value,
    target_channel,
    target_mask_active,
    target_normalized_value,
):
    """Score one autonomous operation without rounding or averaging.

    This pure evaluator receives an already generated value and a target only
    after inference. It has no model, encoder, decoder, memory, optimizer, or
    checkpoint-control input.
    """

    if not isinstance(family_id, str) or not family_id:
        raise GraphEncoderError(
            "invalid_operation_fidelity_input", "family_id must be nonempty"
        )
    if (
        not isinstance(representation_variant, str)
        or not representation_variant
    ):
        raise GraphEncoderError(
            "invalid_operation_fidelity_input",
            "representation_variant must be nonempty",
        )
    if (
        isinstance(operation_index, bool)
        or not isinstance(operation_index, int)
        or operation_index < 0
    ):
        raise GraphEncoderError(
            "invalid_operation_fidelity_input",
            "operation_index must be a nonnegative integer",
        )
    if target_operation_type not in OPERATION_TYPES:
        raise GraphEncoderError(
            "invalid_operation_fidelity_input",
            "target operation must be extrude or revolve",
        )

    expected_channel, scale, threshold = _operation_contract(
        target_operation_type
    )
    reasons = []
    if predicted_operation_type != target_operation_type:
        reasons.append("wrong_operation_type")
    if target_channel != expected_channel:
        reasons.append("wrong_target_channel")
    if target_mask_active is not True:
        reasons.append("target_channel_missing_or_masked")
    if predicted_channel != expected_channel:
        reasons.append("wrong_predicted_channel")
    if predicted_mask_active is not True:
        reasons.append("predicted_channel_missing_or_masked")

    predicted_normalized = _finite_or_none(predicted_normalized_value)
    target_normalized = _finite_or_none(target_normalized_value)
    if predicted_normalized is None:
        reasons.append("predicted_value_missing_or_nonfinite")
    if target_normalized is None:
        reasons.append("target_value_missing_or_nonfinite")

    predicted_physical = (
        None if predicted_normalized is None else predicted_normalized * scale
    )
    target_physical = (
        None if target_normalized is None else target_normalized * scale
    )
    signed_error = (
        None
        if predicted_physical is None or target_physical is None
        else predicted_physical - target_physical
    )
    absolute_error = None if signed_error is None else abs(signed_error)
    within_tolerance = bool(
        absolute_error is not None and absolute_error < threshold
    )
    if absolute_error is not None and not within_tolerance:
        reasons.append("absolute_physical_error_not_strictly_below_threshold")

    passed = not reasons and within_tolerance
    return {
        "version": OPERATION_GEOMETRY_FIDELITY_VERSION,
        "family_id": family_id,
        "representation_variant": representation_variant,
        "sample_record_identity": "{}::{}".format(
            family_id, representation_variant
        ),
        "operation_index": operation_index,
        "target_operation_type": target_operation_type,
        "predicted_operation_type": predicted_operation_type,
        "target_channel": target_channel,
        "predicted_channel": predicted_channel,
        "expected_channel": expected_channel,
        "target_mask_active": target_mask_active is True,
        "predicted_mask_active": predicted_mask_active is True,
        "predicted_normalized_value": predicted_normalized,
        "predicted_physical_value": predicted_physical,
        "target_normalized_value": target_normalized,
        "target_physical_value": target_physical,
        "signed_physical_error": signed_error,
        "absolute_physical_error": absolute_error,
        "absolute_physical_error_threshold": threshold,
        "comparison": "absolute_physical_error_strictly_less_than_threshold",
        "rounded_for_comparison": False,
        "status": "pass" if passed else "fail",
        "failure_reasons": reasons,
        "first_failure_stage": (
            "no failure" if passed else "operation geometry fidelity"
        ),
    }


def operation_geometry_fidelity_gate(
    *,
    arm,
    subset_identity,
    condition_result,
    examples_by_family,
    checkpoint_identity,
    epoch,
    access_flags,
    protocol_version,
    gate_version,
    checkpoint_role,
    operation_magnitude_parameterization,
):
    """Score all P_true operations and conservatively aggregate families."""

    if arm not in ("flat", "typed_graph"):
        raise GraphEncoderError(
            "invalid_repaired_sufficiency_arm", "unknown encoder arm"
        )
    if subset_identity not in ("tiny", "scaled"):
        raise GraphEncoderError(
            "invalid_repaired_sufficiency_subset", "unknown subset"
        )
    if epoch != 200:
        raise GraphEncoderError(
            "wrong_repaired_checkpoint_epoch",
            "operation fidelity requires strict epoch 200",
        )
    if (
        getattr(condition_result, "condition", None) != "P_true"
        or getattr(condition_result, "available", None) is not True
    ):
        raise GraphEncoderError(
            "nonautonomous_operation_fidelity_input",
            "only available autonomous P_true output is eligible",
        )
    examples = dict(examples_by_family)
    predictions = tuple(getattr(condition_result, "predictions", ()))
    prediction_ids = tuple(item.family_id for item in predictions)
    if (
        len(prediction_ids) != len(set(prediction_ids))
        or set(prediction_ids) != set(examples)
        or len(examples) != (4 if subset_identity == "tiny" else 32)
    ):
        raise GraphEncoderError(
            "metric_target_alignment_failure",
            "fidelity predictions and physical-family targets differ",
        )

    evidence = []
    summaries = {}
    for prediction in sorted(predictions, key=lambda item: item.family_id):
        example = examples[prediction.family_id]
        rows = _prediction_operation_records(prediction, example)
        evidence.extend(rows)
        summaries[prediction.family_id] = summarize_operation_fidelity_family(
            rows,
            family_id=prediction.family_id,
            operation_template=example.metadata.operation_template,
            representation_variants=example.metadata.geometry_encodings,
            source_sample_ids=example.metadata.sample_ids,
        )

    failing_ids = [
        family_id
        for family_id in sorted(summaries)
        if summaries[family_id]["status"] != "pass"
    ]
    return {
        "version": protocol_version,
        "gate_version": gate_version,
        "fidelity_contract_version": OPERATION_GEOMETRY_FIDELITY_VERSION,
        "gate_name": "{}_operation_geometry_fidelity".format(
            subset_identity
        ),
        "status": "pass" if not failing_ids else "fail",
        "arm": arm,
        "seed": 2026,
        "subset_identity": subset_identity,
        "physical_family_count": len(examples),
        "generation_condition": "P_true",
        "teacher_forced_gate_eligible": False,
        "checkpoint_epoch": epoch,
        "checkpoint_identity": str(checkpoint_identity),
        "checkpoint_role": checkpoint_role,
        "operation_magnitude_parameterization": (
            operation_magnitude_parameterization
        ),
        "contract": operation_fidelity_contract(),
        "sample_operation_evidence": evidence,
        "family_summaries": summaries,
        "failing_family_ids": failing_ids,
        "failed_operations_by_family": {
            family_id: summaries[family_id]["failing_operation_records"]
            for family_id in failing_ids
        },
        "structured_reason": (
            "all_operation_geometry_fidelity_criteria_satisfied"
            if not failing_ids
            else "one_or_more_operations_failed_geometry_fidelity"
        ),
        "access": dict(access_flags),
    }


def summarize_operation_fidelity_family(
    records,
    *,
    family_id,
    operation_template,
    representation_variants,
    source_sample_ids=(),
):
    """Conservatively summarize already scored operation/variant records."""

    rows = tuple(records)
    if not rows or any(item.get("family_id") != family_id for item in rows):
        raise GraphEncoderError(
            "metric_target_alignment_failure",
            "family summary requires nonempty aligned operation records",
        )
    failures = [item for item in rows if item.get("status") != "pass"]
    errors = [
        item.get("absolute_physical_error")
        for item in rows
        if isinstance(item.get("absolute_physical_error"), (int, float))
        and not isinstance(item.get("absolute_physical_error"), bool)
        and math.isfinite(float(item.get("absolute_physical_error")))
    ]
    return {
        "family_id": family_id,
        "operation_template": operation_template,
        "representation_variants": list(representation_variants),
        "source_sample_ids": list(source_sample_ids),
        "variant_targets_verified_physically_equivalent_by_loader": True,
        "operation_record_count": len(rows),
        "maximum_absolute_physical_error": max(errors) if errors else None,
        "undefined_operation_value_present": any(
            item.get("absolute_physical_error") is None for item in rows
        ),
        "aggregation": "all_records_must_pass_no_averaging",
        "status": "pass" if not failures else "fail",
        "failing_operation_records": [
            {
                "sample_record_identity": item.get("sample_record_identity"),
                "operation_index": item.get("operation_index"),
                "operation_type": item.get("target_operation_type"),
                "failure_reasons": item.get("failure_reasons", []),
                "absolute_physical_error": item.get(
                    "absolute_physical_error"
                ),
                "threshold": item.get(
                    "absolute_physical_error_threshold"
                ),
                "first_failure_stage": item.get("first_failure_stage"),
            }
            for item in failures
        ],
    }


def _prediction_operation_records(prediction, example):
    target = example.target
    constrained = getattr(prediction, "constrained_prediction", None)
    node_prediction = getattr(constrained, "node_prediction", None)
    graph = getattr(constrained, "graph", None)
    raw_nodes = tuple(getattr(node_prediction, "raw_nodes", ()))
    predicted_operation_indices = tuple(
        getattr(node_prediction, "predicted_operation_node_indices", ())
    )
    predicted_node_type_ids = tuple(getattr(graph, "node_type_ids", ()))
    variants = tuple(example.metadata.geometry_encodings)
    if not variants:
        variants = ("physical",)
    rows = []
    for operation_index, target_node_index in enumerate(
        target.operation_sequence
    ):
        target_type = _node_type(target.node_type_ids, target_node_index)
        target_channel, unused_scale, unused_threshold = _operation_contract(
            target_type
        )
        del unused_scale, unused_threshold
        predicted_node_index = (
            predicted_operation_indices[operation_index]
            if operation_index < len(predicted_operation_indices)
            else None
        )
        predicted_type = _node_type(
            predicted_node_type_ids, predicted_node_index
        )
        predicted_node = _at(raw_nodes, predicted_node_index)
        predicted_mask = tuple(
            getattr(predicted_node, "derived_geometry_mask", ())
        )
        predicted_geometry = tuple(
            getattr(predicted_node, "normalized_geometry", ())
        )
        active_operation_channels = tuple(
            channel
            for channel in (EXTRUSION_CHANNEL, REVOLVE_CHANNEL)
            if _at(predicted_mask, channel) is True
        )
        predicted_channel = (
            active_operation_channels[0]
            if len(active_operation_channels) == 1
            else None
        )
        target_mask = _at(_at(target.geometry_mask, target_node_index), target_channel)
        target_value = _at(_at(target.geometry, target_node_index), target_channel)
        predicted_value = _at(predicted_geometry, target_channel)
        for variant in variants:
            rows.append(operation_fidelity_record(
                family_id=prediction.family_id,
                representation_variant=variant,
                operation_index=operation_index,
                target_operation_type=target_type,
                predicted_operation_type=predicted_type,
                predicted_channel=predicted_channel,
                predicted_mask_active=_at(predicted_mask, target_channel),
                predicted_normalized_value=predicted_value,
                target_channel=target_channel,
                target_mask_active=target_mask,
                target_normalized_value=target_value,
            ))
    return rows


def _operation_contract(operation_type):
    _validate_repository_scales()
    if operation_type == "extrude":
        return (
            EXTRUSION_CHANNEL,
            EXTRUSION_SCALE,
            EXTRUSION_ABSOLUTE_ERROR_MAX_EXCLUSIVE,
        )
    if operation_type == "revolve":
        return (
            REVOLVE_CHANNEL,
            REVOLVE_SCALE,
            REVOLVE_ABSOLUTE_ERROR_MAX_EXCLUSIVE,
        )
    raise GraphEncoderError(
        "invalid_operation_fidelity_input",
        "operation type must be extrude or revolve",
    )


def _validate_repository_scales():
    if (
        float(GEOMETRY_CHANNEL_SCALES[EXTRUSION_CHANNEL]) != EXTRUSION_SCALE
        or float(GEOMETRY_CHANNEL_SCALES[REVOLVE_CHANNEL]) != REVOLVE_SCALE
    ):
        raise GraphEncoderError(
            "operation_fidelity_contract_mismatch",
            "authoritative geometry scales differ from the frozen protocol",
        )


def _node_type(values, index):
    value = _at(values, index)
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    if not 0 <= value < len(NODE_TYPES.tokens):
        return None
    token = NODE_TYPES.tokens[value]
    return token if token in OPERATION_TYPES else token


def _at(values, index):
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        return None
    try:
        return values[index]
    except (IndexError, KeyError, TypeError):
        return None


def _finite_or_none(value):
    if isinstance(value, bool) or value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(numeric):
        return None
    return 0.0 if numeric == 0.0 else numeric
