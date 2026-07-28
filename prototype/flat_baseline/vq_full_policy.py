"""Deterministic code-usage policy for full train-kmeans retraining."""

from __future__ import annotations

import math


ACTIVE_CODE_MINIMUM = 2
PERPLEXITY_MINIMUM = 2.0
CONSECUTIVE_VIOLATION_LIMIT = 2


class UsagePolicyError(ValueError):
    """A required code-usage value is malformed or non-finite."""


def usage_snapshot(record):
    diagnostics = record["diagnostics"]
    active = diagnostics["active_code_count"]
    perplexity = diagnostics["codebook_perplexity"]
    if (
        isinstance(active, bool)
        or not isinstance(active, int)
        or active < 0
        or isinstance(perplexity, bool)
        or not isinstance(perplexity, (int, float))
        or not math.isfinite(float(perplexity))
        or diagnostics.get("finite") is not True
        or diagnostics.get("ema_state_consistent") is not True
    ):
        raise UsagePolicyError("code-usage diagnostics are not finite")
    return {
        "epoch": record["epoch"],
        "active_code_count": active,
        "codebook_perplexity": float(perplexity),
    }


def threshold_violation(snapshot):
    return (
        snapshot["active_code_count"] < ACTIVE_CODE_MINIMUM
        or snapshot["codebook_perplexity"] < PERPLEXITY_MINIMUM
    )


def empty_collapse_state():
    return {
        "consecutive_violations": 0,
        "violation_count": 0,
        "maximum_consecutive_violations": 0,
    }


def training_epoch_monitor(state, record):
    snapshot = usage_snapshot(record)
    violation = threshold_violation(snapshot)
    consecutive = (
        state["consecutive_violations"] + 1 if violation else 0
    )
    updated = {
        "consecutive_violations": consecutive,
        "violation_count": state["violation_count"] + int(violation),
        "maximum_consecutive_violations": max(
            state["maximum_consecutive_violations"], consecutive
        ),
    }
    monitor = {
        "mode": "train",
        "epoch": snapshot["epoch"],
        "warning": violation,
        "warning_code": (
            "ASSIGNMENT_COLLAPSE_RISK" if violation else None
        ),
        "active_code_count_below_2": (
            snapshot["active_code_count"] < ACTIVE_CODE_MINIMUM
        ),
        "codebook_perplexity_below_2": (
            snapshot["codebook_perplexity"] < PERPLEXITY_MINIMUM
        ),
        "consecutive_training_violations": consecutive,
        "stop": consecutive >= CONSECUTIVE_VIOLATION_LIMIT,
    }
    return updated, monitor


def validation_epoch_monitor(record):
    snapshot = usage_snapshot(record)
    violation = threshold_violation(snapshot)
    return {
        "mode": "validation",
        "epoch": snapshot["epoch"],
        "warning": violation,
        "warning_code": (
            "ASSIGNMENT_COLLAPSE_RISK" if violation else None
        ),
        "active_code_count_below_2": (
            snapshot["active_code_count"] < ACTIVE_CODE_MINIMUM
        ),
        "codebook_perplexity_below_2": (
            snapshot["codebook_perplexity"] < PERPLEXITY_MINIMUM
        ),
        "stop": False,
    }


def code_usage_trend(records):
    ordered = sorted(
        records, key=lambda item: (item["epoch"], item["mode"])
    )
    by_mode = {
        mode: [
            usage_snapshot(record)
            for record in ordered
            if record["mode"] == mode
        ]
        for mode in ("train", "validation")
    }
    pilot = {}
    for mode in ("train", "validation"):
        matches = [
            item for item in by_mode[mode] if item["epoch"] == 5
        ]
        if len(matches) != 1:
            raise UsagePolicyError(
                "pilot epoch 5 {} usage is required".format(mode)
            )
        pilot[mode] = matches[0]
    snapshots = by_mode["train"] + by_mode["validation"]
    if not snapshots:
        raise UsagePolicyError("at least one usage record is required")
    state = empty_collapse_state()
    for record in ordered:
        if record["mode"] == "train":
            state, unused = training_epoch_monitor(state, record)
            del unused
    final_train = by_mode["train"][-5:]
    direction = _trend_direction(final_train)
    return {
        "summary_scope": (
            "all available pilot and full-retraining train and validation "
            "epoch records; violation streaks use training records only"
        ),
        "thresholds": {
            "active_code_count_minimum": ACTIVE_CODE_MINIMUM,
            "codebook_perplexity_minimum": PERPLEXITY_MINIMUM,
            "consecutive_training_violation_limit": (
                CONSECUTIVE_VIOLATION_LIMIT
            ),
        },
        "pilot_epoch_5": pilot,
        "final_five_train_epochs": final_train,
        "final_five_validation_epochs": by_mode["validation"][-5:],
        "active_code_count": {
            "minimum": min(
                item["active_code_count"] for item in snapshots
            ),
            "maximum": max(
                item["active_code_count"] for item in snapshots
            ),
        },
        "codebook_perplexity": {
            "minimum": min(
                item["codebook_perplexity"] for item in snapshots
            ),
            "maximum": max(
                item["codebook_perplexity"] for item in snapshots
            ),
        },
        "collapse_threshold_violation_count": state["violation_count"],
        "maximum_consecutive_violations": state[
            "maximum_consecutive_violations"
        ],
        "violation_scope": "completed_training_epochs",
        "trend": direction,
        "trend_rule": (
            "compare the first and last of the final five available training "
            "epochs: increasing requires both metrics nondecreasing and at "
            "least one increasing; decreasing requires both nonincreasing "
            "and at least one decreasing; all equal or mixed movement is stable"
        ),
    }


def final_training_decision(
    completed_epoch,
    authoritative_budget,
    stopped_reason,
    best_checkpoint_provenance_valid,
    best_required_values_finite,
    best_validation_record,
    test_partition_evaluated,
    trend,
):
    base = {
        "completed_epoch": completed_epoch,
        "test_partition_evaluated": test_partition_evaluated,
        "code_usage_trend": trend,
    }
    if stopped_reason is not None:
        return dict(base, decision="FAIL_ASSIGNMENT_COLLAPSE",
                    reason=stopped_reason)
    if test_partition_evaluated:
        return dict(base, decision="FAIL_PROVENANCE",
                    reason="test_partition_was_evaluated")
    if completed_epoch != authoritative_budget:
        return dict(base, decision="FAIL_OTHER",
                    reason="authoritative_epoch_budget_not_completed")
    if not best_checkpoint_provenance_valid:
        return dict(base, decision="FAIL_PROVENANCE",
                    reason="selected_best_checkpoint_provenance")
    if not best_required_values_finite:
        return dict(base, decision="FAIL_NUMERICAL",
                    reason="selected_best_checkpoint_nonfinite")
    try:
        selected = usage_snapshot(best_validation_record)
    except (KeyError, TypeError, UsagePolicyError):
        return dict(base, decision="FAIL_NUMERICAL",
                    reason="selected_validation_usage_nonfinite")
    if threshold_violation(selected):
        return dict(base, decision="FAIL_ASSIGNMENT_COLLAPSE",
                    reason="selected_validation_usage_gate")
    return dict(
        base,
        decision="PASS_FOR_EVALUATION",
        reason="full_training_and_selected_validation_gates_passed",
    )


def _trend_direction(records):
    if len(records) < 2:
        return "stable"
    first = records[0]
    last = records[-1]
    changes = (
        last["active_code_count"] - first["active_code_count"],
        last["codebook_perplexity"] - first["codebook_perplexity"],
    )
    if all(value >= 0 for value in changes) and any(
        value > 0 for value in changes
    ):
        return "increasing"
    if all(value <= 0 for value in changes) and any(
        value < 0 for value in changes
    ):
        return "decreasing"
    return "stable"
