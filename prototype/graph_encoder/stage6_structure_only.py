"""Prospective structure-only Stage 6 scoring and artifact contract.

This module deliberately contains no corpus, checkpoint, model, or training
loader.  A separately authorized scientific execution must produce the
target-free autonomous records consumed here; this entry point validates and
scores those records exactly once.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path

from .errors import GraphEncoderError


PROTOCOL_VERSION = "GE1-STAGE6-STRUCTURE-ONLY-COMPARISON-v1"
INPUT_VERSION = "GE1-STAGE6-STRUCTURE-ONLY-INPUT-v1"
SCORE_VERSION = "GE1-STAGE6-STRUCTURAL-PREFIX-v1"
AGGREGATION_VERSION = "GE1-STAGE6-PAIRED-SEED-AGGREGATION-v1"
MEMORY_VERSION = "GE1-STAGE6-STRUCTURAL-MEMORY-v1"
ARTIFACT_VERSION = "GE1-STAGE6-STRUCTURE-ONLY-ARTIFACT-v1"

ARMS = ("flat", "typed_graph")
CONDITIONS = ("P_true", "P_shuffle", "P_mean")
COHORTS = ("train", "development")
TEMPLATES = ("E", "R", "EE", "RE")
FULL_SEEDS = (2026, 2027, 2028)
FALLBACK_SEEDS = (2026, 2027)
TRAIN_FAMILY_COUNT = 407
DEVELOPMENT_FAMILY_COUNT = 45
EPOCHS = 200
BATCH_SIZE = 8
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0
CLIP_NORM = 1.0
PRIMARY_THRESHOLD = 0.10
MEMORY_RATIO_THRESHOLD = 0.80
STRUCTURAL_FIELDS = (
    "grammar_complete",
    "operation_types_exact",
    "chronological_order_exact",
    "depends_on_exact",
    "ownership_attachments_exact",
    "canonical_graph_exact",
)
FAILURE_STAGES = (
    "grammar_complete",
    "operation_types_exact",
    "chronological_order_exact",
    "canonical_graph_exact",
    "depends_on_exact",
    "ownership_attachments_exact",
    "no_failure",
)
STRUCTURAL_RECORD_FIELDS = {
    "k", "grammar_valid_complete", "canonicalization_succeeded",
    "prefix_construction_succeeded", "requested_node_count",
    "node_type_ids", "operation_type_ids", "canonical_edges",
    "depends_on_edges", "ownership_attachment_edges",
}
AUTHORITY = {
    "scientific_execution_authorized_by_implementation": False,
    "stage7_authorized": False,
    "rr_accessed": False,
    "er_accessed": False,
    "iid_accessed": False,
    "history_depth_accessed": False,
    "geometry_extrapolation_accessed": False,
    "cad_kernel_executed": False,
    "geometry_repair_authorized": False,
}


def protocol_config():
    return {
        "protocol_version": PROTOCOL_VERSION,
        "training_family_count": TRAIN_FAMILY_COUNT,
        "development_family_count": DEVELOPMENT_FAMILY_COUNT,
        "allowed_templates": list(TEMPLATES),
        "full_seeds": list(FULL_SEEDS),
        "fallback_seeds": list(FALLBACK_SEEDS),
        "arms": list(ARMS),
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "optimizer": "AdamW",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "gradient_clip_norm": CLIP_NORM,
        "checkpoint_epoch": EPOCHS,
        "early_stopping": False,
        "development_checkpoint_selection": False,
        "warm_start": False,
        "primary_threshold": PRIMARY_THRESHOLD,
        "memory_ratio_threshold": MEMORY_RATIO_THRESHOLD,
        "primary_uses_conversion": False,
        "primary_uses_analytic_validity": False,
        "primary_uses_geometry": False,
        "target_joined_after_generation": True,
    }


def _finite_number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GraphEncoderError("invalid_stage6_record", name + " is not numeric")
    value = float(value)
    if not math.isfinite(value):
        raise GraphEncoderError("invalid_stage6_record", name + " is nonfinite")
    return value


def score_structural_prefix(target_operation_count, prefix_evidence,
                            predicted_operation_group_count=None):
    """Return the longest leading operation prefix satisfying structure only."""

    if (
        isinstance(target_operation_count, bool)
        or not isinstance(target_operation_count, int)
        or target_operation_count not in (1, 2)
    ):
        raise GraphEncoderError(
            "invalid_stage6_denominator", "operation count must be one or two"
        )
    if predicted_operation_group_count is None:
        predicted_operation_group_count = target_operation_count
    if (
        isinstance(predicted_operation_group_count, bool)
        or not isinstance(predicted_operation_group_count, int)
        or predicted_operation_group_count < 0
    ):
        raise GraphEncoderError(
            "invalid_stage6_denominator", "predicted operation count differs"
        )
    rows = tuple(prefix_evidence)
    if tuple(row.get("k") for row in rows) != tuple(
        range(1, target_operation_count + 1)
    ):
        raise GraphEncoderError(
            "invalid_stage6_prefix_evidence", "one ordered row per prefix is required"
        )
    selected = 0
    first_failure = "no_failure"
    normalized_rows = []
    prefix_open = True
    for row in rows:
        if set(row) != {"k", *STRUCTURAL_FIELDS}:
            raise GraphEncoderError(
                "invalid_stage6_prefix_evidence", "structural fields differ"
            )
        if any(not isinstance(row[field], bool) for field in STRUCTURAL_FIELDS):
            raise GraphEncoderError(
                "invalid_stage6_prefix_evidence", "structural values must be bool"
            )
        passed = all(row[field] for field in STRUCTURAL_FIELDS)
        if prefix_open and passed:
            selected = row["k"]
        else:
            prefix_open = False
            if first_failure == "no_failure":
                first_failure = next(
                    field for field in STRUCTURAL_FIELDS if not row[field]
                )
        normalized_rows.append({
            "k": row["k"],
            **{field: row[field] for field in STRUCTURAL_FIELDS},
            "structurally_correct": passed,
        })
    exact_generation = predicted_operation_group_count == target_operation_count
    if selected == target_operation_count and not exact_generation:
        selected = max(0, target_operation_count - 1)
    return {
        "version": SCORE_VERSION,
        "target_operation_count": target_operation_count,
        "selected_k": selected,
        "normalized_structural_prefix": selected / float(target_operation_count),
        "predicted_operation_group_count": predicted_operation_group_count,
        "under_generation": predicted_operation_group_count < target_operation_count,
        "exact_generation": exact_generation,
        "over_generation": predicted_operation_group_count > target_operation_count,
        "first_failure_stage": first_failure,
        "prefixes": normalized_rows,
        "conversion_used": False,
        "analytic_validity_used": False,
        "geometry_used": False,
    }


def structural_prefix_evidence(predicted_prefixes, target_prefixes):
    """Derive structural booleans from canonical, geometry-free records."""

    predicted = tuple(predicted_prefixes)
    target = tuple(target_prefixes)
    if len(predicted) != len(target) or not predicted:
        raise GraphEncoderError(
            "invalid_stage6_structural_record", "prefix collections differ"
        )
    evidence = []
    from prototype.node_grammar import V5_NODE_GRAMMAR, validate_complete_node_sequence
    for observed, expected in zip(predicted, target):
        if set(observed) != STRUCTURAL_RECORD_FIELDS or set(expected) != STRUCTURAL_RECORD_FIELDS:
            raise GraphEncoderError(
                "invalid_stage6_structural_record", "record fields differ"
            )
        if observed["k"] != expected["k"]:
            raise GraphEncoderError(
                "invalid_stage6_structural_record", "prefix indices differ"
            )
        for record in (observed, expected):
            if (
                not isinstance(record["grammar_valid_complete"], bool)
                or not isinstance(record["canonicalization_succeeded"], bool)
                or not isinstance(record["prefix_construction_succeeded"], bool)
                or isinstance(record["requested_node_count"], bool)
                or not isinstance(record["requested_node_count"], int)
                or record["requested_node_count"] < 1
                or (
                    record["prefix_construction_succeeded"]
                    and record["requested_node_count"] != len(record["node_type_ids"])
                )
            ):
                raise GraphEncoderError(
                    "invalid_stage6_structural_record", "status must be Boolean"
                )
            for name in ("node_type_ids", "operation_type_ids"):
                if any(isinstance(item, bool) or not isinstance(item, int)
                       for item in record[name]):
                    raise GraphEncoderError(
                        "invalid_stage6_structural_record", name + " differs"
                    )
            for name in ("canonical_edges", "depends_on_edges", "ownership_attachment_edges"):
                normalized = tuple(tuple(item) for item in record[name])
                if any(len(item) != 3 or any(isinstance(value, bool) or not isinstance(value, int)
                                             for value in item) for item in normalized):
                    raise GraphEncoderError(
                        "invalid_stage6_structural_record", name + " differs"
                    )
                if normalized != tuple(sorted(set(normalized))):
                    raise GraphEncoderError(
                        "invalid_stage6_structural_record", name + " is not canonical"
                    )
            try:
                validate_complete_node_sequence(
                    tuple(record["node_type_ids"]), record["requested_node_count"],
                    V5_NODE_GRAMMAR,
                )
                computed_grammar = True
            except Exception:
                computed_grammar = False
            if record["grammar_valid_complete"] is not computed_grammar:
                raise GraphEncoderError(
                    "invalid_stage6_structural_record", "grammar evidence differs"
                )
        if (
            expected["grammar_valid_complete"] is not True
            or expected["canonicalization_succeeded"] is not True
            or expected["prefix_construction_succeeded"] is not True
        ):
            raise GraphEncoderError(
                "invalid_stage6_structural_target", "target prefix is invalid"
            )
        operation_types_exact = sorted(observed["operation_type_ids"]) == sorted(
            expected["operation_type_ids"]
        )
        chronological_exact = observed["operation_type_ids"] == expected["operation_type_ids"]
        canonical_exact = (
            observed["canonicalization_succeeded"]
            and observed["node_type_ids"] == expected["node_type_ids"]
            and observed["canonical_edges"] == expected["canonical_edges"]
        )
        evidence.append({
            "k": observed["k"],
            "grammar_complete": (
                observed["grammar_valid_complete"]
                and observed["canonicalization_succeeded"]
                and observed["prefix_construction_succeeded"]
            ),
            "operation_types_exact": operation_types_exact,
            "chronological_order_exact": chronological_exact,
            "canonical_graph_exact": canonical_exact,
            "depends_on_exact": observed["depends_on_edges"] == expected["depends_on_edges"],
            "ownership_attachments_exact": (
                observed["ownership_attachment_edges"]
                == expected["ownership_attachment_edges"]
            ),
        })
    return evidence


def score_family_record(record):
    required = {
        "family_id", "template", "cohort", "arm", "seed", "condition",
        "representation_variant_id", "batch_identity", "memory_source_family_id",
        "target_operation_count", "autonomous", "target_joined_after_generation",
        "predicted_operation_group_count",
        "raw_prediction_preserved", "constrained_prediction_preserved",
        "predicted_structural_prefixes", "target_structural_prefixes", "secondary",
    }
    if set(record) != required:
        raise GraphEncoderError("invalid_stage6_record", "family fields differ")
    if (
        not isinstance(record["family_id"], str)
        or not record["family_id"]
        or record["template"] not in TEMPLATES
        or record["cohort"] not in COHORTS
        or record["arm"] not in ARMS
        or record["seed"] not in FULL_SEEDS
        or record["condition"] not in CONDITIONS
        or not isinstance(record["representation_variant_id"], str)
        or not record["representation_variant_id"]
        or not isinstance(record["batch_identity"], str)
        or not record["batch_identity"]
        or record["autonomous"] is not True
        or record["target_joined_after_generation"] is not True
        or record["raw_prediction_preserved"] is not True
        or record["constrained_prediction_preserved"] is not True
    ):
        raise GraphEncoderError("invalid_stage6_record", "family identity differs")
    source = record["memory_source_family_id"]
    if (
        (record["condition"] == "P_true" and source != record["family_id"])
        or (record["condition"] == "P_shuffle" and (
            not isinstance(source, str) or not source or source == record["family_id"]
        ))
        or (record["condition"] == "P_mean" and (
            not isinstance(source, str) or not source.startswith("batch_mean:")
        ))
    ):
        raise GraphEncoderError("invalid_stage6_record", "memory assignment differs")
    evidence = structural_prefix_evidence(
        record["predicted_structural_prefixes"], record["target_structural_prefixes"]
    )
    score = score_structural_prefix(
        record["target_operation_count"], evidence,
        record["predicted_operation_group_count"],
    )
    secondary = record["secondary"]
    required_secondary = {
        "exact_complete_node_sequence", "exact_operation_type_sequence",
        "exact_graph", "depends_on_true_positive", "depends_on_predicted",
        "depends_on_target", "depends_on_exact", "reference_attachment_exact",
        "strict_conversion", "analytic_validity", "geometry_metrics",
        "operation_magnitude_metrics",
    }
    if set(secondary) != required_secondary:
        raise GraphEncoderError("invalid_stage6_record", "secondary fields differ")
    for name in (
        "exact_complete_node_sequence", "exact_operation_type_sequence",
        "exact_graph", "depends_on_exact", "reference_attachment_exact",
        "strict_conversion", "analytic_validity",
    ):
        if not isinstance(secondary[name], bool):
            raise GraphEncoderError("invalid_stage6_record", name + " must be bool")
    for name in ("depends_on_true_positive", "depends_on_predicted", "depends_on_target"):
        if isinstance(secondary[name], bool) or not isinstance(secondary[name], int) or secondary[name] < 0:
            raise GraphEncoderError("invalid_stage6_record", name + " must be nonnegative int")
    if secondary["depends_on_true_positive"] > min(
        secondary["depends_on_predicted"], secondary["depends_on_target"]
    ):
        raise GraphEncoderError("invalid_stage6_record", "depends_on counts differ")
    return {
        "version": SCORE_VERSION,
        "family_id": record["family_id"],
        "template": record["template"],
        "cohort": record["cohort"],
        "arm": record["arm"],
        "seed": record["seed"],
        "condition": record["condition"],
        "representation_variant_id": record["representation_variant_id"],
        "batch_identity": record["batch_identity"],
        "memory_source_family_id": record["memory_source_family_id"],
        "autonomous": record["autonomous"],
        "target_joined_after_generation": record["target_joined_after_generation"],
        "raw_prediction_preserved": record["raw_prediction_preserved"],
        "constrained_prediction_preserved": record["constrained_prediction_preserved"],
        "predicted_operation_group_count": record["predicted_operation_group_count"],
        "predicted_structural_prefixes": record["predicted_structural_prefixes"],
        "target_structural_prefixes": record["target_structural_prefixes"],
        "score": score,
        "secondary": secondary,
    }


def _mean(values):
    values = tuple(values)
    if not values:
        raise GraphEncoderError("invalid_stage6_aggregation", "empty mean")
    return sum(values) / float(len(values))


def paired_seed_aggregation(scored, retained_seeds):
    """Pair families within seed, then average seed effects without pooling."""

    seeds = tuple(retained_seeds)
    if seeds not in (FULL_SEEDS, FALLBACK_SEEDS):
        raise GraphEncoderError("invalid_stage6_seeds", "retained seeds differ")
    development = [
        row for row in scored
        if row["cohort"] == "development" and row["condition"] == "P_true"
    ]
    effects = []
    template_effects = defaultdict(list)
    for seed in seeds:
        by_arm = {
            arm: {row["family_id"]: row for row in development
                  if row["seed"] == seed and row["arm"] == arm}
            for arm in ARMS
        }
        if set(by_arm["flat"]) != set(by_arm["typed_graph"]):
            raise GraphEncoderError("unpaired_stage6_families", str(seed))
        if len(by_arm["flat"]) != DEVELOPMENT_FAMILY_COUNT:
            raise GraphEncoderError("invalid_stage6_family_count", str(seed))
        paired = []
        by_template = defaultdict(list)
        for family_id in sorted(by_arm["flat"]):
            flat = by_arm["flat"][family_id]
            graph = by_arm["typed_graph"][family_id]
            if flat["template"] != graph["template"]:
                raise GraphEncoderError("unpaired_stage6_families", family_id)
            delta = (
                graph["score"]["normalized_structural_prefix"]
                - flat["score"]["normalized_structural_prefix"]
            )
            paired.append(delta)
            by_template[flat["template"]].append(delta)
        seed_effect = _mean(paired)
        effects.append({
            "seed": seed,
            "paired_family_count": len(paired),
            "graph_minus_flat_mean": seed_effect,
        })
        for template in TEMPLATES:
            if by_template[template]:
                template_effects[template].append(_mean(by_template[template]))
    mean_effect = _mean(item["graph_minus_flat_mean"] for item in effects)
    fallback_nonnegative = (
        seeds != FALLBACK_SEEDS
        or all(item["graph_minus_flat_mean"] >= 0.0 for item in effects)
    )
    return {
        "version": AGGREGATION_VERSION,
        "retained_seeds": list(seeds),
        "seed_effects": effects,
        "retained_seed_mean_effect": mean_effect,
        "primary_threshold": PRIMARY_THRESHOLD,
        "threshold_pass": mean_effect >= PRIMARY_THRESHOLD,
        "fallback_seed_nonnegative_pass": fallback_nonnegative,
        "template_effects": {
            name: _mean(values) for name, values in sorted(template_effects.items())
        },
        "seeds_pooled_as_independent_families": False,
    }


def structure_memory_gate_for_cohort(scored, retained_seeds, cohort):
    if cohort not in COHORTS:
        raise GraphEncoderError("invalid_stage6_partition", str(cohort))
    rows = [row for row in scored if row["seed"] in retained_seeds]
    output = {}
    for arm in ARMS:
        values = {}
        for condition in CONDITIONS:
            selected = [
                row["score"]["normalized_structural_prefix"] for row in rows
                if row["cohort"] == cohort and row["arm"] == arm
                and row["condition"] == condition
            ]
            values[condition] = _mean(selected)
        true = values["P_true"]
        shuffle_ratio = None if true <= 0.0 else values["P_shuffle"] / true
        mean_ratio = None if true <= 0.0 else values["P_mean"] / true
        passed = (
            true > 0.0 and shuffle_ratio <= MEMORY_RATIO_THRESHOLD
            and mean_ratio <= MEMORY_RATIO_THRESHOLD
        )
        output[arm] = {
            **values,
            "R_shuffle": shuffle_ratio,
            "R_mean": mean_ratio,
            "threshold": MEMORY_RATIO_THRESHOLD,
            "pass": passed,
            "geometry_failure_can_zero_score": False,
        }
    return output


def structure_memory_gate(scored, retained_seeds):
    output = {
        cohort: structure_memory_gate_for_cohort(scored, retained_seeds, cohort)
        for cohort in COHORTS
    }
    return {
        "version": MEMORY_VERSION,
        "cohorts": output,
        "overall_pass": all(
            output[cohort][arm]["pass"] for cohort in COHORTS for arm in ARMS
        ),
    }


def secondary_evidence(scored, retained_seeds):
    """Aggregate required structural secondary evidence without gate leakage."""

    rows = [
        row for row in scored
        if row["seed"] in retained_seeds and row["condition"] == "P_true"
    ]
    groups = {}
    for cohort in COHORTS:
        for arm in ARMS:
            for seed in retained_seeds:
                selected = [
                    row for row in rows if row["cohort"] == cohort
                    and row["arm"] == arm and row["seed"] == seed
                ]
                by_template = {}
                for template in TEMPLATES:
                    template_rows = [row for row in selected if row["template"] == template]
                    if template_rows:
                        by_template[template] = _secondary_group(template_rows)
                groups["{}:{}:{}".format(cohort, arm, seed)] = {
                    **_secondary_group(selected),
                    "by_template": by_template,
                }
    return {
        "groups": groups,
        "train_structural_ceilings_reported": True,
        "geometry_metrics_report_only_per_family": True,
        "operation_magnitude_metrics_report_only_per_family": True,
        "strict_conversion_report_only": True,
        "analytic_validity_report_only": True,
    }


def _secondary_group(rows):
    if not rows:
        raise GraphEncoderError("invalid_stage6_aggregation", "empty secondary group")
    secondary = [row["secondary"] for row in rows]
    tp = sum(row["depends_on_true_positive"] for row in secondary)
    predicted = sum(row["depends_on_predicted"] for row in secondary)
    target = sum(row["depends_on_target"] for row in secondary)
    return {
        "family_count": len(rows),
        "mean_structural_prefix": _mean(
            row["score"]["normalized_structural_prefix"] for row in rows
        ),
        "exact_complete_node_sequence_rate": _mean(
            float(row["exact_complete_node_sequence"]) for row in secondary
        ),
        "exact_operation_type_sequence_rate": _mean(
            float(row["exact_operation_type_sequence"]) for row in secondary
        ),
        "exact_graph_rate": _mean(float(row["exact_graph"]) for row in secondary),
        "under_generation_rate": _mean(
            float(row["score"]["under_generation"]) for row in rows
        ),
        "exact_generation_rate": _mean(
            float(row["score"]["exact_generation"]) for row in rows
        ),
        "over_generation_rate": _mean(
            float(row["score"]["over_generation"]) for row in rows
        ),
        "depends_on_precision": None if predicted == 0 else tp / float(predicted),
        "depends_on_recall": None if target == 0 else tp / float(target),
        "depends_on_exact_rate": _mean(
            float(row["depends_on_exact"]) for row in secondary
        ),
        "reference_attachment_exact_rate": _mean(
            float(row["reference_attachment_exact"]) for row in secondary
        ),
        "strict_conversion_rate_report_only": _mean(
            float(row["strict_conversion"]) for row in secondary
        ),
        "analytic_validity_rate_report_only": _mean(
            float(row["analytic_validity"]) for row in secondary
        ),
        "first_failure_counts": dict(sorted(Counter(
            row["score"]["first_failure_stage"] for row in rows
        ).items())),
    }


def interpretation_category(primary, validity_gates_pass):
    """Apply the single frozen directional ADR-0014 interpretation rule."""

    if not isinstance(validity_gates_pass, bool):
        raise GraphEncoderError("invalid_stage6_interpretation", "validity differs")
    if not validity_gates_pass:
        return "inconclusive"
    effect = _finite_number(
        primary.get("retained_seed_mean_effect"), "retained_seed_mean_effect"
    )
    if effect >= PRIMARY_THRESHOLD:
        return (
            "graph_supported"
            if primary.get("fallback_seed_nonnegative_pass") is True
            else "inconclusive"
        )
    if effect <= -PRIMARY_THRESHOLD:
        return "flat_supported"
    subgroup_effects = [
        _finite_number(row.get("graph_minus_flat_mean"), "seed effect")
        for row in primary.get("seed_effects", ())
    ] + [
        _finite_number(value, "template effect")
        for value in primary.get("template_effects", {}).values()
    ]
    if any(abs(value) >= PRIMARY_THRESHOLD for value in subgroup_effects):
        return "mixed"
    return "null"


def validate_governance_evidence(payload, *, require_producer_artifact=False,
                                 expected_commit=None):
    """Reject infrastructure evidence failures instead of scoring them."""

    source = payload.get("source_evidence")
    inputs = payload.get("input_evidence")
    bundle = payload.get("checkpoint_bundle_reference")
    execution = payload.get("execution_evidence")
    if (
        not isinstance(source, dict)
        or source.get("verification_status") != "pass"
        or source.get("source_clean") is not True
        or source.get("detached_head") is not True
        or not isinstance(source.get("source_commit"), str)
        or len(source["source_commit"]) != 40
        or not isinstance(source.get("source_tree_sha256"), str)
        or len(source["source_tree_sha256"]) != 64
        or (expected_commit is not None and source["source_commit"] != expected_commit)
    ):
        raise GraphEncoderError("invalid_stage6_provenance", "source evidence differs")
    if (
        not isinstance(inputs, dict)
        or inputs.get("verification_status") != "pass"
        or inputs.get("input_policy_version") is None
        or any(
            not isinstance(inputs.get(cohort), dict)
            or inputs[cohort].get("verification_status") != "pass"
            or not isinstance(inputs[cohort].get("index_identity_sha256"), str)
            or len(inputs[cohort]["index_identity_sha256"]) != 64
            or inputs[cohort].get("expected_allowlist_sha256")
            != inputs[cohort].get("observed_allowlist_sha256")
            or inputs[cohort].get("expected_payload_digests_sha256")
            != inputs[cohort].get("observed_payload_digests_sha256")
            or inputs[cohort].get("expected_payload_sha256")
            != inputs[cohort].get("observed_payload_sha256")
            for cohort in ("train", "development")
        )
    ):
        raise GraphEncoderError("invalid_stage6_input_evidence", "input evidence differs")
    if (
        not isinstance(bundle, dict)
        or bundle.get("verification_status") != "pass"
        or bundle.get("source_evidence") != source
        or bundle.get("input_evidence") != inputs
        or not isinstance(bundle.get("bundle_sha256"), str)
        or len(bundle["bundle_sha256"]) != 64
    ):
        raise GraphEncoderError("invalid_stage6_checkpoint_bundle", "reference differs")
    if (
        not isinstance(execution, dict)
        or set(execution) != {
            "selected_execution_device", "runtime_identity",
            "timing_hardware_identity", "timing_version",
            "device_selected_only_by_timing", "cuda_peak_memory_bytes",
            "verification_status",
        }
        or execution.get("verification_status") != "pass"
        or execution.get("device_selected_only_by_timing") is not True
    ):
        raise GraphEncoderError(
            "invalid_stage6_execution_evidence", "execution evidence differs"
        )
    from .stage6_timing import (
        TIMING_VERSION, validate_runtime_identity, validate_timing_evidence,
    )
    if execution.get("timing_version") != TIMING_VERSION:
        raise GraphEncoderError(
            "invalid_stage6_execution_evidence", "timing version differs"
        )
    selection = validate_timing_evidence(
        payload.get("timing_fallback", {}).get("evidence", {}),
        required_device=execution.get("selected_execution_device"),
    )
    device = execution.get("selected_execution_device")
    validate_runtime_identity(execution.get("runtime_identity"), device)
    from .stage6_device import timing_hardware_identity
    timing_record = payload["timing_fallback"]["evidence"]
    peak_memory = execution.get("cuda_peak_memory_bytes")
    if execution.get("timing_hardware_identity") != selection[
        "timing_hardware_identity"
    ] or execution.get("timing_hardware_identity") != timing_hardware_identity(
        execution["runtime_identity"]
    ) or (device == "cpu" and peak_memory is not None) or (
        device == "cuda:0" and (
            isinstance(peak_memory, bool)
            or not isinstance(peak_memory, int)
            or peak_memory <= 0
        )
    ) or timing_record.get("source_identity") != {
        "source_commit": source["source_commit"],
        "source_tree_sha256": source["source_tree_sha256"],
    } or timing_record.get("train_input_identity") != {
        "index_identity_sha256": inputs["train"]["index_identity_sha256"],
        "payload_digests_sha256": inputs["train"][
            "observed_payload_digests_sha256"
        ],
    }:
        raise GraphEncoderError(
            "invalid_stage6_execution_evidence", "hardware identity differs"
        )
    if require_producer_artifact:
        producer = payload.get("producer_artifact_evidence")
        if (
            not isinstance(producer, dict)
            or producer.get("verification_status") != "pass"
            or producer.get("checkpoint_bundle_sha256") != bundle["bundle_sha256"]
            or not isinstance(producer.get("producer_artifact_sha256"), str)
            or len(producer["producer_artifact_sha256"]) != 64
        ):
            raise GraphEncoderError(
                "invalid_stage6_producer_artifact", "producer evidence differs"
            )
    return True


def _validate_execution_metadata(payload):
    if payload.get("schema_version") != INPUT_VERSION:
        raise GraphEncoderError("invalid_stage6_input", "schema differs")
    seeds = tuple(payload.get("retained_seeds", ()))
    timing = payload.get("timing_fallback", {})
    if seeds == FULL_SEEDS:
        if timing.get("invoked") is not False:
            raise GraphEncoderError("invalid_stage6_timing", "unexpected fallback")
    elif seeds == FALLBACK_SEEDS:
        if (
            timing.get("invoked") is not True
            or timing.get("prospective") is not True
            or timing.get("observed_results_used") is not False
            or not timing.get("reason")
        ):
            raise GraphEncoderError("invalid_stage6_timing", "fallback not prospective")
    else:
        raise GraphEncoderError("invalid_stage6_seeds", "seed set differs")
    partition = payload.get("partition", {})
    if (
        partition.get("training_name") != "operation_template_train"
        or partition.get("training_family_count") != TRAIN_FAMILY_COUNT
        or partition.get("development_name") != "operation_template_development"
        or partition.get("development_family_count") != DEVELOPMENT_FAMILY_COUNT
        or partition.get("rr_accessed") is not False
        or partition.get("er_accessed") is not False
    ):
        raise GraphEncoderError("invalid_stage6_partition", "partition differs")
    runs = payload.get("training_runs", [])
    selected_device = payload.get("execution_evidence", {}).get(
        "selected_execution_device"
    )
    if selected_device not in ("cpu", "cuda:0"):
        raise GraphEncoderError(
            "invalid_stage6_execution_evidence", "selected device differs"
        )
    expected = {(arm, seed) for arm in ARMS for seed in seeds}
    observed = {(row.get("arm"), row.get("seed")) for row in runs}
    if observed != expected or len(runs) != len(expected):
        raise GraphEncoderError("invalid_stage6_training", "run matrix differs")
    capacities = {}
    for row in runs:
        if (
            row.get("fresh_initialization") is not True
            or row.get("epochs") != EPOCHS
            or row.get("batch_size") != BATCH_SIZE
            or row.get("optimizer") != "AdamW"
            or row.get("learning_rate") != LEARNING_RATE
            or row.get("weight_decay") != WEIGHT_DECAY
            or row.get("gradient_clip_norm") != CLIP_NORM
            or row.get("checkpoint_epoch") != EPOCHS
            or row.get("early_stopping") is not False
            or row.get("development_used_for_selection") is not False
            or row.get("warm_start") is not False
            or row.get("training_family_count") != TRAIN_FAMILY_COUNT
            or row.get("execution_device") != selected_device
        ):
            raise GraphEncoderError("invalid_stage6_training", "frozen run differs")
        capacities[row["arm"]] = row.get("trainable_parameter_count")
        for name in ("final_loss", "plateau_relative_improvement"):
            _finite_number(row.get(name), name)
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0
           for value in capacities.values()):
        raise GraphEncoderError("invalid_stage6_capacity", "parameter count differs")
    return seeds


def validate_scored_alignment(scored, seeds):
    """Recheck the complete family/arm/seed/condition alignment matrix."""

    expected_keys = {
        (cohort, arm, seed, condition)
        for cohort in COHORTS for arm in ARMS for seed in seeds
        for condition in CONDITIONS
    }
    grouped = defaultdict(list)
    for row in scored:
        if row["seed"] not in seeds:
            raise GraphEncoderError("invalid_stage6_seeds", "extra record seed")
        grouped[(row["cohort"], row["arm"], row["seed"], row["condition"])].append(row)
    if set(grouped) != expected_keys:
        raise GraphEncoderError("invalid_stage6_record_matrix", "groups differ")
    family_sets = {}
    for key, rows in grouped.items():
        expected = TRAIN_FAMILY_COUNT if key[0] == "train" else DEVELOPMENT_FAMILY_COUNT
        if len(rows) != expected or len({row["family_id"] for row in rows}) != expected:
            raise GraphEncoderError("invalid_stage6_family_count", str(key))
        observed_families = frozenset(row["family_id"] for row in rows)
        prior = family_sets.setdefault(key[0], observed_families)
        if observed_families != prior:
            raise GraphEncoderError("invalid_stage6_alignment", str(key))
        batches = defaultdict(set)
        for row in rows:
            batches[row["batch_identity"]].add(row["family_id"])
        for row in rows:
            if (
                row["condition"] == "P_shuffle"
                and row["memory_source_family_id"]
                not in batches[row["batch_identity"]]
            ):
                raise GraphEncoderError(
                    "invalid_stage6_alignment", "shuffle donor outside batch"
                )
            if (
                row["condition"] == "P_mean"
                and row["memory_source_family_id"]
                != "batch_mean:" + row["batch_identity"]
            ):
                raise GraphEncoderError(
                    "invalid_stage6_alignment", "mean identity differs"
                )
    target_identities = {}
    for row in scored:
        identity = (
            row["representation_variant_id"], row["template"],
            row["batch_identity"],
            row["score"]["target_operation_count"],
            _canonical(row["target_structural_prefixes"]),
        )
        key = (row["cohort"], row["family_id"])
        previous = target_identities.setdefault(key, identity)
        if identity != previous:
            raise GraphEncoderError("invalid_stage6_alignment", str(key))
    return True


def summarize_execution(payload):
    seeds = _validate_execution_metadata(payload)
    validate_governance_evidence(payload)
    scored = [score_family_record(row) for row in payload.get("family_records", [])]
    validate_scored_alignment(scored, seeds)
    primary = paired_seed_aggregation(scored, seeds)
    memory = structure_memory_gate(scored, seeds)
    secondary = secondary_evidence(scored, seeds)
    runs = payload["training_runs"]
    capacity_pass = len({
        row["trainable_parameter_count"] for row in runs if row["arm"] == "flat"
    }) == 1 and len({
        row["trainable_parameter_count"] for row in runs if row["arm"] == "typed_graph"
    }) == 1 and all(row.get("capacity_parity_pass") is True for row in runs)
    optimization_pass = all(row.get("optimization_reliable") is True for row in runs)
    provenance_pass = payload.get("provenance_valid") is True
    artifact_inputs_valid = payload.get("artifact_inputs_valid") is True
    validity = all((capacity_pass, optimization_pass, provenance_pass,
                    artifact_inputs_valid, memory["overall_pass"]))
    category = interpretation_category(primary, validity)
    stage_counts = Counter(
        row["score"]["first_failure_stage"] for row in scored
        if row["cohort"] == "development" and row["condition"] == "P_true"
    )
    return scored, {
        "protocol_version": PROTOCOL_VERSION,
        "primary": primary,
        "memory": memory,
        "secondary_evidence": secondary,
        "capacity_pass": capacity_pass,
        "optimization_pass": optimization_pass,
        "provenance_pass": provenance_pass,
        "artifact_inputs_valid": artifact_inputs_valid,
        "validity_gates_pass": validity,
        "interpretation_category": category,
        "development_first_failure_counts": dict(sorted(stage_counts.items())),
        "geometry_and_operation_magnitude_report_only": True,
        "strict_conversion_and_analytic_validity_report_only": True,
        "claim_boundary": "structural_reconstruction_and_generalization_only",
        **AUTHORITY,
    }


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False)


def _write(path, raw):
    Path(path).write_bytes(raw)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def create_artifact(payload, output_dir, expected_commit, job_id):
    root = Path(output_dir)
    if root.exists() or root.is_symlink() or not root.parent.is_dir():
        raise GraphEncoderError("unsafe_stage6_output", str(root))
    staging = root.with_name(root.name + ".incomplete-" + str(job_id))
    if staging.exists() or staging.is_symlink():
        raise GraphEncoderError("unsafe_stage6_output", str(staging))
    scored, summary = summarize_execution(payload)
    validate_governance_evidence(
        payload, require_producer_artifact=True, expected_commit=expected_commit
    )
    staging.mkdir()
    try:
        resolved = {
            "artifact_version": ARTIFACT_VERSION,
            "protocol": protocol_config(),
            "source_commit": expected_commit,
            "job_id": str(job_id),
            "retained_seeds": payload["retained_seeds"],
            "partition": payload["partition"],
            "timing_fallback": payload["timing_fallback"],
            "training_runs": payload["training_runs"],
            "provenance_valid": payload["provenance_valid"],
            "artifact_inputs_valid": payload["artifact_inputs_valid"],
            "source_evidence": payload.get("source_evidence"),
            "input_evidence": payload.get("input_evidence"),
            "checkpoint_bundle_reference": payload.get("checkpoint_bundle_reference"),
            "producer_artifact_evidence": payload.get("producer_artifact_evidence"),
            "execution_evidence": payload.get("execution_evidence"),
            **AUTHORITY,
        }
        _write(staging / "resolved_config.json", (_canonical(resolved) + "\n").encode())
        _write(staging / "family_metrics.jsonl", (
            "".join(_canonical(row) + "\n" for row in scored)
        ).encode())
        summary.update({
            "artifact_version": ARTIFACT_VERSION,
            "source_commit": expected_commit,
            "job_id": str(job_id),
            "family_metric_record_count": len(scored),
        })
        _write(staging / "summary.json", (_canonical(summary) + "\n").encode())
        names = ("resolved_config.json", "family_metrics.jsonl", "summary.json")
        manifest = {
            "schema_version": ARTIFACT_VERSION,
            "artifacts": [{
                "path": name, "byte_size": (staging / name).stat().st_size,
                "sha256": _sha(staging / name),
            } for name in names],
        }
        _write(staging / "artifact_manifest.json", (_canonical(manifest) + "\n").encode())
        sum_names = ("artifact_manifest.json",) + names
        _write(staging / "SHA256SUMS", "".join(
            "{}  {}\n".format(_sha(staging / name), name) for name in sum_names
        ).encode())
        os.replace(str(staging), str(root))
    except Exception:
        raise
    return verify_artifact(root, expected_commit=expected_commit, expected_job_id=str(job_id))


def verify_artifact(path, *, expected_commit, expected_job_id):
    root = Path(path)
    expected = {
        "resolved_config.json", "family_metrics.jsonl", "summary.json",
        "artifact_manifest.json", "SHA256SUMS",
    }
    if (
        not root.is_dir() or root.is_symlink()
        or {p.name for p in root.iterdir()} != expected
        or any(item.is_symlink() or not item.is_file() for item in root.iterdir())
    ):
        raise GraphEncoderError("invalid_stage6_artifact", "file set differs")
    for name in ("resolved_config.json", "summary.json", "artifact_manifest.json"):
        raw = (root / name).read_text(encoding="utf-8")
        try:
            value = json.loads(
                raw,
                parse_constant=lambda constant: (_ for _ in ()).throw(
                    ValueError("nonfinite " + constant)
                ),
            )
        except (ValueError, json.JSONDecodeError) as exc:
            raise GraphEncoderError("invalid_stage6_artifact", "malformed JSON") from exc
        if raw != _canonical(value) + "\n":
            raise GraphEncoderError("invalid_stage6_artifact", name + " not canonical")
    manifest = json.loads((root / "artifact_manifest.json").read_text())
    if manifest.get("schema_version") != ARTIFACT_VERSION:
        raise GraphEncoderError("invalid_stage6_artifact", "schema differs")
    if tuple(row.get("path") for row in manifest.get("artifacts", [])) != (
        "resolved_config.json", "family_metrics.jsonl", "summary.json"
    ):
        raise GraphEncoderError("invalid_stage6_artifact", "manifest coverage differs")
    for row in manifest.get("artifacts", []):
        target = root / row.get("path", "")
        if not target.is_file() or target.is_symlink() or target.stat().st_size != row.get("byte_size") or _sha(target) != row.get("sha256"):
            raise GraphEncoderError("invalid_stage6_artifact", "manifest mismatch")
    try:
        lines = tuple(
            tuple(line.split("  ", 1))
            for line in (root / "SHA256SUMS").read_text().splitlines()
        )
    except (OSError, UnicodeError) as exc:
        raise GraphEncoderError("invalid_stage6_artifact", "malformed checksums") from exc
    if len(lines) != 4 or any(len(row) != 2 or len(row[0]) != 64 for row in lines):
        raise GraphEncoderError("invalid_stage6_artifact", "checksum count differs")
    observed_checksum_names = []
    for digest, name in lines:
        if name not in expected - {"SHA256SUMS"} or _sha(root / name) != digest:
            raise GraphEncoderError("invalid_stage6_artifact", "checksum differs")
        observed_checksum_names.append(name)
    if tuple(observed_checksum_names) != (
        "artifact_manifest.json", "resolved_config.json",
        "family_metrics.jsonl", "summary.json",
    ):
        raise GraphEncoderError("invalid_stage6_artifact", "checksum coverage differs")
    resolved = json.loads((root / "resolved_config.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    if (
        resolved.get("source_commit") != expected_commit
        or resolved.get("job_id") != expected_job_id
        or resolved.get("protocol") != protocol_config()
        or summary.get("source_commit") != expected_commit
        or summary.get("job_id") != expected_job_id
        or any(resolved.get(key) != value for key, value in AUTHORITY.items())
        or any(summary.get(key) != value for key, value in AUTHORITY.items())
    ):
        raise GraphEncoderError("invalid_stage6_artifact", "identity differs")
    validate_governance_evidence({
        "source_evidence": resolved.get("source_evidence"),
        "input_evidence": resolved.get("input_evidence"),
        "checkpoint_bundle_reference": resolved.get("checkpoint_bundle_reference"),
        "producer_artifact_evidence": resolved.get("producer_artifact_evidence"),
        "execution_evidence": resolved.get("execution_evidence"),
        "timing_fallback": resolved.get("timing_fallback"),
    }, require_producer_artifact=True, expected_commit=expected_commit)
    _validate_execution_metadata({
        "schema_version": INPUT_VERSION,
        "retained_seeds": resolved.get("retained_seeds"),
        "timing_fallback": resolved.get("timing_fallback"),
        "partition": resolved.get("partition"),
        "training_runs": resolved.get("training_runs"),
        "execution_evidence": resolved.get("execution_evidence"),
    })
    records = []
    raw = (root / "family_metrics.jsonl").read_text(encoding="utf-8")
    if not raw.endswith("\n"):
        raise GraphEncoderError("invalid_stage6_artifact", "JSONL final LF absent")
    for line in raw.splitlines():
        try:
            value = json.loads(
                line,
                parse_constant=lambda constant: (_ for _ in ()).throw(
                    ValueError("nonfinite " + constant)
                ),
            )
        except (ValueError, json.JSONDecodeError) as exc:
            raise GraphEncoderError("invalid_stage6_artifact", "malformed JSONL") from exc
        if line != _canonical(value):
            raise GraphEncoderError("invalid_stage6_artifact", "JSONL not canonical")
        records.append(value)
    if len(records) != summary.get("family_metric_record_count"):
        raise GraphEncoderError("invalid_stage6_artifact", "record count differs")
    for row in records:
        original = {
            key: row.get(key) for key in (
                "family_id", "template", "cohort", "arm", "seed", "condition",
                "representation_variant_id", "batch_identity", "memory_source_family_id",
                "autonomous", "target_joined_after_generation",
                "raw_prediction_preserved", "constrained_prediction_preserved",
                "predicted_operation_group_count",
                "predicted_structural_prefixes", "target_structural_prefixes",
                "secondary",
            )
        }
        original["target_operation_count"] = row.get("score", {}).get(
            "target_operation_count"
        )
        recomputed = score_family_record(original)
        if recomputed != row:
            raise GraphEncoderError("invalid_stage6_artifact", "score differs")
    seeds = tuple(resolved.get("retained_seeds", ()))
    validate_scored_alignment(records, seeds)
    primary = paired_seed_aggregation(records, seeds)
    memory = structure_memory_gate(records, seeds)
    secondary = secondary_evidence(records, seeds)
    runs = resolved.get("training_runs", [])
    capacity_pass = (
        len({row.get("trainable_parameter_count") for row in runs if row.get("arm") == "flat"}) == 1
        and len({row.get("trainable_parameter_count") for row in runs if row.get("arm") == "typed_graph"}) == 1
        and all(row.get("capacity_parity_pass") is True for row in runs)
    )
    optimization_pass = all(row.get("optimization_reliable") is True for row in runs)
    validity = all((capacity_pass, optimization_pass,
                    resolved.get("provenance_valid") is True,
                    resolved.get("artifact_inputs_valid") is True,
                    memory["overall_pass"]))
    category = interpretation_category(primary, validity)
    if (
        summary.get("primary") != primary
        or summary.get("memory") != memory
        or summary.get("secondary_evidence") != secondary
        or summary.get("capacity_pass") is not capacity_pass
        or summary.get("optimization_pass") is not optimization_pass
        or summary.get("provenance_pass") is not True
        or summary.get("artifact_inputs_valid") is not True
        or summary.get("validity_gates_pass") is not validity
        or summary.get("interpretation_category") != category
    ):
        raise GraphEncoderError("invalid_stage6_artifact", "summary recomputation differs")
    return {
        "verification_status": "pass",
        "artifact_version": ARTIFACT_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "source_commit": expected_commit,
        "job_id": expected_job_id,
        "family_metric_record_count": len(records),
        "interpretation_category": summary.get("interpretation_category"),
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--producer-artifact", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--job-id", default=os.environ.get("SLURM_JOB_ID"))
    args = parser.parse_args(argv)
    if not args.job_id or not str(args.job_id).isdigit():
        raise GraphEncoderError("invalid_stage6_job_id", "decimal job ID required")
    from .stage6_structure_only_producer import load_execution_record
    payload = load_execution_record(args.producer_artifact)
    result = create_artifact(payload, args.output_dir, args.expected_commit, args.job_id)
    print(_canonical({"event": "stage6_structure_only_completed", **result}), flush=True)


if __name__ == "__main__":
    main()
