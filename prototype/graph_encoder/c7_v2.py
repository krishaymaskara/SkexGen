"""Prospective GE1 C7-v2 train-only sufficiency and memory-use protocol."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import platform
import socket
import sys
import time

try:
    import torch
except ImportError:  # Pure contract and artifact tests remain torch-free.
    torch = None

from .errors import GraphEncoderError
from .metrics import (
    METRICS_SCHEMA_VERSION,
    PRIMARY_REPORTING_CONTRACT_VERSION,
    donor_template_agreement,
    intervention_ratios,
    parameter_count_record,
    peak_memory_record,
    receptive_field_record,
    validate_primary_report,
)
from .optimization_diagnostic import (
    FROZEN_FAMILY_IDS,
    FROZEN_FAMILY_IDS_SHA256,
    validate_slurm_job_id,
)
from .partitions import (
    AUTHORITATIVE_FILE_SHA256,
    C7_ACCESSIBLE_TEMPLATES,
    load_train,
    select_c7_sufficiency_subsets,
)
from .pilot import (
    C7_ARTIFACT_MANIFEST_VERSION,
    C7_CHECKPOINT_EPOCH,
    C7_GATE_VERSION,
    C7_PILOT_VERSION,
    _assert_matched_disjoint,
    _assert_model_collections_disjoint,
    _assert_model_states_equal,
    _atomic_write_bytes,
    _atomic_write_json,
    _atomic_write_jsonl,
    _canonical_json_text,
    _file_sha256,
    _finite_metric,
    _is_within,
    _memory_criterion,
    _read_canonical_jsonl,
    _regular_artifact_files,
    _safe_relative_path,
    _source_identity,
    _verify_source_unchanged,
)


C7_V2_PROTOCOL_VERSION = "GE1-C7-SUFFICIENCY-v2"
C7_V2_METRICS_VERSION = "GE1-C7-METRICS-v2"
C7_V2_ARTIFACT_VERSION = "GE1-C7-ARTIFACT-v2"
C7_V2_SEED = 2026
C7_V2_CHECKPOINT_EPOCH = 200
C7_V2_BATCH_SIZE = 8
C7_V2_MEMORY_RATIO_MAX = 0.80
C7_V2_ARMS = ("flat", "typed_graph")
C7_V2_SUBSETS = ("tiny", "scaled")
C7_V2_GATE_STATUSES = ("pass", "fail", "not_run")
C7_V2_CHECKPOINT_ROLE = "c7_v2_gate_only_not_full_comparison"
C7_V2_TINY_FAMILY_IDS_SHA256 = FROZEN_FAMILY_IDS_SHA256
C7_V2_SCALED_FAMILY_IDS_SHA256 = (
    "0888d517561ca24c99065465f8a2378e0e8e457f4bd20024e7f983956456f9ce"
)
C7_V2_REQUIRED_GATE_NAMES = tuple(
    "{}.{}".format(arm, gate)
    for arm in C7_V2_ARMS
    for gate in (
        "tiny_exact_sufficiency",
        "scaled_exact_sufficiency",
        "scaled_memory_use",
    )
)
C7_V2_PROTECTED_ACCESS_FIELDS = (
    "development_accessed",
    "systematic_rr_accessed",
    "test_er_accessed",
    "iid_accessed",
    "history_depth_accessed",
    "geometry_extrapolation_accessed",
)


@dataclass(frozen=True)
class C7V2Configuration:
    """Resolved prospective budget without mutating the frozen C6 config."""

    epochs: int = C7_V2_CHECKPOINT_EPOCH
    checkpoint_epoch: int = C7_V2_CHECKPOINT_EPOCH
    batch_size: int = C7_V2_BATCH_SIZE
    seed: int = C7_V2_SEED
    early_stopping: bool = False
    best_checkpoint_selection: bool = False
    outcome_dependent_extension: bool = False
    warm_start: bool = False
    resume_from_c7_v1: bool = False
    resume_from_optimization_diagnostic: bool = False
    reuse_diagnostic_checkpoints: bool = False

    def validate(self):
        if self != C7V2Configuration():
            raise GraphEncoderError(
                "invalid_c7_v2_configuration",
                "C7-v2 configuration differs from the accepted protocol",
            )
        return self

    def to_dict(self):
        self.validate()
        return {
            "protocol_version": C7_V2_PROTOCOL_VERSION,
            "epochs": self.epochs,
            "checkpoint_epoch": self.checkpoint_epoch,
            "checkpoint_selection": "fixed_epoch_200",
            "batch_size": self.batch_size,
            "seed": self.seed,
            "governing_coordinate": "epochs",
            "early_stopping": self.early_stopping,
            "best_checkpoint_selection": self.best_checkpoint_selection,
            "outcome_dependent_extension": self.outcome_dependent_extension,
            "warm_start": self.warm_start,
            "resume_from_c7_v1": self.resume_from_c7_v1,
            "resume_from_optimization_diagnostic": (
                self.resume_from_optimization_diagnostic
            ),
            "reuse_diagnostic_checkpoints": self.reuse_diagnostic_checkpoints,
        }


@dataclass
class C7V2AccessTracker:
    operation_template_manifest_accessed: object = False
    operation_template_train_payload_accessed: object = False
    tiny_train_payload_accessed: object = False
    scaled_train_payload_accessed: object = False

    def declarations(self, *, completed=False):
        result = {
            "operation_template_manifest_accessed": (
                self.operation_template_manifest_accessed
            ),
            "operation_template_train_payload_accessed": (
                self.operation_template_train_payload_accessed
            ),
            "tiny_train_payload_accessed": self.tiny_train_payload_accessed,
            "scaled_train_payload_accessed": self.scaled_train_payload_accessed,
            "c7_v2_execution_completed": bool(completed),
            "stage6_performed": False,
            "c8_or_later_performed": False,
            "decoder_repair_implemented_or_invoked": False,
        }
        result.update({name: False for name in C7_V2_PROTECTED_ACCESS_FIELDS})
        return result


def c7_v2_training_arithmetic(family_count):
    """Return the exact accepted epoch-200 exposure arithmetic."""

    if family_count not in (4, 32, 407):
        raise GraphEncoderError(
            "invalid_c7_v2_subset_size",
            "C7-v2 arithmetic requires 4, 32, or 407 families",
        )
    steps_per_epoch = int(math.ceil(family_count / float(C7_V2_BATCH_SIZE)))
    result = {
        "family_count": family_count,
        "epochs": C7_V2_CHECKPOINT_EPOCH,
        "batch_size": C7_V2_BATCH_SIZE,
        "steps_per_epoch": steps_per_epoch,
        "optimizer_steps": steps_per_epoch * C7_V2_CHECKPOINT_EPOCH,
        "training_example_presentations": family_count * C7_V2_CHECKPOINT_EPOCH,
    }
    if family_count == 407:
        result["stage6_authorized_by_c7_v2_required"] = True
        result["optimizer_steps_are_approximate"] = True
    return result


def validate_frozen_c7_v2_selection(selection):
    """Require the exact immutable C7-v1 cohorts, selected metadata-only."""

    if (
        tuple(selection.tiny_family_ids) != FROZEN_FAMILY_IDS
        or selection.tiny_family_ids_sha256 != C7_V2_TINY_FAMILY_IDS_SHA256
        or selection.scaled_family_ids_sha256
        != C7_V2_SCALED_FAMILY_IDS_SHA256
        or len(selection.scaled_family_ids) != 32
        or not set(FROZEN_FAMILY_IDS).issubset(selection.scaled_family_ids)
    ):
        raise GraphEncoderError(
            "c7_v2_family_identity_mismatch",
            "metadata selection differs from the immutable C7-v1 cohorts",
        )
    templates = dict(selection.selected_templates)
    for template in C7_ACCESSIBLE_TEMPLATES:
        if sum(
            templates.get(family_id) == template
            for family_id in selection.scaled_family_ids
        ) != 8:
            raise GraphEncoderError(
                "c7_v2_family_identity_mismatch",
                "scaled cohort template composition differs",
            )
    record = selection.to_dict()
    record.update({
        "c7_v2_protocol_version": C7_V2_PROTOCOL_VERSION,
        "source_selection_identity_preserved": True,
        "formal_c7_v1_cohorts_reused_without_reselection": True,
    })
    return record


def validate_c7_v2_reloaded_checkpoint(resume_state, *, expected_commit, job_id):
    """Require the selected epoch-200 checkpoint and its execution provenance."""

    payload = getattr(resume_state, "payload", None)
    provenance = payload.get("provenance", {}) if isinstance(payload, dict) else {}
    if (
        getattr(resume_state, "completed_epoch", None)
        != C7_V2_CHECKPOINT_EPOCH
        or not isinstance(payload, dict)
        or payload.get("completed_epoch") != C7_V2_CHECKPOINT_EPOCH
        or payload.get("selected_checkpoint_epoch")
        != C7_V2_CHECKPOINT_EPOCH
        or payload.get("selected_experimental_checkpoint") is not True
        or provenance.get("git_commit") != expected_commit
        or provenance.get("slurm_job_id") != job_id
    ):
        raise GraphEncoderError(
            "c7_v2_checkpoint_reload_failure",
            "strict epoch-200 reload or provenance validation failed",
        )
    return resume_state


def exact_sufficiency_gate(
    *, arm, subset_identity, condition_metrics, templates_by_family,
    checkpoint_identity, epoch, access_flags, generation_condition="P_true"
):
    """Apply the unchanged exact, unrounded, per-family criteria at epoch 200."""

    _validate_arm_subset(arm, subset_identity)
    if generation_condition != "P_true":
        raise GraphEncoderError(
            "nonautonomous_c7_v2_gate_input",
            "only autonomous P_true generation may satisfy C7-v2",
        )
    if epoch != C7_V2_CHECKPOINT_EPOCH:
        raise GraphEncoderError(
            "wrong_c7_v2_checkpoint_epoch",
            "only strict epoch 200 may satisfy C7-v2",
        )
    if (
        not isinstance(condition_metrics, dict)
        or condition_metrics.get("condition") != "P_true"
        or condition_metrics.get("available") is not True
    ):
        raise GraphEncoderError(
            "malformed_c7_v2_metrics",
            "exact gate requires available P_true metrics",
        )
    family_values = condition_metrics.get("family_values")
    templates = dict(templates_by_family)
    expected_count = 4 if subset_identity == "tiny" else 32
    if (
        not isinstance(family_values, dict)
        or set(family_values) != set(templates)
        or len(family_values) != expected_count
    ):
        raise GraphEncoderError(
            "metric_target_alignment_failure",
            "C7-v2 metrics and physical-family target map differ",
        )

    fields = (
        "exact_node_sequence",
        "exact_graph",
        "strict_conversion",
        "complete_executable_validity",
    )
    criteria = {}
    failures = []
    for field in fields:
        passing = 0
        for family_id in sorted(family_values):
            observed = _finite_metric(family_values[family_id].get(field), field)
            if observed == 1.0:
                passing += 1
            else:
                failures.append((family_id, field, observed))
        criteria[field] = {
            "required_value": 1.0,
            "passing_family_count": passing,
            "physical_family_denominator": expected_count,
            "rounded_for_comparison": False,
        }
    dependency_ids = tuple(sorted(
        family_id for family_id, template in templates.items()
        if template in ("EE", "RE")
    ))
    dependency_passing = 0
    for family_id in dependency_ids:
        observed = _finite_metric(
            family_values[family_id].get("depends_on_exactness"),
            "depends_on_exactness",
        )
        if observed == 1.0:
            dependency_passing += 1
        else:
            failures.append((family_id, "depends_on_exactness", observed))
    criteria["depends_on_exactness_for_EE_RE"] = {
        "required_value": 1.0,
        "applicable_templates": ["EE", "RE"],
        "passing_family_count": dependency_passing,
        "physical_family_denominator": len(dependency_ids),
        "rounded_for_comparison": False,
    }
    by_family = {}
    for family_id, field, observed in failures:
        by_family.setdefault(family_id, []).append({
            "field": field,
            "required_value": 1.0,
            "observed_value": observed,
        })
    passed = not by_family
    return {
        "version": C7_V2_PROTOCOL_VERSION,
        "gate_name": "{}_exact_sufficiency".format(subset_identity),
        "status": "pass" if passed else "fail",
        "arm": arm,
        "seed": C7_V2_SEED,
        "subset_identity": subset_identity,
        "physical_family_count": expected_count,
        "generation_condition": "P_true",
        "teacher_forced_gate_eligible": False,
        "checkpoint_epoch": C7_V2_CHECKPOINT_EPOCH,
        "checkpoint_identity": str(checkpoint_identity),
        "checkpoint_role": C7_V2_CHECKPOINT_ROLE,
        "criteria": criteria,
        "failing_family_ids": sorted(by_family),
        "failed_fields_by_family": {
            family_id: by_family[family_id] for family_id in sorted(by_family)
        },
        "structured_reason": (
            "all_exact_criteria_satisfied"
            if passed else "one_or_more_families_failed_exact_criteria"
        ),
        "access": dict(access_flags),
    }


def not_run_gate(*, arm, gate_name, reason, access_flags):
    """Create a dependent scaled-gate record that cannot pass."""

    _validate_arm_subset(arm, "scaled")
    if gate_name not in ("scaled_exact_sufficiency", "scaled_memory_use"):
        raise GraphEncoderError("invalid_c7_v2_gate", "invalid not-run gate")
    if not isinstance(reason, str) or not reason:
        raise GraphEncoderError("invalid_c7_v2_gate", "not_run requires reason")
    return {
        "version": C7_V2_PROTOCOL_VERSION,
        "gate_name": gate_name,
        "status": "not_run",
        "arm": arm,
        "seed": C7_V2_SEED,
        "subset_identity": "scaled",
        "physical_family_count": 32,
        "checkpoint_epoch": None,
        "checkpoint_identity": None,
        "checkpoint_role": C7_V2_CHECKPOINT_ROLE,
        "criteria": {},
        "failing_family_ids": [],
        "failed_fields_by_family": {},
        "structured_reason": reason,
        "access": dict(access_flags),
    }


def memory_use_gate(
    *, arm, condition_metrics, autonomous_result, templates_by_family,
    checkpoint_identity, epoch, access_flags
):
    """Apply the unchanged exact unrounded scaled autonomous-memory gate."""

    _validate_arm_subset(arm, "scaled")
    if epoch != C7_V2_CHECKPOINT_EPOCH:
        raise GraphEncoderError(
            "wrong_c7_v2_checkpoint_epoch",
            "only strict epoch 200 may satisfy C7-v2",
        )
    metrics = tuple(condition_metrics)
    by_condition = {
        item.get("condition"): item for item in metrics if isinstance(item, dict)
    }
    if set(by_condition) != {"P_true", "P_shuffle", "P_mean"}:
        raise GraphEncoderError(
            "malformed_c7_v2_metrics", "memory gate requires three conditions"
        )
    family_ids = tuple(sorted(templates_by_family))
    if len(family_ids) != 32:
        raise GraphEncoderError(
            "invalid_c7_v2_subset_size", "memory gate requires 32 families"
        )
    for name in ("P_true", "P_shuffle", "P_mean"):
        record = by_condition[name]
        values = record.get("family_values")
        if record.get("available") is True and (
            not isinstance(values, dict) or set(values) != set(family_ids)
        ):
            raise GraphEncoderError(
                "metric_target_alignment_failure",
                "memory-condition families differ from scaled targets",
            )

    ratios = intervention_ratios(metrics)
    report_valid = True
    try:
        validate_primary_report({"intervention_ratios": ratios})
    except GraphEncoderError:
        report_valid = False
    criteria = {
        "P_true_positive": _memory_criterion(
            ratios.get("P_true"), comparison="greater_than", threshold=0.0
        ),
        "R_shuffle_at_most_0_80": _memory_criterion(
            ratios.get("R_shuffle"), comparison="less_than_or_equal",
            threshold=C7_V2_MEMORY_RATIO_MAX,
        ),
        "R_mean_at_most_0_80": _memory_criterion(
            ratios.get("R_mean"), comparison="less_than_or_equal",
            threshold=C7_V2_MEMORY_RATIO_MAX,
        ),
    }
    for criterion in criteria.values():
        criterion["physical_family_denominator"] = 32
        criterion["aggregation"] = "physical_family_macro"
    batch_membership = _validate_scaled_batch_membership(
        autonomous_result, family_ids
    )
    conditions = {item.condition: item for item in autonomous_result.conditions}
    if set(conditions) != {"P_true", "P_shuffle", "P_mean"}:
        raise GraphEncoderError(
            "malformed_c7_v2_autonomous_result",
            "autonomous result requires all memory conditions",
        )
    memory_evidence = {
        name: {
            "available": bool(conditions[name].available),
            "memory_altered": bool(conditions[name].memory_altered),
            "alteration_possible": bool(conditions[name].alteration_possible),
            "unavailable_reason": conditions[name].unavailable_reason,
        }
        for name in ("P_true", "P_shuffle", "P_mean")
    }
    donor = [
        donor_template_agreement(
            conditions[name].memory_assignments,
            templates_by_family,
            condition=name,
        )
        for name in ("P_true", "P_shuffle", "P_mean")
    ]
    passed = report_valid and all(item["passed"] for item in criteria.values())
    reasons = []
    if not report_valid:
        reasons.append("incomplete_primary_report")
    reasons.extend(name for name, item in criteria.items() if not item["passed"])
    return {
        "version": C7_V2_PROTOCOL_VERSION,
        "gate_name": "scaled_memory_use",
        "status": "pass" if passed else "fail",
        "arm": arm,
        "seed": C7_V2_SEED,
        "subset_identity": "scaled",
        "physical_family_count": 32,
        "checkpoint_epoch": C7_V2_CHECKPOINT_EPOCH,
        "checkpoint_identity": str(checkpoint_identity),
        "checkpoint_role": C7_V2_CHECKPOINT_ROLE,
        "criteria": criteria,
        "intervention_ratios": ratios,
        "primary_reporting_contract": PRIMARY_REPORTING_CONTRACT_VERSION,
        "donor_template_agreement": donor,
        "memory_evidence": memory_evidence,
        "deterministic_batch_membership": batch_membership,
        "failing_family_ids": [],
        "failed_fields_by_family": {},
        "structured_reason": (
            "all_memory_use_criteria_satisfied" if passed else ";".join(reasons)
        ),
        "access": dict(access_flags),
    }


def overall_c7_v2_decision(gates, *, infrastructure_failure=False):
    """Return only C7-v2-scoped authorization and next-path fields."""

    if infrastructure_failure:
        return {
            "version": C7_V2_PROTOCOL_VERSION,
            "c7_v2_overall_gate_pass": False,
            "stage6_authorized_by_c7_v2": False,
            "preauthorized_decoder_repair_path_next": False,
            "comparison_inconclusive": False,
            "c7_v2_execution_completed": False,
            "structured_reason": "infrastructure_failure",
        }
    by_name = dict(gates)
    if set(by_name) != set(C7_V2_REQUIRED_GATE_NAMES):
        raise GraphEncoderError(
            "incomplete_c7_v2_gate_set", "decision requires all six gates"
        )
    for name, gate in by_name.items():
        if not isinstance(gate, dict) or gate.get("status") not in C7_V2_GATE_STATUSES:
            raise GraphEncoderError(
                "malformed_c7_v2_gate", "{} has invalid status".format(name)
            )
    overall = all(
        by_name[name]["status"] == "pass" for name in C7_V2_REQUIRED_GATE_NAMES
    )
    exact_names = tuple(
        name for name in C7_V2_REQUIRED_GATE_NAMES
        if "exact_sufficiency" in name
    )
    repair = any(by_name[name]["status"] == "fail" for name in exact_names)
    inconclusive = (not overall) and (not repair)
    return {
        "version": C7_V2_PROTOCOL_VERSION,
        "c7_v2_overall_gate_pass": overall,
        "stage6_authorized_by_c7_v2": overall,
        "preauthorized_decoder_repair_path_next": repair,
        "comparison_inconclusive": inconclusive,
        "c7_v2_execution_completed": True,
        "structured_reason": (
            "all_six_c7_v2_arm_gates_passed"
            if overall else (
                "autonomous_exact_sufficiency_failure"
                if repair else "memory_gate_failure_or_dependent_gate_not_run"
            )
        ),
    }


def _validate_arm_subset(arm, subset_identity):
    if arm not in C7_V2_ARMS:
        raise GraphEncoderError("invalid_c7_v2_arm", "unknown arm")
    if subset_identity not in C7_V2_SUBSETS:
        raise GraphEncoderError("invalid_c7_v2_subset", "unknown subset")


def _validate_scaled_batch_membership(autonomous_result, family_ids):
    true = next(
        (item for item in autonomous_result.conditions if item.condition == "P_true"),
        None,
    )
    if true is None:
        raise GraphEncoderError(
            "malformed_c7_v2_autonomous_result", "P_true condition is absent"
        )
    membership = tuple(sorted(
        (tuple(families) for unused_identity, families in true.batch_membership),
        key=lambda item: item,
    ))
    expected = tuple(
        family_ids[start:start + C7_V2_BATCH_SIZE]
        for start in range(0, len(family_ids), C7_V2_BATCH_SIZE)
    )
    if membership != expected or len(membership) != 4 or any(
        len(item) != 8 for item in membership
    ):
        raise GraphEncoderError(
            "invalid_c7_v2_batch_membership",
            "scaled evaluation must be four sorted batches of eight",
        )
    return [list(item) for item in membership]


def c7_v2_metrics_envelope(
    *, arm, checkpoint_identity, condition_metrics, autonomous_result,
    templates_by_family, training_result, parameter_counts, exact_gate,
    memory_gate, access_flags, total_run_seconds, staging=None
):
    """Build the separately versioned scaled C7-v2 metrics record."""

    ratios = intervention_ratios(condition_metrics)
    record = {
        "schema_version": C7_V2_METRICS_VERSION,
        "protocol_version": C7_V2_PROTOCOL_VERSION,
        "source_condition_metrics_schema": METRICS_SCHEMA_VERSION,
        "primary_reporting_contract": PRIMARY_REPORTING_CONTRACT_VERSION,
        "arm": arm,
        "seed": C7_V2_SEED,
        "epoch": C7_V2_CHECKPOINT_EPOCH,
        "checkpoint_identity": str(checkpoint_identity),
        "checkpoint_role": C7_V2_CHECKPOINT_ROLE,
        "scientific_result": True,
        "formal_c7_v2_gate": True,
        "formal_c7_v1_result_changed": False,
        "optimization_diagnostic_result_changed": False,
        "training": _training_record_for_artifact(
            training_result, staging=staging
        ),
        "conditions": list(condition_metrics),
        "intervention_ratios": ratios,
        "donor_template_agreement": [
            donor_template_agreement(
                item.memory_assignments,
                templates_by_family,
                condition=item.condition,
            )
            for item in autonomous_result.conditions
        ],
        "parameter_counts": parameter_counts,
        "receptive_field": receptive_field_record(),
        "peak_memory": peak_memory_record(),
        "wall_time": {
            "clock": "time.perf_counter_monotonic",
            "total_subset_arm_seconds": float(total_run_seconds),
            "autonomous_encoding_seconds": autonomous_result.encoding_seconds,
            "autonomous_evaluation_seconds_by_condition": {
                item.condition: item.elapsed_seconds
                for item in autonomous_result.conditions
            },
            "metric_computation_seconds_by_condition": {
                item["condition"]: item["metric_computation_seconds"]
                for item in condition_metrics
            },
        },
        "exact_sufficiency_gate": exact_gate,
        "memory_use_gate": memory_gate,
        "access": dict(access_flags),
    }
    return validate_primary_report(record)


def run_c7_v2(
    *, corpus_dir, output_dir, repository_root, expected_commit,
    access_tracker=None
):
    """Run the prospective formal gate and atomically publish its artifact."""

    job_id = _require_authoritative_runtime()
    tracker = access_tracker or C7V2AccessTracker()
    source = _source_identity(repository_root, expected_commit)
    final, staging = prepare_c7_v2_output(
        output_dir, repository_root=repository_root, corpus_dir=corpus_dir,
        job_id=job_id,
    )
    events = []
    run_started = time.perf_counter()
    tracker.operation_template_manifest_accessed = "not_confirmed_on_failure"
    selection = select_c7_sufficiency_subsets(corpus_dir)
    tracker.operation_template_manifest_accessed = True
    selection_record = validate_frozen_c7_v2_selection(selection)
    events.append({"event": "c7_v2_subset_selection", **selection_record})

    from .config import GE1TrainingConfig
    from .model import build_legacy_matched_ge1_models

    protocol_config = C7V2Configuration().validate()
    inherited_training_config = GE1TrainingConfig()
    inherited_training_config.validate()
    gates = {}
    parameter_inventories = {}
    model_configurations = {}

    tiny_examples = _load_selected_train(
        corpus_dir, selection.tiny_family_ids, tracker, subset_identity="tiny"
    )
    tiny_models = build_legacy_matched_ge1_models(seed=C7_V2_SEED)
    _assert_matched_disjoint(*tiny_models)
    parameter_inventories["tiny"] = parameter_count_record(*tiny_models)
    model_configurations.update({
        model.config.encoder: model.config.to_dict() for model in tiny_models
    })
    for arm, model in zip(C7_V2_ARMS, tiny_models):
        result = _train_evaluate_arm(
            model=model,
            examples=tiny_examples,
            subset_identity="tiny",
            staging=staging,
            repository_root=repository_root,
            expected_commit=expected_commit,
            job_id=job_id,
            training_config=inherited_training_config,
            parameter_counts=parameter_inventories["tiny"],
            events=events,
            tracker=tracker,
        )
        gates["{}.tiny_exact_sufficiency".format(arm)] = result["exact_gate"]

    both_tiny_pass = all(
        gates["{}.tiny_exact_sufficiency".format(arm)]["status"] == "pass"
        for arm in C7_V2_ARMS
    )
    if both_tiny_pass:
        scaled_examples = _load_selected_train(
            corpus_dir,
            selection.scaled_family_ids,
            tracker,
            subset_identity="scaled",
        )
        scaled_models = build_legacy_matched_ge1_models(seed=C7_V2_SEED)
        _assert_matched_disjoint(*scaled_models)
        _assert_model_collections_disjoint(tiny_models, scaled_models)
        parameter_inventories["scaled"] = parameter_count_record(*scaled_models)
        if parameter_inventories["scaled"] != parameter_inventories["tiny"]:
            raise GraphEncoderError(
                "c7_v2_model_lifecycle_failure",
                "fresh tiny and scaled pairs have different capacities",
            )
        for arm, model in zip(C7_V2_ARMS, scaled_models):
            result = _train_evaluate_arm(
                model=model,
                examples=scaled_examples,
                subset_identity="scaled",
                staging=staging,
                repository_root=repository_root,
                expected_commit=expected_commit,
                job_id=job_id,
                training_config=inherited_training_config,
                parameter_counts=parameter_inventories["scaled"],
                events=events,
                tracker=tracker,
            )
            gates["{}.scaled_exact_sufficiency".format(arm)] = (
                result["exact_gate"]
            )
            gates["{}.scaled_memory_use".format(arm)] = result["memory_gate"]
    else:
        if tracker.scaled_train_payload_accessed is not False:
            raise GraphEncoderError(
                "c7_v2_access_dependency_failure",
                "scaled payload state changed before both tiny arms passed",
            )
        for arm in C7_V2_ARMS:
            for gate_name in (
                "scaled_exact_sufficiency", "scaled_memory_use"
            ):
                gate = not_run_gate(
                    arm=arm,
                    gate_name=gate_name,
                    reason="both_tiny_exact_sufficiency_gates_did_not_pass",
                    access_flags=tracker.declarations(),
                )
                gates["{}.{}".format(arm, gate_name)] = gate
                events.append({"event": "c7_v2_gate_result", **gate})

    decision = overall_c7_v2_decision(gates)
    decision.update(tracker.declarations(completed=True))
    events.append({"event": "c7_v2_overall_gate_decision", **decision})
    batches = tuple(
        selection.scaled_family_ids[start:start + C7_V2_BATCH_SIZE]
        for start in range(
            0, len(selection.scaled_family_ids), C7_V2_BATCH_SIZE
        )
    )
    resolved = {
        "protocol_version": C7_V2_PROTOCOL_VERSION,
        "metrics_version": C7_V2_METRICS_VERSION,
        "artifact_version": C7_V2_ARTIFACT_VERSION,
        "source": source,
        "governed_source_digest": source["source_tree_sha256"],
        "historical_records": {
            "formal_c7_v1": {
                "pilot_version": C7_PILOT_VERSION,
                "gate_version": C7_GATE_VERSION,
                "artifact_version": C7_ARTIFACT_MANIFEST_VERSION,
                "checkpoint_epoch": C7_CHECKPOINT_EPOCH,
                "immutable_result_changed": False,
            },
            "optimization_diagnostic": {
                "job_id": "3344907",
                "exact_commit": (
                    "fbc6073f63da9f0e10b5db8c0c0d4786a48ce0c0"
                ),
                "result_changed": False,
                "checkpoints_reused": False,
            },
        },
        "authoritative_manifest": {
            "name": "operation_template",
            "sha256": AUTHORITATIVE_FILE_SHA256,
            "authorized_partition": "train",
            "authorized_templates": list(C7_ACCESSIBLE_TEMPLATES),
        },
        "subset_selection": selection_record,
        "seed": C7_V2_SEED,
        "model_configurations": model_configurations,
        "c7_v2_configuration": protocol_config.to_dict(),
        "inherited_optimizer_loss_configuration": (
            inherited_training_config.to_dict()
        ),
        "training_arithmetic": {
            "tiny": c7_v2_training_arithmetic(4),
            "scaled": c7_v2_training_arithmetic(32),
            "eventual_stage6_if_authorized": c7_v2_training_arithmetic(407),
            "maximum_c7_v2_training_runs": 4,
        },
        "model_lifecycle": {
            "fresh_matched_pair_per_subset": True,
            "shared_initialization_between_arms": True,
            "mutable_objects_disjoint": True,
            "tiny_to_scaled_warm_start": False,
            "c7_v1_checkpoint_reuse": False,
            "diagnostic_checkpoint_reuse": False,
            "checkpoint_role": C7_V2_CHECKPOINT_ROLE,
        },
        "checkpoint_selection": {
            "selected_epoch": C7_V2_CHECKPOINT_EPOCH,
            "strict_reload_required": True,
            "earlier_epoch_gate_eligible": False,
            "training_loss_selection": False,
            "best_checkpoint_selection": False,
            "early_stopping": False,
            "outcome_dependent_extension": False,
        },
        "gate_dependency": {
            "both_tiny_arms_required_before_scaled_payload_access": True,
            "both_tiny_arms_passed": both_tiny_pass,
        },
        "gate_thresholds": {
            "per_family_exact_required": 1.0,
            "depends_on_exactness_required_for": ["EE", "RE"],
            "P_true_comparison": "strictly_greater_than_zero",
            "R_shuffle_max_inclusive": C7_V2_MEMORY_RATIO_MAX,
            "R_mean_max_inclusive": C7_V2_MEMORY_RATIO_MAX,
            "hidden_epsilon": False,
            "rounded_comparison": False,
        },
        "evaluation": {
            "order": list(selection.scaled_family_ids),
            "batch_size": C7_V2_BATCH_SIZE,
            "batch_membership": [list(item) for item in batches],
            "memory_conditions": ["P_true", "P_shuffle", "P_mean"],
            "teacher_forced_gate_eligible": False,
        },
        "parameter_inventory": parameter_inventories,
        "runtime": {
            "python": platform.python_version(),
            "pytorch": str(torch.__version__),
            "device": "cpu",
            "cuda_available": bool(torch.cuda.is_available()),
            "host": socket.gethostname(),
            "slurm_job_id": job_id,
        },
        "access": tracker.declarations(completed=True),
        "formal_c7_v2_execution": True,
        "stage6_performed": False,
        "c8_or_later_performed": False,
        "decoder_repair_implemented_or_invoked": False,
    }
    run_metadata = {
        "event": "c7_v2_run_metadata",
        "protocol_version": C7_V2_PROTOCOL_VERSION,
        "metrics_version": C7_V2_METRICS_VERSION,
        "artifact_version": C7_V2_ARTIFACT_VERSION,
        "source": source,
        "seed": C7_V2_SEED,
        "slurm_job_id": job_id,
        "access": tracker.declarations(completed=True),
    }
    completion = {
        "event": "c7_v2_execution_completed",
        "slurm_job_id": job_id,
        **decision,
        "elapsed_seconds": time.perf_counter() - run_started,
    }
    events = [run_metadata] + events + [completion]
    _atomic_write_json(staging / "resolved_config.json", resolved)
    _atomic_write_jsonl(staging / "metrics.jsonl", events)
    _verify_source_unchanged(source, repository_root, expected_commit)
    finalize_c7_v2_artifact(staging, final)
    verified = verify_c7_v2_artifact(
        final,
        expected_commit=expected_commit,
        expected_slurm_job_id=job_id,
    )
    return {
        "artifact_path": str(final),
        "artifact_verification": verified,
        "overall_decision": decision,
        "completion_event": completion,
    }


def _train_evaluate_arm(
    *, model, examples, subset_identity, staging, repository_root,
    expected_commit, job_id, training_config, parameter_counts, events,
    tracker
):
    from .autonomous import autonomous_input_from_paired, run_autonomous_evaluation
    from .batching import build_paired_batch
    from .metrics import score_condition
    from .model import build_ge1_model
    from .provenance import training_partition_identity
    from .training import load_training_checkpoint, run_ge1_training

    arm = model.config.encoder
    run_start = time.perf_counter()
    checkpoint_dir = staging / "checkpoints" / subset_identity / arm
    events.append({
        "event": "c7_v2_training_started",
        "arm": arm,
        "subset_identity": subset_identity,
        "seed": C7_V2_SEED,
        "checkpoint_role": C7_V2_CHECKPOINT_ROLE,
        "fresh_model": True,
        "resume_checkpoint": False,
        "expected_arithmetic": c7_v2_training_arithmetic(len(examples)),
        "access": tracker.declarations(),
    })
    training = run_ge1_training(
        model,
        examples,
        training_config=training_config,
        checkpoint_directory=checkpoint_dir,
        repository_root=repository_root,
        expected_commit=expected_commit,
        final_epoch=C7_V2_CHECKPOINT_EPOCH,
        engineering_smoke=False,
        checkpoint_epochs=(C7_V2_CHECKPOINT_EPOCH,),
        fixed_protocol_final_epoch=C7_V2_CHECKPOINT_EPOCH,
        checkpoint_coordinate="epoch",
        selected_checkpoint_epoch=C7_V2_CHECKPOINT_EPOCH,
    )
    arithmetic = c7_v2_training_arithmetic(len(examples))
    if (
        training.completed_epoch != C7_V2_CHECKPOINT_EPOCH
        or training.optimizer_steps != arithmetic["optimizer_steps"]
        or training.training_example_presentations
        != arithmetic["training_example_presentations"]
        or len(training.epoch_records) != C7_V2_CHECKPOINT_EPOCH
        or len(training.checkpoint_paths) != 1
        or training.selected_experimental_checkpoint
        != training.checkpoint_paths[0]
    ):
        raise GraphEncoderError(
            "c7_v2_training_arithmetic_failure",
            "training differs from the fixed epoch-200 contract",
        )
    selected_path = Path(training.checkpoint_paths[0])
    training_record = _training_record_for_artifact(training, staging=staging)
    events.append({
        "event": "c7_v2_epoch_200_checkpoint_written",
        "arm": arm,
        "subset_identity": subset_identity,
        "epoch": C7_V2_CHECKPOINT_EPOCH,
        "checkpoint_identity": selected_path.relative_to(staging).as_posix(),
        "checkpoint_sha256": _file_sha256(selected_path),
        "training": training_record,
        "access": tracker.declarations(),
    })

    reloaded = build_ge1_model(model.config)
    optimizer = torch.optim.AdamW(
        reloaded.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    partition_identity = training_partition_identity(
        tuple(item.physical_family_id for item in examples)
    )
    resume = load_training_checkpoint(
        selected_path,
        model=reloaded,
        optimizer=optimizer,
        training_config=training_config,
        partition_identity=partition_identity,
        expected_code_revision=expected_commit,
        restore_rng=True,
        expected_selected_checkpoint_epoch=C7_V2_CHECKPOINT_EPOCH,
    )
    validate_c7_v2_reloaded_checkpoint(
        resume, expected_commit=expected_commit, job_id=job_id
    )
    _assert_model_states_equal(model, reloaded)
    relative_checkpoint = selected_path.relative_to(staging).as_posix()
    events.append({
        "event": "c7_v2_checkpoint_reloaded",
        "arm": arm,
        "subset_identity": subset_identity,
        "epoch": C7_V2_CHECKPOINT_EPOCH,
        "checkpoint_identity": relative_checkpoint,
        "checkpoint_sha256": _file_sha256(selected_path),
        "checkpoint_role": C7_V2_CHECKPOINT_ROLE,
        "strict_reload": True,
        "fresh_model_constructed": True,
        "source_commit": expected_commit,
        "slurm_job_id": job_id,
        "access": tracker.declarations(),
    })

    ordered = tuple(sorted(examples, key=lambda item: item.physical_family_id))
    input_batches = tuple(
        autonomous_input_from_paired(
            build_paired_batch(ordered[start:start + C7_V2_BATCH_SIZE]), arm
        )
        for start in range(0, len(ordered), C7_V2_BATCH_SIZE)
    )
    autonomous = run_autonomous_evaluation(
        reloaded, input_batches, seed=C7_V2_SEED
    )
    targets = {item.physical_family_id: item.target for item in ordered}
    metrics = tuple(
        score_condition(condition, targets)
        for condition in autonomous.conditions
    )
    templates = {
        item.physical_family_id: item.metadata.operation_template
        for item in ordered
    }
    condition_by_name = {item["condition"]: item for item in metrics}
    for condition, metric in zip(autonomous.conditions, metrics):
        events.append({
            "event": "c7_v2_autonomous_condition_metrics",
            "arm": arm,
            "subset_identity": subset_identity,
            "condition": condition.condition,
            "metrics": metric,
            "memory_assignments": [
                list(item) for item in condition.memory_assignments
            ],
            "memory_altered": condition.memory_altered,
            "alteration_possible": condition.alteration_possible,
            "batch_membership": [
                [identity, list(families)]
                for identity, families in condition.batch_membership
            ],
            "access": tracker.declarations(),
        })
    exact = exact_sufficiency_gate(
        arm=arm,
        subset_identity=subset_identity,
        condition_metrics=condition_by_name["P_true"],
        templates_by_family=templates,
        checkpoint_identity=relative_checkpoint,
        epoch=C7_V2_CHECKPOINT_EPOCH,
        access_flags=tracker.declarations(),
    )
    events.append({"event": "c7_v2_gate_result", **exact})
    memory = None
    if subset_identity == "scaled":
        memory = memory_use_gate(
            arm=arm,
            condition_metrics=metrics,
            autonomous_result=autonomous,
            templates_by_family=templates,
            checkpoint_identity=relative_checkpoint,
            epoch=C7_V2_CHECKPOINT_EPOCH,
            access_flags=tracker.declarations(),
        )
        events.append({"event": "c7_v2_gate_result", **memory})
        envelope = c7_v2_metrics_envelope(
            arm=arm,
            checkpoint_identity=relative_checkpoint,
            condition_metrics=metrics,
            autonomous_result=autonomous,
            templates_by_family=templates,
            training_result=training,
            parameter_counts=parameter_counts,
            exact_gate=exact,
            memory_gate=memory,
            access_flags=tracker.declarations(),
            total_run_seconds=time.perf_counter() - run_start,
            staging=staging,
        )
        events.append({
            "event": "c7_v2_scaled_metrics_envelope",
            "record": envelope,
        })
    return {"exact_gate": exact, "memory_gate": memory}


def _load_selected_train(corpus_dir, family_ids, tracker, *, subset_identity):
    if subset_identity not in C7_V2_SUBSETS:
        raise GraphEncoderError("invalid_c7_v2_subset", "unknown subset")
    if subset_identity == "scaled" and tracker.scaled_train_payload_accessed is False:
        tracker.scaled_train_payload_accessed = "not_confirmed_on_failure"
    if subset_identity == "tiny" and tracker.tiny_train_payload_accessed is False:
        tracker.tiny_train_payload_accessed = "not_confirmed_on_failure"
    if tracker.operation_template_train_payload_accessed is not True:
        tracker.operation_template_train_payload_accessed = (
            "not_confirmed_on_failure"
        )
    examples = tuple(load_train(corpus_dir, family_ids))
    observed = tuple(sorted(item.physical_family_id for item in examples))
    if observed != tuple(family_ids):
        raise GraphEncoderError(
            "c7_v2_payload_alignment_failure",
            "loaded train families differ from metadata selection",
        )
    tracker.operation_template_train_payload_accessed = True
    if subset_identity == "tiny":
        tracker.tiny_train_payload_accessed = True
    else:
        tracker.scaled_train_payload_accessed = True
    return examples


def _training_record_for_artifact(training_result, staging=None):
    record = training_result.to_dict()
    if staging is not None:
        root = Path(staging)
        record["checkpoint_paths"] = [
            Path(item).relative_to(root).as_posix()
            for item in training_result.checkpoint_paths
        ]
        selected = training_result.selected_experimental_checkpoint
        record["selected_experimental_checkpoint"] = (
            None if selected is None else Path(selected).relative_to(root).as_posix()
        )
    record.update({
        "checkpoint_role": C7_V2_CHECKPOINT_ROLE,
        "full_comparison_checkpoint": False,
        "fixed_checkpoint_epoch": C7_V2_CHECKPOINT_EPOCH,
        "early_stopping": False,
        "best_checkpoint_selection": False,
        "outcome_dependent_extension": False,
        "warm_start": False,
    })
    return record


def prepare_c7_v2_output(
    output_dir, *, repository_root, corpus_dir, job_id=None
):
    """Create a new external incomplete directory for a C7-v2 artifact."""

    identity = validate_slurm_job_id(job_id)
    raw = Path(output_dir)
    if raw.is_symlink() or raw.exists():
        raise GraphEncoderError(
            "c7_v2_output_exists", "output must be a new non-symlink path"
        )
    final = raw.resolve()
    repository = Path(repository_root).resolve()
    corpus = Path(corpus_dir).resolve()
    if _is_within(final, repository) or _is_within(final, corpus):
        raise GraphEncoderError(
            "unsafe_c7_v2_output_location",
            "output must be outside repository and corpus",
        )
    final.parent.mkdir(parents=True, exist_ok=True)
    staging = final.with_name(final.name + ".incomplete-" + identity)
    if staging.exists() or staging.is_symlink():
        raise GraphEncoderError(
            "c7_v2_staging_exists", "incomplete staging path exists"
        )
    staging.mkdir()
    (staging / "checkpoints").mkdir()
    return final, staging


def finalize_c7_v2_artifact(staging_dir, final_dir):
    """Write both integrity layers, verify them, and atomically publish."""

    staging = Path(staging_dir)
    final = Path(final_dir)
    if staging.parent.resolve() != final.parent.resolve():
        raise GraphEncoderError(
            "unsafe_c7_v2_output_location",
            "staging and final directories must share a parent",
        )
    if final.exists() or final.is_symlink():
        raise GraphEncoderError("c7_v2_output_exists", "final output exists")
    required = (
        staging / "resolved_config.json",
        staging / "metrics.jsonl",
        staging / "checkpoints",
    )
    if any(not item.exists() for item in required):
        raise GraphEncoderError(
            "incomplete_c7_v2_artifact", "required artifact is absent"
        )
    ordinary = _regular_artifact_files(
        staging, excluded=("artifact_manifest.json", "SHA256SUMS")
    )
    checkpoints = tuple(
        path for path in ordinary if path.startswith("checkpoints/")
    )
    if len(checkpoints) not in (2, 4):
        raise GraphEncoderError(
            "incomplete_c7_v2_artifact",
            "artifact must contain two or four selected checkpoints",
        )
    manifest = {
        "schema_version": C7_V2_ARTIFACT_VERSION,
        "artifacts": [
            {
                "path": path,
                "byte_size": (staging / path).stat().st_size,
                "sha256": _file_sha256(staging / path),
            }
            for path in ordinary
        ],
    }
    _atomic_write_json(staging / "artifact_manifest.json", manifest)
    checksum_paths = _regular_artifact_files(staging, excluded=("SHA256SUMS",))
    checksum_text = "".join(
        "{}  {}\n".format(_file_sha256(staging / path), path)
        for path in checksum_paths
    )
    _atomic_write_bytes(staging / "SHA256SUMS", checksum_text.encode("utf-8"))
    verify_c7_v2_artifact(staging, allow_incomplete_name=True)
    os.replace(str(staging), str(final))
    return final


def verify_c7_v2_artifact(
    path, *, allow_incomplete_name=False, expected_commit=None,
    expected_slurm_job_id=None
):
    """Verify paths, identities, hashes, provenance, events, and access."""

    root = Path(path)
    if not root.is_dir() or root.is_symlink():
        raise GraphEncoderError(
            "invalid_c7_v2_artifact", "artifact root must be a real directory"
        )
    if ".incomplete-" in root.name and not allow_incomplete_name:
        raise GraphEncoderError(
            "invalid_c7_v2_artifact", "incomplete artifact cannot be final"
        )
    manifest_path = root / "artifact_manifest.json"
    checksums_path = root / "SHA256SUMS"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        checksum_raw = checksums_path.read_bytes()
    except (OSError, ValueError) as exc:
        raise GraphEncoderError(
            "invalid_c7_v2_artifact", "integrity metadata is unreadable"
        ) from exc
    if manifest.get("schema_version") != C7_V2_ARTIFACT_VERSION:
        raise GraphEncoderError(
            "invalid_c7_v2_artifact", "artifact identity differs"
        )
    rows = manifest.get("artifacts")
    if not isinstance(rows, list):
        raise GraphEncoderError("invalid_c7_v2_artifact", "artifact rows absent")
    listed = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != {
            "path", "byte_size", "sha256"
        }:
            raise GraphEncoderError(
                "invalid_c7_v2_artifact", "artifact row is malformed"
            )
        relative = _safe_relative_path(row["path"])
        target = root / relative
        if not target.is_file() or target.is_symlink():
            raise GraphEncoderError(
                "invalid_c7_v2_artifact", "listed path is not a regular file"
            )
        if (
            target.stat().st_size != row["byte_size"]
            or _file_sha256(target) != row["sha256"]
        ):
            raise GraphEncoderError(
                "c7_v2_artifact_integrity_failure",
                "artifact size or SHA-256 differs",
            )
        listed.append(relative)
    expected_ordinary = _regular_artifact_files(
        root, excluded=("artifact_manifest.json", "SHA256SUMS")
    )
    if (
        listed != sorted(listed)
        or len(listed) != len(set(listed))
        or tuple(listed) != expected_ordinary
    ):
        raise GraphEncoderError(
            "invalid_c7_v2_artifact", "manifest coverage differs"
        )
    if not checksum_raw.endswith(b"\n"):
        raise GraphEncoderError(
            "invalid_c7_v2_artifact", "SHA256SUMS requires final LF"
        )
    try:
        checksum_lines = checksum_raw.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise GraphEncoderError(
            "invalid_c7_v2_artifact", "SHA256SUMS is not UTF-8"
        ) from exc
    checksum_paths = []
    for line in checksum_lines:
        parts = line.split("  ", 1)
        if len(parts) != 2 or len(parts[0]) != 64:
            raise GraphEncoderError(
                "invalid_c7_v2_artifact", "SHA256SUMS row is malformed"
            )
        relative = _safe_relative_path(parts[1])
        if _file_sha256(root / relative) != parts[0]:
            raise GraphEncoderError(
                "c7_v2_artifact_integrity_failure", "SHA256SUMS hash differs"
            )
        checksum_paths.append(relative)
    expected_checksum_paths = _regular_artifact_files(
        root, excluded=("SHA256SUMS",)
    )
    if tuple(checksum_paths) != expected_checksum_paths:
        raise GraphEncoderError(
            "invalid_c7_v2_artifact", "SHA256SUMS coverage differs"
        )

    events = _read_canonical_jsonl(root / "metrics.jsonl")
    if not events or events[-1].get("event") != "c7_v2_execution_completed":
        raise GraphEncoderError(
            "invalid_c7_v2_artifact", "terminal completion event is absent"
        )
    observed_events = {item.get("event") for item in events}
    required_events = {
        "c7_v2_run_metadata",
        "c7_v2_subset_selection",
        "c7_v2_training_started",
        "c7_v2_epoch_200_checkpoint_written",
        "c7_v2_checkpoint_reloaded",
        "c7_v2_autonomous_condition_metrics",
        "c7_v2_gate_result",
        "c7_v2_overall_gate_decision",
        "c7_v2_execution_completed",
    }
    if not required_events.issubset(observed_events):
        raise GraphEncoderError(
            "invalid_c7_v2_artifact", "required JSONL events are absent"
        )
    try:
        resolved = json.loads(
            (root / "resolved_config.json").read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as exc:
        raise GraphEncoderError(
            "invalid_c7_v2_artifact", "resolved configuration is unreadable"
        ) from exc
    if (
        resolved.get("protocol_version") != C7_V2_PROTOCOL_VERSION
        or resolved.get("metrics_version") != C7_V2_METRICS_VERSION
        or resolved.get("artifact_version") != C7_V2_ARTIFACT_VERSION
        or resolved.get("checkpoint_selection", {}).get("selected_epoch")
        != C7_V2_CHECKPOINT_EPOCH
        or resolved.get("source", {}).get("git_dirty") is not False
    ):
        raise GraphEncoderError(
            "invalid_c7_v2_artifact", "resolved identities are malformed"
        )
    runtime = resolved.get("runtime", {})
    selection = resolved.get("subset_selection", {})
    protocol_configuration = resolved.get("c7_v2_configuration", {})
    arithmetic = resolved.get("training_arithmetic", {})
    history = resolved.get("historical_records", {})
    if (
        runtime.get("python") != "3.8.13"
        or str(runtime.get("pytorch", "")).split("+")[0] != "1.11.0"
        or runtime.get("device") != "cpu"
        or runtime.get("cuda_available") is not False
        or selection.get("tiny_family_ids_sha256")
        != C7_V2_TINY_FAMILY_IDS_SHA256
        or selection.get("scaled_family_ids_sha256")
        != C7_V2_SCALED_FAMILY_IDS_SHA256
        or protocol_configuration != C7V2Configuration().to_dict()
        or arithmetic.get("tiny") != c7_v2_training_arithmetic(4)
        or arithmetic.get("scaled") != c7_v2_training_arithmetic(32)
        or arithmetic.get("eventual_stage6_if_authorized")
        != c7_v2_training_arithmetic(407)
        or history.get("formal_c7_v1", {}).get("checkpoint_epoch")
        != C7_CHECKPOINT_EPOCH
        or history.get("formal_c7_v1", {}).get("immutable_result_changed")
        is not False
        or history.get("optimization_diagnostic", {}).get("result_changed")
        is not False
        or history.get("optimization_diagnostic", {}).get("checkpoints_reused")
        is not False
    ):
        raise GraphEncoderError(
            "invalid_c7_v2_artifact",
            "environment, cohort, budget, or historical boundary differs",
        )
    source_commit = resolved["source"].get("git_commit")
    if (
        not isinstance(source_commit, str)
        or len(source_commit) != 40
        or any(character not in "0123456789abcdef" for character in source_commit)
    ):
        raise GraphEncoderError(
            "invalid_c7_v2_artifact", "source commit is not a full lowercase hash"
        )
    slurm_job_id = validate_slurm_job_id(
        runtime.get("slurm_job_id")
    )
    if expected_commit is not None and source_commit != expected_commit:
        raise GraphEncoderError(
            "invalid_c7_v2_artifact", "source commit differs"
        )
    if (
        expected_slurm_job_id is not None
        and slurm_job_id != validate_slurm_job_id(expected_slurm_job_id)
    ):
        raise GraphEncoderError(
            "invalid_c7_v2_artifact", "Slurm identity differs"
        )
    terminal = events[-1]
    for name in (
        "c7_v2_overall_gate_pass",
        "stage6_authorized_by_c7_v2",
        "preauthorized_decoder_repair_path_next",
        "comparison_inconclusive",
        "c7_v2_execution_completed",
    ):
        if not isinstance(terminal.get(name), bool):
            raise GraphEncoderError(
                "invalid_c7_v2_artifact", "terminal decision is malformed"
            )
    if (
        terminal["stage6_authorized_by_c7_v2"]
        is not terminal["c7_v2_overall_gate_pass"]
        or terminal.get("slurm_job_id") != slurm_job_id
        or terminal.get("operation_template_manifest_accessed") is not True
        or terminal.get("operation_template_train_payload_accessed") is not True
        or terminal.get("tiny_train_payload_accessed") is not True
        or not isinstance(terminal.get("scaled_train_payload_accessed"), bool)
        or terminal.get("development_accessed") is not False
        or terminal.get("systematic_rr_accessed") is not False
        or terminal.get("test_er_accessed") is not False
        or terminal.get("iid_accessed") is not False
        or terminal.get("history_depth_accessed") is not False
        or terminal.get("geometry_extrapolation_accessed") is not False
        or terminal.get("stage6_performed") is not False
        or terminal.get("c8_or_later_performed") is not False
        or terminal.get("decoder_repair_implemented_or_invoked") is not False
    ):
        raise GraphEncoderError(
            "invalid_c7_v2_artifact", "authorization or access differs"
        )
    gate_rows = [
        item for item in events if item.get("event") == "c7_v2_gate_result"
    ]
    gates = {
        "{}.{}".format(item.get("arm"), item.get("gate_name")): item
        for item in gate_rows
    }
    if (
        len(gate_rows) != len(C7_V2_REQUIRED_GATE_NAMES)
        or set(gates) != set(C7_V2_REQUIRED_GATE_NAMES)
        or any(item.get("version") != C7_V2_PROTOCOL_VERSION for item in gate_rows)
    ):
        raise GraphEncoderError(
            "invalid_c7_v2_artifact", "complete versioned gate set is absent"
        )
    scaled_statuses = tuple(
        gates["{}.{}".format(arm, gate_name)]["status"]
        for arm in C7_V2_ARMS
        for gate_name in ("scaled_exact_sufficiency", "scaled_memory_use")
    )
    if (
        terminal["scaled_train_payload_accessed"] is False
        and scaled_statuses != ("not_run",) * 4
    ) or (
        terminal["scaled_train_payload_accessed"] is True
        and "not_run" in scaled_statuses
    ):
        raise GraphEncoderError(
            "invalid_c7_v2_artifact", "scaled access and gate dependency differ"
        )
    expected_decision = overall_c7_v2_decision(gates)
    for name in (
        "c7_v2_overall_gate_pass",
        "stage6_authorized_by_c7_v2",
        "preauthorized_decoder_repair_path_next",
        "comparison_inconclusive",
        "c7_v2_execution_completed",
    ):
        if terminal.get(name) is not expected_decision[name]:
            raise GraphEncoderError(
                "invalid_c7_v2_artifact", "terminal decision differs from gates"
            )
    checkpoint_paths = tuple(
        path for path in expected_ordinary if path.startswith("checkpoints/")
    )
    expected_checkpoint_count = (
        4 if terminal.get("scaled_train_payload_accessed") is True else 2
    )
    if len(checkpoint_paths) != expected_checkpoint_count:
        raise GraphEncoderError(
            "invalid_c7_v2_artifact", "selected checkpoint count differs"
        )
    written = [
        item for item in events
        if item.get("event") == "c7_v2_epoch_200_checkpoint_written"
    ]
    reloaded = [
        item for item in events
        if item.get("event") == "c7_v2_checkpoint_reloaded"
    ]
    if len(written) != expected_checkpoint_count or len(reloaded) != len(written):
        raise GraphEncoderError(
            "invalid_c7_v2_artifact", "checkpoint event count differs"
        )
    by_identity = {item["checkpoint_identity"]: item for item in written}
    if set(by_identity) != set(checkpoint_paths):
        raise GraphEncoderError(
            "invalid_c7_v2_artifact", "checkpoint identities differ"
        )
    event_positions = {id(item): index for index, item in enumerate(events)}
    for reload_event in reloaded:
        identity = reload_event.get("checkpoint_identity")
        write_event = by_identity.get(identity)
        if (
            write_event is None
            or event_positions[id(write_event)] >= event_positions[id(reload_event)]
            or reload_event.get("epoch") != C7_V2_CHECKPOINT_EPOCH
            or reload_event.get("strict_reload") is not True
            or reload_event.get("fresh_model_constructed") is not True
            or reload_event.get("source_commit") != source_commit
            or reload_event.get("slurm_job_id") != slurm_job_id
            or reload_event.get("checkpoint_sha256")
            != _file_sha256(root / identity)
            or write_event.get("checkpoint_sha256")
            != reload_event.get("checkpoint_sha256")
        ):
            raise GraphEncoderError(
                "invalid_c7_v2_artifact",
                "checkpoint write/reload provenance differs",
            )
    return {
        "protocol_version": C7_V2_PROTOCOL_VERSION,
        "artifact_manifest_sha256": _file_sha256(manifest_path),
        "sha256sums_sha256": _file_sha256(checksums_path),
        "regular_file_count": len(expected_checksum_paths) + 1,
        "checkpoint_count": len(checkpoint_paths),
        "source_commit": source_commit,
        "slurm_job_id": slurm_job_id,
        "c7_v2_overall_gate_pass": terminal["c7_v2_overall_gate_pass"],
        "stage6_authorized_by_c7_v2": terminal[
            "stage6_authorized_by_c7_v2"
        ],
        "preauthorized_decoder_repair_path_next": terminal[
            "preauthorized_decoder_repair_path_next"
        ],
    }


def _require_authoritative_runtime():
    if torch is None:
        raise RuntimeError("formal C7-v2 execution requires PyTorch")
    if sys.version_info[:3] != (3, 8, 13):
        raise GraphEncoderError(
            "environment_mismatch", "C7-v2 requires Python 3.8.13"
        )
    if str(torch.__version__).split("+")[0] != "1.11.0":
        raise GraphEncoderError(
            "environment_mismatch", "C7-v2 requires PyTorch 1.11.0"
        )
    if torch.cuda.is_available():
        raise GraphEncoderError(
            "environment_mismatch", "C7-v2 must execute on CPU"
        )
    return validate_slurm_job_id()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--expected-commit", required=True)
    arguments = parser.parse_args(argv)
    tracker = C7V2AccessTracker()
    try:
        result = run_c7_v2(
            corpus_dir=arguments.corpus_dir,
            output_dir=arguments.output_dir,
            repository_root=arguments.repository_root,
            expected_commit=arguments.expected_commit,
            access_tracker=tracker,
        )
    except Exception as exc:
        failure = {
            "event": "c7_v2_infrastructure_failure",
            "code": getattr(exc, "code", type(exc).__name__),
            "detail": str(exc),
            **overall_c7_v2_decision({}, infrastructure_failure=True),
            **tracker.declarations(completed=False),
        }
        print(_canonical_json_text(failure), file=sys.stderr, flush=True)
        return 1
    terminal = {
        "event": "c7_v2_terminal_execution_completed",
        "artifact_path": result["artifact_path"],
        **result["overall_decision"],
        **tracker.declarations(completed=True),
    }
    print(_canonical_json_text(terminal), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
