"""Formal train-only GE1 C7 sufficiency and autonomous memory-use gates."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import tempfile
import time

try:
    import torch
except ImportError:  # Pure gate, selection, and artifact tests are torch-free.
    torch = None

from prototype.flat_baseline.provenance import source_tree_sha256

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
from .partitions import (
    AUTHORITATIVE_FILE_SHA256,
    C7_ACCESSIBLE_TEMPLATES,
    C7_SELECTION_VERSION,
    load_train,
    select_c7_sufficiency_subsets,
)


C7_PILOT_VERSION = "GE1-C7-PILOT-v1"
C7_GATE_VERSION = "GE1-C7-SUFFICIENCY-GATES-v1"
C7_ARTIFACT_MANIFEST_VERSION = "GE1-C7-ARTIFACT-MANIFEST-v1"
C7_GATE_SEED = 2026
C7_CHECKPOINT_EPOCH = 50
C7_BATCH_SIZE = 8
C7_MEMORY_RATIO_MAX = 0.80
C7_GATE_STATUSES = ("pass", "fail", "not_run")
C7_SUBSETS = ("tiny", "scaled")
C7_ARMS = ("flat", "typed_graph")
C7_CHECKPOINT_ROLE = "c7_gate_only_not_full_comparison"
C7_REQUIRED_GATE_NAMES = tuple(
    "{}.{}".format(arm, name)
    for arm in C7_ARMS
    for name in (
        "tiny_exact_sufficiency",
        "scaled_exact_sufficiency",
        "scaled_memory_use",
    )
)
C7_ACCESS_FALSE_FIELDS = (
    "development_accessed",
    "systematic_rr_accessed",
    "test_er_accessed",
    "iid_accessed",
    "history_depth_accessed",
    "geometry_extrapolation_accessed",
)


@dataclass
class C7AccessTracker:
    operation_template_manifest_accessed: object = False
    operation_template_train_payload_accessed: object = False

    def declarations(self, *, gate_completed=False):
        values = {
            "operation_template_manifest_accessed": (
                self.operation_template_manifest_accessed
            ),
            "operation_template_train_payload_accessed": (
                self.operation_template_train_payload_accessed
            ),
            "c7_gate_completed": bool(gate_completed),
            "c8_or_later_performed": False,
        }
        values.update({name: False for name in C7_ACCESS_FALSE_FIELDS})
        return values


def c7_training_arithmetic(family_count):
    """Return and validate the frozen C7 presentations and optimizer steps."""

    if family_count not in (4, 32):
        raise GraphEncoderError(
            "invalid_c7_subset_size", "C7 subset must contain 4 or 32 families"
        )
    steps_per_epoch = int(math.ceil(family_count / float(C7_BATCH_SIZE)))
    return {
        "family_count": family_count,
        "epochs": C7_CHECKPOINT_EPOCH,
        "batch_size": C7_BATCH_SIZE,
        "steps_per_epoch": steps_per_epoch,
        "optimizer_steps": steps_per_epoch * C7_CHECKPOINT_EPOCH,
        "training_example_presentations": family_count * C7_CHECKPOINT_EPOCH,
    }


def validate_c7_reloaded_checkpoint(resume_state):
    """Require the C6 strict loader to have restored selected epoch 50."""

    payload = getattr(resume_state, "payload", None)
    if (
        getattr(resume_state, "completed_epoch", None) != C7_CHECKPOINT_EPOCH
        or not isinstance(payload, dict)
        or payload.get("selected_experimental_checkpoint") is not True
        or payload.get("completed_epoch") != C7_CHECKPOINT_EPOCH
    ):
        raise GraphEncoderError(
            "c7_checkpoint_reload_failure", "strict epoch-50 reload failed"
        )
    return resume_state


def exact_sufficiency_gate(
    *, arm, subset_identity, condition_metrics, templates_by_family,
    checkpoint_identity, epoch, access_flags, generation_condition="P_true"
):
    """Apply the exact per-family autonomous C7 sufficiency criteria."""

    _validate_arm_subset(arm, subset_identity)
    if generation_condition != "P_true":
        raise GraphEncoderError(
            "nonautonomous_gate_input",
            "only autonomous P_true generation may satisfy C7 exactness",
        )
    if epoch != C7_CHECKPOINT_EPOCH:
        raise GraphEncoderError(
            "wrong_c7_checkpoint_epoch", "only strict epoch 50 may satisfy C7"
        )
    if not isinstance(condition_metrics, dict):
        raise GraphEncoderError("malformed_c7_metrics", "metrics must be an object")
    if condition_metrics.get("condition") != "P_true":
        raise GraphEncoderError(
            "nonautonomous_gate_input", "exact gate requires P_true metrics"
        )
    if condition_metrics.get("available") is not True:
        raise GraphEncoderError(
            "unavailable_c7_metrics", "P_true exact metrics must be available"
        )
    family_values = condition_metrics.get("family_values")
    if not isinstance(family_values, dict) or not family_values:
        raise GraphEncoderError(
            "malformed_c7_metrics", "exact metrics require physical-family values"
        )
    templates = dict(templates_by_family)
    if set(family_values) != set(templates):
        raise GraphEncoderError(
            "metric_target_alignment_failure",
            "C7 family metrics and template map differ",
        )
    expected_count = 4 if subset_identity == "tiny" else 32
    if len(family_values) != expected_count:
        raise GraphEncoderError(
            "invalid_c7_subset_size",
            "{} exact gate requires {} families".format(
                subset_identity, expected_count
            ),
        )

    base_fields = (
        "exact_node_sequence",
        "exact_graph",
        "strict_conversion",
        "complete_executable_validity",
    )
    criteria = {}
    failing = []
    for field in base_fields:
        passing = 0
        for family_id in sorted(family_values):
            value = _finite_metric(family_values[family_id].get(field), field)
            if value == 1.0:
                passing += 1
            else:
                failing.append((family_id, field, value))
        criteria[field] = {
            "required_value": 1.0,
            "passing_family_count": passing,
            "physical_family_denominator": len(family_values),
        }

    dependency_ids = tuple(sorted(
        family_id
        for family_id, template in templates.items()
        if template in ("EE", "RE")
    ))
    dependency_passing = 0
    for family_id in dependency_ids:
        value = _finite_metric(
            family_values[family_id].get("depends_on_exactness"),
            "depends_on_exactness",
        )
        if value == 1.0:
            dependency_passing += 1
        else:
            failing.append((family_id, "depends_on_exactness", value))
    criteria["depends_on_exactness_for_EE_RE"] = {
        "required_value": 1.0,
        "applicable_templates": ["EE", "RE"],
        "passing_family_count": dependency_passing,
        "physical_family_denominator": len(dependency_ids),
    }

    failures_by_family = {}
    for family_id, field, observed in failing:
        failures_by_family.setdefault(family_id, []).append({
            "field": field,
            "required_value": 1.0,
            "observed_value": observed,
        })
    passed = not failures_by_family
    return {
        "version": C7_GATE_VERSION,
        "gate_name": "{}_exact_sufficiency".format(subset_identity),
        "status": "pass" if passed else "fail",
        "arm": arm,
        "seed": C7_GATE_SEED,
        "subset_identity": subset_identity,
        "physical_family_count": len(family_values),
        "generation_condition": "P_true",
        "teacher_forced_gate_eligible": False,
        "checkpoint_epoch": epoch,
        "checkpoint_identity": str(checkpoint_identity),
        "checkpoint_role": C7_CHECKPOINT_ROLE,
        "criteria": criteria,
        "failing_family_ids": sorted(failures_by_family),
        "failed_fields_by_family": {
            family_id: failures_by_family[family_id]
            for family_id in sorted(failures_by_family)
        },
        "structured_reason": (
            "all_exact_criteria_satisfied"
            if passed else "one_or_more_families_failed_exact_criteria"
        ),
        "access": dict(access_flags),
    }


def not_run_gate(*, arm, gate_name, subset_identity, reason, access_flags):
    """Create an explicit non-passing dependent-gate record."""

    _validate_arm_subset(arm, subset_identity)
    if gate_name not in (
        "scaled_exact_sufficiency", "scaled_memory_use"
    ):
        raise GraphEncoderError("invalid_c7_gate", "invalid not-run gate name")
    if not isinstance(reason, str) or not reason:
        raise GraphEncoderError("invalid_c7_gate", "not_run requires a reason")
    return {
        "version": C7_GATE_VERSION,
        "gate_name": gate_name,
        "status": "not_run",
        "arm": arm,
        "seed": C7_GATE_SEED,
        "subset_identity": subset_identity,
        "physical_family_count": 32,
        "checkpoint_epoch": None,
        "checkpoint_identity": None,
        "checkpoint_role": C7_CHECKPOINT_ROLE,
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
    """Apply the exact unrounded scaled C7 autonomous memory-use gate."""

    if arm not in C7_ARMS:
        raise GraphEncoderError("invalid_c7_arm", "unknown C7 arm")
    if epoch != C7_CHECKPOINT_EPOCH:
        raise GraphEncoderError(
            "wrong_c7_checkpoint_epoch", "only strict epoch 50 may satisfy C7"
        )
    metrics = tuple(condition_metrics)
    by_condition = {
        item.get("condition"): item
        for item in metrics
        if isinstance(item, dict)
    }
    if set(by_condition) != {"P_true", "P_shuffle", "P_mean"}:
        raise GraphEncoderError(
            "malformed_c7_metrics", "memory gate requires all three conditions"
        )
    family_ids = tuple(sorted(templates_by_family))
    if len(family_ids) != 32:
        raise GraphEncoderError(
            "invalid_c7_subset_size", "memory gate requires 32 families"
        )
    for name in ("P_true", "P_shuffle", "P_mean"):
        record = by_condition[name]
        values = record.get("family_values")
        if record.get("available") is True and (
            not isinstance(values, dict) or set(values) != set(family_ids)
        ):
            raise GraphEncoderError(
                "metric_target_alignment_failure",
                "memory condition families differ from the scaled set",
            )

    ratios = intervention_ratios(metrics)
    primary_record = {"intervention_ratios": ratios}
    report_valid = True
    try:
        validate_primary_report(primary_record)
    except GraphEncoderError:
        report_valid = False

    criteria = {
        "P_true_positive": _memory_criterion(
            ratios.get("P_true"), comparison="greater_than", threshold=0.0
        ),
        "R_shuffle_at_most_0_80": _memory_criterion(
            ratios.get("R_shuffle"), comparison="less_than_or_equal",
            threshold=C7_MEMORY_RATIO_MAX,
        ),
        "R_mean_at_most_0_80": _memory_criterion(
            ratios.get("R_mean"), comparison="less_than_or_equal",
            threshold=C7_MEMORY_RATIO_MAX,
        ),
    }
    for item in criteria.values():
        item["physical_family_denominator"] = 32
        item["aggregation"] = "physical_family_macro"
    batch_membership = _validate_scaled_batch_membership(
        autonomous_result, family_ids
    )
    conditions = {
        item.condition: item for item in autonomous_result.conditions
    }
    if set(conditions) != {"P_true", "P_shuffle", "P_mean"}:
        raise GraphEncoderError(
            "malformed_c7_autonomous_result",
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
    reasons.extend(
        name for name, item in criteria.items() if not item["passed"]
    )
    return {
        "version": C7_GATE_VERSION,
        "gate_name": "scaled_memory_use",
        "status": "pass" if passed else "fail",
        "arm": arm,
        "seed": C7_GATE_SEED,
        "subset_identity": "scaled",
        "physical_family_count": 32,
        "physical_family_denominator": 32,
        "checkpoint_epoch": epoch,
        "checkpoint_identity": str(checkpoint_identity),
        "checkpoint_role": C7_CHECKPOINT_ROLE,
        "criteria": criteria,
        "intervention_ratios": ratios,
        "primary_reporting_contract": PRIMARY_REPORTING_CONTRACT_VERSION,
        "donor_template_agreement": donor,
        "memory_evidence": memory_evidence,
        "deterministic_batch_membership": batch_membership,
        "failing_family_ids": [],
        "failed_fields_by_family": {},
        "structured_reason": (
            "all_memory_use_criteria_satisfied"
            if passed else ";".join(reasons)
        ),
        "access": dict(access_flags),
    }


def overall_c7_decision(gates, *, infrastructure_failure=False):
    """Return the only fields that may authorize Stage 6 or the C8 repair."""

    if infrastructure_failure:
        return {
            "version": C7_GATE_VERSION,
            "overall_gate_pass": False,
            "stage6_authorized_by_c7": False,
            "preauthorized_decoder_repair_triggered": False,
            "comparison_inconclusive": False,
            "c7_execution_completed": False,
            "structured_reason": "infrastructure_failure",
        }
    by_name = dict(gates)
    if set(by_name) != set(C7_REQUIRED_GATE_NAMES):
        raise GraphEncoderError(
            "incomplete_c7_gate_set", "overall decision requires all six gates"
        )
    for name, gate in by_name.items():
        if not isinstance(gate, dict) or gate.get("status") not in C7_GATE_STATUSES:
            raise GraphEncoderError(
                "malformed_c7_gate", "{} has an invalid status".format(name)
            )
    overall = all(
        by_name[name]["status"] == "pass" for name in C7_REQUIRED_GATE_NAMES
    )
    exact_names = tuple(
        name for name in C7_REQUIRED_GATE_NAMES if "exact_sufficiency" in name
    )
    repair = any(by_name[name]["status"] == "fail" for name in exact_names)
    inconclusive = (not overall) and (not repair)
    return {
        "version": C7_GATE_VERSION,
        "overall_gate_pass": overall,
        "stage6_authorized_by_c7": overall,
        "preauthorized_decoder_repair_triggered": repair,
        "comparison_inconclusive": inconclusive,
        "c7_execution_completed": True,
        "structured_reason": (
            "all_six_arm_gates_passed"
            if overall else (
                "autonomous_exact_sufficiency_failure"
                if repair else "memory_gate_failure_or_dependent_gate_not_run"
            )
        ),
    }


def c7_metrics_envelope(
    *, arm, checkpoint_identity, condition_metrics, autonomous_result,
    templates_by_family, training_result, parameter_counts,
    exact_gate, memory_gate, access_flags, total_run_seconds, staging=None
):
    """Build a C7-native scientific envelope without mutating C6 identity."""

    ratios = intervention_ratios(condition_metrics)
    record = {
        "schema_version": C7_GATE_VERSION,
        "source_condition_metrics_schema": METRICS_SCHEMA_VERSION,
        "primary_reporting_contract": PRIMARY_REPORTING_CONTRACT_VERSION,
        "pilot_version": C7_PILOT_VERSION,
        "arm": arm,
        "seed": C7_GATE_SEED,
        "epoch": C7_CHECKPOINT_EPOCH,
        "checkpoint_identity": str(checkpoint_identity),
        "checkpoint_role": C7_CHECKPOINT_ROLE,
        "scientific_result": True,
        "formal_c7_gate": True,
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


def run_pilot(
    *, corpus_dir, output_dir, repository_root, expected_commit,
    access_tracker=None
):
    """Run the formal C7 train-only gate and atomically publish its artifact."""

    _require_authoritative_runtime()
    tracker = access_tracker or C7AccessTracker()
    source = _source_identity(repository_root, expected_commit)
    final, staging = prepare_c7_output(
        output_dir, repository_root=repository_root, corpus_dir=corpus_dir
    )
    events = []
    run_started = time.perf_counter()
    tracker.operation_template_manifest_accessed = "not_confirmed_on_failure"
    selection = select_c7_sufficiency_subsets(corpus_dir)
    tracker.operation_template_manifest_accessed = True
    selection_record = selection.to_dict()
    events.append({"event": "subset_selection", **selection_record})

    from .config import GE1TrainingConfig
    from .model import (
        build_legacy_matched_ge1_models as build_matched_ge1_models,
    )

    training_config = GE1TrainingConfig()
    training_config.validate()
    selected_templates = dict(selection.selected_templates)
    gates = {}
    parameter_inventories = {}
    model_configurations = {}
    prior_models = []

    tiny_examples = _load_selected_train(
        corpus_dir, selection.tiny_family_ids, tracker
    )
    tiny_models = build_matched_ge1_models(seed=C7_GATE_SEED)
    _assert_matched_disjoint(*tiny_models)
    prior_models.extend(tiny_models)
    parameter_inventories["tiny"] = parameter_count_record(*tiny_models)
    model_configurations.update({
        model.config.encoder: model.config.to_dict() for model in tiny_models
    })
    for arm, model in zip(C7_ARMS, tiny_models):
        result = _train_evaluate_arm(
            model=model,
            examples=tiny_examples,
            subset_identity="tiny",
            staging=staging,
            repository_root=repository_root,
            expected_commit=expected_commit,
            training_config=training_config,
            parameter_counts=parameter_inventories["tiny"],
            events=events,
            tracker=tracker,
        )
        gates["{}.tiny_exact_sufficiency".format(arm)] = result["exact_gate"]

    passing_tiny_arms = tuple(
        arm for arm in C7_ARMS
        if gates["{}.tiny_exact_sufficiency".format(arm)]["status"] == "pass"
    )
    if passing_tiny_arms:
        scaled_examples = _load_selected_train(
            corpus_dir, selection.scaled_family_ids, tracker
        )
        scaled_models = build_matched_ge1_models(seed=C7_GATE_SEED)
        _assert_matched_disjoint(*scaled_models)
        _assert_model_collections_disjoint(prior_models, scaled_models)
        parameter_inventories["scaled"] = parameter_count_record(*scaled_models)
        if parameter_inventories["scaled"] != parameter_inventories["tiny"]:
            raise GraphEncoderError(
                "c7_model_lifecycle_failure",
                "tiny and scaled fresh pairs have different parameter inventories",
            )
        for arm, model in zip(C7_ARMS, scaled_models):
            if arm not in passing_tiny_arms:
                for gate_name in (
                    "scaled_exact_sufficiency", "scaled_memory_use"
                ):
                    gate = not_run_gate(
                        arm=arm,
                        gate_name=gate_name,
                        subset_identity="scaled",
                        reason="tiny_exact_sufficiency_did_not_pass",
                        access_flags=tracker.declarations(),
                    )
                    gates["{}.{}".format(arm, gate_name)] = gate
                    events.append({"event": "gate_result", **gate})
                continue
            result = _train_evaluate_arm(
                model=model,
                examples=scaled_examples,
                subset_identity="scaled",
                staging=staging,
                repository_root=repository_root,
                expected_commit=expected_commit,
                training_config=training_config,
                parameter_counts=parameter_inventories["scaled"],
                events=events,
                tracker=tracker,
            )
            gates["{}.scaled_exact_sufficiency".format(arm)] = (
                result["exact_gate"]
            )
            gates["{}.scaled_memory_use".format(arm)] = result["memory_gate"]
    else:
        for arm in C7_ARMS:
            for gate_name in (
                "scaled_exact_sufficiency", "scaled_memory_use"
            ):
                gate = not_run_gate(
                    arm=arm,
                    gate_name=gate_name,
                    subset_identity="scaled",
                    reason="tiny_exact_sufficiency_did_not_pass",
                    access_flags=tracker.declarations(),
                )
                gates["{}.{}".format(arm, gate_name)] = gate
                events.append({"event": "gate_result", **gate})

    decision = overall_c7_decision(gates)
    decision.update(tracker.declarations(gate_completed=True))
    events.append({"event": "overall_gate_decision", **decision})

    batches = tuple(
        selection.scaled_family_ids[start:start + C7_BATCH_SIZE]
        for start in range(0, len(selection.scaled_family_ids), C7_BATCH_SIZE)
    )
    resolved = {
        "pilot_version": C7_PILOT_VERSION,
        "gate_version": C7_GATE_VERSION,
        "artifact_manifest_version": C7_ARTIFACT_MANIFEST_VERSION,
        "source": source,
        "governed_source_digest": source["source_tree_sha256"],
        "authoritative_manifest": {
            "name": "operation_template",
            "sha256": AUTHORITATIVE_FILE_SHA256,
            "authorized_partition": "train",
            "authorized_templates": list(C7_ACCESSIBLE_TEMPLATES),
        },
        "subset_selection": selection_record,
        "seed": C7_GATE_SEED,
        "model_configurations": model_configurations,
        "training_configuration": training_config.to_dict(),
        "training_arithmetic": {
            "tiny": c7_training_arithmetic(4),
            "scaled": c7_training_arithmetic(32),
            "total_gate_training_runs_if_all_dependencies_pass": 4,
        },
        "model_lifecycle": {
            "fresh_matched_pair_per_subset": True,
            "shared_initialization_between_arms": True,
            "mutable_objects_disjoint": True,
            "tiny_to_scaled_warm_start": False,
            "full_comparison_warm_start_authorized": False,
            "checkpoint_role": C7_CHECKPOINT_ROLE,
        },
        "checkpoint_selection": {
            "selected_epoch": C7_CHECKPOINT_EPOCH,
            "strict_reload_required": True,
            "earlier_epoch_gate_eligible": False,
            "training_loss_selection": False,
        },
        "gate_thresholds": {
            "per_family_exact_required": 1.0,
            "depends_on_exactness_required_for": ["EE", "RE"],
            "P_true_comparison": "strictly_greater_than_zero",
            "R_shuffle_max_inclusive": C7_MEMORY_RATIO_MAX,
            "R_mean_max_inclusive": C7_MEMORY_RATIO_MAX,
            "hidden_epsilon": False,
            "rounded_comparison": False,
        },
        "evaluation": {
            "order": list(selection.scaled_family_ids),
            "batch_size": C7_BATCH_SIZE,
            "batch_membership": [list(item) for item in batches],
            "memory_conditions": ["P_true", "P_shuffle", "P_mean"],
            "shuffle": "seed_2026_full_sorted_set_cyclic_derangement",
            "mean": "batch_local_per_latent_token_and_feature",
            "teacher_forced_gate_eligible": False,
        },
        "parameter_inventory": parameter_inventories,
        "runtime": {
            "python": platform.python_version(),
            "pytorch": str(torch.__version__),
            "device": "cpu",
            "cuda_available": bool(torch.cuda.is_available()),
            "host": socket.gethostname(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        },
        "access": tracker.declarations(gate_completed=True),
        "formal_c7_execution": True,
        "c8_or_later_performed": False,
    }
    run_metadata = {
        "event": "run_metadata",
        "pilot_version": C7_PILOT_VERSION,
        "gate_version": C7_GATE_VERSION,
        "artifact_manifest_version": C7_ARTIFACT_MANIFEST_VERSION,
        "source": source,
        "seed": C7_GATE_SEED,
        "access": tracker.declarations(gate_completed=True),
    }
    completion = {
        "event": "c7_pilot_execution_completed",
        **decision,
        "elapsed_seconds": time.perf_counter() - run_started,
    }
    events = [run_metadata] + events + [completion]
    _atomic_write_json(staging / "resolved_config.json", resolved)
    _atomic_write_jsonl(staging / "metrics.jsonl", events)
    _verify_source_unchanged(source, repository_root, expected_commit)
    finalize_c7_artifact(staging, final)
    verified = verify_c7_artifact(final)
    return {
        "artifact_path": str(final),
        "artifact_verification": verified,
        "overall_decision": decision,
        "completion_event": completion,
    }


def prepare_c7_output(output_dir, *, repository_root, corpus_dir):
    """Validate a new external final path and create its incomplete sibling."""

    raw = Path(output_dir)
    if raw.is_symlink() or raw.exists():
        raise GraphEncoderError(
            "c7_output_exists", "C7 output must be a new non-symlink path"
        )
    final = raw.resolve()
    repository = Path(repository_root).resolve()
    corpus = Path(corpus_dir).resolve()
    if _is_within(final, repository) or _is_within(final, corpus):
        raise GraphEncoderError(
            "unsafe_c7_output_location",
            "C7 output must be outside the repository and corpus",
        )
    final.parent.mkdir(parents=True, exist_ok=True)
    job_identity = os.environ.get("SLURM_JOB_ID") or str(os.getpid())
    staging = final.with_name(final.name + ".incomplete-" + job_identity)
    if staging.exists() or staging.is_symlink():
        raise GraphEncoderError(
            "c7_staging_exists", "C7 incomplete staging path already exists"
        )
    staging.mkdir()
    (staging / "checkpoints").mkdir()
    return final, staging


def finalize_c7_artifact(staging_dir, final_dir):
    """Write and verify both integrity layers, then atomically rename."""

    staging = Path(staging_dir)
    final = Path(final_dir)
    if staging.parent.resolve() != final.parent.resolve():
        raise GraphEncoderError(
            "unsafe_c7_output_location",
            "C7 staging and final directories must share one parent",
        )
    if final.exists() or final.is_symlink():
        raise GraphEncoderError("c7_output_exists", "final C7 output exists")
    required = (
        staging / "resolved_config.json",
        staging / "metrics.jsonl",
        staging / "checkpoints",
    )
    if any(not item.exists() for item in required):
        raise GraphEncoderError(
            "incomplete_c7_artifact", "required ordinary artifact is absent"
        )
    ordinary = _regular_artifact_files(
        staging, excluded=("artifact_manifest.json", "SHA256SUMS")
    )
    if not any(path.startswith("checkpoints/") for path in ordinary):
        raise GraphEncoderError(
            "incomplete_c7_artifact", "C7 artifact has no checkpoints"
        )
    manifest = {
        "schema_version": C7_ARTIFACT_MANIFEST_VERSION,
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
    verify_c7_artifact(staging, allow_incomplete_name=True)
    os.replace(str(staging), str(final))
    return final


def verify_c7_artifact(path, *, allow_incomplete_name=False):
    """Independently verify paths, sizes, hashes, JSONL, and final event."""

    root = Path(path)
    if not root.is_dir() or root.is_symlink():
        raise GraphEncoderError(
            "invalid_c7_artifact", "artifact root must be a real directory"
        )
    if ".incomplete-" in root.name and not allow_incomplete_name:
        raise GraphEncoderError(
            "invalid_c7_artifact", "incomplete artifact cannot be final"
        )
    manifest_path = root / "artifact_manifest.json"
    checksums_path = root / "SHA256SUMS"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        checksum_raw = checksums_path.read_bytes()
    except (OSError, ValueError) as exc:
        raise GraphEncoderError(
            "invalid_c7_artifact", "integrity metadata is unreadable"
        ) from exc
    if manifest.get("schema_version") != C7_ARTIFACT_MANIFEST_VERSION:
        raise GraphEncoderError(
            "invalid_c7_artifact", "artifact manifest identity differs"
        )
    rows = manifest.get("artifacts")
    if not isinstance(rows, list):
        raise GraphEncoderError("invalid_c7_artifact", "artifact rows missing")
    listed = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != {
            "path", "byte_size", "sha256"
        }:
            raise GraphEncoderError(
                "invalid_c7_artifact", "artifact manifest row is malformed"
            )
        relative = _safe_relative_path(row["path"])
        target = root / relative
        if not target.is_file() or target.is_symlink():
            raise GraphEncoderError(
                "invalid_c7_artifact", "listed artifact is not a regular file"
            )
        if target.stat().st_size != row["byte_size"]:
            raise GraphEncoderError(
                "c7_artifact_integrity_failure", "artifact byte size differs"
            )
        if _file_sha256(target) != row["sha256"]:
            raise GraphEncoderError(
                "c7_artifact_integrity_failure", "artifact SHA-256 differs"
            )
        listed.append(relative)
    if listed != sorted(listed) or len(listed) != len(set(listed)):
        raise GraphEncoderError(
            "invalid_c7_artifact", "manifest paths must be sorted and unique"
        )
    expected_ordinary = _regular_artifact_files(
        root, excluded=("artifact_manifest.json", "SHA256SUMS")
    )
    if tuple(listed) != expected_ordinary:
        raise GraphEncoderError(
            "invalid_c7_artifact", "manifest does not cover ordinary artifacts"
        )
    if not checksum_raw.endswith(b"\n"):
        raise GraphEncoderError(
            "invalid_c7_artifact", "SHA256SUMS requires a final LF"
        )
    try:
        checksum_lines = checksum_raw.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise GraphEncoderError(
            "invalid_c7_artifact", "SHA256SUMS is not UTF-8"
        ) from exc
    checksum_rows = []
    for line in checksum_lines:
        parts = line.split("  ", 1)
        if len(parts) != 2 or len(parts[0]) != 64:
            raise GraphEncoderError(
                "invalid_c7_artifact", "SHA256SUMS row is malformed"
            )
        relative = _safe_relative_path(parts[1])
        checksum_rows.append(relative)
        if _file_sha256(root / relative) != parts[0]:
            raise GraphEncoderError(
                "c7_artifact_integrity_failure", "SHA256SUMS hash differs"
            )
    expected_checksum_paths = _regular_artifact_files(
        root, excluded=("SHA256SUMS",)
    )
    if tuple(checksum_rows) != expected_checksum_paths:
        raise GraphEncoderError(
            "invalid_c7_artifact", "SHA256SUMS paths differ from regular files"
        )
    events = _read_canonical_jsonl(root / "metrics.jsonl")
    if not events or events[-1].get("event") != "c7_pilot_execution_completed":
        raise GraphEncoderError(
            "invalid_c7_artifact", "final C7 completion event is absent"
        )
    observed_events = {item.get("event") for item in events}
    required_events = {
        "run_metadata",
        "subset_selection",
        "training_started",
        "training_completed",
        "checkpoint_reloaded",
        "autonomous_condition_metrics",
        "gate_result",
        "overall_gate_decision",
        "c7_pilot_execution_completed",
    }
    if not required_events.issubset(observed_events):
        raise GraphEncoderError(
            "invalid_c7_artifact", "required C7 JSONL events are absent"
        )
    try:
        resolved = json.loads(
            (root / "resolved_config.json").read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as exc:
        raise GraphEncoderError(
            "invalid_c7_artifact", "resolved C7 configuration is unreadable"
        ) from exc
    if (
        resolved.get("pilot_version") != C7_PILOT_VERSION
        or resolved.get("gate_version") != C7_GATE_VERSION
        or resolved.get("artifact_manifest_version")
        != C7_ARTIFACT_MANIFEST_VERSION
        or "c7_or_later_performed" in resolved
    ):
        raise GraphEncoderError(
            "invalid_c7_artifact", "resolved C7 identities are malformed"
        )
    terminal = events[-1]
    for name in (
        "overall_gate_pass",
        "stage6_authorized_by_c7",
        "preauthorized_decoder_repair_triggered",
    ):
        if not isinstance(terminal.get(name), bool):
            raise GraphEncoderError(
                "invalid_c7_artifact", "terminal decision field is malformed"
            )
    if terminal["stage6_authorized_by_c7"] is not terminal["overall_gate_pass"]:
        raise GraphEncoderError(
            "invalid_c7_artifact", "Stage 6 authorization differs from C7 pass"
        )
    return {
        "artifact_manifest_sha256": _file_sha256(manifest_path),
        "sha256sums_sha256": _file_sha256(checksums_path),
        "regular_file_count": len(expected_checksum_paths) + 1,
        "overall_gate_pass": terminal["overall_gate_pass"],
        "stage6_authorized_by_c7": terminal["stage6_authorized_by_c7"],
        "preauthorized_decoder_repair_triggered": terminal[
            "preauthorized_decoder_repair_triggered"
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--expected-commit", required=True)
    arguments = parser.parse_args(argv)
    tracker = C7AccessTracker()
    try:
        result = run_pilot(
            corpus_dir=arguments.corpus_dir,
            output_dir=arguments.output_dir,
            repository_root=arguments.repository_root,
            expected_commit=arguments.expected_commit,
            access_tracker=tracker,
        )
    except Exception as exc:
        code = getattr(exc, "code", type(exc).__name__)
        failure = {
            "event": "c7_pilot_infrastructure_failure",
            "code": code,
            "detail": str(exc),
            **tracker.declarations(gate_completed=False),
        }
        print(_canonical_json_text(failure), file=sys.stderr, flush=True)
        return 1
    terminal = {
        "event": "c7_pilot_execution_completed",
        "artifact_path": result["artifact_path"],
        **result["overall_decision"],
        **tracker.declarations(gate_completed=True),
    }
    print(_canonical_json_text(terminal), flush=True)
    return 0


def _train_evaluate_arm(
    *, model, examples, subset_identity, staging, repository_root,
    expected_commit, training_config, parameter_counts, events, tracker
):
    from .autonomous import autonomous_input_from_paired, run_autonomous_evaluation
    from .batching import build_paired_batch
    from .model import build_ge1_model
    from .provenance import training_partition_identity
    from .training import load_training_checkpoint, run_ge1_training
    from .metrics import score_condition

    arm = model.config.encoder
    run_start = time.perf_counter()
    checkpoint_dir = staging / "checkpoints" / subset_identity / arm
    events.append({
        "event": "training_started",
        "arm": arm,
        "subset_identity": subset_identity,
        "seed": C7_GATE_SEED,
        "checkpoint_role": C7_CHECKPOINT_ROLE,
        "expected_arithmetic": c7_training_arithmetic(len(examples)),
        "access": tracker.declarations(),
    })
    training = run_ge1_training(
        model,
        examples,
        training_config=training_config,
        checkpoint_directory=checkpoint_dir,
        repository_root=repository_root,
        expected_commit=expected_commit,
        final_epoch=C7_CHECKPOINT_EPOCH,
        engineering_smoke=False,
    )
    arithmetic = c7_training_arithmetic(len(examples))
    if (
        training.completed_epoch != C7_CHECKPOINT_EPOCH
        or training.optimizer_steps != arithmetic["optimizer_steps"]
        or training.training_example_presentations
        != arithmetic["training_example_presentations"]
        or len(training.checkpoint_paths) != C7_CHECKPOINT_EPOCH
        or training.selected_experimental_checkpoint
        != training.checkpoint_paths[-1]
    ):
        raise GraphEncoderError(
            "c7_training_arithmetic_failure",
            "completed training differs from the frozen C7 arithmetic",
        )
    events.append({
        "event": "training_completed",
        "arm": arm,
        "subset_identity": subset_identity,
        "checkpoint_role": C7_CHECKPOINT_ROLE,
        "training": _training_record_for_artifact(training, staging=staging),
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
    selected_path = Path(training.checkpoint_paths[-1])
    resume = load_training_checkpoint(
        selected_path,
        model=reloaded,
        optimizer=optimizer,
        training_config=training_config,
        partition_identity=partition_identity,
        expected_code_revision=expected_commit,
        restore_rng=True,
    )
    validate_c7_reloaded_checkpoint(resume)
    _assert_model_states_equal(model, reloaded)
    relative_checkpoint = selected_path.relative_to(staging).as_posix()
    events.append({
        "event": "checkpoint_reloaded",
        "arm": arm,
        "subset_identity": subset_identity,
        "epoch": C7_CHECKPOINT_EPOCH,
        "checkpoint_identity": relative_checkpoint,
        "checkpoint_role": C7_CHECKPOINT_ROLE,
        "strict_reload": True,
        "access": tracker.declarations(),
    })

    ordered = tuple(sorted(examples, key=lambda item: item.physical_family_id))
    input_batches = []
    for start in range(0, len(ordered), C7_BATCH_SIZE):
        paired = build_paired_batch(ordered[start:start + C7_BATCH_SIZE])
        input_batches.append(autonomous_input_from_paired(paired, arm))
    autonomous = run_autonomous_evaluation(
        reloaded, tuple(input_batches), seed=C7_GATE_SEED
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
            "event": "autonomous_condition_metrics",
            "arm": arm,
            "subset_identity": subset_identity,
            "condition": condition.condition,
            "metrics": metric,
            "memory_assignments": [list(item) for item in condition.memory_assignments],
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
        epoch=C7_CHECKPOINT_EPOCH,
        access_flags=tracker.declarations(),
    )
    events.append({"event": "gate_result", **exact})
    memory = None
    if subset_identity == "scaled":
        memory = memory_use_gate(
            arm=arm,
            condition_metrics=metrics,
            autonomous_result=autonomous,
            templates_by_family=templates,
            checkpoint_identity=relative_checkpoint,
            epoch=C7_CHECKPOINT_EPOCH,
            access_flags=tracker.declarations(),
        )
        events.append({"event": "gate_result", **memory})
        envelope = c7_metrics_envelope(
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
            "event": "scaled_c7_metrics_envelope",
            "record": envelope,
        })
    return {"exact_gate": exact, "memory_gate": memory}


def _load_selected_train(corpus_dir, family_ids, tracker):
    if tracker.operation_template_train_payload_accessed is not True:
        tracker.operation_template_train_payload_accessed = (
            "not_confirmed_on_failure"
        )
    examples = tuple(load_train(corpus_dir, family_ids))
    tracker.operation_template_train_payload_accessed = True
    observed = tuple(sorted(item.physical_family_id for item in examples))
    if observed != tuple(family_ids):
        raise GraphEncoderError(
            "c7_payload_alignment_failure",
            "loaded train families differ from metadata selection",
        )
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
            None if selected is None
            else Path(selected).relative_to(root).as_posix()
        )
    record["checkpoint_role"] = C7_CHECKPOINT_ROLE
    record["full_comparison_checkpoint"] = False
    return record


def _validate_scaled_batch_membership(autonomous_result, family_ids):
    true = next(
        (item for item in autonomous_result.conditions if item.condition == "P_true"),
        None,
    )
    if true is None:
        raise GraphEncoderError(
            "malformed_c7_autonomous_result", "P_true condition is absent"
        )
    membership = tuple(sorted(
        (tuple(families) for unused_identity, families in true.batch_membership),
        key=lambda item: item,
    ))
    expected = tuple(
        family_ids[start:start + C7_BATCH_SIZE]
        for start in range(0, len(family_ids), C7_BATCH_SIZE)
    )
    if membership != expected or len(membership) != 4 or any(
        len(item) != 8 for item in membership
    ):
        raise GraphEncoderError(
            "invalid_c7_batch_membership",
            "scaled evaluation must be four sorted batches of eight",
        )
    return [list(item) for item in membership]


def _memory_criterion(value, *, comparison, threshold):
    numeric = (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )
    if not numeric:
        reason = "unavailable_or_malformed"
        if isinstance(value, dict) and isinstance(value.get("reason"), str):
            reason = value["reason"]
        return {
            "observed_value": value,
            "comparison": comparison,
            "threshold": threshold,
            "passed": False,
            "failure_reason": reason,
            "rounded_for_comparison": False,
            "epsilon_added": False,
        }
    observed = float(value)
    if comparison == "greater_than":
        passed = observed > threshold
    elif comparison == "less_than_or_equal":
        passed = observed <= threshold
    else:
        raise ValueError("unknown memory comparison")
    return {
        "observed_value": observed,
        "comparison": comparison,
        "threshold": threshold,
        "passed": passed,
        "failure_reason": None if passed else "threshold_not_satisfied",
        "rounded_for_comparison": False,
        "epsilon_added": False,
    }


def _finite_metric(value, name):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise GraphEncoderError(
            "malformed_c7_metrics", "{} must be finite numeric".format(name)
        )
    return float(value)


def _validate_arm_subset(arm, subset_identity):
    if arm not in C7_ARMS:
        raise GraphEncoderError("invalid_c7_arm", "unknown C7 arm")
    if subset_identity not in C7_SUBSETS:
        raise GraphEncoderError("invalid_c7_subset", "unknown C7 subset")


def _assert_matched_disjoint(flat_model, graph_model):
    flat_state = flat_model.decoder.state_dict()
    graph_state = graph_model.decoder.state_dict()
    if list(flat_state) != list(graph_state):
        raise GraphEncoderError(
            "c7_model_lifecycle_failure", "shared decoder keys differ"
        )
    for name in flat_state:
        if not torch.equal(flat_state[name], graph_state[name]):
            raise GraphEncoderError(
                "c7_model_lifecycle_failure",
                "shared decoder initialization differs at {}".format(name),
            )
    if {id(item) for item in flat_model.parameters()} & {
        id(item) for item in graph_model.parameters()
    }:
        raise GraphEncoderError(
            "c7_model_lifecycle_failure", "matched arms share mutable parameters"
        )


def _assert_model_collections_disjoint(previous, current):
    old = {id(item) for model in previous for item in model.parameters()}
    new = {id(item) for model in current for item in model.parameters()}
    if old & new:
        raise GraphEncoderError(
            "c7_model_lifecycle_failure", "tiny and scaled models share parameters"
        )


def _assert_model_states_equal(expected, observed):
    expected_state = expected.state_dict()
    observed_state = observed.state_dict()
    if list(expected_state) != list(observed_state):
        raise GraphEncoderError(
            "c7_checkpoint_reload_failure", "strict reload keys differ"
        )
    for name in expected_state:
        if not torch.equal(expected_state[name], observed_state[name]):
            raise GraphEncoderError(
                "c7_checkpoint_reload_failure",
                "strict reload tensor differs at {}".format(name),
            )


def _source_identity(repository_root, expected_commit):
    root = Path(repository_root).resolve()
    actual_root = _git(root, ("rev-parse", "--show-toplevel"))
    if Path(actual_root).resolve() != root:
        raise GraphEncoderError(
            "wrong_repository_root", "repository root is not the Git top level"
        )
    commit = _git(root, ("rev-parse", "HEAD"))
    status = _git(
        root,
        ("status", "--porcelain=v1", "--untracked-files=all"),
        allow_empty=True,
    )
    if commit != expected_commit:
        raise GraphEncoderError("wrong_git_commit", "checked-out commit differs")
    if status:
        raise GraphEncoderError("dirty_source_tree", "C7 requires clean source")
    branch = _git(
        root, ("symbolic-ref", "--quiet", "--short", "HEAD"),
        allow_failure=True,
    )
    return {
        "git_commit": commit,
        "git_branch": branch or None,
        "detached_head": not bool(branch),
        "git_dirty": False,
        "git_status_porcelain": [],
        "source_tree_sha256": source_tree_sha256(root),
    }


def _verify_source_unchanged(initial, repository_root, expected_commit):
    current = _source_identity(repository_root, expected_commit)
    if current != initial:
        raise GraphEncoderError(
            "source_changed_during_c7", "source identity changed during execution"
        )


def _git(root, arguments, allow_empty=False, allow_failure=False):
    result = subprocess.run(
        ("git",) + tuple(arguments),
        cwd=str(root),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    if allow_failure and result.returncode != 0:
        return ""
    value = result.stdout.rstrip("\r\n") if allow_empty else result.stdout.strip()
    if result.returncode != 0 or (not allow_empty and not value):
        raise GraphEncoderError("git_provenance_failure", "Git command failed")
    return value


def _require_authoritative_runtime():
    if torch is None:
        raise RuntimeError("formal C7 execution requires PyTorch")
    if sys.version_info[:3] != (3, 8, 13):
        raise GraphEncoderError(
            "environment_mismatch", "formal C7 requires Python 3.8.13"
        )
    if str(torch.__version__).split("+")[0] != "1.11.0":
        raise GraphEncoderError(
            "environment_mismatch", "formal C7 requires PyTorch 1.11.0"
        )
    if torch.cuda.is_available():
        raise GraphEncoderError(
            "environment_mismatch", "formal C7 must execute on CPU"
        )


def _atomic_write_json(path, value):
    _atomic_write_bytes(path, _canonical_json_bytes(value))


def _atomic_write_jsonl(path, values):
    payload = b"".join(_canonical_json_bytes(value) for value in values)
    _atomic_write_bytes(path, payload)


def _atomic_write_bytes(path, payload):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix="." + destination.name + ".tmp-", dir=str(destination.parent)
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, str(destination))
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def _canonical_json_bytes(value):
    try:
        text = _canonical_json_text(value)
    except (TypeError, ValueError) as exc:
        raise GraphEncoderError(
            "noncanonical_c7_json", "C7 JSON must be finite and serializable"
        ) from exc
    return (text + "\n").encode("utf-8")


def _canonical_json_text(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def _read_canonical_jsonl(path):
    try:
        raw = Path(path).read_bytes()
    except OSError as exc:
        raise GraphEncoderError(
            "invalid_c7_artifact", "metrics JSONL is unreadable"
        ) from exc
    if not raw.endswith(b"\n"):
        raise GraphEncoderError(
            "invalid_c7_artifact", "metrics JSONL requires a final LF"
        )
    events = []
    for line in raw.splitlines():
        try:
            event = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise GraphEncoderError(
                "invalid_c7_artifact", "metrics JSONL row is invalid"
            ) from exc
        if _canonical_json_bytes(event).rstrip(b"\n") != line:
            raise GraphEncoderError(
                "invalid_c7_artifact", "metrics JSONL row is not canonical"
            )
        events.append(event)
    return events


def _regular_artifact_files(root, *, excluded):
    result = []
    for path in Path(root).rglob("*"):
        if path.is_symlink():
            raise GraphEncoderError(
                "invalid_c7_artifact", "artifact may not contain symlinks"
            )
        if path.is_file():
            relative = path.relative_to(root).as_posix()
            if relative not in excluded:
                _safe_relative_path(relative)
                result.append(relative)
    return tuple(sorted(result))


def _safe_relative_path(value):
    if not isinstance(value, str) or not value:
        raise GraphEncoderError("invalid_c7_artifact", "artifact path is empty")
    candidate = Path(value)
    if candidate.is_absolute() or ".." in candidate.parts or candidate.as_posix() != value:
        raise GraphEncoderError(
            "invalid_c7_artifact", "artifact path is not a safe relative path"
        )
    if value in ("artifact_manifest.json", "SHA256SUMS"):
        return value
    return value


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_within(path, parent):
    try:
        Path(path).relative_to(parent)
    except ValueError:
        return False
    return True


if __name__ == "__main__":
    raise SystemExit(main())
