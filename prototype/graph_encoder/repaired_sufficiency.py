"""Prospective repaired GE1 train-sufficiency scientific protocol.

This module is deliberately separate from :mod:`c7_v2`. It reuses stable
training, autonomous, metric, and artifact primitives without changing the
immutable C7-v2 identities or behavior.
"""

from __future__ import annotations

import argparse
import copy
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
except ImportError:  # Pure protocol and artifact tests remain torch-free.
    torch = None

from . import c7_v2
from .decoder_contract import POSITIVE_OPERATION_MAGNITUDE_PARAMETERIZATION
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
from .operation_fidelity import (
    OPERATION_GEOMETRY_FIDELITY_VERSION,
    operation_fidelity_contract,
    operation_geometry_fidelity_gate,
)
from .optimization_diagnostic import validate_slurm_job_id
from .partitions import (
    AUTHORITATIVE_FILE_SHA256,
    C7_ACCESSIBLE_TEMPLATES,
    load_train,
    select_c7_sufficiency_subsets,
)
from .pilot import (
    _assert_matched_disjoint,
    _assert_model_collections_disjoint,
    _assert_model_states_equal,
    _atomic_write_bytes,
    _atomic_write_json,
    _atomic_write_jsonl,
    _file_sha256,
    _is_within,
    _read_canonical_jsonl,
    _regular_artifact_files,
    _safe_relative_path,
    _source_identity,
    _verify_source_unchanged,
)


REPAIRED_PROTOCOL_VERSION = "GE1-C7-REPAIRED-SUFFICIENCY-v1"
REPAIRED_METRICS_VERSION = "GE1-C7-REPAIRED-METRICS-v1"
REPAIRED_GATE_VERSION = "GE1-C7-REPAIRED-GATES-v1"
REPAIRED_ARTIFACT_VERSION = "GE1-C7-REPAIRED-ARTIFACT-v1"
REPAIRED_CHECKPOINT_ROLE = (
    "ge1_c7_repaired_sufficiency_only_not_full_comparison"
)
REPAIRED_SEED = 2026
REPAIRED_CHECKPOINT_EPOCH = 200
REPAIRED_BATCH_SIZE = 8
REPAIRED_MEMORY_RATIO_MAX = 0.80
REPAIRED_ARMS = ("flat", "typed_graph")
REPAIRED_SUBSETS = ("tiny", "scaled")
REPAIRED_GATE_STATUSES = ("pass", "fail", "not_run")
REPAIRED_REQUIRED_GATE_NAMES = tuple(
    "{}.{}".format(arm, gate)
    for arm in REPAIRED_ARMS
    for gate in (
        "tiny_exact_sufficiency",
        "tiny_operation_geometry_fidelity",
        "scaled_exact_sufficiency",
        "scaled_operation_geometry_fidelity",
        "scaled_memory_use",
    )
)
REPAIRED_PROTECTED_ACCESS_FIELDS = (
    "development_accessed",
    "systematic_rr_accessed",
    "test_er_accessed",
    "iid_accessed",
    "history_depth_accessed",
    "geometry_extrapolation_accessed",
    "other_corpus_accessed",
)
IMMUTABLE_C7_V2_COMMIT = "e325d5ad97957c08da4a19b4261560e8a4a472a4"
IMMUTABLE_C7_V2_JOB = "3344981"
REPAIR_VALIDATION_COMMIT = "12167ce7d0dc025c4b297b7ccb8a3011580bf0fc"
REPAIR_VALIDATION_JOB = "3345044"


@dataclass(frozen=True)
class RepairedSufficiencyConfiguration:
    """Exact prospective budget and nonadaptive execution choices."""

    epochs: int = REPAIRED_CHECKPOINT_EPOCH
    checkpoint_epoch: int = REPAIRED_CHECKPOINT_EPOCH
    batch_size: int = REPAIRED_BATCH_SIZE
    seed: int = REPAIRED_SEED
    operation_magnitude_parameterization: str = (
        POSITIVE_OPERATION_MAGNITUDE_PARAMETERIZATION
    )
    early_stopping: bool = False
    best_checkpoint_selection: bool = False
    outcome_dependent_extension: bool = False
    warm_start: bool = False
    checkpoint_reuse: bool = False

    def validate(self):
        if self != RepairedSufficiencyConfiguration():
            raise GraphEncoderError(
                "invalid_repaired_sufficiency_configuration",
                "configuration differs from accepted ADR-0010",
            )
        return self

    def to_dict(self):
        self.validate()
        return {
            "protocol_version": REPAIRED_PROTOCOL_VERSION,
            "metrics_version": REPAIRED_METRICS_VERSION,
            "gate_version": REPAIRED_GATE_VERSION,
            "artifact_version": REPAIRED_ARTIFACT_VERSION,
            "epochs": self.epochs,
            "checkpoint_epoch": self.checkpoint_epoch,
            "checkpoint_selection": "fixed_epoch_200",
            "batch_size": self.batch_size,
            "seed": self.seed,
            "operation_magnitude_parameterization": (
                self.operation_magnitude_parameterization
            ),
            "operation_geometry_fidelity": operation_fidelity_contract(),
            "early_stopping": self.early_stopping,
            "best_checkpoint_selection": self.best_checkpoint_selection,
            "outcome_dependent_extension": (
                self.outcome_dependent_extension
            ),
            "warm_start": self.warm_start,
            "checkpoint_reuse": self.checkpoint_reuse,
        }


@dataclass
class RepairedSufficiencyAccessTracker:
    operation_template_manifest_accessed: object = False
    operation_template_train_payload_accessed: object = False
    tiny_train_payload_accessed: object = False
    scaled_train_payload_accessed: object = False

    def declarations(self, *, completed=False):
        values = {
            "operation_template_manifest_accessed": (
                self.operation_template_manifest_accessed
            ),
            "operation_template_train_payload_accessed": (
                self.operation_template_train_payload_accessed
            ),
            "tiny_train_payload_accessed": self.tiny_train_payload_accessed,
            "scaled_train_payload_accessed": self.scaled_train_payload_accessed,
            "repaired_sufficiency_execution_completed": bool(completed),
            "positive_magnitude_repair_invoked": True,
            "additional_repair_implemented_or_invoked": False,
            "cad_kernel_available": False,
            "cad_kernel_used": False,
            "stage6_performed": False,
            "c8_or_later_performed": False,
        }
        values.update({name: False for name in REPAIRED_PROTECTED_ACCESS_FIELDS})
        return values


def repaired_training_arithmetic(family_count):
    """Return exact accepted tiny/scaled epoch-200 exposure arithmetic."""

    if family_count not in (4, 32):
        raise GraphEncoderError(
            "invalid_repaired_subset_size",
            "repaired sufficiency requires 4 or 32 physical families",
        )
    steps_per_epoch = int(math.ceil(family_count / float(REPAIRED_BATCH_SIZE)))
    return {
        "family_count": family_count,
        "epochs": REPAIRED_CHECKPOINT_EPOCH,
        "batch_size": REPAIRED_BATCH_SIZE,
        "steps_per_epoch": steps_per_epoch,
        "optimizer_steps": steps_per_epoch * REPAIRED_CHECKPOINT_EPOCH,
        "training_example_presentations": (
            family_count * REPAIRED_CHECKPOINT_EPOCH
        ),
    }


def validate_repaired_selection(selection):
    """Reuse the frozen C7 cohorts while applying a new protocol identity."""

    source = c7_v2.validate_frozen_c7_v2_selection(selection)
    record = copy.deepcopy(source)
    record.update({
        "protocol_version": REPAIRED_PROTOCOL_VERSION,
        "source_selection_protocol": c7_v2.C7_V2_PROTOCOL_VERSION,
        "immutable_c7_v2_result_changed": False,
        "cohorts_reused_without_reselection": True,
    })
    return record


def validate_repaired_reloaded_checkpoint(
    resume_state, *, expected_commit, job_id
):
    """Require strict epoch-200 recovery and the repaired identity."""

    c7_v2.validate_c7_v2_reloaded_checkpoint(
        resume_state, expected_commit=expected_commit, job_id=job_id
    )
    payload = getattr(resume_state, "payload", None)
    model_config = payload.get("model_config", {}) if isinstance(payload, dict) else {}
    provenance = payload.get("provenance", {}) if isinstance(payload, dict) else {}
    if (
        model_config.get("operation_magnitude_parameterization")
        != POSITIVE_OPERATION_MAGNITUDE_PARAMETERIZATION
        or provenance.get("operation_magnitude_parameterization")
        != POSITIVE_OPERATION_MAGNITUDE_PARAMETERIZATION
    ):
        raise GraphEncoderError(
            "invalid_repaired_training_checkpoint",
            "recovery checkpoint lacks the repaired parameterization identity",
        )
    return resume_state


def repaired_exact_sufficiency_gate(**kwargs):
    """Apply the unchanged C7-v2 exact criteria under a new identity."""

    base = c7_v2.exact_sufficiency_gate(**kwargs)
    record = copy.deepcopy(base)
    record.update({
        "version": REPAIRED_PROTOCOL_VERSION,
        "gate_version": REPAIRED_GATE_VERSION,
        "checkpoint_role": REPAIRED_CHECKPOINT_ROLE,
        "operation_magnitude_parameterization": (
            POSITIVE_OPERATION_MAGNITUDE_PARAMETERIZATION
        ),
        "source_exact_gate_contract": c7_v2.C7_V2_PROTOCOL_VERSION,
        "analytic_complete_validity_source_field": (
            "complete_executable_validity"
        ),
        "cad_kernel_required_or_used": False,
    })
    family_values = kwargs["condition_metrics"].get("family_values", {})
    record["first_failure_stage_by_family"] = {
        family_id: family_values[family_id].get("stage_of_first_failure")
        for family_id in record["failing_family_ids"]
    }
    return record


def repaired_memory_use_gate(**kwargs):
    """Apply the unchanged C7-v2 memory gate under a new identity."""

    base = c7_v2.memory_use_gate(**kwargs)
    record = copy.deepcopy(base)
    record.update({
        "version": REPAIRED_PROTOCOL_VERSION,
        "gate_version": REPAIRED_GATE_VERSION,
        "checkpoint_role": REPAIRED_CHECKPOINT_ROLE,
        "operation_magnitude_parameterization": (
            POSITIVE_OPERATION_MAGNITUDE_PARAMETERIZATION
        ),
        "source_memory_gate_contract": c7_v2.C7_V2_PROTOCOL_VERSION,
    })
    return record


def repaired_not_run_gate(*, arm, gate_name, reason, access_flags):
    """Create one dependent scaled gate that cannot pass."""

    if arm not in REPAIRED_ARMS:
        raise GraphEncoderError(
            "invalid_repaired_sufficiency_arm", "unknown encoder arm"
        )
    if gate_name not in (
        "scaled_exact_sufficiency",
        "scaled_operation_geometry_fidelity",
        "scaled_memory_use",
    ):
        raise GraphEncoderError(
            "invalid_repaired_sufficiency_gate", "invalid not-run gate"
        )
    if not isinstance(reason, str) or not reason:
        raise GraphEncoderError(
            "invalid_repaired_sufficiency_gate", "not_run requires a reason"
        )
    return {
        "version": REPAIRED_PROTOCOL_VERSION,
        "gate_version": REPAIRED_GATE_VERSION,
        "gate_name": gate_name,
        "status": "not_run",
        "arm": arm,
        "seed": REPAIRED_SEED,
        "subset_identity": "scaled",
        "physical_family_count": 32,
        "checkpoint_epoch": None,
        "checkpoint_identity": None,
        "checkpoint_role": REPAIRED_CHECKPOINT_ROLE,
        "operation_magnitude_parameterization": (
            POSITIVE_OPERATION_MAGNITUDE_PARAMETERIZATION
        ),
        "criteria": {},
        "failing_family_ids": [],
        "failed_fields_by_family": {},
        "structured_reason": reason,
        "access": dict(access_flags),
    }


def overall_repaired_sufficiency_decision(
    gates, *, infrastructure_failure=False
):
    """Return repaired-protocol decisions without authorizing later work."""

    if infrastructure_failure:
        return {
            "version": REPAIRED_PROTOCOL_VERSION,
            "infrastructure_completed": False,
            "repaired_scientific_gate_pass": False,
            "repaired_scientific_failure": False,
            "stage6_authorized_by_repaired_sufficiency": False,
            "comparison_inconclusive": False,
            "scaled_authorized": False,
            "cad_kernel_available": False,
            "further_intervention_authorized": False,
            "repaired_sufficiency_execution_completed": False,
            "structured_reason": "infrastructure_failure",
        }
    by_name = dict(gates)
    if set(by_name) != set(REPAIRED_REQUIRED_GATE_NAMES):
        raise GraphEncoderError(
            "incomplete_repaired_sufficiency_gate_set",
            "decision requires all ten arm gates",
        )
    if any(
        not isinstance(gate, dict)
        or gate.get("status") not in REPAIRED_GATE_STATUSES
        for gate in by_name.values()
    ):
        raise GraphEncoderError(
            "malformed_repaired_sufficiency_gate",
            "gate has an invalid status",
        )
    overall = all(
        by_name[name]["status"] == "pass"
        for name in REPAIRED_REQUIRED_GATE_NAMES
    )
    exact_or_fidelity_names = tuple(
        name
        for name in REPAIRED_REQUIRED_GATE_NAMES
        if "exact_sufficiency" in name
        or "operation_geometry_fidelity" in name
    )
    scientific_failure = any(
        by_name[name]["status"] == "fail"
        for name in exact_or_fidelity_names
    )
    memory_failure = any(
        by_name["{}.scaled_memory_use".format(arm)]["status"] == "fail"
        for arm in REPAIRED_ARMS
    )
    inconclusive = bool(not overall and not scientific_failure and memory_failure)
    scaled_authorized = all(
        by_name["{}.tiny_exact_sufficiency".format(arm)]["status"] == "pass"
        and by_name[
            "{}.tiny_operation_geometry_fidelity".format(arm)
        ]["status"] == "pass"
        for arm in REPAIRED_ARMS
    )
    return {
        "version": REPAIRED_PROTOCOL_VERSION,
        "infrastructure_completed": True,
        "repaired_scientific_gate_pass": overall,
        "repaired_scientific_failure": scientific_failure,
        "stage6_authorized_by_repaired_sufficiency": overall,
        "comparison_inconclusive": inconclusive,
        "scaled_authorized": scaled_authorized,
        "cad_kernel_available": False,
        "further_intervention_authorized": False,
        "repaired_sufficiency_execution_completed": True,
        "structured_reason": (
            "all_ten_repaired_arm_gates_passed"
            if overall
            else (
                "exact_or_operation_geometry_fidelity_failure"
                if scientific_failure
                else "memory_only_failure_or_dependent_gate_not_run"
            )
        ),
    }


def repaired_metrics_envelope(
    *,
    arm,
    checkpoint_identities,
    condition_metrics,
    autonomous_result,
    templates_by_family,
    training_result,
    parameter_counts,
    exact_gate,
    fidelity_gate,
    memory_gate,
    access_flags,
    total_run_seconds,
    staging=None,
):
    """Build a separately versioned scaled metrics record."""

    ratios = intervention_ratios(condition_metrics)
    record = {
        "schema_version": REPAIRED_METRICS_VERSION,
        "protocol_version": REPAIRED_PROTOCOL_VERSION,
        "gate_version": REPAIRED_GATE_VERSION,
        "source_condition_metrics_schema": METRICS_SCHEMA_VERSION,
        "primary_reporting_contract": PRIMARY_REPORTING_CONTRACT_VERSION,
        "operation_fidelity_contract": OPERATION_GEOMETRY_FIDELITY_VERSION,
        "operation_magnitude_parameterization": (
            POSITIVE_OPERATION_MAGNITUDE_PARAMETERIZATION
        ),
        "arm": arm,
        "seed": REPAIRED_SEED,
        "epoch": REPAIRED_CHECKPOINT_EPOCH,
        "checkpoint_identities": dict(checkpoint_identities),
        "checkpoint_role": REPAIRED_CHECKPOINT_ROLE,
        "scientific_result": True,
        "immutable_c7_v2_result_changed": False,
        "repair_engineering_validation_result_changed": False,
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
        "operation_geometry_fidelity_gate": fidelity_gate,
        "memory_use_gate": memory_gate,
        "other_geometry_metrics_report_only": True,
        "access": dict(access_flags),
    }
    return validate_primary_report(record)


def run_repaired_sufficiency(
    *, corpus_dir, output_dir, repository_root, expected_commit,
    access_tracker=None
):
    """Run the prospective protocol and atomically publish its artifact."""

    job_id = _require_authoritative_runtime()
    tracker = access_tracker or RepairedSufficiencyAccessTracker()
    source = _source_identity(repository_root, expected_commit)
    final, staging = prepare_repaired_output(
        output_dir,
        repository_root=repository_root,
        corpus_dir=corpus_dir,
        job_id=job_id,
    )
    events = []
    started = time.perf_counter()
    tracker.operation_template_manifest_accessed = "not_confirmed_on_failure"
    selection = select_c7_sufficiency_subsets(corpus_dir)
    tracker.operation_template_manifest_accessed = True
    selection_record = validate_repaired_selection(selection)
    events.append({
        "event": "repaired_sufficiency_subset_selection",
        **selection_record,
    })

    from .config import GE1TrainingConfig
    from .model import build_matched_ge1_models

    protocol_config = RepairedSufficiencyConfiguration().validate()
    training_config = GE1TrainingConfig()
    training_config.validate()
    gates = {}
    parameter_inventories = {}
    model_configurations = {}

    tiny_examples = _load_selected_train(
        corpus_dir, selection.tiny_family_ids, tracker,
        subset_identity="tiny"
    )
    tiny_models = build_matched_ge1_models(seed=REPAIRED_SEED)
    _assert_repaired_matched_pair(tiny_models)
    parameter_inventories["tiny"] = parameter_count_record(*tiny_models)
    model_configurations["tiny"] = {
        model.config.encoder: model.config.to_dict() for model in tiny_models
    }
    for arm, model in zip(REPAIRED_ARMS, tiny_models):
        result = _train_evaluate_arm(
            model=model,
            examples=tiny_examples,
            subset_identity="tiny",
            staging=staging,
            repository_root=repository_root,
            expected_commit=expected_commit,
            job_id=job_id,
            training_config=training_config,
            parameter_counts=parameter_inventories["tiny"],
            events=events,
            tracker=tracker,
        )
        gates["{}.tiny_exact_sufficiency".format(arm)] = result[
            "exact_gate"
        ]
        gates[
            "{}.tiny_operation_geometry_fidelity".format(arm)
        ] = result["fidelity_gate"]

    both_tiny_pass = all(
        gates["{}.tiny_exact_sufficiency".format(arm)]["status"] == "pass"
        and gates[
            "{}.tiny_operation_geometry_fidelity".format(arm)
        ]["status"] == "pass"
        for arm in REPAIRED_ARMS
    )
    if both_tiny_pass:
        scaled_examples = _load_selected_train(
            corpus_dir,
            selection.scaled_family_ids,
            tracker,
            subset_identity="scaled",
        )
        scaled_models = build_matched_ge1_models(seed=REPAIRED_SEED)
        _assert_repaired_matched_pair(scaled_models)
        _assert_model_collections_disjoint(tiny_models, scaled_models)
        parameter_inventories["scaled"] = parameter_count_record(*scaled_models)
        if parameter_inventories["scaled"] != parameter_inventories["tiny"]:
            raise GraphEncoderError(
                "repaired_model_lifecycle_failure",
                "fresh tiny and scaled pairs differ in capacity",
            )
        model_configurations["scaled"] = {
            model.config.encoder: model.config.to_dict()
            for model in scaled_models
        }
        for arm, model in zip(REPAIRED_ARMS, scaled_models):
            result = _train_evaluate_arm(
                model=model,
                examples=scaled_examples,
                subset_identity="scaled",
                staging=staging,
                repository_root=repository_root,
                expected_commit=expected_commit,
                job_id=job_id,
                training_config=training_config,
                parameter_counts=parameter_inventories["scaled"],
                events=events,
                tracker=tracker,
            )
            gates["{}.scaled_exact_sufficiency".format(arm)] = result[
                "exact_gate"
            ]
            gates[
                "{}.scaled_operation_geometry_fidelity".format(arm)
            ] = result["fidelity_gate"]
            gates["{}.scaled_memory_use".format(arm)] = result["memory_gate"]
    else:
        if tracker.scaled_train_payload_accessed is not False:
            raise GraphEncoderError(
                "repaired_access_dependency_failure",
                "scaled payload state changed before both tiny arms passed",
            )
        for arm in REPAIRED_ARMS:
            for gate_name in (
                "scaled_exact_sufficiency",
                "scaled_operation_geometry_fidelity",
                "scaled_memory_use",
            ):
                gate = repaired_not_run_gate(
                    arm=arm,
                    gate_name=gate_name,
                    reason=(
                        "both_tiny_exact_and_operation_fidelity_gates_"
                        "did_not_pass"
                    ),
                    access_flags=tracker.declarations(),
                )
                gates["{}.{}".format(arm, gate_name)] = gate
                events.append({
                    "event": "repaired_sufficiency_gate_result", **gate
                })

    decision = overall_repaired_sufficiency_decision(gates)
    decision.update(tracker.declarations(completed=True))
    events.append({
        "event": "repaired_sufficiency_overall_gate_decision", **decision
    })
    resolved = _resolved_configuration(
        source=source,
        selection_record=selection_record,
        protocol_config=protocol_config,
        training_config=training_config,
        model_configurations=model_configurations,
        parameter_inventories=parameter_inventories,
        tracker=tracker,
        job_id=job_id,
        both_tiny_pass=both_tiny_pass,
        scaled_family_ids=selection.scaled_family_ids,
    )
    run_metadata = {
        "event": "repaired_sufficiency_run_metadata",
        "protocol_version": REPAIRED_PROTOCOL_VERSION,
        "metrics_version": REPAIRED_METRICS_VERSION,
        "gate_version": REPAIRED_GATE_VERSION,
        "artifact_version": REPAIRED_ARTIFACT_VERSION,
        "operation_magnitude_parameterization": (
            POSITIVE_OPERATION_MAGNITUDE_PARAMETERIZATION
        ),
        "source": source,
        "seed": REPAIRED_SEED,
        "slurm_job_id": job_id,
        "access": tracker.declarations(completed=True),
    }
    completion = {
        "event": "repaired_sufficiency_execution_completed",
        "protocol_version": REPAIRED_PROTOCOL_VERSION,
        "operation_magnitude_parameterization": (
            POSITIVE_OPERATION_MAGNITUDE_PARAMETERIZATION
        ),
        "slurm_job_id": job_id,
        **decision,
        "elapsed_seconds": time.perf_counter() - started,
    }
    events = [run_metadata] + events + [completion]
    _atomic_write_json(staging / "resolved_config.json", resolved)
    _atomic_write_jsonl(staging / "metrics.jsonl", events)
    _verify_source_unchanged(source, repository_root, expected_commit)
    finalize_repaired_artifact(staging, final)
    verified = verify_repaired_artifact(
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
    """Train, strictly reload, generate autonomously, then introduce targets."""

    from .autonomous import autonomous_input_from_paired, run_autonomous_evaluation
    from .batching import build_paired_batch
    from .checkpoint import load_ge1_checkpoint, save_ge1_checkpoint
    from .metrics import score_condition
    from .model import build_ge1_model
    from .provenance import training_partition_identity
    from .training import load_training_checkpoint, run_ge1_training

    if (
        model.config.operation_magnitude_parameterization
        != POSITIVE_OPERATION_MAGNITUDE_PARAMETERIZATION
    ):
        raise GraphEncoderError(
            "legacy_checkpoint_ineligible",
            "repaired protocol requires the positive parameterization",
        )
    arm = model.config.encoder
    run_start = time.perf_counter()
    checkpoint_dir = staging / "checkpoints" / subset_identity / arm
    events.append({
        "event": "repaired_sufficiency_training_started",
        "arm": arm,
        "subset_identity": subset_identity,
        "seed": REPAIRED_SEED,
        "checkpoint_role": REPAIRED_CHECKPOINT_ROLE,
        "operation_magnitude_parameterization": (
            POSITIVE_OPERATION_MAGNITUDE_PARAMETERIZATION
        ),
        "fresh_model": True,
        "resume_checkpoint": False,
        "training_configuration": _repaired_training_configuration(
            training_config
        ),
        "expected_arithmetic": repaired_training_arithmetic(len(examples)),
        "access": tracker.declarations(),
    })
    training = run_ge1_training(
        model,
        examples,
        training_config=training_config,
        checkpoint_directory=checkpoint_dir,
        repository_root=repository_root,
        expected_commit=expected_commit,
        final_epoch=REPAIRED_CHECKPOINT_EPOCH,
        engineering_smoke=False,
        checkpoint_epochs=(REPAIRED_CHECKPOINT_EPOCH,),
        fixed_protocol_final_epoch=REPAIRED_CHECKPOINT_EPOCH,
        checkpoint_coordinate="epoch",
        selected_checkpoint_epoch=REPAIRED_CHECKPOINT_EPOCH,
    )
    arithmetic = repaired_training_arithmetic(len(examples))
    if (
        training.completed_epoch != REPAIRED_CHECKPOINT_EPOCH
        or training.optimizer_steps != arithmetic["optimizer_steps"]
        or training.training_example_presentations
        != arithmetic["training_example_presentations"]
        or len(training.epoch_records) != REPAIRED_CHECKPOINT_EPOCH
        or len(training.checkpoint_paths) != 1
        or training.selected_experimental_checkpoint
        != training.checkpoint_paths[0]
    ):
        raise GraphEncoderError(
            "repaired_training_arithmetic_failure",
            "training differs from the fixed epoch-200 contract",
        )
    recovery_path = Path(training.checkpoint_paths[0])
    recovery_relative = recovery_path.relative_to(staging).as_posix()
    recovery_sha256 = _file_sha256(recovery_path)

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
        recovery_path,
        model=reloaded,
        optimizer=optimizer,
        training_config=training_config,
        partition_identity=partition_identity,
        expected_code_revision=expected_commit,
        restore_rng=True,
        expected_selected_checkpoint_epoch=REPAIRED_CHECKPOINT_EPOCH,
    )
    validate_repaired_reloaded_checkpoint(
        resume, expected_commit=expected_commit, job_id=job_id
    )
    _assert_model_states_equal(model, reloaded)

    inference_path = checkpoint_dir / "inference-epoch-0200.pt"
    save_ge1_checkpoint(inference_path, reloaded, code_revision=expected_commit)
    inference_reloaded, inference_payload = load_ge1_checkpoint(
        inference_path,
        expected_code_revision=expected_commit,
        expected_operation_magnitude_parameterization=(
            POSITIVE_OPERATION_MAGNITUDE_PARAMETERIZATION
        ),
    )
    _assert_model_states_equal(reloaded, inference_reloaded)
    if (
        inference_payload.get("operation_magnitude_parameterization")
        != POSITIVE_OPERATION_MAGNITUDE_PARAMETERIZATION
    ):
        raise GraphEncoderError(
            "invalid_repaired_inference_checkpoint",
            "inference checkpoint lacks the repaired identity",
        )
    inference_relative = inference_path.relative_to(staging).as_posix()
    inference_sha256 = _file_sha256(inference_path)
    checkpoint_identities = {
        "recovery": recovery_relative,
        "inference": inference_relative,
    }
    events.append({
        "event": "repaired_sufficiency_epoch_200_checkpoints_written",
        "arm": arm,
        "subset_identity": subset_identity,
        "epoch": REPAIRED_CHECKPOINT_EPOCH,
        "checkpoint_role": REPAIRED_CHECKPOINT_ROLE,
        "operation_magnitude_parameterization": (
            POSITIVE_OPERATION_MAGNITUDE_PARAMETERIZATION
        ),
        "checkpoint_identities": checkpoint_identities,
        "checkpoint_sha256": {
            "recovery": recovery_sha256,
            "inference": inference_sha256,
        },
        "training": _training_record_for_artifact(training, staging=staging),
        "access": tracker.declarations(),
    })
    events.append({
        "event": "repaired_sufficiency_checkpoints_reloaded",
        "arm": arm,
        "subset_identity": subset_identity,
        "epoch": REPAIRED_CHECKPOINT_EPOCH,
        "checkpoint_identities": checkpoint_identities,
        "checkpoint_sha256": {
            "recovery": recovery_sha256,
            "inference": inference_sha256,
        },
        "checkpoint_role": REPAIRED_CHECKPOINT_ROLE,
        "operation_magnitude_parameterization": (
            POSITIVE_OPERATION_MAGNITUDE_PARAMETERIZATION
        ),
        "strict_recovery_reload": True,
        "strict_inference_reload": True,
        "fresh_models_constructed": True,
        "source_commit": expected_commit,
        "slurm_job_id": job_id,
        "access": tracker.declarations(),
    })

    ordered = tuple(sorted(examples, key=lambda item: item.physical_family_id))
    input_batches = tuple(
        autonomous_input_from_paired(
            build_paired_batch(ordered[start:start + REPAIRED_BATCH_SIZE]), arm
        )
        for start in range(0, len(ordered), REPAIRED_BATCH_SIZE)
    )
    autonomous = run_autonomous_evaluation(
        inference_reloaded, input_batches, seed=REPAIRED_SEED
    )

    # Target construction is intentionally after autonomous generation. No
    # target is accepted by the model or autonomous interfaces above.
    targets = {item.physical_family_id: item.target for item in ordered}
    examples_by_family = {
        item.physical_family_id: item for item in ordered
    }
    metrics = tuple(
        score_condition(condition, targets)
        for condition in autonomous.conditions
    )
    templates = {
        item.physical_family_id: item.metadata.operation_template
        for item in ordered
    }
    condition_by_name = {item["condition"]: item for item in metrics}
    autonomous_by_name = {
        item.condition: item for item in autonomous.conditions
    }
    for condition, metric in zip(autonomous.conditions, metrics):
        events.append({
            "event": "repaired_sufficiency_autonomous_condition_metrics",
            "arm": arm,
            "subset_identity": subset_identity,
            "condition": condition.condition,
            "operation_magnitude_parameterization": (
                POSITIVE_OPERATION_MAGNITUDE_PARAMETERIZATION
            ),
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
            "targets_entered_after_autonomous_generation": True,
            "access": tracker.declarations(),
        })
    exact = repaired_exact_sufficiency_gate(
        arm=arm,
        subset_identity=subset_identity,
        condition_metrics=condition_by_name["P_true"],
        templates_by_family=templates,
        checkpoint_identity=inference_relative,
        epoch=REPAIRED_CHECKPOINT_EPOCH,
        access_flags=tracker.declarations(),
    )
    fidelity = operation_geometry_fidelity_gate(
        arm=arm,
        subset_identity=subset_identity,
        condition_result=autonomous_by_name["P_true"],
        examples_by_family=examples_by_family,
        checkpoint_identity=inference_relative,
        epoch=REPAIRED_CHECKPOINT_EPOCH,
        access_flags=tracker.declarations(),
        protocol_version=REPAIRED_PROTOCOL_VERSION,
        gate_version=REPAIRED_GATE_VERSION,
        checkpoint_role=REPAIRED_CHECKPOINT_ROLE,
        operation_magnitude_parameterization=(
            POSITIVE_OPERATION_MAGNITUDE_PARAMETERIZATION
        ),
    )
    for gate in (exact, fidelity):
        events.append({
            "event": "repaired_sufficiency_gate_result", **gate
        })
    memory = None
    if subset_identity == "scaled":
        memory = repaired_memory_use_gate(
            arm=arm,
            condition_metrics=metrics,
            autonomous_result=autonomous,
            templates_by_family=templates,
            checkpoint_identity=inference_relative,
            epoch=REPAIRED_CHECKPOINT_EPOCH,
            access_flags=tracker.declarations(),
        )
        events.append({
            "event": "repaired_sufficiency_gate_result", **memory
        })
        envelope = repaired_metrics_envelope(
            arm=arm,
            checkpoint_identities=checkpoint_identities,
            condition_metrics=metrics,
            autonomous_result=autonomous,
            templates_by_family=templates,
            training_result=training,
            parameter_counts=parameter_counts,
            exact_gate=exact,
            fidelity_gate=fidelity,
            memory_gate=memory,
            access_flags=tracker.declarations(),
            total_run_seconds=time.perf_counter() - run_start,
            staging=staging,
        )
        events.append({
            "event": "repaired_sufficiency_scaled_metrics_envelope",
            "record": envelope,
        })
    return {
        "exact_gate": exact,
        "fidelity_gate": fidelity,
        "memory_gate": memory,
        "checkpoint_identities": checkpoint_identities,
    }


def _assert_repaired_matched_pair(models):
    _assert_matched_disjoint(*models)
    for model in models:
        if (
            model.config.operation_magnitude_parameterization
            != POSITIVE_OPERATION_MAGNITUDE_PARAMETERIZATION
            or model.decoder.operation_magnitude_parameterization
            != POSITIVE_OPERATION_MAGNITUDE_PARAMETERIZATION
        ):
            raise GraphEncoderError(
                "legacy_checkpoint_ineligible",
                "both arms must use the repaired parameterization",
            )


def _load_selected_train(corpus_dir, family_ids, tracker, *, subset_identity):
    if subset_identity not in REPAIRED_SUBSETS:
        raise GraphEncoderError(
            "invalid_repaired_sufficiency_subset", "unknown subset"
        )
    if subset_identity == "scaled":
        if tracker.scaled_train_payload_accessed is False:
            tracker.scaled_train_payload_accessed = "not_confirmed_on_failure"
    else:
        if tracker.tiny_train_payload_accessed is False:
            tracker.tiny_train_payload_accessed = "not_confirmed_on_failure"
    if tracker.operation_template_train_payload_accessed is not True:
        tracker.operation_template_train_payload_accessed = (
            "not_confirmed_on_failure"
        )
    examples = tuple(load_train(corpus_dir, family_ids))
    observed = tuple(sorted(item.physical_family_id for item in examples))
    if observed != tuple(family_ids):
        raise GraphEncoderError(
            "repaired_payload_alignment_failure",
            "loaded train families differ from metadata selection",
        )
    tracker.operation_template_train_payload_accessed = True
    if subset_identity == "tiny":
        tracker.tiny_train_payload_accessed = True
    else:
        tracker.scaled_train_payload_accessed = True
    return examples


def _repaired_training_configuration(training_config):
    return {
        "protocol_version": REPAIRED_PROTOCOL_VERSION,
        "operation_magnitude_parameterization": (
            POSITIVE_OPERATION_MAGNITUDE_PARAMETERIZATION
        ),
        "inherited_training_configuration": training_config.to_dict(),
        "fixed_epoch": REPAIRED_CHECKPOINT_EPOCH,
        "fixed_checkpoint_only": True,
        "early_stopping": False,
        "best_loss_selection": False,
        "warm_start": False,
        "checkpoint_reuse": False,
    }


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
            None
            if selected is None
            else Path(selected).relative_to(root).as_posix()
        )
    record.update({
        "protocol_version": REPAIRED_PROTOCOL_VERSION,
        "checkpoint_role": REPAIRED_CHECKPOINT_ROLE,
        "operation_magnitude_parameterization": (
            POSITIVE_OPERATION_MAGNITUDE_PARAMETERIZATION
        ),
        "full_comparison_checkpoint": False,
        "fixed_checkpoint_epoch": REPAIRED_CHECKPOINT_EPOCH,
        "early_stopping": False,
        "best_checkpoint_selection": False,
        "outcome_dependent_extension": False,
        "warm_start": False,
    })
    return record


def _resolved_configuration(
    *, source, selection_record, protocol_config, training_config,
    model_configurations, parameter_inventories, tracker, job_id,
    both_tiny_pass, scaled_family_ids
):
    batches = tuple(
        scaled_family_ids[start:start + REPAIRED_BATCH_SIZE]
        for start in range(0, len(scaled_family_ids), REPAIRED_BATCH_SIZE)
    )
    return {
        "protocol_version": REPAIRED_PROTOCOL_VERSION,
        "metrics_version": REPAIRED_METRICS_VERSION,
        "gate_version": REPAIRED_GATE_VERSION,
        "artifact_version": REPAIRED_ARTIFACT_VERSION,
        "operation_fidelity_contract": OPERATION_GEOMETRY_FIDELITY_VERSION,
        "operation_magnitude_parameterization": (
            POSITIVE_OPERATION_MAGNITUDE_PARAMETERIZATION
        ),
        "source": source,
        "governed_source_digest": source["source_tree_sha256"],
        "historical_records": {
            "immutable_c7_v2": {
                "commit": IMMUTABLE_C7_V2_COMMIT,
                "job_id": IMMUTABLE_C7_V2_JOB,
                "result_changed": False,
                "code_or_schema_modified": False,
            },
            "operation_magnitude_engineering_validation": {
                "commit": REPAIR_VALIDATION_COMMIT,
                "job_id": REPAIR_VALIDATION_JOB,
                "scientific_result": False,
                "result_changed": False,
            },
        },
        "authoritative_manifest": {
            "name": "operation_template",
            "sha256": AUTHORITATIVE_FILE_SHA256,
            "authorized_partition": "train",
            "authorized_templates": list(C7_ACCESSIBLE_TEMPLATES),
        },
        "subset_selection": selection_record,
        "seed": REPAIRED_SEED,
        "model_configurations": model_configurations,
        "repaired_sufficiency_configuration": protocol_config.to_dict(),
        "training_configuration": _repaired_training_configuration(
            training_config
        ),
        "training_arithmetic": {
            "tiny": repaired_training_arithmetic(4),
            "scaled": repaired_training_arithmetic(32),
            "maximum_training_runs": 4,
        },
        "model_lifecycle": {
            "fresh_matched_pair_per_subset": True,
            "shared_initialization_between_arms": True,
            "mutable_objects_disjoint": True,
            "tiny_to_scaled_warm_start": False,
            "legacy_checkpoint_eligible": False,
            "prior_checkpoint_reuse": False,
            "checkpoint_role": REPAIRED_CHECKPOINT_ROLE,
        },
        "checkpoint_selection": {
            "selected_epoch": REPAIRED_CHECKPOINT_EPOCH,
            "recovery_and_inference_strict_reload_required": True,
            "earlier_epoch_gate_eligible": False,
            "training_loss_selection": False,
            "best_checkpoint_selection": False,
            "early_stopping": False,
            "outcome_dependent_extension": False,
        },
        "gate_dependency": {
            "both_tiny_arms_exact_and_fidelity_required_before_scaled": True,
            "both_tiny_arms_passed": both_tiny_pass,
        },
        "gate_thresholds": {
            "per_family_exact_required": 1.0,
            "depends_on_exactness_required_for": ["EE", "RE"],
            "operation_geometry_fidelity": operation_fidelity_contract(),
            "P_true_comparison": "strictly_greater_than_zero",
            "R_shuffle_max_inclusive": REPAIRED_MEMORY_RATIO_MAX,
            "R_mean_max_inclusive": REPAIRED_MEMORY_RATIO_MAX,
            "rounded_comparison": False,
        },
        "evaluation": {
            "order": list(scaled_family_ids),
            "batch_size": REPAIRED_BATCH_SIZE,
            "batch_membership": [list(item) for item in batches],
            "memory_conditions": ["P_true", "P_shuffle", "P_mean"],
            "fidelity_condition": "P_true",
            "targets_enter_after_autonomous_generation": True,
            "teacher_forced_gate_eligible": False,
        },
        "parameter_inventory": parameter_inventories,
        "runtime": {
            "python": platform.python_version(),
            "pytorch": str(torch.__version__),
            "device": "cpu",
            "cuda_available": bool(torch.cuda.is_available()),
            "cpu_threads": int(torch.get_num_threads()),
            "host": socket.gethostname(),
            "slurm_job_id": job_id,
        },
        "access": tracker.declarations(completed=True),
        "formal_repaired_sufficiency_execution": True,
        "cad_kernel_available": False,
        "stage6_performed": False,
        "c8_or_later_performed": False,
    }


def prepare_repaired_output(
    output_dir, *, repository_root, corpus_dir, job_id=None
):
    """Create a new external incomplete artifact directory."""

    identity = validate_slurm_job_id(job_id)
    raw = Path(output_dir)
    if raw.is_symlink() or raw.exists():
        raise GraphEncoderError(
            "repaired_output_exists", "output must be a new non-symlink path"
        )
    final = raw.resolve()
    repository = Path(repository_root).resolve()
    corpus = Path(corpus_dir).resolve()
    if _is_within(final, repository) or _is_within(final, corpus):
        raise GraphEncoderError(
            "unsafe_repaired_output_location",
            "output must be outside repository and corpus",
        )
    final.parent.mkdir(parents=True, exist_ok=True)
    staging = final.with_name(final.name + ".incomplete-" + identity)
    if staging.exists() or staging.is_symlink():
        raise GraphEncoderError(
            "repaired_staging_exists", "incomplete staging path exists"
        )
    staging.mkdir()
    (staging / "checkpoints").mkdir()
    return final, staging


def finalize_repaired_artifact(staging_dir, final_dir):
    """Write both integrity layers, verify, and atomically publish."""

    staging = Path(staging_dir)
    final = Path(final_dir)
    if staging.parent.resolve() != final.parent.resolve():
        raise GraphEncoderError(
            "unsafe_repaired_output_location",
            "staging and final must share a parent",
        )
    if final.exists() or final.is_symlink():
        raise GraphEncoderError("repaired_output_exists", "final output exists")
    required = (
        staging / "resolved_config.json",
        staging / "metrics.jsonl",
        staging / "checkpoints",
    )
    if any(not item.exists() for item in required):
        raise GraphEncoderError(
            "incomplete_repaired_artifact", "required artifact is absent"
        )
    ordinary = _regular_artifact_files(
        staging, excluded=("artifact_manifest.json", "SHA256SUMS")
    )
    checkpoints = tuple(
        path for path in ordinary if path.startswith("checkpoints/")
    )
    if len(checkpoints) not in (4, 8):
        raise GraphEncoderError(
            "incomplete_repaired_artifact",
            "artifact requires recovery and inference checkpoints per arm",
        )
    manifest = {
        "schema_version": REPAIRED_ARTIFACT_VERSION,
        "protocol_version": REPAIRED_PROTOCOL_VERSION,
        "operation_magnitude_parameterization": (
            POSITIVE_OPERATION_MAGNITUDE_PARAMETERIZATION
        ),
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
    verify_repaired_artifact(staging, allow_incomplete_name=True)
    os.replace(str(staging), str(final))
    return final


def verify_repaired_artifact(
    path, *, allow_incomplete_name=False, expected_commit=None,
    expected_slurm_job_id=None
):
    """Verify integrity, versioning, gates, provenance, and access semantics."""

    root = Path(path)
    if not root.is_dir() or root.is_symlink():
        raise GraphEncoderError(
            "invalid_repaired_artifact", "artifact root must be a real directory"
        )
    if ".incomplete-" in root.name and not allow_incomplete_name:
        raise GraphEncoderError(
            "invalid_repaired_artifact", "incomplete artifact cannot be final"
        )
    manifest_path = root / "artifact_manifest.json"
    sums_path = root / "SHA256SUMS"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        checksum_raw = sums_path.read_bytes()
    except (OSError, ValueError) as exc:
        raise GraphEncoderError(
            "invalid_repaired_artifact", "integrity metadata is unreadable"
        ) from exc
    if (
        manifest.get("schema_version") != REPAIRED_ARTIFACT_VERSION
        or manifest.get("protocol_version") != REPAIRED_PROTOCOL_VERSION
        or manifest.get("operation_magnitude_parameterization")
        != POSITIVE_OPERATION_MAGNITUDE_PARAMETERIZATION
    ):
        raise GraphEncoderError(
            "invalid_repaired_artifact", "artifact identities differ"
        )
    rows = manifest.get("artifacts")
    if not isinstance(rows, list):
        raise GraphEncoderError(
            "invalid_repaired_artifact", "artifact rows are absent"
        )
    listed = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != {
            "path", "byte_size", "sha256"
        }:
            raise GraphEncoderError(
                "invalid_repaired_artifact", "artifact row is malformed"
            )
        relative = _safe_relative_path(row["path"])
        target = root / relative
        if (
            not target.is_file()
            or target.is_symlink()
            or target.stat().st_size != row["byte_size"]
            or _file_sha256(target) != row["sha256"]
        ):
            raise GraphEncoderError(
                "repaired_artifact_integrity_failure",
                "artifact size or SHA-256 differs",
            )
        listed.append(relative)
    ordinary = _regular_artifact_files(
        root, excluded=("artifact_manifest.json", "SHA256SUMS")
    )
    if tuple(listed) != ordinary or len(listed) != len(set(listed)):
        raise GraphEncoderError(
            "invalid_repaired_artifact", "manifest coverage differs"
        )
    if not checksum_raw.endswith(b"\n"):
        raise GraphEncoderError(
            "invalid_repaired_artifact", "SHA256SUMS requires final LF"
        )
    checksum_paths = []
    try:
        checksum_lines = checksum_raw.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise GraphEncoderError(
            "invalid_repaired_artifact", "SHA256SUMS is not UTF-8"
        ) from exc
    for line in checksum_lines:
        parts = line.split("  ", 1)
        if len(parts) != 2 or len(parts[0]) != 64:
            raise GraphEncoderError(
                "invalid_repaired_artifact", "SHA256SUMS row is malformed"
            )
        relative = _safe_relative_path(parts[1])
        if _file_sha256(root / relative) != parts[0]:
            raise GraphEncoderError(
                "repaired_artifact_integrity_failure",
                "SHA256SUMS hash differs",
            )
        checksum_paths.append(relative)
    expected_sums = _regular_artifact_files(root, excluded=("SHA256SUMS",))
    if tuple(checksum_paths) != expected_sums:
        raise GraphEncoderError(
            "invalid_repaired_artifact", "SHA256SUMS coverage differs"
        )

    try:
        resolved = json.loads(
            (root / "resolved_config.json").read_text(encoding="utf-8")
        )
        events = _read_canonical_jsonl(root / "metrics.jsonl")
    except (OSError, ValueError) as exc:
        raise GraphEncoderError(
            "invalid_repaired_artifact", "artifact JSON is unreadable"
        ) from exc
    if not events or events[-1].get("event") != (
        "repaired_sufficiency_execution_completed"
    ):
        raise GraphEncoderError(
            "invalid_repaired_artifact", "terminal completion event is absent"
        )
    required_events = {
        "repaired_sufficiency_run_metadata",
        "repaired_sufficiency_subset_selection",
        "repaired_sufficiency_training_started",
        "repaired_sufficiency_epoch_200_checkpoints_written",
        "repaired_sufficiency_checkpoints_reloaded",
        "repaired_sufficiency_autonomous_condition_metrics",
        "repaired_sufficiency_gate_result",
        "repaired_sufficiency_overall_gate_decision",
        "repaired_sufficiency_execution_completed",
    }
    if not required_events.issubset(
        {item.get("event") for item in events}
    ):
        raise GraphEncoderError(
            "invalid_repaired_artifact", "required events are absent"
        )
    if (
        resolved.get("protocol_version") != REPAIRED_PROTOCOL_VERSION
        or resolved.get("metrics_version") != REPAIRED_METRICS_VERSION
        or resolved.get("gate_version") != REPAIRED_GATE_VERSION
        or resolved.get("artifact_version") != REPAIRED_ARTIFACT_VERSION
        or resolved.get("operation_magnitude_parameterization")
        != POSITIVE_OPERATION_MAGNITUDE_PARAMETERIZATION
        or resolved.get("source", {}).get("git_dirty") is not False
        or resolved.get("repaired_sufficiency_configuration")
        != RepairedSufficiencyConfiguration().to_dict()
        or resolved.get("training_arithmetic", {}).get("tiny")
        != repaired_training_arithmetic(4)
        or resolved.get("training_arithmetic", {}).get("scaled")
        != repaired_training_arithmetic(32)
    ):
        raise GraphEncoderError(
            "invalid_repaired_artifact",
            "resolved identity, configuration, or arithmetic differs",
        )
    history = resolved.get("historical_records", {})
    if (
        history.get("immutable_c7_v2", {}).get("commit")
        != IMMUTABLE_C7_V2_COMMIT
        or history.get("immutable_c7_v2", {}).get("job_id")
        != IMMUTABLE_C7_V2_JOB
        or history.get("immutable_c7_v2", {}).get("result_changed") is not False
        or history.get("operation_magnitude_engineering_validation", {}).get(
            "result_changed"
        ) is not False
    ):
        raise GraphEncoderError(
            "invalid_repaired_artifact", "historical boundary differs"
        )
    runtime = resolved.get("runtime", {})
    if (
        runtime.get("python") != "3.8.13"
        or str(runtime.get("pytorch", "")).split("+")[0] != "1.11.0"
        or runtime.get("device") != "cpu"
        or runtime.get("cuda_available") is not False
        or runtime.get("cpu_threads") != 1
    ):
        raise GraphEncoderError(
            "invalid_repaired_artifact", "authoritative environment differs"
        )
    source_commit = resolved["source"].get("git_commit")
    slurm_job_id = runtime.get("slurm_job_id")
    if expected_commit is not None and source_commit != expected_commit:
        raise GraphEncoderError(
            "invalid_repaired_artifact", "source commit differs"
        )
    if (
        expected_slurm_job_id is not None
        and slurm_job_id != str(expected_slurm_job_id)
    ):
        raise GraphEncoderError(
            "invalid_repaired_artifact", "Slurm identity differs"
        )
    terminal = events[-1]
    if (
        terminal.get("slurm_job_id") != slurm_job_id
        or terminal.get("operation_magnitude_parameterization")
        != POSITIVE_OPERATION_MAGNITUDE_PARAMETERIZATION
        or terminal.get("cad_kernel_available") is not False
        or terminal.get("stage6_performed") is not False
        or terminal.get("c8_or_later_performed") is not False
        or any(terminal.get(name) is not False for name in REPAIRED_PROTECTED_ACCESS_FIELDS)
    ):
        raise GraphEncoderError(
            "invalid_repaired_artifact", "terminal provenance or access differs"
        )
    gate_rows = [
        item
        for item in events
        if item.get("event") == "repaired_sufficiency_gate_result"
    ]
    gates = {
        "{}.{}".format(item.get("arm"), item.get("gate_name")): item
        for item in gate_rows
    }
    if (
        len(gate_rows) != len(REPAIRED_REQUIRED_GATE_NAMES)
        or set(gates) != set(REPAIRED_REQUIRED_GATE_NAMES)
        or any(
            item.get("version") != REPAIRED_PROTOCOL_VERSION
            or item.get("gate_version") != REPAIRED_GATE_VERSION
            or item.get("operation_magnitude_parameterization")
            != POSITIVE_OPERATION_MAGNITUDE_PARAMETERIZATION
            for item in gate_rows
        )
    ):
        raise GraphEncoderError(
            "invalid_repaired_artifact", "complete versioned gate set is absent"
        )
    expected_decision = overall_repaired_sufficiency_decision(gates)
    for name, expected in expected_decision.items():
        if terminal.get(name) != expected:
            raise GraphEncoderError(
                "invalid_repaired_artifact",
                "terminal decision differs from gate records",
            )
    scaled_statuses = tuple(
        gates["{}.{}".format(arm, gate)]["status"]
        for arm in REPAIRED_ARMS
        for gate in (
            "scaled_exact_sufficiency",
            "scaled_operation_geometry_fidelity",
            "scaled_memory_use",
        )
    )
    scaled_access = terminal.get("scaled_train_payload_accessed")
    if (
        scaled_access is False and scaled_statuses != ("not_run",) * 6
    ) or (scaled_access is True and "not_run" in scaled_statuses):
        raise GraphEncoderError(
            "invalid_repaired_artifact", "scaled access dependency differs"
        )
    checkpoint_paths = tuple(
        path for path in ordinary if path.startswith("checkpoints/")
    )
    expected_count = 8 if scaled_access is True else 4
    if len(checkpoint_paths) != expected_count:
        raise GraphEncoderError(
            "invalid_repaired_artifact", "checkpoint count differs"
        )
    written = [
        item
        for item in events
        if item.get("event")
        == "repaired_sufficiency_epoch_200_checkpoints_written"
    ]
    reloaded = [
        item
        for item in events
        if item.get("event") == "repaired_sufficiency_checkpoints_reloaded"
    ]
    if len(written) * 2 != expected_count or len(reloaded) != len(written):
        raise GraphEncoderError(
            "invalid_repaired_artifact", "checkpoint event count differs"
        )
    event_positions = {id(item): index for index, item in enumerate(events)}
    written_by_pair = {
        (item.get("subset_identity"), item.get("arm")): item
        for item in written
    }
    for gate in gates.values():
        if gate.get("status") == "not_run":
            if gate.get("checkpoint_identity") is not None:
                raise GraphEncoderError(
                    "invalid_repaired_artifact",
                    "not-run gate unexpectedly names a checkpoint",
                )
            continue
        pair = (gate.get("subset_identity"), gate.get("arm"))
        expected_inference = written_by_pair.get(pair, {}).get(
            "checkpoint_identities", {}
        ).get("inference")
        if (
            expected_inference is None
            or gate.get("checkpoint_identity") != expected_inference
        ):
            raise GraphEncoderError(
                "invalid_repaired_artifact",
                "gate does not name the strict inference reload",
            )
    for reload_event in reloaded:
        key = (reload_event.get("subset_identity"), reload_event.get("arm"))
        write_event = written_by_pair.get(key)
        if (
            write_event is None
            or event_positions[id(write_event)] >= event_positions[id(reload_event)]
            or reload_event.get("epoch") != REPAIRED_CHECKPOINT_EPOCH
            or reload_event.get("strict_recovery_reload") is not True
            or reload_event.get("strict_inference_reload") is not True
            or reload_event.get("source_commit") != source_commit
            or reload_event.get("slurm_job_id") != slurm_job_id
            or reload_event.get("checkpoint_identities")
            != write_event.get("checkpoint_identities")
            or reload_event.get("checkpoint_sha256")
            != write_event.get("checkpoint_sha256")
        ):
            raise GraphEncoderError(
                "invalid_repaired_artifact",
                "checkpoint write/reload provenance differs",
            )
        for kind, identity in reload_event["checkpoint_identities"].items():
            if (
                kind not in ("recovery", "inference")
                or identity not in checkpoint_paths
                or reload_event["checkpoint_sha256"].get(kind)
                != _file_sha256(root / identity)
            ):
                raise GraphEncoderError(
                    "invalid_repaired_artifact", "checkpoint hash differs"
                )
    return {
        "protocol_version": REPAIRED_PROTOCOL_VERSION,
        "artifact_manifest_sha256": _file_sha256(manifest_path),
        "sha256sums_sha256": _file_sha256(sums_path),
        "regular_file_count": len(expected_sums) + 1,
        "checkpoint_count": len(checkpoint_paths),
        "source_commit": source_commit,
        "slurm_job_id": slurm_job_id,
        "repaired_scientific_gate_pass": terminal[
            "repaired_scientific_gate_pass"
        ],
        "stage6_authorized_by_repaired_sufficiency": terminal[
            "stage6_authorized_by_repaired_sufficiency"
        ],
    }


def _require_authoritative_runtime():
    if torch is None:
        raise RuntimeError("repaired scientific execution requires PyTorch")
    if sys.version_info[:3] != (3, 8, 13):
        raise GraphEncoderError(
            "environment_mismatch", "protocol requires Python 3.8.13"
        )
    if str(torch.__version__).split("+")[0] != "1.11.0":
        raise GraphEncoderError(
            "environment_mismatch", "protocol requires PyTorch 1.11.0"
        )
    if torch.cuda.is_available() or torch.get_num_threads() != 1:
        raise GraphEncoderError(
            "environment_mismatch", "protocol requires one-thread CPU"
        )
    return validate_slurm_job_id()


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run prospective repaired GE1 train sufficiency"
    )
    parser.add_argument("--corpus-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--expected-commit", required=True)
    arguments = parser.parse_args(argv)
    try:
        result = run_repaired_sufficiency(
            corpus_dir=arguments.corpus_dir,
            output_dir=arguments.output_dir,
            repository_root=arguments.repository_root,
            expected_commit=arguments.expected_commit,
        )
    except Exception as exc:
        failure = {
            "event": "repaired_sufficiency_infrastructure_failure",
            "failure_type": type(exc).__name__,
            "failure_code": getattr(exc, "code", None),
            **overall_repaired_sufficiency_decision(
                {}, infrastructure_failure=True
            ),
        }
        print(json.dumps(failure, sort_keys=True, separators=(",", ":")))
        return 1
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
