"""Post-C7 fixed-update optimization-sufficiency diagnostic."""

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
import sys
import time

try:
    import torch
except ImportError:  # Pure contract and artifact tests remain torch-free.
    torch = None

from .errors import GraphEncoderError
from .metrics import (
    donor_template_agreement,
    intervention_ratios,
    parameter_count_record,
    score_condition,
    validate_primary_report,
)
from .partitions import (
    AUTHORITATIVE_FILE_SHA256,
    load_train,
    select_c7_sufficiency_subsets,
    selected_family_ids_sha256,
)
from .pilot import (
    _assert_matched_disjoint,
    _atomic_write_bytes,
    _atomic_write_json,
    _atomic_write_jsonl,
    _canonical_json_text,
    _file_sha256,
    _is_within,
    _read_canonical_jsonl,
    _regular_artifact_files,
    _safe_relative_path,
    _source_identity,
    _verify_source_unchanged,
)


DIAGNOSTIC_VERSION = "GE1-C7-OPTIMIZATION-SUFFICIENCY-DIAGNOSTIC-v1"
TRAJECTORY_VERSION = "GE1-C7-OPTIMIZATION-TRAJECTORY-v1"
ARTIFACT_VERSION = "GE1-C7-OPTIMIZATION-ARTIFACT-v1"
SEED = 2026
MAXIMUM_OPTIMIZER_UPDATES = 500
TRAINING_EXAMPLE_PRESENTATIONS = 2000
EFFECTIVE_EXAMPLES_PER_UPDATE = 4
AUTONOMOUS_MILESTONES = (50, 100, 200, 500)
RECOVERY_CHECKPOINT_INTERVAL = 25
CHECKPOINT_UPDATES = tuple(range(25, MAXIMUM_OPTIMIZER_UPDATES + 1, 25))
ARMS = ("flat", "typed_graph")
CHECKPOINT_ROLE = "post_c7_optimization_diagnostic_only"
INTERPRETATION_CATEGORIES = (
    "undertraining_supported_both_arms",
    "undertraining_or_optimization_difference_supported_one_arm",
    "still_improving_at_update_500",
    "plateaued_without_exact_sufficiency",
    "infrastructure_failure",
)
FROZEN_FAMILIES_BY_TEMPLATE = (
    (
        "E",
        "sf_d11886750826237040bec2a8580aa378e367b3e7ba51dd972c57b476408c470c",
    ),
    (
        "R",
        "sf_5bbeb6143e53c2d259f24a60b576708dd8267b0ad49ed62c31c966e36e503ac4",
    ),
    (
        "EE",
        "sf_0626a6cb7e6b08cd7cae372a0d61d028cdc57c0d5bfd708a46397024e37d72f1",
    ),
    (
        "RE",
        "sf_fe8e4fd3246a0c94e25c661b0e879fa9099d098c8f558424898e72f3822746ff",
    ),
)
FROZEN_FAMILY_IDS = tuple(sorted(item[1] for item in FROZEN_FAMILIES_BY_TEMPLATE))
FROZEN_FAMILY_IDS_SHA256 = (
    "c07e90675099cb2ea58575fe91683dc80b70cf3fa85fa6fa8de0c44dbe9783d6"
)
PROTECTED_ACCESS_FIELDS = (
    "development_accessed",
    "systematic_rr_accessed",
    "test_er_accessed",
    "iid_accessed",
    "history_depth_accessed",
    "geometry_extrapolation_accessed",
)
FORBIDDEN_DECISION_FIELDS = (
    "overall_gate_pass",
    "stage6_authorized_by_c7",
    "preauthorized_decoder_repair_triggered",
)
_MISSING_ARGUMENT = object()


@dataclass
class DiagnosticAccessTracker:
    operation_template_manifest_accessed: object = False
    operation_template_train_payload_accessed: object = False

    def declarations(self, *, completed=False):
        values = {
            "operation_template_manifest_accessed": (
                self.operation_template_manifest_accessed
            ),
            "operation_template_train_payload_accessed": (
                self.operation_template_train_payload_accessed
            ),
            "optimization_diagnostic_completed": bool(completed),
            "original_c7_result_changed": False,
            "stage6_authorized": False,
            "decoder_repair_invoked": False,
            "c8_or_later_performed": False,
        }
        values.update({name: False for name in PROTECTED_ACCESS_FIELDS})
        return values


def diagnostic_training_arithmetic():
    """Return the governed update-based budget and assert its consistency."""

    if (
        MAXIMUM_OPTIMIZER_UPDATES * EFFECTIVE_EXAMPLES_PER_UPDATE
        != TRAINING_EXAMPLE_PRESENTATIONS
    ):
        raise AssertionError("diagnostic presentation arithmetic differs")
    return {
        "governing_coordinate": "optimizer_updates",
        "maximum_optimizer_updates": MAXIMUM_OPTIMIZER_UPDATES,
        "effective_examples_per_update": EFFECTIVE_EXAMPLES_PER_UPDATE,
        "training_example_presentations": TRAINING_EXAMPLE_PRESENTATIONS,
        "batch_size": 8,
        "one_update_currently_equals_one_epoch": True,
        "scientific_measurement_milestones": list(AUTONOMOUS_MILESTONES),
        "recovery_checkpoint_interval_updates": RECOVERY_CHECKPOINT_INTERVAL,
        "retained_checkpoint_updates": list(CHECKPOINT_UPDATES),
        "retained_checkpoint_count_per_arm": len(CHECKPOINT_UPDATES),
        "early_stopping": False,
        "extension_after_observation": False,
    }


def validate_slurm_job_id(value=_MISSING_ARGUMENT):
    """Require the decimal Slurm identity that binds every artifact record."""

    observed = (
        os.environ.get("SLURM_JOB_ID")
        if value is _MISSING_ARGUMENT
        else value
    )
    if (
        not isinstance(observed, str)
        or not observed
        or not observed.isascii()
        or not observed.isdecimal()
    ):
        raise GraphEncoderError(
            "invalid_slurm_job_id",
            "SLURM_JOB_ID must be a nonempty decimal string",
        )
    return observed


def validate_frozen_selection(selection):
    """Require the existing metadata-only selector to reproduce formal C7."""

    observed = tuple(selection.tiny_family_ids)
    observed_hash = selected_family_ids_sha256(observed)
    if observed != FROZEN_FAMILY_IDS or observed_hash != FROZEN_FAMILY_IDS_SHA256:
        raise GraphEncoderError(
            "diagnostic_family_identity_mismatch",
            "metadata selector did not reproduce the formal C7 tiny cohort",
        )
    templates = dict(selection.selected_templates)
    expected_templates = {family: template for template, family in FROZEN_FAMILIES_BY_TEMPLATE}
    if {family: templates.get(family) for family in observed} != expected_templates:
        raise GraphEncoderError(
            "diagnostic_family_identity_mismatch",
            "selected family templates differ from formal C7",
        )
    return {
        "selector_version": selection.version,
        "metadata_only": True,
        "family_ids": list(observed),
        "family_ids_sha256": observed_hash,
        "templates_by_family": dict(sorted(expected_templates.items())),
        "payload_aware_selection": False,
    }


def exact_sufficiency_at_milestone(
    *, arm, optimizer_update, condition_metrics, templates_by_family,
    checkpoint_identity
):
    """Apply the autonomous, unrounded, physical-family exact criteria."""

    if arm not in ARMS or optimizer_update not in AUTONOMOUS_MILESTONES:
        raise GraphEncoderError("invalid_diagnostic_milestone", "unknown arm or update")
    if not isinstance(condition_metrics, dict):
        raise GraphEncoderError("malformed_diagnostic_metrics", "metrics must be object")
    if (
        condition_metrics.get("condition") != "P_true"
        or condition_metrics.get("available") is not True
    ):
        raise GraphEncoderError(
            "nonautonomous_diagnostic_input",
            "only available autonomous P_true metrics determine sufficiency",
        )
    values = condition_metrics.get("family_values")
    templates = dict(templates_by_family)
    if not isinstance(values, dict) or set(values) != set(FROZEN_FAMILY_IDS):
        raise GraphEncoderError(
            "metric_target_alignment_failure",
            "diagnostic metrics must contain exactly the frozen four families",
        )
    if set(templates) != set(values):
        raise GraphEncoderError(
            "metric_target_alignment_failure", "template and metric families differ"
        )
    required = (
        "exact_node_sequence",
        "exact_graph",
        "strict_conversion",
        "complete_executable_validity",
    )
    failures = {}
    criteria = {}
    for field in required:
        passing = 0
        for family_id in FROZEN_FAMILY_IDS:
            value = _exact_metric(values[family_id].get(field), field)
            if value == 1.0:
                passing += 1
            else:
                failures.setdefault(family_id, []).append({
                    "field": field, "required_value": 1.0, "observed_value": value,
                })
        criteria[field] = {
            "required_value": 1.0,
            "passing_family_count": passing,
            "physical_family_denominator": 4,
        }
    dependency_families = tuple(sorted(
        family for family, template in templates.items() if template in ("EE", "RE")
    ))
    dependency_passing = 0
    for family_id in dependency_families:
        value = _exact_metric(
            values[family_id].get("depends_on_exactness"), "depends_on_exactness"
        )
        if value == 1.0:
            dependency_passing += 1
        else:
            failures.setdefault(family_id, []).append({
                "field": "depends_on_exactness",
                "required_value": 1.0,
                "observed_value": value,
            })
    criteria["depends_on_exactness_for_EE_RE"] = {
        "required_value": 1.0,
        "passing_family_count": dependency_passing,
        "physical_family_denominator": len(dependency_families),
        "applicable_templates": ["EE", "RE"],
    }
    exact = not failures
    return {
        "trajectory_version": TRAJECTORY_VERSION,
        "arm": arm,
        "optimizer_update": optimizer_update,
        "checkpoint_identity": checkpoint_identity,
        "checkpoint_role": CHECKPOINT_ROLE,
        "generation_condition": "P_true",
        "teacher_forced_sufficiency_eligible": False,
        "exact_sufficient": exact,
        "criteria": criteria,
        "failing_family_ids": sorted(failures),
        "failed_fields_by_family": {
            family: failures[family] for family in sorted(failures)
        },
    }


def diagnostic_interpretation(arm_records, *, infrastructure_failure=False):
    """Apply exactly one preregistered post-C7 interpretation category."""

    if infrastructure_failure:
        return {
            "interpretation_category": "infrastructure_failure",
            **_fixed_interpretation_fields(),
        }
    if set(arm_records) != set(ARMS):
        raise GraphEncoderError(
            "incomplete_diagnostic_trajectory", "both arm trajectories are required"
        )
    summaries = {}
    for arm in ARMS:
        records = tuple(arm_records[arm])
        if tuple(item.get("optimizer_update") for item in records) != AUTONOMOUS_MILESTONES:
            raise GraphEncoderError(
                "incomplete_diagnostic_trajectory", "milestone sequence differs"
            )
        exact_updates = tuple(
            item["optimizer_update"] for item in records if item.get("exact_sufficient") is True
        )
        summaries[arm] = {
            "ever": bool(exact_updates),
            "first": exact_updates[0] if exact_updates else None,
            "at_500": records[-1].get("exact_sufficient") is True,
            "plateau": bool(records[-1].get("plateau_observed")),
        }
    exact_arm_count = sum(summaries[arm]["ever"] for arm in ARMS)
    if exact_arm_count == 2:
        category = "undertraining_supported_both_arms"
    elif exact_arm_count == 1:
        category = "undertraining_or_optimization_difference_supported_one_arm"
    elif not all(summaries[arm]["plateau"] for arm in ARMS):
        category = "still_improving_at_update_500"
    else:
        category = "plateaued_without_exact_sufficiency"
    return {
        "interpretation_category": category,
        "flat_ever_exact_at_predetermined_milestone": summaries["flat"]["ever"],
        "typed_graph_ever_exact_at_predetermined_milestone": summaries["typed_graph"]["ever"],
        "flat_first_exact_update": summaries["flat"]["first"],
        "typed_graph_first_exact_update": summaries["typed_graph"]["first"],
        "flat_exact_at_update_500": summaries["flat"]["at_500"],
        "typed_graph_exact_at_update_500": summaries["typed_graph"]["at_500"],
        "flat_plateau_observed": summaries["flat"]["plateau"],
        "typed_graph_plateau_observed": summaries["typed_graph"]["plateau"],
        "optimization_budget_explanation": (
            "Both arms trained continuously from fresh matched initialization "
            "through the predetermined 500-update maximum; milestones did not "
            "select, stop, or extend training."
        ),
        "original_c7_result_changed": False,
        "stage6_authorized": False,
        "decoder_repair_invoked": False,
        "c8_or_later_performed": False,
    }


def run_optimization_diagnostic(
    *, corpus_dir, output_dir, repository_root, expected_commit,
    access_tracker=None
):
    """Run the governed diagnostic and atomically publish a complete artifact."""

    _require_authoritative_runtime()
    job_id = validate_slurm_job_id()
    tracker = access_tracker or DiagnosticAccessTracker()
    source = _source_identity(repository_root, expected_commit)
    final, staging = prepare_diagnostic_output(
        output_dir,
        repository_root=repository_root,
        corpus_dir=corpus_dir,
        slurm_job_id=job_id,
    )
    started = time.perf_counter()
    events = []
    tracker.operation_template_manifest_accessed = "not_confirmed_on_failure"
    selection = select_c7_sufficiency_subsets(corpus_dir)
    tracker.operation_template_manifest_accessed = True
    selection_record = validate_frozen_selection(selection)
    events.append({"event": "diagnostic_subset_selection", **selection_record})
    examples = _load_frozen_train(corpus_dir, tracker)

    from .config import GE1TrainingConfig
    from .model import build_legacy_matched_ge1_models

    training_config = GE1TrainingConfig()
    training_config.validate()
    flat_model, graph_model = build_legacy_matched_ge1_models(seed=SEED)
    _assert_matched_disjoint(flat_model, graph_model)
    parameter_counts = parameter_count_record(flat_model, graph_model)
    events.append({
        "event": "matched_fresh_initialization",
        "seed": SEED,
        "fresh_initialization": True,
        "formal_c7_checkpoint_resume": False,
        "arm_warm_start": False,
        "shared_components_byte_identical": True,
        "mutable_parameters_and_buffers_disjoint": True,
        "checkpoint_role": CHECKPOINT_ROLE,
        "parameter_counts": parameter_counts,
        "slurm_job_id": job_id,
    })

    trajectories = {}
    training_records = {}
    for model in (flat_model, graph_model):
        arm = model.config.encoder
        arm_result = _train_and_measure_arm(
            model=model,
            examples=examples,
            staging=staging,
            repository_root=repository_root,
            expected_commit=expected_commit,
            training_config=training_config,
            job_id=job_id,
            tracker=tracker,
            events=events,
        )
        trajectories[arm] = arm_result["milestone_summaries"]
        training_records[arm] = arm_result["training_record"]

    interpretation = diagnostic_interpretation(trajectories)
    events.append({"event": "diagnostic_interpretation", **interpretation})
    access = tracker.declarations(completed=True)
    resolved = {
        "diagnostic_version": DIAGNOSTIC_VERSION,
        "trajectory_version": TRAJECTORY_VERSION,
        "artifact_version": ARTIFACT_VERSION,
        "source": source,
        "governed_source_digest": source["source_tree_sha256"],
        "runtime": {
            "python": platform.python_version(),
            "pytorch": str(torch.__version__),
            "device": "cpu",
            "cuda_available": bool(torch.cuda.is_available()),
            "host": socket.gethostname(),
            "slurm_job_id": job_id,
        },
        "authoritative_manifest": {
            "name": "operation_template",
            "sha256": AUTHORITATIVE_FILE_SHA256,
            "authorized_partition": "train",
        },
        "selection": selection_record,
        "model_lifecycle": {
            "seed": SEED,
            "fresh_matched_initialization": True,
            "shared_components_byte_identical": True,
            "mutable_parameters_and_buffers_disjoint": True,
            "formal_c7_checkpoint_resume": False,
            "arm_warm_start": False,
            "full_comparison_checkpoint": False,
            "checkpoint_role": CHECKPOINT_ROLE,
        },
        "training_configuration": training_config.to_dict(),
        "training_arithmetic": diagnostic_training_arithmetic(),
        "autonomous_measurement": {
            "conditions": ["P_true", "P_shuffle", "P_mean"],
            "sufficiency_condition": "P_true",
            "targets_scored_only_after_generation": True,
            "teacher_forced_sufficiency_eligible": False,
            "exact_unrounded_required": 1.0,
        },
        "training_records": training_records,
        "interpretation": interpretation,
        "access": access,
        "formal_c7_result": {
            "job_id": "3344505",
            "result_changed": False,
            "formal_c7_stage6_authorization_was_false": True,
        },
    }
    run_metadata = {
        "event": "optimization_diagnostic_run_metadata",
        "diagnostic_version": DIAGNOSTIC_VERSION,
        "trajectory_version": TRAJECTORY_VERSION,
        "artifact_version": ARTIFACT_VERSION,
        "source": source,
        "slurm_job_id": job_id,
        "access": access,
    }
    completion = {
        "event": "optimization_diagnostic_completed",
        "diagnostic_version": DIAGNOSTIC_VERSION,
        "interpretation_category": interpretation["interpretation_category"],
        "slurm_job_id": job_id,
        "elapsed_seconds": time.perf_counter() - started,
        **access,
    }
    events = [run_metadata] + events + [completion]
    _assert_no_formal_c7_decision_fields(resolved)
    _assert_no_formal_c7_decision_fields(events)
    _atomic_write_json(staging / "resolved_config.json", resolved)
    _atomic_write_jsonl(staging / "metrics.jsonl", events)
    _verify_source_unchanged(source, repository_root, expected_commit)
    finalize_diagnostic_artifact(staging, final)
    verified = verify_diagnostic_artifact(final, expected_slurm_job_id=job_id)
    return {
        "artifact_path": str(final),
        "artifact_verification": verified,
        "interpretation": interpretation,
        "completion_event": completion,
    }


def prepare_diagnostic_output(
    output_dir, *, repository_root, corpus_dir, slurm_job_id
):
    """Create a new external incomplete directory for atomic publication."""

    job_id = validate_slurm_job_id(slurm_job_id)
    raw = Path(output_dir)
    if raw.exists() or raw.is_symlink():
        raise GraphEncoderError(
            "diagnostic_output_exists", "diagnostic output must be a new path"
        )
    final = raw.resolve()
    repository = Path(repository_root).resolve()
    corpus = Path(corpus_dir).resolve()
    if _is_within(final, repository) or _is_within(final, corpus):
        raise GraphEncoderError(
            "unsafe_diagnostic_output_location",
            "diagnostic output must be outside repository and corpus",
        )
    final.parent.mkdir(parents=True, exist_ok=True)
    staging = final.with_name(final.name + ".incomplete-" + job_id)
    if staging.exists() or staging.is_symlink():
        raise GraphEncoderError(
            "diagnostic_staging_exists", "incomplete diagnostic path exists"
        )
    staging.mkdir()
    (staging / "checkpoints").mkdir()
    return final, staging


def finalize_diagnostic_artifact(staging_dir, final_dir):
    """Write both integrity layers, verify, and atomically publish."""

    staging = Path(staging_dir)
    final = Path(final_dir)
    if staging.parent.resolve() != final.parent.resolve():
        raise GraphEncoderError(
            "unsafe_diagnostic_output_location", "staging and final parents differ"
        )
    if final.exists() or final.is_symlink():
        raise GraphEncoderError("diagnostic_output_exists", "final output exists")
    required = (
        staging / "resolved_config.json",
        staging / "metrics.jsonl",
        staging / "checkpoints",
    )
    if any(not item.exists() for item in required):
        raise GraphEncoderError(
            "incomplete_diagnostic_artifact", "required artifact is absent"
        )
    ordinary = _regular_artifact_files(
        staging, excluded=("artifact_manifest.json", "SHA256SUMS")
    )
    if not any(item.startswith("checkpoints/") for item in ordinary):
        raise GraphEncoderError(
            "incomplete_diagnostic_artifact", "artifact has no checkpoints"
        )
    manifest = {
        "schema_version": ARTIFACT_VERSION,
        "artifacts": [
            {
                "path": relative,
                "byte_size": (staging / relative).stat().st_size,
                "sha256": _file_sha256(staging / relative),
            }
            for relative in ordinary
        ],
    }
    _atomic_write_json(staging / "artifact_manifest.json", manifest)
    checksum_paths = _regular_artifact_files(staging, excluded=("SHA256SUMS",))
    text = "".join(
        "{}  {}\n".format(_file_sha256(staging / relative), relative)
        for relative in checksum_paths
    )
    _atomic_write_bytes(staging / "SHA256SUMS", text.encode("utf-8"))
    job_id = _resolved_slurm_job_id(staging)
    verify_diagnostic_artifact(
        staging, expected_slurm_job_id=job_id, allow_incomplete_name=True
    )
    os.replace(str(staging), str(final))
    return final


def verify_diagnostic_artifact(
    path, *, expected_slurm_job_id, allow_incomplete_name=False
):
    """Verify integrity, completion, access, and structured Slurm identity."""

    job_id = validate_slurm_job_id(expected_slurm_job_id)
    root = Path(path)
    if not root.is_dir() or root.is_symlink():
        raise GraphEncoderError(
            "invalid_diagnostic_artifact", "artifact root must be a real directory"
        )
    if ".incomplete-" in root.name and not allow_incomplete_name:
        raise GraphEncoderError(
            "invalid_diagnostic_artifact", "incomplete artifact cannot be final"
        )
    manifest_path = root / "artifact_manifest.json"
    checksums_path = root / "SHA256SUMS"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        checksum_raw = checksums_path.read_bytes()
        resolved = json.loads((root / "resolved_config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise GraphEncoderError(
            "invalid_diagnostic_artifact", "artifact metadata is unreadable"
        ) from exc
    if manifest.get("schema_version") != ARTIFACT_VERSION:
        raise GraphEncoderError(
            "invalid_diagnostic_artifact", "artifact identity differs"
        )
    rows = manifest.get("artifacts")
    if not isinstance(rows, list):
        raise GraphEncoderError("invalid_diagnostic_artifact", "manifest rows missing")
    listed = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"path", "byte_size", "sha256"}:
            raise GraphEncoderError(
                "invalid_diagnostic_artifact", "manifest row is malformed"
            )
        relative = _safe_relative_path(row["path"])
        target = root / relative
        if not target.is_file() or target.is_symlink():
            raise GraphEncoderError(
                "invalid_diagnostic_artifact", "listed path is not a regular file"
            )
        if target.stat().st_size != row["byte_size"] or _file_sha256(target) != row["sha256"]:
            raise GraphEncoderError(
                "diagnostic_artifact_integrity_failure", "manifest value differs"
            )
        listed.append(relative)
    ordinary = _regular_artifact_files(
        root, excluded=("artifact_manifest.json", "SHA256SUMS")
    )
    if tuple(listed) != ordinary or listed != sorted(set(listed)):
        raise GraphEncoderError(
            "invalid_diagnostic_artifact", "manifest paths differ from files"
        )
    if not checksum_raw.endswith(b"\n"):
        raise GraphEncoderError(
            "invalid_diagnostic_artifact", "SHA256SUMS requires final LF"
        )
    checksum_rows = []
    for line in checksum_raw.decode("utf-8").splitlines():
        parts = line.split("  ", 1)
        if len(parts) != 2 or len(parts[0]) != 64:
            raise GraphEncoderError(
                "invalid_diagnostic_artifact", "checksum row is malformed"
            )
        relative = _safe_relative_path(parts[1])
        checksum_rows.append(relative)
        if _file_sha256(root / relative) != parts[0]:
            raise GraphEncoderError(
                "diagnostic_artifact_integrity_failure", "checksum hash differs"
            )
    expected_checksums = _regular_artifact_files(root, excluded=("SHA256SUMS",))
    if tuple(checksum_rows) != expected_checksums:
        raise GraphEncoderError(
            "invalid_diagnostic_artifact", "checksum paths differ from files"
        )
    events = _read_canonical_jsonl(root / "metrics.jsonl")
    required_events = {
        "optimization_diagnostic_run_metadata",
        "diagnostic_subset_selection",
        "matched_fresh_initialization",
        "diagnostic_training_completed",
        "checkpoint_provenance",
        "diagnostic_milestone_measurement",
        "diagnostic_interpretation",
        "optimization_diagnostic_completed",
    }
    if not events or events[-1].get("event") != "optimization_diagnostic_completed":
        raise GraphEncoderError(
            "invalid_diagnostic_artifact", "terminal completion event is absent"
        )
    if not required_events.issubset({item.get("event") for item in events}):
        raise GraphEncoderError(
            "invalid_diagnostic_artifact", "required diagnostic events are absent"
        )
    if (
        resolved.get("diagnostic_version") != DIAGNOSTIC_VERSION
        or resolved.get("trajectory_version") != TRAJECTORY_VERSION
        or resolved.get("artifact_version") != ARTIFACT_VERSION
        or resolved.get("runtime", {}).get("slurm_job_id") != job_id
    ):
        raise GraphEncoderError(
            "diagnostic_provenance_failure", "resolved identity or Slurm job differs"
        )
    _assert_no_formal_c7_decision_fields(resolved)
    _assert_no_formal_c7_decision_fields(events)
    run_metadata = events[0]
    terminal = events[-1]
    if run_metadata.get("slurm_job_id") != job_id or terminal.get("slurm_job_id") != job_id:
        raise GraphEncoderError(
            "diagnostic_provenance_failure", "run/terminal Slurm identity differs"
        )
    checkpoint_events = [
        item for item in events if item.get("event") == "checkpoint_provenance"
    ]
    expected_checkpoint_count = len(CHECKPOINT_UPDATES) * len(ARMS)
    if len(checkpoint_events) != expected_checkpoint_count:
        raise GraphEncoderError(
            "diagnostic_provenance_failure", "checkpoint provenance count differs"
        )
    if any(item.get("slurm_job_id") != job_id for item in checkpoint_events):
        raise GraphEncoderError(
            "diagnostic_provenance_failure", "checkpoint Slurm identity differs"
        )
    checkpoint_files = tuple(
        root / item for item in ordinary if item.startswith("checkpoints/")
    )
    if len(checkpoint_files) != expected_checkpoint_count:
        raise GraphEncoderError(
            "diagnostic_provenance_failure", "checkpoint file count differs"
        )
    if torch is not None:
        for checkpoint_path in checkpoint_files:
            if _checkpoint_provenance(checkpoint_path).get("slurm_job_id") != job_id:
                raise GraphEncoderError(
                    "diagnostic_provenance_failure",
                    "checkpoint payload Slurm identity differs",
                )
    milestone_events = [
        item for item in events if item.get("event") == "diagnostic_milestone_measurement"
    ]
    observed = tuple((item.get("arm"), item.get("optimizer_update")) for item in milestone_events)
    expected = tuple((arm, update) for arm in ARMS for update in AUTONOMOUS_MILESTONES)
    if observed != expected:
        raise GraphEncoderError(
            "incomplete_diagnostic_trajectory", "milestone events differ"
        )
    interpretation = next(
        item for item in events if item.get("event") == "diagnostic_interpretation"
    )
    if interpretation.get("interpretation_category") not in INTERPRETATION_CATEGORIES[:-1]:
        raise GraphEncoderError(
            "invalid_diagnostic_interpretation", "completed category is invalid"
        )
    for name in (
        "original_c7_result_changed", "stage6_authorized",
        "decoder_repair_invoked", "c8_or_later_performed",
    ):
        if terminal.get(name) is not False:
            raise GraphEncoderError(
                "invalid_diagnostic_artifact", "terminal boundary field differs"
            )
    for name in PROTECTED_ACCESS_FIELDS:
        if terminal.get(name) is not False:
            raise GraphEncoderError(
                "protected_partition_access", "protected access declaration differs"
            )
    return {
        "artifact_manifest_sha256": _file_sha256(manifest_path),
        "sha256sums_sha256": _file_sha256(checksums_path),
        "regular_file_count": len(expected_checksums) + 1,
        "slurm_job_id": job_id,
        "interpretation_category": interpretation["interpretation_category"],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--expected-commit", required=True)
    arguments = parser.parse_args(argv)
    tracker = DiagnosticAccessTracker()
    try:
        result = run_optimization_diagnostic(
            corpus_dir=arguments.corpus_dir,
            output_dir=arguments.output_dir,
            repository_root=arguments.repository_root,
            expected_commit=arguments.expected_commit,
            access_tracker=tracker,
        )
    except Exception as exc:
        failure = {
            "event": "optimization_diagnostic_infrastructure_failure",
            "interpretation_category": "infrastructure_failure",
            "code": getattr(exc, "code", type(exc).__name__),
            "detail": str(exc),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            **tracker.declarations(completed=False),
        }
        print(_canonical_json_text(failure), file=sys.stderr, flush=True)
        return 1
    terminal = {
        "event": "optimization_diagnostic_terminal_success",
        "artifact_path": result["artifact_path"],
        "slurm_job_id": validate_slurm_job_id(),
        "interpretation_category": result["interpretation"]["interpretation_category"],
        **tracker.declarations(completed=True),
    }
    print(_canonical_json_text(terminal), flush=True)
    return 0


def _train_and_measure_arm(
    *, model, examples, staging, repository_root, expected_commit,
    training_config, job_id, tracker, events
):
    from .autonomous import autonomous_input_from_paired, run_autonomous_evaluation
    from .batching import build_paired_batch
    from .model import build_ge1_model
    from .provenance import training_partition_identity
    from .training import load_training_checkpoint, run_ge1_training

    arm = model.config.encoder
    checkpoint_dir = staging / "checkpoints" / arm
    events.append({
        "event": "diagnostic_training_started",
        "arm": arm,
        "seed": SEED,
        "slurm_job_id": job_id,
        "arithmetic": diagnostic_training_arithmetic(),
        "resume_checkpoint": False,
        "checkpoint_role": CHECKPOINT_ROLE,
        "access": tracker.declarations(),
    })
    training = run_ge1_training(
        model,
        examples,
        training_config=training_config,
        checkpoint_directory=checkpoint_dir,
        repository_root=repository_root,
        expected_commit=expected_commit,
        final_epoch=MAXIMUM_OPTIMIZER_UPDATES,
        engineering_smoke=False,
        checkpoint_epochs=CHECKPOINT_UPDATES,
        extended_final_epoch=MAXIMUM_OPTIMIZER_UPDATES,
        checkpoint_coordinate="update",
        selected_checkpoint_epoch=None,
    )
    if (
        training.completed_epoch != MAXIMUM_OPTIMIZER_UPDATES
        or training.optimizer_steps != MAXIMUM_OPTIMIZER_UPDATES
        or training.training_example_presentations != TRAINING_EXAMPLE_PRESENTATIONS
        or len(training.epoch_records) != MAXIMUM_OPTIMIZER_UPDATES
        or len(training.checkpoint_paths) != len(CHECKPOINT_UPDATES)
        or training.selected_experimental_checkpoint is not None
    ):
        raise GraphEncoderError(
            "diagnostic_training_arithmetic_failure",
            "training result differs from the fixed update contract",
        )
    observed_updates = tuple(
        int(Path(item).stem.rsplit("update", 1)[1]) for item in training.checkpoint_paths
    )
    if observed_updates != CHECKPOINT_UPDATES:
        raise GraphEncoderError(
            "invalid_checkpoint_schedule", "retained checkpoint updates differ"
        )
    training_record = training.to_dict()
    training_record["checkpoint_paths"] = [
        Path(item).relative_to(staging).as_posix() for item in training.checkpoint_paths
    ]
    training_record.update({
        "governing_coordinate": "optimizer_updates",
        "maximum_optimizer_updates": MAXIMUM_OPTIMIZER_UPDATES,
        "diagnostic_checkpoint_selection": False,
        "formal_c7_checkpoint_resume": False,
        "checkpoint_role": CHECKPOINT_ROLE,
    })
    events.append({
        "event": "diagnostic_training_completed",
        "arm": arm,
        "slurm_job_id": job_id,
        "training": training_record,
        "access": tracker.declarations(),
    })
    by_update = dict(zip(CHECKPOINT_UPDATES, training.checkpoint_paths))
    templates = {
        item.physical_family_id: item.metadata.operation_template for item in examples
    }
    targets = {item.physical_family_id: item.target for item in examples}
    ordered = tuple(sorted(examples, key=lambda item: item.physical_family_id))
    partition_identity = training_partition_identity(FROZEN_FAMILY_IDS)
    summaries = []
    for update in AUTONOMOUS_MILESTONES:
        checkpoint_path = Path(by_update[update])
        reloaded = build_ge1_model(model.config)
        optimizer = torch.optim.AdamW(
            reloaded.parameters(),
            lr=training_config.learning_rate,
            weight_decay=training_config.weight_decay,
        )
        resume = load_training_checkpoint(
            checkpoint_path,
            model=reloaded,
            optimizer=optimizer,
            training_config=training_config,
            partition_identity=partition_identity,
            expected_code_revision=expected_commit,
            restore_rng=False,
            expected_selected_checkpoint_epoch=None,
        )
        relative = checkpoint_path.relative_to(staging).as_posix()
        checkpoint_job = resume.payload.get("provenance", {}).get("slurm_job_id")
        if (
            resume.completed_epoch != update
            or resume.optimizer_step_count != update
            or resume.training_example_presentations
            != update * EFFECTIVE_EXAMPLES_PER_UPDATE
            or checkpoint_job != job_id
        ):
            raise GraphEncoderError(
                "diagnostic_checkpoint_reload_failure",
                "milestone checkpoint counters or Slurm identity differ",
            )
        events.append({
            "event": "checkpoint_reloaded",
            "arm": arm,
            "optimizer_update": update,
            "checkpoint_identity": relative,
            "checkpoint_role": CHECKPOINT_ROLE,
            "strict_reload": True,
            "fresh_model_constructed": True,
            "slurm_job_id": job_id,
            "access": tracker.declarations(),
        })
        autonomous_input = autonomous_input_from_paired(
            build_paired_batch(ordered), arm
        )
        autonomous = run_autonomous_evaluation(
            reloaded, (autonomous_input,), seed=SEED
        )
        metrics = tuple(
            score_condition(condition, targets) for condition in autonomous.conditions
        )
        by_condition = {item["condition"]: item for item in metrics}
        exact = exact_sufficiency_at_milestone(
            arm=arm,
            optimizer_update=update,
            condition_metrics=by_condition["P_true"],
            templates_by_family=templates,
            checkpoint_identity=relative,
        )
        ratios = intervention_ratios(metrics)
        validate_primary_report({"intervention_ratios": ratios})
        donors = [
            donor_template_agreement(
                condition.memory_assignments,
                templates,
                condition=condition.condition,
            )
            for condition in autonomous.conditions
        ]
        memory_evidence = {
            condition.condition: {
                "available": condition.available,
                "unavailable_reason": condition.unavailable_reason,
                "memory_altered": condition.memory_altered,
                "alteration_possible": condition.alteration_possible,
                "memory_assignments": [list(item) for item in condition.memory_assignments],
            }
            for condition in autonomous.conditions
        }
        summary = dict(exact)
        summary["plateau_observed"] = (
            training.plateau_state.first_plateau_epoch is not None
            and training.plateau_state.first_plateau_epoch <= update
        )
        summaries.append(summary)
        events.append({
            "event": "diagnostic_milestone_measurement",
            "arm": arm,
            "optimizer_update": update,
            "epoch_coordinate": update,
            "slurm_job_id": job_id,
            "checkpoint_identity": relative,
            "checkpoint_written_before_evaluation": True,
            "strict_reload": True,
            "autonomous_target_free_generation": True,
            "target_scoring_after_generation": True,
            "condition_metrics": list(metrics),
            "primary_reporting": ratios,
            "donor_template_agreement": donors,
            "memory_alteration_evidence": memory_evidence,
            "exact_sufficiency": summary,
            "access": tracker.declarations(),
        })
    for update, checkpoint_path in zip(CHECKPOINT_UPDATES, training.checkpoint_paths):
        provenance = _checkpoint_provenance(checkpoint_path)
        if provenance.get("slurm_job_id") != job_id:
            raise GraphEncoderError(
                "diagnostic_provenance_failure", "checkpoint Slurm job differs"
            )
        events.append({
            "event": "checkpoint_provenance",
            "arm": arm,
            "optimizer_update": update,
            "checkpoint_identity": Path(checkpoint_path).relative_to(staging).as_posix(),
            "slurm_job_id": provenance["slurm_job_id"],
            "git_commit": provenance.get("git_commit"),
            "source_tree_sha256": provenance.get("source_tree_sha256"),
        })
    return {"training_record": training_record, "milestone_summaries": tuple(summaries)}


def _load_frozen_train(corpus_dir, tracker):
    tracker.operation_template_train_payload_accessed = "not_confirmed_on_failure"
    examples = tuple(load_train(corpus_dir, FROZEN_FAMILY_IDS))
    tracker.operation_template_train_payload_accessed = True
    observed = tuple(sorted(item.physical_family_id for item in examples))
    if observed != FROZEN_FAMILY_IDS:
        raise GraphEncoderError(
            "diagnostic_payload_alignment_failure",
            "loaded payload differs from the frozen four-family cohort",
        )
    return examples


def _checkpoint_provenance(path):
    if torch is None:
        raise RuntimeError("checkpoint provenance validation requires PyTorch")
    payload = torch.load(str(path), map_location="cpu")
    provenance = payload.get("provenance") if isinstance(payload, dict) else None
    if not isinstance(provenance, dict):
        raise GraphEncoderError(
            "diagnostic_provenance_failure", "checkpoint provenance is missing"
        )
    return provenance


def _resolved_slurm_job_id(root):
    try:
        resolved = json.loads(
            (Path(root) / "resolved_config.json").read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as exc:
        raise GraphEncoderError(
            "invalid_diagnostic_artifact", "resolved configuration is unreadable"
        ) from exc
    return validate_slurm_job_id(resolved.get("runtime", {}).get("slurm_job_id"))


def _fixed_interpretation_fields():
    return {
        "flat_ever_exact_at_predetermined_milestone": False,
        "typed_graph_ever_exact_at_predetermined_milestone": False,
        "flat_first_exact_update": None,
        "typed_graph_first_exact_update": None,
        "flat_exact_at_update_500": False,
        "typed_graph_exact_at_update_500": False,
        "flat_plateau_observed": False,
        "typed_graph_plateau_observed": False,
        "optimization_budget_explanation": "execution did not complete",
        "original_c7_result_changed": False,
        "stage6_authorized": False,
        "decoder_repair_invoked": False,
        "c8_or_later_performed": False,
    }


def _exact_metric(value, field):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise GraphEncoderError(
            "malformed_diagnostic_metrics", "{} must be finite numeric".format(field)
        )
    return float(value)


def _assert_no_formal_c7_decision_fields(value):
    if isinstance(value, dict):
        overlap = set(value).intersection(FORBIDDEN_DECISION_FIELDS)
        if overlap:
            raise GraphEncoderError(
                "formal_c7_decision_mutation",
                "diagnostic contains forbidden formal C7 fields: {}".format(
                    ", ".join(sorted(overlap))
                ),
            )
        for child in value.values():
            _assert_no_formal_c7_decision_fields(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _assert_no_formal_c7_decision_fields(child)


def _require_authoritative_runtime():
    if torch is None:
        raise RuntimeError("optimization diagnostic requires PyTorch")
    if sys.version_info[:3] != (3, 8, 13):
        raise GraphEncoderError(
            "environment_mismatch", "Python 3.8.13 is required"
        )
    if str(torch.__version__).split("+")[0] != "1.11.0":
        raise GraphEncoderError(
            "environment_mismatch", "PyTorch 1.11.0 is required"
        )
    if torch.cuda.is_available():
        raise GraphEncoderError("environment_mismatch", "CPU execution is required")
    if torch.get_num_threads() != 1:
        raise GraphEncoderError(
            "environment_mismatch", "one PyTorch CPU thread is required"
        )


if __name__ == "__main__":
    raise SystemExit(main())
