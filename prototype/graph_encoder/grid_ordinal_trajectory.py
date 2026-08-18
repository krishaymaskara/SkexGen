"""Corpus-free actual-head/loss grid-ordinal optimization trajectory.

This engineering diagnostic intentionally instantiates only the unchanged
grid-magnitude head and loss.  It uses generated fixed decoder states, writes
no checkpoint, and makes no claim about a corpus-trained model or job 3352404.
"""

from __future__ import annotations

import argparse
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
except ImportError:  # Pure contract and artifact tests remain locally runnable.
    torch = None

from .config import grid_frozen_encoder_config
from .errors import GraphEncoderError
from .grid_magnitude import (
    GRID_MAGNITUDE_PARAMETERIZATION,
    NORMALIZED_GRIDS,
    OPERATION_NODE_TYPE_IDS,
    OPERATION_TYPES,
    PHYSICAL_GRIDS,
    SERIALIZED_CHANNELS,
    build_grid_magnitude_head,
    grid_contract_metadata,
)
if torch is not None:
    from .losses import grid_magnitude_terms
else:  # Preserve corpus-free pure contract and artifact validation locally.
    grid_magnitude_terms = None
from .pilot import (
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


DIAGNOSTIC_VERSION = "GE1-GRID-ORDINAL-TRAJECTORY-DIAGNOSTIC-v1"
TRAJECTORY_VERSION = "GE1-GRID-ORDINAL-TRAJECTORY-v1"
ARTIFACT_VERSION = "GE1-GRID-ORDINAL-TRAJECTORY-ARTIFACT-v1"
SEED = 2026
MODEL_WIDTH = 32
BATCH_SIZE = 8
NODES_PER_SAMPLE = 1
OPTIMIZER_UPDATES = 200
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0
GRADIENT_CLIP_NORM = 1.0
RAW_GAP_INITIALIZATIONS = (0.0, 0.5)
TARGET_CLASSES = (0, 1, 2, 3, 4)
PRIMARY_CLASSES = (1, 3)
EXPECTED_CONDITION_COUNT = 20
EXPECTED_TRAJECTORY_RECORD_COUNT = EXPECTED_CONDITION_COUNT * (
    OPTIMIZER_UPDATES + 1
)
_MISSING = object()

ACCESS_RECORD = {
    "corpus_accessed": False,
    "manifest_accessed": False,
    "train_payload_accessed": False,
    "preserved_feature_payload_accessed": False,
    "checkpoint_accessed": False,
    "model_artifact_accessed": False,
    "repaired_artifact_accessed": False,
    "cad_kernel_accessed": False,
    "protected_partition_accessed": False,
    "scientific_execution": False,
    "scientific_training": False,
    "scientific_inference": False,
    "engineering_only": True,
    "scientific_job_authorized": False,
    "repair_authorized": False,
    "another_job_authorized": False,
    "stage6_authorized": False,
    "c8_authorized": False,
}


def validate_slurm_job_id(value=_MISSING):
    observed = os.environ.get("SLURM_JOB_ID") if value is _MISSING else value
    if (
        not isinstance(observed, str)
        or not observed
        or not observed.isascii()
        or not observed.isdecimal()
    ):
        raise GraphEncoderError(
            "invalid_grid_trajectory_slurm_job_id",
            "SLURM_JOB_ID must be a nonempty decimal string",
        )
    return observed


def diagnostic_arithmetic():
    return {
        "seed": SEED,
        "model_width": MODEL_WIDTH,
        "batch_size": BATCH_SIZE,
        "nodes_per_sample": NODES_PER_SAMPLE,
        "optimizer_updates_per_condition": OPTIMIZER_UPDATES,
        "recorded_steps_per_condition": OPTIMIZER_UPDATES + 1,
        "condition_count": EXPECTED_CONDITION_COUNT,
        "trajectory_record_count": EXPECTED_TRAJECTORY_RECORD_COUNT,
        "optimizer": "AdamW",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "gradient_clip_norm": GRADIENT_CLIP_NORM,
        "fixed_generated_decoder_states": True,
        "generated_state_trainable": False,
        "head_parameters_only": True,
    }


def condition_matrix():
    rows = []
    for operation_type in OPERATION_TYPES:
        for raw_gap_initialization in RAW_GAP_INITIALIZATIONS:
            for target_class in TARGET_CLASSES:
                rows.append({
                    "condition_id": condition_id(
                        operation_type, raw_gap_initialization, target_class
                    ),
                    "operation_type": operation_type,
                    "raw_gap_initialization": raw_gap_initialization,
                    "target_class": target_class,
                    "role": (
                        "primary" if target_class in PRIMARY_CLASSES else "control"
                    ),
                    "fresh_seed": SEED,
                })
    if len(rows) != EXPECTED_CONDITION_COUNT:
        raise AssertionError("grid trajectory condition count differs")
    return tuple(rows)


def condition_id(operation_type, raw_gap_initialization, target_class):
    gap = str(float(raw_gap_initialization)).replace(".", "p")
    return "{}-gap{}-class{}".format(operation_type, gap, target_class)


def _require_runtime():
    if torch is None:
        raise RuntimeError("grid ordinal trajectory requires PyTorch")
    if sys.version_info[:3] != (3, 8, 13):
        raise GraphEncoderError("environment_mismatch", "Python 3.8.13 required")
    if str(torch.__version__).split("+")[0] != "1.11.0":
        raise GraphEncoderError("environment_mismatch", "PyTorch 1.11.0 required")
    if torch.cuda.is_available() or torch.get_num_threads() != 1:
        raise GraphEncoderError(
            "environment_mismatch", "one-thread CPU PyTorch required"
        )


def _fixed_states():
    states = torch.zeros(BATCH_SIZE, NODES_PER_SAMPLE, MODEL_WIDTH)
    states[..., 0] = 1.0
    states.requires_grad_(False)
    return states


def _generated_target(operation_type, target_class):
    position = OPERATION_TYPES.index(operation_type)
    node_type_ids = torch.full(
        (BATCH_SIZE, NODES_PER_SAMPLE),
        OPERATION_NODE_TYPE_IDS[position],
        dtype=torch.long,
    )
    node_mask = torch.ones(BATCH_SIZE, NODES_PER_SAMPLE, dtype=torch.bool)
    geometry = torch.zeros(BATCH_SIZE, NODES_PER_SAMPLE, 39)
    geometry_mask = torch.zeros(BATCH_SIZE, NODES_PER_SAMPLE, 39, dtype=torch.bool)
    channel = SERIALIZED_CHANNELS[operation_type]
    geometry[..., channel] = NORMALIZED_GRIDS[operation_type][target_class]
    geometry_mask[..., channel] = True
    return {
        "node_type_ids": node_type_ids,
        "node_mask": node_mask,
        "geometry": geometry,
        "geometry_mask": geometry_mask,
    }


def _vector_norm(value):
    if value is None:
        return 0.0
    return float(value.detach().double().norm().cpu().item())


def _gradient_record(head, position, global_before):
    projection = head.projections[position].weight.grad
    first_bias = head.first_bias.grad[position:position + 1]
    raw_gaps = head.bias_gaps.grad[position]
    return {
        "projection": _vector_norm(projection),
        "first_bias": _vector_norm(first_bias),
        "raw_gaps": _vector_norm(raw_gaps),
        "global_before_clipping": float(global_before),
        "clip_norm": GRADIENT_CLIP_NORM,
        "values_are_after_clipping": True,
    }


def _state_record(
    *, condition, step, head, states, target, config, gradient_update_follows
):
    position = OPERATION_TYPES.index(condition["operation_type"])
    logits = head(states)
    terms = grid_magnitude_terms(logits, target, config)
    loss = terms.weighted[condition["operation_type"]]
    loss.backward()
    global_before = torch.nn.utils.clip_grad_norm_(
        tuple(head.parameters()), GRADIENT_CLIP_NORM
    )
    target_logits = logits[0, 0, position]
    probabilities = torch.sigmoid(target_logits)
    decoded_class = int(terms.predicted_classes[condition["operation_type"]][0, 0])
    ordered_biases = head.ordered_biases()[position]
    projection_score = head.projections[position](states)[0, 0, 0]
    raw_gaps = head.bias_gaps[position]
    target_class = condition["target_class"]
    normalized = NORMALIZED_GRIDS[condition["operation_type"]]
    physical = PHYSICAL_GRIDS[condition["operation_type"]]
    values = {
        "event": "grid_ordinal_trajectory_step",
        "version": TRAJECTORY_VERSION,
        "condition_id": condition["condition_id"],
        "condition_role": condition["role"],
        "operation_type": condition["operation_type"],
        "target_class": target_class,
        "target_normalized_value": normalized[target_class],
        "target_physical_value": physical[target_class],
        "raw_gap_initialization": condition["raw_gap_initialization"],
        "step": step,
        "state_coordinate": (
            "initialization" if step == 0 else "post_optimizer_update"
        ),
        "optimizer_updates_completed": step,
        "gradient_update_follows_record": bool(gradient_update_follows),
        "loss": float(loss.detach().cpu().item()),
        "projection_score": float(projection_score.detach().cpu().item()),
        "first_bias": float(head.first_bias[position].detach().cpu().item()),
        "raw_gap_values": [
            float(value) for value in raw_gaps.detach().cpu().tolist()
        ],
        "ordered_biases": [
            float(value) for value in ordered_biases.detach().cpu().tolist()
        ],
        "logits": [float(value) for value in target_logits.detach().cpu().tolist()],
        "sigmoid_probabilities": [
            float(value) for value in probabilities.detach().cpu().tolist()
        ],
        "decoded_class": decoded_class,
        "decoded_normalized_value": normalized[decoded_class],
        "decoded_physical_value": physical[decoded_class],
        "minimum_decision_margin": float(
            target_logits.detach().abs().min().cpu().item()
        ),
        "gradient_norms": _gradient_record(head, position, global_before),
        "decoded_class_equals_target": decoded_class == target_class,
        "engineering_only": True,
        "scientific_training": False,
    }
    _assert_finite_trajectory_record(values)
    return values, loss


def _assert_finite_trajectory_record(record):
    for name in (
        "loss", "projection_score", "first_bias", "minimum_decision_margin"
    ):
        if not math.isfinite(record[name]):
            raise GraphEncoderError("nonfinite_grid_trajectory", name)
    for name in ("raw_gap_values", "ordered_biases", "logits", "sigmoid_probabilities"):
        if any(not math.isfinite(value) for value in record[name]):
            raise GraphEncoderError("nonfinite_grid_trajectory", name)
    if any(
        not math.isfinite(value)
        for key, value in record["gradient_norms"].items()
        if key not in ("values_are_after_clipping",)
    ):
        raise GraphEncoderError("nonfinite_grid_trajectory", "gradient_norms")


def run_condition(operation_type, raw_gap_initialization, target_class):
    """Run one independently initialized real-head/loss condition."""

    if operation_type not in OPERATION_TYPES:
        raise ValueError("unknown operation type")
    if raw_gap_initialization not in RAW_GAP_INITIALIZATIONS:
        raise ValueError("unknown raw-gap initialization")
    if target_class not in TARGET_CLASSES:
        raise ValueError("unknown target class")
    torch.manual_seed(SEED)
    head = build_grid_magnitude_head(MODEL_WIDTH)
    if raw_gap_initialization != 0.0:
        with torch.no_grad():
            head.bias_gaps.fill_(raw_gap_initialization)
    condition = next(
        row for row in condition_matrix()
        if row["operation_type"] == operation_type
        and row["raw_gap_initialization"] == raw_gap_initialization
        and row["target_class"] == target_class
    )
    states = _fixed_states()
    target = _generated_target(operation_type, target_class)
    config = grid_frozen_encoder_config("flat")
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    records = []
    for step in range(OPTIMIZER_UPDATES + 1):
        optimizer.zero_grad(set_to_none=True)
        record, unused_loss = _state_record(
            condition=condition,
            step=step,
            head=head,
            states=states,
            target=target,
            config=config,
            gradient_update_follows=step < OPTIMIZER_UPDATES,
        )
        records.append(record)
        if step < OPTIMIZER_UPDATES:
            optimizer.step()
    if states.grad is not None or states.requires_grad:
        raise GraphEncoderError(
            "generated_state_became_trainable", "fixed state must remain frozen"
        )
    return tuple(records)


def summarize_condition(records):
    values = tuple(records)
    if len(values) != OPTIMIZER_UPDATES + 1:
        raise GraphEncoderError("incomplete_grid_trajectory", "step count differs")
    expected_steps = tuple(range(OPTIMIZER_UPDATES + 1))
    if tuple(row["step"] for row in values) != expected_steps:
        raise GraphEncoderError("incomplete_grid_trajectory", "steps differ")
    first = values[0]
    target_class = first["target_class"]
    correct_steps = [
        row["step"] for row in values if row["decoded_class_equals_target"]
    ]
    first_correct = correct_steps[0] if correct_steps else None
    transitions = []
    previous = first["decoded_class"]
    for row in values[1:]:
        if row["decoded_class"] != previous:
            transitions.append({
                "step": row["step"],
                "from_class": previous,
                "to_class": row["decoded_class"],
            })
            previous = row["decoded_class"]
    persisted = (
        None if first_correct is None else all(
            row["decoded_class_equals_target"]
            for row in values[first_correct:]
        )
    )
    final = values[-1]
    return {
        "condition_id": first["condition_id"],
        "condition_role": first["condition_role"],
        "operation_type": first["operation_type"],
        "target_class": target_class,
        "target_normalized_value": first["target_normalized_value"],
        "target_physical_value": first["target_physical_value"],
        "raw_gap_initialization": first["raw_gap_initialization"],
        "initial_decoded_class": first["decoded_class"],
        "final_decoded_class": final["decoded_class"],
        "first_correct_step": first_correct,
        "correctness_persisted_after_first_correct": persisted,
        "class_transitions": transitions,
        "final_loss": final["loss"],
        "final_projection_score": final["projection_score"],
        "final_raw_gap_values": final["raw_gap_values"],
        "final_ordered_biases": final["ordered_biases"],
        "final_logits": final["logits"],
        "final_sigmoid_probabilities": final["sigmoid_probabilities"],
        "final_minimum_decision_margin": final["minimum_decision_margin"],
        "class_1_remained_class_0_through_step_200": (
            all(row["decoded_class"] == 0 for row in values)
            if target_class == 1 else None
        ),
        "class_3_remained_class_4_through_step_200": (
            all(row["decoded_class"] == 4 for row in values)
            if target_class == 3 else None
        ),
        "diagnostic_gap_0p5_corrected_by_step_200": (
            final["decoded_class_equals_target"]
            if first["raw_gap_initialization"] == 0.5
            and target_class in PRIMARY_CLASSES else None
        ),
    }


def hypothesis_summary(condition_summaries):
    indexed = {row["condition_id"]: row for row in condition_summaries}
    class1_final = {}
    class3_final = {}
    class1_all_steps = {}
    class3_all_steps = {}
    corrected = {}
    for operation_type in OPERATION_TYPES:
        class1_summary = indexed[condition_id(operation_type, 0.0, 1)]
        class3_summary = indexed[condition_id(operation_type, 0.0, 3)]
        class1_final[operation_type] = (
            class1_summary["final_decoded_class"] == 0
        )
        class3_final[operation_type] = (
            class3_summary["final_decoded_class"] == 4
        )
        class1_all_steps[operation_type] = class1_summary[
            "class_1_remained_class_0_through_step_200"
        ]
        class3_all_steps[operation_type] = class3_summary[
            "class_3_remained_class_4_through_step_200"
        ]
        corrected[operation_type] = {
            str(target_class): indexed[
                condition_id(operation_type, 0.5, target_class)
            ]["diagnostic_gap_0p5_corrected_by_step_200"]
            for target_class in PRIMARY_CLASSES
        }
    trap = all(class1_final.values()) and all(class3_final.values())
    intervention = all(
        value for operation in corrected.values() for value in operation.values()
    )
    if trap and intervention:
        category = "predicted_trap_and_gap_intervention_pattern_observed"
    elif trap:
        category = "predicted_zero_gap_trap_observed_intervention_not_uniform"
    else:
        category = "predicted_zero_gap_trap_not_uniformly_observed"
    return {
        "interpretation_category": category,
        "zero_gap_class_1_final_class_0": class1_final,
        "zero_gap_class_3_final_class_4": class3_final,
        "zero_gap_class_1_remained_class_0_all_steps": class1_all_steps,
        "zero_gap_class_3_remained_class_4_all_steps": class3_all_steps,
        "diagnostic_gap_0p5_corrected_by_step_200": corrected,
        "predicted_zero_gap_trap_observed_at_step_200_for_both_operations": trap,
        "diagnostic_gap_intervention_corrected_all_primary_conditions": intervention,
        "outcome_is_an_artifact_validity_gate": False,
    }


def _prepare_output(output_dir, repository_root, job_id):
    final = Path(output_dir).resolve()
    repository = Path(repository_root).resolve()
    if final.exists() or final.is_symlink():
        raise GraphEncoderError("grid_trajectory_output_exists", str(final))
    if _is_within(final, repository):
        raise GraphEncoderError(
            "unsafe_grid_trajectory_output", "artifact must be outside repository"
        )
    if not final.parent.is_dir() or final.parent.is_symlink():
        raise GraphEncoderError(
            "unsafe_grid_trajectory_output", "artifact parent must already exist"
        )
    staging = final.with_name(final.name + ".incomplete-" + job_id)
    if staging.exists() or staging.is_symlink():
        raise GraphEncoderError("grid_trajectory_staging_exists", str(staging))
    staging.mkdir()
    return final, staging


def run_grid_ordinal_trajectory(*, output_dir, repository_root, expected_commit):
    _require_runtime()
    job_id = validate_slurm_job_id()
    source = _source_identity(repository_root, expected_commit)
    final, staging = _prepare_output(output_dir, repository_root, job_id)
    started = time.perf_counter()
    records = []
    summaries = []
    for condition in condition_matrix():
        condition_records = run_condition(
            condition["operation_type"],
            condition["raw_gap_initialization"],
            condition["target_class"],
        )
        records.extend(condition_records)
        summaries.append(summarize_condition(condition_records))
    interpretation = hypothesis_summary(summaries)
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
            "cpu_threads": torch.get_num_threads(),
            "host": socket.gethostname(),
            "slurm_job_id": job_id,
        },
        "arithmetic": diagnostic_arithmetic(),
        "condition_matrix": list(condition_matrix()),
        "grid_contract": grid_contract_metadata(),
        "configuration": grid_frozen_encoder_config("flat").to_dict(),
        "generated_state": {
            "shape": [BATCH_SIZE, NODES_PER_SAMPLE, MODEL_WIDTH],
            "first_component": 1.0,
            "remaining_components": 0.0,
            "fixed": True,
            "requires_grad": False,
        },
        "implementation_reuse": {
            "head": "prototype.graph_encoder.grid_magnitude.build_grid_magnitude_head",
            "loss": "prototype.graph_encoder.losses.grid_magnitude_terms",
            "head_reimplemented": False,
            "loss_reimplemented": False,
            "source_default_changed": False,
            "gap_0p5_is_in_memory_diagnostic_only": True,
        },
        "access": dict(ACCESS_RECORD),
    }
    summary = {
        "diagnostic_version": DIAGNOSTIC_VERSION,
        "trajectory_version": TRAJECTORY_VERSION,
        "artifact_version": ARTIFACT_VERSION,
        "source_commit": expected_commit,
        "slurm_job_id": job_id,
        "condition_count": len(summaries),
        "trajectory_record_count": len(records),
        "conditions": summaries,
        "hypothesis_evaluation": interpretation,
        "interpretation_boundary": {
            "actual_repository_head_and_loss_on_generated_fixed_states_only": True,
            "job_3352404_checkpoint_behavior_proven": False,
            "full_model_or_encoder_behavior_proven": False,
            "corpus_behavior_proven": False,
            "exact_defect_identified": False,
            "initialization_change_authorized": False,
        },
        "diagnostic_completed": True,
        "elapsed_seconds": time.perf_counter() - started,
        **dict(ACCESS_RECORD),
    }
    _atomic_write_json(staging / "resolved_config.json", resolved)
    _atomic_write_jsonl(staging / "trajectory.jsonl", records)
    _atomic_write_json(staging / "summary.json", summary)
    _verify_source_unchanged(source, repository_root, expected_commit)
    finalize_artifact(staging, final)
    verified = verify_artifact(
        final, expected_commit=expected_commit, expected_slurm_job_id=job_id
    )
    return {
        "artifact_path": str(final),
        "artifact_verification": verified,
        "hypothesis_evaluation": interpretation,
        "completion": {
            "event": "grid_ordinal_trajectory_completed",
            "commit": expected_commit,
            "job_id": job_id,
            "diagnostic_completed": True,
            **dict(ACCESS_RECORD),
        },
    }


def finalize_artifact(staging_dir, final_dir):
    staging = Path(staging_dir)
    final = Path(final_dir)
    if staging.parent.resolve() != final.parent.resolve():
        raise GraphEncoderError("unsafe_grid_trajectory_output", "parents differ")
    if final.exists() or final.is_symlink():
        raise GraphEncoderError("grid_trajectory_output_exists", str(final))
    required = {
        "resolved_config.json", "trajectory.jsonl", "summary.json"
    }
    ordinary = _regular_artifact_files(
        staging, excluded=("artifact_manifest.json", "SHA256SUMS")
    )
    if set(ordinary) != required:
        raise GraphEncoderError(
            "incomplete_grid_trajectory_artifact", "ordinary files differ"
        )
    manifest = {
        "schema_version": ARTIFACT_VERSION,
        "diagnostic_version": DIAGNOSTIC_VERSION,
        "trajectory_version": TRAJECTORY_VERSION,
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
    checksums = "".join(
        "{}  {}\n".format(_file_sha256(staging / relative), relative)
        for relative in checksum_paths
    )
    _atomic_write_bytes(staging / "SHA256SUMS", checksums.encode("utf-8"))
    job_id = json.loads(
        (staging / "resolved_config.json").read_text(encoding="utf-8")
    )["runtime"]["slurm_job_id"]
    commit = json.loads(
        (staging / "resolved_config.json").read_text(encoding="utf-8")
    )["source"]["git_commit"]
    verify_artifact(
        staging,
        expected_commit=commit,
        expected_slurm_job_id=job_id,
        allow_incomplete_name=True,
    )
    os.replace(str(staging), str(final))
    return final


def _canonical_json_file(path):
    raw = Path(path).read_bytes()
    if not raw.endswith(b"\n"):
        raise GraphEncoderError("invalid_grid_trajectory_artifact", "final LF absent")
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise GraphEncoderError(
            "invalid_grid_trajectory_artifact", "JSON unreadable"
        ) from exc
    if raw.decode("utf-8") != _canonical_json_text(value) + "\n":
        raise GraphEncoderError(
            "invalid_grid_trajectory_artifact", "JSON is not canonical"
        )
    return value


def verify_artifact(
    path, *, expected_commit, expected_slurm_job_id, allow_incomplete_name=False
):
    job_id = validate_slurm_job_id(expected_slurm_job_id)
    root = Path(path)
    if not root.is_dir() or root.is_symlink():
        raise GraphEncoderError("invalid_grid_trajectory_artifact", "root differs")
    if ".incomplete-" in root.name and not allow_incomplete_name:
        raise GraphEncoderError("invalid_grid_trajectory_artifact", "incomplete root")
    expected_names = {
        "SHA256SUMS", "artifact_manifest.json", "resolved_config.json",
        "summary.json", "trajectory.jsonl",
    }
    observed_names = {
        item.relative_to(root).as_posix()
        for item in root.rglob("*") if item.is_file()
    }
    if observed_names != expected_names:
        raise GraphEncoderError(
            "invalid_grid_trajectory_artifact", "exact five-file set differs"
        )
    manifest = _canonical_json_file(root / "artifact_manifest.json")
    resolved = _canonical_json_file(root / "resolved_config.json")
    summary = _canonical_json_file(root / "summary.json")
    if (
        manifest.get("schema_version") != ARTIFACT_VERSION
        or manifest.get("diagnostic_version") != DIAGNOSTIC_VERSION
        or manifest.get("trajectory_version") != TRAJECTORY_VERSION
    ):
        raise GraphEncoderError("invalid_grid_trajectory_artifact", "identity differs")
    listed = []
    for row in manifest.get("artifacts", []):
        if set(row) != {"path", "byte_size", "sha256"}:
            raise GraphEncoderError("invalid_grid_trajectory_artifact", "manifest row")
        relative = _safe_relative_path(row["path"])
        target = root / relative
        if (
            not target.is_file() or target.is_symlink()
            or target.stat().st_size != row["byte_size"]
            or _file_sha256(target) != row["sha256"]
        ):
            raise GraphEncoderError(
                "grid_trajectory_artifact_integrity_failure", relative
            )
        listed.append(relative)
    ordinary = _regular_artifact_files(
        root, excluded=("artifact_manifest.json", "SHA256SUMS")
    )
    if tuple(listed) != ordinary:
        raise GraphEncoderError("invalid_grid_trajectory_artifact", "coverage differs")
    checksum_raw = (root / "SHA256SUMS").read_bytes()
    if not checksum_raw.endswith(b"\n"):
        raise GraphEncoderError("invalid_grid_trajectory_artifact", "checksum LF")
    checksum_paths = []
    for line in checksum_raw.decode("utf-8").splitlines():
        parts = line.split("  ", 1)
        if len(parts) != 2 or len(parts[0]) != 64:
            raise GraphEncoderError("invalid_grid_trajectory_artifact", "checksum row")
        relative = _safe_relative_path(parts[1])
        if _file_sha256(root / relative) != parts[0]:
            raise GraphEncoderError(
                "grid_trajectory_artifact_integrity_failure", relative
            )
        checksum_paths.append(relative)
    expected_sums = _regular_artifact_files(root, excluded=("SHA256SUMS",))
    if tuple(checksum_paths) != expected_sums:
        raise GraphEncoderError("invalid_grid_trajectory_artifact", "checksum coverage")
    trajectory = _read_canonical_jsonl(root / "trajectory.jsonl")
    conditions = condition_matrix()
    if len(trajectory) != EXPECTED_TRAJECTORY_RECORD_COUNT:
        raise GraphEncoderError("incomplete_grid_trajectory", "record count differs")
    for condition in conditions:
        rows = [
            row for row in trajectory
            if row.get("condition_id") == condition["condition_id"]
        ]
        if (
            len(rows) != OPTIMIZER_UPDATES + 1
            or tuple(row.get("step") for row in rows)
            != tuple(range(OPTIMIZER_UPDATES + 1))
            or any(row.get("scientific_training") is not False for row in rows)
        ):
            raise GraphEncoderError("incomplete_grid_trajectory", "condition differs")
    runtime = resolved.get("runtime", {})
    if (
        resolved.get("diagnostic_version") != DIAGNOSTIC_VERSION
        or resolved.get("trajectory_version") != TRAJECTORY_VERSION
        or resolved.get("artifact_version") != ARTIFACT_VERSION
        or resolved.get("arithmetic") != diagnostic_arithmetic()
        or resolved.get("condition_matrix") != list(conditions)
        or resolved.get("source", {}).get("git_commit") != expected_commit
        or resolved.get("source", {}).get("git_branch") is not None
        or resolved.get("source", {}).get("detached_head") is not True
        or resolved.get("source", {}).get("git_dirty") is not False
        or resolved.get("source", {}).get("git_status_porcelain") != []
        or runtime.get("slurm_job_id") != job_id
        or runtime.get("python") != "3.8.13"
        or str(runtime.get("pytorch", "")).split("+")[0] != "1.11.0"
        or runtime.get("device") != "cpu"
        or runtime.get("cuda_available") is not False
        or runtime.get("cpu_threads") != 1
    ):
        raise GraphEncoderError("invalid_grid_trajectory_artifact", "resolved differs")
    if (
        summary.get("source_commit") != expected_commit
        or summary.get("slurm_job_id") != job_id
        or summary.get("condition_count") != EXPECTED_CONDITION_COUNT
        or summary.get("trajectory_record_count") != EXPECTED_TRAJECTORY_RECORD_COUNT
        or summary.get("diagnostic_completed") is not True
        or len(summary.get("conditions", [])) != EXPECTED_CONDITION_COUNT
    ):
        raise GraphEncoderError("invalid_grid_trajectory_artifact", "summary differs")
    for record in (resolved.get("access", {}), summary):
        if any(record.get(key) != value for key, value in ACCESS_RECORD.items()):
            raise GraphEncoderError(
                "invalid_grid_trajectory_artifact", "authority boundary differs"
            )
    return {
        "diagnostic_version": DIAGNOSTIC_VERSION,
        "source_commit": expected_commit,
        "slurm_job_id": job_id,
        "condition_count": EXPECTED_CONDITION_COUNT,
        "trajectory_record_count": EXPECTED_TRAJECTORY_RECORD_COUNT,
        "regular_file_count": len(expected_sums) + 1,
        "artifact_manifest_sha256": _file_sha256(root / "artifact_manifest.json"),
        "sha256sums_sha256": _file_sha256(root / "SHA256SUMS"),
        "interpretation_category": summary["hypothesis_evaluation"][
            "interpretation_category"
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--expected-commit", required=True)
    arguments = parser.parse_args(argv)
    try:
        result = run_grid_ordinal_trajectory(
            output_dir=arguments.output_dir,
            repository_root=arguments.repository_root,
            expected_commit=arguments.expected_commit,
        )
    except Exception as exc:
        failure = {
            "event": "grid_ordinal_trajectory_infrastructure_failure",
            "failure_type": type(exc).__name__,
            "failure_code": getattr(exc, "code", None),
            "detail": str(exc),
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "diagnostic_completed": False,
            **dict(ACCESS_RECORD),
        }
        print(_canonical_json_text(failure), file=sys.stderr, flush=True)
        return 1
    print(_canonical_json_text(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
