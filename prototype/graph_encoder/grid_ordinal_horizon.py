"""Corpus-free extended-horizon diagnostic for the real grid ordinal head."""

from __future__ import annotations

import argparse
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
except ImportError:  # Pure contracts and artifact verification remain runnable.
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
else:
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


DIAGNOSTIC_VERSION = "GE1-GRID-ORDINAL-HORIZON-DIAGNOSTIC-v1"
TRAJECTORY_VERSION = "GE1-GRID-ORDINAL-HORIZON-TRAJECTORY-v1"
ARTIFACT_VERSION = "GE1-GRID-ORDINAL-HORIZON-ARTIFACT-v1"
SEED = 2026
MODEL_WIDTH = 32
BATCH_SIZE = 8
NODES_PER_SAMPLE = 1
OPTIMIZER_UPDATES = 2000
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0
GRADIENT_CLIP_NORM = 1.0
RAW_GAP_INITIALIZATION = 0.0
TARGET_CLASSES = (2, 3, 4)
CHECKPOINT_STEPS = (200, 500, 1000, 2000)
LOGIT_CHANGE_INTERVALS = ((200, 500), (500, 1000), (1000, 2000))
EXPECTED_CONDITION_COUNT = len(OPERATION_TYPES) * len(TARGET_CLASSES)
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
            "invalid_grid_horizon_slurm_job_id",
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
        "raw_gap_initialization": RAW_GAP_INITIALIZATION,
        "target_classes": list(TARGET_CLASSES),
        "checkpoint_steps": list(CHECKPOINT_STEPS),
        "logit_change_intervals": [list(value) for value in LOGIT_CHANGE_INTERVALS],
        "fixed_generated_decoder_states": True,
        "generated_state_trainable": False,
        "head_parameters_only": True,
    }


def condition_id(operation_type, target_class):
    return "{}-class{}".format(operation_type, target_class)


def condition_matrix():
    rows = tuple({
        "condition_id": condition_id(operation_type, target_class),
        "operation_type": operation_type,
        "target_class": target_class,
        "raw_gap_initialization": RAW_GAP_INITIALIZATION,
        "fresh_seed": SEED,
    } for operation_type in OPERATION_TYPES for target_class in TARGET_CLASSES)
    if len(rows) != EXPECTED_CONDITION_COUNT:
        raise AssertionError("grid horizon condition count differs")
    return rows


def _require_runtime():
    if torch is None:
        raise RuntimeError("grid ordinal horizon requires PyTorch")
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


def _tensor_values(value):
    return [float(item) for item in value.detach().reshape(-1).cpu().tolist()]


def _vector_norm(value):
    return float(value.detach().double().norm().cpu().item())


def _state_step(value):
    if hasattr(value, "item"):
        value = value.item()
    return int(value)


def _optimizer_tensor_state(optimizer, parameter, name):
    state = optimizer.state.get(parameter, {})
    value = state.get(name)
    if value is None:
        return torch.zeros_like(parameter)
    return value


def _optimizer_state_record(optimizer, head, position):
    projection = head.projections[position].weight
    first_bias = head.first_bias
    raw_gaps = head.bias_gaps

    projection_state = optimizer.state.get(projection, {})
    first_bias_state = optimizer.state.get(first_bias, {})
    raw_gap_state = optimizer.state.get(raw_gaps, {})
    steps = {
        _state_step(value.get("step", 0))
        for value in (projection_state, first_bias_state, raw_gap_state)
    }
    if len(steps) != 1:
        raise GraphEncoderError(
            "grid_horizon_optimizer_state_diverged", "AdamW steps differ"
        )
    return {
        "step": steps.pop(),
        "projection": {
            "exp_avg": _tensor_values(
                _optimizer_tensor_state(optimizer, projection, "exp_avg")
            ),
            "exp_avg_sq": _tensor_values(
                _optimizer_tensor_state(optimizer, projection, "exp_avg_sq")
            ),
        },
        "first_bias": {
            "exp_avg": float(_optimizer_tensor_state(
                optimizer, first_bias, "exp_avg"
            )[position].detach().cpu().item()),
            "exp_avg_sq": float(_optimizer_tensor_state(
                optimizer, first_bias, "exp_avg_sq"
            )[position].detach().cpu().item()),
        },
        "raw_gaps": {
            "exp_avg": _tensor_values(_optimizer_tensor_state(
                optimizer, raw_gaps, "exp_avg"
            )[position]),
            "exp_avg_sq": _tensor_values(_optimizer_tensor_state(
                optimizer, raw_gaps, "exp_avg_sq"
            )[position]),
        },
    }


def _gradient_state(head, position):
    return {
        "projection": head.projections[position].weight.grad.detach().clone(),
        "first_bias": head.first_bias.grad[position:position + 1].detach().clone(),
        "raw_gaps": head.bias_gaps.grad[position].detach().clone(),
    }


def _gradient_group_record(before, after):
    return {
        "before_clipping": _tensor_values(before),
        "after_clipping": _tensor_values(after),
        "before_clipping_norm": _vector_norm(before),
        "after_clipping_norm": _vector_norm(after),
    }


def _global_gradient_norm(parameters):
    squared = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            squared += float(
                parameter.grad.detach().double().pow(2).sum().cpu().item()
            )
    return math.sqrt(squared)


def _gradient_record(head, position):
    parameters = tuple(head.parameters())
    before = _gradient_state(head, position)
    global_before = torch.nn.utils.clip_grad_norm_(
        parameters, GRADIENT_CLIP_NORM
    )
    after = _gradient_state(head, position)
    global_before_value = float(global_before.detach().cpu().item())
    global_after_value = _global_gradient_norm(parameters)
    return {
        "projection": _gradient_group_record(
            before["projection"], after["projection"]
        ),
        "first_bias": _gradient_group_record(
            before["first_bias"], after["first_bias"]
        ),
        "raw_gaps": _gradient_group_record(
            before["raw_gaps"], after["raw_gaps"]
        ),
        "global_before_clipping": global_before_value,
        "global_after_clipping": global_after_value,
        "clip_norm": GRADIENT_CLIP_NORM,
        "clipping_occurred": global_before_value > GRADIENT_CLIP_NORM,
    }


def _state_record(
    *, condition, step, head, optimizer, states, target, config,
    gradient_update_follows
):
    position = OPERATION_TYPES.index(condition["operation_type"])
    logits = head(states)
    terms = grid_magnitude_terms(logits, target, config)
    loss = terms.weighted[condition["operation_type"]]
    loss.backward()
    gradients = _gradient_record(head, position)
    target_logits = logits[0, 0, position]
    probabilities = torch.sigmoid(target_logits)
    decoded_class = int(terms.predicted_classes[condition["operation_type"]][0, 0])
    ordered_biases = head.ordered_biases()[position]
    projection_score = head.projections[position](states)[0, 0, 0]
    raw_gaps = head.bias_gaps[position]
    target_class = condition["target_class"]
    operation_type = condition["operation_type"]
    normalized = NORMALIZED_GRIDS[operation_type]
    physical = PHYSICAL_GRIDS[operation_type]
    record = {
        "event": "grid_ordinal_horizon_step",
        "version": TRAJECTORY_VERSION,
        "condition_id": condition["condition_id"],
        "operation_type": operation_type,
        "target_class": target_class,
        "target_normalized_value": normalized[target_class],
        "target_physical_value": physical[target_class],
        "raw_gap_initialization": RAW_GAP_INITIALIZATION,
        "step": step,
        "state_coordinate": (
            "initialization" if step == 0 else "post_optimizer_update"
        ),
        "optimizer_updates_completed": step,
        "gradient_update_follows_record": bool(gradient_update_follows),
        "loss": float(loss.detach().cpu().item()),
        "projection_score": float(projection_score.detach().cpu().item()),
        "first_bias": float(head.first_bias[position].detach().cpu().item()),
        "raw_gap_values": _tensor_values(raw_gaps),
        "ordered_biases": _tensor_values(ordered_biases),
        "logits": _tensor_values(target_logits),
        "sigmoid_probabilities": _tensor_values(probabilities),
        "decoded_class": decoded_class,
        "decoded_normalized_value": normalized[decoded_class],
        "decoded_physical_value": physical[decoded_class],
        "minimum_decision_margin": float(
            target_logits.detach().abs().min().cpu().item()
        ),
        "gradients": gradients,
        "adamw_state": _optimizer_state_record(optimizer, head, position),
        "decoded_class_equals_target": decoded_class == target_class,
        "engineering_only": True,
        "scientific_training": False,
    }
    _assert_finite_record(record)
    return record


def _assert_finite_record(record):
    scalar_paths = (
        record["loss"], record["projection_score"], record["first_bias"],
        record["minimum_decision_margin"],
        record["gradients"]["global_before_clipping"],
        record["gradients"]["global_after_clipping"],
    )
    if any(not math.isfinite(value) for value in scalar_paths):
        raise GraphEncoderError("nonfinite_grid_horizon", "scalar")
    sequences = [
        record["raw_gap_values"], record["ordered_biases"], record["logits"],
        record["sigmoid_probabilities"],
    ]
    for group in ("projection", "first_bias", "raw_gaps"):
        values = record["gradients"][group]
        sequences.extend((values["before_clipping"], values["after_clipping"]))
    for group in ("projection", "raw_gaps"):
        values = record["adamw_state"][group]
        sequences.extend((values["exp_avg"], values["exp_avg_sq"]))
    sequences.extend((
        [record["adamw_state"]["first_bias"]["exp_avg"]],
        [record["adamw_state"]["first_bias"]["exp_avg_sq"]],
    ))
    if any(not math.isfinite(value) for values in sequences for value in values):
        raise GraphEncoderError("nonfinite_grid_horizon", "sequence")


def _validate_loaded_record(record, condition):
    gradients = record.get("gradients", {})
    adamw = record.get("adamw_state", {})
    if (
        record.get("version") != TRAJECTORY_VERSION
        or record.get("condition_id") != condition["condition_id"]
        or record.get("operation_type") != condition["operation_type"]
        or record.get("target_class") != condition["target_class"]
        or record.get("raw_gap_initialization") != RAW_GAP_INITIALIZATION
        or record.get("optimizer_updates_completed") != record.get("step")
        or record.get("gradient_update_follows_record")
        != (record.get("step") < OPTIMIZER_UPDATES)
        or adamw.get("step") != record.get("step")
        or record.get("engineering_only") is not True
        or record.get("scientific_training") is not False
        or len(record.get("raw_gap_values", [])) != 3
        or len(record.get("ordered_biases", [])) != 4
        or len(record.get("logits", [])) != 4
        or len(record.get("sigmoid_probabilities", [])) != 4
        or len(gradients.get("projection", {}).get("before_clipping", [])) != 32
        or len(gradients.get("projection", {}).get("after_clipping", [])) != 32
        or len(gradients.get("first_bias", {}).get("before_clipping", [])) != 1
        or len(gradients.get("first_bias", {}).get("after_clipping", [])) != 1
        or len(gradients.get("raw_gaps", {}).get("before_clipping", [])) != 3
        or len(gradients.get("raw_gaps", {}).get("after_clipping", [])) != 3
        or gradients.get("clip_norm") != GRADIENT_CLIP_NORM
        or not isinstance(gradients.get("clipping_occurred"), bool)
        or len(adamw.get("projection", {}).get("exp_avg", [])) != 32
        or len(adamw.get("projection", {}).get("exp_avg_sq", [])) != 32
        or len(adamw.get("raw_gaps", {}).get("exp_avg", [])) != 3
        or len(adamw.get("raw_gaps", {}).get("exp_avg_sq", [])) != 3
    ):
        raise GraphEncoderError("invalid_grid_horizon_artifact", "record differs")
    _assert_finite_record(record)


def run_condition(operation_type, target_class):
    if operation_type not in OPERATION_TYPES:
        raise ValueError("unknown operation type")
    if target_class not in TARGET_CLASSES:
        raise ValueError("unknown target class")
    torch.manual_seed(SEED)
    head = build_grid_magnitude_head(MODEL_WIDTH)
    condition = next(
        row for row in condition_matrix()
        if row["operation_type"] == operation_type
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
        records.append(_state_record(
            condition=condition,
            step=step,
            head=head,
            optimizer=optimizer,
            states=states,
            target=target,
            config=config,
            gradient_update_follows=step < OPTIMIZER_UPDATES,
        ))
        if step < OPTIMIZER_UPDATES:
            optimizer.step()
    if states.requires_grad or states.grad is not None:
        raise GraphEncoderError(
            "generated_state_became_trainable", "fixed state must remain frozen"
        )
    return tuple(records)


def _checkpoint_snapshot(record):
    gradients = record["gradients"]
    adamw = record["adamw_state"]
    return {
        "step": record["step"],
        "loss": record["loss"],
        "logits": record["logits"],
        "sigmoid_probabilities": record["sigmoid_probabilities"],
        "decoded_class": record["decoded_class"],
        "decoded_normalized_value": record["decoded_normalized_value"],
        "decoded_physical_value": record["decoded_physical_value"],
        "projection_score": record["projection_score"],
        "first_bias": record["first_bias"],
        "raw_gap_values": record["raw_gap_values"],
        "ordered_biases": record["ordered_biases"],
        "minimum_decision_margin": record["minimum_decision_margin"],
        "gradient_global_before_clipping": gradients["global_before_clipping"],
        "gradient_global_after_clipping": gradients["global_after_clipping"],
        "clipping_occurred": gradients["clipping_occurred"],
        "adamw_step": adamw["step"],
        "adamw_projection_exp_avg_norm": math.sqrt(sum(
            value * value for value in adamw["projection"]["exp_avg"]
        )),
        "adamw_projection_exp_avg_sq_norm": math.sqrt(sum(
            value * value for value in adamw["projection"]["exp_avg_sq"]
        )),
        "adamw_first_bias_exp_avg": adamw["first_bias"]["exp_avg"],
        "adamw_first_bias_exp_avg_sq": adamw["first_bias"]["exp_avg_sq"],
        "adamw_raw_gap_exp_avg": adamw["raw_gaps"]["exp_avg"],
        "adamw_raw_gap_exp_avg_sq": adamw["raw_gaps"]["exp_avg_sq"],
        "decoded_class_equals_target": record["decoded_class_equals_target"],
    }


def summarize_condition(records):
    values = tuple(records)
    if len(values) != OPTIMIZER_UPDATES + 1:
        raise GraphEncoderError("incomplete_grid_horizon", "step count differs")
    if tuple(row["step"] for row in values) != tuple(range(OPTIMIZER_UPDATES + 1)):
        raise GraphEncoderError("incomplete_grid_horizon", "steps differ")
    first = values[0]
    final = values[-1]
    correct = [row["step"] for row in values if row["decoded_class_equals_target"]]
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
    first_positive = {
        str(cut): next(
            (row["step"] for row in values if row["logits"][cut] > 0.0),
            None,
        )
        for cut in range(4)
    }
    checkpoints = {
        str(step): _checkpoint_snapshot(values[step]) for step in CHECKPOINT_STEPS
    }
    changes = {}
    for start, end in LOGIT_CHANGE_INTERVALS:
        delta = [
            values[end]["logits"][index] - values[start]["logits"][index]
            for index in range(4)
        ]
        changes["{}-{}".format(start, end)] = {
            "logit_delta": delta,
            "max_absolute_logit_delta": max(abs(value) for value in delta),
            "loss_delta": values[end]["loss"] - values[start]["loss"],
            "decoded_class_changed": (
                values[end]["decoded_class"] != values[start]["decoded_class"]
            ),
        }
    crossed_zero = {
        str(cut): any(
            values[step - 1]["logits"][cut] <= 0.0
            and values[step]["logits"][cut] > 0.0
            for step in range(1, len(values))
        )
        for cut in range(4)
    }
    return {
        "condition_id": first["condition_id"],
        "operation_type": first["operation_type"],
        "target_class": first["target_class"],
        "target_normalized_value": first["target_normalized_value"],
        "target_physical_value": first["target_physical_value"],
        "raw_gap_initialization": first["raw_gap_initialization"],
        "initial_decoded_class": first["decoded_class"],
        "final_decoded_class": final["decoded_class"],
        "first_correct_step": correct[0] if correct else None,
        "correctness_persisted_after_first_correct": (
            None if not correct else all(
                row["decoded_class_equals_target"] for row in values[correct[0]:]
            )
        ),
        "class_transitions": transitions,
        "first_positive_step_by_cut": first_positive,
        "cut_crossed_zero": crossed_zero,
        "upper_cuts_crossed_zero": {
            str(cut): crossed_zero[str(cut)] for cut in range(1, 4)
        },
        "checkpoint_values": checkpoints,
        "interval_changes": changes,
        "final_class_correct": final["decoded_class_equals_target"],
        "final_loss": final["loss"],
        "final_logits": final["logits"],
        "final_sigmoid_probabilities": final["sigmoid_probabilities"],
        "final_projection_score": final["projection_score"],
        "final_first_bias": final["first_bias"],
        "final_raw_gap_values": final["raw_gap_values"],
        "final_ordered_biases": final["ordered_biases"],
        "final_minimum_decision_margin": final["minimum_decision_margin"],
        "clipping_record_count": sum(
            row["gradients"]["clipping_occurred"] for row in values
        ),
        "no_class_transition_from_1000_through_2000": all(
            row["decoded_class"] == values[1000]["decoded_class"]
            for row in values[1000:]
        ),
    }


def horizon_summary(condition_summaries):
    values = tuple(condition_summaries)
    final_correct = [row["condition_id"] for row in values if row["final_class_correct"]]
    correct_by_checkpoint = {
        str(step): [
            row["condition_id"] for row in values
            if row["checkpoint_values"][str(step)]["decoded_class_equals_target"]
        ]
        for step in CHECKPOINT_STEPS
    }
    if len(final_correct) == len(values):
        category = "all_upper_class_conditions_correct_by_step_2000"
    elif final_correct:
        category = "mixed_upper_class_progress_by_step_2000"
    else:
        category = "no_upper_class_condition_correct_by_step_2000"
    return {
        "interpretation_category": category,
        "final_correct_condition_ids": final_correct,
        "final_correct_condition_count": len(final_correct),
        "correct_condition_ids_by_checkpoint": correct_by_checkpoint,
        "all_upper_cuts_crossed_by_condition": {
            row["condition_id"]: all(row["upper_cuts_crossed_zero"].values())
            for row in values
        },
        "outcome_is_an_artifact_validity_gate": False,
    }


def _prepare_output(output_dir, repository_root, job_id):
    final = Path(output_dir).resolve()
    repository = Path(repository_root).resolve()
    if final.exists() or final.is_symlink():
        raise GraphEncoderError("grid_horizon_output_exists", str(final))
    if _is_within(final, repository):
        raise GraphEncoderError(
            "unsafe_grid_horizon_output", "artifact must be outside repository"
        )
    if not final.parent.is_dir() or final.parent.is_symlink():
        raise GraphEncoderError(
            "unsafe_grid_horizon_output", "artifact parent must already exist"
        )
    staging = final.with_name(final.name + ".incomplete-" + job_id)
    if staging.exists() or staging.is_symlink():
        raise GraphEncoderError("grid_horizon_staging_exists", str(staging))
    staging.mkdir()
    return final, staging


def run_grid_ordinal_horizon(*, output_dir, repository_root, expected_commit):
    _require_runtime()
    job_id = validate_slurm_job_id()
    source = _source_identity(repository_root, expected_commit)
    final, staging = _prepare_output(output_dir, repository_root, job_id)
    started = time.perf_counter()
    records = []
    summaries = []
    for condition in condition_matrix():
        condition_records = run_condition(
            condition["operation_type"], condition["target_class"]
        )
        records.extend(condition_records)
        summaries.append(summarize_condition(condition_records))
    interpretation = horizon_summary(summaries)
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
        "grid_parameterization": GRID_MAGNITUDE_PARAMETERIZATION,
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
        "horizon_evaluation": interpretation,
        "interpretation_boundary": {
            "actual_repository_head_and_loss_on_generated_fixed_states_only": True,
            "job_3352404_checkpoint_behavior_proven": False,
            "full_model_or_encoder_behavior_proven": False,
            "corpus_behavior_proven": False,
            "exact_defect_identified": False,
            "initialization_or_source_change_authorized": False,
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
        "horizon_evaluation": interpretation,
        "completion": {
            "event": "grid_ordinal_horizon_completed",
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
        raise GraphEncoderError("unsafe_grid_horizon_output", "parents differ")
    if final.exists() or final.is_symlink():
        raise GraphEncoderError("grid_horizon_output_exists", str(final))
    required = {"resolved_config.json", "trajectory.jsonl", "summary.json"}
    ordinary = _regular_artifact_files(
        staging, excluded=("artifact_manifest.json", "SHA256SUMS")
    )
    if set(ordinary) != required:
        raise GraphEncoderError(
            "incomplete_grid_horizon_artifact", "ordinary files differ"
        )
    manifest = {
        "schema_version": ARTIFACT_VERSION,
        "diagnostic_version": DIAGNOSTIC_VERSION,
        "trajectory_version": TRAJECTORY_VERSION,
        "artifacts": [{
            "path": relative,
            "byte_size": (staging / relative).stat().st_size,
            "sha256": _file_sha256(staging / relative),
        } for relative in ordinary],
    }
    _atomic_write_json(staging / "artifact_manifest.json", manifest)
    checksum_paths = _regular_artifact_files(staging, excluded=("SHA256SUMS",))
    _atomic_write_bytes(
        staging / "SHA256SUMS",
        "".join(
            "{}  {}\n".format(_file_sha256(staging / relative), relative)
            for relative in checksum_paths
        ).encode("utf-8"),
    )
    resolved = json.loads(
        (staging / "resolved_config.json").read_text(encoding="utf-8")
    )
    verify_artifact(
        staging,
        expected_commit=resolved["source"]["git_commit"],
        expected_slurm_job_id=resolved["runtime"]["slurm_job_id"],
        allow_incomplete_name=True,
    )
    os.replace(str(staging), str(final))
    return final


def _canonical_json_file(path):
    raw = Path(path).read_bytes()
    if not raw.endswith(b"\n"):
        raise GraphEncoderError("invalid_grid_horizon_artifact", "final LF absent")
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise GraphEncoderError(
            "invalid_grid_horizon_artifact", "JSON unreadable"
        ) from exc
    if raw.decode("utf-8") != _canonical_json_text(value) + "\n":
        raise GraphEncoderError(
            "invalid_grid_horizon_artifact", "JSON is not canonical"
        )
    return value


def verify_artifact(
    path, *, expected_commit, expected_slurm_job_id, allow_incomplete_name=False
):
    job_id = validate_slurm_job_id(expected_slurm_job_id)
    root = Path(path)
    if not root.is_dir() or root.is_symlink():
        raise GraphEncoderError("invalid_grid_horizon_artifact", "root differs")
    if ".incomplete-" in root.name and not allow_incomplete_name:
        raise GraphEncoderError("invalid_grid_horizon_artifact", "incomplete root")
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
            "invalid_grid_horizon_artifact", "exact five-file set differs"
        )
    manifest = _canonical_json_file(root / "artifact_manifest.json")
    resolved = _canonical_json_file(root / "resolved_config.json")
    summary = _canonical_json_file(root / "summary.json")
    if (
        manifest.get("schema_version") != ARTIFACT_VERSION
        or manifest.get("diagnostic_version") != DIAGNOSTIC_VERSION
        or manifest.get("trajectory_version") != TRAJECTORY_VERSION
    ):
        raise GraphEncoderError("invalid_grid_horizon_artifact", "identity differs")
    listed = []
    for row in manifest.get("artifacts", []):
        if set(row) != {"path", "byte_size", "sha256"}:
            raise GraphEncoderError("invalid_grid_horizon_artifact", "manifest row")
        relative = _safe_relative_path(row["path"])
        target = root / relative
        if (
            not target.is_file() or target.is_symlink()
            or target.stat().st_size != row["byte_size"]
            or _file_sha256(target) != row["sha256"]
        ):
            raise GraphEncoderError(
                "grid_horizon_artifact_integrity_failure", relative
            )
        listed.append(relative)
    ordinary = _regular_artifact_files(
        root, excluded=("artifact_manifest.json", "SHA256SUMS")
    )
    if tuple(listed) != ordinary:
        raise GraphEncoderError("invalid_grid_horizon_artifact", "coverage differs")
    checksum_raw = (root / "SHA256SUMS").read_bytes()
    if not checksum_raw.endswith(b"\n"):
        raise GraphEncoderError("invalid_grid_horizon_artifact", "checksum LF")
    checksum_paths = []
    for line in checksum_raw.decode("utf-8").splitlines():
        parts = line.split("  ", 1)
        if len(parts) != 2 or len(parts[0]) != 64:
            raise GraphEncoderError("invalid_grid_horizon_artifact", "checksum row")
        relative = _safe_relative_path(parts[1])
        if _file_sha256(root / relative) != parts[0]:
            raise GraphEncoderError(
                "grid_horizon_artifact_integrity_failure", relative
            )
        checksum_paths.append(relative)
    expected_sums = _regular_artifact_files(root, excluded=("SHA256SUMS",))
    if tuple(checksum_paths) != expected_sums:
        raise GraphEncoderError(
            "invalid_grid_horizon_artifact", "checksum coverage differs"
        )
    trajectory = _read_canonical_jsonl(root / "trajectory.jsonl")
    conditions = condition_matrix()
    if len(trajectory) != EXPECTED_TRAJECTORY_RECORD_COUNT:
        raise GraphEncoderError("incomplete_grid_horizon", "record count differs")
    recomputed = []
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
            raise GraphEncoderError("incomplete_grid_horizon", "condition differs")
        for row in rows:
            _validate_loaded_record(row, condition)
        recomputed.append(summarize_condition(rows))
    runtime = resolved.get("runtime", {})
    source = resolved.get("source", {})
    if (
        resolved.get("diagnostic_version") != DIAGNOSTIC_VERSION
        or resolved.get("trajectory_version") != TRAJECTORY_VERSION
        or resolved.get("artifact_version") != ARTIFACT_VERSION
        or resolved.get("arithmetic") != diagnostic_arithmetic()
        or resolved.get("condition_matrix") != list(conditions)
        or source.get("git_commit") != expected_commit
        or source.get("git_branch") is not None
        or source.get("detached_head") is not True
        or source.get("git_dirty") is not False
        or source.get("git_status_porcelain") != []
        or runtime.get("slurm_job_id") != job_id
        or runtime.get("python") != "3.8.13"
        or str(runtime.get("pytorch", "")).split("+")[0] != "1.11.0"
        or runtime.get("device") != "cpu"
        or runtime.get("cuda_available") is not False
        or runtime.get("cpu_threads") != 1
    ):
        raise GraphEncoderError("invalid_grid_horizon_artifact", "resolved differs")
    if (
        summary.get("source_commit") != expected_commit
        or summary.get("slurm_job_id") != job_id
        or summary.get("condition_count") != EXPECTED_CONDITION_COUNT
        or summary.get("trajectory_record_count")
        != EXPECTED_TRAJECTORY_RECORD_COUNT
        or summary.get("diagnostic_completed") is not True
        or summary.get("conditions") != recomputed
        or summary.get("horizon_evaluation") != horizon_summary(recomputed)
    ):
        raise GraphEncoderError("invalid_grid_horizon_artifact", "summary differs")
    for record in (resolved.get("access", {}), summary):
        if any(record.get(key) != value for key, value in ACCESS_RECORD.items()):
            raise GraphEncoderError(
                "invalid_grid_horizon_artifact", "authority boundary differs"
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
        "interpretation_category": summary["horizon_evaluation"][
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
        result = run_grid_ordinal_horizon(
            output_dir=arguments.output_dir,
            repository_root=arguments.repository_root,
            expected_commit=arguments.expected_commit,
        )
    except Exception as exc:
        failure = {
            "event": "grid_ordinal_horizon_infrastructure_failure",
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
