"""Deterministic, common C6 training and strict recovery checkpoints."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import time

try:
    import numpy as np
except ImportError:  # NumPy state is recorded only when NumPy is available.
    np = None

try:
    import torch
except ImportError:  # Static C6 contracts remain importable without PyTorch.
    torch = None

from .batching import build_paired_batch
from .config import (
    CHECKPOINT_EPOCH,
    CHECKPOINT_SCHEMA,
    GE1TrainingConfig,
    MODEL_FAMILY,
    PLATEAU_MOVING_BEST_WINDOW_EPOCHS,
    PLATEAU_RELATIVE_IMPROVEMENT_THRESHOLD,
    PLATEAU_START_EPOCH,
    TRAINING_BATCH_SIZE,
    TRAINING_EPOCHS,
    TRAINING_GRADIENT_CLIP_NORM,
    TRAINING_LEARNING_RATE,
    TRAINING_OPTIMIZER,
    TRAINING_WEIGHT_DECAY,
)
from .errors import GraphEncoderError
from .provenance import (
    authorize_c6_provenance,
    configuration_digest,
    sha256_json,
    training_partition_identity,
    verify_c6_provenance,
)


C6_TRAINING_CONTRACT_VERSION = "GE1-C6-TRAINING-v1"
TRAINING_STATE_SCHEMA = "GE1-TRAINING-STATE-v1"
PLATEAU_EPSILON = 1e-12
SMOKE_EPOCHS = 2
FULL_TRAIN_STEPS_PER_EPOCH = 51
FULL_TRAIN_STEPS_PER_RUN = 2550
FULL_TRAINING_EXAMPLE_PRESENTATIONS = 20350

TRAINING_CHECKPOINT_FIELDS = frozenset({
    "training_state_schema",
    "checkpoint_schema",
    "model_family",
    "arm_identity",
    "encoder_type",
    "model_config",
    "model_config_sha256",
    "training_config",
    "training_config_sha256",
    "model_state",
    "optimizer_name",
    "optimizer_state",
    "completed_epoch",
    "optimizer_step_count",
    "training_example_presentations",
    "seed",
    "rng_states",
    "plateau_state",
    "epoch_records",
    "provenance",
    "source_digest",
    "partition_identity",
    "partition_identity_sha256",
    "run_identity",
    "selected_checkpoint_epoch",
    "selected_experimental_checkpoint",
    "family_order_history",
})


class C6TrainingError(GraphEncoderError):
    """Terminal training failure with a target-free structured record."""

    def __init__(self, code, detail, failure_record):
        self.failure_record = dict(failure_record)
        super().__init__(code, detail)


@dataclass(frozen=True)
class PlateauEvaluation:
    epoch: int
    recent_best: float
    previous_best: float
    relative_improvement: float
    plateau: bool


@dataclass(frozen=True)
class PlateauState:
    epsilon: float
    threshold: float
    window_epochs: int
    first_eligible_epoch: int
    first_plateau_epoch: object
    epoch_loss_history: tuple
    evaluations: tuple

    def to_dict(self):
        return {
            "epsilon": self.epsilon,
            "threshold": self.threshold,
            "window_epochs": self.window_epochs,
            "first_eligible_epoch": self.first_eligible_epoch,
            "first_plateau_epoch": self.first_plateau_epoch,
            "epoch_loss_history": list(self.epoch_loss_history),
            "evaluations": [asdict(item) for item in self.evaluations],
        }


@dataclass(frozen=True)
class EpochTrainingRecord:
    epoch: int
    mean_loss: float
    mean_loss_components: tuple
    optimizer_steps: int
    training_example_presentations: int
    family_order: tuple
    batch_boundaries: tuple
    pre_clip_gradient_norms: tuple
    data_preparation_seconds: float
    training_seconds: float

    def to_dict(self):
        values = asdict(self)
        values["mean_loss_components"] = dict(self.mean_loss_components)
        values["family_order"] = list(self.family_order)
        values["batch_boundaries"] = [list(item) for item in self.batch_boundaries]
        values["pre_clip_gradient_norms"] = list(self.pre_clip_gradient_norms)
        return values


@dataclass(frozen=True)
class TrainingRunResult:
    version: str
    run_identity: str
    arm: str
    seed: int
    model_configuration: tuple
    training_configuration: tuple
    completed_epoch: int
    optimizer_steps: int
    training_example_presentations: int
    epoch_records: tuple
    plateau_state: PlateauState
    checkpoint_paths: tuple
    checkpoint_opportunities: int
    selected_checkpoint_epoch: int
    selected_experimental_checkpoint: object
    provenance: object
    timing: tuple

    def to_dict(self):
        return {
            "version": self.version,
            "run_identity": self.run_identity,
            "arm": self.arm,
            "seed": self.seed,
            "model_configuration": dict(self.model_configuration),
            "training_configuration": dict(self.training_configuration),
            "completed_epoch": self.completed_epoch,
            "optimizer_steps": self.optimizer_steps,
            "training_example_presentations": self.training_example_presentations,
            "epoch_records": [item.to_dict() for item in self.epoch_records],
            "plateau_state": self.plateau_state.to_dict(),
            "checkpoint_paths": list(self.checkpoint_paths),
            "checkpoint_opportunities": self.checkpoint_opportunities,
            "selected_checkpoint_epoch": self.selected_checkpoint_epoch,
            "selected_experimental_checkpoint": self.selected_experimental_checkpoint,
            "provenance": self.provenance.to_dict(),
            "timing": dict(self.timing),
        }


@dataclass(frozen=True)
class TrainingResumeState:
    completed_epoch: int
    optimizer_step_count: int
    training_example_presentations: int
    epoch_records: tuple
    plateau_state: PlateauState
    family_order_history: tuple
    data_order_random_state: object
    payload: dict


def training_contract_metadata():
    """Return the exact frozen training and diagnostic selection contract."""

    return {
        "version": C6_TRAINING_CONTRACT_VERSION,
        "epochs": TRAINING_EPOCHS,
        "batch_size": TRAINING_BATCH_SIZE,
        "optimizer": TRAINING_OPTIMIZER,
        "learning_rate": TRAINING_LEARNING_RATE,
        "weight_decay": TRAINING_WEIGHT_DECAY,
        "gradient_clip_global_norm": TRAINING_GRADIENT_CLIP_NORM,
        "seeds": [2026, 2027, 2028],
        "checkpoint_opportunity": "after_each_successfully_completed_epoch",
        "selected_experimental_checkpoint": "fixed_epoch_50",
        "early_stopping": False,
        "development_checkpoint_selection": False,
        "training_loss_checkpoint_selection": False,
        "best_checkpoint_selection": False,
        "earlier_checkpoint_roles": ["recovery", "provenance", "debugging", "predetermined_diagnostic"],
        "plateau_diagnostic_only": True,
        "training_example_presentations_definition": (
            "cumulative actual physical examples in successfully completed "
            "optimizer steps; excludes evaluation, padding, interventions, "
            "and failed steps"
        ),
    }


def plateau_state(epoch_losses, first_plateau_epoch=None):
    """Apply the frozen two-window diagnostic plateau definition exactly."""

    losses = tuple(float(value) for value in epoch_losses)
    if any(not math.isfinite(value) for value in losses):
        raise GraphEncoderError("nonfinite_metric", "epoch loss history is nonfinite")
    evaluations = []
    frozen_first = first_plateau_epoch
    window = PLATEAU_MOVING_BEST_WINDOW_EPOCHS
    for epoch in range(PLATEAU_START_EPOCH, len(losses) + 1):
        recent = min(losses[epoch - window:epoch])
        previous = min(losses[epoch - 2 * window:epoch - window])
        improvement = (previous - recent) / max(abs(previous), PLATEAU_EPSILON)
        detected = improvement < PLATEAU_RELATIVE_IMPROVEMENT_THRESHOLD
        evaluations.append(PlateauEvaluation(
            epoch, recent, previous, improvement, detected
        ))
        if detected and frozen_first is None:
            frozen_first = epoch
    return PlateauState(
        PLATEAU_EPSILON,
        PLATEAU_RELATIVE_IMPROVEMENT_THRESHOLD,
        window,
        PLATEAU_START_EPOCH,
        frozen_first,
        losses,
        tuple(evaluations),
    )


def deterministic_family_order(family_ids, seed, epoch):
    """Pure diagnostic order; identical for both arms for a seed and epoch."""

    ordered = tuple(sorted(family_ids))
    if len(ordered) != len(set(ordered)):
        raise GraphEncoderError("duplicate_family_id", "family IDs must be unique")
    material = "{}:{}".format(int(seed), int(epoch)).encode("utf-8")
    local_seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
    generator = random.Random(local_seed)
    values = list(ordered)
    generator.shuffle(values)
    return tuple(values)


def run_ge1_training(
    model,
    examples,
    *,
    training_config,
    checkpoint_directory,
    repository_root,
    expected_commit,
    final_epoch=TRAINING_EPOCHS,
    engineering_smoke=False,
    resume_checkpoint=None,
    provenance_context=None,
    provenance_verifier=verify_c6_provenance,
    checkpoint_epochs=None,
    extended_final_epoch=None,
    checkpoint_coordinate="epoch",
    selected_checkpoint_epoch=CHECKPOINT_EPOCH,
):
    """Run one common training loop for either arm.

    ``final_epoch`` may be 1 or 2 only for an explicitly labeled engineering
    smoke.  The additive post-C7 optimization diagnostic may instead pass an
    equal ``extended_final_epoch`` and an explicit sparse checkpoint schedule.
    Defaults preserve the C6/formal-C7 every-epoch behavior and fixed epoch-50
    selection. The diagnostic explicitly supplies no selected checkpoint.
    """

    _require_torch()
    # Imported only for real training so the frozen plateau/checkpoint metadata
    # remains inspectable on documentation hosts without PyTorch installed.
    from .losses import common_ge1_loss
    if not isinstance(training_config, GE1TrainingConfig):
        raise TypeError("training_config must be GE1TrainingConfig")
    training_config.validate()
    model.config.validate()
    _validate_execution_epoch(
        final_epoch, engineering_smoke, extended_final_epoch=extended_final_epoch
    )
    checkpoint_schedule = _validate_checkpoint_schedule(
        final_epoch, checkpoint_epochs, checkpoint_coordinate
    )
    ordered_examples = _validate_training_examples(examples)
    family_ids = tuple(item.physical_family_id for item in ordered_examples)
    partition_identity = training_partition_identity(family_ids)
    partition_digest = sha256_json(partition_identity)
    config_digest = configuration_digest(
        model.config, training_config, partition_identity
    )
    output_dir = Path(checkpoint_directory).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _validate_output_location(repository_root, output_dir)
    run_identity = "ge1-{}-seed{}".format(model.config.encoder, model.config.seed)
    if provenance_context is None:
        provenance_context = authorize_c6_provenance(
            repository_root,
            expected_commit=expected_commit,
            configuration_sha256=config_digest,
            partition_identity_sha256=partition_digest,
            seed=model.config.seed,
            encoder_arm=model.config.encoder,
            device="cpu",
            torch_module=torch,
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    data_order_random = random.Random(model.config.seed + 7919)
    completed_epoch = 0
    optimizer_steps = 0
    presentations = 0
    epoch_records = []
    family_order_history = []
    plateau = plateau_state(())
    if resume_checkpoint is None:
        _seed_run(model.config.seed)
    else:
        resumed = load_training_checkpoint(
            resume_checkpoint,
            model=model,
            optimizer=optimizer,
            training_config=training_config,
            partition_identity=partition_identity,
            expected_code_revision=expected_commit,
            restore_rng=True,
        )
        completed_epoch = resumed.completed_epoch
        optimizer_steps = resumed.optimizer_step_count
        presentations = resumed.training_example_presentations
        epoch_records = list(resumed.epoch_records)
        plateau = resumed.plateau_state
        family_order_history = list(resumed.family_order_history)
        data_order_random.setstate(resumed.data_order_random_state)
    if completed_epoch >= final_epoch:
        raise GraphEncoderError(
            "invalid_training_resume",
            "checkpoint epoch must precede the requested final epoch",
        )

    by_id = {item.physical_family_id: item for item in ordered_examples}
    checkpoint_paths = []
    data_seconds_total = 0.0
    training_seconds_total = 0.0
    checkpoint_seconds_total = 0.0
    run_start = time.perf_counter()
    for epoch in range(completed_epoch + 1, final_epoch + 1):
        epoch_order = list(sorted(family_ids))
        data_order_random.shuffle(epoch_order)
        epoch_order = tuple(epoch_order)
        family_order_history.append(epoch_order)
        component_sums = {}
        epoch_examples = 0
        epoch_steps = 0
        gradient_norms = []
        boundaries = []
        epoch_data_seconds = 0.0
        epoch_training_seconds = 0.0
        model.train()
        for batch_index, start in enumerate(
            range(0, len(epoch_order), training_config.batch_size)
        ):
            batch_ids = epoch_order[start:start + training_config.batch_size]
            boundaries.append(batch_ids)
            data_start = time.perf_counter()
            paired = build_paired_batch(tuple(by_id[item] for item in batch_ids))
            tensors = _training_tensors(paired, model.config.encoder)
            epoch_data_seconds += time.perf_counter() - data_start
            train_start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            try:
                teacher = model.teacher_forced(
                    tensors["encoder_input"],
                    tensors["target"],
                    tensors["profile_targets"],
                )
                _assert_finite_tree(
                    teacher.decoder_output,
                    run_identity=run_identity,
                    arm=model.config.encoder,
                    seed=model.config.seed,
                    epoch=epoch,
                    batch_index=batch_index,
                    family_ids=batch_ids,
                    failed_stage="model_output",
                )
                loss = common_ge1_loss(
                    teacher.decoder_output,
                    tensors["target"],
                    tensors["profile_targets"],
                    model.config,
                )
            except C6TrainingError:
                raise
            except Exception as exc:
                _terminal_failure(
                    "training_step_failure",
                    str(exc),
                    run_identity,
                    model.config.encoder,
                    model.config.seed,
                    epoch,
                    batch_index,
                    batch_ids,
                    "forward_or_loss",
                    type(exc).__name__,
                    False,
                )
            _assert_finite_loss(
                loss, run_identity, model.config.encoder, model.config.seed,
                epoch, batch_index, batch_ids
            )
            loss.total.backward()
            _assert_finite_gradients(
                model, run_identity, model.config.encoder, model.config.seed,
                epoch, batch_index, batch_ids, "pre_clip_gradient"
            )
            pre_clip = torch.nn.utils.clip_grad_norm_(
                model.parameters(), training_config.gradient_clip_norm
            )
            if not bool(torch.isfinite(torch.as_tensor(pre_clip)).item()):
                _terminal_failure(
                    "nonfinite_gradient", "pre-clip global norm is nonfinite",
                    run_identity, model.config.encoder, model.config.seed,
                    epoch, batch_index, batch_ids, "gradient_clipping",
                    "global_norm", False,
                )
            _assert_finite_gradients(
                model, run_identity, model.config.encoder, model.config.seed,
                epoch, batch_index, batch_ids, "post_clip_gradient"
            )
            optimizer.step()
            _assert_finite_optimizer(
                optimizer, run_identity, model.config.encoder, model.config.seed,
                epoch, batch_index, batch_ids
            )
            batch_size = len(batch_ids)
            optimizer_steps += 1
            epoch_steps += 1
            presentations += batch_size
            epoch_examples += batch_size
            gradient_norms.append(float(pre_clip.item()))
            for name, values in loss.per_example.items():
                component_sums[name] = component_sums.get(name, 0.0) + float(
                    values.detach().sum().item()
                )
            epoch_training_seconds += time.perf_counter() - train_start

        if epoch_examples != len(family_ids):
            raise AssertionError("one epoch must present every physical family once")
        means = tuple(sorted(
            (name, value / float(epoch_examples))
            for name, value in component_sums.items()
        ))
        mean_loss = dict(means)["total"]
        epoch_record = EpochTrainingRecord(
            epoch,
            mean_loss,
            means,
            optimizer_steps,
            presentations,
            epoch_order,
            tuple(boundaries),
            tuple(gradient_norms),
            epoch_data_seconds,
            epoch_training_seconds,
        )
        epoch_records.append(epoch_record)
        plateau = plateau_state(
            tuple(item.mean_loss for item in epoch_records),
            first_plateau_epoch=plateau.first_plateau_epoch,
        )
        data_seconds_total += epoch_data_seconds
        training_seconds_total += epoch_training_seconds
        if epoch not in checkpoint_schedule:
            continue
        checkpoint_path = output_dir / (
            "{}-seed{}-{}{:04d}.pt".format(
                model.config.encoder,
                model.config.seed,
                checkpoint_coordinate,
                epoch,
            )
        )
        checkpoint_start = time.perf_counter()
        save_training_checkpoint(
            checkpoint_path,
            model=model,
            optimizer=optimizer,
            training_config=training_config,
            completed_epoch=epoch,
            optimizer_step_count=optimizer_steps,
            training_example_presentations=presentations,
            plateau=plateau,
            epoch_records=tuple(epoch_records),
            family_order_history=tuple(family_order_history),
            data_order_random_state=data_order_random.getstate(),
            provenance_context=provenance_context,
            provenance_verifier=provenance_verifier,
            partition_identity=partition_identity,
            expected_code_revision=expected_commit,
            selected_checkpoint_epoch=selected_checkpoint_epoch,
        )
        checkpoint_seconds_total += time.perf_counter() - checkpoint_start
        checkpoint_paths.append(str(checkpoint_path))

    selected = (
        checkpoint_paths[-1]
        if selected_checkpoint_epoch is not None
        and final_epoch == selected_checkpoint_epoch
        else None
    )
    timing = (
        ("data_preparation_seconds", data_seconds_total),
        ("training_seconds", training_seconds_total),
        ("checkpoint_and_provenance_seconds", checkpoint_seconds_total),
        ("total_training_run_seconds", time.perf_counter() - run_start),
        ("clock", "time.perf_counter_monotonic"),
    )
    return TrainingRunResult(
        C6_TRAINING_CONTRACT_VERSION,
        run_identity,
        model.config.encoder,
        model.config.seed,
        tuple(sorted(model.config.to_dict().items())),
        tuple(sorted(training_config.to_dict().items())),
        final_epoch,
        optimizer_steps,
        presentations,
        tuple(epoch_records),
        plateau,
        tuple(checkpoint_paths),
        final_epoch,
        selected_checkpoint_epoch,
        selected,
        provenance_context.authorized,
        timing,
    )


def training_checkpoint_payload(
    *,
    model,
    optimizer,
    training_config,
    completed_epoch,
    optimizer_step_count,
    training_example_presentations,
    plateau,
    epoch_records,
    family_order_history,
    data_order_random_state,
    provenance,
    partition_identity,
    expected_code_revision,
    selected_checkpoint_epoch=CHECKPOINT_EPOCH,
):
    """Build the exact strict C6 recovery payload without target content."""

    model_config = model.config.to_dict()
    training_values = training_config.to_dict()
    return {
        "training_state_schema": TRAINING_STATE_SCHEMA,
        "checkpoint_schema": CHECKPOINT_SCHEMA,
        "model_family": MODEL_FAMILY,
        "arm_identity": model.config.arm_identity,
        "encoder_type": model.config.encoder,
        "model_config": model_config,
        "model_config_sha256": sha256_json(model_config),
        "training_config": training_values,
        "training_config_sha256": sha256_json(training_values),
        "model_state": model.state_dict(),
        "optimizer_name": TRAINING_OPTIMIZER,
        "optimizer_state": optimizer.state_dict(),
        "completed_epoch": int(completed_epoch),
        "optimizer_step_count": int(optimizer_step_count),
        "training_example_presentations": int(training_example_presentations),
        "seed": int(model.config.seed),
        "rng_states": _capture_rng_states(data_order_random_state),
        "plateau_state": plateau.to_dict(),
        "epoch_records": [item.to_dict() for item in epoch_records],
        "provenance": provenance.to_dict(),
        "source_digest": provenance.source_tree_sha256,
        "partition_identity": partition_identity,
        "partition_identity_sha256": sha256_json(partition_identity),
        "run_identity": "ge1-{}-seed{}".format(
            model.config.encoder, model.config.seed
        ),
        "selected_checkpoint_epoch": selected_checkpoint_epoch,
        "selected_experimental_checkpoint": (
            selected_checkpoint_epoch is not None
            and completed_epoch == selected_checkpoint_epoch
        ),
        "family_order_history": [list(item) for item in family_order_history],
    }


def save_training_checkpoint(
    path,
    *,
    model,
    optimizer,
    training_config,
    completed_epoch,
    optimizer_step_count,
    training_example_presentations,
    plateau,
    epoch_records,
    family_order_history,
    data_order_random_state,
    provenance_context,
    provenance_verifier,
    partition_identity,
    expected_code_revision,
    selected_checkpoint_epoch=CHECKPOINT_EPOCH,
):
    """Recheck provenance, then atomically publish one complete checkpoint."""

    _require_torch()
    current_config_digest = configuration_digest(
        model.config, training_config, partition_identity
    )
    current_partition_digest = sha256_json(partition_identity)
    try:
        provenance = provenance_verifier(
            provenance_context,
            torch_module=torch,
            configuration_sha256=current_config_digest,
            partition_identity_sha256=current_partition_digest,
        )
    except TypeError:
        # Narrow test doubles may expose the historical one-argument shape.
        provenance = provenance_verifier(provenance_context, torch_module=torch)
    if provenance.git_commit != expected_code_revision:
        raise GraphEncoderError("wrong_git_commit", "checkpoint commit differs")
    payload = training_checkpoint_payload(
        model=model,
        optimizer=optimizer,
        training_config=training_config,
        completed_epoch=completed_epoch,
        optimizer_step_count=optimizer_step_count,
        training_example_presentations=training_example_presentations,
        plateau=plateau,
        epoch_records=epoch_records,
        family_order_history=family_order_history,
        data_order_random_state=data_order_random_state,
        provenance=provenance,
        partition_identity=partition_identity,
        expected_code_revision=expected_code_revision,
        selected_checkpoint_epoch=selected_checkpoint_epoch,
    )
    final_path = Path(path)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = final_path.with_name(
        ".{}.tmp-{}".format(final_path.name, os.getpid())
    )
    if temporary.exists():
        temporary.unlink()
    try:
        with temporary.open("wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(str(temporary), str(final_path))
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
    return payload


def load_training_checkpoint(
    path,
    *,
    model,
    optimizer,
    training_config,
    partition_identity,
    expected_code_revision,
    restore_rng,
    expected_selected_checkpoint_epoch=CHECKPOINT_EPOCH,
):
    """Strictly reload model, optimizer, counters, plateau, and RNG state."""

    _require_torch()
    payload = torch.load(str(path), map_location="cpu")
    if not isinstance(payload, dict):
        raise GraphEncoderError("invalid_training_checkpoint", "payload must be a dict")
    missing = TRAINING_CHECKPOINT_FIELDS - set(payload)
    unexpected = set(payload) - TRAINING_CHECKPOINT_FIELDS
    if missing or unexpected:
        raise GraphEncoderError(
            "invalid_training_checkpoint",
            "field mismatch missing={} unexpected={}".format(
                sorted(missing), sorted(unexpected)
            ),
        )
    expected_metadata = {
        "training_state_schema": TRAINING_STATE_SCHEMA,
        "checkpoint_schema": CHECKPOINT_SCHEMA,
        "model_family": MODEL_FAMILY,
        "arm_identity": model.config.arm_identity,
        "encoder_type": model.config.encoder,
        "model_config": model.config.to_dict(),
        "model_config_sha256": sha256_json(model.config.to_dict()),
        "training_config": training_config.to_dict(),
        "training_config_sha256": sha256_json(training_config.to_dict()),
        "optimizer_name": TRAINING_OPTIMIZER,
        "seed": model.config.seed,
        "partition_identity": partition_identity,
        "partition_identity_sha256": sha256_json(partition_identity),
        "run_identity": "ge1-{}-seed{}".format(
            model.config.encoder, model.config.seed
        ),
        "selected_checkpoint_epoch": expected_selected_checkpoint_epoch,
    }
    for name, expected in expected_metadata.items():
        if payload[name] != expected:
            raise GraphEncoderError(
                "invalid_training_checkpoint", "metadata field {} differs".format(name)
            )
    provenance = payload["provenance"]
    if not isinstance(provenance, dict) or provenance.get("git_commit") != expected_code_revision:
        raise GraphEncoderError(
            "invalid_training_checkpoint", "checkpoint provenance commit differs"
        )
    if payload["source_digest"] != provenance.get("source_tree_sha256"):
        raise GraphEncoderError(
            "invalid_training_checkpoint", "source digest differs from provenance"
        )
    _strict_state_load(model, payload["model_state"])
    try:
        optimizer.load_state_dict(payload["optimizer_state"])
    except Exception as exc:
        raise GraphEncoderError(
            "invalid_training_checkpoint", "strict optimizer reload failed"
        ) from exc
    _assert_optimizer_state_finite_plain(optimizer)
    epoch_records = tuple(_epoch_record_from_dict(item) for item in payload["epoch_records"])
    plateau = _plateau_from_dict(payload["plateau_state"])
    if payload["completed_epoch"] != len(epoch_records):
        raise GraphEncoderError(
            "invalid_training_checkpoint", "completed epoch and history length differ"
        )
    expected_selected = (
        expected_selected_checkpoint_epoch is not None
        and payload["completed_epoch"] == expected_selected_checkpoint_epoch
    )
    if payload["selected_experimental_checkpoint"] is not expected_selected:
        raise GraphEncoderError(
            "invalid_training_checkpoint", "fixed checkpoint selection flag differs"
        )
    if restore_rng:
        data_order_state = _restore_rng_states(payload["rng_states"])
    else:
        data_order_state = payload["rng_states"]["data_order_random"]
    return TrainingResumeState(
        payload["completed_epoch"],
        payload["optimizer_step_count"],
        payload["training_example_presentations"],
        epoch_records,
        plateau,
        tuple(tuple(item) for item in payload["family_order_history"]),
        data_order_state,
        payload,
    )


def _training_tensors(paired, encoder):
    from prototype.constrained_profile_decoder import profile_targets_for_loss

    flat = paired.flat_input.to_torch(torch)
    target = paired.target.to_torch(torch)
    profiles = profile_targets_for_loss(paired.target, flat["geometry"])
    if encoder == "flat":
        encoder_input = flat
    elif encoder == "typed_graph":
        encoder_input = paired.graph_input.to_torch(torch)
        bookkeeping = paired.graph_bookkeeping.to_torch(torch)
        encoder_input["graph_offsets"] = bookkeeping["graph_offsets"]
    else:
        raise GraphEncoderError("invalid_configuration", "unknown encoder arm")
    return {
        "encoder_input": encoder_input,
        "target": target,
        "profile_targets": profiles,
    }


def _validate_execution_epoch(
    final_epoch, engineering_smoke, *, extended_final_epoch=None
):
    if isinstance(final_epoch, bool) or not isinstance(final_epoch, int):
        raise GraphEncoderError("invalid_training_budget", "final epoch must be integer")
    if extended_final_epoch is not None:
        if (
            isinstance(extended_final_epoch, bool)
            or not isinstance(extended_final_epoch, int)
            or extended_final_epoch <= TRAINING_EPOCHS
            or final_epoch != extended_final_epoch
            or engineering_smoke
        ):
            raise GraphEncoderError(
                "invalid_training_budget",
                "extended final epoch must be an equal integer above epoch 50",
            )
        return
    allowed = (1, SMOKE_EPOCHS) if engineering_smoke else (TRAINING_EPOCHS,)
    if final_epoch not in allowed:
        raise GraphEncoderError(
            "invalid_training_budget",
            "only epoch 1/2 engineering smoke or the frozen epoch 50 run is allowed",
        )


def _validate_checkpoint_schedule(
    final_epoch, checkpoint_epochs, checkpoint_coordinate
):
    if checkpoint_coordinate not in ("epoch", "update"):
        raise GraphEncoderError(
            "invalid_checkpoint_schedule",
            "checkpoint coordinate must be epoch or update",
        )
    if checkpoint_epochs is None:
        return frozenset(range(1, final_epoch + 1))
    values = tuple(checkpoint_epochs)
    if (
        not values
        or any(isinstance(item, bool) or not isinstance(item, int) for item in values)
        or values != tuple(sorted(set(values)))
        or values[0] < 1
        or values[-1] != final_epoch
    ):
        raise GraphEncoderError(
            "invalid_checkpoint_schedule",
            "checkpoint epochs must be sorted, unique, positive, and end at final epoch",
        )
    return frozenset(values)


def _validate_training_examples(examples):
    values = tuple(examples)
    if not values:
        raise GraphEncoderError("empty_training_set", "training examples are empty")
    ordered = tuple(sorted(values, key=lambda item: item.physical_family_id))
    ids = tuple(item.physical_family_id for item in ordered)
    if len(ids) != len(set(ids)):
        raise GraphEncoderError("duplicate_family_id", "training families must be unique")
    if any(item.split_name != "operation_template" or item.partition != "train" for item in ordered):
        raise GraphEncoderError(
            "protected_partition_access",
            "C6 training accepts only operation_template.train physical examples",
        )
    return ordered


def _validate_output_location(repository_root, output_dir):
    root = Path(repository_root).resolve()
    output = Path(output_dir).resolve()
    try:
        output.relative_to(root)
    except ValueError:
        return
    probe = output / ".ge1-c6-output-probe"
    result = subprocess.run(
        ("git", "check-ignore", "--quiet", "--no-index", str(probe)),
        cwd=str(root),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise GraphEncoderError(
            "ungoverned_output_location",
            "checkpoint output inside the repository must be Git-ignored",
        )


def _assert_finite_loss(loss, run_identity, arm, seed, epoch, batch_index, family_ids):
    fields = loss.as_dict()
    for name, value in fields.items():
        if not bool(torch.isfinite(value).all().item()):
            _terminal_failure(
                "nonfinite_loss", "loss component is nonfinite", run_identity,
                arm, seed, epoch, batch_index, family_ids, "loss", name, False
            )
    for name, values in loss.per_example.items():
        if not bool(torch.isfinite(values).all().item()):
            _terminal_failure(
                "nonfinite_loss_component", "per-example loss is nonfinite",
                run_identity, arm, seed, epoch, batch_index, family_ids,
                "loss_component", name, False
            )


def _assert_finite_tree(value, **failure):
    for name, tensor in _named_tensors(value):
        if tensor.dtype.is_floating_point and not bool(torch.isfinite(tensor).all().item()):
            _terminal_failure(
                "nonfinite_model_output", "model output is nonfinite",
                failure["run_identity"], failure["arm"], failure["seed"],
                failure["epoch"], failure["batch_index"], failure["family_ids"],
                failure["failed_stage"], name, False
            )


def _named_tensors(value, prefix="output"):
    if torch.is_tensor(value):
        return ((prefix, value),)
    if isinstance(value, dict):
        result = []
        for name, item in value.items():
            result.extend(_named_tensors(item, prefix + "." + str(name)))
        return tuple(result)
    if isinstance(value, (tuple, list)):
        result = []
        for index, item in enumerate(value):
            result.extend(_named_tensors(item, prefix + "[{}]".format(index)))
        return tuple(result)
    if hasattr(value, "__dataclass_fields__"):
        result = []
        for name in value.__dataclass_fields__:
            result.extend(_named_tensors(getattr(value, name), prefix + "." + name))
        return tuple(result)
    return ()


def _assert_finite_gradients(model, run_identity, arm, seed, epoch, batch_index, family_ids, stage):
    for name, parameter in model.named_parameters():
        gradient = parameter.grad
        if gradient is not None and not bool(torch.isfinite(gradient).all().item()):
            _terminal_failure(
                "nonfinite_gradient", "parameter gradient is nonfinite",
                run_identity, arm, seed, epoch, batch_index, family_ids,
                stage, name, False
            )


def _assert_finite_optimizer(optimizer, run_identity, arm, seed, epoch, batch_index, family_ids):
    for parameter_index, state in enumerate(optimizer.state.values()):
        for name, value in state.items():
            if torch.is_tensor(value) and value.dtype.is_floating_point and not bool(torch.isfinite(value).all().item()):
                _terminal_failure(
                    "nonfinite_optimizer_state", "optimizer state is nonfinite",
                    run_identity, arm, seed, epoch, batch_index, family_ids,
                    "optimizer_step", "{}.{}".format(parameter_index, name), False
                )


def _assert_optimizer_state_finite_plain(optimizer):
    for state in optimizer.state.values():
        for value in state.values():
            if torch.is_tensor(value) and value.dtype.is_floating_point and not bool(torch.isfinite(value).all().item()):
                raise GraphEncoderError(
                    "invalid_training_checkpoint", "optimizer state is nonfinite"
                )


def _terminal_failure(code, detail, run_identity, arm, seed, epoch, batch_index, family_ids, failed_stage, name, finite):
    record = {
        "run_identity": run_identity,
        "arm": arm,
        "seed": seed,
        "epoch": epoch,
        "batch_index": batch_index,
        "family_ids": list(family_ids),
        "failed_stage": failed_stage,
        "failed_tensor_or_component": name,
        "finite": bool(finite),
        "target_payload_serialized": False,
    }
    raise C6TrainingError(code, detail, record)


def _capture_rng_states(data_order_random_state):
    states = {
        "python_global": random.getstate(),
        "numpy": None if np is None else np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": None,
        "data_order_random": data_order_random_state,
    }
    if torch.cuda.is_available():
        states["torch_cuda"] = torch.cuda.get_rng_state_all()
    return states


def _restore_rng_states(states):
    required = {"python_global", "numpy", "torch_cpu", "torch_cuda", "data_order_random"}
    if not isinstance(states, dict) or set(states) != required:
        raise GraphEncoderError("invalid_training_checkpoint", "RNG state fields differ")
    random.setstate(states["python_global"])
    if states["numpy"] is not None:
        if np is None:
            raise GraphEncoderError("invalid_training_checkpoint", "NumPy RNG cannot be restored")
        np.random.set_state(states["numpy"])
    torch.set_rng_state(states["torch_cpu"])
    if states["torch_cuda"] is not None:
        if not torch.cuda.is_available():
            raise GraphEncoderError("invalid_training_checkpoint", "CUDA RNG cannot be restored")
        torch.cuda.set_rng_state_all(states["torch_cuda"])
    return states["data_order_random"]


def _seed_run(seed):
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
    torch.manual_seed(seed)
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True)


def _strict_state_load(model, state):
    if not isinstance(state, dict):
        raise GraphEncoderError("invalid_training_checkpoint", "model_state must be a dict")
    expected = model.state_dict()
    if list(state) != list(expected):
        raise GraphEncoderError("invalid_training_checkpoint", "model state keys differ")
    for name, expected_value in expected.items():
        value = state[name]
        if not torch.is_tensor(value) or value.dtype != expected_value.dtype or tuple(value.shape) != tuple(expected_value.shape):
            raise GraphEncoderError(
                "invalid_training_checkpoint", "model tensor {} differs".format(name)
            )
    try:
        model.load_state_dict(state, strict=True)
    except Exception as exc:
        raise GraphEncoderError("invalid_training_checkpoint", "strict model reload failed") from exc


def _epoch_record_from_dict(value):
    return EpochTrainingRecord(
        value["epoch"],
        value["mean_loss"],
        tuple(sorted(value["mean_loss_components"].items())),
        value["optimizer_steps"],
        value["training_example_presentations"],
        tuple(value["family_order"]),
        tuple(tuple(item) for item in value["batch_boundaries"]),
        tuple(value["pre_clip_gradient_norms"]),
        value["data_preparation_seconds"],
        value["training_seconds"],
    )


def _plateau_from_dict(value):
    evaluations = tuple(PlateauEvaluation(**item) for item in value["evaluations"])
    return PlateauState(
        value["epsilon"], value["threshold"], value["window_epochs"],
        value["first_eligible_epoch"], value["first_plateau_epoch"],
        tuple(value["epoch_loss_history"]), evaluations
    )


def _require_torch():
    if torch is None:
        raise RuntimeError("C6 training requires PyTorch")
