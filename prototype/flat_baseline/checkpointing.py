"""Atomic checkpoints and reproducible epoch-boundary resumption."""

from __future__ import annotations

import os
from pathlib import Path
import random
import tempfile
import math

from .training_config import TrainingConfig


class CheckpointError(ValueError):
    """A checkpoint is malformed or incompatible with the requested run."""

    def __init__(self, code, detail):
        self.code = code
        self.detail = detail
        super().__init__("{}: {}".format(code, detail))


def checkpoint_payload(
    model,
    optimizer,
    epoch,
    global_step,
    model_config,
    training_config,
    best_validation_metric,
    torch_module,
    data_state=None,
    checkpoint_kind="last",
):
    if checkpoint_kind not in ("last", "best"):
        raise CheckpointError(
            "invalid_checkpoint_kind",
            "checkpoint_kind must be 'last' or 'best'",
        )
    return {
        "checkpoint_version": 1,
        "checkpoint_kind": checkpoint_kind,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "model_config": model_config.to_dict(),
        "training_config": training_config.to_dict(),
        "best_validation_metric": best_validation_metric,
        "rng_state": capture_rng_state(torch_module),
        "data_state": data_state,
    }


def save_checkpoint(path, payload, torch_module):
    """Write and fsync a sibling temporary file before atomic replacement."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix="." + destination.name + ".",
            suffix=".tmp",
            dir=str(destination.parent),
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            torch_module.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(str(temporary_path), str(destination))
    except Exception:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
        raise


def load_checkpoint(
    path,
    model,
    optimizer,
    model_config,
    training_config,
    torch_module,
    map_location,
    restore_rng=True,
    expected_data_state=None,
):
    checkpoint = validated_checkpoint(
        path,
        model_config,
        training_config,
        torch_module,
        expected_data_state,
    )
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    _optimizer_to(optimizer, map_location, torch_module)
    if restore_rng:
        restore_rng_state(checkpoint["rng_state"], torch_module)
    return checkpoint


def validated_checkpoint(
    path,
    model_config,
    training_config,
    torch_module,
    expected_data_state=None,
):
    """Read a compatible checkpoint without mutating a model or RNG state."""

    # Keep CPU and CUDA RNG byte tensors on the CPU while deserializing.
    checkpoint = torch_module.load(str(path), map_location="cpu")
    _validate_payload(checkpoint)
    if checkpoint["model_config"] != model_config.to_dict():
        raise CheckpointError(
            "incompatible_model_config",
            "checkpoint model configuration differs from the requested model",
        )
    try:
        saved_training = TrainingConfig(**checkpoint["training_config"])
        saved_signature = saved_training.resume_signature()
    except (TypeError, ValueError) as exc:
        raise CheckpointError(
            "malformed_checkpoint",
            "checkpoint training configuration is invalid",
        ) from exc
    if saved_signature != training_config.resume_signature():
        raise CheckpointError(
            "incompatible_training_config",
            "checkpoint optimization configuration differs from the request",
        )
    if expected_data_state is not None:
        saved_data_state = checkpoint["data_state"]
        if not _compatible_data_state(
            saved_data_state, expected_data_state, saved_training
        ):
            raise CheckpointError(
                "incompatible_training_data",
                "checkpoint physical-family partitions differ from the request",
            )
    return checkpoint


def _compatible_data_state(saved, expected, training_config):
    if not isinstance(saved, dict) or not isinstance(expected, dict):
        return False
    if any(saved.get(name) != value for name, value in expected.items()):
        return False
    extra = set(saved) - set(expected)
    provenance = {
        "initialization_mode",
        "initialization_algorithm",
        "initialization_seed",
        "initialization_pseudo_count_policy",
        "initialization_report_sha256",
    }
    if not extra:
        return True
    if extra != provenance:
        return False
    return (
        saved["initialization_mode"] == training_config.vq_init
        and saved["initialization_seed"] == training_config.seed
        and isinstance(saved["initialization_algorithm"], str)
        and bool(saved["initialization_algorithm"])
        and isinstance(saved["initialization_pseudo_count_policy"], str)
        and bool(saved["initialization_pseudo_count_policy"])
        and isinstance(saved["initialization_report_sha256"], str)
        and len(saved["initialization_report_sha256"]) == 64
    )


def capture_rng_state(torch_module):
    state = {
        "python": random.getstate(),
        "torch_cpu": torch_module.get_rng_state(),
        "torch_cuda": None,
    }
    if torch_module.cuda.is_available():
        state["torch_cuda"] = torch_module.cuda.get_rng_state_all()
    return state


def restore_rng_state(state, torch_module):
    random.setstate(state["python"])
    torch_module.set_rng_state(state["torch_cpu"])
    if state.get("torch_cuda") is not None and torch_module.cuda.is_available():
        torch_module.cuda.set_rng_state_all(state["torch_cuda"])


def _validate_payload(checkpoint):
    required = {
        "checkpoint_version",
        "checkpoint_kind",
        "model_state",
        "optimizer_state",
        "epoch",
        "global_step",
        "model_config",
        "training_config",
        "best_validation_metric",
        "rng_state",
        "data_state",
    }
    if not isinstance(checkpoint, dict) or set(checkpoint) != required:
        raise CheckpointError(
            "malformed_checkpoint",
            "checkpoint fields do not match version 1",
        )
    if checkpoint["checkpoint_version"] != 1:
        raise CheckpointError(
            "unsupported_checkpoint",
            "checkpoint_version must be 1",
        )
    if checkpoint["checkpoint_kind"] not in ("last", "best"):
        raise CheckpointError(
            "malformed_checkpoint",
            "checkpoint_kind must be 'last' or 'best'",
        )
    for name in ("epoch", "global_step"):
        value = checkpoint[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
        ):
            raise CheckpointError(
                "malformed_checkpoint",
                "{} must be a nonnegative integer".format(name),
            )
    best = checkpoint["best_validation_metric"]
    if best is not None and (
        isinstance(best, bool)
        or not isinstance(best, (int, float))
        or not math.isfinite(float(best))
    ):
        raise CheckpointError(
            "malformed_checkpoint",
            "best_validation_metric must be finite or null",
        )


def _optimizer_to(optimizer, device, torch_module):
    for state in optimizer.state.values():
        for name, value in tuple(state.items()):
            if torch_module.is_tensor(value):
                state[name] = value.to(device)
