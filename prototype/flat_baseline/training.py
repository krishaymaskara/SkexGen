"""Reproducible teacher-forced optimization for B0-FLAT-MIXED-VQ."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import random
import sys

import torch

from .checkpointing import (
    CheckpointError,
    checkpoint_payload,
    load_checkpoint,
    save_checkpoint,
    validated_checkpoint,
)
from .config import FlatBaselineConfig
from .data import (
    load_training_data,
    make_data_loader,
    validate_model_selection_partitions,
)
from .losses import flat_mixed_vq_loss
from .model import FlatMixedVQModel
from .provenance import source_state
from .run_logging import JsonlLogger
from .training_config import LOSS_METRICS, TrainingConfig


class TrainingError(RuntimeError):
    """A run cannot continue without violating the training contract."""

    def __init__(self, code, detail):
        self.code = code
        self.detail = detail
        super().__init__("{}: {}".format(code, detail))


@dataclass(frozen=True)
class RunResult:
    output_dir: str
    last_checkpoint: str
    best_checkpoint: str
    metrics_path: str
    completed_epoch: int
    global_step: int
    best_validation_metric: float


class _MetricAccumulator:
    def __init__(self):
        self.examples = 0
        self.loss_sums = {name: 0.0 for name in LOSS_METRICS}
        self.assignment_counts = None

    def add(self, losses, output, example_count):
        if (
            isinstance(example_count, bool)
            or not isinstance(example_count, int)
            or example_count <= 0
        ):
            raise TrainingError(
                "invalid_example_count",
                "example_count must be a positive integer",
            )
        if set(losses.per_example) != set(LOSS_METRICS):
            raise TrainingError(
                "invalid_per_example_metrics",
                "per-example loss fields do not match the metric contract",
            )
        pending_sums = {}
        for name in LOSS_METRICS:
            values = losses.per_example[name].detach().to(
                device="cpu", dtype=torch.float64
            )
            if values.dim() != 1 or values.numel() != example_count:
                raise TrainingError(
                    "invalid_per_example_metrics",
                    "{} must contain one value per example".format(name),
                )
            if not bool(torch.isfinite(values).all().item()):
                raise TrainingError(
                    "nonfinite_metric",
                    "{} became non-finite".format(name),
                )
            pending_sums[name] = float(values.sum().item())
        raw_counts = output.assignment_counts.detach()
        if (
            raw_counts.dim() != 1
            or raw_counts.numel() == 0
            or raw_counts.dtype != torch.long
        ):
            raise TrainingError(
                "invalid_assignment_counts",
                "assignment_counts must be a nonempty rank-one torch.long tensor",
            )
        if bool((raw_counts < 0).any().item()):
            raise TrainingError(
                "invalid_assignment_counts",
                "assignment_counts must be nonnegative",
            )
        counts = raw_counts.to(device="cpu", dtype=torch.float64)
        if self.assignment_counts is None:
            self.assignment_counts = torch.zeros_like(counts)
        elif self.assignment_counts.numel() != counts.numel():
            raise TrainingError(
                "inconsistent_codebook_width",
                "assignment_counts width changed within one epoch",
            )
        self.examples += example_count
        for name, value in pending_sums.items():
            self.loss_sums[name] += value
        self.assignment_counts.add_(counts)

    def summary(self):
        if self.examples <= 0:
            raise TrainingError("empty_epoch", "an epoch processed no examples")
        if self.assignment_counts is None:
            raise TrainingError(
                "missing_assignment_counts",
                "an epoch reported no codebook assignments",
            )
        divisor = float(self.examples)
        total_assignments = float(self.assignment_counts.sum().item())
        active_count = int((self.assignment_counts > 0).sum().item())
        codebook_size = self.assignment_counts.numel()
        utilization = float(active_count) / float(codebook_size)
        perplexity = 0.0
        if total_assignments > 0.0:
            probabilities = self.assignment_counts / total_assignments
            nonzero = probabilities > 0
            entropy = -(
                probabilities[nonzero] * torch.log(probabilities[nonzero])
            ).sum()
            perplexity = float(torch.exp(entropy).item())
        if not all(
            math.isfinite(value)
            for value in (utilization, perplexity, float(active_count))
        ):
            raise TrainingError(
                "nonfinite_metric",
                "epoch-global codebook diagnostics became non-finite",
            )
        return {
            "number_of_examples": self.examples,
            "metrics": {
                name: self.loss_sums[name] / divisor for name in LOSS_METRICS
            },
            "codebook": {
                "active_code_count": active_count,
                "codebook_perplexity": perplexity,
                "codebook_utilization": utilization,
            },
        }


def seed_everything(seed, torch_module=torch):
    """Seed every RNG used by this implementation."""

    random.seed(seed)
    torch_module.manual_seed(seed)
    if torch_module.cuda.is_available():
        torch_module.cuda.manual_seed_all(seed)
    if hasattr(torch_module.backends, "cudnn"):
        torch_module.backends.cudnn.benchmark = False
        torch_module.backends.cudnn.deterministic = True


def resolve_device(selection, torch_module=torch):
    if selection == "auto":
        selection = "cuda" if torch_module.cuda.is_available() else "cpu"
    if selection == "cuda" and not torch_module.cuda.is_available():
        raise TrainingError(
            "cuda_unavailable",
            "CUDA was requested but torch.cuda.is_available() is false",
        )
    return torch_module.device(selection)


def train_epoch(
    model,
    optimizer,
    examples,
    model_config,
    training_config,
    device,
    epoch,
    global_step,
    torch_module=torch,
):
    """Train one epoch with active EMA updates and finite-gradient checks."""

    model.train()
    loader = make_data_loader(
        examples,
        training_config.batch_size,
        training_config.dataloader_workers,
        training_config.seed + epoch,
        True,
        torch_module,
    )
    accumulator = _MetricAccumulator()
    for batch in loader:
        inputs, target = _torch_batch(batch, device, torch_module)
        optimizer.zero_grad()
        output = model(target=target, **inputs)
        losses = flat_mixed_vq_loss(output, target, model_config)
        _require_finite_tensor(losses.total, "nonfinite_loss", torch_module)
        losses.total.backward()
        _require_finite_gradients(model, torch_module)
        gradient_norm = torch_module.nn.utils.clip_grad_norm_(
            model.parameters(), training_config.gradient_clip_norm
        )
        _require_finite_tensor(
            gradient_norm, "nonfinite_gradient_norm", torch_module
        )
        _require_finite_gradients(
            model, torch_module, "nonfinite_postclip_gradient"
        )
        optimizer.step()
        _require_finite_model_state(model, torch_module)
        _require_finite_optimizer_state(optimizer, torch_module)
        current_count = len(batch.family_ids)
        accumulator.add(losses, output, current_count)
        global_step += 1
    return accumulator.summary(), global_step


def evaluate_epoch(
    model,
    examples,
    model_config,
    training_config,
    device,
    torch_module=torch,
):
    """Evaluate deterministically without changing parameters or EMA buffers."""

    was_training = model.training
    model.eval()
    loader = make_data_loader(
        examples,
        training_config.batch_size,
        training_config.dataloader_workers,
        training_config.seed,
        False,
        torch_module,
    )
    accumulator = _MetricAccumulator()
    try:
        with torch_module.no_grad():
            for batch in loader:
                inputs, target = _torch_batch(batch, device, torch_module)
                output = model(target=target, **inputs)
                losses = flat_mixed_vq_loss(output, target, model_config)
                _require_finite_tensor(
                    losses.total, "nonfinite_validation_loss", torch_module
                )
                accumulator.add(losses, output, len(batch.family_ids))
    finally:
        if was_training:
            model.train()
    return accumulator.summary()


def train_run(
    corpus_dir,
    split_name="iid",
    train_partition="train",
    validation_partition="validation",
    model_config=None,
    training_config=None,
    resume_checkpoint=None,
    overfit_families=None,
    torch_module=torch,
):
    """Execute a complete epoch-boundary-resumable training run."""

    validate_model_selection_partitions(
        train_partition, validation_partition
    )
    model_config = model_config or FlatBaselineConfig()
    training_config = training_config or TrainingConfig()
    model_config.validate()
    training_config.validate()
    output_dir = Path(training_config.output_dir)
    resume_path = (
        None
        if resume_checkpoint is None
        else Path(resume_checkpoint).resolve()
    )
    _protect_output_directory(output_dir, resume_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = JsonlLogger(output_dir / "metrics.jsonl")
    try:
        data = load_training_data(
            corpus_dir,
            split_name,
            train_partition,
            validation_partition,
            overfit_families,
        )
        seed_everything(training_config.seed, torch_module)
        device = resolve_device(training_config.device, torch_module)
        model = FlatMixedVQModel(model_config).to(device)
        optimizer = torch_module.optim.AdamW(
            model.parameters(),
            lr=training_config.learning_rate,
            weight_decay=training_config.weight_decay,
        )
        start_epoch = 1
        global_step = 0
        best_metric = None
        best_path = output_dir / "best.pt"
        last_path = output_dir / "last.pt"
        data_state = _data_state(
            data, split_name, train_partition, validation_partition
        )
        if resume_path is not None:
            restored = load_checkpoint(
                resume_path,
                model,
                optimizer,
                model_config,
                training_config,
                torch_module,
                device,
                expected_data_state=data_state,
            )
            start_epoch = restored["epoch"] + 1
            global_step = restored["global_step"]
            best_metric = restored["best_validation_metric"]
            if best_metric is not None:
                _preserve_resumed_best(
                    resume_path,
                    restored["checkpoint_kind"],
                    best_path,
                    best_metric,
                    model_config,
                    training_config,
                    data_state,
                    torch_module,
                    restored["epoch"],
                    restored["global_step"],
                )
        if start_epoch > training_config.epochs:
            raise TrainingError(
                "resume_after_final_epoch",
                "checkpoint already completed the requested epoch count",
            )
        logger.write(
            _metadata_record(
                model_config,
                training_config,
                data,
                split_name,
                train_partition,
                validation_partition,
                device,
                torch_module,
                resume_path,
            )
        )
        completed_epoch = start_epoch - 1
        for epoch in range(start_epoch, training_config.epochs + 1):
            training_summary, global_step = train_epoch(
                model,
                optimizer,
                data.train_examples,
                model_config,
                training_config,
                device,
                epoch,
                global_step,
                torch_module,
            )
            logger.write(
                _epoch_record(
                    "train",
                    epoch,
                    global_step,
                    training_summary,
                    optimizer,
                )
            )
            should_validate = (
                epoch % training_config.validation_interval == 0
                or epoch == training_config.epochs
            )
            if should_validate:
                validation_summary = evaluate_epoch(
                    model,
                    data.validation_examples,
                    model_config,
                    training_config,
                    device,
                    torch_module,
                )
                logger.write(
                    _epoch_record(
                        "validation",
                        epoch,
                        global_step,
                        validation_summary,
                        optimizer,
                    )
                )
                candidate = validation_summary["metrics"][
                    training_config.checkpoint_selection_metric
                ]
                if best_metric is None or candidate < best_metric:
                    best_metric = candidate
                    payload = checkpoint_payload(
                        model,
                        optimizer,
                        epoch,
                        global_step,
                        model_config,
                        training_config,
                        best_metric,
                        torch_module,
                        data_state,
                        "best",
                    )
                    save_checkpoint(best_path, payload, torch_module)
                    logger.write(
                        _checkpoint_record(
                            "best", epoch, global_step, best_metric, best_path
                        )
                    )
            should_checkpoint = (
                epoch % training_config.checkpoint_interval == 0
                or epoch == training_config.epochs
            )
            if should_checkpoint:
                payload = checkpoint_payload(
                    model,
                    optimizer,
                    epoch,
                    global_step,
                    model_config,
                    training_config,
                    best_metric,
                    torch_module,
                    data_state,
                    "last",
                )
                save_checkpoint(last_path, payload, torch_module)
                logger.write(
                    _checkpoint_record(
                        "last", epoch, global_step, best_metric, last_path
                    )
                )
            completed_epoch = epoch
        if not best_path.is_file():
            raise TrainingError(
                "missing_best_checkpoint",
                "completed run has no best.pt checkpoint",
            )
        return RunResult(
            str(output_dir),
            str(last_path),
            str(best_path),
            str(logger.path),
            completed_epoch,
            global_step,
            float(best_metric),
        )
    except Exception as exc:
        logger.write(
            {
                "event": "failure",
                "failure_type": type(exc).__name__,
                "failure_code": getattr(exc, "code", "unexpected_training_error"),
                "detail": str(exc),
            }
        )
        raise


def _torch_batch(batch, device, torch_module):
    inputs = {
        name: tensor.to(device)
        for name, tensor in batch.to_torch(torch_module).items()
    }
    target = {
        name: tensor.to(device)
        for name, tensor in batch.target.to_torch(torch_module).items()
    }
    return inputs, target


def _require_finite_tensor(value, code, torch_module):
    if not bool(torch_module.isfinite(value.detach()).all().item()):
        raise TrainingError(code, "encountered a non-finite tensor")


def _require_finite_gradients(
    model, torch_module, code="nonfinite_gradient"
):
    for name, parameter in model.named_parameters():
        if parameter.grad is not None and not bool(
            torch_module.isfinite(parameter.grad).all().item()
        ):
            raise TrainingError(
                code,
                "parameter {!r} has a non-finite gradient".format(name),
            )


def _require_finite_model_state(model, torch_module):
    for kind, values in (
        ("parameter", model.named_parameters()),
        ("buffer", model.named_buffers()),
    ):
        for name, value in values:
            if _is_checked_tensor(value, torch_module) and not bool(
                torch_module.isfinite(value).all().item()
            ):
                raise TrainingError(
                    "nonfinite_model_state",
                    "{} {!r} became non-finite".format(kind, name),
                )


def _require_finite_optimizer_state(optimizer, torch_module):
    for parameter_index, state in enumerate(optimizer.state.values()):
        for path, value in _nested_tensors(state, torch_module):
            if _is_checked_tensor(value, torch_module) and not bool(
                torch_module.isfinite(value).all().item()
            ):
                raise TrainingError(
                    "nonfinite_optimizer_state",
                    "optimizer state {}{} became non-finite".format(
                        parameter_index, path
                    ),
                )


def _nested_tensors(value, torch_module, path=""):
    if torch_module.is_tensor(value):
        yield path, value
    elif isinstance(value, dict):
        for key in sorted(value, key=lambda item: str(item)):
            for item in _nested_tensors(
                value[key],
                torch_module,
                "{}[{}]".format(path, key),
            ):
                yield item
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            for item in _nested_tensors(
                child,
                torch_module,
                "{}[{}]".format(path, index),
            ):
                yield item


def _is_checked_tensor(value, torch_module):
    return torch_module.is_floating_point(value) or torch_module.is_complex(
        value
    )


def _metadata_record(
    model_config,
    training_config,
    data,
    split_name,
    train_partition,
    validation_partition,
    device,
    torch_module,
    resume_checkpoint,
):
    current_source = source_state()
    return {
        "event": "run_metadata",
        "model_config": model_config.to_dict(),
        "training_config": training_config.to_dict(),
        "git_commit": current_source["git_commit"],
        "git_dirty": current_source["git_dirty"],
        "git_status_porcelain": current_source["git_status_porcelain"],
        "source_tree_sha256": current_source["source_tree_sha256"],
        "python_version": sys.version.split()[0],
        "pytorch_version": str(torch_module.__version__),
        "resolved_device": str(device),
        "split_manifest": split_name,
        "train_partition": train_partition,
        "validation_partition": validation_partition,
        "train_family_ids": list(data.train_family_ids),
        "validation_family_ids": list(data.validation_family_ids),
        "resumed": resume_checkpoint is not None,
        "resume_checkpoint": (
            None if resume_checkpoint is None else str(resume_checkpoint)
        ),
        "determinism_note": (
            "seeded and deterministic evaluation; bitwise GPU determinism "
            "is not guaranteed by this milestone"
        ),
    }


def _data_state(data, split_name, train_partition, validation_partition):
    return {
        "split_manifest": split_name,
        "train_partition": train_partition,
        "validation_partition": validation_partition,
        "train_family_ids": list(data.train_family_ids),
        "validation_family_ids": list(data.validation_family_ids),
    }


def _protect_output_directory(output_dir, resume_path):
    if resume_path is not None:
        resume_directory = resume_path.parent
        if resume_directory.resolve() == output_dir.resolve():
            return
    managed = tuple(
        name
        for name in ("metrics.jsonl", "last.pt", "best.pt")
        if (output_dir / name).exists()
    )
    if managed:
        raise TrainingError(
            "managed_output_collision",
            "output contains managed artifacts: {}".format(
                ", ".join(managed)
            ),
        )


def _preserve_resumed_best(
    resume_path,
    resume_checkpoint_kind,
    destination,
    restored_metric,
    model_config,
    training_config,
    data_state,
    torch_module,
    resumed_epoch,
    resumed_global_step,
):
    associated = (
        resume_path
        if resume_checkpoint_kind == "best"
        else resume_path.parent / "best.pt"
    )
    if not associated.is_file():
        raise TrainingError(
            "missing_best_checkpoint",
            "restored best metric requires associated best.pt",
        )
    best_checkpoint = validated_checkpoint(
        associated,
        model_config,
        training_config,
        torch_module,
        data_state,
    )
    if best_checkpoint["checkpoint_kind"] != "best":
        raise CheckpointError(
            "inconsistent_best_checkpoint",
            "associated checkpoint is not marked as best",
        )
    associated_metric = best_checkpoint["best_validation_metric"]
    if associated_metric is None or float(associated_metric) != float(
        restored_metric
    ):
        raise CheckpointError(
            "inconsistent_best_checkpoint",
            "associated best.pt does not preserve the restored best metric",
        )
    if best_checkpoint["epoch"] > resumed_epoch:
        raise CheckpointError(
            "inconsistent_best_checkpoint",
            "associated best.pt is from a later epoch",
        )
    if best_checkpoint["global_step"] > resumed_global_step:
        raise CheckpointError(
            "inconsistent_best_checkpoint",
            "associated best.pt is from a later global step",
        )
    if associated.resolve() != destination.resolve():
        save_checkpoint(destination, best_checkpoint, torch_module)


def _epoch_record(mode, epoch, global_step, summary, optimizer):
    return {
        "event": "{}_epoch".format(mode),
        "epoch": epoch,
        "global_step": global_step,
        "learning_rate": float(optimizer.param_groups[0]["lr"]),
        "number_of_examples": summary["number_of_examples"],
        "metrics": summary["metrics"],
        "codebook": summary["codebook"],
    }


def _checkpoint_record(kind, epoch, global_step, best_metric, path):
    return {
        "event": "checkpoint",
        "checkpoint_kind": kind,
        "epoch": epoch,
        "global_step": global_step,
        "best_validation_metric": best_metric,
        "relative_path": path.name,
    }
