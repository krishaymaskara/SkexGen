"""Serializable plumbing configuration for teacher-forced baseline training."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math


LOSS_METRICS = (
    "total",
    "node_type",
    "categorical_attributes",
    "geometry",
    "edge_presence",
    "edge_type",
    "operation_pointer",
    "vq_commitment",
)
DEVICE_CHOICES = ("auto", "cpu", "cuda")


class TrainingConfigurationError(ValueError):
    """A training-loop setting is malformed or unsupported."""


@dataclass(frozen=True)
class TrainingConfig:
    """Small reproducibility defaults; these are not final experiment settings."""

    seed: int = 0
    epochs: int = 2
    batch_size: int = 4
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    gradient_clip_norm: float = 1.0
    validation_interval: int = 1
    checkpoint_interval: int = 1
    dataloader_workers: int = 0
    device: str = "auto"
    output_dir: str = "flat_baseline_run"
    checkpoint_selection_metric: str = "total"

    def validate(self):
        _nonnegative_integer(self.seed, "seed")
        for name in (
            "epochs",
            "batch_size",
            "validation_interval",
            "checkpoint_interval",
        ):
            _positive_integer(getattr(self, name), name)
        _nonnegative_integer(self.dataloader_workers, "dataloader_workers")
        _positive_finite(self.learning_rate, "learning_rate")
        _nonnegative_finite(self.weight_decay, "weight_decay")
        _positive_finite(self.gradient_clip_norm, "gradient_clip_norm")
        if self.device not in DEVICE_CHOICES:
            raise TrainingConfigurationError(
                "device must be one of {}".format(", ".join(DEVICE_CHOICES))
            )
        if not isinstance(self.output_dir, str) or not self.output_dir.strip():
            raise TrainingConfigurationError("output_dir must be a nonempty string")
        if self.checkpoint_selection_metric not in LOSS_METRICS:
            raise TrainingConfigurationError(
                "checkpoint_selection_metric must name a validation loss"
            )

    def to_dict(self):
        self.validate()
        return asdict(self)

    def to_json(self):
        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

    def resume_signature(self):
        """Settings which must match to continue the same optimization run."""

        values = self.to_dict()
        # A run may be extended, moved to another output directory, or resumed
        # on another device without changing its optimization semantics.
        for name in ("epochs", "output_dir", "device"):
            values.pop(name)
        return values


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise TrainingConfigurationError(
            "{} must be a positive integer".format(name)
        )


def _nonnegative_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TrainingConfigurationError(
            "{} must be a nonnegative integer".format(name)
        )


def _positive_finite(value, name):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise TrainingConfigurationError(
            "{} must be finite and strictly positive".format(name)
        )


def _nonnegative_finite(value, name):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise TrainingConfigurationError(
            "{} must be finite and nonnegative".format(name)
        )
