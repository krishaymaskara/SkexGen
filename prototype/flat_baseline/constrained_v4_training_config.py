"""Versioned configuration for train-only constrained V4 smoke training."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math

from prototype.profile_geometry import PROFILE_EXTENT_MAX, PROFILE_EXTENT_MIN

from .constrained_v4_config import (
    CONSTRAINED_PROFILE_CHECKPOINT_VERSION,
    CONSTRAINED_PROFILE_DECODER_CONTRACT_VERSION,
    CONSTRAINED_PROFILE_FAMILY_ORDER,
    CONSTRAINED_PROFILE_MODEL_CONFIG_VERSION,
    CONSTRAINED_PROFILE_MODEL_NAME,
)


V4_TINY_SELECTION_IDENTITY = "train_metadata_greedy_coverage_v1"
V4_TRAINING_PARTITION = "train"
V4_NO_VALIDATION_PARTITION = "none"
V4_TRAINING_DTYPES = ("float32",)
V4_TRAINING_DEVICES = ("auto", "cpu", "cuda")
V4_TINY_VQ_INITIALIZATIONS = ("normal",)


class ConstrainedV4TrainingConfigurationError(ValueError):
    """A V4 tiny-training setting violates the frozen contract."""


@dataclass(frozen=True)
class ConstrainedV4TrainingConfig:
    """Deterministic bounded settings; not a full-experiment configuration."""

    model_name: str = CONSTRAINED_PROFILE_MODEL_NAME
    model_config_version: int = CONSTRAINED_PROFILE_MODEL_CONFIG_VERSION
    decoder_contract_version: int = (
        CONSTRAINED_PROFILE_DECODER_CONTRACT_VERSION
    )
    checkpoint_version: int = CONSTRAINED_PROFILE_CHECKPOINT_VERSION
    seed: int = 2026
    device: str = "cpu"
    dtype: str = "float32"
    batch_size: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    gradient_clip_norm: float = 1.0
    maximum_steps: int = 500
    logging_cadence: int = 10
    checkpoint_cadence: int = 500
    output_dir: str = "constrained_v4_tiny_overfit"
    extent_min: float = PROFILE_EXTENT_MIN
    extent_max: float = PROFILE_EXTENT_MAX
    profile_family_order: tuple = CONSTRAINED_PROFILE_FAMILY_ORDER
    node_type_loss_weight: float = 1.0
    categorical_loss_weight: float = 1.0
    remaining_geometry_loss_weight: float = 1.0
    profile_family_loss_weight: float = 1.0
    profile_parameter_loss_weight: float = 1.0
    edge_presence_loss_weight: float = 1.0
    edge_type_loss_weight: float = 1.0
    operation_pointer_loss_weight: float = 1.0
    vq_loss_weight: float = 1.0
    vq_initialization: str = "normal"
    training_partition: str = V4_TRAINING_PARTITION
    validation_partition: str = V4_NO_VALIDATION_PARTITION
    test_partition_accessed: bool = False
    split_name: str = "iid"
    tiny_subset_size: int = 8
    tiny_overfit_selection_identity: str = V4_TINY_SELECTION_IDENTITY
    require_clean_source: bool = True

    def validate(self):
        fixed = (
            ("model_name", CONSTRAINED_PROFILE_MODEL_NAME),
            ("model_config_version", CONSTRAINED_PROFILE_MODEL_CONFIG_VERSION),
            (
                "decoder_contract_version",
                CONSTRAINED_PROFILE_DECODER_CONTRACT_VERSION,
            ),
            ("checkpoint_version", CONSTRAINED_PROFILE_CHECKPOINT_VERSION),
            ("extent_min", PROFILE_EXTENT_MIN),
            ("extent_max", PROFILE_EXTENT_MAX),
            ("profile_family_order", CONSTRAINED_PROFILE_FAMILY_ORDER),
            ("training_partition", V4_TRAINING_PARTITION),
            ("validation_partition", V4_NO_VALIDATION_PARTITION),
            ("test_partition_accessed", False),
            (
                "tiny_overfit_selection_identity",
                V4_TINY_SELECTION_IDENTITY,
            ),
        )
        for name, expected in fixed:
            if getattr(self, name) != expected:
                raise ConstrainedV4TrainingConfigurationError(
                    "{} must equal {!r}".format(name, expected)
                )
        _nonnegative_integer(self.seed, "seed")
        for name in (
            "batch_size",
            "maximum_steps",
            "logging_cadence",
            "checkpoint_cadence",
            "tiny_subset_size",
        ):
            _positive_integer(getattr(self, name), name)
        if self.maximum_steps > 500:
            raise ConstrainedV4TrainingConfigurationError(
                "maximum_steps must not exceed the bounded ceiling 500"
            )
        if not 6 <= self.tiny_subset_size <= 12:
            raise ConstrainedV4TrainingConfigurationError(
                "tiny_subset_size must be between 6 and 12"
            )
        _positive_finite(self.learning_rate, "learning_rate")
        _nonnegative_finite(self.weight_decay, "weight_decay")
        _positive_finite(self.gradient_clip_norm, "gradient_clip_norm")
        for name in _LOSS_WEIGHT_FIELDS:
            _positive_finite(getattr(self, name), name)
        if self.device not in V4_TRAINING_DEVICES:
            raise ConstrainedV4TrainingConfigurationError(
                "unsupported device"
            )
        if self.dtype not in V4_TRAINING_DTYPES:
            raise ConstrainedV4TrainingConfigurationError(
                "unsupported dtype"
            )
        if self.vq_initialization not in V4_TINY_VQ_INITIALIZATIONS:
            raise ConstrainedV4TrainingConfigurationError(
                "tiny-overfit VQ initialization must be fresh normal"
            )
        for name in ("output_dir", "split_name"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ConstrainedV4TrainingConfigurationError(
                    "{} must be a nonempty string".format(name)
                )
        if not isinstance(self.require_clean_source, bool):
            raise ConstrainedV4TrainingConfigurationError(
                "require_clean_source must be Boolean"
            )

    def to_dict(self):
        self.validate()
        values = asdict(self)
        values["profile_family_order"] = list(self.profile_family_order)
        return values

    def to_json(self):
        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

    def loss_weights(self):
        values = self.to_dict()
        return {name: values[name] for name in _LOSS_WEIGHT_FIELDS}


_LOSS_WEIGHT_FIELDS = (
    "node_type_loss_weight",
    "categorical_loss_weight",
    "remaining_geometry_loss_weight",
    "profile_family_loss_weight",
    "profile_parameter_loss_weight",
    "edge_presence_loss_weight",
    "edge_type_loss_weight",
    "operation_pointer_loss_weight",
    "vq_loss_weight",
)


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConstrainedV4TrainingConfigurationError(
            "{} must be a positive integer".format(name)
        )


def _nonnegative_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ConstrainedV4TrainingConfigurationError(
            "{} must be a nonnegative integer".format(name)
        )


def _positive_finite(value, name):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ConstrainedV4TrainingConfigurationError(
            "{} must be finite and positive".format(name)
        )


def _nonnegative_finite(value, name):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise ConstrainedV4TrainingConfigurationError(
            "{} must be finite and nonnegative".format(name)
        )
