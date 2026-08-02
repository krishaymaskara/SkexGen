"""Frozen configuration and partition boundary for the V6 IID pilot."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json


PILOT_IDENTITY = (
    "B0-FLAT-CONSTRAINED-PROFILE-CANONICAL-PLANE-"
    "NODE-CATEGORIES-PREFIX-GRAMMAR-CONSTRAINED-AXIS-v6-iid-pilot-v1"
)
PILOT_CONFIG_VERSION = 1
PILOT_EPOCHS = 2
PILOT_BATCH_SIZE = 8
PILOT_VALIDATION_INTERVAL = 1
PILOT_CHECKPOINT_INTERVAL = 1
PILOT_TRAIN_PARTITION = "train"
PILOT_VALIDATION_PARTITION = "validation"
PILOT_VALIDATION_IDENTITY = "iid_validation"
PILOT_EXPECTED_TRAIN_EXAMPLES = 544
PILOT_EXPECTED_VALIDATION_EXAMPLES = 68
PILOT_EXPECTED_OPTIMIZER_STEPS = 136
PILOT_EXPECTED_EXAMPLES_PROCESSED = 1088


class ConstrainedV6PilotError(RuntimeError):
    """The bounded pilot contract cannot continue safely."""

    def __init__(self, code, detail):
        self.code = code
        self.detail = detail
        super().__init__("{}: {}".format(code, detail))


@dataclass(frozen=True)
class ConstrainedV6PilotConfig:
    pilot_identity: str = PILOT_IDENTITY
    pilot_config_version: int = PILOT_CONFIG_VERSION
    seed: int = 2026
    epochs: int = PILOT_EPOCHS
    batch_size: int = PILOT_BATCH_SIZE
    validation_interval: int = PILOT_VALIDATION_INTERVAL
    checkpoint_interval: int = PILOT_CHECKPOINT_INTERVAL
    device: str = "cpu"
    split_name: str = "iid"
    training_partition: str = PILOT_TRAIN_PARTITION
    validation_partition: str = PILOT_VALIDATION_PARTITION
    validation_partition_identity: str = PILOT_VALIDATION_IDENTITY
    vq_initialization: str = "normal"
    checkpoint_selection_metric: str = "ordinary_validation_total_loss"
    output_dir: str = "constrained_v6_constrained_axis_iid_pilot"
    require_clean_source: bool = True
    systematic_partition_accessed: bool = False
    test_partition_accessed: bool = False

    def validate(self):
        if type(self.systematic_partition_accessed) is not bool or type(
            self.test_partition_accessed
        ) is not bool:
            raise ConstrainedV6PilotError(
                "invalid_pilot_configuration",
                "partition access facts must be Boolean",
            )
        fixed = (
            ("pilot_identity", PILOT_IDENTITY),
            ("pilot_config_version", PILOT_CONFIG_VERSION),
            ("epochs", PILOT_EPOCHS),
            ("batch_size", PILOT_BATCH_SIZE),
            ("validation_interval", PILOT_VALIDATION_INTERVAL),
            ("checkpoint_interval", PILOT_CHECKPOINT_INTERVAL),
            ("split_name", "iid"),
            ("training_partition", PILOT_TRAIN_PARTITION),
            ("validation_partition", PILOT_VALIDATION_PARTITION),
            (
                "validation_partition_identity",
                PILOT_VALIDATION_IDENTITY,
            ),
            ("vq_initialization", "normal"),
            (
                "checkpoint_selection_metric",
                "ordinary_validation_total_loss",
            ),
            ("systematic_partition_accessed", False),
            ("test_partition_accessed", False),
        )
        for name, expected in fixed:
            if getattr(self, name) != expected:
                raise ConstrainedV6PilotError(
                    "invalid_pilot_configuration",
                    "{} must equal {!r}".format(name, expected),
                )
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
        ):
            raise ConstrainedV6PilotError(
                "invalid_pilot_configuration",
                "seed must be a nonnegative integer",
            )
        if self.device not in ("cpu", "cuda"):
            raise ConstrainedV6PilotError(
                "invalid_pilot_configuration", "unsupported device"
            )
        if not isinstance(self.output_dir, str) or not self.output_dir.strip():
            raise ConstrainedV6PilotError(
                "invalid_pilot_configuration",
                "output_dir must be a nonempty string",
            )
        if not isinstance(self.require_clean_source, bool):
            raise ConstrainedV6PilotError(
                "invalid_pilot_configuration",
                "require_clean_source must be Boolean",
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


def validate_pilot_partition_authorization(config):
    config.validate()
    if (
        config.training_partition != PILOT_TRAIN_PARTITION
        or config.validation_partition != PILOT_VALIDATION_PARTITION
        or config.validation_partition_identity != PILOT_VALIDATION_IDENTITY
        or config.systematic_partition_accessed
        or config.test_partition_accessed
    ):
        raise ConstrainedV6PilotError(
            "unauthorized_partition",
            "pilot permits only train plus ordinary IID validation",
        )
