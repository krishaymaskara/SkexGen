"""Frozen authorization for the Stage 2H reference-plane diagnostic."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json


DIAGNOSTIC_IDENTITY = (
    "B0-FLAT-CONSTRAINED-PROFILE-v2-reference-plane-diagnostic-v1"
)
DIAGNOSTIC_VERSION = 1
EXPECTED_DIAGNOSTIC_SOURCE_BASE = (
    "d62889b241b2a07876e2fe5f3c82c82bce7851c9"
)
EXPECTED_PILOT_SOURCE_COMMIT = (
    "9e662310b41b3750a49547b95fa6af7a9235a8b4"
)
EXPECTED_CHECKPOINT_SHA256 = (
    "516c480ad7477bd93d14ece7247057a831760a7399552845def2ac2846eb85b7"
)
EXPECTED_GLOBAL_STEP = 136
EXPECTED_EPOCH = 2
EXPECTED_TRAIN_COUNT = 544
EXPECTED_VALIDATION_COUNT = 68


class ReferencePlaneDiagnosticError(RuntimeError):
    """The frozen reference-plane diagnostic cannot continue safely."""

    def __init__(self, code, detail):
        self.code = code
        self.detail = detail
        super().__init__("{}: {}".format(code, detail))


@dataclass(frozen=True)
class ReferencePlaneDiagnosticConfig:
    diagnostic_identity: str = DIAGNOSTIC_IDENTITY
    diagnostic_version: int = DIAGNOSTIC_VERSION
    expected_diagnostic_source_base: str = EXPECTED_DIAGNOSTIC_SOURCE_BASE
    expected_pilot_source_commit: str = EXPECTED_PILOT_SOURCE_COMMIT
    expected_checkpoint_sha256: str = EXPECTED_CHECKPOINT_SHA256
    expected_global_step: int = EXPECTED_GLOBAL_STEP
    expected_epoch: int = EXPECTED_EPOCH
    expected_train_count: int = EXPECTED_TRAIN_COUNT
    expected_validation_count: int = EXPECTED_VALIDATION_COUNT
    split_name: str = "iid"
    training_partition: str = "train"
    validation_partition: str = "validation"
    validation_partition_identity: str = "iid_validation"
    batch_size: int = 8
    device: str = "cpu"
    output_dir: str = "constrained_v2_reference_plane_diagnostic"
    require_clean_source: bool = True
    systematic_partition_accessed: bool = False
    test_partition_accessed: bool = False

    def validate(self):
        fixed = (
            ("diagnostic_identity", DIAGNOSTIC_IDENTITY),
            ("diagnostic_version", DIAGNOSTIC_VERSION),
            ("expected_diagnostic_source_base", EXPECTED_DIAGNOSTIC_SOURCE_BASE),
            ("expected_pilot_source_commit", EXPECTED_PILOT_SOURCE_COMMIT),
            ("expected_checkpoint_sha256", EXPECTED_CHECKPOINT_SHA256),
            ("expected_global_step", EXPECTED_GLOBAL_STEP),
            ("expected_epoch", EXPECTED_EPOCH),
            ("expected_train_count", EXPECTED_TRAIN_COUNT),
            ("expected_validation_count", EXPECTED_VALIDATION_COUNT),
            ("split_name", "iid"),
            ("training_partition", "train"),
            ("validation_partition", "validation"),
            ("validation_partition_identity", "iid_validation"),
            ("batch_size", 8),
            ("systematic_partition_accessed", False),
            ("test_partition_accessed", False),
        )
        for name, expected in fixed:
            value = getattr(self, name)
            if value != expected or type(value) is not type(expected):
                raise ReferencePlaneDiagnosticError(
                    "invalid_diagnostic_configuration",
                    "{} must equal {!r}".format(name, expected),
                )
        if self.device not in ("cpu", "cuda"):
            raise ReferencePlaneDiagnosticError(
                "invalid_diagnostic_configuration", "unsupported device"
            )
        if not isinstance(self.output_dir, str) or not self.output_dir.strip():
            raise ReferencePlaneDiagnosticError(
                "invalid_diagnostic_configuration",
                "output_dir must be a nonempty string",
            )
        if not isinstance(self.require_clean_source, bool):
            raise ReferencePlaneDiagnosticError(
                "invalid_diagnostic_configuration",
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


def validate_reference_plane_partition_authorization(config):
    config.validate()
    if (
        config.training_partition != "train"
        or config.validation_partition != "validation"
        or config.validation_partition_identity != "iid_validation"
        or config.systematic_partition_accessed is not False
        or config.test_partition_accessed is not False
    ):
        raise ReferencePlaneDiagnosticError(
            "unauthorized_partition",
            "diagnostic permits only train plus ordinary IID validation",
        )
