"""Frozen configuration and version constants for counterfactual edits."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from typing import Any

from prototype.controlled_data.config import GeneratorConfig


GENERATOR_VERSION = "1.0"
EDIT_PAIR_SCHEMA_VERSION = 1
EDIT_IDENTITY_VERSION = 1
EDIT_SAMPLE_IDENTITY_VERSION = 1
ORIENTATION_POLICY_VERSION = "counterfactual-orientation-v1"
SELECTION_POLICY_VERSION = "counterfactual-greedy-matching-v1"
EDIT_CANONICALIZATION_VERSION = "counterfactual-json-v1"


class EditConfigurationError(ValueError):
    """The requested edit corpus cannot satisfy its frozen contract."""


@dataclass(frozen=True)
class CounterfactualConfig:
    num_edit_families: int
    seed: int = 0
    factor_config: GeneratorConfig = field(
        default_factory=lambda: GeneratorConfig(num_source_families=68, seed=0)
    )
    iid_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1)
    validation_ratio: float = 0.1
    require_full_coverage: bool = True

    def validate(self) -> None:
        if isinstance(self.num_edit_families, bool) or not isinstance(
            self.num_edit_families, int
        ):
            raise EditConfigurationError("num_edit_families must be an integer")
        if self.num_edit_families <= 0:
            raise EditConfigurationError("num_edit_families must be positive")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise EditConfigurationError("seed must be an integer")
        self.factor_config.validate()

    def normalized(self) -> dict[str, Any]:
        self.validate()
        value = asdict(self)
        return _normalize(value)

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.normalized())

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


def primary_token_declarations() -> tuple[tuple[str, str, int, str], ...]:
    from prototype.controlled_data.factors import ExtentBand, OperationTemplate

    from .model import EditType

    return tuple(
        (edit_type.value, template.value, position, band.value)
        for edit_type in EditType
        for template in OperationTemplate
        for position, operation in enumerate(template.operations)
        if (
            edit_type in (EditType.PROFILE_EXTENT, EditType.OPERATION_DIRECTION)
            or (
                edit_type is EditType.EXTRUSION_DISTANCE
                and operation == "extrude"
            )
            or (
                edit_type is EditType.REVOLVE_ANGLE
                and operation == "revolve"
            )
            or (
                edit_type is EditType.BOOLEAN_MODE
                and len(template.operations) == 2
                and position == 1
            )
        )
        for band in ExtentBand
    )


def required_minimum() -> int:
    return len(primary_token_declarations())


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _normalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _normalize(value[key]) for key in sorted(value)}
    if isinstance(value, (tuple, list)):
        return [_normalize(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise EditConfigurationError(f"unsupported configuration value {type(value).__name__}")
