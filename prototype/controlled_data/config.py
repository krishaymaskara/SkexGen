"""Frozen configuration and constants for controlled corpus generation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any


GENERATOR_VERSION = "1.0"
CANONICALIZATION_VERSION = "cad-history-json-v1"
REPRESENTATION_SCHEMA_VERSION = 1
PHYSICAL_ID_DECIMAL_PLACES = 12
LENGTH_QUANTIZATION_SCALE = 0.125
ANGLE_QUANTIZATION_SCALE = 1.0


class ConfigurationError(ValueError):
    """A generator request cannot satisfy the frozen corpus contract."""


@dataclass(frozen=True)
class GeneratorConfig:
    """All values that affect bounded corpus selection or physical geometry."""

    num_source_families: int
    seed: int = 0
    sketch_extents: tuple[float, ...] = (1.0, 1.5, 2.0, 2.5, 3.0)
    extrusion_distances: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 3.0)
    revolution_angles: tuple[float, ...] = (45.0, 90.0, 180.0, 270.0, 360.0)
    in_range_extent_max: float = 2.0
    extrapolation_extent_min: float = 2.5
    iid_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1)
    validation_ratio: float = 0.1
    require_full_coverage: bool = True

    def validate(self) -> None:
        if isinstance(self.num_source_families, bool) or not isinstance(self.num_source_families, int):
            raise ConfigurationError("num_source_families must be an integer (booleans are rejected)")
        if self.num_source_families <= 0:
            raise ConfigurationError("num_source_families must be positive")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ConfigurationError("seed must be an integer (booleans are rejected)")
        for name, values in (
            ("sketch_extents", self.sketch_extents),
            ("extrusion_distances", self.extrusion_distances),
            ("revolution_angles", self.revolution_angles),
        ):
            if not isinstance(values, tuple) or not values:
                raise ConfigurationError(f"{name} must be a nonempty tuple")
            if len(set(values)) != len(values):
                raise ConfigurationError(f"{name} must not contain duplicates")
            for value in values:
                _finite_number(value, name)
            canonical = [round(float(value), PHYSICAL_ID_DECIMAL_PLACES) for value in values]
            if len(set(canonical)) != len(canonical):
                raise ConfigurationError(
                    f"{name} contains distinct values that collide after "
                    f"{PHYSICAL_ID_DECIMAL_PLACES}-decimal physical-ID normalization"
                )
        if any(value <= 0 for value in self.sketch_extents):
            raise ConfigurationError("sketch extents must be positive")
        if any(value <= 0 for value in self.extrusion_distances):
            raise ConfigurationError("extrusion distances must be positive")
        if any(not 0 < value <= 360 for value in self.revolution_angles):
            raise ConfigurationError("revolution angles must lie in (0, 360]")
        for extent in self.sketch_extents:
            _require_quantizable(extent / 4, LENGTH_QUANTIZATION_SCALE, "sketch_extents")
        for distance in self.extrusion_distances:
            _require_quantizable(distance, LENGTH_QUANTIZATION_SCALE, "extrusion_distances")
        for angle in self.revolution_angles:
            _require_quantizable(angle, ANGLE_QUANTIZATION_SCALE, "revolution_angles")
        _finite_number(self.in_range_extent_max, "in_range_extent_max")
        _finite_number(self.extrapolation_extent_min, "extrapolation_extent_min")
        if self.in_range_extent_max >= self.extrapolation_extent_min:
            raise ConfigurationError("extent bands must be disjoint")
        if not any(value <= self.in_range_extent_max for value in self.sketch_extents):
            raise ConfigurationError("sketch_extents has no training-range value")
        if not any(value >= self.extrapolation_extent_min for value in self.sketch_extents):
            raise ConfigurationError("sketch_extents has no extrapolation-range value")
        if any(self.in_range_extent_max < value < self.extrapolation_extent_min for value in self.sketch_extents):
            raise ConfigurationError("sketch_extents contains a value in the excluded band gap")
        if not isinstance(self.iid_ratios, tuple) or len(self.iid_ratios) != 3:
            raise ConfigurationError("iid_ratios must be a three-value tuple")
        for ratio in self.iid_ratios:
            _finite_number(ratio, "iid_ratios")
            if ratio < 0:
                raise ConfigurationError("IID ratios must be nonnegative")
        if not math.isclose(sum(self.iid_ratios), 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ConfigurationError("IID ratios must sum to 1")
        _finite_number(self.validation_ratio, "validation_ratio")
        if not 0 < self.validation_ratio < 1:
            raise ConfigurationError("validation_ratio must lie in (0, 1)")
        if not isinstance(self.require_full_coverage, bool):
            raise ConfigurationError("require_full_coverage must be a boolean")
        minimum = required_coverage_family_count(self)
        if self.require_full_coverage and self.num_source_families < minimum:
            raise ConfigurationError(
                f"at least {minimum} source families are required for the configured "
                "per-template, per-extent-band coverage anchors"
            )

    def normalized(self) -> dict[str, Any]:
        self.validate()
        raw = asdict(self)
        return _normalize_json_value(raw)

    def canonical_bytes(self) -> bytes:
        return json.dumps(
            self.normalized(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f"{name} values must be numbers (booleans are rejected)")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ConfigurationError(f"{name} values must be finite")
    return 0.0 if numeric == 0.0 else numeric


def _require_quantizable(value: float, scale: float, name: str) -> None:
    code = round(float(value) / scale)
    if not math.isclose(code * scale, float(value), rel_tol=0.0, abs_tol=1e-12):
        raise ConfigurationError(
            f"{name} value {value!r} is not exactly representable with quantization scale {scale}"
        )


def required_coverage_family_count(config: GeneratorConfig) -> int:
    """Families needed by diagonal anchors in all template/band blocks."""

    total = 0
    for template in ("E", "R", "EE", "ER", "RE", "RR"):
        parameter_sizes = [
            len(config.extrusion_distances) if operation == "E" else len(config.revolution_angles)
            for operation in template
        ]
        anchors_per_band = max(
            3,  # primitive families
            3,  # reference planes
            2,  # both direction values
            2 if len(template) == 2 else 1,  # later JOIN/CUT where applicable
            *parameter_sizes,
        )
        total += 2 * anchors_per_band  # in-range and extrapolation blocks
    return total


def _normalize_json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _normalize_json_value(value[key]) for key in sorted(value)}
    if isinstance(value, (tuple, list)):
        return [_normalize_json_value(item) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return _finite_number(value, "configuration")
    raise ConfigurationError(f"unsupported configuration value {type(value).__name__}")
