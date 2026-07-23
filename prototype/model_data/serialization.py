"""Canonical environment-independent serialization for contract regression tests."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
import math

from .errors import ModelDataError


def canonical_record_json(value):
    return json.dumps(
        _normalize(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _normalize(value):
    if is_dataclass(value):
        return _normalize(asdict(value))
    if isinstance(value, dict):
        return {key: _normalize(value[key]) for key in sorted(value)}
    if isinstance(value, (tuple, list)):
        return [_normalize(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ModelDataError("nonfinite_record", repr(value))
        return 0.0 if value == 0.0 else value
    if value is None or isinstance(value, (bool, int, str)):
        return value
    raise ModelDataError("unsupported_record_value", type(value).__name__)
