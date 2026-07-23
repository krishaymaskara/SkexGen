"""Encoding-independent source identity and encoding-specific sample identity."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

from prototype.representation.model import CADHistory
from prototype.representation.serialization import history_to_json

from .config import (
    CANONICALIZATION_VERSION,
    PHYSICAL_ID_DECIMAL_PLACES,
    REPRESENTATION_SCHEMA_VERSION,
)


def canonical_physical_source_bytes(history: CADHistory) -> bytes:
    """Canonical bytes over decoded physical values and canonical structure."""

    canonical_variant = json.loads(history_to_json(history))
    physical = _decode_numeric_records(canonical_variant)
    return _canonical_json_bytes(physical)


def source_family_id(history: CADHistory) -> str:
    return "sf_" + hashlib.sha256(canonical_physical_source_bytes(history)).hexdigest()


def sample_id(history: CADHistory) -> str:
    variant_bytes = history_to_json(history).encode("utf-8")
    identity_bytes = (
        f"schema={REPRESENTATION_SCHEMA_VERSION};canonicalization={CANONICALIZATION_VERSION};".encode(
            "ascii"
        )
        + variant_bytes
    )
    return "sv_" + hashlib.sha256(identity_bytes).hexdigest()


def _decode_numeric_records(value: Any) -> Any:
    if isinstance(value, dict):
        if set(value) in ({"encoding", "values"}, {"encoding", "values", "scale", "offset"}):
            encoding = value.get("encoding")
            raw_values = value.get("values")
            if not isinstance(raw_values, list):
                raise ValueError("numeric record values must be a list")
            if encoding == "continuous":
                return [_canonical_number(item) for item in raw_values]
            if encoding == "quantized":
                scale = _canonical_number(value.get("scale"))
                offset = _canonical_number(value.get("offset"))
                if scale <= 0:
                    raise ValueError("quantized scale must be positive")
                decoded = []
                for code in raw_values:
                    if isinstance(code, bool) or not isinstance(code, int):
                        raise ValueError("quantized codes must be integers")
                    decoded.append(_canonical_number(code * scale + offset))
                return decoded
            raise ValueError(f"unsupported geometry encoding {encoding!r}")
        return {key: _decode_numeric_records(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_decode_numeric_records(item) for item in value]
    if isinstance(value, float):
        return _canonical_number(value)
    return value


def _canonical_number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("physical values must be numbers and booleans are rejected")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError("physical values must be finite")
    # Twelve decimal places are well below the representation validator's
    # geometric tolerances and remove affine-decode noise such as 3 * 0.1.
    normalized = round(numeric, PHYSICAL_ID_DECIMAL_PLACES)
    return 0.0 if normalized == 0.0 else normalized


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
