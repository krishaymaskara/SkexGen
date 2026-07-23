"""Directed physical-pair and encoding-specific edit-sample identities."""

from __future__ import annotations

from functools import lru_cache
import hashlib
from typing import Any

from prototype.controlled_data.builders import build_history
from prototype.controlled_data.factors import PhysicalSource
from prototype.controlled_data.identity import source_family_id
from prototype.representation.model import GeometryEncoding

from .config import (
    EDIT_IDENTITY_VERSION,
    EDIT_SAMPLE_IDENTITY_VERSION,
    canonical_json_bytes,
)
from .model import EditTarget, EditType


@lru_cache(maxsize=None)
def endpoint_source_family_id(source: PhysicalSource) -> str:
    history = build_history(source, GeometryEncoding.CONTINUOUS)
    return source_family_id(history)


def edit_family_id(
    edit_type: EditType,
    target: EditTarget,
    base_source_family_id: str,
    edited_source_family_id: str,
) -> str:
    descriptor = {
        "edit_identity_version": EDIT_IDENTITY_VERSION,
        "edit_type": edit_type.value,
        "target": target.to_dict(),
        "base_source_family_id": base_source_family_id,
        "edited_source_family_id": edited_source_family_id,
    }
    return "ef_" + hashlib.sha256(canonical_json_bytes(descriptor)).hexdigest()


def edit_sample_id(
    family_id: str,
    geometry_encoding: str,
    base_sample_id: str,
    edited_sample_id: str,
) -> str:
    descriptor = {
        "edit_sample_identity_version": EDIT_SAMPLE_IDENTITY_VERSION,
        "edit_family_id": family_id,
        "geometry_encoding": geometry_encoding,
        "base_sample_id": base_sample_id,
        "edited_sample_id": edited_sample_id,
    }
    return "ev_" + hashlib.sha256(canonical_json_bytes(descriptor)).hexdigest()


def physical_source_descriptor(source: PhysicalSource) -> dict[str, Any]:
    return {
        "operation_template": source.operation_template.value,
        "primitive_family": source.primitive_family.value,
        "reference_plane": source.reference_plane.value,
        "sketch_extents": list(source.sketch_extents),
        "directions": [item.value for item in source.directions],
        "later_boolean_mode": (
            source.later_boolean_mode.value
            if source.later_boolean_mode is not None
            else None
        ),
        "operation_parameters": list(source.operation_parameters),
        "extent_band": source.extent_band.value,
    }


def physical_source_bytes(source: PhysicalSource) -> bytes:
    return canonical_json_bytes(physical_source_descriptor(source))
