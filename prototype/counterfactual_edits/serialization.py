"""Strict deterministic serialization for encoding-specific edit pairs."""

from __future__ import annotations

import json
from typing import Any

from .config import EDIT_PAIR_SCHEMA_VERSION, canonical_json_bytes
from .model import (
    AddressKind,
    EditSample,
    EditTarget,
    EditType,
    LocalityGroundTruth,
    RepresentationAddress,
)


class EditSerializationError(ValueError):
    pass


def edit_sample_to_json(sample: EditSample) -> str:
    return canonical_json_bytes(edit_sample_to_dict(sample)).decode("utf-8")


def edit_sample_to_dict(sample: EditSample) -> dict[str, Any]:
    return {
        "edit_pair_schema_version": EDIT_PAIR_SCHEMA_VERSION,
        "edit_sample_id": sample.edit_sample_id,
        "edit_family_id": sample.edit_family_id,
        "geometry_encoding": sample.geometry_encoding,
        "base": {
            "source_family_id": sample.base_source_family_id,
            "sample_id": sample.base_sample_id,
        },
        "edited": {
            "source_family_id": sample.edited_source_family_id,
            "sample_id": sample.edited_sample_id,
        },
        "edit": {
            "edit_type": sample.edit_type.value,
            **sample.target.to_dict(),
            "before": sample.before,
            "after": sample.after,
        },
        "locality": sample.locality.to_dict(),
    }


def edit_sample_from_json(payload: str) -> EditSample:
    try:
        value = json.loads(payload, object_pairs_hook=_no_duplicates)
    except Exception as exc:
        if isinstance(exc, EditSerializationError):
            raise
        raise EditSerializationError(f"malformed edit JSON: {type(exc).__name__}") from exc
    root = _object(
        value,
        {
            "edit_pair_schema_version",
            "edit_sample_id",
            "edit_family_id",
            "geometry_encoding",
            "base",
            "edited",
            "edit",
            "locality",
        },
    )
    if _integer(root["edit_pair_schema_version"]) != EDIT_PAIR_SCHEMA_VERSION:
        raise EditSerializationError("unsupported edit_pair_schema_version")
    base = _object(root["base"], {"source_family_id", "sample_id"})
    edited = _object(root["edited"], {"source_family_id", "sample_id"})
    edit = _object(
        root["edit"],
        {
            "edit_type",
            "operation_index",
            "node_id",
            "changed_field",
            "before",
            "after",
        },
    )
    locality = _locality(root["locality"])
    return EditSample(
        _string(root["edit_sample_id"]),
        _string(root["edit_family_id"]),
        _geometry_encoding(root["geometry_encoding"]),
        _string(base["source_family_id"]),
        _string(edited["source_family_id"]),
        _string(base["sample_id"]),
        _string(edited["sample_id"]),
        _enum(EditType, edit["edit_type"]),
        EditTarget(
            _integer(edit["operation_index"]),
            _string(edit["node_id"]),
            _string(edit["changed_field"]),
        ),
        _scalar(edit["before"]),
        _scalar(edit["after"]),
        locality,
    )


def _locality(value: Any) -> LocalityGroundTruth:
    obj = _object(
        value,
        {
            "allowed_changed_paths",
            "expected_unchanged_paths",
            "expected_unchanged_edges",
            "expected_operation_sequence",
            "causal_downstream_operation_ids",
            "structural_graph_edit_distance",
            "semantic_parameter_edit_distance",
        },
    )
    return LocalityGroundTruth(
        tuple(_address(item) for item in _list(obj["allowed_changed_paths"])),
        tuple(_address(item) for item in _list(obj["expected_unchanged_paths"])),
        tuple(
            tuple(_string(part) for part in _exact_list(item, 3))
            for item in _list(obj["expected_unchanged_edges"])
        ),
        tuple(_string(item) for item in _list(obj["expected_operation_sequence"])),
        tuple(
            _string(item) for item in _list(obj["causal_downstream_operation_ids"])
        ),
        _integer(obj["structural_graph_edit_distance"]),
        _integer(obj["semantic_parameter_edit_distance"]),
    )


def _address(value: Any) -> RepresentationAddress:
    if not isinstance(value, dict):
        raise EditSerializationError("address must be an object")
    required = {"kind", "owner_id", "field_path"}
    allowed = required | {"element_id"}
    if not required.issubset(value) or set(value) - allowed:
        raise EditSerializationError("address has missing or unknown fields")
    return RepresentationAddress(
        _enum(AddressKind, value["kind"]),
        _string(value["owner_id"]),
        _string(value["field_path"]),
        _string(value["element_id"]) if "element_id" in value else None,
    )


def _object(value: Any, fields: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise EditSerializationError(
            f"object fields differ: expected {sorted(fields)!r}"
        )
    return value


def _list(value: Any) -> list[Any]:
    if not isinstance(value, list):
        raise EditSerializationError("expected a list")
    return value


def _exact_list(value: Any, length: int) -> list[Any]:
    result = _list(value)
    if len(result) != length:
        raise EditSerializationError(f"expected a {length}-item list")
    return result


def _string(value: Any) -> str:
    if not isinstance(value, str):
        raise EditSerializationError("expected a string")
    return value


def _integer(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise EditSerializationError("expected an integer")
    return value


def _scalar(value: Any) -> Any:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise EditSerializationError("edit value must be a finite scalar")
    if isinstance(value, float) and not (float("-inf") < value < float("inf")):
        raise EditSerializationError("edit value must be finite")
    return value


def _geometry_encoding(value: Any) -> str:
    result = _string(value)
    if result not in ("continuous", "quantized"):
        raise EditSerializationError("unknown geometry encoding")
    return result


def _enum(enum_type, value: Any):
    try:
        return enum_type(_string(value))
    except ValueError as exc:
        raise EditSerializationError(f"unknown {enum_type.__name__}") from exc


def _no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise EditSerializationError(f"duplicate field {key!r}")
        result[key] = value
    return result
