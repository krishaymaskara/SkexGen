"""Exact representation-path differencing for v1 localized edits."""

from __future__ import annotations

import json
from typing import Any

from prototype.representation.model import CADHistory
from prototype.representation.serialization import history_to_json

from .config import EditConfigurationError
from .model import (
    AddressKind,
    EditTarget,
    EditType,
    LocalityGroundTruth,
    RepresentationAddress,
)


def build_locality_ground_truth(
    base: CADHistory,
    edited: CADHistory,
    edit_type: EditType,
    target: EditTarget,
) -> LocalityGroundTruth:
    base_map, base_edges, base_sequence = _address_map(base)
    edited_map, edited_edges, edited_sequence = _address_map(edited)
    if set(base_map) != set(edited_map):
        raise EditConfigurationError("v1 edit changes the representation address set")
    if base_edges != edited_edges:
        raise EditConfigurationError("v1 edit changes graph edges")
    if base_sequence != edited_sequence:
        raise EditConfigurationError("v1 edit changes operation order")
    changed = tuple(sorted(
        (address for address in base_map if base_map[address] != edited_map[address]),
        key=_address_key,
    ))
    if not changed:
        raise EditConfigurationError("edit produces no representation change")
    for address in changed:
        if not _is_allowed(address, edit_type, target):
            raise EditConfigurationError(
                f"undeclared representation change at {address.to_dict()!r}"
            )
    unchanged = tuple(sorted((set(base_map) - set(changed)), key=_address_key))
    downstream = base_sequence[target.operation_index :]
    result = LocalityGroundTruth(
        allowed_changed_paths=changed,
        expected_unchanged_paths=unchanged,
        expected_unchanged_edges=base_edges,
        expected_operation_sequence=base_sequence,
        causal_downstream_operation_ids=downstream,
    )
    validate_locality(base, edited, result, edit_type, target)
    return result


def validate_locality(
    base: CADHistory,
    edited: CADHistory,
    expected: LocalityGroundTruth,
    edit_type: EditType,
    target: EditTarget,
) -> None:
    base_map, base_edges, base_sequence = _address_map(base)
    edited_map, edited_edges, edited_sequence = _address_map(edited)
    if set(base_map) != set(edited_map):
        raise EditConfigurationError("v1 edit changes the representation address set")
    if base_edges != edited_edges:
        raise EditConfigurationError("v1 edit changes graph edges")
    if base_sequence != edited_sequence:
        raise EditConfigurationError("v1 edit changes operation order")
    if isinstance(target.operation_index, bool) or not isinstance(
        target.operation_index, int
    ) or not 0 <= target.operation_index < len(base_sequence):
        raise EditConfigurationError("edit target operation index is invalid")

    changed = {
        address
        for address in base_map
        if base_map[address] != edited_map[address]
    }
    unchanged = set(base_map) - changed
    allowed_sequence = expected.allowed_changed_paths
    unchanged_sequence = expected.expected_unchanged_paths
    if len(allowed_sequence) != len(set(allowed_sequence)):
        raise EditConfigurationError("allowed changed paths contain duplicates")
    if len(unchanged_sequence) != len(set(unchanged_sequence)):
        raise EditConfigurationError("expected unchanged paths contain duplicates")
    allowed = set(allowed_sequence)
    declared_unchanged = set(unchanged_sequence)
    if allowed & declared_unchanged:
        raise EditConfigurationError("changed and unchanged locality paths overlap")
    if allowed | declared_unchanged != set(base_map):
        raise EditConfigurationError(
            "locality paths do not exhaust the authoritative address set"
        )
    if changed != allowed:
        raise EditConfigurationError("actual changed paths differ from locality ground truth")
    if unchanged != declared_unchanged:
        raise EditConfigurationError(
            "expected unchanged paths differ from authoritative unchanged paths"
        )
    if any(not _is_allowed(address, edit_type, target) for address in changed):
        raise EditConfigurationError("changed path is not approved for the declared edit")
    if base_edges != expected.expected_unchanged_edges or edited_edges != base_edges:
        raise EditConfigurationError("edge preservation check failed")
    if base_sequence != expected.expected_operation_sequence or edited_sequence != base_sequence:
        raise EditConfigurationError("operation-order preservation check failed")
    if expected.structural_graph_edit_distance != 0:
        raise EditConfigurationError("structural graph edit distance must be zero")
    if expected.semantic_parameter_edit_distance != 1:
        raise EditConfigurationError("semantic parameter edit distance must be one")
    downstream = base_sequence[target.operation_index :]
    if expected.causal_downstream_operation_ids != downstream:
        raise EditConfigurationError("causal downstream operation IDs are incorrect")


def _address_map(
    history: CADHistory,
) -> tuple[
    dict[RepresentationAddress, Any],
    tuple[tuple[str, str, str], ...],
    tuple[str, ...],
]:
    value = json.loads(history_to_json(history))
    result: dict[RepresentationAddress, Any] = {}
    for node in value["structure"]["nodes"]:
        owner = node["node_id"]
        for path, leaf in _flatten(
            {key: item for key, item in node.items() if key != "node_id"}
        ):
            _insert_address(
                result,
                RepresentationAddress(AddressKind.NODE_FIELD, owner, path),
                leaf,
            )
    for record in value["geometry"]["node_geometry"]:
        for path, leaf in _flatten(record["geometry"]):
            _insert_address(
                result,
                RepresentationAddress(
                    AddressKind.NODE_GEOMETRY, record["node_id"], path
                ),
                leaf,
            )
    for record in value["geometry"]["sketch_element_geometry"]:
        for path, leaf in _flatten(record["geometry"]):
            _insert_address(
                result,
                RepresentationAddress(
                    AddressKind.SKETCH_ELEMENT_GEOMETRY,
                    record["sketch_id"],
                    path,
                    record["element_id"],
                ),
                leaf,
            )
    edges = tuple(
        sorted(
            (
                edge["source_id"],
                edge["edge_type"],
                edge["target_id"],
            )
            for edge in value["structure"]["edges"]
        )
    )
    return result, edges, tuple(value["structure"]["operation_sequence"])


def _insert_address(
    result: dict[RepresentationAddress, Any],
    address: RepresentationAddress,
    value: Any,
) -> None:
    if address in result:
        raise EditConfigurationError(
            f"duplicate representation address {address.to_dict()!r}"
        )
    result[address] = value


def _flatten(value: Any, prefix: str = ""):
    if isinstance(value, dict):
        if set(value) in (
            {"encoding", "values"},
            {"encoding", "values", "scale", "offset"},
        ):
            encoding = value["encoding"]
            values = value["values"]
            if encoding == "continuous":
                decoded = values
            elif encoding == "quantized":
                decoded = [
                    code * value["scale"] + value["offset"] for code in values
                ]
            else:
                raise EditConfigurationError(
                    f"unknown geometry encoding {encoding!r}"
                )
            for index, item in enumerate(decoded):
                yield f"{prefix}.values[{index}]", (
                    0.0 if float(item) == 0.0 else float(item)
                )
            return
        for key in sorted(value):
            path = f"{prefix}.{key}" if prefix else key
            yield from _flatten(value[key], path)
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            yield from _flatten(item, f"{prefix}[{index}]")
        return
    yield prefix, value


def _is_allowed(
    address: RepresentationAddress, edit_type: EditType, target: EditTarget
) -> bool:
    if edit_type is EditType.PROFILE_EXTENT:
        return (
            address.kind is AddressKind.SKETCH_ELEMENT_GEOMETRY
            and address.owner_id == target.node_id
        )
    if edit_type is EditType.EXTRUSION_DISTANCE:
        return (
            address.kind is AddressKind.NODE_GEOMETRY
            and address.owner_id == target.node_id
            and address.field_path.startswith("distance.")
        )
    if edit_type is EditType.REVOLVE_ANGLE:
        return (
            address.kind is AddressKind.NODE_GEOMETRY
            and address.owner_id == target.node_id
            and address.field_path.startswith("angle_degrees.")
        )
    if edit_type is EditType.OPERATION_DIRECTION:
        return (
            address.kind is AddressKind.NODE_FIELD
            and address.owner_id == target.node_id
            and address.field_path == "direction"
        )
    if edit_type is EditType.BOOLEAN_MODE:
        return (
            address.kind is AddressKind.NODE_FIELD
            and address.owner_id == target.node_id
            and address.field_path == "boolean_mode"
        )
    return False


def _address_key(address: RepresentationAddress):
    return (
        address.kind.value,
        address.owner_id,
        address.element_id or "",
        address.field_path,
    )
