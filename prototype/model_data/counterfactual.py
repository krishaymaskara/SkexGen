"""Physical counterfactual evaluation pairs with preservation paths."""

from __future__ import annotations

import json
from pathlib import Path

from prototype.counterfactual_edits.identity import edit_family_id, edit_sample_id
from prototype.counterfactual_edits.locality import validate_locality
from prototype.counterfactual_edits.serialization import edit_sample_from_json
from prototype.representation.serialization import history_from_json

from .errors import ModelDataError
from .loader import load_physical_examples
from .records import AttributePath, CounterfactualExample


_PAIR_RECORD_FIELDS = {
    "edit_sample_id",
    "edit_family_id",
    "geometry_encoding",
    "base_source_family_id",
    "edited_source_family_id",
    "base_sample_id",
    "edited_sample_id",
    "relative_json_path",
    "schema_valid",
    "serialization_round_trip_valid",
    "kernel_status",
}
_COUNTERFACTUAL_MANIFEST_FIELDS = {
    "generator_version",
    "edit_pair_schema_version",
    "edit_canonicalization_version",
    "representation_schema_version",
    "representation_canonicalization_version",
    "feasibility_policy_version",
    "orientation_policy_version",
    "selection_policy_version",
    "normalized_configuration",
    "configuration_sha256",
    "generation_seed",
    "candidate_counts",
    "total_edit_family_count",
    "total_edit_sample_count",
    "total_physical_endpoint_count",
    "total_endpoint_sample_count",
    "families",
    "samples",
}
_EDIT_FAMILY_FIELDS = {
    "edit_family_id",
    "edit_type",
    "target",
    "before",
    "after",
    "base_source_family_id",
    "edited_source_family_id",
    "base_physical_source",
    "edited_physical_source",
    "operation_template",
    "history_depth",
    "primitive_family",
    "reference_plane",
    "history_max_sketch_extent",
    "sketch_extent_band",
    "locality",
    "schema_valid",
    "serialization_round_trip_valid",
    "both_endpoints_feasible",
    "kernel_status",
}
_EVALUATION_PARTITIONS = {
    "train",
    "validation",
    "test",
    "secondary_systematic_validation",
}


def load_counterfactual_examples(corpus_dir, split_name="iid"):
    root = Path(corpus_dir)
    physical = {
        item.physical_family_id: item
        for item in load_physical_examples(root, split_name)
    }
    manifest = _read(root / "counterfactual_manifest.json")
    unknown_manifest_fields = set(manifest) - _COUNTERFACTUAL_MANIFEST_FIELDS
    if unknown_manifest_fields:
        raise ModelDataError(
            "counterfactual_manifest",
            f"unknown fields {sorted(unknown_manifest_fields)!r}",
        )
    split = _read(root / "manifests" / f"{split_name}.json")
    if split.get("authoritative_assignment_unit") != "edit_family_id":
        raise ModelDataError(
            "counterfactual_split_unit",
            "counterfactual partition authority must be edit_family_id",
        )
    assignments = _edit_assignments(split.get("families"))
    split_samples = split.get("samples")
    if not isinstance(split_samples, list):
        raise ModelDataError("counterfactual_split", "samples must be a list")
    split_sample_ids = set()
    for item in split_samples:
        if not isinstance(item, dict):
            raise ModelDataError("counterfactual_split", "invalid sample assignment")
        unknown = set(item) - (_PAIR_RECORD_FIELDS | {"partition"})
        if unknown:
            raise ModelDataError(
                "counterfactual_split",
                f"unknown sample fields {sorted(unknown)!r}",
            )
        family_id = item.get("edit_family_id")
        current_sample_id = item.get("edit_sample_id")
        if not isinstance(current_sample_id, str) or current_sample_id in split_sample_ids:
            raise ModelDataError(
                "counterfactual_split", "duplicate or invalid edit sample assignment"
            )
        split_sample_ids.add(current_sample_id)
        if assignments.get(family_id) != item.get("partition"):
            raise ModelDataError(
                "counterfactual_leakage",
                "edit variants do not inherit the edit-family partition",
            )
    grouped = {}
    seen_ids = set()
    manifest_samples = manifest.get("samples")
    if not isinstance(manifest_samples, list):
        raise ModelDataError(
            "counterfactual_manifest", "samples must be a list"
        )
    for record in manifest_samples:
        _require_pair_record(record)
        current_edit_sample_id = record.get("edit_sample_id")
        if current_edit_sample_id in seen_ids:
            raise ModelDataError("duplicate_edit_sample", str(current_edit_sample_id))
        seen_ids.add(current_edit_sample_id)
        grouped.setdefault(record.get("edit_family_id"), []).append(record)
    if split_sample_ids != seen_ids:
        raise ModelDataError(
            "counterfactual_split_mismatch",
            "split must assign every edit sample exactly once",
        )
    _verify_manifest_families(manifest, grouped)
    if set(grouped) != set(assignments):
        raise ModelDataError(
            "counterfactual_split_mismatch",
            "edit-family manifest and split sets differ",
        )

    result = []
    for family_id in sorted(grouped):
        variants = {}
        parsed = {}
        for record in grouped[family_id]:
            encoding = record.get("geometry_encoding")
            if encoding in variants:
                raise ModelDataError("duplicate_edit_encoding", str(encoding))
            path = _safe(root, record.get("relative_json_path"))
            payload = path.read_text(encoding="utf-8")
            payload = payload[:-1] if payload.endswith("\n") else payload
            try:
                sample = edit_sample_from_json(payload)
            except Exception as exc:
                raise ModelDataError(
                    "counterfactual_deserialization",
                    f"{family_id}: {type(exc).__name__}",
                ) from exc
            if (
                sample.edit_sample_id != record.get("edit_sample_id")
                or sample.edit_family_id != family_id
                or sample.geometry_encoding != encoding
                or sample.base_source_family_id
                != record.get("base_source_family_id")
                or sample.edited_source_family_id
                != record.get("edited_source_family_id")
                or sample.base_sample_id != record.get("base_sample_id")
                or sample.edited_sample_id != record.get("edited_sample_id")
            ):
                raise ModelDataError(
                    "counterfactual_identity", f"{family_id} pair metadata disagrees"
                )
            expected_family_id = edit_family_id(
                sample.edit_type,
                sample.target,
                sample.base_source_family_id,
                sample.edited_source_family_id,
            )
            expected_sample_id = edit_sample_id(
                expected_family_id,
                sample.geometry_encoding,
                sample.base_sample_id,
                sample.edited_sample_id,
            )
            if family_id != expected_family_id or sample.edit_sample_id != expected_sample_id:
                raise ModelDataError(
                    "counterfactual_identity", f"{family_id} identity hash disagrees"
                )
            variants[encoding] = record
            parsed[encoding] = sample
        if set(variants) != {"continuous", "quantized"}:
            raise ModelDataError(
                "incomplete_edit_family", f"{family_id} lacks one encoding"
            )
        continuous = parsed["continuous"]
        quantized = parsed["quantized"]
        if (
            continuous.base_source_family_id
            != quantized.base_source_family_id
            or continuous.edited_source_family_id
            != quantized.edited_source_family_id
            or continuous.edit_type is not quantized.edit_type
            or continuous.target != quantized.target
            or continuous.locality != quantized.locality
        ):
            raise ModelDataError(
                "counterfactual_physical_disagreement",
                f"{family_id} variants disagree",
            )
        source = physical.get(continuous.base_source_family_id)
        target = physical.get(continuous.edited_source_family_id)
        if source is None or target is None:
            raise ModelDataError(
                "counterfactual_endpoint", f"{family_id} endpoint is absent"
            )
        for variant in parsed.values():
            if (
                variant.base_sample_id not in source.metadata.sample_ids
                or variant.edited_sample_id not in target.metadata.sample_ids
            ):
                raise ModelDataError(
                    "counterfactual_endpoint",
                    f"{family_id} endpoint sample is absent",
                )
        partition = assignments[family_id]
        if source.partition != partition or target.partition != partition:
            raise ModelDataError(
                "counterfactual_leakage",
                f"{family_id} endpoints cross partitions",
            )
        _verify_endpoint_assignments(
            split.get("endpoints"),
            family_id,
            partition,
            source.physical_family_id,
            target.physical_family_id,
        )
        try:
            validate_locality(
                history_from_json(source.canonical_reconstruction_json),
                history_from_json(target.canonical_reconstruction_json),
                continuous.locality,
                continuous.edit_type,
                continuous.target,
            )
        except Exception as exc:
            raise ModelDataError(
                "counterfactual_locality",
                f"{family_id} locality is stale or incomplete: {type(exc).__name__}",
            ) from exc
        changed_paths = tuple(
            _path(item) for item in continuous.locality.allowed_changed_paths
        )
        unchanged_paths = tuple(
            _path(item) for item in continuous.locality.expected_unchanged_paths
        )
        if (
            len(changed_paths) != len(set(changed_paths))
            or len(unchanged_paths) != len(set(unchanged_paths))
            or set(changed_paths).intersection(unchanged_paths)
        ):
            raise ModelDataError(
                "counterfactual_locality",
                f"{family_id} locality paths overlap or contain duplicates",
            )
        result.append(
            CounterfactualExample(
                source,
                target,
                family_id,
                continuous.edit_sample_id,
                continuous.edit_type.value,
                changed_paths,
                unchanged_paths,
                split_name,
                partition,
            )
        )
    return tuple(result)


def _edit_assignments(records):
    if not isinstance(records, list):
        raise ModelDataError("counterfactual_split", "families must be a list")
    assignments = {}
    for item in records:
        if not isinstance(item, dict):
            raise ModelDataError("counterfactual_split", "invalid family assignment")
        unknown = set(item) - (_EDIT_FAMILY_FIELDS | {"partition"})
        if unknown:
            raise ModelDataError(
                "counterfactual_split",
                f"unknown family fields {sorted(unknown)!r}",
            )
        family_id = item.get("edit_family_id")
        partition = item.get("partition")
        if not isinstance(family_id, str) or not isinstance(partition, str):
            raise ModelDataError("counterfactual_split", "invalid family assignment")
        if partition not in _EVALUATION_PARTITIONS:
            raise ModelDataError("counterfactual_split", "unknown partition")
        if family_id in assignments:
            raise ModelDataError(
                "duplicate_edit_family",
                f"{family_id} has more than one authoritative assignment",
            )
        assignments[family_id] = partition
    return assignments


def _verify_manifest_families(manifest, grouped):
    records = manifest.get("families")
    if not isinstance(records, list):
        raise ModelDataError("counterfactual_manifest", "families must be a list")
    observed = set()
    for item in records:
        if not isinstance(item, dict):
            raise ModelDataError("counterfactual_manifest", "invalid family record")
        unknown = set(item) - _EDIT_FAMILY_FIELDS
        if unknown:
            raise ModelDataError(
                "counterfactual_manifest",
                f"unknown family fields {sorted(unknown)!r}",
            )
        family_id = item.get("edit_family_id")
        if not isinstance(family_id, str) or family_id in observed:
            raise ModelDataError("duplicate_edit_family", str(family_id))
        observed.add(family_id)
    if observed != set(grouped):
        raise ModelDataError(
            "counterfactual_manifest", "family and sample sets differ"
        )
    for key, expected in (
        ("total_edit_family_count", len(grouped)),
        ("total_edit_sample_count", sum(len(items) for items in grouped.values())),
    ):
        if key in manifest and manifest[key] != expected:
            raise ModelDataError("counterfactual_count", f"{key} disagrees")


def _verify_endpoint_assignments(
    records, family_id, partition, source_family_id, target_family_id
):
    if not isinstance(records, list):
        raise ModelDataError("counterfactual_split", "endpoints must be a list")
    expected = {
        ("base", source_family_id),
        ("edited", target_family_id),
    }
    observed = {
        (item.get("role"), item.get("source_family_id"))
        for item in records
        if isinstance(item, dict)
        and item.get("edit_family_id") == family_id
        and item.get("partition") == partition
    }
    if observed != expected or sum(
        1
        for item in records
        if isinstance(item, dict) and item.get("edit_family_id") == family_id
    ) != 2:
        raise ModelDataError(
            "counterfactual_leakage",
            f"{family_id} endpoint assignments are incomplete or conflicting",
        )


def _path(value):
    return AttributePath(
        value.kind.value,
        value.owner_id,
        value.field_path,
        value.element_id,
    )


def _require_pair_record(record):
    if not isinstance(record, dict):
        raise ModelDataError("counterfactual_manifest", "pair record must be an object")
    unknown = set(record) - _PAIR_RECORD_FIELDS
    required = {
        "edit_sample_id",
        "edit_family_id",
        "geometry_encoding",
        "base_source_family_id",
        "edited_source_family_id",
        "base_sample_id",
        "edited_sample_id",
        "relative_json_path",
        "schema_valid",
        "serialization_round_trip_valid",
        "kernel_status",
    }
    if unknown or set(record) < required:
        raise ModelDataError(
            "counterfactual_manifest",
            f"pair record fields differ: unknown={sorted(unknown)!r}",
        )
    if record["geometry_encoding"] not in {"continuous", "quantized"}:
        raise ModelDataError(
            "counterfactual_manifest", "unknown pair geometry encoding"
        )
    if (
        record["schema_valid"] is not True
        or record["serialization_round_trip_valid"] is not True
        or record["kernel_status"] != "not_checked"
    ):
        raise ModelDataError(
            "counterfactual_manifest", "pair validity fields are inconsistent"
        )


def _safe(root, relative):
    if not isinstance(relative, str):
        raise ModelDataError("counterfactual_path", "path must be a string")
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != relative:
        raise ModelDataError("counterfactual_path", relative)
    target = root / path
    if not target.is_file():
        raise ModelDataError("counterfactual_path", f"missing {relative}")
    return target


def _read(path):
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_no_duplicates,
            parse_constant=_reject_nonfinite,
        )
    except ModelDataError:
        raise
    except Exception as exc:
        raise ModelDataError(
            "counterfactual_json", f"{path.name}: {type(exc).__name__}"
        ) from exc


def _no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ModelDataError("counterfactual_json", f"duplicate field {key!r}")
        result[key] = value
    return result


def _reject_nonfinite(value):
    raise ModelDataError("counterfactual_json", f"nonfinite value {value!r}")
