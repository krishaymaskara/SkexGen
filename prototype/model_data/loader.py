"""Strict collapse of encoded corpus variants into physical examples."""

from __future__ import annotations

import json
from pathlib import Path

from prototype.controlled_data.identity import (
    canonical_physical_source_bytes,
    sample_id,
    source_family_id,
)
from prototype.representation.serialization import history_from_json, history_to_json

from .canonical import (
    canonical_nodes_and_edges,
    infer_metadata,
    reconstruction_target,
)
from .errors import ModelDataError
from .records import FamilyMetadata, PhysicalExample


_PHYSICAL_METADATA_FIELDS = {
    "operation_template",
    "history_depth",
    "operation_sequence",
    "primitive_family",
    "boolean_modes",
    "operation_directions",
    "profile_axis_dependency",
    "body_dependency",
    "sketch_extents",
    "history_max_sketch_extent",
    "sketch_extent_band",
    "extrusion_distances",
    "revolution_angles",
    "reference_plane",
    "extent_order_relation",
    "direction_relation",
}
_SAMPLE_FIELDS = {
    "source_family_id",
    "sample_id",
    "geometry_encoding",
    "relative_json_path",
    "schema_valid",
    "serialization_round_trip_valid",
    "kernel_status",
    "endpoint_role_occurrences",
    *_PHYSICAL_METADATA_FIELDS,
}
_FAMILY_FIELDS = {
    "source_family_id",
    "selection_order",
    "sample_ids",
    "schema_valid",
    "serialization_round_trip_valid",
    "kernel_status",
    *_PHYSICAL_METADATA_FIELDS,
}
_CORPUS_FIELDS = {
    "generator_version",
    "feasibility_policy_version",
    "representation_schema_version",
    "canonicalization_version",
    "normalized_configuration",
    "configuration_sha256",
    "generation_seed",
    "candidate_source_family_count",
    "raw_candidate_source_family_count",
    "total_source_family_count",
    "total_sample_variant_count",
    "families",
    "samples",
}
_SPLIT_FIELDS = {
    "manifest_version",
    "split_policy_version",
    "corpus_configuration_sha256",
    "name",
    "definition",
    "authoritative_assignment_unit",
    "partition_counts",
    "coverage",
    "families",
    "samples",
    "endpoints",
}
_PARTITIONS = {
    "train",
    "validation",
    "test",
    "secondary_systematic_validation",
}
_SUPPORTED_MANIFEST_FILES = {
    "iid.json",
    "operation_template.json",
    "history_depth.json",
    "geometry_extrapolation.json",
    "endpoint_exclusions.json",
}


def load_physical_examples(corpus_dir, split_name="iid"):
    root = Path(corpus_dir)
    _validate_split_name(split_name)
    corpus = _load_json(root / "corpus_manifest.json")
    split = _load_json(root / "manifests" / f"{split_name}.json")
    _reject_unknown_fields(corpus, _CORPUS_FIELDS, "corpus manifest")
    _reject_unknown_fields(split, _SPLIT_FIELDS, "split manifest")
    samples = corpus.get("samples")
    if not isinstance(samples, list):
        raise ModelDataError("malformed_corpus", "corpus samples must be a list")
    for record in samples:
        _require_record(record)
    _verify_corpus_counts(corpus, samples)
    assignments = _split_assignments(split, samples)
    grouped = {}
    seen_sample_ids = set()
    seen_paths = set()
    for record in samples:
        current_sample_id = record["sample_id"]
        if current_sample_id in seen_sample_ids:
            raise ModelDataError("duplicate_sample", current_sample_id)
        seen_sample_ids.add(current_sample_id)
        relative = record["relative_json_path"]
        if relative in seen_paths:
            raise ModelDataError("duplicate_history_path", relative)
        seen_paths.add(relative)
        grouped.setdefault(record["source_family_id"], []).append(record)
    _verify_family_records(corpus.get("families"), grouped)
    declared_family_count = corpus.get("total_source_family_count")
    if declared_family_count is not None and declared_family_count != len(grouped):
        raise ModelDataError(
            "inconsistent_count", "source-family count disagrees with records"
        )
    if set(grouped) != set(assignments):
        raise ModelDataError(
            "split_corpus_mismatch",
            "split and corpus physical-family sets differ",
        )

    examples = []
    for family_id in sorted(grouped):
        variants = grouped[family_id]
        if len(variants) != 2:
            raise ModelDataError(
                "variant_count",
                "exactly two representation variants are required",
                family_id,
            )
        _verify_variant_metadata_consistency(variants, family_id)
        by_encoding = {}
        physical_bytes = None
        histories = {}
        identity_errors = []
        for record in variants:
            encoding = record["geometry_encoding"]
            if encoding in by_encoding:
                raise ModelDataError(
                    "duplicate_encoding", encoding, family_id
                )
            path = _safe_path(root, record["relative_json_path"])
            try:
                payload = path.read_text(encoding="utf-8")
                payload = payload[:-1] if payload.endswith("\n") else payload
                history = history_from_json(payload)
                canonical = history_to_json(history)
            except Exception as exc:
                raise ModelDataError(
                    "history_deserialization",
                    f"{record['relative_json_path']}: {type(exc).__name__}",
                    family_id,
                ) from exc
            if canonical != payload:
                raise ModelDataError(
                    "noncanonical_history", record["relative_json_path"], family_id
                )
            current_physical = canonical_physical_source_bytes(history)
            if physical_bytes is None:
                physical_bytes = current_physical
            elif current_physical != physical_bytes:
                raise ModelDataError(
                    "physical_disagreement",
                    "encoding variants decode to different physical semantics",
                    family_id,
                )
            if sample_id(history) != record["sample_id"]:
                identity_errors.append(("sample_identity", record["sample_id"]))
            if source_family_id(history) != family_id:
                identity_errors.append(("family_identity", family_id))
            _verify_history_encoding(history, encoding, family_id)
            by_encoding[encoding] = record
            histories[encoding] = history
        if set(by_encoding) != {"continuous", "quantized"}:
            raise ModelDataError(
                "incomplete_family",
                "exactly one continuous and one quantized variant are required",
                family_id,
            )
        if identity_errors:
            code, detail = identity_errors[0]
            raise ModelDataError(code, detail, family_id)
        history = histories["continuous"]
        template, primitive_family, reference_plane = infer_metadata(history)
        operation_sequence = history.structure.operation_sequence
        for record in variants:
            _verify_optional_metadata(
                record,
                template,
                primitive_family,
                reference_plane,
                len(operation_sequence),
                family_id,
            )
        nodes, edges = canonical_nodes_and_edges(history)
        target = reconstruction_target(nodes, edges, operation_sequence)
        examples.append(
            PhysicalExample(
                family_id,
                FamilyMetadata(
                    template,
                    primitive_family,
                    reference_plane,
                    len(operation_sequence),
                    ("continuous", "quantized"),
                    tuple(sorted(by_encoding[item]["sample_id"] for item in by_encoding)),
                ),
                split_name,
                assignments[family_id],
                operation_sequence,
                nodes,
                edges,
                target,
                history_to_json(history),
            )
        )
    _verify_corpus_closure(root, seen_paths)
    return tuple(examples)


def _split_assignments(split, corpus_samples):
    unit = split.get("authoritative_assignment_unit")
    result = {}
    if unit == "source_family_id":
        records = split.get("families")
        if not isinstance(records, list):
            raise ModelDataError("malformed_split", "families must be a list")
        for item in records:
            if not isinstance(item, dict):
                raise ModelDataError("malformed_split", "invalid family assignment")
            _reject_unknown_fields(
                item, _FAMILY_FIELDS | {"partition"}, "split family record"
            )
            _assign(result, item.get("source_family_id"), item.get("partition"))
        split_samples = split.get("samples")
        if not isinstance(split_samples, list):
            raise ModelDataError("malformed_split", "samples must be a list")
        expected_samples = {
            item["sample_id"]: item["source_family_id"] for item in corpus_samples
        }
        observed_samples = {}
        for item in split_samples:
            if not isinstance(item, dict):
                raise ModelDataError("malformed_split", "invalid sample assignment")
            _reject_unknown_fields(
                item, _SAMPLE_FIELDS | {"partition"}, "split sample record"
            )
            current_sample_id = item.get("sample_id")
            family_id = item.get("source_family_id")
            partition = item.get("partition")
            if (
                not isinstance(current_sample_id, str)
                or current_sample_id in observed_samples
                or expected_samples.get(current_sample_id) != family_id
            ):
                raise ModelDataError(
                    "split_sample_mismatch",
                    "split samples are duplicate, unknown, or assigned to another family",
                )
            observed_samples[current_sample_id] = family_id
            if result.get(family_id) != partition:
                raise ModelDataError("split_leakage", "encoding variants cross partitions")
        if observed_samples != expected_samples:
            raise ModelDataError(
                "incomplete_split",
                "split must assign every encoding variant exactly once",
            )
    elif unit == "edit_family_id":
        records = split.get("endpoints")
        if not isinstance(records, list):
            raise ModelDataError("malformed_split", "endpoints must be a list")
        for item in records:
            if not isinstance(item, dict):
                raise ModelDataError("malformed_split", "invalid endpoint assignment")
            _reject_unknown_fields(
                item,
                {"edit_family_id", "role", "source_family_id", "partition"},
                "split endpoint record",
            )
            _assign(result, item.get("source_family_id"), item.get("partition"))
    else:
        raise ModelDataError(
            "unsupported_split_unit", f"unsupported assignment unit {unit!r}"
        )
    return result


def _assign(result, family_id, partition):
    if (
        not isinstance(family_id, str)
        or not isinstance(partition, str)
        or partition not in _PARTITIONS
    ):
        raise ModelDataError("malformed_split", "invalid family assignment")
    if family_id in result:
        raise ModelDataError(
            "duplicate_split_family",
            f"{family_id} has more than one authoritative assignment",
        )
    result[family_id] = partition


def _require_record(record):
    required_strings = (
        "source_family_id",
        "sample_id",
        "geometry_encoding",
        "relative_json_path",
    )
    required = {
        *required_strings,
        "schema_valid",
        "serialization_round_trip_valid",
        "kernel_status",
    }
    if not isinstance(record, dict):
        raise ModelDataError("malformed_sample", "sample record must be an object")
    _reject_unknown_fields(record, _SAMPLE_FIELDS, "sample record")
    if not required.issubset(record) or any(
        not isinstance(record.get(key), str) or not record[key]
        for key in required_strings
    ):
        raise ModelDataError("malformed_sample", "sample record fields are invalid")
    if record["geometry_encoding"] not in {"continuous", "quantized"}:
        raise ModelDataError("malformed_sample", "unknown geometry encoding")
    for key in ("schema_valid", "serialization_round_trip_valid"):
        if key in record and record[key] is not True:
            raise ModelDataError("malformed_sample", f"{key} must be true")
    if "kernel_status" in record and record["kernel_status"] != "not_checked":
        raise ModelDataError("malformed_sample", "kernel_status must be not_checked")


def _verify_history_encoding(history, expected, family_id):
    encodings = {
        value.encoding.value
        for record in (
            *history.geometry.node_geometry,
            *history.geometry.sketch_element_geometry,
        )
        for value in vars(record.geometry).values()
        if hasattr(value, "encoding")
    }
    if encodings != {expected}:
        raise ModelDataError(
            "encoding_mismatch", f"expected {expected}, observed {encodings}", family_id
        )


def _verify_optional_metadata(
    record,
    template,
    primitive_family,
    reference_plane,
    history_depth,
    family_id,
):
    for key, expected in (
        ("operation_template", template),
        ("primitive_family", primitive_family),
        ("reference_plane", reference_plane),
        ("history_depth", history_depth),
    ):
        if key in record and record[key] != expected:
            raise ModelDataError(
                "metadata_mismatch", f"{key} disagrees with history", family_id
            )


def _verify_variant_metadata_consistency(records, family_id):
    declared = [
        {
            key: record[key]
            for key in sorted(_PHYSICAL_METADATA_FIELDS)
            if key in record
        }
        for record in records
    ]
    if any(item != declared[0] for item in declared[1:]):
        raise ModelDataError(
            "encoding_metadata_disagreement",
            "encoding variants declare different physical metadata",
            family_id,
        )


def _safe_path(root, relative):
    if not isinstance(relative, str) or not relative:
        raise ModelDataError("unsafe_path", repr(relative))
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != relative:
        raise ModelDataError("unsafe_path", relative)
    target = root / path
    if not target.is_file():
        raise ModelDataError("missing_history", relative)
    return target


def _validate_split_name(split_name):
    if (
        not isinstance(split_name, str)
        or not split_name
        or Path(split_name).name != split_name
        or split_name in {".", ".."}
    ):
        raise ModelDataError("unsafe_split_name", repr(split_name))


def _verify_corpus_counts(corpus, samples):
    sample_count = corpus.get("total_sample_variant_count")
    family_count = corpus.get("total_source_family_count")
    for name, value in (
        ("total_sample_variant_count", sample_count),
        ("total_source_family_count", family_count),
    ):
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value < 0
        ):
            raise ModelDataError("malformed_count", f"{name} must be a nonnegative integer")
    if sample_count is not None and sample_count != len(samples):
        raise ModelDataError("inconsistent_count", "sample count disagrees with records")


def _verify_family_records(records, grouped):
    if records is None:
        return
    if not isinstance(records, list):
        raise ModelDataError("malformed_corpus", "families must be a list")
    observed = {}
    for item in records:
        if not isinstance(item, dict):
            raise ModelDataError("malformed_corpus", "family record must be an object")
        _reject_unknown_fields(item, _FAMILY_FIELDS, "family record")
        family_id = item.get("source_family_id")
        if not isinstance(family_id, str) or family_id in observed:
            raise ModelDataError("duplicate_family", str(family_id))
        sample_ids = item.get("sample_ids")
        if sample_ids is not None and (
            not isinstance(sample_ids, list)
            or sorted(sample_ids)
            != sorted(record["sample_id"] for record in grouped.get(family_id, ()))
        ):
            raise ModelDataError("family_sample_mismatch", str(family_id))
        for key in _PHYSICAL_METADATA_FIELDS:
            if key in item and any(
                record.get(key) != item[key] for record in grouped.get(family_id, ())
            ):
                raise ModelDataError(
                    "family_metadata_mismatch", f"{family_id}: {key}"
                )
        for key in ("schema_valid", "serialization_round_trip_valid"):
            if key in item and item[key] is not True:
                raise ModelDataError("malformed_corpus", f"{key} must be true")
        if "kernel_status" in item and item["kernel_status"] != "not_checked":
            raise ModelDataError("malformed_corpus", "kernel_status must be not_checked")
        observed[family_id] = item
    if set(observed) != set(grouped):
        raise ModelDataError("family_record_mismatch", "family and sample sets differ")


def _verify_corpus_closure(root, history_paths):
    expected = {"corpus_manifest.json", *history_paths}
    counterfactual_path = root / "counterfactual_manifest.json"
    if counterfactual_path.is_file():
        counterfactual = _load_json(counterfactual_path)
        expected.add("counterfactual_manifest.json")
        records = counterfactual.get("samples")
        if not isinstance(records, list):
            raise ModelDataError(
                "malformed_counterfactual_manifest", "samples must be a list"
            )
        for record in records:
            if not isinstance(record, dict):
                raise ModelDataError(
                    "malformed_counterfactual_manifest", "sample must be an object"
                )
            relative = record.get("relative_json_path")
            _safe_path(root, relative)
            expected.add(relative)
    manifests = root / "manifests"
    if not manifests.is_dir():
        raise ModelDataError("missing_manifests", "manifests directory is absent")
    expected.update(
        path.relative_to(root).as_posix()
        for path in manifests.iterdir()
        if path.is_file()
        and path.name in _SUPPORTED_MANIFEST_FILES
    )
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    if actual != expected:
        raise ModelDataError(
            "corpus_not_closed",
            f"unexpected={sorted(actual - expected)!r}, "
            f"missing={sorted(expected - actual)!r}",
        )


def _reject_unknown_fields(value, allowed, label):
    if not isinstance(value, dict):
        raise ModelDataError("malformed_manifest", f"{label} must be an object")
    unknown = set(value) - allowed
    if unknown:
        raise ModelDataError(
            "unknown_manifest_field", f"{label}: {sorted(unknown)!r}"
        )


def _load_json(path):
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
            "json_loading", f"{path.name}: {type(exc).__name__}"
        ) from exc


def _no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ModelDataError("duplicate_json_field", repr(key))
        result[key] = value
    return result


def _reject_nonfinite(value):
    raise ModelDataError("nonfinite_json", repr(value))
