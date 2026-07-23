"""Validated edit-corpus assembly and atomic publication."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

from prototype.controlled_data.builders import build_history
from prototype.controlled_data.config import (
    CANONICALIZATION_VERSION,
    FEASIBILITY_POLICY_VERSION,
    REPRESENTATION_SCHEMA_VERSION,
)
from prototype.controlled_data.identity import sample_id, source_family_id
from prototype.controlled_data.factors import (
    ExtentBand,
    OperationTemplate,
    PhysicalSource,
    PrimitiveFamily,
    ReferencePlane,
)
from prototype.representation.model import (
    BooleanMode,
    Direction,
    GeometryEncoding,
)
from prototype.representation.serialization import history_from_json, history_to_json
from prototype.representation.validation import validate_history

from .candidates import candidate_counts, select_oriented_candidates
from .config import (
    EDIT_CANONICALIZATION_VERSION,
    EDIT_PAIR_SCHEMA_VERSION,
    GENERATOR_VERSION,
    ORIENTATION_POLICY_VERSION,
    SELECTION_POLICY_VERSION,
    CounterfactualConfig,
    EditConfigurationError,
)
from .edits import before_after, validate_edit_endpoints
from .identity import (
    edit_family_id,
    edit_sample_id,
    physical_source_descriptor,
)
from .locality import build_locality_ground_truth, validate_locality
from .model import EditFamily, EditSample, OrientedCandidate
from .serialization import edit_sample_from_json, edit_sample_to_json
from .splits import build_split_manifests


class EditGenerationError(RuntimeError):
    pass


def generate_corpus(
    output_dir: str | os.PathLike[str],
    config: CounterfactualConfig,
) -> dict[str, Any]:
    config.validate()
    final = Path(output_dir)
    if final.exists():
        raise EditConfigurationError(f"output directory already exists: {final}")
    if not final.parent.exists():
        raise EditConfigurationError("output parent directory does not exist")
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{final.name}.tmp-", dir=final.parent)
    )
    try:
        histories_dir = temporary / "histories"
        pairs_dir = temporary / "pairs"
        manifests_dir = temporary / "manifests"
        histories_dir.mkdir()
        pairs_dir.mkdir()
        manifests_dir.mkdir()

        selected = select_oriented_candidates(config)
        families = [_build_family(item, config) for item in selected]
        family_records = []
        edit_sample_records = []
        endpoint_records: dict[str, dict[str, Any]] = {}
        written_sample_ids: set[str] = set()
        written_edit_sample_ids: set[str] = set()

        for family in families:
            sample_records = []
            for encoding in (
                GeometryEncoding.CONTINUOUS,
                GeometryEncoding.QUANTIZED,
            ):
                base_history = build_history(family.base_source, encoding)
                edited_history = build_history(family.edited_source, encoding)
                _validate_endpoint_history(
                    base_history, family.base_source_family_id
                )
                _validate_endpoint_history(
                    edited_history, family.edited_source_family_id
                )
                validate_locality(
                    base_history,
                    edited_history,
                    family.locality,
                    family.edit_type,
                    family.target,
                )
                base_sample_id = sample_id(base_history)
                edited_sample_id = sample_id(edited_history)
                for role, history, source_id, current_sample_id, source in (
                    (
                        "base",
                        base_history,
                        family.base_source_family_id,
                        base_sample_id,
                        family.base_source,
                    ),
                    (
                        "edited",
                        edited_history,
                        family.edited_source_family_id,
                        edited_sample_id,
                        family.edited_source,
                    ),
                ):
                    relative_path = f"histories/{current_sample_id}.json"
                    payload = history_to_json(history)
                    if current_sample_id in written_sample_ids:
                        raise EditGenerationError(
                            f"endpoint sample ID collision {current_sample_id}"
                        )
                    _write_text(temporary / relative_path, payload + "\n")
                    written_sample_ids.add(current_sample_id)
                    endpoint_records[current_sample_id] = {
                        "source_family_id": source_id,
                        "sample_id": current_sample_id,
                        "geometry_encoding": encoding.value,
                        "operation_template": source.operation_template.value,
                        "primitive_family": source.primitive_family.value,
                        "relative_json_path": relative_path,
                        "schema_valid": True,
                        "serialization_round_trip_valid": True,
                        "kernel_status": "not_checked",
                        "endpoint_role_occurrences": [role],
                    }

                current_edit_sample_id = edit_sample_id(
                    family.edit_family_id,
                    encoding.value,
                    base_sample_id,
                    edited_sample_id,
                )
                if current_edit_sample_id in written_edit_sample_ids:
                    raise EditGenerationError(
                        f"duplicate edit sample ID {current_edit_sample_id}"
                    )
                written_edit_sample_ids.add(current_edit_sample_id)
                edit_sample = EditSample(
                    current_edit_sample_id,
                    family.edit_family_id,
                    encoding.value,
                    family.base_source_family_id,
                    family.edited_source_family_id,
                    base_sample_id,
                    edited_sample_id,
                    family.edit_type,
                    family.target,
                    family.before,
                    family.after,
                    family.locality,
                )
                pair_payload = edit_sample_to_json(edit_sample)
                restored = edit_sample_from_json(pair_payload)
                if edit_sample_to_json(restored) != pair_payload:
                    raise EditGenerationError("edit-sample round trip changed bytes")
                relative_pair_path = f"pairs/{current_edit_sample_id}.json"
                _write_text(temporary / relative_pair_path, pair_payload + "\n")
                record = {
                    "edit_sample_id": current_edit_sample_id,
                    "edit_family_id": family.edit_family_id,
                    "geometry_encoding": encoding.value,
                    "base_source_family_id": family.base_source_family_id,
                    "edited_source_family_id": family.edited_source_family_id,
                    "base_sample_id": base_sample_id,
                    "edited_sample_id": edited_sample_id,
                    "relative_json_path": relative_pair_path,
                    "schema_valid": True,
                    "serialization_round_trip_valid": True,
                    "kernel_status": "not_checked",
                }
                sample_records.append(record)
                edit_sample_records.append(record)

            family_record = _family_record(family)
            family_record["samples"] = sorted(
                sample_records, key=lambda item: item["edit_sample_id"]
            )
            family_records.append(family_record)

        if len(family_records) != config.num_edit_families:
            raise EditGenerationError("edit-family count differs from request")
        if len(edit_sample_records) != 2 * config.num_edit_families:
            raise EditGenerationError("each edit family must have two variants")
        if len(endpoint_records) != 4 * config.num_edit_families:
            raise EditGenerationError("endpoint-disjoint corpus must have four samples per pair")

        split_manifests, exclusions = build_split_manifests(family_records, config)
        counts = candidate_counts(config)
        counterfactual_manifest = {
            "generator_version": GENERATOR_VERSION,
            "edit_pair_schema_version": EDIT_PAIR_SCHEMA_VERSION,
            "edit_canonicalization_version": EDIT_CANONICALIZATION_VERSION,
            "representation_schema_version": REPRESENTATION_SCHEMA_VERSION,
            "representation_canonicalization_version": CANONICALIZATION_VERSION,
            "feasibility_policy_version": FEASIBILITY_POLICY_VERSION,
            "orientation_policy_version": ORIENTATION_POLICY_VERSION,
            "selection_policy_version": SELECTION_POLICY_VERSION,
            "normalized_configuration": config.normalized(),
            "configuration_sha256": config.sha256(),
            "generation_seed": config.seed,
            "candidate_counts": counts,
            "total_edit_family_count": len(family_records),
            "total_edit_sample_count": len(edit_sample_records),
            "total_physical_endpoint_count": 2 * len(family_records),
            "total_endpoint_sample_count": len(endpoint_records),
            "families": sorted(
                (
                    {key: value for key, value in family.items() if key != "samples"}
                    for family in family_records
                ),
                key=lambda item: item["edit_family_id"],
            ),
            "samples": sorted(
                edit_sample_records, key=lambda item: item["edit_sample_id"]
            ),
        }
        corpus_manifest = {
            "generator_version": GENERATOR_VERSION,
            "representation_schema_version": REPRESENTATION_SCHEMA_VERSION,
            "canonicalization_version": CANONICALIZATION_VERSION,
            "configuration_sha256": config.sha256(),
            "generation_seed": config.seed,
            "total_source_family_count": 2 * len(family_records),
            "total_sample_variant_count": len(endpoint_records),
            "samples": sorted(
                endpoint_records.values(), key=lambda item: item["sample_id"]
            ),
        }
        _write_json(
            temporary / "counterfactual_manifest.json",
            counterfactual_manifest,
        )
        _write_json(temporary / "corpus_manifest.json", corpus_manifest)
        for name, manifest in split_manifests.items():
            _write_json(manifests_dir / f"{name}.json", manifest)
        _write_json(manifests_dir / "endpoint_exclusions.json", exclusions)
        _verify_tree(
            temporary,
            counterfactual_manifest,
            corpus_manifest,
            split_manifests,
            exclusions,
            config,
        )
        os.rename(temporary, final)
        return counterfactual_manifest
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _build_family(
    item: OrientedCandidate, config: CounterfactualConfig
) -> EditFamily:
    candidate = item.identified.candidate
    base = item.base_source
    edited = item.edited_source
    validate_edit_endpoints(
        candidate.edit_type,
        candidate.target,
        base,
        edited,
        config.factor_config,
    )
    base_history = build_history(base, GeometryEncoding.CONTINUOUS)
    edited_history = build_history(edited, GeometryEncoding.CONTINUOUS)
    base_source_id = source_family_id(base_history)
    edited_source_id = source_family_id(edited_history)
    if {
        base_source_id,
        edited_source_id,
    } != {
        item.identified.lower_source_family_id,
        item.identified.higher_source_family_id,
    }:
        raise EditGenerationError("selected endpoint identities changed")
    locality = build_locality_ground_truth(
        base_history, edited_history, candidate.edit_type, candidate.target
    )
    before, after = before_after(
        candidate.edit_type, base, edited, candidate.target.operation_index
    )
    family_id = edit_family_id(
        candidate.edit_type,
        candidate.target,
        base_source_id,
        edited_source_id,
    )
    return EditFamily(
        base,
        edited,
        base_source_id,
        edited_source_id,
        candidate.edit_type,
        candidate.target,
        before,
        after,
        locality,
        family_id,
    )


def _family_record(family: EditFamily) -> dict[str, Any]:
    return {
        "edit_family_id": family.edit_family_id,
        "edit_type": family.edit_type.value,
        "target": family.target.to_dict(),
        "before": family.before,
        "after": family.after,
        "base_source_family_id": family.base_source_family_id,
        "edited_source_family_id": family.edited_source_family_id,
        "base_physical_source": physical_source_descriptor(family.base_source),
        "edited_physical_source": physical_source_descriptor(family.edited_source),
        "operation_template": family.base_source.operation_template.value,
        "history_depth": family.base_source.history_depth,
        "primitive_family": family.base_source.primitive_family.value,
        "reference_plane": family.base_source.reference_plane.value,
        "history_max_sketch_extent": max(family.base_source.sketch_extents),
        "sketch_extent_band": family.base_source.extent_band.value,
        "locality": family.locality.to_dict(),
        "schema_valid": True,
        "serialization_round_trip_valid": True,
        "both_endpoints_feasible": True,
        "kernel_status": "not_checked",
    }


def _validate_endpoint_history(history, expected_source_id):
    validate_history(history)
    payload = history_to_json(history)
    restored = history_from_json(payload)
    if history_to_json(restored) != payload:
        raise EditGenerationError("endpoint serialization round trip changed bytes")
    if source_family_id(history) != expected_source_id:
        raise EditGenerationError("endpoint source-family identity mismatch")


def _write_json(path: Path, value: Any) -> None:
    _write_text(
        path,
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
    )


def _write_text(path: Path, payload: str) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _verify_tree(root, counterfactual, corpus, splits, exclusions, config):
    checks = {
        root / "counterfactual_manifest.json": counterfactual,
        root / "corpus_manifest.json": corpus,
        root / "manifests" / "endpoint_exclusions.json": exclusions,
    }
    checks.update(
        {
            root / "manifests" / f"{name}.json": manifest
            for name, manifest in splits.items()
        }
    )
    for path, expected in checks.items():
        if _strict_json_file(path) != expected:
            raise EditGenerationError(f"written manifest failed verification: {path.name}")

    history_paths = _unique_references(corpus["samples"], "relative_json_path", "history")
    pair_paths = _unique_references(
        counterfactual["samples"], "relative_json_path", "pair"
    )
    fixed_paths = {
        "counterfactual_manifest.json",
        "corpus_manifest.json",
        "manifests/endpoint_exclusions.json",
        *{f"manifests/{name}.json" for name in splits},
    }
    expected_files = fixed_paths | set(history_paths) | set(pair_paths)
    actual_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    if actual_files != expected_files:
        raise EditGenerationError(
            "published file set differs: "
            f"missing={sorted(expected_files - actual_files)!r}, "
            f"unexpected={sorted(actual_files - expected_files)!r}"
        )

    families = _family_lookup(counterfactual, config)
    source_lookup = {}
    for family_id, item in families.items():
        for role in ("base", "edited"):
            source = item[f"{role}_source"]
            source_id = item[f"{role}_source_family_id"]
            previous = source_lookup.setdefault(source_id, source)
            if previous != source:
                raise EditGenerationError(
                    f"source-family descriptor collision in {family_id}"
                )
    if len(source_lookup) != 2 * len(families):
        raise EditGenerationError("physical endpoints are not globally disjoint")

    endpoint_by_sample = {}
    endpoint_by_source_encoding = {}
    histories = {}
    for endpoint in corpus["samples"]:
        current_sample_id = endpoint["sample_id"]
        if current_sample_id in endpoint_by_sample:
            raise EditGenerationError("duplicate endpoint sample manifest record")
        source_id = endpoint["source_family_id"]
        encoding = _geometry_encoding(endpoint["geometry_encoding"])
        source = source_lookup.get(source_id)
        if source is None:
            raise EditGenerationError("endpoint source ID has no physical descriptor")
        if endpoint["relative_json_path"] != f"histories/{current_sample_id}.json":
            raise EditGenerationError("endpoint history path does not match sample ID")
        key = (source_id, encoding.value)
        if key in endpoint_by_source_encoding:
            raise EditGenerationError("duplicate source/encoding endpoint manifest record")
        path = root / _safe_relative_path(endpoint["relative_json_path"])
        payload = path.read_text(encoding="utf-8")
        payload = payload[:-1] if payload.endswith("\n") else payload
        history = history_from_json(payload)
        canonical = history_to_json(history)
        expected_history = build_history(source, encoding)
        if (
            canonical != payload
            or canonical != history_to_json(expected_history)
            or sample_id(history) != current_sample_id
            or source_family_id(history) != source_id
            or endpoint["operation_template"] != source.operation_template.value
            or endpoint["primitive_family"] != source.primitive_family.value
        ):
            raise EditGenerationError("written endpoint failed complete verification")
        _validate_endpoint_history(history, source_id)
        endpoint_by_sample[current_sample_id] = endpoint
        endpoint_by_source_encoding[key] = endpoint
        histories[current_sample_id] = history

    if len(endpoint_by_source_encoding) != 2 * len(source_lookup):
        raise EditGenerationError(
            "every physical endpoint must have continuous and quantized variants"
        )

    seen_edit_ids = set()
    for record in counterfactual["samples"]:
        path = root / _safe_relative_path(record["relative_json_path"])
        payload = path.read_text(encoding="utf-8")
        payload = payload[:-1] if payload.endswith("\n") else payload
        restored = edit_sample_from_json(payload)
        if edit_sample_to_json(restored) != payload:
            raise EditGenerationError("written edit sample changed canonical bytes")
        if restored.edit_sample_id in seen_edit_ids:
            raise EditGenerationError("duplicate edit sample manifest record")
        seen_edit_ids.add(restored.edit_sample_id)
        family = families.get(restored.edit_family_id)
        if family is None:
            raise EditGenerationError("edit sample references an unknown edit family")
        encoding = _geometry_encoding(restored.geometry_encoding)
        base_endpoint = endpoint_by_source_encoding.get(
            (restored.base_source_family_id, encoding.value)
        )
        edited_endpoint = endpoint_by_source_encoding.get(
            (restored.edited_source_family_id, encoding.value)
        )
        if base_endpoint is None or edited_endpoint is None:
            raise EditGenerationError("edit sample references an unknown endpoint")
        expected_family_id = edit_family_id(
            family["edit_type"],
            family["target"],
            family["base_source_family_id"],
            family["edited_source_family_id"],
        )
        expected_sample_id = edit_sample_id(
            expected_family_id,
            encoding.value,
            base_endpoint["sample_id"],
            edited_endpoint["sample_id"],
        )
        if record["relative_json_path"] != f"pairs/{expected_sample_id}.json":
            raise EditGenerationError("pair path does not match edit sample ID")
        expected_record = {
            "edit_sample_id": expected_sample_id,
            "edit_family_id": expected_family_id,
            "geometry_encoding": encoding.value,
            "base_source_family_id": family["base_source_family_id"],
            "edited_source_family_id": family["edited_source_family_id"],
            "base_sample_id": base_endpoint["sample_id"],
            "edited_sample_id": edited_endpoint["sample_id"],
            "relative_json_path": record["relative_json_path"],
            "schema_valid": True,
            "serialization_round_trip_valid": True,
            "kernel_status": "not_checked",
        }
        if record != expected_record:
            raise EditGenerationError("pair manifest record is internally inconsistent")
        if (
            restored.edit_sample_id != expected_sample_id
            or restored.edit_family_id != expected_family_id
            or restored.base_source_family_id != family["base_source_family_id"]
            or restored.edited_source_family_id != family["edited_source_family_id"]
            or restored.base_sample_id != base_endpoint["sample_id"]
            or restored.edited_sample_id != edited_endpoint["sample_id"]
            or restored.edit_type is not family["edit_type"]
            or restored.target != family["target"]
            or restored.before != family["before"]
            or restored.after != family["after"]
            or restored.locality.to_dict() != family["locality"].to_dict()
        ):
            raise EditGenerationError("written edit sample metadata is inconsistent")
        validate_edit_endpoints(
            family["edit_type"],
            family["target"],
            family["base_source"],
            family["edited_source"],
            config.factor_config,
        )
        validate_locality(
            histories[base_endpoint["sample_id"]],
            histories[edited_endpoint["sample_id"]],
            restored.locality,
            restored.edit_type,
            restored.target,
        )

    if len(seen_edit_ids) != 2 * len(families):
        raise EditGenerationError(
            "every edit family must have continuous and quantized pair samples"
        )


def _unique_references(records, field, label):
    result = []
    seen = set()
    for record in records:
        value = record.get(field)
        relative = _safe_relative_path(value).as_posix()
        if relative in seen:
            raise EditGenerationError(f"duplicate {label} file reference {relative!r}")
        seen.add(relative)
        result.append(relative)
    return tuple(result)


def _safe_relative_path(value):
    if not isinstance(value, str) or not value:
        raise EditGenerationError("manifest path must be a nonempty string")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        raise EditGenerationError(f"unsafe relative path {value!r}")
    return path


def _strict_json_file(path):
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_json_fields,
            parse_constant=_reject_nonfinite_json,
        )
    except EditGenerationError:
        raise
    except Exception as exc:
        raise EditGenerationError(
            f"cannot strictly parse {path.name}: {type(exc).__name__}"
        ) from exc


def _reject_duplicate_json_fields(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise EditGenerationError(f"duplicate JSON field {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_json(value):
    raise EditGenerationError(f"nonfinite JSON value {value!r}")


def _family_lookup(counterfactual, config):
    result = {}
    for raw in counterfactual["families"]:
        family_id = raw["edit_family_id"]
        if family_id in result:
            raise EditGenerationError("duplicate edit family manifest record")
        edit_type = _edit_type(raw["edit_type"])
        target = _edit_target(raw["target"])
        base = _physical_source(raw["base_physical_source"])
        edited = _physical_source(raw["edited_physical_source"])
        validate_edit_endpoints(edit_type, target, base, edited, config.factor_config)
        base_id = source_family_id(build_history(base, GeometryEncoding.CONTINUOUS))
        edited_id = source_family_id(
            build_history(edited, GeometryEncoding.CONTINUOUS)
        )
        expected_id = edit_family_id(edit_type, target, base_id, edited_id)
        before, after = before_after(edit_type, base, edited, target.operation_index)
        locality = build_locality_ground_truth(
            build_history(base, GeometryEncoding.CONTINUOUS),
            build_history(edited, GeometryEncoding.CONTINUOUS),
            edit_type,
            target,
        )
        if (
            raw["base_source_family_id"] != base_id
            or raw["edited_source_family_id"] != edited_id
            or family_id != expected_id
            or raw["before"] != before
            or raw["after"] != after
            or raw["locality"] != locality.to_dict()
            or raw["operation_template"] != base.operation_template.value
            or raw["history_depth"] != base.history_depth
            or raw["primitive_family"] != base.primitive_family.value
            or raw["reference_plane"] != base.reference_plane.value
            or raw["history_max_sketch_extent"] != max(base.sketch_extents)
            or raw["sketch_extent_band"] != base.extent_band.value
            or raw["schema_valid"] is not True
            or raw["serialization_round_trip_valid"] is not True
            or raw["both_endpoints_feasible"] is not True
            or raw["kernel_status"] != "not_checked"
        ):
            raise EditGenerationError("edit family manifest is internally inconsistent")
        result[family_id] = {
            "edit_type": edit_type,
            "target": target,
            "base_source": base,
            "edited_source": edited,
            "base_source_family_id": base_id,
            "edited_source_family_id": edited_id,
            "before": before,
            "after": after,
            "locality": locality,
        }
    return result


def _physical_source(value):
    if not isinstance(value, dict) or set(value) != {
        "operation_template",
        "primitive_family",
        "reference_plane",
        "sketch_extents",
        "directions",
        "later_boolean_mode",
        "operation_parameters",
        "extent_band",
    }:
        raise EditGenerationError("physical source descriptor fields differ")
    try:
        return PhysicalSource(
            OperationTemplate(value["operation_template"]),
            PrimitiveFamily(value["primitive_family"]),
            ReferencePlane(value["reference_plane"]),
            tuple(value["sketch_extents"]),
            tuple(Direction(item) for item in value["directions"]),
            (
                BooleanMode(value["later_boolean_mode"])
                if value["later_boolean_mode"] is not None
                else None
            ),
            tuple(value["operation_parameters"]),
            ExtentBand(value["extent_band"]),
        )
    except Exception as exc:
        raise EditGenerationError("invalid physical source descriptor") from exc


def _geometry_encoding(value):
    try:
        return GeometryEncoding(value)
    except Exception as exc:
        raise EditGenerationError(f"invalid geometry encoding {value!r}") from exc


def _edit_type(value):
    from .model import EditType

    try:
        return EditType(value)
    except Exception as exc:
        raise EditGenerationError(f"invalid edit type {value!r}") from exc


def _edit_target(value):
    from .model import EditTarget

    if not isinstance(value, dict) or set(value) != {
        "operation_index",
        "node_id",
        "changed_field",
    }:
        raise EditGenerationError("invalid edit target fields")
    return EditTarget(
        value["operation_index"],
        value["node_id"],
        value["changed_field"],
    )
