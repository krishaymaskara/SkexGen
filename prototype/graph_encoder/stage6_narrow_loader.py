"""Strict partition-scoped loader for future ADR-0014 execution.

This module never discovers a corpus manifest.  It accepts one independently
reviewed narrow index and an exact payload-only root, then reconstructs physical
examples with the same canonical identity primitives used by model-data.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from prototype.controlled_data.identity import (
    canonical_physical_source_bytes,
    sample_id,
    source_family_id,
)
from prototype.model_data.canonical import (
    canonical_nodes_and_edges,
    infer_metadata,
    reconstruction_target,
)
from prototype.model_data.records import FamilyMetadata, PhysicalExample
from prototype.representation.serialization import history_from_json, history_to_json

from .errors import GraphEncoderError
from .partitions import AUTHORITATIVE_FILE_SHA256
from .stage6_structure_only import (
    DEVELOPMENT_FAMILY_COUNT,
    PROTOCOL_VERSION,
    TEMPLATES,
    TRAIN_FAMILY_COUNT,
)


NARROW_INDEX_VERSION = "GE1-STAGE6-NARROW-INDEX-v1"
NARROW_LOADER_VERSION = "GE1-STAGE6-NARROW-LOADER-v1"
PARTITIONS = {
    "train": {
        "physical_partition": "train",
        "family_count": TRAIN_FAMILY_COUNT,
        "assignment_sha256": (
            "42d61d2224ae1a2110279f66d374516f8b5f220271407913147d8be390c66595"
        ),
    },
    "development": {
        "physical_partition": "validation",
        "family_count": DEVELOPMENT_FAMILY_COUNT,
        "assignment_sha256": (
            "9af48e34e0ae2e5fd266d2336b6adecdad543c75a2c470ea132b9c7a593afaf6"
        ),
    },
}
INDEX_FIELDS = {
    "schema_version", "protocol_version", "partition_identity",
    "parent_manifest_sha256", "assignment_identity", "assignment_sha256",
    "expected_family_count", "family_ids", "samples", "payload_allowlist",
}
SAMPLE_FIELDS = {
    "source_family_id", "sample_id", "geometry_encoding",
    "relative_payload_path", "payload_sha256", "operation_template",
    "primitive_family", "reference_plane", "history_depth",
}
FORBIDDEN_COMPONENTS = {
    "rr", "er", "iid", "test", "secondary_systematic_validation",
    "history_depth", "geometry_extrapolation", "checkpoints", "checkpoint",
    "model_artifact", "repaired_artifact", "preserved_feature",
}


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False)


def _loads(raw, label):
    try:
        return json.loads(
            raw,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError("nonfinite " + value)
            ),
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise GraphEncoderError("invalid_stage6_narrow_index", label) from exc


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest_lines(values):
    return hashlib.sha256(("\n".join(values) + "\n").encode("utf-8")).hexdigest()


def _safe_root(path, label):
    raw = Path(path)
    if raw.is_symlink() or not raw.is_dir() or raw.name != "payloads":
        raise GraphEncoderError("unsafe_stage6_narrow_root", label)
    if any(part.lower() in FORBIDDEN_COMPONENTS for part in raw.parts):
        raise GraphEncoderError("unauthorized_stage6_input", label)
    return raw.resolve()


def _safe_relative(relative):
    if not isinstance(relative, str) or not relative:
        raise GraphEncoderError("unsafe_stage6_narrow_path", repr(relative))
    path = Path(relative)
    if (
        path.is_absolute() or ".." in path.parts or path.as_posix() != relative
        or any(part.lower() in FORBIDDEN_COMPONENTS for part in path.parts)
    ):
        raise GraphEncoderError("unsafe_stage6_narrow_path", relative)
    return path


def validate_stage6_narrow_index(index_path, payload_root, partition_identity):
    """Validate exact index/root contents without invoking a corpus loader."""

    if partition_identity not in PARTITIONS:
        raise GraphEncoderError("invalid_stage6_narrow_partition", str(partition_identity))
    index = Path(index_path)
    if (
        index.is_symlink() or not index.is_file()
        or any(part.lower() in FORBIDDEN_COMPONENTS for part in index.parts)
    ):
        raise GraphEncoderError("invalid_stage6_narrow_index", "index is not regular")
    root = _safe_root(payload_root, partition_identity)
    raw = index.read_text(encoding="utf-8")
    document = _loads(raw, partition_identity)
    if raw != _canonical(document) + "\n" or not isinstance(document, dict):
        raise GraphEncoderError("invalid_stage6_narrow_index", "canonical JSON differs")
    expected = PARTITIONS[partition_identity]
    if (
        set(document) != INDEX_FIELDS
        or document.get("schema_version") != NARROW_INDEX_VERSION
        or document.get("protocol_version") != PROTOCOL_VERSION
        or document.get("partition_identity") != partition_identity
        or document.get("parent_manifest_sha256") != AUTHORITATIVE_FILE_SHA256
        or document.get("assignment_identity")
        != "operation_template:" + partition_identity
        or document.get("assignment_sha256") != expected["assignment_sha256"]
        or document.get("expected_family_count") != expected["family_count"]
    ):
        raise GraphEncoderError("invalid_stage6_narrow_index", "identity differs")
    family_ids = tuple(document.get("family_ids", ()))
    samples = tuple(document.get("samples", ()))
    allowlist = tuple(document.get("payload_allowlist", ()))
    if (
        family_ids != tuple(sorted(family_ids))
        or len(family_ids) != expected["family_count"]
        or len(set(family_ids)) != len(family_ids)
        or any(not isinstance(value, str) or not value for value in family_ids)
        or allowlist != tuple(sorted(allowlist))
        or len(allowlist) != 2 * expected["family_count"]
        or len(set(allowlist)) != len(allowlist)
    ):
        raise GraphEncoderError("invalid_stage6_narrow_index", "counts/order differ")
    grouped = {family_id: [] for family_id in family_ids}
    expected_hashes = {}
    expected_order = []
    for record in samples:
        if not isinstance(record, dict) or set(record) != SAMPLE_FIELDS:
            raise GraphEncoderError("invalid_stage6_narrow_index", "sample fields differ")
        family_id = record["source_family_id"]
        encoding = record["geometry_encoding"]
        relative = record["relative_payload_path"]
        digest = record["payload_sha256"]
        if (
            family_id not in grouped or encoding not in ("continuous", "quantized")
            or record["operation_template"] not in TEMPLATES
            or not all(isinstance(record[name], str) and record[name]
                       for name in ("sample_id", "primitive_family", "reference_plane"))
            or isinstance(record["history_depth"], bool)
            or not isinstance(record["history_depth"], int)
            or record["history_depth"] not in (1, 2)
            or not isinstance(digest, str) or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise GraphEncoderError("invalid_stage6_narrow_index", "sample differs")
        _safe_relative(relative)
        if relative in expected_hashes:
            raise GraphEncoderError("invalid_stage6_narrow_index", "duplicate payload")
        expected_hashes[relative] = digest
        expected_order.append((family_id, encoding, relative))
        grouped[family_id].append(record)
    if (
        tuple(expected_order) != tuple(sorted(expected_order))
        or tuple(sorted(expected_hashes)) != allowlist
        or any(len(rows) != 2 or {row["geometry_encoding"] for row in rows}
               != {"continuous", "quantized"} for rows in grouped.values())
    ):
        raise GraphEncoderError("invalid_stage6_narrow_index", "variant matrix differs")
    if any(path.is_symlink() for path in root.rglob("*")):
        raise GraphEncoderError("unsafe_stage6_narrow_path", "symlink exposed")
    observed_files = tuple(sorted(
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    ))
    if observed_files != allowlist:
        raise GraphEncoderError("unauthorized_stage6_input", "allowlist differs")
    observed_hashes = {}
    for relative in allowlist:
        target = root / _safe_relative(relative)
        if target.is_symlink() or not target.is_file():
            raise GraphEncoderError("unsafe_stage6_narrow_path", relative)
        observed_hashes[relative] = _sha(target)
        if observed_hashes[relative] != expected_hashes[relative]:
            raise GraphEncoderError("stage6_input_hash_mismatch", relative)
    evidence = {
        "loader_version": NARROW_LOADER_VERSION,
        "index_schema_version": NARROW_INDEX_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "partition_identity": partition_identity,
        "index_identity_sha256": _sha(index),
        "parent_manifest_sha256": AUTHORITATIVE_FILE_SHA256,
        "assignment_sha256": expected["assignment_sha256"],
        "family_count": len(family_ids),
        "expected_allowlist_sha256": _digest_lines(allowlist),
        "observed_allowlist_sha256": _digest_lines(observed_files),
        "expected_payload_digests_sha256": hashlib.sha256(
            _canonical(expected_hashes).encode("utf-8")
        ).hexdigest(),
        "observed_payload_digests_sha256": hashlib.sha256(
            _canonical(observed_hashes).encode("utf-8")
        ).hexdigest(),
        "expected_payload_sha256": dict(sorted(expected_hashes.items())),
        "observed_payload_sha256": dict(sorted(observed_hashes.items())),
        "root_read_only_required": True,
        "unexpected_file_count": 0,
        "verification_status": "pass",
    }
    return document, grouped, evidence


def _history_encoding(history):
    return {
        value.encoding.value
        for record in (
            *history.geometry.node_geometry,
            *history.geometry.sketch_element_geometry,
        )
        for value in vars(record.geometry).values()
        if hasattr(value, "encoding")
    }


def _load(index_path, payload_root, partition_identity):
    document, grouped, evidence = validate_stage6_narrow_index(
        index_path, payload_root, partition_identity
    )
    root = Path(payload_root).resolve()
    examples = []
    physical_partition = PARTITIONS[partition_identity]["physical_partition"]
    for family_id in document["family_ids"]:
        records = grouped[family_id]
        histories = {}
        physical_bytes = None
        declared_metadata = None
        for record in records:
            path = root / record["relative_payload_path"]
            payload = path.read_text(encoding="utf-8")
            payload = payload[:-1] if payload.endswith("\n") else payload
            try:
                history = history_from_json(payload)
                canonical = history_to_json(history)
            except Exception as exc:
                raise GraphEncoderError(
                    "invalid_stage6_narrow_payload", record["relative_payload_path"]
                ) from exc
            if canonical != payload:
                raise GraphEncoderError("invalid_stage6_narrow_payload", "noncanonical")
            if sample_id(history) != record["sample_id"] or source_family_id(history) != family_id:
                raise GraphEncoderError("invalid_stage6_narrow_payload", "identity differs")
            if _history_encoding(history) != {record["geometry_encoding"]}:
                raise GraphEncoderError("invalid_stage6_narrow_payload", "encoding differs")
            current_physical = canonical_physical_source_bytes(history)
            if physical_bytes is not None and current_physical != physical_bytes:
                raise GraphEncoderError("invalid_stage6_narrow_payload", "variants differ")
            physical_bytes = current_physical
            metadata = (
                record["operation_template"], record["primitive_family"],
                record["reference_plane"], record["history_depth"],
            )
            if declared_metadata is not None and metadata != declared_metadata:
                raise GraphEncoderError("invalid_stage6_narrow_payload", "metadata differs")
            declared_metadata = metadata
            histories[record["geometry_encoding"]] = history
        history = histories["continuous"]
        inferred = infer_metadata(history)
        if inferred != declared_metadata[:3] or len(history.structure.operation_sequence) != declared_metadata[3]:
            raise GraphEncoderError("invalid_stage6_narrow_payload", "inference differs")
        nodes, edges = canonical_nodes_and_edges(history)
        target = reconstruction_target(nodes, edges, history.structure.operation_sequence)
        examples.append(PhysicalExample(
            family_id,
            FamilyMetadata(
                inferred[0], inferred[1], inferred[2], declared_metadata[3],
                ("continuous", "quantized"),
                tuple(sorted(record["sample_id"] for record in records)),
            ),
            "operation_template", physical_partition,
            history.structure.operation_sequence, nodes, edges, target,
            history_to_json(history),
        ))
    return tuple(examples), evidence


def load_stage6_train(index_path, payload_root):
    return _load(index_path, payload_root, "train")


def load_stage6_development(index_path, payload_root):
    return _load(index_path, payload_root, "development")
