"""Governed, non-overwriting preparation of ADR-0014 narrow input packages."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re

from prototype.controlled_data.identity import sample_id, source_family_id
from prototype.model_data.canonical import infer_metadata
from prototype.representation.serialization import history_from_json, history_to_json

from .errors import GraphEncoderError
from .partitions import (
    AUTHORITATIVE_FILE_SHA256,
    AUTHORITATIVE_RELATIVE_FILE,
    _verify_authoritative_manifest,
    load_development,
    load_train,
)
from .stage6_narrow_loader import (
    NARROW_INDEX_VERSION,
    PARTITIONS,
    load_stage6_development,
    load_stage6_train,
)
from .stage6_structure_only import PROTOCOL_VERSION, TEMPLATES


BUILDER_VERSION = "GE1-STAGE6-NARROW-BUILDER-v1"
PREPARATION_VERSION = "GE1-STAGE6-NARROW-PREPARATION-v1"
RECEIPT_NAME = "stage6-preparation-receipt.json"
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
PARTITION_CHOICES = ("train", "development", "both")


def _canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    )


def _loads(raw, label):
    def no_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate field")
            result[key] = value
        return result

    try:
        return json.loads(
            raw,
            object_pairs_hook=no_duplicates,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError("nonfinite " + value)
            ),
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise GraphEncoderError(
            "invalid_stage6_narrow_preparation", label
        ) from exc


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write(path, payload):
    target = Path(path)
    if target.exists() or target.is_symlink():
        raise GraphEncoderError("unsafe_stage6_narrow_output", str(target))
    with target.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _job_id(value):
    value = str(value)
    if not value.isdigit() or int(value) <= 0:
        raise GraphEncoderError(
            "invalid_stage6_preparation_job_id", "positive decimal job ID required"
        )
    return value


def _safe_root(path, label):
    root = Path(path)
    if root.is_symlink() or not root.is_dir():
        raise GraphEncoderError("unsafe_stage6_narrow_source", label)
    resolved = root.resolve()
    if len(resolved.parts) < 3:
        raise GraphEncoderError("unsafe_stage6_narrow_source", label)
    return resolved


def _safe_output_parent(path, corpus_root):
    parent = Path(path)
    if parent.is_symlink() or not parent.is_dir():
        raise GraphEncoderError("unsafe_stage6_narrow_output", str(parent))
    resolved = parent.resolve()
    if len(resolved.parts) < 3 or resolved == corpus_root:
        raise GraphEncoderError("unsafe_stage6_narrow_output", str(parent))
    if corpus_root in resolved.parents or resolved in corpus_root.parents:
        raise GraphEncoderError(
            "unsafe_stage6_narrow_output", "source and output roots overlap"
        )
    return resolved


def _safe_relative(value):
    if not isinstance(value, str) or not value:
        raise GraphEncoderError("unsafe_stage6_narrow_source_path", repr(value))
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts or relative.as_posix() != value:
        raise GraphEncoderError("unsafe_stage6_narrow_source_path", value)
    return relative


def _read_manifest(root, name):
    path = root / name
    if path.is_symlink() or not path.is_file():
        raise GraphEncoderError("invalid_stage6_narrow_preparation", name)
    return _loads(path.read_text(encoding="utf-8"), name)


def _authoritative_examples(corpus_root, partition):
    """Run accepted authority/full-corpus validation, then expose one partition."""

    from prototype.model_data.errors import ModelDataError
    try:
        _verify_authoritative_manifest(corpus_root)
        examples = load_train(corpus_root) if partition == "train" else load_development(
            corpus_root
        )
    except ModelDataError as exc:
        raise GraphEncoderError(
            "invalid_stage6_narrow_preparation", "authoritative corpus validation failed"
        ) from exc
    corpus = _read_manifest(corpus_root, "corpus_manifest.json")
    records = corpus.get("samples")
    if not isinstance(records, list):
        raise GraphEncoderError(
            "invalid_stage6_narrow_preparation", "corpus samples differ"
        )
    by_sample = {}
    for record in records:
        if not isinstance(record, dict):
            raise GraphEncoderError(
                "invalid_stage6_narrow_preparation", "sample record differs"
            )
        current = record.get("sample_id")
        if not isinstance(current, str) or current in by_sample:
            raise GraphEncoderError(
                "invalid_stage6_narrow_preparation", "sample identity differs"
            )
        by_sample[current] = record
    selected = {}
    for example in examples:
        for current in example.metadata.sample_ids:
            if current not in by_sample:
                raise GraphEncoderError(
                    "invalid_stage6_narrow_preparation", "selected sample absent"
                )
            selected[current] = by_sample[current]
    return tuple(examples), selected


def _validated_source_payload(root, record, example):
    relative = _safe_relative(record.get("relative_json_path"))
    path = root / relative
    current = root
    unsafe = False
    for component in relative.parts:
        current = current / component
        unsafe = unsafe or current.is_symlink()
    if unsafe or not path.is_file():
        raise GraphEncoderError("unsafe_stage6_narrow_source_path", relative.as_posix())
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
        without_lf = text[:-1] if text.endswith("\n") else text
        history = history_from_json(without_lf)
    except Exception as exc:
        raise GraphEncoderError(
            "invalid_stage6_narrow_source_payload", relative.as_posix()
        ) from exc
    if text != history_to_json(history) + "\n":
        raise GraphEncoderError(
            "invalid_stage6_narrow_source_payload", "noncanonical JSON"
        )
    if (
        sample_id(history) != record.get("sample_id")
        or source_family_id(history) != example.physical_family_id
    ):
        raise GraphEncoderError(
            "invalid_stage6_narrow_source_payload", "identity differs"
        )
    inferred = infer_metadata(history)
    expected = (
        example.metadata.operation_template,
        example.metadata.primitive_family,
        example.metadata.reference_plane,
    )
    if inferred != expected or len(history.structure.operation_sequence) != example.metadata.history_depth:
        raise GraphEncoderError(
            "invalid_stage6_narrow_source_payload", "metadata differs"
        )
    encoding = record.get("geometry_encoding")
    if encoding not in ("continuous", "quantized"):
        raise GraphEncoderError(
            "invalid_stage6_narrow_source_payload", "encoding differs"
        )
    return raw, history, encoding


def _package_index(corpus_root, staging, partition, examples, source_records):
    policy = PARTITIONS[partition]
    if len(examples) != policy["family_count"]:
        raise GraphEncoderError(
            "invalid_stage6_narrow_preparation", "family count differs"
        )
    payload_root = staging / "payloads"
    payload_root.mkdir()
    samples = []
    allowlist = []
    family_ids = tuple(sorted(item.physical_family_id for item in examples))
    if len(set(family_ids)) != len(family_ids):
        raise GraphEncoderError(
            "invalid_stage6_narrow_preparation", "duplicate family"
        )
    by_family = {item.physical_family_id: item for item in examples}
    for family_id in family_ids:
        if not SAFE_ID.fullmatch(family_id):
            raise GraphEncoderError(
                "unsafe_stage6_narrow_source_path", "unsafe trusted family identity"
            )
        example = by_family[family_id]
        if example.metadata.operation_template not in TEMPLATES:
            raise GraphEncoderError(
                "protected_partition_access", "only E/R/EE/RE are exportable"
            )
        if len(example.metadata.sample_ids) != 2:
            raise GraphEncoderError(
                "invalid_stage6_narrow_preparation", "variant count differs"
            )
        physical = None
        encodings = set()
        for current_sample in sorted(example.metadata.sample_ids):
            if not SAFE_ID.fullmatch(current_sample) or current_sample not in source_records:
                raise GraphEncoderError(
                    "unsafe_stage6_narrow_source_path", "sample identity differs"
                )
            record = source_records[current_sample]
            raw, history, encoding = _validated_source_payload(
                corpus_root, record, example
            )
            from prototype.controlled_data.identity import canonical_physical_source_bytes
            current_physical = canonical_physical_source_bytes(history)
            if physical is not None and current_physical != physical:
                raise GraphEncoderError(
                    "invalid_stage6_narrow_source_payload", "physical variants differ"
                )
            physical = current_physical
            if encoding in encodings:
                raise GraphEncoderError(
                    "invalid_stage6_narrow_preparation", "duplicate encoding"
                )
            encodings.add(encoding)
            relative = "families/{}/{}-{}.json".format(
                family_id, encoding, current_sample
            )
            target = payload_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.is_symlink() or target.exists():
                raise GraphEncoderError("unsafe_stage6_narrow_output", relative)
            _write(target, raw)
            digest = hashlib.sha256(raw).hexdigest()
            allowlist.append(relative)
            samples.append({
                "source_family_id": family_id,
                "sample_id": current_sample,
                "geometry_encoding": encoding,
                "relative_payload_path": relative,
                "payload_sha256": digest,
                "operation_template": example.metadata.operation_template,
                "primitive_family": example.metadata.primitive_family,
                "reference_plane": example.metadata.reference_plane,
                "history_depth": example.metadata.history_depth,
            })
        if encodings != {"continuous", "quantized"}:
            raise GraphEncoderError(
                "invalid_stage6_narrow_preparation", "encoding matrix differs"
            )
    document = {
        "schema_version": NARROW_INDEX_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "partition_identity": partition,
        "parent_manifest_sha256": AUTHORITATIVE_FILE_SHA256,
        "assignment_identity": "operation_template:" + partition,
        "assignment_sha256": policy["assignment_sha256"],
        "expected_family_count": policy["family_count"],
        "family_ids": list(family_ids),
        "samples": sorted(samples, key=lambda row: (
            row["source_family_id"], row["geometry_encoding"],
            row["relative_payload_path"],
        )),
        "payload_allowlist": sorted(allowlist),
    }
    _write(staging / "index.json", (_canonical(document) + "\n").encode("utf-8"))
    loader = load_stage6_train if partition == "train" else load_stage6_development
    loaded, evidence = loader(staging / "index.json", payload_root)
    if len(loaded) != policy["family_count"] or evidence["verification_status"] != "pass":
        raise GraphEncoderError(
            "invalid_stage6_narrow_preparation", "independent loader rejected package"
        )
    return evidence


def _receipt(records, job_id):
    return {
        "schema_version": PREPARATION_VERSION,
        "builder_version": BUILDER_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "job_id": job_id,
        "source_manifest_sha256": AUTHORITATIVE_FILE_SHA256,
        "assignment_sha256": {
            name: PARTITIONS[name]["assignment_sha256"] for name in records
        },
        "packages": {
            name: {
                "index_sha256": evidence["index_identity_sha256"],
                "expected_allowlist_sha256": evidence["expected_allowlist_sha256"],
                "observed_allowlist_sha256": evidence["observed_allowlist_sha256"],
                "expected_payload_digests_sha256": evidence[
                    "expected_payload_digests_sha256"
                ],
                "observed_payload_digests_sha256": evidence[
                    "observed_payload_digests_sha256"
                ],
                "family_count": evidence["family_count"],
                "sample_count": evidence["family_count"] * 2,
                "verification_status": "pass",
            }
            for name, evidence in sorted(records.items())
        },
        "source_access": {
            "authoritative_manifest_accessed": True,
            "train_payload_accessed": "train" in records,
            "development_payload_accessed": "development" in records,
            "rr_accessed": False,
            "er_accessed": False,
            "iid_accessed": False,
            "history_depth_accessed": False,
            "geometry_extrapolation_accessed": False,
            "checkpoint_accessed": False,
            "model_artifact_accessed": False,
            "repaired_artifact_accessed": False,
            "preserved_feature_artifact_accessed": False,
        },
        "source_absolute_payload_paths_recorded": False,
        "verification_status": "pass",
    }


def build_stage6_narrow_packages(*, corpus_root, output_parent, job_id,
                                 expected_source_manifest_sha256, partition):
    """Prepare one or both packages and publish only after all checks pass."""

    job_id = _job_id(job_id)
    if partition not in PARTITION_CHOICES:
        raise GraphEncoderError(
            "invalid_stage6_narrow_partition", repr(partition)
        )
    if expected_source_manifest_sha256 != AUTHORITATIVE_FILE_SHA256:
        raise GraphEncoderError(
            "manifest_authority_failure", "expected source manifest differs"
        )
    source = _safe_root(corpus_root, "corpus root")
    manifest = source / AUTHORITATIVE_RELATIVE_FILE
    if manifest.is_symlink() or not manifest.is_file() or _sha(manifest) != AUTHORITATIVE_FILE_SHA256:
        raise GraphEncoderError(
            "manifest_authority_failure", "authoritative manifest SHA-256 mismatch"
        )
    parent = _safe_output_parent(output_parent, source)
    selected = ("train", "development") if partition == "both" else (partition,)
    receipt_name = RECEIPT_NAME if partition == "both" else (
        "stage6-{}-preparation-receipt.json".format(partition)
    )
    final_paths = {name: parent / ("stage6-" + name) for name in selected}
    staging_paths = {
        name: parent / ("stage6-{}.incomplete-{}".format(name, job_id))
        for name in selected
    }
    receipt_final = parent / receipt_name
    receipt_staging = parent / (receipt_name + ".incomplete-" + job_id)
    if receipt_final.exists() or receipt_final.is_symlink() or receipt_staging.exists() or receipt_staging.is_symlink():
        raise GraphEncoderError("unsafe_stage6_narrow_output", receipt_name)
    for name in selected:
        if final_paths[name].exists() or final_paths[name].is_symlink():
            raise GraphEncoderError("unsafe_stage6_narrow_output", str(final_paths[name]))
        if staging_paths[name].exists() or staging_paths[name].is_symlink():
            raise GraphEncoderError("unsafe_stage6_narrow_output", str(staging_paths[name]))
    evidence = {}
    for name in selected:
        staging_paths[name].mkdir()
        examples, records = _authoritative_examples(source, name)
        evidence[name] = _package_index(
            source, staging_paths[name], name, examples, records
        )
    receipt = _receipt(evidence, job_id)
    _write(receipt_staging, (_canonical(receipt) + "\n").encode("utf-8"))
    published = []
    try:
        for name in selected:
            os.replace(str(staging_paths[name]), str(final_paths[name]))
            published.append(name)
        os.replace(str(receipt_staging), str(receipt_final))
    except Exception:
        for name in reversed(published):
            if final_paths[name].exists() and not staging_paths[name].exists():
                os.replace(str(final_paths[name]), str(staging_paths[name]))
        raise
    return verify_preparation_receipt(parent, receipt_name=receipt_name)


def verify_preparation_receipt(output_parent, *, receipt_name=RECEIPT_NAME):
    parent = Path(output_parent)
    permitted_receipts = {
        RECEIPT_NAME,
        "stage6-train-preparation-receipt.json",
        "stage6-development-preparation-receipt.json",
    }
    if receipt_name not in permitted_receipts:
        raise GraphEncoderError(
            "invalid_stage6_narrow_preparation", "receipt name differs"
        )
    receipt_path = parent / receipt_name
    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise GraphEncoderError("invalid_stage6_narrow_preparation", "receipt absent")
    raw = receipt_path.read_text(encoding="utf-8")
    receipt = _loads(raw, "receipt")
    required = {
        "schema_version", "builder_version", "protocol_version", "job_id",
        "source_manifest_sha256", "assignment_sha256", "packages",
        "source_access", "source_absolute_payload_paths_recorded",
        "verification_status",
    }
    package_names = set(receipt.get("packages", {})) if isinstance(receipt, dict) else set()
    expected_names = {"train", "development"} if receipt_name == RECEIPT_NAME else {
        "train" if "train" in receipt_name else "development"
    }
    expected_assignments = {
        name: PARTITIONS[name]["assignment_sha256"] for name in expected_names
    }
    expected_access = {
        "authoritative_manifest_accessed": True,
        "train_payload_accessed": "train" in expected_names,
        "development_payload_accessed": "development" in expected_names,
        "rr_accessed": False,
        "er_accessed": False,
        "iid_accessed": False,
        "history_depth_accessed": False,
        "geometry_extrapolation_accessed": False,
        "checkpoint_accessed": False,
        "model_artifact_accessed": False,
        "repaired_artifact_accessed": False,
        "preserved_feature_artifact_accessed": False,
    }
    if (
        raw != _canonical(receipt) + "\n" or set(receipt) != required
        or receipt.get("schema_version") != PREPARATION_VERSION
        or receipt.get("builder_version") != BUILDER_VERSION
        or receipt.get("protocol_version") != PROTOCOL_VERSION
        or receipt.get("source_manifest_sha256") != AUTHORITATIVE_FILE_SHA256
        or package_names != expected_names
        or receipt.get("assignment_sha256") != expected_assignments
        or receipt.get("source_access") != expected_access
        or receipt.get("source_absolute_payload_paths_recorded") is not False
    ):
        raise GraphEncoderError("invalid_stage6_narrow_preparation", "receipt differs")
    _job_id(receipt.get("job_id"))
    for name, recorded in receipt.get("packages", {}).items():
        root = parent / ("stage6-" + name)
        loader = load_stage6_train if name == "train" else load_stage6_development
        _, evidence = loader(root / "index.json", root / "payloads")
        expected = {
            "index_sha256": evidence["index_identity_sha256"],
            "expected_allowlist_sha256": evidence["expected_allowlist_sha256"],
            "observed_allowlist_sha256": evidence["observed_allowlist_sha256"],
            "expected_payload_digests_sha256": evidence[
                "expected_payload_digests_sha256"
            ],
            "observed_payload_digests_sha256": evidence[
                "observed_payload_digests_sha256"
            ],
            "family_count": evidence["family_count"],
            "sample_count": evidence["family_count"] * 2,
            "verification_status": "pass",
        }
        if recorded != expected:
            raise GraphEncoderError(
                "invalid_stage6_narrow_preparation", name + " receipt differs"
            )
    if not receipt.get("packages") or receipt.get("verification_status") != "pass":
        raise GraphEncoderError("invalid_stage6_narrow_preparation", "receipt status")
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-root", required=True)
    parser.add_argument("--output-parent", required=True)
    parser.add_argument("--job-id", default=os.environ.get("SLURM_JOB_ID"))
    parser.add_argument("--expected-source-manifest-sha256", required=True)
    parser.add_argument("--partition", choices=PARTITION_CHOICES, required=True)
    args = parser.parse_args(argv)
    result = build_stage6_narrow_packages(
        corpus_root=args.corpus_root,
        output_parent=args.output_parent,
        job_id=args.job_id,
        expected_source_manifest_sha256=args.expected_source_manifest_sha256,
        partition=args.partition,
    )
    print(_canonical({
        "event": "stage6_narrow_preparation_completed",
        "receipt": result,
    }), flush=True)


if __name__ == "__main__":
    main()
