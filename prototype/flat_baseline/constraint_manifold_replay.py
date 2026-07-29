"""Read-only diagnostic replay of frozen Phase B raw predictions."""

from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import sys
import tempfile
from types import SimpleNamespace

from prototype.model_data.geometry import (
    GEOMETRY_CHANNEL_SCALES,
    GEOMETRY_WIDTH,
    MAX_PRIMITIVES,
    PRIMITIVE_SLOT_WIDTH,
)
from prototype.model_data.vocab import (
    NODE_TYPES,
    PRIMITIVE_TYPES,
    REFERENCE_PLANES,
)

from .artifact_manifest import (
    EVALUATION_ARTIFACTS,
    MANIFEST_ENTRY_COUNT as SOURCE_MANIFEST_ENTRY_COUNT,
    REPORT_NAME,
    WORKFLOW_EVIDENCE_ARTIFACTS,
)
from .conversion import (
    FAILURE_CODE_ORDER,
    STAGE_ORDER,
    validate_and_convert_raw_prediction,
)
from .evaluate_length_conditioned import (
    EvaluationError,
    _atomic_no_replace,
    _fsync_directory,
    _json_safe,
    _raw_from_json,
    _stage_two_prediction,
)


PATHS = ("teacher_forced", "predicted_history")
ARMS = (
    "baseline",
    "semantic_constants",
    "model_category_projection",
    "oracle_category_diagnostic",
)
ORACLE_ARM = "oracle_category_diagnostic"
EXPECTED_FAMILY_COUNT = 68
EXPECTED_RAW_PATH_COUNT = 136
EXPECTED_REPLAY_RECORD_COUNT = 544
EXPECTED_SLURM_JOB_ID = "3329040"
EXPECTED_CHECKPOINT_SHA256 = (
    "282988af00a2dc9a53e14ceb270537d35"
    "f85d5339dc989634f88af5f77931267"
)
EXPECTED_EVALUATION_COMMIT = (
    "ce4abca8f450745a8832c0189aaae4a7d8263ec0"
)
EXPECTED_EVALUATION_SCHEMA_VERSION = 2
VALIDATOR_SOURCE_PATHS = (
    "prototype/flat_baseline/conversion.py",
    "prototype/flat_baseline/evaluate_length_conditioned.py",
)
OUTPUT_ARTIFACTS = (
    "replay_records.jsonl",
    "family_transitions.csv",
    "aggregate_summary.json",
    "projection_failures.csv",
    "run_metadata.json",
)
OUTPUT_MANIFEST = "sha256-manifest.txt"
PREREGISTRATION = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "specifications"
    / "flat_baseline_phase_b_constraint_manifold_replay.md"
)
PROFILE_PATTERNS = {
    "circle": ("circle", "<none>", "<none>", "<none>"),
    "rectangle_lines": ("line", "line", "line", "line"),
    "capsule_line_arc": ("arc", "arc", "line", "line"),
}
PATTERN_TO_PROFILE = {
    pattern: family for family, pattern in PROFILE_PATTERNS.items()
}
PLANE_FRAMES = {
    "XY": (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
    "XZ": (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
    "YZ": (0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
}
TRANSITION_COLUMNS = (
    "family_id",
    "path",
    "arm",
    "operation_template",
    "profile_family",
    "baseline_controlled_domain_valid",
    "controlled_domain_valid",
    "invalid_to_valid",
    "baseline_failure_combination",
    "failure_combination",
    "primary_failure_code",
    "supported_profile_family_eligible",
    "supported_sketch_count",
    "sketch_count",
    "projection_attempt_count",
    "projection_success_count",
    "projection_unavailable_count",
)
PROJECTION_FAILURE_COLUMNS = (
    "family_id",
    "path",
    "arm",
    "node_position",
    "transformation_kind",
    "field",
    "decoded_profile_pattern",
    "requested_profile_family",
    "reason",
)


class ReplayError(ValueError):
    """A replay input, transformation, or publication violated its contract."""

    def __init__(self, code, detail):
        self.code = code
        self.detail = detail
        super().__init__("{}: {}".format(code, detail))


@dataclass(frozen=True)
class OracleCategories:
    reference_plane: str
    primitive_family: str


@dataclass(frozen=True)
class BundleEvidence:
    bundle_path: str
    manifest_sha256: str
    entry_sha256: dict
    entry_stats: dict
    hidden_entries: tuple


@dataclass(frozen=True)
class ValidatorEvidence:
    aggregate_sha256: str
    evaluation_commit: str
    per_file_sha256: dict


def run_replay(bundle, output):
    """Validate one immutable bundle, replay four arms, and publish once."""

    source = Path(os.path.abspath(os.fspath(bundle)))
    destination = Path(os.path.abspath(os.fspath(output)))
    evidence = validate_source_manifest(source)
    validator_evidence = validate_validator_source()
    loaded = _load_bundle(source)
    artifacts = make_replay_artifacts(
        loaded["raw_records"],
        loaded["example_records"],
        loaded["run_metadata"],
        evidence,
        scheduler_job_id=loaded["scheduler_job_id"],
        validator_evidence=validator_evidence,
    )
    final_evidence = validate_source_manifest(source)
    if final_evidence != evidence:
        raise ReplayError(
            "source_bundle_changed",
            "source bundle changed after initial manifest validation",
        )
    publication = _publish_artifacts(destination, artifacts)
    return {
        "output": str(destination),
        "publication_backend": publication["backend"],
        "replay_record_count": EXPECTED_REPLAY_RECORD_COUNT,
        "manifest_entry_count": len(OUTPUT_ARTIFACTS),
    }


def validate_source_manifest(bundle):
    """Verify all 15 stable source paths before any scientific parsing."""

    root = Path(bundle)
    if root.is_symlink() or not root.is_dir():
        raise ReplayError(
            "invalid_source_bundle",
            "bundle must be a non-symlink directory",
        )
    manifest = root / OUTPUT_MANIFEST
    if manifest.is_symlink() or not manifest.is_file():
        raise ReplayError(
            "invalid_source_manifest",
            "source manifest must be a regular file",
        )
    expected = _source_relative_paths()
    public = {
        item.name for item in root.iterdir() if not item.name.startswith(".")
    }
    if public != {"evaluation", "workflow-evidence", REPORT_NAME, OUTPUT_MANIFEST}:
        raise ReplayError(
            "source_public_set_mismatch",
            "stable source namespace entries differ",
        )
    _require_exact_regular_files(root / "evaluation", EVALUATION_ARTIFACTS)
    _require_exact_regular_files(
        root / "workflow-evidence", WORKFLOW_EVIDENCE_ARTIFACTS
    )
    report = root / REPORT_NAME
    if report.is_symlink() or not report.is_file():
        raise ReplayError(
            "invalid_source_report",
            "stable Gate A-D report must be a regular file",
        )
    content = manifest.read_bytes()
    lines = content.splitlines()
    if len(lines) != SOURCE_MANIFEST_ENTRY_COUNT:
        raise ReplayError(
            "source_manifest_count",
            "source manifest must contain exactly 15 entries",
        )
    entries = {}
    stats = {}
    for raw_line in lines:
        try:
            digest_bytes, path_bytes = raw_line.split(b"  ", 1)
            digest = digest_bytes.decode("ascii")
            archived_path = path_bytes.decode("utf-8")
        except (ValueError, UnicodeError) as exc:
            raise ReplayError(
                "invalid_source_manifest",
                "manifest line encoding differs",
            ) from exc
        if (
            len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ReplayError(
                "invalid_source_manifest",
                "manifest digest differs",
            )
        relative = _stable_relative_path(archived_path, expected)
        if relative in entries:
            raise ReplayError(
                "duplicate_source_manifest_path",
                relative,
            )
        path = root / PurePosixPath(relative)
        before = path.stat()
        if path.is_symlink() or not stat.S_ISREG(before.st_mode):
            raise ReplayError(
                "invalid_source_artifact",
                relative,
            )
        actual = _file_sha256(path)
        after = path.stat()
        fingerprint = _stat_fingerprint(before)
        if fingerprint != _stat_fingerprint(after):
            raise ReplayError(
                "source_artifact_changed",
                relative,
            )
        if actual != digest:
            raise ReplayError(
                "source_hash_mismatch",
                relative,
            )
        entries[relative] = digest
        stats[relative] = fingerprint
    if set(entries) != expected:
        raise ReplayError(
            "source_manifest_set_mismatch",
            "source manifest stable paths differ",
        )
    return BundleEvidence(
        str(root),
        hashlib.sha256(content).hexdigest(),
        dict(sorted(entries.items())),
        dict(sorted(stats.items())),
        tuple(sorted(
            item.name for item in root.iterdir()
            if item.name.startswith(".")
        )),
    )


def validate_validator_source():
    """Pin the unchanged validator implementation to the evaluation commit."""

    repository = Path(__file__).resolve().parents[2]
    per_file = {}
    differences = []
    for relative in VALIDATOR_SOURCE_PATHS:
        current = _file_sha256(repository / relative)
        committed = hashlib.sha256(
            _git_bytes(
                repository,
                ("show", "{}:{}".format(EXPECTED_EVALUATION_COMMIT, relative)),
            )
        ).hexdigest()
        per_file[relative] = current
        if current != committed:
            differences.append(
                "{} expected={} actual={}".format(relative, committed, current)
            )
    if differences:
        raise ReplayError(
            "validator_source_mismatch",
            "; ".join(differences),
        )
    aggregate = hashlib.sha256(
        _json_text(dict(sorted(per_file.items()))).encode("utf-8")
    ).hexdigest()
    return ValidatorEvidence(
        aggregate,
        EXPECTED_EVALUATION_COMMIT,
        dict(sorted(per_file.items())),
    )


def make_replay_artifacts(
    raw_records,
    examples,
    metadata,
    evidence,
    *,
    scheduler_job_id=EXPECTED_SLURM_JOB_ID,
    validator_evidence=None,
):
    """Return deterministic replay artifact bytes without publishing them."""

    if validator_evidence is None:
        validator_evidence = validate_validator_source()
    max_operations = _validate_loaded_contract(
        raw_records,
        examples,
        metadata,
        scheduler_job_id=scheduler_job_id,
    )
    original_by_family = {
        item["family_id"]: item for item in raw_records
    }
    records = []
    projection_failures = []
    for example in examples:
        family_id = example["family_id"]
        raw_record = original_by_family[family_id]
        for path in PATHS:
            original = _raw_from_json(raw_record[path])
            baseline_input = _stage_two_prediction(original, path)
            baseline = validate_and_convert_raw_prediction(
                baseline_input,
                max_operations=max_operations,
            )
            expected = example["stored_conversions"][path]
            if _json_safe(baseline) != expected:
                raise ReplayError(
                    "baseline_conversion_mismatch",
                    "{}.{}".format(family_id, path),
                )
            baseline_codes = baseline.controlled_domain.failure_codes
            for arm in ARMS:
                oracle = (
                    _oracle_categories(example, arm)
                    if arm == ORACLE_ARM else None
                )
                try:
                    transformed, changes, unavailable = transform_prediction(
                        original,
                        arm,
                        oracle=oracle,
                    )
                except ReplayError as exc:
                    raise ReplayError(
                        exc.code,
                        "{}.{}.{}: {}".format(
                            family_id, path, arm, exc.detail
                        ),
                    ) from exc
                result = validate_and_convert_raw_prediction(
                    _stage_two_prediction(transformed, path),
                    max_operations=max_operations,
                )
                if arm == "baseline" and _json_safe(result) != expected:
                    raise ReplayError(
                        "baseline_replay_mismatch",
                        "{}.{}".format(family_id, path),
                    )
                eligibility = _profile_eligibility(transformed)
                failures = tuple(
                    dict(item, family_id=family_id, path=path, arm=arm)
                    for item in unavailable
                )
                projection_failures.extend(failures)
                profile_unavailable = sum(
                    item["transformation_kind"] == "profile_projection"
                    for item in unavailable
                )
                record = {
                    "arm": arm,
                    "baseline_controlled_domain_valid": (
                        baseline.controlled_domain.valid
                    ),
                    "baseline_failure_codes": list(baseline_codes),
                    "conversion": _json_safe(result),
                    "family_id": family_id,
                    "invalid_to_valid": (
                        not baseline.controlled_domain.valid
                        and result.controlled_domain.valid
                    ),
                    "operation_template": example["operation_template"],
                    "oracle_assisted": arm == ORACLE_ARM,
                    "path": path,
                    "profile_family": example["primitive_family"],
                    "profile_projection": {
                        "attempt_count": sum(
                            item["kind"].endswith("_profile")
                            for item in changes
                        ) + profile_unavailable,
                        "success_count": sum(
                            item["kind"].endswith("_profile")
                            and not item["is_no_op"]
                            for item in changes
                        ),
                        "on_manifold_no_op_count": sum(
                            item["kind"].endswith("_profile")
                            and item["is_no_op"]
                            for item in changes
                        ),
                        "unavailable_count": profile_unavailable,
                    },
                    "supported_profile_eligibility": eligibility,
                    "transformations": changes,
                    "raw_completion": transformed is not None,
                }
                records.append(record)
    if len(records) != EXPECTED_REPLAY_RECORD_COUNT:
        raise ReplayError(
            "replay_record_count",
            "exactly 544 replay rows are required",
        )
    _validate_record_order(records)
    aggregate = _aggregate_records(records)
    metadata_record = _run_metadata(
        metadata,
        evidence,
        records,
        scheduler_job_id=scheduler_job_id,
        validator_evidence=validator_evidence,
    )
    artifacts = {
        "replay_records.jsonl": _json_lines(records),
        "family_transitions.csv": _transition_csv(records),
        "aggregate_summary.json": _json_document(aggregate),
        "projection_failures.csv": _projection_failure_csv(
            projection_failures
        ),
        "run_metadata.json": _json_document(metadata_record),
    }
    return artifacts


def transform_prediction(raw, arm, *, oracle=None):
    """Apply one arm without accepting family metadata outside the oracle arm."""

    if arm not in ARMS:
        raise ReplayError("unknown_replay_arm", str(arm))
    if arm == ORACLE_ARM:
        if not isinstance(oracle, OracleCategories):
            raise ReplayError(
                "oracle_metadata_required",
                "Arm 4 requires restricted oracle categories",
            )
    elif oracle is not None:
        raise ReplayError(
            "oracle_metadata_leakage",
            "oracle categories are forbidden outside Arm 4",
        )
    if raw is None:
        return None, [], []
    if arm == "baseline":
        return raw, [], []
    transformed = copy.deepcopy(raw)
    nodes = list(transformed.raw_nodes)
    changes = []
    unavailable = []
    for index, node in enumerate(nodes):
        token = _node_token(node)
        current = node
        if arm == ORACLE_ARM and token == "reference_plane":
            current, category_change = _set_plane_category(
                current, oracle.reference_plane
            )
            changes.append(category_change)
        if arm == ORACLE_ARM and token == "sketch":
            current, category_changes = _set_profile_category(
                current, oracle.primitive_family
            )
            changes.extend(category_changes)
        if token == "reference_plane":
            current, change, failure = _canonical_plane(current)
            if change is not None:
                changes.append(change)
            if failure is not None:
                unavailable.append(failure)
        elif token == "axis":
            current, change = _canonical_axis(current)
            changes.append(change)
        if (
            token == "sketch"
            and arm in ("model_category_projection", ORACLE_ARM)
        ):
            current, change, failure = _project_sketch(current)
            if change is not None:
                changes.append(change)
            if failure is not None:
                unavailable.append(failure)
        nodes[index] = current
    transformed.raw_nodes = tuple(nodes)
    return transformed, changes, unavailable


def _load_bundle(bundle):
    evaluation = bundle / "evaluation"
    raw_records = _strict_json_lines(evaluation / "raw_predictions.jsonl")
    examples_raw = _strict_json_lines(evaluation / "examples.jsonl")
    metadata = _strict_json(evaluation / "run_metadata.json")
    scheduler_job_id = _scheduler_job_id(
        bundle / "workflow-evidence" / "scheduler.txt"
    )
    examples = []
    for item in examples_raw:
        try:
            authoritative = item["authoritative_metadata"]
            examples.append({
                "family_id": item["family_id"],
                "operation_template": authoritative["operation_template"],
                "primitive_family": authoritative["primitive_family"],
                "reference_plane": authoritative["reference_plane"],
                "stored_conversions": {
                    path: item[path]["conversion"] for path in PATHS
                },
            })
        except (KeyError, TypeError) as exc:
            raise ReplayError(
                "invalid_example_artifact",
                "required replay fields are absent",
            ) from exc
    return {
        "raw_records": raw_records,
        "example_records": examples,
        "run_metadata": metadata,
        "scheduler_job_id": scheduler_job_id,
    }


def _validate_loaded_contract(
    raw_records, examples, metadata, *, scheduler_job_id
):
    if (
        len(raw_records) != EXPECTED_FAMILY_COUNT
        or len(examples) != EXPECTED_FAMILY_COUNT
    ):
        raise ReplayError(
            "source_family_count",
            "exactly 68 family records are required",
        )
    raw_ids = tuple(item.get("family_id") for item in raw_records)
    example_ids = tuple(item.get("family_id") for item in examples)
    if (
        raw_ids != tuple(sorted(raw_ids))
        or raw_ids != example_ids
        or len(set(raw_ids)) != EXPECTED_FAMILY_COUNT
    ):
        raise ReplayError(
            "source_family_order",
            "family IDs must be unique, aligned, and lexicographically ordered",
        )
    if any(set(item) != {"family_id", *PATHS} for item in raw_records):
        raise ReplayError(
            "invalid_raw_artifact",
            "raw record schema differs",
        )
    selected = tuple(metadata.get("selected_family_ids", ()))
    authoritative = tuple(
        metadata.get("authoritative_validation_family_ids", ())
    )
    payload = metadata.get("payload_access", {})
    if (
        selected != raw_ids
        or authoritative != raw_ids
        or metadata.get("selected_family_count") != EXPECTED_FAMILY_COUNT
        or metadata.get("partition") != "validation"
        or metadata.get("test_partition_evaluated") is not False
        or payload.get("train_family_records_loaded") != 0
        or payload.get("validation_family_records_loaded")
        != EXPECTED_FAMILY_COUNT
        or payload.get("test_family_records_loaded") != 0
    ):
        raise ReplayError(
            "source_metadata_contract",
            "frozen validation-only metadata differs",
        )
    identity_checks = (
        (
            "checkpoint_sha256",
            metadata.get("checkpoint_sha256"),
            EXPECTED_CHECKPOINT_SHA256,
        ),
        (
            "repository_commit",
            metadata.get("repository_commit"),
            EXPECTED_EVALUATION_COMMIT,
        ),
        (
            "repaired_full_contract",
            metadata.get("repaired_full_contract"),
            True,
        ),
        (
            "evaluation_schema_version",
            metadata.get("evaluation_schema_version"),
            EXPECTED_EVALUATION_SCHEMA_VERSION,
        ),
        ("slurm_job_id", scheduler_job_id, EXPECTED_SLURM_JOB_ID),
    )
    differences = [
        "{} expected={!r} actual={!r}".format(name, expected, actual)
        for name, actual, expected in identity_checks
        if actual != expected
    ]
    if metadata.get("repaired_full_contract") is not True:
        differences.append(
            "repaired_full_contract must be the JSON boolean true"
        )
    schema = metadata.get("evaluation_schema_version")
    if isinstance(schema, bool) or not isinstance(schema, int):
        differences.append(
            "evaluation_schema_version must be the JSON integer 2"
        )
    if differences:
        raise ReplayError(
            "source_bundle_identity",
            "; ".join(differences),
        )
    if len(raw_records) * len(PATHS) != EXPECTED_RAW_PATH_COUNT:
        raise ReplayError(
            "source_raw_path_count",
            "exactly 136 raw path records are required",
        )
    configuration = metadata.get("model_configuration", {})
    max_operations = configuration.get("max_operations")
    if (
        isinstance(max_operations, bool)
        or not isinstance(max_operations, int)
        or max_operations < 1
    ):
        raise ReplayError(
            "source_max_operations",
            "frozen max_operations is absent",
        )
    return max_operations


def _oracle_categories(example, arm):
    if arm != ORACLE_ARM:
        raise ReplayError(
            "oracle_metadata_leakage",
            "authoritative categories are inaccessible outside Arm 4",
        )
    plane = example["reference_plane"]
    family = example["primitive_family"]
    if plane not in PLANE_FRAMES or family not in PROFILE_PATTERNS:
        raise ReplayError(
            "invalid_oracle_metadata",
            example["family_id"],
        )
    return OracleCategories(plane, family)


def _set_plane_category(node, plane):
    categorical = list(node.categorical_ids)
    if len(categorical) <= 3:
        raise ReplayError(
            "invalid_raw_node_width",
            "reference-plane categorical_ids is too short",
        )
    original = categorical[3]
    categorical[3] = REFERENCE_PLANES.id(plane)
    changed = _replace_namespace(
        node, categorical_ids=tuple(categorical)
    )
    return changed, _categorical_change(
        node.position,
        "oracle_reference_plane_category",
        "categorical_ids",
        [3],
        [original],
        [categorical[3]],
        "authoritative_family_category",
    )


def _set_profile_category(node, family):
    pattern = PROFILE_PATTERNS[family]
    categorical = list(node.categorical_ids)
    if len(categorical) < 8:
        raise ReplayError(
            "invalid_raw_node_width",
            "sketch categorical_ids is too short",
        )
    original_categories = list(categorical[4:8])
    categorical[4:8] = [PRIMITIVE_TYPES.id(item) for item in pattern]
    old_mask = list(node.derived_geometry_mask)
    if len(old_mask) < GEOMETRY_WIDTH:
        raise ReplayError(
            "invalid_raw_node_width",
            "sketch derived_geometry_mask is too short",
        )
    expected_mask = _derived_geometry_mask("sketch", categorical)
    new_mask = list(old_mask)
    new_mask[9:33] = expected_mask[9:33]
    changed = _replace_namespace(
        node,
        categorical_ids=tuple(categorical),
        derived_geometry_mask=tuple(new_mask),
    )
    category_change = _categorical_change(
        node.position,
        "oracle_profile_category",
        "categorical_ids",
        [4, 5, 6, 7],
        original_categories,
        categorical[4:8],
        "authoritative_family_category",
    )
    mask_change = _categorical_change(
        node.position,
        "oracle_geometry_mask",
        "derived_geometry_mask",
        list(range(9, 33)),
        old_mask[9:33],
        new_mask[9:33],
        "category_derived_applicability",
    )
    return changed, [category_change, mask_change]


def _canonical_plane(node):
    try:
        plane = REFERENCE_PLANES.tokens[node.categorical_ids[3]]
    except (IndexError, TypeError):
        return node, None, _semantic_failure(
            node, "reference_plane", "invalid_decoded_plane_category"
        )
    frame = PLANE_FRAMES.get(plane)
    if frame is None:
        return node, None, _semantic_failure(
            node, "reference_plane", "invalid_decoded_plane_category"
        )
    changed, change = _replace_geometry(
        node,
        tuple(range(9)),
        frame,
        "reference_plane",
        "normalized_geometry",
        "model_decoded_plane_canonical_frame",
    )
    return changed, change, None


def _canonical_axis(node):
    return _replace_geometry(
        node,
        (33, 34, 35, 36),
        (0.0, 0.0, 0.0, 1.0),
        "axis",
        "normalized_geometry",
        "controlled_axis_constant",
    )


def _project_sketch(node):
    try:
        pattern = tuple(
            PRIMITIVE_TYPES.tokens[item]
            for item in node.categorical_ids[4:8]
        )
    except (IndexError, TypeError):
        return node, None, _projection_failure(
            node, (), "", "invalid_decoded_profile_category"
        )
    family = PATTERN_TO_PROFILE.get(pattern)
    if family is None:
        return node, None, _projection_failure(
            node,
            pattern,
            "",
            "unsupported_model_profile_pattern",
        )
    if family == "circle":
        return _project_circle(node, pattern)
    projected = (
        _rectangle_projection(node.normalized_geometry)
        if family == "rectangle_lines"
        else _capsule_projection(node.normalized_geometry)
    )
    if isinstance(projected, str):
        return node, None, _projection_failure(
            node, pattern, family, projected
        )
    channels, values, method = projected
    changed, change = _replace_geometry(
        node,
        channels,
        values,
        family.replace("_lines", "").replace("_line_arc", "") + "_profile",
        "normalized_geometry",
        method,
    )
    return changed, change, None


def _project_circle(node, pattern):
    channels = (9, 10, 11)
    try:
        values = tuple(float(node.normalized_geometry[index]) for index in channels)
    except IndexError:
        return node, None, _projection_failure(
            node, pattern, "circle", "malformed_geometry_width"
        )
    except (TypeError, ValueError, OverflowError):
        return node, None, _projection_failure(
            node, pattern, "circle", "nonfinite_or_nonnumeric_observation"
        )
    if not all(math.isfinite(item) for item in values):
        return node, None, _projection_failure(
            node, pattern, "circle", "nonfinite_or_nonnumeric_observation"
        )
    physical_radius = values[2] * GEOMETRY_CHANNEL_SCALES[11]
    if physical_radius <= 0.0:
        return node, None, _projection_failure(
            node, pattern, "circle", "nonpositive_radius_has_no_nearest_strict_projection"
        )
    changed, change = _replace_geometry(
        node,
        channels,
        values,
        "circle_profile",
        "normalized_geometry",
        "identity_on_positive_circle_manifold",
    )
    return changed, change, None


def _rectangle_projection(geometry):
    points = (
        ((9, 10), (-0.5, -0.25)),
        ((11, 12), (0.5, -0.25)),
        ((15, 16), (0.5, -0.25)),
        ((17, 18), (0.5, 0.25)),
        ((21, 22), (0.5, 0.25)),
        ((23, 24), (-0.5, 0.25)),
        ((27, 28), (-0.5, 0.25)),
        ((29, 30), (-0.5, -0.25)),
    )
    fitted = _fit_center_extent(geometry, points)
    if isinstance(fitted, str):
        return fitted
    cx, cy, extent = fitted
    p0 = (cx - extent / 2.0, cy - extent / 4.0)
    p1 = (cx + extent / 2.0, cy - extent / 4.0)
    p2 = (cx + extent / 2.0, cy + extent / 4.0)
    p3 = (cx - extent / 2.0, cy + extent / 4.0)
    channel_points = (
        ((9, 10), p0), ((11, 12), p1),
        ((15, 16), p1), ((17, 18), p2),
        ((21, 22), p2), ((23, 24), p3),
        ((27, 28), p3), ((29, 30), p0),
    )
    return _normalized_projection(
        channel_points,
        "euclidean_least_squares_center_extent_rectangle",
    )


def _capsule_projection(geometry):
    points = (
        ((9, 10), (0.25, -0.25)),
        ((11, 12), (0.5, 0.0)),
        ((13, 14), (0.25, 0.25)),
        ((15, 16), (-0.25, 0.25)),
        ((17, 18), (-0.5, 0.0)),
        ((19, 20), (-0.25, -0.25)),
        ((21, 22), (-0.25, -0.25)),
        ((23, 24), (0.25, -0.25)),
        ((27, 28), (0.25, 0.25)),
        ((29, 30), (-0.25, 0.25)),
    )
    fitted = _fit_center_extent(geometry, points)
    if isinstance(fitted, str):
        return fitted
    cx, cy, extent = fitted
    lower_left = (cx - extent / 4.0, cy - extent / 4.0)
    lower_right = (cx + extent / 4.0, cy - extent / 4.0)
    upper_right = (cx + extent / 4.0, cy + extent / 4.0)
    upper_left = (cx - extent / 4.0, cy + extent / 4.0)
    right_mid = (cx + extent / 2.0, cy)
    left_mid = (cx - extent / 2.0, cy)
    channel_points = (
        ((9, 10), lower_right),
        ((11, 12), right_mid),
        ((13, 14), upper_right),
        ((15, 16), upper_left),
        ((17, 18), left_mid),
        ((19, 20), lower_left),
        ((21, 22), lower_left),
        ((23, 24), lower_right),
        ((27, 28), upper_right),
        ((29, 30), upper_left),
    )
    return _normalized_projection(
        channel_points,
        "euclidean_least_squares_center_extent_capsule",
    )


def _fit_center_extent(geometry, points):
    rows = []
    observations = []
    try:
        for (x_channel, y_channel), (x_extent, y_extent) in points:
            x = (
                float(geometry[x_channel])
                * GEOMETRY_CHANNEL_SCALES[x_channel]
            )
            y = (
                float(geometry[y_channel])
                * GEOMETRY_CHANNEL_SCALES[y_channel]
            )
            rows.extend(((1.0, 0.0, x_extent), (0.0, 1.0, y_extent)))
            observations.extend((x, y))
    except IndexError:
        return "malformed_geometry_width"
    except (TypeError, ValueError, OverflowError):
        return "nonfinite_or_nonnumeric_observation"
    if not all(math.isfinite(item) for item in observations):
        return "nonfinite_or_nonnumeric_observation"
    normal = [[0.0] * 3 for _ in range(3)]
    right = [0.0] * 3
    for row, observed in zip(rows, observations):
        for first in range(3):
            right[first] += row[first] * observed
            for second in range(3):
                normal[first][second] += row[first] * row[second]
    solution = _solve_three_by_three(normal, right)
    if solution is None:
        return "singular_least_squares_system"
    if solution[2] <= 0.0:
        return "nonpositive_fitted_extent"
    return tuple(solution)


def _solve_three_by_three(matrix, right):
    augmented = [
        [float(item) for item in row] + [float(value)]
        for row, value in zip(matrix, right)
    ]
    for column in range(3):
        pivot = max(
            range(column, 3),
            key=lambda row: abs(augmented[row][column]),
        )
        if abs(augmented[pivot][column]) <= 1e-15:
            return None
        augmented[column], augmented[pivot] = (
            augmented[pivot], augmented[column]
        )
        divisor = augmented[column][column]
        augmented[column] = [
            item / divisor for item in augmented[column]
        ]
        for row in range(3):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [
                item - factor * pivot_item
                for item, pivot_item in zip(
                    augmented[row], augmented[column]
                )
            ]
    return [augmented[index][3] for index in range(3)]


def _normalized_projection(channel_points, method):
    channels = []
    values = []
    for pair, point in channel_points:
        for channel, physical in zip(pair, point):
            normalized = physical / GEOMETRY_CHANNEL_SCALES[channel]
            if not math.isfinite(normalized) or not -1.0 <= normalized <= 1.0:
                return "projected_geometry_out_of_range"
            channels.append(channel)
            values.append(0.0 if normalized == 0.0 else normalized)
    return tuple(channels), tuple(values), method


def _replace_geometry(node, channels, projected, kind, field, method):
    geometry = list(node.normalized_geometry)
    try:
        original = [geometry[index] for index in channels]
    except IndexError as exc:
        raise ReplayError(
            "invalid_raw_node_width",
            "node {} normalized_geometry is too short for {}".format(
                node.position, kind
            ),
        ) from exc
    for channel, value in zip(channels, projected):
        geometry[channel] = float(value)
    change = _numeric_change(
        node.position,
        kind,
        field,
        list(channels),
        original,
        list(projected),
        method,
    )
    return _replace_namespace(
        node, normalized_geometry=tuple(geometry)
    ), change


def _numeric_change(
    position, kind, field, indices, original, projected, method
):
    physical_residuals = []
    for index, left, right in zip(indices, original, projected):
        try:
            difference = (
                (float(right) - float(left))
                * GEOMETRY_CHANNEL_SCALES[index]
            )
        except (TypeError, ValueError, OverflowError) as exc:
            raise ReplayError(
                "nonfinite_raw_geometry",
                "node {} {} contains a nonnumeric value".format(
                    position, kind
                ),
            ) from exc
        if not math.isfinite(difference):
            raise ReplayError(
                "nonfinite_raw_geometry",
                "node {} {} contains a nonfinite value".format(
                    position, kind
                ),
            )
        physical_residuals.append(difference)
    count = len(physical_residuals)
    l2 = math.sqrt(sum(item * item for item in physical_residuals))
    rms = l2 / math.sqrt(count) if count else None
    maximum = max(abs(item) for item in physical_residuals) if count else None
    return {
        "channel_or_category_indices": indices,
        "field": field,
        "kind": kind,
        "method": method,
        "node_position": position,
        "is_no_op": list(original) == list(projected),
        "original_values": original,
        "projected_values": list(projected),
        "projection_residual_l2_physical": l2,
        "projection_residual_max_abs_physical": maximum,
        "projection_residual_rms_physical": rms,
    }


def _categorical_change(
    position, kind, field, indices, original, projected, method
):
    return {
        "channel_or_category_indices": indices,
        "field": field,
        "kind": kind,
        "method": method,
        "node_position": position,
        "is_no_op": list(original) == list(projected),
        "original_values": original,
        "projected_values": list(projected),
        "projection_residual_l2_physical": None,
        "projection_residual_max_abs_physical": None,
        "projection_residual_rms_physical": None,
    }


def _projection_failure(node, pattern, family, reason):
    return {
        "decoded_profile_pattern": "|".join(pattern),
        "field": "normalized_geometry",
        "node_position": node.position,
        "reason": reason,
        "requested_profile_family": family,
        "transformation_kind": "profile_projection",
    }


def _semantic_failure(node, kind, reason):
    return {
        "decoded_profile_pattern": "",
        "field": "categorical_ids",
        "node_position": node.position,
        "reason": reason,
        "requested_profile_family": "",
        "transformation_kind": kind,
    }


def _profile_eligibility(raw):
    if raw is None:
        return {
            "family_eligible": False,
            "sketch_count": 0,
            "supported_sketch_count": 0,
        }
    sketch_count = 0
    supported = 0
    for node in raw.raw_nodes:
        if _node_token(node) != "sketch":
            continue
        sketch_count += 1
        try:
            pattern = tuple(
                PRIMITIVE_TYPES.tokens[item]
                for item in node.categorical_ids[4:8]
            )
        except (IndexError, TypeError):
            pattern = ()
        if pattern in PATTERN_TO_PROFILE:
            supported += 1
    return {
        "family_eligible": sketch_count > 0 and supported == sketch_count,
        "sketch_count": sketch_count,
        "supported_sketch_count": supported,
    }


def _derived_geometry_mask(node_type, categorical):
    mask = [False] * GEOMETRY_WIDTH
    if node_type == "reference_plane":
        mask[0:9] = [True] * 9
    elif node_type == "sketch":
        widths = {"line": 4, "arc": 6, "circle": 3}
        for slot in range(MAX_PRIMITIVES):
            primitive = PRIMITIVE_TYPES.tokens[categorical[4 + slot]]
            width = widths.get(primitive, 0)
            start = 9 + slot * PRIMITIVE_SLOT_WIDTH
            mask[start:start + width] = [True] * width
    elif node_type == "axis":
        mask[33:37] = [True] * 4
    elif node_type == "extrude":
        mask[37] = True
    elif node_type == "revolve":
        mask[38] = True
    return tuple(mask)


def _aggregate_records(records):
    by_path_arm = {}
    for path in PATHS:
        for arm in ARMS:
            selected = [
                item for item in records
                if item["path"] == path and item["arm"] == arm
            ]
            by_path_arm.setdefault(path, {})[arm] = _aggregate_group(selected)
    return {
        "arms": by_path_arm,
        "diagnostic_interpretation": _diagnostic_interpretation(by_path_arm),
        "frozen_original_result": {
            "gate_c": "failed_frozen",
            "gate_d": "failed_frozen",
            "gate_e": "not_run",
            "replay_can_change_gates": False,
        },
        "replay_record_count": len(records),
    }


def _diagnostic_interpretation(by_path_arm):
    per_path = {}
    oracle_deltas = {}
    for path in PATHS:
        arm2 = by_path_arm[path]["semantic_constants"][
            "invalid_to_valid_count"
        ]
        arm3 = by_path_arm[path]["model_category_projection"][
            "invalid_to_valid_count"
        ]
        arm4_valid = by_path_arm[path]["oracle_category_diagnostic"][
            "controlled_domain_valid_count"
        ]
        arm3_valid = by_path_arm[path]["model_category_projection"][
            "controlled_domain_valid_count"
        ]
        oracle_delta = arm4_valid - arm3_valid
        oracle_deltas[path] = oracle_delta
        per_path[path] = {
            "semantic_constant_sensitivity": _effect_label(arm2),
            "constraint_projection_rescue_count": arm3,
            "oracle_additional_valid_count": oracle_delta,
            "oracle_category_sensitivity": _effect_label(oracle_delta),
        }
    arm3_teacher = by_path_arm["teacher_forced"][
        "model_category_projection"
    ]["invalid_to_valid_count"]
    arm3_predicted = by_path_arm["predicted_history"][
        "model_category_projection"
    ]["invalid_to_valid_count"]
    if arm3_teacher >= 17 and arm3_predicted >= 17:
        constraint_label = "strong"
    elif arm3_teacher or arm3_predicted:
        constraint_label = "limited_or_path_specific"
    else:
        constraint_label = "none"
    return {
        "autoregressive_sensitivity": {
            arm: _autoregressive_sensitivity(by_path_arm, arm)
            for arm in ("model_category_projection", ORACLE_ARM)
        },
        "categorical_oracle_sensitivity": {
            "combined_label": _effect_label(max(oracle_deltas.values())),
            "combination_rule": "maximum_either_path_additional_valid",
            "per_path_additional_valid_count": oracle_deltas,
        },
        "constraint_parameterization_sensitivity": constraint_label,
        "per_path": per_path,
        "threshold_family_count": 17,
    }


def _autoregressive_sensitivity(by_path_arm, arm):
    teacher = by_path_arm["teacher_forced"][arm][
        "controlled_domain_valid_count"
    ]
    predicted = by_path_arm["predicted_history"][arm][
        "controlled_domain_valid_count"
    ]
    ratio = None if teacher == 0 else predicted / teacher
    return {
        "label": (
            "not_material"
            if teacher == 0 or predicted >= 0.8 * teacher
            else "material"
        ),
        "predicted_history_valid_count": predicted,
        "survival_ratio": ratio,
        "teacher_forced_valid_count": teacher,
        "threshold": "predicted_history < 0.8 * teacher_forced",
    }


def _aggregate_group(records):
    count = len(records)
    raw_complete = sum(item["raw_completion"] for item in records)
    raw_valid = sum(
        item["conversion"]["raw_integrity"]["valid"] for item in records
    )
    target_valid = sum(
        item["conversion"]["reconstruction_target"]["valid"]
        for item in records
    )
    controlled_valid = sum(
        item["conversion"]["controlled_domain"]["valid"]
        for item in records
    )
    primary = {code: 0 for code in _failure_codes()}
    any_failure = {code: 0 for code in _failure_codes()}
    transition_groups = {}
    residuals = {}
    for item in records:
        conversion = item["conversion"]
        first = conversion["primary_failure"]
        if first is not None:
            primary[first["code"]] += 1
        for code in conversion["controlled_domain"]["failure_codes"]:
            any_failure[code] += 1
        combination = _failure_combination(item["baseline_failure_codes"])
        group = transition_groups.setdefault(combination, {
            "attempted_count": 0,
            "baseline_invalid_count": 0,
            "invalid_to_valid_count": 0,
            "still_invalid_count": 0,
        })
        group["attempted_count"] += 1
        if not item["baseline_controlled_domain_valid"]:
            group["baseline_invalid_count"] += 1
            if item["invalid_to_valid"]:
                group["invalid_to_valid_count"] += 1
            else:
                group["still_invalid_count"] += 1
        for transformation in item["transformations"]:
            value = transformation["projection_residual_l2_physical"]
            maximum = transformation[
                "projection_residual_max_abs_physical"
            ]
            if value is not None and not transformation["is_no_op"]:
                residuals.setdefault(transformation["kind"], []).append(
                    (value, maximum)
                )
    strata = {}
    for field, label in (
        ("operation_template", "by_operation_template"),
        ("profile_family", "by_profile_family"),
    ):
        strata[label] = {
            value: _aggregate_stratum(
                [item for item in records if item[field] == value]
            )
            for value in sorted({item[field] for item in records})
        }
    family_eligible = sum(
        item["supported_profile_eligibility"]["family_eligible"]
        for item in records
    )
    supported_sketches = sum(
        item["supported_profile_eligibility"]["supported_sketch_count"]
        for item in records
    )
    sketches = sum(
        item["supported_profile_eligibility"]["sketch_count"]
        for item in records
    )
    return {
        "any_failure_counts": [
            {"code": code, "count": any_failure[code]}
            for code in _failure_codes()
        ],
        "attempted_count": count,
        "controlled_domain_valid_count": controlled_valid,
        "controlled_domain_valid_rate": _rate(controlled_valid, count),
        "invalid_to_valid_count": sum(item["invalid_to_valid"] for item in records),
        "primary_failure_counts": [
            {"code": code, "count": primary[code]}
            for code in _failure_codes()
        ],
        "projection_residual_distributions": {
            kind: _residual_distribution(values)
            for kind, values in sorted(residuals.items())
        },
        "raw_completion_count": raw_complete,
        "raw_completion_rate": _rate(raw_complete, count),
        "raw_integrity_valid_count": raw_valid,
        "raw_integrity_valid_rate": _rate(raw_valid, count),
        "reconstruction_target_valid_count": target_valid,
        "reconstruction_target_valid_rate": _rate(target_valid, count),
        "strata": strata,
        "supported_profile_eligibility": {
            "eligible_family_count": family_eligible,
            "eligible_family_rate": _rate(family_eligible, count),
            "sketch_count": sketches,
            "supported_sketch_count": supported_sketches,
            "supported_sketch_rate": _rate(supported_sketches, sketches),
        },
        "transitions_by_original_failure_combination": [
            dict({"failure_combination": name}, **transition_groups[name])
            for name in sorted(transition_groups)
        ],
    }


def _aggregate_stratum(records):
    count = len(records)
    valid = sum(
        item["conversion"]["controlled_domain"]["valid"]
        for item in records
    )
    return {
        "attempted_count": count,
        "controlled_domain_valid_count": valid,
        "controlled_domain_valid_rate": _rate(valid, count),
        "invalid_to_valid_count": sum(item["invalid_to_valid"] for item in records),
        "primary_failure_counts": _compact_failure_counts(records, primary=True),
        "any_failure_counts": _compact_failure_counts(records, primary=False),
    }


def _compact_failure_counts(records, *, primary):
    counts = {code: 0 for code in _failure_codes()}
    for item in records:
        conversion = item["conversion"]
        if primary:
            failure = conversion["primary_failure"]
            codes = () if failure is None else (failure["code"],)
        else:
            codes = conversion["controlled_domain"]["failure_codes"]
        for code in codes:
            counts[code] += 1
    return [
        {"code": code, "count": counts[code]}
        for code in _failure_codes() if counts[code]
    ]


def _residual_distribution(values):
    l2 = sorted(item[0] for item in values)
    maxima = [item[1] for item in values]
    return {
        "count": len(l2),
        "maximum": l2[-1],
        "maximum_absolute_component": max(maxima),
        "mean": sum(l2) / len(l2),
        "minimum": l2[0],
        "p25": _quantile(l2, 0.25),
        "p50": _quantile(l2, 0.50),
        "p75": _quantile(l2, 0.75),
        "p90": _quantile(l2, 0.90),
        "p95": _quantile(l2, 0.95),
        "rms": math.sqrt(sum(item * item for item in l2) / len(l2)),
    }


def _quantile(values, fraction):
    if len(values) == 1:
        return values[0]
    position = fraction * (len(values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return values[lower]
    weight = position - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def _run_metadata(
    source_metadata,
    evidence,
    records,
    *,
    scheduler_job_id,
    validator_evidence,
):
    repository = Path(__file__).resolve().parents[2]
    commit = _git_output(repository, ("rev-parse", "HEAD"))
    status = _git_output(repository, ("status", "--porcelain"))
    source_ids = [item["family_id"] for item in records if (
        item["path"] == PATHS[0] and item["arm"] == ARMS[0]
    )]
    return {
        "access_contract": {
            "authoritative_categories_used_for_transformation_arms": [
                ORACLE_ARM
            ],
            "corpus_payload_accessed": False,
            "model_or_checkpoint_loaded": False,
            "opencascade_run": False,
            "slurm_run": False,
            "target_continuous_geometry_accessed": False,
            "test_family_payload_accessed": False,
            "torch_runtime_loaded": "torch" in sys.modules,
            "training_run": False,
        },
        "arm_order": list(ARMS),
        "diagnostic_only": True,
        "family_count": EXPECTED_FAMILY_COUNT,
        "family_ids": source_ids,
        "frozen_original_result": {
            "gate_c": "failed_frozen",
            "gate_d": "failed_frozen",
            "gate_e": "not_run",
            "may_be_reinterpreted_by_replay": False,
        },
        "path_order": list(PATHS),
        "preregistration_path": str(
            PREREGISTRATION.relative_to(repository)
        ),
        "preregistration_sha256": _file_sha256(PREREGISTRATION),
        "quantile_method": "linear_interpolation_over_sorted_values",
        "replay_record_count": len(records),
        "repository_commit": commit,
        "repository_dirty": bool(status),
        "source_bundle": {
            "path": evidence.bundle_path,
            "checkpoint_epoch": source_metadata.get("checkpoint_epoch"),
            "checkpoint_global_step": source_metadata.get(
                "checkpoint_global_step"
            ),
            "checkpoint_sha256": source_metadata.get("checkpoint_sha256"),
            "evaluation_repository_commit": source_metadata.get(
                "repository_commit"
            ),
            "manifest_entry_sha256": evidence.entry_sha256,
            "manifest_sha256": evidence.manifest_sha256,
            "root_hidden_entries": list(evidence.hidden_entries),
            "slurm_job_id": scheduler_job_id,
        },
        "tool_path": str(Path(__file__).resolve().relative_to(repository)),
        "tool_sha256": _file_sha256(Path(__file__).resolve()),
        "validator": (
            "prototype.flat_baseline.conversion."
            "validate_and_convert_raw_prediction"
        ),
        "validator_source": {
            "aggregate_sha256": validator_evidence.aggregate_sha256,
            "evaluation_commit": validator_evidence.evaluation_commit,
            "per_file_sha256": validator_evidence.per_file_sha256,
        },
    }


def _transition_csv(records):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream, fieldnames=TRANSITION_COLUMNS, lineterminator="\n"
    )
    writer.writeheader()
    for item in records:
        conversion = item["conversion"]
        eligibility = item["supported_profile_eligibility"]
        projection = item["profile_projection"]
        writer.writerow({
            "family_id": item["family_id"],
            "path": item["path"],
            "arm": item["arm"],
            "operation_template": item["operation_template"],
            "profile_family": item["profile_family"],
            "baseline_controlled_domain_valid": _csv_bool(
                item["baseline_controlled_domain_valid"]
            ),
            "controlled_domain_valid": _csv_bool(
                conversion["controlled_domain"]["valid"]
            ),
            "invalid_to_valid": _csv_bool(item["invalid_to_valid"]),
            "baseline_failure_combination": _failure_combination(
                item["baseline_failure_codes"]
            ),
            "failure_combination": _failure_combination(
                conversion["controlled_domain"]["failure_codes"]
            ),
            "primary_failure_code": (
                ""
                if conversion["primary_failure"] is None
                else conversion["primary_failure"]["code"]
            ),
            "supported_profile_family_eligible": _csv_bool(
                eligibility["family_eligible"]
            ),
            "supported_sketch_count": eligibility["supported_sketch_count"],
            "sketch_count": eligibility["sketch_count"],
            "projection_attempt_count": projection["attempt_count"],
            "projection_success_count": projection["success_count"],
            "projection_unavailable_count": projection["unavailable_count"],
        })
    return stream.getvalue().encode("utf-8")


def _projection_failure_csv(failures):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=PROJECTION_FAILURE_COLUMNS,
        lineterminator="\n",
    )
    writer.writeheader()
    for item in failures:
        writer.writerow({name: item[name] for name in PROJECTION_FAILURE_COLUMNS})
    return stream.getvalue().encode("utf-8")


def _publish_artifacts(destination, artifacts):
    destination = Path(destination)
    parent = destination.parent
    if not parent.is_dir() or parent.is_symlink():
        raise ReplayError(
            "invalid_output_parent",
            "output parent must be an existing non-symlink directory",
        )
    if destination.exists() or destination.is_symlink():
        raise ReplayError(
            "output_collision",
            "diagnostic output already exists",
        )
    temporary = Path(tempfile.mkdtemp(
        prefix="." + destination.name + ".tmp-",
        dir=str(parent),
    ))
    published = False
    outcome = None
    try:
        for name in OUTPUT_ARTIFACTS:
            _write_fsynced(temporary / name, artifacts[name])
        manifest = b"".join(
            "{}  {}\n".format(
                hashlib.sha256(artifacts[name]).hexdigest(), name
            ).encode("utf-8")
            for name in sorted(OUTPUT_ARTIFACTS)
        )
        _write_fsynced(temporary / OUTPUT_MANIFEST, manifest)
        _fsync_directory(temporary)
        if {
            item.name for item in temporary.iterdir()
        } != set(OUTPUT_ARTIFACTS + (OUTPUT_MANIFEST,)):
            raise ReplayError(
                "output_artifact_set",
                "diagnostic artifact set differs before publication",
            )
        outcome = _atomic_no_replace(temporary, destination)
        published = True
        temporary = None
        try:
            _fsync_directory(parent)
        except OSError as exc:
            raise ReplayError(
                "output_published_but_parent_unsynced",
                str(exc),
            ) from exc
    except ReplayError:
        raise
    except (EvaluationError, OSError) as exc:
        code = (
            "output_published_but_parent_unsynced"
            if published else "output_publication_failure"
        )
        raise ReplayError(
            code,
            str(exc),
        ) from exc
    finally:
        if temporary is not None and temporary.exists():
            shutil.rmtree(str(temporary))
    return {
        "backend": outcome["publication_backend"],
        "details": outcome,
    }


def _write_fsynced(path, content):
    with path.open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def _source_relative_paths():
    return {
        *("evaluation/" + name for name in EVALUATION_ARTIFACTS),
        *(
            "workflow-evidence/" + name
            for name in WORKFLOW_EVIDENCE_ARTIFACTS
        ),
        REPORT_NAME,
    }


def _stable_relative_path(archived_path, expected):
    normalized = PurePosixPath(archived_path).as_posix()
    matches = [
        item for item in expected
        if normalized == item or normalized.endswith("/" + item)
    ]
    if len(matches) != 1:
        raise ReplayError(
            "invalid_source_manifest_path",
            archived_path,
        )
    return matches[0]


def _require_exact_regular_files(root, expected):
    if root.is_symlink() or not root.is_dir():
        raise ReplayError(
            "invalid_source_artifact_directory",
            root.name,
        )
    entries = tuple(root.iterdir())
    if {item.name for item in entries} != set(expected) or any(
        item.is_symlink() or not item.is_file() for item in entries
    ):
        raise ReplayError(
            "source_artifact_set_mismatch",
            root.name,
        )


def _strict_json(path):
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_constant,
        )
    except (OSError, ValueError) as exc:
        raise ReplayError("invalid_json_artifact", path.name) from exc


def _strict_json_lines(path):
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        return [
            json.loads(line, parse_constant=_reject_constant)
            for line in lines
        ]
    except (OSError, ValueError) as exc:
        raise ReplayError("invalid_jsonl_artifact", path.name) from exc


def _scheduler_job_id(path):
    try:
        matches = [
            line.split("=", 1)[1]
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.startswith("slurm_job_id=")
        ]
    except (OSError, UnicodeError) as exc:
        raise ReplayError(
            "invalid_scheduler_evidence",
            "scheduler evidence is unreadable",
        ) from exc
    if (
        len(matches) != 1
        or not matches[0]
        or not matches[0].isdigit()
    ):
        raise ReplayError(
            "invalid_scheduler_evidence",
            "exactly one numeric slurm_job_id is required",
        )
    return matches[0]


def _reject_constant(value):
    raise ValueError("nonfinite JSON constant {}".format(value))


def _json_document(value):
    return (_json_text(value) + "\n").encode("utf-8")


def _json_lines(values):
    return "".join(_json_text(item) + "\n" for item in values).encode("utf-8")


def _json_text(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _node_token(node):
    try:
        return NODE_TYPES.tokens[node.node_type_id]
    except (IndexError, TypeError):
        return ""


def _replace_namespace(value, **changes):
    values = vars(value).copy()
    values.update(changes)
    return SimpleNamespace(**values)


def _failure_codes():
    return tuple(
        code for stage in STAGE_ORDER for code in FAILURE_CODE_ORDER[stage]
    )


def _failure_combination(codes):
    return "|".join(codes) if codes else "<valid>"


def _effect_label(count):
    if count < 0:
        return "regression"
    if count >= 17:
        return "strong"
    if count > 0:
        return "limited"
    return "none"


def _rate(numerator, denominator):
    return None if denominator == 0 else numerator / denominator


def _csv_bool(value):
    return "true" if value else "false"


def _validate_record_order(records):
    expected = []
    family_ids = sorted({
        item["family_id"] for item in records
    })
    for family_id in family_ids:
        for path in PATHS:
            for arm in ARMS:
                expected.append((family_id, path, arm))
    actual = [
        (item["family_id"], item["path"], item["arm"]) for item in records
    ]
    if actual != expected or len(actual) != len(set(actual)):
        raise ReplayError(
            "replay_record_order",
            "family/path/arm order differs",
        )


def _stat_fingerprint(value):
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
    )


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _git_output(repository, arguments):
    try:
        completed = subprocess.run(
            ("git",) + tuple(arguments),
            cwd=str(repository),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            universal_newlines=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ReplayError(
            "repository_provenance",
            "git {}".format(" ".join(arguments)),
        ) from exc
    return completed.stdout.strip()


def _git_bytes(repository, arguments):
    try:
        completed = subprocess.run(
            ("git",) + tuple(arguments),
            cwd=str(repository),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ReplayError(
            "repository_provenance",
            "git {}".format(" ".join(arguments)),
        ) from exc
    return completed.stdout


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run the preregistered Phase B constraint-manifold replay"
    )
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args(argv)
    try:
        result = run_replay(arguments.bundle, arguments.output)
    except (ReplayError, EvaluationError, OSError, ValueError) as exc:
        sys.stderr.write("constraint_manifold_replay_failure: {}\n".format(exc))
        return 1
    sys.stdout.write(
        "replay_records={}\noutput={}\npublication_backend={}\n".format(
            result["replay_record_count"],
            result["output"],
            result["publication_backend"],
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
