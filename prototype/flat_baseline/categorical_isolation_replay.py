"""Preregistered 2-by-2 categorical-isolation replay for Phase B."""

from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import stat
import sys
import tempfile

from .constraint_manifold_replay import (
    ARMS as PARENT_ARMS,
    EXPECTED_FAMILY_COUNT,
    EXPECTED_REPLAY_RECORD_COUNT,
    ORACLE_ARM as PARENT_ORACLE_ARM,
    PATHS,
    PROJECTION_FAILURE_COLUMNS,
    ReplayError,
    OracleCategories,
    _canonical_axis,
    _canonical_plane,
    _categorical_change,
    _derived_geometry_mask,
    _file_sha256,
    _git_output,
    _git_bytes,
    _json_document,
    _json_lines,
    _json_safe,
    _load_bundle,
    _node_token,
    PLANE_FRAMES,
    PROFILE_PATTERNS,
    _profile_eligibility,
    _project_sketch,
    _replace_namespace,
    _set_plane_category,
    _set_profile_category,
    _stage_two_prediction,
    _stat_fingerprint,
    _validate_loaded_contract,
    validate_source_manifest,
    validate_validator_source,
)
from .conversion import validate_and_convert_raw_prediction
from .evaluate_length_conditioned import (
    EvaluationError,
    _atomic_no_replace,
    _fsync_directory,
    _raw_from_json,
)
from prototype.model_data.geometry import GEOMETRY_WIDTH


ARMS = (
    "model_plane_model_profile",
    "oracle_plane_model_profile",
    "model_plane_oracle_profile",
    "oracle_plane_oracle_profile",
)
MODEL_MODEL_ARM = ARMS[0]
PLANE_ONLY_ARM = ARMS[1]
PROFILE_ONLY_ARM = ARMS[2]
FULL_ORACLE_ARM = ARMS[3]
PARENT_ARM_MAP = {
    MODEL_MODEL_ARM: "model_category_projection",
    FULL_ORACLE_ARM: PARENT_ORACLE_ARM,
}
EXPECTED_PARENT_COMMIT = "e0829773589a07ab471bc3932bf91578107b5467"
EXPECTED_SOURCE_MANIFEST_SHA256 = (
    "a13d5f35b345b6814912145a56e0cce0"
    "290ae677b030a1215b61e92558862541"
)
EXPECTED_PARENT_MANIFEST_SHA256 = (
    "6ff46389fb3947b3448e5e52860833d0"
    "592e83f70849536ddfd4aeb32f73b4b2"
)
PARENT_ARTIFACTS = (
    "aggregate_summary.json",
    "family_transitions.csv",
    "projection_failures.csv",
    "replay_records.jsonl",
    "run_metadata.json",
)
OUTPUT_ARTIFACTS = (
    "factorial_records.jsonl",
    "family_transitions.csv",
    "aggregate_summary.json",
    "projection_failures.csv",
    "capsule_regressions.json",
    "run_metadata.json",
)
OUTPUT_MANIFEST = "sha256-manifest.txt"
EXPECTED_ROW_COUNT = EXPECTED_FAMILY_COUNT * len(PATHS) * len(ARMS)
PREREGISTRATION = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "specifications"
    / "flat_baseline_phase_b_categorical_isolation_replay.md"
)
CAPSULE_REGRESSION_IDS = (
    "sf_7b44d531d7f3646bd54685ebfa22a0cf2aca0115aa4eb5ac1cebae7f9375375e",
    "sf_85bfcef3e95d63fcc36a4cdeae8497a2909cabe3c58eed3687a9efd917b3aeaf",
    "sf_96d5102c2a138cd471e8699841125d3da5953645df906f86c3e23ccb1e97983a",
    "sf_a4071c6e4d9144b58494e0e9d7a6322421789d448827bbef61532416671576de",
    "sf_c29d2f2ca87aeffe549395ce758d0d77b8435a69d5c3cdfb1b7e5d9c4a8a1caf",
)
TRANSITION_COLUMNS = (
    "family_id",
    "path",
    "arm",
    "operation_template",
    "profile_family",
    "first_operation",
    "model_model_controlled_domain_valid",
    "controlled_domain_valid",
    "invalid_to_valid",
    "valid_to_invalid",
    "failure_combination",
    "projection_unavailability_reasons",
)


@dataclass(frozen=True)
class PlaneOracle:
    reference_plane: str


@dataclass(frozen=True)
class ProfileOracle:
    primitive_family: str


@dataclass(frozen=True)
class ParentReplayEvidence:
    path: str
    manifest_sha256: str
    entry_sha256: dict
    entry_stats: dict


@dataclass(frozen=True)
class ParentImplementationEvidence:
    commit: str
    path: str
    sha256: str


def run_categorical_isolation(source_bundle, parent_replay, output):
    """Validate two immutable inputs, construct artifacts, and publish once."""

    source = Path(os.path.abspath(os.fspath(source_bundle)))
    parent = Path(os.path.abspath(os.fspath(parent_replay)))
    destination = Path(os.path.abspath(os.fspath(output)))
    _require_distinct_output(destination, source, parent)
    source_evidence = validate_source_manifest(source)
    if source_evidence.manifest_sha256 != EXPECTED_SOURCE_MANIFEST_SHA256:
        raise ReplayError(
            "source_manifest_identity",
            "recovered source manifest digest differs",
        )
    parent_evidence = validate_parent_replay_manifest(parent)
    validator_evidence = validate_validator_source()
    parent_implementation = validate_parent_implementation_source()
    loaded = _load_bundle(source)
    parent_loaded = _load_parent_replay(parent)
    artifacts = make_categorical_isolation_artifacts(
        loaded,
        parent_loaded,
        source_evidence,
        parent_evidence,
        validator_evidence,
        parent_implementation,
    )
    if validate_source_manifest(source) != source_evidence:
        raise ReplayError(
            "source_bundle_changed",
            "source bundle changed during categorical isolation",
        )
    if validate_parent_replay_manifest(parent) != parent_evidence:
        raise ReplayError(
            "parent_replay_changed",
            "completed replay changed during categorical isolation",
        )
    publication = _publish_artifacts(destination, artifacts)
    return {
        "manifest_entry_count": len(OUTPUT_ARTIFACTS),
        "output": str(destination),
        "publication_backend": publication["publication_backend"],
        "record_count": EXPECTED_ROW_COUNT,
    }


def validate_parent_replay_manifest(parent):
    """Validate the exact completed-replay artifact set without parsing it."""

    root = Path(parent)
    if root.is_symlink() or not root.is_dir():
        raise ReplayError(
            "invalid_parent_replay",
            "parent replay must be a non-symlink directory",
        )
    expected = set(PARENT_ARTIFACTS + (OUTPUT_MANIFEST,))
    entries = tuple(root.iterdir())
    if {item.name for item in entries} != expected or any(
        item.is_symlink() or not item.is_file() for item in entries
    ):
        raise ReplayError(
            "parent_artifact_set_mismatch",
            "completed replay artifact set differs",
        )
    manifest = root / OUTPUT_MANIFEST
    manifest_bytes = manifest.read_bytes()
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    if manifest_sha256 != EXPECTED_PARENT_MANIFEST_SHA256:
        raise ReplayError(
            "parent_manifest_identity",
            "completed replay manifest digest differs",
        )
    lines = manifest_bytes.splitlines()
    if len(lines) != len(PARENT_ARTIFACTS):
        raise ReplayError(
            "parent_manifest_count",
            "completed replay manifest must contain five entries",
        )
    hashes = {}
    stats = {}
    for line in lines:
        try:
            digest_bytes, name_bytes = line.split(b"  ", 1)
            digest = digest_bytes.decode("ascii")
            name = name_bytes.decode("utf-8")
        except (ValueError, UnicodeError) as exc:
            raise ReplayError(
                "invalid_parent_manifest",
                "completed replay manifest encoding differs",
            ) from exc
        if (
            name not in PARENT_ARTIFACTS
            or name in hashes
            or len(digest) != 64
            or any(item not in "0123456789abcdef" for item in digest)
        ):
            raise ReplayError(
                "invalid_parent_manifest",
                "completed replay manifest entry differs",
            )
        path = root / name
        before = path.stat()
        if not stat.S_ISREG(before.st_mode):
            raise ReplayError("invalid_parent_artifact", name)
        actual = _file_sha256(path)
        after = path.stat()
        fingerprint = _stat_fingerprint(before)
        if fingerprint != _stat_fingerprint(after):
            raise ReplayError("parent_artifact_changed", name)
        if actual != digest:
            raise ReplayError("parent_hash_mismatch", name)
        hashes[name] = digest
        stats[name] = fingerprint
    if set(hashes) != set(PARENT_ARTIFACTS):
        raise ReplayError(
            "parent_manifest_set_mismatch",
            "completed replay stable paths differ",
        )
    return ParentReplayEvidence(
        str(root),
        manifest_sha256,
        dict(sorted(hashes.items())),
        dict(sorted(stats.items())),
    )


def validate_parent_implementation_source():
    """Require unchanged parent projection and transformation semantics."""

    repository = Path(__file__).resolve().parents[2]
    relative = "prototype/flat_baseline/constraint_manifold_replay.py"
    current = _file_sha256(repository / relative)
    committed = hashlib.sha256(
        _git_bytes(
            repository,
            ("show", "{}:{}".format(EXPECTED_PARENT_COMMIT, relative)),
        )
    ).hexdigest()
    if current != committed:
        raise ReplayError(
            "parent_implementation_source_mismatch",
            "{} expected={} actual={}".format(
                relative, committed, current
            ),
        )
    return ParentImplementationEvidence(
        EXPECTED_PARENT_COMMIT, relative, current
    )


def make_categorical_isolation_artifacts(
    loaded,
    parent_loaded,
    source_evidence,
    parent_evidence,
    validator_evidence,
    parent_implementation,
):
    """Construct deterministic artifact bytes without publishing them."""

    raw_records = loaded["raw_records"]
    examples = loaded["example_records"]
    source_metadata = loaded["run_metadata"]
    max_operations = _validate_loaded_contract(
        raw_records,
        examples,
        source_metadata,
        scheduler_job_id=loaded["scheduler_job_id"],
    )
    _validate_parent_contract(
        parent_loaded,
        source_evidence,
        tuple(item["family_id"] for item in raw_records),
    )
    raw_by_family = {item["family_id"]: item for item in raw_records}
    parent_by_key = {
        (item["family_id"], item["path"], item["arm"]): item
        for item in parent_loaded["records"]
    }
    records = []
    failures = []
    for example in examples:
        family_id = example["family_id"]
        for path in PATHS:
            original = _raw_from_json(raw_by_family[family_id][path])
            original_conversion = validate_and_convert_raw_prediction(
                _stage_two_prediction(original, path),
                max_operations=max_operations,
            )
            stored = example["stored_conversions"][path]
            if _json_safe(original_conversion) != stored:
                raise ReplayError(
                    "source_baseline_mismatch",
                    "{}.{}".format(family_id, path),
                )
            model_model_valid = None
            pending = []
            for arm in ARMS:
                transformed, changes, unavailable = _transform_for_example(
                    original, arm, example
                )
                conversion = validate_and_convert_raw_prediction(
                    _stage_two_prediction(transformed, path),
                    max_operations=max_operations,
                )
                if arm == MODEL_MODEL_ARM:
                    model_model_valid = conversion.controlled_domain.valid
                pending.append((
                    arm, transformed, changes, unavailable, conversion
                ))
            for arm, transformed, changes, unavailable, conversion in pending:
                if arm in PARENT_ARM_MAP:
                    legacy = _legacy_record(
                        example,
                        path,
                        PARENT_ARM_MAP[arm],
                        original_conversion,
                        transformed,
                        changes,
                        unavailable,
                        conversion,
                    )
                    parent_record = parent_by_key[
                        (family_id, path, PARENT_ARM_MAP[arm])
                    ]
                    if legacy != parent_record:
                        raise ReplayError(
                            "parent_arm_reconciliation_mismatch",
                            "{}.{}.{}".format(family_id, path, arm),
                        )
                unavailable_rows = [
                    dict(item, family_id=family_id, path=path, arm=arm)
                    for item in unavailable
                ]
                failures.extend(unavailable_rows)
                valid = conversion.controlled_domain.valid
                record = {
                    "arm": arm,
                    "controlled_domain_valid": valid,
                    "conversion": _json_safe(conversion),
                    "family_id": family_id,
                    "failure_combination": _failure_combination(
                        conversion.controlled_domain.failure_codes
                    ),
                    "first_operation": _first_operation(
                        example["operation_template"]
                    ),
                    "invalid_to_valid": (
                        not model_model_valid and valid
                    ),
                    "model_model_controlled_domain_valid": model_model_valid,
                    "operation_template": example["operation_template"],
                    "oracle_capabilities": _arm_capabilities(arm),
                    "path": path,
                    "profile_family": example["primitive_family"],
                    "projection_unavailability": unavailable,
                    "transformations": changes,
                    "valid_to_invalid": model_model_valid and not valid,
                }
                records.append(record)
    _validate_record_order(records)
    if len(records) != EXPECTED_ROW_COUNT:
        raise ReplayError(
            "factorial_record_count",
            "exactly 544 categorical-isolation rows are required",
        )
    capsule = _capsule_regression_report(
        records,
        failures,
        parent_loaded["projection_failures"],
    )
    aggregate = _aggregate(records)
    metadata = _run_metadata(
        source_metadata,
        source_evidence,
        parent_evidence,
        validator_evidence,
        parent_implementation,
        records,
        scheduler_job_id=loaded["scheduler_job_id"],
    )
    return {
        "factorial_records.jsonl": _json_lines(records),
        "family_transitions.csv": _transition_csv(records),
        "aggregate_summary.json": _json_document(aggregate),
        "projection_failures.csv": _projection_failure_csv(failures),
        "capsule_regressions.json": _json_document(capsule),
        "run_metadata.json": _json_document(metadata),
    }


def transform_categorical_isolation(
    raw,
    arm,
    *,
    plane_oracle=None,
    profile_oracle=None,
):
    """Apply exactly one factorial arm with capability-typed oracle access."""

    _validate_capabilities(arm, plane_oracle, profile_oracle)
    if raw is None:
        return None, [], []
    if arm == MODEL_MODEL_ARM:
        from .constraint_manifold_replay import transform_prediction
        return transform_prediction(raw, "model_category_projection")
    if arm == FULL_ORACLE_ARM:
        from .constraint_manifold_replay import transform_prediction
        return transform_prediction(
            raw,
            PARENT_ORACLE_ARM,
            oracle=OracleCategories(
                plane_oracle.reference_plane,
                profile_oracle.primitive_family,
            ),
        )
    transformed = copy.deepcopy(raw)
    nodes = list(transformed.raw_nodes)
    changes = []
    unavailable = []
    for index, node in enumerate(nodes):
        token = _node_token(node)
        current = node
        if arm == PLANE_ONLY_ARM and token == "reference_plane":
            current, category_change = _set_plane_category(
                current, plane_oracle.reference_plane
            )
            changes.append(category_change)
            current, mask_change = _set_plane_mask(current)
            changes.append(mask_change)
        if arm == PROFILE_ONLY_ARM and token == "sketch":
            current, category_changes = _set_profile_category(
                current, profile_oracle.primitive_family
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
        elif token == "sketch":
            current, change, failure = _project_sketch(current)
            if change is not None:
                changes.append(change)
            if failure is not None:
                unavailable.append(failure)
        nodes[index] = current
    transformed.raw_nodes = tuple(nodes)
    return transformed, changes, unavailable


def _transform_for_example(original, arm, example):
    if arm == MODEL_MODEL_ARM:
        return transform_categorical_isolation(original, arm)
    if arm == PLANE_ONLY_ARM:
        return transform_categorical_isolation(
            original,
            arm,
            plane_oracle=PlaneOracle(_oracle_plane(example)),
        )
    if arm == PROFILE_ONLY_ARM:
        return transform_categorical_isolation(
            original,
            arm,
            profile_oracle=ProfileOracle(_oracle_profile(example)),
        )
    return transform_categorical_isolation(
        original,
        arm,
        plane_oracle=PlaneOracle(_oracle_plane(example)),
        profile_oracle=ProfileOracle(_oracle_profile(example)),
    )


def _oracle_plane(example):
    family_id = example.get("family_id", "<unknown>")
    try:
        plane = example["reference_plane"]
    except (KeyError, TypeError) as exc:
        raise ReplayError("invalid_oracle_metadata", family_id) from exc
    if plane not in PLANE_FRAMES:
        raise ReplayError("invalid_oracle_metadata", family_id)
    return plane


def _oracle_profile(example):
    family_id = example.get("family_id", "<unknown>")
    try:
        profile = example["primitive_family"]
    except (KeyError, TypeError) as exc:
        raise ReplayError("invalid_oracle_metadata", family_id) from exc
    if profile not in PROFILE_PATTERNS:
        raise ReplayError("invalid_oracle_metadata", family_id)
    return profile


def _validate_capabilities(arm, plane_oracle, profile_oracle):
    if arm not in ARMS:
        raise ReplayError("unknown_factorial_arm", str(arm))
    expected_plane = arm in (PLANE_ONLY_ARM, FULL_ORACLE_ARM)
    expected_profile = arm in (PROFILE_ONLY_ARM, FULL_ORACLE_ARM)
    if expected_plane != isinstance(plane_oracle, PlaneOracle):
        raise ReplayError(
            "plane_oracle_capability",
            "{} plane capability differs".format(arm),
        )
    if expected_profile != isinstance(profile_oracle, ProfileOracle):
        raise ReplayError(
            "profile_oracle_capability",
            "{} profile capability differs".format(arm),
        )


def _set_plane_mask(node):
    old_mask = list(node.derived_geometry_mask)
    if len(old_mask) < GEOMETRY_WIDTH:
        raise ReplayError(
            "invalid_raw_node_width",
            "reference-plane derived_geometry_mask is too short",
        )
    expected = _derived_geometry_mask(
        "reference_plane", node.categorical_ids
    )
    new_mask = list(old_mask)
    new_mask[0:9] = expected[0:9]
    changed = _replace_namespace(
        node, derived_geometry_mask=tuple(new_mask)
    )
    change = _categorical_change(
        node.position,
        "oracle_reference_plane_mask",
        "derived_geometry_mask",
        list(range(9)),
        old_mask[0:9],
        new_mask[0:9],
        "plane_category_derived_applicability",
    )
    if not change["is_no_op"]:
        raise ReplayError(
            "plane_mask_rederivation_not_no_op",
            "reference-plane mask differs at node {}".format(node.position),
        )
    return changed, change


def _legacy_record(
    example,
    path,
    parent_arm,
    baseline,
    transformed,
    changes,
    unavailable,
    result,
):
    profile_unavailable = sum(
        item["transformation_kind"] == "profile_projection"
        for item in unavailable
    )
    return {
        "arm": parent_arm,
        "baseline_controlled_domain_valid": (
            baseline.controlled_domain.valid
        ),
        "baseline_failure_codes": list(
            baseline.controlled_domain.failure_codes
        ),
        "conversion": _json_safe(result),
        "family_id": example["family_id"],
        "invalid_to_valid": (
            not baseline.controlled_domain.valid
            and result.controlled_domain.valid
        ),
        "operation_template": example["operation_template"],
        "oracle_assisted": parent_arm == PARENT_ORACLE_ARM,
        "path": path,
        "profile_family": example["primitive_family"],
        "profile_projection": {
            "attempt_count": sum(
                item["kind"].endswith("_profile") for item in changes
            ) + profile_unavailable,
            "on_manifold_no_op_count": sum(
                item["kind"].endswith("_profile") and item["is_no_op"]
                for item in changes
            ),
            "success_count": sum(
                item["kind"].endswith("_profile") and not item["is_no_op"]
                for item in changes
            ),
            "unavailable_count": profile_unavailable,
        },
        "raw_completion": transformed is not None,
        "supported_profile_eligibility": _profile_eligibility(transformed),
        "transformations": changes,
    }


def _load_parent_replay(parent):
    try:
        records = [
            json.loads(line)
            for line in (
                parent / "replay_records.jsonl"
            ).read_text(encoding="utf-8").splitlines()
        ]
        metadata = json.loads(
            (parent / "run_metadata.json").read_text(encoding="utf-8")
        )
        with (parent / "projection_failures.csv").open(
            encoding="utf-8", newline=""
        ) as stream:
            failures = list(csv.DictReader(stream))
    except (OSError, ValueError, csv.Error) as exc:
        raise ReplayError(
            "invalid_parent_artifact",
            "completed replay scientific artifact is malformed",
        ) from exc
    return {
        "metadata": metadata,
        "projection_failures": failures,
        "records": records,
    }


def _validate_parent_contract(parent, source_evidence, family_ids):
    records = parent["records"]
    metadata = parent["metadata"]
    access = metadata.get("access_contract", {})
    if (
        len(records) != EXPECTED_REPLAY_RECORD_COUNT
        or metadata.get("replay_record_count") != EXPECTED_REPLAY_RECORD_COUNT
        or metadata.get("repository_commit") != EXPECTED_PARENT_COMMIT
        or metadata.get("repository_dirty") is not False
        or tuple(metadata.get("family_ids", ())) != family_ids
        or metadata.get("source_bundle", {}).get("manifest_sha256")
        != source_evidence.manifest_sha256
        or metadata.get("source_bundle", {}).get("manifest_entry_sha256")
        != source_evidence.entry_sha256
        or tuple(metadata.get("arm_order", ())) != PARENT_ARMS
        or tuple(metadata.get("path_order", ())) != PATHS
        or metadata.get("diagnostic_only") is not True
        or any(
            access.get(name) is not False
            for name in (
                "corpus_payload_accessed",
                "model_or_checkpoint_loaded",
                "opencascade_run",
                "slurm_run",
                "target_continuous_geometry_accessed",
                "test_family_payload_accessed",
                "torch_runtime_loaded",
                "training_run",
            )
        )
        or metadata.get("frozen_original_result", {}).get("gate_c")
        != "failed_frozen"
        or metadata.get("frozen_original_result", {}).get("gate_d")
        != "failed_frozen"
        or metadata.get("frozen_original_result", {}).get("gate_e")
        != "not_run"
    ):
        raise ReplayError(
            "parent_replay_contract",
            "completed replay provenance differs",
        )
    expected = [
        (family_id, path, arm)
        for family_id in family_ids
        for path in PATHS
        for arm in PARENT_ARMS
    ]
    actual = [
        (item.get("family_id"), item.get("path"), item.get("arm"))
        for item in records
    ]
    if actual != expected or len(actual) != len(set(actual)):
        raise ReplayError(
            "parent_replay_order",
            "completed replay row order differs",
        )
    _validate_parent_capsule_cohort(parent["projection_failures"])


def _validate_parent_capsule_cohort(failures):
    matching = [
        item for item in failures
        if item.get("path") == "predicted_history"
        and item.get("arm") == PARENT_ORACLE_ARM
        and item.get("reason") == "nonpositive_fitted_extent"
    ]
    actual_ids = tuple(sorted(item.get("family_id") for item in matching))
    if (
        actual_ids != tuple(sorted(CAPSULE_REGRESSION_IDS))
        or any(
            item.get("node_position") != "4"
            or item.get("decoded_profile_pattern")
            != "arc|arc|line|line"
            or item.get("requested_profile_family")
            != "capsule_line_arc"
            for item in matching
        )
    ):
        raise ReplayError(
            "parent_capsule_regression_cohort",
            "five parent capsule regressions differ",
        )


def _aggregate(records):
    arms = {
        path: {
            arm: _aggregate_group([
                item for item in records
                if item["path"] == path and item["arm"] == arm
            ])
            for arm in ARMS
        }
        for path in PATHS
    }
    effects = {
        path: _factorial_effects({
            arm: arms[path][arm]["controlled_domain_valid_count"]
            for arm in ARMS
        })
        for path in PATHS
    }
    return {
        "arms": arms,
        "factorial_effects": effects,
        "frozen_original_result": {
            "gate_c": "failed_frozen",
            "gate_d": "failed_frozen",
            "gate_e": "not_run",
            "replay_can_change_gates": False,
        },
        "record_count": len(records),
    }


def _aggregate_group(records):
    def strata(field):
        return {
            value: _transition_counts([
                item for item in records if item[field] == value
            ])
            for value in sorted({item[field] for item in records})
        }

    result = _transition_counts(records)
    failure_counts = {}
    reason_counts = {}
    for item in records:
        combination = item["failure_combination"]
        failure_counts[combination] = failure_counts.get(combination, 0) + 1
        for failure in item["projection_unavailability"]:
            reason = failure["reason"]
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    result.update({
        "failure_combination_counts": [
            {"combination": name, "count": failure_counts[name]}
            for name in sorted(failure_counts)
        ],
        "projection_unavailability_reason_counts": [
            {"reason": name, "count": reason_counts[name]}
            for name in sorted(reason_counts)
        ],
        "strata": {
            "by_first_operation": strata("first_operation"),
            "by_operation_template": strata("operation_template"),
            "by_profile_family": strata("profile_family"),
        },
    })
    return result


def _transition_counts(records):
    attempted = len(records)
    valid = sum(item["controlled_domain_valid"] for item in records)
    invalid_to_valid = sum(item["invalid_to_valid"] for item in records)
    valid_to_invalid = sum(item["valid_to_invalid"] for item in records)
    return {
        "attempted_count": attempted,
        "controlled_domain_valid_count": valid,
        "controlled_domain_valid_rate": (
            None if attempted == 0 else valid / attempted
        ),
        "invalid_to_valid_count": invalid_to_valid,
        "valid_to_invalid_count": valid_to_invalid,
    }


def _factorial_effects(valid_counts):
    model_model = valid_counts[MODEL_MODEL_ARM]
    plane_only = valid_counts[PLANE_ONLY_ARM]
    profile_only = valid_counts[PROFILE_ONLY_ARM]
    full_oracle = valid_counts[FULL_ORACLE_ARM]
    effects = {
        "combined_gain": full_oracle - model_model,
        "interaction": (
            full_oracle - plane_only - profile_only + model_model
        ),
        "plane_only_gain": plane_only - model_model,
        "profile_only_gain": profile_only - model_model,
        "valid_counts": {
            arm: valid_counts[arm] for arm in ARMS
        },
    }
    if (
        effects["interaction"]
        != effects["combined_gain"]
        - effects["plane_only_gain"]
        - effects["profile_only_gain"]
    ):
        raise ReplayError(
            "factorial_arithmetic",
            "factorial interaction reconciliation differs",
        )
    return effects


def _capsule_regression_report(records, failures, parent_failures):
    _validate_parent_capsule_cohort(parent_failures)
    actual_families = {
        item["family_id"] for item in failures
        if item["path"] == "predicted_history"
        and item["reason"] == "nonpositive_fitted_extent"
    }
    expected_families = set(CAPSULE_REGRESSION_IDS)
    if actual_families != expected_families:
        raise ReplayError(
            "capsule_regression_family_set",
            "expected={} actual={}".format(
                sorted(expected_families), sorted(actual_families)
            ),
        )
    rows = []
    for family_id in CAPSULE_REGRESSION_IDS:
        family_records = [
            item for item in records
            if item["family_id"] == family_id
            and item["path"] == "predicted_history"
        ]
        if (
            len(family_records) != len(ARMS)
            or any(
                item["operation_template"] != "EE"
                or item["profile_family"] != "capsule_line_arc"
                for item in family_records
            )
        ):
            raise ReplayError(
                "capsule_regression_record",
                family_id,
            )
        by_arm = {}
        for arm in ARMS:
            matching = [
                item for item in failures
                if item["family_id"] == family_id
                and item["path"] == "predicted_history"
                and item["arm"] == arm
                and item["reason"] == "nonpositive_fitted_extent"
            ]
            by_arm[arm] = {
                "node_positions": sorted(
                    item["node_position"] for item in matching
                ),
                "nonpositive_fitted_extent": bool(matching),
            }
        first = next(
            (
                arm for arm in ARMS
                if by_arm[arm]["nonpositive_fitted_extent"]
            ),
            None,
        )
        if not by_arm[FULL_ORACLE_ARM]["nonpositive_fitted_extent"]:
            raise ReplayError(
                "capsule_regression_full_oracle_reconciliation",
                family_id,
            )
        rows.append({
            "family_id": family_id,
            "first_appearance_arm": first,
            "operation_template": "EE",
            "path": "predicted_history",
            "per_arm": by_arm,
            "profile_family": "capsule_line_arc",
            "registered_arm_order": list(ARMS),
        })
    return {
        "cohort_count": len(rows),
        "reason": "nonpositive_fitted_extent",
        "rows": rows,
    }


def _run_metadata(
    source_metadata,
    source_evidence,
    parent_evidence,
    validator_evidence,
    parent_implementation,
    records,
    *,
    scheduler_job_id,
):
    repository = Path(__file__).resolve().parents[2]
    return {
        "access_contract": {
            "arm_oracle_capabilities": {
                arm: _arm_capabilities(arm) for arm in ARMS
            },
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
        "family_ids": [
            item["family_id"] for item in records
            if item["path"] == PATHS[0] and item["arm"] == ARMS[0]
        ],
        "frozen_original_result": {
            "gate_c": "failed_frozen",
            "gate_d": "failed_frozen",
            "gate_e": "not_run",
            "may_be_reinterpreted_by_replay": False,
        },
        "parent_replay": {
            "entry_sha256": parent_evidence.entry_sha256,
            "manifest_sha256": parent_evidence.manifest_sha256,
            "path": parent_evidence.path,
            "repository_commit": EXPECTED_PARENT_COMMIT,
        },
        "parent_replay_implementation": {
            "commit": parent_implementation.commit,
            "path": parent_implementation.path,
            "sha256": parent_implementation.sha256,
        },
        "path_order": list(PATHS),
        "preregistration_path": str(
            PREREGISTRATION.relative_to(repository)
        ),
        "preregistration_sha256": _file_sha256(PREREGISTRATION),
        "record_count": len(records),
        "repository_commit": _git_output(
            repository, ("rev-parse", "HEAD")
        ),
        "repository_dirty": bool(
            _git_output(repository, ("status", "--porcelain"))
        ),
        "source_bundle": {
            "checkpoint_epoch": source_metadata.get("checkpoint_epoch"),
            "checkpoint_global_step": source_metadata.get(
                "checkpoint_global_step"
            ),
            "checkpoint_sha256": source_metadata.get("checkpoint_sha256"),
            "entry_sha256": source_evidence.entry_sha256,
            "evaluation_repository_commit": source_metadata.get(
                "repository_commit"
            ),
            "manifest_sha256": source_evidence.manifest_sha256,
            "path": source_evidence.bundle_path,
            "slurm_job_id": scheduler_job_id,
        },
        "tool_path": str(Path(__file__).resolve().relative_to(repository)),
        "tool_sha256": _file_sha256(Path(__file__).resolve()),
        "validator_source": {
            "aggregate_sha256": validator_evidence.aggregate_sha256,
            "evaluation_commit": validator_evidence.evaluation_commit,
            "per_file_sha256": validator_evidence.per_file_sha256,
        },
    }


def _arm_capabilities(arm):
    return {
        "authoritative_primitive_family": (
            arm in (PROFILE_ONLY_ARM, FULL_ORACLE_ARM)
        ),
        "authoritative_reference_plane": (
            arm in (PLANE_ONLY_ARM, FULL_ORACLE_ARM)
        ),
    }


def _first_operation(template):
    if not isinstance(template, str) or not template:
        raise ReplayError(
            "invalid_operation_template",
            "operation template is absent",
        )
    first = template[0]
    if first not in ("E", "R"):
        raise ReplayError(
            "invalid_operation_template",
            template,
        )
    return first


def _failure_combination(codes):
    return "|".join(codes) if codes else "<valid>"


def _validate_record_order(records):
    family_ids = sorted({item["family_id"] for item in records})
    expected = [
        (family_id, path, arm)
        for family_id in family_ids
        for path in PATHS
        for arm in ARMS
    ]
    actual = [
        (item["family_id"], item["path"], item["arm"])
        for item in records
    ]
    if actual != expected or len(actual) != len(set(actual)):
        raise ReplayError(
            "factorial_record_order",
            "categorical-isolation row order differs",
        )


def _transition_csv(records):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream, fieldnames=TRANSITION_COLUMNS, lineterminator="\n"
    )
    writer.writeheader()
    for item in records:
        writer.writerow({
            "family_id": item["family_id"],
            "path": item["path"],
            "arm": item["arm"],
            "operation_template": item["operation_template"],
            "profile_family": item["profile_family"],
            "first_operation": item["first_operation"],
            "model_model_controlled_domain_valid": _csv_bool(
                item["model_model_controlled_domain_valid"]
            ),
            "controlled_domain_valid": _csv_bool(
                item["controlled_domain_valid"]
            ),
            "invalid_to_valid": _csv_bool(item["invalid_to_valid"]),
            "valid_to_invalid": _csv_bool(item["valid_to_invalid"]),
            "failure_combination": item["failure_combination"],
            "projection_unavailability_reasons": "|".join(
                failure["reason"]
                for failure in item["projection_unavailability"]
            ),
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
        writer.writerow({
            name: item[name] for name in PROJECTION_FAILURE_COLUMNS
        })
    return stream.getvalue().encode("utf-8")


def _csv_bool(value):
    return "true" if value else "false"


def _require_distinct_output(destination, source, parent):
    output_real = os.path.realpath(os.fspath(destination))
    for immutable in (source, parent):
        immutable_real = os.path.realpath(os.fspath(immutable))
        try:
            common = os.path.commonpath((output_real, immutable_real))
        except ValueError:
            common = None
        if common in (output_real, immutable_real):
            raise ReplayError(
                "output_overlaps_immutable_input",
                immutable_real,
            )


def _publish_artifacts(destination, artifacts):
    if set(artifacts) != set(OUTPUT_ARTIFACTS):
        raise ReplayError(
            "output_artifact_set",
            "categorical-isolation artifact keys differ",
        )
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
            "categorical-isolation output already exists",
        )
    temporary = Path(tempfile.mkdtemp(
        prefix="." + destination.name + ".tmp-",
        dir=str(parent),
    ))
    published = False
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
        if {item.name for item in temporary.iterdir()} != set(
            OUTPUT_ARTIFACTS + (OUTPUT_MANIFEST,)
        ):
            raise ReplayError(
                "output_artifact_set",
                "categorical-isolation artifact set differs",
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
        return outcome
    except ReplayError:
        raise
    except (EvaluationError, OSError) as exc:
        raise ReplayError(
            (
                "output_published_but_parent_unsynced"
                if published else "output_publication_failure"
            ),
            str(exc),
        ) from exc
    finally:
        if temporary is not None and temporary.exists():
            shutil.rmtree(str(temporary))


def _write_fsynced(path, content):
    with path.open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run the preregistered Phase B categorical isolation"
    )
    parser.add_argument("--source-bundle", required=True)
    parser.add_argument("--parent-replay", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args(argv)
    try:
        result = run_categorical_isolation(
            arguments.source_bundle,
            arguments.parent_replay,
            arguments.output,
        )
    except (ReplayError, EvaluationError, OSError, ValueError) as exc:
        sys.stderr.write(
            "categorical_isolation_replay_failure: {}\n".format(exc)
        )
        return 1
    sys.stdout.write(
        "records={}\noutput={}\npublication_backend={}\n".format(
            result["record_count"],
            result["output"],
            result["publication_backend"],
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
