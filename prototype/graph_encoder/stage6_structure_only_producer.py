"""Governed scientific producer for the prospective structure-only Stage 6.

Importing this module performs no input discovery or scientific work.  The
future entry point requires exact narrow input declarations and separate
execution authority before it can load train/development examples.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile

try:
    import torch
except ImportError:  # Pure contracts and artifact verification remain usable.
    torch = None

from .errors import GraphEncoderError
from .decoder_contract import (
    AUTONOMOUS_STOP_NODE_GENERATION_IDENTITY,
    GRID_SOFTMAX_OPERATION_MAGNITUDE_PARAMETERIZATION,
)
from .stage6_structure_only import (
    ARMS,
    COHORTS,
    CONDITIONS,
    DEVELOPMENT_FAMILY_COUNT,
    FALLBACK_SEEDS,
    FULL_SEEDS,
    INPUT_VERSION,
    PROTOCOL_VERSION,
    TEMPLATES,
    TRAIN_FAMILY_COUNT,
    score_family_record,
    summarize_execution,
)
from .stage6_timing import TIMING_VERSION


PRODUCER_VERSION = "GE1-STAGE6-STRUCTURE-ONLY-PRODUCER-v1"
EXECUTION_RECORD_VERSION = "GE1-STAGE6-STRUCTURE-ONLY-EXECUTION-RECORD-v1"
PRODUCER_ARTIFACT_VERSION = "GE1-STAGE6-STRUCTURE-ONLY-PRODUCER-ARTIFACT-v1"
CHECKPOINT_VERSION = "GE1-STAGE6-STRUCTURE-ONLY-CHECKPOINT-v1"
CHECKPOINT_BUNDLE_VERSION = "GE1-STAGE6-STRUCTURE-ONLY-CHECKPOINT-BUNDLE-v1"
INPUT_POLICY_VERSION = "GE1-STAGE6-STRUCTURE-ONLY-INPUT-POLICY-v1"
LEGACY_TIMING_VERSION = "GE1-STAGE6-STRUCTURE-ONLY-TIMING-v1"
ACCESS_VERSION = "GE1-STAGE6-STRUCTURE-ONLY-ACCESS-v1"
CHECKPOINT_IDENTITY_FIELDS = {
    "version", "protocol_version", "source_commit", "source_digest", "arm",
    "seed", "epoch", "model_config", "operation_magnitude_parameterization",
    "node_generation_identity",
    "training_partition_identity", "training_partition_hashes",
    "training_arithmetic", "parameter_count", "checkpoint_schema",
    "checkpoint_sha256", "execution_device", "runtime_identity",
    "timing_hardware_identity", "cuda_rng_preserved",
    "governed_map_location", "external_checkpoint", "warm_start",
    "checkpoint_reuse",
}

EPOCHS = 200
BATCH_SIZE = 8
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0
CLIP_NORM = 1.0
CAPACITY_DIFFERENCE_MAX = 0.05
# Historical postmortem modules import this literal to verify the superseded
# artifact.  The active producer never reads it and applies no cross-arm gate.
TRAIN_CEILING_SHORTFALL_DIFFERENCE_MAX = 0.05
TIMING_CONTINGENCY = 0.20
UNRESOLVED_HASH = "UNRESOLVED_REVIEWER_INPUT"
PRODUCER_FILES = (
    "resolved_config.json",
    "training_histories.jsonl",
    "checkpoint_identities.jsonl",
    "family_records.jsonl",
    "artifact_manifest.json",
    "SHA256SUMS",
)
ACCESS_DECLARATIONS = {
    "rr_accessed": False,
    "er_accessed": False,
    "iid_accessed": False,
    "history_depth_accessed": False,
    "geometry_extrapolation_accessed": False,
    "external_checkpoint_accessed": False,
    "model_artifact_accessed": False,
    "repaired_artifact_accessed": False,
    "preserved_feature_payload_accessed": False,
    "cad_kernel_executed": False,
    "stage7_performed": False,
}
FORBIDDEN_INPUT_TOKENS = (
    "secondary_systematic_validation", "systematic_rr", "test_er",
    "geometry_extrapolation", "history_depth", "checkpoint",
    "model_artifact", "repaired_artifact", "preserved_feature",
)


def producer_config():
    return {
        "producer_version": PRODUCER_VERSION,
        "execution_record_version": EXECUTION_RECORD_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "checkpoint_version": CHECKPOINT_VERSION,
        "input_policy_version": INPUT_POLICY_VERSION,
        "default_seeds": list(FULL_SEEDS),
        "fallback_seeds": list(FALLBACK_SEEDS),
        "arms": list(ARMS),
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "optimizer": "AdamW",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "gradient_clip_norm": CLIP_NORM,
        "fixed_checkpoint_epoch": EPOCHS,
        "capacity_difference_max": CAPACITY_DIFFERENCE_MAX,
        "plateau_window_epochs": 5,
        "plateau_relative_improvement_threshold": 0.01,
        "plateau_first_eligible_epoch": 10,
        "plateau_required_by_epoch": EPOCHS,
        "train_ceiling_cross_arm_gate_applied": False,
        "train_ceiling_cross_arm_difference_role": "diagnostic_only",
        "operation_magnitude_parameterization": (
            GRID_SOFTMAX_OPERATION_MAGNITUDE_PARAMETERIZATION
        ),
        "node_generation_identity": (
            AUTONOMOUS_STOP_NODE_GENERATION_IDENTITY
        ),
        "geometry_can_determine_optimization_reliability": False,
        "complete_geometric_validity_can_determine_optimization_reliability": False,
        "early_stopping": False,
        "best_checkpoint_selection": False,
        "development_checkpoint_selection": False,
        "warm_start": False,
        "checkpoint_reuse": False,
        "timing_version": TIMING_VERSION,
        "supported_execution_devices": ["cpu", "cuda:0"],
        "execution_device_selected_only_by_timing": True,
    }


def validate_legacy_timing_evidence(record):
    """Validate timing-v1 only for historical regression; never authorize v2."""

    required = {
        "version", "measured_before_scientific_outcomes", "observed_results_used",
        "corpus_accessed", "mode", "seconds_per_epoch_by_arm",
        "projected_three_seed_seconds", "projected_three_seed_seconds_with_contingency",
        "projected_two_seed_seconds", "projected_two_seed_seconds_with_contingency",
        "contingency_fraction", "available_wall_seconds",
        "three_seed_feasible", "two_seed_feasible", "fallback_reason",
    }
    if set(record) != required or record.get("version") != LEGACY_TIMING_VERSION:
        raise GraphEncoderError("invalid_stage6_timing", "timing fields differ")
    if (
        record["measured_before_scientific_outcomes"] is not True
        or record["observed_results_used"] is not False
        or not isinstance(record["corpus_accessed"], bool)
        or record["mode"] not in ("corpus_free", "authorized_train_only")
        or (record["mode"] == "corpus_free" and record["corpus_accessed"] is not False)
        or record["contingency_fraction"] != TIMING_CONTINGENCY
    ):
        raise GraphEncoderError("invalid_stage6_timing", "timing boundary differs")
    per_arm = record["seconds_per_epoch_by_arm"]
    if set(per_arm) != set(ARMS):
        raise GraphEncoderError("invalid_stage6_timing", "arm timing differs")
    numeric_names = (
        "projected_three_seed_seconds", "projected_three_seed_seconds_with_contingency",
        "projected_two_seed_seconds", "projected_two_seed_seconds_with_contingency",
        "available_wall_seconds",
    )
    values = list(per_arm.values()) + [record[name] for name in numeric_names]
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(float(value)) or value <= 0 for value in values):
        raise GraphEncoderError("invalid_stage6_timing", "timing value differs")
    projected_three = sum(float(per_arm[arm]) for arm in ARMS) * EPOCHS * len(FULL_SEEDS)
    projected_two = sum(float(per_arm[arm]) for arm in ARMS) * EPOCHS * len(FALLBACK_SEEDS)
    if not math.isclose(record["projected_three_seed_seconds"], projected_three,
                        rel_tol=1e-12, abs_tol=1e-9):
        raise GraphEncoderError("invalid_stage6_timing", "projection differs")
    if not math.isclose(record["projected_two_seed_seconds"], projected_two,
                        rel_tol=1e-12, abs_tol=1e-9):
        raise GraphEncoderError("invalid_stage6_timing", "fallback projection differs")
    full_with_contingency = projected_three * (1.0 + TIMING_CONTINGENCY)
    two_with_contingency = projected_two * (1.0 + TIMING_CONTINGENCY)
    if (
        not math.isclose(record["projected_three_seed_seconds_with_contingency"],
                         full_with_contingency, rel_tol=1e-12, abs_tol=1e-9)
        or not math.isclose(record["projected_two_seed_seconds_with_contingency"],
                            two_with_contingency, rel_tol=1e-12, abs_tol=1e-9)
    ):
        raise GraphEncoderError("invalid_stage6_timing", "contingency differs")
    feasible = full_with_contingency <= record["available_wall_seconds"]
    two_feasible = two_with_contingency <= record["available_wall_seconds"]
    if record["three_seed_feasible"] is not feasible:
        raise GraphEncoderError("invalid_stage6_timing", "feasibility differs")
    if record["two_seed_feasible"] is not two_feasible:
        raise GraphEncoderError("invalid_stage6_timing", "fallback feasibility differs")
    if feasible:
        if record["fallback_reason"] is not None:
            raise GraphEncoderError("invalid_stage6_timing", "unexpected fallback")
        return FULL_SEEDS
    if not two_feasible or not record["fallback_reason"]:
        raise GraphEncoderError("stage6_resource_infeasible", "fallback unavailable")
    return FALLBACK_SEEDS


def validate_timing_evidence(record, *, allow_fallback=False,
                             required_device=None, return_selection=False):
    """Require timing-v2; timing-v1 cannot authorize the current producer."""

    del allow_fallback  # Feasibility, not an imperative flag, governs fallback.
    from .stage6_timing import validate_timing_evidence as validate_v2
    selection = validate_v2(record, required_device=required_device)
    return selection if return_selection else selection["retained_seeds"]


def unresolved_timing_record():
    """Configuration placeholder; it is intentionally ineligible to execute."""

    return {
        "version": TIMING_VERSION,
        "status": "unresolved_prospective_measurement_required",
        "fallback_invoked": False,
        "selected_seeds": list(FULL_SEEDS),
        "contingency_fraction": TIMING_CONTINGENCY,
        "scientific_outcomes_observed": False,
    }


def validate_input_declaration(record):
    required = {
        "version", "train", "development", "unexpected_paths",
        "broad_parent_mounted", "external_inputs",
    }
    if set(record) != required or record.get("version") != INPUT_POLICY_VERSION:
        raise GraphEncoderError("invalid_stage6_input_policy", "fields differ")
    if record["unexpected_paths"] or record["broad_parent_mounted"] is not False:
        raise GraphEncoderError("unauthorized_stage6_input", "input exposure differs")
    if record["external_inputs"]:
        raise GraphEncoderError("unauthorized_stage6_input", "external input declared")
    for name, expected_partition, expected_count in (
        ("train", "operation_template_train", TRAIN_FAMILY_COUNT),
        ("development", "operation_template_development", DEVELOPMENT_FAMILY_COUNT),
    ):
        item = record[name]
        required_item = {
            "partition", "family_count", "root", "root_is_read_only",
            "authorized_files", "expected_sha256", "observed_sha256",
        }
        if (
            set(item) != required_item
            or item["partition"] != expected_partition
            or item["family_count"] != expected_count
            or item["root_is_read_only"] is not True
            or not isinstance(item["root"], str) or not item["root"]
            or not item["authorized_files"]
            or set(item["expected_sha256"]) != set(item["authorized_files"])
            or set(item["observed_sha256"]) != set(item["authorized_files"])
        ):
            raise GraphEncoderError("invalid_stage6_input_policy", name + " differs")
        material = "\n".join([item["root"]] + list(item["authorized_files"])).lower()
        if any(token in material for token in FORBIDDEN_INPUT_TOKENS):
            raise GraphEncoderError("unauthorized_stage6_input", name + " path differs")
        forbidden_components = {"rr", "er", "iid", "test"}
        if any(
            component.lower() in forbidden_components
            for raw_path in [item["root"]] + list(item["authorized_files"])
            for component in Path(raw_path).parts
        ):
            raise GraphEncoderError("unauthorized_stage6_input", name + " path differs")
        for relative in item["authorized_files"]:
            path = Path(relative)
            if path.is_absolute() or ".." in path.parts:
                raise GraphEncoderError("unauthorized_stage6_input", "unsafe path")
        for relative in item["authorized_files"]:
            expected = item["expected_sha256"][relative]
            observed = item["observed_sha256"][relative]
            if expected == UNRESOLVED_HASH or observed == UNRESOLVED_HASH:
                raise GraphEncoderError(
                    "unresolved_stage6_input_hash", name + " hash unresolved"
                )
            if (
                not isinstance(expected, str) or len(expected) != 64
                or any(char not in "0123456789abcdef" for char in expected)
                or observed != expected
            ):
                raise GraphEncoderError("stage6_input_hash_mismatch", relative)
    return True


def verify_declared_input_files(record):
    """Hash exactly the narrow allowlist and reject every exposed extra file."""

    validate_input_declaration(record)
    for cohort in ("train", "development"):
        item = record[cohort]
        root = Path(item["root"]).resolve()
        if not root.is_dir() or root.is_symlink():
            raise GraphEncoderError("invalid_stage6_input_policy", cohort + " root")
        observed_files = tuple(sorted(
            path.relative_to(root).as_posix() for path in root.rglob("*")
            if path.is_file()
        ))
        if observed_files != tuple(item["authorized_files"]):
            raise GraphEncoderError("unauthorized_stage6_input", cohort + " file set")
        for relative in observed_files:
            target = root / relative
            if target.is_symlink() or _sha(target) != item["expected_sha256"][relative]:
                raise GraphEncoderError("stage6_input_hash_mismatch", relative)
    return True


def unresolved_input_declaration():
    return {
        "version": INPUT_POLICY_VERSION,
        "train": {
            "partition": "operation_template_train",
            "family_count": TRAIN_FAMILY_COUNT,
            "root": "UNRESOLVED_NARROW_TRAIN_ROOT",
            "root_is_read_only": True,
            "authorized_files": ["UNRESOLVED_TRAIN_INDEX.json"],
            "expected_sha256": {"UNRESOLVED_TRAIN_INDEX.json": UNRESOLVED_HASH},
            "observed_sha256": {"UNRESOLVED_TRAIN_INDEX.json": UNRESOLVED_HASH},
        },
        "development": {
            "partition": "operation_template_development",
            "family_count": DEVELOPMENT_FAMILY_COUNT,
            "root": "UNRESOLVED_NARROW_DEVELOPMENT_ROOT",
            "root_is_read_only": True,
            "authorized_files": ["UNRESOLVED_DEVELOPMENT_INDEX.json"],
            "expected_sha256": {
                "UNRESOLVED_DEVELOPMENT_INDEX.json": UNRESOLVED_HASH
            },
            "observed_sha256": {
                "UNRESOLVED_DEVELOPMENT_INDEX.json": UNRESOLVED_HASH
            },
        },
        "unexpected_paths": [],
        "broad_parent_mounted": False,
        "external_inputs": [],
    }


def capacity_gate(flat_count, graph_count):
    for value in (flat_count, graph_count):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise GraphEncoderError("invalid_stage6_capacity", "count differs")
    difference = abs(graph_count - flat_count) / float(flat_count)
    return {
        "flat_trainable_parameters": flat_count,
        "typed_graph_trainable_parameters": graph_count,
        "relative_difference": difference,
        "maximum_inclusive": CAPACITY_DIFFERENCE_MAX,
        "pass": difference <= CAPACITY_DIFFERENCE_MAX,
    }


def optimization_reliability(training_runs, train_scores, retained_seeds):
    """Apply arm-internal checks; report cross-arm train scores diagnostically."""

    from .training import plateau_state

    runs = {(row["arm"], row["seed"]): row for row in training_runs}
    expected = {(arm, seed) for arm in ARMS for seed in retained_seeds}
    if set(runs) != expected or len(training_runs) != len(expected):
        raise GraphEncoderError("invalid_stage6_optimization", "run matrix differs")
    per_run = []
    for arm, seed in sorted(expected, key=lambda item: (item[1], item[0])):
        row = runs[(arm, seed)]
        losses = tuple(row.get("epoch_losses", ()))
        gradients = tuple(row.get("epoch_gradient_norms", ()))
        finite = (
            len(losses) == EPOCHS and len(gradients) == EPOCHS
            and all(math.isfinite(float(value)) for value in losses + gradients)
        )
        plateau = plateau_state(losses) if finite else None
        plateau_by_200 = finite and plateau.first_plateau_epoch is not None
        fixed = row.get("completed_epoch") == EPOCHS and row.get("checkpoint_epoch") == EPOCHS
        per_run.append({
            "arm": arm, "seed": seed, "finite_losses_and_gradients": finite,
            "first_plateau_epoch": None if plateau is None else plateau.first_plateau_epoch,
            "plateau_by_epoch_200": plateau_by_200,
            "fixed_epoch_200_checkpoint": fixed,
            "pass_before_train_ceiling": finite and plateau_by_200 and fixed,
        })
    ceilings = []
    for seed in retained_seeds:
        values = {
            arm: float(train_scores[(arm, seed)]) for arm in ARMS
        }
        if any(not math.isfinite(value) or value < 0.0 or value > 1.0
               for value in values.values()):
            raise GraphEncoderError("invalid_stage6_optimization", "ceiling differs")
        shortfalls = {arm: 1.0 - value for arm, value in values.items()}
        difference = abs(shortfalls["typed_graph"] - shortfalls["flat"])
        ceilings.append({
            "seed": seed, "structural_train_scores": values,
            "structural_train_ceiling_shortfalls": shortfalls,
            "absolute_shortfall_difference": difference,
            "binding": False,
            "role": "diagnostic_only",
        })
    overall = all(row["pass_before_train_ceiling"] for row in per_run)
    return {
        "runs": per_run,
        "train_ceiling_comparisons": ceilings,
        "geometry_used": False,
        "complete_geometric_validity_used": False,
        "pass": overall,
    }


def checkpoint_identity(*, arm, seed, source_commit, source_digest,
                        model_config, partition_identity,
                        partition_hashes, parameter_count,
                        training_arithmetic, checkpoint_sha256,
                        execution_device="cpu", runtime_identity=None,
                        timing_hardware_identity=None,
                        cuda_rng_preserved=False):
    if arm not in ARMS or seed not in FULL_SEEDS:
        raise GraphEncoderError("invalid_stage6_checkpoint", "arm/seed differs")
    from .stage6_device import validate_execution_device
    execution_device = validate_execution_device(execution_device)
    if runtime_identity is None:
        runtime_identity = {
            "execution_device": execution_device,
            "historical_fixture_default": True,
        }
    if timing_hardware_identity is None:
        timing_hardware_identity = {
            "execution_device": execution_device,
            "historical_fixture_default": True,
        }
    record = {
        "version": CHECKPOINT_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "source_commit": source_commit,
        "source_digest": source_digest,
        "arm": arm,
        "seed": seed,
        "epoch": EPOCHS,
        "model_config": model_config,
        "operation_magnitude_parameterization": (
            GRID_SOFTMAX_OPERATION_MAGNITUDE_PARAMETERIZATION
        ),
        "node_generation_identity": (
            AUTONOMOUS_STOP_NODE_GENERATION_IDENTITY
        ),
        "training_partition_identity": partition_identity,
        "training_partition_hashes": partition_hashes,
        "training_arithmetic": training_arithmetic,
        "parameter_count": parameter_count,
        "checkpoint_schema": model_config.get("checkpoint_schema"),
        "checkpoint_sha256": checkpoint_sha256,
        "execution_device": execution_device,
        "runtime_identity": runtime_identity,
        "timing_hardware_identity": timing_hardware_identity,
        "cuda_rng_preserved": bool(cuda_rng_preserved),
        "governed_map_location": execution_device,
        "external_checkpoint": False,
        "warm_start": False,
        "checkpoint_reuse": False,
    }
    validate_checkpoint_identity(record, expected=record)
    return record


def validate_checkpoint_identity(record, *, expected):
    if set(record) != CHECKPOINT_IDENTITY_FIELDS or set(expected) != CHECKPOINT_IDENTITY_FIELDS:
        raise GraphEncoderError("invalid_stage6_checkpoint", "fields differ")
    for name, value in expected.items():
        if record.get(name) != value:
            raise GraphEncoderError("invalid_stage6_checkpoint", name + " differs")
    if (
        record.get("version") != CHECKPOINT_VERSION
        or record.get("protocol_version") != PROTOCOL_VERSION
        or record.get("epoch") != EPOCHS
        or record.get("operation_magnitude_parameterization")
        != GRID_SOFTMAX_OPERATION_MAGNITUDE_PARAMETERIZATION
        or record.get("node_generation_identity")
        != AUTONOMOUS_STOP_NODE_GENERATION_IDENTITY
        or record.get("external_checkpoint") is not False
        or record.get("warm_start") is not False
        or record.get("checkpoint_reuse") is not False
        or record.get("execution_device") not in ("cpu", "cuda:0")
        or record.get("governed_map_location") != record.get("execution_device")
        or not isinstance(record.get("runtime_identity"), dict)
        or not isinstance(record.get("timing_hardware_identity"), dict)
        or record.get("cuda_rng_preserved")
        is not (record.get("execution_device") == "cuda:0")
    ):
        raise GraphEncoderError("invalid_stage6_checkpoint", "identity differs")
    lengths = {"source_commit": 40, "source_digest": 64, "checkpoint_sha256": 64}
    for name, length in lengths.items():
        value = record.get(name)
        if (
            not isinstance(value, str) or len(value) != length
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise GraphEncoderError("invalid_stage6_checkpoint", name + " malformed")
    return True


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write(path, text):
    destination = Path(path)
    descriptor, temporary = tempfile.mkstemp(
        prefix="." + destination.name + ".tmp-", dir=str(destination.parent)
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(text.encode("utf-8"))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, str(destination))
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _job_id(value):
    value = str(value)
    if not value.isdigit() or int(value) <= 0:
        raise GraphEncoderError("invalid_stage6_job_id", "decimal job ID required")
    return value


def create_checkpoint_bundle(checkpoints, output_dir, *, job_id, source_evidence,
                             input_evidence, retained_seeds):
    """Atomically retain the exact recovered epoch-200 wrapper checkpoints."""

    job_id = _job_id(job_id)
    rows = tuple(checkpoints)
    expected_pairs = {(arm, seed) for arm in ARMS for seed in retained_seeds}
    if {(row["identity"]["arm"], row["identity"]["seed"]) for row in rows} != expected_pairs:
        raise GraphEncoderError("invalid_stage6_checkpoint_bundle", "matrix differs")
    root = Path(output_dir)
    if root.exists() or root.is_symlink() or not root.parent.is_dir():
        raise GraphEncoderError("unsafe_stage6_checkpoint_bundle", str(root))
    staging = root.with_name(root.name + ".incomplete-" + job_id)
    if staging.exists() or staging.is_symlink():
        raise GraphEncoderError("unsafe_stage6_checkpoint_bundle", str(staging))
    staging.mkdir()
    identities = []
    for row in sorted(rows, key=lambda value: (value["identity"]["seed"], value["identity"]["arm"])):
        identity = row["identity"]
        source = Path(row["path"])
        name = "stage6-{}-seed{}.pt".format(identity["arm"], identity["seed"])
        target = staging / name
        if source.is_symlink() or not source.is_file():
            raise GraphEncoderError("invalid_stage6_checkpoint_bundle", name)
        shutil.copyfile(str(source), str(target))
        if _sha(target) != identity["stage6_wrapper_sha256"]:
            raise GraphEncoderError("invalid_stage6_checkpoint_bundle", "wrapper hash differs")
        identities.append({**identity, "bundle_path": name})
    resolved = {
        "schema_version": CHECKPOINT_BUNDLE_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "checkpoint_version": CHECKPOINT_VERSION,
        "retained_seeds": list(retained_seeds),
        "source_evidence": source_evidence,
        "input_evidence": input_evidence,
        "checkpoint_identities": identities,
        "job_id": job_id,
    }
    _write(staging / "resolved_config.json", _canonical(resolved) + "\n")
    ordinary = ("resolved_config.json",) + tuple(row["bundle_path"] for row in identities)
    manifest = {
        "schema_version": CHECKPOINT_BUNDLE_VERSION,
        "artifacts": [{
            "path": name, "byte_size": (staging / name).stat().st_size,
            "sha256": _sha(staging / name),
        } for name in ordinary],
    }
    _write(staging / "artifact_manifest.json", _canonical(manifest) + "\n")
    checksum_names = ("artifact_manifest.json",) + ordinary
    _write(staging / "SHA256SUMS", "".join(
        "{}  {}\n".format(_sha(staging / name), name) for name in checksum_names
    ))
    os.replace(str(staging), str(root))
    return verify_checkpoint_bundle(root, retained_seeds=retained_seeds)


def _checksum_rows(path, governed_code):
    rows = []
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
        for line in lines:
            parts = line.split("  ", 1)
            if len(parts) != 2 or len(parts[0]) != 64:
                raise ValueError("malformed checksum")
            rows.append(tuple(parts))
    except (OSError, UnicodeError, ValueError) as exc:
        raise GraphEncoderError(governed_code, "malformed SHA256SUMS") from exc
    return tuple(rows)


def verify_checkpoint_bundle(path, *, retained_seeds):
    root = Path(path)
    expected_names = {
        "resolved_config.json", "artifact_manifest.json", "SHA256SUMS",
        *("stage6-{}-seed{}.pt".format(arm, seed)
          for seed in retained_seeds for arm in ARMS),
    }
    if (
        root.is_symlink() or not root.is_dir()
        or {item.name for item in root.iterdir()} != expected_names
        or any(item.is_symlink() or not item.is_file() for item in root.iterdir())
    ):
        raise GraphEncoderError("invalid_stage6_checkpoint_bundle", "file set differs")
    resolved = _load_json_file(root / "resolved_config.json", "invalid_stage6_checkpoint_bundle")
    manifest = _load_json_file(root / "artifact_manifest.json", "invalid_stage6_checkpoint_bundle")
    identities = resolved.get("checkpoint_identities", ())
    expected_pairs = {(arm, seed) for arm in ARMS for seed in retained_seeds}
    if (
        resolved.get("schema_version") != CHECKPOINT_BUNDLE_VERSION
        or resolved.get("protocol_version") != PROTOCOL_VERSION
        or resolved.get("checkpoint_version") != CHECKPOINT_VERSION
        or tuple(resolved.get("retained_seeds", ())) != tuple(retained_seeds)
        or {(row.get("arm"), row.get("seed")) for row in identities} != expected_pairs
        or len(identities) != len(expected_pairs)
        or manifest.get("schema_version") != CHECKPOINT_BUNDLE_VERSION
    ):
        raise GraphEncoderError("invalid_stage6_checkpoint_bundle", "identity differs")
    for identity in identities:
        bundle_path = identity.get("bundle_path")
        core = {key: value for key, value in identity.items()
                if key not in ("bundle_path", "stage6_wrapper_sha256")}
        validate_checkpoint_identity(core, expected=core)
        if _sha(root / bundle_path) != identity.get("stage6_wrapper_sha256"):
            raise GraphEncoderError("invalid_stage6_checkpoint_bundle", "checkpoint differs")
    declared = tuple(row.get("path") for row in manifest.get("artifacts", ()))
    expected_ordinary = ("resolved_config.json",) + tuple(
        row["bundle_path"] for row in identities
    )
    if declared != expected_ordinary:
        raise GraphEncoderError("invalid_stage6_checkpoint_bundle", "manifest differs")
    for row in manifest["artifacts"]:
        target = root / row["path"]
        if target.stat().st_size != row.get("byte_size") or _sha(target) != row.get("sha256"):
            raise GraphEncoderError("invalid_stage6_checkpoint_bundle", "manifest hash differs")
    checksums = _checksum_rows(root / "SHA256SUMS", "invalid_stage6_checkpoint_bundle")
    expected_checksum_names = ("artifact_manifest.json",) + expected_ordinary
    if tuple(name for unused, name in checksums) != expected_checksum_names:
        raise GraphEncoderError("invalid_stage6_checkpoint_bundle", "checksum coverage differs")
    if any(_sha(root / name) != digest for digest, name in checksums):
        raise GraphEncoderError("invalid_stage6_checkpoint_bundle", "checksum differs")
    return {
        "schema_version": CHECKPOINT_BUNDLE_VERSION,
        "bundle_sha256": _sha(root / "SHA256SUMS"),
        "checkpoint_count": len(identities),
        "checkpoint_identities": identities,
        "source_evidence": resolved["source_evidence"],
        "input_evidence": resolved["input_evidence"],
        "verification_status": "pass",
    }


def _load_json_file(path, governed_code):
    target = Path(path)
    if target.is_symlink() or not target.is_file():
        raise GraphEncoderError(governed_code, "JSON file is not regular")
    raw = target.read_text(encoding="utf-8")
    try:
        value = json.loads(
            raw,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ValueError("nonfinite " + constant)
            ),
        )
    except (ValueError, json.JSONDecodeError) as exc:
        raise GraphEncoderError(governed_code, "malformed JSON") from exc
    if raw != _canonical(value) + "\n":
        raise GraphEncoderError(governed_code, "noncanonical JSON")
    return value


def create_producer_artifact(payload, checkpoint_identities, output_dir, *,
                             job_id, checkpoint_bundle_reference=None):
    """Atomically publish a finalizer-compatible derived execution record."""

    job_id = _job_id(job_id)
    if not isinstance(checkpoint_bundle_reference, dict):
        raise GraphEncoderError("invalid_stage6_producer_artifact", "bundle absent")
    root = Path(output_dir)
    if root.exists() or root.is_symlink() or not root.parent.is_dir():
        raise GraphEncoderError("unsafe_stage6_producer_output", str(root))
    staging = root.with_name(root.name + ".incomplete-" + job_id)
    if staging.exists() or staging.is_symlink():
        raise GraphEncoderError("unsafe_stage6_producer_output", str(staging))
    scored, summary = summarize_execution(payload)
    del scored, summary  # Validation is outcome-independent.
    staging.mkdir()
    try:
        resolved = {
            key: copy.deepcopy(value) for key, value in payload.items()
            if key not in ("training_runs", "family_records")
        }
        resolved.update({
            "producer_version": PRODUCER_VERSION,
            "execution_record_version": EXECUTION_RECORD_VERSION,
            "producer_artifact_version": PRODUCER_ARTIFACT_VERSION,
            "producer_config": producer_config(),
            "checkpoint_identity_count": len(checkpoint_identities),
            "checkpoint_bundle_reference": checkpoint_bundle_reference,
            "job_id": job_id,
            **ACCESS_DECLARATIONS,
        })
        _write(staging / "resolved_config.json", _canonical(resolved) + "\n")
        _write(staging / "training_histories.jsonl", "".join(
            _canonical(row) + "\n" for row in payload["training_runs"]
        ))
        _write(staging / "checkpoint_identities.jsonl", "".join(
            _canonical(row) + "\n" for row in checkpoint_identities
        ))
        _write(staging / "family_records.jsonl", "".join(
            _canonical(row) + "\n" for row in payload["family_records"]
        ))
        ordinary = PRODUCER_FILES[:4]
        manifest = {
            "schema_version": PRODUCER_ARTIFACT_VERSION,
            "artifacts": [{
                "path": name,
                "byte_size": (staging / name).stat().st_size,
                "sha256": _sha(staging / name),
            } for name in ordinary],
        }
        _write(staging / "artifact_manifest.json", _canonical(manifest) + "\n")
        checksum_names = ("artifact_manifest.json",) + ordinary
        _write(staging / "SHA256SUMS", "".join(
            "{}  {}\n".format(_sha(staging / name), name)
            for name in checksum_names
        ))
        os.replace(str(staging), str(root))
    except Exception:
        raise
    return verify_producer_artifact(root)


def _read_jsonl(path):
    target = Path(path)
    if target.is_symlink() or not target.is_file():
        raise GraphEncoderError("invalid_stage6_producer_artifact", "JSONL not regular")
    raw = target.read_text(encoding="utf-8")
    if not raw.endswith("\n"):
        raise GraphEncoderError("invalid_stage6_producer_artifact", "final LF absent")
    rows = []
    for line in raw.splitlines():
        try:
            value = json.loads(
                line,
                parse_constant=lambda constant: (_ for _ in ()).throw(
                    ValueError("nonfinite " + constant)
                ),
            )
        except (ValueError, json.JSONDecodeError) as exc:
            raise GraphEncoderError(
                "invalid_stage6_producer_artifact", "malformed JSONL"
            ) from exc
        if line != _canonical(value):
            raise GraphEncoderError("invalid_stage6_producer_artifact", "JSONL differs")
        rows.append(value)
    return rows


def load_execution_record(path):
    """Verify a producer artifact and return the finalizer input in memory."""

    verification = verify_producer_artifact(path, return_payload=True)
    payload = verification.pop("payload")
    payload["producer_artifact_evidence"] = {
        "schema_version": verification["producer_artifact_version"],
        "producer_artifact_sha256": verification["producer_artifact_sha256"],
        "checkpoint_bundle_sha256": verification["checkpoint_bundle_sha256"],
        "verification_status": "pass",
    }
    return payload


def verify_producer_artifact(path, *, return_payload=False):
    root = Path(path)
    if (
        not root.is_dir() or root.is_symlink()
        or tuple(sorted(item.name for item in root.iterdir()))
        != tuple(sorted(PRODUCER_FILES))
        or any(item.is_symlink() or not item.is_file() for item in root.iterdir())
    ):
        raise GraphEncoderError("invalid_stage6_producer_artifact", "file set differs")
    for name in ("resolved_config.json", "artifact_manifest.json"):
        _load_json_file(root / name, "invalid_stage6_producer_artifact")
    manifest = _load_json_file(
        root / "artifact_manifest.json", "invalid_stage6_producer_artifact"
    )
    ordinary = PRODUCER_FILES[:4]
    if (
        manifest.get("schema_version") != PRODUCER_ARTIFACT_VERSION
        or tuple(row.get("path") for row in manifest.get("artifacts", ())) != ordinary
    ):
        raise GraphEncoderError("invalid_stage6_producer_artifact", "manifest differs")
    for row in manifest["artifacts"]:
        target = root / row["path"]
        if target.stat().st_size != row["byte_size"] or _sha(target) != row["sha256"]:
            raise GraphEncoderError("invalid_stage6_producer_artifact", "hash differs")
    checksum_names = ("artifact_manifest.json",) + ordinary
    lines = _checksum_rows(
        root / "SHA256SUMS", "invalid_stage6_producer_artifact"
    )
    if tuple(name for unused, name in lines) != checksum_names:
        raise GraphEncoderError("invalid_stage6_producer_artifact", "checksum coverage")
    for digest, name in lines:
        if _sha(root / name) != digest:
            raise GraphEncoderError("invalid_stage6_producer_artifact", "checksum differs")
    resolved = _load_json_file(
        root / "resolved_config.json", "invalid_stage6_producer_artifact"
    )
    training = _read_jsonl(root / "training_histories.jsonl")
    checkpoints = _read_jsonl(root / "checkpoint_identities.jsonl")
    families = _read_jsonl(root / "family_records.jsonl")
    if (
        resolved.get("producer_version") != PRODUCER_VERSION
        or resolved.get("execution_record_version") != EXECUTION_RECORD_VERSION
        or resolved.get("producer_artifact_version") != PRODUCER_ARTIFACT_VERSION
        or resolved.get("producer_config") != producer_config()
        or resolved.get("checkpoint_identity_count") != len(checkpoints)
        or not isinstance(resolved.get("checkpoint_bundle_reference"), dict)
        or not str(resolved.get("job_id", "")).isdigit()
        or any(resolved.get(key) != value for key, value in ACCESS_DECLARATIONS.items())
    ):
        raise GraphEncoderError("invalid_stage6_producer_artifact", "identity differs")
    payload = {
        key: value for key, value in resolved.items()
        if key not in {
            "producer_version", "execution_record_version",
            "producer_artifact_version", "producer_config",
            "checkpoint_identity_count", *ACCESS_DECLARATIONS,
        }
    }
    payload["training_runs"] = training
    payload["family_records"] = families
    retained = tuple(payload.get("retained_seeds", ()))
    execution = payload.get("execution_evidence", {})
    if set(execution) != {
        "selected_execution_device", "runtime_identity",
        "timing_hardware_identity", "timing_version",
        "device_selected_only_by_timing", "cuda_peak_memory_bytes",
        "verification_status",
    } or execution.get("timing_version") != TIMING_VERSION or execution.get(
        "device_selected_only_by_timing"
    ) is not True or execution.get("verification_status") != "pass":
        raise GraphEncoderError(
            "invalid_stage6_producer_artifact", "execution evidence differs"
        )
    timing_selection = validate_timing_evidence(
        payload.get("timing_fallback", {}).get("evidence", {}),
        required_device=execution["selected_execution_device"],
        return_selection=True,
    )
    from .stage6_device import timing_hardware_identity
    from .stage6_timing import validate_runtime_identity
    validate_runtime_identity(
        execution["runtime_identity"], execution["selected_execution_device"]
    )
    peak_memory = execution["cuda_peak_memory_bytes"]
    if (
        retained != timing_selection["retained_seeds"]
        or execution["timing_hardware_identity"]
        != timing_selection["timing_hardware_identity"]
        or execution["timing_hardware_identity"]
        != timing_hardware_identity(execution["runtime_identity"])
        or (
            execution["selected_execution_device"] == "cpu"
            and peak_memory is not None
        )
        or (
            execution["selected_execution_device"] == "cuda:0"
            and (
                isinstance(peak_memory, bool)
                or not isinstance(peak_memory, int)
                or peak_memory <= 0
            )
        )
        or any(
            row.get("execution_device") != execution["selected_execution_device"]
            or row.get("runtime_identity") != execution["runtime_identity"]
            or row.get("timing_hardware_identity")
            != execution["timing_hardware_identity"]
            for row in checkpoints
        )
    ):
        raise GraphEncoderError(
            "invalid_stage6_producer_artifact", "device provenance differs"
        )
    expected_checkpoints = {(arm, seed) for arm in ARMS for seed in retained}
    observed_checkpoints = {(row.get("arm"), row.get("seed")) for row in checkpoints}
    if observed_checkpoints != expected_checkpoints or len(checkpoints) != len(expected_checkpoints):
        raise GraphEncoderError("invalid_stage6_producer_artifact", "checkpoint matrix")
    for row in checkpoints:
        wrapper_sha = row.get("stage6_wrapper_sha256")
        core = {key: value for key, value in row.items()
                if key != "stage6_wrapper_sha256"}
        if (
            set(row) != {*core, "stage6_wrapper_sha256"}
            or not isinstance(wrapper_sha, str) or len(wrapper_sha) != 64
            or any(character not in "0123456789abcdef" for character in wrapper_sha)
        ):
            raise GraphEncoderError(
                "invalid_stage6_producer_artifact", "wrapper hash differs"
            )
        validate_checkpoint_identity(core, expected=core)
    bundle = resolved["checkpoint_bundle_reference"]
    bundle_rows = [{key: value for key, value in row.items() if key != "bundle_path"}
                   for row in bundle.get("checkpoint_identities", ())]
    if (
        bundle.get("schema_version") != CHECKPOINT_BUNDLE_VERSION
        or bundle.get("verification_status") != "pass"
        or not isinstance(bundle.get("bundle_sha256"), str)
        or len(bundle["bundle_sha256"]) != 64
        or bundle_rows != checkpoints
        or bundle.get("source_evidence") != payload.get("source_evidence")
        or bundle.get("input_evidence") != payload.get("input_evidence")
    ):
        raise GraphEncoderError("invalid_stage6_producer_artifact", "bundle reference differs")
    scored = [score_family_record(row) for row in families]
    train_scores = {}
    for arm in ARMS:
        for seed in retained:
            values = [
                row["score"]["normalized_structural_prefix"] for row in scored
                if row["cohort"] == "train" and row["condition"] == "P_true"
                and row["arm"] == arm and row["seed"] == seed
            ]
            if len(values) != TRAIN_FAMILY_COUNT:
                raise GraphEncoderError(
                    "invalid_stage6_producer_artifact", "train ceiling matrix"
                )
            train_scores[(arm, seed)] = sum(values) / float(len(values))
    reliability = optimization_reliability(training, train_scores, retained)
    if reliability != payload.get("optimization_reliability"):
        raise GraphEncoderError(
            "invalid_stage6_producer_artifact", "optimization evidence differs"
        )
    reliability_runs = {
        (row["arm"], row["seed"]): row for row in reliability["runs"]
    }
    for row in training:
        expected_reliable = reliability_runs[
            (row["arm"], row["seed"])
        ]["pass_before_train_ceiling"]
        if row.get("optimization_reliable") is not expected_reliable:
            raise GraphEncoderError(
                "invalid_stage6_producer_artifact", "run reliability differs"
            )
    for seed in retained:
        counts = {
            row["arm"]: row["trainable_parameter_count"] for row in training
            if row["seed"] == seed
        }
        capacity = capacity_gate(counts["flat"], counts["typed_graph"])
        if any(
            row.get("capacity_parity_pass") is not capacity["pass"]
            for row in training if row["seed"] == seed
        ):
            raise GraphEncoderError(
                "invalid_stage6_producer_artifact", "capacity evidence differs"
            )
    summarize_execution(payload)
    result = {
        "producer_artifact_version": PRODUCER_ARTIFACT_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "retained_seeds": list(retained),
        "training_run_count": len(training),
        "checkpoint_identity_count": len(checkpoints),
        "family_record_count": len(families),
        "outcome_controls_validity": False,
        "producer_artifact_sha256": _sha(root / "SHA256SUMS"),
        "checkpoint_bundle_sha256": bundle["bundle_sha256"],
    }
    if return_payload:
        result["payload"] = payload
    return result


def save_stage6_checkpoint(generic_checkpoint_path, stage6_path, identity):
    """Wrap one newly produced C6 checkpoint in the Stage 6 identity."""

    if torch is None:
        raise RuntimeError("Stage 6 checkpoint writing requires PyTorch")
    validate_checkpoint_identity(identity, expected=identity)
    if _sha(generic_checkpoint_path) != identity["checkpoint_sha256"]:
        raise GraphEncoderError(
            "invalid_stage6_checkpoint", "generic checkpoint hash differs"
        )
    generic = torch.load(str(generic_checkpoint_path), map_location="cpu")
    if (
        not isinstance(generic, dict)
        or generic.get("completed_epoch") != EPOCHS
        or generic.get("selected_checkpoint_epoch") != EPOCHS
        or generic.get("encoder_type") != identity["arm"]
        or generic.get("seed") != identity["seed"]
        or generic.get("model_config") != identity["model_config"]
        or generic.get("checkpoint_schema") != identity["checkpoint_schema"]
        or generic.get("provenance", {}).get("git_commit")
        != identity["source_commit"]
        or generic.get("provenance", {}).get("device")
        != identity["execution_device"]
    ):
        raise GraphEncoderError("invalid_stage6_checkpoint", "generic payload differs")
    cuda_state = generic.get("rng_states", {}).get("torch_cuda")
    if (cuda_state is not None) is not identity["cuda_rng_preserved"]:
        raise GraphEncoderError(
            "invalid_stage6_checkpoint", "CUDA RNG preservation differs"
        )
    wrapper = {
        "version": CHECKPOINT_VERSION,
        "identity": identity,
        "training_checkpoint": generic,
    }
    final = Path(stage6_path)
    temporary = final.with_name("." + final.name + ".tmp")
    try:
        torch.save(wrapper, str(temporary))
        os.replace(str(temporary), str(final))
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
    return _sha(final)


def load_stage6_checkpoint(path, *, expected_identity, model, optimizer,
                           training_config, partition_identity,
                           restore_rng=False, execution_device="cpu",
                           portable_cpu_mapping=False,
                           expected_wrapper_sha256=None):
    """Strictly recover through the existing C6 loader into a fresh model."""

    if torch is None:
        raise RuntimeError("Stage 6 checkpoint recovery requires PyTorch")
    from .training import load_training_checkpoint

    from .stage6_device import validate_execution_device
    execution_device = validate_execution_device(execution_device)
    identity_device = expected_identity.get("execution_device")
    if execution_device != identity_device and not (
        portable_cpu_mapping and execution_device == "cpu"
    ):
        raise GraphEncoderError(
            "invalid_stage6_checkpoint", "governed map_location differs"
        )
    if expected_wrapper_sha256 is not None and (
        not isinstance(expected_wrapper_sha256, str)
        or len(expected_wrapper_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in expected_wrapper_sha256
        )
        or _sha(path) != expected_wrapper_sha256
    ):
        raise GraphEncoderError(
            "invalid_stage6_checkpoint", "Stage 6 wrapper hash differs"
        )
    wrapper = torch.load(str(path), map_location=execution_device)
    if not isinstance(wrapper, dict) or set(wrapper) != {
        "version", "identity", "training_checkpoint"
    } or wrapper.get("version") != CHECKPOINT_VERSION:
        raise GraphEncoderError("invalid_stage6_checkpoint", "wrapper differs")
    validate_checkpoint_identity(wrapper["identity"], expected=expected_identity)
    descriptor, temporary_name = tempfile.mkstemp(suffix=".pt")
    os.close(descriptor)
    try:
        torch.save(wrapper["training_checkpoint"], temporary_name)
        resume = load_training_checkpoint(
            temporary_name,
            model=model,
            optimizer=optimizer,
            training_config=training_config,
            partition_identity=partition_identity,
            expected_code_revision=expected_identity["source_commit"],
            restore_rng=restore_rng,
            expected_selected_checkpoint_epoch=EPOCHS,
            map_location=execution_device,
        )
    finally:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
    if resume.completed_epoch != EPOCHS:
        raise GraphEncoderError("invalid_stage6_checkpoint", "epoch differs")
    return resume


def _edge_tuples(graph):
    return tuple(sorted({
        (int(edge.source), int(edge.destination), int(edge.edge_type_id))
        for edge in graph.directed_typed_edges
    }))


def _operation_ids(node_type_ids):
    from prototype.model_data.vocab import NODE_TYPES

    return tuple(
        int(value) for value in node_type_ids
        if NODE_TYPES.tokens[int(value)] in ("extrude", "revolve")
    )


def _grammar_valid(node_ids, requested_node_count):
    from prototype.node_grammar import V5_NODE_GRAMMAR, validate_complete_node_sequence

    try:
        validate_complete_node_sequence(
            tuple(node_ids), int(requested_node_count), V5_NODE_GRAMMAR
        )
    except Exception:
        return False
    return True


def _structural_record(k, graph, *, canonicalized, prefix_constructed=True):
    from prototype.graph_baseline.graph_contract import graph_edge_class_id
    from .metrics import ATTACHMENT_EDGE_NAMES

    edges = _edge_tuples(graph)
    depends_id = graph_edge_class_id("depends_on")
    attachment_ids = {graph_edge_class_id(name) for name in ATTACHMENT_EDGE_NAMES}
    node_ids = tuple(int(value) for value in graph.node_type_ids)
    return {
        "k": int(k),
        "grammar_valid_complete": _grammar_valid(node_ids, len(node_ids)),
        "canonicalization_succeeded": bool(canonicalized),
        "prefix_construction_succeeded": bool(prefix_constructed),
        "requested_node_count": len(node_ids),
        "node_type_ids": list(node_ids),
        "operation_type_ids": list(_operation_ids(node_ids)),
        "canonical_edges": [list(item) for item in edges],
        "depends_on_edges": [list(item) for item in edges if item[2] == depends_id],
        "ownership_attachment_edges": [
            list(item) for item in edges if item[2] in attachment_ids
        ],
    }


def structural_records_from_prediction(prediction, target):
    """Extract geometry-free autonomous and target prefix records."""

    from prototype.graph_baseline.graph_contract import (
        graph_from_reconstruction_target,
        model_data_edge_id_from_graph_class,
    )
    from .canonicalization import GraphCanonicalizationInput, canonicalize_graph
    from .metrics import _canonical_prefix_prediction

    operation_count = len(target.operation_sequence)
    if operation_count not in (1, 2):
        raise GraphEncoderError("invalid_stage6_target", "operation count differs")
    expected_graph = graph_from_reconstruction_target(target)
    target_records = []
    for k in range(1, operation_count + 1):
        stop = int(target.operation_sequence[k - 1]) + 1
        target_records.append(_structural_record(
            k, _graph_prefix(expected_graph, stop),
            canonicalized=True,
        ))
    graph = prediction.graph
    narrow = GraphCanonicalizationInput(
        graph.node_type_ids,
        (
            tuple(item.source for item in graph.directed_typed_edges),
            tuple(item.destination for item in graph.directed_typed_edges),
        ),
        tuple(model_data_edge_id_from_graph_class(item.edge_type_id)
              for item in graph.directed_typed_edges),
        tuple(node.categorical_ids for node in prediction.node_prediction.raw_nodes),
        tuple(node.normalized_geometry for node in prediction.node_prediction.raw_nodes),
        tuple(node.derived_geometry_mask for node in prediction.node_prediction.raw_nodes),
    )
    try:
        canonical = canonicalize_graph(narrow)
    except Exception:
        return (
            [_failed_structural_record(
                k, target_records[k - 1]["requested_node_count"],
                canonicalized=False, grammar_valid=False,
            ) for k in range(1, operation_count + 1)],
            target_records,
            len(_operation_ids(graph.node_type_ids)),
        )
    predicted_records = []
    for k in range(1, operation_count + 1):
        if len(canonical.operation_sequence) < k:
            predicted_records.append(_failed_structural_record(
                k, target_records[k - 1]["requested_node_count"],
                canonicalized=True, grammar_valid=False,
            ))
            continue
        stop = canonical.operation_sequence[k - 1] + 1
        grammar_valid = _grammar_valid(canonical.node_type_ids[:stop], stop)
        try:
            prefix = _canonical_prefix_prediction(
                prediction, canonical.canonical_to_old[:stop]
            )
        except Exception:
            predicted_records.append(_failed_structural_record(
                k, stop, canonicalized=True, grammar_valid=grammar_valid,
            ))
            continue
        predicted_records.append(_structural_record(
            k, prefix.graph, canonicalized=True
        ))
    return predicted_records, target_records, len(canonical.operation_sequence)


def _graph_prefix(graph, stop):
    """Return the minimal graph-shaped immutable prefix used by the scorer."""

    from dataclasses import dataclass

    @dataclass(frozen=True)
    class PrefixGraph:
        node_type_ids: tuple
        directed_typed_edges: tuple

    return PrefixGraph(
        tuple(graph.node_type_ids[:stop]),
        tuple(edge for edge in graph.directed_typed_edges
              if edge.source < stop and edge.destination < stop),
    )


def _failed_structural_record(k, requested_node_count, *, canonicalized,
                              grammar_valid):
    return {
        "k": int(k),
        "grammar_valid_complete": bool(grammar_valid),
        "canonicalization_succeeded": bool(canonicalized),
        "prefix_construction_succeeded": False,
        "requested_node_count": int(requested_node_count),
        "node_type_ids": [],
        "operation_type_ids": [],
        "canonical_edges": [],
        "depends_on_edges": [],
        "ownership_attachment_edges": [],
    }


def family_record_from_prediction(item, target, *, template, representation_identity,
                                  cohort, arm, seed):
    """Join a target only after autonomous generation and preserve all outcomes."""

    from .metrics import _score_one

    if item.condition not in CONDITIONS or arm not in ARMS or cohort not in COHORTS:
        raise GraphEncoderError("invalid_stage6_alignment", "identity differs")
    if item.family_id == "" or template not in TEMPLATES:
        raise GraphEncoderError("invalid_stage6_alignment", "family/template differs")
    predicted, expected, predicted_operation_count = structural_records_from_prediction(
        item.constrained_prediction, target
    )
    report = _score_one(item, target, grid_identity=True)
    full_predicted = predicted[-1]
    full_expected = expected[-1]
    predicted_edges = {tuple(edge) for edge in full_predicted["depends_on_edges"]}
    expected_edges = {tuple(edge) for edge in full_expected["depends_on_edges"]}
    attachment_predicted = {
        tuple(edge) for edge in full_predicted["ownership_attachment_edges"]
    }
    attachment_expected = {
        tuple(edge) for edge in full_expected["ownership_attachment_edges"]
    }
    outcome = item.converted_prediction
    converted = None if outcome.raised_failure else outcome.result
    analytic = bool(converted is not None and converted.controlled_domain.valid)
    strict = bool(converted is not None and converted.reconstruction_target.valid)
    return {
        "family_id": item.family_id,
        "template": template,
        "cohort": cohort,
        "arm": arm,
        "seed": int(seed),
        "condition": item.condition,
        "representation_variant_id": representation_identity,
        "batch_identity": item.batch_identity,
        "memory_source_family_id": item.memory_source_family_id,
        "target_operation_count": len(target.operation_sequence),
        "predicted_operation_group_count": predicted_operation_count,
        "autonomous": True,
        "target_joined_after_generation": True,
        "raw_prediction_preserved": item.raw_prediction is not None,
        "constrained_prediction_preserved": item.constrained_prediction is not None,
        "predicted_structural_prefixes": predicted,
        "target_structural_prefixes": expected,
        "secondary": {
            "exact_complete_node_sequence": (
                full_predicted["node_type_ids"] == full_expected["node_type_ids"]
            ),
            "exact_operation_type_sequence": (
                full_predicted["operation_type_ids"]
                == full_expected["operation_type_ids"]
            ),
            "exact_graph": bool(report["exact_graph"]),
            "depends_on_true_positive": len(predicted_edges & expected_edges),
            "depends_on_predicted": len(predicted_edges),
            "depends_on_target": len(expected_edges),
            "depends_on_exact": predicted_edges == expected_edges,
            "reference_attachment_exact": attachment_predicted == attachment_expected,
            "strict_conversion": strict,
            "analytic_validity": analytic,
            "geometry_metrics": report["geometry_error_by_channel_family"],
            "operation_magnitude_metrics": report.get("grid_magnitude", {
                "available": False, "reason": "not_grid_parameterized"
            }),
        },
    }


def generate_autonomous_records(model, examples, *, cohort, arm, seed,
                                execution_device="cpu"):
    """Generate all memory conditions before constructing any target mapping."""

    from .autonomous import autonomous_input_from_paired, run_autonomous_evaluation
    from .batching import build_paired_batch

    ordered = tuple(sorted(examples, key=lambda item: item.physical_family_id))
    expected_count = TRAIN_FAMILY_COUNT if cohort == "train" else DEVELOPMENT_FAMILY_COUNT
    if len(ordered) != expected_count or len({item.physical_family_id for item in ordered}) != expected_count:
        raise GraphEncoderError("invalid_stage6_family_count", cohort)
    input_batches = tuple(
        autonomous_input_from_paired(
            build_paired_batch(ordered[start:start + BATCH_SIZE]), arm,
            execution_device=execution_device,
        )
        for start in range(0, len(ordered), BATCH_SIZE)
    )
    autonomous = run_autonomous_evaluation(
        model, input_batches, seed=seed, shuffle_scope="batch",
        execution_device=execution_device,
    )
    # This is the first point targets enter the evaluation path.
    targets = {item.physical_family_id: item.target for item in ordered}
    templates = {
        item.physical_family_id: item.metadata.operation_template for item in ordered
    }
    representations = {
        item.physical_family_id: hashlib.sha256(
            ("\n".join(item.metadata.sample_ids) + "\n").encode("utf-8")
        ).hexdigest()
        for item in ordered
    }
    records = []
    for condition in autonomous.conditions:
        if not condition.available or len(condition.predictions) != expected_count:
            raise GraphEncoderError("invalid_stage6_memory_condition", condition.condition)
        for prediction in condition.predictions:
            if prediction.family_id not in targets:
                raise GraphEncoderError("invalid_stage6_alignment", prediction.family_id)
            records.append(family_record_from_prediction(
                prediction, targets[prediction.family_id],
                template=templates[prediction.family_id],
                representation_identity=representations[prediction.family_id],
                cohort=cohort,
                arm=arm, seed=seed,
            ))
    return records


def _training_run_record(training, *, arm, seed, parameter_count,
                         capacity_pass, execution_device="cpu"):
    losses = [float(row.mean_loss) for row in training.epoch_records]
    gradient_norms = []
    for row in training.epoch_records:
        values = tuple(float(value) for value in row.pre_clip_gradient_norms)
        if not values:
            raise GraphEncoderError("invalid_stage6_optimization", "gradient history empty")
        gradient_norms.append(max(values))
    evaluations = training.plateau_state.evaluations
    final_relative = None if not evaluations else float(evaluations[-1].relative_improvement)
    return {
        "arm": arm,
        "seed": seed,
        "fresh_initialization": True,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "optimizer": "AdamW",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "gradient_clip_norm": CLIP_NORM,
        "checkpoint_epoch": EPOCHS,
        "early_stopping": False,
        "development_used_for_selection": False,
        "warm_start": False,
        "training_family_count": TRAIN_FAMILY_COUNT,
        "trainable_parameter_count": parameter_count,
        "capacity_parity_pass": bool(capacity_pass),
        "optimization_reliable": False,
        "final_loss": losses[-1],
        "plateau_relative_improvement": final_relative,
        "completed_epoch": training.completed_epoch,
        "optimizer_steps": training.optimizer_steps,
        "training_example_presentations": training.training_example_presentations,
        "epoch_losses": losses,
        "epoch_gradient_norms": gradient_norms,
        "first_plateau_epoch": training.plateau_state.first_plateau_epoch,
        "selected_checkpoint_is_fixed_epoch_200": True,
        "checkpoint_reuse": False,
        "execution_device": execution_device,
    }


def _train_arm(model, train_examples, *, work_root, repository_root,
               expected_commit, train_input_evidence, parameter_count,
               capacity_pass, execution_device, runtime_identity,
               timing_hardware):
    from .config import GE1TrainingConfig
    from .model import build_ge1_model
    from .provenance import training_partition_identity
    from .training import run_ge1_training

    arm = model.config.encoder
    seed = model.config.seed
    training_config = GE1TrainingConfig()
    training_config.validate()
    checkpoint_dir = Path(work_root) / "generic" / "{}-{}".format(arm, seed)
    training = run_ge1_training(
        model,
        train_examples,
        training_config=training_config,
        checkpoint_directory=checkpoint_dir,
        repository_root=repository_root,
        expected_commit=expected_commit,
        final_epoch=EPOCHS,
        engineering_smoke=False,
        checkpoint_epochs=(EPOCHS,),
        fixed_protocol_final_epoch=EPOCHS,
        checkpoint_coordinate="epoch",
        selected_checkpoint_epoch=EPOCHS,
        execution_device=execution_device,
    )
    if (
        training.completed_epoch != EPOCHS
        or len(training.epoch_records) != EPOCHS
        or len(training.checkpoint_paths) != 1
        or training.selected_experimental_checkpoint != training.checkpoint_paths[0]
    ):
        raise GraphEncoderError("invalid_stage6_training", "epoch/checkpoint differs")
    generic_path = Path(training.checkpoint_paths[0])
    generic_sha = _sha(generic_path)
    partition_identity = training_partition_identity(
        tuple(item.physical_family_id for item in train_examples)
    )
    provenance = training.provenance
    arithmetic = {
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "optimizer": "AdamW",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "gradient_clip_norm": CLIP_NORM,
        "optimizer_steps": training.optimizer_steps,
        "training_example_presentations": training.training_example_presentations,
    }
    identity = checkpoint_identity(
        arm=arm,
        seed=seed,
        source_commit=expected_commit,
        source_digest=provenance.source_tree_sha256,
        model_config=model.config.to_dict(),
        partition_identity=partition_identity,
        partition_hashes={
            "index_identity_sha256": train_input_evidence["index_identity_sha256"],
            "payload_digests_sha256": train_input_evidence[
                "observed_payload_digests_sha256"
            ],
        },
        parameter_count=parameter_count,
        training_arithmetic=arithmetic,
        checkpoint_sha256=generic_sha,
        execution_device=execution_device,
        runtime_identity=runtime_identity,
        timing_hardware_identity=timing_hardware,
        cuda_rng_preserved=execution_device == "cuda:0",
    )
    wrapper_path = Path(work_root) / "stage6-{}-seed{}.pt".format(arm, seed)
    wrapper_sha = save_stage6_checkpoint(generic_path, wrapper_path, identity)
    identity = {**identity, "stage6_wrapper_sha256": wrapper_sha}
    training_record = _training_run_record(
        training, arm=arm, seed=seed, parameter_count=parameter_count,
        capacity_pass=capacity_pass, execution_device=execution_device,
    )
    return {
        "trained_model": model,
        "training_record": training_record,
        "identity": identity,
        "wrapper_path": str(wrapper_path),
        "training_config": training_config,
        "partition_identity": partition_identity,
    }


def _recover_arm(trained, *, execution_device):
    from .model import build_ge1_model

    model = trained["trained_model"]
    identity = trained["identity"]
    expected_for_loader = {key: value for key, value in identity.items()
                           if key != "stage6_wrapper_sha256"}
    fresh = build_ge1_model(model.config).to(execution_device)
    optimizer = torch.optim.AdamW(
        fresh.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    resume = load_stage6_checkpoint(
        trained["wrapper_path"],
        expected_identity=expected_for_loader,
        model=fresh,
        optimizer=optimizer,
        training_config=trained["training_config"],
        partition_identity=trained["partition_identity"],
        restore_rng=True,
        execution_device=execution_device,
        expected_wrapper_sha256=identity["stage6_wrapper_sha256"],
    )
    if resume.completed_epoch != EPOCHS:
        raise GraphEncoderError("invalid_stage6_checkpoint", "recovery differs")
    expected_state = model.state_dict()
    recovered_state = fresh.state_dict()
    if list(expected_state) != list(recovered_state) or any(
        not torch.equal(expected_state[name], recovered_state[name])
        for name in expected_state
    ):
        raise GraphEncoderError("invalid_stage6_checkpoint", "tensor reload differs")
    if not _state_tree_equal(
        optimizer.state_dict(), resume.payload.get("optimizer_state")
    ):
        raise GraphEncoderError(
            "invalid_stage6_checkpoint", "optimizer reload differs"
        )
    rng_states = resume.payload.get("rng_states", {})
    expected_cpu_rng = rng_states.get("torch_cpu")
    if hasattr(expected_cpu_rng, "cpu"):
        expected_cpu_rng = expected_cpu_rng.cpu()
    if expected_cpu_rng is None or not torch.equal(
        torch.get_rng_state(), expected_cpu_rng
    ):
        raise GraphEncoderError(
            "invalid_stage6_checkpoint", "CPU RNG reload differs"
        )
    if execution_device == "cuda:0":
        expected_cuda_rng = rng_states.get("torch_cuda")
        observed_cuda_rng = torch.cuda.get_rng_state_all()
        if (
            not isinstance(expected_cuda_rng, (tuple, list))
            or len(expected_cuda_rng) != len(observed_cuda_rng)
            or any(
                not torch.equal(observed, expected.cpu())
                for observed, expected in zip(observed_cuda_rng, expected_cuda_rng)
            )
        ):
            raise GraphEncoderError(
                "invalid_stage6_checkpoint", "CUDA RNG reload differs"
            )
    return fresh


def _state_tree_equal(left, right):
    if torch is not None and torch.is_tensor(left):
        return torch.is_tensor(right) and torch.equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and set(left) == set(right) and all(
            _state_tree_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (tuple, list)):
        return isinstance(right, type(left)) and len(left) == len(right) and all(
            _state_tree_equal(a, b) for a, b in zip(left, right)
        )
    return left == right


def run_stage6_producer(*, train_index, train_root, development_index,
                        development_root, timing_evidence, output_dir,
                        checkpoint_bundle_dir, repository_root, expected_commit,
                        job_id, allow_fallback=False,
                        required_execution_device=None):
    """Execute the governed lifecycle after all prospective gates pass."""

    from .pilot import _source_identity, _verify_source_unchanged

    job_id = _job_id(job_id)
    timing_selection = validate_timing_evidence(
        timing_evidence, allow_fallback=allow_fallback,
        required_device=required_execution_device, return_selection=True,
    )
    retained_seeds = timing_selection["retained_seeds"]
    execution_device = timing_selection["execution_device"]
    _require_runtime(execution_device)
    from .stage6_device import (
        configure_stage6_runtime, require_timing_hardware,
    )
    runtime_identity = configure_stage6_runtime(
        torch, execution_device, seed=FULL_SEEDS[0]
    )
    require_timing_hardware(
        runtime_identity, timing_selection["timing_hardware_identity"]
    )
    if execution_device == "cuda:0":
        torch.cuda.reset_peak_memory_stats(0)
    source = _source_identity(repository_root, expected_commit)
    if source["detached_head"] is not True or source["git_dirty"] is not False:
        raise GraphEncoderError("invalid_stage6_source", "detached clean source required")
    source_evidence = {
        "source_commit": source["git_commit"],
        "source_tree_sha256": source["source_tree_sha256"],
        "source_clean": True,
        "detached_head": True,
        "verification_status": "pass",
    }
    if timing_evidence["source_identity"] != {
        "source_commit": source_evidence["source_commit"],
        "source_tree_sha256": source_evidence["source_tree_sha256"],
    }:
        raise GraphEncoderError(
            "invalid_stage6_timing", "timing source identity differs"
        )
    from .model import build_matched_ge1_models
    from .stage6_narrow_loader import (
        NARROW_LOADER_VERSION, load_stage6_development, load_stage6_train,
    )
    from .stage6_structure_only import (
        score_family_record, structure_memory_gate_for_cohort,
    )

    train_examples, train_input_evidence = load_stage6_train(train_index, train_root)
    if len(train_examples) != TRAIN_FAMILY_COUNT:
        raise GraphEncoderError("invalid_stage6_family_count", "train differs")
    if timing_evidence["train_input_identity"] != {
        "index_identity_sha256": train_input_evidence["index_identity_sha256"],
        "payload_digests_sha256": train_input_evidence[
            "observed_payload_digests_sha256"
        ],
    }:
        raise GraphEncoderError(
            "invalid_stage6_timing", "timing train input identity differs"
        )
    training_runs = []
    trained_runs = []
    recovered_models = []
    train_records = []
    work_root = Path(output_dir).with_name(
        Path(output_dir).name + ".work.incomplete-" + job_id
    )
    if work_root.exists() or work_root.is_symlink():
        raise GraphEncoderError("unsafe_stage6_producer_output", str(work_root))
    work_root.mkdir()
    models = []
    for seed in retained_seeds:
        flat, graph = build_matched_ge1_models(
            seed,
            operation_magnitude_parameterization=(
                GRID_SOFTMAX_OPERATION_MAGNITUDE_PARAMETERIZATION
            ),
            node_generation_identity=(
                AUTONOMOUS_STOP_NODE_GENERATION_IDENTITY
            ),
        )
        flat = flat.to(execution_device)
        graph = graph.to(execution_device)
        counts = {
            "flat": sum(parameter.numel() for parameter in flat.parameters()
                        if parameter.requires_grad),
            "typed_graph": sum(parameter.numel() for parameter in graph.parameters()
                              if parameter.requires_grad),
        }
        capacity = capacity_gate(counts["flat"], counts["typed_graph"])
        if not capacity["pass"]:
            raise GraphEncoderError("invalid_stage6_capacity", "capacity gate failed")
        models.extend((model, counts[model.config.encoder], capacity["pass"])
                      for model in (flat, graph))
    # All six/four training runs complete before any strict recovery.
    for model, parameter_count, capacity_pass in models:
        trained = _train_arm(
            model, train_examples, work_root=work_root,
            repository_root=repository_root, expected_commit=expected_commit,
            train_input_evidence=train_input_evidence,
            parameter_count=parameter_count, capacity_pass=capacity_pass,
            execution_device=execution_device,
            runtime_identity=runtime_identity,
            timing_hardware=timing_selection["timing_hardware_identity"],
        )
        trained_runs.append(trained)
        training_runs.append(trained["training_record"])
    # Every trained checkpoint is strictly recovered before train inference.
    for trained in trained_runs:
        recovered_models.append((
            _recover_arm(trained, execution_device=execution_device),
            trained["identity"]["arm"],
            trained["identity"]["seed"],
        ))
    for model, arm, seed in recovered_models:
        train_records.extend(generate_autonomous_records(
            model, train_examples, cohort="train", arm=arm, seed=seed,
            execution_device=execution_device,
        ))
    scored_train = [score_family_record(row) for row in train_records]
    train_scores = {}
    for arm in ARMS:
        for seed in retained_seeds:
            values = [
                row["score"]["normalized_structural_prefix"] for row in scored_train
                if row["condition"] == "P_true"
                and row["arm"] == arm and row["seed"] == seed
            ]
            train_scores[(arm, seed)] = sum(values) / float(len(values))
    reliability = optimization_reliability(
        training_runs, train_scores, retained_seeds
    )
    reliability_by_run = {
        (row["arm"], row["seed"]): row for row in reliability["runs"]
    }
    for row in training_runs:
        key = (row["arm"], row["seed"])
        row["optimization_reliable"] = reliability_by_run[key][
            "pass_before_train_ceiling"
        ]
    train_memory = structure_memory_gate_for_cohort(
        scored_train, retained_seeds, "train"
    )
    if not reliability["pass"] or not all(row["pass"] for row in train_memory.values()):
        raise GraphEncoderError(
            "stage6_train_reliability_failure",
            "development remains closed after failed train-side gates",
        )
    # Development is first validated and loaded only after all train-side work.
    development_examples, development_input_evidence = load_stage6_development(
        development_index, development_root
    )
    if len(development_examples) != DEVELOPMENT_FAMILY_COUNT:
        raise GraphEncoderError("invalid_stage6_family_count", "development differs")
    development_records = []
    for model, arm, seed in recovered_models:
        development_records.extend(generate_autonomous_records(
            model, development_examples, cohort="development", arm=arm,
            seed=seed, execution_device=execution_device,
        ))
    family_records = train_records + development_records
    input_evidence = {
        "input_policy_version": INPUT_POLICY_VERSION,
        "narrow_loader_version": NARROW_LOADER_VERSION,
        "train": train_input_evidence,
        "development": development_input_evidence,
        "verification_status": "pass",
    }
    bundle_rows = [{
        "path": row["wrapper_path"], "identity": row["identity"]
    } for row in trained_runs]
    bundle = create_checkpoint_bundle(
        bundle_rows, checkpoint_bundle_dir, job_id=job_id,
        source_evidence=source_evidence, input_evidence=input_evidence,
        retained_seeds=retained_seeds,
    )
    fallback = tuple(retained_seeds) == FALLBACK_SEEDS
    payload = {
        "schema_version": INPUT_VERSION,
        "retained_seeds": list(retained_seeds),
        "timing_fallback": {
            "invoked": fallback,
            "prospective": True,
            "observed_results_used": False,
            "reason": timing_selection["fallback_reason"] if fallback else None,
            "evidence": timing_evidence,
        },
        "execution_evidence": {
            "selected_execution_device": execution_device,
            "runtime_identity": runtime_identity,
            "timing_hardware_identity": timing_selection[
                "timing_hardware_identity"
            ],
            "timing_version": TIMING_VERSION,
            "device_selected_only_by_timing": True,
            "cuda_peak_memory_bytes": (
                int(torch.cuda.max_memory_allocated(0))
                if execution_device == "cuda:0" else None
            ),
            "verification_status": "pass",
        },
        "partition": {
            "training_name": "operation_template_train",
            "training_family_count": TRAIN_FAMILY_COUNT,
            "development_name": "operation_template_development",
            "development_family_count": DEVELOPMENT_FAMILY_COUNT,
            "rr_accessed": False,
            "er_accessed": False,
        },
        "training_runs": training_runs,
        "family_records": family_records,
        "provenance_valid": True,
        "artifact_inputs_valid": True,
        "producer_protocol": PRODUCER_VERSION,
        "execution_record_version": EXECUTION_RECORD_VERSION,
        "source_evidence": source_evidence,
        "input_evidence": input_evidence,
        "checkpoint_bundle_reference": bundle,
        "optimization_reliability": reliability,
        **ACCESS_DECLARATIONS,
    }
    summarize_execution(payload)
    _verify_source_unchanged(source, repository_root, expected_commit)
    result = create_producer_artifact(
        payload, [row["identity"] for row in trained_runs], output_dir,
        job_id=job_id, checkpoint_bundle_reference=bundle,
    )
    shutil.rmtree(str(work_root))
    return result


def _require_runtime(execution_device="cpu"):
    if torch is None:
        raise RuntimeError("Stage 6 producer requires PyTorch")
    if sys.version_info[:3] != (3, 8, 13):
        raise GraphEncoderError("environment_mismatch", "Python 3.8.13 required")
    if str(torch.__version__).split("+")[0] != "1.11.0":
        raise GraphEncoderError("environment_mismatch", "PyTorch 1.11.0 required")
    if torch.get_num_threads() != 1:
        raise GraphEncoderError("environment_mismatch", "one thread required")
    from .stage6_device import validate_execution_device
    device = validate_execution_device(execution_device)
    if device == "cuda:0" and (
        not torch.cuda.is_available() or torch.cuda.device_count() != 1
    ):
        raise GraphEncoderError("stage6_cuda_unavailable", "cuda:0 required")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-index", required=True)
    parser.add_argument("--train-root", required=True)
    parser.add_argument("--development-index", required=True)
    parser.add_argument("--development-root", required=True)
    parser.add_argument("--timing-evidence", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint-bundle-dir", required=True)
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--job-id", default=os.environ.get("SLURM_JOB_ID"))
    parser.add_argument("--allow-fallback", action="store_true")
    parser.add_argument(
        "--require-selected-device", choices=("cpu", "cuda:0"), required=True
    )
    args = parser.parse_args(argv)
    timing = _load_json_file(args.timing_evidence, "invalid_stage6_timing")
    try:
        result = run_stage6_producer(
            train_index=args.train_index,
            train_root=args.train_root,
            development_index=args.development_index,
            development_root=args.development_root,
            timing_evidence=timing,
            output_dir=args.output_dir,
            checkpoint_bundle_dir=args.checkpoint_bundle_dir,
            repository_root=args.repository_root,
            expected_commit=args.expected_commit,
            job_id=args.job_id,
            allow_fallback=args.allow_fallback,
            required_execution_device=args.require_selected_device,
        )
    except GraphEncoderError as exc:
        if exc.code == "stage6_train_reliability_failure":
            print(_canonical({
                "event": "stage6_structure_only_producer_scientific_failure",
                "code": exc.code, "detail": exc.detail,
            }), flush=True)
            raise SystemExit(42)
        raise
    print(_canonical({"event": "stage6_structure_only_producer_completed", **result}),
          flush=True)


if __name__ == "__main__":
    main()
