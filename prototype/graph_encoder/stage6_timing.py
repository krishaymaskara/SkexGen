"""Prospective CPU/CUDA hardware timing for governed ADR-0014 selection."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time

try:
    import torch
except ImportError:  # Pure timing contracts remain locally testable.
    torch = None

from .errors import GraphEncoderError
from .decoder_contract import (
    GRID_ORDINAL_OPERATION_MAGNITUDE_PARAMETERIZATION,
    LEGACY_NODE_GENERATION_IDENTITY,
    NODE_GENERATION_IDENTITIES,
)
from .stage6_device import (
    CUBLAS_WORKSPACE_CONFIG,
    DEVICE_POLICY_VERSION,
    SUPPORTED_EXECUTION_DEVICES,
    configure_stage6_runtime,
    timing_hardware_identity,
    validate_execution_device,
)
from .stage6_structure_only import ARMS, FALLBACK_SEEDS, FULL_SEEDS, PROTOCOL_VERSION


TIMING_VERSION = "GE1-STAGE6-STRUCTURE-ONLY-TIMING-v2"
TIMING_MEASUREMENT_VERSION = "GE1-STAGE6-STRUCTURE-ONLY-TIMING-MEASUREMENT-v2"
TIMING_ARTIFACT_VERSION = "GE1-STAGE6-STRUCTURE-ONLY-TIMING-ARTIFACT-v2"
SELECTION_ALGORITHM = "GE1-STAGE6-FASTEST-FEASIBLE-DEVICE-v1"
TIMED_EPOCHS = 5
TIMED_REPETITIONS = 3
PRODUCTION_EPOCHS = 200
CONTINGENCY_FRACTION = 0.20
TIMING_FILES = ("timing.json", "artifact_manifest.json", "SHA256SUMS")


def _canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    )


def _finite_positive(value, label):
    if (
        isinstance(value, bool) or not isinstance(value, (int, float))
        or not math.isfinite(float(value)) or float(value) <= 0.0
    ):
        raise GraphEncoderError("invalid_stage6_timing", label)
    return float(value)


def _hex(value, length, label):
    if (
        not isinstance(value, str) or len(value) != length
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise GraphEncoderError("invalid_stage6_timing", label)
    return value


def _job_id(value):
    value = str(value)
    if not value.isdigit() or int(value) <= 0:
        raise GraphEncoderError("invalid_stage6_job_id", "positive decimal job ID required")
    return value


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path, code="invalid_stage6_timing"):
    target = Path(path)
    if target.is_symlink() or not target.is_file():
        raise GraphEncoderError(code, "JSON input is not regular")
    raw = target.read_text(encoding="utf-8")
    try:
        value = json.loads(
            raw,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ValueError("nonfinite " + constant)
            ),
        )
    except (ValueError, json.JSONDecodeError) as exc:
        raise GraphEncoderError(code, "malformed JSON") from exc
    if raw != _canonical(value) + "\n":
        raise GraphEncoderError(code, "noncanonical JSON")
    return value


def _write(path, text):
    target = Path(path)
    if target.exists() or target.is_symlink():
        raise GraphEncoderError("unsafe_stage6_timing_output", str(target))
    with target.open("xb") as stream:
        stream.write(text.encode("utf-8"))
        stream.flush()
        os.fsync(stream.fileno())


def validate_runtime_identity(record, device):
    required = {
        "device_policy_version", "execution_device", "python_version",
        "torch_version", "torch_cuda_build_version", "cuda_available",
        "visible_cuda_device_count", "gpu", "cudnn_version",
        "deterministic_algorithms", "tf32_matmul_allowed",
        "tf32_cudnn_allowed", "cublas_workspace_config", "cpu_threads",
        "cpu_host", "cpu_processor", "platform",
        "cuda_rng_preservation_required",
    }
    if not isinstance(record, dict) or set(record) != required:
        raise GraphEncoderError("invalid_stage6_timing", "runtime fields differ")
    if (
        record["device_policy_version"] != DEVICE_POLICY_VERSION
        or record["execution_device"] != device
        or record["python_version"] != "3.8.13"
        or str(record["torch_version"]).split("+")[0] != "1.11.0"
        or not isinstance(record["cuda_available"], bool)
        or isinstance(record["visible_cuda_device_count"], bool)
        or not isinstance(record["visible_cuda_device_count"], int)
        or record["visible_cuda_device_count"] < 0
        or (
            record["cuda_available"]
            and record["visible_cuda_device_count"] < 1
        )
        or (
            not record["cuda_available"]
            and record["visible_cuda_device_count"] != 0
        )
        or record["cpu_threads"] != 1
        or not isinstance(record["cpu_host"], str)
        or not record["cpu_host"]
        or not isinstance(record["cpu_processor"], str)
        or not isinstance(record["platform"], str)
        or not record["platform"]
        or (
            record["cudnn_version"] is not None
            and (
                isinstance(record["cudnn_version"], bool)
                or not isinstance(record["cudnn_version"], int)
            )
        )
        or record["deterministic_algorithms"] is not True
        or record["tf32_matmul_allowed"] is not False
        or record["tf32_cudnn_allowed"] is not False
    ):
        raise GraphEncoderError("invalid_stage6_timing", "runtime identity differs")
    if device == "cuda:0":
        gpu = record.get("gpu")
        if (
            record["cuda_available"] is not True
            or record["visible_cuda_device_count"] != 1
            or record["torch_cuda_build_version"] is None
            or record["cublas_workspace_config"] != CUBLAS_WORKSPACE_CONFIG
            or record["cuda_rng_preservation_required"] is not True
            or not isinstance(gpu, dict)
            or set(gpu) != {
                "visible_device_identity", "name", "compute_capability",
                "total_memory_bytes",
            }
            or not gpu["name"]
            or not isinstance(gpu["visible_device_identity"], str)
            or not gpu["visible_device_identity"]
            or not isinstance(gpu["compute_capability"], list)
            or len(gpu["compute_capability"]) != 2
            or any(isinstance(value, bool) or not isinstance(value, int)
                   for value in gpu["compute_capability"])
            or isinstance(gpu["total_memory_bytes"], bool)
            or not isinstance(gpu["total_memory_bytes"], int)
            or gpu["total_memory_bytes"] <= 0
        ):
            raise GraphEncoderError("invalid_stage6_timing", "CUDA runtime differs")
    elif record["gpu"] is not None or record["cuda_rng_preservation_required"] is not False:
        raise GraphEncoderError("invalid_stage6_timing", "CPU runtime differs")
    return True


def _measurement_record(raw):
    if not isinstance(raw, dict) or set(raw) != {"warmup", "timed"}:
        raise GraphEncoderError("invalid_stage6_timing", "raw timing fields differ")
    warmup = _finite_positive(raw["warmup"], "warmup duration differs")
    timed = tuple(raw["timed"])
    if len(timed) != TIMED_REPETITIONS:
        raise GraphEncoderError("invalid_stage6_timing", "timed repetition count differs")
    rows = []
    for index, duration in enumerate(timed, 1):
        duration = _finite_positive(duration, "timed duration differs")
        rows.append({
            "measurement_index": index,
            "fresh_model": True,
            "warm_start": False,
            "checkpoint_reuse": False,
            "timed_epochs": TIMED_EPOCHS,
            "duration_seconds": duration,
            "seconds_per_epoch": duration / float(TIMED_EPOCHS),
        })
    conservative = max(row["seconds_per_epoch"] for row in rows)
    return {
        "warmup": {
            "fresh_model": True,
            "included_in_measurement": False,
            "epochs": TIMED_EPOCHS,
            "duration_seconds": warmup,
        },
        "timed_measurements": rows,
        "conservative_seconds_per_epoch": conservative,
        "conservative_rule": "maximum_of_three_fresh_measurements",
    }


def build_timing_record(*, source_commit, source_tree_sha256,
                        train_index_identity_sha256,
                        train_payload_digests_sha256, runtime_identities,
                        raw_measurements, available_wall_seconds_by_device,
                        node_generation_identity=LEGACY_NODE_GENERATION_IDENTITY):
    """Build one outcome-free timing-v2 record and select device/seeds."""

    _hex(source_commit, 40, "source commit")
    _hex(source_tree_sha256, 64, "source digest")
    _hex(train_index_identity_sha256, 64, "train index digest")
    _hex(train_payload_digests_sha256, 64, "train payload digest")
    if node_generation_identity not in NODE_GENERATION_IDENTITIES:
        raise GraphEncoderError(
            "invalid_stage6_timing", "node-generation identity differs"
        )
    if (
        set(runtime_identities) != set(SUPPORTED_EXECUTION_DEVICES)
        or set(raw_measurements) != set(SUPPORTED_EXECUTION_DEVICES)
        or set(available_wall_seconds_by_device) != set(SUPPORTED_EXECUTION_DEVICES)
    ):
        raise GraphEncoderError("invalid_stage6_timing", "candidate matrix differs")
    candidates = {}
    for device in SUPPORTED_EXECUTION_DEVICES:
        runtime = runtime_identities[device]
        validate_runtime_identity(runtime, device)
        if set(raw_measurements[device]) != set(ARMS):
            raise GraphEncoderError("invalid_stage6_timing", "arm matrix differs")
        arms = {
            arm: _measurement_record(raw_measurements[device][arm])
            for arm in ARMS
        }
        per_epoch = sum(
            arms[arm]["conservative_seconds_per_epoch"] for arm in ARMS
        )
        raw_three = per_epoch * PRODUCTION_EPOCHS * len(FULL_SEEDS)
        raw_two = per_epoch * PRODUCTION_EPOCHS * len(FALLBACK_SEEDS)
        wall = _finite_positive(
            available_wall_seconds_by_device[device], "available wall time differs"
        )
        candidates[device] = {
            "runtime_identity": runtime,
            "timing_hardware_identity": timing_hardware_identity(runtime),
            "arms": arms,
            "available_wall_seconds": wall,
            "projected_three_seed_seconds": raw_three,
            "projected_three_seed_seconds_with_contingency": raw_three * 1.2,
            "projected_two_seed_seconds": raw_two,
            "projected_two_seed_seconds_with_contingency": raw_two * 1.2,
            "three_seed_feasible": raw_three * 1.2 <= wall,
            "two_seed_feasible": raw_two * 1.2 <= wall,
        }
    full = [device for device in SUPPORTED_EXECUTION_DEVICES
            if candidates[device]["three_seed_feasible"]]
    fallback = False
    if full:
        selected = min(
            full,
            key=lambda device: (
                candidates[device]["projected_three_seed_seconds_with_contingency"],
                0 if device == "cpu" else 1,
            ),
        )
        seeds = FULL_SEEDS
        reason = "fastest feasible three-seed contingent projection"
        fallback_reason = None
    else:
        two = [device for device in SUPPORTED_EXECUTION_DEVICES
               if candidates[device]["two_seed_feasible"]]
        if not two:
            raise GraphEncoderError(
                "stage6_resource_infeasible", "no candidate fits two seeds"
            )
        selected = min(
            two,
            key=lambda device: (
                candidates[device]["projected_two_seed_seconds_with_contingency"],
                0 if device == "cpu" else 1,
            ),
        )
        seeds = FALLBACK_SEEDS
        fallback = True
        reason = "no three-seed candidate fits; fastest feasible two-seed projection"
        fallback_reason = "three seeds exceed every prospectively supplied allocation"
    return {
        "version": TIMING_VERSION,
        "measurement_version": TIMING_MEASUREMENT_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "operation_magnitude_parameterization": (
            GRID_ORDINAL_OPERATION_MAGNITUDE_PARAMETERIZATION
        ),
        "node_generation_identity": node_generation_identity,
        "selection_algorithm": SELECTION_ALGORITHM,
        "source_identity": {
            "source_commit": source_commit,
            "source_tree_sha256": source_tree_sha256,
        },
        "train_input_identity": {
            "index_identity_sha256": train_index_identity_sha256,
            "payload_digests_sha256": train_payload_digests_sha256,
        },
        "measured_before_scientific_outcomes": True,
        "observed_results_used": False,
        "development_accessed": False,
        "train_payload_accessed": True,
        "candidate_devices": list(SUPPORTED_EXECUTION_DEVICES),
        "timed_epochs_per_measurement": TIMED_EPOCHS,
        "timed_repetitions_per_arm_device": TIMED_REPETITIONS,
        "production_epochs": PRODUCTION_EPOCHS,
        "contingency_fraction": CONTINGENCY_FRACTION,
        "candidates": candidates,
        "selected_device": selected,
        "selected_seeds": list(seeds),
        "selected_timing_hardware_identity": candidates[selected][
            "timing_hardware_identity"
        ],
        "selection_reason": reason,
        "fallback_invoked": fallback,
        "fallback_reason": fallback_reason,
        "outcome_fields_recorded": False,
    }


def validate_timing_evidence(record, *, required_device=None):
    """Independently recompute all v2 measurements, projections, and selection."""

    required = {
        "version", "measurement_version", "protocol_version",
        "operation_magnitude_parameterization", "node_generation_identity",
        "selection_algorithm", "source_identity", "train_input_identity",
        "measured_before_scientific_outcomes", "observed_results_used",
        "development_accessed", "train_payload_accessed", "candidate_devices",
        "timed_epochs_per_measurement", "timed_repetitions_per_arm_device",
        "production_epochs", "contingency_fraction", "candidates",
        "selected_device", "selected_seeds", "selected_timing_hardware_identity",
        "selection_reason", "fallback_invoked", "fallback_reason",
        "outcome_fields_recorded",
    }
    if not isinstance(record, dict) or set(record) != required:
        raise GraphEncoderError("invalid_stage6_timing", "timing-v2 fields differ")
    if (
        record["version"] != TIMING_VERSION
        or record["measurement_version"] != TIMING_MEASUREMENT_VERSION
        or record["protocol_version"] != PROTOCOL_VERSION
        or record["operation_magnitude_parameterization"]
        != GRID_ORDINAL_OPERATION_MAGNITUDE_PARAMETERIZATION
        or record["node_generation_identity"] not in NODE_GENERATION_IDENTITIES
        or record["selection_algorithm"] != SELECTION_ALGORITHM
        or record["measured_before_scientific_outcomes"] is not True
        or record["observed_results_used"] is not False
        or record["development_accessed"] is not False
        or record["train_payload_accessed"] is not True
        or record["candidate_devices"] != list(SUPPORTED_EXECUTION_DEVICES)
        or record["timed_epochs_per_measurement"] != TIMED_EPOCHS
        or record["timed_repetitions_per_arm_device"] != TIMED_REPETITIONS
        or record["production_epochs"] != PRODUCTION_EPOCHS
        or record["contingency_fraction"] != CONTINGENCY_FRACTION
        or record["outcome_fields_recorded"] is not False
    ):
        raise GraphEncoderError("invalid_stage6_timing", "timing-v2 identity differs")
    source = record["source_identity"]
    train = record["train_input_identity"]
    if set(source) != {"source_commit", "source_tree_sha256"} or set(train) != {
        "index_identity_sha256", "payload_digests_sha256"
    }:
        raise GraphEncoderError("invalid_stage6_timing", "identity fields differ")
    _hex(source["source_commit"], 40, "source commit")
    _hex(source["source_tree_sha256"], 64, "source digest")
    _hex(train["index_identity_sha256"], 64, "train index digest")
    _hex(train["payload_digests_sha256"], 64, "train payload digest")
    if set(record["candidates"]) != set(SUPPORTED_EXECUTION_DEVICES):
        raise GraphEncoderError("invalid_stage6_timing", "candidates differ")
    raw = {}
    runtimes = {}
    walls = {}
    for device in SUPPORTED_EXECUTION_DEVICES:
        candidate = record["candidates"][device]
        required_candidate = {
            "runtime_identity", "timing_hardware_identity", "arms",
            "available_wall_seconds", "projected_three_seed_seconds",
            "projected_three_seed_seconds_with_contingency",
            "projected_two_seed_seconds",
            "projected_two_seed_seconds_with_contingency",
            "three_seed_feasible", "two_seed_feasible",
        }
        if set(candidate) != required_candidate or set(candidate["arms"]) != set(ARMS):
            raise GraphEncoderError("invalid_stage6_timing", "candidate fields differ")
        validate_runtime_identity(candidate["runtime_identity"], device)
        if candidate["timing_hardware_identity"] != timing_hardware_identity(
            candidate["runtime_identity"]
        ):
            raise GraphEncoderError("invalid_stage6_timing", "hardware identity differs")
        runtimes[device] = candidate["runtime_identity"]
        walls[device] = candidate["available_wall_seconds"]
        raw[device] = {}
        for arm in ARMS:
            arm_record = candidate["arms"][arm]
            if set(arm_record) != {
                "warmup", "timed_measurements", "conservative_seconds_per_epoch",
                "conservative_rule",
            }:
                raise GraphEncoderError("invalid_stage6_timing", "arm fields differ")
            warmup = arm_record["warmup"]
            if set(warmup) != {
                "fresh_model", "included_in_measurement", "epochs", "duration_seconds"
            } or warmup["fresh_model"] is not True or warmup[
                "included_in_measurement"
            ] is not False or warmup["epochs"] != TIMED_EPOCHS:
                raise GraphEncoderError("invalid_stage6_timing", "warmup differs")
            timed = arm_record["timed_measurements"]
            if len(timed) != TIMED_REPETITIONS:
                raise GraphEncoderError("invalid_stage6_timing", "timed rows differ")
            durations = []
            for index, row in enumerate(timed, 1):
                if set(row) != {
                    "measurement_index", "fresh_model", "warm_start",
                    "checkpoint_reuse", "timed_epochs", "duration_seconds",
                    "seconds_per_epoch",
                } or row["measurement_index"] != index or row["fresh_model"] is not True or row[
                    "warm_start"
                ] is not False or row["checkpoint_reuse"] is not False or row[
                    "timed_epochs"
                ] != TIMED_EPOCHS:
                    raise GraphEncoderError("invalid_stage6_timing", "timed row differs")
                duration = _finite_positive(row["duration_seconds"], "duration differs")
                if not math.isclose(
                    row["seconds_per_epoch"], duration / TIMED_EPOCHS,
                    rel_tol=1e-12, abs_tol=1e-12,
                ):
                    raise GraphEncoderError("invalid_stage6_timing", "per-epoch differs")
                durations.append(duration)
            if arm_record["conservative_rule"] != "maximum_of_three_fresh_measurements":
                raise GraphEncoderError("invalid_stage6_timing", "conservative rule differs")
            if not math.isclose(
                arm_record["conservative_seconds_per_epoch"],
                max(durations) / TIMED_EPOCHS, rel_tol=1e-12, abs_tol=1e-12,
            ):
                raise GraphEncoderError("invalid_stage6_timing", "conservative value differs")
            raw[device][arm] = {
                "warmup": warmup["duration_seconds"], "timed": durations,
            }
    rebuilt = build_timing_record(
        source_commit=source["source_commit"],
        source_tree_sha256=source["source_tree_sha256"],
        train_index_identity_sha256=train["index_identity_sha256"],
        train_payload_digests_sha256=train["payload_digests_sha256"],
        runtime_identities=runtimes, raw_measurements=raw,
        available_wall_seconds_by_device=walls,
        node_generation_identity=record["node_generation_identity"],
    )
    if rebuilt != record:
        raise GraphEncoderError("invalid_stage6_timing", "recomputed timing differs")
    selected = validate_execution_device(record["selected_device"])
    if required_device is not None and selected != validate_execution_device(required_device):
        raise GraphEncoderError(
            "stage6_timing_device_mismatch", "runner and timing selection differ"
        )
    return {
        "execution_device": selected,
        "retained_seeds": tuple(record["selected_seeds"]),
        "timing_hardware_identity": record["selected_timing_hardware_identity"],
        "fallback_invoked": record["fallback_invoked"],
        "fallback_reason": record["fallback_reason"],
        "operation_magnitude_parameterization": record[
            "operation_magnitude_parameterization"
        ],
        "node_generation_identity": record["node_generation_identity"],
        "verification_status": "pass",
    }


def create_timing_artifact(record, output_dir, *, job_id):
    job_id = _job_id(job_id)
    validate_timing_evidence(record)
    root = Path(output_dir)
    if root.exists() or root.is_symlink() or not root.parent.is_dir():
        raise GraphEncoderError("unsafe_stage6_timing_output", str(root))
    staging = root.with_name(root.name + ".incomplete-" + job_id)
    if staging.exists() or staging.is_symlink():
        raise GraphEncoderError("unsafe_stage6_timing_output", str(staging))
    staging.mkdir()
    _write(staging / "timing.json", _canonical(record) + "\n")
    manifest = {
        "schema_version": TIMING_ARTIFACT_VERSION,
        "artifacts": [{
            "path": "timing.json",
            "byte_size": (staging / "timing.json").stat().st_size,
            "sha256": _sha(staging / "timing.json"),
        }],
    }
    _write(staging / "artifact_manifest.json", _canonical(manifest) + "\n")
    _write(staging / "SHA256SUMS", "{}  artifact_manifest.json\n{}  timing.json\n".format(
        _sha(staging / "artifact_manifest.json"), _sha(staging / "timing.json")
    ))
    os.replace(str(staging), str(root))
    return verify_timing_artifact(root)


def verify_timing_artifact(path, *, required_device=None):
    root = Path(path)
    if (
        root.is_symlink() or not root.is_dir()
        or tuple(sorted(item.name for item in root.iterdir())) != tuple(sorted(TIMING_FILES))
        or any(item.is_symlink() or not item.is_file() for item in root.iterdir())
    ):
        raise GraphEncoderError("invalid_stage6_timing_artifact", "file set differs")
    record = _load_json(root / "timing.json", "invalid_stage6_timing_artifact")
    selection = validate_timing_evidence(record, required_device=required_device)
    manifest = _load_json(root / "artifact_manifest.json", "invalid_stage6_timing_artifact")
    expected_manifest = {
        "schema_version": TIMING_ARTIFACT_VERSION,
        "artifacts": [{
            "path": "timing.json", "byte_size": (root / "timing.json").stat().st_size,
            "sha256": _sha(root / "timing.json"),
        }],
    }
    if manifest != expected_manifest:
        raise GraphEncoderError("invalid_stage6_timing_artifact", "manifest differs")
    checksums = (root / "SHA256SUMS").read_text(encoding="utf-8")
    expected_checksums = "{}  artifact_manifest.json\n{}  timing.json\n".format(
        _sha(root / "artifact_manifest.json"), _sha(root / "timing.json")
    )
    if checksums != expected_checksums:
        raise GraphEncoderError("invalid_stage6_timing_artifact", "checksums differ")
    return {
        "schema_version": TIMING_ARTIFACT_VERSION,
        "timing_sha256": _sha(root / "timing.json"),
        "artifact_sha256": _sha(root / "SHA256SUMS"),
        **selection,
    }


def _bounded_measurement(train_examples, *, arm, device, repository_root,
                         expected_commit, work_root, measurement_name,
                         node_generation_identity):
    from .config import GE1TrainingConfig
    from .grid_magnitude import GRID_MAGNITUDE_PARAMETERIZATION
    from .model import build_matched_ge1_models
    from .training import run_ge1_training

    configure_stage6_runtime(torch, device, seed=2026)
    flat, graph = build_matched_ge1_models(
        2026,
        operation_magnitude_parameterization=GRID_MAGNITUDE_PARAMETERIZATION,
        node_generation_identity=node_generation_identity,
    )
    model = (flat if arm == "flat" else graph).to(device)
    directory = Path(work_root) / measurement_name
    if device == "cuda:0":
        torch.cuda.synchronize(0)
    started = time.perf_counter()
    run_ge1_training(
        model, train_examples, training_config=GE1TrainingConfig(),
        checkpoint_directory=directory, repository_root=repository_root,
        expected_commit=expected_commit, final_epoch=TIMED_EPOCHS,
        checkpoint_epochs=(TIMED_EPOCHS,), selected_checkpoint_epoch=None,
        timing_protocol_final_epoch=TIMED_EPOCHS, execution_device=device,
    )
    if device == "cuda:0":
        torch.cuda.synchronize(0)
    duration = time.perf_counter() - started
    del model, flat, graph
    if device == "cuda:0":
        torch.cuda.empty_cache()
    return duration


def run_hardware_timing(*, train_index, train_root, output_dir,
                        repository_root, expected_commit, job_id,
                        available_wall_seconds_cpu,
                        available_wall_seconds_cuda,
                        node_generation_identity=LEGACY_NODE_GENERATION_IDENTITY):
    if torch is None:
        raise RuntimeError("Stage 6 hardware timing requires PyTorch")
    import sys
    if sys.version_info[:3] != (3, 8, 13):
        raise GraphEncoderError("environment_mismatch", "Python 3.8.13 required")
    job_id = _job_id(job_id)
    if node_generation_identity not in NODE_GENERATION_IDENTITIES:
        raise GraphEncoderError(
            "invalid_stage6_timing", "node-generation identity differs"
        )
    from .pilot import _source_identity
    from .stage6_narrow_loader import load_stage6_train

    source = _source_identity(repository_root, expected_commit)
    if source["detached_head"] is not True or source["git_dirty"] is not False:
        raise GraphEncoderError("invalid_stage6_source", "clean detached source required")
    train_examples, input_evidence = load_stage6_train(train_index, train_root)
    runtime_identities = {}
    raw = {}
    work = Path(output_dir).with_name(Path(output_dir).name + ".work.incomplete-" + job_id)
    if work.exists() or work.is_symlink():
        raise GraphEncoderError("unsafe_stage6_timing_output", str(work))
    work.mkdir()
    for device in SUPPORTED_EXECUTION_DEVICES:
        runtime_identities[device] = configure_stage6_runtime(torch, device, seed=2026)
        raw[device] = {}
        for arm in ARMS:
            warmup = _bounded_measurement(
                train_examples, arm=arm, device=device,
                repository_root=repository_root, expected_commit=expected_commit,
                work_root=work, measurement_name="{}-{}-warmup".format(
                    device.replace(":", "-"), arm
                ),
                node_generation_identity=node_generation_identity,
            )
            durations = []
            for repetition in range(1, TIMED_REPETITIONS + 1):
                durations.append(_bounded_measurement(
                    train_examples, arm=arm, device=device,
                    repository_root=repository_root, expected_commit=expected_commit,
                    work_root=work,
                    measurement_name="{}-{}-timed-{}".format(
                        device.replace(":", "-"), arm, repetition
                    ),
                    node_generation_identity=node_generation_identity,
                ))
            raw[device][arm] = {"warmup": warmup, "timed": durations}
    record = build_timing_record(
        source_commit=source["git_commit"],
        source_tree_sha256=source["source_tree_sha256"],
        train_index_identity_sha256=input_evidence["index_identity_sha256"],
        train_payload_digests_sha256=input_evidence[
            "observed_payload_digests_sha256"
        ],
        runtime_identities=runtime_identities, raw_measurements=raw,
        available_wall_seconds_by_device={
            "cpu": available_wall_seconds_cpu,
            "cuda:0": available_wall_seconds_cuda,
        },
        node_generation_identity=node_generation_identity,
    )
    result = create_timing_artifact(record, output_dir, job_id=job_id)
    shutil.rmtree(str(work))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-index", required=True)
    parser.add_argument("--train-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--job-id", default=os.environ.get("SLURM_JOB_ID"))
    parser.add_argument("--available-wall-seconds-cpu", type=float, required=True)
    parser.add_argument("--available-wall-seconds-cuda", type=float, required=True)
    parser.add_argument(
        "--node-generation-identity",
        choices=NODE_GENERATION_IDENTITIES,
        default=LEGACY_NODE_GENERATION_IDENTITY,
    )
    args = parser.parse_args(argv)
    result = run_hardware_timing(
        train_index=args.train_index, train_root=args.train_root,
        output_dir=args.output_dir, repository_root=args.repository_root,
        expected_commit=args.expected_commit, job_id=args.job_id,
        available_wall_seconds_cpu=args.available_wall_seconds_cpu,
        available_wall_seconds_cuda=args.available_wall_seconds_cuda,
        node_generation_identity=args.node_generation_identity,
    )
    print(_canonical({"event": "stage6_hardware_timing_completed", **result}),
          flush=True)


if __name__ == "__main__":
    main()
