"""Read-only ADR-0016 linear probe from GE1 memory to node count.

Derived from ``stage6_zero_memory_diagnostic.py`` at commit 12521f6.  The
copied source-retention, checkpoint-authentication, artifact, and access
boundaries remain the basis of this narrower prerequisite diagnostic.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import socket
import sys
import time

try:
    import torch
except ImportError:  # Pure contracts and artifact verification remain runnable.
    torch = None

from .errors import GraphEncoderError
from .pilot import (
    _atomic_write_bytes,
    _atomic_write_json,
    _atomic_write_jsonl,
    _canonical_json_text,
    _file_sha256,
    _is_within,
    _read_canonical_jsonl,
    _regular_artifact_files,
    _safe_relative_path,
    _source_identity,
    _verify_source_unchanged,
)
from .stage6_structure_only import (
    ARMS,
    FULL_SEEDS,
    PROTOCOL_VERSION,
    TRAIN_FAMILY_COUNT,
    score_family_record,
)
from .stage6_structure_only_producer import (
    BATCH_SIZE,
    family_record_from_prediction,
)
from .stage6_train_gate_postmortem import (
    PRODUCER_JOB_ID,
    PRODUCER_SOURCE_COMMIT,
    _load_retained_checkpoint,
    _sanitized_train_evidence,
    _validate_checkpoint_identity,
    source_commit_python_sha256,
    verify_artifact as verify_postmortem_artifact,
)


MEMORY_COUNT_PROBE_VERSION = "GE1-MEMORY-COUNT-LINEAR-PROBE-v1"
MEMORY_COUNT_ARTIFACT_VERSION = "GE1-MEMORY-COUNT-PROBE-ARTIFACT-v1"
MEMORY_COUNT_FEATURE_VERSION = "GE1-MEMORY-COUNT-FEATURES-v1"
MEMORY_COUNT_CLASSES = (4, 5, 7, 8)
MEMORY_COUNT_FEATURES = ("memory", "prequant")
MEMORY_COUNT_TEST_FRACTION = 0.20
MEMORY_COUNT_REGULARIZATION = 1e-3
MEMORY_COUNT_MAX_ITER = 300
MEMORY_COUNT_ACCURACY_MIN = 0.90
MEMORY_COUNT_CONTROL_MARGIN_MIN = 0.10

# Retained aliases belong to the copied zero-memory verifier below.  The
# ADR-0016 entry point and artifact use the MEMORY_COUNT_* identities.
DIAGNOSTIC_VERSION = "GE1-STAGE6-ZERO-MEMORY-DIAGNOSTIC-v1"
ARTIFACT_VERSION = "GE1-STAGE6-ZERO-MEMORY-ARTIFACT-v1"
RECORD_VERSION = "GE1-STAGE6-ZERO-MEMORY-RECORD-v1"
P_ZERO = "P_zero"
PRIOR_POSTMORTEM_COMMIT = "ddf9617fb124fbc18f38302943f0c982019bfee9"
PRIOR_POSTMORTEM_JOB_ID = "3355134"
PRIOR_POSTMORTEM_DIAGNOSTIC_VERSION = (
    "GE1-STAGE6-TRAIN-GATE-POSTMORTEM-v1"
)
PRIOR_POSTMORTEM_ARTIFACT_VERSION = (
    "GE1-STAGE6-TRAIN-GATE-POSTMORTEM-ARTIFACT-v1"
)
PRIOR_POSTMORTEM_DIRECTORY = (
    "ge1-stage6-train-gate-postmortem-{}-{}".format(
        PRIOR_POSTMORTEM_COMMIT, PRIOR_POSTMORTEM_JOB_ID
    )
)
PRIOR_POSTMORTEM_SHA256SUMS_SHA256 = (
    "927b6407542e8019c05ade07ce5c987462ab730bb7399e394e45c130c61e73f3"
)
PRIOR_POSTMORTEM_FILE_SHA256 = {
    "artifact_manifest.json": (
        "1287a0d4397b6322291cece39e3c37449d4449713edc01ea29d3752f7e6a8a04"
    ),
    "resolved_config.json": (
        "24d0bfbf5ab2fe11283d29434e1d4713dd26fd83e4dc9669d3f75a5c34e0c669"
    ),
    "summary.json": (
        "0719022fc691d6c2e31ac2373735957246d298ba54bd269a023367416963f7cd"
    ),
    "train_family_records.jsonl": (
        "9ccbf1b11d320b2f245669be44be284c0e35d73806ce9c6f4dac22bc866bb33e"
    ),
    "training_histories.jsonl": (
        "96a6d63f61f9684bc0542c5b2913ba5ab353b0608f727a57f4a406f8f724fb89"
    ),
}
EXPECTED_CHECKPOINT_SHA256 = {
    "stage6-flat-seed2026.pt": (
        "21c1573602704a86aa7d981ce71be0d1d8697c5ca96545e16268bfcf0ba30b2e"
    ),
    "stage6-typed_graph-seed2026.pt": (
        "b1f24705ab71b319e5f4963ebd8d9cd84a10b5c104d0f80a1df03c5fca379bcc"
    ),
    "stage6-flat-seed2027.pt": (
        "6e0c6c499baa273caabad895768841099c00d46ac3abc2e9e8ff88cdba970a01"
    ),
    "stage6-typed_graph-seed2027.pt": (
        "0208e4f0064b255396e03f5f348134fb4d0819fecc66b225b88898de36978425"
    ),
    "stage6-flat-seed2028.pt": (
        "e06ef970718933e2a3b56d054247bb8d5aa940cc6680c4bc1da81ab00cfeaad4"
    ),
    "stage6-typed_graph-seed2028.pt": (
        "44b3eebd60a2f03545ba6c44998bb5bb4e05b5d0613a4da1fad2e6bb6626f327"
    ),
}
EXPECTED_WRAPPER_NAMES = tuple(sorted(EXPECTED_CHECKPOINT_SHA256))
EXPECTED_ZERO_RECORD_COUNT = len(ARMS) * len(FULL_SEEDS) * TRAIN_FAMILY_COUNT
ORDINARY_FILES = (
    "resolved_config.json",
    "summary.json",
    "zero_memory_records.jsonl",
)
ARTIFACT_FILES = ORDINARY_FILES + ("artifact_manifest.json", "SHA256SUMS")
ACCESS_RECORD = {
    "authorized_narrow_train_accessed": True,
    "retained_job_checkpoint_wrappers_accessed": True,
    "prior_postmortem_artifact_accessed": True,
    "zero_memory_train_only_inference_performed": True,
    "training_performed": False,
    "backward_pass_performed": False,
    "optimizer_step_performed": False,
    "checkpoint_modified": False,
    "development_accessed": False,
    "rr_accessed": False,
    "er_test_accessed": False,
    "iid_accessed": False,
    "history_depth_partition_accessed": False,
    "geometry_extrapolation_accessed": False,
    "corpus_accessed": False,
    "broad_manifest_accessed": False,
    "unrelated_checkpoint_accessed": False,
    "external_checkpoint_accessed": False,
    "model_artifact_accessed": False,
    "repaired_artifact_accessed": False,
    "scientific_repair_performed": False,
}


def _fail(code, detail):
    raise GraphEncoderError(code, detail)


def _job_id(value, code="invalid_stage6_zero_memory_job_id"):
    value = str(value)
    if not value.isascii() or not value.isdecimal() or int(value) <= 0:
        _fail(code, "positive decimal job ID required")
    return value


def _require_sha256(value, code, detail):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail(code, detail)
    return value


def parse_checkpoint_paths(values):
    """Require exactly the six frozen producer-job wrapper paths."""

    result = {}
    for value in values:
        if not isinstance(value, str) or value.count("=") != 1:
            _fail("invalid_stage6_zero_memory_checkpoints", "NAME=PATH required")
        name, raw_path = value.split("=", 1)
        path = Path(raw_path)
        if (
            name not in EXPECTED_CHECKPOINT_SHA256
            or name in result
            or not raw_path
            or path.name != name
            or not path.is_absolute()
            or ".." in path.parts
        ):
            _fail("invalid_stage6_zero_memory_checkpoints", value)
        result[name] = path
    if tuple(sorted(result)) != EXPECTED_WRAPPER_NAMES:
        _fail("invalid_stage6_zero_memory_checkpoints", "wrapper matrix differs")
    parents = {path.parent for path in result.values()}
    expected_parent = (
        "ge1-stage6-structure-only-producer-{}-{}.work.incomplete-{}".format(
            PRODUCER_SOURCE_COMMIT, PRODUCER_JOB_ID, PRODUCER_JOB_ID
        )
    )
    if len(parents) != 1 or next(iter(parents)).name != expected_parent:
        _fail("invalid_stage6_zero_memory_checkpoints", "retained root differs")
    return result


def _require_runtime():
    if torch is None:
        raise RuntimeError("Stage 6 zero-memory diagnostic requires PyTorch")
    if sys.version_info[:3] != (3, 8, 13):
        _fail("environment_mismatch", "Python 3.8.13 required")
    if str(torch.__version__).split("+")[0] != "1.11.0":
        _fail("environment_mismatch", "PyTorch 1.11.0 required")
    if torch.cuda.is_available() or torch.get_num_threads() != 1:
        _fail("environment_mismatch", "one-thread CPU execution required")


def zero_memory_tensor(memory, torch_module=None):
    """Replace only memory values, preserving every tensor property."""

    runtime = torch if torch_module is None else torch_module
    if runtime is None:
        raise RuntimeError("zero-memory intervention requires PyTorch")
    zero = runtime.zeros_like(memory)
    if (
        zero.shape != memory.shape
        or zero.dtype != memory.dtype
        or zero.device != memory.device
    ):
        _fail("invalid_stage6_zero_memory_tensor", "tensor properties changed")
    return zero


def run_zero_memory_evaluation(model, input_batches, *, execution_device="cpu"):
    """Encode once and decode only exact-zero memory through the shared path."""

    if torch is None:
        raise RuntimeError("zero-memory inference requires PyTorch")
    from .autonomous import EncodedAutonomousRow, _decode_condition
    from .stage6_device import assert_tensor_tree_device, validate_execution_device

    execution_device = validate_execution_device(execution_device)
    batches = tuple(input_batches)
    if not batches:
        _fail("empty_stage6_zero_memory_set", "input batches are empty")
    all_ids = tuple(item for batch in batches for item in batch.family_ids)
    if all_ids != tuple(sorted(all_ids)) or len(all_ids) != len(set(all_ids)):
        _fail("invalid_stage6_zero_memory_order", "families differ")
    for batch in batches:
        assert_tensor_tree_device(
            batch.encoder_input, execution_device, "zero-memory input"
        )
    devices = {str(parameter.device) for parameter in model.parameters()}
    if devices and devices != {execution_device}:
        _fail("stage6_mixed_device", "zero-memory model placement differs")

    was_training = model.training
    model.eval()
    encoded_rows = []
    encoding_start = time.perf_counter()
    try:
        with torch.no_grad():
            for batch in batches:
                encoded = model.encode(batch.encoder_input)
                if encoded.memory.size(0) != len(batch.family_ids):
                    _fail("memory_alignment_failure", "memory rows do not align")
                for index, family_id in enumerate(batch.family_ids):
                    encoded_rows.append(EncodedAutonomousRow(
                        family_id,
                        encoded.memory[index:index + 1].detach().clone(),
                        int(batch.node_counts[index]),
                        batch.batch_identity,
                    ))
    finally:
        model.train(was_training)
    encoding_seconds = time.perf_counter() - encoding_start

    supplied = []
    evidence = {}
    intervention_start = time.perf_counter()
    for row in encoded_rows:
        zero = zero_memory_tensor(row.memory)
        supplied.append((row, "zero_memory", zero))
        evidence[row.family_id] = {
            "torch_zeros_like_used": True,
            "memory_shape": list(zero.shape),
            "memory_dtype": str(zero.dtype),
            "memory_device": str(zero.device),
            "memory_element_count": int(zero.numel()),
            "memory_nonzero_count": int(torch.count_nonzero(zero).item()),
            "recipient_node_count": row.node_count,
            "batch_identity": row.batch_identity,
            "only_memory_values_replaced": True,
            "recipient_node_count_preserved": True,
            "decoder_non_memory_inputs_preserved": True,
            "encoder_input_mutated": False,
        }
    intervention_seconds = time.perf_counter() - intervention_start
    condition = _decode_condition(
        model,
        P_ZERO,
        tuple(supplied),
        True,
        True,
        (),
        intervention_seconds,
    )
    return {
        "family_order": all_ids,
        "encoding_seconds": float(encoding_seconds),
        "condition": condition,
        "intervention_evidence": evidence,
    }


def zero_memory_family_record_from_prediction(
    prediction, target, *, template, representation_identity, cohort, arm, seed
):
    """Build a P_zero record through the unchanged production record path."""

    if prediction.condition != P_ZERO:
        _fail("invalid_stage6_zero_memory_alignment", "condition differs")
    if prediction.memory_source_family_id != "zero_memory":
        _fail("invalid_stage6_zero_memory_alignment", "memory source differs")
    compatible_prediction = replace(
        prediction,
        condition="P_true",
        memory_source_family_id=prediction.family_id,
    )
    compatible_record = family_record_from_prediction(
        compatible_prediction,
        target,
        template=template,
        representation_identity=representation_identity,
        cohort=cohort,
        arm=arm,
        seed=seed,
    )
    if not isinstance(compatible_record, dict) or (
        compatible_record.get("family_id") != prediction.family_id
        or compatible_record.get("arm") != arm
        or compatible_record.get("seed") != int(seed)
        or compatible_record.get("cohort") != cohort
        or compatible_record.get("condition") != "P_true"
        or compatible_record.get("memory_source_family_id")
        != prediction.family_id
    ):
        _fail(
            "invalid_stage6_zero_memory_alignment",
            "compatibility record differs",
        )
    diagnostic_record = dict(compatible_record)
    diagnostic_record["condition"] = P_ZERO
    diagnostic_record["memory_source_family_id"] = "zero_memory"
    return diagnostic_record


def score_zero_family_record(record):
    """Apply the unchanged structural scorer to the new intervention record."""

    if record.get("condition") != P_ZERO:
        _fail("invalid_stage6_zero_memory_record", "condition differs")
    if record.get("memory_source_family_id") != "zero_memory":
        _fail("invalid_stage6_zero_memory_record", "memory source differs")
    compatible = dict(record)
    compatible["condition"] = "P_true"
    compatible["memory_source_family_id"] = compatible.get("family_id")
    scored = score_family_record(compatible)
    return scored["score"]


def _record_digest(record):
    return hashlib.sha256(
        _canonical_json_text(record).encode("utf-8")
    ).hexdigest()


def _prior_true_baseline(records):
    selected = [row for row in records if row.get("condition") == "P_true"]
    expected = len(ARMS) * len(FULL_SEEDS) * TRAIN_FAMILY_COUNT
    if len(selected) != expected:
        _fail("invalid_stage6_zero_memory_prior", "true record count differs")
    result = {}
    family_sets = {}
    for record in selected:
        scored = score_family_record(record)
        key = (record["arm"], record["seed"], record["family_id"])
        if key in result or record["cohort"] != "train":
            _fail("invalid_stage6_zero_memory_prior", "true matrix differs")
        result[key] = {
            "template": record["template"],
            "score": scored["score"]["normalized_structural_prefix"],
            "record_sha256": _record_digest(record),
        }
        family_sets.setdefault((record["arm"], record["seed"]), set()).add(
            record["family_id"]
        )
    expected_groups = {(arm, seed) for arm in ARMS for seed in FULL_SEEDS}
    if set(family_sets) != expected_groups or any(
        len(values) != TRAIN_FAMILY_COUNT for values in family_sets.values()
    ):
        _fail("invalid_stage6_zero_memory_prior", "true coverage differs")
    common = None
    for values in family_sets.values():
        if common is None:
            common = values
        elif values != common:
            _fail("invalid_stage6_zero_memory_prior", "families differ")
    return result


def _verify_exact_prior_files(root):
    observed = tuple(sorted(
        item.relative_to(root).as_posix()
        for item in root.rglob("*") if item.is_file()
    ))
    expected = tuple(sorted((*PRIOR_POSTMORTEM_FILE_SHA256, "SHA256SUMS")))
    if observed != expected:
        _fail("invalid_stage6_zero_memory_prior", "file set differs")
    for name, expected_digest in PRIOR_POSTMORTEM_FILE_SHA256.items():
        if _file_sha256(root / name) != expected_digest:
            _fail("stage6_zero_memory_prior_integrity_failure", name)
    if _file_sha256(root / "SHA256SUMS") != PRIOR_POSTMORTEM_SHA256SUMS_SHA256:
        _fail("stage6_zero_memory_prior_integrity_failure", "SHA256SUMS")


def authenticate_prior_postmortem(path):
    """Authenticate the immutable job-3355134 artifact before baseline use."""

    root = Path(path)
    if (
        root.name != PRIOR_POSTMORTEM_DIRECTORY
        or root.is_symlink()
        or not root.is_dir()
    ):
        _fail("invalid_stage6_zero_memory_prior", "root identity differs")
    _verify_exact_prior_files(root)
    verification = verify_postmortem_artifact(
        root,
        expected_commit=PRIOR_POSTMORTEM_COMMIT,
        expected_slurm_job_id=PRIOR_POSTMORTEM_JOB_ID,
    )
    if (
        verification.get("verification_status") != "pass"
        or verification.get("diagnostic_version")
        != PRIOR_POSTMORTEM_DIAGNOSTIC_VERSION
        or verification.get("artifact_version")
        != PRIOR_POSTMORTEM_ARTIFACT_VERSION
        or verification.get("producer_job_id") != PRODUCER_JOB_ID
        or verification.get("producer_source_commit") != PRODUCER_SOURCE_COMMIT
        or verification.get("train_family_record_count") != 7326
    ):
        _fail("invalid_stage6_zero_memory_prior", "verification differs")
    records = _read_canonical_jsonl(root / "train_family_records.jsonl")
    baseline = _prior_true_baseline(records)
    resolved = _canonical_json_file(root / "resolved_config.json")
    prior_summary = _canonical_json_file(root / "summary.json")
    train_input_evidence = resolved.get("train_input_evidence")
    checkpoint_wrapper_sha256 = {
        row.get("wrapper_name"): row.get("stage6_wrapper_sha256")
        for row in resolved.get("checkpoint_evidence", ())
    }
    prior_analysis = prior_summary.get("analysis", {})
    aggregate_mean = {
        arm: prior_analysis.get("train_memory_gate_by_arm", {}).get(arm, {}).get(
            "P_mean"
        )
        for arm in ARMS
    }
    per_seed_mean = {
        str(seed): {
            arm: prior_analysis.get("train_memory_gate_by_seed_and_arm", {}).get(
                str(seed), {}
            ).get(arm, {}).get("P_mean")
            for arm in ARMS
        }
        for seed in FULL_SEEDS
    }
    mean_values = tuple(aggregate_mean.values()) + tuple(
        per_seed_mean[str(seed)][arm] for seed in FULL_SEEDS for arm in ARMS
    )
    if (
        not isinstance(train_input_evidence, dict)
        or train_input_evidence.get("verification_status") != "pass"
        or checkpoint_wrapper_sha256 != EXPECTED_CHECKPOINT_SHA256
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in mean_values
        )
    ):
        _fail("invalid_stage6_zero_memory_prior", "input identity differs")
    _verify_exact_prior_files(root)
    return {
        "diagnostic_version": PRIOR_POSTMORTEM_DIAGNOSTIC_VERSION,
        "artifact_version": PRIOR_POSTMORTEM_ARTIFACT_VERSION,
        "source_commit": PRIOR_POSTMORTEM_COMMIT,
        "slurm_job_id": PRIOR_POSTMORTEM_JOB_ID,
        "producer_source_commit": PRODUCER_SOURCE_COMMIT,
        "producer_job_id": PRODUCER_JOB_ID,
        "directory_name": PRIOR_POSTMORTEM_DIRECTORY,
        "sha256sums_sha256": PRIOR_POSTMORTEM_SHA256SUMS_SHA256,
        "file_sha256": dict(PRIOR_POSTMORTEM_FILE_SHA256),
        "train_input_evidence": train_input_evidence,
        "checkpoint_wrapper_sha256": checkpoint_wrapper_sha256,
        "mean_memory_reference": {
            "source": "authenticated job-3355134 summary",
            "aggregate_by_arm": aggregate_mean,
            "per_seed_and_arm": per_seed_mean,
            "threshold": None,
            "validity_gate_in_this_diagnostic": False,
        },
        "verification_status": "pass",
        "true_record_count": len(baseline),
    }, baseline


def generate_zero_memory_records(model, examples, *, arm, seed):
    """Generate only P_zero before joining targets for structural scoring."""

    from .autonomous import autonomous_input_from_paired
    from .batching import build_paired_batch

    ordered = tuple(sorted(examples, key=lambda item: item.physical_family_id))
    if (
        len(ordered) != TRAIN_FAMILY_COUNT
        or len({item.physical_family_id for item in ordered}) != TRAIN_FAMILY_COUNT
    ):
        _fail("invalid_stage6_zero_memory_train", "family count differs")
    input_batches = tuple(
        autonomous_input_from_paired(
            build_paired_batch(ordered[start:start + BATCH_SIZE]),
            arm,
            execution_device="cpu",
        )
        for start in range(0, len(ordered), BATCH_SIZE)
    )
    autonomous = run_zero_memory_evaluation(
        model, input_batches, execution_device="cpu"
    )
    condition = autonomous["condition"]
    if (
        condition.condition != P_ZERO
        or not condition.available
        or len(condition.predictions) != TRAIN_FAMILY_COUNT
    ):
        _fail("invalid_stage6_zero_memory_condition", "condition differs")

    # This is the first point targets enter the diagnostic path.
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
    for prediction in condition.predictions:
        family_id = prediction.family_id
        if family_id not in targets:
            _fail("invalid_stage6_zero_memory_alignment", family_id)
        family_record = zero_memory_family_record_from_prediction(
            prediction,
            targets[family_id],
            template=templates[family_id],
            representation_identity=representations[family_id],
            cohort="train",
            arm=arm,
            seed=seed,
        )
        records.append({
            "family_record": family_record,
            "intervention_evidence": autonomous["intervention_evidence"][family_id],
        })
    return records


def complete_zero_memory_records(generated, prior_baseline):
    """Join authenticated prior scores to target-free zero-memory outputs."""

    output = []
    for item in generated:
        family = item["family_record"]
        key = (family["arm"], family["seed"], family["family_id"])
        prior = prior_baseline.get(key)
        if prior is None or prior["template"] != family["template"]:
            _fail("invalid_stage6_zero_memory_alignment", str(key))
        zero_score_record = score_zero_family_record(family)
        zero = float(zero_score_record["normalized_structural_prefix"])
        true = float(prior["score"])
        ratio = None if true <= 0.0 else zero / true
        output.append({
            "version": RECORD_VERSION,
            "arm": family["arm"],
            "seed": family["seed"],
            "family_id": family["family_id"],
            "template": family["template"],
            "support": 1,
            "true_memory_score": true,
            "zero_memory_score": zero,
            "zero_over_true": ratio,
            "score_difference": zero - true,
            "prior_true_record_sha256": prior["record_sha256"],
            "zero_memory_score_record": zero_score_record,
            "zero_memory_family_record": family,
            "intervention_evidence": item["intervention_evidence"],
        })
    return output


def _mean(values):
    values = tuple(values)
    if not values:
        _fail("invalid_stage6_zero_memory_aggregation", "empty values")
    if any(not math.isfinite(float(value)) for value in values):
        _fail("invalid_stage6_zero_memory_aggregation", "nonfinite values")
    return sum(float(value) for value in values) / float(len(values))


def _group_summary(rows):
    true = _mean(row["true_memory_score"] for row in rows)
    zero = _mean(row["zero_memory_score"] for row in rows)
    return {
        "support": len(rows),
        "P_true": true,
        "P_zero": zero,
        "zero_over_true": None if true <= 0.0 else zero / true,
        "score_difference": zero - true,
    }


def validate_zero_memory_records(records, prior_baseline):
    expected_fields = {
        "version", "arm", "seed", "family_id", "template", "support",
        "true_memory_score", "zero_memory_score", "zero_over_true",
        "score_difference", "prior_true_record_sha256",
        "zero_memory_score_record", "zero_memory_family_record",
        "intervention_evidence",
    }
    expected_intervention = {
        "torch_zeros_like_used", "memory_shape", "memory_dtype",
        "memory_device", "memory_element_count", "memory_nonzero_count",
        "recipient_node_count", "batch_identity",
        "only_memory_values_replaced", "recipient_node_count_preserved",
        "decoder_non_memory_inputs_preserved", "encoder_input_mutated",
    }
    expected_groups = {(arm, seed) for arm in ARMS for seed in FULL_SEEDS}
    groups = defaultdict(list)
    seen = set()
    for row in records:
        if set(row) != expected_fields or row.get("version") != RECORD_VERSION:
            _fail("invalid_stage6_zero_memory_record", "fields differ")
        key = (row["arm"], row["seed"], row["family_id"])
        if key in seen or key not in prior_baseline:
            _fail("invalid_stage6_zero_memory_record", "identity differs")
        seen.add(key)
        family = row["zero_memory_family_record"]
        prior = prior_baseline[key]
        score = score_zero_family_record(family)
        true = float(prior["score"])
        zero = float(score["normalized_structural_prefix"])
        expected_ratio = None if true <= 0.0 else zero / true
        intervention = row["intervention_evidence"]
        if (
            row["arm"] not in ARMS
            or row["seed"] not in FULL_SEEDS
            or row["support"] != 1
            or row["template"] != prior["template"]
            or family["arm"] != row["arm"]
            or family["seed"] != row["seed"]
            or family["family_id"] != row["family_id"]
            or family["template"] != row["template"]
            or family["cohort"] != "train"
            or family["condition"] != P_ZERO
            or family["memory_source_family_id"] != "zero_memory"
            or row["true_memory_score"] != true
            or row["zero_memory_score"] != zero
            or row["zero_over_true"] != expected_ratio
            or row["score_difference"] != zero - true
            or row["prior_true_record_sha256"] != prior["record_sha256"]
            or row["zero_memory_score_record"] != score
            or set(intervention) != expected_intervention
            or intervention["torch_zeros_like_used"] is not True
            or intervention["memory_nonzero_count"] != 0
            or intervention["memory_element_count"] <= 0
            or intervention["recipient_node_count"] <= 0
            or intervention["batch_identity"] != family["batch_identity"]
            or intervention["only_memory_values_replaced"] is not True
            or intervention["recipient_node_count_preserved"] is not True
            or intervention["decoder_non_memory_inputs_preserved"] is not True
            or intervention["encoder_input_mutated"] is not False
        ):
            _fail("invalid_stage6_zero_memory_record", "content differs")
        for value in (true, zero, row["score_difference"]):
            if not math.isfinite(value):
                _fail("invalid_stage6_zero_memory_record", "nonfinite value")
        groups[(row["arm"], row["seed"])].append(row)
    if (
        len(records) != EXPECTED_ZERO_RECORD_COUNT
        or set(groups) != expected_groups
        or any(len(values) != TRAIN_FAMILY_COUNT for values in groups.values())
        or seen != set(prior_baseline)
    ):
        _fail("invalid_stage6_zero_memory_record", "coverage differs")
    common = None
    for values in groups.values():
        families = {row["family_id"] for row in values}
        if common is None:
            common = families
        elif families != common:
            _fail("invalid_stage6_zero_memory_record", "families differ")
    return True


def analyze_zero_memory(records, prior_baseline, prior_mean_memory):
    validate_zero_memory_records(records, prior_baseline)
    per_seed = []
    for seed in FULL_SEEDS:
        for arm in ARMS:
            selected = [
                row for row in records if row["seed"] == seed and row["arm"] == arm
            ]
            per_seed.append({"arm": arm, "seed": seed, **_group_summary(selected)})
    aggregate = {}
    templates = {}
    family_contributions = []
    for arm in ARMS:
        arm_rows = [row for row in records if row["arm"] == arm]
        aggregate[arm] = _group_summary(arm_rows)
        template_rows = {}
        for template in ("E", "R", "EE", "RE"):
            selected = [row for row in arm_rows if row["template"] == template]
            template_rows[template] = _group_summary(selected)
        templates[arm] = template_rows
        by_family = defaultdict(list)
        for row in arm_rows:
            by_family[row["family_id"]].append(row)
        for family_id in sorted(by_family):
            selected = by_family[family_id]
            if len(selected) != len(FULL_SEEDS):
                _fail("invalid_stage6_zero_memory_aggregation", family_id)
            family_contributions.append({
                "arm": arm,
                "family_id": family_id,
                "template": selected[0]["template"],
                **_group_summary(selected),
            })
    family_balanced = {}
    for arm in ARMS:
        true = _mean(templates[arm][name]["P_true"] for name in ("E", "R", "EE", "RE"))
        zero = _mean(templates[arm][name]["P_zero"] for name in ("E", "R", "EE", "RE"))
        family_balanced[arm] = {
            "description": "equal E/R/EE/RE template macro mean; descriptive only",
            "template_support": 4,
            "P_true": true,
            "P_zero": zero,
            "zero_over_true": None if true <= 0.0 else zero / true,
            "score_difference": zero - true,
            "threshold": None,
            "validity_gate": False,
        }
    zero_vs_prior_mean = {}
    for arm in ARMS:
        mean_value = float(prior_mean_memory["aggregate_by_arm"][arm])
        zero_value = aggregate[arm]["P_zero"]
        zero_vs_prior_mean[arm] = {
            "authenticated_prior_P_mean": mean_value,
            "P_zero": zero_value,
            "zero_minus_prior_mean": zero_value - mean_value,
            "zero_over_prior_mean": (
                None if mean_value <= 0.0 else zero_value / mean_value
            ),
            "threshold": None,
            "validity_gate": False,
        }
    return {
        "diagnostic_version": DIAGNOSTIC_VERSION,
        "per_seed_and_arm": per_seed,
        "aggregate_by_arm": aggregate,
        "by_arm_and_template": templates,
        "family_balanced_secondary": family_balanced,
        "zero_vs_authenticated_prior_mean": zero_vs_prior_mean,
        "family_contributions": family_contributions,
        "aggregation": {
            "primary": "same pooled family-replica mean as ADR-0014 train memory",
            "records_per_arm": len(FULL_SEEDS) * TRAIN_FAMILY_COUNT,
            "records_per_arm_seed": TRAIN_FAMILY_COUNT,
            "family_balanced_secondary_is_gate": False,
        },
        "new_pass_fail_threshold_created": False,
        "scientific_outcome_controls_artifact_validity": False,
        "encoder_superiority_inferred": False,
        "development_conclusion_available": False,
    }


def _canonical_json_file(path):
    raw = Path(path).read_bytes()
    if not raw.endswith(b"\n"):
        _fail("invalid_stage6_zero_memory_artifact", "final LF absent")
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise GraphEncoderError(
            "invalid_stage6_zero_memory_artifact", "JSON unreadable"
        ) from exc
    if raw.decode("utf-8") != _canonical_json_text(value) + "\n":
        _fail("invalid_stage6_zero_memory_artifact", "JSON not canonical")
    return value


def _prepare_output(output_dir, input_roots, repository_root, job_id):
    final = Path(output_dir).resolve()
    protected = tuple(Path(value).resolve() for value in input_roots) + (
        Path(repository_root).resolve(),
    )
    if (
        final.exists()
        or final.is_symlink()
        or not final.parent.is_dir()
        or any(_is_within(final, value) or _is_within(value, final) for value in protected)
    ):
        _fail("unsafe_stage6_zero_memory_output", str(final))
    staging = final.with_name(final.name + ".incomplete-" + job_id)
    if staging.exists() or staging.is_symlink():
        _fail("unsafe_stage6_zero_memory_output", str(staging))
    staging.mkdir()
    return final, staging


def _verify_checksums(root):
    manifest = _canonical_json_file(root / "artifact_manifest.json")
    if (
        manifest.get("schema_version") != ARTIFACT_VERSION
        or manifest.get("diagnostic_version") != DIAGNOSTIC_VERSION
    ):
        _fail("invalid_stage6_zero_memory_artifact", "manifest identity differs")
    ordinary = _regular_artifact_files(
        root, excluded=("artifact_manifest.json", "SHA256SUMS")
    )
    listed = []
    for row in manifest.get("artifacts", ()):
        if set(row) != {"path", "byte_size", "sha256"}:
            _fail("invalid_stage6_zero_memory_artifact", "manifest row differs")
        relative = _safe_relative_path(row["path"])
        target = root / relative
        _require_sha256(
            row["sha256"], "invalid_stage6_zero_memory_artifact", "manifest hash"
        )
        if (
            target.is_symlink()
            or not target.is_file()
            or target.stat().st_size != row["byte_size"]
            or _file_sha256(target) != row["sha256"]
        ):
            _fail("stage6_zero_memory_integrity_failure", relative)
        listed.append(relative)
    if tuple(listed) != ordinary:
        _fail("invalid_stage6_zero_memory_artifact", "manifest coverage differs")
    raw = (root / "SHA256SUMS").read_bytes()
    if not raw.endswith(b"\n"):
        _fail("invalid_stage6_zero_memory_artifact", "checksum LF absent")
    names = []
    for line in raw.decode("utf-8").splitlines():
        parts = line.split("  ", 1)
        if len(parts) != 2:
            _fail("invalid_stage6_zero_memory_artifact", "checksum row differs")
        digest = _require_sha256(
            parts[0], "invalid_stage6_zero_memory_artifact", "checksum hash"
        )
        relative = _safe_relative_path(parts[1])
        if _file_sha256(root / relative) != digest:
            _fail("stage6_zero_memory_integrity_failure", relative)
        names.append(relative)
    expected = _regular_artifact_files(root, excluded=("SHA256SUMS",))
    if tuple(names) != expected:
        _fail("invalid_stage6_zero_memory_artifact", "checksum coverage differs")
    return manifest


def finalize_artifact(staging_dir, final_dir, *, prior_postmortem_artifact):
    staging = Path(staging_dir)
    final = Path(final_dir)
    if staging.parent.resolve() != final.parent.resolve():
        _fail("unsafe_stage6_zero_memory_output", "parents differ")
    ordinary = _regular_artifact_files(
        staging, excluded=("artifact_manifest.json", "SHA256SUMS")
    )
    if ordinary != tuple(sorted(ORDINARY_FILES)):
        _fail("incomplete_stage6_zero_memory_artifact", "ordinary files differ")
    manifest = {
        "schema_version": ARTIFACT_VERSION,
        "diagnostic_version": DIAGNOSTIC_VERSION,
        "artifacts": [{
            "path": relative,
            "byte_size": (staging / relative).stat().st_size,
            "sha256": _file_sha256(staging / relative),
        } for relative in ordinary],
    }
    _atomic_write_json(staging / "artifact_manifest.json", manifest)
    checksum_paths = _regular_artifact_files(staging, excluded=("SHA256SUMS",))
    _atomic_write_bytes(
        staging / "SHA256SUMS",
        "".join(
            "{}  {}\n".format(_file_sha256(staging / name), name)
            for name in checksum_paths
        ).encode("utf-8"),
    )
    resolved = _canonical_json_file(staging / "resolved_config.json")
    verify_artifact(
        staging,
        expected_commit=resolved["diagnostic_source"]["git_commit"],
        expected_slurm_job_id=resolved["runtime"]["slurm_job_id"],
        prior_postmortem_artifact=prior_postmortem_artifact,
        allow_incomplete_name=True,
    )
    os.replace(str(staging), str(final))
    return final


def verify_artifact(
    path,
    *,
    expected_commit,
    expected_slurm_job_id,
    prior_postmortem_artifact,
    allow_incomplete_name=False,
):
    job_id = _job_id(expected_slurm_job_id)
    root = Path(path)
    if not root.is_dir() or root.is_symlink():
        _fail("invalid_stage6_zero_memory_artifact", "root differs")
    if ".incomplete-" in root.name and not allow_incomplete_name:
        _fail("invalid_stage6_zero_memory_artifact", "incomplete root")
    observed = tuple(sorted(
        item.relative_to(root).as_posix()
        for item in root.rglob("*") if item.is_file()
    ))
    if observed != tuple(sorted(ARTIFACT_FILES)):
        _fail("invalid_stage6_zero_memory_artifact", "file set differs")
    _verify_checksums(root)
    prior_evidence, prior_baseline = authenticate_prior_postmortem(
        prior_postmortem_artifact
    )
    resolved = _canonical_json_file(root / "resolved_config.json")
    summary = _canonical_json_file(root / "summary.json")
    records = _read_canonical_jsonl(root / "zero_memory_records.jsonl")
    if len(records) != EXPECTED_ZERO_RECORD_COUNT:
        _fail("invalid_stage6_zero_memory_artifact", "record count differs")
    analysis = analyze_zero_memory(
        records, prior_baseline, prior_evidence["mean_memory_reference"]
    )
    source = resolved.get("diagnostic_source", {})
    runtime = resolved.get("runtime", {})
    train_evidence = resolved.get("train_input_evidence", {})
    checkpoints = resolved.get("checkpoint_evidence", ())
    producer_source_digest = resolved.get("producer_source_tree_sha256")
    if (
        resolved.get("diagnostic_version") != DIAGNOSTIC_VERSION
        or resolved.get("artifact_version") != ARTIFACT_VERSION
        or resolved.get("protocol_version") != PROTOCOL_VERSION
        or resolved.get("producer_source_commit") != PRODUCER_SOURCE_COMMIT
        or resolved.get("producer_job_id") != PRODUCER_JOB_ID
        or resolved.get("prior_postmortem_evidence") != prior_evidence
        or source.get("git_commit") != expected_commit
        or source.get("git_branch") is not None
        or source.get("detached_head") is not True
        or source.get("git_dirty") is not False
        or source.get("git_status_porcelain") != []
        or runtime.get("slurm_job_id") != job_id
        or runtime.get("python") != "3.8.13"
        or str(runtime.get("pytorch", "")).split("+")[0] != "1.11.0"
        or runtime.get("device") != "cpu"
        or runtime.get("cuda_available") is not False
        or runtime.get("cpu_threads") != 1
        or train_evidence != prior_evidence.get("train_input_evidence")
        or len(checkpoints) != len(EXPECTED_WRAPPER_NAMES)
        or not isinstance(producer_source_digest, str)
        or len(producer_source_digest) != 64
    ):
        _fail("invalid_stage6_zero_memory_artifact", "resolved identity differs")
    observed_checkpoint_hashes = {}
    for row in checkpoints:
        identity = row.get("identity", {})
        name = row.get("wrapper_name")
        if (
            set(row) != {
                "wrapper_name", "stage6_wrapper_sha256", "identity",
                "strict_recovery_passed", "checkpoint_modified",
            }
            or name not in EXPECTED_CHECKPOINT_SHA256
            or row.get("stage6_wrapper_sha256")
            != EXPECTED_CHECKPOINT_SHA256.get(name)
            or row.get("strict_recovery_passed") is not True
            or row.get("checkpoint_modified") is not False
        ):
            _fail("invalid_stage6_zero_memory_artifact", "checkpoint differs")
        observed_checkpoint_hashes[name] = row["stage6_wrapper_sha256"]
        _validate_checkpoint_identity(
            identity,
            arm=identity.get("arm"),
            seed=identity.get("seed"),
            model_config=identity.get("model_config"),
            parameter_count=identity.get("parameter_count"),
            partition_identity=identity.get("training_partition_identity"),
            train_input_evidence=train_evidence,
            producer_source_digest=producer_source_digest,
        )
    if observed_checkpoint_hashes != EXPECTED_CHECKPOINT_SHA256:
        _fail("invalid_stage6_zero_memory_artifact", "checkpoint matrix differs")
    if (
        summary.get("diagnostic_version") != DIAGNOSTIC_VERSION
        or summary.get("artifact_version") != ARTIFACT_VERSION
        or summary.get("diagnostic_source_commit") != expected_commit
        or summary.get("slurm_job_id") != job_id
        or summary.get("producer_source_commit") != PRODUCER_SOURCE_COMMIT
        or summary.get("producer_job_id") != PRODUCER_JOB_ID
        or summary.get("prior_postmortem_evidence") != prior_evidence
        or summary.get("zero_memory_record_count") != len(records)
        or summary.get("analysis") != analysis
        or summary.get("diagnostic_completed") is not True
        or summary.get("outcome_controls_artifact_validity") is not False
        or summary.get("access") != ACCESS_RECORD
        or resolved.get("access") != ACCESS_RECORD
    ):
        _fail("invalid_stage6_zero_memory_artifact", "summary differs")
    return {
        "diagnostic_version": DIAGNOSTIC_VERSION,
        "artifact_version": ARTIFACT_VERSION,
        "source_commit": expected_commit,
        "slurm_job_id": job_id,
        "producer_source_commit": PRODUCER_SOURCE_COMMIT,
        "producer_job_id": PRODUCER_JOB_ID,
        "prior_postmortem_commit": PRIOR_POSTMORTEM_COMMIT,
        "prior_postmortem_job_id": PRIOR_POSTMORTEM_JOB_ID,
        "checkpoint_count": len(checkpoints),
        "zero_memory_record_count": len(records),
        "regular_file_count": len(ARTIFACT_FILES),
        "sha256sums_sha256": _file_sha256(root / "SHA256SUMS"),
        "outcome_controls_artifact_validity": False,
        "verification_status": "pass",
    }


def run_zero_memory_diagnostic(
    *,
    checkpoint_paths,
    train_index,
    train_root,
    prior_postmortem_artifact,
    output_dir,
    repository_root,
    expected_commit,
    slurm_job_id,
):
    _require_runtime()
    job_id = _job_id(slurm_job_id)
    source = _source_identity(repository_root, expected_commit)
    if source["detached_head"] is not True:
        _fail("invalid_stage6_zero_memory_source", "detached checkout required")
    checkpoints = parse_checkpoint_paths(checkpoint_paths)

    # The prior artifact is authenticated before any train loader or checkpoint.
    prior_evidence, prior_baseline = authenticate_prior_postmortem(
        prior_postmortem_artifact
    )
    repository = Path(repository_root).resolve()
    train_package = Path(train_index).resolve().parent
    train_payload_root = Path(train_root).resolve()
    prior_root = Path(prior_postmortem_artifact).resolve()
    retained_root = next(iter(checkpoints.values())).parent.resolve()
    if (
        Path(train_index).name != "index.json"
        or train_payload_root != train_package / "payloads"
        or any(_is_within(value, repository) for value in (
            train_package, prior_root, retained_root
        ))
        or any(_is_within(left, right) or _is_within(right, left)
               for index, left in enumerate((train_package, prior_root, retained_root))
               for right in (train_package, prior_root, retained_root)[index + 1:])
    ):
        _fail("unsafe_stage6_zero_memory_inputs", "input roots overlap or differ")
    final, staging = _prepare_output(
        output_dir,
        (train_package, prior_root, retained_root),
        repository_root,
        job_id,
    )

    from .stage6_narrow_loader import load_stage6_train
    train_examples, train_input_evidence = load_stage6_train(train_index, train_root)
    train_input_evidence = _sanitized_train_evidence(train_input_evidence)
    if len(train_examples) != TRAIN_FAMILY_COUNT:
        _fail("invalid_stage6_zero_memory_train", "family count differs")
    if train_input_evidence != prior_evidence["train_input_evidence"]:
        _fail("invalid_stage6_zero_memory_train", "input identity differs")
    train_family_ids = tuple(sorted(
        item.physical_family_id for item in train_examples
    ))
    producer_source_digest = source_commit_python_sha256(
        repository_root, PRODUCER_SOURCE_COMMIT
    )
    checkpoint_evidence = []
    validated_models = []
    for seed in FULL_SEEDS:
        for arm in ARMS:
            name = "stage6-{}-seed{}.pt".format(arm, seed)
            model, unused_training, evidence = _load_retained_checkpoint(
                checkpoints[name],
                expected_wrapper_sha256=EXPECTED_CHECKPOINT_SHA256[name],
                arm=arm,
                seed=seed,
                train_family_ids=train_family_ids,
                train_input_evidence=train_input_evidence,
                producer_source_digest=producer_source_digest,
            )
            checkpoint_evidence.append(evidence)
            validated_models.append((model, arm, seed))

    generated = []
    for model, arm, seed in validated_models:
        generated.extend(generate_zero_memory_records(
            model, train_examples, arm=arm, seed=seed
        ))
    records = complete_zero_memory_records(generated, prior_baseline)
    analysis = analyze_zero_memory(
        records, prior_baseline, prior_evidence["mean_memory_reference"]
    )
    resolved = {
        "diagnostic_version": DIAGNOSTIC_VERSION,
        "artifact_version": ARTIFACT_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "producer_source_commit": PRODUCER_SOURCE_COMMIT,
        "producer_source_tree_sha256": producer_source_digest,
        "producer_job_id": PRODUCER_JOB_ID,
        "prior_postmortem_evidence": prior_evidence,
        "diagnostic_source": source,
        "runtime": {
            "python": platform.python_version(),
            "pytorch": str(torch.__version__),
            "device": "cpu",
            "cuda_available": bool(torch.cuda.is_available()),
            "cpu_threads": int(torch.get_num_threads()),
            "host": socket.gethostname(),
            "slurm_job_id": job_id,
        },
        "train_input_evidence": train_input_evidence,
        "checkpoint_evidence": checkpoint_evidence,
        "zero_memory_intervention": {
            "implementation": "torch.zeros_like immediately before decoder use",
            "memory_values_zeroed": True,
            "shape_dtype_device_preserved": True,
            "recipient_node_counts_preserved": True,
            "masks_positions_constraints_preserved": True,
            "decoder_unchanged": True,
            "targets_joined_after_generation": True,
            "additional_conditions": [],
            "cpu_round_trip": False,
        },
        "implementation_reuse": {
            "checkpoint_recovery": (
                "prototype.graph_encoder.stage6_train_gate_postmortem."
                "_load_retained_checkpoint"
            ),
            "decoder_path": "prototype.graph_encoder.autonomous._decode_condition",
            "family_record": (
                "prototype.graph_encoder.stage6_structure_only_producer."
                "family_record_from_prediction"
            ),
            "structural_scoring": (
                "prototype.graph_encoder.stage6_structure_only.score_family_record"
            ),
            "training_repeated": False,
            "threshold_changed": False,
        },
        "access": dict(ACCESS_RECORD),
    }
    summary = {
        "diagnostic_version": DIAGNOSTIC_VERSION,
        "artifact_version": ARTIFACT_VERSION,
        "diagnostic_source_commit": expected_commit,
        "slurm_job_id": job_id,
        "producer_source_commit": PRODUCER_SOURCE_COMMIT,
        "producer_job_id": PRODUCER_JOB_ID,
        "prior_postmortem_evidence": prior_evidence,
        "zero_memory_record_count": len(records),
        "analysis": analysis,
        "diagnostic_completed": True,
        "outcome_controls_artifact_validity": False,
        "access": dict(ACCESS_RECORD),
    }
    _atomic_write_json(staging / "resolved_config.json", resolved)
    _atomic_write_jsonl(staging / "zero_memory_records.jsonl", records)
    _atomic_write_json(staging / "summary.json", summary)
    _verify_source_unchanged(source, repository_root, expected_commit)
    finalize_artifact(
        staging, final, prior_postmortem_artifact=prior_postmortem_artifact
    )
    verified = verify_artifact(
        final,
        expected_commit=expected_commit,
        expected_slurm_job_id=job_id,
        prior_postmortem_artifact=prior_postmortem_artifact,
    )
    return {
        "event": "stage6_zero_memory_diagnostic_completed",
        "artifact_path": str(final),
        "verification": verified,
        "analysis": analysis,
        "diagnostic_completed": True,
        "outcome_controls_artifact_validity": False,
    }


MEMORY_COUNT_ARTIFACT_FILES = (
    "artifact_manifest.json",
    "features.jsonl",
    "probe_results.json",
    "resolved_config.json",
    "SHA256SUMS",
)
MEMORY_COUNT_ACCESS_RECORD = {
    "authorized_narrow_train_accessed": True,
    "retained_job_checkpoint_wrappers_accessed": True,
    "prior_postmortem_artifact_accessed": True,
    "encoder_forward_performed": True,
    "disposable_linear_probe_fit_performed": True,
    "majority_baseline_computed": True,
    "shuffled_label_control_computed": True,
    "ge1_training_performed": False,
    "ge1_backward_pass_performed": False,
    "ge1_optimizer_step_performed": False,
    "ge1_checkpoint_modified": False,
    "development_accessed": False,
    "rr_accessed": False,
    "er_test_accessed": False,
    "iid_accessed": False,
    "history_depth_partition_accessed": False,
    "geometry_extrapolation_accessed": False,
    "other_corpus_accessed": False,
    "cad_kernel_executed": False,
    "stage6_performed": False,
}


def deterministic_count_split(rows):
    """Return one family-aligned, class-stratified train/test split."""

    values = tuple(sorted(rows, key=lambda item: item["family_id"]))
    if len(values) != TRAIN_FAMILY_COUNT:
        _fail("invalid_memory_count_probe_rows", "train family count differs")
    by_count = defaultdict(list)
    for row in values:
        count = row.get("node_count")
        if count not in MEMORY_COUNT_CLASSES:
            _fail("invalid_memory_count_probe_label", repr(count))
        by_count[count].append(row)
    if set(by_count) != set(MEMORY_COUNT_CLASSES):
        _fail("invalid_memory_count_probe_label", "class coverage differs")
    train_ids = set()
    test_ids = set()
    per_class = {}
    for count in MEMORY_COUNT_CLASSES:
        ordered = sorted(
            by_count[count],
            key=lambda item: hashlib.sha256(
                "{}:{}:{}".format(
                    MEMORY_COUNT_PROBE_VERSION, count, item["family_id"]
                ).encode("utf-8")
            ).hexdigest(),
        )
        test_count = max(1, int(round(len(ordered) * MEMORY_COUNT_TEST_FRACTION)))
        test_count = min(test_count, len(ordered) - 1)
        current_test = {item["family_id"] for item in ordered[:test_count]}
        current_train = {item["family_id"] for item in ordered[test_count:]}
        test_ids.update(current_test)
        train_ids.update(current_train)
        per_class[str(count)] = {
            "total": len(ordered),
            "train": len(current_train),
            "test": len(current_test),
        }
    if train_ids & test_ids or train_ids | test_ids != {
        item["family_id"] for item in values
    }:
        _fail("invalid_memory_count_probe_split", "family split differs")
    return (
        tuple(item for item in values if item["family_id"] in train_ids),
        tuple(item for item in values if item["family_id"] in test_ids),
        {
            "method": "sha256_ordered_stratified_within_train",
            "test_fraction": MEMORY_COUNT_TEST_FRACTION,
            "train_family_count": len(train_ids),
            "test_family_count": len(test_ids),
            "per_class": per_class,
            "train_family_ids_sha256": hashlib.sha256(
                ("\n".join(sorted(train_ids)) + "\n").encode("utf-8")
            ).hexdigest(),
            "test_family_ids_sha256": hashlib.sha256(
                ("\n".join(sorted(test_ids)) + "\n").encode("utf-8")
            ).hexdigest(),
        },
    )


def _count_class_index(node_count):
    try:
        return MEMORY_COUNT_CLASSES.index(int(node_count))
    except (TypeError, ValueError) as exc:
        raise GraphEncoderError(
            "invalid_memory_count_probe_label", repr(node_count)
        ) from exc


def _count_metrics(truth, predicted):
    truth = tuple(int(value) for value in truth)
    predicted = tuple(int(value) for value in predicted)
    if not truth or len(truth) != len(predicted):
        _fail("invalid_memory_count_probe_metrics", "rows differ")
    confusion = [
        [0] * len(MEMORY_COUNT_CLASSES) for _ in MEMORY_COUNT_CLASSES
    ]
    for expected, observed in zip(truth, predicted):
        if (
            expected not in range(len(MEMORY_COUNT_CLASSES))
            or observed not in range(len(MEMORY_COUNT_CLASSES))
        ):
            _fail("invalid_memory_count_probe_metrics", "class differs")
        confusion[expected][observed] += 1
    recalls = []
    for index, row in enumerate(confusion):
        support = sum(row)
        if support <= 0:
            _fail("invalid_memory_count_probe_metrics", "zero class support")
        recalls.append(row[index] / float(support))
    return {
        "accuracy": sum(
            int(expected == observed)
            for expected, observed in zip(truth, predicted)
        ) / float(len(truth)),
        "balanced_accuracy": sum(recalls) / float(len(recalls)),
        "per_class_recall": {
            str(MEMORY_COUNT_CLASSES[index]): value
            for index, value in enumerate(recalls)
        },
        "confusion_matrix": confusion,
        "row_count": len(truth),
    }


def _standardize_count_features(train_rows, test_rows, feature_name):
    train = torch.tensor(
        [row[feature_name] for row in train_rows], dtype=torch.float64
    )
    test = torch.tensor(
        [row[feature_name] for row in test_rows], dtype=torch.float64
    )
    if (
        train.dim() != 2
        or test.dim() != 2
        or train.size(1) != test.size(1)
        or train.size(0) <= 0
        or test.size(0) <= 0
    ):
        _fail("invalid_memory_count_probe_features", feature_name)
    mean = train.mean(dim=0)
    std = train.std(dim=0, unbiased=False)
    safe = torch.where(std == 0.0, torch.ones_like(std), std)
    train = (train - mean) / safe
    test = (test - mean) / safe
    train[:, std == 0.0] = 0.0
    test[:, std == 0.0] = 0.0
    return train, test, {
        "population_mean": [float(value) for value in mean.tolist()],
        "population_standard_deviation": [
            float(value) for value in std.tolist()
        ],
        "zero_variance_feature_count": int((std == 0.0).sum().item()),
    }


def _fit_count_linear(train_x, train_labels, test_x):
    class_count = len(MEMORY_COUNT_CLASSES)
    weights = torch.zeros(
        (class_count, train_x.size(1)), dtype=torch.float64, requires_grad=True
    )
    intercept = torch.zeros(
        (class_count,), dtype=torch.float64, requires_grad=True
    )
    labels = torch.tensor(tuple(train_labels), dtype=torch.long)
    optimizer = torch.optim.LBFGS(
        (weights, intercept),
        lr=1.0,
        max_iter=MEMORY_COUNT_MAX_ITER,
        history_size=100,
        tolerance_grad=1e-9,
        tolerance_change=1e-12,
        line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad(set_to_none=True)
        logits = train_x.mm(weights.t()) + intercept
        loss = torch.nn.functional.cross_entropy(logits, labels)
        loss = loss + (
            0.5 * MEMORY_COUNT_REGULARIZATION * weights.square().sum()
        )
        if not bool(torch.isfinite(loss).item()):
            _fail("nonfinite_memory_count_probe", "objective")
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        logits = train_x.mm(weights.t()) + intercept
        final = torch.nn.functional.cross_entropy(logits, labels) + (
            0.5 * MEMORY_COUNT_REGULARIZATION * weights.square().sum()
        )
        predicted = (test_x.mm(weights.t()) + intercept).argmax(dim=1)
    return tuple(int(value) for value in predicted.tolist()), {
        "dtype": "float64",
        "device": "cpu",
        "optimizer": "LBFGS",
        "max_iter": MEMORY_COUNT_MAX_ITER,
        "regularization": MEMORY_COUNT_REGULARIZATION,
        "intercept_penalized": False,
        "parameter_count": int(weights.numel() + intercept.numel()),
        "final_objective": float(final.item()),
    }


def fit_count_probe(rows, feature_name):
    train_rows, test_rows, split = deterministic_count_split(rows)
    train_x, test_x, preprocessing = _standardize_count_features(
        train_rows, test_rows, feature_name
    )
    train_labels = tuple(
        _count_class_index(item["node_count"]) for item in train_rows
    )
    test_labels = tuple(
        _count_class_index(item["node_count"]) for item in test_rows
    )
    predicted, fit = _fit_count_linear(train_x, train_labels, test_x)
    observed = _count_metrics(test_labels, predicted)

    counts = {
        label: train_labels.count(label) for label in range(len(MEMORY_COUNT_CLASSES))
    }
    majority = max(counts, key=lambda value: (counts[value], -value))
    majority_metrics = _count_metrics(
        test_labels, (majority,) * len(test_labels)
    )

    shuffled = list(train_labels)
    shuffle_seed = int.from_bytes(hashlib.sha256(
        "{}:{}".format(MEMORY_COUNT_PROBE_VERSION, feature_name).encode("utf-8")
    ).digest()[:8], "big")
    random.Random(shuffle_seed).shuffle(shuffled)
    shuffled_predicted, shuffled_fit = _fit_count_linear(
        train_x, tuple(shuffled), test_x
    )
    shuffled_metrics = _count_metrics(test_labels, shuffled_predicted)
    recovered = (
        observed["balanced_accuracy"] >= MEMORY_COUNT_ACCURACY_MIN
        and observed["balanced_accuracy"]
        >= majority_metrics["balanced_accuracy"] + MEMORY_COUNT_CONTROL_MARGIN_MIN
        and observed["balanced_accuracy"]
        >= shuffled_metrics["balanced_accuracy"] + MEMORY_COUNT_CONTROL_MARGIN_MIN
    )
    return {
        "feature_name": feature_name,
        "feature_dimension": len(rows[0][feature_name]),
        "classes": list(MEMORY_COUNT_CLASSES),
        "split": split,
        "preprocessing": preprocessing,
        "fit": fit,
        "test_metrics": observed,
        "majority_class_baseline": {
            "class": MEMORY_COUNT_CLASSES[majority],
            "metrics": majority_metrics,
        },
        "shuffled_label_control": {
            "seed": shuffle_seed,
            "fit": shuffled_fit,
            "metrics": shuffled_metrics,
        },
        "recovery_rule": {
            "balanced_accuracy_minimum_inclusive": MEMORY_COUNT_ACCURACY_MIN,
            "margin_over_each_control_minimum_inclusive": (
                MEMORY_COUNT_CONTROL_MARGIN_MIN
            ),
        },
        "linearly_recoverable": bool(recovered),
    }


def extract_count_features(model, examples, *, arm, seed):
    """Encode authorized train examples and detach memory/prequant only."""

    from .autonomous import autonomous_input_from_paired
    from .batching import build_paired_batch

    ordered = tuple(sorted(examples, key=lambda item: item.physical_family_id))
    if len(ordered) != TRAIN_FAMILY_COUNT or model.config.encoder != arm:
        _fail("invalid_memory_count_probe_rows", "model/train alignment differs")
    model.requires_grad_(False)
    model.eval()
    rows = []
    with torch.no_grad():
        for start in range(0, len(ordered), BATCH_SIZE):
            autonomous = autonomous_input_from_paired(
                build_paired_batch(ordered[start:start + BATCH_SIZE]), arm
            )
            encoded = model.encode(autonomous.encoder_input)
            expected = len(autonomous.family_ids)
            if (
                tuple(encoded.memory.shape)
                != (expected, model.config.latent_tokens, model.config.model_dim)
                or tuple(encoded.prequant.shape)
                != (expected, model.config.latent_tokens, model.config.bottleneck_dim)
            ):
                _fail("invalid_memory_count_probe_features", "shape differs")
            for index, family_id in enumerate(autonomous.family_ids):
                rows.append({
                    "version": MEMORY_COUNT_FEATURE_VERSION,
                    "arm": arm,
                    "seed": int(seed),
                    "family_id": family_id,
                    "node_count": int(autonomous.node_counts[index]),
                    "memory": [
                        float(value) for value in
                        encoded.memory[index].detach().cpu().reshape(-1).tolist()
                    ],
                    "prequant": [
                        float(value) for value in
                        encoded.prequant[index].detach().cpu().reshape(-1).tolist()
                    ],
                })
    if (
        len(rows) != TRAIN_FAMILY_COUNT
        or len({item["family_id"] for item in rows}) != TRAIN_FAMILY_COUNT
        or any(len(item["memory"]) != 64 for item in rows)
        or any(len(item["prequant"]) != 32 for item in rows)
    ):
        _fail("invalid_memory_count_probe_features", "coverage differs")
    return tuple(rows)


def _count_probe_checksums(root):
    raw = (root / "SHA256SUMS").read_bytes()
    if not raw.endswith(b"\n"):
        _fail("invalid_memory_count_probe_artifact", "checksum LF absent")
    names = []
    for line in raw.decode("utf-8").splitlines():
        parts = line.split("  ", 1)
        if len(parts) != 2:
            _fail("invalid_memory_count_probe_artifact", "checksum row differs")
        digest = _require_sha256(
            parts[0], "invalid_memory_count_probe_artifact", "checksum hash"
        )
        relative = _safe_relative_path(parts[1])
        if _file_sha256(root / relative) != digest:
            _fail("memory_count_probe_integrity_failure", relative)
        names.append(relative)
    expected = _regular_artifact_files(root, excluded=("SHA256SUMS",))
    if tuple(names) != expected:
        _fail("invalid_memory_count_probe_artifact", "checksum coverage differs")


def verify_memory_count_artifact(
    path, *, expected_commit, expected_slurm_job_id, allow_incomplete_name=False
):
    job_id = _job_id(expected_slurm_job_id, "invalid_memory_count_probe_job_id")
    root = Path(path)
    if (
        not root.is_dir()
        or root.is_symlink()
        or (".incomplete-" in root.name and not allow_incomplete_name)
    ):
        _fail("invalid_memory_count_probe_artifact", "root differs")
    observed = tuple(sorted(
        item.relative_to(root).as_posix()
        for item in root.rglob("*") if item.is_file()
    ))
    if observed != tuple(sorted(MEMORY_COUNT_ARTIFACT_FILES)):
        _fail("invalid_memory_count_probe_artifact", "file set differs")
    _count_probe_checksums(root)
    manifest = _canonical_json_file(root / "artifact_manifest.json")
    resolved = _canonical_json_file(root / "resolved_config.json")
    results = _canonical_json_file(root / "probe_results.json")
    features = _read_canonical_jsonl(root / "features.jsonl")
    source = resolved.get("diagnostic_source", {})
    runtime = resolved.get("runtime", {})
    if (
        manifest.get("schema_version") != MEMORY_COUNT_ARTIFACT_VERSION
        or manifest.get("diagnostic_version") != MEMORY_COUNT_PROBE_VERSION
        or resolved.get("diagnostic_version") != MEMORY_COUNT_PROBE_VERSION
        or resolved.get("artifact_version") != MEMORY_COUNT_ARTIFACT_VERSION
        or source.get("git_commit") != expected_commit
        or source.get("git_branch") is not None
        or source.get("detached_head") is not True
        or source.get("git_dirty") is not False
        or runtime.get("slurm_job_id") != job_id
        or runtime.get("python") != "3.8.13"
        or str(runtime.get("pytorch", "")).split("+")[0] != "1.11.0"
        or runtime.get("device") != "cpu"
        or runtime.get("cuda_available") is not False
        or runtime.get("cpu_threads") != 1
        or resolved.get("access") != MEMORY_COUNT_ACCESS_RECORD
        or results.get("diagnostic_version") != MEMORY_COUNT_PROBE_VERSION
        or results.get("feature_row_count") != len(features)
        or len(features) != len(ARMS) * len(FULL_SEEDS) * TRAIN_FAMILY_COUNT
        or results.get("outcome_controls_artifact_validity") is not False
    ):
        _fail("invalid_memory_count_probe_artifact", "content differs")
    return {
        "diagnostic_version": MEMORY_COUNT_PROBE_VERSION,
        "artifact_version": MEMORY_COUNT_ARTIFACT_VERSION,
        "source_commit": expected_commit,
        "slurm_job_id": job_id,
        "feature_row_count": len(features),
        "probe_result_count": len(results.get("results", ())),
        "prerequisite_pass": bool(results.get("prerequisite_pass")),
        "sha256sums_sha256": _file_sha256(root / "SHA256SUMS"),
        "verification_status": "pass",
    }


def _finalize_memory_count_artifact(staging, final):
    staging = Path(staging)
    final = Path(final)
    ordinary = _regular_artifact_files(
        staging, excluded=("artifact_manifest.json", "SHA256SUMS")
    )
    if ordinary != ("features.jsonl", "probe_results.json", "resolved_config.json"):
        _fail("incomplete_memory_count_probe_artifact", "ordinary files differ")
    manifest = {
        "schema_version": MEMORY_COUNT_ARTIFACT_VERSION,
        "diagnostic_version": MEMORY_COUNT_PROBE_VERSION,
        "artifacts": [{
            "path": relative,
            "byte_size": (staging / relative).stat().st_size,
            "sha256": _file_sha256(staging / relative),
        } for relative in ordinary],
    }
    _atomic_write_json(staging / "artifact_manifest.json", manifest)
    checksum_paths = _regular_artifact_files(staging, excluded=("SHA256SUMS",))
    _atomic_write_bytes(
        staging / "SHA256SUMS",
        "".join(
            "{}  {}\n".format(_file_sha256(staging / name), name)
            for name in checksum_paths
        ).encode("utf-8"),
    )
    resolved = _canonical_json_file(staging / "resolved_config.json")
    verify_memory_count_artifact(
        staging,
        expected_commit=resolved["diagnostic_source"]["git_commit"],
        expected_slurm_job_id=resolved["runtime"]["slurm_job_id"],
        allow_incomplete_name=True,
    )
    os.replace(str(staging), str(final))
    return final


def run_memory_count_probe(
    *, checkpoint_paths, train_index, train_root, prior_postmortem_artifact,
    output_dir, repository_root, expected_commit, slurm_job_id
):
    _require_runtime()
    job_id = _job_id(slurm_job_id, "invalid_memory_count_probe_job_id")
    source = _source_identity(repository_root, expected_commit)
    if source["detached_head"] is not True or source["git_dirty"] is not False:
        _fail("invalid_memory_count_probe_source", "detached clean checkout required")
    checkpoints = parse_checkpoint_paths(checkpoint_paths)
    prior_evidence, unused_prior = authenticate_prior_postmortem(
        prior_postmortem_artifact
    )
    del unused_prior
    repository = Path(repository_root).resolve()
    train_package = Path(train_index).resolve().parent
    train_payload_root = Path(train_root).resolve()
    prior_root = Path(prior_postmortem_artifact).resolve()
    retained_root = next(iter(checkpoints.values())).parent.resolve()
    if (
        Path(train_index).name != "index.json"
        or train_payload_root != train_package / "payloads"
        or any(_is_within(value, repository) for value in (
            train_package, prior_root, retained_root
        ))
    ):
        _fail("unsafe_memory_count_probe_inputs", "input roots differ")
    final, staging = _prepare_output(
        output_dir,
        (train_package, prior_root, retained_root),
        repository_root,
        job_id,
    )

    from .stage6_narrow_loader import load_stage6_train
    train_examples, train_input_evidence = load_stage6_train(train_index, train_root)
    train_input_evidence = _sanitized_train_evidence(train_input_evidence)
    if (
        len(train_examples) != TRAIN_FAMILY_COUNT
        or train_input_evidence != prior_evidence["train_input_evidence"]
    ):
        _fail("invalid_memory_count_probe_train", "train identity differs")
    train_family_ids = tuple(sorted(
        item.physical_family_id for item in train_examples
    ))
    producer_source_digest = source_commit_python_sha256(
        repository_root, PRODUCER_SOURCE_COMMIT
    )
    checkpoint_evidence = []
    feature_rows = []
    results = []
    for seed in FULL_SEEDS:
        for arm in ARMS:
            name = "stage6-{}-seed{}.pt".format(arm, seed)
            before = _file_sha256(checkpoints[name])
            model, unused_training, evidence = _load_retained_checkpoint(
                checkpoints[name],
                expected_wrapper_sha256=EXPECTED_CHECKPOINT_SHA256[name],
                arm=arm,
                seed=seed,
                train_family_ids=train_family_ids,
                train_input_evidence=train_input_evidence,
                producer_source_digest=producer_source_digest,
            )
            del unused_training
            rows = extract_count_features(
                model, train_examples, arm=arm, seed=seed
            )
            if _file_sha256(checkpoints[name]) != before:
                _fail("memory_count_probe_checkpoint_modified", name)
            checkpoint_evidence.append(evidence)
            feature_rows.extend(rows)
            for feature_name in MEMORY_COUNT_FEATURES:
                result = fit_count_probe(rows, feature_name)
                result.update({"arm": arm, "seed": int(seed)})
                results.append(result)
    aggregate = {}
    for arm in ARMS:
        aggregate[arm] = {}
        for feature_name in MEMORY_COUNT_FEATURES:
            selected = [
                row for row in results
                if row["arm"] == arm and row["feature_name"] == feature_name
            ]
            aggregate[arm][feature_name] = {
                "mean_balanced_accuracy": sum(
                    row["test_metrics"]["balanced_accuracy"] for row in selected
                ) / float(len(selected)),
                "all_seeds_linearly_recoverable": all(
                    row["linearly_recoverable"] for row in selected
                ),
            }
    prerequisite = all(
        row["linearly_recoverable"] for row in results
    )
    resolved = {
        "diagnostic_version": MEMORY_COUNT_PROBE_VERSION,
        "artifact_version": MEMORY_COUNT_ARTIFACT_VERSION,
        "producer_source_commit": PRODUCER_SOURCE_COMMIT,
        "producer_source_tree_sha256": producer_source_digest,
        "producer_job_id": PRODUCER_JOB_ID,
        "prior_postmortem_evidence": prior_evidence,
        "diagnostic_source": source,
        "runtime": {
            "python": platform.python_version(),
            "pytorch": str(torch.__version__),
            "device": "cpu",
            "cuda_available": bool(torch.cuda.is_available()),
            "cpu_threads": int(torch.get_num_threads()),
            "host": socket.gethostname(),
            "slurm_job_id": job_id,
        },
        "train_input_evidence": train_input_evidence,
        "checkpoint_evidence": checkpoint_evidence,
        "probe_contract": {
            "classes": list(MEMORY_COUNT_CLASSES),
            "features": list(MEMORY_COUNT_FEATURES),
            "regularization": MEMORY_COUNT_REGULARIZATION,
            "max_iter": MEMORY_COUNT_MAX_ITER,
            "accuracy_minimum": MEMORY_COUNT_ACCURACY_MIN,
            "control_margin_minimum": MEMORY_COUNT_CONTROL_MARGIN_MIN,
            "split_within_train": True,
            "target_model_parameters_modified": False,
        },
        "access": dict(MEMORY_COUNT_ACCESS_RECORD),
    }
    probe_results = {
        "diagnostic_version": MEMORY_COUNT_PROBE_VERSION,
        "feature_row_count": len(feature_rows),
        "results": results,
        "aggregate": aggregate,
        "prerequisite_pass": prerequisite,
        "interpretation": (
            "node_count_linearly_recoverable_from_both_feature_spaces_all_checkpoints"
            if prerequisite else
            "node_count_not_linearly_recoverable_under_frozen_rule"
        ),
        "outcome_controls_artifact_validity": False,
    }
    _atomic_write_json(staging / "resolved_config.json", resolved)
    _atomic_write_jsonl(staging / "features.jsonl", feature_rows)
    _atomic_write_json(staging / "probe_results.json", probe_results)
    _verify_source_unchanged(source, repository_root, expected_commit)
    _finalize_memory_count_artifact(staging, final)
    verified = verify_memory_count_artifact(
        final,
        expected_commit=expected_commit,
        expected_slurm_job_id=job_id,
    )
    return {
        "event": "memory_count_probe_completed",
        "artifact_path": str(final),
        "verification": verified,
        "prerequisite_pass": prerequisite,
        "aggregate": aggregate,
        "outcome_controls_artifact_validity": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--train-index", required=True)
    parser.add_argument("--train-root", required=True)
    parser.add_argument("--prior-postmortem-artifact", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--producer-job-id", required=True)
    parser.add_argument("--prior-postmortem-job-id", required=True)
    parser.add_argument("--slurm-job-id", default=os.environ.get("SLURM_JOB_ID"))
    args = parser.parse_args(argv)
    if _job_id(args.producer_job_id) != PRODUCER_JOB_ID:
        _fail("invalid_stage6_zero_memory_producer_job", args.producer_job_id)
    if _job_id(args.prior_postmortem_job_id) != PRIOR_POSTMORTEM_JOB_ID:
        _fail(
            "invalid_stage6_zero_memory_prior_job",
            args.prior_postmortem_job_id,
        )
    try:
        result = run_memory_count_probe(
            checkpoint_paths=args.checkpoint,
            train_index=args.train_index,
            train_root=args.train_root,
            prior_postmortem_artifact=args.prior_postmortem_artifact,
            output_dir=args.output_dir,
            repository_root=args.repository_root,
            expected_commit=args.expected_commit,
            slurm_job_id=args.slurm_job_id,
        )
    except Exception as exc:
        failure = {
            "event": "memory_count_probe_infrastructure_failure",
            "failure_type": type(exc).__name__,
            "failure_code": getattr(exc, "code", None),
            "detail": str(exc),
            "diagnostic_completed": False,
            "ge1_training_performed": False,
            "ge1_backward_pass_performed": False,
            "ge1_optimizer_step_performed": False,
            "ge1_checkpoint_modified": False,
            "development_accessed": False,
        }
        print(_canonical_json_text(failure), file=sys.stderr, flush=True)
        return 1
    print(_canonical_json_text(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
