"""Read-only train-gate postmortem for failed ADR-0014 producer job 3354961."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys

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
    CONDITIONS,
    FULL_SEEDS,
    MEMORY_RATIO_THRESHOLD,
    PROTOCOL_VERSION,
    TRAIN_FAMILY_COUNT,
    score_family_record,
    structure_memory_gate_for_cohort,
)
from .stage6_structure_only_producer import (
    BATCH_SIZE,
    CHECKPOINT_VERSION,
    CLIP_NORM,
    EPOCHS,
    LEARNING_RATE,
    TRAIN_CEILING_SHORTFALL_DIFFERENCE_MAX,
    WEIGHT_DECAY,
    generate_autonomous_records,
    load_stage6_checkpoint,
    optimization_reliability as _current_optimization_reliability,
    validate_checkpoint_identity,
)


DIAGNOSTIC_VERSION = "GE1-STAGE6-TRAIN-GATE-POSTMORTEM-v1"
ARTIFACT_VERSION = "GE1-STAGE6-TRAIN-GATE-POSTMORTEM-ARTIFACT-v1"
PRODUCER_SOURCE_COMMIT = "5f4542f86756dae44435af27a6e072db6f27a8ef"
PRODUCER_JOB_ID = "3354961"
HISTORICAL_INNER_SLURM_JOB_ID = None
EXPECTED_OPTIMIZER_STEPS = EPOCHS * ((TRAIN_FAMILY_COUNT + BATCH_SIZE - 1) // BATCH_SIZE)
EXPECTED_TRAINING_PRESENTATIONS = EPOCHS * TRAIN_FAMILY_COUNT
EXPECTED_FAMILY_RECORD_COUNT = (
    len(ARMS) * len(FULL_SEEDS) * len(CONDITIONS) * TRAIN_FAMILY_COUNT
)
ORDINARY_FILES = (
    "resolved_config.json",
    "train_family_records.jsonl",
    "training_histories.jsonl",
    "summary.json",
)
ARTIFACT_FILES = ORDINARY_FILES + ("artifact_manifest.json", "SHA256SUMS")
EXPECTED_WRAPPER_NAMES = tuple(
    "stage6-{}-seed{}.pt".format(arm, seed)
    for seed in FULL_SEEDS for arm in ARMS
)
ACCESS_RECORD = {
    "authorized_narrow_train_accessed": True,
    "retained_job_checkpoint_wrappers_accessed": True,
    "train_only_autonomous_inference_performed": True,
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
    "manifest_accessed": False,
    "unrelated_checkpoint_accessed": False,
    "model_artifact_accessed": False,
    "repaired_artifact_accessed": False,
    "scientific_repair_performed": False,
}
PRODUCER_EXECUTION_EVIDENCE = {
    "job_id": PRODUCER_JOB_ID,
    "source_commit": PRODUCER_SOURCE_COMMIT,
    "slurm_state": "FAILED",
    "exit_code": "42:0",
    "elapsed": "02:07:20",
    "max_rss": "855460K",
    "allocated_cpus": 1,
    "focused_tests_declared": 49,
    "focused_tests_run": 49,
    "focused_tests_skipped": 0,
    "complete_tests_declared": 620,
    "complete_tests_run": 620,
    "complete_tests_failed": 0,
    "complete_tests_errored": 0,
    "complete_tests_expected_cuda_skips": 6,
    "scientific_failure_code": "stage6_train_reliability_failure",
    "development_accessed": False,
    "checkpoint_wrapper_count": 6,
    "inner_checkpoint_slurm_job_id": HISTORICAL_INNER_SLURM_JOB_ID,
    "inner_checkpoint_job_id_not_forwarded_by_cleanenv": True,
    "evidence_source": (
        "reviewed supplied stdout, stderr, sacct transcript, and "
        "retained-work inventory"
    ),
}


def _fail(code, detail):
    raise GraphEncoderError(code, detail)


def _job_id(value, code="invalid_stage6_postmortem_job_id"):
    value = str(value)
    if not value.isascii() or not value.isdecimal() or int(value) <= 0:
        _fail(code, "positive decimal job ID required")
    return value


def parse_checkpoint_hashes(values):
    """Parse exactly one reviewed SHA-256 for every retained wrapper."""

    result = {}
    for value in values:
        if not isinstance(value, str) or value.count("=") != 1:
            _fail("invalid_stage6_postmortem_hashes", "NAME=SHA256 required")
        name, digest = value.split("=", 1)
        if (
            name not in EXPECTED_WRAPPER_NAMES
            or name in result
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            _fail("invalid_stage6_postmortem_hashes", value)
        result[name] = digest
    if tuple(sorted(result)) != tuple(sorted(EXPECTED_WRAPPER_NAMES)):
        _fail("invalid_stage6_postmortem_hashes", "wrapper matrix differs")
    return result


def source_commit_python_sha256(repository_root, commit):
    """Recompute the historical prototype-Python digest from Git objects."""

    root = Path(repository_root).resolve()
    if (
        not isinstance(commit, str)
        or len(commit) != 40
        or any(character not in "0123456789abcdef" for character in commit)
    ):
        _fail("invalid_stage6_postmortem_source", "commit malformed")
    try:
        listing = subprocess.check_output(
            ("git", "ls-tree", "-r", "--name-only", commit, "--", "prototype"),
            cwd=str(root),
        ).decode("utf-8")
    except (OSError, subprocess.SubprocessError, UnicodeError) as exc:
        raise GraphEncoderError(
            "invalid_stage6_postmortem_source", "historical tree unavailable"
        ) from exc
    paths = tuple(sorted(
        line for line in listing.splitlines()
        if line.endswith(".py") and "__pycache__" not in Path(line).parts
    ))
    if not paths:
        _fail("invalid_stage6_postmortem_source", "historical Python tree empty")
    digest = hashlib.sha256()
    for relative in paths:
        try:
            content = subprocess.check_output(
                ("git", "show", "{}:{}".format(commit, relative)), cwd=str(root)
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise GraphEncoderError(
                "invalid_stage6_postmortem_source", relative
            ) from exc
        name = relative.encode("utf-8")
        digest.update(len(name).to_bytes(8, byteorder="big"))
        digest.update(name)
        digest.update(len(content).to_bytes(8, byteorder="big"))
        digest.update(content)
    return digest.hexdigest()


def _require_runtime():
    if torch is None:
        raise RuntimeError("Stage 6 train-gate postmortem requires PyTorch")
    if sys.version_info[:3] != (3, 8, 13):
        _fail("environment_mismatch", "Python 3.8.13 required")
    if str(torch.__version__).split("+")[0] != "1.11.0":
        _fail("environment_mismatch", "PyTorch 1.11.0 required")
    if torch.cuda.is_available() or torch.get_num_threads() != 1:
        _fail("environment_mismatch", "one-thread CPU execution required")


def _expected_training_arithmetic():
    return {
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "optimizer": "AdamW",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "gradient_clip_norm": CLIP_NORM,
        "optimizer_steps": EXPECTED_OPTIMIZER_STEPS,
        "training_example_presentations": EXPECTED_TRAINING_PRESENTATIONS,
    }


def _validate_checkpoint_identity(
    identity, *, arm, seed, model_config, parameter_count, partition_identity,
    train_input_evidence, producer_source_digest,
):
    validate_checkpoint_identity(identity, expected=identity)
    expected_hashes = {
        "index_identity_sha256": train_input_evidence["index_identity_sha256"],
        "payload_digests_sha256": train_input_evidence[
            "observed_payload_digests_sha256"
        ],
    }
    direct = {
        "source_commit": PRODUCER_SOURCE_COMMIT,
        "source_digest": producer_source_digest,
        "arm": arm,
        "seed": seed,
        "epoch": EPOCHS,
        "model_config": model_config,
        "training_partition_identity": partition_identity,
        "training_partition_hashes": expected_hashes,
        "training_arithmetic": _expected_training_arithmetic(),
        "parameter_count": parameter_count,
        "checkpoint_schema": model_config["checkpoint_schema"],
        "execution_device": "cpu",
        "governed_map_location": "cpu",
        "cuda_rng_preserved": False,
        "external_checkpoint": False,
        "warm_start": False,
        "checkpoint_reuse": False,
    }
    for name, expected in direct.items():
        if identity.get(name) != expected:
            _fail("invalid_stage6_postmortem_checkpoint", name + " differs")
    runtime = identity["runtime_identity"]
    if (
        runtime.get("execution_device") != "cpu"
        or runtime.get("python_version") != "3.8.13"
        or str(runtime.get("torch_version", "")).split("+")[0] != "1.11.0"
        or runtime.get("cpu_threads") != 1
        or runtime.get("gpu") is not None
        or runtime.get("deterministic_algorithms") is not True
        or runtime.get("tf32_matmul_allowed") is not False
        or runtime.get("tf32_cudnn_allowed") is not False
        or runtime.get("cuda_rng_preservation_required") is not False
    ):
        _fail("invalid_stage6_postmortem_checkpoint", "runtime differs")
    from .stage6_device import timing_hardware_identity
    if identity["timing_hardware_identity"] != timing_hardware_identity(runtime):
        _fail("invalid_stage6_postmortem_checkpoint", "hardware identity differs")
    return True


def _training_record(resume, *, arm, seed, wrapper_name, wrapper_sha256):
    losses = [float(row.mean_loss) for row in resume.epoch_records]
    gradients = []
    for row in resume.epoch_records:
        values = tuple(float(value) for value in row.pre_clip_gradient_norms)
        if not values:
            _fail("invalid_stage6_postmortem_checkpoint", "gradient history empty")
        gradients.append(max(values))
    from .training import plateau_state
    recomputed_plateau = plateau_state(losses)
    if resume.plateau_state.to_dict() != recomputed_plateau.to_dict():
        _fail("invalid_stage6_postmortem_checkpoint", "plateau state differs")
    return {
        "arm": arm,
        "seed": seed,
        "completed_epoch": resume.completed_epoch,
        "checkpoint_epoch": EPOCHS,
        "optimizer_steps": resume.optimizer_step_count,
        "training_example_presentations": resume.training_example_presentations,
        "epoch_losses": losses,
        "epoch_gradient_norms": gradients,
        "wrapper_name": wrapper_name,
        "wrapper_sha256": wrapper_sha256,
        "diagnostic_read_only_recovery": True,
        "training_repeated": False,
        "optimizer_step_performed": False,
    }


def _validate_retained_inner_checkpoint(
    resume, *, arm, seed, train_family_ids, producer_source_digest,
):
    """Validate every retained inner field with field-specific failures."""

    def require(name, observed, expected):
        if observed != expected:
            _fail(
                "invalid_stage6_postmortem_checkpoint",
                "inner checkpoint field {} differs".format(name),
            )

    require("completed_epoch", resume.completed_epoch, EPOCHS)
    require(
        "optimizer_step_count",
        resume.optimizer_step_count,
        EXPECTED_OPTIMIZER_STEPS,
    )
    require(
        "training_example_presentations",
        resume.training_example_presentations,
        EXPECTED_TRAINING_PRESENTATIONS,
    )
    require("epoch_records.count", len(resume.epoch_records), EPOCHS)
    require("family_order_history.count", len(resume.family_order_history), EPOCHS)
    expected_families = tuple(train_family_ids)
    for index, order in enumerate(resume.family_order_history):
        try:
            observed_families = tuple(sorted(order))
        except (TypeError, ValueError):
            observed_families = None
        require(
            "family_order_history[{}]".format(index),
            observed_families,
            expected_families,
        )

    payload = resume.payload
    if not isinstance(payload, dict):
        _fail(
            "invalid_stage6_postmortem_checkpoint",
            "inner checkpoint field payload differs",
        )
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        _fail(
            "invalid_stage6_postmortem_checkpoint",
            "inner checkpoint field provenance differs",
        )
    expected_provenance = {
        "git_commit": PRODUCER_SOURCE_COMMIT,
        "source_tree_sha256": producer_source_digest,
        "device": "cpu",
        "slurm_job_id": HISTORICAL_INNER_SLURM_JOB_ID,
        "encoder_arm": arm,
        "seed": seed,
    }
    for name, expected in expected_provenance.items():
        if name not in provenance:
            _fail(
                "invalid_stage6_postmortem_checkpoint",
                "inner checkpoint field provenance.{} is absent".format(name),
            )
        require("provenance." + name, provenance[name], expected)

    rng_states = payload.get("rng_states")
    if not isinstance(rng_states, dict):
        _fail(
            "invalid_stage6_postmortem_checkpoint",
            "inner checkpoint field rng_states differs",
        )
    if "torch_cuda" not in rng_states:
        _fail(
            "invalid_stage6_postmortem_checkpoint",
            "inner checkpoint field rng_states.torch_cuda is absent",
        )
    require("rng_states.torch_cuda", rng_states["torch_cuda"], None)
    return True


def _load_retained_checkpoint(
    path, *, expected_wrapper_sha256, arm, seed, train_family_ids,
    train_input_evidence, producer_source_digest,
):
    from .config import GE1TrainingConfig, grid_frozen_encoder_config
    from .model import build_ge1_model
    from .provenance import training_partition_identity

    target = Path(path)
    expected_name = "stage6-{}-seed{}.pt".format(arm, seed)
    if target.name != expected_name or target.is_symlink() or not target.is_file():
        _fail("invalid_stage6_postmortem_checkpoint", expected_name)
    observed_wrapper_sha256 = _file_sha256(target)
    if observed_wrapper_sha256 != expected_wrapper_sha256:
        _fail("invalid_stage6_postmortem_checkpoint", "wrapper checksum differs")
    wrapper = torch.load(str(target), map_location="cpu")
    if (
        not isinstance(wrapper, dict)
        or set(wrapper) != {"version", "identity", "training_checkpoint"}
        or wrapper.get("version") != CHECKPOINT_VERSION
    ):
        _fail("invalid_stage6_postmortem_checkpoint", "wrapper structure differs")
    config = grid_frozen_encoder_config(arm, seed)
    model = build_ge1_model(config).to("cpu")
    parameter_count = sum(
        value.numel() for value in model.parameters() if value.requires_grad
    )
    partition_identity = training_partition_identity(train_family_ids)
    identity = wrapper["identity"]
    _validate_checkpoint_identity(
        identity,
        arm=arm,
        seed=seed,
        model_config=config.to_dict(),
        parameter_count=parameter_count,
        partition_identity=partition_identity,
        train_input_evidence=train_input_evidence,
        producer_source_digest=producer_source_digest,
    )
    training_config = GE1TrainingConfig()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    resume = load_stage6_checkpoint(
        target,
        expected_identity=identity,
        model=model,
        optimizer=optimizer,
        training_config=training_config,
        partition_identity=partition_identity,
        restore_rng=False,
        execution_device="cpu",
        expected_wrapper_sha256=expected_wrapper_sha256,
    )
    _validate_retained_inner_checkpoint(
        resume,
        arm=arm,
        seed=seed,
        train_family_ids=train_family_ids,
        producer_source_digest=producer_source_digest,
    )
    record = _training_record(
        resume, arm=arm, seed=seed, wrapper_name=expected_name,
        wrapper_sha256=observed_wrapper_sha256,
    )
    evidence = {
        "wrapper_name": expected_name,
        "stage6_wrapper_sha256": observed_wrapper_sha256,
        "identity": identity,
        "strict_recovery_passed": True,
        "checkpoint_modified": False,
    }
    return model, record, evidence


def _validate_train_scored_matrix(scored):
    expected_groups = {
        (arm, seed, condition)
        for arm in ARMS for seed in FULL_SEEDS for condition in CONDITIONS
    }
    grouped = defaultdict(list)
    for row in scored:
        if row.get("cohort") != "train":
            _fail("invalid_stage6_postmortem_records", "non-train row")
        grouped[(row["arm"], row["seed"], row["condition"])].append(row)
    if set(grouped) != expected_groups:
        _fail("invalid_stage6_postmortem_records", "group matrix differs")
    family_set = None
    for key, rows in grouped.items():
        observed = frozenset(row["family_id"] for row in rows)
        if len(rows) != TRAIN_FAMILY_COUNT or len(observed) != TRAIN_FAMILY_COUNT:
            _fail("invalid_stage6_postmortem_records", str(key))
        if family_set is None:
            family_set = observed
        elif observed != family_set:
            _fail("invalid_stage6_postmortem_records", "family alignment differs")
        batches = defaultdict(set)
        for row in rows:
            batches[row["batch_identity"]].add(row["family_id"])
        for row in rows:
            if (
                row["condition"] == "P_shuffle"
                and row["memory_source_family_id"]
                not in batches[row["batch_identity"]]
            ):
                _fail("invalid_stage6_postmortem_records", "shuffle crosses batch")
    return True


def _predicate(name, observed, operator, threshold, passed):
    return {
        "name": name,
        "observed": observed,
        "operator": operator,
        "threshold": threshold,
        "pass": bool(passed),
    }


def _historical_optimization_reliability(training_runs, train_scores, seeds):
    """Reconstruct the job-3354961 gate after its prospective removal.

    ADR-0016 changes only the future producer.  This v1 postmortem remains an
    exact verifier of the historical artifact and therefore reattaches the old
    threshold to the now-diagnostic comparison rows locally.
    """

    current = _current_optimization_reliability(
        training_runs, train_scores, seeds
    )
    comparisons = []
    for source in current["train_ceiling_comparisons"]:
        row = dict(source)
        row["maximum_inclusive"] = TRAIN_CEILING_SHORTFALL_DIFFERENCE_MAX
        row["pass"] = (
            row["absolute_shortfall_difference"]
            <= TRAIN_CEILING_SHORTFALL_DIFFERENCE_MAX
        )
        comparisons.append(row)
    result = dict(current)
    result["train_ceiling_comparisons"] = comparisons
    result["pass"] = (
        all(row["pass_before_train_ceiling"] for row in current["runs"])
        and all(row["pass"] for row in comparisons)
    )
    return result


def analyze_postmortem(training_runs, family_records):
    """Recompute the exact failed train-side gates without changing them."""

    scored = [score_family_record(row) for row in family_records]
    _validate_train_scored_matrix(scored)
    runs = {(row["arm"], row["seed"]): row for row in training_runs}
    expected_runs = {(arm, seed) for arm in ARMS for seed in FULL_SEEDS}
    if set(runs) != expected_runs or len(training_runs) != len(expected_runs):
        _fail("invalid_stage6_postmortem_training", "run matrix differs")
    train_scores = {}
    for arm, seed in sorted(expected_runs):
        values = [
            row["score"]["normalized_structural_prefix"] for row in scored
            if row["arm"] == arm and row["seed"] == seed
            and row["condition"] == "P_true"
        ]
        if len(values) != TRAIN_FAMILY_COUNT:
            _fail("invalid_stage6_postmortem_records", "P_true matrix differs")
        train_scores[(arm, seed)] = sum(values) / float(len(values))
    reliability = _historical_optimization_reliability(
        list(training_runs), train_scores, FULL_SEEDS
    )
    reliability_runs = {
        (row["arm"], row["seed"]): row for row in reliability["runs"]
    }
    ceiling_by_seed = {
        row["seed"]: row for row in reliability["train_ceiling_comparisons"]
    }
    memory_overall = structure_memory_gate_for_cohort(
        scored, FULL_SEEDS, "train"
    )
    memory_by_seed = {
        str(seed): structure_memory_gate_for_cohort(scored, (seed,), "train")
        for seed in FULL_SEEDS
    }
    rows = []
    for seed in FULL_SEEDS:
        ceiling = ceiling_by_seed[seed]
        for arm in ARMS:
            source = runs[(arm, seed)]
            optimization = reliability_runs[(arm, seed)]
            memory = memory_by_seed[str(seed)][arm]
            finite_losses = (
                len(source.get("epoch_losses", ())) == EPOCHS
                and all(math.isfinite(float(value))
                        for value in source.get("epoch_losses", ()))
            )
            finite_gradients = (
                len(source.get("epoch_gradient_norms", ())) == EPOCHS
                and all(math.isfinite(float(value))
                        for value in source.get("epoch_gradient_norms", ()))
            )
            predicates = [
                _predicate(
                    "finite_losses", finite_losses, "is", True, finite_losses
                ),
                _predicate(
                    "finite_gradients", finite_gradients, "is", True,
                    finite_gradients,
                ),
                _predicate(
                    "plateau_by_epoch_200",
                    optimization["plateau_by_epoch_200"], "is", True,
                    optimization["plateau_by_epoch_200"],
                ),
                _predicate(
                    "fixed_epoch_200_checkpoint",
                    optimization["fixed_epoch_200_checkpoint"], "is", True,
                    optimization["fixed_epoch_200_checkpoint"],
                ),
                _predicate(
                    "per_seed_shortfall_difference",
                    ceiling["absolute_shortfall_difference"], "<=",
                    TRAIN_CEILING_SHORTFALL_DIFFERENCE_MAX, ceiling["pass"],
                ),
                _predicate("P_true", memory["P_true"], ">", 0.0,
                           memory["P_true"] > 0.0),
                _predicate(
                    "shuffle_over_true", memory["R_shuffle"], "<=",
                    MEMORY_RATIO_THRESHOLD,
                    memory["R_shuffle"] is not None
                    and memory["R_shuffle"] <= MEMORY_RATIO_THRESHOLD,
                ),
                _predicate(
                    "mean_over_true", memory["R_mean"], "<=",
                    MEMORY_RATIO_THRESHOLD,
                    memory["R_mean"] is not None
                    and memory["R_mean"] <= MEMORY_RATIO_THRESHOLD,
                ),
            ]
            rows.append({
                "arm": arm,
                "seed": seed,
                "finite_loss_status": finite_losses,
                "finite_gradient_status": finite_gradients,
                "finite_losses_and_gradients": optimization[
                    "finite_losses_and_gradients"
                ],
                "first_plateau_epoch": optimization["first_plateau_epoch"],
                "plateau_by_epoch_200": optimization["plateau_by_epoch_200"],
                "fixed_epoch_200_checkpoint": optimization[
                    "fixed_epoch_200_checkpoint"
                ],
                "true_memory_train_structural_score": train_scores[(arm, seed)],
                "train_ceiling_shortfall": ceiling[
                    "structural_train_ceiling_shortfalls"
                ][arm],
                "per_seed_flat_graph_shortfall_difference": ceiling[
                    "absolute_shortfall_difference"
                ],
                "per_seed_shortfall_difference_maximum_inclusive": (
                    TRAIN_CEILING_SHORTFALL_DIFFERENCE_MAX
                ),
                "per_seed_shortfall_gate_pass": ceiling["pass"],
                "P_true": memory["P_true"],
                "P_shuffle": memory["P_shuffle"],
                "P_mean": memory["P_mean"],
                "shuffle_over_true": memory["R_shuffle"],
                "mean_over_true": memory["R_mean"],
                "memory_ratio_maximum_inclusive": MEMORY_RATIO_THRESHOLD,
                "per_seed_memory_gate_pass": memory["pass"],
                "predicates": predicates,
                "failed_predicates": [
                    value["name"] for value in predicates if not value["pass"]
                ],
            })
    failed_optimization_runs = [
        "{}:{}".format(row["arm"], row["seed"])
        for row in reliability["runs"] if not row["pass_before_train_ceiling"]
    ]
    failed_ceiling_seeds = [
        row["seed"] for row in reliability["train_ceiling_comparisons"]
        if not row["pass"]
    ]
    failed_memory_arms = [
        arm for arm in ARMS if not memory_overall[arm]["pass"]
    ]
    failed_components = []
    if failed_optimization_runs:
        failed_components.append("optimization_reliability")
    if failed_ceiling_seeds:
        failed_components.append("per_seed_train_ceiling")
    if failed_memory_arms:
        failed_components.append("train_memory_use_gate")
    producer_predicates = []
    for row in reliability["runs"]:
        prefix = "optimization.{}.{}".format(row["arm"], row["seed"])
        producer_predicates.extend((
            _predicate(
                prefix + ".finite_losses_and_gradients",
                row["finite_losses_and_gradients"], "is", True,
                row["finite_losses_and_gradients"],
            ),
            _predicate(
                prefix + ".plateau_by_epoch_200",
                row["plateau_by_epoch_200"], "is", True,
                row["plateau_by_epoch_200"],
            ),
            _predicate(
                prefix + ".fixed_epoch_200_checkpoint",
                row["fixed_epoch_200_checkpoint"], "is", True,
                row["fixed_epoch_200_checkpoint"],
            ),
        ))
    for row in reliability["train_ceiling_comparisons"]:
        producer_predicates.append(_predicate(
            "train_ceiling.seed{}.absolute_shortfall_difference".format(
                row["seed"]
            ),
            row["absolute_shortfall_difference"], "<=",
            TRAIN_CEILING_SHORTFALL_DIFFERENCE_MAX, row["pass"],
        ))
    for arm in ARMS:
        memory = memory_overall[arm]
        producer_predicates.extend((
            _predicate(
                "train_memory.{}.P_true".format(arm), memory["P_true"],
                ">", 0.0, memory["P_true"] > 0.0,
            ),
            _predicate(
                "train_memory.{}.shuffle_over_true".format(arm),
                memory["R_shuffle"], "<=", MEMORY_RATIO_THRESHOLD,
                memory["R_shuffle"] is not None
                and memory["R_shuffle"] <= MEMORY_RATIO_THRESHOLD,
            ),
            _predicate(
                "train_memory.{}.mean_over_true".format(arm),
                memory["R_mean"], "<=", MEMORY_RATIO_THRESHOLD,
                memory["R_mean"] is not None
                and memory["R_mean"] <= MEMORY_RATIO_THRESHOLD,
            ),
        ))
    return {
        "diagnostic_version": DIAGNOSTIC_VERSION,
        "producer_source_commit": PRODUCER_SOURCE_COMMIT,
        "producer_job_id": PRODUCER_JOB_ID,
        "arms_and_seeds": rows,
        "optimization_reliability": reliability,
        "train_memory_gate_by_arm": memory_overall,
        "train_memory_gate_by_seed_and_arm": memory_by_seed,
        "failed_optimization_runs": failed_optimization_runs,
        "failed_train_ceiling_seeds": failed_ceiling_seeds,
        "failed_train_memory_arms": failed_memory_arms,
        "exact_producer_predicates": producer_predicates,
        "exact_failed_producer_predicates": [
            row["name"] for row in producer_predicates if not row["pass"]
        ],
        "minimal_combined_reason": {
            "stage6_error_code": "stage6_train_reliability_failure",
            "failed_components": failed_components,
            "development_remained_closed": True,
        },
        "reproduces_stage6_train_reliability_failure": bool(failed_components),
        "all_train_side_gates_pass": not bool(failed_components),
        "outcome_is_artifact_validity_gate": False,
        "encoder_superiority_inferred": False,
    }


def _canonical_json_file(path):
    raw = Path(path).read_bytes()
    if not raw.endswith(b"\n"):
        _fail("invalid_stage6_postmortem_artifact", "final LF absent")
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise GraphEncoderError(
            "invalid_stage6_postmortem_artifact", "JSON unreadable"
        ) from exc
    if raw.decode("utf-8") != _canonical_json_text(value) + "\n":
        _fail("invalid_stage6_postmortem_artifact", "JSON not canonical")
    return value


def _prepare_output(output_dir, repository_root, retained_root, train_package_root,
                    job_id):
    final = Path(output_dir).resolve()
    if (
        final.exists() or final.is_symlink() or not final.parent.is_dir()
        or any(_is_within(final, Path(value).resolve()) for value in (
            repository_root, retained_root, train_package_root
        ))
    ):
        _fail("unsafe_stage6_postmortem_output", str(final))
    staging = final.with_name(final.name + ".incomplete-" + job_id)
    if staging.exists() or staging.is_symlink():
        _fail("unsafe_stage6_postmortem_output", str(staging))
    staging.mkdir()
    return final, staging


def finalize_artifact(staging_dir, final_dir):
    staging = Path(staging_dir)
    final = Path(final_dir)
    if staging.parent.resolve() != final.parent.resolve():
        _fail("unsafe_stage6_postmortem_output", "parents differ")
    ordinary = _regular_artifact_files(
        staging, excluded=("artifact_manifest.json", "SHA256SUMS")
    )
    if ordinary != tuple(sorted(ORDINARY_FILES)):
        _fail("incomplete_stage6_postmortem_artifact", "ordinary files differ")
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
        allow_incomplete_name=True,
    )
    os.replace(str(staging), str(final))
    return final


def _verify_checksums(root):
    manifest = _canonical_json_file(root / "artifact_manifest.json")
    if (
        manifest.get("schema_version") != ARTIFACT_VERSION
        or manifest.get("diagnostic_version") != DIAGNOSTIC_VERSION
    ):
        _fail("invalid_stage6_postmortem_artifact", "manifest identity differs")
    ordinary = _regular_artifact_files(
        root, excluded=("artifact_manifest.json", "SHA256SUMS")
    )
    listed = []
    for row in manifest.get("artifacts", ()):
        if set(row) != {"path", "byte_size", "sha256"}:
            _fail("invalid_stage6_postmortem_artifact", "manifest row differs")
        relative = _safe_relative_path(row["path"])
        target = root / relative
        if (
            target.is_symlink() or not target.is_file()
            or target.stat().st_size != row["byte_size"]
            or _file_sha256(target) != row["sha256"]
        ):
            _fail("stage6_postmortem_integrity_failure", relative)
        listed.append(relative)
    if tuple(listed) != ordinary:
        _fail("invalid_stage6_postmortem_artifact", "manifest coverage differs")
    raw = (root / "SHA256SUMS").read_bytes()
    if not raw.endswith(b"\n"):
        _fail("invalid_stage6_postmortem_artifact", "checksum LF absent")
    names = []
    for line in raw.decode("utf-8").splitlines():
        parts = line.split("  ", 1)
        if len(parts) != 2 or len(parts[0]) != 64:
            _fail("invalid_stage6_postmortem_artifact", "checksum row differs")
        relative = _safe_relative_path(parts[1])
        if _file_sha256(root / relative) != parts[0]:
            _fail("stage6_postmortem_integrity_failure", relative)
        names.append(relative)
    expected = _regular_artifact_files(root, excluded=("SHA256SUMS",))
    if tuple(names) != expected:
        _fail("invalid_stage6_postmortem_artifact", "checksum coverage differs")
    return manifest


def verify_artifact(path, *, expected_commit, expected_slurm_job_id,
                    allow_incomplete_name=False):
    job_id = _job_id(expected_slurm_job_id)
    root = Path(path)
    if not root.is_dir() or root.is_symlink():
        _fail("invalid_stage6_postmortem_artifact", "root differs")
    if ".incomplete-" in root.name and not allow_incomplete_name:
        _fail("invalid_stage6_postmortem_artifact", "incomplete root")
    observed = tuple(sorted(
        item.relative_to(root).as_posix()
        for item in root.rglob("*") if item.is_file()
    ))
    if observed != tuple(sorted(ARTIFACT_FILES)):
        _fail("invalid_stage6_postmortem_artifact", "file set differs")
    _verify_checksums(root)
    resolved = _canonical_json_file(root / "resolved_config.json")
    summary = _canonical_json_file(root / "summary.json")
    training = _read_canonical_jsonl(root / "training_histories.jsonl")
    families = _read_canonical_jsonl(root / "train_family_records.jsonl")
    if len(training) != len(ARMS) * len(FULL_SEEDS):
        _fail("invalid_stage6_postmortem_artifact", "training count differs")
    if len(families) != EXPECTED_FAMILY_RECORD_COUNT:
        _fail("invalid_stage6_postmortem_artifact", "family count differs")
    analysis = analyze_postmortem(training, families)
    source = resolved.get("diagnostic_source", {})
    runtime = resolved.get("runtime", {})
    checkpoints = resolved.get("checkpoint_evidence", ())
    train_evidence = resolved.get("train_input_evidence", {})
    producer_source_digest = resolved.get("producer_source_tree_sha256")
    if (
        resolved.get("diagnostic_version") != DIAGNOSTIC_VERSION
        or resolved.get("artifact_version") != ARTIFACT_VERSION
        or resolved.get("protocol_version") != PROTOCOL_VERSION
        or resolved.get("producer_source_commit") != PRODUCER_SOURCE_COMMIT
        or resolved.get("producer_job_id") != PRODUCER_JOB_ID
        or resolved.get("producer_execution_evidence")
        != PRODUCER_EXECUTION_EVIDENCE
        or not isinstance(producer_source_digest, str)
        or len(producer_source_digest) != 64
        or any(character not in "0123456789abcdef"
               for character in producer_source_digest)
        or train_evidence.get("verification_status") != "pass"
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
        or len(checkpoints) != len(ARMS) * len(FULL_SEEDS)
        or {(row.get("identity", {}).get("arm"),
             row.get("identity", {}).get("seed")) for row in checkpoints}
        != {(arm, seed) for arm in ARMS for seed in FULL_SEEDS}
        or {row.get("wrapper_name") for row in checkpoints}
        != set(EXPECTED_WRAPPER_NAMES)
    ):
        _fail("invalid_stage6_postmortem_artifact", "resolved identity differs")
    for row in checkpoints:
        identity = row.get("identity", {})
        expected_name = "stage6-{}-seed{}.pt".format(
            identity.get("arm"), identity.get("seed")
        )
        digest = row.get("stage6_wrapper_sha256")
        if (
            set(row) != {
                "wrapper_name", "stage6_wrapper_sha256", "identity",
                "strict_recovery_passed", "checkpoint_modified",
            }
            or row["wrapper_name"] not in EXPECTED_WRAPPER_NAMES
            or row["wrapper_name"] != expected_name
            or row["strict_recovery_passed"] is not True
            or row["checkpoint_modified"] is not False
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            _fail("invalid_stage6_postmortem_artifact", "checkpoint evidence differs")
        _validate_checkpoint_identity(
            identity,
            arm=identity["arm"],
            seed=identity["seed"],
            model_config=identity["model_config"],
            parameter_count=identity["parameter_count"],
            partition_identity=identity["training_partition_identity"],
            train_input_evidence=train_evidence,
            producer_source_digest=producer_source_digest,
        )
    if (
        summary.get("diagnostic_version") != DIAGNOSTIC_VERSION
        or summary.get("artifact_version") != ARTIFACT_VERSION
        or summary.get("diagnostic_source_commit") != expected_commit
        or summary.get("slurm_job_id") != job_id
        or summary.get("producer_source_commit") != PRODUCER_SOURCE_COMMIT
        or summary.get("producer_job_id") != PRODUCER_JOB_ID
        or summary.get("producer_execution_evidence")
        != PRODUCER_EXECUTION_EVIDENCE
        or summary.get("training_run_count") != len(training)
        or summary.get("train_family_record_count") != len(families)
        or summary.get("analysis") != analysis
        or summary.get("diagnostic_completed") is not True
        or summary.get("outcome_controls_artifact_validity") is not False
    ):
        _fail("invalid_stage6_postmortem_artifact", "summary differs")
    for record in (resolved.get("access", {}), summary.get("access", {})):
        if record != ACCESS_RECORD:
            _fail("invalid_stage6_postmortem_artifact", "access record differs")
    return {
        "diagnostic_version": DIAGNOSTIC_VERSION,
        "artifact_version": ARTIFACT_VERSION,
        "source_commit": expected_commit,
        "slurm_job_id": job_id,
        "producer_source_commit": PRODUCER_SOURCE_COMMIT,
        "producer_job_id": PRODUCER_JOB_ID,
        "training_run_count": len(training),
        "train_family_record_count": len(families),
        "checkpoint_count": len(checkpoints),
        "regular_file_count": len(ARTIFACT_FILES),
        "sha256sums_sha256": _file_sha256(root / "SHA256SUMS"),
        "reproduces_stage6_train_reliability_failure": analysis[
            "reproduces_stage6_train_reliability_failure"
        ],
        "outcome_controls_artifact_validity": False,
        "verification_status": "pass",
    }


def _sanitized_train_evidence(evidence):
    return {
        name: value for name, value in evidence.items()
        if name not in ("expected_payload_sha256", "observed_payload_sha256")
    }


def run_postmortem(
    *, retained_work_root, train_index, train_root, checkpoint_hashes,
    output_dir, repository_root, expected_commit, slurm_job_id,
):
    _require_runtime()
    job_id = _job_id(slurm_job_id)
    source = _source_identity(repository_root, expected_commit)
    if source["detached_head"] is not True:
        _fail("invalid_stage6_postmortem_source", "detached checkout required")
    retained = Path(retained_work_root)
    expected_retained_name = (
        "ge1-stage6-structure-only-producer-{}-{}.work.incomplete-{}".format(
            PRODUCER_SOURCE_COMMIT, PRODUCER_JOB_ID, PRODUCER_JOB_ID
        )
    )
    if (
        retained.is_symlink() or not retained.is_dir()
        or retained.name != expected_retained_name
    ):
        _fail("invalid_stage6_postmortem_retained_root", str(retained))
    hashes = parse_checkpoint_hashes(checkpoint_hashes)
    train_package = Path(train_index).resolve().parent
    repository = Path(repository_root).resolve()
    retained_resolved = retained.resolve()
    train_payload_root = Path(train_root).resolve()
    if (
        Path(train_index).name != "index.json"
        or train_payload_root != train_package / "payloads"
        or _is_within(retained_resolved, repository)
        or _is_within(retained_resolved, train_package)
        or _is_within(train_package, repository)
        or _is_within(train_package, retained_resolved)
    ):
        _fail("unsafe_stage6_postmortem_input_roots", "roots overlap or differ")
    final, staging = _prepare_output(
        output_dir, repository_root, retained, train_package, job_id
    )
    from .stage6_narrow_loader import load_stage6_train
    train_examples, train_input_evidence = load_stage6_train(train_index, train_root)
    if len(train_examples) != TRAIN_FAMILY_COUNT:
        _fail("invalid_stage6_postmortem_train", "family count differs")
    train_family_ids = tuple(
        sorted(item.physical_family_id for item in train_examples)
    )
    producer_source_digest = source_commit_python_sha256(
        repository_root, PRODUCER_SOURCE_COMMIT
    )
    training = []
    family_records = []
    checkpoint_evidence = []
    validated_models = []
    for seed in FULL_SEEDS:
        for arm in ARMS:
            name = "stage6-{}-seed{}.pt".format(arm, seed)
            model, record, evidence = _load_retained_checkpoint(
                retained / name,
                expected_wrapper_sha256=hashes[name],
                arm=arm,
                seed=seed,
                train_family_ids=train_family_ids,
                train_input_evidence=train_input_evidence,
                producer_source_digest=producer_source_digest,
            )
            training.append(record)
            checkpoint_evidence.append(evidence)
            validated_models.append((model, arm, seed))
    for model, arm, seed in validated_models:
        family_records.extend(generate_autonomous_records(
            model, train_examples, cohort="train", arm=arm, seed=seed,
            execution_device="cpu",
        ))
    analysis = analyze_postmortem(training, family_records)
    resolved = {
        "diagnostic_version": DIAGNOSTIC_VERSION,
        "artifact_version": ARTIFACT_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "producer_source_commit": PRODUCER_SOURCE_COMMIT,
        "producer_source_tree_sha256": producer_source_digest,
        "producer_job_id": PRODUCER_JOB_ID,
        "producer_execution_evidence": dict(PRODUCER_EXECUTION_EVIDENCE),
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
        "train_input_evidence": _sanitized_train_evidence(train_input_evidence),
        "checkpoint_evidence": checkpoint_evidence,
        "implementation_reuse": {
            "autonomous_evaluation": (
                "prototype.graph_encoder.stage6_structure_only_producer."
                "generate_autonomous_records"
            ),
            "structural_scoring": (
                "prototype.graph_encoder.stage6_structure_only.score_family_record"
            ),
            "optimization_reliability": (
                "prototype.graph_encoder.stage6_structure_only_producer."
                "optimization_reliability"
            ),
            "memory_gate": (
                "prototype.graph_encoder.stage6_structure_only."
                "structure_memory_gate_for_cohort"
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
        "producer_execution_evidence": dict(PRODUCER_EXECUTION_EVIDENCE),
        "training_run_count": len(training),
        "train_family_record_count": len(family_records),
        "analysis": analysis,
        "diagnostic_completed": True,
        "outcome_controls_artifact_validity": False,
        "access": dict(ACCESS_RECORD),
    }
    _atomic_write_json(staging / "resolved_config.json", resolved)
    _atomic_write_jsonl(staging / "training_histories.jsonl", training)
    _atomic_write_jsonl(staging / "train_family_records.jsonl", family_records)
    _atomic_write_json(staging / "summary.json", summary)
    _verify_source_unchanged(source, repository_root, expected_commit)
    finalize_artifact(staging, final)
    verified = verify_artifact(
        final, expected_commit=expected_commit, expected_slurm_job_id=job_id
    )
    return {
        "event": "stage6_train_gate_postmortem_completed",
        "artifact_path": str(final),
        "verification": verified,
        "analysis": analysis,
        "diagnostic_completed": True,
        "outcome_controls_artifact_validity": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retained-work-root", required=True)
    parser.add_argument("--train-index", required=True)
    parser.add_argument("--train-root", required=True)
    parser.add_argument("--checkpoint-sha256", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--producer-job-id", required=True)
    parser.add_argument("--slurm-job-id", default=os.environ.get("SLURM_JOB_ID"))
    args = parser.parse_args(argv)
    if _job_id(args.producer_job_id) != PRODUCER_JOB_ID:
        _fail("invalid_stage6_postmortem_producer_job", args.producer_job_id)
    try:
        result = run_postmortem(
            retained_work_root=args.retained_work_root,
            train_index=args.train_index,
            train_root=args.train_root,
            checkpoint_hashes=args.checkpoint_sha256,
            output_dir=args.output_dir,
            repository_root=args.repository_root,
            expected_commit=args.expected_commit,
            slurm_job_id=args.slurm_job_id,
        )
    except Exception as exc:
        failure = {
            "event": "stage6_train_gate_postmortem_infrastructure_failure",
            "failure_type": type(exc).__name__,
            "failure_code": getattr(exc, "code", None),
            "detail": str(exc),
            "diagnostic_completed": False,
            "training_performed": False,
            "backward_pass_performed": False,
            "optimizer_step_performed": False,
            "checkpoint_modified": False,
            "development_accessed": False,
        }
        print(_canonical_json_text(failure), file=sys.stderr, flush=True)
        return 1
    print(_canonical_json_text(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
