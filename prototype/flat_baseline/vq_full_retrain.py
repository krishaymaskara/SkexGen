"""Exact epoch-boundary promotion of the accepted train-kmeans VQ pilot."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile

import torch

from prototype.model_data.loader import load_physical_examples

from .checkpointing import (
    CheckpointError,
    checkpoint_payload,
    load_checkpoint,
    save_checkpoint,
    validated_checkpoint,
)
from .config import FlatBaselineConfig
from .data import load_training_data
from .evaluate_length_conditioned import (
    EvaluationError,
    _atomic_no_replace,
    _fsync_directory,
    _preflight_output,
    _remove_temporary_directory,
    checkpoint_sha256,
)
from .model import FlatMixedVQModel
from .provenance import source_state
from .run_logging import JsonlLogger
from .training import (
    TrainingError,
    evaluate_epoch,
    resolve_device,
    train_epoch,
)
from .training_config import TrainingConfig
from .vq_initialization import initialization_report_sha256
from .vq_full_policy import (
    UsagePolicyError,
    code_usage_trend,
    empty_collapse_state,
    final_training_decision,
    training_epoch_monitor,
    usage_snapshot,
    validation_epoch_monitor,
)
from .vq_pilot import PilotInstrumentation, _pilot_epoch_record, _write_json


FULL_DECISIONS = (
    "PASS_FOR_EVALUATION",
    "FAIL_ASSIGNMENT_COLLAPSE",
    "FAIL_NUMERICAL",
    "FAIL_PROVENANCE",
    "FAIL_OTHER",
)
REQUIRED_TREATMENT = {
    "vq_init": "train-kmeans",
    "seed": 2026,
    "learning_rate": 1e-3,
}


class FullRetrainError(RuntimeError):
    """The promotion cannot continue without violating its fixed contract."""

    def __init__(self, code, detail):
        self.code = code
        self.detail = detail
        super().__init__("{}: {}".format(code, detail))


def authoritative_epoch_budget(baseline_run, torch_module=torch):
    """Reconcile the original checkpoint and run-metadata epoch authority."""

    root = Path(baseline_run)
    checkpoint = torch_module.load(
        str(root / "best.pt"), map_location="cpu"
    )
    try:
        checkpoint_config = TrainingConfig(
            **checkpoint["training_config"]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise FullRetrainError(
            "baseline_provenance",
            "original baseline checkpoint has invalid training provenance",
        ) from exc
    metadata = []
    try:
        with (root / "metrics.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                record = json.loads(line)
                if record.get("event") == "run_metadata":
                    metadata.append(record)
    except (OSError, TypeError, ValueError) as exc:
        raise FullRetrainError(
            "baseline_provenance",
            "original baseline metrics are missing or malformed",
        ) from exc
    if not metadata:
        raise FullRetrainError(
            "baseline_provenance",
            "original baseline has no run_metadata epoch authority",
        )
    try:
        budgets = {
            TrainingConfig(**record["training_config"]).epochs
            for record in metadata
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise FullRetrainError(
            "baseline_provenance",
            "original baseline run_metadata is invalid",
        ) from exc
    if budgets != {checkpoint_config.epochs}:
        raise FullRetrainError(
            "baseline_provenance",
            "checkpoint and run metadata disagree on the full epoch budget",
        )
    return checkpoint_config.epochs


def _promotion_signature(training):
    values = training.to_dict()
    for name in ("epochs", "output_dir", "device", "vq_init"):
        values.pop(name)
    return values


def _read_json(path):
    try:
        with Path(path).open(encoding="utf-8") as stream:
            return json.load(stream)
    except (OSError, TypeError, ValueError) as exc:
        raise FullRetrainError(
            "pilot_provenance", "{} is missing or malformed".format(path)
        ) from exc


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _copy_and_fsync(source, destination):
    shutil.copyfile(str(source), str(destination))
    with Path(destination).open("rb") as stream:
        os.fsync(stream.fileno())


def _validate_pilot(pilot_run, torch_module):
    root = Path(pilot_run)
    decision = _read_json(root / "pilot_decision.json")
    configuration = _read_json(root / "run_configuration.json")
    initialization = _read_json(root / "initialization_report.json")
    partitions = _read_json(root / "partition_provenance.json")
    if (
        decision.get("decision") != "PASS_FOR_FULL_RETRAIN"
        or decision.get("completed_epoch") != 5
        or decision.get("test_partition_evaluated") is not False
    ):
        raise FullRetrainError(
            "pilot_provenance", "the required successful pilot is absent"
        )
    training = TrainingConfig(**configuration["training_config"])
    if any(
        getattr(training, name) != value
        for name, value in REQUIRED_TREATMENT.items()
    ):
        raise FullRetrainError(
            "pilot_provenance", "pilot treatment differs from the accepted one"
        )
    if configuration["model_config"]["codebook_size"] != 32:
        raise FullRetrainError(
            "pilot_provenance", "pilot codebook size is not 32"
        )
    if (
        partitions.get("test_partition_evaluated") is not False
        or partitions.get("test_family_count_used_for_initialization") != 0
    ):
        raise FullRetrainError(
            "pilot_provenance", "pilot used the test partition"
        )
    if initialization_report_sha256(initialization) != initialization.get(
        "report_sha256"
    ):
        raise FullRetrainError(
            "pilot_provenance", "pilot initialization hash is inconsistent"
        )
    provenance = configuration.get("initialization_provenance")
    if not isinstance(provenance, dict) or (
        provenance.get("mode") != "train-kmeans"
        or provenance.get("seed") != 2026
        or provenance.get("algorithm") != initialization.get("method")
        or provenance.get("pseudo_count_policy")
        != initialization.get("pseudo_count_policy")
        or provenance.get("report_sha256")
        != initialization.get("report_sha256")
    ):
        raise FullRetrainError(
            "pilot_provenance",
            "pilot initialization provenance does not match its report",
        )
    checkpoint = torch_module.load(
        str(root / "last.pt"), map_location="cpu"
    )
    data_state = checkpoint.get("data_state")
    if (
        checkpoint.get("epoch") != 5
        or checkpoint.get("checkpoint_kind") != "last"
        or not isinstance(data_state, dict)
        or data_state.get("initialization_mode") != "train-kmeans"
        or data_state.get("initialization_report_sha256")
        != initialization["report_sha256"]
    ):
        raise FullRetrainError(
            "pilot_provenance", "pilot last checkpoint is not resumable"
        )
    return configuration, partitions, checkpoint


def _decision(value, reason, completed_epoch):
    if value not in FULL_DECISIONS:
        raise FullRetrainError("invalid_decision", value)
    return {
        "decision": value,
        "reason": reason,
        "completed_epoch": completed_epoch,
        "test_partition_evaluated": False,
    }


def _manifest(root):
    hashes = {
        path.name: _sha256(path)
        for path in sorted(Path(root).iterdir(), key=lambda item: item.name)
        if path.name != "artifact_manifest.json"
    }
    _write_json(Path(root) / "artifact_manifest.json", {
        "artifact_names": sorted(hashes),
        "artifact_sha256": hashes,
    })


def _read_epoch_records(path):
    records = []
    try:
        with Path(path).open(encoding="utf-8") as stream:
            for line in stream:
                record = json.loads(line)
                if record.get("mode") in ("train", "validation"):
                    records.append(record)
    except (OSError, TypeError, ValueError) as exc:
        raise FullRetrainError(
            "pilot_provenance", "pilot epoch metrics are malformed"
        ) from exc
    return records


def _selected_best_state(
    best_path,
    epoch_records,
    model_config,
    training_config,
    data_state,
    torch_module,
):
    try:
        checkpoint = validated_checkpoint(
            best_path,
            model_config,
            training_config,
            torch_module,
            expected_data_state=data_state,
        )
    except (CheckpointError, OSError, TypeError, ValueError):
        return False, False, None, None
    matches = [
        record for record in epoch_records
        if record["mode"] == "validation"
        and record["epoch"] == checkpoint["epoch"]
    ]
    if checkpoint["checkpoint_kind"] != "best" or len(matches) != 1:
        return False, False, None, checkpoint
    record = matches[0]
    metric_name = training_config.checkpoint_selection_metric
    metric = record.get("metrics", {}).get(metric_name)
    checkpoint_metric = checkpoint.get("best_validation_metric")
    finite = (
        isinstance(metric, (int, float))
        and not isinstance(metric, bool)
        and math.isfinite(float(metric))
        and isinstance(checkpoint_metric, (int, float))
        and not isinstance(checkpoint_metric, bool)
        and math.isfinite(float(checkpoint_metric))
        and float(metric) == float(checkpoint_metric)
        and all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            for value in record.get("metrics", {}).values()
        )
    )
    finite = finite and _checkpoint_values_finite(
        checkpoint, torch_module
    )
    try:
        usage_snapshot(record)
    except (KeyError, TypeError, UsagePolicyError):
        finite = False
    return True, finite, record, checkpoint


def _checkpoint_values_finite(checkpoint, torch_module):
    pending = [
        checkpoint.get("model_state"),
        checkpoint.get("optimizer_state"),
    ]
    while pending:
        value = pending.pop()
        if isinstance(value, dict):
            pending.extend(value.values())
        elif isinstance(value, (list, tuple)):
            pending.extend(value)
        elif torch_module.is_tensor(value) and (
            torch_module.is_floating_point(value)
            or torch_module.is_complex(value)
        ):
            if not bool(torch_module.isfinite(value).all().item()):
                return False
    return True


def run_full_retraining(
    corpus_dir,
    baseline_run,
    pilot_run,
    output_dir,
    reviewed_commit,
    expected_counts=None,
    device_override="cpu",
    torch_module=torch,
):
    """Resume the successful pilot through the authoritative full budget."""

    destination = Path(output_dir)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        _preflight_output(destination)
    except EvaluationError as exc:
        raise FullRetrainError(exc.code, exc.detail) from exc
    pilot_root = Path(pilot_run).resolve()
    if destination.resolve() == pilot_root:
        raise FullRetrainError(
            "output_collision", "full output cannot overwrite the pilot"
        )
    source = source_state()
    if source["git_commit"] != reviewed_commit or source["git_dirty"] is not False:
        raise FullRetrainError(
            "source_provenance",
            "reviewed commit and clean source tree are required",
        )
    budget = authoritative_epoch_budget(baseline_run, torch_module)
    configuration, partitions, pilot_checkpoint = (
        _validate_pilot(pilot_root, torch_module)
    )
    baseline_checkpoint = torch_module.load(
        str(Path(baseline_run) / "best.pt"), map_location="cpu"
    )
    try:
        baseline_model = FlatBaselineConfig(
            **baseline_checkpoint["model_config"]
        )
        baseline_training = TrainingConfig(
            **baseline_checkpoint["training_config"]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise FullRetrainError(
            "baseline_provenance",
            "original baseline configurations are invalid",
        ) from exc
    pilot_training = TrainingConfig(**configuration["training_config"])
    if (
        baseline_model.to_dict() != configuration["model_config"]
        or _promotion_signature(baseline_training)
        != _promotion_signature(pilot_training)
    ):
        raise FullRetrainError(
            "baseline_provenance",
            "pilot changed settings beyond the accepted initialization treatment",
        )
    if budget <= pilot_checkpoint["epoch"]:
        raise FullRetrainError(
            "baseline_provenance",
            "full epoch budget must extend beyond the completed pilot",
        )
    physical = tuple(load_physical_examples(corpus_dir, "iid"))
    counts = {
        name: sum(item.partition == name for item in physical)
        for name in ("train", "validation", "test")
    }
    if expected_counts is not None and counts != expected_counts:
        raise FullRetrainError(
            "partition_mismatch", "authoritative partition counts differ"
        )
    data = load_training_data(corpus_dir, "iid", "train", "validation")
    model_config = FlatBaselineConfig(**configuration["model_config"])
    training = replace(
        pilot_training,
        epochs=budget,
        output_dir=str(destination),
        device=device_override,
    )
    training.validate()
    device = resolve_device(training.device, torch_module)
    model = FlatMixedVQModel(model_config).to(device)
    optimizer = torch_module.optim.AdamW(
        model.parameters(),
        lr=training.learning_rate,
        weight_decay=training.weight_decay,
    )
    restored = load_checkpoint(
        pilot_root / "last.pt",
        model,
        optimizer,
        model_config,
        training,
        torch_module,
        device,
        restore_rng=True,
        expected_data_state=pilot_checkpoint["data_state"],
    )
    temporary = Path(tempfile.mkdtemp(
        prefix="." + destination.name + ".tmp-",
        dir=str(destination.parent),
    ))
    completed = restored["epoch"]
    best_metric = restored["best_validation_metric"]
    global_step = restored["global_step"]
    epoch_records = []
    try:
        _copy_and_fsync(
            pilot_root / "initialization_report.json",
            temporary / "initialization_report.json",
        )
        _copy_and_fsync(
            pilot_root / "epoch_metrics.jsonl",
            temporary / "epoch_metrics.jsonl",
        )
        _copy_and_fsync(
            pilot_root / "best.pt", temporary / "best.pt"
        )
        _copy_and_fsync(
            pilot_root / "last.pt", temporary / "last.pt"
        )
        full_configuration = {
            "experiment": "flat-mixed-vq-train-kmeans-full",
            "resume_strategy": "exact_epoch_boundary_resume",
            "resume_checkpoint": str(pilot_root / "last.pt"),
            "resume_checkpoint_sha256": checkpoint_sha256(
                pilot_root / "last.pt"
            ),
            "pilot_run": str(pilot_root),
            "pilot_decision": "PASS_FOR_FULL_RETRAIN",
            "authoritative_epoch_budget": budget,
            "epoch_budget_authority": {
                "checkpoint": str(Path(baseline_run) / "best.pt"),
                "run_metadata": str(Path(baseline_run) / "metrics.jsonl"),
            },
            "model_config": model_config.to_dict(),
            "training_config": training.to_dict(),
            "scheduler_present": False,
            "resume_state_contract": (
                "model_and_vq_ema",
                "optimizer",
                "python_and_torch_rng",
                "completed_epoch_and_global_step",
                "treatment_and_initialization_provenance",
            ),
            "data_order_contract": "epoch_seed_equals_training_seed_plus_epoch",
            "reviewed_commit": reviewed_commit,
            "source": source,
            "initialization_provenance": configuration[
                "initialization_provenance"
            ],
            "test_partition_evaluated": False,
        }
        _write_json(temporary / "run_configuration.json", full_configuration)
        _write_json(temporary / "partition_provenance.json", partitions)
        epoch_logger = JsonlLogger(temporary / "epoch_metrics.jsonl")
        epoch_records = _read_epoch_records(
            temporary / "epoch_metrics.jsonl"
        )
        pilot_five = [
            record for record in epoch_records
            if record["mode"] == "train" and record["epoch"] == 5
        ]
        if len(pilot_five) != 1:
            raise FullRetrainError(
                "pilot_provenance",
                "one pilot epoch-5 training record is required",
            )
        collapse_state = empty_collapse_state()
        collapse_state, unused_monitor = training_epoch_monitor(
            collapse_state, pilot_five[0]
        )
        del unused_monitor
        stopped = None
        for epoch in range(restored["epoch"] + 1, budget + 1):
            train_summary, global_step = train_epoch(
                model,
                optimizer,
                data.train_examples,
                model_config,
                training,
                device,
                epoch,
                global_step,
                torch_module,
                PilotInstrumentation(None, torch_module),
            )
            train_record = _pilot_epoch_record(
                "train", epoch, global_step, train_summary, optimizer
            )
            collapse_state, train_record["collapse_monitor"] = (
                training_epoch_monitor(collapse_state, train_record)
            )
            epoch_logger.write(train_record)
            epoch_records.append(train_record)
            completed = epoch
            if train_record["collapse_monitor"]["stop"]:
                stopped = (
                    "train_epoch_{}_two_consecutive_usage_violations"
                ).format(epoch)
            validation_summary = None
            if stopped is None:
                validation_summary = evaluate_epoch(
                    model,
                    data.validation_examples,
                    model_config,
                    training,
                    device,
                    torch_module,
                    PilotInstrumentation(None, torch_module),
                    epoch,
                )
                validation_record = _pilot_epoch_record(
                    "validation",
                    epoch,
                    global_step,
                    validation_summary,
                    optimizer,
                )
                validation_record["collapse_monitor"] = (
                    validation_epoch_monitor(validation_record)
                )
                epoch_logger.write(validation_record)
                epoch_records.append(validation_record)
                candidate = validation_summary["metrics"][
                    training.checkpoint_selection_metric
                ]
                if best_metric is None or candidate < best_metric:
                    best_metric = candidate
                    save_checkpoint(
                        temporary / "best.pt",
                        checkpoint_payload(
                            model, optimizer, epoch, global_step,
                            model_config, training, best_metric, torch_module,
                            restored["data_state"], "best",
                        ),
                        torch_module,
                    )
            save_checkpoint(
                temporary / "last.pt",
                checkpoint_payload(
                    model, optimizer, epoch, global_step,
                    model_config, training, best_metric, torch_module,
                    restored["data_state"], "last",
                ),
                torch_module,
            )
            if stopped is not None:
                break
        trend = code_usage_trend(epoch_records)
        (
            best_provenance_valid,
            best_values_finite,
            best_validation_record,
            best_checkpoint,
        ) = _selected_best_state(
            temporary / "best.pt",
            epoch_records,
            model_config,
            training,
            restored["data_state"],
            torch_module,
        )
        decision = final_training_decision(
            completed_epoch=completed,
            authoritative_budget=budget,
            stopped_reason=stopped,
            best_checkpoint_provenance_valid=best_provenance_valid,
            best_required_values_finite=best_values_finite,
            best_validation_record=best_validation_record,
            test_partition_evaluated=False,
            trend=trend,
        )
        decision["selected_best_checkpoint"] = (
            None
            if best_checkpoint is None
            else {
                "epoch": best_checkpoint["epoch"],
                "global_step": best_checkpoint["global_step"],
                "sha256": checkpoint_sha256(temporary / "best.pt"),
                "provenance_valid": best_provenance_valid,
                "required_values_finite": best_values_finite,
                "validation_usage": (
                    None
                    if best_validation_record is None
                    else usage_snapshot(best_validation_record)
                ),
            }
        )
        _write_json(temporary / "training_decision.json", decision)
        _manifest(temporary)
        _fsync_directory(temporary)
        _atomic_no_replace(temporary, destination)
        temporary = None
        _fsync_directory(destination.parent)
        return decision
    except (
        FullRetrainError,
        TrainingError,
        UsagePolicyError,
        OSError,
        TypeError,
        ValueError,
    ) as exc:
        decision = _decision(
            "FAIL_NUMERICAL"
            if (
                isinstance(exc, UsagePolicyError)
                or getattr(exc, "code", "").startswith("nonfinite")
            )
            else "FAIL_OTHER",
            getattr(exc, "code", type(exc).__name__),
            completed,
        )
        try:
            decision["code_usage_trend"] = code_usage_trend(epoch_records)
        except (KeyError, TypeError, UsagePolicyError):
            decision["code_usage_trend"] = None
        _write_json(temporary / "training_decision.json", decision)
        _manifest(temporary)
        _fsync_directory(temporary)
        _atomic_no_replace(temporary, destination)
        temporary = None
        return decision
    finally:
        if temporary is not None and temporary.exists():
            _remove_temporary_directory(temporary)


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", required=True)
    parser.add_argument("--baseline-run", required=True)
    parser.add_argument("--pilot-run", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reviewed-commit", required=True)
    parser.add_argument("--vq-init", choices=("train-kmeans",), required=True)
    parser.add_argument("--device", choices=("cpu",), required=True)
    parser.add_argument("--expected-train-count", type=int, required=True)
    parser.add_argument("--expected-validation-count", type=int, required=True)
    parser.add_argument("--expected-test-count", type=int, required=True)
    return parser


def main(argv=None):
    arguments = _parser().parse_args(argv)
    try:
        result = run_full_retraining(
            arguments.corpus_dir,
            arguments.baseline_run,
            arguments.pilot_run,
            arguments.output_dir,
            arguments.reviewed_commit,
            {
                "train": arguments.expected_train_count,
                "validation": arguments.expected_validation_count,
                "test": arguments.expected_test_count,
            },
            arguments.device,
        )
    except (FullRetrainError, OSError, TypeError, ValueError) as exc:
        sys.stderr.write("full_retrain_failure: {}\n".format(exc))
        return 2
    sys.stdout.write(json.dumps(
        result, sort_keys=True, separators=(",", ":"), allow_nan=False
    ) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
