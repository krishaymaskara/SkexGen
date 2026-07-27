"""Five-epoch train-kmeans treatment pilot for the flat mixed VQ baseline."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile

import torch

from prototype.model_data.loader import load_physical_examples

from .checkpointing import checkpoint_payload, save_checkpoint
from .config import FlatBaselineConfig
from .data import load_training_data
from .evaluate_length_conditioned import (
    EvaluationError,
    _atomic_no_replace,
    _fsync_directory,
    _json_document,
    _preflight_output,
    _remove_temporary_directory,
    checkpoint_sha256,
)
from .model import FlatMixedVQModel
from .provenance import source_state
from .run_logging import JsonlLogger
from .training import (
    TrainingError,
    _data_state,
    _epoch_record,
    evaluate_epoch,
    resolve_device,
    seed_everything,
    train_epoch,
)
from .training_config import LOSS_METRICS, TrainingConfig
from .vq_initialization import (
    DISTINCT_TOLERANCE,
    PSEUDO_COUNT_POLICY,
    VQInitializationError,
    _distinct_count,
    _squared_distances,
    initialization_report_sha256,
    initialize_train_codebook,
)


PILOT_EPOCHS = 5
EARLY_LOG_STEPS = frozenset((0, 1, 2, 4, 8, 16))
PILOT_DECISIONS = (
    "PASS_FOR_FULL_RETRAIN",
    "FAIL_ASSIGNMENT_COLLAPSE",
    "FAIL_NUMERICAL",
    "FAIL_PROVENANCE",
    "FAIL_OTHER",
)


class PilotError(RuntimeError):
    """The bounded treatment cannot continue under its scientific contract."""

    def __init__(self, code, detail):
        self.code = code
        self.detail = detail
        super().__init__("{}: {}".format(code, detail))


class PilotInstrumentation:
    """Read-only prequant and code-usage capture around existing forwards."""

    def __init__(self, step_logger=None, torch_module=torch):
        self.step_logger = step_logger
        self.torch = torch_module
        self.handle = None
        self.mode = None
        self.epoch = None
        self.pending = []
        self.vectors = []
        self.counts = []

    def begin(self, model, mode, epoch):
        if self.handle is not None:
            raise PilotError(
                "instrumentation_state", "instrumentation is already active"
            )
        self.mode = mode
        self.epoch = epoch
        self.pending = []
        self.vectors = []
        self.counts = []

        def capture(module, inputs, output):
            del module, inputs
            self.pending.append(output.detach().to(
                device="cpu", dtype=self.torch.float64
            ))

        self.handle = model.to_codebook.register_forward_hook(capture)

    def observe_batch(self, model, output, losses, global_step):
        if len(self.pending) != 1:
            raise PilotError(
                "prequant_capture",
                "one prequant tensor is required for every model forward",
            )
        vectors = self.pending.pop().reshape(-1, model.vq.embedding_dim)
        counts = output.assignment_counts.detach().to(
            device="cpu", dtype=self.torch.long
        )
        self.vectors.append(vectors)
        self.counts.append(counts)
        if (
            self.step_logger is not None
            and self.mode == "train"
            and self.epoch == 1
            and global_step in EARLY_LOG_STEPS
        ):
            record = {
                "event": "early_code_usage",
                "epoch": self.epoch,
                "global_step": global_step,
                "losses": _loss_values(losses),
            }
            record.update(_diagnostics(
                vectors, counts, model.vq, self.torch
            ))
            self.step_logger.write(record)

    def finish(self, model):
        if self.handle is None:
            raise PilotError(
                "instrumentation_state", "instrumentation is not active"
            )
        self.handle.remove()
        self.handle = None
        if self.pending:
            raise PilotError(
                "prequant_capture", "captured prequant tensors were not consumed"
            )
        if not self.vectors or not self.counts:
            raise PilotError(
                "empty_instrumentation", "epoch produced no instrumented batches"
            )
        vectors = self.torch.cat(self.vectors, dim=0)
        counts = self.torch.stack(self.counts, dim=0).sum(dim=0)
        return _diagnostics(vectors, counts, model.vq, self.torch)

    def abort(self):
        """Remove the observation hook without masking the training failure."""

        if self.handle is not None:
            self.handle.remove()
            self.handle = None
        self.pending = []


def pilot_decision(epoch_records):
    """Apply the fixed epoch-2 stop and epoch-5 qualification rules."""

    indexed = {
        (record["mode"], record["epoch"]): record
        for record in epoch_records
    }
    if any(
        record["diagnostics"]["finite"] is not True
        or record["diagnostics"]["ema_state_consistent"] is not True
        for record in epoch_records
    ):
        return _decision("FAIL_NUMERICAL", "nonfinite_or_inconsistent_state")
    train_two = indexed.get(("train", 2))
    if train_two is None:
        return _decision("FAIL_OTHER", "missing_train_epoch_2")
    if (
        train_two["diagnostics"]["active_code_count"] < 2
        or train_two["diagnostics"]["codebook_perplexity"] < 2.0
    ):
        return _decision(
            "FAIL_ASSIGNMENT_COLLAPSE", "epoch_2_assignment_gate"
        )
    train_five = indexed.get(("train", 5))
    validation_five = indexed.get(("validation", 5))
    if train_five is None or validation_five is None:
        return _decision("FAIL_OTHER", "pilot_trajectory_is_incomplete")
    qualifying = (
        train_five["diagnostics"]["active_code_count"] >= 2
        and train_five["diagnostics"]["codebook_perplexity"] >= 2.0
        and validation_five["diagnostics"]["active_code_count"] >= 2
        and validation_five["diagnostics"]["codebook_perplexity"] >= 2.0
        and train_two["diagnostics"]["distinct_prequant_vector_count"] >= 2
        and train_five["diagnostics"]["distinct_prequant_vector_count"] >= 2
        and validation_five["diagnostics"][
            "distinct_prequant_vector_count"
        ] >= 2
    )
    if qualifying:
        return _decision("PASS_FOR_FULL_RETRAIN", "all_pilot_gates_passed")
    return _decision(
        "FAIL_ASSIGNMENT_COLLAPSE", "epoch_5_assignment_gate"
    )


def run_pilot(
    corpus_dir,
    baseline_checkpoint,
    output_dir,
    reviewed_commit,
    expected_counts=None,
    torch_module=torch,
):
    """Run and atomically publish the bounded treatment pilot."""

    destination = Path(output_dir)
    destination.parent.mkdir(parents=True, exist_ok=True)
    completed_epoch = 0
    try:
        _preflight_output(destination)
    except EvaluationError as exc:
        raise PilotError(exc.code, exc.detail) from exc
    source = source_state()
    if source["git_commit"] != reviewed_commit or source["git_dirty"] is not False:
        raise PilotError(
            "source_provenance",
            "reviewed commit and clean source tree are required",
        )
    physical = tuple(load_physical_examples(corpus_dir, "iid"))
    partition_ids = {
        name: tuple(sorted(
            item.physical_family_id
            for item in physical
            if item.partition == name
        ))
        for name in ("train", "validation", "test")
    }
    counts = {name: len(values) for name, values in partition_ids.items()}
    if expected_counts is not None and counts != expected_counts:
        raise PilotError(
            "partition_mismatch",
            "authoritative partition counts differ from the pilot contract",
        )
    data = load_training_data(corpus_dir, "iid", "train", "validation")
    checkpoint = torch_module.load(
        str(baseline_checkpoint), map_location="cpu"
    )
    try:
        model_config = FlatBaselineConfig(**checkpoint["model_config"])
        baseline_training = TrainingConfig(**checkpoint["training_config"])
    except (KeyError, TypeError, ValueError) as exc:
        raise PilotError(
            "baseline_checkpoint_provenance",
            "baseline checkpoint configurations are invalid",
        ) from exc
    expected_data = checkpoint.get("data_state")
    if not isinstance(expected_data, dict) or (
        tuple(expected_data.get("train_family_ids", ()))
        != data.train_family_ids
        or tuple(expected_data.get("validation_family_ids", ()))
        != data.validation_family_ids
    ):
        raise PilotError(
            "baseline_checkpoint_provenance",
            "baseline checkpoint partitions differ from authoritative data",
        )
    temporary = Path(tempfile.mkdtemp(
        prefix="." + destination.name + ".tmp-",
        dir=str(destination.parent),
    ))
    try:
        training_config = replace(
            baseline_training,
            epochs=PILOT_EPOCHS,
            output_dir=str(destination),
            vq_init="train-kmeans",
        )
        training_config.validate()
        seed_everything(training_config.seed, torch_module)
        device = resolve_device(training_config.device, torch_module)
        model = FlatMixedVQModel(model_config).to(device)
        initialization = initialize_train_codebook(
            model, data, training_config, device, torch_module
        )
        if initialization_report_sha256(initialization) != initialization[
            "report_sha256"
        ]:
            raise PilotError(
                "initialization_report_hash",
                "initialization report self-hash is inconsistent",
            )
        optimizer = torch_module.optim.AdamW(
            model.parameters(),
            lr=training_config.learning_rate,
            weight_decay=training_config.weight_decay,
        )
        data_state = _data_state(
            data, "iid", "train", "validation"
        )
        data_state.update({
            "experiment": "flat-mixed-vq-train-kmeans-pilot",
            "initialization_mode": "train-kmeans",
            "initialization_algorithm": initialization["method"],
            "initialization_seed": training_config.seed,
            "initialization_pseudo_count_policy": PSEUDO_COUNT_POLICY,
            "initialization_report_sha256": initialization["report_sha256"],
            "reviewed_commit": reviewed_commit,
            "source_tree_sha256": source["source_tree_sha256"],
        })
        configuration = {
            "experiment": "flat-mixed-vq-train-kmeans-pilot",
            "pilot_epochs": PILOT_EPOCHS,
            "model_config": model_config.to_dict(),
            "training_config": training_config.to_dict(),
            "baseline_checkpoint": str(Path(baseline_checkpoint)),
            "baseline_checkpoint_sha256": checkpoint_sha256(
                baseline_checkpoint
            ),
            "reviewed_commit": reviewed_commit,
            "source": source,
            "initialization_provenance": {
                "mode": "train-kmeans",
                "algorithm": initialization["method"],
                "seed": training_config.seed,
                "pseudo_count_policy": PSEUDO_COUNT_POLICY,
                "report_sha256": initialization["report_sha256"],
            },
            "test_partition_evaluated": False,
        }
        partitions = {
            "partition_counts": counts,
            "partition_ids": {
                name: list(values) for name, values in partition_ids.items()
            },
            "initialization_partition": "train",
            "validation_family_count_used_for_initialization": 0,
            "test_family_count_used_for_initialization": 0,
            "test_partition_evaluated": False,
        }
        _write_json(temporary / "run_configuration.json", configuration)
        _write_json(temporary / "partition_provenance.json", partitions)
        _write_json(temporary / "initialization_report.json", initialization)
        step_logger = JsonlLogger(temporary / "code_usage_steps.jsonl")
        epoch_logger = JsonlLogger(temporary / "epoch_metrics.jsonl")
        last_path = temporary / "last.pt"
        best_path = temporary / "best.pt"
        epoch_records = []
        global_step = 0
        best_metric = None
        for epoch in range(1, PILOT_EPOCHS + 1):
            train_instrumentation = PilotInstrumentation(
                step_logger, torch_module
            )
            train_summary, global_step = train_epoch(
                model,
                optimizer,
                data.train_examples,
                model_config,
                training_config,
                device,
                epoch,
                global_step,
                torch_module,
                train_instrumentation,
            )
            train_record = _pilot_epoch_record(
                "train", epoch, global_step, train_summary, optimizer
            )
            epoch_records.append(train_record)
            epoch_logger.write(train_record)
            validation_instrumentation = PilotInstrumentation(
                None, torch_module
            )
            validation_summary = evaluate_epoch(
                model,
                data.validation_examples,
                model_config,
                training_config,
                device,
                torch_module,
                validation_instrumentation,
                epoch,
            )
            validation_record = _pilot_epoch_record(
                "validation",
                epoch,
                global_step,
                validation_summary,
                optimizer,
            )
            epoch_records.append(validation_record)
            epoch_logger.write(validation_record)
            completed_epoch = epoch
            candidate = validation_summary["metrics"][
                training_config.checkpoint_selection_metric
            ]
            if best_metric is None or candidate < best_metric:
                best_metric = candidate
                save_checkpoint(
                    best_path,
                    checkpoint_payload(
                        model,
                        optimizer,
                        epoch,
                        global_step,
                        model_config,
                        training_config,
                        best_metric,
                        torch_module,
                        data_state,
                        "best",
                    ),
                    torch_module,
                )
            save_checkpoint(
                last_path,
                checkpoint_payload(
                    model,
                    optimizer,
                    epoch,
                    global_step,
                    model_config,
                    training_config,
                    best_metric,
                    torch_module,
                    data_state,
                    "last",
                ),
                torch_module,
            )
            if epoch == 2:
                early = pilot_decision(epoch_records)
                if early["decision"] == "FAIL_ASSIGNMENT_COLLAPSE":
                    break
        decision = pilot_decision(epoch_records)
        decision.update({
            "completed_epoch": max(
                record["epoch"] for record in epoch_records
            ),
            "test_partition_evaluated": False,
        })
        _write_json(temporary / "pilot_decision.json", decision)
        _write_manifest(temporary)
        _fsync_directory(temporary)
        _atomic_no_replace(temporary, destination)
        temporary = None
        _fsync_directory(destination.parent)
        return decision
    except (TrainingError, VQInitializationError) as exc:
        failure = _decision_for_exception(exc)
        failure.update({
            "completed_epoch": completed_epoch,
            "failure_code": exc.code,
            "detail": exc.detail,
            "test_partition_evaluated": False,
        })
        _write_json(temporary / "pilot_decision.json", failure)
        _write_manifest(temporary)
        _fsync_directory(temporary)
        _atomic_no_replace(temporary, destination)
        temporary = None
        _fsync_directory(destination.parent)
        return failure
    except EvaluationError as exc:
        raise PilotError(exc.code, exc.detail) from exc
    finally:
        if temporary is not None and temporary.exists():
            _remove_temporary_directory(temporary)


def _diagnostics(vectors, counts, quantizer, torch_module):
    vectors = vectors.to(device="cpu", dtype=torch_module.float64)
    counts = counts.to(device="cpu", dtype=torch_module.float64)
    embedding = quantizer.embedding.detach().to(
        device="cpu", dtype=torch_module.float64
    )
    ema_counts = quantizer.ema_cluster_size.detach().to(
        device="cpu", dtype=torch_module.float64
    )
    ema_weight = quantizer.ema_weight.detach().to(
        device="cpu", dtype=torch_module.float64
    )
    total = float(counts.sum().item())
    probabilities = counts / total if total > 0 else counts
    nonzero = probabilities > 0
    entropy = float(-(
        probabilities[nonzero] * torch_module.log(probabilities[nonzero])
    ).sum().item()) if bool(nonzero.any().item()) else 0.0
    perplexity = math.exp(entropy)
    variances = vectors.var(dim=0, unbiased=False)
    distances = _squared_distances(vectors, embedding)
    if embedding.size(0) >= 2:
        nearest = distances.topk(2, dim=1, largest=False).values
        margins = nearest[:, 1] - nearest[:, 0]
        mean_margin = float(margins.mean().item())
        minimum_margin = float(margins.min().item())
    else:
        mean_margin = None
        minimum_margin = None
    finite_tensors = (
        vectors,
        counts,
        embedding,
        ema_counts,
        ema_weight,
        variances,
        distances,
    )
    finite = all(
        bool(torch_module.isfinite(value).all().item())
        for value in finite_tensors
    ) and all(
        value is None or math.isfinite(value)
        for value in (
            entropy,
            perplexity,
            mean_margin,
            minimum_margin,
        )
    )
    ema_total = ema_counts.sum()
    smoothed = (
        (ema_counts + quantizer.epsilon)
        / (
            ema_total
            + ema_counts.numel() * quantizer.epsilon
        )
        * ema_total
    )
    implied_embedding = (
        ema_weight / smoothed.unsqueeze(1).clamp_min(quantizer.epsilon)
    )
    ema_consistent = (
        finite
        and ema_counts.shape == counts.shape
        and ema_weight.shape == embedding.shape
        and bool((ema_counts >= 0).all().item())
        and bool(torch_module.allclose(
            implied_embedding,
            embedding,
            rtol=1e-5,
            atol=1e-6,
        ))
    )
    active = int((counts > 0).sum().item())
    return {
        "active_code_count": active,
        "codebook_utilization": active / float(counts.numel()),
        "assignment_entropy": entropy,
        "codebook_perplexity": perplexity,
        "assignment_histogram": [
            int(value) for value in counts.tolist()
        ],
        "prequant_variance": {
            "minimum": _finite_float_or_none(variances.min().item()),
            "mean": _finite_float_or_none(variances.mean().item()),
            "maximum": _finite_float_or_none(variances.max().item()),
        },
        "distinct_prequant_vector_count": _distinct_count(
            vectors, DISTINCT_TOLERANCE
        ),
        "distinct_tolerance": DISTINCT_TOLERANCE,
        "mean_nearest_code_margin": _finite_float_or_none(mean_margin),
        "minimum_nearest_code_margin": _finite_float_or_none(minimum_margin),
        "ema_cluster_counts": [
            float(value) for value in ema_counts.tolist()
        ],
        "dead_ema_code_count": int(
            (ema_counts <= quantizer.epsilon).sum().item()
        ),
        "unassigned_code_count": int((counts == 0).sum().item()),
        "finite": finite,
        "ema_state_consistent": ema_consistent,
    }


def _loss_values(losses):
    return {
        name: float(losses.per_example[name].detach().mean().item())
        for name in LOSS_METRICS
    }


def _finite_float_or_none(value):
    if value is None:
        return None
    converted = float(value)
    return converted if math.isfinite(converted) else None


def _pilot_epoch_record(mode, epoch, global_step, summary, optimizer):
    record = _epoch_record(
        mode, epoch, global_step, summary, optimizer
    )
    record["mode"] = mode
    return record


def _decision(value, reason):
    if value not in PILOT_DECISIONS:
        raise PilotError("invalid_decision", value)
    return {"decision": value, "reason": reason}


def _decision_for_exception(exc):
    code = getattr(exc, "code", "")
    if code.startswith("nonfinite") or code in (
        "inconsistent_ema_state",
        "initialization_report_hash",
    ):
        return _decision("FAIL_NUMERICAL", code)
    if code in (
        "source_provenance",
        "partition_mismatch",
        "baseline_checkpoint_provenance",
        "initialization_partition_overlap",
        "invalid_initialization_authority",
    ):
        return _decision("FAIL_PROVENANCE", code)
    return _decision("FAIL_OTHER", code or type(exc).__name__)


def _write_json(path, value):
    content = _json_document(value)
    with Path(path).open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def _write_manifest(root):
    hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.iterdir(), key=lambda item: item.name)
        if path.name != "artifact_manifest.json"
    }
    _write_json(root / "artifact_manifest.json", {
        "artifact_sha256": hashes,
        "artifact_names": sorted(hashes),
    })


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", required=True)
    parser.add_argument("--baseline-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reviewed-commit", required=True)
    parser.add_argument(
        "--vq-init", choices=("train-kmeans",), required=True
    )
    parser.add_argument("--expected-train-count", type=int)
    parser.add_argument("--expected-validation-count", type=int)
    parser.add_argument("--expected-test-count", type=int)
    return parser


def main(argv=None):
    arguments = _parser().parse_args(argv)
    counts = None
    supplied = (
        arguments.expected_train_count,
        arguments.expected_validation_count,
        arguments.expected_test_count,
    )
    if any(value is not None for value in supplied):
        if any(value is None for value in supplied):
            sys.stderr.write(
                "vq_pilot_failure: incomplete expected partition counts\n"
            )
            return 2
        counts = dict(zip(("train", "validation", "test"), supplied))
    try:
        result = run_pilot(
            arguments.corpus_dir,
            arguments.baseline_checkpoint,
            arguments.output_dir,
            arguments.reviewed_commit,
            counts,
        )
    except (PilotError, OSError, TypeError, ValueError) as exc:
        decision = _decision_for_exception(exc)
        sys.stderr.write(json.dumps(
            decision, sort_keys=True, separators=(",", ":")
        ) + "\n")
        sys.stderr.write("vq_pilot_failure: {}\n".format(exc))
        return 2
    sys.stdout.write(json.dumps(
        result, sort_keys=True, separators=(",", ":"), allow_nan=False
    ) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
