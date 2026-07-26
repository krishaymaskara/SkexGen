"""Read-only validation and summarization of quiescent B0 training runs."""

from __future__ import annotations

import csv
from dataclasses import fields
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import tempfile

from .checkpointing import CheckpointError, _validate_payload
from .config import FlatBaselineConfig
from .training_config import LOSS_METRICS, TrainingConfig


ANALYSIS_SCHEMA_VERSION = 1
_REL_TOL = 1e-7
_ABS_TOL = 1e-9
_LOSS_REL_TOL = 1e-6
_KNOWN_EVENTS = {
    "run_metadata",
    "train_epoch",
    "validation_epoch",
    "checkpoint",
    "failure",
}
_METADATA_FIELDS = {
    "event",
    "model_config",
    "training_config",
    "git_commit",
    "git_dirty",
    "git_status_porcelain",
    "source_tree_sha256",
    "python_version",
    "pytorch_version",
    "resolved_device",
    "split_manifest",
    "train_partition",
    "validation_partition",
    "train_family_ids",
    "validation_family_ids",
    "resumed",
    "resume_checkpoint",
    "determinism_note",
}
_EPOCH_FIELDS = {
    "event",
    "epoch",
    "global_step",
    "learning_rate",
    "number_of_examples",
    "metrics",
    "codebook",
}
_CHECKPOINT_FIELDS = {
    "event",
    "checkpoint_kind",
    "epoch",
    "global_step",
    "best_validation_metric",
    "relative_path",
}
_FAILURE_FIELDS = {"event", "failure_type", "failure_code", "detail"}
_CODEBOOK_FIELDS = {
    "active_code_count",
    "codebook_utilization",
    "codebook_perplexity",
}
_COMPONENT_WEIGHTS = {
    "node_type": "node_type_loss_weight",
    "categorical_attributes": "categorical_loss_weight",
    "geometry": "geometry_loss_weight",
    "edge_presence": "edge_presence_loss_weight",
    "edge_type": "edge_type_loss_weight",
    "operation_pointer": "operation_pointer_loss_weight",
    "vq_commitment": "vq_loss_weight",
}


class RunAnalysisError(ValueError):
    """A run directory cannot be interpreted under analysis schema version 1."""

    def __init__(self, code, detail):
        self.code = code
        self.detail = detail
        super().__init__("{}: {}".format(code, detail))


def analyze_run(run_dir, torch_module=None):
    """Validate a quiescent run directory and return its deterministic summary."""

    root = Path(run_dir)
    if not root.is_dir():
        raise RunAnalysisError("missing_run_directory", str(root))
    root = root.resolve(strict=True)
    metrics_path = root / "metrics.jsonl"
    if not metrics_path.is_file():
        raise RunAnalysisError("missing_metrics", "metrics.jsonl is absent")

    records, schema_warnings = _read_records(metrics_path)
    metadata_records = [
        item for item in records if item["event"] == "run_metadata"
    ]
    if not metadata_records:
        raise RunAnalysisError("missing_metadata", "no run_metadata record")
    metadata = _validate_metadata(metadata_records, schema_warnings)
    model_config = metadata["model_config"]
    training_config = metadata["training_config"]
    if Path(training_config.output_dir).resolve() != root.resolve():
        raise RunAnalysisError(
            "output_directory_mismatch",
            "embedded training output_dir does not identify the analyzed run",
        )

    failure = _validate_terminal_failure(records, schema_warnings)
    train_records = _epoch_records(
        records, "train_epoch", model_config, schema_warnings
    )
    validation_records = _epoch_records(
        records, "validation_epoch", model_config, schema_warnings
    )
    _validate_epoch_order(
        train_records,
        validation_records,
        metadata_records,
        training_config,
        records,
    )
    _validate_activity_chronology(records, training_config)
    train_steps = {
        item["epoch"]: item["global_step"] for item in train_records
    }
    checkpoint_events = _checkpoint_events(
        records, root, schema_warnings, train_steps
    )
    checkpoints = _inspect_checkpoints(
        root,
        checkpoint_events,
        metadata,
        torch_module,
    )
    _qualify_checkpoint_evidence(
        checkpoints, metadata_records, train_records
    )
    best = _reconcile_best(
        validation_records,
        train_records,
        training_config,
        checkpoints,
    )
    status, status_reasons = _classify_status(
        train_records,
        validation_records,
        training_config,
        checkpoints,
        failure,
    )

    diagnostic_warnings = _diagnostic_warnings(
        train_records,
        validation_records,
        model_config,
        best,
        status,
    )
    warnings = schema_warnings + diagnostic_warnings
    latest_train = train_records[-1] if train_records else None
    latest_validation = validation_records[-1] if validation_records else None
    final = _epoch_snapshot(latest_train, validation_records)

    return {
        "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
        "run_status": status,
        "status_reasons": status_reasons,
        "planned_epochs": training_config.epochs,
        "epochs_completed": len(train_records),
        "latest_train_epoch": (
            None if latest_train is None else latest_train["epoch"]
        ),
        "latest_validation_epoch": (
            None if latest_validation is None else latest_validation["epoch"]
        ),
        "best": best,
        "final": final,
        "vq": {
            "train": _vq_series(train_records),
            "validation": _vq_series(validation_records),
            "statistics": {
                "train": _vq_statistics(train_records),
                "validation": _vq_statistics(validation_records),
            },
        },
        "epoch_records": {
            "train": train_records,
            "validation": validation_records,
        },
        "model_config": model_config.to_dict(),
        "training_config": training_config.to_dict(),
        "checkpoints": checkpoints,
        "terminal_failure": failure,
        "warnings": warnings,
        "warning_thresholds": _warning_thresholds(),
        "scientific_limitations": [
            (
                "Teacher-forced metrics do not establish autonomous decoding; "
                "the decoder can rely on shifted ground-truth context."
            ),
            (
                "Codebook underuse may reflect latent or posterior bypass, not "
                "only excessive codebook capacity."
            ),
            (
                "Raw per-code assignments and code identities were not persisted, "
                "so dead-code identity and turnover cannot be measured."
            ),
            (
                "OpenCascade executability and extrude/revolve reconstruction "
                "are separate evaluations not performed here."
            ),
            "This analyzer cannot declare the Week 3 gate passed.",
        ],
    }


def write_analysis(run_dir, output_dir, torch_module=None):
    """Analyze and atomically publish summary.json and metrics.csv."""

    destination = Path(output_dir)
    if destination.exists():
        raise RunAnalysisError(
            "analysis_output_exists", "output directory already exists"
        )
    if not destination.parent.is_dir():
        raise RunAnalysisError(
            "missing_output_parent", "output parent directory is absent"
        )
    summary = analyze_run(run_dir, torch_module=torch_module)
    temporary = Path(
        tempfile.mkdtemp(
            prefix="." + destination.name + ".tmp-",
            dir=str(destination.parent),
        )
    )
    try:
        _write_json(temporary / "summary.json", summary)
        _write_csv(temporary / "metrics.csv", summary)
        os.rename(str(temporary), str(destination))
    except Exception:
        if temporary.exists():
            shutil.rmtree(str(temporary))
        raise
    return summary


def terminal_summary(summary):
    """Return a concise human-readable report."""

    best = summary["best"]
    final = summary["final"]
    lines = [
        "B0 run analysis: {}".format(summary["run_status"]),
        "epochs: {}/{}".format(
            summary["epochs_completed"], summary["planned_epochs"]
        ),
        "best epoch: {} ({}={})".format(
            _display(best["epoch"]),
            best["selection_metric"],
            _display(best["selection_value"]),
        ),
        "final train total: {}".format(
            _display(
                None
                if final["train_metrics"] is None
                else final["train_metrics"]["total"]
            )
        ),
        "final validation total: {}".format(
            _display(
                None
                if final["validation_metrics"] is None
                else final["validation_metrics"]["total"]
            )
        ),
        "warnings: {}".format(len(summary["warnings"])),
    ]
    if summary["terminal_failure"] is not None:
        lines.append(
            "failure: {} ({})".format(
                summary["terminal_failure"]["failure_code"],
                summary["terminal_failure"]["failure_type"],
            )
        )
    return "\n".join(lines)


def _read_records(path):
    try:
        payload = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise RunAnalysisError(
            "metrics_read_failure", type(exc).__name__
        ) from exc
    if not payload:
        raise RunAnalysisError("empty_metrics", "metrics.jsonl is empty")
    if not payload.endswith("\n"):
        raise RunAnalysisError(
            "noncanonical_jsonl", "metrics.jsonl must end with one newline"
        )
    warnings = []
    records = []
    for line_number, line in enumerate(payload.splitlines(), 1):
        if not line:
            raise RunAnalysisError(
                "blank_jsonl_line", "line {}".format(line_number)
            )
        try:
            record = json.loads(
                line,
                object_pairs_hook=_no_duplicate_keys,
                parse_constant=_reject_nonfinite_constant,
            )
        except RunAnalysisError:
            raise
        except Exception as exc:
            raise RunAnalysisError(
                "malformed_jsonl",
                "line {}: {}".format(line_number, type(exc).__name__),
            ) from exc
        if not isinstance(record, dict):
            raise RunAnalysisError(
                "malformed_record", "line {} is not an object".format(line_number)
            )
        try:
            canonical = json.dumps(
                record,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
        except ValueError as exc:
            raise RunAnalysisError(
                "nonfinite_json_constant", "line {}".format(line_number)
            ) from exc
        if canonical != line:
            raise RunAnalysisError(
                "noncanonical_jsonl", "line {}".format(line_number)
            )
        event = record.get("event")
        if event not in _KNOWN_EVENTS:
            raise RunAnalysisError(
                "unknown_event",
                "line {} event {!r}".format(line_number, event),
            )
        records.append(record)
    return records, warnings


def _validate_metadata(records, warnings):
    parsed = []
    for index, record in enumerate(records):
        _required_fields(record, _METADATA_FIELDS, "run_metadata", warnings)
        try:
            _require_complete_configuration(
                record["model_config"], FlatBaselineConfig, "model_config"
            )
            _require_complete_configuration(
                record["training_config"], TrainingConfig, "training_config"
            )
            model = FlatBaselineConfig(**record["model_config"])
            training = TrainingConfig(**record["training_config"])
            model.validate()
            training.validate()
        except (TypeError, ValueError) as exc:
            raise RunAnalysisError(
                "invalid_embedded_configuration", str(exc)
            ) from exc
        for name in ("train_family_ids", "validation_family_ids"):
            values = record[name]
            if (
                not isinstance(values, list)
                or not values
                or any(not isinstance(item, str) for item in values)
                or len(values) != len(set(values))
            ):
                raise RunAnalysisError(
                    "invalid_metadata", "{} is malformed".format(name)
                )
        if set(record["train_family_ids"]) & set(record["validation_family_ids"]):
            raise RunAnalysisError(
                "metadata_family_leakage", "train and validation families overlap"
            )
        if not isinstance(record["resumed"], bool):
            raise RunAnalysisError("invalid_metadata", "resumed must be boolean")
        resume_checkpoint = record["resume_checkpoint"]
        if record["resumed"]:
            if (
                isinstance(resume_checkpoint, bool)
                or not isinstance(resume_checkpoint, str)
                or not resume_checkpoint
            ):
                raise RunAnalysisError(
                    "invalid_metadata",
                    "resumed metadata requires a nonempty resume_checkpoint string",
                )
        elif resume_checkpoint is not None:
            raise RunAnalysisError(
                "invalid_metadata",
                "non-resumed metadata requires a null resume_checkpoint",
            )
        if index > 0 and not record["resumed"]:
            raise RunAnalysisError(
                "incompatible_metadata",
                "subsequent metadata must describe a resume",
            )
        for name in (
            "git_commit",
            "source_tree_sha256",
            "python_version",
            "pytorch_version",
            "resolved_device",
            "split_manifest",
            "train_partition",
            "validation_partition",
            "determinism_note",
        ):
            if not isinstance(record[name], str) or not record[name]:
                raise RunAnalysisError(
                    "invalid_metadata", "{} is malformed".format(name)
                )
        if (
            not isinstance(record["git_dirty"], bool)
            or not isinstance(record["git_status_porcelain"], list)
            or any(
                not isinstance(item, str)
                for item in record["git_status_porcelain"]
            )
        ):
            raise RunAnalysisError(
                "invalid_metadata", "Git provenance is malformed"
            )
        parsed.append((record, model, training))

    first_record, first_model, first_training = parsed[0]
    previous_epochs = first_training.epochs
    compatibility_fields = (
        "split_manifest",
        "train_partition",
        "validation_partition",
        "train_family_ids",
        "validation_family_ids",
    )
    for record, model, training in parsed[1:]:
        if model.to_dict() != first_model.to_dict():
            raise RunAnalysisError(
                "incompatible_metadata", "model configuration changed"
            )
        if training.resume_signature() != first_training.resume_signature():
            raise RunAnalysisError(
                "incompatible_metadata", "training resume signature changed"
            )
        if training.epochs < previous_epochs:
            raise RunAnalysisError(
                "incompatible_metadata", "requested epochs decreased on resume"
            )
        if any(record[name] != first_record[name] for name in compatibility_fields):
            raise RunAnalysisError(
                "incompatible_metadata", "training data selection changed"
            )
        previous_epochs = training.epochs
        if (
            record["git_commit"] != first_record["git_commit"]
            or record["source_tree_sha256"] != first_record["source_tree_sha256"]
            or record["git_status_porcelain"]
            != first_record["git_status_porcelain"]
        ):
            warnings.append(
                _warning(
                    "metadata_provenance_changed",
                    "Source provenance changed across resumed metadata records.",
                    "identical Git and source-tree provenance",
                    [],
                    {"metadata_record_index": records.index(record) + 1},
                )
            )
    last_record, last_model, last_training = parsed[-1]
    return {
        "record": last_record,
        "model_config": last_model,
        "training_config": last_training,
        "data_state": {
            name: last_record[name]
            for name in compatibility_fields
        },
    }


def _validate_terminal_failure(records, warnings):
    failures = [
        (index, item)
        for index, item in enumerate(records)
        if item["event"] == "failure"
    ]
    if not failures:
        return None
    if len(failures) != 1 or failures[0][0] != len(records) - 1:
        raise RunAnalysisError(
            "nonterminal_failure",
            "failure must be the single final log record",
        )
    record = failures[0][1]
    _required_fields(record, _FAILURE_FIELDS, "failure", warnings)
    for name in ("failure_type", "failure_code", "detail"):
        if not isinstance(record[name], str) or not record[name]:
            raise RunAnalysisError(
                "invalid_failure_record", "{} is malformed".format(name)
            )
    return {
        name: record[name]
        for name in ("failure_type", "failure_code", "detail")
    }


def _epoch_records(records, event, model_config, warnings):
    result = []
    for record in records:
        if record["event"] != event:
            continue
        _required_fields(record, _EPOCH_FIELDS, event, warnings)
        _positive_integer(record["epoch"], "epoch")
        _nonnegative_integer(record["global_step"], "global_step")
        _positive_integer(record["number_of_examples"], "number_of_examples")
        _finite_number(record["learning_rate"], "learning_rate", positive=True)
        metrics = record["metrics"]
        if not isinstance(metrics, dict):
            raise RunAnalysisError("invalid_metrics", event)
        missing = set(LOSS_METRICS) - set(metrics)
        if missing:
            raise RunAnalysisError(
                "missing_metric", "{}: {}".format(event, sorted(missing))
            )
        extra = set(metrics) - set(LOSS_METRICS)
        if extra:
            warnings.append(
                _warning(
                    "unknown_fields",
                    "A known event contains forward-compatible unknown fields.",
                    "analysis schema version 1 fields",
                    [record["epoch"]],
                    {"location": "metrics", "fields": sorted(extra)},
                )
            )
        normalized_metrics = {}
        for name in LOSS_METRICS:
            normalized_metrics[name] = _finite_number(
                metrics[name], "{}.{}".format(event, name), nonnegative=True
            )
        codebook = record["codebook"]
        if not isinstance(codebook, dict):
            raise RunAnalysisError("invalid_codebook", event)
        missing_codebook = _CODEBOOK_FIELDS - set(codebook)
        if missing_codebook:
            raise RunAnalysisError(
                "missing_codebook_metric",
                "{}: {}".format(event, sorted(missing_codebook)),
            )
        extra_codebook = set(codebook) - _CODEBOOK_FIELDS
        if extra_codebook:
            warnings.append(
                _warning(
                    "unknown_fields",
                    "A known event contains forward-compatible unknown fields.",
                    "analysis schema version 1 fields",
                    [record["epoch"]],
                    {"location": "codebook", "fields": sorted(extra_codebook)},
                )
            )
        active = codebook["active_code_count"]
        _nonnegative_integer(active, "active_code_count")
        if active > model_config.codebook_size:
            raise RunAnalysisError(
                "invalid_codebook", "active count exceeds codebook size"
            )
        utilization = _finite_number(
            codebook["codebook_utilization"],
            "codebook_utilization",
            nonnegative=True,
        )
        perplexity = _finite_number(
            codebook["codebook_perplexity"],
            "codebook_perplexity",
            nonnegative=True,
        )
        if utilization > 1.0:
            raise RunAnalysisError(
                "invalid_codebook", "diagnostic exceeds codebook bounds"
            )
        expected_utilization = float(active) / float(model_config.codebook_size)
        if not math.isclose(
            utilization, expected_utilization, rel_tol=_REL_TOL, abs_tol=_ABS_TOL
        ):
            raise RunAnalysisError(
                "inconsistent_codebook", "utilization disagrees with active count"
            )
        if active == 0:
            if not math.isclose(
                perplexity, 0.0, rel_tol=_REL_TOL, abs_tol=_ABS_TOL
            ):
                raise RunAnalysisError(
                    "inconsistent_codebook",
                    "zero active codes require zero perplexity",
                )
        elif (
            perplexity < 1.0
            and not math.isclose(
                perplexity, 1.0, rel_tol=_REL_TOL, abs_tol=_ABS_TOL
            )
        ) or (
            perplexity > float(active)
            and not math.isclose(
                perplexity,
                float(active),
                rel_tol=_REL_TOL,
                abs_tol=_ABS_TOL,
            )
        ):
            raise RunAnalysisError(
                "inconsistent_codebook",
                "perplexity must lie between one and the active-code count",
            )
        _validate_weighted_total(normalized_metrics, model_config, record["epoch"])
        result.append(
            {
                "epoch": record["epoch"],
                "global_step": record["global_step"],
                "learning_rate": float(record["learning_rate"]),
                "number_of_examples": record["number_of_examples"],
                "metrics": normalized_metrics,
                "codebook": {
                    "active_code_count": active,
                    "codebook_utilization": utilization,
                    "codebook_perplexity": perplexity,
                },
            }
        )
    return result


def _validate_weighted_total(metrics, model_config, epoch):
    contribution = sum(
        float(getattr(model_config, field)) * metrics[name]
        for name, field in _COMPONENT_WEIGHTS.items()
    )
    if not math.isclose(
        contribution,
        metrics["total"],
        rel_tol=_LOSS_REL_TOL,
        abs_tol=_ABS_TOL,
    ):
        raise RunAnalysisError(
            "inconsistent_total_loss",
            "epoch {} total={} weighted_components={}".format(
                epoch, metrics["total"], contribution
            ),
        )


def _validate_epoch_order(
    train, validation, metadata_records, training_config, all_records
):
    _strict_epochs(train, "train")
    _strict_epochs(validation, "validation")
    train_epochs = {item["epoch"] for item in train}
    if any(item["epoch"] not in train_epochs for item in validation):
        raise RunAnalysisError(
            "validation_without_train",
            "every validation epoch must have a train epoch",
        )
    train_steps = {item["epoch"]: item["global_step"] for item in train}
    for item in validation:
        if item["global_step"] != train_steps[item["epoch"]]:
            raise RunAnalysisError(
                "inconsistent_epoch_global_step",
                "validation epoch {} uses step {}, expected {}".format(
                    item["epoch"],
                    item["global_step"],
                    train_steps[item["epoch"]],
                ),
            )
    if train:
        first = train[0]["epoch"]
        first_metadata_resumed = metadata_records[0]["resumed"]
        if first != 1 and not first_metadata_resumed:
            raise RunAnalysisError(
                "missing_train_epoch", "ordinary run does not start at epoch 1"
            )
        expected = list(range(first, train[-1]["epoch"] + 1))
        if [item["epoch"] for item in train] != expected:
            raise RunAnalysisError(
                "missing_train_epoch", "training epochs are not consecutive"
            )

    allowed_extra_validation = {
        item["training_config"]["epochs"] for item in metadata_records
    }
    validation_epochs = {item["epoch"] for item in validation}
    for item in validation:
        epoch = item["epoch"]
        if (
            epoch % training_config.validation_interval != 0
            and epoch not in allowed_extra_validation
            and epoch != training_config.epochs
        ):
            raise RunAnalysisError(
                "unexpected_validation_epoch", str(epoch)
            )
    if train:
        latest = train[-1]["epoch"]
        for epoch in range(train[0]["epoch"], latest + 1):
            scheduled = (
                epoch % training_config.validation_interval == 0
                or epoch == training_config.epochs
                or epoch in allowed_extra_validation
            )
            if scheduled and epoch not in validation_epochs and epoch != latest:
                raise RunAnalysisError(
                    "missing_validation_epoch", str(epoch)
                )

    previous_step = -1
    for record in all_records:
        if record["event"] not in (
            "train_epoch",
            "validation_epoch",
            "checkpoint",
        ):
            continue
        step = record["global_step"]
        _nonnegative_integer(step, "global_step")
        if step < previous_step:
            raise RunAnalysisError(
                "decreasing_global_step", "{} < {}".format(step, previous_step)
            )
        previous_step = step


def _validate_activity_chronology(records, final_training_config):
    if records[0]["event"] != "run_metadata":
        raise RunAnalysisError(
            "activity_before_metadata",
            "the first log record must be run_metadata",
        )
    active_training = None
    train_counts = None
    validation_counts = None
    seen_train = set()
    for record in records:
        event = record["event"]
        if event == "run_metadata":
            active_training = TrainingConfig(**record["training_config"])
            train_counts = len(record["train_family_ids"])
            validation_counts = len(record["validation_family_ids"])
            continue
        if event not in ("train_epoch", "validation_epoch", "checkpoint"):
            continue
        if active_training is None:
            raise RunAnalysisError(
                "activity_before_metadata", event
            )
        epoch = record["epoch"]
        if (
            epoch > active_training.epochs
            or epoch > final_training_config.epochs
        ):
            raise RunAnalysisError(
                "epoch_exceeds_plan",
                "epoch {} exceeds active configured limit {}".format(
                    epoch, active_training.epochs
                ),
            )
        if event == "train_epoch":
            seen_train.add(epoch)
            expected_examples = train_counts
        else:
            if epoch not in seen_train:
                raise RunAnalysisError(
                    "activity_before_train",
                    "{} precedes train epoch {}".format(event, epoch),
                )
            expected_examples = (
                validation_counts if event == "validation_epoch" else None
            )
        if expected_examples is not None and (
            record["number_of_examples"] != expected_examples
        ):
            raise RunAnalysisError(
                "inconsistent_example_count",
                "{} epoch {} reports {}, expected {}".format(
                    event,
                    epoch,
                    record["number_of_examples"],
                    expected_examples,
                ),
            )
        if event in ("train_epoch", "validation_epoch") and not math.isclose(
            float(record["learning_rate"]),
            float(active_training.learning_rate),
            rel_tol=_REL_TOL,
            abs_tol=_ABS_TOL,
        ):
            raise RunAnalysisError(
                "inconsistent_learning_rate",
                "{} epoch {}".format(event, epoch),
            )


def _strict_epochs(records, label):
    previous = 0
    for record in records:
        if record["epoch"] <= previous:
            raise RunAnalysisError(
                "nonincreasing_epoch", "{} epoch {}".format(label, record["epoch"])
            )
        previous = record["epoch"]


def _checkpoint_events(records, root, warnings, train_steps):
    result = []
    seen_train = set()
    seen_validation = set()
    previous_epoch = {"best": 0, "last": 0}
    for record in records:
        if record["event"] == "train_epoch":
            seen_train.add(record["epoch"])
            continue
        if record["event"] == "validation_epoch":
            seen_validation.add(record["epoch"])
            continue
        if record["event"] != "checkpoint":
            continue
        _required_fields(record, _CHECKPOINT_FIELDS, "checkpoint", warnings)
        if record["checkpoint_kind"] not in ("best", "last"):
            raise RunAnalysisError("invalid_checkpoint_event", "unknown kind")
        kind = record["checkpoint_kind"]
        _positive_integer(record["epoch"], "checkpoint epoch")
        _nonnegative_integer(record["global_step"], "checkpoint global_step")
        expected_step = train_steps.get(record["epoch"])
        if expected_step is None:
            raise RunAnalysisError(
                "checkpoint_without_train", str(record["epoch"])
            )
        if record["global_step"] != expected_step:
            raise RunAnalysisError(
                "inconsistent_epoch_global_step",
                "{} checkpoint epoch {} uses step {}, expected {}".format(
                    kind,
                    record["epoch"],
                    record["global_step"],
                    expected_step,
                ),
            )
        if record["epoch"] <= previous_epoch[kind]:
            raise RunAnalysisError(
                "nonincreasing_checkpoint_epoch", kind
            )
        if record["epoch"] not in seen_train:
            raise RunAnalysisError(
                "checkpoint_without_train", str(record["epoch"])
            )
        if kind == "best" and record["epoch"] not in seen_validation:
            raise RunAnalysisError(
                "best_checkpoint_without_validation", str(record["epoch"])
            )
        previous_epoch[kind] = record["epoch"]
        metric = record["best_validation_metric"]
        if metric is not None:
            _finite_number(metric, "best_validation_metric")
        relative = record["relative_path"]
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
            or Path(relative).name != relative
            or relative != record["checkpoint_kind"] + ".pt"
        ):
            raise RunAnalysisError(
                "unsafe_checkpoint_path", repr(relative)
            )
        path = root / relative
        if not path.is_file():
            raise RunAnalysisError(
                "missing_logged_checkpoint", relative
            )
        _contained_checkpoint_path(root, path)
        result.append(dict(record))
    return result


def _inspect_checkpoints(root, events, metadata, torch_module):
    result = {}
    for kind in ("best", "last"):
        path = root / (kind + ".pt")
        if not path.is_file():
            result[kind] = {
                "relative_path": kind + ".pt",
                "exists": False,
                "epoch": None,
                "global_step": None,
                "best_validation_metric": None,
                "checkpoint_kind": None,
                "checkpoint_version": None,
                "logged": False,
                "considered": False,
                "preserved_relocated_best": False,
            }
            continue
        resolved_path = _contained_checkpoint_path(root, path)
        checkpoint = _load_checkpoint(resolved_path, torch_module)
        try:
            _validate_payload(checkpoint)
        except CheckpointError as exc:
            raise RunAnalysisError(
                "invalid_checkpoint", "{}: {}".format(kind, exc)
            ) from exc
        if checkpoint["checkpoint_kind"] != kind:
            raise RunAnalysisError(
                "invalid_checkpoint", "{} kind mismatch".format(kind)
            )
        if checkpoint["model_config"] != metadata["model_config"].to_dict():
            raise RunAnalysisError(
                "checkpoint_configuration_mismatch",
                "{} model configuration".format(kind),
            )
        try:
            _require_complete_configuration(
                checkpoint["training_config"],
                TrainingConfig,
                "checkpoint training_config",
            )
            saved_training = TrainingConfig(**checkpoint["training_config"])
            saved_training.validate()
        except (TypeError, ValueError) as exc:
            raise RunAnalysisError(
                "invalid_checkpoint", "{} training configuration".format(kind)
            ) from exc
        if (
            saved_training.resume_signature()
            != metadata["training_config"].resume_signature()
        ):
            raise RunAnalysisError(
                "checkpoint_configuration_mismatch",
                "{} training configuration".format(kind),
            )
        if checkpoint["data_state"] != metadata["data_state"]:
            raise RunAnalysisError(
                "checkpoint_data_mismatch", kind
            )
        matching_events = [
            item for item in events if item["checkpoint_kind"] == kind
        ]
        if matching_events:
            latest = matching_events[-1]
            if (
                latest["epoch"] != checkpoint["epoch"]
                or latest["global_step"] != checkpoint["global_step"]
                or not _optional_close(
                    latest["best_validation_metric"],
                    checkpoint["best_validation_metric"],
                )
            ):
                raise RunAnalysisError(
                    "checkpoint_event_mismatch", kind
                )
        result[kind] = {
            "relative_path": kind + ".pt",
            "exists": True,
            "epoch": checkpoint["epoch"],
            "global_step": checkpoint["global_step"],
            "best_validation_metric": checkpoint["best_validation_metric"],
            "checkpoint_kind": checkpoint["checkpoint_kind"],
            "checkpoint_version": checkpoint["checkpoint_version"],
            "logged": bool(matching_events),
            "considered": False,
            "preserved_relocated_best": False,
        }
    return result


def _contained_checkpoint_path(root, path):
    try:
        resolved_root = Path(root).resolve(strict=True)
        resolved_path = Path(path).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise RunAnalysisError(
            "unsafe_checkpoint_path", Path(path).name
        ) from exc
    if (
        resolved_path.parent != resolved_root
        or not resolved_path.is_file()
    ):
        raise RunAnalysisError(
            "unsafe_checkpoint_path", Path(path).name
        )
    return resolved_path


def _qualify_checkpoint_evidence(checkpoints, metadata_records, train):
    for checkpoint in checkpoints.values():
        checkpoint["considered"] = checkpoint["exists"] and checkpoint["logged"]
    best = checkpoints["best"]
    if best["considered"] or not best["exists"] or not train:
        return
    first_metadata = metadata_records[0]
    first_train_epoch = train[0]["epoch"]
    if (
        first_metadata["resumed"]
        and isinstance(first_metadata["resume_checkpoint"], str)
        and first_metadata["resume_checkpoint"]
        and best["epoch"] < first_train_epoch
    ):
        best["considered"] = True
        best["preserved_relocated_best"] = True


def _load_checkpoint(path, torch_module):
    if torch_module is None:
        try:
            import torch as torch_module
        except ImportError as exc:
            raise RunAnalysisError(
                "pytorch_unavailable",
                "checkpoint inspection requires PyTorch",
            ) from exc
    try:
        return torch_module.load(str(path), map_location="cpu")
    except Exception as exc:
        raise RunAnalysisError(
            "checkpoint_load_failure",
            "{}: {}".format(path.name, type(exc).__name__),
        ) from exc


def _reconcile_best(validation, train, training_config, checkpoints):
    selection = training_config.checkpoint_selection_metric
    best_checkpoint = checkpoints["best"]
    last_checkpoint = checkpoints["last"]
    if (
        best_checkpoint["considered"]
        and last_checkpoint["considered"]
        and not _optional_close(
            best_checkpoint["best_validation_metric"],
            last_checkpoint["best_validation_metric"],
        )
    ):
        raise RunAnalysisError(
            "inconsistent_best_checkpoint",
            "best.pt and last.pt preserve different best metrics",
        )
    if train:
        latest_train_epoch = train[-1]["epoch"]
        for kind in ("best", "last"):
            checkpoint = checkpoints[kind]
            if (
                checkpoint["considered"]
                and checkpoint["epoch"] > latest_train_epoch
            ):
                raise RunAnalysisError(
                    "checkpoint_ahead_of_metrics", kind
                )
    if not validation:
        if best_checkpoint["considered"]:
            return {
                "epoch": best_checkpoint["epoch"],
                "selection_metric": selection,
                "selection_value": best_checkpoint["best_validation_metric"],
                "train_metrics": None,
                "validation_metrics": None,
                "gaps": None,
                "limitation_reason": (
                    "Best checkpoint predates the available metric history."
                ),
            }
        return {
            "epoch": None,
            "selection_metric": selection,
            "selection_value": None,
            "train_metrics": None,
            "validation_metrics": None,
            "gaps": None,
            "limitation_reason": "No validation epoch is available.",
        }
    local_best = min(validation, key=lambda item: item["metrics"][selection])
    if best_checkpoint["considered"]:
        checkpoint_epoch = best_checkpoint["epoch"]
        checkpoint_metric = best_checkpoint["best_validation_metric"]
        available = {item["epoch"]: item for item in validation}
        if checkpoint_epoch in available:
            observed = available[checkpoint_epoch]["metrics"][selection]
            if not _optional_close(observed, checkpoint_metric):
                raise RunAnalysisError(
                    "inconsistent_best_checkpoint",
                    "checkpoint metric disagrees with validation",
                )
            if checkpoint_epoch != local_best["epoch"]:
                raise RunAnalysisError(
                    "inconsistent_best_checkpoint",
                    "strict-improvement best epoch disagrees",
                )
        elif checkpoint_epoch >= validation[0]["epoch"]:
            raise RunAnalysisError(
                "inconsistent_best_checkpoint",
                "best epoch is missing inside available history",
            )
        elif checkpoint_metric > local_best["metrics"][selection] and not math.isclose(
            checkpoint_metric,
            local_best["metrics"][selection],
            rel_tol=_REL_TOL,
            abs_tol=_ABS_TOL,
        ):
            raise RunAnalysisError(
                "inconsistent_best_checkpoint",
                "earlier preserved best is worse than available validation",
            )
        else:
            return {
                "epoch": checkpoint_epoch,
                "selection_metric": selection,
                "selection_value": checkpoint_metric,
                "train_metrics": None,
                "validation_metrics": None,
                "gaps": None,
                "limitation_reason": (
                    "Best checkpoint predates the available metric history."
                ),
            }
    selected = local_best
    train_by_epoch = {item["epoch"]: item for item in train}
    train_metrics = train_by_epoch[selected["epoch"]]["metrics"]
    validation_metrics = selected["metrics"]
    return {
        "epoch": selected["epoch"],
        "selection_metric": selection,
        "selection_value": validation_metrics[selection],
        "train_metrics": dict(train_metrics),
        "validation_metrics": dict(validation_metrics),
        "gaps": _gaps(train_metrics, validation_metrics),
        "limitation_reason": None,
    }


def _classify_status(train, validation, config, checkpoints, failure):
    latest = None if not train else train[-1]["epoch"]
    validation_epochs = {item["epoch"] for item in validation}
    complete_epochs = (
        latest == config.epochs
        and len(train) == config.epochs
        and all(
            epoch in validation_epochs
            for epoch in range(1, config.epochs + 1)
            if epoch % config.validation_interval == 0
            or epoch == config.epochs
        )
    )
    complete_checkpoints = (
        checkpoints["last"]["considered"]
        and checkpoints["last"]["epoch"] == config.epochs
        and checkpoints["best"]["considered"]
    )
    if failure is None and complete_epochs and complete_checkpoints:
        return "complete", ["all configured epochs and checkpoints are complete"]
    reasons = []
    if failure is not None:
        reasons.append("run ended with a structured failure")
    if latest != config.epochs:
        reasons.append("configured final epoch was not reached")
    if latest is not None and latest not in validation_epochs:
        reasons.append("latest train epoch has no validation record")
    if not checkpoints["last"]["considered"]:
        reasons.append("last.pt has no accepted checkpoint evidence")
    if validation and not checkpoints["best"]["considered"]:
        reasons.append("best.pt has no accepted checkpoint evidence")
    return "incomplete", reasons or ["run is a valid incomplete prefix"]


def _epoch_snapshot(train_record, validation_records):
    if train_record is None:
        return {
            "epoch": None,
            "train_metrics": None,
            "validation_metrics": None,
            "gaps": None,
            "validation_limitation_reason": "No training epoch is available.",
        }
    validation_by_epoch = {
        item["epoch"]: item for item in validation_records
    }
    validation = validation_by_epoch.get(train_record["epoch"])
    return {
        "epoch": train_record["epoch"],
        "train_metrics": dict(train_record["metrics"]),
        "validation_metrics": (
            None if validation is None else dict(validation["metrics"])
        ),
        "gaps": (
            None
            if validation is None
            else _gaps(train_record["metrics"], validation["metrics"])
        ),
        "validation_limitation_reason": (
            None
            if validation is not None
            else "Validation was not recorded at the latest train epoch."
        ),
    }


def _vq_series(records):
    return [
        {"epoch": item["epoch"], **item["codebook"]}
        for item in records
    ]


def _vq_statistics(records):
    result = {}
    for name in (
        "active_code_count",
        "codebook_utilization",
        "codebook_perplexity",
    ):
        values = [item["codebook"][name] for item in records]
        result[name] = {
            "minimum": None if not values else min(values),
            "maximum": None if not values else max(values),
            "final": None if not values else values[-1],
        }
    return result


def _diagnostic_warnings(train, validation, config, best, status):
    warnings = []
    paired = _paired(train, validation)
    warnings.extend(_degradation_warnings(paired))
    warnings.extend(_nonimproving_warnings(validation))
    warnings.extend(_erratic_warnings(validation))
    warnings.extend(
        _consecutive_threshold_warning(
            train,
            "active_code_collapse",
            "Sustained use of at most 12.5% of the codebook.",
            "utilization <= 0.125 for 5 consecutive train epochs after epoch 5",
            lambda item: (
                item["epoch"] > 5
                and item["codebook"]["codebook_utilization"] <= 0.125
            ),
            "codebook_utilization",
        )
    )
    warnings.extend(
        _consecutive_threshold_warning(
            train,
            "substantial_codebook_underuse",
            "Sustained use of less than half of the codebook.",
            "utilization < 0.50 for 5 consecutive train epochs at epoch 10 or later",
            lambda item: (
                item["epoch"] >= 10
                and item["codebook"]["codebook_utilization"] < 0.50
            ),
            "codebook_utilization",
        )
    )
    warnings.extend(
        _consecutive_threshold_warning(
            train,
            "perplexity_near_one",
            "Codebook perplexity remains close to one.",
            "perplexity <= 1.5 for 5 consecutive train epochs",
            lambda item: item["codebook"]["codebook_perplexity"] <= 1.5,
            "codebook_perplexity",
        )
    )
    warnings.extend(_dominance_warnings(train, validation, config))
    if status == "complete" and best["epoch"] is not None and train:
        final_epoch = train[-1]["epoch"]
        threshold = max(10, int(math.ceil(0.25 * final_epoch)))
        if final_epoch - best["epoch"] >= threshold:
            warnings.append(
                _warning(
                    "best_checkpoint_early",
                    "The selected best checkpoint occurred much earlier than the final epoch.",
                    "final_epoch - best_epoch >= {}".format(threshold),
                    [best["epoch"], final_epoch],
                    {
                        "best_epoch": best["epoch"],
                        "final_epoch": final_epoch,
                        "epoch_difference": final_epoch - best["epoch"],
                    },
                )
            )
    return warnings


def _degradation_warnings(paired):
    for start in range(0, max(0, len(paired) - 4)):
        window = paired[start : start + 5]
        train_values = [item[0]["metrics"]["total"] for item in window]
        validation_values = [item[1]["metrics"]["total"] for item in window]
        scale = max(abs(validation_values[0]), _ABS_TOL)
        if (
            all(b < a for a, b in zip(train_values, train_values[1:]))
            and all(b > a for a, b in zip(validation_values, validation_values[1:]))
            and (validation_values[-1] - validation_values[0]) / scale >= 0.05
        ):
            return [
                _warning(
                    "sustained_validation_degradation",
                    "Training improves while validation degrades monotonically.",
                    "5 paired epochs and at least 5% validation increase",
                    [item[0]["epoch"] for item in window],
                    {
                        "train_total": train_values,
                        "validation_total": validation_values,
                        "relative_validation_increase": (
                            validation_values[-1] - validation_values[0]
                        )
                        / scale,
                    },
                )
            ]
    return []


def _nonimproving_warnings(validation):
    if len(validation) < 10:
        return []
    window = validation[-10:]
    first = min(item["metrics"]["total"] for item in window[:5])
    last = min(item["metrics"]["total"] for item in window[5:])
    improvement = (first - last) / max(abs(first), _ABS_TOL)
    if improvement >= 0.01:
        return []
    return [
        _warning(
            "non_improving_validation",
            "Recent validation loss has not improved meaningfully.",
            "less than 1% best-loss improvement between two 5-point halves",
            [item["epoch"] for item in window],
            {
                "first_half_best": first,
                "last_half_best": last,
                "relative_improvement": improvement,
            },
        )
    ]


def _erratic_warnings(validation):
    if len(validation) < 8:
        return []
    values = [item["metrics"]["total"] for item in validation]
    ordered = sorted(abs(value) for value in values)
    median = statistics.median(ordered)
    threshold = 0.01 * max(median, _ABS_TOL)
    deltas = [
        after - before for before, after in zip(values, values[1:])
        if abs(after - before) >= threshold
    ]
    if len(deltas) < 5:
        return []
    flips = sum(
        (before < 0 < after) or (before > 0 > after)
        for before, after in zip(deltas, deltas[1:])
    )
    ratio = float(flips) / float(len(deltas) - 1)
    if ratio < 0.60:
        return []
    return [
        _warning(
            "highly_erratic_validation",
            "Validation loss changes direction repeatedly at meaningful amplitude.",
            "at least 5 deltas and sign-change ratio >= 0.60",
            [item["epoch"] for item in validation],
            {
                "significant_delta_count": len(deltas),
                "sign_change_count": flips,
                "sign_change_ratio": ratio,
                "minimum_delta_magnitude": threshold,
            },
        )
    ]


def _consecutive_threshold_warning(
    records, code, explanation, threshold, predicate, evidence_name
):
    for start in range(0, max(0, len(records) - 4)):
        window = records[start : start + 5]
        if all(predicate(item) for item in window):
            return [
                _warning(
                    code,
                    explanation,
                    threshold,
                    [item["epoch"] for item in window],
                    {
                        evidence_name: [
                            item["codebook"][evidence_name] for item in window
                        ]
                    },
                )
            ]
    return []


def _dominance_warnings(train, validation, config):
    for mode, records in (("train", train), ("validation", validation)):
        for metric, field in _COMPONENT_WEIGHTS.items():
            ratios = []
            for item in records:
                total = item["metrics"]["total"]
                ratio = (
                    0.0
                    if total <= _ABS_TOL
                    else (
                        float(getattr(config, field))
                        * item["metrics"][metric]
                        / total
                    )
                )
                ratios.append((item["epoch"], ratio))
            for start in range(0, max(0, len(ratios) - 2)):
                window = ratios[start : start + 3]
                if all(value >= 0.70 for _, value in window):
                    return [
                        _warning(
                            "component_dominance",
                            "One weighted loss contribution dominates the logged total.",
                            "weighted component / total >= 0.70 for 3 consecutive records",
                            [epoch for epoch, _ in window],
                            {
                                "mode": mode,
                                "component": metric,
                                "weight": float(getattr(config, field)),
                                "contribution_ratios": [
                                    value for _, value in window
                                ],
                            },
                        )
                    ]
    return []


def _warning(code, explanation, threshold, epochs, evidence):
    return {
        "code": code,
        "explanation": explanation,
        "threshold": threshold,
        "affected_epochs": epochs,
        "evidence": evidence,
    }


def _warning_thresholds():
    return {
        "sustained_validation_degradation": (
            "5 paired epochs with monotonically decreasing train total, "
            "monotonically increasing validation total, and >=5% validation increase"
        ),
        "non_improving_validation": (
            "<1% best-loss improvement between the two 5-point halves "
            "of the latest 10 validation records"
        ),
        "highly_erratic_validation": (
            ">=5 deltas of >=1% median-loss magnitude with >=60% sign changes"
        ),
        "active_code_collapse": (
            "utilization <=0.125 for 5 train epochs after epoch 5"
        ),
        "substantial_codebook_underuse": (
            "utilization <0.50 for 5 train epochs at epoch 10 or later"
        ),
        "perplexity_near_one": "perplexity <=1.5 for 5 train epochs",
        "component_dominance": (
            "weighted component contribution >=70% of total for 3 records"
        ),
        "best_checkpoint_early": (
            "final-best epoch gap >=max(10, ceil(25% of final epoch))"
        ),
    }


def _write_json(path, value):
    path.write_text(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_csv(path, summary):
    train = summary["epoch_records"]["train"]
    validation_by_epoch = {
        item["epoch"]: item
        for item in summary["epoch_records"]["validation"]
    }
    fields = [
        "epoch",
        "global_step",
        "learning_rate",
        "train_examples",
        "validation_examples",
    ]
    for prefix in ("train", "validation", "gap"):
        fields.extend("{}_{}".format(prefix, name) for name in LOSS_METRICS)
    for prefix in ("train", "validation"):
        fields.extend(
            "{}_{}".format(prefix, name)
            for name in (
                "active_code_count",
                "codebook_utilization",
                "codebook_perplexity",
            )
        )
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for train_record in train:
            epoch = train_record["epoch"]
            validation = validation_by_epoch.get(epoch)
            row = {
                "epoch": epoch,
                "global_step": train_record["global_step"],
                "learning_rate": train_record["learning_rate"],
                "train_examples": train_record["number_of_examples"],
                "validation_examples": (
                    "" if validation is None else validation["number_of_examples"]
                ),
            }
            for name in LOSS_METRICS:
                row["train_" + name] = train_record["metrics"][name]
                row["validation_" + name] = (
                    "" if validation is None else validation["metrics"][name]
                )
                row["gap_" + name] = (
                    ""
                    if validation is None
                    else validation["metrics"][name]
                    - train_record["metrics"][name]
                )
            for name in (
                "active_code_count",
                "codebook_utilization",
                "codebook_perplexity",
            ):
                row["train_" + name] = train_record["codebook"][name]
                row["validation_" + name] = (
                    "" if validation is None else validation["codebook"][name]
                )
            writer.writerow(row)


def _gaps(train_metrics, validation_metrics):
    return {
        name: validation_metrics[name] - train_metrics[name]
        for name in LOSS_METRICS
    }


def _paired(train, validation):
    validation_by_epoch = {item["epoch"]: item for item in validation}
    return [
        (item, validation_by_epoch[item["epoch"]])
        for item in train
        if item["epoch"] in validation_by_epoch
    ]


def _required_fields(record, required, label, warnings):
    missing = required - set(record)
    if missing:
        raise RunAnalysisError(
            "missing_event_field",
            "{}: {}".format(label, sorted(missing)),
        )
    unknown = set(record) - required
    if unknown:
        epoch = record.get("epoch")
        warnings.append(
            _warning(
                "unknown_fields",
                "A known event contains forward-compatible unknown fields.",
                "analysis schema version 1 fields",
                [] if epoch is None else [epoch],
                {"event": label, "fields": sorted(unknown)},
            )
        )


def _require_complete_configuration(value, configuration_type, label):
    if not isinstance(value, dict):
        raise RunAnalysisError(
            "invalid_embedded_configuration", "{} is not an object".format(label)
        )
    expected = {item.name for item in fields(configuration_type)}
    if set(value) != expected:
        raise RunAnalysisError(
            "invalid_embedded_configuration",
            "{} fields differ: missing={} unknown={}".format(
                label,
                sorted(expected - set(value)),
                sorted(set(value) - expected),
            ),
        )


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RunAnalysisError(
            "invalid_integer", "{} must be positive".format(name)
        )


def _nonnegative_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RunAnalysisError(
            "invalid_integer", "{} must be nonnegative".format(name)
        )


def _finite_number(value, name, positive=False, nonnegative=False):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise RunAnalysisError(
            "nonfinite_or_invalid_number", name
        )
    result = float(value)
    if positive and result <= 0.0:
        raise RunAnalysisError("invalid_number_range", name)
    if nonnegative and result < 0.0:
        raise RunAnalysisError("invalid_number_range", name)
    return result


def _optional_close(first, second):
    if first is None or second is None:
        return first is second
    return math.isclose(
        float(first), float(second), rel_tol=_REL_TOL, abs_tol=_ABS_TOL
    )


def _no_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise RunAnalysisError("duplicate_json_key", key)
        result[key] = value
    return result


def _reject_nonfinite_constant(value):
    raise RunAnalysisError("nonfinite_json_constant", value)


def _display(value):
    if value is None:
        return "unavailable"
    if isinstance(value, float):
        return "{:.8g}".format(value)
    return str(value)
