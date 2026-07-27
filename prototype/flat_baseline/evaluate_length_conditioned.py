"""Deterministic paired evaluation and atomic scientific-artifact publication."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import ctypes
from dataclasses import dataclass, fields, is_dataclass, replace
import errno
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace

from prototype.controlled_data.config import REPRESENTATION_SCHEMA_VERSION
from prototype.model_data.adapters import adapt_flat_mixed
from prototype.model_data.batching import collate_flat
from prototype.model_data.loader import load_physical_examples

from .checkpointing import _validate_payload
from .config import FlatBaselineConfig
from .conversion import validate_and_convert_raw_prediction
from .metrics import aggregate_prediction_metrics, evaluate_prediction
from .provenance import source_state
from .training_config import TrainingConfig


EVALUATION_SCHEMA_VERSION = 1
PATH_NAMES = ("teacher_forced", "predicted_history")
REQUIRED_ARTIFACTS = (
    "conversion_failures.csv",
    "examples.jsonl",
    "metrics.csv",
    "run_metadata.json",
    "summary.json",
)
RAW_ARTIFACT = "raw_predictions.jsonl"
METRICS_COLUMNS = (
    "path",
    "stratum_kind",
    "stratum_value",
    "attempted_example_count",
    "raw_completion_rate",
    "raw_integrity_valid_rate",
    "reconstruction_target_valid_rate",
    "controlled_domain_valid_rate",
    "node_type_token_accuracy",
    "exact_node_type_sequence_rate",
    "finite_geometry_mae",
    "finite_geometry_rmse",
    "complete_finite_geometry_rate",
    "exact_operation_type_sequence_rate",
    "pointer_overall_accuracy",
    "edge_micro_precision",
    "edge_micro_recall",
    "edge_micro_f1",
)
FAILURE_COLUMNS = (
    "family_id",
    "path",
    "designation",
    "code",
    "layer",
    "stage",
    "location",
    "detail",
)
LIMITATION = (
    "The target canonical node count is supplied. Node count partially reveals "
    "the target template: 4→E, 5→R, 7→EE, 8→ER/RE, and 9→RR. Template results "
    "are not autonomous template classification."
)


class EvaluationError(RuntimeError):
    """An operational evaluation contract failed."""

    def __init__(self, code, detail):
        self.code = code
        self.detail = detail
        super().__init__("{}: {}".format(code, detail))


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        raise EvaluationError("usage", message)


@dataclass(frozen=True)
class PathEvaluation:
    conversion: object
    metrics: object


@dataclass(frozen=True)
class ExampleEvaluation:
    family_id: str
    partition: str
    metadata: object
    target_node_count: int
    target_operation_sequence: tuple[str, ...]
    latent_indices: tuple[int, ...]
    teacher_forced: PathEvaluation
    predicted_history: PathEvaluation


def build_parser():
    parser = _Parser(prog="python3 -m prototype.flat_baseline.evaluate_length_conditioned")
    parser.add_argument("--corpus-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument(
        "--partition", required=True, choices=("train", "validation", "test")
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", required=True, type=_positive_integer)
    parser.add_argument("--device", required=True, type=_device_request)
    parser.add_argument("--family-limit", type=_positive_integer)
    parser.add_argument("--allow-test-evaluation", action="store_true")
    parser.add_argument("--write-raw-predictions", action="store_true")
    return parser


def parse_arguments(argv=None):
    arguments = build_parser().parse_args(argv)
    if arguments.partition == "test" and not arguments.allow_test_evaluation:
        raise EvaluationError(
            "test_evaluation_forbidden",
            "--allow-test-evaluation is required for the test partition",
        )
    return arguments


def select_examples(examples, partition, family_limit=None):
    selected = tuple(item for item in examples if item.partition == partition)
    family_ids = tuple(item.physical_family_id for item in selected)
    if len(family_ids) != len(set(family_ids)):
        raise EvaluationError("duplicate_family_id", "selected family IDs repeat")
    selected = tuple(sorted(selected, key=lambda item: item.physical_family_id))
    if family_limit is not None:
        selected = selected[:family_limit]
    if not selected:
        raise EvaluationError("empty_selection", "no authoritative families selected")
    return selected


def _partition_ids(examples, partition):
    identifiers = tuple(sorted(
        item.physical_family_id
        for item in examples
        if item.partition == partition
    ))
    if len(identifiers) != len(set(identifiers)):
        raise EvaluationError(
            "duplicate_family_id",
            "{} family IDs repeat".format(partition),
        )
    return identifiers


def evaluate_records(
    examples,
    *,
    batch_size,
    max_operations,
    decode_batch,
):
    """Evaluate paired raw predictions in authoritative family order."""

    records = []
    raw_records = []
    for start in range(0, len(examples), batch_size):
        physical_batch = examples[start : start + batch_size]
        paired = decode_batch(physical_batch)
        teacher = tuple(paired.teacher_forced)
        predicted = tuple(paired.predicted_history)
        latent = tuple(tuple(row) for row in paired.latent_indices)
        expected = len(physical_batch)
        if not (len(teacher) == len(predicted) == len(latent) == expected):
            raise EvaluationError(
                "decode_count_mismatch",
                "paired decoder output count differs from authoritative batch",
            )
        for index, example in enumerate(physical_batch):
            path_records = []
            for path, raw in zip(
                PATH_NAMES, (teacher[index], predicted[index])
            ):
                stage_two_raw = _stage_two_prediction(raw, path)
                conversion = validate_and_convert_raw_prediction(
                    stage_two_raw, max_operations=max_operations
                )
                metric = evaluate_prediction(
                    stage_two_raw,
                    example.target,
                    family_id=example.physical_family_id,
                    max_operations=max_operations,
                )
                if not _conversion_matches_metrics(conversion, metric):
                    raise EvaluationError(
                        "conversion_metric_mismatch",
                        "{} conversion and metrics differ".format(path),
                    )
                path_records.append(PathEvaluation(conversion, metric))
            records.append(
                ExampleEvaluation(
                    example.physical_family_id,
                    example.partition,
                    example.metadata,
                    len(example.target.node_type_ids),
                    tuple(example.operation_sequence),
                    latent[index],
                    path_records[0],
                    path_records[1],
                )
            )
            raw_records.append({
                "family_id": example.physical_family_id,
                "teacher_forced": teacher[index],
                "predicted_history": predicted[index],
            })
    if len(records) != len(examples):
        raise EvaluationError(
            "processed_count_mismatch",
            "processed record count differs from selected family count",
        )
    if tuple(item.family_id for item in records) != tuple(
        item.physical_family_id for item in examples
    ):
        raise EvaluationError("family_order_mismatch", "evaluation reordered families")
    return tuple(records), tuple(raw_records)


def _stage_two_prediction(raw, path):
    """Return the provenance-only compatibility view required by Stage 2."""

    if (
        path == "teacher_forced"
        and raw is not None
        and getattr(raw, "prefix_feedback", None) == "shifted_target_prefix"
    ):
        if isinstance(raw, SimpleNamespace):
            values = vars(raw).copy()
            values["prefix_feedback"] = "raw_argmax_with_derived_geometry_mask"
            return SimpleNamespace(**values)
        try:
            return replace(
                raw,
                prefix_feedback="raw_argmax_with_derived_geometry_mask",
            )
        except TypeError as exc:
            raise EvaluationError(
                "invalid_teacher_forced_record",
                "teacher-forced raw prediction is not a dataclass",
            ) from exc
    return raw


def _conversion_matches_metrics(conversion, metric):
    return (
        conversion.raw_integrity == metric.raw_integrity
        and conversion.reconstruction_target == metric.reconstruction_target
        and conversion.controlled_domain == metric.controlled_domain
        and conversion.primary_failure == metric.primary_failure
        and conversion.secondary_failures == metric.secondary_failures
    )


def make_artifacts(records, raw_records, metadata, examples, write_raw):
    metadata_by_family = {
        item.physical_family_id: item for item in examples
    }
    aggregates = {}
    for path in PATH_NAMES:
        metrics = tuple(getattr(item, path).metrics for item in records)
        aggregates[path] = aggregate_prediction_metrics(
            metrics, metadata_by_family=metadata_by_family
        )
    aggregate_json = {
        name: _aggregate_json(value) for name, value in aggregates.items()
    }
    summary = {
        "evaluation_schema_version": EVALUATION_SCHEMA_VERSION,
        "attempted_example_count": len(records),
        "teacher_forced": aggregate_json["teacher_forced"],
        "predicted_history": aggregate_json["predicted_history"],
        "predicted_minus_teacher_forced": _semantic_difference(
            aggregate_json["predicted_history"],
            aggregate_json["teacher_forced"],
        ),
    }
    artifacts = {
        "run_metadata.json": _json_document(metadata),
        "examples.jsonl": _json_lines(_example_json(item) for item in records),
        "summary.json": _json_document(summary),
        "metrics.csv": _metrics_csv(aggregates),
        "conversion_failures.csv": _failures_csv(records),
    }
    if write_raw:
        artifacts[RAW_ARTIFACT] = _json_lines(
            _raw_json_safe(item) for item in raw_records
        )
    return artifacts


def publish_artifacts(
    output_dir,
    artifacts,
    *,
    expected_ids=None,
    expected_partition=None,
    authoritative_validation_ids=None,
    authoritative_test_ids=(),
    authoritative_examples=None,
    reviewed_commit=None,
    expected_checkpoint_sha256=None,
):
    destination = Path(output_dir)
    _preflight_output(destination)
    parent = destination.parent
    temporary = None
    try:
        temporary = Path(tempfile.mkdtemp(
            prefix="." + destination.name + ".tmp-", dir=str(parent)
        ))
        for name in sorted(artifacts):
            path = temporary / name
            with path.open("wb") as stream:
                stream.write(artifacts[name])
                stream.flush()
                os.fsync(stream.fileno())
        _fsync_directory(temporary)
        validate_artifact_directory(
            temporary,
            expected_ids=expected_ids,
            expected_partition=expected_partition,
            authoritative_validation_ids=authoritative_validation_ids,
            authoritative_test_ids=authoritative_test_ids,
            authoritative_examples=authoritative_examples,
            reviewed_commit=reviewed_commit,
            expected_checkpoint_sha256=expected_checkpoint_sha256,
        )
        _atomic_no_replace(temporary, destination)
        temporary = None
        _fsync_directory(parent)
    except EvaluationError:
        raise
    except (OSError, ValueError, TypeError) as exc:
        raise EvaluationError(
            "publication_failure", type(exc).__name__
        ) from exc
    finally:
        if temporary is not None and temporary.exists():
            _remove_temporary_directory(temporary)


def completed_exit_code(records):
    structural_invalid = any(
        not getattr(item, path).conversion.raw_integrity.valid
        for item in records
        for path in PATH_NAMES
    )
    return 2 if structural_invalid else 0


def validate_artifact_directory(
    directory,
    raw_enabled=None,
    *,
    expected_ids=None,
    expected_partition=None,
    authoritative_validation_ids=None,
    authoritative_test_ids=(),
    authoritative_examples=None,
    reviewed_commit=None,
    expected_checkpoint_sha256=None,
):
    """Validate the complete logical publication, before or after rename."""

    root = Path(directory)
    metadata = _strict_document(root / "run_metadata.json")
    enabled = metadata.get("raw_predictions_published")
    if not isinstance(enabled, bool):
        raise EvaluationError("invalid_metadata", "raw publication flag is invalid")
    if raw_enabled is not None and raw_enabled != enabled:
        raise EvaluationError("raw_publication_mismatch", "raw setting differs")
    expected = set(REQUIRED_ARTIFACTS)
    if enabled:
        expected.add(RAW_ARTIFACT)
    actual = {path.name for path in root.iterdir() if path.is_file()}
    if actual != expected or any(not path.is_file() for path in root.iterdir()):
        raise EvaluationError("artifact_set_mismatch", "published artifact set is wrong")
    for path in sorted(root.iterdir()):
        _validate_file_encoding(path)
    summary = _strict_document(root / "summary.json")
    examples = _strict_lines(root / "examples.jsonl")
    raw = _strict_lines(root / RAW_ARTIFACT) if enabled else ()
    metric_rows = _strict_csv(root / "metrics.csv", METRICS_COLUMNS)
    failure_rows = _strict_csv(
        root / "conversion_failures.csv", FAILURE_COLUMNS
    )
    identifiers = _validate_metadata_and_examples(
        metadata,
        summary,
        examples,
        raw,
        expected_ids,
        expected_partition,
        authoritative_validation_ids,
        authoritative_test_ids,
        reviewed_commit,
        expected_checkpoint_sha256,
    )
    _reconcile_examples_with_authority(
        examples,
        raw,
        metadata,
        summary,
        authoritative_examples,
    )
    _validate_metric_rows(summary, metric_rows)
    _reconcile_failure_rows(examples, failure_rows)
    _reconcile_summary(examples, summary)
    return {
        "metadata": metadata,
        "summary": summary,
        "examples": examples,
        "raw": raw,
        "metrics": metric_rows,
        "failures": failure_rows,
        "family_ids": identifiers,
    }


def _validate_file_encoding(path):
    try:
        content = path.read_bytes()
        text = content.decode("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise EvaluationError("invalid_artifact_encoding", path.name) from exc
    if not content.endswith(b"\n") or content.endswith(b"\n\n"):
        raise EvaluationError("invalid_final_newline", path.name)
    if b"NaN" in content or b"Infinity" in content:
        raise EvaluationError("nonfinite_serialization", path.name)
    if ".tmp-" in text:
        raise EvaluationError("temporary_path_leakage", path.name)


def _strict_document(path):
    try:
        content = path.read_bytes()
        value = json.loads(
            content.decode("utf-8"), parse_constant=_reject_json_constant
        )
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise EvaluationError("invalid_json_artifact", path.name) from exc
    if content != _json_document(value):
        raise EvaluationError("noncanonical_json", path.name)
    return value


def _strict_lines(path):
    try:
        content = path.read_bytes()
        lines = content.decode("utf-8").splitlines()
        values = tuple(
            json.loads(line, parse_constant=_reject_json_constant)
            for line in lines
        )
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise EvaluationError("invalid_jsonl_artifact", path.name) from exc
    if content != _json_lines(values):
        raise EvaluationError("noncanonical_jsonl", path.name)
    return values


def _strict_csv(path, columns):
    try:
        content = path.read_bytes()
        with path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            rows = tuple(reader)
            field_names = tuple(reader.fieldnames or ())
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise EvaluationError("invalid_csv_artifact", path.name) from exc
    if field_names != tuple(columns):
        raise EvaluationError("invalid_csv_header", path.name)
    if any(set(row) != set(columns) or None in row for row in rows):
        raise EvaluationError("invalid_csv_row", path.name)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream, fieldnames=columns, lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(rows)
    if content != stream.getvalue().encode("utf-8"):
        raise EvaluationError("noncanonical_csv", path.name)
    return rows


def _validate_metadata_and_examples(
    metadata,
    summary,
    examples,
    raw,
    expected_ids,
    expected_partition,
    authoritative_validation_ids,
    authoritative_test_ids,
    reviewed_commit,
    expected_checkpoint_sha256,
):
    if metadata.get("evaluation_schema_version") != EVALUATION_SCHEMA_VERSION:
        raise EvaluationError("invalid_evaluation_schema", "metadata")
    if metadata.get("representation_schema_version") != REPRESENTATION_SCHEMA_VERSION:
        raise EvaluationError("invalid_representation_schema", "metadata")
    if summary.get("evaluation_schema_version") != EVALUATION_SCHEMA_VERSION:
        raise EvaluationError("invalid_evaluation_schema", "summary")
    partition = metadata.get("partition")
    if expected_partition is not None and partition != expected_partition:
        raise EvaluationError("partition_mismatch", "metadata partition differs")
    identifiers = tuple(item.get("family_id") for item in examples)
    declared = tuple(metadata.get("selected_family_ids", ()))
    expected = declared if expected_ids is None else tuple(expected_ids)
    if identifiers != declared or identifiers != expected:
        raise EvaluationError("family_identity_mismatch", "selected IDs differ")
    if len(identifiers) != len(set(identifiers)) or identifiers != tuple(sorted(identifiers)):
        raise EvaluationError("family_order_mismatch", "family IDs are not unique/sorted")
    if metadata.get("selected_family_count") != len(identifiers):
        raise EvaluationError("selected_count_mismatch", "metadata count differs")
    if summary.get("attempted_example_count") != len(identifiers):
        raise EvaluationError("summary_count_mismatch", "attempted count differs")
    if any(item.get("partition") != partition for item in examples):
        raise EvaluationError("example_partition_mismatch", "example partition differs")
    if any(any(path not in item for path in PATH_NAMES) for item in examples):
        raise EvaluationError("missing_path_record", "paired path is absent")
    if raw and tuple(item.get("family_id") for item in raw) != identifiers:
        raise EvaluationError("raw_family_mismatch", "raw records differ")
    if raw and any(any(path not in item for path in PATH_NAMES) for item in raw):
        raise EvaluationError("missing_raw_path", "raw paired path is absent")
    _validate_record_schemas(examples, raw, summary)
    _validate_authoritative_ids(
        metadata, identifiers, authoritative_validation_ids
    )
    test_ids = set(authoritative_test_ids)
    derived_test_evaluation = partition == "test" or bool(test_ids & set(identifiers))
    if partition == "validation" and derived_test_evaluation:
        raise EvaluationError("test_family_selected", "validation contains a test family")
    if metadata.get("test_partition_evaluated") is not derived_test_evaluation:
        raise EvaluationError("test_partition_semantics", "test evaluation flag differs")
    if LIMITATION not in metadata.get("limitation", ""):
        raise EvaluationError("missing_limitation", "supplied-count limitation is absent")
    if reviewed_commit is not None and metadata.get("repository_commit") != reviewed_commit:
        raise EvaluationError("repository_commit_mismatch", "reviewed commit differs")
    digest = metadata.get("checkpoint_sha256")
    if not _lower_sha256(digest):
        raise EvaluationError("invalid_checkpoint_digest", "checkpoint digest is invalid")
    if (
        expected_checkpoint_sha256 is not None
        and digest != expected_checkpoint_sha256
    ):
        raise EvaluationError(
            "checkpoint_digest_mismatch",
            "published checkpoint digest differs from reviewed checkpoint",
        )
    return identifiers


def _validate_record_schemas(examples, raw, summary):
    example_fields = {
        "family_id",
        "partition",
        "authoritative_metadata",
        "target_node_count",
        "target_operation_sequence",
        "latent_indices",
        *PATH_NAMES,
    }
    if any(set(item) != example_fields for item in examples):
        raise EvaluationError("invalid_example_schema", "example fields differ")
    if any(
        not isinstance(item[path], dict)
        or set(item[path]) != {"conversion", "metrics"}
        for item in examples
        for path in PATH_NAMES
    ):
        raise EvaluationError("invalid_example_schema", "path fields differ")
    if set(summary) != {
        "evaluation_schema_version",
        "attempted_example_count",
        *PATH_NAMES,
        "predicted_minus_teacher_forced",
    }:
        raise EvaluationError("invalid_summary_schema", "summary fields differ")
    raw_record_fields = {"family_id", *PATH_NAMES}
    if any(set(item) != raw_record_fields for item in raw):
        raise EvaluationError("invalid_raw_schema", "raw record fields differ")
    for item in raw:
        for path in PATH_NAMES:
            _validate_raw_prediction_schema(item[path])


def _validate_raw_prediction_schema(value):
    if value is None:
        return
    prediction_fields = {
        "latent_indices",
        "node_count",
        "node_count_source",
        "termination_reason",
        "termination_is_learned",
        "prefix_feedback",
        "raw_nodes",
        "raw_edges",
        "predicted_operation_node_indices",
        "predicted_operation_count",
        "operation_count_exceeds_limit",
        "raw_operation_pointers",
    }
    if not isinstance(value, dict) or not set(value) <= prediction_fields:
        raise EvaluationError("invalid_raw_schema", "prediction fields differ")
    nested = (
        (
            "raw_nodes",
            {
                "position",
                "node_type_id",
                "categorical_ids",
                "normalized_geometry",
                "derived_geometry_mask",
            },
        ),
        (
            "raw_edges",
            {
                "source_index",
                "target_index",
                "presence_logit",
                "present",
                "edge_type_id",
            },
        ),
        (
            "raw_operation_pointers",
            {"query_index", "selected_node_index", "selected_logit"},
        ),
    )
    for name, expected in nested:
        records = value.get(name)
        if isinstance(records, list) and any(
            not isinstance(record, dict) or not set(record) <= expected
            for record in records
        ):
            raise EvaluationError("invalid_raw_schema", name)


def _validate_authoritative_ids(metadata, selected, external_ids):
    authoritative = tuple(metadata.get("authoritative_validation_family_ids", ()))
    if metadata.get("authoritative_validation_family_ids_sha256") != (
        _identifier_sha256(authoritative)
    ):
        raise EvaluationError(
            "authoritative_validation_hash_mismatch",
            "validation-ID hash differs",
        )
    if external_ids is not None and authoritative != tuple(external_ids):
        raise EvaluationError(
            "authoritative_validation_mismatch",
            "metadata validation IDs differ from authoritative loader",
        )
    checkpoint_ids = tuple(
        metadata.get("checkpoint_data_state", {}).get(
            "validation_family_ids", ()
        )
    )
    if authoritative != checkpoint_ids:
        raise EvaluationError(
            "checkpoint_family_mismatch",
            "checkpoint validation IDs differ from authoritative loader",
        )
    limit = metadata.get("family_limit")
    expected = authoritative if limit is None else authoritative[:limit]
    if metadata.get("partition") == "validation" and selected != expected:
        raise EvaluationError(
            "selected_validation_mismatch",
            "selected IDs are not the authoritative validation prefix",
        )


def _reconcile_examples_with_authority(
    examples,
    raw_records,
    metadata,
    summary,
    authoritative_examples,
):
    for example in examples:
        for path in PATH_NAMES:
            _reconcile_stored_conversion_and_metrics(example[path], path)
    if authoritative_examples is None:
        return
    authoritative_examples = tuple(authoritative_examples)
    try:
        authoritative_by_id = {
            item.physical_family_id: item for item in authoritative_examples
        }
    except (AttributeError, TypeError) as exc:
        raise EvaluationError(
            "invalid_authoritative_examples",
            "authoritative examples are malformed",
        ) from exc
    identifiers = tuple(item["family_id"] for item in examples)
    if (
        len(authoritative_by_id) != len(authoritative_examples)
        or set(authoritative_by_id) != set(identifiers)
    ):
        raise EvaluationError(
            "authoritative_example_mismatch",
            "authoritative example identities differ",
        )
    raw_by_id = {
        item.get("family_id"): item for item in raw_records
    }
    if raw_records and (
        len(raw_by_id) != len(raw_records)
        or set(raw_by_id) != set(identifiers)
    ):
        raise EvaluationError("raw_family_mismatch", "raw records differ")
    configuration = metadata.get("model_configuration")
    max_operations = (
        configuration.get("max_operations")
        if isinstance(configuration, dict)
        else None
    )
    if (
        raw_records
        and (
            isinstance(max_operations, bool)
            or not isinstance(max_operations, int)
            or max_operations < 1
        )
    ):
        raise EvaluationError(
            "invalid_metadata",
            "model max_operations is invalid",
        )
    recomputed_metrics = {path: [] for path in PATH_NAMES}
    for stored in examples:
        family_id = stored["family_id"]
        authoritative = authoritative_by_id[family_id]
        if (
            stored.get("partition") != authoritative.partition
            or stored.get("authoritative_metadata")
            != _json_safe(authoritative.metadata)
            or stored.get("target_node_count")
            != len(authoritative.target.node_type_ids)
            or stored.get("target_operation_sequence")
            != list(authoritative.operation_sequence)
        ):
            raise EvaluationError(
                "authoritative_example_mismatch",
                family_id,
            )
        if not raw_records:
            continue
        raw_record = raw_by_id[family_id]
        latent_indices = stored.get("latent_indices")
        if not isinstance(latent_indices, list):
            raise EvaluationError("latent_index_mismatch", family_id)
        for path in PATH_NAMES:
            raw = _raw_from_json(raw_record[path])
            raw_latent = getattr(raw, "latent_indices", None)
            if isinstance(raw_latent, tuple) and raw_latent != tuple(latent_indices):
                raise EvaluationError(
                    "latent_index_mismatch",
                    family_id + "." + path,
                )
            stage_two_raw = _stage_two_prediction(raw, path)
            conversion = validate_and_convert_raw_prediction(
                stage_two_raw,
                max_operations=max_operations,
            )
            metric = evaluate_prediction(
                stage_two_raw,
                authoritative.target,
                family_id=family_id,
                max_operations=max_operations,
            )
            recomputed_metrics[path].append(metric)
            expected_path = {
                "conversion": _json_safe(conversion),
                "metrics": _json_safe(metric),
            }
            if stored[path] != expected_path:
                raise EvaluationError(
                    "raw_example_mismatch",
                    family_id + "." + path,
                )
    if raw_records:
        recomputed = {
            path: _aggregate_json(aggregate_prediction_metrics(
                tuple(recomputed_metrics[path]),
                metadata_by_family=authoritative_by_id,
            ))
            for path in PATH_NAMES
        }
        if any(summary.get(path) != recomputed[path] for path in PATH_NAMES):
            raise EvaluationError(
                "raw_summary_mismatch",
                "summary does not match recomputed raw metrics",
            )
        expected_gap = _semantic_difference(
            recomputed["predicted_history"],
            recomputed["teacher_forced"],
        )
        if summary.get("predicted_minus_teacher_forced") != expected_gap:
            raise EvaluationError(
                "raw_summary_mismatch",
                "metric gap does not match recomputed raw metrics",
            )


def _reconcile_stored_conversion_and_metrics(path_record, path):
    if not isinstance(path_record, dict):
        raise EvaluationError("invalid_example_path", path)
    conversion = path_record.get("conversion")
    metric = path_record.get("metrics")
    if not isinstance(conversion, dict) or not isinstance(metric, dict):
        raise EvaluationError("invalid_example_path", path)
    pairs = (
        ("raw_integrity", "raw_integrity"),
        ("reconstruction_target", "reconstruction_target"),
        ("controlled_domain", "controlled_domain"),
        ("primary_failure", "primary_failure"),
        ("secondary_failures", "secondary_failures"),
    )
    if any(conversion.get(left) != metric.get(right) for left, right in pairs):
        raise EvaluationError(
            "conversion_metric_mismatch",
            path,
        )


def _raw_from_json(value):
    if isinstance(value, list):
        return tuple(_raw_from_json(item) for item in value)
    if isinstance(value, dict):
        if set(value) == {"nonfinite_float"}:
            label = value["nonfinite_float"]
            if label == "nan":
                return float("nan")
            if label == "positive_infinity":
                return float("inf")
            if label == "negative_infinity":
                return float("-inf")
            raise EvaluationError(
                "invalid_nonfinite_tag",
                str(label),
            )
        return SimpleNamespace(
            **{name: _raw_from_json(item) for name, item in value.items()}
        )
    return value


def _validate_metric_rows(summary, rows):
    expected = []
    aggregate_by_key = {}
    for path in PATH_NAMES:
        aggregate = summary.get(path)
        if not isinstance(aggregate, dict):
            raise EvaluationError("missing_summary_path", path)
        groups = (
            ("overall", {"": aggregate.get("overall")}),
            ("operation_template", aggregate.get("by_operation_template")),
            ("primitive_family", aggregate.get("by_primitive_family")),
            (
                "target_operation_type_sequence",
                aggregate.get("by_target_operation_type_sequence"),
            ),
        )
        for kind, values in groups:
            if not isinstance(values, dict):
                raise EvaluationError("invalid_summary_strata", kind)
            for value in sorted(values):
                key = (path, kind, value)
                expected.append(key)
                aggregate_by_key[key] = values[value]
    actual = tuple(
        (row["path"], row["stratum_kind"], row["stratum_value"])
        for row in rows
    )
    if actual != tuple(expected) or len(actual) != len(set(actual)):
        raise EvaluationError("metrics_row_mismatch", "metric rows/ordering differ")
    for row in rows:
        key = (row["path"], row["stratum_kind"], row["stratum_value"])
        expected_row = _metric_json_row(aggregate_by_key[key])
        for column in METRICS_COLUMNS[3:]:
            if row[column] != _csv_scalar(expected_row[column]):
                raise EvaluationError("metrics_value_mismatch", ".".join(key + (column,)))


def _metric_json_row(metric):
    validity = metric["validity"]
    return {
        "attempted_example_count": metric["attempted_example_count"],
        "raw_completion_rate": validity["raw_completion_rate"],
        "raw_integrity_valid_rate": validity["raw_integrity_valid_rate"],
        "reconstruction_target_valid_rate": validity["reconstruction_target_valid_rate"],
        "controlled_domain_valid_rate": validity["controlled_domain_valid_rate"],
        "node_type_token_accuracy": metric["nodes"]["node_type_token_accuracy"],
        "exact_node_type_sequence_rate": metric["nodes"]["exact_node_type_sequence_rate"],
        "finite_geometry_mae": metric["geometry"]["finite_applicable_geometry_mae"],
        "finite_geometry_rmse": metric["geometry"]["finite_applicable_geometry_rmse"],
        "complete_finite_geometry_rate": metric["geometry"]["complete_finite_example_rate"],
        "exact_operation_type_sequence_rate": metric["operations"]["exact_operation_type_sequence_rate"],
        "pointer_overall_accuracy": metric["pointers"]["overall_pointer_accuracy"],
        "edge_micro_precision": metric["edges"]["micro_precision"],
        "edge_micro_recall": metric["edges"]["micro_recall"],
        "edge_micro_f1": metric["edges"]["micro_f1"],
    }


def _reconcile_failure_rows(examples, rows):
    expected = []
    for example in examples:
        for path in PATH_NAMES:
            conversion = example[path]["conversion"]
            failures = []
            primary = conversion["primary_failure"]
            if primary is not None:
                failures.append(("primary", primary))
            failures.extend(
                ("secondary", item)
                for item in conversion["secondary_failures"]
            )
            for designation, failure in failures:
                expected.append({
                    "family_id": example["family_id"],
                    "path": path,
                    "designation": designation,
                    "code": failure["code"],
                    "layer": failure["layer"],
                    "stage": failure["stage"],
                    "location": _location_text(failure["location"]),
                    "detail": failure["detail"],
                })
    if list(rows) != expected:
        raise EvaluationError(
            "conversion_failure_mismatch",
            "conversion failure records do not reconcile",
        )


def _reconcile_summary(examples, summary):
    count = len(examples)
    for path in PATH_NAMES:
        records = [item[path] for item in examples]
        validity = summary[path]["overall"]["validity"]
        expected_counts = {
            "attempted_example_count": count,
            "raw_completion_count": sum(
                bool(item["metrics"]["raw_completion"]) for item in records
            ),
            "raw_integrity_valid_count": sum(
                bool(item["conversion"]["raw_integrity"]["valid"]) for item in records
            ),
            "reconstruction_target_valid_count": sum(
                bool(item["conversion"]["reconstruction_target"]["valid"])
                for item in records
            ),
            "controlled_domain_valid_count": sum(
                bool(item["conversion"]["controlled_domain"]["valid"])
                for item in records
            ),
        }
        for field, expected in expected_counts.items():
            if validity.get(field) != expected:
                raise EvaluationError("summary_validity_mismatch", path + "." + field)
        rate_pairs = (
            ("raw_completion_count", "raw_completion_rate"),
            ("raw_integrity_valid_count", "raw_integrity_valid_rate"),
            (
                "reconstruction_target_valid_count",
                "reconstruction_target_valid_rate",
            ),
            ("controlled_domain_valid_count", "controlled_domain_valid_rate"),
        )
        for count_field, rate_field in rate_pairs:
            expected_rate = (
                None
                if count == 0
                else expected_counts[count_field] / count
            )
            if validity.get(rate_field) != expected_rate:
                raise EvaluationError(
                    "summary_validity_mismatch", path + "." + rate_field
                )
        primary, any_failure = _failure_counters(records)
        _reconcile_failure_aggregates(
            validity.get("primary_failure_counts"), primary, path, "primary"
        )
        _reconcile_failure_aggregates(
            validity.get("any_failure_counts"), any_failure, path, "any"
        )
        _reconcile_stratum_counts(examples, summary[path], path)


def _reconcile_stratum_counts(examples, aggregate, path):
    specifications = (
        (
            "by_operation_template",
            lambda item: item["authoritative_metadata"]["operation_template"],
        ),
        (
            "by_primitive_family",
            lambda item: item["authoritative_metadata"]["primitive_family"],
        ),
        (
            "by_target_operation_type_sequence",
            lambda item: ",".join(
                item[path]["metrics"]["operations"][
                    "target_operation_type_sequence"
                ]
            ),
        ),
    )
    for name, key_function in specifications:
        expected = Counter(key_function(item) for item in examples)
        observed = aggregate.get(name)
        if not isinstance(observed, dict) or set(observed) != set(expected):
            raise EvaluationError("summary_stratum_mismatch", path + "." + name)
        for key, expected_count in expected.items():
            if observed[key].get("attempted_example_count") != expected_count:
                raise EvaluationError(
                    "summary_stratum_mismatch",
                    path + "." + name + "." + key,
                )


def _failure_counters(records):
    primary = Counter()
    any_failure = Counter()
    for item in records:
        conversion = item["conversion"]
        first = conversion["primary_failure"]
        codes = set()
        if first is not None:
            primary[first["code"]] += 1
            codes.add(first["code"])
        codes.update(
            failure["code"] for failure in conversion["secondary_failures"]
        )
        any_failure.update(codes)
    return primary, any_failure


def _reconcile_failure_aggregates(values, expected, path, designation):
    if not isinstance(values, list):
        raise EvaluationError("summary_failure_mismatch", path + "." + designation)
    observed_codes = [item.get("code") for item in values]
    if len(observed_codes) != len(set(observed_codes)):
        raise EvaluationError("summary_failure_mismatch", "duplicate failure code")
    for item in values:
        if item.get("count") != expected[item.get("code")]:
            raise EvaluationError(
                "summary_failure_mismatch",
                path + "." + designation + "." + str(item.get("code")),
            )
    if any(code not in observed_codes for code in expected):
        raise EvaluationError("summary_failure_mismatch", "failure code is absent")


def _csv_scalar(value):
    return "" if value is None else str(value)


def _lower_sha256(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _identifier_sha256(identifiers):
    return hashlib.sha256(
        ("\n".join(identifiers) + "\n").encode("utf-8")
    ).hexdigest()


def checkpoint_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def corpus_identity(corpus_dir, split_manifest):
    root = Path(corpus_dir)
    corpus_path = root / "corpus_manifest.json"
    split_path = root / "manifests" / (split_manifest + ".json")
    corpus = _read_strict_json(corpus_path)
    split = _read_strict_json(split_path)
    configuration = corpus.get("normalized_configuration")
    declared_hash = corpus.get("configuration_sha256")
    declared_schema = corpus.get("representation_schema_version")
    split_hash = split.get("corpus_configuration_sha256")
    identity_values = (configuration, declared_hash, declared_schema, split_hash)
    if all(value is None for value in identity_values):
        return {
            "corpus_configuration_sha256": None,
            "corpus_manifest_sha256": checkpoint_sha256(corpus_path),
            "split_manifest_sha256": checkpoint_sha256(split_path),
            "corpus_identity_validation": "unavailable_in_fixture_manifest",
        }
    if (
        not isinstance(configuration, dict)
        or not _lower_sha256(declared_hash)
        or declared_schema != REPRESENTATION_SCHEMA_VERSION
        or split_hash != declared_hash
    ):
        raise EvaluationError("invalid_corpus_identity", "identity fields are inconsistent")
    calculated_hash = hashlib.sha256(
        json.dumps(
            configuration,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    if calculated_hash != declared_hash:
        raise EvaluationError("invalid_corpus_identity", "configuration hash differs")
    return {
        "corpus_configuration_sha256": declared_hash,
        "corpus_manifest_sha256": checkpoint_sha256(corpus_path),
        "split_manifest_sha256": checkpoint_sha256(split_path),
        "corpus_identity_validation": "validated",
    }


def load_and_validate_checkpoint(
    path, torch_module, authoritative_validation_ids, split
):
    digest = checkpoint_sha256(path)
    try:
        checkpoint = torch_module.load(str(path), map_location="cpu")
        _validate_payload(checkpoint)
        model_config = FlatBaselineConfig(**checkpoint["model_config"])
        training_config = TrainingConfig(**checkpoint["training_config"])
        model_config.validate()
        training_config.validate()
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        raise EvaluationError("invalid_checkpoint", type(exc).__name__) from exc
    _validate_checkpoint_data_state(
        checkpoint["data_state"], authoritative_validation_ids, split
    )
    return digest, checkpoint, model_config, training_config


def run(
    arguments,
    torch_module=None,
    *,
    model_factory=None,
    paired_decoder=None,
    source_state_provider=None,
):
    output = Path(arguments.output_dir)
    _preflight_output(output)
    if torch_module is None:
        try:
            import torch as torch_module
        except ImportError as exc:
            raise EvaluationError("pytorch_unavailable", "PyTorch is required") from exc
    identity = corpus_identity(arguments.corpus_dir, arguments.split_manifest)
    examples_all = tuple(load_physical_examples(
        arguments.corpus_dir, arguments.split_manifest
    ))
    validation_ids = _partition_ids(examples_all, "validation")
    test_ids = _partition_ids(examples_all, "test")
    digest, checkpoint, model_config, training_config = (
        load_and_validate_checkpoint(
            arguments.checkpoint,
            torch_module,
            validation_ids,
            arguments.split_manifest,
        )
    )
    selected = select_examples(
        examples_all, arguments.partition, arguments.family_limit
    )
    if arguments.partition == "validation":
        expected_selected = (
            validation_ids
            if arguments.family_limit is None
            else validation_ids[:arguments.family_limit]
        )
        if tuple(item.physical_family_id for item in selected) != expected_selected:
            raise EvaluationError(
                "selected_validation_mismatch",
                "selection is not the authoritative validation prefix",
            )
    device = _resolve_device(arguments.device, torch_module)
    if paired_decoder is None:
        from .autonomous import decode_paired_from_batch as paired_decoder
    if model_factory is None:
        from .model import FlatMixedVQModel as model_factory

    model = model_factory(model_config).to(device)
    _strict_load_model(model, checkpoint["model_state"])

    def decode_batch(items):
        batch = collate_flat(tuple(adapt_flat_mixed(item) for item in items))
        inputs = _mapping_to_device(batch.to_torch(torch_module), device)
        target = _mapping_to_device(batch.target.to_torch(torch_module), device)
        counts = tuple(len(item.target.node_type_ids) for item in items)
        return paired_decoder(
            model,
            (inputs, target),
            node_counts=counts,
            node_count_source="target_canonical_metadata",
        )

    records, raw_records = evaluate_records(
        selected,
        batch_size=arguments.batch_size,
        max_operations=model_config.max_operations,
        decode_batch=decode_batch,
    )
    metadata = _run_metadata(
        arguments,
        selected,
        digest,
        checkpoint,
        model_config,
        training_config,
        device,
        validation_ids,
        test_ids,
        source_state_provider,
    )
    metadata.update(identity)
    artifacts = make_artifacts(
        records,
        raw_records,
        metadata,
        selected,
        arguments.write_raw_predictions,
    )
    publish_artifacts(
        output,
        artifacts,
        expected_ids=tuple(item.physical_family_id for item in selected),
        expected_partition=arguments.partition,
        authoritative_validation_ids=validation_ids,
        authoritative_test_ids=test_ids,
        authoritative_examples=selected,
        reviewed_commit=metadata["repository_commit"],
        expected_checkpoint_sha256=digest,
    )
    return completed_exit_code(records)


def main(argv=None):
    try:
        return run(parse_arguments(argv))
    except EvaluationError as exc:
        sys.stderr.write("{}: {}\n".format(exc.code, exc.detail))
        return 1
    except Exception as exc:
        sys.stderr.write("internal_error: {}\n".format(type(exc).__name__))
        return 1


def _run_metadata(
    arguments,
    selected,
    checkpoint_digest,
    checkpoint,
    model_config,
    training_config,
    device,
    authoritative_validation_ids,
    authoritative_test_ids,
    source_state_provider,
):
    repository = Path(__file__).resolve().parents[2]
    state = (
        source_state(repository)
        if source_state_provider is None
        else source_state_provider(repository)
    )
    selected_ids = tuple(item.physical_family_id for item in selected)
    test_partition_evaluated = (
        arguments.partition == "test"
        or bool(set(selected_ids) & set(authoritative_test_ids))
    )
    return {
        "evaluation_schema_version": EVALUATION_SCHEMA_VERSION,
        "representation_schema_version": REPRESENTATION_SCHEMA_VERSION,
        "repository_branch": _git_value(repository, "branch", "--show-current"),
        "repository_commit": state["git_commit"],
        "source_tree_sha256": state["source_tree_sha256"],
        "source_dirty": state["git_dirty"],
        "dirty_state_policy": "recorded; official Slurm workflow requires clean",
        "checkpoint_path": str(Path(arguments.checkpoint).resolve()),
        "checkpoint_sha256": checkpoint_digest,
        "checkpoint_kind": checkpoint["checkpoint_kind"],
        "checkpoint_epoch": checkpoint["epoch"],
        "checkpoint_global_step": checkpoint["global_step"],
        "model_configuration": model_config.to_dict(),
        "training_configuration": training_config.to_dict(),
        "checkpoint_data_state": checkpoint["data_state"],
        "checkpoint_original_repository_commit": {"status": "unavailable"},
        "checkpoint_original_corpus_hashes": {"status": "unavailable"},
        "authoritative_loader_success": True,
        "split_manifest": arguments.split_manifest,
        "partition": arguments.partition,
        "authoritative_validation_family_ids": list(
            authoritative_validation_ids
        ),
        "authoritative_validation_family_ids_sha256": _identifier_sha256(
            authoritative_validation_ids
        ),
        "selected_family_ids": list(selected_ids),
        "selected_family_count": len(selected),
        "family_limit": arguments.family_limit,
        "batch_size": arguments.batch_size,
        "requested_device": arguments.device,
        "resolved_device": str(device),
        "supplied_node_count_policy": "target_canonical_metadata",
        "teacher_forced_prefix_feedback": "shifted_target_prefix",
        "stage_two_teacher_forced_provenance_adapter": (
            "prefix_feedback_only; raw artifact remains shifted_target_prefix"
        ),
        "predicted_history_prefix_feedback": "raw_argmax_with_derived_geometry_mask",
        "edge_presence_decision_threshold": 0.0,
        "raw_predictions_published": arguments.write_raw_predictions,
        "test_partition_evaluated": test_partition_evaluated,
        "limitation": LIMITATION,
    }


def _validate_checkpoint_data_state(
    data_state, authoritative_validation_ids, split
):
    if not isinstance(data_state, dict):
        raise EvaluationError("invalid_checkpoint_data_state", "data_state is not a map")
    if data_state.get("split_manifest") != split:
        raise EvaluationError("checkpoint_split_mismatch", "split manifest differs")
    if data_state.get("validation_partition") != "validation":
        raise EvaluationError("checkpoint_partition_mismatch", "validation")
    checkpoint_ids = data_state.get("validation_family_ids")
    if (
        not isinstance(checkpoint_ids, list)
        or tuple(checkpoint_ids) != tuple(authoritative_validation_ids)
    ):
        raise EvaluationError("checkpoint_family_mismatch", "validation")


def _aggregate_json(value):
    raw = _json_safe(value)
    return {
        "overall": raw["overall"],
        "by_operation_template": {
            name: metrics for name, metrics in raw["by_operation_template"]
        },
        "by_primitive_family": {
            name: metrics for name, metrics in raw["by_primitive_family"]
        },
        "by_target_operation_type_sequence": {
            name: metrics
            for name, metrics in raw["by_target_operation_type_sequence"]
        },
    }


def _example_json(record):
    value = {
        "family_id": record.family_id,
        "partition": record.partition,
        "authoritative_metadata": record.metadata,
        "target_node_count": record.target_node_count,
        "target_operation_sequence": record.target_operation_sequence,
        "latent_indices": record.latent_indices,
    }
    for path in PATH_NAMES:
        path_record = getattr(record, path)
        value[path] = {
            "conversion": path_record.conversion,
            "metrics": path_record.metrics,
        }
    return _json_safe(value)


def _metrics_csv(aggregates):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=METRICS_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for path in PATH_NAMES:
        aggregate = aggregates[path]
        groups = (
            ("overall", "", (("", aggregate.overall),)),
            ("operation_template", "", aggregate.by_operation_template),
            ("primitive_family", "", aggregate.by_primitive_family),
            (
                "target_operation_type_sequence",
                "",
                aggregate.by_target_operation_type_sequence,
            ),
        )
        for kind, _, rows in groups:
            for value, metric in sorted(rows, key=lambda item: item[0]):
                writer.writerow(_metric_row(path, kind, value, metric))
    return stream.getvalue().encode("utf-8")


def _metric_row(path, kind, value, metric):
    return {
        "path": path,
        "stratum_kind": kind,
        "stratum_value": value,
        "attempted_example_count": metric.attempted_example_count,
        "raw_completion_rate": metric.validity.raw_completion_rate,
        "raw_integrity_valid_rate": metric.validity.raw_integrity_valid_rate,
        "reconstruction_target_valid_rate": metric.validity.reconstruction_target_valid_rate,
        "controlled_domain_valid_rate": metric.validity.controlled_domain_valid_rate,
        "node_type_token_accuracy": metric.nodes.node_type_token_accuracy,
        "exact_node_type_sequence_rate": metric.nodes.exact_node_type_sequence_rate,
        "finite_geometry_mae": metric.geometry.finite_applicable_geometry_mae,
        "finite_geometry_rmse": metric.geometry.finite_applicable_geometry_rmse,
        "complete_finite_geometry_rate": metric.geometry.complete_finite_example_rate,
        "exact_operation_type_sequence_rate": metric.operations.exact_operation_type_sequence_rate,
        "pointer_overall_accuracy": metric.pointers.overall_pointer_accuracy,
        "edge_micro_precision": metric.edges.micro_precision,
        "edge_micro_recall": metric.edges.micro_recall,
        "edge_micro_f1": metric.edges.micro_f1,
    }


def _failures_csv(records):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=FAILURE_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for record in records:
        for path in PATH_NAMES:
            conversion = getattr(record, path).conversion
            failures = ()
            if conversion.primary_failure is not None:
                failures += (("primary", conversion.primary_failure),)
            failures += tuple(
                ("secondary", failure)
                for failure in conversion.secondary_failures
            )
            for designation, failure in failures:
                writer.writerow({
                    "family_id": record.family_id,
                    "path": path,
                    "designation": designation,
                    "code": failure.code,
                    "layer": failure.layer,
                    "stage": failure.stage,
                    "location": _location_text(failure.location),
                    "detail": failure.detail,
                })
    return stream.getvalue().encode("utf-8")


def _json_safe(value):
    if is_dataclass(value):
        return {
            field.name: _json_safe(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, dict):
        return {
            str(key): _json_safe(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise EvaluationError("nonfinite_serialization", "nonfinite JSON value")
        return value
    raise EvaluationError(
        "unsupported_serialization_type", type(value).__name__
    )


def _raw_json_safe(value):
    """Encode IEEE nonfinite raw evidence without emitting forbidden JSON tokens."""

    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            label = "nan"
        elif value > 0:
            label = "positive_infinity"
        else:
            label = "negative_infinity"
        return {"nonfinite_float": label}
    if is_dataclass(value):
        return {
            field.name: _raw_json_safe(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, dict):
        return {
            str(key): _raw_json_safe(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_raw_json_safe(item) for item in value]
    return _json_safe(value)


def _semantic_difference(predicted, teacher):
    if isinstance(predicted, dict) and isinstance(teacher, dict):
        result = {}
        for key in sorted(set(predicted) | set(teacher)):
            if key in predicted and key in teacher:
                result[key] = _semantic_difference(predicted[key], teacher[key])
            else:
                result[key] = {
                    "predicted_history": predicted.get(key),
                    "teacher_forced": teacher.get(key),
                }
        return result
    if isinstance(predicted, list) and isinstance(teacher, list):
        semantic_key = _repeated_record_key(predicted, teacher)
        if semantic_key is not None:
            return _keyed_record_difference(
                predicted, teacher, semantic_key
            )
        if predicted == teacher:
            return predicted
        return {
            "predicted_history": predicted,
            "teacher_forced": teacher,
        }
    if (
        isinstance(predicted, (int, float))
        and not isinstance(predicted, bool)
        and isinstance(teacher, (int, float))
        and not isinstance(teacher, bool)
    ):
        return predicted - teacher
    if predicted == teacher:
        return predicted
    return {
        "predicted_history": predicted,
        "teacher_forced": teacher,
    }


def _repeated_record_key(left, right):
    records = left + right
    if not records or not all(isinstance(item, dict) for item in records):
        return None
    for key in ("field_name", "code"):
        if all(isinstance(item.get(key), str) for item in records):
            return key
    return None


def _keyed_record_difference(predicted, teacher, semantic_key):
    predicted_by_key = {item[semantic_key]: item for item in predicted}
    teacher_by_key = {item[semantic_key]: item for item in teacher}
    if len(predicted_by_key) != len(predicted) or len(teacher_by_key) != len(teacher):
        raise EvaluationError(
            "ambiguous_gap_records",
            "repeated {} values".format(semantic_key),
        )
    result = []
    for key in sorted(set(predicted_by_key) | set(teacher_by_key)):
        if key in predicted_by_key and key in teacher_by_key:
            difference = _semantic_difference(
                predicted_by_key[key], teacher_by_key[key]
            )
            difference[semantic_key] = key
        else:
            difference = {
                semantic_key: key,
                "predicted_history": predicted_by_key.get(key),
                "teacher_forced": teacher_by_key.get(key),
            }
        result.append(difference)
    return result


def _json_document(value):
    return (_json_text(value) + "\n").encode("utf-8")


def _json_lines(values):
    return ("".join(_json_text(value) + "\n" for value in values)).encode("utf-8")


def _json_text(value):
    return json.dumps(
        _json_safe(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _read_strict_json(path):
    try:
        return json.loads(
            Path(path).read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
        )
    except (OSError, ValueError) as exc:
        raise EvaluationError("invalid_json_input", Path(path).name) from exc


def _reject_json_constant(value):
    raise ValueError("nonfinite JSON constant")


def _location_text(location):
    if location is None:
        return ""
    if isinstance(location, tuple):
        return ".".join(str(item) for item in location)
    return str(location)


def _positive_integer(value):
    try:
        integer = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if str(integer) != value or integer <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return integer


def _device_request(value):
    if value == "cpu" or value == "cuda":
        return value
    if value.startswith("cuda:") and value[5:].isdigit():
        return value
    raise argparse.ArgumentTypeError("device must be cpu, cuda, or cuda:N")


def _resolve_device(request, torch_module):
    if request.startswith("cuda") and not torch_module.cuda.is_available():
        raise EvaluationError("cuda_unavailable", "requested CUDA is unavailable")
    try:
        device = torch_module.device(request)
    except (TypeError, RuntimeError) as exc:
        raise EvaluationError("unsupported_device", request) from exc
    return device


def _mapping_to_device(mapping, device):
    return {name: value.to(device) for name, value in mapping.items()}


def publish_json_report(path, value):
    destination = Path(path)
    _preflight_output(destination)
    temporary = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix="." + destination.name + ".tmp-",
            dir=str(destination.parent),
        )
        temporary = Path(temporary_name)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(_json_document(value))
            stream.flush()
            os.fsync(stream.fileno())
        _strict_document(temporary)
        _atomic_no_replace(temporary, destination)
        temporary = None
        _fsync_directory(destination.parent)
    except EvaluationError:
        raise
    except (OSError, TypeError, ValueError) as exc:
        raise EvaluationError("report_publication_failure", type(exc).__name__) from exc
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _atomic_no_replace(source, destination):
    """Atomically rename without replacement; never fall back to os.rename."""

    if sys.platform.startswith("linux"):
        _linux_rename_noreplace(source, destination)
        return
    if sys.platform == "darwin":
        _darwin_rename_noreplace(source, destination)
        return
    raise EvaluationError(
        "unsupported_no_replace",
        "no supported atomic no-replace rename primitive",
    )


def _linux_rename_noreplace(source, destination):
    library = ctypes.CDLL(None, use_errno=True)
    function = getattr(library, "renameat2", None)
    if function is None:
        raise EvaluationError("unsupported_no_replace", "renameat2 is unavailable")
    function.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    function.restype = ctypes.c_int
    result = function(
        -100,
        os.fsencode(str(source)),
        -100,
        os.fsencode(str(destination)),
        1,
    )
    if result != 0:
        _raise_rename_error(ctypes.get_errno())


def _darwin_rename_noreplace(source, destination):
    library = ctypes.CDLL(None, use_errno=True)
    function = getattr(library, "renamex_np", None)
    if function is None:
        raise EvaluationError("unsupported_no_replace", "renamex_np is unavailable")
    function.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint)
    function.restype = ctypes.c_int
    result = function(
        os.fsencode(str(source)),
        os.fsencode(str(destination)),
        4,
    )
    if result != 0:
        _raise_rename_error(ctypes.get_errno())


def _raise_rename_error(error_number):
    if error_number in (errno.EEXIST, errno.ENOTEMPTY):
        raise EvaluationError("output_collision", "destination already exists")
    if error_number in (errno.ENOSYS, errno.EINVAL, errno.ENOTSUP):
        raise EvaluationError(
            "unsupported_no_replace",
            "atomic no-replace rename is unsupported",
        )
    raise EvaluationError(
        "publication_failure",
        "atomic rename failed with errno {}".format(error_number),
    )


def _fsync_directory(path):
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(str(path), flags)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError as exc:
        if exc.errno not in (errno.EINVAL, errno.ENOTSUP):
            raise


def _preflight_output(destination):
    if destination.exists():
        raise EvaluationError("output_collision", "output directory already exists")
    if not destination.parent.is_dir():
        raise EvaluationError("invalid_output_parent", "output parent is not a directory")


def _strict_load_model(model, state):
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise EvaluationError("incompatible_model_state", "strict load failed") from exc


def _git_value(repository, *arguments):
    try:
        result = subprocess.run(
            ("git",) + arguments,
            cwd=str(repository),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            universal_newlines=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def _remove_temporary_directory(path):
    for child in path.iterdir():
        if child.is_file():
            child.unlink()
    path.rmdir()


if __name__ == "__main__":
    sys.exit(main())
