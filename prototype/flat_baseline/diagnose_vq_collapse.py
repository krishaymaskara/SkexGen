"""Deterministic, train/validation-only diagnosis of flat-baseline VQ collapse."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile

from prototype.model_data.loader import load_physical_examples

from .evaluate_length_conditioned import (
    EvaluationError,
    _atomic_no_replace,
    _fsync_directory,
    _identifier_sha256,
    _json_document,
    _preflight_output,
    _remove_temporary_directory,
    _resolve_publication_directory,
    checkpoint_sha256,
    corpus_identity,
    publish_json_report,
)
from .provenance import source_state


DIAGNOSTIC_SCHEMA_VERSION = 1
REQUIRED_ARTIFACTS = (
    "run_metadata.json",
    "checkpoint_inventory.json",
    "partition_summary.json",
    "latent_usage.csv",
    "prequant_summary.json",
    "codebook_summary.json",
    "decoder_sensitivity.json",
    "loss_masking_audit.json",
    "gradient_probe.json",
    "root_cause_report.json",
)
MANIFEST_ARTIFACT = "artifact_manifest.json"
LATENT_USAGE_COLUMNS = (
    "checkpoint_sha256",
    "checkpoint_relative_path",
    "partition",
    "code_index",
    "assignment_count",
    "assignment_fraction",
)
HYPOTHESES = (
    "encoder-output collapse",
    "nearest-code assignment collapse",
    "codebook-embedding collapse",
    "decoder ignores latent",
    "loss/mask imbalance",
    "gradient-flow failure",
    "training-time collapse",
    "unresolved or combined mechanisms",
)


class DiagnosisError(RuntimeError):
    """A diagnosis contract failed without changing training state."""

    def __init__(self, code, detail):
        self.code = code
        self.detail = detail
        super().__init__("{}: {}".format(code, detail))


def partition_reconciliation(examples, checkpoint_data_state=None):
    """Reconcile the complete authoritative corpus before any limit."""

    partitions = {}
    all_ids = []
    for item in examples:
        family_id = item.physical_family_id
        partition = item.partition
        if not isinstance(family_id, str) or not family_id:
            raise DiagnosisError("invalid_family_id", "family ID is invalid")
        if partition not in ("train", "validation", "test"):
            raise DiagnosisError(
                "invalid_partition", "unexpected partition {!r}".format(partition)
            )
        partitions.setdefault(partition, []).append(family_id)
        all_ids.append(family_id)
    if len(all_ids) != len(set(all_ids)):
        raise DiagnosisError("duplicate_family", "family IDs are not unique")
    ordered = {
        name: tuple(sorted(partitions.get(name, ())))
        for name in ("train", "validation", "test")
    }
    checkpoint_matches = None
    checkpoint_ids = None
    if checkpoint_data_state is not None:
        if not isinstance(checkpoint_data_state, dict):
            checkpoint_matches = False
        else:
            checkpoint_ids = {
                "train": tuple(checkpoint_data_state.get("train_family_ids", ())),
                "validation": tuple(
                    checkpoint_data_state.get("validation_family_ids", ())
                ),
            }
            checkpoint_matches = (
                checkpoint_data_state.get("train_partition") == "train"
                and checkpoint_data_state.get("validation_partition")
                == "validation"
                and checkpoint_ids["train"] == ordered["train"]
                and checkpoint_ids["validation"] == ordered["validation"]
            )
    return {
        "total_physical_family_count": len(all_ids),
        "partition_counts": {
            name: len(ordered[name]) for name in ordered
        },
        "partition_id_sha256": {
            name: _identifier_sha256(ordered[name]) for name in ordered
        },
        "checkpoint_partition_ids": (
            None
            if checkpoint_ids is None
            else {name: list(values) for name, values in checkpoint_ids.items()}
        ),
        "checkpoint_partition_ids_match": checkpoint_matches,
        "test_partition_evaluated": False,
        "_ids": ordered,
    }


def select_permitted_ids(reconciliation, train_limit=None, validation_limit=None):
    selected = {}
    for name, limit in (
        ("train", train_limit),
        ("validation", validation_limit),
    ):
        values = reconciliation["_ids"][name]
        if limit is not None:
            if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
                raise DiagnosisError(
                    "invalid_limit", "{} limit must be positive".format(name)
                )
            values = values[:limit]
        selected[name] = values
    test_ids = set(reconciliation["_ids"]["test"])
    if test_ids & (set(selected["train"]) | set(selected["validation"])):
        raise DiagnosisError(
            "test_partition_leakage", "test family selected for diagnosis"
        )
    return selected


def latent_statistics(assignments, codebook_size):
    """Return reconciled entropy, perplexity, and collapse statistics."""

    if isinstance(codebook_size, bool) or not isinstance(codebook_size, int):
        raise DiagnosisError("invalid_codebook_size", "codebook size is invalid")
    if codebook_size <= 0:
        raise DiagnosisError("invalid_codebook_size", "codebook size is invalid")
    values = tuple(assignments)
    if any(
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        or value >= codebook_size
        for value in values
    ):
        raise DiagnosisError("invalid_assignment", "assignment is out of range")
    histogram = [0] * codebook_size
    for value in values:
        histogram[value] += 1
    total = len(values)
    probabilities = [
        count / float(total) if total else 0.0 for count in histogram
    ]
    entropy = -sum(
        probability * math.log(probability)
        for probability in probabilities
        if probability > 0.0
    )
    active = sum(count > 0 for count in histogram)
    return {
        "latent_unit_count": total,
        "active_code_count": active,
        "histogram": histogram,
        "utilization_fraction": active / float(codebook_size),
        "empirical_entropy_nats": entropy,
        "perplexity": math.exp(entropy) if total else 0.0,
        "maximum_code_share": max(probabilities) if probabilities else 0.0,
    }


def histogram_jensen_shannon(left, right):
    if len(left) != len(right):
        raise DiagnosisError("histogram_shape", "histograms have different widths")
    left_total = float(sum(left))
    right_total = float(sum(right))
    if left_total <= 0.0 or right_total <= 0.0:
        return None
    left_p = [value / left_total for value in left]
    right_p = [value / right_total for value in right]
    middle = [(a + b) / 2.0 for a, b in zip(left_p, right_p)]

    def divergence(values, reference):
        return sum(
            value * math.log(value / target)
            for value, target in zip(values, reference)
            if value > 0.0
        )

    return 0.5 * divergence(left_p, middle) + 0.5 * divergence(
        right_p, middle
    )


def summarize_vectors(vectors, distinct_tolerance=1e-6):
    """Pure bounded vector summary used by tests and JSON validation."""

    if (
        isinstance(distinct_tolerance, bool)
        or not isinstance(distinct_tolerance, (int, float))
        or not math.isfinite(float(distinct_tolerance))
        or float(distinct_tolerance) <= 0.0
    ):
        raise DiagnosisError(
            "invalid_tolerance", "distinct tolerance must be finite and positive"
        )
    rows = tuple(tuple(float(value) for value in row) for row in vectors)
    if not rows:
        return {
            "shape": [0, 0],
            "finite_value_count": 0,
            "nonfinite_value_count": 0,
            "numerically_distinct_vector_count": 0,
            "overall_variance": None,
            "norm": _empty_distribution(),
            "pairwise_distance": _empty_distribution(),
        }
    width = len(rows[0])
    if width == 0 or any(len(row) != width for row in rows):
        raise DiagnosisError("vector_shape", "vectors are not rectangular")
    finite = [value for row in rows for value in row if math.isfinite(value)]
    nonfinite = len(rows) * width - len(finite)
    per_dimension_mean = []
    per_dimension_std = []
    for index in range(width):
        values = [row[index] for row in rows]
        if all(math.isfinite(value) for value in values):
            mean = sum(values) / len(values)
            variance = sum((value - mean) ** 2 for value in values) / len(values)
            per_dimension_mean.append(mean)
            per_dimension_std.append(math.sqrt(variance))
        else:
            per_dimension_mean.append(None)
            per_dimension_std.append(None)
    representatives = []
    for row in rows:
        if not all(math.isfinite(value) for value in row):
            continue
        if not any(
            max(abs(left - right) for left, right in zip(row, existing))
            <= distinct_tolerance
            for existing in representatives
        ):
            representatives.append(row)
    norms = [
        math.sqrt(sum(value * value for value in row))
        for row in rows
        if all(math.isfinite(value) for value in row)
    ]
    distances = []
    for left_index, left in enumerate(rows):
        if not all(math.isfinite(value) for value in left):
            continue
        for right in rows[left_index + 1 :]:
            if all(math.isfinite(value) for value in right):
                distances.append(math.sqrt(sum(
                    (a - b) ** 2 for a, b in zip(left, right)
                )))
    overall_variance = None
    if finite and nonfinite == 0:
        mean = sum(finite) / len(finite)
        overall_variance = sum(
            (value - mean) ** 2 for value in finite
        ) / len(finite)
    return {
        "shape": [len(rows), width],
        "finite_value_count": len(finite),
        "nonfinite_value_count": nonfinite,
        "numerically_distinct_vector_count": len(representatives),
        "distinct_tolerance": distinct_tolerance,
        "distinct_vector_equivalence": (
            "deterministic_greedy_maximum_absolute_channel_difference"
        ),
        "per_dimension_mean": per_dimension_mean,
        "per_dimension_std": per_dimension_std,
        "overall_variance": overall_variance,
        "norm": distribution(norms),
        "pairwise_distance": distribution(distances),
    }


def codebook_statistics(embeddings, active_codes=(), tolerance=1e-6):
    summary = summarize_vectors(embeddings, tolerance)
    rows = tuple(tuple(float(value) for value in row) for row in embeddings)
    active = set(active_codes)
    if any(
        isinstance(index, bool)
        or not isinstance(index, int)
        or index < 0
        or index >= len(rows)
        for index in active
    ):
        raise DiagnosisError("invalid_active_code", "active code is out of range")
    summary.update({
        "embedding_count": len(rows),
        "near_duplicate_entry_count": (
            len(rows) - summary["numerically_distinct_vector_count"]
        ),
        "active_codes": sorted(active),
        "dead_code_count": len(rows) - len(active),
        "dead_code_scope": (
            "unobserved assignments in selected train/validation families; "
            "not proof of historical inactivity"
        ),
        "unobserved_code_count_in_selected_partitions": (
            len(rows) - len(active)
        ),
        "distance_convention": "squared_euclidean_argmin",
        "pairwise_summary_distance_convention": "euclidean",
        "codebook_effectively_collapsed": (
            summary["numerically_distinct_vector_count"] <= 1
        ),
        "inactive_codes_distinguishable_but_unassigned": (
            len(rows) > len(active)
            and summary["numerically_distinct_vector_count"] > len(active)
        ),
    })
    return summary


def decoder_sensitivity_statistics(
    interventions, tolerance=1e-6, class_width=1
):
    """Summarize head outputs supplied as code -> flat numeric vector."""

    if not interventions:
        raise DiagnosisError("empty_intervention", "no code intervention supplied")
    codes = sorted(interventions)
    width = len(interventions[codes[0]])
    if any(len(interventions[code]) != width for code in codes):
        raise DiagnosisError("intervention_shape", "interventions are misaligned")
    if (
        isinstance(tolerance, bool)
        or not isinstance(tolerance, (int, float))
        or not math.isfinite(float(tolerance))
        or float(tolerance) <= 0.0
    ):
        raise DiagnosisError(
            "invalid_tolerance", "sensitivity tolerance must be finite and positive"
        )
    values = [[float(value) for value in interventions[code]] for code in codes]
    if any(not math.isfinite(value) for row in values for value in row):
        raise DiagnosisError(
            "nonfinite_intervention", "decoder intervention contains nonfinite values"
        )
    if width == 0:
        return {
            "code_count": len(codes),
            "value_count_per_code": 0,
            "logit_variance_mean": None,
            "maximum_absolute_difference": None,
            "mean_absolute_difference": None,
            "decision_type": (
                "continuous"
                if class_width is None
                else ("binary" if class_width == 1 else "argmax")
            ),
            "argmax_or_binary_decision_change_fraction": None,
            "distinct_complete_decision_count": None,
            "materially_sensitive": None,
            "code_17_maximum_absolute_difference_from_other_codes": None,
            "tolerance": tolerance,
        }
    reference = values[0]
    absolute = [
        abs(value - reference[index])
        for row in values[1:]
        for index, value in enumerate(row)
    ]
    decisions = _intervention_decisions(values, class_width)
    if decisions is None:
        changed_fraction = None
        distinct_decisions = None
    else:
        changed = sum(
            decision != decisions[0][index]
            for row in decisions[1:]
            for index, decision in enumerate(row)
        )
        denominator = max((len(values) - 1) * len(decisions[0]), 1)
        changed_fraction = changed / denominator
        distinct_decisions = len(set(decisions))
    column_variances = []
    for index in range(width):
        column = [row[index] for row in values]
        mean = sum(column) / len(column)
        column_variances.append(
            sum((value - mean) ** 2 for value in column) / len(column)
        )
    return {
        "code_count": len(codes),
        "value_count_per_code": width,
        "logit_variance_mean": (
            sum(column_variances) / len(column_variances)
        ),
        "maximum_absolute_difference": max(absolute) if absolute else 0.0,
        "mean_absolute_difference": (
            sum(absolute) / len(absolute) if absolute else 0.0
        ),
        "decision_type": (
            "continuous"
            if class_width is None
            else ("binary" if class_width == 1 else "argmax")
        ),
        "argmax_or_binary_decision_change_fraction": changed_fraction,
        "distinct_complete_decision_count": distinct_decisions,
        "materially_sensitive": (
            max(absolute) > tolerance if absolute else False
        ),
        "code_17_maximum_absolute_difference_from_other_codes": (
            _code_reference_difference(values, codes, 17)
        ),
        "tolerance": tolerance,
    }


def loss_masking_accounting(records):
    """Aggregate exact supplied denominators and class counts."""

    totals = {}
    class_counts = {}
    for record in records:
        for name, value in record.get("counts", {}).items():
            totals[name] = totals.get(name, 0) + int(value)
        for name, values in record.get("class_counts", {}).items():
            target = class_counts.setdefault(name, {})
            for key, value in values.items():
                target[str(key)] = target.get(str(key), 0) + int(value)
    majority = {}
    for name, values in class_counts.items():
        count = sum(values.values())
        majority[name] = (
            max(values.values()) / float(count) if count else None
        )
    proportions = {}
    for result_name, numerator, denominator in (
        ("pad_token_proportion", "padded_node_slots", "total_node_slots"),
        (
            "geometry_applicable_channel_proportion",
            "geometry_applicable_channels",
            "geometry_total_channels",
        ),
        (
            "geometry_applicable_within_supervised_nodes_proportion",
            "geometry_applicable_channels",
            "geometry_supervised_node_channels",
        ),
        (
            "edge_present_proportion",
            "edge_present_pairs",
            "edge_valid_pairs",
        ),
        (
            "pointer_supervised_slot_proportion",
            "pointer_supervised_slots",
            "pointer_total_slots",
        ),
    ):
        denominator_value = totals.get(denominator, 0)
        proportions[result_name] = (
            totals.get(numerator, 0) / float(denominator_value)
            if denominator_value else None
        )
    return {
        "counts": {name: totals[name] for name in sorted(totals)},
        "class_counts": {
            name: {key: values[key] for key in sorted(values)}
            for name, values in sorted(class_counts.items())
        },
        "majority_class_baseline_accuracy": {
            name: majority[name] for name in sorted(majority)
        },
        "proportions": proportions,
    }


def aggregate_gradients(parameters):
    """Aggregate fake or real named gradient observations deterministically."""

    records = []
    total_squared = 0.0
    for name, gradient in sorted(parameters):
        if gradient is None:
            records.append({
                "name": name,
                "status": "missing",
                "norm": None,
            })
            continue
        values = tuple(float(value) for value in gradient)
        if any(not math.isfinite(value) for value in values):
            status = "nonfinite"
            norm = None
        else:
            norm = math.sqrt(sum(value * value for value in values))
            status = "zero" if norm == 0.0 else "finite"
            total_squared += norm * norm
        records.append({"name": name, "status": status, "norm": norm})
    return {
        "aggregate_norm": math.sqrt(total_squared),
        "parameters": records,
        "missing_gradient_count": sum(
            item["status"] == "missing" for item in records
        ),
        "zero_gradient_count": sum(
            item["status"] == "zero" for item in records
        ),
        "nonfinite_gradient_count": sum(
            item["status"] == "nonfinite" for item in records
        ),
    }


def checkpoint_inventory(run_dir, torch_module=None):
    root = Path(run_dir)
    paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in (".pt", ".pth", ".ckpt")
    )
    records = []
    payloads = {}
    for path in paths:
        relative_path = path.relative_to(root).as_posix()
        symlink = path.is_symlink()
        record = {
            "relative_path": relative_path,
            "sha256": checkpoint_sha256(path),
            "is_symlink": symlink,
            "managed_training_checkpoint": relative_path in (
                "best.pt", "last.pt"
            ),
            "load_status": (
                "incompatible:symlink"
                if symlink
                else "not_inspected_without_pytorch"
            ),
            "checkpoint_version": None,
            "checkpoint_kind": None,
            "epoch": None,
            "global_step": None,
            "has_model_state": False,
            "has_optimizer_state": False,
            "has_rng_state": False,
            "has_training_metadata": False,
            "strict_model_load": False,
            "timeline_usable": False,
        }
        if torch_module is not None and not symlink:
            try:
                payload = torch_module.load(str(path), map_location="cpu")
                if not isinstance(payload, dict):
                    raise TypeError("checkpoint payload is not a mapping")
                record.update({
                    "load_status": "loaded",
                    "checkpoint_version": payload.get("checkpoint_version"),
                    "checkpoint_kind": payload.get("checkpoint_kind"),
                    "epoch": payload.get("epoch"),
                    "global_step": payload.get("global_step"),
                    "has_model_state": isinstance(
                        payload.get("model_state"), dict
                    ),
                    "has_optimizer_state": isinstance(
                        payload.get("optimizer_state"), dict
                    ),
                    "has_rng_state": isinstance(payload.get("rng_state"), dict),
                    "has_training_metadata": isinstance(
                        payload.get("training_config"), dict
                    ),
                    "data_state_train_family_count": (
                        len(payload.get("data_state", {}).get(
                            "train_family_ids", ()
                        ))
                        if isinstance(payload.get("data_state"), dict)
                        else None
                    ),
                    "data_state_validation_family_count": (
                        len(payload.get("data_state", {}).get(
                            "validation_family_ids", ()
                        ))
                        if isinstance(payload.get("data_state"), dict)
                        else None
                    ),
                })
                payloads[record["relative_path"]] = payload
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                record["load_status"] = "incompatible:{}".format(
                    type(exc).__name__
                )
        records.append(record)
    return records, payloads


def history_inventory(run_dir):
    root = Path(run_dir)
    records = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in (
            ".jsonl", ".json", ".log", ".txt", ".csv"
        ):
            continue
        record = {
            "relative_path": path.relative_to(root).as_posix(),
            "sha256": checkpoint_sha256(path),
            "recorded_fields": [],
            "parse_status": "not_jsonl",
            "event_count": None,
        }
        if path.suffix.lower() == ".jsonl":
            try:
                events = [
                    json.loads(line)
                    for line in path.read_text(encoding="utf-8").splitlines()
                    if line
                ]
                record["parse_status"] = "parsed"
                record["event_count"] = len(events)
                record["recorded_fields"] = sorted({
                    key for event in events if isinstance(event, dict)
                    for key in event
                })
            except (OSError, UnicodeDecodeError, ValueError):
                record["parse_status"] = "invalid_jsonl"
        records.append(record)
    return records


def checkpoint_timeline_limitation(records):
    usable = sum(bool(record.get("timeline_usable")) for record in records)
    return (
        None
        if usable > 1
        else "collapse time cannot be reconstructed from one checkpoint"
    )


def _checkpoint_analysis_key(record):
    epoch = record.get("epoch")
    step = record.get("global_step")
    return (
        -1 if not isinstance(step, int) or isinstance(step, bool) else step,
        -1 if not isinstance(epoch, int) or isinstance(epoch, bool) else epoch,
        record["relative_path"],
    )


def _managed_checkpoint_contract(record, payload):
    relative = record.get("relative_path")
    return (
        relative in ("best.pt", "last.pt")
        and not record.get("is_symlink", False)
        and payload.get("checkpoint_kind") == Path(relative).stem
    )


def build_artifacts(payloads):
    required = set(REQUIRED_ARTIFACTS)
    if set(payloads) != required:
        raise DiagnosisError(
            "artifact_set", "diagnostic payload names are incomplete"
        )
    artifacts = {}
    for name in REQUIRED_ARTIFACTS:
        value = payloads[name]
        if name == "latent_usage.csv":
            artifacts[name] = latent_usage_csv(value)
        else:
            artifacts[name] = _json_document(value)
    hashes = {
        name: hashlib.sha256(content).hexdigest()
        for name, content in sorted(artifacts.items())
    }
    artifacts[MANIFEST_ARTIFACT] = _json_document({
        "diagnostic_schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        "artifact_sha256": hashes,
        "manifest_self_hash_excluded": True,
    })
    return artifacts


def latent_usage_csv(rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream, fieldnames=LATENT_USAGE_COLUMNS, lineterminator="\n"
    )
    writer.writeheader()
    for row in rows:
        writer.writerow({
            name: row[name] for name in LATENT_USAGE_COLUMNS
        })
    return stream.getvalue().encode("utf-8")


def validate_diagnostic_artifacts(
    directory,
    *,
    authoritative_examples=None,
    training_run_dir=None,
    reviewed_commit=None,
):
    try:
        root = _resolve_publication_directory(directory)
    except EvaluationError as exc:
        raise DiagnosisError(exc.code, exc.detail) from exc
    expected = set(REQUIRED_ARTIFACTS) | {MANIFEST_ARTIFACT}
    actual = {
        path.name for path in root.iterdir()
        if path.is_file() and not path.is_symlink()
    }
    if actual != expected:
        raise DiagnosisError("artifact_set", "diagnostic artifact set differs")
    manifest = _strict_json(root / MANIFEST_ARTIFACT)
    hashes = manifest.get("artifact_sha256")
    if not isinstance(hashes, dict) or set(hashes) != set(REQUIRED_ARTIFACTS):
        raise DiagnosisError("artifact_manifest", "manifest names differ")
    for name in REQUIRED_ARTIFACTS:
        content = (root / name).read_bytes()
        if not content.endswith(b"\n") or content.endswith(b"\n\n"):
            raise DiagnosisError("artifact_newline", name)
        if hashlib.sha256(content).hexdigest() != hashes[name]:
            raise DiagnosisError("artifact_hash", name)
        if name.endswith(".json"):
            _strict_json(root / name)
    metadata = _strict_json(root / "run_metadata.json")
    partition = _strict_json(root / "partition_summary.json")
    checkpoint_document = _strict_json(root / "checkpoint_inventory.json")
    checkpoints = checkpoint_document["checkpoints"]
    if metadata.get("checkpoint_sha256") != [
        record.get("sha256") for record in checkpoints
    ]:
        raise DiagnosisError(
            "checkpoint_reconciliation",
            "metadata checkpoint hashes differ from inventory",
        )
    if (
        reviewed_commit is not None
        and metadata.get("repository_commit") != reviewed_commit
    ):
        raise DiagnosisError(
            "repository_commit", "artifact commit differs from reviewed commit"
        )
    if metadata.get("test_partition_evaluated") is not False:
        raise DiagnosisError("test_partition", "metadata permits test evaluation")
    if partition.get("test_partition_evaluated") is not False:
        raise DiagnosisError("test_partition", "partition summary permits test")
    if set(partition.get("selected_family_ids", {})) != {
        "train", "validation"
    }:
        raise DiagnosisError(
            "test_partition", "selected partitions are not train/validation only"
        )
    with (root / "latent_usage.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        reader = csv.DictReader(stream)
        rows = tuple(reader)
        if tuple(reader.fieldnames or ()) != LATENT_USAGE_COLUMNS:
            raise DiagnosisError("latent_header", "latent CSV header differs")
    if (root / "latent_usage.csv").read_bytes() != latent_usage_csv(rows):
        raise DiagnosisError("noncanonical_csv", "latent_usage.csv")
    _validate_partition_authority(partition, authoritative_examples)
    _validate_checkpoint_files(
        checkpoints, training_run_dir
    )
    selected_ids = partition.get("selected_family_ids", {})
    if authoritative_examples is not None or training_run_dir is not None:
        _validate_scientific_records(
            root, checkpoints, rows, selected_ids
        )
    for record in checkpoints:
        digest = record["sha256"]
        for partition_name in ("train", "validation"):
            matching = [
                row for row in rows
                if row["checkpoint_sha256"] == digest
                and row["partition"] == partition_name
            ]
            if record.get("timeline_usable") and not matching:
                raise DiagnosisError(
                    "latent_reconciliation",
                    "usable checkpoint lacks latent rows",
                )
            if record.get("timeline_usable"):
                if len(matching) != record["codebook_size"]:
                    raise DiagnosisError(
                        "latent_reconciliation",
                        "latent histogram width differs from codebook",
                    )
                observed = sum(int(row["assignment_count"]) for row in matching)
                expected_count = (
                    len(selected_ids[partition_name])
                    * record["latent_tokens"]
                )
                if observed != expected_count:
                    raise DiagnosisError(
                        "latent_reconciliation",
                        "latent assignment count differs from selected units",
                    )
                if {
                    int(row["code_index"]) for row in matching
                } != set(range(record["codebook_size"])):
                    raise DiagnosisError(
                        "latent_reconciliation",
                        "latent code rows are incomplete",
                    )
                fraction = sum(
                    float(row["assignment_fraction"]) for row in matching
                )
                if abs(fraction - 1.0) > 1e-12:
                    raise DiagnosisError(
                        "latent_reconciliation",
                        "latent assignment fractions do not sum to one",
                    )
    root_report = _strict_json(root / "root_cause_report.json")
    if tuple(
        item.get("hypothesis")
        for item in root_report.get("hypotheses", ())
    ) != HYPOTHESES:
        raise DiagnosisError(
            "root_cause_rows", "root-cause hypothesis inventory differs"
        )
    complete_hashes = dict(hashes)
    complete_hashes[MANIFEST_ARTIFACT] = checkpoint_sha256(
        root / MANIFEST_ARTIFACT
    )
    return {
        "artifact_sha256": complete_hashes,
        "root_cause_answers": root_report.get("answers"),
        "test_partition_evaluated": False,
    }


def publish_diagnostic_artifacts(
    output_dir,
    artifacts,
    *,
    authoritative_examples=None,
    training_run_dir=None,
    reviewed_commit=None,
):
    destination = Path(output_dir)
    try:
        _preflight_output(destination)
    except EvaluationError as exc:
        raise DiagnosisError(exc.code, exc.detail) from exc
    temporary = None
    try:
        temporary = Path(tempfile.mkdtemp(
            prefix="." + destination.name + ".tmp-",
            dir=str(destination.parent),
        ))
        for name in sorted(artifacts):
            with (temporary / name).open("wb") as stream:
                stream.write(artifacts[name])
                stream.flush()
                os.fsync(stream.fileno())
        _fsync_directory(temporary)
        validate_diagnostic_artifacts(
            temporary,
            authoritative_examples=authoritative_examples,
            training_run_dir=training_run_dir,
            reviewed_commit=reviewed_commit,
        )
        _atomic_no_replace(temporary, destination)
        temporary = None
        _fsync_directory(destination.parent)
    except DiagnosisError:
        raise
    except EvaluationError as exc:
        raise DiagnosisError(exc.code, exc.detail) from exc
    except (OSError, TypeError, ValueError) as exc:
        raise DiagnosisError("publication_failure", type(exc).__name__) from exc
    finally:
        if temporary is not None and temporary.exists():
            _remove_temporary_directory(temporary)


def _validate_partition_authority(partition, authoritative_examples):
    counts = partition.get("partition_counts")
    if (
        not isinstance(counts, dict)
        or partition.get("total_physical_family_count")
        != sum(counts.get(name, -1) for name in ("train", "validation", "test"))
    ):
        raise DiagnosisError(
            "partition_reconciliation", "partition counts do not reconcile"
        )
    selected = partition.get("selected_family_ids", {})
    selected_hashes = partition.get("selected_family_id_sha256", {})
    for name in ("train", "validation"):
        values = tuple(selected.get(name, ()))
        if values != tuple(sorted(values)) or len(values) != len(set(values)):
            raise DiagnosisError(
                "partition_reconciliation", name + " selected IDs differ"
            )
        if selected_hashes.get(name) != _identifier_sha256(values):
            raise DiagnosisError(
                "partition_reconciliation", name + " selected hash differs"
            )
    if authoritative_examples is None:
        return
    reconciliation = partition_reconciliation(
        tuple(authoritative_examples)
    )
    if (
        partition.get("total_physical_family_count")
        != reconciliation["total_physical_family_count"]
        or partition.get("partition_counts")
        != reconciliation["partition_counts"]
        or partition.get("partition_id_sha256")
        != reconciliation["partition_id_sha256"]
    ):
        raise DiagnosisError(
            "partition_reconciliation",
            "artifact partitions differ from authoritative loader",
        )
    for name in ("train", "validation"):
        values = tuple(selected[name])
        if values != reconciliation["_ids"][name][:len(values)]:
            raise DiagnosisError(
                "partition_reconciliation",
                name + " selection is not the authoritative prefix",
            )
    test_ids = set(reconciliation["_ids"]["test"])
    if test_ids & (
        set(selected["train"]) | set(selected["validation"])
    ):
        raise DiagnosisError(
            "test_partition_leakage", "test family appears in selection"
        )


def _validate_checkpoint_files(checkpoints, training_run_dir):
    if training_run_dir is None:
        return
    root = Path(training_run_dir)
    candidates = tuple(sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in (".pt", ".pth", ".ckpt")
    ))
    by_relative = {
        record.get("relative_path"): record for record in checkpoints
    }
    actual = tuple(path.relative_to(root).as_posix() for path in candidates)
    if tuple(sorted(by_relative)) != actual:
        raise DiagnosisError(
            "checkpoint_reconciliation",
            "checkpoint inventory paths differ from training run",
        )
    for path in candidates:
        relative = path.relative_to(root).as_posix()
        record = by_relative[relative]
        if record.get("is_symlink") is not path.is_symlink():
            raise DiagnosisError(
                "checkpoint_reconciliation", relative + " symlink status differs"
            )
        if checkpoint_sha256(path) != record.get("sha256"):
            raise DiagnosisError(
                "checkpoint_reconciliation", relative + " hash differs"
            )


def _validate_scientific_records(root, checkpoints, latent_rows, selected_ids):
    usable = tuple(
        record for record in checkpoints if record.get("timeline_usable")
    )
    checkpoint_keys = {
        (record["sha256"], record["relative_path"]) for record in usable
    }
    partition_keys = {
        (digest, path, partition)
        for digest, path in checkpoint_keys
        for partition in ("train", "validation")
    }

    def records(name):
        value = _strict_json(root / name).get("records")
        if not isinstance(value, list):
            raise DiagnosisError(
                "scientific_reconciliation", name + " records are invalid"
            )
        return value

    prequant = records("prequant_summary.json")
    losses = records("loss_masking_audit.json")
    sensitivity = records("decoder_sensitivity.json")
    codebooks = records("codebook_summary.json")
    gradients = records("gradient_probe.json")

    def partition_record_keys(values):
        return {
            (
                item.get("checkpoint_sha256"),
                item.get("checkpoint_relative_path"),
                item.get("partition"),
            )
            for item in values
        }

    if (
        len(prequant) != len(partition_keys)
        or partition_record_keys(prequant) != partition_keys
        or len(losses) != len(partition_keys)
        or partition_record_keys(losses) != partition_keys
        or len(sensitivity) != len(partition_keys)
        or partition_record_keys(sensitivity) != partition_keys
    ):
        raise DiagnosisError(
            "scientific_reconciliation",
            "partition-level diagnostic records differ from usable checkpoints",
        )
    checkpoint_record_keys = {
        (
            item.get("checkpoint_sha256"),
            item.get("checkpoint_relative_path"),
        )
        for item in codebooks
    }
    gradient_keys = {
        (
            item.get("checkpoint_sha256"),
            item.get("checkpoint_relative_path"),
        )
        for item in gradients
    }
    if (
        len(codebooks) != len(checkpoint_keys)
        or checkpoint_record_keys != checkpoint_keys
        or len(gradients) != len(checkpoint_keys)
        or gradient_keys != checkpoint_keys
    ):
        raise DiagnosisError(
            "scientific_reconciliation",
            "checkpoint-level diagnostic records differ",
        )
    rows_by_key = {}
    for row in latent_rows:
        key = (
            row["checkpoint_sha256"],
            row["checkpoint_relative_path"],
            row["partition"],
        )
        rows_by_key.setdefault(key, []).append(row)
    if set(rows_by_key) != partition_keys:
        raise DiagnosisError(
            "scientific_reconciliation", "latent rows differ from checkpoints"
        )
    for item in prequant:
        key = (
            item["checkpoint_sha256"],
            item["checkpoint_relative_path"],
            item["partition"],
        )
        ordered = sorted(
            rows_by_key[key], key=lambda row: int(row["code_index"])
        )
        histogram = [
            int(row["assignment_count"]) for row in ordered
        ]
        statistics = item.get("latent_statistics", {})
        if (
            statistics.get("histogram") != histogram
            or statistics.get("latent_unit_count") != sum(histogram)
            or statistics.get("active_code_count")
            != sum(count > 0 for count in histogram)
        ):
            raise DiagnosisError(
                "scientific_reconciliation",
                "prequant latent statistics differ from latent CSV",
            )
    for item in sensitivity:
        partition = item["partition"]
        families = item.get("family_ids")
        if (
            not isinstance(families, list)
            or not families
            or not set(families) <= set(selected_ids[partition])
        ):
            raise DiagnosisError(
                "scientific_reconciliation",
                "decoder sensitivity families differ from selection",
            )
    for item in gradients:
        families = item.get("family_ids")
        if (
            not isinstance(families, list)
            or not families
            or not set(families) <= set(selected_ids["train"])
            or item.get("backward_loss") != "flat_mixed_vq_loss.total"
            or item.get("optimizer_created") is not False
            or item.get("optimizer_step_called") is not False
        ):
            raise DiagnosisError(
                "scientific_reconciliation",
                "gradient probe contract differs",
            )


def run_diagnosis(arguments, torch_module=None):
    """Run bounded analysis without decoding or selecting test examples."""

    for name in ("batch_size", "sensitivity_family_limit"):
        value = getattr(arguments, name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise DiagnosisError(
                "invalid_limit", "{} must be a positive integer".format(name)
            )
    try:
        _preflight_output(Path(arguments.output_dir))
    except EvaluationError as exc:
        raise DiagnosisError(exc.code, exc.detail) from exc
    if torch_module is None:
        try:
            import torch as torch_module
        except ImportError as exc:
            raise DiagnosisError("pytorch_unavailable", "PyTorch is required") from exc
    from .config import FlatBaselineConfig
    from .checkpointing import _validate_payload
    from .data import load_training_data, make_data_loader
    from .losses import dense_edge_targets, flat_mixed_vq_loss
    from .model import FlatMixedVQModel
    from .training import _torch_batch
    from .training_config import TrainingConfig

    repository = Path(__file__).resolve().parents[2]
    repository_state = source_state(repository)
    if repository_state["git_commit"] != arguments.reviewed_commit:
        raise DiagnosisError(
            "repository_commit",
            "reviewed commit differs from the live repository",
        )
    physical = tuple(load_physical_examples(
        arguments.corpus_dir, arguments.split_manifest
    ))
    inventory, payloads = checkpoint_inventory(
        arguments.training_run_dir, torch_module
    )
    history = history_inventory(arguments.training_run_dir)
    checkpoint_data_state = None
    for record in sorted(inventory, key=_checkpoint_analysis_key):
        payload = payloads.get(record["relative_path"])
        if (
            record.get("managed_training_checkpoint")
            and not record.get("is_symlink", False)
            and payload is not None
            and isinstance(payload.get("data_state"), dict)
        ):
            checkpoint_data_state = payload["data_state"]
            break
    reconciliation = partition_reconciliation(
        physical, checkpoint_data_state
    )
    selected_ids = select_permitted_ids(
        reconciliation,
        arguments.train_limit,
        arguments.validation_limit,
    )
    data = load_training_data(
        arguments.corpus_dir,
        arguments.split_manifest,
        "train",
        "validation",
        None,
    )
    by_partition = {
        "train": tuple(
            item for item in data.train_examples
            if item.physical_family_id in set(selected_ids["train"])
        ),
        "validation": tuple(
            item for item in data.validation_examples
            if item.physical_family_id in set(selected_ids["validation"])
        ),
    }
    if any(not values for values in by_partition.values()):
        raise DiagnosisError("empty_partition", "train and validation are required")

    latent_rows = []
    prequant_records = []
    codebook_records = []
    sensitivity_records = []
    loss_records = []
    gradient_records = []
    usable_payloads = []
    for record in sorted(inventory, key=_checkpoint_analysis_key):
        payload = payloads.get(record["relative_path"])
        if payload is None:
            continue
        try:
            _validate_payload(payload)
            model_config = FlatBaselineConfig(**payload["model_config"])
            training_config = TrainingConfig(**payload["training_config"])
            model_config.validate()
            training_config.validate()
            model = FlatMixedVQModel(model_config)
            model.load_state_dict(payload["model_state"], strict=True)
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            record["load_status"] = "incompatible:{}".format(type(exc).__name__)
            continue
        record["strict_model_load"] = True
        record["codebook_size"] = model_config.codebook_size
        record["latent_tokens"] = model_config.latent_tokens
        managed_checkpoint = _managed_checkpoint_contract(record, payload)
        record["timeline_usable"] = (
            managed_checkpoint
            and not record.get("is_symlink", False)
            and payload.get("data_state", {}).get("split_manifest")
            == arguments.split_manifest
            and payload.get("data_state", {}).get("train_partition") == "train"
            and payload.get("data_state", {}).get("validation_partition")
            == "validation"
            and payload.get("data_state", {}).get("train_family_ids")
            == list(reconciliation["_ids"]["train"])
            and payload.get("data_state", {}).get("validation_family_ids")
            == list(reconciliation["_ids"]["validation"])
        )
        record["checkpoint_partition_ids_match"] = record["timeline_usable"]
        record["timeline_exclusion_reason"] = (
            None
            if record["timeline_usable"]
            else (
                "not a provenance-backed root best.pt/last.pt checkpoint"
                if not managed_checkpoint or record.get("is_symlink", False)
                else "checkpoint data_state differs from authoritative partitions"
            )
        )
        if not record["timeline_usable"]:
            continue
        usable_payloads.append((record, payload, model_config, training_config))
        model.eval()
        codebook = model.vq.embedding.detach().cpu().tolist()
        all_active = set()
        for partition_name in ("train", "validation"):
            measurement = _measure_partition(
                model,
                by_partition[partition_name],
                training_config,
                torch_module,
                _torch_batch,
                make_data_loader,
                flat_mixed_vq_loss,
                dense_edge_targets,
                arguments.batch_size,
            )
            stats = latent_statistics(
                measurement["assignments"], model_config.codebook_size
            )
            all_active.update(
                index for index, count in enumerate(stats["histogram"]) if count
            )
            for index, count in enumerate(stats["histogram"]):
                latent_rows.append({
                    "checkpoint_sha256": record["sha256"],
                    "checkpoint_relative_path": record["relative_path"],
                    "partition": partition_name,
                    "code_index": index,
                    "assignment_count": count,
                    "assignment_fraction": (
                        count / float(stats["latent_unit_count"])
                        if stats["latent_unit_count"] else 0.0
                    ),
                })
            prequant_records.append({
                "checkpoint_sha256": record["sha256"],
                "checkpoint_relative_path": record["relative_path"],
                "partition": partition_name,
                "capture_point": (
                    "FlatMixedVQModel.to_codebook output passed directly "
                    "to EMAVectorQuantizer"
                ),
                "latent_statistics": stats,
                "family_level_consistency_fraction": measurement[
                    "family_consistency"
                ],
                "assignment_margin_convention": (
                    "second_nearest_squared_distance_minus_"
                    "nearest_squared_distance"
                ),
                "assignment_margin": distribution(measurement["margins"]),
                "distance_to_assigned_code": distribution(
                    measurement["assigned_distances"]
                ),
                "within_between_family_variation": measurement["variation"],
                "vectors": _tensor_vector_summary(
                    measurement["prequant"], torch_module
                ),
                "loss_and_masking": measurement["loss_audit"],
            })
            loss_records.append({
                "checkpoint_sha256": record["sha256"],
                "checkpoint_relative_path": record["relative_path"],
                "partition": partition_name,
                **measurement["loss_audit"],
            })
        train_hist = prequant_records[-2]["latent_statistics"]["histogram"]
        validation_hist = prequant_records[-1]["latent_statistics"]["histogram"]
        prequant_records[-2]["train_validation_histogram_js_divergence"] = (
            histogram_jensen_shannon(train_hist, validation_hist)
        )
        prequant_records[-1]["train_validation_histogram_js_divergence"] = (
            histogram_jensen_shannon(train_hist, validation_hist)
        )
        codebook_record = codebook_statistics(codebook, all_active)
        codebook_record.update({
            "checkpoint_sha256": record["sha256"],
            "checkpoint_relative_path": record["relative_path"],
            "effective_rank": _effective_rank(
                model.vq.embedding.detach().cpu(), torch_module
            ),
        })
        if 17 < len(codebook):
            codebook_record["distance_from_code_17"] = [
                math.sqrt(sum(
                    (value - reference) ** 2
                    for value, reference in zip(row, codebook[17])
                ))
                for row in codebook
            ]
        else:
            codebook_record["distance_from_code_17"] = None
        codebook_records.append(codebook_record)
        for partition_name in ("train", "validation"):
            sensitivity_records.append(_decoder_sensitivity(
                model,
                by_partition[partition_name][
                    : arguments.sensitivity_family_limit
                ],
                record,
                partition_name,
                torch_module,
                _torch_batch,
                make_data_loader,
                dense_edge_targets,
            ))
        gradient_records.append(_gradient_probe(
            payload,
            model_config,
            training_config,
            by_partition["train"],
            record,
            torch_module,
            _torch_batch,
            make_data_loader,
            flat_mixed_vq_loss,
            FlatMixedVQModel,
            Path(arguments.training_run_dir) / record["relative_path"],
        ))

    if not usable_payloads:
        raise DiagnosisError(
            "no_usable_checkpoint", "no checkpoint strictly matched the corpus"
        )
    history_summary = _history_summary(arguments.training_run_dir, history)
    root_report = classify_root_causes(
        prequant_records,
        codebook_records,
        sensitivity_records,
        loss_records,
        gradient_records,
        inventory,
        history_summary,
    )
    identity = corpus_identity(arguments.corpus_dir, arguments.split_manifest)
    public_partition = {
        key: value for key, value in reconciliation.items() if key != "_ids"
    }
    public_partition["selected_family_ids"] = {
        name: list(values) for name, values in selected_ids.items()
    }
    public_partition["selected_family_id_sha256"] = {
        name: _identifier_sha256(values)
        for name, values in selected_ids.items()
    }
    payload_documents = {
        "run_metadata.json": {
            "diagnostic_schema_version": DIAGNOSTIC_SCHEMA_VERSION,
            "repository_commit": arguments.reviewed_commit,
            "source_tree_sha256": repository_state["source_tree_sha256"],
            "source_dirty": repository_state["git_dirty"],
            "corpus_identity": identity,
            "checkpoint_sha256": [
                record["sha256"] for record in inventory
            ],
            "quantizer_update_mechanism": (
                "EMA buffers updated only in training mode; no codebook parameter"
            ),
            "quantizer_distance_convention": "squared_euclidean_argmin",
            "prequant_capture_point": (
                "FlatMixedVQModel.to_codebook forward output"
            ),
            "test_partition_evaluated": False,
            "diagnostic_intervention_not_evaluation": True,
        },
        "checkpoint_inventory.json": {
            "checkpoints": inventory,
            "history_files": history,
            "timeline_checkpoint_count": sum(
                record["timeline_usable"] for record in inventory
            ),
            "timeline_limitation": checkpoint_timeline_limitation(inventory),
        },
        "partition_summary.json": public_partition,
        "latent_usage.csv": latent_rows,
        "prequant_summary.json": {"records": prequant_records},
        "codebook_summary.json": {"records": codebook_records},
        "decoder_sensitivity.json": {
            "diagnostic_intervention_not_evaluation": True,
            "records": sensitivity_records,
        },
        "loss_masking_audit.json": {
            "uses_exact_flat_mixed_vq_loss": True,
            "records": loss_records,
        },
        "gradient_probe.json": {
            "optimizer_created": False,
            "optimizer_step_called": False,
            "checkpoint_mutated": False,
            "records": gradient_records,
        },
        "root_cause_report.json": root_report,
    }
    artifacts = build_artifacts(payload_documents)
    publish_diagnostic_artifacts(
        arguments.output_dir,
        artifacts,
        authoritative_examples=physical,
        training_run_dir=arguments.training_run_dir,
        reviewed_commit=arguments.reviewed_commit,
    )
    return validate_diagnostic_artifacts(
        arguments.output_dir,
        authoritative_examples=physical,
        training_run_dir=arguments.training_run_dir,
        reviewed_commit=arguments.reviewed_commit,
    )


def _measure_partition(
    model,
    examples,
    training_config,
    torch_module,
    torch_batch,
    make_data_loader,
    loss_function,
    dense_edge_targets,
    batch_size,
):
    assignments = []
    prequant = []
    margins = []
    assigned_distances = []
    family_consistency = []
    loss_sums = {}
    example_count = 0
    accounting_records = []
    hook_values = []

    def capture(module, inputs, output):
        del module, inputs
        hook_values.append(output.detach())

    handle = model.to_codebook.register_forward_hook(capture)
    loader = make_data_loader(
        examples, batch_size, 0, training_config.seed, False, torch_module
    )
    try:
        with torch_module.no_grad():
            for batch in loader:
                inputs, target = torch_batch(
                    batch, torch_module.device("cpu"), torch_module
                )
                hook_values[:] = []
                output = model(target=target, **inputs)
                if len(hook_values) != 1:
                    raise DiagnosisError(
                        "prequant_capture", "to_codebook hook count differs"
                    )
                current = hook_values[0].cpu()
                indices = output.code_indices.detach().cpu()
                if current.shape[:-1] != indices.shape:
                    raise DiagnosisError(
                        "prequant_shape", "prequant and assignments differ"
                    )
                prequant.append(current)
                flat = current.reshape(-1, current.size(-1))
                codebook = model.vq.embedding.detach().cpu()
                distances = (
                    flat.pow(2).sum(1, keepdim=True)
                    + codebook.pow(2).sum(1).unsqueeze(0)
                    - 2.0 * flat.matmul(codebook.t())
                )
                recomputed = distances.argmin(dim=1).view_as(indices)
                if not torch_module.equal(recomputed, indices):
                    raise DiagnosisError(
                        "assignment_reconciliation",
                        "captured prequant vectors do not reproduce VQ assignments",
                    )
                nearest = distances.topk(2, largest=False, dim=1).values
                margins.extend((nearest[:, 1] - nearest[:, 0]).tolist())
                assigned_distances.extend(nearest[:, 0].tolist())
                assignments.extend(indices.reshape(-1).tolist())
                family_consistency.extend([
                    int(row.unique().numel() == 1) for row in indices
                ])
                losses = loss_function(output, target, model.config)
                current_count = len(batch.family_ids)
                example_count += current_count
                for name, values in losses.per_example.items():
                    loss_sums[name] = loss_sums.get(name, 0.0) + float(
                        values.detach().sum().item()
                    )
                accounting_records.append(
                    _batch_loss_accounting(
                        target, output, dense_edge_targets
                    )
                )
    finally:
        handle.remove()
    counts = loss_masking_accounting(accounting_records)
    before = {
        name: loss_sums[name] / float(example_count)
        for name in sorted(loss_sums)
    }
    weights = {
        "node_type": model.config.node_type_loss_weight,
        "categorical_attributes": model.config.categorical_loss_weight,
        "geometry": model.config.geometry_loss_weight,
        "edge_presence": model.config.edge_presence_loss_weight,
        "edge_type": model.config.edge_type_loss_weight,
        "operation_pointer": model.config.operation_pointer_loss_weight,
        "vq_commitment": model.config.vq_loss_weight,
    }
    after = {
        name: before[name] * weights[name] for name in weights
    }
    weighted_total = sum(after.values())
    return {
        "assignments": assignments,
        "prequant": torch_module.cat(prequant, dim=0),
        "margins": margins,
        "assigned_distances": assigned_distances,
        "family_consistency": (
            sum(family_consistency) / float(len(family_consistency))
        ),
        "variation": _partition_variation(
            torch_module.cat(prequant, dim=0), torch_module
        ),
        "loss_audit": {
            "component_before_weighting": before,
            "component_after_weighting": after,
            "relative_weighted_contribution": {
                name: (
                    after[name] / weighted_total if weighted_total else None
                )
                for name in sorted(after)
            },
            **counts,
        },
    }


def _batch_loss_accounting(target, output, dense_edge_targets):
    node_mask = target["node_mask"]
    geometry_mask = target["geometry_mask"] & node_mask.unsqueeze(-1)
    presence, edge_types, valid_pairs = dense_edge_targets(
        target, output.edge_presence_logits.size(1)
    )
    positives = presence & valid_pairs
    negatives = ~presence & valid_pairs
    counts = {
        "padded_node_slots": int((~node_mask).sum().item()),
        "supervised_node_slots": int(node_mask.sum().item()),
        "total_node_slots": node_mask.numel(),
        "geometry_applicable_channels": int(geometry_mask.sum().item()),
        "geometry_total_channels": geometry_mask.numel(),
        "geometry_supervised_node_channels": int(
            node_mask.sum().item() * target["geometry_mask"].size(-1)
        ),
        "edge_present_pairs": int(positives.sum().item()),
        "edge_absent_pairs": int(negatives.sum().item()),
        "edge_valid_pairs": int(valid_pairs.sum().item()),
        "pointer_supervised_slots": int(
            target["operation_mask"].sum().item()
        ),
        "pointer_total_slots": target["operation_mask"].numel(),
        "node_type_supervised_positions": int(node_mask.sum().item()),
        "categorical_supervised_positions": int(
            node_mask.sum().item()
            * target["categorical_attributes"].size(-1)
        ),
        "edge_type_supervised_positions": int(positives.sum().item()),
        "vq_latent_units": output.code_indices.numel(),
    }
    class_counts = {
        "node_type": _masked_class_counts(
            target["node_type_ids"], node_mask
        ),
        "edge_presence": {
            "0": counts["edge_absent_pairs"],
            "1": counts["edge_present_pairs"],
        },
        "edge_type": _masked_class_counts(edge_types, positives),
        "operation_pointer": _masked_class_counts(
            target["operation_sequence"], target["operation_mask"]
        ),
    }
    for index in range(target["categorical_attributes"].size(-1)):
        class_counts["categorical_{}".format(index)] = _masked_class_counts(
            target["categorical_attributes"][..., index], node_mask
        )
    return {"counts": counts, "class_counts": class_counts}


def _masked_class_counts(values, mask):
    selected = values[mask].detach().cpu().reshape(-1).tolist()
    result = {}
    for value in selected:
        key = str(int(value))
        result[key] = result.get(key, 0) + 1
    return result


def _tensor_vector_summary(tensor, torch_module):
    flat = tensor.reshape(-1, tensor.size(-1)).to(dtype=torch_module.float64)
    summary = summarize_vectors(flat.tolist())
    summary["effective_rank"] = _effective_rank(flat, torch_module)
    return summary


def _effective_rank(tensor, torch_module):
    values = tensor.to(dtype=torch_module.float64)
    if values.dim() != 2 or values.size(0) < 2:
        return 0.0
    if not bool(torch_module.isfinite(values).all().item()):
        return None
    centered = values - values.mean(dim=0, keepdim=True)
    singular = torch_module.linalg.svdvals(centered)
    squared = singular.pow(2)
    total = squared.sum()
    if float(total.item()) <= 0.0:
        return 0.0
    probabilities = squared / total
    positive = probabilities > 0
    entropy = -(probabilities[positive] * probabilities[positive].log()).sum()
    return float(entropy.exp().item())


def _partition_variation(tensor, torch_module):
    values = tensor.to(dtype=torch_module.float64)
    within = []
    for family in values:
        for left_index in range(family.size(0)):
            for right_index in range(left_index + 1, family.size(0)):
                within.append(float(
                    (family[left_index] - family[right_index]).norm().item()
                ))
    family_means = values.mean(dim=1)
    between = []
    for left_index in range(family_means.size(0)):
        for right_index in range(left_index + 1, family_means.size(0)):
            between.append(float(
                (family_means[left_index] - family_means[right_index])
                .norm().item()
            ))
    return {
        "within_family_distance": distribution(within),
        "between_family_distance": distribution(between),
    }


def _decoder_sensitivity(
    model,
    examples,
    checkpoint_record,
    partition,
    torch_module,
    torch_batch,
    make_data_loader,
    dense_edge_targets,
):
    loader = make_data_loader(examples, len(examples), 0, 0, False, torch_module)
    batch = next(iter(loader))
    _, target = torch_batch(batch, torch_module.device("cpu"), torch_module)
    target_categories = torch_module.cat((
        target["node_type_ids"].unsqueeze(-1),
        target["categorical_attributes"],
    ), dim=-1)
    masks = _decoder_supervision_masks(
        target,
        model.config.max_operations,
        dense_edge_targets,
        torch_module,
    )
    heads = {}
    head_class_widths = {}
    family_outputs = {}
    with torch_module.no_grad():
        for code in range(model.config.codebook_size):
            indices = torch_module.full(
                (len(examples), model.config.latent_tokens),
                code,
                dtype=torch_module.long,
            )
            memory = model.memory_from_indices(indices)
            prefix = model.decode_prefix(
                memory,
                target_categories[:, :-1],
                target["geometry"][:, :-1],
                target["geometry_mask"][:, :-1],
                target["node_mask"][:, :-1],
            )
            relations = model.decode_relations(
                prefix.decoded_states, target["node_mask"]
            )
            tensors = {
                "node_type": prefix.node_type_logits,
                "geometry": prefix.geometry,
                "edge_presence": relations.edge_presence_logits,
                "edge_type": relations.edge_type_logits,
                "operation_pointer": relations.operation_pointer_logits,
            }
            for index, tensor in enumerate(prefix.categorical_logits):
                tensors["categorical_{}".format(index)] = tensor
            for name, tensor in tensors.items():
                mask = masks[name]
                selected = tensor[mask]
                heads.setdefault(name, {})[code] = (
                    selected.detach().cpu().reshape(-1).tolist()
                )
                family_outputs.setdefault(name, {})[code] = (
                    _masked_family_rows(tensor, mask)
                )
                if name == "geometry":
                    head_class_widths[name] = None
                elif name == "edge_presence":
                    head_class_widths[name] = 1
                else:
                    head_class_widths[name] = tensor.size(-1)
    complete_predictions = []
    for code in range(model.config.codebook_size):
        complete = []
        for name in sorted(heads):
            decisions = _intervention_decisions(
                [heads[name][code]], head_class_widths[name]
            )
            if decisions is not None:
                complete.extend(decisions[0])
        complete_predictions.append(tuple(complete))
    return {
        "checkpoint_sha256": checkpoint_record["sha256"],
        "checkpoint_relative_path": checkpoint_record["relative_path"],
        "partition": partition,
        "family_ids": list(batch.family_ids),
        "forced_prefix": "authoritative_teacher_forced_target_prefix",
        "latent_intervention": "all latent positions filled with each code",
        "mask_contract": {
            "node_and_categorical": "authoritative node_mask",
            "geometry": "authoritative applicable geometry channels",
            "edge_presence": "valid nonpadding nondiagonal pairs",
            "edge_type": "authoritative present valid pairs",
            "operation_pointer": "authoritative supervised operation slots",
        },
        "supervised_unit_counts": {
            name: int(mask.sum().item())
            for name, mask in sorted(masks.items())
        },
        "head_statistics": {
            name: decoder_sensitivity_statistics(
                values, class_width=head_class_widths[name]
            )
            for name, values in sorted(heads.items())
        },
        "same_forced_code_17_family_mean_logit_variation": {
            name: _family_row_difference(
                values[17] if 17 in values else values[sorted(values)[0]]
            )
            for name, values in sorted(family_outputs.items())
        },
        "distinct_complete_discrete_predictions": len(
            set(complete_predictions)
        ),
    }


def _masked_family_rows(tensor, mask):
    return [
        tensor[index][mask[index]].detach().cpu().reshape(-1).tolist()
        for index in range(tensor.size(0))
    ]


def _decoder_supervision_masks(
    target, max_operations, dense_edge_targets, torch_module
):
    node_mask = target["node_mask"]
    presence, _, valid_pairs = dense_edge_targets(
        target, node_mask.size(1)
    )
    pointer_slots = torch_module.zeros(
        node_mask.size(0),
        max_operations,
        dtype=torch_module.bool,
        device=node_mask.device,
    )
    operation_width = min(
        target["operation_mask"].size(1), max_operations
    )
    pointer_slots[:, :operation_width] = target[
        "operation_mask"
    ][:, :operation_width]
    masks = {
        "node_type": node_mask,
        "geometry": target["geometry_mask"] & node_mask.unsqueeze(-1),
        "edge_presence": valid_pairs,
        "edge_type": presence & valid_pairs,
        "operation_pointer": pointer_slots,
    }
    for index in range(target["categorical_attributes"].size(-1)):
        masks["categorical_{}".format(index)] = node_mask
    return masks


def _gradient_probe(
    payload,
    model_config,
    training_config,
    examples,
    checkpoint_record,
    torch_module,
    torch_batch,
    make_data_loader,
    loss_function,
    model_type,
    checkpoint_path,
):
    random.seed(training_config.seed)
    torch_module.manual_seed(training_config.seed)
    model = model_type(model_config)
    model.load_state_dict(payload["model_state"], strict=True)
    model.train()
    loader = make_data_loader(
        examples[: training_config.batch_size],
        min(training_config.batch_size, len(examples)),
        0,
        training_config.seed,
        False,
        torch_module,
    )
    batch = next(iter(loader))
    inputs, target = torch_batch(batch, torch_module.device("cpu"), torch_module)
    for parameter in model.parameters():
        parameter.grad = None
    output = model(target=target, **inputs)
    losses = loss_function(output, target, model_config)
    losses.total.backward()
    parameters = []
    module_groups = {}
    for name, parameter in model.named_parameters():
        gradient = (
            None
            if parameter.grad is None
            else parameter.grad.detach().cpu().reshape(-1).tolist()
        )
        parameters.append((name, gradient))
        group = _parameter_group(name)
        module_groups.setdefault(group, []).append((name, gradient))
    after_digest = checkpoint_sha256(checkpoint_path)
    if after_digest != checkpoint_record["sha256"]:
        raise DiagnosisError(
            "checkpoint_mutation", "gradient probe changed checkpoint bytes"
        )
    return {
        "checkpoint_sha256": checkpoint_record["sha256"],
        "checkpoint_relative_path": checkpoint_record["relative_path"],
        "fixed_seed": training_config.seed,
        "family_ids": list(batch.family_ids),
        "loss_components": {
            name: float(value.detach().item())
            for name, value in losses.as_dict().items()
        },
        "all_parameters": aggregate_gradients(parameters),
        "module_groups": {
            name: aggregate_gradients(values)
            for name, values in sorted(module_groups.items())
        },
        "quantizer_update_mechanism": (
            "EMA buffers mutate on isolated training-mode forward; "
            "codebook embedding is not a parameter"
        ),
        "codebook_parameter_count": 0,
        "codebook_gradient_status": "not_applicable_ema_buffer",
        "optimizer_created": False,
        "optimizer_step_called": False,
        "source_checkpoint_sha256_after_probe": after_digest,
        "backward_loss": "flat_mixed_vq_loss.total",
        "disposable_model_instance": True,
        "training_mode_ema_buffer_updates_isolated": True,
    }


def _parameter_group(name):
    if name.startswith("to_codebook."):
        return "pre_vq_projection"
    if name.startswith("from_codebook."):
        return "memory_projection"
    if name.startswith((
        "encoder.",
        "field_embeddings.",
        "geometry_projection.",
        "geometry_mask_projection.",
        "node_position_embedding.",
        "input_norm.",
        "latent_queries",
    )):
        return "encoder"
    if name.startswith((
        "decoder.",
        "decoder_position_embedding.",
        "decoder_input_norm.",
        "bos",
    )):
        return "decoder"
    if name.startswith((
        "node_type_head.", "categorical_heads.", "geometry_head.",
        "edge_", "operation_",
    )):
        return "decoder_heads"
    return "other"


def _history_summary(run_dir, inventory):
    parsed = []
    for record in inventory:
        if record["parse_status"] != "parsed":
            continue
        path = Path(run_dir) / record["relative_path"]
        events = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line
        ]
        parsed.extend(events)
    fields = sorted({
        key for event in parsed if isinstance(event, dict) for key in event
    })
    epoch_events = []
    for event in parsed:
        if not isinstance(event, dict) or event.get("event") not in (
            "train_epoch", "validation_epoch"
        ):
            continue
        epoch_events.append({
            name: event.get(name)
            for name in (
                "event",
                "epoch",
                "global_step",
                "learning_rate",
                "number_of_examples",
                "metrics",
                "codebook",
            )
        })
    epoch_events.sort(key=lambda item: (
        item["epoch"],
        item["event"],
        item["global_step"],
    ))
    metadata_events = [
        event for event in parsed
        if isinstance(event, dict) and event.get("event") == "run_metadata"
    ]
    selection_metrics = sorted({
        event.get("training_config", {}).get("checkpoint_selection_metric")
        for event in metadata_events
        if isinstance(event.get("training_config"), dict)
        and event["training_config"].get("checkpoint_selection_metric")
        is not None
    })
    return {
        "event_count": len(parsed),
        "recorded_top_level_fields": fields,
        "has_component_losses": any(
            isinstance(event.get("metrics"), dict) for event in parsed
        ),
        "has_codebook_utilization": any(
            isinstance(event.get("codebook"), dict)
            and "codebook_utilization" in event["codebook"]
            for event in parsed
        ),
        "has_codebook_perplexity": any(
            isinstance(event.get("codebook"), dict)
            and "codebook_perplexity" in event["codebook"]
            for event in parsed
        ),
        "has_learning_rate": any("learning_rate" in event for event in parsed),
        "historical_commitment_loss_available": any(
            isinstance(event.get("metrics"), dict)
            and "vq_commitment" in event["metrics"]
            for event in parsed
        ),
        "checkpoint_selection_metrics": selection_metrics,
        "recorded_epoch_events": epoch_events,
    }


def classify_root_causes(
    prequant,
    codebook,
    sensitivity,
    losses,
    gradients,
    inventory,
    history,
):
    latest_prequant = prequant[-1]
    latest_codebook = codebook[-1]
    encoder_constant = (
        latest_prequant["vectors"]["numerically_distinct_vector_count"] <= 1
    )
    one_code = latest_prequant["latent_statistics"]["active_code_count"] == 1
    histogram = latest_prequant["latent_statistics"]["histogram"]
    dominant_code = (
        max(range(len(histogram)), key=lambda index: histogram[index])
        if histogram and sum(histogram)
        else None
    )
    codebook_collapsed = (
        latest_codebook["numerically_distinct_vector_count"] <= 1
    )
    decoder_sensitive = any(
        head["materially_sensitive"] is True
        for record in sensitivity[-2:]
        for head in record["head_statistics"].values()
    )
    encoder_gradient = gradients[-1]["module_groups"].get("encoder", {})
    projection_gradient = gradients[-1]["module_groups"].get(
        "pre_vq_projection", {}
    )
    encoder_gradient_present = (
        encoder_gradient.get("aggregate_norm", 0.0) > 0.0
        and encoder_gradient.get("nonfinite_gradient_count", 0) == 0
    )
    projection_gradient_present = (
        projection_gradient.get("aggregate_norm", 0.0) > 0.0
        and projection_gradient.get("nonfinite_gradient_count", 0) == 0
    )
    gradient_present = (
        encoder_gradient_present and projection_gradient_present
    )
    timeline_count = sum(record["timeline_usable"] for record in inventory)
    history_codebook_timeline = (
        len(history.get("recorded_epoch_events", ())) > 1
        and all(
            isinstance(event.get("codebook"), dict)
            for event in history["recorded_epoch_events"]
        )
    )
    time_identifiable = timeline_count > 1 or history_codebook_timeline
    latest_loss = losses[-1]
    proportions = latest_loss.get("proportions", {})
    majority = latest_loss.get("majority_class_baseline_accuracy", {})
    imbalance_evidence = []
    imbalanced_heads = sorted(
        name
        for name, value in majority.items()
        if name != "edge_presence"
        and value is not None
        and value >= 0.8
    )
    if imbalanced_heads:
        imbalance_evidence.append(
            "unbalanced-loss heads with >=80% majority-class frequency: "
            + ",".join(imbalanced_heads)
        )
    if (
        proportions.get(
            "geometry_applicable_within_supervised_nodes_proportion"
        ) is not None
        and proportions[
            "geometry_applicable_within_supervised_nodes_proportion"
        ] < 0.25
    ):
        imbalance_evidence.append(
            "fewer than 25% of geometry channels are applicable"
        )
    contributions = latest_loss.get("relative_weighted_contribution", {})
    dominant_loss = (
        max(
            contributions,
            key=lambda name: (
                -1.0 if contributions[name] is None else contributions[name]
            ),
        )
        if contributions else None
    )
    if one_code and dominant_code == 17:
        if encoder_constant:
            code_17_explanation = (
                "effectively constant prequant vectors select code 17; inspect "
                "assigned distances to distinguish encoder from code placement"
            )
        elif codebook_collapsed:
            code_17_explanation = (
                "varied prequant vectors encounter an effectively collapsed "
                "codebook whose deterministic argmin selects code 17"
            )
        else:
            code_17_explanation = (
                "varied prequant vectors all lie in code 17's nearest-neighbor "
                "region; assignment margins and code-17 distances quantify it"
            )
    else:
        code_17_explanation = (
            "code 17 is not the sole dominant code in this checkpoint record"
        )
    rows = []

    def row(name, supporting, against, confidence):
        rows.append({
            "hypothesis": name,
            "evidence_supporting": supporting,
            "evidence_against": against,
            "confidence": confidence,
        })

    row(
        "encoder-output collapse",
        ["prequant_summary.records[-1].vectors"] if encoder_constant else [],
        [] if encoder_constant else ["prequant vectors are numerically varied"],
        "high" if encoder_constant else "medium",
    )
    row(
        "nearest-code assignment collapse",
        ["one active code with varied prequant vectors"]
        if one_code and not encoder_constant else [],
        [] if one_code else ["multiple codes are active"],
        "high" if one_code and not encoder_constant else "low",
    )
    row(
        "codebook-embedding collapse",
        ["codebook entries are numerically identical"] if codebook_collapsed else [],
        [] if codebook_collapsed else ["codebook entries remain distinguishable"],
        "high" if codebook_collapsed else "medium",
    )
    row(
        "decoder ignores latent",
        [] if decoder_sensitive else ["decoder intervention changes are below tolerance"],
        ["one or more decoder heads respond to code changes"]
        if decoder_sensitive else [],
        "high" if not decoder_sensitive else "medium",
    )
    row(
        "loss/mask imbalance",
        imbalance_evidence,
        [
            "padding is excluded by node/geometry masks",
            "edge presence is explicitly balanced between present and absent",
        ],
        "medium" if imbalance_evidence else "low",
    )
    row(
        "gradient-flow failure",
        [] if gradient_present else ["encoder gradient is absent, zero, or nonfinite"],
        ["encoder receives finite nonzero gradient"] if gradient_present else [],
        "medium",
    )
    row(
        "training-time collapse",
        (
            ["multiple compatible checkpoints permit a state timeline"]
            if timeline_count > 1 else []
        ) + (
            ["epoch logs contain a codebook-utilization timeline"]
            if history_codebook_timeline else []
        ),
        (
            ["neither multiple compatible checkpoints nor codebook epoch logs"]
            if not time_identifiable else []
        ),
        "medium" if time_identifiable else "unresolved",
    )
    supported_mechanisms = []
    if encoder_constant:
        supported_mechanisms.append("encoder-output collapse")
    if one_code and not encoder_constant:
        supported_mechanisms.append("nearest-code assignment collapse")
    if codebook_collapsed:
        supported_mechanisms.append("codebook-embedding collapse")
    if not decoder_sensitive:
        supported_mechanisms.append("decoder ignores latent")
    if imbalance_evidence:
        supported_mechanisms.append("loss/mask imbalance")
    if not gradient_present:
        supported_mechanisms.append("gradient-flow failure")
    if time_identifiable:
        supported_mechanisms.append("training-time collapse")
    combined = len(supported_mechanisms) > 1
    row(
        "unresolved or combined mechanisms",
        (
            ["multiple mechanisms have direct supporting observations"]
            if combined else []
        ),
        (
            ["one mechanism has direct support"]
            if len(supported_mechanisms) == 1 else []
        ),
        "high" if combined else "unresolved",
    )
    if tuple(item["hypothesis"] for item in rows) != HYPOTHESES:
        raise DiagnosisError("root_cause_rows", "hypothesis ordering differs")
    return {
        "observations": {
            "encoder_outputs_effectively_constant": encoder_constant,
            "single_active_code": one_code,
            "codebook_effectively_constant": codebook_collapsed,
            "decoder_materially_sensitive": decoder_sensitive,
            "encoder_finite_nonzero_gradient": gradient_present,
            "encoder_module_finite_nonzero_gradient": (
                encoder_gradient_present
            ),
            "pre_vq_projection_finite_nonzero_gradient": (
                projection_gradient_present
            ),
            "usable_checkpoint_count": timeline_count,
            "history_codebook_timeline_available": history_codebook_timeline,
            "history_observability": history,
        },
        "hypotheses": rows,
        "answers": {
            "encoder_collapsed_before_quantization": (
                "yes" if encoder_constant else "no"
            ),
            "codebook_embeddings_collapsed": (
                "yes" if codebook_collapsed else "no"
            ),
            "why_code_17": (
                code_17_explanation
            ),
            "decoder_responds_to_latent_changes": decoder_sensitive,
            "loss_explanation": {
                "dominant_weighted_component": dominant_loss,
                "imbalance_evidence": imbalance_evidence,
                "padding_directly_supervised": False,
                "edge_presence_balanced": True,
            },
            "gradient_reached_encoder": gradient_present,
            "supported_mechanisms": supported_mechanisms,
            "combined_mechanisms_supported": combined,
            "gradient_reached_quantizer": (
                "not applicable: codebook is an EMA buffer updated from "
                "training-mode assignments, not an optimizer parameter"
            ),
            "collapse_time_identifiable": time_identifiable,
            "unavailable_evidence": (
                []
                if timeline_count > 1
                else ["intermediate compatible model states"]
            ),
            "smallest_retraining_experiment": (
                "not implemented; run a bounded train-only seed-controlled "
                "experiment instrumented with per-epoch prequant variance, "
                "assignment entropy, EMA codebook distances, and decoder "
                "sensitivity after review of this diagnosis"
            ),
            "retraining_gates": {
                "active_code_count_minimum": 2,
                "perplexity_minimum": 2.0,
                "prequant_distinct_vector_count_minimum": 2,
                "finite_nonzero_encoder_gradient_required": True,
                "decoder_material_sensitivity_required": True,
                "test_partition_evaluated": False,
            },
        },
        "test_partition_evaluated": False,
    }


def distribution(values):
    converted = [float(value) for value in values]
    finite = sorted(value for value in converted if math.isfinite(value))
    if not finite:
        result = _empty_distribution()
        result["nonfinite_count"] = len(converted)
        return result
    return {
        "count": len(finite),
        "nonfinite_count": len(converted) - len(finite),
        "minimum": finite[0],
        "maximum": finite[-1],
        "mean": sum(finite) / len(finite),
        "median": _median(finite),
    }


def _empty_distribution():
    return {
        "count": 0,
        "nonfinite_count": 0,
        "minimum": None,
        "maximum": None,
        "mean": None,
        "median": None,
    }


def _median(values):
    middle = len(values) // 2
    if len(values) % 2:
        return values[middle]
    return (values[middle - 1] + values[middle]) / 2.0


def _tolerance_bucket(value, tolerance):
    if not math.isfinite(value):
        if math.isnan(value):
            label = "nan"
        elif value > 0:
            label = "positive_infinity"
        else:
            label = "negative_infinity"
        return ("nonfinite", label)
    return int(round(value / tolerance))


def _binary_decision(value):
    return int(value >= 0.0)


def _intervention_decisions(values, class_width):
    if class_width is None:
        return None
    if class_width == 1:
        return [
            tuple(_binary_decision(value) for value in row)
            for row in values
        ]
    if (
        isinstance(class_width, bool)
        or not isinstance(class_width, int)
        or class_width <= 1
        or any(len(row) % class_width for row in values)
    ):
        raise DiagnosisError(
            "intervention_shape", "class width does not divide logits"
        )
    decisions = []
    for row in values:
        current = []
        for start in range(0, len(row), class_width):
            chunk = row[start : start + class_width]
            current.append(max(
                range(class_width), key=lambda index: chunk[index]
            ))
        decisions.append(tuple(current))
    return decisions


def _code_reference_difference(values, codes, reference_code):
    if reference_code not in codes or len(codes) <= 1:
        return None
    reference = values[codes.index(reference_code)]
    return max(
        abs(value - reference[index])
        for code, row in zip(codes, values)
        if code != reference_code
        for index, value in enumerate(row)
    )


def _family_row_difference(rows):
    means = [
        sum(float(value) for value in row) / len(row)
        if row else None
        for row in rows
    ]
    differences = [
        abs(left - right)
        for left_index, left in enumerate(means)
        if left is not None
        for right in means[left_index + 1 :]
        if right is not None
    ]
    result = distribution(differences)
    result["family_count"] = len(rows)
    result["family_with_values_count"] = sum(
        value is not None for value in means
    )
    result["statistic"] = "pairwise_absolute_difference_of_family_mean_logits"
    return result


def _strict_json(path):
    try:
        content = Path(path).read_bytes()
        value = json.loads(
            content.decode("utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(value)
            ),
        )
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise DiagnosisError("invalid_json", Path(path).name) from exc
    if content != _json_document(value):
        raise DiagnosisError("noncanonical_json", Path(path).name)
    return value


def _require_identical_smoke_replay(first, second):
    if first["artifact_sha256"] != second["artifact_sha256"]:
        raise DiagnosisError(
            "replay_mismatch",
            "smoke diagnostic artifact bytes are not identical",
        )
    return True


def build_final_report(full_result, smoke_first, smoke_second):
    return {
        "diagnostic_schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        "artifact_sha256": full_result["artifact_sha256"],
        "root_cause_answers": full_result["root_cause_answers"],
        "byte_identical_smoke_replay": _require_identical_smoke_replay(
            smoke_first, smoke_second
        ),
        "full_diagnosis_verified": True,
        "test_partition_evaluated": False,
    }


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--corpus-dir", required=True)
    run.add_argument("--training-run-dir", required=True)
    run.add_argument("--output-dir", required=True)
    run.add_argument("--split-manifest", default="iid")
    run.add_argument("--reviewed-commit", required=True)
    run.add_argument("--batch-size", type=int, default=32)
    run.add_argument("--train-limit", type=int)
    run.add_argument("--validation-limit", type=int)
    run.add_argument("--sensitivity-family-limit", type=int, default=4)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--output", required=True)
    verify.add_argument("--corpus-dir", required=True)
    verify.add_argument("--training-run-dir", required=True)
    verify.add_argument("--split-manifest", default="iid")
    verify.add_argument("--reviewed-commit", required=True)
    verify.add_argument("--compare-output")
    verify.add_argument("--report-output")
    verify.add_argument("--smoke-output-a")
    verify.add_argument("--smoke-output-b")
    return parser


def main(argv=None):
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "run":
            result = run_diagnosis(arguments)
        else:
            authoritative = tuple(load_physical_examples(
                arguments.corpus_dir, arguments.split_manifest
            ))
            result = validate_diagnostic_artifacts(
                arguments.output,
                authoritative_examples=authoritative,
                training_run_dir=arguments.training_run_dir,
                reviewed_commit=arguments.reviewed_commit,
            )
            if arguments.compare_output:
                other = validate_diagnostic_artifacts(
                    arguments.compare_output,
                    authoritative_examples=authoritative,
                    training_run_dir=arguments.training_run_dir,
                    reviewed_commit=arguments.reviewed_commit,
                )
                result["byte_identical_smoke_replay"] = (
                    _require_identical_smoke_replay(result, other)
                )
            if arguments.report_output:
                if not (
                    arguments.smoke_output_a
                    and arguments.smoke_output_b
                ):
                    raise DiagnosisError(
                        "missing_smoke_replay",
                        "final report requires both smoke outputs",
                    )
                smoke_results = [
                    validate_diagnostic_artifacts(
                        output,
                        authoritative_examples=authoritative,
                        training_run_dir=arguments.training_run_dir,
                        reviewed_commit=arguments.reviewed_commit,
                    )
                    for output in (
                        arguments.smoke_output_a,
                        arguments.smoke_output_b,
                    )
                ]
                report = build_final_report(
                    result, smoke_results[0], smoke_results[1]
                )
                publish_json_report(arguments.report_output, report)
                result = report
    except (DiagnosisError, OSError, TypeError, ValueError) as exc:
        sys.stderr.write("vq_diagnosis_failure: {}\n".format(exc))
        return 1
    sys.stdout.write(json.dumps(
        result, sort_keys=True, separators=(",", ":"), allow_nan=False
    ) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
