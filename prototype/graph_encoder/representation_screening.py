"""Short artifact-only screening over preserved GE1 representation features.

This module is deliberately narrower than accepted ADR-0011.  It verifies the
three immutable files preserved from timed-out job 3346513 and runs only the
existing non-permuted real-label probe pipelines.  It cannot extract features,
load a corpus or checkpoint, construct a GE1 model, or authorize a repair.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
import os
from pathlib import Path
import platform
import socket
import sys
import time

try:
    import torch
except ImportError:  # Pure contract tests remain available without PyTorch.
    torch = None

from .errors import GraphEncoderError
from .optimization_diagnostic import validate_slurm_job_id
from .pilot import (
    _atomic_write_bytes,
    _atomic_write_json,
    _atomic_write_jsonl,
    _file_sha256,
    _is_within,
    _read_canonical_jsonl,
    _safe_relative_path,
    _source_identity,
    _verify_source_unchanged,
)
from .representation_probe import (
    FEATURE_DIMENSIONS,
    FEATURE_LEVELS,
    LINEAR_HISTORY_SIZE,
    LINEAR_MAX_ITER,
    LINEAR_PARAMETER_COUNTS,
    LINEAR_TOLERANCE_CHANGE,
    LINEAR_TOLERANCE_GRAD,
    MLP_L2,
    MLP_LEARNING_RATE,
    MLP_PARAMETER_COUNTS,
    MLP_UPDATES,
    MLP_WIDTH,
    OPERATION_GRIDS,
    OPERATION_TYPES,
    PROBE_ARMS,
    PROBE_FEATURE_VERSION,
    PROBE_KINDS,
    PROBE_LABEL_VERSION,
    PROBE_PROTOCOL_VERSION,
    REGULARIZATION_GRID,
    REPAIRED_SEED,
    SCALED_EXTRUSION_COUNT,
    SCALED_FAMILY_COUNT,
    SCALED_OPERATION_COUNT,
    SCALED_REVOLVE_COUNT,
    fit_probe_pipeline,
    validate_feature_rows,
)


SCREEN_PROTOCOL_VERSION = "GE1-C7-REPAIRED-REPRESENTATION-SCREEN-v1"
SCREEN_ARTIFACT_VERSION = (
    "GE1-C7-REPAIRED-REPRESENTATION-SCREEN-ARTIFACT-v1"
)
SCREEN_RESULT_VERSION = "GE1-C7-REPAIRED-REPRESENTATION-SCREEN-RESULTS-v1"
SOURCE_FEATURE_COMMIT = "3ea43650bb793d8d55b3d61a1edb70ec7a920089"
SOURCE_FEATURE_JOB_ID = "3346513"
SCREEN_INPUT_HASHES = {
    "detached_features.pt": (
        "87dab4841f64f0c1b27db6676852da7cd980ba1bde61e81809fde3af745cc01c"
    ),
    "feature_manifest.json": (
        "29220bda0677636627069ce6565bf367065f1ceaa3c2d86d3174058838598450"
    ),
    "labels.json": (
        "70f82727ebe35613bcbae1267924d9a327d9744a757b6d076c8028e261cfe71f"
    ),
}
SCREEN_PIPELINE_COUNT = 16
SCREEN_CANDIDATE_ACCURACY_MIN = 0.40
SCREEN_CANDIDATE_BALANCED_MIN = 0.40
SCREEN_EXPECTED_ORDINARY_FILES = (
    "metrics.jsonl",
    "resolved_config.json",
    "screening_results.json",
)
SCREEN_EXPECTED_FILES = SCREEN_EXPECTED_ORDINARY_FILES + (
    "artifact_manifest.json",
    "SHA256SUMS",
)
LABEL_ROW_FIELDS = frozenset({
    "class_index",
    "class_physical_value",
    "family_id",
    "label_key",
    "operation_index",
    "operation_template",
    "operation_type",
})
SCREEN_COMBINATIONS = tuple(itertools.product(
    PROBE_ARMS, OPERATION_TYPES, FEATURE_LEVELS, PROBE_KINDS
))


def screen_contract():
    """Return the frozen preliminary-screening contract."""

    return {
        "protocol_version": SCREEN_PROTOCOL_VERSION,
        "source_feature_artifact": {
            "git_commit": SOURCE_FEATURE_COMMIT,
            "slurm_job_id": SOURCE_FEATURE_JOB_ID,
            "full_adr0011_probe_completed": False,
            "sha256": copy.deepcopy(SCREEN_INPUT_HASHES),
        },
        "input": {
            "feature_schema": PROBE_FEATURE_VERSION,
            "label_schema": PROBE_LABEL_VERSION,
            "feature_rows": 96,
            "label_rows": 48,
            "physical_families": 32,
            "feature_dimensions": dict(FEATURE_DIMENSIONS),
            "extrusion_operations": 32,
            "revolve_operations": 16,
        },
        "pipeline": {
            "count": SCREEN_PIPELINE_COUNT,
            "arms": list(PROBE_ARMS),
            "operation_types": list(OPERATION_TYPES),
            "features": list(FEATURE_LEVELS),
            "probes": list(PROBE_KINDS),
            "production_function": "representation_probe.fit_probe_pipeline",
            "label_permutations": 0,
            "physical_family_lofo": True,
            "nested_linear_regularization_selection": True,
            "train_fold_only_standardization": True,
            "linear": {
                "regularization_grid": list(REGULARIZATION_GRID),
                "max_iter": LINEAR_MAX_ITER,
                "history_size": LINEAR_HISTORY_SIZE,
                "tolerance_grad": LINEAR_TOLERANCE_GRAD,
                "tolerance_change": LINEAR_TOLERANCE_CHANGE,
                "line_search_fn": "strong_wolfe",
                "dtype": "float64",
                "parameter_counts": dict(LINEAR_PARAMETER_COUNTS),
            },
            "mlp": {
                "hidden_width": MLP_WIDTH,
                "activation": "tanh",
                "l2_weights_only": MLP_L2,
                "optimizer": "Adam",
                "learning_rate": MLP_LEARNING_RATE,
                "updates": MLP_UPDATES,
                "base_seed": REPAIRED_SEED,
                "parameter_counts": dict(MLP_PARAMETER_COUNTS),
            },
        },
        "candidate_flag": {
            "lofo_accuracy_min": SCREEN_CANDIDATE_ACCURACY_MIN,
            "lofo_balanced_accuracy_min": SCREEN_CANDIDATE_BALANCED_MIN,
            "purpose": "descriptive_prioritization_for_full_controls_only",
            "statistical_significance": "unavailable_without_permutations",
        },
        "authority": {
            "completes_or_replaces_adr0011": False,
            "formal_representation_conclusion": False,
            "authorizes_model_or_decoder_repair": False,
            "authorizes_stage6": False,
            "authorizes_c8": False,
        },
    }


def screening_access_declarations(*, input_accessed=False, completed=False):
    return {
        "preserved_feature_label_artifact_accessed": bool(input_accessed),
        "corpus_accessed": False,
        "operation_template_manifest_accessed": False,
        "source_repaired_artifact_accessed": False,
        "source_repaired_checkpoint_accessed": False,
        "development_accessed": False,
        "systematic_rr_accessed": False,
        "test_er_accessed": False,
        "iid_accessed": False,
        "history_depth_accessed": False,
        "geometry_extrapolation_accessed": False,
        "other_corpus_accessed": False,
        "ge1_model_constructed_or_loaded": False,
        "ge1_encoder_or_decoder_constructed_or_loaded": False,
        "ge1_training_performed": False,
        "ge1_backward_pass_performed": False,
        "ge1_optimizer_constructed": False,
        "ge1_checkpoint_written": False,
        "disposable_probe_optimization_performed": bool(completed),
        "label_permutations_performed": False,
        "model_or_decoder_repair_implemented_or_invoked": False,
        "categorical_head_implemented": False,
        "geometry_loss_changed": False,
        "calibration_repair_implemented": False,
        "stage6_performed": False,
        "c8_or_later_performed": False,
        "representation_screening_completed": bool(completed),
        "formal_adr0011_representation_probe_completed": False,
    }


def candidate_for_full_controls(lofo_metrics):
    if not isinstance(lofo_metrics, dict):
        raise GraphEncoderError(
            "invalid_screening_metrics", "LOFO metrics must be an object"
        )
    values = []
    for name in ("accuracy", "balanced_accuracy"):
        value = lofo_metrics.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) <= 1.0
        ):
            raise GraphEncoderError(
                "invalid_screening_metrics", "{} is invalid".format(name)
            )
        values.append(float(value))
    return (
        values[0] >= SCREEN_CANDIDATE_ACCURACY_MIN
        and values[1] >= SCREEN_CANDIDATE_BALANCED_MIN
    )


def _validate_input_hashes(observed):
    if dict(observed) != SCREEN_INPUT_HASHES:
        raise GraphEncoderError(
            "screening_input_hash_mismatch", "preserved input hashes differ"
        )
    return dict(observed)


def _validate_labels(payload):
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version",
        "labels_joined_after_ge1_models_released",
        "grouping_unit",
        "class_grids",
        "rows",
    }:
        raise GraphEncoderError(
            "invalid_screening_labels", "label payload fields differ"
        )
    if (
        payload["schema_version"] != PROBE_LABEL_VERSION
        or payload["labels_joined_after_ge1_models_released"] is not True
        or payload["grouping_unit"] != "physical_family"
        or payload["class_grids"] != {
            name: list(values) for name, values in OPERATION_GRIDS.items()
        }
        or not isinstance(payload["rows"], list)
        or len(payload["rows"]) != SCALED_OPERATION_COUNT
    ):
        raise GraphEncoderError(
            "invalid_screening_labels", "label identity or count differs"
        )
    rows = tuple(payload["rows"])
    if len({item.get("label_key") for item in rows}) != len(rows):
        raise GraphEncoderError(
            "invalid_screening_labels", "label keys are not unique"
        )
    families = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != LABEL_ROW_FIELDS:
            raise GraphEncoderError(
                "invalid_screening_labels", "label row fields differ"
            )
        operation_type = row.get("operation_type")
        operation_index = row.get("operation_index")
        class_index = row.get("class_index")
        if (
            operation_type not in OPERATION_TYPES
            or operation_index not in (0, 1)
            or isinstance(operation_index, bool)
            or isinstance(class_index, bool)
            or not isinstance(class_index, int)
            or class_index not in range(5)
            or row.get("operation_template") not in ("E", "R", "EE", "RE")
            or not isinstance(row.get("family_id"), str)
            or not row["family_id"]
            or row.get("label_key") != "{}:{}:{}".format(
                row["family_id"], operation_index, operation_type
            )
            or float(row.get("class_physical_value"))
            != float(OPERATION_GRIDS[operation_type][class_index])
        ):
            raise GraphEncoderError(
                "invalid_screening_labels", "label row is malformed"
            )
        families.setdefault(row["family_id"], []).append(row)
    if (
        len(families) != SCALED_FAMILY_COUNT
        or sum(item["operation_type"] == "extrude" for item in rows)
        != SCALED_EXTRUSION_COUNT
        or sum(item["operation_type"] == "revolve" for item in rows)
        != SCALED_REVOLVE_COUNT
    ):
        raise GraphEncoderError(
            "invalid_screening_labels", "label family or operation coverage differs"
        )
    expected_by_template = {
        "E": ((0, "extrude"),),
        "R": ((0, "revolve"),),
        "EE": ((0, "extrude"), (1, "extrude")),
        "RE": ((0, "revolve"), (1, "extrude")),
    }
    template_counts = {name: 0 for name in expected_by_template}
    for family_rows in families.values():
        templates = {item["operation_template"] for item in family_rows}
        if len(templates) != 1:
            raise GraphEncoderError(
                "invalid_screening_labels", "family template differs"
            )
        template = templates.pop()
        observed = tuple(sorted(
            (item["operation_index"], item["operation_type"])
            for item in family_rows
        ))
        if observed != expected_by_template[template]:
            raise GraphEncoderError(
                "invalid_screening_labels", "family operation pattern differs"
            )
        template_counts[template] += 1
    if template_counts != {"E": 8, "R": 8, "EE": 8, "RE": 8}:
        raise GraphEncoderError(
            "invalid_screening_labels", "template coverage differs"
        )
    return rows


def _validate_feature_manifest(payload, feature_path):
    expected_fields = {
        "schema_version",
        "protocol_version",
        "feature_file",
        "feature_file_byte_size",
        "feature_file_sha256",
        "written_and_hashed_before_label_join",
        "target_or_label_content_present",
        "model_state_present",
        "optimizer_state_present",
        "row_count",
        "feature_dimensions",
    }
    if not isinstance(payload, dict) or set(payload) != expected_fields:
        raise GraphEncoderError(
            "invalid_screening_feature_manifest", "manifest fields differ"
        )
    if (
        payload["schema_version"] != PROBE_FEATURE_VERSION
        or payload["protocol_version"] != PROBE_PROTOCOL_VERSION
        or payload["feature_file"] != "detached_features.pt"
        or payload["feature_file_byte_size"] != feature_path.stat().st_size
        or payload["feature_file_sha256"] != SCREEN_INPUT_HASHES[
            "detached_features.pt"
        ]
        or payload["written_and_hashed_before_label_join"] is not True
        or payload["target_or_label_content_present"] is not False
        or payload["model_state_present"] is not False
        or payload["optimizer_state_present"] is not False
        or payload["row_count"] != 96
        or payload["feature_dimensions"] != FEATURE_DIMENSIONS
    ):
        raise GraphEncoderError(
            "invalid_screening_feature_manifest", "manifest contract differs"
        )
    return payload


def _validate_feature_label_alignment(feature_rows, label_rows):
    labels = {item["label_key"]: item for item in label_rows}
    grouped = {}
    for feature in feature_rows:
        label = labels.get(feature["label_key"])
        if label is None or any(
            feature[name] != label[name]
            for name in ("family_id", "operation_index", "operation_type")
        ):
            raise GraphEncoderError(
                "screening_input_alignment_failure",
                "feature and label identity differs",
            )
        grouped.setdefault(feature["label_key"], []).append(feature["arm"])
    if (
        set(grouped) != set(labels)
        or any(tuple(sorted(arms)) != PROBE_ARMS for arms in grouped.values())
    ):
        raise GraphEncoderError(
            "screening_input_alignment_failure",
            "each label requires one row per arm",
        )
    return True


def load_screening_inputs(input_dir):
    """Verify and load only the three frozen feature/label input files."""

    if torch is None:
        raise RuntimeError("representation screening requires PyTorch")
    root = Path(input_dir)
    if not root.is_dir() or root.is_symlink():
        raise GraphEncoderError(
            "invalid_screening_input", "input root must be a real directory"
        )
    paths = {name: root / name for name in SCREEN_INPUT_HASHES}
    if any(not path.is_file() or path.is_symlink() for path in paths.values()):
        raise GraphEncoderError(
            "invalid_screening_input", "required input file is absent or a symlink"
        )
    observed_hashes = _validate_input_hashes({
        name: _file_sha256(path) for name, path in paths.items()
    })
    manifest = json.loads(paths["feature_manifest.json"].read_text("utf-8"))
    _validate_feature_manifest(manifest, paths["detached_features.pt"])
    labels = json.loads(paths["labels.json"].read_text("utf-8"))
    label_rows = _validate_labels(labels)
    features = torch.load(
        str(paths["detached_features.pt"]), map_location="cpu"
    )
    if not isinstance(features, dict) or set(features) != {
        "schema_version",
        "protocol_version",
        "target_free",
        "feature_dimensions",
        "rows",
    }:
        raise GraphEncoderError(
            "invalid_screening_features", "feature payload fields differ"
        )
    if (
        features["schema_version"] != PROBE_FEATURE_VERSION
        or features["protocol_version"] != PROBE_PROTOCOL_VERSION
        or features["target_free"] is not True
        or features["feature_dimensions"] != FEATURE_DIMENSIONS
    ):
        raise GraphEncoderError(
            "invalid_screening_features", "feature payload identity differs"
        )
    feature_rows = validate_feature_rows(features["rows"])
    _validate_feature_label_alignment(feature_rows, label_rows)
    return {
        "feature_rows": feature_rows,
        "label_rows": label_rows,
        "input_hashes": observed_hashes,
        "feature_manifest": manifest,
    }


def _joined_rows(feature_rows, label_rows, arm):
    labels = {item["label_key"]: item for item in label_rows}
    result = []
    for feature in feature_rows:
        if feature["arm"] != arm:
            continue
        row = dict(feature)
        label = labels[feature["label_key"]]
        row["operation_template"] = label["operation_template"]
        row["class_index"] = label["class_index"]
        result.append(row)
    return tuple(result)


def _class_support(label_rows):
    result = {}
    for operation_type in OPERATION_TYPES:
        result[operation_type] = [
            sum(
                item["operation_type"] == operation_type
                and item["class_index"] == class_index
                for item in label_rows
            )
            for class_index in range(5)
        ]
    return result


def _validate_metric_record(metrics, expected_count):
    if not isinstance(metrics, dict):
        raise GraphEncoderError(
            "invalid_screening_results", "metrics record is absent"
        )
    for name in ("accuracy", "balanced_accuracy"):
        value = metrics.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) <= 1.0
        ):
            raise GraphEncoderError(
                "invalid_screening_results", "metric value differs"
            )
    if (
        metrics.get("observation_count") != expected_count
        or len(metrics.get("confusion_matrix", ())) != 5
        or any(len(row) != 5 for row in metrics["confusion_matrix"])
        or len(metrics.get("class_support", ())) != 5
        or len(metrics.get("per_class_recall", ())) != 5
    ):
        raise GraphEncoderError(
            "invalid_screening_results", "five-class metric shape differs"
        )


def _validate_pipeline_record(record):
    expected_fields = {
        "arm",
        "operation_type",
        "feature_level",
        "probe_kind",
        "derived_parameter_count",
        "optimization_contract",
        "resubstitution",
        "lofo",
        "candidate_for_full_controls",
        "screening_flag_rule",
        "permutation_significance",
    }
    if not isinstance(record, dict) or set(record) != expected_fields:
        raise GraphEncoderError(
            "invalid_screening_results", "pipeline fields differ"
        )
    identity = (
        record["arm"],
        record["operation_type"],
        record["feature_level"],
        record["probe_kind"],
    )
    if identity not in SCREEN_COMBINATIONS:
        raise GraphEncoderError(
            "invalid_screening_results", "pipeline identity differs"
        )
    expected_count = (
        SCALED_EXTRUSION_COUNT
        if record["operation_type"] == "extrude"
        else SCALED_REVOLVE_COUNT
    )
    _validate_metric_record(record.get("resubstitution", {}).get("metrics"), expected_count)
    _validate_metric_record(record.get("lofo", {}).get("metrics"), expected_count)
    expected_parameters = (
        LINEAR_PARAMETER_COUNTS
        if record["probe_kind"] == "linear" else MLP_PARAMETER_COUNTS
    )[record["feature_level"]]
    if record["derived_parameter_count"] != expected_parameters:
        raise GraphEncoderError(
            "invalid_screening_results", "probe parameter count differs"
        )
    folds = record["lofo"].get("folds")
    if not isinstance(folds, list) or not folds:
        raise GraphEncoderError(
            "invalid_screening_results", "LOFO folds are absent"
        )
    for fold in folds:
        if not isinstance(fold, dict) or set(fold) != {
            "held_out_family_id", "selected_lambda", "inner_selection", "seed", "fit"
        }:
            raise GraphEncoderError(
                "invalid_screening_results", "LOFO fold fields differ"
            )
        if record["probe_kind"] == "linear":
            if fold["selected_lambda"] not in REGULARIZATION_GRID or fold["seed"] is not None:
                raise GraphEncoderError(
                    "invalid_screening_results", "linear fold contract differs"
                )
        elif (
            fold["selected_lambda"] is not None
            or fold["inner_selection"] is not None
            or isinstance(fold["seed"], bool)
            or not isinstance(fold["seed"], int)
        ):
            raise GraphEncoderError(
                "invalid_screening_results", "MLP fold contract differs"
            )
    resubstitution = record["resubstitution"]
    if record["probe_kind"] == "linear":
        if (
            resubstitution.get("selected_lambda") not in REGULARIZATION_GRID
            or resubstitution.get("seed") is not None
            or not isinstance(resubstitution.get("regularization_selection"), list)
        ):
            raise GraphEncoderError(
                "invalid_screening_results", "linear resubstitution differs"
            )
    elif (
        resubstitution.get("selected_lambda") is not None
        or resubstitution.get("regularization_selection") is not None
        or isinstance(resubstitution.get("seed"), bool)
        or not isinstance(resubstitution.get("seed"), int)
    ):
        raise GraphEncoderError(
            "invalid_screening_results", "MLP resubstitution differs"
        )
    expected_flag = candidate_for_full_controls(record["lofo"]["metrics"])
    if (
        record["candidate_for_full_controls"] is not expected_flag
        or record["screening_flag_rule"] != {
            "lofo_accuracy_min": SCREEN_CANDIDATE_ACCURACY_MIN,
            "lofo_balanced_accuracy_min": SCREEN_CANDIDATE_BALANCED_MIN,
        }
        or record["permutation_significance"] != {
            "available": False,
            "reason": "screening_runs_no_label_permutations",
        }
    ):
        raise GraphEncoderError(
            "invalid_screening_results", "screening interpretation differs"
        )
    return identity


def validate_screening_results(results):
    if not isinstance(results, dict) or set(results) != {
        "schema_version",
        "protocol_version",
        "source_feature_artifact",
        "execution_source",
        "environment",
        "input_hashes",
        "class_support",
        "pipeline_count",
        "pipelines",
        "screening_summary",
    }:
        raise GraphEncoderError(
            "invalid_screening_results", "screening result fields differ"
        )
    if (
        results["schema_version"] != SCREEN_RESULT_VERSION
        or results["protocol_version"] != SCREEN_PROTOCOL_VERSION
        or results["source_feature_artifact"] != {
            "git_commit": SOURCE_FEATURE_COMMIT,
            "slurm_job_id": SOURCE_FEATURE_JOB_ID,
            "adr0011_probe_completed": False,
        }
        or results["input_hashes"] != SCREEN_INPUT_HASHES
        or not isinstance(results["execution_source"], dict)
        or not isinstance(results["execution_source"].get("git_commit"), str)
        or not results["execution_source"]["git_commit"]
        or not isinstance(results["execution_source"].get("slurm_job_id"), str)
        or not results["execution_source"]["slurm_job_id"].isdigit()
        or not isinstance(results["environment"], dict)
        or results["environment"].get("python") != "3.8.13"
        or str(results["environment"].get("pytorch", "")).split("+")[0]
        != "1.11.0"
        or results["environment"].get("device") != "cpu"
        or results["environment"].get("cuda_available") is not False
        or results["environment"].get("cpu_threads") != 1
        or results["pipeline_count"] != SCREEN_PIPELINE_COUNT
        or len(results["pipelines"]) != SCREEN_PIPELINE_COUNT
    ):
        raise GraphEncoderError(
            "invalid_screening_results", "screening identity or count differs"
        )
    identities = tuple(_validate_pipeline_record(item) for item in results["pipelines"])
    if identities != SCREEN_COMBINATIONS:
        raise GraphEncoderError(
            "incomplete_screening_pipelines", "all 16 ordered pipelines are required"
        )
    class_support = results.get("class_support")
    if (
        not isinstance(class_support, dict)
        or set(class_support) != set(OPERATION_TYPES)
        or any(
            not isinstance(class_support[name], list)
            or len(class_support[name]) != 5
            or any(isinstance(value, bool) or not isinstance(value, int) or value < 0
                   for value in class_support[name])
            for name in OPERATION_TYPES
        )
        or sum(class_support["extrude"]) != SCALED_EXTRUSION_COUNT
        or sum(class_support["revolve"]) != SCALED_REVOLVE_COUNT
    ):
        raise GraphEncoderError(
            "invalid_screening_results", "class support differs"
        )
    candidate_count = sum(
        item["candidate_for_full_controls"] for item in results["pipelines"]
    )
    if results["screening_summary"] != {
        "candidate_pipeline_count": candidate_count,
        "permutation_significance_available": False,
        "permutation_significance_unavailable_reason": (
            "screening_deliberately_runs_no_label_permutations"
        ),
        "adr0011_representation_probe_complete": False,
        "formal_representation_conclusion_available": False,
        "authorizes_model_or_decoder_repair": False,
        "authorizes_stage6": False,
        "authorizes_c8": False,
    }:
        raise GraphEncoderError(
            "invalid_screening_results", "screening summary differs"
        )
    return results


def run_real_label_screen(feature_rows, label_rows, input_hashes, *,
                          execution_source, environment):
    """Run exactly 16 production probe pipelines without permutations."""

    pipelines = []
    for arm, operation_type, feature_level, probe_kind in SCREEN_COMBINATIONS:
        rows = _joined_rows(feature_rows, label_rows, arm)
        selected = tuple(
            item for item in rows if item["operation_type"] == operation_type
        )
        labels_by_key = {item["key"]: item["class_index"] for item in selected}
        result = fit_probe_pipeline(
            selected,
            labels_by_key,
            arm=arm,
            operation_type=operation_type,
            feature_level=feature_level,
            probe_kind=probe_kind,
        )
        result["candidate_for_full_controls"] = candidate_for_full_controls(
            result["lofo"]["metrics"]
        )
        result["screening_flag_rule"] = {
            "lofo_accuracy_min": SCREEN_CANDIDATE_ACCURACY_MIN,
            "lofo_balanced_accuracy_min": SCREEN_CANDIDATE_BALANCED_MIN,
        }
        result["permutation_significance"] = {
            "available": False,
            "reason": "screening_runs_no_label_permutations",
        }
        pipelines.append(result)
    results = {
        "schema_version": SCREEN_RESULT_VERSION,
        "protocol_version": SCREEN_PROTOCOL_VERSION,
        "source_feature_artifact": {
            "git_commit": SOURCE_FEATURE_COMMIT,
            "slurm_job_id": SOURCE_FEATURE_JOB_ID,
            "adr0011_probe_completed": False,
        },
        "execution_source": dict(execution_source),
        "environment": dict(environment),
        "input_hashes": dict(input_hashes),
        "class_support": _class_support(label_rows),
        "pipeline_count": len(pipelines),
        "pipelines": pipelines,
        "screening_summary": {
            "candidate_pipeline_count": sum(
                item["candidate_for_full_controls"] for item in pipelines
            ),
            "permutation_significance_available": False,
            "permutation_significance_unavailable_reason": (
                "screening_deliberately_runs_no_label_permutations"
            ),
            "adr0011_representation_probe_complete": False,
            "formal_representation_conclusion_available": False,
            "authorizes_model_or_decoder_repair": False,
            "authorizes_stage6": False,
            "authorizes_c8": False,
        },
    }
    return validate_screening_results(results)


def _regular_files(root, excluded=()):
    excluded_values = set(excluded)
    paths = []
    for path in Path(root).rglob("*"):
        if path.is_symlink():
            raise GraphEncoderError(
                "invalid_screening_artifact", "artifact symlinks are forbidden"
            )
        if path.is_file():
            relative = path.relative_to(root).as_posix()
            if relative not in excluded_values:
                paths.append(relative)
    return tuple(sorted(paths))


def prepare_screening_output(output_dir, *, repository_root, input_dir, job_id=None):
    identity = validate_slurm_job_id(job_id)
    raw = Path(output_dir)
    if raw.exists() or raw.is_symlink():
        raise GraphEncoderError(
            "screening_output_exists", "output must be a new non-symlink path"
        )
    final = raw.resolve()
    if any(_is_within(final, Path(root).resolve()) for root in (repository_root, input_dir)):
        raise GraphEncoderError(
            "unsafe_screening_output_location",
            "output must be outside repository and preserved input",
        )
    final.parent.mkdir(parents=True, exist_ok=True)
    staging = final.with_name(final.name + ".incomplete-" + identity)
    if staging.exists() or staging.is_symlink():
        raise GraphEncoderError(
            "screening_staging_exists", "incomplete output already exists"
        )
    staging.mkdir()
    return final, staging


def finalize_screening_artifact(staging_dir, final_dir):
    staging = Path(staging_dir)
    final = Path(final_dir)
    ordinary = _regular_files(
        staging, excluded=("artifact_manifest.json", "SHA256SUMS")
    )
    if ordinary != SCREEN_EXPECTED_ORDINARY_FILES:
        raise GraphEncoderError(
            "incomplete_screening_artifact",
            "all three ordinary artifact files are required",
        )
    manifest = {
        "schema_version": SCREEN_ARTIFACT_VERSION,
        "protocol_version": SCREEN_PROTOCOL_VERSION,
        "source_feature_artifact": {
            "git_commit": SOURCE_FEATURE_COMMIT,
            "slurm_job_id": SOURCE_FEATURE_JOB_ID,
            "sha256": dict(SCREEN_INPUT_HASHES),
        },
        "artifacts": [
            {
                "path": name,
                "byte_size": (staging / name).stat().st_size,
                "sha256": _file_sha256(staging / name),
            }
            for name in ordinary
        ],
    }
    _atomic_write_json(staging / "artifact_manifest.json", manifest)
    checksum_paths = _regular_files(staging, excluded=("SHA256SUMS",))
    checksum_text = "".join(
        "{}  {}\n".format(_file_sha256(staging / name), name)
        for name in checksum_paths
    )
    _atomic_write_bytes(staging / "SHA256SUMS", checksum_text.encode("utf-8"))
    verify_screening_artifact(staging, allow_incomplete_name=True)
    os.replace(str(staging), str(final))
    return final


def verify_screening_artifact(path, *, allow_incomplete_name=False,
                              expected_commit=None, expected_slurm_job_id=None):
    root = Path(path)
    if not root.is_dir() or root.is_symlink():
        raise GraphEncoderError(
            "invalid_screening_artifact", "artifact root must be a real directory"
        )
    if ".incomplete-" in root.name and not allow_incomplete_name:
        raise GraphEncoderError(
            "invalid_screening_artifact", "incomplete artifact cannot be final"
        )
    if _regular_files(root) != tuple(sorted(SCREEN_EXPECTED_FILES)):
        raise GraphEncoderError(
            "incomplete_screening_artifact", "five artifact files are required"
        )
    manifest = json.loads((root / "artifact_manifest.json").read_text("utf-8"))
    if (
        manifest.get("schema_version") != SCREEN_ARTIFACT_VERSION
        or manifest.get("protocol_version") != SCREEN_PROTOCOL_VERSION
        or manifest.get("source_feature_artifact") != {
            "git_commit": SOURCE_FEATURE_COMMIT,
            "slurm_job_id": SOURCE_FEATURE_JOB_ID,
            "sha256": SCREEN_INPUT_HASHES,
        }
    ):
        raise GraphEncoderError(
            "invalid_screening_artifact", "artifact manifest identity differs"
        )
    listed = []
    for row in manifest.get("artifacts", ()):
        if not isinstance(row, dict) or set(row) != {"path", "byte_size", "sha256"}:
            raise GraphEncoderError(
                "invalid_screening_artifact", "artifact row is malformed"
            )
        relative = _safe_relative_path(row["path"])
        target = root / relative
        if (
            not target.is_file()
            or target.is_symlink()
            or target.stat().st_size != row["byte_size"]
            or _file_sha256(target) != row["sha256"]
        ):
            raise GraphEncoderError(
                "screening_artifact_integrity_failure", "artifact hash differs"
            )
        listed.append(relative)
    if tuple(listed) != SCREEN_EXPECTED_ORDINARY_FILES:
        raise GraphEncoderError(
            "invalid_screening_artifact", "artifact coverage differs"
        )
    checksum = (root / "SHA256SUMS").read_bytes()
    if not checksum.endswith(b"\n"):
        raise GraphEncoderError(
            "invalid_screening_artifact", "SHA256SUMS requires a final LF"
        )
    checksum_paths = []
    for line in checksum.decode("utf-8").splitlines():
        parts = line.split("  ", 1)
        if len(parts) != 2 or len(parts[0]) != 64:
            raise GraphEncoderError(
                "invalid_screening_artifact", "SHA256SUMS row is malformed"
            )
        relative = _safe_relative_path(parts[1])
        if _file_sha256(root / relative) != parts[0]:
            raise GraphEncoderError(
                "screening_artifact_integrity_failure", "checksum differs"
            )
        checksum_paths.append(relative)
    if tuple(checksum_paths) != _regular_files(root, excluded=("SHA256SUMS",)):
        raise GraphEncoderError(
            "invalid_screening_artifact", "SHA256SUMS coverage differs"
        )
    resolved = json.loads((root / "resolved_config.json").read_text("utf-8"))
    results = json.loads((root / "screening_results.json").read_text("utf-8"))
    validate_screening_results(results)
    events = _read_canonical_jsonl(root / "metrics.jsonl")
    if (
        resolved.get("protocol_version") != SCREEN_PROTOCOL_VERSION
        or resolved.get("screen_contract") != screen_contract()
        or resolved.get("input_hashes") != SCREEN_INPUT_HASHES
        or resolved.get("runtime", {}).get("python") != "3.8.13"
        or str(resolved.get("runtime", {}).get("pytorch", "")).split("+")[0]
        != "1.11.0"
        or resolved.get("runtime", {}).get("device") != "cpu"
        or resolved.get("runtime", {}).get("cuda_available") is not False
        or resolved.get("runtime", {}).get("cpu_threads") != 1
    ):
        raise GraphEncoderError(
            "invalid_screening_artifact", "resolved configuration differs"
        )
    if expected_commit is not None and resolved.get("source", {}).get("git_commit") != expected_commit:
        raise GraphEncoderError(
            "invalid_screening_artifact", "screening commit differs"
        )
    if (
        expected_slurm_job_id is not None
        and resolved.get("runtime", {}).get("slurm_job_id")
        != str(expected_slurm_job_id)
    ):
        raise GraphEncoderError(
            "invalid_screening_artifact", "screening Slurm job differs"
        )
    if not events or events[-1].get("event") != "representation_screening_completed":
        raise GraphEncoderError(
            "invalid_screening_artifact", "terminal completion event is absent"
        )
    access = screening_access_declarations(input_accessed=True, completed=True)
    if any(events[-1].get(name) != value for name, value in access.items()):
        raise GraphEncoderError(
            "invalid_screening_artifact", "terminal access declarations differ"
        )
    if resolved.get("access") != access:
        raise GraphEncoderError(
            "invalid_screening_artifact", "resolved access declarations differ"
        )
    return {
        "protocol_version": SCREEN_PROTOCOL_VERSION,
        "file_count": len(_regular_files(root)),
        "pipeline_count": len(results["pipelines"]),
        "candidate_pipeline_count": results["screening_summary"][
            "candidate_pipeline_count"
        ],
        "integrity_verified": True,
    }


def _require_screening_runtime():
    if torch is None:
        raise RuntimeError("representation screening requires PyTorch")
    job_id = validate_slurm_job_id()
    if platform.python_version() != "3.8.13":
        raise GraphEncoderError(
            "invalid_screening_runtime", "Python 3.8.13 is required"
        )
    if str(torch.__version__).split("+")[0] != "1.11.0":
        raise GraphEncoderError(
            "invalid_screening_runtime", "PyTorch 1.11.0 is required"
        )
    if torch.cuda.is_available() or torch.get_num_threads() != 1:
        raise GraphEncoderError(
            "invalid_screening_runtime", "CPU-only one-thread execution is required"
        )
    return job_id


def run_representation_screening(*, input_dir, output_dir, repository_root,
                                 expected_commit):
    """Run the exact 16 real-label pipelines and no permutation controls."""

    job_id = _require_screening_runtime()
    source = _source_identity(repository_root, expected_commit)
    final, staging = prepare_screening_output(
        output_dir,
        repository_root=repository_root,
        input_dir=input_dir,
        job_id=job_id,
    )
    started = time.perf_counter()
    events = [{
        "event": "representation_screening_started",
        "protocol_version": SCREEN_PROTOCOL_VERSION,
        "source": source,
        **screening_access_declarations(),
    }]
    inputs = load_screening_inputs(input_dir)
    events.append({
        "event": "representation_screening_inputs_verified",
        "input_hashes": inputs["input_hashes"],
        **screening_access_declarations(input_accessed=True),
    })
    runtime = {
        "python": platform.python_version(),
        "pytorch": str(torch.__version__),
        "device": "cpu",
        "cuda_available": bool(torch.cuda.is_available()),
        "cpu_threads": int(torch.get_num_threads()),
        "host": socket.gethostname(),
        "slurm_job_id": job_id,
    }
    execution_source = dict(source)
    execution_source["slurm_job_id"] = job_id
    results = run_real_label_screen(
        inputs["feature_rows"],
        inputs["label_rows"],
        inputs["input_hashes"],
        execution_source=execution_source,
        environment=runtime,
    )
    _atomic_write_json(staging / "screening_results.json", results)
    access = screening_access_declarations(input_accessed=True, completed=True)
    resolved = {
        "protocol_version": SCREEN_PROTOCOL_VERSION,
        "artifact_version": SCREEN_ARTIFACT_VERSION,
        "source": source,
        "source_feature_artifact": {
            "git_commit": SOURCE_FEATURE_COMMIT,
            "slurm_job_id": SOURCE_FEATURE_JOB_ID,
            "full_adr0011_probe_completed": False,
        },
        "input_hashes": inputs["input_hashes"],
        "screen_contract": screen_contract(),
        "runtime": runtime,
        "access": access,
        "permutation_significance_available": False,
        "adr0011_representation_probe_complete": False,
        "formal_representation_conclusion_available": False,
        "stage6_authorized": False,
        "c8_authorized": False,
        "model_or_decoder_repair_authorized": False,
    }
    _atomic_write_json(staging / "resolved_config.json", resolved)
    events.append({
        "event": "representation_screening_completed",
        "protocol_version": SCREEN_PROTOCOL_VERSION,
        "pipeline_count": len(results["pipelines"]),
        "candidate_pipeline_count": results["screening_summary"][
            "candidate_pipeline_count"
        ],
        "permutation_significance_available": False,
        "adr0011_representation_probe_complete": False,
        "formal_representation_conclusion_available": False,
        "elapsed_seconds": time.perf_counter() - started,
        **access,
    })
    _atomic_write_jsonl(staging / "metrics.jsonl", events)
    _verify_source_unchanged(source, repository_root, expected_commit)
    finalize_screening_artifact(staging, final)
    verified = verify_screening_artifact(
        final,
        expected_commit=expected_commit,
        expected_slurm_job_id=job_id,
    )
    return {"artifact_path": str(final), "artifact_verification": verified}


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run the artifact-only GE1 representation screening"
    )
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--expected-commit", required=True)
    arguments = parser.parse_args(argv)
    result = run_representation_screening(
        input_dir=arguments.input_dir,
        output_dir=arguments.output_dir,
        repository_root=arguments.repository_root,
        expected_commit=arguments.expected_commit,
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
