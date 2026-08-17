"""Artifact-only closed-form GE1 representation readout.

This additive diagnostic reads only the three detached, target-isolated files
preserved by representation-probe job 3346513.  It never imports a GE1 model,
checkpoint loader, corpus loader, training loop, or repair implementation.
All fitting is disposable float64 CPU ridge regression with exact reuse of
label-independent SVD factorizations.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
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
except ImportError:  # Contract and documentation tests remain torch-free.
    torch = None

from .errors import GraphEncoderError
from .optimization_diagnostic import validate_slurm_job_id
from .pilot import (
    _atomic_write_bytes,
    _atomic_write_json,
    _file_sha256,
    _is_within,
    _read_canonical_jsonl,
    _safe_relative_path,
    _source_identity,
    _verify_source_unchanged,
)
from .representation_probe import (
    FEATURE_DIMENSIONS,
    FIDELITY_ERROR_LIMITS,
    OPERATION_GRIDS,
    OPERATION_TYPES,
    PROBE_ARMS,
    PROBE_FEATURE_VERSION,
    PROBE_LABEL_VERSION,
    PROBE_PROTOCOL_VERSION,
    classification_metrics,
    fit_monotonic_thresholds,
    grouped_threshold_predictions,
    nearest_grid_predictions,
    validate_feature_rows,
)


READOUT_PROTOCOL_VERSION = "GE1-C7-CLOSED-FORM-READOUT-v2"
READOUT_ARTIFACT_VERSION = "GE1-C7-CLOSED-FORM-READOUT-ARTIFACT-v2"
READOUT_RESULT_VERSION = "GE1-C7-CLOSED-FORM-READOUT-RESULTS-v2"
READOUT_SEED = 2026
SOURCE_FEATURE_COMMIT = "3ea43650bb793d8d55b3d61a1edb70ec7a920089"
SOURCE_FEATURE_JOB_ID = "3346513"
READOUT_INPUT_HASHES = {
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
RIDGE_LAMBDAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
PERMUTATION_LADDER = (999, 499)
TIMING_LIMIT_SECONDS = 45 * 60
TIMING_SAMPLE_PERMUTATIONS = 7
TIMING_BASELINE_PERMUTATIONS = 3
TIMING_SAFETY_FACTOR = 1.25
TIMING_FINALIZATION_SECONDS = 180.0
ACCESS_BALANCED_MIN = 0.40
MATERIAL_MARGIN = 0.10
P_VALUE_MAX = 0.05
FEATURE_CONTRACTS = {
    "C": {"nominal_dimension": 32, "operation_types": ("extrude", "revolve")},
    "P": {"nominal_dimension": 32, "operation_types": ("extrude", "revolve")},
    "D-additive": {"nominal_dimension": 34, "operation_types": ("extrude",)},
    "D-gated": {"nominal_dimension": 66, "operation_types": ("extrude",)},
    "context-only": {"nominal_dimension": 2, "operation_types": ("extrude",)},
}
COHORT_CONTRACTS = {
    "full": {
        "extrude": {"families": 24, "operations": 32},
        "revolve": {"families": 16, "operations": 16},
    },
    "single_extrusion_E_RE": {
        "extrude": {"families": 16, "operations": 16},
    },
}
EXPECTED_ORDINARY_FILES = (
    "metrics.jsonl",
    "readout_results.json",
    "resolved_config.json",
)
EXPECTED_FILES = EXPECTED_ORDINARY_FILES + (
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
LIMITATIONS = (
    "observational_accessibility_diagnostic_only",
    "does_not_identify_an_exact_code_defect",
    "does_not_prove_information_absence",
    "does_not_prove_architectural_capacity_exhaustion",
    "does_not_prove_latent_tokens_or_bottleneck_dim_binding",
    "does_not_predict_or_prove_a_48_of_48_categorical_gate",
    "does_not_establish_encoder_superiority",
    "does_not_authorize_repair_stage6_c8_or_protected_access",
)


def readout_contract():
    """Return the frozen ADR-0012 implementation contract."""

    return {
        "protocol_version": READOUT_PROTOCOL_VERSION,
        "artifact_version": READOUT_ARTIFACT_VERSION,
        "results_version": READOUT_RESULT_VERSION,
        "source_feature_artifact": {
            "git_commit": SOURCE_FEATURE_COMMIT,
            "slurm_job_id": SOURCE_FEATURE_JOB_ID,
            "sha256": dict(READOUT_INPUT_HASHES),
            "adr0011_status": "accepted_and_incomplete",
            "representation_screen_status": "separate_and_incomplete",
        },
        "cohorts": copy.deepcopy(COHORT_CONTRACTS),
        "features": copy.deepcopy(FEATURE_CONTRACTS),
        "scalar_baseline": {
            "identity": "A/B grouped scalar common finite-precision weak order",
            "coordinate_specific_analyses": "reported_descriptively_only",
            "order_contract": (
                "collapse_exact_ties_in_either_coordinate_transitively_"
                "and_fail_on_strict_inversion"
            ),
            "boundary_representation": "ordered_upper_anchor_no_numeric_midpoints",
            "held_out_same_gap_policy": "smaller_class_index",
            "permutation_rule": "refit_same_common_boundaries_for_every_target",
        },
        "estimator": {
            "name": "five_output_closed_form_ridge",
            "objective": (
                "(1/(2n))*||Y-XW-b||_F^2+(lambda/2)*||W||_F^2"
            ),
            "intercept": "unpenalized",
            "dtype": "float64",
            "device": "cpu",
            "classes": 5,
            "lambda_grid": list(RIDGE_LAMBDAS),
            "argmax_tie_break": "smallest_class_index",
            "selection_order": (
                "maximum_balanced_accuracy_then_maximum_raw_accuracy_"
                "then_largest_lambda"
            ),
            "outer_split": "physical_family_lofo",
            "inner_split": "physical_family_lofo",
            "solver": "thin_svd_exact_label_independent_reuse",
        },
        "permutations": {
            "seed": READOUT_SEED,
            "seed_parts": [
                "protocol_identity", "seed", "operation_type", "permutation_index"
            ],
            "ladder": list(PERMUTATION_LADDER),
            "projection_limit_seconds": TIMING_LIMIT_SECONDS,
            "global_correction": "single_step_centered_maxT_per_metric",
        },
        "primary_hypothesis_count": 16,
        "limitations": list(LIMITATIONS),
    }


def readout_access_declarations(*, input_accessed=False, completed=False):
    return {
        "preserved_feature_label_artifact_accessed": bool(input_accessed),
        "preserved_input_content_access_status": (
            "verified_and_loaded" if input_accessed else "not_accessed"
        ),
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
        "ge1_inference_performed": False,
        "ge1_training_performed": False,
        "ge1_backward_pass_performed": False,
        "ge1_optimizer_constructed": False,
        "ge1_checkpoint_written": False,
        "model_or_decoder_repair_implemented_or_invoked": False,
        "stage6_performed": False,
        "c8_or_later_performed": False,
        "closed_form_readout_completed": bool(completed),
        "formal_adr0011_representation_probe_completed": False,
    }


def _validate_hashes(observed, expected=None):
    frozen = READOUT_INPUT_HASHES if expected is None else dict(expected)
    if dict(observed) != frozen:
        raise GraphEncoderError(
            "readout_input_hash_mismatch", "preserved input hashes differ"
        )
    return dict(observed)


def _validate_labels(payload):
    expected_top = {
        "schema_version",
        "labels_joined_after_ge1_models_released",
        "grouping_unit",
        "class_grids",
        "rows",
    }
    if not isinstance(payload, dict) or set(payload) != expected_top:
        raise GraphEncoderError("invalid_readout_labels", "label fields differ")
    if (
        payload["schema_version"] != PROBE_LABEL_VERSION
        or payload["labels_joined_after_ge1_models_released"] is not True
        or payload["grouping_unit"] != "physical_family"
        or payload["class_grids"]
        != {name: list(values) for name, values in OPERATION_GRIDS.items()}
        or not isinstance(payload["rows"], list)
        or len(payload["rows"]) != 48
    ):
        raise GraphEncoderError(
            "invalid_readout_labels", "label identity or count differs"
        )
    rows = tuple(payload["rows"])
    if len({row.get("label_key") for row in rows}) != 48:
        raise GraphEncoderError("invalid_readout_labels", "label keys differ")
    by_family = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != LABEL_ROW_FIELDS:
            raise GraphEncoderError("invalid_readout_labels", "label row differs")
        operation_type = row.get("operation_type")
        operation_index = row.get("operation_index")
        class_index = row.get("class_index")
        if (
            operation_type not in OPERATION_TYPES
            or isinstance(operation_index, bool)
            or operation_index not in (0, 1)
            or isinstance(class_index, bool)
            or not isinstance(class_index, int)
            or class_index not in range(5)
            or row.get("operation_template") not in ("E", "R", "EE", "RE")
            or row.get("label_key") != "{}:{}:{}".format(
                row.get("family_id"), operation_index, operation_type
            )
            or float(row.get("class_physical_value"))
            != float(OPERATION_GRIDS[operation_type][class_index])
        ):
            raise GraphEncoderError("invalid_readout_labels", "label row malformed")
        by_family.setdefault(row["family_id"], []).append(row)
    expected_patterns = {
        "E": ((0, "extrude"),),
        "R": ((0, "revolve"),),
        "EE": ((0, "extrude"), (1, "extrude")),
        "RE": ((0, "revolve"), (1, "extrude")),
    }
    template_counts = {name: 0 for name in expected_patterns}
    for family_rows in by_family.values():
        templates = {row["operation_template"] for row in family_rows}
        if len(templates) != 1:
            raise GraphEncoderError("invalid_readout_labels", "family template differs")
        template = templates.pop()
        pattern = tuple(sorted(
            (row["operation_index"], row["operation_type"])
            for row in family_rows
        ))
        if pattern != expected_patterns[template]:
            raise GraphEncoderError("invalid_readout_labels", "operation pattern differs")
        template_counts[template] += 1
    if (
        len(by_family) != 32
        or template_counts != {"E": 8, "R": 8, "EE": 8, "RE": 8}
        or sum(row["operation_type"] == "extrude" for row in rows) != 32
        or sum(row["operation_type"] == "revolve" for row in rows) != 16
    ):
        raise GraphEncoderError("invalid_readout_labels", "cohort structure differs")
    return rows


def _validate_manifest(payload, feature_path, expected_hashes):
    fields = {
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
    if not isinstance(payload, dict) or set(payload) != fields:
        raise GraphEncoderError("invalid_readout_manifest", "manifest fields differ")
    if (
        payload["schema_version"] != PROBE_FEATURE_VERSION
        or payload["protocol_version"] != PROBE_PROTOCOL_VERSION
        or payload["feature_file"] != "detached_features.pt"
        or payload["feature_file_byte_size"] != feature_path.stat().st_size
        or payload["feature_file_sha256"] != expected_hashes["detached_features.pt"]
        or payload["written_and_hashed_before_label_join"] is not True
        or payload["target_or_label_content_present"] is not False
        or payload["model_state_present"] is not False
        or payload["optimizer_state_present"] is not False
        or payload["row_count"] != 96
        or payload["feature_dimensions"] != FEATURE_DIMENSIONS
    ):
        raise GraphEncoderError("invalid_readout_manifest", "manifest contract differs")
    return payload


def _validate_alignment(feature_rows, label_rows):
    labels = {row["label_key"]: row for row in label_rows}
    arms_by_label = {}
    for feature in feature_rows:
        label = labels.get(feature["label_key"])
        if label is None or any(
            feature[name] != label[name]
            for name in ("family_id", "operation_index", "operation_type")
        ):
            raise GraphEncoderError(
                "readout_input_alignment_failure", "feature/label identity differs"
            )
        arms_by_label.setdefault(feature["label_key"], []).append(feature["arm"])
    if (
        set(arms_by_label) != set(labels)
        or any(tuple(sorted(arms)) != PROBE_ARMS for arms in arms_by_label.values())
    ):
        raise GraphEncoderError(
            "readout_input_alignment_failure", "arm/label coverage differs"
        )
    return True


def load_readout_inputs(input_dir, *, expected_hashes=None, synthetic_test=False):
    """Load features before labels and verify the exact target-isolation order."""

    if torch is None:
        raise RuntimeError("closed-form readout requires PyTorch")
    if expected_hashes is not None and not synthetic_test:
        raise GraphEncoderError(
            "unsafe_hash_override", "input hashes may be overridden only by synthetic tests"
        )
    frozen = READOUT_INPUT_HASHES if expected_hashes is None else dict(expected_hashes)
    root = Path(input_dir)
    if not root.is_dir() or root.is_symlink():
        raise GraphEncoderError("invalid_readout_input", "input root differs")
    paths = {name: root / name for name in READOUT_INPUT_HASHES}
    if any(not path.is_file() or path.is_symlink() for path in paths.values()):
        raise GraphEncoderError("invalid_readout_input", "required file differs")
    hashes = _validate_hashes(
        {name: _file_sha256(path) for name, path in paths.items()}, frozen
    )
    manifest = json.loads(paths["feature_manifest.json"].read_text("utf-8"))
    _validate_manifest(manifest, paths["detached_features.pt"], frozen)
    # Target-free payload is validated and detached before labels are parsed.
    payload = torch.load(str(paths["detached_features.pt"]), map_location="cpu")
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version", "protocol_version", "target_free", "feature_dimensions", "rows"
    }:
        raise GraphEncoderError("invalid_readout_features", "feature payload differs")
    if (
        payload["schema_version"] != PROBE_FEATURE_VERSION
        or payload["protocol_version"] != PROBE_PROTOCOL_VERSION
        or payload["target_free"] is not True
        or payload["feature_dimensions"] != FEATURE_DIMENSIONS
    ):
        raise GraphEncoderError("invalid_readout_features", "feature identity differs")
    feature_rows = validate_feature_rows(payload["rows"])
    del payload
    label_payload = json.loads(paths["labels.json"].read_text("utf-8"))
    label_rows = _validate_labels(label_payload)
    _validate_alignment(feature_rows, label_rows)
    return {
        "feature_rows": feature_rows,
        "label_rows": label_rows,
        "input_hashes": hashes,
        "feature_manifest": manifest,
        "label_join_order": "target_free_payload_validated_and_released_before_labels_parsed",
    }


def join_detached_rows(feature_rows, label_rows, arm):
    if arm not in PROBE_ARMS:
        raise GraphEncoderError("invalid_readout_arm", "unknown arm")
    labels = {row["label_key"]: row for row in label_rows}
    joined = []
    for feature in feature_rows:
        if feature["arm"] != arm:
            continue
        row = dict(feature)
        label = labels[feature["label_key"]]
        row["operation_template"] = label["operation_template"]
        row["class_index"] = label["class_index"]
        joined.append(row)
    return tuple(sorted(joined, key=lambda row: row["key"]))


def select_cohort(rows, operation_type, cohort):
    if operation_type not in OPERATION_TYPES:
        raise GraphEncoderError("invalid_operation_type", "unknown operation type")
    selected = tuple(row for row in rows if row["operation_type"] == operation_type)
    if cohort == "single_extrusion_E_RE":
        if operation_type != "extrude":
            raise GraphEncoderError(
                "redundant_revolve_sensitivity", "full revolve is already one-to-one"
            )
        selected = tuple(
            row for row in selected if row["operation_template"] in ("E", "RE")
        )
    elif cohort != "full":
        raise GraphEncoderError("invalid_readout_cohort", "unknown cohort")
    expected = COHORT_CONTRACTS[cohort][operation_type]
    if (
        len(selected) != expected["operations"]
        or len({row["family_id"] for row in selected}) != expected["families"]
    ):
        raise GraphEncoderError("invalid_readout_cohort", "cohort counts differ")
    return tuple(sorted(selected, key=lambda row: row["key"]))


def construct_features(rows, feature_identity, operation_type):
    """Construct C/P/slot features exclusively from target-free row fields."""

    contract = FEATURE_CONTRACTS.get(feature_identity)
    if contract is None or operation_type not in contract["operation_types"]:
        raise GraphEncoderError("degenerate_readout_feature", "feature is not executable")
    selected = tuple(rows)
    vectors = []
    for row in selected:
        if row["operation_type"] != operation_type:
            raise GraphEncoderError("invalid_readout_feature_rows", "operation type differs")
        c_value = tuple(float(value) for value in row["C"])
        p_value = tuple(float(value) for value in row["D"][:32])
        slot = tuple(float(value) for value in row["D"][-2:])
        if feature_identity == "C":
            vector = c_value
            kinds = ("continuous",) * 32
        elif feature_identity == "P":
            vector = p_value
            kinds = ("continuous",) * 32
        elif feature_identity == "D-additive":
            vector = p_value + slot
            kinds = ("continuous",) * 32 + ("binary",) * 2
        elif feature_identity == "D-gated":
            vector = (
                tuple(value * slot[0] for value in p_value)
                + tuple(value * slot[1] for value in p_value)
                + slot
            )
            kinds = ("continuous",) * 64 + ("binary",) * 2
        else:
            vector = slot
            kinds = ("binary",) * 2
        vectors.append(vector)
    if any(len(vector) != contract["nominal_dimension"] for vector in vectors):
        raise GraphEncoderError("invalid_readout_features", "feature dimension differs")
    return {
        "matrix": torch.tensor(vectors, dtype=torch.float64, device="cpu"),
        "column_kinds": kinds,
        "nominal_dimension": contract["nominal_dimension"],
        "feature_identity": feature_identity,
    }


def revolve_degeneracy_records(arm):
    reason = (
        "all_revolve_operations_are_slot_0_after_zero_variance_removal_"
        "context_is_constant_and_additive_and_gated_equal_P"
    )
    return [
        {
            "status": "degenerate_by_construction",
            "arm": arm,
            "operation_type": "revolve",
            "cohort": "full",
            "feature_identity": feature,
            "nominal_dimension": FEATURE_CONTRACTS[feature]["nominal_dimension"],
            "reason": reason,
            "fit_executed": False,
            "counted_as_primary_test": False,
        }
        for feature in ("D-additive", "D-gated", "context-only")
    ]


def revolve_sensitivity_record(arm):
    return {
        "status": "not_applicable_full_cohort_already_one_to_one",
        "arm": arm,
        "operation_type": "revolve",
        "cohort": "sensitivity",
        "fit_executed": False,
        "counted_as_primary_test": False,
        "reason": "R_plus_RE_is_the_complete_16_family_revolve_cohort",
    }


def _seed_from_parts(*parts):
    material = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def permute_family_blocks(rows, operation_type, permutation_index):
    """Permute ordered family label blocks with an arm/feature-free seed."""

    if operation_type not in OPERATION_TYPES or int(permutation_index) < 0:
        raise GraphEncoderError("invalid_readout_permutation", "permutation identity differs")
    selected = tuple(rows)
    by_template = {}
    for row in selected:
        if row["operation_type"] != operation_type:
            raise GraphEncoderError("invalid_readout_permutation", "operation type differs")
        by_template.setdefault(row["operation_template"], {}).setdefault(
            row["family_id"], []
        ).append(row)
    generator = random.Random(_seed_from_parts(
        READOUT_PROTOCOL_VERSION, READOUT_SEED, operation_type, permutation_index
    ))
    result = {}
    for template in sorted(by_template):
        families = sorted(by_template[template])
        blocks = []
        for family in families:
            ordered = sorted(
                by_template[template][family], key=lambda row: row["operation_index"]
            )
            blocks.append(tuple(row["class_index"] for row in ordered))
        donor_indices = list(range(len(families)))
        generator.shuffle(donor_indices)
        for recipient_index, family in enumerate(families):
            recipients = sorted(
                by_template[template][family], key=lambda row: row["operation_index"]
            )
            donor = blocks[donor_indices[recipient_index]]
            if len(recipients) != len(donor):
                raise GraphEncoderError("invalid_readout_permutation", "block size differs")
            for row, label in zip(recipients, donor):
                result[row["key"]] = label
    if set(result) != {row["key"] for row in selected}:
        raise GraphEncoderError("invalid_readout_permutation", "coverage differs")
    return result


def target_matrix(rows, permutation_count, permutation_source_rows=None):
    selected = tuple(rows)
    source = selected if permutation_source_rows is None else tuple(permutation_source_rows)
    values = [[row["class_index"] for row in selected]]
    for index in range(int(permutation_count)):
        labels = permute_family_blocks(source, selected[0]["operation_type"], index)
        values.append([labels[row["key"]] for row in selected])
    return torch.tensor(values, dtype=torch.long, device="cpu")


def metric_record(labels, predictions):
    record = classification_metrics(
        [int(value) for value in labels], [int(value) for value in predictions]
    )
    predicted_count = [
        sum(int(value) == class_index for value in predictions)
        for class_index in range(5)
    ]
    record["predicted_class_count"] = predicted_count
    record["class_masking_detected"] = any(
        support > 0 and predicted_count[index] == 0
        for index, support in enumerate(record["class_support"])
    )
    record["masking_limit"] = (
        "negative_ridge_result_is_limited_by_class_masking"
        if record["class_masking_detected"] else None
    )
    return record


def _batch_metric_values(labels, predictions):
    truth = labels.to(dtype=torch.long, device="cpu")
    predicted = predictions.to(dtype=torch.long, device="cpu")
    accuracy = (truth == predicted).to(torch.float64).mean(dim=1)
    recalls = []
    for class_index in range(5):
        support = (truth == class_index).sum(dim=1)
        correct = ((truth == class_index) & (predicted == class_index)).sum(dim=1)
        recalls.append(torch.where(
            support > 0,
            correct.to(torch.float64) / support.clamp(min=1).to(torch.float64),
            torch.zeros_like(accuracy),
        ))
    balanced = torch.stack(recalls, dim=1).mean(dim=1)
    return accuracy, balanced


@dataclass
class PreprocessedFold:
    train: object
    test: object
    retained_columns: tuple
    zero_variance_columns: tuple
    means: tuple
    scales: tuple
    column_kinds: tuple


def preprocess_fold(train, test, column_kinds):
    if torch is None:
        raise RuntimeError("ridge preprocessing requires PyTorch")
    train_value = torch.as_tensor(train, dtype=torch.float64, device="cpu")
    test_value = torch.as_tensor(test, dtype=torch.float64, device="cpu")
    kinds = tuple(column_kinds)
    if (
        train_value.dim() != 2
        or test_value.dim() != 2
        or train_value.size(1) != test_value.size(1)
        or train_value.size(1) != len(kinds)
        or any(kind not in ("continuous", "binary") for kind in kinds)
    ):
        raise GraphEncoderError("invalid_ridge_features", "matrix shape differs")
    means_all = train_value.mean(dim=0)
    centered = train_value - means_all
    variance = (centered * centered).mean(dim=0)
    retained = tuple(
        index for index in range(train_value.size(1)) if float(variance[index]) > 0.0
    )
    dropped = tuple(index for index in range(train_value.size(1)) if index not in retained)
    if retained:
        indices = torch.tensor(retained, dtype=torch.long)
        means = means_all.index_select(0, indices)
        selected_kinds = tuple(kinds[index] for index in retained)
        scales = torch.ones(len(retained), dtype=torch.float64)
        for local_index, kind in enumerate(selected_kinds):
            if kind == "continuous":
                scales[local_index] = torch.sqrt(variance[retained[local_index]])
        transformed_train = (
            train_value.index_select(1, indices) - means
        ) / scales
        transformed_test = (
            test_value.index_select(1, indices) - means
        ) / scales
    else:
        means = torch.empty(0, dtype=torch.float64)
        scales = torch.empty(0, dtype=torch.float64)
        selected_kinds = ()
        transformed_train = train_value[:, :0]
        transformed_test = test_value[:, :0]
    return PreprocessedFold(
        train=transformed_train,
        test=transformed_test,
        retained_columns=retained,
        zero_variance_columns=dropped,
        means=tuple(float(value) for value in means),
        scales=tuple(float(value) for value in scales),
        column_kinds=selected_kinds,
    )


class RidgeSVDCache:
    """One exact label-independent decomposition for every lambda and RHS."""

    def __init__(self, train, test, column_kinds, timing=None):
        started = time.perf_counter()
        self.preprocessing = preprocess_fold(train, test, column_kinds)
        self.n = int(self.preprocessing.train.size(0))
        if self.preprocessing.train.size(1):
            self.u, self.s, self.vh = torch.linalg.svd(
                self.preprocessing.train, full_matrices=False
            )
        else:
            self.u = torch.empty((self.n, 0), dtype=torch.float64)
            self.s = torch.empty((0,), dtype=torch.float64)
            self.vh = torch.empty((0, 0), dtype=torch.float64)
        self.operators = []
        if self.s.numel():
            test_basis = self.preprocessing.test.matmul(self.vh.t())
            for ridge_lambda in RIDGE_LAMBDAS:
                multiplier = self.s / (
                    self.s * self.s + self.n * float(ridge_lambda)
                )
                self.operators.append(
                    (test_basis * multiplier).matmul(self.u.t())
                )
        else:
            self.operators = [
                torch.zeros(
                    (self.preprocessing.test.size(0), self.n),
                    dtype=torch.float64,
                )
                for unused in RIDGE_LAMBDAS
            ]
        elapsed = time.perf_counter() - started
        if timing is not None:
            timing["factorization_seconds"] += elapsed
            timing["factorization_count"] += 1

    def scores(self, targets, ridge_lambda):
        target = torch.as_tensor(targets, dtype=torch.float64, device="cpu")
        individual = target.dim() == 2
        if individual:
            target = target.unsqueeze(0)
        if target.dim() != 3 or target.size(1) != self.n or target.size(2) != 5:
            raise GraphEncoderError("invalid_ridge_targets", "target matrix differs")
        mean = target.mean(dim=1, keepdim=True)
        centered = target - mean
        try:
            lambda_index = RIDGE_LAMBDAS.index(float(ridge_lambda))
        except ValueError:
            raise GraphEncoderError("invalid_ridge_lambda", "lambda is outside grid")
        score = torch.matmul(
            self.operators[lambda_index].unsqueeze(0), centered
        ) + mean
        return score[0] if individual else score

    def scores_all_lambdas(self, targets):
        target = torch.as_tensor(targets, dtype=torch.float64, device="cpu")
        individual = target.dim() == 2
        if individual:
            target = target.unsqueeze(0)
        if target.dim() != 3 or target.size(1) != self.n or target.size(2) != 5:
            raise GraphEncoderError("invalid_ridge_targets", "target matrix differs")
        mean = target.mean(dim=1, keepdim=True)
        centered = target - mean
        values = torch.stack([
            torch.matmul(operator.unsqueeze(0), centered) + mean
            for operator in self.operators
        ], dim=0)
        return values[:, 0] if individual else values


def uncached_reference_scores(train, test, column_kinds, targets, ridge_lambda):
    """Simple primal solve retained only as a synthetic-test oracle."""

    processed = preprocess_fold(train, test, column_kinds)
    target = torch.as_tensor(targets, dtype=torch.float64, device="cpu")
    individual = target.dim() == 2
    if individual:
        target = target.unsqueeze(0)
    mean = target.mean(dim=1, keepdim=True)
    centered = target - mean
    dimension = processed.train.size(1)
    if dimension:
        gram = processed.train.t().matmul(processed.train)
        gram = gram + processed.train.size(0) * float(ridge_lambda) * torch.eye(
            dimension, dtype=torch.float64
        )
        rhs = torch.matmul(processed.train.t().unsqueeze(0), centered)
        weights = torch.linalg.solve(gram.unsqueeze(0), rhs)
        score = torch.matmul(processed.test.unsqueeze(0), weights) + mean
    else:
        score = mean.expand(target.size(0), processed.test.size(0), 5)
    return score[0] if individual else score


def _one_hot(labels):
    return torch.nn.functional.one_hot(labels.to(torch.long), num_classes=5).to(
        torch.float64
    )


def select_lambdas_balanced_then_raw(metric_table):
    """Select per-target lambdas, resolving exact ties toward largest lambda."""

    balanced = metric_table["balanced_accuracy"]
    accuracy = metric_table["accuracy"]
    selected = []
    for target_index in range(balanced.size(1)):
        candidates = []
        for lambda_index, ridge_lambda in enumerate(RIDGE_LAMBDAS):
            candidates.append((
                float(balanced[lambda_index, target_index]),
                float(accuracy[lambda_index, target_index]),
                float(ridge_lambda),
                lambda_index,
            ))
        selected.append(max(candidates)[3])
    return torch.tensor(selected, dtype=torch.long)


def _nested_lambda_selection(matrix, kinds, families, indices, targets, timing):
    subset_families = tuple(sorted({families[index] for index in indices}))
    target_count = targets.size(0)
    accuracy = torch.zeros((len(RIDGE_LAMBDAS), target_count), dtype=torch.float64)
    balanced = torch.zeros_like(accuracy)
    predictions = torch.empty(
        (len(RIDGE_LAMBDAS), target_count, len(indices)), dtype=torch.long
    )
    local_by_global = {global_index: local for local, global_index in enumerate(indices)}
    inner_records = []
    for held in subset_families:
        train_indices = [index for index in indices if families[index] != held]
        test_indices = [index for index in indices if families[index] == held]
        cache = RidgeSVDCache(
            matrix[train_indices], matrix[test_indices], kinds, timing=timing
        )
        train_targets = _one_hot(targets[:, train_indices])
        all_predictions = cache.scores_all_lambdas(train_targets).argmax(dim=3)
        for lambda_index, unused_lambda in enumerate(RIDGE_LAMBDAS):
            observed = all_predictions[lambda_index]
            destinations = [local_by_global[index] for index in test_indices]
            predictions[lambda_index][:, destinations] = observed
        inner_records.append({
            "held_out_family_id": held,
            "train_family_ids": sorted(set(families[index] for index in train_indices)),
            "retained_dimension": len(cache.preprocessing.retained_columns),
            "zero_variance_columns": list(cache.preprocessing.zero_variance_columns),
        })
    truth = targets[:, indices]
    for lambda_index in range(len(RIDGE_LAMBDAS)):
        raw, bal = _batch_metric_values(truth, predictions[lambda_index])
        accuracy[lambda_index] = raw
        balanced[lambda_index] = bal
    selected = select_lambdas_balanced_then_raw({
        "accuracy": accuracy, "balanced_accuracy": balanced
    })
    observed_table = [
        {
            "lambda": ridge_lambda,
            "accuracy": float(accuracy[index, 0]),
            "balanced_accuracy": float(balanced[index, 0]),
            "metrics": metric_record(
                [int(value) for value in truth[0]],
                [int(value) for value in predictions[index, 0]],
            ),
        }
        for index, ridge_lambda in enumerate(RIDGE_LAMBDAS)
    ]
    return selected, observed_table, inner_records


def fit_ridge_pipeline(rows, feature_identity, target_labels, *, arm,
                       operation_type, cohort, timing=None):
    """Fit all true/permuted targets with nested family LOFO and cached SVDs."""

    if timing is None:
        timing = {"factorization_seconds": 0.0, "factorization_count": 0}
    selected_rows = tuple(rows)
    features = construct_features(selected_rows, feature_identity, operation_type)
    matrix = features["matrix"]
    kinds = features["column_kinds"]
    targets = torch.as_tensor(target_labels, dtype=torch.long, device="cpu")
    if targets.dim() != 2 or targets.size(1) != len(selected_rows):
        raise GraphEncoderError("invalid_ridge_targets", "target count differs")
    families = tuple(row["family_id"] for row in selected_rows)
    unique_families = tuple(sorted(set(families)))
    predictions = torch.empty_like(targets)
    folds = []
    for held in unique_families:
        train_indices = [index for index, family in enumerate(families) if family != held]
        test_indices = [index for index, family in enumerate(families) if family == held]
        selected_lambda_indices, selection, inner_records = _nested_lambda_selection(
            matrix, kinds, families, train_indices, targets, timing
        )
        cache = RidgeSVDCache(
            matrix[train_indices], matrix[test_indices], kinds, timing=timing
        )
        train_targets = _one_hot(targets[:, train_indices])
        stacked = cache.scores_all_lambdas(train_targets).argmax(dim=3)
        for target_index in range(targets.size(0)):
            predictions[target_index, test_indices] = stacked[
                selected_lambda_indices[target_index], target_index
            ]
        observed_fold_predictions = [
            int(value) for value in stacked[selected_lambda_indices[0], 0]
        ]
        folds.append({
            "held_out_family_id": held,
            "train_family_ids": sorted(set(families[index] for index in train_indices)),
            "selected_lambda": RIDGE_LAMBDAS[int(selected_lambda_indices[0])],
            "inner_selection": selection,
            "inner_folds": inner_records,
            "retained_dimension": len(cache.preprocessing.retained_columns),
            "retained_columns": list(cache.preprocessing.retained_columns),
            "zero_variance_columns": list(cache.preprocessing.zero_variance_columns),
            "preprocessing_means": list(cache.preprocessing.means),
            "preprocessing_scales": list(cache.preprocessing.scales),
            "preprocessing_column_kinds": list(cache.preprocessing.column_kinds),
            "held_out_metrics": metric_record(
                [int(targets[0, index]) for index in test_indices],
                observed_fold_predictions,
            ),
            "held_out_predictions_by_operation_key": {
                selected_rows[index]["key"]: observed_fold_predictions[local]
                for local, index in enumerate(test_indices)
            },
        })
    raw_values, balanced_values = _batch_metric_values(targets, predictions)
    full_indices = list(range(len(selected_rows)))
    resub_lambda_indices, resub_selection, unused_inner = _nested_lambda_selection(
        matrix, kinds, families, full_indices, targets, timing
    )
    resub_cache = RidgeSVDCache(matrix, matrix, kinds, timing=timing)
    one_hot = _one_hot(targets)
    resub_by_lambda = resub_cache.scores_all_lambdas(one_hot).argmax(dim=3)
    resub_predictions = torch.empty_like(targets)
    for target_index in range(targets.size(0)):
        resub_predictions[target_index] = resub_by_lambda[
            resub_lambda_indices[target_index], target_index
        ]
    observed_truth = [int(value) for value in targets[0]]
    observed_lofo = [int(value) for value in predictions[0]]
    observed_resub = [int(value) for value in resub_predictions[0]]
    return {
        "record": {
            "status": "executed",
            "arm": arm,
            "operation_type": operation_type,
            "cohort": cohort,
            "feature_identity": feature_identity,
            "nominal_dimension": features["nominal_dimension"],
            "column_kinds": list(kinds),
            "lambda_grid": list(RIDGE_LAMBDAS),
            "objective": readout_contract()["estimator"]["objective"],
            "folds": folds,
            "lofo": {
                "metrics": metric_record(observed_truth, observed_lofo),
                "predictions_by_operation_key": {
                    row["key"]: observed_lofo[index]
                    for index, row in enumerate(selected_rows)
                },
            },
            "resubstitution": {
                "generalization_status": "non_generalizing_within_cohort_fit_only",
                "selected_lambda": RIDGE_LAMBDAS[int(resub_lambda_indices[0])],
                "regularization_selection": resub_selection,
                "metrics": metric_record(observed_truth, observed_resub),
                "predictions_by_operation_key": {
                    row["key"]: observed_resub[index]
                    for index, row in enumerate(selected_rows)
                },
            },
            "timing": {},
        },
        "raw_values": [float(value) for value in raw_values],
        "balanced_values": [float(value) for value in balanced_values],
        "predictions": predictions,
    }


def _nearest_rank_95(values):
    ordered = sorted(float(value) for value in values)
    return ordered[int(math.ceil(0.95 * len(ordered))) - 1]


def marginal_permutation_result(values):
    observed = float(values[0])
    null = [float(value) for value in values[1:]]
    if not null:
        raise GraphEncoderError("invalid_permutation_count", "null is empty")
    mean = sum(null) / len(null)
    variance = sum((value - mean) ** 2 for value in null) / len(null)
    return {
        "observed_statistic": observed,
        "null_count": len(null),
        "null_mean": mean,
        "null_population_standard_deviation": math.sqrt(variance),
        "permutation_95th_percentile": _nearest_rank_95(null),
        "permutation_maximum": max(null),
        "raw_empirical_p_value": (
            1.0 + sum(value >= observed for value in null)
        ) / (len(null) + 1.0),
    }


def centered_global_max_t(hypotheses):
    """Apply one single-step centered maxT family to synchronized nulls."""

    if len(hypotheses) != 16:
        raise GraphEncoderError("invalid_primary_family", "exactly 16 hypotheses required")
    lengths = {len(values) for values in hypotheses.values()}
    if len(lengths) != 1 or next(iter(lengths)) < 2:
        raise GraphEncoderError("invalid_primary_family", "null lengths differ")
    count = next(iter(lengths)) - 1
    centered_nulls = {}
    output = {}
    for identity, values in hypotheses.items():
        marginal = marginal_permutation_result(values)
        mean = marginal["null_mean"]
        centered_nulls[identity] = [float(value) - mean for value in values[1:]]
        marginal["observed_centered_statistic"] = float(values[0]) - mean
        output[identity] = marginal
    max_null = [
        max(centered_nulls[identity][index] for identity in centered_nulls)
        for index in range(count)
    ]
    for identity in output:
        observed = output[identity]["observed_centered_statistic"]
        output[identity]["global_centered_maxT_adjusted_p_value"] = (
            1.0 + sum(value >= observed for value in max_null)
        ) / (count + 1.0)
        output[identity]["maxT_family_size"] = 16
    return {
        "hypotheses": output,
        "max_null_summary": {
            "count": count,
            "mean": sum(max_null) / count,
            "permutation_95th_percentile": _nearest_rank_95(max_null),
            "maximum": max(max_null),
        },
    }


def select_permutation_count(projected_999_seconds, projected_499_seconds):
    if float(projected_999_seconds) <= TIMING_LIMIT_SECONDS:
        return {"status": "pass", "selected_permutation_count": 999,
                "reason": "B999_projected_at_or_below_45_minutes"}
    if float(projected_499_seconds) <= TIMING_LIMIT_SECONDS:
        return {"status": "pass", "selected_permutation_count": 499,
                "reason": "B999_above_45_minutes_B499_at_or_below_45_minutes"}
    return {"status": "abort_before_input_access", "selected_permutation_count": None,
            "reason": "both_B999_and_B499_project_above_45_minutes"}


def _finite_json_value(value):
    """Serialize threshold endpoints symbolically without changing fitting."""

    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            raise GraphEncoderError("nonfinite_readout_result", "NaN is forbidden")
        return "infinity" if value > 0 else "-infinity"
    if isinstance(value, dict):
        return {key: _finite_json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_finite_json_value(item) for item in value]
    if isinstance(value, tuple):
        return [_finite_json_value(item) for item in value]
    return value


def common_scalar_order(rows):
    """Build the canonical weak order shared by A and B.

    Numeric distances are deliberately excluded. Strict inversions are an
    invalid positive mapping, while an exact tie in either finite-precision
    coordinate collapses the observations into one common-information block.
    Tie collapse is transitive so the result is a single deterministic weak
    order rather than an outcome-dependent choice of A or B.
    """

    selected = tuple(rows)
    if not selected:
        raise GraphEncoderError("invalid_common_scalar_order", "rows are empty")
    keys = tuple(row["key"] for row in selected)
    if len(set(keys)) != len(keys):
        raise GraphEncoderError("invalid_common_scalar_order", "keys are not unique")
    pairs = []
    for row in selected:
        a_value = float(row["A_physical"])
        b_value = float(row["B_raw_logit"])
        if not math.isfinite(a_value) or not math.isfinite(b_value):
            raise GraphEncoderError(
                "invalid_common_scalar_order", "A/B values must be finite"
            )
        pairs.append((a_value, b_value))

    parents = list(range(len(selected)))

    def root(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left, right):
        left_root = root(left)
        right_root = root(right)
        if left_root != right_root:
            parents[right_root] = left_root

    for left in range(len(selected)):
        for right in range(left + 1, len(selected)):
            a_left, b_left = pairs[left]
            a_right, b_right = pairs[right]
            a_order = (a_left > a_right) - (a_left < a_right)
            b_order = (b_left > b_right) - (b_left < b_right)
            if a_order * b_order < 0:
                raise GraphEncoderError(
                    "positive_mapping_monotonicity_failure",
                    "strict A/B ordering inversion between {} and {}".format(
                        keys[left], keys[right]
                    ),
                )
            if a_order == 0 or b_order == 0:
                union(left, right)

    components = {}
    for index in range(len(selected)):
        components.setdefault(root(index), []).append(index)
    ordered_components = sorted(
        components.values(),
        key=lambda indices: (
            min(pairs[index][0] for index in indices),
            min(pairs[index][1] for index in indices),
            tuple(sorted(keys[index] for index in indices)),
        ),
    )
    blocks = []
    rank_by_key = {}
    previous = None
    for rank, indices in enumerate(ordered_components):
        a_values = [pairs[index][0] for index in indices]
        b_values = [pairs[index][1] for index in indices]
        block = {
            "rank": rank,
            "operation_keys": sorted(keys[index] for index in indices),
            "A_physical_minimum": min(a_values),
            "A_physical_maximum": max(a_values),
            "B_raw_logit_minimum": min(b_values),
            "B_raw_logit_maximum": max(b_values),
            "collapsed_by_finite_precision_tie": len(indices) > 1,
        }
        if previous is not None and not (
            previous["A_physical_maximum"] < block["A_physical_minimum"]
            and previous["B_raw_logit_maximum"] < block["B_raw_logit_minimum"]
        ):
            raise GraphEncoderError(
                "positive_mapping_monotonicity_failure",
                "A/B weak-order blocks are not strictly ordered",
            )
        for index in indices:
            rank_by_key[keys[index]] = rank
        blocks.append(block)
        previous = block
    return {
        "algorithm": "transitive_intersection_of_A_B_finite_precision_weak_orders",
        "numeric_midpoints_used": False,
        "strict_inversion_policy": "fail",
        "tie_policy": "collapse_exact_ties_in_either_coordinate_transitively",
        "common_information_interpretation": (
            "only distinctions preserved by both existing scalar coordinates"
        ),
        "block_count": len(blocks),
        "coordinate_pairs_by_key": {
            key: [pairs[index][0], pairs[index][1]]
            for index, key in sorted(enumerate(keys), key=lambda item: item[1])
        },
        "rank_by_key": rank_by_key,
        "blocks": blocks,
    }


def _common_boundary_candidates(ranks):
    ordered = sorted(set(int(rank) for rank in ranks))
    if not ordered:
        raise GraphEncoderError("invalid_common_scalar_order", "training ranks are empty")
    return tuple(
        [{"kind": "negative_infinity"}]
        + [
            {"kind": "before_common_rank", "upper_anchor_rank": rank}
            for rank in ordered[1:]
        ]
        + [{"kind": "positive_infinity"}]
    )


def _common_boundary_position(boundary):
    if boundary.get("kind") == "negative_infinity":
        return -float("inf")
    if boundary.get("kind") == "positive_infinity":
        return float("inf")
    if boundary.get("kind") == "before_common_rank":
        return float(boundary["upper_anchor_rank"])
    raise GraphEncoderError("invalid_common_scalar_boundary", "unknown boundary kind")


def apply_common_order_boundaries(ranks, boundaries):
    """Apply four order anchors; same-gap values stay in the smaller class."""

    cuts = tuple(boundaries)
    positions = tuple(_common_boundary_position(boundary) for boundary in cuts)
    if len(cuts) != 4 or any(
        left > right for left, right in zip(positions, positions[1:])
    ):
        raise GraphEncoderError(
            "invalid_common_scalar_boundary", "four nondecreasing boundaries required"
        )
    predictions = []
    for value in ranks:
        rank = int(value)
        predictions.append(sum(
            boundary["kind"] == "negative_infinity"
            or (
                boundary["kind"] == "before_common_rank"
                and rank >= int(boundary["upper_anchor_rank"])
            )
            for boundary in cuts
        ))
    return tuple(predictions)


def fit_common_order_boundaries(ranks, labels):
    """Fit exact five-class monotonic boundaries on canonical common ranks."""

    scalar_ranks = tuple(int(rank) for rank in ranks)
    truth = tuple(int(label) for label in labels)
    if not scalar_ranks or len(scalar_ranks) != len(truth):
        raise GraphEncoderError(
            "invalid_common_scalar_order", "rank and label lengths differ"
        )
    if any(label < 0 or label > 4 for label in truth):
        raise GraphEncoderError("invalid_common_scalar_order", "class index differs")
    candidates = _common_boundary_candidates(scalar_ranks)
    prefix_matches = []
    for predicted_class in range(5):
        counts = []
        for boundary in candidates:
            if boundary["kind"] == "negative_infinity":
                count = 0
            elif boundary["kind"] == "positive_infinity":
                count = sum(label == predicted_class for label in truth)
            else:
                anchor = int(boundary["upper_anchor_rank"])
                count = sum(
                    label == predicted_class and rank < anchor
                    for rank, label in zip(scalar_ranks, truth)
                )
            counts.append(count)
        prefix_matches.append(tuple(counts))

    states = {
        index: (prefix_matches[0][index], (index,))
        for index in range(len(candidates))
    }
    for class_index in range(1, 4):
        next_states = {}
        for boundary in range(len(candidates)):
            best = None
            for previous in range(boundary + 1):
                previous_score, previous_path = states[previous]
                segment = (
                    prefix_matches[class_index][boundary]
                    - prefix_matches[class_index][previous]
                )
                candidate = (previous_score + segment, previous_path + (boundary,))
                if best is None or candidate[0] > best[0] or (
                    candidate[0] == best[0] and candidate[1] < best[1]
                ):
                    best = candidate
            next_states[boundary] = best
        states = next_states
    best = None
    total_by_class = [truth.count(index) for index in range(5)]
    for boundary, (score, path) in states.items():
        final_score = score + total_by_class[4] - prefix_matches[4][boundary]
        candidate = (final_score, path)
        if best is None or candidate[0] > best[0] or (
            candidate[0] == best[0] and candidate[1] < best[1]
        ):
            best = candidate
    fitted = tuple(dict(candidates[index]) for index in best[1])
    predictions = apply_common_order_boundaries(scalar_ranks, fitted)
    return {
        "algorithm": "dynamic_programming_on_common_order_anchors",
        "tie_break": "lexicographically_earliest_boundary_tuple",
        "same_gap_policy": "smaller_class_index",
        "boundaries": list(fitted),
        "metrics": classification_metrics(truth, predictions),
        "predictions": list(predictions),
    }


def common_grouped_scalar_predictions(rows, order_contract=None):
    """Fit one LOFO scalar baseline on distinctions shared by A and B."""

    values = tuple(rows)
    common_order = common_scalar_order(values) if order_contract is None else order_contract
    if set(common_order["rank_by_key"]) != {row["key"] for row in values}:
        raise GraphEncoderError(
            "invalid_common_scalar_order", "common-order keys differ from rows"
        )
    observed_pairs = {
        row["key"]: [float(row["A_physical"]), float(row["B_raw_logit"])]
        for row in values
    }
    if common_order.get("coordinate_pairs_by_key") != observed_pairs:
        raise GraphEncoderError(
            "invalid_common_scalar_order", "common-order coordinates differ from rows"
        )
    ranks = common_order["rank_by_key"]
    families = tuple(sorted(set(row["family_id"] for row in values)))
    if len(families) < 2:
        raise GraphEncoderError(
            "insufficient_probe_families", "grouped boundaries need two families"
        )
    predictions = {}
    folds = []
    for held in families:
        train = tuple(row for row in values if row["family_id"] != held)
        test = tuple(row for row in values if row["family_id"] == held)
        fitted = fit_common_order_boundaries(
            tuple(ranks[row["key"]] for row in train),
            tuple(row["class_index"] for row in train),
        )
        observed = apply_common_order_boundaries(
            tuple(ranks[row["key"]] for row in test), fitted["boundaries"]
        )
        for row, prediction in zip(test, observed):
            predictions[row["key"]] = prediction
        folds.append({
            "held_out_family_id": held,
            "boundaries": fitted["boundaries"],
            "train_family_ids": sorted(set(row["family_id"] for row in train)),
            "same_gap_policy": fitted["same_gap_policy"],
        })
    ordered = tuple(sorted(values, key=lambda row: row["key"]))
    truth = tuple(row["class_index"] for row in ordered)
    predicted = tuple(predictions[row["key"]] for row in ordered)
    prediction_view = dict(predictions)
    return {
        "split": "leave_one_physical_family_out",
        "baseline_identity": "A/B grouped scalar common finite-precision weak order",
        "coordinate_selection": "none_single_prospectively_frozen_common_fit",
        "metrics": classification_metrics(truth, predicted),
        "predictions_by_key": prediction_view,
        "coordinate_prediction_equivalence": {
            "status": "identical_by_construction_single_common_fit",
            "A_physical_predictions_by_key": dict(prediction_view),
            "B_raw_logit_predictions_by_key": dict(prediction_view),
        },
        "folds": folds,
        "common_order": common_order,
    }


def _closed_form_scalar_analysis(rows, operation_type):
    """Report separate coordinate analyses without choosing the common baseline."""

    selected = tuple(sorted(rows, key=lambda row: row["key"]))
    labels = tuple(row["class_index"] for row in selected)
    physical = tuple(float(row["A_physical"]) for row in selected)
    logits = tuple(float(row["B_raw_logit"]) for row in selected)
    fidelity_pass_by_key = {
        row["key"]: abs(
            row["A_physical"]
            - OPERATION_GRIDS[operation_type][row["class_index"]]
        ) < FIDELITY_ERROR_LIMITS[operation_type]
        for row in selected
    }
    family_pass = {}
    for row in selected:
        family_pass.setdefault(row["family_id"], True)
        family_pass[row["family_id"]] = (
            family_pass[row["family_id"]] and fidelity_pass_by_key[row["key"]]
        )
    ranges = {}
    for name, scalars in (("A_physical", physical), ("B_raw_logit", logits)):
        ranges[name] = {}
        for label in range(5):
            observed = tuple(
                value for value, truth in zip(scalars, labels) if truth == label
            )
            ranges[name][str(label)] = {
                "support": len(observed),
                "minimum": min(observed) if observed else None,
                "maximum": max(observed) if observed else None,
            }
    return {
        "operation_type": operation_type,
        "class_order": list(OPERATION_GRIDS[operation_type]),
        "per_class_scalar_ranges": ranges,
        "rank_order_A": [
            {"key": selected[index]["key"], "value": physical[index],
             "class_index": labels[index]}
            for index in sorted(
                range(len(selected)), key=lambda index: (physical[index], index)
            )
        ],
        "rank_order_B": [
            {"key": selected[index]["key"], "value": logits[index],
             "class_index": labels[index]}
            for index in sorted(
                range(len(selected)), key=lambda index: (logits[index], index)
            )
        ],
        "existing_fidelity_gate": {
            "operation_accuracy": sum(fidelity_pass_by_key.values()) / len(selected),
            "family_all_operations_accuracy": sum(family_pass.values()) / len(family_pass),
            "comparison": "unrounded_absolute_physical_error_strictly_below_limit",
        },
        "nearest_grid": classification_metrics(
            labels, nearest_grid_predictions(physical, operation_type)
        ),
        "A_scalar_only_resubstitution_ceiling": fit_monotonic_thresholds(
            physical, labels
        ),
        "B_monotonic_resubstitution_ceiling": fit_monotonic_thresholds(
            logits, labels
        ),
        "A_grouped_threshold": grouped_threshold_predictions(
            selected, value_key="A_physical"
        ),
        "B_grouped_threshold": grouped_threshold_predictions(
            selected, value_key="B_raw_logit"
        ),
        "coordinate_specific_grouped_role": (
            "descriptive_only_not_used_for_primary_baseline_or_selection"
        ),
    }


def _scalar_targets(rows, targets):
    raw_values = []
    balanced_values = []
    observed_analysis = None
    observed_common = None
    common_order = common_scalar_order(rows)
    for target_index in range(targets.size(0)):
        relabeled = []
        for row_index, row in enumerate(rows):
            value = dict(row)
            value["class_index"] = int(targets[target_index, row_index])
            relabeled.append(value)
        common = common_grouped_scalar_predictions(relabeled, common_order)
        raw_values.append(float(common["metrics"]["accuracy"]))
        balanced_values.append(float(common["metrics"]["balanced_accuracy"]))
        if target_index == 0:
            observed_analysis = _finite_json_value(
                _closed_form_scalar_analysis(relabeled, rows[0]["operation_type"])
            )
            ordered = tuple(sorted(relabeled, key=lambda row: row["key"]))
            predicted = [common["predictions_by_key"][row["key"]] for row in ordered]
            observed_common = _finite_json_value(dict(common))
            observed_common.update({
                "status": "verified_single_common_order_fit",
                "identity": "A/B grouped scalar",
                "metrics": metric_record(
                    [row["class_index"] for row in ordered], predicted
                ),
                "predictions_by_operation_key": dict(common["predictions_by_key"]),
            })
    return {
        "analysis": observed_analysis,
        "common_grouped": observed_common,
        "raw_values": raw_values,
        "balanced_values": balanced_values,
    }


def _primary_definitions():
    definitions = []
    for arm in PROBE_ARMS:
        definitions.extend([
            ("{}:extrude:C_access".format(arm), arm, "extrude", "access", "C", None),
            ("{}:extrude:D-gated_access".format(arm), arm, "extrude", "access", "D-gated", None),
            ("{}:extrude:C_minus_AB".format(arm), arm, "extrude", "difference", "C", "A/B"),
            ("{}:extrude:D-gated_minus_C".format(arm), arm, "extrude", "difference", "D-gated", "C"),
            ("{}:revolve:C_access".format(arm), arm, "revolve", "access", "C", None),
            ("{}:revolve:P_access".format(arm), arm, "revolve", "access", "P", None),
            ("{}:revolve:C_minus_AB".format(arm), arm, "revolve", "difference", "C", "A/B"),
            ("{}:revolve:P_minus_C".format(arm), arm, "revolve", "difference", "P", "C"),
        ])
    return tuple(definitions)


def _interpret_primary(primary, pipeline_lookup):
    interpretations = []
    by_id = {row["hypothesis_id"]: row for row in primary}
    for arm in PROBE_ARMS:
        for operation_type, upstream in (("extrude", "D-gated"), ("revolve", "P")):
            access = {}
            for feature in ("C", upstream):
                identity = "{}:{}:{}_access".format(arm, operation_type, feature)
                row = by_id[identity]
                pipeline = pipeline_lookup[(arm, operation_type, "full", feature)]["record"]
                access[feature] = (
                    pipeline["lofo"]["metrics"]["balanced_accuracy"] >= ACCESS_BALANCED_MIN
                    and row["raw_accuracy"]["observed_statistic"]
                    > row["raw_accuracy"]["permutation_95th_percentile"]
                    and row["raw_accuracy"]["global_centered_maxT_adjusted_p_value"] <= P_VALUE_MAX
                    and row["balanced_accuracy"]["global_centered_maxT_adjusted_p_value"] <= P_VALUE_MAX
                )
            c_diff = by_id["{}:{}:C_minus_AB".format(arm, operation_type)]
            decoder_statement = (
                access["C"]
                and c_diff["raw_accuracy"]["observed_statistic"] >= MATERIAL_MARGIN
                and c_diff["balanced_accuracy"]["observed_statistic"] >= MATERIAL_MARGIN
                and c_diff["raw_accuracy"]["global_centered_maxT_adjusted_p_value"] <= P_VALUE_MAX
                and c_diff["balanced_accuracy"]["global_centered_maxT_adjusted_p_value"] <= P_VALUE_MAX
            )
            upstream_diff_id = "{}:{}:{}_minus_C".format(arm, operation_type, upstream)
            upstream_diff = by_id[upstream_diff_id]
            conditioning = (
                access[upstream]
                and not access["C"]
                and upstream_diff["raw_accuracy"]["observed_statistic"] >= MATERIAL_MARGIN
                and upstream_diff["balanced_accuracy"]["observed_statistic"] >= MATERIAL_MARGIN
                and upstream_diff["raw_accuracy"]["global_centered_maxT_adjusted_p_value"] <= P_VALUE_MAX
                and upstream_diff["balanced_accuracy"]["global_centered_maxT_adjusted_p_value"] <= P_VALUE_MAX
            )
            interpretations.append({
                "arm": arm,
                "operation_type": operation_type,
                "C_controlled_accessibility_evidence": access["C"],
                "upstream_feature": upstream,
                "upstream_controlled_accessibility_evidence": access[upstream],
                "decoder_state_materially_exceeds_scalar_path": decoder_statement,
                "decoder_conditioning_candidate_supported": conditioning,
                "licensed_decoder_statement": (
                    "Magnitude-class information reaches the operation decoder state but is not exploited by the existing scalar output path."
                    if decoder_statement else None
                ),
                "licensed_conditioning_statement": (
                    "Magnitude information is accessible at prequant but not linearly accessible at C under the tested readouts; the transformation or conditioning between them is a candidate failure region."
                    if conditioning else None
                ),
            })
    return interpretations


def _combined_summaries(pipeline_lookup):
    summaries = []
    for arm in PROBE_ARMS:
        for identity, features in (
            ("C_across_operation_types", {"extrude": "C", "revolve": "C"}),
            ("primary_upstream_across_operation_types", {"extrude": "D-gated", "revolve": "P"}),
        ):
            truths = []
            predictions = []
            family_correct = {}
            for operation_type in OPERATION_TYPES:
                run = pipeline_lookup[(arm, operation_type, "full", features[operation_type])]
                record = run["record"]
                rows = run["rows"]
                for row in rows:
                    truth = row["class_index"]
                    prediction = record["lofo"]["predictions_by_operation_key"][row["key"]]
                    truths.append(truth)
                    predictions.append(prediction)
                    family_correct.setdefault(row["family_id"], True)
                    family_correct[row["family_id"]] = (
                        family_correct[row["family_id"]] and truth == prediction
                    )
            summaries.append({
                "arm": arm,
                "identity": identity,
                "status": "descriptive_not_a_primary_hypothesis",
                "operation_metrics": metric_record(truths, predictions),
                "physical_family_all_operations_accuracy": (
                    sum(family_correct.values()) / float(len(family_correct))
                ),
                "physical_family_count": len(family_correct),
            })
    return summaries


def run_closed_form_analysis(feature_rows, label_rows, permutation_count, *,
                             execution_source=None, environment=None,
                             input_hashes=None, emit=None):
    """Run the complete 24-pipeline closed-form analysis on supplied rows."""

    if torch is None:
        raise RuntimeError("closed-form readout requires PyTorch")
    if int(permutation_count) < 1:
        raise GraphEncoderError("invalid_permutation_count", "at least one null required")
    started = time.perf_counter()
    timing = {"factorization_seconds": 0.0, "factorization_count": 0}
    pipeline_lookup = {}
    scalar_lookup = {}
    records = []
    structured = []
    for arm in PROBE_ARMS:
        joined = join_detached_rows(feature_rows, label_rows, arm)
        full_extrude_rows = None
        for operation_type in OPERATION_TYPES:
            rows = select_cohort(joined, operation_type, "full")
            if operation_type == "extrude":
                full_extrude_rows = rows
            targets = target_matrix(rows, permutation_count)
            scalar = _scalar_targets(rows, targets)
            scalar_lookup[(arm, operation_type)] = scalar
            features = (
                ("C", "P", "D-additive", "D-gated", "context-only")
                if operation_type == "extrude" else ("C", "P")
            )
            for feature in features:
                pipeline_started = time.perf_counter()
                factorization_count_started = timing["factorization_count"]
                factorization_seconds_started = timing["factorization_seconds"]
                run = fit_ridge_pipeline(
                    rows, feature, targets, arm=arm, operation_type=operation_type,
                    cohort="full", timing=timing,
                )
                run["record"]["timing"] = {
                    "elapsed_seconds": time.perf_counter() - pipeline_started,
                    "factorization_count": (
                        timing["factorization_count"] - factorization_count_started
                    ),
                    "factorization_seconds": (
                        timing["factorization_seconds"] - factorization_seconds_started
                    ),
                }
                run["rows"] = rows
                pipeline_lookup[(arm, operation_type, "full", feature)] = run
                records.append(run["record"])
                if emit is not None:
                    emit({
                        "event": "closed_form_pipeline_completed",
                        "arm": arm,
                        "operation_type": operation_type,
                        "cohort": "full",
                        "feature_identity": feature,
                        "permutation_targets_completed": permutation_count,
                    })
                    emit({
                        "event": "closed_form_permutation_progress",
                        "arm": arm,
                        "operation_type": operation_type,
                        "cohort": "full",
                        "feature_identity": feature,
                        "completed_permutation_targets": permutation_count,
                    })
            if operation_type == "revolve":
                structured.extend(revolve_degeneracy_records(arm))
                structured.append(revolve_sensitivity_record(arm))
        sensitivity_rows = select_cohort(joined, "extrude", "single_extrusion_E_RE")
        sensitivity_targets = target_matrix(
            sensitivity_rows, permutation_count,
            permutation_source_rows=full_extrude_rows,
        )
        for feature in ("C", "P", "D-additive", "D-gated", "context-only"):
            pipeline_started = time.perf_counter()
            factorization_count_started = timing["factorization_count"]
            factorization_seconds_started = timing["factorization_seconds"]
            run = fit_ridge_pipeline(
                sensitivity_rows, feature, sensitivity_targets, arm=arm,
                operation_type="extrude", cohort="single_extrusion_E_RE", timing=timing,
            )
            run["record"]["timing"] = {
                "elapsed_seconds": time.perf_counter() - pipeline_started,
                "factorization_count": (
                    timing["factorization_count"] - factorization_count_started
                ),
                "factorization_seconds": (
                    timing["factorization_seconds"] - factorization_seconds_started
                ),
            }
            run["rows"] = sensitivity_rows
            pipeline_lookup[(arm, "extrude", "single_extrusion_E_RE", feature)] = run
            records.append(run["record"])
            if emit is not None:
                emit({
                    "event": "closed_form_pipeline_completed",
                    "arm": arm,
                    "operation_type": "extrude",
                    "cohort": "single_extrusion_E_RE",
                    "feature_identity": feature,
                    "permutation_targets_completed": permutation_count,
                })
                emit({
                    "event": "closed_form_permutation_progress",
                    "arm": arm,
                    "operation_type": "extrude",
                    "cohort": "single_extrusion_E_RE",
                    "feature_identity": feature,
                    "completed_permutation_targets": permutation_count,
                })
    definitions = _primary_definitions()
    metric_families = {"raw_accuracy": {}, "balanced_accuracy": {}}
    metadata = {}
    for identity, arm, operation_type, kind, left, right in definitions:
        left_run = pipeline_lookup[(arm, operation_type, "full", left)]
        for metric_name, array_name in (
            ("raw_accuracy", "raw_values"), ("balanced_accuracy", "balanced_values")
        ):
            values = left_run[array_name]
            if kind == "difference":
                if right == "A/B":
                    right_values = scalar_lookup[(arm, operation_type)][array_name]
                else:
                    right_values = pipeline_lookup[
                        (arm, operation_type, "full", right)
                    ][array_name]
                values = [left_value - right_value for left_value, right_value in zip(
                    values, right_values
                )]
            metric_families[metric_name][identity] = values
        metadata[identity] = {
            "hypothesis_id": identity,
            "arm": arm,
            "operation_type": operation_type,
            "kind": kind,
            "left": left,
            "right": right,
            "difference_recomputed_inside_every_synchronized_permutation": kind == "difference",
        }
    corrected = {
        metric_name: centered_global_max_t(values)
        for metric_name, values in metric_families.items()
    }
    primary = []
    for identity, unused_arm, unused_type, unused_kind, unused_left, unused_right in definitions:
        row = dict(metadata[identity])
        for metric_name in ("raw_accuracy", "balanced_accuracy"):
            row[metric_name] = corrected[metric_name]["hypotheses"][identity]
        primary.append(row)
    # Every executed pipeline gets a marginal permutation record; maxT is only primary.
    primary_access_by_pipeline = {
        (row["arm"], row["operation_type"], row["left"]): row
        for row in primary if row["kind"] == "access"
    }
    for record in records:
        run = pipeline_lookup[(
            record["arm"], record["operation_type"], record["cohort"],
            record["feature_identity"],
        )]
        record["permutation"] = {
            "raw_accuracy": marginal_permutation_result(run["raw_values"]),
            "balanced_accuracy": marginal_permutation_result(run["balanced_values"]),
            "centered_maxT": {"status": "secondary_not_in_primary_family"},
        }
        primary_row = primary_access_by_pipeline.get((
            record["arm"], record["operation_type"], record["feature_identity"]
        )) if record["cohort"] == "full" else None
        if primary_row is not None:
            record["permutation"]["centered_maxT"] = {
                "status": "primary_global_family",
                "raw_accuracy": primary_row["raw_accuracy"],
                "balanced_accuracy": primary_row["balanced_accuracy"],
            }
    scalar_records = []
    for arm in PROBE_ARMS:
        for operation_type in OPERATION_TYPES:
            scalar = scalar_lookup[(arm, operation_type)]
            scalar_records.append({
                "arm": arm,
                "operation_type": operation_type,
                "cohort": "full",
                "A_and_B_analysis": scalar["analysis"],
                "A_B_grouped_scalar": scalar["common_grouped"],
                "permutation": {
                    "raw_accuracy": marginal_permutation_result(scalar["raw_values"]),
                    "balanced_accuracy": marginal_permutation_result(
                        scalar["balanced_values"]
                    ),
                },
            })
    elapsed = time.perf_counter() - started
    return {
        "schema_version": READOUT_RESULT_VERSION,
        "protocol_version": READOUT_PROTOCOL_VERSION,
        "execution_source": dict(execution_source or {"synthetic": True}),
        "input_hashes": dict(input_hashes or {"synthetic": True}),
        "environment": dict(environment or {
            "python": platform.python_version(),
            "pytorch": str(torch.__version__),
            "device": "cpu",
            "cuda_available": bool(torch.cuda.is_available()),
            "cpu_threads": int(torch.get_num_threads()),
        }),
        "permutation_count": int(permutation_count),
        "primary_hypothesis_count": 16,
        "primary_hypotheses": primary,
        "global_maxT": {
            metric_name: value["max_null_summary"]
            for metric_name, value in corrected.items()
        },
        "executed_pipeline_count": len(records),
        "executed_pipelines": records,
        "structured_nonexecuted_count": len(structured),
        "structured_nonexecuted": structured,
        "scalar_analyses": scalar_records,
        "combined_descriptive_summaries": _combined_summaries(pipeline_lookup),
        "prospective_interpretation": _interpret_primary(primary, pipeline_lookup),
        "timing": {
            "elapsed_seconds": elapsed,
            "factorization_seconds": timing["factorization_seconds"],
            "factorization_count": timing["factorization_count"],
            "factorization_reuse": (
                "each_X_fold_SVD_reused_exactly_for_all_lambdas_and_true_permuted_RHS"
            ),
        },
        "limitations": list(LIMITATIONS),
        "authority": {
            "formal_adr0011_representation_probe_completed": False,
            "representation_screen_completed_or_reinterpreted": False,
            "model_or_decoder_repair_authorized": False,
            "stage6_authorized": False,
            "c8_authorized": False,
            "protected_access_authorized": False,
        },
    }


def validate_readout_results(results, expected_permutation_count=None):
    if not isinstance(results, dict):
        raise GraphEncoderError("invalid_readout_results", "results are absent")
    if (
        results.get("schema_version") != READOUT_RESULT_VERSION
        or results.get("protocol_version") != READOUT_PROTOCOL_VERSION
        or results.get("primary_hypothesis_count") != 16
        or len(results.get("primary_hypotheses", ())) != 16
        or results.get("executed_pipeline_count") != 24
        or len(results.get("executed_pipelines", ())) != 24
        or results.get("structured_nonexecuted_count") != 8
        or len(results.get("structured_nonexecuted", ())) != 8
        or results.get("limitations") != list(LIMITATIONS)
    ):
        raise GraphEncoderError("invalid_readout_results", "result identity/count differs")
    if (
        expected_permutation_count is not None
        and results.get("permutation_count") != int(expected_permutation_count)
    ):
        raise GraphEncoderError("invalid_readout_results", "permutation count differs")
    identities = {
        (row["arm"], row["operation_type"], row["cohort"], row["feature_identity"])
        for row in results["executed_pipelines"]
    }
    expected = set()
    for arm in PROBE_ARMS:
        for feature in ("C", "P", "D-additive", "D-gated", "context-only"):
            expected.add((arm, "extrude", "full", feature))
            expected.add((arm, "extrude", "single_extrusion_E_RE", feature))
        for feature in ("C", "P"):
            expected.add((arm, "revolve", "full", feature))
    if identities != expected:
        raise GraphEncoderError("invalid_readout_results", "pipeline coverage differs")
    primary_ids = tuple(row.get("hypothesis_id") for row in results["primary_hypotheses"])
    if primary_ids != tuple(row[0] for row in _primary_definitions()):
        raise GraphEncoderError("invalid_readout_results", "primary coverage/order differs")
    structured = results["structured_nonexecuted"]
    degeneracies = {
        (row.get("arm"), row.get("feature_identity"))
        for row in structured
        if row.get("status") == "degenerate_by_construction"
    }
    not_applicable = {
        row.get("arm") for row in structured
        if row.get("status") == "not_applicable_full_cohort_already_one_to_one"
    }
    if (
        degeneracies != {
            (arm, feature) for arm in PROBE_ARMS
            for feature in ("D-additive", "D-gated", "context-only")
        }
        or not_applicable != set(PROBE_ARMS)
    ):
        raise GraphEncoderError("invalid_readout_results", "structured coverage differs")
    if results.get("authority") != {
        "formal_adr0011_representation_probe_completed": False,
        "representation_screen_completed_or_reinterpreted": False,
        "model_or_decoder_repair_authorized": False,
        "stage6_authorized": False,
        "c8_authorized": False,
        "protected_access_authorized": False,
    }:
        raise GraphEncoderError("invalid_readout_results", "authority differs")
    return results


def synthetic_fixture():
    """Create the exact cohort in memory for tests and timing only."""

    labels = []
    features = []
    patterns = {
        "E": ((0, "extrude"),),
        "R": ((0, "revolve"),),
        "EE": ((0, "extrude"), (1, "extrude")),
        "RE": ((0, "revolve"), (1, "extrude")),
    }
    family_index = 0
    for template in ("E", "R", "EE", "RE"):
        for unused in range(8):
            family_id = "synthetic-family-{:02d}".format(family_index)
            family_p = tuple(
                math.sin((family_index + 1) * (index + 1) / 17.0)
                for index in range(32)
            )
            for operation_index, operation_type in patterns[template]:
                label_key = "{}:{}:{}".format(
                    family_id, operation_index, operation_type
                )
                class_index = (family_index + operation_index) % 5
                labels.append({
                    "class_index": class_index,
                    "class_physical_value": OPERATION_GRIDS[operation_type][class_index],
                    "family_id": family_id,
                    "label_key": label_key,
                    "operation_index": operation_index,
                    "operation_template": template,
                    "operation_type": operation_type,
                })
                type_context = (1.0, 0.0) if operation_type == "extrude" else (0.0, 1.0)
                slot_context = (1.0, 0.0) if operation_index == 0 else (0.0, 1.0)
                normalized = OPERATION_GRIDS[operation_type][class_index] / (
                    4.0 if operation_type == "extrude" else 360.0
                )
                raw_logit = math.log(normalized / (1.0 - normalized)) if normalized < 1.0 else 40.0
                for arm_index, arm in enumerate(PROBE_ARMS):
                    c_value = tuple(
                        family_p[index] + 0.05 * operation_index + 0.01 * arm_index
                        for index in range(32)
                    )
                    features.append({
                        "key": "{}:{}".format(arm, label_key),
                        "label_key": label_key,
                        "arm": arm,
                        "family_id": family_id,
                        "operation_index": operation_index,
                        "operation_type": operation_type,
                        "A_normalized": normalized,
                        "A_physical": OPERATION_GRIDS[operation_type][class_index],
                        "B_raw_logit": raw_logit,
                        "C": c_value,
                        "D": family_p + type_context + slot_context,
                    })
            family_index += 1
    return tuple(features), tuple(labels)


def synthetic_timing_preflight(elapsed_repository_preflight_seconds, emit=None):
    if torch is None:
        raise RuntimeError("synthetic timing requires PyTorch")
    if emit is not None:
        emit({"event": "closed_form_synthetic_timing_started",
              "sample_permutations": TIMING_SAMPLE_PERMUTATIONS})
    features, labels = synthetic_fixture()
    timing_started = time.perf_counter()
    baseline_results = run_closed_form_analysis(
        features, labels, TIMING_BASELINE_PERMUTATIONS,
        execution_source={"synthetic_timing": True}, emit=None,
    )
    baseline_measured = time.perf_counter() - timing_started
    sample_started = time.perf_counter()
    results = run_closed_form_analysis(
        features, labels, TIMING_SAMPLE_PERMUTATIONS,
        execution_source={"synthetic_timing": True}, emit=None,
    )
    measured = time.perf_counter() - sample_started
    timing_elapsed = time.perf_counter() - timing_started
    target_delta = TIMING_SAMPLE_PERMUTATIONS - TIMING_BASELINE_PERMUTATIONS
    per_target = max(0.0, measured - baseline_measured) / float(target_delta)
    fixed = max(
        results["timing"]["factorization_seconds"],
        baseline_measured - per_target * (TIMING_BASELINE_PERMUTATIONS + 1),
    )

    def projected(permutations):
        analysis = fixed + per_target * (permutations + 1)
        return (
            float(elapsed_repository_preflight_seconds)
            + timing_elapsed
            + TIMING_SAFETY_FACTOR * analysis
            + TIMING_FINALIZATION_SECONDS
        )

    projected_999 = projected(999)
    projected_499 = projected(499)
    decision = select_permutation_count(projected_999, projected_499)
    record = {
        "protocol_version": READOUT_PROTOCOL_VERSION,
        "synthetic_only": True,
        "actual_cohort_cardinalities": copy.deepcopy(COHORT_CONTRACTS),
        "feature_dimensions_exercised": {
            name: value["nominal_dimension"] for name, value in FEATURE_CONTRACTS.items()
        },
        "nested_lambda_selection_exercised": True,
        "batched_permutation_targets_exercised": True,
        "metrics_and_global_maxT_exercised": True,
        "sample_permutation_count": TIMING_SAMPLE_PERMUTATIONS,
        "baseline_permutation_count": TIMING_BASELINE_PERMUTATIONS,
        "measured_baseline_complete_seconds": baseline_measured,
        "measured_complete_sample_seconds": measured,
        "measured_timing_preflight_total_seconds": timing_elapsed,
        "estimated_fixed_analysis_seconds": fixed,
        "measured_factorization_seconds": results["timing"]["factorization_seconds"],
        "measured_factorization_count": results["timing"]["factorization_count"],
        "estimated_variable_seconds_per_target": per_target,
        "elapsed_repository_preflight_seconds": float(elapsed_repository_preflight_seconds),
        "projected_finalization_and_verification_seconds": TIMING_FINALIZATION_SECONDS,
        "safety_factor": TIMING_SAFETY_FACTOR,
        "projection_limit_seconds": TIMING_LIMIT_SECONDS,
        "projected_complete_seconds": {"999": projected_999, "499": projected_499},
        **decision,
    }
    if emit is not None:
        emit({"event": "closed_form_synthetic_timing_completed", **record})
        emit({
            "event": "closed_form_timing_gate_{}".format(
                "passed" if decision["status"] == "pass" else "aborted"
            ),
            "selected_permutation_count": decision["selected_permutation_count"],
            "reason": decision["reason"],
        })
    return record


class EventRecorder:
    def __init__(self, path):
        self.path = Path(path)
        self.events = []

    def emit(self, record):
        value = dict(record)
        value.setdefault("protocol_version", READOUT_PROTOCOL_VERSION)
        value.setdefault("wall_time_unix", time.time())
        payload = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode(
            "utf-8"
        )
        with self.path.open("ab") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        self.events.append(value)
        print(payload.decode("utf-8").rstrip("\n"), flush=True)


def _regular_files(root, excluded=()):
    excluded_values = set(excluded)
    paths = []
    for path in Path(root).rglob("*"):
        if path.is_symlink():
            raise GraphEncoderError("invalid_readout_artifact", "symlinks are forbidden")
        if path.is_file():
            relative = path.relative_to(root).as_posix()
            if relative not in excluded_values:
                paths.append(relative)
    return tuple(sorted(paths))


def prepare_readout_output(output_dir, *, repository_root, input_dir, job_id=None):
    identity = validate_slurm_job_id(job_id)
    raw = Path(output_dir)
    if raw.exists() or raw.is_symlink():
        raise GraphEncoderError("readout_output_exists", "output must be new")
    final = raw.resolve()
    if any(_is_within(final, Path(root).resolve()) for root in (repository_root, input_dir)):
        raise GraphEncoderError("unsafe_readout_output_location", "output location differs")
    final.parent.mkdir(parents=True, exist_ok=True)
    staging = final.with_name(final.name + ".incomplete-" + identity)
    if staging.exists() or staging.is_symlink():
        raise GraphEncoderError("readout_staging_exists", "staging already exists")
    staging.mkdir()
    return final, staging


def finalize_readout_artifact(staging_dir, final_dir):
    staging = Path(staging_dir)
    final = Path(final_dir)
    ordinary = _regular_files(staging, excluded=("artifact_manifest.json", "SHA256SUMS"))
    if ordinary != EXPECTED_ORDINARY_FILES:
        raise GraphEncoderError("incomplete_readout_artifact", "ordinary files differ")
    manifest = {
        "schema_version": READOUT_ARTIFACT_VERSION,
        "protocol_version": READOUT_PROTOCOL_VERSION,
        "source_feature_artifact": {
            "git_commit": SOURCE_FEATURE_COMMIT,
            "slurm_job_id": SOURCE_FEATURE_JOB_ID,
            "sha256": dict(READOUT_INPUT_HASHES),
        },
        "artifacts": [
            {"path": name, "byte_size": (staging / name).stat().st_size,
             "sha256": _file_sha256(staging / name)}
            for name in ordinary
        ],
        "limitations": list(LIMITATIONS),
        "authority": {
            "model_or_decoder_repair_authorized": False,
            "stage6_authorized": False,
            "c8_authorized": False,
            "protected_access_authorized": False,
        },
    }
    _atomic_write_json(staging / "artifact_manifest.json", manifest)
    checksum_paths = _regular_files(staging, excluded=("SHA256SUMS",))
    checksum = "".join(
        "{}  {}\n".format(_file_sha256(staging / name), name)
        for name in checksum_paths
    )
    _atomic_write_bytes(staging / "SHA256SUMS", checksum.encode("utf-8"))
    verify_readout_artifact(staging, allow_incomplete_name=True)
    os.replace(str(staging), str(final))
    descriptor = os.open(str(final.parent), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return final


def verify_readout_artifact(path, *, allow_incomplete_name=False,
                            expected_commit=None, expected_slurm_job_id=None):
    root = Path(path)
    if not root.is_dir() or root.is_symlink():
        raise GraphEncoderError("invalid_readout_artifact", "root differs")
    if ".incomplete-" in root.name and not allow_incomplete_name:
        raise GraphEncoderError("invalid_readout_artifact", "incomplete is not final")
    if _regular_files(root) != tuple(sorted(EXPECTED_FILES)):
        raise GraphEncoderError("incomplete_readout_artifact", "five files required")
    manifest = json.loads((root / "artifact_manifest.json").read_text("utf-8"))
    if (
        manifest.get("schema_version") != READOUT_ARTIFACT_VERSION
        or manifest.get("protocol_version") != READOUT_PROTOCOL_VERSION
        or manifest.get("source_feature_artifact") != {
            "git_commit": SOURCE_FEATURE_COMMIT,
            "slurm_job_id": SOURCE_FEATURE_JOB_ID,
            "sha256": READOUT_INPUT_HASHES,
        }
    ):
        raise GraphEncoderError("invalid_readout_artifact", "manifest differs")
    listed = []
    for row in manifest.get("artifacts", ()):
        if not isinstance(row, dict) or set(row) != {"path", "byte_size", "sha256"}:
            raise GraphEncoderError("invalid_readout_artifact", "manifest row differs")
        relative = _safe_relative_path(row["path"])
        target = root / relative
        if (
            not target.is_file() or target.is_symlink()
            or target.stat().st_size != row["byte_size"]
            or _file_sha256(target) != row["sha256"]
        ):
            raise GraphEncoderError("readout_artifact_integrity_failure", "hash differs")
        listed.append(relative)
    if tuple(listed) != EXPECTED_ORDINARY_FILES:
        raise GraphEncoderError("invalid_readout_artifact", "manifest coverage differs")
    checksum_paths = []
    checksum_bytes = (root / "SHA256SUMS").read_bytes()
    if not checksum_bytes.endswith(b"\n"):
        raise GraphEncoderError("invalid_readout_artifact", "checksum final LF absent")
    for line in checksum_bytes.decode("utf-8").splitlines():
        parts = line.split("  ", 1)
        if len(parts) != 2 or len(parts[0]) != 64:
            raise GraphEncoderError("invalid_readout_artifact", "checksum row differs")
        relative = _safe_relative_path(parts[1])
        if _file_sha256(root / relative) != parts[0]:
            raise GraphEncoderError("readout_artifact_integrity_failure", "checksum differs")
        checksum_paths.append(relative)
    if tuple(checksum_paths) != _regular_files(root, excluded=("SHA256SUMS",)):
        raise GraphEncoderError("invalid_readout_artifact", "checksum coverage differs")
    resolved = json.loads((root / "resolved_config.json").read_text("utf-8"))
    results = json.loads((root / "readout_results.json").read_text("utf-8"))
    validate_readout_results(results, resolved.get("selected_permutation_count"))
    events = _read_canonical_jsonl(root / "metrics.jsonl")
    if not events or events[-1].get("event") != "closed_form_readout_completed":
        raise GraphEncoderError("invalid_readout_artifact", "terminal event absent")
    if resolved.get("protocol_version") != READOUT_PROTOCOL_VERSION:
        raise GraphEncoderError("invalid_readout_artifact", "resolved protocol differs")
    if expected_commit is not None and resolved.get("source", {}).get("git_commit") != expected_commit:
        raise GraphEncoderError("invalid_readout_artifact", "commit differs")
    if (
        expected_slurm_job_id is not None
        and resolved.get("runtime", {}).get("slurm_job_id") != str(expected_slurm_job_id)
    ):
        raise GraphEncoderError("invalid_readout_artifact", "job differs")
    return {
        "protocol_version": READOUT_PROTOCOL_VERSION,
        "file_count": 5,
        "pipeline_count": results["executed_pipeline_count"],
        "primary_hypothesis_count": 16,
        "integrity_verified": True,
    }


def _require_runtime():
    if torch is None:
        raise RuntimeError("closed-form readout requires PyTorch")
    job_id = validate_slurm_job_id()
    if platform.python_version() != "3.8.13":
        raise GraphEncoderError("invalid_readout_runtime", "Python 3.8.13 required")
    if str(torch.__version__).split("+")[0] != "1.11.0":
        raise GraphEncoderError("invalid_readout_runtime", "PyTorch 1.11.0 required")
    if torch.cuda.is_available() or torch.get_num_threads() != 1:
        raise GraphEncoderError("invalid_readout_runtime", "one-thread CPU required")
    return job_id


def run_authoritative_readout(*, input_dir, output_dir, repository_root,
                              expected_commit, timing_record_path):
    job_id = _require_runtime()
    source = _source_identity(repository_root, expected_commit)
    timing_record = json.loads(Path(timing_record_path).read_text("utf-8"))
    if (
        timing_record.get("protocol_version") != READOUT_PROTOCOL_VERSION
        or timing_record.get("status") != "pass"
        or timing_record.get("selected_permutation_count") not in PERMUTATION_LADDER
    ):
        raise GraphEncoderError("invalid_timing_authorization", "timing gate did not pass")
    final, staging = prepare_readout_output(
        output_dir, repository_root=repository_root, input_dir=input_dir, job_id=job_id
    )
    recorder = EventRecorder(staging / "metrics.jsonl")
    recorder.emit({
        "event": "closed_form_readout_started",
        "source": source,
        "timing_gate": timing_record,
        "limitations": list(LIMITATIONS),
        **readout_access_declarations(),
    })
    inputs = load_readout_inputs(input_dir)
    recorder.emit({
        "event": "closed_form_preserved_inputs_verified",
        "input_hashes": inputs["input_hashes"],
        "label_join_order": inputs["label_join_order"],
        **readout_access_declarations(input_accessed=True),
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
    results = run_closed_form_analysis(
        inputs["feature_rows"], inputs["label_rows"],
        timing_record["selected_permutation_count"],
        execution_source=execution_source, environment=runtime,
        input_hashes=inputs["input_hashes"], emit=recorder.emit,
    )
    validate_readout_results(results, timing_record["selected_permutation_count"])
    _atomic_write_json(staging / "readout_results.json", results)
    access = readout_access_declarations(input_accessed=True, completed=True)
    resolved = {
        "protocol_version": READOUT_PROTOCOL_VERSION,
        "artifact_version": READOUT_ARTIFACT_VERSION,
        "results_version": READOUT_RESULT_VERSION,
        "source": source,
        "source_feature_artifact": {
            "git_commit": SOURCE_FEATURE_COMMIT,
            "slurm_job_id": SOURCE_FEATURE_JOB_ID,
            "sha256": dict(READOUT_INPUT_HASHES),
        },
        "input_hashes": inputs["input_hashes"],
        "selected_permutation_count": timing_record["selected_permutation_count"],
        "timing_gate": timing_record,
        "runtime": runtime,
        "contract": readout_contract(),
        "access": access,
        "limitations": list(LIMITATIONS),
    }
    _atomic_write_json(staging / "resolved_config.json", resolved)
    recorder.emit({
        "event": "closed_form_readout_completed",
        "pipeline_count": results["executed_pipeline_count"],
        "primary_hypothesis_count": 16,
        "complete_negative_result_is_success": True,
        "limitations": list(LIMITATIONS),
        **access,
    })
    _verify_source_unchanged(source, repository_root, expected_commit)
    finalize_readout_artifact(staging, final)
    verified = verify_readout_artifact(
        final, expected_commit=expected_commit, expected_slurm_job_id=job_id
    )
    print(json.dumps({
        "event": "closed_form_artifact_validated", **verified
    }, sort_keys=True, separators=(",", ":")), flush=True)
    return {"artifact_path": str(final), "artifact_verification": verified}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    timing_parser = subparsers.add_parser("synthetic-timing")
    timing_parser.add_argument("--elapsed-repository-preflight-seconds", type=float, required=True)
    timing_parser.add_argument("--output", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--input-dir", required=True)
    run_parser.add_argument("--output-dir", required=True)
    run_parser.add_argument("--repository-root", required=True)
    run_parser.add_argument("--expected-commit", required=True)
    run_parser.add_argument("--timing-record", required=True)
    arguments = parser.parse_args(argv)
    if arguments.command == "synthetic-timing":
        def emit(record):
            print(json.dumps(record, sort_keys=True, separators=(",", ":")), flush=True)
        result = synthetic_timing_preflight(
            arguments.elapsed_repository_preflight_seconds, emit=emit
        )
        _atomic_write_json(arguments.output, result)
        return 0 if result["status"] == "pass" else 2
    result = run_authoritative_readout(
        input_dir=arguments.input_dir,
        output_dir=arguments.output_dir,
        repository_root=arguments.repository_root,
        expected_commit=arguments.expected_commit,
        timing_record_path=arguments.timing_record,
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
