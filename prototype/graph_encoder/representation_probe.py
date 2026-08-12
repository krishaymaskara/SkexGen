"""Read-only GE1 repaired-checkpoint representation probe.

This additive module implements accepted ADR-0011.  It never trains or
modifies a GE1 model.  The scientific entry point verifies and reproduces the
immutable repaired-sufficiency result before fitting disposable probes to a
target-free detached feature artifact.
"""

from __future__ import annotations

import argparse
import ast
import bisect
import copy
from dataclasses import dataclass
import hashlib
import itertools
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
except ImportError:  # Pure contract and statistical tests remain torch-free.
    torch = None

from prototype.model_data.geometry import GEOMETRY_CHANNEL_SCALES
from prototype.model_data.vocab import NODE_TYPES

from .autonomous import (
    AUTONOMOUS_EVALUATION_VERSION,
    P_TRUE,
    AutonomousConditionResult,
    AutonomousEvaluationResult,
    AutonomousPrediction,
    autonomous_input_from_paired,
)
from .batching import build_paired_batch
from .c7_v2 import C7_V2_SCALED_FAMILY_IDS_SHA256
from .decoder_contract import POSITIVE_OPERATION_MAGNITUDE_PARAMETERIZATION
from .errors import GraphEncoderError
from .metrics import score_condition
from .operation_fidelity import operation_geometry_fidelity_gate
from .optimization_diagnostic import validate_slurm_job_id
from .partitions import (
    AUTHORITATIVE_FILE_SHA256,
    load_train,
    select_c7_sufficiency_subsets,
    selected_family_ids_sha256,
)
from .pilot import (
    _atomic_write_bytes,
    _atomic_write_json,
    _atomic_write_jsonl,
    _canonical_json_text,
    _file_sha256,
    _is_within,
    _read_canonical_jsonl,
    _safe_relative_path,
    _source_identity,
    _verify_source_unchanged,
)
from .provenance import sha256_json, training_partition_identity
from .repaired_sufficiency import (
    REPAIRED_CHECKPOINT_EPOCH,
    REPAIRED_CHECKPOINT_ROLE,
    REPAIRED_GATE_VERSION,
    REPAIRED_PROTOCOL_VERSION,
    REPAIRED_SEED,
    verify_repaired_artifact,
)


PROBE_PROTOCOL_VERSION = "GE1-C7-REPAIRED-REPRESENTATION-PROBE-v1"
PROBE_ARTIFACT_VERSION = "GE1-C7-REPAIRED-REPRESENTATION-PROBE-ARTIFACT-v1"
PROBE_FEATURE_VERSION = "GE1-C7-REPAIRED-DETACHED-FEATURES-v1"
PROBE_LABEL_VERSION = "GE1-C7-REPAIRED-MAGNITUDE-LABELS-v1"
PROBE_RESULT_VERSION = "GE1-C7-REPAIRED-PROBE-RESULTS-v1"
SOURCE_REPAIRED_COMMIT = "705eb820f7a15d64fe650df7350b1553b2bc8172"
SOURCE_REPAIRED_JOB_ID = "3345280"
SOURCE_REPAIRED_ARTIFACT_PATH = (
    "/scratch/network/km6349/ge1_repaired_sufficiency_artifacts/"
    "ge1-repaired-705eb820f7a15d64fe650df7350b1553b2bc8172-3345280"
)
SOURCE_DIRECT_HASHES = {
    "artifact_manifest.json": (
        "63b495d245277bb6e3a181e122722def1cfd42ca6b05fe73cbaab2f5c90663e2"
    ),
    "metrics.jsonl": (
        "39bd071bc682034d582a7cfffa6a225dddb3e2cf1902ddf68890545df21b148f"
    ),
    "resolved_config.json": (
        "c54102bb4c801c8955b9495581732188a88317fd676457961d361e7c31afcd4f"
    ),
    "SHA256SUMS": (
        "04c97c3d795c8dd757ab5ee1bcff5421abe67d83d9dd674f9155f27de55edbe8"
    ),
}
SOURCE_CHECKPOINTS = {
    "flat": {
        "recovery": {
            "path": "checkpoints/scaled/flat/flat-seed2026-epoch0200.pt",
            "sha256": (
                "b61666a307c56a7b59a91671b2e0b06914bb1e76d4a0b48a60a0a501eb3cdb31"
            ),
        },
        "inference": {
            "path": "checkpoints/scaled/flat/inference-epoch-0200.pt",
            "sha256": (
                "f7c37677b8e707bfc35f1dcfd61ad13ec118b77528053a2553af39f6b2520b28"
            ),
        },
    },
    "typed_graph": {
        "recovery": {
            "path": (
                "checkpoints/scaled/typed_graph/"
                "typed_graph-seed2026-epoch0200.pt"
            ),
            "sha256": (
                "fffdb0ead00c05aa6d99f3ee0e471cccec63040e659130ae03f74bbc0dd73ec9"
            ),
        },
        "inference": {
            "path": (
                "checkpoints/scaled/typed_graph/inference-epoch-0200.pt"
            ),
            "sha256": (
                "8b3733df88a8bd73851a2e7979a503e7c18ef75acbbd4b626f524657aba58cb5"
            ),
        },
    },
}
PROBE_ARMS = ("flat", "typed_graph")
OPERATION_TYPES = ("extrude", "revolve")
FEATURE_LEVELS = ("C", "D")
PROBE_KINDS = ("linear", "mlp")
FEATURE_DIMENSIONS = {"A": 1, "B": 1, "C": 32, "D": 36}
LINEAR_PARAMETER_COUNTS = {"C": 165, "D": 185}
MLP_PARAMETER_COUNTS = {"C": 309, "D": 341}
REGULARIZATION_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
LINEAR_MAX_ITER = 500
LINEAR_HISTORY_SIZE = 100
LINEAR_TOLERANCE_GRAD = 1e-9
LINEAR_TOLERANCE_CHANGE = 1e-12
MLP_WIDTH = 8
MLP_L2 = 1e-2
MLP_LEARNING_RATE = 1e-2
MLP_UPDATES = 2000
PERMUTATION_COUNT = 100
ACCESS_BALANCED_MIN = 0.40
ACCESS_P_VALUE_MAX = 0.05
MATERIAL_MARGIN = 0.10
SCALED_FAMILY_COUNT = 32
SCALED_OPERATION_COUNT = 48
SCALED_EXTRUSION_COUNT = 32
SCALED_REVOLVE_COUNT = 16
OPERATION_GRIDS = {
    "extrude": (0.5, 1.0, 1.5, 2.0, 3.0),
    "revolve": (45.0, 90.0, 180.0, 270.0, 360.0),
}
NORMALIZED_GRIDS = {
    "extrude": (0.125, 0.25, 0.375, 0.5, 0.75),
    "revolve": (0.125, 0.25, 0.5, 0.75, 1.0),
}
FIDELITY_ERROR_LIMITS = {"extrude": 0.25, "revolve": 22.5}
PROTECTED_ACCESS_FIELDS = (
    "development_accessed",
    "systematic_rr_accessed",
    "test_er_accessed",
    "iid_accessed",
    "history_depth_accessed",
    "geometry_extrapolation_accessed",
    "other_corpus_accessed",
)
EXPECTED_ARTIFACT_FILES = (
    "detached_features.pt",
    "feature_manifest.json",
    "labels.json",
    "metrics.jsonl",
    "probe_results.json",
    "resolved_config.json",
    "scalar_analysis.json",
)
FEATURE_ROW_FIELDS = frozenset({
    "key",
    "label_key",
    "arm",
    "family_id",
    "operation_index",
    "operation_type",
    "A_normalized",
    "A_physical",
    "B_raw_logit",
    "C",
    "D",
})
FORBIDDEN_FEATURE_KEYS = frozenset({
    "target",
    "targets",
    "target_geometry",
    "target_magnitude",
    "label",
    "labels",
    "class_index",
    "cad_history",
    "model_state",
    "optimizer_state",
    "checkpoint",
})


@dataclass
class RepresentationProbeAccessTracker:
    operation_template_manifest_accessed: object = False
    operation_template_train_payload_accessed: object = False
    scaled_train_payload_accessed: object = False
    source_artifact_accessed: object = False
    source_checkpoint_accessed: object = False

    def declarations(self, *, completed=False):
        record = {
            "operation_template_manifest_accessed": (
                self.operation_template_manifest_accessed
            ),
            "operation_template_train_payload_accessed": (
                self.operation_template_train_payload_accessed
            ),
            "scaled_train_payload_accessed": self.scaled_train_payload_accessed,
            "source_repaired_artifact_accessed": self.source_artifact_accessed,
            "source_repaired_checkpoint_accessed": (
                self.source_checkpoint_accessed
            ),
            "source_repaired_result_changed": False,
            "source_checkpoints_modified": False,
            "ge1_training_performed": False,
            "ge1_backward_pass_performed": False,
            "ge1_optimizer_constructed": False,
            "ge1_checkpoint_written": False,
            "ge1_parameters_modified": False,
            "probe_training_performed": bool(completed),
            "probe_parameters_only": True,
            "model_or_decoder_repair_implemented_or_invoked": False,
            "cad_kernel_available": False,
            "cad_kernel_used": False,
            "stage6_performed": False,
            "c8_or_later_performed": False,
            "representation_probe_completed": bool(completed),
        }
        record.update({name: False for name in PROTECTED_ACCESS_FIELDS})
        return record


def probe_contract():
    """Return the frozen statistical and feature contract."""

    return {
        "protocol_version": PROBE_PROTOCOL_VERSION,
        "source": {
            "protocol_version": REPAIRED_PROTOCOL_VERSION,
            "commit": SOURCE_REPAIRED_COMMIT,
            "slurm_job_id": SOURCE_REPAIRED_JOB_ID,
            "artifact_path": SOURCE_REPAIRED_ARTIFACT_PATH,
            "direct_sha256": copy.deepcopy(SOURCE_DIRECT_HASHES),
            "checkpoints": copy.deepcopy(SOURCE_CHECKPOINTS),
        },
        "cohort": {
            "partition": "operation_template.train",
            "family_count": SCALED_FAMILY_COUNT,
            "family_ids_sha256": C7_V2_SCALED_FAMILY_IDS_SHA256,
            "template_counts": {"E": 8, "R": 8, "EE": 8, "RE": 8},
            "operation_count": SCALED_OPERATION_COUNT,
            "extrusion_count": SCALED_EXTRUSION_COUNT,
            "revolve_count": SCALED_REVOLVE_COUNT,
            "representation_variant": "continuous",
            "grouping_unit": "physical_family",
        },
        "features": {
            "A": "positive_mapped_active_normalized_operation_magnitude",
            "B": "active_remaining_geometry_head_logit",
            "C": "pre_head_shared_decoder_state",
            "D": (
                "flattened_2x16_continuous_memory_plus_predicted_type_and_"
                "canonical_slot_one_hot"
            ),
            "dimensions": dict(FEATURE_DIMENSIONS),
        },
        "linear": {
            "regularization_grid": list(REGULARIZATION_GRID),
            "max_iter": LINEAR_MAX_ITER,
            "history_size": LINEAR_HISTORY_SIZE,
            "tolerance_grad": LINEAR_TOLERANCE_GRAD,
            "tolerance_change": LINEAR_TOLERANCE_CHANGE,
            "line_search_fn": "strong_wolfe",
            "dtype": "float64",
            "initialization": "zeros",
            "parameter_counts": dict(LINEAR_PARAMETER_COUNTS),
        },
        "mlp": {
            "hidden_width": MLP_WIDTH,
            "activation": "tanh",
            "l2_weights_only": MLP_L2,
            "optimizer": "Adam",
            "learning_rate": MLP_LEARNING_RATE,
            "updates": MLP_UPDATES,
            "seed": REPAIRED_SEED,
            "parameter_counts": dict(MLP_PARAMETER_COUNTS),
        },
        "permutations": PERMUTATION_COUNT,
        "interpretation": {
            "balanced_accuracy_min": ACCESS_BALANCED_MIN,
            "p_value_max": ACCESS_P_VALUE_MAX,
            "raw_accuracy_must_exceed_null_p95": True,
            "material_margin": MATERIAL_MARGIN,
        },
    }


def _validate_label(label):
    if isinstance(label, bool) or not isinstance(label, int) or not 0 <= label < 5:
        raise GraphEncoderError(
            "invalid_probe_label", "magnitude class must be an integer in [0,4]"
        )
    return label


def classification_metrics(labels, predictions):
    """Return fixed-five-class metrics without external dependencies."""

    truth = tuple(_validate_label(item) for item in labels)
    predicted = tuple(_validate_label(item) for item in predictions)
    if not truth or len(truth) != len(predicted):
        raise GraphEncoderError(
            "invalid_probe_predictions", "prediction and label lengths differ"
        )
    confusion = [[0 for unused in range(5)] for unused in range(5)]
    for expected, observed in zip(truth, predicted):
        confusion[expected][observed] += 1
    support = [sum(row) for row in confusion]
    recall = [
        (float(confusion[index][index]) / count if count else 0.0)
        for index, count in enumerate(support)
    ]
    correct = sum(confusion[index][index] for index in range(5))
    return {
        "accuracy": float(correct) / len(truth),
        "balanced_accuracy": sum(recall) / 5.0,
        "confusion_matrix": confusion,
        "class_support": support,
        "per_class_recall": recall,
        "observation_count": len(truth),
    }


def nearest_grid_predictions(values, operation_type):
    """Map scalars to the closest grid point; smaller magnitude wins ties."""

    grid = OPERATION_GRIDS.get(operation_type)
    if grid is None:
        raise GraphEncoderError("invalid_operation_type", "unknown operation type")
    result = []
    for value in values:
        scalar = float(value)
        if not math.isfinite(scalar):
            raise GraphEncoderError("nonfinite_probe_feature", "scalar is nonfinite")
        unused_distance, unused_target, index = min(
            (abs(scalar - target), target, index)
            for index, target in enumerate(grid)
        )
        result.append(index)
    return tuple(result)


def _threshold_candidates(values):
    ordered = sorted(set(float(value) for value in values))
    if not ordered or any(not math.isfinite(value) for value in ordered):
        raise GraphEncoderError(
            "invalid_scalar_analysis", "threshold values must be finite"
        )
    return tuple(
        [-float("inf")]
        + [
            (left + right) / 2.0
            for left, right in zip(ordered[:-1], ordered[1:])
        ]
        + [float("inf")]
    )


def apply_monotonic_thresholds(values, thresholds):
    cuts = tuple(float(item) for item in thresholds)
    if len(cuts) != 4 or any(left > right for left, right in zip(cuts, cuts[1:])):
        raise GraphEncoderError(
            "invalid_scalar_thresholds", "four nondecreasing thresholds required"
        )
    return tuple(bisect.bisect_right(cuts, float(value)) for value in values)


def fit_monotonic_thresholds(values, labels):
    """Fit the exact five-class ordered ceiling by dynamic programming."""

    scalars = tuple(float(value) for value in values)
    truth = tuple(_validate_label(item) for item in labels)
    if not scalars or len(scalars) != len(truth):
        raise GraphEncoderError(
            "invalid_scalar_analysis", "scalar and label lengths differ"
        )
    candidates = _threshold_candidates(scalars)
    # A candidate threshold partitions observations with value <= threshold.
    prefix_matches = []
    for predicted_class in range(5):
        counts = []
        for threshold in candidates:
            counts.append(sum(
                int(label == predicted_class and value <= threshold)
                for value, label in zip(scalars, truth)
            ))
        prefix_matches.append(tuple(counts))

    # State after assigning classes 0..class_index: boundary tuple and score.
    states = {index: (prefix_matches[0][index], (index,))
              for index in range(len(candidates))}
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
    thresholds = tuple(candidates[index] for index in best[1])
    predictions = apply_monotonic_thresholds(scalars, thresholds)
    return {
        "algorithm": "dynamic_programming_with_empty_intervals",
        "tie_break": "lexicographically_smallest_threshold_tuple",
        "thresholds": list(thresholds),
        "metrics": classification_metrics(truth, predictions),
        "predictions": list(predictions),
    }


def grouped_threshold_predictions(rows, *, value_key):
    """Apply threshold fitting under leave-one-physical-family-out splits."""

    values = tuple(rows)
    families = tuple(sorted(set(item["family_id"] for item in values)))
    if len(families) < 2:
        raise GraphEncoderError(
            "insufficient_probe_families", "grouped thresholds need two families"
        )
    predictions = {}
    folds = []
    for held in families:
        train = tuple(item for item in values if item["family_id"] != held)
        test = tuple(item for item in values if item["family_id"] == held)
        fitted = fit_monotonic_thresholds(
            tuple(item[value_key] for item in train),
            tuple(item["class_index"] for item in train),
        )
        observed = apply_monotonic_thresholds(
            tuple(item[value_key] for item in test), fitted["thresholds"]
        )
        for row, prediction in zip(test, observed):
            predictions[row["key"]] = prediction
        folds.append({
            "held_out_family_id": held,
            "thresholds": fitted["thresholds"],
            "train_family_ids": sorted(set(item["family_id"] for item in train)),
        })
    ordered = tuple(sorted(values, key=lambda item: item["key"]))
    truth = tuple(item["class_index"] for item in ordered)
    predicted = tuple(predictions[item["key"]] for item in ordered)
    return {
        "split": "leave_one_physical_family_out",
        "metrics": classification_metrics(truth, predicted),
        "predictions_by_key": predictions,
        "folds": folds,
    }


def scalar_analysis(rows, operation_type):
    """Compute the frozen A/B scalar analyses for one arm and type."""

    selected = tuple(sorted(
        (item for item in rows if item["operation_type"] == operation_type),
        key=lambda item: item["key"],
    ))
    expected = (
        SCALED_EXTRUSION_COUNT if operation_type == "extrude"
        else SCALED_REVOLVE_COUNT
    )
    if len(selected) != expected:
        raise GraphEncoderError(
            "invalid_probe_operation_count", "operation count differs"
        )
    labels = tuple(item["class_index"] for item in selected)
    physical = tuple(item["A_physical"] for item in selected)
    logits = tuple(item["B_raw_logit"] for item in selected)
    fidelity_pass_by_key = {
        item["key"]: abs(
            item["A_physical"]
            - OPERATION_GRIDS[operation_type][item["class_index"]]
        ) < FIDELITY_ERROR_LIMITS[operation_type]
        for item in selected
    }
    family_pass = {}
    for item in selected:
        family_pass.setdefault(item["family_id"], True)
        family_pass[item["family_id"]] = (
            family_pass[item["family_id"]] and fidelity_pass_by_key[item["key"]]
        )
    a_ceiling = fit_monotonic_thresholds(physical, labels)
    b_ceiling = fit_monotonic_thresholds(logits, labels)
    if a_ceiling["metrics"] != b_ceiling["metrics"]:
        raise GraphEncoderError(
            "positive_mapping_monotonicity_failure",
            "A and B monotonic ceilings differ",
        )
    ranges = {}
    for name, values in (("A_physical", physical), ("B_raw_logit", logits)):
        ranges[name] = {}
        for label in range(5):
            observed = tuple(
                value for value, truth in zip(values, labels) if truth == label
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
            {
                "key": selected[index]["key"],
                "value": physical[index],
                "class_index": labels[index],
            }
            for index in sorted(
                range(len(selected)), key=lambda item: (physical[item], item)
            )
        ],
        "rank_order_B": [
            {
                "key": selected[index]["key"],
                "value": logits[index],
                "class_index": labels[index],
            }
            for index in sorted(
                range(len(selected)), key=lambda item: (logits[item], item)
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
        "A_scalar_only_resubstitution_ceiling": a_ceiling,
        "B_monotonic_resubstitution_ceiling": b_ceiling,
        "A_grouped_threshold": grouped_threshold_predictions(
            selected, value_key="A_physical"
        ),
        "B_grouped_threshold": grouped_threshold_predictions(
            selected, value_key="B_raw_logit"
        ),
    }


def geometry_loss_audit():
    """Return the accepted source-derived analytic loss accounting."""

    return {
        "audit_type": "read_only_source_derived_analytic_audit",
        "source_contract": validate_geometry_loss_source_contract(),
        "smooth_l1_beta": 1.0,
        "prediction_minimum": "torch.finfo(torch.float32).tiny",
        "prediction_maximum": 1.0,
        "normalized_target_grids": {
            name: list(values) for name, values in NORMALIZED_GRIDS.items()
        },
        "signed_normalized_error_envelopes": {
            "extrude": ["epsilon - 0.75", 0.875],
            "revolve": ["epsilon - 1.0", 0.875],
        },
        "strictly_linear_above_beta_region_reached": False,
        "template_coefficients": {
            "E": {"active_channels": 1, "extrude": "1", "total": "1"},
            "R": {"active_channels": 5, "revolve": "1/5", "total": "1/5"},
            "EE": {"active_channels": 2, "extrude_each": "1/2", "total": "1"},
            "RE": {
                "active_channels": 6,
                "revolve": "1/6",
                "extrude": "1/6",
                "total": "1/3",
            },
        },
        "cohort_average_coefficients": {
            "all_operation_channels": "19/30",
            "extrusion": "13/24",
            "revolve": "11/120",
            "axis": "11/30",
        },
        "per_operation_mean_coefficients": {
            "extrusion_over_32": "13/24",
            "revolve_over_16": "11/60",
            "all_48": "19/45",
        },
        "interpretation": (
            "masked-mean dilution is distinct from the equal-template cohort "
            "average and is not itself classified as a defect"
        ),
    }


def validate_geometry_loss_source_contract(source_text=None):
    """Prove the common loss leaves PyTorch's Smooth-L1 beta at 1.0."""

    if source_text is None:
        source_path = Path(__file__).resolve().parents[1] / "flat_baseline/losses.py"
        source_text = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source_text)
    function = next(
        (
            node for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_per_example_smooth_l1"
        ),
        None,
    )
    if function is None:
        raise GraphEncoderError(
            "geometry_loss_source_mismatch", "Smooth-L1 helper is absent"
        )
    calls = [
        node for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "smooth_l1_loss"
    ]
    if len(calls) != 1 or any(item.arg == "beta" for item in calls[0].keywords):
        raise GraphEncoderError(
            "geometry_loss_source_mismatch",
            "Smooth-L1 call count or beta override differs",
        )
    return {
        "path": "prototype/flat_baseline/losses.py",
        "function": "_per_example_smooth_l1",
        "common_loss_call_chain": [
            "prototype.graph_encoder.losses.common_ge1_loss",
            "prototype.graph_baseline.losses.graph_v1_loss",
            "prototype.flat_baseline.constrained_v4_losses.constrained_profile_v4_loss",
            "prototype.flat_baseline.losses._per_example_smooth_l1",
        ],
        "smooth_l1_call_count": 1,
        "beta_keyword_present": False,
        "pytorch_1_11_default_beta": 1.0,
    }


def _nearest_rank_95(values):
    ordered = sorted(float(item) for item in values)
    if not ordered:
        raise GraphEncoderError("empty_permutation_null", "null is empty")
    rank = int(math.ceil(0.95 * len(ordered)))
    return ordered[rank - 1]


def permutation_summary(observed, null_values):
    values = tuple(float(item) for item in null_values)
    if len(values) != PERMUTATION_COUNT:
        raise GraphEncoderError(
            "invalid_permutation_count", "exactly 100 permutations required"
        )
    mean = sum(values) / len(values)
    variance = sum((item - mean) ** 2 for item in values) / len(values)
    return {
        "count": len(values),
        "mean": mean,
        "population_standard_deviation": math.sqrt(variance),
        "nearest_rank_95th_percentile": _nearest_rank_95(values),
        "maximum": max(values),
        "empirical_one_sided_p_value": (
            1.0 + sum(item >= float(observed) for item in values)
        ) / 101.0,
    }


def _seed_from_parts(*parts):
    material = "|".join(str(item) for item in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def permute_family_label_blocks(rows, *, arm, operation_type, feature_level,
                                probe_kind, permutation_index):
    """Permute complete family label blocks within template and type."""

    if not 0 <= int(permutation_index) < PERMUTATION_COUNT:
        raise GraphEncoderError(
            "invalid_permutation_index", "permutation index is outside [0,99]"
        )
    selected = tuple(rows)
    by_template = {}
    for row in selected:
        if row["operation_type"] != operation_type:
            raise GraphEncoderError(
                "invalid_permutation_rows", "operation types differ"
            )
        by_template.setdefault(row["operation_template"], {}).setdefault(
            row["family_id"], []
        ).append(row)
    result = {}
    generator = random.Random(_seed_from_parts(
        PROBE_PROTOCOL_VERSION,
        REPAIRED_SEED,
        arm,
        operation_type,
        feature_level,
        probe_kind,
        permutation_index,
    ))
    for template in sorted(by_template):
        families = sorted(by_template[template])
        blocks = []
        for family in families:
            ordered = sorted(
                by_template[template][family], key=lambda item: item["operation_index"]
            )
            blocks.append(tuple(item["class_index"] for item in ordered))
        donor_indices = list(range(len(families)))
        generator.shuffle(donor_indices)
        for recipient_index, family in enumerate(families):
            recipient_rows = sorted(
                by_template[template][family], key=lambda item: item["operation_index"]
            )
            donor = blocks[donor_indices[recipient_index]]
            if len(recipient_rows) != len(donor):
                raise GraphEncoderError(
                    "invalid_permutation_block", "family block sizes differ"
                )
            for row, label in zip(recipient_rows, donor):
                result[row["key"]] = label
    if set(result) != {item["key"] for item in selected}:
        raise GraphEncoderError(
            "invalid_permutation_coverage", "permutation coverage differs"
        )
    return result


def _standardize(train_features, test_features, continuous_count):
    if torch is None:
        raise RuntimeError("probe fitting requires PyTorch")
    train = torch.as_tensor(train_features, dtype=torch.float64, device="cpu")
    test = torch.as_tensor(test_features, dtype=torch.float64, device="cpu")
    if train.dim() != 2 or test.dim() != 2 or train.size(1) != test.size(1):
        raise GraphEncoderError(
            "invalid_probe_features", "feature matrices are incompatible"
        )
    count = int(continuous_count)
    mean = train[:, :count].mean(dim=0)
    std = train[:, :count].std(dim=0, unbiased=False)
    safe = torch.where(std == 0.0, torch.ones_like(std), std)
    train_continuous = (train[:, :count] - mean) / safe
    test_continuous = (test[:, :count] - mean) / safe
    train_continuous[:, std == 0.0] = 0.0
    test_continuous[:, std == 0.0] = 0.0
    return (
        torch.cat((train_continuous, train[:, count:]), dim=1),
        torch.cat((test_continuous, test[:, count:]), dim=1),
        {"mean": mean.tolist(), "population_standard_deviation": std.tolist()},
    )


def _linear_fit_predict(train_x, train_y, test_x, regularization,
                        *, max_iter=LINEAR_MAX_ITER):
    if torch is None:
        raise RuntimeError("linear probe requires PyTorch")
    feature_count = int(train_x.size(1))
    weights = torch.zeros((5, feature_count), dtype=torch.float64, requires_grad=True)
    intercept = torch.zeros((5,), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        (weights, intercept),
        lr=1.0,
        max_iter=int(max_iter),
        history_size=LINEAR_HISTORY_SIZE,
        tolerance_grad=LINEAR_TOLERANCE_GRAD,
        tolerance_change=LINEAR_TOLERANCE_CHANGE,
        line_search_fn="strong_wolfe",
    )
    labels = torch.as_tensor(train_y, dtype=torch.long)

    def closure():
        optimizer.zero_grad(set_to_none=True)
        logits = train_x.mm(weights.t()) + intercept
        loss = torch.nn.functional.cross_entropy(logits, labels)
        loss = loss + 0.5 * float(regularization) * weights.square().sum()
        if not bool(torch.isfinite(loss).item()):
            raise GraphEncoderError("nonfinite_probe_loss", "linear loss is nonfinite")
        loss.backward()
        return loss

    final_loss = optimizer.step(closure)
    with torch.no_grad():
        predictions = (test_x.mm(weights.t()) + intercept).argmax(dim=1)
    if not bool(torch.isfinite(torch.as_tensor(final_loss)).item()):
        raise GraphEncoderError("nonfinite_probe_loss", "linear loss is nonfinite")
    return tuple(int(item) for item in predictions.tolist()), {
        "final_objective": float(final_loss),
        "parameter_count": weights.numel() + intercept.numel(),
    }


def _mlp_fit_predict(train_x, train_y, test_x, seed,
                     *, updates=MLP_UPDATES):
    if torch is None:
        raise RuntimeError("MLP probe requires PyTorch")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed) % (2 ** 63 - 1))
        model = torch.nn.Sequential(
            torch.nn.Linear(train_x.size(1), MLP_WIDTH, dtype=torch.float64),
            torch.nn.Tanh(),
            torch.nn.Linear(MLP_WIDTH, 5, dtype=torch.float64),
        )
    optimizer = torch.optim.Adam(model.parameters(), lr=MLP_LEARNING_RATE)
    labels = torch.as_tensor(train_y, dtype=torch.long)
    final = None
    for unused_step in range(int(updates)):
        optimizer.zero_grad(set_to_none=True)
        logits = model(train_x)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        weight_penalty = sum(
            parameter.square().sum()
            for name, parameter in model.named_parameters()
            if name.endswith("weight")
        )
        final = loss + MLP_L2 * weight_penalty
        if not bool(torch.isfinite(final).item()):
            raise GraphEncoderError("nonfinite_probe_loss", "MLP loss is nonfinite")
        final.backward()
        optimizer.step()
    with torch.no_grad():
        predictions = model(test_x).argmax(dim=1)
    return tuple(int(item) for item in predictions.tolist()), {
        "final_objective": float(final.detach().item()),
        "parameter_count": sum(item.numel() for item in model.parameters()),
        "updates": int(updates),
        "seed": int(seed),
    }


def _rows_for_probe(rows, labels_by_key, feature_level):
    selected = tuple(sorted(rows, key=lambda item: item["key"]))
    return tuple({
        "key": item["key"],
        "family_id": item["family_id"],
        "features": item[feature_level],
        "label": _validate_label(labels_by_key[item["key"]]),
    } for item in selected)


def _fit_one_split(train, test, *, feature_level, probe_kind,
                   regularization=None, seed=None, test_limits=None):
    continuous = 32
    train_x, test_x, preprocessing = _standardize(
        [item["features"] for item in train],
        [item["features"] for item in test],
        continuous,
    )
    train_y = [item["label"] for item in train]
    if probe_kind == "linear":
        predictions, fit = _linear_fit_predict(
            train_x,
            train_y,
            test_x,
            regularization,
            max_iter=(
                LINEAR_MAX_ITER if test_limits is None
                else test_limits.get("linear_max_iter", LINEAR_MAX_ITER)
            ),
        )
    elif probe_kind == "mlp":
        predictions, fit = _mlp_fit_predict(
            train_x,
            train_y,
            test_x,
            seed,
            updates=(
                MLP_UPDATES if test_limits is None
                else test_limits.get("mlp_updates", MLP_UPDATES)
            ),
        )
    else:
        raise GraphEncoderError("invalid_probe_kind", "unknown probe kind")
    expected_count = (
        LINEAR_PARAMETER_COUNTS[feature_level]
        if probe_kind == "linear" else MLP_PARAMETER_COUNTS[feature_level]
    )
    if fit["parameter_count"] != expected_count:
        raise GraphEncoderError(
            "probe_parameter_count_mismatch", "derived parameter count differs"
        )
    return predictions, {"preprocessing": preprocessing, "fit": fit}


def _choose_linear_regularization(rows, *, feature_level, test_limits=None):
    families = tuple(sorted(set(item["family_id"] for item in rows)))
    if len(families) < 2:
        raise GraphEncoderError(
            "insufficient_probe_families", "inner LOFO needs two families"
        )
    grid = REGULARIZATION_GRID
    if test_limits is not None:
        grid = tuple(test_limits.get("regularization_grid", grid))
    candidates = []
    for regularization in grid:
        truth = []
        predicted = []
        for held in families:
            train = tuple(item for item in rows if item["family_id"] != held)
            test = tuple(item for item in rows if item["family_id"] == held)
            observed, unused_evidence = _fit_one_split(
                train,
                test,
                feature_level=feature_level,
                probe_kind="linear",
                regularization=regularization,
                test_limits=test_limits,
            )
            truth.extend(item["label"] for item in test)
            predicted.extend(observed)
        metrics = classification_metrics(truth, predicted)
        candidates.append((regularization, metrics))
    selected = max(
        candidates,
        key=lambda item: (
            item[1]["balanced_accuracy"], item[1]["accuracy"], item[0]
        ),
    )
    return selected[0], [
        {"lambda": value, "metrics": metrics} for value, metrics in candidates
    ]


def fit_probe_pipeline(rows, labels_by_key, *, arm, operation_type,
                       feature_level, probe_kind, test_limits=None):
    """Fit resubstitution and nested-family-LOFO disposable probes."""

    prepared = _rows_for_probe(rows, labels_by_key, feature_level)
    families = tuple(sorted(set(item["family_id"] for item in prepared)))
    if len(families) < 3:
        raise GraphEncoderError(
            "insufficient_probe_families", "nested LOFO requires three families"
        )
    resub_lambda = None
    resub_selection = None
    if probe_kind == "linear":
        resub_lambda, resub_selection = _choose_linear_regularization(
            prepared, feature_level=feature_level, test_limits=test_limits
        )
    resub_seed = _seed_from_parts(
        PROBE_PROTOCOL_VERSION, REPAIRED_SEED, arm, operation_type,
        feature_level, "__full__"
    )
    resub_predictions, resub_fit = _fit_one_split(
        prepared,
        prepared,
        feature_level=feature_level,
        probe_kind=probe_kind,
        regularization=resub_lambda,
        seed=resub_seed,
        test_limits=test_limits,
    )
    lofo_predictions = {}
    folds = []
    for held in families:
        train = tuple(item for item in prepared if item["family_id"] != held)
        test = tuple(item for item in prepared if item["family_id"] == held)
        selected_lambda = None
        selection = None
        if probe_kind == "linear":
            selected_lambda, selection = _choose_linear_regularization(
                train, feature_level=feature_level, test_limits=test_limits
            )
        seed = _seed_from_parts(
            PROBE_PROTOCOL_VERSION, REPAIRED_SEED, arm, operation_type,
            feature_level, held
        )
        observed, fit = _fit_one_split(
            train,
            test,
            feature_level=feature_level,
            probe_kind=probe_kind,
            regularization=selected_lambda,
            seed=seed,
            test_limits=test_limits,
        )
        for item, value in zip(test, observed):
            lofo_predictions[item["key"]] = value
        folds.append({
            "held_out_family_id": held,
            "selected_lambda": selected_lambda,
            "inner_selection": selection,
            "seed": seed if probe_kind == "mlp" else None,
            "fit": fit,
        })
    labels = tuple(item["label"] for item in prepared)
    lofo_ordered = tuple(lofo_predictions[item["key"]] for item in prepared)
    return {
        "arm": arm,
        "operation_type": operation_type,
        "feature_level": feature_level,
        "probe_kind": probe_kind,
        "derived_parameter_count": resub_fit["fit"]["parameter_count"],
        "optimization_contract": (
            {
                "dtype": "float64",
                "device": "cpu",
                "initialization": "zero_weights_and_intercept",
                "objective": "mean_cross_entropy_plus_0.5_lambda_weight_l2",
                "intercept_penalized": False,
                "optimizer": "LBFGS",
                "max_iter": LINEAR_MAX_ITER,
                "history_size": LINEAR_HISTORY_SIZE,
                "tolerance_grad": LINEAR_TOLERANCE_GRAD,
                "tolerance_change": LINEAR_TOLERANCE_CHANGE,
                "line_search_fn": "strong_wolfe",
            }
            if probe_kind == "linear" else {
                "dtype": "float64",
                "device": "cpu",
                "hidden_width": MLP_WIDTH,
                "activation": "tanh",
                "dropout": False,
                "normalization": False,
                "objective": "mean_cross_entropy_plus_0.01_weight_matrix_l2",
                "bias_penalized": False,
                "optimizer": "Adam",
                "learning_rate": MLP_LEARNING_RATE,
                "updates": MLP_UPDATES,
            }
        ),
        "resubstitution": {
            "selected_lambda": resub_lambda,
            "regularization_selection": resub_selection,
            "seed": resub_seed if probe_kind == "mlp" else None,
            "metrics": classification_metrics(labels, resub_predictions),
            "predictions_by_key": {
                item["key"]: value
                for item, value in zip(prepared, resub_predictions)
            },
        },
        "lofo": {
            "metrics": classification_metrics(labels, lofo_ordered),
            "predictions_by_key": lofo_predictions,
            "folds": folds,
        },
    }


def run_permutation_controls(rows, true_labels, *, arm, operation_type,
                             feature_level, probe_kind, observed_result,
                             test_limits=None):
    """Rerun the complete pipeline under 100 deterministic block permutations."""

    if (
        not {item["key"] for item in rows}.issubset(set(true_labels))
        or any(
            _validate_label(true_labels[item["key"]]) != item["class_index"]
            for item in rows
        )
    ):
        raise GraphEncoderError(
            "probe_label_alignment_failure", "true-label coverage differs"
        )
    null_raw = []
    null_balanced = []
    expected_count = PERMUTATION_COUNT
    if test_limits is not None:
        expected_count = int(test_limits.get("permutation_count", expected_count))
    if expected_count != PERMUTATION_COUNT and not test_limits:
        raise AssertionError("scientific permutation count cannot change")
    for index in range(expected_count):
        labels = permute_family_label_blocks(
            rows,
            arm=arm,
            operation_type=operation_type,
            feature_level=feature_level,
            probe_kind=probe_kind,
            permutation_index=index,
        )
        result = fit_probe_pipeline(
            rows,
            labels,
            arm=arm,
            operation_type=operation_type,
            feature_level=feature_level,
            probe_kind=probe_kind,
            test_limits=test_limits,
        )
        null_raw.append(result["lofo"]["metrics"]["accuracy"])
        null_balanced.append(result["lofo"]["metrics"]["balanced_accuracy"])
    if expected_count != PERMUTATION_COUNT:
        return {
            "test_only_reduced_permutation_count": expected_count,
            "raw_values": null_raw,
            "balanced_values": null_balanced,
        }
    return {
        "raw_accuracy": permutation_summary(
            observed_result["lofo"]["metrics"]["accuracy"], null_raw
        ),
        "balanced_accuracy": permutation_summary(
            observed_result["lofo"]["metrics"]["balanced_accuracy"],
            null_balanced,
        ),
        "seed_stream": {
            "derivation": "sha256_first_8_bytes_big_endian",
            "parts": [
                PROBE_PROTOCOL_VERSION,
                REPAIRED_SEED,
                arm,
                operation_type,
                feature_level,
                probe_kind,
                "permutation_index_0_through_99",
            ],
            "family_blocks_permuted_within_template_and_operation_type": True,
        },
    }


def accessibility_decision(probe_result, scalar_grouped_metrics):
    """Apply the frozen prospective interpretation thresholds."""

    metrics = probe_result["lofo"]["metrics"]
    permutation = probe_result["label_permutation"]
    raw_null = permutation["raw_accuracy"]
    balanced_null = permutation["balanced_accuracy"]
    accessible = (
        metrics["balanced_accuracy"] >= ACCESS_BALANCED_MIN
        and balanced_null["empirical_one_sided_p_value"] <= ACCESS_P_VALUE_MAX
        and metrics["accuracy"] > raw_null["nearest_rank_95th_percentile"]
    )
    material = (
        accessible
        and metrics["accuracy"]
        >= scalar_grouped_metrics["accuracy"] + MATERIAL_MARGIN
        and metrics["balanced_accuracy"]
        >= scalar_grouped_metrics["balanced_accuracy"] + MATERIAL_MARGIN
    )
    return {
        "held_out_family_accessibility_evidence": accessible,
        "criteria": {
            "balanced_accuracy_at_least_0_40": (
                metrics["balanced_accuracy"] >= ACCESS_BALANCED_MIN
            ),
            "permutation_p_value_at_most_0_05": (
                balanced_null["empirical_one_sided_p_value"]
                <= ACCESS_P_VALUE_MAX
            ),
            "raw_accuracy_above_permutation_p95": (
                metrics["accuracy"] > raw_null["nearest_rank_95th_percentile"]
            ),
        },
        "materially_exceeds_scalar_grouped_threshold": material,
        "raw_accuracy_difference_from_scalar": (
            metrics["accuracy"] - scalar_grouped_metrics["accuracy"]
        ),
        "balanced_accuracy_difference_from_scalar": (
            metrics["balanced_accuracy"]
            - scalar_grouped_metrics["balanced_accuracy"]
        ),
    }


def combined_prediction_metrics(rows, labels_by_key, extrusion_predictions,
                                revolve_predictions):
    """Combine independently fitted type-specific predictions conservatively."""

    combined = dict(extrusion_predictions)
    if set(combined) & set(revolve_predictions):
        raise GraphEncoderError(
            "probe_prediction_overlap", "operation-type predictions overlap"
        )
    combined.update(revolve_predictions)
    expected = {item["key"] for item in rows}
    if set(combined) != expected:
        raise GraphEncoderError(
            "incomplete_probe_predictions", "combined prediction coverage differs"
        )
    ordered = tuple(sorted(rows, key=lambda item: item["key"]))
    operation_metrics = classification_metrics(
        [labels_by_key[item["key"]] for item in ordered],
        [combined[item["key"]] for item in ordered],
    )
    by_family = {}
    for item in ordered:
        by_family.setdefault(item["family_id"], []).append(
            combined[item["key"]] == labels_by_key[item["key"]]
        )
    return {
        "exact_operation_accuracy": operation_metrics["accuracy"],
        "physical_family_all_operations_correct_accuracy": (
            sum(all(values) for values in by_family) / len(by_family)
        ),
        "operation_count": len(ordered),
        "physical_family_count": len(by_family),
    }


def _state_sha256(model):
    if torch is None:
        raise RuntimeError("model hashing requires PyTorch")
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(tensor.dtype).encode("ascii") + b"\0")
        digest.update(json.dumps(list(tensor.shape)).encode("ascii") + b"\0")
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _assert_model_states_equal(left, right):
    left_state = left.state_dict()
    right_state = right.state_dict()
    if tuple(left_state) != tuple(right_state) or any(
        not torch.equal(left_state[name], right_state[name]) for name in left_state
    ):
        raise GraphEncoderError(
            "source_checkpoint_state_mismatch",
            "strict recovery and inference model states differ",
        )


def _validate_source_hashes(source_artifact):
    root = Path(source_artifact)
    observed = {}
    for relative, expected in SOURCE_DIRECT_HASHES.items():
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise GraphEncoderError(
                "source_repaired_artifact_mismatch",
                "required source file is not a regular file",
            )
        observed[relative] = _file_sha256(path)
        if observed[relative] != expected:
            raise GraphEncoderError(
                "source_repaired_artifact_mismatch",
                "direct source SHA-256 differs for {}".format(relative),
            )
    checkpoint_hashes = {}
    for arm in PROBE_ARMS:
        checkpoint_hashes[arm] = {}
        for role in ("recovery", "inference"):
            contract = SOURCE_CHECKPOINTS[arm][role]
            path = root / contract["path"]
            if not path.is_file() or path.is_symlink():
                raise GraphEncoderError(
                    "source_repaired_checkpoint_mismatch",
                    "source checkpoint is not a regular file",
                )
            observed_hash = _file_sha256(path)
            if observed_hash != contract["sha256"]:
                raise GraphEncoderError(
                    "source_repaired_checkpoint_mismatch",
                    "{} {} checkpoint SHA-256 differs".format(arm, role),
                )
            checkpoint_hashes[arm][role] = observed_hash
    return {"direct": observed, "checkpoints": checkpoint_hashes}


def _source_repaired_evidence(source_artifact):
    """Verify the complete immutable source and obtain scaled P_true records."""

    root = Path(source_artifact)
    verification = verify_repaired_artifact(
        root,
        expected_commit=SOURCE_REPAIRED_COMMIT,
        expected_slurm_job_id=SOURCE_REPAIRED_JOB_ID,
    )
    hashes = _validate_source_hashes(root)
    events = _read_canonical_jsonl(root / "metrics.jsonl")
    original_metrics = {}
    original_fidelity = {}
    checkpoint_events = {}
    for event in events:
        if (
            event.get("event") == "repaired_sufficiency_autonomous_condition_metrics"
            and event.get("subset_identity") == "scaled"
            and event.get("condition") == P_TRUE
        ):
            original_metrics[event["arm"]] = event["metrics"]
        if (
            event.get("event") == "repaired_sufficiency_gate_result"
            and event.get("subset_identity") == "scaled"
            and event.get("gate_name") == "scaled_operation_geometry_fidelity"
        ):
            original_fidelity[event["arm"]] = event
        if (
            event.get("event") == "repaired_sufficiency_checkpoints_reloaded"
            and event.get("subset_identity") == "scaled"
        ):
            checkpoint_events[event["arm"]] = event
    if (
        set(original_metrics) != set(PROBE_ARMS)
        or set(original_fidelity) != set(PROBE_ARMS)
        or set(checkpoint_events) != set(PROBE_ARMS)
    ):
        raise GraphEncoderError(
            "source_repaired_artifact_mismatch",
            "scaled source evidence is incomplete",
        )
    for arm in PROBE_ARMS:
        event = checkpoint_events[arm]
        expected = SOURCE_CHECKPOINTS[arm]
        if (
            event.get("epoch") != REPAIRED_CHECKPOINT_EPOCH
            or event.get("strict_recovery_reload") is not True
            or event.get("strict_inference_reload") is not True
            or event.get("checkpoint_identities", {}).get("recovery")
            != expected["recovery"]["path"]
            or event.get("checkpoint_identities", {}).get("inference")
            != expected["inference"]["path"]
            or event.get("checkpoint_sha256", {}).get("recovery")
            != expected["recovery"]["sha256"]
            or event.get("checkpoint_sha256", {}).get("inference")
            != expected["inference"]["sha256"]
        ):
            raise GraphEncoderError(
                "source_repaired_artifact_mismatch",
                "source checkpoint reload evidence differs",
            )
    resolved = json.loads((root / "resolved_config.json").read_text("utf-8"))
    scaled_ids = tuple(resolved.get("subset_selection", {}).get("scaled_family_ids", ()))
    if (
        len(scaled_ids) != SCALED_FAMILY_COUNT
        or tuple(sorted(scaled_ids)) != scaled_ids
        or selected_family_ids_sha256(scaled_ids)
        != C7_V2_SCALED_FAMILY_IDS_SHA256
    ):
        raise GraphEncoderError(
            "source_repaired_cohort_mismatch", "source scaled cohort differs"
        )
    return {
        "verification": verification,
        "hashes": hashes,
        "original_metrics": original_metrics,
        "original_fidelity": original_fidelity,
        "checkpoint_events": checkpoint_events,
        "scaled_family_ids": scaled_ids,
        "resolved_config": resolved,
    }


def _load_recovery_model_read_only(path, *, arm, expected_sha256,
                                   partition_identity):
    """Strictly load repaired training state without constructing an optimizer."""

    if torch is None:
        raise RuntimeError("checkpoint loading requires PyTorch")
    from .config import (
        CHECKPOINT_SCHEMA,
        GE1TrainingConfig,
        MODEL_FAMILY,
        TRAINING_OPTIMIZER,
        frozen_encoder_config,
    )
    from .model import build_ge1_model
    from .training import TRAINING_CHECKPOINT_FIELDS, TRAINING_STATE_SCHEMA

    checkpoint = Path(path)
    if _file_sha256(checkpoint) != expected_sha256:
        raise GraphEncoderError(
            "source_repaired_checkpoint_mismatch", "recovery hash differs"
        )
    payload = torch.load(str(checkpoint), map_location="cpu")
    if not isinstance(payload, dict) or set(payload) != set(TRAINING_CHECKPOINT_FIELDS):
        raise GraphEncoderError(
            "invalid_probe_recovery_checkpoint", "recovery field set differs"
        )
    config = frozen_encoder_config(
        arm,
        REPAIRED_SEED,
        operation_magnitude_parameterization=(
            POSITIVE_OPERATION_MAGNITUDE_PARAMETERIZATION
        ),
    )
    training = GE1TrainingConfig()
    training.validate()
    expected = {
        "training_state_schema": TRAINING_STATE_SCHEMA,
        "checkpoint_schema": CHECKPOINT_SCHEMA,
        "model_family": MODEL_FAMILY,
        "arm_identity": config.arm_identity,
        "encoder_type": arm,
        "model_config": config.to_dict(),
        "model_config_sha256": sha256_json(config.to_dict()),
        "training_config": training.to_dict(),
        "training_config_sha256": sha256_json(training.to_dict()),
        "optimizer_name": TRAINING_OPTIMIZER,
        "completed_epoch": REPAIRED_CHECKPOINT_EPOCH,
        "optimizer_step_count": 800,
        "training_example_presentations": 6400,
        "seed": REPAIRED_SEED,
        "partition_identity": partition_identity,
        "partition_identity_sha256": sha256_json(partition_identity),
        "run_identity": "ge1-{}-seed{}".format(arm, REPAIRED_SEED),
        "selected_checkpoint_epoch": REPAIRED_CHECKPOINT_EPOCH,
        "selected_experimental_checkpoint": True,
    }
    for name, value in expected.items():
        if payload.get(name) != value:
            raise GraphEncoderError(
                "invalid_probe_recovery_checkpoint",
                "recovery metadata field {} differs".format(name),
            )
    provenance = payload.get("provenance", {})
    if (
        provenance.get("git_commit") != SOURCE_REPAIRED_COMMIT
        or provenance.get("slurm_job_id") != SOURCE_REPAIRED_JOB_ID
        or provenance.get("encoder_arm") != arm
        or provenance.get("operation_magnitude_parameterization")
        != POSITIVE_OPERATION_MAGNITUDE_PARAMETERIZATION
        or provenance.get("python_version") != "3.8.13"
        or str(provenance.get("pytorch_version", "")).split("+")[0] != "1.11.0"
        or provenance.get("device") != "cpu"
    ):
        raise GraphEncoderError(
            "invalid_probe_recovery_checkpoint", "recovery provenance differs"
        )
    if not isinstance(payload.get("optimizer_state"), dict):
        raise GraphEncoderError(
            "invalid_probe_recovery_checkpoint", "optimizer evidence is malformed"
        )
    model = build_ge1_model(config)
    try:
        model.load_state_dict(payload["model_state"], strict=True)
    except Exception as exc:
        raise GraphEncoderError(
            "invalid_probe_recovery_checkpoint", "strict recovery load failed"
        ) from exc
    model.requires_grad_(False)
    model.eval()
    return model, payload


def _load_inference_model_read_only(path, *, arm, expected_sha256):
    if torch is None:
        raise RuntimeError("checkpoint loading requires PyTorch")
    from .checkpoint import load_ge1_checkpoint

    if _file_sha256(path) != expected_sha256:
        raise GraphEncoderError(
            "source_repaired_checkpoint_mismatch", "inference hash differs"
        )
    model, payload = load_ge1_checkpoint(
        path,
        expected_code_revision=SOURCE_REPAIRED_COMMIT,
        expected_operation_magnitude_parameterization=(
            POSITIVE_OPERATION_MAGNITUDE_PARAMETERIZATION
        ),
    )
    if payload.get("encoder_type") != arm:
        raise GraphEncoderError(
            "invalid_probe_inference_checkpoint", "inference arm differs"
        )
    model.requires_grad_(False)
    model.eval()
    return model, payload


def _operation_type_one_hot(operation_type):
    if operation_type == "extrude":
        return (1.0, 0.0)
    if operation_type == "revolve":
        return (0.0, 1.0)
    raise GraphEncoderError(
        "invalid_autonomous_operation_type", "predicted node is not an operation"
    )


def _slot_one_hot(operation_index):
    if operation_index == 0:
        return (1.0, 0.0)
    if operation_index == 1:
        return (0.0, 1.0)
    raise GraphEncoderError(
        "invalid_autonomous_operation_slot", "controlled histories have two slots"
    )


def extract_target_free_features(model, input_batches, *, arm):
    """Run P_true and detach A-D without accepting any target argument."""

    if torch is None:
        raise RuntimeError("feature extraction requires PyTorch")
    if arm not in PROBE_ARMS or model.config.encoder != arm:
        raise GraphEncoderError("invalid_probe_arm", "model arm differs")
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise GraphEncoderError(
            "ge1_gradient_path_enabled",
            "every GE1 parameter must have requires_grad disabled",
        )
    batches = tuple(input_batches)
    all_ids = tuple(item for batch in batches for item in batch.family_ids)
    if all_ids != tuple(sorted(all_ids)) or len(all_ids) != len(set(all_ids)):
        raise GraphEncoderError(
            "invalid_autonomous_order", "batches must be unique and family-sorted"
        )
    rows = []
    predictions = []
    assignments = []
    membership = []
    encoding_seconds = 0.0
    decode_seconds = 0.0
    model.eval()
    with torch.no_grad():
        for batch in batches:
            start = time.perf_counter()
            encoded = model.encode(batch.encoder_input)
            encoding_seconds += time.perf_counter() - start
            if (
                tuple(encoded.memory.shape[1:]) != (2, 16)
                or encoded.memory.size(0) != len(batch.family_ids)
            ):
                raise GraphEncoderError(
                    "probe_memory_shape_mismatch", "encoder memory must be [B,2,16]"
                )
            membership.append((batch.batch_identity, batch.family_ids))
            for index, family_id in enumerate(batch.family_ids):
                memory = encoded.memory[index:index + 1].detach().clone()
                decode_start = time.perf_counter()
                output = model.decoder(
                    memory,
                    node_counts=torch.tensor(
                        (int(batch.node_counts[index]),), dtype=torch.long
                    ),
                    node_count_source="target_free_input_node_count",
                )
                decode_seconds += time.perf_counter() - decode_start
                raw = output.raw_prediction[0]
                constrained = output.constrained_prediction[0]
                converted = output.converted_prediction[0]
                prediction = AutonomousPrediction(
                    family_id,
                    P_TRUE,
                    family_id,
                    batch.batch_identity,
                    raw,
                    constrained,
                    converted,
                )
                predictions.append(prediction)
                assignments.append((family_id, family_id))
                states = raw.prefix_output.decoded_states.detach().cpu()
                logits = model.decoder.raw_remaining_geometry(
                    raw.prefix_output.decoded_states
                ).detach().cpu()
                operation_nodes = tuple(
                    constrained.node_prediction.predicted_operation_node_indices
                )
                if len(operation_nodes) not in (1, 2):
                    raise GraphEncoderError(
                        "invalid_autonomous_operation_count",
                        "controlled prediction must contain one or two operations",
                    )
                memory_values = tuple(float(item) for item in memory.cpu().reshape(-1))
                if len(memory_values) != 32:
                    raise GraphEncoderError(
                        "probe_memory_shape_mismatch", "flattened memory must have 32 values"
                    )
                for operation_index, node_index in enumerate(operation_nodes):
                    node = constrained.node_prediction.raw_nodes[node_index]
                    operation_type = NODE_TYPES.tokens[node.node_type_id]
                    type_one_hot = _operation_type_one_hot(operation_type)
                    slot_one_hot = _slot_one_hot(operation_index)
                    channel = 37 if operation_type == "extrude" else 38
                    other_channel = 38 if channel == 37 else 37
                    if (
                        not bool(node.derived_geometry_mask[channel])
                        or bool(node.derived_geometry_mask[other_channel])
                    ):
                        raise GraphEncoderError(
                            "probe_active_channel_mismatch",
                            "autonomous predicted type does not select one active channel",
                        )
                    compact_channel = channel - 33
                    normalized = float(node.normalized_geometry[channel])
                    physical = normalized * float(GEOMETRY_CHANNEL_SCALES[channel])
                    hidden = tuple(float(item) for item in states[0, node_index])
                    d_feature = memory_values + type_one_hot + slot_one_hot
                    if len(hidden) != 32 or len(d_feature) != 36:
                        raise GraphEncoderError(
                            "probe_feature_shape_mismatch", "C or D dimension differs"
                        )
                    rows.append({
                        "key": "{}:{}:{}:{}".format(
                            arm, family_id, operation_index, operation_type
                        ),
                        "label_key": "{}:{}:{}".format(
                            family_id, operation_index, operation_type
                        ),
                        "arm": arm,
                        "family_id": family_id,
                        "operation_index": operation_index,
                        "operation_type": operation_type,
                        "A_normalized": normalized,
                        "A_physical": physical,
                        "B_raw_logit": float(
                            logits[0, node_index, compact_channel].item()
                        ),
                        "C": hidden,
                        "D": d_feature,
                    })
    condition = AutonomousConditionResult(
        AUTONOMOUS_EVALUATION_VERSION,
        P_TRUE,
        True,
        None,
        tuple(predictions),
        tuple(assignments),
        tuple(membership),
        False,
        False,
        (),
        0.0,
        decode_seconds,
    )
    autonomous = AutonomousEvaluationResult(
        AUTONOMOUS_EVALUATION_VERSION,
        all_ids,
        encoding_seconds,
        (condition,),
    )
    return tuple(rows), autonomous


def _stable_condition_metrics(metrics):
    stable = copy.deepcopy(metrics)
    stable.pop("metric_computation_seconds", None)
    return stable


def _stable_fidelity_gate(gate):
    stable = copy.deepcopy(gate)
    # Source and reproduction differ only in diagnostic checkpoint path/access.
    for name in (
        "event", "version", "gate_version", "checkpoint_identity", "checkpoint_role",
        "access", "operation_magnitude_parameterization",
    ):
        stable.pop(name, None)
    return stable


def reproduce_source_evidence(*, arm, autonomous, examples, original_metrics,
                              original_fidelity):
    """Require exact stable source scoring before any feature is accepted."""

    if len(autonomous.conditions) != 1 or autonomous.conditions[0].condition != P_TRUE:
        raise GraphEncoderError(
            "probe_reproduction_failure", "only one P_true condition is eligible"
        )
    targets = {item.physical_family_id: item.target for item in examples}
    reproduced_metrics = score_condition(autonomous.conditions[0], targets)
    if _stable_condition_metrics(reproduced_metrics) != _stable_condition_metrics(
        original_metrics
    ):
        raise GraphEncoderError(
            "probe_reproduction_failure", "stable P_true source metrics differ"
        )
    fidelity = operation_geometry_fidelity_gate(
        arm=arm,
        subset_identity="scaled",
        condition_result=autonomous.conditions[0],
        examples_by_family={item.physical_family_id: item for item in examples},
        checkpoint_identity=SOURCE_CHECKPOINTS[arm]["inference"]["path"],
        epoch=REPAIRED_CHECKPOINT_EPOCH,
        access_flags={},
        protocol_version=REPAIRED_PROTOCOL_VERSION,
        gate_version=REPAIRED_GATE_VERSION,
        checkpoint_role=REPAIRED_CHECKPOINT_ROLE,
        operation_magnitude_parameterization=(
            POSITIVE_OPERATION_MAGNITUDE_PARAMETERIZATION
        ),
    )
    if _stable_fidelity_gate(fidelity) != _stable_fidelity_gate(original_fidelity):
        raise GraphEncoderError(
            "probe_reproduction_failure", "source operation fidelity differs"
        )
    family_values = reproduced_metrics.get("family_values", {})
    for family_id, values in family_values.items():
        if any(values.get(name) != 1.0 for name in (
            "exact_node_sequence", "exact_graph", "depends_on_exactness",
            "strict_conversion", "complete_executable_validity",
        )):
            raise GraphEncoderError(
                "probe_reproduction_failure",
                "source structural or validity result differs for {}".format(family_id),
            )
    return {
        "status": "exact_stable_source_reproduction_passed",
        "arm": arm,
        "stable_condition_metrics_sha256": hashlib.sha256(
            _canonical_json_text(_stable_condition_metrics(reproduced_metrics)).encode(
                "utf-8"
            )
        ).hexdigest(),
        "stable_fidelity_gate_sha256": hashlib.sha256(
            _canonical_json_text(_stable_fidelity_gate(fidelity)).encode("utf-8")
        ).hexdigest(),
        "family_count": len(family_values),
    }


def validate_feature_rows(rows):
    values = tuple(rows)
    def reject_forbidden(value):
        if isinstance(value, dict):
            forbidden = set(value) & FORBIDDEN_FEATURE_KEYS
            if forbidden:
                raise GraphEncoderError(
                    "target_leakage_in_features",
                    "target or label key is present: {}".format(sorted(forbidden)),
                )
            for nested in value.values():
                reject_forbidden(nested)
        elif isinstance(value, (list, tuple)):
            for nested in value:
                reject_forbidden(nested)

    reject_forbidden(values)
    if len(values) != len(PROBE_ARMS) * SCALED_OPERATION_COUNT:
        raise GraphEncoderError(
            "invalid_detached_features", "feature operation count differs"
        )
    if len({item.get("key") for item in values}) != len(values):
        raise GraphEncoderError(
            "invalid_detached_features", "feature keys are not unique"
        )
    for row in values:
        operation_type = row.get("operation_type") if isinstance(row, dict) else None
        operation_index = row.get("operation_index") if isinstance(row, dict) else None
        expected_context = None
        if operation_type in OPERATION_TYPES and operation_index in (0, 1):
            expected_context = (
                _operation_type_one_hot(operation_type)
                + _slot_one_hot(operation_index)
            )
        if (
            set(row) != FEATURE_ROW_FIELDS
            or row.get("arm") not in PROBE_ARMS
            or operation_type not in OPERATION_TYPES
            or isinstance(operation_index, bool)
            or operation_index not in (0, 1)
            or len(row.get("C", ())) != 32
            or len(row.get("D", ())) != 36
            or any(
                not math.isfinite(float(value))
                for name in ("A_normalized", "A_physical", "B_raw_logit")
                for value in (row.get(name),)
            )
            or any(not math.isfinite(float(value)) for value in row["C"] + row["D"])
            or tuple(float(value) for value in row["D"][-4:]) != expected_context
            or not 0.0 < float(row["A_normalized"]) <= 1.0
            or float(row["A_physical"]) != (
                float(row["A_normalized"])
                * float(GEOMETRY_CHANNEL_SCALES[
                    37 if operation_type == "extrude" else 38
                ])
            )
            or row.get("key") != "{}:{}:{}:{}".format(
                row.get("arm"), row.get("family_id"), operation_index,
                operation_type,
            )
            or row.get("label_key") != "{}:{}:{}".format(
                row.get("family_id"), operation_index, operation_type
            )
        ):
            raise GraphEncoderError(
                "invalid_detached_features", "feature row is malformed"
            )
    for arm in PROBE_ARMS:
        selected = tuple(item for item in values if item["arm"] == arm)
        if (
            len(selected) != SCALED_OPERATION_COUNT
            or sum(item["operation_type"] == "extrude" for item in selected)
            != SCALED_EXTRUSION_COUNT
            or sum(item["operation_type"] == "revolve" for item in selected)
            != SCALED_REVOLVE_COUNT
            or len({item["family_id"] for item in selected}) != SCALED_FAMILY_COUNT
        ):
            raise GraphEncoderError(
                "invalid_detached_features", "arm coverage differs"
            )
    return values


def build_labels_after_model_release(examples, feature_rows):
    """Join controlled magnitude classes only after GE1 models are released."""

    by_family = {item.physical_family_id: item for item in examples}
    features_by_label_key = {}
    for row in feature_rows:
        features_by_label_key.setdefault(row["label_key"], []).append(row)
    if (
        len(features_by_label_key) != SCALED_OPERATION_COUNT
        or any(
            tuple(sorted(item["arm"] for item in rows)) != PROBE_ARMS
            for rows in features_by_label_key.values()
        )
    ):
        raise GraphEncoderError(
            "probe_label_alignment_failure",
            "every physical operation must have one feature per arm",
        )
    labels = []
    for label_key in sorted(features_by_label_key):
        arm_rows = features_by_label_key[label_key]
        row = arm_rows[0]
        if any(
            (
                item["family_id"], item["operation_index"], item["operation_type"]
            )
            != (row["family_id"], row["operation_index"], row["operation_type"])
            for item in arm_rows[1:]
        ):
            raise GraphEncoderError(
                "probe_label_alignment_failure", "arm feature keys disagree"
            )
        example = by_family[row["family_id"]]
        operation_nodes = tuple(example.target.operation_sequence)
        if row["operation_index"] >= len(operation_nodes):
            raise GraphEncoderError(
                "probe_label_alignment_failure", "operation slot is absent"
            )
        node_index = operation_nodes[row["operation_index"]]
        target_type = NODE_TYPES.tokens[example.target.node_type_ids[node_index]]
        if target_type != row["operation_type"]:
            raise GraphEncoderError(
                "probe_label_alignment_failure",
                "autonomous predicted operation type differs from target",
            )
        channel = 37 if target_type == "extrude" else 38
        normalized = float(example.target.geometry[node_index][channel])
        physical = normalized * float(GEOMETRY_CHANNEL_SCALES[channel])
        grid = OPERATION_GRIDS[target_type]
        matches = [index for index, value in enumerate(grid) if physical == value]
        if len(matches) != 1 or not bool(example.target.geometry_mask[node_index][channel]):
            raise GraphEncoderError(
                "probe_label_alignment_failure", "target is outside controlled grid"
            )
        labels.append({
            "label_key": label_key,
            "family_id": row["family_id"],
            "operation_index": row["operation_index"],
            "operation_type": target_type,
            "operation_template": example.metadata.operation_template,
            "class_index": matches[0],
            "class_physical_value": grid[matches[0]],
        })
    return {
        "schema_version": PROBE_LABEL_VERSION,
        "labels_joined_after_ge1_models_released": True,
        "grouping_unit": "physical_family",
        "class_grids": {name: list(values) for name, values in OPERATION_GRIDS.items()},
        "rows": labels,
    }


def _analysis_rows(feature_rows, label_payload, arm):
    labels = {item["label_key"]: item for item in label_payload["rows"]}
    result = []
    for feature in feature_rows:
        if feature["arm"] != arm:
            continue
        label = labels[feature["label_key"]]
        row = dict(feature)
        row.update({
            "operation_template": label["operation_template"],
            "class_index": label["class_index"],
        })
        result.append(row)
    return tuple(result)


def run_all_probe_analyses(feature_rows, label_payload):
    """Run the complete frozen scalar, linear, MLP, and permutation analyses."""

    scalar = {"schema_version": PROBE_RESULT_VERSION, "arms": {}}
    results = {
        "schema_version": PROBE_RESULT_VERSION,
        "pipelines": [],
        "combined": [],
        "interpretations": [],
    }
    for arm in PROBE_ARMS:
        rows = _analysis_rows(feature_rows, label_payload, arm)
        scalar["arms"][arm] = {
            operation_type: scalar_analysis(rows, operation_type)
            for operation_type in OPERATION_TYPES
        }
        combined_fidelity = {}
        for row in rows:
            target = OPERATION_GRIDS[row["operation_type"]][row["class_index"]]
            passed = abs(row["A_physical"] - target) < FIDELITY_ERROR_LIMITS[
                row["operation_type"]
            ]
            combined_fidelity.setdefault(row["family_id"], []).append(passed)
        scalar["arms"][arm]["combined_existing_fidelity_gate"] = {
            "operation_accuracy": (
                sum(
                    passed
                    for values in combined_fidelity.values()
                    for passed in values
                ) / SCALED_OPERATION_COUNT
            ),
            "physical_family_all_operations_correct_accuracy": (
                sum(all(values) for values in combined_fidelity.values())
                / SCALED_FAMILY_COUNT
            ),
            "operation_count": SCALED_OPERATION_COUNT,
            "physical_family_count": SCALED_FAMILY_COUNT,
        }
        labels_by_key = {item["key"]: item["class_index"] for item in rows}
        pipeline_map = {}
        for operation_type, feature_level, probe_kind in itertools.product(
            OPERATION_TYPES, FEATURE_LEVELS, PROBE_KINDS
        ):
            selected = tuple(
                item for item in rows if item["operation_type"] == operation_type
            )
            result = fit_probe_pipeline(
                selected,
                labels_by_key,
                arm=arm,
                operation_type=operation_type,
                feature_level=feature_level,
                probe_kind=probe_kind,
            )
            result["label_permutation"] = run_permutation_controls(
                selected,
                labels_by_key,
                arm=arm,
                operation_type=operation_type,
                feature_level=feature_level,
                probe_kind=probe_kind,
                observed_result=result,
            )
            grouped = scalar["arms"][arm][operation_type][
                "A_grouped_threshold"
            ]["metrics"]
            decision = accessibility_decision(result, grouped)
            if feature_level != "C":
                decision["materially_exceeds_scalar_grouped_threshold"] = False
                decision["material_scalar_comparison_applicable"] = False
                decision["material_scalar_comparison_reason"] = (
                    "frozen material-excess rule applies to decoder-state C only"
                )
            else:
                decision["material_scalar_comparison_applicable"] = True
                decision["material_scalar_comparison_reason"] = None
            result["interpretation"] = decision
            result["scalar_only_resubstitution_ceiling"] = scalar["arms"][arm][
                operation_type
            ]["A_scalar_only_resubstitution_ceiling"]["metrics"]
            result["difference_from_scalar_grouped_threshold"] = {
                "raw_accuracy": (
                    result["lofo"]["metrics"]["accuracy"] - grouped["accuracy"]
                ),
                "balanced_accuracy": (
                    result["lofo"]["metrics"]["balanced_accuracy"]
                    - grouped["balanced_accuracy"]
                ),
            }
            pipeline_map[(operation_type, feature_level, probe_kind)] = result
            results["pipelines"].append(result)
        for feature_level, probe_kind in itertools.product(FEATURE_LEVELS, PROBE_KINDS):
            extrude = pipeline_map[("extrude", feature_level, probe_kind)]
            revolve = pipeline_map[("revolve", feature_level, probe_kind)]
            results["combined"].append({
                "arm": arm,
                "feature_level": feature_level,
                "probe_kind": probe_kind,
                "resubstitution": combined_prediction_metrics(
                    rows,
                    labels_by_key,
                    extrude["resubstitution"]["predictions_by_key"],
                    revolve["resubstitution"]["predictions_by_key"],
                ),
                "lofo": combined_prediction_metrics(
                    rows,
                    labels_by_key,
                    extrude["lofo"]["predictions_by_key"],
                    revolve["lofo"]["predictions_by_key"],
                ),
            })
        for operation_type in OPERATION_TYPES:
            c_linear = pipeline_map[(operation_type, "C", "linear")]
            c_mlp = pipeline_map[(operation_type, "C", "mlp")]
            d_linear = pipeline_map[(operation_type, "D", "linear")]
            d_mlp = pipeline_map[(operation_type, "D", "mlp")]
            c_access = any(
                item["interpretation"]["held_out_family_accessibility_evidence"]
                for item in (c_linear, c_mlp)
            )
            d_access = any(
                item["interpretation"]["held_out_family_accessibility_evidence"]
                for item in (d_linear, d_mlp)
            )
            c_material = any(
                item["interpretation"][
                    "materially_exceeds_scalar_grouped_threshold"
                ]
                for item in (c_linear, c_mlp)
            )
            results["interpretations"].append({
                "arm": arm,
                "operation_type": operation_type,
                "decoder_state_accessibility_any_probe": c_access,
                "encoder_memory_accessibility_any_probe": d_access,
                "decoder_state_materially_exceeds_scalar_any_probe": c_material,
                "D_accessible_while_C_not": d_access and not c_access,
                "C_linear_lacks_but_C_mlp_has": (
                    not c_linear["interpretation"][
                        "held_out_family_accessibility_evidence"
                    ]
                    and c_mlp["interpretation"][
                        "held_out_family_accessibility_evidence"
                    ]
                ),
                "D_linear_lacks_but_D_mlp_has": (
                    not d_linear["interpretation"][
                        "held_out_family_accessibility_evidence"
                    ]
                    and d_mlp["interpretation"][
                        "held_out_family_accessibility_evidence"
                    ]
                ),
                "every_controlled_probe_lacks_accessibility_evidence": (
                    not c_access and not d_access
                ),
                "authorizes_model_or_decoder_change": False,
                "authorizes_stage6_or_c8": False,
            })
    scalar["geometry_loss_audit"] = geometry_loss_audit()
    return scalar, results


def _regular_files(root, excluded=()):
    excluded_values = set(excluded)
    paths = []
    for path in Path(root).rglob("*"):
        if path.is_symlink():
            raise GraphEncoderError(
                "invalid_probe_artifact", "artifact symlinks are forbidden"
            )
        if path.is_file():
            relative = path.relative_to(root).as_posix()
            if relative not in excluded_values:
                paths.append(relative)
    return tuple(sorted(paths))


def prepare_probe_output(output_dir, *, repository_root, corpus_dir,
                         source_artifact, job_id=None):
    identity = validate_slurm_job_id(job_id)
    raw = Path(output_dir)
    if raw.exists() or raw.is_symlink():
        raise GraphEncoderError(
            "probe_output_exists", "output must be a new non-symlink path"
        )
    final = raw.resolve()
    roots = tuple(Path(item).resolve() for item in (
        repository_root, corpus_dir, source_artifact
    ))
    if any(_is_within(final, root) for root in roots):
        raise GraphEncoderError(
            "unsafe_probe_output_location",
            "output must be outside repository, corpus, and source artifact",
        )
    final.parent.mkdir(parents=True, exist_ok=True)
    staging = final.with_name(final.name + ".incomplete-" + identity)
    if staging.exists() or staging.is_symlink():
        raise GraphEncoderError(
            "probe_staging_exists", "incomplete output already exists"
        )
    staging.mkdir()
    return final, staging


def _validate_feature_payload(payload):
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version", "protocol_version", "target_free", "feature_dimensions",
        "rows",
    }:
        raise GraphEncoderError(
            "invalid_detached_features", "feature payload field set differs"
        )
    if (
        payload["schema_version"] != PROBE_FEATURE_VERSION
        or payload["protocol_version"] != PROBE_PROTOCOL_VERSION
        or payload["target_free"] is not True
        or payload["feature_dimensions"] != FEATURE_DIMENSIONS
    ):
        raise GraphEncoderError(
            "invalid_detached_features", "feature identity differs"
        )
    validate_feature_rows(payload["rows"])
    return payload


def write_target_free_features(staging, rows):
    """Write and hash detached features before any label file exists."""

    if torch is None:
        raise RuntimeError("feature serialization requires PyTorch")
    root = Path(staging)
    if (root / "labels.json").exists():
        raise GraphEncoderError(
            "target_isolation_order_failure", "labels exist before features"
        )
    payload = {
        "schema_version": PROBE_FEATURE_VERSION,
        "protocol_version": PROBE_PROTOCOL_VERSION,
        "target_free": True,
        "feature_dimensions": dict(FEATURE_DIMENSIONS),
        "rows": list(validate_feature_rows(rows)),
    }
    path = root / "detached_features.pt"
    torch.save(payload, str(path))
    observed = torch.load(str(path), map_location="cpu")
    _validate_feature_payload(observed)
    feature_hash = _file_sha256(path)
    manifest = {
        "schema_version": PROBE_FEATURE_VERSION,
        "protocol_version": PROBE_PROTOCOL_VERSION,
        "feature_file": "detached_features.pt",
        "feature_file_byte_size": path.stat().st_size,
        "feature_file_sha256": feature_hash,
        "written_and_hashed_before_label_join": True,
        "target_or_label_content_present": False,
        "model_state_present": False,
        "optimizer_state_present": False,
        "row_count": len(payload["rows"]),
        "feature_dimensions": dict(FEATURE_DIMENSIONS),
    }
    _atomic_write_json(root / "feature_manifest.json", manifest)
    return payload, manifest


def finalize_probe_artifact(staging_dir, final_dir):
    staging = Path(staging_dir)
    final = Path(final_dir)
    ordinary = _regular_files(
        staging, excluded=("artifact_manifest.json", "SHA256SUMS")
    )
    if ordinary != EXPECTED_ARTIFACT_FILES:
        raise GraphEncoderError(
            "incomplete_probe_artifact", "probe artifact file set differs"
        )
    forbidden_suffixes = (".ckpt", ".pth")
    if any(path.endswith(forbidden_suffixes) for path in ordinary):
        raise GraphEncoderError(
            "checkpoint_write_forbidden", "probe artifact cannot contain checkpoints"
        )
    manifest = {
        "schema_version": PROBE_ARTIFACT_VERSION,
        "protocol_version": PROBE_PROTOCOL_VERSION,
        "artifacts": [
            {
                "path": path,
                "byte_size": (staging / path).stat().st_size,
                "sha256": _file_sha256(staging / path),
            }
            for path in ordinary
        ],
    }
    _atomic_write_json(staging / "artifact_manifest.json", manifest)
    checksum_paths = _regular_files(staging, excluded=("SHA256SUMS",))
    checksum_text = "".join(
        "{}  {}\n".format(_file_sha256(staging / path), path)
        for path in checksum_paths
    )
    _atomic_write_bytes(staging / "SHA256SUMS", checksum_text.encode("utf-8"))
    verify_probe_artifact(staging, allow_incomplete_name=True)
    os.replace(str(staging), str(final))
    return final


def verify_probe_artifact(path, *, allow_incomplete_name=False,
                          expected_commit=None, expected_slurm_job_id=None):
    """Verify integrity, target isolation, provenance, and non-authorization."""

    root = Path(path)
    if not root.is_dir() or root.is_symlink():
        raise GraphEncoderError(
            "invalid_probe_artifact", "artifact root must be a real directory"
        )
    if ".incomplete-" in root.name and not allow_incomplete_name:
        raise GraphEncoderError(
            "invalid_probe_artifact", "incomplete artifact cannot be final"
        )
    manifest = json.loads((root / "artifact_manifest.json").read_text("utf-8"))
    if (
        manifest.get("schema_version") != PROBE_ARTIFACT_VERSION
        or manifest.get("protocol_version") != PROBE_PROTOCOL_VERSION
    ):
        raise GraphEncoderError(
            "invalid_probe_artifact", "artifact identity differs"
        )
    rows = manifest.get("artifacts")
    if not isinstance(rows, list):
        raise GraphEncoderError("invalid_probe_artifact", "artifact rows are absent")
    listed = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"path", "byte_size", "sha256"}:
            raise GraphEncoderError(
                "invalid_probe_artifact", "artifact row is malformed"
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
                "probe_artifact_integrity_failure", "artifact size or hash differs"
            )
        listed.append(relative)
    ordinary = _regular_files(root, excluded=("artifact_manifest.json", "SHA256SUMS"))
    if tuple(listed) != ordinary or ordinary != EXPECTED_ARTIFACT_FILES:
        raise GraphEncoderError(
            "invalid_probe_artifact", "artifact manifest coverage differs"
        )
    checksum = (root / "SHA256SUMS").read_bytes()
    if not checksum.endswith(b"\n"):
        raise GraphEncoderError(
            "invalid_probe_artifact", "SHA256SUMS requires a final LF"
        )
    checksum_paths = []
    for line in checksum.decode("utf-8").splitlines():
        parts = line.split("  ", 1)
        if len(parts) != 2 or len(parts[0]) != 64:
            raise GraphEncoderError(
                "invalid_probe_artifact", "SHA256SUMS row is malformed"
            )
        relative = _safe_relative_path(parts[1])
        if _file_sha256(root / relative) != parts[0]:
            raise GraphEncoderError(
                "probe_artifact_integrity_failure", "SHA256SUMS hash differs"
            )
        checksum_paths.append(relative)
    if tuple(checksum_paths) != _regular_files(root, excluded=("SHA256SUMS",)):
        raise GraphEncoderError(
            "invalid_probe_artifact", "SHA256SUMS coverage differs"
        )
    resolved = json.loads((root / "resolved_config.json").read_text("utf-8"))
    labels = json.loads((root / "labels.json").read_text("utf-8"))
    scalar = json.loads((root / "scalar_analysis.json").read_text("utf-8"))
    probe_results = json.loads((root / "probe_results.json").read_text("utf-8"))
    events = _read_canonical_jsonl(root / "metrics.jsonl")
    if torch is None:
        raise RuntimeError("feature artifact verification requires PyTorch")
    features = torch.load(str(root / "detached_features.pt"), map_location="cpu")
    _validate_feature_payload(features)
    feature_manifest = json.loads((root / "feature_manifest.json").read_text("utf-8"))
    if (
        feature_manifest.get("feature_file_sha256")
        != _file_sha256(root / "detached_features.pt")
        or feature_manifest.get("written_and_hashed_before_label_join") is not True
        or feature_manifest.get("target_or_label_content_present") is not False
        or labels.get("schema_version") != PROBE_LABEL_VERSION
        or labels.get("labels_joined_after_ge1_models_released") is not True
        or len(labels.get("rows", ())) != SCALED_OPERATION_COUNT
        or len({item.get("label_key") for item in labels.get("rows", ())})
        != SCALED_OPERATION_COUNT
        or {item.get("label_key") for item in labels.get("rows", ())}
        != {item.get("label_key") for item in features["rows"]}
        or scalar.get("schema_version") != PROBE_RESULT_VERSION
        or probe_results.get("schema_version") != PROBE_RESULT_VERSION
    ):
        raise GraphEncoderError(
            "invalid_probe_artifact", "feature/label/result identity differs"
        )
    if (
        resolved.get("protocol_version") != PROBE_PROTOCOL_VERSION
        or resolved.get("source_repaired_result", {}).get("commit")
        != SOURCE_REPAIRED_COMMIT
        or resolved.get("source_repaired_result", {}).get("slurm_job_id")
        != SOURCE_REPAIRED_JOB_ID
        or resolved.get("source_repaired_result", {}).get("result_changed") is not False
        or resolved.get("probe_contract") != probe_contract()
        or resolved.get("runtime", {}).get("python") != "3.8.13"
        or str(resolved.get("runtime", {}).get("pytorch", "")).split("+")[0]
        != "1.11.0"
        or resolved.get("runtime", {}).get("device") != "cpu"
        or resolved.get("runtime", {}).get("cuda_available") is not False
        or resolved.get("runtime", {}).get("cpu_threads") != 1
    ):
        raise GraphEncoderError(
            "invalid_probe_artifact", "resolved contract or runtime differs"
        )
    if expected_commit is not None and resolved.get("source", {}).get("git_commit") != expected_commit:
        raise GraphEncoderError("invalid_probe_artifact", "diagnostic commit differs")
    if (
        expected_slurm_job_id is not None
        and resolved.get("runtime", {}).get("slurm_job_id")
        != str(expected_slurm_job_id)
    ):
        raise GraphEncoderError("invalid_probe_artifact", "Slurm job differs")
    if not events or events[-1].get("event") != "representation_probe_completed":
        raise GraphEncoderError(
            "invalid_probe_artifact", "terminal completion event is absent"
        )
    terminal = events[-1]
    if (
        terminal.get("representation_probe_completed") is not True
        or terminal.get("source_repaired_result_changed") is not False
        or terminal.get("ge1_training_performed") is not False
        or terminal.get("ge1_backward_pass_performed") is not False
        or terminal.get("ge1_optimizer_constructed") is not False
        or terminal.get("ge1_checkpoint_written") is not False
        or terminal.get("model_or_decoder_repair_implemented_or_invoked") is not False
        or terminal.get("stage6_performed") is not False
        or terminal.get("c8_or_later_performed") is not False
        or any(terminal.get(name) is not False for name in PROTECTED_ACCESS_FIELDS)
    ):
        raise GraphEncoderError(
            "invalid_probe_artifact", "terminal access or authority differs"
        )
    return {
        "protocol_version": PROBE_PROTOCOL_VERSION,
        "file_count": len(_regular_files(root)),
        "feature_row_count": len(features["rows"]),
        "label_row_count": len(labels["rows"]),
        "pipeline_count": len(probe_results["pipelines"]),
        "integrity_verified": True,
    }


def _require_authoritative_runtime():
    if torch is None:
        raise RuntimeError("representation probe requires PyTorch")
    job_id = validate_slurm_job_id()
    if platform.python_version() != "3.8.13":
        raise GraphEncoderError(
            "invalid_probe_runtime", "Python 3.8.13 is required"
        )
    if str(torch.__version__).split("+")[0] != "1.11.0":
        raise GraphEncoderError(
            "invalid_probe_runtime", "PyTorch 1.11.0 is required"
        )
    if torch.cuda.is_available() or torch.get_num_threads() != 1:
        raise GraphEncoderError(
            "invalid_probe_runtime", "CPU-only one-thread execution is required"
        )
    return job_id


def _load_scaled_examples(corpus_dir, family_ids, tracker):
    tracker.operation_template_train_payload_accessed = "not_confirmed_on_failure"
    tracker.scaled_train_payload_accessed = "not_confirmed_on_failure"
    examples = tuple(load_train(corpus_dir, family_ids))
    observed = tuple(sorted(item.physical_family_id for item in examples))
    if observed != tuple(family_ids):
        raise GraphEncoderError(
            "probe_payload_alignment_failure", "loaded family IDs differ"
        )
    for item in examples:
        if tuple(item.metadata.geometry_encodings) != ("continuous", "quantized"):
            raise GraphEncoderError(
                "probe_representation_variant_failure",
                "continuous/quantized source pairing differs",
            )
    tracker.operation_template_train_payload_accessed = True
    tracker.scaled_train_payload_accessed = True
    return examples


def run_representation_probe(*, corpus_dir, source_artifact, output_dir,
                             repository_root, expected_commit,
                             access_tracker=None):
    """Execute accepted ADR-0011 under the separately authorized runner."""

    job_id = _require_authoritative_runtime()
    tracker = access_tracker or RepresentationProbeAccessTracker()
    source = _source_identity(repository_root, expected_commit)
    final, staging = prepare_probe_output(
        output_dir,
        repository_root=repository_root,
        corpus_dir=corpus_dir,
        source_artifact=source_artifact,
        job_id=job_id,
    )
    started = time.perf_counter()
    events = [{
        "event": "representation_probe_started",
        "protocol_version": PROBE_PROTOCOL_VERSION,
        "source": source,
        "source_repaired_result_changed": False,
        "access": tracker.declarations(),
    }]
    tracker.source_artifact_accessed = "not_confirmed_on_failure"
    tracker.source_checkpoint_accessed = "not_confirmed_on_failure"
    source_evidence = _source_repaired_evidence(source_artifact)
    tracker.source_artifact_accessed = True
    tracker.source_checkpoint_accessed = True
    events.append({
        "event": "representation_probe_source_artifact_verified",
        "source_hashes": source_evidence["hashes"],
        "source_repaired_result_changed": False,
        "access": tracker.declarations(),
    })

    tracker.operation_template_manifest_accessed = "not_confirmed_on_failure"
    selection = select_c7_sufficiency_subsets(corpus_dir)
    tracker.operation_template_manifest_accessed = True
    scaled_ids = tuple(selection.scaled_family_ids)
    if (
        scaled_ids != source_evidence["scaled_family_ids"]
        or selected_family_ids_sha256(scaled_ids)
        != C7_V2_SCALED_FAMILY_IDS_SHA256
    ):
        raise GraphEncoderError(
            "source_repaired_cohort_mismatch", "manifest and source cohort differ"
        )
    examples = _load_scaled_examples(corpus_dir, scaled_ids, tracker)
    ordered = tuple(sorted(examples, key=lambda item: item.physical_family_id))
    partition_identity = training_partition_identity(scaled_ids)
    input_batches_by_arm = {
        arm: tuple(
            autonomous_input_from_paired(
                build_paired_batch(ordered[start:start + 8]), arm
            )
            for start in range(0, len(ordered), 8)
        )
        for arm in PROBE_ARMS
    }
    all_features = []
    reproductions = []
    model_hashes = {}
    for arm in PROBE_ARMS:
        checkpoint = SOURCE_CHECKPOINTS[arm]
        recovery, unused_recovery_payload = _load_recovery_model_read_only(
            Path(source_artifact) / checkpoint["recovery"]["path"],
            arm=arm,
            expected_sha256=checkpoint["recovery"]["sha256"],
            partition_identity=partition_identity,
        )
        inference, unused_inference_payload = _load_inference_model_read_only(
            Path(source_artifact) / checkpoint["inference"]["path"],
            arm=arm,
            expected_sha256=checkpoint["inference"]["sha256"],
        )
        _assert_model_states_equal(recovery, inference)
        before = _state_sha256(inference)
        features, autonomous = extract_target_free_features(
            inference, input_batches_by_arm[arm], arm=arm
        )
        reproduction = reproduce_source_evidence(
            arm=arm,
            autonomous=autonomous,
            examples=ordered,
            original_metrics=source_evidence["original_metrics"][arm],
            original_fidelity=source_evidence["original_fidelity"][arm],
        )
        after = _state_sha256(inference)
        if before != after or any(parameter.grad is not None for parameter in inference.parameters()):
            raise GraphEncoderError(
                "ge1_model_mutation_detected", "GE1 state or gradient buffers changed"
            )
        model_hashes[arm] = {
            "recovery_state_sha256": _state_sha256(recovery),
            "inference_state_sha256_before": before,
            "inference_state_sha256_after": after,
            "state_equal_between_recovery_and_inference": True,
        }
        all_features.extend(features)
        reproductions.append(reproduction)
        del recovery
        del inference
        del unused_recovery_payload
        del unused_inference_payload
        del autonomous
    feature_payload, feature_manifest = write_target_free_features(
        staging, tuple(all_features)
    )
    events.append({
        "event": "representation_probe_target_free_features_written",
        "feature_manifest": feature_manifest,
        "source_reproduction": reproductions,
        "model_state_hashes": model_hashes,
        "ge1_models_released_before_label_join": True,
        "access": tracker.declarations(),
    })

    # The only target-derived artifact is created after all GE1 models have
    # been released and the target-free feature file has been written/hashed.
    label_payload = build_labels_after_model_release(ordered, feature_payload["rows"])
    _atomic_write_json(staging / "labels.json", label_payload)
    scalar, probe_results = run_all_probe_analyses(
        feature_payload["rows"], label_payload
    )
    _atomic_write_json(staging / "scalar_analysis.json", scalar)
    _atomic_write_json(staging / "probe_results.json", probe_results)

    source_hashes_after = _validate_source_hashes(source_artifact)
    if source_hashes_after != source_evidence["hashes"]:
        raise GraphEncoderError(
            "source_repaired_artifact_mutation", "source hashes changed"
        )
    resolved = {
        "protocol_version": PROBE_PROTOCOL_VERSION,
        "artifact_version": PROBE_ARTIFACT_VERSION,
        "source": source,
        "source_repaired_result": {
            "protocol_version": REPAIRED_PROTOCOL_VERSION,
            "commit": SOURCE_REPAIRED_COMMIT,
            "slurm_job_id": SOURCE_REPAIRED_JOB_ID,
            "artifact_path": str(Path(source_artifact).resolve()),
            "result_changed": False,
            "hashes_before": source_evidence["hashes"],
            "hashes_after": source_hashes_after,
        },
        "authoritative_manifest": {
            "sha256": AUTHORITATIVE_FILE_SHA256,
            "partition": "train",
            "scaled_family_ids": list(scaled_ids),
            "scaled_family_ids_sha256": C7_V2_SCALED_FAMILY_IDS_SHA256,
        },
        "probe_contract": probe_contract(),
        "target_isolation": {
            "inference_accepted_targets": False,
            "feature_extraction_accepted_targets": False,
            "feature_file_written_and_hashed_before_label_join": True,
            "ge1_models_released_before_label_join": True,
            "probe_optimizers_contain_only_disposable_probe_parameters": True,
            "all_ge1_requires_grad_disabled": True,
            "feature_selection_sources": {
                "operation_type": "autonomous_prediction",
                "operation_node": "autonomous_predicted_operation_indices",
                "operation_slot": "canonical_autonomous_operation_order",
                "geometry_channel": "autonomous_predicted_operation_type",
            },
        },
        "feature_manifest": feature_manifest,
        "source_reproduction": reproductions,
        "model_state_hashes": model_hashes,
        "runtime": {
            "python": platform.python_version(),
            "pytorch": str(torch.__version__),
            "device": "cpu",
            "cuda_available": bool(torch.cuda.is_available()),
            "cpu_threads": int(torch.get_num_threads()),
            "host": socket.gethostname(),
            "slurm_job_id": job_id,
        },
        "access": tracker.declarations(completed=True),
        "stage6_authorized": False,
        "c8_authorized": False,
        "scientific_model_or_decoder_repair_authorized": False,
    }
    _atomic_write_json(staging / "resolved_config.json", resolved)
    events.append({
        "event": "representation_probe_completed",
        "protocol_version": PROBE_PROTOCOL_VERSION,
        "source_repaired_result_changed": False,
        "source_hashes_unchanged": True,
        "feature_row_count": len(feature_payload["rows"]),
        "label_row_count": len(label_payload["rows"]),
        "pipeline_count": len(probe_results["pipelines"]),
        "elapsed_seconds": time.perf_counter() - started,
        **tracker.declarations(completed=True),
    })
    _atomic_write_jsonl(staging / "metrics.jsonl", events)
    _verify_source_unchanged(source, repository_root, expected_commit)
    finalize_probe_artifact(staging, final)
    verified = verify_probe_artifact(
        final,
        expected_commit=expected_commit,
        expected_slurm_job_id=job_id,
    )
    return {"artifact_path": str(final), "artifact_verification": verified}


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run the accepted read-only GE1 representation probe"
    )
    parser.add_argument("--corpus-dir", required=True)
    parser.add_argument("--source-artifact", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--expected-commit", required=True)
    arguments = parser.parse_args(argv)
    result = run_representation_probe(
        corpus_dir=arguments.corpus_dir,
        source_artifact=arguments.source_artifact,
        output_dir=arguments.output_dir,
        repository_root=arguments.repository_root,
        expected_commit=arguments.expected_commit,
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
