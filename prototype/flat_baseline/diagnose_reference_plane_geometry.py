"""Read-only Stage 2H diagnosis of frozen V2 reference-plane geometry."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
import sys

import torch

from prototype.constrained_profile_decoder import profile_targets_for_loss
from prototype.controlled_data.builders import _plane_geometry
from prototype.controlled_data.factors import OperationTemplate, ReferencePlane
from prototype.model_data.batching import collate_flat
from prototype.model_data.canonical import _reference_plane
from prototype.model_data.geometry import (
    GEOMETRY_CHANNELS,
    GEOMETRY_CHANNEL_SCALES,
    GEOMETRY_WIDTH,
    LENGTH_SCALE,
    NORMALIZED_MAX,
    NORMALIZED_MIN,
    decode,
    denormalize_applicable_geometry,
)
from prototype.model_data.vocab import NODE_TYPES, REFERENCE_PLANES
from prototype.profile_geometry import PROFILE_FAMILIES
from prototype.profile_geometry_torch import canonicalize_profile_tensors
from prototype.representation.model import GeometryEncoding
from prototype.representation.serialization import _geometry_to_dict
from prototype.representation.validation import FRAME_ORTHONORMAL_TOLERANCE

from .autonomous import RawDecodedNode, derive_geometry_mask
from .checkpointing import capture_rng_state, restore_rng_state
from .constrained_v2 import (
    NON_PROFILE_GEOMETRY_CHANNELS,
    NON_PROFILE_GEOMETRY_INDICES,
    REMAINING_CATEGORICAL_TARGET_INDICES,
    ConstrainedProfileV2Model,
    scatter_non_profile_geometry,
    select_non_profile_geometry,
)
from .constrained_v2_config import ConstrainedProfileV2Config
from .constrained_v2_conversion import (
    _prediction_row,
    construct_v2_predicted_node_tensors,
    validate_and_convert_v2_teacher_forced_prediction,
)
from .constrained_v2_pilot import load_pilot_data
from .constrained_v2_pilot_config import ConstrainedV2PilotConfig
from .constrained_v2_reference_plane_diagnostic_config import (
    ReferencePlaneDiagnosticConfig,
    ReferencePlaneDiagnosticError,
    validate_reference_plane_partition_authorization,
)
from .conversion import _Failures, _GEOMETRY_TOLERANCE, _validate_plane
from .diagnose_constrained_v2_pilot import (
    _assert_explicit_vq_state,
    _clone_tree,
    _raw_prediction_is_finite,
    _raw_profile_records_are_canonical,
    _replace_oracle_relations,
    _require_finite_json,
    _sha256,
    _tree_equal,
    _vq_matches_snapshot,
    audit_frozen_checkpoint_payload,
)
from .provenance import source_state
from .run_logging import JsonlLogger


REFERENCE_PLANE_INDICES = tuple(range(9))
OTHER_NON_PROFILE_INDICES = tuple(range(33, 39))
REFERENCE_PLANE_CHANNELS = tuple(GEOMETRY_CHANNELS[:9])
REFERENCE_PLANE_TOLERANCE = 1e-6
DEGENERATE_VECTOR_EPSILON = 1e-12
MATERIAL_VALIDITY_IMPROVEMENT = 0.10
HIGH_VALIDITY = 0.50
EXPECTED_H0_FAILURES = {"invalid_reference_plane_geometry": 68}
CANONICAL_FRAMES = {
    "XY": ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    "XZ": ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    "YZ": ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
}
ATOMIC_CONSTRAINT_ORDER = (
    "applicability_mask",
    "finiteness",
    "normalized_bounds",
    "origin_zero",
    "x_axis_nonzero",
    "y_axis_nonzero",
    "x_axis_unit",
    "y_axis_unit",
    "axes_orthogonal",
    "categorical_x_axis",
    "categorical_y_axis",
    "canonical_frame",
)
ARM_CONTRACTS = {
    "H0": {
        "label": "stage_2g_arm_e_reproduction",
        "oracle": True,
        "replacements": ["complete_discrete_structure"],
        "retained_predictions": [
            "compact_profile_parameters", "all_non_profile_geometry"
        ],
    },
    "H1": {
        "label": "oracle_reference_plane_geometry_only",
        "oracle": True,
        "replacements": ["reference_plane_geometry_channels_0_through_8"],
        "retained_predictions": [
            "compact_profile_parameters", "other_non_profile_geometry"
        ],
    },
    "H2": {
        "label": "oracle_all_non_profile_geometry",
        "oracle": True,
        "replacements": ["serialized_channels_0_through_8_and_33_through_38"],
        "retained_predictions": ["compact_profile_parameters"],
    },
    "H3": {
        "label": "target_free_projected_reference_plane",
        "oracle": True,
        "replacements": ["deterministic_projection_of_predicted_channels_0_through_8"],
        "retained_predictions": [
            "compact_profile_parameters", "other_non_profile_geometry"
        ],
        "target_geometry_used": False,
    },
    "H4": {
        "label": "not_applicable_no_separable_bounded_scalar",
        "oracle": True,
        "enabled": False,
        "reason": (
            "canonical plane orientation fixes both axes and controlled origin "
            "is the constant zero; no independent bounded scalar remains"
        ),
    },
    "H5": {
        "label": "oracle_compact_profile_parameters_only",
        "oracle": True,
        "replacements": ["compact_profile_parameters"],
        "retained_predictions": ["all_non_profile_geometry"],
    },
}


@dataclass(frozen=True)
class ReferencePlaneDiagnosticResult:
    output_dir: str
    metrics_path: str
    examples_path: str
    classifications: tuple
    checkpoint_sha256: str


def reference_plane_contract():
    """Return and cross-check the authoritative serialized plane contract."""

    expected_channels = (
        "plane_origin_x", "plane_origin_y", "plane_origin_z",
        "plane_x_axis_x", "plane_x_axis_y", "plane_x_axis_z",
        "plane_y_axis_x", "plane_y_axis_y", "plane_y_axis_z",
    )
    expected_scales = (LENGTH_SCALE,) * 3 + (1.0,) * 6
    discrepancies = []
    if REFERENCE_PLANE_CHANNELS != expected_channels:
        discrepancies.append("serialized_channel_order")
    if tuple(GEOMETRY_CHANNEL_SCALES[:9]) != expected_scales:
        discrepancies.append("normalization_scales")
    if tuple(NON_PROFILE_GEOMETRY_INDICES[:9]) != REFERENCE_PLANE_INDICES:
        discrepancies.append("v2_gather_order")
    if tuple(NON_PROFILE_GEOMETRY_CHANNELS[:9]) != expected_channels:
        discrepancies.append("v2_channel_names")
    if _GEOMETRY_TOLERANCE != REFERENCE_PLANE_TOLERANCE:
        discrepancies.append("strict_converter_tolerance")
    generated = {}
    for name in ("XY", "XZ", "YZ"):
        geometry = _plane_geometry(
            ReferencePlane(name), GeometryEncoding.CONTINUOUS
        )
        origin = decode(geometry.origin)
        x_axis = decode(geometry.x_axis)
        y_axis = decode(geometry.y_axis)
        generated[name] = {
            "origin": list(origin),
            "x_axis": list(x_axis),
            "y_axis": list(y_axis),
        }
        if origin != (0.0, 0.0, 0.0) or (x_axis, y_axis) != CANONICAL_FRAMES[name]:
            discrepancies.append("controlled_builder_{}".format(name))
        if _reference_plane(geometry) != name:
            discrepancies.append("model_data_inference_{}".format(name))
        serialized = _geometry_to_dict(geometry)
        if list(serialized) != ["geometry_type", "origin", "x_axis", "y_axis"]:
            discrepancies.append("serialization_order_{}".format(name))
        categories = [REFERENCE_PLANES.id(None)] * 9
        categories[3] = REFERENCE_PLANES.id(name)
        mask = derive_geometry_mask(
            NODE_TYPES.id("reference_plane"), tuple(categories)
        )
        if mask != (True,) * 9 + (False,) * (GEOMETRY_WIDTH - 9):
            discrepancies.append("applicability_mask_{}".format(name))
        node = RawDecodedNode(
            0,
            NODE_TYPES.id("reference_plane"),
            tuple(categories),
            tuple(origin) + tuple(x_axis) + tuple(y_axis)
            + (0.0,) * (GEOMETRY_WIDTH - 9),
            mask,
        )
        physical = tuple(origin) + tuple(x_axis) + tuple(y_axis) \
            + (None,) * (GEOMETRY_WIDTH - 9)
        failures = _Failures()
        _validate_plane(node, physical, 0, failures)
        if failures.items:
            discrepancies.append("strict_converter_{}".format(name))
    if discrepancies:
        raise ReferencePlaneDiagnosticError(
            "reference_plane_contract_disagreement", repr(discrepancies)
        )
    channels = []
    meanings = (
        "origin_x", "origin_y", "origin_z",
        "x_axis_x", "x_axis_y", "x_axis_z",
        "y_axis_x", "y_axis_y", "y_axis_z",
    )
    for index, (channel, meaning, scale) in enumerate(
        zip(expected_channels, meanings, expected_scales)
    ):
        channels.append({
            "index": index,
            "channel": channel,
            "physical_meaning": meaning,
            "normalized_range": [NORMALIZED_MIN, NORMALIZED_MAX],
            "physical_scale": scale,
            "units": "length" if index < 3 else "dimensionless",
            "applicable_node_type": "reference_plane",
        })
    result = {
        "serialization_order": ["origin", "x_axis", "y_axis"],
        "channels": channels,
        "canonical_frames": {
            name: {"origin": [0.0, 0.0, 0.0],
                   "x_axis": list(frame[0]), "y_axis": list(frame[1])}
            for name, frame in CANONICAL_FRAMES.items()
        },
        "schema_constraints": {
            "axis_norm_absolute_tolerance": FRAME_ORTHONORMAL_TOLERANCE,
            "axis_dot_absolute_tolerance": FRAME_ORTHONORMAL_TOLERANCE,
            "origin_constraint": "finite_only_in_general_schema",
        },
        "controlled_constraints": {
            "origin": [0.0, 0.0, 0.0],
            "categorical_frame_absolute_tolerance": REFERENCE_PLANE_TOLERANCE,
            "no_clipping": True,
        },
        "sign_convention": "exact categorical axes listed above",
        "handedness": (
            "no serialized normal or third axis; x_cross_y is derived only"
        ),
        "padding_and_non_applicable": (
            "all values zero and all applicability bits false"
        ),
        "independent_implementations_checked": [
            "controlled_builder", "representation_serializer",
            "representation_validator", "model_data_normalization",
            "model_data_plane_inference", "v2_non_profile_gather_scatter",
            "autonomous_mask_derivation", "strict_converter",
        ],
        "generated_frames": generated,
    }
    _require_finite_json(result)
    return result


def evaluate_reference_plane(values, mask, plane_name):
    """Evaluate every atomic plane predicate without collapsing failures."""

    try:
        row = tuple(float(value) for value in values)
        applicability = tuple(mask)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("plane values and mask must be iterable numerics") from exc
    if len(row) != 9:
        raise ValueError("reference-plane values must have width 9")
    if len(applicability) not in (9, GEOMETRY_WIDTH):
        raise ValueError("reference-plane mask must have width 9 or 39")
    if plane_name not in CANONICAL_FRAMES:
        raise ValueError("unsupported reference-plane category")
    expected_mask = (
        (True,) * 9 if len(applicability) == 9
        else (True,) * 9 + (False,) * (GEOMETRY_WIDTH - 9)
    )
    finite = all(math.isfinite(value) for value in row)
    constraints = {}
    constraints["applicability_mask"] = _predicate(
        applicability == expected_mask,
        float(sum(a != b for a, b in zip(applicability, expected_mask))),
    )
    constraints["finiteness"] = _predicate(
        finite, 0.0 if finite else None
    )
    bounded = finite and all(NORMALIZED_MIN <= value <= NORMALIZED_MAX for value in row)
    bound_violation = (
        max(max(abs(value) - NORMALIZED_MAX, 0.0) for value in row)
        if finite else None
    )
    constraints["normalized_bounds"] = _predicate(bounded, bound_violation)
    if finite:
        origin = tuple(value * LENGTH_SCALE for value in row[:3])
        x_axis = row[3:6]
        y_axis = row[6:9]
        x_norm = _norm(x_axis)
        y_norm = _norm(y_axis)
        dot = _dot(x_axis, y_axis)
        normal = _cross3(x_axis, y_axis)
        normal_norm = _norm(normal)
        determinant = normal_norm * normal_norm
        expected_x, expected_y = CANONICAL_FRAMES[plane_name]
        origin_error = max(abs(value) for value in origin)
        x_unit_error = abs(x_norm - 1.0)
        y_unit_error = abs(y_norm - 1.0)
        orthogonality_error = abs(dot)
        x_category_error = max(
            abs(a - b) for a, b in zip(x_axis, expected_x)
        )
        y_category_error = max(
            abs(a - b) for a, b in zip(y_axis, expected_y)
        )
        canonical_error = max(origin_error, x_category_error, y_category_error)
        constraints["origin_zero"] = _predicate(
            origin_error <= REFERENCE_PLANE_TOLERANCE, origin_error
        )
        constraints["x_axis_nonzero"] = _predicate(
            x_norm > DEGENERATE_VECTOR_EPSILON,
            max(DEGENERATE_VECTOR_EPSILON - x_norm, 0.0),
        )
        constraints["y_axis_nonzero"] = _predicate(
            y_norm > DEGENERATE_VECTOR_EPSILON,
            max(DEGENERATE_VECTOR_EPSILON - y_norm, 0.0),
        )
        constraints["x_axis_unit"] = _predicate(
            x_unit_error <= FRAME_ORTHONORMAL_TOLERANCE, x_unit_error
        )
        constraints["y_axis_unit"] = _predicate(
            y_unit_error <= FRAME_ORTHONORMAL_TOLERANCE, y_unit_error
        )
        constraints["axes_orthogonal"] = _predicate(
            orthogonality_error <= FRAME_ORTHONORMAL_TOLERANCE,
            orthogonality_error,
        )
        constraints["categorical_x_axis"] = _predicate(
            x_category_error <= REFERENCE_PLANE_TOLERANCE, x_category_error
        )
        constraints["categorical_y_axis"] = _predicate(
            y_category_error <= REFERENCE_PLANE_TOLERANCE, y_category_error
        )
        constraints["canonical_frame"] = _predicate(
            canonical_error <= REFERENCE_PLANE_TOLERANCE, canonical_error
        )
        metrics = {
            "origin_norm": _norm(origin),
            "x_axis_norm": x_norm,
            "y_axis_norm": y_norm,
            "axis_dot": dot,
            "orthogonality_error": orthogonality_error,
            "derived_normal": list(normal),
            "derived_normal_norm": normal_norm,
            "derived_basis_determinant": determinant,
            "x_axis_angular_error_degrees": _angular_error(x_axis, expected_x),
            "y_axis_angular_error_degrees": _angular_error(y_axis, expected_y),
            "canonical_frame_max_error": canonical_error,
        }
    else:
        for name in ATOMIC_CONSTRAINT_ORDER[3:]:
            constraints[name] = _predicate(False, None)
        metrics = {
            "origin_norm": None,
            "x_axis_norm": None,
            "y_axis_norm": None,
            "axis_dot": None,
            "orthogonality_error": None,
            "derived_normal": None,
            "derived_normal_norm": None,
            "derived_basis_determinant": None,
            "x_axis_angular_error_degrees": None,
            "y_axis_angular_error_degrees": None,
            "canonical_frame_max_error": None,
        }
    failed = [name for name in ATOMIC_CONSTRAINT_ORDER if not constraints[name]["passed"]]
    mathematical = all(constraints[name]["passed"] for name in (
        "finiteness", "normalized_bounds", "x_axis_nonzero",
        "y_axis_nonzero", "x_axis_unit", "y_axis_unit", "axes_orthogonal",
    ))
    result = {
        "constraints": constraints,
        "failed_constraints": failed,
        "first_violated_predicate": failed[0] if failed else None,
        "mathematical_form_pass": mathematical,
        "controlled_contract_pass": not failed,
        "failure_subtype": (
            "valid" if not failed
            else "mathematically_malformed" if not mathematical
            else "well_formed_but_controlled_frame_mismatch"
        ),
        "metrics": metrics,
    }
    _require_finite_json(result)
    return result


def project_reference_plane(values):
    """Project a predicted plane to an orthonormal frame without targets."""

    try:
        row = tuple(float(value) for value in values)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("predicted plane must contain numeric values") from exc
    if len(row) != 9:
        raise ValueError("predicted plane must have width 9")
    if not all(math.isfinite(value) for value in row):
        raise ValueError("predicted plane must be finite")
    if any(not NORMALIZED_MIN <= value <= NORMALIZED_MAX for value in row):
        raise ValueError("predicted plane must be within normalized bounds")
    x_axis = row[3:6]
    if _norm(x_axis) <= DEGENERATE_VECTOR_EPSILON:
        x_axis = (1.0, 0.0, 0.0)
    else:
        x_axis = _scale(x_axis, 1.0 / _norm(x_axis))
    y_raw = row[6:9]
    y_axis = _subtract(y_raw, _scale(x_axis, _dot(y_raw, x_axis)))
    if _norm(y_axis) <= DEGENERATE_VECTOR_EPSILON:
        fallback = min(
            ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            key=lambda axis: (abs(_dot(axis, x_axis)), axis),
        )
        y_axis = _subtract(
            fallback, _scale(x_axis, _dot(fallback, x_axis))
        )
    y_axis = _scale(y_axis, 1.0 / _norm(y_axis))
    projected = (0.0, 0.0, 0.0) + tuple(x_axis) + tuple(y_axis)
    if not all(NORMALIZED_MIN <= value <= NORMALIZED_MAX for value in projected):
        raise AssertionError("orthonormal projection left normalized bounds")
    return tuple(0.0 if value == 0.0 else float(value) for value in projected)


def build_reference_plane_arms(output, target, profiles, physical_examples):
    """Build H0-H5 with explicit, narrow oracle replacement boundaries."""

    h0 = _build_h0(output, target, profiles, physical_examples)
    arms = {"H0": h0, "H1": [], "H2": [], "H3": [], "H4": (), "H5": []}
    for batch_index, (prediction, example) in enumerate(zip(h0, physical_examples)):
        arms["H1"].append(_replace_reference_plane_geometry(prediction, example))
        arms["H2"].append(_replace_all_non_profile_geometry(prediction, example))
        arms["H3"].append(_project_prediction_reference_plane(prediction))
        arms["H5"].append(_replace_compact_profile_parameters(
            prediction, profiles, batch_index, len(example.nodes)
        ))
    return {
        name: tuple(values) if isinstance(values, list) else values
        for name, values in arms.items()
    }


def run_frozen_reference_plane_diagnostic(
    corpus_dir, checkpoint_path, config=None
):
    config = config or ReferencePlaneDiagnosticConfig()
    validate_reference_plane_partition_authorization(config)
    output_dir = Path(config.output_dir)
    checkpoint_path = Path(checkpoint_path)
    resolved_output = output_dir.resolve()
    checkpoint_parent = checkpoint_path.resolve().parent
    if resolved_output == checkpoint_parent or checkpoint_parent in resolved_output.parents:
        raise ReferencePlaneDiagnosticError(
            "unsafe_output_path", "output must be outside the frozen pilot directory"
        )
    if output_dir.exists() or output_dir.is_symlink():
        raise ReferencePlaneDiagnosticError(
            "output_collision", "diagnostic output already exists"
        )
    provenance = source_state()
    if config.require_clean_source and (
        provenance["git_dirty"] is not False or provenance["git_commit"] is None
    ):
        raise ReferencePlaneDiagnosticError(
            "dirty_source", "diagnostic requires clean committed source"
        )
    checkpoint_hash_before = _sha256(checkpoint_path)
    if checkpoint_hash_before != config.expected_checkpoint_sha256:
        raise ReferencePlaneDiagnosticError(
            "checkpoint_hash_mismatch", "frozen checkpoint SHA-256 differs"
        )
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    checkpoint_audit = audit_frozen_checkpoint_payload(checkpoint, config)
    pilot_config = ConstrainedV2PilotConfig(**checkpoint["pilot_config"])
    pilot_config = replace(
        pilot_config,
        output_dir=config.output_dir,
        require_clean_source=config.require_clean_source,
    )
    data = load_pilot_data(corpus_dir, pilot_config)
    if (
        data.train.metadata() != checkpoint["training_partition_state"]
        or data.validation.metadata() != checkpoint["validation_partition_state"]
    ):
        raise ReferencePlaneDiagnosticError(
            "partition_fingerprint_mismatch",
            "loaded authorized partitions differ from checkpoint",
        )
    model_payload = dict(checkpoint["model_config"])
    if "profile_family_order" in model_payload:
        model_payload["profile_family_order"] = tuple(
            model_payload["profile_family_order"]
        )
    model_config = ConstrainedProfileV2Config(**model_payload)
    model_config.validate()
    if config.device == "cuda" and not torch.cuda.is_available():
        raise ReferencePlaneDiagnosticError(
            "unavailable_device", "CUDA was requested but is unavailable"
        )
    device = torch.device(config.device)
    construction_rng = capture_rng_state(torch)
    model = ConstrainedProfileV2Model(model_config).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    restore_rng_state(construction_rng, torch)
    _assert_explicit_vq_state(model, checkpoint["vq_state"])
    model.eval()

    contract = reference_plane_contract()
    output_dir.mkdir(parents=True)
    logger = JsonlLogger(output_dir / "metrics.jsonl")
    example_logger = JsonlLogger(output_dir / "reference_plane_examples.jsonl")
    logger.write({
        "event": "diagnostic_metadata",
        "diagnostic_config": config.to_dict(),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_hash_before,
        "checkpoint_audit": checkpoint_audit,
        "training_partition": data.train.metadata(),
        "validation_partition": data.validation.metadata(),
        "source_provenance": provenance,
        "arm_contracts": ARM_CONTRACTS,
        "training_occurred": False,
        "optimizer_instantiated": False,
        "training_function_called": False,
        "systematic_partition_accessed": False,
        "test_partition_accessed": False,
    })
    logger.write({"event": "reference_plane_contract", "contract": contract})
    logger.write({
        "event": "vq_stage2g_finding",
        "train": {"active_code_count": 1, "dominant_code_fraction": 1.0,
                  "perplexity": 1.0},
        "iid_validation": {"active_code_count": 1,
                           "dominant_code_fraction": 1.0, "perplexity": 1.0},
        "prequantized_variance_nonzero": True,
        "stage2g_collapse_classification": "neither",
        "interpretation": "quantizer_utilization_collapse_without_proven_encoder_or_codebook_vector_collapse",
        "recomputed": False,
    })

    model_state_before = _clone_tree(model.state_dict())
    rng_before = capture_rng_state(torch)
    train = _partition_diagnostic(
        model, data.train, model_config, config.batch_size, device,
        include_arms=False,
    )
    validation = _partition_diagnostic(
        model, data.validation, model_config, config.batch_size, device,
        include_arms=True,
    )
    for identity, summary in (("train", train), ("iid_validation", validation)):
        logger.write({
            "event": "reference_plane_partition_summary",
            "partition_identity": identity,
            "summary": summary["partition_summary"],
            "systematic_partition_accessed": False,
            "test_partition_accessed": False,
        })
        logger.write({
            "event": "reference_plane_constraint_summary",
            "partition_identity": identity,
            "summary": summary["constraint_summary"],
            "systematic_partition_accessed": False,
            "test_partition_accessed": False,
        })
        logger.write({
            "event": "reference_plane_channel_audit",
            "partition_identity": identity,
            "summary": summary["channel_audit"],
            "systematic_partition_accessed": False,
            "test_partition_accessed": False,
        })
    for name in ("H0", "H1", "H2", "H3", "H4", "H5"):
        logger.write({
            "event": "reference_plane_arm_summary",
            "arm": name,
            "contract": ARM_CONTRACTS[name],
            "summary": validation["arm_summaries"][name],
            "systematic_partition_accessed": False,
            "test_partition_accessed": False,
        })
    for record in validation["example_records"]:
        example_logger.write({
            "event": "reference_plane_example",
            **record,
            "systematic_partition_accessed": False,
            "test_partition_accessed": False,
        })
    h0 = validation["arm_summaries"]["H0"]
    if (
        h0["total_count"] != config.expected_validation_count
        or h0["valid_count"] != 0
        or h0["failure_reason_histogram"] != EXPECTED_H0_FAILURES
    ):
        raise ReferencePlaneDiagnosticError(
            "h0_reproduction_mismatch", repr(h0)
        )
    classifications = classify_reference_plane_diagnostic(
        validation["arm_summaries"]
    )
    logger.write({
        "event": "reference_plane_classification",
        "classifications": list(classifications),
        "rules": _classification_rules(),
        "systematic_partition_accessed": False,
        "test_partition_accessed": False,
    })
    checkpoint_hash_after = _sha256(checkpoint_path)
    state_audit = {
        "checkpoint_sha256_exact": checkpoint_hash_after == config.expected_checkpoint_sha256,
        "checkpoint_bytes_unchanged": checkpoint_hash_after == checkpoint_hash_before,
        "model_state_unchanged": _tree_equal(model_state_before, model.state_dict()),
        "vq_and_ema_state_unchanged": _vq_matches_snapshot(model, model_state_before),
        "rng_state_unchanged": _tree_equal(rng_before, capture_rng_state(torch)),
        "model_in_evaluation_mode": model.training is False,
        "no_parameter_gradients_created": all(
            parameter.grad is None for parameter in model.parameters()
        ),
        "optimizer_instantiated": False,
        "training_function_called": False,
    }
    if not all(value is True for name, value in state_audit.items()
               if name not in ("optimizer_instantiated", "training_function_called")):
        raise ReferencePlaneDiagnosticError(
            "diagnostic_mutated_state", repr(state_audit)
        )
    if state_audit["optimizer_instantiated"] is not False or state_audit["training_function_called"] is not False:
        raise ReferencePlaneDiagnosticError(
            "forbidden_execution", repr(state_audit)
        )
    logger.write({
        "event": "checkpoint_state_audit",
        **state_audit,
        "checkpoint_sha256_before": checkpoint_hash_before,
        "checkpoint_sha256_after": checkpoint_hash_after,
        "systematic_partition_accessed": False,
        "test_partition_accessed": False,
    })
    terminal = {
        "event": "terminal_success",
        "classifications": list(classifications),
        "checkpoint_sha256": checkpoint_hash_after,
        "validation_family_count": len(data.validation.family_ids),
        "training_occurred": False,
        "systematic_partition_accessed": False,
        "test_partition_accessed": False,
    }
    _require_finite_json(terminal)
    logger.write(terminal)
    return ReferencePlaneDiagnosticResult(
        str(output_dir), str(logger.path), str(example_logger.path),
        classifications, checkpoint_hash_after,
    )


def classify_reference_plane_diagnostic(summaries):
    h0 = summaries["H0"]["validity"]
    h1 = summaries["H1"]["validity"]
    h2 = summaries["H2"]["validity"]
    h3 = summaries["H3"]["validity"]
    classifications = []
    if h0 == 0.0 and h1 >= HIGH_VALIDITY:
        classifications.append("reference_plane_geometry_is_immediate_blocker")
    if h2 - h1 >= MATERIAL_VALIDITY_IMPROVEMENT:
        classifications.append("additional_non_profile_geometry_also_matters")
    if h1 >= HIGH_VALIDITY and h3 >= h1 - MATERIAL_VALIDITY_IMPROVEMENT:
        classifications.append("deterministic_projection_is_viable_direction")
    if h1 >= HIGH_VALIDITY and h3 < h1 - MATERIAL_VALIDITY_IMPROVEMENT:
        classifications.append("reference_plane_prediction_accuracy_is_inadequate")
    if h2 >= HIGH_VALIDITY:
        classifications.append("constrained_profile_decoder_supported_with_oracle_non_profile_geometry")
    if h1 == 0.0 and h2 == 0.0:
        classifications.append("routing_or_validator_contract_requires_reaudit")
    if not classifications:
        classifications.append("insufficient_evidence")
    return tuple(classifications)


def _partition_diagnostic(
    model, partition, model_config, batch_size, device, *, include_arms
):
    records = []
    routing = []
    arm_rows = {name: [] for name in ("H0", "H1", "H2", "H3", "H5")}
    with torch.no_grad():
        for start in range(0, len(partition.flat_examples), batch_size):
            flat = partition.flat_examples[start:start + batch_size]
            physical = partition.physical_examples[start:start + batch_size]
            batch = collate_flat(flat)
            if batch.family_ids != tuple(item.physical_family_id for item in physical):
                raise ReferencePlaneDiagnosticError(
                    "batch_alignment", "physical and flat examples differ"
                )
            inputs = {
                name: value.to(device)
                for name, value in batch.to_torch(torch).items()
            }
            target = {
                name: value.to(device)
                for name, value in batch.target.to_torch(torch).items()
            }
            profiles = profile_targets_for_loss(batch.target, inputs["geometry"])
            output = model(target=target, profile_targets=profiles, **inputs)
            arms = build_reference_plane_arms(
                output, target, profiles, physical
            )
            routing.append(_channel_routing_audit(
                output, target, arms["H0"], physical
            ))
            for batch_index, example in enumerate(physical):
                plane_position = _one_plane_position(example)
                plane_name = example.nodes[plane_position].reference_plane
                prediction = arms["H0"][batch_index]
                predicted = prediction.raw_nodes[plane_position]
                authoritative = example.target.geometry[plane_position]
                pre = evaluate_reference_plane(
                    predicted.normalized_geometry[:9],
                    predicted.derived_geometry_mask,
                    plane_name,
                )
                projected = project_reference_plane(
                    predicted.normalized_geometry[:9]
                )
                post = evaluate_reference_plane(
                    projected, predicted.derived_geometry_mask, plane_name
                )
                authoritative_evaluation = evaluate_reference_plane(
                    authoritative[:9], example.target.geometry_mask[plane_position],
                    plane_name,
                )
                arm_success = {}
                arm_failures = {}
                if include_arms:
                    for name in ("H0", "H1", "H2", "H3", "H5"):
                        raw = arms[name][batch_index]
                        conversion = validate_and_convert_v2_teacher_forced_prediction(
                            raw, max_operations=model_config.max_operations
                        )
                        arm_rows[name].append((example, raw, conversion))
                        arm_success[name] = bool(conversion.controlled_domain.valid)
                        arm_failures[name] = (
                            conversion.primary_failure.code
                            if conversion.primary_failure is not None else "valid"
                        )
                    arm_success["H4"] = None
                    arm_failures["H4"] = "not_applicable"
                record = {
                    "physical_family_id": example.physical_family_id,
                    "node_position": plane_position,
                    "operation_index": None,
                    "operation_count": len(example.operation_sequence),
                    "operation_family": "+".join(
                        OperationTemplate(example.metadata.operation_template).operations
                    ),
                    "authoritative_profile_family": example.metadata.primitive_family,
                    "reference_plane_category": plane_name,
                    "predicted_reference_plane": list(
                        predicted.normalized_geometry[:9]
                    ),
                    "authoritative_reference_plane": list(authoritative[:9]),
                    "pre_projection": pre,
                    "post_projection": post,
                    "authoritative_evaluation": authoritative_evaluation,
                    "arm_success": arm_success,
                    "arm_failure_code": arm_failures,
                }
                _require_finite_json(record)
                records.append(record)
    arm_summaries = {}
    if include_arms:
        for name in ("H0", "H1", "H2", "H3", "H5"):
            arm_summaries[name] = _reference_plane_arm_summary(name, arm_rows[name])
        arm_summaries["H4"] = {
            "arm": "H4", "enabled": False,
            "reason": ARM_CONTRACTS["H4"]["reason"],
        }
    return {
        "partition_summary": _partition_summary(records),
        "constraint_summary": _constraint_summary(records),
        "channel_audit": _merge_channel_audits(routing),
        "arm_summaries": arm_summaries,
        "example_records": records,
    }


def _build_h0(output, target, profiles, physical_examples):
    oracle_node_types = target["node_type_ids"]
    oracle_retained = target["categorical_attributes"][
        ..., REMAINING_CATEGORICAL_TARGET_INDICES
    ]
    family_logits = output.profile_family_logits.clone()
    selected = profiles.sketch_mask.reshape(-1)
    flat_logits = family_logits.reshape(-1, len(PROFILE_FAMILIES))
    family_ids = profiles.family_ids.reshape(-1)
    flat_logits[selected] = -1.0
    indexes = torch.nonzero(selected, as_tuple=False).reshape(-1)
    flat_logits[indexes, family_ids[indexes]] = 1.0
    records = construct_v2_predicted_node_tensors(
        oracle_node_types,
        oracle_retained,
        family_logits,
        output.raw_profile_parameters,
        output.non_profile_geometry,
    )
    return tuple(
        _replace_oracle_relations(
            _prediction_row(
                output, records, target["node_mask"], batch_index,
                "diagnostic_oracle_discrete_structure",
            ),
            example,
        )
        for batch_index, example in enumerate(physical_examples)
    )


def _replace_reference_plane_geometry(prediction, example):
    position = _one_plane_position(example)
    nodes = list(prediction.raw_nodes)
    current = nodes[position]
    geometry = list(current.normalized_geometry)
    geometry[:9] = example.target.geometry[position][:9]
    nodes[position] = replace(current, normalized_geometry=tuple(geometry))
    return replace(prediction, raw_nodes=tuple(nodes))


def _replace_all_non_profile_geometry(prediction, example):
    nodes = []
    for position, current in enumerate(prediction.raw_nodes):
        geometry = list(current.normalized_geometry)
        authoritative = example.target.geometry[position]
        for channel in NON_PROFILE_GEOMETRY_INDICES:
            geometry[channel] = authoritative[channel]
        nodes.append(replace(current, normalized_geometry=tuple(geometry)))
    return replace(prediction, raw_nodes=tuple(nodes))


def _project_prediction_reference_plane(prediction):
    nodes = []
    for current in prediction.raw_nodes:
        if current.node_type_id == NODE_TYPES.id("reference_plane"):
            geometry = list(current.normalized_geometry)
            geometry[:9] = project_reference_plane(geometry[:9])
            current = replace(current, normalized_geometry=tuple(geometry))
        nodes.append(current)
    return replace(prediction, raw_nodes=tuple(nodes))


def _replace_compact_profile_parameters(
    prediction, profiles, batch_index, node_count
):
    family_ids = profiles.family_ids[batch_index, :node_count]
    parameters = profiles.parameters[batch_index, :node_count]
    sketch_mask = profiles.sketch_mask[batch_index, :node_count]
    canonical = canonicalize_profile_tensors(
        family_ids, parameters, sketch_mask
    )
    nodes = []
    constrained = list(prediction.constrained_profile_parameters)
    for position, current in enumerate(prediction.raw_nodes):
        if bool(sketch_mask[position].item()):
            geometry = list(current.normalized_geometry)
            geometry[9:33] = canonical.geometry[position, 9:33].detach().cpu().tolist()
            current = replace(current, normalized_geometry=tuple(geometry))
            constrained[position] = tuple(
                float(value) for value in parameters[position].detach().cpu().tolist()
            )
        nodes.append(current)
    return replace(
        prediction,
        raw_nodes=tuple(nodes),
        constrained_profile_parameters=tuple(constrained),
    )


def _channel_routing_audit(output, target, h0, physical_examples):
    scattered = scatter_non_profile_geometry(output.non_profile_geometry)
    gathered = select_non_profile_geometry(scattered)
    if not torch.equal(gathered, output.non_profile_geometry):
        raise ReferencePlaneDiagnosticError(
            "channel_routing", "non-profile scatter/gather is not exact"
        )
    if not torch.equal(scattered, output.non_profile_geometry_scattered):
        raise ReferencePlaneDiagnosticError(
            "channel_routing", "model scattered tensor differs"
        )
    applicable = 0
    padding = 0
    denormalization_verified = True
    for batch_index, (prediction, example) in enumerate(zip(h0, physical_examples)):
        node_count = int(target["node_mask"][batch_index].sum().item())
        if node_count != len(prediction.raw_nodes) or node_count != len(example.nodes):
            raise ReferencePlaneDiagnosticError(
                "channel_routing", "teacher-forced node alignment differs"
            )
        padding += target["node_mask"].size(1) - node_count
        position = _one_plane_position(example)
        raw = prediction.raw_nodes[position]
        expected = tuple(
            float(value) for value in output.non_profile_geometry[
                batch_index, position, :9
            ].detach().cpu().tolist()
        )
        if raw.normalized_geometry[:9] != expected:
            raise ReferencePlaneDiagnosticError(
                "channel_routing", "H0 did not use V2 plane-head channels"
            )
        if raw.derived_geometry_mask != tuple(
            bool(value) for value in target["geometry_mask"][
                batch_index, position
            ].detach().cpu().tolist()
        ):
            raise ReferencePlaneDiagnosticError(
                "channel_routing", "H0 reference-plane mask is misaligned"
            )
        physical = denormalize_applicable_geometry(
            raw.normalized_geometry, raw.derived_geometry_mask
        )
        expected_physical = tuple(
            raw.normalized_geometry[index] * GEOMETRY_CHANNEL_SCALES[index]
            for index in REFERENCE_PLANE_INDICES
        )
        if physical[:9] != expected_physical:
            denormalization_verified = False
            raise ReferencePlaneDiagnosticError(
                "channel_routing", "reference-plane denormalization differs"
            )
        applicable += 1
    return {
        "batch_count": 1,
        "applicable_reference_plane_nodes": applicable,
        "padding_nodes_excluded": padding,
        "source_tensor": "ConstrainedProfileV2Output.non_profile_geometry",
        "source_tensor_width": int(output.non_profile_geometry.size(-1)),
        "source_offsets": list(range(9)),
        "serialized_indices": list(REFERENCE_PLANE_INDICES),
        "scatter_gather_exact": True,
        "model_scattered_tensor_exact": True,
        "teacher_forced_position_alignment": True,
        "mask_alignment": True,
        "denormalization_verified": denormalization_verified,
        "compact_profile_parameter_source_used": False,
        "v1_geometry_head_used": False,
    }


def _merge_channel_audits(items):
    return {
        "batch_count": sum(item["batch_count"] for item in items),
        "applicable_reference_plane_nodes": sum(
            item["applicable_reference_plane_nodes"] for item in items
        ),
        "padding_nodes_excluded": sum(item["padding_nodes_excluded"] for item in items),
        "source_tensor": "ConstrainedProfileV2Output.non_profile_geometry",
        "source_tensor_width": 15,
        "source_offsets": list(range(9)),
        "serialized_indices": list(REFERENCE_PLANE_INDICES),
        "scatter_gather_exact": all(item["scatter_gather_exact"] for item in items),
        "model_scattered_tensor_exact": all(item["model_scattered_tensor_exact"] for item in items),
        "teacher_forced_position_alignment": all(item["teacher_forced_position_alignment"] for item in items),
        "mask_alignment": all(item["mask_alignment"] for item in items),
        "denormalization_verified": all(
            item["denormalization_verified"] for item in items
        ),
        "compact_profile_parameter_source_used": False,
        "v1_geometry_head_used": False,
        "denormalization_scales": list(GEOMETRY_CHANNEL_SCALES[:9]),
    }


def _partition_summary(records):
    return {
        "example_count": len(records),
        "applicable_reference_plane_node_count": len(records),
        "predicted_scalar_distributions": {
            name: _distribution([record["predicted_reference_plane"][index] for record in records])
            for index, name in enumerate(REFERENCE_PLANE_CHANNELS)
        },
        "authoritative_scalar_distributions": {
            name: _distribution([record["authoritative_reference_plane"][index] for record in records])
            for index, name in enumerate(REFERENCE_PLANE_CHANNELS)
        },
        "predicted_metric_distributions": {
            name: _distribution([
                record["pre_projection"]["metrics"][name] for record in records
                if record["pre_projection"]["metrics"][name] is not None
            ])
            for name in (
                "origin_norm", "x_axis_norm", "y_axis_norm", "axis_dot",
                "orthogonality_error", "derived_normal_norm",
                "derived_basis_determinant", "x_axis_angular_error_degrees",
                "y_axis_angular_error_degrees", "canonical_frame_max_error",
            )
        },
        "authoritative_metric_distributions": {
            name: _distribution([
                record["authoritative_evaluation"]["metrics"][name]
                for record in records
                if record["authoritative_evaluation"]["metrics"][name] is not None
            ])
            for name in (
                "origin_norm", "x_axis_norm", "y_axis_norm", "axis_dot",
                "orthogonality_error", "derived_normal_norm",
                "derived_basis_determinant", "x_axis_angular_error_degrees",
                "y_axis_angular_error_degrees", "canonical_frame_max_error",
            )
        },
    }


def _constraint_summary(records):
    atomic = {}
    for name in ATOMIC_CONSTRAINT_ORDER:
        entries = [record["pre_projection"]["constraints"][name] for record in records]
        magnitudes = [entry["violation_magnitude"] for entry in entries
                      if entry["violation_magnitude"] is not None]
        atomic[name] = {
            "pass_count": sum(entry["passed"] for entry in entries),
            "fail_count": sum(not entry["passed"] for entry in entries),
            "maximum_violation_magnitude": max(magnitudes) if magnitudes else None,
            "mean_violation_magnitude": (
                sum(magnitudes) / float(len(magnitudes)) if magnitudes else None
            ),
        }
    first = {}
    for record in records:
        key = record["pre_projection"]["first_violated_predicate"] or "valid"
        first[key] = first.get(key, 0) + 1
    return {
        "atomic_constraints": atomic,
        "first_violated_predicate_histogram": first,
        "by_node_position": _constraint_group(records, "node_position"),
        "by_operation_count": _constraint_group(records, "operation_count"),
        "by_operation_family": _constraint_group(records, "operation_family"),
        "pre_projection_controlled_pass_count": sum(
            record["pre_projection"]["controlled_contract_pass"] for record in records
        ),
        "post_projection_mathematical_pass_count": sum(
            record["post_projection"]["mathematical_form_pass"] for record in records
        ),
        "post_projection_controlled_pass_count": sum(
            record["post_projection"]["controlled_contract_pass"] for record in records
        ),
    }


def _constraint_group(records, field):
    groups = {}
    for record in records:
        key = str(record[field])
        entry = groups.setdefault(key, {"count": 0, "mathematical_pass_count": 0,
                                        "controlled_pass_count": 0,
                                        "atomic_failure_counts": {},
                                        "first_violation_histogram": {}})
        entry["count"] += 1
        entry["mathematical_pass_count"] += int(
            record["pre_projection"]["mathematical_form_pass"]
        )
        entry["controlled_pass_count"] += int(
            record["pre_projection"]["controlled_contract_pass"]
        )
        for name in record["pre_projection"]["failed_constraints"]:
            failures = entry["atomic_failure_counts"]
            failures[name] = failures.get(name, 0) + 1
        first = record["pre_projection"]["first_violated_predicate"] or "valid"
        histogram = entry["first_violation_histogram"]
        histogram[first] = histogram.get(first, 0) + 1
    return groups


def _reference_plane_arm_summary(name, rows):
    total = len(rows)
    valid = sum(result.controlled_domain.valid for _, _, result in rows)
    failures = {}
    subtypes = {}
    groups = {
        "validity_by_node_count": ({}, {}),
        "validity_by_operation_count": ({}, {}),
        "validity_by_operation_family": ({}, {}),
        "validity_by_authoritative_profile_family": ({}, {}),
    }
    for example, raw, result in rows:
        if result.primary_failure is not None:
            code = result.primary_failure.code
            failures[code] = failures.get(code, 0) + 1
        position = _one_plane_position(example)
        evaluation = evaluate_reference_plane(
            raw.raw_nodes[position].normalized_geometry[:9],
            raw.raw_nodes[position].derived_geometry_mask,
            example.nodes[position].reference_plane,
        )
        subtype = evaluation["failure_subtype"]
        subtypes[subtype] = subtypes.get(subtype, 0) + 1
        template = OperationTemplate(example.metadata.operation_template)
        keys = (
            str(len(example.nodes)), str(len(example.operation_sequence)),
            "+".join(template.operations), example.metadata.primitive_family,
        )
        for (totals, successes), key in zip(groups.values(), keys):
            totals[key] = totals.get(key, 0) + 1
            if result.controlled_domain.valid:
                successes[key] = successes.get(key, 0) + 1
    summary = {
        "arm": name,
        "enabled": True,
        "oracle_intervention": True,
        "total_count": total,
        "valid_count": valid,
        "validity": valid / float(total),
        "finite_rate": sum(_raw_prediction_is_finite(raw) for _, raw, _ in rows) / float(total),
        "canonical_profile_rate": sum(_raw_profile_records_are_canonical(raw) for _, raw, _ in rows) / float(total),
        "conversion_success_rate": sum(result.reconstruction_target.valid for _, _, result in rows) / float(total),
        "failure_reason_histogram": failures,
        "failure_subtype_histogram": subtypes,
    }
    for group_name, (totals, successes) in groups.items():
        summary[group_name] = {
            key: {"count": totals[key], "valid_count": successes.get(key, 0),
                  "validity": successes.get(key, 0) / float(totals[key])}
            for key in sorted(totals)
        }
    _require_finite_json(summary)
    return summary


def _one_plane_position(example):
    positions = [
        index for index, node in enumerate(example.nodes)
        if node.node_type == "reference_plane"
    ]
    if len(positions) != 1:
        raise ReferencePlaneDiagnosticError(
            "reference_plane_count", "expected exactly one reference plane"
        )
    return positions[0]


def _predicate(passed, violation):
    return {"passed": bool(passed), "violation_magnitude": violation}


def _norm(values):
    return math.sqrt(sum(value * value for value in values))


def _dot(left, right):
    return sum(a * b for a, b in zip(left, right))


def _cross3(left, right):
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _scale(values, scale):
    return tuple(value * scale for value in values)


def _subtract(left, right):
    return tuple(a - b for a, b in zip(left, right))


def _angular_error(predicted, expected):
    norm = _norm(predicted)
    if norm <= DEGENERATE_VECTOR_EPSILON:
        return None
    cosine = max(-1.0, min(1.0, _dot(predicted, expected) / norm))
    return math.degrees(math.acos(cosine))


def _distribution(values):
    finite = [float(value) for value in values]
    if not finite:
        return {"count": 0, "min": 0.0, "mean": 0.0, "max": 0.0, "std": 0.0}
    mean = sum(finite) / float(len(finite))
    return {
        "count": len(finite), "min": min(finite), "mean": mean,
        "max": max(finite),
        "std": math.sqrt(sum((value - mean) ** 2 for value in finite) / len(finite)),
    }


def _classification_rules():
    return {
        "material_validity_improvement": MATERIAL_VALIDITY_IMPROVEMENT,
        "high_validity": HIGH_VALIDITY,
        "h1_immediate_blocker": "H0 == 0 and H1 >= high_validity",
        "h2_additional_geometry": "H2 - H1 >= material improvement",
        "h3_projection_viable": "H1 high and H3 within material improvement of H1",
        "h3_accuracy_problem": "H1 high and H3 materially below H1",
        "h2_profile_support": "H2 >= high_validity",
        "reaudit": "H1 == 0 and H2 == 0",
    }


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    config = replace(
        ReferencePlaneDiagnosticConfig(),
        output_dir=args.output_dir,
        device=args.device,
    )
    try:
        result = run_frozen_reference_plane_diagnostic(
            args.corpus_dir, args.checkpoint, config
        )
    except Exception as exc:
        _append_terminal_failure(config, exc)
        print(json.dumps({
            "event": "terminal_failure",
            "error_code": getattr(exc, "code", "diagnostic_failure"),
            "error_type": type(exc).__name__,
            "detail": str(exc),
            "training_occurred": False,
            "systematic_partition_accessed": False,
            "test_partition_accessed": False,
        }, sort_keys=True, separators=(",", ":"), allow_nan=False), file=sys.stderr)
        return 2
    print(json.dumps({
        "event": "terminal_success",
        "output_dir": result.output_dir,
        "metrics_path": result.metrics_path,
        "examples_path": result.examples_path,
        "classifications": list(result.classifications),
        "checkpoint_sha256": result.checkpoint_sha256,
        "training_occurred": False,
        "systematic_partition_accessed": False,
        "test_partition_accessed": False,
    }, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0


def _append_terminal_failure(config, exc):
    if getattr(exc, "code", None) == "output_collision":
        return
    path = Path(config.output_dir) / "metrics.jsonl"
    if not path.is_file():
        return
    try:
        records = path.read_text(encoding="utf-8").splitlines()
        first = json.loads(records[0])
        last = json.loads(records[-1])
        if (
            first.get("event") != "diagnostic_metadata"
            or first.get("diagnostic_config", {}).get("diagnostic_identity")
            != config.diagnostic_identity
            or last.get("event") in ("terminal_success", "terminal_failure")
        ):
            return
    except (OSError, ValueError, IndexError, TypeError):
        return
    JsonlLogger(path).write({
        "event": "terminal_failure",
        "error_code": getattr(exc, "code", "diagnostic_failure"),
        "error_type": type(exc).__name__,
        "detail": str(exc),
        "training_occurred": False,
        "systematic_partition_accessed": False,
        "test_partition_accessed": False,
    })


if __name__ == "__main__":
    raise SystemExit(main())
