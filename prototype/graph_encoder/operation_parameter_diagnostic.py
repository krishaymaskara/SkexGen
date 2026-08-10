"""Read-only C7-v2 operation-parameter diagnostic.

This additive diagnostic reuses the immutable scaled epoch-200 checkpoints
from authoritative C7-v2 job 3344981.  It performs target-free P_true
inference only after verifying the source artifact and checkpoint identities,
reproduces the finalized C7-v2 metrics, and publishes only minimal scalar and
categorical traces for the five operation-parameter failures.

The controlled-domain validity check is analytic.  No CAD kernel is part of
the repository contract, so executor results and legal-but-geometrically-
incompatible classifications are explicitly unavailable.
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
import socket
import sys
import time

try:
    import torch
except ImportError:  # Pure contract tests remain available without PyTorch.
    torch = None

from prototype.model_data.geometry import (
    GEOMETRY_CHANNELS,
    GEOMETRY_CHANNEL_SCALES,
    LENGTH_SCALE,
)
from prototype.model_data.vocab import BOOLEAN_MODES, DIRECTIONS, EDGE_TYPES, NODE_TYPES
from prototype.profile_geometry import extract_profile_parameters

from .c7_v2 import (
    C7_V2_ARMS,
    C7_V2_BATCH_SIZE,
    C7_V2_CHECKPOINT_EPOCH,
    C7_V2_SCALED_FAMILY_IDS_SHA256,
    C7_V2_SEED,
    validate_frozen_c7_v2_selection,
    verify_c7_v2_artifact,
)
from .errors import GraphEncoderError
from .optimization_diagnostic import validate_slurm_job_id
from .partitions import (
    AUTHORITATIVE_FILE_SHA256,
    load_train,
    select_c7_sufficiency_subsets,
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


DIAGNOSTIC_PROTOCOL_VERSION = (
    "GE1-C7-V2-OPERATION-PARAMETER-DIAGNOSTIC-v1"
)
DIAGNOSTIC_ARTIFACT_VERSION = (
    "GE1-C7-V2-OPERATION-PARAMETER-ARTIFACT-v1"
)
DIAGNOSTIC_RECORD_VERSION = (
    "GE1-C7-V2-OPERATION-PARAMETER-RECORD-v1"
)
SOURCE_C7_V2_COMMIT = "e325d5ad97957c08da4a19b4261560e8a4a472a4"
SOURCE_C7_V2_JOB_ID = "3344981"
SOURCE_C7_V2_METRICS_SHA256 = (
    "fe151f45df045dd18182053a42e304600b7c272f9d072339707fdc00dc24e2fd"
)
SOURCE_C7_V2_ARTIFACT_MANIFEST_SHA256 = (
    "9c526c89dda8fb72bfafc0c922c73fae6461b0f5728eaaeaeea31d0335994d37"
)
SOURCE_CHECKPOINTS = {
    "flat": {
        "relative_path": (
            "checkpoints/scaled/flat/flat-seed2026-epoch0200.pt"
        ),
        "sha256": (
            "ff628c7e1c5391869237658a00941efbe2481cb6c272d7c9e56affad2a99e992"
        ),
    },
    "typed_graph": {
        "relative_path": (
            "checkpoints/scaled/typed_graph/"
            "typed_graph-seed2026-epoch0200.pt"
        ),
        "sha256": (
            "7535c8bd4e01ac586beff431a228bc2c244b7ff5d97faf745b5d2125d35e0cd2"
        ),
    },
}
FAILED_FAMILIES_BY_ARM = {
    "flat": (
        "sf_140d5984ae7b954a9b0232bc3f1d1e0bf24b73971e54ccce6dddfd253ff522e6",
        "sf_93d4027a05325354f073f6ebe3375e1d2fe52a1fe1fc6c914fdcaef98e7d169c",
    ),
    "typed_graph": (
        "sf_69cc286ec6c7d7993635edb269176860e99dbaf9877d467a6e0aa31afdd44378",
        "sf_caab9609073e3323d6d748a4d4a05174bd7978bb938f86e69d28441dcc6dc24c",
        "sf_fe25554783fbed5e372dbd12b4db1ee6a590f9eb90ef5f674314b161ad4303c3",
    ),
}
FAILED_FAMILY_IDS = tuple(sorted(
    family_id
    for values in FAILED_FAMILIES_BY_ARM.values()
    for family_id in values
))
EXPECTED_TEMPLATES = {
    FAILED_FAMILIES_BY_ARM["flat"][0]: "RE",
    FAILED_FAMILIES_BY_ARM["flat"][1]: "RE",
    FAILED_FAMILIES_BY_ARM["typed_graph"][0]: "EE",
    FAILED_FAMILIES_BY_ARM["typed_graph"][1]: "E",
    FAILED_FAMILIES_BY_ARM["typed_graph"][2]: "EE",
}
PROTECTED_ACCESS_FIELDS = (
    "development_accessed",
    "systematic_rr_accessed",
    "test_er_accessed",
    "iid_accessed",
    "history_depth_accessed",
    "geometry_extrapolation_accessed",
    "other_corpus_accessed",
)
FORBIDDEN_DIAGNOSTIC_KEYS = frozenset({
    "cad_history",
    "canonical_reconstruction_json",
    "directed_typed_edges",
    "edge_index",
    "edges",
    "graph",
    "history",
    "model_state",
    "node_type_ids",
    "nodes",
    "optimizer_state",
    "raw_cad_history",
    "sample_payload",
})
EXECUTOR_UNAVAILABLE = {
    "available": False,
    "result": None,
    "reason": "no_cad_kernel_in_evaluation_contract",
}
GEOMETRIC_INCOMPATIBILITY_UNASSESSABLE = {
    "assessable": False,
    "classification": None,
    "reason": "no_cad_kernel_in_evaluation_contract",
}


@dataclass
class DiagnosticAccessTracker:
    operation_template_manifest_accessed: object = False
    operation_template_train_payload_accessed: object = False
    scaled_train_payload_accessed: object = False

    def declarations(self, *, completed=False):
        result = {
            "operation_template_manifest_accessed": (
                self.operation_template_manifest_accessed
            ),
            "operation_template_train_payload_accessed": (
                self.operation_template_train_payload_accessed
            ),
            "scaled_train_payload_accessed": self.scaled_train_payload_accessed,
            "source_c7_v2_artifact_read_only": True,
            "source_checkpoints_modified": False,
            "optimization_performed": False,
            "backward_pass_performed": False,
            "training_performed": False,
            "checkpoint_written": False,
            "model_parameters_modified_after_reload": False,
            "decoder_repair_implemented_or_invoked": False,
            "geometry_repair_implemented_or_invoked": False,
            "stage6_performed": False,
            "c8_or_later_performed": False,
            "diagnostic_completed": bool(completed),
        }
        result.update({name: False for name in PROTECTED_ACCESS_FIELDS})
        return result


def corrected_validity_contract():
    """Return the reviewer-authorized semantic correction."""

    return {
        "strict_conversion": (
            "reconstruction_target.valid from the existing strict converter"
        ),
        "analytic_controlled_domain_validity": (
            "controlled_domain.valid from the existing analytic validator"
        ),
        "invalid_operation_parameter_semantics": (
            "active denormalized extrude_distance or revolve_angle is <= 0.0"
        ),
        "executor_result": dict(EXECUTOR_UNAVAILABLE),
        "legal_parameter_geometric_incompatibility": dict(
            GEOMETRIC_INCOMPATIBILITY_UNASSESSABLE
        ),
    }


def classify_operation_parameter(normalized_value, channel_index, mask_active):
    """Classify one scalar under the frozen analytic parameter contract."""

    if channel_index not in (37, 38):
        raise GraphEncoderError(
            "invalid_diagnostic_channel",
            "operation parameter channel must be 37 or 38",
        )
    if not isinstance(mask_active, bool):
        raise GraphEncoderError(
            "invalid_diagnostic_mask", "geometry-mask evidence must be Boolean"
        )
    if isinstance(normalized_value, bool):
        raise GraphEncoderError(
            "invalid_diagnostic_value", "operation parameter must be numeric"
        )
    try:
        normalized = float(normalized_value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise GraphEncoderError(
            "invalid_diagnostic_value", "operation parameter must be numeric"
        ) from exc
    if not math.isfinite(normalized):
        raise GraphEncoderError(
            "invalid_diagnostic_value", "operation parameter must be finite"
        )
    scale = float(GEOMETRY_CHANNEL_SCALES[channel_index])
    physical = normalized * scale
    if not mask_active:
        classification = "inactive_channel"
        analytic_legal = False
    elif physical < 0.0:
        classification = "negative_value"
        analytic_legal = False
    elif physical == 0.0:
        classification = "zero_boundary_value"
        analytic_legal = False
    else:
        classification = "positive_analytic_legal_value"
        analytic_legal = True
    return {
        "classification": classification,
        "analytic_legal": analytic_legal,
        "normalized_value": 0.0 if normalized == 0.0 else normalized,
        "normalized_within_declared_network_range": -1.0 <= normalized <= 1.0,
        "physical_value": 0.0 if physical == 0.0 else physical,
        "physical_boundary_margin_above_zero": (
            0.0 if physical == 0.0 else physical
        ),
        "normalized_contract_min_inclusive": -1.0,
        "normalized_contract_max_inclusive": 1.0,
        "physical_legal_min_exclusive": 0.0,
        "physical_legal_max_inclusive": scale,
    }


def validate_checkpoint_identity_fields(
    payload,
    *,
    expected_arm,
    expected_commit=SOURCE_C7_V2_COMMIT,
    expected_job_id=SOURCE_C7_V2_JOB_ID,
):
    """Pure identity gate used before any model state is loaded."""

    if expected_arm not in C7_V2_ARMS:
        raise GraphEncoderError(
            "invalid_diagnostic_arm", "checkpoint arm is not a GE1 arm"
        )
    if not isinstance(payload, dict):
        raise GraphEncoderError(
            "invalid_diagnostic_checkpoint", "checkpoint payload must be a dict"
        )
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        raise GraphEncoderError(
            "invalid_diagnostic_checkpoint", "checkpoint provenance is absent"
        )
    expected = {
        "encoder_type": expected_arm,
        "completed_epoch": C7_V2_CHECKPOINT_EPOCH,
        "selected_checkpoint_epoch": C7_V2_CHECKPOINT_EPOCH,
        "selected_experimental_checkpoint": True,
        "seed": C7_V2_SEED,
    }
    for name, value in expected.items():
        if payload.get(name) != value:
            raise GraphEncoderError(
                "invalid_diagnostic_checkpoint",
                "checkpoint field {} differs".format(name),
            )
    if (
        provenance.get("git_commit") != expected_commit
        or provenance.get("slurm_job_id") != expected_job_id
        or provenance.get("encoder_arm") != expected_arm
        or provenance.get("device") != "cpu"
        or provenance.get("python_version") != "3.8.13"
        or str(provenance.get("pytorch_version", "")).split("+")[0]
        != "1.11.0"
    ):
        raise GraphEncoderError(
            "invalid_diagnostic_checkpoint",
            "checkpoint commit, job, arm, or runtime provenance differs",
        )
    if payload.get("source_digest") != provenance.get("source_tree_sha256"):
        raise GraphEncoderError(
            "invalid_diagnostic_checkpoint",
            "checkpoint source digest and provenance differ",
        )
    return payload


def validate_checkpoint_file_hash(path, expected_sha256):
    """Require a regular immutable checkpoint with its frozen SHA-256."""

    checkpoint = Path(path)
    if (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha256)
    ):
        raise GraphEncoderError(
            "invalid_diagnostic_checkpoint", "expected SHA-256 is malformed"
        )
    if not checkpoint.is_file() or checkpoint.is_symlink():
        raise GraphEncoderError(
            "invalid_diagnostic_checkpoint", "checkpoint is not a regular file"
        )
    observed = _file_sha256(checkpoint)
    if observed != expected_sha256:
        raise GraphEncoderError(
            "diagnostic_checkpoint_hash_mismatch", "checkpoint SHA-256 differs"
        )
    return observed


def stable_condition_metrics(metrics):
    """Remove only nondeterministic wall-clock evidence from C7 metrics."""

    if not isinstance(metrics, dict):
        raise GraphEncoderError(
            "invalid_source_c7_metrics", "condition metrics must be a dict"
        )
    stable = copy.deepcopy(metrics)
    stable.pop("metric_computation_seconds", None)
    return stable


def stable_metrics_sha256(metrics):
    return hashlib.sha256(
        _canonical_json_text(stable_condition_metrics(metrics)).encode("utf-8")
    ).hexdigest()


def validate_reproduced_c7_metrics(
    original_metrics,
    reproduced_metrics,
    *,
    arm,
    templates_by_family,
):
    """Require exact stable P_true reproduction before diagnosis."""

    original = stable_condition_metrics(original_metrics)
    reproduced = stable_condition_metrics(reproduced_metrics)
    if reproduced != original:
        raise GraphEncoderError(
            "c7_v2_metric_reproduction_failure",
            "{} stable P_true metrics differ from job 3344981".format(arm),
        )
    values = reproduced.get("family_values")
    if not isinstance(values, dict) or set(values) != set(templates_by_family):
        raise GraphEncoderError(
            "c7_v2_metric_reproduction_failure",
            "reproduced physical-family membership differs",
        )
    observed_failed = tuple(sorted(
        family_id
        for family_id, value in values.items()
        if value.get("complete_executable_validity") != 1.0
    ))
    if observed_failed != FAILED_FAMILIES_BY_ARM[arm]:
        raise GraphEncoderError(
            "c7_v2_metric_reproduction_failure",
            "{} reproduced failure set differs".format(arm),
        )
    for family_id, row in values.items():
        template = templates_by_family[family_id]
        required = (
            row.get("exact_node_sequence") == 1.0
            and row.get("exact_graph") == 1.0
            and row.get("strict_conversion") == 1.0
            and row.get("depends_on_exactness") == 1.0
        )
        if not required:
            raise GraphEncoderError(
                "c7_v2_metric_reproduction_failure",
                "{} structural or conversion reproduction differs".format(
                    family_id
                ),
            )
        if template not in ("E", "R", "EE", "RE"):
            raise GraphEncoderError(
                "c7_v2_metric_reproduction_failure",
                "reproduced template is outside the scaled cohort",
            )
    return {
        "arm": arm,
        "status": "exact_stable_reproduction_passed",
        "stable_metrics_sha256": stable_metrics_sha256(reproduced),
        "physical_family_count": len(values),
        "failed_family_ids": list(observed_failed),
        "exact_node_sequence_count": sum(
            int(item["exact_node_sequence"] == 1.0) for item in values.values()
        ),
        "exact_graph_count": sum(
            int(item["exact_graph"] == 1.0) for item in values.values()
        ),
        "strict_conversion_count": sum(
            int(item["strict_conversion"] == 1.0) for item in values.values()
        ),
        "complete_executable_validity_count": sum(
            int(item["complete_executable_validity"] == 1.0)
            for item in values.values()
        ),
    }


def validate_minimal_diagnostic_record(record):
    """Reject raw histories, model state, or incomplete arm/family coverage."""

    if not isinstance(record, dict):
        raise GraphEncoderError(
            "invalid_diagnostic_record", "diagnostic record must be a dict"
        )
    if record.get("schema_version") != DIAGNOSTIC_RECORD_VERSION:
        raise GraphEncoderError(
            "invalid_diagnostic_record", "diagnostic record identity differs"
        )

    def walk(value):
        if isinstance(value, dict):
            forbidden = FORBIDDEN_DIAGNOSTIC_KEYS & set(value)
            if forbidden:
                raise GraphEncoderError(
                    "raw_payload_disclosure_forbidden",
                    "diagnostic record contains forbidden keys {}".format(
                        sorted(forbidden)
                    ),
                )
            for nested in value.values():
                walk(nested)
        elif isinstance(value, list):
            for nested in value:
                walk(nested)

    walk(record)
    families = record.get("family_diagnostics")
    if not isinstance(families, list):
        raise GraphEncoderError(
            "invalid_diagnostic_record", "family diagnostics are absent"
        )
    if tuple(sorted(item.get("family_id") for item in families)) != FAILED_FAMILY_IDS:
        raise GraphEncoderError(
            "invalid_diagnostic_record", "failed-family coverage differs"
        )
    for family in families:
        arms = family.get("arm_results")
        if not isinstance(arms, list) or tuple(
            sorted(item.get("arm") for item in arms)
        ) != C7_V2_ARMS:
            raise GraphEncoderError(
                "invalid_diagnostic_record",
                "both arms must be recorded for every failed family",
            )
        for arm in arms:
            executor = arm.get("executor_result")
            geometric = arm.get("legal_parameter_geometric_incompatibility")
            if executor != EXECUTOR_UNAVAILABLE or geometric != (
                GEOMETRIC_INCOMPATIBILITY_UNASSESSABLE
            ):
                raise GraphEncoderError(
                    "invalid_diagnostic_record",
                    "corrected no-kernel contract is missing",
                )
    return record


def _require_authoritative_runtime():
    if torch is None:
        raise RuntimeError("diagnostic execution requires PyTorch")
    job_id = validate_slurm_job_id()
    if platform.python_version() != "3.8.13":
        raise GraphEncoderError(
            "invalid_diagnostic_runtime", "Python 3.8.13 is required"
        )
    if str(torch.__version__).split("+")[0] != "1.11.0":
        raise GraphEncoderError(
            "invalid_diagnostic_runtime", "PyTorch 1.11.0 is required"
        )
    if torch.cuda.is_available():
        raise GraphEncoderError(
            "invalid_diagnostic_runtime", "diagnostic must be CPU-only"
        )
    if torch.get_num_threads() != 1:
        raise GraphEncoderError(
            "invalid_diagnostic_runtime", "one CPU thread is required"
        )
    return job_id


def _state_sha256(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(tensor.dtype).encode("ascii") + b"\0")
        digest.update(json.dumps(list(tensor.shape)).encode("ascii") + b"\0")
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _load_read_only_model_checkpoint(
    path,
    *,
    arm,
    expected_sha256,
    partition_identity,
):
    """Strictly reconstruct a fresh inference model without an optimizer."""

    if torch is None:
        raise RuntimeError("checkpoint loading requires PyTorch")
    from .config import (
        CHECKPOINT_SCHEMA,
        GE1TrainingConfig,
        MODEL_FAMILY,
        TRAINING_OPTIMIZER,
        legacy_frozen_encoder_config,
    )
    from .model import build_ge1_model
    from .provenance import sha256_json
    from .training import TRAINING_CHECKPOINT_FIELDS, TRAINING_STATE_SCHEMA

    checkpoint = Path(path)
    validate_checkpoint_file_hash(checkpoint, expected_sha256)
    payload = torch.load(str(checkpoint), map_location="cpu")
    validate_checkpoint_identity_fields(payload, expected_arm=arm)
    if set(payload) != set(TRAINING_CHECKPOINT_FIELDS):
        raise GraphEncoderError(
            "invalid_diagnostic_checkpoint", "checkpoint field set differs"
        )
    config = legacy_frozen_encoder_config(arm, C7_V2_SEED)
    historical_model_config = config.to_dict()
    historical_model_config.pop("operation_magnitude_parameterization")
    training_config = GE1TrainingConfig()
    training_config.validate()
    expected = {
        "training_state_schema": TRAINING_STATE_SCHEMA,
        "checkpoint_schema": CHECKPOINT_SCHEMA,
        "model_family": MODEL_FAMILY,
        "arm_identity": config.arm_identity,
        "model_config": historical_model_config,
        "model_config_sha256": sha256_json(historical_model_config),
        "training_config": training_config.to_dict(),
        "training_config_sha256": sha256_json(training_config.to_dict()),
        "optimizer_name": TRAINING_OPTIMIZER,
        "optimizer_step_count": 800,
        "training_example_presentations": 6400,
        "partition_identity": partition_identity,
        "partition_identity_sha256": sha256_json(partition_identity),
        "run_identity": "ge1-{}-seed2026".format(arm),
    }
    for name, value in expected.items():
        if payload.get(name) != value:
            raise GraphEncoderError(
                "invalid_diagnostic_checkpoint",
                "checkpoint metadata field {} differs".format(name),
            )
    if not isinstance(payload.get("optimizer_state"), dict):
        raise GraphEncoderError(
            "invalid_diagnostic_checkpoint", "stored optimizer evidence is malformed"
        )
    model = build_ge1_model(config)
    state = payload.get("model_state")
    if not isinstance(state, dict):
        raise GraphEncoderError(
            "invalid_diagnostic_checkpoint", "model state is malformed"
        )
    try:
        model.load_state_dict(state, strict=True)
    except Exception as exc:
        raise GraphEncoderError(
            "invalid_diagnostic_checkpoint", "strict model-state load failed"
        ) from exc
    model.eval()
    return model, payload, _state_sha256(model)


def _source_c7_evidence(source_artifact):
    root = Path(source_artifact)
    verification = verify_c7_v2_artifact(
        root,
        expected_commit=SOURCE_C7_V2_COMMIT,
        expected_slurm_job_id=SOURCE_C7_V2_JOB_ID,
    )
    if _file_sha256(root / "metrics.jsonl") != SOURCE_C7_V2_METRICS_SHA256:
        raise GraphEncoderError(
            "source_c7_v2_artifact_mismatch", "source metrics SHA-256 differs"
        )
    if (
        _file_sha256(root / "artifact_manifest.json")
        != SOURCE_C7_V2_ARTIFACT_MANIFEST_SHA256
    ):
        raise GraphEncoderError(
            "source_c7_v2_artifact_mismatch", "source manifest SHA-256 differs"
        )
    events = _read_canonical_jsonl(root / "metrics.jsonl")
    originals = {}
    reloads = {}
    for event in events:
        if (
            event.get("event") == "c7_v2_autonomous_condition_metrics"
            and event.get("subset_identity") == "scaled"
            and event.get("condition") == "P_true"
        ):
            originals[event["arm"]] = event
        if (
            event.get("event") == "c7_v2_checkpoint_reloaded"
            and event.get("subset_identity") == "scaled"
        ):
            reloads[event["arm"]] = event
    if set(originals) != set(C7_V2_ARMS) or set(reloads) != set(C7_V2_ARMS):
        raise GraphEncoderError(
            "source_c7_v2_artifact_mismatch", "scaled C7-v2 evidence is incomplete"
        )
    for arm in C7_V2_ARMS:
        frozen = SOURCE_CHECKPOINTS[arm]
        reload = reloads[arm]
        if (
            reload.get("checkpoint_identity") != frozen["relative_path"]
            or reload.get("checkpoint_sha256") != frozen["sha256"]
            or reload.get("epoch") != C7_V2_CHECKPOINT_EPOCH
            or reload.get("strict_reload") is not True
        ):
            raise GraphEncoderError(
                "source_c7_v2_artifact_mismatch",
                "{} checkpoint event differs".format(arm),
            )
    return verification, originals, reloads


def _profile_extent_for_operation(target, operation_node_index):
    uses_profile = EDGE_TYPES.id("uses_profile")
    defined_in = EDGE_TYPES.id("defined_in")
    edges = tuple(zip(target.edge_index[0], target.edge_index[1], target.edge_type_ids))
    profiles = [
        destination
        for source, destination, edge_type in edges
        if source == operation_node_index and edge_type == uses_profile
    ]
    if len(profiles) != 1:
        raise GraphEncoderError(
            "diagnostic_target_alignment_failure",
            "operation must use exactly one profile",
        )
    sketches = [
        destination
        for source, destination, edge_type in edges
        if source == profiles[0] and edge_type == defined_in
    ]
    if len(sketches) != 1:
        raise GraphEncoderError(
            "diagnostic_target_alignment_failure",
            "profile must be defined in exactly one sketch",
        )
    parameters = extract_profile_parameters(target, sketches[0])
    return float(parameters.extent) * LENGTH_SCALE


def _failure_record(failure):
    if failure is None:
        return None
    return {
        "code": failure.code,
        "layer": failure.layer,
        "stage": failure.stage,
        "location": failure.location,
        "detail": failure.detail,
    }


def _trace_arm_family(model, prediction, example, arm):
    from .metrics import score_prediction_prefix

    constrained = prediction.constrained_prediction
    converted_outcome = prediction.converted_prediction
    if converted_outcome.raised_failure:
        raise GraphEncoderError(
            "diagnostic_reproduction_failure",
            "full strict conversion unexpectedly raised",
        )
    conversion = converted_outcome.result
    prefix = score_prediction_prefix(
        constrained, len(example.target.operation_sequence)
    )
    prefix_output = prediction.raw_prediction.prefix_output
    with torch.no_grad():
        pre_tanh = model.decoder.remaining_geometry_head(
            prefix_output.decoded_states
        ).detach().cpu()
    operation_records = []
    for operation_index, node_index in enumerate(example.target.operation_sequence):
        node = constrained.node_prediction.raw_nodes[node_index]
        operation_type = NODE_TYPES.tokens[node.node_type_id]
        if operation_type not in ("extrude", "revolve"):
            raise GraphEncoderError(
                "diagnostic_target_alignment_failure",
                "operation sequence contains a non-operation node",
            )
        channel = 37 if operation_type == "extrude" else 38
        compact_channel = channel - 33
        predicted_normalized = float(node.normalized_geometry[channel])
        target_normalized = float(example.target.geometry[node_index][channel])
        predicted = classify_operation_parameter(
            predicted_normalized,
            channel,
            bool(node.derived_geometry_mask[channel]),
        )
        target = classify_operation_parameter(
            target_normalized,
            channel,
            bool(example.target.geometry_mask[node_index][channel]),
        )
        predicted_physical = predicted["physical_value"]
        target_physical = target["physical_value"]
        scale = float(GEOMETRY_CHANNEL_SCALES[channel])
        categorical = node.categorical_ids
        target_categorical = example.target.categorical_attributes[node_index]
        record = {
            "operation_index": operation_index,
            "operation_node_index": int(node_index),
            "operation_type": operation_type,
            "geometry_channel_index": channel,
            "geometry_channel_name": GEOMETRY_CHANNELS[channel],
            "raw_head_output_pre_tanh": float(
                pre_tanh[0, node_index, compact_channel].item()
            ),
            "normalized_predicted_value_post_tanh": predicted_normalized,
            "geometry_mask_active": bool(node.derived_geometry_mask[channel]),
            "target_geometry_mask_active": bool(
                example.target.geometry_mask[node_index][channel]
            ),
            "denormalization": {
                "formula": "physical = normalized * scale",
                "scale": scale,
                "offset": 0.0,
                "physical_units": (
                    "controlled_length_unit"
                    if operation_type == "extrude" else "degrees"
                ),
            },
            "physical_predicted_value": predicted_physical,
            "normalized_target_value": target_normalized,
            "physical_target_value": target_physical,
            "signed_physical_error": predicted_physical - target_physical,
            "absolute_physical_error": abs(
                predicted_physical - target_physical
            ),
            "analytic_parameter_contract": predicted,
            "target_parameter_contract": target,
            "predicted_direction": DIRECTIONS.tokens[categorical[2]],
            "target_direction": DIRECTIONS.tokens[target_categorical[2]],
            "predicted_boolean_mode": BOOLEAN_MODES.tokens[categorical[1]],
            "target_boolean_mode": BOOLEAN_MODES.tokens[target_categorical[1]],
            "separate_extent_categorical": {
                "available": False,
                "value": None,
                "reason": "schema_uses_parameter_magnitude_and_direction_only",
            },
            "preceding_profile_physical_extent": _profile_extent_for_operation(
                example.target, node_index
            ),
            "normalization_consistency": (
                abs(predicted_physical - predicted_normalized * scale) == 0.0
            ),
            "channel_selection_consistency": (
                (operation_type == "extrude" and channel == 37)
                or (operation_type == "revolve" and channel == 38)
            ),
            "direction_boolean_interaction_evaluated_by_parameter_check": False,
        }
        operation_records.append(record)
    result = {
        "arm": arm,
        "strict_conversion_succeeded": bool(
            conversion.reconstruction_target.valid
        ),
        "analytic_controlled_domain_validity": bool(
            conversion.controlled_domain.valid
        ),
        "primary_failure": _failure_record(conversion.primary_failure),
        "secondary_failures": [
            _failure_record(item) for item in conversion.secondary_failures
        ],
        "longest_executable_operation_prefix": (
            prefix.unnormalized_longest_executable_prefix
        ),
        "normalized_longest_executable_operation_prefix": (
            prefix.normalized_longest_executable_operation_prefix
        ),
        "failed_operation_index": (
            None
            if prefix.unnormalized_longest_executable_prefix
            == len(example.target.operation_sequence)
            else prefix.unnormalized_longest_executable_prefix
        ),
        "first_failed_stage": prefix.first_failed_stage,
        "first_failure_code": prefix.first_failure_code,
        "executor_result": dict(EXECUTOR_UNAVAILABLE),
        "legal_parameter_geometric_incompatibility": dict(
            GEOMETRIC_INCOMPATIBILITY_UNASSESSABLE
        ),
        "operations": operation_records,
    }
    return result


def _attach_controls(family_diagnostics):
    for family in family_diagnostics:
        by_arm = {item["arm"]: item for item in family["arm_results"]}
        for arm, result in by_arm.items():
            other = "typed_graph" if arm == "flat" else "flat"
            control_operations = by_arm[other]["operations"]
            if len(control_operations) != len(result["operations"]):
                raise GraphEncoderError(
                    "diagnostic_control_alignment_failure",
                    "arm operation counts differ",
                )
            for operation, control in zip(result["operations"], control_operations):
                if operation["operation_index"] != control["operation_index"]:
                    raise GraphEncoderError(
                        "diagnostic_control_alignment_failure",
                        "arm operation indices differ",
                    )
                operation["other_arm_control"] = {
                    "arm": other,
                    "normalized_predicted_value_post_tanh": control[
                        "normalized_predicted_value_post_tanh"
                    ],
                    "physical_predicted_value": control[
                        "physical_predicted_value"
                    ],
                    "analytic_parameter_classification": control[
                        "analytic_parameter_contract"
                    ]["classification"],
                    "analytic_controlled_domain_validity": by_arm[other][
                        "analytic_controlled_domain_validity"
                    ],
                }


def _classify_failed_family(family):
    failing_arm = family["failing_arm"]
    result = next(
        item for item in family["arm_results"] if item["arm"] == failing_arm
    )
    failed_index = result["failed_operation_index"]
    nonpositive = [
        item
        for item in result["operations"]
        if not item["analytic_parameter_contract"]["analytic_legal"]
    ]
    if not nonpositive or result["first_failure_code"] != "invalid_operation_parameter":
        raise GraphEncoderError(
            "diagnostic_classification_failure",
            "failed family lacks nonpositive operation-parameter evidence",
        )
    out_of_range = any(
        not item["analytic_parameter_contract"][
            "normalized_within_declared_network_range"
        ]
        for item in result["operations"]
    )
    normalization_mismatch = any(
        not item["normalization_consistency"] for item in result["operations"]
    )
    mask_or_channel_mismatch = any(
        not item["geometry_mask_active"]
        or not item["target_geometry_mask_active"]
        or not item["channel_selection_consistency"]
        for item in result["operations"]
    )
    hypotheses = {
        "unconstrained_or_out_of_network_range_prediction": {
            "status": "supported" if out_of_range else "not_supported",
            "observed": out_of_range,
            "evidence": "post_tanh_value_checked_against_closed_interval_-1_1",
        },
        "zero_negative_or_boundary_value": {
            "status": "supported",
            "observed": True,
            "evidence": "active_denormalized_parameter_is_less_than_or_equal_to_zero",
        },
        "normalization_or_denormalization_error": {
            "status": (
                "supported" if normalization_mismatch else "not_supported"
            ),
            "observed": normalization_mismatch,
            "evidence": "recorded_value_equals_normalized_value_times_frozen_scale",
        },
        "units_mismatch": {
            "status": "not_supported",
            "observed": False,
            "evidence": "channel_37_uses_length_scale_and_channel_38_uses_degrees",
        },
        "direction_or_boolean_interaction": {
            "status": "not_supported_by_primary_failure",
            "observed": False,
            "evidence": "primary_failure_is_invalid_operation_parameter",
        },
        "legal_but_geometrically_incompatible_parameter": {
            "status": "unassessable",
            "observed": None,
            "reason": "no_cad_kernel_in_evaluation_contract",
        },
        "encoder_memory_error": {
            "status": "unresolved",
            "observed": None,
            "reason": "no_authorized_memory_or_head_intervention_in_this_diagnostic",
        },
        "shared_geometry_head_or_reconstruction_path_problem": {
            "status": "unresolved",
            "observed": None,
            "reason": "no_authorized_memory_or_head_intervention_in_this_diagnostic",
        },
        "mask_channel_conversion_or_evaluation_defect": {
            "status": (
                "supported" if mask_or_channel_mismatch else "not_supported"
            ),
            "observed": mask_or_channel_mismatch,
            "evidence": (
                "mask_channel_or_reproduction_mismatch"
                if mask_or_channel_mismatch
                else "active_masks_channels_and_exact_c7_metric_reproduction_agree"
            ),
        },
    }
    return {
        "status": "classified",
        "observed_mechanism": "nonpositive_active_operation_parameter",
        "failed_operation_index": failed_index,
        "nonpositive_operations": [
            {
                "operation_index": item["operation_index"],
                "operation_type": item["operation_type"],
                "classification": item["analytic_parameter_contract"][
                    "classification"
                ],
                "normalized_value": item[
                    "normalized_predicted_value_post_tanh"
                ],
                "physical_value": item["physical_predicted_value"],
                "raw_head_output_pre_tanh": item["raw_head_output_pre_tanh"],
            }
            for item in nonpositive
        ],
        "normalization_or_denormalization_mismatch_observed": (
            normalization_mismatch
        ),
        "mask_or_channel_selection_mismatch_observed": mask_or_channel_mismatch,
        "units_mismatch_observed": False,
        "direction_or_boolean_rejection_observed": False,
        "executor_result": dict(EXECUTOR_UNAVAILABLE),
        "legal_parameter_geometric_incompatibility": dict(
            GEOMETRIC_INCOMPATIBILITY_UNASSESSABLE
        ),
        "hypothesis_classification": hypotheses,
        "root_cause_attribution": {
            "resolved": False,
            "reason": (
                "scalar evidence localizes the failure to a nonpositive shared-"
                "decoder output, but cannot separate encoder-memory causation "
                "from the shared geometry-head response without an additional "
                "intervention"
            ),
        },
    }


def _prepare_output(output_dir, *, repository_root, corpus_dir, source_artifact, job_id):
    raw = Path(output_dir)
    if raw.exists() or raw.is_symlink():
        raise GraphEncoderError(
            "diagnostic_output_exists", "output must be a new non-symlink path"
        )
    final = raw.resolve()
    roots = tuple(
        Path(item).resolve()
        for item in (repository_root, corpus_dir, source_artifact)
    )
    if any(_is_within(final, root) for root in roots):
        raise GraphEncoderError(
            "unsafe_diagnostic_output_location",
            "output must be outside repository, corpus, and source artifact",
        )
    final.parent.mkdir(parents=True, exist_ok=True)
    staging = final.with_name(final.name + ".incomplete-" + job_id)
    if staging.exists() or staging.is_symlink():
        raise GraphEncoderError(
            "diagnostic_staging_exists", "incomplete output already exists"
        )
    staging.mkdir()
    return final, staging


def _regular_files(root, excluded=()):
    excluded_values = set(excluded)
    result = []
    for path in Path(root).rglob("*"):
        if path.is_symlink():
            raise GraphEncoderError(
                "invalid_diagnostic_artifact", "artifact symlinks are forbidden"
            )
        if path.is_file():
            relative = path.relative_to(root).as_posix()
            if relative not in excluded_values:
                result.append(relative)
    return tuple(sorted(result))


def finalize_diagnostic_artifact(staging_dir, final_dir):
    staging = Path(staging_dir)
    final = Path(final_dir)
    required = {
        "diagnosis.json",
        "metrics.jsonl",
        "resolved_config.json",
    }
    ordinary = set(_regular_files(
        staging, excluded=("artifact_manifest.json", "SHA256SUMS")
    ))
    if ordinary != required:
        raise GraphEncoderError(
            "incomplete_diagnostic_artifact",
            "diagnostic artifact file set differs",
        )
    if any(path.endswith(".pt") for path in ordinary):
        raise GraphEncoderError(
            "checkpoint_write_forbidden", "diagnostic cannot publish checkpoints"
        )
    manifest = {
        "schema_version": DIAGNOSTIC_ARTIFACT_VERSION,
        "artifacts": [
            {
                "path": path,
                "byte_size": (staging / path).stat().st_size,
                "sha256": _file_sha256(staging / path),
            }
            for path in sorted(ordinary)
        ],
    }
    _atomic_write_json(staging / "artifact_manifest.json", manifest)
    checksum_paths = _regular_files(staging, excluded=("SHA256SUMS",))
    checksums = "".join(
        "{}  {}\n".format(_file_sha256(staging / path), path)
        for path in checksum_paths
    )
    _atomic_write_bytes(staging / "SHA256SUMS", checksums.encode("utf-8"))
    verify_diagnostic_artifact(staging, allow_incomplete_name=True)
    os.replace(str(staging), str(final))
    return final


def verify_diagnostic_artifact(
    path,
    *,
    allow_incomplete_name=False,
    expected_commit=None,
    expected_slurm_job_id=None,
):
    root = Path(path)
    if not root.is_dir() or root.is_symlink():
        raise GraphEncoderError(
            "invalid_diagnostic_artifact", "artifact must be a real directory"
        )
    if ".incomplete-" in root.name and not allow_incomplete_name:
        raise GraphEncoderError(
            "invalid_diagnostic_artifact", "incomplete artifact is not final"
        )
    manifest = json.loads((root / "artifact_manifest.json").read_text("utf-8"))
    if manifest.get("schema_version") != DIAGNOSTIC_ARTIFACT_VERSION:
        raise GraphEncoderError(
            "invalid_diagnostic_artifact", "artifact identity differs"
        )
    rows = manifest.get("artifacts")
    if not isinstance(rows, list):
        raise GraphEncoderError(
            "invalid_diagnostic_artifact", "artifact rows are absent"
        )
    listed = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != {
            "path", "byte_size", "sha256"
        }:
            raise GraphEncoderError(
                "invalid_diagnostic_artifact", "artifact row is malformed"
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
                "diagnostic_artifact_integrity_failure",
                "artifact file size or SHA-256 differs",
            )
        listed.append(relative)
    expected = _regular_files(
        root, excluded=("artifact_manifest.json", "SHA256SUMS")
    )
    if tuple(listed) != expected:
        raise GraphEncoderError(
            "invalid_diagnostic_artifact", "manifest coverage differs"
        )
    checksum_raw = (root / "SHA256SUMS").read_bytes()
    if not checksum_raw.endswith(b"\n"):
        raise GraphEncoderError(
            "invalid_diagnostic_artifact", "SHA256SUMS requires final LF"
        )
    checksum_paths = []
    for line in checksum_raw.decode("utf-8").splitlines():
        parts = line.split("  ", 1)
        if len(parts) != 2 or len(parts[0]) != 64:
            raise GraphEncoderError(
                "invalid_diagnostic_artifact", "SHA256SUMS row is malformed"
            )
        relative = _safe_relative_path(parts[1])
        if _file_sha256(root / relative) != parts[0]:
            raise GraphEncoderError(
                "diagnostic_artifact_integrity_failure",
                "SHA256SUMS hash differs",
            )
        checksum_paths.append(relative)
    if tuple(checksum_paths) != _regular_files(root, excluded=("SHA256SUMS",)):
        raise GraphEncoderError(
            "invalid_diagnostic_artifact", "SHA256SUMS coverage differs"
        )
    resolved = json.loads((root / "resolved_config.json").read_text("utf-8"))
    diagnosis = json.loads((root / "diagnosis.json").read_text("utf-8"))
    validate_minimal_diagnostic_record(diagnosis)
    events = _read_canonical_jsonl(root / "metrics.jsonl")
    if not events or events[-1].get("event") != "diagnostic_completed":
        raise GraphEncoderError(
            "invalid_diagnostic_artifact", "terminal event is absent"
        )
    if (
        resolved.get("protocol_version") != DIAGNOSTIC_PROTOCOL_VERSION
        or resolved.get("artifact_version") != DIAGNOSTIC_ARTIFACT_VERSION
        or resolved.get("source_c7_v2", {}).get("commit")
        != SOURCE_C7_V2_COMMIT
        or resolved.get("source_c7_v2", {}).get("job_id")
        != SOURCE_C7_V2_JOB_ID
        or resolved.get("source_c7_v2_immutable") is not True
    ):
        raise GraphEncoderError(
            "invalid_diagnostic_artifact", "resolved identities differ"
        )
    if expected_commit is not None and (
        resolved.get("diagnostic_source", {}).get("git_commit") != expected_commit
    ):
        raise GraphEncoderError(
            "invalid_diagnostic_artifact", "diagnostic commit differs"
        )
    if expected_slurm_job_id is not None and (
        resolved.get("runtime", {}).get("slurm_job_id")
        != expected_slurm_job_id
    ):
        raise GraphEncoderError(
            "invalid_diagnostic_artifact", "diagnostic Slurm job differs"
        )
    access = resolved.get("access", {})
    if (
        access.get("operation_template_manifest_accessed") is not True
        or access.get("operation_template_train_payload_accessed") is not True
        or access.get("scaled_train_payload_accessed") is not True
        or access.get("diagnostic_completed") is not True
        or any(access.get(name) is not False for name in PROTECTED_ACCESS_FIELDS)
        or access.get("training_performed") is not False
        or access.get("checkpoint_written") is not False
        or access.get("decoder_repair_implemented_or_invoked") is not False
        or access.get("stage6_performed") is not False
        or access.get("c8_or_later_performed") is not False
    ):
        raise GraphEncoderError(
            "invalid_diagnostic_artifact", "access or non-authorization differs"
        )
    return {
        "protocol_version": DIAGNOSTIC_PROTOCOL_VERSION,
        "artifact_version": DIAGNOSTIC_ARTIFACT_VERSION,
        "artifact_manifest_sha256": _file_sha256(root / "artifact_manifest.json"),
        "sha256sums_sha256": _file_sha256(root / "SHA256SUMS"),
        "family_diagnostic_count": len(diagnosis["family_diagnostics"]),
        "diagnostic_completed": True,
    }


def run_operation_parameter_diagnostic(
    *,
    corpus_dir,
    source_c7_v2_artifact,
    output_dir,
    repository_root,
    expected_commit,
    access_tracker=None,
):
    """Reproduce job 3344981 and diagnose only its five failed families."""

    job_id = _require_authoritative_runtime()
    tracker = access_tracker or DiagnosticAccessTracker()
    source = _source_identity(repository_root, expected_commit)
    final, staging = _prepare_output(
        output_dir,
        repository_root=repository_root,
        corpus_dir=corpus_dir,
        source_artifact=source_c7_v2_artifact,
        job_id=job_id,
    )
    started = time.perf_counter()
    events = []
    source_verification, original_events, unused_reload_events = (
        _source_c7_evidence(source_c7_v2_artifact)
    )
    del unused_reload_events
    events.append({
        "event": "source_c7_v2_artifact_verified",
        "commit": SOURCE_C7_V2_COMMIT,
        "job_id": SOURCE_C7_V2_JOB_ID,
        "verification": source_verification,
        "access": tracker.declarations(),
    })

    tracker.operation_template_manifest_accessed = "not_confirmed_on_failure"
    selection = select_c7_sufficiency_subsets(corpus_dir)
    tracker.operation_template_manifest_accessed = True
    selection_record = validate_frozen_c7_v2_selection(selection)
    if selection.scaled_family_ids_sha256 != C7_V2_SCALED_FAMILY_IDS_SHA256:
        raise GraphEncoderError(
            "diagnostic_cohort_mismatch", "scaled cohort hash differs"
        )
    templates = dict(selection.selected_templates)
    if any(templates.get(key) != value for key, value in EXPECTED_TEMPLATES.items()):
        raise GraphEncoderError(
            "diagnostic_cohort_mismatch", "failed-family templates differ"
        )
    events.append({
        "event": "diagnostic_scaled_cohort_selected",
        "selection": selection_record,
        "access": tracker.declarations(),
    })

    tracker.operation_template_train_payload_accessed = "not_confirmed_on_failure"
    tracker.scaled_train_payload_accessed = "not_confirmed_on_failure"
    examples = tuple(load_train(corpus_dir, selection.scaled_family_ids))
    if tuple(item.physical_family_id for item in examples) != selection.scaled_family_ids:
        raise GraphEncoderError(
            "diagnostic_payload_alignment_failure", "loaded cohort differs"
        )
    tracker.operation_template_train_payload_accessed = True
    tracker.scaled_train_payload_accessed = True

    from .autonomous import autonomous_input_from_paired, run_autonomous_evaluation
    from .batching import build_paired_batch
    from .metrics import score_condition
    from .provenance import training_partition_identity

    partition_identity = training_partition_identity(selection.scaled_family_ids)
    ordered = tuple(sorted(examples, key=lambda item: item.physical_family_id))
    examples_by_id = {item.physical_family_id: item for item in ordered}
    targets = {item.physical_family_id: item.target for item in ordered}
    original_predictions = {}
    models = {}
    reproduction = []
    checkpoint_records = []
    source_root = Path(source_c7_v2_artifact)
    for arm in C7_V2_ARMS:
        checkpoint = SOURCE_CHECKPOINTS[arm]
        path = source_root / checkpoint["relative_path"]
        model, payload, before_state = _load_read_only_model_checkpoint(
            path,
            arm=arm,
            expected_sha256=checkpoint["sha256"],
            partition_identity=partition_identity,
        )
        models[arm] = model
        input_batches = tuple(
            autonomous_input_from_paired(
                build_paired_batch(ordered[start:start + C7_V2_BATCH_SIZE]),
                arm,
            )
            for start in range(0, len(ordered), C7_V2_BATCH_SIZE)
        )
        autonomous = run_autonomous_evaluation(
            model, input_batches, seed=C7_V2_SEED
        )
        true_condition = next(
            item for item in autonomous.conditions if item.condition == "P_true"
        )
        metrics = score_condition(true_condition, targets)
        source_event = original_events[arm]
        if [list(item) for item in true_condition.memory_assignments] != (
            source_event.get("memory_assignments")
        ):
            raise GraphEncoderError(
                "c7_v2_metric_reproduction_failure",
                "{} P_true memory assignments differ".format(arm),
            )
        batch_membership = [
            [identity, list(family_ids)]
            for identity, family_ids in true_condition.batch_membership
        ]
        if batch_membership != source_event.get("batch_membership"):
            raise GraphEncoderError(
                "c7_v2_metric_reproduction_failure",
                "{} P_true batch membership differs".format(arm),
            )
        reproduced = validate_reproduced_c7_metrics(
            source_event["metrics"],
            metrics,
            arm=arm,
            templates_by_family=templates,
        )
        reproduced["batch_membership"] = batch_membership
        reproduced["memory_assignments"] = [
            list(item) for item in true_condition.memory_assignments
        ]
        after_state = _state_sha256(model)
        if before_state != after_state:
            raise GraphEncoderError(
                "diagnostic_model_mutation", "inference changed model state"
            )
        reproduction.append(reproduced)
        original_predictions[arm] = {
            item.family_id: item for item in true_condition.predictions
        }
        checkpoint_records.append({
            "arm": arm,
            "relative_path": checkpoint["relative_path"],
            "sha256": checkpoint["sha256"],
            "source_commit": payload["provenance"]["git_commit"],
            "source_job_id": payload["provenance"]["slurm_job_id"],
            "epoch": payload["completed_epoch"],
            "model_state_sha256_before_inference": before_state,
            "model_state_sha256_after_inference": after_state,
            "model_state_unchanged": True,
            "checkpoint_modified": False,
        })
        events.append({
            "event": "diagnostic_arm_reproduction_passed",
            "arm": arm,
            "checkpoint": checkpoint_records[-1],
            "reproduction": reproduced,
            "access": tracker.declarations(),
        })

    family_diagnostics = []
    for family_id in FAILED_FAMILY_IDS:
        failing_arm = next(
            arm for arm, values in FAILED_FAMILIES_BY_ARM.items()
            if family_id in values
        )
        family = {
            "family_id": family_id,
            "operation_template": templates[family_id],
            "failing_arm": failing_arm,
            "arm_results": [
                _trace_arm_family(
                    models[arm],
                    original_predictions[arm][family_id],
                    examples_by_id[family_id],
                    arm,
                )
                for arm in C7_V2_ARMS
            ],
        }
        family_diagnostics.append(family)
    _attach_controls(family_diagnostics)
    for family in family_diagnostics:
        family["classification"] = _classify_failed_family(family)

    diagnosis = {
        "schema_version": DIAGNOSTIC_RECORD_VERSION,
        "protocol_version": DIAGNOSTIC_PROTOCOL_VERSION,
        "source_c7_v2": {
            "commit": SOURCE_C7_V2_COMMIT,
            "job_id": SOURCE_C7_V2_JOB_ID,
            "immutable_result_changed": False,
            "metrics_sha256": SOURCE_C7_V2_METRICS_SHA256,
            "scaled_family_ids_sha256": C7_V2_SCALED_FAMILY_IDS_SHA256,
        },
        "corrected_validity_contract": corrected_validity_contract(),
        "reproduction": reproduction,
        "checkpoint_verification": checkpoint_records,
        "family_diagnostics": family_diagnostics,
        "access": tracker.declarations(completed=True),
    }
    validate_minimal_diagnostic_record(diagnosis)
    events.append({
        "event": "diagnostic_classification_completed",
        "family_count": len(family_diagnostics),
        "all_cases_classified": True,
        "access": tracker.declarations(),
    })
    resolved = {
        "protocol_version": DIAGNOSTIC_PROTOCOL_VERSION,
        "artifact_version": DIAGNOSTIC_ARTIFACT_VERSION,
        "record_version": DIAGNOSTIC_RECORD_VERSION,
        "diagnostic_source": source,
        "source_c7_v2": {
            "commit": SOURCE_C7_V2_COMMIT,
            "job_id": SOURCE_C7_V2_JOB_ID,
            "artifact_path": str(Path(source_c7_v2_artifact).resolve()),
            "metrics_sha256": SOURCE_C7_V2_METRICS_SHA256,
            "artifact_manifest_sha256": (
                SOURCE_C7_V2_ARTIFACT_MANIFEST_SHA256
            ),
            "checkpoint_files": copy.deepcopy(SOURCE_CHECKPOINTS),
        },
        "source_c7_v2_immutable": True,
        "authoritative_manifest": {
            "name": "operation_template",
            "sha256": AUTHORITATIVE_FILE_SHA256,
            "partition": "train",
        },
        "scaled_family_ids": list(selection.scaled_family_ids),
        "scaled_family_ids_sha256": selection.scaled_family_ids_sha256,
        "failed_family_ids": list(FAILED_FAMILY_IDS),
        "memory_condition": "P_true",
        "batch_size": C7_V2_BATCH_SIZE,
        "seed": C7_V2_SEED,
        "checkpoint_epoch": C7_V2_CHECKPOINT_EPOCH,
        "corrected_validity_contract": corrected_validity_contract(),
        "runtime": {
            "python": platform.python_version(),
            "pytorch": str(torch.__version__),
            "device": "cpu",
            "cuda_available": bool(torch.cuda.is_available()),
            "host": socket.gethostname(),
            "slurm_job_id": job_id,
        },
        "access": tracker.declarations(completed=True),
    }
    completion = {
        "event": "diagnostic_completed",
        "protocol_version": DIAGNOSTIC_PROTOCOL_VERSION,
        "diagnostic_completed": True,
        "source_c7_v2_result_changed": False,
        "all_five_cases_classified": True,
        "stage6_authorized": False,
        "repair_authorized_or_performed": False,
        "c8_or_later_performed": False,
        "elapsed_seconds": time.perf_counter() - started,
        "access": tracker.declarations(completed=True),
    }
    events = [{
        "event": "diagnostic_run_metadata",
        "protocol_version": DIAGNOSTIC_PROTOCOL_VERSION,
        "diagnostic_source": source,
        "source_c7_v2_commit": SOURCE_C7_V2_COMMIT,
        "source_c7_v2_job_id": SOURCE_C7_V2_JOB_ID,
        "slurm_job_id": job_id,
        "access": tracker.declarations(completed=True),
    }] + events + [completion]
    _atomic_write_json(staging / "diagnosis.json", diagnosis)
    _atomic_write_json(staging / "resolved_config.json", resolved)
    _atomic_write_jsonl(staging / "metrics.jsonl", events)
    _verify_source_unchanged(source, repository_root, expected_commit)
    finalize_diagnostic_artifact(staging, final)
    verified = verify_diagnostic_artifact(
        final,
        expected_commit=expected_commit,
        expected_slurm_job_id=job_id,
    )
    return {
        "artifact_path": str(final),
        "artifact_verification": verified,
        "completion_event": completion,
    }


def _parser():
    parser = argparse.ArgumentParser(
        description="GE1 C7-v2 read-only operation-parameter diagnostic"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--corpus-dir", required=True)
    run.add_argument("--source-c7-v2-artifact", required=True)
    run.add_argument("--output-dir", required=True)
    run.add_argument("--repository-root", required=True)
    run.add_argument("--expected-commit", required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--artifact-dir", required=True)
    verify.add_argument("--expected-commit", required=True)
    verify.add_argument("--expected-slurm-job-id", required=True)
    return parser


def main(argv=None):
    arguments = _parser().parse_args(argv)
    if arguments.command == "run":
        result = run_operation_parameter_diagnostic(
            corpus_dir=arguments.corpus_dir,
            source_c7_v2_artifact=arguments.source_c7_v2_artifact,
            output_dir=arguments.output_dir,
            repository_root=arguments.repository_root,
            expected_commit=arguments.expected_commit,
        )
        print(json.dumps({
            "event": "operation_parameter_diagnostic_terminal_completed",
            "artifact_path": result["artifact_path"],
            "diagnostic_completed": True,
            "source_c7_v2_result_changed": False,
            "stage6_authorized": False,
            "repair_authorized_or_performed": False,
            "c8_or_later_performed": False,
            "operation_template_manifest_accessed": True,
            "operation_template_train_payload_accessed": True,
            "scaled_train_payload_accessed": True,
            "development_accessed": False,
            "systematic_rr_accessed": False,
            "test_er_accessed": False,
            "iid_accessed": False,
            "history_depth_accessed": False,
            "geometry_extrapolation_accessed": False,
        }, sort_keys=True, separators=(",", ":")), flush=True)
        return 0
    verified = verify_diagnostic_artifact(
        arguments.artifact_dir,
        expected_commit=arguments.expected_commit,
        expected_slurm_job_id=arguments.expected_slurm_job_id,
    )
    print(json.dumps({
        "event": "operation_parameter_diagnostic_artifact_verified",
        **verified,
    }, sort_keys=True, separators=(",", ":")), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
