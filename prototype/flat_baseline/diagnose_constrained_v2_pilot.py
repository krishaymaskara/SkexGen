"""Read-only frozen-checkpoint diagnosis for the Stage 2F IID pilot."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import re
import sys

import torch
from torch.nn import functional as F

from prototype.constrained_profile_decoder import profile_targets_for_loss
from prototype.controlled_data.factors import OperationTemplate
from prototype.model_data.batching import collate_flat
from prototype.model_data.vocab import (
    EDGE_TYPES,
    NODE_TYPES,
)
from prototype.profile_geometry import PROFILE_FAMILIES
from prototype.profile_geometry_torch import (
    canonicalize_profile_tensors,
    constrain_profile_parameters,
)

from .autonomous import (
    RawDecodedEdge,
    RawDecodedNode,
    RawDecodedPointer,
    derive_geometry_mask,
)
from .checkpointing import capture_rng_state, restore_rng_state
from .constrained_v2 import (
    REMAINING_CATEGORICAL_TARGET_INDICES,
    ConstrainedProfileV2Model,
)
from .constrained_v2_autonomous import (
    greedy_decode_v2,
    validate_and_convert_v2_autonomous_prediction,
)
from .constrained_v2_config import ConstrainedProfileV2Config
from .constrained_v2_conversion import (
    _prediction_row,
    construct_v2_predicted_node_tensors,
    v2_teacher_forced_predictions,
    validate_and_convert_v2_teacher_forced_prediction,
)
from .constrained_v2_losses import constrained_profile_v2_loss
from .constrained_v2_pilot import (
    PILOT_CHECKPOINT_FIELDS,
    _raw_prediction_is_finite,
    _raw_profile_records_are_canonical,
    load_pilot_data,
)
from .constrained_v2_pilot_config import ConstrainedV2PilotConfig
from .constrained_v2_pilot_diagnostic_config import (
    ConstrainedV2PilotDiagnosticConfig,
    ConstrainedV2PilotDiagnosticError,
    EXPECTED_EPOCH,
    EXPECTED_GLOBAL_STEP,
    validate_diagnostic_partition_authorization,
)
from .provenance import source_state
from .run_logging import JsonlLogger
from .training import resolve_device


HYBRID_ARM_CONTRACTS = {
    "A": {
        "label": "fully_autonomous",
        "history": "generated",
        "oracle": False,
        "replacements": [],
    },
    "B": {
        "label": "teacher_forced_history_predicted_current",
        "history": "authoritative_prior_nodes",
        "oracle": False,
        "replacements": [],
    },
    "C": {
        "label": "oracle_node_types",
        "history": "authoritative_prior_nodes",
        "oracle": True,
        "replacements": ["node_type"],
        "derived_consequences": [
            "geometry_applicability_mask_for_replaced_node_type"
        ],
    },
    "D": {
        "label": "oracle_node_types_and_retained_categories",
        "history": "authoritative_prior_nodes",
        "oracle": True,
        "replacements": ["node_type", "retained_categorical_fields"],
        "derived_consequences": [
            "geometry_applicability_mask_for_replaced_discrete_fields"
        ],
    },
    "E": {
        "label": "oracle_discrete_structure",
        "history": "authoritative_prior_nodes",
        "oracle": True,
        "replacements": [
            "node_type",
            "retained_categorical_fields",
            "profile_family",
            "edge_presence_and_type",
            "operation_pointers_and_counts",
        ],
        "derived_consequences": [
            "canonical_primitive_slots",
            "geometry_applicability_mask_for_oracle_discrete_structure",
        ],
    },
    "F": {
        "label": "autonomous_oracle_family_only",
        "history": "generated",
        "oracle": True,
        "replacements": ["profile_family_at_aligned_generated_sketches"],
        "derived_consequences": [
            "canonical_primitive_slots",
            "constrained_parameters",
            "canonical_profile_geometry_and_mask",
        ],
    },
}

ENCODER_VARIANCE_THRESHOLD = 1e-6
ENCODER_DISTANCE_THRESHOLD = 1e-3
CODEBOOK_NORM_STD_THRESHOLD = 1e-6
EMA_LIVE_THRESHOLD = 1e-6
MATERIAL_VALIDITY_IMPROVEMENT = 0.10
HIGH_ORACLE_DISCRETE_VALIDITY = 0.50
LOW_AUTONOMOUS_VALIDITY = 0.10
_NODE_LOCATION = re.compile(r"raw_nodes\[(\d+)\]")


@dataclass(frozen=True)
class DiagnosticResult:
    output_dir: str
    metrics_path: str
    first_failures_path: str
    classifications: tuple
    checkpoint_sha256: str


def audit_frozen_checkpoint_payload(
    checkpoint,
    config,
    *,
    expected_epoch=EXPECTED_EPOCH,
    expected_global_step=EXPECTED_GLOBAL_STEP,
):
    """Audit identity before constructing a model or opening corpus payloads."""

    validate_diagnostic_partition_authorization(config)
    if (
        not isinstance(checkpoint, dict)
        or set(checkpoint) != PILOT_CHECKPOINT_FIELDS
    ):
        raise ConstrainedV2PilotDiagnosticError(
            "malformed_checkpoint", "pilot checkpoint fields differ"
        )
    required = (
        ("checkpoint_version", 2),
        ("checkpoint_kind", "pilot_fixed_epoch"),
        ("model_name", "B0-FLAT-CONSTRAINED-PROFILE-v2"),
        ("model_config_version", 2),
        ("decoder_contract_version", 2),
        ("pilot_identity", "B0-FLAT-CONSTRAINED-PROFILE-v2-iid-pilot-v1"),
        ("epoch", expected_epoch),
        ("global_step", expected_global_step),
        ("completed_epochs", expected_epoch),
        ("systematic_partition_accessed", False),
        ("test_partition_accessed", False),
    )
    for name, expected in required:
        value = checkpoint.get(name)
        if value != expected or type(value) is not type(expected):
            raise ConstrainedV2PilotDiagnosticError(
                "incompatible_checkpoint",
                "{} must equal {!r}".format(name, expected),
            )
    provenance = checkpoint.get("source_provenance")
    if (
        not isinstance(provenance, dict)
        or provenance.get("git_commit")
        != config.expected_pilot_source_commit
    ):
        raise ConstrainedV2PilotDiagnosticError(
            "incompatible_checkpoint", "pilot source commit differs"
        )
    training = checkpoint.get("training_partition_state")
    validation = checkpoint.get("validation_partition_state")
    if (
        not isinstance(training, dict)
        or training.get("partition") != "train"
        or training.get("partition_identity") != "train"
        or training.get("family_count") != config.expected_train_count
        or not isinstance(validation, dict)
        or validation.get("partition") != "validation"
        or validation.get("partition_identity") != "iid_validation"
        or validation.get("family_count")
        != config.expected_validation_count
    ):
        raise ConstrainedV2PilotDiagnosticError(
            "incompatible_checkpoint", "partition identity differs"
        )
    return {
        "checkpoint_version": 2,
        "model_name": checkpoint["model_name"],
        "source_commit": provenance["git_commit"],
        "epoch": checkpoint["epoch"],
        "global_step": checkpoint["global_step"],
        "systematic_partition_accessed": False,
        "test_partition_accessed": False,
    }


def vq_assignment_summary(vectors, codebook, ema_cluster_size):
    """Return deterministic assignment and separation evidence."""

    if (
        not torch.is_tensor(vectors)
        or not torch.is_tensor(codebook)
        or not torch.is_tensor(ema_cluster_size)
        or vectors.dim() != 2
        or codebook.dim() != 2
        or vectors.size(1) != codebook.size(1)
        or ema_cluster_size.shape != (codebook.size(0),)
        or vectors.size(0) == 0
    ):
        raise ValueError("VQ diagnostic tensors are misaligned")
    if not all(torch.isfinite(item).all() for item in (
        vectors, codebook, ema_cluster_size
    )):
        raise ValueError("VQ diagnostic tensors must be finite")
    squared = (
        vectors.pow(2).sum(dim=1, keepdim=True)
        + codebook.pow(2).sum(dim=1).unsqueeze(0)
        - 2.0 * torch.matmul(vectors, codebook.t())
    ).clamp_min(0.0)
    sorted_distances, sorted_ids = squared.sort(dim=1)
    assigned = sorted_ids[:, 0]
    nearest = sorted_distances[:, 0].sqrt()
    second = sorted_distances[:, 1].sqrt()
    counts = torch.bincount(assigned, minlength=codebook.size(0))
    probabilities = counts.double() / float(assigned.numel())
    nonzero = probabilities > 0
    perplexity = torch.exp(
        -(probabilities[nonzero] * probabilities[nonzero].log()).sum()
    )
    pairwise = (
        torch.pdist(vectors.double())
        if vectors.size(0) > 1
        else vectors.new_zeros((0,), dtype=torch.double)
    )
    vector_std = vectors.double().std(dim=0, unbiased=False)
    vector_mean = vectors.double().mean(dim=0)
    vector_variance = vectors.double().var(dim=0, unbiased=False)
    norms = vectors.double().norm(dim=1)
    codebook_norms = codebook.double().norm(dim=1)
    margin = second - nearest
    result = {
        "total_latent_tokens": int(vectors.size(0)),
        "active_code_ids": torch.nonzero(
            counts > 0, as_tuple=False
        ).reshape(-1).tolist(),
        "assignment_count_per_code": counts.tolist(),
        "active_code_count": int((counts > 0).sum().item()),
        "perplexity": float(perplexity.item()),
        "dominant_code_fraction": float(
            counts.max().double().item() / float(assigned.numel())
        ),
        "prequantized_mean_per_dimension": vector_mean.tolist(),
        "prequantized_std_per_dimension": vector_std.tolist(),
        "aggregate_prequantized_variance": float(
            vector_variance.mean().item()
        ),
        "inter_vector_distance": _tensor_distribution(pairwise),
        "distance_to_assigned_code": _tensor_distribution(nearest),
        "distance_to_second_nearest_code": _tensor_distribution(second),
        "nearest_second_margin": _tensor_distribution(margin),
        "codebook_vector_norms": codebook_norms.tolist(),
        "codebook_norm_distribution": _tensor_distribution(codebook_norms),
        "ema_cluster_sizes": ema_cluster_size.double().tolist(),
        "dead_code_count": int(
            (ema_cluster_size <= EMA_LIVE_THRESHOLD).sum().item()
        ),
        "encoder_output_norm": _tensor_distribution(norms),
    }
    result["collapse_classification"] = classify_vq_collapse(result)
    _require_finite_json(result)
    return result


def classify_vq_collapse(summary):
    encoder = (
        summary["aggregate_prequantized_variance"]
        <= ENCODER_VARIANCE_THRESHOLD
        and summary["inter_vector_distance"]["mean"]
        <= ENCODER_DISTANCE_THRESHOLD
    )
    codebook = (
        summary["codebook_norm_distribution"]["std"]
        <= CODEBOOK_NORM_STD_THRESHOLD
        or len(summary["ema_cluster_sizes"]) - summary["dead_code_count"] <= 1
    )
    if encoder and codebook:
        return "both"
    if encoder:
        return "encoder_collapse"
    if codebook:
        return "codebook_or_ema_collapse"
    return "neither"


def family_prediction_summary(logits, targets, predicted_classes):
    """Summarize family evidence, allowing a non-sketch prediction column."""

    logits = torch.as_tensor(logits, dtype=torch.float64)
    targets = torch.as_tensor(targets, dtype=torch.long)
    predicted = torch.as_tensor(predicted_classes, dtype=torch.long)
    if (
        logits.dim() != 2
        or logits.size(1) != len(PROFILE_FAMILIES)
        or targets.shape != (logits.size(0),)
        or predicted.shape != targets.shape
        or logits.size(0) == 0
        or not torch.isfinite(logits).all()
        or int(targets.min().item()) < 0
        or int(targets.max().item()) >= len(PROFILE_FAMILIES)
        or int(predicted.min().item()) < 0
        or int(predicted.max().item()) > len(PROFILE_FAMILIES)
    ):
        raise ValueError("family diagnostic inputs are invalid")
    class_count = len(PROFILE_FAMILIES)
    confusion = torch.zeros(
        class_count, class_count + 1, dtype=torch.long
    )
    for target, prediction in zip(targets.tolist(), predicted.tolist()):
        confusion[target, prediction] += 1
    probabilities = torch.softmax(logits, dim=-1)
    entropy = -(probabilities * probabilities.clamp_min(1e-300).log()).sum(-1)
    top = logits.topk(2, dim=-1).values
    margin = top[:, 0] - top[:, 1]
    names = tuple(item.value for item in PROFILE_FAMILIES)
    per_class = {}
    recalls = []
    for class_id, name in enumerate(names):
        true_positive = int(confusion[class_id, class_id].item())
        support = int(confusion[class_id].sum().item())
        predicted_count = int(confusion[:, class_id].sum().item())
        recall = true_positive / float(support) if support else 0.0
        precision = (
            true_positive / float(predicted_count)
            if predicted_count else 0.0
        )
        recalls.append(recall)
        per_class[name] = {
            "precision": precision,
            "recall": recall,
            "support": support,
            "predicted_count": predicted_count,
        }
    histogram = {
        name: int((predicted == index).sum().item())
        for index, name in enumerate(names)
    }
    histogram["non_sketch"] = int(
        (predicted == class_count).sum().item()
    )
    result = {
        "example_position_count": int(targets.numel()),
        "confusion_matrix": confusion.tolist(),
        "confusion_columns": list(names) + ["non_sketch"],
        "predicted_class_histogram": histogram,
        "accuracy": float(
            sum(confusion[index, index].item() for index in range(class_count))
            / float(targets.numel())
        ),
        "balanced_accuracy": float(sum(recalls) / len(recalls)),
        "per_class": per_class,
        "average_logit_margin": float(margin.mean().item()),
        "average_entropy": float(entropy.mean().item()),
        "family_loss": float(F.cross_entropy(logits, targets).item()),
    }
    _require_finite_json(result)
    return result


def build_hybrid_arms(
    output,
    target,
    profiles,
    physical_examples,
    autonomous_predictions,
):
    """Build the frozen A-F matrix without modifying model outputs."""

    node_mask = target["node_mask"]
    teacher = v2_teacher_forced_predictions(
        output,
        node_mask=node_mask,
        node_count_source="authorized_validation_length",
    )
    predicted_node_types = output.node_type_logits.argmax(dim=-1)
    predicted_retained = torch.stack(
        tuple(logits.argmax(dim=-1) for logits in output.categorical_logits),
        dim=-1,
    )
    oracle_node_types = target["node_type_ids"]
    oracle_retained = target["categorical_attributes"][
        ..., REMAINING_CATEGORICAL_TARGET_INDICES
    ]
    forced_family_logits = output.profile_family_logits.clone()
    selected = profiles.sketch_mask.reshape(-1)
    forced_flat = forced_family_logits.reshape(-1, len(PROFILE_FAMILIES))
    family_flat = profiles.family_ids.reshape(-1)
    forced_flat[selected] = -1.0
    selected_indices = torch.nonzero(
        selected, as_tuple=False
    ).reshape(-1)
    forced_flat[
        selected_indices, family_flat[selected_indices]
    ] = 1.0

    records_c = construct_v2_predicted_node_tensors(
        oracle_node_types,
        predicted_retained,
        output.profile_family_logits,
        output.raw_profile_parameters,
        output.non_profile_geometry,
    )
    records_d = construct_v2_predicted_node_tensors(
        oracle_node_types,
        oracle_retained,
        output.profile_family_logits,
        output.raw_profile_parameters,
        output.non_profile_geometry,
    )
    records_e = construct_v2_predicted_node_tensors(
        oracle_node_types,
        oracle_retained,
        forced_family_logits,
        output.raw_profile_parameters,
        output.non_profile_geometry,
    )
    arms = {name: [] for name in HYBRID_ARM_CONTRACTS}
    for index, physical in enumerate(physical_examples):
        arms["A"].append(autonomous_predictions[index])
        arms["B"].append(teacher[index])
        arms["C"].append(_prediction_row(
            output, records_c, node_mask, index,
            "diagnostic_oracle_node_types",
        ))
        arms["D"].append(_prediction_row(
            output, records_d, node_mask, index,
            "diagnostic_oracle_node_types_and_categories",
        ))
        predicted_e = _prediction_row(
            output, records_e, node_mask, index,
            "diagnostic_oracle_discrete_structure",
        )
        arms["E"].append(_replace_oracle_relations(predicted_e, physical))
        family_ids = profiles.family_ids[index, :len(physical.nodes)]
        arms["F"].append(_replace_autonomous_family(
            autonomous_predictions[index], family_ids
        ))
    return {name: tuple(values) for name, values in arms.items()}


def classify_diagnostic(arm_summaries, vq_summaries):
    classifications = []
    a = arm_summaries["A"]["validity"]
    b = arm_summaries["B"]["validity"]
    e = arm_summaries["E"]["validity"]
    f = arm_summaries["F"]["validity"]
    if e >= HIGH_ORACLE_DISCRETE_VALIDITY and a <= LOW_AUTONOMOUS_VALIDITY:
        classifications.append("discrete_grammar_is_primary_bottleneck")
    if f - a >= MATERIAL_VALIDITY_IMPROVEMENT:
        classifications.append("family_classification_is_primary_bottleneck")
    if b - a >= MATERIAL_VALIDITY_IMPROVEMENT:
        classifications.append("autoregressive_exposure_is_primary_bottleneck")
    profile_codes = {"invalid_primitive_geometry", "invalid_loop_constraint"}
    e_profile_failures = sum(
        count for code, count in arm_summaries["E"][
            "failure_reason_histogram"
        ].items() if code in profile_codes
    )
    if (
        e < HIGH_ORACLE_DISCRETE_VALIDITY
        and e_profile_failures > arm_summaries["E"]["total_count"] / 2.0
    ):
        classifications.append("profile_geometry_is_primary_bottleneck")
    train_vq = vq_summaries["train"]
    validation_vq = vq_summaries["iid_validation"]
    if (
        train_vq["active_code_count"] == 1
        and validation_vq["active_code_count"] == 1
        and train_vq["dominant_code_fraction"] >= 0.95
        and validation_vq["dominant_code_fraction"] >= 0.95
        and (
            train_vq["collapse_classification"] != "neither"
            or validation_vq["collapse_classification"] != "neither"
        )
    ):
        classifications.append("vq_collapse_is_likely_contributor")
    if not classifications:
        classifications.append("insufficient_evidence")
    return tuple(classifications)


def run_frozen_pilot_diagnostic(
    corpus_dir,
    checkpoint_path,
    config=None,
    epoch1_checkpoint_path=None,
):
    config = config or ConstrainedV2PilotDiagnosticConfig()
    validate_diagnostic_partition_authorization(config)
    output_dir = Path(config.output_dir)
    checkpoint_path = Path(checkpoint_path)
    checkpoint_parent = checkpoint_path.resolve().parent
    resolved_output = output_dir.resolve()
    if (
        resolved_output == checkpoint_parent
        or checkpoint_parent in resolved_output.parents
    ):
        raise ConstrainedV2PilotDiagnosticError(
            "unsafe_output_path",
            "diagnostic output must be outside the frozen pilot directory",
        )
    if output_dir.exists() or output_dir.is_symlink():
        raise ConstrainedV2PilotDiagnosticError(
            "output_collision", "diagnostic output already exists"
        )
    provenance = source_state()
    if config.require_clean_source and (
        provenance["git_dirty"] is not False
        or provenance["git_commit"] is None
    ):
        raise ConstrainedV2PilotDiagnosticError(
            "dirty_source", "diagnostic requires clean committed source"
        )
    checkpoint_hash_before = _sha256(checkpoint_path)
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
        or data.validation.metadata()
        != checkpoint["validation_partition_state"]
    ):
        raise ConstrainedV2PilotDiagnosticError(
            "partition_fingerprint_mismatch",
            "loaded authorized partitions differ from checkpoint",
        )
    model_config_payload = dict(checkpoint["model_config"])
    if "profile_family_order" in model_config_payload:
        model_config_payload["profile_family_order"] = tuple(
            model_config_payload["profile_family_order"]
        )
    model_config = ConstrainedProfileV2Config(**model_config_payload)
    model_config.validate()
    device = resolve_device(config.device, torch)
    construction_rng = capture_rng_state(torch)
    model = ConstrainedProfileV2Model(model_config).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    restore_rng_state(construction_rng, torch)
    _assert_explicit_vq_state(model, checkpoint["vq_state"])
    model.eval()

    output_dir.mkdir(parents=True)
    logger = JsonlLogger(output_dir / "metrics.jsonl")
    first_failure_logger = JsonlLogger(
        output_dir / "first_failures.jsonl"
    )
    logger.write({
        "event": "diagnostic_metadata",
        "diagnostic_config": config.to_dict(),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_hash_before,
        "checkpoint_audit": checkpoint_audit,
        "training_partition": data.train.metadata(),
        "validation_partition": data.validation.metadata(),
        "source_provenance": provenance,
        "hybrid_arm_contracts": HYBRID_ARM_CONTRACTS,
        "training_occurred": False,
        "systematic_partition_accessed": False,
        "test_partition_accessed": False,
    })

    model_state_before = _clone_tree(model.state_dict())
    rng_before = capture_rng_state(torch)
    vq_summaries = {
        "train": _vq_partition_summary(
            model, data.train, config.batch_size, device
        ),
        "iid_validation": _vq_partition_summary(
            model, data.validation, config.batch_size, device
        ),
    }
    for partition_identity, summary in vq_summaries.items():
        logger.write({
            "event": "vq_partition_summary",
            "partition_identity": partition_identity,
            "summary": summary,
            "systematic_partition_accessed": False,
            "test_partition_accessed": False,
        })

    epoch_comparison = _optional_epoch1_comparison(
        epoch1_checkpoint_path,
        checkpoint_path,
        config,
        model_config,
        data,
        device,
    )
    validation = _validation_diagnostic(
        model, data.validation, model_config, config.batch_size, device
    )
    logger.write({
        "event": "family_teacher_forced_summary",
        "partition_identity": "iid_validation",
        "summary": validation["family_teacher_forced"],
        "systematic_partition_accessed": False,
        "test_partition_accessed": False,
    })
    logger.write({
        "event": "family_autonomous_summary",
        "partition_identity": "iid_validation",
        "summary": validation["family_autonomous"],
        "capsule_zero_prediction_diagnosis": validation[
            "capsule_zero_prediction_diagnosis"
        ],
        "systematic_partition_accessed": False,
        "test_partition_accessed": False,
    })
    for arm_name in HYBRID_ARM_CONTRACTS:
        logger.write({
            "event": "hybrid_arm_summary",
            "arm": arm_name,
            "contract": HYBRID_ARM_CONTRACTS[arm_name],
            "summary": validation["arm_summaries"][arm_name],
            "systematic_partition_accessed": False,
            "test_partition_accessed": False,
        })
    for record in validation["first_failure_records"]:
        first_failure_logger.write({
            "event": "first_failure_record",
            **record,
            "systematic_partition_accessed": False,
            "test_partition_accessed": False,
        })
    logger.write({
        "event": "first_failure_aggregate",
        "summary": validation["first_failure_aggregate"],
        "systematic_partition_accessed": False,
        "test_partition_accessed": False,
    })

    classifications = classify_diagnostic(
        validation["arm_summaries"], vq_summaries
    )
    logger.write({
        "event": "diagnostic_classification",
        "classifications": list(classifications),
        "rules": _classification_rules(),
        "epoch1_epoch2_comparison": epoch_comparison,
        "systematic_partition_accessed": False,
        "test_partition_accessed": False,
    })

    checkpoint_hash_after = _sha256(checkpoint_path)
    state_audit = {
        "model_state_unchanged": _tree_equal(
            model_state_before, model.state_dict()
        ),
        "vq_state_unchanged": _vq_matches_snapshot(
            model, model_state_before
        ),
        "rng_state_unchanged": _tree_equal(
            rng_before, capture_rng_state(torch)
        ),
        "checkpoint_bytes_unchanged": (
            checkpoint_hash_before == checkpoint_hash_after
        ),
        "model_in_evaluation_mode": model.training is False,
        "no_parameter_gradients_created": all(
            parameter.grad is None for parameter in model.parameters()
        ),
    }
    if not all(value is True for value in state_audit.values()):
        raise ConstrainedV2PilotDiagnosticError(
            "diagnostic_mutated_state", repr(state_audit)
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
    return DiagnosticResult(
        str(output_dir),
        str(logger.path),
        str(first_failure_logger.path),
        classifications,
        checkpoint_hash_after,
    )


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--epoch1-checkpoint")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    config = replace(
        ConstrainedV2PilotDiagnosticConfig(),
        output_dir=args.output_dir,
        device=args.device,
    )
    try:
        result = run_frozen_pilot_diagnostic(
            args.corpus_dir,
            args.checkpoint,
            config,
            epoch1_checkpoint_path=args.epoch1_checkpoint,
        )
    except Exception as exc:
        _append_terminal_failure(config, exc)
        print(json.dumps(
            {
                "event": "terminal_failure",
                "error_code": getattr(exc, "code", "diagnostic_failure"),
                "error_type": type(exc).__name__,
                "detail": str(exc),
                "training_occurred": False,
                "systematic_partition_accessed": False,
                "test_partition_accessed": False,
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ), file=sys.stderr)
        return 2
    print(json.dumps(
        {
            "event": "terminal_success",
            "output_dir": result.output_dir,
            "metrics_path": result.metrics_path,
            "first_failures_path": result.first_failures_path,
            "classifications": list(result.classifications),
            "checkpoint_sha256": result.checkpoint_sha256,
            "training_occurred": False,
            "systematic_partition_accessed": False,
            "test_partition_accessed": False,
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ))
    return 0


def _vq_partition_summary(model, partition, batch_size, device):
    vectors = []
    with torch.no_grad():
        for start in range(0, len(partition.flat_examples), batch_size):
            batch = collate_flat(
                partition.flat_examples[start:start + batch_size]
            )
            inputs = {
                name: value.to(device)
                for name, value in batch.to_torch(torch).items()
            }
            vectors.append(_encode_prequantized(model, inputs).cpu())
    flattened = torch.cat(vectors, dim=0).reshape(
        -1, model.config.codebook_dim
    )
    summary = vq_assignment_summary(
        flattened,
        model.vq.embedding.detach().cpu(),
        model.vq.ema_cluster_size.detach().cpu(),
    )
    summary["total_encoded_examples"] = len(partition.flat_examples)
    return summary


def _encode_prequantized(model, inputs):
    categorical_ids = inputs["categorical_ids"]
    geometry = inputs["geometry"]
    geometry_mask = inputs["geometry_mask"]
    padding_mask = inputs["padding_mask"]
    model._validate_encoder_inputs(
        categorical_ids, geometry, geometry_mask, padding_mask
    )
    node_count = categorical_ids.size(1)
    positions = torch.arange(
        node_count, device=categorical_ids.device
    ).unsqueeze(0)
    content = model._record_content(
        categorical_ids, geometry, geometry_mask
    )
    nodes = model.input_norm(
        content + model.node_position_embedding(positions)
    )
    queries = model.latent_queries.unsqueeze(0).expand(
        categorical_ids.size(0), -1, -1
    )
    encoder_input = torch.cat((queries, nodes), dim=1)
    query_mask = torch.zeros(
        categorical_ids.size(0),
        model.config.latent_tokens,
        dtype=torch.bool,
        device=padding_mask.device,
    )
    encoder_padding = torch.cat((query_mask, ~padding_mask), dim=1)
    encoded = model.encoder(
        encoder_input, src_key_padding_mask=encoder_padding
    )
    return model.to_codebook(
        encoded[:, :model.config.latent_tokens]
    ).detach().contiguous()


def _validation_diagnostic(model, partition, model_config, batch_size, device):
    family_teacher_logits = []
    family_teacher_targets = []
    family_teacher_predictions = []
    family_auto_logits = []
    family_auto_targets = []
    family_auto_predictions = []
    capsule_logit_argmax_count = 0
    capsule_generated_sketch_count = 0
    capsule_support = 0
    teacher_family_loss_sum = 0.0
    teacher_example_count = 0
    arm_rows = {name: [] for name in HYBRID_ARM_CONTRACTS}
    first_failure_records = []
    with torch.no_grad():
        for start in range(0, len(partition.flat_examples), batch_size):
            flat = partition.flat_examples[start:start + batch_size]
            physical = partition.physical_examples[start:start + batch_size]
            batch = collate_flat(flat)
            inputs = {
                name: value.to(device)
                for name, value in batch.to_torch(torch).items()
            }
            target = {
                name: value.to(device)
                for name, value in batch.target.to_torch(torch).items()
            }
            profiles = profile_targets_for_loss(
                batch.target, inputs["geometry"]
            )
            output = model(
                target=target, profile_targets=profiles, **inputs
            )
            losses = constrained_profile_v2_loss(
                output, target, profiles, model_config
            )
            teacher_family_loss_sum += float(
                losses.per_example["profile_family"].double().sum().item()
            )
            teacher_example_count += len(batch.family_ids)
            selected = profiles.sketch_mask
            selected_logits = output.profile_family_logits[selected]
            selected_targets = profiles.family_ids[selected]
            family_teacher_logits.extend(selected_logits.cpu().tolist())
            family_teacher_targets.extend(selected_targets.cpu().tolist())
            family_teacher_predictions.extend(
                selected_logits.argmax(-1).cpu().tolist()
            )
            node_counts = target["node_mask"].sum(dim=1)
            autonomous = greedy_decode_v2(
                model,
                inputs,
                node_counts=node_counts,
                node_count_source="authorized_validation_length",
            )
            arms = build_hybrid_arms(
                output, target, profiles, physical, autonomous
            )
            for index, example in enumerate(physical):
                results = {}
                for arm_name in HYBRID_ARM_CONTRACTS:
                    raw = arms[arm_name][index]
                    result = _convert_arm(
                        arm_name, raw, model_config.max_operations
                    )
                    results[arm_name] = result
                    arm_rows[arm_name].append((example, raw, result))
                record = _first_failure_record(
                    example,
                    arms["A"][index],
                    results["A"],
                    profiles.family_ids[index],
                    results,
                )
                first_failure_records.append(record)
                raw = arms["A"][index]
                for position, node in enumerate(example.nodes):
                    if node.node_type != "sketch":
                        continue
                    true_id = int(profiles.family_ids[index, position].item())
                    logits = raw.profile_family_logits[position]
                    family_auto_logits.append(logits)
                    family_auto_targets.append(true_id)
                    generated_sketch = (
                        raw.raw_nodes[position].node_type_id
                        == NODE_TYPES.id("sketch")
                    )
                    prediction = (
                        raw.predicted_profile_family_ids[position]
                        if generated_sketch else len(PROFILE_FAMILIES)
                    )
                    family_auto_predictions.append(prediction)
                    if true_id == 2:
                        capsule_support += 1
                        capsule_logit_argmax_count += int(
                            max(range(3), key=lambda item: logits[item]) == 2
                        )
                        capsule_generated_sketch_count += int(
                            generated_sketch
                        )
    teacher_summary = family_prediction_summary(
        family_teacher_logits,
        family_teacher_targets,
        family_teacher_predictions,
    )
    teacher_summary["family_loss"] = (
        teacher_family_loss_sum / float(teacher_example_count)
    )
    autonomous_summary = family_prediction_summary(
        family_auto_logits,
        family_auto_targets,
        family_auto_predictions,
    )
    autonomous_summary["capsule_logit_argmax_count"] = (
        capsule_logit_argmax_count
    )
    autonomous_summary["capsule_generated_sketch_count"] = (
        capsule_generated_sketch_count
    )
    autonomous_summary["capsule_support"] = capsule_support
    capsule_diagnosis = _capsule_zero_diagnosis(autonomous_summary)
    arm_summaries = {
        name: _arm_summary(name, rows)
        for name, rows in arm_rows.items()
    }
    return {
        "family_teacher_forced": teacher_summary,
        "family_autonomous": autonomous_summary,
        "capsule_zero_prediction_diagnosis": capsule_diagnosis,
        "arm_summaries": arm_summaries,
        "first_failure_records": first_failure_records,
        "first_failure_aggregate": _aggregate_first_failures(
            first_failure_records
        ),
    }


def _arm_summary(arm_name, rows):
    total = len(rows)
    valid = sum(result.controlled_domain.valid for _, _, result in rows)
    finite = sum(_raw_prediction_is_finite(raw) for _, raw, _ in rows)
    canonical = sum(
        _raw_profile_records_are_canonical(raw) for _, raw, _ in rows
    )
    converted = sum(
        result.reconstruction_target.valid for _, _, result in rows
    )
    failures = {}
    groupings = {
        "validity_by_operation_count": ({}, {}),
        "validity_by_node_count": ({}, {}),
        "validity_by_operation_family": ({}, {}),
        "validity_by_authoritative_profile_family": ({}, {}),
    }
    for example, unused_raw, result in rows:
        del unused_raw
        if result.primary_failure is not None:
            code = result.primary_failure.code
            failures[code] = failures.get(code, 0) + 1
        operation_template = OperationTemplate(
            example.metadata.operation_template
        )
        keys = (
            str(len(example.operation_sequence)),
            str(len(example.nodes)),
            "+".join(operation_template.operations),
            example.metadata.primitive_family,
        )
        for (totals, successes), key in zip(groupings.values(), keys):
            totals[key] = totals.get(key, 0) + 1
            if result.controlled_domain.valid:
                successes[key] = successes.get(key, 0) + 1
    result = {
        "arm": arm_name,
        "oracle_intervention": HYBRID_ARM_CONTRACTS[arm_name]["oracle"],
        "total_count": total,
        "valid_count": valid,
        "validity": valid / float(total),
        "finite_rate": finite / float(total),
        "canonical_profile_rate": canonical / float(total),
        "conversion_success_rate": converted / float(total),
        "failure_reason_histogram": failures,
    }
    for name, (totals, successes) in groupings.items():
        result[name] = _rates(totals, successes)
    _require_finite_json(result)
    return result


def _first_failure_record(
    example, raw, conversion, authoritative_family_ids, arm_results
):
    failures = (
        (() if conversion.primary_failure is None
         else (conversion.primary_failure,))
        + tuple(conversion.secondary_failures)
    )
    located = []
    for item in failures:
        match = _NODE_LOCATION.search(item.location)
        if match:
            located.append((int(match.group(1)), item))
    if located:
        position, failure = min(located, key=lambda item: item[0])
    else:
        position = -1
        failure = conversion.primary_failure
    predicted_node = (
        raw.raw_nodes[position] if 0 <= position < len(raw.raw_nodes) else None
    )
    authoritative_node = (
        example.nodes[position] if 0 <= position < len(example.nodes) else None
    )
    target_attributes = (
        example.target.categorical_attributes[position]
        if authoritative_node is not None else None
    )
    predicted_retained = (
        tuple(predicted_node.categorical_ids[index] for index in (0, 1, 2, 3, 8))
        if predicted_node is not None else None
    )
    authoritative_retained = (
        tuple(target_attributes[index] for index in (0, 1, 2, 3, 8))
        if target_attributes is not None else None
    )
    predicted_family = None
    authoritative_family = None
    if 0 <= position < len(raw.predicted_profile_family_ids):
        family_id = raw.predicted_profile_family_ids[position]
        predicted_family = (
            PROFILE_FAMILIES[family_id].value
            if 0 <= family_id < len(PROFILE_FAMILIES) else "non_sketch"
        )
    if (
        authoritative_node is not None
        and authoritative_node.node_type == "sketch"
    ):
        family_id = int(authoritative_family_ids[position].item())
        authoritative_family = PROFILE_FAMILIES[family_id].value
    mismatches = []
    if predicted_retained is not None and authoritative_retained is not None:
        names = (
            "operation_type", "boolean_mode", "direction",
            "reference_plane", "loop_role",
        )
        mismatches = [
            name for name, predicted, authoritative in zip(
                names, predicted_retained, authoritative_retained
            ) if predicted != authoritative
        ]
    operation_index = None
    if position >= 0:
        operation_types = {"extrude", "revolve"}
        prior = [
            node for node in example.nodes[:position + 1]
            if node.node_type in operation_types
        ]
        operation_index = len(prior) - 1 if prior else None
    return {
        "physical_family_id": example.physical_family_id,
        "earliest_failing_node_position": position,
        "predicted_node_type": (
            NODE_TYPES.tokens[predicted_node.node_type_id]
            if predicted_node is not None else None
        ),
        "authoritative_node_type": (
            authoritative_node.node_type
            if authoritative_node is not None else None
        ),
        "predicted_retained_categories": predicted_retained,
        "authoritative_retained_categories": authoritative_retained,
        "applicability_field_mismatches": mismatches,
        "predicted_family": predicted_family,
        "authoritative_family": authoritative_family,
        "generated_prefix_length": max(position, 0),
        "failure_code": failure.code if failure else "valid",
        "failure_stage": failure.stage if failure else None,
        "failure_location": failure.location if failure else None,
        "reference_plane_failure_subtype": (
            "basis_or_origin"
            if failure and failure.code == "invalid_reference_plane_geometry"
            else None
        ),
        "operation_index": operation_index,
        "operation_count": len(example.operation_sequence),
        "hybrid_arm_success": {
            name: bool(result.controlled_domain.valid)
            for name, result in arm_results.items()
        },
    }


def _aggregate_first_failures(records):
    positions = {}
    node_confusion = {}
    applicability = {}
    reference_subtypes = {}
    operation_counts = {}
    failures_by_operation_count = {}
    for record in records:
        _increment(positions, str(record["earliest_failing_node_position"]))
        key = "{}->{}".format(
            record["authoritative_node_type"], record["predicted_node_type"]
        )
        _increment(node_confusion, key)
        for name in record["applicability_field_mismatches"]:
            _increment(applicability, name)
        subtype = record["reference_plane_failure_subtype"]
        if subtype is not None:
            _increment(reference_subtypes, subtype)
        _increment(operation_counts, str(record["operation_count"]))
        operation_key = str(record["operation_count"])
        failures = failures_by_operation_count.setdefault(operation_key, {})
        _increment(failures, record["failure_code"])
    return {
        "record_count": len(records),
        "earliest_failing_node_position_histogram": positions,
        "node_type_confusion_at_failure": node_confusion,
        "applicability_field_mismatch_counts": applicability,
        "reference_plane_failure_subtypes": reference_subtypes,
        "operation_count_histogram": operation_counts,
        "failure_histogram_by_operation_count": failures_by_operation_count,
    }


def _replace_oracle_relations(prediction, example):
    index = {node.node_id: position for position, node in enumerate(example.nodes)}
    edge_lookup = {
        (index[edge.source_id], index[edge.target_id]): EDGE_TYPES.id(edge.edge_type)
        for edge in example.edges
    }
    edges = []
    for source in range(len(example.nodes)):
        for target in range(len(example.nodes)):
            edge_type = edge_lookup.get((source, target), EDGE_TYPES.id(None))
            present = (source, target) in edge_lookup
            edges.append(RawDecodedEdge(
                source, target, 1.0 if present else -1.0, present, edge_type
            ))
    operation_nodes = tuple(index[item] for item in example.operation_sequence)
    pointers = tuple(
        RawDecodedPointer(query, node_index, 1.0)
        for query, node_index in enumerate(operation_nodes)
    )
    return replace(
        prediction,
        raw_edges=tuple(edges),
        predicted_operation_node_indices=operation_nodes,
        predicted_operation_count=len(operation_nodes),
        operation_count_exceeds_limit=False,
        raw_operation_pointers=pointers,
    )


def _replace_autonomous_family(prediction, authoritative_family_ids):
    family_ids = list(prediction.predicted_profile_family_ids)
    parameters = list(prediction.constrained_profile_parameters)
    nodes = list(prediction.raw_nodes)
    sketch_id = NODE_TYPES.id("sketch")
    for position, node in enumerate(nodes):
        if node.node_type_id != sketch_id:
            continue
        authoritative = int(authoritative_family_ids[position].item())
        if not 0 <= authoritative < len(PROFILE_FAMILIES):
            continue
        raw_parameters = torch.tensor(
            prediction.raw_profile_parameters[position], dtype=torch.float32
        ).unsqueeze(0)
        family = torch.tensor([authoritative], dtype=torch.long)
        sketch = torch.tensor([True], dtype=torch.bool)
        constrained = constrain_profile_parameters(
            raw_parameters, family, sketch
        )
        canonical = canonicalize_profile_tensors(
            family, constrained, sketch
        )
        categories = list(node.categorical_ids)
        categories[4:8] = canonical.primitive_type_ids[0].tolist()
        geometry = list(node.normalized_geometry)
        geometry[9:33] = canonical.geometry[0, 9:33].tolist()
        mask = derive_geometry_mask(node.node_type_id, tuple(categories))
        nodes[position] = RawDecodedNode(
            node.position,
            node.node_type_id,
            tuple(categories),
            tuple(float(value) for value in geometry),
            mask,
        )
        family_ids[position] = authoritative
        parameters[position] = tuple(
            float(value) for value in constrained[0].tolist()
        )
    return replace(
        prediction,
        raw_nodes=tuple(nodes),
        predicted_profile_family_ids=tuple(family_ids),
        constrained_profile_parameters=tuple(parameters),
    )


def _convert_arm(arm_name, raw, max_operations):
    if arm_name in ("A", "F"):
        return validate_and_convert_v2_autonomous_prediction(
            raw, max_operations=max_operations
        )
    return validate_and_convert_v2_teacher_forced_prediction(
        raw, max_operations=max_operations
    )


def _capsule_zero_diagnosis(summary):
    capsule_predictions = summary["predicted_class_histogram"][
        PROFILE_FAMILIES[2].value
    ]
    if capsule_predictions:
        return "capsule_was_predicted"
    logits_never_maximal = summary["capsule_logit_argmax_count"] == 0
    nodes_suppressed = (
        summary["capsule_generated_sketch_count"]
        < summary["capsule_support"]
    )
    if logits_never_maximal and nodes_suppressed:
        return "both"
    if logits_never_maximal:
        return "capsule_logits_never_maximal"
    if nodes_suppressed:
        return "generated_node_types_suppress_capsule_positions"
    return "another_reason"


def _optional_epoch1_comparison(
    supplied_path, epoch2_path, config, model_config, data, device
):
    path = (
        Path(supplied_path)
        if supplied_path is not None
        else Path(epoch2_path).with_name("epoch-0001.pt")
    )
    if not path.is_file():
        return {"available": False, "path": str(path)}
    checkpoint = torch.load(str(path), map_location="cpu")
    audit_frozen_checkpoint_payload(
        checkpoint, config, expected_epoch=1, expected_global_step=68
    )
    construction_rng = capture_rng_state(torch)
    model = ConstrainedProfileV2Model(model_config).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    restore_rng_state(construction_rng, torch)
    model.eval()
    return {
        "available": True,
        "path": str(path),
        "checkpoint_sha256": _sha256(path),
        "train": _vq_partition_summary(
            model, data.train, config.batch_size, device
        ),
        "iid_validation": _vq_partition_summary(
            model, data.validation, config.batch_size, device
        ),
    }


def _assert_explicit_vq_state(model, expected):
    actual = model.vq.state_dict()
    if (
        not isinstance(expected, dict)
        or list(expected) != list(actual)
        or any(not torch.equal(expected[name], actual[name]) for name in actual)
    ):
        raise ConstrainedV2PilotDiagnosticError(
            "incompatible_checkpoint", "explicit VQ state differs"
        )


def _classification_rules():
    return {
        "material_validity_improvement": MATERIAL_VALIDITY_IMPROVEMENT,
        "high_oracle_discrete_validity": HIGH_ORACLE_DISCRETE_VALIDITY,
        "low_autonomous_validity": LOW_AUTONOMOUS_VALIDITY,
        "encoder_variance_threshold": ENCODER_VARIANCE_THRESHOLD,
        "encoder_distance_threshold": ENCODER_DISTANCE_THRESHOLD,
        "codebook_norm_std_threshold": CODEBOOK_NORM_STD_THRESHOLD,
        "ema_live_threshold": EMA_LIVE_THRESHOLD,
        "vq_causality_claimed": False,
    }


def _tensor_distribution(values):
    values = values.double().reshape(-1)
    if values.numel() == 0:
        return {"count": 0, "min": 0.0, "mean": 0.0,
                "median": 0.0, "max": 0.0, "std": 0.0}
    return {
        "count": int(values.numel()),
        "min": float(values.min().item()),
        "mean": float(values.mean().item()),
        "median": float(values.median().item()),
        "max": float(values.max().item()),
        "std": float(values.std(unbiased=False).item()),
    }


def _rates(totals, successes):
    return {
        key: {
            "count": totals[key],
            "valid_count": successes.get(key, 0),
            "validity": successes.get(key, 0) / float(totals[key]),
        }
        for key in sorted(totals)
    }


def _increment(values, key):
    values[key] = values.get(key, 0) + 1


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _clone_tree(value):
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, dict):
        return {name: _clone_tree(item) for name, item in value.items()}
    if isinstance(value, list):
        return [_clone_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_tree(item) for item in value)
    return value


def _tree_equal(expected, actual):
    if torch.is_tensor(expected):
        return (
            torch.is_tensor(actual)
            and expected.dtype == actual.dtype
            and expected.shape == actual.shape
            and expected.device == actual.device
            and torch.equal(expected, actual)
        )
    if isinstance(expected, dict):
        return (
            isinstance(actual, dict)
            and list(expected) == list(actual)
            and all(_tree_equal(expected[name], actual[name]) for name in expected)
        )
    if isinstance(expected, (list, tuple)):
        return (
            type(expected) is type(actual)
            and len(expected) == len(actual)
            and all(_tree_equal(a, b) for a, b in zip(expected, actual))
        )
    return type(expected) is type(actual) and expected == actual


def _vq_matches_snapshot(model, snapshot):
    return all(
        _tree_equal(snapshot["vq." + name], value)
        for name, value in model.vq.state_dict().items()
    )


def _require_finite_json(value, path="$"):
    if value is None or isinstance(value, (bool, str)):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ConstrainedV2PilotDiagnosticError(
                "nonfinite_json", "{} is nonfinite".format(path)
            )
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _require_finite_json(item, "{}[{}]".format(path, index))
        return
    if isinstance(value, dict):
        for name, item in value.items():
            _require_finite_json(item, "{}.{}".format(path, name))
        return
    raise ConstrainedV2PilotDiagnosticError(
        "invalid_json_value", "{} has unsupported type".format(path)
    )


def _append_terminal_failure(config, exc):
    if getattr(exc, "code", None) == "output_collision":
        return
    path = Path(config.output_dir) / "metrics.jsonl"
    if not path.is_file():
        return
    try:
        records = path.read_text(encoding="utf-8").splitlines()
        first = json.loads(records[0])
        if (
            first.get("event") != "diagnostic_metadata"
            or first.get("diagnostic_config", {}).get(
                "diagnostic_identity"
            ) != config.diagnostic_identity
        ):
            return
        if json.loads(records[-1]).get("event") in (
            "terminal_success", "terminal_failure"
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
