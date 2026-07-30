"""Train-fixture-only diagnosis of V2 profile-family joint optimization."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import sys
import tempfile

import torch
from torch.nn import functional as F

from prototype.constrained_profile_decoder import (
    profile_family_loss,
    profile_targets_for_loss,
)
from prototype.flat_baseline.constrained_v2 import (
    ConstrainedProfileV2Model,
)
from prototype.flat_baseline.constrained_v2_config import (
    ConstrainedProfileV2Config,
)
from prototype.flat_baseline.constrained_v2_losses import (
    constrained_profile_v2_loss,
)
from prototype.flat_baseline.constrained_v2_training import (
    build_v2_optimizer,
    constrained_v2_training_step,
    load_v2_tiny_training_selection,
)
from prototype.flat_baseline.constrained_v2_training_config import (
    ConstrainedV2TrainingConfig,
)
from prototype.flat_baseline.tests.test_constrained_v2_training import (
    _sources,
)
from prototype.model_data.batching import collate_flat
from prototype.model_data.tests.fixtures import write_physical_corpus
from prototype.model_data.vocab import NODE_TYPES
from prototype.profile_geometry import PROFILE_FAMILIES


CHECKPOINT_STEPS = (0, 1, 5, 10, 25, 50, 100, 200, 500)
CONFLICT_STEPS = (0, 10, 100)
REFERENCE_INITIAL_FAMILY_LOSS = 1.084914207458496
REFERENCE_STEP_100_FAMILY_LOSS = 1.4846277236938477
ARM_NAMES = (
    "A_cached_features_family_head_only",
    "B_full_family_only_frozen_vq",
    "B_full_family_only_production_vq",
    "C_full_objective_frozen_vq",
    "D_full_production_objective",
)


class DiagnosticFailure(RuntimeError):
    pass


class JsonlEmitter:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("x", encoding="utf-8")

    def emit(self, payload):
        line = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        self.stream.write(line + "\n")
        self.stream.flush()
        print(line, flush=True)

    def close(self):
        self.stream.close()


class DiagnosticContext:
    def __init__(self, root):
        self.corpus = Path(root) / "corpus"
        write_physical_corpus(self.corpus, _sources())
        self.training_config = ConstrainedV2TrainingConfig(
            seed=37,
            batch_size=8,
            learning_rate=1e-3,
            maximum_steps=500,
            logging_cadence=10,
            checkpoint_cadence=500,
            output_dir=str(Path(root) / "unused-run"),
            require_clean_source=False,
        )
        self.model_config = ConstrainedProfileV2Config(
            model_dim=32,
            num_heads=4,
            feedforward_dim=64,
            encoder_layers=1,
            decoder_layers=1,
            dropout=0.0,
            max_nodes=10,
            max_operations=2,
            latent_tokens=2,
            codebook_size=16,
            codebook_dim=16,
            edge_pair_dim=32,
        )
        selection = load_v2_tiny_training_selection(
            self.corpus, self.training_config
        )
        self.selected_ids = selection.selected_example_ids
        self.batch = collate_flat(selection.examples)
        tensor_batch = self.batch.to_torch(torch)
        self.inputs = dict(tensor_batch)
        self.target = self.batch.target.to_torch(torch)
        self.profile_targets = profile_targets_for_loss(
            self.batch.target, self.inputs["geometry"]
        )
        torch.manual_seed(self.training_config.seed)
        initial_model = ConstrainedProfileV2Model(self.model_config)
        self.initial_state = copy.deepcopy(initial_model.state_dict())
        self.reference_family_ids = (
            self.profile_targets.family_ids.detach().clone()
        )
        self.reference_sketch_mask = (
            self.profile_targets.sketch_mask.detach().clone()
        )

    def clone_model(self):
        torch.manual_seed(self.training_config.seed)
        model = ConstrainedProfileV2Model(self.model_config)
        model.load_state_dict(self.initial_state, strict=True)
        return model


def _forward(model, context):
    return model(
        target=context.target,
        profile_targets=context.profile_targets,
        **context.inputs
    )


def _losses(model, context):
    output = _forward(model, context)
    losses = constrained_profile_v2_loss(
        output,
        context.target,
        context.profile_targets,
        context.model_config,
    )
    return output, losses


def _configured_per_example_total(losses, config):
    return (
        config.node_type_loss_weight * losses.per_example["node_type"]
        + config.categorical_loss_weight
        * losses.per_example["categorical_attributes"]
        + config.geometry_loss_weight
        * losses.per_example["non_profile_geometry"]
        + config.profile_family_loss_weight
        * losses.per_example["profile_family"]
        + config.profile_parameter_loss_weight
        * losses.per_example["profile_parameter"]
        + config.edge_presence_loss_weight
        * losses.per_example["edge_presence"]
        + config.edge_type_loss_weight
        * losses.per_example["edge_type"]
        + config.operation_pointer_loss_weight
        * losses.per_example["operation_pointer"]
        + config.vq_loss_weight * losses.per_example["vq_commitment"]
    )


def _tensor_l2(value):
    return float(value.detach().double().norm().cpu().item())


def _vq_reference(model, output, prequant):
    return {
        "embedding": model.vq.embedding.detach().clone(),
        "ema_cluster_size": model.vq.ema_cluster_size.detach().clone(),
        "ema_weight": model.vq.ema_weight.detach().clone(),
        "assignments": output.code_indices.detach().clone(),
        "prequant": prequant.detach().clone(),
        "quantized_memory": output.quantized_memory.detach().clone(),
    }


def _vq_drift(model, output, prequant, reference):
    assignment_changes = (
        output.code_indices.detach() != reference["assignments"]
    ).long().sum(dim=1)
    return {
        "codebook_change_l2": _tensor_l2(
            model.vq.embedding - reference["embedding"]
        ),
        "ema_cluster_size_change_l2": _tensor_l2(
            model.vq.ema_cluster_size - reference["ema_cluster_size"]
        ),
        "ema_weight_change_l2": _tensor_l2(
            model.vq.ema_weight - reference["ema_weight"]
        ),
        "assignment_changes_by_example": [
            int(value) for value in assignment_changes.cpu().tolist()
        ],
        "code_assignments_by_example": [
            [int(value) for value in row]
            for row in output.code_indices.detach().cpu().tolist()
        ],
        "prequantized_vector_change_l2": _tensor_l2(
            prequant - reference["prequant"]
        ),
        "quantized_memory_change_l2": _tensor_l2(
            output.quantized_memory - reference["quantized_memory"]
        ),
    }


def _loss_metrics(losses):
    return {
        "total_loss": float(losses.total.detach().cpu().item()),
        "profile_family_loss": float(
            losses.profile_family.detach().cpu().item()
        ),
        "profile_parameter_loss": float(
            losses.profile_parameter.detach().cpu().item()
        ),
        "node_type_loss": float(losses.node_type.detach().cpu().item()),
        "remaining_categorical_loss": float(
            losses.categorical_attributes.detach().cpu().item()
        ),
        "non_profile_geometry_loss": float(
            losses.non_profile_geometry.detach().cpu().item()
        ),
        "edge_presence_loss": float(
            losses.edge_presence.detach().cpu().item()
        ),
        "edge_type_loss": float(losses.edge_type.detach().cpu().item()),
        "operation_pointer_loss": float(
            losses.operation_pointer.detach().cpu().item()
        ),
        "vq_commitment_loss": float(
            losses.vq_commitment.detach().cpu().item()
        ),
    }


def _family_metrics(output, context):
    mask = context.profile_targets.sketch_mask
    targets = context.profile_targets.family_ids
    logits = output.profile_family_logits
    selected_logits = logits[mask]
    selected_targets = targets[mask]
    predicted = selected_logits.argmax(dim=-1)
    probabilities = torch.softmax(selected_logits, dim=-1)
    entropy = -(
        probabilities * torch.log_softmax(selected_logits, dim=-1)
    ).sum(dim=-1).mean()
    per_class = {}
    for class_id, family in enumerate(PROFILE_FAMILIES):
        selected = selected_targets == class_id
        per_class[family.value] = {
            "count": int(selected.long().sum().item()),
            "cross_entropy": float(F.cross_entropy(
                selected_logits[selected],
                selected_targets[selected],
            ).detach().cpu().item()),
            "accuracy": float(
                (predicted[selected] == selected_targets[selected])
                .float().mean().detach().cpu().item()
            ),
        }
    return {
        "family_accuracy": float(
            (predicted == selected_targets)
            .float().mean().detach().cpu().item()
        ),
        "family_logit_entropy": float(entropy.detach().cpu().item()),
        "per_class": per_class,
    }


def _alignment_metrics(context, output):
    extracted = profile_targets_for_loss(
        context.batch.target, context.inputs["geometry"]
    )
    mask = extracted.sketch_mask
    family_ids = extracted.family_ids
    target_node_mask = context.target["node_mask"]
    target_sketch_mask = (
        target_node_mask
        & (
            context.target["node_type_ids"]
            == NODE_TYPES.id("sketch")
        )
    )
    return {
        "sketch_positions_unchanged": bool(torch.equal(
            mask, context.reference_sketch_mask
        )),
        "family_target_ids_unchanged": bool(torch.equal(
            family_ids, context.reference_family_ids
        )),
        "family_class_order_unchanged": (
            tuple(item.value for item in PROFILE_FAMILIES)
            == context.model_config.profile_family_order
        ),
        "target_extraction_deterministic": bool(
            torch.equal(mask, context.profile_targets.sketch_mask)
            and torch.equal(family_ids, context.profile_targets.family_ids)
        ),
        "only_real_sketches_selected": bool(
            (mask & ~target_node_mask).long().sum().item() == 0
        ),
        "sketch_mask_matches_target_node_types": bool(
            torch.equal(mask, target_sketch_mask)
        ),
        "non_sketch_family_ids_are_sentinel": bool(
            (family_ids[~mask] == -1).all().item()
        ),
        "family_applicable_count": int(mask.long().sum().item()),
        "logit_node_alignment_shape": [
            int(value)
            for value in output.profile_family_logits.shape
        ],
        "logit_node_alignment_valid": (
            output.profile_family_logits.shape
            == family_ids.shape + (len(PROFILE_FAMILIES),)
        ),
        "sketch_positions_by_example": [
            [
                int(value)
                for value in torch.nonzero(row, as_tuple=False)
                .reshape(-1).cpu().tolist()
            ]
            for row in mask
        ],
        "family_target_ids_by_example": [
            [int(value) for value in family_ids[index][mask[index]].cpu().tolist()]
            for index in range(mask.size(0))
        ],
    }


def deterministic_snapshot(model, context, vq_reference=None):
    was_training = model.training
    vq_was_training = model.vq.training
    captured = []

    def capture(unused_module, unused_inputs, output):
        del unused_module, unused_inputs
        captured.append(output.detach().clone())

    handle = model.to_codebook.register_forward_hook(capture)
    try:
        model.eval()
        with torch.no_grad():
            output, losses = _losses(model, context)
    finally:
        handle.remove()
        model.train(was_training)
        model.vq.train(vq_was_training)
    if len(captured) != 1:
        raise DiagnosticFailure("prequantized vectors were not captured once")
    metrics = {
        **_loss_metrics(losses),
        **_family_metrics(output, context),
        **_alignment_metrics(context, output),
        "active_vq_code_count": int(output.active_code_count.item()),
        "vq_perplexity": float(output.codebook_perplexity.item()),
    }
    if vq_reference is not None:
        metrics["vq_drift"] = _vq_drift(
            model, output, captured[0], vq_reference
        )
    return metrics, _vq_reference(model, output, captured[0])


def verify_initial_arithmetic(model, context):
    model.eval()
    output, losses = _losses(model, context)
    configured = _configured_per_example_total(
        losses, context.model_config
    )
    exact_per_example = torch.equal(
        configured, losses.per_example["total"]
    )
    exact_reported = torch.equal(
        losses.total, losses.per_example["total"].mean()
    )
    family_gradient = torch.autograd.grad(
        losses.total, output.profile_family_logits
    )[0]
    return {
        "exact_configured_per_example_sum": exact_per_example,
        "exact_reported_total_mean": exact_reported,
        "reported_total_loss": float(losses.total.detach().item()),
        "configured_total_loss": float(configured.mean().detach().item()),
        "absolute_difference": float(
            (losses.total - configured.mean()).abs().detach().item()
        ),
        "profile_family_weight": (
            context.model_config.profile_family_loss_weight
        ),
        "profile_family_total_gradient_norm": _tensor_l2(
            family_gradient
        ),
        "profile_family_contribution_positive": bool(
            context.model_config.profile_family_loss_weight > 0.0
            and family_gradient.norm().item() > 0.0
        ),
    }


def detached_logit_direction_check(model, context):
    model.eval()
    with torch.no_grad():
        output = _forward(model, context)
    mask = context.profile_targets.sketch_mask
    targets = context.profile_targets.family_ids[mask]
    logits = output.profile_family_logits[mask].detach().clone()
    logits.requires_grad_(True)
    before = F.cross_entropy(logits, targets)
    gradient = torch.autograd.grad(before, logits)[0]
    step_size = 1e-4
    after = F.cross_entropy(logits - step_size * gradient, targets)
    return {
        "step_size": step_size,
        "before": float(before.detach().item()),
        "after": float(after.detach().item()),
        "gradient_norm": _tensor_l2(gradient),
        "decreased": bool(after.item() < before.item()),
    }


def full_family_direction_check(context):
    model = context.clone_model()
    model.eval()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)
    optimizer.zero_grad()
    output = _forward(model, context)
    term = profile_family_loss(
        output.profile_family_logits,
        context.profile_targets.family_ids,
        context.profile_targets.sketch_mask,
    )
    before = float(term.loss.detach().item())
    term.loss.backward()
    gradient_norm = math.sqrt(sum(
        float(parameter.grad.detach().double().pow(2).sum().item())
        for parameter in model.parameters()
        if parameter.grad is not None
    ))
    optimizer.step()
    with torch.no_grad():
        after_output = _forward(model, context)
        after = profile_family_loss(
            after_output.profile_family_logits,
            context.profile_targets.family_ids,
            context.profile_targets.sketch_mask,
        ).loss
    return {
        "step_size": 1e-5,
        "before": before,
        "after": float(after.item()),
        "gradient_norm": gradient_norm,
        "decreased": bool(after.item() < before),
    }


def _require_finite_model(model):
    for name, value in tuple(model.named_parameters()) + tuple(
        model.named_buffers()
    ):
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise DiagnosticFailure(
                "nonfinite model state {!r}".format(name)
            )


def _diagnostic_step(model, optimizer, context, objective, freeze_vq):
    model.train()
    if freeze_vq:
        model.vq.eval()
    optimizer.zero_grad()
    output, losses = _losses(model, context)
    if objective == "family":
        optimized_loss = (
            context.model_config.profile_family_loss_weight
            * losses.profile_family
        )
    elif objective == "full":
        optimized_loss = losses.total
    else:
        raise AssertionError(objective)
    if not torch.isfinite(optimized_loss):
        raise DiagnosticFailure("nonfinite optimized loss")
    optimized_loss.backward()
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    if not gradients or any(
        not torch.isfinite(gradient).all() for gradient in gradients
    ):
        raise DiagnosticFailure("missing or nonfinite diagnostic gradients")
    norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), context.training_config.gradient_clip_norm
    )
    if not torch.isfinite(norm):
        raise DiagnosticFailure("nonfinite diagnostic gradient norm")
    optimizer.step()
    _require_finite_model(model)


def _cached_family_step(model, optimizer, features, context):
    model.profile_heads.family_head.train()
    optimizer.zero_grad()
    logits = model.profile_heads.family_head(features)
    loss = profile_family_loss(
        logits,
        context.profile_targets.family_ids,
        context.profile_targets.sketch_mask,
    ).loss
    loss.backward()
    optimizer.step()


def _flatten_gradients(gradients, parameters):
    return torch.cat(tuple(
        (
            torch.zeros_like(parameter)
            if gradient is None else gradient
        ).reshape(-1)
        for gradient, parameter in zip(gradients, parameters)
    ))


def _gradient_pair_metrics(first, second):
    first_norm = first.double().norm()
    second_norm = second.double().norm()
    dot = torch.dot(first.double(), second.double())
    denominator = first_norm * second_norm
    cosine = dot / denominator if denominator.item() else dot * 0.0
    return {
        "family_gradient_norm": float(first_norm.item()),
        "other_gradient_norm": float(second_norm.item()),
        "dot_product": float(dot.item()),
        "cosine_similarity": float(cosine.item()),
        "combined_gradient_norm": float(
            (first.double() + second.double()).norm().item()
        ),
    }


def gradient_conflict_metrics(model, context):
    was_training = model.training
    vq_was_training = model.vq.training
    try:
        model.eval()
        output, losses = _losses(model, context)
        family = (
            context.model_config.profile_family_loss_weight
            * losses.profile_family
        )
        other = losses.total - family
        named_parameters = tuple(
            (name, parameter)
            for name, parameter in model.named_parameters()
            if name.startswith("decoder.")
        )
        parameters = tuple(item[1] for item in named_parameters)
        family_gradients = torch.autograd.grad(
            family,
            parameters,
            retain_graph=True,
            allow_unused=True,
        )
        other_gradients = torch.autograd.grad(
            other,
            parameters,
            retain_graph=True,
            allow_unused=True,
        )
        family_state_gradient = torch.autograd.grad(
            family,
            output.decoded_states,
            retain_graph=True,
        )[0]
        other_state_gradient = torch.autograd.grad(
            other,
            output.decoded_states,
        )[0]
    finally:
        model.train(was_training)
        model.vq.train(vq_was_training)
    parameter_metrics = _gradient_pair_metrics(
        _flatten_gradients(family_gradients, parameters),
        _flatten_gradients(other_gradients, parameters),
    )
    state_metrics = _gradient_pair_metrics(
        family_state_gradient.reshape(-1),
        other_state_gradient.reshape(-1),
    )
    return {
        "shared_parameter_scope": [
            name for name, unused in named_parameters
        ],
        "shared_decoder_parameters": parameter_metrics,
        "decoded_node_states": state_metrics,
    }


def run_arm(name, context, emitter):
    model = context.clone_model()
    optimizer = None
    cached_features = None
    if name == "A_cached_features_family_head_only":
        model.eval()
        with torch.no_grad():
            cached_features = _forward(
                model, context
            ).decoded_states.detach()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for parameter in model.profile_heads.family_head.parameters():
            parameter.requires_grad_(True)
        optimizer = torch.optim.AdamW(
            model.profile_heads.family_head.parameters(),
            lr=context.training_config.learning_rate,
            weight_decay=context.training_config.weight_decay,
        )
    else:
        optimizer = build_v2_optimizer(
            model, context.training_config
        )

    trajectories = {}
    conflicts = {}
    baseline_vq = None
    for step in range(0, CHECKPOINT_STEPS[-1] + 1):
        if step in CHECKPOINT_STEPS:
            snapshot, current_vq = deterministic_snapshot(
                model, context, baseline_vq
            )
            if baseline_vq is None:
                baseline_vq = current_vq
                snapshot["vq_drift"] = {
                    "codebook_change_l2": 0.0,
                    "ema_cluster_size_change_l2": 0.0,
                    "ema_weight_change_l2": 0.0,
                    "assignment_changes_by_example": [0] * 8,
                    "code_assignments_by_example": [
                        [int(value) for value in row]
                        for row in current_vq["assignments"].cpu().tolist()
                    ],
                    "prequantized_vector_change_l2": 0.0,
                    "quantized_memory_change_l2": 0.0,
                }
            trajectories[step] = snapshot
            emitter.emit({
                "event": "arm_snapshot",
                "arm": name,
                "step": step,
                **snapshot,
            })
        if (
            name == "D_full_production_objective"
            and step in CONFLICT_STEPS
        ):
            conflicts[step] = gradient_conflict_metrics(model, context)
            emitter.emit({
                "event": "gradient_conflict",
                "arm": name,
                "step": step,
                **conflicts[step],
            })
        if step == CHECKPOINT_STEPS[-1]:
            break
        next_step = step + 1
        if name == "A_cached_features_family_head_only":
            _cached_family_step(
                model, optimizer, cached_features, context
            )
        elif name == "B_full_family_only_frozen_vq":
            _diagnostic_step(
                model, optimizer, context, "family", True
            )
        elif name == "B_full_family_only_production_vq":
            _diagnostic_step(
                model, optimizer, context, "family", False
            )
        elif name == "C_full_objective_frozen_vq":
            _diagnostic_step(
                model, optimizer, context, "full", True
            )
        elif name == "D_full_production_objective":
            constrained_v2_training_step(
                model,
                optimizer,
                context.batch,
                context.model_config,
                context.training_config,
                torch.device("cpu"),
                global_step=next_step,
                examples_processed=next_step * 8,
            )
        else:
            raise AssertionError(name)
    return trajectories, conflicts


def classify(
    trajectories,
    conflicts,
    arithmetic,
    logit_check,
    family_check,
):
    initial = trajectories["D_full_production_objective"][0][
        "profile_family_loss"
    ]
    production_100 = trajectories["D_full_production_objective"][100][
        "profile_family_loss"
    ]
    production_500 = trajectories["D_full_production_objective"][500][
        "profile_family_loss"
    ]
    cached_final = trajectories[
        "A_cached_features_family_head_only"
    ][500]["profile_family_loss"]
    frozen_family_final = trajectories[
        "B_full_family_only_frozen_vq"
    ][500]["profile_family_loss"]
    ema_family_final = trajectories[
        "B_full_family_only_production_vq"
    ][500]["profile_family_loss"]
    frozen_full_100 = trajectories[
        "C_full_objective_frozen_vq"
    ][100]["profile_family_loss"]
    frozen_full_500 = trajectories[
        "C_full_objective_frozen_vq"
    ][500]["profile_family_loss"]
    production_drift = trajectories[
        "D_full_production_objective"
    ][100]["vq_drift"]
    frozen_drift = trajectories[
        "C_full_objective_frozen_vq"
    ][100]["vq_drift"]
    production_conflicts = conflicts[
        "D_full_production_objective"
    ]
    conflict_cosines = tuple(
        value[scope]["cosine_similarity"]
        for value in production_conflicts.values()
        for scope in (
            "shared_decoder_parameters",
            "decoded_node_states",
        )
    )
    all_alignment = all(
        snapshot["sketch_positions_unchanged"]
        and snapshot["family_target_ids_unchanged"]
        and snapshot["family_class_order_unchanged"]
        and snapshot["target_extraction_deterministic"]
        and snapshot["only_real_sketches_selected"]
        and snapshot["sketch_mask_matches_target_node_types"]
        and snapshot["non_sketch_family_ids_are_sentinel"]
        and snapshot["logit_node_alignment_valid"]
        and snapshot["family_applicable_count"] == 11
        for arm in trajectories.values()
        for snapshot in arm.values()
    )
    if (
        not arithmetic["exact_configured_per_example_sum"]
        or not arithmetic["exact_reported_total_mean"]
        or not arithmetic["profile_family_contribution_positive"]
        or not logit_check["decreased"]
        or not family_check["decreased"]
        or cached_final >= initial
        or not all_alignment
    ):
        return {
            "classification": "production_defect",
            "proposed_correction": (
                "review the failing arithmetic, gradient-direction, cached-head, "
                "or target-alignment invariant reported by this diagnostic"
            ),
        }
    if (
        frozen_full_100 < initial
        and production_100 > initial
        and frozen_family_final < initial
        and production_drift["codebook_change_l2"] > 0.0
        and sum(
            production_drift["assignment_changes_by_example"]
        ) > 0
        and frozen_drift["codebook_change_l2"] == 0.0
    ):
        return {
            "classification": "vq_ema_interference",
            "proposed_correction": (
                "review a bounded VQ-EMA freeze or warmup policy before changing "
                "the production workflow"
            ),
        }
    if (
        frozen_family_final < initial
        and ema_family_final < initial
        and frozen_full_500 >= initial
        and min(conflict_cosines) < -0.1
    ):
        return {
            "classification": "general_multi_objective_interference",
            "proposed_correction": (
                "review objective normalization or a bounded family warmup; do "
                "not change weights without a separately reviewed experiment"
            ),
        }
    if (
        production_100 > initial
        and production_500 < initial
    ):
        return {
            "classification": "insufficient_optimization_horizon",
            "proposed_correction": (
                "extend only the bounded regression-test horizon to the first "
                "predeclared checkpoint at or below 500 that beats step zero"
            ),
        }
    return {
        "classification": "joint_optimization_behavior_unclassified",
        "proposed_correction": (
            "review the emitted VQ-drift and gradient-conflict evidence before "
            "changing any production policy"
        ),
            "frozen_full_step_500_family_loss": frozen_full_500,
            "minimum_observed_conflict_cosine": min(conflict_cosines),
        }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-jsonl", required=True)
    args = parser.parse_args(argv)
    emitter = JsonlEmitter(args.output_jsonl)
    try:
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as temporary:
            context = DiagnosticContext(temporary)
            emitter.emit({
                "event": "diagnostic_metadata",
                "seed": context.training_config.seed,
                "device": "cpu",
                "dtype": "float32",
                "python_version": sys.version.split()[0],
                "pytorch_version": torch.__version__,
                "checkpoints": list(CHECKPOINT_STEPS),
                "selected_example_ids": list(context.selected_ids),
                "training_partition": "synthetic_train_fixture",
                "validation_partition": "none",
                "test_partition_accessed": False,
            })
            initial_model = context.clone_model()
            arithmetic = verify_initial_arithmetic(
                initial_model, context
            )
            logit_check = detached_logit_direction_check(
                initial_model, context
            )
            family_check = full_family_direction_check(context)
            emitter.emit({
                "event": "initial_invariants",
                "loss_arithmetic": arithmetic,
                "detached_logit_direction": logit_check,
                "full_model_family_direction": family_check,
            })
            trajectories = {}
            conflicts = {}
            for arm in ARM_NAMES:
                trajectories[arm], conflicts[arm] = run_arm(
                    arm, context, emitter
                )
            production = trajectories[
                "D_full_production_objective"
            ]
            reproduced = {
                "initial_reference": REFERENCE_INITIAL_FAMILY_LOSS,
                "initial_observed": production[0][
                    "profile_family_loss"
                ],
                "initial_reproduced": math.isclose(
                    production[0]["profile_family_loss"],
                    REFERENCE_INITIAL_FAMILY_LOSS,
                    rel_tol=0.0,
                    abs_tol=1e-6,
                ),
                "step_100_reference": REFERENCE_STEP_100_FAMILY_LOSS,
                "step_100_observed": production[100][
                    "profile_family_loss"
                ],
                "step_100_reproduced": math.isclose(
                    production[100]["profile_family_loss"],
                    REFERENCE_STEP_100_FAMILY_LOSS,
                    rel_tol=0.0,
                    abs_tol=1e-6,
                ),
            }
            decision = classify(
                trajectories,
                conflicts,
                arithmetic,
                logit_check,
                family_check,
            )
            terminal = {
                "event": "diagnostic_complete",
                "reproduction": reproduced,
                "decision": decision,
                "test_partition_accessed": False,
            }
            emitter.emit(terminal)
            print(
                (
                    "diagnostic complete: {}; production family CE "
                    "step0={:.9g}, step100={:.9g}, step500={:.9g}"
                ).format(
                    decision["classification"],
                    production[0]["profile_family_loss"],
                    production[100]["profile_family_loss"],
                    production[500]["profile_family_loss"],
                ),
                file=sys.stderr,
            )
    except Exception as exc:
        emitter.emit({
            "event": "diagnostic_failure",
            "error_type": type(exc).__name__,
            "detail": str(exc),
            "test_partition_accessed": False,
        })
        return 2
    finally:
        emitter.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
