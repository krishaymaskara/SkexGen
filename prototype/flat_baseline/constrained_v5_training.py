"""Minimal train-only optimization and checkpoint workflow for V5."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import torch

from prototype.constrained_profile_decoder import profile_targets_for_loss
from prototype.model_data.adapters import adapt_flat_mixed
from prototype.model_data.batching import collate_flat
from prototype.model_data.loader import load_partition_physical_examples
from prototype.model_data.vocab import NODE_TYPES
from prototype.node_grammar import V5_NODE_GRAMMAR
from prototype.profile_geometry import PROFILE_FAMILIES
from prototype.representation.model import NodeType

from .checkpointing import capture_rng_state, save_checkpoint
from .constrained_v5 import ConstrainedProfileV5Model
from .constrained_v5_config import ConstrainedProfileV5Config
from .constrained_v5_losses import constrained_profile_v5_loss
from .constrained_v5_training_config import (
    ConstrainedV5TrainingConfig,
    V5_TINY_SELECTION_IDENTITY,
)
from .provenance import source_state
from .run_logging import JsonlLogger
from .training import (
    _require_finite_gradients,
    _require_finite_model_state,
    _require_finite_optimizer_state,
    _require_finite_tensor,
    resolve_device,
    seed_everything,
)


V5_TRAINING_LOSS_FIELDS = (
    "total_loss",
    "node_type_loss",
    "remaining_categorical_loss",
    "reference_plane_category_loss",
    "profile_family_loss",
    "profile_parameter_loss",
    "remaining_geometry_loss",
    "edge_presence_loss",
    "edge_type_loss",
    "operation_pointer_loss",
    "vq_commitment_loss",
)
V5_TINY_EVALUATION_MILESTONES = (100, 200)
V5_CHECKPOINT_FIELDS = {
    "checkpoint_version",
    "model_name",
    "model_config_version",
    "decoder_contract_version",
    "learned_geometry_channel_indices",
    "canonical_plane_contract_id",
    "base_geometry_contract_id",
    "categorical_selection_contract_id",
    "categorical_selection_contract",
    "node_grammar_contract_id",
    "node_grammar_contract",
    "node_vocabulary",
    "valid_requested_node_counts",
    "operation_limit",
    "completion_algorithm_id",
    "model_config",
    "training_config",
    "profile_family_order",
    "extent_bounds",
    "loss_weights",
    "model_state",
    "optimizer_state",
    "vq_state",
    "epoch",
    "global_step",
    "selected_tiny_overfit_ids",
    "data_state",
    "rng_state",
    "source_provenance",
}


class ConstrainedV5TrainingError(RuntimeError):
    """The bounded V5 workflow cannot continue safely."""

    def __init__(self, code, detail):
        self.code = code
        self.detail = detail
        super().__init__("{}: {}".format(code, detail))


@dataclass(frozen=True)
class TinyOverfitSelection:
    examples: tuple
    selected_example_ids: tuple
    selection_rule: str
    family_counts: dict
    exact_operation_types: tuple
    operation_families: tuple
    node_count_distribution: tuple
    missing_coverage: tuple
    corpus_dir: str
    train_selection_sha256: str
    split_name: str
    training_partition: str

    def metadata(self):
        return {
            "selected_example_ids": list(self.selected_example_ids),
            "selection_rule": self.selection_rule,
            "family_counts": dict(self.family_counts),
            "exact_operation_types": list(self.exact_operation_types),
            "operation_families": list(self.operation_families),
            "node_count_distribution": list(self.node_count_distribution),
            "missing_coverage": list(self.missing_coverage),
            "corpus_dir": self.corpus_dir,
            "train_selection_sha256": self.train_selection_sha256,
            "split_name": self.split_name,
            "training_partition": self.training_partition,
            "validation_partition": "none",
            "test_partition_accessed": False,
        }


@dataclass(frozen=True)
class V5TrainingStepResult:
    metrics: dict
    gradient_parameter_names: tuple


@dataclass(frozen=True)
class V5LossSnapshot:
    metrics: dict


@dataclass(frozen=True)
class V5TinyOverfitResult:
    output_dir: str
    metrics_path: str
    final_checkpoint: str
    global_step: int
    selected_example_ids: tuple
    success_criteria: dict


def load_v5_tiny_training_selection(
    corpus_dir,
    training_config: ConstrainedV5TrainingConfig,
) -> TinyOverfitSelection:
    """Load only train payloads, then select deterministic metadata coverage."""

    training_config.validate()
    physical = load_partition_physical_examples(
        corpus_dir,
        training_config.split_name,
        training_config.training_partition,
    )
    return select_v5_tiny_training_examples(
        physical,
        corpus_dir=str(Path(corpus_dir).resolve()),
        split_name=training_config.split_name,
        subset_size=training_config.tiny_subset_size,
    )


def select_v5_tiny_training_examples(
    physical_examples,
    *,
    corpus_dir,
    split_name,
    subset_size,
) -> TinyOverfitSelection:
    """Greedily cover frozen training metadata with stable family-ID ties."""

    ordered = tuple(sorted(
        physical_examples, key=lambda item: item.physical_family_id
    ))
    if not ordered or any(item.partition != "train" for item in ordered):
        raise ConstrainedV5TrainingError(
            "unauthorized_training_data",
            "tiny selection requires nonempty train-only examples",
        )
    if isinstance(subset_size, bool) or not isinstance(subset_size, int):
        raise TypeError("subset_size must be an integer")
    if not 6 <= subset_size <= 12:
        raise ValueError("subset_size must be between 6 and 12")
    if len(ordered) < subset_size:
        raise ConstrainedV5TrainingError(
            "insufficient_training_examples",
            "training partition has {} examples, but {} were requested".format(
                len(ordered), subset_size
            ),
        )

    selected = []
    selected_ids = set()
    family_names = tuple(item.value for item in PROFILE_FAMILIES)
    operation_coverage = {
        item.physical_family_id: operation_coverage_for_example(item)
        for item in ordered
    }
    goals = tuple(
        ("family:{}".format(name), lambda item, name=name:
         item.metadata.primitive_family == name)
        for name in family_names
    ) + (
        (
            "operation:extrude",
            lambda item: "extrude"
            in operation_coverage[item.physical_family_id][1],
        ),
        (
            "operation:revolve",
            lambda item: "revolve"
            in operation_coverage[item.physical_family_id][1],
        ),
        ("multi_operation", lambda item: len(item.operation_sequence) > 1),
    )
    for unused_name, predicate in goals:
        del unused_name
        candidate = next(
            (
                item
                for item in ordered
                if item.physical_family_id not in selected_ids
                and predicate(item)
            ),
            None,
        )
        if candidate is not None:
            selected.append(candidate)
            selected_ids.add(candidate.physical_family_id)
    represented_lengths = {len(item.nodes) for item in selected}
    for item in ordered:
        if len(selected) >= min(subset_size, len(ordered)):
            break
        if (
            item.physical_family_id not in selected_ids
            and len(item.nodes) not in represented_lengths
        ):
            selected.append(item)
            selected_ids.add(item.physical_family_id)
            represented_lengths.add(len(item.nodes))
    for item in ordered:
        if len(selected) >= min(subset_size, len(ordered)):
            break
        if item.physical_family_id not in selected_ids:
            selected.append(item)
            selected_ids.add(item.physical_family_id)
    selected = tuple(sorted(
        selected, key=lambda item: item.physical_family_id
    ))
    observed_families = {
        name: sum(item.metadata.primitive_family == name for item in selected)
        for name in family_names
    }
    exact_operations = tuple(sorted({
        operation
        for item in selected
        for operation in operation_coverage[item.physical_family_id][0]
    }))
    operation_families = tuple(sorted({
        family
        for item in selected
        for family in operation_coverage[item.physical_family_id][1]
    }))
    missing = tuple(
        name for name, predicate in goals
        if not any(predicate(item) for item in selected)
    )
    adapted = tuple(adapt_flat_mixed(item) for item in selected)
    selected_example_ids = tuple(
        item.physical_family_id for item in selected
    )
    selection_identity = {
        "corpus_dir": str(corpus_dir),
        "selected_example_ids": list(selected_example_ids),
        "selection_rule": V5_TINY_SELECTION_IDENTITY,
        "split_name": str(split_name),
        "training_partition": "train",
    }
    train_selection_sha256 = hashlib.sha256(
        json.dumps(
            selection_identity,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return TinyOverfitSelection(
        adapted,
        selected_example_ids,
        V5_TINY_SELECTION_IDENTITY,
        observed_families,
        exact_operations,
        operation_families,
        tuple(sorted(len(item.nodes) for item in selected)),
        missing,
        str(corpus_dir),
        train_selection_sha256,
        str(split_name),
        "train",
    )


def operation_coverage_for_example(physical_example):
    """Return exact operation IDs and typed extrusion/revolution families."""

    nodes = {
        node.node_id: node for node in physical_example.nodes
    }
    if len(nodes) != len(physical_example.nodes):
        raise ConstrainedV5TrainingError(
            "invalid_operation_coverage",
            "canonical node IDs must be unique",
        )
    supported = {
        NodeType.EXTRUDE.value,
        NodeType.REVOLVE.value,
    }
    families = []
    for operation_id in physical_example.operation_sequence:
        node = nodes.get(operation_id)
        if (
            node is None
            or node.node_type not in supported
            or node.operation_type != node.node_type
        ):
            raise ConstrainedV5TrainingError(
                "unsupported_operation_type",
                "operation {!r} has no supported typed node".format(
                    operation_id
                ),
            )
        families.append(node.operation_type)
    return tuple(physical_example.operation_sequence), tuple(families)


def build_v5_optimizer(model, training_config, torch_module=torch):
    """Construct one AdamW group containing every V5 parameter exactly once."""

    if not isinstance(model, ConstrainedProfileV5Model):
        raise TypeError("model must be ConstrainedProfileV5Model")
    training_config.validate()
    parameters = tuple(
        parameter for parameter in model.parameters()
        if parameter.requires_grad
    )
    if len({id(parameter) for parameter in parameters}) != len(parameters):
        raise ConstrainedV5TrainingError(
            "duplicate_optimizer_parameter", "model parameters are duplicated"
        )
    optimizer = torch_module.optim.AdamW(
        parameters,
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    optimized = tuple(
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
    )
    if (
        len(optimized) != len(parameters)
        or len({id(item) for item in optimized}) != len(optimized)
        or {id(item) for item in optimized}
        != {id(item) for item in parameters}
    ):
        raise ConstrainedV5TrainingError(
            "optimizer_parameter_mismatch",
            "optimizer does not contain every intended V5 parameter",
        )
    return optimizer


def constrained_v5_training_step(
    model,
    optimizer,
    batch,
    model_config,
    training_config,
    device,
    *,
    global_step,
    examples_processed,
    torch_module=torch,
) -> V5TrainingStepResult:
    """Run one finite teacher-forced V5 optimization step."""

    _validate_model_and_training_config(model_config, training_config)
    model.train()
    inputs = {
        name: value.to(device)
        for name, value in batch.to_torch(torch_module).items()
    }
    target = {
        name: value.to(device)
        for name, value in batch.target.to_torch(torch_module).items()
    }
    profile_targets = profile_targets_for_loss(
        batch.target, inputs["geometry"]
    )
    optimizer.zero_grad()
    output = model(
        target=target, profile_targets=profile_targets, **inputs
    )
    _validate_training_output(output, torch_module)
    losses = constrained_profile_v5_loss(
        output, target, profile_targets, model_config
    )
    for name, value in losses.as_dict().items():
        _require_finite_tensor(
            value, "nonfinite_{}_loss".format(name), torch_module
        )
    losses.total.backward()
    _require_finite_gradients(model, torch_module)
    gradient_names = tuple(
        name
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    )
    missing = tuple(
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    )
    if missing:
        raise ConstrainedV5TrainingError(
            "missing_gradient",
            "parameters lack gradients: {}".format(", ".join(missing)),
        )
    gradient_norm = torch_module.nn.utils.clip_grad_norm_(
        model.parameters(), training_config.gradient_clip_norm
    )
    _require_finite_tensor(
        gradient_norm, "nonfinite_gradient_norm", torch_module
    )
    _require_finite_gradients(
        model, torch_module, "nonfinite_postclip_gradient"
    )
    optimizer.step()
    _require_finite_model_state(model, torch_module)
    _require_finite_optimizer_state(optimizer, torch_module)
    metrics = _step_metrics(
        losses,
        output,
        optimizer,
        gradient_norm,
        global_step,
        examples_processed,
    )
    return V5TrainingStepResult(metrics, gradient_names)


def constrained_v5_evaluation_snapshot(
    model,
    batch,
    model_config,
    training_config,
    device,
    *,
    torch_module=torch,
) -> V5LossSnapshot:
    """Evaluate one fixed teacher-forced batch without gradients or EMA updates."""

    _validate_model_and_training_config(model_config, training_config)
    was_training = model.training
    try:
        model.eval()
        inputs = {
            name: value.to(device)
            for name, value in batch.to_torch(torch_module).items()
        }
        target = {
            name: value.to(device)
            for name, value in batch.target.to_torch(torch_module).items()
        }
        profile_targets = profile_targets_for_loss(
            batch.target, inputs["geometry"]
        )
        with torch_module.no_grad():
            output = model(
                target=target,
                profile_targets=profile_targets,
                **inputs
            )
            _validate_training_output(output, torch_module)
            losses = constrained_profile_v5_loss(
                output, target, profile_targets, model_config
            )
        for name, value in losses.as_dict().items():
            _require_finite_tensor(
                value,
                "nonfinite_evaluation_{}_loss".format(name),
                torch_module,
            )
        return V5LossSnapshot(_loss_metrics(losses))
    finally:
        model.train(was_training)


def v5_checkpoint_payload(
    model,
    optimizer,
    model_config,
    training_config,
    selection,
    *,
    global_step,
    source_provenance,
    torch_module=torch,
):
    """Build the strict version-5 checkpoint payload."""

    _validate_model_and_training_config(model_config, training_config)
    data_state = selection.metadata()
    return {
        "checkpoint_version": 5,
        "model_name": model_config.model_name,
        "model_config_version": model_config.model_config_version,
        "decoder_contract_version": model_config.decoder_contract_version,
        "learned_geometry_channel_indices": list(
            model_config.learned_geometry_channel_indices
        ),
        "canonical_plane_contract_id": model_config.canonical_plane_contract_id,
        "base_geometry_contract_id": model_config.base_geometry_contract_id,
        "categorical_selection_contract_id": (
            model_config.categorical_selection_contract_id
        ),
        "categorical_selection_contract": model_config.categorical_selection_contract,
        "node_grammar_contract_id": model_config.node_grammar_contract_id,
        "node_grammar_contract": model_config.node_grammar_contract,
        "node_vocabulary": list(NODE_TYPES.tokens),
        "valid_requested_node_counts": list(
            model_config.valid_requested_node_counts
        ),
        "operation_limit": model_config.max_operations,
        "completion_algorithm_id": model_config.completion_algorithm_id,
        "model_config": model_config.to_dict(),
        "training_config": training_config.to_dict(),
        "profile_family_order": list(model_config.profile_family_order),
        "extent_bounds": [
            model_config.profile_extent_min,
            model_config.profile_extent_max,
        ],
        "loss_weights": training_config.loss_weights(),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "vq_state": {
            name: value
            for name, value in model.vq.state_dict().items()
        },
        "epoch": 0,
        "global_step": global_step,
        "selected_tiny_overfit_ids": list(selection.selected_example_ids),
        "data_state": data_state,
        "rng_state": capture_rng_state(torch_module),
        "source_provenance": source_provenance,
    }


def save_v5_checkpoint(path, payload, torch_module=torch):
    destination = Path(path)
    if destination.exists() or destination.is_symlink():
        raise ConstrainedV5TrainingError(
            "checkpoint_collision", "checkpoint destination already exists"
        )
    save_checkpoint(destination, payload, torch_module)


def load_v5_checkpoint(
    path,
    model,
    optimizer,
    model_config,
    training_config,
    selection,
    *,
    map_location,
    torch_module=torch,
):
    checkpoint = torch_module.load(str(path), map_location="cpu")
    _validate_v5_checkpoint(
        checkpoint, model_config, training_config, selection
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    for state in optimizer.state.values():
        for name, value in tuple(state.items()):
            if torch_module.is_tensor(value):
                state[name] = value.to(map_location)
    _require_finite_model_state(model, torch_module)
    _require_finite_optimizer_state(optimizer, torch_module)
    return checkpoint


def run_v5_tiny_overfit(
    corpus_dir,
    model_config=None,
    training_config=None,
    *,
    torch_module=torch,
) -> V5TinyOverfitResult:
    """Run the bounded train-only smoke, checkpoint it, and reload strictly."""

    model_config = model_config or ConstrainedProfileV5Config()
    training_config = training_config or ConstrainedV5TrainingConfig()
    _validate_model_and_training_config(model_config, training_config)
    provenance = source_state()
    if training_config.require_clean_source and (
        provenance["git_dirty"] is not False
        or provenance["git_commit"] is None
    ):
        raise ConstrainedV5TrainingError(
            "dirty_source",
            "tiny-overfit requires identifiable clean committed source",
        )
    output_dir = Path(training_config.output_dir)
    if output_dir.exists() or output_dir.is_symlink():
        raise ConstrainedV5TrainingError(
            "output_collision", "output directory already exists"
        )
    selection = load_v5_tiny_training_selection(
        corpus_dir, training_config
    )
    output_dir.mkdir(parents=True)
    logger = JsonlLogger(output_dir / "metrics.jsonl")
    logger.write({
        "event": "run_metadata",
        "model_config": model_config.to_dict(),
        "training_config": training_config.to_dict(),
        "selection": selection.metadata(),
        "training_partition": "train",
        "validation_partition": "none",
        "test_partition_accessed": False,
        "source_provenance": provenance,
        "vq_initialization": {
            "mode": "normal",
            "seed": training_config.seed,
            "available_prequant_vector_count": (
                len(selection.examples) * model_config.latent_tokens
            ),
            "minimum_kmeans_vector_count": model_config.codebook_size,
            "enough_vectors_by_count": (
                len(selection.examples) * model_config.latent_tokens
                >= model_config.codebook_size
            ),
            "kmeans_suitability_established": False,
            "reason": (
                "fresh normal is the bounded smoke default; distinct-vector "
                "suitability is not assessed by fitting on this tiny subset"
            ),
        },
    })
    seed_everything(training_config.seed, torch_module)
    device = resolve_device(training_config.device, torch_module)
    model = ConstrainedProfileV5Model(model_config).to(device)
    optimizer = build_v5_optimizer(model, training_config, torch_module)
    batches = tuple(
        collate_flat(selection.examples[index:index + training_config.batch_size])
        for index in range(0, len(selection.examples), training_config.batch_size)
    )
    initial_snapshot = _aggregate_evaluation_snapshots(
        model,
        batches,
        model_config,
        training_config,
        device,
        torch_module=torch_module,
    )
    logger.write({
        "event": "initial_evaluation_snapshot",
        **initial_snapshot.metrics,
    })
    family_before = model.profile_heads.family_head.weight.detach().clone()
    parameter_before = model.profile_heads.parameter_head.weight.detach().clone()
    history = []
    examples_processed = 0
    for step in range(1, training_config.maximum_steps + 1):
        batch = batches[(step - 1) % len(batches)]
        examples_processed += len(batch.family_ids)
        result = constrained_v5_training_step(
            model,
            optimizer,
            batch,
            model_config,
            training_config,
            device,
            global_step=step,
            examples_processed=examples_processed,
            torch_module=torch_module,
        )
        history.append(result.metrics)
        if step == 1 or step % training_config.logging_cadence == 0:
            logger.write({"event": "training_step", **result.metrics})
        if (
            step in V5_TINY_EVALUATION_MILESTONES
            and step < training_config.maximum_steps
        ):
            milestone_snapshot = _aggregate_evaluation_snapshots(
                model,
                batches,
                model_config,
                training_config,
                device,
                torch_module=torch_module,
            )
            logger.write({
                "event": "milestone_evaluation_snapshot",
                "global_step": step,
                **milestone_snapshot.metrics,
            })
        if step % training_config.checkpoint_cadence == 0:
            payload = v5_checkpoint_payload(
                model,
                optimizer,
                model_config,
                training_config,
                selection,
                global_step=step,
                source_provenance=provenance,
                torch_module=torch_module,
            )
            save_v5_checkpoint(
                output_dir / "step-{:06d}.pt".format(step),
                payload,
                torch_module,
            )
    final_snapshot = _aggregate_evaluation_snapshots(
        model,
        batches,
        model_config,
        training_config,
        device,
        torch_module=torch_module,
    )
    logger.write({
        "event": "final_evaluation_snapshot",
        **final_snapshot.metrics,
    })
    criteria = _tiny_success_criteria(
        history,
        initial_snapshot,
        final_snapshot,
        model,
        family_before,
        parameter_before,
        selection,
    )
    final_path = output_dir / "final.pt"
    payload = v5_checkpoint_payload(
        model,
        optimizer,
        model_config,
        training_config,
        selection,
        global_step=training_config.maximum_steps,
        source_provenance=provenance,
        torch_module=torch_module,
    )
    save_v5_checkpoint(final_path, payload, torch_module)
    reloaded = ConstrainedProfileV5Model(model_config).to(device)
    reloaded_optimizer = build_v5_optimizer(
        reloaded, training_config, torch_module
    )
    load_v5_checkpoint(
        final_path,
        reloaded,
        reloaded_optimizer,
        model_config,
        training_config,
        selection,
        map_location=device,
        torch_module=torch_module,
    )
    _assert_reload_forward_close(
        model, reloaded, batches, device, torch_module
    )
    logger.write({
        "event": "terminal_success",
        "global_step": training_config.maximum_steps,
        "success_criteria": criteria,
        "training_partition": "train",
        "validation_partition": "none",
        "test_partition_accessed": False,
    })
    return V5TinyOverfitResult(
        str(output_dir),
        str(logger.path),
        str(final_path),
        training_config.maximum_steps,
        selection.selected_example_ids,
        criteria,
    )


def _validate_model_and_training_config(model_config, training_config):
    if not isinstance(model_config, ConstrainedProfileV5Config):
        raise TypeError("model_config must be ConstrainedProfileV5Config")
    if not isinstance(training_config, ConstrainedV5TrainingConfig):
        raise TypeError(
            "training_config must be ConstrainedV5TrainingConfig"
        )
    model_config.validate()
    training_config.validate()
    expected = {
        "node_type_loss_weight": model_config.node_type_loss_weight,
        "categorical_loss_weight": model_config.categorical_loss_weight,
        "remaining_geometry_loss_weight": model_config.geometry_loss_weight,
        "profile_family_loss_weight": model_config.profile_family_loss_weight,
        "profile_parameter_loss_weight": (
            model_config.profile_parameter_loss_weight
        ),
        "edge_presence_loss_weight": model_config.edge_presence_loss_weight,
        "edge_type_loss_weight": model_config.edge_type_loss_weight,
        "operation_pointer_loss_weight": (
            model_config.operation_pointer_loss_weight
        ),
        "vq_loss_weight": model_config.vq_loss_weight,
    }
    if training_config.loss_weights() != expected:
        raise ConstrainedV5TrainingError(
            "loss_weight_mismatch",
            "training and model loss weights must match exactly",
        )


def _step_metrics(
    losses, output, optimizer, gradient_norm, global_step, examples_processed
):
    metrics = {
        "total_loss": float(losses.total.detach().cpu().item()),
        "node_type_loss": float(losses.node_type.detach().cpu().item()),
        "remaining_categorical_loss": float(
            losses.categorical_attributes.detach().cpu().item()
        ),
        "reference_plane_category_loss": float(
            losses.reference_plane_category.detach().cpu().item()
        ),
        "profile_family_loss": float(
            losses.profile_family.detach().cpu().item()
        ),
        "profile_parameter_loss": float(
            losses.profile_parameter.detach().cpu().item()
        ),
        "remaining_geometry_loss": float(
            losses.remaining_geometry.detach().cpu().item()
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
        "profile_family_count": losses.profile_family_count,
        "profile_parameter_count": losses.profile_parameter_count,
        "active_vq_code_count": int(output.active_code_count.item()),
        "vq_perplexity": float(output.codebook_perplexity.item()),
        "learning_rate": float(optimizer.param_groups[0]["lr"]),
        "gradient_norm": float(gradient_norm.detach().cpu().item()),
        "global_step": global_step,
        "examples_processed": examples_processed,
    }
    if set(V5_TRAINING_LOSS_FIELDS) - set(metrics):
        raise AssertionError("V5 loss logging is incomplete")
    if not all(
        math.isfinite(float(value))
        for value in metrics.values()
    ):
        raise ConstrainedV5TrainingError(
            "nonfinite_log_metric", "training metrics must be finite"
        )
    return metrics


def _loss_metrics(losses):
    metrics = {
        "total_loss": float(losses.total.detach().cpu().item()),
        "node_type_loss": float(losses.node_type.detach().cpu().item()),
        "remaining_categorical_loss": float(
            losses.categorical_attributes.detach().cpu().item()
        ),
        "reference_plane_category_loss": float(
            losses.reference_plane_category.detach().cpu().item()
        ),
        "profile_family_loss": float(
            losses.profile_family.detach().cpu().item()
        ),
        "profile_parameter_loss": float(
            losses.profile_parameter.detach().cpu().item()
        ),
        "remaining_geometry_loss": float(
            losses.remaining_geometry.detach().cpu().item()
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
        "profile_family_count": losses.profile_family_count,
        "profile_parameter_count": losses.profile_parameter_count,
    }
    if not all(math.isfinite(float(value)) for value in metrics.values()):
        raise ConstrainedV5TrainingError(
            "nonfinite_evaluation_metric",
            "evaluation snapshot metrics must be finite",
        )
    return metrics


def _aggregate_evaluation_snapshots(
    model,
    batches,
    model_config,
    training_config,
    device,
    *,
    torch_module,
):
    weighted = {name: 0.0 for name in V5_TRAINING_LOSS_FIELDS}
    family_count = 0
    parameter_count = 0
    example_count = 0
    for batch in batches:
        current_count = len(batch.family_ids)
        snapshot = constrained_v5_evaluation_snapshot(
            model,
            batch,
            model_config,
            training_config,
            device,
            torch_module=torch_module,
        )
        for name in V5_TRAINING_LOSS_FIELDS:
            weighted[name] += snapshot.metrics[name] * current_count
        family_count += snapshot.metrics["profile_family_count"]
        parameter_count += snapshot.metrics["profile_parameter_count"]
        example_count += current_count
    if example_count <= 0:
        raise ConstrainedV5TrainingError(
            "empty_evaluation_snapshot",
            "evaluation snapshots require at least one training example",
        )
    return V5LossSnapshot({
        **{
            name: weighted[name] / float(example_count)
            for name in V5_TRAINING_LOSS_FIELDS
        },
        "profile_family_count": family_count,
        "profile_parameter_count": parameter_count,
    })


def _validate_v5_checkpoint(checkpoint, model_config, training_config, selection):
    if not isinstance(checkpoint, dict) or set(checkpoint) != V5_CHECKPOINT_FIELDS:
        raise ConstrainedV5TrainingError(
            "malformed_checkpoint", "V5 checkpoint fields are invalid"
        )
    if checkpoint["checkpoint_version"] != 5:
        raise ConstrainedV5TrainingError(
            "unsupported_checkpoint", "checkpoint_version must be 5"
        )
    if (
        checkpoint["epoch"] != 0
        or isinstance(checkpoint["global_step"], bool)
        or not isinstance(checkpoint["global_step"], int)
        or checkpoint["global_step"] < 0
        or not isinstance(checkpoint["model_state"], dict)
        or not isinstance(checkpoint["optimizer_state"], dict)
        or not isinstance(checkpoint["source_provenance"], dict)
        or set(checkpoint["source_provenance"])
        != {
            "git_commit", "git_dirty", "git_status_porcelain",
            "source_tree_sha256",
        }
        or not isinstance(
            checkpoint["source_provenance"].get("git_commit"), str
        )
        or not checkpoint["source_provenance"]["git_commit"]
        or type(checkpoint["source_provenance"].get("git_dirty")) is not bool
        or not isinstance(
            checkpoint["source_provenance"].get("git_status_porcelain"), str
        )
        or not isinstance(
            checkpoint["source_provenance"].get("source_tree_sha256"), str
        )
        or not isinstance(checkpoint["rng_state"], dict)
        or set(checkpoint["rng_state"])
        != {"python", "torch_cpu", "torch_cuda"}
    ):
        raise ConstrainedV5TrainingError(
            "malformed_checkpoint", "V5 checkpoint state metadata is invalid"
        )
    expected = (
        ("model_name", model_config.model_name),
        ("model_config_version", model_config.model_config_version),
        ("decoder_contract_version", model_config.decoder_contract_version),
        (
            "learned_geometry_channel_indices",
            list(model_config.learned_geometry_channel_indices),
        ),
        ("canonical_plane_contract_id", model_config.canonical_plane_contract_id),
        ("base_geometry_contract_id", model_config.base_geometry_contract_id),
        (
            "categorical_selection_contract_id",
            model_config.categorical_selection_contract_id,
        ),
        (
            "categorical_selection_contract",
            model_config.categorical_selection_contract,
        ),
        ("node_grammar_contract_id", model_config.node_grammar_contract_id),
        ("node_grammar_contract", V5_NODE_GRAMMAR.to_dict()),
        ("node_vocabulary", list(NODE_TYPES.tokens)),
        (
            "valid_requested_node_counts",
            list(model_config.valid_requested_node_counts),
        ),
        ("operation_limit", model_config.max_operations),
        ("completion_algorithm_id", model_config.completion_algorithm_id),
        ("model_config", model_config.to_dict()),
        ("training_config", training_config.to_dict()),
        ("profile_family_order", list(model_config.profile_family_order)),
        (
            "extent_bounds",
            [model_config.profile_extent_min, model_config.profile_extent_max],
        ),
        ("loss_weights", training_config.loss_weights()),
        (
            "selected_tiny_overfit_ids",
            list(selection.selected_example_ids),
        ),
        ("data_state", selection.metadata()),
    )
    for name, value in expected:
        if checkpoint[name] != value:
            raise ConstrainedV5TrainingError(
                "incompatible_checkpoint",
                "{} differs from the requested V5 run".format(name),
            )
    vq_state = checkpoint["vq_state"]
    expected_vq_names = {
        name[3:]
        for name in checkpoint["model_state"]
        if name.startswith("vq.")
    }
    if (
        not isinstance(vq_state, dict)
        or set(vq_state) != expected_vq_names
        or any(
        "vq." + name not in checkpoint["model_state"]
        or not torch.equal(
            value, checkpoint["model_state"]["vq." + name]
        )
        for name, value in vq_state.items()
        )
    ):
        raise ConstrainedV5TrainingError(
            "malformed_checkpoint", "explicit VQ state disagrees with model state"
        )


def _tiny_success_criteria(
    history,
    initial_snapshot,
    final_snapshot,
    model,
    family_before,
    parameter_before,
    selection,
):
    initial = initial_snapshot.metrics
    final = final_snapshot.metrics
    criteria = {
        "total_loss_decreased": (
            final["total_loss"] < initial["total_loss"]
        ),
        "profile_family_loss_decreased": (
            final["profile_family_loss"]
            < initial["profile_family_loss"]
        ),
        "profile_parameter_loss_decreased": (
            final["profile_parameter_loss"]
            < initial["profile_parameter_loss"]
        ),
        "family_head_changed": not torch.equal(
            family_before, model.profile_heads.family_head.weight
        ),
        "parameter_head_changed": not torch.equal(
            parameter_before, model.profile_heads.parameter_head.weight
        ),
        "all_profile_families_present": all(
            count > 0 for count in selection.family_counts.values()
        ),
        "all_losses_finite": all(
            math.isfinite(item[name])
            for item in history
            for name in V5_TRAINING_LOSS_FIELDS
        ),
        "constrained_profiles_constructed": True,
    }
    if not all(criteria.values()):
        raise ConstrainedV5TrainingError(
            "tiny_overfit_criteria_failed", repr(criteria)
        )
    return criteria


def _validate_training_output(output, torch_module):
    for name in (
        "constrained_profile_parameters",
        "training_profile_geometry",
        "training_geometry",
    ):
        _require_finite_tensor(
            getattr(output, name),
            "nonfinite_{}".format(name),
            torch_module,
        )
    for name in (
        "training_profile_geometry_mask",
        "training_geometry_mask",
    ):
        value = getattr(output, name)
        if value.dtype != torch_module.bool:
            raise ConstrainedV5TrainingError(
                "invalid_geometry_mask",
                "{} must be Boolean".format(name),
            )


def _assert_reload_forward_close(model, reloaded, batches, device, torch_module):
    model.eval()
    reloaded.eval()
    for batch in batches:
        inputs = {
            name: value.to(device)
            for name, value in batch.to_torch(torch_module).items()
        }
        target = {
            name: value.to(device)
            for name, value in batch.target.to_torch(torch_module).items()
        }
        profiles = profile_targets_for_loss(
            batch.target, inputs["geometry"]
        )
        with torch_module.no_grad():
            first = model(
                target=target, profile_targets=profiles, **inputs
            )
            second = reloaded(
                target=target, profile_targets=profiles, **inputs
            )
        for name in (
            "node_type_logits",
            "profile_family_logits",
            "raw_profile_parameters",
            "remaining_geometry",
            "edge_presence_logits",
            "edge_type_logits",
            "operation_pointer_logits",
            "quantized_memory",
        ):
            torch_module.testing.assert_close(
                getattr(first, name),
                getattr(second, name),
                rtol=1e-5,
                atol=2e-7,
            )
