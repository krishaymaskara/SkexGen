"""Frozen two-epoch train/IID-validation pilot for graph V1."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

import torch

from prototype.constrained_profile_decoder import profile_targets_for_loss
from prototype.flat_baseline.checkpointing import capture_rng_state, save_checkpoint
from prototype.flat_baseline.constrained_v6_config import ConstrainedProfileV6Config
from prototype.flat_baseline.constrained_v6_pilot import (
    autonomous_validation as v6_autonomous_validation,
    load_pilot_data,
    teacher_forced_validation as v6_teacher_forced_validation,
)
from prototype.flat_baseline.constrained_v6_pilot_config import ConstrainedV6PilotConfig
from prototype.flat_baseline.run_logging import JsonlLogger
from prototype.flat_baseline.training import seed_everything
from prototype.model_data.batching import collate_flat

from .autonomous import greedy_decode_graph_v1
from .checkpoint import (
    GRAPH_CHECKPOINT_FIELDS,
    graph_model_metadata,
    validate_graph_checkpoint,
)
from .config import GraphV1Config
from .conversion import (
    graph_v1_teacher_forced_predictions,
    validate_and_convert_graph_prediction,
)
from .losses import graph_v1_loss
from .metrics import (
    FROZEN_V6_REFERENCE,
    finish_graph_metrics,
    new_graph_metrics,
    update_graph_outcome,
    update_pair_metrics,
    update_raw_graph_outcome,
)
from .model import GraphV1Model
from .pilot_config import (
    EXPECTED_EXAMPLES_PROCESSED,
    EXPECTED_OPTIMIZER_STEPS,
    GraphPilotConfig,
    validate_partition_authorization,
)
from .provenance import (
    GRAPH_SOURCE_BRANCH,
    collect_graph_source_provenance,
    validate_graph_source_provenance,
    verify_graph_source_provenance,
)
from .training import build_graph_optimizer, graph_training_step
from .training_config import GraphTrainingConfig


GRAPH_LOSS_FIELDS = (
    "total", "node_type", "categorical_attributes",
    "reference_plane_category", "remaining_geometry", "profile_family",
    "profile_parameter", "graph_edge", "none_class", "positive_edge",
    "edge_type", "vq_commitment",
)


@dataclass(frozen=True)
class GraphPilotResult:
    output_dir: str
    metrics_path: str
    selected_checkpoint: str
    final_checkpoint: str
    completed_epochs: int
    global_step: int
    examples_processed: int
    terminal_summary: dict


def run_graph_v1_pilot(
    corpus_dir,
    pilot_config=None,
    model_config=None,
    *,
    repository_root,
    reviewed_commit
):
    pilot_config = pilot_config or GraphPilotConfig()
    model_config = model_config or GraphV1Config()
    training_config = GraphTrainingConfig()
    validate_partition_authorization(pilot_config)
    model_config.validate()
    training_config.validate()
    repository_root = Path(repository_root).resolve()
    provenance = collect_graph_source_provenance(repository_root)
    validate_graph_source_provenance(
        provenance,
        expected_commit=reviewed_commit,
        expected_branch=GRAPH_SOURCE_BRANCH,
    )
    output_dir = Path(pilot_config.output_dir)
    if output_dir.exists() or output_dir.is_symlink():
        raise ValueError("graph pilot output must be new and immutable")
    v6_data_config = ConstrainedV6PilotConfig(
        require_clean_source=pilot_config.require_clean_source
    )
    data = load_pilot_data(corpus_dir, v6_data_config)
    output_dir.mkdir(parents=True)
    logger = JsonlLogger(output_dir / "metrics.jsonl")
    seed_everything(pilot_config.seed, torch)
    device = torch.device("cpu")
    model = GraphV1Model(model_config).to(device)
    optimizer = build_graph_optimizer(model, training_config)
    configured_steps = pilot_config.epochs * int(math.ceil(
        len(data.train.flat_examples) / float(pilot_config.batch_size)
    ))
    if configured_steps != EXPECTED_OPTIMIZER_STEPS:
        raise ValueError("configured graph step budget differs")
    metadata = {
        "event": "run_metadata",
        "pilot_config": pilot_config.to_dict(),
        "model_config": model_config.to_dict(),
        "training_config": training_config.to_dict(),
        "model_metadata": graph_model_metadata(model, model_config),
        "training_partition": data.train.metadata(),
        "validation_partition": data.validation.metadata(),
        "configured_maximum_steps": configured_steps,
        "frozen_v6_reference": dict(FROZEN_V6_REFERENCE),
        "source_provenance": provenance,
        "systematic_partition_accessed": False,
        "test_partition_accessed": False,
    }
    _require_finite_json(metadata)
    logger.write(metadata)
    initial_train = teacher_forced_graph_validation(
        model, data.train, model_config, pilot_config.batch_size, device
    )
    global_step = 0
    examples_processed = 0
    validations = {}
    checkpoint_paths = {}
    for epoch in range(1, pilot_config.epochs + 1):
        generator = torch.Generator().manual_seed(pilot_config.seed + epoch)
        order = torch.randperm(len(data.train.flat_examples), generator=generator).tolist()
        for start in range(0, len(order), pilot_config.batch_size):
            examples = tuple(data.train.flat_examples[index] for index in order[
                start:start + pilot_config.batch_size
            ])
            batch = collate_flat(examples)
            inputs = {name: value.to(device) for name, value in batch.to_torch(torch).items()}
            target = {name: value.to(device) for name, value in batch.target.to_torch(torch).items()}
            profiles = profile_targets_for_loss(batch.target, inputs["geometry"])
            unused_output, losses = graph_training_step(
                model, optimizer, inputs, target, profiles, model_config, training_config
            )
            del unused_output
            global_step += 1
            examples_processed += len(batch.family_ids)
            logger.write({
                "event": "training_step",
                "epoch": epoch,
                "global_step": global_step,
                "examples_processed": examples_processed,
                "losses": {name: float(value.detach().item())
                           for name, value in losses.as_dict().items()},
            })
        teacher = teacher_forced_graph_validation(
            model, data.validation, model_config, pilot_config.batch_size, device
        )
        autonomous = autonomous_graph_validation(
            model, data.validation, model_config, pilot_config.batch_size, device
        )
        validations[epoch] = {"teacher_forced": teacher, "autonomous": autonomous}
        payload = graph_checkpoint_payload(
            model, optimizer, model_config, pilot_config, training_config,
            data, epoch, global_step, examples_processed, configured_steps,
            validations[epoch], provenance, repository_root=repository_root,
        )
        checkpoint_path = output_dir / "epoch-{:04d}.pt".format(epoch)
        save_checkpoint(checkpoint_path, payload, torch)
        checkpoint_paths[epoch] = checkpoint_path
        logger.write({
            "event": "epoch_validation", "epoch": epoch,
            "teacher_forced": teacher, "autonomous": autonomous,
            "checkpoint": str(checkpoint_path),
        })
    selected_epoch = min(
        validations,
        key=lambda item: validations[item]["teacher_forced"]["total_loss"],
    )
    selected_path = checkpoint_paths[selected_epoch]
    final_path = checkpoint_paths[pilot_config.epochs]
    selected_reload = _reload_and_validate(
        selected_path, model_config, pilot_config, training_config, data, device,
        provenance, repository_root,
    )
    final_reload = _reload_and_validate(
        final_path, model_config, pilot_config, training_config, data, device,
        provenance, repository_root,
    )
    final_train = teacher_forced_graph_validation(
        model, data.train, model_config, pilot_config.batch_size, device
    )
    scientific = {
        "configured_budget_completed": (
            global_step == EXPECTED_OPTIMIZER_STEPS
            and examples_processed == EXPECTED_EXAMPLES_PROCESSED
        ),
        "train_total_loss_decreased": final_train["total_loss"] < initial_train["total_loss"],
        "validation_losses_finite": all(
            math.isfinite(validations[pilot_config.epochs]["teacher_forced"][
                name + "_loss"
            ]) for name in GRAPH_LOSS_FIELDS
        ),
        "all_validation_families_present": all(
            row["applicable_sketch_count"] > 0
            for row in validations[pilot_config.epochs]["teacher_forced"]
            ["inherited_v6_metrics"]["compact_parameter_metrics"].values()
        ),
        "autonomous_completed": (
            validations[pilot_config.epochs]["autonomous"]["example_count"] == 68
        ),
        "autonomous_outcomes_classified": (
            len(validations[pilot_config.epochs]["autonomous"]["graph_metrics"]["outcomes"]) == 68
        ),
        "strict_selected_reload_reproduced": selected_reload,
        "strict_final_reload_reproduced": final_reload,
        "constrained_categorical_contract_satisfied": (
            validations[pilot_config.epochs]["autonomous"]
            ["inherited_v6_metrics"]["categorical_selection_metrics"]
            ["aggregate"]
            ["total_constrained_categorical_applicability_violations"] == 0
        ),
        "constrained_node_grammar_contract_satisfied": (
            validations[pilot_config.epochs]["autonomous"]
            ["inherited_v6_metrics"]["node_grammar_selection_metrics"]
            ["constrained_grammar_violation_count"] == 0
            and validations[pilot_config.epochs]["autonomous"]
            ["inherited_v6_metrics"]["constrained_grammar_valid_sequence_count"]
            == len(data.validation.flat_examples)
        ),
        "constrained_axis_geometry_contract_satisfied": (
            validations[pilot_config.epochs]["autonomous"]
            ["inherited_v6_metrics"]["axis_geometry_metrics"]
            ["constrained_invalid_axis_count"] == 0
            and validations[pilot_config.epochs]["autonomous"]
            ["inherited_v6_metrics"]["axis_geometry_metrics"]
            ["constrained_valid_axis_count"]
            == validations[pilot_config.epochs]["autonomous"]
            ["inherited_v6_metrics"]["axis_geometry_metrics"]
            ["applicable_axis_node_count"]
        ),
        "graph_representation_contract_satisfied": (
            validations[pilot_config.epochs]["autonomous"]["graph_metrics"]
            ["example_count"] == len(data.validation.flat_examples)
        ),
        "graph_predictions_finite": (
            validations[pilot_config.epochs]["autonomous"]["graph_metrics"]
            ["nonfinite_logit_count"] == 0
        ),
        "graph_converter_outcomes_classified": (
            sum(validations[pilot_config.epochs]["autonomous"]["graph_metrics"]
                ["first_failure_histogram"].values())
            == len(data.validation.flat_examples)
        ),
    }
    acceptance = _require_acceptance(scientific, {
        "systematic_partition_accessed": False,
        "test_partition_accessed": False,
    })
    terminal = {
        "event": "terminal_success",
        "acceptance": acceptance,
        "selected_epoch": selected_epoch,
        "selected_checkpoint": str(selected_path),
        "final_checkpoint": str(final_path),
        "completed_epochs": pilot_config.epochs,
        "global_step": global_step,
        "examples_processed": examples_processed,
        "final_validation_teacher_forced": validations[pilot_config.epochs]["teacher_forced"],
        "final_validation_autonomous": validations[pilot_config.epochs]["autonomous"],
        "systematic_partition_accessed": False,
        "test_partition_accessed": False,
    }
    _require_finite_json(terminal)
    logger.write(terminal)
    return GraphPilotResult(
        str(output_dir), str(output_dir / "metrics.jsonl"), str(selected_path),
        str(final_path), pilot_config.epochs, global_step, examples_processed, terminal
    )


def teacher_forced_graph_validation(model, partition, model_config, batch_size, device):
    was_training = model.training
    loss_sums = {name: 0.0 for name in GRAPH_LOSS_FIELDS}
    metrics = new_graph_metrics()
    example_count = 0
    try:
        model.eval()
        with torch.no_grad():
            for start in range(0, len(partition.flat_examples), batch_size):
                batch = collate_flat(partition.flat_examples[start:start + batch_size])
                inputs = {name: value.to(device) for name, value in batch.to_torch(torch).items()}
                target = {name: value.to(device) for name, value in batch.target.to_torch(torch).items()}
                profiles = profile_targets_for_loss(batch.target, inputs["geometry"])
                output = model(target=target, profile_targets=profiles, **inputs)
                losses = graph_v1_loss(output, target, profiles, model_config)
                predictions = graph_v1_teacher_forced_predictions(
                    output, node_mask=target["node_mask"],
                    node_count_source="teacher_forced_authorized_length",
                )
                physical_by_id = {
                    item.physical_family_id: item
                    for item in partition.physical_examples
                }
                for family_id, prediction in zip(batch.family_ids, predictions):
                    physical = physical_by_id[family_id]
                    result = validate_and_convert_graph_prediction(prediction)
                    update_pair_metrics(metrics, prediction, physical.target)
                    update_raw_graph_outcome(metrics, prediction, physical.target)
                    update_graph_outcome(metrics, prediction, physical.target, result, physical.metadata)
                for name, values in losses.per_example.items():
                    loss_sums[name] += float(values.double().sum().item())
                example_count += len(batch.family_ids)
    finally:
        model.train(was_training)
    inherited = v6_teacher_forced_validation(
        model, partition, ConstrainedProfileV6Config(), batch_size, device
    )
    summary = {
        "example_count": example_count,
        **{name + "_loss": value / float(example_count)
           for name, value in loss_sums.items()},
        "graph_metrics": finish_graph_metrics(metrics),
        "inherited_v6_metrics": inherited,
    }
    _require_finite_json(summary)
    return summary


def autonomous_graph_validation(model, partition, model_config, batch_size, device):
    was_training = model.training
    metrics = new_graph_metrics()
    try:
        model.eval()
        for start in range(0, len(partition.flat_examples), batch_size):
            batch = collate_flat(partition.flat_examples[start:start + batch_size])
            inputs = {name: value.to(device) for name, value in batch.to_torch(torch).items()}
            counts = torch.tensor(
                [sum(row) for row in batch.target.node_mask],
                dtype=torch.long, device=device,
            )
            predictions = greedy_decode_graph_v1(
                model, inputs, node_counts=counts,
                node_count_source="authorized_validation_length",
            )
            physical_by_id = {
                item.physical_family_id: item for item in partition.physical_examples
            }
            for family_id, prediction in zip(batch.family_ids, predictions):
                physical = physical_by_id[family_id]
                result = validate_and_convert_graph_prediction(prediction)
                update_pair_metrics(metrics, prediction, physical.target)
                update_raw_graph_outcome(metrics, prediction, physical.target)
                update_graph_outcome(metrics, prediction, physical.target, result, physical.metadata)
    finally:
        model.train(was_training)
    summary = {
        "example_count": len(partition.flat_examples),
        "graph_metrics": finish_graph_metrics(metrics),
        "inherited_v6_metrics": v6_autonomous_validation(
            model, partition, ConstrainedProfileV6Config(), batch_size, device
        ),
    }
    _require_finite_json(summary)
    return summary


def graph_checkpoint_payload(
    model, optimizer, model_config, pilot_config, training_config, data,
    epoch, global_step, examples_processed, configured_steps,
    validation_summaries, provenance, *, repository_root,
):
    verified_provenance = verify_graph_source_provenance(
        repository_root, provenance
    )
    payload = {
        **graph_model_metadata(model, model_config),
        "checkpoint_kind": "pilot_fixed_epoch",
        "pilot_config": pilot_config.to_dict(),
        "training_config": training_config.to_dict(),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "vq_state": model.vq.state_dict(),
        "rng_state": capture_rng_state(torch),
        "epoch": epoch,
        "global_step": global_step,
        "examples_processed": examples_processed,
        "configured_maximum_steps": configured_steps,
        "completed_epochs": epoch,
        "training_partition_state": data.train.metadata(),
        "validation_partition_state": data.validation.metadata(),
        "validation_summaries": validation_summaries,
        "source_provenance": verified_provenance,
        "systematic_partition_accessed": False,
        "test_partition_accessed": False,
    }
    if set(payload) != GRAPH_CHECKPOINT_FIELDS:
        raise ValueError("graph checkpoint field set differs")
    validate_graph_checkpoint(
        payload, model, model_config, pilot_config, training_config, torch,
        expected_source_provenance=provenance,
    )
    return payload


def _reload_and_validate(
    path, model_config, pilot_config, training_config, data, device,
    provenance, repository_root,
):
    verify_graph_source_provenance(repository_root, provenance)
    model = GraphV1Model(model_config).to(device)
    optimizer = build_graph_optimizer(model, training_config)
    payload = torch.load(str(path), map_location="cpu")
    validate_graph_checkpoint(
        payload, model, model_config, pilot_config, training_config, torch,
        expected_source_provenance=provenance,
    )
    model.load_state_dict(payload["model_state"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state"])
    summary = teacher_forced_graph_validation(
        model, data.validation, model_config, pilot_config.batch_size, device
    )
    expected = payload["validation_summaries"]["teacher_forced"]
    return summary == expected


def _require_acceptance(scientific, access):
    result = dict(scientific)
    result["systematic_partition_not_accessed"] = (
        access.get("systematic_partition_accessed") is False
    )
    result["test_partition_not_accessed"] = (
        access.get("test_partition_accessed") is False
    )
    unmet = tuple(name for name, value in result.items() if value is not True)
    if unmet:
        raise ValueError("graph_pilot_acceptance_failed: {}".format(unmet))
    return result


def _require_finite_json(value, path="$ "):
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError("{} is nonfinite".format(path))
        return
    if isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _require_finite_json(item, "{}[{}]".format(path, index))
        return
    if isinstance(value, dict):
        for name, item in value.items():
            _require_finite_json(item, "{}.{}".format(path, name))
        return
    raise ValueError("{} is not JSON-compatible".format(path))
