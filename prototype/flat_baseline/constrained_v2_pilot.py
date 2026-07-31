"""Bounded V2 train-and-ordinary-IID-validation engineering pilot."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F

from prototype.constrained_profile_decoder import profile_targets_for_loss
from prototype.controlled_data.factors import OperationTemplate
from prototype.model_data.adapters import adapt_flat_mixed
from prototype.model_data.batching import collate_flat
from prototype.model_data.loader import load_partition_physical_examples
from prototype.model_data.vocab import NODE_TYPES
from prototype.profile_geometry import (
    PROFILE_FAMILIES,
    canonical_primitive_type_ids_from_family_id,
    construct_profile_geometry,
    profile_family_from_id,
)

from .checkpointing import capture_rng_state
from .constrained_v2 import ConstrainedProfileV2Model
from .constrained_v2_autonomous import (
    greedy_decode_v2,
    validate_and_convert_v2_autonomous_prediction,
)
from .constrained_v2_config import ConstrainedProfileV2Config
from .constrained_v2_losses import constrained_profile_v2_loss
from .constrained_v2_training import (
    V2_TRAINING_LOSS_FIELDS,
    build_v2_optimizer,
    constrained_v2_training_step,
    save_v2_checkpoint,
)
from .constrained_v2_training_config import ConstrainedV2TrainingConfig
from .constrained_v2_pilot_config import (
    ConstrainedV2PilotConfig,
    ConstrainedV2PilotError,
    validate_pilot_partition_authorization,
)
from .data import make_data_loader
from .provenance import source_state
from .run_logging import JsonlLogger
from .training import resolve_device, seed_everything


PILOT_CHECKPOINT_FIELDS = {
    "checkpoint_version",
    "checkpoint_kind",
    "model_name",
    "model_config_version",
    "decoder_contract_version",
    "pilot_identity",
    "pilot_config",
    "model_config",
    "optimization_config",
    "model_state",
    "optimizer_state",
    "vq_state",
    "epoch",
    "global_step",
    "examples_processed",
    "configured_maximum_steps",
    "completed_epochs",
    "training_partition_state",
    "validation_partition_state",
    "training_sampler_state",
    "validation_cadence",
    "validation_summaries",
    "rng_state",
    "source_provenance",
    "systematic_partition_accessed",
    "test_partition_accessed",
}


@dataclass(frozen=True)
class PilotPartition:
    physical_examples: tuple
    flat_examples: tuple
    family_ids: tuple
    fingerprint: str
    partition: str
    identity: str

    def metadata(self):
        return {
            "split_name": "iid",
            "partition": self.partition,
            "partition_identity": self.identity,
            "family_count": len(self.family_ids),
            "family_ids": list(self.family_ids),
            "corpus_fingerprint": self.fingerprint,
        }


@dataclass(frozen=True)
class PilotData:
    train: PilotPartition
    validation: PilotPartition


@dataclass(frozen=True)
class PilotResult:
    output_dir: str
    metrics_path: str
    final_checkpoint: str
    selected_checkpoint: str
    completed_epochs: int
    global_step: int
    examples_processed: int
    terminal_summary: dict


def load_pilot_data(corpus_dir, config):
    validate_pilot_partition_authorization(config)
    train = load_partition_physical_examples(
        corpus_dir, config.split_name, config.training_partition
    )
    validation = load_partition_physical_examples(
        corpus_dir, config.split_name, config.validation_partition
    )
    train_ids = tuple(item.physical_family_id for item in train)
    validation_ids = tuple(item.physical_family_id for item in validation)
    if set(train_ids) & set(validation_ids):
        raise ConstrainedV2PilotError(
            "partition_overlap", "train and validation families overlap"
        )
    return PilotData(
        _pilot_partition(
            train, train_ids, "train", "train"
        ),
        _pilot_partition(
            validation,
            validation_ids,
            "validation",
            "iid_validation",
        ),
    )


def _pilot_partition(physical, family_ids, partition, identity):
    if not physical:
        raise ConstrainedV2PilotError(
            "empty_partition", "{} is empty".format(identity)
        )
    digest = hashlib.sha256()
    for item in physical:
        encoded_id = item.physical_family_id.encode("utf-8")
        encoded_record = item.canonical_reconstruction_json.encode("utf-8")
        digest.update(len(encoded_id).to_bytes(8, "big"))
        digest.update(encoded_id)
        digest.update(len(encoded_record).to_bytes(8, "big"))
        digest.update(encoded_record)
    return PilotPartition(
        tuple(physical),
        tuple(adapt_flat_mixed(item) for item in physical),
        tuple(family_ids),
        digest.hexdigest(),
        partition,
        identity,
    )


def pilot_optimization_config(config, configured_maximum_steps):
    config.validate()
    return ConstrainedV2TrainingConfig(
        seed=config.seed,
        device=config.device,
        batch_size=config.batch_size,
        maximum_steps=configured_maximum_steps,
        output_dir=config.output_dir,
        require_clean_source=config.require_clean_source,
    )


def pilot_optimization_metadata(config):
    values = config.to_dict()
    names = (
        "model_name",
        "model_config_version",
        "decoder_contract_version",
        "checkpoint_version",
        "seed",
        "device",
        "dtype",
        "batch_size",
        "learning_rate",
        "weight_decay",
        "gradient_clip_norm",
        "maximum_steps",
        "extent_min",
        "extent_max",
        "profile_family_order",
        "node_type_loss_weight",
        "categorical_loss_weight",
        "non_profile_geometry_loss_weight",
        "profile_family_loss_weight",
        "profile_parameter_loss_weight",
        "edge_presence_loss_weight",
        "edge_type_loss_weight",
        "operation_pointer_loss_weight",
        "vq_loss_weight",
        "vq_initialization",
    )
    return {name: values[name] for name in names}


def deterministic_epoch_family_ids(examples, batch_size, seed, epoch):
    generator = torch.Generator()
    generator.manual_seed(seed + epoch)
    order = torch.randperm(len(examples), generator=generator).tolist()
    return tuple(
        tuple(
            examples[index].physical_family_id
            for index in order[start:start + batch_size]
        )
        for start in range(0, len(order), batch_size)
    )


def teacher_forced_validation(
    model,
    partition,
    model_config,
    batch_size,
    device,
):
    was_training = model.training
    vq_before = {
        name: value.detach().clone()
        for name, value in model.vq.state_dict().items()
    }
    loss_sums = {name: 0.0 for name in V2_TRAINING_LOSS_FIELDS}
    family_confusion = [
        [0 for _ in PROFILE_FAMILIES] for _ in PROFILE_FAMILIES
    ]
    family_absolute = [0.0 for _ in PROFILE_FAMILIES]
    family_smooth_l1 = [0.0 for _ in PROFILE_FAMILIES]
    family_parameter_values = [0 for _ in PROFILE_FAMILIES]
    family_applicable = [0 for _ in PROFILE_FAMILIES]
    code_counts = torch.zeros(model_config.codebook_size, dtype=torch.long)
    example_count = 0
    try:
        model.eval()
        with torch.no_grad():
            for start in range(0, len(partition.flat_examples), batch_size):
                batch = collate_flat(
                    partition.flat_examples[start:start + batch_size]
                )
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
                current = len(batch.family_ids)
                for name, values in losses.per_example.items():
                    metric = (
                        "remaining_categorical_loss"
                        if name == "categorical_attributes"
                        else "{}_loss".format(name)
                    )
                    if metric not in loss_sums:
                        continue
                    loss_sums[metric] += float(values.double().sum().item())
                selected = profiles.sketch_mask
                target_ids = profiles.family_ids[selected]
                predicted_ids = output.profile_family_logits[selected].argmax(-1)
                for target_id, predicted_id in zip(
                    target_ids.cpu().tolist(),
                    predicted_ids.cpu().tolist(),
                ):
                    family_confusion[target_id][predicted_id] += 1
                    family_applicable[target_id] += 1
                differences = (
                    output.constrained_profile_parameters
                    - profiles.parameters
                )
                for class_id, _ in enumerate(PROFILE_FAMILIES):
                    class_mask = selected & (profiles.family_ids == class_id)
                    if class_mask.any():
                        values = differences[class_mask]
                        family_absolute[class_id] += float(
                            values.abs().double().sum().item()
                        )
                        family_smooth_l1[class_id] += float(
                            F.smooth_l1_loss(
                                output.constrained_profile_parameters[class_mask],
                                profiles.parameters[class_mask],
                                reduction="sum",
                            ).double().item()
                        )
                        family_parameter_values[class_id] += values.numel()
                code_counts.add_(
                    torch.bincount(
                        output.code_indices.detach().cpu().reshape(-1),
                        minlength=model_config.codebook_size,
                    )
                )
                example_count += current
    finally:
        model.train(was_training)
    _assert_vq_unchanged(model, vq_before)
    losses = {
        name: value / float(example_count)
        for name, value in loss_sums.items()
    }
    probabilities = code_counts.double() / max(int(code_counts.sum()), 1)
    nonzero = probabilities > 0
    perplexity = float(torch.exp(
        -(probabilities[nonzero] * torch.log(probabilities[nonzero])).sum()
    ).item())
    per_family_accuracy = {}
    compact_parameters = {}
    for class_id, family in enumerate(PROFILE_FAMILIES):
        row = family_confusion[class_id]
        count = sum(row)
        per_family_accuracy[family.value] = (
            float(row[class_id]) / float(count) if count else 0.0
        )
        denominator = family_parameter_values[class_id]
        compact_parameters[family.value] = {
            "applicable_sketch_count": family_applicable[class_id],
            "parameter_value_count": denominator,
            "mae": (
                family_absolute[class_id] / float(denominator)
                if denominator else 0.0
            ),
            "smooth_l1": (
                family_smooth_l1[class_id] / float(denominator)
                if denominator else 0.0
            ),
        }
    correct = sum(
        family_confusion[index][index]
        for index in range(len(PROFILE_FAMILIES))
    )
    total_family = sum(sum(row) for row in family_confusion)
    summary = {
        "example_count": example_count,
        **losses,
        "profile_family_count": total_family,
        "profile_parameter_count": total_family * 3,
        "family_accuracy": (
            float(correct) / float(total_family)
            if total_family else 0.0
        ),
        "family_confusion_matrix": family_confusion,
        "family_accuracy_per_class": per_family_accuracy,
        "compact_parameter_metrics": compact_parameters,
        "active_vq_code_count": int((code_counts > 0).sum().item()),
        "vq_perplexity": perplexity,
    }
    _require_finite_json(summary)
    return summary


def autonomous_validation(
    model,
    partition,
    model_config,
    batch_size,
    device,
):
    was_training = model.training
    vq_before = {
        name: value.detach().clone()
        for name, value in model.vq.state_dict().items()
    }
    failure_histogram = {}
    valid_count = 0
    finite_count = 0
    canonical_count = 0
    conversion_count = 0
    family_totals = {}
    family_valid = {}
    node_totals = {}
    node_valid = {}
    operation_family_totals = {}
    operation_family_valid = {}
    operation_count_totals = {}
    operation_count_valid = {}
    family_confusion = [
        [0 for _ in range(len(PROFILE_FAMILIES) + 1)]
        for _ in PROFILE_FAMILIES
    ]
    outcomes = []
    try:
        model.eval()
        for start in range(0, len(partition.flat_examples), batch_size):
            flat = partition.flat_examples[start:start + batch_size]
            physical = partition.physical_examples[start:start + batch_size]
            batch = collate_flat(flat)
            inputs = {
                name: value.to(device)
                for name, value in batch.to_torch(torch).items()
            }
            node_counts = torch.tensor(
                [
                    sum(row)
                    for row in batch.target.node_mask
                ],
                dtype=torch.long,
                device=device,
            )
            raw_predictions = greedy_decode_v2(
                model,
                inputs,
                node_counts=node_counts,
                node_count_source="authorized_validation_length",
            )
            for example, raw in zip(physical, raw_predictions):
                result = validate_and_convert_v2_autonomous_prediction(
                    raw, max_operations=model_config.max_operations
                )
                valid = bool(result.controlled_domain.valid)
                converted = bool(result.reconstruction_target.valid)
                finite = _raw_prediction_is_finite(raw)
                canonical = _raw_profile_records_are_canonical(raw)
                code = (
                    "valid"
                    if result.primary_failure is None
                    else result.primary_failure.code
                )
                if result.primary_failure is not None:
                    failure_histogram[code] = (
                        failure_histogram.get(code, 0) + 1
                    )
                valid_count += int(valid)
                conversion_count += int(converted)
                finite_count += int(finite)
                canonical_count += int(canonical)
                family = example.metadata.primitive_family
                node_bucket = str(len(example.nodes))
                operation_template = OperationTemplate(
                    example.metadata.operation_template
                )
                operation_families = "+".join(
                    operation_template.operations
                )
                operation_count = str(len(example.operation_sequence))
                _increment(family_totals, family)
                _increment(node_totals, node_bucket)
                _increment(operation_family_totals, operation_families)
                _increment(operation_count_totals, operation_count)
                if valid:
                    _increment(family_valid, family)
                    _increment(node_valid, node_bucket)
                    _increment(operation_family_valid, operation_families)
                    _increment(operation_count_valid, operation_count)
                true_id = tuple(
                    item.value for item in PROFILE_FAMILIES
                ).index(family)
                for position, node in enumerate(example.nodes):
                    if node.node_type != "sketch":
                        continue
                    predicted_id = (
                        raw.predicted_profile_family_ids[position]
                        if position < len(raw.predicted_profile_family_ids)
                        else -1
                    )
                    column_id = (
                        predicted_id
                        if 0 <= predicted_id < len(PROFILE_FAMILIES)
                        else len(PROFILE_FAMILIES)
                    )
                    family_confusion[true_id][column_id] += 1
                outcomes.append({
                    "physical_family_id": example.physical_family_id,
                    "valid": valid,
                    "conversion_success": converted,
                    "finite": finite,
                    "canonical_profiles": canonical,
                    "failure_code": code,
                })
    finally:
        model.train(was_training)
    _assert_vq_unchanged(model, vq_before)
    total = len(partition.physical_examples)
    summary = {
        "example_count": total,
        "total_count": total,
        "valid_count": valid_count,
        "overall_autonomous_validity": valid_count / float(total),
        "validity_by_profile_family": _rates(family_totals, family_valid),
        "validity_by_node_count_bucket": _rates(node_totals, node_valid),
        "validity_by_operation_family": _rates(
            operation_family_totals, operation_family_valid
        ),
        "validity_by_operation_count": _rates(
            operation_count_totals, operation_count_valid
        ),
        "failure_reason_histogram": failure_histogram,
        "predicted_family_confusion_matrix": family_confusion,
        "finite_prediction_rate": finite_count / float(total),
        "canonical_profile_rate": canonical_count / float(total),
        "conversion_success_rate": conversion_count / float(total),
        "outcomes": outcomes,
    }
    _require_finite_json(summary)
    return summary


def _increment(values, key):
    values[key] = values.get(key, 0) + 1


def _rates(totals, successes):
    return {
        key: {
            "count": totals[key],
            "valid_count": successes.get(key, 0),
            "validity": successes.get(key, 0) / float(totals[key]),
        }
        for key in sorted(totals)
    }


def _raw_prediction_is_finite(raw):
    values = []
    for node in raw.raw_nodes:
        values.extend(node.normalized_geometry)
    for edge in raw.raw_edges:
        values.append(edge.presence_logit)
    for pointer in raw.raw_operation_pointers:
        values.append(pointer.selected_logit)
    for row in raw.profile_family_logits:
        values.extend(row)
    for row in raw.raw_profile_parameters:
        values.extend(row)
    for row in raw.constrained_profile_parameters:
        values.extend(row)
    return all(math.isfinite(float(value)) for value in values)


def _raw_profile_records_are_canonical(raw):
    if (
        len(raw.raw_nodes) != len(raw.predicted_profile_family_ids)
        or len(raw.raw_nodes) != len(raw.constrained_profile_parameters)
    ):
        return False
    sketch_id = NODE_TYPES.id("sketch")
    for node, family_id, parameters in zip(
        raw.raw_nodes,
        raw.predicted_profile_family_ids,
        raw.constrained_profile_parameters,
    ):
        if node.node_type_id == sketch_id:
            if not 0 <= family_id < len(PROFILE_FAMILIES):
                return False
            if (
                tuple(node.categorical_ids[4:8])
                != canonical_primitive_type_ids_from_family_id(family_id)
                or len(parameters) != 3
            ):
                return False
            try:
                geometry, mask = construct_profile_geometry(
                    profile_family_from_id(family_id), *parameters
                )
            except (TypeError, ValueError):
                return False
            if tuple(node.derived_geometry_mask[9:33]) != mask[9:33]:
                return False
            if any(
                not math.isclose(
                    node.normalized_geometry[index],
                    geometry[index],
                    rel_tol=1e-5,
                    abs_tol=2e-7,
                )
                for index in range(9, 33)
            ):
                return False
        elif family_id != -1:
            return False
    return True


def _assert_vq_unchanged(model, before):
    for name, expected in before.items():
        if not torch.equal(expected, model.vq.state_dict()[name]):
            raise ConstrainedV2PilotError(
                "validation_mutated_vq",
                "validation changed VQ state {!r}".format(name),
            )


def _require_finite_json(value, path="$"):
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ConstrainedV2PilotError(
                "nonfinite_json", "{} is nonfinite".format(path)
            )
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _require_finite_json(item, "{}[{}]".format(path, index))
        return
    if isinstance(value, dict):
        for name, item in value.items():
            _require_finite_json(item, "{}.{}".format(path, name))
        return
    raise ConstrainedV2PilotError(
        "invalid_json_value", "{} has unsupported type".format(path)
    )


def pilot_checkpoint_payload(
    model,
    optimizer,
    model_config,
    optimization_config,
    pilot_config,
    data,
    *,
    epoch,
    global_step,
    examples_processed,
    configured_maximum_steps,
    teacher_summary,
    autonomous_summary,
    source_provenance,
):
    return {
        "checkpoint_version": 2,
        "checkpoint_kind": "pilot_fixed_epoch",
        "model_name": model_config.model_name,
        "model_config_version": model_config.model_config_version,
        "decoder_contract_version": model_config.decoder_contract_version,
        "pilot_identity": pilot_config.pilot_identity,
        "pilot_config": pilot_config.to_dict(),
        "model_config": model_config.to_dict(),
        "optimization_config": pilot_optimization_metadata(
            optimization_config
        ),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "vq_state": dict(model.vq.state_dict()),
        "epoch": epoch,
        "global_step": global_step,
        "examples_processed": examples_processed,
        "configured_maximum_steps": configured_maximum_steps,
        "completed_epochs": epoch,
        "training_partition_state": data.train.metadata(),
        "validation_partition_state": data.validation.metadata(),
        "training_sampler_state": {
            "base_seed": pilot_config.seed,
            "completed_epoch": epoch,
            "next_epoch_seed": pilot_config.seed + epoch + 1,
        },
        "validation_cadence": pilot_config.validation_interval,
        "validation_summaries": {
            "teacher_forced": teacher_summary,
            "autonomous": autonomous_summary,
        },
        "rng_state": capture_rng_state(torch),
        "source_provenance": source_provenance,
        "systematic_partition_accessed": False,
        "test_partition_accessed": False,
    }


def load_pilot_checkpoint(
    path,
    model,
    optimizer,
    model_config,
    optimization_config,
    pilot_config,
    data,
    *,
    map_location,
):
    checkpoint = torch.load(str(path), map_location="cpu")
    if (
        not isinstance(checkpoint, dict)
        or set(checkpoint) != PILOT_CHECKPOINT_FIELDS
    ):
        raise ConstrainedV2PilotError(
            "malformed_checkpoint", "pilot checkpoint fields differ"
        )
    steps_per_epoch = int(math.ceil(
        len(data.train.flat_examples) / float(pilot_config.batch_size)
    ))
    expected_epoch = checkpoint["epoch"]
    if (
        isinstance(expected_epoch, bool)
        or not isinstance(expected_epoch, int)
        or not 1 <= expected_epoch <= pilot_config.epochs
        or checkpoint["completed_epochs"] != expected_epoch
        or checkpoint["global_step"] != expected_epoch * steps_per_epoch
        or checkpoint["examples_processed"]
        != expected_epoch * len(data.train.flat_examples)
        or checkpoint["configured_maximum_steps"]
        != pilot_config.epochs * steps_per_epoch
        or checkpoint["training_sampler_state"] != {
            "base_seed": pilot_config.seed,
            "completed_epoch": expected_epoch,
            "next_epoch_seed": pilot_config.seed + expected_epoch + 1,
        }
        or set(checkpoint["validation_summaries"])
        != {"teacher_forced", "autonomous"}
        or not isinstance(checkpoint["source_provenance"], dict)
        or set(checkpoint["source_provenance"])
        != {
            "git_commit",
            "git_dirty",
            "git_status_porcelain",
            "source_tree_sha256",
        }
        or not isinstance(checkpoint["rng_state"], dict)
        or set(checkpoint["rng_state"])
        != {"python", "torch_cpu", "torch_cuda"}
    ):
        raise ConstrainedV2PilotError(
            "malformed_checkpoint", "pilot checkpoint state is invalid"
        )
    expected = (
        ("checkpoint_version", 2),
        ("checkpoint_kind", "pilot_fixed_epoch"),
        ("model_name", model_config.model_name),
        ("model_config_version", model_config.model_config_version),
        ("decoder_contract_version", model_config.decoder_contract_version),
        ("pilot_identity", pilot_config.pilot_identity),
        ("pilot_config", pilot_config.to_dict()),
        ("model_config", model_config.to_dict()),
        (
            "optimization_config",
            pilot_optimization_metadata(optimization_config),
        ),
        ("training_partition_state", data.train.metadata()),
        ("validation_partition_state", data.validation.metadata()),
        ("validation_cadence", pilot_config.validation_interval),
        ("systematic_partition_accessed", False),
        ("test_partition_accessed", False),
    )
    for name, value in expected:
        if checkpoint[name] != value:
            raise ConstrainedV2PilotError(
                "incompatible_checkpoint", "{} differs".format(name)
            )
    vq_state = checkpoint["vq_state"]
    expected_vq_names = {
        name[3:]
        for name in checkpoint["model_state"]
        if name.startswith("vq.")
    }
    if (
        not isinstance(checkpoint["model_state"], dict)
        or not isinstance(checkpoint["optimizer_state"], dict)
        or not isinstance(vq_state, dict)
        or set(vq_state) != expected_vq_names
        or any(
            "vq." + name not in checkpoint["model_state"]
            or not torch.equal(
                value, checkpoint["model_state"]["vq." + name]
            )
            for name, value in vq_state.items()
        )
    ):
        raise ConstrainedV2PilotError(
            "malformed_checkpoint",
            "pilot model, optimizer, or explicit VQ state is invalid",
        )
    _require_finite_tensor_tree(
        checkpoint["model_state"], "model_state"
    )
    _require_finite_tensor_tree(
        checkpoint["optimizer_state"], "optimizer_state"
    )
    _require_finite_json(checkpoint["validation_summaries"])
    model.load_state_dict(checkpoint["model_state"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    for state in optimizer.state.values():
        for name, value in tuple(state.items()):
            if torch.is_tensor(value):
                state[name] = value.to(map_location)
    return checkpoint


def _require_finite_tensor_tree(value, path):
    if torch.is_tensor(value):
        if (
            value.dtype.is_floating_point
            and not bool(torch.isfinite(value).all().item())
        ):
            raise ConstrainedV2PilotError(
                "nonfinite_checkpoint", "{} is nonfinite".format(path)
            )
        return
    if isinstance(value, dict):
        for name, item in value.items():
            _require_finite_tensor_tree(
                item, "{}.{}".format(path, name)
            )
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _require_finite_tensor_tree(
                item, "{}[{}]".format(path, index)
            )
        return
    if value is None or isinstance(value, (bool, int, float, str)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ConstrainedV2PilotError(
                "nonfinite_checkpoint", "{} is nonfinite".format(path)
            )
        return
    raise ConstrainedV2PilotError(
        "malformed_checkpoint",
        "{} has unsupported state type".format(path),
    )


def run_constrained_v2_pilot(
    corpus_dir,
    pilot_config=None,
    model_config=None,
):
    config = pilot_config or ConstrainedV2PilotConfig()
    try:
        return _run_constrained_v2_pilot(
            corpus_dir, config, model_config
        )
    except Exception as exc:
        _append_terminal_failure_if_possible(config, exc)
        raise


def _run_constrained_v2_pilot(
    corpus_dir,
    pilot_config=None,
    model_config=None,
):
    pilot_config = pilot_config or ConstrainedV2PilotConfig()
    model_config = model_config or ConstrainedProfileV2Config()
    validate_pilot_partition_authorization(pilot_config)
    model_config.validate()
    provenance = source_state()
    if pilot_config.require_clean_source and (
        provenance["git_dirty"] is not False
        or provenance["git_commit"] is None
    ):
        raise ConstrainedV2PilotError(
            "dirty_source", "pilot requires clean committed source"
        )
    output_dir = Path(pilot_config.output_dir)
    if output_dir.exists() or output_dir.is_symlink():
        raise ConstrainedV2PilotError(
            "output_collision", "pilot output already exists"
        )
    data = load_pilot_data(corpus_dir, pilot_config)
    configured_maximum_steps = pilot_config.epochs * int(math.ceil(
        len(data.train.flat_examples) / float(pilot_config.batch_size)
    ))
    output_dir.mkdir(parents=True)
    logger = JsonlLogger(output_dir / "metrics.jsonl")
    optimization = pilot_optimization_config(
        pilot_config, configured_maximum_steps
    )
    seed_everything(pilot_config.seed, torch)
    device = resolve_device(pilot_config.device, torch)
    model = ConstrainedProfileV2Model(model_config).to(device)
    optimizer = build_v2_optimizer(model, optimization)
    logger.write({
        "event": "run_metadata",
        "pilot_config": pilot_config.to_dict(),
        "model_config": model_config.to_dict(),
        "optimization_config": pilot_optimization_metadata(optimization),
        "configured_maximum_steps": configured_maximum_steps,
        "training_partition": data.train.metadata(),
        "validation_partition": data.validation.metadata(),
        "systematic_partition_accessed": False,
        "test_partition_accessed": False,
        "source_provenance": provenance,
        "initialization": {
            "source": "fresh_v2",
            "vq_initialization": "normal",
            "seed": pilot_config.seed,
        },
    })
    initial_train = teacher_forced_validation(
        model,
        data.train,
        model_config,
        pilot_config.batch_size,
        device,
    )
    global_step = 0
    examples_processed = 0
    validations = {}
    checkpoint_paths = {}
    for epoch in range(1, pilot_config.epochs + 1):
        loader = make_data_loader(
            data.train.flat_examples,
            pilot_config.batch_size,
            0,
            pilot_config.seed + epoch,
            True,
            torch,
        )
        for batch in loader:
            global_step += 1
            current = len(batch.family_ids)
            examples_processed += current
            result = constrained_v2_training_step(
                model,
                optimizer,
                batch,
                model_config,
                optimization,
                device,
                global_step=global_step,
                examples_processed=examples_processed,
            )
            logger.write({
                "event": "training_step",
                "epoch": epoch,
                **result.metrics,
                "systematic_partition_accessed": False,
                "test_partition_accessed": False,
            })
        logger.write({
            "event": "training_epoch_complete",
            "epoch": epoch,
            "global_step": global_step,
            "examples_processed": examples_processed,
            "sampler_seed": pilot_config.seed + epoch,
            "systematic_partition_accessed": False,
            "test_partition_accessed": False,
        })
        teacher = teacher_forced_validation(
            model,
            data.validation,
            model_config,
            pilot_config.batch_size,
            device,
        )
        autonomous = autonomous_validation(
            model,
            data.validation,
            model_config,
            pilot_config.batch_size,
            device,
        )
        validations[epoch] = {
            "teacher_forced": teacher,
            "autonomous": autonomous,
        }
        identity = "epoch-{:04d}".format(epoch)
        common = {
            "global_step": global_step,
            "epoch": epoch,
            "examples_processed": examples_processed,
            "checkpoint_identity": identity,
            "partition_identity": "iid_validation",
            "systematic_partition_accessed": False,
            "test_partition_accessed": False,
        }
        logger.write({
            "event": "validation_teacher_forced",
            **common,
            "summary": teacher,
        })
        logger.write({
            "event": "validation_autonomous",
            **common,
            "summary": autonomous,
        })
        checkpoint = output_dir / "{}.pt".format(identity)
        payload = pilot_checkpoint_payload(
            model,
            optimizer,
            model_config,
            optimization,
            pilot_config,
            data,
            epoch=epoch,
            global_step=global_step,
            examples_processed=examples_processed,
            configured_maximum_steps=configured_maximum_steps,
            teacher_summary=teacher,
            autonomous_summary=autonomous,
            source_provenance=provenance,
        )
        save_v2_checkpoint(checkpoint, payload)
        checkpoint_paths[epoch] = checkpoint
        logger.write({
            "event": "checkpoint_saved",
            **common,
            "path": str(checkpoint),
            "checkpoint_kind": "pilot_fixed_epoch",
        })
    final_train = teacher_forced_validation(
        model,
        data.train,
        model_config,
        pilot_config.batch_size,
        device,
    )
    selected_epoch = min(
        validations,
        key=lambda epoch: (
            validations[epoch]["teacher_forced"]["total_loss"],
            epoch,
        ),
    )
    selected_path = checkpoint_paths[selected_epoch]
    reloaded = ConstrainedProfileV2Model(model_config).to(device)
    reloaded_optimizer = build_v2_optimizer(reloaded, optimization)
    checkpoint = load_pilot_checkpoint(
        selected_path,
        reloaded,
        reloaded_optimizer,
        model_config,
        optimization,
        pilot_config,
        data,
        map_location=device,
    )
    vq_before = {
        name: value.detach().clone()
        for name, value in reloaded.vq.state_dict().items()
    }
    reproduced_teacher = teacher_forced_validation(
        reloaded,
        data.validation,
        model_config,
        pilot_config.batch_size,
        device,
    )
    reproduced_autonomous = autonomous_validation(
        reloaded,
        data.validation,
        model_config,
        pilot_config.batch_size,
        device,
    )
    _assert_nested_close(
        checkpoint["validation_summaries"]["teacher_forced"],
        reproduced_teacher,
    )
    _assert_nested_close(
        checkpoint["validation_summaries"]["autonomous"],
        reproduced_autonomous,
    )
    _assert_vq_unchanged(reloaded, vq_before)
    logger.write({
        "event": "checkpoint_reload_validation",
        "epoch": selected_epoch,
        "global_step": checkpoint["global_step"],
        "examples_processed": checkpoint["examples_processed"],
        "checkpoint_identity": "epoch-{:04d}".format(selected_epoch),
        "partition_identity": "iid_validation",
        "teacher_forced_reproduced": True,
        "autonomous_reproduced": True,
        "vq_ema_unchanged": True,
        "reload_role": (
            "selected_and_final"
            if selected_epoch == pilot_config.epochs
            else "selected"
        ),
        "systematic_partition_accessed": False,
        "test_partition_accessed": False,
    })
    final_path = checkpoint_paths[pilot_config.epochs]
    if final_path == selected_path:
        final_reload_reproduced = True
    else:
        final_reloaded = ConstrainedProfileV2Model(model_config).to(device)
        final_optimizer = build_v2_optimizer(
            final_reloaded, optimization
        )
        final_checkpoint = load_pilot_checkpoint(
            final_path,
            final_reloaded,
            final_optimizer,
            model_config,
            optimization,
            pilot_config,
            data,
            map_location=device,
        )
        final_vq_before = {
            name: value.detach().clone()
            for name, value in final_reloaded.vq.state_dict().items()
        }
        final_reproduced_teacher = teacher_forced_validation(
            final_reloaded,
            data.validation,
            model_config,
            pilot_config.batch_size,
            device,
        )
        final_reproduced_autonomous = autonomous_validation(
            final_reloaded,
            data.validation,
            model_config,
            pilot_config.batch_size,
            device,
        )
        _assert_nested_close(
            final_checkpoint["validation_summaries"][
                "teacher_forced"
            ],
            final_reproduced_teacher,
        )
        _assert_nested_close(
            final_checkpoint["validation_summaries"]["autonomous"],
            final_reproduced_autonomous,
        )
        _assert_vq_unchanged(final_reloaded, final_vq_before)
        final_reload_reproduced = True
        logger.write({
            "event": "checkpoint_reload_validation",
            "epoch": pilot_config.epochs,
            "global_step": final_checkpoint["global_step"],
            "examples_processed": final_checkpoint[
                "examples_processed"
            ],
            "checkpoint_identity": "epoch-{:04d}".format(
                pilot_config.epochs
            ),
            "partition_identity": "iid_validation",
            "teacher_forced_reproduced": True,
            "autonomous_reproduced": True,
            "vq_ema_unchanged": True,
            "reload_role": "final",
            "systematic_partition_accessed": False,
            "test_partition_accessed": False,
        })
    scientific_acceptance = {
        "configured_budget_completed": (
            global_step == configured_maximum_steps
            and examples_processed
            == len(data.train.flat_examples) * pilot_config.epochs
        ),
        "train_total_loss_decreased": (
            final_train["total_loss"] < initial_train["total_loss"]
        ),
        "validation_losses_finite": all(math.isfinite(
            validations[pilot_config.epochs]["teacher_forced"][name]
        ) for name in (
            "total_loss",
            "profile_family_loss",
            "profile_parameter_loss",
        )),
        "all_validation_families_present": all(
            metrics["applicable_sketch_count"] > 0
            for metrics in validations[pilot_config.epochs][
                "teacher_forced"
            ]["compact_parameter_metrics"].values()
        ),
        "autonomous_completed": (
            len(validations[pilot_config.epochs]["autonomous"]["outcomes"])
            == len(data.validation.flat_examples)
        ),
        "autonomous_outcomes_classified": (
            validations[pilot_config.epochs]["autonomous"]["valid_count"]
            + sum(validations[pilot_config.epochs]["autonomous"][
                    "failure_reason_histogram"
                ].values())
            == len(data.validation.flat_examples)
        ),
        "strict_selected_reload_reproduced": True,
        "strict_final_reload_reproduced": final_reload_reproduced,
    }
    acceptance = _require_pilot_acceptance(
        scientific_acceptance,
        {
            "systematic_partition_accessed": (
                pilot_config.systematic_partition_accessed
            ),
            "test_partition_accessed": (
                pilot_config.test_partition_accessed
            ),
        },
    )
    terminal = {
        "acceptance": acceptance,
        "configured_maximum_steps": configured_maximum_steps,
        "completed_epochs": pilot_config.epochs,
        "global_step": global_step,
        "examples_processed": examples_processed,
        "diagnostic_selected_epoch": selected_epoch,
        "diagnostic_selection_metric": "ordinary_validation_total_loss",
        "final_validation_teacher_forced": validations[
            pilot_config.epochs
        ]["teacher_forced"],
        "final_validation_autonomous": validations[
            pilot_config.epochs
        ]["autonomous"],
    }
    _write_terminal_success(logger, terminal)
    return PilotResult(
        str(output_dir),
        str(logger.path),
        str(checkpoint_paths[pilot_config.epochs]),
        str(selected_path),
        pilot_config.epochs,
        global_step,
        examples_processed,
        terminal,
    )


def _append_terminal_failure_if_possible(config, exc):
    if getattr(exc, "code", None) == "output_collision":
        return
    output_dir = Path(config.output_dir)
    metrics_path = output_dir / "metrics.jsonl"
    if not output_dir.is_dir() or not metrics_path.exists():
        return
    last_event = None
    try:
        lines = metrics_path.read_text(encoding="utf-8").splitlines()
        if not lines:
            return
        first = json.loads(lines[0])
        if (
            first.get("event") != "run_metadata"
            or first.get("pilot_config", {}).get("pilot_identity")
            != config.pilot_identity
        ):
            return
        last_event = json.loads(lines[-1]).get("event")
    except (OSError, ValueError, TypeError):
        last_event = None
    if last_event in ("terminal_success", "terminal_failure"):
        return
    JsonlLogger(metrics_path).write({
        "event": "terminal_failure",
        "error_code": getattr(exc, "code", "pilot_failure"),
        "error_type": type(exc).__name__,
        "detail": str(exc),
        "systematic_partition_accessed": False,
        "test_partition_accessed": False,
    })


def _write_terminal_success(logger, terminal):
    _require_finite_json(terminal)
    logger.write({
        "event": "terminal_success",
        **terminal,
        "systematic_partition_accessed": False,
        "test_partition_accessed": False,
    })


def _require_pilot_acceptance(scientific_criteria, access_state):
    criteria = dict(scientific_criteria)
    access = access_state if isinstance(access_state, dict) else {}
    criteria["systematic_partition_not_accessed"] = (
        access.get("systematic_partition_accessed") is False
    )
    criteria["test_partition_not_accessed"] = (
        access.get("test_partition_accessed") is False
    )
    unmet = tuple(
        name for name, value in criteria.items() if value is not True
    )
    if unmet:
        raise ConstrainedV2PilotError(
            "pilot_acceptance_failed",
            repr({"criteria": criteria, "unmet": unmet}),
        )
    return criteria


def _assert_nested_close(expected, actual, path="$"):
    if type(expected) is not type(actual):
        raise ConstrainedV2PilotError(
            "reload_mismatch", "{} type differs".format(path)
        )
    if isinstance(expected, dict):
        if set(expected) != set(actual):
            raise ConstrainedV2PilotError(
                "reload_mismatch", "{} keys differ".format(path)
            )
        for name in expected:
            _assert_nested_close(
                expected[name], actual[name], "{}.{}".format(path, name)
            )
    elif isinstance(expected, list):
        if len(expected) != len(actual):
            raise ConstrainedV2PilotError(
                "reload_mismatch", "{} length differs".format(path)
            )
        for index, value in enumerate(expected):
            _assert_nested_close(
                value, actual[index], "{}[{}]".format(path, index)
            )
    elif isinstance(expected, float):
        if not math.isclose(expected, actual, rel_tol=1e-5, abs_tol=2e-7):
            raise ConstrainedV2PilotError(
                "reload_mismatch", "{} differs".format(path)
            )
    elif expected != actual:
        raise ConstrainedV2PilotError(
            "reload_mismatch", "{} differs".format(path)
        )
