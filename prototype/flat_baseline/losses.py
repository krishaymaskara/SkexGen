"""Independently normalized reconstruction losses for the flat baseline."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass
class FlatMixedVQLoss:
    total: torch.Tensor
    node_type: torch.Tensor
    categorical_attributes: torch.Tensor
    geometry: torch.Tensor
    edge_presence: torch.Tensor
    edge_type: torch.Tensor
    operation_pointer: torch.Tensor
    vq_commitment: torch.Tensor
    per_example: dict

    def as_dict(self):
        return {
            "total": self.total,
            "node_type": self.node_type,
            "categorical_attributes": self.categorical_attributes,
            "geometry": self.geometry,
            "edge_presence": self.edge_presence,
            "edge_type": self.edge_type,
            "operation_pointer": self.operation_pointer,
            "vq_commitment": self.vq_commitment,
        }


def dense_edge_targets(target, node_count):
    """Convert globally offset ReconstructionBatch edges to per-graph matrices."""

    node_mask = target["node_mask"]
    batch_size = node_mask.size(0)
    device = node_mask.device
    presence = torch.zeros(
        batch_size, node_count, node_count, dtype=torch.bool, device=device
    )
    edge_types = torch.zeros(
        batch_size, node_count, node_count, dtype=torch.long, device=device
    )
    valid_pairs = node_mask.unsqueeze(2) & node_mask.unsqueeze(1)
    diagonal = torch.eye(node_count, dtype=torch.bool, device=device).unsqueeze(0)
    valid_pairs = valid_pairs & ~diagonal

    node_counts = node_mask.long().sum(dim=1)
    node_offsets = torch.cat(
        (
            torch.zeros(1, dtype=torch.long, device=device),
            node_counts.cumsum(dim=0),
        )
    )
    edge_offsets = target["edge_offsets"]
    if edge_offsets.numel() != batch_size + 1:
        raise ValueError("edge_offsets must have B+1 entries")
    for batch_index in range(batch_size):
        start = int(edge_offsets[batch_index].item())
        stop = int(edge_offsets[batch_index + 1].item())
        if stop < start:
            raise ValueError("edge_offsets must be monotonic")
        source = (
            target["edge_index"][0, start:stop] - node_offsets[batch_index]
        )
        destination = (
            target["edge_index"][1, start:stop] - node_offsets[batch_index]
        )
        current_count = int(node_counts[batch_index].item())
        if source.numel() and (
            int(source.min().item()) < 0
            or int(destination.min().item()) < 0
            or int(source.max().item()) >= current_count
            or int(destination.max().item()) >= current_count
        ):
            raise ValueError("edge_index disagrees with graph offsets")
        presence[batch_index, source, destination] = True
        edge_types[batch_index, source, destination] = target[
            "edge_type_ids"
        ][start:stop]
    return presence, edge_types, valid_pairs


def flat_mixed_vq_loss(output, target, config):
    """Return scalar means plus exact independently normalized example losses."""

    node_mask = target["node_mask"]
    per_node_type = _per_example_selected_cross_entropy(
        output.node_type_logits, target["node_type_ids"], node_mask
    )
    per_categorical_parts = tuple(
        _per_example_selected_cross_entropy(
            logits,
            target["categorical_attributes"][..., index],
            node_mask,
        )
        for index, logits in enumerate(output.categorical_logits)
    )
    per_categorical = torch.stack(per_categorical_parts).mean(dim=0)

    applicable = target["geometry_mask"] & node_mask.unsqueeze(-1)
    per_geometry = _per_example_smooth_l1(
        output.geometry, target["geometry"], applicable
    )

    presence_target, edge_type_target, valid_pairs = dense_edge_targets(
        target, output.edge_presence_logits.size(1)
    )
    positives = valid_pairs & presence_target
    negatives = valid_pairs & ~presence_target
    per_edge_presence = _per_example_balanced_presence(
        output.edge_presence_logits, positives, negatives
    )
    per_edge_type = _per_example_selected_cross_entropy(
        output.edge_type_logits, edge_type_target, positives
    )

    operation_count = target["operation_sequence"].size(1)
    pointer_logits = output.operation_pointer_logits[:, :operation_count]
    per_operation_pointer = _per_example_selected_cross_entropy(
        pointer_logits,
        target["operation_sequence"],
        target["operation_mask"],
    )
    per_vq = output.vq_per_example_loss
    batch_size = node_mask.size(0)
    if per_vq.dim() != 1 or per_vq.numel() != batch_size:
        raise ValueError("vq_per_example_loss must have shape [B]")
    per_total = (
        config.node_type_loss_weight * per_node_type
        + config.categorical_loss_weight * per_categorical
        + config.geometry_loss_weight * per_geometry
        + config.edge_presence_loss_weight * per_edge_presence
        + config.edge_type_loss_weight * per_edge_type
        + config.operation_pointer_loss_weight * per_operation_pointer
        + config.vq_loss_weight * per_vq
    )
    per_example = {
        "total": per_total,
        "node_type": per_node_type,
        "categorical_attributes": per_categorical,
        "geometry": per_geometry,
        "edge_presence": per_edge_presence,
        "edge_type": per_edge_type,
        "operation_pointer": per_operation_pointer,
        "vq_commitment": per_vq,
    }
    return FlatMixedVQLoss(
        per_total.mean(),
        per_node_type.mean(),
        per_categorical.mean(),
        per_geometry.mean(),
        per_edge_presence.mean(),
        per_edge_type.mean(),
        per_operation_pointer.mean(),
        per_vq.mean(),
        per_example,
    )


def _per_example_selected_cross_entropy(logits, targets, mask):
    values = []
    for index in range(logits.size(0)):
        selected = mask[index]
        if selected.any():
            value = F.cross_entropy(
                logits[index][selected],
                targets[index][selected],
                reduction="mean",
            )
        else:
            value = logits[index].sum() * 0.0
        values.append(value)
    return torch.stack(values)


def _per_example_smooth_l1(prediction, target, mask):
    values = []
    for index in range(prediction.size(0)):
        selected = mask[index]
        if selected.any():
            value = F.smooth_l1_loss(
                prediction[index][selected],
                target[index][selected],
                reduction="mean",
            )
        else:
            value = prediction[index].sum() * 0.0
        values.append(value)
    return torch.stack(values)


def _per_example_balanced_presence(logits, positives, negatives):
    values = []
    for index in range(logits.size(0)):
        parts = []
        if positives[index].any():
            selected = logits[index][positives[index]]
            parts.append(
                F.binary_cross_entropy_with_logits(
                    selected, torch.ones_like(selected)
                )
            )
        if negatives[index].any():
            selected = logits[index][negatives[index]]
            parts.append(
                F.binary_cross_entropy_with_logits(
                    selected, torch.zeros_like(selected)
                )
            )
        value = (
            torch.stack(parts).mean()
            if parts
            else logits[index].sum() * 0.0
        )
        values.append(value)
    return torch.stack(values)
