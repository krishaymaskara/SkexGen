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
    """Return independently normalized named components and their weighted sum."""

    node_mask = target["node_mask"]
    node_type = _selected_cross_entropy(
        output.node_type_logits, target["node_type_ids"], node_mask
    )
    categorical_parts = tuple(
        _selected_cross_entropy(
            logits,
            target["categorical_attributes"][..., index],
            node_mask,
        )
        for index, logits in enumerate(output.categorical_logits)
    )
    categorical = torch.stack(categorical_parts).mean()

    applicable = target["geometry_mask"] & node_mask.unsqueeze(-1)
    if applicable.any():
        geometry = F.smooth_l1_loss(
            output.geometry[applicable],
            target["geometry"][applicable],
            reduction="mean",
        )
    else:
        geometry = output.geometry.sum() * 0.0

    presence_target, edge_type_target, valid_pairs = dense_edge_targets(
        target, output.edge_presence_logits.size(1)
    )
    positives = valid_pairs & presence_target
    negatives = valid_pairs & ~presence_target
    presence_parts = []
    if positives.any():
        presence_parts.append(
            F.binary_cross_entropy_with_logits(
                output.edge_presence_logits[positives],
                torch.ones_like(output.edge_presence_logits[positives]),
            )
        )
    if negatives.any():
        presence_parts.append(
            F.binary_cross_entropy_with_logits(
                output.edge_presence_logits[negatives],
                torch.zeros_like(output.edge_presence_logits[negatives]),
            )
        )
    edge_presence = (
        torch.stack(presence_parts).mean()
        if presence_parts
        else output.edge_presence_logits.sum() * 0.0
    )
    edge_type = _selected_cross_entropy(
        output.edge_type_logits, edge_type_target, positives
    )

    operation_count = target["operation_sequence"].size(1)
    pointer_logits = output.operation_pointer_logits[:, :operation_count]
    operation_pointer = _selected_cross_entropy(
        pointer_logits,
        target["operation_sequence"],
        target["operation_mask"],
    )
    vq = output.vq_loss
    total = (
        config.node_type_loss_weight * node_type
        + config.categorical_loss_weight * categorical
        + config.geometry_loss_weight * geometry
        + config.edge_presence_loss_weight * edge_presence
        + config.edge_type_loss_weight * edge_type
        + config.operation_pointer_loss_weight * operation_pointer
        + config.vq_loss_weight * vq
    )
    return FlatMixedVQLoss(
        total,
        node_type,
        categorical,
        geometry,
        edge_presence,
        edge_type,
        operation_pointer,
        vq,
    )


def _selected_cross_entropy(logits, targets, mask):
    if mask.any():
        return F.cross_entropy(logits[mask], targets[mask], reduction="mean")
    return logits.sum() * 0.0
