"""Per-example normalized graph-edge loss replacing all flat structure losses."""

from __future__ import annotations

from dataclasses import dataclass, fields

import torch
from torch.nn import functional as F

from prototype.flat_baseline.constrained_v4 import ConstrainedProfileV4Output
from prototype.flat_baseline.constrained_v4_config import ConstrainedProfileV4Config
from prototype.flat_baseline.constrained_v4_losses import constrained_profile_v4_loss

from .config import GraphV1Config
from .graph_tensors import graph_targets_from_reconstruction_batch
from .model import GraphV1Output


@dataclass(frozen=True)
class GraphV1Loss:
    total: torch.Tensor
    node_type: torch.Tensor
    categorical_attributes: torch.Tensor
    reference_plane_category: torch.Tensor
    remaining_geometry: torch.Tensor
    profile_family: torch.Tensor
    profile_parameter: torch.Tensor
    graph_edge: torch.Tensor
    none_class: torch.Tensor
    positive_edge: torch.Tensor
    edge_type: torch.Tensor
    vq_commitment: torch.Tensor
    profile_family_count: int
    profile_parameter_count: int
    positive_pair_count: int
    negative_pair_count: int
    per_example: dict

    def as_dict(self):
        return {
            "total": self.total,
            "node_type": self.node_type,
            "categorical_attributes": self.categorical_attributes,
            "reference_plane_category": self.reference_plane_category,
            "remaining_geometry": self.remaining_geometry,
            "profile_family": self.profile_family,
            "profile_parameter": self.profile_parameter,
            "graph_edge": self.graph_edge,
            "none_class": self.none_class,
            "positive_edge": self.positive_edge,
            "edge_type": self.edge_type,
            "vq_commitment": self.vq_commitment,
        }


def graph_v1_loss(output, target, profile_targets, config):
    if not isinstance(output, GraphV1Output):
        raise TypeError("output must be GraphV1Output")
    if not isinstance(config, GraphV1Config):
        raise TypeError("config must be GraphV1Config")
    config.validate()
    compatible_config = _v4_config(config)
    compatible_output = ConstrainedProfileV4Output(**{
        field.name: getattr(output, field.name)
        for field in fields(ConstrainedProfileV4Output)
    })
    base = constrained_profile_v4_loss(
        compatible_output, target, profile_targets, compatible_config
    )
    graph_targets = graph_targets_from_reconstruction_batch(
        target, output.graph_edge_logits.size(1)
    )
    per_graph = _per_example_graph_cross_entropy(
        output.graph_edge_logits,
        graph_targets.edge_class_ids,
        graph_targets.active_pair_mask,
    )
    none_mask = graph_targets.active_pair_mask & (graph_targets.edge_class_ids == 0)
    positive_mask = graph_targets.active_pair_mask & (graph_targets.edge_class_ids != 0)
    per_none = _per_example_graph_cross_entropy(
        output.graph_edge_logits, graph_targets.edge_class_ids, none_mask
    )
    per_positive = _per_example_graph_cross_entropy(
        output.graph_edge_logits, graph_targets.edge_class_ids, positive_mask
    )
    per_total = (
        base.per_example["total"]
        - compatible_config.edge_presence_loss_weight
        * base.per_example["edge_presence"]
        - compatible_config.edge_type_loss_weight
        * base.per_example["edge_type"]
        - compatible_config.operation_pointer_loss_weight
        * base.per_example["operation_pointer"]
        + config.graph_edge_loss_weight * per_graph
    )
    per_example = {
        "total": per_total,
        "node_type": base.per_example["node_type"],
        "categorical_attributes": base.per_example["categorical_attributes"],
        "reference_plane_category": base.per_example["reference_plane_category"],
        "remaining_geometry": base.per_example["remaining_geometry"],
        "profile_family": base.per_example["profile_family"],
        "profile_parameter": base.per_example["profile_parameter"],
        "graph_edge": per_graph,
        "none_class": per_none,
        "positive_edge": per_positive,
        "edge_type": per_positive,
        "vq_commitment": base.per_example["vq_commitment"],
    }
    for name, values in per_example.items():
        if not torch.isfinite(values).all():
            raise ValueError("{} graph loss must be finite".format(name))
    return GraphV1Loss(
        per_total.mean(),
        base.node_type,
        base.categorical_attributes,
        base.reference_plane_category,
        base.remaining_geometry,
        base.profile_family,
        base.profile_parameter,
        per_graph.mean(),
        per_none.mean(),
        per_positive.mean(),
        per_positive.mean(),
        base.vq_commitment,
        base.profile_family_count,
        base.profile_parameter_count,
        int(positive_mask.sum().item()),
        int(none_mask.sum().item()),
        per_example,
    )


def _per_example_graph_cross_entropy(logits, targets, mask):
    if (
        logits.dim() != 4
        or targets.shape != logits.shape[:3]
        or mask.shape != targets.shape
        or targets.dtype != torch.long
        or mask.dtype != torch.bool
    ):
        raise ValueError("graph loss tensors are misaligned")
    values = []
    for index in range(logits.size(0)):
        selected = mask[index]
        if selected.any():
            value = F.cross_entropy(
                logits[index][selected], targets[index][selected], reduction="mean"
            )
        else:
            value = logits[index].sum() * 0.0
        values.append(value)
    return torch.stack(values)


def _v4_config(config):
    names = {field.name for field in fields(ConstrainedProfileV4Config)}
    identity = {
        "model_name", "model_config_version", "decoder_contract_version",
        "checkpoint_version", "profile_family_order",
        "learned_geometry_channel_indices", "canonical_plane_contract_id",
        "base_geometry_contract_id", "categorical_selection_contract_id",
        "categorical_selection_contract",
    }
    return ConstrainedProfileV4Config(**{
        name: getattr(config, name) for name in names - identity
    })
