"""Integrated teacher-forced losses for the constrained-profile V4 model."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from prototype.constrained_profile_decoder import (
    ProfileHeadOutput,
    constrained_profile_loss,
)
from prototype.profile_geometry_torch import TensorProfileTargets
from prototype.model_data.vocab import NODE_TYPES

from .constrained_v4 import (
    REMAINING_CATEGORICAL_TARGET_INDICES,
    ConstrainedProfileV4Output,
    select_remaining_geometry,
    validate_profile_targets_against_tensor_target,
)
from .constrained_v4_config import ConstrainedProfileV4Config
from .losses import (
    _per_example_balanced_presence,
    _per_example_selected_cross_entropy,
    _per_example_smooth_l1,
    dense_edge_targets,
)


@dataclass(frozen=True)
class ConstrainedProfileV4Loss:
    """All retained V4 losses with profile terms reported separately."""

    total: torch.Tensor
    node_type: torch.Tensor
    categorical_attributes: torch.Tensor
    reference_plane_category: torch.Tensor
    remaining_geometry: torch.Tensor
    profile_family: torch.Tensor
    profile_parameter: torch.Tensor
    edge_presence: torch.Tensor
    edge_type: torch.Tensor
    operation_pointer: torch.Tensor
    vq_commitment: torch.Tensor
    profile_family_count: int
    profile_parameter_count: int
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
            "edge_presence": self.edge_presence,
            "edge_type": self.edge_type,
            "operation_pointer": self.operation_pointer,
            "vq_commitment": self.vq_commitment,
        }


def constrained_profile_v4_loss(
    output,
    target,
    profile_targets,
    config,
):
    """Return independently normalized V4 reconstruction losses."""

    if not isinstance(output, ConstrainedProfileV4Output):
        raise TypeError("output must be ConstrainedProfileV4Output")
    if not isinstance(profile_targets, TensorProfileTargets):
        raise TypeError("profile_targets must be TensorProfileTargets")
    if not isinstance(config, ConstrainedProfileV4Config):
        raise TypeError("config must be ConstrainedProfileV4Config")
    config.validate()
    validate_profile_targets_against_tensor_target(
        profile_targets, target
    )

    node_mask = target["node_mask"]
    per_node_type = _per_example_selected_cross_entropy(
        output.node_type_logits, target["node_type_ids"], node_mask
    )
    if len(output.categorical_logits) != len(
        REMAINING_CATEGORICAL_TARGET_INDICES
    ):
        raise ValueError("V4 must expose exactly five categorical heads")
    categorical_masks = tuple(
        node_mask & (target["node_type_ids"] == NODE_TYPES.id("reference_plane"))
        if target_index == 3 else node_mask
        for target_index in REMAINING_CATEGORICAL_TARGET_INDICES
    )
    per_categorical_parts = tuple(
        _per_example_selected_cross_entropy(
            logits,
            target["categorical_attributes"][..., target_index],
            applicable_mask,
        )
        for logits, target_index, applicable_mask in zip(
            output.categorical_logits,
            REMAINING_CATEGORICAL_TARGET_INDICES,
            categorical_masks,
        )
    )
    per_categorical = torch.stack(per_categorical_parts).mean(dim=0)
    per_reference_plane_category = per_categorical_parts[3]

    target_remaining = select_remaining_geometry(target["geometry"])
    authoritative_mask = (
        select_remaining_geometry(target["geometry_mask"])
        & node_mask.unsqueeze(-1)
    )
    if not torch.equal(
        output.remaining_geometry_mask, authoritative_mask
    ):
        raise ValueError(
            "remaining geometry mask disagrees with authoritative target"
        )
    per_remaining_geometry = _per_example_smooth_l1(
        output.remaining_geometry,
        target_remaining,
        authoritative_mask,
    )

    profile = constrained_profile_loss(
        ProfileHeadOutput(
            output.profile_family_logits,
            output.raw_profile_parameters,
        ),
        profile_targets,
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
        + config.geometry_loss_weight * per_remaining_geometry
        + config.profile_family_loss_weight
        * profile.per_example_family_loss
        + config.profile_parameter_loss_weight
        * profile.per_example_parameter_loss
        + config.edge_presence_loss_weight * per_edge_presence
        + config.edge_type_loss_weight * per_edge_type
        + config.operation_pointer_loss_weight * per_operation_pointer
        + config.vq_loss_weight * per_vq
    )
    per_example = {
        "total": per_total,
        "node_type": per_node_type,
        "categorical_attributes": per_categorical,
        "reference_plane_category": per_reference_plane_category,
        "remaining_geometry": per_remaining_geometry,
        "profile_family": profile.per_example_family_loss,
        "profile_parameter": profile.per_example_parameter_loss,
        "edge_presence": per_edge_presence,
        "edge_type": per_edge_type,
        "operation_pointer": per_operation_pointer,
        "vq_commitment": per_vq,
    }
    for name, values in per_example.items():
        if not torch.isfinite(values).all():
            raise ValueError("{} V4 loss must be finite".format(name))
    return ConstrainedProfileV4Loss(
        per_total.mean(),
        per_node_type.mean(),
        per_categorical.mean(),
        per_reference_plane_category.mean(),
        per_remaining_geometry.mean(),
        profile.family_loss,
        profile.parameter_loss,
        per_edge_presence.mean(),
        per_edge_type.mean(),
        per_operation_pointer.mean(),
        per_vq.mean(),
        profile.family_count,
        profile.parameter_count,
        per_example,
    )
