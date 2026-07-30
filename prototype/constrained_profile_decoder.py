"""Shared neural heads and losses for constrained controlled profiles."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from prototype.profile_geometry import (
    NO_PROFILE_FAMILY_ID,
    PROFILE_FAMILIES,
)
from prototype.profile_geometry_torch import (
    TensorProfileTargets,
    canonicalize_profile_tensors,
    constrain_profile_parameters,
    extract_profile_target_tensors,
)


PROFILE_FAMILY_CLASS_COUNT = len(PROFILE_FAMILIES)
PROFILE_PARAMETER_COUNT = 3

# These unweighted utilities are intended to enter a successor model with the
# same initial weights as one categorical field and the legacy geometry field.
DEFAULT_PROFILE_FAMILY_LOSS_WEIGHT = 1.0
DEFAULT_PROFILE_PARAMETER_LOSS_WEIGHT = 1.0


@dataclass(frozen=True)
class ProfileHeadOutput:
    """Raw profile predictions aligned with decoded node states."""

    family_logits: torch.Tensor
    raw_parameters: torch.Tensor


@dataclass(frozen=True)
class ProfileLossTerm:
    """One independently normalized profile loss and its accounting."""

    loss: torch.Tensor
    per_example: torch.Tensor
    count: int
    per_example_count: torch.Tensor


@dataclass(frozen=True)
class ConstrainedProfileLoss:
    """Reusable family and compact-parameter losses."""

    family_loss: torch.Tensor
    parameter_loss: torch.Tensor
    family_count: int
    parameter_count: int
    per_example_family_loss: torch.Tensor
    per_example_parameter_loss: torch.Tensor
    per_example_family_count: torch.Tensor
    per_example_parameter_count: torch.Tensor


class ConstrainedProfileHeads(nn.Module):
    """Direct profile-family and compact-parameter projections."""

    def __init__(self, decoded_state_dim: int):
        super().__init__()
        if (
            isinstance(decoded_state_dim, bool)
            or not isinstance(decoded_state_dim, int)
            or decoded_state_dim <= 0
        ):
            raise ValueError("decoded_state_dim must be a positive integer")
        self.decoded_state_dim = decoded_state_dim
        self.family_head = nn.Linear(
            decoded_state_dim, PROFILE_FAMILY_CLASS_COUNT
        )
        self.parameter_head = nn.Linear(
            decoded_state_dim, PROFILE_PARAMETER_COUNT
        )

    def forward(
        self, decoded_node_states: torch.Tensor
    ) -> ProfileHeadOutput:
        if not torch.is_tensor(decoded_node_states):
            raise TypeError("decoded node states must be a PyTorch tensor")
        if not decoded_node_states.dtype.is_floating_point:
            raise TypeError("decoded node states must be floating point")
        if decoded_node_states.dim() != 3:
            raise ValueError(
                "decoded node states must have shape [B, N, D]"
            )
        if decoded_node_states.size(-1) != self.decoded_state_dim:
            raise ValueError(
                "decoded node-state width does not match decoded_state_dim"
            )
        if decoded_node_states.dtype != self.family_head.weight.dtype:
            raise TypeError(
                "decoded node states must match profile-head dtype"
            )
        if decoded_node_states.device != self.family_head.weight.device:
            raise ValueError(
                "decoded node states must share the profile-head device"
            )
        if not torch.isfinite(decoded_node_states).all():
            raise ValueError("decoded node states must be finite")

        family_logits = self.family_head(decoded_node_states)
        raw_parameters = self.parameter_head(decoded_node_states)
        if (
            not torch.isfinite(family_logits).all()
            or not torch.isfinite(raw_parameters).all()
        ):
            raise ValueError("profile-head outputs must be finite")
        return ProfileHeadOutput(family_logits, raw_parameters)


def profile_targets_for_loss(
    target, reference_parameters: torch.Tensor
) -> TensorProfileTargets:
    """Extract targets on the device and dtype used by parameter loss."""

    if not torch.is_tensor(reference_parameters):
        raise TypeError("reference parameters must be a PyTorch tensor")
    if not reference_parameters.dtype.is_floating_point:
        raise TypeError("reference parameters must be floating point")
    return extract_profile_target_tensors(
        target,
        torch_module=torch,
        device=reference_parameters.device,
        dtype=reference_parameters.dtype,
    )


def profile_family_loss(
    family_logits: torch.Tensor,
    target_family_ids: torch.Tensor,
    sketch_mask: torch.Tensor,
) -> ProfileLossTerm:
    """Cross-entropy independently normalized over sketches per example."""

    _validate_family_loss_inputs(
        family_logits, target_family_ids, sketch_mask
    )
    per_example = []
    for batch_index in range(family_logits.size(0)):
        selected = sketch_mask[batch_index]
        if selected.any():
            value = F.cross_entropy(
                family_logits[batch_index][selected],
                target_family_ids[batch_index][selected],
                reduction="mean",
            )
        else:
            value = family_logits[batch_index].sum() * 0.0
        if not torch.isfinite(value):
            raise ValueError("profile-family loss must be finite")
        per_example.append(value)
    values = torch.stack(per_example)
    counts = sketch_mask.long().sum(dim=1)
    return ProfileLossTerm(
        values.mean(),
        values,
        int(counts.sum().item()),
        counts,
    )


def profile_parameter_loss(
    raw_parameters: torch.Tensor,
    target_family_ids: torch.Tensor,
    target_parameters: torch.Tensor,
    sketch_mask: torch.Tensor,
) -> ProfileLossTerm:
    """Smooth L1 on three compact parameters using target-family routing."""

    _validate_parameter_loss_inputs(
        raw_parameters,
        target_family_ids,
        target_parameters,
        sketch_mask,
    )
    # This shared call also validates every applicable target against the
    # canonical family-specific domain without defining another convention.
    canonicalize_profile_tensors(
        target_family_ids, target_parameters, sketch_mask
    )
    constrained = constrain_profile_parameters(
        raw_parameters, target_family_ids, sketch_mask
    )

    per_example = []
    for batch_index in range(raw_parameters.size(0)):
        selected = sketch_mask[batch_index]
        if selected.any():
            value = F.smooth_l1_loss(
                constrained[batch_index][selected],
                target_parameters[batch_index][selected],
                reduction="mean",
            )
        else:
            value = raw_parameters[batch_index].sum() * 0.0
        if not torch.isfinite(value):
            raise ValueError("profile-parameter loss must be finite")
        per_example.append(value)
    values = torch.stack(per_example)
    family_counts = sketch_mask.long().sum(dim=1)
    counts = family_counts * PROFILE_PARAMETER_COUNT
    return ProfileLossTerm(
        values.mean(),
        values,
        int(counts.sum().item()),
        counts,
    )


def constrained_profile_loss(
    output: ProfileHeadOutput,
    targets: TensorProfileTargets,
) -> ConstrainedProfileLoss:
    """Return both unweighted Stage 2A profile losses and exact counts."""

    if not isinstance(output, ProfileHeadOutput):
        raise TypeError("output must be a ProfileHeadOutput")
    if not isinstance(targets, TensorProfileTargets):
        raise TypeError("targets must be TensorProfileTargets")
    family = profile_family_loss(
        output.family_logits, targets.family_ids, targets.sketch_mask
    )
    parameters = profile_parameter_loss(
        output.raw_parameters,
        targets.family_ids,
        targets.parameters,
        targets.sketch_mask,
    )
    return ConstrainedProfileLoss(
        family.loss,
        parameters.loss,
        family.count,
        parameters.count,
        family.per_example,
        parameters.per_example,
        family.per_example_count,
        parameters.per_example_count,
    )


def _validate_family_loss_inputs(
    family_logits, target_family_ids, sketch_mask
) -> None:
    _validate_family_targets(target_family_ids, sketch_mask)
    if (
        not torch.is_tensor(family_logits)
        or not family_logits.dtype.is_floating_point
    ):
        raise TypeError("profile-family logits must be floating point")
    if (
        family_logits.dim() != 3
        or family_logits.size(-1) != PROFILE_FAMILY_CLASS_COUNT
    ):
        raise ValueError("profile-family logits must have shape [B, N, 3]")
    if family_logits.size(0) == 0:
        raise ValueError("profile losses require a positive batch size")
    if family_logits.shape[:-1] != target_family_ids.shape:
        raise ValueError("profile-family logits and targets are misaligned")
    if (
        family_logits.device != target_family_ids.device
        or family_logits.device != sketch_mask.device
    ):
        raise ValueError("profile-family loss tensors must share one device")
    if not torch.isfinite(family_logits).all():
        raise ValueError("profile-family logits must be finite")


def _validate_parameter_loss_inputs(
    raw_parameters, target_family_ids, target_parameters, sketch_mask
) -> None:
    _validate_family_targets(target_family_ids, sketch_mask)
    for value, name in (
        (raw_parameters, "raw profile parameters"),
        (target_parameters, "target profile parameters"),
    ):
        if not torch.is_tensor(value) or not value.dtype.is_floating_point:
            raise TypeError("{} must be floating point".format(name))
        if value.dim() != 3 or value.size(-1) != PROFILE_PARAMETER_COUNT:
            raise ValueError("{} must have shape [B, N, 3]".format(name))
        if not torch.isfinite(value).all():
            raise ValueError("{} must be finite".format(name))
    if raw_parameters.size(0) == 0:
        raise ValueError("profile losses require a positive batch size")
    if raw_parameters.shape != target_parameters.shape:
        raise ValueError("raw and target profile parameters are misaligned")
    if raw_parameters.shape[:-1] != target_family_ids.shape:
        raise ValueError("profile parameters and family targets are misaligned")
    if raw_parameters.dtype != target_parameters.dtype:
        raise TypeError("raw and target profile parameters must share a dtype")
    if (
        raw_parameters.device != target_parameters.device
        or raw_parameters.device != target_family_ids.device
        or raw_parameters.device != sketch_mask.device
    ):
        raise ValueError("profile-parameter loss tensors must share one device")


def _validate_family_targets(target_family_ids, sketch_mask) -> None:
    if (
        not torch.is_tensor(target_family_ids)
        or target_family_ids.dtype != torch.long
    ):
        raise TypeError("target profile family IDs must use torch.long")
    if (
        not torch.is_tensor(sketch_mask)
        or sketch_mask.dtype != torch.bool
    ):
        raise TypeError("profile sketch mask must use torch.bool")
    if target_family_ids.dim() != 2 or sketch_mask.dim() != 2:
        raise ValueError("profile family targets must have shape [B, N]")
    if target_family_ids.shape != sketch_mask.shape:
        raise ValueError("profile family targets and sketch mask are misaligned")
    supported = (
        (target_family_ids >= 0)
        & (target_family_ids < PROFILE_FAMILY_CLASS_COUNT)
    )
    if (sketch_mask & ~supported).any():
        raise ValueError(
            "sketch positions contain unsupported profile family IDs"
        )
    if (
        (~sketch_mask)
        & (target_family_ids != NO_PROFILE_FAMILY_ID)
    ).any():
        raise ValueError(
            "non-sketch positions must use the no-profile family ID"
        )
