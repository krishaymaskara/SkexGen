"""Batched PyTorch construction for the controlled V6 axis contract."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from prototype.model_data.geometry import GEOMETRY_WIDTH
from prototype.model_data.vocab import NODE_TYPES

from .axis_geometry import CANONICAL_AXIS_CHANNELS


@dataclass(frozen=True)
class ConstrainedAxisTensors:
    geometry: torch.Tensor
    geometry_mask: torch.Tensor
    raw_axis_geometry: torch.Tensor
    constrained_axis_geometry: torch.Tensor
    axis_geometry_correction_mask: torch.Tensor
    axis_node_mask: torch.Tensor


def constrain_axis_geometry_tensors(
    node_type_ids,
    raw_remaining_geometry,
    geometry,
    geometry_mask,
    active_mask=None,
):
    """Replace only applicable constrained-axis channels with the canonical row."""

    leading = _validate_inputs(
        node_type_ids,
        raw_remaining_geometry,
        geometry,
        geometry_mask,
        active_mask,
    )
    if active_mask is None:
        active_mask = torch.ones_like(node_type_ids, dtype=torch.bool)
    axis_mask = active_mask & (node_type_ids == NODE_TYPES.id("axis"))
    canonical = raw_remaining_geometry.new_tensor(
        CANONICAL_AXIS_CHANNELS
    ).expand(leading + (6,))
    constrained_evidence = torch.where(
        axis_mask.unsqueeze(-1),
        canonical,
        torch.zeros_like(raw_remaining_geometry),
    ).contiguous()
    applicable = torch.tensor(
        (True, True, True, True, False, False),
        dtype=torch.bool,
        device=node_type_ids.device,
    ).expand(leading + (6,))
    correction = (
        axis_mask.unsqueeze(-1)
        & applicable
        & (raw_remaining_geometry != canonical)
    ).contiguous()
    result_geometry = geometry.clone()
    result_geometry[..., 33:37] = torch.where(
        axis_mask.unsqueeze(-1),
        canonical[..., :4],
        result_geometry[..., 33:37],
    )
    result_geometry = torch.where(
        geometry_mask, result_geometry, torch.zeros_like(result_geometry)
    ).contiguous()
    expected_axis_mask = torch.zeros_like(geometry_mask)
    expected_axis_mask[..., 33:37] = axis_mask.unsqueeze(-1)
    actual_axis_mask = geometry_mask & axis_mask.unsqueeze(-1)
    if not torch.equal(actual_axis_mask, expected_axis_mask):
        raise ValueError("axis geometry mask differs from channels 33--36")
    return ConstrainedAxisTensors(
        result_geometry,
        geometry_mask.contiguous(),
        raw_remaining_geometry.contiguous(),
        constrained_evidence,
        correction,
        axis_mask.contiguous(),
    )


def _validate_inputs(
    node_type_ids,
    raw_remaining_geometry,
    geometry,
    geometry_mask,
    active_mask,
):
    if not torch.is_tensor(node_type_ids) or node_type_ids.dtype != torch.long:
        raise TypeError("node_type_ids must be a long tensor")
    leading = tuple(node_type_ids.shape)
    if (
        not torch.is_tensor(raw_remaining_geometry)
        or not raw_remaining_geometry.dtype.is_floating_point
        or tuple(raw_remaining_geometry.shape) != leading + (6,)
        or raw_remaining_geometry.device != node_type_ids.device
    ):
        raise ValueError("raw_remaining_geometry must align as [..., 6]")
    if (
        not torch.is_tensor(geometry)
        or not geometry.dtype.is_floating_point
        or tuple(geometry.shape) != leading + (GEOMETRY_WIDTH,)
        or geometry.device != node_type_ids.device
        or geometry.dtype != raw_remaining_geometry.dtype
    ):
        raise ValueError("geometry must align as [..., 39]")
    if (
        not torch.is_tensor(geometry_mask)
        or geometry_mask.dtype != torch.bool
        or tuple(geometry_mask.shape) != leading + (GEOMETRY_WIDTH,)
        or geometry_mask.device != node_type_ids.device
    ):
        raise ValueError("geometry_mask must align as Boolean [..., 39]")
    if active_mask is not None and (
        not torch.is_tensor(active_mask)
        or active_mask.dtype != torch.bool
        or tuple(active_mask.shape) != leading
        or active_mask.device != node_type_ids.device
    ):
        raise ValueError("active_mask must align with node_type_ids")
    if (
        not torch.isfinite(raw_remaining_geometry).all()
        or not torch.isfinite(geometry).all()
    ):
        raise ValueError("axis construction inputs must be finite")
    if node_type_ids.numel() and (
        int(node_type_ids.min().item()) < 0
        or int(node_type_ids.max().item()) >= len(NODE_TYPES.tokens)
    ):
        raise ValueError("node_type_ids contain an unsupported ID")
    return leading
