"""PyTorch 1.11 tensor construction for canonical reference planes."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from prototype.model_data.geometry import GEOMETRY_WIDTH
from prototype.model_data.vocab import NODE_TYPES, REFERENCE_PLANES

from .reference_plane_geometry import (
    REFERENCE_PLANE_ABSOLUTE_TOLERANCE,
    ReferencePlaneGeometryError,
    canonical_reference_plane_values,
    supported_reference_planes,
)


VALID_REFERENCE_PLANE_NAMES = supported_reference_planes()
VALID_REFERENCE_PLANE_IDS = tuple(
    REFERENCE_PLANES.id(name) for name in VALID_REFERENCE_PLANE_NAMES
)
REFERENCE_PLANE_NODE_TYPE_ID = NODE_TYPES.id("reference_plane")

if tuple(REFERENCE_PLANES.tokens[index] for index in VALID_REFERENCE_PLANE_IDS) \
        != VALID_REFERENCE_PLANE_NAMES:
    raise AssertionError("reference-plane vocabulary IDs changed")


@dataclass(frozen=True)
class TensorReferencePlaneGeometry:
    geometry: torch.Tensor
    geometry_mask: torch.Tensor
    applicable_mask: torch.Tensor


def canonicalize_reference_plane_tensors(
    node_type_ids,
    reference_plane_ids,
    floating_reference,
    node_mask=None,
):
    """Construct width-39 canonical planes for applicable predicted nodes."""

    leading_shape = _validate_inputs(
        node_type_ids, reference_plane_ids, floating_reference, node_mask
    )
    if node_mask is None:
        node_mask = torch.ones(
            leading_shape, dtype=torch.bool, device=node_type_ids.device
        )
    applicable = node_mask & (node_type_ids == REFERENCE_PLANE_NODE_TYPE_ID)
    valid_category = torch.zeros_like(applicable)
    for category_id in VALID_REFERENCE_PLANE_IDS:
        valid_category = valid_category | (reference_plane_ids == category_id)
    invalid = applicable & ~valid_category
    if invalid.any():
        positions = torch.nonzero(invalid, as_tuple=False).detach().cpu().tolist()
        values = reference_plane_ids[invalid].detach().cpu().tolist()
        raise ReferencePlaneGeometryError(
            "invalid_predicted_reference_plane_category",
            "positions={} ids={}".format(positions, values),
        )
    geometry = floating_reference.new_zeros(leading_shape + (GEOMETRY_WIDTH,))
    geometry_mask = torch.zeros(
        leading_shape + (GEOMETRY_WIDTH,),
        dtype=torch.bool,
        device=node_type_ids.device,
    )
    plane_region = geometry[..., :9]
    for name, category_id in zip(
        VALID_REFERENCE_PLANE_NAMES, VALID_REFERENCE_PLANE_IDS
    ):
        selected = applicable & (reference_plane_ids == category_id)
        if selected.any():
            values = floating_reference.new_tensor(
                canonical_reference_plane_values(name)
            )
            plane_region = torch.where(
                selected.unsqueeze(-1), values, plane_region
            )
    geometry[..., :9] = plane_region
    geometry_mask[..., :9] = applicable.unsqueeze(-1).expand(
        leading_shape + (9,)
    )
    return TensorReferencePlaneGeometry(
        geometry.contiguous(), geometry_mask.contiguous(), applicable
    )


def validate_authoritative_reference_plane_tensors(
    node_type_ids,
    categorical_attributes,
    geometry,
    geometry_mask,
    node_mask,
):
    """Reject controlled targets that disagree with their plane category."""

    if (
        not torch.is_tensor(categorical_attributes)
        or categorical_attributes.dtype != torch.long
        or categorical_attributes.shape != node_type_ids.shape + (9,)
    ):
        raise ValueError("categorical attributes must have shape [B,N,9]")
    expected_shape = node_type_ids.shape + (GEOMETRY_WIDTH,)
    if (
        not torch.is_tensor(geometry)
        or geometry.shape != expected_shape
        or not geometry.dtype.is_floating_point
        or not torch.is_tensor(geometry_mask)
        or geometry_mask.shape != expected_shape
        or geometry_mask.dtype != torch.bool
    ):
        raise ValueError("authoritative geometry tensors are misaligned")
    canonical = canonicalize_reference_plane_tensors(
        node_type_ids,
        categorical_attributes[..., 3],
        geometry[..., 33:39],
        node_mask,
    )
    applicable = canonical.applicable_mask
    if applicable.any():
        if not torch.equal(
            geometry_mask[..., :9][applicable],
            canonical.geometry_mask[..., :9][applicable],
        ):
            raise ReferencePlaneGeometryError(
                "authoritative_reference_plane_mask_mismatch",
                "applicable reference-plane mask differs",
            )
        differences = (
            geometry[..., :9][applicable]
            - canonical.geometry[..., :9][applicable]
        ).abs()
        if (differences > REFERENCE_PLANE_ABSOLUTE_TOLERANCE).any():
            raise ReferencePlaneGeometryError(
                "authoritative_reference_plane_geometry_mismatch",
                "maximum normalized error {!r}".format(
                    float(differences.max().item())
                ),
            )
    non_applicable = ~applicable
    if geometry_mask[..., :9][non_applicable].any():
        raise ReferencePlaneGeometryError(
            "authoritative_reference_plane_mask_mismatch",
            "non-reference-plane channels must be inapplicable",
        )
    return canonical


def _validate_inputs(
    node_type_ids, reference_plane_ids, floating_reference, node_mask
):
    if not torch.is_tensor(node_type_ids) or node_type_ids.dtype != torch.long:
        raise TypeError("node type IDs must use torch.long")
    if node_type_ids.dim() < 1 or node_type_ids.numel() == 0:
        raise ValueError("node type IDs need nonempty leading dimensions")
    leading_shape = tuple(node_type_ids.shape)
    if (
        not torch.is_tensor(reference_plane_ids)
        or reference_plane_ids.dtype != torch.long
        or tuple(reference_plane_ids.shape) != leading_shape
    ):
        raise ValueError("reference-plane IDs must align with node types")
    if (
        not torch.is_tensor(floating_reference)
        or not floating_reference.dtype.is_floating_point
        or tuple(floating_reference.shape[:-1]) != leading_shape
        or floating_reference.size(-1) == 0
    ):
        raise ValueError("floating reference must align with node types")
    if (
        node_type_ids.device != reference_plane_ids.device
        or node_type_ids.device != floating_reference.device
    ):
        raise ValueError("reference-plane tensors must share one device")
    if not torch.isfinite(floating_reference).all():
        raise ValueError("floating reference must be finite")
    if node_mask is not None and (
        not torch.is_tensor(node_mask)
        or node_mask.dtype != torch.bool
        or tuple(node_mask.shape) != leading_shape
        or node_mask.device != node_type_ids.device
    ):
        raise ValueError("node mask must align with node types")
    return leading_shape
