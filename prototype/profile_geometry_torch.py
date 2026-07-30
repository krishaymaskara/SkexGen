"""Differentiable tensor operations for the shared profile contract."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from prototype.model_data.geometry import (
    GEOMETRY_WIDTH,
    NORMALIZED_MAX,
    NORMALIZED_MIN,
)
from prototype.model_data.records import ReconstructionBatch, ReconstructionTarget
from prototype.model_data.vocab import NODE_TYPES, PRIMITIVE_TYPES
from prototype.profile_geometry import (
    NO_PROFILE_FAMILY_ID,
    PROFILE_EXTENT_MAX,
    PROFILE_EXTENT_MIN,
    PROFILE_FAMILIES,
    PROFILE_TEMPLATES,
    extract_profile_parameters_from_row,
    profile_family_id,
)


@dataclass(frozen=True)
class TensorCanonicalProfile:
    """Canonical tensor outputs aligned to the supplied leading dimensions."""

    primitive_type_ids: Any
    geometry: Any
    geometry_mask: Any


@dataclass(frozen=True)
class TensorProfileTargets:
    """Compact family and parameter targets for sketches in a target container."""

    family_ids: Any
    parameters: Any
    sketch_mask: Any


def constrain_profile_parameters(
    raw_parameters,
    family_ids,
    sketch_mask,
    *,
    extent_min=PROFILE_EXTENT_MIN,
    extent_max=PROFILE_EXTENT_MAX,
):
    """Map raw ``[..., 3]`` tensors into the configured profile domain."""

    torch = _torch_for(raw_parameters)
    _validate_profile_tensor_shapes(
        raw_parameters, family_ids, sketch_mask, torch
    )
    lower, upper = _validated_extent_bounds(extent_min, extent_max)
    if not torch.isfinite(raw_parameters).all():
        raise ValueError("raw profile parameters must be finite")

    safe_family_ids = _safe_family_ids(family_ids, sketch_mask, torch)
    margin_table = raw_parameters.new_tensor(
        tuple(template.center_extent_margins for template in PROFILE_TEMPLATES)
    )
    margins = margin_table[safe_family_ids]
    extent = lower + (upper - lower) * torch.sigmoid(raw_parameters[..., 2])
    center_scale = 1.0 - margins * extent.unsqueeze(-1)
    centers = torch.tanh(raw_parameters[..., :2]) * center_scale
    parameters = torch.cat((centers, extent.unsqueeze(-1)), dim=-1)
    return torch.where(
        sketch_mask.unsqueeze(-1),
        parameters,
        torch.zeros_like(parameters),
    )


def canonicalize_profile_tensors(
    family_ids,
    constrained_parameters,
    sketch_mask,
    *,
    extent_min=PROFILE_EXTENT_MIN,
    extent_max=PROFILE_EXTENT_MAX,
) -> TensorCanonicalProfile:
    """Build primitive IDs, 39-channel geometry, and masks differentiably."""

    torch = _torch_for(constrained_parameters)
    _validate_profile_tensor_shapes(
        constrained_parameters, family_ids, sketch_mask, torch
    )
    lower, upper = _validated_extent_bounds(extent_min, extent_max)
    if not torch.isfinite(constrained_parameters).all():
        raise ValueError("constrained profile parameters must be finite")
    safe_family_ids = _safe_family_ids(family_ids, sketch_mask, torch)
    _validate_constrained_domain(
        constrained_parameters,
        safe_family_ids,
        sketch_mask,
        lower,
        upper,
        torch,
    )

    coefficient_table = constrained_parameters.new_tensor(
        tuple(
            template.geometry_coefficients for template in PROFILE_TEMPLATES
        )
    )
    bias_table = constrained_parameters.new_tensor(
        tuple(template.geometry_bias for template in PROFILE_TEMPLATES)
    )
    mask_table = torch.tensor(
        tuple(template.geometry_mask for template in PROFILE_TEMPLATES),
        dtype=torch.bool,
        device=constrained_parameters.device,
    )
    primitive_table = torch.tensor(
        tuple(template.primitive_type_ids for template in PROFILE_TEMPLATES),
        dtype=torch.long,
        device=constrained_parameters.device,
    )

    coefficients = coefficient_table[safe_family_ids]
    bias = bias_table[safe_family_ids]
    geometry = (
        coefficients * constrained_parameters.unsqueeze(-2)
    ).sum(dim=-1) + bias
    geometry = torch.where(
        sketch_mask.unsqueeze(-1), geometry, torch.zeros_like(geometry)
    )
    geometry_mask = mask_table[safe_family_ids] & sketch_mask.unsqueeze(-1)
    primitive_type_ids = primitive_table[safe_family_ids]
    none_ids = torch.full_like(
        primitive_type_ids, PRIMITIVE_TYPES.id(None)
    )
    primitive_type_ids = torch.where(
        sketch_mask.unsqueeze(-1), primitive_type_ids, none_ids
    )

    if not torch.isfinite(geometry).all():
        raise ValueError("canonical profile geometry must be finite")
    applicable = geometry[geometry_mask]
    if applicable.numel() and (
        (applicable < NORMALIZED_MIN).any()
        or (applicable > NORMALIZED_MAX).any()
    ):
        raise ValueError("canonical profile geometry is outside normalized bounds")
    return TensorCanonicalProfile(
        primitive_type_ids, geometry, geometry_mask
    )


def extract_profile_target_tensors(
    target,
    *,
    torch_module=None,
    device=None,
    dtype=None,
    extent_min=PROFILE_EXTENT_MIN,
    extent_max=PROFILE_EXTENT_MAX,
) -> TensorProfileTargets:
    """Extract compact tensor targets from one target or padded target batch."""

    torch = _resolve_torch(torch_module)
    lower, upper = _validated_extent_bounds(extent_min, extent_max)
    dtype = torch.float32 if dtype is None else dtype
    if not getattr(dtype, "is_floating_point", False):
        raise TypeError("profile target parameter dtype must be floating point")

    if isinstance(target, ReconstructionTarget):
        node_type_rows = (target.node_type_ids,)
        attribute_rows = (target.categorical_attributes,)
        geometry_rows = (target.geometry,)
        geometry_mask_rows = (target.geometry_mask,)
        node_mask_rows = ((True,) * len(target.node_type_ids),)
        squeeze = True
    elif isinstance(target, ReconstructionBatch):
        node_type_rows = target.node_type_ids
        attribute_rows = target.categorical_attributes
        geometry_rows = target.geometry
        geometry_mask_rows = target.geometry_mask
        node_mask_rows = target.node_mask
        squeeze = False
    else:
        raise TypeError(
            "target must be a ReconstructionTarget or ReconstructionBatch"
        )

    _validate_target_batch_alignment(
        node_type_rows,
        attribute_rows,
        geometry_rows,
        geometry_mask_rows,
        node_mask_rows,
    )
    family_rows = []
    parameter_rows = []
    sketch_rows = []
    sketch_id = NODE_TYPES.id("sketch")
    for node_types, attributes, geometry, masks, node_mask in zip(
        node_type_rows,
        attribute_rows,
        geometry_rows,
        geometry_mask_rows,
        node_mask_rows,
    ):
        families = []
        parameters = []
        sketches = []
        for node_type, row, values, mask, present in zip(
            node_types, attributes, geometry, masks, node_mask
        ):
            if not isinstance(present, bool):
                raise ValueError("target node mask must contain only Booleans")
            is_sketch = present and node_type == sketch_id
            if is_sketch:
                if len(row) != 9:
                    raise ValueError(
                        "target categorical attributes must have width 9"
                    )
                extracted = extract_profile_parameters_from_row(
                    row[4:8], values, mask
                )
                if not lower <= extracted.extent <= upper:
                    raise ValueError(
                        "target profile extent is outside configured bounds"
                    )
                families.append(profile_family_id(extracted.family))
                parameters.append(
                    (
                        extracted.center_x,
                        extracted.center_y,
                        extracted.extent,
                    )
                )
            else:
                families.append(NO_PROFILE_FAMILY_ID)
                parameters.append((0.0, 0.0, 0.0))
            sketches.append(is_sketch)
        family_rows.append(tuple(families))
        parameter_rows.append(tuple(parameters))
        sketch_rows.append(tuple(sketches))

    result = TensorProfileTargets(
        torch.tensor(
            family_rows, dtype=torch.long, device=device
        ).contiguous(),
        torch.tensor(
            parameter_rows, dtype=dtype, device=device
        ).contiguous(),
        torch.tensor(
            sketch_rows, dtype=torch.bool, device=device
        ).contiguous(),
    )
    if not squeeze:
        return result
    return TensorProfileTargets(
        result.family_ids[0],
        result.parameters[0],
        result.sketch_mask[0],
    )


def _validate_profile_tensor_shapes(
    parameters, family_ids, sketch_mask, torch
) -> None:
    if not torch.is_tensor(parameters) or not parameters.dtype.is_floating_point:
        raise TypeError("profile parameters must be a floating-point tensor")
    if not torch.is_tensor(family_ids) or family_ids.dtype != torch.long:
        raise TypeError("profile family IDs must be a torch.long tensor")
    if not torch.is_tensor(sketch_mask) or sketch_mask.dtype != torch.bool:
        raise TypeError("sketch mask must be a torch.bool tensor")
    if parameters.dim() < 1 or parameters.size(-1) != 3:
        raise ValueError("profile parameters must have shape [..., 3]")
    expected = parameters.shape[:-1]
    if family_ids.shape != expected or sketch_mask.shape != expected:
        raise ValueError(
            "family IDs and sketch mask must match parameter leading dimensions"
        )
    if (
        family_ids.device != parameters.device
        or sketch_mask.device != parameters.device
    ):
        raise ValueError("profile tensors must share one device")


def _safe_family_ids(family_ids, sketch_mask, torch):
    supported = (family_ids >= 0) & (family_ids < len(PROFILE_FAMILIES))
    if (sketch_mask & ~supported).any():
        raise ValueError("sketch positions contain unsupported profile family IDs")
    if ((~sketch_mask) & (family_ids != NO_PROFILE_FAMILY_ID)).any():
        raise ValueError(
            "non-sketch positions must use the no-profile family ID"
        )
    return torch.where(sketch_mask, family_ids, torch.zeros_like(family_ids))


def _validate_constrained_domain(
    parameters,
    safe_family_ids,
    sketch_mask,
    lower,
    upper,
    torch,
) -> None:
    if ((~sketch_mask).unsqueeze(-1) & (parameters != 0.0)).any():
        raise ValueError("non-sketch profile parameters must be zero")
    if not sketch_mask.any():
        return
    selected = parameters[sketch_mask]
    extent = selected[:, 2]
    if (extent < lower).any() or (extent > upper).any():
        raise ValueError("profile extent is outside configured bounds")
    margins = parameters.new_tensor(
        tuple(template.center_extent_margins for template in PROFILE_TEMPLATES)
    )[safe_family_ids][sketch_mask]
    limits = 1.0 - margins * extent.unsqueeze(-1)
    if (selected[:, :2].abs() > limits).any():
        raise ValueError("profile center would generate out-of-bounds geometry")


def _validated_extent_bounds(extent_min, extent_max):
    for value, name in (
        (extent_min, "extent_min"),
        (extent_max, "extent_max"),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise ValueError("{} must be finite numeric".format(name))
    lower = float(extent_min)
    upper = float(extent_max)
    if not 0.0 < lower < upper <= NORMALIZED_MAX:
        raise ValueError(
            "extent bounds must satisfy 0 < extent_min < extent_max <= 1"
        )
    return lower, upper


def _validate_target_batch_alignment(
    node_types, attributes, geometry, geometry_masks, node_masks
) -> None:
    batch_size = len(node_types)
    if any(
        len(rows) != batch_size
        for rows in (attributes, geometry, geometry_masks, node_masks)
    ):
        raise ValueError("target batch fields have inconsistent batch sizes")
    for rows in zip(node_types, attributes, geometry, geometry_masks, node_masks):
        width = len(rows[0])
        if any(len(row) != width for row in rows[1:]):
            raise ValueError("target batch node fields are misaligned")


def _torch_for(value):
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for tensor profile geometry") from exc
    if not torch.is_tensor(value):
        raise TypeError("profile parameters must be a PyTorch tensor")
    return torch


def _resolve_torch(torch_module):
    if torch_module is not None:
        return torch_module
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for tensor profile targets") from exc
    return torch
