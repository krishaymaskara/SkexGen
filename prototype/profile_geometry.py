"""Canonical normalized geometry for the controlled profile families."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence, Tuple

from prototype.controlled_data.factors import PrimitiveFamily
from prototype.model_data.geometry import (
    GEOMETRY_WIDTH,
    MAX_PRIMITIVES,
    NORMALIZED_MAX,
    NORMALIZED_MIN,
    PRIMITIVE_SLOT_WIDTH,
)
from prototype.model_data.records import ReconstructionTarget
from prototype.model_data.vocab import NODE_TYPES, PRIMITIVE_TYPES


PrimitiveTypeIds = Tuple[int, int, int, int]
ProfileGeometry = Tuple[Tuple[float, ...], Tuple[bool, ...]]

_NONE_ID = PRIMITIVE_TYPES.id(None)
_ARC_ID = PRIMITIVE_TYPES.id("arc")
_CIRCLE_ID = PRIMITIVE_TYPES.id("circle")
_LINE_ID = PRIMITIVE_TYPES.id("line")

_FAMILY_TO_PATTERN = {
    PrimitiveFamily.CIRCLE: (_CIRCLE_ID, _NONE_ID, _NONE_ID, _NONE_ID),
    PrimitiveFamily.RECTANGLE_LINES: (_LINE_ID,) * MAX_PRIMITIVES,
    # Canonical sketch slots are sorted by element ID: arc_1, arc_3,
    # line_0, line_2. This differs from the builder's loop traversal.
    PrimitiveFamily.CAPSULE_LINE_ARC: (
        _ARC_ID,
        _ARC_ID,
        _LINE_ID,
        _LINE_ID,
    ),
}
_PATTERN_TO_FAMILY = {
    pattern: family for family, pattern in _FAMILY_TO_PATTERN.items()
}


@dataclass(frozen=True)
class ProfileParameters:
    """Compact normalized parameters for one canonical controlled profile."""

    family: PrimitiveFamily
    center_x: float
    center_y: float
    extent: float


def profile_family_from_primitive_type_ids(
    primitive_type_ids: Sequence[int],
) -> PrimitiveFamily:
    """Map the existing four categorical primitive slots to a profile family."""

    try:
        pattern = tuple(primitive_type_ids)
    except TypeError as exc:
        raise ValueError(
            "primitive type IDs must be an iterable of four integers"
        ) from exc
    if len(pattern) != MAX_PRIMITIVES:
        raise ValueError("primitive type IDs must contain exactly four entries")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in pattern):
        raise ValueError("primitive type IDs must contain only integers")
    try:
        return _PATTERN_TO_FAMILY[pattern]
    except KeyError as exc:
        raise ValueError(
            "unsupported canonical primitive pattern {!r}".format(pattern)
        ) from exc


def canonical_primitive_type_ids(family: PrimitiveFamily) -> PrimitiveTypeIds:
    """Return the exact existing four-slot categorical pattern for ``family``."""

    _validate_family(family)
    return _FAMILY_TO_PATTERN[family]


def extract_profile_parameters(
    target: ReconstructionTarget,
    sketch_node_index: int,
) -> ProfileParameters:
    """Extract and validate compact parameters from one canonical sketch row."""

    if not isinstance(target, ReconstructionTarget):
        raise TypeError("target must be a ReconstructionTarget")
    if (
        isinstance(sketch_node_index, bool)
        or not isinstance(sketch_node_index, int)
        or not 0 <= sketch_node_index < len(target.node_type_ids)
    ):
        raise ValueError("sketch node index is outside the target")
    if target.node_type_ids[sketch_node_index] != NODE_TYPES.id("sketch"):
        raise ValueError("selected target node is not a sketch")

    try:
        attributes = tuple(target.categorical_attributes[sketch_node_index])
        geometry = tuple(target.geometry[sketch_node_index])
        mask = tuple(target.geometry_mask[sketch_node_index])
    except (IndexError, TypeError) as exc:
        raise ValueError("target sketch fields are malformed or misaligned") from exc
    if len(attributes) != 9:
        raise ValueError("target sketch categorical attributes must have width 9")
    if len(geometry) != GEOMETRY_WIDTH:
        raise ValueError(
            "target sketch geometry must have width {}".format(GEOMETRY_WIDTH)
        )
    if len(mask) != GEOMETRY_WIDTH:
        raise ValueError(
            "target sketch geometry mask must have width {}".format(GEOMETRY_WIDTH)
        )
    if any(not isinstance(item, bool) for item in mask):
        raise ValueError("target sketch geometry mask must contain only Booleans")

    normalized = tuple(
        _finite_float(item, "target sketch geometry") for item in geometry
    )
    if any(
        item < NORMALIZED_MIN or item > NORMALIZED_MAX for item in normalized
    ):
        raise ValueError("target sketch geometry is outside normalized bounds")

    family = profile_family_from_primitive_type_ids(attributes[4:8])
    if family is PrimitiveFamily.CIRCLE:
        center_x, center_y, radius = normalized[9:12]
        extent = radius * 2.0
    elif family is PrimitiveFamily.RECTANGLE_LINES:
        lower_left = normalized[9:11]
        lower_right = normalized[11:13]
        upper_left = normalized[27:29]
        center_x = (lower_left[0] + lower_right[0]) / 2.0
        center_y = (lower_left[1] + upper_left[1]) / 2.0
        extent = lower_right[0] - lower_left[0]
    else:
        right_midpoint = normalized[11:13]
        left_midpoint = normalized[17:19]
        center_x = (right_midpoint[0] + left_midpoint[0]) / 2.0
        center_y = (right_midpoint[1] + left_midpoint[1]) / 2.0
        extent = right_midpoint[0] - left_midpoint[0]

    expected_geometry, expected_mask = construct_profile_geometry(
        family, center_x, center_y, extent
    )
    if mask != expected_mask:
        raise ValueError(
            "target sketch geometry mask is not canonical for its family"
        )
    if normalized != expected_geometry:
        raise ValueError(
            "target sketch geometry is inconsistent with its canonical family"
        )
    return ProfileParameters(family, center_x, center_y, extent)


def construct_profile_geometry(
    family: PrimitiveFamily,
    center_x: float,
    center_y: float,
    extent: float,
) -> ProfileGeometry:
    """Construct the canonical 39-channel normalized sketch row and mask."""

    _validate_family(family)
    cx = _finite_float(center_x, "center_x")
    cy = _finite_float(center_y, "center_y")
    size = _finite_float(extent, "extent")
    if size <= 0.0:
        raise ValueError("extent must be strictly positive")
    if not NORMALIZED_MIN <= cx <= NORMALIZED_MAX:
        raise ValueError("center_x is outside normalized bounds")
    if not NORMALIZED_MIN <= cy <= NORMALIZED_MAX:
        raise ValueError("center_y is outside normalized bounds")
    if size > NORMALIZED_MAX:
        raise ValueError("extent is outside normalized bounds")

    values = [0.0] * GEOMETRY_WIDTH
    mask = [False] * GEOMETRY_WIDTH
    if family is PrimitiveFamily.CIRCLE:
        _put_primitive(values, mask, 0, (cx, cy, size / 2.0))
    elif family is PrimitiveFamily.RECTANGLE_LINES:
        lower_left = (cx - size / 2.0, cy - size / 4.0)
        lower_right = (cx + size / 2.0, cy - size / 4.0)
        upper_right = (cx + size / 2.0, cy + size / 4.0)
        upper_left = (cx - size / 2.0, cy + size / 4.0)
        _put_primitive(values, mask, 0, (*lower_left, *lower_right))
        _put_primitive(values, mask, 1, (*lower_right, *upper_right))
        _put_primitive(values, mask, 2, (*upper_right, *upper_left))
        _put_primitive(values, mask, 3, (*upper_left, *lower_left))
    else:
        radius = size / 4.0
        lower_left = (cx - radius, cy - radius)
        lower_right = (cx + radius, cy - radius)
        upper_right = (cx + radius, cy + radius)
        upper_left = (cx - radius, cy + radius)
        right_midpoint = (cx + size / 2.0, cy)
        left_midpoint = (cx - size / 2.0, cy)
        _put_primitive(
            values,
            mask,
            0,
            (*lower_right, *right_midpoint, *upper_right),
        )
        _put_primitive(
            values,
            mask,
            1,
            (*upper_left, *left_midpoint, *lower_left),
        )
        _put_primitive(values, mask, 2, (*lower_left, *lower_right))
        _put_primitive(values, mask, 3, (*upper_right, *upper_left))

    if any(
        item < NORMALIZED_MIN or item > NORMALIZED_MAX
        for item, applicable in zip(values, mask)
        if applicable
    ):
        raise ValueError("constructed profile geometry is outside normalized bounds")
    return tuple(values), tuple(mask)


def _validate_family(family: PrimitiveFamily) -> None:
    if not isinstance(family, PrimitiveFamily) or family not in _FAMILY_TO_PATTERN:
        raise ValueError("unsupported profile family {!r}".format(family))


def _finite_float(value: float, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError("{} must be a finite number".format(name))
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("{} must be a finite number".format(name)) from exc
    if not math.isfinite(result):
        raise ValueError("{} must be finite".format(name))
    return 0.0 if result == 0.0 else result


def _put_primitive(values, mask, slot, primitive_values) -> None:
    start = 9 + slot * PRIMITIVE_SLOT_WIDTH
    for offset, item in enumerate(primitive_values):
        value = _finite_float(item, "constructed profile geometry")
        values[start + offset] = value
        mask[start + offset] = True
