"""Canonical normalized geometry for the controlled profile families."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence, Tuple

from prototype.controlled_data.config import GeneratorConfig
from prototype.controlled_data.factors import PrimitiveFamily
from prototype.model_data.geometry import (
    GEOMETRY_WIDTH,
    LENGTH_SCALE,
    MAX_PRIMITIVES,
    NORMALIZED_MAX,
    NORMALIZED_MIN,
)
from prototype.model_data.records import ReconstructionTarget
from prototype.model_data.vocab import NODE_TYPES, PRIMITIVE_TYPES


PrimitiveTypeIds = Tuple[int, int, int, int]
ProfileGeometry = Tuple[Tuple[float, ...], Tuple[bool, ...]]
GeometryCoefficients = Tuple[Tuple[float, float, float], ...]

PROFILE_FAMILIES = (
    PrimitiveFamily.CIRCLE,
    PrimitiveFamily.RECTANGLE_LINES,
    PrimitiveFamily.CAPSULE_LINE_ARC,
)
NO_PROFILE_FAMILY_ID = -1

_DEFAULT_PHYSICAL_EXTENTS = (
    GeneratorConfig.__dataclass_fields__["sketch_extents"].default
)
PROFILE_EXTENT_MIN = min(_DEFAULT_PHYSICAL_EXTENTS) / LENGTH_SCALE
PROFILE_EXTENT_MAX = max(_DEFAULT_PHYSICAL_EXTENTS) / LENGTH_SCALE

_NONE_ID = PRIMITIVE_TYPES.id(None)
_ARC_ID = PRIMITIVE_TYPES.id("arc")
_CIRCLE_ID = PRIMITIVE_TYPES.id("circle")
_LINE_ID = PRIMITIVE_TYPES.id("line")
_ZERO_COEFFICIENTS = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class CanonicalProfileTemplate:
    """One affine source of primitive types, geometry, and applicability."""

    family: PrimitiveFamily
    primitive_type_ids: PrimitiveTypeIds
    geometry_coefficients: GeometryCoefficients
    geometry_bias: Tuple[float, ...]
    geometry_mask: Tuple[bool, ...]
    center_extent_margins: Tuple[float, float]


@dataclass(frozen=True)
class ProfileParameters:
    """Compact normalized parameters for one canonical controlled profile."""

    family: PrimitiveFamily
    center_x: float
    center_y: float
    extent: float


def _template(
    family: PrimitiveFamily,
    primitive_type_ids: PrimitiveTypeIds,
    channel_coefficients,
    center_extent_margins: Tuple[float, float],
) -> CanonicalProfileTemplate:
    coefficients = [_ZERO_COEFFICIENTS] * GEOMETRY_WIDTH
    bias = [0.0] * GEOMETRY_WIDTH
    mask = [False] * GEOMETRY_WIDTH
    for channel, values in channel_coefficients:
        if not 0 <= channel < GEOMETRY_WIDTH or mask[channel]:
            raise AssertionError("profile template channel is invalid or duplicated")
        coefficients[channel] = values
        mask[channel] = True
    return CanonicalProfileTemplate(
        family,
        primitive_type_ids,
        tuple(coefficients),
        tuple(bias),
        tuple(mask),
        center_extent_margins,
    )


_CIRCLE_TEMPLATE = _template(
    PrimitiveFamily.CIRCLE,
    (_CIRCLE_ID, _NONE_ID, _NONE_ID, _NONE_ID),
    (
        (9, (1.0, 0.0, 0.0)),
        (10, (0.0, 1.0, 0.0)),
        (11, (0.0, 0.0, 0.5)),
    ),
    # The schema stores center and radius, not circle boundary points. Existing
    # validation bounds those stored channels independently.
    (0.0, 0.0),
)

_RECTANGLE_TEMPLATE = _template(
    PrimitiveFamily.RECTANGLE_LINES,
    (_LINE_ID, _LINE_ID, _LINE_ID, _LINE_ID),
    (
        (9, (1.0, 0.0, -0.5)),
        (10, (0.0, 1.0, -0.25)),
        (11, (1.0, 0.0, 0.5)),
        (12, (0.0, 1.0, -0.25)),
        (15, (1.0, 0.0, 0.5)),
        (16, (0.0, 1.0, -0.25)),
        (17, (1.0, 0.0, 0.5)),
        (18, (0.0, 1.0, 0.25)),
        (21, (1.0, 0.0, 0.5)),
        (22, (0.0, 1.0, 0.25)),
        (23, (1.0, 0.0, -0.5)),
        (24, (0.0, 1.0, 0.25)),
        (27, (1.0, 0.0, -0.5)),
        (28, (0.0, 1.0, 0.25)),
        (29, (1.0, 0.0, -0.5)),
        (30, (0.0, 1.0, -0.25)),
    ),
    (0.5, 0.25),
)

_CAPSULE_TEMPLATE = _template(
    PrimitiveFamily.CAPSULE_LINE_ARC,
    # Canonical slots are sorted by element ID: arc_1, arc_3, line_0, line_2.
    (_ARC_ID, _ARC_ID, _LINE_ID, _LINE_ID),
    (
        (9, (1.0, 0.0, 0.25)),
        (10, (0.0, 1.0, -0.25)),
        (11, (1.0, 0.0, 0.5)),
        (12, (0.0, 1.0, 0.0)),
        (13, (1.0, 0.0, 0.25)),
        (14, (0.0, 1.0, 0.25)),
        (15, (1.0, 0.0, -0.25)),
        (16, (0.0, 1.0, 0.25)),
        (17, (1.0, 0.0, -0.5)),
        (18, (0.0, 1.0, 0.0)),
        (19, (1.0, 0.0, -0.25)),
        (20, (0.0, 1.0, -0.25)),
        (21, (1.0, 0.0, -0.25)),
        (22, (0.0, 1.0, -0.25)),
        (23, (1.0, 0.0, 0.25)),
        (24, (0.0, 1.0, -0.25)),
        (27, (1.0, 0.0, 0.25)),
        (28, (0.0, 1.0, 0.25)),
        (29, (1.0, 0.0, -0.25)),
        (30, (0.0, 1.0, 0.25)),
    ),
    (0.5, 0.25),
)

PROFILE_TEMPLATES = (
    _CIRCLE_TEMPLATE,
    _RECTANGLE_TEMPLATE,
    _CAPSULE_TEMPLATE,
)
_FAMILY_TO_TEMPLATE = {
    template.family: template for template in PROFILE_TEMPLATES
}
_PATTERN_TO_FAMILY_ID = {
    template.primitive_type_ids: class_id
    for class_id, template in enumerate(PROFILE_TEMPLATES)
}

if tuple(template.family for template in PROFILE_TEMPLATES) != PROFILE_FAMILIES:
    raise AssertionError("profile templates must follow the frozen class order")


def profile_family_id(family: PrimitiveFamily) -> int:
    """Map a supported family enum to its frozen class ID."""

    _validate_family(family)
    return PROFILE_FAMILIES.index(family)


def profile_family_from_id(class_id: int) -> PrimitiveFamily:
    """Map one frozen class ID to its family enum."""

    if (
        isinstance(class_id, bool)
        or not isinstance(class_id, int)
        or not 0 <= class_id < len(PROFILE_FAMILIES)
    ):
        raise ValueError("unsupported profile family ID {!r}".format(class_id))
    return PROFILE_FAMILIES[class_id]


def profile_family_id_from_primitive_type_ids(
    primitive_type_ids: Sequence[int],
) -> int:
    """Map the existing four categorical primitive slots to a class ID."""

    pattern = _validated_primitive_pattern(primitive_type_ids)
    try:
        return _PATTERN_TO_FAMILY_ID[pattern]
    except KeyError as exc:
        raise ValueError(
            "unsupported canonical primitive pattern {!r}".format(pattern)
        ) from exc


def canonical_primitive_type_ids_from_family_id(
    class_id: int,
) -> PrimitiveTypeIds:
    """Return deterministic primitive slots for one frozen class ID."""

    profile_family_from_id(class_id)
    return PROFILE_TEMPLATES[class_id].primitive_type_ids


def canonical_profile_template(
    family: PrimitiveFamily,
) -> CanonicalProfileTemplate:
    """Return the one authoritative affine template for ``family``."""

    _validate_family(family)
    return _FAMILY_TO_TEMPLATE[family]


def profile_family_from_primitive_type_ids(
    primitive_type_ids: Sequence[int],
) -> PrimitiveFamily:
    """Map the existing four categorical primitive slots to a profile family."""

    return profile_family_from_id(
        profile_family_id_from_primitive_type_ids(primitive_type_ids)
    )


def canonical_primitive_type_ids(family: PrimitiveFamily) -> PrimitiveTypeIds:
    """Return the exact existing four-slot categorical pattern for ``family``."""

    return canonical_profile_template(family).primitive_type_ids


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
    return extract_profile_parameters_from_row(attributes[4:8], geometry, mask)


def extract_profile_parameters_from_row(
    primitive_type_ids: Sequence[int],
    geometry,
    geometry_mask,
) -> ProfileParameters:
    """Extract parameters from one canonical sketch row without target coupling."""

    try:
        normalized = tuple(
            _finite_float(item, "target sketch geometry") for item in geometry
        )
        mask = tuple(geometry_mask)
    except TypeError as exc:
        raise ValueError("target sketch geometry fields must be iterable") from exc
    if len(normalized) != GEOMETRY_WIDTH:
        raise ValueError(
            "target sketch geometry must have width {}".format(GEOMETRY_WIDTH)
        )
    if len(mask) != GEOMETRY_WIDTH:
        raise ValueError(
            "target sketch geometry mask must have width {}".format(GEOMETRY_WIDTH)
        )
    if any(not isinstance(item, bool) for item in mask):
        raise ValueError("target sketch geometry mask must contain only Booleans")
    if any(
        item < NORMALIZED_MIN or item > NORMALIZED_MAX for item in normalized
    ):
        raise ValueError("target sketch geometry is outside normalized bounds")

    family = profile_family_from_primitive_type_ids(primitive_type_ids)
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

    template = canonical_profile_template(family)
    parameters = (
        _finite_float(center_x, "center_x"),
        _finite_float(center_y, "center_y"),
        _finite_float(extent, "extent"),
    )
    if parameters[2] <= 0.0:
        raise ValueError("extent must be strictly positive")
    if not NORMALIZED_MIN <= parameters[0] <= NORMALIZED_MAX:
        raise ValueError("center_x is outside normalized bounds")
    if not NORMALIZED_MIN <= parameters[1] <= NORMALIZED_MAX:
        raise ValueError("center_y is outside normalized bounds")
    if parameters[2] > NORMALIZED_MAX:
        raise ValueError("extent is outside normalized bounds")

    values = []
    for coefficients, bias, applicable in zip(
        template.geometry_coefficients,
        template.geometry_bias,
        template.geometry_mask,
    ):
        if not applicable:
            values.append(0.0)
            continue
        value = (
            parameters[0] * coefficients[0]
            + parameters[1] * coefficients[1]
            + parameters[2] * coefficients[2]
            + bias
        )
        values.append(0.0 if value == 0.0 else value)

    if any(
        item < NORMALIZED_MIN or item > NORMALIZED_MAX
        for item, applicable in zip(values, template.geometry_mask)
        if applicable
    ):
        raise ValueError("constructed profile geometry is outside normalized bounds")
    return tuple(values), template.geometry_mask


def _validate_family(family: PrimitiveFamily) -> None:
    if not isinstance(family, PrimitiveFamily) or family not in _FAMILY_TO_TEMPLATE:
        raise ValueError("unsupported profile family {!r}".format(family))


def _validated_primitive_pattern(
    primitive_type_ids: Sequence[int],
) -> PrimitiveTypeIds:
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
    return pattern


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
