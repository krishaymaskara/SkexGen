"""Canonical controlled reference-plane geometry shared by decoder versions."""

from __future__ import annotations

from enum import Enum
import math

from prototype.controlled_data.factors import ReferencePlane
from prototype.model_data.geometry import (
    GEOMETRY_CHANNEL_SCALES,
    GEOMETRY_WIDTH,
    denormalize_applicable_geometry,
    empty_geometry,
    place_node_geometry,
)
from prototype.representation.model import NumericValue, PlaneGeometry


CANONICAL_PLANE_CONTRACT_ID = "controlled-reference-plane-canonical-v1"
REFERENCE_PLANE_GEOMETRY_INDICES = tuple(range(9))
REFERENCE_PLANE_SERIALIZED_CHANNELS = (
    "plane_origin_x", "plane_origin_y", "plane_origin_z",
    "plane_x_axis_x", "plane_x_axis_y", "plane_x_axis_z",
    "plane_y_axis_x", "plane_y_axis_y", "plane_y_axis_z",
)
REFERENCE_PLANE_CHANNEL_SCALES = (4.0, 4.0, 4.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
REFERENCE_PLANE_ABSOLUTE_TOLERANCE = 1e-6

_PHYSICAL_FRAMES = {
    ReferencePlane.XY: (
        0.0, 0.0, 0.0,
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
    ),
    ReferencePlane.XZ: (
        0.0, 0.0, 0.0,
        1.0, 0.0, 0.0,
        0.0, 0.0, 1.0,
    ),
    ReferencePlane.YZ: (
        0.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
    ),
}

if tuple(GEOMETRY_CHANNEL_SCALES[:9]) != REFERENCE_PLANE_CHANNEL_SCALES:
    raise AssertionError("reference-plane normalization scales changed")


class ReferencePlaneGeometryError(ValueError):
    """A plane category or canonical geometry contract is invalid."""

    def __init__(self, code, detail):
        self.code = code
        self.detail = detail
        super().__init__("{}: {}".format(code, detail))


def canonical_reference_plane_geometry(plane):
    """Return normalized width-39 geometry and mask for one valid plane."""

    resolved = _resolve_plane(plane)
    physical = _PHYSICAL_FRAMES[resolved]
    values, applicability = empty_geometry()
    place_node_geometry(
        values,
        applicability,
        PlaneGeometry(
            NumericValue.continuous(*physical[:3]),
            NumericValue.continuous(*physical[3:6]),
            NumericValue.continuous(*physical[6:9]),
        ),
    )
    geometry = tuple(values)
    mask = tuple(applicability)
    decoded = denormalize_applicable_geometry(geometry, mask)
    if any(
        not math.isclose(decoded[index], physical[index], rel_tol=0.0,
                         abs_tol=REFERENCE_PLANE_ABSOLUTE_TOLERANCE)
        for index in REFERENCE_PLANE_GEOMETRY_INDICES
    ):
        raise AssertionError("canonical plane normalization does not invert")
    return geometry, mask


def canonical_reference_plane_values(plane):
    """Return the applicable normalized channels in serialized order."""

    geometry, unused_mask = canonical_reference_plane_geometry(plane)
    return geometry[:9]


def canonical_reference_plane_physical_values(plane):
    """Return physical origin/x-axis/y-axis values in serialized order."""

    return _PHYSICAL_FRAMES[_resolve_plane(plane)]


def supported_reference_planes():
    return tuple(item.value for item in ReferencePlane)


def _resolve_plane(plane):
    if isinstance(plane, ReferencePlane):
        return plane
    if isinstance(plane, Enum):
        plane = plane.value
    if isinstance(plane, str):
        try:
            return ReferencePlane(plane)
        except ValueError:
            pass
    raise ReferencePlaneGeometryError(
        "invalid_predicted_reference_plane_category", repr(plane)
    )
