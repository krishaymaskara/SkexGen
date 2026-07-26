"""Fixed normalized geometry channels shared by all adapters."""

from __future__ import annotations

import math

from prototype.representation.model import (
    ArcGeometry,
    AxisGeometry,
    CircleGeometry,
    ExtrudeGeometry,
    LineGeometry,
    PlaneGeometry,
    RevolveGeometry,
)

from .errors import ModelDataError


LENGTH_SCALE = 4.0
ANGLE_SCALE = 360.0
PRIMITIVE_SLOT_WIDTH = 6
MAX_PRIMITIVES = 4

GEOMETRY_CHANNELS = (
    "plane_origin_x",
    "plane_origin_y",
    "plane_origin_z",
    "plane_x_axis_x",
    "plane_x_axis_y",
    "plane_x_axis_z",
    "plane_y_axis_x",
    "plane_y_axis_y",
    "plane_y_axis_z",
    *tuple(
        f"primitive_{slot}_{field}"
        for slot in range(MAX_PRIMITIVES)
        for field in ("x0", "y0", "x1", "y1", "x2", "y2")
    ),
    "axis_point_x",
    "axis_point_y",
    "axis_direction_x",
    "axis_direction_y",
    "extrude_distance",
    "revolve_angle",
)
GEOMETRY_WIDTH = len(GEOMETRY_CHANNELS)
NORMALIZED_MIN = -1.0
NORMALIZED_MAX = 1.0
GEOMETRY_CHANNEL_SCALES = (
    *(LENGTH_SCALE,) * 3,
    *(1.0,) * 6,
    *(LENGTH_SCALE,) * (MAX_PRIMITIVES * PRIMITIVE_SLOT_WIDTH),
    *(LENGTH_SCALE,) * 2,
    *(1.0,) * 2,
    LENGTH_SCALE,
    ANGLE_SCALE,
)

if len(GEOMETRY_CHANNEL_SCALES) != GEOMETRY_WIDTH:
    raise AssertionError("geometry channel scales must match geometry width")


def decode(value) -> tuple[float, ...]:
    if value.encoding.value == "continuous":
        raw = value.values
    else:
        raw = tuple(code * value.scale + value.offset for code in value.values)
    return tuple(0.0 if float(item) == 0.0 else float(item) for item in raw)


def empty_geometry():
    return [0.0] * GEOMETRY_WIDTH, [False] * GEOMETRY_WIDTH


def denormalize_applicable_geometry(values, mask):
    """Return physical values for applicable channels and ``None`` otherwise."""

    try:
        normalized_values = tuple(values)
        applicability = tuple(mask)
    except TypeError as exc:
        raise ModelDataError(
            "invalid_geometry_container",
            "geometry values and mask must be iterable",
        ) from exc
    if len(normalized_values) != GEOMETRY_WIDTH:
        raise ModelDataError(
            "invalid_geometry_width",
            "geometry values must contain exactly {} entries".format(
                GEOMETRY_WIDTH
            ),
        )
    if len(applicability) != GEOMETRY_WIDTH:
        raise ModelDataError(
            "invalid_geometry_mask_width",
            "geometry mask must contain exactly {} entries".format(
                GEOMETRY_WIDTH
            ),
        )
    result = []
    for index, (value, applicable, scale) in enumerate(
        zip(normalized_values, applicability, GEOMETRY_CHANNEL_SCALES)
    ):
        if not isinstance(applicable, bool):
            raise ModelDataError(
                "invalid_geometry_mask_type",
                "geometry mask channel {} must be Boolean".format(index),
            )
        if not applicable:
            result.append(None)
            continue
        if isinstance(value, bool):
            raise ModelDataError(
                "invalid_geometry_value_type",
                "geometry channel {} must be numeric".format(index),
            )
        try:
            normalized = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ModelDataError(
                "invalid_geometry_value_type",
                "geometry channel {} must be numeric".format(index),
            ) from exc
        if not math.isfinite(normalized):
            raise ModelDataError(
                "nonfinite_geometry",
                "geometry channel {} is nonfinite".format(index),
            )
        if not NORMALIZED_MIN <= normalized <= NORMALIZED_MAX:
            raise ModelDataError(
                "geometry_out_of_range",
                "geometry channel {} is outside [{}, {}]".format(
                    index, NORMALIZED_MIN, NORMALIZED_MAX
                ),
            )
        physical = normalized * scale
        result.append(0.0 if physical == 0.0 else physical)
    return tuple(result)


def place_node_geometry(values, mask, geometry):
    if isinstance(geometry, PlaneGeometry):
        _put(values, mask, 0, decode(geometry.origin), LENGTH_SCALE)
        _put(values, mask, 3, decode(geometry.x_axis), 1.0)
        _put(values, mask, 6, decode(geometry.y_axis), 1.0)
    elif isinstance(geometry, AxisGeometry):
        _put(values, mask, 33, decode(geometry.point), LENGTH_SCALE)
        _put(values, mask, 35, decode(geometry.direction), 1.0)
    elif isinstance(geometry, ExtrudeGeometry):
        _put(values, mask, 37, decode(geometry.distance), LENGTH_SCALE)
    elif isinstance(geometry, RevolveGeometry):
        _put(values, mask, 38, decode(geometry.angle_degrees), ANGLE_SCALE)
    else:
        raise ModelDataError(
            "unsupported_node_geometry", type(geometry).__name__
        )


def place_primitive_geometry(values, mask, slot, geometry):
    if not 0 <= slot < MAX_PRIMITIVES:
        raise ModelDataError("too_many_primitives", f"primitive slot {slot}")
    start = 9 + slot * PRIMITIVE_SLOT_WIDTH
    if isinstance(geometry, LineGeometry):
        physical = (*decode(geometry.start), *decode(geometry.end))
    elif isinstance(geometry, ArcGeometry):
        physical = (
            *decode(geometry.start),
            *decode(geometry.midpoint),
            *decode(geometry.end),
        )
    elif isinstance(geometry, CircleGeometry):
        center = decode(geometry.center)
        radius = decode(geometry.radius)
        physical = (*center, radius[0])
    else:
        raise ModelDataError(
            "unsupported_primitive_geometry", type(geometry).__name__
        )
    _put(values, mask, start, physical, LENGTH_SCALE)


def _put(values, mask, start, physical, scale):
    for offset, item in enumerate(physical):
        normalized = float(item) / scale
        if not math.isfinite(normalized):
            raise ModelDataError(
                "nonfinite_geometry", f"channel {start + offset} is nonfinite"
            )
        if not NORMALIZED_MIN <= normalized <= NORMALIZED_MAX:
            raise ModelDataError(
                "geometry_out_of_range",
                f"channel {start + offset} normalized value {normalized!r} "
                f"is outside [{NORMALIZED_MIN}, {NORMALIZED_MAX}]",
            )
        values[start + offset] = 0.0 if normalized == 0.0 else normalized
        mask[start + offset] = True
