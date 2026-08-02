"""Deterministic controlled-domain revolve-axis geometry contract."""

from __future__ import annotations

from dataclasses import dataclass
import math


AXIS_GEOMETRY_CONTRACT_ID = "controlled-revolve-axis-geometry-v6"
AXIS_CONSTRUCTION_ALGORITHM_ID = "universal-sketch-local-axis-v1"
AXIS_GEOMETRY_CHANNEL_INDICES = (33, 34, 35, 36, 37, 38)
AXIS_APPLICABLE_CHANNEL_INDICES = (33, 34, 35, 36)
AXIS_SERIALIZATION_ORDER = (
    "axis_point_x",
    "axis_point_y",
    "axis_direction_x",
    "axis_direction_y",
    "extrude_distance",
    "revolve_angle",
)
AXIS_COORDINATE_SPACE = "sketch_local_2d"
AXIS_CHANNEL_SCALES = (4.0, 4.0, 1.0, 1.0, 4.0, 360.0)
CANONICAL_NORMALIZED_AXIS = (0.0, 0.0, 0.0, 1.0)
CANONICAL_PHYSICAL_AXIS = (0.0, 0.0, 0.0, 1.0)
CANONICAL_AXIS_CHANNELS = CANONICAL_NORMALIZED_AXIS + (0.0, 0.0)
CANONICAL_AXIS_MASK = (True, True, True, True, False, False)
RAW_CONSTRAINED_AXIS_EVIDENCE_POLICY = (
    "raw-six-channel-shadow-constrained-axis-feedback-v1"
)


class AxisGeometryContractError(ValueError):
    """Axis geometry or applicability violates the frozen V6 contract."""

    def __init__(self, code, detail):
        self.code = code
        self.detail = detail
        super().__init__("{}: {}".format(code, detail))


@dataclass(frozen=True)
class CanonicalAxisGeometry:
    normalized_channels: tuple
    geometry_mask: tuple


def construct_canonical_axis_geometry():
    """Return the sole controlled-corpus axis in channels 33--38 order."""

    return CanonicalAxisGeometry(
        CANONICAL_AXIS_CHANNELS,
        CANONICAL_AXIS_MASK,
    )


def validate_canonical_axis_geometry(normalized_channels, geometry_mask):
    """Strictly verify an externally supplied controlled V6 axis row."""

    try:
        channels = tuple(normalized_channels)
        mask = tuple(geometry_mask)
    except TypeError as exc:
        raise AxisGeometryContractError(
            "invalid_axis_geometry", "axis geometry and mask must be iterable"
        ) from exc
    if len(channels) != 6 or len(mask) != 6:
        raise AxisGeometryContractError(
            "invalid_axis_geometry", "axis geometry and mask must have width six"
        )
    if any(type(value) is not bool for value in mask):
        raise AxisGeometryContractError(
            "invalid_axis_geometry_mask", "axis mask entries must be Boolean"
        )
    if mask != CANONICAL_AXIS_MASK:
        raise AxisGeometryContractError(
            "invalid_axis_geometry_mask", "axis mask differs from channels 33--36"
        )
    values = []
    for index, value in enumerate(channels):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise AxisGeometryContractError(
                "invalid_axis_geometry", "axis channel {} is not numeric".format(index)
            )
        numeric = float(value)
        if not math.isfinite(numeric):
            raise AxisGeometryContractError(
                "invalid_axis_geometry", "axis channel {} is nonfinite".format(index)
            )
        values.append(0.0 if numeric == 0.0 else numeric)
    if tuple(values) != CANONICAL_AXIS_CHANNELS:
        raise AxisGeometryContractError(
            "invalid_axis_geometry",
            "controlled axes require point (0,0) and direction (0,1)",
        )
    return construct_canonical_axis_geometry()


def axis_geometry_metadata():
    """Return deterministic JSON-compatible V6 axis provenance."""

    return {
        "contract_id": AXIS_GEOMETRY_CONTRACT_ID,
        "construction_algorithm_id": AXIS_CONSTRUCTION_ALGORITHM_ID,
        "serialization_order": list(AXIS_SERIALIZATION_ORDER),
        "channel_indices": list(AXIS_GEOMETRY_CHANNEL_INDICES),
        "applicable_channel_indices": list(AXIS_APPLICABLE_CHANNEL_INDICES),
        "coordinate_space": AXIS_COORDINATE_SPACE,
        "channel_scales": list(AXIS_CHANNEL_SCALES),
        "canonical_normalized_channels": list(CANONICAL_AXIS_CHANNELS),
        "canonical_physical_axis": list(CANONICAL_PHYSICAL_AXIS),
        "canonical_mask": list(CANONICAL_AXIS_MASK),
        "raw_constrained_evidence_policy": (
            RAW_CONSTRAINED_AXIS_EVIDENCE_POLICY
        ),
    }
