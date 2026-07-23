"""Pure analytical Boolean-feasibility policy for the frozen v1 builders."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Any

from prototype.representation.model import BooleanMode, Direction


class FeasibilityStatus(str, Enum):
    ACCEPTED = "accepted"
    CUT_NO_POSITIVE_VOLUME_OVERLAP = "cut_no_positive_volume_overlap"
    CUT_COMPLETE_SUBTRACTION = "cut_complete_subtraction"
    JOIN_TOOL_CONTAINED = "join_tool_contained"
    JOIN_DUPLICATE_GEOMETRY = "join_duplicate_geometry"
    JOIN_DISCONNECTED = "join_disconnected"
    MIXED_CONTAINMENT_NOT_CERTIFIED = "mixed_containment_not_certified"


@dataclass(frozen=True)
class FeasibilityResult:
    """Structured result retained independently of CAD-kernel behavior."""

    status: FeasibilityStatus
    detail: str

    @property
    def accepted(self) -> bool:
        return self.status is FeasibilityStatus.ACCEPTED


def evaluate_feasibility(source: Any) -> FeasibilityResult:
    """Classify one physical factor assignment before history construction.

    The policy is exact for same-kind EE/RR histories. Mixed ER/RE histories
    use exact sector-overlap tests plus conservative profile-specific
    containment certificates and explicit non-containment witnesses.
    """

    template = _enum_value(source.operation_template)
    if template in {"E", "R"}:
        return _result(FeasibilityStatus.ACCEPTED, "standalone NEW_BODY feature")
    if source.later_boolean_mode not in {BooleanMode.JOIN, BooleanMode.CUT}:
        raise ValueError("depth-two feasibility requires JOIN or CUT")
    if template == "EE":
        return _evaluate_ee(source)
    if template == "RR":
        return _evaluate_rr(source)
    if template == "ER":
        return _evaluate_er(source)
    if template == "RE":
        return _evaluate_re(source)
    raise ValueError(f"unsupported operation template {template!r}")


def extent_order_relation(source: Any) -> str:
    """Return the second sketch's symbolic extent relation to the first."""

    if len(source.sketch_extents) == 1:
        return "single"
    first, second = source.sketch_extents
    if second < first:
        return "smaller"
    if second > first:
        return "larger"
    return "equal"


def direction_relation(source: Any) -> str:
    if len(source.directions) == 1:
        return "single"
    return "same" if source.directions[0] is source.directions[1] else "opposite"


def _evaluate_ee(source: Any) -> FeasibilityResult:
    extent_1, extent_2 = source.sketch_extents
    distance_1, distance_2 = source.operation_parameters
    same = direction_relation(source) == "same"
    if source.later_boolean_mode is BooleanMode.JOIN:
        if same and extent_2 == extent_1 and distance_2 == distance_1:
            return _result(
                FeasibilityStatus.JOIN_DUPLICATE_GEOMETRY,
                "same-direction extrusions have identical profile and interval",
            )
        if same and extent_2 <= extent_1 and distance_2 <= distance_1:
            return _result(
                FeasibilityStatus.JOIN_TOOL_CONTAINED,
                "second extrusion is contained in the current extrusion",
            )
        # Opposite directions share the complete profile intersection on z=0,
        # so the union is face-connected and adds positive volume.
        return _result(FeasibilityStatus.ACCEPTED, "extrusion union adds volume")

    if not same:
        return _result(
            FeasibilityStatus.CUT_NO_POSITIVE_VOLUME_OVERLAP,
            "opposite-side extrusions meet only on the profile plane",
        )
    if extent_2 >= extent_1 and distance_2 >= distance_1:
        return _result(
            FeasibilityStatus.CUT_COMPLETE_SUBTRACTION,
            "second extrusion contains the current extrusion",
        )
    return _result(FeasibilityStatus.ACCEPTED, "extrusion cut overlaps and leaves residual")


def _evaluate_rr(source: Any) -> FeasibilityResult:
    extent_1, extent_2 = source.sketch_extents
    angle_1, angle_2 = source.operation_parameters
    same = direction_relation(source) == "same"
    both_full = angle_1 == 360.0 and angle_2 == 360.0
    if source.later_boolean_mode is BooleanMode.JOIN:
        if extent_2 == extent_1 and ((same and angle_2 == angle_1) or both_full):
            return _result(
                FeasibilityStatus.JOIN_DUPLICATE_GEOMETRY,
                "revolves occupy identical profile and angular sector",
            )
        sector_contained = (same and angle_2 <= angle_1) or angle_1 == 360.0
        if extent_2 <= extent_1 and sector_contained:
            return _result(
                FeasibilityStatus.JOIN_TOOL_CONTAINED,
                "second revolve is contained in the current revolve",
            )
        return _result(FeasibilityStatus.ACCEPTED, "revolve union adds volume")

    if not same and angle_1 + angle_2 <= 360.0:
        return _result(
            FeasibilityStatus.CUT_NO_POSITIVE_VOLUME_OVERLAP,
            "opposite revolve sectors have no positive angular overlap",
        )
    sector_contains_current = (same and angle_2 >= angle_1) or angle_2 == 360.0
    if extent_2 >= extent_1 and sector_contains_current:
        return _result(
            FeasibilityStatus.CUT_COMPLETE_SUBTRACTION,
            "second revolve contains the current revolve",
        )
    return _result(FeasibilityStatus.ACCEPTED, "revolve cut overlaps and leaves residual")


def _evaluate_er(source: Any) -> FeasibilityResult:
    extrusion_extent, revolve_extent = source.sketch_extents
    extrusion_distance, revolve_angle = source.operation_parameters
    opposite = direction_relation(source) == "opposite"
    if source.later_boolean_mode is BooleanMode.JOIN:
        # Both operations include a positive-area part of the common profile
        # plane, and the revolve always contributes outside the extrusion.
        return _result(FeasibilityStatus.ACCEPTED, "mixed union is face-connected and adds volume")

    if not opposite and revolve_angle != 360.0:
        return _result(
            FeasibilityStatus.CUT_NO_POSITIVE_VOLUME_OVERLAP,
            "revolve sector lies on the opposite side of the extrusion",
        )
    angular_contains = revolve_angle == 360.0 or (
        opposite and revolve_angle >= _extrusion_half_angle(extrusion_distance)
    )
    if angular_contains and _extrusion_within_revolve_profile(
        source.primitive_family,
        extrusion_extent,
        revolve_extent,
        extrusion_distance,
    ):
        return _result(
            FeasibilityStatus.CUT_COMPLETE_SUBTRACTION,
            "revolve angular sector and radial profile contain the extrusion",
        )
    if (
        extrusion_extent > revolve_extent
        or (opposite and revolve_angle < _extrusion_half_angle(extrusion_distance))
        or _radial_tip(extrusion_extent, extrusion_distance) > 0.5 + revolve_extent
    ):
        return _result(
            FeasibilityStatus.ACCEPTED,
            "mixed cut has overlap and an explicit extrusion residual witness",
        )
    return _result(
        FeasibilityStatus.MIXED_CONTAINMENT_NOT_CERTIFIED,
        "overlap exists but complete-subtraction safety is not analytically certified",
    )


def _evaluate_re(source: Any) -> FeasibilityResult:
    revolve_extent, extrusion_extent = source.sketch_extents
    revolve_angle, extrusion_distance = source.operation_parameters
    opposite = direction_relation(source) == "opposite"
    if source.later_boolean_mode is BooleanMode.CUT:
        if not opposite and revolve_angle != 360.0:
            return _result(
                FeasibilityStatus.CUT_NO_POSITIVE_VOLUME_OVERLAP,
                "extrusion lies outside the current revolve sector",
            )
        # A one-sided extrusion cannot contain the revolve material on the
        # other side of the profile plane, so positive overlap leaves residual.
        return _result(FeasibilityStatus.ACCEPTED, "mixed cut overlaps and leaves revolve residual")

    if not opposite and revolve_angle < 360.0:
        return _result(
            FeasibilityStatus.ACCEPTED,
            "same-direction extrusion adds volume on the opposite side of the profile plane",
        )
    angular_contains = revolve_angle == 360.0 or (
        opposite and revolve_angle >= _extrusion_half_angle(extrusion_distance)
    )
    if angular_contains and _extrusion_within_revolve_profile(
        source.primitive_family,
        extrusion_extent,
        revolve_extent,
        extrusion_distance,
    ):
        return _result(
            FeasibilityStatus.JOIN_TOOL_CONTAINED,
            "extrusion is contained in the current revolve",
        )
    if (
        (opposite and revolve_angle < _extrusion_half_angle(extrusion_distance))
        or extrusion_extent > revolve_extent
        or _radial_tip(extrusion_extent, extrusion_distance) > 0.5 + revolve_extent
    ):
        return _result(
            FeasibilityStatus.ACCEPTED,
            "mixed union has an explicit extrusion volume-addition witness",
        )
    return _result(
        FeasibilityStatus.MIXED_CONTAINMENT_NOT_CERTIFIED,
        "union is connected but extrusion containment is not analytically certified",
    )


def _extrusion_half_angle(distance: float) -> float:
    return math.degrees(math.atan2(distance, 0.5))


def _radial_tip(extent: float, distance: float) -> float:
    return math.hypot(0.5 + extent, distance)


def _extrusion_within_revolve_profile(
    primitive_family: Any,
    extrusion_extent: float,
    revolve_extent: float,
    extrusion_distance: float,
) -> bool:
    if extrusion_extent > revolve_extent:
        return False
    family = _enum_value(primitive_family)
    if family == "rectangle_lines":
        outer_minimum = 0.5 + revolve_extent
    elif family == "circle":
        outer_minimum = (
            0.5
            + revolve_extent / 2.0
            + math.sqrt((revolve_extent / 2.0) ** 2 - (extrusion_extent / 2.0) ** 2)
        )
    elif family == "capsule_line_arc":
        outer_minimum = (
            0.5
            + 3.0 * revolve_extent / 4.0
            + math.sqrt((revolve_extent / 4.0) ** 2 - (extrusion_extent / 4.0) ** 2)
        )
    else:
        raise ValueError(f"unsupported primitive family {family!r}")
    return _radial_tip(extrusion_extent, extrusion_distance) <= outer_minimum


def _enum_value(value: Any) -> str:
    return value.value if isinstance(value, Enum) else str(value)


def _result(status: FeasibilityStatus, detail: str) -> FeasibilityResult:
    return FeasibilityResult(status, detail)
