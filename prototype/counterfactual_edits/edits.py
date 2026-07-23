"""Apply and validate exactly one frozen semantic-factor edit."""

from __future__ import annotations

from dataclasses import fields, replace
from typing import Any

from prototype.controlled_data.config import GeneratorConfig
from prototype.controlled_data.factors import (
    ExtentBand,
    OperationTemplate,
    PhysicalSource,
    PrimitiveFamily,
    ReferencePlane,
)
from prototype.controlled_data.feasibility import evaluate_feasibility
from prototype.representation.model import BooleanMode, Direction

from .config import EditConfigurationError
from .model import EditTarget, EditType


def target_for(edit_type: EditType, source: PhysicalSource, position: int) -> EditTarget:
    operation = source.operation_template.operations[position]
    suffix = position + 1
    if edit_type is EditType.PROFILE_EXTENT:
        return EditTarget(position, f"sketch_{suffix}", "sketch_extent")
    if edit_type is EditType.EXTRUSION_DISTANCE:
        if operation != "extrude":
            raise EditConfigurationError("extrusion-distance target is not an extrude")
        return EditTarget(position, f"extrude_{suffix}", "distance")
    if edit_type is EditType.REVOLVE_ANGLE:
        if operation != "revolve":
            raise EditConfigurationError("revolve-angle target is not a revolve")
        return EditTarget(position, f"revolve_{suffix}", "angle_degrees")
    if edit_type is EditType.OPERATION_DIRECTION:
        return EditTarget(position, f"{operation}_{suffix}", "direction")
    if edit_type is EditType.BOOLEAN_MODE:
        if source.history_depth != 2 or position != 1:
            raise EditConfigurationError("Boolean-mode edits target operation index 1")
        return EditTarget(position, f"{operation}_{suffix}", "boolean_mode")
    raise EditConfigurationError(f"unsupported edit type {edit_type!r}")


def before_after(
    edit_type: EditType,
    base: PhysicalSource,
    edited: PhysicalSource,
    position: int,
) -> tuple[Any, Any]:
    if edit_type is EditType.PROFILE_EXTENT:
        return base.sketch_extents[position], edited.sketch_extents[position]
    if edit_type in (EditType.EXTRUSION_DISTANCE, EditType.REVOLVE_ANGLE):
        return base.operation_parameters[position], edited.operation_parameters[position]
    if edit_type is EditType.OPERATION_DIRECTION:
        return base.directions[position].value, edited.directions[position].value
    if edit_type is EditType.BOOLEAN_MODE:
        return base.later_boolean_mode.value, edited.later_boolean_mode.value
    raise EditConfigurationError(f"unsupported edit type {edit_type!r}")


def semantic_difference_count(left: PhysicalSource, right: PhysicalSource) -> int:
    _validate_source_shape(left)
    _validate_source_shape(right)
    differences = 0
    sequence_fields = {
        "sketch_extents",
        "directions",
        "operation_parameters",
    }
    scalar_fields = (
        "operation_template",
        "primitive_family",
        "reference_plane",
        "later_boolean_mode",
        "configured_extent_band",
    )
    known = sequence_fields | set(scalar_fields)
    actual = {item.name for item in fields(PhysicalSource)}
    if actual != known:
        raise EditConfigurationError(
            f"PhysicalSource fields changed without exhaustive edit validation: "
            f"{sorted(actual ^ known)!r}"
        )
    differences += sum(getattr(left, name) != getattr(right, name) for name in scalar_fields)
    for name in sorted(sequence_fields):
        left_values = getattr(left, name)
        right_values = getattr(right, name)
        if len(left_values) != len(right_values):
            raise EditConfigurationError(
                f"{name} tuple lengths differ between edit endpoints"
            )
        differences += sum(
            left_values[index] != right_values[index]
            for index in range(len(left_values))
        )
    return differences


def validate_edit_endpoints(
    edit_type: EditType,
    target: EditTarget,
    base: PhysicalSource,
    edited: PhysicalSource,
    config: GeneratorConfig,
) -> None:
    _validate_source_shape(base, config)
    _validate_source_shape(edited, config)
    if semantic_difference_count(base, edited) != 1:
        raise EditConfigurationError("edit endpoints do not differ by exactly one factor")
    expected = target_for(edit_type, base, target.operation_index)
    if expected != target:
        raise EditConfigurationError("edit target does not match its semantic factor")
    if not evaluate_feasibility(base).accepted or not evaluate_feasibility(edited).accepted:
        raise EditConfigurationError("both edit endpoints must be analytically feasible")
    before, after = before_after(edit_type, base, edited, target.operation_index)
    if edit_type is EditType.PROFILE_EXTENT:
        domain = config.sketch_extents
        if base.extent_band is not edited.extent_band:
            raise EditConfigurationError("profile-extent edit crosses an extent band")
    elif edit_type is EditType.EXTRUSION_DISTANCE:
        domain = config.extrusion_distances
    elif edit_type is EditType.REVOLVE_ANGLE:
        domain = config.revolution_angles
    else:
        domain = ()
    if domain:
        left = domain.index(before)
        right = domain.index(after)
        if abs(left - right) != 1:
            raise EditConfigurationError("numeric edit endpoints are not adjacent")
    if edit_type is EditType.OPERATION_DIRECTION:
        if {before, after} != {Direction.POSITIVE.value, Direction.NEGATIVE.value}:
            raise EditConfigurationError("operation-direction edit is not a strict flip")
    if edit_type is EditType.BOOLEAN_MODE:
        if {before, after} != {BooleanMode.JOIN.value, BooleanMode.CUT.value}:
            raise EditConfigurationError("Boolean-mode edit is not JOIN/CUT")


def _validate_source_shape(
    source: PhysicalSource, config: GeneratorConfig | None = None
) -> None:
    """Reject malformed factor records before any zip or builder truncation."""

    if not isinstance(source, PhysicalSource):
        raise EditConfigurationError("edit endpoint must be a PhysicalSource")
    if not isinstance(source.operation_template, OperationTemplate):
        raise EditConfigurationError("operation_template must be an OperationTemplate")
    if not isinstance(source.primitive_family, PrimitiveFamily):
        raise EditConfigurationError("primitive_family must be a PrimitiveFamily")
    if not isinstance(source.reference_plane, ReferencePlane):
        raise EditConfigurationError("reference_plane must be a ReferencePlane")
    if not isinstance(source.configured_extent_band, ExtentBand):
        raise EditConfigurationError("configured_extent_band must be an ExtentBand")

    depth = len(source.operation_template.operations)
    for name in ("sketch_extents", "directions", "operation_parameters"):
        values = getattr(source, name)
        if not isinstance(values, tuple):
            raise EditConfigurationError(f"{name} must be a tuple")
        if len(values) != depth:
            raise EditConfigurationError(
                f"{name} must contain exactly {depth} entries for "
                f"{source.operation_template.value}"
            )
    if any(not isinstance(item, Direction) for item in source.directions):
        raise EditConfigurationError("directions must contain Direction values")
    if depth == 1:
        if source.later_boolean_mode is not None:
            raise EditConfigurationError(
                "depth-one histories must not define a later Boolean mode"
            )
    elif source.later_boolean_mode not in (BooleanMode.JOIN, BooleanMode.CUT):
        raise EditConfigurationError(
            "depth-two histories require a JOIN or CUT later Boolean mode"
        )
    if config is not None:
        actual_band = extent_band(source.sketch_extents, config)
        if actual_band is not source.configured_extent_band:
            raise EditConfigurationError(
                "configured extent band disagrees with sketch extents"
            )


def extent_band(extents: tuple[float, ...], config: GeneratorConfig) -> ExtentBand:
    maximum = max(extents)
    if maximum <= config.in_range_extent_max:
        return ExtentBand.IN_RANGE
    if maximum >= config.extrapolation_extent_min:
        return ExtentBand.EXTRAPOLATION
    raise EditConfigurationError("edited extent lies in the excluded band gap")
