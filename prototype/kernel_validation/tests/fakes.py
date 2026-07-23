"""Deterministic fake kernel used by standard-library tests."""

from __future__ import annotations

from dataclasses import dataclass
import math

from prototype.controlled_data.builders import build_history
from prototype.controlled_data.factors import (
    ExtentBand,
    OperationTemplate,
    PhysicalSource,
    PrimitiveFamily,
    ReferencePlane,
)
from prototype.controlled_data.identity import sample_id, source_family_id
from prototype.representation.model import BooleanMode, Direction, GeometryEncoding
from prototype.representation.serialization import history_to_json

from prototype.kernel_validation.model import InspectedShape, SampleInput, ShapeMetrics


@dataclass(frozen=True)
class FakeShape:
    volume: float
    bbox: tuple[float, float, float, float, float, float] = (0, 0, 0, 1, 1, 1)
    valid: bool = True
    null: bool = False
    solids: int = 1
    faces: int = 6
    edges: int = 12
    vertices: int = 8


class FakeAdapter:
    def __init__(
        self,
        *,
        feature_volumes=(10.0, 4.0),
        no_effect_join=False,
        no_effect_cut=False,
        fail_method=None,
        fail_exception=None,
    ):
        self.feature_volumes = iter(feature_volumes)
        self.no_effect_join = no_effect_join
        self.no_effect_cut = no_effect_cut
        self.fail_method = fail_method
        self.fail_exception = fail_exception or RuntimeError("fake kernel failure")

    def backend_versions(self):
        return {"pythonocc_version": "fake-1", "opencascade_version": "fake-2"}

    def _fail(self, name):
        if self.fail_method == name:
            raise self.fail_exception

    def make_wire(self, primitives):
        self._fail("make_wire")
        return ("wire", primitives)

    def wire_is_closed(self, wire):
        self._fail("wire_is_closed")
        return True

    def shape_is_valid(self, shape):
        self._fail("shape_is_valid")
        return True

    def make_face(self, wire):
        self._fail("make_face")
        return ("face", wire)

    def make_axis(self, point, direction):
        self._fail("make_axis")
        return (point, direction)

    def extrude(self, face, vector):
        self._fail("extrude")
        return FakeShape(next(self.feature_volumes))

    def revolve(self, face, axis, angle_radians):
        self._fail("revolve")
        return FakeShape(next(self.feature_volumes))

    def join(self, current, feature):
        self._fail("join")
        volume = current.volume if self.no_effect_join else current.volume + feature.volume
        return FakeShape(volume)

    def cut(self, current, feature):
        self._fail("cut")
        volume = current.volume if self.no_effect_cut else current.volume - feature.volume / 2
        return FakeShape(volume)

    def inspect_solid(self, shape):
        self._fail("inspect_solid")
        metrics = ShapeMetrics(
            shape.null,
            shape.valid,
            "solid" if shape.solids == 1 else "compound",
            shape.volume,
            shape.bbox,
            shape.solids,
            shape.faces,
            shape.edges,
            shape.vertices,
        )
        return InspectedShape(shape, metrics)


def make_history(
    template="E",
    encoding="continuous",
    later_mode="join",
    primitive_family="rectangle_lines",
    reference_plane="XY",
):
    operation_template = OperationTemplate(template)
    depth = len(template)
    source = PhysicalSource(
        operation_template=operation_template,
        primitive_family=PrimitiveFamily(primitive_family),
        reference_plane=ReferencePlane(reference_plane),
        sketch_extents=(1.0,) * depth,
        directions=(Direction.POSITIVE,) * depth,
        later_boolean_mode=BooleanMode(later_mode) if depth == 2 else None,
        operation_parameters=tuple(1.0 if op == "extrude" else 90.0 for op in operation_template.operations),
        configured_extent_band=ExtentBand.IN_RANGE,
    )
    return build_history(source, GeometryEncoding(encoding))


def make_sample(
    template="E",
    encoding="continuous",
    later_mode="join",
    primitive_family="rectangle_lines",
    reference_plane="XY",
):
    history = make_history(template, encoding, later_mode, primitive_family, reference_plane)
    return SampleInput(
        source_family_id=source_family_id(history),
        sample_id=sample_id(history),
        geometry_encoding=encoding,
        operation_template=template,
        primitive_family=primitive_family,
        relative_json_path=f"samples/{sample_id(history)}.json",
        payload=history_to_json(history),
    )
