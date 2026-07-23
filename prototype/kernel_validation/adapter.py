"""PythonOCC-free kernel adapter protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Tuple, Union

from .model import InspectedShape


Point3 = Tuple[float, float, float]
Vector3 = Tuple[float, float, float]


@dataclass(frozen=True)
class WorldLine:
    start: Point3
    end: Point3


@dataclass(frozen=True)
class WorldArc:
    start: Point3
    midpoint: Point3
    end: Point3


@dataclass(frozen=True)
class WorldCircle:
    center: Point3
    normal: Vector3
    x_direction: Vector3
    radius: float


WorldPrimitive = Union[WorldLine, WorldArc, WorldCircle]


class KernelAdapterError(RuntimeError):
    """An expected, deterministic kernel-builder failure."""


class KernelAdapter(Protocol):
    def backend_versions(self) -> dict[str, str | None]: ...
    def make_wire(self, primitives: tuple[WorldPrimitive, ...]) -> object: ...
    def wire_is_closed(self, wire: object) -> bool: ...
    def shape_is_valid(self, shape: object) -> bool: ...
    def make_face(self, wire: object) -> object: ...
    def make_axis(self, point: Point3, direction: Vector3) -> object: ...
    def extrude(self, face: object, vector: Vector3) -> object: ...
    def revolve(self, face: object, axis: object, angle_radians: float) -> object: ...
    def join(self, current: object, feature: object) -> object: ...
    def cut(self, current: object, feature: object) -> object: ...
    def inspect_solid(self, shape: object) -> InspectedShape: ...
