"""Physical factors, indexable factor blocks, and bounded deterministic selection."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import itertools
import math

from prototype.representation.model import BooleanMode, Direction

from .config import ConfigurationError, GeneratorConfig, required_coverage_family_count


class OperationTemplate(str, Enum):
    E = "E"
    R = "R"
    EE = "EE"
    ER = "ER"
    RE = "RE"
    RR = "RR"

    @property
    def operations(self) -> tuple[str, ...]:
        return tuple("extrude" if item == "E" else "revolve" for item in self.value)


class PrimitiveFamily(str, Enum):
    RECTANGLE_LINES = "rectangle_lines"
    CIRCLE = "circle"
    CAPSULE_LINE_ARC = "capsule_line_arc"


class ReferencePlane(str, Enum):
    XY = "XY"
    XZ = "XZ"
    YZ = "YZ"


class ExtentBand(str, Enum):
    IN_RANGE = "in_range"
    EXTRAPOLATION = "extrapolation"


@dataclass(frozen=True)
class PhysicalSource:
    operation_template: OperationTemplate
    primitive_family: PrimitiveFamily
    reference_plane: ReferencePlane
    sketch_extents: tuple[float, ...]
    directions: tuple[Direction, ...]
    later_boolean_mode: BooleanMode | None
    operation_parameters: tuple[float, ...]
    configured_extent_band: ExtentBand

    @property
    def history_depth(self) -> int:
        return len(self.operation_template.operations)

    @property
    def extent_band(self) -> ExtentBand:
        return self.configured_extent_band

    def physical_metadata(self) -> dict[str, object]:
        modes = [BooleanMode.NEW_BODY.value]
        if self.later_boolean_mode is not None:
            modes.append(self.later_boolean_mode.value)
        distances = [
            value
            for kind, value in zip(self.operation_template.operations, self.operation_parameters)
            if kind == "extrude"
        ]
        angles = [
            value
            for kind, value in zip(self.operation_template.operations, self.operation_parameters)
            if kind == "revolve"
        ]
        return {
            "operation_template": self.operation_template.value,
            "history_depth": self.history_depth,
            "operation_sequence": list(self.operation_template.operations),
            "primitive_family": self.primitive_family.value,
            "boolean_modes": modes,
            "operation_directions": [item.value for item in self.directions],
            "profile_axis_dependency": "per_operation_same_sketch",
            "body_dependency": "linear_previous_operation",
            "sketch_extents": list(self.sketch_extents),
            "history_max_sketch_extent": max(self.sketch_extents),
            "sketch_extent_band": self.extent_band.value,
            "extrusion_distances": distances,
            "revolution_angles": angles,
            "reference_plane": self.reference_plane.value,
        }


@dataclass(frozen=True)
class FactorBlock:
    template: OperationTemplate
    extent_band: ExtentBand
    extents: tuple[tuple[float, ...], ...]
    parameter_domains: tuple[tuple[float, ...], ...]
    parameters: tuple[tuple[float, ...], ...]

    @classmethod
    def create(
        cls, template: OperationTemplate, band: ExtentBand, config: GeneratorConfig
    ) -> FactorBlock:
        depth = len(template.operations)
        all_extents = itertools.product(config.sketch_extents, repeat=depth)
        if band is ExtentBand.IN_RANGE:
            extents = tuple(items for items in all_extents if max(items) <= config.in_range_extent_max)
        else:
            extents = tuple(items for items in all_extents if max(items) >= config.extrapolation_extent_min)
        parameter_domains = tuple(
            config.extrusion_distances if operation == "extrude" else config.revolution_angles
            for operation in template.operations
        )
        parameters = tuple(itertools.product(*parameter_domains))
        return cls(template, band, extents, parameter_domains, parameters)

    @property
    def radices(self) -> tuple[int, ...]:
        depth = len(self.template.operations)
        boolean_count = 1 if depth == 1 else 2
        return (len(PrimitiveFamily), len(ReferencePlane), len(self.extents), 2**depth, boolean_count, len(self.parameters))

    @property
    def size(self) -> int:
        result = 1
        for radix in self.radices:
            result *= radix
        return result

    def decode(self, index: int) -> PhysicalSource:
        if self.size == 0:
            raise ConfigurationError(f"cannot decode empty factor block {self.name}")
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < self.size:
            raise ConfigurationError(f"factor index {index!r} is outside block {self.name}")
        digits = _mixed_radix_decode(index, self.radices)
        primitive = tuple(PrimitiveFamily)[digits[0]]
        plane = tuple(ReferencePlane)[digits[1]]
        extents = self.extents[digits[2]]
        depth = len(self.template.operations)
        directions = tuple(
            Direction.NEGATIVE if digits[3] & (1 << operation_index) else Direction.POSITIVE
            for operation_index in range(depth)
        )
        later_mode = None
        if depth == 2:
            later_mode = (BooleanMode.JOIN, BooleanMode.CUT)[digits[4]]
        return PhysicalSource(
            self.template,
            primitive,
            plane,
            tuple(float(item) for item in extents),
            directions,
            later_mode,
            tuple(float(item) for item in self.parameters[digits[5]]),
            self.extent_band,
        )

    @property
    def name(self) -> str:
        return f"{self.template.value}:{self.extent_band.value}"


def factor_blocks(config: GeneratorConfig) -> tuple[FactorBlock, ...]:
    config.validate()
    blocks = tuple(
        FactorBlock.create(template, band, config)
        for template in OperationTemplate
        for band in ExtentBand
    )
    empty = [block.name for block in blocks if block.size == 0]
    if empty:
        raise ConfigurationError(f"empty factor blocks: {empty!r}")
    return blocks


def select_sources(config: GeneratorConfig) -> tuple[PhysicalSource, ...]:
    """Select a bounded, deterministic subset without materializing the full space."""

    config.validate()
    blocks = factor_blocks(config)
    total = sum(block.size for block in blocks)
    if config.num_source_families > total:
        raise ConfigurationError(
            f"requested {config.num_source_families} source families but the factor space contains {total}"
        )
    minimum = required_coverage_family_count(config)
    if config.require_full_coverage and config.num_source_families < minimum:
        raise ConfigurationError(
            f"coverage anchors require {minimum} source families for this configuration"
        )
    selected: dict[tuple[str, int], PhysicalSource] = {}

    if config.require_full_coverage:
        for block in blocks:
            for index in _coverage_anchor_indices(block, config):
                selected[(block.name, index)] = block.decode(index)

    cursors = {block.name: 0 for block in blocks}
    while len(selected) < config.num_source_families:
        made_progress = False
        for block in blocks:
            if len(selected) >= config.num_source_families:
                break
            cursor = cursors[block.name]
            while cursor < block.size:
                index = _permuted_index(block, cursor, config)
                cursor += 1
                key = (block.name, index)
                if key not in selected:
                    selected[key] = block.decode(index)
                    made_progress = True
                    break
            cursors[block.name] = cursor
        if not made_progress:
            raise ConfigurationError("factor-space selection exhausted before reaching the requested count")

    if len(selected) != config.num_source_families:
        raise ConfigurationError("coverage anchors exceed num_source_families")
    return tuple(selected[key] for key in sorted(selected))


def total_candidate_count(config: GeneratorConfig) -> int:
    return sum(block.size for block in factor_blocks(config))


def _coverage_anchor_indices(
    block: FactorBlock, config: GeneratorConfig
) -> tuple[int, ...]:
    """Five diagonal anchors cover every size-five numerical grid per block."""

    if block.size == 0:
        return ()
    target_count = min(
        max(
            len(PrimitiveFamily),
            len(ReferencePlane),
            2,
            2 if len(block.template.operations) == 2 else 1,
            *(len(domain) for domain in block.parameter_domains),
        ),
        block.size,
    )
    digest = hashlib.sha256(
        b"controlled-data-anchor-v1\0"
        + config.canonical_bytes()
        + b"\0"
        + block.name.encode("ascii")
    ).digest()
    offsets = tuple(digest[index] for index in range(len(block.radices) + len(block.parameter_domains)))
    result: list[int] = []
    seen: set[int] = set()
    attempt = 0
    max_attempts = max(20, block.size)
    while len(result) < target_count and attempt < max_attempts:
        digits = list(
            (attempt + offsets[position]) % radix
            for position, radix in enumerate(block.radices)
        )
        desired_parameters = tuple(
            domain[(attempt + offsets[len(block.radices) + position]) % len(domain)]
            for position, domain in enumerate(block.parameter_domains)
        )
        digits[5] = block.parameters.index(desired_parameters)
        index = _mixed_radix_encode(tuple(digits), block.radices)
        if index not in seen:
            seen.add(index)
            result.append(index)
        attempt += 1
    if len(result) != target_count:
        raise ConfigurationError(f"could not construct unique coverage anchors for {block.name}")
    return tuple(result)


def _permuted_index(block: FactorBlock, position: int, config: GeneratorConfig) -> int:
    size = block.size
    if size == 0:
        raise ConfigurationError(f"cannot permute empty factor block {block.name}")
    if size == 1:
        if position != 0:
            raise ConfigurationError("position is outside the unit factor block")
        return 0
    if isinstance(position, bool) or not isinstance(position, int) or not 0 <= position < size:
        raise ConfigurationError(f"permutation position {position!r} is outside block {block.name}")
    digest = hashlib.sha256(
        b"controlled-data-affine-v1\0"
        + config.canonical_bytes()
        + b"\0"
        + str(config.seed).encode("ascii")
        + b"\0"
        + block.name.encode("ascii")
    ).digest()
    candidate = int.from_bytes(digest[:16], "big") % size
    multiplier = candidate or 1
    while math.gcd(multiplier, size) != 1:
        multiplier = (multiplier + 1) % size
        if multiplier == 0:
            multiplier = 1
    offset = int.from_bytes(digest[16:], "big") % size
    return (multiplier * position + offset) % size


def _mixed_radix_decode(index: int, radices: tuple[int, ...]) -> tuple[int, ...]:
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ConfigurationError("mixed-radix index must be a nonnegative integer")
    if any(radix <= 0 for radix in radices):
        raise ConfigurationError("mixed-radix bases must be positive")
    digits = [0] * len(radices)
    remainder = index
    for position in range(len(radices) - 1, -1, -1):
        digits[position] = remainder % radices[position]
        remainder //= radices[position]
    if remainder:
        raise ConfigurationError("mixed-radix index exceeds its factor space")
    return tuple(digits)


def _mixed_radix_encode(digits: tuple[int, ...], radices: tuple[int, ...]) -> int:
    if len(digits) != len(radices) or any(radix <= 0 for radix in radices):
        raise ConfigurationError("invalid mixed-radix shape")
    result = 0
    for digit, radix in zip(digits, radices):
        if isinstance(digit, bool) or not isinstance(digit, int) or not 0 <= digit < radix:
            raise ConfigurationError("mixed-radix digit is outside its base")
        result = result * radix + digit
    return result
