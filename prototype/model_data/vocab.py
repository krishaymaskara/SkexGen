"""Traversal-independent categorical vocabularies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from prototype.representation.model import (
    BooleanMode,
    Direction,
    EdgeType,
    LoopRole,
    NodeType,
    PrimitiveType,
)
from prototype.controlled_data.factors import ReferencePlane

from .errors import ModelDataError


NONE_TOKEN = "<none>"
PAD_TOKEN = "<pad>"


@dataclass(frozen=True)
class Vocabulary:
    name: str
    tokens: tuple[str, ...]

    def id(self, token: str | None) -> int:
        value = NONE_TOKEN if token is None else token
        try:
            return self.tokens.index(value)
        except ValueError as exc:
            raise ModelDataError(
                "unknown_vocabulary_token",
                f"{self.name} has no token {value!r}",
            ) from exc

    @property
    def pad_id(self) -> int:
        return self.id(PAD_TOKEN)


def _vocabulary(name: str, values: Iterable[str]) -> Vocabulary:
    return Vocabulary(name, (PAD_TOKEN, NONE_TOKEN, *tuple(sorted(set(values)))))


NODE_TYPES = _vocabulary("node_type", (item.value for item in NodeType))
EDGE_TYPES = _vocabulary("edge_type", (item.value for item in EdgeType))
OPERATION_TYPES = _vocabulary("operation_type", ("extrude", "revolve"))
PRIMITIVE_TYPES = _vocabulary(
    "primitive_type", (item.value for item in PrimitiveType)
)
BOOLEAN_MODES = _vocabulary("boolean_mode", (item.value for item in BooleanMode))
DIRECTIONS = _vocabulary("direction", (item.value for item in Direction))
REFERENCE_PLANES = _vocabulary(
    "reference_plane", (item.value for item in ReferencePlane)
)
LOOP_ROLES = _vocabulary("loop_role", (item.value for item in LoopRole))

ALL_VOCABULARIES = (
    NODE_TYPES,
    EDGE_TYPES,
    OPERATION_TYPES,
    PRIMITIVE_TYPES,
    BOOLEAN_MODES,
    DIRECTIONS,
    REFERENCE_PLANES,
    LOOP_ROLES,
)

MAX_PRIMITIVES = 4
CATEGORICAL_ATTRIBUTE_FIELDS = (
    "operation_type",
    "boolean_mode",
    "direction",
    "reference_plane",
    "primitive_0",
    "primitive_1",
    "primitive_2",
    "primitive_3",
    "loop_role",
)
FLAT_CATEGORICAL_FIELDS = ("node_type", *CATEGORICAL_ATTRIBUTE_FIELDS)
