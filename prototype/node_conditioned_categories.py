"""Frozen categorical-selection contract for the controlled V4 decoder."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

from prototype.model_data.vocab import (
    BOOLEAN_MODES,
    DIRECTIONS,
    LOOP_ROLES,
    NODE_TYPES,
    OPERATION_TYPES,
    REFERENCE_PLANES,
)


V4_CATEGORICAL_CONTRACT_ID = "controlled-node-conditioned-categories-v4"
V4_CATEGORICAL_CONTRACT_VERSION = 4
ALL_VALID_NODE_TYPE_IDS = tuple(range(len(NODE_TYPES.tokens)))
NODE_TYPE_ID_TO_SEMANTIC_NAME = MappingProxyType({
    index: name for index, name in enumerate(NODE_TYPES.tokens)
})
SEMANTIC_NAME_TO_NODE_TYPE_ID = MappingProxyType({
    name: index for index, name in NODE_TYPE_ID_TO_SEMANTIC_NAME.items()
})


class NodeConditionedCategoricalError(ValueError):
    """A V4 categorical contract or selection is invalid."""

    def __init__(self, code, detail):
        self.code = code
        self.detail = detail
        super().__init__("{}: {}".format(code, detail))


@dataclass(frozen=True)
class V4CategoricalFieldContract:
    name: str
    output_position: int
    target_position: int
    class_order: tuple
    sentinel_id: int
    padding_id: int
    padding_behavior: str
    applicable_node_types: tuple
    valid_ids_by_node_type: tuple

    def valid_ids(self, node_type):
        for name, values in self.valid_ids_by_node_type:
            if name == node_type:
                return values
        return ()

    def to_metadata(self):
        return {
            "name": self.name,
            "output_position": self.output_position,
            "target_position": self.target_position,
            "class_order": list(self.class_order),
            "sentinel_id": self.sentinel_id,
            "padding_id": self.padding_id,
            "padding_behavior": self.padding_behavior,
            "applicable_node_types": list(self.applicable_node_types),
            "valid_ids_by_node_type": {
                name: list(values)
                for name, values in self.valid_ids_by_node_type
            },
        }


def _field(name, output_position, target_position, vocabulary, valid):
    applicable = tuple(name for name, unused_ids in valid)
    contract = V4CategoricalFieldContract(
        name,
        output_position,
        target_position,
        tuple(vocabulary.tokens),
        vocabulary.id(None),
        vocabulary.pad_id,
        "emit_non_applicable_sentinel",
        applicable,
        tuple(valid),
    )
    if contract.sentinel_id == contract.padding_id:
        raise AssertionError("sentinel and padding IDs must be distinct")
    for node_type in applicable:
        values = contract.valid_ids(node_type)
        if not values or any(
            value in (contract.sentinel_id, contract.padding_id)
            or not 0 <= value < len(contract.class_order)
            for value in values
        ):
            raise AssertionError("applicable fields need valid real classes")
    return contract


_OPERATIONS = ("extrude", "revolve")
V4_RETAINED_CATEGORICAL_FIELDS = (
    _field(
        "operation_type", 0, 0, OPERATION_TYPES,
        tuple((name, (OPERATION_TYPES.id(name),)) for name in _OPERATIONS),
    ),
    _field(
        "boolean_mode", 1, 1, BOOLEAN_MODES,
        tuple(
            (name, tuple(range(2, len(BOOLEAN_MODES.tokens))))
            for name in _OPERATIONS
        ),
    ),
    _field(
        "direction", 2, 2, DIRECTIONS,
        tuple(
            (name, tuple(range(2, len(DIRECTIONS.tokens))))
            for name in _OPERATIONS
        ),
    ),
    _field(
        "reference_plane", 3, 3, REFERENCE_PLANES,
        (("reference_plane", tuple(range(2, len(REFERENCE_PLANES.tokens)))),),
    ),
    _field(
        "loop_role", 4, 8, LOOP_ROLES,
        (("sketch", (LOOP_ROLES.id("outer"),)),),
    ),
)

V4_RETAINED_CATEGORICAL_FIELD_ORDER = tuple(
    field.name for field in V4_RETAINED_CATEGORICAL_FIELDS
)
V4_RETAINED_CATEGORICAL_TARGET_INDICES = tuple(
    field.target_position for field in V4_RETAINED_CATEGORICAL_FIELDS
)
V4_NODE_TYPE_APPLICABILITY = tuple(
    (
        node_name,
        tuple(
            field.name
            for field in V4_RETAINED_CATEGORICAL_FIELDS
            if field.valid_ids(node_name)
        ),
    )
    for node_name in NODE_TYPES.tokens
)

if V4_RETAINED_CATEGORICAL_FIELD_ORDER != (
    "operation_type", "boolean_mode", "direction", "reference_plane", "loop_role"
):
    raise AssertionError("retained categorical field order changed")
if V4_RETAINED_CATEGORICAL_TARGET_INDICES != (0, 1, 2, 3, 8):
    raise AssertionError("retained categorical target positions changed")


def v4_categorical_contract_metadata():
    return {
        "contract_id": V4_CATEGORICAL_CONTRACT_ID,
        "contract_version": V4_CATEGORICAL_CONTRACT_VERSION,
        "fields": [field.to_metadata() for field in V4_RETAINED_CATEGORICAL_FIELDS],
    }


def validate_node_conditioned_categorical_row(node_type_id, categorical_ids):
    """Validate one retained five-field row without repairing it."""

    if isinstance(node_type_id, bool) or not isinstance(node_type_id, int):
        raise NodeConditionedCategoricalError(
            "invalid_predicted_node_type", repr(node_type_id)
        )
    if not 0 <= node_type_id < len(NODE_TYPES.tokens):
        raise NodeConditionedCategoricalError(
            "invalid_predicted_node_type", repr(node_type_id)
        )
    node_type = NODE_TYPE_ID_TO_SEMANTIC_NAME[node_type_id]
    if node_type in ("<pad>", "<none>"):
        if tuple(categorical_ids) == tuple(
            field.sentinel_id for field in V4_RETAINED_CATEGORICAL_FIELDS
        ):
            return
        raise NodeConditionedCategoricalError(
            "invalid_node_conditioned_categorical_selection",
            "{} rows require categorical sentinels".format(node_type),
        )
    if len(categorical_ids) != len(V4_RETAINED_CATEGORICAL_FIELDS):
        raise NodeConditionedCategoricalError(
            "invalid_node_conditioned_categorical_selection",
            "retained categorical width must be five",
        )
    for field, value in zip(V4_RETAINED_CATEGORICAL_FIELDS, categorical_ids):
        valid = field.valid_ids(node_type)
        expected = valid if valid else (field.sentinel_id,)
        if isinstance(value, bool) or not isinstance(value, int) or value not in expected:
            raise NodeConditionedCategoricalError(
                "invalid_node_conditioned_categorical_selection",
                "{}={!r} is invalid for {}".format(field.name, value, node_type),
            )
