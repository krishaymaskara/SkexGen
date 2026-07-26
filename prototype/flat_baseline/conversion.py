"""Strict Phase B conversion of raw flat-baseline predictions."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from types import MappingProxyType

from prototype.model_data.geometry import (
    GEOMETRY_WIDTH,
    denormalize_applicable_geometry,
)
from prototype.model_data.errors import ModelDataError
from prototype.model_data.records import ReconstructionTarget
from prototype.representation.validation import LOOP_CONTINUITY_TOLERANCE
from prototype.model_data.vocab import (
    BOOLEAN_MODES,
    DIRECTIONS,
    EDGE_TYPES,
    LOOP_ROLES,
    NODE_TYPES,
    OPERATION_TYPES,
    PRIMITIVE_TYPES,
    REFERENCE_PLANES,
)


STAGE_ORDER = (
    "raw_decode",
    "raw_envelope",
    "raw_numeric_integrity",
    "flat_target_structure",
    "controlled_node_grammar",
    "controlled_geometry",
    "controlled_operations",
    "controlled_edges",
    "cross_record_consistency",
)

FAILURE_CODE_ORDER = MappingProxyType({
    "raw_decode": ("missing_raw_prediction",),
    "raw_envelope": (
        "missing_raw_field",
        "invalid_raw_field_type",
        "invalid_node_count",
        "node_count_mismatch",
        "invalid_node_position",
        "invalid_categorical_width",
        "invalid_geometry_width",
        "invalid_geometry_mask_width",
        "invalid_edge_matrix",
        "invalid_edge_matrix_order",
        "invalid_pointer_records",
        "invalid_pointer_query_order",
        "inconsistent_operation_metadata",
        "invalid_raw_metadata",
    ),
    "raw_numeric_integrity": (
        "nonfinite_geometry",
        "nonfinite_edge_logit",
        "nonfinite_pointer_logit",
    ),
    "flat_target_structure": (
        "invalid_node_type_id",
        "node_type_sentinel",
        "invalid_categorical_id",
        "categorical_pad_sentinel",
        "invalid_geometry_mask_type",
        "invalid_edge_type_id",
        "edge_type_sentinel",
        "invalid_pointer_type",
        "invalid_pointer_index",
    ),
    "controlled_node_grammar": (
        "unsupported_operation_count",
        "invalid_node_grammar",
        "unsupported_profile_pattern",
    ),
    "controlled_geometry": (
        "node_category_applicability",
        "geometry_mask_applicability",
        "geometry_out_of_range",
        "invalid_reference_plane_geometry",
        "invalid_axis_geometry",
        "invalid_operation_parameter",
        "invalid_primitive_geometry",
        "invalid_loop_constraint",
    ),
    "controlled_operations": (
        "operation_limit_exceeded",
        "operation_type_mismatch",
        "invalid_boolean_sequence",
    ),
    "controlled_edges": ("missing_required_edge", "unexpected_edge"),
    "cross_record_consistency": (
        "operation_metadata_mismatch",
        "operation_pointer_inconsistency",
        "operation_edge_inconsistency",
    ),
})

LAYER_ORDER = MappingProxyType(
    {"raw_integrity": 0, "reconstruction_target": 1, "controlled_domain": 2}
)
_STAGE_INDEX = {name: index for index, name in enumerate(STAGE_ORDER)}
_CODE_INDEX = {
    stage: {code: index for index, code in enumerate(codes)}
    for stage, codes in FAILURE_CODE_ORDER.items()
}
_LOCATION_PART = re.compile(r"(\d+)")
_NONE = NODE_TYPES.id(None)
_PAD = NODE_TYPES.pad_id
_GEOMETRY_TOLERANCE = 1e-6
# Kept local to avoid importing the PyTorch-dependent Phase A decoder module.
_RAW_PREFIX_FEEDBACK = "raw_argmax_with_derived_geometry_mask"
_REQUESTED_LENGTH_TERMINATION = "requested_node_count_reached"
_ATTRIBUTE_VOCABS = (
    OPERATION_TYPES,
    BOOLEAN_MODES,
    DIRECTIONS,
    REFERENCE_PLANES,
    PRIMITIVE_TYPES,
    PRIMITIVE_TYPES,
    PRIMITIVE_TYPES,
    PRIMITIVE_TYPES,
    LOOP_ROLES,
)


@dataclass(frozen=True)
class LayerValidity:
    valid: bool
    failure_codes: tuple[str, ...]


@dataclass(frozen=True)
class ConversionFailure:
    code: str
    layer: str
    stage: str
    location: str
    detail: str


@dataclass(frozen=True)
class ConversionResult:
    raw_integrity: LayerValidity
    reconstruction_target: LayerValidity
    controlled_domain: LayerValidity
    reconstruction_target_candidate: ReconstructionTarget | None
    primary_failure: ConversionFailure | None
    secondary_failures: tuple[ConversionFailure, ...]


@dataclass(frozen=True)
class _RawContext:
    raw: object
    addressable_node_count: int | None
    nodes_usable: bool
    edges_usable: bool
    pointers_usable: bool
    metadata_usable: bool
    target_usable: bool


class _Failures:
    def __init__(self):
        self.items = []

    def add(self, code, layer, stage, location, detail):
        if code not in _CODE_INDEX.get(stage, {}):
            raise AssertionError("unregistered conversion failure code")
        failure = ConversionFailure(code, layer, stage, location, detail)
        if failure not in self.items:
            self.items.append(failure)

    def ordered(self):
        return tuple(sorted(self.items, key=_failure_key))

    def has_layer(self, layer):
        return any(item.layer == layer for item in self.items)

    def has_stage(self, stage):
        return any(item.stage == stage for item in self.items)


def validate_and_convert_raw_prediction(raw_prediction, *, max_operations):
    """Validate one Phase A result and construct an authoritative flat target."""

    failures = _Failures()
    if isinstance(max_operations, bool) or not isinstance(max_operations, int):
        raise TypeError("max_operations must be a non-Boolean integer")
    if max_operations < 1:
        raise ValueError("max_operations must be positive")
    context = _validate_raw_envelope(raw_prediction, max_operations, failures)
    if context is not None:
        _validate_raw_numeric(context, failures)
        _validate_flat_structure(context, failures)
    candidate = None
    if (
        context is not None and context.target_usable
        and not _has_prerequisite_failures(failures, 3)
    ):
        candidate = _build_target(context.raw)
        _validate_controlled(context.raw, max_operations, failures)
    ordered = failures.ordered()
    raw_codes = _codes_through_layer(ordered, "raw_integrity")
    target_codes = _codes_through_layer(ordered, "reconstruction_target")
    domain_codes = _codes_through_layer(ordered, "controlled_domain")
    return ConversionResult(
        LayerValidity(not raw_codes, raw_codes),
        LayerValidity(not target_codes, target_codes),
        LayerValidity(not domain_codes, domain_codes),
        candidate,
        ordered[0] if ordered else None,
        ordered[1:] if ordered else (),
    )


def _validate_raw_envelope(raw, max_operations, failures):
    if raw is None:
        failures.add(
            "missing_raw_prediction", "raw_integrity", "raw_decode", "$",
            "raw prediction is absent",
        )
        return None
    required = (
        "latent_indices", "node_count", "node_count_source",
        "termination_reason", "termination_is_learned", "prefix_feedback",
        "raw_nodes", "raw_edges", "predicted_operation_node_indices",
        "predicted_operation_count", "operation_count_exceeds_limit",
        "raw_operation_pointers",
    )
    missing = tuple(name for name in required if not hasattr(raw, name))
    for name in missing:
        failures.add(
            "missing_raw_field", "raw_integrity", "raw_envelope", name,
            "required raw field is absent",
        )
    collections = ("latent_indices", "raw_nodes", "raw_edges",
                   "predicted_operation_node_indices", "raw_operation_pointers")
    for name in collections:
        if hasattr(raw, name) and not isinstance(getattr(raw, name), tuple):
            failures.add(
                "invalid_raw_field_type", "raw_integrity", "raw_envelope", name,
                "raw collection must be a tuple",
            )
    nodes_collection = hasattr(raw, "raw_nodes") and isinstance(raw.raw_nodes, tuple)
    edges_collection = hasattr(raw, "raw_edges") and isinstance(raw.raw_edges, tuple)
    pointer_collection = (
        hasattr(raw, "raw_operation_pointers")
        and isinstance(raw.raw_operation_pointers, tuple)
    )
    count = getattr(raw, "node_count", None)
    count_valid = hasattr(raw, "node_count")
    if count_valid and (
        isinstance(count, bool) or not isinstance(count, int) or count <= 0
    ):
        failures.add(
            "invalid_node_count", "raw_integrity", "raw_envelope", "node_count",
            "node count must be a positive non-Boolean integer",
        )
        count_valid = False
    actual_count = len(raw.raw_nodes) if nodes_collection else None
    if count_valid and nodes_collection and actual_count != count:
        failures.add(
            "node_count_mismatch", "raw_integrity", "raw_envelope", "raw_nodes",
            "declared node count does not match raw node records",
        )
    node_start = len(failures.items)
    if nodes_collection:
        _validate_raw_nodes(raw.raw_nodes, failures)
    nodes_usable = nodes_collection and not _new_failures(failures, node_start)
    edge_start = len(failures.items)
    if edges_collection and actual_count is not None:
        _validate_raw_edges(raw.raw_edges, actual_count, failures)
    edges_usable = (
        edges_collection and actual_count is not None
        and not _new_failures(failures, edge_start)
    )
    pointer_start = len(failures.items)
    if pointer_collection:
        _validate_raw_pointers(raw.raw_operation_pointers, failures)
    pointers_usable = pointer_collection and not _new_failures(failures, pointer_start)
    latent_usable = (
        hasattr(raw, "latent_indices") and isinstance(raw.latent_indices, tuple)
    )
    operation_indices_usable = (
        hasattr(raw, "predicted_operation_node_indices")
        and isinstance(raw.predicted_operation_node_indices, tuple)
    )
    metadata_start = len(failures.items)
    _validate_raw_metadata(
        raw,
        actual_count,
        max_operations,
        failures,
        latent_usable=latent_usable,
        operation_indices_usable=operation_indices_usable,
        pointers_container_usable=pointer_collection,
        pointer_records_usable=pointers_usable,
        count_valid=count_valid,
    )
    metadata_usable = (
        not missing and count_valid and actual_count is not None
        and not _new_failures(failures, metadata_start)
    )
    target_usable = (
        count_valid and actual_count == count and nodes_usable and edges_usable
        and pointers_usable and metadata_usable
    )
    return _RawContext(
        raw, actual_count, nodes_usable, edges_usable, pointers_usable,
        metadata_usable, target_usable,
    )


def _validate_raw_nodes(nodes, failures):
    for index, node in enumerate(nodes):
        location = "raw_nodes[{}]".format(index)
        fields = (
            "position", "node_type_id", "categorical_ids",
            "normalized_geometry", "derived_geometry_mask",
        )
        if any(not hasattr(node, name) for name in fields):
            failures.add(
                "invalid_raw_field_type", "raw_integrity", "raw_envelope",
                location, "raw node record is missing required fields",
            )
            continue
        if (
            isinstance(node.position, bool)
            or not isinstance(node.position, int)
            or node.position != index
        ):
            failures.add(
                "invalid_node_position", "raw_integrity", "raw_envelope",
                location + ".position", "node positions must be canonical and contiguous",
            )
        for name, width, code in (
            ("categorical_ids", 9, "invalid_categorical_width"),
            ("normalized_geometry", GEOMETRY_WIDTH, "invalid_geometry_width"),
            ("derived_geometry_mask", GEOMETRY_WIDTH, "invalid_geometry_mask_width"),
        ):
            value = getattr(node, name)
            if not isinstance(value, tuple) or len(value) != width:
                failures.add(
                    code, "raw_integrity", "raw_envelope",
                    location + "." + name,
                    "{} must contain exactly {} entries".format(name, width),
                )
        if (
            isinstance(node.normalized_geometry, tuple)
            and len(node.normalized_geometry) == GEOMETRY_WIDTH
        ):
            for channel, value in enumerate(node.normalized_geometry):
                if type(value) is not float:
                    failures.add(
                        "invalid_raw_field_type", "raw_integrity",
                        "raw_envelope",
                        location + ".normalized_geometry[{}]".format(channel),
                        "generated geometry values must be Python floats",
                    )


def _validate_raw_edges(edges, count, failures):
    if len(edges) != count * count:
        failures.add(
            "invalid_edge_matrix", "raw_integrity", "raw_envelope", "raw_edges",
            "edge matrix must contain one entry for every ordered node pair",
        )
        return
    for offset, edge in enumerate(edges):
        source, target = divmod(offset, count)
        location = "raw_edges[{}]".format(offset)
        fields = (
            "source_index", "target_index", "presence_logit", "present",
            "edge_type_id",
        )
        if any(not hasattr(edge, name) for name in fields):
            failures.add(
                "invalid_edge_matrix", "raw_integrity", "raw_envelope", location,
                "edge entry is missing required fields",
            )
            continue
        if (
            isinstance(edge.source_index, bool)
            or not isinstance(edge.source_index, int)
            or isinstance(edge.target_index, bool)
            or not isinstance(edge.target_index, int)
            or edge.source_index != source
            or edge.target_index != target
        ):
            failures.add(
                "invalid_edge_matrix_order", "raw_integrity", "raw_envelope",
                location, "edge entries must use complete source-major ordering",
            )
        if not isinstance(edge.present, bool):
            failures.add(
                "invalid_edge_matrix", "raw_integrity", "raw_envelope",
                location + ".present", "edge presence must be Boolean",
            )
        elif _finite_number(edge.presence_logit):
            if edge.present != (float(edge.presence_logit) >= 0.0):
                failures.add(
                    "invalid_edge_matrix", "raw_integrity", "raw_envelope",
                    location + ".present",
                    "edge presence must preserve the Phase A logit threshold",
                )
        if type(edge.presence_logit) is not float:
            failures.add(
                "invalid_raw_field_type", "raw_integrity", "raw_envelope",
                location + ".presence_logit",
                "retained edge presence logits must be Python floats",
            )


def _validate_raw_pointers(pointers, failures):
    for index, pointer in enumerate(pointers):
        location = "raw_operation_pointers[{}]".format(index)
        fields = ("query_index", "selected_node_index", "selected_logit")
        if any(not hasattr(pointer, name) for name in fields):
            failures.add(
                "invalid_pointer_records", "raw_integrity", "raw_envelope",
                location, "pointer record is missing required fields",
            )
        elif (
            isinstance(pointer.query_index, bool)
            or not isinstance(pointer.query_index, int)
            or pointer.query_index != index
        ):
            failures.add(
                "invalid_pointer_query_order", "raw_integrity", "raw_envelope",
                location + ".query_index",
                "pointer queries must be canonical and contiguous",
            )
        if hasattr(pointer, "selected_logit") and type(pointer.selected_logit) is not float:
            failures.add(
                "invalid_raw_field_type", "raw_integrity", "raw_envelope",
                location + ".selected_logit",
                "retained pointer logits must be Python floats",
            )


def _validate_raw_metadata(
    raw,
    count,
    max_operations,
    failures,
    *,
    latent_usable,
    operation_indices_usable,
    pointers_container_usable,
    pointer_records_usable,
    count_valid,
):
    if (
        not hasattr(raw, "node_count_source")
        or not isinstance(raw.node_count_source, str)
        or not raw.node_count_source
        or not hasattr(raw, "termination_reason")
        or raw.termination_reason != _REQUESTED_LENGTH_TERMINATION
        or not hasattr(raw, "prefix_feedback")
        or raw.prefix_feedback != _RAW_PREFIX_FEEDBACK
        or not hasattr(raw, "termination_is_learned")
        or raw.termination_is_learned is not False
    ):
        failures.add(
            "invalid_raw_metadata", "raw_integrity", "raw_envelope", "metadata",
            "raw decoding policies must match the Phase A contract",
        )
    if latent_usable and any(
        isinstance(item, bool) or not isinstance(item, int) or item < 0
        for item in raw.latent_indices
    ):
        failures.add(
            "invalid_raw_metadata", "raw_integrity", "raw_envelope",
            "latent_indices",
            "latent indices must be nonnegative non-Boolean integers",
        )
    if not operation_indices_usable:
        return
    indices = raw.predicted_operation_node_indices
    indices_typed = not any(
        isinstance(item, bool) or not isinstance(item, int) for item in indices
    )
    if not indices_typed:
        failures.add(
            "inconsistent_operation_metadata", "raw_integrity", "raw_envelope",
            "predicted_operation_node_indices",
            "operation-node metadata must contain non-Boolean integers",
        )
    if (
        indices_typed and count_valid and count is not None
        and any(item < 0 or item >= count for item in indices)
    ):
        failures.add(
            "inconsistent_operation_metadata", "raw_integrity", "raw_envelope",
            "predicted_operation_node_indices",
            "operation-node metadata contains an out-of-range position",
        )
    if (
        indices_typed and count_valid and count is not None
        and indices != tuple(sorted(set(indices)))
    ):
        failures.add(
            "inconsistent_operation_metadata", "raw_integrity", "raw_envelope",
            "predicted_operation_node_indices",
            "operation-node metadata must be unique and canonically ordered",
        )
    if (
        indices_typed
        and hasattr(raw, "predicted_operation_count")
        and (
            isinstance(raw.predicted_operation_count, bool)
            or not isinstance(raw.predicted_operation_count, int)
            or raw.predicted_operation_count != len(indices)
        )
    ):
        failures.add(
            "inconsistent_operation_metadata", "raw_integrity", "raw_envelope",
            "predicted_operation_count",
            "declared operation metadata is internally inconsistent",
        )
    if (
        indices_typed
        and pointers_container_usable
        and pointer_records_usable
        and len(raw.raw_operation_pointers) != min(len(indices), max_operations)
    ):
        failures.add(
            "inconsistent_operation_metadata", "raw_integrity", "raw_envelope",
            "raw_operation_pointers",
            "pointer record count does not match the Phase A policy",
        )
    if (
        indices_typed
        and hasattr(raw, "operation_count_exceeds_limit")
        and (
            not isinstance(raw.operation_count_exceeds_limit, bool)
            or raw.operation_count_exceeds_limit != (len(indices) > max_operations)
        )
    ):
        failures.add(
            "inconsistent_operation_metadata", "raw_integrity", "raw_envelope",
            "operation_count_exceeds_limit",
            "operation-limit metadata does not match the configured limit",
        )


def _validate_raw_numeric(context, failures):
    raw = context.raw
    for node_index, node in enumerate(raw.raw_nodes if hasattr(raw, "raw_nodes") and isinstance(raw.raw_nodes, tuple) else ()):
        values = getattr(node, "normalized_geometry", ())
        for channel, value in enumerate(values if isinstance(values, tuple) else ()):
            if type(value) is float and not math.isfinite(value):
                failures.add(
                    "nonfinite_geometry", "raw_integrity", "raw_numeric_integrity",
                    "raw_nodes[{}].normalized_geometry[{}]".format(node_index, channel),
                    "generated geometry must be finite",
                )
    for index, edge in enumerate(raw.raw_edges if hasattr(raw, "raw_edges") and isinstance(raw.raw_edges, tuple) else ()):
        if not hasattr(edge, "presence_logit"):
            continue
        if (
            type(edge.presence_logit) is float
            and not math.isfinite(edge.presence_logit)
        ):
            failures.add(
                "nonfinite_edge_logit", "raw_integrity", "raw_numeric_integrity",
                "raw_edges[{}].presence_logit".format(index),
                "retained edge presence logit must be finite",
            )
    for index, pointer in enumerate(raw.raw_operation_pointers if hasattr(raw, "raw_operation_pointers") and isinstance(raw.raw_operation_pointers, tuple) else ()):
        if not hasattr(pointer, "selected_logit"):
            continue
        if (
            type(pointer.selected_logit) is float
            and not math.isfinite(pointer.selected_logit)
        ):
            failures.add(
                "nonfinite_pointer_logit", "raw_integrity", "raw_numeric_integrity",
                "raw_operation_pointers[{}].selected_logit".format(index),
                "retained pointer logit must be finite",
            )


def _validate_flat_structure(context, failures):
    raw = context.raw
    if context.nodes_usable:
        for node_index, node in enumerate(raw.raw_nodes):
            _validate_id(
                node.node_type_id, NODE_TYPES, False, "invalid_node_type_id",
                "node_type_sentinel",
                "raw_nodes[{}].node_type_id".format(node_index),
                failures,
            )
            for field, (value, vocab) in enumerate(
                zip(node.categorical_ids, _ATTRIBUTE_VOCABS)
            ):
                _validate_id(
                    value, vocab, True, "invalid_categorical_id",
                    "categorical_pad_sentinel",
                    "raw_nodes[{}].categorical_ids[{}]".format(node_index, field),
                    failures,
                )
            for channel, value in enumerate(node.derived_geometry_mask):
                if not isinstance(value, bool):
                    failures.add(
                        "invalid_geometry_mask_type", "reconstruction_target",
                        "flat_target_structure",
                        "raw_nodes[{}].derived_geometry_mask[{}]".format(
                            node_index, channel
                        ),
                        "geometry mask entries must be actual Booleans",
                    )
    if context.edges_usable:
        for index, edge in enumerate(raw.raw_edges):
            if not edge.present:
                continue
            location = "raw_edges[{}]".format(index)
            _validate_id(
                edge.edge_type_id, EDGE_TYPES, False, "invalid_edge_type_id",
                "edge_type_sentinel", location + ".edge_type_id", failures,
            )
    if context.pointers_usable:
        for index, pointer in enumerate(raw.raw_operation_pointers):
            value = pointer.selected_node_index
            location = "raw_operation_pointers[{}].selected_node_index".format(index)
            if isinstance(value, bool) or not isinstance(value, int):
                failures.add(
                    "invalid_pointer_type", "reconstruction_target",
                    "flat_target_structure", location,
                    "operation pointer must be a non-Boolean integer",
                )
            elif (
                context.addressable_node_count is not None
                and not 0 <= value < context.addressable_node_count
            ):
                failures.add(
                    "invalid_pointer_index", "reconstruction_target",
                    "flat_target_structure", location,
                    "operation pointer must reference an existing node",
                )


def _validate_id(value, vocab, allow_none, invalid_code, sentinel_code, location, failures):
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < len(vocab.tokens):
        failures.add(
            invalid_code, "reconstruction_target", "flat_target_structure",
            location, "{} must be a legal non-Boolean vocabulary ID".format(vocab.name),
        )
    elif value == _PAD or (not allow_none and value == _NONE):
        failures.add(
            sentinel_code, "reconstruction_target", "flat_target_structure",
            location, "sentinel is not legal in an unpadded target field",
        )


def _build_target(raw):
    present = tuple(sorted(
        (edge for edge in raw.raw_edges if edge.present),
        key=lambda edge: (edge.source_index, edge.edge_type_id, edge.target_index),
    ))
    attributes = tuple(node.categorical_ids for node in raw.raw_nodes)
    return ReconstructionTarget(
        node_type_ids=tuple(node.node_type_id for node in raw.raw_nodes),
        categorical_attributes=attributes,
        edge_index=(
            tuple(edge.source_index for edge in present),
            tuple(edge.target_index for edge in present),
        ),
        edge_type_ids=tuple(edge.edge_type_id for edge in present),
        boolean_mode_targets=tuple(row[1] for row in attributes),
        operation_sequence=tuple(
            pointer.selected_node_index for pointer in raw.raw_operation_pointers
        ),
        geometry=tuple(node.normalized_geometry for node in raw.raw_nodes),
        geometry_mask=tuple(node.derived_geometry_mask for node in raw.raw_nodes),
    )


def _validate_controlled(raw, max_operations, failures):
    node_tokens = tuple(NODE_TYPES.tokens[node.node_type_id] for node in raw.raw_nodes)
    grammar = _parse_grammar(node_tokens, failures)
    categories_usable = grammar is not None
    if categories_usable:
        geometry_categories_usable = _validate_categories(
            raw, node_tokens, failures
        )
        _validate_geometry(
            raw, node_tokens, geometry_categories_usable, failures
        )
        _validate_operations(raw, grammar, max_operations, failures)
    edges_usable = not any(
        item.code in {"invalid_edge_type_id", "edge_type_sentinel"}
        for item in failures.items
    )
    if grammar is not None and edges_usable:
        _validate_edges(raw, grammar, failures)
    pointers_usable = not any(
        item.code in {"invalid_pointer_type", "invalid_pointer_index"}
        for item in failures.items
    )
    if grammar is not None:
        _validate_cross_records(
            raw,
            grammar,
            failures,
            pointers_usable=pointers_usable,
            edges_usable=edges_usable,
        )
    return not failures.has_layer("controlled_domain")


def _parse_grammar(tokens, failures):
    operations = []
    cursor = 0
    if not tokens or tokens[0] != "reference_plane":
        failures.add(
            "invalid_node_grammar", "controlled_domain",
            "controlled_node_grammar", "raw_nodes[0]",
            "controlled node grammar must begin with one reference plane",
        )
        return None
    cursor = 1
    while cursor < len(tokens):
        start = cursor
        if tokens[cursor:cursor + 2] != ("sketch", "profile"):
            failures.add(
                "invalid_node_grammar", "controlled_domain",
                "controlled_node_grammar", "raw_nodes[{}]".format(cursor),
                "each feature must begin with sketch and profile nodes",
            )
            return None
        cursor += 2
        axis = None
        if cursor < len(tokens) and tokens[cursor] == "axis":
            axis = cursor
            cursor += 1
        if cursor >= len(tokens) or tokens[cursor] not in ("extrude", "revolve"):
            failures.add(
                "invalid_node_grammar", "controlled_domain",
                "controlled_node_grammar", "raw_nodes[{}]".format(cursor),
                "each feature must end with an extrude or revolve node",
            )
            return None
        kind = tokens[cursor]
        if (kind == "revolve") != (axis is not None):
            failures.add(
                "invalid_node_grammar", "controlled_domain",
                "controlled_node_grammar", "raw_nodes[{}]".format(cursor),
                "only revolve features require an axis node",
            )
            return None
        operations.append((start, start + 1, axis, cursor, kind))
        cursor += 1
    if len(operations) not in (1, 2):
        failures.add(
            "unsupported_operation_count", "controlled_domain",
            "controlled_node_grammar", "raw_nodes",
            "controlled grammar supports exactly one or two operations",
        )
        return None
    return tuple(operations)


def _validate_categories(raw, tokens, failures):
    geometry_usable = []
    for index, (node, token) in enumerate(zip(raw.raw_nodes, tokens)):
        row = node.categorical_ids
        applicable = [False] * 9
        if token == "reference_plane":
            applicable[3] = True
            node_geometry_usable = row[3] >= 2
        elif token == "sketch":
            primitives = tuple(PRIMITIVE_TYPES.tokens[item] for item in row[4:8])
            allowed = (
                ("circle", "<none>", "<none>", "<none>"),
                ("line", "line", "line", "line"),
                ("arc", "arc", "line", "line"),
            )
            if primitives not in allowed:
                failures.add(
                    "unsupported_profile_pattern", "controlled_domain",
                    "controlled_node_grammar",
                    "raw_nodes[{}].categorical_ids[4]".format(index),
                    "sketch primitives must encode circle, rectangle, or capsule",
                )
            node_geometry_usable = primitives in allowed
            applicable[4:8] = [
                primitive != "<none>" for primitive in primitives
            ]
            applicable[8] = True
        elif token in ("extrude", "revolve"):
            applicable[0:3] = [True] * 3
            node_geometry_usable = True
        else:
            node_geometry_usable = True
        for field, should_apply in enumerate(applicable):
            value = row[field]
            invalid = value < 2 if should_apply else value != _NONE
            if invalid:
                failures.add(
                    "node_category_applicability", "controlled_domain",
                    "controlled_geometry",
                    "raw_nodes[{}].categorical_ids[{}]".format(index, field),
                    (
                        "applicable categorical field must contain a real value"
                        if should_apply else
                        "non-applicable categorical field must contain <none>"
                    ),
                )
        if token == "sketch" and row[8] != LOOP_ROLES.id("outer"):
            failures.add(
                "invalid_loop_constraint", "controlled_domain",
                "controlled_geometry",
                "raw_nodes[{}].categorical_ids[8]".format(index),
                "controlled sketches require one outer loop",
            )
        geometry_usable.append(node_geometry_usable)
    return tuple(geometry_usable)


def _expected_mask(node_type, row):
    mask = [False] * GEOMETRY_WIDTH
    if node_type == "reference_plane":
        mask[0:9] = [True] * 9
    elif node_type == "sketch":
        widths = {"line": 4, "arc": 6, "circle": 3}
        for slot, primitive_id in enumerate(row[4:8]):
            primitive = PRIMITIVE_TYPES.tokens[primitive_id]
            width = widths.get(primitive, 0)
            start = 9 + slot * 6
            mask[start:start + width] = [True] * width
    elif node_type == "axis":
        mask[33:37] = [True] * 4
    elif node_type == "extrude":
        mask[37] = True
    elif node_type == "revolve":
        mask[38] = True
    return tuple(mask)


def _validate_geometry(raw, tokens, category_usable, failures):
    for index, (node, token) in enumerate(zip(raw.raw_nodes, tokens)):
        if not category_usable[index]:
            continue
        expected = _expected_mask(token, node.categorical_ids)
        if node.derived_geometry_mask != expected:
            failures.add(
                "geometry_mask_applicability", "controlled_domain",
                "controlled_geometry", "raw_nodes[{}].derived_geometry_mask".format(index),
                "geometry mask does not match node and primitive applicability",
            )
            continue
        try:
            physical = denormalize_applicable_geometry(
                node.normalized_geometry, node.derived_geometry_mask
            )
        except ModelDataError:
            failures.add(
                "geometry_out_of_range", "controlled_domain",
                "controlled_geometry", "raw_nodes[{}].normalized_geometry".format(index),
                "applicable normalized geometry is outside the authoritative contract",
            )
            continue
        if token == "reference_plane":
            _validate_plane(node, physical, index, failures)
        elif token == "axis":
            if not _close_tuple(physical[33:37], (0.0, 0.0, 0.0, 1.0)):
                failures.add(
                    "invalid_axis_geometry", "controlled_domain",
                    "controlled_geometry", "raw_nodes[{}].normalized_geometry".format(index),
                    "controlled revolve axes require point (0,0) and direction (0,1)",
                )
        elif token in ("extrude", "revolve"):
            channel = 37 if token == "extrude" else 38
            if physical[channel] is None or physical[channel] <= 0.0:
                failures.add(
                    "invalid_operation_parameter", "controlled_domain",
                    "controlled_geometry",
                    "raw_nodes[{}].normalized_geometry[{}]".format(index, channel),
                    "controlled operation parameter must be positive",
                )
        elif token == "sketch":
            _validate_primitives(node, physical, index, failures)


def _validate_plane(node, physical, index, failures):
    plane = REFERENCE_PLANES.tokens[node.categorical_ids[3]]
    frames = {
        "XY": (1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        "XZ": (1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        "YZ": (0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
    }
    expected = (0.0, 0.0, 0.0) + frames.get(plane, ())
    if len(expected) != 9 or not _close_tuple(physical[0:9], expected):
        failures.add(
            "invalid_reference_plane_geometry", "controlled_domain",
            "controlled_geometry", "raw_nodes[{}].normalized_geometry".format(index),
            "reference-plane frame must match its categorical plane",
        )


def _validate_primitives(node, physical, index, failures):
    primitives = tuple(
        PRIMITIVE_TYPES.tokens[item] for item in node.categorical_ids[4:8]
    )
    ok = True
    if primitives[0] == "circle":
        ok = physical[11] is not None and physical[11] > 0.0
    elif primitives == ("line", "line", "line", "line"):
        entries = tuple(_primitive_entry(physical, slot, "line") for slot in range(4))
        ok = _valid_primitive_loop(entries)
    elif primitives == ("arc", "arc", "line", "line"):
        order = ((2, "line"), (0, "arc"), (3, "line"), (1, "arc"))
        entries = tuple(_primitive_entry(physical, slot, kind) for slot, kind in order)
        ok = _valid_primitive_loop(entries)
    if not ok:
        failures.add(
            "invalid_primitive_geometry", "controlled_domain",
            "controlled_geometry", "raw_nodes[{}].normalized_geometry".format(index),
            "primitive geometry must form the selected controlled profile family",
        )


def _primitive_entry(physical, slot, kind):
    start = 9 + slot * 6
    if kind == "line":
        return kind, physical[start:start + 2], None, physical[start + 2:start + 4]
    return (
        kind,
        physical[start:start + 2],
        physical[start + 2:start + 4],
        physical[start + 4:start + 6],
    )


def _valid_primitive_loop(entries):
    for kind, start, midpoint, end in entries:
        if _close_tuple(start, end):
            return False
        if kind == "arc":
            if _close_tuple(start, midpoint) or _close_tuple(midpoint, end):
                return False
            if abs(_cross(start, midpoint, end)) <= LOOP_CONTINUITY_TOLERANCE:
                return False
    if any(
        not _close_tuple(entry[3], entries[(index + 1) % len(entries)][1])
        for index, entry in enumerate(entries)
    ):
        return False
    polygon = []
    for kind, start, midpoint, end in entries:
        polygon.append(start)
        if midpoint is not None:
            polygon.append(midpoint)
    if abs(_polygon_area2(polygon)) <= LOOP_CONTINUITY_TOLERANCE:
        return False
    return not _polygon_self_intersects(polygon)


def _cross(start, middle, end):
    return (
        (middle[0] - start[0]) * (end[1] - start[1])
        - (middle[1] - start[1]) * (end[0] - start[0])
    )


def _polygon_area2(points):
    return sum(
        point[0] * points[(index + 1) % len(points)][1]
        - point[1] * points[(index + 1) % len(points)][0]
        for index, point in enumerate(points)
    )


def _polygon_self_intersects(points):
    count = len(points)
    for first in range(count):
        for second in range(first + 1, count):
            if second in (first, (first + 1) % count):
                continue
            if first == 0 and second == count - 1:
                continue
            if _segments_intersect(
                points[first], points[(first + 1) % count],
                points[second], points[(second + 1) % count],
            ):
                return True
    return False


def _segments_intersect(a, b, c, d):
    first = (_cross(a, b, c), _cross(a, b, d))
    second = (_cross(c, d, a), _cross(c, d, b))
    tolerance = LOOP_CONTINUITY_TOLERANCE
    if (
        first[0] * first[1] < -(tolerance ** 2)
        and second[0] * second[1] < -(tolerance ** 2)
    ):
        return True
    return (
        (abs(first[0]) <= tolerance and _point_on_segment(c, a, b))
        or (abs(first[1]) <= tolerance and _point_on_segment(d, a, b))
        or (abs(second[0]) <= tolerance and _point_on_segment(a, c, d))
        or (abs(second[1]) <= tolerance and _point_on_segment(b, c, d))
    )


def _point_on_segment(point, start, end):
    tolerance = LOOP_CONTINUITY_TOLERANCE
    return (
        min(start[0], end[0]) - tolerance
        <= point[0]
        <= max(start[0], end[0]) + tolerance
        and min(start[1], end[1]) - tolerance
        <= point[1]
        <= max(start[1], end[1]) + tolerance
    )


def _validate_operations(raw, grammar, max_operations, failures):
    positions = tuple(item[3] for item in grammar)
    if len(positions) > max_operations:
        failures.add(
            "operation_limit_exceeded", "controlled_domain",
            "controlled_operations", "predicted_operation_count",
            "predicted operation count exceeds configured maximum",
        )
    for order, (_, _, _, position, kind) in enumerate(grammar):
        row = raw.raw_nodes[position].categorical_ids
        if OPERATION_TYPES.tokens[row[0]] != kind:
            failures.add(
                "operation_type_mismatch", "controlled_domain",
                "controlled_operations",
                "raw_nodes[{}].categorical_ids[0]".format(position),
                "operation category must match operation node type",
            )
        allowed = ("new_body",) if order == 0 else ("join", "cut")
        if BOOLEAN_MODES.tokens[row[1]] not in allowed:
            failures.add(
                "invalid_boolean_sequence", "controlled_domain",
                "controlled_operations",
                "raw_nodes[{}].categorical_ids[1]".format(position),
                "first operation must be new_body and later operations join or cut",
            )


def _required_edges(grammar):
    placed_on = EDGE_TYPES.id("placed_on")
    defined_in = EDGE_TYPES.id("defined_in")
    uses_profile = EDGE_TYPES.id("uses_profile")
    uses_axis = EDGE_TYPES.id("uses_axis")
    result = set()
    for sketch, profile, axis, operation, _ in grammar:
        result.add((sketch, 0, placed_on))
        result.add((profile, sketch, defined_in))
        result.add((operation, profile, uses_profile))
        if axis is not None:
            result.add((axis, sketch, defined_in))
            result.add((operation, axis, uses_axis))
    return result


def _validate_edges(raw, grammar, failures):
    depends_on = EDGE_TYPES.id("depends_on")
    actual = {
        (edge.source_index, edge.target_index, edge.edge_type_id)
        for edge in raw.raw_edges
        if edge.present and edge.edge_type_id != depends_on
    }
    required = _required_edges(grammar)
    for source, target, edge_type in sorted(required - actual):
        failures.add(
            "missing_required_edge", "controlled_domain", "controlled_edges",
            "edge[{},{}]".format(source, target),
            "required controlled relationship is absent",
        )
    for source, target, edge_type in sorted(actual - required):
        failures.add(
            "unexpected_edge", "controlled_domain", "controlled_edges",
            "edge[{},{}]".format(source, target),
            "present relationship is outside the controlled flat grammar",
        )


def _validate_cross_records(
    raw, grammar, failures, *, pointers_usable, edges_usable
):
    positions = tuple(item[3] for item in grammar)
    if raw.predicted_operation_node_indices != positions:
        failures.add(
            "operation_metadata_mismatch", "controlled_domain",
            "cross_record_consistency", "predicted_operation_node_indices",
            "operation metadata must match operation nodes in canonical order",
        )
    if pointers_usable:
        selected = tuple(
            item.selected_node_index for item in raw.raw_operation_pointers
        )
        if selected != positions[:len(raw.raw_operation_pointers)]:
            failures.add(
                "operation_pointer_inconsistency", "controlled_domain",
                "cross_record_consistency", "raw_operation_pointers",
                "operation pointers must select chronological operation nodes",
            )
    if not edges_usable:
        return
    depends_on = EDGE_TYPES.id("depends_on")
    actual_dependencies = {
        (edge.source_index, edge.target_index)
        for edge in raw.raw_edges
        if edge.present and edge.edge_type_id == depends_on
    }
    expected_dependencies = set(zip(positions[1:], positions[:-1]))
    if actual_dependencies != expected_dependencies:
        failures.add(
            "operation_edge_inconsistency", "controlled_domain",
            "cross_record_consistency", "raw_edges",
            "dependency edges must agree with chronological operation records",
        )


def _finite_number(value):
    # Phase A materializes geometry and retained logits with Python float().
    if type(value) is not float:
        return False
    return math.isfinite(value)


def _close_tuple(left, right):
    return len(left) == len(right) and all(
        a is not None and math.isclose(
            float(a), float(b), rel_tol=0.0, abs_tol=_GEOMETRY_TOLERANCE
        )
        for a, b in zip(left, right)
    )


def _failure_key(failure):
    return (
        _STAGE_INDEX[failure.stage],
        _CODE_INDEX[failure.stage][failure.code],
        _location_key(failure.location),
        LAYER_ORDER[failure.layer],
        failure.detail,
    )


def _location_key(location):
    return tuple(
        int(part) if part.isdigit() else part
        for part in _LOCATION_PART.split(location)
    )


def _codes_through_layer(failures, layer):
    maximum = LAYER_ORDER[layer]
    return tuple(dict.fromkeys(
        item.code for item in failures if LAYER_ORDER[item.layer] <= maximum
    ))


def _has_prerequisite_failures(failures, stage_index):
    return any(_STAGE_INDEX[item.stage] <= stage_index for item in failures.items)


def _new_failures(failures, start):
    return len(failures.items) != start
