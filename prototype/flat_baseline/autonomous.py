"""Raw length-conditioned autoregressive decoding for the flat baseline."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from prototype.model_data.geometry import (
    GEOMETRY_WIDTH,
    MAX_PRIMITIVES,
    PRIMITIVE_SLOT_WIDTH,
)
from prototype.model_data.vocab import NODE_TYPES, PRIMITIVE_TYPES


RAW_PREFIX_FEEDBACK = "raw_argmax_with_derived_geometry_mask"
REQUESTED_LENGTH_TERMINATION = "requested_node_count_reached"


class AutonomousDecodingError(ValueError):
    """A requested raw decoding operation is malformed or unsupported."""

    def __init__(self, code, detail):
        self.code = code
        self.detail = detail
        super().__init__("{}: {}".format(code, detail))


@dataclass(frozen=True)
class RawDecodedNode:
    position: int
    node_type_id: int
    categorical_ids: tuple
    normalized_geometry: tuple
    derived_geometry_mask: tuple


@dataclass(frozen=True)
class RawDecodedEdge:
    source_index: int
    target_index: int
    presence_logit: float
    present: bool
    edge_type_id: int


@dataclass(frozen=True)
class RawDecodedPointer:
    query_index: int
    selected_node_index: int
    selected_logit: float


@dataclass(frozen=True)
class RawDecodedPrediction:
    latent_indices: tuple
    node_count: int
    node_count_source: str
    termination_reason: str
    termination_is_learned: bool
    prefix_feedback: str
    raw_nodes: tuple
    raw_edges: tuple
    predicted_operation_node_indices: tuple
    predicted_operation_count: int
    operation_count_exceeds_limit: bool
    raw_operation_pointers: tuple


def derive_geometry_mask(node_type_id, categorical_ids):
    """Derive the 39-channel applicability mask from raw categorical choices."""

    if (
        isinstance(node_type_id, bool)
        or not isinstance(node_type_id, int)
        or not 0 <= node_type_id < len(NODE_TYPES.tokens)
    ):
        raise AutonomousDecodingError(
            "invalid_node_type_id", repr(node_type_id)
        )
    if len(categorical_ids) != 9:
        raise AutonomousDecodingError(
            "invalid_categorical_width",
            "categorical_ids must contain nine fields",
        )
    mask = [False] * GEOMETRY_WIDTH
    node_type = NODE_TYPES.tokens[node_type_id]
    if node_type == "reference_plane":
        mask[0:9] = [True] * 9
    elif node_type == "sketch":
        for slot in range(MAX_PRIMITIVES):
            primitive_id = categorical_ids[4 + slot]
            if (
                isinstance(primitive_id, bool)
                or not isinstance(primitive_id, int)
                or not 0 <= primitive_id < len(PRIMITIVE_TYPES.tokens)
            ):
                raise AutonomousDecodingError(
                    "invalid_primitive_type_id", repr(primitive_id)
                )
            primitive_type = PRIMITIVE_TYPES.tokens[primitive_id]
            widths = {"line": 4, "arc": 6, "circle": 3}
            width = widths.get(primitive_type, 0)
            start = 9 + slot * PRIMITIVE_SLOT_WIDTH
            mask[start : start + width] = [True] * width
    elif node_type == "axis":
        mask[33:37] = [True] * 4
    elif node_type == "extrude":
        mask[37] = True
    elif node_type == "revolve":
        mask[38] = True
    return tuple(mask)


def decode_length_conditioned(
    model,
    latent_indices,
    *,
    node_counts,
    node_count_source="caller_supplied",
    prefix_feedback=RAW_PREFIX_FEEDBACK,
):
    """Greedily decode exactly the externally supplied node count per example."""

    if prefix_feedback != RAW_PREFIX_FEEDBACK:
        raise AutonomousDecodingError(
            "unsupported_prefix_feedback", repr(prefix_feedback)
        )
    if not isinstance(node_count_source, str) or not node_count_source:
        raise AutonomousDecodingError(
            "invalid_node_count_source",
            "node_count_source must be a nonempty string",
        )
    if latent_indices.dtype != torch.long:
        raise TypeError("latent indices must use torch.long")
    if latent_indices.dim() != 2:
        raise ValueError(
            "latent indices must have shape [B, latent_tokens]"
        )
    counts = _validated_node_counts(
        node_counts, latent_indices.size(0), model.config.max_nodes
    )

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            memory = model.memory_from_indices(latent_indices)
            results = tuple(
                _decode_one(
                    model,
                    memory[index : index + 1],
                    latent_indices[index],
                    node_count,
                    node_count_source,
                    prefix_feedback,
                )
                for index, node_count in enumerate(counts)
            )
    finally:
        if was_training:
            model.train()
    return results


def _decode_one(
    model,
    memory,
    latent_indices,
    node_count,
    node_count_source,
    prefix_feedback,
):
    device = memory.device
    categories = torch.empty(
        (1, 0, 10), dtype=torch.long, device=device
    )
    geometry = torch.empty(
        (1, 0, GEOMETRY_WIDTH), dtype=memory.dtype, device=device
    )
    geometry_mask = torch.empty(
        (1, 0, GEOMETRY_WIDTH), dtype=torch.bool, device=device
    )
    raw_nodes = []

    for position in range(node_count):
        output = model.decode_prefix(
            memory, categories, geometry, geometry_mask
        )
        node_type = output.node_type_logits[:, -1].argmax(dim=-1)
        attributes = torch.stack(
            tuple(item[:, -1].argmax(dim=-1) for item in output.categorical_logits),
            dim=-1,
        )
        current_geometry = output.geometry[:, -1]
        node_type_value = int(node_type.item())
        attribute_values = tuple(
            int(item) for item in attributes[0].detach().cpu().tolist()
        )
        current_mask_values = derive_geometry_mask(
            node_type_value, attribute_values
        )
        current_mask = torch.tensor(
            current_mask_values, dtype=torch.bool, device=device
        ).unsqueeze(0)
        current_categories = torch.cat(
            (node_type.unsqueeze(-1), attributes), dim=-1
        ).unsqueeze(1)
        categories = torch.cat((categories, current_categories), dim=1)
        geometry = torch.cat(
            (geometry, current_geometry.unsqueeze(1)), dim=1
        )
        geometry_mask = torch.cat(
            (geometry_mask, current_mask.unsqueeze(1)), dim=1
        )
        raw_nodes.append(
            RawDecodedNode(
                position,
                node_type_value,
                attribute_values,
                tuple(
                    float(item)
                    for item in current_geometry[0].detach().cpu().tolist()
                ),
                current_mask_values,
            )
        )

    aligned = model.decode_prefix(
        memory,
        categories[:, :-1],
        geometry[:, :-1],
        geometry_mask[:, :-1],
    )
    node_mask = torch.ones(
        (1, node_count), dtype=torch.bool, device=device
    )
    relations = model.decode_relations(aligned.decoded_states, node_mask)
    raw_edges = []
    for source in range(node_count):
        for target in range(node_count):
            presence_logit = float(
                relations.edge_presence_logits[0, source, target].item()
            )
            edge_type = int(
                relations.edge_type_logits[0, source, target]
                .argmax(dim=-1)
                .item()
            )
            raw_edges.append(
                RawDecodedEdge(
                    source,
                    target,
                    presence_logit,
                    presence_logit >= 0.0,
                    edge_type,
                )
            )

    operation_ids = {
        NODE_TYPES.id("extrude"),
        NODE_TYPES.id("revolve"),
    }
    operation_nodes = tuple(
        node.position
        for node in raw_nodes
        if node.node_type_id in operation_ids
    )
    pointer_count = min(
        len(operation_nodes), model.config.max_operations
    )
    pointers = []
    for query_index in range(pointer_count):
        logits = relations.operation_pointer_logits[0, query_index]
        selected = int(logits.argmax(dim=-1).item())
        pointers.append(
            RawDecodedPointer(
                query_index,
                selected,
                float(logits[selected].item()),
            )
        )
    return RawDecodedPrediction(
        tuple(int(item) for item in latent_indices.detach().cpu().tolist()),
        node_count,
        node_count_source,
        REQUESTED_LENGTH_TERMINATION,
        False,
        prefix_feedback,
        tuple(raw_nodes),
        tuple(raw_edges),
        operation_nodes,
        len(operation_nodes),
        len(operation_nodes) > model.config.max_operations,
        tuple(pointers),
    )


def _validated_node_counts(node_counts, batch_size, max_nodes):
    try:
        counts = tuple(node_counts)
    except TypeError as exc:
        raise AutonomousDecodingError(
            "invalid_node_counts", "node_counts must be an iterable"
        ) from exc
    if len(counts) != batch_size:
        raise AutonomousDecodingError(
            "node_count_batch_mismatch",
            "node_counts must contain one value per latent row",
        )
    for value in counts:
        if isinstance(value, bool) or not isinstance(value, int):
            raise AutonomousDecodingError(
                "invalid_node_count",
                "node counts must be non-Boolean integers",
            )
        if value <= 0:
            raise AutonomousDecodingError(
                "invalid_node_count", "node counts must be positive"
            )
        if value > max_nodes:
            raise AutonomousDecodingError(
                "node_count_exceeds_limit",
                "node count {} exceeds checkpoint max_nodes {}".format(
                    value, max_nodes
                ),
            )
    return counts
