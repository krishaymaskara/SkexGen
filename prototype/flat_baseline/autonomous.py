"""Raw length-conditioned autoregressive decoding for the flat baseline."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import torch

from prototype.model_data.geometry import (
    GEOMETRY_WIDTH,
    MAX_PRIMITIVES,
    PRIMITIVE_SLOT_WIDTH,
)
from prototype.model_data.vocab import NODE_TYPES, PRIMITIVE_TYPES


RAW_PREFIX_FEEDBACK = "raw_argmax_with_derived_geometry_mask"
TEACHER_FORCED_PREFIX_FEEDBACK = "shifted_target_prefix"
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


@dataclass(frozen=True)
class PairedDecodedPredictions:
    latent_indices: tuple
    teacher_forced: tuple
    predicted_history: tuple
    node_count_source: str
    teacher_forced_prefix_feedback: str
    predicted_history_prefix_feedback: str


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

    _validate_latent_indices(model, latent_indices)
    counts = _validated_node_counts(
        node_counts, latent_indices.size(0), model.config.max_nodes
    )
    _validate_provenance(node_count_source, prefix_feedback)
    with _evaluation_mode(model):
        memory = model.memory_from_indices(latent_indices)
        _validate_memory(model, memory, latent_indices)
        return _decode_length_conditioned_rows(
            model,
            memory,
            latent_indices,
            counts,
            node_count_source,
            prefix_feedback,
        )


def decode_length_conditioned_from_memory(
    model,
    memory,
    latent_indices,
    *,
    node_counts,
    node_count_source,
    prefix_feedback,
):
    """Decode generated prefixes from caller-supplied lookup memory."""

    _validate_memory(model, memory, latent_indices)
    counts = _validated_node_counts(
        node_counts, memory.size(0), model.config.max_nodes
    )
    _validate_provenance(node_count_source, prefix_feedback)
    with _evaluation_mode(model):
        return _decode_length_conditioned_rows(
            model,
            memory,
            latent_indices,
            counts,
            node_count_source,
            prefix_feedback,
        )


def decode_teacher_forced_from_memory(
    model,
    memory,
    latent_indices,
    target,
    *,
    node_count_source,
):
    """Decode model outputs from exact shifted authoritative prefixes."""

    _validate_memory(model, memory, latent_indices)
    _validate_node_count_source(node_count_source)
    counts = _validated_teacher_target(
        model, target, memory, latent_indices
    )
    with _evaluation_mode(model):
        return tuple(
            _decode_teacher_forced_one(
                model,
                memory[index : index + 1],
                latent_indices[index],
                target,
                index,
                count,
                node_count_source,
            )
            for index, count in enumerate(counts)
        )


def decode_paired_from_batch(
    model,
    authoritative_batch,
    *,
    node_counts,
    node_count_source,
):
    """Encode once and decode both prefix policies from one lookup memory."""

    inputs, target = _paired_batch_parts(authoritative_batch)
    _validate_authoritative_mask_alignment(inputs, target)
    _validate_node_count_source(node_count_source)
    batch_size = _encoder_batch_size(inputs)
    counts = _validated_node_counts(
        node_counts, batch_size, model.config.max_nodes
    )
    with _evaluation_mode(model):
        encoded = model.encode_to_memory(
            inputs["categorical_ids"],
            inputs["geometry"],
            inputs["geometry_mask"],
            inputs["padding_mask"],
        )
        latent_indices = encoded.vq.indices
        _validate_latent_indices(model, latent_indices)
        memory = model.memory_from_indices(latent_indices)
        _validate_memory(model, memory, latent_indices)
        target_counts = _validated_teacher_target(
            model, target, memory, latent_indices
        )
        if counts != target_counts:
            raise AutonomousDecodingError(
                "target_node_count_mismatch",
                "node_counts must match authoritative target node masks",
            )
        teacher = decode_teacher_forced_from_memory(
            model,
            memory,
            latent_indices,
            target,
            node_count_source=node_count_source,
        )
        predicted = decode_length_conditioned_from_memory(
            model,
            memory,
            latent_indices,
            node_counts=counts,
            node_count_source=node_count_source,
            prefix_feedback=RAW_PREFIX_FEEDBACK,
        )
    provenance = tuple(
        tuple(int(value) for value in row.detach().cpu().tolist())
        for row in latent_indices
    )
    return PairedDecodedPredictions(
        provenance,
        teacher,
        predicted,
        node_count_source,
        TEACHER_FORCED_PREFIX_FEEDBACK,
        RAW_PREFIX_FEEDBACK,
    )


def _decode_length_conditioned_rows(
    model,
    memory,
    latent_indices,
    counts,
    node_count_source,
    prefix_feedback,
):
    return tuple(
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
    return _finish_prediction(
        model,
        aligned,
        tuple(raw_nodes),
        latent_indices,
        node_count,
        node_count_source,
        prefix_feedback,
    )


def _decode_teacher_forced_one(
    model,
    memory,
    latent_indices,
    target,
    batch_index,
    node_count,
    node_count_source,
):
    node_types = target["node_type_ids"][
        batch_index : batch_index + 1, :node_count
    ]
    attributes = target["categorical_attributes"][
        batch_index : batch_index + 1, :node_count
    ]
    categories = torch.cat((node_types.unsqueeze(-1), attributes), dim=-1)
    geometry = target["geometry"][
        batch_index : batch_index + 1, :node_count
    ]
    geometry_mask = target["geometry_mask"][
        batch_index : batch_index + 1, :node_count
    ]
    aligned = model.decode_prefix(
        memory,
        categories[:, :-1],
        geometry[:, :-1],
        geometry_mask[:, :-1],
    )
    raw_nodes = _raw_nodes_from_output(aligned, node_count)
    return _finish_prediction(
        model,
        aligned,
        raw_nodes,
        latent_indices,
        node_count,
        node_count_source,
        TEACHER_FORCED_PREFIX_FEEDBACK,
    )


def _raw_nodes_from_output(output, node_count):
    raw_nodes = []
    for position in range(node_count):
        node_type = int(
            output.node_type_logits[0, position].argmax(dim=-1).item()
        )
        attributes = tuple(
            int(logits[0, position].argmax(dim=-1).item())
            for logits in output.categorical_logits
        )
        mask = derive_geometry_mask(node_type, attributes)
        raw_nodes.append(
            RawDecodedNode(
                position,
                node_type,
                attributes,
                tuple(
                    float(value)
                    for value in output.geometry[
                        0, position
                    ].detach().cpu().tolist()
                ),
                mask,
            )
        )
    return tuple(raw_nodes)


def _finish_prediction(
    model,
    aligned,
    raw_nodes,
    latent_indices,
    node_count,
    node_count_source,
    prefix_feedback,
):
    device = aligned.decoded_states.device
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
        raw_nodes,
        tuple(raw_edges),
        operation_nodes,
        len(operation_nodes),
        len(operation_nodes) > model.config.max_operations,
        tuple(pointers),
    )


@contextmanager
def _evaluation_mode(model):
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            yield
    finally:
        model.train(was_training)


def _validate_memory(model, memory, latent_indices):
    if not torch.is_tensor(memory):
        raise TypeError("memory must be a torch.Tensor")
    if memory.dim() != 3:
        raise ValueError(
            "memory must have shape [B, latent_tokens, model_dim]"
        )
    if (
        memory.size(0) == 0
        or memory.size(1) != model.config.latent_tokens
        or memory.size(2) != model.config.model_dim
    ):
        raise ValueError(
            "memory must have shape [B, latent_tokens, model_dim]"
        )
    if not memory.dtype.is_floating_point:
        raise TypeError("memory must be floating point")
    if memory.dtype != model.bos.dtype:
        raise TypeError("memory dtype must match model dtype")
    if memory.device != model.bos.device:
        raise ValueError("memory device must match model device")
    _validate_latent_indices(model, latent_indices)
    if latent_indices.size(0) != memory.size(0):
        raise ValueError("latent indices and memory batch sizes disagree")
    if latent_indices.device != memory.device:
        raise ValueError("latent indices and memory must share a device")


def _validate_latent_indices(model, latent_indices):
    if not torch.is_tensor(latent_indices):
        raise TypeError("latent indices must be a torch.Tensor")
    if latent_indices.dtype != torch.long:
        raise TypeError("latent indices must use torch.long")
    if latent_indices.dim() == 2 and latent_indices.size(0) == 0:
        raise ValueError("latent indices require a positive batch size")
    if (
        latent_indices.dim() != 2
        or latent_indices.size(1) != model.config.latent_tokens
    ):
        raise ValueError(
            "latent indices must have shape [B, latent_tokens]"
        )
    if latent_indices.numel() and (
        int(latent_indices.min().item()) < 0
        or int(latent_indices.max().item()) >= model.config.codebook_size
    ):
        raise ValueError("latent index is outside the configured codebook")


def _validate_provenance(node_count_source, prefix_feedback):
    _validate_node_count_source(node_count_source)
    if prefix_feedback != RAW_PREFIX_FEEDBACK:
        raise AutonomousDecodingError(
            "unsupported_prefix_feedback", repr(prefix_feedback)
        )


def _validate_node_count_source(node_count_source):
    if not isinstance(node_count_source, str) or not node_count_source:
        raise AutonomousDecodingError(
            "invalid_node_count_source",
            "node_count_source must be a nonempty string",
        )


def _validated_teacher_target(model, target, memory, latent_indices):
    required = (
        "node_type_ids",
        "categorical_attributes",
        "geometry",
        "geometry_mask",
        "node_mask",
    )
    if not isinstance(target, dict) or any(
        name not in target for name in required
    ):
        raise ValueError("target must contain teacher-forced node tensors")
    values = tuple(target[name] for name in required)
    if not all(torch.is_tensor(value) for value in values):
        raise TypeError("teacher-forced target values must be tensors")
    node_types, attributes, geometry, geometry_mask, node_mask = values
    batch_size = memory.size(0)
    if node_types.dim() != 2:
        raise ValueError("target node_type_ids must have shape [B, N]")
    padded_count = node_types.size(1)
    if (
        node_types.shape != (batch_size, padded_count)
        or attributes.shape != (batch_size, padded_count, 9)
        or geometry.shape != (batch_size, padded_count, GEOMETRY_WIDTH)
        or geometry_mask.shape != geometry.shape
        or node_mask.shape != (batch_size, padded_count)
    ):
        raise ValueError("teacher-forced target tensors are misaligned")
    if node_types.dtype != torch.long or attributes.dtype != torch.long:
        raise TypeError("teacher-forced categories must use torch.long")
    if geometry.dtype != memory.dtype:
        raise TypeError("target geometry dtype must match memory dtype")
    if geometry_mask.dtype != torch.bool or node_mask.dtype != torch.bool:
        raise TypeError("teacher-forced masks must use torch.bool")
    if any(value.device != memory.device for value in values):
        raise ValueError("teacher-forced target must share memory device")
    if latent_indices.size(0) != batch_size:
        raise ValueError("target and latent batch sizes disagree")
    counts = []
    for row in node_mask:
        count = int(row.sum().item())
        if (
            count <= 0
            or count > model.config.max_nodes
            or not bool(row[:count].all().item())
            or bool(row[count:].any().item())
        ):
            raise AutonomousDecodingError(
                "invalid_target_node_mask",
                "target node masks must be positive contiguous prefixes",
            )
        counts.append(count)
    return tuple(counts)


def _paired_batch_parts(authoritative_batch):
    if (
        not isinstance(authoritative_batch, tuple)
        or len(authoritative_batch) != 2
    ):
        raise TypeError(
            "authoritative_batch must be an (inputs, target) tuple"
        )
    inputs, target = authoritative_batch
    required = (
        "categorical_ids",
        "geometry",
        "geometry_mask",
        "padding_mask",
    )
    if not isinstance(inputs, dict) or any(
        name not in inputs for name in required
    ):
        raise ValueError("authoritative inputs are missing encoder tensors")
    return inputs, target


def _encoder_batch_size(inputs):
    categorical_ids = inputs["categorical_ids"]
    if not torch.is_tensor(categorical_ids) or categorical_ids.dim() != 3:
        raise ValueError("categorical_ids must have shape [B, N, 10]")
    if categorical_ids.size(0) == 0:
        raise ValueError("authoritative batch must be nonempty")
    return categorical_ids.size(0)


def _validate_authoritative_mask_alignment(inputs, target):
    padding_mask = inputs.get("padding_mask")
    if not isinstance(target, dict) or "node_mask" not in target:
        raise ValueError("authoritative target is missing node_mask")
    node_mask = target["node_mask"]
    if not torch.is_tensor(padding_mask) or not torch.is_tensor(node_mask):
        raise TypeError("authoritative node masks must be tensors")
    if padding_mask.shape != node_mask.shape:
        raise ValueError(
            "input padding_mask and target node_mask shapes disagree"
        )
    if padding_mask.dtype != torch.bool or node_mask.dtype != torch.bool:
        raise TypeError("authoritative node masks must use torch.bool")
    if padding_mask.device != node_mask.device:
        raise ValueError(
            "input padding_mask and target node_mask devices disagree"
        )
    if not torch.equal(padding_mask, node_mask):
        raise ValueError("input padding_mask and target node_mask disagree")


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
