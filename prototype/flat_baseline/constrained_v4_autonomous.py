"""Autonomous predicted-history decoding for the constrained V4 flat model."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Mapping, Tuple

import torch

from prototype.model_data.geometry import GEOMETRY_WIDTH
from prototype.model_data.vocab import (
    BOOLEAN_MODES,
    DIRECTIONS,
    EDGE_TYPES,
    LOOP_ROLES,
    NODE_TYPES,
    OPERATION_TYPES,
    REFERENCE_PLANES,
)
from prototype.profile_geometry import PROFILE_FAMILIES
from prototype.node_conditioned_categories import V4_CATEGORICAL_CONTRACT_ID

from .autonomous import (
    RAW_PREFIX_FEEDBACK,
    REQUESTED_LENGTH_TERMINATION,
    RawDecodedEdge,
    RawDecodedNode,
    RawDecodedPointer,
)
from .constrained_v4 import (
    REMAINING_GEOMETRY_WIDTH,
    ConstrainedProfileV4Model,
    ConstrainedProfileV4PrefixOutput,
)
from .constrained_v4_conversion import (
    V4TeacherForcedRawPrediction,
    _validate_v4_raw_contract,
    construct_v4_predicted_node_tensors,
)
from .conversion import ConversionResult, validate_and_convert_raw_prediction


V4_ENCODED_MEMORY_SOURCE = "v4_encoder_quantized_memory"
V4_AUTONOMOUS_PREFIX_FEEDBACK = "v4_predicted_constrained_records"
_ENCODER_FIELDS = (
    "categorical_ids",
    "geometry",
    "geometry_mask",
    "padding_mask",
)
_RETAINED_VOCABULARIES = (
    OPERATION_TYPES,
    BOOLEAN_MODES,
    DIRECTIONS,
    REFERENCE_PLANES,
    LOOP_ROLES,
)


@dataclass(frozen=True)
class V4EncodedMemory:
    """Quantized V4 memory and immutable provenance needed for decoding."""

    memory: torch.Tensor
    code_indices: torch.Tensor
    encoding_source: str
    model_name: str


@dataclass(frozen=True)
class V4AutonomousRawPrediction(V4TeacherForcedRawPrediction):
    """Autonomous raw evidence with explicit encoded-memory provenance."""

    encoded_memory_source: str


def encode_v4_to_memory(
    model: ConstrainedProfileV4Model,
    source_batch: Mapping,
) -> V4EncodedMemory:
    """Encode source-only flat tensors without accepting reconstruction targets."""

    _validate_v4_model(model)
    _validate_source_batch(model, source_batch)
    with _evaluation_mode(model):
        encoded = model.encode_to_memory(
            source_batch["categorical_ids"],
            source_batch["geometry"],
            source_batch["geometry_mask"],
            source_batch["padding_mask"],
        )
    memory = encoded.memory.detach().contiguous()
    code_indices = encoded.vq.indices.detach().contiguous()
    result = V4EncodedMemory(
        memory,
        code_indices,
        V4_ENCODED_MEMORY_SOURCE,
        model.config.model_name,
    )
    _validate_encoded_memory(model, result)
    return result


def greedy_decode_v4_from_memory(
    model: ConstrainedProfileV4Model,
    memory: V4EncodedMemory,
    *,
    node_counts: torch.Tensor,
    node_count_source: str,
) -> Tuple[V4AutonomousRawPrediction, ...]:
    """Greedily decode complete predicted V4 records to authorized lengths."""

    _validate_v4_model(model)
    _validate_encoded_memory(model, memory)
    counts = _validated_node_counts(
        node_counts,
        memory.memory.size(0),
        model.config.max_nodes,
        memory.memory.device,
    )
    _validate_node_count_source(node_count_source)
    with _evaluation_mode(model):
        return tuple(
            _decode_one(
                model,
                memory.memory[index : index + 1],
                memory.code_indices[index],
                count,
                node_count_source,
                memory.encoding_source,
            )
            for index, count in enumerate(counts)
        )


def greedy_decode_v4(
    model: ConstrainedProfileV4Model,
    source_batch: Mapping,
    *,
    node_counts: torch.Tensor,
    node_count_source: str,
) -> Tuple[V4AutonomousRawPrediction, ...]:
    """Encode once and autonomously decode from the resulting V4 memory."""

    memory = encode_v4_to_memory(model, source_batch)
    return greedy_decode_v4_from_memory(
        model,
        memory,
        node_counts=node_counts,
        node_count_source=node_count_source,
    )


def validate_and_convert_v4_autonomous_prediction(
    prediction: V4AutonomousRawPrediction,
    *,
    max_operations: int,
) -> ConversionResult:
    """Run the existing strict converter through a V4 provenance adapter."""

    if not isinstance(prediction, V4AutonomousRawPrediction):
        raise TypeError(
            "prediction must be a V4AutonomousRawPrediction"
        )
    if prediction.prefix_feedback != V4_AUTONOMOUS_PREFIX_FEEDBACK:
        raise ValueError("V4 autonomous prediction has invalid feedback")
    if prediction.encoded_memory_source != V4_ENCODED_MEMORY_SOURCE:
        raise ValueError("V4 autonomous prediction has invalid memory source")
    _validate_v4_raw_contract(prediction)
    compatible = replace(prediction, prefix_feedback=RAW_PREFIX_FEEDBACK)
    return validate_and_convert_raw_prediction(
        compatible, max_operations=max_operations
    )


def _decode_one(
    model,
    memory,
    code_indices,
    node_count,
    node_count_source,
    encoded_memory_source,
):
    device = memory.device
    categories = torch.empty((1, 0, 10), dtype=torch.long, device=device)
    geometry = torch.empty(
        (1, 0, GEOMETRY_WIDTH), dtype=memory.dtype, device=device
    )
    geometry_mask = torch.empty(
        (1, 0, GEOMETRY_WIDTH), dtype=torch.bool, device=device
    )
    raw_nodes = []
    family_logits = []
    family_ids = []
    raw_parameters = []
    constrained_parameters = []
    raw_categorical_ids = []
    conditioned_categorical_ids = []
    categorical_corrections = []

    for position in range(node_count):
        output = model.decode_prefix(
            memory, categories, geometry, geometry_mask
        )
        _validate_prefix_output(model, output, position + 1, memory)
        node_type_ids = output.node_type_logits[:, -1].argmax(dim=-1)
        records = construct_v4_predicted_node_tensors(
            node_type_ids,
            tuple(logits[:, -1] for logits in output.categorical_logits),
            output.profile_family_logits[:, -1],
            output.raw_profile_parameters[:, -1],
            output.remaining_geometry[:, -1],
        )
        current_categories = torch.cat(
            (node_type_ids.unsqueeze(-1), records.categorical_ids), dim=-1
        ).unsqueeze(1)
        categories = torch.cat(
            (categories, current_categories), dim=1
        ).contiguous()
        geometry = torch.cat(
            (geometry, records.geometry.unsqueeze(1)), dim=1
        ).contiguous()
        geometry_mask = torch.cat(
            (geometry_mask, records.geometry_mask.unsqueeze(1)), dim=1
        ).contiguous()
        raw_nodes.append(
            _raw_node_from_record(records, position)
        )
        family_logits.append(
            tuple(
                float(value)
                for value in output.profile_family_logits[
                    0, -1
                ].detach().cpu().tolist()
            )
        )
        family_ids.append(int(records.family_ids[0].item()))
        raw_parameters.append(
            tuple(
                float(value)
                for value in output.raw_profile_parameters[
                    0, -1
                ].detach().cpu().tolist()
            )
        )
        constrained_parameters.append(
            tuple(
                float(value)
                for value in records.constrained_parameters[
                    0
                ].detach().cpu().tolist()
            )
        )
        raw_categorical_ids.append(tuple(
            int(value)
            for value in records.raw_categorical_argmax_ids[0].cpu().tolist()
        ))
        conditioned_categorical_ids.append(tuple(
            int(value)
            for value in records.node_conditioned_categorical_ids[0].cpu().tolist()
        ))
        categorical_corrections.append(tuple(
            bool(value)
            for value in records.categorical_correction_mask[0].cpu().tolist()
        ))

    aligned = model.decode_prefix(
        memory,
        categories[:, :-1],
        geometry[:, :-1],
        geometry_mask[:, :-1],
    )
    _validate_prefix_output(model, aligned, node_count, memory)
    node_mask = torch.ones(
        (1, node_count), dtype=torch.bool, device=device
    )
    relations = model.decode_relations(aligned.decoded_states, node_mask)
    _validate_relation_output(model, relations, node_count, memory)
    edges = _raw_edges(relations, node_count)
    operation_nodes = _operation_nodes(raw_nodes)
    pointers = _raw_pointers(
        relations, node_count, len(operation_nodes), model.config.max_operations
    )
    return V4AutonomousRawPrediction(
        tuple(
            int(value) for value in code_indices.detach().cpu().tolist()
        ),
        node_count,
        node_count_source,
        REQUESTED_LENGTH_TERMINATION,
        False,
        V4_AUTONOMOUS_PREFIX_FEEDBACK,
        tuple(raw_nodes),
        edges,
        operation_nodes,
        len(operation_nodes),
        len(operation_nodes) > model.config.max_operations,
        pointers,
        tuple(family_logits),
        tuple(family_ids),
        tuple(raw_parameters),
        tuple(constrained_parameters),
        tuple(raw_categorical_ids),
        tuple(conditioned_categorical_ids),
        tuple(categorical_corrections),
        V4_CATEGORICAL_CONTRACT_ID,
        encoded_memory_source,
    )


def _raw_node_from_record(records, position):
    return RawDecodedNode(
        position,
        int(records.node_type_ids[0].item()),
        tuple(
            int(value)
            for value in records.categorical_ids[
                0
            ].detach().cpu().tolist()
        ),
        tuple(
            float(value)
            for value in records.geometry[0].detach().cpu().tolist()
        ),
        tuple(
            bool(value)
            for value in records.geometry_mask[0].detach().cpu().tolist()
        ),
    )


def _raw_edges(relations, node_count):
    result = []
    for source in range(node_count):
        for destination in range(node_count):
            presence_logit = float(
                relations.edge_presence_logits[
                    0, source, destination
                ].item()
            )
            edge_type_id = int(
                relations.edge_type_logits[
                    0, source, destination
                ].argmax(dim=-1).item()
            )
            result.append(
                RawDecodedEdge(
                    source,
                    destination,
                    presence_logit,
                    presence_logit >= 0.0,
                    edge_type_id,
                )
            )
    return tuple(result)


def _operation_nodes(raw_nodes):
    operation_ids = {
        NODE_TYPES.id("extrude"),
        NODE_TYPES.id("revolve"),
    }
    return tuple(
        node.position
        for node in raw_nodes
        if node.node_type_id in operation_ids
    )


def _raw_pointers(relations, node_count, operation_count, max_operations):
    result = []
    for query_index in range(min(operation_count, max_operations)):
        logits = relations.operation_pointer_logits[
            0, query_index, :node_count
        ]
        selected = int(logits.argmax(dim=-1).item())
        result.append(
            RawDecodedPointer(
                query_index,
                selected,
                float(logits[selected].item()),
            )
        )
    return tuple(result)


@contextmanager
def _evaluation_mode(model):
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            yield
    finally:
        model.train(was_training)


def _validate_v4_model(model):
    if not isinstance(model, ConstrainedProfileV4Model):
        raise TypeError("model must be ConstrainedProfileV4Model")


def _validate_source_batch(model, source_batch):
    if not isinstance(source_batch, Mapping):
        raise TypeError("source_batch must be a mapping of encoder tensors")
    if set(source_batch) != set(_ENCODER_FIELDS):
        raise ValueError(
            "source_batch must contain only the four encoder tensors"
        )
    values = tuple(source_batch[name] for name in _ENCODER_FIELDS)
    if not all(torch.is_tensor(value) for value in values):
        raise TypeError("source_batch values must be tensors")
    model._validate_encoder_inputs(*values)
    categorical_ids, geometry, geometry_mask, padding_mask = values
    if categorical_ids.size(0) == 0:
        raise ValueError("source_batch must have a positive batch size")
    if geometry.dtype != model.bos.dtype:
        raise TypeError("source geometry dtype must match model dtype")
    if any(value.device != model.bos.device for value in values):
        raise ValueError("source tensors must share model device")
    if not torch.isfinite(geometry).all():
        raise ValueError("source geometry must be finite")
    for row in padding_mask:
        count = int(row.sum().item())
        if (
            count <= 0
            or not bool(row[:count].all().item())
            or bool(row[count:].any().item())
        ):
            raise ValueError(
                "source padding masks must be positive contiguous prefixes"
            )
    if geometry_mask.shape != geometry.shape:
        raise AssertionError("validated source geometry masks are misaligned")
    if categorical_ids.shape[:2] != padding_mask.shape:
        raise AssertionError("validated source categories are misaligned")


def _validate_encoded_memory(model, encoded):
    if not isinstance(encoded, V4EncodedMemory):
        raise TypeError("memory must be V4EncodedMemory")
    memory = encoded.memory
    indices = encoded.code_indices
    expected_memory = (
        indices.size(0) if torch.is_tensor(indices) and indices.dim() == 2 else 0,
        model.config.latent_tokens,
        model.config.model_dim,
    )
    if (
        not torch.is_tensor(memory)
        or tuple(memory.shape) != expected_memory
        or memory.size(0) == 0
    ):
        raise ValueError(
            "memory must have shape [B, latent_tokens, model_dim]"
        )
    if not memory.dtype.is_floating_point:
        raise TypeError("memory must be floating point")
    if memory.dtype != model.bos.dtype or memory.device != model.bos.device:
        raise ValueError("memory must match model dtype and device")
    if not torch.isfinite(memory).all():
        raise ValueError("memory must be finite")
    if (
        not torch.is_tensor(indices)
        or indices.dtype != torch.long
        or tuple(indices.shape)
        != (memory.size(0), model.config.latent_tokens)
        or indices.device != memory.device
    ):
        raise ValueError(
            "code_indices must have shape [B, latent_tokens]"
        )
    if indices.numel() and (
        int(indices.min().item()) < 0
        or int(indices.max().item()) >= model.config.codebook_size
    ):
        raise ValueError("code index is outside the configured codebook")
    if encoded.encoding_source != V4_ENCODED_MEMORY_SOURCE:
        raise ValueError("encoded memory has invalid provenance")
    if encoded.model_name != model.config.model_name:
        raise ValueError("encoded memory belongs to a different model identity")


def _validated_node_counts(
    node_counts, batch_size, max_nodes, device
):
    if not torch.is_tensor(node_counts):
        raise TypeError("node_counts must be a torch.Tensor")
    if node_counts.dtype != torch.long:
        raise TypeError("node_counts must use torch.long")
    if tuple(node_counts.shape) != (batch_size,):
        raise ValueError("node_counts must have shape [B]")
    if node_counts.device != device:
        raise ValueError("node_counts must share encoded-memory device")
    if node_counts.numel() and int(node_counts.min().item()) <= 0:
        raise ValueError("node_counts must be positive")
    if node_counts.numel() and int(node_counts.max().item()) > max_nodes:
        raise ValueError("node_count exceeds configured max_nodes")
    return tuple(int(value) for value in node_counts.detach().cpu().tolist())


def _validate_node_count_source(node_count_source):
    if not isinstance(node_count_source, str) or not node_count_source:
        raise ValueError("node_count_source must be a nonempty string")


def _validate_prefix_output(model, output, output_length, memory):
    if not isinstance(output, ConstrainedProfileV4PrefixOutput):
        raise TypeError("decode_prefix returned an invalid V4 output")
    expected = (memory.size(0), output_length)
    floats = (
        (
            "decoded_states",
            output.decoded_states,
            expected + (model.config.model_dim,),
        ),
        (
            "node_type_logits",
            output.node_type_logits,
            expected + (len(NODE_TYPES.tokens),),
        ),
        (
            "profile_family_logits",
            output.profile_family_logits,
            expected + (len(PROFILE_FAMILIES),),
        ),
        (
            "raw_profile_parameters",
            output.raw_profile_parameters,
            expected + (3,),
        ),
        (
            "remaining_geometry",
            output.remaining_geometry,
            expected + (REMAINING_GEOMETRY_WIDTH,),
        ),
    )
    for name, value, shape in floats:
        if not torch.is_tensor(value) or not value.dtype.is_floating_point:
            raise TypeError("{} must be floating point".format(name))
        if tuple(value.shape) != shape:
            raise ValueError("{} is misaligned".format(name))
        if value.dtype != memory.dtype or value.device != memory.device:
            raise ValueError("{} must match memory".format(name))
        if not torch.isfinite(value).all():
            raise ValueError("{} must be finite".format(name))
    if (
        not isinstance(output.categorical_logits, tuple)
        or len(output.categorical_logits) != len(_RETAINED_VOCABULARIES)
    ):
        raise ValueError("decode_prefix must return five categorical logits")
    for logits, vocabulary in zip(
        output.categorical_logits, _RETAINED_VOCABULARIES
    ):
        if (
            not torch.is_tensor(logits)
            or not logits.dtype.is_floating_point
            or tuple(logits.shape) != expected + (len(vocabulary.tokens),)
            or logits.dtype != memory.dtype
            or logits.device != memory.device
        ):
            raise ValueError("categorical logits are misaligned")
        if not torch.isfinite(logits).all():
            raise ValueError("categorical logits must be finite")


def _validate_relation_output(model, output, node_count, memory):
    expected = (1, node_count)
    fields = (
        (
            "edge_presence_logits",
            output.edge_presence_logits,
            expected + (node_count,),
        ),
        (
            "edge_type_logits",
            output.edge_type_logits,
            expected + (node_count, len(EDGE_TYPES.tokens)),
        ),
        (
            "operation_pointer_logits",
            output.operation_pointer_logits,
            (1, model.config.max_operations, node_count),
        ),
    )
    for name, value, shape in fields:
        if (
            not torch.is_tensor(value)
            or not value.dtype.is_floating_point
            or tuple(value.shape) != shape
            or value.dtype != memory.dtype
            or value.device != memory.device
        ):
            raise ValueError("{} is misaligned".format(name))
        if not torch.isfinite(value).all():
            raise ValueError("{} must be finite".format(name))
