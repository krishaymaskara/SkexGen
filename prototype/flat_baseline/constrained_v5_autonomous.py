"""Target-free predicted-history decoding for prefix-grammar V5."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping, Tuple

import torch

from prototype.node_grammar import (
    CONTROLLED_OPERATION_LIMIT,
    NODE_GRAMMAR_CONTRACT_ID,
    NodeGrammarError,
    legal_next_node_ids,
)

from .autonomous import RAW_PREFIX_FEEDBACK, REQUESTED_LENGTH_TERMINATION
from .constrained_v4_autonomous import (
    _evaluation_mode,
    _operation_nodes,
    _raw_edges,
    _raw_node_from_record,
    _raw_pointers,
    _validate_node_count_source,
    _validate_prefix_output,
    _validate_relation_output,
    _validate_source_batch,
    _validated_node_counts,
)
from .constrained_v4_conversion import _validate_v4_raw_contract
from .constrained_v5 import ConstrainedProfileV5Model
from .constrained_v5_conversion import (
    V5TeacherForcedRawPrediction,
    construct_v5_predicted_node_tensors,
    validate_v5_prediction_contract,
)
from .conversion import ConversionResult, validate_and_convert_raw_prediction


V5_ENCODED_MEMORY_SOURCE = "v5_encoder_quantized_memory"
V5_AUTONOMOUS_PREFIX_FEEDBACK = "v5_predicted_grammar_constrained_records"


class V5AutonomousEvaluationError(NodeGrammarError):
    def __init__(self, detail):
        super().__init__("invalid_v5_node_selection", detail)


@dataclass(frozen=True)
class V5EncodedMemory:
    memory: torch.Tensor
    code_indices: torch.Tensor
    encoding_source: str
    model_name: str


@dataclass(frozen=True)
class V5AutonomousRawPrediction(V5TeacherForcedRawPrediction):
    encoded_memory_source: str


def encode_v5_to_memory(model, source_batch):
    _validate_v5_model(model)
    _validate_source_batch(model, source_batch)
    with _evaluation_mode(model):
        encoded = model.encode_to_memory(
            source_batch["categorical_ids"],
            source_batch["geometry"],
            source_batch["geometry_mask"],
            source_batch["padding_mask"],
        )
    result = V5EncodedMemory(
        encoded.memory.detach().contiguous(),
        encoded.vq.indices.detach().contiguous(),
        V5_ENCODED_MEMORY_SOURCE,
        model.config.model_name,
    )
    _validate_encoded_memory(model, result)
    return result


def greedy_decode_v5_from_memory(
    model,
    memory,
    *,
    node_counts,
    node_count_source,
) -> Tuple[V5AutonomousRawPrediction, ...]:
    _validate_v5_model(model)
    _validate_encoded_memory(model, memory)
    counts = _validated_node_counts(
        node_counts,
        memory.memory.size(0),
        model.config.max_nodes,
        memory.memory.device,
    )
    _validate_node_count_source(node_count_source)
    for count in counts:
        legal_next_node_ids((), count)
    results = []
    with _evaluation_mode(model):
        for index, count in enumerate(counts):
            prediction = _decode_one(
                model,
                memory.memory[index:index + 1],
                memory.code_indices[index],
                count,
                node_count_source,
                memory.encoding_source,
            )
            validate_v5_autonomous_evaluation_result(
                prediction,
                requested_node_count=count,
                max_nodes=model.config.max_nodes,
            )
            results.append(prediction)
    return tuple(results)


def greedy_decode_v5(
    model,
    source_batch: Mapping,
    *,
    node_counts,
    node_count_source,
) -> Tuple[V5AutonomousRawPrediction, ...]:
    memory = encode_v5_to_memory(model, source_batch)
    return greedy_decode_v5_from_memory(
        model,
        memory,
        node_counts=node_counts,
        node_count_source=node_count_source,
    )


def validate_and_convert_v5_autonomous_prediction(
    prediction, *, max_operations
) -> ConversionResult:
    if max_operations != CONTROLLED_OPERATION_LIMIT:
        raise NodeGrammarError(
            "invalid_v5_grammar_contract",
            "strict V5 conversion requires the controlled operation limit",
        )
    if not isinstance(prediction, V5AutonomousRawPrediction):
        raise NodeGrammarError(
            "invalid_v5_node_selection", "prediction has the wrong V5 type"
        )
    if prediction.prefix_feedback != V5_AUTONOMOUS_PREFIX_FEEDBACK:
        raise NodeGrammarError(
            "invalid_v5_node_selection", "autonomous feedback provenance is wrong"
        )
    if prediction.encoded_memory_source != V5_ENCODED_MEMORY_SOURCE:
        raise NodeGrammarError(
            "invalid_v5_node_selection", "encoded memory provenance is wrong"
        )
    validate_v5_prediction_contract(
        prediction, require_complete_sequence=True
    )
    _validate_v4_raw_contract(prediction)
    return validate_and_convert_raw_prediction(
        replace(prediction, prefix_feedback=RAW_PREFIX_FEEDBACK),
        max_operations=max_operations,
    )


def validate_v5_autonomous_evaluation_result(
    prediction, *, requested_node_count=None, max_nodes=None
):
    if not isinstance(prediction, V5AutonomousRawPrediction):
        raise NodeGrammarError(
            "invalid_v5_node_selection", "result has the wrong V5 type"
        )
    if (
        isinstance(requested_node_count, bool)
        or not isinstance(requested_node_count, int)
        or isinstance(max_nodes, bool)
        or not isinstance(max_nodes, int)
        or not 1 <= requested_node_count <= max_nodes
    ):
        raise NodeGrammarError(
            "invalid_v5_requested_node_count", "request context is malformed"
        )
    if prediction.node_count != requested_node_count:
        raise NodeGrammarError(
            "invalid_v5_node_selection", "generated count differs from request"
        )
    validate_v5_prediction_contract(
        prediction, require_complete_sequence=True
    )
    return prediction


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
    geometry = memory.new_empty((1, 0, 39))
    geometry_mask = torch.empty((1, 0, 39), dtype=torch.bool, device=device)
    raw_nodes = []
    raw_shadow_nodes = []
    family_logits = []
    family_ids = []
    raw_parameters = []
    constrained_parameters = []
    raw_categorical_ids = []
    conditioned_categorical_ids = []
    categorical_corrections = []
    raw_node_ids = []
    constrained_node_ids = []
    node_corrections = []
    legal_masks = []
    state_evidence = []
    requested = torch.tensor([node_count], dtype=torch.long, device=device)

    for position in range(node_count):
        output = model.decode_prefix(memory, categories, geometry, geometry_mask)
        _validate_prefix_output(model, output, position + 1, memory)
        selected = construct_v5_predicted_node_tensors(
            output.node_type_logits[:, -1],
            (tuple(constrained_node_ids),),
            requested,
            tuple(logits[:, -1] for logits in output.categorical_logits),
            output.profile_family_logits[:, -1],
            output.raw_profile_parameters[:, -1],
            output.remaining_geometry[:, -1],
        )
        records = selected.records
        current_categories = torch.cat(
            (
                selected.grammar_constrained_node_type_ids.unsqueeze(-1),
                records.categorical_ids,
            ),
            dim=-1,
        ).unsqueeze(1)
        categories = torch.cat((categories, current_categories), dim=1).contiguous()
        geometry = torch.cat((geometry, records.geometry.unsqueeze(1)), dim=1).contiguous()
        geometry_mask = torch.cat(
            (geometry_mask, records.geometry_mask.unsqueeze(1)), dim=1
        ).contiguous()
        raw_nodes.append(_raw_node_from_record(records, position))
        raw_shadow_nodes.append(
            _raw_node_from_record(selected.raw_shadow_records, position)
        )
        family_logits.append(tuple(float(value) for value in output.profile_family_logits[0, -1].cpu().tolist()))
        family_ids.append(int(records.family_ids[0].item()))
        raw_parameters.append(tuple(float(value) for value in output.raw_profile_parameters[0, -1].cpu().tolist()))
        constrained_parameters.append(tuple(float(value) for value in records.constrained_parameters[0].cpu().tolist()))
        raw_categorical_ids.append(tuple(int(value) for value in records.raw_categorical_argmax_ids[0].cpu().tolist()))
        conditioned_categorical_ids.append(tuple(int(value) for value in records.node_conditioned_categorical_ids[0].cpu().tolist()))
        categorical_corrections.append(tuple(bool(value) for value in records.categorical_correction_mask[0].cpu().tolist()))
        raw_node_ids.append(int(selected.raw_node_type_argmax_ids[0].item()))
        constrained_node_ids.append(int(selected.grammar_constrained_node_type_ids[0].item()))
        node_corrections.append(bool(selected.node_type_correction_mask[0].item()))
        legal_masks.append(tuple(bool(value) for value in selected.legal_node_type_mask[0].cpu().tolist()))
        state_evidence.append(selected.grammar_state_evidence[0])

    aligned = model.decode_prefix(
        memory, categories[:, :-1], geometry[:, :-1], geometry_mask[:, :-1]
    )
    _validate_prefix_output(model, aligned, node_count, memory)
    node_mask = torch.ones((1, node_count), dtype=torch.bool, device=device)
    relations = model.decode_relations(aligned.decoded_states, node_mask)
    _validate_relation_output(model, relations, node_count, memory)
    edges = _raw_edges(relations, node_count)
    operation_nodes = _operation_nodes(raw_nodes)
    raw_shadow_operation_nodes = _operation_nodes(raw_shadow_nodes)
    pointers = _raw_pointers(
        relations, node_count, len(operation_nodes), model.config.max_operations
    )
    raw_shadow_pointers = _raw_pointers(
        relations,
        node_count,
        len(raw_shadow_operation_nodes),
        model.config.max_operations,
    )
    return V5AutonomousRawPrediction(
        tuple(int(value) for value in code_indices.cpu().tolist()),
        node_count,
        node_count_source,
        REQUESTED_LENGTH_TERMINATION,
        False,
        V5_AUTONOMOUS_PREFIX_FEEDBACK,
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
        model.config.categorical_selection_contract_id,
        tuple(raw_node_ids),
        tuple(constrained_node_ids),
        tuple(node_corrections),
        tuple(legal_masks),
        tuple(state_evidence),
        tuple(raw_shadow_nodes),
        raw_shadow_operation_nodes,
        len(raw_shadow_operation_nodes),
        len(raw_shadow_operation_nodes) > model.config.max_operations,
        raw_shadow_pointers,
        NODE_GRAMMAR_CONTRACT_ID,
        encoded_memory_source,
    )


def _validate_v5_model(model):
    if not isinstance(model, ConstrainedProfileV5Model):
        raise TypeError("model must be ConstrainedProfileV5Model")
    model.config.validate()


def _validate_encoded_memory(model, encoded):
    if not isinstance(encoded, V5EncodedMemory):
        raise TypeError("memory must be V5EncodedMemory")
    memory = encoded.memory
    indices = encoded.code_indices
    if (
        not torch.is_tensor(memory)
        or tuple(memory.shape[1:])
        != (model.config.latent_tokens, model.config.model_dim)
        or memory.size(0) == 0
        or memory.dtype != model.bos.dtype
        or memory.device != model.bos.device
        or not torch.isfinite(memory).all()
    ):
        raise ValueError("encoded V5 memory is invalid")
    if (
        not torch.is_tensor(indices)
        or indices.dtype != torch.long
        or tuple(indices.shape) != (memory.size(0), model.config.latent_tokens)
        or indices.device != memory.device
    ):
        raise ValueError("encoded V5 code indices are invalid")
    if indices.numel() and (
        int(indices.min().item()) < 0
        or int(indices.max().item()) >= model.config.codebook_size
    ):
        raise ValueError("encoded V5 code index is outside the codebook")
    if encoded.encoding_source != V5_ENCODED_MEMORY_SOURCE:
        raise ValueError("encoded V5 memory provenance is wrong")
    if encoded.model_name != model.config.model_name:
        raise ValueError("encoded V5 memory model identity is wrong")
