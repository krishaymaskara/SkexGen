"""GE1-owned `<pad>`-terminated autonomous node generation.

Derived from ``prototype/flat_baseline/constrained_v6_autonomous.py`` at
commit 12521f64c73bf5f818156fe62722e64a0537d59c.  The frozen source remains the
sole legacy path; this copy removes target-count and transition masking only
for ADR-0016's opt-in identity.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping, Tuple

import torch

from prototype.axis_geometry import AXIS_GEOMETRY_CONTRACT_ID
from prototype.model_data.vocab import NODE_TYPES
from prototype.node_grammar import (
    CONTROLLED_OPERATION_LIMIT,
    NODE_GRAMMAR_CONTRACT_ID,
    NodeGrammarError,
)
from prototype.node_grammar_torch import (
    AUTONOMOUS_STOP_NODE_GENERATION_IDENTITY,
    select_prefix_conditioned_node_types,
)

from prototype.flat_baseline.autonomous import RAW_PREFIX_FEEDBACK
from prototype.flat_baseline.constrained_v4_autonomous import (
    _evaluation_mode,
    _operation_nodes,
    _raw_edges,
    _raw_node_from_record,
    _raw_pointers,
    _validate_prefix_output,
    _validate_relation_output,
)
from prototype.flat_baseline.constrained_v4_conversion import (
    _validate_v4_raw_contract,
    construct_v4_predicted_node_tensors,
)
from prototype.flat_baseline.constrained_v6 import ConstrainedProfileV6Model
from prototype.flat_baseline.constrained_v6_conversion import (
    V6PredictedNodeTensors,
    V6TeacherForcedRawPrediction,
    constrain_v6_node_records,
    validate_v6_prediction_contract,
)
from prototype.flat_baseline.conversion import (
    ConversionResult,
    validate_and_convert_raw_prediction,
)


V6_ENCODED_MEMORY_SOURCE = "v6_encoder_quantized_memory"
V6_AUTONOMOUS_PREFIX_FEEDBACK = "v6_predicted_grammar_constrained_records"
AUTONOMOUS_STOP_PREFIX_FEEDBACK = "ge1_unconstrained_argmax_records"
LEARNED_PAD_TERMINATION = "learned_pad_node_type"
GENERATION_CAP_TERMINATION = "max_nodes_reached_without_pad"


class V6AutonomousEvaluationError(NodeGrammarError):
    def __init__(self, detail):
        super().__init__("invalid_v6_node_selection", detail)


@dataclass(frozen=True)
class V6EncodedMemory:
    memory: torch.Tensor
    code_indices: torch.Tensor
    encoding_source: str
    model_name: str


@dataclass(frozen=True)
class V6AutonomousRawPrediction(V6TeacherForcedRawPrediction):
    encoded_memory_source: str


@dataclass(frozen=True)
class AutonomousStopRawPrediction(V6AutonomousRawPrediction):
    terminator_node_type_id: object
    terminator_position: object
    generation_cap_reached: bool


def greedy_decode_from_memory(
    model, memory
) -> Tuple[AutonomousStopRawPrediction, ...]:
    """Decode each memory row until `<pad>` or the configured cap."""

    _validate_v6_model(model)
    _validate_encoded_memory(model, memory)
    results = []
    with _evaluation_mode(model):
        for index in range(memory.memory.size(0)):
            prediction = _decode_one(
                model,
                memory.memory[index:index + 1],
                memory.code_indices[index],
                memory.encoding_source,
            )
            validate_autonomous_stop_evaluation_result(
                prediction, max_nodes=model.config.max_nodes
            )
            results.append(prediction)
    return tuple(results)


def validate_and_convert_autonomous_stop_prediction(
    prediction, *, max_operations
) -> ConversionResult:
    if max_operations != CONTROLLED_OPERATION_LIMIT:
        raise NodeGrammarError(
            "invalid_v6_grammar_contract",
            "strict V6 conversion requires the controlled operation limit",
        )
    if not isinstance(prediction, AutonomousStopRawPrediction):
        raise NodeGrammarError(
            "invalid_v6_node_selection", "prediction has the wrong V6 type"
        )
    if prediction.prefix_feedback != AUTONOMOUS_STOP_PREFIX_FEEDBACK:
        raise NodeGrammarError(
            "invalid_v6_node_selection", "autonomous feedback provenance is wrong"
        )
    if prediction.encoded_memory_source != V6_ENCODED_MEMORY_SOURCE:
        raise NodeGrammarError(
            "invalid_v6_node_selection", "encoded memory provenance is wrong"
        )
    if prediction.generation_cap_reached:
        raise NodeGrammarError(
            "invalid_v6_node_selection",
            "generation cap is an explicit failed conversion outcome",
        )
    validate_v6_prediction_contract(prediction, require_complete_sequence=True)
    _validate_v4_raw_contract(prediction)
    return validate_and_convert_raw_prediction(
        replace(prediction, prefix_feedback=RAW_PREFIX_FEEDBACK),
        max_operations=max_operations,
    )


def validate_autonomous_stop_evaluation_result(prediction, *, max_nodes=None):
    if not isinstance(prediction, AutonomousStopRawPrediction):
        raise NodeGrammarError(
            "invalid_v6_node_selection", "result has the wrong V6 type"
        )
    if (
        isinstance(max_nodes, bool)
        or not isinstance(max_nodes, int)
        or max_nodes <= 0
    ):
        raise NodeGrammarError(
            "invalid_v6_requested_node_count", "request context is malformed"
        )
    count = prediction.node_count
    if isinstance(count, bool) or not isinstance(count, int) or not 0 <= count <= max_nodes:
        raise NodeGrammarError(
            "invalid_v6_node_selection", "generated count is outside the cap"
        )
    cap = prediction.generation_cap_reached
    if (
        cap
        and (
            count != max_nodes
            or prediction.terminator_node_type_id is not None
            or prediction.terminator_position is not None
            or prediction.termination_reason != GENERATION_CAP_TERMINATION
            or prediction.termination_is_learned is not False
        )
    ):
        raise NodeGrammarError(
            "invalid_v6_node_selection", "cap termination evidence differs"
        )
    if (
        not cap
        and (
            prediction.terminator_node_type_id != NODE_TYPES.pad_id
            or prediction.terminator_position != count
            or count >= max_nodes
            or prediction.termination_reason != LEARNED_PAD_TERMINATION
            or prediction.termination_is_learned is not True
        )
    ):
        raise NodeGrammarError(
            "invalid_v6_node_selection", "learned termination evidence differs"
        )
    evidence = (
        prediction.raw_nodes,
        prediction.raw_node_type_argmax_ids,
        prediction.grammar_constrained_node_type_ids,
        prediction.node_type_correction_mask,
        prediction.legal_node_type_masks,
        prediction.grammar_state_evidence,
    )
    if any(len(item) != count for item in evidence):
        raise NodeGrammarError(
            "invalid_v6_node_selection", "node evidence is misaligned"
        )
    if (
        prediction.raw_node_type_argmax_ids
        != prediction.grammar_constrained_node_type_ids
        or any(prediction.node_type_correction_mask)
        or any(
            not all(mask) or len(mask) != len(NODE_TYPES.tokens)
            for mask in prediction.legal_node_type_masks
        )
    ):
        raise NodeGrammarError(
            "invalid_v6_node_selection", "unconstrained selection was corrected"
        )
    if count:
        validate_v6_prediction_contract(
            prediction, require_complete_sequence=False
        )
    return prediction


# Retain the copied validator name as a compatibility/audit alias, but its
# signature deliberately has no requested node count.
validate_v6_autonomous_evaluation_result = (
    validate_autonomous_stop_evaluation_result
)


def _construct_unconstrained_predicted_node_tensors(
    selection,
    categorical_logits,
    profile_family_logits,
    raw_profile_parameters,
    remaining_geometry,
):
    """Reuse frozen V4/category and V6/axis record construction after argmax."""

    records = construct_v4_predicted_node_tensors(
        selection.grammar_constrained_node_type_ids,
        categorical_logits,
        profile_family_logits,
        raw_profile_parameters,
        remaining_geometry,
        None,
    )
    records, axes = constrain_v6_node_records(
        records, remaining_geometry, None
    )
    raw_shadow = construct_v4_predicted_node_tensors(
        selection.raw_node_type_argmax_ids,
        categorical_logits,
        profile_family_logits,
        raw_profile_parameters,
        remaining_geometry,
        None,
    )
    return V6PredictedNodeTensors(
        records,
        raw_shadow,
        selection.raw_node_type_argmax_ids,
        selection.grammar_constrained_node_type_ids,
        selection.node_type_correction_mask,
        selection.legal_node_type_mask,
        selection.grammar_state_evidence,
        axes.raw_axis_geometry,
        axes.constrained_axis_geometry,
        axes.axis_geometry_correction_mask,
    )


def _decode_one(
    model,
    memory,
    code_indices,
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
    raw_axis_geometry = []
    constrained_axis_geometry = []
    axis_geometry_corrections = []
    terminator_node_type_id = None
    terminator_position = None
    for position in range(model.config.max_nodes):
        output = model.decode_prefix(memory, categories, geometry, geometry_mask)
        _validate_prefix_output(model, output, position + 1, memory)
        selection = select_prefix_conditioned_node_types(
            output.node_type_logits[:, -1],
            (tuple(constrained_node_ids),),
            None,
            current_positions=torch.tensor(
                (position,), dtype=torch.long, device=device
            ),
            node_generation_identity=(
                AUTONOMOUS_STOP_NODE_GENERATION_IDENTITY
            ),
        )
        selected_id = int(
            selection.grammar_constrained_node_type_ids[0].item()
        )
        if selected_id == NODE_TYPES.pad_id:
            terminator_node_type_id = selected_id
            terminator_position = position
            break
        selected = _construct_unconstrained_predicted_node_tensors(
            selection,
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
        raw_axis_geometry.append(tuple(
            float(value)
            for value in selected.raw_axis_geometry[0].cpu().tolist()
        ))
        constrained_axis_geometry.append(tuple(
            float(value)
            for value in selected.constrained_axis_geometry[0].cpu().tolist()
        ))
        axis_geometry_corrections.append(tuple(
            bool(value)
            for value in selected.axis_geometry_correction_mask[0].cpu().tolist()
        ))

    node_count = len(raw_nodes)
    generation_cap_reached = terminator_node_type_id is None
    if node_count:
        aligned = model.decode_prefix(
            memory, categories[:, :-1], geometry[:, :-1], geometry_mask[:, :-1]
        )
        _validate_prefix_output(model, aligned, node_count, memory)
        node_mask = torch.ones((1, node_count), dtype=torch.bool, device=device)
        relations = model.decode_relations(aligned.decoded_states, node_mask)
        _validate_relation_output(model, relations, node_count, memory)
        edges = _raw_edges(relations, node_count)
    else:
        relations = None
        edges = ()
    operation_nodes = _operation_nodes(raw_nodes)
    raw_shadow_operation_nodes = _operation_nodes(raw_shadow_nodes)
    pointers = (
        _raw_pointers(
            relations, node_count, len(operation_nodes),
            model.config.max_operations
        )
        if relations is not None else ()
    )
    raw_shadow_pointers = (
        _raw_pointers(
            relations,
            node_count,
            len(raw_shadow_operation_nodes),
            model.config.max_operations,
        )
        if relations is not None else ()
    )
    return AutonomousStopRawPrediction(
        tuple(int(value) for value in code_indices.cpu().tolist()),
        node_count,
        "autonomous_stop_symbol",
        (
            GENERATION_CAP_TERMINATION
            if generation_cap_reached else LEARNED_PAD_TERMINATION
        ),
        not generation_cap_reached,
        AUTONOMOUS_STOP_PREFIX_FEEDBACK,
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
        tuple(raw_axis_geometry),
        tuple(constrained_axis_geometry),
        tuple(axis_geometry_corrections),
        AXIS_GEOMETRY_CONTRACT_ID,
        encoded_memory_source,
        terminator_node_type_id,
        terminator_position,
        generation_cap_reached,
    )


def _validate_v6_model(model):
    if not isinstance(model, ConstrainedProfileV6Model):
        raise TypeError("model must be ConstrainedProfileV6Model")
    model.config.validate()


def _validate_encoded_memory(model, encoded):
    if not isinstance(encoded, V6EncodedMemory):
        raise TypeError("memory must be V6EncodedMemory")
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
        raise ValueError("encoded V6 memory is invalid")
    if (
        not torch.is_tensor(indices)
        or indices.dtype != torch.long
        or tuple(indices.shape) != (memory.size(0), model.config.latent_tokens)
        or indices.device != memory.device
    ):
        raise ValueError("encoded V6 code indices are invalid")
    if indices.numel() and (
        int(indices.min().item()) < 0
        or int(indices.max().item()) >= model.config.codebook_size
    ):
        raise ValueError("encoded V6 code index is outside the codebook")
    if encoded.encoding_source != V6_ENCODED_MEMORY_SOURCE:
        raise ValueError("encoded V6 memory provenance is wrong")
    if encoded.model_name != model.config.model_name:
        raise ValueError("encoded V6 memory model identity is wrong")
