"""Teacher-forced prediction conversion for prefix-grammar V6."""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from prototype.axis_geometry import (
    AXIS_GEOMETRY_CONTRACT_ID,
    CANONICAL_AXIS_CHANNELS,
    AxisGeometryContractError,
    validate_canonical_axis_geometry,
)
from prototype.axis_geometry_torch import constrain_axis_geometry_tensors
from prototype.model_data.vocab import NODE_TYPES
from prototype.node_grammar import (
    CONTROLLED_OPERATION_LIMIT,
    NODE_GRAMMAR_CONTRACT_ID,
    NodeGrammarError,
    V5_NODE_GRAMMAR,
    legal_next_node_ids,
    validate_complete_node_sequence,
)
from prototype.node_grammar_torch import select_prefix_conditioned_node_types

from .autonomous import RAW_PREFIX_FEEDBACK, TEACHER_FORCED_PREFIX_FEEDBACK
from .constrained_v4_conversion import (
    V4PredictedNodeTensors,
    V4TeacherForcedRawPrediction,
    _prediction_row,
    construct_v4_predicted_node_tensors,
    validate_and_convert_v4_teacher_forced_prediction,
)
from .constrained_v6 import ConstrainedProfileV6Output
from .conversion import ConversionResult, validate_and_convert_raw_prediction


@dataclass(frozen=True)
class V6PredictedNodeTensors:
    records: V4PredictedNodeTensors
    raw_shadow_records: V4PredictedNodeTensors
    raw_node_type_argmax_ids: torch.Tensor
    grammar_constrained_node_type_ids: torch.Tensor
    node_type_correction_mask: torch.Tensor
    legal_node_type_mask: torch.Tensor
    grammar_state_evidence: tuple
    raw_axis_geometry: torch.Tensor
    constrained_axis_geometry: torch.Tensor
    axis_geometry_correction_mask: torch.Tensor


@dataclass(frozen=True)
class V6TeacherForcedRawPrediction(V4TeacherForcedRawPrediction):
    raw_node_type_argmax_ids: tuple
    grammar_constrained_node_type_ids: tuple
    node_type_correction_mask: tuple
    legal_node_type_masks: tuple
    grammar_state_evidence: tuple
    same_history_raw_shadow_nodes: tuple
    same_history_raw_shadow_operation_node_indices: tuple
    same_history_raw_shadow_operation_count: int
    same_history_raw_shadow_operation_count_exceeds_limit: bool
    same_history_raw_shadow_operation_pointers: tuple
    node_grammar_contract_id: str
    raw_axis_geometry: tuple
    constrained_axis_geometry: tuple
    axis_geometry_correction_mask: tuple
    axis_geometry_contract_id: str


def constrain_v6_node_records(records, remaining_geometry, active_mask):
    """Apply the V6 axis contract after V5/V4 node and category selection."""

    axes = constrain_axis_geometry_tensors(
        records.node_type_ids,
        remaining_geometry,
        records.geometry,
        records.geometry_mask,
        active_mask,
    )
    return (
        replace(
            records,
            geometry=axes.geometry,
            geometry_mask=axes.geometry_mask,
        ),
        axes,
    )


def construct_v6_predicted_node_tensors(
    node_type_logits,
    generated_prefix_node_ids,
    requested_node_counts,
    categorical_logits,
    profile_family_logits,
    raw_profile_parameters,
    remaining_geometry,
    *,
    active_mask=None,
):
    selection = select_prefix_conditioned_node_types(
        node_type_logits,
        generated_prefix_node_ids,
        requested_node_counts,
        current_positions=torch.tensor(
            [len(prefix) for prefix in generated_prefix_node_ids],
            dtype=torch.long,
            device=node_type_logits.device,
        ),
        active_mask=active_mask,
    )
    constrained = construct_v4_predicted_node_tensors(
        selection.grammar_constrained_node_type_ids,
        categorical_logits,
        profile_family_logits,
        raw_profile_parameters,
        remaining_geometry,
        active_mask,
    )
    constrained, axes = constrain_v6_node_records(
        constrained, remaining_geometry, active_mask
    )
    raw_shadow = construct_v4_predicted_node_tensors(
        selection.raw_node_type_argmax_ids,
        categorical_logits,
        profile_family_logits,
        raw_profile_parameters,
        remaining_geometry,
        active_mask,
    )
    return V6PredictedNodeTensors(
        constrained,
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


def v6_teacher_forced_predictions(output, *, node_mask, node_count_source):
    """Report grammar-constrained current nodes from shifted prefixes only."""

    if not isinstance(output, ConstrainedProfileV6Output):
        raise TypeError("output must be ConstrainedProfileV6Output")
    if not torch.is_tensor(node_mask) or node_mask.dtype != torch.bool:
        raise TypeError("node_mask must be a Boolean tensor")
    if tuple(node_mask.shape) != tuple(output.node_type_logits.shape[:2]):
        raise ValueError("node_mask must align with V6 node logits")
    if not isinstance(node_count_source, str) or not node_count_source:
        raise ValueError("node_count_source must be a nonempty string")
    batch_size, maximum_nodes = node_mask.shape
    requested = node_mask.sum(dim=1).to(dtype=torch.long)
    constrained_ids = torch.full_like(
        requested[:, None].expand(-1, maximum_nodes), NODE_TYPES.id(None)
    )
    raw_ids = torch.full_like(constrained_ids, NODE_TYPES.id(None))
    corrections = torch.zeros_like(constrained_ids, dtype=torch.bool)
    legal_masks = torch.zeros(
        batch_size,
        maximum_nodes,
        output.node_type_logits.size(-1),
        dtype=torch.bool,
        device=node_mask.device,
    )
    evidence_by_position = []
    with torch.no_grad():
        for position in range(maximum_nodes):
            prefixes = tuple(
                tuple(
                    int(value)
                    for value in output.teacher_forced_prefix_node_ids[
                        row, 1:position + 1
                    ].detach().cpu().tolist()
                )
                if bool(node_mask[row, position].item()) else ()
                for row in range(batch_size)
            )
            selection = select_prefix_conditioned_node_types(
                output.node_type_logits[:, position],
                prefixes,
                requested,
                current_positions=torch.full(
                    (batch_size,),
                    position,
                    dtype=torch.long,
                    device=node_mask.device,
                ),
                active_mask=node_mask[:, position],
            )
            raw_ids[:, position] = selection.raw_node_type_argmax_ids
            constrained_ids[:, position] = (
                selection.grammar_constrained_node_type_ids
            )
            corrections[:, position] = selection.node_type_correction_mask
            legal_masks[:, position] = selection.legal_node_type_mask
            evidence_by_position.append(selection.grammar_state_evidence)

        constrained_records = construct_v4_predicted_node_tensors(
            constrained_ids,
            output.categorical_logits,
            output.profile_family_logits,
            output.raw_profile_parameters,
            output.remaining_geometry,
            node_mask,
        )
        constrained_records, axes = constrain_v6_node_records(
            constrained_records, output.remaining_geometry, node_mask
        )
        raw_records = construct_v4_predicted_node_tensors(
            raw_ids,
            output.categorical_logits,
            output.profile_family_logits,
            output.raw_profile_parameters,
            output.remaining_geometry,
            node_mask,
        )
        predictions = []
        for row in range(batch_size):
            base = _prediction_row(
                output, constrained_records, node_mask, row, node_count_source
            )
            raw_base = _prediction_row(
                output, raw_records, node_mask, row, node_count_source
            )
            count = int(requested[row].item())
            predictions.append(V6TeacherForcedRawPrediction(
                **vars(base),
                raw_node_type_argmax_ids=tuple(
                    int(value) for value in raw_ids[row, :count].cpu().tolist()
                ),
                grammar_constrained_node_type_ids=tuple(
                    int(value)
                    for value in constrained_ids[row, :count].cpu().tolist()
                ),
                node_type_correction_mask=tuple(
                    bool(value)
                    for value in corrections[row, :count].cpu().tolist()
                ),
                legal_node_type_masks=tuple(
                    tuple(bool(value) for value in mask)
                    for mask in legal_masks[row, :count].cpu().tolist()
                ),
                grammar_state_evidence=tuple(
                    evidence_by_position[position][row]
                    for position in range(count)
                ),
                same_history_raw_shadow_nodes=raw_base.raw_nodes,
                same_history_raw_shadow_operation_node_indices=(
                    raw_base.predicted_operation_node_indices
                ),
                same_history_raw_shadow_operation_count=(
                    raw_base.predicted_operation_count
                ),
                same_history_raw_shadow_operation_count_exceeds_limit=(
                    raw_base.operation_count_exceeds_limit
                ),
                same_history_raw_shadow_operation_pointers=(
                    raw_base.raw_operation_pointers
                ),
                node_grammar_contract_id=NODE_GRAMMAR_CONTRACT_ID,
                raw_axis_geometry=tuple(
                    tuple(float(value) for value in axis)
                    for axis in axes.raw_axis_geometry[row, :count].cpu().tolist()
                ),
                constrained_axis_geometry=tuple(
                    tuple(float(value) for value in axis)
                    for axis in axes.constrained_axis_geometry[
                        row, :count
                    ].cpu().tolist()
                ),
                axis_geometry_correction_mask=tuple(
                    tuple(bool(value) for value in mask)
                    for mask in axes.axis_geometry_correction_mask[
                        row, :count
                    ].cpu().tolist()
                ),
                axis_geometry_contract_id=AXIS_GEOMETRY_CONTRACT_ID,
            ))
        return tuple(predictions)


def validate_v6_prediction_contract(prediction, *, require_complete_sequence=False):
    if not isinstance(prediction, V6TeacherForcedRawPrediction):
        raise NodeGrammarError(
            "invalid_v6_node_selection", "prediction has the wrong V6 type"
        )
    if prediction.node_grammar_contract_id != NODE_GRAMMAR_CONTRACT_ID:
        raise NodeGrammarError(
            "invalid_v6_grammar_contract", "prediction grammar metadata is missing"
        )
    if prediction.axis_geometry_contract_id != AXIS_GEOMETRY_CONTRACT_ID:
        raise AxisGeometryContractError(
            "invalid_axis_geometry", "prediction axis contract metadata is missing"
        )
    count = prediction.node_count
    evidence = (
        prediction.raw_node_type_argmax_ids,
        prediction.grammar_constrained_node_type_ids,
        prediction.node_type_correction_mask,
        prediction.legal_node_type_masks,
        prediction.grammar_state_evidence,
        prediction.same_history_raw_shadow_nodes,
        prediction.raw_nodes,
        prediction.raw_axis_geometry,
        prediction.constrained_axis_geometry,
        prediction.axis_geometry_correction_mask,
    )
    if any(len(item) != count for item in evidence):
        raise NodeGrammarError(
            "invalid_v6_node_selection", "V6 node evidence is misaligned"
        )
    axis_id = NODE_TYPES.id("axis")
    for position, node in enumerate(prediction.raw_nodes):
        constrained_axis = prediction.constrained_axis_geometry[position]
        correction = prediction.axis_geometry_correction_mask[position]
        if len(prediction.raw_axis_geometry[position]) != 6 or len(
            constrained_axis
        ) != 6 or len(correction) != 6:
            raise AxisGeometryContractError(
                "invalid_axis_geometry", "axis evidence width differs from six"
            )
        if node.node_type_id == axis_id:
            validate_canonical_axis_geometry(
                node.normalized_geometry[33:39],
                node.derived_geometry_mask[33:39],
            )
            if tuple(constrained_axis) != CANONICAL_AXIS_CHANNELS:
                raise AxisGeometryContractError(
                    "invalid_axis_geometry", "constrained axis evidence differs"
                )
            expected_correction = tuple(
                index < 4 and raw != canonical
                for index, (raw, canonical) in enumerate(zip(
                    prediction.raw_axis_geometry[position],
                    CANONICAL_AXIS_CHANNELS,
                ))
            )
            if tuple(correction) != expected_correction:
                raise AxisGeometryContractError(
                    "invalid_axis_geometry", "axis correction evidence differs"
                )
        elif (
            tuple(constrained_axis) != (0.0,) * 6
            or any(correction)
        ):
            raise AxisGeometryContractError(
                "invalid_axis_geometry", "non-axis node carries constrained axis evidence"
            )
    if (
        prediction.same_history_raw_shadow_operation_count
        != len(prediction.same_history_raw_shadow_operation_node_indices)
        or prediction.same_history_raw_shadow_operation_count_exceeds_limit
        != (
            prediction.same_history_raw_shadow_operation_count
            > CONTROLLED_OPERATION_LIMIT
        )
    ):
        raise NodeGrammarError(
            "invalid_v6_node_selection",
            "same-history raw-shadow operation evidence is inconsistent",
        )
    constrained = tuple(node.node_type_id for node in prediction.raw_nodes)
    if constrained != prediction.grammar_constrained_node_type_ids:
        raise NodeGrammarError(
            "invalid_v6_node_selection", "generated nodes are not constrained IDs"
        )
    if tuple(
        raw != selected
        for raw, selected in zip(
            prediction.raw_node_type_argmax_ids,
            prediction.grammar_constrained_node_type_ids,
        )
    ) != prediction.node_type_correction_mask:
        raise NodeGrammarError(
            "invalid_v6_node_selection", "node correction evidence differs"
        )
    for selected, legal_mask in zip(
        prediction.grammar_constrained_node_type_ids,
        prediction.legal_node_type_masks,
    ):
        if (
            not isinstance(legal_mask, tuple)
            or len(legal_mask) != len(NODE_TYPES.tokens)
            or any(type(value) is not bool for value in legal_mask)
            or not legal_mask[selected]
        ):
            raise NodeGrammarError(
                "invalid_v6_node_selection",
                "selected node is absent from its legal mask",
            )
    if require_complete_sequence:
        validate_complete_node_sequence(constrained, count, V5_NODE_GRAMMAR)
        for position, selected in enumerate(constrained):
            legal = legal_next_node_ids(
                constrained[:position], count, V5_NODE_GRAMMAR
            )
            expected_mask = tuple(
                node_id in legal for node_id in range(len(NODE_TYPES.tokens))
            )
            if (
                selected not in legal
                or prediction.legal_node_type_masks[position] != expected_mask
            ):
                raise NodeGrammarError(
                    "invalid_v6_node_selection",
                    "stored legal mask differs from constrained-prefix grammar",
                )
    return prediction


def validate_and_convert_v6_teacher_forced_prediction(
    prediction, *, max_operations
) -> ConversionResult:
    if max_operations != CONTROLLED_OPERATION_LIMIT:
        raise NodeGrammarError(
            "invalid_v6_grammar_contract",
            "strict V6 conversion requires the controlled operation limit",
        )
    if not isinstance(prediction, V6TeacherForcedRawPrediction):
        raise NodeGrammarError(
            "invalid_v6_node_selection", "prediction has the wrong V6 type"
        )
    if prediction.prefix_feedback != TEACHER_FORCED_PREFIX_FEEDBACK:
        raise NodeGrammarError(
            "invalid_v6_node_selection", "teacher-forced provenance is wrong"
        )
    validate_v6_prediction_contract(
        prediction, require_complete_sequence=True
    )
    return validate_and_convert_v4_teacher_forced_prediction(
        prediction, max_operations=max_operations
    )


def same_history_raw_shadow_for_reporting(prediction):
    """Return the read-only per-step raw shadow; never use it for feedback."""

    validate_v6_prediction_contract(prediction)
    return replace(
        prediction,
        raw_nodes=prediction.same_history_raw_shadow_nodes,
        predicted_operation_node_indices=(
            prediction.same_history_raw_shadow_operation_node_indices
        ),
        predicted_operation_count=(
            prediction.same_history_raw_shadow_operation_count
        ),
        operation_count_exceeds_limit=(
            prediction.same_history_raw_shadow_operation_count_exceeds_limit
        ),
        raw_operation_pointers=(
            prediction.same_history_raw_shadow_operation_pointers
        ),
        prefix_feedback=RAW_PREFIX_FEEDBACK,
    )


def convert_same_history_raw_shadow(prediction, *, max_operations):
    shadow = same_history_raw_shadow_for_reporting(prediction)
    return validate_and_convert_raw_prediction(
        shadow, max_operations=max_operations
    )
