"""Prediction evidence and independent strict conversion for graph V1."""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from prototype.axis_geometry import AXIS_GEOMETRY_CONTRACT_ID
from prototype.flat_baseline.autonomous import (
    RAW_PREFIX_FEEDBACK,
    RawDecodedEdge,
    RawDecodedPointer,
)
from prototype.flat_baseline.constrained_v6_conversion import (
    V6TeacherForcedRawPrediction,
    constrain_v6_node_records,
    validate_v6_prediction_contract,
)
from prototype.flat_baseline.constrained_v4_conversion import (
    _prediction_row,
    construct_v4_predicted_node_tensors,
)
from prototype.flat_baseline.conversion import validate_and_convert_raw_prediction
from prototype.model_data.vocab import EDGE_TYPES, NODE_TYPES
from prototype.node_grammar import NODE_GRAMMAR_CONTRACT_ID

from .graph_contract import (
    GRAPH_CONTRACT_VERSION,
    GRAPH_REPRESENTATION_NAME,
    GRAPH_TENSOR_VERSION,
    CanonicalGraphRecord,
    DirectedTypedEdge,
    GraphContractError,
    model_data_edge_id_from_graph_class,
    validate_graph_record,
)
from .graph_tensors import mask_graph_edge_logits
from .model import GraphV1Output


@dataclass(frozen=True)
class GraphV1Prediction:
    node_prediction: object
    graph: CanonicalGraphRecord
    raw_graph_edge_predictions: tuple
    masked_graph_edge_predictions: tuple
    graph_edge_correction_mask: tuple
    graph_contract_version: int
    raw_self_edge_prediction_count: int
    raw_inactive_node_edge_prediction_count: int
    graph_edge_correction_count: int


def graph_v1_teacher_forced_predictions(output, *, node_mask, node_count_source):
    if not isinstance(output, GraphV1Output):
        raise TypeError("output must be GraphV1Output")
    if (
        not torch.is_tensor(node_mask)
        or node_mask.dtype != torch.bool
        or tuple(node_mask.shape) != tuple(output.node_type_logits.shape[:2])
    ):
        raise TypeError("node_mask must be a Boolean tensor aligned with node logits")
    if not isinstance(node_count_source, str) or not node_count_source:
        raise ValueError("node_count_source must be a nonempty string")
    nodes = _graph_teacher_forced_node_predictions(
        output, node_mask=node_mask, node_count_source=node_count_source
    )
    masked = mask_graph_edge_logits(
        output.graph_edge_logits,
        output.authoritative_graph_node_type_ids,
        node_mask,
    )
    results = []
    for row, node_prediction in enumerate(nodes):
        count = node_prediction.node_count
        full_count = node_mask.size(1)
        raw_row = masked.raw_class_ids[row]
        diagonal = torch.eye(
            full_count, dtype=torch.bool, device=raw_row.device
        )
        inactive = ~(
            node_mask[row].unsqueeze(1) & node_mask[row].unsqueeze(0)
        ) & ~diagonal
        results.append(graph_prediction_from_evidence(
            node_prediction,
            masked.raw_class_ids[row, :count, :count].detach().cpu().tolist(),
            masked.masked_class_ids[row, :count, :count].detach().cpu().tolist(),
            masked.correction_mask[row, :count, :count].detach().cpu().tolist(),
            raw_self_edge_prediction_count=int(
                ((raw_row != 0) & diagonal).sum().item()
            ),
            raw_inactive_node_edge_prediction_count=int(
                ((raw_row != 0) & inactive).sum().item()
            ),
            graph_edge_correction_count=int(
                masked.correction_mask[row].sum().item()
            ),
        ))
    return tuple(results)


@torch.no_grad()
def _graph_teacher_forced_node_predictions(output, *, node_mask, node_count_source):
    """Build V6-compatible records from the single complete graph sequence."""

    constrained_records = construct_v4_predicted_node_tensors(
        output.authoritative_graph_node_type_ids,
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
        output.graph_raw_node_type_argmax_ids,
        output.categorical_logits,
        output.profile_family_logits,
        output.raw_profile_parameters,
        output.remaining_geometry,
        node_mask,
    )
    requested = node_mask.long().sum(dim=1)
    predictions = []
    for row in range(node_mask.size(0)):
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
                int(value) for value in output.graph_raw_node_type_argmax_ids[
                    row, :count
                ].cpu().tolist()
            ),
            grammar_constrained_node_type_ids=tuple(
                int(value) for value in output.authoritative_graph_node_type_ids[
                    row, :count
                ].cpu().tolist()
            ),
            node_type_correction_mask=tuple(
                bool(value) for value in output.graph_node_type_correction_mask[
                    row, :count
                ].cpu().tolist()
            ),
            legal_node_type_masks=tuple(
                tuple(bool(value) for value in mask)
                for mask in output.graph_legal_node_type_mask[
                    row, :count
                ].cpu().tolist()
            ),
            grammar_state_evidence=tuple(
                output.graph_grammar_state_evidence[position][row]
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


def graph_prediction_from_evidence(
    node_prediction, raw_classes, masked_classes, correction_mask,
    *, raw_self_edge_prediction_count=None,
    raw_inactive_node_edge_prediction_count=0,
    graph_edge_correction_count=None,
):
    count = node_prediction.node_count
    matrices = (raw_classes, masked_classes, correction_mask)
    if any(len(matrix) != count or any(len(row) != count for row in matrix)
           for matrix in matrices):
        raise GraphContractError("malformed_graph_prediction", "pair evidence width differs")
    node_ids = tuple(node.node_type_id for node in node_prediction.raw_nodes)
    edges = tuple(sorted(
        (
            DirectedTypedEdge(source, destination, int(masked_classes[source][destination]))
            for source in range(count)
            for destination in range(count)
            if int(masked_classes[source][destination]) != 0
        ),
        key=lambda item: (item.source, item.edge_type_id, item.destination),
    ))
    graph = CanonicalGraphRecord(
        GRAPH_REPRESENTATION_NAME,
        GRAPH_CONTRACT_VERSION,
        GRAPH_TENSOR_VERSION,
        node_ids,
        edges,
        (True,) * count,
        tuple(tuple(source != destination for destination in range(count))
              for source in range(count)),
        count,
    )
    validate_graph_record(graph)
    if raw_self_edge_prediction_count is None:
        raw_self_edge_prediction_count = sum(
            int(raw_classes[index][index] != 0) for index in range(count)
        )
    if graph_edge_correction_count is None:
        graph_edge_correction_count = sum(
            int(correction_mask[source][destination])
            for source in range(count) for destination in range(count)
        )
    return GraphV1Prediction(
        node_prediction,
        graph,
        tuple(tuple(int(value) for value in row) for row in raw_classes),
        tuple(tuple(int(value) for value in row) for row in masked_classes),
        tuple(tuple(bool(value) for value in row) for row in correction_mask),
        GRAPH_CONTRACT_VERSION,
        int(raw_self_edge_prediction_count),
        int(raw_inactive_node_edge_prediction_count),
        int(graph_edge_correction_count),
    )


def validate_and_convert_graph_prediction(prediction, *, max_operations=2):
    """Verify graph and V6 node contracts, then use existing CAD conversion."""

    if not isinstance(prediction, GraphV1Prediction):
        raise TypeError("prediction must be GraphV1Prediction")
    if prediction.graph_contract_version != GRAPH_CONTRACT_VERSION:
        raise GraphContractError("graph_contract_mismatch", "prediction version differs")
    validate_v6_prediction_contract(
        prediction.node_prediction, require_complete_sequence=True
    )
    validate_graph_record(prediction.graph)
    if tuple(node.node_type_id for node in prediction.node_prediction.raw_nodes) != (
        prediction.graph.node_type_ids
    ):
        raise GraphContractError("graph_node_mismatch", "prediction nodes differ")
    count = prediction.graph.node_count
    edge_by_pair = {
        (item.source, item.destination): item
        for item in prediction.graph.directed_typed_edges
    }
    raw_edges = []
    for source in range(count):
        for destination in range(count):
            edge = edge_by_pair.get((source, destination))
            raw_edges.append(RawDecodedEdge(
                source,
                destination,
                1.0 if edge is not None else -1.0,
                edge is not None,
                (
                    model_data_edge_id_from_graph_class(edge.edge_type_id)
                    if edge is not None else EDGE_TYPES.id(None)
                ),
            ))
    operations = tuple(
        index for index, node_id in enumerate(prediction.graph.node_type_ids)
        if NODE_TYPES.tokens[node_id] in ("extrude", "revolve")
    )
    pointers = tuple(
        RawDecodedPointer(index, node_index, 1.0)
        for index, node_index in enumerate(operations)
    )
    converted_input = replace(
        prediction.node_prediction,
        raw_edges=tuple(raw_edges),
        predicted_operation_node_indices=operations,
        predicted_operation_count=len(operations),
        operation_count_exceeds_limit=len(operations) > max_operations,
        raw_operation_pointers=pointers,
        prefix_feedback=RAW_PREFIX_FEEDBACK,
    )
    return validate_and_convert_raw_prediction(
        converted_input, max_operations=max_operations
    )
