"""Target-free V6 node generation followed by graph-native edge prediction."""

from __future__ import annotations

from typing import Mapping

import torch

from prototype.flat_baseline.constrained_v6_autonomous import (
    encode_v6_to_memory,
    greedy_decode_v6_from_memory,
)

from .conversion import graph_prediction_from_evidence
from .graph_tensors import mask_graph_edge_logits
from .model import GraphV1Model


def greedy_decode_graph_v1(
    model,
    source_batch: Mapping,
    *,
    node_counts,
    node_count_source,
):
    """Decode constrained nodes and directed graph edges without targets."""

    if not isinstance(model, GraphV1Model):
        raise TypeError("model must be GraphV1Model")
    memory = encode_v6_to_memory(model, source_batch)
    node_predictions = greedy_decode_v6_from_memory(
        model,
        memory,
        node_counts=node_counts,
        node_count_source=node_count_source,
    )
    results = []
    for batch_index, node_prediction in enumerate(node_predictions):
        count = node_prediction.node_count
        prefix_nodes = node_prediction.raw_nodes[:-1]
        categories = torch.tensor(
            [[(node.node_type_id, *node.categorical_ids) for node in prefix_nodes]],
            dtype=torch.long,
            device=memory.memory.device,
        )
        geometry = memory.memory.new_tensor(
            [[node.normalized_geometry for node in prefix_nodes]]
        )
        geometry_mask = torch.tensor(
            [[node.derived_geometry_mask for node in prefix_nodes]],
            dtype=torch.bool,
            device=memory.memory.device,
        )
        prefix_output = model.decode_prefix(
            memory.memory[batch_index:batch_index + 1],
            categories,
            geometry,
            geometry_mask,
        )
        node_ids = torch.tensor(
            [[node.node_type_id for node in node_prediction.raw_nodes]],
            dtype=torch.long,
            device=memory.memory.device,
        )
        active = torch.ones((1, count), dtype=torch.bool, device=memory.memory.device)
        logits = model.decode_graph_edges(
            prefix_output.decoded_states,
            node_ids,
            memory.memory[batch_index:batch_index + 1],
            active,
        )
        masked = mask_graph_edge_logits(logits, node_ids, active)
        results.append(graph_prediction_from_evidence(
            node_prediction,
            masked.raw_class_ids[0].detach().cpu().tolist(),
            masked.masked_class_ids[0].detach().cpu().tolist(),
            masked.correction_mask[0].detach().cpu().tolist(),
        ))
    return tuple(results)
