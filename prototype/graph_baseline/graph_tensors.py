"""Dense ordered-pair tensorization and minimal graph-v1 masking."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from prototype.model_data.vocab import EDGE_TYPES, NODE_TYPES

from .graph_contract import (
    GRAPH_EDGE_CLASS_ORDER,
    edge_type_is_compatible,
    graph_class_from_model_data_edge_id,
)


@dataclass(frozen=True)
class GraphTargetTensors:
    edge_class_ids: torch.Tensor
    active_node_mask: torch.Tensor
    active_pair_mask: torch.Tensor
    node_counts: torch.Tensor


@dataclass(frozen=True)
class MaskedGraphPredictions:
    raw_class_ids: torch.Tensor
    masked_class_ids: torch.Tensor
    correction_mask: torch.Tensor
    allowed_class_mask: torch.Tensor
    active_pair_mask: torch.Tensor


def graph_targets_from_reconstruction_batch(target, maximum_nodes=None):
    """Create mutually exclusive graph classes with per-example dense pairs."""

    if not isinstance(target, dict) or not {
        "node_type_ids", "node_mask", "edge_index", "edge_type_ids", "edge_offsets"
    }.issubset(target):
        raise ValueError("target must come from ReconstructionBatch.to_torch()")
    node_mask = target["node_mask"]
    if node_mask.dtype != torch.bool or node_mask.dim() != 2:
        raise TypeError("node_mask must be a Boolean [B,N] tensor")
    batch_size, available_nodes = node_mask.shape
    node_count = available_nodes if maximum_nodes is None else maximum_nodes
    if node_count != available_nodes:
        raise ValueError("maximum_nodes must equal the padded target width")
    device = node_mask.device
    classes = torch.zeros(
        batch_size, node_count, node_count, dtype=torch.long, device=device
    )
    active_pairs = node_mask.unsqueeze(2) & node_mask.unsqueeze(1)
    diagonal = torch.eye(node_count, dtype=torch.bool, device=device).unsqueeze(0)
    active_pairs = (active_pairs & ~diagonal).contiguous()
    counts = node_mask.long().sum(dim=1)
    node_offsets = torch.cat((
        torch.zeros(1, dtype=torch.long, device=device), counts.cumsum(dim=0)
    ))
    edge_offsets = target["edge_offsets"]
    if tuple(edge_offsets.shape) != (batch_size + 1,):
        raise ValueError("edge_offsets must have B+1 entries")
    for batch_index in range(batch_size):
        start = int(edge_offsets[batch_index].item())
        stop = int(edge_offsets[batch_index + 1].item())
        if start < 0 or stop < start or stop > target["edge_type_ids"].numel():
            raise ValueError("edge_offsets must be bounded and monotonic")
        source = target["edge_index"][0, start:stop] - node_offsets[batch_index]
        destination = target["edge_index"][1, start:stop] - node_offsets[batch_index]
        current_count = int(counts[batch_index].item())
        if source.numel() and (
            int(source.min().item()) < 0
            or int(destination.min().item()) < 0
            or int(source.max().item()) >= current_count
            or int(destination.max().item()) >= current_count
        ):
            raise ValueError("edge_index disagrees with graph offsets")
        if source.numel() and (source == destination).any():
            raise ValueError("self edges are forbidden")
        pairs = set()
        for local, (source_id, destination_id) in enumerate(zip(
            source.detach().cpu().tolist(), destination.detach().cpu().tolist()
        )):
            pair = (int(source_id), int(destination_id))
            if pair in pairs:
                raise ValueError("multiple graph edges on one ordered pair")
            pairs.add(pair)
            existing_id = int(target["edge_type_ids"][start + local].item())
            if existing_id in (EDGE_TYPES.pad_id, EDGE_TYPES.id(None)):
                raise ValueError("present target edge cannot use a sentinel")
            class_id = graph_class_from_model_data_edge_id(existing_id)
            source_node = int(target["node_type_ids"][batch_index, pair[0]].item())
            destination_node = int(
                target["node_type_ids"][batch_index, pair[1]].item()
            )
            if not edge_type_is_compatible(source_node, destination_node, class_id):
                raise ValueError("target edge violates node-type compatibility")
            classes[batch_index, pair[0], pair[1]] = class_id
    return GraphTargetTensors(
        classes.contiguous(), node_mask.contiguous(), active_pairs,
        counts.contiguous()
    )


def graph_type_allowed_mask(node_type_ids, active_node_mask):
    if (
        node_type_ids.dtype != torch.long
        or active_node_mask.dtype != torch.bool
        or tuple(node_type_ids.shape) != tuple(active_node_mask.shape)
    ):
        raise ValueError("node IDs and active mask must align as [B,N]")
    if node_type_ids.numel() and (
        int(node_type_ids.min().item()) < 0
        or int(node_type_ids.max().item()) >= len(NODE_TYPES.tokens)
    ):
        raise ValueError("node type ID is unsupported")
    batch_size, count = node_type_ids.shape
    result = torch.zeros(
        batch_size, count, count, len(GRAPH_EDGE_CLASS_ORDER),
        dtype=torch.bool, device=node_type_ids.device,
    )
    active_pairs = active_node_mask.unsqueeze(2) & active_node_mask.unsqueeze(1)
    diagonal = torch.eye(count, dtype=torch.bool, device=node_type_ids.device)
    active_pairs = active_pairs & ~diagonal.unsqueeze(0)
    result[..., 0] = active_pairs
    for source in range(count):
        for destination in range(count):
            pair_active = active_pairs[:, source, destination]
            if not pair_active.any():
                continue
            for class_id in range(1, len(GRAPH_EDGE_CLASS_ORDER)):
                allowed_rows = []
                for row in range(batch_size):
                    allowed_rows.append(
                        bool(pair_active[row].item())
                        and edge_type_is_compatible(
                            int(node_type_ids[row, source].item()),
                            int(node_type_ids[row, destination].item()),
                            class_id,
                        )
                    )
                result[:, source, destination, class_id] = torch.tensor(
                    allowed_rows, dtype=torch.bool, device=node_type_ids.device
                )
    return result.contiguous(), active_pairs.contiguous()


def mask_graph_edge_logits(logits, node_type_ids, active_node_mask):
    if (
        not torch.is_tensor(logits)
        or not logits.dtype.is_floating_point
        or logits.dim() != 4
        or logits.size(-1) != len(GRAPH_EDGE_CLASS_ORDER)
        or tuple(logits.shape[:3])
        != tuple(node_type_ids.shape) + (node_type_ids.size(1),)
        or logits.device != node_type_ids.device
        or not torch.isfinite(logits).all()
    ):
        raise ValueError("graph logits must be finite [B,N,N,C]")
    allowed, active_pairs = graph_type_allowed_mask(
        node_type_ids, active_node_mask
    )
    minimum = torch.finfo(logits.dtype).min
    masked_logits = logits.masked_fill(~allowed, minimum)
    raw = logits.argmax(dim=-1)
    masked = masked_logits.argmax(dim=-1)
    masked = torch.where(active_pairs, masked, torch.zeros_like(masked))
    correction = (raw != masked).contiguous()
    return MaskedGraphPredictions(
        raw.contiguous(), masked.contiguous(), correction, allowed, active_pairs
    )
