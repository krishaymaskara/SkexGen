"""Sparse, position-free typed relational message passing for GE1 C4."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from prototype.model_data.vocab import EDGE_TYPES


SEMANTIC_EDGE_TYPE_IDS = tuple(
    EDGE_TYPES.id(name)
    for name in (
        "placed_on",
        "defined_in",
        "uses_profile",
        "uses_axis",
        "depends_on",
    )
)
SEMANTIC_RELATION_COUNT = len(SEMANTIC_EDGE_TYPE_IDS)
DIRECTED_RELATION_CHANNELS = 2 * SEMANTIC_RELATION_COUNT


class BasisRelationalLayer(nn.Module):
    """One residual layer with ten basis-decomposed message channels.

    Stored edges keep their dependent-to-reference orientation. Channels
    ``0..4`` send along that stored orientation; channels ``5..9`` send the
    same five semantic relations in reverse. Aggregation computes one mean per
    receiver and directed channel, then sums those means without an additional
    active-channel normalization.
    """

    def __init__(self, model_dim, basis_count=2, dropout=0.0):
        super().__init__()
        if model_dim <= 0:
            raise ValueError("model_dim must be positive")
        if basis_count != 2:
            raise ValueError("C4 requires exactly two relation bases")
        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.model_dim = int(model_dim)
        self.basis_count = int(basis_count)
        self.relation_bases = nn.Parameter(
            torch.empty(self.basis_count, self.model_dim, self.model_dim)
        )
        self.relation_mixing = nn.Parameter(
            torch.empty(DIRECTED_RELATION_CHANNELS, self.basis_count)
        )
        self.self_projection = nn.Linear(self.model_dim, self.model_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(self.model_dim)

        nn.init.normal_(
            self.relation_bases,
            mean=0.0,
            std=1.0 / math.sqrt(float(self.model_dim)),
        )
        nn.init.xavier_uniform_(self.relation_mixing)

        lookup = torch.full(
            (len(EDGE_TYPES.tokens),),
            -1,
            dtype=torch.long,
        )
        for channel, edge_type_id in enumerate(SEMANTIC_EDGE_TYPE_IDS):
            lookup[edge_type_id] = channel
        self.register_buffer("edge_type_to_channel", lookup)

    def relation_transforms(self):
        """Return the ten transforms in the span of the two learned bases."""

        return torch.einsum(
            "cb,bij->cij", self.relation_mixing, self.relation_bases
        )

    def aggregate_messages(self, nodes, edge_index, edge_type_ids):
        """Return the sum of per-receiver, per-channel message means."""

        self._validate_inputs(nodes, edge_index, edge_type_ids)
        node_count = nodes.size(0)
        aggregated = nodes.new_zeros((node_count, self.model_dim))
        if edge_type_ids.numel() == 0:
            return aggregated

        base_channels = self.edge_type_to_channel[edge_type_ids]
        if bool((base_channels < 0).any().item()):
            raise ValueError("edge_type_ids must name one of five semantic relations")

        stored_sources = edge_index[0]
        stored_destinations = edge_index[1]
        senders = torch.cat((stored_sources, stored_destinations), dim=0)
        receivers = torch.cat((stored_destinations, stored_sources), dim=0)
        channels = torch.cat(
            (base_channels, base_channels + SEMANTIC_RELATION_COUNT),
            dim=0,
        )
        transforms = self.relation_transforms()

        for channel in range(DIRECTED_RELATION_CHANNELS):
            selected = channels == channel
            if not bool(selected.any().item()):
                continue
            channel_senders = senders[selected]
            channel_receivers = receivers[selected]
            messages = F.linear(nodes[channel_senders], transforms[channel])
            channel_sum = nodes.new_zeros((node_count, self.model_dim))
            channel_sum.index_add_(0, channel_receivers, messages)
            channel_degree = nodes.new_zeros((node_count, 1))
            channel_degree.index_add_(
                0,
                channel_receivers,
                nodes.new_ones((channel_receivers.numel(), 1)),
            )
            channel_mean = channel_sum / channel_degree.clamp_min(1.0)
            aggregated = aggregated + channel_mean
        return aggregated

    def forward(self, nodes, edge_index, edge_type_ids):
        messages = self.self_projection(nodes)
        messages = messages + self.aggregate_messages(
            nodes, edge_index, edge_type_ids
        )
        update = self.dropout(F.gelu(messages))
        return self.norm(nodes + update)

    def _validate_inputs(self, nodes, edge_index, edge_type_ids):
        if nodes.dim() != 2 or nodes.size(1) != self.model_dim:
            raise ValueError("nodes must have shape [N, model_dim]")
        if not nodes.dtype.is_floating_point:
            raise TypeError("nodes must be floating point")
        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            raise ValueError("edge_index must have shape [2, E]")
        if edge_index.dtype != torch.long:
            raise TypeError("edge_index must use torch.long")
        if edge_type_ids.dim() != 1 or edge_type_ids.size(0) != edge_index.size(1):
            raise ValueError("edge_type_ids must align with edge_index")
        if edge_type_ids.dtype != torch.long:
            raise TypeError("edge_type_ids must use torch.long")
        if (
            nodes.device != edge_index.device
            or nodes.device != edge_type_ids.device
        ):
            raise ValueError("nodes and edges must share one device")
        if edge_index.numel() and (
            int(edge_index.min().item()) < 0
            or int(edge_index.max().item()) >= nodes.size(0)
        ):
            raise ValueError("edge endpoint is outside the node tensor")
        if edge_type_ids.numel() and (
            int(edge_type_ids.min().item()) < 0
            or int(edge_type_ids.max().item()) >= self.edge_type_to_channel.numel()
        ):
            raise ValueError("edge type ID is outside the vocabulary")


class RelationalEncoderCore(nn.Module):
    """Exactly three position-free basis-decomposed relational layers."""

    def __init__(
        self,
        model_dim,
        layer_count=3,
        basis_count=2,
        dropout=0.0,
    ):
        super().__init__()
        if layer_count != 3:
            raise ValueError("C4 requires exactly three relational layers")
        self.layers = nn.ModuleList(
            BasisRelationalLayer(model_dim, basis_count, dropout)
            for _ in range(layer_count)
        )

    def forward(self, nodes, edge_index, edge_type_ids):
        result = nodes
        for layer in self.layers:
            result = layer(result, edge_index, edge_type_ids)
        return result
