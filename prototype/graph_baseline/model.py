"""Graph V1 C1 with a shared pair decoder and directed position bias."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from prototype.constrained_profile_decoder import ConstrainedProfileHeads
from prototype.flat_baseline.constrained_v4 import (
    REMAINING_GEOMETRY_WIDTH,
    _REMAINING_CATEGORICAL_VOCABULARIES,
    _scatter_remaining_mask,
    scatter_remaining_geometry,
    select_remaining_geometry,
)
from prototype.flat_baseline.constrained_v6 import (
    ConstrainedProfileV6Model,
    ConstrainedProfileV6Output,
)
from prototype.flat_baseline.model import FlatMixedVQModel, RelationDecoderOutput
from prototype.model_data.vocab import EDGE_TYPES, NODE_TYPES
from prototype.node_grammar_torch import select_prefix_conditioned_node_types
from prototype.profile_geometry_torch import (
    canonicalize_profile_tensors,
    constrain_profile_parameters,
)
from prototype.reference_plane_geometry_torch import canonicalize_reference_plane_tensors

from .config import (
    GRAPH_PAIR_HIDDEN_DIM,
    GRAPH_POSITION_BIAS_RANK,
    GraphV1Config,
)
from .graph_contract import GRAPH_EDGE_CLASS_ORDER


@dataclass(frozen=True)
class GraphV1Output(ConstrainedProfileV6Output):
    graph_edge_logits: torch.Tensor
    graph_main_pair_logits: torch.Tensor
    graph_position_bias_logits: torch.Tensor
    graph_raw_node_type_argmax_ids: torch.Tensor
    authoritative_graph_node_type_ids: torch.Tensor
    graph_node_type_correction_mask: torch.Tensor
    graph_legal_node_type_mask: torch.Tensor
    graph_grammar_state_evidence: tuple


@dataclass(frozen=True)
class GraphNodeSelection:
    raw_node_type_argmax_ids: torch.Tensor
    authoritative_node_type_ids: torch.Tensor
    node_type_correction_mask: torch.Tensor
    legal_node_type_mask: torch.Tensor
    grammar_state_evidence: tuple


@dataclass(frozen=True)
class GraphEdgeLogitComponents:
    main_pair_logits: torch.Tensor
    position_bias_logits: torch.Tensor
    edge_logits: torch.Tensor


class GraphV1Model(ConstrainedProfileV6Model):
    """Replace all trainable flat structural heads with one pairwise classifier."""

    def __init__(self, config=None):
        resolved = config or GraphV1Config()
        if not isinstance(resolved, GraphV1Config):
            raise TypeError("graph model requires GraphV1Config")
        resolved.validate()
        FlatMixedVQModel.__init__(self, resolved)
        del self.categorical_heads
        del self.geometry_head
        del self.edge_source
        del self.edge_target
        del self.edge_presence_head
        del self.edge_type_head
        del self.operation_queries
        del self.operation_keys
        width = self.config.model_dim
        self.remaining_categorical_heads = nn.ModuleList(
            nn.Linear(width, len(vocabulary.tokens))
            for vocabulary in _REMAINING_CATEGORICAL_VOCABULARIES
        )
        self.profile_heads = ConstrainedProfileHeads(width)
        self.remaining_geometry_head = nn.Linear(width, REMAINING_GEOMETRY_WIDTH)
        self.graph_edge_decoder = nn.Sequential(
            nn.Linear(7 * width + 1, GRAPH_PAIR_HIDDEN_DIM),
            nn.GELU(),
            nn.Linear(GRAPH_PAIR_HIDDEN_DIM, len(GRAPH_EDGE_CLASS_ORDER)),
        )
        self.source_position_factor = nn.Embedding(
            self.config.max_nodes, GRAPH_POSITION_BIAS_RANK
        )
        self.destination_position_factor = nn.Embedding(
            self.config.max_nodes, GRAPH_POSITION_BIAS_RANK
        )
        self.position_class_projection = nn.Linear(
            GRAPH_POSITION_BIAS_RANK,
            len(GRAPH_EDGE_CLASS_ORDER),
            bias=False,
        )

    def forward(
        self, categorical_ids, geometry, geometry_mask, padding_mask,
        target, profile_targets,
    ):
        self._validate_inputs(
            categorical_ids, geometry, geometry_mask, padding_mask, target
        )
        self._validate_profile_targets(profile_targets, target)
        encoded = self.encode_to_memory(
            categorical_ids, geometry, geometry_mask, padding_mask
        )
        states = self._teacher_forced_states(
            categorical_ids, padding_mask, target, encoded.memory
        )
        node_logits = self.node_type_head(states)
        categorical_logits = tuple(head(states) for head in self.remaining_categorical_heads)
        profile_output = self.profile_heads(states)
        constrained_parameters = constrain_profile_parameters(
            profile_output.raw_parameters,
            profile_targets.family_ids,
            profile_targets.sketch_mask,
            extent_min=self.config.profile_extent_min,
            extent_max=self.config.profile_extent_max,
        )
        canonical_profile = canonicalize_profile_tensors(
            profile_targets.family_ids,
            constrained_parameters,
            profile_targets.sketch_mask,
            extent_min=self.config.profile_extent_min,
            extent_max=self.config.profile_extent_max,
        )
        remaining = torch.tanh(self.remaining_geometry_head(states))
        scattered = scatter_remaining_geometry(remaining)
        selected_mask = (
            select_remaining_geometry(target["geometry_mask"])
            & target["node_mask"].unsqueeze(-1)
        )
        scattered_mask = _scatter_remaining_mask(selected_mask)
        canonical_plane = canonicalize_reference_plane_tensors(
            target["node_type_ids"],
            target["categorical_attributes"][..., 3],
            remaining,
            target["node_mask"],
        )
        training_geometry = canonical_plane.geometry + canonical_profile.geometry + scattered
        training_mask = (
            canonical_plane.geometry_mask
            | canonical_profile.geometry_mask
            | scattered_mask
        )
        graph_nodes = select_complete_graph_node_sequence(
            node_logits, target["node_mask"]
        )
        graph_components = self.decode_graph_edge_components(
            states, graph_nodes.authoritative_node_type_ids,
            encoded.memory, target["node_mask"]
        )
        batch_size, node_count = target["node_mask"].shape
        neutral_presence = states.new_zeros(batch_size, node_count, node_count)
        neutral_types = states.new_zeros(
            batch_size, node_count, node_count, len(EDGE_TYPES.tokens)
        )
        neutral_pointers = states.new_zeros(
            batch_size, self.config.max_operations, node_count
        )
        prefix = target["node_type_ids"].clone()
        prefix[:, 1:] = target["node_type_ids"][:, :-1]
        prefix[:, 0] = NODE_TYPES.pad_id
        return GraphV1Output(
            states,
            node_logits,
            categorical_logits,
            profile_output.family_logits,
            profile_output.raw_parameters,
            constrained_parameters,
            remaining,
            scattered,
            selected_mask,
            canonical_plane.geometry,
            canonical_plane.geometry_mask,
            canonical_profile.primitive_type_ids,
            canonical_profile.geometry,
            canonical_profile.geometry_mask,
            training_geometry,
            training_mask,
            neutral_presence,
            neutral_types,
            neutral_pointers,
            encoded.memory,
            encoded.vq.loss,
            encoded.vq.per_example_loss,
            encoded.vq.indices,
            encoded.vq.assignment_counts,
            encoded.vq.active_code_count,
            encoded.vq.utilization,
            encoded.vq.perplexity,
            prefix.contiguous(),
            graph_components.edge_logits,
            graph_components.main_pair_logits,
            graph_components.position_bias_logits,
            graph_nodes.raw_node_type_argmax_ids,
            graph_nodes.authoritative_node_type_ids,
            graph_nodes.node_type_correction_mask,
            graph_nodes.legal_node_type_mask,
            graph_nodes.grammar_state_evidence,
        )

    def decode_graph_edges(self, states, node_type_ids, memory, node_mask):
        return self.decode_graph_edge_components(
            states, node_type_ids, memory, node_mask
        ).edge_logits

    def decode_graph_edge_components(
        self, states, node_type_ids, memory, node_mask
    ):
        if states.dim() != 3 or states.shape[:2] != node_type_ids.shape:
            raise ValueError("states and constrained node IDs must align")
        batch_size, count, width = states.shape
        if width != self.config.model_dim or node_mask.shape != (batch_size, count):
            raise ValueError("graph decoder inputs have invalid shape")
        positions = torch.arange(count, device=states.device)
        position_values = self.decoder_position_embedding(positions)
        position_values = position_values.unsqueeze(0).expand(batch_size, -1, -1)
        type_values = self.field_embeddings[0](node_type_ids)
        global_context = memory.mean(dim=1).unsqueeze(1).unsqueeze(1)
        global_context = global_context.expand(-1, count, count, -1)
        source_state = states.unsqueeze(2).expand(-1, -1, count, -1)
        destination_state = states.unsqueeze(1).expand(-1, count, -1, -1)
        source_type = type_values.unsqueeze(2).expand(-1, -1, count, -1)
        destination_type = type_values.unsqueeze(1).expand(-1, count, -1, -1)
        source_position = position_values.unsqueeze(2).expand(-1, -1, count, -1)
        destination_position = position_values.unsqueeze(1).expand(-1, count, -1, -1)
        relative = (
            positions.view(1, count, 1) - positions.view(1, 1, count)
        ).to(states.dtype) / float(max(self.config.max_nodes - 1, 1))
        relative = relative.unsqueeze(-1).expand(batch_size, -1, -1, -1)
        features = torch.cat((
            source_state, destination_state, source_type, destination_type,
            source_position, destination_position, global_context, relative,
        ), dim=-1)
        main_pair_logits = self.graph_edge_decoder(features)
        source_factors = self.source_position_factor(positions).view(
            1, count, 1, GRAPH_POSITION_BIAS_RANK
        )
        destination_factors = self.destination_position_factor(positions).view(
            1, 1, count, GRAPH_POSITION_BIAS_RANK
        )
        position_latent = source_factors * destination_factors
        position_bias_logits = self.position_class_projection(
            position_latent
        ).expand(batch_size, -1, -1, -1)
        return GraphEdgeLogitComponents(
            main_pair_logits.contiguous(),
            position_bias_logits.contiguous(),
            (main_pair_logits + position_bias_logits).contiguous(),
        )

    def decode_relations(self, decoded_states, node_mask):
        """Compatibility-only neutral output; never used as graph structure."""

        batch_size, count = node_mask.shape
        return RelationDecoderOutput(
            decoded_states.new_zeros(batch_size, count, count),
            decoded_states.new_zeros(
                batch_size, count, count, len(EDGE_TYPES.tokens)
            ),
            decoded_states.new_zeros(
                batch_size, self.config.max_operations, count
            ),
        )


def select_complete_graph_node_sequence(logits, node_mask):
    """Roll out one complete constrained sequence from logits and lengths."""

    batch_size, count = node_mask.shape
    requested = node_mask.long().sum(dim=1)
    constrained = torch.full(
        (batch_size, count), NODE_TYPES.id(None), dtype=torch.long,
        device=logits.device,
    )
    raw = torch.full_like(constrained, NODE_TYPES.id(None))
    corrections = torch.zeros_like(constrained, dtype=torch.bool)
    legal_masks = torch.zeros(
        batch_size, count, len(NODE_TYPES.tokens), dtype=torch.bool,
        device=logits.device,
    )
    evidence = []
    with torch.no_grad():
        for position in range(count):
            prefixes = tuple(
                tuple(int(value) for value in constrained[row, :position].tolist())
                if bool(node_mask[row, position].item()) else ()
                for row in range(batch_size)
            )
            selected = select_prefix_conditioned_node_types(
                logits[:, position], prefixes, requested,
                current_positions=torch.full(
                    (batch_size,), position, dtype=torch.long, device=logits.device
                ),
                active_mask=node_mask[:, position],
            )
            raw[:, position] = selected.raw_node_type_argmax_ids
            constrained[:, position] = selected.grammar_constrained_node_type_ids
            corrections[:, position] = selected.node_type_correction_mask
            legal_masks[:, position] = selected.legal_node_type_mask
            evidence.append(selected.grammar_state_evidence)
    return GraphNodeSelection(
        raw.contiguous(), constrained.contiguous(), corrections.contiguous(),
        legal_masks.contiguous(), tuple(evidence),
    )


def graph_decoder_parameter_count(model):
    return main_pair_mlp_parameter_count(model) + position_bias_parameter_count(model)


def main_pair_mlp_parameter_count(model):
    return sum(
        parameter.numel() for parameter in model.graph_edge_decoder.parameters()
    )


def position_bias_parameter_count(model):
    modules = (
        model.source_position_factor,
        model.destination_position_factor,
        model.position_class_projection,
    )
    return sum(
        parameter.numel()
        for module in modules for parameter in module.parameters()
    )
