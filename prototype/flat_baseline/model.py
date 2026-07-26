"""Flat mixed VQ encoder and teacher-forced CAD-history decoder."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from prototype.model_data.geometry import GEOMETRY_WIDTH
from prototype.model_data.vocab import (
    BOOLEAN_MODES,
    DIRECTIONS,
    EDGE_TYPES,
    LOOP_ROLES,
    NODE_TYPES,
    OPERATION_TYPES,
    PRIMITIVE_TYPES,
    REFERENCE_PLANES,
)

from .config import FlatBaselineConfig
from .vq import EMAVectorQuantizer


_INPUT_VOCABULARIES = (
    NODE_TYPES,
    OPERATION_TYPES,
    BOOLEAN_MODES,
    DIRECTIONS,
    REFERENCE_PLANES,
    PRIMITIVE_TYPES,
    PRIMITIVE_TYPES,
    PRIMITIVE_TYPES,
    PRIMITIVE_TYPES,
    LOOP_ROLES,
)
_ATTRIBUTE_VOCABULARIES = _INPUT_VOCABULARIES[1:]


@dataclass
class FlatMixedVQOutput:
    decoded_states: torch.Tensor
    node_type_logits: torch.Tensor
    categorical_logits: tuple
    geometry: torch.Tensor
    edge_presence_logits: torch.Tensor
    edge_type_logits: torch.Tensor
    operation_pointer_logits: torch.Tensor
    quantized_memory: torch.Tensor
    vq_loss: torch.Tensor
    vq_per_example_loss: torch.Tensor
    code_indices: torch.Tensor
    assignment_counts: torch.Tensor
    active_code_count: torch.Tensor
    codebook_utilization: torch.Tensor
    codebook_perplexity: torch.Tensor


@dataclass
class EncodedMemory:
    memory: torch.Tensor
    vq: object


@dataclass
class PrefixDecoderOutput:
    decoded_states: torch.Tensor
    node_type_logits: torch.Tensor
    categorical_logits: tuple
    geometry: torch.Tensor


@dataclass
class RelationDecoderOutput:
    edge_presence_logits: torch.Tensor
    edge_type_logits: torch.Tensor
    operation_pointer_logits: torch.Tensor


class FlatMixedVQModel(nn.Module):
    """One-stream flat baseline with no dependency edges in its encoder."""

    def __init__(self, config=None):
        super().__init__()
        self.config = config or FlatBaselineConfig()
        self.config.validate()
        d_model = self.config.model_dim

        self.field_embeddings = nn.ModuleList(
            nn.Embedding(len(vocabulary.tokens), d_model)
            for vocabulary in _INPUT_VOCABULARIES
        )
        self.geometry_projection = nn.Linear(GEOMETRY_WIDTH, d_model)
        self.geometry_mask_projection = nn.Linear(GEOMETRY_WIDTH, d_model)
        self.node_position_embedding = nn.Embedding(
            self.config.max_nodes, d_model
        )
        self.input_norm = nn.LayerNorm(d_model)
        self.latent_queries = nn.Parameter(
            torch.empty(self.config.latent_tokens, d_model)
        )
        nn.init.normal_(self.latent_queries, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=self.config.num_heads,
            dim_feedforward=self.config.feedforward_dim,
            dropout=self.config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.config.encoder_layers,
            norm=nn.LayerNorm(d_model),
        )
        self.to_codebook = nn.Linear(d_model, self.config.codebook_dim)
        self.vq = EMAVectorQuantizer(
            self.config.codebook_size,
            self.config.codebook_dim,
            self.config.commitment_cost,
            self.config.ema_decay,
            self.config.ema_epsilon,
        )
        self.from_codebook = nn.Linear(self.config.codebook_dim, d_model)

        self.bos = nn.Parameter(torch.empty(d_model))
        nn.init.normal_(self.bos, std=0.02)
        self.decoder_position_embedding = nn.Embedding(
            self.config.max_nodes, d_model
        )
        self.decoder_input_norm = nn.LayerNorm(d_model)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=self.config.num_heads,
            dim_feedforward=self.config.feedforward_dim,
            dropout=self.config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=self.config.decoder_layers,
            norm=nn.LayerNorm(d_model),
        )

        self.node_type_head = nn.Linear(d_model, len(NODE_TYPES.tokens))
        self.categorical_heads = nn.ModuleList(
            nn.Linear(d_model, len(vocabulary.tokens))
            for vocabulary in _ATTRIBUTE_VOCABULARIES
        )
        self.geometry_head = nn.Linear(d_model, GEOMETRY_WIDTH)

        self.edge_source = nn.Linear(d_model, self.config.edge_pair_dim)
        self.edge_target = nn.Linear(d_model, self.config.edge_pair_dim)
        self.edge_presence_head = nn.Linear(self.config.edge_pair_dim, 1)
        self.edge_type_head = nn.Linear(
            self.config.edge_pair_dim, len(EDGE_TYPES.tokens)
        )
        self.operation_queries = nn.Parameter(
            torch.empty(self.config.max_operations, d_model)
        )
        nn.init.normal_(self.operation_queries, std=0.02)
        self.operation_keys = nn.Linear(d_model, d_model)

    def forward(
        self,
        categorical_ids,
        geometry,
        geometry_mask,
        padding_mask,
        target,
    ):
        self._validate_inputs(
            categorical_ids, geometry, geometry_mask, padding_mask, target
        )
        encoded = self.encode_to_memory(
            categorical_ids, geometry, geometry_mask, padding_mask
        )
        target_categories = torch.cat(
            (
                target["node_type_ids"].unsqueeze(-1),
                target["categorical_attributes"],
            ),
            dim=-1,
        )
        target_content = self._record_content(
            target_categories,
            target["geometry"],
            target["geometry_mask"],
        )
        batch_size = categorical_ids.size(0)
        bos = self.bos.view(1, 1, -1).expand(batch_size, 1, -1)
        shifted = torch.cat((bos, target_content[:, :-1]), dim=1)
        decoder_valid = torch.cat(
            (
                torch.ones(
                    batch_size,
                    1,
                    dtype=torch.bool,
                    device=padding_mask.device,
                ),
                target["node_mask"][:, :-1],
            ),
            dim=1,
        )
        decoded_states = self._decode_embedded_prefix(
            shifted, encoded.memory, decoder_valid
        )
        decoded = PrefixDecoderOutput(
            decoded_states,
            self.node_type_head(decoded_states),
            tuple(head(decoded_states) for head in self.categorical_heads),
            torch.tanh(self.geometry_head(decoded_states)),
        )
        relations = self.decode_relations(
            decoded.decoded_states, target["node_mask"]
        )

        return FlatMixedVQOutput(
            decoded.decoded_states,
            decoded.node_type_logits,
            decoded.categorical_logits,
            decoded.geometry,
            relations.edge_presence_logits,
            relations.edge_type_logits,
            relations.operation_pointer_logits,
            encoded.memory,
            encoded.vq.loss,
            encoded.vq.per_example_loss,
            encoded.vq.indices,
            encoded.vq.assignment_counts,
            encoded.vq.active_code_count,
            encoded.vq.utilization,
            encoded.vq.perplexity,
        )

    def encode_to_memory(
        self, categorical_ids, geometry, geometry_mask, padding_mask
    ):
        """Encode flat records and return quantized decoder memory."""

        self._validate_encoder_inputs(
            categorical_ids, geometry, geometry_mask, padding_mask
        )
        batch_size, node_count = categorical_ids.shape[:2]
        positions = torch.arange(
            node_count, device=categorical_ids.device
        ).unsqueeze(0)
        content = self._record_content(
            categorical_ids, geometry, geometry_mask
        )
        nodes = self.input_norm(
            content + self.node_position_embedding(positions)
        )
        queries = self.latent_queries.unsqueeze(0).expand(
            batch_size, -1, -1
        )
        encoder_input = torch.cat((queries, nodes), dim=1)
        query_mask = torch.zeros(
            batch_size,
            self.config.latent_tokens,
            dtype=torch.bool,
            device=padding_mask.device,
        )
        encoder_padding = torch.cat((query_mask, ~padding_mask), dim=1)
        encoded = self.encoder(
            encoder_input, src_key_padding_mask=encoder_padding
        )
        latent = self.to_codebook(
            encoded[:, : self.config.latent_tokens]
        )
        vq = self.vq(latent)
        return EncodedMemory(self.from_codebook(vq.quantized), vq)

    def memory_from_indices(self, indices):
        """Look up supplied mixed-code indices without running or updating VQ."""

        if indices.dtype != torch.long:
            raise TypeError("latent indices must use torch.long")
        if indices.dim() != 2 or indices.size(1) != self.config.latent_tokens:
            raise ValueError(
                "latent indices must have shape [B, latent_tokens]"
            )
        if indices.size(0) == 0:
            raise ValueError("latent indices require a positive batch size")
        if indices.numel() and (
            int(indices.min().item()) < 0
            or int(indices.max().item()) >= self.config.codebook_size
        ):
            raise ValueError("latent index is outside the configured codebook")
        quantized = torch.nn.functional.embedding(
            indices, self.vq.embedding
        )
        return self.from_codebook(quantized)

    def decode_prefix(
        self,
        memory,
        categorical_prefix,
        geometry_prefix,
        geometry_mask_prefix,
        prefix_mask=None,
    ):
        """Decode BOS plus the declared preceding node-record prefix."""

        if memory.dim() != 3 or memory.size(1) != self.config.latent_tokens:
            raise ValueError(
                "memory must have shape [B, latent_tokens, model_dim]"
            )
        if memory.size(2) != self.config.model_dim:
            raise ValueError("memory width disagrees with model_dim")
        batch_size = memory.size(0)
        prefix_length = categorical_prefix.size(1)
        if (
            categorical_prefix.dim() != 3
            or categorical_prefix.shape
            != (batch_size, prefix_length, len(_INPUT_VOCABULARIES))
        ):
            raise ValueError(
                "categorical prefix must have shape [B, P, 10]"
            )
        if categorical_prefix.dtype != torch.long:
            raise TypeError("categorical prefix must use torch.long")
        expected_geometry = (batch_size, prefix_length, GEOMETRY_WIDTH)
        if geometry_prefix.shape != expected_geometry:
            raise ValueError("geometry prefix must have shape [B, P, 39]")
        if geometry_mask_prefix.shape != expected_geometry:
            raise ValueError("geometry-mask prefix must align with geometry")
        if not geometry_prefix.dtype.is_floating_point:
            raise TypeError("geometry prefix must be floating point")
        if geometry_mask_prefix.dtype != torch.bool:
            raise TypeError("geometry-mask prefix must use torch.bool")
        output_length = prefix_length + 1
        if output_length > self.config.max_nodes:
            raise ValueError("decoded prefix exceeds configured max_nodes")
        if prefix_mask is None:
            prefix_mask = torch.ones(
                batch_size,
                prefix_length,
                dtype=torch.bool,
                device=memory.device,
            )
        if prefix_mask.shape != (batch_size, prefix_length):
            raise ValueError("prefix mask must have shape [B, P]")
        if prefix_mask.dtype != torch.bool:
            raise TypeError("prefix mask must use torch.bool")
        if (
            categorical_prefix.device != memory.device
            or geometry_prefix.device != memory.device
            or geometry_mask_prefix.device != memory.device
            or prefix_mask.device != memory.device
        ):
            raise ValueError("decoder prefix tensors must share memory device")

        prefix_content = self._record_content(
            categorical_prefix, geometry_prefix, geometry_mask_prefix
        )
        bos = self.bos.view(1, 1, -1).expand(batch_size, 1, -1)
        shifted = torch.cat((bos, prefix_content), dim=1)
        decoder_valid = torch.cat(
            (
                torch.ones(
                    batch_size,
                    1,
                    dtype=torch.bool,
                    device=memory.device,
                ),
                prefix_mask,
            ),
            dim=1,
        )
        decoded = self._decode_embedded_prefix(
            shifted, memory, decoder_valid
        )
        return PrefixDecoderOutput(
            decoded,
            self.node_type_head(decoded),
            tuple(head(decoded) for head in self.categorical_heads),
            torch.tanh(self.geometry_head(decoded)),
        )

    def _decode_embedded_prefix(
        self, shifted, memory, decoder_valid
    ):
        """Decode an embedded BOS-plus-prefix sequence with shared masking."""

        output_length = shifted.size(1)
        positions = torch.arange(
            output_length, device=shifted.device
        ).unsqueeze(0)
        decoder_input = self.decoder_input_norm(
            shifted + self.decoder_position_embedding(positions)
        )
        causal_mask = torch.triu(
            torch.ones(
                output_length,
                output_length,
                dtype=torch.bool,
                device=shifted.device,
            ),
            diagonal=1,
        )
        return self.decoder(
            decoder_input,
            memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=~decoder_valid,
        )

    def decode_relations(self, decoded_states, node_mask):
        """Predict directed edges and ordered operation pointers."""

        if decoded_states.dim() != 3:
            raise ValueError("decoded states must have shape [B, N, model_dim]")
        batch_size, node_count, width = decoded_states.shape
        if width != self.config.model_dim:
            raise ValueError("decoded-state width disagrees with model_dim")
        if node_mask.shape != (batch_size, node_count):
            raise ValueError("node mask must have shape [B, N]")
        if node_mask.dtype != torch.bool:
            raise TypeError("node mask must use torch.bool")
        source = self.edge_source(decoded_states).unsqueeze(2)
        destination = self.edge_target(decoded_states).unsqueeze(1)
        pairs = torch.tanh(source + destination)
        edge_presence = self.edge_presence_head(pairs).squeeze(-1)
        edge_types = self.edge_type_head(pairs)
        operation_logits = torch.einsum(
            "od,bnd->bon",
            self.operation_queries,
            self.operation_keys(decoded_states),
        )
        minimum = torch.finfo(operation_logits.dtype).min
        operation_logits = operation_logits.masked_fill(
            ~node_mask.unsqueeze(1), minimum
        )
        return RelationDecoderOutput(
            edge_presence, edge_types, operation_logits
        )

    def _record_content(self, categorical_ids, geometry, geometry_mask):
        categorical = None
        for index, embedding in enumerate(self.field_embeddings):
            current = embedding(categorical_ids[..., index])
            categorical = current if categorical is None else categorical + current
        masked_geometry = geometry * geometry_mask.to(geometry.dtype)
        return (
            categorical
            + self.geometry_projection(masked_geometry)
            + self.geometry_mask_projection(geometry_mask.to(geometry.dtype))
        )

    def _validate_inputs(
        self, categorical_ids, geometry, geometry_mask, padding_mask, target
    ):
        self._validate_encoder_inputs(
            categorical_ids, geometry, geometry_mask, padding_mask
        )
        batch_size, node_count = categorical_ids.shape[:2]
        required = {
            "node_type_ids",
            "categorical_attributes",
            "geometry",
            "geometry_mask",
            "node_mask",
            "operation_sequence",
            "operation_mask",
            "edge_index",
            "edge_type_ids",
            "edge_offsets",
        }
        if not isinstance(target, dict) or not required.issubset(target):
            raise ValueError("target must come from ReconstructionBatch.to_torch()")
        if not torch.equal(padding_mask, target["node_mask"]):
            raise ValueError("input padding_mask and target node_mask disagree")
        if target["operation_sequence"].size(1) > self.config.max_operations:
            raise ValueError("operation count exceeds configured max_operations")

    def _validate_encoder_inputs(
        self, categorical_ids, geometry, geometry_mask, padding_mask
    ):
        if categorical_ids.dim() != 3 or categorical_ids.size(-1) != 10:
            raise ValueError("categorical_ids must have shape [B, N, 10]")
        batch_size, node_count = categorical_ids.shape[:2]
        if node_count > self.config.max_nodes:
            raise ValueError("node count exceeds configured max_nodes")
        if geometry.shape != (batch_size, node_count, GEOMETRY_WIDTH):
            raise ValueError("geometry must have shape [B, N, 39]")
        if geometry_mask.shape != geometry.shape:
            raise ValueError("geometry_mask must align with geometry")
        if padding_mask.shape != (batch_size, node_count):
            raise ValueError("padding_mask must have shape [B, N]")
        if categorical_ids.dtype != torch.long:
            raise TypeError("categorical_ids must use torch.long")
        if not geometry.dtype.is_floating_point:
            raise TypeError("geometry must be floating point")
        if geometry_mask.dtype != torch.bool or padding_mask.dtype != torch.bool:
            raise TypeError("all masks must use torch.bool")
