"""Continuous flat and typed-graph encoder arms for GE1 C4."""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace

import torch
from torch import nn

from prototype.flat_baseline.config import FlatBaselineConfig
from prototype.flat_baseline.model import FlatMixedVQModel
from prototype.model_data.geometry import GEOMETRY_WIDTH

from .config import GE1Config
from .relational import RelationalEncoderCore


# This arithmetic-only capacity adjustment is frozen before any encoder
# behavior or training result is observed. The graph feed-forward width stays
# at the inherited 64; the flat Transformer width is the sole adjusted field.
FLAT_CAPACITY_MATCHED_FEEDFORWARD_WIDTH = 192
GRAPH_FEEDFORWARD_WIDTH = 64
CAPACITY_TOLERANCE_PERCENT = 5.0


@dataclass(frozen=True)
class EncodedMemory:
    """Common continuous GE1 encoder result.

    ``prequant`` is the continuous output of ``to_codebook`` and is exposed
    for diagnostics only. ``memory`` is
    ``from_codebook(prequant)`` and is the only tensor intended for the shared
    decoder. No nearest-code assignment occurs in either C4 arm.
    """

    memory: torch.Tensor
    prequant: torch.Tensor


def default_encoder_config(encoder, seed=2026):
    """Return the capacity-matched C4 configuration for one encoder arm."""

    width = (
        FLAT_CAPACITY_MATCHED_FEEDFORWARD_WIDTH
        if encoder == "flat"
        else GRAPH_FEEDFORWARD_WIDTH
    )
    config = replace(
        GE1Config(encoder=encoder, seed=seed),
        encoder_feedforward_width=width,
    )
    config.validate()
    return config


class FlatProgramEncoder(nn.Module):
    """Inherited chronological encoder with a continuous VQ bypass."""

    def __init__(self, config=None, inherited_model=None):
        super().__init__()
        self.config = config or default_encoder_config("flat")
        self.config.validate()
        if self.config.encoder != "flat":
            raise ValueError("FlatProgramEncoder requires encoder='flat'")

        source = inherited_model
        if source is None:
            source = FlatMixedVQModel(_flat_baseline_config(self.config))
        _validate_inherited_model(source, self.config)

        # Deep copies preserve the inherited state and computation without
        # retaining its quantizer, decoder, or shared live Parameters.
        self.field_embeddings = copy.deepcopy(source.field_embeddings)
        self.geometry_projection = copy.deepcopy(source.geometry_projection)
        self.geometry_mask_projection = copy.deepcopy(
            source.geometry_mask_projection
        )
        self.node_position_embedding = copy.deepcopy(
            source.node_position_embedding
        )
        self.input_norm = copy.deepcopy(source.input_norm)
        self.latent_queries = nn.Parameter(source.latent_queries.detach().clone())
        self.encoder = copy.deepcopy(source.encoder)
        self.to_codebook = copy.deepcopy(source.to_codebook)
        self.from_codebook = copy.deepcopy(source.from_codebook)

    @classmethod
    def from_inherited(cls, inherited_model, config=None):
        return cls(config=config, inherited_model=inherited_model)

    def forward(self, categorical_ids, geometry, geometry_mask, padding_mask):
        self._validate_inputs(
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
        prequant = self.to_codebook(
            encoded[:, : self.config.latent_tokens]
        )
        memory = self.from_codebook(prequant).contiguous()
        return EncodedMemory(memory=memory, prequant=prequant)

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
        self, categorical_ids, geometry, geometry_mask, padding_mask
    ):
        if categorical_ids.dim() != 3 or categorical_ids.size(-1) != 10:
            raise ValueError("categorical_ids must have shape [B, N, 10]")
        batch_size, node_count = categorical_ids.shape[:2]
        if batch_size <= 0 or node_count <= 0 or node_count > self.config.max_nodes:
            raise ValueError("flat input has an unsupported batch or node count")
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
        if not bool(padding_mask.any(dim=1).all().item()):
            raise ValueError("every flat example requires at least one real node")


class TypedGraphProgramEncoder(nn.Module):
    """Three-layer position-free typed graph encoder and local pooling."""

    def __init__(self, config=None):
        super().__init__()
        self.config = config or default_encoder_config("typed_graph")
        self.config.validate()
        if self.config.encoder != "typed_graph":
            raise ValueError("TypedGraphProgramEncoder requires encoder='typed_graph'")
        if self.config.relation_basis_count != 2:
            raise ValueError("C4 requires exactly two relation bases")
        if self.config.encoder_feedforward_width != GRAPH_FEEDFORWARD_WIDTH:
            raise ValueError("C4 graph feed-forward width must remain 64")

        source = FlatMixedVQModel(_flat_baseline_config(self.config))
        self.field_embeddings = copy.deepcopy(source.field_embeddings)
        self.geometry_projection = copy.deepcopy(source.geometry_projection)
        self.geometry_mask_projection = copy.deepcopy(
            source.geometry_mask_projection
        )
        self.input_norm = copy.deepcopy(source.input_norm)
        self.relational = RelationalEncoderCore(
            self.config.model_dim,
            self.config.relational_layers,
            self.config.relation_basis_count,
            self.config.dropout,
        )
        self.latent_queries = nn.Parameter(source.latent_queries.detach().clone())
        self.pool_attention = nn.MultiheadAttention(
            self.config.model_dim,
            self.config.attention_heads,
            dropout=self.config.dropout,
            batch_first=True,
        )
        self.pool_attention_dropout = nn.Dropout(self.config.dropout)
        self.pool_attention_norm = nn.LayerNorm(self.config.model_dim)
        self.pool_feedforward = nn.Sequential(
            nn.Linear(
                self.config.model_dim,
                self.config.encoder_feedforward_width,
            ),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(
                self.config.encoder_feedforward_width,
                self.config.model_dim,
            ),
        )
        self.pool_feedforward_dropout = nn.Dropout(self.config.dropout)
        self.pool_output_norm = nn.LayerNorm(self.config.model_dim)
        self.to_codebook = copy.deepcopy(source.to_codebook)
        self.from_codebook = copy.deepcopy(source.from_codebook)

    def initialize_nodes(
        self,
        node_type_ids,
        categorical_attributes,
        geometry,
        geometry_mask,
    ):
        """Embed semantic node content without any bookkeeping feature."""

        self._validate_node_content(
            node_type_ids,
            categorical_attributes,
            geometry,
            geometry_mask,
        )
        categorical_ids = torch.cat(
            (node_type_ids.unsqueeze(-1), categorical_attributes),
            dim=-1,
        )
        categorical = None
        for index, embedding in enumerate(self.field_embeddings):
            current = embedding(categorical_ids[:, index])
            categorical = current if categorical is None else categorical + current
        masked_geometry = geometry * geometry_mask.to(geometry.dtype)
        return self.input_norm(
            categorical
            + self.geometry_projection(masked_geometry)
            + self.geometry_mask_projection(geometry_mask.to(geometry.dtype))
        )

    def encode_nodes(
        self,
        node_type_ids,
        categorical_attributes,
        geometry,
        geometry_mask,
        edge_index,
        edge_type_ids,
    ):
        """Diagnostic node-level output used for equivariance checks."""

        nodes = self.initialize_nodes(
            node_type_ids,
            categorical_attributes,
            geometry,
            geometry_mask,
        )
        return self.relational(nodes, edge_index, edge_type_ids)

    def forward(
        self,
        node_type_ids,
        categorical_attributes,
        geometry,
        geometry_mask,
        edge_index,
        edge_type_ids,
        graph_offsets,
    ):
        nodes = self.initialize_nodes(
            node_type_ids,
            categorical_attributes,
            geometry,
            geometry_mask,
        )
        self.relational.layers[0]._validate_inputs(
            nodes,
            edge_index,
            edge_type_ids,
        )
        offsets = self._validate_graph_offsets(graph_offsets, nodes.size(0))
        self._validate_graph_local_edges(edge_index, offsets)
        nodes = self.relational(nodes, edge_index, edge_type_ids)
        pooled = []
        for start, stop in zip(offsets, offsets[1:]):
            graph_nodes = nodes[start:stop].unsqueeze(0)
            queries = self.latent_queries.unsqueeze(0)
            attended, unused_weights = self.pool_attention(
                queries,
                graph_nodes,
                graph_nodes,
                need_weights=False,
            )
            del unused_weights
            attended = self.pool_attention_norm(
                queries + self.pool_attention_dropout(attended)
            )
            feedforward = self.pool_feedforward(attended)
            pooled.append(
                self.pool_output_norm(
                    attended + self.pool_feedforward_dropout(feedforward)
                )
            )
        pooled_queries = torch.cat(pooled, dim=0)
        prequant = self.to_codebook(pooled_queries)
        memory = self.from_codebook(prequant).contiguous()
        return EncodedMemory(memory=memory, prequant=prequant)

    def _validate_node_content(
        self,
        node_type_ids,
        categorical_attributes,
        geometry,
        geometry_mask,
    ):
        if node_type_ids.dim() != 1 or node_type_ids.numel() <= 0:
            raise ValueError("node_type_ids must have shape [N] with N > 0")
        node_count = node_type_ids.size(0)
        if categorical_attributes.shape != (node_count, 9):
            raise ValueError("categorical_attributes must have shape [N, 9]")
        if geometry.shape != (node_count, GEOMETRY_WIDTH):
            raise ValueError("geometry must have shape [N, 39]")
        if geometry_mask.shape != geometry.shape:
            raise ValueError("geometry_mask must align with geometry")
        if node_type_ids.dtype != torch.long:
            raise TypeError("node_type_ids must use torch.long")
        if categorical_attributes.dtype != torch.long:
            raise TypeError("categorical_attributes must use torch.long")
        if not geometry.dtype.is_floating_point:
            raise TypeError("geometry must be floating point")
        if geometry_mask.dtype != torch.bool:
            raise TypeError("geometry_mask must use torch.bool")
        devices = {
            node_type_ids.device,
            categorical_attributes.device,
            geometry.device,
            geometry_mask.device,
        }
        if len(devices) != 1:
            raise ValueError("semantic node tensors must share one device")

    def _validate_graph_offsets(self, graph_offsets, node_count):
        if graph_offsets.dim() != 1 or graph_offsets.numel() < 2:
            raise ValueError("graph_offsets must have shape [B + 1]")
        if graph_offsets.dtype != torch.long:
            raise TypeError("graph_offsets must use torch.long")
        offsets = tuple(int(value) for value in graph_offsets.detach().cpu().tolist())
        if (
            offsets[0] != 0
            or offsets[-1] != node_count
            or any(left >= right for left, right in zip(offsets, offsets[1:]))
        ):
            raise ValueError("graph_offsets must define nonempty node intervals")
        return offsets

    def _validate_graph_local_edges(self, edge_index, offsets):
        if edge_index.numel() == 0:
            return
        covered = torch.zeros(
            edge_index.size(1),
            dtype=torch.bool,
            device=edge_index.device,
        )
        for start, stop in zip(offsets, offsets[1:]):
            local = (
                (edge_index[0] >= start)
                & (edge_index[0] < stop)
                & (edge_index[1] >= start)
                & (edge_index[1] < stop)
            )
            covered = covered | local
        if not bool(covered.all().item()):
            raise ValueError("edge crosses a graph-local pooling boundary")


def encoder_parameter_report(encoder):
    """Return exact non-overlapping and requested nested C4 counts."""

    if isinstance(encoder, FlatProgramEncoder):
        return {
            "node_content_initialization": _count_parameters(
                encoder.field_embeddings,
                encoder.geometry_projection,
                encoder.geometry_mask_projection,
                encoder.input_norm,
            ),
            "chronological_position_embedding": _count_parameters(
                encoder.node_position_embedding
            ),
            "latent_queries": encoder.latent_queries.numel(),
            "transformer_encoder": _count_parameters(encoder.encoder),
            "bottleneck_projection": _count_parameters(
                encoder.to_codebook, encoder.from_codebook
            ),
            "complete_flat_encoder": _count_parameters(encoder),
        }
    if isinstance(encoder, TypedGraphProgramEncoder):
        report = {
            "node_initialization": _count_parameters(
                encoder.field_embeddings,
                encoder.geometry_projection,
                encoder.geometry_mask_projection,
                encoder.input_norm,
            ),
            "relation_bases": sum(
                layer.relation_bases.numel()
                for layer in encoder.relational.layers
            ),
            "relation_direction_mixing": sum(
                layer.relation_mixing.numel()
                for layer in encoder.relational.layers
            ),
            "graph_pooling": _count_parameters(
                encoder.pool_attention,
                encoder.pool_attention_norm,
                encoder.pool_feedforward,
                encoder.pool_output_norm,
            )
            + encoder.latent_queries.numel(),
            "bottleneck_projection": _count_parameters(
                encoder.to_codebook, encoder.from_codebook
            ),
            "complete_graph_encoder": _count_parameters(encoder),
        }
        for index, layer in enumerate(encoder.relational.layers):
            report["relational_layer_{}".format(index)] = _count_parameters(layer)
        return report
    raise TypeError("encoder must be a C4 program encoder")


def capacity_difference_percent(flat_encoder, graph_encoder):
    """Return absolute trainable-count difference relative to flat control."""

    flat_count = _count_parameters(flat_encoder)
    graph_count = _count_parameters(graph_encoder)
    if flat_count <= 0:
        raise ValueError("flat encoder must have trainable parameters")
    return abs(graph_count - flat_count) * 100.0 / float(flat_count)


def _count_parameters(*modules):
    parameters = {}
    for module in modules:
        for parameter in module.parameters():
            if parameter.requires_grad:
                parameters[id(parameter)] = parameter
    return sum(parameter.numel() for parameter in parameters.values())


def _flat_baseline_config(config):
    return FlatBaselineConfig(
        model_dim=config.model_dim,
        num_heads=config.attention_heads,
        feedforward_dim=config.encoder_feedforward_width,
        encoder_layers=config.flat_encoder_layers,
        decoder_layers=config.decoder_layers,
        dropout=config.dropout,
        max_nodes=config.max_nodes,
        max_operations=config.max_operations,
        latent_tokens=config.latent_tokens,
        codebook_dim=config.bottleneck_dim,
    )


def _validate_inherited_model(model, config):
    if not isinstance(model, FlatMixedVQModel):
        raise TypeError("inherited_model must be FlatMixedVQModel")
    expected = _flat_baseline_config(config)
    fields = (
        "model_dim",
        "num_heads",
        "feedforward_dim",
        "encoder_layers",
        "dropout",
        "max_nodes",
        "latent_tokens",
        "codebook_dim",
    )
    for name in fields:
        if getattr(model.config, name) != getattr(expected, name):
            raise ValueError("inherited flat encoder configuration mismatch")
