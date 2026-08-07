"""One common per-example normalized loss for both GE1 encoder arms."""

from __future__ import annotations

from prototype.graph_baseline.config import GraphV1Config
from prototype.graph_baseline.losses import GraphV1Loss, graph_v1_loss

from .config import GE1Config
from .decoder_contract import COMMON_LOSS_VERSION


GE1Loss = GraphV1Loss
LOSS_CONTRACT_VERSION = COMMON_LOSS_VERSION


def common_ge1_loss(output, target, profile_targets, config):
    """Use the retained Graph V1 per-example component and total assembly.

    Graph V1 already normalizes each component independently per example and
    then takes the batch mean.  C5 therefore introduces no legacy aggregation
    difference.  The continuous GE1 decoder supplies an exactly zero VQ term.
    """

    if not isinstance(config, GE1Config):
        raise TypeError("config must be GE1Config")
    config.validate()
    inherited = GraphV1Config(
        model_dim=config.model_dim,
        num_heads=config.attention_heads,
        decoder_layers=config.decoder_layers,
        max_nodes=config.max_nodes,
        max_operations=config.max_operations,
        latent_tokens=config.latent_tokens,
        codebook_dim=config.bottleneck_dim,
        dropout=config.dropout,
        node_type_loss_weight=config.node_type_loss_weight,
        categorical_loss_weight=config.categorical_loss_weight,
        geometry_loss_weight=config.geometry_loss_weight,
        profile_family_loss_weight=config.profile_family_loss_weight,
        profile_parameter_loss_weight=config.profile_parameter_loss_weight,
        graph_edge_loss_weight=config.graph_edge_loss_weight,
    )
    inherited.validate()
    return graph_v1_loss(output, target, profile_targets, inherited)
