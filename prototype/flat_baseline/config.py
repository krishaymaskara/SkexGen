"""Explicit serializable plumbing configuration for B0-FLAT-MIXED-VQ."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math


class FlatBaselineConfigurationError(ValueError):
    """The baseline architecture or loss configuration is invalid."""


@dataclass(frozen=True)
class FlatBaselineConfig:
    """Small defaults intended for tests and plumbing, not final experiments."""

    model_dim: int = 32
    num_heads: int = 4
    feedforward_dim: int = 64
    encoder_layers: int = 1
    decoder_layers: int = 1
    dropout: float = 0.0
    max_nodes: int = 16
    max_operations: int = 2
    latent_tokens: int = 2
    codebook_size: int = 16
    codebook_dim: int = 16
    commitment_cost: float = 0.25
    ema_decay: float = 0.99
    ema_epsilon: float = 1e-5
    edge_pair_dim: int = 32
    node_type_loss_weight: float = 1.0
    categorical_loss_weight: float = 1.0
    geometry_loss_weight: float = 1.0
    edge_presence_loss_weight: float = 1.0
    edge_type_loss_weight: float = 1.0
    operation_pointer_loss_weight: float = 1.0
    vq_loss_weight: float = 1.0

    def validate(self):
        positive_integers = (
            "model_dim",
            "num_heads",
            "feedforward_dim",
            "encoder_layers",
            "decoder_layers",
            "max_nodes",
            "max_operations",
            "latent_tokens",
            "codebook_size",
            "codebook_dim",
            "edge_pair_dim",
        )
        for name in positive_integers:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise FlatBaselineConfigurationError(
                    "{} must be a positive integer".format(name)
                )
        if self.model_dim % self.num_heads != 0:
            raise FlatBaselineConfigurationError(
                "model_dim must be divisible by num_heads"
            )
        _range(self.dropout, "dropout", 0.0, 1.0, upper_inclusive=False)
        _range(self.commitment_cost, "commitment_cost", 0.0, None)
        _range(
            self.ema_decay,
            "ema_decay",
            0.0,
            1.0,
            upper_inclusive=False,
        )
        _range(self.ema_epsilon, "ema_epsilon", 0.0, None, lower_inclusive=False)
        for name in (
            "node_type_loss_weight",
            "categorical_loss_weight",
            "geometry_loss_weight",
            "edge_presence_loss_weight",
            "edge_type_loss_weight",
            "operation_pointer_loss_weight",
            "vq_loss_weight",
        ):
            _range(getattr(self, name), name, 0.0, None)

    def to_dict(self):
        self.validate()
        return asdict(self)

    def to_json(self):
        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )


def _range(
    value,
    name,
    lower,
    upper,
    lower_inclusive=True,
    upper_inclusive=True,
):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FlatBaselineConfigurationError("{} must be numeric".format(name))
    numeric = float(value)
    if not math.isfinite(numeric):
        raise FlatBaselineConfigurationError("{} must be finite".format(name))
    lower_ok = numeric >= lower if lower_inclusive else numeric > lower
    upper_ok = (
        True
        if upper is None
        else (numeric <= upper if upper_inclusive else numeric < upper)
    )
    if not lower_ok or not upper_ok:
        right = "infinity" if upper is None else str(upper)
        raise FlatBaselineConfigurationError(
            "{} is outside the supported interval [{}, {}]".format(
                name, lower, right
            )
        )
