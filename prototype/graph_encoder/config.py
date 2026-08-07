"""Frozen, deterministic C1 configuration for GE1."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math

from prototype.flat_baseline.config import FlatBaselineConfig
from prototype.graph_baseline.config import GraphV1Config

from .errors import GraphEncoderError


MODEL_FAMILY = "GE1-MODEL-v1"
FLAT_ARM = "GE1-FLAT-CHRONOLOGICAL-CONTINUOUS-V1"
GRAPH_ARM = "GE1-TYPED-GRAPH-POSITION-FREE-CONTINUOUS-V1"
SHARED_DECODER = "GE1-SHARED-TYPED-EDGE-DECODER-V1"
CHECKPOINT_SCHEMA = "GE1-CHECKPOINT-v1"
PROTOCOL_IDENTITY = "GE1-STAGE0-PREREG-v1"
EXPERIMENT_IDENTITY = "GE1-SHARED-DECODER-ENCODER-COMPARISON"

ENCODERS = ("flat", "typed_graph")
BOTTLENECK_MODE = "continuous"
AUTHORIZED_SEEDS = (2026, 2027, 2028)
PLANNED_SEEDS = AUTHORIZED_SEEDS
AUTHORIZED_FALLBACK_SEEDS = (2026, 2027)

RELATIONAL_LAYERS = 3
INITIAL_RELATION_BASIS_COUNT = 2

TRAINING_EPOCHS = 50
TRAINING_BATCH_SIZE = 8
TRAINING_OPTIMIZER = "AdamW"
TRAINING_LEARNING_RATE = 1e-3
TRAINING_WEIGHT_DECAY = 0.0
TRAINING_GRADIENT_CLIP_NORM = 1.0
CHECKPOINT_SELECTION = "fixed_epoch_50"
CHECKPOINT_EPOCH = 50

PLATEAU_RELATIVE_IMPROVEMENT_THRESHOLD = 0.01
PLATEAU_MOVING_BEST_WINDOW_EPOCHS = 5
PLATEAU_START_EPOCH = 10
NORMALIZED_PREFIX_TRAIN_CEILING_SHORTFALL_MAX = 0.05
COMPLETE_VALIDITY_TRAIN_CEILING_SHORTFALL_MAX = 0.10

CAPACITY_ADJUSTMENT_FIELDS = (
    "relation_basis_count",
    "encoder_feedforward_width",
)

_FLAT_CONTRACT = FlatBaselineConfig()
_GRAPH_CONTRACT = GraphV1Config()


@dataclass(frozen=True)
class GE1Config:
    """Immutable model configuration for one of the two accepted GE1 arms."""

    encoder: str
    seed: int = AUTHORIZED_SEEDS[0]
    bottleneck_mode: str = BOTTLENECK_MODE
    model_family: str = MODEL_FAMILY
    shared_decoder: str = SHARED_DECODER
    checkpoint_schema: str = CHECKPOINT_SCHEMA
    protocol_identity: str = PROTOCOL_IDENTITY
    experiment_identity: str = EXPERIMENT_IDENTITY
    model_dim: int = _FLAT_CONTRACT.model_dim
    attention_heads: int = _FLAT_CONTRACT.num_heads
    flat_encoder_layers: int = _FLAT_CONTRACT.encoder_layers
    decoder_layers: int = _FLAT_CONTRACT.decoder_layers
    max_nodes: int = _FLAT_CONTRACT.max_nodes
    max_operations: int = _FLAT_CONTRACT.max_operations
    latent_tokens: int = _FLAT_CONTRACT.latent_tokens
    bottleneck_dim: int = _FLAT_CONTRACT.codebook_dim
    relational_layers: int = RELATIONAL_LAYERS
    relation_basis_count: int = INITIAL_RELATION_BASIS_COUNT
    encoder_feedforward_width: int = _FLAT_CONTRACT.feedforward_dim
    dropout: float = _FLAT_CONTRACT.dropout
    node_type_loss_weight: float = _GRAPH_CONTRACT.node_type_loss_weight
    categorical_loss_weight: float = _GRAPH_CONTRACT.categorical_loss_weight
    geometry_loss_weight: float = _GRAPH_CONTRACT.geometry_loss_weight
    profile_family_loss_weight: float = _GRAPH_CONTRACT.profile_family_loss_weight
    profile_parameter_loss_weight: float = _GRAPH_CONTRACT.profile_parameter_loss_weight
    graph_edge_loss_weight: float = _GRAPH_CONTRACT.graph_edge_loss_weight

    @property
    def arm_identity(self):
        if self.encoder == "flat":
            return FLAT_ARM
        if self.encoder == "typed_graph":
            return GRAPH_ARM
        raise GraphEncoderError(
            "invalid_configuration",
            "encoder must be one of flat, typed_graph",
        )

    def validate(self):
        if self.encoder not in ENCODERS:
            raise GraphEncoderError(
                "invalid_configuration",
                "encoder must be one of flat, typed_graph",
            )
        _integer(self.seed, "seed", positive=True)
        if self.seed not in AUTHORIZED_SEEDS:
            raise GraphEncoderError(
                "unauthorized_configuration",
                "seed must be one of 2026, 2027, 2028",
            )
        if self.bottleneck_mode != BOTTLENECK_MODE:
            raise GraphEncoderError(
                "unauthorized_configuration",
                "bottleneck_mode must equal continuous",
            )
        fixed_identities = (
            ("model_family", MODEL_FAMILY),
            ("shared_decoder", SHARED_DECODER),
            ("checkpoint_schema", CHECKPOINT_SCHEMA),
            ("protocol_identity", PROTOCOL_IDENTITY),
            ("experiment_identity", EXPERIMENT_IDENTITY),
        )
        for name, expected in fixed_identities:
            if getattr(self, name) != expected:
                raise GraphEncoderError(
                    "identity_mismatch",
                    "{} must equal {}".format(name, expected),
                )

        positive_integer_fields = (
            "model_dim",
            "attention_heads",
            "flat_encoder_layers",
            "decoder_layers",
            "max_nodes",
            "max_operations",
            "latent_tokens",
            "bottleneck_dim",
            "relational_layers",
            "relation_basis_count",
            "encoder_feedforward_width",
        )
        for name in positive_integer_fields:
            _integer(getattr(self, name), name, positive=True)
        if self.model_dim % self.attention_heads != 0:
            raise GraphEncoderError(
                "invalid_configuration",
                "model_dim must be divisible by attention_heads",
            )

        inherited_dimensions = (
            ("model_dim", _FLAT_CONTRACT.model_dim),
            ("attention_heads", _FLAT_CONTRACT.num_heads),
            ("flat_encoder_layers", _FLAT_CONTRACT.encoder_layers),
            ("decoder_layers", _FLAT_CONTRACT.decoder_layers),
            ("max_nodes", _FLAT_CONTRACT.max_nodes),
            ("max_operations", _FLAT_CONTRACT.max_operations),
            ("latent_tokens", _FLAT_CONTRACT.latent_tokens),
            ("bottleneck_dim", _FLAT_CONTRACT.codebook_dim),
            ("relational_layers", RELATIONAL_LAYERS),
        )
        for name, expected in inherited_dimensions:
            if getattr(self, name) != expected:
                raise GraphEncoderError(
                    "unauthorized_configuration",
                    "{} must equal the frozen value {}".format(name, expected),
                )
        _finite_range(self.dropout, "dropout", 0.0, 1.0, upper_inclusive=False)
        if float(self.dropout) != float(_FLAT_CONTRACT.dropout):
            raise GraphEncoderError(
                "unauthorized_configuration",
                "dropout must equal the inherited value {}".format(
                    _FLAT_CONTRACT.dropout
                ),
            )

        inherited_losses = (
            ("node_type_loss_weight", _GRAPH_CONTRACT.node_type_loss_weight),
            ("categorical_loss_weight", _GRAPH_CONTRACT.categorical_loss_weight),
            ("geometry_loss_weight", _GRAPH_CONTRACT.geometry_loss_weight),
            (
                "profile_family_loss_weight",
                _GRAPH_CONTRACT.profile_family_loss_weight,
            ),
            (
                "profile_parameter_loss_weight",
                _GRAPH_CONTRACT.profile_parameter_loss_weight,
            ),
            ("graph_edge_loss_weight", _GRAPH_CONTRACT.graph_edge_loss_weight),
        )
        for name, expected in inherited_losses:
            _finite_range(getattr(self, name), name, 0.0, None)
            if float(getattr(self, name)) != float(expected):
                raise GraphEncoderError(
                    "unauthorized_configuration",
                    "{} must equal the inherited value {}".format(name, expected),
                )

    def to_dict(self):
        self.validate()
        values = asdict(self)
        values["arm_identity"] = self.arm_identity
        return values

    def to_json(self):
        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )


@dataclass(frozen=True)
class GE1TrainingConfig:
    """Immutable training policy shared by later GE1 runners."""

    retained_seeds: tuple = PLANNED_SEEDS
    planned_seeds: tuple = PLANNED_SEEDS
    authorized_fallback_seeds: tuple = AUTHORIZED_FALLBACK_SEEDS
    epochs: int = TRAINING_EPOCHS
    batch_size: int = TRAINING_BATCH_SIZE
    optimizer: str = TRAINING_OPTIMIZER
    learning_rate: float = TRAINING_LEARNING_RATE
    weight_decay: float = TRAINING_WEIGHT_DECAY
    gradient_clip_norm: float = TRAINING_GRADIENT_CLIP_NORM
    checkpoint_selection: str = CHECKPOINT_SELECTION
    checkpoint_epoch: int = CHECKPOINT_EPOCH
    plateau_relative_improvement_threshold: float = (
        PLATEAU_RELATIVE_IMPROVEMENT_THRESHOLD
    )
    plateau_moving_best_window_epochs: int = PLATEAU_MOVING_BEST_WINDOW_EPOCHS
    plateau_start_epoch: int = PLATEAU_START_EPOCH
    normalized_prefix_train_ceiling_shortfall_max: float = (
        NORMALIZED_PREFIX_TRAIN_CEILING_SHORTFALL_MAX
    )
    complete_validity_train_ceiling_shortfall_max: float = (
        COMPLETE_VALIDITY_TRAIN_CEILING_SHORTFALL_MAX
    )

    def validate(self):
        if self.planned_seeds != PLANNED_SEEDS:
            raise GraphEncoderError(
                "identity_mismatch",
                "planned_seeds must equal (2026, 2027, 2028)",
            )
        if self.authorized_fallback_seeds != AUTHORIZED_FALLBACK_SEEDS:
            raise GraphEncoderError(
                "identity_mismatch",
                "authorized_fallback_seeds must equal (2026, 2027)",
            )
        if self.retained_seeds not in (PLANNED_SEEDS, AUTHORIZED_FALLBACK_SEEDS):
            raise GraphEncoderError(
                "unauthorized_configuration",
                "retained_seeds must be the planned or authorized fallback set",
            )
        for name in (
            "planned_seeds",
            "authorized_fallback_seeds",
            "retained_seeds",
        ):
            values = getattr(self, name)
            if any(isinstance(item, bool) or not isinstance(item, int) for item in values):
                raise GraphEncoderError(
                    "invalid_configuration",
                    "{} must contain integer seeds".format(name),
                )

        fixed = (
            ("epochs", TRAINING_EPOCHS),
            ("batch_size", TRAINING_BATCH_SIZE),
            ("optimizer", TRAINING_OPTIMIZER),
            ("learning_rate", TRAINING_LEARNING_RATE),
            ("weight_decay", TRAINING_WEIGHT_DECAY),
            ("gradient_clip_norm", TRAINING_GRADIENT_CLIP_NORM),
            ("checkpoint_selection", CHECKPOINT_SELECTION),
            ("checkpoint_epoch", CHECKPOINT_EPOCH),
            (
                "plateau_relative_improvement_threshold",
                PLATEAU_RELATIVE_IMPROVEMENT_THRESHOLD,
            ),
            (
                "plateau_moving_best_window_epochs",
                PLATEAU_MOVING_BEST_WINDOW_EPOCHS,
            ),
            ("plateau_start_epoch", PLATEAU_START_EPOCH),
            (
                "normalized_prefix_train_ceiling_shortfall_max",
                NORMALIZED_PREFIX_TRAIN_CEILING_SHORTFALL_MAX,
            ),
            (
                "complete_validity_train_ceiling_shortfall_max",
                COMPLETE_VALIDITY_TRAIN_CEILING_SHORTFALL_MAX,
            ),
        )
        integer_names = {
            "epochs",
            "batch_size",
            "checkpoint_epoch",
            "plateau_moving_best_window_epochs",
            "plateau_start_epoch",
        }
        numeric_names = {
            "learning_rate",
            "weight_decay",
            "gradient_clip_norm",
            "plateau_relative_improvement_threshold",
            "normalized_prefix_train_ceiling_shortfall_max",
            "complete_validity_train_ceiling_shortfall_max",
        }
        for name, expected in fixed:
            value = getattr(self, name)
            if name in integer_names:
                _integer(value, name, positive=True)
            elif name in numeric_names:
                _finite_range(value, name, 0.0, None)
            if value != expected:
                raise GraphEncoderError(
                    "unauthorized_configuration",
                    "{} must equal the frozen value {}".format(name, expected),
                )

    @property
    def uses_timing_fallback(self):
        return self.retained_seeds == AUTHORIZED_FALLBACK_SEEDS

    def to_dict(self):
        self.validate()
        values = asdict(self)
        for name in (
            "retained_seeds",
            "planned_seeds",
            "authorized_fallback_seeds",
        ):
            values[name] = list(values[name])
        values["uses_timing_fallback"] = self.uses_timing_fallback
        return values

    def to_json(self):
        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )


def _integer(value, name, positive=False):
    if isinstance(value, bool) or not isinstance(value, int):
        raise GraphEncoderError(
            "invalid_configuration", "{} must be an integer".format(name)
        )
    if positive and value <= 0:
        raise GraphEncoderError(
            "invalid_configuration", "{} must be positive".format(name)
        )


def _finite_range(value, name, lower, upper, upper_inclusive=True):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GraphEncoderError(
            "invalid_configuration", "{} must be numeric".format(name)
        )
    numeric = float(value)
    if not math.isfinite(numeric):
        raise GraphEncoderError(
            "invalid_configuration", "{} must be finite".format(name)
        )
    lower_ok = numeric >= lower
    upper_ok = True
    if upper is not None:
        upper_ok = numeric <= upper if upper_inclusive else numeric < upper
    if not lower_ok or not upper_ok:
        raise GraphEncoderError(
            "invalid_configuration",
            "{} is outside the supported range".format(name),
        )
