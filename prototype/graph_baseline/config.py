"""Frozen configuration for the single corrected graph-native decoder."""

from __future__ import annotations

from dataclasses import dataclass, field

from prototype.axis_geometry import (
    AXIS_CONSTRUCTION_ALGORITHM_ID,
    AXIS_COORDINATE_SPACE,
    AXIS_GEOMETRY_CHANNEL_INDICES,
    AXIS_GEOMETRY_CONTRACT_ID,
    RAW_CONSTRAINED_AXIS_EVIDENCE_POLICY,
    axis_geometry_metadata,
)
from prototype.flat_baseline.config import (
    FlatBaselineConfig,
    FlatBaselineConfigurationError,
)
from prototype.flat_baseline.constrained_v4_config import (
    V4_BASE_GEOMETRY_CONTRACT_ID,
    V4_LEARNED_GEOMETRY_CHANNEL_INDICES,
)
from prototype.node_conditioned_categories import (
    V4_CATEGORICAL_CONTRACT_ID,
    v4_categorical_contract_metadata,
)
from prototype.node_grammar import (
    COMPLETION_ALGORITHM_ID,
    NODE_GRAMMAR_CONTRACT_ID,
    VALID_REQUESTED_NODE_COUNTS,
    node_grammar_metadata,
)
from prototype.profile_geometry import PROFILE_EXTENT_MAX, PROFILE_EXTENT_MIN, PROFILE_FAMILIES
from prototype.reference_plane_geometry import CANONICAL_PLANE_CONTRACT_ID

from .graph_contract import graph_contract_metadata


GRAPH_MODEL_NAME = "B0-GRAPH-NATIVE-EDGE-DECODER-V1-POSITION-BIAS-C1"
GRAPH_CHECKPOINT_VERSION = 2
GRAPH_MODEL_CONFIG_VERSION = 2
GRAPH_DECODER_CONTRACT_VERSION = 2
GRAPH_PAIR_FEATURE_ORDER = (
    "source_decoded_state",
    "destination_decoded_state",
    "source_constrained_node_type_embedding",
    "destination_constrained_node_type_embedding",
    "source_position_embedding",
    "destination_position_embedding",
    "global_quantized_program_context",
    "signed_relative_serialized_position",
)
GRAPH_PAIR_HIDDEN_DIM = 14
GRAPH_POSITION_BIAS_RANK = 6
GRAPH_LOSS_NORMALIZATION = "mean-active-pairs-per-example-then-batch-mean-v1"
GRAPH_NODE_PATH_ID = "frozen-v6-generated-constrained-node-path"
GRAPH_SCIENTIFIC_CORRECTION_INDEX = 1
GRAPH_SCIENTIFIC_CORRECTION_LIMIT = 1
PARENT_GRAPH_COMMIT = "089b9f3d0e5a61fb19ef3fa05e993fc4eceffdcb"
PARENT_GRAPH_PILOT_JOB = 3341942
GRAPH_CORRECTION_HYPOTHESIS = (
    "explicit-directed-ordered-position-bias-for-repeated-instance-alignment"
)


@dataclass(frozen=True)
class GraphV1Config(FlatBaselineConfig):
    model_name: str = GRAPH_MODEL_NAME
    checkpoint_version: int = GRAPH_CHECKPOINT_VERSION
    model_config_version: int = GRAPH_MODEL_CONFIG_VERSION
    decoder_contract_version: int = GRAPH_DECODER_CONTRACT_VERSION
    profile_family_order: tuple = tuple(item.value for item in PROFILE_FAMILIES)
    learned_geometry_channel_indices: tuple = V4_LEARNED_GEOMETRY_CHANNEL_INDICES
    canonical_plane_contract_id: str = CANONICAL_PLANE_CONTRACT_ID
    base_geometry_contract_id: str = V4_BASE_GEOMETRY_CONTRACT_ID
    categorical_selection_contract_id: str = V4_CATEGORICAL_CONTRACT_ID
    categorical_selection_contract: dict = field(
        default_factory=v4_categorical_contract_metadata
    )
    node_grammar_contract_id: str = NODE_GRAMMAR_CONTRACT_ID
    node_grammar_contract: dict = field(default_factory=node_grammar_metadata)
    valid_requested_node_counts: tuple = VALID_REQUESTED_NODE_COUNTS
    completion_algorithm_id: str = COMPLETION_ALGORITHM_ID
    axis_geometry_contract_id: str = AXIS_GEOMETRY_CONTRACT_ID
    axis_geometry_contract: dict = field(default_factory=axis_geometry_metadata)
    axis_construction_algorithm_id: str = AXIS_CONSTRUCTION_ALGORITHM_ID
    axis_geometry_channel_indices: tuple = AXIS_GEOMETRY_CHANNEL_INDICES
    axis_coordinate_space: str = AXIS_COORDINATE_SPACE
    axis_raw_constrained_evidence_policy: str = RAW_CONSTRAINED_AXIS_EVIDENCE_POLICY
    profile_extent_min: float = PROFILE_EXTENT_MIN
    profile_extent_max: float = PROFILE_EXTENT_MAX
    profile_family_loss_weight: float = 1.0
    profile_parameter_loss_weight: float = 1.0
    graph_contract: dict = field(default_factory=graph_contract_metadata)
    pair_feature_order: tuple = GRAPH_PAIR_FEATURE_ORDER
    pair_hidden_dim: int = GRAPH_PAIR_HIDDEN_DIM
    position_bias_rank: int = GRAPH_POSITION_BIAS_RANK
    graph_loss_normalization: str = GRAPH_LOSS_NORMALIZATION
    inherited_node_path_id: str = GRAPH_NODE_PATH_ID
    graph_edge_loss_weight: float = 1.0

    def validate(self):
        FlatBaselineConfig.validate(self)
        fixed = (
            ("model_name", GRAPH_MODEL_NAME),
            ("checkpoint_version", GRAPH_CHECKPOINT_VERSION),
            ("model_config_version", GRAPH_MODEL_CONFIG_VERSION),
            ("decoder_contract_version", GRAPH_DECODER_CONTRACT_VERSION),
            ("profile_family_order", tuple(item.value for item in PROFILE_FAMILIES)),
            ("learned_geometry_channel_indices", V4_LEARNED_GEOMETRY_CHANNEL_INDICES),
            ("canonical_plane_contract_id", CANONICAL_PLANE_CONTRACT_ID),
            ("base_geometry_contract_id", V4_BASE_GEOMETRY_CONTRACT_ID),
            ("categorical_selection_contract_id", V4_CATEGORICAL_CONTRACT_ID),
            ("categorical_selection_contract", v4_categorical_contract_metadata()),
            ("node_grammar_contract_id", NODE_GRAMMAR_CONTRACT_ID),
            ("node_grammar_contract", node_grammar_metadata()),
            ("valid_requested_node_counts", VALID_REQUESTED_NODE_COUNTS),
            ("completion_algorithm_id", COMPLETION_ALGORITHM_ID),
            ("axis_geometry_contract_id", AXIS_GEOMETRY_CONTRACT_ID),
            ("axis_geometry_contract", axis_geometry_metadata()),
            ("axis_construction_algorithm_id", AXIS_CONSTRUCTION_ALGORITHM_ID),
            ("axis_geometry_channel_indices", AXIS_GEOMETRY_CHANNEL_INDICES),
            ("axis_coordinate_space", AXIS_COORDINATE_SPACE),
            ("axis_raw_constrained_evidence_policy", RAW_CONSTRAINED_AXIS_EVIDENCE_POLICY),
            ("profile_extent_min", PROFILE_EXTENT_MIN),
            ("profile_extent_max", PROFILE_EXTENT_MAX),
            ("graph_contract", graph_contract_metadata()),
            ("pair_feature_order", GRAPH_PAIR_FEATURE_ORDER),
            ("pair_hidden_dim", GRAPH_PAIR_HIDDEN_DIM),
            ("position_bias_rank", GRAPH_POSITION_BIAS_RANK),
            ("graph_loss_normalization", GRAPH_LOSS_NORMALIZATION),
            ("inherited_node_path_id", GRAPH_NODE_PATH_ID),
            ("graph_edge_loss_weight", 1.0),
        )
        for name, expected in fixed:
            if getattr(self, name) != expected:
                raise FlatBaselineConfigurationError(
                    "{} must equal {!r} for graph V1".format(name, expected)
                )
        if self.max_operations != 2:
            raise FlatBaselineConfigurationError("graph V1 requires two operations max")
