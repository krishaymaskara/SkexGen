"""Versioned configuration for the prefix-grammar constrained V6 model."""

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
from prototype.node_grammar import (
    COMPLETION_ALGORITHM_ID,
    NODE_GRAMMAR_CONTRACT_ID,
    VALID_REQUESTED_NODE_COUNTS,
    node_grammar_metadata,
)
from prototype.node_conditioned_categories import (
    V4_CATEGORICAL_CONTRACT_ID,
    v4_categorical_contract_metadata,
)
from prototype.profile_geometry import PROFILE_EXTENT_MAX, PROFILE_EXTENT_MIN, PROFILE_FAMILIES
from prototype.reference_plane_geometry import CANONICAL_PLANE_CONTRACT_ID

from .config import FlatBaselineConfig, FlatBaselineConfigurationError
from .constrained_v4_config import (
    V4_BASE_GEOMETRY_CONTRACT_ID,
    V4_LEARNED_GEOMETRY_CHANNEL_INDICES,
)


CONSTRAINED_PROFILE_MODEL_NAME = (
    "B0-FLAT-CONSTRAINED-PROFILE-CANONICAL-PLANE-"
    "NODE-CATEGORIES-PREFIX-GRAMMAR-CONSTRAINED-AXIS-v6"
)
CONSTRAINED_PROFILE_MODEL_CONFIG_VERSION = 6
CONSTRAINED_PROFILE_DECODER_CONTRACT_VERSION = 6
CONSTRAINED_PROFILE_CHECKPOINT_VERSION = 6
CONSTRAINED_PROFILE_FAMILY_ORDER = tuple(item.value for item in PROFILE_FAMILIES)
V6_LEARNED_GEOMETRY_CHANNEL_INDICES = V4_LEARNED_GEOMETRY_CHANNEL_INDICES
V6_BASE_GEOMETRY_CONTRACT_ID = V4_BASE_GEOMETRY_CONTRACT_ID


@dataclass(frozen=True)
class ConstrainedProfileV6Config(FlatBaselineConfig):
    model_name: str = CONSTRAINED_PROFILE_MODEL_NAME
    model_config_version: int = CONSTRAINED_PROFILE_MODEL_CONFIG_VERSION
    decoder_contract_version: int = CONSTRAINED_PROFILE_DECODER_CONTRACT_VERSION
    checkpoint_version: int = CONSTRAINED_PROFILE_CHECKPOINT_VERSION
    profile_family_order: tuple = CONSTRAINED_PROFILE_FAMILY_ORDER
    learned_geometry_channel_indices: tuple = V6_LEARNED_GEOMETRY_CHANNEL_INDICES
    canonical_plane_contract_id: str = CANONICAL_PLANE_CONTRACT_ID
    base_geometry_contract_id: str = V6_BASE_GEOMETRY_CONTRACT_ID
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
    axis_raw_constrained_evidence_policy: str = (
        RAW_CONSTRAINED_AXIS_EVIDENCE_POLICY
    )
    profile_extent_min: float = PROFILE_EXTENT_MIN
    profile_extent_max: float = PROFILE_EXTENT_MAX
    profile_family_loss_weight: float = 1.0
    profile_parameter_loss_weight: float = 1.0

    def validate(self):
        super().validate()
        fixed = (
            ("model_name", CONSTRAINED_PROFILE_MODEL_NAME),
            ("model_config_version", CONSTRAINED_PROFILE_MODEL_CONFIG_VERSION),
            ("decoder_contract_version", CONSTRAINED_PROFILE_DECODER_CONTRACT_VERSION),
            ("checkpoint_version", CONSTRAINED_PROFILE_CHECKPOINT_VERSION),
            ("profile_family_order", CONSTRAINED_PROFILE_FAMILY_ORDER),
            ("learned_geometry_channel_indices", V6_LEARNED_GEOMETRY_CHANNEL_INDICES),
            ("canonical_plane_contract_id", CANONICAL_PLANE_CONTRACT_ID),
            ("base_geometry_contract_id", V6_BASE_GEOMETRY_CONTRACT_ID),
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
            (
                "axis_raw_constrained_evidence_policy",
                RAW_CONSTRAINED_AXIS_EVIDENCE_POLICY,
            ),
            ("profile_extent_min", PROFILE_EXTENT_MIN),
            ("profile_extent_max", PROFILE_EXTENT_MAX),
        )
        for name, expected in fixed:
            if getattr(self, name) != expected:
                raise FlatBaselineConfigurationError(
                    "{} must equal {!r} for the V6 contract".format(name, expected)
                )
        if self.max_operations != 2:
            raise FlatBaselineConfigurationError(
                "V6 controlled node grammar requires max_operations == 2"
            )
        if self.profile_family_loss_weight != self.categorical_loss_weight:
            raise FlatBaselineConfigurationError(
                "profile-family loss weight must match categorical loss weight"
            )
        if self.profile_parameter_loss_weight != self.geometry_loss_weight:
            raise FlatBaselineConfigurationError(
                "profile-parameter loss weight must match geometry loss weight"
            )
