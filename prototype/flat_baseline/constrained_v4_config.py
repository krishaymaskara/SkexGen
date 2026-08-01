"""Versioned configuration for the constrained-profile V4 flat model."""

from __future__ import annotations

from dataclasses import dataclass, field
import math

from prototype.profile_geometry import (
    PROFILE_EXTENT_MAX,
    PROFILE_EXTENT_MIN,
    PROFILE_FAMILIES,
)
from prototype.reference_plane_geometry import CANONICAL_PLANE_CONTRACT_ID
from prototype.node_conditioned_categories import (
    V4_CATEGORICAL_CONTRACT_ID,
    v4_categorical_contract_metadata,
)

from .config import FlatBaselineConfig, FlatBaselineConfigurationError


CONSTRAINED_PROFILE_MODEL_NAME = (
    "B0-FLAT-CONSTRAINED-PROFILE-CANONICAL-PLANE-"
    "NODE-CONDITIONED-CATEGORIES-v4"
)
CONSTRAINED_PROFILE_MODEL_CONFIG_VERSION = 4
CONSTRAINED_PROFILE_DECODER_CONTRACT_VERSION = 4
CONSTRAINED_PROFILE_CHECKPOINT_VERSION = 4
CONSTRAINED_PROFILE_FAMILY_ORDER = tuple(
    family.value for family in PROFILE_FAMILIES
)
V4_LEARNED_GEOMETRY_CHANNEL_INDICES = (33, 34, 35, 36, 37, 38)
V4_BASE_GEOMETRY_CONTRACT_ID = (
    "B0-flat-constrained-profile-canonical-plane-geometry-v3"
)


@dataclass(frozen=True)
class ConstrainedProfileV4Config(FlatBaselineConfig):
    """Frozen V4 identity layered over the shared flat-model dimensions."""

    model_name: str = CONSTRAINED_PROFILE_MODEL_NAME
    model_config_version: int = CONSTRAINED_PROFILE_MODEL_CONFIG_VERSION
    decoder_contract_version: int = (
        CONSTRAINED_PROFILE_DECODER_CONTRACT_VERSION
    )
    checkpoint_version: int = CONSTRAINED_PROFILE_CHECKPOINT_VERSION
    profile_family_order: tuple = CONSTRAINED_PROFILE_FAMILY_ORDER
    learned_geometry_channel_indices: tuple = V4_LEARNED_GEOMETRY_CHANNEL_INDICES
    canonical_plane_contract_id: str = CANONICAL_PLANE_CONTRACT_ID
    base_geometry_contract_id: str = V4_BASE_GEOMETRY_CONTRACT_ID
    categorical_selection_contract_id: str = V4_CATEGORICAL_CONTRACT_ID
    categorical_selection_contract: dict = field(
        default_factory=v4_categorical_contract_metadata
    )
    profile_extent_min: float = PROFILE_EXTENT_MIN
    profile_extent_max: float = PROFILE_EXTENT_MAX
    profile_family_loss_weight: float = 1.0
    profile_parameter_loss_weight: float = 1.0

    def validate(self):
        expected_categorical = v4_categorical_contract_metadata()
        super().validate()
        expected_metadata = (
            ("model_name", CONSTRAINED_PROFILE_MODEL_NAME),
            (
                "model_config_version",
                CONSTRAINED_PROFILE_MODEL_CONFIG_VERSION,
            ),
            (
                "decoder_contract_version",
                CONSTRAINED_PROFILE_DECODER_CONTRACT_VERSION,
            ),
            ("checkpoint_version", CONSTRAINED_PROFILE_CHECKPOINT_VERSION),
            ("profile_family_order", CONSTRAINED_PROFILE_FAMILY_ORDER),
            (
                "learned_geometry_channel_indices",
                V4_LEARNED_GEOMETRY_CHANNEL_INDICES,
            ),
            ("canonical_plane_contract_id", CANONICAL_PLANE_CONTRACT_ID),
            ("base_geometry_contract_id", V4_BASE_GEOMETRY_CONTRACT_ID),
            ("categorical_selection_contract_id", V4_CATEGORICAL_CONTRACT_ID),
            ("categorical_selection_contract", expected_categorical),
            ("profile_extent_min", PROFILE_EXTENT_MIN),
            ("profile_extent_max", PROFILE_EXTENT_MAX),
        )
        for name, expected in expected_metadata:
            if getattr(self, name) != expected:
                raise FlatBaselineConfigurationError(
                    "{} must equal {!r} for the V4 contract".format(
                        name, expected
                    )
                )
        _finite_nonnegative(
            self.profile_family_loss_weight,
            "profile_family_loss_weight",
        )
        _finite_nonnegative(
            self.profile_parameter_loss_weight,
            "profile_parameter_loss_weight",
        )
        if self.profile_family_loss_weight != self.categorical_loss_weight:
            raise FlatBaselineConfigurationError(
                "profile-family loss weight must match categorical loss weight"
            )
        if self.profile_parameter_loss_weight != self.geometry_loss_weight:
            raise FlatBaselineConfigurationError(
                "profile-parameter loss weight must match geometry loss weight"
            )


def _finite_nonnegative(value, name):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or value < 0.0
    ):
        raise FlatBaselineConfigurationError(
            "{} must be finite and nonnegative".format(name)
        )
