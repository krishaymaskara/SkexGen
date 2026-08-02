"""Numerically V4-equivalent losses for the V5 model identity."""

from __future__ import annotations

from dataclasses import fields

from .constrained_v4_config import ConstrainedProfileV4Config
from .constrained_v4 import ConstrainedProfileV4Output
from .constrained_v4_losses import constrained_profile_v4_loss
from .constrained_v5 import ConstrainedProfileV5Output
from .constrained_v5_config import ConstrainedProfileV5Config


def constrained_profile_v5_loss(output, target, profile_targets, config):
    if not isinstance(output, ConstrainedProfileV5Output):
        raise TypeError("output must be ConstrainedProfileV5Output")
    if not isinstance(config, ConstrainedProfileV5Config):
        raise TypeError("config must be ConstrainedProfileV5Config")
    config.validate()
    v4_names = {field.name for field in fields(ConstrainedProfileV4Config)}
    identity = {
        "model_name", "model_config_version", "decoder_contract_version",
        "checkpoint_version", "profile_family_order",
        "learned_geometry_channel_indices", "canonical_plane_contract_id",
        "base_geometry_contract_id", "categorical_selection_contract_id",
        "categorical_selection_contract",
    }
    values = {
        name: getattr(config, name)
        for name in v4_names - identity
    }
    compatible_config = ConstrainedProfileV4Config(**values)
    compatible_output = ConstrainedProfileV4Output(
        **{
            field.name: getattr(output, field.name)
            for field in fields(ConstrainedProfileV4Output)
        }
    )
    return constrained_profile_v4_loss(
        compatible_output, target, profile_targets, compatible_config
    )
