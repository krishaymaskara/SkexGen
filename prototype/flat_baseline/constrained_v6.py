"""V6 model identity with V4 trainable architecture and prefix grammar evidence."""

from __future__ import annotations

from dataclasses import dataclass

from torch import nn

from prototype.constrained_profile_decoder import ConstrainedProfileHeads

from .constrained_v4 import (
    REMAINING_GEOMETRY_WIDTH,
    ConstrainedProfileV4Model,
    ConstrainedProfileV4Output,
    _REMAINING_CATEGORICAL_VOCABULARIES,
)
from .constrained_v6_config import ConstrainedProfileV6Config
from .model import FlatMixedVQModel


@dataclass(frozen=True)
class ConstrainedProfileV6Output(ConstrainedProfileV4Output):
    """V4 numerical outputs plus the shifted prefix used for V6 reporting."""

    teacher_forced_prefix_node_ids: object


class ConstrainedProfileV6Model(ConstrainedProfileV4Model):
    """Parameter-shape-identical V4 successor with a V6 decoding identity."""

    def __init__(self, config=None):
        resolved = config or ConstrainedProfileV6Config()
        if not isinstance(resolved, ConstrainedProfileV6Config):
            raise TypeError("V6 model requires ConstrainedProfileV6Config")
        resolved.validate()
        FlatMixedVQModel.__init__(self, resolved)
        del self.categorical_heads
        del self.geometry_head
        d_model = self.config.model_dim
        self.remaining_categorical_heads = nn.ModuleList(
            nn.Linear(d_model, len(vocabulary.tokens))
            for vocabulary in _REMAINING_CATEGORICAL_VOCABULARIES
        )
        self.profile_heads = ConstrainedProfileHeads(d_model)
        self.remaining_geometry_head = nn.Linear(d_model, REMAINING_GEOMETRY_WIDTH)

    def forward(
        self,
        categorical_ids,
        geometry,
        geometry_mask,
        padding_mask,
        target,
        profile_targets,
    ):
        output = ConstrainedProfileV4Model.forward(
            self,
            categorical_ids,
            geometry,
            geometry_mask,
            padding_mask,
            target,
            profile_targets,
        )
        prefix = target["node_type_ids"].clone()
        prefix[:, 1:] = target["node_type_ids"][:, :-1]
        prefix[:, 0] = 0
        return ConstrainedProfileV6Output(
            **vars(output), teacher_forced_prefix_node_ids=prefix.contiguous()
        )
