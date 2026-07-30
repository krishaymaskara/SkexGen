"""Teacher-forced V2 flat model with constrained controlled profiles."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from prototype.constrained_profile_decoder import (
    ConstrainedProfileHeads,
)
from prototype.model_data.geometry import (
    GEOMETRY_CHANNELS,
    GEOMETRY_WIDTH,
    NORMALIZED_MAX,
    NORMALIZED_MIN,
)
from prototype.model_data.vocab import (
    BOOLEAN_MODES,
    DIRECTIONS,
    LOOP_ROLES,
    NODE_TYPES,
    OPERATION_TYPES,
    REFERENCE_PLANES,
)
from prototype.profile_geometry_torch import (
    TensorProfileTargets,
    canonicalize_profile_tensors,
    constrain_profile_parameters,
)

from .constrained_v2_config import ConstrainedProfileV2Config
from .model import FlatMixedVQModel


NON_PROFILE_GEOMETRY_INDICES = (
    *tuple(range(0, 9)),
    *tuple(range(33, 39)),
)
NON_PROFILE_GEOMETRY_CHANNELS = tuple(
    GEOMETRY_CHANNELS[index] for index in NON_PROFILE_GEOMETRY_INDICES
)
NON_PROFILE_GEOMETRY_WIDTH = len(NON_PROFILE_GEOMETRY_INDICES)

REMAINING_CATEGORICAL_TARGET_INDICES = (0, 1, 2, 3, 8)
REMAINING_CATEGORICAL_FIELDS = (
    "operation_type",
    "boolean_mode",
    "direction",
    "reference_plane",
    "loop_role",
)
_REMAINING_CATEGORICAL_VOCABULARIES = (
    OPERATION_TYPES,
    BOOLEAN_MODES,
    DIRECTIONS,
    REFERENCE_PLANES,
    LOOP_ROLES,
)

if NON_PROFILE_GEOMETRY_WIDTH != 15:
    raise AssertionError("the V2 non-profile geometry contract must have width 15")


@dataclass(frozen=True)
class ConstrainedProfileV2Output:
    """Teacher-forced V2 predictions and training-only canonical profiles."""

    decoded_states: torch.Tensor
    node_type_logits: torch.Tensor
    categorical_logits: tuple
    profile_family_logits: torch.Tensor
    raw_profile_parameters: torch.Tensor
    constrained_profile_parameters: torch.Tensor
    non_profile_geometry: torch.Tensor
    non_profile_geometry_scattered: torch.Tensor
    non_profile_geometry_mask: torch.Tensor
    training_primitive_type_ids: torch.Tensor
    training_profile_geometry: torch.Tensor
    training_profile_geometry_mask: torch.Tensor
    training_geometry: torch.Tensor
    training_geometry_mask: torch.Tensor
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


def scatter_non_profile_geometry(values: torch.Tensor) -> torch.Tensor:
    """Scatter the frozen 15-channel order into the serialized width 39."""

    if not torch.is_tensor(values):
        raise TypeError("non-profile geometry must be a PyTorch tensor")
    if not values.dtype.is_floating_point:
        raise TypeError("non-profile geometry must be floating point")
    if values.dim() < 1 or values.size(-1) != NON_PROFILE_GEOMETRY_WIDTH:
        raise ValueError(
            "non-profile geometry must have shape [..., 15]"
        )
    if not torch.isfinite(values).all():
        raise ValueError("non-profile geometry must be finite")
    if values.numel() and (
        (values < NORMALIZED_MIN).any()
        or (values > NORMALIZED_MAX).any()
    ):
        raise ValueError("non-profile geometry is outside normalized bounds")
    result = values.new_zeros(values.shape[:-1] + (GEOMETRY_WIDTH,))
    result[..., NON_PROFILE_GEOMETRY_INDICES] = values
    return result.contiguous()


def select_non_profile_geometry(values: torch.Tensor) -> torch.Tensor:
    """Gather the frozen non-profile channels from serialized geometry."""

    if not torch.is_tensor(values):
        raise TypeError("serialized geometry must be a PyTorch tensor")
    if values.dim() < 1 or values.size(-1) != GEOMETRY_WIDTH:
        raise ValueError("serialized geometry must have shape [..., 39]")
    return values[..., NON_PROFILE_GEOMETRY_INDICES].contiguous()


def validate_profile_targets_against_tensor_target(
    profile_targets, target
) -> None:
    """Confirm compact targets exactly encode the authoritative tensor target."""

    if not isinstance(profile_targets, TensorProfileTargets):
        raise TypeError(
            "profile_targets must come from profile_targets_for_loss()"
        )
    expected = target["node_mask"].shape
    if (
        profile_targets.family_ids.shape != expected
        or profile_targets.parameters.shape != expected + (3,)
        or profile_targets.sketch_mask.shape != expected
    ):
        raise ValueError("profile targets do not align with decoder targets")
    if (
        profile_targets.family_ids.device != target["node_mask"].device
        or profile_targets.parameters.device != target["geometry"].device
        or profile_targets.sketch_mask.device != target["node_mask"].device
    ):
        raise ValueError("profile and decoder targets must share one device")
    if profile_targets.parameters.dtype != target["geometry"].dtype:
        raise TypeError("profile parameters must match target geometry dtype")
    authoritative_sketch_mask = (
        target["node_mask"]
        & (target["node_type_ids"] == NODE_TYPES.id("sketch"))
    )
    if not torch.equal(
        profile_targets.sketch_mask, authoritative_sketch_mask
    ):
        raise ValueError(
            "profile sketch mask disagrees with authoritative node structure"
        )
    canonical = canonicalize_profile_tensors(
        profile_targets.family_ids,
        profile_targets.parameters,
        profile_targets.sketch_mask,
    )
    selected = profile_targets.sketch_mask
    if selected.any():
        if not torch.equal(
            canonical.primitive_type_ids[selected],
            target["categorical_attributes"][..., 4:8][selected],
        ):
            raise ValueError(
                "profile family IDs disagree with authoritative primitive slots"
            )
        target_profile_mask = target["geometry_mask"][..., 9:33]
        if not torch.equal(
            canonical.geometry_mask[..., 9:33][selected],
            target_profile_mask[selected],
        ):
            raise ValueError(
                "compact profile targets disagree with authoritative mask"
            )
        applicable = canonical.geometry_mask & selected.unsqueeze(-1)
        if not torch.equal(
            canonical.geometry[applicable],
            target["geometry"][applicable],
        ):
            raise ValueError(
                "compact profile targets disagree with authoritative geometry"
            )


class ConstrainedProfileV2Model(FlatMixedVQModel):
    """V2 successor retaining V1 encoder/VQ/causal-decoder internals."""

    def __init__(self, config=None):
        resolved = config or ConstrainedProfileV2Config()
        if not isinstance(resolved, ConstrainedProfileV2Config):
            raise TypeError("V2 model requires ConstrainedProfileV2Config")
        super().__init__(resolved)

        # V1 created these output modules during shared initialization. Remove
        # them completely so no primitive-slot or 39-channel output parameters
        # remain in the V2 state dictionary.
        del self.categorical_heads
        del self.geometry_head

        d_model = self.config.model_dim
        self.remaining_categorical_heads = nn.ModuleList(
            nn.Linear(d_model, len(vocabulary.tokens))
            for vocabulary in _REMAINING_CATEGORICAL_VOCABULARIES
        )
        self.profile_heads = ConstrainedProfileHeads(d_model)
        self.non_profile_geometry_head = nn.Linear(
            d_model, NON_PROFILE_GEOMETRY_WIDTH
        )

    def forward(
        self,
        categorical_ids,
        geometry,
        geometry_mask,
        padding_mask,
        target,
        profile_targets,
    ):
        self._validate_inputs(
            categorical_ids, geometry, geometry_mask, padding_mask, target
        )
        self._validate_profile_targets(profile_targets, target)
        encoded = self.encode_to_memory(
            categorical_ids, geometry, geometry_mask, padding_mask
        )
        decoded_states = self._teacher_forced_states(
            categorical_ids, padding_mask, target, encoded.memory
        )

        categorical_logits = tuple(
            head(decoded_states)
            for head in self.remaining_categorical_heads
        )
        profile_output = self.profile_heads(decoded_states)
        constrained_parameters = constrain_profile_parameters(
            profile_output.raw_parameters,
            profile_targets.family_ids,
            profile_targets.sketch_mask,
            extent_min=self.config.profile_extent_min,
            extent_max=self.config.profile_extent_max,
        )
        canonical = canonicalize_profile_tensors(
            profile_targets.family_ids,
            constrained_parameters,
            profile_targets.sketch_mask,
            extent_min=self.config.profile_extent_min,
            extent_max=self.config.profile_extent_max,
        )

        non_profile_geometry = torch.tanh(
            self.non_profile_geometry_head(decoded_states)
        )
        scattered = scatter_non_profile_geometry(non_profile_geometry)
        selected_mask = (
            select_non_profile_geometry(target["geometry_mask"])
            & target["node_mask"].unsqueeze(-1)
        )
        scattered_mask = _scatter_non_profile_mask(selected_mask)
        training_geometry = scattered + canonical.geometry
        training_geometry_mask = scattered_mask | canonical.geometry_mask

        relations = self.decode_relations(
            decoded_states, target["node_mask"]
        )
        return ConstrainedProfileV2Output(
            decoded_states,
            self.node_type_head(decoded_states),
            categorical_logits,
            profile_output.family_logits,
            profile_output.raw_parameters,
            constrained_parameters,
            non_profile_geometry,
            scattered,
            selected_mask,
            canonical.primitive_type_ids,
            canonical.geometry,
            canonical.geometry_mask,
            training_geometry,
            training_geometry_mask,
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

    def decode_prefix(self, *args, **kwargs):
        del args, kwargs
        raise NotImplementedError(
            "V2 autonomous or prefix prediction is outside Stage 2B"
        )

    def _teacher_forced_states(
        self, categorical_ids, padding_mask, target, memory
    ):
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
        return self._decode_embedded_prefix(
            shifted, memory, decoder_valid
        )

    def _validate_profile_targets(self, profile_targets, target):
        validate_profile_targets_against_tensor_target(
            profile_targets, target
        )


def _scatter_non_profile_mask(mask):
    if (
        not torch.is_tensor(mask)
        or mask.dtype != torch.bool
        or mask.dim() < 1
        or mask.size(-1) != NON_PROFILE_GEOMETRY_WIDTH
    ):
        raise ValueError("non-profile geometry mask must have shape [..., 15]")
    result = torch.zeros(
        mask.shape[:-1] + (GEOMETRY_WIDTH,),
        dtype=torch.bool,
        device=mask.device,
    )
    result[..., NON_PROFILE_GEOMETRY_INDICES] = mask
    return result.contiguous()
