"""Teacher-forced V3 flat model with constrained controlled profiles."""

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
from prototype.reference_plane_geometry_torch import (
    canonicalize_reference_plane_tensors,
    validate_authoritative_reference_plane_tensors,
)

from .constrained_v3_config import (
    V3_LEARNED_GEOMETRY_CHANNEL_INDICES,
    ConstrainedProfileV3Config,
)
from .model import FlatMixedVQModel


REMAINING_GEOMETRY_INDICES = V3_LEARNED_GEOMETRY_CHANNEL_INDICES
REMAINING_GEOMETRY_CHANNELS = tuple(
    GEOMETRY_CHANNELS[index] for index in REMAINING_GEOMETRY_INDICES
)
REMAINING_GEOMETRY_WIDTH = len(REMAINING_GEOMETRY_INDICES)

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

if REMAINING_GEOMETRY_WIDTH != 6 or REMAINING_GEOMETRY_INDICES != tuple(range(33, 39)):
    raise AssertionError("the V3 learned geometry contract must be channels 33-38")


@dataclass(frozen=True)
class ConstrainedProfileV3Output:
    """Teacher-forced V3 predictions and training-only canonical profiles."""

    decoded_states: torch.Tensor
    node_type_logits: torch.Tensor
    categorical_logits: tuple
    profile_family_logits: torch.Tensor
    raw_profile_parameters: torch.Tensor
    constrained_profile_parameters: torch.Tensor
    remaining_geometry: torch.Tensor
    remaining_geometry_scattered: torch.Tensor
    remaining_geometry_mask: torch.Tensor
    training_reference_plane_geometry: torch.Tensor
    training_reference_plane_geometry_mask: torch.Tensor
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


@dataclass(frozen=True)
class ConstrainedProfileV3PrefixOutput:
    """V3 predictions for BOS plus a generated-record prefix."""

    decoded_states: torch.Tensor
    node_type_logits: torch.Tensor
    categorical_logits: tuple
    profile_family_logits: torch.Tensor
    raw_profile_parameters: torch.Tensor
    remaining_geometry: torch.Tensor


def scatter_remaining_geometry(values: torch.Tensor) -> torch.Tensor:
    """Scatter learned V3 channels 33-38 into serialized width 39."""

    if not torch.is_tensor(values):
        raise TypeError("remaining geometry must be a PyTorch tensor")
    if not values.dtype.is_floating_point:
        raise TypeError("remaining geometry must be floating point")
    if values.dim() < 1 or values.size(-1) != REMAINING_GEOMETRY_WIDTH:
        raise ValueError(
            "remaining geometry must have shape [..., 6]"
        )
    if not torch.isfinite(values).all():
        raise ValueError("remaining geometry must be finite")
    if values.numel() and (
        (values < NORMALIZED_MIN).any()
        or (values > NORMALIZED_MAX).any()
    ):
        raise ValueError("remaining geometry is outside normalized bounds")
    result = values.new_zeros(values.shape[:-1] + (GEOMETRY_WIDTH,))
    result[..., REMAINING_GEOMETRY_INDICES] = values
    return result.contiguous()


def select_remaining_geometry(values: torch.Tensor) -> torch.Tensor:
    """Gather serialized channels 33-38 in the exact learned V3 order."""

    if not torch.is_tensor(values):
        raise TypeError("serialized geometry must be a PyTorch tensor")
    if values.dim() < 1 or values.size(-1) != GEOMETRY_WIDTH:
        raise ValueError("serialized geometry must have shape [..., 39]")
    return values[..., REMAINING_GEOMETRY_INDICES].contiguous()


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


class ConstrainedProfileV3Model(FlatMixedVQModel):
    """V3 successor retaining V1 encoder/VQ/causal-decoder internals."""

    def __init__(self, config=None):
        resolved = config or ConstrainedProfileV3Config()
        if not isinstance(resolved, ConstrainedProfileV3Config):
            raise TypeError("V3 model requires ConstrainedProfileV3Config")
        super().__init__(resolved)

        # V1 created these output modules during shared initialization. Remove
        # them completely so no primitive-slot or 39-channel output parameters
        # remain in the V3 state dictionary.
        del self.categorical_heads
        del self.geometry_head

        d_model = self.config.model_dim
        self.remaining_categorical_heads = nn.ModuleList(
            nn.Linear(d_model, len(vocabulary.tokens))
            for vocabulary in _REMAINING_CATEGORICAL_VOCABULARIES
        )
        self.profile_heads = ConstrainedProfileHeads(d_model)
        self.remaining_geometry_head = nn.Linear(
            d_model, REMAINING_GEOMETRY_WIDTH
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

        remaining_geometry = torch.tanh(
            self.remaining_geometry_head(decoded_states)
        )
        scattered = scatter_remaining_geometry(remaining_geometry)
        selected_mask = (
            select_remaining_geometry(target["geometry_mask"])
            & target["node_mask"].unsqueeze(-1)
        )
        scattered_mask = _scatter_remaining_mask(selected_mask)
        canonical_plane = canonicalize_reference_plane_tensors(
            target["node_type_ids"],
            target["categorical_attributes"][..., 3],
            remaining_geometry,
            target["node_mask"],
        )
        training_geometry = (
            canonical_plane.geometry + canonical.geometry + scattered
        )
        training_geometry_mask = (
            canonical_plane.geometry_mask
            | canonical.geometry_mask
            | scattered_mask
        )

        relations = self.decode_relations(
            decoded_states, target["node_mask"]
        )
        return ConstrainedProfileV3Output(
            decoded_states,
            self.node_type_head(decoded_states),
            categorical_logits,
            profile_output.family_logits,
            profile_output.raw_parameters,
            constrained_parameters,
            remaining_geometry,
            scattered,
            selected_mask,
            canonical_plane.geometry,
            canonical_plane.geometry_mask,
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

    def decode_prefix(
        self,
        memory,
        categorical_prefix,
        geometry_prefix,
        geometry_mask_prefix,
        prefix_mask=None,
    ):
        """Decode BOS plus complete preceding V3 node records."""

        batch_size, prefix_length = _validate_v3_prefix_inputs(
            self,
            memory,
            categorical_prefix,
            geometry_prefix,
            geometry_mask_prefix,
            prefix_mask,
        )
        if prefix_mask is None:
            prefix_mask = torch.ones(
                batch_size,
                prefix_length,
                dtype=torch.bool,
                device=memory.device,
            )
        prefix_content = self._record_content(
            categorical_prefix,
            geometry_prefix,
            geometry_mask_prefix,
        )
        bos = self.bos.view(1, 1, -1).expand(batch_size, 1, -1)
        shifted = torch.cat((bos, prefix_content), dim=1)
        decoder_valid = torch.cat(
            (
                torch.ones(
                    batch_size,
                    1,
                    dtype=torch.bool,
                    device=memory.device,
                ),
                prefix_mask,
            ),
            dim=1,
        )
        decoded_states = self._decode_embedded_prefix(
            shifted, memory, decoder_valid
        )
        profile_output = self.profile_heads(decoded_states)
        return ConstrainedProfileV3PrefixOutput(
            decoded_states,
            self.node_type_head(decoded_states),
            tuple(
                head(decoded_states)
                for head in self.remaining_categorical_heads
            ),
            profile_output.family_logits,
            profile_output.raw_parameters,
            torch.tanh(self.remaining_geometry_head(decoded_states)),
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
        validate_authoritative_reference_plane_tensors(
            target["node_type_ids"],
            target["categorical_attributes"],
            target["geometry"],
            target["geometry_mask"],
            target["node_mask"],
        )


def _scatter_remaining_mask(mask):
    if (
        not torch.is_tensor(mask)
        or mask.dtype != torch.bool
        or mask.dim() < 1
        or mask.size(-1) != REMAINING_GEOMETRY_WIDTH
    ):
        raise ValueError("remaining geometry mask must have shape [..., 6]")
    result = torch.zeros(
        mask.shape[:-1] + (GEOMETRY_WIDTH,),
        dtype=torch.bool,
        device=mask.device,
    )
    result[..., REMAINING_GEOMETRY_INDICES] = mask
    return result.contiguous()


def _validate_v3_prefix_inputs(
    model,
    memory,
    categorical_prefix,
    geometry_prefix,
    geometry_mask_prefix,
    prefix_mask,
):
    if (
        not torch.is_tensor(memory)
        or memory.dim() != 3
        or memory.size(1) != model.config.latent_tokens
        or memory.size(2) != model.config.model_dim
        or memory.size(0) == 0
    ):
        raise ValueError(
            "memory must have shape [B, latent_tokens, model_dim]"
        )
    if not memory.dtype.is_floating_point:
        raise TypeError("memory must be floating point")
    if memory.dtype != model.bos.dtype:
        raise TypeError("memory dtype must match model dtype")
    if memory.device != model.bos.device:
        raise ValueError("memory device must match model device")
    if not torch.isfinite(memory).all():
        raise ValueError("memory must be finite")

    batch_size = memory.size(0)
    if not torch.is_tensor(categorical_prefix):
        raise TypeError("categorical prefix must be a tensor")
    if categorical_prefix.dim() != 3:
        raise ValueError("categorical prefix must have shape [B, P, 10]")
    prefix_length = categorical_prefix.size(1)
    if categorical_prefix.shape != (batch_size, prefix_length, 10):
        raise ValueError("categorical prefix must have shape [B, P, 10]")
    if categorical_prefix.dtype != torch.long:
        raise TypeError("categorical prefix must use torch.long")
    expected_geometry = (batch_size, prefix_length, GEOMETRY_WIDTH)
    if (
        not torch.is_tensor(geometry_prefix)
        or tuple(geometry_prefix.shape) != expected_geometry
    ):
        raise ValueError("geometry prefix must have shape [B, P, 39]")
    if (
        not torch.is_tensor(geometry_mask_prefix)
        or tuple(geometry_mask_prefix.shape) != expected_geometry
    ):
        raise ValueError("geometry-mask prefix must align with geometry")
    if not geometry_prefix.dtype.is_floating_point:
        raise TypeError("geometry prefix must be floating point")
    if geometry_prefix.dtype != memory.dtype:
        raise TypeError("geometry prefix dtype must match memory dtype")
    if geometry_mask_prefix.dtype != torch.bool:
        raise TypeError("geometry-mask prefix must use torch.bool")
    if prefix_length + 1 > model.config.max_nodes:
        raise ValueError("decoded prefix exceeds configured max_nodes")
    if prefix_mask is not None and (
        not torch.is_tensor(prefix_mask)
        or tuple(prefix_mask.shape) != (batch_size, prefix_length)
    ):
        raise ValueError("prefix mask must have shape [B, P]")
    if prefix_mask is not None and prefix_mask.dtype != torch.bool:
        raise TypeError("prefix mask must use torch.bool")
    tensors = (
        categorical_prefix,
        geometry_prefix,
        geometry_mask_prefix,
    )
    if prefix_mask is not None:
        tensors = tensors + (prefix_mask,)
    if any(value.device != memory.device for value in tensors):
        raise ValueError("decoder prefix tensors must share memory device")
    if not torch.isfinite(geometry_prefix).all():
        raise ValueError("geometry prefix must be finite")
    return batch_size, prefix_length
