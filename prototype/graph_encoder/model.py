"""Thin C5 GE1 encoder-selection wrapper and matched model construction."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .config import (
    GE1Config,
    frozen_encoder_config,
    legacy_frozen_encoder_config,
    validate_authorized_seed,
)
from .decoder_contract import POSITIVE_OPERATION_MAGNITUDE_PARAMETERIZATION
from .decoder_contract import LEGACY_NODE_GENERATION_IDENTITY
from .encoders import FlatProgramEncoder, TypedGraphProgramEncoder
from .shared_decoder import SharedGE1Decoder, copied_shared_decoder


@dataclass(frozen=True)
class GE1TeacherForcedOutput:
    """GE1-owned wrapper; the frozen `GraphV1Output` is never extended.

    `grid_magnitude_logits` is `None` for both historical scalar identities so
    every existing caller keeps its exact previous shape and semantics.
    """

    decoder_output: object
    prequant: torch.Tensor
    grid_magnitude_logits: object = None


class GE1Model(nn.Module):
    """Select an encoder before the common-memory boundary, then decode."""

    def __init__(self, config, encoder, decoder):
        super().__init__()
        if not isinstance(config, GE1Config):
            raise TypeError("config must be GE1Config")
        config.validate()
        expected_encoder = (
            FlatProgramEncoder
            if config.encoder == "flat"
            else TypedGraphProgramEncoder
        )
        if not isinstance(encoder, expected_encoder):
            raise TypeError("encoder implementation disagrees with configuration")
        if encoder.config != config:
            raise ValueError("encoder state is labeled with a different configuration")
        if not isinstance(decoder, SharedGE1Decoder):
            raise TypeError("decoder must be SharedGE1Decoder")
        if (
            decoder.operation_magnitude_parameterization
            != config.operation_magnitude_parameterization
        ):
            raise ValueError(
                "decoder operation-magnitude parameterization disagrees "
                "with configuration"
            )
        if decoder.node_generation_identity != config.node_generation_identity:
            raise ValueError(
                "decoder node-generation identity disagrees with configuration"
            )
        self.config = config
        self.encoder = encoder
        self.decoder = decoder

    def encode(self, encoder_input):
        """Adapt the selected input only before obtaining common memory."""

        if not isinstance(encoder_input, dict):
            raise TypeError("encoder_input must be a tensor dictionary")
        return self.encoder(**encoder_input)

    def forward(
        self, encoder_input, *, node_counts=None, node_count_source=None
    ):
        encoded = self.encode(encoder_input)
        return self.decoder(
            encoded.memory,
            node_counts=node_counts,
            node_count_source=node_count_source,
        )

    def teacher_forced(self, encoder_input, target, profile_targets):
        """Training/parity-only entry point, separate from inference."""

        encoded = self.encode(encoder_input)
        output = self.decoder.teacher_forced(
            encoded.memory, target, profile_targets
        )
        magnitude_logits = None
        if getattr(self.decoder, "uses_grid_magnitude", False):
            # Applied to the already-computed decoder states, so the ordinal
            # gradient reaches the shared decoder trunk without a second
            # forward pass through it.
            magnitude_logits = self.decoder.grid_magnitude_logits(
                output.decoded_states
            )
        return GE1TeacherForcedOutput(
            output, encoded.prequant, magnitude_logits
        )


def _construct_with_local_seed(seed, constructor):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return constructor()


def canonical_shared_decoder(
    seed,
    operation_magnitude_parameterization=(
        POSITIVE_OPERATION_MAGNITUDE_PARAMETERIZATION
    ),
    node_generation_identity=LEGACY_NODE_GENERATION_IDENTITY,
):
    """Construct the arm-independent canonical decoder exactly once."""

    validate_authorized_seed(seed)
    return _construct_with_local_seed(
        seed,
        lambda: SharedGE1Decoder(
            operation_magnitude_parameterization=(
                operation_magnitude_parameterization
            ),
            node_generation_identity=node_generation_identity,
        ),
    )


def build_ge1_model(config, canonical_decoder=None):
    """Build one arm from its frozen configuration."""

    if not isinstance(config, GE1Config):
        raise TypeError("config must be GE1Config")
    config.validate()
    decoder = (
        copied_shared_decoder(canonical_decoder)
        if canonical_decoder is not None
        else canonical_shared_decoder(
            config.seed,
            config.operation_magnitude_parameterization,
            config.node_generation_identity,
        )
    )
    if (
        decoder.operation_magnitude_parameterization
        != config.operation_magnitude_parameterization
    ):
        raise ValueError(
            "canonical decoder operation-magnitude parameterization differs"
        )
    if decoder.node_generation_identity != config.node_generation_identity:
        raise ValueError("canonical decoder node-generation identity differs")
    encoder_type = (
        FlatProgramEncoder
        if config.encoder == "flat"
        else TypedGraphProgramEncoder
    )
    encoder = _construct_with_local_seed(
        config.seed, lambda: encoder_type(config=config)
    )
    return GE1Model(config, encoder, decoder)


def build_matched_ge1_models(
    seed=2026,
    encoder_order=("flat", "typed_graph"),
    *,
    operation_magnitude_parameterization=(
        POSITIVE_OPERATION_MAGNITUDE_PARAMETERIZATION
    ),
    node_generation_identity=LEGACY_NODE_GENERATION_IDENTITY,
):
    """Build both arms from one canonical decoder, independent of arm order."""

    order = tuple(encoder_order)
    if set(order) != {"flat", "typed_graph"} or len(order) != 2:
        raise ValueError("encoder_order must contain flat and typed_graph once")
    parameterization = operation_magnitude_parameterization
    canonical = canonical_shared_decoder(
        seed, parameterization, node_generation_identity
    )
    models = {}
    for arm in order:
        config = frozen_encoder_config(
            arm,
            seed,
            operation_magnitude_parameterization=parameterization,
            node_generation_identity=node_generation_identity,
        )
        models[arm] = build_ge1_model(config, canonical)
    return models["flat"], models["typed_graph"]


def build_legacy_matched_ge1_models(
    seed=2026, encoder_order=("flat", "typed_graph")
):
    """Build the explicit historical `tanh` pair for immutable protocols."""

    legacy = legacy_frozen_encoder_config("typed_graph", seed)
    return build_matched_ge1_models(
        seed,
        encoder_order,
        operation_magnitude_parameterization=(
            legacy.operation_magnitude_parameterization
        ),
    )
