"""Thin C5 GE1 encoder-selection wrapper and matched model construction."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .config import GE1Config, frozen_encoder_config
from .encoders import FlatProgramEncoder, TypedGraphProgramEncoder
from .shared_decoder import SharedGE1Decoder, copied_shared_decoder


@dataclass(frozen=True)
class GE1TeacherForcedOutput:
    decoder_output: object
    prequant: torch.Tensor


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
        self.config = config
        self.encoder = encoder
        self.decoder = decoder

    def encode(self, encoder_input):
        """Adapt the selected input only before obtaining common memory."""

        if not isinstance(encoder_input, dict):
            raise TypeError("encoder_input must be a tensor dictionary")
        return self.encoder(**encoder_input)

    def forward(
        self, encoder_input, *, node_counts, node_count_source
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
        return GE1TeacherForcedOutput(output, encoded.prequant)


def _construct_with_local_seed(seed, constructor):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return constructor()


def canonical_shared_decoder(seed):
    """Construct the arm-independent canonical decoder exactly once."""

    frozen_encoder_config("flat", seed)
    return _construct_with_local_seed(seed, SharedGE1Decoder)


def build_ge1_model(config, canonical_decoder=None):
    """Build one arm from its frozen configuration."""

    if not isinstance(config, GE1Config):
        raise TypeError("config must be GE1Config")
    config.validate()
    decoder = (
        copied_shared_decoder(canonical_decoder)
        if canonical_decoder is not None
        else canonical_shared_decoder(config.seed)
    )
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
    seed=2026, encoder_order=("flat", "typed_graph")
):
    """Build both arms from one canonical decoder, independent of arm order."""

    order = tuple(encoder_order)
    if set(order) != {"flat", "typed_graph"} or len(order) != 2:
        raise ValueError("encoder_order must contain flat and typed_graph once")
    canonical = canonical_shared_decoder(seed)
    models = {}
    for arm in order:
        config = frozen_encoder_config(arm, seed)
        models[arm] = build_ge1_model(config, canonical)
    return models["flat"], models["typed_graph"]
