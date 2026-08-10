"""Strict C5 GE1 model checkpoint identity and reload contract."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import re

import torch

from .config import (
    BOTTLENECK_MODE,
    CHECKPOINT_SCHEMA,
    MODEL_FAMILY,
    frozen_encoder_config,
)
from .decoder_contract import (
    OUTPUT_POSITION_CONTRACT_VERSION,
    POSITIVE_OPERATION_MAGNITUDE_PARAMETERIZATION,
    SHARED_DECODER_VERSION,
    output_position_contract_metadata,
)
from .model import build_ge1_model


AUTHORITATIVE_OPERATION_TEMPLATE_SHA256 = (
    "a9ac86a6dede054fbbba57e0906b210bab26036c3f5c150b332038f78d2dadb7"
)
CHECKPOINT_FIELDS = frozenset({
    "checkpoint_schema",
    "model_family",
    "arm_identity",
    "encoder_type",
    "encoder_version",
    "encoder_feedforward_width",
    "relation_basis_count",
    "shared_decoder_version",
    "output_position_contract_version",
    "output_position_contract",
    "operation_magnitude_parameterization",
    "bottleneck_mode",
    "config",
    "config_sha256",
    "architectural_sizes",
    "parameter_inventory",
    "buffer_inventory",
    "decoder_state_keys",
    "code_revision",
    "operation_template_manifest_sha256",
    "model_state",
})


class GE1CheckpointError(ValueError):
    def __init__(self, detail):
        self.code = "invalid_ge1_checkpoint"
        self.detail = detail
        super().__init__("{}: {}".format(self.code, detail))


def _config_sha256(config):
    return hashlib.sha256(config.to_json().encode("utf-8")).hexdigest()


def _inventory(named_values):
    return [
        {
            "name": name,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
        for name, value in named_values
    ]


def ge1_checkpoint_payload(model, *, code_revision):
    """Create the exact C5 inference-model payload; no training state exists."""

    if not re.fullmatch(r"[0-9a-f]{40}", code_revision or ""):
        raise GE1CheckpointError("code_revision must be a lowercase commit hash")
    config = model.config
    config.validate()
    return {
        "checkpoint_schema": CHECKPOINT_SCHEMA,
        "model_family": MODEL_FAMILY,
        "arm_identity": config.arm_identity,
        "encoder_type": config.encoder,
        "encoder_version": config.arm_identity,
        "encoder_feedforward_width": config.encoder_feedforward_width,
        "relation_basis_count": config.relation_basis_count,
        "shared_decoder_version": SHARED_DECODER_VERSION,
        "output_position_contract_version": OUTPUT_POSITION_CONTRACT_VERSION,
        "output_position_contract": output_position_contract_metadata(),
        "operation_magnitude_parameterization": (
            config.operation_magnitude_parameterization
        ),
        "bottleneck_mode": BOTTLENECK_MODE,
        "config": config.to_dict(),
        "config_sha256": _config_sha256(config),
        "architectural_sizes": {
            "model_dim": config.model_dim,
            "latent_tokens": config.latent_tokens,
            "bottleneck_dim": config.bottleneck_dim,
            "max_nodes": config.max_nodes,
            "max_operations": config.max_operations,
            "relational_layers": config.relational_layers,
        },
        "parameter_inventory": _inventory(model.named_parameters()),
        "buffer_inventory": _inventory(model.named_buffers()),
        "decoder_state_keys": list(model.decoder.state_dict().keys()),
        "code_revision": code_revision,
        "operation_template_manifest_sha256": (
            AUTHORITATIVE_OPERATION_TEMPLATE_SHA256
        ),
        "model_state": model.state_dict(),
    }


def save_ge1_checkpoint(path, model, *, code_revision):
    torch.save(
        ge1_checkpoint_payload(model, code_revision=code_revision), str(path)
    )


def load_ge1_checkpoint(
    path,
    *,
    expected_code_revision=None,
    expected_operation_magnitude_parameterization=(
        POSITIVE_OPERATION_MAGNITUDE_PARAMETERIZATION
    ),
):
    """Validate exact identity and reconstruct one selected arm strictly."""

    payload = torch.load(str(path), map_location="cpu")
    if not isinstance(payload, Mapping):
        raise GE1CheckpointError("payload must be a mapping")
    missing = CHECKPOINT_FIELDS - set(payload)
    unexpected = set(payload) - CHECKPOINT_FIELDS
    if missing or unexpected:
        raise GE1CheckpointError(
            "field mismatch missing={} unexpected={}".format(
                sorted(missing), sorted(unexpected)
            )
        )
    if expected_code_revision is not None and (
        payload["code_revision"] != expected_code_revision
    ):
        raise GE1CheckpointError("code revision differs")
    encoder = payload["encoder_type"]
    config_values = payload["config"]
    if not isinstance(config_values, Mapping):
        raise GE1CheckpointError("config must be a mapping")
    seed = config_values.get("seed")
    try:
        config = frozen_encoder_config(
            encoder,
            seed,
            operation_magnitude_parameterization=(
                expected_operation_magnitude_parameterization
            ),
        )
    except Exception as exc:
        raise GE1CheckpointError("configuration cannot be reconstructed") from exc
    if (
        payload["operation_magnitude_parameterization"]
        != expected_operation_magnitude_parameterization
    ):
        raise GE1CheckpointError(
            "operation-magnitude parameterization differs"
        )
    expected = ge1_checkpoint_payload(
        build_ge1_model(config), code_revision=payload["code_revision"]
    )
    metadata_fields = CHECKPOINT_FIELDS - {"model_state"}
    for name in metadata_fields:
        if payload[name] != expected[name]:
            raise GE1CheckpointError("metadata field {} differs".format(name))
    state = payload["model_state"]
    if not isinstance(state, Mapping):
        raise GE1CheckpointError("model_state must be a mapping")
    if list(state.keys()) != list(expected["model_state"].keys()):
        raise GE1CheckpointError("model state keys or ordering differ")
    for name, expected_value in expected["model_state"].items():
        value = state[name]
        if (
            not torch.is_tensor(value)
            or value.dtype != expected_value.dtype
            or tuple(value.shape) != tuple(expected_value.shape)
        ):
            raise GE1CheckpointError(
                "model state tensor {} has incompatible shape or dtype".format(
                    name
                )
            )
    model = build_ge1_model(config)
    try:
        model.load_state_dict(state, strict=True)
    except Exception as exc:
        raise GE1CheckpointError("strict model-state reload failed") from exc
    return model, payload
