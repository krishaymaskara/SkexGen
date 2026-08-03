"""Single strict graph-v1 pilot checkpoint schema."""

from __future__ import annotations

from collections.abc import Mapping

from prototype.axis_geometry import axis_geometry_metadata
from prototype.flat_baseline.constrained_v6_checkpoint import canonical_grammar_metadata
from prototype.model_data.vocab import NODE_TYPES
from prototype.node_conditioned_categories import v4_categorical_contract_metadata

from .config import (
    GRAPH_CORRECTION_HYPOTHESIS,
    GRAPH_SCIENTIFIC_CORRECTION_INDEX,
    GRAPH_SCIENTIFIC_CORRECTION_LIMIT,
    PARENT_GRAPH_COMMIT,
    PARENT_GRAPH_PILOT_JOB,
    GraphV1Config,
)
from .graph_contract import graph_contract_metadata
from .model import (
    graph_decoder_parameter_count,
    main_pair_mlp_parameter_count,
    position_bias_parameter_count,
)
from .provenance import (
    GRAPH_SOURCE_BRANCH,
    GraphProvenanceError,
    validate_graph_source_provenance,
)


GRAPH_CHECKPOINT_FIELDS = frozenset({
    "checkpoint_version", "checkpoint_kind", "model_name",
    "model_config_version", "decoder_contract_version",
    "graph_contract_version", "graph_contract", "graph_vocabulary",
    "graph_direction_convention", "active_pair_construction",
    "minimal_mask_contract", "pair_feature_contract",
    "pair_decoder_architecture", "loss_normalization",
    "inherited_node_path", "node_grammar_contract",
    "categorical_selection_contract", "axis_geometry_contract",
    "node_vocabulary", "model_config", "parameter_counts",
    "scientific_correction_index", "scientific_correction_limit",
    "parent_graph_commit", "parent_graph_pilot_job", "correction_hypothesis",
    "pilot_config", "training_config", "model_state", "optimizer_state",
    "vq_state", "rng_state", "epoch", "global_step",
    "examples_processed", "configured_maximum_steps", "completed_epochs",
    "training_partition_state", "validation_partition_state",
    "validation_summaries", "source_provenance",
    "systematic_partition_accessed", "test_partition_accessed",
})


class GraphCheckpointError(ValueError):
    def __init__(self, detail):
        self.code = "malformed_graph_checkpoint"
        self.detail = detail
        super().__init__("{}: {}".format(self.code, detail))


def graph_model_metadata(model, config):
    config.validate()
    graph = graph_contract_metadata()
    total = sum(parameter.numel() for parameter in model.parameters())
    decoder = graph_decoder_parameter_count(model)
    main_pair = main_pair_mlp_parameter_count(model)
    position_bias = position_bias_parameter_count(model)
    return {
        "checkpoint_version": config.checkpoint_version,
        "model_name": config.model_name,
        "model_config_version": config.model_config_version,
        "decoder_contract_version": config.decoder_contract_version,
        "graph_contract_version": 1,
        "graph_contract": graph,
        "graph_vocabulary": graph["edge_vocabulary"],
        "graph_direction_convention": graph["direction_convention"],
        "active_pair_construction": graph["active_pair_construction"],
        "minimal_mask_contract": graph["minimal_mask_contract"],
        "pair_feature_contract": list(config.pair_feature_order),
        "pair_decoder_architecture": {
            "kind": "shared_ordered_pair_mlp-plus-directed-position-bias",
            "input_width": 7 * config.model_dim + 1,
            "hidden_width": config.pair_hidden_dim,
            "output_width": len(graph["edge_vocabulary"]),
            "activation": "gelu",
            "position_bias_rank": config.position_bias_rank,
            "position_factor_rows": config.max_nodes,
            "position_factor_directionality": "distinct-source-and-destination",
            "combination": "main-pair-logits-plus-position-logits-before-mask",
        },
        "loss_normalization": config.graph_loss_normalization,
        "inherited_node_path": config.inherited_node_path_id,
        "node_grammar_contract": canonical_grammar_metadata(),
        "categorical_selection_contract": v4_categorical_contract_metadata(),
        "axis_geometry_contract": axis_geometry_metadata(),
        "node_vocabulary": list(NODE_TYPES.tokens),
        "model_config": config.to_dict(),
        "scientific_correction_index": GRAPH_SCIENTIFIC_CORRECTION_INDEX,
        "scientific_correction_limit": GRAPH_SCIENTIFIC_CORRECTION_LIMIT,
        "parent_graph_commit": PARENT_GRAPH_COMMIT,
        "parent_graph_pilot_job": PARENT_GRAPH_PILOT_JOB,
        "correction_hypothesis": GRAPH_CORRECTION_HYPOTHESIS,
        "parameter_counts": {
            "graph_model_total": total,
            "graph_edge_decoder": decoder,
            "main_pair_mlp": main_pair,
            "position_bias": position_bias,
            "source_position_factor": model.source_position_factor.weight.numel(),
            "destination_position_factor": (
                model.destination_position_factor.weight.numel()
            ),
            "position_class_projection": (
                model.position_class_projection.weight.numel()
            ),
            "frozen_v6_structural_heads": 3496,
            "frozen_v6_total": 32866,
            "absolute_total_difference": abs(32866 - total),
        },
    }


def validate_graph_checkpoint(
    payload,
    model,
    model_config,
    pilot_config,
    training_config,
    torch_module,
    *,
    expected_source_provenance
):
    if not isinstance(payload, Mapping):
        raise GraphCheckpointError("payload must be a mapping")
    missing = GRAPH_CHECKPOINT_FIELDS - set(payload)
    extra = set(payload) - GRAPH_CHECKPOINT_FIELDS
    if missing or extra:
        raise GraphCheckpointError("field mismatch missing={} extra={}".format(
            sorted(missing), sorted(extra)
        ))
    expected = graph_model_metadata(model, model_config)
    for name, value in expected.items():
        if payload[name] != value:
            raise GraphCheckpointError("metadata field {} differs".format(name))
    if payload["checkpoint_kind"] != "pilot_fixed_epoch":
        raise GraphCheckpointError("checkpoint kind differs")
    if payload["pilot_config"] != pilot_config.to_dict():
        raise GraphCheckpointError("pilot config differs")
    if payload["training_config"] != training_config.to_dict():
        raise GraphCheckpointError("training config differs")
    for name in ("model_state", "optimizer_state", "vq_state", "rng_state"):
        if not isinstance(payload[name], Mapping):
            raise GraphCheckpointError("{} must be a mapping".format(name))
    expected_state_keys = list(model.state_dict().keys())
    if list(payload["model_state"].keys()) != expected_state_keys:
        raise GraphCheckpointError("model state keys or ordering differ")
    for name in ("epoch", "global_step", "examples_processed", "configured_maximum_steps", "completed_epochs"):
        value = payload[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise GraphCheckpointError("{} must be nonnegative integer".format(name))
    for name in ("systematic_partition_accessed", "test_partition_accessed"):
        if payload[name] is not False:
            raise GraphCheckpointError("{} must be exactly false".format(name))
    provenance = payload["source_provenance"]
    expected_provenance = expected_source_provenance
    expected_commit = (
        expected_provenance.get("git_commit")
        if isinstance(expected_provenance, Mapping) else None
    )
    expected_digest = (
        expected_provenance.get("source_tree_sha256")
        if isinstance(expected_provenance, Mapping) else None
    )
    try:
        validate_graph_source_provenance(
            provenance,
            expected_commit=expected_commit,
            expected_branch=GRAPH_SOURCE_BRANCH,
            expected_digest=expected_digest,
        )
    except GraphProvenanceError as exc:
        raise GraphCheckpointError(str(exc))
    for name, value in payload["model_state"].items():
        if not isinstance(name, str) or not torch_module.is_tensor(value):
            raise GraphCheckpointError("model state is malformed")
    expected_vq_keys = list(model.vq.state_dict().keys())
    if list(payload["vq_state"].keys()) != expected_vq_keys:
        raise GraphCheckpointError("VQ state keys or ordering differ")
    for name, value in payload["vq_state"].items():
        model_value = payload["model_state"].get("vq." + name)
        if (
            not torch_module.is_tensor(value)
            or not torch_module.is_tensor(model_value)
            or value.dtype != model_value.dtype
            or tuple(value.shape) != tuple(model_value.shape)
            or not torch_module.equal(value, model_value)
        ):
            raise GraphCheckpointError("VQ state differs from model state")
    return payload
