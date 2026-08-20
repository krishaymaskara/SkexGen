"""Frozen C5 shared-decoder and output-position contracts.

This module is deliberately free of PyTorch imports.  It records the neural
position signals used after the common GE1 memory boundary and distinguishes
them from tensor bookkeeping and from positions forbidden in the graph
encoder.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


SHARED_DECODER_VERSION = "GE1-SHARED-TYPED-EDGE-DECODER-V1"
OUTPUT_POSITION_CONTRACT_VERSION = "GE1-DECODER-OUTPUT-POSITIONS-v1"
COMMON_LOSS_VERSION = "GE1-COMMON-GRAPH-LOSS-v1"
AUTONOMOUS_OUTPUT_VERSION = "GE1-AUTONOMOUS-OUTPUT-v1"
C5_IMPLEMENTATION_SCOPE = "GE1-C5-SHARED-DECODER-PARITY-v1"
LEGACY_OPERATION_MAGNITUDE_PARAMETERIZATION = (
    "GE1-OPERATION-MAGNITUDE-TANH-LEGACY-v1"
)
POSITIVE_OPERATION_MAGNITUDE_PARAMETERIZATION = (
    "GE1-OPERATION-MAGNITUDE-POSITIVE-v1"
)
GRID_ORDINAL_OPERATION_MAGNITUDE_PARAMETERIZATION = (
    "GE1-OPERATION-MAGNITUDE-GRID-ORDINAL-v1"
)
GRID_SOFTMAX_OPERATION_MAGNITUDE_PARAMETERIZATION = (
    "GE1-OPERATION-MAGNITUDE-GRID-SOFTMAX-v1"
)
# Historical identities keep their exact order and membership; each grid
# identity is appended so no existing tuple index or prefix changes.
SCALAR_OPERATION_MAGNITUDE_PARAMETERIZATIONS = (
    LEGACY_OPERATION_MAGNITUDE_PARAMETERIZATION,
    POSITIVE_OPERATION_MAGNITUDE_PARAMETERIZATION,
)
GRID_OPERATION_MAGNITUDE_PARAMETERIZATIONS = (
    GRID_ORDINAL_OPERATION_MAGNITUDE_PARAMETERIZATION,
    GRID_SOFTMAX_OPERATION_MAGNITUDE_PARAMETERIZATION,
)
OPERATION_MAGNITUDE_PARAMETERIZATIONS = (
    SCALAR_OPERATION_MAGNITUDE_PARAMETERIZATIONS
    + GRID_OPERATION_MAGNITUDE_PARAMETERIZATIONS
)
OPERATION_MAGNITUDE_EPSILON_POLICY = "torch.finfo(dtype).tiny"
OPERATION_MAGNITUDE_SERIALIZED_CHANNELS = (37, 38)
OPERATION_MAGNITUDE_COMPACT_CHANNELS = (4, 5)

# Versioned decoder/output/checkpoint identities.  The v1 values are the exact
# historical literals and must never change; the grid identity introduces v2
# so a legacy or positive-sigmoid checkpoint can never be loaded into a grid
# model, or the reverse, without a typed failure.
GRID_SHARED_DECODER_VERSION = "GE1-SHARED-TYPED-EDGE-DECODER-V2"
GRID_OUTPUT_POSITION_CONTRACT_VERSION = "GE1-DECODER-OUTPUT-POSITIONS-v2"
LEGACY_CHECKPOINT_SCHEMA = "GE1-CHECKPOINT-v1"
GRID_CHECKPOINT_SCHEMA = "GE1-CHECKPOINT-v2"
# ADR-0015.  The softmax head's state-dict keys and shapes differ from the
# ordinal head's, so it takes its own decoder version and checkpoint schema
# rather than inheriting the grid-ordinal ones.  Output positions genuinely do
# not change between the two grid identities, so v2 is reused there.
GRID_SOFTMAX_SHARED_DECODER_VERSION = "GE1-SHARED-TYPED-EDGE-DECODER-V3"
GRID_SOFTMAX_CHECKPOINT_SCHEMA = "GE1-CHECKPOINT-v3"

# ADR-0016 adds an independent node-generation identity.  Magnitude and node
# generation are separate axes: the new scientific configuration combines the
# existing grid-softmax head with learned stopping, while every prior
# magnitude identity keeps the exact requested-count path by default.
LEGACY_NODE_GENERATION_IDENTITY = (
    "GE1-REQUESTED-COUNT-GRAMMAR-MASKED-v1"
)
AUTONOMOUS_STOP_NODE_GENERATION_IDENTITY = (
    "GE1-PAD-TERMINATED-UNCONSTRAINED-NODES-v1"
)
NODE_GENERATION_IDENTITIES = (
    LEGACY_NODE_GENERATION_IDENTITY,
    AUTONOMOUS_STOP_NODE_GENERATION_IDENTITY,
)
AUTONOMOUS_STOP_NODE_GENERATION_CONTRACT_VERSION = (
    "GE1-AUTONOMOUS-STOP-NODE-GENERATION-CONTRACT-v1"
)
AUTONOMOUS_STOP_SHARED_DECODER_VERSION = (
    "GE1-SHARED-TYPED-EDGE-DECODER-V4"
)
AUTONOMOUS_STOP_OUTPUT_POSITION_CONTRACT_VERSION = (
    "GE1-DECODER-OUTPUT-POSITIONS-v3"
)
AUTONOMOUS_STOP_CHECKPOINT_SCHEMA = "GE1-CHECKPOINT-v4"


def uses_grid_magnitude(parameterization):
    """Return whether one parameterization decodes from a frozen grid.

    Membership, not equality: both the ordinal and the softmax identity route
    through the grid loss, the grid-only configuration fields, and the
    exclusion of the compact magnitude channels from the scalar mask.  The
    checkpoint and decoder identities deliberately do *not* use this predicate,
    because those must separate the two grid heads.
    """

    return parameterization in GRID_OPERATION_MAGNITUDE_PARAMETERIZATIONS


def uses_grid_softmax_magnitude(parameterization):
    """Return whether one parameterization is the grid-softmax identity."""

    return parameterization == GRID_SOFTMAX_OPERATION_MAGNITUDE_PARAMETERIZATION


def uses_autonomous_stop(node_generation_identity):
    """Return whether node generation learns `<pad>` termination."""

    if node_generation_identity not in NODE_GENERATION_IDENTITIES:
        raise ValueError("unknown node-generation identity")
    return (
        node_generation_identity == AUTONOMOUS_STOP_NODE_GENERATION_IDENTITY
    )


def shared_decoder_version_for(
    parameterization,
    node_generation_identity=LEGACY_NODE_GENERATION_IDENTITY,
):
    """Return the decoder identity implied by a magnitude parameterization."""

    if uses_autonomous_stop(node_generation_identity):
        return AUTONOMOUS_STOP_SHARED_DECODER_VERSION
    if uses_grid_softmax_magnitude(parameterization):
        return GRID_SOFTMAX_SHARED_DECODER_VERSION
    if uses_grid_magnitude(parameterization):
        return GRID_SHARED_DECODER_VERSION
    return SHARED_DECODER_VERSION


def output_position_contract_version_for(
    parameterization,
    node_generation_identity=LEGACY_NODE_GENERATION_IDENTITY,
):
    """Return the output-position identity implied by a parameterization.

    Both grid identities share `...-POSITIONS-v2`: the magnitude readout
    changes, the decoder output positions do not.
    """

    if uses_autonomous_stop(node_generation_identity):
        return AUTONOMOUS_STOP_OUTPUT_POSITION_CONTRACT_VERSION
    if uses_grid_magnitude(parameterization):
        return GRID_OUTPUT_POSITION_CONTRACT_VERSION
    return OUTPUT_POSITION_CONTRACT_VERSION


def checkpoint_schema_for(
    parameterization,
    node_generation_identity=LEGACY_NODE_GENERATION_IDENTITY,
):
    """Return the checkpoint schema implied by a parameterization.

    The softmax identity is checkpoint incompatible with the ordinal one:
    `projections.*.weight` is `[5, model_dim]` rather than `[1, model_dim]`,
    `projections.*.bias` is new, and `first_bias` and `bias_gaps` do not
    exist.  A shared schema would let a governed schema check pass on a
    structurally incompatible state dict, so each grid head owns a schema.
    """

    if uses_autonomous_stop(node_generation_identity):
        return AUTONOMOUS_STOP_CHECKPOINT_SCHEMA
    if uses_grid_softmax_magnitude(parameterization):
        return GRID_SOFTMAX_CHECKPOINT_SCHEMA
    if uses_grid_magnitude(parameterization):
        return GRID_CHECKPOINT_SCHEMA
    return LEGACY_CHECKPOINT_SCHEMA


@dataclass(frozen=True)
class OutputPositionSignal:
    name: str
    source: str
    shape: str
    dtype: str
    construction: str
    routing: str
    neural_influence: bool
    applies_identically_to_both_arms: bool = True
    enters_encoder: bool = False


OUTPUT_POSITION_SIGNALS = (
    OutputPositionSignal(
        "decoder_output_step_absolute_embedding",
        "decoder_position_embedding(torch.arange(output_length))",
        "[1, output_length, model_dim] (broadcast over batch)",
        "floating point, matching decoder memory",
        "learned absolute serialized output-step embedding",
        "added to BOS/shifted-prefix content before decoder_input_norm",
        True,
    ),
    OutputPositionSignal(
        "pair_source_absolute_embedding",
        "the same decoder_position_embedding at the source pair index",
        "[batch, nodes, nodes, model_dim]",
        "floating point, matching decoded states",
        "learned absolute serialized source position",
        "concatenated into the frozen main pair-MLP feature vector",
        True,
    ),
    OutputPositionSignal(
        "pair_destination_absolute_embedding",
        "the same decoder_position_embedding at the destination pair index",
        "[batch, nodes, nodes, model_dim]",
        "floating point, matching decoded states",
        "learned absolute serialized destination position",
        "concatenated into the frozen main pair-MLP feature vector",
        True,
    ),
    OutputPositionSignal(
        "pair_signed_relative_serialized_position",
        "(source_index - destination_index) / max(max_nodes - 1, 1)",
        "[batch, nodes, nodes, 1]",
        "floating point, matching decoded states",
        "constructed signed relative output position",
        "concatenated into the frozen main pair-MLP feature vector",
        True,
    ),
)


BOOKKEEPING_ONLY_VALUES = (
    {
        "name": "pair_source_and_destination_indices",
        "permitted_use": (
            "address decoded output rows, form ordered pairs, and construct "
            "the four frozen semantic decoder-position signals"
        ),
        "neural_feature": False,
    },
    {
        "name": "output_node_mask",
        "permitted_use": (
            "requested-length grammar activity, padding exclusion, and active "
            "non-self pair masking"
        ),
        "neural_feature": False,
    },
    {
        "name": "graph_offsets",
        "permitted_use": "graph-local encoder validation, slicing, and pooling only",
        "neural_feature": False,
    },
    {
        "name": "edge_offsets_and_node_graph_ids",
        "permitted_use": "C3 batching validation and output alignment only",
        "neural_feature": False,
    },
)


EXCLUDED_POSITION_SIGNALS = (
    "typed_graph_encoder_chronological_position",
    "typed_graph_encoder_absolute_local_index",
    "graph_v1_c1_source_position_factor",
    "graph_v1_c1_destination_position_factor",
    "graph_v1_c1_position_class_projection",
)


def output_position_contract_metadata(
    node_generation_identity=LEGACY_NODE_GENERATION_IDENTITY,
):
    """Return a deterministic JSON-compatible copy of the frozen inventory."""

    autonomous_stop = uses_autonomous_stop(node_generation_identity)
    result = {
        "version": (
            AUTONOMOUS_STOP_OUTPUT_POSITION_CONTRACT_VERSION
            if autonomous_stop else OUTPUT_POSITION_CONTRACT_VERSION
        ),
        "semantic_signals": [asdict(item) for item in OUTPUT_POSITION_SIGNALS],
        "bookkeeping_only_values": [dict(item) for item in BOOKKEEPING_ONLY_VALUES],
        "excluded_signals": list(EXCLUDED_POSITION_SIGNALS),
    }
    if autonomous_stop:
        result["learned_terminator"] = {
            "node_type_id": 0,
            "token": "<pad>",
            "supervised_positions_per_example": 1,
            "occupies_decoder_output_position": True,
            "enters_graph_pair_enumeration": False,
        }
    return result


def operation_magnitude_contract_metadata(parameterization):
    """Return the versioned operation-magnitude neural-output contract."""

    if parameterization not in OPERATION_MAGNITUDE_PARAMETERIZATIONS:
        raise ValueError("unknown operation-magnitude parameterization")
    legacy = parameterization == LEGACY_OPERATION_MAGNITUDE_PARAMETERIZATION
    return {
        "parameterization": parameterization,
        "legacy": legacy,
        "serialized_channels": list(OPERATION_MAGNITUDE_SERIALIZED_CHANNELS),
        "compact_channels": list(OPERATION_MAGNITUDE_COMPACT_CHANNELS),
        "mapping": (
            "tanh(raw)"
            if legacy
            else "eps + (1 - eps) * sigmoid(raw)"
        ),
        "epsilon_policy": None if legacy else OPERATION_MAGNITUDE_EPSILON_POLICY,
        "normalized_domain": "[-1, 1]" if legacy else "(0, 1]",
        "direction_is_separate_categorical": True,
        "applies_identically_to_both_arms": True,
        "cad_kernel_validity_claimed": False,
    }
