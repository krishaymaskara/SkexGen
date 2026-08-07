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


def output_position_contract_metadata():
    """Return a deterministic JSON-compatible copy of the frozen inventory."""

    return {
        "version": OUTPUT_POSITION_CONTRACT_VERSION,
        "semantic_signals": [asdict(item) for item in OUTPUT_POSITION_SIGNALS],
        "bookkeeping_only_values": [dict(item) for item in BOOKKEEPING_ONLY_VALUES],
        "excluded_signals": list(EXCLUDED_POSITION_SIGNALS),
    }
