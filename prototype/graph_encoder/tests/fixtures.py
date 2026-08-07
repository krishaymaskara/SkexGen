"""Procedural in-memory PhysicalExample fixtures shared by GE1 unit tests."""

from __future__ import annotations

from dataclasses import dataclass

from prototype.controlled_data.builders import build_history
from prototype.controlled_data.identity import sample_id, source_family_id
from prototype.graph_encoder.canonicalization import GraphCanonicalizationInput
from prototype.model_data.adapters import adapt_typed_graph
from prototype.model_data.canonical import (
    canonical_nodes_and_edges,
    infer_metadata,
    reconstruction_target,
)
from prototype.model_data.records import FamilyMetadata, PhysicalExample
from prototype.model_data.tests.fixtures import source
from prototype.representation.model import GeometryEncoding
from prototype.representation.serialization import history_to_json


@dataclass(frozen=True)
class ProceduralFixture:
    physical: PhysicalExample
    graph: GraphCanonicalizationInput
    target: object


def procedural_fixture(template):
    """Build one six-family fixture without opening manifests or payloads."""

    physical_source = source(template)
    continuous = build_history(physical_source, GeometryEncoding.CONTINUOUS)
    quantized = build_history(physical_source, GeometryEncoding.QUANTIZED)
    nodes, edges = canonical_nodes_and_edges(continuous)
    target = reconstruction_target(
        nodes, edges, continuous.structure.operation_sequence
    )
    operation_template, primitive_family, reference_plane = infer_metadata(continuous)
    continuous_family = source_family_id(continuous)
    if source_family_id(quantized) != continuous_family:
        raise AssertionError("procedural encodings disagree on family identity")
    physical = PhysicalExample(
        continuous_family,
        FamilyMetadata(
            operation_template,
            primitive_family,
            reference_plane,
            len(continuous.structure.operation_sequence),
            ("continuous", "quantized"),
            tuple(sorted((sample_id(continuous), sample_id(quantized)))),
        ),
        "procedural_ge1_fixture",
        "procedural_only",
        continuous.structure.operation_sequence,
        nodes,
        edges,
        target,
        history_to_json(continuous),
    )
    adapted = adapt_typed_graph(physical)
    graph = GraphCanonicalizationInput(
        adapted.node_type_ids,
        adapted.edge_index,
        adapted.edge_type_ids,
        adapted.categorical_attributes,
        adapted.geometry,
        adapted.geometry_mask,
    )
    return ProceduralFixture(physical, graph, target)
