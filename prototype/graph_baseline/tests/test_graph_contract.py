"""Pure contract enumeration for the controlled graph representation."""

from __future__ import annotations

from dataclasses import replace
import itertools
import unittest

from prototype.controlled_data.builders import build_history
from prototype.controlled_data.factors import PrimitiveFamily, ReferencePlane
from prototype.model_data.canonical import canonical_nodes_and_edges, reconstruction_target
from prototype.model_data.tests.fixtures import source
from prototype.representation.model import BooleanMode, Direction, GeometryEncoding

from prototype.graph_baseline.graph_contract import (
    GRAPH_EDGE_CLASS_ORDER,
    GRAPH_REPRESENTATION_NAME,
    DirectedTypedEdge,
    GraphContractError,
    graph_from_reconstruction_target,
    position_only_canonical_edges,
    reconstruction_target_from_graph,
    validate_graph_record,
)


class GraphContractTests(unittest.TestCase):
    def test_exact_identity_vocabulary_and_direction(self):
        self.assertEqual(GRAPH_REPRESENTATION_NAME, "B0-CONTROLLED-CAD-TYPED-GRAPH-v1")
        self.assertEqual(GRAPH_EDGE_CLASS_ORDER, (
            "<none>", "defined_in", "depends_on", "placed_on",
            "uses_axis", "uses_profile",
        ))

    def test_every_controlled_combination_round_trips_exactly(self):
        observed = {}
        for template in ("E", "R", "EE", "ER", "RE", "RR"):
            observed[template] = set()
            for primitive in PrimitiveFamily:
                for plane in ReferencePlane:
                    for directions in itertools.product(tuple(Direction), repeat=len(template)):
                        modes = (
                            (BooleanMode.JOIN, BooleanMode.CUT)
                            if len(template) == 2 else (BooleanMode.JOIN,)
                        )
                        for mode in modes:
                            for encoding in GeometryEncoding:
                                history = build_history(source(
                                    template, primitive, plane=plane,
                                    extents=(1.0,) * len(template),
                                    directions=directions, mode=mode,
                                ), encoding)
                                nodes, edges = canonical_nodes_and_edges(history)
                                target = reconstruction_target(
                                    nodes, edges, history.structure.operation_sequence
                                )
                                graph = graph_from_reconstruction_target(target)
                                restored = reconstruction_target_from_graph(graph, target)
                                self.assertEqual(restored, target)
                                self.assertEqual(
                                    graph.directed_typed_edges,
                                    position_only_canonical_edges(graph.node_type_ids),
                                )
                                observed[template].add((
                                    graph.node_type_ids, graph.directed_typed_edges
                                ))
        self.assertTrue(all(len(values) == 1 for values in observed.values()))
        self.assertEqual(
            {name: len(next(iter(values))[1]) for name, values in observed.items()},
            {"E": 3, "R": 5, "EE": 7, "ER": 9, "RE": 9, "RR": 11},
        )

    def test_malformed_duplicate_self_wrong_type_and_order_fail(self):
        history = build_history(source("R"), GeometryEncoding.CONTINUOUS)
        nodes, edges = canonical_nodes_and_edges(history)
        target = reconstruction_target(nodes, edges, history.structure.operation_sequence)
        graph = graph_from_reconstruction_target(target)
        first = graph.directed_typed_edges[0]
        cases = (
            replace(graph, directed_typed_edges=(first,) + graph.directed_typed_edges),
            replace(graph, directed_typed_edges=(DirectedTypedEdge(0, 0, 1),)),
            replace(graph, directed_typed_edges=(DirectedTypedEdge(0, 1, 3),)),
            replace(graph, directed_typed_edges=tuple(reversed(graph.directed_typed_edges))),
        )
        for malformed in cases:
            with self.subTest(malformed=malformed):
                with self.assertRaises(GraphContractError):
                    validate_graph_record(malformed)
