"""Exhaustive small-grid canonical alignment across all v1 templates."""

from __future__ import annotations

import tempfile
import unittest

from prototype.controlled_data.factors import PrimitiveFamily, ReferencePlane
from prototype.model_data.adapters import adapt_flat_mixed, adapt_typed_graph
from prototype.model_data.loader import load_physical_examples
from prototype.model_data.tests.fixtures import source, write_physical_corpus
from prototype.model_data.vocab import BOOLEAN_MODES, EDGE_TYPES, NODE_TYPES
from prototype.representation.model import BooleanMode, Direction
from prototype.representation.serialization import history_from_json


class CanonicalAlignmentTests(unittest.TestCase):
    def test_all_templates_and_primitives_have_exact_flat_graph_alignment(self):
        sources = []
        templates = ("E", "R", "EE", "ER", "RE", "RR")
        primitives = tuple(PrimitiveFamily)
        planes = tuple(ReferencePlane)
        for template in templates:
            for primitive_index, primitive in enumerate(primitives):
                plane = planes[primitive_index]
                if len(template) == 1:
                    for direction in (Direction.POSITIVE, Direction.NEGATIVE):
                        sources.append(
                            source(
                                template,
                                primitive,
                                plane,
                                directions=(direction,),
                            )
                        )
                else:
                    for mode in (BooleanMode.JOIN, BooleanMode.CUT):
                        for directions in (
                            (Direction.POSITIVE, Direction.POSITIVE),
                            (Direction.POSITIVE, Direction.NEGATIVE),
                        ):
                            sources.append(
                                source(
                                    template,
                                    primitive,
                                    plane,
                                    directions=directions,
                                    mode=mode,
                                )
                            )

        with tempfile.TemporaryDirectory() as temporary:
            write_physical_corpus(temporary, sources)
            examples = load_physical_examples(temporary)

        self.assertEqual(len(examples), 60)
        self.assertEqual(
            {item.metadata.operation_template for item in examples}, set(templates)
        )
        self.assertEqual(
            {item.metadata.primitive_family for item in examples},
            {item.value for item in primitives},
        )
        for physical in examples:
            flat = adapt_flat_mixed(physical)
            graph = adapt_typed_graph(physical)
            target = physical.target
            node_positions = {
                node.node_id: index for index, node in enumerate(physical.nodes)
            }
            restored = history_from_json(physical.canonical_reconstruction_json)

            self.assertEqual(flat.target, graph.target)
            self.assertEqual(graph.target, target)
            self.assertEqual(
                tuple(row[0] for row in flat.categorical_ids),
                tuple(NODE_TYPES.id(node.node_type) for node in physical.nodes),
            )
            self.assertEqual(
                tuple(row[1:] for row in flat.categorical_ids),
                graph.categorical_attributes,
            )
            self.assertEqual(flat.geometry, graph.geometry)
            self.assertEqual(flat.geometry_mask, graph.geometry_mask)
            self.assertEqual(
                graph.edge_index,
                (
                    tuple(node_positions[edge.source_id] for edge in physical.edges),
                    tuple(node_positions[edge.target_id] for edge in physical.edges),
                ),
            )
            self.assertEqual(
                graph.edge_type_ids,
                tuple(EDGE_TYPES.id(edge.edge_type) for edge in physical.edges),
            )
            self.assertEqual(
                graph.operation_sequence,
                tuple(
                    node_positions[node_id]
                    for node_id in physical.operation_sequence
                ),
            )
            self.assertEqual(
                restored.structure.operation_sequence, physical.operation_sequence
            )
            self.assertEqual(
                target.boolean_mode_targets,
                tuple(
                    BOOLEAN_MODES.id(node.boolean_mode) for node in physical.nodes
                ),
            )


if __name__ == "__main__":
    unittest.main()
