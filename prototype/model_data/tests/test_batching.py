"""Mixed-length deterministic batching tests."""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace

from prototype.model_data.adapters import adapt_flat_mixed, adapt_typed_graph
from prototype.model_data.batching import collate_flat, collate_graph
from prototype.model_data.errors import ModelDataError
from prototype.model_data.loader import load_physical_examples
from prototype.model_data.serialization import canonical_record_json
from prototype.model_data.tests.fixtures import source, write_physical_corpus


class BatchingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        write_physical_corpus(
            self.temporary.name, (source("E"), source("RR"), source("ER"))
        )
        self.examples = load_physical_examples(self.temporary.name)

    def test_flat_batch_pads_mixed_history_lengths(self):
        items = tuple(adapt_flat_mixed(item) for item in reversed(self.examples))
        batch = collate_flat(items)
        self.assertEqual(batch.family_ids, tuple(sorted(batch.family_ids)))
        self.assertEqual(len(batch.padding_mask), 3)
        maximum = max(len(item.nodes) for item in self.examples)
        self.assertTrue(all(len(row) == maximum for row in batch.padding_mask))
        expected = {
            item.physical_family_id: len(item.nodes) for item in self.examples
        }
        self.assertEqual(
            tuple(sum(row) for row in batch.padding_mask),
            tuple(expected[item] for item in batch.family_ids),
        )
        self.assertEqual(batch.target.node_mask, batch.padding_mask)

    def test_graph_batch_offsets_and_masks_are_consistent(self):
        items = tuple(adapt_typed_graph(item) for item in reversed(self.examples))
        batch = collate_graph(items)
        self.assertEqual(batch.graph_offsets[0], 0)
        self.assertEqual(batch.graph_offsets[-1], len(batch.node_type_ids))
        self.assertEqual(batch.edge_offsets[0], 0)
        self.assertEqual(batch.edge_offsets[-1], len(batch.edge_type_ids))
        self.assertEqual(len(batch.node_graph_ids), len(batch.node_type_ids))
        for graph_index in range(len(batch.family_ids)):
            start, stop = batch.graph_offsets[graph_index : graph_index + 2]
            self.assertEqual(
                batch.node_graph_ids[start:stop], (graph_index,) * (stop - start)
            )
        self.assertTrue(
            all(index < len(batch.node_type_ids) for row in batch.edge_index for index in row)
        )
        by_id = {item.physical_family_id: item for item in items}
        for graph_index, family_id in enumerate(batch.family_ids):
            node_offset = batch.graph_offsets[graph_index]
            edge_start, edge_stop = batch.edge_offsets[graph_index : graph_index + 2]
            local = by_id[family_id]
            self.assertEqual(
                batch.edge_index[0][edge_start:edge_stop],
                tuple(node_offset + value for value in local.edge_index[0]),
            )
            self.assertEqual(
                batch.edge_index[1][edge_start:edge_stop],
                tuple(node_offset + value for value in local.edge_index[1]),
            )
            self.assertEqual(
                batch.geometry_mask[
                    batch.graph_offsets[graph_index] : batch.graph_offsets[graph_index + 1]
                ],
                local.geometry_mask,
            )

    def test_batching_is_order_independent_and_byte_identical(self):
        flat = tuple(adapt_flat_mixed(item) for item in self.examples)
        graph = tuple(adapt_typed_graph(item) for item in self.examples)
        self.assertEqual(
            canonical_record_json(collate_flat(flat)),
            canonical_record_json(collate_flat(tuple(reversed(flat)))),
        )
        self.assertEqual(
            canonical_record_json(collate_graph(graph)),
            canonical_record_json(collate_graph(tuple(reversed(graph)))),
        )

    def test_duplicate_family_in_batch_is_rejected(self):
        item = adapt_flat_mixed(self.examples[0])
        with self.assertRaises(ModelDataError) as caught:
            collate_flat((item, item))
        self.assertEqual(caught.exception.code, "duplicate_batch_family")

    def test_empty_batch_is_rejected_clearly(self):
        for collator in (collate_flat, collate_graph):
            with self.subTest(collator=collator.__name__):
                with self.assertRaises(ModelDataError) as caught:
                    collator(())
                self.assertEqual(caught.exception.code, "empty_batch")

    def test_empty_edge_graph_and_single_operation_collate(self):
        single = next(
            adapt_typed_graph(item)
            for item in self.examples
            if item.metadata.history_depth == 1
        )
        empty_target = replace(
            single.target,
            edge_index=((), ()),
            edge_type_ids=(),
        )
        empty_edges = replace(
            single,
            edge_index=((), ()),
            edge_type_ids=(),
            target=empty_target,
        )
        batch = collate_graph((empty_edges,))
        self.assertEqual(batch.edge_index, ((), ()))
        self.assertEqual(batch.edge_offsets, (0, 0))
        self.assertEqual(batch.target.edge_offsets, (0, 0))
        self.assertEqual(batch.target.operation_mask, ((True,),))
        self.assertEqual(
            batch.geometry_mask,
            empty_edges.geometry_mask,
        )


if __name__ == "__main__":
    unittest.main()
