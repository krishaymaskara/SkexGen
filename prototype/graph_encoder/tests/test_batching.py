"""Procedural C3 paired-batching, separation, and rejection tests."""

from __future__ import annotations

import ast
from dataclasses import fields, replace
import inspect
from pathlib import Path
import unittest
from unittest.mock import patch

from prototype.graph_encoder.batching import (
    C3_FAILURE_CODES,
    CANONICAL_TARGET_MISMATCH,
    COLLATED_TARGET_MISMATCH,
    CROSS_GRAPH_EDGE,
    DUPLICATE_FAMILY_ID,
    EMPTY_PAIRED_BATCH,
    FAMILY_ALIGNMENT_MISMATCH,
    FLAT_GRAPH_TARGET_MISMATCH,
    INVALID_EDGE_OFFSETS,
    INVALID_FLAT_PADDING,
    INVALID_GRAPH_OFFSETS,
    INVALID_NODE_GRAPH_IDS,
    INVALID_NODE_MASK,
    INVALID_PERMUTATION,
    INVALID_PHYSICAL_EXAMPLE,
    FlatEncoderInput,
    GraphBookkeeping,
    GraphNodeContent,
    GraphSemanticInput,
    PairedBatch,
    _split_graph_batch,
    build_paired_batch,
    permute_graph,
)
from prototype.graph_encoder.canonicalization import (
    GraphCanonicalizationInput,
    canonicalize_graph,
)
from prototype.graph_encoder.errors import GraphEncoderError
from prototype.model_data.adapters import adapt_flat_mixed, adapt_typed_graph
from prototype.model_data.batching import collate_flat, collate_graph
from prototype.model_data.serialization import canonical_record_json

from prototype.graph_encoder.tests.fixtures import procedural_fixture


try:
    import torch
except ImportError:
    torch = None


TORCH_REASON = "real PyTorch execution is deferred to the authoritative environment"
TEMPLATES = ("E", "R", "EE", "ER", "RE", "RR")


class _FakeTensor:
    def __init__(self, value, dtype):
        self.value = value
        self.dtype = dtype
        self.contiguous_calls = 0

    def contiguous(self):
        self.contiguous_calls += 1
        return self


class _FakeTorch:
    long = "long"
    float32 = "float32"
    bool = "bool"

    @staticmethod
    def tensor(value, dtype):
        return _FakeTensor(value, dtype)


def _record_bytes(value):
    return canonical_record_json(value).encode("utf-8")


def _examples():
    return tuple(procedural_fixture(template).physical for template in TEMPLATES)


def _assert_code(testcase, code, function):
    with testcase.assertRaises(GraphEncoderError) as caught:
        function()
    testcase.assertEqual(caught.exception.code, code)


class PairedBatchTests(unittest.TestCase):
    def test_all_six_templates_share_exact_authoritative_targets(self):
        for template in TEMPLATES:
            fixture = procedural_fixture(template)
            canonical = canonicalize_graph(fixture.graph)
            target = fixture.target
            self.assertEqual(
                (
                    canonical.node_type_ids,
                    canonical.categorical_attributes,
                    canonical.geometry,
                    canonical.geometry_mask,
                    canonical.edge_index,
                    canonical.edge_type_ids,
                    canonical.operation_sequence,
                ),
                (
                    target.node_type_ids,
                    target.categorical_attributes,
                    target.geometry,
                    target.geometry_mask,
                    target.edge_index,
                    target.edge_type_ids,
                    target.operation_sequence,
                ),
            )
            paired = build_paired_batch((fixture.physical,))
            flat_target = collate_flat((adapt_flat_mixed(fixture.physical),)).target
            graph_target = collate_graph((adapt_typed_graph(fixture.physical),)).target
            self.assertEqual(_record_bytes(flat_target), _record_bytes(graph_target))
            self.assertEqual(_record_bytes(paired.target), _record_bytes(flat_target))

    def test_mixed_lengths_padding_offsets_edges_and_family_alignment(self):
        examples = _examples()
        paired = build_paired_batch(tuple(reversed(examples)))
        self.assertIsInstance(paired, PairedBatch)
        self.assertEqual(
            paired.family_ids,
            tuple(sorted(item.physical_family_id for item in examples)),
        )
        by_id = {item.physical_family_id: item for item in examples}
        counts = tuple(len(by_id[item].target.node_type_ids) for item in paired.family_ids)
        edge_counts = tuple(len(by_id[item].target.edge_type_ids) for item in paired.family_ids)
        maximum = max(counts)
        self.assertEqual(
            paired.flat_input.padding_mask,
            tuple(
                (True,) * count + (False,) * (maximum - count)
                for count in counts
            ),
        )
        expected_graph_offsets = [0]
        expected_edge_offsets = [0]
        for count in counts:
            expected_graph_offsets.append(expected_graph_offsets[-1] + count)
        for count in edge_counts:
            expected_edge_offsets.append(expected_edge_offsets[-1] + count)
        self.assertEqual(
            paired.graph_bookkeeping.graph_offsets,
            tuple(expected_graph_offsets),
        )
        self.assertEqual(
            paired.graph_bookkeeping.edge_offsets,
            tuple(expected_edge_offsets),
        )
        self.assertEqual(paired.graph_input.edge_index, paired.target.edge_index)
        self.assertEqual(paired.graph_input.edge_type_ids, paired.target.edge_type_ids)
        for graph_index in range(len(paired.family_ids)):
            node_start, node_stop = paired.graph_bookkeeping.graph_offsets[
                graph_index : graph_index + 2
            ]
            edge_start, edge_stop = paired.graph_bookkeeping.edge_offsets[
                graph_index : graph_index + 2
            ]
            self.assertEqual(
                paired.graph_bookkeeping.node_graph_ids[node_start:node_stop],
                (graph_index,) * (node_stop - node_start),
            )
            for source, destination in zip(
                paired.graph_input.edge_index[0][edge_start:edge_stop],
                paired.graph_input.edge_index[1][edge_start:edge_stop],
            ):
                self.assertTrue(node_start <= source < node_stop)
                self.assertTrue(node_start <= destination < node_stop)

    def test_forward_reverse_and_arbitrary_order_are_byte_identical(self):
        examples = _examples()
        arbitrary = tuple(examples[index] for index in (2, 5, 0, 4, 1, 3))
        reference = _record_bytes(build_paired_batch(examples))
        self.assertEqual(reference, _record_bytes(build_paired_batch(tuple(reversed(examples)))))
        self.assertEqual(reference, _record_bytes(build_paired_batch(arbitrary)))
        self.assertEqual(reference, _record_bytes(build_paired_batch(examples)))

    def test_primary_builder_never_invokes_permutation_augmentation(self):
        example = procedural_fixture("RR").physical
        with patch(
            "prototype.graph_encoder.batching.permute_graph",
            side_effect=AssertionError("automatic permutation is forbidden"),
        ) as mocked:
            build_paired_batch((example,))
        mocked.assert_not_called()
        source = inspect.getsource(build_paired_batch)
        self.assertNotIn("permute_graph(", source)

    def test_low_level_empty_edges_pack_without_becoming_a_valid_program(self):
        fixture = procedural_fixture("E")
        typed = adapt_typed_graph(fixture.physical)
        empty_target = replace(
            typed.target,
            edge_index=((), ()),
            edge_type_ids=(),
        )
        empty_graph = replace(
            typed,
            edge_index=((), ()),
            edge_type_ids=(),
            target=empty_target,
        )
        inherited = collate_graph((empty_graph,))
        semantic, bookkeeping = _split_graph_batch(
            inherited,
            expected_family_ids=(empty_graph.physical_family_id,),
            expected_node_counts=(len(empty_graph.node_type_ids),),
            expected_edge_counts=(0,),
        )
        self.assertEqual(semantic.edge_index, ((), ()))
        self.assertEqual(semantic.edge_type_ids, ())
        self.assertEqual(bookkeeping.edge_offsets, (0, 0))
        narrow = GraphCanonicalizationInput(
            semantic.node_type_ids,
            semantic.edge_index,
            semantic.edge_type_ids,
            semantic.categorical_attributes,
            semantic.geometry,
            semantic.geometry_mask,
        )
        with self.assertRaises(GraphEncoderError):
            canonicalize_graph(narrow)
        invalid_physical = replace(fixture.physical, target=empty_target)
        with self.assertRaises(GraphEncoderError):
            build_paired_batch((invalid_physical,))


class PermutationTests(unittest.TestCase):
    def test_explicit_permutation_reorders_every_field_and_relabels_edges(self):
        fixture = procedural_fixture("RR")
        graph = fixture.graph
        permutation = tuple(reversed(range(len(graph.node_type_ids))))
        permuted = permute_graph(graph, permutation)
        inverse = tuple(reversed(range(len(graph.node_type_ids))))
        self.assertEqual(
            permuted.node_type_ids,
            tuple(graph.node_type_ids[old] for old in permutation),
        )
        self.assertEqual(
            permuted.categorical_attributes,
            tuple(graph.categorical_attributes[old] for old in permutation),
        )
        self.assertEqual(
            permuted.geometry,
            tuple(graph.geometry[old] for old in permutation),
        )
        self.assertEqual(
            permuted.geometry_mask,
            tuple(graph.geometry_mask[old] for old in permutation),
        )
        self.assertEqual(
            permuted.edge_index,
            (
                tuple(inverse[value] for value in graph.edge_index[0]),
                tuple(inverse[value] for value in graph.edge_index[1]),
            ),
        )
        self.assertEqual(permuted.edge_type_ids, graph.edge_type_ids)
        canonical = canonicalize_graph(permuted)
        self.assertEqual(canonical.node_type_ids, fixture.target.node_type_ids)
        self.assertEqual(canonical.edge_index, fixture.target.edge_index)
        self.assertEqual(canonical.geometry, fixture.target.geometry)
        self.assertEqual(canonical.operation_sequence, fixture.target.operation_sequence)

    def test_invalid_permutations_are_rejected(self):
        graph = procedural_fixture("E").graph
        count = len(graph.node_type_ids)
        invalid = (
            (),
            tuple(range(count - 1)),
            (0, 1, 2, 2),
            (False, 1, 2, 3),
            (-1, 1, 2, 3),
            (0, 1, 2, count),
            None,
        )
        for permutation in invalid:
            with self.subTest(permutation=permutation):
                _assert_code(
                    self,
                    INVALID_PERMUTATION,
                    lambda permutation=permutation: permute_graph(graph, permutation),
                )
        _assert_code(
            self,
            INVALID_PERMUTATION,
            lambda: permute_graph(object(), tuple(range(count))),
        )


class BoundaryAndTensorTests(unittest.TestCase):
    def test_encoder_records_are_structurally_target_and_metadata_free(self):
        self.assertEqual(
            {field.name for field in fields(FlatEncoderInput)},
            {"categorical_ids", "geometry", "geometry_mask", "padding_mask"},
        )
        self.assertEqual(
            {field.name for field in fields(GraphNodeContent)},
            {"node_type_ids", "categorical_attributes", "geometry", "geometry_mask"},
        )
        self.assertEqual(
            {field.name for field in fields(GraphSemanticInput)},
            {"node_content", "edge_index", "edge_type_ids"},
        )
        self.assertEqual(
            {field.name for field in fields(GraphBookkeeping)},
            {"graph_offsets", "edge_offsets", "node_graph_ids", "node_mask"},
        )
        forbidden_target = {
            "target",
            "operation_sequence",
            "boolean_mode_targets",
            "family_template",
            "operation_template",
            "split",
            "split_name",
            "partition",
        }
        forbidden_bookkeeping = {
            "graph_offsets",
            "edge_offsets",
            "node_graph_ids",
            "node_mask",
            "padding_position",
            "family_ids",
        }
        for record in (FlatEncoderInput, GraphNodeContent, GraphSemanticInput):
            names = {field.name for field in fields(record)}
            self.assertTrue(names.isdisjoint(forbidden_target))
        self.assertTrue(
            {field.name for field in fields(GraphNodeContent)}.isdisjoint(
                forbidden_bookkeeping
            )
        )
        self.assertEqual(
            tuple(inspect.signature(GraphNodeContent).parameters),
            ("node_type_ids", "categorical_attributes", "geometry", "geometry_mask"),
        )

    def test_encoder_tensor_dictionaries_keep_target_and_bookkeeping_separate(self):
        paired = build_paired_batch(_examples())
        flat_names = set(paired.flat_input.__dataclass_fields__)
        semantic_names = set(paired.graph_input.__dataclass_fields__)
        node_names = set(paired.graph_input.node_content.__dataclass_fields__)
        bookkeeping_names = set(paired.graph_bookkeeping.__dataclass_fields__)
        self.assertTrue(flat_names.isdisjoint({"target", "operation_sequence"}))
        self.assertTrue(semantic_names.isdisjoint({"target", "operation_sequence"}))
        self.assertTrue(bookkeeping_names.isdisjoint({"target", "operation_sequence"}))
        self.assertTrue(node_names.isdisjoint(bookkeeping_names))
        self.assertFalse(hasattr(paired, "to_torch"))
        flat = paired.flat_input.to_torch(_FakeTorch)
        semantic = paired.graph_input.to_torch(_FakeTorch)
        node_content = paired.graph_input.node_content.to_torch(_FakeTorch)
        bookkeeping = paired.graph_bookkeeping.to_torch(_FakeTorch)
        self.assertEqual(
            set(flat),
            {"categorical_ids", "geometry", "geometry_mask", "padding_mask"},
        )
        self.assertEqual(
            set(semantic),
            {
                "node_type_ids",
                "categorical_attributes",
                "geometry",
                "geometry_mask",
                "edge_index",
                "edge_type_ids",
            },
        )
        self.assertEqual(
            set(bookkeeping),
            {"graph_offsets", "edge_offsets", "node_graph_ids", "node_mask"},
        )
        self.assertEqual(
            set(node_content),
            {"node_type_ids", "categorical_attributes", "geometry", "geometry_mask"},
        )
        self.assertTrue(set(semantic).isdisjoint(bookkeeping))
        self.assertTrue(
            {"target", "operation_sequence", "boolean_mode_targets"}.isdisjoint(
                set(flat) | set(semantic) | set(bookkeeping)
            )
        )
        self.assertTrue(
            all(
                tensor.contiguous_calls == 1
                for collection in (flat, semantic, node_content, bookkeeping)
                for tensor in collection.values()
            )
        )

    def test_batching_source_has_no_loader_or_automatic_random_dependency(self):
        import prototype.graph_encoder.batching as module

        path = Path(module.__file__)
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        forbidden_calls = {
            "load_train",
            "load_development",
            "partition_family_ids",
            "load_partition_physical_examples",
            "load_physical_examples",
            "random",
            "shuffle",
        }
        observed_calls = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                function = node.func
                if isinstance(function, ast.Name):
                    observed_calls.add(function.id)
                elif isinstance(function, ast.Attribute):
                    observed_calls.add(function.attr)
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                rendered = ast.dump(node)
                self.assertNotIn("model_data.loader", rendered)
                self.assertNotIn("manifest", rendered)
        self.assertTrue(observed_calls.isdisjoint(forbidden_calls))

    @unittest.skipUnless(torch is not None, TORCH_REASON)
    def test_cpu_real_tensor_smoke_keeps_boundaries_separate(self):
        paired = build_paired_batch(_examples())
        flat = paired.flat_input.to_torch(torch)
        semantic = paired.graph_input.to_torch(torch)
        node_content = paired.graph_input.node_content.to_torch(torch)
        bookkeeping = paired.graph_bookkeeping.to_torch(torch)
        target = paired.target.to_torch(torch)
        self.assertEqual(
            set(flat),
            {"categorical_ids", "geometry", "geometry_mask", "padding_mask"},
        )
        self.assertEqual(
            set(semantic),
            {
                "node_type_ids",
                "categorical_attributes",
                "geometry",
                "geometry_mask",
                "edge_index",
                "edge_type_ids",
            },
        )
        self.assertEqual(
            set(bookkeeping),
            {"graph_offsets", "edge_offsets", "node_graph_ids", "node_mask"},
        )
        self.assertTrue(set(node_content).isdisjoint(bookkeeping))
        self.assertIn("operation_sequence", target)
        for collection in (flat, semantic, node_content, bookkeeping, target):
            for tensor in collection.values():
                self.assertEqual(tensor.device.type, "cpu")
                self.assertTrue(tensor.is_contiguous())
        self.assertEqual(flat["categorical_ids"].dtype, torch.long)
        self.assertEqual(flat["geometry"].dtype, torch.float32)
        self.assertEqual(flat["padding_mask"].dtype, torch.bool)
        self.assertEqual(semantic["edge_index"].shape[0], 2)
        self.assertEqual(semantic["geometry"].shape[1], 39)
        self.assertEqual(bookkeeping["graph_offsets"].shape[0], len(TEMPLATES) + 1)


class FailureTaxonomyTests(unittest.TestCase):
    def test_failure_taxonomy_is_exact(self):
        self.assertEqual(
            set(C3_FAILURE_CODES),
            {
                EMPTY_PAIRED_BATCH,
                INVALID_PHYSICAL_EXAMPLE,
                DUPLICATE_FAMILY_ID,
                FAMILY_ALIGNMENT_MISMATCH,
                CANONICAL_TARGET_MISMATCH,
                FLAT_GRAPH_TARGET_MISMATCH,
                COLLATED_TARGET_MISMATCH,
                INVALID_PERMUTATION,
                INVALID_FLAT_PADDING,
                INVALID_GRAPH_OFFSETS,
                INVALID_EDGE_OFFSETS,
                INVALID_NODE_GRAPH_IDS,
                CROSS_GRAPH_EDGE,
                INVALID_NODE_MASK,
            },
        )
        self.assertEqual(len(C3_FAILURE_CODES), len(set(C3_FAILURE_CODES)))

    def test_empty_invalid_and_duplicate_inputs_have_distinct_codes(self):
        example = procedural_fixture("E").physical
        _assert_code(self, EMPTY_PAIRED_BATCH, lambda: build_paired_batch(()))
        _assert_code(
            self,
            INVALID_PHYSICAL_EXAMPLE,
            lambda: build_paired_batch((object(),)),
        )
        _assert_code(
            self,
            INVALID_PHYSICAL_EXAMPLE,
            lambda: build_paired_batch(None),
        )
        _assert_code(
            self,
            DUPLICATE_FAMILY_ID,
            lambda: build_paired_batch((example, example)),
        )

    def test_adapted_family_mismatch_is_rejected(self):
        example = procedural_fixture("E").physical
        original = adapt_typed_graph

        def mismatched(item):
            return replace(original(item), physical_family_id="different-family")

        with patch("prototype.graph_encoder.batching.adapt_typed_graph", mismatched):
            _assert_code(
                self,
                FAMILY_ALIGNMENT_MISMATCH,
                lambda: build_paired_batch((example,)),
            )

    def test_per_example_flat_graph_target_mismatch_is_rejected(self):
        example = procedural_fixture("E").physical
        original = adapt_typed_graph

        def mismatched(item):
            graph = original(item)
            boolean_targets = list(graph.target.boolean_mode_targets)
            boolean_targets[0] += 1
            return replace(
                graph,
                target=replace(
                    graph.target,
                    boolean_mode_targets=tuple(boolean_targets),
                ),
            )

        with patch("prototype.graph_encoder.batching.adapt_typed_graph", mismatched):
            _assert_code(
                self,
                FLAT_GRAPH_TARGET_MISMATCH,
                lambda: build_paired_batch((example,)),
            )

    def test_c2_canonical_target_mismatch_is_terminal(self):
        example = procedural_fixture("EE").physical
        target = replace(
            example.target,
            operation_sequence=tuple(reversed(example.target.operation_sequence)),
        )
        malformed = replace(example, target=target)
        _assert_code(
            self,
            CANONICAL_TARGET_MISMATCH,
            lambda: build_paired_batch((malformed,)),
        )

    def test_collated_target_mismatch_is_rejected(self):
        example = procedural_fixture("EE").physical
        original = collate_graph

        def mismatched(items):
            batch = original(items)
            rows = list(batch.target.operation_sequence)
            rows[0] = tuple(reversed(rows[0]))
            return replace(
                batch,
                target=replace(batch.target, operation_sequence=tuple(rows)),
            )

        with patch("prototype.graph_encoder.batching.collate_graph", mismatched):
            _assert_code(
                self,
                COLLATED_TARGET_MISMATCH,
                lambda: build_paired_batch((example,)),
            )

    def test_invalid_flat_padding_is_rejected(self):
        example = procedural_fixture("E").physical
        original = collate_flat

        def malformed(items):
            batch = original(items)
            return replace(batch, padding_mask=((False,) * len(batch.padding_mask[0]),))

        with patch("prototype.graph_encoder.batching.collate_flat", malformed):
            _assert_code(
                self,
                INVALID_FLAT_PADDING,
                lambda: build_paired_batch((example,)),
            )

    def test_invalid_graph_and_edge_offsets_are_rejected(self):
        graph_batch = collate_graph((adapt_typed_graph(procedural_fixture("E").physical),))
        bad_graph = replace(
            graph_batch,
            graph_offsets=(0, len(graph_batch.node_type_ids) - 1),
        )
        bad_edge = replace(
            graph_batch,
            edge_offsets=(0, len(graph_batch.edge_type_ids) - 1),
        )
        _assert_code(
            self,
            INVALID_GRAPH_OFFSETS,
            lambda: _split_graph_batch(bad_graph),
        )
        _assert_code(
            self,
            INVALID_EDGE_OFFSETS,
            lambda: _split_graph_batch(bad_edge),
        )

    def test_invalid_node_graph_ids_and_masks_are_rejected(self):
        graph_batch = collate_graph((adapt_typed_graph(procedural_fixture("R").physical),))
        bad_ids = replace(
            graph_batch,
            node_graph_ids=(1,) + graph_batch.node_graph_ids[1:],
        )
        mask = list(graph_batch.node_mask[0])
        mask[0] = False
        bad_mask = replace(graph_batch, node_mask=(tuple(mask),))
        _assert_code(
            self,
            INVALID_NODE_GRAPH_IDS,
            lambda: _split_graph_batch(bad_ids),
        )
        _assert_code(
            self,
            INVALID_NODE_MASK,
            lambda: _split_graph_batch(bad_mask),
        )

    def test_cross_graph_edges_are_rejected(self):
        examples = tuple(
            adapt_typed_graph(procedural_fixture(template).physical)
            for template in ("E", "R")
        )
        graph_batch = collate_graph(examples)
        sources = list(graph_batch.edge_index[0])
        second_edge = graph_batch.edge_offsets[1]
        sources[second_edge] = 0
        crossed = replace(
            graph_batch,
            edge_index=(tuple(sources), graph_batch.edge_index[1]),
        )
        _assert_code(
            self,
            CROSS_GRAPH_EDGE,
            lambda: _split_graph_batch(crossed),
        )


if __name__ == "__main__":
    unittest.main()
