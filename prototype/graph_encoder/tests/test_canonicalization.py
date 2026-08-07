"""Procedural parity and rejection tests for position-free C2."""

from __future__ import annotations

from dataclasses import replace
import ast
import hashlib
import inspect
import itertools
import math
from pathlib import Path
import unittest

from prototype.graph_encoder.canonicalization import (
    AMBIGUOUS_OPERATION_CHAIN,
    AXIS_PROFILE_SKETCH_MISMATCH,
    DEFENSIVE_ONLY_FAILURE_CODES,
    DISCONNECTED_OPERATION_CHAIN,
    DUPLICATE_EDGE,
    FAILURE_CODES,
    REACHABLE_FAILURE_CODES,
    INCOMPATIBLE_TYPED_EDGE,
    INVALID_AXIS_SKETCH_TARGET_TYPE,
    INVALID_AXIS_TARGET_TYPE,
    INVALID_EDGE_TYPE_ID,
    INVALID_NODE_TYPE_ID,
    INVALID_PROFILE_SKETCH_TARGET_TYPE,
    INVALID_PROFILE_TARGET_TYPE,
    MALFORMED_EDGE_INDEX,
    MALFORMED_NODE_FIELDS,
    MISSING_FORBIDDEN_OR_MULTIPLE_OPERATION_AXIS,
    MISSING_OR_MULTIPLE_AXIS_DEFINED_IN,
    MISSING_OR_MULTIPLE_OPERATION_PROFILE,
    MISSING_OR_MULTIPLE_PROFILE_DEFINED_IN,
    MISSING_OR_MULTIPLE_SKETCH_PLACED_ON,
    MULTIPLY_ASSIGNED_NODE,
    OPERATION_CHAIN_BRANCHING,
    OPERATION_CHAIN_CYCLE,
    OUT_OF_RANGE_EDGE_ENDPOINT,
    PLACEMENT_ON_NON_PLANE_NODE,
    PLACEMENT_ON_WRONG_SHARED_PLANE,
    SELF_EDGE,
    UNSUPPORTED_OPERATION_COUNT,
    UNSUPPORTED_PLANE_COUNT,
    UNASSIGNED_NODE,
    GraphCanonicalizationInput,
    _assert_shared_plane,
    _operation_order,
    canonicalize_graph,
)
from prototype.graph_encoder.errors import GraphEncoderError
from prototype.model_data.vocab import EDGE_TYPES, NODE_TYPES

from prototype.graph_encoder.tests.fixtures import procedural_fixture as _fixture


_RANK_DIGESTS = {
    7: "54130a190a4278b945bd8da932d187e0c50730508d98a4b5c540d5d9af1bb5d2",
    8: "d95c521c159093f09de5a460d506050104f973c0605f9c46ed42fd6e94eb0fb5",
    9: "2b666713370a605d5d3bb6ddf4c70974ac250230251c296f5d8b3a8641486ac9",
}


def _permute(graph, permutation):
    count = len(graph.node_type_ids)
    if tuple(sorted(permutation)) != tuple(range(count)):
        raise AssertionError("test permutation is not one-to-one")
    old_to_new = [None] * count
    for new, old in enumerate(permutation):
        old_to_new[old] = new
    return GraphCanonicalizationInput(
        tuple(graph.node_type_ids[old] for old in permutation),
        (
            tuple(old_to_new[source] for source in graph.edge_index[0]),
            tuple(old_to_new[destination] for destination in graph.edge_index[1]),
        ),
        graph.edge_type_ids,
        tuple(graph.categorical_attributes[old] for old in permutation),
        tuple(graph.geometry[old] for old in permutation),
        tuple(graph.geometry_mask[old] for old in permutation),
    )


def _assert_parity(testcase, fixture, permutation):
    incoming = _permute(fixture.graph, permutation)
    result = canonicalize_graph(incoming)
    target = fixture.target
    testcase.assertEqual(result.node_type_ids, target.node_type_ids)
    testcase.assertEqual(result.categorical_attributes, target.categorical_attributes)
    testcase.assertEqual(result.geometry, target.geometry)
    testcase.assertEqual(result.geometry_mask, target.geometry_mask)
    testcase.assertEqual(result.edge_index, target.edge_index)
    testcase.assertEqual(result.edge_type_ids, target.edge_type_ids)
    testcase.assertEqual(result.operation_sequence, target.operation_sequence)
    expected_canonical_to_old = tuple(permutation.index(index) for index in range(len(permutation)))
    testcase.assertEqual(result.canonical_to_old, expected_canonical_to_old)
    testcase.assertEqual(result.old_to_canonical, tuple(permutation))
    for canonical, old in enumerate(result.canonical_to_old):
        testcase.assertEqual(result.old_to_canonical[old], canonical)
    testcase.assertEqual(result, canonicalize_graph(incoming))


def _distributed_ranks(count):
    total = math.factorial(count)
    return tuple((index * total) // 200 for index in range(200))


def _unrank_permutation(count, rank):
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("count must be positive")
    total = math.factorial(count)
    if isinstance(rank, bool) or not isinstance(rank, int) or not 0 <= rank < total:
        raise ValueError("rank is outside the permutation space")
    available = list(range(count))
    result = []
    remainder = rank
    for remaining in range(count, 0, -1):
        block = math.factorial(remaining - 1)
        position, remainder = divmod(remainder, block)
        result.append(available.pop(position))
    return tuple(result)


def _edge_tuples(graph):
    return list(zip(graph.edge_index[0], graph.edge_index[1], graph.edge_type_ids))


def _outgoing_to_by_source_type(outgoing):
    """Adapt a plain operation-dependency map to the internal edge index."""

    depends_on = EDGE_TYPES.id("depends_on")
    return {
        (source, depends_on): tuple(sorted(targets))
        for source, targets in outgoing.items()
        if targets
    }


def _with_edges(graph, edges):
    return replace(
        graph,
        edge_index=(
            tuple(item[0] for item in edges),
            tuple(item[1] for item in edges),
        ),
        edge_type_ids=tuple(item[2] for item in edges),
    )


def _append_node(graph, node_type_id, clone_index=0):
    return replace(
        graph,
        node_type_ids=graph.node_type_ids + (node_type_id,),
        categorical_attributes=(
            graph.categorical_attributes + (graph.categorical_attributes[clone_index],)
        ),
        geometry=graph.geometry + (graph.geometry[clone_index],),
        geometry_mask=graph.geometry_mask + (graph.geometry_mask[clone_index],),
    )


def _indices(graph, token):
    node_id = NODE_TYPES.id(token)
    return tuple(index for index, value in enumerate(graph.node_type_ids) if value == node_id)


def _replace_edge(edges, edge_type, source=None, destination=None, new_source=None, new_destination=None):
    result = []
    replaced = False
    for current_source, current_destination, current_type in edges:
        if (
            not replaced
            and current_type == edge_type
            and (source is None or current_source == source)
            and (destination is None or current_destination == destination)
        ):
            result.append((
                current_source if new_source is None else new_source,
                current_destination if new_destination is None else new_destination,
                current_type,
            ))
            replaced = True
        else:
            result.append((current_source, current_destination, current_type))
    if not replaced:
        raise AssertionError("test edge was not found")
    return result


class CanonicalizationParityTests(unittest.TestCase):
    def test_unpermuted_all_six_controlled_templates_match_authority(self):
        for template in ("E", "R", "EE", "ER", "RE", "RR"):
            fixture = _fixture(template)
            with self.subTest(template=template):
                _assert_parity(
                    self, fixture, tuple(range(len(fixture.graph.node_type_ids)))
                )

    def test_exhaustive_single_operation_permutations(self):
        expected_counts = {"E": 24, "R": 120}
        expected_nodes = {"E": 4, "R": 5}
        for template in ("E", "R"):
            fixture = _fixture(template)
            count = len(fixture.graph.node_type_ids)
            self.assertEqual(count, expected_nodes[template])
            permutations = tuple(itertools.permutations(range(count)))
            self.assertEqual(len(permutations), expected_counts[template])
            for permutation in permutations:
                _assert_parity(self, fixture, permutation)

    def test_exact_distributed_two_operation_permutations(self):
        for template in ("EE", "ER", "RE", "RR"):
            fixture = _fixture(template)
            count = len(fixture.graph.node_type_ids)
            ranks = _distributed_ranks(count)
            self.assertEqual(len(ranks), 200)
            self.assertEqual(len(set(ranks)), 200)
            digest = hashlib.sha256(
                ("\n".join(str(item) for item in ranks) + "\n").encode("ascii")
            ).hexdigest()
            self.assertEqual(digest, _RANK_DIGESTS[count])
            permutations = tuple(_unrank_permutation(count, rank) for rank in ranks)
            self.assertEqual(len(set(permutations)), 200)
            for permutation in permutations:
                _assert_parity(self, fixture, permutation)

    def test_factorial_unranking_matches_lexicographic_permutations(self):
        expected = tuple(itertools.permutations(range(5)))
        actual = tuple(_unrank_permutation(5, rank) for rank in range(math.factorial(5)))
        self.assertEqual(actual, expected)

    def test_named_adversarial_permutations(self):
        for template in ("EE", "ER", "RE", "RR"):
            fixture = _fixture(template)
            graph = fixture.graph
            count = len(graph.node_type_ids)
            named = {"reverse": tuple(reversed(range(count)))}
            for label, token in (("sketch_swap", "sketch"), ("profile_swap", "profile")):
                values = _indices(graph, token)
                permutation = list(range(count))
                permutation[values[0]], permutation[values[1]] = (
                    permutation[values[1]], permutation[values[0]]
                )
                named[label] = tuple(permutation)
            operations = tuple(
                index
                for index, node_type in enumerate(graph.node_type_ids)
                if node_type in (NODE_TYPES.id("extrude"), NODE_TYPES.id("revolve"))
            )
            permutation = list(range(count))
            permutation[operations[0]], permutation[operations[1]] = (
                permutation[operations[1]], permutation[operations[0]]
            )
            named["operation_swap"] = tuple(permutation)
            if template == "RR":
                axes = _indices(graph, "axis")
                permutation = list(range(count))
                permutation[axes[0]], permutation[axes[1]] = (
                    permutation[axes[1]], permutation[axes[0]]
                )
                named["axis_swap"] = tuple(permutation)
            first_operation, second_operation = fixture.target.operation_sequence
            group_swap = (
                (0,)
                + tuple(range(first_operation + 1, second_operation + 1))
                + tuple(range(1, first_operation + 1))
            )
            named["second_group_first"] = group_swap
            for name, permutation in sorted(named.items()):
                with self.subTest(template=template, permutation=name):
                    _assert_parity(self, fixture, permutation)

    def test_edge_order_uses_integer_model_data_vocabulary_ids(self):
        fixture = _fixture("RR")
        reversed_edges = tuple(reversed(_edge_tuples(fixture.graph)))
        result = canonicalize_graph(_with_edges(fixture.graph, reversed_edges))
        self.assertEqual(result.edge_index, fixture.target.edge_index)
        self.assertEqual(result.edge_type_ids, fixture.target.edge_type_ids)
        triples = tuple(zip(result.edge_index[0], result.edge_type_ids, result.edge_index[1]))
        self.assertEqual(triples, tuple(sorted(triples)))
        self.assertTrue(all(0 <= value < len(EDGE_TYPES.tokens) for value in result.edge_type_ids))


class CanonicalizationRejectionTests(unittest.TestCase):
    def assertCode(self, graph, code):
        with self.assertRaises(GraphEncoderError) as caught:
            canonicalize_graph(graph)
        self.assertEqual(caught.exception.code, code)

    def test_every_documented_failure_code_has_a_negative_case(self):
        e = _fixture("E").graph
        r = _fixture("R").graph
        ee = _fixture("EE").graph
        placed_on = EDGE_TYPES.id("placed_on")
        defined_in = EDGE_TYPES.id("defined_in")
        uses_profile = EDGE_TYPES.id("uses_profile")
        uses_axis = EDGE_TYPES.id("uses_axis")
        depends_on = EDGE_TYPES.id("depends_on")

        e_edges = _edge_tuples(e)
        r_edges = _edge_tuples(r)
        ee_edges = _edge_tuples(ee)
        e_plane = _indices(e, "reference_plane")[0]
        e_sketch = _indices(e, "sketch")[0]
        e_profile = _indices(e, "profile")[0]
        e_operation = _indices(e, "extrude")[0]
        r_plane = _indices(r, "reference_plane")[0]
        r_sketch = _indices(r, "sketch")[0]
        r_profile = _indices(r, "profile")[0]
        r_axis = _indices(r, "axis")[0]
        r_operation = _indices(r, "revolve")[0]
        ee_operations = _indices(ee, "extrude")
        ee_profiles = _indices(ee, "profile")
        ee_sketches = _indices(ee, "sketch")

        malformed_nodes = replace(e, geometry=e.geometry[:-1])
        bad_node_ids = list(e.node_type_ids)
        bad_node_ids[e_sketch] = 999
        invalid_node = replace(e, node_type_ids=tuple(bad_node_ids))
        bad_edge_types = list(e.edge_type_ids)
        bad_edge_types[0] = 999
        invalid_edge = replace(e, edge_type_ids=tuple(bad_edge_types))
        malformed_edge_index = replace(e, edge_index=(e.edge_index[0][:-1], e.edge_index[1]))
        out_of_range = _with_edges(e, e_edges + [(e_sketch, len(e.node_type_ids), placed_on)])
        incompatible = _with_edges(e, e_edges + [(e_operation, e_plane, placed_on)])
        duplicate = _with_edges(e, e_edges + [e_edges[0]])
        self_edge = _with_edges(e, e_edges + [(e_sketch, e_sketch, placed_on)])

        no_plane = GraphCanonicalizationInput(
            (NODE_TYPES.id("sketch"),),
            ((), ()),
            (),
            (e.categorical_attributes[e_sketch],),
            (e.geometry[e_sketch],),
            (e.geometry_mask[e_sketch],),
        )
        no_operation = GraphCanonicalizationInput(
            e.node_type_ids[:-1],
            (
                tuple(item[0] for item in e_edges if item[0] != e_operation),
                tuple(item[1] for item in e_edges if item[0] != e_operation),
            ),
            tuple(item[2] for item in e_edges if item[0] != e_operation),
            e.categorical_attributes[:-1],
            e.geometry[:-1],
            e.geometry_mask[:-1],
        )

        branch = _append_node(ee, NODE_TYPES.id("extrude"), ee_operations[0])
        branch_operation = len(branch.node_type_ids) - 1
        branch = _with_edges(
            branch,
            ee_edges
            + [
                (branch_operation, ee_operations[0], depends_on),
                (branch_operation, ee_operations[1], depends_on),
            ],
        )
        cycle = _with_edges(
            ee, ee_edges + [(ee_operations[0], ee_operations[1], depends_on)]
        )
        disconnected = _append_node(ee, NODE_TYPES.id("extrude"), ee_operations[0])
        ambiguous = _with_edges(
            ee, [item for item in ee_edges if item[2] != depends_on]
        )
        missing_profile = _with_edges(
            e, [item for item in e_edges if item[2] != uses_profile]
        )
        invalid_profile = _with_edges(
            e,
            _replace_edge(
                e_edges, uses_profile, source=e_operation, new_destination=e_sketch
            ),
        )
        missing_profile_defined = _with_edges(
            e, [item for item in e_edges if item[2] != defined_in]
        )
        invalid_profile_sketch = _with_edges(
            e,
            _replace_edge(
                e_edges, defined_in, source=e_profile, new_destination=e_plane
            ),
        )
        missing_axis = _with_edges(
            r, [item for item in r_edges if item[2] != uses_axis]
        )
        invalid_axis = _with_edges(
            r,
            _replace_edge(
                r_edges, uses_axis, source=r_operation, new_destination=r_profile
            ),
        )
        missing_axis_defined = _with_edges(
            r,
            [
                item
                for item in r_edges
                if not (item[0] == r_axis and item[2] == defined_in)
            ],
        )
        invalid_axis_sketch = _with_edges(
            r,
            _replace_edge(
                r_edges, defined_in, source=r_axis, new_destination=r_plane
            ),
        )
        mismatch = _append_node(r, NODE_TYPES.id("sketch"), r_sketch)
        extra_sketch = len(mismatch.node_type_ids) - 1
        mismatch_edges = _replace_edge(
            r_edges, defined_in, source=r_axis, new_destination=extra_sketch
        )
        mismatch = _with_edges(
            mismatch, mismatch_edges + [(extra_sketch, r_plane, placed_on)]
        )
        missing_placement = _with_edges(
            e, [item for item in e_edges if item[2] != placed_on]
        )
        nonplane_placement = _with_edges(
            e,
            _replace_edge(
                e_edges, placed_on, source=e_sketch, new_destination=e_profile
            ),
        )
        wrong_plane = _append_node(ee, NODE_TYPES.id("reference_plane"), e_plane)
        second_plane = len(wrong_plane.node_type_ids) - 1
        wrong_plane = _with_edges(
            wrong_plane,
            _replace_edge(
                ee_edges,
                placed_on,
                source=ee_sketches[1],
                new_destination=second_plane,
            ),
        )
        multiply_assigned = _with_edges(
            ee,
            _replace_edge(
                ee_edges,
                uses_profile,
                source=ee_operations[1],
                new_destination=ee_profiles[0],
            ),
        )
        unassigned = _append_node(e, NODE_TYPES.id("sketch"), e_sketch)
        unassigned_sketch = len(unassigned.node_type_ids) - 1
        unassigned = _with_edges(
            unassigned, e_edges + [(unassigned_sketch, e_plane, placed_on)]
        )

        scope_masking_cases = (
            (UNSUPPORTED_OPERATION_COUNT, branch),
            (UNSUPPORTED_OPERATION_COUNT, disconnected),
            (UNSUPPORTED_PLANE_COUNT, wrong_plane),
        )

        cases = (
            (MALFORMED_NODE_FIELDS, malformed_nodes),
            (INVALID_NODE_TYPE_ID, invalid_node),
            (INVALID_EDGE_TYPE_ID, invalid_edge),
            (MALFORMED_EDGE_INDEX, malformed_edge_index),
            (OUT_OF_RANGE_EDGE_ENDPOINT, out_of_range),
            (INCOMPATIBLE_TYPED_EDGE, incompatible),
            (DUPLICATE_EDGE, duplicate),
            (SELF_EDGE, self_edge),
            (UNSUPPORTED_PLANE_COUNT, no_plane),
            (UNSUPPORTED_OPERATION_COUNT, no_operation),
            (OPERATION_CHAIN_CYCLE, cycle),
            (AMBIGUOUS_OPERATION_CHAIN, ambiguous),
            (MISSING_OR_MULTIPLE_OPERATION_PROFILE, missing_profile),
            (INVALID_PROFILE_TARGET_TYPE, invalid_profile),
            (MISSING_OR_MULTIPLE_PROFILE_DEFINED_IN, missing_profile_defined),
            (INVALID_PROFILE_SKETCH_TARGET_TYPE, invalid_profile_sketch),
            (MISSING_FORBIDDEN_OR_MULTIPLE_OPERATION_AXIS, missing_axis),
            (INVALID_AXIS_TARGET_TYPE, invalid_axis),
            (MISSING_OR_MULTIPLE_AXIS_DEFINED_IN, missing_axis_defined),
            (INVALID_AXIS_SKETCH_TARGET_TYPE, invalid_axis_sketch),
            (AXIS_PROFILE_SKETCH_MISMATCH, mismatch),
            (MISSING_OR_MULTIPLE_SKETCH_PLACED_ON, missing_placement),
            (PLACEMENT_ON_NON_PLANE_NODE, nonplane_placement),
            (MULTIPLY_ASSIGNED_NODE, multiply_assigned),
            (UNASSIGNED_NODE, unassigned),
        )
        self.assertEqual({item[0] for item in cases}, set(REACHABLE_FAILURE_CODES))
        self.assertEqual(
            set(REACHABLE_FAILURE_CODES) | set(DEFENSIVE_ONLY_FAILURE_CODES),
            set(FAILURE_CODES),
        )
        self.assertFalse(
            set(REACHABLE_FAILURE_CODES) & set(DEFENSIVE_ONLY_FAILURE_CODES)
        )
        for code, graph in cases:
            with self.subTest(code=code):
                self.assertCode(graph, code)

        # Requirement: an out-of-scope graph must report the scope violation
        # rather than a downstream chain or placement defect.
        for code, graph in scope_masking_cases:
            with self.subTest(scope=code):
                self.assertCode(graph, code)

    def test_defensive_only_codes_are_unreachable_but_still_guarded(self):
        """The three gated codes stay covered by calling the guards directly."""

        operations = (0, 1, 2)
        branching = {0: (), 1: (0,), 2: (0,)}
        with self.assertRaises(GraphEncoderError) as raised:
            _operation_order(operations, _outgoing_to_by_source_type(branching))
        self.assertEqual(raised.exception.code, OPERATION_CHAIN_BRANCHING)

        disconnected = {0: (), 1: (0,), 2: ()}
        with self.assertRaises(GraphEncoderError) as raised:
            _operation_order(operations, _outgoing_to_by_source_type(disconnected))
        self.assertEqual(raised.exception.code, DISCONNECTED_OPERATION_CHAIN)

        with self.assertRaises(GraphEncoderError) as raised:
            _assert_shared_plane(0, (0, 1))
        self.assertEqual(raised.exception.code, PLACEMENT_ON_WRONG_SHARED_PLANE)

        with self.assertRaises(GraphEncoderError) as raised:
            _assert_shared_plane(0, (1, 1))
        self.assertEqual(raised.exception.code, PLACEMENT_ON_WRONG_SHARED_PLANE)

    def test_controlled_scope_is_rejected_before_relational_recovery(self):
        """Three operations are out of scope whether or not the chain is valid."""

        ee = _fixture("EE").graph
        ee_edges = _edge_tuples(ee)
        operations = _indices(ee, "extrude")
        third = _append_node(ee, NODE_TYPES.id("extrude"), operations[0])
        third_operation = len(third.node_type_ids) - 1
        valid_chain = _with_edges(
            third, ee_edges + [(third_operation, operations[1], EDGE_TYPES.id("depends_on"))]
        )
        self.assertCode(valid_chain, UNSUPPORTED_OPERATION_COUNT)
        self.assertCode(third, UNSUPPORTED_OPERATION_COUNT)

    def test_invalid_operation_chains_are_not_resolved_by_node_row_order(self):
        ee = _fixture("EE").graph
        depends_on = EDGE_TYPES.id("depends_on")
        operations = _indices(ee, "extrude")
        edges = _edge_tuples(ee)

        branch = _append_node(ee, NODE_TYPES.id("extrude"), operations[0])
        branch_operation = len(branch.node_type_ids) - 1
        branch = _with_edges(
            branch,
            edges
            + [
                (branch_operation, operations[0], depends_on),
                (branch_operation, operations[1], depends_on),
            ],
        )
        cycle = _with_edges(
            ee, edges + [(operations[0], operations[1], depends_on)]
        )
        ambiguous = _with_edges(
            ee, [item for item in edges if item[2] != depends_on]
        )

        # `branch` needs a third operation, so the controlled-scope gate now
        # rejects it first. The point of this test is unchanged: whichever code
        # applies must not depend on incoming node row order.
        for code, graph in (
            (UNSUPPORTED_OPERATION_COUNT, branch),
            (OPERATION_CHAIN_CYCLE, cycle),
            (AMBIGUOUS_OPERATION_CHAIN, ambiguous),
        ):
            permutations = (
                tuple(range(len(graph.node_type_ids))),
                tuple(reversed(range(len(graph.node_type_ids)))),
            )
            for permutation in permutations:
                with self.subTest(code=code, permutation=permutation):
                    self.assertCode(_permute(graph, permutation), code)

    def test_extra_reference_plane_is_a_controlled_scope_error(self):
        graph = _fixture("E").graph
        plane = _indices(graph, "reference_plane")[0]
        graph = _append_node(graph, NODE_TYPES.id("reference_plane"), plane)
        self.assertCode(graph, UNSUPPORTED_PLANE_COUNT)


class CanonicalizationStaticContractTests(unittest.TestCase):
    def test_public_input_excludes_chronology_targets_and_batching(self):
        fields = set(GraphCanonicalizationInput.__dataclass_fields__)
        self.assertEqual(
            fields,
            {
                "node_type_ids",
                "edge_index",
                "edge_type_ids",
                "categorical_attributes",
                "geometry",
                "geometry_mask",
            },
        )
        forbidden = {
            "operation_sequence",
            "target",
            "partition",
            "split",
            "family_template",
            "node_ids",
            "padding",
            "graph_offsets",
            "edge_offsets",
            "node_graph_ids",
        }
        self.assertTrue(fields.isdisjoint(forbidden))
        self.assertEqual(tuple(inspect.signature(canonicalize_graph).parameters), ("graph",))

    def test_batch_shaped_nested_node_fields_are_rejected(self):
        graph = _fixture("E").graph
        batch_shaped = replace(
            graph,
            node_type_ids=(graph.node_type_ids,),
            categorical_attributes=(graph.categorical_attributes,),
            geometry=(graph.geometry,),
            geometry_mask=(graph.geometry_mask,),
        )
        with self.assertRaises(GraphEncoderError) as caught:
            canonicalize_graph(batch_shaped)
        self.assertEqual(caught.exception.code, INVALID_NODE_TYPE_ID)

    def test_c2_source_has_no_loader_or_position_dependency(self):
        import prototype.graph_encoder.canonicalization as module

        path = Path(module.__file__)
        source_text = path.read_text(encoding="utf-8")
        tree = ast.parse(source_text)
        forbidden_tokens = (
            "load_train",
            "load_development",
            "partition_family_ids",
            "load_partition_physical_examples",
            "load_physical_examples",
            "graph_offsets",
            "node_graph_ids",
        )
        for token in forbidden_tokens:
            self.assertNotIn(token, source_text)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                rendered = ast.dump(node)
                self.assertNotIn("loader", rendered)
                self.assertNotIn("partitions", rendered)


if __name__ == "__main__":
    unittest.main()
