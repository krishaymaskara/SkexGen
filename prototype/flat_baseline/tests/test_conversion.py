"""Focused Phase B strict-conversion tests."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass, replace
import math
import tempfile
from types import SimpleNamespace
import unittest

from prototype.controlled_data.factors import PrimitiveFamily
from prototype.flat_baseline.conversion import (
    ConversionFailure,
    ConversionResult,
    LayerValidity,
    _segments_intersect,
    validate_and_convert_raw_prediction,
)
from prototype.model_data.loader import load_physical_examples
from prototype.model_data.tests.fixtures import source, write_physical_corpus
from prototype.model_data.vocab import (
    BOOLEAN_MODES,
    EDGE_TYPES,
    NODE_TYPES,
    OPERATION_TYPES,
)

RAW_PREFIX_FEEDBACK = "raw_argmax_with_derived_geometry_mask"
REQUESTED_LENGTH_TERMINATION = "requested_node_count_reached"


@dataclass(frozen=True)
class RawDecodedNode:
    position: int
    node_type_id: int
    categorical_ids: tuple
    normalized_geometry: tuple
    derived_geometry_mask: tuple


@dataclass(frozen=True)
class RawDecodedEdge:
    source_index: int
    target_index: int
    presence_logit: float
    present: bool
    edge_type_id: int


@dataclass(frozen=True)
class RawDecodedPointer:
    query_index: int
    selected_node_index: int
    selected_logit: float


@dataclass(frozen=True)
class RawDecodedPrediction:
    latent_indices: tuple
    node_count: int
    node_count_source: str
    termination_reason: str
    termination_is_learned: bool
    prefix_feedback: str
    raw_nodes: tuple
    raw_edges: tuple
    predicted_operation_node_indices: tuple
    predicted_operation_count: int
    operation_count_exceeds_limit: bool
    raw_operation_pointers: tuple


def _raw_from_target(target, max_operations=2):
    count = len(target.node_type_ids)
    typed_edges = {
        (source_index, target_index): edge_type
        for source_index, target_index, edge_type in zip(
            target.edge_index[0], target.edge_index[1], target.edge_type_ids
        )
    }
    nodes = tuple(
        RawDecodedNode(index, node_type, attributes, geometry, mask)
        for index, (node_type, attributes, geometry, mask) in enumerate(
            zip(
                target.node_type_ids,
                target.categorical_attributes,
                target.geometry,
                target.geometry_mask,
            )
        )
    )
    edges = tuple(
        RawDecodedEdge(
            source_index,
            target_index,
            1.0 if (source_index, target_index) in typed_edges else -1.0,
            (source_index, target_index) in typed_edges,
            typed_edges.get((source_index, target_index), 0),
        )
        for source_index in range(count)
        for target_index in range(count)
    )
    operation_indices = tuple(
        index
        for index, node_type in enumerate(target.node_type_ids)
        if NODE_TYPES.tokens[node_type] in ("extrude", "revolve")
    )
    pointers = tuple(
        RawDecodedPointer(query, selected, 1.0)
        for query, selected in enumerate(
            target.operation_sequence[:max_operations]
        )
    )
    return RawDecodedPrediction(
        (3, 7),
        count,
        "caller_supplied",
        REQUESTED_LENGTH_TERMINATION,
        False,
        RAW_PREFIX_FEEDBACK,
        nodes,
        edges,
        operation_indices,
        len(operation_indices),
        len(operation_indices) > max_operations,
        pointers,
    )


def _with_node(raw, index, **changes):
    nodes = list(raw.raw_nodes)
    nodes[index] = replace(nodes[index], **changes)
    return replace(raw, raw_nodes=tuple(nodes))


def _with_edge(raw, index, **changes):
    edges = list(raw.raw_edges)
    edges[index] = replace(edges[index], **changes)
    return replace(raw, raw_edges=tuple(edges))


def _with_pointer(raw, index, **changes):
    pointers = list(raw.raw_operation_pointers)
    pointers[index] = replace(pointers[index], **changes)
    return replace(raw, raw_operation_pointers=tuple(pointers))


def _with_geometry(raw, node_index, updates):
    geometry = list(raw.raw_nodes[node_index].normalized_geometry)
    for channel, value in updates.items():
        geometry[channel] = value
    return _with_node(raw, node_index, normalized_geometry=tuple(geometry))


def _without_fields(raw, *names):
    values = {
        name: value for name, value in vars(raw).items()
        if name not in names
    }
    return SimpleNamespace(**values)


class ConversionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        write_physical_corpus(
            cls.temporary.name,
            (
                source("E", PrimitiveFamily.RECTANGLE_LINES),
                source("R", PrimitiveFamily.CIRCLE),
                source("EE", PrimitiveFamily.CAPSULE_LINE_ARC),
                source("ER", PrimitiveFamily.RECTANGLE_LINES),
                source("RE", PrimitiveFamily.CIRCLE),
                source("RR", PrimitiveFamily.CAPSULE_LINE_ARC),
            ),
        )
        cls.examples = {
            item.metadata.operation_template: item
            for item in load_physical_examples(cls.temporary.name)
        }

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def raw(self, template="ER"):
        return _raw_from_target(self.examples[template].target)

    def test_all_templates_and_profiles_convert_exactly(self):
        for template, example in self.examples.items():
            result = validate_and_convert_raw_prediction(
                _raw_from_target(example.target), max_operations=2
            )
            self.assertTrue(result.raw_integrity.valid, template)
            self.assertTrue(result.reconstruction_target.valid, template)
            self.assertTrue(result.controlled_domain.valid, template)
            self.assertEqual(result.reconstruction_target_candidate, example.target)
            self.assertIsNone(result.primary_failure)
            self.assertEqual(result.secondary_failures, ())

    def test_real_phase_a_record_integration_when_pytorch_is_available(self):
        try:
            from prototype.flat_baseline.autonomous import (
                RawDecodedEdge as ActualEdge,
                RawDecodedNode as ActualNode,
                RawDecodedPointer as ActualPointer,
                RawDecodedPrediction as ActualPrediction,
            )
        except ModuleNotFoundError as exc:
            if exc.name == "torch":
                self.skipTest("real Phase A integration requires PyTorch")
            raise
        raw = self.raw("ER")
        actual = ActualPrediction(
            raw.latent_indices,
            raw.node_count,
            raw.node_count_source,
            raw.termination_reason,
            raw.termination_is_learned,
            raw.prefix_feedback,
            tuple(ActualNode(*vars(node).values()) for node in raw.raw_nodes),
            tuple(ActualEdge(*vars(edge).values()) for edge in raw.raw_edges),
            raw.predicted_operation_node_indices,
            raw.predicted_operation_count,
            raw.operation_count_exceeds_limit,
            tuple(
                ActualPointer(*vars(pointer).values())
                for pointer in raw.raw_operation_pointers
            ),
        )
        result = validate_and_convert_raw_prediction(actual, max_operations=2)
        self.assertEqual(
            result.reconstruction_target_candidate,
            self.examples["ER"].target,
        )

    def test_validity_implications_and_immutable_records(self):
        results = (
            validate_and_convert_raw_prediction(self.raw(), max_operations=2),
            validate_and_convert_raw_prediction(
                _with_node(self.raw(), 0, node_type_id=0), max_operations=2
            ),
            validate_and_convert_raw_prediction(
                _with_node(
                    self.raw(), 0,
                    categorical_ids=(
                        *self.raw().raw_nodes[0].categorical_ids[:3],
                        self.raw().raw_nodes[0].categorical_ids[3],
                        2, 1, 1, 1, 1,
                    ),
                ),
                max_operations=2,
            ),
        )
        for result in results:
            if result.controlled_domain.valid:
                self.assertTrue(result.reconstruction_target.valid)
            if result.reconstruction_target.valid:
                self.assertTrue(result.raw_integrity.valid)
        with self.assertRaises(FrozenInstanceError):
            results[0].raw_integrity.valid = False
        for record in (
            LayerValidity(True, ()),
            ConversionFailure("x", "x", "x", "x", "x"),
            results[0],
        ):
            self.assertIsNotNone(hash(record))

    def test_raw_valid_target_invalid_has_no_candidate(self):
        raw = _with_node(self.raw(), 0, node_type_id=0)
        result = validate_and_convert_raw_prediction(raw, max_operations=2)
        self.assertTrue(result.raw_integrity.valid)
        self.assertFalse(result.reconstruction_target.valid)
        self.assertFalse(result.controlled_domain.valid)
        self.assertIsNone(result.reconstruction_target_candidate)
        self.assertEqual(result.primary_failure.code, "node_type_sentinel")

    def test_target_valid_domain_invalid_retains_candidate(self):
        raw = self.raw()
        attributes = list(raw.raw_nodes[0].categorical_ids)
        attributes[4] = 2
        raw = _with_node(raw, 0, categorical_ids=tuple(attributes))
        result = validate_and_convert_raw_prediction(raw, max_operations=2)
        self.assertTrue(result.raw_integrity.valid)
        self.assertTrue(result.reconstruction_target.valid)
        self.assertFalse(result.controlled_domain.valid)
        self.assertIsNotNone(result.reconstruction_target_candidate)
        self.assertEqual(result.primary_failure.code, "node_category_applicability")

    def test_malformed_envelope_and_inconsistent_counts(self):
        result = validate_and_convert_raw_prediction(object(), max_operations=2)
        self.assertFalse(result.raw_integrity.valid)
        self.assertIsNone(result.reconstruction_target_candidate)
        self.assertEqual(result.primary_failure.code, "missing_raw_field")

        raw = replace(self.raw(), node_count=self.raw().node_count + 1)
        result = validate_and_convert_raw_prediction(raw, max_operations=2)
        self.assertIn("node_count_mismatch", result.raw_integrity.failure_codes)

    def test_remaining_registered_failure_codes_are_reachable(self):
        raw = self.raw()
        cases = []
        cases.append((None, "missing_raw_prediction"))
        cases.append((replace(raw, raw_edges=list(raw.raw_edges)),
                      "invalid_raw_field_type"))
        cases.append((replace(raw, node_count=0), "invalid_node_count"))
        cases.append((_with_node(raw, 0, position=1), "invalid_node_position"))
        cases.append((_with_node(raw, 0, categorical_ids=(1,)),
                      "invalid_categorical_width"))
        cases.append((_with_node(raw, 0, normalized_geometry=(0.0,)),
                      "invalid_geometry_width"))
        cases.append((_with_node(raw, 0, derived_geometry_mask=(False,)),
                      "invalid_geometry_mask_width"))
        cases.append((replace(raw, predicted_operation_count=0),
                      "inconsistent_operation_metadata"))
        cases.append((_with_node(raw, 0, node_type_id=len(NODE_TYPES.tokens)),
                      "invalid_node_type_id"))
        for changed, expected in cases:
            result = validate_and_convert_raw_prediction(changed, max_operations=2)
            self.assertIn(expected, result.raw_integrity.failure_codes
                          + result.reconstruction_target.failure_codes)

        plane = raw.raw_nodes[0]
        plane_only = RawDecodedPrediction(
            raw.latent_indices, 1, raw.node_count_source, raw.termination_reason,
            raw.termination_is_learned, raw.prefix_feedback,
            (replace(plane, position=0),),
            (RawDecodedEdge(0, 0, -1.0, False, 0),),
            (), 0, False, (),
        )
        result = validate_and_convert_raw_prediction(plane_only, max_operations=2)
        self.assertEqual(result.primary_failure.code, "unsupported_operation_count")

        attributes = list(raw.raw_nodes[1].categorical_ids)
        attributes[4:8] = (2, 3, 1, 1)
        result = validate_and_convert_raw_prediction(
            _with_node(raw, 1, categorical_ids=tuple(attributes)),
            max_operations=2,
        )
        self.assertEqual(result.primary_failure.code, "unsupported_profile_pattern")
        self.assertNotIn(
            "geometry_mask_applicability",
            result.controlled_domain.failure_codes,
        )
        self.assertNotIn(
            "invalid_primitive_geometry",
            result.controlled_domain.failure_codes,
        )

        controlled_cases = (
            (_with_geometry(raw, 0, {0: 1.1}), "geometry_out_of_range"),
            (_with_geometry(raw, 0, {3: 0.0}),
             "invalid_reference_plane_geometry"),
            (_with_geometry(self.raw("R"), 3, {36: 0.0}),
             "invalid_axis_geometry"),
            (_with_geometry(raw, raw.predicted_operation_node_indices[0], {37: 0.0}),
             "invalid_operation_parameter"),
        )
        for changed, expected in controlled_cases:
            result = validate_and_convert_raw_prediction(changed, max_operations=2)
            self.assertEqual(result.primary_failure.code, expected)

        absent_index = next(
            index for index, edge in enumerate(raw.raw_edges) if not edge.present
        )
        unexpected = _with_edge(
            raw, absent_index, presence_logit=1.0, present=True,
            edge_type_id=EDGE_TYPES.id("placed_on"),
        )
        result = validate_and_convert_raw_prediction(unexpected, max_operations=2)
        self.assertEqual(result.primary_failure.code, "unexpected_edge")

    def test_exact_phase_a_policy_strings_are_required(self):
        valid = validate_and_convert_raw_prediction(self.raw(), max_operations=2)
        self.assertTrue(valid.raw_integrity.valid)
        for field, alternative in (
            ("termination_reason", "other_termination"),
            ("prefix_feedback", "other_prefix_feedback"),
        ):
            raw = replace(self.raw(), **{field: alternative})
            first = validate_and_convert_raw_prediction(raw, max_operations=2)
            second = validate_and_convert_raw_prediction(raw, max_operations=2)
            self.assertEqual(first, second)
            self.assertEqual(first.primary_failure.code, "invalid_raw_metadata")
            self.assertEqual(first.primary_failure.stage, "raw_envelope")

    def test_edge_matrix_shape_order_and_phase_a_presence(self):
        raw = self.raw()
        incomplete = replace(raw, raw_edges=raw.raw_edges[:-1])
        result = validate_and_convert_raw_prediction(incomplete, max_operations=2)
        self.assertEqual(result.primary_failure.code, "invalid_edge_matrix")

    def test_canonical_sparse_edge_order_differs_without_mutating_raw(self):
        raw = self.raw("R")
        before = raw.raw_edges
        raw_present = tuple(
            (edge.source_index, edge.edge_type_id, edge.target_index)
            for edge in raw.raw_edges if edge.present
        )
        result = validate_and_convert_raw_prediction(raw, max_operations=2)
        candidate = result.reconstruction_target_candidate
        canonical = tuple(zip(
            candidate.edge_index[0], candidate.edge_type_ids,
            candidate.edge_index[1],
        ))
        self.assertNotEqual(raw_present, canonical)
        self.assertEqual(canonical, tuple(sorted(raw_present)))
        self.assertIs(raw.raw_edges, before)
        self.assertEqual(raw.raw_edges, before)
        repeated = validate_and_convert_raw_prediction(raw, max_operations=2)
        self.assertEqual(repeated.reconstruction_target_candidate, candidate)

        reordered = _with_edge(raw, 0, source_index=1)
        result = validate_and_convert_raw_prediction(reordered, max_operations=2)
        self.assertEqual(result.primary_failure.code, "invalid_edge_matrix_order")

        inconsistent = _with_edge(raw, 0, presence_logit=0.0, present=False)
        result = validate_and_convert_raw_prediction(inconsistent, max_operations=2)
        self.assertEqual(result.primary_failure.code, "invalid_edge_matrix")

    def test_nonfinite_geometry_edge_and_pointer_are_independent(self):
        raw = self.raw()
        geometry = list(raw.raw_nodes[0].normalized_geometry)
        geometry[0] = float("nan")
        raw = _with_node(raw, 0, normalized_geometry=tuple(geometry))
        raw = _with_edge(raw, 0, presence_logit=float("inf"))
        raw = _with_pointer(raw, 0, selected_logit=float("nan"))
        result = validate_and_convert_raw_prediction(raw, max_operations=2)
        codes = (result.primary_failure.code,) + tuple(
            item.code for item in result.secondary_failures
        )
        self.assertEqual(
            codes,
            (
                "nonfinite_geometry",
                "nonfinite_edge_logit",
                "nonfinite_pointer_logit",
            ),
        )
        self.assertIsNone(result.reconstruction_target_candidate)

    def test_record_family_gating_preserves_independent_numeric_failures(self):
        raw = self.raw()
        geometry = list(raw.raw_nodes[0].normalized_geometry)
        geometry[0] = math.nan
        malformed_pointer = replace(
            _with_node(raw, 0, normalized_geometry=tuple(geometry)),
            raw_operation_pointers=(object(),) + raw.raw_operation_pointers[1:],
        )
        malformed_pointer = _with_edge(
            malformed_pointer, 0, presence_logit=math.inf
        )
        result = validate_and_convert_raw_prediction(
            malformed_pointer, max_operations=2
        )
        self.assertEqual(
            tuple(f.code for f in (result.primary_failure,) + result.secondary_failures),
            ("invalid_pointer_records", "nonfinite_geometry", "nonfinite_edge_logit"),
        )

        malformed_edges = replace(
            _with_pointer(raw, 0, selected_logit=math.nan),
            raw_edges=raw.raw_edges[:-1],
        )
        result = validate_and_convert_raw_prediction(malformed_edges, max_operations=2)
        self.assertIn("nonfinite_pointer_logit", result.raw_integrity.failure_codes)

        malformed_nodes = replace(
            _with_edge(_with_pointer(raw, 0, selected_logit=math.nan), 0,
                       presence_logit=math.inf),
            raw_nodes=(object(),) + raw.raw_nodes[1:],
        )
        result = validate_and_convert_raw_prediction(malformed_nodes, max_operations=2)
        self.assertIn("nonfinite_edge_logit", result.raw_integrity.failure_codes)
        self.assertIn("nonfinite_pointer_logit", result.raw_integrity.failure_codes)

        mismatch = replace(raw, node_count=raw.node_count + 1)
        result = validate_and_convert_raw_prediction(mismatch, max_operations=2)
        self.assertEqual(result.raw_integrity.failure_codes, ("node_count_mismatch",))

    def test_pointer_record_shape_query_order_types_and_ranges(self):
        raw = self.raw()
        malformed = replace(
            raw, raw_operation_pointers=(object(),) + raw.raw_operation_pointers[1:]
        )
        result = validate_and_convert_raw_prediction(malformed, max_operations=2)
        self.assertEqual(result.primary_failure.code, "invalid_pointer_records")

        result = validate_and_convert_raw_prediction(
            _with_pointer(raw, 0, query_index=1), max_operations=2
        )
        self.assertEqual(result.primary_failure.code, "invalid_pointer_query_order")

        result = validate_and_convert_raw_prediction(
            _with_pointer(raw, 0, selected_node_index=True), max_operations=2
        )
        self.assertEqual(result.primary_failure.code, "invalid_pointer_type")

        result = validate_and_convert_raw_prediction(
            _with_pointer(raw, 0, selected_node_index=raw.node_count),
            max_operations=2,
        )
        self.assertEqual(result.primary_failure.code, "invalid_pointer_index")
        for invalid_query in (False, True, 0.0, "0"):
            result = validate_and_convert_raw_prediction(
                _with_pointer(raw, 0, query_index=invalid_query),
                max_operations=2,
            )
            self.assertEqual(
                result.primary_failure.code, "invalid_pointer_query_order"
            )

    def test_phase_a_numeric_fields_require_python_floats(self):
        raw = self.raw()
        for value in ("0.0", 0, True, object()):
            for changed in (
                _with_geometry(raw, 0, {0: value}),
                _with_edge(raw, 0, presence_logit=value),
                _with_pointer(raw, 0, selected_logit=value),
            ):
                result = validate_and_convert_raw_prediction(
                    changed, max_operations=2
                )
                self.assertEqual(
                    result.primary_failure.code, "invalid_raw_field_type"
                )
                self.assertEqual(result.primary_failure.stage, "raw_envelope")

    def test_malformed_metadata_collections_return_results(self):
        raw = self.raw()
        for field in (
            "latent_indices",
            "predicted_operation_node_indices",
            "raw_operation_pointers",
        ):
            changed = replace(raw, **{field: object()})
            first = validate_and_convert_raw_prediction(changed, max_operations=2)
            second = validate_and_convert_raw_prediction(changed, max_operations=2)
            self.assertEqual(first, second)
            self.assertEqual(first.primary_failure.code, "invalid_raw_field_type")
        malformed_element = replace(
            raw,
            raw_operation_pointers=(object(),) + raw.raw_operation_pointers[1:],
        )
        result = validate_and_convert_raw_prediction(
            malformed_element, max_operations=2
        )
        self.assertEqual(result.primary_failure.code, "invalid_pointer_records")

    def test_unaddressable_raw_nodes_do_not_break_pointer_validation(self):
        raw = self.raw()
        malformed_cases = (
            (
                replace(raw, raw_nodes=object()),
                replace(
                    _with_pointer(raw, 0, selected_node_index=True),
                    raw_nodes=object(),
                ),
            ),
            (
                replace(raw, raw_nodes=None),
                replace(
                    _with_pointer(raw, 0, selected_node_index=True),
                    raw_nodes=None,
                ),
            ),
            (
                _without_fields(raw, "raw_nodes"),
                _without_fields(
                    _with_pointer(raw, 0, selected_node_index=True),
                    "raw_nodes",
                ),
            ),
        )
        for valid_pointers, invalid_pointer_scalar in malformed_cases:
            valid_first = validate_and_convert_raw_prediction(
                valid_pointers, max_operations=2
            )
            valid_second = validate_and_convert_raw_prediction(
                valid_pointers, max_operations=2
            )
            self.assertIsInstance(valid_first, ConversionResult)
            self.assertEqual(valid_first, valid_second)
            self.assertNotIn(
                "invalid_pointer_index",
                valid_first.reconstruction_target.failure_codes,
            )

            invalid_first = validate_and_convert_raw_prediction(
                invalid_pointer_scalar, max_operations=2
            )
            invalid_second = validate_and_convert_raw_prediction(
                invalid_pointer_scalar, max_operations=2
            )
            self.assertIsInstance(invalid_first, ConversionResult)
            self.assertEqual(invalid_first, invalid_second)
            self.assertIn(
                "invalid_pointer_type",
                invalid_first.reconstruction_target.failure_codes,
            )
            self.assertNotIn(
                "invalid_pointer_index",
                invalid_first.reconstruction_target.failure_codes,
            )

    def test_missing_node_count_preserves_independent_validation(self):
        raw = self.raw()
        changed = replace(raw, termination_reason="invalid")
        changed = _with_edge(changed, 0, presence_logit=math.inf)
        changed = _with_pointer(changed, 0, selected_logit=math.nan)
        changed = _without_fields(changed, "node_count")
        first = validate_and_convert_raw_prediction(changed, max_operations=2)
        second = validate_and_convert_raw_prediction(changed, max_operations=2)
        self.assertIsInstance(first, ConversionResult)
        self.assertEqual(first, second)
        failures = (first.primary_failure,) + first.secondary_failures
        self.assertIn(
            ("missing_raw_field", "node_count"),
            tuple((failure.code, failure.location) for failure in failures),
        )
        self.assertIn(
            ("invalid_raw_metadata", "metadata"),
            tuple((failure.code, failure.location) for failure in failures),
        )
        self.assertIn(
            "nonfinite_edge_logit", first.raw_integrity.failure_codes
        )
        self.assertIn(
            "nonfinite_pointer_logit", first.raw_integrity.failure_codes
        )
        self.assertIsNone(first.reconstruction_target_candidate)

    def test_count_independent_metadata_survives_invalid_node_count(self):
        raw = replace(
            self.raw(),
            node_count=0,
            termination_reason="invalid",
            latent_indices=object(),
        )
        result = validate_and_convert_raw_prediction(raw, max_operations=2)
        failures = (result.primary_failure,) + result.secondary_failures
        self.assertIn(
            ("invalid_node_count", "node_count"),
            tuple((failure.code, failure.location) for failure in failures),
        )
        self.assertIn(
            ("invalid_raw_field_type", "latent_indices"),
            tuple((failure.code, failure.location) for failure in failures),
        )
        self.assertIn(
            ("invalid_raw_metadata", "metadata"),
            tuple((failure.code, failure.location) for failure in failures),
        )

    def test_illegal_and_sentinel_categorical_ids(self):
        raw = self.raw()
        attributes = list(raw.raw_nodes[0].categorical_ids)
        attributes[0] = len(OPERATION_TYPES.tokens)
        illegal = _with_node(raw, 0, categorical_ids=tuple(attributes))
        result = validate_and_convert_raw_prediction(illegal, max_operations=2)
        self.assertEqual(result.primary_failure.code, "invalid_categorical_id")

        attributes[0] = 0
        sentinel = _with_node(raw, 0, categorical_ids=tuple(attributes))
        result = validate_and_convert_raw_prediction(sentinel, max_operations=2)
        self.assertEqual(result.primary_failure.code, "categorical_pad_sentinel")

    def test_edge_endpoint_and_type_validation(self):
        raw = self.raw()
        present_index = next(
            index for index, edge in enumerate(raw.raw_edges) if edge.present
        )
        result = validate_and_convert_raw_prediction(
            _with_edge(raw, present_index, target_index=raw.node_count),
            max_operations=2,
        )
        self.assertEqual(result.primary_failure.code, "invalid_edge_matrix_order")
        result = validate_and_convert_raw_prediction(
            _with_edge(raw, present_index, edge_type_id=True), max_operations=2
        )
        self.assertEqual(result.primary_failure.code, "invalid_edge_type_id")
        result = validate_and_convert_raw_prediction(
            _with_edge(raw, present_index, edge_type_id=0), max_operations=2
        )
        self.assertEqual(result.primary_failure.code, "edge_type_sentinel")

    def test_controlled_grammar_category_mask_and_boolean_failures(self):
        raw = self.raw()
        grammar = _with_node(
            raw, 1, node_type_id=NODE_TYPES.id("profile")
        )
        result = validate_and_convert_raw_prediction(grammar, max_operations=2)
        self.assertEqual(result.primary_failure.code, "invalid_node_grammar")
        self.assertFalse(any(
            item.code == "geometry_mask_applicability"
            for item in result.secondary_failures
        ))

        attributes = list(raw.raw_nodes[1].categorical_ids)
        attributes[3] = 2
        category = _with_node(raw, 1, categorical_ids=tuple(attributes))
        result = validate_and_convert_raw_prediction(category, max_operations=2)
        self.assertEqual(result.primary_failure.code, "node_category_applicability")

        mask = list(raw.raw_nodes[1].derived_geometry_mask)
        mask[0] = True
        masked = _with_node(raw, 1, derived_geometry_mask=tuple(mask))
        result = validate_and_convert_raw_prediction(masked, max_operations=2)
        self.assertEqual(result.primary_failure.code, "geometry_mask_applicability")

        non_boolean = list(raw.raw_nodes[1].derived_geometry_mask)
        non_boolean[0] = 0
        result = validate_and_convert_raw_prediction(
            _with_node(raw, 1, derived_geometry_mask=tuple(non_boolean)),
            max_operations=2,
        )
        self.assertEqual(result.primary_failure.code, "invalid_geometry_mask_type")

        operation = self.examples["ER"].target.operation_sequence[0]
        attributes = list(raw.raw_nodes[operation].categorical_ids)
        attributes[1] = BOOLEAN_MODES.id("join")
        boolean = _with_node(raw, operation, categorical_ids=tuple(attributes))
        result = validate_and_convert_raw_prediction(boolean, max_operations=2)
        self.assertEqual(result.primary_failure.code, "invalid_boolean_sequence")

    def test_category_applicability_is_separate_from_semantic_agreement(self):
        raw = self.raw()
        operation = raw.predicted_operation_node_indices[0]
        attributes = list(raw.raw_nodes[operation].categorical_ids)
        attributes[0] = OPERATION_TYPES.id("revolve")
        result = validate_and_convert_raw_prediction(
            _with_node(raw, operation, categorical_ids=tuple(attributes)),
            max_operations=2,
        )
        self.assertEqual(
            tuple(f.code for f in (result.primary_failure,) + result.secondary_failures),
            ("operation_type_mismatch",),
        )

        sketch = 1
        attributes = list(raw.raw_nodes[sketch].categorical_ids)
        attributes[8] = 2  # "inner"
        result = validate_and_convert_raw_prediction(
            _with_node(raw, sketch, categorical_ids=tuple(attributes)),
            max_operations=2,
        )
        self.assertEqual(
            tuple(f.code for f in (result.primary_failure,) + result.secondary_failures),
            ("invalid_loop_constraint",),
        )

        attributes = list(raw.raw_nodes[operation].categorical_ids)
        attributes[2] = 1
        result = validate_and_convert_raw_prediction(
            _with_node(raw, operation, categorical_ids=tuple(attributes)),
            max_operations=2,
        )
        self.assertEqual(result.primary_failure.code, "node_category_applicability")

    def test_category_gating_is_per_node_and_loop_role_is_independent(self):
        raw = self.raw()
        sketch_attributes = list(raw.raw_nodes[1].categorical_ids)
        sketch_attributes[4:8] = (2, 3, 1, 1)
        operation = raw.predicted_operation_node_indices[0]
        changed = _with_node(
            raw, 1, categorical_ids=tuple(sketch_attributes)
        )
        changed = _with_geometry(changed, operation, {37: 0.0})
        result = validate_and_convert_raw_prediction(changed, max_operations=2)
        self.assertIn(
            "unsupported_profile_pattern",
            result.controlled_domain.failure_codes,
        )
        self.assertIn(
            "invalid_operation_parameter",
            result.controlled_domain.failure_codes,
        )
        self.assertNotIn(
            "geometry_mask_applicability",
            result.controlled_domain.failure_codes,
        )

        plane_attributes = list(raw.raw_nodes[0].categorical_ids)
        plane_attributes[3] = 1
        changed = _with_node(
            raw, 0, categorical_ids=tuple(plane_attributes)
        )
        changed = _with_geometry(changed, 0, {3: 0.0})
        result = validate_and_convert_raw_prediction(changed, max_operations=2)
        self.assertIn(
            "node_category_applicability",
            result.controlled_domain.failure_codes,
        )
        self.assertNotIn(
            "invalid_reference_plane_geometry",
            result.controlled_domain.failure_codes,
        )

        loop_attributes = list(raw.raw_nodes[1].categorical_ids)
        loop_attributes[8] = 2
        changed = _with_node(
            raw, 1, categorical_ids=tuple(loop_attributes)
        )
        geometry = changed.raw_nodes[1].normalized_geometry
        changed = _with_geometry(
            changed, 1, {11: geometry[9], 12: geometry[10]}
        )
        result = validate_and_convert_raw_prediction(changed, max_operations=2)
        self.assertIn(
            "invalid_loop_constraint", result.controlled_domain.failure_codes
        )
        self.assertIn(
            "invalid_primitive_geometry", result.controlled_domain.failure_codes
        )

    def test_adversarial_rectangle_and_capsule_geometry_is_rejected(self):
        rectangle = self.raw("E")
        sketch = 1
        cases = (
            ("zero_length_line",
             {11: rectangle.raw_nodes[sketch].normalized_geometry[9],
              12: rectangle.raw_nodes[sketch].normalized_geometry[10]}),
            ("bow_tie_self_intersection",
             {9: 0.0, 10: 0.0, 11: 1.0, 12: 1.0,
             15: 1.0, 16: 1.0, 17: 0.0, 18: 1.0,
             21: 0.0, 22: 1.0, 23: 1.0, 24: 0.0,
             27: 1.0, 28: 0.0, 29: 0.0, 30: 0.0}),
            ("two_disconnected_two_edge_cycles",
             {9: 0.0, 10: 0.0, 11: 1.0, 12: 0.0,
             15: 1.0, 16: 0.0, 17: 0.0, 18: 0.0,
             21: 0.25, 22: 0.5, 23: 0.75, 24: 0.5,
             27: 0.75, 28: 0.5, 29: 0.25, 30: 0.5}),
            ("open_loop", {11: 0.25, 12: 0.25}),
            ("zero_area",
             {9: 0.0, 10: 0.0, 11: 0.25, 12: 0.0,
              15: 0.25, 16: 0.0, 17: 0.5, 18: 0.0,
              21: 0.5, 22: 0.0, 23: 0.75, 24: 0.0,
              27: 0.75, 28: 0.0, 29: 0.0, 30: 0.0}),
        )
        for name, updates in cases:
            with self.subTest(name=name):
                result = validate_and_convert_raw_prediction(
                    _with_geometry(rectangle, sketch, updates), max_operations=2
                )
                self.assertEqual(
                    result.primary_failure.code, "invalid_primitive_geometry"
                )

    def test_nonadjacent_contact_and_collinear_overlap_intersect(self):
        self.assertTrue(
            _segments_intersect(
                (0.0, 0.0), (1.0, 0.0),
                (0.5, 0.0), (0.5, 1.0),
            )
        )
        self.assertTrue(
            _segments_intersect(
                (0.0, 0.0), (1.0, 0.0),
                (0.25, 0.0), (0.75, 0.0),
            )
        )

        capsule = self.raw("EE")
        sketch = 1
        geometry = capsule.raw_nodes[sketch].normalized_geometry
        capsule_cases = (
            ("coincident_arc_points", {11: geometry[9], 12: geometry[10]}),
            ("collinear_arc",
             {11: (geometry[9] + geometry[13]) / 2,
              12: (geometry[10] + geometry[14]) / 2}),
            ("disconnected_capsule", {21: 0.75, 22: 0.75}),
        )
        for name, updates in capsule_cases:
            with self.subTest(name=name):
                result = validate_and_convert_raw_prediction(
                    _with_geometry(capsule, sketch, updates), max_operations=2
                )
                self.assertEqual(
                    result.primary_failure.code, "invalid_primitive_geometry"
                )

    def test_missing_required_relationship_and_operation_limit(self):
        raw = self.raw()
        present_index = next(
            index for index, edge in enumerate(raw.raw_edges)
            if edge.present and EDGE_TYPES.tokens[edge.edge_type_id] == "placed_on"
        )
        missing = _with_edge(
            raw, present_index, presence_logit=-1.0, present=False
        )
        result = validate_and_convert_raw_prediction(missing, max_operations=2)
        self.assertEqual(result.primary_failure.code, "missing_required_edge")

        result = validate_and_convert_raw_prediction(
            _raw_from_target(self.examples["EE"].target, max_operations=1),
            max_operations=1,
        )
        self.assertEqual(result.primary_failure.code, "operation_limit_exceeded")

    def test_cross_record_consistency(self):
        raw = self.raw()
        raw = replace(
            raw,
            predicted_operation_node_indices=(
                raw.predicted_operation_node_indices[0] - 1,
                raw.predicted_operation_node_indices[1],
            ),
        )
        result = validate_and_convert_raw_prediction(raw, max_operations=2)
        self.assertEqual(result.primary_failure.code, "operation_metadata_mismatch")

    def test_stage8_dependency_inconsistency_after_stages_zero_through_seven_pass(self):
        raw = self.raw()
        dependency_index = next(
            index for index, edge in enumerate(raw.raw_edges)
            if edge.present and EDGE_TYPES.tokens[edge.edge_type_id] == "depends_on"
        )
        raw = _with_edge(
            raw, dependency_index, presence_logit=-1.0, present=False
        )
        result = validate_and_convert_raw_prediction(raw, max_operations=2)
        self.assertTrue(result.raw_integrity.valid)
        self.assertTrue(result.reconstruction_target.valid)
        self.assertFalse(result.controlled_domain.valid)
        self.assertEqual(
            result.primary_failure.code, "operation_edge_inconsistency"
        )
        self.assertEqual(
            result.primary_failure.stage, "cross_record_consistency"
        )
        self.assertEqual(result.secondary_failures, ())

    def test_stage8_metadata_survives_unrelated_stage7_edge_failure(self):
        raw = self.raw()
        placed_on_index = next(
            index for index, edge in enumerate(raw.raw_edges)
            if edge.present and EDGE_TYPES.tokens[edge.edge_type_id] == "placed_on"
        )
        raw = _with_edge(
            raw, placed_on_index, presence_logit=-1.0, present=False
        )
        raw = replace(
            raw,
            predicted_operation_node_indices=(
                raw.predicted_operation_node_indices[0] - 1,
                raw.predicted_operation_node_indices[1],
            ),
        )
        result = validate_and_convert_raw_prediction(raw, max_operations=2)
        self.assertEqual(result.primary_failure.code, "missing_required_edge")
        self.assertEqual(result.primary_failure.stage, "controlled_edges")
        self.assertIn(
            ("operation_metadata_mismatch", "cross_record_consistency"),
            tuple(
                (failure.code, failure.stage)
                for failure in result.secondary_failures
            ),
        )

    def test_stage8_pointer_consistency_is_independent_of_edges(self):
        raw = self.raw()
        raw = _with_pointer(
            raw,
            0,
            selected_node_index=raw.predicted_operation_node_indices[0] - 1,
        )
        result = validate_and_convert_raw_prediction(raw, max_operations=2)
        self.assertEqual(
            result.primary_failure.code, "operation_pointer_inconsistency"
        )
        self.assertEqual(
            result.primary_failure.stage, "cross_record_consistency"
        )

    def test_failure_order_and_details_are_repeatable(self):
        raw = self.raw()
        geometry = list(raw.raw_nodes[1].normalized_geometry)
        geometry[0] = math.inf
        raw = _with_node(raw, 1, normalized_geometry=tuple(geometry))
        raw = _with_edge(raw, 2, presence_logit=math.nan)
        first = validate_and_convert_raw_prediction(raw, max_operations=2)
        second = validate_and_convert_raw_prediction(raw, max_operations=2)
        self.assertEqual(first, second)
        self.assertEqual(first.primary_failure, second.primary_failure)
        self.assertEqual(first.secondary_failures, second.secondary_failures)
        for failure in (first.primary_failure,) + first.secondary_failures:
            self.assertNotIn("/Users/", failure.detail)
            self.assertNotIn("0x", failure.detail)


if __name__ == "__main__":
    unittest.main()
