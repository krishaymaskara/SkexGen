"""Focused and adversarial Phase B prediction-metric tests."""

from __future__ import annotations

from dataclasses import fields, replace
import inspect
import math
import tempfile
from types import SimpleNamespace
import unittest

from prototype.controlled_data.factors import PrimitiveFamily
from prototype.flat_baseline.metrics import (
    AggregateEvaluation,
    PredictionMetrics,
    aggregate_prediction_metrics,
    evaluate_prediction,
)
from prototype.flat_baseline.tests.test_conversion import (
    RawDecodedEdge,
    RawDecodedNode,
    RawDecodedPointer,
    _raw_from_target,
    _with_edge,
    _with_geometry,
    _with_node,
    _with_pointer,
)
from prototype.model_data.geometry import (
    GEOMETRY_CHANNEL_SCALES,
    GEOMETRY_WIDTH,
)
from prototype.model_data.loader import load_physical_examples
from prototype.model_data.records import ReconstructionTarget
from prototype.model_data.tests.fixtures import source, write_physical_corpus
from prototype.model_data.vocab import (
    CATEGORICAL_ATTRIBUTE_FIELDS,
    EDGE_TYPES,
    NODE_TYPES,
)


def _without_field(raw, name):
    return SimpleNamespace(**{
        field: value for field, value in vars(raw).items()
        if field != name
    })


def _all_edges_absent(raw):
    return replace(
        raw,
        raw_edges=tuple(
            replace(edge, presence_logit=-1.0, present=False)
            for edge in raw.raw_edges
        ),
    )


def _target_without_edges(target):
    return replace(
        target,
        edge_index=((), ()),
        edge_type_ids=(),
    )


def _target_without_operations(target):
    return replace(target, operation_sequence=())


class PredictionMetricTests(unittest.TestCase):
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
        examples = load_physical_examples(cls.temporary.name)
        cls.examples = {
            item.metadata.operation_template: item for item in examples
        }

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def evaluate(self, raw=None, template="ER", target=None, family_id=None):
        example = self.examples[template]
        if raw is None:
            raw = _raw_from_target(example.target)
        return evaluate_prediction(
            raw,
            example.target if target is None else target,
            family_id=family_id or example.physical_family_id,
            max_operations=2,
        )

    def metadata_for(self, family_id, template="ER"):
        return replace(
            self.examples[template], physical_family_id=family_id
        )

    def test_perfect_predictions_and_all_authoritative_strata(self):
        records = []
        for template, example in self.examples.items():
            raw = _raw_from_target(example.target)
            before = (raw, example.target, example.metadata)
            metric = self.evaluate(raw, template)
            self.assertIsInstance(metric, PredictionMetrics)
            self.assertTrue(metric.raw_completion)
            self.assertTrue(metric.raw_integrity.valid)
            self.assertTrue(metric.reconstruction_target.valid)
            self.assertTrue(metric.controlled_domain.valid)
            self.assertIsNone(metric.primary_failure)
            self.assertEqual(metric.secondary_failures, ())
            self.assertEqual(
                metric.nodes.node_type_correct_positions,
                len(example.target.node_type_ids),
            )
            self.assertEqual(metric.nodes.node_type_token_accuracy, 1.0)
            self.assertTrue(metric.nodes.exact_node_type_sequence)
            self.assertEqual(
                tuple(item.field_name for item in metric.nodes.categorical_fields),
                CATEGORICAL_ATTRIBUTE_FIELDS,
            )
            self.assertTrue(all(
                item.accuracy == 1.0
                for item in metric.nodes.categorical_fields
            ))
            self.assertTrue(metric.nodes.exact_nine_attribute_match)
            self.assertTrue(metric.nodes.exact_complete_ten_field_match)
            self.assertEqual(metric.nodes.geometry_mask_cell_accuracy, 1.0)
            self.assertTrue(metric.nodes.exact_geometry_mask_match)
            self.assertEqual(
                metric.geometry.finite_predicted_applicable_geometry_channel_count,
                metric.geometry.target_applicable_geometry_channel_count,
            )
            self.assertEqual(
                metric.geometry.nonfinite_predicted_applicable_geometry_channel_count,
                0,
            )
            self.assertEqual(
                metric.geometry.unavailable_predicted_applicable_geometry_channel_count,
                0,
            )
            self.assertEqual(
                metric.geometry.finite_applicable_geometry_absolute_error_sum,
                0.0,
            )
            self.assertEqual(metric.geometry.finite_applicable_geometry_mae, 0.0)
            self.assertEqual(metric.geometry.finite_applicable_geometry_rmse, 0.0)
            self.assertTrue(metric.geometry.complete_finite_geometry)
            expected_operations = tuple(
                "extrude" if item == "E" else "revolve"
                for item in template
            )
            self.assertEqual(
                metric.operations.predicted_operation_type_sequence,
                expected_operations,
            )
            self.assertTrue(metric.operations.exact_operation_type_sequence)
            self.assertEqual(metric.pointers.overall_pointer_accuracy, 1.0)
            self.assertEqual(metric.pointers.conditional_pointer_accuracy, 1.0)
            self.assertTrue(
                metric.pointers.exact_operation_sequence_pointer_match
            )
            self.assertEqual(metric.edges.precision, 1.0)
            self.assertEqual(metric.edges.recall, 1.0)
            self.assertEqual(metric.edges.f1, 1.0)
            self.assertEqual(metric.edges.edge_type_accuracy, 1.0)
            self.assertTrue(metric.edges.exact_untyped_presence_set_match)
            self.assertTrue(metric.edges.exact_typed_edge_set_match)
            self.assertEqual(before, (raw, example.target, example.metadata))
            records.append(metric)

        metadata = {
            item.physical_family_id: item
            for item in self.examples.values()
        }
        aggregate = aggregate_prediction_metrics(
            tuple(reversed(records)), metadata_by_family=metadata
        )
        repeated = aggregate_prediction_metrics(
            records, metadata_by_family=dict(reversed(tuple(metadata.items())))
        )
        self.assertIsInstance(aggregate, AggregateEvaluation)
        self.assertEqual(aggregate, repeated)
        self.assertEqual(aggregate.overall.attempted_example_count, 6)
        self.assertEqual(
            aggregate.overall.validity.raw_completion_rate, 1.0
        )
        self.assertEqual(
            aggregate.overall.validity.raw_integrity_valid_rate, 1.0
        )
        self.assertEqual(
            aggregate.overall.validity.reconstruction_target_valid_rate, 1.0
        )
        self.assertEqual(
            aggregate.overall.validity.controlled_domain_valid_rate, 1.0
        )
        self.assertEqual(
            aggregate.overall.nodes.node_type_token_accuracy, 1.0
        )
        self.assertEqual(
            aggregate.overall.nodes.exact_node_type_sequence_rate, 1.0
        )
        self.assertTrue(all(
            item.accuracy == 1.0
            for item in aggregate.overall.nodes.categorical_fields
        ))
        self.assertEqual(
            aggregate.overall.nodes.exact_nine_attribute_match_rate, 1.0
        )
        self.assertEqual(
            aggregate.overall.nodes.exact_complete_ten_field_match_rate,
            1.0,
        )
        self.assertEqual(
            aggregate.overall.nodes.geometry_mask_cell_accuracy, 1.0
        )
        self.assertEqual(
            aggregate.overall.nodes.exact_geometry_mask_match_rate, 1.0
        )
        self.assertEqual(
            aggregate.overall.geometry.finite_applicable_geometry_mae, 0.0
        )
        self.assertEqual(
            aggregate.overall.geometry.finite_applicable_geometry_rmse, 0.0
        )
        self.assertEqual(
            aggregate.overall.geometry.complete_finite_example_rate, 1.0
        )
        self.assertEqual(
            aggregate.overall.operations.predicted_zero_operation_rate, 0.0
        )
        self.assertEqual(
            aggregate.overall.operations.predicted_target_excess_rate, 0.0
        )
        self.assertEqual(
            aggregate.overall.operations
            .predicted_configured_limit_excess_rate,
            0.0,
        )
        self.assertEqual(
            aggregate.overall.operations
            .exact_operation_type_sequence_rate,
            1.0,
        )
        self.assertEqual(
            aggregate.overall.pointers.overall_pointer_accuracy, 1.0
        )
        self.assertEqual(
            aggregate.overall.pointers.conditional_pointer_accuracy, 1.0
        )
        self.assertEqual(
            aggregate.overall.pointers
            .exact_operation_sequence_pointer_match_rate,
            1.0,
        )
        self.assertEqual(aggregate.overall.edges.micro_precision, 1.0)
        self.assertEqual(aggregate.overall.edges.micro_recall, 1.0)
        self.assertEqual(aggregate.overall.edges.micro_f1, 1.0)
        self.assertEqual(
            aggregate.overall.edges.micro_edge_type_accuracy, 1.0
        )
        self.assertEqual(aggregate.overall.edges.macro_precision, 1.0)
        self.assertEqual(aggregate.overall.edges.macro_recall, 1.0)
        self.assertEqual(aggregate.overall.edges.macro_f1, 1.0)
        self.assertEqual(
            aggregate.overall.edges.macro_edge_type_accuracy, 1.0
        )
        self.assertEqual(
            aggregate.overall.edges.exact_untyped_presence_set_match_rate,
            1.0,
        )
        self.assertEqual(
            aggregate.overall.edges.exact_typed_edge_set_match_rate, 1.0
        )
        self.assertEqual(
            tuple(key for key, _ in aggregate.by_operation_template),
            ("E", "EE", "ER", "R", "RE", "RR"),
        )
        self.assertEqual(
            tuple(key for key, _ in aggregate.by_primitive_family),
            ("capsule_line_arc", "circle", "rectangle_lines"),
        )
        self.assertEqual(
            tuple(
                key for key, _
                in aggregate.by_target_operation_type_sequence
            ),
            (
                "extrude",
                "extrude,extrude",
                "extrude,revolve",
                "revolve",
                "revolve,extrude",
                "revolve,revolve",
            ),
        )

    def test_conversion_is_always_recomputed_with_explicit_limit(self):
        target = self.examples["ER"].target
        missing = evaluate_prediction(
            None, target, family_id="missing", max_operations=2
        )
        self.assertFalse(missing.raw_completion)
        self.assertEqual(missing.primary_failure.code, "missing_raw_prediction")
        self.assertFalse(missing.raw_integrity.valid)
        self.assertFalse(missing.reconstruction_target.valid)
        self.assertFalse(missing.controlled_domain.valid)
        raw = _raw_from_target(target, max_operations=2)
        explicit_limit = evaluate_prediction(
            raw,
            target,
            family_id="explicit_limit",
            max_operations=1,
        )
        self.assertEqual(
            explicit_limit.primary_failure.code,
            "inconsistent_operation_metadata",
        )
        self.assertEqual(
            explicit_limit.primary_failure.location,
            "operation_count_exceeds_limit",
        )
        self.assertTrue(
            explicit_limit.operations.predicted_exceeds_configured_limit
        )
        self.assertFalse(
            self.evaluate(raw).operations
            .predicted_exceeds_configured_limit
        )
        self.assertNotIn(
            "conversion_result",
            inspect.signature(evaluate_prediction).parameters,
        )
        with self.assertRaises(TypeError):
            evaluate_prediction(
                raw,
                target,
                family_id="stale",
                max_operations=2,
                conversion_result=object(),
            )

    def test_node_categories_rows_and_masks_retain_target_denominators(self):
        raw = _raw_from_target(self.examples["ER"].target)
        target_count = len(raw.raw_nodes)
        wrong_type = _with_node(
            raw, 0, node_type_id=NODE_TYPES.id("sketch")
        )
        metric = self.evaluate(wrong_type)
        self.assertEqual(
            metric.nodes.node_type_correct_positions, target_count - 1
        )
        self.assertFalse(metric.nodes.exact_node_type_sequence)
        self.assertFalse(metric.nodes.exact_complete_ten_field_match)

        for field in range(9):
            attributes = list(raw.raw_nodes[0].categorical_ids)
            attributes[field] = 0 if attributes[field] != 0 else 1
            metric = self.evaluate(
                _with_node(raw, 0, categorical_ids=tuple(attributes))
            )
            self.assertEqual(
                metric.nodes.categorical_fields[field].correct_positions,
                target_count - 1,
            )
            self.assertFalse(metric.nodes.exact_nine_attribute_match)
            self.assertFalse(metric.nodes.exact_complete_ten_field_match)

        missing = replace(raw, raw_nodes=raw.raw_nodes[:-1])
        extra_node = replace(
            raw.raw_nodes[-1], position=len(raw.raw_nodes)
        )
        extra = replace(raw, raw_nodes=raw.raw_nodes + (extra_node,))
        malformed = replace(
            raw, raw_nodes=(object(),) + raw.raw_nodes[1:]
        )
        misaligned = _with_node(raw, 0, position=1)
        unavailable = replace(raw, raw_nodes=object())
        for changed in (
            missing, extra, malformed, misaligned, unavailable
        ):
            metric = self.evaluate(changed)
            self.assertEqual(
                metric.nodes.node_type_target_positions, target_count
            )
            self.assertEqual(
                metric.nodes.geometry_mask_target_cells,
                target_count * GEOMETRY_WIDTH,
            )
            self.assertFalse(metric.nodes.exact_node_type_sequence)
            self.assertFalse(metric.nodes.exact_nine_attribute_match)
            self.assertFalse(metric.nodes.exact_complete_ten_field_match)
        misaligned_metric = self.evaluate(misaligned)
        self.assertEqual(
            misaligned_metric.nodes.node_type_correct_positions,
            target_count - 1,
        )
        self.assertGreater(
            misaligned_metric.geometry
            .unavailable_predicted_applicable_geometry_channel_count,
            0,
        )
        self.assertFalse(misaligned_metric.operations.available)
        self.assertFalse(self.evaluate(unavailable).nodes.node_records_available)

        mask = list(raw.raw_nodes[0].derived_geometry_mask)
        mask[0] = not mask[0]
        metric = self.evaluate(
            _with_node(raw, 0, derived_geometry_mask=tuple(mask))
        )
        self.assertEqual(
            metric.nodes.geometry_mask_matching_cells,
            target_count * GEOMETRY_WIDTH - 1,
        )
        self.assertFalse(metric.nodes.exact_geometry_mask_match)

    def test_node_and_operation_complete_structure_availability(self):
        raw = _raw_from_target(self.examples["ER"].target)
        target_count = len(raw.raw_nodes)
        extra_node = replace(raw.raw_nodes[-1], position=target_count)
        malformed_category = list(raw.raw_nodes[0].categorical_ids)
        malformed_category[0] = True
        malformed_mask = list(raw.raw_nodes[0].derived_geometry_mask)
        malformed_mask[0] = 1
        cases = (
            (replace(raw, raw_nodes=()), False),
            (replace(raw, raw_nodes=raw.raw_nodes[:-1]), False),
            (replace(raw, raw_nodes=raw.raw_nodes + (extra_node,)), False),
            (replace(raw, raw_nodes=(object(),) + raw.raw_nodes[1:]), False),
            (_with_node(raw, 0, position=1), False),
            (_with_node(raw, 0, node_type_id=True), False),
            (
            _with_node(
                raw, 0, categorical_ids=tuple(malformed_category)
            ), True),
            (
            _with_node(
                raw,
                0,
                categorical_ids=raw.raw_nodes[0].categorical_ids[:-1],
            ), True),
            (
            _with_node(
                raw, 0, derived_geometry_mask=tuple(malformed_mask)
            ), True),
            (
            _with_node(
                raw,
                0,
                derived_geometry_mask=raw.raw_nodes[0]
                .derived_geometry_mask[:-1],
            ), True),
        )
        for changed, operations_available in cases:
            metric = self.evaluate(changed)
            self.assertFalse(metric.nodes.node_records_available)
            self.assertFalse(metric.nodes.exact_node_type_sequence)
            self.assertFalse(metric.nodes.exact_nine_attribute_match)
            self.assertFalse(metric.nodes.exact_geometry_mask_match)
            self.assertEqual(
                metric.operations.available, operations_available
            )
            if not operations_available:
                self.assertIsNone(
                    metric.operations.predicted_operation_count
                )
            self.assertEqual(
                metric.nodes.node_type_target_positions, target_count
            )

        out_of_vocabulary = _with_node(
            raw, 0, node_type_id=len(NODE_TYPES.tokens) + 10
        )
        metric = self.evaluate(out_of_vocabulary)
        self.assertTrue(metric.nodes.node_records_available)
        self.assertFalse(metric.nodes.exact_node_type_sequence)
        self.assertFalse(metric.operations.available)
        categories = list(raw.raw_nodes[0].categorical_ids)
        categories[0] = 10_000
        metric = self.evaluate(
            _with_node(raw, 0, categorical_ids=tuple(categories))
        )
        self.assertTrue(metric.nodes.node_records_available)
        self.assertFalse(metric.nodes.exact_nine_attribute_match)

        zero_nodes = tuple(
            replace(node, node_type_id=NODE_TYPES.id("profile"))
            for node in raw.raw_nodes
        )
        zero = self.evaluate(replace(raw, raw_nodes=zero_nodes))
        self.assertTrue(zero.operations.available)
        self.assertEqual(zero.operations.predicted_operation_count, 0)
        self.assertTrue(zero.operations.predicted_zero_operations)

    def test_geometry_physical_scales_masks_and_out_of_range(self):
        raw = _raw_from_target(self.examples["ER"].target)
        target = self.examples["ER"].target
        applicable = [
            (node, channel)
            for node, mask in enumerate(target.geometry_mask)
            for channel, value in enumerate(mask)
            if value
        ]
        scale_classes = {}
        for node, channel in applicable:
            scale_classes.setdefault(
                GEOMETRY_CHANNEL_SCALES[channel], (node, channel)
            )
        self.assertEqual(set(scale_classes), {1.0, 4.0, 360.0})
        for scale, (node, channel) in scale_classes.items():
            predicted = target.geometry[node][channel] + 0.25
            changed = _with_geometry(raw, node, {channel: predicted})
            metric = self.evaluate(changed)
            self.assertAlmostEqual(
                metric.geometry.finite_applicable_geometry_absolute_error_sum,
                0.25 * scale,
            )
            self.assertAlmostEqual(
                metric.geometry.finite_applicable_geometry_squared_error_sum,
                (0.25 * scale) ** 2,
            )

        node, channel = applicable[0]
        out_of_range = _with_geometry(raw, node, {channel: 2.0})
        metric = self.evaluate(out_of_range)
        self.assertFalse(metric.controlled_domain.valid)
        self.assertEqual(
            metric.geometry.finite_predicted_applicable_geometry_channel_count,
            metric.geometry.target_applicable_geometry_channel_count,
        )
        self.assertGreater(
            metric.geometry.finite_applicable_geometry_absolute_error_sum,
            0.0,
        )

        mask = list(raw.raw_nodes[node].derived_geometry_mask)
        mask[channel] = False
        metric = self.evaluate(
            _with_node(raw, node, derived_geometry_mask=tuple(mask))
        )
        self.assertEqual(
            metric.geometry.finite_predicted_applicable_geometry_channel_count,
            metric.geometry.target_applicable_geometry_channel_count,
        )

    def test_geometry_nonfinite_unavailable_and_zero_denominators(self):
        raw = _raw_from_target(self.examples["ER"].target)
        target = self.examples["ER"].target
        node, channels = next(
            (index, tuple(
                channel for channel, value in enumerate(mask) if value
            ))
            for index, mask in enumerate(target.geometry_mask)
            if any(mask)
        )
        channel = channels[0]
        for value in (math.nan, math.inf, -math.inf):
            metric = self.evaluate(
                _with_geometry(raw, node, {channel: value})
            )
            self.assertEqual(
                metric.geometry.nonfinite_predicted_applicable_geometry_channel_count,
                1,
            )
            self.assertFalse(metric.geometry.complete_finite_geometry)
            self.assertTrue(math.isfinite(
                metric.geometry.finite_applicable_geometry_absolute_error_sum
            ))

        unavailable_cases = [
            _with_geometry(raw, node, {channel: value})
            for value in ("0.0", 0, True, object())
        ]
        unavailable_cases.extend((
            _with_node(
                raw,
                node,
                normalized_geometry=raw.raw_nodes[node].normalized_geometry[
                    :channel
                ],
            ),
            replace(raw, raw_nodes=raw.raw_nodes[:node]),
        ))
        for changed in unavailable_cases:
            metric = self.evaluate(changed)
            self.assertGreater(
                metric.geometry.unavailable_predicted_applicable_geometry_channel_count,
                0,
            )
            self.assertFalse(metric.geometry.complete_finite_geometry)

        all_unavailable = replace(
            raw,
            raw_nodes=tuple(
                replace(
                    node_record,
                    normalized_geometry=("invalid",) * GEOMETRY_WIDTH,
                )
                for node_record in raw.raw_nodes
            ),
        )
        metric = self.evaluate(all_unavailable)
        self.assertEqual(
            metric.geometry.finite_predicted_applicable_geometry_channel_count,
            0,
        )
        self.assertEqual(
            metric.geometry.finite_applicable_geometry_absolute_error_sum, 0.0
        )
        self.assertEqual(
            metric.geometry.finite_applicable_geometry_squared_error_sum, 0.0
        )
        self.assertIsNone(metric.geometry.finite_applicable_geometry_mae)
        self.assertIsNone(metric.geometry.finite_applicable_geometry_rmse)

        empty_mask = tuple(
            (False,) * GEOMETRY_WIDTH for _ in target.geometry_mask
        )
        no_applicable_target = replace(target, geometry_mask=empty_mask)
        metric = self.evaluate(raw, target=no_applicable_target)
        self.assertEqual(
            metric.geometry.target_applicable_geometry_channel_count, 0
        )
        self.assertTrue(metric.geometry.complete_finite_geometry)
        self.assertIsNone(metric.geometry.finite_applicable_geometry_mae)
        self.assertIsNone(metric.geometry.finite_applicable_geometry_rmse)

    def test_geometry_aggregate_uses_global_channel_sufficient_statistics(self):
        raw = _raw_from_target(self.examples["ER"].target)
        target = self.examples["ER"].target
        applicable = [
            (node, channel)
            for node, mask in enumerate(target.geometry_mask)
            for channel, value in enumerate(mask)
            if value
        ]
        first_node, first_channel = applicable[0]
        first = self.evaluate(
            _with_geometry(
                raw,
                first_node,
                {
                    first_channel:
                    target.geometry[first_node][first_channel] + 0.25
                },
            ),
            family_id="first",
        )
        keep_node, keep_channel = applicable[-1]
        sparse_nodes = []
        for node_index, node in enumerate(raw.raw_nodes):
            values = ["invalid"] * GEOMETRY_WIDTH
            if node_index == keep_node:
                values[keep_channel] = (
                    target.geometry[keep_node][keep_channel] + 0.5
                )
            sparse_nodes.append(
                replace(node, normalized_geometry=tuple(values))
            )
        second = self.evaluate(
            replace(raw, raw_nodes=tuple(sparse_nodes)),
            family_id="second",
        )
        metadata = {
            "first": self.metadata_for("first"),
            "second": self.metadata_for("second"),
        }
        aggregate = aggregate_prediction_metrics(
            (first, second), metadata_by_family=metadata
        ).overall.geometry
        expected_mae = (
            first.geometry.finite_applicable_geometry_absolute_error_sum
            + second.geometry.finite_applicable_geometry_absolute_error_sum
        ) / (
            first.geometry.finite_predicted_applicable_geometry_channel_count
            + second.geometry.finite_predicted_applicable_geometry_channel_count
        )
        mean_of_examples = (
            first.geometry.finite_applicable_geometry_mae
            + second.geometry.finite_applicable_geometry_mae
        ) / 2
        self.assertAlmostEqual(
            aggregate.finite_applicable_geometry_mae, expected_mae
        )
        self.assertNotAlmostEqual(
            aggregate.finite_applicable_geometry_mae, mean_of_examples
        )
        self.assertEqual(aggregate.complete_finite_example_rate, 0.5)

    def test_geometry_overflow_is_unavailable_without_nonfinite_raw_count(self):
        raw = _raw_from_target(self.examples["ER"].target)
        target = self.examples["ER"].target
        applicable = [
            (node, channel)
            for node, mask in enumerate(target.geometry_mask)
            for channel, value in enumerate(mask)
            if value
        ]
        scale_node, scale_channel = next(
            item for item in applicable
            if GEOMETRY_CHANNEL_SCALES[item[1]] == 360.0
        )
        unit_channels = tuple(
            item for item in applicable
            if GEOMETRY_CHANNEL_SCALES[item[1]] == 1.0
        )
        square_node, square_channel = unit_channels[0]

        scaling = self.evaluate(
            _with_geometry(
                raw, scale_node, {scale_channel: 1.0e308}
            )
        )
        squaring = self.evaluate(
            _with_geometry(
                raw, square_node, {square_channel: 1.0e200}
            )
        )
        target_rows = list(target.geometry)
        target_row = list(target_rows[square_node])
        target_row[square_channel] = -1.0e308
        target_rows[square_node] = tuple(target_row)
        subtraction_target = replace(
            target, geometry=tuple(target_rows)
        )
        subtraction = self.evaluate(
            _with_geometry(
                raw, square_node, {square_channel: 1.0e308}
            ),
            target=subtraction_target,
        )
        for metric in (scaling, squaring, subtraction):
            self.assertEqual(
                metric.geometry
                .nonfinite_predicted_applicable_geometry_channel_count,
                0,
            )
            self.assertEqual(
                metric.geometry
                .unavailable_predicted_applicable_geometry_channel_count,
                1,
            )
            self.assertFalse(metric.geometry.complete_finite_geometry)
            self._assert_floats_finite(metric)

    def test_geometry_checked_per_example_and_aggregate_accumulation(self):
        raw = _raw_from_target(self.examples["ER"].target)
        target = self.examples["ER"].target
        unit_channels = tuple(
            (node, channel)
            for node, mask in enumerate(target.geometry_mask)
            for channel, value in enumerate(mask)
            if value and GEOMETRY_CHANNEL_SCALES[channel] == 1.0
        )
        self.assertGreaterEqual(len(unit_channels), 2)
        accumulated = raw
        for node, channel in unit_channels[:2]:
            accumulated = _with_geometry(
                accumulated, node, {channel: 1.0e154}
            )
        metric = self.evaluate(accumulated)
        self.assertEqual(
            metric.geometry
            .unavailable_predicted_applicable_geometry_channel_count,
            1,
        )
        self.assertFalse(metric.geometry.complete_finite_geometry)
        self._assert_floats_finite(metric)

        node, channel = unit_channels[0]
        first = self.evaluate(
            _with_geometry(raw, node, {channel: 1.0e154}),
            family_id="overflow_first",
        )
        second = self.evaluate(
            _with_geometry(raw, node, {channel: 1.0e154}),
            family_id="overflow_second",
        )
        aggregate = aggregate_prediction_metrics(
            (first, second),
            metadata_by_family={
                "overflow_first": self.metadata_for("overflow_first"),
                "overflow_second": self.metadata_for("overflow_second"),
            },
        ).overall.geometry
        self.assertFalse(aggregate.absolute_error_aggregation_overflow)
        self.assertTrue(aggregate.squared_error_aggregation_overflow)
        self.assertIsNotNone(
            aggregate.finite_applicable_geometry_absolute_error_sum
        )
        self.assertIsNone(
            aggregate.finite_applicable_geometry_squared_error_sum
        )
        self.assertIsNotNone(aggregate.finite_applicable_geometry_mae)
        self.assertIsNone(aggregate.finite_applicable_geometry_rmse)
        self._assert_floats_finite(aggregate)

    def test_operation_templates_counts_types_and_unavailability(self):
        for template in ("E", "R", "EE", "ER", "RE", "RR"):
            metric = self.evaluate(template=template)
            self.assertTrue(metric.operations.available)
            self.assertEqual(
                metric.operations.predicted_operation_count, len(template)
            )
            self.assertEqual(
                metric.operations.predicted_operation_type_sequence,
                tuple(
                    "extrude" if item == "E" else "revolve"
                    for item in template
                ),
            )
            self.assertTrue(metric.operations.exact_operation_type_sequence)

        raw = _raw_from_target(self.examples["ER"].target)
        zero_nodes = tuple(
            replace(
                node,
                node_type_id=(
                    NODE_TYPES.id("profile")
                    if NODE_TYPES.tokens[node.node_type_id]
                    in ("extrude", "revolve")
                    else node.node_type_id
                ),
            )
            for node in raw.raw_nodes
        )
        zero = self.evaluate(replace(raw, raw_nodes=zero_nodes))
        self.assertTrue(zero.operations.predicted_zero_operations)
        self.assertEqual(zero.operations.predicted_operation_count, 0)

        changed_nodes = list(raw.raw_nodes)
        for index in (0, 1, 2):
            changed_nodes[index] = replace(
                changed_nodes[index],
                node_type_id=NODE_TYPES.id("extrude"),
            )
        excess = self.evaluate(replace(raw, raw_nodes=tuple(changed_nodes)))
        self.assertTrue(excess.operations.predicted_exceeds_target)
        self.assertTrue(
            excess.operations.predicted_exceeds_configured_limit
        )

        operation = raw.predicted_operation_node_indices[0]
        original = NODE_TYPES.tokens[raw.raw_nodes[operation].node_type_id]
        wrong = "revolve" if original == "extrude" else "extrude"
        wrong_type = self.evaluate(
            _with_node(raw, operation, node_type_id=NODE_TYPES.id(wrong))
        )
        self.assertFalse(
            wrong_type.operations.exact_operation_type_sequence
        )

        unavailable = self.evaluate(replace(raw, raw_nodes=object()))
        self.assertFalse(unavailable.operations.available)
        self.assertIsNone(unavailable.operations.predicted_operation_count)
        self.assertIsNone(unavailable.operations.predicted_zero_operations)

        no_target_operations = self.evaluate(
            raw, target=_target_without_operations(
                self.examples["ER"].target
            )
        )
        self.assertEqual(
            no_target_operations.operations.target_operation_count, 0
        )
        self.assertTrue(
            no_target_operations.operations.predicted_exceeds_target
        )
        self.assertFalse(
            no_target_operations.operations.exact_operation_type_sequence
        )

    def test_pointer_overall_conditional_extra_missing_and_empty_target(self):
        raw = _raw_from_target(self.examples["ER"].target)
        missing = self.evaluate(
            replace(raw, raw_operation_pointers=raw.raw_operation_pointers[:1])
        )
        self.assertEqual(missing.pointers.target_operation_slots, 2)
        self.assertEqual(missing.pointers.correct_target_operation_slots, 1)
        self.assertEqual(missing.pointers.comparable_pointer_queries, 1)
        self.assertEqual(missing.pointers.overall_pointer_accuracy, 0.5)
        self.assertEqual(missing.pointers.conditional_pointer_accuracy, 1.0)
        self.assertEqual(missing.pointers.missing_target_slot_predictions, 1)
        self.assertFalse(
            missing.pointers.exact_operation_sequence_pointer_match
        )

        incorrect = self.evaluate(
            _with_pointer(raw, 0, selected_node_index=0)
        )
        self.assertEqual(incorrect.pointers.overall_pointer_accuracy, 0.5)
        self.assertEqual(
            incorrect.pointers.conditional_pointer_accuracy, 0.5
        )

        extra_record = RawDecodedPointer(
            len(raw.raw_operation_pointers), 0, 1.0
        )
        extra = self.evaluate(replace(
            raw,
            raw_operation_pointers=raw.raw_operation_pointers
            + (extra_record,),
        ))
        self.assertEqual(extra.pointers.extra_predicted_pointer_records, 1)
        self.assertFalse(extra.pointers.exact_operation_sequence_pointer_match)

        unavailable = self.evaluate(
            replace(raw, raw_operation_pointers=object())
        )
        self.assertFalse(unavailable.pointers.available)
        self.assertEqual(
            unavailable.pointers.missing_target_slot_predictions, 2
        )
        self.assertIsNone(
            unavailable.pointers.conditional_pointer_accuracy
        )

        no_target = self.evaluate(
            replace(raw, raw_operation_pointers=()),
            target=_target_without_operations(self.examples["ER"].target),
        )
        self.assertIsNone(no_target.pointers.overall_pointer_accuracy)
        self.assertIsNone(no_target.pointers.conditional_pointer_accuracy)
        self.assertTrue(
            no_target.pointers.exact_operation_sequence_pointer_match
        )

    def test_operation_and_pointer_aggregate_rates_use_all_attempts(self):
        raw = _raw_from_target(self.examples["ER"].target)
        zero_nodes = tuple(
            replace(
                node,
                node_type_id=(
                    NODE_TYPES.id("profile")
                    if NODE_TYPES.tokens[node.node_type_id]
                    in ("extrude", "revolve")
                    else node.node_type_id
                ),
            )
            for node in raw.raw_nodes
        )
        zero_missing = replace(
            raw,
            raw_nodes=zero_nodes,
            raw_operation_pointers=raw.raw_operation_pointers[:1],
        )
        excess_nodes = list(raw.raw_nodes)
        for index in (0, 1, 2):
            excess_nodes[index] = replace(
                excess_nodes[index],
                node_type_id=NODE_TYPES.id("extrude"),
            )
        excess_incorrect = replace(
            _with_pointer(raw, 0, selected_node_index=0),
            raw_nodes=tuple(excess_nodes),
        )
        unavailable = replace(
            raw,
            raw_nodes=object(),
            raw_operation_pointers=object(),
        )
        records = (
            self.evaluate(zero_missing, family_id="zero"),
            self.evaluate(excess_incorrect, family_id="excess"),
            self.evaluate(unavailable, family_id="unavailable"),
        )
        metadata = {
            record.family_id: self.metadata_for(record.family_id)
            for record in records
        }
        aggregate = aggregate_prediction_metrics(
            records, metadata_by_family=metadata
        ).overall
        self.assertEqual(aggregate.operations.available_record_count, 2)
        self.assertEqual(aggregate.operations.unavailable_record_count, 1)
        self.assertEqual(
            aggregate.operations.predicted_operation_count_defined_record_count,
            2,
        )
        self.assertEqual(
            aggregate.operations.predicted_zero_operation_rate, 1 / 3
        )
        self.assertEqual(
            aggregate.operations.predicted_target_excess_rate, 1 / 3
        )
        self.assertEqual(
            aggregate.operations.predicted_configured_limit_excess_rate,
            1 / 3,
        )
        self.assertEqual(aggregate.pointers.available_record_count, 2)
        self.assertEqual(aggregate.pointers.unavailable_record_count, 1)
        self.assertEqual(aggregate.pointers.target_operation_slots, 6)
        self.assertEqual(
            aggregate.pointers.correct_target_operation_slots, 2
        )
        self.assertEqual(
            aggregate.pointers.overall_pointer_accuracy, 1 / 3
        )
        self.assertEqual(
            aggregate.pointers.comparable_pointer_queries, 3
        )
        self.assertEqual(
            aggregate.pointers.conditional_pointer_accuracy, 2 / 3
        )

    def test_edge_set_metrics_types_errors_and_corner_conventions(self):
        raw = _raw_from_target(self.examples["ER"].target)
        target = self.examples["ER"].target
        present = next(
            index for index, edge in enumerate(raw.raw_edges) if edge.present
        )
        absent = next(
            index for index, edge in enumerate(raw.raw_edges)
            if not edge.present
        )
        wrong_type_id = next(
            value for value in range(2, len(EDGE_TYPES.tokens))
            if value != raw.raw_edges[present].edge_type_id
        )
        wrong_type = self.evaluate(
            _with_edge(raw, present, edge_type_id=wrong_type_id)
        )
        self.assertTrue(wrong_type.edges.exact_untyped_presence_set_match)
        self.assertFalse(wrong_type.edges.exact_typed_edge_set_match)
        self.assertLess(wrong_type.edges.edge_type_accuracy, 1.0)

        false_positive = self.evaluate(_with_edge(
            raw,
            absent,
            presence_logit=1.0,
            present=True,
            edge_type_id=EDGE_TYPES.id("placed_on"),
        ))
        self.assertEqual(false_positive.edges.false_positive_pairs, 1)

        false_negative = self.evaluate(_with_edge(
            raw, present, presence_logit=-1.0, present=False
        ))
        self.assertEqual(false_negative.edges.false_negative_pairs, 1)

        empty_target = _target_without_edges(target)
        both_empty = self.evaluate(
            _all_edges_absent(raw), target=empty_target
        )
        self.assertEqual(
            (both_empty.edges.precision, both_empty.edges.recall,
             both_empty.edges.f1),
            (1.0, 1.0, 1.0),
        )
        self.assertIsNone(both_empty.edges.edge_type_accuracy)

        predicted_empty = self.evaluate(_all_edges_absent(raw))
        self.assertEqual(
            (predicted_empty.edges.precision, predicted_empty.edges.recall,
             predicted_empty.edges.f1),
            (0.0, 0.0, 0.0),
        )

        target_empty = self.evaluate(raw, target=empty_target)
        self.assertEqual(
            (target_empty.edges.precision, target_empty.edges.recall,
             target_empty.edges.f1),
            (0.0, 1.0, 0.0),
        )

        unavailable = self.evaluate(
            replace(raw, raw_edges=raw.raw_edges[:-1])
        )
        self.assertFalse(unavailable.edges.available)
        self.assertIsNone(unavailable.edges.predicted_present_pair_count)
        self.assertIsNone(unavailable.edges.precision)
        self.assertFalse(unavailable.edges.exact_untyped_presence_set_match)

        reordered = self.evaluate(
            _with_edge(raw, 0, source_index=1)
        )
        self.assertFalse(reordered.edges.available)
        inconsistent_presence = self.evaluate(
            _with_edge(raw, 0, presence_logit=1.0, present=False)
        )
        self.assertFalse(inconsistent_presence.edges.available)
        nonfinite_logit = self.evaluate(
            _with_edge(raw, 0, presence_logit=math.nan)
        )
        self.assertFalse(nonfinite_logit.raw_integrity.valid)
        self.assertTrue(nonfinite_logit.edges.available)
        self.assertEqual(
            nonfinite_logit.edges.exact_typed_edge_set_match,
            self.evaluate(raw).edges.exact_typed_edge_set_match,
        )

    def test_edge_availability_is_target_sized_and_node_independent(self):
        raw = _raw_from_target(self.examples["ER"].target)
        baseline = self.evaluate(raw).edges
        node_mutations = (
            _without_field(raw, "raw_nodes"),
            replace(raw, raw_nodes=None),
            replace(raw, raw_nodes=object()),
            replace(raw, raw_nodes=()),
            replace(raw, raw_nodes=(object(),) + raw.raw_nodes[1:]),
            _with_node(raw, 0, position=1),
            replace(raw, raw_nodes=raw.raw_nodes[:-1]),
        )
        for changed in node_mutations:
            self.assertEqual(self.evaluate(changed).edges, baseline)

        nonempty_target_empty_records = self.evaluate(
            replace(raw, raw_nodes=(), raw_edges=())
        )
        self.assertFalse(nonempty_target_empty_records.edges.available)
        empty_target = replace(
            self.examples["ER"].target,
            node_type_ids=(),
            categorical_attributes=(),
            edge_index=((), ()),
            edge_type_ids=(),
            boolean_mode_targets=(),
            operation_sequence=(),
            geometry=(),
            geometry_mask=(),
        )
        empty_raw = replace(
            raw,
            node_count=0,
            raw_nodes=(),
            raw_edges=(),
            predicted_operation_node_indices=(),
            predicted_operation_count=0,
            raw_operation_pointers=(),
        )
        self.assertTrue(
            self.evaluate(empty_raw, target=empty_target).edges.available
        )
        target_count = len(self.examples["ER"].target.node_type_ids)
        self.assertEqual(len(raw.raw_edges), target_count ** 2)
        wrong_node_length = replace(raw, raw_nodes=raw.raw_nodes[:-1])
        self.assertTrue(self.evaluate(wrong_node_length).edges.available)

        pointer_and_geometry = replace(
            _with_geometry(raw, 0, {0: "invalid"}),
            raw_operation_pointers=object(),
        )
        self.assertEqual(self.evaluate(pointer_and_geometry).edges, baseline)

    def test_edge_micro_macro_and_defined_counts_use_sufficient_statistics(self):
        raw = _raw_from_target(self.examples["ER"].target)
        first = self.evaluate(
            raw, family_id="perfect"
        )
        single_raw = _raw_from_target(self.examples["E"].target)
        second = self.evaluate(
            _all_edges_absent(single_raw),
            template="E",
            family_id="empty",
        )
        third = self.evaluate(
            replace(raw, raw_edges=raw.raw_edges[:-1]),
            family_id="unavailable",
        )
        metadata = {
            "perfect": self.metadata_for("perfect"),
            "empty": self.metadata_for("empty", "E"),
            "unavailable": self.metadata_for("unavailable"),
        }
        aggregate = aggregate_prediction_metrics(
            (first, second, third), metadata_by_family=metadata
        ).overall.edges
        self.assertEqual(aggregate.available_record_count, 2)
        self.assertEqual(aggregate.unavailable_record_count, 1)
        self.assertEqual(aggregate.macro_precision_defined_record_count, 2)
        self.assertEqual(aggregate.macro_recall_defined_record_count, 2)
        self.assertEqual(aggregate.macro_f1_defined_record_count, 2)
        self.assertNotEqual(aggregate.micro_recall, aggregate.macro_recall)
        self.assertEqual(
            aggregate.exact_typed_edge_set_match_rate, 1 / 3
        )
        self.assertEqual(
            aggregate.micro_true_positive_pairs,
            first.edges.true_positive_pairs
            + second.edges.true_positive_pairs,
        )
        self.assertEqual(
            aggregate.micro_false_negative_pairs,
            first.edges.false_negative_pairs
            + second.edges.false_negative_pairs,
        )

        all_unavailable = aggregate_prediction_metrics(
            (third,),
            metadata_by_family={
                "unavailable": self.metadata_for("unavailable")
            },
        ).overall.edges
        self.assertEqual(all_unavailable.available_record_count, 0)
        self.assertIsNone(all_unavailable.micro_precision)
        self.assertIsNone(all_unavailable.micro_recall)
        self.assertIsNone(all_unavailable.micro_f1)
        self.assertIsNone(all_unavailable.micro_edge_type_accuracy)
        self.assertIsNone(all_unavailable.macro_precision)
        self.assertEqual(
            all_unavailable.macro_precision_defined_record_count, 0
        )

    def test_validity_rates_failure_order_and_any_failure_deduplication(self):
        raw = _raw_from_target(self.examples["ER"].target)
        missing = evaluate_prediction(
            None,
            self.examples["ER"].target,
            family_id="missing",
            max_operations=2,
        )
        values = list(raw.raw_nodes[0].normalized_geometry)
        values[0] = 0
        values[1] = 0
        raw_invalid = self.evaluate(
            _with_node(raw, 0, normalized_geometry=tuple(values)),
            family_id="raw_invalid",
        )
        controlled_invalid = self.evaluate(
            _with_geometry(raw, 0, {3: 0.0}),
            family_id="controlled_invalid",
        )
        metadata = {
            key: self.metadata_for(key)
            for key in ("missing", "raw_invalid", "controlled_invalid")
        }
        aggregate = aggregate_prediction_metrics(
            (missing, raw_invalid, controlled_invalid),
            metadata_by_family=metadata,
        ).overall
        validity = aggregate.validity
        self.assertEqual(validity.attempted_example_count, 3)
        self.assertEqual(validity.raw_completion_rate, 2 / 3)
        self.assertEqual(validity.raw_integrity_valid_rate, 1 / 3)
        self.assertEqual(
            validity.reconstruction_target_valid_rate, 1 / 3
        )
        self.assertEqual(validity.controlled_domain_valid_rate, 0.0)
        primary = {item.code: item.count
                   for item in validity.primary_failure_counts}
        any_failure = {item.code: item.count
                       for item in validity.any_failure_counts}
        self.assertEqual(primary["missing_raw_prediction"], 1)
        self.assertEqual(primary["invalid_raw_field_type"], 1)
        self.assertEqual(
            primary["invalid_reference_plane_geometry"], 1
        )
        self.assertEqual(any_failure["invalid_raw_field_type"], 1)
        self.assertEqual(
            tuple(item.code for item in validity.primary_failure_counts),
            tuple(item.code for item in validity.any_failure_counts),
        )

    def test_aggregation_rejects_family_and_metadata_mismatches(self):
        record = self.evaluate(family_id="family")
        metadata = self.examples["ER"].metadata
        authoritative = self.metadata_for("family")
        accepted = aggregate_prediction_metrics(
            (record,), metadata_by_family={"family": authoritative}
        )
        self.assertEqual(accepted.overall.attempted_example_count, 1)
        with self.assertRaisesRegex(ValueError, "duplicate metric"):
            aggregate_prediction_metrics(
                (record, record),
                metadata_by_family={"family": authoritative},
            )
        with self.assertRaisesRegex(ValueError, "missing metadata"):
            aggregate_prediction_metrics((record,), metadata_by_family={})
        with self.assertRaisesRegex(ValueError, "unknown"):
            aggregate_prediction_metrics(
                (record,),
                metadata_by_family={
                    "family": metadata,
                    "unknown": metadata,
                },
            )
        inconsistent = replace(
            self.examples["ER"], physical_family_id="different"
        )
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            aggregate_prediction_metrics(
                (record,), metadata_by_family={"family": inconsistent}
            )
        with self.assertRaises(TypeError):
            aggregate_prediction_metrics(
                (record,), metadata_by_family={"family": object()}
            )
        with self.assertRaisesRegex(TypeError, "PhysicalExample"):
            aggregate_prediction_metrics(
                (record,), metadata_by_family={"family": metadata}
            )
        inconsistent_depth = replace(metadata, history_depth=99)
        with self.assertRaisesRegex(ValueError, "history depth"):
            aggregate_prediction_metrics(
                (record,),
                metadata_by_family={
                    "family": replace(
                        authoritative, metadata=inconsistent_depth
                    )
                },
            )
        unsupported_template = replace(
            authoritative,
            metadata=replace(metadata, operation_template="EX"),
        )
        with self.assertRaisesRegex(ValueError, "operation template"):
            aggregate_prediction_metrics(
                (record,),
                metadata_by_family={"family": unsupported_template},
            )
        unsupported_primitive = replace(
            authoritative,
            metadata=replace(metadata, primitive_family="triangle"),
        )
        with self.assertRaisesRegex(ValueError, "primitive family"):
            aggregate_prediction_metrics(
                (record,),
                metadata_by_family={"family": unsupported_primitive},
            )
        operation_nodes = list(authoritative.nodes)
        operation_id = authoritative.operation_sequence[0]
        operation_index = next(
            index for index, node in enumerate(operation_nodes)
            if node.node_id == operation_id
        )
        operation_nodes[operation_index] = replace(
            operation_nodes[operation_index], operation_type="loft"
        )
        unsupported_operation = replace(
            authoritative, nodes=tuple(operation_nodes)
        )
        with self.assertRaisesRegex(ValueError, "operation type sequence"):
            aggregate_prediction_metrics(
                (record,),
                metadata_by_family={"family": unsupported_operation},
            )
        swapped = self.examples["E"]
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            aggregate_prediction_metrics(
                (record,), metadata_by_family={"family": swapped}
            )

    def test_malformed_mutations_never_raise_and_metrics_remain_finite(self):
        raw = _raw_from_target(self.examples["ER"].target)
        mutations = (
            None,
            object(),
            replace(raw, raw_nodes=object()),
            replace(raw, raw_nodes=(object(),) + raw.raw_nodes[1:]),
            replace(raw, raw_edges=object()),
            replace(raw, raw_edges=raw.raw_edges[:-1]),
            replace(raw, raw_operation_pointers=object()),
            replace(
                raw,
                raw_operation_pointers=(object(),)
                + raw.raw_operation_pointers[1:],
            ),
            _without_field(raw, "node_count"),
            _without_field(raw, "raw_nodes"),
            _with_geometry(raw, 0, {0: math.nan, 1: math.inf}),
            _with_edge(raw, 0, presence_logit=math.nan),
            _with_pointer(raw, 0, selected_logit=math.inf),
        )
        records = []
        for index, mutation in enumerate(mutations):
            first = evaluate_prediction(
                mutation,
                self.examples["ER"].target,
                family_id="mutation_{:02d}".format(index),
                max_operations=2,
            )
            second = evaluate_prediction(
                mutation,
                self.examples["ER"].target,
                family_id="mutation_{:02d}".format(index),
                max_operations=2,
            )
            self.assertEqual(first, second)
            self._assert_floats_finite(first)
            records.append(first)
        metadata = {
            record.family_id: self.metadata_for(record.family_id)
            for record in records
        }
        inputs_before = (tuple(records), dict(metadata))
        aggregate = aggregate_prediction_metrics(
            records, metadata_by_family=metadata
        )
        repeated = aggregate_prediction_metrics(
            tuple(reversed(records)),
            metadata_by_family=dict(reversed(tuple(metadata.items()))),
        )
        self.assertEqual(aggregate, repeated)
        self.assertEqual(inputs_before, (tuple(records), metadata))
        self._assert_floats_finite(aggregate)
        overall = aggregate.overall
        attempted = len(records)
        self.assertEqual(overall.attempted_example_count, attempted)
        self.assertEqual(
            overall.nodes.available_record_count,
            sum(record.nodes.node_records_available for record in records),
        )
        self.assertEqual(
            overall.nodes.unavailable_record_count,
            attempted - overall.nodes.available_record_count,
        )
        for name in (
            "node_type_correct_positions",
            "node_type_target_positions",
            "geometry_mask_matching_cells",
            "geometry_mask_target_cells",
        ):
            self.assertEqual(
                getattr(overall.nodes, name),
                sum(getattr(record.nodes, name) for record in records),
            )
        for name, source in (
            ("exact_node_type_sequence_count", "exact_node_type_sequence"),
            (
                "exact_nine_attribute_match_count",
                "exact_nine_attribute_match",
            ),
            (
                "exact_complete_ten_field_match_count",
                "exact_complete_ten_field_match",
            ),
            (
                "exact_geometry_mask_match_count",
                "exact_geometry_mask_match",
            ),
        ):
            self.assertEqual(
                getattr(overall.nodes, name),
                sum(getattr(record.nodes, source) for record in records),
            )
        for index, field in enumerate(overall.nodes.categorical_fields):
            self.assertEqual(
                field.correct_positions,
                sum(
                    record.nodes.categorical_fields[index].correct_positions
                    for record in records
                ),
            )
            self.assertEqual(
                field.target_positions,
                sum(
                    record.nodes.categorical_fields[index].target_positions
                    for record in records
                ),
            )
        for name in (
            "target_applicable_geometry_channel_count",
            "finite_predicted_applicable_geometry_channel_count",
            "nonfinite_predicted_applicable_geometry_channel_count",
            "unavailable_predicted_applicable_geometry_channel_count",
        ):
            self.assertEqual(
                getattr(overall.geometry, name),
                sum(getattr(record.geometry, name) for record in records),
            )
        self.assertEqual(
            overall.geometry.complete_finite_example_count,
            sum(
                record.geometry.complete_finite_geometry
                for record in records
            ),
        )
        available_operations = tuple(
            record.operations
            for record in records if record.operations.available
        )
        self.assertEqual(
            overall.operations.available_record_count,
            len(available_operations),
        )
        self.assertEqual(
            overall.operations.unavailable_record_count,
            attempted - len(available_operations),
        )
        self.assertEqual(
            overall.operations.target_operation_count_sum,
            sum(record.operations.target_operation_count for record in records),
        )
        self.assertEqual(
            overall.operations.predicted_operation_count_sum,
            sum(
                record.predicted_operation_count
                for record in available_operations
            ),
        )
        self.assertEqual(
            overall.operations.predicted_operation_count_defined_record_count,
            len(available_operations),
        )
        for name, source in (
            (
                "predicted_zero_operation_count",
                "predicted_zero_operations",
            ),
            (
                "predicted_target_excess_count",
                "predicted_exceeds_target",
            ),
            (
                "predicted_configured_limit_excess_count",
                "predicted_exceeds_configured_limit",
            ),
        ):
            self.assertEqual(
                getattr(overall.operations, name),
                sum(getattr(record, source) for record in available_operations),
            )
        self.assertEqual(
            overall.operations.exact_operation_type_sequence_count,
            sum(
                record.operations.exact_operation_type_sequence
                for record in records
            ),
        )
        self.assertEqual(
            overall.pointers.available_record_count,
            sum(record.pointers.available for record in records),
        )
        self.assertEqual(
            overall.pointers.unavailable_record_count,
            attempted - overall.pointers.available_record_count,
        )
        for name in (
            "target_operation_slots",
            "correct_target_operation_slots",
            "comparable_pointer_queries",
            "correct_comparable_queries",
            "missing_target_slot_predictions",
            "extra_predicted_pointer_records",
        ):
            self.assertEqual(
                getattr(overall.pointers, name),
                sum(getattr(record.pointers, name) for record in records),
            )
        self.assertEqual(
            overall.pointers
            .exact_operation_sequence_pointer_match_count,
            sum(
                record.pointers.exact_operation_sequence_pointer_match
                for record in records
            ),
        )
        available_edges = tuple(
            record.edges for record in records if record.edges.available
        )
        self.assertEqual(
            overall.edges.available_record_count, len(available_edges)
        )
        self.assertEqual(
            overall.edges.unavailable_record_count,
            attempted - len(available_edges),
        )
        for name, source in (
            ("micro_true_positive_pairs", "true_positive_pairs"),
            ("micro_false_positive_pairs", "false_positive_pairs"),
            ("micro_false_negative_pairs", "false_negative_pairs"),
            (
                "micro_correctly_typed_shared_pairs",
                "correctly_typed_shared_pair_count",
            ),
            ("micro_shared_present_pairs", "shared_present_pair_count"),
        ):
            self.assertEqual(
                getattr(overall.edges, name),
                sum(getattr(record, source) for record in available_edges),
            )
        for name, source in (
            ("macro_precision_defined_record_count", "precision"),
            ("macro_recall_defined_record_count", "recall"),
            ("macro_f1_defined_record_count", "f1"),
            (
                "macro_edge_type_accuracy_defined_record_count",
                "edge_type_accuracy",
            ),
        ):
            self.assertEqual(
                getattr(overall.edges, name),
                sum(
                    getattr(record, source) is not None
                    for record in available_edges
                ),
            )
        self.assertEqual(
            overall.edges.exact_untyped_presence_set_match_count,
            sum(
                record.edges.exact_untyped_presence_set_match
                for record in records
            ),
        )
        self.assertEqual(
            overall.edges.exact_typed_edge_set_match_count,
            sum(
                record.edges.exact_typed_edge_set_match
                for record in records
            ),
        )
        validity = overall.validity
        self.assertEqual(validity.attempted_example_count, attempted)
        for name, source in (
            ("raw_completion_count", "raw_completion"),
            ("raw_integrity_valid_count", "raw_integrity"),
            (
                "reconstruction_target_valid_count",
                "reconstruction_target",
            ),
            ("controlled_domain_valid_count", "controlled_domain"),
        ):
            expected = sum(
                getattr(record, source)
                if source == "raw_completion"
                else getattr(record, source).valid
                for record in records
            )
            self.assertEqual(getattr(validity, name), expected)
        for aggregate_failure, code in (
            (validity.primary_failure_counts, "primary"),
            (validity.any_failure_counts, "any"),
        ):
            for item in aggregate_failure:
                if code == "primary":
                    expected = sum(
                        record.primary_failure is not None
                        and record.primary_failure.code == item.code
                        for record in records
                    )
                else:
                    expected = sum(
                        item.code in {
                            failure.code
                            for failure in (
                                (
                                    (record.primary_failure,)
                                    if record.primary_failure else ()
                                )
                                + record.secondary_failures
                            )
                        }
                        for record in records
                    )
                self.assertEqual(item.count, expected)
        self.assertEqual(
            aggregate.overall.geometry.target_applicable_geometry_channel_count,
            sum(
                record.geometry.target_applicable_geometry_channel_count
                for record in records
            ),
        )
        self.assertEqual(
            aggregate.overall.edges.micro_true_positive_pairs,
            sum(
                record.edges.true_positive_pairs
                for record in records if record.edges.available
            ),
        )

    def _assert_floats_finite(self, value):
        if isinstance(value, float):
            self.assertTrue(math.isfinite(value))
            return
        if isinstance(value, tuple):
            for item in value:
                self._assert_floats_finite(item)
            return
        if hasattr(value, "__dataclass_fields__"):
            for field in fields(value):
                self._assert_floats_finite(getattr(value, field.name))


if __name__ == "__main__":
    unittest.main()
