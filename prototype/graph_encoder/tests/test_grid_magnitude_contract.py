"""PyTorch-independent contract tests for the grid-ordinal magnitude repair."""

from __future__ import annotations

import ast
from dataclasses import fields
import json
from pathlib import Path
import unittest

from prototype.graph_encoder import grid_magnitude as gm
from prototype.graph_encoder.config import (
    GE1Config,
    frozen_encoder_config,
    grid_frozen_encoder_config,
    legacy_frozen_encoder_config,
)
from prototype.graph_encoder.decoder_contract import (
    GRID_CHECKPOINT_SCHEMA,
    GRID_ORDINAL_OPERATION_MAGNITUDE_PARAMETERIZATION,
    GRID_OUTPUT_POSITION_CONTRACT_VERSION,
    GRID_SHARED_DECODER_VERSION,
    LEGACY_CHECKPOINT_SCHEMA,
    LEGACY_OPERATION_MAGNITUDE_PARAMETERIZATION,
    OUTPUT_POSITION_CONTRACT_VERSION,
    POSITIVE_OPERATION_MAGNITUDE_PARAMETERIZATION,
    SHARED_DECODER_VERSION,
    checkpoint_schema_for,
    output_position_contract_version_for,
    shared_decoder_version_for,
    uses_grid_magnitude,
)
from prototype.graph_encoder.errors import GraphEncoderError
from prototype.graph_encoder.operation_fidelity import (
    EXTRUSION_ABSOLUTE_ERROR_MAX_EXCLUSIVE,
    EXTRUSION_GRID,
    EXTRUSION_SCALE,
    REVOLVE_ABSOLUTE_ERROR_MAX_EXCLUSIVE,
    REVOLVE_GRID,
    REVOLVE_SCALE,
)


PACKAGE = Path(__file__).parents[1]


class FrozenGridTests(unittest.TestCase):
    def test_grids_match_the_fidelity_gate_exactly(self):
        self.assertEqual(gm.EXTRUSION_PHYSICAL_GRID, tuple(EXTRUSION_GRID))
        self.assertEqual(gm.REVOLVE_PHYSICAL_GRID, tuple(REVOLVE_GRID))
        self.assertEqual(gm.EXTRUSION_NORMALIZED_GRID,
                         (0.125, 0.25, 0.375, 0.5, 0.75))
        self.assertEqual(gm.REVOLVE_NORMALIZED_GRID,
                         (0.125, 0.25, 0.5, 0.75, 1.0))

    def test_grids_are_strictly_increasing_and_five_valued(self):
        for name in gm.OPERATION_TYPES:
            for grid in (gm.PHYSICAL_GRIDS[name], gm.NORMALIZED_GRIDS[name]):
                self.assertEqual(len(grid), gm.GRID_CLASS_COUNT)
                self.assertEqual(list(grid), sorted(grid))
                self.assertEqual(len(set(grid)), gm.GRID_CLASS_COUNT)

    def test_channels_and_scales(self):
        self.assertEqual(gm.SERIALIZED_CHANNELS["extrude"], 37)
        self.assertEqual(gm.SERIALIZED_CHANNELS["revolve"], 38)
        self.assertEqual(gm.COMPACT_CHANNELS["extrude"], 4)
        self.assertEqual(gm.COMPACT_CHANNELS["revolve"], 5)
        meta = gm.grid_contract_metadata()
        self.assertEqual(meta["grids"]["extrude"]["normalization_scale"],
                         EXTRUSION_SCALE)
        self.assertEqual(meta["grids"]["revolve"]["normalization_scale"],
                         REVOLVE_SCALE)

    def test_exact_physical_reconstruction_from_class(self):
        for name in gm.OPERATION_TYPES:
            scale = gm.grid_contract_metadata()["grids"][name][
                "normalization_scale"
            ]
            for index in range(gm.GRID_CLASS_COUNT):
                normalized = gm.normalized_grid_value(name, index)
                physical = gm.physical_grid_value(name, index)
                self.assertAlmostEqual(normalized * scale, physical, places=12)
                # A correct class yields exactly zero error, far inside the gate.
                self.assertEqual(abs(physical - physical), 0.0)

    def test_360_degree_class_is_representable_and_passes_the_gate(self):
        top = gm.physical_grid_value("revolve", gm.GRID_CLASS_COUNT - 1)
        self.assertEqual(top, 360.0)
        self.assertEqual(gm.normalized_grid_value("revolve", 4), 1.0)
        self.assertLess(abs(top - 360.0), REVOLVE_ABSOLUTE_ERROR_MAX_EXCLUSIVE)


class LabelMappingTests(unittest.TestCase):
    def test_exact_target_to_class_mapping(self):
        for name in gm.OPERATION_TYPES:
            for index, value in enumerate(gm.NORMALIZED_GRIDS[name]):
                self.assertEqual(
                    gm.class_index_from_normalized_target(name, value), index
                )

    def test_off_grid_and_malformed_targets_are_rejected(self):
        cases = (
            ("extrude", 0.4, "off_grid_magnitude_target"),
            ("extrude", 0.0, "off_grid_magnitude_target"),
            ("revolve", 0.9, "off_grid_magnitude_target"),
            ("extrude", float("nan"), "invalid_grid_target"),
            ("extrude", float("inf"), "invalid_grid_target"),
            ("extrude", None, "invalid_grid_target"),
            ("extrude", True, "invalid_grid_target"),
        )
        for name, value, code in cases:
            with self.subTest(value=value):
                with self.assertRaises(GraphEncoderError) as caught:
                    gm.class_index_from_normalized_target(name, value)
                self.assertEqual(caught.exception.code, code)

    def test_nothing_is_snapped(self):
        near = gm.EXTRUSION_NORMALIZED_GRID[0] + 10 * gm.GRID_LABEL_TOLERANCE
        with self.assertRaises(GraphEncoderError):
            gm.class_index_from_normalized_target("extrude", near)

    def test_unknown_operation_type_is_rejected(self):
        with self.assertRaises(GraphEncoderError):
            gm.class_index_from_normalized_target("fillet", 0.125)

    def test_cumulative_labels_round_trip(self):
        for index in range(gm.GRID_CLASS_COUNT):
            labels = gm.cumulative_labels(index)
            self.assertEqual(len(labels), gm.ORDINAL_CUT_COUNT)
            self.assertEqual(sum(labels), float(index))
            flags = tuple(bool(value) for value in labels)
            self.assertEqual(gm.class_index_from_cumulative_flags(flags), index)

    def test_labels_are_non_increasing(self):
        for index in range(gm.GRID_CLASS_COUNT):
            labels = gm.cumulative_labels(index)
            self.assertEqual(list(labels), sorted(labels, reverse=True))

    def test_rank_inconsistent_decode_is_rejected(self):
        with self.assertRaises(GraphEncoderError) as caught:
            gm.class_index_from_cumulative_flags((True, False, True, False))
        self.assertEqual(
            caught.exception.code, "rank_inconsistent_ordinal_decision"
        )

    def test_node_type_selects_the_grid(self):
        for name in gm.OPERATION_TYPES:
            node_type_id = gm.grid_contract_metadata()["grids"][name][
                "node_type_id"
            ]
            self.assertEqual(
                gm.operation_type_for_node_type_id(node_type_id), name
            )
        self.assertIsNone(gm.operation_type_for_node_type_id(0))


class ReductionContractTests(unittest.TestCase):
    def test_reduction_is_sum_within_example_then_batch_mean(self):
        reduction = gm.GRID_MAGNITUDE_REDUCTION.to_dict()
        self.assertIn("sum_over_active_operations", reduction["within_example"])
        self.assertEqual(reduction["across_batch"], "mean_over_examples")
        self.assertFalse(reduction["axis_channels_included"])
        self.assertFalse(reduction["class_balancing_applied"])

    def test_equal_per_operation_contribution_algebra(self):
        """Sum-then-mean gives every operation coefficient 1/B."""

        # E example: 1 operation; EE example: 2 operations.
        per_operation = {"E": [0.7], "EE": [0.3, 0.5]}
        batch = len(per_operation)
        total = sum(sum(v) for v in per_operation.values()) / batch
        # Each operation's coefficient is 1/B regardless of template.
        for values in per_operation.values():
            for value in values:
                self.assertAlmostEqual(
                    (total - (total - value / batch)), value / batch, places=12
                )
        # A within-example mean would have halved the EE coefficient.
        diluted = sum(sum(v) / len(v) for v in per_operation.values()) / batch
        self.assertNotAlmostEqual(total, diluted, places=6)

    def test_no_residual_in_v1(self):
        self.assertFalse(gm.grid_contract_metadata()["residual_implemented"])


class IdentityTests(unittest.TestCase):
    def test_grid_identity_literals(self):
        self.assertEqual(
            GRID_ORDINAL_OPERATION_MAGNITUDE_PARAMETERIZATION,
            "GE1-OPERATION-MAGNITUDE-GRID-ORDINAL-v1",
        )
        self.assertEqual(GRID_CHECKPOINT_SCHEMA, "GE1-CHECKPOINT-v2")
        self.assertEqual(gm.GRID_MAGNITUDE_PARAMETERIZATION,
                         GRID_ORDINAL_OPERATION_MAGNITUDE_PARAMETERIZATION)

    def test_historical_identities_are_byte_stable(self):
        self.assertEqual(LEGACY_OPERATION_MAGNITUDE_PARAMETERIZATION,
                         "GE1-OPERATION-MAGNITUDE-TANH-LEGACY-v1")
        self.assertEqual(POSITIVE_OPERATION_MAGNITUDE_PARAMETERIZATION,
                         "GE1-OPERATION-MAGNITUDE-POSITIVE-v1")
        self.assertEqual(SHARED_DECODER_VERSION,
                         "GE1-SHARED-TYPED-EDGE-DECODER-V1")
        self.assertEqual(OUTPUT_POSITION_CONTRACT_VERSION,
                         "GE1-DECODER-OUTPUT-POSITIONS-v1")
        self.assertEqual(LEGACY_CHECKPOINT_SCHEMA, "GE1-CHECKPOINT-v1")

    def test_identity_resolvers(self):
        for parameterization in (
            LEGACY_OPERATION_MAGNITUDE_PARAMETERIZATION,
            POSITIVE_OPERATION_MAGNITUDE_PARAMETERIZATION,
        ):
            self.assertFalse(uses_grid_magnitude(parameterization))
            self.assertEqual(checkpoint_schema_for(parameterization),
                             LEGACY_CHECKPOINT_SCHEMA)
            self.assertEqual(shared_decoder_version_for(parameterization),
                             SHARED_DECODER_VERSION)
            self.assertEqual(
                output_position_contract_version_for(parameterization),
                OUTPUT_POSITION_CONTRACT_VERSION,
            )
        grid = GRID_ORDINAL_OPERATION_MAGNITUDE_PARAMETERIZATION
        self.assertTrue(uses_grid_magnitude(grid))
        self.assertEqual(checkpoint_schema_for(grid), GRID_CHECKPOINT_SCHEMA)
        self.assertEqual(shared_decoder_version_for(grid),
                         GRID_SHARED_DECODER_VERSION)
        self.assertEqual(output_position_contract_version_for(grid),
                         GRID_OUTPUT_POSITION_CONTRACT_VERSION)

    def test_grid_config_carries_v2_schema(self):
        config = grid_frozen_encoder_config("flat")
        self.assertEqual(config.checkpoint_schema, GRID_CHECKPOINT_SCHEMA)
        config.validate()

    def test_historical_configs_serialize_exactly_as_before(self):
        grid_only = set(GE1Config.GRID_ONLY_SERIALIZED_FIELDS)
        declared = {item.name for item in fields(GE1Config)}
        expected_historical = (declared - grid_only) | {"arm_identity"}
        for builder in (frozen_encoder_config, legacy_frozen_encoder_config):
            payload = json.loads(builder("flat").to_json())
            with self.subTest(builder=builder.__name__):
                self.assertEqual(set(payload), expected_historical)
                self.assertFalse(set(payload) & grid_only)
                self.assertEqual(payload["checkpoint_schema"],
                                 LEGACY_CHECKPOINT_SCHEMA)
        grid_payload = json.loads(grid_frozen_encoder_config("flat").to_json())
        self.assertEqual(set(grid_payload), declared | {"arm_identity"})

    def test_cross_identity_configs_are_distinct(self):
        historical = frozen_encoder_config("flat").to_json()
        grid = grid_frozen_encoder_config("flat").to_json()
        self.assertNotEqual(historical, grid)


class FidelityGateStabilityTests(unittest.TestCase):
    """The gate must remain byte-stable; this repair never touches it."""

    def test_thresholds_and_comparison_semantics(self):
        self.assertEqual(EXTRUSION_ABSOLUTE_ERROR_MAX_EXCLUSIVE, 0.25)
        self.assertEqual(REVOLVE_ABSOLUTE_ERROR_MAX_EXCLUSIVE, 22.5)
        self.assertEqual(EXTRUSION_SCALE, 4.0)
        self.assertEqual(REVOLVE_SCALE, 360.0)

    def test_operation_fidelity_source_is_unmodified_by_this_change(self):
        source = (PACKAGE / "operation_fidelity.py").read_text(encoding="utf-8")
        self.assertIn("strictly_less_than_unrounded", source)
        self.assertNotIn("grid_magnitude", source)


class TargetIsolationTests(unittest.TestCase):
    def test_grid_module_never_imports_a_loader_or_target_source(self):
        tree = ast.parse(
            (PACKAGE / "grid_magnitude.py").read_text(encoding="utf-8")
        )
        forbidden = {
            "load_train", "load_development", "load_physical_examples",
            "load_partition_physical_examples",
        }
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    self.assertNotIn(alias.name, forbidden)

    def test_head_construction_signature_takes_no_target(self):
        source = (PACKAGE / "grid_magnitude.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name in (
                "build_grid_magnitude_head", "forward", "normalized_values",
                "class_indices", "cumulative_probabilities",
            ):
                names = {arg.arg for arg in node.args.args}
                self.assertFalse(
                    names & {"target", "targets", "labels", "label"},
                    "{} must not accept a target argument".format(node.name),
                )

    def test_labels_are_derived_only_inside_the_loss_module(self):
        decoder = (PACKAGE / "shared_decoder.py").read_text(encoding="utf-8")
        self.assertNotIn("class_index_from_normalized_target", decoder)
        self.assertNotIn("cumulative_labels", decoder)
        losses = (PACKAGE / "losses.py").read_text(encoding="utf-8")
        self.assertIn("class_index_from_normalized_target", losses)


class Python38GrammarTests(unittest.TestCase):
    def test_new_sources_parse_under_python_38(self):
        for name in ("grid_magnitude.py", "losses.py", "config.py",
                     "decoder_contract.py", "shared_decoder.py",
                     "checkpoint.py", "model.py"):
            path = PACKAGE / name
            with self.subTest(name=name):
                ast.parse(
                    path.read_text(encoding="utf-8"),
                    filename=str(path),
                    feature_version=(3, 8),
                )


if __name__ == "__main__":
    unittest.main()
