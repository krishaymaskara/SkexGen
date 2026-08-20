"""PyTorch-independent contract tests for the ADR-0015 softmax magnitude head.

These cover the new identity only.  The ADR-0013 ordinal contract keeps its own
pinned suite, and this module additionally asserts that nothing in that suite's
subject matter moved.
"""

from __future__ import annotations

import ast
from dataclasses import fields
import json
from pathlib import Path
import re
import unittest

from prototype.graph_encoder import grid_magnitude as gm
from prototype.graph_encoder.config import (
    GE1Config,
    frozen_encoder_config,
    grid_frozen_encoder_config,
    grid_softmax_frozen_encoder_config,
    legacy_frozen_encoder_config,
)
from prototype.graph_encoder.decoder_contract import (
    GRID_CHECKPOINT_SCHEMA,
    GRID_OPERATION_MAGNITUDE_PARAMETERIZATIONS,
    GRID_ORDINAL_OPERATION_MAGNITUDE_PARAMETERIZATION,
    GRID_OUTPUT_POSITION_CONTRACT_VERSION,
    GRID_SHARED_DECODER_VERSION,
    GRID_SOFTMAX_CHECKPOINT_SCHEMA,
    GRID_SOFTMAX_OPERATION_MAGNITUDE_PARAMETERIZATION,
    GRID_SOFTMAX_SHARED_DECODER_VERSION,
    LEGACY_CHECKPOINT_SCHEMA,
    LEGACY_OPERATION_MAGNITUDE_PARAMETERIZATION,
    OPERATION_MAGNITUDE_PARAMETERIZATIONS,
    OUTPUT_POSITION_CONTRACT_VERSION,
    POSITIVE_OPERATION_MAGNITUDE_PARAMETERIZATION,
    SCALAR_OPERATION_MAGNITUDE_PARAMETERIZATIONS,
    SHARED_DECODER_VERSION,
    checkpoint_schema_for,
    output_position_contract_version_for,
    shared_decoder_version_for,
    uses_grid_magnitude,
    uses_grid_softmax_magnitude,
)
from prototype.graph_encoder.errors import GraphEncoderError


PACKAGE = Path(__file__).parents[1]
REPOSITORY = PACKAGE.parents[1]
SOFTMAX = GRID_SOFTMAX_OPERATION_MAGNITUDE_PARAMETERIZATION
ORDINAL = GRID_ORDINAL_OPERATION_MAGNITUDE_PARAMETERIZATION


class SoftmaxIdentityTests(unittest.TestCase):
    def test_identity_literals(self):
        self.assertEqual(SOFTMAX, "GE1-OPERATION-MAGNITUDE-GRID-SOFTMAX-v1")
        self.assertEqual(gm.GRID_SOFTMAX_MAGNITUDE_PARAMETERIZATION, SOFTMAX)
        self.assertEqual(
            gm.GRID_SOFTMAX_MAGNITUDE_CONTRACT_VERSION,
            "GE1-GRID-SOFTMAX-MAGNITUDE-CONTRACT-v1",
        )
        self.assertEqual(
            gm.GRID_SOFTMAX_MAGNITUDE_LOSS_VERSION,
            "GE1-GRID-SOFTMAX-MAGNITUDE-LOSS-v1",
        )
        self.assertEqual(GRID_SOFTMAX_SHARED_DECODER_VERSION,
                         "GE1-SHARED-TYPED-EDGE-DECODER-V3")
        self.assertEqual(GRID_SOFTMAX_CHECKPOINT_SCHEMA, "GE1-CHECKPOINT-v3")

    def test_new_identity_is_appended_without_moving_any_existing_index(self):
        self.assertEqual(
            OPERATION_MAGNITUDE_PARAMETERIZATIONS,
            SCALAR_OPERATION_MAGNITUDE_PARAMETERIZATIONS + (ORDINAL, SOFTMAX),
        )
        self.assertEqual(OPERATION_MAGNITUDE_PARAMETERIZATIONS[:3], (
            LEGACY_OPERATION_MAGNITUDE_PARAMETERIZATION,
            POSITIVE_OPERATION_MAGNITUDE_PARAMETERIZATION,
            ORDINAL,
        ))
        self.assertEqual(
            GRID_OPERATION_MAGNITUDE_PARAMETERIZATIONS, (ORDINAL, SOFTMAX)
        )
        self.assertEqual(gm.GRID_MAGNITUDE_PARAMETERIZATIONS,
                         (ORDINAL, SOFTMAX))

    def test_grid_predicate_is_membership_over_both_grid_identities(self):
        for identity in (ORDINAL, SOFTMAX):
            self.assertTrue(uses_grid_magnitude(identity))
        for identity in SCALAR_OPERATION_MAGNITUDE_PARAMETERIZATIONS:
            self.assertFalse(uses_grid_magnitude(identity))
        self.assertTrue(uses_grid_softmax_magnitude(SOFTMAX))
        others = (ORDINAL,) + SCALAR_OPERATION_MAGNITUDE_PARAMETERIZATIONS
        for identity in others:
            self.assertFalse(uses_grid_softmax_magnitude(identity))

    def test_softmax_gets_its_own_decoder_version_and_checkpoint_schema(self):
        """A shared schema would pass a governed check on an incompatible state."""

        self.assertEqual(shared_decoder_version_for(SOFTMAX),
                         GRID_SOFTMAX_SHARED_DECODER_VERSION)
        self.assertEqual(checkpoint_schema_for(SOFTMAX),
                         GRID_SOFTMAX_CHECKPOINT_SCHEMA)
        self.assertNotEqual(shared_decoder_version_for(SOFTMAX),
                            shared_decoder_version_for(ORDINAL))
        self.assertNotEqual(checkpoint_schema_for(SOFTMAX),
                            checkpoint_schema_for(ORDINAL))
        schemas = {
            checkpoint_schema_for(identity)
            for identity in OPERATION_MAGNITUDE_PARAMETERIZATIONS
        }
        self.assertEqual(schemas, {
            LEGACY_CHECKPOINT_SCHEMA,
            GRID_CHECKPOINT_SCHEMA,
            GRID_SOFTMAX_CHECKPOINT_SCHEMA,
        })

    def test_output_positions_are_shared_because_they_do_not_change(self):
        self.assertEqual(output_position_contract_version_for(SOFTMAX),
                         GRID_OUTPUT_POSITION_CONTRACT_VERSION)
        self.assertEqual(output_position_contract_version_for(ORDINAL),
                         GRID_OUTPUT_POSITION_CONTRACT_VERSION)


class Adr0013ImmutabilityTests(unittest.TestCase):
    """ADR-0013's identity, resolvers, and helpers must not have moved."""

    def test_ordinal_resolvers_are_unchanged(self):
        self.assertEqual(ORDINAL, "GE1-OPERATION-MAGNITUDE-GRID-ORDINAL-v1")
        self.assertEqual(gm.GRID_MAGNITUDE_PARAMETERIZATION, ORDINAL)
        self.assertEqual(shared_decoder_version_for(ORDINAL),
                         GRID_SHARED_DECODER_VERSION)
        self.assertEqual(checkpoint_schema_for(ORDINAL), GRID_CHECKPOINT_SCHEMA)

    def test_historical_scalar_resolvers_are_unchanged(self):
        for identity in SCALAR_OPERATION_MAGNITUDE_PARAMETERIZATIONS:
            with self.subTest(identity=identity):
                self.assertEqual(shared_decoder_version_for(identity),
                                 SHARED_DECODER_VERSION)
                self.assertEqual(output_position_contract_version_for(identity),
                                 OUTPUT_POSITION_CONTRACT_VERSION)
                self.assertEqual(checkpoint_schema_for(identity),
                                 LEGACY_CHECKPOINT_SCHEMA)

    def test_ordinal_helpers_and_cut_count_are_retained(self):
        self.assertEqual(gm.ORDINAL_CUT_COUNT, 4)
        self.assertEqual(gm.cumulative_labels(3), (1.0, 1.0, 1.0, 0.0))
        self.assertEqual(
            gm.class_index_from_cumulative_flags((True, True, False, False)), 2
        )
        self.assertTrue(callable(gm.build_grid_magnitude_head))
        losses = (PACKAGE / "losses.py").read_text(encoding="utf-8")
        self.assertIn("def _cumulative_label_tensor(", losses)
        self.assertIn("binary_cross_entropy_with_logits", losses)

    def test_sufficiency_protocol_still_requires_the_ordinal_identity(self):
        source = (PACKAGE / "grid_magnitude_sufficiency.py").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            "GRID_ORDINAL_OPERATION_MAGNITUDE_PARAMETERIZATION", source
        )
        self.assertNotIn("SOFTMAX", source)

    def test_pinned_ordinal_suite_counts_are_unchanged(self):
        """Three committed exact-commit runners assert these literals."""

        loader = unittest.defaultTestLoader
        expected = (
            ("test_grid_magnitude_contract", 28),
            ("test_grid_magnitude_runtime", 14),
            ("test_grid_magnitude_integration", 18),
        )
        for name, count in expected:
            suite = loader.loadTestsFromName(
                "prototype.graph_encoder.tests." + name
            )
            with self.subTest(suite=name):
                self.assertEqual(suite.countTestCases(), count)


class SoftmaxContractMetadataTests(unittest.TestCase):
    def test_softmax_record_drops_the_cut_count_and_restates_the_decode(self):
        record = gm.grid_contract_metadata(SOFTMAX)
        self.assertEqual(record["version"],
                         gm.GRID_SOFTMAX_MAGNITUDE_CONTRACT_VERSION)
        self.assertEqual(record["parameterization"], SOFTMAX)
        self.assertEqual(record["decode_rule"], "argmax_over_class_logits")
        self.assertEqual(record["rank_consistency"],
                         "none_independent_class_logits")
        self.assertNotIn("ordinal_cut_count", record)
        self.assertEqual(record["class_count"], gm.GRID_CLASS_COUNT)
        self.assertIs(record["class_balancing_applied"], False)
        self.assertIs(record["residual_implemented"], False)
        json.dumps(record, sort_keys=True, allow_nan=False)

    def test_ordinal_record_is_byte_identical_with_and_without_an_argument(self):
        default = gm.grid_contract_metadata()
        explicit = gm.grid_contract_metadata(ORDINAL)
        self.assertEqual(default, explicit)
        self.assertEqual(default["version"], gm.GRID_MAGNITUDE_CONTRACT_VERSION)
        self.assertEqual(default["ordinal_cut_count"], gm.ORDINAL_CUT_COUNT)
        self.assertEqual(default["decode_rule"],
                         "count_of_cumulative_logits_greater_than_zero")
        self.assertEqual(
            default["rank_consistency"],
            "structural_shared_weight_with_decreasing_biases",
        )

    def test_only_the_readout_fields_differ_between_the_two_records(self):
        ordinal = gm.grid_contract_metadata(ORDINAL)
        softmax = gm.grid_contract_metadata(SOFTMAX)
        changed = {
            key for key in set(ordinal) | set(softmax)
            if ordinal.get(key) != softmax.get(key)
        }
        self.assertEqual(changed, {
            "version", "parameterization", "ordinal_cut_count",
            "decode_rule", "rank_consistency",
        })
        self.assertEqual(ordinal["grids"], softmax["grids"])

    def test_unknown_parameterization_is_rejected(self):
        for identity in SCALAR_OPERATION_MAGNITUDE_PARAMETERIZATIONS + ("x",):
            with self.subTest(identity=identity):
                with self.assertRaises(GraphEncoderError):
                    gm.grid_contract_metadata(identity)


class SoftmaxReductionContractTests(unittest.TestCase):
    def test_only_the_per_operation_step_changes(self):
        ordinal = gm.GRID_MAGNITUDE_REDUCTION.to_dict()
        softmax = gm.GRID_SOFTMAX_MAGNITUDE_REDUCTION.to_dict()
        changed = {key for key in ordinal if ordinal[key] != softmax[key]}
        self.assertEqual(changed, {"version", "per_operation"})
        self.assertEqual(
            softmax["per_operation"],
            "cross_entropy_over_five_independent_class_logits",
        )
        self.assertEqual(softmax["version"],
                         gm.GRID_SOFTMAX_MAGNITUDE_LOSS_VERSION)

    def test_equal_per_operation_contribution_survives(self):
        softmax = gm.GRID_SOFTMAX_MAGNITUDE_REDUCTION.to_dict()
        self.assertEqual(softmax["within_example"],
                         "sum_over_active_operations_of_that_type")
        self.assertEqual(softmax["across_batch"], "mean_over_examples")
        self.assertEqual(softmax["per_operation_coefficient"],
                         "1/batch_size_for_every_active_operation")
        self.assertIs(softmax["axis_channels_included"], False)
        self.assertIs(softmax["class_balancing_applied"], False)


class SoftmaxConfigurationTests(unittest.TestCase):
    def test_softmax_config_carries_the_v3_schema(self):
        config = grid_softmax_frozen_encoder_config("flat")
        self.assertEqual(config.operation_magnitude_parameterization, SOFTMAX)
        self.assertEqual(config.checkpoint_schema,
                         GRID_SOFTMAX_CHECKPOINT_SCHEMA)
        config.validate()

    def test_every_default_is_unchanged_so_the_identity_is_opt_in(self):
        declared = GE1Config.__dataclass_fields__[
            "operation_magnitude_parameterization"
        ]
        self.assertEqual(declared.default,
                         POSITIVE_OPERATION_MAGNITUDE_PARAMETERIZATION)
        self.assertEqual(
            frozen_encoder_config("flat").operation_magnitude_parameterization,
            POSITIVE_OPERATION_MAGNITUDE_PARAMETERIZATION,
        )
        self.assertEqual(
            grid_frozen_encoder_config(
                "flat"
            ).operation_magnitude_parameterization,
            ORDINAL,
        )

    def test_grid_only_fields_serialize_for_both_grid_identities(self):
        declared = {item.name for item in fields(GE1Config)}
        grid_only = set(GE1Config.GRID_ONLY_SERIALIZED_FIELDS)
        for builder in (grid_frozen_encoder_config,
                        grid_softmax_frozen_encoder_config):
            payload = json.loads(builder("flat").to_json())
            with self.subTest(builder=builder.__name__):
                self.assertEqual(set(payload), declared | {"arm_identity"})
                self.assertTrue(grid_only <= set(payload))
        for builder in (frozen_encoder_config, legacy_frozen_encoder_config):
            payload = json.loads(builder("flat").to_json())
            with self.subTest(builder=builder.__name__):
                self.assertFalse(set(payload) & grid_only)
                self.assertEqual(payload["checkpoint_schema"],
                                 LEGACY_CHECKPOINT_SCHEMA)

    def test_all_four_identity_configurations_are_distinct(self):
        payloads = [
            frozen_encoder_config("flat").to_json(),
            legacy_frozen_encoder_config("flat").to_json(),
            grid_frozen_encoder_config("flat").to_json(),
            grid_softmax_frozen_encoder_config("flat").to_json(),
        ]
        self.assertEqual(len(set(payloads)), 4)


class SoftmaxIsolationTests(unittest.TestCase):
    def test_the_cut_count_never_leaves_its_two_owning_modules(self):
        """No Stage 6, producer, metric, or audit module may assume a width."""

        owners = {"grid_magnitude.py", "losses.py"}
        for path in sorted(PACKAGE.glob("*.py")):
            if path.name in owners:
                continue
            with self.subTest(module=path.name):
                self.assertNotIn(
                    "ORDINAL_CUT_COUNT", path.read_text(encoding="utf-8")
                )

    def test_no_other_module_hardcodes_a_grid_logit_shape(self):
        pattern = re.compile(r"\[\.\.\., 2, [45]\]")
        allowed = {"grid_magnitude.py", "shared_decoder.py"}
        for path in sorted(PACKAGE.glob("*.py")):
            if path.name in allowed:
                continue
            with self.subTest(module=path.name):
                self.assertIsNone(
                    pattern.search(path.read_text(encoding="utf-8"))
                )

    def test_metrics_module_needs_no_logit_shape_knowledge(self):
        source = (PACKAGE / "grid_magnitude_metrics.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("ORDINAL_CUT_COUNT", source)
        self.assertNotIn("argmax", source)

    def test_labels_are_still_derived_only_inside_the_loss_module(self):
        decoder = (PACKAGE / "shared_decoder.py").read_text(encoding="utf-8")
        self.assertNotIn("class_index_from_normalized_target", decoder)
        grid = (PACKAGE / "grid_magnitude.py").read_text(encoding="utf-8")
        for node in ast.walk(ast.parse(grid)):
            if isinstance(node, ast.FunctionDef):
                self.assertNotIn("label_tensor", node.name)
        losses = (PACKAGE / "losses.py").read_text(encoding="utf-8")
        self.assertIn("def _class_index_label_tensor(", losses)

    def test_softmax_head_interfaces_accept_no_target(self):
        source = (PACKAGE / "grid_magnitude.py").read_text(encoding="utf-8")
        forbidden = {"target", "targets", "label", "labels"}
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.FunctionDef) and node.name in (
                "build_grid_softmax_magnitude_head", "forward", "class_indices",
                "class_probabilities", "normalized_values",
            ):
                names = {item.arg for item in node.args.args}
                with self.subTest(interface=node.name):
                    self.assertFalse(names & forbidden)


class SoftmaxGrammarAndDocumentationTests(unittest.TestCase):
    def test_changed_sources_parse_under_python_38(self):
        names = (
            "grid_magnitude.py", "losses.py", "config.py",
            "decoder_contract.py", "shared_decoder.py", "checkpoint.py",
            "model.py", "grid_magnitude_metrics.py",
            "tests/test_grid_softmax_magnitude_contract.py",
            "tests/test_grid_softmax_magnitude_runtime.py",
        )
        for name in names:
            path = PACKAGE / name
            with self.subTest(name=name):
                ast.parse(
                    path.read_text(encoding="utf-8"),
                    filename=str(path),
                    feature_version=(3, 8),
                )

    def test_governance_records_exist(self):
        decision = (
            REPOSITORY / "docs" / "decisions"
            / "ADR-0015-ge1-grid-softmax-operation-magnitude-classification.md"
        )
        specification = (
            REPOSITORY / "docs" / "specifications"
            / "ge1_grid_softmax_magnitude.md"
        )
        for path in (decision, specification):
            with self.subTest(path=path.name):
                self.assertTrue(path.is_file())
                text = path.read_text(encoding="utf-8")
                self.assertIn(SOFTMAX, text)
                self.assertIn("3353008", text)


if __name__ == "__main__":
    unittest.main()
