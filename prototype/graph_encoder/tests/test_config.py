"""Frozen GE1 configuration contract tests."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import json
import unittest

from prototype.graph_encoder.config import (
    AUTHORIZED_FALLBACK_SEEDS,
    CAPACITY_ADJUSTMENT_FIELDS,
    CHECKPOINT_SCHEMA,
    EXPERIMENT_IDENTITY,
    FLAT_ARM,
    GE1Config,
    GE1TrainingConfig,
    GRAPH_ARM,
    MODEL_FAMILY,
    PLANNED_SEEDS,
    PROTOCOL_IDENTITY,
    SHARED_DECODER,
)
from prototype.graph_encoder.errors import GraphEncoderError


class GE1ConfigTests(unittest.TestCase):
    def test_exact_identities_and_derived_arm(self):
        flat = GE1Config("flat")
        graph = GE1Config("typed_graph")
        self.assertEqual(flat.model_family, MODEL_FAMILY)
        self.assertEqual(flat.shared_decoder, SHARED_DECODER)
        self.assertEqual(flat.checkpoint_schema, CHECKPOINT_SCHEMA)
        self.assertEqual(flat.protocol_identity, PROTOCOL_IDENTITY)
        self.assertEqual(flat.experiment_identity, EXPERIMENT_IDENTITY)
        self.assertEqual(flat.arm_identity, FLAT_ARM)
        self.assertEqual(graph.arm_identity, GRAPH_ARM)

    def test_immutable_deterministic_finite_serialization(self):
        config = GE1Config("typed_graph")
        with self.assertRaises(FrozenInstanceError):
            config.seed = 2027
        self.assertEqual(config.to_json(), GE1Config("typed_graph").to_json())
        self.assertEqual(json.loads(config.to_json()), config.to_dict())
        self.assertNotIn("NaN", config.to_json())
        self.assertNotIn("Infinity", config.to_json())

    def test_only_capacity_adjustment_fields_are_variable(self):
        self.assertEqual(
            CAPACITY_ADJUSTMENT_FIELDS,
            ("relation_basis_count", "encoder_feedforward_width"),
        )
        adjusted = replace(
            GE1Config("typed_graph"),
            relation_basis_count=1,
            encoder_feedforward_width=128,
        )
        adjusted.validate()

    def test_invalid_encoder_seed_bottleneck_and_identity(self):
        cases = (
            (replace(GE1Config("flat"), encoder="graph"), "invalid_configuration"),
            (replace(GE1Config("flat"), seed=2025), "unauthorized_configuration"),
            (replace(GE1Config("flat"), seed=True), "invalid_configuration"),
            (
                replace(GE1Config("flat"), bottleneck_mode="discrete"),
                "unauthorized_configuration",
            ),
            (
                replace(GE1Config("flat"), model_family="GE1-MODEL-v2"),
                "identity_mismatch",
            ),
            (
                replace(GE1Config("flat"), checkpoint_schema="checkpoint"),
                "identity_mismatch",
            ),
        )
        for config, code in cases:
            with self.subTest(config=config):
                with self.assertRaises(GraphEncoderError) as caught:
                    config.validate()
                self.assertEqual(caught.exception.code, code)

    def test_invalid_dimensions_layers_heads_dropout_and_losses(self):
        cases = (
            replace(GE1Config("flat"), model_dim=0),
            replace(GE1Config("flat"), model_dim=True),
            replace(GE1Config("flat"), attention_heads=3),
            replace(GE1Config("flat"), relational_layers=2),
            replace(GE1Config("flat"), relation_basis_count=0),
            replace(GE1Config("flat"), encoder_feedforward_width=False),
            replace(GE1Config("flat"), dropout=0.5),
            replace(GE1Config("flat"), dropout=float("nan")),
            replace(GE1Config("flat"), dropout=float("inf")),
            replace(GE1Config("flat"), node_type_loss_weight=0.0),
            replace(GE1Config("flat"), graph_edge_loss_weight=True),
            replace(GE1Config("flat"), geometry_loss_weight=float("nan")),
        )
        for config in cases:
            with self.subTest(config=config):
                with self.assertRaises(GraphEncoderError):
                    config.validate()


class GE1TrainingConfigTests(unittest.TestCase):
    def test_frozen_training_policy_and_epoch_50_selection(self):
        config = GE1TrainingConfig()
        config.validate()
        self.assertEqual(config.planned_seeds, (2026, 2027, 2028))
        self.assertEqual(config.authorized_fallback_seeds, (2026, 2027))
        self.assertEqual(config.epochs, 50)
        self.assertEqual(config.batch_size, 8)
        self.assertEqual(config.optimizer, "AdamW")
        self.assertEqual(config.learning_rate, 1e-3)
        self.assertEqual(config.weight_decay, 0.0)
        self.assertEqual(config.gradient_clip_norm, 1.0)
        self.assertEqual(config.checkpoint_selection, "fixed_epoch_50")
        self.assertEqual(config.checkpoint_epoch, 50)
        self.assertEqual(config.plateau_relative_improvement_threshold, 0.01)
        self.assertEqual(config.plateau_moving_best_window_epochs, 5)
        self.assertEqual(config.plateau_start_epoch, 10)
        self.assertEqual(
            config.normalized_prefix_train_ceiling_shortfall_max, 0.05
        )
        self.assertEqual(
            config.complete_validity_train_ceiling_shortfall_max, 0.10
        )
        self.assertEqual(json.loads(config.to_json()), config.to_dict())

    def test_only_exact_timing_fallback_is_accepted(self):
        fallback = replace(
            GE1TrainingConfig(), retained_seeds=AUTHORIZED_FALLBACK_SEEDS
        )
        fallback.validate()
        self.assertTrue(fallback.uses_timing_fallback)
        self.assertFalse(GE1TrainingConfig().uses_timing_fallback)
        with self.assertRaises(GraphEncoderError):
            replace(GE1TrainingConfig(), retained_seeds=(2026,)).validate()
        with self.assertRaises(GraphEncoderError):
            replace(GE1TrainingConfig(), planned_seeds=(2026, 2027)).validate()

    def test_frozen_training_values_reject_changes_and_nonfinite_values(self):
        cases = (
            replace(GE1TrainingConfig(), epochs=49),
            replace(GE1TrainingConfig(), batch_size=True),
            replace(GE1TrainingConfig(), optimizer="SGD"),
            replace(GE1TrainingConfig(), learning_rate=float("nan")),
            replace(GE1TrainingConfig(), weight_decay=float("inf")),
            replace(GE1TrainingConfig(), gradient_clip_norm=0.0),
            replace(GE1TrainingConfig(), checkpoint_selection="train_loss"),
            replace(GE1TrainingConfig(), checkpoint_epoch=49),
            replace(
                GE1TrainingConfig(),
                normalized_prefix_train_ceiling_shortfall_max=0.06,
            ),
        )
        for config in cases:
            with self.subTest(config=config):
                with self.assertRaises(GraphEncoderError):
                    config.validate()

    def test_seed_constants_are_exact(self):
        self.assertEqual(PLANNED_SEEDS, (2026, 2027, 2028))
        self.assertEqual(AUTHORIZED_FALLBACK_SEEDS, (2026, 2027))


if __name__ == "__main__":
    unittest.main()
