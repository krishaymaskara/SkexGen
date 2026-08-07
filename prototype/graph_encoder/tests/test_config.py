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
    FROZEN_FLAT_FEEDFORWARD_WIDTH,
    FROZEN_GRAPH_FEEDFORWARD_WIDTH,
    FROZEN_RELATION_BASIS_COUNT,
    GE1Config,
    GE1TrainingConfig,
    GRAPH_ARM,
    MODEL_FAMILY,
    PLANNED_SEEDS,
    PROTOCOL_IDENTITY,
    SHARED_DECODER,
    frozen_encoder_config,
    frozen_feedforward_width,
)
from prototype.graph_encoder.errors import GraphEncoderError


def _flat():
    """The complete frozen flat configuration; width differs per arm."""

    return frozen_encoder_config("flat")


class GE1ConfigTests(unittest.TestCase):
    def test_exact_identities_and_derived_arm(self):
        flat = _flat()
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

    def test_capacity_adjustment_fields_are_frozen_at_selected_values(self):
        """The Stage 4 match is complete, so both former knobs are now fixed."""

        self.assertEqual(
            CAPACITY_ADJUSTMENT_FIELDS,
            ("relation_basis_count", "encoder_feedforward_width"),
        )
        self.assertEqual(FROZEN_RELATION_BASIS_COUNT, 2)
        self.assertEqual(FROZEN_FLAT_FEEDFORWARD_WIDTH, 192)
        self.assertEqual(FROZEN_GRAPH_FEEDFORWARD_WIDTH, 64)
        self.assertEqual(frozen_feedforward_width("flat"), 192)
        self.assertEqual(frozen_feedforward_width("typed_graph"), 64)
        with self.assertRaises(GraphEncoderError):
            frozen_feedforward_width("graph")

        for encoder, width in (("flat", 192), ("typed_graph", 64)):
            config = frozen_encoder_config(encoder)
            self.assertEqual(config.encoder_feedforward_width, width)
            self.assertEqual(config.relation_basis_count, 2)
            config.validate()

        rejected = (
            replace(_flat(), relation_basis_count=1),
            replace(_flat(), relation_basis_count=3),
            replace(_flat(), encoder_feedforward_width=64),
            replace(_flat(), encoder_feedforward_width=128),
            replace(GE1Config("typed_graph"), relation_basis_count=1),
            replace(GE1Config("typed_graph"), encoder_feedforward_width=192),
        )
        for config in rejected:
            with self.subTest(config=config):
                with self.assertRaises(GraphEncoderError) as caught:
                    config.validate()
                self.assertEqual(
                    caught.exception.code, "unauthorized_configuration"
                )

    def test_bare_flat_config_is_incomplete_without_the_frozen_width(self):
        """`GE1Config('flat')` alone carries the graph width and must fail."""

        with self.assertRaises(GraphEncoderError) as caught:
            GE1Config("flat").validate()
        self.assertEqual(caught.exception.code, "unauthorized_configuration")

    def test_invalid_encoder_seed_bottleneck_and_identity(self):
        cases = (
            (replace(_flat(), encoder="graph"), "invalid_configuration"),
            (replace(_flat(), seed=2025), "unauthorized_configuration"),
            (replace(_flat(), seed=True), "invalid_configuration"),
            (
                replace(_flat(), bottleneck_mode="discrete"),
                "unauthorized_configuration",
            ),
            (
                replace(_flat(), model_family="GE1-MODEL-v2"),
                "identity_mismatch",
            ),
            (
                replace(_flat(), checkpoint_schema="checkpoint"),
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
            replace(_flat(), model_dim=0),
            replace(_flat(), model_dim=True),
            replace(_flat(), attention_heads=3),
            replace(_flat(), relational_layers=2),
            replace(_flat(), relation_basis_count=0),
            replace(_flat(), encoder_feedforward_width=False),
            replace(_flat(), dropout=0.5),
            replace(_flat(), dropout=float("nan")),
            replace(_flat(), dropout=float("inf")),
            replace(_flat(), node_type_loss_weight=0.0),
            replace(_flat(), graph_edge_loss_weight=True),
            replace(_flat(), geometry_loss_weight=float("nan")),
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
