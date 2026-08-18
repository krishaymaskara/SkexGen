"""Real-PyTorch tests for the generated-state grid-ordinal trajectory."""

from __future__ import annotations

import math
import unittest
from unittest.mock import patch

try:
    import torch
except ImportError:  # Authoritative execution occurs in the Adroit container.
    torch = None

from prototype.graph_encoder import grid_ordinal_trajectory as diagnostic
from prototype.graph_encoder import grid_magnitude as gm

if torch is not None:
    from prototype.graph_encoder import losses
else:
    losses = None


REASON = "grid ordinal trajectory runtime tests require PyTorch"


@unittest.skipIf(torch is None, REASON)
class GridOrdinalTrajectoryRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.zero_class1 = diagnostic.run_condition("extrude", 0.0, 1)
        cls.gap_class1 = diagnostic.run_condition("extrude", 0.5, 1)

    def test_real_generated_target_uses_frozen_ids_channels_and_masks(self):
        target = diagnostic._generated_target("revolve", 3)
        self.assertTrue(torch.equal(
            target["node_type_ids"],
            torch.full((8, 1), gm.OPERATION_NODE_TYPE_IDS[1], dtype=torch.long),
        ))
        self.assertTrue(bool(target["node_mask"].all().item()))
        self.assertEqual(int(target["geometry_mask"].sum().item()), 8)
        self.assertTrue(bool(target["geometry_mask"][..., 38].all().item()))
        self.assertTrue(bool((target["geometry"][..., 38] == 0.75).all().item()))

    def test_current_default_and_diagnostic_intervention_are_separate(self):
        self.assertEqual(self.zero_class1[0]["raw_gap_values"], [0.0, 0.0, 0.0])
        self.assertEqual(self.gap_class1[0]["raw_gap_values"], [0.5, 0.5, 0.5])
        torch.manual_seed(2026)
        unchanged = gm.build_grid_magnitude_head(32)
        self.assertTrue(bool((unchanged.bias_gaps == 0.0).all().item()))

    def test_exactly_200_updates_produce_201_states(self):
        self.assertEqual(len(self.zero_class1), 201)
        self.assertEqual([row["step"] for row in self.zero_class1], list(range(201)))
        self.assertEqual(self.zero_class1[0]["state_coordinate"], "initialization")
        self.assertEqual(self.zero_class1[-1]["state_coordinate"], "post_optimizer_update")
        self.assertFalse(self.zero_class1[-1]["gradient_update_follows_record"])

    def test_every_required_state_field_is_finite_and_shaped(self):
        for row in self.zero_class1:
            self.assertEqual(len(row["raw_gap_values"]), 3)
            self.assertEqual(len(row["ordered_biases"]), 4)
            self.assertEqual(len(row["logits"]), 4)
            self.assertEqual(len(row["sigmoid_probabilities"]), 4)
            for name in ("loss", "projection_score", "first_bias", "minimum_decision_margin"):
                self.assertTrue(math.isfinite(row[name]))
            self.assertEqual(
                set(row["gradient_norms"]),
                {"projection", "first_bias", "raw_gaps", "global_before_clipping", "clip_norm", "values_are_after_clipping"},
            )
            self.assertTrue(row["engineering_only"])
            self.assertFalse(row["scientific_training"])

    def test_each_condition_is_fresh_and_deterministic(self):
        repeated = diagnostic.run_condition("extrude", 0.0, 1)
        self.assertEqual(self.zero_class1, repeated)

    def test_generated_state_never_receives_gradient(self):
        states = diagnostic._fixed_states()
        self.assertFalse(states.requires_grad)
        self.assertIsNone(states.grad)
        diagnostic.run_condition("revolve", 0.0, 3)
        self.assertFalse(states.requires_grad)
        self.assertIsNone(states.grad)

    def test_real_head_and_loss_are_invoked_without_reimplementation(self):
        with patch.object(
            diagnostic, "build_grid_magnitude_head",
            wraps=gm.build_grid_magnitude_head,
        ) as head_call, patch.object(
            diagnostic, "grid_magnitude_terms",
            wraps=losses.grid_magnitude_terms,
        ) as loss_call:
            rows = diagnostic.run_condition("revolve", 0.5, 3)
        self.assertEqual(head_call.call_count, 1)
        self.assertEqual(loss_call.call_count, 201)
        self.assertEqual(len(rows), 201)

    def test_observed_outcome_never_controls_artifact_validity(self):
        summary = diagnostic.summarize_condition(self.gap_class1)
        self.assertIn(summary["final_decoded_class"], range(5))
        self.assertIn(
            summary["diagnostic_gap_0p5_corrected_by_step_200"],
            (True, False),
        )


if __name__ == "__main__":
    unittest.main()
