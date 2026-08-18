"""Real-PyTorch tests for the extended grid-ordinal horizon diagnostic."""

from __future__ import annotations

import math
import unittest
from unittest.mock import patch

try:
    import torch
except ImportError:
    torch = None

from prototype.graph_encoder import grid_ordinal_horizon as diagnostic
from prototype.graph_encoder import grid_magnitude as gm

if torch is not None:
    from prototype.graph_encoder import losses
else:
    losses = None


REASON = "grid ordinal horizon runtime tests require PyTorch"


@unittest.skipIf(torch is None, REASON)
class GridOrdinalHorizonRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.records = diagnostic.run_condition("extrude", 2)

    def test_real_target_uses_frozen_ids_channel_and_masks(self):
        target = diagnostic._generated_target("revolve", 4)
        self.assertTrue(torch.equal(
            target["node_type_ids"],
            torch.full((8, 1), gm.OPERATION_NODE_TYPE_IDS[1], dtype=torch.long),
        ))
        self.assertTrue(bool(target["node_mask"].all().item()))
        self.assertEqual(int(target["geometry_mask"].sum().item()), 8)
        self.assertTrue(bool(target["geometry_mask"][..., 38].all().item()))
        self.assertTrue(bool((target["geometry"][..., 38] == 1.0).all().item()))

    def test_exactly_2000_updates_and_optimizer_steps(self):
        self.assertEqual(len(self.records), 2001)
        self.assertEqual([row["step"] for row in self.records], list(range(2001)))
        self.assertEqual(self.records[0]["raw_gap_values"], [0.0, 0.0, 0.0])
        self.assertEqual(self.records[0]["adamw_state"]["step"], 0)
        self.assertEqual(self.records[-1]["adamw_state"]["step"], 2000)
        self.assertFalse(self.records[-1]["gradient_update_follows_record"])

    def test_gradients_before_after_and_clipping_are_complete(self):
        for row in (self.records[0], self.records[200], self.records[-1]):
            gradients = row["gradients"]
            self.assertEqual(len(gradients["projection"]["before_clipping"]), 32)
            self.assertEqual(len(gradients["raw_gaps"]["before_clipping"]), 3)
            self.assertEqual(len(gradients["first_bias"]["before_clipping"]), 1)
            self.assertTrue(math.isfinite(gradients["global_before_clipping"]))
            self.assertLessEqual(gradients["global_after_clipping"], 1.00001)

    def test_adamw_moments_are_recorded_for_active_parameters(self):
        state = self.records[-1]["adamw_state"]
        self.assertEqual(len(state["projection"]["exp_avg"]), 32)
        self.assertEqual(len(state["projection"]["exp_avg_sq"]), 32)
        self.assertEqual(len(state["raw_gaps"]["exp_avg"]), 3)
        self.assertTrue(math.isfinite(state["first_bias"]["exp_avg"]))

    def test_fixed_state_never_receives_gradient(self):
        states = diagnostic._fixed_states()
        self.assertFalse(states.requires_grad)
        self.assertIsNone(states.grad)

    def test_real_head_and_loss_are_invoked(self):
        with patch.object(diagnostic, "OPTIMIZER_UPDATES", 1), patch.object(
            diagnostic, "build_grid_magnitude_head", wraps=gm.build_grid_magnitude_head
        ) as head_call, patch.object(
            diagnostic, "grid_magnitude_terms", wraps=losses.grid_magnitude_terms
        ) as loss_call:
            rows = diagnostic.run_condition("revolve", 3)
        self.assertEqual(head_call.call_count, 1)
        self.assertEqual(loss_call.call_count, 2)
        self.assertEqual(len(rows), 2)

    def test_observed_outcome_is_not_a_validity_gate(self):
        summary = diagnostic.summarize_condition(self.records)
        self.assertIn(summary["final_decoded_class"], range(5))
        self.assertIsInstance(summary["final_class_correct"], bool)


if __name__ == "__main__":
    unittest.main()
