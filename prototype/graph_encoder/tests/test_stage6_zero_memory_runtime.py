"""Real-PyTorch runtime contracts for the Stage 6 zero-memory intervention."""

from __future__ import annotations

from types import SimpleNamespace
import unittest

try:
    import torch
except ImportError:
    torch = None

from prototype.graph_encoder.autonomous import AutonomousInputBatch
from prototype.graph_encoder.stage6_zero_memory_diagnostic import (
    P_ZERO,
    run_zero_memory_evaluation,
    zero_memory_tensor,
)


@unittest.skipUnless(torch is not None, "PyTorch is unavailable")
class Stage6ZeroMemoryRuntimeTests(unittest.TestCase):
    def test_zeros_like_is_exact_and_preserves_tensor_properties(self):
        memory = torch.tensor(
            [[[1.5, -2.0], [3.25, 4.0]]], dtype=torch.float64
        )
        original = memory.clone()
        zero = zero_memory_tensor(memory)
        self.assertTrue(torch.equal(zero, torch.zeros_like(memory)))
        self.assertEqual(tuple(zero.shape), tuple(memory.shape))
        self.assertEqual(zero.dtype, memory.dtype)
        self.assertEqual(zero.device, memory.device)
        self.assertTrue(torch.equal(memory, original))

    def test_evaluation_changes_only_memory_and_preserves_counts_masks_and_order(self):
        class Model:
            def __init__(self):
                self.training = True
                self.calls = []

            def parameters(self):
                return iter(())

            def eval(self):
                self.training = False

            def train(self, value):
                self.training = value

            def encode(self, encoder_input):
                return SimpleNamespace(memory=encoder_input["memory"])

            def decoder(self, memory, *, node_counts, node_count_source):
                self.calls.append((
                    memory.detach().clone(),
                    node_counts.detach().clone(),
                    node_count_source,
                ))
                value = SimpleNamespace(
                    raw_prediction=("raw",),
                    constrained_prediction=("constrained",),
                    converted_prediction=("converted",),
                )
                return value

        original_memory = torch.tensor([
            [[1.0, 2.0]],
            [[3.0, 4.0]],
        ], dtype=torch.float32)
        original_mask = torch.tensor([[True], [False]], dtype=torch.bool)
        encoder_input = {
            "memory": original_memory.clone(),
            "legal_node_type_masks": original_mask.clone(),
        }
        batch = AutonomousInputBatch(
            ("family-a", "family-b"), encoder_input, (4, 8), "batch-identity"
        )
        model = Model()
        result = run_zero_memory_evaluation(model, (batch,), execution_device="cpu")
        condition = result["condition"]
        self.assertEqual(condition.condition, P_ZERO)
        self.assertTrue(condition.available)
        self.assertEqual(len(condition.predictions), 2)
        self.assertEqual(len(model.calls), 2)
        self.assertTrue(model.training)
        self.assertTrue(torch.equal(encoder_input["memory"], original_memory))
        self.assertTrue(torch.equal(
            encoder_input["legal_node_type_masks"], original_mask
        ))
        for index, (memory, node_counts, source) in enumerate(model.calls):
            self.assertEqual(torch.count_nonzero(memory).item(), 0)
            self.assertEqual(memory.dtype, original_memory.dtype)
            self.assertEqual(memory.device, original_memory.device)
            self.assertEqual(node_counts.tolist(), [(4, 8)[index]])
            self.assertEqual(source, "target_free_input_node_count")
        for family_id, evidence in result["intervention_evidence"].items():
            self.assertIn(family_id, ("family-a", "family-b"))
            self.assertEqual(evidence["memory_nonzero_count"], 0)
            self.assertTrue(evidence["only_memory_values_replaced"])
            self.assertTrue(evidence["recipient_node_count_preserved"])
            self.assertTrue(evidence["decoder_non_memory_inputs_preserved"])
            self.assertFalse(evidence["encoder_input_mutated"])


if __name__ == "__main__":
    unittest.main()
