"""Real-PyTorch C7 model lifecycle and autonomous-path tests."""

from __future__ import annotations

from dataclasses import replace
import unittest

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "real PyTorch is required")
class C7RuntimeTests(unittest.TestCase):
    def test_fresh_tiny_and_scaled_pairs_are_matched_and_disjoint(self):
        from prototype.graph_encoder.model import build_matched_ge1_models
        from prototype.graph_encoder.pilot import (
            _assert_matched_disjoint, _assert_model_collections_disjoint,
        )

        tiny = build_matched_ge1_models(seed=2026)
        scaled = build_matched_ge1_models(seed=2026)
        _assert_matched_disjoint(*tiny)
        _assert_matched_disjoint(*scaled)
        _assert_model_collections_disjoint(tiny, scaled)
        for tiny_model, scaled_model in zip(tiny, scaled):
            for name, value in tiny_model.state_dict().items():
                self.assertTrue(
                    torch.equal(value, scaled_model.state_dict()[name]), name
                )

    def test_arm_construction_order_cannot_change_initialization(self):
        from prototype.graph_encoder.model import build_matched_ge1_models

        normal = build_matched_ge1_models(
            seed=2026, encoder_order=("flat", "typed_graph")
        )
        reverse = build_matched_ge1_models(
            seed=2026, encoder_order=("typed_graph", "flat")
        )
        for first, second in zip(normal, reverse):
            self.assertEqual(list(first.state_dict()), list(second.state_dict()))
            for name, value in first.state_dict().items():
                self.assertTrue(torch.equal(value, second.state_dict()[name]), name)

    def test_state_comparison_detects_mutation(self):
        from prototype.graph_encoder.errors import GraphEncoderError
        from prototype.graph_encoder.model import build_matched_ge1_models
        from prototype.graph_encoder.pilot import _assert_model_states_equal

        first = build_matched_ge1_models(seed=2026)[0]
        second = build_matched_ge1_models(seed=2026)[0]
        _assert_model_states_equal(first, second)
        with torch.no_grad():
            next(second.parameters()).add_(1.0)
        with self.assertRaises(GraphEncoderError) as caught:
            _assert_model_states_equal(first, second)
        self.assertEqual(caught.exception.code, "c7_checkpoint_reload_failure")

    def test_scaled_autonomous_path_uses_four_batches_and_frozen_interventions(self):
        from prototype.graph_encoder.autonomous import (
            autonomous_input_from_paired, run_autonomous_evaluation,
        )
        from prototype.graph_encoder.batching import build_paired_batch
        from prototype.graph_encoder.metrics import score_condition
        from prototype.graph_encoder.model import build_matched_ge1_models
        from prototype.graph_encoder.pilot import memory_use_gate
        from prototype.graph_encoder.tests.fixtures import procedural_fixture

        examples = []
        for template in ("E", "R", "EE", "RE"):
            base = procedural_fixture(template).physical
            for index in range(8):
                family_id = "{}_c7_{:02d}".format(template, index)
                examples.append(replace(
                    base,
                    physical_family_id=family_id,
                    metadata=replace(
                        base.metadata,
                        sample_ids=(
                            family_id + "_continuous",
                            family_id + "_quantized",
                        ),
                    ),
                    split_name="operation_template",
                    partition="train",
                ))
        ordered = tuple(sorted(examples, key=lambda item: item.physical_family_id))
        model = build_matched_ge1_models(seed=2026)[0]
        batches = tuple(
            autonomous_input_from_paired(
                build_paired_batch(ordered[start:start + 8]), "flat"
            )
            for start in range(0, len(ordered), 8)
        )
        autonomous = run_autonomous_evaluation(model, batches, seed=2026)
        targets = {item.physical_family_id: item.target for item in ordered}
        metrics = tuple(
            score_condition(condition, targets)
            for condition in autonomous.conditions
        )
        templates = {
            item.physical_family_id: item.metadata.operation_template
            for item in ordered
        }
        gate = memory_use_gate(
            arm="flat",
            condition_metrics=metrics,
            autonomous_result=autonomous,
            templates_by_family=templates,
            checkpoint_identity="procedural-epoch-0050.pt",
            epoch=50,
            access_flags={"development_accessed": False},
        )
        self.assertIn(gate["status"], ("pass", "fail"))
        self.assertEqual(len(gate["deterministic_batch_membership"]), 4)
        self.assertEqual(set(gate["intervention_ratios"]), {
            "P_true", "P_shuffle", "P_mean", "R_shuffle", "R_mean"
        })
        by_name = {item.condition: item for item in autonomous.conditions}
        self.assertTrue(by_name["P_shuffle"].available)
        self.assertTrue(all(
            recipient != donor
            for recipient, donor in by_name["P_shuffle"].memory_assignments
        ))
        self.assertFalse(by_name["P_mean"].singleton_mean_batches)


if __name__ == "__main__":
    unittest.main()
