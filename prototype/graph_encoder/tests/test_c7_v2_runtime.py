"""Real-PyTorch C7-v2 model-lifecycle and epoch-200 orchestration tests."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "real PyTorch is required")
class C7V2RuntimeTests(unittest.TestCase):
    def test_tiny_and_scaled_pairs_are_fresh_matched_and_disjoint(self):
        from prototype.graph_encoder.model import build_matched_ge1_models
        from prototype.graph_encoder.pilot import (
            _assert_matched_disjoint,
            _assert_model_collections_disjoint,
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

    def test_epoch_200_is_written_then_strictly_reloaded_before_evaluation(self):
        from prototype.graph_encoder import autonomous, metrics, training
        from prototype.graph_encoder.c7_v2 import (
            C7V2AccessTracker,
            _train_evaluate_arm,
        )
        from prototype.graph_encoder.config import GE1TrainingConfig
        from prototype.graph_encoder.model import build_matched_ge1_models
        from prototype.graph_encoder.tests.fixtures import procedural_fixture

        examples = tuple(
            procedural_fixture(template).physical
            for template in ("E", "R", "EE", "RE")
        )
        model = build_matched_ge1_models(seed=2026)[0]
        configuration = GE1TrainingConfig()
        family_ids = tuple(sorted(item.physical_family_id for item in examples))
        templates = {
            item.physical_family_id: item.metadata.operation_template
            for item in examples
        }

        def fake_training(unused_model, unused_examples, **kwargs):
            self.assertEqual(kwargs["final_epoch"], 200)
            self.assertEqual(kwargs["checkpoint_epochs"], (200,))
            self.assertEqual(kwargs["fixed_protocol_final_epoch"], 200)
            self.assertEqual(kwargs["selected_checkpoint_epoch"], 200)
            self.assertNotIn("resume_checkpoint", kwargs)
            path = Path(kwargs["checkpoint_directory"]) / "epoch-0200.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"epoch-200-selected-checkpoint")
            return SimpleNamespace(
                completed_epoch=200,
                optimizer_steps=200,
                training_example_presentations=800,
                epoch_records=tuple(range(200)),
                checkpoint_paths=(str(path),),
                selected_experimental_checkpoint=str(path),
                to_dict=lambda: {
                    "completed_epoch": 200,
                    "optimizer_steps": 200,
                    "training_example_presentations": 800,
                    "checkpoint_paths": [str(path)],
                    "selected_experimental_checkpoint": str(path),
                },
            )

        def fake_reload(unused_path, **kwargs):
            self.assertTrue(kwargs["restore_rng"])
            self.assertEqual(kwargs["expected_selected_checkpoint_epoch"], 200)
            kwargs["model"].load_state_dict(model.state_dict(), strict=True)
            return SimpleNamespace(
                completed_epoch=200,
                payload={
                    "completed_epoch": 200,
                    "selected_checkpoint_epoch": 200,
                    "selected_experimental_checkpoint": True,
                    "provenance": {
                        "git_commit": "a" * 40,
                        "slurm_job_id": "123456",
                    },
                },
            )

        conditions = tuple(
            SimpleNamespace(
                condition=name,
                memory_assignments=tuple((item, item) for item in family_ids),
                memory_altered=False,
                alteration_possible=False,
                batch_membership=(("batch", family_ids),),
                elapsed_seconds=0.0,
                unavailable_reason=None,
                available=True,
            )
            for name in ("P_true", "P_shuffle", "P_mean")
        )
        autonomous_result = SimpleNamespace(
            conditions=conditions,
            encoding_seconds=0.0,
        )

        def fake_score(condition, unused_targets):
            return {
                "condition": condition.condition,
                "available": True,
                "family_values": {
                    family_id: {
                        "exact_node_sequence": 1.0,
                        "exact_graph": 1.0,
                        "strict_conversion": 1.0,
                        "complete_executable_validity": 1.0,
                        "depends_on_exactness": 1.0,
                    }
                    for family_id in family_ids
                },
                "metric_computation_seconds": 0.0,
            }

        with tempfile.TemporaryDirectory(prefix="c7-v2-runtime-") as temporary:
            staging = Path(temporary)
            events = []
            with mock.patch.object(
                training, "run_ge1_training", side_effect=fake_training
            ), mock.patch.object(
                training, "load_training_checkpoint", side_effect=fake_reload
            ), mock.patch.object(
                autonomous, "run_autonomous_evaluation",
                return_value=autonomous_result,
            ), mock.patch.object(metrics, "score_condition", side_effect=fake_score):
                result = _train_evaluate_arm(
                    model=model,
                    examples=examples,
                    subset_identity="tiny",
                    staging=staging,
                    repository_root=staging,
                    expected_commit="a" * 40,
                    job_id="123456",
                    training_config=configuration,
                    parameter_counts={},
                    events=events,
                    tracker=C7V2AccessTracker(
                        operation_template_manifest_accessed=True,
                        operation_template_train_payload_accessed=True,
                        tiny_train_payload_accessed=True,
                    ),
                )
        self.assertEqual(result["exact_gate"]["status"], "pass")
        names = [item["event"] for item in events]
        self.assertLess(
            names.index("c7_v2_epoch_200_checkpoint_written"),
            names.index("c7_v2_checkpoint_reloaded"),
        )
        self.assertLess(
            names.index("c7_v2_checkpoint_reloaded"),
            names.index("c7_v2_autonomous_condition_metrics"),
        )


if __name__ == "__main__":
    unittest.main()
