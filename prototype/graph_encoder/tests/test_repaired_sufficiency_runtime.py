"""Real-PyTorch tests for the prospective repaired sufficiency path."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from prototype.graph_encoder.tests.fixtures import procedural_fixture


try:
    import torch
except ImportError:
    torch = None


TORCH_REASON = "repaired sufficiency runtime tests require real PyTorch"


@unittest.skipUnless(torch is not None, TORCH_REASON)
class RepairedSufficiencyRuntimeTests(unittest.TestCase):
    def _examples(self):
        return tuple(
            procedural_fixture(template).physical
            for template in ("E", "R", "EE", "RE")
        )

    def test_both_repaired_arms_construct_and_take_synthetic_training_step(self):
        from prototype.constrained_profile_decoder import profile_targets_for_loss
        from prototype.graph_encoder.autonomous import autonomous_input_from_paired
        from prototype.graph_encoder.batching import build_paired_batch
        from prototype.graph_encoder.losses import common_ge1_loss
        from prototype.graph_encoder.model import build_matched_ge1_models

        paired = build_paired_batch(self._examples())
        target = paired.target.to_torch(torch)
        flat = paired.flat_input.to_torch(torch)
        profiles = profile_targets_for_loss(paired.target, flat["geometry"])
        for model in build_matched_ge1_models(seed=2026):
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            autonomous_input = autonomous_input_from_paired(
                paired, model.config.encoder, torch
            )
            output = model.teacher_forced(
                autonomous_input.encoder_input, target, profiles
            )
            loss = common_ge1_loss(
                output.decoder_output, target, profiles, model.config
            )
            self.assertTrue(torch.isfinite(loss.total))
            optimizer.zero_grad(set_to_none=True)
            loss.total.backward()
            optimizer.step()
            self.assertTrue(all(
                torch.isfinite(parameter).all().item()
                for parameter in model.parameters()
            ))

    def test_matched_shared_decoder_initialization_is_preserved(self):
        from prototype.graph_encoder.model import build_matched_ge1_models

        flat, graph = build_matched_ge1_models(seed=2026)
        self.assertEqual(
            tuple(flat.decoder.state_dict()), tuple(graph.decoder.state_dict())
        )
        for name, value in flat.decoder.state_dict().items():
            self.assertTrue(torch.equal(value, graph.decoder.state_dict()[name]), name)
            self.assertIsNot(value, graph.decoder.state_dict()[name])

    def test_strict_repaired_inference_and_recovery_identity_reload_succeeds(self):
        from prototype.graph_encoder.checkpoint import (
            load_ge1_checkpoint,
            save_ge1_checkpoint,
        )
        from prototype.graph_encoder.decoder_contract import (
            POSITIVE_OPERATION_MAGNITUDE_PARAMETERIZATION,
        )
        from prototype.graph_encoder.model import build_matched_ge1_models
        from prototype.graph_encoder.repaired_sufficiency import (
            validate_repaired_reloaded_checkpoint,
        )

        model = build_matched_ge1_models(seed=2026)[0]
        with tempfile.TemporaryDirectory(prefix="repaired-reload-") as temporary:
            path = Path(temporary) / "inference.pt"
            save_ge1_checkpoint(path, model, code_revision="a" * 40)
            loaded, payload = load_ge1_checkpoint(
                path, expected_code_revision="a" * 40
            )
        self.assertEqual(
            payload["operation_magnitude_parameterization"],
            POSITIVE_OPERATION_MAGNITUDE_PARAMETERIZATION,
        )
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, loaded.state_dict()[name]), name)
        resume = SimpleNamespace(
            completed_epoch=200,
            payload={
                "completed_epoch": 200,
                "selected_checkpoint_epoch": 200,
                "selected_experimental_checkpoint": True,
                "model_config": model.config.to_dict(),
                "provenance": {
                    "git_commit": "a" * 40,
                    "slurm_job_id": "123456",
                    "operation_magnitude_parameterization": (
                        POSITIVE_OPERATION_MAGNITUDE_PARAMETERIZATION
                    ),
                },
            },
        )
        self.assertIs(
            validate_repaired_reloaded_checkpoint(
                resume, expected_commit="a" * 40, job_id="123456"
            ),
            resume,
        )

    def test_legacy_repaired_inference_checkpoint_interchange_fails(self):
        from prototype.graph_encoder.checkpoint import (
            GE1CheckpointError,
            load_ge1_checkpoint,
            save_ge1_checkpoint,
        )
        from prototype.graph_encoder.decoder_contract import (
            LEGACY_OPERATION_MAGNITUDE_PARAMETERIZATION,
        )
        from prototype.graph_encoder.model import build_legacy_matched_ge1_models

        legacy = build_legacy_matched_ge1_models(seed=2026)[0]
        with tempfile.TemporaryDirectory(prefix="repaired-legacy-") as temporary:
            path = Path(temporary) / "legacy.pt"
            save_ge1_checkpoint(path, legacy, code_revision="a" * 40)
            with self.assertRaises(GE1CheckpointError):
                load_ge1_checkpoint(path, expected_code_revision="a" * 40)
            loaded, payload = load_ge1_checkpoint(
                path,
                expected_code_revision="a" * 40,
                expected_operation_magnitude_parameterization=(
                    LEGACY_OPERATION_MAGNITUDE_PARAMETERIZATION
                ),
            )
        self.assertEqual(
            loaded.config.operation_magnitude_parameterization,
            LEGACY_OPERATION_MAGNITUDE_PARAMETERIZATION,
        )
        self.assertEqual(
            payload["operation_magnitude_parameterization"],
            LEGACY_OPERATION_MAGNITUDE_PARAMETERIZATION,
        )

    def test_autonomous_output_feeds_fidelity_only_after_generation(self):
        from prototype.graph_encoder.autonomous import (
            autonomous_input_from_paired,
            run_autonomous_evaluation,
        )
        from prototype.graph_encoder.batching import build_paired_batch
        from prototype.graph_encoder.decoder_contract import (
            POSITIVE_OPERATION_MAGNITUDE_PARAMETERIZATION,
        )
        from prototype.graph_encoder.model import build_matched_ge1_models
        from prototype.graph_encoder.operation_fidelity import (
            operation_geometry_fidelity_gate,
        )
        from prototype.graph_encoder.repaired_sufficiency import (
            REPAIRED_CHECKPOINT_ROLE,
            REPAIRED_GATE_VERSION,
            REPAIRED_PROTOCOL_VERSION,
        )

        examples = self._examples()
        paired = build_paired_batch(examples)
        model = build_matched_ge1_models(seed=2026)[0]
        input_batch = autonomous_input_from_paired(paired, "flat", torch)
        autonomous = run_autonomous_evaluation(model, (input_batch,), seed=2026)
        true = next(item for item in autonomous.conditions if item.condition == "P_true")
        gate = operation_geometry_fidelity_gate(
            arm="flat",
            subset_identity="tiny",
            condition_result=true,
            examples_by_family={item.physical_family_id: item for item in examples},
            checkpoint_identity="synthetic",
            epoch=200,
            access_flags={},
            protocol_version=REPAIRED_PROTOCOL_VERSION,
            gate_version=REPAIRED_GATE_VERSION,
            checkpoint_role=REPAIRED_CHECKPOINT_ROLE,
            operation_magnitude_parameterization=(
                POSITIVE_OPERATION_MAGNITUDE_PARAMETERIZATION
            ),
        )
        self.assertEqual(gate["generation_condition"], "P_true")
        self.assertEqual(len(gate["family_summaries"]), 4)
        self.assertTrue(gate["sample_operation_evidence"])

    def test_autonomous_model_interface_rejects_target_tensor(self):
        from prototype.graph_encoder.autonomous import run_autonomous_evaluation
        from prototype.graph_encoder.model import build_matched_ge1_models

        model = build_matched_ge1_models(seed=2026)[0]
        with self.assertRaises(TypeError):
            run_autonomous_evaluation(
                model, (), seed=2026, target=torch.zeros(1)
            )

    def test_operation_magnitude_gradients_are_finite(self):
        from prototype.graph_encoder.shared_decoder import SharedGE1Decoder

        decoder = SharedGE1Decoder()
        raw = torch.tensor(
            ((-100.0, -2.0, 0.0, 2.0, -10.0, 10.0),),
            requires_grad=True,
        )
        values = decoder.parameterize_remaining_geometry(raw)
        operation_loss = values[..., 4:6].sum()
        operation_loss.backward()
        self.assertIsNotNone(raw.grad)
        self.assertTrue(torch.isfinite(raw.grad).all())
        self.assertTrue(torch.isfinite(values[..., 4:6]).all())

    def test_repaired_transformation_is_positive_finite_and_bounded(self):
        from prototype.graph_encoder.shared_decoder import SharedGE1Decoder

        decoder = SharedGE1Decoder()
        maximum = torch.finfo(torch.float32).max
        logits = torch.tensor((-maximum, -10.0, 0.0, 10.0, maximum))
        raw = logits.view(-1, 1).expand(-1, 6).contiguous()
        operation = decoder.parameterize_remaining_geometry(raw)[..., 4:6]
        self.assertTrue(torch.isfinite(operation).all())
        self.assertTrue((operation > 0.0).all())
        self.assertTrue((operation <= 1.0).all())


if __name__ == "__main__":
    unittest.main()
