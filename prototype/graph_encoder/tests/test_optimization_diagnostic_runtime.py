"""Real-PyTorch optimization-diagnostic lifecycle and checkpoint tests."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import random
import tempfile
import unittest

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "real PyTorch is required")
class OptimizationDiagnosticRuntimeTests(unittest.TestCase):
    def test_fresh_matched_pair_has_equal_shared_state_and_disjoint_objects(self):
        from prototype.graph_encoder.model import build_matched_ge1_models
        from prototype.graph_encoder.pilot import _assert_matched_disjoint

        flat, graph = build_matched_ge1_models(seed=2026)
        _assert_matched_disjoint(flat, graph)
        for name, value in flat.decoder.state_dict().items():
            self.assertTrue(torch.equal(value, graph.decoder.state_dict()[name]), name)

    def test_milestone_checkpoint_strict_reload_and_slurm_provenance(self):
        from prototype.graph_encoder.config import GE1TrainingConfig
        from prototype.graph_encoder.model import build_ge1_model, build_matched_ge1_models
        from prototype.graph_encoder.optimization_diagnostic import _checkpoint_provenance
        from prototype.graph_encoder.provenance import C6Provenance, training_partition_identity
        from prototype.graph_encoder.training import (
            EpochTrainingRecord,
            load_training_checkpoint,
            plateau_state,
            save_training_checkpoint,
        )

        model = build_matched_ge1_models(seed=2026)[0]
        training_config = GE1TrainingConfig()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=training_config.learning_rate,
            weight_decay=training_config.weight_decay,
        )
        family_ids = tuple("fixture-family-{}".format(index) for index in range(4))
        partition = training_partition_identity(family_ids)
        records = tuple(
            EpochTrainingRecord(
                epoch,
                1.0 / epoch,
                (("total", 1.0 / epoch),),
                epoch,
                epoch * 4,
                family_ids,
                (family_ids,),
                (1.0,),
                0.0,
                0.0,
            )
            for epoch in range(1, 101)
        )
        plateau = plateau_state(tuple(item.mean_loss for item in records))
        provenance = C6Provenance(
            "GE1-C6-PROVENANCE-v1",
            None,
            True,
            "a" * 40,
            False,
            (),
            "b" * 64,
            "c" * 64,
            "d" * 64,
            "3.8.13",
            "1.11.0",
            "cpu",
            "fixture-host",
            "123456",
            None,
            2026,
            "flat",
            "GE1-CHECKPOINT-v1",
        )
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "flat-seed2026-update0100.pt"
            save_training_checkpoint(
                path,
                model=model,
                optimizer=optimizer,
                training_config=training_config,
                completed_epoch=100,
                optimizer_step_count=100,
                training_example_presentations=400,
                plateau=plateau,
                epoch_records=records,
                family_order_history=tuple(family_ids for unused in range(100)),
                data_order_random_state=random.Random(2026).getstate(),
                provenance_context=object(),
                provenance_verifier=lambda unused, **kwargs: provenance,
                partition_identity=partition,
                expected_code_revision="a" * 40,
            )
            reloaded = build_ge1_model(model.config)
            reloaded_optimizer = torch.optim.AdamW(
                reloaded.parameters(),
                lr=training_config.learning_rate,
                weight_decay=training_config.weight_decay,
            )
            state = load_training_checkpoint(
                path,
                model=reloaded,
                optimizer=reloaded_optimizer,
                training_config=training_config,
                partition_identity=partition,
                expected_code_revision="a" * 40,
                restore_rng=False,
            )
            self.assertEqual(state.completed_epoch, 100)
            self.assertEqual(state.optimizer_step_count, 100)
            self.assertEqual(state.training_example_presentations, 400)
            self.assertEqual(_checkpoint_provenance(path)["slurm_job_id"], "123456")
            for name, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, reloaded.state_dict()[name]), name)


if __name__ == "__main__":
    unittest.main()
