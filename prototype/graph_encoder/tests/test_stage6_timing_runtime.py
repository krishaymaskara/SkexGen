"""Bounded real-PyTorch timing-path coverage for ADR-0014."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from pathlib import Path
import tempfile
import unittest

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


@unittest.skipIf(torch is None, "ADR-0014 timing runtime requires PyTorch")
class Stage6TimingRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_real_matched_flat_graph_five_epoch_timing_lifecycle(self):
        from prototype.graph_encoder.config import GE1TrainingConfig
        from prototype.graph_encoder.grid_magnitude import GRID_MAGNITUDE_PARAMETERIZATION
        from prototype.graph_encoder.model import build_matched_ge1_models
        from prototype.graph_encoder.provenance import C6Provenance
        from prototype.graph_encoder.tests.fixtures import procedural_fixture
        from prototype.graph_encoder.training import run_ge1_training
        base = procedural_fixture("E").physical
        examples = tuple(replace(
            base, physical_family_id="timing-{}".format(index),
            split_name="operation_template", partition="train",
            metadata=replace(base.metadata, sample_ids=(
                "timing-{}-continuous".format(index),
                "timing-{}-quantized".format(index),
            )),
        ) for index in range(4))
        flat, graph = build_matched_ge1_models(
            2026, operation_magnitude_parameterization=GRID_MAGNITUDE_PARAMETERIZATION
        )
        with tempfile.TemporaryDirectory() as temporary:
            for model in (flat, graph):
                provenance = C6Provenance(
                    "GE1-C6-PROVENANCE-v1", None, True, "a" * 40, False, (),
                    "b" * 64, "c" * 64, "d" * 64, "3.8.13", "1.11.0",
                    "cpu", "fixture-host", "123", None, 2026,
                    model.config.encoder, model.config.checkpoint_schema,
                )
                result = run_ge1_training(
                    model, examples, training_config=GE1TrainingConfig(),
                    checkpoint_directory=Path(temporary) / model.config.encoder,
                    repository_root=temporary, expected_commit="a" * 40,
                    final_epoch=5, checkpoint_epochs=(5,),
                    selected_checkpoint_epoch=None,
                    timing_protocol_final_epoch=5, execution_device="cpu",
                    provenance_context=SimpleNamespace(authorized=provenance),
                    provenance_verifier=lambda unused, **kwargs: provenance,
                )
                self.assertEqual(result.completed_epoch, 5)
                self.assertEqual(len(result.epoch_records), 5)
                self.assertEqual(result.checkpoint_opportunities, 5)
                self.assertTrue(all(row.training_seconds >= 0.0 for row in result.epoch_records))


if __name__ == "__main__":
    unittest.main()
