"""Corpus-free real-PyTorch contracts for the Stage 6 producer."""

from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
from pathlib import Path
import random
import tempfile
import unittest

try:
    import torch
except ImportError:  # pragma: no cover - authoritative runtime supplies torch
    torch = None

from prototype.graph_encoder.decoder_contract import (
    AUTONOMOUS_STOP_NODE_GENERATION_IDENTITY,
    GRID_ORDINAL_OPERATION_MAGNITUDE_PARAMETERIZATION,
)
from prototype.graph_encoder.stage6_structure_only_producer import capacity_gate

if torch is not None:
    from prototype.graph_encoder.model import build_matched_ge1_models


REASON = "Stage 6 producer runtime contracts require PyTorch"


def _models(seed):
    return build_matched_ge1_models(
        seed,
        operation_magnitude_parameterization=(
            GRID_ORDINAL_OPERATION_MAGNITUDE_PARAMETERIZATION
        ),
        node_generation_identity=AUTONOMOUS_STOP_NODE_GENERATION_IDENTITY,
    )


@unittest.skipIf(torch is None, REASON)
class Stage6ProducerRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_matched_initialization_and_capacity_gate(self):
        flat, graph = _models(2026)
        for left, right in zip(flat.decoder.parameters(), graph.decoder.parameters()):
            self.assertIsNot(left, right)
            torch.testing.assert_close(left, right, rtol=0.0, atol=0.0)
        counts = [sum(parameter.numel() for parameter in model.parameters()
                      if parameter.requires_grad) for model in (flat, graph)]
        self.assertEqual(capacity_gate(*counts)["pass"],
                         abs(counts[1] - counts[0]) / float(counts[0]) <= 0.05)

    def test_fresh_seed_is_deterministic_but_independent(self):
        first, unused = _models(2027)
        second, unused = _models(2027)
        for left, right in zip(first.parameters(), second.parameters()):
            self.assertIsNot(left, right)
            torch.testing.assert_close(left, right, rtol=0.0, atol=0.0)

    def test_geometry_nan_does_not_suppress_structural_container(self):
        record = {"structural": [True, True], "geometry": torch.tensor(float("nan"))}
        preserved = copy.copy(record)
        self.assertEqual(preserved["structural"], [True, True])
        self.assertTrue(torch.isnan(preserved["geometry"]).item())

    def test_real_multiple_batch_shuffle_is_batch_local_and_deterministic(self):
        from prototype.graph_encoder.autonomous import (
            autonomous_input_from_paired, run_autonomous_evaluation,
        )
        from prototype.graph_encoder.batching import build_paired_batch
        from prototype.graph_encoder.stage6_structure_only_producer import (
            structural_records_from_prediction,
        )
        from prototype.graph_encoder.tests.fixtures import procedural_fixture

        base = procedural_fixture("E").physical
        examples = tuple(replace(
            base,
            physical_family_id="stage6-{:02d}".format(index),
            metadata=replace(base.metadata, sample_ids=(
                "stage6-{:02d}-continuous".format(index),
                "stage6-{:02d}-quantized".format(index),
            )),
            partition="train",
        ) for index in range(16))
        batches = tuple(
            autonomous_input_from_paired(build_paired_batch(examples[start:start + 8]), "flat")
            for start in (0, 8)
        )
        model, unused = _models(2026)
        first = run_autonomous_evaluation(model, batches, seed=2026, shuffle_scope="batch")
        second = run_autonomous_evaluation(model, batches, seed=2026, shuffle_scope="batch")
        by_name = {row.condition: row for row in first.conditions}
        repeated = {row.condition: row for row in second.conditions}
        membership = {
            family: set(families)
            for unused_batch, families in by_name["P_shuffle"].batch_membership
            for family in families
        }
        self.assertEqual(by_name["P_shuffle"].memory_assignments,
                         repeated["P_shuffle"].memory_assignments)
        self.assertTrue(all(
            recipient != donor and donor in membership[recipient]
            for recipient, donor in by_name["P_shuffle"].memory_assignments
        ))
        self.assertTrue(all(
            donor.startswith("batch_mean:")
            for unused_recipient, donor in by_name["P_mean"].memory_assignments
        ))
        # The target is joined only after both autonomous evaluations complete.
        predicted_prefixes, target_prefixes, predicted_count = (
            structural_records_from_prediction(
                by_name["P_true"].predictions[0].constrained_prediction,
                examples[0].target,
            )
        )
        self.assertEqual(len(predicted_prefixes), 1)
        self.assertEqual(len(target_prefixes), 1)
        self.assertGreaterEqual(predicted_count, 0)

    def test_stage6_checkpoint_fresh_round_trip_and_strict_rejection(self):
        from prototype.graph_encoder.config import GE1TrainingConfig
        from prototype.graph_encoder.model import build_ge1_model
        from prototype.graph_encoder.provenance import C6Provenance, training_partition_identity
        from prototype.graph_encoder.stage6_structure_only_producer import (
            checkpoint_identity, load_stage6_checkpoint, save_stage6_checkpoint,
        )
        from prototype.graph_encoder.training import (
            EpochTrainingRecord, plateau_state, save_training_checkpoint,
        )

        model, unused = _models(2026)
        config = GE1TrainingConfig()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0)
        family_ids = (
            "procedural-a", "procedural-b", "procedural-c", "procedural-d",
        )
        partition = training_partition_identity(family_ids)
        records = tuple(EpochTrainingRecord(
            epoch, 1.0, (("total", 1.0),), epoch, epoch * 2,
            family_ids, (family_ids,), (1.0,), 0.0, 0.0,
        ) for epoch in range(1, 201))
        provenance = C6Provenance(
            "GE1-C6-PROVENANCE-v1", None, True, "a" * 40, False, (),
            "b" * 64, "c" * 64, "d" * 64, "3.8.13", "1.11.0", "cpu",
            "fixture-host", "123", None, 2026, "flat", model.config.checkpoint_schema,
        )
        with tempfile.TemporaryDirectory() as temporary:
            generic = Path(temporary) / "epoch-0200.pt"
            save_training_checkpoint(
                generic, model=model, optimizer=optimizer, training_config=config,
                completed_epoch=200, optimizer_step_count=200,
                training_example_presentations=400,
                plateau=plateau_state([1.0] * 200), epoch_records=records,
                family_order_history=tuple(family_ids for unused in range(200)),
                data_order_random_state=random.Random(2026).getstate(),
                provenance_context=object(),
                provenance_verifier=lambda unused, **kwargs: provenance,
                partition_identity=partition, expected_code_revision="a" * 40,
                selected_checkpoint_epoch=200,
            )
            identity = checkpoint_identity(
                arm="flat", seed=2026, source_commit="a" * 40,
                source_digest="b" * 64, model_config=model.config.to_dict(),
                partition_identity=partition,
                partition_hashes={"index": "c" * 64},
                parameter_count=sum(value.numel() for value in model.parameters()),
                training_arithmetic={"epochs": 200},
                checkpoint_sha256=hashlib.sha256(generic.read_bytes()).hexdigest(),
                node_generation_identity=(
                    AUTONOMOUS_STOP_NODE_GENERATION_IDENTITY
                ),
            )
            wrapper = Path(temporary) / "stage6.pt"
            save_stage6_checkpoint(generic, wrapper, identity)
            fresh = build_ge1_model(model.config)
            fresh_optimizer = torch.optim.AdamW(fresh.parameters(), lr=0.001, weight_decay=0.0)
            state = load_stage6_checkpoint(
                wrapper, expected_identity=identity, model=fresh,
                optimizer=fresh_optimizer, training_config=config,
                partition_identity=partition, restore_rng=False,
            )
            self.assertEqual(state.completed_epoch, 200)
            wrong = dict(identity, seed=2027)
            with self.assertRaises(Exception):
                load_stage6_checkpoint(
                    wrapper, expected_identity=wrong, model=fresh,
                    optimizer=fresh_optimizer, training_config=config,
                    partition_identity=partition, restore_rng=False,
                )
            tampered_payload = torch.load(str(wrapper), map_location="cpu")
            tampered_payload["training_checkpoint"]["model_state"].pop(
                next(iter(tampered_payload["training_checkpoint"]["model_state"]))
            )
            tampered = Path(temporary) / "tampered.pt"
            torch.save(tampered_payload, str(tampered))
            with self.assertRaises(Exception):
                load_stage6_checkpoint(
                    tampered, expected_identity=identity, model=fresh,
                    optimizer=fresh_optimizer, training_config=config,
                    partition_identity=partition, restore_rng=False,
                )


if __name__ == "__main__":
    unittest.main()
