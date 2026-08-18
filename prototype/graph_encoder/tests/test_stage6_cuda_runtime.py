"""Authoritative-environment CUDA contracts for ADR-0014."""

from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import os
from pathlib import Path
import random
import tempfile
import unittest

try:
    import torch
except ImportError:  # pragma: no cover - authoritative GPU runtime supplies torch
    torch = None


CUDA_AVAILABLE = bool(torch is not None and torch.cuda.is_available())
REASON = "ADR-0014 CUDA contracts require an available CUDA device"


@unittest.skipUnless(CUDA_AVAILABLE, REASON)
class Stage6CudaRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        from prototype.graph_encoder.stage6_device import configure_stage6_runtime
        cls.runtime = configure_stage6_runtime(torch, "cuda:0", seed=2026)

    def examples(self, count=8):
        from prototype.graph_encoder.tests.fixtures import procedural_fixture
        base = procedural_fixture("E").physical
        return tuple(replace(
            base,
            physical_family_id="cuda-fixture-{:02d}".format(index),
            metadata=replace(base.metadata, sample_ids=(
                "cuda-fixture-{:02d}-continuous".format(index),
                "cuda-fixture-{:02d}-quantized".format(index),
            )),
            partition="train",
        ) for index in range(count))

    def test_matched_models_and_every_training_tensor_are_cuda(self):
        from prototype.graph_encoder.batching import build_paired_batch
        from prototype.graph_encoder.grid_magnitude import GRID_MAGNITUDE_PARAMETERIZATION
        from prototype.graph_encoder.model import build_matched_ge1_models
        from prototype.graph_encoder.training import _training_tensors
        flat, graph = build_matched_ge1_models(
            2026, operation_magnitude_parameterization=GRID_MAGNITUDE_PARAMETERIZATION
        )
        for model in (flat.to("cuda:0"), graph.to("cuda:0")):
            self.assertEqual({str(value.device) for value in model.parameters()}, {"cuda:0"})
            tensors = _training_tensors(
                build_paired_batch(self.examples()), model.config.encoder,
                execution_device="cuda:0",
            )
            devices = []
            def visit(value):
                if torch.is_tensor(value):
                    devices.append(str(value.device))
                elif isinstance(value, dict):
                    for item in value.values():
                        visit(item)
            visit(tensors)
            self.assertTrue(devices)
            self.assertEqual(set(devices), {"cuda:0"})

    def test_bounded_optimizer_step_is_finite_clipped_and_repeatable(self):
        from prototype.graph_encoder.batching import build_paired_batch
        from prototype.graph_encoder.grid_magnitude import GRID_MAGNITUDE_PARAMETERIZATION
        from prototype.graph_encoder.losses import common_ge1_loss
        from prototype.graph_encoder.model import build_matched_ge1_models
        from prototype.graph_encoder.training import _training_tensors
        states = []
        for unused_repeat in range(2):
            torch.manual_seed(2026)
            torch.cuda.manual_seed_all(2026)
            flat, graph = build_matched_ge1_models(
                2026, operation_magnitude_parameterization=GRID_MAGNITUDE_PARAMETERIZATION
            )
            repeat_states = []
            for model in (flat.to("cuda:0"), graph.to("cuda:0")):
                tensors = _training_tensors(
                    build_paired_batch(self.examples()), model.config.encoder,
                    execution_device="cuda:0",
                )
                optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0)
                optimizer.zero_grad(set_to_none=True)
                output = model.teacher_forced(
                    tensors["encoder_input"], tensors["target"],
                    tensors["profile_targets"],
                )
                loss = common_ge1_loss(
                    output.decoder_output, tensors["target"],
                    tensors["profile_targets"], model.config,
                    grid_magnitude_logits=output.grid_magnitude_logits,
                )
                self.assertTrue(torch.isfinite(loss.total).item())
                loss.total.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                self.assertTrue(torch.isfinite(norm).item())
                self.assertTrue(all(
                    parameter.grad is None or torch.isfinite(parameter.grad).all().item()
                    for parameter in model.parameters()
                ))
                optimizer.step()
                repeat_states.append(tuple(
                    value.detach().cpu().clone() for value in model.state_dict().values()
                ))
            states.append(repeat_states)
        for first_arm, second_arm in zip(states[0], states[1]):
            for first, second in zip(first_arm, second_arm):
                torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)

    def test_autonomous_true_shuffle_mean_are_cuda_and_batch_local(self):
        from prototype.graph_encoder.autonomous import (
            autonomous_input_from_paired, run_autonomous_evaluation,
        )
        from prototype.graph_encoder.batching import build_paired_batch
        from prototype.graph_encoder.grid_magnitude import GRID_MAGNITUDE_PARAMETERIZATION
        from prototype.graph_encoder.model import build_matched_ge1_models
        examples = self.examples(16)
        batches = tuple(autonomous_input_from_paired(
            build_paired_batch(examples[start:start + 8]), "flat",
            execution_device="cuda:0",
        ) for start in (0, 8))
        model, unused = build_matched_ge1_models(
            2026, operation_magnitude_parameterization=GRID_MAGNITUDE_PARAMETERIZATION
        )
        result = run_autonomous_evaluation(
            model.to("cuda:0"), batches, seed=2026, shuffle_scope="batch",
            execution_device="cuda:0",
        )
        by_name = {row.condition: row for row in result.conditions}
        self.assertEqual(set(by_name), {"P_true", "P_shuffle", "P_mean"})
        membership = {
            family: set(families)
            for unused_batch, families in by_name["P_shuffle"].batch_membership
            for family in families
        }
        self.assertTrue(all(
            recipient != donor and donor in membership[recipient]
            for recipient, donor in by_name["P_shuffle"].memory_assignments
        ))

    def test_cuda_checkpoint_fresh_recovery_restores_model_optimizer_and_rng(self):
        from prototype.graph_encoder.config import GE1TrainingConfig
        from prototype.graph_encoder.grid_magnitude import GRID_MAGNITUDE_PARAMETERIZATION
        from prototype.graph_encoder.model import build_ge1_model, build_matched_ge1_models
        from prototype.graph_encoder.provenance import C6Provenance, training_partition_identity
        from prototype.graph_encoder.stage6_device import timing_hardware_identity
        from prototype.graph_encoder.stage6_structure_only_producer import (
            checkpoint_identity, load_stage6_checkpoint, save_stage6_checkpoint,
        )
        from prototype.graph_encoder.training import (
            EpochTrainingRecord, plateau_state, save_training_checkpoint,
        )
        model, unused = build_matched_ge1_models(
            2026, operation_magnitude_parameterization=GRID_MAGNITUDE_PARAMETERIZATION
        )
        model = model.to("cuda:0")
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0)
        optimizer.zero_grad(set_to_none=True)
        for parameter in model.parameters():
            parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        config = GE1TrainingConfig()
        family_ids = ("fixture-a", "fixture-b", "fixture-c", "fixture-d")
        partition = training_partition_identity(family_ids)
        records = tuple(EpochTrainingRecord(
            epoch, 1.0, (("total", 1.0),), epoch, epoch * 2,
            family_ids, (family_ids,), (1.0,), 0.0, 0.0,
        ) for epoch in range(1, 201))
        provenance = C6Provenance(
            "GE1-C6-PROVENANCE-v1", None, True, "a" * 40, False, (),
            "b" * 64, "c" * 64, "d" * 64, "3.8.13", "1.11.0", "cuda:0",
            "fixture-host", "123", None, 2026, "flat", model.config.checkpoint_schema,
        )
        with tempfile.TemporaryDirectory() as temporary:
            generic = Path(temporary) / "generic.pt"
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
                partition_identity=partition, partition_hashes={"index": "c" * 64},
                parameter_count=sum(value.numel() for value in model.parameters()),
                training_arithmetic={"epochs": 200},
                checkpoint_sha256=hashlib.sha256(generic.read_bytes()).hexdigest(),
                execution_device="cuda:0", runtime_identity=self.runtime,
                timing_hardware_identity=timing_hardware_identity(self.runtime),
                cuda_rng_preserved=True,
            )
            wrapper = Path(temporary) / "wrapper.pt"
            wrapper_sha = save_stage6_checkpoint(generic, wrapper, identity)
            fresh = build_ge1_model(model.config).to("cuda:0")
            fresh_optimizer = torch.optim.AdamW(fresh.parameters(), lr=0.001, weight_decay=0.0)
            state = load_stage6_checkpoint(
                wrapper, expected_identity=identity, model=fresh,
                optimizer=fresh_optimizer, training_config=config,
                partition_identity=partition, restore_rng=True,
                execution_device="cuda:0",
                expected_wrapper_sha256=wrapper_sha,
            )
            self.assertEqual(state.completed_epoch, 200)
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value, fresh.state_dict()[name], rtol=0.0, atol=0.0)
            expected_optimizer = optimizer.state_dict()
            recovered_optimizer = fresh_optimizer.state_dict()
            self.assertEqual(
                expected_optimizer["param_groups"],
                recovered_optimizer["param_groups"],
            )
            self.assertEqual(
                set(expected_optimizer["state"]),
                set(recovered_optimizer["state"]),
            )
            for parameter_id, expected_fields in expected_optimizer["state"].items():
                recovered_fields = recovered_optimizer["state"][parameter_id]
                self.assertEqual(set(expected_fields), set(recovered_fields))
                for field, expected_value in expected_fields.items():
                    recovered_value = recovered_fields[field]
                    if torch.is_tensor(expected_value):
                        torch.testing.assert_close(
                            expected_value, recovered_value, rtol=0.0, atol=0.0
                        )
                    else:
                        self.assertEqual(expected_value, recovered_value)
            self.assertTrue(state.payload["rng_states"]["torch_cuda"] is not None)
            torch.testing.assert_close(
                torch.get_rng_state(),
                state.payload["rng_states"]["torch_cpu"].cpu(),
                rtol=0.0, atol=0.0,
            )
            for observed, expected in zip(
                torch.cuda.get_rng_state_all(),
                state.payload["rng_states"]["torch_cuda"],
            ):
                torch.testing.assert_close(
                    observed, expected.cpu(), rtol=0.0, atol=0.0
                )
            bad = dict(identity, execution_device="cpu")
            with self.assertRaises(Exception):
                load_stage6_checkpoint(
                    wrapper, expected_identity=bad, model=fresh,
                    optimizer=fresh_optimizer, training_config=config,
                    partition_identity=partition, restore_rng=True,
                    execution_device="cuda:0",
                )
            bad_gpu = copy.deepcopy(identity)
            bad_gpu["runtime_identity"]["gpu"]["name"] = "wrong-gpu"
            with self.assertRaises(Exception):
                load_stage6_checkpoint(
                    wrapper, expected_identity=bad_gpu, model=fresh,
                    optimizer=fresh_optimizer, training_config=config,
                    partition_identity=partition, restore_rng=True,
                    execution_device="cuda:0",
                    expected_wrapper_sha256=wrapper_sha,
                )
            tampered_tensor = Path(temporary) / "tampered-tensor.pt"
            tampered_payload = torch.load(str(wrapper), map_location="cpu")
            first_name = next(iter(tampered_payload["training_checkpoint"]["model_state"]))
            tampered_payload["training_checkpoint"]["model_state"][first_name].add_(1)
            torch.save(tampered_payload, str(tampered_tensor))
            with self.assertRaises(Exception):
                load_stage6_checkpoint(
                    tampered_tensor, expected_identity=identity, model=fresh,
                    optimizer=fresh_optimizer, training_config=config,
                    partition_identity=partition, restore_rng=True,
                    execution_device="cuda:0",
                    expected_wrapper_sha256=wrapper_sha,
                )
            tampered_rng = Path(temporary) / "tampered-rng.pt"
            tampered_payload = torch.load(str(wrapper), map_location="cpu")
            tampered_payload["training_checkpoint"]["rng_states"]["torch_cuda"] = None
            torch.save(tampered_payload, str(tampered_rng))
            with self.assertRaises(Exception):
                load_stage6_checkpoint(
                    tampered_rng, expected_identity=identity, model=fresh,
                    optimizer=fresh_optimizer, training_config=config,
                    partition_identity=partition, restore_rng=True,
                    execution_device="cuda:0",
                    expected_wrapper_sha256=wrapper_sha,
                )

    def test_cuda_configuration_disables_tf32_and_has_no_fallback(self):
        self.assertTrue(torch.are_deterministic_algorithms_enabled())
        self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
        self.assertFalse(torch.backends.cudnn.allow_tf32)
        self.assertEqual(self.runtime["execution_device"], "cuda:0")
        self.assertEqual(self.runtime["visible_cuda_device_count"], 1)

    def test_procedural_cuda_producer_record_is_finalizer_compatible(self):
        from prototype.graph_encoder.stage6_device import timing_hardware_identity
        from prototype.graph_encoder.stage6_structure_only import (
            ARMS, FULL_SEEDS, create_artifact, verify_artifact,
        )
        from prototype.graph_encoder.stage6_structure_only_producer import (
            checkpoint_identity, create_producer_artifact, load_execution_record,
        )
        from prototype.graph_encoder.stage6_timing import build_timing_record
        from prototype.graph_encoder.tests.test_stage6_structure_only_producer_contract import (
            _bundle, _payload,
        )
        from prototype.graph_encoder.tests.test_stage6_timing_contract import (
            runtime as fixture_runtime,
        )

        payload = _payload()
        timing = build_timing_record(
            source_commit="a" * 40,
            source_tree_sha256="b" * 64,
            train_index_identity_sha256="c" * 64,
            train_payload_digests_sha256="d" * 64,
            runtime_identities={
                "cpu": fixture_runtime("cpu"),
                "cuda:0": self.runtime,
            },
            raw_measurements={
                "cpu": {
                    arm: {"warmup": 10.0, "timed": [10.0, 10.0, 10.0]}
                    for arm in ARMS
                },
                "cuda:0": {
                    arm: {"warmup": 5.0, "timed": [5.0, 5.0, 5.0]}
                    for arm in ARMS
                },
            },
            available_wall_seconds_by_device={
                "cpu": 100000.0, "cuda:0": 100000.0,
            },
        )
        hardware = timing_hardware_identity(self.runtime)
        payload["timing_fallback"]["evidence"] = timing
        payload["execution_evidence"] = {
            "selected_execution_device": "cuda:0",
            "runtime_identity": self.runtime,
            "timing_hardware_identity": hardware,
            "timing_version": timing["version"],
            "device_selected_only_by_timing": True,
            "cuda_peak_memory_bytes": 1,
            "verification_status": "pass",
        }
        for row in payload["training_runs"]:
            row["execution_device"] = "cuda:0"
        checkpoints = []
        for arm in ARMS:
            for seed in FULL_SEEDS:
                checkpoints.append({
                    **checkpoint_identity(
                        arm=arm, seed=seed, source_commit="a" * 40,
                        source_digest="b" * 64,
                        model_config={"checkpoint_schema": "GE1-C6-CHECKPOINT-v2"},
                        partition_identity="operation_template_train-407",
                        partition_hashes={"index.json": "c" * 64},
                        parameter_count=1000,
                        training_arithmetic={"epochs": 200, "batch_size": 8},
                        checkpoint_sha256="d" * 64,
                        execution_device="cuda:0", runtime_identity=self.runtime,
                        timing_hardware_identity=hardware,
                        cuda_rng_preserved=True,
                    ),
                    "stage6_wrapper_sha256": "e" * 64,
                })
        bundle = _bundle(checkpoints, payload)
        payload["checkpoint_bundle_reference"] = bundle
        with tempfile.TemporaryDirectory() as temporary:
            producer_path = Path(temporary) / "producer"
            create_producer_artifact(
                payload, checkpoints, producer_path, job_id="123",
                checkpoint_bundle_reference=bundle,
            )
            final_payload = load_execution_record(producer_path)
            final_path = Path(temporary) / "final"
            create_artifact(
                final_payload, final_path, expected_commit="a" * 40,
                job_id="124",
            )
            self.assertEqual(
                verify_artifact(
                    final_path, expected_commit="a" * 40,
                    expected_job_id="124",
                )["verification_status"],
                "pass",
            )


if __name__ == "__main__":
    unittest.main()
