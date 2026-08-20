"""PyTorch-independent contracts for ADR-0017 exploratory Stage 6."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

from prototype.graph_encoder.decoder_contract import (
    AUTONOMOUS_STOP_NODE_GENERATION_IDENTITY,
    GRID_ORDINAL_OPERATION_MAGNITUDE_PARAMETERIZATION,
    LEGACY_NODE_GENERATION_IDENTITY,
)
from prototype.graph_encoder.errors import GraphEncoderError
from prototype.graph_encoder.stage6_structure_only import (
    FULL_SEEDS,
    create_artifact,
    summarize_execution,
)
from prototype.graph_encoder.stage6_structure_only_producer import (
    EPOCHS,
    SMOKE_EPOCHS,
    checkpoint_identity,
    optimization_reliability,
    producer_config,
)
from prototype.graph_encoder.stage6_timing import build_timing_record
from prototype.graph_encoder.tests.test_stage6_structure_only_contract import (
    synthetic_payload,
)
from prototype.graph_encoder.tests.test_stage6_timing_contract import (
    record as timing_record,
)
from prototype.graph_encoder.training import _validate_execution_epoch


ROOT = Path(__file__).parents[3]
STOP = AUTONOMOUS_STOP_NODE_GENERATION_IDENTITY
ORDINAL = GRID_ORDINAL_OPERATION_MAGNITUDE_PARAMETERIZATION


def _marked_payload(*, smoke):
    payload = synthetic_payload()
    payload.update({
        "operation_magnitude_parameterization": ORDINAL,
        "node_generation_identity": STOP,
        "exploratory_development_access": True,
        "smoke_protocol": smoke,
        "protocol_final_epoch": SMOKE_EPOCHS if smoke else EPOCHS,
        "stage6_result_eligible": False,
        "train_side_gate_evidence": {
            "optimization_pass": False,
            "structural_memory_pass": False,
            "overall_pass": False,
            "exploratory_continue_applied": True,
        },
    })
    if smoke:
        for row in payload["training_runs"]:
            row.update({
                "epochs": SMOKE_EPOCHS,
                "checkpoint_epoch": SMOKE_EPOCHS,
                "smoke_protocol": True,
                "optimization_reliable": False,
                "plateau_relative_improvement": None,
            })
    else:
        for row in payload["training_runs"]:
            row["smoke_protocol"] = False
    return payload


class ExploratoryStage6Contracts(unittest.TestCase):
    def test_default_producer_remains_legacy_blocking_and_result_eligible(self):
        config = producer_config()
        self.assertEqual(
            config["node_generation_identity"], LEGACY_NODE_GENERATION_IDENTITY
        )
        self.assertEqual(config["operation_magnitude_parameterization"], ORDINAL)
        self.assertFalse(config["exploratory_development_access"])
        self.assertFalse(config["smoke_protocol"])
        self.assertTrue(config["stage6_result_eligible"])

    def test_explicit_exploratory_smoke_is_two_epoch_and_ineligible(self):
        config = producer_config(
            STOP, exploratory_development_access=True, smoke_protocol=True
        )
        self.assertEqual(config["epochs"], 2)
        self.assertEqual(config["fixed_checkpoint_epoch"], 2)
        self.assertFalse(config["stage6_result_eligible"])

    def test_training_budget_accepts_only_exact_smoke_two(self):
        self.assertIsNone(_validate_execution_epoch(
            2, False, smoke_protocol_final_epoch=2
        ))
        for final_epoch, declared in ((1, 2), (2, 1), (3, 3)):
            with self.assertRaises(GraphEncoderError):
                _validate_execution_epoch(
                    final_epoch, False,
                    smoke_protocol_final_epoch=declared,
                )

    def test_smoke_reliability_records_failure_without_weakening_real_gate(self):
        runs = [{
            "arm": arm,
            "seed": seed,
            "epoch_losses": [1.0, 1.0],
            "epoch_gradient_norms": [0.5, 0.5],
            "completed_epoch": 2,
            "checkpoint_epoch": 2,
        } for arm in ("flat", "typed_graph") for seed in FULL_SEEDS]
        scores = {
            (arm, seed): 0.5
            for arm in ("flat", "typed_graph") for seed in FULL_SEEDS
        }
        result = optimization_reliability(
            runs, scores, FULL_SEEDS,
            protocol_final_epoch=2, smoke_protocol=True,
        )
        self.assertFalse(result["pass"])
        self.assertTrue(all(
            not row["pass_before_train_ceiling"] for row in result["runs"]
        ))

    def test_exploratory_and_smoke_summaries_are_forced_inconclusive(self):
        for smoke in (False, True):
            unused, summary = summarize_execution(_marked_payload(smoke=smoke))
            self.assertEqual(summary["interpretation_category"], "inconclusive")
            self.assertFalse(summary["stage6_result_eligible"])
            self.assertFalse(summary["validity_gates_pass"])

    def test_finalized_smoke_manifest_and_config_cannot_claim_result(self):
        payload = _marked_payload(smoke=True)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "artifact"
            result = create_artifact(payload, root, "a" * 40, "123")
            manifest = json.loads((root / "artifact_manifest.json").read_text())
            resolved = json.loads((root / "resolved_config.json").read_text())
            summary = json.loads((root / "summary.json").read_text())
            self.assertTrue(result["smoke_protocol"])
            self.assertTrue(manifest["smoke_protocol"])
            self.assertTrue(resolved["exploratory_development_access"])
            self.assertFalse(summary["stage6_result_eligible"])
            self.assertEqual(summary["interpretation_category"], "inconclusive")

    def test_smoke_checkpoint_identity_is_strict_and_uses_v4_model(self):
        row = checkpoint_identity(
            arm="flat", seed=2026, source_commit="a" * 40,
            source_digest="b" * 64,
            model_config={
                "checkpoint_schema": "GE1-CHECKPOINT-v4",
                "operation_magnitude_parameterization": ORDINAL,
                "node_generation_identity": STOP,
            },
            partition_identity="operation_template_train-407",
            partition_hashes={"index": "c" * 64}, parameter_count=1000,
            training_arithmetic={
                "epochs": 2, "batch_size": 8, "smoke_protocol": True,
            },
            checkpoint_sha256="d" * 64, protocol_final_epoch=2,
            node_generation_identity=STOP,
        )
        self.assertEqual(row["epoch"], 2)
        self.assertEqual(row["checkpoint_schema"], "GE1-CHECKPOINT-v4")
        broken = copy.deepcopy(row)
        broken["training_arithmetic"]["smoke_protocol"] = False
        with self.assertRaises(GraphEncoderError):
            from prototype.graph_encoder.stage6_structure_only_producer import (
                validate_checkpoint_identity,
            )
            validate_checkpoint_identity(broken, expected=broken)

    def test_timing_identity_is_bound_and_tamper_evident(self):
        record = timing_record(cpu=1.0, cuda=2.0)
        rebuilt = build_timing_record(
            source_commit=record["source_identity"]["source_commit"],
            source_tree_sha256=record["source_identity"]["source_tree_sha256"],
            train_index_identity_sha256=record["train_input_identity"][
                "index_identity_sha256"
            ],
            train_payload_digests_sha256=record["train_input_identity"][
                "payload_digests_sha256"
            ],
            runtime_identities={
                device: row["runtime_identity"]
                for device, row in record["candidates"].items()
            },
            raw_measurements={
                device: {
                    arm: {
                        "warmup": values["warmup"]["duration_seconds"],
                        "timed": [
                            item["duration_seconds"]
                            for item in values["timed_measurements"]
                        ],
                    }
                    for arm, values in row["arms"].items()
                }
                for device, row in record["candidates"].items()
            },
            available_wall_seconds_by_device={
                device: row["available_wall_seconds"]
                for device, row in record["candidates"].items()
            },
            node_generation_identity=STOP,
        )
        self.assertEqual(rebuilt["node_generation_identity"], STOP)
        self.assertEqual(rebuilt["operation_magnitude_parameterization"], ORDINAL)

    def test_runners_require_identity_and_exploratory_markers(self):
        producer = (ROOT / (
            "prototype/graph_encoder/adroit/"
            "ge1_stage6_structure_only_producer.slurm"
        )).read_text()
        timing = (ROOT / (
            "prototype/graph_encoder/adroit/"
            "ge1_stage6_hardware_timing_gpu.slurm"
        )).read_text()
        self.assertIn("NODE_GENERATION_IDENTITY", producer)
        self.assertIn("EXPLORATORY_DEVELOPMENT_ACCESS", producer)
        self.assertIn("--smoke-epochs", producer)
        self.assertIn("--node-generation-identity", timing)
        self.assertNotIn("\nsbatch ", producer)
        self.assertNotIn("\nsbatch ", timing)


if __name__ == "__main__":
    unittest.main()
