"""Pure contracts for the ADR-0014 train-gate postmortem."""

from __future__ import annotations

import copy
import inspect
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest

from prototype.graph_encoder.errors import GraphEncoderError
from prototype.graph_encoder.pilot import _atomic_write_json, _atomic_write_jsonl
from prototype.graph_encoder.stage6_structure_only import (
    ARMS,
    FULL_SEEDS,
    PROTOCOL_VERSION,
)
from prototype.graph_encoder.stage6_structure_only_producer import (
    checkpoint_identity,
)
from prototype.graph_encoder.stage6_train_gate_postmortem import (
    ACCESS_RECORD,
    ARTIFACT_FILES,
    ARTIFACT_VERSION,
    DIAGNOSTIC_VERSION,
    EXPECTED_FAMILY_RECORD_COUNT,
    EXPECTED_OPTIMIZER_STEPS,
    EXPECTED_TRAINING_PRESENTATIONS,
    EXPECTED_WRAPPER_NAMES,
    HISTORICAL_INNER_SLURM_JOB_ID,
    PRODUCER_JOB_ID,
    PRODUCER_EXECUTION_EVIDENCE,
    PRODUCER_SOURCE_COMMIT,
    _expected_training_arithmetic,
    _validate_retained_inner_checkpoint,
    analyze_postmortem,
    finalize_artifact,
    parse_checkpoint_hashes,
    source_commit_python_sha256,
    run_postmortem,
    verify_artifact,
)
from prototype.graph_encoder.stage6_train_gate_postmortem_audit import audit
from prototype.graph_encoder.tests.test_stage6_structure_only_contract import (
    synthetic_payload,
)


DIAGNOSTIC_COMMIT = "d" * 40
DIAGNOSTIC_JOB = "3355999"
INNER_SOURCE_DIGEST = "e" * 64
INNER_FAMILY_IDS = ("family-a", "family-b")


def _training_runs():
    return [
        {
            "arm": arm,
            "seed": seed,
            "completed_epoch": 200,
            "checkpoint_epoch": 200,
            "epoch_losses": [1.0] * 200,
            "epoch_gradient_norms": [0.25] * 200,
        }
        for seed in FULL_SEEDS for arm in ARMS
    ]


def _inner_resume():
    return SimpleNamespace(
        completed_epoch=200,
        optimizer_step_count=EXPECTED_OPTIMIZER_STEPS,
        training_example_presentations=EXPECTED_TRAINING_PRESENTATIONS,
        epoch_records=[None] * 200,
        family_order_history=[INNER_FAMILY_IDS] * 200,
        payload={
            "provenance": {
                "git_commit": PRODUCER_SOURCE_COMMIT,
                "source_tree_sha256": INNER_SOURCE_DIGEST,
                "device": "cpu",
                "slurm_job_id": None,
                "encoder_arm": "flat",
                "seed": 2026,
            },
            "rng_states": {"torch_cuda": None},
        },
    )


def _validate_inner(resume):
    return _validate_retained_inner_checkpoint(
        resume,
        arm="flat",
        seed=2026,
        train_family_ids=INNER_FAMILY_IDS,
        producer_source_digest=INNER_SOURCE_DIGEST,
    )


def _train_records():
    return [
        row for row in synthetic_payload()["family_records"]
        if row["cohort"] == "train"
    ]


def _checkpoint_evidence():
    runtime = {
        "execution_device": "cpu",
        "python_version": "3.8.13",
        "torch_version": "1.11.0",
        "torch_cuda_build_version": None,
        "cpu_processor": "fixture-cpu",
        "platform": "fixture-platform",
        "cpu_threads": 1,
        "gpu": None,
        "deterministic_algorithms": True,
        "tf32_matmul_allowed": False,
        "tf32_cudnn_allowed": False,
        "cuda_rng_preservation_required": False,
    }
    from prototype.graph_encoder.stage6_device import timing_hardware_identity
    output = []
    for seed in FULL_SEEDS:
        for arm in ARMS:
            name = "stage6-{}-seed{}.pt".format(arm, seed)
            identity = checkpoint_identity(
                arm=arm,
                seed=seed,
                source_commit=PRODUCER_SOURCE_COMMIT,
                source_digest="e" * 64,
                model_config={"checkpoint_schema": "fixture"},
                partition_identity={"partition": "train"},
                partition_hashes={
                    "index_identity_sha256": "f" * 64,
                    "payload_digests_sha256": "1" * 64,
                },
                parameter_count=1,
                training_arithmetic=_expected_training_arithmetic(),
                checkpoint_sha256="a" * 64,
                execution_device="cpu",
                runtime_identity=runtime,
                timing_hardware_identity=timing_hardware_identity(runtime),
            )
            output.append({
                "wrapper_name": name,
                "stage6_wrapper_sha256": "b" * 64,
                "identity": identity,
                "strict_recovery_passed": True,
                "checkpoint_modified": False,
            })
    return output


class Stage6TrainGatePostmortemContractTests(unittest.TestCase):
    def test_historical_cleanenv_proves_inner_slurm_job_id_is_none(self):
        root = Path(__file__).resolve().parents[3]
        runner = subprocess.check_output((
            "git", "show", PRODUCER_SOURCE_COMMIT
            + ":prototype/graph_encoder/adroit/"
            "ge1_stage6_structure_only_producer.slurm",
        ), cwd=str(root), universal_newlines=True)
        provenance = subprocess.check_output((
            "git", "show", PRODUCER_SOURCE_COMMIT
            + ":prototype/graph_encoder/provenance.py",
        ), cwd=str(root), universal_newlines=True)
        producer = subprocess.check_output((
            "git", "show", PRODUCER_SOURCE_COMMIT
            + ":prototype/graph_encoder/stage6_structure_only_producer.py",
        ), cwd=str(root), universal_newlines=True)
        container = runner[
            runner.index("container_python()"):
            runner.index("test -d \"$REPOSITORY/.git\"")
        ]
        self.assertIn("apptainer exec --cleanenv", container)
        self.assertNotIn("SLURM_JOB_ID", container)
        self.assertIn('--job-id "$SLURM_JOB_ID"', runner)
        self.assertIn('os.environ.get("SLURM_JOB_ID")', provenance)
        run_body = producer[
            producer.index("def run_stage6_producer("):
            producer.index("def _require_runtime(")
        ]
        self.assertNotIn('os.environ["SLURM_JOB_ID"]', run_body)
        self.assertIs(HISTORICAL_INNER_SLURM_JOB_ID, None)

    def test_exact_historical_inner_provenance_passes(self):
        self.assertTrue(_validate_inner(_inner_resume()))

    def test_missing_or_incorrect_inner_job_id_fails_exactly(self):
        for value in ("3354961", "3355101", "arbitrary"):
            resume = _inner_resume()
            resume.payload["provenance"]["slurm_job_id"] = value
            with self.subTest(value=value), self.assertRaises(
                GraphEncoderError
            ) as raised:
                _validate_inner(resume)
            self.assertEqual(
                raised.exception.detail,
                "inner checkpoint field provenance.slurm_job_id differs",
            )
        resume = _inner_resume()
        del resume.payload["provenance"]["slurm_job_id"]
        with self.assertRaises(GraphEncoderError) as raised:
            _validate_inner(resume)
        self.assertEqual(
            raised.exception.detail,
            "inner checkpoint field provenance.slurm_job_id is absent",
        )

    def test_every_other_inner_field_remains_strict_before_inference(self):
        cases = []
        resume = _inner_resume()
        resume.completed_epoch = 199
        cases.append(("completed_epoch", resume))
        resume = _inner_resume()
        resume.optimizer_step_count -= 1
        cases.append(("optimizer_step_count", resume))
        resume = _inner_resume()
        resume.training_example_presentations -= 1
        cases.append(("training_example_presentations", resume))
        resume = _inner_resume()
        resume.epoch_records.pop()
        cases.append(("epoch_records.count", resume))
        resume = _inner_resume()
        resume.family_order_history.pop()
        cases.append(("family_order_history.count", resume))
        resume = _inner_resume()
        resume.family_order_history[0] = ("family-a",)
        cases.append(("family_order_history[0]", resume))
        for name, value in (
            ("git_commit", "f" * 40),
            ("source_tree_sha256", "f" * 64),
            ("device", "cuda:0"),
            ("encoder_arm", "typed_graph"),
            ("seed", 2027),
        ):
            resume = _inner_resume()
            resume.payload["provenance"][name] = value
            cases.append(("provenance." + name, resume))
        resume = _inner_resume()
        resume.payload["rng_states"]["torch_cuda"] = [1]
        cases.append(("rng_states.torch_cuda", resume))
        for field, resume in cases:
            with self.subTest(field=field), self.assertRaises(
                GraphEncoderError
            ) as raised:
                _validate_inner(resume)
            self.assertEqual(
                raised.exception.detail,
                "inner checkpoint field {} differs".format(field),
            )

        source = inspect.getsource(run_postmortem)
        validation = source.index("validated_models.append((model, arm, seed))")
        inference_loop = source.index("for model, arm, seed in validated_models:")
        inference = source.index("generate_autonomous_records(", inference_loop)
        self.assertLess(validation, inference_loop)
        self.assertLess(inference_loop, inference)
        for forbidden in (".backward(", "optimizer.step(", "run_ge1_training("):
            self.assertNotIn(forbidden, source)

    def test_separate_frozen_identity_and_access_contract(self):
        self.assertEqual(DIAGNOSTIC_VERSION, "GE1-STAGE6-TRAIN-GATE-POSTMORTEM-v1")
        self.assertEqual(
            ARTIFACT_VERSION,
            "GE1-STAGE6-TRAIN-GATE-POSTMORTEM-ARTIFACT-v1",
        )
        self.assertEqual(PRODUCER_JOB_ID, "3354961")
        self.assertEqual(
            PRODUCER_SOURCE_COMMIT,
            "5f4542f86756dae44435af27a6e072db6f27a8ef",
        )
        self.assertEqual(PRODUCER_EXECUTION_EVIDENCE["exit_code"], "42:0")
        self.assertEqual(PRODUCER_EXECUTION_EVIDENCE["max_rss"], "855460K")
        self.assertFalse(PRODUCER_EXECUTION_EVIDENCE["development_accessed"])
        self.assertIs(
            PRODUCER_EXECUTION_EVIDENCE["inner_checkpoint_slurm_job_id"], None
        )
        self.assertTrue(ACCESS_RECORD["authorized_narrow_train_accessed"])
        self.assertTrue(ACCESS_RECORD["train_only_autonomous_inference_performed"])
        for name in (
            "training_performed",
            "backward_pass_performed",
            "optimizer_step_performed",
            "checkpoint_modified",
            "development_accessed",
            "rr_accessed",
            "er_test_accessed",
            "iid_accessed",
            "corpus_accessed",
            "manifest_accessed",
            "model_artifact_accessed",
            "repaired_artifact_accessed",
            "scientific_repair_performed",
        ):
            self.assertIs(ACCESS_RECORD[name], False)

    def test_exact_checkpoint_hash_matrix_is_required(self):
        values = ["{}={}".format(name, "a" * 64) for name in EXPECTED_WRAPPER_NAMES]
        observed = parse_checkpoint_hashes(values)
        self.assertEqual(tuple(sorted(observed)), tuple(sorted(EXPECTED_WRAPPER_NAMES)))
        for invalid in (
            values[:-1],
            values + [values[0]],
            values[:-1] + [EXPECTED_WRAPPER_NAMES[-1] + "=" + "A" * 64],
        ):
            with self.assertRaises(GraphEncoderError):
                parse_checkpoint_hashes(invalid)

    def test_historical_source_digest_uses_commit_objects(self):
        root = Path(__file__).resolve().parents[3]
        observed = source_commit_python_sha256(root, PRODUCER_SOURCE_COMMIT)
        self.assertEqual(len(observed), 64)
        self.assertTrue(all(character in "0123456789abcdef" for character in observed))
        with self.assertRaises(GraphEncoderError):
            source_commit_python_sha256(root, "not-a-commit")

    def test_exact_train_gates_are_recomputed_and_failed_predicates_reported(self):
        records = _train_records()
        self.assertEqual(len(records), EXPECTED_FAMILY_RECORD_COUNT)
        result = analyze_postmortem(_training_runs(), records)
        self.assertEqual(result["failed_optimization_runs"], [])
        self.assertEqual(result["failed_train_ceiling_seeds"], list(FULL_SEEDS))
        self.assertEqual(result["failed_train_memory_arms"], [])
        self.assertEqual(
            result["minimal_combined_reason"]["failed_components"],
            ["per_seed_train_ceiling"],
        )
        self.assertEqual(len(result["exact_failed_producer_predicates"]), 3)
        self.assertTrue(all(
            name.startswith("train_ceiling.seed")
            for name in result["exact_failed_producer_predicates"]
        ))
        self.assertTrue(result["reproduces_stage6_train_reliability_failure"])
        self.assertFalse(result["outcome_is_artifact_validity_gate"])
        self.assertFalse(result["encoder_superiority_inferred"])
        self.assertEqual(len(result["arms_and_seeds"]), 6)
        for row in result["arms_and_seeds"]:
            self.assertIn("per_seed_shortfall_difference", row["failed_predicates"])
            self.assertTrue(row["per_seed_memory_gate_pass"])

    def test_record_matrix_tampering_is_rejected(self):
        records = _train_records()
        with self.assertRaises(GraphEncoderError):
            analyze_postmortem(_training_runs(), records[:-1])
        changed = copy.deepcopy(records)
        changed[0]["cohort"] = "development"
        with self.assertRaises(GraphEncoderError):
            analyze_postmortem(_training_runs(), changed)

    def test_artifact_finalizes_for_failed_scientific_gate_and_detects_tamper(self):
        training = _training_runs()
        families = _train_records()
        analysis = analyze_postmortem(training, families)
        resolved = {
            "diagnostic_version": DIAGNOSTIC_VERSION,
            "artifact_version": ARTIFACT_VERSION,
            "protocol_version": PROTOCOL_VERSION,
            "producer_source_commit": PRODUCER_SOURCE_COMMIT,
            "producer_source_tree_sha256": "e" * 64,
            "producer_job_id": PRODUCER_JOB_ID,
            "producer_execution_evidence": dict(PRODUCER_EXECUTION_EVIDENCE),
            "diagnostic_source": {
                "git_commit": DIAGNOSTIC_COMMIT,
                "git_branch": None,
                "detached_head": True,
                "git_dirty": False,
                "git_status_porcelain": [],
            },
            "runtime": {
                "slurm_job_id": DIAGNOSTIC_JOB,
                "python": "3.8.13",
                "pytorch": "1.11.0",
                "device": "cpu",
                "cuda_available": False,
                "cpu_threads": 1,
            },
            "checkpoint_evidence": _checkpoint_evidence(),
            "train_input_evidence": {
                "index_identity_sha256": "f" * 64,
                "observed_payload_digests_sha256": "1" * 64,
                "verification_status": "pass",
            },
            "access": dict(ACCESS_RECORD),
        }
        summary = {
            "diagnostic_version": DIAGNOSTIC_VERSION,
            "artifact_version": ARTIFACT_VERSION,
            "diagnostic_source_commit": DIAGNOSTIC_COMMIT,
            "slurm_job_id": DIAGNOSTIC_JOB,
            "producer_source_commit": PRODUCER_SOURCE_COMMIT,
            "producer_job_id": PRODUCER_JOB_ID,
            "producer_execution_evidence": dict(PRODUCER_EXECUTION_EVIDENCE),
            "training_run_count": 6,
            "train_family_record_count": len(families),
            "analysis": analysis,
            "diagnostic_completed": True,
            "outcome_controls_artifact_validity": False,
            "access": dict(ACCESS_RECORD),
        }
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            staging = parent / "artifact.incomplete-3355999"
            final = parent / "artifact"
            staging.mkdir()
            _atomic_write_json(staging / "resolved_config.json", resolved)
            _atomic_write_jsonl(staging / "train_family_records.jsonl", families)
            _atomic_write_jsonl(staging / "training_histories.jsonl", training)
            _atomic_write_json(staging / "summary.json", summary)
            finalize_artifact(staging, final)
            self.assertEqual(
                tuple(sorted(item.name for item in final.iterdir())),
                tuple(sorted(ARTIFACT_FILES)),
            )
            verification = verify_artifact(
                final,
                expected_commit=DIAGNOSTIC_COMMIT,
                expected_slurm_job_id=DIAGNOSTIC_JOB,
            )
            self.assertEqual(verification["verification_status"], "pass")
            self.assertTrue(verification["reproduces_stage6_train_reliability_failure"])
            self.assertFalse(verification["outcome_controls_artifact_validity"])
            with (final / "summary.json").open("ab") as handle:
                handle.write(b" ")
            with self.assertRaises(GraphEncoderError):
                verify_artifact(
                    final,
                    expected_commit=DIAGNOSTIC_COMMIT,
                    expected_slurm_job_id=DIAGNOSTIC_JOB,
                )

    def test_source_and_runner_pass_strict_access_audit(self):
        root = Path(__file__).resolve().parents[1]
        runner = root / "adroit" / "ge1_stage6_train_gate_postmortem_cpu.slurm"
        result = audit(root, runner)
        self.assertEqual(result["training_call_count"], 0)
        self.assertEqual(result["optimizer_step_call_count"], 0)
        self.assertEqual(result["development_loader_call_count"], 0)
        self.assertEqual(result["diagnostic_invocation_count"], 1)
        self.assertEqual(result["bind_count"], 4)
        self.assertEqual(result["protected_bind_count"], 0)


if __name__ == "__main__":
    unittest.main()
