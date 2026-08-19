"""Pure contracts for the ADR-0014 train-only zero-memory diagnostic."""

from __future__ import annotations

import copy
import inspect
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from prototype.graph_encoder.errors import GraphEncoderError
from prototype.graph_encoder.pilot import _atomic_write_json, _atomic_write_jsonl
from prototype.graph_encoder.stage6_structure_only import ARMS, FULL_SEEDS, score_family_record
from prototype.graph_encoder.stage6_zero_memory_audit import audit
from prototype.graph_encoder.stage6_zero_memory_diagnostic import (
    ACCESS_RECORD,
    ARTIFACT_FILES,
    ARTIFACT_VERSION,
    DIAGNOSTIC_VERSION,
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_WRAPPER_NAMES,
    EXPECTED_ZERO_RECORD_COUNT,
    P_ZERO,
    PRIOR_POSTMORTEM_COMMIT,
    PRIOR_POSTMORTEM_FILE_SHA256,
    PRIOR_POSTMORTEM_JOB_ID,
    PRIOR_POSTMORTEM_SHA256SUMS_SHA256,
    PRODUCER_JOB_ID,
    PRODUCER_SOURCE_COMMIT,
    RECORD_VERSION,
    _prior_true_baseline,
    analyze_zero_memory,
    complete_zero_memory_records,
    finalize_artifact,
    parse_checkpoint_paths,
    score_zero_family_record,
    validate_zero_memory_records,
    verify_artifact,
    zero_memory_tensor,
)
from prototype.graph_encoder.tests.test_stage6_structure_only_contract import (
    synthetic_payload,
)


DIAGNOSTIC_COMMIT = "d" * 40
DIAGNOSTIC_JOB = "3355998"


class _FakeTensor:
    def __init__(self, values, shape=(1, 2), dtype="float32", device="cpu"):
        self.values = values
        self.shape = shape
        self.dtype = dtype
        self.device = device


class _FakeTorch:
    @staticmethod
    def zeros_like(memory):
        return _FakeTensor(
            [0 for unused in memory.values],
            shape=memory.shape,
            dtype=memory.dtype,
            device=memory.device,
        )


def _prior_evidence():
    return {
        "diagnostic_version": "GE1-STAGE6-TRAIN-GATE-POSTMORTEM-v1",
        "artifact_version": "GE1-STAGE6-TRAIN-GATE-POSTMORTEM-ARTIFACT-v1",
        "source_commit": PRIOR_POSTMORTEM_COMMIT,
        "slurm_job_id": PRIOR_POSTMORTEM_JOB_ID,
        "producer_source_commit": PRODUCER_SOURCE_COMMIT,
        "producer_job_id": PRODUCER_JOB_ID,
        "directory_name": (
            "ge1-stage6-train-gate-postmortem-{}-{}".format(
                PRIOR_POSTMORTEM_COMMIT, PRIOR_POSTMORTEM_JOB_ID
            )
        ),
        "sha256sums_sha256": PRIOR_POSTMORTEM_SHA256SUMS_SHA256,
        "file_sha256": dict(PRIOR_POSTMORTEM_FILE_SHA256),
        "train_input_evidence": {"verification_status": "pass"},
        "checkpoint_wrapper_sha256": dict(EXPECTED_CHECKPOINT_SHA256),
        "mean_memory_reference": {
            "source": "authenticated job-3355134 summary",
            "aggregate_by_arm": {"flat": 0.75, "typed_graph": 0.8},
            "per_seed_and_arm": {
                str(seed): {"flat": 0.75, "typed_graph": 0.8}
                for seed in FULL_SEEDS
            },
            "threshold": None,
            "validity_gate_in_this_diagnostic": False,
        },
        "verification_status": "pass",
        "true_record_count": 2442,
    }


def _zero_fixture():
    true_records = [
        copy.deepcopy(row) for row in synthetic_payload()["family_records"]
        if row["cohort"] == "train" and row["condition"] == "P_true"
    ]
    baseline = _prior_true_baseline(true_records)
    generated = []
    for true in true_records:
        zero = copy.deepcopy(true)
        zero["condition"] = P_ZERO
        zero["memory_source_family_id"] = "zero_memory"
        generated.append({
            "family_record": zero,
            "intervention_evidence": {
                "torch_zeros_like_used": True,
                "memory_shape": [1, 2, 32],
                "memory_dtype": "torch.float32",
                "memory_device": "cpu",
                "memory_element_count": 64,
                "memory_nonzero_count": 0,
                "recipient_node_count": zero["target_structural_prefixes"][-1][
                    "requested_node_count"
                ],
                "batch_identity": zero["batch_identity"],
                "only_memory_values_replaced": True,
                "recipient_node_count_preserved": True,
                "decoder_non_memory_inputs_preserved": True,
                "encoder_input_mutated": False,
            },
        })
    records = complete_zero_memory_records(generated, baseline)
    return baseline, records


def _checkpoint_evidence():
    result = []
    for seed in FULL_SEEDS:
        for arm in ARMS:
            name = "stage6-{}-seed{}.pt".format(arm, seed)
            result.append({
                "wrapper_name": name,
                "stage6_wrapper_sha256": EXPECTED_CHECKPOINT_SHA256[name],
                "identity": {
                    "arm": arm,
                    "seed": seed,
                    "model_config": {},
                    "parameter_count": 1,
                    "training_partition_identity": {},
                },
                "strict_recovery_passed": True,
                "checkpoint_modified": False,
            })
    return result


class Stage6ZeroMemoryContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.baseline, cls.records = _zero_fixture()

    def test_separate_identity_exact_inputs_and_no_threshold(self):
        self.assertEqual(DIAGNOSTIC_VERSION, "GE1-STAGE6-ZERO-MEMORY-DIAGNOSTIC-v1")
        self.assertEqual(ARTIFACT_VERSION, "GE1-STAGE6-ZERO-MEMORY-ARTIFACT-v1")
        self.assertEqual(RECORD_VERSION, "GE1-STAGE6-ZERO-MEMORY-RECORD-v1")
        self.assertEqual(PRODUCER_JOB_ID, "3354961")
        self.assertEqual(
            PRODUCER_SOURCE_COMMIT,
            "5f4542f86756dae44435af27a6e072db6f27a8ef",
        )
        self.assertEqual(PRIOR_POSTMORTEM_JOB_ID, "3355134")
        self.assertEqual(
            PRIOR_POSTMORTEM_COMMIT,
            "ddf9617fb124fbc18f38302943f0c982019bfee9",
        )
        self.assertEqual(len(EXPECTED_CHECKPOINT_SHA256), 6)
        self.assertEqual(len(PRIOR_POSTMORTEM_FILE_SHA256), 5)
        analysis = analyze_zero_memory(
            self.records, self.baseline, _prior_evidence()["mean_memory_reference"]
        )
        self.assertFalse(analysis["new_pass_fail_threshold_created"])
        self.assertFalse(analysis["scientific_outcome_controls_artifact_validity"])
        self.assertFalse(analysis["encoder_superiority_inferred"])
        self.assertFalse(analysis["development_conclusion_available"])

    def test_exact_six_checkpoint_paths_are_required(self):
        parent = Path(
            "/safe/ge1-stage6-structure-only-producer-{}-{}.work.incomplete-{}".format(
                PRODUCER_SOURCE_COMMIT, PRODUCER_JOB_ID, PRODUCER_JOB_ID
            )
        )
        values = ["{}={}".format(name, parent / name) for name in EXPECTED_WRAPPER_NAMES]
        observed = parse_checkpoint_paths(values)
        self.assertEqual(tuple(sorted(observed)), EXPECTED_WRAPPER_NAMES)
        for invalid in (
            values[:-1],
            values + [values[0]],
            values[:-1] + ["unexpected.pt=/safe/unexpected.pt"],
            [value.replace(".work.incomplete-3354961", ".work") for value in values],
        ):
            with self.subTest(invalid=invalid), self.assertRaises(GraphEncoderError):
                parse_checkpoint_paths(invalid)

    def test_zeros_like_preserves_shape_dtype_device_and_only_values_change(self):
        memory = _FakeTensor([3.0, -2.0], shape=(1, 2, 1), dtype="float64")
        zero = zero_memory_tensor(memory, torch_module=_FakeTorch)
        self.assertEqual(zero.values, [0, 0])
        self.assertEqual(zero.shape, memory.shape)
        self.assertEqual(zero.dtype, memory.dtype)
        self.assertEqual(zero.device, memory.device)
        self.assertEqual(memory.values, [3.0, -2.0])

    def test_zero_condition_reuses_unchanged_structural_score(self):
        row = self.records[0]
        zero = row["zero_memory_family_record"]
        compatible = copy.deepcopy(zero)
        compatible["condition"] = "P_true"
        compatible["memory_source_family_id"] = compatible["family_id"]
        self.assertEqual(
            score_zero_family_record(zero), score_family_record(compatible)["score"]
        )
        changed = copy.deepcopy(zero)
        changed["condition"] = "P_mean"
        with self.assertRaises(GraphEncoderError):
            score_zero_family_record(changed)

    def test_record_coverage_aggregation_and_family_balancing_are_exact(self):
        self.assertEqual(len(self.records), EXPECTED_ZERO_RECORD_COUNT)
        self.assertTrue(validate_zero_memory_records(self.records, self.baseline))
        analysis = analyze_zero_memory(
            self.records, self.baseline, _prior_evidence()["mean_memory_reference"]
        )
        self.assertEqual(len(analysis["per_seed_and_arm"]), 6)
        self.assertEqual(len(analysis["family_contributions"]), 814)
        self.assertEqual(analysis["aggregation"]["records_per_arm"], 1221)
        for arm in ARMS:
            aggregate = analysis["aggregate_by_arm"][arm]
            self.assertEqual(aggregate["support"], 1221)
            self.assertEqual(aggregate["P_true"], aggregate["P_zero"])
            self.assertEqual(aggregate["zero_over_true"], 1.0)
            secondary = analysis["family_balanced_secondary"][arm]
            self.assertEqual(secondary["template_support"], 4)
            self.assertIsNone(secondary["threshold"])
            self.assertFalse(secondary["validity_gate"])

    def test_arbitrary_condition_duplicate_missing_and_tampering_fail(self):
        cases = []
        changed = copy.deepcopy(self.records)
        changed[0]["zero_memory_family_record"]["condition"] = "P_noise"
        cases.append(changed)
        cases.append(self.records[:-1])
        changed = copy.deepcopy(self.records)
        changed.append(copy.deepcopy(changed[0]))
        cases.append(changed)
        changed = copy.deepcopy(self.records)
        changed[0]["zero_memory_score"] = 0.25
        cases.append(changed)
        changed = copy.deepcopy(self.records)
        changed[0]["intervention_evidence"]["memory_nonzero_count"] = 1
        cases.append(changed)
        for value in cases:
            with self.subTest(length=len(value)), self.assertRaises(GraphEncoderError):
                validate_zero_memory_records(value, self.baseline)

    def test_prior_authentication_precedes_train_and_checkpoint_access(self):
        from prototype.graph_encoder.stage6_zero_memory_diagnostic import (
            authenticate_prior_postmortem,
            run_zero_memory_diagnostic,
        )
        authentication_source = inspect.getsource(authenticate_prior_postmortem)
        first_hash_check = authentication_source.index("_verify_exact_prior_files(")
        verifier = authentication_source.index("verify_postmortem_artifact(")
        baseline = authentication_source.index("_prior_true_baseline(")
        second_hash_check = authentication_source.rindex("_verify_exact_prior_files(")
        self.assertLess(first_hash_check, verifier)
        self.assertLess(verifier, baseline)
        self.assertLess(baseline, second_hash_check)
        source = inspect.getsource(run_zero_memory_diagnostic)
        prior = source.index("authenticate_prior_postmortem(")
        loader = source.index("load_stage6_train(")
        checkpoint = source.index("_load_retained_checkpoint(")
        inference = source.index("generate_zero_memory_records(")
        self.assertLess(prior, loader)
        self.assertLess(prior, checkpoint)
        self.assertLess(checkpoint, inference)
        for forbidden in (".backward(", "optimizer.step(", "run_ge1_training("):
            self.assertNotIn(forbidden, source)

    def test_artifact_finalizes_independently_of_scientific_outcome(self):
        prior_evidence = _prior_evidence()
        analysis = analyze_zero_memory(
            self.records, self.baseline, _prior_evidence()["mean_memory_reference"]
        )
        resolved = {
            "diagnostic_version": DIAGNOSTIC_VERSION,
            "artifact_version": ARTIFACT_VERSION,
            "protocol_version": "GE1-STAGE6-STRUCTURE-ONLY-COMPARISON-v1",
            "producer_source_commit": PRODUCER_SOURCE_COMMIT,
            "producer_source_tree_sha256": "e" * 64,
            "producer_job_id": PRODUCER_JOB_ID,
            "prior_postmortem_evidence": prior_evidence,
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
            "train_input_evidence": {"verification_status": "pass"},
            "access": dict(ACCESS_RECORD),
        }
        summary = {
            "diagnostic_version": DIAGNOSTIC_VERSION,
            "artifact_version": ARTIFACT_VERSION,
            "diagnostic_source_commit": DIAGNOSTIC_COMMIT,
            "slurm_job_id": DIAGNOSTIC_JOB,
            "producer_source_commit": PRODUCER_SOURCE_COMMIT,
            "producer_job_id": PRODUCER_JOB_ID,
            "prior_postmortem_evidence": prior_evidence,
            "zero_memory_record_count": len(self.records),
            "analysis": analysis,
            "diagnostic_completed": True,
            "outcome_controls_artifact_validity": False,
            "access": dict(ACCESS_RECORD),
        }
        authentication = mock.patch(
            "prototype.graph_encoder.stage6_zero_memory_diagnostic."
            "authenticate_prior_postmortem",
            return_value=(prior_evidence, self.baseline),
        )
        checkpoint_validation = mock.patch(
            "prototype.graph_encoder.stage6_zero_memory_diagnostic."
            "_validate_checkpoint_identity",
            return_value=True,
        )
        with tempfile.TemporaryDirectory() as directory, authentication, checkpoint_validation:
            parent = Path(directory)
            staging = parent / "artifact.incomplete-3355998"
            final = parent / "artifact"
            staging.mkdir()
            _atomic_write_json(staging / "resolved_config.json", resolved)
            _atomic_write_json(staging / "summary.json", summary)
            _atomic_write_jsonl(staging / "zero_memory_records.jsonl", self.records)
            finalize_artifact(
                staging, final, prior_postmortem_artifact=parent / "prior"
            )
            self.assertEqual(
                tuple(sorted(item.name for item in final.iterdir())),
                tuple(sorted(ARTIFACT_FILES)),
            )
            result = verify_artifact(
                final,
                expected_commit=DIAGNOSTIC_COMMIT,
                expected_slurm_job_id=DIAGNOSTIC_JOB,
                prior_postmortem_artifact=parent / "prior",
            )
            self.assertEqual(result["verification_status"], "pass")
            self.assertFalse(result["outcome_controls_artifact_validity"])
            with (final / "summary.json").open("ab") as handle:
                handle.write(b" ")
            with self.assertRaises(GraphEncoderError):
                verify_artifact(
                    final,
                    expected_commit=DIAGNOSTIC_COMMIT,
                    expected_slurm_job_id=DIAGNOSTIC_JOB,
                    prior_postmortem_artifact=parent / "prior",
                )

    def test_source_and_runner_pass_strict_access_audit(self):
        root = Path(__file__).resolve().parents[1]
        runner = root / "adroit" / "ge1_stage6_zero_memory_cpu.slurm"
        result = audit(root, runner)
        self.assertEqual(result["torch_zeros_like_call_count"], 1)
        self.assertEqual(result["training_call_count"], 0)
        self.assertEqual(result["optimizer_step_call_count"], 0)
        self.assertEqual(result["development_loader_call_count"], 0)
        self.assertEqual(result["diagnostic_invocation_count"], 1)
        self.assertEqual(result["exact_checkpoint_file_bind_count"], 6)
        self.assertEqual(result["bind_count"], 10)
        self.assertEqual(result["protected_bind_count"], 0)


if __name__ == "__main__":
    unittest.main()
