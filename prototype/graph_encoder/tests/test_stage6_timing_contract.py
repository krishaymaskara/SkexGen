"""Device-independent contracts for ADR-0014 timing-v2."""

from __future__ import annotations

import copy
from pathlib import Path
import tempfile
import unittest

from prototype.graph_encoder.errors import GraphEncoderError
from prototype.graph_encoder.stage6_timing import (
    CONTINGENCY_FRACTION,
    SELECTION_ALGORITHM,
    TIMED_EPOCHS,
    TIMING_VERSION,
    build_timing_record,
    create_timing_artifact,
    validate_timing_evidence,
    verify_timing_artifact,
)


def runtime(device, *, gpu_name="Fixture GPU"):
    cuda = device == "cuda:0"
    return {
        "device_policy_version": "GE1-STAGE6-DEVICE-POLICY-v1",
        "execution_device": device,
        "python_version": "3.8.13",
        "torch_version": "1.11.0+cu113",
        "torch_cuda_build_version": "11.3",
        "cuda_available": True,
        "visible_cuda_device_count": 1,
        "gpu": ({
            "visible_device_identity": "0", "name": gpu_name,
            "compute_capability": [8, 0], "total_memory_bytes": 40000000000,
        } if cuda else None),
        "cudnn_version": 8200,
        "deterministic_algorithms": True,
        "tf32_matmul_allowed": False,
        "tf32_cudnn_allowed": False,
        "cublas_workspace_config": ":4096:8",
        "cpu_threads": 1,
        "cpu_host": "fixture-host",
        "cpu_processor": "fixture-cpu",
        "platform": "fixture-platform",
        "cuda_rng_preservation_required": cuda,
    }


def record(*, cpu=10.0, cuda=5.0, cpu_wall=50000.0, cuda_wall=50000.0):
    raw = {
        "cpu": {
            "flat": {"warmup": cpu * 5, "timed": [cpu * 4, cpu * 5, cpu * 3]},
            "typed_graph": {"warmup": cpu * 5, "timed": [cpu * 5, cpu * 4, cpu * 3]},
        },
        "cuda:0": {
            "flat": {"warmup": cuda * 5, "timed": [cuda * 4, cuda * 5, cuda * 3]},
            "typed_graph": {"warmup": cuda * 5, "timed": [cuda * 5, cuda * 4, cuda * 3]},
        },
    }
    return build_timing_record(
        source_commit="a" * 40, source_tree_sha256="b" * 64,
        train_index_identity_sha256="c" * 64,
        train_payload_digests_sha256="d" * 64,
        runtime_identities={"cpu": runtime("cpu"), "cuda:0": runtime("cuda:0")},
        raw_measurements=raw,
        available_wall_seconds_by_device={"cpu": cpu_wall, "cuda:0": cuda_wall},
    )


class Stage6TimingContractTests(unittest.TestCase):
    def test_exact_schema_slowest_of_three_and_warmup_excluded(self):
        value = record()
        self.assertEqual(value["version"], TIMING_VERSION)
        self.assertEqual(value["selection_algorithm"], SELECTION_ALGORITHM)
        arm = value["candidates"]["cuda:0"]["arms"]["flat"]
        self.assertFalse(arm["warmup"]["included_in_measurement"])
        self.assertEqual(len(arm["timed_measurements"]), 3)
        self.assertEqual(arm["conservative_seconds_per_epoch"], 5.0)
        self.assertEqual(TIMED_EPOCHS, 5)
        self.assertEqual(CONTINGENCY_FRACTION, 0.20)

    def test_fastest_feasible_device_retains_three_seed_priority(self):
        value = record(cpu=10.0, cuda=5.0)
        selected = validate_timing_evidence(value)
        self.assertEqual(selected["execution_device"], "cuda:0")
        self.assertEqual(selected["retained_seeds"], (2026, 2027, 2028))
        # CUDA two seeds is faster, but CPU is the only three-seed-feasible candidate.
        value = record(cpu=10.0, cuda=5.0, cpu_wall=15000.0, cuda_wall=6000.0)
        self.assertEqual(validate_timing_evidence(value)["execution_device"], "cpu")
        self.assertEqual(validate_timing_evidence(value)["retained_seeds"], (2026, 2027, 2028))

    def test_exact_tie_prefers_cpu(self):
        value = record(cpu=5.0, cuda=5.0)
        self.assertEqual(validate_timing_evidence(value)["execution_device"], "cpu")

    def test_exact_full_boundary_is_inclusive(self):
        probe = record()
        boundary = probe["candidates"]["cuda:0"][
            "projected_three_seed_seconds_with_contingency"
        ]
        value = record(cpu_wall=boundary, cuda_wall=boundary)
        self.assertEqual(validate_timing_evidence(value)["retained_seeds"], (2026, 2027, 2028))

    def test_two_seed_fallback_and_exact_boundary(self):
        probe = record()
        two = probe["candidates"]["cuda:0"][
            "projected_two_seed_seconds_with_contingency"
        ]
        value = record(cpu_wall=1.0, cuda_wall=two)
        selected = validate_timing_evidence(value)
        self.assertEqual(selected["retained_seeds"], (2026, 2027))
        self.assertTrue(selected["fallback_invoked"])
        self.assertTrue(selected["fallback_reason"])

    def test_neither_feasible_fails_closed(self):
        with self.assertRaisesRegex(GraphEncoderError, "stage6_resource_infeasible"):
            record(cpu_wall=1.0, cuda_wall=1.0)

    def test_v1_malformed_nonfinite_negative_and_outcome_informed_fail(self):
        with self.assertRaises(GraphEncoderError):
            validate_timing_evidence({"version": "GE1-STAGE6-STRUCTURE-ONLY-TIMING-v1"})
        for mutation in ("negative", "nonfinite", "outcome"):
            value = record()
            if mutation == "negative":
                value["candidates"]["cpu"]["arms"]["flat"][
                    "timed_measurements"
                ][0]["duration_seconds"] = -1.0
            elif mutation == "nonfinite":
                value["candidates"]["cpu"]["available_wall_seconds"] = float("inf")
            else:
                value["observed_results_used"] = True
            with self.assertRaises(GraphEncoderError):
                validate_timing_evidence(value)

    def test_raw_projection_contingency_and_hardware_tampering_fail(self):
        value = record()
        for path in ("projection", "hardware", "measurement"):
            broken = copy.deepcopy(value)
            if path == "projection":
                broken["candidates"]["cuda:0"][
                    "projected_three_seed_seconds_with_contingency"
                ] += 1.0
            elif path == "hardware":
                broken["selected_timing_hardware_identity"]["gpu"]["name"] = "other"
            else:
                broken["candidates"]["cuda:0"]["arms"]["flat"][
                    "conservative_seconds_per_epoch"
                ] += 1.0
            with self.assertRaises(GraphEncoderError):
                validate_timing_evidence(broken)

    def test_required_device_rejects_silent_override(self):
        value = record()
        self.assertEqual(value["selected_device"], "cuda:0")
        with self.assertRaisesRegex(GraphEncoderError, "stage6_timing_device_mismatch"):
            validate_timing_evidence(value, required_device="cpu")

    def test_atomic_artifact_is_canonical_checksummed_and_tamper_evident(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "timing"
            result = create_timing_artifact(record(), root, job_id="123")
            self.assertEqual(result["verification_status"], "pass")
            self.assertEqual(verify_timing_artifact(root)["artifact_sha256"], result["artifact_sha256"])
            (root / "timing.json").write_text("{}\n", encoding="utf-8")
            with self.assertRaises(GraphEncoderError):
                verify_timing_artifact(root)

    def test_timing_schema_contains_no_scientific_outcomes_or_development(self):
        value = record()
        text = str(value).lower()
        for forbidden in ("accuracy", "prediction", "model_quality"):
            self.assertNotIn(forbidden, text)
        self.assertFalse(value["development_accessed"])
        self.assertTrue(value["train_payload_accessed"])


if __name__ == "__main__":
    unittest.main()
