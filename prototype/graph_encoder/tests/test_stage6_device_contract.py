"""Pure CPU-default and runner contracts for ADR-0014 device support."""

from __future__ import annotations

import inspect
from pathlib import Path
import unittest

from prototype.graph_encoder.errors import GraphEncoderError
from prototype.graph_encoder.stage6_device import validate_execution_device
from prototype.graph_encoder.stage6_device_audit import audit


class Stage6DeviceContractTests(unittest.TestCase):
    def test_supported_devices_are_exact_and_invalid_values_do_not_fallback(self):
        self.assertEqual(validate_execution_device("cpu"), "cpu")
        self.assertEqual(validate_execution_device("cuda:0"), "cuda:0")
        for invalid in ("cuda", "cuda:1", None, "auto"):
            with self.assertRaises(GraphEncoderError):
                validate_execution_device(invalid)

    def test_shared_training_and_autonomous_defaults_remain_cpu(self):
        from prototype.graph_encoder.autonomous import (
            autonomous_input_from_paired, run_autonomous_evaluation,
        )
        from prototype.graph_encoder.training import run_ge1_training
        self.assertEqual(
            inspect.signature(run_ge1_training).parameters[
                "execution_device"
            ].default,
            "cpu",
        )
        self.assertEqual(
            inspect.signature(autonomous_input_from_paired).parameters[
                "execution_device"
            ].default,
            "cpu",
        )
        self.assertEqual(
            inspect.signature(run_autonomous_evaluation).parameters[
                "execution_device"
            ].default,
            "cpu",
        )

    def test_cpu_gpu_timing_and_producer_runners_bind_device_selection(self):
        repository = Path(__file__).resolve().parents[3]
        result = audit(
            repository,
            repository / "prototype/graph_encoder/adroit/ge1_stage6_hardware_timing_gpu.slurm",
            repository / "prototype/graph_encoder/adroit/ge1_stage6_structure_only_producer_gpu.slurm",
            repository / "prototype/graph_encoder/adroit/ge1_stage6_structure_only_producer.slurm",
        )
        self.assertTrue(result["cuda_opt_in"])
        self.assertEqual(result["shared_training_default"], "cpu")
        self.assertFalse(result["selected_device_overridable"])


if __name__ == "__main__":
    unittest.main()
