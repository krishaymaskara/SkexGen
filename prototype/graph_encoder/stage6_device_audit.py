"""Audit opt-in CUDA plumbing and unchanged CPU defaults."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import re


AUDIT_VERSION = "GE1-STAGE6-DEVICE-AUDIT-v1"


def _default(path, function, parameter):
    tree = ast.parse(Path(path).read_text(encoding="utf-8"), filename=str(path),
                     feature_version=(3, 8))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function:
            for arg, default in zip(node.args.kwonlyargs, node.args.kw_defaults):
                if arg.arg == parameter:
                    return ast.literal_eval(default)
            positional = node.args.args
            for arg, default in zip(positional[-len(node.args.defaults):], node.args.defaults):
                if arg.arg == parameter:
                    return ast.literal_eval(default)
    raise AssertionError("missing default {}.{}".format(function, parameter))


def audit(repository_root=".", timing_runner=None, gpu_runner=None, cpu_runner=None):
    root = Path(repository_root)
    training = root / "prototype/graph_encoder/training.py"
    autonomous = root / "prototype/graph_encoder/autonomous.py"
    producer = root / "prototype/graph_encoder/stage6_structure_only_producer.py"
    if _default(training, "run_ge1_training", "execution_device") != "cpu":
        raise AssertionError("shared training CPU default changed")
    if _default(autonomous, "autonomous_input_from_paired", "execution_device") != "cpu":
        raise AssertionError("autonomous input CPU default changed")
    if _default(autonomous, "run_autonomous_evaluation", "execution_device") != "cpu":
        raise AssertionError("autonomous evaluation CPU default changed")
    source = producer.read_text(encoding="utf-8")
    for required in (
        "required_execution_device", "configure_stage6_runtime",
        "require_timing_hardware", ".to(execution_device)",
        "execution_device=execution_device", "cuda_peak_memory_bytes",
    ):
        if required not in source:
            raise AssertionError("missing Stage 6 device control: " + required)
    result = {
        "version": AUDIT_VERSION,
        "shared_training_default": "cpu",
        "shared_autonomous_default": "cpu",
        "supported_devices": ["cpu", "cuda:0"],
        "cuda_opt_in": True,
        "selected_device_overridable": False,
        "scientific_arithmetic_changed": False,
        "finalizer_tensor_independent": True,
    }
    for label, path, device, nv, binds in (
        ("timing", timing_runner, None, True, 3),
        ("gpu_producer", gpu_runner, "cuda:0", True, 4),
        ("cpu_producer", cpu_runner, "cpu", False, 4),
    ):
        if path is None:
            continue
        shell = Path(path).read_text(encoding="utf-8")
        executable = "\n".join(
            line for line in shell.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
        if executable.count("--bind") != binds or ("--nv" in executable) is not nv:
            raise AssertionError(label + " runner device/bind boundary differs")
        if re.search(r"(^|\s)sbatch(\s|$)", executable):
            raise AssertionError(label + " runner submits itself")
        if device is not None and "--require-selected-device " + device not in executable:
            raise AssertionError(label + " runner does not bind timing selection")
    return result


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", default=".")
    parser.add_argument("--timing-runner")
    parser.add_argument("--gpu-runner")
    parser.add_argument("--cpu-runner")
    args = parser.parse_args(argv)
    print(json.dumps(audit(
        args.repository_root, args.timing_runner, args.gpu_runner, args.cpu_runner
    ), sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
