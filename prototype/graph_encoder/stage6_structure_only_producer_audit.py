"""Static lifecycle and access audit for the Stage 6 producer."""

from __future__ import annotations

import ast
from pathlib import Path
import re


AUDIT_VERSION = "GE1-STAGE6-STRUCTURE-ONLY-PRODUCER-AUDIT-v1"


def audit(repository_root, runner_path):
    root = Path(repository_root)
    producer = root / "prototype/graph_encoder/stage6_structure_only_producer.py"
    runner = Path(runner_path)
    tree = ast.parse(producer.read_text(encoding="utf-8"), filename=str(producer),
                     feature_version=(3, 8))
    source = producer.read_text(encoding="utf-8")
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    loader_names = {
        getattr(node.func, "id", None) for node in calls
        if getattr(node.func, "id", None) in {
            "load_stage6_train", "load_stage6_development"
        }
    }
    if loader_names != {"load_stage6_train", "load_stage6_development"}:
        raise AssertionError("producer must contain exactly the two authorized loaders")
    if "load_train(" in source or "load_development(" in source:
        raise AssertionError("generic full-corpus loader appears in producer")
    for required in (
        'shuffle_scope="batch"', "_grammar_valid(",
        "create_checkpoint_bundle(", "verify_checkpoint_bundle(",
        "validate_scored_alignment", "source_evidence", "input_evidence",
        "from .stage6_timing import TIMING_VERSION", "execution_device",
        "timing_hardware_identity", "cuda_rng_preserved",
    ):
        if required not in source and required not in (
            root / "prototype/graph_encoder/stage6_structure_only.py"
        ).read_text(encoding="utf-8"):
            raise AssertionError("missing governed producer boundary: " + required)
    timing_index = source.index("validate_timing_evidence(", source.index("def run_stage6_producer"))
    source_index = source.index("_source_identity(", source.index("def run_stage6_producer"))
    train_index = source.index("load_stage6_train(train_index, train_root)", source.index("def run_stage6_producer"))
    if not timing_index < source_index < train_index:
        raise AssertionError("timing and hash gates must precede scientific loaders")

    shell = runner.read_text(encoding="utf-8")
    executable = "\n".join(
        line for line in shell.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    if re.search(r"(^|\s)sbatch(\s|$)", executable):
        raise AssertionError("runner must not submit itself")
    required_binds = (
        '--bind "$REPOSITORY:$REPOSITORY:ro"',
        '--bind "$TRAIN_INPUT_ROOT:$TRAIN_INPUT_ROOT:ro"',
        '--bind "$DEVELOPMENT_INPUT_ROOT:$DEVELOPMENT_INPUT_ROOT:ro"',
        '--bind "$RUN_PARENT:$RUN_PARENT:rw"',
    )
    if executable.count("--bind") != 4 or any(item not in executable for item in required_binds):
        raise AssertionError("producer runner bind set differs")
    gpu_runner = runner.name.endswith("_gpu.slurm")
    if gpu_runner:
        if "--nv" not in executable or "--require-selected-device cuda:0" not in executable:
            raise AssertionError("GPU producer device boundary differs")
    elif "--nv" in executable or "--require-selected-device cpu" not in executable:
        raise AssertionError("CPU producer device boundary differs")
    invocation = re.findall(
        r"-m\s+prototype\.graph_encoder\.stage6_structure_only_producer(?=\s|$)",
        executable,
    )
    if len(invocation) != 1:
        raise AssertionError("producer must be invoked exactly once")
    for forbidden in (
        "RR_ROOT", "ER_ROOT", "IID_ROOT", "HISTORY_DEPTH_ROOT",
        "GEOMETRY_EXTRAPOLATION_ROOT", "CORPUS_ROOT", "CHECKPOINT_ROOT",
        "MODEL_ARTIFACT_ROOT", "REPAIRED_ARTIFACT_ROOT",
    ):
        if forbidden in executable:
            raise AssertionError("runner exposes a forbidden input: " + forbidden)
    for event in (
        "terminal_success", "scientific_failure", "infrastructure_failure",
        "terminal_timeout",
    ):
        if event not in executable:
            raise AssertionError("missing telemetry: " + event)
    return {
        "version": AUDIT_VERSION,
        "producer_invocation_count": 1,
        "bind_count": 4,
        "authorized_scientific_loaders": sorted(loader_names),
        "prospective_gates_precede_loaders": True,
        "protected_bind_count": 0,
        "submission_command_invoked": False,
        "selected_device_bound_to_timing": True,
        "gpu_passthrough": gpu_runner,
    }


def main(argv=None):
    import argparse
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument("runner")
    parser.add_argument("--repository-root", default=".")
    args = parser.parse_args(argv)
    print(json.dumps(audit(args.repository_root, args.runner), sort_keys=True,
                     separators=(",", ":")))


if __name__ == "__main__":
    main()
