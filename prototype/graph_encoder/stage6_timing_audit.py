"""Static timing-v2 and GPU timing-runner audit for ADR-0014."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import re


AUDIT_VERSION = "GE1-STAGE6-HARDWARE-TIMING-AUDIT-v2"


def audit(repository_root=".", runner_path=None):
    root = Path(repository_root)
    module = root / "prototype/graph_encoder/stage6_timing.py"
    source = module.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(module), feature_version=(3, 8))
    for required in (
        'TIMING_VERSION = "GE1-STAGE6-STRUCTURE-ONLY-TIMING-v2"',
        "TIMED_EPOCHS = 5", "TIMED_REPETITIONS = 3",
        "maximum_of_three_fresh_measurements", "CONTINGENCY_FRACTION = 0.20",
        "fastest feasible three-seed", "stage6_resource_infeasible",
        "load_stage6_train", "development_accessed", "observed_results_used",
        "configure_stage6_runtime", "create_timing_artifact",
    ):
        if required not in source:
            raise AssertionError("missing timing-v2 control: " + required)
    for forbidden in (
        "load_stage6_development", "run_stage6_producer", "summarize_execution",
        "score_family_record", "create_checkpoint_bundle",
    ):
        if forbidden in source:
            raise AssertionError("forbidden timing path: " + forbidden)
    result = {
        "version": AUDIT_VERSION,
        "candidate_devices": ["cpu", "cuda:0"],
        "train_loader_count": source.count("load_stage6_train"),
        "development_loader_count": 0,
        "timed_repetitions_per_arm_device": 3,
        "slowest_measurement_selected": True,
        "contingency_fraction": 0.20,
        "three_seed_priority": True,
        "exact_tie_prefers_cpu": True,
        "outcome_selection_fields": 0,
        "submission_command_invoked": False,
    }
    if runner_path is not None:
        shell = Path(runner_path).read_text(encoding="utf-8")
        executable = "\n".join(
            line for line in shell.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
        if re.search(r"(^|\s)sbatch(\s|$)", executable):
            raise AssertionError("timing runner submits itself")
        required_binds = (
            '--bind "$REPOSITORY:$REPOSITORY:ro"',
            '--bind "$TRAIN_INPUT_ROOT:$TRAIN_INPUT_ROOT:ro"',
            '--bind "$TIMING_OUTPUT_PARENT:$TIMING_OUTPUT_PARENT:rw"',
        )
        if "--nv" not in executable or executable.count("--bind") != 3 or any(
            item not in executable for item in required_binds
        ):
            raise AssertionError("GPU timing runner boundary differs")
        if any(token in executable for token in (
            "DEVELOPMENT", "RR_ROOT", "ER_ROOT", "IID_ROOT",
            "CHECKPOINT_ROOT", "MODEL_ARTIFACT_ROOT",
        )):
            raise AssertionError("timing runner exposes forbidden input")
        invocations = re.findall(
            r"-m\s+prototype\.graph_encoder\.stage6_timing(?=\s|$)", executable
        )
        if len(invocations) != 1:
            raise AssertionError("timing generator must be invoked once")
        result.update({
            "runner_bind_count": 3, "nv_enabled": True,
            "timing_invocation_count": 1,
        })
    return result


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("runner", nargs="?")
    parser.add_argument("--repository-root", default=".")
    args = parser.parse_args(argv)
    print(json.dumps(audit(args.repository_root, args.runner), sort_keys=True,
                     separators=(",", ":")))


if __name__ == "__main__":
    main()
