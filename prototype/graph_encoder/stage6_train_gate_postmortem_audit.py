"""Structural and access audit for the ADR-0014 train-gate postmortem."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import re
import sys


AUDIT_VERSION = "GE1-STAGE6-TRAIN-GATE-POSTMORTEM-AUDIT-v1"
DIAGNOSTIC_MODULE = "prototype.graph_encoder.stage6_train_gate_postmortem"


def _shell_source(source):
    return "\n".join(
        line for line in source.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )


def audit(package_root, runner_path):
    root = Path(package_root)
    diagnostic = root / "stage6_train_gate_postmortem.py"
    runner = Path(runner_path)
    source = diagnostic.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(diagnostic), feature_version=(3, 8))
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    required_imports = {
        "generate_autonomous_records",
        "load_stage6_checkpoint",
        "load_stage6_train",
        "optimization_reliability",
        "score_family_record",
        "structure_memory_gate_for_cohort",
        "validate_checkpoint_identity",
    }
    if not required_imports.issubset(imports):
        raise AssertionError("postmortem omits governed implementation reuse")
    forbidden_imports = {
        "load_stage6_development",
        "load_train",
        "load_development",
        "load_partition_physical_examples",
        "run_ge1_training",
    }
    if imports & forbidden_imports:
        raise AssertionError("postmortem imports a prohibited entry point")
    forbidden_source = (
        ".backward(", ".step(", "optimizer.step", "run_ge1_training(",
        "load_stage6_development(", "--development", "DEVELOPMENT_INPUT",
        "RR_ROOT", "ER_ROOT", "IID_ROOT", "CORPUS_ROOT", "MANIFEST",
        "MODEL_ARTIFACT", "REPAIRED_ARTIFACT",
    )
    if any(value in source for value in forbidden_source):
        raise AssertionError("postmortem source contains prohibited work")
    if source.count("generate_autonomous_records(") != 1:
        raise AssertionError("postmortem autonomous path differs")
    if source.count("load_stage6_train(") != 1:
        raise AssertionError("postmortem train-loader path differs")

    runner_source = runner.read_text(encoding="utf-8")
    shell = _shell_source(runner_source)
    if re.search(r"(^|\s)sbatch(\s|$)", shell):
        raise AssertionError("runner submits a Slurm job")
    required_binds = (
        '--bind "$REPOSITORY:$REPOSITORY:ro"',
        '--bind "$RETAINED_WORK_ROOT:$RETAINED_WORK_ROOT:ro"',
        '--bind "$TRAIN_INPUT_ROOT:$TRAIN_INPUT_ROOT:ro"',
        '--bind "$RUN_PARENT:$RUN_PARENT:rw"',
    )
    if shell.count("--bind") != 4 or any(
        value not in shell for value in required_binds
    ):
        raise AssertionError("postmortem bind set differs")
    if "--nv" in shell:
        raise AssertionError("postmortem runner must remain CPU-only")
    invocations = re.findall(
        r"-m\s+" + re.escape(DIAGNOSTIC_MODULE) + r"(?=\s|$)", shell
    )
    if len(invocations) != 1:
        raise AssertionError("postmortem diagnostic must run exactly once")
    forbidden_shell = (
        "DEVELOPMENT_INPUT", "RR_ROOT", "ER_ROOT", "IID_ROOT",
        "HISTORY_DEPTH_ROOT", "GEOMETRY_EXTRAPOLATION_ROOT", "CORPUS_ROOT",
        "MANIFEST", "MODEL_ARTIFACT", "REPAIRED_ARTIFACT",
        "run_ge1_training", "scientific_training.py",
    )
    if any(value in shell for value in forbidden_shell):
        raise AssertionError("postmortem runner exposes a prohibited resource")
    for required in (
        "git symbolic-ref -q --short HEAD",
        "git status --porcelain=v1 --untracked-files=all",
        "git rev-parse --show-superproject-working-tree",
        "EXPECTED_RUNNER_SHA256",
        "stage6_train_gate_postmortem_terminal_success",
        "stage6_train_gate_postmortem_terminal_failure",
        "stage6_train_gate_postmortem_terminal_timeout",
    ):
        if required not in shell:
            raise AssertionError("postmortem runner omits " + required)
    return {
        "version": AUDIT_VERSION,
        "source_ast_audit": "pass",
        "unchanged_scientific_function_reuse": True,
        "training_call_count": 0,
        "backward_call_count": 0,
        "optimizer_step_call_count": 0,
        "development_loader_call_count": 0,
        "train_loader_call_count": 1,
        "diagnostic_invocation_count": 1,
        "bind_count": 4,
        "protected_bind_count": 0,
        "submission_command_invoked": False,
    }


def main(argv=None):
    values = list(sys.argv[1:] if argv is None else argv)
    if len(values) != 1:
        raise SystemExit(
            "usage: python -m prototype.graph_encoder."
            "stage6_train_gate_postmortem_audit RUNNER"
        )
    root = Path(__file__).resolve().parent
    result = audit(root, values[0])
    print(json.dumps(result, sort_keys=True, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
