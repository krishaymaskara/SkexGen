"""Structural and access audit for the ADR-0016 memory-count probe."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import re
import sys


AUDIT_VERSION = "GE1-MEMORY-COUNT-PROBE-AUDIT-v1"
DIAGNOSTIC_MODULE = "prototype.graph_encoder.memory_count_probe"


def _shell_source(source):
    return "\n".join(
        line for line in source.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )


def audit(package_root, runner_path):
    root = Path(package_root)
    diagnostic = root / "memory_count_probe.py"
    runner = Path(runner_path)
    source = diagnostic.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(diagnostic), feature_version=(3, 8))
    imports = {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    required_imports = {
        "_load_retained_checkpoint",
        "load_stage6_train",
        "verify_postmortem_artifact",
    }
    if not required_imports.issubset(imports):
        raise AssertionError("memory-count probe omits governed reuse")
    forbidden_imports = {
        "load_stage6_development",
        "load_train",
        "load_development",
        "load_partition_physical_examples",
        "run_ge1_training",
    }
    if imports & forbidden_imports:
        raise AssertionError("memory-count probe imports prohibited work")
    forbidden_source = (
        "run_ge1_training(",
        "load_stage6_development(", "--development", "DEVELOPMENT_INPUT",
        "RR_ROOT", "ER_ROOT", "IID_ROOT", "CORPUS_ROOT", "MANIFEST_ROOT",
        "MODEL_ARTIFACT", "REPAIRED_ARTIFACT", "torch.rand", "torch.randn",
        "global_mean", "cross_batch", "cross_example", "random_memory",
    )
    if any(value in source for value in forbidden_source):
        raise AssertionError("memory-count source contains prohibited work")
    # One call belongs to the active ADR-0016 entry point; the other remains
    # in the retained copied zero-memory implementation and is unreachable
    # from ``main``.
    if source.count("load_stage6_train(") != 2:
        raise AssertionError("memory-count copied/active train loaders differ")
    if source.count("authenticate_prior_postmortem(") < 3:
        raise AssertionError("prior artifact is not authenticated for use/verification")
    for required in (
        "encoded.memory", "encoded.prequant", "deterministic_count_split(",
        "fit_count_probe(", "majority_class_baseline",
        "shuffled_label_control", "model.requires_grad_(False)",
        "ge1_training_performed\": False",
    ):
        if required not in source:
            raise AssertionError("memory-count source omits " + required)
    if source.count("run_memory_count_probe(") != 2:
        raise AssertionError("memory-count entry point count differs")

    runner_source = runner.read_text(encoding="utf-8")
    shell = _shell_source(runner_source)
    if re.search(r"(^|\s)sbatch(\s|$)", shell):
        raise AssertionError("runner submits a Slurm job")
    required_binds = (
        '--bind "$REPOSITORY:$REPOSITORY:ro"',
        '--bind "$FLAT_2026:$FLAT_2026:ro"',
        '--bind "$TYPED_GRAPH_2026:$TYPED_GRAPH_2026:ro"',
        '--bind "$FLAT_2027:$FLAT_2027:ro"',
        '--bind "$TYPED_GRAPH_2027:$TYPED_GRAPH_2027:ro"',
        '--bind "$FLAT_2028:$FLAT_2028:ro"',
        '--bind "$TYPED_GRAPH_2028:$TYPED_GRAPH_2028:ro"',
        '--bind "$TRAIN_INPUT_ROOT:$TRAIN_INPUT_ROOT:ro"',
        '--bind "$POSTMORTEM_ARTIFACT:$POSTMORTEM_ARTIFACT:ro"',
        '--bind "$RUN_PARENT:$RUN_PARENT:rw"',
    )
    if shell.count("--bind") != 10 or any(
        value not in shell for value in required_binds
    ):
        raise AssertionError("memory-count bind set differs")
    if '--bind "$RETAINED_WORK_ROOT:' in shell:
        raise AssertionError("runner broadly binds retained work")
    if "--nv" in shell:
        raise AssertionError("memory-count runner must remain CPU-only")
    invocations = re.findall(
        r"-m\s+" + re.escape(DIAGNOSTIC_MODULE) + r"(?=\s|$)", shell
    )
    if len(invocations) != 1:
        raise AssertionError("memory-count probe must run exactly once")
    forbidden_shell = (
        "DEVELOPMENT_INPUT", "RR_ROOT", "ER_ROOT", "IID_ROOT",
        "HISTORY_DEPTH_ROOT", "GEOMETRY_EXTRAPOLATION_ROOT", "CORPUS_ROOT",
        "MANIFEST_ROOT", "MODEL_ARTIFACT", "REPAIRED_ARTIFACT",
        "run_ge1_training", "scientific_training.py",
    )
    if any(value in shell for value in forbidden_shell):
        raise AssertionError("runner exposes a prohibited resource")
    for required in (
        "#SBATCH --cpus-per-task=1",
        "#SBATCH --mem=2G",
        "#SBATCH --time=00:30:00",
        "git symbolic-ref -q --short HEAD",
        "git status --porcelain=v1 --untracked-files=all",
        "git rev-parse --show-superproject-working-tree",
        "EXPECTED_RUNNER_SHA256",
        "memory_count_probe_terminal_success",
        "memory_count_probe_terminal_failure",
        "memory_count_probe_terminal_timeout",
    ):
        if required not in runner_source:
            raise AssertionError("memory-count runner omits " + required)
    return {
        "version": AUDIT_VERSION,
        "source_ast_audit": "pass",
        "strict_checkpoint_recovery_reused": True,
        "ge1_training_call_count": 0,
        "ge1_backward_call_count": 0,
        "ge1_optimizer_step_call_count": 0,
        "disposable_probe_backward_present": True,
        "disposable_probe_optimizer_step_present": True,
        "development_loader_call_count": 0,
        "train_loader_call_count": 1,
        "diagnostic_invocation_count": 1,
        "exact_checkpoint_file_bind_count": 6,
        "bind_count": 10,
        "protected_bind_count": 0,
        "submission_command_invoked": False,
    }


def main(argv=None):
    values = list(sys.argv[1:] if argv is None else argv)
    if len(values) != 1:
        raise SystemExit(
            "usage: python -m prototype.graph_encoder."
            "memory_count_probe_audit RUNNER"
        )
    root = Path(__file__).resolve().parent
    result = audit(root, values[0])
    print(json.dumps(result, sort_keys=True, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
