"""Structural audit for the corpus-free grid-ordinal trajectory diagnostic."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import re
import sys

from .grid_magnitude_audit import _executable_shell_source


AUDIT_VERSION = "GE1-GRID-ORDINAL-TRAJECTORY-SOURCE-AUDIT-v1"
DIAGNOSTIC_MODULE = "prototype.graph_encoder.grid_ordinal_trajectory"


def audit_grid_ordinal_trajectory(package_root, runner_path):
    root = Path(package_root)
    runner = Path(runner_path)
    diagnostic_path = root / "grid_ordinal_trajectory.py"
    source = diagnostic_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(diagnostic_path))
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    required = {
        "build_grid_magnitude_head",
        "grid_magnitude_terms",
        "NORMALIZED_GRIDS",
        "PHYSICAL_GRIDS",
        "OPERATION_NODE_TYPE_IDS",
        "SERIALIZED_CHANNELS",
        "grid_frozen_encoder_config",
    }
    if not required.issubset(imports):
        raise AssertionError("diagnostic does not import the real frozen implementation")
    forbidden = {
        "load_train", "load_development", "load_partition_physical_examples",
        "load_ge1_checkpoint", "load_training_checkpoint", "build_ge1_model",
        "run_ge1_training", "run_grid_magnitude_sufficiency",
    }
    if imports & forbidden:
        raise AssertionError("diagnostic imports a scientific or protected entry point")
    calls = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    if not {"build_grid_magnitude_head", "grid_magnitude_terms"}.issubset(calls):
        raise AssertionError("diagnostic does not call the real head and loss")
    if "torch.optim.AdamW" not in source or "clip_grad_norm_" not in source:
        raise AssertionError("frozen optimizer or clipping is absent")
    forbidden_source = (
        "prototype.graph_encoder.partitions",
        "prototype.graph_encoder.training",
        "prototype.graph_encoder.model",
        "prototype.graph_encoder.autonomous",
        "--corpus-dir",
        "--checkpoint",
        "--manifest",
    )
    if any(value in source for value in forbidden_source):
        raise AssertionError("diagnostic source declares a forbidden data path")

    shell = _executable_shell_source(runner.read_text(encoding="utf-8"))
    if re.search(r"(^|\s)sbatch(\s|$)", shell):
        raise AssertionError("runner submits a Slurm job")
    if shell.count("--bind") != 2:
        raise AssertionError("runner must declare exactly two binds")
    for value in (
        '--bind "$REPOSITORY:$REPOSITORY:ro"',
        '--bind "$RUN_PARENT:$RUN_PARENT:rw"',
    ):
        if value not in shell:
            raise AssertionError("runner bind mode differs")
    invocation = re.findall(
        r"-m\s+" + re.escape(DIAGNOSTIC_MODULE) + r"(?=\s|$)", shell
    )
    if len(invocation) != 1:
        raise AssertionError("diagnostic entry point must run exactly once")
    forbidden_shell = (
        "CORPUS_DIR", "MANIFEST", "CHECKPOINT", "PRESERVED", "MODEL_ARTIFACT",
        "REPAIRED_ARTIFACT", "scientific_training.py",
        "grid_magnitude_sufficiency ", "run_ge1_training",
    )
    if any(value in shell for value in forbidden_shell):
        raise AssertionError("runner declares a forbidden resource or entry point")
    if re.search(r"-m\s+prototype\.(?:kernel_validation|cad|repair|stage6|c8)", shell):
        raise AssertionError("runner invokes prohibited later-stage work")
    for required_shell in (
        "git symbolic-ref -q --short HEAD",
        "git status --porcelain=v1 --untracked-files=all",
        "git rev-parse --show-superproject-working-tree",
        "grid_ordinal_trajectory_terminal_timeout",
        "scientific_execution",
        "engineering_only",
        "another_job_authorized",
    ):
        if required_shell not in shell:
            raise AssertionError("runner omits {}".format(required_shell))
    return {
        "version": AUDIT_VERSION,
        "python_ast_import_audit": "pass",
        "real_head_and_loss_call_audit": "pass",
        "runner_structural_audit": "pass",
        "bind_count": 2,
        "repository_bind": "read_only",
        "artifact_bind": "read_write",
        "corpus_bind_count": 0,
        "checkpoint_bind_count": 0,
        "diagnostic_entry_point_count": 1,
        "scientific_entry_point_count": 0,
        "submission_command_invoked": False,
        "protected_input_declared": False,
    }


def main(argv=None):
    values = list(sys.argv[1:] if argv is None else argv)
    if len(values) != 1:
        raise SystemExit(
            "usage: python -m prototype.graph_encoder."
            "grid_ordinal_trajectory_audit RUNNER"
        )
    root = Path(__file__).resolve().parent
    result = audit_grid_ordinal_trajectory(root, values[0])
    print(json.dumps(result, sort_keys=True, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
