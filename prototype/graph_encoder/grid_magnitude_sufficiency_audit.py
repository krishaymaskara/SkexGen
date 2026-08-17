"""Structural audit for the one authorized ADR-0013 scientific runner."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import re
import shlex
import sys

from .grid_magnitude_audit import _executable_shell_source


AUDIT_VERSION = "GE1-GRID-MAGNITUDE-SUFFICIENCY-SOURCE-AUDIT-v1"
SCIENTIFIC_MODULE = "prototype.graph_encoder.grid_magnitude_sufficiency"
PROTECTED_TOKENS = (
    "--development",
    "--systematic-rr",
    "--test-er",
    "--iid",
    "--history-depth",
    "--geometry-extrapolation",
    "--preserved-payload",
    "--checkpoint",
    "--model-artifact",
    "--repaired-artifact",
)


def audit_grid_magnitude_sufficiency(package_root, runner_path):
    """Require a train-only, exact-commit, non-submitting execution path."""

    root = Path(package_root)
    runner = Path(runner_path)
    source_path = root / "grid_magnitude_sufficiency.py"
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(source_path))
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    forbidden_imports = {
        "load_development",
        "load_partition_physical_examples",
        "build_legacy_matched_ge1_models",
    }
    if imports & forbidden_imports:
        raise AssertionError("scientific source imports a forbidden entry point")

    run = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "run_grid_magnitude_sufficiency"
    )
    selected_loads = [
        node for node in ast.walk(run)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_load_selected_train"
    ]
    if len(selected_loads) != 2:
        raise AssertionError("scientific source must declare tiny and scaled loads")
    subset_values = {
        keyword.value.value
        for call in selected_loads
        for keyword in call.keywords
        if keyword.arg == "subset_identity"
        and isinstance(keyword.value, ast.Constant)
    }
    if subset_values != {"tiny", "scaled"}:
        raise AssertionError("scientific source loads a non-frozen subset")

    shell = _executable_shell_source(runner.read_text(encoding="utf-8"))
    commands = []
    for line in shell.splitlines():
        try:
            tokens = shlex.split(line, comments=True, posix=True)
        except ValueError:
            continue
        if tokens:
            commands.append(tokens)
    if any(tokens[0] == "sbatch" for tokens in commands):
        raise AssertionError("scientific runner submits a Slurm job")
    if shell.count("--bind") != 3:
        raise AssertionError("scientific runner requires exactly three binds")
    required_binds = (
        '--bind "$REPOSITORY:$REPOSITORY:ro"',
        '--bind "$CORPUS_DIR:$CORPUS_DIR:ro"',
        '--bind "$RUN_PARENT:$RUN_PARENT:rw"',
    )
    if any(value not in shell for value in required_binds):
        raise AssertionError("scientific runner bind modes differ")
    scientific_entry_points = re.findall(
        r"-m\s+" + re.escape(SCIENTIFIC_MODULE) + r"(?=\s|$)", shell
    )
    if len(scientific_entry_points) != 1:
        raise AssertionError("scientific entry point must execute exactly once")
    if any(token in shell for token in PROTECTED_TOKENS):
        raise AssertionError("scientific runner declares a prohibited input")
    if re.search(r"-m\s+prototype\.(?:kernel_validation|cad|repair|stage6|c8)", shell):
        raise AssertionError("scientific runner invokes prohibited work")
    for required in (
        "git symbolic-ref -q --short HEAD",
        "git status --porcelain=v1 --untracked-files=all",
        "git rev-parse --show-superproject-working-tree",
        "grid_magnitude_sufficiency_terminal_timeout",
        "another_scientific_job_authorized",
        "additional_repair_authorized",
    ):
        if required not in shell:
            raise AssertionError("scientific runner omits {}".format(required))
    return {
        "version": AUDIT_VERSION,
        "python_ast_import_audit": "pass",
        "train_only_load_audit": "pass",
        "runner_structural_audit": "pass",
        "bind_count": 3,
        "repository_bind": "read_only",
        "corpus_bind": "read_only",
        "artifact_bind": "read_write",
        "scientific_entry_point_count": 1,
        "submission_command_invoked": False,
        "protected_input_declared": False,
        "later_stage_invoked": False,
    }


def main(argv=None):
    values = list(sys.argv[1:] if argv is None else argv)
    if len(values) != 1:
        raise SystemExit(
            "usage: python -m prototype.graph_encoder."
            "grid_magnitude_sufficiency_audit RUNNER"
        )
    root = Path(__file__).resolve().parent
    record = audit_grid_magnitude_sufficiency(root, values[0])
    print(json.dumps(record, sort_keys=True, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
