"""Structural audit for the corpus-free grid-ordinal horizon diagnostic."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import re
import sys

from .grid_magnitude_audit import _executable_shell_source
from .stage6_structure_only_producer_audit import CPU_DISCOVERY_CUDA_SKIP_IDS


AUDIT_VERSION = "GE1-GRID-ORDINAL-HORIZON-SOURCE-AUDIT-v1"
DIAGNOSTIC_MODULE = "prototype.graph_encoder.grid_ordinal_horizon"


def validate_cpu_horizon_complete_discovery(declared, result):
    """Accept only the six CUDA-only skips in complete CPU discovery."""

    skipped_ids = tuple(sorted(test.id() for test, unused_reason in result.skipped))
    telemetry = {
        "event": "grid_ordinal_horizon_complete_graph_encoder_suite",
        "declared_tests": declared,
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(skipped_ids),
        "skip_ids": list(skipped_ids),
    }
    if result.testsRun != declared:
        raise AssertionError("complete discovery did not report every declared test")
    if result.failures or result.errors or not result.wasSuccessful():
        raise AssertionError("complete discovery has failures or errors")
    if len(skipped_ids) != len(set(skipped_ids)):
        raise AssertionError("complete discovery reported duplicate skip IDs")
    if skipped_ids != CPU_DISCOVERY_CUDA_SKIP_IDS:
        raise AssertionError("complete discovery CUDA skip allowlist differs")
    return telemetry


def audit_grid_ordinal_horizon(package_root, runner_path):
    root = Path(package_root)
    runner = Path(runner_path)
    diagnostic_path = root / "grid_ordinal_horizon.py"
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
        raise AssertionError("horizon diagnostic omits a real frozen import")
    forbidden = {
        "load_train", "load_development", "load_partition_physical_examples",
        "load_ge1_checkpoint", "load_training_checkpoint", "build_ge1_model",
        "run_ge1_training", "run_grid_magnitude_sufficiency",
    }
    if imports & forbidden:
        raise AssertionError("horizon diagnostic imports a prohibited entry point")
    calls = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    if not {"build_grid_magnitude_head", "grid_magnitude_terms"}.issubset(calls):
        raise AssertionError("horizon diagnostic does not call the real head/loss")
    for value in (
        "torch.optim.AdamW", "clip_grad_norm_", '"exp_avg"', '"exp_avg_sq"',
        "global_before_clipping", "global_after_clipping",
    ):
        if value not in source:
            raise AssertionError("horizon diagnostic omits {}".format(value))
    forbidden_source = (
        "prototype.graph_encoder.partitions",
        "prototype.graph_encoder.training",
        "prototype.graph_encoder.model",
        "prototype.graph_encoder.autonomous",
        "--corpus-dir", "--checkpoint", "--manifest",
    )
    if any(value in source for value in forbidden_source):
        raise AssertionError("horizon source declares a prohibited data path")

    runner_source = runner.read_text(encoding="utf-8")
    shell = _executable_shell_source(runner_source)
    if re.search(r"(^|\s)sbatch(\s|$)", shell):
        raise AssertionError("runner submits a Slurm job")
    for required_shell in (
        "declared = suite.countTestCases()",
        "validate_cpu_horizon_complete_discovery",
        '("prototype.graph_encoder.tests.test_grid_ordinal_horizon_contract", 17)',
        '("prototype.graph_encoder.tests.test_grid_ordinal_horizon_runtime", 7)',
        '("prototype.graph_encoder.tests.test_grid_magnitude_contract", 28)',
        '("prototype.graph_encoder.tests.test_grid_magnitude_runtime", 14)',
        '("prototype.graph_encoder.tests.test_grid_magnitude_integration", 18)',
        '("prototype.graph_encoder.tests.test_grid_magnitude_sufficiency_contract", 31)',
    ):
        if required_shell not in runner_source:
            raise AssertionError("runner discovery contract differs: " + required_shell)
    if (
        "result.testsRun != 540" in runner_source
        or "all 540 graph-encoder tests" in runner_source
    ):
        raise AssertionError("runner retains a fixed complete-discovery total")
    if re.search(
        r"result\.testsRun\s*(?:==|!=|<=|>=|<|>)\s*\d+", runner_source
    ):
        raise AssertionError("runner hardcodes a complete-discovery total")
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
        raise AssertionError("horizon diagnostic must run exactly once")
    forbidden_shell = (
        "CORPUS_DIR", "MANIFEST", "CHECKPOINT", "PRESERVED", "MODEL_ARTIFACT",
        "REPAIRED_ARTIFACT", "scientific_training.py",
        "grid_magnitude_sufficiency ", "run_ge1_training",
    )
    if any(value in shell for value in forbidden_shell):
        raise AssertionError("runner declares a prohibited resource")
    if re.search(r"-m\s+prototype\.(?:kernel_validation|cad|repair|stage6|c8)", shell):
        raise AssertionError("runner invokes prohibited later-stage work")
    for required_shell in (
        "git symbolic-ref -q --short HEAD",
        "git status --porcelain=v1 --untracked-files=all",
        "git rev-parse --show-superproject-working-tree",
        "grid_ordinal_horizon_terminal_timeout",
        "scientific_execution", "engineering_only", "another_job_authorized",
    ):
        if required_shell not in shell:
            raise AssertionError("runner omits {}".format(required_shell))
    preservation = re.search(
        r"allowed = \{(?P<body>.*?)\}\noutput = subprocess\.check_output",
        runner_source,
        re.DOTALL,
    )
    expected_paths = {
        "prototype/graph_encoder/README.md",
        "prototype/graph_encoder/grid_ordinal_horizon.py",
        "prototype/graph_encoder/grid_ordinal_horizon_audit.py",
        "prototype/graph_encoder/adroit/ge1_grid_ordinal_horizon_cpu.slurm",
        "prototype/graph_encoder/tests/test_grid_ordinal_horizon_contract.py",
        "prototype/graph_encoder/tests/test_grid_ordinal_horizon_runtime.py",
    }
    if preservation is None or set(re.findall(
        r'"([^"]+)"', preservation.group("body")
    )) != expected_paths:
        raise AssertionError("runner horizon preservation allowlist differs")
    return {
        "version": AUDIT_VERSION,
        "python_ast_import_audit": "pass",
        "real_head_and_loss_call_audit": "pass",
        "optimizer_state_audit": "pass",
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
            "grid_ordinal_horizon_audit RUNNER"
        )
    root = Path(__file__).resolve().parent
    result = audit_grid_ordinal_horizon(root, values[0])
    print(json.dumps(result, sort_keys=True, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
