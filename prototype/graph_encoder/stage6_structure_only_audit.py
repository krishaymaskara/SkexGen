"""Static access and boundary audit for structure-only Stage 6."""

from __future__ import annotations

import ast
from pathlib import Path
import re

from .stage6_structure_only_producer_audit import CPU_DISCOVERY_CUDA_SKIP_IDS


AUDIT_VERSION = "GE1-STAGE6-STRUCTURE-ONLY-SOURCE-AUDIT-v1"


def validate_cpu_finalizer_complete_discovery(declared, result):
    """Accept only the six CUDA-only skips in complete CPU discovery."""

    skipped_ids = tuple(sorted(test.id() for test, unused_reason in result.skipped))
    telemetry = {
        "event": "stage6_structure_only_complete_tests",
        "declared": declared,
        "run": result.testsRun,
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


def audit(repository_root, runner_path):
    root = Path(repository_root)
    module = root / "prototype/graph_encoder/stage6_structure_only.py"
    runner = Path(runner_path)
    source = module.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(module), feature_version=(3, 8))
    imported = {
        node.module for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    } | {
        alias.name for node in ast.walk(tree) if isinstance(node, ast.Import)
        for alias in node.names
    }
    forbidden_imports = {
        "prototype.graph_encoder.partitions",
        "prototype.graph_encoder.training",
        "prototype.graph_encoder.model",
        "prototype.graph_encoder.autonomous",
        "prototype.graph_baseline.conversion",
    }
    if imported & forbidden_imports:
        raise AssertionError("Stage 6 scoring imports scientific execution code")
    for forbidden in (
        "load_train", "load_development", "load_partition_physical_examples",
        "run_ge1_training", "torch.load", "validate_and_convert",
    ):
        if forbidden in source:
            raise AssertionError("forbidden scoring dependency: " + forbidden)
    if "execution_evidence" not in source or "timing_hardware_identity" not in source:
        raise AssertionError("finalizer must preserve device provenance")
    if "--execution-record" in source or 'parser.add_argument("--producer-artifact", required=True)' not in source:
        raise AssertionError("authoritative finalization must require producer artifact")
    shell = runner.read_text(encoding="utf-8")
    executable = "\n".join(
        line for line in shell.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    if re.search(r"(^|\s)sbatch(\s|$)", executable):
        raise AssertionError("runner must not submit itself")
    for required in (
        "declared = suite.countTestCases()",
        "validate_cpu_finalizer_complete_discovery",
        "all focused Stage 6 tests must pass with zero skips",
    ):
        if required not in executable:
            raise AssertionError("finalizer discovery policy differs: " + required)
    if "complete graph-encoder discovery must pass with zero skips" in executable:
        raise AssertionError("CPU finalizer broadly rejects the exact CUDA-only skips")
    if re.search(
        r"result\.testsRun\s*(?:==|!=|<=|>=|<|>)\s*\d+", executable
    ):
        raise AssertionError("CPU finalizer hardcodes a complete-discovery total")
    if executable.count("--bind") != 3:
        raise AssertionError("runner must have exactly three governed binds")
    for required in (
        '--bind "$REPOSITORY:$REPOSITORY:ro"',
        '--bind "$STAGE6_PRODUCER_ARTIFACT:$STAGE6_PRODUCER_ARTIFACT:ro"',
        '--bind "$RUN_PARENT:$RUN_PARENT:rw"',
    ):
        if required not in executable:
            raise AssertionError("runner bind differs")
    invocation = re.findall(
        r"-m\s+prototype\.graph_encoder\.stage6_structure_only(?=\s|$)",
        executable,
    )
    if len(invocation) != 1:
        raise AssertionError("Stage 6 entry point must be invoked exactly once")
    for forbidden in (
        "RR_DIR", "ER_DIR", "IID_DIR", "HISTORY_DEPTH_DIR",
        "GEOMETRY_EXTRAPOLATION_DIR", "CORPUS_DIR", "CHECKPOINT_DIR",
    ):
        if forbidden in executable:
            raise AssertionError("runner exposes protected/scientific input: " + forbidden)
    return {
        "version": AUDIT_VERSION,
        "entry_point_count": len(invocation),
        "bind_count": executable.count("--bind"),
        "repository_read_only": True,
        "producer_artifact_read_only": True,
        "artifact_parent_read_write": True,
        "protected_partition_inputs": 0,
        "scientific_loader_imports": 0,
        "submission_command_invoked": False,
        "tensor_and_device_independent": True,
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
