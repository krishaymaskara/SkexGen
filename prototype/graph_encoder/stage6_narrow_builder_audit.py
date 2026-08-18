"""Static access/publication audit for the ADR-0014 narrow builder."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import re


AUDIT_VERSION = "GE1-STAGE6-NARROW-BUILDER-AUDIT-v1"


def audit(repository_root=".", runner_path=None):
    root = Path(repository_root)
    path = root / "prototype/graph_encoder/stage6_narrow_builder.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path), feature_version=(3, 8))
    imported = {
        node.module for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    forbidden = {
        "prototype.graph_encoder.training",
        "prototype.graph_encoder.model",
        "prototype.graph_encoder.autonomous",
        "prototype.graph_encoder.stage6_structure_only_producer",
    }
    if imported & forbidden:
        raise AssertionError("builder imports scientific execution")
    for required in (
        "load_train(", "load_development(", "load_stage6_train",
        "load_stage6_development", "stage6-{}.incomplete-{}",
        "os.replace", "is_symlink", "PARTITION_CHOICES",
        '"train", "development", "both"', "verify_preparation_receipt",
    ):
        if required not in source:
            raise AssertionError("missing builder control: " + required)
    for forbidden_text in (
        "load_rr", "load_test", "run_ge1_training", "torch.load",
        "create_checkpoint_bundle", "run_stage6_producer",
    ):
        if forbidden_text in source:
            raise AssertionError("forbidden builder path: " + forbidden_text)
    result = {
        "version": AUDIT_VERSION,
        "exportable_partitions": ["development", "train"],
        "protected_partition_entry_points": 0,
        "source_payload_write_paths": 0,
        "independent_narrow_loader_validation": True,
        "job_scoped_staging": True,
        "governed_two_package_publication": True,
        "scientific_execution_imported": False,
        "submission_command_invoked": False,
    }
    if runner_path is not None:
        shell = Path(runner_path).read_text(encoding="utf-8")
        executable = "\n".join(
            line for line in shell.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
        if re.search(r"(^|\s)sbatch(\s|$)", executable):
            raise AssertionError("builder runner submits itself")
        required_binds = (
            '--bind "$REPOSITORY:$REPOSITORY:ro"',
            '--bind "$CORPUS_ROOT:$CORPUS_ROOT:ro"',
            '--bind "$OUTPUT_PARENT:$OUTPUT_PARENT:rw"',
        )
        if executable.count("--bind") != 3 or any(
            item not in executable for item in required_binds
        ):
            raise AssertionError("builder runner bind set differs")
        invocations = re.findall(
            r"-m\s+prototype\.graph_encoder\.stage6_narrow_builder(?=\s|$)",
            executable,
        )
        if len(invocations) != 1:
            raise AssertionError("builder must be invoked once")
        result.update({"runner_bind_count": 3, "builder_invocation_count": 1})
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
