"""Structural, corpus-free source audit for ADR-0013 engineering validation."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import re
import shlex
import sys

from .stage6_structure_only_producer_audit import CPU_DISCOVERY_CUDA_SKIP_IDS


GRID_MAGNITUDE_SOURCE_AUDIT_VERSION = "GE1-GRID-MAGNITUDE-SOURCE-AUDIT-v1"

PRODUCTION_FILES = (
    "grid_magnitude.py",
    "grid_magnitude_metrics.py",
    "losses.py",
    "model.py",
    "shared_decoder.py",
)
FORBIDDEN_IMPORT_PARTS = (
    "loader",
    "corpus",
    "manifest",
    "checkpoint",
    "payload",
    "artifact",
    "partition",
    "preserved_payload",
)
GRID_INTERFACE_FUNCTIONS = {
    "grid_magnitude_logits",
    "parameterize_remaining_geometry",
    "forward",
    "class_indices",
    "normalized_values",
}
FORBIDDEN_INTERFACE_ARGUMENTS = {
    "target",
    "targets",
    "label",
    "labels",
    "target_class",
    "target_classes",
}
SCIENTIFIC_PATH_WORDS = (
    "CORPUS",
    "MANIFEST",
    "PAYLOAD",
    "CHECKPOINT",
    "ARTIFACT",
    "PARTITION",
)
SCIENTIFIC_ARGUMENTS = re.compile(
    r"--(?:corpus|manifest|payload|checkpoint|artifact|partition)(?:[-_][A-Za-z0-9]+)*\b"
)
SCIENTIFIC_MODULES = re.compile(
    r"prototype[./]graph_encoder[./](?:"
    r"training|pilot|c6_smoke|c7_v2|repaired_sufficiency|"
    r"representation_probe|operation_parameter_diagnostic|"
    r"optimization_diagnostic)(?:\.py)?\b"
)


def validate_cpu_softmax_complete_discovery(declared, result):
    """Accept only the six CUDA-only skips in complete CPU discovery."""

    skipped_ids = tuple(sorted(test.id() for test, unused_reason in result.skipped))
    telemetry = {
        "event": "grid_softmax_magnitude_complete_graph_encoder_suite",
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


def audit_grid_magnitude_sources(package_root, runner_path):
    """Raise on a forbidden structural dependency or executable shell path."""

    root = Path(package_root)
    runner = Path(runner_path)
    imported_modules = {}
    for relative in PRODUCTION_FILES:
        path = root / relative
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imports = _imported_modules(tree)
        forbidden = tuple(sorted(
            name for name in imports
            if any(part in name.lower() for part in FORBIDDEN_IMPORT_PARTS)
        ))
        if forbidden:
            raise AssertionError(
                "{} imports forbidden modules {}".format(relative, forbidden)
            )
        imported_modules[relative] = sorted(imports)

    _audit_grid_interfaces(root)
    _audit_label_derivation(root)
    shell = _executable_shell_source(
        runner.read_text(encoding="utf-8")
    )
    _audit_shell(shell)
    return {
        "version": GRID_MAGNITUDE_SOURCE_AUDIT_VERSION,
        "production_file_count": len(PRODUCTION_FILES),
        "runner": str(runner),
        "python_ast_import_audit": "pass",
        "target_free_grid_interface_audit": "pass",
        "loss_only_tensor_label_derivation_audit": "pass",
        "runner_structural_audit": "pass",
        "repository_read_only_bind_count": 1,
        "scientific_data_path_declared": False,
        "submission_command_invoked": False,
        "scientific_command_invoked": False,
    }


def _imported_modules(tree):
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def _audit_grid_interfaces(root):
    sources = (root / "grid_magnitude.py", root / "shared_decoder.py")
    checked = set()
    for path in sources:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name in GRID_INTERFACE_FUNCTIONS
            ):
                # ``forward`` is audited only on the ordinal head.
                if node.name == "forward" and path.name != "grid_magnitude.py":
                    continue
                arguments = {
                    item.arg
                    for item in (
                        list(node.args.posonlyargs)
                        + list(node.args.args)
                        + list(node.args.kwonlyargs)
                    )
                }
                forbidden = arguments & FORBIDDEN_INTERFACE_ARGUMENTS
                if forbidden:
                    raise AssertionError(
                        "{} accepts forbidden arguments {}".format(
                            node.name, sorted(forbidden)
                        )
                    )
                checked.add(node.name)
    missing = GRID_INTERFACE_FUNCTIONS - checked
    if missing:
        raise AssertionError(
            "grid interfaces missing from structural audit: {}".format(
                sorted(missing)
            )
        )


def _audit_label_derivation(root):
    for path in root.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if "label_tensor" in node.name and path.name != "losses.py":
                raise AssertionError(
                    "tensor label derivation must remain in losses.py: {}".format(
                        path
                    )
                )
    loss_source = (root / "losses.py").read_text(encoding="utf-8")
    if "def _cumulative_label_tensor(" not in loss_source:
        raise AssertionError("ordinal tensor labels are missing from losses.py")


def _executable_shell_source(source):
    """Remove comments and here-document bodies before command inspection."""

    retained = []
    heredoc_end = None
    for raw_line in source.splitlines():
        stripped = raw_line.strip()
        if heredoc_end is not None:
            if stripped == heredoc_end:
                heredoc_end = None
            continue
        if not stripped or stripped.startswith("#"):
            continue
        match = re.search(r"<<-?['\"]?([A-Za-z_][A-Za-z0-9_]*)['\"]?", raw_line)
        command = raw_line
        if match:
            heredoc_end = match.group(1)
            command = raw_line[:match.start()]
        command = _strip_shell_comment(command)
        if command.strip():
            retained.append(command.rstrip())
    return "\n".join(retained).replace("\\\n", " ")


def _strip_shell_comment(line):
    quote = None
    escaped = False
    for index, character in enumerate(line):
        if escaped:
            escaped = False
            continue
        if character == "\\" and quote != "'":
            escaped = True
            continue
        if character in ("'", '"'):
            if quote is None:
                quote = character
            elif quote == character:
                quote = None
            continue
        if character == "#" and quote is None:
            return line[:index]
    return line


def _audit_shell(shell):
    command_tokens = []
    for line in shell.splitlines():
        try:
            tokens = shlex.split(line, comments=True, posix=True)
        except ValueError:
            continue
        if tokens:
            command_tokens.append(tokens)
    if any(tokens[0] == "sbatch" for tokens in command_tokens):
        raise AssertionError("runner invokes a Slurm submission command")

    assignments = re.findall(
        r"(?m)^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)=", shell
    )
    forbidden_assignments = tuple(sorted(
        name for name in assignments
        if any(word in name.upper() for word in SCIENTIFIC_PATH_WORDS)
        and name not in {"PYTHONPYCACHEPREFIX"}
    ))
    if forbidden_assignments:
        raise AssertionError(
            "runner declares scientific-data path variables {}".format(
                forbidden_assignments
            )
        )
    if SCIENTIFIC_ARGUMENTS.search(shell):
        raise AssertionError("runner contains a scientific-data argument")

    bind_count = len(re.findall(r"(?:^|\s)--bind(?:\s|=)", shell))
    if bind_count != 1:
        raise AssertionError(
            "runner must contain exactly one Apptainer bind, found {}".format(
                bind_count
            )
        )
    if '--bind "$REPOSITORY:$REPOSITORY:ro"' not in shell:
        raise AssertionError("repository bind must be exact and read-only")
    if SCIENTIFIC_MODULES.search(shell):
        raise AssertionError("runner invokes a scientific execution module")


def main(argv=None):
    values = list(sys.argv[1:] if argv is None else argv)
    if len(values) != 1:
        raise SystemExit("usage: python -m prototype.graph_encoder.grid_magnitude_audit RUNNER")
    root = Path(__file__).resolve().parent
    record = audit_grid_magnitude_sources(root, values[0])
    print(json.dumps(record, sort_keys=True, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
