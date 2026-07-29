"""Strict stable-path hashing for repaired Phase B workflow publications."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import stat
import sys
import tempfile

from .evaluate_length_conditioned import (
    EvaluationError,
    RAW_ARTIFACT,
    REQUIRED_ARTIFACTS,
    _atomic_no_replace,
    _fsync_directory,
    _resolve_publication_directory,
    _resolve_publication_file,
)


EVALUATION_ARTIFACTS = frozenset(REQUIRED_ARTIFACTS + (RAW_ARTIFACT,))
WORKFLOW_EVIDENCE_ARTIFACTS = frozenset({
    "checkpoint.sha256",
    "container.sha256",
    "corpus-manifests.sha256",
    "environment.json",
    "partition.json",
    "regressions.txt",
    "repository.txt",
    "scheduler.txt",
})
REPORT_NAME = "gate-a-to-d-inputs.json"
MANIFEST_NAME = "sha256-manifest.txt"
MANIFEST_ENTRY_COUNT = 15


def stable_manifest_entries(namespace):
    """Return all required hashes labeled only by stable publication paths."""

    root = Path(os.path.abspath(os.fspath(namespace)))
    if not root.is_dir() or root.is_symlink():
        raise EvaluationError(
            "invalid_manifest_namespace",
            "namespace must be a non-symlink directory",
        )
    evaluation_public = root / "evaluation"
    report_public = root / REPORT_NAME
    evaluation_root, evaluation_backing = _stable_publication(
        evaluation_public, expect_directory=True
    )
    report_file, report_backing = _stable_publication(
        report_public, expect_directory=False
    )
    evidence_root = root / "workflow-evidence"
    if not evidence_root.is_dir() or evidence_root.is_symlink():
        raise EvaluationError(
            "invalid_manifest_evidence",
            "workflow evidence must be a non-symlink directory",
        )
    _require_exact_regular_files(evaluation_root, EVALUATION_ARTIFACTS)
    _require_exact_regular_files(
        evidence_root, WORKFLOW_EVIDENCE_ARTIFACTS
    )
    manifest_public = root / MANIFEST_NAME
    manifest_backing = None
    if manifest_public.is_symlink():
        _, manifest_backing = _stable_publication(
            manifest_public, expect_directory=False
        )
    elif manifest_public.exists() and not manifest_public.is_file():
        raise EvaluationError(
            "invalid_manifest_output",
            "manifest path has the wrong type",
        )
    required_public = {
        "evaluation",
        "workflow-evidence",
        REPORT_NAME,
    }
    if manifest_public.exists() or manifest_public.is_symlink():
        required_public.add(MANIFEST_NAME)
    actual_public = {
        item.name for item in root.iterdir() if not item.name.startswith(".")
    }
    if actual_public != required_public:
        raise EvaluationError(
            "manifest_namespace_set_mismatch",
            "stable namespace entries differ",
        )
    expected_hidden = {
        item.name
        for item in (evaluation_backing, report_backing, manifest_backing)
        if item is not None
    }
    actual_hidden = {
        item.name for item in root.iterdir() if item.name.startswith(".")
    }
    if actual_hidden != expected_hidden:
        raise EvaluationError(
            "manifest_hidden_set_mismatch",
            "hidden namespace entries differ from publication backings",
        )
    sources = []
    for name in sorted(EVALUATION_ARTIFACTS):
        sources.append((
            "evaluation/" + name,
            evaluation_root / name,
        ))
    for name in sorted(WORKFLOW_EVIDENCE_ARTIFACTS):
        sources.append((
            "workflow-evidence/" + name,
            evidence_root / name,
        ))
    sources.append((REPORT_NAME, report_file))
    entries = tuple(
        (_stable_display_path(root, relative), _stable_file_sha256(source))
        for relative, source in sorted(sources)
    )
    if len(entries) != MANIFEST_ENTRY_COUNT:
        raise EvaluationError(
            "manifest_entry_count_mismatch",
            "exactly 15 stable artifact paths are required",
        )
    return entries


def manifest_bytes(namespace):
    return "".join(
        "{}  {}\n".format(digest, path)
        for path, digest in stable_manifest_entries(namespace)
    ).encode("utf-8")


def publish_manifest(namespace, output):
    """Atomically publish a new manifest without replacing any destination."""

    root, destination = _manifest_destination(namespace, output)
    content = manifest_bytes(root)
    temporary = None
    try:
        descriptor, name = tempfile.mkstemp(
            prefix="." + destination.name + ".tmp-",
            dir=str(root),
        )
        temporary = Path(name)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        _atomic_no_replace(temporary, destination)
        temporary = None
        _fsync_directory(root)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return content


def replace_incomplete_manifest(namespace, output):
    """Replace only the known nine-entry post-validation failure manifest."""

    root, destination = _manifest_destination(namespace, output)
    if destination.is_symlink() or not destination.is_file():
        raise EvaluationError(
            "invalid_incomplete_manifest",
            "recovery requires an existing regular manifest",
        )
    old_stat = destination.stat()
    old_content = destination.read_bytes()
    current = stable_manifest_entries(root)
    by_path = dict(current)
    expected_old_paths = tuple(
        str(root / "workflow-evidence" / name)
        for name in sorted(WORKFLOW_EVIDENCE_ARTIFACTS)
    ) + (str(root / REPORT_NAME),)
    expected_old = tuple(
        (path, by_path[path]) for path in expected_old_paths
    )
    expected_old_content = "".join(
        "{}  {}\n".format(digest, path)
        for path, digest in expected_old
    ).encode("utf-8")
    if old_content != expected_old_content:
        raise EvaluationError(
            "invalid_incomplete_manifest",
            "existing manifest is not the exact validated nine-entry subset",
        )
    content = "".join(
        "{}  {}\n".format(digest, path) for path, digest in current
    ).encode("utf-8")
    temporary = None
    try:
        descriptor, name = tempfile.mkstemp(
            prefix="." + destination.name + ".recovery-",
            dir=str(root),
        )
        temporary = Path(name)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        latest = destination.stat()
        if (
            latest.st_dev,
            latest.st_ino,
            latest.st_size,
            latest.st_mtime_ns,
        ) != (
            old_stat.st_dev,
            old_stat.st_ino,
            old_stat.st_size,
            old_stat.st_mtime_ns,
        ) or destination.read_bytes() != old_content:
            raise EvaluationError(
                "incomplete_manifest_changed",
                "manifest changed during recovery validation",
            )
        os.replace(str(temporary), str(destination))
        temporary = None
        _fsync_directory(root)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    if destination.read_bytes() != content:
        raise EvaluationError(
            "manifest_recovery_failure",
            "replacement manifest bytes differ",
        )
    return content


def _manifest_destination(namespace, output):
    root = Path(os.path.abspath(os.fspath(namespace)))
    destination = Path(os.path.abspath(os.fspath(output)))
    if destination != root / MANIFEST_NAME:
        raise EvaluationError(
            "invalid_manifest_output",
            "manifest must use the stable namespace path",
        )
    return root, destination


def _stable_publication(public, *, expect_directory):
    backing = None
    if public.is_symlink():
        target = os.readlink(str(public))
        required_prefix = "." + public.name + ".tmp-"
        if not target.startswith(required_prefix) or "/" in target:
            raise EvaluationError(
                "invalid_publication_symlink",
                "publication symlink is not its generated hidden sibling",
            )
        backing = public.parent / target
    resolved = (
        _resolve_publication_directory(public)
        if expect_directory
        else _resolve_publication_file(public)
    )
    return resolved, backing


def _require_exact_regular_files(root, expected):
    actual = {item.name for item in root.iterdir()}
    if actual != set(expected) or any(
        item.is_symlink() or not item.is_file() for item in root.iterdir()
    ):
        raise EvaluationError(
            "manifest_artifact_set_mismatch",
            "artifact directory does not contain its exact required files",
        )


def _stable_display_path(root, relative):
    return str(root / Path(relative))


def _stable_file_sha256(path):
    before = path.stat()
    if not stat.S_ISREG(before.st_mode) or path.is_symlink():
        raise EvaluationError(
            "invalid_manifest_artifact",
            "manifest source is not a regular file",
        )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    after = path.stat()
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise EvaluationError(
            "manifest_artifact_changed",
            "artifact changed while it was hashed",
        )
    return digest


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--replace-incomplete", action="store_true")
    arguments = parser.parse_args(argv)
    try:
        content = (
            replace_incomplete_manifest(arguments.namespace, arguments.output)
            if arguments.replace_incomplete
            else publish_manifest(arguments.namespace, arguments.output)
        )
    except (EvaluationError, OSError, ValueError) as exc:
        sys.stderr.write("artifact_manifest_failure: {}\n".format(exc))
        return 1
    sys.stdout.write(
        "artifact_manifest_entries={}\n".format(len(content.splitlines()))
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
