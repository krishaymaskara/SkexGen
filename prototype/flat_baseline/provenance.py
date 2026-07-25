"""Deterministic repository and exact-source identity for run metadata."""

from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess


def source_state(repository=None):
    repository = (
        Path(repository)
        if repository is not None
        else Path(__file__).resolve().parents[2]
    )
    status = None
    try:
        commit_result = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=str(repository),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            universal_newlines=True,
        )
        status_result = subprocess.run(
            (
                "git",
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
            ),
            cwd=str(repository),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            universal_newlines=True,
        )
    except (OSError, subprocess.SubprocessError):
        commit = None
    else:
        commit = commit_result.stdout.strip() or None
        status = tuple(
            sorted(
                line for line in status_result.stdout.splitlines() if line
            )
        )
    return {
        "git_commit": commit,
        "git_dirty": None if status is None else bool(status),
        "git_status_porcelain": None if status is None else list(status),
        "source_tree_sha256": source_tree_sha256(repository),
    }


def source_tree_sha256(repository):
    """Hash sorted relative paths and raw bytes for prototype Python sources."""

    repository = Path(repository)
    digest = hashlib.sha256()
    prototype = repository / "prototype"
    paths = sorted(
        path
        for path in prototype.rglob("*.py")
        if "__pycache__" not in path.parts
    )
    for path in paths:
        relative = path.relative_to(repository).as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(8, byteorder="big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, byteorder="big"))
        digest.update(content)
    return digest.hexdigest()
