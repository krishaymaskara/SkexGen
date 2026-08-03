"""Strict source identity for graph V1 authorization and checkpoints."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import subprocess

from prototype.flat_baseline.provenance import source_tree_sha256


GRAPH_SOURCE_BRANCH = "graph-profile-decoder"
GRAPH_SOURCE_PROVENANCE_FIELDS = frozenset({
    "git_branch",
    "git_commit",
    "git_dirty",
    "git_status_porcelain",
    "source_tree_sha256",
})


class GraphProvenanceError(ValueError):
    """Structured graph source-provenance failure."""

    def __init__(self, code, detail):
        self.code = code
        self.detail = detail
        super().__init__("{}: {}".format(code, detail))


def collect_graph_source_provenance(repository_root):
    """Collect branch, commit, cleanliness, and digest at an explicit Git root."""

    if repository_root is None:
        raise GraphProvenanceError(
            "missing_repository_root", "repository_root must be explicit"
        )
    root = Path(repository_root).resolve()
    actual_root = Path(_git(
        root, ("rev-parse", "--show-toplevel"), "missing_repository_root"
    )).resolve()
    if actual_root != root:
        raise GraphProvenanceError(
            "wrong_repository_root",
            "expected={!r} actual={!r}".format(str(root), str(actual_root)),
        )
    branch = _git(
        root,
        ("symbolic-ref", "--quiet", "--short", "HEAD"),
        "missing_git_branch",
    )
    commit = _git(
        root, ("rev-parse", "--verify", "HEAD"), "missing_git_commit"
    )
    status_output = _git(
        root,
        ("status", "--porcelain=v1", "--untracked-files=all"),
        "malformed_git_status",
        allow_empty=True,
    )
    status = [line for line in status_output.splitlines() if line]
    return {
        "git_branch": branch,
        "git_commit": commit,
        "git_dirty": bool(status),
        "git_status_porcelain": status,
        "source_tree_sha256": source_tree_sha256(root),
    }


def validate_graph_source_provenance(
    provenance,
    *,
    expected_commit,
    expected_branch=GRAPH_SOURCE_BRANCH,
    expected_digest=None
):
    """Validate the exact serialized graph provenance contract."""

    if not isinstance(provenance, Mapping):
        raise GraphProvenanceError(
            "malformed_source_provenance", "provenance must be a mapping"
        )
    extra = set(provenance) - GRAPH_SOURCE_PROVENANCE_FIELDS
    missing_codes = (
        ("git_branch", "missing_git_branch"),
        ("git_commit", "missing_git_commit"),
        ("git_dirty", "dirty_source_tree"),
        ("git_status_porcelain", "malformed_git_status"),
        ("source_tree_sha256", "missing_source_tree_digest"),
    )
    for name, code in missing_codes:
        if name not in provenance:
            raise GraphProvenanceError(code, "field {!r} is missing".format(name))
    if extra:
        raise GraphProvenanceError(
            "malformed_source_provenance",
            "unexpected fields={}".format(sorted(extra)),
        )

    branch = provenance["git_branch"]
    if not isinstance(branch, str) or not branch:
        raise GraphProvenanceError(
            "missing_git_branch", "expected={!r} actual={!r}".format(
                expected_branch, branch
            )
        )
    if branch != expected_branch:
        raise GraphProvenanceError(
            "wrong_git_branch", "expected={!r} actual={!r}".format(
                expected_branch, branch
            )
        )

    commit = provenance["git_commit"]
    if not isinstance(commit, str) or len(commit) != 40:
        raise GraphProvenanceError(
            "missing_git_commit", "expected={!r} actual={!r}".format(
                expected_commit, commit
            )
        )
    if commit != expected_commit:
        raise GraphProvenanceError(
            "wrong_git_commit", "expected={!r} actual={!r}".format(
                expected_commit, commit
            )
        )

    status = provenance["git_status_porcelain"]
    if (
        not isinstance(status, list)
        or any(not isinstance(line, str) or not line for line in status)
    ):
        raise GraphProvenanceError(
            "malformed_git_status", "expected list[str] actual={!r}".format(
                status
            )
        )
    dirty = provenance["git_dirty"]
    if type(dirty) is not bool or dirty is not bool(status):
        raise GraphProvenanceError(
            "dirty_source_tree",
            "git_dirty={!r} status={!r}".format(dirty, status),
        )
    if dirty:
        raise GraphProvenanceError(
            "dirty_source_tree", "git_status_porcelain={!r}".format(status)
        )

    digest = provenance["source_tree_sha256"]
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise GraphProvenanceError(
            "missing_source_tree_digest", "actual={!r}".format(digest)
        )
    if expected_digest is not None and digest != expected_digest:
        raise GraphProvenanceError(
            "source_tree_digest_mismatch",
            "expected={!r} actual={!r}".format(expected_digest, digest),
        )
    return provenance


def verify_graph_source_provenance(repository_root, authorized_provenance):
    """Recollect and require exact equivalence with authorized clean source."""

    expected_commit = (
        authorized_provenance.get("git_commit")
        if isinstance(authorized_provenance, Mapping) else None
    )
    validate_graph_source_provenance(
        authorized_provenance,
        expected_commit=expected_commit,
        expected_branch=GRAPH_SOURCE_BRANCH,
    )
    actual = collect_graph_source_provenance(repository_root)
    validate_graph_source_provenance(
        actual,
        expected_commit=expected_commit,
        expected_branch=GRAPH_SOURCE_BRANCH,
        expected_digest=authorized_provenance.get("source_tree_sha256"),
    )
    return actual


def _git(root, arguments, error_code, allow_empty=False):
    command = ("git",) + tuple(arguments)
    try:
        result = subprocess.run(
            command,
            cwd=str(root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
    except OSError as exc:
        raise GraphProvenanceError(
            error_code, "command={!r} error={!r}".format(command, str(exc))
        )
    value = (
        result.stdout.rstrip("\r\n")
        if allow_empty else result.stdout.strip()
    )
    if result.returncode != 0 or (not allow_empty and not value):
        raise GraphProvenanceError(
            error_code,
            "command={!r} exit_status={} stderr={!r}".format(
                command, result.returncode, result.stderr.strip()
            ),
        )
    return value
