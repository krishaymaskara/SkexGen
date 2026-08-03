"""Graph V1 source-provenance schema and real-Git regressions."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from prototype.graph_baseline.provenance import (
    GRAPH_SOURCE_BRANCH,
    GraphProvenanceError,
    collect_graph_source_provenance,
    validate_graph_source_provenance,
    verify_graph_source_provenance,
)


def initialize_temporary_graph_repository(root):
    """Create a clean real Git repository with one prototype source file."""

    root = Path(root)
    root.mkdir(parents=True)
    _git(root, "init")
    _git(root, "checkout", "-b", GRAPH_SOURCE_BRANCH)
    _git(root, "config", "user.name", "Graph Provenance Test")
    _git(root, "config", "user.email", "graph-provenance@example.invalid")
    (root / "prototype").mkdir()
    (root / "prototype" / "sample.py").write_text("VALUE = 1\n")
    (root / ".gitignore").write_text("__pycache__/\n")
    _git(root, "add", ".gitignore", "prototype/sample.py")
    _git(root, "commit", "-m", "temporary graph source")
    return _git(root, "rev-parse", "HEAD")


def _git(root, *arguments):
    result = subprocess.run(
        ("git",) + arguments,
        cwd=str(root),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    return result.stdout.strip()


class GraphProvenanceTests(unittest.TestCase):
    def test_job_3339787_real_git_branch_cleanliness_and_structured_failures(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "source"
            commit = initialize_temporary_graph_repository(root)
            provenance = collect_graph_source_provenance(root)
            self.assertEqual(provenance, {
                "git_branch": GRAPH_SOURCE_BRANCH,
                "git_commit": commit,
                "git_dirty": False,
                "git_status_porcelain": [],
                "source_tree_sha256": provenance["source_tree_sha256"],
            })
            self.assertIsInstance(provenance["git_status_porcelain"], list)
            json.dumps(provenance, allow_nan=False)
            validate_graph_source_provenance(
                provenance, expected_commit=commit
            )

            for name, code in (
                ("git_branch", "missing_git_branch"),
                ("git_commit", "missing_git_commit"),
                ("git_dirty", "dirty_source_tree"),
                ("git_status_porcelain", "malformed_git_status"),
                ("source_tree_sha256", "missing_source_tree_digest"),
            ):
                missing = dict(provenance)
                del missing[name]
                self._assert_code(
                    code,
                    validate_graph_source_provenance,
                    missing,
                    expected_commit=commit,
                )

            original_cwd = Path.cwd()
            try:
                os.chdir(temporary)
                from_other_cwd = collect_graph_source_provenance(root)
            finally:
                os.chdir(str(original_cwd))
            self.assertEqual(from_other_cwd, provenance)
            self.assertEqual(
                verify_graph_source_provenance(root, provenance), provenance
            )

            malformed = dict(provenance, git_branch=None)
            self._assert_code(
                "missing_git_branch",
                validate_graph_source_provenance,
                malformed,
                expected_commit=commit,
            )
            malformed = dict(provenance, git_branch="wrong")
            self._assert_code(
                "wrong_git_branch",
                validate_graph_source_provenance,
                malformed,
                expected_commit=commit,
            )
            malformed = dict(provenance, git_status_porcelain=())
            self._assert_code(
                "malformed_git_status",
                validate_graph_source_provenance,
                malformed,
                expected_commit=commit,
            )
            self._assert_code(
                "wrong_git_commit",
                validate_graph_source_provenance,
                provenance,
                expected_commit="0" * 40,
            )
            self._assert_code(
                "source_tree_digest_mismatch",
                validate_graph_source_provenance,
                provenance,
                expected_commit=commit,
                expected_digest="0" * 64,
            )

            _git(root, "checkout", "--detach", commit)
            self._assert_code(
                "missing_git_branch", collect_graph_source_provenance, root
            )
            _git(root, "checkout", GRAPH_SOURCE_BRANCH)

            source = root / "prototype" / "sample.py"
            source.write_text("VALUE = 2\n")
            dirty = collect_graph_source_provenance(root)
            self.assertTrue(dirty["git_dirty"])
            self.assertEqual(
                dirty["git_status_porcelain"], [" M prototype/sample.py"]
            )
            self._assert_code(
                "dirty_source_tree",
                validate_graph_source_provenance,
                dirty,
                expected_commit=commit,
            )
            _git(root, "checkout", "--", "prototype/sample.py")

            untracked = root / "untracked.txt"
            untracked.write_text("not ignored\n")
            dirty = collect_graph_source_provenance(root)
            self.assertIn("?? untracked.txt", dirty["git_status_porcelain"])
            self._assert_code(
                "dirty_source_tree",
                validate_graph_source_provenance,
                dirty,
                expected_commit=commit,
            )
            untracked.unlink()

            cache = root / "prototype" / "__pycache__"
            cache.mkdir()
            (cache / "sample.pyc").write_bytes(b"ignored")
            ignored = collect_graph_source_provenance(root)
            self.assertFalse(ignored["git_dirty"])
            self.assertEqual(ignored["git_status_porcelain"], [])
            validate_graph_source_provenance(
                ignored, expected_commit=commit
            )

    def _assert_code(self, code, function, *args, **kwargs):
        with self.assertRaises(GraphProvenanceError) as caught:
            function(*args, **kwargs)
        self.assertEqual(caught.exception.code, code)


if __name__ == "__main__":
    unittest.main()
