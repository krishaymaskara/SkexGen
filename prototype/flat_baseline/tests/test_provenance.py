"""Dependency-light exact-source provenance tests."""

from __future__ import annotations

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from prototype.flat_baseline.provenance import (
    source_state,
    source_tree_sha256,
)


class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.repository = Path(self.temporary.name)
        (self.repository / "prototype" / "nested").mkdir(parents=True)
        (self.repository / "prototype" / "a.py").write_bytes(b"a = 1\n")
        (self.repository / "prototype" / "nested" / "b.py").write_bytes(
            b"b = 2\n"
        )

    def test_digest_is_stable_and_sensitive_to_path_and_raw_contents(self):
        first = source_tree_sha256(self.repository)
        second = source_tree_sha256(self.repository)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 64)
        (self.repository / "prototype" / "nested" / "b.py").write_bytes(
            b"b = 3\n"
        )
        self.assertNotEqual(first, source_tree_sha256(self.repository))

    def test_add_delete_and_rename_change_the_digest(self):
        original = source_tree_sha256(self.repository)
        added = self.repository / "prototype" / "added.py"
        added.write_bytes(b"value = 3\n")
        after_add = source_tree_sha256(self.repository)
        self.assertNotEqual(original, after_add)
        added.unlink()
        self.assertEqual(original, source_tree_sha256(self.repository))

        renamed = self.repository / "prototype" / "renamed.py"
        (self.repository / "prototype" / "a.py").rename(renamed)
        self.assertNotEqual(original, source_tree_sha256(self.repository))
        renamed.unlink()
        self.assertNotEqual(original, source_tree_sha256(self.repository))

    def test_path_and_content_framing_prevents_ambiguous_concatenation(self):
        first = self.repository / "first"
        second = self.repository / "second"
        (first / "prototype").mkdir(parents=True)
        (second / "prototype").mkdir(parents=True)
        first_path = b"prototype/a.py"
        first_content = b"b.pyc"
        second_path = b"prototype/a.pyb.py"
        second_content = b"c"
        self.assertEqual(
            first_path + first_content, second_path + second_content
        )
        (first / "prototype" / "a.py").write_bytes(first_content)
        (second / "prototype" / "a.pyb.py").write_bytes(second_content)
        self.assertNotEqual(
            source_tree_sha256(first), source_tree_sha256(second)
        )

    def test_clean_and_dirty_states_cannot_look_identical(self):
        clean = (
            SimpleNamespace(stdout="abc123\n"),
            SimpleNamespace(stdout=""),
        )
        dirty = (
            SimpleNamespace(stdout="abc123\n"),
            SimpleNamespace(
                stdout=(
                    "?? prototype/z.py\n"
                    " M prototype/flat_baseline/training.py\n"
                )
            ),
        )
        with mock.patch(
            "prototype.flat_baseline.provenance.subprocess.run",
            side_effect=clean,
        ):
            clean_state = source_state(self.repository)
        (self.repository / "prototype" / "a.py").write_bytes(b"a = 9\n")
        with mock.patch(
            "prototype.flat_baseline.provenance.subprocess.run",
            side_effect=dirty,
        ):
            dirty_state = source_state(self.repository)
        self.assertFalse(clean_state["git_dirty"])
        self.assertEqual(clean_state["git_status_porcelain"], [])
        self.assertTrue(dirty_state["git_dirty"])
        self.assertEqual(
            dirty_state["git_status_porcelain"],
            [
                " M prototype/flat_baseline/training.py",
                "?? prototype/z.py",
            ],
        )
        self.assertEqual(clean_state["git_commit"], dirty_state["git_commit"])
        self.assertNotEqual(
            clean_state["source_tree_sha256"],
            dirty_state["source_tree_sha256"],
        )

    def test_git_subprocess_failure_is_explicitly_unknown(self):
        with mock.patch(
            "prototype.flat_baseline.provenance.subprocess.run",
            side_effect=OSError("git unavailable"),
        ):
            state = source_state(self.repository)
        self.assertIsNone(state["git_commit"])
        self.assertIsNone(state["git_dirty"])
        self.assertIsNone(state["git_status_porcelain"])
        self.assertEqual(len(state["source_tree_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
