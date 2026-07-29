"""Regression tests for stable-path Phase B artifact manifests."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from prototype.flat_baseline.artifact_manifest import (
    EVALUATION_ARTIFACTS,
    MANIFEST_ENTRY_COUNT,
    MANIFEST_NAME,
    REPORT_NAME,
    WORKFLOW_EVIDENCE_ARTIFACTS,
    manifest_bytes,
    publish_manifest,
    replace_incomplete_manifest,
    stable_manifest_entries,
)
from prototype.flat_baseline.evaluate_length_conditioned import EvaluationError


class ArtifactManifestTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.namespace = Path(self.temporary.name) / "publication"
        self.namespace.mkdir()
        self.evaluation_backing = (
            self.namespace / ".evaluation.tmp-fixture"
        )
        self.evaluation_backing.mkdir()
        for name in EVALUATION_ARTIFACTS:
            (self.evaluation_backing / name).write_bytes(
                ("evaluation:" + name + "\n").encode("utf-8")
            )
        os.symlink(
            self.evaluation_backing.name,
            str(self.namespace / "evaluation"),
        )
        self.evidence = self.namespace / "workflow-evidence"
        self.evidence.mkdir()
        for name in WORKFLOW_EVIDENCE_ARTIFACTS:
            (self.evidence / name).write_bytes(
                ("evidence:" + name + "\n").encode("utf-8")
            )
        self.report_backing = (
            self.namespace / ("." + REPORT_NAME + ".tmp-fixture")
        )
        self.report_backing.write_bytes(b'{"validated":true}\n')
        os.symlink(
            self.report_backing.name,
            str(self.namespace / REPORT_NAME),
        )

    def test_old_find_omits_symlinked_evaluation_and_stable_manifest_has_15(self):
        old = subprocess.run(
            (
                "find",
                str(self.namespace / "evaluation"),
                str(self.evidence),
                "-type",
                "f",
                "-print",
            ),
            check=True,
            stdout=subprocess.PIPE,
            universal_newlines=True,
        ).stdout.splitlines()
        self.assertEqual(len(old), 8)
        self.assertFalse(any("/evaluation/" in path for path in old))
        self.assertEqual(len(old) + 1, 9)

        entries = stable_manifest_entries(self.namespace)
        self.assertEqual(len(entries), MANIFEST_ENTRY_COUNT)
        expected_relative = {
            "evaluation/" + name for name in EVALUATION_ARTIFACTS
        } | {
            "workflow-evidence/" + name
            for name in WORKFLOW_EVIDENCE_ARTIFACTS
        } | {REPORT_NAME}
        root_text = str(self.namespace) + os.sep
        relative = {path[len(root_text):] for path, _ in entries}
        self.assertEqual(relative, expected_relative)
        for stable_path, digest in entries:
            self.assertFalse(Path(stable_path).name.startswith("."))
            self.assertNotIn(".tmp-", stable_path)
            self.assertEqual(
                digest,
                hashlib.sha256(Path(stable_path).read_bytes()).hexdigest(),
            )

    def test_broken_or_redirected_publication_symlinks_fail(self):
        cases = (
            ("evaluation", ".evaluation.tmp-missing"),
            ("evaluation", "redirected-evaluation"),
            (REPORT_NAME, "." + REPORT_NAME + ".tmp-missing"),
            (REPORT_NAME, "redirected-report"),
        )
        for public_name, target in cases:
            with self.subTest(public_name=public_name, target=target):
                public = self.namespace / public_name
                original = os.readlink(str(public))
                public.unlink()
                target_path = self.namespace / target
                if target.startswith("redirected"):
                    if public_name == "evaluation":
                        target_path.mkdir()
                    else:
                        target_path.write_bytes(b"redirected\n")
                os.symlink(target, str(public))
                try:
                    with self.assertRaisesRegex(
                        EvaluationError, "invalid_publication_symlink"
                    ):
                        stable_manifest_entries(self.namespace)
                finally:
                    public.unlink()
                    if target_path.exists():
                        if target_path.is_dir():
                            target_path.rmdir()
                        else:
                            target_path.unlink()
                    os.symlink(original, str(public))

    def test_missing_or_extra_artifacts_fail(self):
        missing = self.evaluation_backing / sorted(
            EVALUATION_ARTIFACTS
        )[0]
        content = missing.read_bytes()
        missing.unlink()
        with self.assertRaisesRegex(
            EvaluationError, "manifest_artifact_set_mismatch"
        ):
            stable_manifest_entries(self.namespace)
        missing.write_bytes(content)

        extra = self.evidence / "extra.txt"
        extra.write_bytes(b"extra\n")
        with self.assertRaisesRegex(
            EvaluationError, "manifest_artifact_set_mismatch"
        ):
            stable_manifest_entries(self.namespace)

    def test_recovery_accepts_only_exact_nine_entry_subset(self):
        entries = stable_manifest_entries(self.namespace)
        by_path = dict(entries)
        old_paths = tuple(
            str(self.evidence / name)
            for name in sorted(WORKFLOW_EVIDENCE_ARTIFACTS)
        ) + (str(self.namespace / REPORT_NAME),)
        old_entries = tuple(
            (path, by_path[path]) for path in old_paths
        )
        self.assertEqual(len(old_entries), 9)
        manifest = self.namespace / MANIFEST_NAME
        manifest.write_text("".join(
            "{}  {}\n".format(digest, path)
            for path, digest in old_entries
        ))
        replacement = replace_incomplete_manifest(
            self.namespace, manifest
        )
        self.assertEqual(len(replacement.splitlines()), 15)
        self.assertEqual(manifest.read_bytes(), manifest_bytes(self.namespace))
        self.assertFalse(tuple(
            self.namespace.glob("." + MANIFEST_NAME + ".recovery-*")
        ))

        manifest.write_text("".join(
            "{}  {}\n".format(digest, path)
            for path, digest in old_entries[:-1]
        ))
        with self.assertRaisesRegex(
            EvaluationError, "invalid_incomplete_manifest"
        ):
            replace_incomplete_manifest(self.namespace, manifest)

    def test_new_manifest_publication_is_atomic_no_replace(self):
        manifest = self.namespace / MANIFEST_NAME
        expected = manifest_bytes(self.namespace)
        self.assertEqual(
            publish_manifest(self.namespace, manifest), expected
        )
        self.assertEqual(manifest.read_bytes(), expected)
        with self.assertRaisesRegex(EvaluationError, "output_collision"):
            publish_manifest(self.namespace, manifest)
        self.assertEqual(manifest.read_bytes(), expected)

    def test_slurm_and_recovery_workflows_use_strict_manifest_contract(self):
        root = Path(__file__).resolve().parents[1] / "adroit"
        workflow = (
            root / "evaluate_repaired_validation_cpu.slurm"
        ).read_text()
        recovery = (
            root / "finalize_repaired_validation_3329040.sh"
        ).read_text()
        self.assertIn(
            "prototype.flat_baseline.artifact_manifest", workflow
        )
        self.assertNotIn(
            'find "$EVALUATION" "$EVIDENCE" -type f', workflow
        )
        self.assertIn(
            'test "$(wc -l < "$HASH_MANIFEST")" -eq 15', workflow
        )
        for required in (
            'JOB_ID="3329040"',
            'RUN_COMMIT="ce4abca8f450745a8832c0189aaae4a7d8263ec0"',
            "original_slurm_state=FAILED",
            "--existing-report \"$REPORT\"",
            "--replace-incomplete",
            'sha256sum -c "$HASH_MANIFEST"',
        ):
            self.assertIn(required, recovery)


if __name__ == "__main__":
    unittest.main()
