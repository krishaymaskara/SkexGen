from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tools.check_documentation import (
    broken_local_links,
    index_coverage_errors,
    structured_record_errors,
    unified_layout_errors,
)


class DocumentationCheckerTests(unittest.TestCase):
    def test_broken_local_links_reports_only_missing_local_targets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            docs = root / "docs"
            docs.mkdir()
            source = docs / "page.md"
            target = docs / "present.md"
            target.write_text("# Present\n", encoding="utf-8")
            source.write_text(
                "[present](present.md) [anchor](#part) "
                "[external](https://example.com) [missing](absent.md)\n",
                encoding="utf-8",
            )

            errors = broken_local_links(root, [source, target])

            self.assertEqual(
                errors,
                ["docs/page.md: missing local link target 'absent.md'"],
            )

    def test_structured_record_requires_semantic_sections(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            experiments = root / "docs" / "experiments"
            experiments.mkdir(parents=True)
            (experiments / "incomplete.md").write_text(
                "# Run\n\n## Record status\n\n## Question\n\n",
                encoding="utf-8",
            )

            errors = structured_record_errors(root)

            self.assertTrue(any("results or findings" in error for error in errors))
            self.assertTrue(any("artifacts and integrity" in error for error in errors))
            self.assertTrue(any("related records" in error for error in errors))

    def test_compact_historical_record_is_grandfathered(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            experiments = root / "docs" / "experiments"
            experiments.mkdir(parents=True)
            (experiments / "historical.md").write_text(
                "# Historical\n\n## Purpose and status\n\nPreserved summary.\n",
                encoding="utf-8",
            )

            self.assertEqual(structured_record_errors(root), [])

    def test_indexes_require_discoverability_and_verified_local_records(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            experiments = root / "docs" / "experiments"
            milestones = root / "docs" / "milestones"
            experiments.mkdir(parents=True)
            milestones.mkdir()
            (root / "docs" / "README.md").write_text(
                "# Docs\n", encoding="utf-8"
            )
            (experiments / "README.md").write_text(
                "| Run | source | job | path | partition | evidence | record |\n"
                "|---|---|---|---|---|---|---|\n"
                "| Run | abc | 1 | local | none | `verified-local` | Backfill needed |\n",
                encoding="utf-8",
            )
            (milestones / "README.md").write_text(
                "# Milestones\n", encoding="utf-8"
            )
            (experiments / "run.md").write_text("# Run\n", encoding="utf-8")
            (milestones / "stage.md").write_text("# Stage\n", encoding="utf-8")

            errors = index_coverage_errors(root)

            self.assertTrue(any("absent from experiment evidence index" in e for e in errors))
            self.assertTrue(any("absent from milestone index" in e for e in errors))
            self.assertTrue(any("lacks a durable record" in e for e in errors))

    def test_unified_layout_rejects_legacy_notes_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            notes = root / "notes"
            notes.mkdir()
            (notes / "old.md").write_text("# Old\n", encoding="utf-8")

            errors = unified_layout_errors(root)

            self.assertIn(
                "notes/old.md: legacy Markdown must be migrated into docs/",
                errors,
            )


if __name__ == "__main__":
    unittest.main()
