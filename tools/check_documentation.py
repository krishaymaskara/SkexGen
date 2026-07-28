#!/usr/bin/env python3
"""Validate durable documentation structure without third-party dependencies."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence
from urllib.parse import unquote


LINK_PATTERN = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
HEADING_PATTERN = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)


def markdown_files(root: Path) -> List[Path]:
    """Return repository Markdown files, excluding Git internals."""

    return sorted(
        path
        for path in root.rglob("*.md")
        if ".git" not in path.parts and path.is_file()
    )


def _link_target(raw_target: str) -> str:
    target = raw_target.strip()
    if target.startswith("<") and ">" in target:
        return target[1 : target.index(">")]
    # Local documentation links in this repository do not use titles. This
    # fallback still handles the conventional `(path "title")` form.
    return re.split(r"\s+[\"']", target, maxsplit=1)[0]


def broken_local_links(root: Path, files: Iterable[Path]) -> List[str]:
    """Return errors for local Markdown links whose targets do not exist."""

    errors: List[str] = []
    for source in files:
        text = source.read_text(encoding="utf-8")
        for match in LINK_PATTERN.finditer(text):
            raw_target = _link_target(match.group(1))
            if (
                not raw_target
                or raw_target.startswith("#")
                or re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", raw_target)
            ):
                continue
            path_part = unquote(raw_target.split("#", 1)[0])
            if not path_part:
                continue
            target = (source.parent / path_part).resolve()
            if not target.exists():
                relative_source = source.relative_to(root)
                errors.append(
                    f"{relative_source}: missing local link target {raw_target!r}"
                )
    return errors


def _headings(text: str) -> List[str]:
    return [heading.strip().lower() for heading in HEADING_PATTERN.findall(text)]


def structured_record_errors(root: Path) -> List[str]:
    """Check required semantic sections in template-style experiment records."""

    experiments = root / "docs" / "experiments"
    errors: List[str] = []
    if not experiments.is_dir():
        return ["docs/experiments: directory is missing"]

    requirements = {
        "question": ("question",),
        "results or findings": ("result", "finding"),
        "decision or interpretation": ("decision", "interpretation"),
        "limitations": ("limitation", "unsupported"),
        "artifacts and integrity": ("artifact", "integrity"),
        "related records": ("related record",),
    }

    for record in sorted(experiments.glob("*.md")):
        if record.name == "README.md":
            continue
        text = record.read_text(encoding="utf-8")
        headings = _headings(text)
        if not any("record status" in heading for heading in headings):
            # Compact historical records predate the structured template.
            continue
        for label, alternatives in requirements.items():
            if not any(
                alternative in heading
                for heading in headings
                for alternative in alternatives
            ):
                errors.append(
                    f"{record.relative_to(root)}: missing {label} section"
                )
    return errors


def index_coverage_errors(root: Path) -> List[str]:
    """Ensure experiment and milestone pages remain discoverable."""

    errors: List[str] = []
    docs_index_path = root / "docs" / "README.md"
    experiment_index_path = root / "docs" / "experiments" / "README.md"
    milestone_index_path = root / "docs" / "milestones" / "README.md"

    required_indexes = (docs_index_path, experiment_index_path, milestone_index_path)
    for index in required_indexes:
        if not index.is_file():
            errors.append(f"{index.relative_to(root)}: required index is missing")
    if errors:
        return errors

    docs_index = docs_index_path.read_text(encoding="utf-8")
    experiment_index = experiment_index_path.read_text(encoding="utf-8")
    milestone_index = milestone_index_path.read_text(encoding="utf-8")

    for record in sorted((root / "docs" / "experiments").glob("*.md")):
        if record.name == "README.md":
            continue
        if f"({record.name})" not in experiment_index:
            errors.append(
                f"{record.relative_to(root)}: absent from experiment evidence index"
            )
        docs_target = f"(experiments/{record.name})"
        if docs_target not in docs_index:
            errors.append(
                f"{record.relative_to(root)}: absent from documentation index"
            )

    for milestone in sorted((root / "docs" / "milestones").glob("*.md")):
        if milestone.name == "README.md":
            continue
        if f"({milestone.name})" not in milestone_index:
            errors.append(
                f"{milestone.relative_to(root)}: absent from milestone index"
            )
        docs_target = f"(milestones/{milestone.name})"
        if docs_target not in docs_index:
            errors.append(
                f"{milestone.relative_to(root)}: absent from documentation index"
            )

    for line_number, line in enumerate(experiment_index.splitlines(), start=1):
        if (
            "`verified-local`" not in line
            or not line.lstrip().startswith("|")
            or line.count("|") < 8
        ):
            continue
        if "Backfill needed" in line or not re.search(r"\[Record\]\([^)]+\.md\)", line):
            errors.append(
                "docs/experiments/README.md:"
                f"{line_number}: verified-local evidence lacks a durable record"
            )

    return errors


def unified_layout_errors(root: Path) -> List[str]:
    """Enforce the consolidated documentation layout and collection indexes."""

    errors: List[str] = []
    legacy_notes = root / "notes"
    if legacy_notes.is_dir():
        for path in sorted(legacy_notes.rglob("*.md")):
            errors.append(
                f"{path.relative_to(root)}: legacy Markdown must be migrated into docs/"
            )

    root_readme = root / "README.md"
    if root_readme.is_file():
        text = root_readme.read_text(encoding="utf-8")
        if "The original upstream SkexGen README is preserved below" in text:
            errors.append(
                "README.md: embedded upstream README must live in docs/reference/"
            )

    docs_index_path = root / "docs" / "README.md"
    if not docs_index_path.is_file():
        return errors
    docs_index = docs_index_path.read_text(encoding="utf-8")

    for collection_name in ("reference", "reports", "specifications"):
        collection = root / "docs" / collection_name
        if not collection.is_dir():
            errors.append(f"docs/{collection_name}: required collection is missing")
            continue
        collection_index = collection / "README.md"
        if not collection_index.is_file():
            errors.append(
                f"docs/{collection_name}/README.md: required index is missing"
            )
            continue
        collection_text = collection_index.read_text(encoding="utf-8")
        if f"({collection_name}/README.md)" not in docs_index:
            errors.append(
                f"docs/{collection_name}/README.md: absent from documentation index"
            )
        for page in sorted(collection.glob("*.md")):
            if page.name == "README.md":
                continue
            if f"({page.name})" not in collection_text:
                errors.append(
                    f"{page.relative_to(root)}: absent from {collection_name} index"
                )

    return errors


def validate(root: Path) -> List[str]:
    root = root.resolve()
    files = markdown_files(root)
    errors: List[str] = []
    errors.extend(broken_local_links(root, files))
    errors.extend(structured_record_errors(root))
    errors.extend(index_coverage_errors(root))
    errors.extend(unified_layout_errors(root))
    return sorted(errors)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="repository root (defaults to the parent of tools/)",
    )
    args = parser.parse_args(argv)

    errors = validate(args.root)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        print(f"Documentation validation failed with {len(errors)} error(s).")
        return 1

    file_count = len(markdown_files(args.root.resolve()))
    print(f"Documentation validation passed for {file_count} Markdown files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
