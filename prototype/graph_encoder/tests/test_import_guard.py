"""Prevent unrestricted model-data loaders from escaping C1 boundaries.

This is an accident guard, not an airtight security boundary. It scans static
import statements only, so `import prototype.model_data` followed by attribute
access, `importlib`, or any other dynamic lookup would not be detected. Its
purpose is to make an inadvertent widening of data access fail loudly in review,
not to withstand a determined bypass.
"""

from __future__ import annotations

import ast
from pathlib import Path
import tempfile
import unittest

import prototype.graph_encoder as graph_encoder


FORBIDDEN_NAMES = {
    "load_counterfactual_examples",
    "load_partition_physical_examples",
    "load_physical_examples",
    "partition_family_ids",
}

# `partitions.py` is the single audited loader boundary and is exempt by design.
EXEMPT_FILENAMES = {"partitions.py"}
# Tests construct planted violations deliberately; caches and generated trees
# are not source. Every other module, at any depth, must be scanned.
EXCLUDED_DIRECTORIES = {"tests", "__pycache__", "adroit", "generated"}


def _forbidden_imports(paths):
    violations = []
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "prototype.model_data.loader":
                        violations.append((path.name, alias.name))
            elif isinstance(node, ast.ImportFrom):
                if node.module not in (
                    "prototype.model_data",
                    "prototype.model_data.loader",
                ):
                    continue
                for alias in node.names:
                    if alias.name == "*" or alias.name in FORBIDDEN_NAMES:
                        violations.append((path.name, alias.name))
    return violations


class ImportGuardTests(unittest.TestCase):
    def test_production_modules_do_not_import_unrestricted_loaders(self):
        package = Path(graph_encoder.__file__).parent
        paths = tuple(
            path
            for path in package.rglob("*.py")
            if path.name not in EXEMPT_FILENAMES
            and not (set(path.relative_to(package).parts) & EXCLUDED_DIRECTORIES)
        )
        self.assertTrue(paths)
        self.assertEqual(_forbidden_imports(paths), [])

    def test_scan_is_recursive_and_excludes_only_declared_directories(self):
        package = Path(graph_encoder.__file__).parent
        scanned = {
            path.relative_to(package).as_posix()
            for path in package.rglob("*.py")
            if path.name not in EXEMPT_FILENAMES
            and not (set(path.relative_to(package).parts) & EXCLUDED_DIRECTORIES)
        }
        self.assertIn("canonicalization.py", scanned)
        self.assertNotIn("partitions.py", scanned)
        self.assertFalse({name for name in scanned if name.startswith("tests/")})
        self.assertFalse({name for name in scanned if "__pycache__" in name})

    def test_scanner_detects_a_planted_violation(self):
        with tempfile.TemporaryDirectory() as temporary:
            planted = Path(temporary) / "violation.py"
            planted.write_text(
                "from prototype.model_data.loader import "
                "load_partition_physical_examples\n",
                encoding="utf-8",
            )
            self.assertEqual(
                _forbidden_imports((planted,)),
                [("violation.py", "load_partition_physical_examples")],
            )

    def test_public_package_does_not_reexport_unrestricted_loaders(self):
        self.assertTrue(FORBIDDEN_NAMES.isdisjoint(graph_encoder.__all__))
        for name in FORBIDDEN_NAMES:
            self.assertFalse(hasattr(graph_encoder, name))


if __name__ == "__main__":
    unittest.main()
