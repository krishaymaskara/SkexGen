"""Prevent unrestricted model-data loaders from escaping C1 boundaries."""

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
            for path in package.glob("*.py")
            if path.name != "partitions.py" and "cache" not in path.parts
        )
        self.assertTrue(paths)
        self.assertEqual(_forbidden_imports(paths), [])

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
