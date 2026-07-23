"""Python 3.8 grammar compatibility guard."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path


class CompatibilityTests(unittest.TestCase):
    def test_all_package_sources_parse_with_python_38_grammar(self):
        root = Path(__file__).resolve().parents[1]
        failures = []
        for path in sorted(root.rglob("*.py")):
            try:
                ast.parse(path.read_text(encoding="utf-8"), feature_version=(3, 8))
            except SyntaxError as exc:
                failures.append("%s:%s" % (path.relative_to(root), exc.lineno))
        self.assertEqual(failures, [])


if __name__ == "__main__":
    unittest.main()
