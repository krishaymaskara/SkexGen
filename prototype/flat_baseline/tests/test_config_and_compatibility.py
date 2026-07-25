"""Configuration, imports, and source-level compatibility tests."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

from prototype.flat_baseline.config import (
    FlatBaselineConfig,
    FlatBaselineConfigurationError,
)


class ConfigurationTests(unittest.TestCase):
    def test_configuration_is_explicit_validated_and_canonical(self):
        config = FlatBaselineConfig()
        config.validate()
        self.assertEqual(config.to_json(), FlatBaselineConfig().to_json())
        self.assertEqual(set(config.to_dict()), set(config.__dataclass_fields__))

    def test_invalid_configuration_is_rejected(self):
        invalid = (
            FlatBaselineConfig(model_dim=30, num_heads=4),
            FlatBaselineConfig(latent_tokens=0),
            FlatBaselineConfig(dropout=1.0),
            FlatBaselineConfig(ema_decay=float("nan")),
            FlatBaselineConfig(codebook_size=True),
            FlatBaselineConfig(vq_loss_weight=-1.0),
        )
        for config in invalid:
            with self.subTest(config=config):
                with self.assertRaises(FlatBaselineConfigurationError):
                    config.validate()


class CompatibilityTests(unittest.TestCase):
    def test_python_38_grammar_and_forbidden_dependencies(self):
        root = Path(__file__).resolve().parents[1]
        failures = []
        for path in sorted(root.rglob("*.py")):
            source = path.read_text(encoding="utf-8")
            try:
                ast.parse(source, filename=str(path), feature_version=(3, 8))
            except SyntaxError as exc:
                failures.append("{}:{}".format(path.name, exc.lineno))
            self.assertNotIn("." + "cuda" + "(", source, str(path))
            self.assertNotIn("torch" + "_geometric", source, str(path))
        self.assertEqual(failures, [])

    def test_public_import_contract_is_declared(self):
        import prototype.flat_baseline as package

        self.assertEqual(
            package.__all__,
            (
                "FlatBaselineConfig",
                "TrainingConfig",
                "FlatMixedVQModel",
                "flat_mixed_vq_loss",
            ),
        )


if __name__ == "__main__":
    unittest.main()
