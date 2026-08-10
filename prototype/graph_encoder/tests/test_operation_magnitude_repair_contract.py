"""Pure contracts for the prospective positive operation-magnitude repair."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import ast
import unittest

from prototype.controlled_data.config import GeneratorConfig
from prototype.graph_encoder.config import (
    frozen_encoder_config,
    legacy_frozen_encoder_config,
)
from prototype.graph_encoder.decoder_contract import (
    LEGACY_OPERATION_MAGNITUDE_PARAMETERIZATION,
    OPERATION_MAGNITUDE_COMPACT_CHANNELS,
    OPERATION_MAGNITUDE_EPSILON_POLICY,
    OPERATION_MAGNITUDE_SERIALIZED_CHANNELS,
    POSITIVE_OPERATION_MAGNITUDE_PARAMETERIZATION,
    operation_magnitude_contract_metadata,
)
from prototype.graph_encoder.errors import GraphEncoderError
from prototype.model_data.geometry import ANGLE_SCALE, LENGTH_SCALE


ROOT = Path(__file__).resolve().parents[3]


class OperationMagnitudeContractTests(unittest.TestCase):
    def test_controlled_targets_are_positive_bounded_and_include_one(self):
        config = GeneratorConfig(num_source_families=680)
        normalized_extrusions = tuple(
            value / LENGTH_SCALE for value in config.extrusion_distances
        )
        normalized_revolves = tuple(
            value / ANGLE_SCALE for value in config.revolution_angles
        )
        active = normalized_extrusions + normalized_revolves
        self.assertTrue(all(0.0 < value <= 1.0 for value in active))
        self.assertEqual(max(normalized_extrusions), 0.75)
        self.assertNotIn(1.0, normalized_extrusions)
        self.assertIn(1.0, normalized_revolves)
        self.assertEqual(config.revolution_angles[-1], 360.0)

    def test_versioned_configuration_has_explicit_legacy_and_repaired_modes(self):
        repaired = frozen_encoder_config("flat")
        legacy = legacy_frozen_encoder_config("flat")
        self.assertEqual(
            repaired.operation_magnitude_parameterization,
            POSITIVE_OPERATION_MAGNITUDE_PARAMETERIZATION,
        )
        self.assertEqual(
            legacy.operation_magnitude_parameterization,
            LEGACY_OPERATION_MAGNITUDE_PARAMETERIZATION,
        )
        self.assertEqual(
            repaired.to_dict()["operation_magnitude_parameterization"],
            POSITIVE_OPERATION_MAGNITUDE_PARAMETERIZATION,
        )
        with self.assertRaises(GraphEncoderError) as caught:
            replace(
                repaired,
                operation_magnitude_parameterization="unversioned",
            ).validate()
        self.assertEqual(caught.exception.code, "unauthorized_configuration")

    def test_metadata_freezes_channels_mapping_and_no_kernel_claim(self):
        repaired = operation_magnitude_contract_metadata(
            POSITIVE_OPERATION_MAGNITUDE_PARAMETERIZATION
        )
        legacy = operation_magnitude_contract_metadata(
            LEGACY_OPERATION_MAGNITUDE_PARAMETERIZATION
        )
        self.assertEqual(OPERATION_MAGNITUDE_SERIALIZED_CHANNELS, (37, 38))
        self.assertEqual(OPERATION_MAGNITUDE_COMPACT_CHANNELS, (4, 5))
        self.assertEqual(
            repaired["epsilon_policy"], OPERATION_MAGNITUDE_EPSILON_POLICY
        )
        self.assertEqual(repaired["normalized_domain"], "(0, 1]")
        self.assertEqual(legacy["mapping"], "tanh(raw)")
        self.assertFalse(repaired["cad_kernel_validity_claimed"])
        self.assertTrue(repaired["direction_is_separate_categorical"])

    def test_immutable_protocols_select_legacy_explicitly(self):
        expected = {
            "pilot.py": "build_legacy_matched_ge1_models",
            "optimization_diagnostic.py": "build_legacy_matched_ge1_models",
            "c7_v2.py": "build_legacy_matched_ge1_models",
            "operation_parameter_diagnostic.py": (
                "legacy_frozen_encoder_config"
            ),
        }
        package = ROOT / "prototype" / "graph_encoder"
        for name, marker in expected.items():
            with self.subTest(path=name):
                source = (package / name).read_text(encoding="utf-8")
                self.assertIn(marker, source)
                ast.parse(source, filename=name, feature_version=(3, 8))

    def test_repair_does_not_introduce_graph_encoder_position_features(self):
        source = (
            ROOT / "prototype" / "graph_encoder" / "encoders.py"
        ).read_text(encoding="utf-8")
        graph_encoder_source = source.split(
            "class TypedGraphProgramEncoder", 1
        )[1].split("\ndef encoder_parameter_report", 1)[0]
        self.assertNotIn("node_position_embedding", graph_encoder_source)
        self.assertNotIn("chronological_position", graph_encoder_source)


if __name__ == "__main__":
    unittest.main()
