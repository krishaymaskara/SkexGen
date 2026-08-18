"""Corpus-free real-PyTorch integration for Stage 6 structure-only prep."""

from __future__ import annotations

import unittest

try:
    import torch
except ImportError:  # pragma: no cover - authoritative runtime supplies torch
    torch = None

from prototype.graph_encoder.config import grid_frozen_encoder_config
from prototype.graph_encoder.grid_magnitude import GRID_MAGNITUDE_PARAMETERIZATION
from prototype.graph_encoder.stage6_structure_only import (
    STRUCTURAL_FIELDS,
    score_structural_prefix,
)

if torch is not None:
    from prototype.graph_encoder.model import build_matched_ge1_models


REASON = "Stage 6 real-PyTorch integration requires PyTorch"


@unittest.skipIf(torch is None, REASON)
class Stage6StructureOnlyRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_accepted_grid_models_remain_matched_and_disjoint(self):
        flat, graph = build_matched_ge1_models(
            2026,
            operation_magnitude_parameterization=GRID_MAGNITUDE_PARAMETERIZATION,
        )
        self.assertEqual(flat.config, grid_frozen_encoder_config("flat", 2026))
        self.assertEqual(graph.config, grid_frozen_encoder_config("typed_graph", 2026))
        flat_decoder = tuple(flat.decoder.parameters())
        graph_decoder = tuple(graph.decoder.parameters())
        self.assertEqual(len(flat_decoder), len(graph_decoder))
        for left, right in zip(flat_decoder, graph_decoder):
            self.assertIsNot(left, right)
            torch.testing.assert_close(left, right, rtol=0.0, atol=0.0)

    def test_tensor_checks_reduce_to_pure_structural_evidence(self):
        predicted = torch.tensor((3, 4, 6, 7), dtype=torch.long)
        target = torch.tensor((3, 4, 6, 7), dtype=torch.long)
        exact = bool(torch.equal(predicted, target))
        evidence = [{"k": 1, **{field: exact for field in STRUCTURAL_FIELDS}}]
        score = score_structural_prefix(1, evidence)
        self.assertEqual(score["normalized_structural_prefix"], 1.0)
        self.assertFalse(score["geometry_used"])

    def test_geometry_failure_does_not_enter_structure_score(self):
        evidence = [{"k": 1, **{field: True for field in STRUCTURAL_FIELDS}}]
        score = score_structural_prefix(1, evidence)
        geometry_failure = torch.tensor(float("nan"))
        self.assertTrue(bool(torch.isnan(geometry_failure).item()))
        self.assertEqual(score["normalized_structural_prefix"], 1.0)


if __name__ == "__main__":
    unittest.main()
