"""Real-PyTorch tests for the grid-ordinal magnitude repair.

These require a PyTorch runtime and are expected to execute on the Adroit
engineering-validation runner.  They are corpus-free: every fixture is
synthetic and no corpus, manifest, checkpoint, or preserved payload is opened.
"""

from __future__ import annotations

import unittest

try:
    import torch
except ImportError:  # pragma: no cover - exercised only without PyTorch
    torch = None

from prototype.graph_encoder import grid_magnitude as gm
from prototype.graph_encoder.config import (
    frozen_encoder_config,
    grid_frozen_encoder_config,
    legacy_frozen_encoder_config,
)
from prototype.graph_encoder.decoder_contract import (
    GRID_CHECKPOINT_SCHEMA,
    GRID_SHARED_DECODER_VERSION,
    LEGACY_CHECKPOINT_SCHEMA,
    SHARED_DECODER_VERSION,
)

REASON = "grid-magnitude runtime tests require the authoritative PyTorch runtime"


@unittest.skipIf(torch is None, REASON)
class GridHeadTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2026)
        self.width = 32
        self.head = gm.build_grid_magnitude_head(self.width)

    def test_output_shape_and_dtype(self):
        states = torch.randn(3, 7, self.width, dtype=torch.float32)
        logits = self.head(states)
        self.assertEqual(
            tuple(logits.shape),
            (3, 7, len(gm.OPERATION_TYPES), gm.ORDINAL_CUT_COUNT),
        )
        self.assertTrue(torch.isfinite(logits).all())

    def test_rank_consistency_is_structural_for_random_inputs(self):
        """Biases are decreasing by construction, so cuts never cross."""

        with torch.no_grad():
            for parameter in self.head.parameters():
                parameter.normal_(0.0, 3.0)
        states = torch.randn(64, 5, self.width) * 5.0
        logits = self.head(states)
        differences = logits[..., 1:] - logits[..., :-1]
        self.assertTrue(bool((differences <= 0).all().item()))

    def test_ordered_biases_are_strictly_decreasing(self):
        with torch.no_grad():
            self.head.bias_gaps.normal_(0.0, 2.0)
        biases = self.head.ordered_biases()
        differences = biases[..., 1:] - biases[..., :-1]
        self.assertTrue(bool((differences < 0).all().item()))

    def test_decoded_values_are_exact_frozen_grid_members(self):
        states = torch.randn(16, 4, self.width)
        logits = self.head(states)
        values = self.head.normalized_values(logits)
        for position, name in enumerate(gm.OPERATION_TYPES):
            allowed = torch.tensor(gm.NORMALIZED_GRIDS[name])
            observed = values[..., position].reshape(-1)
            for item in observed:
                self.assertTrue(
                    bool((allowed == item).any().item()),
                    "decoded value {} is not a frozen grid member".format(item),
                )

    def test_all_five_classes_are_reachable_for_both_grids(self):
        """Sweep the shared projection so every class index is produced."""

        for position in range(len(gm.OPERATION_TYPES)):
            seen = set()
            with torch.no_grad():
                self.head.projections[position].weight.zero_()
                self.head.projections[position].weight[0, 0] = 1.0
                self.head.first_bias.zero_()
                self.head.bias_gaps.fill_(0.5413)  # softplus(0.5413) ~= 1.0
            for value in torch.linspace(-6.0, 6.0, 200):
                states = torch.zeros(1, 1, self.width)
                states[0, 0, 0] = value
                logits = self.head(states)
                seen.add(int(self.head.class_indices(logits)[0, 0, position]))
            self.assertEqual(seen, set(range(gm.GRID_CLASS_COUNT)))

    def test_finite_nonzero_gradients_through_the_head(self):
        states = torch.randn(4, 3, self.width, requires_grad=True)
        logits = self.head(states)
        labels = torch.zeros_like(logits)
        labels[..., :2] = 1.0
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, labels
        )
        loss.backward()
        for name, parameter in self.head.named_parameters():
            with self.subTest(parameter=name):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertIsNotNone(states.grad)
        self.assertTrue(bool((states.grad.abs().sum() > 0).item()))

    def test_decode_carries_no_gradient(self):
        states = torch.randn(2, 2, self.width, requires_grad=True)
        values = self.head.normalized_values(self.head(states))
        self.assertFalse(values.requires_grad)


@unittest.skipIf(torch is None, REASON)
class SharedDecoderIntegrationTests(unittest.TestCase):
    """Identity wiring, structural preservation, and trunk gradient flow."""

    def _decoder(self, parameterization):
        from prototype.graph_encoder.shared_decoder import SharedGE1Decoder

        return SharedGE1Decoder(
            operation_magnitude_parameterization=parameterization
        )

    def test_grid_identity_instantiates_the_head_and_v2_versions(self):
        decoder = self._decoder(gm.GRID_MAGNITUDE_PARAMETERIZATION)
        self.assertTrue(decoder.uses_grid_magnitude)
        self.assertTrue(hasattr(decoder, "grid_magnitude_head"))
        self.assertEqual(decoder.shared_decoder_version,
                         GRID_SHARED_DECODER_VERSION)

    def test_historical_identities_have_no_grid_head(self):
        for config in (frozen_encoder_config("flat"),
                       legacy_frozen_encoder_config("flat")):
            decoder = self._decoder(
                config.operation_magnitude_parameterization
            )
            with self.subTest(identity=config.operation_magnitude_parameterization):
                self.assertFalse(decoder.uses_grid_magnitude)
                self.assertFalse(hasattr(decoder, "grid_magnitude_head"))
                self.assertEqual(decoder.shared_decoder_version,
                                 SHARED_DECODER_VERSION)

    def test_historical_parameterization_is_numerically_unchanged(self):
        """tanh and positive-sigmoid outputs must match their exact formulas."""

        raw = torch.randn(2, 5, 6, dtype=torch.float32)
        legacy = self._decoder(
            legacy_frozen_encoder_config("flat").operation_magnitude_parameterization
        )
        torch.testing.assert_close(
            legacy.parameterize_remaining_geometry(raw), torch.tanh(raw),
            rtol=0.0, atol=0.0,
        )
        positive = self._decoder(
            frozen_encoder_config("flat").operation_magnitude_parameterization
        )
        epsilon = torch.finfo(raw.dtype).tiny
        expected = torch.cat(
            (
                torch.tanh(raw)[..., :4],
                epsilon + (1.0 - epsilon) * torch.sigmoid(raw[..., 4:]),
            ),
            dim=-1,
        )
        torch.testing.assert_close(
            positive.parameterize_remaining_geometry(raw), expected,
            rtol=0.0, atol=0.0,
        )

    def test_grid_parameterization_requires_decoded_states(self):
        decoder = self._decoder(gm.GRID_MAGNITUDE_PARAMETERIZATION)
        raw = torch.randn(2, 5, 6)
        with self.assertRaises(ValueError):
            decoder.parameterize_remaining_geometry(raw)

    def test_grid_parameterization_keeps_axis_channels_as_tanh(self):
        decoder = self._decoder(gm.GRID_MAGNITUDE_PARAMETERIZATION)
        raw = torch.randn(2, 5, 6)
        states = torch.randn(2, 5, decoder.config.model_dim)
        remaining = decoder.parameterize_remaining_geometry(raw, states)
        torch.testing.assert_close(
            remaining[..., :4], torch.tanh(raw)[..., :4], rtol=0.0, atol=0.0
        )
        for position, name in enumerate(gm.OPERATION_TYPES):
            allowed = torch.tensor(gm.NORMALIZED_GRIDS[name])
            for item in remaining[..., 4 + position].reshape(-1):
                self.assertTrue(bool((allowed == item).any().item()))

    def test_magnitude_head_cannot_reach_structural_logits(self):
        """Perturbing only the grid head must leave structural outputs fixed."""

        decoder = self._decoder(gm.GRID_MAGNITUDE_PARAMETERIZATION).eval()
        states = torch.randn(2, 6, decoder.config.model_dim)
        before = decoder.node_type_head(states).clone()
        with torch.no_grad():
            for parameter in decoder.grid_magnitude_head.parameters():
                parameter.add_(torch.randn_like(parameter) * 5.0)
        after = decoder.node_type_head(states)
        torch.testing.assert_close(before, after, rtol=0.0, atol=0.0)


@unittest.skipIf(torch is None, REASON)
class CheckpointIdentityTests(unittest.TestCase):
    def test_schemas_differ_between_grid_and_historical_models(self):
        self.assertEqual(
            grid_frozen_encoder_config("flat").checkpoint_schema,
            GRID_CHECKPOINT_SCHEMA,
        )
        self.assertEqual(
            frozen_encoder_config("flat").checkpoint_schema,
            LEGACY_CHECKPOINT_SCHEMA,
        )


if __name__ == "__main__":
    unittest.main()
