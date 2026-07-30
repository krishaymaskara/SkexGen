"""Focused tests for the additive constrained-profile heads and losses."""

from __future__ import annotations

import ast
import math
from pathlib import Path
import unittest

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from prototype.constrained_profile_decoder import (
        DEFAULT_PROFILE_FAMILY_LOSS_WEIGHT,
        DEFAULT_PROFILE_PARAMETER_LOSS_WEIGHT,
        PROFILE_FAMILY_CLASS_COUNT,
        PROFILE_PARAMETER_COUNT,
        ConstrainedProfileHeads,
        ProfileHeadOutput,
        constrained_profile_loss,
        profile_family_loss,
        profile_parameter_loss,
        profile_targets_for_loss,
    )
    from prototype.profile_geometry_torch import (
        TensorProfileTargets,
        canonicalize_profile_tensors,
        constrain_profile_parameters,
    )

from prototype.controlled_data.builders import build_history
from prototype.controlled_data.factors import PrimitiveFamily
from prototype.model_data.batching import _collate_targets
from prototype.model_data.canonical import (
    canonical_nodes_and_edges,
    reconstruction_target,
)
from prototype.model_data.tests.fixtures import source
from prototype.profile_geometry import (
    NO_PROFILE_FAMILY_ID,
    PROFILE_EXTENT_MAX,
    PROFILE_EXTENT_MIN,
    PROFILE_FAMILIES,
)
from prototype.representation.model import GeometryEncoding


def _target(template, family, extent=2.0):
    history = build_history(
        source(template, family, extents=(extent,) * len(template)),
        GeometryEncoding.CONTINUOUS,
    )
    nodes, edges = canonical_nodes_and_edges(history)
    return reconstruction_target(
        nodes, edges, history.structure.operation_sequence
    )


@unittest.skipIf(torch is None, "PyTorch is unavailable")
class ConstrainedProfileDecoderTests(unittest.TestCase):
    def test_heads_preserve_batch_and_node_dimensions(self):
        heads = ConstrainedProfileHeads(7)
        decoded = torch.randn((2, 5, 7), dtype=torch.float32)
        output = heads(decoded)
        self.assertEqual(output.family_logits.shape, (2, 5, 3))
        self.assertEqual(output.raw_parameters.shape, (2, 5, 3))
        self.assertEqual(PROFILE_FAMILY_CLASS_COUNT, 3)
        self.assertEqual(PROFILE_PARAMETER_COUNT, 3)
        self.assertTrue(torch.isfinite(output.family_logits).all())
        self.assertTrue(torch.isfinite(output.raw_parameters).all())
        self.assertTrue(output.family_logits.is_contiguous())
        self.assertTrue(output.raw_parameters.is_contiguous())

    def test_heads_validate_dimension_shape_dtype_and_finiteness(self):
        for width in (0, -1, True, 2.5):
            with self.subTest(width=width):
                with self.assertRaises(ValueError):
                    ConstrainedProfileHeads(width)
        heads = ConstrainedProfileHeads(4)
        invalid = (
            torch.zeros((2, 4)),
            torch.zeros((2, 3, 5)),
            torch.zeros((2, 3, 4), dtype=torch.long),
        )
        for value in invalid:
            with self.subTest(shape=tuple(value.shape), dtype=value.dtype):
                with self.assertRaises((TypeError, ValueError)):
                    heads(value)
        nonfinite = torch.zeros((1, 1, 4))
        nonfinite[0, 0, 0] = math.inf
        with self.assertRaisesRegex(ValueError, "finite"):
            heads(nonfinite)

    def test_heads_follow_ordinary_linear_initialization(self):
        torch.manual_seed(19)
        expected_family = torch.nn.Linear(5, 3)
        expected_parameters = torch.nn.Linear(5, 3)
        torch.manual_seed(19)
        heads = ConstrainedProfileHeads(5)
        self.assertTrue(
            torch.equal(heads.family_head.weight, expected_family.weight)
        )
        self.assertTrue(
            torch.equal(heads.family_head.bias, expected_family.bias)
        )
        self.assertTrue(
            torch.equal(heads.parameter_head.weight, expected_parameters.weight)
        )
        self.assertTrue(
            torch.equal(heads.parameter_head.bias, expected_parameters.bias)
        )

    def test_family_loss_uses_only_sketches_and_exact_denominator(self):
        logits = torch.tensor(
            (
                ((2.0, 0.0, -1.0), (-1.0, 0.0, 2.0), (9.0, -9.0, 0.0)),
                ((0.5, 1.5, -0.5), (8.0, -8.0, 0.0), (0.0, 0.0, 0.0)),
            ),
            requires_grad=True,
        )
        family_ids = torch.tensor(
            (
                (0, 2, NO_PROFILE_FAMILY_ID),
                (
                    1,
                    NO_PROFILE_FAMILY_ID,
                    NO_PROFILE_FAMILY_ID,
                ),
            ),
            dtype=torch.long,
        )
        mask = family_ids != NO_PROFILE_FAMILY_ID
        result = profile_family_loss(logits, family_ids, mask)
        expected_first = torch.nn.functional.cross_entropy(
            logits[0, :2], family_ids[0, :2], reduction="sum"
        ) / 2.0
        expected_second = torch.nn.functional.cross_entropy(
            logits[1, :1], family_ids[1, :1], reduction="sum"
        )
        self.assertTrue(
            torch.allclose(
                result.per_example,
                torch.stack((expected_first, expected_second)),
            )
        )
        self.assertTrue(
            torch.allclose(
                result.loss, (expected_first + expected_second) / 2.0
            )
        )
        self.assertEqual(result.count, 3)
        self.assertEqual(result.per_example_count.tolist(), [2, 1])

    def test_family_loss_no_sketch_is_differentiable_zero(self):
        logits = torch.randn((2, 4, 3), requires_grad=True)
        family_ids = torch.full((2, 4), NO_PROFILE_FAMILY_ID, dtype=torch.long)
        mask = torch.zeros((2, 4), dtype=torch.bool)
        result = profile_family_loss(logits, family_ids, mask)
        self.assertEqual(result.loss.item(), 0.0)
        self.assertEqual(result.count, 0)
        self.assertEqual(result.per_example_count.tolist(), [0, 0])
        result.loss.backward()
        self.assertIsNotNone(logits.grad)
        self.assertFalse(logits.grad.any())

    def test_family_loss_rejects_invalid_applicable_and_non_sketch_ids(self):
        logits = torch.zeros((1, 1, 3))
        mask = torch.tensor(((True,),), dtype=torch.bool)
        for family_id in (-1, 3):
            with self.subTest(family_id=family_id):
                ids = torch.tensor(((family_id,),), dtype=torch.long)
                with self.assertRaisesRegex(ValueError, "unsupported"):
                    profile_family_loss(logits, ids, mask)
        with self.assertRaisesRegex(ValueError, "no-profile"):
            profile_family_loss(
                logits,
                torch.tensor(((0,),), dtype=torch.long),
                torch.tensor(((False,),), dtype=torch.bool),
            )

    def test_parameter_loss_uses_target_family_routing_for_all_families(self):
        family_ids = torch.tensor(((0, 1, 2),), dtype=torch.long)
        mask = torch.ones((1, 3), dtype=torch.bool)
        raw = torch.zeros((1, 3, 3), requires_grad=True)
        target_raw = torch.tensor(
            (((0.4, -0.3, 0.2), (-0.2, 0.5, -0.4), (0.3, 0.2, 0.6)),)
        )
        targets = constrain_profile_parameters(target_raw, family_ids, mask)
        wrong_family_predictions = torch.tensor(
            (((-5.0, -4.0, 7.0), (8.0, -3.0, -2.0), (-1.0, 9.0, -2.0)),)
        )
        combined = constrained_profile_loss(
            ProfileHeadOutput(wrong_family_predictions, raw),
            TensorProfileTargets(family_ids, targets, mask),
        )
        direct = profile_parameter_loss(raw, family_ids, targets, mask)
        self.assertTrue(torch.equal(combined.parameter_loss, direct.loss))
        self.assertEqual(combined.parameter_count, 9)
        self.assertEqual(tuple(PROFILE_FAMILIES), (
            PrimitiveFamily.CIRCLE,
            PrimitiveFamily.RECTANGLE_LINES,
            PrimitiveFamily.CAPSULE_LINE_ARC,
        ))

    def test_parameter_loss_preserves_gradients_through_all_raw_values(self):
        family_ids = torch.tensor(((0, 1, 2),), dtype=torch.long)
        mask = torch.ones((1, 3), dtype=torch.bool)
        raw = torch.zeros((1, 3, 3), requires_grad=True)
        target_raw = torch.full((1, 3, 3), 0.4)
        targets = constrain_profile_parameters(target_raw, family_ids, mask)
        result = profile_parameter_loss(raw, family_ids, targets, mask)
        result.loss.backward()
        self.assertIsNotNone(raw.grad)
        self.assertTrue(torch.isfinite(raw.grad).all())
        self.assertTrue((raw.grad != 0.0).all())
        self.assertFalse(family_ids.requires_grad)

    def test_parameter_loss_has_exact_per_example_denominator_and_counts(self):
        family_ids = torch.tensor(
            ((0, 1), (NO_PROFILE_FAMILY_ID, NO_PROFILE_FAMILY_ID)),
            dtype=torch.long,
        )
        mask = family_ids != NO_PROFILE_FAMILY_ID
        raw = torch.zeros((2, 2, 3), requires_grad=True)
        target_raw = torch.zeros((2, 2, 3))
        target_raw[0, 0] = torch.tensor((0.2, -0.1, 0.3))
        target_raw[0, 1] = torch.tensor((-0.3, 0.2, -0.4))
        targets = constrain_profile_parameters(target_raw, family_ids, mask)
        constrained = constrain_profile_parameters(raw, family_ids, mask)
        expected_first = torch.nn.functional.smooth_l1_loss(
            constrained[0, :2], targets[0, :2], reduction="sum"
        ) / 6.0
        result = profile_parameter_loss(raw, family_ids, targets, mask)
        self.assertTrue(
            torch.allclose(
                result.per_example,
                torch.stack((expected_first, expected_first * 0.0)),
            )
        )
        self.assertTrue(torch.allclose(result.loss, expected_first / 2.0))
        self.assertEqual(result.count, 6)
        self.assertEqual(result.per_example_count.tolist(), [6, 0])

    def test_parameter_loss_no_sketch_is_differentiable_zero(self):
        raw = torch.randn((2, 3, 3), requires_grad=True)
        family_ids = torch.full((2, 3), NO_PROFILE_FAMILY_ID, dtype=torch.long)
        mask = torch.zeros((2, 3), dtype=torch.bool)
        targets = torch.zeros_like(raw)
        result = profile_parameter_loss(raw, family_ids, targets, mask)
        self.assertEqual(result.loss.item(), 0.0)
        self.assertEqual(result.count, 0)
        result.loss.backward()
        self.assertIsNotNone(raw.grad)
        self.assertFalse(raw.grad.any())

    def test_extent_bounds_and_extreme_finite_raw_parameters(self):
        family_ids = torch.tensor(((0, 1, 2),), dtype=torch.long)
        mask = torch.ones((1, 3), dtype=torch.bool)
        raw = torch.tensor(
            (((-1000.0, 1000.0, -1000.0),
              (1000.0, -1000.0, 1000.0),
              (-1000.0, -1000.0, 1000.0)),)
        )
        parameters = constrain_profile_parameters(raw, family_ids, mask)
        self.assertTrue(
            torch.all(parameters[..., 2] >= PROFILE_EXTENT_MIN)
        )
        self.assertTrue(
            torch.all(parameters[..., 2] <= PROFILE_EXTENT_MAX)
        )
        geometry = canonicalize_profile_tensors(
            family_ids, parameters, mask
        ).geometry
        self.assertTrue(torch.isfinite(geometry).all())
        result = profile_parameter_loss(
            raw, family_ids, parameters.detach(), mask
        )
        self.assertEqual(result.loss.item(), 0.0)

    def test_loss_rejects_nonfinite_values_and_incompatible_layouts(self):
        family_ids = torch.tensor(((0,),), dtype=torch.long)
        mask = torch.tensor(((True,),), dtype=torch.bool)
        target = constrain_profile_parameters(
            torch.zeros((1, 1, 3)), family_ids, mask
        )
        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(value=value):
                logits = torch.zeros((1, 1, 3))
                logits[0, 0, 0] = value
                with self.assertRaisesRegex(ValueError, "finite"):
                    profile_family_loss(logits, family_ids, mask)
                raw = torch.zeros((1, 1, 3))
                raw[0, 0, 0] = value
                with self.assertRaisesRegex(ValueError, "finite"):
                    profile_parameter_loss(raw, family_ids, target, mask)
                invalid_target = target.clone()
                invalid_target[0, 0, 0] = value
                with self.assertRaisesRegex(ValueError, "finite"):
                    profile_parameter_loss(
                        torch.zeros((1, 1, 3)),
                        family_ids,
                        invalid_target,
                        mask,
                    )
        with self.assertRaisesRegex(ValueError, "shape"):
            profile_family_loss(torch.zeros((1, 1, 4)), family_ids, mask)
        with self.assertRaisesRegex(ValueError, "misaligned"):
            profile_parameter_loss(
                torch.zeros((1, 2, 3)), family_ids, target, mask
            )
        invalid_extent = target.clone()
        invalid_extent[0, 0, 2] = PROFILE_EXTENT_MIN / 2.0
        with self.assertRaisesRegex(ValueError, "extent"):
            profile_parameter_loss(
                torch.zeros((1, 1, 3)),
                family_ids,
                invalid_extent,
                mask,
            )
        with self.assertRaisesRegex(ValueError, "non-sketch"):
            profile_parameter_loss(
                torch.zeros((1, 1, 3)),
                torch.tensor(
                    ((NO_PROFILE_FAMILY_ID,),), dtype=torch.long
                ),
                torch.ones((1, 1, 3)),
                torch.zeros((1, 1), dtype=torch.bool),
            )

    def test_cpu_float64_device_dtype_and_target_transfer(self):
        first = _target("E", PrimitiveFamily.CIRCLE)
        second = _target("EE", PrimitiveFamily.CAPSULE_LINE_ARC)
        batch = _collate_targets((first, second))
        width = len(batch.node_mask[0])
        reference = torch.zeros((2, width, 3), dtype=torch.float64)
        targets = profile_targets_for_loss(batch, reference)
        self.assertEqual(targets.parameters.dtype, torch.float64)
        self.assertEqual(targets.parameters.device.type, "cpu")
        self.assertEqual(targets.family_ids.dtype, torch.long)
        self.assertEqual(targets.sketch_mask.dtype, torch.bool)
        self.assertEqual(targets.family_ids.shape, (2, width))
        self.assertTrue(targets.parameters.is_contiguous())
        self.assertTrue(targets.family_ids.is_contiguous())
        self.assertTrue(targets.sketch_mask.is_contiguous())
        self.assertTrue(
            torch.all(
                targets.family_ids[~targets.sketch_mask]
                == NO_PROFILE_FAMILY_ID
            )
        )
        self.assertFalse(targets.parameters[~targets.sketch_mask].any())

        heads = ConstrainedProfileHeads(4).double()
        output = heads(torch.zeros((2, width, 4), dtype=torch.float64))
        self.assertEqual(output.family_logits.dtype, torch.float64)
        self.assertEqual(output.raw_parameters.dtype, torch.float64)
        losses = constrained_profile_loss(output, targets)
        self.assertEqual(losses.family_loss.dtype, torch.float64)
        self.assertEqual(losses.parameter_loss.dtype, torch.float64)
        self.assertEqual(losses.family_loss.device.type, "cpu")
        self.assertEqual(losses.parameter_loss.device.type, "cpu")

    def test_default_weights_map_one_to_one_to_existing_objectives(self):
        self.assertEqual(DEFAULT_PROFILE_FAMILY_LOSS_WEIGHT, 1.0)
        self.assertEqual(DEFAULT_PROFILE_PARAMETER_LOSS_WEIGHT, 1.0)


class ConstrainedProfileDecoderSourceTests(unittest.TestCase):
    def test_python_38_grammar_and_pytorch_111_operations(self):
        root = Path(__file__).resolve().parents[2]
        path = root / "constrained_profile_decoder.py"
        source_text = path.read_text(encoding="utf-8")
        ast.parse(
            source_text,
            filename=str(path),
            feature_version=(3, 8),
        )
        forbidden = (
            "torch.compile",
            "torch.asarray",
            "torch.func",
            "torch.vmap",
            "Tensor.scatter_reduce",
        )
        for token in forbidden:
            self.assertNotIn(token, source_text)


if __name__ == "__main__":
    unittest.main()
