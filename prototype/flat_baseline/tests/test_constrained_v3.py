"""Focused V3 canonical-plane model, loss, and conversion tests."""

from __future__ import annotations

from dataclasses import replace
import ast
import inspect
from pathlib import Path
from types import SimpleNamespace
import unittest

try:
    import torch
except ImportError:
    torch = None

from prototype.controlled_data.builders import build_history
from prototype.controlled_data.factors import PrimitiveFamily, ReferencePlane
from prototype.model_data.adapters import adapt_flat_mixed
from prototype.model_data.batching import collate_flat
from prototype.model_data.canonical import canonical_nodes_and_edges, reconstruction_target
from prototype.model_data.geometry import denormalize_applicable_geometry
from prototype.model_data.tests.fixtures import source
from prototype.reference_plane_geometry import (
    CANONICAL_PLANE_CONTRACT_ID,
    ReferencePlaneGeometryError,
    canonical_reference_plane_geometry,
)
from prototype.representation.model import GeometryEncoding
from prototype.flat_baseline.constrained_v2_config import ConstrainedProfileV2Config
from prototype.flat_baseline.constrained_v3_config import (
    CONSTRAINED_PROFILE_CHECKPOINT_VERSION,
    CONSTRAINED_PROFILE_DECODER_CONTRACT_VERSION,
    CONSTRAINED_PROFILE_MODEL_CONFIG_VERSION,
    CONSTRAINED_PROFILE_MODEL_NAME,
    V3_LEARNED_GEOMETRY_CHANNEL_INDICES,
    ConstrainedProfileV3Config,
)

TORCH_REASON = "real PyTorch execution is deferred to the authoritative environment"


def _controlled_batch():
    examples = []
    for index, family in enumerate(PrimitiveFamily):
        history = build_history(
            source("E", family, extents=(1.0 + index,)),
            GeometryEncoding.CONTINUOUS,
        )
        nodes, edges = canonical_nodes_and_edges(history)
        target = reconstruction_target(nodes, edges, history.structure.operation_sequence)
        examples.append(adapt_flat_mixed(SimpleNamespace(
            physical_family_id="v3-{}".format(index), nodes=nodes, target=target
        )))
    return collate_flat(tuple(examples))


class CanonicalPlaneStaticTests(unittest.TestCase):
    def test_exact_frames_normalization_and_masks(self):
        expected = {
            ReferencePlane.XY: (0, 0, 0, 1, 0, 0, 0, 1, 0),
            ReferencePlane.XZ: (0, 0, 0, 1, 0, 0, 0, 0, 1),
            ReferencePlane.YZ: (0, 0, 0, 0, 1, 0, 0, 0, 1),
        }
        for plane, physical in expected.items():
            geometry, mask = canonical_reference_plane_geometry(plane)
            self.assertEqual(len(geometry), 39)
            self.assertEqual(geometry[:9], physical)
            self.assertEqual(mask, (True,) * 9 + (False,) * 30)
            self.assertEqual(
                denormalize_applicable_geometry(geometry, mask)[:9], physical
            )

    def test_invalid_category_is_structured(self):
        for value in (None, "<none>", "bad", 99):
            with self.subTest(value=value):
                with self.assertRaises(ReferencePlaneGeometryError) as caught:
                    canonical_reference_plane_geometry(value)
                self.assertEqual(
                    caught.exception.code,
                    "invalid_predicted_reference_plane_category",
                )

    def test_controlled_builder_planes_match_authoritative_lookup(self):
        for plane in ReferencePlane:
            physical = source(
                "E", PrimitiveFamily.CIRCLE, extents=(1.0,)
            )
            physical = replace(physical, reference_plane=plane)
            history = build_history(physical, GeometryEncoding.CONTINUOUS)
            nodes, edges = canonical_nodes_and_edges(history)
            target = reconstruction_target(
                nodes, edges, history.structure.operation_sequence
            )
            plane_index = next(
                index
                for index, node in enumerate(nodes)
                if node.node_type == "reference_plane"
            )
            geometry, mask = canonical_reference_plane_geometry(plane)
            self.assertEqual(target.geometry[plane_index], geometry)
            self.assertEqual(target.geometry_mask[plane_index], mask)

    def test_exact_v3_identity_and_layout(self):
        config = ConstrainedProfileV3Config()
        config.validate()
        self.assertEqual(
            CONSTRAINED_PROFILE_MODEL_NAME,
            "B0-FLAT-CONSTRAINED-PROFILE-CANONICAL-PLANE-v3",
        )
        self.assertEqual(CONSTRAINED_PROFILE_CHECKPOINT_VERSION, 3)
        self.assertEqual(CONSTRAINED_PROFILE_MODEL_CONFIG_VERSION, 3)
        self.assertEqual(CONSTRAINED_PROFILE_DECODER_CONTRACT_VERSION, 3)
        self.assertEqual(V3_LEARNED_GEOMETRY_CHANNEL_INDICES, tuple(range(33, 39)))
        self.assertEqual(config.canonical_plane_contract_id, CANONICAL_PLANE_CONTRACT_ID)
        self.assertNotEqual(config.model_name, ConstrainedProfileV2Config().model_name)

    def test_contract_metadata_is_immutable(self):
        for change in (
            {"checkpoint_version": 2},
            {"learned_geometry_channel_indices": tuple(range(6))},
            {"canonical_plane_contract_id": "wrong"},
        ):
            with self.assertRaises(ValueError):
                replace(ConstrainedProfileV3Config(), **change).validate()

    def test_python_38_grammar(self):
        root = Path(__file__).resolve().parents[2]
        paths = (
            root / "reference_plane_geometry.py",
            root / "reference_plane_geometry_torch.py",
            Path(__file__).resolve().parents[1] / "constrained_v3.py",
            Path(__file__).resolve().parents[1] / "constrained_v3_conversion.py",
            Path(__file__).resolve().parents[1] / "constrained_v3_autonomous.py",
        )
        for path in paths:
            ast.parse(path.read_text(encoding="utf-8"), str(path), feature_version=(3, 8))


@unittest.skipIf(torch is None, TORCH_REASON)
class CanonicalPlaneTensorTests(unittest.TestCase):
    def setUp(self):
        from prototype.constrained_profile_decoder import profile_targets_for_loss
        from prototype.flat_baseline.constrained_v3 import ConstrainedProfileV3Model
        self.batch = _controlled_batch()
        self.inputs = self.batch.to_torch(torch)
        self.target = self.batch.target.to_torch(torch)
        self.profiles = profile_targets_for_loss(self.batch.target, self.inputs["geometry"])
        self.model = ConstrainedProfileV3Model(ConstrainedProfileV3Config())

    def test_tensor_lookup_shape_dtype_device_mask_and_invalid(self):
        from prototype.model_data.vocab import NODE_TYPES, REFERENCE_PLANES
        from prototype.reference_plane_geometry_torch import canonicalize_reference_plane_tensors
        nodes = torch.tensor([[NODE_TYPES.id("reference_plane"), NODE_TYPES.id("axis")]])
        categories = torch.tensor([[REFERENCE_PLANES.id("XY"), REFERENCE_PLANES.id("<none>")]])
        floating = torch.zeros(1, 2, 6, dtype=torch.float64)
        result = canonicalize_reference_plane_tensors(nodes, categories, floating)
        self.assertEqual(result.geometry.shape, (1, 2, 39))
        self.assertEqual(result.geometry.dtype, torch.float64)
        self.assertTrue(result.geometry.is_contiguous())
        self.assertEqual(result.geometry[0, 0, :9].tolist(), [0, 0, 0, 1, 0, 0, 0, 1, 0])
        self.assertFalse(result.geometry_mask[0, 1].any())
        categories[0, 0] = REFERENCE_PLANES.id("<none>")
        with self.assertRaises(ReferencePlaneGeometryError):
            canonicalize_reference_plane_tensors(nodes, categories, floating)

    def test_model_has_six_channel_head_and_no_plane_head(self):
        self.assertEqual(self.model.remaining_geometry_head.out_features, 6)
        names = tuple(name for name, _ in self.model.named_parameters())
        self.assertFalse(any("plane_geometry_head" in name for name in names))
        self.assertFalse(any("non_profile_geometry_head" in name for name in names))

    def test_forward_loss_gradients_and_plane_target_loss_independence(self):
        from prototype.flat_baseline.constrained_v3_losses import constrained_profile_v3_loss
        output = self.model(target=self.target, profile_targets=self.profiles, **self.inputs)
        self.assertEqual(output.remaining_geometry.shape[-1], 6)
        losses = constrained_profile_v3_loss(output, self.target, self.profiles, self.model.config)
        losses.total.backward()
        self.assertIsNotNone(self.model.remaining_geometry_head.weight.grad)
        self.assertTrue(torch.isfinite(self.model.remaining_geometry_head.weight.grad).all())
        self.assertIsNotNone(self.model.remaining_categorical_heads[3].weight.grad)
        mutated = dict(self.target)
        mutated["geometry"] = self.target["geometry"].clone()
        mutated["geometry"][..., :9] = 0.875
        other = constrained_profile_v3_loss(output, mutated, self.profiles, self.model.config)
        torch.testing.assert_close(losses.remaining_geometry, other.remaining_geometry, rtol=0, atol=0)

    def test_predicted_category_controls_full_construction(self):
        from prototype.model_data.vocab import NODE_TYPES, REFERENCE_PLANES
        from prototype.flat_baseline.constrained_v3_conversion import construct_v3_predicted_node_tensors
        node = torch.tensor([NODE_TYPES.id("reference_plane")])
        retained = torch.zeros(1, 5, dtype=torch.long)
        retained[:, 3] = REFERENCE_PLANES.id("XZ")
        family_logits = torch.zeros(1, 3)
        raw = torch.zeros(1, 3)
        remaining = torch.tensor([[.11, .22, .33, .44, .55, .66]])
        record = construct_v3_predicted_node_tensors(node, retained, family_logits, raw, remaining)
        self.assertEqual(record.geometry.shape, (1, 39))
        self.assertEqual(record.geometry[0, :9].tolist(), [0, 0, 0, 1, 0, 0, 0, 0, 1])
        self.assertTrue(torch.equal(record.geometry[0, 33:39], torch.zeros(6)))
        self.assertTrue(record.geometry_mask[0, :9].all())
        self.assertFalse(record.geometry_mask[0, 9:].any())
        retained[:, 3] = 999
        with self.assertRaises(ReferencePlaneGeometryError) as caught:
            construct_v3_predicted_node_tensors(
                node, retained, family_logits, raw, remaining
            )
        self.assertEqual(
            caught.exception.code,
            "invalid_predicted_reference_plane_category",
        )

    def test_v2_and_v3_state_dicts_are_strictly_incompatible(self):
        from prototype.flat_baseline.constrained_v2 import ConstrainedProfileV2Model
        v2 = ConstrainedProfileV2Model(ConstrainedProfileV2Config())
        with self.assertRaises(RuntimeError):
            self.model.load_state_dict(v2.state_dict(), strict=True)
        with self.assertRaises(RuntimeError):
            v2.load_state_dict(self.model.state_dict(), strict=True)

    def test_autonomous_interfaces_accept_no_target(self):
        from prototype.flat_baseline.constrained_v3_autonomous import greedy_decode_v3, greedy_decode_v3_from_memory
        self.assertNotIn("target", inspect.signature(greedy_decode_v3).parameters)
        self.assertNotIn("target", inspect.signature(greedy_decode_v3_from_memory).parameters)
