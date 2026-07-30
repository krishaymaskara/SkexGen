"""Focused V2 constrained-profile model and integrated-loss tests."""

from __future__ import annotations

from dataclasses import replace
import ast
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import unittest

try:
    import torch
except ImportError:
    torch = None

from prototype.controlled_data.builders import build_history
from prototype.controlled_data.factors import PrimitiveFamily
from prototype.model_data.adapters import adapt_flat_mixed
from prototype.model_data.batching import collate_flat
from prototype.model_data.canonical import (
    canonical_nodes_and_edges,
    reconstruction_target,
)
from prototype.model_data.tests.fixtures import source
from prototype.model_data.vocab import PRIMITIVE_TYPES
from prototype.profile_geometry import (
    NO_PROFILE_FAMILY_ID,
    PROFILE_FAMILIES,
)
from prototype.representation.model import GeometryEncoding

from prototype.flat_baseline.constrained_v2_config import (
    CONSTRAINED_PROFILE_CHECKPOINT_VERSION,
    CONSTRAINED_PROFILE_DECODER_CONTRACT_VERSION,
    CONSTRAINED_PROFILE_FAMILY_ORDER,
    CONSTRAINED_PROFILE_MODEL_CONFIG_VERSION,
    CONSTRAINED_PROFILE_MODEL_NAME,
    ConstrainedProfileV2Config,
)
from prototype.flat_baseline.config import (
    FlatBaselineConfig,
    FlatBaselineConfigurationError,
)

if torch is not None:
    from prototype.constrained_profile_decoder import (
        profile_targets_for_loss,
    )
    from prototype.flat_baseline.constrained_v2 import (
        NON_PROFILE_GEOMETRY_CHANNELS,
        NON_PROFILE_GEOMETRY_INDICES,
        NON_PROFILE_GEOMETRY_WIDTH,
        REMAINING_CATEGORICAL_FIELDS,
        ConstrainedProfileV2Model,
        ConstrainedProfileV2PrefixOutput,
        scatter_non_profile_geometry,
    )
    from prototype.flat_baseline.constrained_v2_losses import (
        constrained_profile_v2_loss,
    )
    from prototype.flat_baseline.constrained_v2_conversion import (
        construct_v2_predicted_node_tensors,
        validate_and_convert_v2_teacher_forced_prediction,
        v2_teacher_forced_predictions,
    )
    from prototype.flat_baseline.model import FlatMixedVQModel
    from prototype.model_data.geometry import GEOMETRY_WIDTH
    from prototype.model_data.vocab import NODE_TYPES
    from prototype.profile_geometry_torch import TensorProfileTargets


TORCH_REASON = "real PyTorch execution is deferred to the Adroit environment"
_V1_SOURCE_HASHES = {
    "config.py": "83ac1ac0095f60596e4ffdc3d248f26d44e99f4d0ee50cee42d939b35458a280",
    "losses.py": "b0629cb919bf1b426b74fec0f2249f040c888d712dbc527f084a11f1bd2d036b",
    "model.py": "c0c81f022a43c9d64fcdf1cd410f3bca532c4635acbb5e085cbb3622d1147714",
    "vq.py": "d94d31113c210113cffe519687394bdbd03c5283959e5ffe1cac82fdbaa7fec3",
}


def _controlled_batch():
    examples = []
    templates = ("E", "EE", "E")
    for index, (family, template) in enumerate(
        zip(PROFILE_FAMILIES, templates)
    ):
        depth = len(template)
        history = build_history(
            source(
                template,
                family,
                extents=(1.0 + index,) * depth,
            ),
            GeometryEncoding.CONTINUOUS,
        )
        nodes, edges = canonical_nodes_and_edges(history)
        target = reconstruction_target(
            nodes, edges, history.structure.operation_sequence
        )
        examples.append(
            adapt_flat_mixed(
                SimpleNamespace(
                    physical_family_id="{:02d}-{}".format(
                        index, family.value
                    ),
                    nodes=nodes,
                    target=target,
                )
            )
        )
    return collate_flat(tuple(examples))


class ConstrainedProfileV2StaticTests(unittest.TestCase):
    def test_v1_model_sources_remain_byte_exact(self):
        root = Path(__file__).resolve().parents[1]
        for name, expected in _V1_SOURCE_HASHES.items():
            with self.subTest(name=name):
                digest = hashlib.sha256(
                    (root / name).read_bytes()
                ).hexdigest()
                self.assertEqual(digest, expected)

    def test_configuration_identity_serialization_and_validation(self):
        config = ConstrainedProfileV2Config()
        config.validate()
        data = config.to_dict()
        self.assertEqual(data["model_name"], CONSTRAINED_PROFILE_MODEL_NAME)
        self.assertEqual(
            data["model_config_version"],
            CONSTRAINED_PROFILE_MODEL_CONFIG_VERSION,
        )
        self.assertEqual(
            data["decoder_contract_version"],
            CONSTRAINED_PROFILE_DECODER_CONTRACT_VERSION,
        )
        self.assertEqual(
            data["checkpoint_version"],
            CONSTRAINED_PROFILE_CHECKPOINT_VERSION,
        )
        self.assertEqual(
            data["profile_family_order"],
            CONSTRAINED_PROFILE_FAMILY_ORDER,
        )
        self.assertEqual(data["profile_extent_min"], 0.25)
        self.assertEqual(data["profile_extent_max"], 0.75)
        self.assertEqual(config.profile_family_loss_weight, 1.0)
        self.assertEqual(config.profile_parameter_loss_weight, 1.0)
        self.assertEqual(
            json.loads(config.to_json()),
            json.loads(config.to_json()),
        )
        self.assertEqual(
            config.to_json(),
            json.dumps(
                data,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ),
        )

    def test_configuration_rejects_changed_contract_metadata(self):
        invalid = (
            {"model_name": "V1"},
            {"model_config_version": 1},
            {"decoder_contract_version": 1},
            {"checkpoint_version": 1},
            {"profile_family_order": tuple(reversed(PROFILE_FAMILIES))},
            {"profile_extent_min": 0.2},
            {"profile_extent_max": 0.8},
            {"profile_family_loss_weight": 2.0},
            {"profile_parameter_loss_weight": 2.0},
            {"profile_family_loss_weight": math.nan},
        )
        for changes in invalid:
            with self.subTest(changes=changes):
                with self.assertRaises(FlatBaselineConfigurationError):
                    replace(ConstrainedProfileV2Config(), **changes).validate()

    def test_python_38_grammar_and_pytorch_111_operations(self):
        root = Path(__file__).resolve().parents[1]
        paths = (
            root / "constrained_v2.py",
            root / "constrained_v2_config.py",
            root / "constrained_v2_losses.py",
        )
        forbidden = (
            "torch.compile",
            "torch.asarray",
            "torch.func",
            "torch.vmap",
            "Tensor.scatter_reduce",
        )
        for path in paths:
            source_text = path.read_text(encoding="utf-8")
            ast.parse(
                source_text,
                filename=str(path),
                feature_version=(3, 8),
            )
            for token in forbidden:
                self.assertNotIn(token, source_text)


@unittest.skipUnless(torch is not None, TORCH_REASON)
class ConstrainedProfileV2TensorTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(41)
        self.batch = _controlled_batch()
        self.inputs = self.batch.to_torch(torch)
        self.target = self.batch.target.to_torch(torch)
        self.profile_targets = profile_targets_for_loss(
            self.batch.target, self.inputs["geometry"]
        )
        self.config = ConstrainedProfileV2Config(
            model_dim=16,
            num_heads=2,
            feedforward_dim=24,
            codebook_dim=8,
            codebook_size=8,
            edge_pair_dim=12,
            latent_tokens=1,
        )
        self.model = ConstrainedProfileV2Model(self.config)

    def _forward(self):
        return self.model(
            target=self.target,
            profile_targets=self.profile_targets,
            **self.inputs
        )

    def test_v2_has_only_required_output_heads(self):
        self.assertFalse(hasattr(self.model, "geometry_head"))
        self.assertFalse(hasattr(self.model, "categorical_heads"))
        self.assertEqual(len(self.model.remaining_categorical_heads), 5)
        self.assertEqual(
            REMAINING_CATEGORICAL_FIELDS,
            (
                "operation_type",
                "boolean_mode",
                "direction",
                "reference_plane",
                "loop_role",
            ),
        )
        self.assertEqual(self.model.profile_heads.family_head.out_features, 3)
        self.assertEqual(
            self.model.profile_heads.parameter_head.out_features, 3
        )
        self.assertEqual(
            self.model.non_profile_geometry_head.out_features, 15
        )
        state_keys = tuple(self.model.state_dict())
        self.assertFalse(
            any(key.startswith("geometry_head.") for key in state_keys)
        )
        self.assertFalse(
            any(key.startswith("categorical_heads.") for key in state_keys)
        )

    def test_non_profile_geometry_order_and_scatter_are_exact(self):
        self.assertEqual(NON_PROFILE_GEOMETRY_WIDTH, 15)
        self.assertEqual(
            NON_PROFILE_GEOMETRY_INDICES,
            (*tuple(range(9)), *tuple(range(33, 39))),
        )
        self.assertEqual(
            NON_PROFILE_GEOMETRY_CHANNELS,
            (
                "plane_origin_x",
                "plane_origin_y",
                "plane_origin_z",
                "plane_x_axis_x",
                "plane_x_axis_y",
                "plane_x_axis_z",
                "plane_y_axis_x",
                "plane_y_axis_y",
                "plane_y_axis_z",
                "axis_point_x",
                "axis_point_y",
                "axis_direction_x",
                "axis_direction_y",
                "extrude_distance",
                "revolve_angle",
            ),
        )
        values = torch.linspace(-1.0, 1.0, 30).reshape(2, 15)
        scattered = scatter_non_profile_geometry(values)
        self.assertEqual(scattered.shape, (2, GEOMETRY_WIDTH))
        self.assertTrue(scattered.is_contiguous())
        self.assertTrue(
            torch.equal(
                scattered[..., NON_PROFILE_GEOMETRY_INDICES], values
            )
        )
        self.assertFalse(scattered[..., 9:33].any())

    def test_scatter_rejects_bad_shape_dtype_nonfinite_and_bounds(self):
        with self.assertRaisesRegex(ValueError, "15"):
            scatter_non_profile_geometry(torch.zeros((2, 14)))
        with self.assertRaises(TypeError):
            scatter_non_profile_geometry(
                torch.zeros((2, 15), dtype=torch.long)
            )
        for value in (math.nan, math.inf, -math.inf, 1.01, -1.01):
            with self.subTest(value=value):
                values = torch.zeros((1, 15))
                values[0, 0] = value
                with self.assertRaises(ValueError):
                    scatter_non_profile_geometry(values)

    def test_forward_shapes_masks_serialized_order_dtype_and_device(self):
        self.model.eval()
        with torch.no_grad():
            output = self._forward()
        batch_size, node_count = self.target["node_mask"].shape
        self.assertEqual(
            output.profile_family_logits.shape,
            (batch_size, node_count, 3),
        )
        self.assertEqual(
            output.raw_profile_parameters.shape,
            (batch_size, node_count, 3),
        )
        self.assertEqual(
            output.constrained_profile_parameters.shape,
            (batch_size, node_count, 3),
        )
        self.assertEqual(
            output.non_profile_geometry.shape,
            (batch_size, node_count, 15),
        )
        self.assertEqual(
            output.non_profile_geometry_scattered.shape,
            (batch_size, node_count, 39),
        )
        self.assertEqual(output.training_geometry.shape, (batch_size, node_count, 39))
        self.assertFalse(output.non_profile_geometry_scattered[..., 9:33].any())
        self.assertTrue(
            torch.equal(
                output.training_geometry[..., 9:33],
                output.training_profile_geometry[..., 9:33],
            )
        )
        self.assertTrue(
            torch.equal(
                output.training_geometry_mask[..., 9:33],
                output.training_profile_geometry_mask[..., 9:33],
            )
        )
        self.assertTrue((~self.target["node_mask"]).any())
        self.assertEqual(
            set(
                self.profile_targets.family_ids[
                    self.profile_targets.sketch_mask
                ].tolist()
            ),
            {0, 1, 2},
        )
        for value in (
            output.decoded_states,
            output.profile_family_logits,
            output.raw_profile_parameters,
            output.constrained_profile_parameters,
            output.non_profile_geometry,
            output.non_profile_geometry_scattered,
            output.training_geometry,
        ):
            self.assertEqual(value.dtype, torch.float32)
            self.assertEqual(value.device.type, "cpu")
            self.assertTrue(value.is_contiguous())
            self.assertTrue(torch.isfinite(value).all())

        capsule_id = PROFILE_FAMILIES.index(
            PrimitiveFamily.CAPSULE_LINE_ARC
        )
        capsule_rows = (
            self.profile_targets.sketch_mask
            & (self.profile_targets.family_ids == capsule_id)
        )
        primitive_ids = output.training_primitive_type_ids[capsule_rows]
        expected = torch.tensor(
            (
                PRIMITIVE_TYPES.id("arc"),
                PRIMITIVE_TYPES.id("arc"),
                PRIMITIVE_TYPES.id("line"),
                PRIMITIVE_TYPES.id("line"),
            ),
            dtype=torch.long,
        )
        self.assertTrue(primitive_ids.numel())
        self.assertTrue(torch.all(primitive_ids == expected))

    def test_teacher_forcing_uses_shifted_authoritative_prefix(self):
        self.model.eval()
        with torch.no_grad():
            original = self._forward()
            changed_target = {
                key: value.clone() for key, value in self.target.items()
            }
            last = int(changed_target["node_mask"][0].sum().item()) - 1
            changed_target["categorical_attributes"][0, last, 0] = 0
            changed_target["geometry"][0, last] = 0.75
            changed_target["geometry_mask"][0, last] = True
            changed = self.model(
                target=changed_target,
                profile_targets=self.profile_targets,
                **self.inputs
            )
        torch.testing.assert_close(
            original.decoded_states[0][self.target["node_mask"][0]],
            changed.decoded_states[0][self.target["node_mask"][0]],
            rtol=0.0,
            atol=0.0,
        )

    def test_target_family_routes_parameters_not_family_logits(self):
        self.model.eval()
        output = self._forward()
        original_loss = constrained_profile_v2_loss(
            output, self.target, self.profile_targets, self.config
        )
        changed_logits = output.profile_family_logits.roll(1, dims=-1)
        changed = replace(output, profile_family_logits=changed_logits)
        changed_loss = constrained_profile_v2_loss(
            changed, self.target, self.profile_targets, self.config
        )
        torch.testing.assert_close(
            original_loss.profile_parameter,
            changed_loss.profile_parameter,
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            output.constrained_profile_parameters,
            replace(
                output, profile_family_logits=changed_logits
            ).constrained_profile_parameters,
            rtol=0.0,
            atol=0.0,
        )

    def test_non_authoritative_compact_targets_are_rejected(self):
        family_ids = self.profile_targets.family_ids.clone()
        sketch = self.profile_targets.sketch_mask.nonzero(
            as_tuple=False
        )[0]
        batch_index = int(sketch[0].item())
        node_index = int(sketch[1].item())
        family_ids[batch_index, node_index] = (
            family_ids[batch_index, node_index] + 1
        ) % 3
        changed = TensorProfileTargets(
            family_ids,
            self.profile_targets.parameters,
            self.profile_targets.sketch_mask,
        )
        with self.assertRaisesRegex(ValueError, "authoritative"):
            self.model(
                target=self.target,
                profile_targets=changed,
                **self.inputs
            )

    def test_integrated_loss_has_only_v2_components_and_exact_counts(self):
        output = self._forward()
        losses = constrained_profile_v2_loss(
            output, self.target, self.profile_targets, self.config
        )
        self.assertEqual(
            tuple(losses.as_dict()),
            (
                "total",
                "node_type",
                "categorical_attributes",
                "non_profile_geometry",
                "profile_family",
                "profile_parameter",
                "edge_presence",
                "edge_type",
                "operation_pointer",
                "vq_commitment",
            ),
        )
        self.assertNotIn("primitive", losses.per_example)
        self.assertNotIn("geometry", losses.as_dict())
        sketch_count = int(self.profile_targets.sketch_mask.sum().item())
        self.assertEqual(losses.profile_family_count, sketch_count)
        self.assertEqual(losses.profile_parameter_count, 3 * sketch_count)
        for value in losses.as_dict().values():
            self.assertTrue(torch.isfinite(value))

    def test_full_forward_backward_reaches_all_retained_paths(self):
        self.model.train()
        output = self._forward()
        losses = constrained_profile_v2_loss(
            output, self.target, self.profile_targets, self.config
        )
        losses.total.backward()
        named = dict(self.model.named_parameters())
        prefixes = (
            "profile_heads.family_head",
            "profile_heads.parameter_head",
            "non_profile_geometry_head",
            "remaining_categorical_heads.0",
            "remaining_categorical_heads.1",
            "remaining_categorical_heads.2",
            "remaining_categorical_heads.3",
            "remaining_categorical_heads.4",
            "node_type_head",
            "decoder.layers.0",
            "encoder.layers.0",
            "to_codebook",
            "from_codebook",
            "edge_presence_head",
            "edge_type_head",
            "operation_keys",
        )
        for prefix in prefixes:
            with self.subTest(prefix=prefix):
                gradients = [
                    parameter.grad
                    for name, parameter in named.items()
                    if name.startswith(prefix)
                    and parameter.grad is not None
                ]
                self.assertTrue(gradients)
                self.assertTrue(
                    all(torch.isfinite(value).all() for value in gradients)
                )
                self.assertGreater(
                    sum(float(value.abs().sum()) for value in gradients),
                    0.0,
                )

    def test_no_sketch_profile_losses_are_differentiable_zero(self):
        self.model.eval()
        output = self._forward()
        family_logits = output.profile_family_logits.detach().clone()
        family_logits.requires_grad_(True)
        raw_parameters = output.raw_profile_parameters.detach().clone()
        raw_parameters.requires_grad_(True)
        output = replace(
            output,
            profile_family_logits=family_logits,
            raw_profile_parameters=raw_parameters,
        )
        family_ids = torch.full_like(
            self.profile_targets.family_ids, NO_PROFILE_FAMILY_ID
        )
        mask = torch.zeros_like(self.profile_targets.sketch_mask)
        parameters = torch.zeros_like(self.profile_targets.parameters)
        profile_targets = TensorProfileTargets(
            family_ids, parameters, mask
        )
        target = {key: value.clone() for key, value in self.target.items()}
        sequence_shape = target["node_mask"].shape
        self.assertEqual(target["node_type_ids"].shape, sequence_shape)
        self.assertEqual(
            target["categorical_attributes"].shape[:2], sequence_shape
        )
        self.assertEqual(target["geometry"].shape[:2], sequence_shape)
        self.assertEqual(target["geometry_mask"].shape[:2], sequence_shape)
        self.assertEqual(profile_targets.family_ids.shape, sequence_shape)
        self.assertEqual(profile_targets.parameters.shape[:2], sequence_shape)
        self.assertEqual(profile_targets.sketch_mask.shape, sequence_shape)

        original_sketches = self.profile_targets.sketch_mask
        original_profiles = (
            target["node_mask"]
            & (target["node_type_ids"] == NODE_TYPES.id("profile"))
        )
        self.assertTrue(
            torch.equal(
                original_sketches.sum(dim=1),
                original_profiles.sum(dim=1),
            )
        )
        target["node_type_ids"][original_sketches] = NODE_TYPES.id("profile")
        target["categorical_attributes"][original_sketches] = (
            self.target["categorical_attributes"][original_profiles]
        )
        target["boolean_mode_targets"][original_sketches] = (
            self.target["boolean_mode_targets"][original_profiles]
        )
        target["geometry"][original_sketches] = self.target["geometry"][
            original_profiles
        ]
        target["geometry_mask"][original_sketches] = (
            self.target["geometry_mask"][original_profiles]
        )
        self.assertTrue(
            torch.equal(target["node_mask"], self.target["node_mask"])
        )
        self.assertFalse(
            (
                target["node_mask"]
                & (target["node_type_ids"] == NODE_TYPES.id("sketch"))
            ).any()
        )
        losses = constrained_profile_v2_loss(
            output, target, profile_targets, self.config
        )
        self.assertEqual(losses.profile_family.item(), 0.0)
        self.assertEqual(losses.profile_parameter.item(), 0.0)
        self.assertEqual(losses.profile_family_count, 0)
        self.assertEqual(losses.profile_parameter_count, 0)
        (losses.profile_family + losses.profile_parameter).backward()
        self.assertIsNotNone(family_logits.grad)
        self.assertIsNotNone(raw_parameters.grad)
        self.assertFalse(family_logits.grad.any())
        self.assertFalse(raw_parameters.grad.any())

    def test_non_sketch_and_padding_do_not_change_profile_losses(self):
        self.model.eval()
        output = self._forward()
        baseline = constrained_profile_v2_loss(
            output, self.target, self.profile_targets, self.config
        )
        inactive = ~self.profile_targets.sketch_mask
        logits = output.profile_family_logits.clone()
        raw = output.raw_profile_parameters.clone()
        logits[inactive] = torch.tensor((100.0, -100.0, 50.0))
        raw[inactive] = 1000.0
        changed = replace(
            output,
            profile_family_logits=logits,
            raw_profile_parameters=raw,
        )
        altered = constrained_profile_v2_loss(
            changed, self.target, self.profile_targets, self.config
        )
        torch.testing.assert_close(
            baseline.profile_family,
            altered.profile_family,
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            baseline.profile_parameter,
            altered.profile_parameter,
            rtol=0.0,
            atol=0.0,
        )

    def test_v1_and_v2_state_dictionaries_are_strictly_incompatible(self):
        v1 = FlatMixedVQModel(
            FlatBaselineConfig(
                model_dim=16,
                num_heads=2,
                feedforward_dim=24,
                codebook_dim=8,
                codebook_size=8,
                edge_pair_dim=12,
                latent_tokens=1,
            )
        )
        with self.assertRaises(RuntimeError):
            self.model.load_state_dict(v1.state_dict(), strict=True)
        with self.assertRaises(RuntimeError):
            v1.load_state_dict(self.model.state_dict(), strict=True)

    def test_fixed_seed_forward_is_deterministic(self):
        torch.manual_seed(79)
        first_model = ConstrainedProfileV2Model(self.config)
        torch.manual_seed(79)
        second_model = ConstrainedProfileV2Model(self.config)
        first_model.eval()
        second_model.eval()
        with torch.no_grad():
            first = first_model(
                target=self.target,
                profile_targets=self.profile_targets,
                **self.inputs
            )
            second = second_model(
                target=self.target,
                profile_targets=self.profile_targets,
                **self.inputs
            )
        for left, right in (
            (first.node_type_logits, second.node_type_logits),
            (first.profile_family_logits, second.profile_family_logits),
            (first.raw_profile_parameters, second.raw_profile_parameters),
            (first.non_profile_geometry, second.non_profile_geometry),
            (first.quantized_memory, second.quantized_memory),
        ):
            torch.testing.assert_close(left, right, rtol=0.0, atol=0.0)

    def test_v2_bos_prefix_matches_first_teacher_forced_output(self):
        self.model.eval()
        vq_before = {
            name: value.detach().clone()
            for name, value in self.model.named_buffers()
            if name.startswith("vq.")
        }
        decode_calls = []
        original_decode = self.model._decode_embedded_prefix

        def capture_decode(shifted, memory, decoder_valid):
            decode_calls.append(
                (
                    shifted.detach().clone(),
                    memory,
                    decoder_valid.detach().clone(),
                    self.model.training,
                    torch.is_grad_enabled(),
                )
            )
            return original_decode(shifted, memory, decoder_valid)

        with mock.patch.object(
            self.model,
            "_decode_embedded_prefix",
            side_effect=capture_decode,
        ):
            with torch.no_grad():
                full = self._forward()
                batch_size, teacher_length = self.target[
                    "node_mask"
                ].shape
                empty_categories = torch.empty(
                    (batch_size, 0, 10), dtype=torch.long
                )
                empty_geometry = torch.empty(
                    (batch_size, 0, GEOMETRY_WIDTH),
                    dtype=torch.float32,
                )
                empty_mask = torch.empty(
                    (batch_size, 0, GEOMETRY_WIDTH), dtype=torch.bool
                )
                prefix = self.model.decode_prefix(
                    full.quantized_memory,
                    empty_categories,
                    empty_geometry,
                    empty_mask,
                )
                one_node_mask = torch.ones(
                    (batch_size, 1), dtype=torch.bool
                )
                prefix_relations = self.model.decode_relations(
                    prefix.decoded_states, one_node_mask
                )

        self.assertEqual(len(decode_calls), 2)
        teacher_call, prefix_call = decode_calls
        self.assertEqual(
            teacher_call[0].shape,
            (batch_size, teacher_length, self.config.model_dim),
        )
        self.assertEqual(
            prefix_call[0].shape,
            (batch_size, 1, self.config.model_dim),
        )
        self.assertEqual(
            teacher_call[1].data_ptr(),
            full.quantized_memory.data_ptr(),
        )
        self.assertEqual(
            prefix_call[1].data_ptr(),
            full.quantized_memory.data_ptr(),
        )
        expected_bos = self.model.bos.detach().view(1, 1, -1).expand(
            batch_size, 1, -1
        )
        self.assertTrue(torch.equal(teacher_call[0][:, :1], expected_bos))
        self.assertTrue(torch.equal(prefix_call[0], expected_bos))
        self.assertTrue(teacher_call[2][:, 0].all())
        self.assertTrue(prefix_call[2].all())
        self.assertFalse(teacher_call[3])
        self.assertFalse(prefix_call[3])
        self.assertFalse(teacher_call[4])
        self.assertFalse(prefix_call[4])
        self.assertFalse(self.model.training)
        for name, expected in vq_before.items():
            torch.testing.assert_close(
                expected,
                dict(self.model.named_buffers())[name],
                rtol=0.0,
                atol=0.0,
            )

        self.assertIsInstance(prefix, ConstrainedProfileV2PrefixOutput)
        continuous_pairs = (
            (
                "decoded_states",
                full.decoded_states[:, :1],
                prefix.decoded_states,
            ),
            (
                "node_type_logits",
                full.node_type_logits[:, :1],
                prefix.node_type_logits,
            ),
            (
                "profile_family_logits",
                full.profile_family_logits[:, :1],
                prefix.profile_family_logits,
            ),
            (
                "raw_profile_parameters",
                full.raw_profile_parameters[:, :1],
                prefix.raw_profile_parameters,
            ),
            (
                "non_profile_geometry",
                full.non_profile_geometry[:, :1],
                prefix.non_profile_geometry,
            ),
            (
                "edge_presence_logits",
                full.edge_presence_logits[:, :1, :1],
                prefix_relations.edge_presence_logits,
            ),
            (
                "edge_type_logits",
                full.edge_type_logits[:, :1, :1],
                prefix_relations.edge_type_logits,
            ),
            (
                "operation_pointer_logits",
                full.operation_pointer_logits[:, :, :1],
                prefix_relations.operation_pointer_logits,
            ),
        )
        continuous_pairs += tuple(
            (
                "categorical_logits_{}".format(index),
                expected[:, :1],
                actual,
            )
            for index, (expected, actual) in enumerate(
                zip(full.categorical_logits, prefix.categorical_logits)
            )
        )
        # The teacher path executes a length-seven causal kernel while the
        # autonomous BOS path executes a length-one kernel. Float32 reduction
        # ordering may therefore differ by roughly one machine epsilon even
        # though batch, memory, BOS, mask, model mode, and semantics are equal.
        for name, teacher_tensor, autonomous_tensor in continuous_pairs:
            with self.subTest(continuous=name):
                self.assertEqual(
                    teacher_tensor.shape, autonomous_tensor.shape
                )
                self.assertEqual(teacher_tensor.dtype, torch.float32)
                self.assertEqual(autonomous_tensor.dtype, torch.float32)
                torch.testing.assert_close(
                    autonomous_tensor,
                    teacher_tensor,
                    rtol=1e-5,
                    atol=2e-7,
                )

        teacher_node_ids = full.node_type_logits[:, :1].argmax(dim=-1)
        autonomous_node_ids = prefix.node_type_logits.argmax(dim=-1)
        self.assertTrue(torch.equal(
            teacher_node_ids, autonomous_node_ids
        ))
        teacher_retained = torch.stack(
            tuple(
                logits[:, :1].argmax(dim=-1)
                for logits in full.categorical_logits
            ),
            dim=-1,
        )
        autonomous_retained = torch.stack(
            tuple(
                logits.argmax(dim=-1)
                for logits in prefix.categorical_logits
            ),
            dim=-1,
        )
        self.assertTrue(torch.equal(
            teacher_retained, autonomous_retained
        ))
        self.assertTrue(torch.equal(
            full.profile_family_logits[:, :1].argmax(dim=-1),
            prefix.profile_family_logits.argmax(dim=-1),
        ))
        self.assertTrue(torch.equal(
            full.edge_type_logits[:, :1, :1].argmax(dim=-1),
            prefix_relations.edge_type_logits.argmax(dim=-1),
        ))
        self.assertTrue(torch.equal(
            full.operation_pointer_logits[:, :, :1].argmax(dim=-1),
            prefix_relations.operation_pointer_logits.argmax(dim=-1),
        ))
        self.assertTrue(torch.equal(
            full.edge_presence_logits[:, :1, :1] >= 0.0,
            prefix_relations.edge_presence_logits >= 0.0,
        ))

        teacher_records = construct_v2_predicted_node_tensors(
            teacher_node_ids,
            teacher_retained,
            full.profile_family_logits[:, :1],
            full.raw_profile_parameters[:, :1],
            full.non_profile_geometry[:, :1],
        )
        autonomous_records = construct_v2_predicted_node_tensors(
            autonomous_node_ids,
            autonomous_retained,
            prefix.profile_family_logits,
            prefix.raw_profile_parameters,
            prefix.non_profile_geometry,
        )
        for teacher_tensor, autonomous_tensor in (
            (
                teacher_records.node_type_ids,
                autonomous_records.node_type_ids,
            ),
            (
                teacher_records.categorical_ids,
                autonomous_records.categorical_ids,
            ),
            (
                teacher_records.family_ids,
                autonomous_records.family_ids,
            ),
            (
                teacher_records.geometry_mask,
                autonomous_records.geometry_mask,
            ),
        ):
            self.assertTrue(torch.equal(
                teacher_tensor, autonomous_tensor
            ))
        teacher_feedback = torch.cat(
            (
                teacher_records.node_type_ids.unsqueeze(-1),
                teacher_records.categorical_ids,
            ),
            dim=-1,
        )
        autonomous_feedback = torch.cat(
            (
                autonomous_records.node_type_ids.unsqueeze(-1),
                autonomous_records.categorical_ids,
            ),
            dim=-1,
        )
        self.assertTrue(torch.equal(
            teacher_feedback, autonomous_feedback
        ))
        torch.testing.assert_close(
            autonomous_records.constrained_parameters,
            teacher_records.constrained_parameters,
            rtol=1e-5,
            atol=2e-7,
        )
        torch.testing.assert_close(
            autonomous_records.geometry,
            teacher_records.geometry,
            rtol=1e-5,
            atol=2e-7,
        )
        operation_ids = torch.tensor(
            (
                NODE_TYPES.id("extrude"),
                NODE_TYPES.id("revolve"),
            )
        )
        teacher_operation_count = (
            teacher_node_ids.unsqueeze(-1) == operation_ids
        ).any(dim=-1).sum(dim=-1)
        autonomous_operation_count = (
            autonomous_node_ids.unsqueeze(-1) == operation_ids
        ).any(dim=-1).sum(dim=-1)
        self.assertTrue(torch.equal(
            teacher_operation_count, autonomous_operation_count
        ))

        first_full = replace(
            full,
            decoded_states=full.decoded_states[:, :1],
            node_type_logits=full.node_type_logits[:, :1],
            categorical_logits=tuple(
                logits[:, :1] for logits in full.categorical_logits
            ),
            profile_family_logits=full.profile_family_logits[:, :1],
            raw_profile_parameters=full.raw_profile_parameters[:, :1],
            constrained_profile_parameters=(
                full.constrained_profile_parameters[:, :1]
            ),
            non_profile_geometry=full.non_profile_geometry[:, :1],
            non_profile_geometry_scattered=(
                full.non_profile_geometry_scattered[:, :1]
            ),
            non_profile_geometry_mask=(
                full.non_profile_geometry_mask[:, :1]
            ),
            training_primitive_type_ids=(
                full.training_primitive_type_ids[:, :1]
            ),
            training_profile_geometry=(
                full.training_profile_geometry[:, :1]
            ),
            training_profile_geometry_mask=(
                full.training_profile_geometry_mask[:, :1]
            ),
            training_geometry=full.training_geometry[:, :1],
            training_geometry_mask=full.training_geometry_mask[:, :1],
            edge_presence_logits=full.edge_presence_logits[:, :1, :1],
            edge_type_logits=full.edge_type_logits[:, :1, :1],
            operation_pointer_logits=full.operation_pointer_logits[:, :, :1],
        )
        first_prefix = replace(
            first_full,
            decoded_states=prefix.decoded_states,
            node_type_logits=prefix.node_type_logits,
            categorical_logits=prefix.categorical_logits,
            profile_family_logits=prefix.profile_family_logits,
            raw_profile_parameters=prefix.raw_profile_parameters,
            non_profile_geometry=prefix.non_profile_geometry,
            edge_presence_logits=prefix_relations.edge_presence_logits,
            edge_type_logits=prefix_relations.edge_type_logits,
            operation_pointer_logits=(
                prefix_relations.operation_pointer_logits
            ),
        )
        teacher_predictions = v2_teacher_forced_predictions(
            first_full,
            node_mask=one_node_mask,
            node_count_source="bos_equivalence",
        )
        autonomous_predictions = v2_teacher_forced_predictions(
            first_prefix,
            node_mask=one_node_mask,
            node_count_source="bos_equivalence",
        )
        for teacher_prediction, autonomous_prediction in zip(
            teacher_predictions, autonomous_predictions
        ):
            teacher_result = (
                validate_and_convert_v2_teacher_forced_prediction(
                    teacher_prediction,
                    max_operations=self.config.max_operations,
                )
            )
            autonomous_result = (
                validate_and_convert_v2_teacher_forced_prediction(
                    autonomous_prediction,
                    max_operations=self.config.max_operations,
                )
            )
            self.assertEqual(
                teacher_result.raw_integrity,
                autonomous_result.raw_integrity,
            )
            self.assertEqual(
                teacher_result.reconstruction_target,
                autonomous_result.reconstruction_target,
            )
            self.assertEqual(
                teacher_result.controlled_domain,
                autonomous_result.controlled_domain,
            )
            self.assertEqual(
                (
                    None
                    if teacher_result.primary_failure is None
                    else teacher_result.primary_failure.code
                ),
                (
                    None
                    if autonomous_result.primary_failure is None
                    else autonomous_result.primary_failure.code
                ),
            )
            self.assertEqual(
                tuple(
                    failure.code
                    for failure in teacher_result.secondary_failures
                ),
                tuple(
                    failure.code
                    for failure in autonomous_result.secondary_failures
                ),
            )


if __name__ == "__main__":
    unittest.main()
