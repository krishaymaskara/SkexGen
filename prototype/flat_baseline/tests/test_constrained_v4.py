"""Focused V4 node-conditioned categorical decoder tests."""

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
from prototype.controlled_data.factors import PrimitiveFamily
from prototype.model_data.adapters import adapt_flat_mixed
from prototype.model_data.batching import collate_flat
from prototype.model_data.canonical import canonical_nodes_and_edges, reconstruction_target
from prototype.model_data.tests.fixtures import source
from prototype.model_data.vocab import (
    BOOLEAN_MODES, DIRECTIONS, LOOP_ROLES, NODE_TYPES,
    OPERATION_TYPES, REFERENCE_PLANES,
)
from prototype.node_conditioned_categories import (
    ALL_VALID_NODE_TYPE_IDS,
    NODE_TYPE_ID_TO_SEMANTIC_NAME,
    SEMANTIC_NAME_TO_NODE_TYPE_ID,
    NodeConditionedCategoricalError,
    V4_CATEGORICAL_CONTRACT_ID,
    V4_RETAINED_CATEGORICAL_FIELD_ORDER,
    V4_RETAINED_CATEGORICAL_FIELDS,
    V4_NODE_TYPE_APPLICABILITY,
    validate_node_conditioned_categorical_row,
    v4_categorical_contract_metadata,
)
from prototype.representation.model import GeometryEncoding
from prototype.flat_baseline.constrained_v3_config import ConstrainedProfileV3Config
from prototype.flat_baseline.constrained_v4_config import (
    CONSTRAINED_PROFILE_CHECKPOINT_VERSION,
    CONSTRAINED_PROFILE_DECODER_CONTRACT_VERSION,
    CONSTRAINED_PROFILE_MODEL_CONFIG_VERSION,
    CONSTRAINED_PROFILE_MODEL_NAME,
    V4_LEARNED_GEOMETRY_CHANNEL_INDICES,
    V4_BASE_GEOMETRY_CONTRACT_ID,
    ConstrainedProfileV4Config,
)

TORCH_REASON = "real PyTorch execution is deferred to the authoritative environment"


def _batch():
    history = build_history(
        source("ER", PrimitiveFamily.CAPSULE_LINE_ARC, extents=(1.0, 2.0)),
        GeometryEncoding.CONTINUOUS,
    )
    nodes, edges = canonical_nodes_and_edges(history)
    target = reconstruction_target(nodes, edges, history.structure.operation_sequence)
    example = adapt_flat_mixed(SimpleNamespace(
        physical_family_id="v4-contract", nodes=nodes, target=target
    ))
    return collate_flat((example,))


class V4CategoricalStaticTests(unittest.TestCase):
    def test_exact_identity_geometry_and_field_order(self):
        config = ConstrainedProfileV4Config()
        config.validate()
        self.assertEqual(
            CONSTRAINED_PROFILE_MODEL_NAME,
            "B0-FLAT-CONSTRAINED-PROFILE-CANONICAL-PLANE-"
            "NODE-CONDITIONED-CATEGORIES-v4",
        )
        self.assertEqual(
            (CONSTRAINED_PROFILE_CHECKPOINT_VERSION,
             CONSTRAINED_PROFILE_MODEL_CONFIG_VERSION,
             CONSTRAINED_PROFILE_DECODER_CONTRACT_VERSION),
            (4, 4, 4),
        )
        self.assertEqual(V4_LEARNED_GEOMETRY_CHANNEL_INDICES, tuple(range(33, 39)))
        self.assertEqual(
            config.base_geometry_contract_id,
            V4_BASE_GEOMETRY_CONTRACT_ID,
        )
        self.assertEqual(
            V4_RETAINED_CATEGORICAL_FIELD_ORDER,
            ("operation_type", "boolean_mode", "direction", "reference_plane", "loop_role"),
        )
        self.assertEqual(config.categorical_selection_contract_id, V4_CATEGORICAL_CONTRACT_ID)
        self.assertEqual(ConstrainedProfileV3Config().checkpoint_version, 3)
        self.assertNotEqual(config.model_name, ConstrainedProfileV3Config().model_name)

    def test_exact_class_orders_sentinels_and_applicability(self):
        vocabularies = (
            OPERATION_TYPES, BOOLEAN_MODES, DIRECTIONS, REFERENCE_PLANES, LOOP_ROLES
        )
        expected_nodes = (
            ("extrude", "revolve"),
            ("extrude", "revolve"),
            ("extrude", "revolve"),
            ("reference_plane",),
            ("sketch",),
        )
        for field, vocab, nodes in zip(
            V4_RETAINED_CATEGORICAL_FIELDS, vocabularies, expected_nodes
        ):
            self.assertEqual(field.class_order, vocab.tokens)
            self.assertEqual(field.sentinel_id, vocab.id(None))
            self.assertEqual(field.padding_id, vocab.pad_id)
            self.assertEqual(field.padding_behavior, "emit_non_applicable_sentinel")
            self.assertEqual(field.applicable_node_types, nodes)
            for node in nodes:
                self.assertTrue(field.valid_ids(node))
        self.assertEqual(
            V4_RETAINED_CATEGORICAL_FIELDS[0].valid_ids("extrude"),
            (OPERATION_TYPES.id("extrude"),),
        )
        self.assertEqual(
            V4_RETAINED_CATEGORICAL_FIELDS[4].valid_ids("sketch"),
            (LOOP_ROLES.id("outer"),),
        )

    def test_every_supported_node_has_complete_selection(self):
        authoritative_ids = set(range(len(NODE_TYPES.tokens)))
        self.assertEqual(set(ALL_VALID_NODE_TYPE_IDS), authoritative_ids)
        self.assertEqual(set(NODE_TYPE_ID_TO_SEMANTIC_NAME), authoritative_ids)
        self.assertEqual(
            set(SEMANTIC_NAME_TO_NODE_TYPE_ID), set(NODE_TYPES.tokens)
        )
        self.assertEqual(
            tuple(name for name, unused in V4_NODE_TYPE_APPLICABILITY),
            NODE_TYPES.tokens,
        )
        sentinel = tuple(field.sentinel_id for field in V4_RETAINED_CATEGORICAL_FIELDS)
        for node in NODE_TYPES.tokens:
            row = list(sentinel)
            for index, field in enumerate(V4_RETAINED_CATEGORICAL_FIELDS):
                valid = field.valid_ids(node)
                if valid:
                    row[index] = valid[0]
            validate_node_conditioned_categorical_row(NODE_TYPES.id(node), row)

    def test_controlled_builder_tensorization_agrees_with_contract(self):
        batch = _batch()
        for node_id, attributes in zip(
            batch.target.node_type_ids[0],
            batch.target.categorical_attributes[0],
        ):
            retained = tuple(attributes[index] for index in (0, 1, 2, 3, 8))
            validate_node_conditioned_categorical_row(node_id, retained)

    def test_contract_metadata_and_config_are_frozen(self):
        self.assertEqual(
            ConstrainedProfileV4Config().categorical_selection_contract,
            v4_categorical_contract_metadata(),
        )
        with self.assertRaises(ValueError):
            replace(
                ConstrainedProfileV4Config(),
                categorical_selection_contract_id="wrong",
            ).validate()

    def test_python_38_grammar(self):
        root = Path(__file__).resolve().parents[2]
        paths = (
            root / "node_conditioned_categories.py",
            root / "node_conditioned_categories_torch.py",
            Path(__file__).resolve().parents[1] / "constrained_v4.py",
            Path(__file__).resolve().parents[1] / "constrained_v4_conversion.py",
            Path(__file__).resolve().parents[1] / "constrained_v4_autonomous.py",
        )
        for path in paths:
            ast.parse(path.read_text(encoding="utf-8"), str(path), feature_version=(3, 8))


@unittest.skipIf(torch is None, TORCH_REASON)
class V4CategoricalTensorTests(unittest.TestCase):
    def _logits(self, shape=(1, 6)):
        result = []
        for field in V4_RETAINED_CATEGORICAL_FIELDS:
            result.append(torch.zeros(shape + (len(field.class_order),)))
        return tuple(result)

    def test_applicable_sentinel_excluded_and_next_valid_selected(self):
        from prototype.node_conditioned_categories_torch import select_node_conditioned_categorical_ids
        logits = list(self._logits((1, 1)))
        logits[3][..., REFERENCE_PLANES.id(None)] = 10
        logits[3][..., REFERENCE_PLANES.id("XZ")] = 9
        result = select_node_conditioned_categorical_ids(
            tuple(logits), torch.tensor([[NODE_TYPES.id("reference_plane")]])
        )
        self.assertEqual(result.raw_categorical_argmax_ids[0, 0, 3], REFERENCE_PLANES.id(None))
        self.assertEqual(result.node_conditioned_categorical_ids[0, 0, 3], REFERENCE_PLANES.id("XZ"))
        self.assertTrue(result.correction_mask[0, 0, 3])

    def test_inapplicable_real_becomes_sentinel_and_valid_stays(self):
        from prototype.node_conditioned_categories_torch import select_node_conditioned_categorical_ids
        logits = list(self._logits((1, 2)))
        logits[0][0, :, OPERATION_TYPES.id("extrude")] = 5
        nodes = torch.tensor([[NODE_TYPES.id("profile"), NODE_TYPES.id("extrude")]])
        result = select_node_conditioned_categorical_ids(tuple(logits), nodes)
        self.assertEqual(result.node_conditioned_categorical_ids[0, 0, 0], OPERATION_TYPES.id(None))
        self.assertEqual(result.node_conditioned_categorical_ids[0, 1, 0], OPERATION_TYPES.id("extrude"))

    def test_padding_mixed_types_ties_dtype_device_and_contiguity(self):
        from prototype.node_conditioned_categories_torch import select_node_conditioned_categorical_ids
        logits = self._logits((1, 6))
        nodes = torch.tensor([[NODE_TYPES.id(name) for name in (
            "reference_plane", "sketch", "profile", "axis", "extrude", "revolve"
        )]])
        mask = torch.tensor([[True, True, True, True, True, False]])
        result = select_node_conditioned_categorical_ids(logits, nodes, mask)
        self.assertEqual(result.node_conditioned_categorical_ids.dtype, torch.long)
        self.assertEqual(result.node_conditioned_categorical_ids.device, nodes.device)
        self.assertTrue(result.node_conditioned_categorical_ids.is_contiguous())
        self.assertEqual(
            result.node_conditioned_categorical_ids[0, -1].tolist(),
            [field.sentinel_id for field in V4_RETAINED_CATEGORICAL_FIELDS],
        )

    def test_nonfinite_shapes_and_node_types_fail_structured(self):
        from prototype.node_conditioned_categories_torch import select_node_conditioned_categorical_ids
        logits = list(self._logits((1, 1)))
        logits[0][0, 0, 0] = float("nan")
        with self.assertRaises(NodeConditionedCategoricalError) as caught:
            select_node_conditioned_categorical_ids(tuple(logits), torch.tensor([[2]]))
        self.assertEqual(caught.exception.code, "invalid_v4_categorical_logits")
        for invalid_id in (-1, len(NODE_TYPES.tokens), len(NODE_TYPES.tokens) + 1):
            with self.subTest(invalid_id=invalid_id):
                with self.assertRaises(NodeConditionedCategoricalError) as caught:
                    select_node_conditioned_categorical_ids(
                        self._logits((1, 1)), torch.tensor([[invalid_id]])
                    )
                self.assertEqual(caught.exception.code, "invalid_predicted_node_type")

    def test_job_3334551_node_type_id_1_regression(self):
        from prototype.node_conditioned_categories_torch import select_node_conditioned_categorical_ids
        self.assertEqual(NODE_TYPES.tokens[1], "<none>")
        nodes = torch.full((16,), 1, dtype=torch.long)
        logits = []
        for field in V4_RETAINED_CATEGORICAL_FIELDS:
            values = torch.zeros(16, len(field.class_order))
            favored = len(field.class_order) - 1
            values[:, favored] = 9.0
            logits.append(values)
        result = select_node_conditioned_categorical_ids(tuple(logits), nodes)
        expected = torch.ones(16, 5, dtype=torch.long)
        self.assertTrue(torch.equal(result.node_conditioned_categorical_ids, expected))
        self.assertEqual(result.node_conditioned_categorical_ids.shape, (16, 5))
        self.assertTrue(result.correction_mask.all())
        for row in result.node_conditioned_categorical_ids.tolist():
            validate_node_conditioned_categorical_row(1, row)

    def test_every_authoritative_node_id_selects_complete_row(self):
        from prototype.node_conditioned_categories_torch import select_node_conditioned_categorical_ids
        nodes = torch.tensor([list(ALL_VALID_NODE_TYPE_IDS)], dtype=torch.long)
        logits = []
        for field in V4_RETAINED_CATEGORICAL_FIELDS:
            values = torch.zeros(1, len(ALL_VALID_NODE_TYPE_IDS), len(field.class_order))
            values[..., field.padding_id] = 20.0
            values[..., field.sentinel_id] = 19.0
            values[..., -1] = 18.0
            logits.append(values)
        result = select_node_conditioned_categorical_ids(tuple(logits), nodes)
        for node_id, row in zip(nodes[0].tolist(), result.node_conditioned_categorical_ids[0].tolist()):
            validate_node_conditioned_categorical_row(node_id, row)

    def test_v3_v4_parameter_count_and_loss_equivalence(self):
        from prototype.constrained_profile_decoder import profile_targets_for_loss
        from prototype.flat_baseline.constrained_v3 import ConstrainedProfileV3Model
        from prototype.flat_baseline.constrained_v3_losses import constrained_profile_v3_loss
        from prototype.flat_baseline.constrained_v4 import ConstrainedProfileV4Model
        from prototype.flat_baseline.constrained_v4_losses import constrained_profile_v4_loss
        batch = _batch()
        inputs = batch.to_torch(torch)
        target = batch.target.to_torch(torch)
        profiles = profile_targets_for_loss(batch.target, inputs["geometry"])
        torch.manual_seed(7)
        v3 = ConstrainedProfileV3Model(ConstrainedProfileV3Config())
        torch.manual_seed(7)
        v4 = ConstrainedProfileV4Model(ConstrainedProfileV4Config())
        self.assertEqual(sum(p.numel() for p in v3.parameters()), sum(p.numel() for p in v4.parameters()))
        v4.load_state_dict(v3.state_dict(), strict=True)
        v3.eval()
        v4.eval()
        out3 = v3(target=target, profile_targets=profiles, **inputs)
        out4 = v4(target=target, profile_targets=profiles, **inputs)
        loss3 = constrained_profile_v3_loss(out3, target, profiles, v3.config)
        loss4 = constrained_profile_v4_loss(out4, target, profiles, v4.config)
        for name in loss3.as_dict():
            torch.testing.assert_close(loss3.as_dict()[name], loss4.as_dict()[name])

    def test_teacher_conversion_preserves_evidence_and_uses_conditioned_ids(self):
        from prototype.flat_baseline.constrained_v4_conversion import construct_v4_predicted_node_tensors
        logits = list(self._logits((1,)))
        logits[3][0, REFERENCE_PLANES.id(None)] = 5
        logits[3][0, REFERENCE_PLANES.id("YZ")] = 4
        records = construct_v4_predicted_node_tensors(
            torch.tensor([NODE_TYPES.id("reference_plane")]), tuple(logits),
            torch.zeros(1, 3), torch.zeros(1, 3), torch.zeros(1, 6),
        )
        self.assertEqual(records.raw_categorical_argmax_ids[0, 3], REFERENCE_PLANES.id(None))
        self.assertEqual(records.categorical_ids[0, 3], REFERENCE_PLANES.id("YZ"))
        self.assertEqual(records.geometry[0, :9].tolist(), [0, 0, 0, 0, 1, 0, 0, 0, 1])

    def test_autonomous_interfaces_accept_no_targets(self):
        from prototype.flat_baseline.constrained_v4_autonomous import greedy_decode_v4, greedy_decode_v4_from_memory
        self.assertNotIn("target", inspect.signature(greedy_decode_v4).parameters)
        self.assertNotIn("target", inspect.signature(greedy_decode_v4_from_memory).parameters)

    def test_real_model_argmax_id_1_reaches_selector_and_construction(self):
        from prototype.constrained_profile_decoder import profile_targets_for_loss
        from prototype.flat_baseline.constrained_v4 import ConstrainedProfileV4Model
        from prototype.flat_baseline.constrained_v4_conversion import (
            construct_v4_predicted_node_tensors,
        )
        batch = _batch()
        inputs = batch.to_torch(torch)
        target = batch.target.to_torch(torch)
        profiles = profile_targets_for_loss(batch.target, inputs["geometry"])
        model = ConstrainedProfileV4Model(ConstrainedProfileV4Config())
        model.eval()
        with torch.no_grad():
            model.node_type_head.weight.zero_()
            model.node_type_head.bias.fill_(-10.0)
            model.node_type_head.bias[1] = 10.0
            output = model(target=target, profile_targets=profiles, **inputs)
            predicted = output.node_type_logits.argmax(dim=-1)
            self.assertTrue(torch.equal(predicted, torch.ones_like(predicted)))
            records = construct_v4_predicted_node_tensors(
                predicted,
                output.categorical_logits,
                output.profile_family_logits,
                output.raw_profile_parameters,
                output.remaining_geometry,
                target["node_mask"],
            )
        self.assertEqual(records.node_conditioned_categorical_ids.shape, predicted.shape + (5,))
        self.assertEqual(records.node_conditioned_categorical_ids.device, predicted.device)
        selected = records.node_conditioned_categorical_ids[target["node_mask"]]
        self.assertTrue(torch.equal(selected, torch.ones_like(selected)))

    def test_autonomous_first_step_id_1_has_no_target_dependency(self):
        from prototype.flat_baseline.constrained_v4_conversion import construct_v4_predicted_node_tensors
        logits = self._logits((1,))
        records = construct_v4_predicted_node_tensors(
            torch.tensor([1]), logits, torch.zeros(1, 3),
            torch.zeros(1, 3), torch.zeros(1, 6),
        )
        self.assertEqual(records.node_conditioned_categorical_ids.tolist(), [[1, 1, 1, 1, 1]])
        self.assertFalse(records.geometry_mask.any())
