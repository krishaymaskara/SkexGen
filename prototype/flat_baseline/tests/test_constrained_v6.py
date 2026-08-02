"""Focused prefix-conditioned node grammar V6 tests."""

from __future__ import annotations

import ast
from dataclasses import replace
import inspect
import itertools
from pathlib import Path
from types import SimpleNamespace
import unittest

try:
    import torch
except ImportError:
    torch = None

from prototype.controlled_data.builders import build_history
from prototype.controlled_data.factors import PrimitiveFamily
from prototype.controlled_data.factors import ReferencePlane
from prototype.axis_geometry import (
    AXIS_CONSTRUCTION_ALGORITHM_ID,
    AXIS_COORDINATE_SPACE,
    AXIS_GEOMETRY_CHANNEL_INDICES,
    AXIS_GEOMETRY_CONTRACT_ID,
    CANONICAL_AXIS_CHANNELS,
    CANONICAL_AXIS_MASK,
    AxisGeometryContractError,
    construct_canonical_axis_geometry,
    validate_canonical_axis_geometry,
)
from prototype.model_data.canonical import canonical_nodes_and_edges, reconstruction_target
from prototype.model_data.adapters import adapt_flat_mixed
from prototype.model_data.batching import collate_flat
from prototype.model_data.tests.fixtures import source
from prototype.model_data.vocab import NODE_TYPES
from prototype.node_grammar import (
    COMPLETION_ALGORITHM_ID,
    NODE_GRAMMAR_CONTRACT_ID,
    NodeGrammarError,
    VALID_REQUESTED_NODE_COUNTS,
    V5_NODE_GRAMMAR,
    enumerate_complete_node_sequences,
    legal_next_node_ids,
    validate_complete_node_sequence,
)
from prototype.representation.model import BooleanMode, Direction, GeometryEncoding


TORCH_REASON = "real PyTorch execution is deferred to the authoritative environment"


def _independent_builder_sequences():
    result = set()
    for template in ("E", "R", "EE", "ER", "RE", "RR"):
        history = build_history(
            source(
                template,
                PrimitiveFamily.CIRCLE,
                extents=(1.0,) * len(template),
            ),
            GeometryEncoding.CONTINUOUS,
        )
        nodes, unused_edges = canonical_nodes_and_edges(history)
        del unused_edges
        result.add(tuple(node.node_type for node in nodes))
    return result


def _batch(template="ER"):
    history = build_history(
        source(
            template,
            PrimitiveFamily.CAPSULE_LINE_ARC,
            extents=tuple(float(index + 1) for index in range(len(template))),
        ),
        GeometryEncoding.CONTINUOUS,
    )
    nodes, edges = canonical_nodes_and_edges(history)
    target = reconstruction_target(nodes, edges, history.structure.operation_sequence)
    example = adapt_flat_mixed(SimpleNamespace(
        physical_family_id="v6-test", nodes=nodes, target=target
    ))
    return collate_flat((example,))


class V6GrammarStaticTests(unittest.TestCase):
    def test_exact_v6_identity_and_immutable_contract(self):
        from prototype.flat_baseline.constrained_v6_config import (
            CONSTRAINED_PROFILE_CHECKPOINT_VERSION,
            CONSTRAINED_PROFILE_DECODER_CONTRACT_VERSION,
            CONSTRAINED_PROFILE_MODEL_CONFIG_VERSION,
            CONSTRAINED_PROFILE_MODEL_NAME,
            ConstrainedProfileV6Config,
        )
        config = ConstrainedProfileV6Config()
        config.validate()
        self.assertEqual(
            CONSTRAINED_PROFILE_MODEL_NAME,
            "B0-FLAT-CONSTRAINED-PROFILE-CANONICAL-PLANE-"
            "NODE-CATEGORIES-PREFIX-GRAMMAR-CONSTRAINED-AXIS-v6",
        )
        self.assertEqual(
            (CONSTRAINED_PROFILE_CHECKPOINT_VERSION,
             CONSTRAINED_PROFILE_MODEL_CONFIG_VERSION,
             CONSTRAINED_PROFILE_DECODER_CONTRACT_VERSION),
            (6, 6, 6),
        )
        self.assertEqual(config.node_grammar_contract_id, NODE_GRAMMAR_CONTRACT_ID)
        self.assertEqual(config.completion_algorithm_id, COMPLETION_ALGORITHM_ID)
        self.assertEqual(config.valid_requested_node_counts, (4, 5, 7, 8, 9))
        self.assertEqual(V5_NODE_GRAMMAR.node_vocabulary, NODE_TYPES.tokens)
        self.assertEqual(V5_NODE_GRAMMAR.to_json(), V5_NODE_GRAMMAR.to_json())
        self.assertEqual(config.learned_geometry_channel_indices, tuple(range(33, 39)))
        self.assertEqual(config.axis_geometry_contract_id, AXIS_GEOMETRY_CONTRACT_ID)
        self.assertEqual(
            config.axis_construction_algorithm_id,
            AXIS_CONSTRUCTION_ALGORITHM_ID,
        )

    def test_axis_contract_exact_channels_scaling_and_builder_domain(self):
        from prototype.model_data.geometry import denormalize_applicable_geometry
        canonical = construct_canonical_axis_geometry()
        self.assertEqual(AXIS_GEOMETRY_CHANNEL_INDICES, (33, 34, 35, 36, 37, 38))
        self.assertEqual(CANONICAL_AXIS_CHANNELS, (0.0, 0.0, 0.0, 1.0, 0.0, 0.0))
        self.assertEqual(CANONICAL_AXIS_MASK, (True, True, True, True, False, False))
        self.assertEqual(AXIS_COORDINATE_SPACE, "sketch_local_2d")
        full_geometry = (0.0,) * 33 + CANONICAL_AXIS_CHANNELS
        full_mask = (False,) * 33 + CANONICAL_AXIS_MASK
        physical = denormalize_applicable_geometry(full_geometry, full_mask)
        self.assertEqual(physical[33:37], (0.0, 0.0, 0.0, 1.0))
        self.assertEqual(physical[37:39], (None, None))
        self.assertEqual(
            validate_canonical_axis_geometry(
                canonical.normalized_channels, canonical.geometry_mask
            ),
            canonical,
        )
        observed = set()
        for template in ("E", "R", "EE", "ER", "RE", "RR"):
            for primitive in PrimitiveFamily:
                for plane in ReferencePlane:
                    directions_by_operation = itertools.product(
                        tuple(Direction), repeat=len(template)
                    )
                    for directions in directions_by_operation:
                        modes = (
                            (BooleanMode.JOIN, BooleanMode.CUT)
                            if len(template) == 2 else (BooleanMode.JOIN,)
                        )
                        for mode in modes:
                            for encoding in (
                                GeometryEncoding.CONTINUOUS,
                                GeometryEncoding.QUANTIZED,
                            ):
                                history = build_history(
                                    source(
                                        template,
                                        primitive,
                                        plane=plane,
                                        extents=(1.0,) * len(template),
                                        directions=directions,
                                        mode=mode,
                                    ),
                                    encoding,
                                )
                                nodes, edges = canonical_nodes_and_edges(history)
                                target = reconstruction_target(
                                    nodes, edges, history.structure.operation_sequence
                                )
                                for node, geometry, mask in zip(
                                    nodes, target.geometry, target.geometry_mask
                                ):
                                    if node.node_type == "axis":
                                        observed.add((geometry[33:39], mask[33:39]))
        self.assertEqual(
            observed,
            {(CANONICAL_AXIS_CHANNELS, CANONICAL_AXIS_MASK)},
        )
        for values, mask in (
            ((0.1, 0.0, 0.0, 1.0, 0.0, 0.0), CANONICAL_AXIS_MASK),
            ((0.0, 0.0, 1.0, 0.0, 0.0, 0.0), CANONICAL_AXIS_MASK),
            ((0.0, 0.0, 0.0, float("nan"), 0.0, 0.0), CANONICAL_AXIS_MASK),
            (CANONICAL_AXIS_CHANNELS, (True, True, True, False, False, False)),
            (CANONICAL_AXIS_CHANNELS, (1, True, True, True, False, False)),
        ):
            with self.subTest(values=values, mask=mask):
                with self.assertRaises(AxisGeometryContractError):
                    validate_canonical_axis_geometry(values, mask)

    def test_grammar_language_equals_independent_builder_language(self):
        grammar = set(enumerate_complete_node_sequences())
        independent = _independent_builder_sequences()
        self.assertEqual(grammar, independent)
        self.assertEqual(tuple(sorted(set(map(len, grammar)))), (4, 5, 7, 8, 9))
        self.assertEqual(len(grammar), 6)

    def test_every_valid_prefix_has_exact_feasible_authoritative_next(self):
        for sequence in enumerate_complete_node_sequences():
            ids = tuple(NODE_TYPES.id(item) for item in sequence)
            for position, authoritative in enumerate(ids):
                legal = legal_next_node_ids(ids[:position], len(ids))
                self.assertIn(authoritative, legal)
            self.assertEqual(validate_complete_node_sequence(ids, len(ids)), sequence)
            with self.assertRaises(NodeGrammarError) as caught:
                legal_next_node_ids(ids, len(ids))
            self.assertEqual(caught.exception.code, "no_valid_node_grammar_continuation")

    def test_exact_length_feasibility_filters_operation_paths(self):
        profile = tuple(NODE_TYPES.id(item) for item in (
            "reference_plane", "sketch", "profile"
        ))
        axis = NODE_TYPES.id("axis")
        extrude = NODE_TYPES.id("extrude")
        self.assertEqual(legal_next_node_ids(profile, 7), (extrude,))
        self.assertEqual(set(legal_next_node_ids(profile, 8)), {axis, extrude})
        self.assertEqual(legal_next_node_ids(profile, 9), (axis,))
        for count in (3, 6, 10, True, None):
            with self.subTest(count=count):
                with self.assertRaises(NodeGrammarError) as caught:
                    legal_next_node_ids((), count)
                self.assertEqual(caught.exception.code, "invalid_v5_requested_node_count")
        impossible = profile + (axis,)
        with self.assertRaises(NodeGrammarError) as caught:
            legal_next_node_ids(impossible, 4)
        self.assertEqual(
            caught.exception.code, "no_valid_node_grammar_continuation"
        )
        self.assertEqual(caught.exception.context["requested_node_count"], 4)
        self.assertEqual(caught.exception.context["current_position"], 4)
        self.assertEqual(caught.exception.context["remaining_positions"], 0)
        self.assertEqual(
            caught.exception.context[
                "legal_transition_candidates_before_completion_filtering"
            ],
            [NODE_TYPES.id("revolve")],
        )

    def test_malformed_prefixes_fail_structured(self):
        malformed = (
            ("sketch",),
            ("reference_plane", "reference_plane"),
            ("reference_plane", "profile"),
            ("reference_plane", "sketch", "axis"),
            ("reference_plane", "sketch", "profile", "revolve"),
            ("reference_plane", "sketch", "profile", "extrude", "axis"),
            ("reference_plane", "sketch", "profile", "axis", "extrude"),
            (
                "reference_plane", "sketch", "profile", "extrude",
                "sketch", "profile", "extrude", "sketch",
            ),
        )
        for tokens in malformed:
            with self.subTest(tokens=tokens):
                ids = tuple(NODE_TYPES.id(item) for item in tokens)
                with self.assertRaises(NodeGrammarError) as caught:
                    legal_next_node_ids(ids, 8)
                self.assertIn(
                    caught.exception.code,
                    ("invalid_v5_generated_prefix", "no_valid_node_grammar_continuation"),
                )
        for sentinel in (NODE_TYPES.pad_id, NODE_TYPES.id(None)):
            with self.assertRaises(NodeGrammarError) as caught:
                legal_next_node_ids((sentinel,), 4)
            self.assertEqual(caught.exception.code, "invalid_v5_generated_prefix")

    def test_python_38_grammar(self):
        root = Path(__file__).resolve().parents[2]
        paths = (
            root / "node_grammar.py",
            root / "node_grammar_torch.py",
            Path(__file__).resolve().parents[1] / "constrained_v6.py",
            Path(__file__).resolve().parents[1] / "constrained_v6_conversion.py",
            Path(__file__).resolve().parents[1] / "constrained_v6_autonomous.py",
        )
        for path in paths:
            ast.parse(path.read_text(), str(path), feature_version=(3, 8))


@unittest.skipIf(torch is None, TORCH_REASON)
class V6GrammarTensorTests(unittest.TestCase):
    def _authoritative_v6_prediction(self, template):
        from prototype.flat_baseline.autonomous import TEACHER_FORCED_PREFIX_FEEDBACK
        from prototype.flat_baseline.constrained_v6_conversion import (
            V6TeacherForcedRawPrediction,
        )
        from prototype.flat_baseline.tests.test_conversion import _raw_from_target
        from prototype.node_conditioned_categories import V4_CATEGORICAL_CONTRACT_ID
        history = build_history(
            source(
                template,
                PrimitiveFamily.CIRCLE,
                extents=(1.0,) * len(template),
            ),
            GeometryEncoding.CONTINUOUS,
        )
        nodes, edges = canonical_nodes_and_edges(history)
        target = reconstruction_target(
            nodes, edges, history.structure.operation_sequence
        )
        base = replace(
            _raw_from_target(target),
            prefix_feedback=TEACHER_FORCED_PREFIX_FEEDBACK,
        )
        node_ids = tuple(target.node_type_ids)
        raw_categories = tuple(
            tuple(row[index] for index in (0, 1, 2, 3, 8))
            for row in target.categorical_attributes
        )
        legal_masks = []
        evidence = []
        for position in range(len(node_ids)):
            legal = legal_next_node_ids(node_ids[:position], len(node_ids))
            mask = tuple(index in legal for index in range(len(NODE_TYPES.tokens)))
            legal_masks.append(mask)
            evidence.append({
                "state": "authoritative-test",
                "requested_node_count": len(node_ids),
            })
        return V6TeacherForcedRawPrediction(
            **vars(base),
            profile_family_logits=((0.0, 0.0, 0.0),) * len(node_ids),
            predicted_profile_family_ids=(-1,) * len(node_ids),
            raw_profile_parameters=((0.0, 0.0, 0.0),) * len(node_ids),
            constrained_profile_parameters=((0.0, 0.0, 0.0),) * len(node_ids),
            raw_categorical_argmax_ids=raw_categories,
            node_conditioned_categorical_ids=raw_categories,
            categorical_correction_mask=((False,) * 5,) * len(node_ids),
            categorical_contract_id=V4_CATEGORICAL_CONTRACT_ID,
            raw_node_type_argmax_ids=node_ids,
            grammar_constrained_node_type_ids=node_ids,
            node_type_correction_mask=(False,) * len(node_ids),
            legal_node_type_masks=tuple(legal_masks),
            grammar_state_evidence=tuple(evidence),
            same_history_raw_shadow_nodes=base.raw_nodes,
            same_history_raw_shadow_operation_node_indices=(
                base.predicted_operation_node_indices
            ),
            same_history_raw_shadow_operation_count=base.predicted_operation_count,
            same_history_raw_shadow_operation_count_exceeds_limit=(
                base.operation_count_exceeds_limit
            ),
            same_history_raw_shadow_operation_pointers=base.raw_operation_pointers,
            node_grammar_contract_id=NODE_GRAMMAR_CONTRACT_ID,
            raw_axis_geometry=tuple(
                tuple(row[33:39]) for row in target.geometry
            ),
            constrained_axis_geometry=tuple(
                tuple(row[33:39])
                if node_id == NODE_TYPES.id("axis") else (0.0,) * 6
                for node_id, row in zip(node_ids, target.geometry)
            ),
            axis_geometry_correction_mask=((False,) * 6,) * len(node_ids),
            axis_geometry_contract_id=AXIS_GEOMETRY_CONTRACT_ID,
        )

    def test_tensor_selection_sentinels_illegal_ties_mixed_and_contiguous(self):
        from prototype.node_grammar_torch import select_prefix_conditioned_node_types
        logits = torch.zeros(3, len(NODE_TYPES.tokens), dtype=torch.float32)
        logits[0, NODE_TYPES.pad_id] = 10
        logits[1, NODE_TYPES.id(None)] = 10
        logits[2, NODE_TYPES.id("axis")] = 10
        counts = torch.tensor([4, 5, 8], dtype=torch.long)
        result = select_prefix_conditioned_node_types(
            logits,
            ((), (), ()),
            counts,
            current_positions=torch.tensor([0, 0, 0]),
        )
        expected = torch.full((3,), NODE_TYPES.id("reference_plane"))
        self.assertTrue(torch.equal(result.grammar_constrained_node_type_ids, expected))
        self.assertTrue(result.node_type_correction_mask.all())
        self.assertEqual(result.legal_node_type_mask.sum(dim=-1).tolist(), [1, 1, 1])
        self.assertEqual(result.raw_node_type_argmax_ids.dtype, torch.long)
        self.assertEqual(result.raw_node_type_argmax_ids.device, logits.device)
        self.assertTrue(result.legal_node_type_mask.is_contiguous())

        profile = tuple(NODE_TYPES.id(item) for item in (
            "reference_plane", "sketch", "profile"
        ))
        tied = select_prefix_conditioned_node_types(
            torch.zeros(1, len(NODE_TYPES.tokens)),
            (profile,),
            torch.tensor([8]),
            current_positions=torch.tensor([3]),
        )
        self.assertEqual(
            tied.grammar_constrained_node_type_ids.item(),
            min(NODE_TYPES.id("axis"), NODE_TYPES.id("extrude")),
        )
        self.assertEqual(tied.legal_node_type_mask.sum().item(), 2)

    def test_valid_raw_argmax_nonfinite_prefix_and_impossible_fail(self):
        from prototype.node_grammar_torch import select_prefix_conditioned_node_types
        logits = torch.zeros(1, len(NODE_TYPES.tokens))
        logits[0, NODE_TYPES.id("reference_plane")] = 3
        result = select_prefix_conditioned_node_types(
            logits,
            ((),),
            torch.tensor([4]),
            current_positions=torch.tensor([0]),
        )
        self.assertEqual(
            result.raw_node_type_argmax_ids.item(),
            result.grammar_constrained_node_type_ids.item(),
        )
        self.assertFalse(result.node_type_correction_mask.item())
        for malformed in (
            logits.fill_(float("nan")),
            torch.zeros(1, len(NODE_TYPES.tokens) - 1),
        ):
            with self.assertRaises(NodeGrammarError):
                select_prefix_conditioned_node_types(
                    malformed,
                    ((),),
                    torch.tensor([4]),
                    current_positions=torch.tensor([0]),
                )
        with self.assertRaises(NodeGrammarError):
            select_prefix_conditioned_node_types(
                torch.zeros(1, len(NODE_TYPES.tokens)),
                ((NODE_TYPES.id("sketch"),),),
                torch.tensor([4]),
                current_positions=torch.tensor([1]),
            )

    def test_v5_v6_parameter_state_loss_gradient_and_vq_equivalence(self):
        from prototype.constrained_profile_decoder import profile_targets_for_loss
        from prototype.flat_baseline.constrained_v5 import ConstrainedProfileV5Model
        from prototype.flat_baseline.constrained_v5_config import ConstrainedProfileV5Config
        from prototype.flat_baseline.constrained_v5_losses import constrained_profile_v5_loss
        from prototype.flat_baseline.constrained_v5_conversion import (
            v5_teacher_forced_predictions,
        )
        from prototype.flat_baseline.constrained_v5_training import build_v5_optimizer
        from prototype.flat_baseline.constrained_v5_training_config import ConstrainedV5TrainingConfig
        from prototype.flat_baseline.constrained_v6 import ConstrainedProfileV6Model
        from prototype.flat_baseline.constrained_v6_config import ConstrainedProfileV6Config
        from prototype.flat_baseline.constrained_v6_losses import constrained_profile_v6_loss
        from prototype.flat_baseline.constrained_v6_conversion import (
            v6_teacher_forced_predictions,
        )
        from prototype.flat_baseline.constrained_v6_training import build_v6_optimizer
        from prototype.flat_baseline.constrained_v6_training_config import ConstrainedV6TrainingConfig
        batch = _batch()
        inputs = batch.to_torch(torch)
        target = batch.target.to_torch(torch)
        profiles = profile_targets_for_loss(batch.target, inputs["geometry"])
        torch.manual_seed(29)
        v5 = ConstrainedProfileV5Model(ConstrainedProfileV5Config())
        torch.manual_seed(29)
        v6 = ConstrainedProfileV6Model(ConstrainedProfileV6Config())
        self.assertEqual(
            sum(parameter.numel() for parameter in v5.parameters()),
            sum(parameter.numel() for parameter in v6.parameters()),
        )
        self.assertEqual(tuple(v5.state_dict()), tuple(v6.state_dict()))
        optimizer5 = build_v5_optimizer(
            v5, ConstrainedV5TrainingConfig(require_clean_source=False)
        )
        optimizer6 = build_v6_optimizer(
            v6, ConstrainedV6TrainingConfig(require_clean_source=False)
        )
        self.assertEqual(
            [len(group["params"]) for group in optimizer5.param_groups],
            [len(group["params"]) for group in optimizer6.param_groups],
        )
        v6.load_state_dict(v5.state_dict(), strict=True)
        v5.eval()
        v6.eval()
        output5 = v5(target=target, profile_targets=profiles, **inputs)
        output6 = v6(target=target, profile_targets=profiles, **inputs)
        for name in vars(output5):
            value5 = getattr(output5, name)
            value6 = getattr(output6, name)
            if torch.is_tensor(value5):
                torch.testing.assert_close(value5, value6, rtol=0, atol=0)
        prediction5 = v5_teacher_forced_predictions(
            output5, node_mask=target["node_mask"], node_count_source="test"
        )[0]
        prediction6 = v6_teacher_forced_predictions(
            output6, node_mask=target["node_mask"], node_count_source="test"
        )[0]
        self.assertEqual(prediction5.raw_edges, prediction6.raw_edges)
        self.assertEqual(
            prediction5.raw_operation_pointers,
            prediction6.raw_operation_pointers,
        )
        self.assertEqual(
            prediction5.same_history_raw_shadow_operation_pointers,
            prediction6.same_history_raw_shadow_operation_pointers,
        )
        self.assertEqual(
            prediction5.predicted_profile_family_ids,
            prediction6.predicted_profile_family_ids,
        )
        self.assertEqual(
            prediction5.constrained_profile_parameters,
            prediction6.constrained_profile_parameters,
        )
        self.assertEqual(
            tuple(node.categorical_ids for node in prediction5.raw_nodes),
            tuple(node.categorical_ids for node in prediction6.raw_nodes),
        )
        self.assertEqual(
            prediction5.same_history_raw_shadow_nodes,
            prediction6.same_history_raw_shadow_nodes,
        )
        loss5 = constrained_profile_v5_loss(
            output5, target, profiles, v5.config
        )
        loss6 = constrained_profile_v6_loss(
            output6, target, profiles, v6.config
        )
        for name, value in loss5.as_dict().items():
            torch.testing.assert_close(value, loss6.as_dict()[name], rtol=0, atol=0)
        loss5.total.backward()
        loss6.total.backward()
        for (name5, parameter5), (name6, parameter6) in zip(
            v5.named_parameters(), v6.named_parameters()
        ):
            self.assertEqual(name5, name6)
            torch.testing.assert_close(
                parameter5.grad, parameter6.grad, rtol=0, atol=0
            )
        self.assertTrue(all(
            parameter.grad is not None
            for parameter in v6.parameters()
            if parameter.requires_grad
        ))
        for name, value in v5.vq.state_dict().items():
            torch.testing.assert_close(
                value, v6.vq.state_dict()[name], rtol=0, atol=0
            )

    def test_strict_v6_conversion_accepts_all_builder_sequences_without_repair(self):
        from prototype.flat_baseline.constrained_v6_conversion import (
            validate_and_convert_v6_teacher_forced_prediction,
        )
        for template in ("E", "R", "EE", "ER", "RE", "RR"):
            with self.subTest(template=template):
                prediction = self._authoritative_v6_prediction(template)
                result = validate_and_convert_v6_teacher_forced_prediction(
                    prediction, max_operations=2
                )
                self.assertTrue(result.controlled_domain.valid)
                invalid = replace(
                    prediction,
                    grammar_constrained_node_type_ids=(
                        NODE_TYPES.id(None),
                    ) + prediction.grammar_constrained_node_type_ids[1:],
                    raw_nodes=(
                        replace(
                            prediction.raw_nodes[0],
                            node_type_id=NODE_TYPES.id(None),
                        ),
                    ) + prediction.raw_nodes[1:],
                )
                with self.assertRaises(NodeGrammarError):
                    validate_and_convert_v6_teacher_forced_prediction(
                        invalid, max_operations=2
                    )

    def test_job_3336430_autonomous_is_target_free_exact_and_grammar_valid(self):
        from prototype.flat_baseline.constrained_v6 import ConstrainedProfileV6Model
        from prototype.flat_baseline.constrained_v6_autonomous import (
            greedy_decode_v6,
            validate_and_convert_v6_autonomous_prediction,
        )
        from prototype.flat_baseline.constrained_v6_config import ConstrainedProfileV6Config
        batch = _batch("ER")
        inputs = batch.to_torch(torch)
        model = ConstrainedProfileV6Model(ConstrainedProfileV6Config())
        with torch.no_grad():
            model.node_type_head.weight.zero_()
            model.node_type_head.bias.zero_()
            model.node_type_head.bias[NODE_TYPES.id(None)] = 10
        self.assertNotIn("target", inspect.signature(greedy_decode_v6).parameters)
        before = {name: value.clone() for name, value in model.state_dict().items()}
        rng_before = torch.get_rng_state().clone()
        result = greedy_decode_v6(
            model,
            inputs,
            node_counts=torch.tensor([8]),
            node_count_source="authorized_test_length",
        )[0]
        target = batch.target.to_torch(torch)
        forbidden_target_shadow = {
            name: value.clone() for name, value in target.items()
        }
        forbidden_target_shadow["node_type_ids"].fill_(NODE_TYPES.id("revolve"))
        forbidden_target_shadow["categorical_attributes"].zero_()
        forbidden_target_shadow["geometry"].fill_(0.75)
        forbidden_target_shadow["operation_sequence"].zero_()
        self.assertFalse(torch.equal(
            forbidden_target_shadow["geometry"], target["geometry"]
        ))
        # The mutated target shadow cannot be supplied to this API; repeating
        # with the same generated context must therefore be exactly unchanged.
        repeated = greedy_decode_v6(
            model,
            inputs,
            node_counts=torch.tensor([8]),
            node_count_source="authorized_test_length",
        )[0]
        self.assertEqual(result, repeated)
        validate_complete_node_sequence(
            result.grammar_constrained_node_type_ids, 8
        )
        self.assertEqual(len(result.raw_nodes), 8)
        self.assertTrue(all(result.node_type_correction_mask))
        self.assertEqual(
            tuple(node.node_type_id for node in result.raw_nodes),
            result.grammar_constrained_node_type_ids,
        )
        self.assertEqual(
            tuple(node.node_type_id for node in result.same_history_raw_shadow_nodes),
            result.raw_node_type_argmax_ids,
        )
        converted = validate_and_convert_v6_autonomous_prediction(
            result, max_operations=2
        )
        self.assertIsNotNone(converted)
        self.assertFalse(hasattr(result, "requested_node_count"))
        invalid_first = replace(
            result.raw_nodes[0], node_type_id=NODE_TYPES.id(None)
        )
        malformed = replace(
            result,
            raw_nodes=(invalid_first,) + result.raw_nodes[1:],
            grammar_constrained_node_type_ids=(NODE_TYPES.id(None),)
            + result.grammar_constrained_node_type_ids[1:],
        )
        with self.assertRaises(NodeGrammarError):
            validate_and_convert_v6_autonomous_prediction(
                malformed, max_operations=2
            )
        with self.assertRaises(NodeGrammarError) as caught:
            validate_and_convert_v6_autonomous_prediction(
                replace(result, node_grammar_contract_id="wrong"),
                max_operations=2,
            )
        self.assertEqual(caught.exception.code, "invalid_v6_grammar_contract")
        for name, value in before.items():
            self.assertTrue(torch.equal(value, model.state_dict()[name]))
        self.assertTrue(torch.equal(rng_before, torch.get_rng_state()))

    def test_job_3334551_raw_none_is_corrected_before_v4_categories(self):
        from prototype.flat_baseline.constrained_v6_conversion import (
            construct_v6_predicted_node_tensors,
        )
        from prototype.node_conditioned_categories import (
            V4_RETAINED_CATEGORICAL_FIELDS,
        )
        node_logits = torch.full((1, len(NODE_TYPES.tokens)), -5.0)
        node_logits[0, NODE_TYPES.id(None)] = 10.0
        categorical_logits = tuple(
            torch.zeros(1, len(field.class_order))
            for field in V4_RETAINED_CATEGORICAL_FIELDS
        )
        result = construct_v6_predicted_node_tensors(
            node_logits,
            ((),),
            torch.tensor([4]),
            categorical_logits,
            torch.zeros(1, 3),
            torch.zeros(1, 3),
            torch.zeros(1, 6),
        )
        self.assertEqual(
            result.raw_node_type_argmax_ids.item(), NODE_TYPES.id(None)
        )
        self.assertEqual(
            result.grammar_constrained_node_type_ids.item(),
            NODE_TYPES.id("reference_plane"),
        )
        self.assertTrue(result.node_type_correction_mask.item())
        self.assertEqual(
            result.records.node_type_ids.item(), NODE_TYPES.id("reference_plane")
        )
        self.assertTrue(result.records.geometry_mask[0, :9].all())
        self.assertFalse(result.raw_shadow_records.geometry_mask[0].any())

    def test_axis_tensor_construction_mixed_no_axis_masks_and_errors(self):
        from prototype.axis_geometry_torch import constrain_axis_geometry_tensors
        axis = NODE_TYPES.id("axis")
        extrude = NODE_TYPES.id("extrude")
        ids = torch.tensor([axis, extrude, axis], dtype=torch.long)
        raw = torch.tensor([
            [0.0, 0.0, 0.0, 1.0, 0.7, -0.4],
            [0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
            [0.5, -0.5, 1.0, 0.0, -0.2, 0.8],
        ], dtype=torch.float32)
        geometry = torch.zeros(3, 39, dtype=torch.float32)
        geometry[:, 33:39] = raw
        mask = torch.zeros(3, 39, dtype=torch.bool)
        mask[0, 33:37] = True
        mask[1, 37] = True
        mask[2, 33:37] = True
        result = constrain_axis_geometry_tensors(
            ids, raw, geometry, mask
        )
        expected = torch.tensor(CANONICAL_AXIS_CHANNELS)
        torch.testing.assert_close(
            result.geometry[[0, 2], 33:39],
            expected.expand(2, 6),
            rtol=0,
            atol=0,
        )
        self.assertFalse(result.axis_geometry_correction_mask[0].any())
        self.assertTrue(result.axis_geometry_correction_mask[2, :4].all())
        self.assertFalse(result.axis_geometry_correction_mask[:, 4:].any())
        self.assertEqual(result.geometry[1, 37].item(), raw[1, 4].item())
        self.assertEqual(result.geometry.dtype, raw.dtype)
        self.assertEqual(result.geometry.device, raw.device)
        self.assertTrue(result.geometry.is_contiguous())
        self.assertTrue(result.geometry_mask.is_contiguous())

        no_axis = constrain_axis_geometry_tensors(
            torch.tensor([extrude]),
            raw[1:2],
            geometry[1:2],
            mask[1:2],
        )
        expected_no_axis = torch.zeros_like(geometry[1:2])
        expected_no_axis[0, 37] = raw[1, 4]
        torch.testing.assert_close(
            no_axis.geometry, expected_no_axis, rtol=0, atol=0
        )
        self.assertEqual(
            torch.nonzero(
                no_axis.geometry[0] != geometry[1], as_tuple=False
            ).flatten().tolist(),
            [33, 34, 35, 36, 38],
        )
        torch.testing.assert_close(
            no_axis.raw_axis_geometry, raw[1:2], rtol=0, atol=0
        )
        self.assertFalse(no_axis.axis_node_mask.any())
        self.assertFalse(no_axis.axis_geometry_correction_mask.any())
        for nonfinite in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(nonfinite=nonfinite):
                with self.assertRaises(ValueError):
                    constrain_axis_geometry_tensors(
                        ids, raw.clone().fill_(nonfinite), geometry, mask
                    )
        wrong_mask = mask.clone()
        wrong_mask[0, 36] = False
        with self.assertRaises(ValueError):
            constrain_axis_geometry_tensors(
                ids, raw, geometry, wrong_mask
            )
        with self.assertRaises(ValueError):
            constrain_axis_geometry_tensors(
                torch.tensor([len(NODE_TYPES.tokens)]),
                raw[:1], geometry[:1], mask[:1],
            )
        double_result = constrain_axis_geometry_tensors(
            ids, raw.double(), geometry.double(), mask
        )
        self.assertEqual(double_result.geometry.dtype, torch.float64)
        self.assertEqual(double_result.raw_axis_geometry.dtype, torch.float64)

    def test_axis_construction_changes_only_axis_rows_from_v5_records(self):
        from prototype.flat_baseline.constrained_v4_conversion import (
            construct_v4_predicted_node_tensors,
        )
        from prototype.flat_baseline.constrained_v6_conversion import (
            constrain_v6_node_records,
        )
        from prototype.node_conditioned_categories import (
            V4_RETAINED_CATEGORICAL_FIELDS,
        )
        node_names = (
            "reference_plane", "sketch", "profile", "extrude", "revolve", "axis"
        )
        node_ids = torch.tensor(
            [NODE_TYPES.id(name) for name in node_names], dtype=torch.long
        )
        raw = torch.tensor(
            [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]] * len(node_names),
            dtype=torch.float32,
        )
        categorical = tuple(
            torch.zeros(len(node_names), len(field.class_order))
            for field in V4_RETAINED_CATEGORICAL_FIELDS
        )
        v5_records = construct_v4_predicted_node_tensors(
            node_ids,
            categorical,
            torch.zeros(len(node_names), 3),
            torch.zeros(len(node_names), 3),
            raw,
        )
        v6_records, evidence = constrain_v6_node_records(
            v5_records, raw, torch.ones(len(node_names), dtype=torch.bool)
        )
        for index, name in enumerate(node_names[:-1]):
            with self.subTest(node_type=name):
                torch.testing.assert_close(
                    v6_records.geometry[index],
                    v5_records.geometry[index],
                    rtol=0,
                    atol=0,
                )
                self.assertTrue(torch.equal(
                    v6_records.geometry_mask[index],
                    v5_records.geometry_mask[index],
                ))
                self.assertFalse(evidence.axis_node_mask[index].item())
                self.assertFalse(
                    v6_records.geometry_mask[index, 33:37].any().item()
                )
        axis_index = len(node_names) - 1
        torch.testing.assert_close(
            v6_records.geometry[axis_index, 33:39],
            torch.tensor(CANONICAL_AXIS_CHANNELS),
            rtol=0,
            atol=0,
        )
        self.assertEqual(
            tuple(v6_records.geometry_mask[axis_index, 33:39].tolist()),
            CANONICAL_AXIS_MASK,
        )
        self.assertEqual(
            tuple(v6_records.geometry[axis_index, 37:39].tolist()),
            (0.0, 0.0),
        )

    def test_v5_grammar_controls_axis_applicability_not_raw_argmax(self):
        from prototype.flat_baseline.constrained_v6_conversion import (
            construct_v6_predicted_node_tensors,
        )
        from prototype.node_conditioned_categories import (
            V4_RETAINED_CATEGORICAL_FIELDS,
        )
        categorical = tuple(
            torch.zeros(1, len(field.class_order))
            for field in V4_RETAINED_CATEGORICAL_FIELDS
        )
        raw_geometry = torch.tensor([[0.2, 0.3, 0.4, 0.5, 0.6, 0.7]])
        logits = torch.full((1, len(NODE_TYPES.tokens)), -5.0)
        logits[0, NODE_TYPES.id("axis")] = 10.0
        first = construct_v6_predicted_node_tensors(
            logits, ((),), torch.tensor([4]), categorical,
            torch.zeros(1, 3), torch.zeros(1, 3), raw_geometry,
        )
        self.assertEqual(
            first.grammar_constrained_node_type_ids.item(),
            NODE_TYPES.id("reference_plane"),
        )
        self.assertFalse(first.records.geometry_mask[0, 33:37].any())
        self.assertFalse(first.axis_geometry_correction_mask.any())

        logits.zero_()
        logits[0, NODE_TYPES.id("extrude")] = 10.0
        prefix = tuple(NODE_TYPES.id(item) for item in (
            "reference_plane", "sketch", "profile"
        ))
        selected = construct_v6_predicted_node_tensors(
            logits, (prefix,), torch.tensor([5]), categorical,
            torch.zeros(1, 3), torch.zeros(1, 3), raw_geometry,
        )
        self.assertEqual(
            selected.grammar_constrained_node_type_ids.item(),
            NODE_TYPES.id("axis"),
        )
        self.assertTrue(selected.records.geometry_mask[0, 33:37].all())
        torch.testing.assert_close(
            selected.records.geometry[0, 33:39],
            torch.tensor(CANONICAL_AXIS_CHANNELS),
            rtol=0,
            atol=0,
        )

    def test_strict_v6_converter_rejects_axis_value_mask_and_leakage(self):
        from prototype.axis_geometry import AxisGeometryContractError
        from prototype.flat_baseline.constrained_v6_conversion import (
            validate_and_convert_v6_teacher_forced_prediction,
        )
        prediction = self._authoritative_v6_prediction("R")
        axis_position = prediction.grammar_constrained_node_type_ids.index(
            NODE_TYPES.id("axis")
        )
        axis_node = prediction.raw_nodes[axis_position]
        geometry = list(axis_node.normalized_geometry)
        geometry[33] = 0.25
        malformed_node = replace(axis_node, normalized_geometry=tuple(geometry))
        malformed = replace(
            prediction,
            raw_nodes=prediction.raw_nodes[:axis_position]
            + (malformed_node,)
            + prediction.raw_nodes[axis_position + 1:],
        )
        with self.assertRaises(AxisGeometryContractError):
            validate_and_convert_v6_teacher_forced_prediction(
                malformed, max_operations=2
            )
        non_axis = self._authoritative_v6_prediction("E")
        node = non_axis.raw_nodes[0]
        geometry = list(node.normalized_geometry)
        geometry[33] = 0.25
        mask = list(node.derived_geometry_mask)
        mask[33] = True
        leaking_node = replace(
            node,
            normalized_geometry=tuple(geometry),
            derived_geometry_mask=tuple(mask),
        )
        leaking = replace(
            non_axis,
            raw_nodes=(leaking_node,) + non_axis.raw_nodes[1:],
        )
        result = validate_and_convert_v6_teacher_forced_prediction(
            leaking, max_operations=2
        )
        self.assertFalse(result.controlled_domain.valid)
        self.assertEqual(
            result.primary_failure.code, "geometry_mask_applicability"
        )
        mask = list(axis_node.derived_geometry_mask)
        mask[36] = False
        malformed_node = replace(axis_node, derived_geometry_mask=tuple(mask))
        malformed = replace(
            prediction,
            raw_nodes=prediction.raw_nodes[:axis_position]
            + (malformed_node,)
            + prediction.raw_nodes[axis_position + 1:],
        )
        with self.assertRaises(AxisGeometryContractError):
            validate_and_convert_v6_teacher_forced_prediction(
                malformed, max_operations=2
            )

    def test_teacher_forced_current_target_mutation_cannot_change_prediction(self):
        from prototype.constrained_profile_decoder import profile_targets_for_loss
        from prototype.flat_baseline.constrained_v6 import ConstrainedProfileV6Model
        from prototype.flat_baseline.constrained_v6_config import ConstrainedProfileV6Config
        from prototype.flat_baseline.constrained_v6_conversion import (
            v6_teacher_forced_predictions,
        )
        batch = _batch("E")
        inputs = batch.to_torch(torch)
        target = batch.target.to_torch(torch)
        profiles = profile_targets_for_loss(batch.target, inputs["geometry"])
        model = ConstrainedProfileV6Model(ConstrainedProfileV6Config())
        model.eval()
        changed = dict(target)
        changed["node_type_ids"] = target["node_type_ids"].clone()
        position = int(target["node_mask"][0].sum().item()) - 1
        changed["node_type_ids"][0, position] = NODE_TYPES.id("revolve")
        with torch.no_grad():
            original = model(target=target, profile_targets=profiles, **inputs)
            mutated = model(target=changed, profile_targets=profiles, **inputs)
        self.assertTrue(torch.equal(
            original.teacher_forced_prefix_node_ids[0, 1:position + 1],
            target["node_type_ids"][0, :position],
        ))
        torch.testing.assert_close(
            original.node_type_logits[0, position],
            mutated.node_type_logits[0, position],
            rtol=0,
            atol=0,
        )
        first = v6_teacher_forced_predictions(
            original,
            node_mask=target["node_mask"],
            node_count_source="test",
        )[0]
        second = v6_teacher_forced_predictions(
            mutated,
            node_mask=target["node_mask"],
            node_count_source="test",
        )[0]
        self.assertEqual(
            first.grammar_constrained_node_type_ids[position],
            second.grammar_constrained_node_type_ids[position],
        )
        self.assertEqual(
            first.raw_node_type_argmax_ids[position],
            second.raw_node_type_argmax_ids[position],
        )

    def test_target_axis_geometry_cannot_change_current_axis_prediction(self):
        from prototype.constrained_profile_decoder import profile_targets_for_loss
        from prototype.flat_baseline.constrained_v6 import ConstrainedProfileV6Model
        from prototype.flat_baseline.constrained_v6_config import ConstrainedProfileV6Config
        from prototype.flat_baseline.constrained_v6_conversion import (
            v6_teacher_forced_predictions,
        )
        batch = _batch("R")
        inputs = batch.to_torch(torch)
        target = batch.target.to_torch(torch)
        profiles = profile_targets_for_loss(batch.target, inputs["geometry"])
        axis_position = target["node_type_ids"][0].tolist().index(
            NODE_TYPES.id("axis")
        )
        changed = dict(target)
        changed["geometry"] = target["geometry"].clone()
        changed["geometry"][0, axis_position, 33:37] = torch.tensor(
            [0.5, -0.5, 1.0, 0.0]
        )
        model = ConstrainedProfileV6Model(ConstrainedProfileV6Config())
        model.eval()
        with torch.no_grad():
            original = model(target=target, profile_targets=profiles, **inputs)
            mutated = model(target=changed, profile_targets=profiles, **inputs)
        first = v6_teacher_forced_predictions(
            original, node_mask=target["node_mask"], node_count_source="test"
        )[0]
        second = v6_teacher_forced_predictions(
            mutated, node_mask=target["node_mask"], node_count_source="test"
        )[0]
        self.assertEqual(
            first.raw_nodes[axis_position], second.raw_nodes[axis_position]
        )
        self.assertEqual(
            first.raw_axis_geometry[axis_position],
            second.raw_axis_geometry[axis_position],
        )
        self.assertEqual(
            first.constrained_axis_geometry[axis_position],
            CANONICAL_AXIS_CHANNELS,
        )

    def test_v6_checkpoint_rejects_v4_and_changed_grammar_metadata(self):
        from prototype.flat_baseline.constrained_v6 import ConstrainedProfileV6Model
        from prototype.flat_baseline.constrained_v6_config import ConstrainedProfileV6Config
        from prototype.flat_baseline.constrained_v6_training import (
            ConstrainedV6TrainingError,
            TinyOverfitSelection,
            _validate_v6_checkpoint,
            build_v6_optimizer,
            v6_checkpoint_payload,
        )
        from prototype.flat_baseline.constrained_v6_training_config import (
            ConstrainedV6TrainingConfig,
        )
        model_config = ConstrainedProfileV6Config()
        training_config = ConstrainedV6TrainingConfig(require_clean_source=False)
        model = ConstrainedProfileV6Model(model_config)
        optimizer = build_v6_optimizer(model, training_config)
        selection = TinyOverfitSelection(
            examples=(),
            selected_example_ids=(),
            selection_rule=training_config.tiny_overfit_selection_identity,
            family_counts={},
            exact_operation_types=(),
            operation_families=(),
            node_count_distribution=(),
            missing_coverage=(),
            corpus_dir="synthetic",
            train_selection_sha256="0" * 64,
            split_name="iid",
            training_partition="train",
        )
        payload = v6_checkpoint_payload(
            model,
            optimizer,
            model_config,
            training_config,
            selection,
            global_step=0,
            source_provenance={
                "git_commit": "a" * 40,
                "git_dirty": False,
                "git_status_porcelain": [],
                "source_tree_sha256": "b" * 64,
            },
        )
        _validate_v6_checkpoint(
            payload, model_config, training_config, selection
        )
        for name, value in (
            ("checkpoint_version", 5),
            ("node_grammar_contract_id", "wrong"),
            ("node_vocabulary", list(reversed(NODE_TYPES.tokens))),
            ("valid_requested_node_counts", [4, 5]),
            ("completion_algorithm_id", "wrong"),
            (
                "node_grammar_contract",
                dict(payload["node_grammar_contract"], terminal_nodes=["extrude"]),
            ),
            ("axis_geometry_contract_id", "wrong"),
            ("axis_construction_algorithm_id", "wrong"),
            ("axis_geometry_channel_indices", [34, 35, 36, 37, 38, 39]),
            (
                "axis_geometry_contract",
                dict(payload["axis_geometry_contract"], coordinate_space="world"),
            ),
        ):
            malformed = dict(payload)
            malformed[name] = value
            with self.subTest(name=name):
                with self.assertRaises(ConstrainedV6TrainingError):
                    _validate_v6_checkpoint(
                        malformed, model_config, training_config, selection
                    )
        missing = dict(payload)
        del missing["node_grammar_contract"]
        with self.assertRaises(ConstrainedV6TrainingError):
            _validate_v6_checkpoint(
                missing, model_config, training_config, selection
            )
        malformed_boolean = dict(payload)
        malformed_boolean["source_provenance"] = dict(
            payload["source_provenance"], git_dirty=0
        )
        with self.assertRaises(ConstrainedV6TrainingError):
            _validate_v6_checkpoint(
                malformed_boolean, model_config, training_config, selection
            )
