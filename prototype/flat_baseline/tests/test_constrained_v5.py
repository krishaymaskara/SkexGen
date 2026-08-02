"""Focused prefix-conditioned node grammar V5 tests."""

from __future__ import annotations

import ast
from dataclasses import replace
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
from prototype.representation.model import GeometryEncoding


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
        physical_family_id="v5-test", nodes=nodes, target=target
    ))
    return collate_flat((example,))


class V5GrammarStaticTests(unittest.TestCase):
    def test_exact_v5_identity_and_immutable_contract(self):
        from prototype.flat_baseline.constrained_v5_config import (
            CONSTRAINED_PROFILE_CHECKPOINT_VERSION,
            CONSTRAINED_PROFILE_DECODER_CONTRACT_VERSION,
            CONSTRAINED_PROFILE_MODEL_CONFIG_VERSION,
            CONSTRAINED_PROFILE_MODEL_NAME,
            ConstrainedProfileV5Config,
        )
        config = ConstrainedProfileV5Config()
        config.validate()
        self.assertEqual(
            CONSTRAINED_PROFILE_MODEL_NAME,
            "B0-FLAT-CONSTRAINED-PROFILE-CANONICAL-PLANE-"
            "NODE-CATEGORIES-PREFIX-GRAMMAR-v5",
        )
        self.assertEqual(
            (CONSTRAINED_PROFILE_CHECKPOINT_VERSION,
             CONSTRAINED_PROFILE_MODEL_CONFIG_VERSION,
             CONSTRAINED_PROFILE_DECODER_CONTRACT_VERSION),
            (5, 5, 5),
        )
        self.assertEqual(config.node_grammar_contract_id, NODE_GRAMMAR_CONTRACT_ID)
        self.assertEqual(config.completion_algorithm_id, COMPLETION_ALGORITHM_ID)
        self.assertEqual(config.valid_requested_node_counts, (4, 5, 7, 8, 9))
        self.assertEqual(V5_NODE_GRAMMAR.node_vocabulary, NODE_TYPES.tokens)
        self.assertEqual(V5_NODE_GRAMMAR.to_json(), V5_NODE_GRAMMAR.to_json())
        self.assertEqual(config.learned_geometry_channel_indices, tuple(range(33, 39)))

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
            Path(__file__).resolve().parents[1] / "constrained_v5.py",
            Path(__file__).resolve().parents[1] / "constrained_v5_conversion.py",
            Path(__file__).resolve().parents[1] / "constrained_v5_autonomous.py",
        )
        for path in paths:
            ast.parse(path.read_text(), str(path), feature_version=(3, 8))


@unittest.skipIf(torch is None, TORCH_REASON)
class V5GrammarTensorTests(unittest.TestCase):
    def _authoritative_v5_prediction(self, template):
        from prototype.flat_baseline.autonomous import TEACHER_FORCED_PREFIX_FEEDBACK
        from prototype.flat_baseline.constrained_v5_conversion import (
            V5TeacherForcedRawPrediction,
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
        return V5TeacherForcedRawPrediction(
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

    def test_v4_v5_parameter_state_and_loss_equivalence(self):
        from prototype.constrained_profile_decoder import profile_targets_for_loss
        from prototype.flat_baseline.constrained_v4 import ConstrainedProfileV4Model
        from prototype.flat_baseline.constrained_v4_config import ConstrainedProfileV4Config
        from prototype.flat_baseline.constrained_v4_losses import constrained_profile_v4_loss
        from prototype.flat_baseline.constrained_v5 import ConstrainedProfileV5Model
        from prototype.flat_baseline.constrained_v5_config import ConstrainedProfileV5Config
        from prototype.flat_baseline.constrained_v5_losses import constrained_profile_v5_loss
        batch = _batch()
        inputs = batch.to_torch(torch)
        target = batch.target.to_torch(torch)
        profiles = profile_targets_for_loss(batch.target, inputs["geometry"])
        torch.manual_seed(29)
        v4 = ConstrainedProfileV4Model(ConstrainedProfileV4Config())
        torch.manual_seed(29)
        v5 = ConstrainedProfileV5Model(ConstrainedProfileV5Config())
        self.assertEqual(
            sum(parameter.numel() for parameter in v4.parameters()),
            sum(parameter.numel() for parameter in v5.parameters()),
        )
        self.assertEqual(tuple(v4.state_dict()), tuple(v5.state_dict()))
        v5.load_state_dict(v4.state_dict(), strict=True)
        v4.eval()
        v5.eval()
        output4 = v4(target=target, profile_targets=profiles, **inputs)
        output5 = v5(target=target, profile_targets=profiles, **inputs)
        for name in vars(output4):
            value4 = getattr(output4, name)
            value5 = getattr(output5, name)
            if torch.is_tensor(value4):
                torch.testing.assert_close(value4, value5, rtol=0, atol=0)
        loss4 = constrained_profile_v4_loss(
            output4, target, profiles, v4.config
        )
        loss5 = constrained_profile_v5_loss(
            output5, target, profiles, v5.config
        )
        for name, value in loss4.as_dict().items():
            torch.testing.assert_close(value, loss5.as_dict()[name], rtol=0, atol=0)
        loss5.total.backward()
        self.assertTrue(all(
            parameter.grad is not None
            for parameter in v5.parameters()
            if parameter.requires_grad
        ))

    def test_strict_v5_conversion_accepts_all_builder_sequences_without_repair(self):
        from prototype.flat_baseline.constrained_v5_conversion import (
            validate_and_convert_v5_teacher_forced_prediction,
        )
        for template in ("E", "R", "EE", "ER", "RE", "RR"):
            with self.subTest(template=template):
                prediction = self._authoritative_v5_prediction(template)
                result = validate_and_convert_v5_teacher_forced_prediction(
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
                    validate_and_convert_v5_teacher_forced_prediction(
                        invalid, max_operations=2
                    )

    def test_job_3336430_autonomous_is_target_free_exact_and_grammar_valid(self):
        from prototype.flat_baseline.constrained_v5 import ConstrainedProfileV5Model
        from prototype.flat_baseline.constrained_v5_autonomous import (
            greedy_decode_v5,
            validate_and_convert_v5_autonomous_prediction,
        )
        from prototype.flat_baseline.constrained_v5_config import ConstrainedProfileV5Config
        batch = _batch("ER")
        inputs = batch.to_torch(torch)
        model = ConstrainedProfileV5Model(ConstrainedProfileV5Config())
        with torch.no_grad():
            model.node_type_head.weight.zero_()
            model.node_type_head.bias.zero_()
            model.node_type_head.bias[NODE_TYPES.id(None)] = 10
        self.assertNotIn("target", inspect.signature(greedy_decode_v5).parameters)
        before = {name: value.clone() for name, value in model.state_dict().items()}
        rng_before = torch.get_rng_state().clone()
        result = greedy_decode_v5(
            model,
            inputs,
            node_counts=torch.tensor([8]),
            node_count_source="authorized_test_length",
        )[0]
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
        converted = validate_and_convert_v5_autonomous_prediction(
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
            validate_and_convert_v5_autonomous_prediction(
                malformed, max_operations=2
            )
        with self.assertRaises(NodeGrammarError) as caught:
            validate_and_convert_v5_autonomous_prediction(
                replace(result, node_grammar_contract_id="wrong"),
                max_operations=2,
            )
        self.assertEqual(caught.exception.code, "invalid_v5_grammar_contract")
        for name, value in before.items():
            self.assertTrue(torch.equal(value, model.state_dict()[name]))
        self.assertTrue(torch.equal(rng_before, torch.get_rng_state()))

    def test_job_3334551_raw_none_is_corrected_before_v4_categories(self):
        from prototype.flat_baseline.constrained_v5_conversion import (
            construct_v5_predicted_node_tensors,
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
        result = construct_v5_predicted_node_tensors(
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

    def test_teacher_forced_current_target_mutation_cannot_change_prediction(self):
        from prototype.constrained_profile_decoder import profile_targets_for_loss
        from prototype.flat_baseline.constrained_v5 import ConstrainedProfileV5Model
        from prototype.flat_baseline.constrained_v5_config import ConstrainedProfileV5Config
        from prototype.flat_baseline.constrained_v5_conversion import (
            v5_teacher_forced_predictions,
        )
        batch = _batch("E")
        inputs = batch.to_torch(torch)
        target = batch.target.to_torch(torch)
        profiles = profile_targets_for_loss(batch.target, inputs["geometry"])
        model = ConstrainedProfileV5Model(ConstrainedProfileV5Config())
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
        first = v5_teacher_forced_predictions(
            original,
            node_mask=target["node_mask"],
            node_count_source="test",
        )[0]
        second = v5_teacher_forced_predictions(
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

    def test_v5_checkpoint_rejects_v4_and_changed_grammar_metadata(self):
        from prototype.flat_baseline.constrained_v5 import ConstrainedProfileV5Model
        from prototype.flat_baseline.constrained_v5_config import ConstrainedProfileV5Config
        from prototype.flat_baseline.constrained_v5_training import (
            ConstrainedV5TrainingError,
            TinyOverfitSelection,
            _validate_v5_checkpoint,
            build_v5_optimizer,
            v5_checkpoint_payload,
        )
        from prototype.flat_baseline.constrained_v5_training_config import (
            ConstrainedV5TrainingConfig,
        )
        model_config = ConstrainedProfileV5Config()
        training_config = ConstrainedV5TrainingConfig(require_clean_source=False)
        model = ConstrainedProfileV5Model(model_config)
        optimizer = build_v5_optimizer(model, training_config)
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
        payload = v5_checkpoint_payload(
            model,
            optimizer,
            model_config,
            training_config,
            selection,
            global_step=0,
            source_provenance={
                "git_commit": "a" * 40,
                "git_dirty": False,
                "git_status_porcelain": "",
                "source_tree_sha256": "b" * 64,
            },
        )
        _validate_v5_checkpoint(
            payload, model_config, training_config, selection
        )
        for name, value in (
            ("checkpoint_version", 4),
            ("node_grammar_contract_id", "wrong"),
            ("node_vocabulary", list(reversed(NODE_TYPES.tokens))),
            ("valid_requested_node_counts", [4, 5]),
            ("completion_algorithm_id", "wrong"),
            (
                "node_grammar_contract",
                dict(payload["node_grammar_contract"], terminal_nodes=["extrude"]),
            ),
        ):
            malformed = dict(payload)
            malformed[name] = value
            with self.subTest(name=name):
                with self.assertRaises(ConstrainedV5TrainingError):
                    _validate_v5_checkpoint(
                        malformed, model_config, training_config, selection
                    )
        missing = dict(payload)
        del missing["node_grammar_contract"]
        with self.assertRaises(ConstrainedV5TrainingError):
            _validate_v5_checkpoint(
                missing, model_config, training_config, selection
            )
        malformed_boolean = dict(payload)
        malformed_boolean["source_provenance"] = dict(
            payload["source_provenance"], git_dirty=0
        )
        with self.assertRaises(ConstrainedV5TrainingError):
            _validate_v5_checkpoint(
                malformed_boolean, model_config, training_config, selection
            )
