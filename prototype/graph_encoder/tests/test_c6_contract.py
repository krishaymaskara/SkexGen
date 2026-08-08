"""PyTorch-independent C6 contract, metric, provenance, and audit tests."""

from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

from prototype.graph_encoder.autonomous import deterministic_derangement
from prototype.graph_encoder.config import GE1TrainingConfig
from prototype.graph_encoder.errors import GraphEncoderError
from prototype.graph_encoder.metrics import (
    FIRST_FAILURE_STAGES,
    PrefixAttempt,
    intervention_ratios,
    prefix_score_from_outcomes,
    receptive_field_record,
)
from prototype.graph_encoder.provenance import (
    authorize_c6_provenance,
    collect_c6_provenance,
    training_partition_identity,
    verify_c6_provenance,
)
from prototype.graph_encoder.training import (
    deterministic_family_order,
    plateau_state,
    training_contract_metadata,
)


ROOT = Path(__file__).resolve().parents[3]


class FrozenTrainingTests(unittest.TestCase):
    def test_frozen_recipe_and_selection(self):
        config = GE1TrainingConfig()
        config.validate()
        self.assertEqual(config.epochs, 50)
        self.assertEqual(config.batch_size, 8)
        self.assertEqual(config.optimizer, "AdamW")
        self.assertEqual(config.learning_rate, 1e-3)
        self.assertEqual(config.weight_decay, 0.0)
        self.assertEqual(config.gradient_clip_norm, 1.0)
        self.assertEqual(config.planned_seeds, (2026, 2027, 2028))
        self.assertEqual(config.checkpoint_selection, "fixed_epoch_50")
        metadata = training_contract_metadata()
        self.assertFalse(metadata["early_stopping"])
        self.assertFalse(metadata["development_checkpoint_selection"])
        self.assertFalse(metadata["training_loss_checkpoint_selection"])
        self.assertEqual(metadata["selected_experimental_checkpoint"], "fixed_epoch_50")

    def test_family_order_is_seed_epoch_pinned_and_arm_independent(self):
        families = ("c", "a", "d", "b")
        first = deterministic_family_order(families, 2026, 1)
        self.assertEqual(first, deterministic_family_order(families, 2026, 1))
        self.assertEqual(set(first), set(families))
        self.assertNotEqual(first, deterministic_family_order(families, 2026, 2))

    def test_duplicate_family_order_rejected(self):
        with self.assertRaises(GraphEncoderError):
            deterministic_family_order(("a", "a"), 2026, 1)


class PlateauTests(unittest.TestCase):
    def test_no_check_before_epoch_ten(self):
        self.assertEqual(plateau_state((1.0,) * 9).evaluations, ())

    def test_exact_windows_and_less_than_rule(self):
        losses = (1.0,) * 5 + (0.99,) * 5
        state = plateau_state(losses)
        item = state.evaluations[0]
        self.assertEqual(item.epoch, 10)
        self.assertEqual(item.previous_best, 1.0)
        self.assertEqual(item.recent_best, 0.99)
        self.assertAlmostEqual(item.relative_improvement, 0.01)
        self.assertFalse(item.plateau)

    def test_less_than_one_percent_and_negative_improvement(self):
        self.assertTrue(plateau_state((1.0,) * 10).evaluations[0].plateau)
        item = plateau_state((1.0,) * 5 + (1.1,) * 5).evaluations[0]
        self.assertLess(item.relative_improvement, 0.0)
        self.assertTrue(item.plateau)

    def test_zero_denominator_epsilon_and_first_epoch_frozen(self):
        state = plateau_state((0.0,) * 10)
        self.assertEqual(state.first_plateau_epoch, 10)
        extended = plateau_state((0.0,) * 10 + (-1.0,), state.first_plateau_epoch)
        self.assertEqual(extended.first_plateau_epoch, 10)

    def test_nonfinite_history_rejected(self):
        with self.assertRaises(GraphEncoderError):
            plateau_state((float("nan"),) * 10)


class InterventionTests(unittest.TestCase):
    def test_derangement_is_sorted_deterministic_and_has_no_fixed_points(self):
        families = ("d", "b", "a", "c")
        first = deterministic_derangement(families, 2026)
        self.assertEqual(first, deterministic_derangement(families, 2026))
        self.assertEqual(tuple(item[0] for item in first), tuple(sorted(families)))
        self.assertTrue(all(left != right for left, right in first))
        self.assertEqual({item[1] for item in first}, set(families))

    def test_shuffle_unavailable_below_two(self):
        with self.assertRaises(GraphEncoderError) as caught:
            deterministic_derangement(("only",), 2026)
        self.assertEqual(caught.exception.code, "shuffle_unavailable")

    def test_ratios_and_zero_true_denominator(self):
        def metric(name, value):
            return {
                "condition": name,
                "available": True,
                "aggregates": {"primary": {"value": value}},
            }
        ratios = intervention_ratios((
            metric("P_true", 0.5), metric("P_shuffle", 0.25),
            metric("P_mean", 0.1),
        ))
        self.assertEqual(ratios["R_shuffle"], 0.5)
        self.assertEqual(ratios["R_mean"], 0.2)
        zero = intervention_ratios((
            metric("P_true", 0.0), metric("P_shuffle", 0.0),
            metric("P_mean", 0.0),
        ))
        self.assertEqual(zero["R_shuffle"]["reason"], "zero_true_denominator")


class PrefixCoreTests(unittest.TestCase):
    def test_ambiguous_canonicalization_scores_zero(self):
        score = prefix_score_from_outcomes(
            2,
            canonicalization_status="failed",
            canonicalization_failure_code="ambiguous_operation_chain",
        )
        self.assertEqual(score.normalized_longest_executable_operation_prefix, 0.0)
        self.assertEqual(score.first_failed_stage, "canonicalization")

    def test_zero_one_and_two_of_two(self):
        failed = PrefixAttempt(1, False, False, "strict conversion", "bad")
        first = PrefixAttempt(1, True, True, "no failure", "valid")
        second = PrefixAttempt(2, True, True, "no failure", "valid")
        failed_second = PrefixAttempt(2, False, False, "strict conversion", "bad")
        zero = prefix_score_from_outcomes(
            2, canonicalization_status="success",
            generated_operation_group_count=2, outcomes=(failed, failed_second)
        )
        half = prefix_score_from_outcomes(
            2, canonicalization_status="success",
            generated_operation_group_count=1, outcomes=(first,)
        )
        full = prefix_score_from_outcomes(
            2, canonicalization_status="success",
            generated_operation_group_count=2, outcomes=(first, second)
        )
        self.assertEqual(zero.normalized_longest_executable_operation_prefix, 0.0)
        self.assertEqual(half.normalized_longest_executable_operation_prefix, 0.5)
        self.assertEqual(full.normalized_longest_executable_operation_prefix, 1.0)
        self.assertEqual(len(full.evaluated_prefixes), 3)

    def test_one_operation_success_and_failure(self):
        valid = PrefixAttempt(1, True, True, "no failure", "valid")
        invalid = PrefixAttempt(1, True, False, "analytic validity", "invalid")
        self.assertEqual(prefix_score_from_outcomes(
            1, canonicalization_status="success",
            generated_operation_group_count=1, outcomes=(valid,)
        ).normalized_longest_executable_operation_prefix, 1.0)
        self.assertEqual(prefix_score_from_outcomes(
            1, canonicalization_status="success",
            generated_operation_group_count=1, outcomes=(invalid,)
        ).normalized_longest_executable_operation_prefix, 0.0)

    def test_invalid_denominator_and_incomplete_record_fail(self):
        with self.assertRaises(GraphEncoderError):
            prefix_score_from_outcomes(0, canonicalization_status="success")
        with self.assertRaises(GraphEncoderError):
            prefix_score_from_outcomes(
                1, canonicalization_status="success",
                generated_operation_group_count=1, outcomes=()
            )

    def test_receptive_field_contract(self):
        record = receptive_field_record()
        self.assertEqual(record["relational_message_passing_layers"], 3)
        self.assertEqual(
            record["verified_maximum_undirected_controlled_template_diameter"], 3
        )
        self.assertTrue(record["maximum_distance_covered"])
        self.assertIn("canonicalization", FIRST_FAILURE_STAGES)

    def test_macro_aggregation_weights_families_equally_and_records_denominators(self):
        from prototype.graph_encoder.metrics import _macro_aggregates, undefined

        geometry = {
            name: {"physical_mae": 1.0}
            for name in (
                "reference_plane", "profile_primitives", "axis",
                "operation_parameter",
            )
        }
        base = {
            "complete_executable_validity": 0.0,
            "valid_single_solid": 0.0,
            "strict_conversion": 0.0,
            "exact_graph": 0.0,
            "exact_node_sequence": 0.0,
            "depends_on_precision": undefined("zero"),
            "depends_on_recall": undefined("zero"),
            "depends_on_exactness": 0.0,
            "attachment_accuracy": 0.0,
            "unnormalized_longest_executable_prefix": 0.0,
            "first_failure_code": "bad",
            "stage_of_first_failure": "edge prediction",
            "geometry_error_by_channel_family": geometry,
        }
        values = {
            "large-sample-family": dict(base, primary=0.0),
            "small-sample-family": dict(base, primary=1.0),
        }
        aggregate = _macro_aggregates(values)
        self.assertEqual(aggregate["primary"]["value"], 0.5)
        self.assertEqual(aggregate["primary"]["physical_family_denominator"], 2)
        self.assertEqual(
            aggregate["depends_on_precision"]["defined_family_denominator"], 0
        )


class ProvenanceTests(unittest.TestCase):
    def test_training_partition_identity_accepts_gate_and_rejects_wrong_full_set(self):
        gate = training_partition_identity(("a", "b", "c", "d"))
        self.assertEqual(gate["selected_family_count"], 4)
        self.assertFalse(gate["protected_payload_content_present"])
        with self.assertRaises(GraphEncoderError) as caught:
            training_partition_identity(tuple(
                "family-{:03d}".format(index) for index in range(407)
            ))
        self.assertEqual(caught.exception.code, "partition_identity_mismatch")

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="ge1-c6-provenance-")
        self.root = Path(self.temporary.name)
        (self.root / "prototype").mkdir()
        (self.root / "prototype" / "source.py").write_text("VALUE = 1\n", encoding="utf-8")
        (self.root / ".gitignore").write_text("outputs/\n", encoding="utf-8")
        self._git("init")
        self._git("config", "user.email", "test@example.com")
        self._git("config", "user.name", "C6 Test")
        self._git("add", ".")
        self._git("commit", "-m", "fixture")
        self.commit = self._git("rev-parse", "HEAD")

    def tearDown(self):
        self.temporary.cleanup()

    def _git(self, *arguments):
        result = subprocess.run(
            ("git",) + arguments,
            cwd=str(self.root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            check=True,
        )
        return result.stdout.strip()

    def _authorize(self):
        return authorize_c6_provenance(
            self.root,
            expected_commit=self.commit,
            configuration_sha256="a" * 64,
            partition_identity_sha256="b" * 64,
            seed=2026,
            encoder_arm="flat",
            device="cpu",
        )

    def test_clean_commit_and_branch_are_accepted(self):
        context = self._authorize()
        self.assertFalse(context.authorized.detached_head)
        self.assertIsNotNone(context.authorized.git_branch)
        self.assertEqual(verify_c6_provenance(context), context.authorized)

    def test_dirty_tree_is_rejected(self):
        context = self._authorize()
        (self.root / "untracked.txt").write_text("dirty\n", encoding="utf-8")
        with self.assertRaises(GraphEncoderError) as caught:
            verify_c6_provenance(context)
        self.assertEqual(caught.exception.code, "dirty_source_tree")

    def test_configuration_and_partition_change_are_rejected(self):
        context = self._authorize()
        with self.assertRaises(GraphEncoderError) as caught:
            verify_c6_provenance(context, configuration_sha256="c" * 64)
        self.assertEqual(caught.exception.code, "configuration_digest_mismatch")
        with self.assertRaises(GraphEncoderError) as caught:
            verify_c6_provenance(context, partition_identity_sha256="d" * 64)
        self.assertEqual(caught.exception.code, "partition_identity_mismatch")

    def test_ignored_outputs_do_not_dirty_or_change_source(self):
        context = self._authorize()
        (self.root / "outputs").mkdir()
        (self.root / "outputs" / "checkpoint.pt").write_text("generated", encoding="utf-8")
        self.assertEqual(verify_c6_provenance(context), context.authorized)

    def test_wrong_commit_rejected(self):
        with self.assertRaises(GraphEncoderError) as caught:
            authorize_c6_provenance(
                self.root, expected_commit="0" * 40,
                configuration_sha256="a" * 64,
                partition_identity_sha256="b" * 64,
                seed=2026, encoder_arm="flat", device="cpu"
            )
        self.assertEqual(caught.exception.code, "wrong_git_commit")

    def test_detached_head_recorded(self):
        self._git("checkout", "--detach", self.commit)
        snapshot = collect_c6_provenance(
            self.root,
            configuration_sha256="a" * 64,
            partition_identity_sha256="b" * 64,
            seed=2026,
            encoder_arm="flat",
            device="cpu",
        )
        self.assertTrue(snapshot.detached_head)
        self.assertIsNone(snapshot.git_branch)


class SourceAuditTests(unittest.TestCase):
    def test_autonomous_public_signatures_have_no_target(self):
        tree = ast.parse((ROOT / "prototype/graph_encoder/autonomous.py").read_text())
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and (
                node.name.startswith("run_autonomous")
                or node.name == "autonomous_input_from_paired"
            ):
                names = [item.arg for item in node.args.args + node.args.kwonlyargs]
                self.assertNotIn("target", names)
                self.assertNotIn("operation_sequence", names)

    def test_one_post_memory_decoder_call_and_no_arm_branch(self):
        source = (ROOT / "prototype/graph_encoder/autonomous.py").read_text()
        tree = ast.parse(source)
        helper = next(
            item for item in tree.body
            if isinstance(item, ast.FunctionDef) and item.name == "_decode_condition"
        )
        text = ast.get_source_segment(source, helper)
        self.assertEqual(text.count("model.decoder("), 1)
        self.assertNotIn("encoder ==", text)
        self.assertNotIn("teacher_forced", text)
        self.assertNotIn("repair", text.lower())

    def test_fixed_checkpoint_selection_audit(self):
        source = (ROOT / "prototype/graph_encoder/training.py").read_text()
        self.assertIn("selected_experimental_checkpoint", source)
        self.assertNotIn("development criterion", source.lower())
        self.assertIn('"best_checkpoint_selection": False', source)

    def test_python38_grammar_for_c6_files(self):
        paths = (
            "provenance.py", "training.py", "autonomous.py", "metrics.py",
            "c6_smoke.py", "tests/test_c6_contract.py",
        )
        for relative in paths:
            path = ROOT / "prototype/graph_encoder" / relative
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path), feature_version=(3, 8))


if __name__ == "__main__":
    unittest.main()
