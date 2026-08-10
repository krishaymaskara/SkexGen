"""Pure/static contracts for the post-C7 optimization diagnostic."""

from __future__ import annotations

import ast
from contextlib import redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

from prototype.graph_encoder.errors import GraphEncoderError
from prototype.graph_encoder import optimization_diagnostic as diagnostic
from prototype.graph_encoder import training


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "prototype/graph_encoder/optimization_diagnostic.py"
SLURM = ROOT / "prototype/graph_encoder/adroit/ge1_optimization_diagnostic_cpu.slurm"


def _metrics(exact=True, condition="P_true"):
    value = 1.0 if exact else 0.0
    family_values = {}
    for family_id in diagnostic.FROZEN_FAMILY_IDS:
        family_values[family_id] = {
            "exact_node_sequence": value,
            "exact_graph": value,
            "strict_conversion": value,
            "complete_executable_validity": value,
            "depends_on_exactness": value,
        }
    return {
        "condition": condition,
        "available": True,
        "family_values": family_values,
    }


def _templates():
    return {
        family: template
        for template, family in diagnostic.FROZEN_FAMILIES_BY_TEMPLATE
    }


def _trajectory(arm, exact_updates=(), plateau=False):
    return tuple({
        "arm": arm,
        "optimizer_update": update,
        "exact_sufficient": update in exact_updates,
        "plateau_observed": plateau,
    } for update in diagnostic.AUTONOMOUS_MILESTONES)


def _artifact_fixture(root, job_id="123456"):
    repository = Path(root) / "repository"
    corpus = Path(root) / "corpus"
    repository.mkdir()
    corpus.mkdir()
    final, staging = diagnostic.prepare_diagnostic_output(
        Path(root) / "outputs" / "optimization-diagnostic",
        repository_root=repository,
        corpus_dir=corpus,
        slurm_job_id=job_id,
    )
    for arm in diagnostic.ARMS:
        directory = staging / "checkpoints" / arm
        directory.mkdir(parents=True)
        for update in diagnostic.CHECKPOINT_UPDATES:
            path = directory / "{}-seed2026-update{:04d}.pt".format(arm, update)
            if diagnostic.torch is None:
                path.write_bytes("{}:{}:{}".format(arm, update, job_id).encode("utf-8"))
            else:
                diagnostic.torch.save({"provenance": {"slurm_job_id": job_id}}, str(path))
    resolved = {
        "diagnostic_version": diagnostic.DIAGNOSTIC_VERSION,
        "trajectory_version": diagnostic.TRAJECTORY_VERSION,
        "artifact_version": diagnostic.ARTIFACT_VERSION,
        "runtime": {"slurm_job_id": job_id},
    }
    interpretation = diagnostic.diagnostic_interpretation({
        "flat": _trajectory("flat", plateau=False),
        "typed_graph": _trajectory("typed_graph", plateau=True),
    })
    events = [{
        "event": "optimization_diagnostic_run_metadata",
        "slurm_job_id": job_id,
    }, {
        "event": "diagnostic_subset_selection",
    }, {
        "event": "matched_fresh_initialization",
    }]
    milestone_events = []
    for arm in diagnostic.ARMS:
        events.append({"event": "diagnostic_training_completed", "arm": arm})
        for update in diagnostic.AUTONOMOUS_MILESTONES:
            milestone_events.append({
                "event": "diagnostic_milestone_measurement",
                "arm": arm,
                "optimizer_update": update,
            })
        for update in diagnostic.CHECKPOINT_UPDATES:
            events.append({
                "event": "checkpoint_provenance",
                "arm": arm,
                "optimizer_update": update,
                "slurm_job_id": job_id,
            })
    events.extend(milestone_events)
    events.append({"event": "diagnostic_interpretation", **interpretation})
    events.append({
        "event": "optimization_diagnostic_completed",
        "slurm_job_id": job_id,
        "original_c7_result_changed": False,
        "stage6_authorized": False,
        "decoder_repair_invoked": False,
        "c8_or_later_performed": False,
        "development_accessed": False,
        "systematic_rr_accessed": False,
        "test_er_accessed": False,
        "iid_accessed": False,
        "history_depth_accessed": False,
        "geometry_extrapolation_accessed": False,
    })
    diagnostic._atomic_write_json(staging / "resolved_config.json", resolved)
    diagnostic._atomic_write_jsonl(staging / "metrics.jsonl", events)
    return final, staging


class FrozenContractTests(unittest.TestCase):
    def test_frozen_identities_family_ids_and_hash(self):
        self.assertEqual(
            diagnostic.DIAGNOSTIC_VERSION,
            "GE1-C7-OPTIMIZATION-SUFFICIENCY-DIAGNOSTIC-v1",
        )
        self.assertEqual(
            diagnostic.TRAJECTORY_VERSION,
            "GE1-C7-OPTIMIZATION-TRAJECTORY-v1",
        )
        self.assertEqual(
            diagnostic.ARTIFACT_VERSION,
            "GE1-C7-OPTIMIZATION-ARTIFACT-v1",
        )
        payload = ("\n".join(diagnostic.FROZEN_FAMILY_IDS) + "\n").encode("utf-8")
        self.assertEqual(hashlib.sha256(payload).hexdigest(), diagnostic.FROZEN_FAMILY_IDS_SHA256)
        self.assertEqual(len(diagnostic.FROZEN_FAMILY_IDS), 4)

    def test_existing_selector_must_reproduce_formal_c7_tiny_set(self):
        selection = SimpleNamespace(
            version="GE1-C7-SUFFICIENCY-v1",
            tiny_family_ids=diagnostic.FROZEN_FAMILY_IDS,
            selected_templates=tuple(
                (family, template)
                for template, family in diagnostic.FROZEN_FAMILIES_BY_TEMPLATE
            ),
        )
        record = diagnostic.validate_frozen_selection(selection)
        self.assertTrue(record["metadata_only"])
        self.assertFalse(record["payload_aware_selection"])
        self.assertEqual(record["family_ids_sha256"], diagnostic.FROZEN_FAMILY_IDS_SHA256)
        selection.tiny_family_ids = diagnostic.FROZEN_FAMILY_IDS[:-1]
        with self.assertRaises(GraphEncoderError):
            diagnostic.validate_frozen_selection(selection)

    def test_update_budget_milestones_and_sparse_schedule(self):
        arithmetic = diagnostic.diagnostic_training_arithmetic()
        self.assertEqual(arithmetic["maximum_optimizer_updates"], 500)
        self.assertEqual(arithmetic["training_example_presentations"], 2000)
        self.assertEqual(arithmetic["effective_examples_per_update"], 4)
        self.assertEqual(tuple(arithmetic["scientific_measurement_milestones"]), (50, 100, 200, 500))
        self.assertEqual(diagnostic.CHECKPOINT_UPDATES, tuple(range(25, 501, 25)))
        self.assertEqual(len(diagnostic.CHECKPOINT_UPDATES), 20)
        self.assertEqual(training._validate_checkpoint_schedule(50, None, "epoch"), frozenset(range(1, 51)))

    def test_training_source_freezes_500_and_has_no_resume_or_early_stop(self):
        source = SOURCE.read_text(encoding="utf-8")
        tree = ast.parse(source)
        calls = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "run_ge1_training"
        ]
        self.assertEqual(len(calls), 1)
        keywords = {item.arg: item.value for item in calls[0].keywords}
        self.assertIsInstance(keywords["final_epoch"], ast.Name)
        self.assertEqual(keywords["final_epoch"].id, "MAXIMUM_OPTIMIZER_UPDATES")
        self.assertIsInstance(keywords["checkpoint_epochs"], ast.Name)
        self.assertEqual(keywords["checkpoint_epochs"].id, "CHECKPOINT_UPDATES")
        self.assertIsInstance(keywords["selected_checkpoint_epoch"], ast.Constant)
        self.assertIsNone(keywords["selected_checkpoint_epoch"].value)
        self.assertNotIn("resume_checkpoint", keywords)
        self.assertNotIn("break", source)

    def test_cli_exposes_only_four_governed_paths(self):
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        options = []
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"
                and node.args
                and isinstance(node.args[0], ast.Constant)
            ):
                options.append(node.args[0].value)
        self.assertEqual(options, [
            "--corpus-dir", "--output-dir", "--repository-root", "--expected-commit"
        ])

    def test_source_does_not_import_or_call_protected_loader(self):
        source = SOURCE.read_text(encoding="utf-8")
        tree = ast.parse(source)
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        imported = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            for alias in node.names
        }
        self.assertNotIn("load_development", names | imported)
        self.assertNotIn("secondary_systematic_validation", source)
        self.assertNotIn("load_partition_physical_examples", source)

    def test_slurm_job_id_is_nonempty_decimal(self):
        self.assertEqual(diagnostic.validate_slurm_job_id("3344505"), "3344505")
        for value in (None, "", "abc", "12-3", 123):
            with self.subTest(value=value), self.assertRaises(GraphEncoderError):
                diagnostic.validate_slurm_job_id(value)


class ExactnessAndInterpretationTests(unittest.TestCase):
    def test_only_autonomous_p_true_exact_unrounded_family_values_pass(self):
        record = diagnostic.exact_sufficiency_at_milestone(
            arm="flat",
            optimizer_update=50,
            condition_metrics=_metrics(True),
            templates_by_family=_templates(),
            checkpoint_identity="update0050.pt",
        )
        self.assertTrue(record["exact_sufficient"])
        nonexact = _metrics(True)
        nonexact["family_values"][diagnostic.FROZEN_FAMILY_IDS[0]]["exact_graph"] = 0.999999
        record = diagnostic.exact_sufficiency_at_milestone(
            arm="flat", optimizer_update=50, condition_metrics=nonexact,
            templates_by_family=_templates(), checkpoint_identity="update0050.pt",
        )
        self.assertFalse(record["exact_sufficient"])
        with self.assertRaises(GraphEncoderError):
            diagnostic.exact_sufficiency_at_milestone(
                arm="flat", optimizer_update=50,
                condition_metrics=_metrics(True, "teacher_forced"),
                templates_by_family=_templates(), checkpoint_identity="update0050.pt",
            )

    def test_depends_on_is_required_for_ee_and_re(self):
        metrics = _metrics(True)
        ee_family = dict(diagnostic.FROZEN_FAMILIES_BY_TEMPLATE)["EE"]
        metrics["family_values"][ee_family]["depends_on_exactness"] = 0.0
        record = diagnostic.exact_sufficiency_at_milestone(
            arm="typed_graph", optimizer_update=100, condition_metrics=metrics,
            templates_by_family=_templates(), checkpoint_identity="update0100.pt",
        )
        self.assertFalse(record["exact_sufficient"])
        self.assertEqual(record["failing_family_ids"], [ee_family])

    def test_all_five_interpretation_categories(self):
        both = diagnostic.diagnostic_interpretation({
            "flat": _trajectory("flat", (100, 200, 500)),
            "typed_graph": _trajectory("typed_graph", (200, 500)),
        })
        one = diagnostic.diagnostic_interpretation({
            "flat": _trajectory("flat", (500,)),
            "typed_graph": _trajectory("typed_graph", plateau=True),
        })
        improving = diagnostic.diagnostic_interpretation({
            "flat": _trajectory("flat", plateau=False),
            "typed_graph": _trajectory("typed_graph", plateau=True),
        })
        plateau = diagnostic.diagnostic_interpretation({
            "flat": _trajectory("flat", plateau=True),
            "typed_graph": _trajectory("typed_graph", plateau=True),
        })
        infrastructure = diagnostic.diagnostic_interpretation({}, infrastructure_failure=True)
        self.assertEqual(both["interpretation_category"], "undertraining_supported_both_arms")
        self.assertEqual(one["interpretation_category"], "undertraining_or_optimization_difference_supported_one_arm")
        self.assertEqual(improving["interpretation_category"], "still_improving_at_update_500")
        self.assertEqual(plateau["interpretation_category"], "plateaued_without_exact_sufficiency")
        self.assertEqual(infrastructure["interpretation_category"], "infrastructure_failure")
        for record in (both, one, improving, plateau, infrastructure):
            self.assertFalse(record["original_c7_result_changed"])
            self.assertFalse(record["stage6_authorized"])
            self.assertFalse(record["decoder_repair_invoked"])
            self.assertFalse(record["c8_or_later_performed"])

    def test_formal_c7_decision_fields_are_rejected(self):
        for name in diagnostic.FORBIDDEN_DECISION_FIELDS:
            with self.assertRaises(GraphEncoderError):
                diagnostic._assert_no_formal_c7_decision_fields({name: False})


class ArtifactContractTests(unittest.TestCase):
    def test_completed_negative_main_exits_zero(self):
        result = {
            "artifact_path": "/external/diagnostic",
            "interpretation": {
                "interpretation_category": "still_improving_at_update_500"
            },
        }
        arguments = (
            "--corpus-dir", "/corpus",
            "--output-dir", "/external/diagnostic",
            "--repository-root", "/repository",
            "--expected-commit", "a" * 40,
        )
        with mock.patch.dict(os.environ, {"SLURM_JOB_ID": "123456"}), mock.patch.object(
            diagnostic, "run_optimization_diagnostic", return_value=result
        ):
            with redirect_stdout(io.StringIO()):
                self.assertEqual(diagnostic.main(arguments), 0)

    def test_completed_negative_diagnostic_finalizes_and_verifies(self):
        with tempfile.TemporaryDirectory() as root:
            final, staging = _artifact_fixture(root)
            diagnostic.finalize_diagnostic_artifact(staging, final)
            verified = diagnostic.verify_diagnostic_artifact(
                final, expected_slurm_job_id="123456"
            )
            self.assertEqual(verified["slurm_job_id"], "123456")
            self.assertEqual(verified["interpretation_category"], "still_improving_at_update_500")
            self.assertTrue(final.is_dir())
            self.assertFalse(staging.exists())

    def test_artifact_mutation_is_detected(self):
        with tempfile.TemporaryDirectory() as root:
            final, staging = _artifact_fixture(root)
            diagnostic.finalize_diagnostic_artifact(staging, final)
            with (final / "metrics.jsonl").open("ab") as stream:
                stream.write(b" ")
            with self.assertRaises(GraphEncoderError) as caught:
                diagnostic.verify_diagnostic_artifact(
                    final, expected_slurm_job_id="123456"
                )
            self.assertEqual(caught.exception.code, "diagnostic_artifact_integrity_failure")

    def test_inconsistent_slurm_identity_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            final, staging = _artifact_fixture(root)
            diagnostic.finalize_diagnostic_artifact(staging, final)
            with self.assertRaises(GraphEncoderError):
                diagnostic.verify_diagnostic_artifact(
                    final, expected_slurm_job_id="999999"
                )

    def test_infrastructure_failure_does_not_finalize(self):
        with tempfile.TemporaryDirectory() as root:
            repository = Path(root) / "repository"
            corpus = Path(root) / "corpus"
            repository.mkdir()
            corpus.mkdir()
            final, staging = diagnostic.prepare_diagnostic_output(
                Path(root) / "outputs" / "failed",
                repository_root=repository,
                corpus_dir=corpus,
                slurm_job_id="123456",
            )
            with self.assertRaises(GraphEncoderError):
                diagnostic.finalize_diagnostic_artifact(staging, final)
            self.assertFalse(final.exists())
            self.assertTrue(staging.exists())


class RunnerStaticTests(unittest.TestCase):
    def test_runner_forwards_slurm_identity_through_cleanenv(self):
        source = SLURM.read_text(encoding="utf-8")
        self.assertIn('--cleanenv', source)
        self.assertIn('--env SLURM_JOB_ID="$SLURM_JOB_ID"', source)
        self.assertIn('[[ "$SLURM_JOB_ID" =~ ^[0-9]+$ ]]', source)

    def test_source_and_test_gates_precede_corpus_access(self):
        source = SLURM.read_text(encoding="utf-8")
        access = source.index("-m prototype.graph_encoder.optimization_diagnostic")
        self.assertLess(source.index("test_optimization_diagnostic_contract"), access)
        self.assertLess(source.index("unittest.defaultTestLoader.discover"), access)
        self.assertLess(source.index("tools/check_documentation.py"), access)
        self.assertLess(source.index("git diff --check"), access)
        self.assertGreater(source.rindex("git status --porcelain", access), access)


if __name__ == "__main__":
    unittest.main()
