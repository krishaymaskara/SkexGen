"""PyTorch-independent prospective C7-v2 protocol and artifact tests."""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

from prototype.graph_encoder import c7_v2
from prototype.graph_encoder import optimization_diagnostic
from prototype.graph_encoder import pilot
from prototype.graph_encoder.errors import GraphEncoderError
from prototype.graph_encoder.training import _validate_execution_epoch


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "prototype/graph_encoder/c7_v2.py"
SLURM = ROOT / "prototype/graph_encoder/adroit/ge1_c7_v2_cpu.slurm"


def _family_fixture(per_template):
    templates = {}
    values = {}
    for template in ("E", "R", "EE", "RE"):
        for index in range(per_template):
            family_id = "{}_{:02d}".format(template, index)
            templates[family_id] = template
            values[family_id] = {
                "exact_node_sequence": 1.0,
                "exact_graph": 1.0,
                "strict_conversion": 1.0,
                "complete_executable_validity": 1.0,
                "depends_on_exactness": 1.0,
            }
    return dict(sorted(templates.items())), dict(sorted(values.items()))


def _gate(arm, name, status="pass"):
    return {
        "version": c7_v2.C7_V2_PROTOCOL_VERSION,
        "gate_name": name,
        "status": status,
        "arm": arm,
    }


def _all_gates(status="pass"):
    return {
        name: _gate(name.split(".")[0], name.split(".")[1], status)
        for name in c7_v2.C7_V2_REQUIRED_GATE_NAMES
    }


def _artifact_fixture(root, *, overall=False, scaled=False):
    root = Path(root)
    repository = root / "repository"
    corpus = root / "corpus"
    repository.mkdir()
    corpus.mkdir()
    final, staging = c7_v2.prepare_c7_v2_output(
        root / "outputs/c7-v2",
        repository_root=repository,
        corpus_dir=corpus,
        job_id="123456",
    )
    identities = [
        "checkpoints/tiny/flat/epoch-0200.pt",
        "checkpoints/tiny/typed_graph/epoch-0200.pt",
    ]
    if scaled:
        identities.extend([
            "checkpoints/scaled/flat/epoch-0200.pt",
            "checkpoints/scaled/typed_graph/epoch-0200.pt",
        ])
    for identity in identities:
        path = staging / identity
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((identity + "\n").encode("ascii"))
    commit = "a" * 40
    c7_v2._atomic_write_json(staging / "resolved_config.json", {
        "protocol_version": c7_v2.C7_V2_PROTOCOL_VERSION,
        "metrics_version": c7_v2.C7_V2_METRICS_VERSION,
        "artifact_version": c7_v2.C7_V2_ARTIFACT_VERSION,
        "source": {"git_commit": commit, "git_dirty": False},
        "checkpoint_selection": {"selected_epoch": 200},
        "runtime": {
            "python": "3.8.13",
            "pytorch": "1.11.0",
            "device": "cpu",
            "cuda_available": False,
            "slurm_job_id": "123456",
        },
        "subset_selection": {
            "tiny_family_ids_sha256": c7_v2.C7_V2_TINY_FAMILY_IDS_SHA256,
            "scaled_family_ids_sha256": c7_v2.C7_V2_SCALED_FAMILY_IDS_SHA256,
        },
        "c7_v2_configuration": c7_v2.C7V2Configuration().to_dict(),
        "training_arithmetic": {
            "tiny": c7_v2.c7_v2_training_arithmetic(4),
            "scaled": c7_v2.c7_v2_training_arithmetic(32),
            "eventual_stage6_if_authorized": (
                c7_v2.c7_v2_training_arithmetic(407)
            ),
        },
        "historical_records": {
            "formal_c7_v1": {
                "checkpoint_epoch": 50,
                "immutable_result_changed": False,
            },
            "optimization_diagnostic": {
                "result_changed": False,
                "checkpoints_reused": False,
            },
        },
    })
    decision = {
        "c7_v2_overall_gate_pass": overall,
        "stage6_authorized_by_c7_v2": overall,
        "preauthorized_decoder_repair_path_next": not overall,
        "comparison_inconclusive": False,
        "c7_v2_execution_completed": True,
    }
    gates = _all_gates()
    if not overall:
        gates["flat.tiny_exact_sufficiency"]["status"] = "fail"
        for arm in c7_v2.C7_V2_ARMS:
            for gate_name in (
                "scaled_exact_sufficiency", "scaled_memory_use"
            ):
                gates["{}.{}".format(arm, gate_name)] = c7_v2.not_run_gate(
                    arm=arm,
                    gate_name=gate_name,
                    reason="both_tiny_exact_sufficiency_gates_did_not_pass",
                    access_flags={"scaled_train_payload_accessed": False},
                )
    protected = {
        name: False for name in c7_v2.C7_V2_PROTECTED_ACCESS_FIELDS
    }
    access = {
        "operation_template_manifest_accessed": True,
        "operation_template_train_payload_accessed": True,
        "tiny_train_payload_accessed": True,
        "scaled_train_payload_accessed": scaled,
        "stage6_performed": False,
        "c8_or_later_performed": False,
        "decoder_repair_implemented_or_invoked": False,
        **protected,
    }
    events = [
        {"event": "c7_v2_run_metadata"},
        {"event": "c7_v2_subset_selection"},
        {"event": "c7_v2_training_started"},
    ]
    for identity in identities:
        digest = c7_v2._file_sha256(staging / identity)
        events.extend([
            {
                "event": "c7_v2_epoch_200_checkpoint_written",
                "checkpoint_identity": identity,
                "checkpoint_sha256": digest,
            },
            {
                "event": "c7_v2_checkpoint_reloaded",
                "checkpoint_identity": identity,
                "checkpoint_sha256": digest,
                "epoch": 200,
                "strict_reload": True,
                "fresh_model_constructed": True,
                "source_commit": commit,
                "slurm_job_id": "123456",
            },
        ])
    events.extend([
        {"event": "c7_v2_autonomous_condition_metrics"},
        *(
            {"event": "c7_v2_gate_result", **gates[name]}
            for name in sorted(gates)
        ),
        {"event": "c7_v2_overall_gate_decision", **decision},
        {
            "event": "c7_v2_execution_completed",
            "slurm_job_id": "123456",
            **decision,
            **access,
        },
    ])
    c7_v2._atomic_write_jsonl(staging / "metrics.jsonl", events)
    return final, staging, commit


class FrozenProtocolTests(unittest.TestCase):
    def test_histories_and_v2_identities_are_distinct(self):
        self.assertEqual(pilot.C7_CHECKPOINT_EPOCH, 50)
        self.assertEqual(pilot.C7_GATE_VERSION, "GE1-C7-SUFFICIENCY-GATES-v1")
        self.assertEqual(
            c7_v2.C7_V2_PROTOCOL_VERSION, "GE1-C7-SUFFICIENCY-v2"
        )
        self.assertEqual(c7_v2.C7_V2_METRICS_VERSION, "GE1-C7-METRICS-v2")
        self.assertEqual(c7_v2.C7_V2_ARTIFACT_VERSION, "GE1-C7-ARTIFACT-v2")
        self.assertEqual(c7_v2.C7_V2_CHECKPOINT_EPOCH, 200)
        self.assertEqual(
            optimization_diagnostic.MAXIMUM_OPTIMIZER_UPDATES, 500
        )

    def test_frozen_arithmetic(self):
        tiny = c7_v2.c7_v2_training_arithmetic(4)
        scaled = c7_v2.c7_v2_training_arithmetic(32)
        stage6 = c7_v2.c7_v2_training_arithmetic(407)
        self.assertEqual((tiny["optimizer_steps"], tiny["training_example_presentations"]), (200, 800))
        self.assertEqual((scaled["optimizer_steps"], scaled["training_example_presentations"]), (800, 6400))
        self.assertEqual((stage6["optimizer_steps"], stage6["training_example_presentations"]), (10200, 81400))

    def test_configuration_forbids_all_adaptive_or_resume_paths(self):
        record = c7_v2.C7V2Configuration().to_dict()
        for name in (
            "early_stopping", "best_checkpoint_selection",
            "outcome_dependent_extension", "warm_start", "resume_from_c7_v1",
            "resume_from_optimization_diagnostic", "reuse_diagnostic_checkpoints",
        ):
            self.assertFalse(record[name], name)
        self.assertEqual(record["checkpoint_selection"], "fixed_epoch_200")

    def test_training_loop_opt_in_preserves_default_and_diagnostic_paths(self):
        _validate_execution_epoch(50, False)
        _validate_execution_epoch(500, False, extended_final_epoch=500)
        _validate_execution_epoch(200, False, fixed_protocol_final_epoch=200)
        for kwargs in (
            {"fixed_protocol_final_epoch": 199},
            {"fixed_protocol_final_epoch": 200, "extended_final_epoch": 200},
        ):
            with self.assertRaises(GraphEncoderError):
                _validate_execution_epoch(200, False, **kwargs)

    def test_frozen_cohort_hashes_are_explicit(self):
        self.assertEqual(
            c7_v2.C7_V2_TINY_FAMILY_IDS_SHA256,
            "c07e90675099cb2ea58575fe91683dc80b70cf3fa85fa6fa8de0c44dbe9783d6",
        )
        self.assertEqual(
            c7_v2.C7_V2_SCALED_FAMILY_IDS_SHA256,
            "0888d517561ca24c99065465f8a2378e0e8e457f4bd20024e7f983956456f9ce",
        )

    def test_only_epoch_200_checkpoint_is_gate_eligible(self):
        templates, values = _family_fixture(1)
        arguments = {
            "arm": "flat",
            "subset_identity": "tiny",
            "condition_metrics": {
                "condition": "P_true", "available": True,
                "family_values": values,
            },
            "templates_by_family": templates,
            "checkpoint_identity": "epoch-0200.pt",
            "access_flags": {},
        }
        for epoch in (50, 199, 500):
            with self.subTest(epoch=epoch), self.assertRaises(GraphEncoderError):
                c7_v2.exact_sufficiency_gate(epoch=epoch, **arguments)
        self.assertEqual(
            c7_v2.exact_sufficiency_gate(epoch=200, **arguments)["status"],
            "pass",
        )

    def test_reload_rejects_c7_v1_and_diagnostic_checkpoints(self):
        for epoch, selected in ((50, True), (200, False), (500, False)):
            resume = SimpleNamespace(
                completed_epoch=epoch,
                payload={
                    "completed_epoch": epoch,
                    "selected_checkpoint_epoch": 50 if epoch == 50 else None,
                    "selected_experimental_checkpoint": selected,
                    "provenance": {"git_commit": "a" * 40, "slurm_job_id": "1"},
                },
            )
            with self.subTest(epoch=epoch), self.assertRaises(GraphEncoderError):
                c7_v2.validate_c7_v2_reloaded_checkpoint(
                    resume, expected_commit="a" * 40, job_id="1"
                )


class GateDecisionTests(unittest.TestCase):
    def test_exact_criteria_and_dependency_are_unchanged(self):
        templates, values = _family_fixture(1)
        values["RE_00"]["depends_on_exactness"] = 0.0
        record = c7_v2.exact_sufficiency_gate(
            arm="typed_graph",
            subset_identity="tiny",
            condition_metrics={
                "condition": "P_true", "available": True,
                "family_values": values,
            },
            templates_by_family=templates,
            checkpoint_identity="epoch-0200.pt",
            epoch=200,
            access_flags={},
        )
        self.assertEqual(record["status"], "fail")
        self.assertEqual(record["failing_family_ids"], ["RE_00"])
        self.assertFalse(
            record["criteria"]["exact_graph"]["rounded_for_comparison"]
        )

    def test_both_tiny_arms_must_pass_or_all_scaled_gates_are_not_run(self):
        gates = _all_gates()
        gates["flat.tiny_exact_sufficiency"]["status"] = "fail"
        for arm in c7_v2.C7_V2_ARMS:
            for name in ("scaled_exact_sufficiency", "scaled_memory_use"):
                gates["{}.{}".format(arm, name)] = c7_v2.not_run_gate(
                    arm=arm, gate_name=name, reason="tiny_failed",
                    access_flags={"scaled_train_payload_accessed": False},
                )
        decision = c7_v2.overall_c7_v2_decision(gates)
        self.assertFalse(decision["stage6_authorized_by_c7_v2"])
        self.assertTrue(decision["preauthorized_decoder_repair_path_next"])

    def test_scaled_exact_failure_returns_to_repair(self):
        gates = _all_gates()
        gates["typed_graph.scaled_exact_sufficiency"]["status"] = "fail"
        decision = c7_v2.overall_c7_v2_decision(gates)
        self.assertTrue(decision["preauthorized_decoder_repair_path_next"])
        self.assertFalse(decision["comparison_inconclusive"])

    def test_memory_only_failure_is_inconclusive(self):
        gates = _all_gates()
        gates["flat.scaled_memory_use"]["status"] = "fail"
        decision = c7_v2.overall_c7_v2_decision(gates)
        self.assertTrue(decision["comparison_inconclusive"])
        self.assertFalse(decision["preauthorized_decoder_repair_path_next"])
        self.assertFalse(decision["stage6_authorized_by_c7_v2"])

    def test_memory_thresholds_are_the_unchanged_exact_boundaries(self):
        self.assertEqual(c7_v2.C7_V2_MEMORY_RATIO_MAX, 0.80)
        at_boundary = c7_v2._memory_criterion(
            0.80, comparison="less_than_or_equal", threshold=0.80
        )
        above_boundary = c7_v2._memory_criterion(
            0.8000000001, comparison="less_than_or_equal", threshold=0.80
        )
        zero_true = c7_v2._memory_criterion(
            0.0, comparison="greater_than", threshold=0.0
        )
        self.assertTrue(at_boundary["passed"])
        self.assertFalse(above_boundary["passed"])
        self.assertFalse(zero_true["passed"])

    def test_only_all_six_passes_authorize_stage6(self):
        decision = c7_v2.overall_c7_v2_decision(_all_gates())
        self.assertTrue(decision["c7_v2_overall_gate_pass"])
        self.assertTrue(decision["stage6_authorized_by_c7_v2"])

    def test_infrastructure_failure_is_non_scientific(self):
        decision = c7_v2.overall_c7_v2_decision(
            {}, infrastructure_failure=True
        )
        self.assertFalse(decision["c7_v2_execution_completed"])
        self.assertFalse(decision["preauthorized_decoder_repair_path_next"])


class ArtifactAndExitTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="ge1-c7-v2-")

    def tearDown(self):
        self.temporary.cleanup()

    def test_completed_scientific_failure_finalizes_and_verifies(self):
        final, staging, commit = _artifact_fixture(self.temporary.name)
        c7_v2.finalize_c7_v2_artifact(staging, final)
        verified = c7_v2.verify_c7_v2_artifact(
            final, expected_commit=commit, expected_slurm_job_id="123456"
        )
        self.assertFalse(verified["c7_v2_overall_gate_pass"])
        self.assertEqual(verified["checkpoint_count"], 2)

    def test_scaled_artifact_requires_four_checkpoints(self):
        final, staging, unused_commit = _artifact_fixture(
            self.temporary.name, overall=True, scaled=True
        )
        c7_v2.finalize_c7_v2_artifact(staging, final)
        self.assertEqual(c7_v2.verify_c7_v2_artifact(final)["checkpoint_count"], 4)

    def test_commit_slurm_checksum_and_clean_source_are_enforced(self):
        final, staging, commit = _artifact_fixture(self.temporary.name)
        c7_v2.finalize_c7_v2_artifact(staging, final)
        for kwargs in (
            {"expected_commit": "b" * 40},
            {"expected_slurm_job_id": "999"},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(GraphEncoderError):
                c7_v2.verify_c7_v2_artifact(final, **kwargs)
        with (final / "metrics.jsonl").open("ab") as stream:
            stream.write(b"{}\n")
        with self.assertRaises(GraphEncoderError) as caught:
            c7_v2.verify_c7_v2_artifact(final, expected_commit=commit)
        self.assertEqual(
            caught.exception.code, "c7_v2_artifact_integrity_failure"
        )

    def test_dirty_source_or_protected_access_cannot_finalize(self):
        final, staging, unused_commit = _artifact_fixture(self.temporary.name)
        resolved_path = staging / "resolved_config.json"
        resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
        resolved["source"]["git_dirty"] = True
        c7_v2._atomic_write_json(resolved_path, resolved)
        with self.assertRaises(GraphEncoderError):
            c7_v2.finalize_c7_v2_artifact(staging, final)

        with tempfile.TemporaryDirectory(prefix="ge1-c7-v2-access-") as other:
            final, staging, unused_commit = _artifact_fixture(other)
            events = c7_v2._read_canonical_jsonl(staging / "metrics.jsonl")
            events[-1]["systematic_rr_accessed"] = True
            c7_v2._atomic_write_jsonl(staging / "metrics.jsonl", events)
            with self.assertRaises(GraphEncoderError):
                c7_v2.finalize_c7_v2_artifact(staging, final)

    def test_scientific_failure_cli_exits_zero_and_infrastructure_exits_nonzero(self):
        result = {
            "artifact_path": "/external/c7-v2",
            "overall_decision": c7_v2.overall_c7_v2_decision(
                {
                    **_all_gates(),
                    "flat.tiny_exact_sufficiency": _gate(
                        "flat", "tiny_exact_sufficiency", "fail"
                    ),
                }
            ),
        }
        with mock.patch.object(c7_v2, "run_c7_v2", return_value=result):
            self.assertEqual(c7_v2.main([
                "--corpus-dir", "x", "--output-dir", "y",
                "--repository-root", "z", "--expected-commit", "a" * 40,
            ]), 0)
        with mock.patch.object(
            c7_v2, "run_c7_v2", side_effect=RuntimeError("preflight")
        ):
            self.assertEqual(c7_v2.main([
                "--corpus-dir", "x", "--output-dir", "y",
                "--repository-root", "z", "--expected-commit", "a" * 40,
            ]), 1)


class StaticProductionTests(unittest.TestCase):
    def test_protected_or_unrestricted_loaders_are_unreachable(self):
        source = SOURCE.read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported = {
            alias.name for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) for alias in node.names
        }
        self.assertNotIn("load_development", imported)
        self.assertNotIn("load_partition_physical_examples", imported)
        self.assertNotIn("secondary_systematic_validation", source)

    def test_scaled_payload_call_is_guarded_by_both_tiny_pass(self):
        source = SOURCE.read_text(encoding="utf-8")
        tree = ast.parse(source)
        run = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "run_c7_v2"
        )
        branch = next(
            node for node in run.body
            if isinstance(node, ast.If)
            and isinstance(node.test, ast.Name)
            and node.test.id == "both_tiny_pass"
        )
        calls = [
            node for node in ast.walk(branch)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_load_selected_train"
        ]
        self.assertEqual(len(calls), 1)
        keywords = {item.arg: item.value for item in calls[0].keywords}
        self.assertEqual(keywords["subset_identity"].value, "scaled")

    def test_no_repair_stage6_or_c8_implementation_call_exists(self):
        source = SOURCE.read_text(encoding="utf-8")
        tree = ast.parse(source)
        called = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name):
                called.add(node.func.id.lower())
            elif isinstance(node.func, ast.Attribute):
                called.add(node.func.attr.lower())
        self.assertFalse({
            name for name in called
            if name.startswith("c8") or name.startswith("stage6")
            or "hierarchical" in name or name == "repair_decoder"
        })

    def test_training_call_has_no_resume_and_one_selected_checkpoint(self):
        source = SOURCE.read_text(encoding="utf-8")
        tree = ast.parse(source)
        function = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_train_evaluate_arm"
        )
        call = next(
            node for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "run_ge1_training"
        )
        keywords = {item.arg: item.value for item in call.keywords}
        self.assertNotIn("resume_checkpoint", keywords)
        self.assertIn("fixed_protocol_final_epoch", keywords)
        self.assertIn("checkpoint_epochs", keywords)
        self.assertEqual(keywords["selected_checkpoint_epoch"].id, "C7_V2_CHECKPOINT_EPOCH")

    def test_slurm_runner_exists_and_is_standalone_checkout_scoped(self):
        source = SLURM.read_text(encoding="utf-8")
        for text in (
            "test -d \"$REPOSITORY/.git\"",
            "test ! -f \"$REPOSITORY/.git\"",
            "git rev-parse HEAD",
            "git status --porcelain=v1 --untracked-files=all",
            "-m prototype.graph_encoder.c7_v2",
            "stage6_authorized_by_c7_v2",
        ):
            self.assertIn(text, source)
        self.assertNotIn("git worktree", source)

    def test_slurm_preflights_precede_manifest_access(self):
        source = SLURM.read_text(encoding="utf-8")
        execution = source.index("-m prototype.graph_encoder.c7_v2")
        for marker in (
            "c7_v2_focused_suite",
            "c7_v2_complete_graph_encoder_suite",
            "documentation_validation=PASS",
            "graph_encoder_compileall=PASS",
            "C7_V2_SOURCE_ACCESS_AUDIT=PASS",
        ):
            self.assertLess(source.index(marker), execution, marker)

    def test_bash_grammar(self):
        self.assertEqual(os.system("bash -n {}".format(SLURM)), 0)


if __name__ == "__main__":
    unittest.main()
