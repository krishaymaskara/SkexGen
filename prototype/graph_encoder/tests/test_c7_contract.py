"""PyTorch-independent C7 selection, gate, artifact, and runner contracts."""

from __future__ import annotations

import ast
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

from prototype.graph_encoder.errors import GraphEncoderError
from prototype.graph_encoder import partitions
from prototype.graph_encoder import pilot


ROOT = Path(__file__).resolve().parents[3]
SLURM = ROOT / "prototype/graph_encoder/adroit/ge1_pilot_cpu.slurm"
PILOT = ROOT / "prototype/graph_encoder/pilot.py"


def _verified_selection_fixture(reversed_records=False):
    family_ids = []
    templates = []
    for template in partitions.C7_ACCESSIBLE_TEMPLATES:
        for index in range(10):
            family_id = "{}_family_{:02d}".format(template, index)
            family_ids.append(family_id)
            templates.append((family_id, template))
    if reversed_records:
        family_ids.reverse()
        templates.reverse()
    return partitions._VerifiedManifest(
        (("train", tuple(family_ids)),), tuple(templates)
    )


def _family_fixture(per_template):
    templates = {}
    values = {}
    for template in partitions.C7_ACCESSIBLE_TEMPLATES:
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


def _condition_metrics(family_ids, true=0.5, shuffled=0.4, mean=0.4):
    result = []
    for name, value in (
        ("P_true", true), ("P_shuffle", shuffled), ("P_mean", mean)
    ):
        result.append({
            "schema_version": "GE1-C6-METRICS-v2",
            "condition": name,
            "available": True,
            "unavailable_reason": None,
            "family_values": {family_id: {} for family_id in family_ids},
            "aggregates": {
                "primary": {
                    "value": value,
                    "defined_family_denominator": len(family_ids),
                    "physical_family_denominator": len(family_ids),
                    "undefined_reason": None,
                },
                "geometry_error_by_channel_family": {},
            },
            "physical_family_denominator": len(family_ids),
            "metric_computation_seconds": 0.0,
        })
    return tuple(result)


def _autonomous_result(family_ids):
    batches = tuple(
        ("batch-{}".format(index // 8), tuple(family_ids[index:index + 8]))
        for index in range(0, len(family_ids), 8)
    )
    shifted = tuple(family_ids[1:]) + tuple(family_ids[:1])
    assignments = {
        "P_true": tuple((item, item) for item in family_ids),
        "P_shuffle": tuple(zip(family_ids, shifted)),
        "P_mean": tuple((item, "batch_mean:fixture") for item in family_ids),
    }
    conditions = tuple(
        SimpleNamespace(
            condition=name,
            available=True,
            unavailable_reason=None,
            memory_assignments=assignments[name],
            memory_altered=name != "P_true",
            alteration_possible=name != "P_true",
            batch_membership=batches,
            elapsed_seconds=0.0,
            intervention_seconds=0.0,
        )
        for name in ("P_true", "P_shuffle", "P_mean")
    )
    return SimpleNamespace(
        family_order=tuple(family_ids),
        encoding_seconds=0.0,
        conditions=conditions,
    )


def _exact_gate(arm, name, status="pass"):
    return {
        "version": pilot.C7_GATE_VERSION,
        "gate_name": name,
        "status": status,
        "arm": arm,
    }


def _all_gates(status="pass"):
    return {
        name: _exact_gate(name.split(".")[0], name.split(".")[1], status)
        for name in pilot.C7_REQUIRED_GATE_NAMES
    }


def _artifact_fixture(root, overall=True):
    repository = Path(root) / "repository"
    corpus = Path(root) / "corpus"
    repository.mkdir()
    corpus.mkdir()
    final, staging = pilot.prepare_c7_output(
        Path(root) / "outputs" / "formal-c7",
        repository_root=repository,
        corpus_dir=corpus,
    )
    checkpoint = staging / "checkpoints/tiny/flat/epoch-0050.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"procedural-checkpoint")
    pilot._atomic_write_json(staging / "resolved_config.json", {
        "pilot_version": pilot.C7_PILOT_VERSION,
        "gate_version": pilot.C7_GATE_VERSION,
        "artifact_manifest_version": pilot.C7_ARTIFACT_MANIFEST_VERSION,
    })
    decision = {
        "overall_gate_pass": overall,
        "stage6_authorized_by_c7": overall,
        "preauthorized_decoder_repair_triggered": not overall,
        "comparison_inconclusive": False,
        "c7_execution_completed": True,
    }
    events = [
        {"event": "run_metadata"},
        {"event": "subset_selection"},
        {"event": "training_started"},
        {"event": "training_completed"},
        {"event": "checkpoint_reloaded"},
        {"event": "autonomous_condition_metrics"},
        {"event": "gate_result"},
        {"event": "overall_gate_decision", **decision},
        {
            "event": "c7_pilot_execution_completed",
            **decision,
            "operation_template_manifest_accessed": True,
            "operation_template_train_payload_accessed": True,
            "development_accessed": False,
            "systematic_rr_accessed": False,
            "test_er_accessed": False,
            "iid_accessed": False,
            "history_depth_accessed": False,
            "geometry_extrapolation_accessed": False,
            "c8_or_later_performed": False,
        },
    ]
    pilot._atomic_write_jsonl(staging / "metrics.jsonl", events)
    return final, staging


class SelectionContractTests(unittest.TestCase):
    def test_exact_ranking_bytes(self):
        template = "EE"
        family = "source-family"
        material = (
            "GE1-C7-SUFFICIENCY-v1\nEE\nsource-family\n"
        ).encode("utf-8")
        self.assertEqual(
            partitions.c7_family_rank(template, family),
            hashlib.sha256(material).hexdigest(),
        )

    def test_selection_is_deterministic_under_reordered_records(self):
        first = partitions._select_c7_from_verified(
            _verified_selection_fixture(False)
        )
        second = partitions._select_c7_from_verified(
            _verified_selection_fixture(True)
        )
        self.assertEqual(first, second)

    def test_exact_counts_templates_and_nestedness(self):
        selection = partitions._select_c7_from_verified(
            _verified_selection_fixture()
        )
        self.assertEqual(len(selection.tiny_family_ids), 4)
        self.assertEqual(len(selection.scaled_family_ids), 32)
        self.assertEqual(len(set(selection.tiny_family_ids)), 4)
        self.assertEqual(len(set(selection.scaled_family_ids)), 32)
        self.assertTrue(
            set(selection.tiny_family_ids).issubset(selection.scaled_family_ids)
        )
        selected_templates = dict(selection.selected_templates)
        for template in partitions.C7_ACCESSIBLE_TEMPLATES:
            self.assertEqual(
                sum(value == template for value in selected_templates.values()),
                8,
            )

    def test_selected_set_hash_uses_real_lf_and_final_lf(self):
        values = ("a", "b", "c")
        expected = hashlib.sha256(b"a\nb\nc\n").hexdigest()
        self.assertEqual(partitions.selected_family_ids_sha256(values), expected)
        with self.assertRaises(GraphEncoderError):
            partitions.selected_family_ids_sha256(tuple(reversed(values)))

    def test_public_selector_is_metadata_only(self):
        verified = _verified_selection_fixture()
        with mock.patch.object(
            partitions, "_verify_authoritative_manifest", return_value=verified
        ) as authority, mock.patch.object(
            partitions, "load_partition_physical_examples"
        ) as payload:
            selection = partitions.select_c7_sufficiency_subsets("fixture")
        authority.assert_called_once_with("fixture")
        payload.assert_not_called()
        self.assertFalse(selection.to_dict()["cad_history_payload_accessed"])

    def test_wrong_authority_is_terminal_before_payload(self):
        failure = GraphEncoderError("manifest_authority_failure", "wrong hash")
        with mock.patch.object(
            partitions, "_verify_authoritative_manifest", side_effect=failure
        ), mock.patch.object(
            partitions, "load_partition_physical_examples"
        ) as payload:
            with self.assertRaises(GraphEncoderError) as caught:
                partitions.select_c7_sufficiency_subsets("fixture")
        self.assertEqual(caught.exception.code, "manifest_authority_failure")
        payload.assert_not_called()

    def test_wrong_template_composition_is_rejected(self):
        verified = _verified_selection_fixture()
        templates = list(verified.family_templates)
        templates[0] = (templates[0][0], "ER")
        malformed = replace(verified, family_templates=tuple(templates))
        with self.assertRaises(GraphEncoderError) as caught:
            partitions._select_c7_from_verified(malformed)
        self.assertEqual(caught.exception.code, "invalid_c7_template_composition")


class TrainingLifecycleContractTests(unittest.TestCase):
    def test_gate_seed_and_training_arithmetic_are_frozen(self):
        self.assertEqual(pilot.C7_GATE_SEED, 2026)
        self.assertEqual(pilot.c7_training_arithmetic(4), {
            "family_count": 4,
            "epochs": 50,
            "batch_size": 8,
            "steps_per_epoch": 1,
            "optimizer_steps": 50,
            "training_example_presentations": 200,
        })
        self.assertEqual(
            pilot.c7_training_arithmetic(32)["optimizer_steps"], 200
        )
        self.assertEqual(
            pilot.c7_training_arithmetic(32)["training_example_presentations"],
            1600,
        )

    def test_invalid_gate_subset_size_is_rejected(self):
        for value in (0, 8, 407):
            with self.subTest(value=value), self.assertRaises(GraphEncoderError):
                pilot.c7_training_arithmetic(value)

    def test_strict_reload_requires_selected_epoch_50(self):
        valid = SimpleNamespace(
            completed_epoch=50,
            payload={
                "completed_epoch": 50,
                "selected_experimental_checkpoint": True,
            },
        )
        self.assertIs(pilot.validate_c7_reloaded_checkpoint(valid), valid)
        for bad in (
            SimpleNamespace(completed_epoch=49, payload=valid.payload),
            SimpleNamespace(
                completed_epoch=50,
                payload={
                    "completed_epoch": 50,
                    "selected_experimental_checkpoint": False,
                },
            ),
        ):
            with self.assertRaises(GraphEncoderError):
                pilot.validate_c7_reloaded_checkpoint(bad)

    def test_cli_exposes_only_four_nongoverned_paths(self):
        tree = ast.parse(PILOT.read_text(encoding="utf-8"))
        options = {
            argument.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
            for argument in node.args[:1]
            if isinstance(argument, ast.Constant)
            and isinstance(argument.value, str)
            and argument.value.startswith("--")
        }
        self.assertEqual(options, {
            "--corpus-dir", "--output-dir", "--repository-root",
            "--expected-commit",
        })

    def test_checkpoint_artifact_role_is_gate_only(self):
        training = SimpleNamespace(
            to_dict=lambda: {
                "checkpoint_paths": [],
                "selected_experimental_checkpoint": None,
            }
        )
        record = pilot._training_record_for_artifact(training)
        self.assertEqual(record["checkpoint_role"], pilot.C7_CHECKPOINT_ROLE)
        self.assertFalse(record["full_comparison_checkpoint"])

    def test_production_has_no_tiny_resume_or_alternate_seed_cli(self):
        source = PILOT.read_text(encoding="utf-8")
        self.assertNotIn("resume_checkpoint=", source)
        self.assertNotIn("--seed", source)
        self.assertIn("build_matched_ge1_models(seed=C7_GATE_SEED)", source)

    def test_scaled_memory_remains_diagnostic_after_scaled_exact_failure(self):
        tree = ast.parse(PILOT.read_text(encoding="utf-8"))
        function = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_train_evaluate_arm"
        )
        scaled_branch = next(
            node for node in function.body
            if isinstance(node, ast.If)
            and isinstance(node.test, ast.Compare)
            and any(
                isinstance(value, ast.Constant) and value.value == "scaled"
                for value in (node.test.left,) + tuple(node.test.comparators)
            )
        )
        calls = {
            node.func.id
            for node in ast.walk(scaled_branch)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        self.assertIn("memory_use_gate", calls)
        self.assertFalse(any(
            isinstance(node, ast.If)
            and any(
                isinstance(name, ast.Name) and name.id == "exact"
                for name in ast.walk(node.test)
            )
            for node in ast.walk(scaled_branch)
        ))


class ExactGateTests(unittest.TestCase):
    def _gate(self, per_template=1, values=None, **overrides):
        templates, default_values = _family_fixture(per_template)
        return pilot.exact_sufficiency_gate(
            arm=overrides.pop("arm", "flat"),
            subset_identity="tiny" if per_template == 1 else "scaled",
            condition_metrics={
                "condition": overrides.pop("condition", "P_true"),
                "available": True,
                "family_values": values or default_values,
            },
            templates_by_family=templates,
            checkpoint_identity="checkpoints/epoch-0050.pt",
            epoch=overrides.pop("epoch", 50),
            access_flags={"development_accessed": False},
            generation_condition=overrides.pop(
                "generation_condition", "P_true"
            ),
            **overrides
        )

    def test_every_family_exact_passes(self):
        gate = self._gate()
        self.assertEqual(gate["status"], "pass")
        self.assertFalse(gate["failing_family_ids"])
        self.assertEqual(
            gate["criteria"]["depends_on_exactness_for_EE_RE"]
            ["physical_family_denominator"],
            2,
        )

    def test_one_failed_family_fails_without_rounding(self):
        unused_templates, values = _family_fixture(1)
        values["E_00"]["exact_graph"] = 0.999999999
        gate = self._gate(values=values)
        self.assertEqual(gate["status"], "fail")
        self.assertEqual(gate["failing_family_ids"], ["E_00"])

    def test_depends_on_is_explicit_for_EE_and_RE(self):
        unused_templates, values = _family_fixture(1)
        values["RE_00"]["depends_on_exactness"] = 0.0
        gate = self._gate(values=values)
        self.assertEqual(gate["status"], "fail")
        self.assertEqual(
            gate["failed_fields_by_family"]["RE_00"][0]["field"],
            "depends_on_exactness",
        )

    def test_teacher_forced_success_cannot_satisfy_gate(self):
        for condition in ("teacher_forced", "P_shuffle"):
            with self.subTest(condition=condition), self.assertRaises(
                GraphEncoderError
            ):
                self._gate(
                    condition=condition, generation_condition=condition
                )

    def test_only_epoch_50_is_eligible(self):
        with self.assertRaises(GraphEncoderError) as caught:
            self._gate(epoch=49)
        self.assertEqual(caught.exception.code, "wrong_c7_checkpoint_epoch")

    def test_not_run_is_structured_and_never_passes(self):
        gate = pilot.not_run_gate(
            arm="flat",
            gate_name="scaled_exact_sufficiency",
            subset_identity="scaled",
            reason="tiny_failed",
            access_flags={},
        )
        self.assertEqual(gate["status"], "not_run")
        self.assertNotEqual(gate["status"], "pass")

    def test_scaled_gate_requires_32_families(self):
        templates, values = _family_fixture(1)
        with self.assertRaises(GraphEncoderError):
            pilot.exact_sufficiency_gate(
                arm="flat",
                subset_identity="scaled",
                condition_metrics={
                    "condition": "P_true", "available": True,
                    "family_values": values,
                },
                templates_by_family=templates,
                checkpoint_identity="epoch50",
                epoch=50,
                access_flags={},
            )


class MemoryGateTests(unittest.TestCase):
    def _gate(self, true=0.5, shuffled=0.4, mean=0.4):
        templates, unused_values = _family_fixture(8)
        family_ids = tuple(templates)
        return pilot.memory_use_gate(
            arm="typed_graph",
            condition_metrics=_condition_metrics(
                family_ids, true=true, shuffled=shuffled, mean=mean
            ),
            autonomous_result=_autonomous_result(family_ids),
            templates_by_family=templates,
            checkpoint_identity="checkpoints/scaled/typed_graph/epoch-0050.pt",
            epoch=50,
            access_flags={},
        )

    def test_exact_inclusive_point_eight_boundary_passes(self):
        gate = self._gate()
        self.assertEqual(gate["status"], "pass")
        self.assertEqual(gate["intervention_ratios"]["R_shuffle"], 0.8)
        self.assertEqual(gate["intervention_ratios"]["R_mean"], 0.8)

    def test_value_above_point_eight_fails(self):
        gate = self._gate(shuffled=0.4000000001)
        self.assertEqual(gate["status"], "fail")
        self.assertFalse(gate["criteria"]["R_shuffle_at_most_0_80"]["passed"])

    def test_true_must_be_strictly_positive(self):
        gate = self._gate(true=0.0, shuffled=0.0, mean=0.0)
        self.assertEqual(gate["status"], "fail")
        self.assertEqual(
            gate["intervention_ratios"]["R_shuffle"]["reason"],
            "zero_true_denominator",
        )

    def test_nonfinite_and_bare_null_values_fail(self):
        for value in (float("nan"), float("inf"), None, {}, True):
            with self.subTest(value=value):
                criterion = pilot._memory_criterion(
                    value, comparison="less_than_or_equal", threshold=0.8
                )
                self.assertFalse(criterion["passed"])

    def test_four_sorted_batches_of_eight_are_recorded(self):
        gate = self._gate()
        batches = gate["deterministic_batch_membership"]
        self.assertEqual(len(batches), 4)
        self.assertTrue(all(len(item) == 8 for item in batches))
        flattened = tuple(item for batch in batches for item in batch)
        self.assertEqual(flattened, tuple(sorted(flattened)))
        for criterion in gate["criteria"].values():
            self.assertEqual(criterion["aggregation"], "physical_family_macro")
            self.assertEqual(criterion["physical_family_denominator"], 32)

    def test_unavailable_condition_is_a_scientific_memory_failure(self):
        templates, unused_values = _family_fixture(8)
        family_ids = tuple(templates)
        metrics = list(_condition_metrics(family_ids))
        metrics[1] = dict(metrics[1], available=False)
        gate = pilot.memory_use_gate(
            arm="flat",
            condition_metrics=metrics,
            autonomous_result=_autonomous_result(family_ids),
            templates_by_family=templates,
            checkpoint_identity="checkpoints/scaled/flat/epoch-0050.pt",
            epoch=50,
            access_flags={},
        )
        self.assertEqual(gate["status"], "fail")
        self.assertFalse(gate["criteria"]["R_shuffle_at_most_0_80"]["passed"])

    def test_donor_availability_states_are_exact(self):
        gate = self._gate()
        donor = {
            item["condition"]: item
            for item in gate["donor_template_agreement"]
        }
        self.assertTrue(donor["P_shuffle"]["available"])
        self.assertEqual(
            donor["P_shuffle"]["chance_baseline"],
            "random_distinct_family_donor",
        )
        self.assertFalse(donor["P_true"]["available"])
        self.assertFalse(donor["P_mean"]["available"])

    def test_all_five_primary_fields_are_present(self):
        ratios = self._gate()["intervention_ratios"]
        self.assertEqual(set(ratios), {
            "P_true", "P_shuffle", "P_mean", "R_shuffle", "R_mean"
        })

    def test_c7_envelope_is_separate_from_c6_smoke_identity(self):
        templates, unused_values = _family_fixture(8)
        family_ids = tuple(templates)
        metrics = _condition_metrics(family_ids)
        autonomous = _autonomous_result(family_ids)
        memory_gate = self._gate()
        training = SimpleNamespace(to_dict=lambda: {
            "checkpoint_paths": ["checkpoints/epoch-0050.pt"],
            "selected_experimental_checkpoint": "checkpoints/epoch-0050.pt",
        })
        record = pilot.c7_metrics_envelope(
            arm="flat",
            checkpoint_identity="checkpoints/epoch-0050.pt",
            condition_metrics=metrics,
            autonomous_result=autonomous,
            templates_by_family=templates,
            training_result=training,
            parameter_counts={},
            exact_gate=_exact_gate("flat", "scaled_exact_sufficiency"),
            memory_gate=memory_gate,
            access_flags={"development_accessed": False},
            total_run_seconds=0.0,
        )
        self.assertEqual(record["schema_version"], pilot.C7_GATE_VERSION)
        self.assertEqual(
            record["source_condition_metrics_schema"],
            "GE1-C6-METRICS-v2",
        )
        self.assertTrue(record["scientific_result"])
        self.assertTrue(record["formal_c7_gate"])
        self.assertNotIn("engineering_smoke_only", record)
        self.assertNotIn("c7_or_later_performed", record)


class OverallDecisionTests(unittest.TestCase):
    def test_all_six_passes_authorize_stage6(self):
        decision = pilot.overall_c7_decision(_all_gates())
        self.assertTrue(decision["overall_gate_pass"])
        self.assertTrue(decision["stage6_authorized_by_c7"])
        self.assertFalse(decision["preauthorized_decoder_repair_triggered"])

    def test_exact_failure_triggers_only_preauthorized_repair(self):
        gates = _all_gates()
        gates["flat.tiny_exact_sufficiency"]["status"] = "fail"
        decision = pilot.overall_c7_decision(gates)
        self.assertFalse(decision["overall_gate_pass"])
        self.assertTrue(decision["preauthorized_decoder_repair_triggered"])
        self.assertFalse(decision["stage6_authorized_by_c7"])

    def test_memory_only_failure_is_inconclusive_without_repair(self):
        gates = _all_gates()
        gates["typed_graph.scaled_memory_use"]["status"] = "fail"
        decision = pilot.overall_c7_decision(gates)
        self.assertTrue(decision["comparison_inconclusive"])
        self.assertFalse(decision["preauthorized_decoder_repair_triggered"])

    def test_infrastructure_failure_never_triggers_repair(self):
        decision = pilot.overall_c7_decision(
            {}, infrastructure_failure=True
        )
        self.assertFalse(decision["c7_execution_completed"])
        self.assertFalse(decision["preauthorized_decoder_repair_triggered"])
        self.assertFalse(decision["stage6_authorized_by_c7"])


class ArtifactContractTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="ge1-c7-artifact-")

    def tearDown(self):
        self.temporary.cleanup()

    def test_successful_scientific_pass_finalizes(self):
        final, staging = _artifact_fixture(self.temporary.name, overall=True)
        pilot.finalize_c7_artifact(staging, final)
        verified = pilot.verify_c7_artifact(final)
        self.assertTrue(verified["overall_gate_pass"])
        self.assertFalse(staging.exists())

    def test_completed_scientific_failure_also_finalizes(self):
        final, staging = _artifact_fixture(self.temporary.name, overall=False)
        pilot.finalize_c7_artifact(staging, final)
        verified = pilot.verify_c7_artifact(final)
        self.assertFalse(verified["overall_gate_pass"])
        self.assertFalse(verified["stage6_authorized_by_c7"])

    def test_infrastructure_failure_never_looks_final(self):
        root = Path(self.temporary.name)
        repository = root / "repository"
        corpus = root / "corpus"
        repository.mkdir()
        corpus.mkdir()
        final, staging = pilot.prepare_c7_output(
            root / "outside" / "run",
            repository_root=repository,
            corpus_dir=corpus,
        )
        with self.assertRaises(GraphEncoderError):
            pilot.finalize_c7_artifact(staging, final)
        self.assertFalse(final.exists())
        self.assertIn(".incomplete-", staging.name)

    def test_existing_and_symlink_outputs_are_rejected(self):
        root = Path(self.temporary.name)
        repository = root / "repository"
        corpus = root / "corpus"
        repository.mkdir()
        corpus.mkdir()
        existing = root / "existing"
        existing.mkdir()
        with self.assertRaises(GraphEncoderError):
            pilot.prepare_c7_output(
                existing, repository_root=repository, corpus_dir=corpus
            )
        link = root / "link"
        try:
            os.symlink(str(existing), str(link))
        except OSError as exc:
            self.skipTest("symlinks unavailable: {}".format(exc))
        with self.assertRaises(GraphEncoderError):
            pilot.prepare_c7_output(
                link, repository_root=repository, corpus_dir=corpus
            )

    def test_repository_and_corpus_contained_outputs_are_rejected(self):
        root = Path(self.temporary.name)
        repository = root / "repository"
        corpus = root / "corpus"
        repository.mkdir()
        corpus.mkdir()
        for output in (repository / "run", corpus / "run"):
            with self.subTest(output=output), self.assertRaises(GraphEncoderError):
                pilot.prepare_c7_output(
                    output, repository_root=repository, corpus_dir=corpus
                )

    def test_checksums_are_sorted_and_lf_terminated(self):
        final, staging = _artifact_fixture(self.temporary.name)
        pilot.finalize_c7_artifact(staging, final)
        raw = (final / "SHA256SUMS").read_bytes()
        self.assertTrue(raw.endswith(b"\n"))
        paths = [line.split("  ", 1)[1] for line in raw.decode().splitlines()]
        self.assertEqual(paths, sorted(paths))

    def test_mutation_is_detected(self):
        final, staging = _artifact_fixture(self.temporary.name)
        pilot.finalize_c7_artifact(staging, final)
        with (final / "metrics.jsonl").open("ab") as stream:
            stream.write(b"{}\n")
        with self.assertRaises(GraphEncoderError) as caught:
            pilot.verify_c7_artifact(final)
        self.assertEqual(caught.exception.code, "c7_artifact_integrity_failure")

    def test_unsafe_manifest_path_is_rejected(self):
        final, staging = _artifact_fixture(self.temporary.name)
        pilot.finalize_c7_artifact(staging, final)
        manifest_path = final / "artifact_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["artifacts"][0]["path"] = "../escape"
        pilot._atomic_write_json(manifest_path, manifest)
        with self.assertRaises(GraphEncoderError) as caught:
            pilot.verify_c7_artifact(final)
        self.assertEqual(caught.exception.code, "invalid_c7_artifact")

    def test_canonical_jsonl_and_c7_identity_exclude_false_c6_field(self):
        final, staging = _artifact_fixture(self.temporary.name)
        pilot.finalize_c7_artifact(staging, final)
        events = pilot._read_canonical_jsonl(final / "metrics.jsonl")
        self.assertEqual(events[-1]["event"], "c7_pilot_execution_completed")
        resolved = json.loads(
            (final / "resolved_config.json").read_text(encoding="utf-8")
        )
        self.assertEqual(resolved["gate_version"], pilot.C7_GATE_VERSION)
        self.assertNotIn("c7_or_later_performed", resolved)

    def test_terminal_event_preserves_all_access_declarations(self):
        final, staging = _artifact_fixture(self.temporary.name)
        pilot.finalize_c7_artifact(staging, final)
        terminal = pilot._read_canonical_jsonl(final / "metrics.jsonl")[-1]
        self.assertTrue(terminal["operation_template_manifest_accessed"])
        self.assertTrue(terminal["operation_template_train_payload_accessed"])
        for name in pilot.C7_ACCESS_FALSE_FIELDS:
            self.assertFalse(terminal[name])


class StaticProductionAndSlurmTests(unittest.TestCase):
    def test_production_cannot_import_development_or_unrestricted_loader(self):
        source = PILOT.read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            for alias in node.names
        }
        self.assertNotIn("load_development", imported)
        self.assertNotIn("load_partition_physical_examples", imported)
        self.assertNotIn("load_development", source)

    def test_no_automatic_repair_or_c8_call_exists(self):
        source = PILOT.read_text(encoding="utf-8")
        tree = ast.parse(source)
        called_names = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name):
                called_names.add(node.func.id.lower())
            elif isinstance(node.func, ast.Attribute):
                called_names.add(node.func.attr.lower())
        self.assertFalse({
            name for name in called_names
            if "hierarchical" in name or name.startswith("c8")
            or name == "repair_decoder"
        })

    def test_slurm_environment_exact_commit_and_clean_tree(self):
        source = SLURM.read_text(encoding="utf-8")
        for literal in (
            "#SBATCH --cpus-per-task=1",
            "CUDA_VISIBLE_DEVICES=\"\"",
            "OMP_NUM_THREADS=1",
            "MKL_NUM_THREADS=1",
            "sys.version_info[:3] == (3, 8, 13)",
            'torch.__version__.split("+")[0] == "1.11.0"',
            'git rev-parse HEAD',
            'git status --porcelain=v1 --untracked-files=all',
        ):
            self.assertIn(literal, source)
        self.assertNotIn("origin/", source)

    def test_slurm_tests_and_audits_precede_corpus_pilot(self):
        source = SLURM.read_text(encoding="utf-8")
        pilot_call = source.index("-m prototype.graph_encoder.pilot")
        for marker in (
            "c7_focused_suite",
            "c7_complete_graph_encoder_suite",
            "model_data_regression_suite=PASS",
            "documentation_validation=PASS",
            "graph_encoder_compileall=PASS",
            "C7_PYTHON38_GRAMMAR_OK",
            "C7_TARGET_LEAKAGE_COMMON_DECODER_SOURCE_AUDIT=PASS",
            "bash -n prototype/graph_encoder/adroit/ge1_pilot_cpu.slurm",
        ):
            self.assertLess(source.index(marker), pilot_call, marker)

    def test_slurm_trap_and_terminal_distinguish_science_from_infrastructure(self):
        source = SLURM.read_text(encoding="utf-8")
        self.assertIn("set -Eeuo pipefail", source)
        self.assertIn("c7_pilot_terminal_failure", source)
        self.assertIn('MANIFEST_ACCESSED=\'"not_confirmed_on_failure"\'', source)
        self.assertIn("c7_pilot_terminal_execution_completed", source)
        self.assertIn('terminal["overall_gate_pass"]', source)
        self.assertIn('terminal["stage6_authorized_by_c7"]', source)

    def test_artifact_validation_precedes_terminal_completion(self):
        source = SLURM.read_text(encoding="utf-8")
        self.assertLess(
            source.index("verify_c7_artifact(root)"),
            source.index('"c7_pilot_terminal_execution_completed"'),
        )

    def test_slurm_bash_grammar(self):
        result = os.system("bash -n {}".format(SLURM))
        self.assertEqual(result, 0)


if __name__ == "__main__":
    unittest.main()
