"""Pure contracts for the prospective Stage 6 structure-only comparison."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from prototype.graph_encoder.errors import GraphEncoderError
from prototype.graph_encoder.stage6_structure_only import (
    ARTIFACT_VERSION,
    DEVELOPMENT_FAMILY_COUNT,
    FALLBACK_SEEDS,
    FULL_SEEDS,
    INPUT_VERSION,
    PRIMARY_THRESHOLD,
    PROTOCOL_VERSION,
    STRUCTURAL_FIELDS,
    TRAIN_FAMILY_COUNT,
    create_artifact,
    interpretation_category,
    paired_seed_aggregation,
    protocol_config,
    score_family_record,
    score_structural_prefix,
    structural_prefix_evidence,
    structure_memory_gate,
    summarize_execution,
    verify_artifact,
)
from prototype.graph_encoder.stage6_structure_only_audit import (
    CPU_DISCOVERY_CUDA_SKIP_IDS,
    audit as audit_finalizer,
    validate_cpu_finalizer_complete_discovery,
)


def _prefix(operation_count, selected):
    rows = []
    for k in range(1, operation_count + 1):
        values = {field: k <= selected for field in STRUCTURAL_FIELDS}
        rows.append({"k": k, **values})
    return rows


def _family(family_id, template, cohort, arm, seed, condition, selected):
    operation_count = 1 if template in ("E", "R") else 2
    from prototype.model_data.vocab import NODE_TYPES
    sequences = {
        "E": (("reference_plane", "sketch", "profile", "extrude"),),
        "R": (("reference_plane", "sketch", "profile", "axis", "revolve"),),
        "EE": (
            ("reference_plane", "sketch", "profile", "extrude"),
            ("reference_plane", "sketch", "profile", "extrude", "sketch", "profile", "extrude"),
        ),
        "RE": (
            ("reference_plane", "sketch", "profile", "axis", "revolve"),
            ("reference_plane", "sketch", "profile", "axis", "revolve", "sketch", "profile", "extrude"),
        ),
    }
    target_prefixes = []
    predicted_prefixes = []
    for k in range(1, operation_count + 1):
        node_ids = [NODE_TYPES.id(name) for name in sequences[template][k - 1]]
        operation_ids = [value for value in node_ids if NODE_TYPES.tokens[value] in ("extrude", "revolve")]
        target = {
            "k": k,
            "grammar_valid_complete": True,
            "canonicalization_succeeded": True,
            "prefix_construction_succeeded": True,
            "requested_node_count": len(node_ids),
            "node_type_ids": node_ids,
            "operation_type_ids": operation_ids,
            "canonical_edges": [[0, 1, 1]],
            "depends_on_edges": [] if k == 1 else [[1, 0, 2]],
            "ownership_attachment_edges": [[0, 1, 1]],
        }
        predicted = copy.deepcopy(target)
        if k > selected:
            predicted["grammar_valid_complete"] = False
            predicted["node_type_ids"][-1] += 10
        target_prefixes.append(target)
        predicted_prefixes.append(predicted)
    batch_identity = cohort + "-batch"
    donor_memory_source = "different-family"
    if "-" in family_id:
        prefix, index = family_id.rsplit("-", 1)
        count = TRAIN_FAMILY_COUNT if cohort == "train" else DEVELOPMENT_FAMILY_COUNT
        numeric_index = int(index)
        batch_start = (numeric_index // 8) * 8
        batch_length = min(8, count - batch_start)
        batch_identity = "{}-batch-{:03d}".format(cohort, numeric_index // 8)
        donor_index = batch_start + ((numeric_index - batch_start + 1) % batch_length)
        donor_memory_source = "{}-{:03d}".format(prefix, donor_index)
    if condition == "P_shuffle":
        memory_source = donor_memory_source
    elif condition == "P_true":
        memory_source = family_id
    else:
        memory_source = "batch_mean:" + batch_identity
    return {
        "family_id": family_id,
        "template": template,
        "cohort": cohort,
        "arm": arm,
        "seed": seed,
        "condition": condition,
        "representation_variant_id": "representation:" + family_id,
        "batch_identity": batch_identity,
        "memory_source_family_id": memory_source,
        "target_operation_count": operation_count,
        "predicted_operation_group_count": operation_count,
        "autonomous": True,
        "target_joined_after_generation": True,
        "raw_prediction_preserved": True,
        "constrained_prediction_preserved": True,
        "predicted_structural_prefixes": predicted_prefixes,
        "target_structural_prefixes": target_prefixes,
        "secondary": {
            "exact_complete_node_sequence": selected == operation_count,
            "exact_operation_type_sequence": selected == operation_count,
            "exact_graph": selected == operation_count,
            "depends_on_true_positive": max(0, operation_count - 1),
            "depends_on_predicted": max(0, operation_count - 1),
            "depends_on_target": max(0, operation_count - 1),
            "depends_on_exact": True,
            "reference_attachment_exact": selected == operation_count,
            "strict_conversion": False,
            "analytic_validity": False,
            "geometry_metrics": {"report_only": True},
            "operation_magnitude_metrics": {"report_only": True},
        },
    }


def synthetic_payload(seeds=FULL_SEEDS):
    records = []
    for cohort, count in (("train", TRAIN_FAMILY_COUNT),
                          ("development", DEVELOPMENT_FAMILY_COUNT)):
        for seed in seeds:
            for arm in ("flat", "typed_graph"):
                for condition in ("P_true", "P_shuffle", "P_mean"):
                    for index in range(count):
                        template = ("E", "R", "EE", "RE")[index % 4]
                        operations = 1 if template in ("E", "R") else 2
                        if condition != "P_true":
                            selected = 0
                        elif arm == "typed_graph":
                            selected = operations
                        else:
                            selected = operations if index % 2 == 0 else 0
                        records.append(_family(
                            "{}-{:03d}".format(cohort, index), template, cohort,
                            arm, seed, condition, selected,
                        ))
    runs = []
    for arm in ("flat", "typed_graph"):
        for seed in seeds:
            runs.append({
                "arm": arm,
                "seed": seed,
                "fresh_initialization": True,
                "epochs": 200,
                "batch_size": 8,
                "optimizer": "AdamW",
                "learning_rate": 0.001,
                "weight_decay": 0.0,
                "gradient_clip_norm": 1.0,
                "checkpoint_epoch": 200,
                "early_stopping": False,
                "development_used_for_selection": False,
                "warm_start": False,
                "training_family_count": 407,
                "trainable_parameter_count": 1000 if arm == "flat" else 1010,
                "capacity_parity_pass": True,
                "optimization_reliable": True,
                "final_loss": 0.1,
                "plateau_relative_improvement": 0.0,
                "execution_device": "cpu",
            })
    fallback = tuple(seeds) == FALLBACK_SEEDS
    source_evidence = {
        "source_commit": "a" * 40, "source_tree_sha256": "b" * 64,
        "source_clean": True, "detached_head": True,
        "verification_status": "pass",
    }
    input_evidence = {
        "input_policy_version": "test", "narrow_loader_version": "test",
        "train": {
            "index_identity_sha256": "c" * 64,
            "expected_allowlist_sha256": "d" * 64,
            "observed_allowlist_sha256": "d" * 64,
            "expected_payload_digests_sha256": "d" * 64,
            "observed_payload_digests_sha256": "d" * 64,
            "expected_payload_sha256": {"a": "e" * 64},
            "observed_payload_sha256": {"a": "e" * 64},
            "verification_status": "pass",
        },
        "development": {
            "index_identity_sha256": "f" * 64,
            "expected_allowlist_sha256": "1" * 64,
            "observed_allowlist_sha256": "1" * 64,
            "expected_payload_digests_sha256": "2" * 64,
            "observed_payload_digests_sha256": "2" * 64,
            "expected_payload_sha256": {"b": "2" * 64},
            "observed_payload_sha256": {"b": "2" * 64},
            "verification_status": "pass",
        },
        "verification_status": "pass",
    }
    bundle = {
        "bundle_sha256": "3" * 64, "source_evidence": source_evidence,
        "input_evidence": input_evidence, "verification_status": "pass",
    }
    from prototype.graph_encoder.tests.test_stage6_timing_contract import record
    timing_evidence = (
        record(cpu=1.0, cuda=2.0)
        if not fallback
        else record(cpu=1.0, cuda=2.0, cpu_wall=1000.0, cuda_wall=1.0)
    )
    return {
        "schema_version": INPUT_VERSION,
        "retained_seeds": list(seeds),
        "timing_fallback": {
            "invoked": fallback,
            "prospective": fallback,
            "observed_results_used": False,
            "reason": "prospective timing projection" if fallback else None,
            "evidence": timing_evidence,
        },
        "execution_evidence": {
            "selected_execution_device": "cpu",
            "runtime_identity": timing_evidence["candidates"]["cpu"][
                "runtime_identity"
            ],
            "timing_hardware_identity": timing_evidence[
                "selected_timing_hardware_identity"
            ],
            "timing_version": timing_evidence["version"],
            "device_selected_only_by_timing": True,
            "cuda_peak_memory_bytes": None,
            "verification_status": "pass",
        },
        "partition": {
            "training_name": "operation_template_train",
            "training_family_count": 407,
            "development_name": "operation_template_development",
            "development_family_count": 45,
            "rr_accessed": False,
            "er_accessed": False,
        },
        "training_runs": runs,
        "family_records": records,
        "provenance_valid": True,
        "artifact_inputs_valid": True,
        "source_evidence": source_evidence,
        "input_evidence": input_evidence,
        "checkpoint_bundle_reference": bundle,
        "producer_artifact_evidence": {
            "schema_version": "test", "producer_artifact_sha256": "4" * 64,
            "checkpoint_bundle_sha256": "3" * 64,
            "verification_status": "pass",
        },
    }


class ScoringTests(unittest.TestCase):
    def test_identity_is_separate(self):
        self.assertEqual(PROTOCOL_VERSION, "GE1-STAGE6-STRUCTURE-ONLY-COMPARISON-v1")
        self.assertEqual(ARTIFACT_VERSION, "GE1-STAGE6-STRUCTURE-ONLY-ARTIFACT-v1")

    def test_complete_two_operation_prefix(self):
        result = score_structural_prefix(2, _prefix(2, 2))
        self.assertEqual(result["selected_k"], 2)
        self.assertEqual(result["normalized_structural_prefix"], 1.0)

    def test_first_failure_stops_later_prefix(self):
        rows = _prefix(2, 0)
        rows[1] = {"k": 2, **{field: True for field in STRUCTURAL_FIELDS}}
        result = score_structural_prefix(2, rows)
        self.assertEqual(result["selected_k"], 0)
        self.assertEqual(result["first_failure_stage"], "grammar_complete")

    def test_geometry_and_conversion_are_absent_from_primary(self):
        result = score_structural_prefix(1, _prefix(1, 1))
        self.assertFalse(result["conversion_used"])
        self.assertFalse(result["analytic_validity_used"])
        self.assertFalse(result["geometry_used"])

    def test_invalid_denominator_fails(self):
        with self.assertRaises(GraphEncoderError):
            score_structural_prefix(3, ())

    def test_evidence_is_derived_from_structural_records(self):
        row = _family("x", "EE", "development", "flat", 2026, "P_true", 1)
        evidence = structural_prefix_evidence(
            row["predicted_structural_prefixes"], row["target_structural_prefixes"]
        )
        self.assertTrue(all(evidence[0][field] for field in STRUCTURAL_FIELDS))
        self.assertFalse(evidence[1]["grammar_complete"])
        self.assertFalse(evidence[1]["canonical_graph_exact"])

    def test_family_requires_autonomous_boundary(self):
        row = _family("x", "E", "development", "flat", 2026, "P_true", 1)
        row["autonomous"] = False
        with self.assertRaises(GraphEncoderError):
            score_family_record(row)

    def test_target_join_must_follow_generation(self):
        row = _family("x", "E", "development", "flat", 2026, "P_true", 1)
        row["target_joined_after_generation"] = False
        with self.assertRaises(GraphEncoderError):
            score_family_record(row)

    def test_over_generation_cannot_receive_full_credit(self):
        exact = score_structural_prefix(2, _prefix(2, 2), 2)
        over = score_structural_prefix(2, _prefix(2, 2), 3)
        under = score_structural_prefix(2, _prefix(2, 1), 1)
        self.assertEqual(exact["normalized_structural_prefix"], 1.0)
        self.assertEqual(over["normalized_structural_prefix"], 0.5)
        self.assertTrue(over["over_generation"])
        self.assertEqual(under["normalized_structural_prefix"], 0.5)
        self.assertTrue(under["under_generation"])

    def test_relation_specific_first_failure_order(self):
        cases = (
            ("depends_on_edges", [[0, 1, 9]], "depends_on_exact"),
            ("ownership_attachment_edges", [[0, 1, 9]], "ownership_attachments_exact"),
            ("canonical_edges", [[0, 1, 9]], "canonical_graph_exact"),
            ("operation_type_ids", None, "chronological_order_exact"),
        )
        for field, value, expected in cases:
            row = _family("x", "RE", "development", "flat", 2026, "P_true", 2)
            if value is None:
                value = list(reversed(
                    row["target_structural_prefixes"][1]["operation_type_ids"]
                ))
            row["predicted_structural_prefixes"][1][field] = value
            result = score_family_record(row)
            self.assertEqual(result["score"]["first_failure_stage"], expected)


class InterpretationTests(unittest.TestCase):
    def _primary(self, effect, subgroups=(), fallback=True):
        return {
            "retained_seed_mean_effect": effect,
            "fallback_seed_nonnegative_pass": fallback,
            "seed_effects": [
                {"graph_minus_flat_mean": value} for value in subgroups
            ],
            "template_effects": {},
        }

    def test_exact_directional_boundaries(self):
        self.assertEqual(interpretation_category(self._primary(0.10), True), "graph_supported")
        self.assertEqual(interpretation_category(self._primary(-0.10), True), "flat_supported")
        self.assertEqual(interpretation_category(self._primary(0.0), True), "null")
        self.assertEqual(interpretation_category(self._primary(0.0, (0.10, -0.10)), True), "mixed")
        self.assertEqual(interpretation_category(self._primary(0.10, fallback=False), True), "inconclusive")
        self.assertEqual(interpretation_category(self._primary(0.10), False), "inconclusive")


class AggregationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = synthetic_payload()
        cls.scored = [score_family_record(row) for row in cls.payload["family_records"]]

    def test_pairs_within_seed_before_seed_mean(self):
        result = paired_seed_aggregation(self.scored, FULL_SEEDS)
        self.assertEqual(len(result["seed_effects"]), 3)
        self.assertTrue(all(row["paired_family_count"] == 45 for row in result["seed_effects"]))
        self.assertFalse(result["seeds_pooled_as_independent_families"])
        self.assertGreaterEqual(result["retained_seed_mean_effect"], PRIMARY_THRESHOLD)

    def test_unpaired_families_fail(self):
        broken = list(self.scored)
        broken.pop(next(index for index, row in enumerate(broken)
                        if row["cohort"] == "development"
                        and row["arm"] == "flat" and row["seed"] == 2026
                        and row["condition"] == "P_true"))
        with self.assertRaises(GraphEncoderError):
            paired_seed_aggregation(broken, FULL_SEEDS)

    def test_structure_memory_gate(self):
        result = structure_memory_gate(self.scored, FULL_SEEDS)
        self.assertTrue(result["overall_pass"])
        self.assertEqual(result["cohorts"]["development"]["flat"]["P_shuffle"], 0.0)

    def test_fallback_is_prospective_and_nonnegative(self):
        payload = synthetic_payload(FALLBACK_SEEDS)
        unused, summary = summarize_execution(payload)
        self.assertTrue(summary["primary"]["fallback_seed_nonnegative_pass"])

    def test_nonprospective_fallback_fails(self):
        payload = synthetic_payload(FALLBACK_SEEDS)
        payload["timing_fallback"]["observed_results_used"] = True
        with self.assertRaises(GraphEncoderError):
            summarize_execution(payload)

    def test_cross_batch_donor_and_moved_batch_record_fail(self):
        payload = copy.deepcopy(self.payload)
        row = next(row for row in payload["family_records"]
                   if row["cohort"] == "train" and row["arm"] == "flat"
                   and row["seed"] == 2026 and row["condition"] == "P_shuffle"
                   and row["family_id"] == "train-000")
        row["memory_source_family_id"] = "train-008"
        with self.assertRaises(GraphEncoderError):
            summarize_execution(payload)
        payload = copy.deepcopy(self.payload)
        row = next(row for row in payload["family_records"]
                   if row["cohort"] == "development" and row["arm"] == "flat"
                   and row["seed"] == 2026 and row["condition"] == "P_mean"
                   and row["family_id"] == "development-000")
        row["batch_identity"] = "development-batch-999"
        row["memory_source_family_id"] = "batch_mean:development-batch-999"
        with self.assertRaises(GraphEncoderError):
            summarize_execution(payload)

    def test_valid_result_is_graph_supported(self):
        unused, summary = summarize_execution(copy.deepcopy(self.payload))
        self.assertEqual(summary["interpretation_category"], "graph_supported")

    def test_memory_failure_is_inconclusive(self):
        payload = copy.deepcopy(self.payload)
        for row in payload["family_records"]:
            if row["condition"] == "P_shuffle":
                row["predicted_structural_prefixes"] = copy.deepcopy(
                    row["target_structural_prefixes"]
                )
        unused, summary = summarize_execution(payload)
        self.assertEqual(summary["interpretation_category"], "inconclusive")


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "artifact"
        self.payload = synthetic_payload(FALLBACK_SEEDS)

    def tearDown(self):
        self.temp.cleanup()

    def test_atomic_five_file_artifact_verifies(self):
        result = create_artifact(self.payload, self.root, "a" * 40, "123")
        self.assertEqual(result["protocol_version"], PROTOCOL_VERSION)
        self.assertEqual(len(tuple(self.root.iterdir())), 5)
        self.assertEqual(
            verify_artifact(self.root, expected_commit="a" * 40, expected_job_id="123"),
            result,
        )

    def test_checksum_tampering_fails(self):
        create_artifact(self.payload, self.root, "a" * 40, "123")
        with (self.root / "summary.json").open("a", encoding="utf-8") as stream:
            stream.write(" ")
        with self.assertRaises(GraphEncoderError):
            verify_artifact(self.root, expected_commit="a" * 40, expected_job_id="123")

    def test_outcome_is_not_artifact_validity_gate(self):
        payload = copy.deepcopy(self.payload)
        payload["training_runs"][0]["optimization_reliable"] = False
        result = create_artifact(payload, self.root, "a" * 40, "123")
        self.assertEqual(result["interpretation_category"], "inconclusive")

    def test_protocol_freezes_epoch_200(self):
        config = protocol_config()
        self.assertEqual(config["epochs"], 200)
        self.assertEqual(config["checkpoint_epoch"], 200)
        self.assertFalse(config["primary_uses_geometry"])


class CpuFinalizerDiscoveryTests(unittest.TestCase):
    def test_exact_cuda_allowlist_is_the_only_complete_discovery_skip_set(self):
        from prototype.graph_encoder.tests.test_stage6_cuda_runtime import (
            Stage6CudaRuntimeTests,
        )
        prefix = "{}.{}.".format(
            Stage6CudaRuntimeTests.__module__, Stage6CudaRuntimeTests.__name__
        )
        discovered = tuple(
            prefix + name for name in sorted(dir(Stage6CudaRuntimeTests))
            if name.startswith("test_")
        )
        self.assertEqual(CPU_DISCOVERY_CUDA_SKIP_IDS, discovered)

        def result_for(skip_ids, tests_run=700, failures=(), errors=()):
            skipped = [
                (SimpleNamespace(id=lambda value=value: value), "CUDA unavailable")
                for value in skip_ids
            ]
            return SimpleNamespace(
                testsRun=tests_run,
                failures=list(failures),
                errors=list(errors),
                skipped=skipped,
                wasSuccessful=lambda: not failures and not errors,
            )

        telemetry = validate_cpu_finalizer_complete_discovery(
            700, result_for(CPU_DISCOVERY_CUDA_SKIP_IDS)
        )
        self.assertEqual(telemetry["declared"], 700)
        self.assertEqual(telemetry["run"], 700)
        self.assertEqual(telemetry["failures"], 0)
        self.assertEqual(telemetry["errors"], 0)
        self.assertEqual(telemetry["skipped"], 6)
        self.assertEqual(telemetry["skip_ids"], list(CPU_DISCOVERY_CUDA_SKIP_IDS))

        arbitrary = "arbitrary.module.ArbitraryTests.test_unrelated_skip"
        for skip_ids in (
            CPU_DISCOVERY_CUDA_SKIP_IDS[:-1],
            CPU_DISCOVERY_CUDA_SKIP_IDS + (CPU_DISCOVERY_CUDA_SKIP_IDS[-1],),
            CPU_DISCOVERY_CUDA_SKIP_IDS[:-1] + (arbitrary,),
            CPU_DISCOVERY_CUDA_SKIP_IDS + (arbitrary,),
        ):
            with self.subTest(skip_ids=skip_ids), self.assertRaises(AssertionError):
                validate_cpu_finalizer_complete_discovery(700, result_for(skip_ids))
        for result in (
            result_for(CPU_DISCOVERY_CUDA_SKIP_IDS, tests_run=699),
            result_for(CPU_DISCOVERY_CUDA_SKIP_IDS, failures=((object(), "failure"),)),
            result_for(CPU_DISCOVERY_CUDA_SKIP_IDS, errors=((object(), "error"),)),
        ):
            with self.assertRaises(AssertionError):
                validate_cpu_finalizer_complete_discovery(700, result)

        repository = Path(__file__).resolve().parents[3]
        runner = repository / "prototype/graph_encoder/adroit/ge1_stage6_structure_only_cpu.slurm"
        self.assertEqual(audit_finalizer(repository, runner)["entry_point_count"], 1)


if __name__ == "__main__":
    unittest.main()
