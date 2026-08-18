"""Pure contracts for the governed Stage 6 scientific producer."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import tempfile
import unittest

from prototype.graph_encoder.errors import GraphEncoderError
from prototype.graph_encoder.grid_magnitude import GRID_MAGNITUDE_PARAMETERIZATION
from prototype.graph_encoder.stage6_structure_only import (
    ARMS, FALLBACK_SEEDS, FULL_SEEDS, summarize_execution,
)
from prototype.graph_encoder.stage6_structure_only_producer import (
    CHECKPOINT_VERSION,
    EPOCHS,
    EXECUTION_RECORD_VERSION,
    INPUT_POLICY_VERSION,
    PRODUCER_ARTIFACT_VERSION,
    PRODUCER_VERSION,
    TIMING_VERSION,
    capacity_gate,
    checkpoint_identity,
    create_producer_artifact,
    create_checkpoint_bundle,
    load_execution_record,
    producer_config,
    optimization_reliability,
    unresolved_input_declaration,
    validate_input_declaration,
    validate_timing_evidence,
    verify_declared_input_files,
    verify_producer_artifact,
    verify_checkpoint_bundle,
    _grammar_valid,
)
from prototype.graph_encoder.stage6_structure_only_producer_audit import audit
from prototype.graph_encoder.tests.test_stage6_structure_only_contract import synthetic_payload


def _timing(feasible=True):
    per_arm = {"flat": 1.0, "typed_graph": 2.0}
    projected = sum(per_arm.values()) * EPOCHS * len(FULL_SEEDS)
    projected_two = sum(per_arm.values()) * EPOCHS * 2
    available = projected * 1.2 + (1.0 if feasible else -1.0)
    return {
        "version": TIMING_VERSION,
        "measured_before_scientific_outcomes": True,
        "observed_results_used": False,
        "corpus_accessed": False,
        "mode": "corpus_free",
        "seconds_per_epoch_by_arm": per_arm,
        "projected_three_seed_seconds": projected,
        "projected_three_seed_seconds_with_contingency": projected * 1.2,
        "projected_two_seed_seconds": projected_two,
        "projected_two_seed_seconds_with_contingency": projected_two * 1.2,
        "contingency_fraction": 0.20,
        "available_wall_seconds": available,
        "three_seed_feasible": feasible,
        "two_seed_feasible": True,
        "fallback_reason": None if feasible else "prospective allocation limit",
    }


def _checkpoint(arm, seed):
    return {
        **checkpoint_identity(
            arm=arm,
            seed=seed,
            source_commit="a" * 40,
            source_digest="b" * 64,
            model_config={"checkpoint_schema": "GE1-C6-CHECKPOINT-v2"},
            partition_identity="operation_template_train-407",
            partition_hashes={"index.json": "c" * 64},
            parameter_count=1000,
            training_arithmetic={"epochs": 200, "batch_size": 8},
            checkpoint_sha256="d" * 64,
        ),
        "stage6_wrapper_sha256": "e" * 64,
    }


def _payload():
    value = synthetic_payload()
    for row in value["training_runs"]:
        row.update({
            "epoch_losses": [1.0] * 200,
            "epoch_gradient_norms": [0.5] * 200,
            "completed_epoch": 200,
        })
    value.update({
        "producer_protocol": PRODUCER_VERSION,
        "execution_record_version": EXECUTION_RECORD_VERSION,
        "source": {"commit": "a" * 40},
        "input_declaration": {"reviewed": True},
    })
    from prototype.graph_encoder.stage6_structure_only import score_family_record
    scored = [score_family_record(row) for row in value["family_records"]]
    scores = {}
    for arm in ARMS:
        for seed in FULL_SEEDS:
            rows = [row["score"]["normalized_structural_prefix"] for row in scored
                    if row["cohort"] == "train" and row["condition"] == "P_true"
                    and row["arm"] == arm and row["seed"] == seed]
            scores[(arm, seed)] = sum(rows) / len(rows)
    reliability = optimization_reliability(value["training_runs"], scores, FULL_SEEDS)
    by_run = {(row["arm"], row["seed"]): row for row in reliability["runs"]}
    by_seed = {row["seed"]: row for row in reliability["train_ceiling_comparisons"]}
    for row in value["training_runs"]:
        row["optimization_reliable"] = (
            by_run[(row["arm"], row["seed"])]["pass_before_train_ceiling"]
            and by_seed[row["seed"]]["pass"]
        )
    value["optimization_reliability"] = reliability
    return value


def _bundle(checkpoints, payload):
    return {
        "schema_version": "GE1-STAGE6-STRUCTURE-ONLY-CHECKPOINT-BUNDLE-v1",
        "bundle_sha256": "f" * 64,
        "checkpoint_count": len(checkpoints),
        "checkpoint_identities": [dict(row, bundle_path=(
            "stage6-{}-seed{}.pt".format(row["arm"], row["seed"])
        )) for row in checkpoints],
        "source_evidence": payload["source_evidence"],
        "input_evidence": payload["input_evidence"],
        "verification_status": "pass",
    }


class ProducerPureContractTests(unittest.TestCase):
    def test_actual_v5_grammar_legal_illegal_incomplete_and_wrong_length(self):
        from prototype.model_data.vocab import NODE_TYPES
        legal = tuple(NODE_TYPES.id(name) for name in (
            "reference_plane", "sketch", "profile", "extrude"
        ))
        illegal = tuple(NODE_TYPES.id(name) for name in (
            "reference_plane", "profile", "sketch", "extrude"
        ))
        self.assertTrue(_grammar_valid(legal, 4))
        self.assertFalse(_grammar_valid(legal[:-1], 3))
        self.assertFalse(_grammar_valid(illegal, 4))
        self.assertFalse(_grammar_valid(legal, 5))
    def test_versioned_frozen_configuration(self):
        config = producer_config()
        self.assertEqual(config["producer_version"], PRODUCER_VERSION)
        self.assertEqual(config["execution_record_version"], EXECUTION_RECORD_VERSION)
        self.assertEqual(config["epochs"], 200)
        self.assertEqual(config["fixed_checkpoint_epoch"], 200)
        self.assertFalse(config["development_checkpoint_selection"])
        self.assertFalse(config["checkpoint_reuse"])

    def test_timing_recomputes_projection_and_fallback_is_prospective(self):
        self.assertEqual(validate_timing_evidence(_timing(), allow_fallback=False), FULL_SEEDS)
        broken = _timing()
        broken["projected_three_seed_seconds"] += 1
        broken["projected_three_seed_seconds_with_contingency"] = broken["projected_three_seed_seconds"] * 1.2
        with self.assertRaises(GraphEncoderError):
            validate_timing_evidence(broken, allow_fallback=False)
        self.assertEqual(
            validate_timing_evidence(_timing(False), allow_fallback=False),
            FALLBACK_SEEDS,
        )

    def test_timing_exact_boundaries_neither_feasible_and_outcome_informed(self):
        full = _timing()
        full["available_wall_seconds"] = full[
            "projected_three_seed_seconds_with_contingency"
        ]
        self.assertEqual(validate_timing_evidence(full, allow_fallback=False), FULL_SEEDS)
        fallback = _timing(False)
        fallback["available_wall_seconds"] = fallback[
            "projected_two_seed_seconds_with_contingency"
        ]
        self.assertEqual(validate_timing_evidence(
            fallback, allow_fallback=False
        ), FALLBACK_SEEDS)
        neither = copy.deepcopy(fallback)
        neither["available_wall_seconds"] -= 1
        with self.assertRaises(GraphEncoderError):
            validate_timing_evidence(neither, allow_fallback=True)
        informed = _timing()
        informed["observed_results_used"] = True
        with self.assertRaises(GraphEncoderError):
            validate_timing_evidence(informed, allow_fallback=True)

    def test_capacity_tolerance_is_inclusive(self):
        self.assertTrue(capacity_gate(1000, 1050)["pass"])
        self.assertFalse(capacity_gate(1000, 1051)["pass"])

    def test_unresolved_and_protected_inputs_cannot_execute(self):
        with self.assertRaises(GraphEncoderError):
            validate_input_declaration(unresolved_input_declaration())
        record = unresolved_input_declaration()
        record["train"]["root"] = "/authorized/RR"
        with self.assertRaises(GraphEncoderError):
            validate_input_declaration(record)

    def test_exact_narrow_input_allowlists_reject_extras(self):
        with tempfile.TemporaryDirectory() as temporary:
            roots = {}
            for cohort in ("train", "development"):
                root = Path(temporary) / cohort
                root.mkdir()
                (root / "index.json").write_text("{}\n", encoding="utf-8")
                digest = hashlib.sha256(b"{}\n").hexdigest()
                roots[cohort] = {
                    "partition": "operation_template_" + cohort,
                    "family_count": 407 if cohort == "train" else 45,
                    "root": str(root), "root_is_read_only": True,
                    "authorized_files": ["index.json"],
                    "expected_sha256": {"index.json": digest},
                    "observed_sha256": {"index.json": digest},
                }
            declaration = {
                "version": INPUT_POLICY_VERSION, **roots,
                "unexpected_paths": [], "broad_parent_mounted": False,
                "external_inputs": [],
            }
            self.assertTrue(verify_declared_input_files(declaration))
            (Path(roots["train"]["root"]) / "extra.json").write_text("{}\n")
            with self.assertRaises(GraphEncoderError):
                verify_declared_input_files(declaration)

    def test_optimization_reliability_uses_only_finite_plateau_and_ceiling(self):
        runs = [{
            "arm": arm, "seed": seed, "epoch_losses": [1.0] * 200,
            "epoch_gradient_norms": [0.5] * 200, "completed_epoch": 200,
            "checkpoint_epoch": 200,
        } for arm in ARMS for seed in FULL_SEEDS]
        scores = {(arm, seed): 0.8 for arm in ARMS for seed in FULL_SEEDS}
        result = optimization_reliability(runs, scores, FULL_SEEDS)
        self.assertTrue(result["pass"])
        self.assertFalse(result["geometry_used"])
        broken = copy.deepcopy(runs)
        broken[0]["epoch_losses"][0] = float("nan")
        self.assertFalse(optimization_reliability(broken, scores, FULL_SEEDS)["pass"])

    def test_checkpoint_identity_is_strict_and_separate(self):
        row = _checkpoint("flat", 2026)
        self.assertEqual(row["version"], CHECKPOINT_VERSION)
        self.assertEqual(row["epoch"], 200)
        self.assertEqual(row["operation_magnitude_parameterization"], GRID_MAGNITUDE_PARAMETERIZATION)
        broken = copy.deepcopy(row)
        broken["source_commit"] = "z" * 40
        with self.assertRaises(GraphEncoderError):
            payload = _payload()
            checkpoints = [broken] + [
                _checkpoint(arm, seed) for arm in ARMS for seed in FULL_SEEDS
                if (arm, seed) != ("flat", 2026)
            ]
            create_producer_artifact(
                payload, checkpoints, Path(tempfile.mkdtemp()) / "artifact",
                job_id="123",
                checkpoint_bundle_reference=_bundle(checkpoints, payload),
            )

    def test_checkpoint_bundle_is_atomic_complete_and_tamper_evident(self):
        payload = _payload()
        with tempfile.TemporaryDirectory() as temporary:
            rows = []
            for arm in ARMS:
                for seed in FULL_SEEDS:
                    path = Path(temporary) / "{}-{}.pt".format(arm, seed)
                    path.write_bytes((arm + str(seed)).encode("utf-8"))
                    identity = _checkpoint(arm, seed)
                    identity["stage6_wrapper_sha256"] = hashlib.sha256(
                        path.read_bytes()
                    ).hexdigest()
                    rows.append({"path": str(path), "identity": identity})
            root = Path(temporary) / "bundle"
            result = create_checkpoint_bundle(
                rows, root, job_id="123", source_evidence=payload["source_evidence"],
                input_evidence=payload["input_evidence"], retained_seeds=FULL_SEEDS,
            )
            self.assertEqual(result["checkpoint_count"], 6)
            self.assertEqual(verify_checkpoint_bundle(
                root, retained_seeds=FULL_SEEDS
            )["verification_status"], "pass")
            (root / "stage6-flat-seed2026.pt").write_bytes(b"tampered")
            with self.assertRaises(GraphEncoderError):
                verify_checkpoint_bundle(root, retained_seeds=FULL_SEEDS)

    def test_atomic_artifact_is_finalizer_compatible_and_outcome_independent(self):
        checkpoints = [_checkpoint(arm, seed) for arm in ARMS for seed in FULL_SEEDS]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "producer"
            payload = _payload()
            result = create_producer_artifact(
                payload, checkpoints, root,
                job_id="123",
                checkpoint_bundle_reference=_bundle(checkpoints, payload),
            )
            self.assertEqual(result["producer_artifact_version"], PRODUCER_ARTIFACT_VERSION)
            payload = load_execution_record(root)
            unused, summary = summarize_execution(payload)
            self.assertEqual(summary["interpretation_category"], "inconclusive")
            self.assertFalse(result["outcome_controls_validity"])

    def test_tamper_duplicate_and_incomplete_artifacts_fail(self):
        checkpoints = [_checkpoint(arm, seed) for arm in ARMS for seed in FULL_SEEDS]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "producer"
            payload = _payload()
            create_producer_artifact(
                payload, checkpoints, root,
                job_id="123",
                checkpoint_bundle_reference=_bundle(checkpoints, payload),
            )
            with (root / "family_records.jsonl").open("a", encoding="utf-8") as stream:
                stream.write("{}\n")
            with self.assertRaises(GraphEncoderError):
                verify_producer_artifact(root)
            (root / "family_records.jsonl").unlink()
            with self.assertRaises(GraphEncoderError):
                verify_producer_artifact(root)

    def test_runner_has_governed_binds_single_invocation_and_telemetry(self):
        repository = Path(__file__).resolve().parents[3]
        runner = repository / "prototype/graph_encoder/adroit/ge1_stage6_structure_only_producer.slurm"
        result = audit(repository, runner)
        self.assertEqual(result["bind_count"], 4)
        self.assertEqual(result["producer_invocation_count"], 1)
        self.assertEqual(result["protected_bind_count"], 0)


if __name__ == "__main__":
    unittest.main()
