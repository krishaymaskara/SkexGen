"""PyTorch-independent contract tests for ADR-0012."""

from __future__ import annotations

import ast
import copy
import json
from pathlib import Path
import tempfile
import unittest

from prototype.graph_encoder.errors import GraphEncoderError
from prototype.graph_encoder import closed_form_readout as readout
from prototype.graph_encoder.pilot import _atomic_write_json, _atomic_write_jsonl


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "prototype/graph_encoder/closed_form_readout.py"
RUNNER = ROOT / "prototype/graph_encoder/adroit/ge1_closed_form_readout_cpu.slurm"


def _joined_rows(arm="flat"):
    features, labels = readout.synthetic_fixture()
    return readout.join_detached_rows(features, labels, arm)


def _negative_results():
    records = []
    for arm in readout.PROBE_ARMS:
        for cohort in ("full", "single_extrusion_E_RE"):
            for feature in ("C", "P", "D-additive", "D-gated", "context-only"):
                records.append({
                    "arm": arm, "operation_type": "extrude", "cohort": cohort,
                    "feature_identity": feature,
                })
        for feature in ("C", "P"):
            records.append({
                "arm": arm, "operation_type": "revolve", "cohort": "full",
                "feature_identity": feature,
            })
    primary = []
    for identity, arm, operation_type, kind, left, right in readout._primary_definitions():
        primary.append({
            "hypothesis_id": identity,
            "arm": arm,
            "operation_type": operation_type,
            "kind": kind,
            "left": left,
            "right": right,
        })
    structured = []
    for arm in readout.PROBE_ARMS:
        structured.extend(readout.revolve_degeneracy_records(arm))
        structured.append(readout.revolve_sensitivity_record(arm))
    return {
        "schema_version": readout.READOUT_RESULT_VERSION,
        "protocol_version": readout.READOUT_PROTOCOL_VERSION,
        "primary_hypothesis_count": 16,
        "primary_hypotheses": primary,
        "executed_pipeline_count": 24,
        "executed_pipelines": records,
        "structured_nonexecuted_count": 8,
        "structured_nonexecuted": structured,
        "limitations": list(readout.LIMITATIONS),
        "authority": {
            "formal_adr0011_representation_probe_completed": False,
            "representation_screen_completed_or_reinterpreted": False,
            "model_or_decoder_repair_authorized": False,
            "stage6_authorized": False,
            "c8_authorized": False,
            "protected_access_authorized": False,
        },
        "permutation_count": 499,
    }


class ClosedFormReadoutContractTests(unittest.TestCase):
    def test_protocol_artifact_and_results_identities_are_exact(self):
        self.assertEqual(readout.READOUT_PROTOCOL_VERSION, "GE1-C7-CLOSED-FORM-READOUT-v1")
        self.assertEqual(
            readout.READOUT_ARTIFACT_VERSION,
            "GE1-C7-CLOSED-FORM-READOUT-ARTIFACT-v1",
        )
        self.assertEqual(
            readout.READOUT_RESULT_VERSION,
            "GE1-C7-CLOSED-FORM-READOUT-RESULTS-v1",
        )

    def test_exact_three_frozen_input_hashes(self):
        self.assertEqual(readout.READOUT_INPUT_HASHES, {
            "detached_features.pt": "87dab4841f64f0c1b27db6676852da7cd980ba1bde61e81809fde3af745cc01c",
            "feature_manifest.json": "29220bda0677636627069ce6565bf367065f1ceaa3c2d86d3174058838598450",
            "labels.json": "70f82727ebe35613bcbae1267924d9a327d9744a757b6d076c8028e261cfe71f",
        })

    def test_contract_freezes_objective_grid_ladder_and_sixteen_hypotheses(self):
        contract = readout.readout_contract()
        self.assertEqual(contract["primary_hypothesis_count"], 16)
        self.assertEqual(contract["estimator"]["lambda_grid"], [
            1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0,
        ])
        self.assertEqual(contract["permutations"]["ladder"], [999, 499])
        self.assertIn("(1/(2n))", contract["estimator"]["objective"])
        self.assertEqual(len(readout._primary_definitions()), 16)

    def test_cohort_structure_and_sensitivity_are_exact(self):
        rows = _joined_rows()
        full_extrude = readout.select_cohort(rows, "extrude", "full")
        full_revolve = readout.select_cohort(rows, "revolve", "full")
        sensitivity = readout.select_cohort(rows, "extrude", "single_extrusion_E_RE")
        self.assertEqual((len(full_extrude), len({r["family_id"] for r in full_extrude})), (32, 24))
        self.assertEqual((len(full_revolve), len({r["family_id"] for r in full_revolve})), (16, 16))
        self.assertEqual((len(sensitivity), len({r["family_id"] for r in sensitivity})), (16, 16))
        self.assertEqual({r["operation_template"] for r in sensitivity}, {"E", "RE"})

    def test_revolve_sensitivity_is_explicitly_not_applicable(self):
        with self.assertRaises(GraphEncoderError) as context:
            readout.select_cohort(_joined_rows(), "revolve", "single_extrusion_E_RE")
        self.assertEqual(context.exception.code, "redundant_revolve_sensitivity")
        record = readout.revolve_sensitivity_record("flat")
        self.assertEqual(record["status"], "not_applicable_full_cohort_already_one_to_one")
        self.assertFalse(record["fit_executed"])

    def test_revolve_feature_degeneracy_is_structured(self):
        records = readout.revolve_degeneracy_records("typed_graph")
        self.assertEqual({row["feature_identity"] for row in records}, {
            "D-additive", "D-gated", "context-only",
        })
        self.assertTrue(all(row["status"] == "degenerate_by_construction" for row in records))
        self.assertTrue(all(not row["counted_as_primary_test"] for row in records))

    def test_family_block_permutation_preserves_template_support_and_EE_order(self):
        rows = readout.select_cohort(_joined_rows(), "extrude", "full")
        observed = readout.permute_family_blocks(rows, "extrude", 11)
        for template in ("E", "EE", "RE"):
            selected = [row for row in rows if row["operation_template"] == template]
            before = sorted(
                tuple(row["class_index"] for row in sorted(group, key=lambda item: item["operation_index"]))
                for group in (
                    [row for row in selected if row["family_id"] == family]
                    for family in sorted({row["family_id"] for row in selected})
                )
            )
            after = sorted(
                tuple(observed[row["key"]] for row in sorted(group, key=lambda item: item["operation_index"]))
                for group in (
                    [row for row in selected if row["family_id"] == family]
                    for family in sorted({row["family_id"] for row in selected})
                )
            )
            self.assertEqual(before, after)

    def test_permutation_is_shared_across_arms_and_features(self):
        flat = readout.select_cohort(_joined_rows("flat"), "extrude", "full")
        graph = readout.select_cohort(_joined_rows("typed_graph"), "extrude", "full")
        flat_values = readout.permute_family_blocks(flat, "extrude", 3)
        graph_values = readout.permute_family_blocks(graph, "extrude", 3)
        by_label_flat = {row["label_key"]: flat_values[row["key"]] for row in flat}
        by_label_graph = {row["label_key"]: graph_values[row["key"]] for row in graph}
        self.assertEqual(by_label_flat, by_label_graph)

    def test_operation_types_have_distinct_seed_strata(self):
        extrude_seed = readout._seed_from_parts(
            readout.READOUT_PROTOCOL_VERSION, 2026, "extrude", 0
        )
        revolve_seed = readout._seed_from_parts(
            readout.READOUT_PROTOCOL_VERSION, 2026, "revolve", 0
        )
        self.assertNotEqual(extrude_seed, revolve_seed)

    def test_timing_ladder_has_only_999_499_or_abort(self):
        self.assertEqual(readout.select_permutation_count(2699, 1200)["selected_permutation_count"], 999)
        self.assertEqual(readout.select_permutation_count(2701, 2699)["selected_permutation_count"], 499)
        self.assertEqual(readout.select_permutation_count(2701, 2701)["status"], "abort_before_input_access")

    def test_raw_empirical_p_value_uses_plus_one_rule(self):
        result = readout.marginal_permutation_result([0.5, 0.6, 0.4, 0.5])
        self.assertEqual(result["raw_empirical_p_value"], 0.75)

    def test_infinite_scalar_threshold_endpoints_serialize_symbolically(self):
        value = readout._finite_json_value({"thresholds": [-float("inf"), 1.0, float("inf")]})
        self.assertEqual(value["thresholds"], ["-infinity", 1.0, "infinity"])
        json.dumps(value, allow_nan=False)

    def test_centered_global_maxT_is_metric_specific_and_synchronized(self):
        values = {
            "h{:02d}".format(index): [0.4 + index / 100.0, 0.2, 0.3, 0.4]
            for index in range(16)
        }
        result = readout.centered_global_max_t(values)
        self.assertEqual(len(result["hypotheses"]), 16)
        self.assertEqual(result["max_null_summary"]["count"], 3)
        self.assertTrue(all(
            row["maxT_family_size"] == 16 for row in result["hypotheses"].values()
        ))

    def test_difference_statistics_are_defined_as_synchronized_differences(self):
        definitions = readout._primary_definitions()
        differences = [row for row in definitions if row[3] == "difference"]
        self.assertEqual(len(differences), 8)
        observed_left = [0.8, 0.3, 0.6]
        observed_right = [0.5, 0.4, 0.2]
        self.assertEqual(
            [left - right for left, right in zip(observed_left, observed_right)],
            [0.30000000000000004, -0.10000000000000003, 0.39999999999999997],
        )

    def test_complete_negative_result_is_valid_and_non_authorizing(self):
        result = readout.validate_readout_results(_negative_results(), 499)
        self.assertFalse(result["authority"]["model_or_decoder_repair_authorized"])
        self.assertEqual(result["executed_pipeline_count"], 24)

    def test_access_declarations_forbid_every_expansive_action(self):
        declarations = readout.readout_access_declarations(input_accessed=True, completed=True)
        self.assertTrue(declarations["preserved_feature_label_artifact_accessed"])
        for name in (
            "corpus_accessed", "source_repaired_checkpoint_accessed",
            "ge1_model_constructed_or_loaded", "ge1_training_performed",
            "ge1_backward_pass_performed", "stage6_performed", "c8_or_later_performed",
        ):
            self.assertFalse(declarations[name])

    def test_target_content_is_rejected_from_detached_feature_rows(self):
        from prototype.graph_encoder.representation_probe import validate_feature_rows

        features, unused_labels = readout.synthetic_fixture()
        changed = [dict(row) for row in features]
        changed[0]["target"] = 1
        with self.assertRaises(GraphEncoderError) as context:
            validate_feature_rows(changed)
        self.assertEqual(context.exception.code, "target_leakage_in_features")

    def test_source_has_no_corpus_checkpoint_model_or_training_import_or_call(self):
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        imports = {
            alias.name for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) for alias in node.names
        }
        calls = {
            node.func.id for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        for forbidden in (
            "load_train", "load_development", "load_ge1_checkpoint",
            "build_ge1_model", "build_matched_ge1_models", "run_ge1_training",
            "extract_target_free_features",
        ):
            self.assertNotIn(forbidden, imports)
            self.assertNotIn(forbidden, calls)

    def test_runner_is_one_hour_cpu_and_mounts_only_source_output_then_inputs(self):
        text = RUNNER.read_text(encoding="utf-8")
        self.assertIn("#SBATCH --time=01:00:00", text)
        self.assertIn("#SBATCH --cpus-per-task=1", text)
        self.assertIn("--bind \"$INPUT_DIR:$INPUT_DIR:ro\"", text)
        self.assertNotIn("CORPUS_DIR", text)
        self.assertNotIn("CHECKPOINT_DIR", text)
        self.assertIn("closed_form_permutation_progress", SOURCE.read_text(encoding="utf-8"))
        self.assertIn("indeterminate_not_terminally_certified", text)
        timing_position = text.index("synthetic-timing")
        input_check_position = text.index('test -d "$INPUT_DIR"')
        self.assertLess(timing_position, input_check_position)

    def test_python_38_ast_and_narrow_exports(self):
        ast.parse(SOURCE.read_text(encoding="utf-8"), feature_version=(3, 8))
        import prototype.graph_encoder as package
        self.assertIn("READOUT_PROTOCOL_VERSION", package.__all__)
        self.assertIn("verify_readout_artifact", package.__all__)

    def test_atomic_finalization_creates_exact_five_files(self):
        with tempfile.TemporaryDirectory(prefix="readout-finalize-") as temporary:
            parent = Path(temporary)
            staging = parent / "artifact.incomplete-123"
            final = parent / "artifact"
            staging.mkdir()
            _atomic_write_jsonl(staging / "metrics.jsonl", [{
                "event": "closed_form_readout_completed",
                "protocol_version": readout.READOUT_PROTOCOL_VERSION,
            }])
            _atomic_write_json(staging / "readout_results.json", _negative_results())
            _atomic_write_json(staging / "resolved_config.json", {
                "protocol_version": readout.READOUT_PROTOCOL_VERSION,
                "selected_permutation_count": 499,
            })
            readout.finalize_readout_artifact(staging, final)
            self.assertFalse(staging.exists())
            self.assertEqual(
                tuple(sorted(path.name for path in final.iterdir())),
                tuple(sorted(readout.EXPECTED_FILES)),
            )
            self.assertTrue(readout.verify_readout_artifact(final)["integrity_verified"])

    def test_incomplete_artifact_is_rejected_as_final(self):
        with tempfile.TemporaryDirectory(prefix="readout-incomplete-") as temporary:
            root = Path(temporary) / "artifact.incomplete-123"
            root.mkdir()
            with self.assertRaises(GraphEncoderError) as context:
                readout.verify_readout_artifact(root)
        self.assertEqual(context.exception.code, "invalid_readout_artifact")


if __name__ == "__main__":
    unittest.main()
