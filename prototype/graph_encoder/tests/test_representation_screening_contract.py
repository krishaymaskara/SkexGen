"""PyTorch-independent contracts for the short representation screening."""

from __future__ import annotations

import ast
import copy
import json
from pathlib import Path
import tempfile
from unittest import mock
import unittest

from prototype.graph_encoder.errors import GraphEncoderError
from prototype.graph_encoder import representation_probe as probe
from prototype.graph_encoder import representation_screening as screen
from prototype.graph_encoder.pilot import _atomic_write_json, _atomic_write_jsonl


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "prototype/graph_encoder/representation_screening.py"
RUNNER = (
    ROOT
    / "prototype/graph_encoder/adroit/ge1_representation_screening_cpu.slurm"
)


def _label_and_feature_rows():
    labels = []
    features = []
    family_index = 0
    for template, family_count in (("E", 8), ("R", 8), ("EE", 8), ("RE", 8)):
        operations = {
            "E": ((0, "extrude"),),
            "R": ((0, "revolve"),),
            "EE": ((0, "extrude"), (1, "extrude")),
            "RE": ((0, "revolve"), (1, "extrude")),
        }[template]
        for unused in range(family_count):
            family_id = "family-{:02d}".format(family_index)
            for operation_index, operation_type in operations:
                label_key = "{}:{}:{}".format(
                    family_id, operation_index, operation_type
                )
                class_index = (family_index + operation_index) % 5
                labels.append({
                    "class_index": class_index,
                    "class_physical_value": probe.OPERATION_GRIDS[
                        operation_type
                    ][class_index],
                    "family_id": family_id,
                    "label_key": label_key,
                    "operation_index": operation_index,
                    "operation_template": template,
                    "operation_type": operation_type,
                })
                for arm in probe.PROBE_ARMS:
                    features.append({
                        "key": "{}:{}".format(arm, label_key),
                        "label_key": label_key,
                        "arm": arm,
                        "family_id": family_id,
                        "operation_index": operation_index,
                        "operation_type": operation_type,
                    })
            family_index += 1
    return tuple(labels), tuple(features)


def _metrics(count, accuracy=0.2, balanced=0.2):
    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced,
        "confusion_matrix": [[0 for unused in range(5)] for unused in range(5)],
        "class_support": [count, 0, 0, 0, 0],
        "per_class_recall": [0.0 for unused in range(5)],
        "observation_count": count,
    }


def _fake_pipeline(*unused_args, **kwargs):
    operation_type = kwargs["operation_type"]
    feature_level = kwargs["feature_level"]
    probe_kind = kwargs["probe_kind"]
    count = 32 if operation_type == "extrude" else 16
    linear = probe_kind == "linear"
    fold = {
        "held_out_family_id": "family",
        "selected_lambda": 1.0 if linear else None,
        "inner_selection": [] if linear else None,
        "seed": None if linear else 2026,
        "fit": {"synthetic": True},
    }
    return {
        "arm": kwargs["arm"],
        "operation_type": operation_type,
        "feature_level": feature_level,
        "probe_kind": probe_kind,
        "derived_parameter_count": (
            probe.LINEAR_PARAMETER_COUNTS
            if linear else probe.MLP_PARAMETER_COUNTS
        )[feature_level],
        "optimization_contract": {"synthetic_contract_record": True},
        "resubstitution": {
            "selected_lambda": 1.0 if linear else None,
            "regularization_selection": [] if linear else None,
            "seed": None if linear else 2026,
            "metrics": _metrics(count),
            "predictions_by_key": {},
            "fit": {"synthetic": True},
        },
        "lofo": {
            "metrics": _metrics(count),
            "predictions_by_key": {},
            "folds": [fold],
        },
    }


def _execution():
    return (
        {"git_commit": "synthetic", "slurm_job_id": "123"},
        {
            "python": "3.8.13",
            "pytorch": "1.11.0",
            "device": "cpu",
            "cuda_available": False,
            "cpu_threads": 1,
            "host": "synthetic",
            "slurm_job_id": "123",
        },
    )


class RepresentationScreeningContractTests(unittest.TestCase):
    def test_identity_hashes_and_sixteen_combinations_are_frozen(self):
        contract = screen.screen_contract()
        self.assertEqual(
            screen.SCREEN_PROTOCOL_VERSION,
            "GE1-C7-REPAIRED-REPRESENTATION-SCREEN-v1",
        )
        self.assertEqual(
            contract["source_feature_artifact"]["git_commit"],
            "3ea43650bb793d8d55b3d61a1edb70ec7a920089",
        )
        self.assertEqual(
            contract["source_feature_artifact"]["slurm_job_id"], "3346513"
        )
        self.assertEqual(len(screen.SCREEN_INPUT_HASHES), 3)
        self.assertEqual(len(screen.SCREEN_COMBINATIONS), 16)
        self.assertEqual(len(set(screen.SCREEN_COMBINATIONS)), 16)
        self.assertEqual(contract["pipeline"]["label_permutations"], 0)

    def test_accepted_probe_settings_are_reused_without_reduction(self):
        contract = screen.screen_contract()["pipeline"]
        self.assertEqual(
            contract["linear"]["regularization_grid"],
            list(probe.REGULARIZATION_GRID),
        )
        self.assertEqual(contract["linear"]["max_iter"], probe.LINEAR_MAX_ITER)
        self.assertEqual(contract["linear"]["history_size"], probe.LINEAR_HISTORY_SIZE)
        self.assertEqual(contract["mlp"]["updates"], probe.MLP_UPDATES)
        self.assertEqual(contract["mlp"]["hidden_width"], probe.MLP_WIDTH)
        self.assertEqual(contract["mlp"]["learning_rate"], probe.MLP_LEARNING_RATE)
        self.assertEqual(contract["linear"]["parameter_counts"], {"C": 165, "D": 185})
        self.assertEqual(contract["mlp"]["parameter_counts"], {"C": 309, "D": 341})

    def test_candidate_flag_requires_both_inclusive_thresholds(self):
        self.assertTrue(screen.candidate_for_full_controls({
            "accuracy": 0.40, "balanced_accuracy": 0.40,
        }))
        self.assertFalse(screen.candidate_for_full_controls({
            "accuracy": 0.399999, "balanced_accuracy": 1.0,
        }))
        self.assertFalse(screen.candidate_for_full_controls({
            "accuracy": 1.0, "balanced_accuracy": 0.399999,
        }))

    def test_input_hash_validator_rejects_any_difference(self):
        self.assertEqual(
            screen._validate_input_hashes(screen.SCREEN_INPUT_HASHES),
            screen.SCREEN_INPUT_HASHES,
        )
        changed = dict(screen.SCREEN_INPUT_HASHES)
        changed["labels.json"] = "0" * 64
        with self.assertRaises(GraphEncoderError) as context:
            screen._validate_input_hashes(changed)
        self.assertEqual(context.exception.code, "screening_input_hash_mismatch")

    def test_source_calls_production_fitter_and_no_permutation_or_loader(self):
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        imports = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            for alias in node.names
        }
        calls = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        self.assertIn("fit_probe_pipeline", imports)
        self.assertIn("fit_probe_pipeline", calls)
        for forbidden in (
            "run_permutation_controls",
            "run_representation_probe",
            "extract_target_free_features",
            "load_train",
            "select_c7_sufficiency_subsets",
            "load_ge1_checkpoint",
            "build_ge1_model",
            "build_matched_ge1_models",
        ):
            self.assertNotIn(forbidden, imports)
            self.assertNotIn(forbidden, calls)

    def test_exact_sixteen_real_label_calls_have_no_test_limits(self):
        labels, features = _label_and_feature_rows()
        source, environment = _execution()
        with mock.patch.object(
            screen, "fit_probe_pipeline", side_effect=_fake_pipeline
        ) as fitter:
            results = screen.run_real_label_screen(
                features,
                labels,
                screen.SCREEN_INPUT_HASHES,
                execution_source=source,
                environment=environment,
            )
        self.assertEqual(fitter.call_count, 16)
        observed = tuple(
            (
                item.kwargs["arm"],
                item.kwargs["operation_type"],
                item.kwargs["feature_level"],
                item.kwargs["probe_kind"],
            )
            for item in fitter.call_args_list
        )
        self.assertEqual(observed, screen.SCREEN_COMBINATIONS)
        self.assertTrue(all("test_limits" not in item.kwargs for item in fitter.call_args_list))
        self.assertEqual(len(results["pipelines"]), 16)
        self.assertFalse(results["screening_summary"]["permutation_significance_available"])

    def test_missing_pipeline_cannot_validate_or_finalize(self):
        labels, features = _label_and_feature_rows()
        source, environment = _execution()
        with mock.patch.object(
            screen, "fit_probe_pipeline", side_effect=_fake_pipeline
        ):
            results = screen.run_real_label_screen(
                features,
                labels,
                screen.SCREEN_INPUT_HASHES,
                execution_source=source,
                environment=environment,
            )
        incomplete = copy.deepcopy(results)
        incomplete["pipelines"].pop()
        incomplete["pipeline_count"] = 15
        with self.assertRaises(GraphEncoderError) as context:
            screen.validate_screening_results(incomplete)
        self.assertEqual(context.exception.code, "invalid_screening_results")

    def test_negative_screening_finalizes_without_authority(self):
        labels, features = _label_and_feature_rows()
        source, environment = _execution()
        with mock.patch.object(
            screen, "fit_probe_pipeline", side_effect=_fake_pipeline
        ):
            results = screen.run_real_label_screen(
                features,
                labels,
                screen.SCREEN_INPUT_HASHES,
                execution_source=source,
                environment=environment,
            )
        self.assertEqual(results["screening_summary"]["candidate_pipeline_count"], 0)
        access = screen.screening_access_declarations(
            input_accessed=True, completed=True
        )
        with tempfile.TemporaryDirectory(prefix="screen-artifact-") as temporary:
            root = Path(temporary)
            staging = root / "screen.incomplete-123"
            final = root / "screen"
            staging.mkdir()
            _atomic_write_json(staging / "screening_results.json", results)
            _atomic_write_json(staging / "resolved_config.json", {
                "protocol_version": screen.SCREEN_PROTOCOL_VERSION,
                "screen_contract": screen.screen_contract(),
                "input_hashes": screen.SCREEN_INPUT_HASHES,
                "source": {"git_commit": "synthetic"},
                "runtime": {
                    "python": "3.8.13",
                    "pytorch": "1.11.0",
                    "device": "cpu",
                    "cuda_available": False,
                    "cpu_threads": 1,
                    "slurm_job_id": "123",
                },
                "access": access,
            })
            _atomic_write_jsonl(staging / "metrics.jsonl", [{
                "event": "representation_screening_completed",
                **access,
            }])
            screen.finalize_screening_artifact(staging, final)
            verified = screen.verify_screening_artifact(
                final, expected_commit="synthetic", expected_slurm_job_id="123"
            )
        self.assertEqual(verified["pipeline_count"], 16)
        self.assertEqual(verified["candidate_pipeline_count"], 0)
        self.assertFalse(access["stage6_performed"])
        self.assertFalse(access["c8_or_later_performed"])
        self.assertFalse(access["model_or_decoder_repair_implemented_or_invoked"])
        self.assertFalse(access["formal_adr0011_representation_probe_completed"])

    def test_runner_is_one_hour_cpu_only_and_mounts_no_corpus_or_checkpoint(self):
        source = RUNNER.read_text(encoding="utf-8")
        self.assertIn("#SBATCH --time=01:00:00", source)
        self.assertIn("#SBATCH --cpus-per-task=1", source)
        self.assertNotIn("#SBATCH --gres", source)
        self.assertNotIn("sbatch ", source)
        self.assertNotIn("CORPUS_DIR", source)
        self.assertNotIn("SOURCE_ARTIFACT", source)
        self.assertNotIn("SOURCE_CHECKPOINT", source)
        self.assertIn("$INPUT_DIR:$INPUT_DIR:ro", source)
        self.assertIn("GE1-C7-REPAIRED-REPRESENTATION-SCREEN-v1", source)


if __name__ == "__main__":
    unittest.main()
