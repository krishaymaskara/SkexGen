"""PyTorch-independent tests for the grid sufficiency protocol."""

from __future__ import annotations

import ast
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

from prototype.graph_encoder.decoder_contract import (
    LEGACY_OPERATION_MAGNITUDE_PARAMETERIZATION,
    GRID_ORDINAL_OPERATION_MAGNITUDE_PARAMETERIZATION,
)
from prototype.graph_encoder.errors import GraphEncoderError
from prototype.graph_encoder.operation_fidelity import (
    EXTRUSION_ABSOLUTE_ERROR_MAX_EXCLUSIVE,
    REVOLVE_ABSOLUTE_ERROR_MAX_EXCLUSIVE,
    operation_fidelity_contract,
    operation_fidelity_record,
    operation_geometry_fidelity_gate,
    summarize_operation_fidelity_family,
)
from prototype.model_data.vocab import NODE_TYPES
from prototype.graph_encoder import grid_magnitude_sufficiency as grid


ROOT = Path(__file__).resolve().parents[3]
GRID_MAGNITUDE_SOURCE = ROOT / "prototype/graph_encoder/grid_magnitude_sufficiency.py"
C7_V2_FILES = {
    "prototype/graph_encoder/c7_v2.py": (
        "67f83efe3ef58478ad3c00b846d43f8928b7c642e42ee8fcbc19cce0f63e01aa"
    ),
    "prototype/graph_encoder/tests/test_c7_v2_contract.py": (
        "c8e607095b9d2444f9bb1438f6784f089bcce6abdafdcd307fb4643050d96330"
    ),
    "prototype/graph_encoder/tests/test_c7_v2_runtime.py": (
        "78721bdbaac7c2a04223e079151de610a90718dcfcbb1525a4cc1b66c800380c"
    ),
}


def _record(operation="extrude", predicted_physical=1.0, target_physical=1.0,
            variant="continuous", **overrides):
    channel = 37 if operation == "extrude" else 38
    scale = 4.0 if operation == "extrude" else 360.0
    values = {
        "family_id": "family",
        "representation_variant": variant,
        "operation_index": 0,
        "target_operation_type": operation,
        "predicted_operation_type": operation,
        "predicted_channel": channel,
        "predicted_mask_active": True,
        "predicted_normalized_value": (
            None if predicted_physical is None else predicted_physical / scale
        ),
        "target_channel": channel,
        "target_mask_active": True,
        "target_normalized_value": target_physical / scale,
    }
    values.update(overrides)
    return operation_fidelity_record(**values)


def _gate(arm, name, status="pass"):
    return {
        "version": grid.GRID_MAGNITUDE_PROTOCOL_VERSION,
        "gate_version": grid.GRID_MAGNITUDE_GATE_VERSION,
        "gate_name": name,
        "status": status,
        "arm": arm,
        "operation_magnitude_parameterization": (
            GRID_ORDINAL_OPERATION_MAGNITUDE_PARAMETERIZATION
        ),
    }


def _all_gates(status="pass"):
    return {
        name: _gate(name.split(".")[0], name.split(".")[1], status)
        for name in grid.GRID_MAGNITUDE_REQUIRED_GATE_NAMES
    }


class OperationFidelityThresholdTests(unittest.TestCase):
    def test_contract_matches_authoritative_grids_scales_and_half_spacing(self):
        contract = operation_fidelity_contract()
        self.assertEqual(contract["extrude"]["physical_grid"], [0.5, 1.0, 1.5, 2.0, 3.0])
        self.assertEqual(contract["revolve"]["physical_grid"], [45.0, 90.0, 180.0, 270.0, 360.0])
        self.assertEqual(contract["extrude"]["normalization_scale"], 4.0)
        self.assertEqual(contract["revolve"]["normalization_scale"], 360.0)
        self.assertEqual(EXTRUSION_ABSOLUTE_ERROR_MAX_EXCLUSIVE, 0.25)
        self.assertEqual(REVOLVE_ABSOLUTE_ERROR_MAX_EXCLUSIVE, 22.5)

    def test_extrusion_just_below_threshold_passes(self):
        result = _record(predicted_physical=1.0 + (0.25 - 1e-12))
        self.assertEqual(result["status"], "pass")
        self.assertFalse(result["rounded_for_comparison"])

    def test_extrusion_at_or_above_threshold_fails(self):
        at = _record(predicted_physical=1.25)
        above = _record(predicted_physical=1.250000000001)
        self.assertEqual(at["status"], "fail")
        self.assertEqual(above["status"], "fail")

    def test_revolve_just_below_threshold_passes(self):
        result = _record(
            operation="revolve",
            predicted_physical=90.0 + (22.5 - 1e-10),
            target_physical=90.0,
        )
        self.assertEqual(result["status"], "pass")

    def test_revolve_at_or_above_threshold_fails(self):
        at = _record(
            operation="revolve", predicted_physical=112.5, target_physical=90.0
        )
        above = _record(
            operation="revolve",
            predicted_physical=112.500000001,
            target_physical=90.0,
        )
        self.assertEqual(at["status"], "fail")
        self.assertEqual(above["status"], "fail")

    def test_comparison_is_unrounded(self):
        result = _record(predicted_physical=1.249999999999)
        self.assertEqual(result["status"], "pass")
        self.assertGreater(result["absolute_physical_error"], 0.24999999999)

    def test_nonfinite_and_missing_values_fail(self):
        for value in (None, float("nan"), float("inf"), -float("inf")):
            result = _record(
                predicted_physical=None,
                predicted_normalized_value=value,
            )
            self.assertEqual(result["status"], "fail")
            self.assertIsNone(result["absolute_physical_error"])

    def test_positive_but_badly_inaccurate_prediction_fails(self):
        result = _record(predicted_physical=3.0, target_physical=0.5)
        self.assertGreater(result["predicted_physical_value"], 0.0)
        self.assertEqual(result["status"], "fail")

    def test_wrong_type_channel_or_mask_fails(self):
        cases = (
            {"predicted_operation_type": "revolve"},
            {"predicted_channel": 38},
            {"predicted_mask_active": False},
            {"target_channel": 38},
            {"target_mask_active": False},
        )
        for values in cases:
            with self.subTest(values=values):
                self.assertEqual(_record(**values)["status"], "fail")


class ConservativeAggregationTests(unittest.TestCase):
    def test_one_failed_operation_fails_two_operation_family(self):
        good = _record()
        bad = _record(
            operation="revolve",
            predicted_physical=180.0,
            target_physical=90.0,
            operation_index=1,
        )
        summary = summarize_operation_fidelity_family(
            (good, bad),
            family_id="family",
            operation_template="ER",
            representation_variants=("continuous",),
        )
        self.assertEqual(summary["status"], "fail")
        self.assertEqual(summary["maximum_absolute_physical_error"], 90.0)

    def test_one_failed_representation_variant_fails_family(self):
        good = _record(variant="continuous")
        bad = _record(
            variant="quantized", predicted_physical=2.0, target_physical=1.0
        )
        summary = summarize_operation_fidelity_family(
            (good, bad),
            family_id="family",
            operation_template="E",
            representation_variants=("continuous", "quantized"),
        )
        self.assertEqual(summary["status"], "fail")
        self.assertEqual(summary["operation_record_count"], 2)

    def test_averaging_cannot_hide_an_outlier(self):
        rows = tuple(_record() for unused in range(99)) + (
            _record(predicted_physical=3.0, target_physical=1.0),
        )
        summary = summarize_operation_fidelity_family(
            rows,
            family_id="family",
            operation_template="E",
            representation_variants=("continuous", "quantized"),
        )
        self.assertEqual(summary["status"], "fail")
        self.assertEqual(summary["aggregation"], "all_records_must_pass_no_averaging")
        self.assertEqual(summary["maximum_absolute_physical_error"], 2.0)

    def test_gate_records_each_variant_and_fails_one_outlier_family(self):
        examples = {}
        predictions = []
        for index in range(4):
            family_id = "E_{:02d}".format(index)
            target_geometry = [0.0] * 39
            target_geometry[37] = 0.25
            target_mask = [False] * 39
            target_mask[37] = True
            examples[family_id] = SimpleNamespace(
                metadata=SimpleNamespace(
                    operation_template="E",
                    geometry_encodings=("continuous", "quantized"),
                    sample_ids=("sample-a-" + family_id, "sample-b-" + family_id),
                ),
                target=SimpleNamespace(
                    operation_sequence=(0,),
                    node_type_ids=(NODE_TYPES.id("extrude"),),
                    geometry=(tuple(target_geometry),),
                    geometry_mask=(tuple(target_mask),),
                ),
            )
            predicted_geometry = list(target_geometry)
            if index == 3:
                predicted_geometry[37] = 0.5
            raw_node = SimpleNamespace(
                normalized_geometry=tuple(predicted_geometry),
                derived_geometry_mask=tuple(target_mask),
            )
            constrained = SimpleNamespace(
                graph=SimpleNamespace(
                    node_type_ids=(NODE_TYPES.id("extrude"),)
                ),
                node_prediction=SimpleNamespace(
                    raw_nodes=(raw_node,),
                    predicted_operation_node_indices=(0,),
                ),
            )
            predictions.append(SimpleNamespace(
                family_id=family_id,
                constrained_prediction=constrained,
            ))
        condition = SimpleNamespace(
            condition="P_true", available=True, predictions=tuple(predictions)
        )
        gate = operation_geometry_fidelity_gate(
            arm="flat",
            subset_identity="tiny",
            condition_result=condition,
            examples_by_family=examples,
            checkpoint_identity="checkpoint",
            epoch=200,
            access_flags={},
            protocol_version=grid.GRID_MAGNITUDE_PROTOCOL_VERSION,
            gate_version=grid.GRID_MAGNITUDE_GATE_VERSION,
            checkpoint_role=grid.GRID_MAGNITUDE_CHECKPOINT_ROLE,
            operation_magnitude_parameterization=(
                GRID_ORDINAL_OPERATION_MAGNITUDE_PARAMETERIZATION
            ),
        )
        self.assertEqual(len(gate["sample_operation_evidence"]), 8)
        self.assertEqual(gate["status"], "fail")
        self.assertEqual(gate["failing_family_ids"], ["E_03"])
        self.assertEqual(
            gate["family_summaries"]["E_03"][
                "maximum_absolute_physical_error"
            ],
            1.0,
        )


class ProtocolDecisionTests(unittest.TestCase):
    def test_frozen_arithmetic_and_identity(self):
        self.assertEqual(
            grid.GridMagnitudeSufficiencyConfiguration().to_dict()[
                "protocol_version"
            ],
            "GE1-C7-GRID-MAGNITUDE-SUFFICIENCY-v1",
        )
        tiny = grid.grid_magnitude_training_arithmetic(4)
        scaled = grid.grid_magnitude_training_arithmetic(32)
        self.assertEqual((tiny["optimizer_steps"], tiny["training_example_presentations"]), (200, 800))
        self.assertEqual((scaled["optimizer_steps"], scaled["training_example_presentations"]), (800, 6400))

    def test_tiny_failure_keeps_every_scaled_gate_not_run(self):
        gates = _all_gates()
        gates["flat.tiny_operation_geometry_fidelity"]["status"] = "fail"
        for arm in grid.GRID_MAGNITUDE_ARMS:
            for name in (
                "scaled_exact_sufficiency",
                "scaled_operation_geometry_fidelity",
                "scaled_memory_use",
            ):
                gates["{}.{}".format(arm, name)] = grid.grid_magnitude_not_run_gate(
                    arm=arm,
                    gate_name=name,
                    reason="tiny_failure",
                    access_flags={"scaled_train_payload_accessed": False},
                )
        decision = grid.overall_grid_magnitude_sufficiency_decision(gates)
        self.assertTrue(decision["grid_magnitude_scientific_failure"])
        self.assertFalse(decision["scaled_authorized"])
        self.assertFalse(decision["stage6_authorized_by_grid_magnitude_sufficiency"])

    def test_both_tiny_arms_must_pass_before_scaled_authorization(self):
        gates = _all_gates()
        gates["typed_graph.tiny_exact_sufficiency"]["status"] = "fail"
        decision = grid.overall_grid_magnitude_sufficiency_decision(gates)
        self.assertFalse(decision["scaled_authorized"])
        gates["typed_graph.tiny_exact_sufficiency"]["status"] = "pass"
        self.assertTrue(
            grid.overall_grid_magnitude_sufficiency_decision(gates)[
                "scaled_authorized"
            ]
        )

    def test_memory_only_failure_is_inconclusive(self):
        gates = _all_gates()
        gates["flat.scaled_memory_use"]["status"] = "fail"
        decision = grid.overall_grid_magnitude_sufficiency_decision(gates)
        self.assertTrue(decision["comparison_inconclusive"])
        self.assertFalse(decision["grid_magnitude_scientific_failure"])
        self.assertFalse(decision["stage6_authorized_by_grid_magnitude_sufficiency"])

    def test_even_a_complete_pass_never_auto_authorizes_later_work(self):
        decision = grid.overall_grid_magnitude_sufficiency_decision(_all_gates())
        self.assertTrue(decision["grid_magnitude_scientific_gate_pass"])
        self.assertFalse(
            decision["stage6_authorized_by_grid_magnitude_sufficiency"]
        )
        self.assertFalse(decision["another_scientific_job_authorized"])
        self.assertFalse(decision["additional_repair_authorized"])

    def test_legacy_recovery_checkpoint_is_ineligible(self):
        state = SimpleNamespace(
            completed_epoch=200,
            payload={
                "completed_epoch": 200,
                "selected_checkpoint_epoch": 200,
                "selected_experimental_checkpoint": True,
                "model_config": {
                    "operation_magnitude_parameterization": (
                        LEGACY_OPERATION_MAGNITUDE_PARAMETERIZATION
                    )
                },
                "provenance": {
                    "git_commit": "a" * 40,
                    "slurm_job_id": "123456",
                    "operation_magnitude_parameterization": (
                        LEGACY_OPERATION_MAGNITUDE_PARAMETERIZATION
                    ),
                },
            },
        )
        with self.assertRaisesRegex(
            GraphEncoderError, "invalid_grid_magnitude_training_checkpoint"
        ):
            grid.validate_grid_magnitude_reloaded_checkpoint(
                state, expected_commit="a" * 40, job_id="123456"
            )

    def test_grid_recovery_checkpoint_identity_is_mandatory(self):
        state = SimpleNamespace(
            completed_epoch=200,
            payload={
                "completed_epoch": 200,
                "selected_checkpoint_epoch": 200,
                "selected_experimental_checkpoint": True,
                "model_config": {
                    "operation_magnitude_parameterization": (
                        GRID_ORDINAL_OPERATION_MAGNITUDE_PARAMETERIZATION
                    )
                },
                "provenance": {
                    "git_commit": "a" * 40,
                    "slurm_job_id": "123456",
                    "operation_magnitude_parameterization": (
                        GRID_ORDINAL_OPERATION_MAGNITUDE_PARAMETERIZATION
                    ),
                },
            },
        )
        self.assertIs(
            grid.validate_grid_magnitude_reloaded_checkpoint(
                state, expected_commit="a" * 40, job_id="123456"
            ),
            state,
        )


class IsolationAndHistoryTests(unittest.TestCase):
    def test_existing_c7_v2_source_and_tests_are_byte_exact(self):
        for relative, expected in C7_V2_FILES.items():
            observed = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
            self.assertEqual(observed, expected, relative)

    def test_targets_enter_only_after_autonomous_generation(self):
        source = GRID_MAGNITUDE_SOURCE.read_text(encoding="utf-8")
        tree = ast.parse(source)
        function = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_train_evaluate_arm"
        )
        function_source = ast.get_source_segment(source, function)
        self.assertLess(
            function_source.index("run_autonomous_evaluation("),
            function_source.index("targets ="),
        )
        calls = [
            node for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "run_autonomous_evaluation"
        ]
        self.assertEqual(len(calls), 1)
        self.assertNotIn("target", [item.arg for item in calls[0].keywords])

    def test_no_target_tensor_is_accepted_by_autonomous_interfaces(self):
        path = ROOT / "prototype/graph_encoder/autonomous.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name in (
                "autonomous_input_from_paired",
                "run_autonomous_evaluation",
                "_decode_condition",
            ):
                names = [item.arg for item in node.args.args + node.args.kwonlyargs]
                self.assertNotIn("target", names)
                self.assertNotIn("target_geometry", names)

    def test_grid_source_does_not_import_protected_loaders(self):
        source = GRID_MAGNITUDE_SOURCE.read_text(encoding="utf-8")
        tree = ast.parse(source)
        imports = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            for alias in node.names
        }
        self.assertNotIn("load_development", imports)
        self.assertNotIn("load_partition_physical_examples", imports)
        self.assertNotIn("secondary_systematic_validation", source)

    def test_model_construction_explicitly_selects_grid_identity(self):
        source = GRID_MAGNITUDE_SOURCE.read_text(encoding="utf-8")
        function = next(
            node for node in ast.parse(source).body
            if isinstance(node, ast.FunctionDef)
            and node.name == "run_grid_magnitude_sufficiency"
        )
        calls = [
            node for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "build_matched_ge1_models"
        ]
        self.assertEqual(len(calls), 2)
        for call in calls:
            keywords = {item.arg: item.value for item in call.keywords}
            self.assertIn("operation_magnitude_parameterization", keywords)

    def test_scientific_runner_contract_passes_structural_audit(self):
        from prototype.graph_encoder.grid_magnitude_sufficiency_audit import (
            audit_grid_magnitude_sufficiency,
        )

        runner = (
            ROOT / "prototype/graph_encoder/adroit/"
            "ge1_grid_magnitude_sufficiency_cpu.slurm"
        )
        result = audit_grid_magnitude_sufficiency(
            ROOT / "prototype/graph_encoder", runner
        )
        self.assertEqual(result["scientific_entry_point_count"], 1)
        self.assertFalse(result["submission_command_invoked"])
        self.assertFalse(result["protected_input_declared"])


class InputVerificationTests(unittest.TestCase):
    def test_metadata_hashes_are_verified_before_payload_loading(self):
        with tempfile.TemporaryDirectory(prefix="grid-input-") as temporary:
            root = Path(temporary)
            operation = root / grid.AUTHORITATIVE_RELATIVE_FILE
            operation.parent.mkdir(parents=True)
            corpus_bytes = b'{"corpus":true}\n'
            operation_bytes = b'{"operation_template":true}\n'
            (root / "corpus_manifest.json").write_bytes(corpus_bytes)
            operation.write_bytes(operation_bytes)
            with mock.patch.object(
                grid,
                "CORPUS_MANIFEST_SHA256",
                hashlib.sha256(corpus_bytes).hexdigest(),
            ), mock.patch.object(
                grid,
                "AUTHORITATIVE_FILE_SHA256",
                hashlib.sha256(operation_bytes).hexdigest(),
            ):
                result = grid.verify_authorized_metadata_inputs(root)
            self.assertEqual(result["status"], "pass")
            self.assertTrue(
                result["verified_before_scientific_payload_loading"]
            )
            self.assertFalse(result["payload_accessed"])

    def test_metadata_hash_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory(prefix="grid-input-") as temporary:
            root = Path(temporary)
            operation = root / grid.AUTHORITATIVE_RELATIVE_FILE
            operation.parent.mkdir(parents=True)
            (root / "corpus_manifest.json").write_bytes(b"wrong\n")
            operation.write_bytes(b"wrong\n")
            with self.assertRaisesRegex(
                GraphEncoderError, "grid_magnitude_input_verification_failure"
            ):
                grid.verify_authorized_metadata_inputs(root)

    def test_payload_paths_cannot_escape_or_name_protected_inputs(self):
        for value in (
            "/absolute.json",
            "../escape.json",
            "manifests/operation_template.json",
            "protected/test.json",
        ):
            with self.subTest(value=value):
                with self.assertRaises(GraphEncoderError):
                    grid._authorized_payload_relative_path(value)

    def test_only_selected_canonical_payloads_are_exact_hash_verified(self):
        from prototype.controlled_data.builders import build_history
        from prototype.controlled_data.identity import (
            sample_id,
            source_family_id,
        )
        from prototype.graph_encoder.tests.fixtures import source
        from prototype.representation.model import GeometryEncoding
        from prototype.representation.serialization import history_to_json

        with tempfile.TemporaryDirectory(prefix="grid-payload-") as temporary:
            root = Path(temporary)
            (root / "samples").mkdir()
            records = []
            family_ids = []
            for template in ("E", "R", "EE", "RE"):
                physical = source(template)
                for encoding in (
                    GeometryEncoding.CONTINUOUS,
                    GeometryEncoding.QUANTIZED,
                ):
                    history = build_history(physical, encoding)
                    current_sample = sample_id(history)
                    current_family = source_family_id(history)
                    relative = "samples/{}.json".format(current_sample)
                    (root / relative).write_text(
                        history_to_json(history) + "\n", encoding="utf-8"
                    )
                    records.append({
                        "sample_id": current_sample,
                        "source_family_id": current_family,
                        "relative_json_path": relative,
                    })
                    family_ids.append(current_family)
            families = tuple(sorted(set(family_ids)))
            manifest = json.dumps(
                {"samples": records}, sort_keys=True, separators=(",", ":")
            ).encode("utf-8") + b"\n"
            (root / "corpus_manifest.json").write_bytes(manifest)
            # This unreadable nonselected file is never named by a selected
            # manifest record and therefore must remain untouched.
            (root / "samples/protected-unselected.json").write_bytes(b"bad")
            with mock.patch.object(
                grid,
                "CORPUS_MANIFEST_SHA256",
                hashlib.sha256(manifest).hexdigest(),
            ):
                result = grid.verify_authorized_payload_inputs(
                    root, families, subset_identity="tiny"
                )
            self.assertEqual(result["family_count"], 4)
            self.assertEqual(result["sample_count"], 8)
            self.assertEqual(len(result["records_sha256"]), 64)
            self.assertTrue(result["verified_before_scientific_loading"])


class ArtifactContractTests(unittest.TestCase):
    def test_separately_versioned_artifact_finalizes_and_verifies(self):
        with tempfile.TemporaryDirectory(prefix="grid-artifact-") as temporary:
            root = Path(temporary)
            repository = root / "repository"
            corpus = root / "corpus"
            repository.mkdir()
            corpus.mkdir()
            final, staging = grid.prepare_grid_magnitude_output(
                root / "outputs/result",
                repository_root=repository,
                corpus_dir=corpus,
                job_id="123456",
            )
            identities = []
            for arm in grid.GRID_MAGNITUDE_ARMS:
                for kind in ("recovery", "inference"):
                    relative = "checkpoints/tiny/{}/{}-epoch-0200.pt".format(
                        arm, kind
                    )
                    path = staging / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes((relative + "\n").encode("ascii"))
                    identities.append(relative)
            commit = "a" * 40
            resolved = {
                "protocol_version": grid.GRID_MAGNITUDE_PROTOCOL_VERSION,
                "metrics_version": grid.GRID_MAGNITUDE_METRICS_VERSION,
                "gate_version": grid.GRID_MAGNITUDE_GATE_VERSION,
                "artifact_version": grid.GRID_MAGNITUDE_ARTIFACT_VERSION,
                "operation_magnitude_parameterization": (
                    GRID_ORDINAL_OPERATION_MAGNITUDE_PARAMETERIZATION
                ),
                "source": {"git_commit": commit, "git_dirty": False},
                "grid_magnitude_sufficiency_configuration": (
                    grid.GridMagnitudeSufficiencyConfiguration().to_dict()
                ),
                "training_arithmetic": {
                    "tiny": grid.grid_magnitude_training_arithmetic(4),
                    "scaled": grid.grid_magnitude_training_arithmetic(32),
                },
                "input_verification": {
                    "metadata": {
                        "status": "pass",
                        "verified_before_scientific_payload_loading": True,
                    },
                    "tiny": {
                        "status": "pass",
                        "verified_before_scientific_loading": True,
                        "family_count": 4,
                        "sample_count": 8,
                    },
                    "scaled": {
                        "status": "not_run",
                        "payload_accessed": False,
                    },
                },
                "historical_records": {
                    "immutable_c7_v2": {
                        "commit": grid.IMMUTABLE_C7_V2_COMMIT,
                        "job_id": grid.IMMUTABLE_C7_V2_JOB,
                        "result_changed": False,
                    },
                    "operation_magnitude_engineering_validation": {
                        "result_changed": False,
                    },
                },
                "runtime": {
                    "python": "3.8.13",
                    "pytorch": "1.11.0",
                    "device": "cpu",
                    "cuda_available": False,
                    "cpu_threads": 1,
                    "slurm_job_id": "123456",
                },
            }
            grid._atomic_write_json(staging / "resolved_config.json", resolved)
            gates = _all_gates()
            gates["flat.tiny_operation_geometry_fidelity"]["status"] = "fail"
            for arm in grid.GRID_MAGNITUDE_ARMS:
                inference_identity = (
                    "checkpoints/tiny/{}/inference-epoch-0200.pt".format(arm)
                )
                for tiny_name in (
                    "tiny_exact_sufficiency",
                    "tiny_operation_geometry_fidelity",
                ):
                    gates["{}.{}".format(arm, tiny_name)].update({
                        "subset_identity": "tiny",
                        "checkpoint_identity": inference_identity,
                    })
                for name in (
                    "scaled_exact_sufficiency",
                    "scaled_operation_geometry_fidelity",
                    "scaled_memory_use",
                ):
                    gates["{}.{}".format(arm, name)] = grid.grid_magnitude_not_run_gate(
                        arm=arm,
                        gate_name=name,
                        reason="tiny_failure",
                        access_flags={"scaled_train_payload_accessed": False},
                    )
            decision = grid.overall_grid_magnitude_sufficiency_decision(gates)
            access = {
                "scaled_train_payload_accessed": False,
                "stage6_performed": False,
                "c8_or_later_performed": False,
                "cad_kernel_available": False,
                "positive_magnitude_repair_invoked": False,
                "grid_magnitude_parameterization_invoked": True,
                **{name: False for name in grid.GRID_MAGNITUDE_PROTECTED_ACCESS_FIELDS},
            }
            events = [
                {"event": "grid_magnitude_sufficiency_run_metadata"},
                {"event": "grid_magnitude_sufficiency_input_verification"},
                {"event": "grid_magnitude_sufficiency_subset_selection"},
                {"event": "grid_magnitude_sufficiency_training_started"},
            ]
            for arm in grid.GRID_MAGNITUDE_ARMS:
                pair = {
                    "recovery": "checkpoints/tiny/{}/recovery-epoch-0200.pt".format(arm),
                    "inference": "checkpoints/tiny/{}/inference-epoch-0200.pt".format(arm),
                }
                hashes = {kind: grid._file_sha256(staging / path) for kind, path in pair.items()}
                events.extend([
                    {
                        "event": "grid_magnitude_sufficiency_epoch_200_checkpoints_written",
                        "arm": arm,
                        "subset_identity": "tiny",
                        "checkpoint_identities": pair,
                        "checkpoint_sha256": hashes,
                    },
                    {
                        "event": "grid_magnitude_sufficiency_checkpoints_reloaded",
                        "arm": arm,
                        "subset_identity": "tiny",
                        "epoch": 200,
                        "checkpoint_identities": pair,
                        "checkpoint_sha256": hashes,
                        "strict_recovery_reload": True,
                        "strict_inference_reload": True,
                        "source_commit": commit,
                        "slurm_job_id": "123456",
                    },
                ])
            events.extend([
                {"event": "grid_magnitude_sufficiency_autonomous_condition_metrics"},
                *(
                    {"event": "grid_magnitude_sufficiency_gate_result", **gates[name]}
                    for name in sorted(gates)
                ),
                {"event": "grid_magnitude_sufficiency_overall_gate_decision", **decision},
                {
                    "event": "grid_magnitude_sufficiency_execution_completed",
                    "slurm_job_id": "123456",
                    "operation_magnitude_parameterization": (
                        GRID_ORDINAL_OPERATION_MAGNITUDE_PARAMETERIZATION
                    ),
                    **decision,
                    **access,
                },
            ])
            grid._atomic_write_jsonl(staging / "metrics.jsonl", events)
            grid.finalize_grid_magnitude_artifact(staging, final)
            verified = grid.verify_grid_magnitude_artifact(
                final,
                expected_commit=commit,
                expected_slurm_job_id="123456",
            )
            self.assertEqual(verified["checkpoint_count"], 4)
            self.assertFalse(verified["grid_magnitude_scientific_gate_pass"])


if __name__ == "__main__":
    unittest.main()
