"""Independent CPU tests for bounded VQ-collapse diagnosis."""

from __future__ import annotations

import csv
from contextlib import redirect_stderr, redirect_stdout
import io
import json
import math
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from prototype.flat_baseline.diagnose_vq_collapse import (
    DiagnosisError,
    HYPOTHESES,
    REQUIRED_ARTIFACTS,
    _decoder_supervision_masks,
    _managed_checkpoint_contract,
    aggregate_gradients,
    build_final_report,
    build_artifacts,
    checkpoint_inventory,
    checkpoint_timeline_limitation,
    classify_root_causes,
    codebook_statistics,
    decoder_sensitivity_statistics,
    histogram_jensen_shannon,
    history_inventory,
    latent_statistics,
    loss_masking_accounting,
    partition_reconciliation,
    publish_diagnostic_artifacts,
    main,
    select_permitted_ids,
    summarize_vectors,
    validate_diagnostic_artifacts,
    validate_workflow_publications,
)
from prototype.flat_baseline.evaluate_length_conditioned import (
    _identifier_sha256,
)

try:
    import torch
except (ImportError, OSError):
    torch = None


class DiagnosisTests(unittest.TestCase):
    def family(self, family_id, partition):
        return SimpleNamespace(
            physical_family_id=family_id,
            partition=partition,
        )

    def reconciliation(self):
        examples = (
            self.family("train-b", "train"),
            self.family("test-a", "test"),
            self.family("validation-a", "validation"),
            self.family("train-a", "train"),
        )
        checkpoint = {
            "train_partition": "train",
            "validation_partition": "validation",
            "train_family_ids": ["train-a", "train-b"],
            "validation_family_ids": ["validation-a"],
        }
        return partition_reconciliation(examples, checkpoint)

    def payloads(self):
        digest = "a" * 64
        inventory = {
            "checkpoints": [{
                "relative_path": "best.pt",
                "sha256": digest,
                "timeline_usable": True,
                "codebook_size": 2,
                "latent_tokens": 2,
            }],
            "history_files": [],
            "timeline_checkpoint_count": 1,
            "timeline_limitation": (
                "collapse time cannot be reconstructed from one checkpoint"
            ),
        }
        partition = {
            "total_physical_family_count": 3,
            "partition_counts": {"train": 1, "validation": 1, "test": 1},
            "partition_id_sha256": {
                "train": "b" * 64,
                "validation": "c" * 64,
                "test": "d" * 64,
            },
            "checkpoint_partition_ids": {
                "train": ["train-a"],
                "validation": ["validation-a"],
            },
            "checkpoint_partition_ids_match": True,
            "selected_family_ids": {
                "train": ["train-a"],
                "validation": ["validation-a"],
            },
            "selected_family_id_sha256": {
                "train": _identifier_sha256(("train-a",)),
                "validation": _identifier_sha256(("validation-a",)),
            },
            "test_partition_evaluated": False,
        }
        latent_rows = []
        for partition_name in ("train", "validation"):
            for code, count in ((0, 2), (1, 0)):
                latent_rows.append({
                    "checkpoint_sha256": digest,
                    "checkpoint_relative_path": "best.pt",
                    "partition": partition_name,
                    "code_index": code,
                    "assignment_count": count,
                    "assignment_fraction": count / 2.0,
                })
        return {
            "run_metadata.json": {
                "diagnostic_schema_version": 1,
                "checkpoint_sha256": [digest],
                "test_partition_evaluated": False,
            },
            "checkpoint_inventory.json": inventory,
            "partition_summary.json": partition,
            "latent_usage.csv": latent_rows,
            "prequant_summary.json": {"records": []},
            "codebook_summary.json": {"records": []},
            "decoder_sensitivity.json": {"records": []},
            "loss_masking_audit.json": {"records": []},
            "gradient_probe.json": {
                "optimizer_created": False,
                "optimizer_step_called": False,
                "checkpoint_mutated": False,
                "records": [],
            },
            "root_cause_report.json": {
                "hypotheses": [
                    {"hypothesis": name} for name in HYPOTHESES
                ],
                "test_partition_evaluated": False,
            },
        }

    def verification_result(self, digest):
        return {
            "artifact_sha256": {"artifact.json": digest},
            "root_cause_answers": {"answer": digest},
            "test_partition_evaluated": False,
        }

    def workflow_publications(self, root, symlink):
        directories = tuple(
            root / name
            for name in ("vq-smoke-a", "vq-smoke-b", "vq-diagnosis")
        )
        files = (root / "vq-diagnosis-report.json",)
        for public in directories + files:
            backing = root / ("." + public.name + ".tmp-backing")
            if public in directories:
                backing.mkdir()
            else:
                backing.write_text("{}\n")
            if symlink:
                public.symlink_to(
                    backing.name, target_is_directory=public in directories
                )
            else:
                backing.rename(public)
        return directories, files

    def test_exact_partition_reconciliation_and_test_exclusion(self):
        result = self.reconciliation()
        self.assertEqual(result["total_physical_family_count"], 4)
        self.assertEqual(
            result["partition_counts"],
            {"train": 2, "validation": 1, "test": 1},
        )
        self.assertTrue(result["checkpoint_partition_ids_match"])
        selected = select_permitted_ids(result, 1, 1)
        self.assertEqual(selected["train"], ("train-a",))
        self.assertEqual(selected["validation"], ("validation-a",))
        self.assertNotIn("test-a", selected["train"] + selected["validation"])
        self.assertFalse(result["test_partition_evaluated"])

    def test_partition_reconciliation_rejects_duplicate_families(self):
        with self.assertRaisesRegex(DiagnosisError, "duplicate_family"):
            partition_reconciliation((
                self.family("same", "train"),
                self.family("same", "test"),
            ))

    def test_latent_histogram_entropy_perplexity_and_all_one_code(self):
        result = latent_statistics((17, 17, 17, 17), 32)
        self.assertEqual(sum(result["histogram"]), 4)
        self.assertEqual(result["active_code_count"], 1)
        self.assertEqual(result["histogram"][17], 4)
        self.assertEqual(result["empirical_entropy_nats"], 0.0)
        self.assertEqual(result["perplexity"], 1.0)
        self.assertEqual(result["maximum_code_share"], 1.0)
        balanced = latent_statistics((0, 0, 1, 1), 2)
        self.assertAlmostEqual(balanced["empirical_entropy_nats"], math.log(2))
        self.assertAlmostEqual(balanced["perplexity"], 2.0)
        self.assertEqual(
            histogram_jensen_shannon((2, 0), (0, 2)), math.log(2)
        )

    def test_varied_prequant_vectors_can_map_to_one_code(self):
        vectors = ((0.0, 0.0), (0.1, 0.0), (0.2, 0.0))
        summary = summarize_vectors(vectors)
        assignments = latent_statistics((0, 0, 0), 2)
        self.assertEqual(summary["numerically_distinct_vector_count"], 3)
        self.assertGreater(summary["overall_variance"], 0.0)
        self.assertEqual(assignments["active_code_count"], 1)

    def test_identical_and_inactive_distinct_codebooks(self):
        collapsed = codebook_statistics(
            ((1.0, 1.0), (1.0, 1.0)), active_codes=(0,)
        )
        self.assertEqual(collapsed["numerically_distinct_vector_count"], 1)
        self.assertEqual(collapsed["near_duplicate_entry_count"], 1)
        distinct = codebook_statistics(
            ((0.0, 0.0), (2.0, 0.0), (0.0, 2.0)),
            active_codes=(0,),
        )
        self.assertEqual(distinct["numerically_distinct_vector_count"], 3)
        self.assertEqual(distinct["dead_code_count"], 2)
        self.assertGreater(distinct["pairwise_distance"]["minimum"], 0.0)

    def test_decoder_sensitivity_and_insensitivity(self):
        insensitive = decoder_sensitivity_statistics({
            0: (1.0, -1.0),
            1: (1.0, -1.0),
        })
        self.assertFalse(insensitive["materially_sensitive"])
        self.assertEqual(insensitive["distinct_complete_decision_count"], 1)
        sensitive = decoder_sensitivity_statistics({
            0: (2.0, 0.0, 0.0, 2.0),
            1: (0.0, 2.0, 2.0, 0.0),
        }, class_width=2)
        self.assertTrue(sensitive["materially_sensitive"])
        self.assertEqual(sensitive["decision_type"], "argmax")
        self.assertEqual(
            sensitive["argmax_or_binary_decision_change_fraction"], 1.0
        )

    def test_loss_denominators_class_frequency_and_pad_dominance(self):
        audit = loss_masking_accounting(({
            "counts": {
                "padded_node_slots": 8,
                "supervised_node_slots": 2,
                "total_node_slots": 10,
                "geometry_applicable_channels": 2,
                "geometry_total_channels": 20,
                "edge_present_pairs": 1,
                "edge_absent_pairs": 9,
                "edge_valid_pairs": 10,
                "pointer_supervised_slots": 1,
                "pointer_total_slots": 4,
            },
            "class_counts": {
                "node_type": {"0": 1, "1": 9},
                "edge_presence": {"0": 9, "1": 1},
            },
        },))
        self.assertEqual(audit["proportions"]["pad_token_proportion"], 0.8)
        self.assertEqual(
            audit["proportions"]["geometry_applicable_channel_proportion"],
            0.1,
        )
        self.assertEqual(
            audit["majority_class_baseline_accuracy"]["node_type"], 0.9
        )

    def test_missing_history_and_checkpoint_timeline_limitations(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(history_inventory(directory), [])
            Path(directory, "z.pt").write_bytes(b"z")
            Path(directory, "a.pt").write_bytes(b"a")
            inventory, payloads = checkpoint_inventory(directory)
        self.assertEqual(
            [record["relative_path"] for record in inventory],
            ["a.pt", "z.pt"],
        )
        self.assertTrue(all(
            not record["managed_training_checkpoint"]
            for record in inventory
        ))
        self.assertEqual(payloads, {})
        self.assertIsNotNone(checkpoint_timeline_limitation([
            {"timeline_usable": True}
        ]))
        self.assertIsNone(checkpoint_timeline_limitation([
            {"timeline_usable": True},
            {"timeline_usable": True},
        ]))

    def test_symlink_checkpoint_is_inventoried_but_never_loaded(self):
        calls = []
        torch_module = SimpleNamespace(
            load=lambda path, map_location: calls.append(
                (path, map_location)
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target.bin"
            target.write_bytes(b"checkpoint-like")
            (root / "linked.pt").symlink_to(target.name)
            inventory, payloads = checkpoint_inventory(root, torch_module)
        self.assertEqual(calls, [])
        self.assertEqual(payloads, {})
        self.assertTrue(inventory[0]["is_symlink"])
        self.assertEqual(
            inventory[0]["load_status"], "incompatible:symlink"
        )

    def test_only_provenance_backed_root_checkpoints_are_managed(self):
        self.assertTrue(_managed_checkpoint_contract(
            {"relative_path": "best.pt", "is_symlink": False},
            {"checkpoint_kind": "best"},
        ))
        self.assertFalse(_managed_checkpoint_contract(
            {"relative_path": "archive/best.pt", "is_symlink": False},
            {"checkpoint_kind": "best"},
        ))
        self.assertFalse(_managed_checkpoint_contract(
            {"relative_path": "best.pt", "is_symlink": True},
            {"checkpoint_kind": "best"},
        ))
        self.assertFalse(_managed_checkpoint_contract(
            {"relative_path": "best.pt", "is_symlink": False},
            {"checkpoint_kind": "last"},
        ))

    def test_gradient_aggregation_missing_zero_finite_nonfinite(self):
        result = aggregate_gradients((
            ("missing", None),
            ("zero", (0.0, 0.0)),
            ("finite", (3.0, 4.0)),
            ("nonfinite", (float("inf"),)),
        ))
        self.assertEqual(result["aggregate_norm"], 5.0)
        self.assertEqual(result["missing_gradient_count"], 1)
        self.assertEqual(result["zero_gradient_count"], 1)
        self.assertEqual(result["nonfinite_gradient_count"], 1)

    def test_invalid_numeric_tolerances_are_rejected(self):
        for value in (0.0, -1.0, float("nan"), float("inf"), True):
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    DiagnosisError, "invalid_tolerance"
                ):
                    summarize_vectors(((0.0,),), value)
                with self.assertRaisesRegex(
                    DiagnosisError, "invalid_tolerance"
                ):
                    decoder_sensitivity_statistics(
                        {0: (0.0,), 1: (1.0,)}, value
                    )

    def test_deterministic_artifacts_and_inconsistent_count_rejection(self):
        first = build_artifacts(self.payloads())
        second = build_artifacts(self.payloads())
        self.assertEqual(first, second)
        self.assertEqual(
            set(first), set(REQUIRED_ARTIFACTS) | {"artifact_manifest.json"}
        )
        for content in first.values():
            self.assertTrue(content.endswith(b"\n"))
            self.assertFalse(content.endswith(b"\n\n"))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, content in first.items():
                (root / name).write_bytes(content)
            validate_diagnostic_artifacts(root)
            rows = list(csv.DictReader(io.StringIO(
                (root / "latent_usage.csv").read_text()
            )))
            rows[0]["assignment_count"] = "1"
            stream = io.StringIO(newline="")
            writer = csv.DictWriter(
                stream, fieldnames=rows[0], lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(rows)
            (root / "latent_usage.csv").write_text(stream.getvalue())
            manifest = json.loads(
                (root / "artifact_manifest.json").read_text()
            )
            import hashlib
            manifest["artifact_sha256"]["latent_usage.csv"] = hashlib.sha256(
                (root / "latent_usage.csv").read_bytes()
            ).hexdigest()
            (root / "artifact_manifest.json").write_text(
                json.dumps(
                    manifest, sort_keys=True, separators=(",", ":")
                ) + "\n"
            )
            with self.assertRaisesRegex(
                DiagnosisError, "latent_reconciliation"
            ):
                validate_diagnostic_artifacts(root)

    def test_publication_collision_preserves_existing_destination(self):
        artifacts = build_artifacts(self.payloads())
        with tempfile.TemporaryDirectory() as directory:
            published = Path(directory) / "published"
            publish_diagnostic_artifacts(published, artifacts)
            self.assertEqual(
                validate_diagnostic_artifacts(published)[
                    "test_partition_evaluated"
                ],
                False,
            )
            output = Path(directory) / "diagnosis"
            output.write_bytes(b"untouched")
            with self.assertRaisesRegex(DiagnosisError, "output_collision"):
                publish_diagnostic_artifacts(output, artifacts)
            self.assertEqual(output.read_bytes(), b"untouched")
            self.assertFalse(tuple(
                Path(directory).glob(".diagnosis.tmp-*")
            ))

    def test_referenced_symlink_backing_objects_are_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            directories, files = self.workflow_publications(root, True)
            referenced = validate_workflow_publications(
                root, directories, files
            )
            self.assertEqual(len(referenced), 4)
            self.assertTrue(all(
                (root / name).exists() for name in referenced
            ))

    def test_dangling_public_symlink_fails_integrity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            public = root / "vq-smoke-a"
            public.symlink_to(".vq-smoke-a.tmp-missing")
            with self.assertRaisesRegex(
                DiagnosisError, "publication_integrity"
            ):
                validate_workflow_publications(root, (public,), ())

    def test_absolute_or_escaping_public_symlink_fails_integrity(self):
        for target in ("/tmp/backing", "../backing"):
            with self.subTest(target=target):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    public = root / "vq-smoke-a"
                    public.symlink_to(target)
                    with self.assertRaisesRegex(
                        DiagnosisError, "publication_integrity"
                    ):
                        validate_workflow_publications(
                            root, (public,), ()
                        )

    def test_unreferenced_temporary_backing_object_fails_integrity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            directories, files = self.workflow_publications(root, True)
            (root / ".vq-unreferenced.tmp-residue").mkdir()
            with self.assertRaisesRegex(
                DiagnosisError, "unreferenced backing object"
            ):
                validate_workflow_publications(
                    root, directories, files
                )

    def test_regular_rename_publications_are_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            directories, files = self.workflow_publications(root, False)
            self.assertEqual(
                validate_workflow_publications(
                    root, directories, files
                ),
                (),
            )

    def test_slurm_uses_reference_aware_publication_integrity_gate(self):
        script = Path(__file__).parents[1] / (
            "adroit/diagnose_vq_collapse_cpu.slurm"
        )
        content = script.read_text()
        self.assertIn("validate_workflow_publications(", content)
        self.assertNotIn(
            """find "$RUN_ROOT" -maxdepth 1 -name '.vq-*.tmp-*'""",
            content,
        )

    def test_gradient_artifact_proves_no_optimizer_or_mutation(self):
        artifacts = build_artifacts(self.payloads())
        gradient = json.loads(artifacts["gradient_probe.json"])
        self.assertFalse(gradient["optimizer_created"])
        self.assertFalse(gradient["optimizer_step_called"])
        self.assertFalse(gradient["checkpoint_mutated"])

    def test_identical_smoke_hashes_pass(self):
        smoke = self.verification_result("a" * 64)
        report = build_final_report(
            self.verification_result("b" * 64), smoke, dict(smoke)
        )
        self.assertTrue(report["byte_identical_smoke_replay"])

    def test_different_smoke_hashes_fail(self):
        with self.assertRaisesRegex(DiagnosisError, "replay_mismatch"):
            build_final_report(
                self.verification_result("c" * 64),
                self.verification_result("a" * 64),
                self.verification_result("b" * 64),
            )

    def test_full_hashes_may_differ_from_identical_smoke_hashes(self):
        full = self.verification_result("f" * 64)
        smoke = self.verification_result("e" * 64)
        report = build_final_report(full, smoke, dict(smoke))
        self.assertEqual(report["artifact_sha256"], full["artifact_sha256"])
        self.assertTrue(report["byte_identical_smoke_replay"])
        self.assertTrue(report["full_diagnosis_verified"])

    def test_final_report_preserves_byte_identical_smoke_replay(self):
        full = self.verification_result("b" * 64)
        smoke = self.verification_result("a" * 64)
        with tempfile.TemporaryDirectory() as directory:
            report_path = Path(directory) / "report.json"
            arguments = [
                "verify",
                "--output", "full",
                "--corpus-dir", "corpus",
                "--training-run-dir", "training",
                "--reviewed-commit", "commit",
                "--smoke-output-a", "smoke-a",
                "--smoke-output-b", "smoke-b",
                "--report-output", str(report_path),
            ]
            with patch(
                "prototype.flat_baseline.diagnose_vq_collapse."
                "load_physical_examples",
                return_value=(),
            ), patch(
                "prototype.flat_baseline.diagnose_vq_collapse."
                "validate_diagnostic_artifacts",
                side_effect=[full, smoke, dict(smoke)],
            ), redirect_stdout(io.StringIO()):
                self.assertEqual(main(arguments), 0)
            report = json.loads(report_path.read_text())
        self.assertIs(report["byte_identical_smoke_replay"], True)
        self.assertIs(report["full_diagnosis_verified"], True)

    def test_final_report_preserves_test_partition_protection(self):
        smoke = self.verification_result("a" * 64)
        report = build_final_report(
            self.verification_result("f" * 64), smoke, dict(smoke)
        )
        self.assertFalse(report["test_partition_evaluated"])

    def test_full_verification_failure_prevents_final_report(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.json"
            common = [
                "verify",
                "--output", "full",
                "--corpus-dir", "corpus",
                "--training-run-dir", "training",
                "--reviewed-commit", "commit",
                "--smoke-output-a", "smoke-a",
                "--smoke-output-b", "smoke-b",
                "--report-output", str(report),
            ]
            with patch(
                "prototype.flat_baseline.diagnose_vq_collapse."
                "load_physical_examples",
                return_value=(),
            ), patch(
                "prototype.flat_baseline.diagnose_vq_collapse."
                "validate_diagnostic_artifacts",
                side_effect=DiagnosisError(
                    "artifact_hash", "full diagnosis is invalid"
                ),
            ), redirect_stderr(io.StringIO()):
                self.assertEqual(main(common), 1)
            self.assertFalse(report.exists())

    def test_slurm_success_gate_never_compares_full_and_smoke_hashes(self):
        script = Path(__file__).parents[1] / (
            "adroit/diagnose_vq_collapse_cpu.slurm"
        )
        content = script.read_text()
        self.assertIn('--compare-output "$SMOKE_B"', content)
        self.assertNotIn('--compare-output "$FULL"', content)
        final_gate = content.split("stage final_report", 1)[1]
        self.assertIn('--output "$FULL"', final_gate)
        self.assertIn('--smoke-output-a "$SMOKE_A"', final_gate)
        self.assertIn('--smoke-output-b "$SMOKE_B"', final_gate)
        self.assertIn(
            'report["byte_identical_smoke_replay"] is True',
            final_gate,
        )
        self.assertIn(
            'report["full_diagnosis_verified"] is True', final_gate
        )

    def test_root_cause_classification_separates_observation_and_inference(self):
        prequant = [{
            "vectors": {"numerically_distinct_vector_count": 3},
            "latent_statistics": {
                "active_code_count": 1,
                "histogram": [0] * 17 + [4] + [0] * 14,
            },
        }]
        codebook = [{"numerically_distinct_vector_count": 32}]
        sensitivity = [{
            "head_statistics": {
                "node_type": {"materially_sensitive": False}
            }
        }]
        losses = [{
            "proportions": {
                "geometry_applicable_channel_proportion": 0.1
            },
            "majority_class_baseline_accuracy": {"node_type": 0.9},
            "relative_weighted_contribution": {"node_type": 0.8},
        }]
        gradients = [{
            "module_groups": {
                "encoder": {
                    "aggregate_norm": 1.0,
                    "nonfinite_gradient_count": 0,
                }
            }
        }]
        report = classify_root_causes(
            prequant,
            codebook,
            sensitivity,
            losses,
            gradients,
            [{"timeline_usable": True}],
            {"event_count": 0},
        )
        self.assertEqual(
            tuple(item["hypothesis"] for item in report["hypotheses"]),
            HYPOTHESES,
        )
        self.assertFalse(
            report["observations"]["encoder_outputs_effectively_constant"]
        )
        self.assertIn(
            "nearest-neighbor",
            report["answers"]["why_code_17"],
        )
        self.assertFalse(report["answers"]["collapse_time_identifiable"])


@unittest.skipUnless(torch is not None, "PyTorch is required")
class PyTorchDiagnosisIntegrationTests(unittest.TestCase):
    def test_decoder_intervention_uses_exact_supervision_masks(self):
        target = {
            "node_mask": torch.tensor([
                [True, True, False],
                [True, False, False],
            ]),
            "geometry_mask": torch.tensor([
                [[True, False], [False, True], [True, True]],
                [[True, True], [True, True], [True, True]],
            ]),
            "operation_mask": torch.tensor([
                [True, False],
                [True, True],
            ]),
            "categorical_attributes": torch.zeros(
                2, 3, 9, dtype=torch.long
            ),
        }

        def dense_edge_targets(current, node_count):
            del current
            valid = torch.tensor([
                [
                    [False, True, False],
                    [True, False, False],
                    [False, False, False],
                ],
                [
                    [False, False, False],
                    [False, False, False],
                    [False, False, False],
                ],
            ])
            presence = torch.zeros_like(valid)
            presence[0, 0, 1] = True
            return presence, torch.zeros(
                2, node_count, node_count, dtype=torch.long
            ), valid

        masks = _decoder_supervision_masks(
            target, 2, dense_edge_targets, torch
        )
        self.assertEqual(int(masks["node_type"].sum()), 3)
        self.assertEqual(int(masks["geometry"].sum()), 4)
        self.assertEqual(int(masks["edge_presence"].sum()), 2)
        self.assertEqual(int(masks["edge_type"].sum()), 1)
        self.assertEqual(int(masks["operation_pointer"].sum()), 3)

    def test_real_quantizer_preassignment_and_ema_contract(self):
        from prototype.flat_baseline.vq import EMAVectorQuantizer

        quantizer = EMAVectorQuantizer(4, 2, 0.25, 0.99, 1e-5)
        values = torch.tensor([[[0.0, 0.0], [1.0, 1.0]]])
        quantizer.eval()
        before = quantizer.embedding.clone()
        with torch.no_grad():
            output = quantizer(values)
        self.assertEqual(output.indices.shape, (1, 2))
        self.assertTrue(torch.equal(before, quantizer.embedding))
        self.assertEqual(int(output.assignment_counts.sum().item()), 2)

    def test_gradient_probe_primitives_do_not_require_optimizer(self):
        parameter = torch.nn.Parameter(torch.tensor([2.0]))
        loss = (parameter * 3.0).sum()
        loss.backward()
        result = aggregate_gradients((
            ("parameter", parameter.grad.detach().tolist()),
        ))
        self.assertEqual(result["nonfinite_gradient_count"], 0)
        self.assertGreater(result["aggregate_norm"], 0.0)


if __name__ == "__main__":
    unittest.main()
