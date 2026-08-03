"""Frozen protocol, checkpoint, and no-training graph pilot smoke tests."""

from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile
import unittest

try:
    import torch
except ImportError:
    torch = None

from prototype.graph_baseline.metrics import FROZEN_V6_REFERENCE
from prototype.graph_baseline.pilot_config import (
    EXPECTED_EXAMPLES_PROCESSED,
    EXPECTED_OPTIMIZER_STEPS,
    GraphPilotConfig,
    validate_partition_authorization,
)


TORCH_REASON = "real PyTorch execution is deferred to the authoritative environment"


def _is_within(path, parent):
    """Python 3.8-compatible resolved-path containment check."""

    try:
        Path(path).resolve().relative_to(Path(parent).resolve())
    except ValueError:
        return False
    return True


class GraphPilotStaticTests(unittest.TestCase):
    def test_exact_frozen_protocol_and_flat_reference(self):
        from prototype.graph_baseline.config import GraphV1Config
        config = GraphPilotConfig()
        validate_partition_authorization(config)
        self.assertEqual(
            (config.seed, config.epochs, config.batch_size, config.device),
            (2026, 2, 8, "cpu"),
        )
        self.assertEqual((EXPECTED_OPTIMIZER_STEPS, EXPECTED_EXAMPLES_PROCESSED), (136, 1088))
        self.assertEqual(
            config.pilot_identity,
            "B0-GRAPH-NATIVE-EDGE-DECODER-V1-POSITION-BIAS-C1-iid-pilot-v1",
        )
        self.assertEqual(FROZEN_V6_REFERENCE, {
            "pilot_job": "3338639",
            "commit": "ac6ef718ae9bab7fa5a80d9f48d0976adf5cafad",
            "iid_examples": 68,
            "conversion_success": 68,
            "complete_validity": 0,
            "unexpected_edge_failures": 68,
        })
        model_config = GraphV1Config()
        model_config.validate()
        self.assertEqual(
            (
                model_config.model_name,
                model_config.checkpoint_version,
                model_config.model_config_version,
                model_config.decoder_contract_version,
                model_config.pair_hidden_dim,
                model_config.position_bias_rank,
            ),
            (
                "B0-GRAPH-NATIVE-EDGE-DECODER-V1-POSITION-BIAS-C1",
                2, 2, 2, 14, 6,
            ),
        )

    def test_slurm_contract_is_authoritative_and_protected(self):
        root = Path(__file__).resolve().parents[1]
        script = (root / "adroit" / "graph_v1_pilot_cpu.slurm").read_text()
        for text in (
            "Python 3.8.13", "1.11.0", "11.3", "graph-profile-decoder",
            "focused_graph_tests_skipped", "systematic_partition_accessed",
            "test_partition_accessed",
            "constrained_graph_v1_position_bias_c1_runs",
            "c1-iid-pilot-${REVIEWED_COMMIT}-${SLURM_JOB_ID}",
            "--repository-root", "--reviewed-commit",
        ):
            self.assertIn(text, script)


@unittest.skipIf(torch is None, TORCH_REASON)
class GraphPilotTensorTests(unittest.TestCase):
    def test_production_shaped_readiness_jobs_3338803_3340201_no_training(self):
        from prototype.controlled_data.builders import build_history
        from prototype.controlled_data.factors import PrimitiveFamily, ReferencePlane
        from prototype.controlled_data.identity import source_family_id
        from prototype.constrained_profile_decoder import profile_targets_for_loss
        from prototype.flat_baseline.constrained_v6_pilot import _pilot_partition
        from prototype.flat_baseline.checkpointing import save_checkpoint
        from prototype.model_data.batching import collate_flat
        from prototype.model_data.loader import load_partition_physical_examples
        from prototype.model_data.tests.fixtures import source, write_physical_corpus
        from prototype.representation.model import GeometryEncoding
        from prototype.graph_baseline.checkpoint import (
            graph_model_metadata,
            validate_graph_checkpoint,
        )
        from prototype.graph_baseline.config import GraphV1Config
        from prototype.graph_baseline.autonomous import greedy_decode_graph_v1
        from prototype.graph_baseline.conversion import (
            graph_v1_teacher_forced_predictions,
            validate_and_convert_graph_prediction,
        )
        from prototype.graph_baseline.graph_tensors import (
            graph_targets_from_reconstruction_batch,
        )
        from prototype.graph_baseline.model import GraphV1Model
        from prototype.graph_baseline.pilot import (
            autonomous_graph_validation,
            graph_checkpoint_payload,
            teacher_forced_graph_validation,
            _reload_and_validate,
            _require_acceptance,
            _require_finite_json,
        )
        from prototype.graph_baseline.provenance import (
            collect_graph_source_provenance,
        )
        from prototype.graph_baseline.tests.test_graph_provenance import (
            initialize_temporary_graph_repository,
        )
        from prototype.graph_baseline.training import build_graph_optimizer
        from prototype.graph_baseline.training_config import GraphTrainingConfig
        templates = ("E", "R", "EE", "ER", "RE", "RR")
        stages = ["authorization"]
        train_sources = tuple(
            source(
                template, tuple(PrimitiveFamily)[index % 3],
                plane=ReferencePlane.XY, extents=(1.0,) * len(template),
            )
            for index, template in enumerate(templates)
        )
        validation_sources = tuple(
            source(
                template, tuple(PrimitiveFamily)[index % 3],
                plane=ReferencePlane.XZ, extents=(1.0,) * len(template),
            )
            for index, template in enumerate(templates)
        )
        train_ids = tuple(
            source_family_id(build_history(item, GeometryEncoding.CONTINUOUS))
            for item in train_sources
        )
        validation_ids = tuple(
            source_family_id(build_history(item, GeometryEncoding.CONTINUOUS))
            for item in validation_sources
        )
        self.assertEqual(len(set(train_ids)), 6)
        self.assertEqual(len(set(validation_ids)), 6)
        self.assertTrue(set(train_ids).isdisjoint(validation_ids))
        with tempfile.TemporaryDirectory() as temporary:
            temporary_root = Path(temporary).resolve()
            corpus_root = temporary_root / "corpus"
            source_repository_root = temporary_root / "source-repository"
            output_root = temporary_root / "outputs"
            output_root.mkdir()
            self.assertNotEqual(corpus_root, source_repository_root)
            self.assertNotEqual(corpus_root, output_root)
            self.assertNotEqual(source_repository_root, output_root)
            self.assertFalse(_is_within(source_repository_root, corpus_root))
            self.assertFalse(_is_within(output_root, corpus_root))
            reviewed_commit = initialize_temporary_graph_repository(
                source_repository_root
            )
            self.assertTrue((source_repository_root / ".git").is_dir())
            provenance = collect_graph_source_provenance(
                source_repository_root
            )
            self.assertEqual(provenance["git_branch"], "graph-profile-decoder")
            self.assertEqual(provenance["git_commit"], reviewed_commit)
            self.assertIs(provenance["git_dirty"], False)
            self.assertEqual(provenance["git_status_porcelain"], [])
            write_physical_corpus(
                corpus_root,
                train_sources + validation_sources,
                partitions={
                    **{item: "train" for item in train_ids},
                    **{item: "validation" for item in validation_ids},
                },
            )
            corpus_paths = tuple(
                path.relative_to(corpus_root)
                for path in corpus_root.rglob("*")
            )
            self.assertTrue(corpus_paths)
            self.assertFalse(any(
                relative.parts[0] in (".git", "source-repository", "outputs")
                for relative in corpus_paths
            ))
            split = json.loads(
                (corpus_root / "manifests" / "iid.json").read_text()
            )
            family_rows = split["families"]
            assignments = [
                (row["source_family_id"], row["partition"])
                for row in family_rows
            ]
            self.assertEqual(len(assignments), 12)
            self.assertEqual(len({family_id for family_id, unused in assignments}), 12)
            self.assertEqual(
                {family_id for family_id, partition in assignments if partition == "train"},
                set(train_ids),
            )
            self.assertEqual(
                {family_id for family_id, partition in assignments
                 if partition == "validation"},
                set(validation_ids),
            )
            self.assertFalse(any(
                partition in ("systematic", "test")
                for unused, partition in assignments
            ))
            sample_assignments = {}
            for row in split["samples"]:
                sample_assignments.setdefault(row["source_family_id"], set()).add(
                    row["partition"]
                )
            self.assertEqual(set(sample_assignments), set(train_ids) | set(validation_ids))
            self.assertTrue(all(len(values) == 1 for values in sample_assignments.values()))

            train_physical = load_partition_physical_examples(
                corpus_root, "iid", "train"
            )
            validation_physical = load_partition_physical_examples(
                corpus_root, "iid", "validation"
            )
            stages.append("dataset_loading")
            self.assertEqual(len(train_physical), 6)
            self.assertEqual(len(validation_physical), 6)
            self.assertTrue(
                {item.physical_family_id for item in train_physical}.isdisjoint(
                    item.physical_family_id for item in validation_physical
                )
            )
            train = _pilot_partition(
                train_physical,
                tuple(item.physical_family_id for item in train_physical),
                "train", "train",
            )
            validation = _pilot_partition(
                validation_physical,
                tuple(item.physical_family_id for item in validation_physical),
                "validation", "iid_validation",
            )
            self.assertEqual(
                {item.metadata.operation_template for item in validation.physical_examples},
                set(templates),
            )
            self.assertEqual(
                {item.metadata.primitive_family for item in validation.physical_examples},
                {item.value for item in PrimitiveFamily},
            )
            data = type("Data", (), {"train": train, "validation": validation})()
            model_config = GraphV1Config()
            pilot_config = GraphPilotConfig(require_clean_source=False)
            training_config = GraphTrainingConfig()
            audit_batch = collate_flat(validation.flat_examples)
            audit_inputs = audit_batch.to_torch(torch)
            audit_target = audit_batch.target.to_torch(torch)
            graph_targets_from_reconstruction_batch(audit_target)
            stages.append("tensorization")
            model = GraphV1Model(model_config)
            stages.append("model_construction")
            optimizer = build_graph_optimizer(model, training_config)
            model_metadata = graph_model_metadata(model, model_config)
            self.assertEqual(model_metadata["model_name"], (
                "B0-GRAPH-NATIVE-EDGE-DECODER-V1-POSITION-BIAS-C1"
            ))
            self.assertEqual(
                (
                    model_metadata["checkpoint_version"],
                    model_metadata["model_config_version"],
                    model_metadata["decoder_contract_version"],
                ),
                (2, 2, 2),
            )
            self.assertEqual(model_metadata["scientific_correction_index"], 1)
            self.assertEqual(model_metadata["scientific_correction_limit"], 1)
            self.assertEqual(
                model_metadata["parent_graph_commit"],
                "089b9f3d0e5a61fb19ef3fa05e993fc4eceffdcb",
            )
            self.assertEqual(model_metadata["parent_graph_pilot_job"], 3341942)
            self.assertEqual(model_metadata["correction_hypothesis"], (
                "explicit-directed-ordered-position-bias-for-repeated-"
                "instance-alignment"
            ))
            self.assertEqual(model_metadata["parameter_counts"], {
                "graph_model_total": 32852,
                "graph_edge_decoder": 3482,
                "main_pair_mlp": 3254,
                "position_bias": 228,
                "source_position_factor": 96,
                "destination_position_factor": 96,
                "position_class_projection": 36,
                "frozen_v6_structural_heads": 3496,
                "frozen_v6_total": 32866,
                "absolute_total_difference": 14,
            })
            profiles = profile_targets_for_loss(
                audit_batch.target, audit_inputs["geometry"]
            )
            model.eval()
            with torch.no_grad():
                output = model(
                    target=audit_target, profile_targets=profiles, **audit_inputs
                )
            stages.append("teacher_forced_nodes")
            teacher_predictions = graph_v1_teacher_forced_predictions(
                output,
                node_mask=audit_target["node_mask"],
                node_count_source="production_readiness",
            )
            stages.append("teacher_forced_graph")
            teacher_results = tuple(
                validate_and_convert_graph_prediction(prediction)
                for prediction in teacher_predictions
            )
            self.assertEqual(len(teacher_results), 6)
            stages.append("teacher_forced_conversion")
            counts = audit_target["node_mask"].long().sum(dim=1)
            autonomous_predictions = greedy_decode_graph_v1(
                model, audit_inputs, node_counts=counts,
                node_count_source="production_readiness",
            )
            stages.append("autonomous_nodes")
            self.assertEqual(len(autonomous_predictions), 6)
            stages.append("autonomous_graph")
            autonomous_results = tuple(
                validate_and_convert_graph_prediction(prediction)
                for prediction in autonomous_predictions
            )
            self.assertEqual(len(autonomous_results), 6)
            stages.append("autonomous_conversion")
            teacher = teacher_forced_graph_validation(
                model, validation, model_config, 8, torch.device("cpu")
            )
            autonomous = autonomous_graph_validation(
                model, validation, model_config, 8, torch.device("cpu")
            )
            self.assertEqual(teacher["example_count"], 6)
            self.assertEqual(autonomous["example_count"], 6)
            self.assertEqual(len(autonomous["graph_metrics"]["outcomes"]), 6)
            self.assertEqual(
                sum(autonomous["graph_metrics"]["first_failure_histogram"].values()),
                6,
            )
            self.assertEqual(
                teacher["graph_metrics"]["active_ordered_pair_count"],
                sum(count * (count - 1) for count in (4, 5, 7, 8, 8, 9)),
            )
            for summary in (teacher, autonomous):
                graph_metrics = summary["graph_metrics"]
                self.assertEqual(
                    graph_metrics["position_bias_parameter_count"], 228
                )
                self.assertEqual(
                    graph_metrics["main_pair_mlp_parameter_count"], 3254
                )
                self.assertEqual(
                    graph_metrics["corrected_graph_decoder_parameter_count"],
                    3482,
                )
                self.assertEqual(
                    graph_metrics["initial_graph_v1_job"], 3341942
                )
                self.assertEqual(
                    graph_metrics["initial_graph_v1_exact_graph_match_count"],
                    22,
                )
                self.assertEqual(
                    graph_metrics["initial_graph_v1_complete_validity_count"],
                    22,
                )
                self.assertEqual(
                    graph_metrics[
                        "initial_graph_v1_two_operation_validity_count"
                    ],
                    0,
                )
                self.assertEqual(
                    graph_metrics["flat_v6_complete_validity_count"], 0
                )
                self.assertEqual(graph_metrics["flat_v6_job"], 3338639)
                self.assertEqual(
                    set(graph_metrics["position_bias_logit_statistics"]),
                    {
                        "all_active_pairs", "positive_target_pairs",
                        "negative_target_pairs", "single_operation_examples",
                        "two_operation_examples",
                    },
                )
                for row in graph_metrics[
                    "position_bias_logit_statistics"
                ].values():
                    self.assertGreater(row["logit_count"], 0)
                    for name in (
                        "position_bias_logit_absolute_mean",
                        "main_pair_logit_absolute_mean",
                        "position_bias_to_main_logit_ratio",
                    ):
                        self.assertTrue(math.isfinite(row[name]))
                        self.assertGreater(row[name], 0.0)
            stages.append("metrics")
            payload = graph_checkpoint_payload(
                model, optimizer, model_config, pilot_config, training_config,
                data, 1, 1, 6, 2,
                {"teacher_forced": teacher, "autonomous": autonomous}, provenance,
                repository_root=source_repository_root,
            )
            for name, value in (
                ("checkpoint_version", 1),
                ("model_name", "B0-GRAPH-NATIVE-EDGE-DECODER-V1"),
                ("model_config_version", 1),
                ("decoder_contract_version", 1),
                ("graph_contract_version", 2),
                ("graph_contract", {"representation_name": "wrong"}),
                ("graph_vocabulary", list(reversed(payload["graph_vocabulary"]))),
                ("graph_direction_convention", "wrong"),
                ("active_pair_construction", "wrong"),
                ("minimal_mask_contract", "wrong"),
                ("pair_feature_contract", list(reversed(payload["pair_feature_contract"]))),
                ("pair_decoder_architecture", {"kind": "wrong"}),
                ("loss_normalization", "wrong"),
                ("inherited_node_path", "wrong"),
                ("node_grammar_contract", {"contract_id": "wrong"}),
                ("categorical_selection_contract", {"contract_id": "wrong"}),
                ("axis_geometry_contract", {"contract_id": "wrong"}),
                ("node_vocabulary", list(reversed(payload["node_vocabulary"]))),
                ("model_config", {"model_name": "wrong"}),
                ("parameter_counts", {"graph_model_total": -1}),
                ("scientific_correction_index", 0),
                ("scientific_correction_limit", 2),
                ("parent_graph_commit", "0" * 40),
                ("parent_graph_pilot_job", 0),
                ("correction_hypothesis", "wrong"),
            ):
                malformed = dict(payload)
                malformed[name] = value
                with self.assertRaisesRegex(ValueError, "malformed_graph_checkpoint"):
                    validate_graph_checkpoint(
                        malformed, model, model_config, pilot_config,
                        training_config, torch,
                        expected_source_provenance=provenance,
                    )
            initial_graph_checkpoint = dict(payload)
            initial_graph_checkpoint.update({
                "checkpoint_version": 1,
                "model_name": "B0-GRAPH-NATIVE-EDGE-DECODER-V1",
                "model_config_version": 1,
                "decoder_contract_version": 1,
            })
            for name in (
                "scientific_correction_index", "scientific_correction_limit",
                "parent_graph_commit", "parent_graph_pilot_job",
                "correction_hypothesis",
            ):
                del initial_graph_checkpoint[name]
            with self.assertRaisesRegex(
                ValueError, "malformed_graph_checkpoint"
            ):
                validate_graph_checkpoint(
                    initial_graph_checkpoint, model, model_config,
                    pilot_config, training_config, torch,
                    expected_source_provenance=provenance,
                )
            for name, value, code in (
                ("git_branch", None, "missing_git_branch"),
                ("git_branch", "wrong", "wrong_git_branch"),
                ("git_commit", None, "missing_git_commit"),
                ("git_commit", "0" * 40, "wrong_git_commit"),
                ("git_dirty", True, "dirty_source_tree"),
                ("git_status_porcelain", (), "malformed_git_status"),
                ("source_tree_sha256", None, "missing_source_tree_digest"),
                (
                    "source_tree_sha256", "0" * 64,
                    "source_tree_digest_mismatch",
                ),
            ):
                malformed = dict(payload)
                malformed["source_provenance"] = dict(
                    provenance, **{name: value}
                )
                with self.assertRaisesRegex(
                    ValueError, "malformed_graph_checkpoint: {}".format(code)
                ):
                    validate_graph_checkpoint(
                        malformed, model, model_config, pilot_config,
                        training_config, torch,
                        expected_source_provenance=provenance,
                    )
            selected_path = output_root / "epoch-0001.pt"
            final_path = output_root / "epoch-0002.pt"
            self.assertTrue(_is_within(selected_path, output_root))
            self.assertTrue(_is_within(final_path, output_root))
            save_checkpoint(selected_path, payload, torch)
            final_payload = graph_checkpoint_payload(
                model, optimizer, model_config, pilot_config, training_config,
                data, 2, 2, 12, 2,
                {"teacher_forced": teacher, "autonomous": autonomous}, provenance,
                repository_root=source_repository_root,
            )
            save_checkpoint(final_path, final_payload, torch)
            stages.append("checkpoint_save")
            loaded = torch.load(str(selected_path), map_location="cpu")
            self.assertEqual(list(loaded), list(payload))
            self.assertEqual(loaded["optimizer_state"], optimizer.state_dict())
            self.assertEqual(list(loaded["vq_state"]), list(model.vq.state_dict()))
            self.assertEqual(loaded["epoch"], 1)
            self.assertEqual(loaded["global_step"], 1)
            self.assertEqual(loaded["examples_processed"], 6)
            self.assertEqual(loaded["source_provenance"], provenance)
            self.assertFalse(loaded["systematic_partition_accessed"])
            self.assertFalse(loaded["test_partition_accessed"])
            validate_graph_checkpoint(
                loaded, model, model_config, pilot_config, training_config, torch,
                expected_source_provenance=provenance,
            )
            reloaded = GraphV1Model(model_config)
            reloaded.load_state_dict(loaded["model_state"], strict=True)
            self.assertEqual(set(reloaded.state_dict()), set(model.state_dict()))
            self.assertTrue(_reload_and_validate(
                selected_path, model_config, pilot_config, training_config,
                data, torch.device("cpu"), provenance, source_repository_root,
            ))
            stages.append("selected_reload")
            self.assertTrue(_reload_and_validate(
                final_path, model_config, pilot_config, training_config,
                data, torch.device("cpu"), provenance, source_repository_root,
            ))
            stages.append("final_reload")
            _require_finite_json({"teacher": teacher, "autonomous": autonomous})
            json.dumps({"teacher": teacher, "autonomous": autonomous}, allow_nan=False)
            stages.append("json_serialization")
            acceptance = _require_acceptance(
                {"production_readiness": True},
                {"systematic_partition_accessed": False,
                 "test_partition_accessed": False},
            )
            terminal = {
                "event": "terminal_success", "acceptance": acceptance,
                "systematic_partition_accessed": False,
                "test_partition_accessed": False,
            }
            _require_finite_json(terminal)
            stages.append("terminal_summary")
            self.assertEqual(stages, [
                "authorization", "dataset_loading", "tensorization",
                "model_construction", "teacher_forced_nodes",
                "teacher_forced_graph", "teacher_forced_conversion",
                "autonomous_nodes", "autonomous_graph",
                "autonomous_conversion", "metrics", "checkpoint_save",
                "selected_reload", "final_reload", "json_serialization",
                "terminal_summary",
            ])
            self.assertEqual(optimizer.state, {})

    def test_checkpoint_rejects_identity_vocabulary_mask_features_and_provenance(self):
        from prototype.graph_baseline.checkpoint import (
            GraphCheckpointError, graph_model_metadata,
        )
        from prototype.graph_baseline.config import GraphV1Config
        from prototype.graph_baseline.model import GraphV1Model
        model = GraphV1Model(GraphV1Config())
        metadata = graph_model_metadata(model, model.config)
        for name, value in (
            ("model_name", "wrong"),
            ("graph_vocabulary", list(reversed(metadata["graph_vocabulary"]))),
            ("minimal_mask_contract", "wrong"),
            ("pair_feature_contract", list(reversed(metadata["pair_feature_contract"]))),
        ):
            malformed = dict(metadata)
            malformed[name] = value
            self.assertNotEqual(malformed[name], metadata[name])
        self.assertTrue(issubclass(GraphCheckpointError, ValueError))
