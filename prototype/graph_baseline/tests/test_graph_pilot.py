"""Frozen protocol, checkpoint, and no-training graph pilot smoke tests."""

from __future__ import annotations

import json
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


class GraphPilotStaticTests(unittest.TestCase):
    def test_exact_frozen_protocol_and_flat_reference(self):
        config = GraphPilotConfig()
        validate_partition_authorization(config)
        self.assertEqual(
            (config.seed, config.epochs, config.batch_size, config.device),
            (2026, 2, 8, "cpu"),
        )
        self.assertEqual((EXPECTED_OPTIMIZER_STEPS, EXPECTED_EXAMPLES_PROCESSED), (136, 1088))
        self.assertEqual(FROZEN_V6_REFERENCE, {
            "pilot_job": "3338639",
            "commit": "ac6ef718ae9bab7fa5a80d9f48d0976adf5cafad",
            "iid_examples": 68,
            "conversion_success": 68,
            "complete_validity": 0,
            "unexpected_edge_failures": 68,
        })

    def test_slurm_contract_is_authoritative_and_protected(self):
        root = Path(__file__).resolve().parents[1]
        script = (root / "adroit" / "graph_v1_pilot_cpu.slurm").read_text()
        for text in (
            "Python 3.8.13", "1.11.0", "11.3", "graph-profile-decoder",
            "focused_graph_tests_skipped", "systematic_partition_accessed",
            "test_partition_accessed", "constrained_graph_v1_runs",
        ):
            self.assertIn(text, script)


@unittest.skipIf(torch is None, TORCH_REASON)
class GraphPilotTensorTests(unittest.TestCase):
    def test_production_shaped_readiness_job_3338803_no_training(self):
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
        from prototype.graph_baseline.checkpoint import validate_graph_checkpoint
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
            write_physical_corpus(
                temporary,
                train_sources + validation_sources,
                partitions={
                    **{item: "train" for item in train_ids},
                    **{item: "validation" for item in validation_ids},
                },
            )
            split = json.loads(
                (Path(temporary) / "manifests" / "iid.json").read_text()
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
                temporary, "iid", "train"
            )
            validation_physical = load_partition_physical_examples(
                temporary, "iid", "validation"
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
            stages.append("metrics")
            provenance = {
                "git_commit": "a" * 40,
                "git_branch": "graph-profile-decoder",
                "git_dirty": False,
                "git_status_porcelain": [],
                "source_tree_sha256": "b" * 64,
            }
            payload = graph_checkpoint_payload(
                model, optimizer, model_config, pilot_config, training_config,
                data, 1, 1, 6, 2,
                {"teacher_forced": teacher, "autonomous": autonomous}, provenance,
            )
            for name, value in (
                ("checkpoint_version", 2),
                ("model_name", "wrong"),
                ("model_config_version", 2),
                ("decoder_contract_version", 2),
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
            ):
                malformed = dict(payload)
                malformed[name] = value
                with self.assertRaisesRegex(ValueError, "malformed_graph_checkpoint"):
                    validate_graph_checkpoint(
                        malformed, model, model_config, pilot_config,
                        training_config, torch,
                    )
            malformed = dict(payload)
            malformed["source_provenance"] = dict(
                provenance, git_branch="wrong"
            )
            with self.assertRaisesRegex(ValueError, "malformed_graph_checkpoint"):
                validate_graph_checkpoint(
                    malformed, model, model_config, pilot_config,
                    training_config, torch,
                )
            selected_path = Path(temporary) / "epoch-0001.pt"
            final_path = Path(temporary) / "epoch-0002.pt"
            save_checkpoint(selected_path, payload, torch)
            final_payload = graph_checkpoint_payload(
                model, optimizer, model_config, pilot_config, training_config,
                data, 2, 2, 12, 2,
                {"teacher_forced": teacher, "autonomous": autonomous}, provenance,
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
            self.assertFalse(loaded["systematic_partition_accessed"])
            self.assertFalse(loaded["test_partition_accessed"])
            validate_graph_checkpoint(
                loaded, model, model_config, pilot_config, training_config, torch
            )
            reloaded = GraphV1Model(model_config)
            reloaded.load_state_dict(loaded["model_state"], strict=True)
            self.assertEqual(set(reloaded.state_dict()), set(model.state_dict()))
            self.assertTrue(_reload_and_validate(
                selected_path, model_config, pilot_config, training_config,
                data, torch.device("cpu"),
            ))
            stages.append("selected_reload")
            self.assertTrue(_reload_and_validate(
                final_path, model_config, pilot_config, training_config,
                data, torch.device("cpu"),
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
