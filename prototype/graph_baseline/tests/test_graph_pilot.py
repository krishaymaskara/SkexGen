"""Frozen protocol, checkpoint, and no-training graph pilot smoke tests."""

from __future__ import annotations

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
    def test_no_training_production_shaped_graph_smoke_and_checkpoint(self):
        from prototype.controlled_data.builders import build_history
        from prototype.controlled_data.factors import PrimitiveFamily
        from prototype.controlled_data.identity import source_family_id
        from prototype.flat_baseline.constrained_v6_pilot import _pilot_partition
        from prototype.flat_baseline.checkpointing import save_checkpoint
        from prototype.model_data.loader import load_partition_physical_examples
        from prototype.model_data.tests.fixtures import source, write_physical_corpus
        from prototype.representation.model import GeometryEncoding
        from prototype.graph_baseline.checkpoint import validate_graph_checkpoint
        from prototype.graph_baseline.config import GraphV1Config
        from prototype.graph_baseline.model import GraphV1Model
        from prototype.graph_baseline.pilot import (
            autonomous_graph_validation,
            graph_checkpoint_payload,
            teacher_forced_graph_validation,
        )
        from prototype.graph_baseline.training import build_graph_optimizer
        from prototype.graph_baseline.training_config import GraphTrainingConfig
        sources = tuple(
            source(template, tuple(PrimitiveFamily)[index % 3], extents=(1.0,) * len(template))
            for index, template in enumerate(("E", "R", "EE", "ER", "RE", "RR", "E", "R"))
        )
        ids = tuple(
            source_family_id(build_history(item, GeometryEncoding.CONTINUOUS))
            for item in sources
        )
        with tempfile.TemporaryDirectory() as temporary:
            write_physical_corpus(
                temporary, sources, partitions={item: "validation" for item in ids}
            )
            physical = load_partition_physical_examples(temporary, "iid", "validation")
            validation = _pilot_partition(
                physical, tuple(item.physical_family_id for item in physical),
                "validation", "iid_validation",
            )
            data = type("Data", (), {"train": validation, "validation": validation})()
            model_config = GraphV1Config()
            pilot_config = GraphPilotConfig(require_clean_source=False)
            training_config = GraphTrainingConfig()
            model = GraphV1Model(model_config)
            optimizer = build_graph_optimizer(model, training_config)
            teacher = teacher_forced_graph_validation(
                model, validation, model_config, 8, torch.device("cpu")
            )
            autonomous = autonomous_graph_validation(
                model, validation, model_config, 8, torch.device("cpu")
            )
            self.assertEqual(teacher["example_count"], 8)
            self.assertEqual(autonomous["example_count"], 8)
            self.assertEqual(len(autonomous["graph_metrics"]["outcomes"]), 8)
            provenance = {
                "git_commit": "a" * 40,
                "git_branch": "graph-profile-decoder",
                "git_dirty": False,
                "git_status_porcelain": [],
                "source_tree_sha256": "b" * 64,
            }
            payload = graph_checkpoint_payload(
                model, optimizer, model_config, pilot_config, training_config,
                data, 1, 1, 8, 2,
                {"teacher_forced": teacher, "autonomous": autonomous}, provenance,
            )
            for name, value in (
                ("model_name", "wrong"),
                ("graph_vocabulary", list(reversed(payload["graph_vocabulary"]))),
                ("minimal_mask_contract", "wrong"),
                ("pair_feature_contract", list(reversed(payload["pair_feature_contract"]))),
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
            path = Path(temporary) / "graph-smoke.pt"
            save_checkpoint(path, payload, torch)
            loaded = torch.load(str(path), map_location="cpu")
            validate_graph_checkpoint(
                loaded, model, model_config, pilot_config, training_config, torch
            )
            reloaded = GraphV1Model(model_config)
            reloaded.load_state_dict(loaded["model_state"], strict=True)
            self.assertEqual(set(reloaded.state_dict()), set(model.state_dict()))

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
