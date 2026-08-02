"""V6 fixed-pilot identity and no-training smoke tests."""

from __future__ import annotations

from dataclasses import replace
import ast
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

try:
    import torch
except ImportError:
    torch = None

from prototype.flat_baseline.constrained_v6_pilot_config import (
    PILOT_EXPECTED_EXAMPLES_PROCESSED,
    PILOT_EXPECTED_OPTIMIZER_STEPS,
    PILOT_EXPECTED_TRAIN_EXAMPLES,
    PILOT_EXPECTED_VALIDATION_EXAMPLES,
    PILOT_IDENTITY,
    ConstrainedV6PilotConfig,
    ConstrainedV6PilotError,
    validate_pilot_partition_authorization,
)


TORCH_REASON = "real PyTorch execution is deferred to the authoritative environment"


class V6PilotStaticTests(unittest.TestCase):
    def test_exact_protocol_and_protected_boundaries(self):
        config = ConstrainedV6PilotConfig()
        config.validate()
        self.assertIn("PREFIX-GRAMMAR-CONSTRAINED-AXIS-v6", PILOT_IDENTITY)
        self.assertEqual((config.seed, config.epochs, config.batch_size), (2026, 2, 8))
        self.assertEqual(
            (PILOT_EXPECTED_TRAIN_EXAMPLES, PILOT_EXPECTED_VALIDATION_EXAMPLES,
             PILOT_EXPECTED_OPTIMIZER_STEPS, PILOT_EXPECTED_EXAMPLES_PROCESSED),
            (544, 68, 136, 1088),
        )
        self.assertFalse(config.systematic_partition_accessed)
        self.assertFalse(config.test_partition_accessed)
        for changes in (
            {"systematic_partition_accessed": True},
            {"test_partition_accessed": True},
            {"validation_partition": "systematic"},
            {"epochs": 3},
            {"batch_size": 4},
        ):
            with self.assertRaises(ConstrainedV6PilotError):
                validate_pilot_partition_authorization(replace(config, **changes))

    def test_python38_and_slurm_contract(self):
        root = Path(__file__).resolve().parents[1]
        for name in (
            "constrained_v6_training.py",
            "constrained_v6_pilot.py",
            "constrained_v6_pilot_config.py",
            "run_constrained_v6_pilot.py",
        ):
            path = root / name
            ast.parse(path.read_text(), str(path), feature_version=(3, 8))
        slurm = (root / "adroit" / "constrained_v6_pilot_cpu.slurm").read_text()
        self.assertIn('torch.__version__.split("+")[0] == "1.11.0"', slurm)
        self.assertIn('torch.version.cuda == "11.3"', slurm)
        self.assertIn("focused_v6_tests_skipped", slurm)
        self.assertIn("constrained_v6_runs", slurm)

    def test_authoritative_checkpoint_schema_and_canonical_grammar(self):
        from prototype.flat_baseline.constrained_v6_checkpoint import (
            V6_PILOT_CHECKPOINT_FIELDS,
            V6_TINY_CHECKPOINT_FIELDS,
            canonical_grammar_metadata,
            required_model_metadata_fields,
            required_pilot_metadata_fields,
            required_training_state_fields,
            required_top_level_fields,
        )
        from prototype.flat_baseline.provenance import source_state
        grammar = canonical_grammar_metadata()
        self.assertEqual(
            set(grammar),
            {
                "contract_id", "node_vocabulary", "start_node",
                "transitions", "terminal_nodes", "valid_requested_node_counts",
                "operation_limit", "completion_algorithm_id",
                "canonical_templates",
            },
        )
        self.assertEqual(grammar["valid_requested_node_counts"], [4, 5, 7, 8, 9])
        self.assertEqual(grammar["operation_limit"], 2)
        self.assertEqual(grammar["start_node"], "reference_plane")
        self.assertEqual(grammar["terminal_nodes"], ["extrude", "revolve"])
        self.assertEqual(
            grammar["completion_algorithm_id"],
            "exhaustive-finite-prefix-completion-v1",
        )
        self.assertEqual(
            required_top_level_fields("tiny_overfit"),
            V6_TINY_CHECKPOINT_FIELDS,
        )
        self.assertEqual(
            required_top_level_fields("pilot_fixed_epoch"),
            V6_PILOT_CHECKPOINT_FIELDS,
        )
        self.assertFalse(
            set(required_model_metadata_fields())
            & set(required_pilot_metadata_fields())
        )
        self.assertTrue(
            set(required_training_state_fields()) <= V6_PILOT_CHECKPOINT_FIELDS
        )
        self.assertTrue({
            "axis_geometry_contract_id",
            "axis_geometry_contract",
            "axis_construction_algorithm_id",
            "axis_geometry_channel_indices",
            "axis_coordinate_space",
            "axis_raw_constrained_evidence_policy",
        } <= V6_PILOT_CHECKPOINT_FIELDS)
        self.assertIsInstance(source_state()["git_status_porcelain"], list)


@unittest.skipIf(torch is None, TORCH_REASON)
class V6PilotTensorTests(unittest.TestCase):
    @staticmethod
    def _checkpoint_fixture():
        from prototype.flat_baseline.constrained_v6 import ConstrainedProfileV6Model
        from prototype.flat_baseline.constrained_v6_config import ConstrainedProfileV6Config
        from prototype.flat_baseline.constrained_v6_pilot import (
            pilot_optimization_config,
        )
        from prototype.flat_baseline.constrained_v6_training import (
            build_v6_optimizer,
        )

        pilot_config = ConstrainedV6PilotConfig(require_clean_source=False)
        model_config = ConstrainedProfileV6Config()
        optimization = pilot_optimization_config(pilot_config, 2)
        model = ConstrainedProfileV6Model(model_config)
        optimizer = build_v6_optimizer(model, optimization)

        def partition_metadata(partition, identity):
            return {
                "split_name": "iid",
                "partition": partition,
                "partition_identity": identity,
                "family_count": 8,
                "family_ids": ["synthetic-{}".format(index) for index in range(8)],
                "corpus_fingerprint": "c" * 64,
            }

        data = SimpleNamespace(
            train=SimpleNamespace(
                flat_examples=(None,) * 8,
                metadata=lambda: partition_metadata("train", "train"),
            ),
            validation=SimpleNamespace(
                flat_examples=(None,) * 8,
                metadata=lambda: partition_metadata(
                    "validation", "iid_validation"
                ),
            ),
        )
        provenance = {
            "git_commit": "a" * 40,
            "git_dirty": False,
            "git_status_porcelain": [],
            "source_tree_sha256": "b" * 64,
        }
        return (
            model, optimizer, model_config, optimization,
            pilot_config, data, provenance,
        )

    def _pilot_payload(self, epoch=1):
        from prototype.flat_baseline.constrained_v6_pilot import (
            pilot_checkpoint_payload,
        )
        fixture = self._checkpoint_fixture()
        model, optimizer, model_config, optimization, pilot_config, data, provenance = fixture
        summary = {"example_count": 8, "total_loss": 1.0 / epoch}
        payload = pilot_checkpoint_payload(
            model,
            optimizer,
            model_config,
            optimization,
            pilot_config,
            data,
            epoch=epoch,
            global_step=epoch,
            examples_processed=epoch * 8,
            configured_maximum_steps=2,
            teacher_summary=summary,
            autonomous_summary={"example_count": 8, "outcomes": []},
            source_provenance=provenance,
        )
        return fixture, payload

    def test_acceptance_includes_positive_node_grammar_predicate(self):
        from prototype.flat_baseline.constrained_v6_pilot import _require_pilot_acceptance
        scientific = {
            "configured_budget_completed": True,
            "constrained_categorical_contract_satisfied": True,
            "constrained_node_grammar_contract_satisfied": True,
            "constrained_axis_geometry_contract_satisfied": True,
        }
        result = _require_pilot_acceptance(scientific, {
            "systematic_partition_accessed": False,
            "test_partition_accessed": False,
        })
        self.assertTrue(all(result.values()))
        scientific["constrained_node_grammar_contract_satisfied"] = False
        with self.assertRaises(ConstrainedV6PilotError):
            _require_pilot_acceptance(scientific, {
                "systematic_partition_accessed": False,
                "test_partition_accessed": False,
            })
        scientific["constrained_node_grammar_contract_satisfied"] = True
        scientific["constrained_axis_geometry_contract_satisfied"] = False
        with self.assertRaises(ConstrainedV6PilotError):
            _require_pilot_acceptance(scientific, {
                "systematic_partition_accessed": False,
                "test_partition_accessed": False,
            })

    def test_job_3338085_no_training_production_shaped_v6_pilot_smoke(self):
        from prototype.constrained_profile_decoder import profile_targets_for_loss
        from prototype.controlled_data.builders import build_history
        from prototype.controlled_data.factors import PrimitiveFamily
        from prototype.controlled_data.identity import source_family_id
        from prototype.flat_baseline.constrained_v6 import ConstrainedProfileV6Model
        from prototype.flat_baseline.constrained_v6_config import ConstrainedProfileV6Config
        from prototype.flat_baseline.constrained_v6_conversion import (
            v6_teacher_forced_predictions,
        )
        from prototype.flat_baseline.constrained_v6_pilot import (
            _pilot_partition,
            _require_finite_json,
            autonomous_validation,
            teacher_forced_validation,
        )
        from prototype.model_data.loader import load_partition_physical_examples
        from prototype.model_data.batching import collate_flat
        from prototype.model_data.tests.fixtures import source, write_physical_corpus
        from prototype.model_data.vocab import NODE_TYPES
        from prototype.node_grammar import legal_next_node_ids
        from prototype.representation.model import GeometryEncoding

        validate_pilot_partition_authorization(ConstrainedV6PilotConfig())
        sources = (
            source("E", PrimitiveFamily.CIRCLE, extents=(1.0,)),
            source("R", PrimitiveFamily.RECTANGLE_LINES, extents=(1.0,)),
            source("EE", PrimitiveFamily.CAPSULE_LINE_ARC, extents=(1.0, 2.0)),
            source("ER", PrimitiveFamily.CIRCLE, extents=(2.0, 3.0)),
            source("RE", PrimitiveFamily.RECTANGLE_LINES, extents=(3.0, 1.0)),
            source("RR", PrimitiveFamily.CAPSULE_LINE_ARC, extents=(2.0, 1.0)),
            source("E", PrimitiveFamily.RECTANGLE_LINES, extents=(2.0,)),
            source("R", PrimitiveFamily.CIRCLE, extents=(2.0,)),
        )
        family_ids = tuple(
            source_family_id(build_history(item, GeometryEncoding.CONTINUOUS))
            for item in sources
        )
        with tempfile.TemporaryDirectory() as temporary:
            write_physical_corpus(
                temporary,
                sources,
                partitions={item: "validation" for item in family_ids},
            )
            physical = load_partition_physical_examples(
                temporary, "iid", "validation"
            )
            partition = _pilot_partition(
                physical,
                tuple(item.physical_family_id for item in physical),
                "validation",
                "iid_validation",
            )
            model_config = ConstrainedProfileV6Config()
            model = ConstrainedProfileV6Model(model_config)
            with torch.no_grad():
                model.node_type_head.weight.zero_()
                model.node_type_head.bias.zero_()
                model.node_type_head.bias[NODE_TYPES.id(None)] = 10.0
                batch = collate_flat(partition.flat_examples)
                inputs = batch.to_torch(torch)
                target = batch.target.to_torch(torch)
                profiles = profile_targets_for_loss(
                    batch.target, inputs["geometry"]
                )
                output = model(
                    target=target, profile_targets=profiles, **inputs
                )
                teacher_predictions = v6_teacher_forced_predictions(
                    output,
                    node_mask=target["node_mask"],
                    node_count_source="independent_smoke_enumeration",
                )
            axis_id = NODE_TYPES.id("axis")
            none_id = NODE_TYPES.id(None)
            independent_teacher_axis_count = 0
            independent_autonomous_axis_count = 0
            enumeration = []
            for example_index, prediction in enumerate(teacher_predictions):
                count = int(target["node_mask"][example_index].sum().item())
                authoritative = tuple(
                    int(value) for value in target["node_type_ids"][
                        example_index, :count
                    ].tolist()
                )
                independently_selected = []
                for position in range(count):
                    prefix = authoritative[:position]
                    legal = legal_next_node_ids(prefix, count)
                    selected = min(legal)
                    independently_selected.append(selected)
                    enumeration.append({
                        "example_index": example_index,
                        "requested_node_count": count,
                        "authoritative_shifted_prefix": prefix,
                        "raw_current_node_argmax": none_id,
                        "grammar_constrained_current_node": selected,
                        "axis_applicable": selected == axis_id,
                    })
                self.assertEqual(
                    prediction.raw_node_type_argmax_ids,
                    (none_id,) * count,
                )
                self.assertEqual(
                    prediction.grammar_constrained_node_type_ids,
                    tuple(independently_selected),
                )
                independent_teacher_axis_count += sum(
                    selected == axis_id for selected in independently_selected
                )

                generated_prefix = []
                for _ in range(count):
                    selected = min(legal_next_node_ids(
                        tuple(generated_prefix), count
                    ))
                    generated_prefix.append(selected)
                independent_autonomous_axis_count += sum(
                    selected == axis_id for selected in generated_prefix
                )
            self.assertEqual(len(enumeration), sum(
                int(row.sum().item()) for row in target["node_mask"]
            ))
            teacher = teacher_forced_validation(
                model, partition, model_config, 8, torch.device("cpu")
            )
            autonomous = autonomous_validation(
                model, partition, model_config, 8, torch.device("cpu")
            )
        self.assertEqual(teacher["example_count"], 8)
        self.assertEqual(autonomous["example_count"], 8)
        grammar = autonomous["node_grammar_selection_metrics"]
        self.assertEqual(grammar["constrained_grammar_violation_count"], 0)
        self.assertEqual(autonomous["constrained_grammar_valid_sequence_count"], 8)
        self.assertEqual(len(autonomous["outcomes"]), 8)
        teacher_axes = teacher["axis_geometry_metrics"]
        autonomous_axes = autonomous["axis_geometry_metrics"]
        self.assertEqual(independent_teacher_axis_count, 7)
        self.assertEqual(independent_autonomous_axis_count, 6)
        self.assertEqual(
            teacher_axes["applicable_axis_node_count"],
            independent_teacher_axis_count,
        )
        self.assertEqual(
            autonomous_axes["applicable_axis_node_count"],
            independent_autonomous_axis_count,
        )
        self.assertEqual(teacher_axes["constrained_invalid_axis_count"], 0)
        self.assertEqual(autonomous_axes["constrained_invalid_axis_count"], 0)
        self.assertEqual(
            autonomous_axes["constrained_valid_axis_count"], 6
        )
        _require_finite_json({"teacher": teacher, "autonomous": autonomous})
        json.dumps(
            {"teacher": teacher, "autonomous": autonomous},
            sort_keys=True,
            allow_nan=False,
        )

    def test_job_3337640_v6_pilot_checkpoint_contract_regression(self):
        from prototype.flat_baseline.constrained_v6 import ConstrainedProfileV6Model
        from prototype.flat_baseline.constrained_v6_checkpoint import (
            V6_PILOT_CHECKPOINT_FIELDS,
        )
        from prototype.flat_baseline.constrained_v6_pilot import (
            load_pilot_checkpoint,
        )
        from prototype.flat_baseline.constrained_v6_training import (
            build_v6_optimizer,
            save_v6_checkpoint,
        )

        fixture, epoch_one = self._pilot_payload(1)
        model, _, model_config, optimization, pilot_config, data, _ = fixture
        validate_pilot_partition_authorization(pilot_config)
        _, epoch_two = self._pilot_payload(2)
        independently_required = {
            "checkpoint_version", "checkpoint_kind", "model_name",
            "model_config_version", "decoder_contract_version",
            "learned_geometry_channel_indices", "canonical_plane_contract_id",
            "base_geometry_contract_id", "categorical_selection_contract_id",
            "categorical_selection_contract", "node_grammar_contract_id",
            "node_grammar_contract", "node_vocabulary",
            "valid_requested_node_counts", "operation_limit",
            "completion_algorithm_id", "axis_geometry_contract_id",
            "axis_geometry_contract", "axis_construction_algorithm_id",
            "axis_geometry_channel_indices", "axis_coordinate_space",
            "axis_raw_constrained_evidence_policy",
            "pilot_identity", "pilot_config",
            "model_config", "optimization_config", "model_state",
            "optimizer_state", "vq_state", "epoch", "global_step",
            "examples_processed", "configured_maximum_steps",
            "completed_epochs", "training_partition_state",
            "validation_partition_state", "training_sampler_state",
            "validation_cadence", "validation_summaries", "rng_state",
            "source_provenance", "systematic_partition_accessed",
            "test_partition_accessed",
        }
        self.assertEqual(set(epoch_one), independently_required)
        self.assertEqual(set(epoch_one), set(V6_PILOT_CHECKPOINT_FIELDS))
        self.assertEqual(epoch_one["source_provenance"]["git_status_porcelain"], [])

        with tempfile.TemporaryDirectory() as temporary:
            first_path = Path(temporary) / "epoch-0001.pt"
            final_path = Path(temporary) / "epoch-0002.pt"
            save_v6_checkpoint(first_path, epoch_one)
            save_v6_checkpoint(final_path, epoch_two)
            for path, expected_epoch in ((first_path, 1), (final_path, 2)):
                reloaded_model = ConstrainedProfileV6Model(model_config)
                reloaded_optimizer = build_v6_optimizer(
                    reloaded_model, optimization
                )
                loaded = load_pilot_checkpoint(
                    path,
                    reloaded_model,
                    reloaded_optimizer,
                    model_config,
                    optimization,
                    pilot_config,
                    data,
                    map_location=torch.device("cpu"),
                )
                self.assertEqual(loaded["epoch"], expected_epoch)
                for name, value in loaded["model_state"].items():
                    torch.testing.assert_close(
                        reloaded_model.state_dict()[name], value, rtol=0, atol=0
                    )
            self.assertTrue(first_path.is_file())
            self.assertTrue(final_path.is_file())
            self.assertNotEqual(first_path, final_path)
            self.assertEqual(
                set(model.state_dict()), set(epoch_one["model_state"])
            )
            terminal_style = {
                "selected_checkpoint": str(first_path),
                "final_checkpoint": str(final_path),
                "strict_selected_reload_reproduced": True,
                "strict_final_reload_reproduced": True,
                "systematic_partition_accessed": False,
                "test_partition_accessed": False,
            }
            json.dumps(terminal_style, sort_keys=True, allow_nan=False)

    def test_v6_checkpoint_mutations_are_structured_malformed_errors(self):
        from prototype.flat_baseline.constrained_v6 import ConstrainedProfileV6Model
        from prototype.flat_baseline.constrained_v6_checkpoint import (
            V6CheckpointValidationError,
            validate_v6_checkpoint_payload,
            validate_v6_pilot_checkpoint_state,
        )
        from prototype.flat_baseline.constrained_v6_pilot import (
            load_pilot_checkpoint,
        )
        from prototype.flat_baseline.constrained_v6_training import (
            build_v6_optimizer,
            save_v6_checkpoint,
        )
        fixture, payload = self._pilot_payload(1)
        _, _, model_config, _, pilot_config, data, _ = fixture
        grammar_mutations = {
            "contract_id": "wrong",
            "completion_algorithm_id": "wrong",
            "node_vocabulary": list(reversed(payload["node_vocabulary"])),
            "transitions": [["START", "sketch"]],
            "terminal_nodes": ["extrude"],
            "valid_requested_node_counts": [4],
            "operation_limit": 1,
            "canonical_templates": [["reference_plane", "extrude"]],
        }
        for name, value in grammar_mutations.items():
            malformed = dict(payload)
            malformed["node_grammar_contract"] = dict(
                payload["node_grammar_contract"], **{name: value}
            )
            with self.subTest(grammar=name):
                with self.assertRaises(V6CheckpointValidationError) as caught:
                    validate_v6_checkpoint_payload(
                        malformed,
                        "pilot_fixed_epoch",
                        model_config,
                        torch_module=torch,
                    )
                self.assertEqual(caught.exception.code, "malformed_checkpoint")
                json.dumps({"detail": caught.exception.detail}, allow_nan=False)

        mutations = []
        for missing in ("model_state", "optimizer_state", "pilot_config"):
            malformed = dict(payload)
            del malformed[missing]
            mutations.append(malformed)
        malformed = dict(payload)
        malformed["unexpected"] = True
        mutations.append(malformed)
        malformed = dict(payload, epoch=True)
        mutations.append(malformed)
        malformed = dict(payload, systematic_partition_accessed=0)
        mutations.append(malformed)
        malformed = dict(payload, systematic_partition_accessed=True)
        mutations.append(malformed)
        malformed = dict(payload, checkpoint_version=5)
        mutations.append(malformed)
        for name, value in (
            ("axis_geometry_contract_id", "wrong"),
            ("axis_construction_algorithm_id", "wrong"),
            ("axis_geometry_channel_indices", [34, 35, 36, 37, 38, 39]),
            ("axis_coordinate_space", "world"),
            ("axis_raw_constrained_evidence_policy", "wrong"),
            (
                "axis_geometry_contract",
                dict(
                    payload["axis_geometry_contract"],
                    canonical_normalized_channels=[1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                ),
            ),
        ):
            malformed = dict(payload)
            malformed[name] = value
            mutations.append(malformed)
        malformed = dict(payload, validation_summaries={
            "teacher_forced": {"total_loss": float("nan")},
            "autonomous": {},
        })
        mutations.append(malformed)
        malformed = dict(payload)
        malformed["rng_state"] = dict(payload["rng_state"], python=(3, (), None))
        mutations.append(malformed)
        malformed = dict(payload)
        malformed["vq_state"] = {}
        mutations.append(malformed)
        malformed = dict(payload)
        malformed["source_provenance"] = dict(
            payload["source_provenance"], git_status_porcelain=""
        )
        mutations.append(malformed)
        for index, malformed in enumerate(mutations):
            with self.subTest(malformed=index):
                with self.assertRaises(V6CheckpointValidationError):
                    validate_v6_checkpoint_payload(
                        malformed,
                        "pilot_fixed_epoch",
                        model_config,
                        torch_module=torch,
                    )
                    validate_v6_pilot_checkpoint_state(
                        malformed, pilot_config, len(data.train.flat_examples)
                    )

        for name, malformed in (
            (
                "training_partition_state",
                dict(payload, training_partition_state={"partition": "wrong"}),
            ),
            (
                "pilot_config",
                dict(
                    payload,
                    pilot_config=dict(payload["pilot_config"], seed=2027),
                ),
            ),
        ):
            with self.subTest(loader_mismatch=name):
                with tempfile.TemporaryDirectory() as temporary:
                    path = Path(temporary) / "malformed.pt"
                    save_v6_checkpoint(path, malformed)
                    reloaded_model = ConstrainedProfileV6Model(model_config)
                    reloaded_optimizer = build_v6_optimizer(
                        reloaded_model, fixture[3]
                    )
                    with self.assertRaises(ConstrainedV6PilotError) as caught:
                        load_pilot_checkpoint(
                            path,
                            reloaded_model,
                            reloaded_optimizer,
                            model_config,
                            fixture[3],
                            pilot_config,
                            data,
                            map_location=torch.device("cpu"),
                        )
                    self.assertEqual(caught.exception.code, "malformed_checkpoint")
