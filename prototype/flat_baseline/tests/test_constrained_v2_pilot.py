"""Bounded train-and-ordinary-validation pilot contract tests."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import replace
import ast
import json
import math
from pathlib import Path
import random
import tempfile
from unittest import mock
import unittest

try:
    import torch
except ImportError:
    torch = None

from prototype.controlled_data.builders import build_history
from prototype.controlled_data.factors import PrimitiveFamily
from prototype.controlled_data.identity import source_family_id
from prototype.model_data.tests.fixtures import source, write_physical_corpus
from prototype.representation.model import GeometryEncoding

from prototype.flat_baseline.constrained_v2_pilot_config import (
    ConstrainedV2PilotConfig,
    ConstrainedV2PilotError,
    PILOT_IDENTITY,
    validate_pilot_partition_authorization,
)
from prototype.flat_baseline.constrained_v2_training_config import (
    ConstrainedV2TrainingConfig,
)

if torch is not None:
    from prototype.flat_baseline.constrained_v2 import (
        ConstrainedProfileV2Model,
    )
    from prototype.flat_baseline.constrained_v2_config import (
        ConstrainedProfileV2Config,
    )
    from prototype.flat_baseline.constrained_v2_pilot import (
        PILOT_CHECKPOINT_FIELDS,
        PilotPartition,
        _require_finite_json,
        _write_terminal_success,
        autonomous_validation,
        deterministic_epoch_family_ids,
        load_pilot_checkpoint,
        load_pilot_data,
        pilot_checkpoint_payload,
        pilot_optimization_config,
        run_constrained_v2_pilot,
        teacher_forced_validation,
    )
    from prototype.flat_baseline.constrained_v2_training import (
        V2_TRAINING_LOSS_FIELDS,
        build_v2_optimizer,
        save_v2_checkpoint,
    )
    from prototype.flat_baseline.provenance import source_state
    from prototype.flat_baseline.run_logging import JsonlLogger


TORCH_REASON = "real PyTorch execution is deferred to the Adroit environment"


def _sources():
    return (
        source("E", PrimitiveFamily.CIRCLE, extents=(1.0,)),
        source("R", PrimitiveFamily.RECTANGLE_LINES, extents=(1.0,)),
        source("ER", PrimitiveFamily.CAPSULE_LINE_ARC, extents=(1.0, 2.0)),
        source("EE", PrimitiveFamily.CIRCLE, extents=(2.0, 3.0)),
        source("R", PrimitiveFamily.RECTANGLE_LINES, extents=(3.0,)),
        source("E", PrimitiveFamily.CAPSULE_LINE_ARC, extents=(2.0,)),
        source("R", PrimitiveFamily.CIRCLE, extents=(2.0,)),
        source("E", PrimitiveFamily.RECTANGLE_LINES, extents=(1.0,)),
        source("R", PrimitiveFamily.CAPSULE_LINE_ARC, extents=(3.0,)),
        source("RR", PrimitiveFamily.CIRCLE, extents=(1.0, 3.0)),
        source("ER", PrimitiveFamily.RECTANGLE_LINES, extents=(2.0, 1.0)),
        source("EE", PrimitiveFamily.CAPSULE_LINE_ARC, extents=(3.0, 2.0)),
    )


def _family_id(physical):
    history = build_history(physical, GeometryEncoding.CONTINUOUS)
    return source_family_id(history)


class ConstrainedV2PilotConfigurationTests(unittest.TestCase):
    def test_canonical_serialization_and_fixed_budget(self):
        config = ConstrainedV2PilotConfig()
        config.validate()
        self.assertEqual(config.pilot_identity, PILOT_IDENTITY)
        self.assertEqual(config.epochs, 2)
        self.assertEqual(config.batch_size, 8)
        self.assertEqual(config.validation_interval, 1)
        self.assertEqual(config.checkpoint_interval, 1)
        self.assertEqual(config.training_partition, "train")
        self.assertEqual(config.validation_partition, "validation")
        self.assertEqual(
            config.validation_partition_identity, "iid_validation"
        )
        self.assertEqual(json.loads(config.to_json()), config.to_dict())
        self.assertEqual(config.to_json(), ConstrainedV2PilotConfig().to_json())
        self.assertFalse(config.systematic_partition_accessed)
        self.assertFalse(config.test_partition_accessed)

    def test_pilot_is_distinct_from_tiny_overfit_configuration(self):
        pilot = ConstrainedV2PilotConfig()
        tiny = ConstrainedV2TrainingConfig()
        self.assertNotEqual(pilot.pilot_identity, tiny.tiny_overfit_selection_identity)
        self.assertEqual(pilot.epochs, 2)
        self.assertEqual(tiny.maximum_steps, 500)
        self.assertEqual(pilot.validation_partition_identity, "iid_validation")
        self.assertEqual(tiny.validation_partition, "none")

    def test_rejects_budget_partition_vq_and_access_changes(self):
        invalid = (
            {"epochs": 3},
            {"batch_size": 4},
            {"validation_interval": 2},
            {"checkpoint_interval": 2},
            {"training_partition": "test"},
            {"validation_partition": "test"},
            {"validation_partition": "systematic"},
            {"validation_partition_identity": "systematic_validation"},
            {"vq_initialization": "train-kmeans"},
            {"systematic_partition_accessed": True},
            {"test_partition_accessed": True},
            {"seed": -1},
            {"seed": True},
            {"device": "auto"},
        )
        for changes in invalid:
            with self.subTest(changes=changes):
                with self.assertRaises(ConstrainedV2PilotError):
                    validate_pilot_partition_authorization(
                        replace(ConstrainedV2PilotConfig(), **changes)
                    )

    def test_python_38_grammar_and_pytorch_111_surface(self):
        root = Path(__file__).resolve().parents[1]
        for name in (
            "constrained_v2_pilot_config.py",
            "constrained_v2_pilot.py",
            "run_constrained_v2_pilot.py",
        ):
            path = root / name
            text = path.read_text(encoding="utf-8")
            ast.parse(text, filename=str(path), feature_version=(3, 8))
            for token in (
                "torch.compile",
                "torch.asarray",
                "torch.func",
                "torch.vmap",
                "Tensor.scatter_reduce",
            ):
                self.assertNotIn(token, text)

    def test_v1_entry_points_do_not_reference_pilot(self):
        root = Path(__file__).resolve().parents[1]
        for name in ("training.py", "train.py", "training_config.py"):
            self.assertNotIn(
                "constrained_v2_pilot",
                (root / name).read_text(encoding="utf-8"),
            )


@unittest.skipUnless(torch is not None, TORCH_REASON)
class ConstrainedV2PilotTensorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.corpus = root / "corpus"
        physical = _sources()
        validation_ids = {_family_id(item) for item in physical[6:]}
        partitions = {
            _family_id(item): (
                "validation"
                if _family_id(item) in validation_ids
                else "train"
            )
            for item in physical
        }
        write_physical_corpus(
            self.corpus, physical, partitions=partitions
        )
        self.pilot_config = ConstrainedV2PilotConfig(
            seed=37,
            output_dir=str(root / "run"),
            require_clean_source=False,
        )
        self.model_config = ConstrainedProfileV2Config(
            model_dim=32,
            num_heads=4,
            feedforward_dim=64,
            encoder_layers=1,
            decoder_layers=1,
            dropout=0.0,
            max_nodes=10,
            max_operations=2,
            latent_tokens=2,
            codebook_size=16,
            codebook_dim=16,
            edge_pair_dim=32,
        )
        self.data = load_pilot_data(self.corpus, self.pilot_config)
        torch.manual_seed(self.pilot_config.seed)
        self.model = ConstrainedProfileV2Model(self.model_config)
        self.optimizer = build_v2_optimizer(
            self.model, pilot_optimization_config(self.pilot_config, 2)
        )

    def test_authorization_precedes_loader_and_model_construction(self):
        invalid = replace(
            self.pilot_config, validation_partition="test"
        )
        with mock.patch(
            "prototype.flat_baseline.constrained_v2_pilot."
            "load_partition_physical_examples"
        ) as loader, mock.patch(
            "prototype.flat_baseline.constrained_v2_pilot."
            "ConstrainedProfileV2Model"
        ) as model:
            with self.assertRaises(ConstrainedV2PilotError):
                run_constrained_v2_pilot(
                    self.corpus, invalid, self.model_config
                )
        loader.assert_not_called()
        model.assert_not_called()

        collision = Path(self.temporary.name) / "collision"
        collision.mkdir()
        collision_config = replace(
            self.pilot_config, output_dir=str(collision)
        )
        with mock.patch(
            "prototype.flat_baseline.constrained_v2_pilot."
            "load_pilot_data"
        ) as data_loader, mock.patch(
            "prototype.flat_baseline.constrained_v2_pilot."
            "ConstrainedProfileV2Model"
        ) as model:
            with self.assertRaisesRegex(
                ConstrainedV2PilotError, "output_collision"
            ):
                run_constrained_v2_pilot(
                    self.corpus, collision_config, self.model_config
                )
        data_loader.assert_not_called()
        model.assert_not_called()
        self.assertEqual(tuple(collision.iterdir()), ())

    def test_scoped_data_and_deterministic_sampler(self):
        self.assertEqual(len(self.data.train.flat_examples), 6)
        self.assertEqual(len(self.data.validation.flat_examples), 6)
        self.assertFalse(
            set(self.data.train.family_ids)
            & set(self.data.validation.family_ids)
        )
        first = deterministic_epoch_family_ids(
            self.data.train.physical_examples, 2, 37, 1
        )
        second = deterministic_epoch_family_ids(
            self.data.train.physical_examples, 2, 37, 1
        )
        changed = deterministic_epoch_family_ids(
            self.data.train.physical_examples, 2, 37, 2
        )
        self.assertEqual(first, second)
        self.assertNotEqual(first, changed)
        self.assertEqual(
            sorted(item for batch in first for item in batch),
            sorted(self.data.train.family_ids),
        )

    def test_teacher_forced_complete_multibatch_metrics_and_no_mutation(self):
        expected_mapping = {
            "a": torch.tensor([1.0, 2.0], dtype=torch.float32),
            "b": torch.tensor([3.0], dtype=torch.float32),
        }
        identical_mapping = OrderedDict((
            ("a", expected_mapping["a"].clone()),
            ("b", expected_mapping["b"].clone()),
        ))
        _assert_state_equal(self, expected_mapping, identical_mapping)
        unequal_mappings = (
            OrderedDict((("a", identical_mapping["a"]),)),
            OrderedDict((
                ("a", identical_mapping["a"]),
                ("b", identical_mapping["b"]),
                ("c", torch.tensor([4.0])),
            )),
            OrderedDict((
                ("b", identical_mapping["b"]),
                ("a", identical_mapping["a"]),
            )),
            OrderedDict((
                ("a", torch.tensor([1.0, 9.0])),
                ("b", identical_mapping["b"]),
            )),
            OrderedDict((
                ("a", identical_mapping["a"].to(torch.float64)),
                ("b", identical_mapping["b"]),
            )),
            OrderedDict((
                ("a", identical_mapping["a"].reshape(1, 2)),
                ("b", identical_mapping["b"]),
            )),
        )
        for actual_mapping in unequal_mappings:
            with self.subTest(actual_mapping=actual_mapping):
                with self.assertRaises(AssertionError):
                    _assert_state_equal(
                        self, expected_mapping, actual_mapping
                    )

        state_before = _clone_state(self.model.state_dict())
        optimizer_before = _clone_state(self.optimizer.state_dict())
        python_rng = random.getstate()
        torch_rng = torch.get_rng_state().clone()
        summary = teacher_forced_validation(
            self.model,
            self.data.validation,
            self.model_config,
            2,
            torch.device("cpu"),
        )
        self.assertEqual(summary["example_count"], 6)
        self.assertTrue(all(
            name in summary for name in V2_TRAINING_LOSS_FIELDS
        ))
        self.assertTrue(all(
            math.isfinite(float(summary[name]))
            for name in V2_TRAINING_LOSS_FIELDS
        ))
        confusion = summary["family_confusion_matrix"]
        self.assertEqual((len(confusion), len(confusion[0])), (3, 3))
        self.assertEqual(
            sum(sum(row) for row in confusion),
            summary["profile_family_count"],
        )
        self.assertTrue(all(
            metrics["applicable_sketch_count"] > 0
            and metrics["parameter_value_count"]
            == 3 * metrics["applicable_sketch_count"]
            and math.isfinite(metrics["mae"])
            and math.isfinite(metrics["smooth_l1"])
            for metrics in summary["compact_parameter_metrics"].values()
        ))
        _assert_state_equal(self, state_before, self.model.state_dict())
        _assert_state_equal(self, optimizer_before, self.optimizer.state_dict())
        self.assertEqual(random.getstate(), python_rng)
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_rng))
        self.assertTrue(self.model.training)

    def test_autonomous_complete_failure_histogram_and_target_independence(self):
        state_before = _clone_state(self.model.state_dict())
        python_rng = random.getstate()
        torch_rng = torch.get_rng_state().clone()
        first = autonomous_validation(
            self.model,
            self.data.validation,
            self.model_config,
            2,
            torch.device("cpu"),
        )
        mutated_flat = tuple(
            replace(
                item,
                target=replace(
                    item.target,
                    categorical_attributes=tuple(
                        tuple(0 for _ in row)
                        for row in item.target.categorical_attributes
                    ),
                    geometry=tuple(
                        tuple(0.0 for _ in row)
                        for row in item.target.geometry
                    ),
                    geometry_mask=tuple(
                        tuple(False for _ in row)
                        for row in item.target.geometry_mask
                    ),
                ),
            )
            for item in self.data.validation.flat_examples
        )
        mutated = replace(
            self.data.validation, flat_examples=mutated_flat
        )
        second = autonomous_validation(
            self.model,
            mutated,
            self.model_config,
            2,
            torch.device("cpu"),
        )
        self.assertEqual(first, second)
        self.assertEqual(first["example_count"], 6)
        self.assertEqual(len(first["outcomes"]), 6)
        self.assertEqual(
            first["valid_count"]
            + sum(first["failure_reason_histogram"].values()),
            6,
        )
        self.assertEqual(
            (len(first["predicted_family_confusion_matrix"]),
             len(first["predicted_family_confusion_matrix"][0])),
            (3, 4),
        )
        self.assertEqual(
            sum(
                sum(row)
                for row in first["predicted_family_confusion_matrix"]
            ),
            sum(item.metadata.history_depth for item in
                self.data.validation.physical_examples),
        )
        _assert_state_equal(self, state_before, self.model.state_dict())
        self.assertEqual(random.getstate(), python_rng)
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_rng))
        self.assertTrue(self.model.training)

    def test_checkpoint_schema_cadence_collision_and_strict_reload(self):
        optimization = pilot_optimization_config(self.pilot_config, 2)
        teacher = teacher_forced_validation(
            self.model, self.data.validation, self.model_config, 2,
            torch.device("cpu"),
        )
        autonomous = autonomous_validation(
            self.model, self.data.validation, self.model_config, 2,
            torch.device("cpu"),
        )
        payload = pilot_checkpoint_payload(
            self.model,
            self.optimizer,
            self.model_config,
            optimization,
            self.pilot_config,
            self.data,
            epoch=1,
            global_step=1,
            examples_processed=6,
            configured_maximum_steps=2,
            teacher_summary=teacher,
            autonomous_summary=autonomous,
            source_provenance=source_state(),
        )
        self.assertEqual(set(payload), PILOT_CHECKPOINT_FIELDS)
        path = Path(self.temporary.name) / "epoch-0001.pt"
        save_v2_checkpoint(path, payload)
        with self.assertRaisesRegex(RuntimeError, "checkpoint_collision"):
            save_v2_checkpoint(path, payload)
        reloaded = ConstrainedProfileV2Model(self.model_config)
        reloaded_optimizer = build_v2_optimizer(reloaded, optimization)
        loaded = load_pilot_checkpoint(
            path,
            reloaded,
            reloaded_optimizer,
            self.model_config,
            optimization,
            self.pilot_config,
            self.data,
            map_location=torch.device("cpu"),
        )
        self.assertEqual(loaded["validation_summaries"]["teacher_forced"], teacher)
        _assert_state_equal(self, self.model.state_dict(), reloaded.state_dict())
        reproduced = teacher_forced_validation(
            reloaded, self.data.validation, self.model_config, 2,
            torch.device("cpu"),
        )
        _assert_nested_close(self, teacher, reproduced)
        malformed = dict(payload)
        malformed.pop("validation_cadence")
        malformed_path = Path(self.temporary.name) / "malformed.pt"
        torch.save(malformed, str(malformed_path))
        with self.assertRaisesRegex(
            ConstrainedV2PilotError, "malformed_checkpoint"
        ):
            load_pilot_checkpoint(
                malformed_path,
                reloaded,
                reloaded_optimizer,
                self.model_config,
                optimization,
                self.pilot_config,
                self.data,
                map_location=torch.device("cpu"),
            )

    def test_finite_json_and_terminal_failure_record(self):
        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    ConstrainedV2PilotError, "nonfinite_json"
                ):
                    _require_finite_json({"value": value})
        output = Path(self.pilot_config.output_dir)
        output.mkdir()
        (output / "metrics.jsonl").write_text(
            json.dumps({
                "event": "run_metadata",
                "pilot_config": self.pilot_config.to_dict(),
            }) + "\n",
            encoding="utf-8",
        )
        with mock.patch(
            "prototype.flat_baseline.constrained_v2_pilot."
            "_run_constrained_v2_pilot",
            side_effect=ConstrainedV2PilotError("synthetic", "failure"),
        ):
            with self.assertRaises(ConstrainedV2PilotError):
                run_constrained_v2_pilot(
                    self.corpus, self.pilot_config, self.model_config
                )
        records = [
            json.loads(line)
            for line in (output / "metrics.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
        ]
        self.assertEqual(records[-1]["event"], "terminal_failure")
        self.assertEqual(records[-1]["error_code"], "synthetic")
        self.assertFalse(records[-1]["systematic_partition_accessed"])
        self.assertFalse(records[-1]["test_partition_accessed"])

        success_path = Path(self.temporary.name) / "success.jsonl"
        _write_terminal_success(
            JsonlLogger(success_path),
            {
                "acceptance": {"configured_budget_completed": True},
                "global_step": 2,
            },
        )
        success = json.loads(
            success_path.read_text(encoding="utf-8").strip()
        )
        self.assertEqual(success["event"], "terminal_success")
        self.assertFalse(success["systematic_partition_accessed"])
        self.assertFalse(success["test_partition_accessed"])


def _clone_state(value):
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, dict):
        return {name: _clone_state(item) for name, item in value.items()}
    if isinstance(value, list):
        return [_clone_state(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_state(item) for item in value)
    return value


def _assert_state_equal(test, expected, actual):
    if torch.is_tensor(expected):
        test.assertTrue(torch.is_tensor(actual))
        test.assertEqual(expected.dtype, actual.dtype)
        test.assertEqual(expected.shape, actual.shape)
        test.assertEqual(expected.device, actual.device)
        test.assertTrue(torch.equal(expected, actual))
    elif isinstance(expected, Mapping):
        test.assertIsInstance(actual, Mapping)
        test.assertEqual(list(expected.keys()), list(actual.keys()))
        for name in expected:
            _assert_state_equal(test, expected[name], actual[name])
    elif isinstance(expected, (list, tuple)):
        test.assertEqual(type(expected), type(actual))
        test.assertEqual(len(expected), len(actual))
        for first, second in zip(expected, actual):
            _assert_state_equal(test, first, second)
    else:
        test.assertEqual(type(expected), type(actual))
        test.assertEqual(expected, actual)


def _assert_nested_close(test, expected, actual):
    test.assertEqual(type(expected), type(actual))
    if isinstance(expected, dict):
        test.assertEqual(set(expected), set(actual))
        for name in expected:
            _assert_nested_close(test, expected[name], actual[name])
    elif isinstance(expected, list):
        test.assertEqual(len(expected), len(actual))
        for first, second in zip(expected, actual):
            _assert_nested_close(test, first, second)
    elif isinstance(expected, float):
        test.assertTrue(
            math.isclose(expected, actual, rel_tol=1e-5, abs_tol=2e-7)
        )
    else:
        test.assertEqual(expected, actual)


if __name__ == "__main__":
    unittest.main()
