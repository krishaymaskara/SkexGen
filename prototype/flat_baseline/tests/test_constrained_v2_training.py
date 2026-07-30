"""Train-only configuration, optimization, and checkpoint tests for V2."""

from __future__ import annotations

from dataclasses import replace
import ast
import json
import math
from pathlib import Path
import tempfile
from unittest import mock
import unittest

try:
    import torch
except ImportError:
    torch = None

from prototype.controlled_data.factors import PrimitiveFamily
from prototype.model_data.loader import load_partition_physical_examples
from prototype.model_data.tests.fixtures import source, write_physical_corpus
from prototype.profile_geometry import PROFILE_FAMILIES

from prototype.flat_baseline.constrained_v2_training_config import (
    ConstrainedV2TrainingConfig,
    ConstrainedV2TrainingConfigurationError,
)

if torch is not None:
    from prototype.flat_baseline.constrained_v2 import (
        ConstrainedProfileV2Model,
    )
    from prototype.flat_baseline.constrained_v2_config import (
        ConstrainedProfileV2Config,
    )
    from prototype.flat_baseline.constrained_v2_training import (
        ConstrainedV2TrainingError,
        V2_CHECKPOINT_FIELDS,
        V2_TRAINING_LOSS_FIELDS,
        build_v2_optimizer,
        constrained_v2_evaluation_snapshot,
        constrained_v2_training_step,
        load_v2_checkpoint,
        load_v2_tiny_training_selection,
        operation_coverage_for_example,
        run_v2_tiny_overfit,
        save_v2_checkpoint,
        select_v2_tiny_training_examples,
        v2_checkpoint_payload,
    )
    from prototype.model_data.batching import collate_flat


TORCH_REASON = "real PyTorch execution is deferred to the Adroit environment"


def _sources():
    return (
        source("E", PrimitiveFamily.CIRCLE, extents=(1.0,)),
        source("R", PrimitiveFamily.CIRCLE, extents=(2.0,)),
        source("ER", PrimitiveFamily.CIRCLE, extents=(3.0, 1.0)),
        source("E", PrimitiveFamily.RECTANGLE_LINES, extents=(2.0,)),
        source("R", PrimitiveFamily.RECTANGLE_LINES, extents=(3.0,)),
        source(
            "ER",
            PrimitiveFamily.RECTANGLE_LINES,
            extents=(1.0, 2.0),
        ),
        source("E", PrimitiveFamily.CAPSULE_LINE_ARC, extents=(3.0,)),
        source(
            "ER",
            PrimitiveFamily.CAPSULE_LINE_ARC,
            extents=(2.0, 3.0),
        ),
    )


class ConstrainedV2TrainingConfigurationTests(unittest.TestCase):
    def test_canonical_serialization_and_frozen_identity(self):
        config = ConstrainedV2TrainingConfig()
        config.validate()
        self.assertEqual(config.to_json(), ConstrainedV2TrainingConfig().to_json())
        self.assertEqual(json.loads(config.to_json()), config.to_dict())
        self.assertEqual(config.training_partition, "train")
        self.assertEqual(config.validation_partition, "none")
        self.assertFalse(config.test_partition_accessed)
        self.assertEqual(
            config.profile_family_order,
            tuple(item.value for item in PROFILE_FAMILIES),
        )

    def test_invalid_identity_numeric_partition_and_vq_values(self):
        invalid = (
            {"model_name": "B0-FLAT-MIXED-VQ"},
            {"checkpoint_version": 1},
            {"profile_family_order": tuple(reversed(PROFILE_FAMILIES))},
            {"extent_min": 0.2},
            {"extent_max": 0.8},
            {"batch_size": 0},
            {"learning_rate": 0.0},
            {"profile_family_loss_weight": 0.0},
            {"weight_decay": math.nan},
            {"gradient_clip_norm": math.inf},
            {"maximum_steps": 501},
            {"tiny_subset_size": 13},
            {"vq_initialization": "train-kmeans"},
            {"training_partition": "validation"},
            {"validation_partition": "validation"},
            {"test_partition_accessed": True},
        )
        for changes in invalid:
            with self.subTest(changes=changes):
                with self.assertRaises(
                    ConstrainedV2TrainingConfigurationError
                ):
                    replace(
                        ConstrainedV2TrainingConfig(), **changes
                    ).validate()

    def test_python_38_grammar_and_pytorch_111_compatibility(self):
        root = Path(__file__).resolve().parents[1]
        for name in (
            "constrained_v2_training_config.py",
            "constrained_v2_training.py",
            "train_constrained_v2.py",
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

    def test_v1_training_entry_points_do_not_reference_v2(self):
        root = Path(__file__).resolve().parents[1]
        for name in ("training.py", "train.py", "training_config.py"):
            text = (root / name).read_text(encoding="utf-8")
            self.assertNotIn("constrained_v2", text)


@unittest.skipUnless(torch is not None, TORCH_REASON)
class ConstrainedV2TrainingTensorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.corpus = Path(self.temporary.name) / "corpus"
        write_physical_corpus(self.corpus, _sources())
        self.training_config = ConstrainedV2TrainingConfig(
            seed=37,
            batch_size=8,
            learning_rate=1e-3,
            maximum_steps=100,
            logging_cadence=10,
            checkpoint_cadence=100,
            output_dir=str(Path(self.temporary.name) / "run"),
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
        self.selection = load_v2_tiny_training_selection(
            self.corpus, self.training_config
        )
        self.batch = collate_flat(self.selection.examples)
        torch.manual_seed(self.training_config.seed)
        self.model = ConstrainedProfileV2Model(self.model_config)
        self.optimizer = build_v2_optimizer(
            self.model, self.training_config
        )

    def _step(self, step=1):
        return constrained_v2_training_step(
            self.model,
            self.optimizer,
            self.batch,
            self.model_config,
            self.training_config,
            torch.device("cpu"),
            global_step=step,
            examples_processed=step * len(self.batch.family_ids),
        )

    def test_train_only_selection_is_deterministic_and_covers_domain(self):
        physical = load_partition_physical_examples(
            self.corpus, "iid", "train"
        )
        first = select_v2_tiny_training_examples(
            physical,
            corpus_dir=str(self.corpus),
            split_name="iid",
            subset_size=8,
        )
        second = select_v2_tiny_training_examples(
            tuple(reversed(physical)),
            corpus_dir=str(self.corpus),
            split_name="iid",
            subset_size=8,
        )
        self.assertEqual(first.metadata(), second.metadata())
        self.assertEqual(len(first.examples), 8)
        self.assertTrue(all(
            count > 0 for count in first.family_counts.values()
        ))
        self.assertEqual(
            set(first.exact_operation_types),
            {"extrude_1", "revolve_1", "revolve_2"},
        )
        self.assertEqual(
            set(first.operation_families), {"extrude", "revolve"}
        )
        self.assertEqual(
            first.selected_example_ids,
            tuple(item.physical_family_id for item in physical),
        )
        self.assertIn(8, first.node_count_distribution)
        self.assertFalse(first.missing_coverage)
        self.assertEqual(first.training_partition, "train")
        self.assertEqual(first.metadata()["validation_partition"], "none")
        self.assertFalse(first.metadata()["test_partition_accessed"])
        self.assertEqual(len(first.train_selection_sha256), 64)
        exact, families = operation_coverage_for_example(physical[0])
        self.assertEqual(exact, physical[0].operation_sequence)
        self.assertEqual(
            families,
            tuple(
                next(
                    node.operation_type
                    for node in physical[0].nodes
                    if node.node_id == operation_id
                )
                for operation_id in exact
            ),
        )

        malformed = replace(
            physical[0], operation_sequence=("loft_1",)
        )
        with self.assertRaisesRegex(
            RuntimeError, "unsupported_operation_type"
        ):
            operation_coverage_for_example(malformed)

    def test_partition_rejection_occurs_before_loader_or_model(self):
        invalid = replace(
            self.training_config, training_partition="test"
        )
        with mock.patch(
            "prototype.flat_baseline.constrained_v2_training."
            "load_partition_physical_examples"
        ) as loader:
            with self.assertRaises(
                ConstrainedV2TrainingConfigurationError
            ):
                load_v2_tiny_training_selection(self.corpus, invalid)
        loader.assert_not_called()

    def test_output_collision_is_rejected_before_data_or_model_access(self):
        output = Path(self.training_config.output_dir)
        output.mkdir()
        with mock.patch(
            "prototype.flat_baseline.constrained_v2_training."
            "load_v2_tiny_training_selection"
        ) as loader, mock.patch(
            "prototype.flat_baseline.constrained_v2_training."
            "ConstrainedProfileV2Model"
        ) as model:
            with self.assertRaisesRegex(RuntimeError, "output_collision"):
                run_v2_tiny_overfit(
                    self.corpus,
                    self.model_config,
                    self.training_config,
                )
        loader.assert_not_called()
        model.assert_not_called()

    def test_optimizer_contains_each_v2_parameter_once(self):
        expected = {
            id(parameter)
            for parameter in self.model.parameters()
            if parameter.requires_grad
        }
        actual = tuple(
            parameter
            for group in self.optimizer.param_groups
            for parameter in group["params"]
        )
        self.assertEqual({id(item) for item in actual}, expected)
        self.assertEqual(len(actual), len(expected))

    def test_one_step_is_finite_logs_profiles_and_updates_ema(self):
        ema_before = {
            name: value.clone()
            for name, value in self.model.vq.state_dict().items()
        }
        result = self._step()
        self.assertTrue(
            set(V2_TRAINING_LOSS_FIELDS).issubset(result.metrics)
        )
        self.assertNotIn("primitive_slot_loss", result.metrics)
        self.assertNotIn("primitive_geometry_loss", result.metrics)
        self.assertGreater(result.metrics["profile_family_count"], 0)
        self.assertGreater(result.metrics["profile_parameter_count"], 0)
        self.assertTrue(all(
            math.isfinite(float(value))
            for value in result.metrics.values()
        ))
        self.assertIn(
            "profile_heads.family_head.weight",
            result.gradient_parameter_names,
        )
        self.assertIn(
            "profile_heads.parameter_head.weight",
            result.gradient_parameter_names,
        )
        self.assertIn(
            "non_profile_geometry_head.weight",
            result.gradient_parameter_names,
        )
        self.assertTrue(any(
            not torch.equal(value, self.model.vq.state_dict()[name])
            for name, value in ema_before.items()
        ))

    def test_total_and_both_profile_losses_decrease_on_repeated_batch(self):
        from prototype.constrained_profile_decoder import (
            profile_targets_for_loss,
        )
        from prototype.flat_baseline.constrained_v2_losses import (
            constrained_profile_v2_loss,
        )

        # Training-step losses observe a changing EMA codebook. Fixed-batch
        # evaluation snapshots provide deterministic before/after comparisons
        # without updating EMA state or sampling different batch compositions.
        initial_ema = {
            name: value.clone()
            for name, value in self.model.vq.state_dict().items()
        }
        self.assertTrue(self.model.training)
        initial = constrained_v2_evaluation_snapshot(
            self.model,
            self.batch,
            self.model_config,
            self.training_config,
            torch.device("cpu"),
        )
        self.assertTrue(self.model.training)
        for name, expected in initial_ema.items():
            torch.testing.assert_close(
                expected,
                self.model.vq.state_dict()[name],
                rtol=0.0,
                atol=0.0,
            )

        inputs = self.batch.to_torch(torch)
        target = self.batch.target.to_torch(torch)
        profile_targets = profile_targets_for_loss(
            self.batch.target, inputs["geometry"]
        )
        self.model.eval()
        output = self.model(
            target=target,
            profile_targets=profile_targets,
            **inputs
        )
        losses = constrained_profile_v2_loss(
            output, target, profile_targets, self.model_config
        )
        self.model.train()
        selected_ids = profile_targets.family_ids[
            profile_targets.sketch_mask
        ]
        self.assertEqual(
            profile_targets.sketch_mask.shape,
            target["node_mask"].shape,
        )
        self.assertEqual(
            int(profile_targets.sketch_mask.sum().item()), 11
        )
        self.assertEqual(
            selected_ids.tolist(),
            [0, 0, 0, 2, 1, 1, 1, 0, 1, 2, 2],
        )
        self.assertTrue(
            (profile_targets.family_ids[
                ~profile_targets.sketch_mask
            ] == -1).all()
        )
        self.assertEqual(
            tuple(item.value for item in PROFILE_FAMILIES),
            self.model_config.profile_family_order,
        )
        self.assertEqual(
            output.profile_family_logits.shape,
            profile_targets.family_ids.shape + (len(PROFILE_FAMILIES),),
        )
        self.assertEqual(
            self.model_config.profile_family_loss_weight, 1.0
        )
        self.assertEqual(losses.profile_family_count, 11)
        self.assertEqual(
            torch.bincount(
                selected_ids, minlength=len(PROFILE_FAMILIES)
            ).tolist(),
            [4, 4, 3],
        )
        self.assertTrue(output.profile_family_logits.requires_grad)
        self.assertTrue(losses.profile_family.requires_grad)
        weighted_total = (
            self.model_config.node_type_loss_weight * losses.node_type
            + self.model_config.categorical_loss_weight
            * losses.categorical_attributes
            + self.model_config.geometry_loss_weight
            * losses.non_profile_geometry
            + self.model_config.profile_family_loss_weight
            * losses.profile_family
            + self.model_config.profile_parameter_loss_weight
            * losses.profile_parameter
            + self.model_config.edge_presence_loss_weight
            * losses.edge_presence
            + self.model_config.edge_type_loss_weight * losses.edge_type
            + self.model_config.operation_pointer_loss_weight
            * losses.operation_pointer
            + self.model_config.vq_loss_weight * losses.vq_commitment
        )
        torch.testing.assert_close(
            losses.total, weighted_total, rtol=1e-6, atol=1e-7
        )
        family_total_gradient = torch.autograd.grad(
            losses.total,
            output.profile_family_logits,
            retain_graph=True,
        )[0]
        parameter_total_gradient = torch.autograd.grad(
            losses.total,
            output.raw_profile_parameters,
            retain_graph=True,
        )[0]
        self.assertTrue(torch.isfinite(family_total_gradient).all())
        self.assertTrue(torch.isfinite(parameter_total_gradient).all())
        self.assertGreater(
            float(family_total_gradient.norm().item()), 0.0
        )
        self.assertGreater(
            float(parameter_total_gradient.norm().item()), 0.0
        )
        optimized = tuple(
            parameter
            for group in self.optimizer.param_groups
            for parameter in group["params"]
        )
        self.assertEqual(
            sum(
                parameter is self.model.profile_heads.family_head.weight
                for parameter in optimized
            ),
            1,
        )
        self.assertEqual(
            sum(
                parameter is self.model.profile_heads.parameter_head.weight
                for parameter in optimized
            ),
            1,
        )
        for name, expected in initial_ema.items():
            torch.testing.assert_close(
                expected,
                self.model.vq.state_dict()[name],
                rtol=0.0,
                atol=0.0,
            )
        del output, losses, weighted_total

        family_before = (
            self.model.profile_heads.family_head.weight.detach().clone()
        )
        parameter_before = (
            self.model.profile_heads.parameter_head.weight.detach().clone()
        )
        family_gradient_norms = []
        parameter_gradient_norms = []
        history = []
        for step in range(1, 101):
            result = self._step(step)
            history.append(result.metrics)
            family_gradient_norms.append(float(
                self.model.profile_heads.family_head.weight.grad.norm().item()
            ))
            parameter_gradient_norms.append(float(
                self.model.profile_heads.parameter_head.weight.grad.norm().item()
            ))
        self.assertTrue(all(
            math.isfinite(value) and value > 0.0
            for value in family_gradient_norms
        ))
        self.assertTrue(all(
            math.isfinite(value) and value > 0.0
            for value in parameter_gradient_norms
        ))
        self.assertFalse(torch.equal(
            family_before, self.model.profile_heads.family_head.weight
        ))
        self.assertFalse(torch.equal(
            parameter_before, self.model.profile_heads.parameter_head.weight
        ))
        self.assertTrue(all(
            math.isfinite(float(value))
            for row in history
            for value in row.values()
        ))
        self.assertTrue(any(
            not torch.equal(value, self.model.vq.state_dict()[name])
            for name, value in initial_ema.items()
        ))

        final_ema = {
            name: value.clone()
            for name, value in self.model.vq.state_dict().items()
        }
        final = constrained_v2_evaluation_snapshot(
            self.model,
            self.batch,
            self.model_config,
            self.training_config,
            torch.device("cpu"),
        )
        self.assertTrue(self.model.training)
        for name, expected in final_ema.items():
            torch.testing.assert_close(
                expected,
                self.model.vq.state_dict()[name],
                rtol=0.0,
                atol=0.0,
            )
        for name in (
            "total_loss",
            "profile_family_loss",
            "profile_parameter_loss",
        ):
            self.assertLess(
                final.metrics[name],
                initial.metrics[name],
                "{}: initial={!r}, final={!r}".format(
                    name, initial.metrics[name], final.metrics[name]
                ),
            )

    def test_nonfinite_loss_gradient_parameter_and_optimizer_failures(self):
        with mock.patch(
            "prototype.flat_baseline.constrained_v2_training."
            "constrained_profile_v2_loss"
        ) as loss_function:
            reference = self._real_loss()
            loss_function.return_value = replace(
                reference, total=reference.total * math.nan
            )
            with self.assertRaisesRegex(RuntimeError, "nonfinite"):
                self._step()

        parameter = self.model.profile_heads.family_head.weight
        hook = parameter.register_hook(
            lambda gradient: gradient * math.nan
        )
        try:
            with self.assertRaisesRegex(RuntimeError, "nonfinite"):
                self._step()
        finally:
            hook.remove()

        original_step = self.optimizer.step
        def corrupt_parameter(*args, **kwargs):
            result = original_step(*args, **kwargs)
            with torch.no_grad():
                parameter.reshape(-1)[0] = math.inf
            return result
        with mock.patch.object(
            self.optimizer, "step", side_effect=corrupt_parameter
        ):
            with self.assertRaisesRegex(RuntimeError, "nonfinite"):
                self._step()

        torch.manual_seed(self.training_config.seed)
        model = ConstrainedProfileV2Model(self.model_config)
        optimizer = build_v2_optimizer(model, self.training_config)
        original_step = optimizer.step

        def corrupt_optimizer(*args, **kwargs):
            result = original_step(*args, **kwargs)
            state = next(iter(optimizer.state.values()))
            tensor = next(
                value for value in state.values()
                if torch.is_tensor(value) and value.is_floating_point()
            )
            tensor.reshape(-1)[0] = math.inf
            return result

        with mock.patch.object(
            optimizer, "step", side_effect=corrupt_optimizer
        ):
            with self.assertRaisesRegex(
                RuntimeError, "nonfinite_optimizer_state"
            ):
                constrained_v2_training_step(
                    model,
                    optimizer,
                    self.batch,
                    self.model_config,
                    self.training_config,
                    torch.device("cpu"),
                    global_step=1,
                    examples_processed=len(self.batch.family_ids),
                )

    def test_fixed_seed_produces_deterministic_one_step(self):
        models = []
        results = []
        for _ in range(2):
            torch.manual_seed(self.training_config.seed)
            model = ConstrainedProfileV2Model(self.model_config)
            optimizer = build_v2_optimizer(
                model, self.training_config
            )
            result = constrained_v2_training_step(
                model,
                optimizer,
                self.batch,
                self.model_config,
                self.training_config,
                torch.device("cpu"),
                global_step=1,
                examples_processed=len(self.batch.family_ids),
            )
            models.append(model)
            results.append(result)
        self.assertEqual(results[0].metrics, results[1].metrics)
        for name, value in models[0].state_dict().items():
            torch.testing.assert_close(
                value, models[1].state_dict()[name], rtol=0.0, atol=0.0
            )

    def _real_loss(self):
        from prototype.constrained_profile_decoder import (
            profile_targets_for_loss,
        )
        from prototype.flat_baseline.constrained_v2_losses import (
            constrained_profile_v2_loss,
        )
        inputs = self.batch.to_torch(torch)
        target = self.batch.target.to_torch(torch)
        profiles = profile_targets_for_loss(
            self.batch.target, inputs["geometry"]
        )
        output = self.model(
            target=target, profile_targets=profiles, **inputs
        )
        return constrained_profile_v2_loss(
            output, target, profiles, self.model_config
        )

    def test_checkpoint_round_trip_strictness_rng_vq_and_collision(self):
        self._step()
        payload = v2_checkpoint_payload(
            self.model,
            self.optimizer,
            self.model_config,
            self.training_config,
            self.selection,
            global_step=1,
            source_provenance={"git_commit": "unit", "git_dirty": False},
        )
        self.assertEqual(set(payload), V2_CHECKPOINT_FIELDS)
        self.assertIn("rng_state", payload)
        self.assertEqual(payload["checkpoint_version"], 2)
        self.assertEqual(
            payload["selected_tiny_overfit_ids"],
            list(self.selection.selected_example_ids),
        )
        path = Path(self.temporary.name) / "v2.pt"
        save_v2_checkpoint(path, payload)
        with self.assertRaisesRegex(RuntimeError, "collision"):
            save_v2_checkpoint(path, payload)
        reloaded = ConstrainedProfileV2Model(self.model_config)
        optimizer = build_v2_optimizer(
            reloaded, self.training_config
        )
        ema_before = {
            name: value.clone()
            for name, value in self.model.vq.state_dict().items()
        }
        restored = load_v2_checkpoint(
            path,
            reloaded,
            optimizer,
            self.model_config,
            self.training_config,
            self.selection,
            map_location=torch.device("cpu"),
        )
        self.assertEqual(restored["global_step"], 1)
        for name, value in self.model.state_dict().items():
            torch.testing.assert_close(
                value, reloaded.state_dict()[name], rtol=0.0, atol=0.0
            )
        reloaded.eval()
        with torch.no_grad():
            self._forward(reloaded)
        for name, expected in ema_before.items():
            torch.testing.assert_close(
                expected,
                reloaded.vq.state_dict()[name],
                rtol=0.0,
                atol=0.0,
            )

        v1 = dict(payload)
        v1["checkpoint_version"] = 1
        v1_path = Path(self.temporary.name) / "v1.pt"
        torch.save(v1, v1_path)
        with self.assertRaisesRegex(RuntimeError, "version"):
            load_v2_checkpoint(
                v1_path,
                reloaded,
                optimizer,
                self.model_config,
                self.training_config,
                self.selection,
                map_location=torch.device("cpu"),
            )

    def _forward(self, model):
        from prototype.constrained_profile_decoder import (
            profile_targets_for_loss,
        )
        inputs = self.batch.to_torch(torch)
        target = self.batch.target.to_torch(torch)
        profiles = profile_targets_for_loss(
            self.batch.target, inputs["geometry"]
        )
        return model(target=target, profile_targets=profiles, **inputs)


if __name__ == "__main__":
    unittest.main()
