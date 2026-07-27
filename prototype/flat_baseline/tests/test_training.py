"""CPU optimization, EMA, logging, checkpointing, and resume tests."""

from __future__ import annotations

import copy
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

try:
    import torch
except ImportError:
    torch = None

from prototype.controlled_data.builders import build_history
from prototype.controlled_data.identity import source_family_id
from prototype.model_data.tests.fixtures import source, write_physical_corpus
from prototype.representation.model import GeometryEncoding

if torch is not None:
    from prototype.flat_baseline.checkpointing import (
        CheckpointError,
        checkpoint_payload,
        load_checkpoint,
        save_checkpoint,
        validated_checkpoint,
    )
    from prototype.flat_baseline.config import FlatBaselineConfig
    from prototype.flat_baseline.data import (
        TrainingDataError,
        load_training_data,
        make_data_loader as real_make_data_loader,
    )
    from prototype.flat_baseline.losses import (
        FlatMixedVQLoss,
        flat_mixed_vq_loss,
    )
    from prototype.flat_baseline.model import FlatMixedVQModel
    from prototype.flat_baseline.run_logging import JsonlLogger
    from prototype.flat_baseline.training import (
        _MetricAccumulator,
        TrainingError,
        evaluate_epoch,
        seed_everything,
        train_epoch,
        train_run,
    )
    from prototype.flat_baseline.training_config import TrainingConfig
    from prototype.model_data.batching import collate_flat


TORCH_REASON = "real PyTorch execution is deferred to the Adroit environment"


def _write_partitioned_corpus(root):
    sources = (source("E"), source("R"), source("ER"), source("EE"))
    family_ids = tuple(
        source_family_id(build_history(item, GeometryEncoding.CONTINUOUS))
        for item in sources
    )
    partitions = {
        family_ids[0]: "train",
        family_ids[1]: "train",
        family_ids[2]: "validation",
        family_ids[3]: "validation",
    }
    write_physical_corpus(root, sources, partitions=partitions)


def _model_config():
    return FlatBaselineConfig(
        model_dim=16,
        num_heads=2,
        feedforward_dim=24,
        encoder_layers=1,
        decoder_layers=1,
        dropout=0.0,
        max_nodes=12,
        max_operations=2,
        latent_tokens=1,
        codebook_size=8,
        codebook_dim=8,
        edge_pair_dim=12,
    )


def _training_config(output_dir, **changes):
    values = {
        "seed": 17,
        "epochs": 1,
        "batch_size": 2,
        "learning_rate": 0.01,
        "weight_decay": 0.0,
        "gradient_clip_norm": 1.0,
        "validation_interval": 1,
        "checkpoint_interval": 1,
        "dataloader_workers": 0,
        "device": "cpu",
        "output_dir": str(output_dir),
        "checkpoint_selection_metric": "total",
    }
    values.update(changes)
    return TrainingConfig(**values)


def _summary(total):
    metrics = {
        "total": total,
        "node_type": total,
        "categorical_attributes": total,
        "geometry": total,
        "edge_presence": total,
        "edge_type": total,
        "operation_pointer": total,
        "vq_commitment": total,
    }
    return {
        "number_of_examples": 1,
        "metrics": metrics,
        "codebook": {
            "active_code_count": 1.0,
            "codebook_perplexity": 1.0,
            "codebook_utilization": 0.125,
        },
    }


def _read_jsonl(path):
    return tuple(
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
    )


def _synthetic_loss(value, count=1):
    per_example = {
        name: value.reshape(1).repeat(count)
        for name in (
            "total",
            "node_type",
            "categorical_attributes",
            "geometry",
            "edge_presence",
            "edge_type",
            "operation_pointer",
            "vq_commitment",
        )
    }
    return FlatMixedVQLoss(
        value,
        value,
        value,
        value,
        value,
        value,
        value,
        value,
        per_example,
    )


def _assert_nested_tensors_equal(test_case, left, right):
    if torch.is_tensor(left):
        torch.testing.assert_close(left, right, rtol=0.0, atol=0.0)
    elif isinstance(left, dict):
        test_case.assertEqual(set(left), set(right))
        for key in left:
            _assert_nested_tensors_equal(test_case, left[key], right[key])
    elif isinstance(left, (list, tuple)):
        test_case.assertEqual(len(left), len(right))
        for left_item, right_item in zip(left, right):
            _assert_nested_tensors_equal(test_case, left_item, right_item)
    else:
        test_case.assertEqual(left, right)


@unittest.skipUnless(torch is not None, TORCH_REASON)
class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.corpus = Path(self.temporary.name) / "corpus"
        _write_partitioned_corpus(self.corpus)
        self.data = load_training_data(self.corpus, "iid")
        self.model_config = _model_config()
        self.training_config = _training_config(
            Path(self.temporary.name) / "run"
        )

    def _model_optimizer(self):
        model = FlatMixedVQModel(self.model_config)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay,
        )
        return model, optimizer

    def _symlink(self, target, link):
        try:
            os.symlink(str(target), str(link))
        except (OSError, NotImplementedError) as exc:
            self.skipTest("symbolic links are unavailable: {}".format(exc))

    def test_seeded_initialization_is_deterministic(self):
        seed_everything(9)
        first = FlatMixedVQModel(self.model_config)
        first_state = {
            name: value.detach().clone()
            for name, value in first.state_dict().items()
        }
        seed_everything(9)
        second = FlatMixedVQModel(self.model_config)
        for name, value in second.state_dict().items():
            torch.testing.assert_close(
                value, first_state[name], rtol=0.0, atol=0.0
            )

    def test_train_kmeans_resume_does_not_rerun_or_change_initialization(self):
        first_config = replace(
            self.training_config,
            vq_init="train-kmeans",
            output_dir=str(Path(self.temporary.name) / "treatment"),
        )

        def trained(
            model,
            optimizer,
            examples,
            model_config,
            training_config,
            device,
            epoch,
            global_step,
            torch_module,
            instrumentation=None,
        ):
            del (
                model,
                optimizer,
                examples,
                model_config,
                training_config,
                device,
                epoch,
                torch_module,
                instrumentation,
            )
            return _summary(1.0), global_step + 1

        with mock.patch(
            "prototype.flat_baseline.vq_initialization."
            "initialize_train_codebook",
            return_value={
                "mode": "train-kmeans",
                "method": "seeded-kmeans++-fixed-lloyd",
                "seed": first_config.seed,
                "pseudo_count_policy": (
                    "cluster-proportions-total-codebook-size"
                ),
                "report_sha256": "a" * 64,
            },
        ) as initialize, mock.patch(
            "prototype.flat_baseline.training.train_epoch",
            side_effect=trained,
        ), mock.patch(
            "prototype.flat_baseline.training.evaluate_epoch",
            return_value=_summary(1.0),
        ):
            first = train_run(
                self.corpus,
                model_config=self.model_config,
                training_config=first_config,
            )
        initialize.assert_called_once()
        treatment_checkpoint = torch.load(
            first.last_checkpoint, map_location="cpu"
        )
        self.assertEqual(
            treatment_checkpoint["training_config"]["vq_init"],
            "train-kmeans",
        )
        self.assertEqual(
            treatment_checkpoint["training_config"]["seed"],
            first_config.seed,
        )
        self.assertEqual(
            treatment_checkpoint["data_state"][
                "initialization_report_sha256"
            ],
            "a" * 64,
        )
        resumed_config = replace(first_config, epochs=2)
        with mock.patch(
            "prototype.flat_baseline.vq_initialization."
            "initialize_train_codebook"
        ) as initialize, mock.patch(
            "prototype.flat_baseline.training.train_epoch",
            side_effect=trained,
        ), mock.patch(
            "prototype.flat_baseline.training.evaluate_epoch",
            return_value=_summary(1.0),
        ):
            train_run(
                self.corpus,
                model_config=self.model_config,
                training_config=resumed_config,
                resume_checkpoint=first.last_checkpoint,
            )
        initialize.assert_not_called()
        with self.assertRaises(CheckpointError) as captured:
            validated_checkpoint(
                first.last_checkpoint,
                self.model_config,
                replace(resumed_config, vq_init="normal"),
                torch,
            )
        self.assertEqual(
            captured.exception.code, "incompatible_training_config"
        )

    def test_treatment_initialization_precedes_optimizer_creation(self):
        config = replace(
            self.training_config,
            vq_init="train-kmeans",
            output_dir=str(Path(self.temporary.name) / "ordering"),
        )
        events = []

        def initialize(*args, **kwargs):
            del args, kwargs
            events.append("initialize")
            return {
                "mode": "train-kmeans",
                "method": "seeded-kmeans++-fixed-lloyd",
                "seed": config.seed,
                "pseudo_count_policy": (
                    "cluster-proportions-total-codebook-size"
                ),
                "report_sha256": "b" * 64,
            }

        real_adamw = torch.optim.AdamW

        def optimizer(*args, **kwargs):
            events.append("optimizer")
            return real_adamw(*args, **kwargs)

        with mock.patch(
            "prototype.flat_baseline.vq_initialization."
            "initialize_train_codebook",
            side_effect=initialize,
        ), mock.patch(
            "prototype.flat_baseline.training.torch.optim.AdamW",
            side_effect=optimizer,
        ), mock.patch(
            "prototype.flat_baseline.training.train_epoch",
            return_value=(_summary(1.0), 1),
        ), mock.patch(
            "prototype.flat_baseline.training.evaluate_epoch",
            return_value=_summary(1.0),
        ):
            train_run(
                self.corpus,
                model_config=self.model_config,
                training_config=config,
            )
        self.assertEqual(events[:2], ["initialize", "optimizer"])

    def test_default_train_run_does_not_invoke_treatment_initializer(self):
        config = replace(
            self.training_config,
            output_dir=str(Path(self.temporary.name) / "normal"),
        )
        with mock.patch(
            "prototype.flat_baseline.vq_initialization."
            "initialize_train_codebook"
        ) as initialize, mock.patch(
            "prototype.flat_baseline.training.train_epoch",
            return_value=(_summary(1.0), 1),
        ), mock.patch(
            "prototype.flat_baseline.training.evaluate_epoch",
            return_value=_summary(1.0),
        ):
            train_run(
                self.corpus,
                model_config=self.model_config,
                training_config=config,
            )
        initialize.assert_not_called()

    def test_instrumented_training_is_state_optimizer_and_rng_neutral(self):
        from prototype.flat_baseline.vq_pilot import PilotInstrumentation

        model_config = replace(self.model_config, dropout=0.2)
        seed_everything(501)
        plain_model = FlatMixedVQModel(model_config)
        seed_everything(501)
        observed_model = FlatMixedVQModel(model_config)
        plain_optimizer = torch.optim.AdamW(
            plain_model.parameters(),
            lr=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay,
        )
        observed_optimizer = torch.optim.AdamW(
            observed_model.parameters(),
            lr=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay,
        )
        seed_everything(777)
        plain_summary, plain_step = train_epoch(
            plain_model,
            plain_optimizer,
            self.data.train_examples,
            model_config,
            self.training_config,
            torch.device("cpu"),
            1,
            0,
            torch,
        )
        plain_rng = torch.get_rng_state().clone()
        seed_everything(777)
        observed_summary, observed_step = train_epoch(
            observed_model,
            observed_optimizer,
            self.data.train_examples,
            model_config,
            self.training_config,
            torch.device("cpu"),
            1,
            0,
            torch,
            PilotInstrumentation(None, torch),
        )
        observed_rng = torch.get_rng_state().clone()
        self.assertEqual(plain_step, observed_step)
        self.assertEqual(
            plain_summary["number_of_examples"],
            observed_summary["number_of_examples"],
        )
        self.assertEqual(
            plain_summary["metrics"], observed_summary["metrics"]
        )
        self.assertEqual(
            plain_summary["codebook"], observed_summary["codebook"]
        )
        self.assertTrue(torch.equal(plain_rng, observed_rng))
        _assert_nested_tensors_equal(
            self,
            plain_model.state_dict(),
            observed_model.state_dict(),
        )
        _assert_nested_tensors_equal(
            self,
            plain_optimizer.state_dict(),
            observed_optimizer.state_dict(),
        )

    def test_train_run_rejects_invalid_partitions_before_model_creation(self):
        invalid = (
            ("test", "validation", "invalid_training_partition"),
            ("train", "test", "invalid_validation_partition"),
            (
                "train",
                "secondary_systematic_validation",
                "invalid_validation_partition",
            ),
            ("unknown", "validation", "invalid_training_partition"),
        )
        with mock.patch(
            "prototype.flat_baseline.training.FlatMixedVQModel"
        ) as constructor:
            for train_partition, validation_partition, code in invalid:
                with self.subTest(
                    train=train_partition, validation=validation_partition
                ):
                    with self.assertRaises(TrainingDataError) as captured:
                        train_run(
                            self.corpus,
                            train_partition=train_partition,
                            validation_partition=validation_partition,
                            model_config=self.model_config,
                            training_config=self.training_config,
                        )
                    self.assertEqual(captured.exception.code, code)
            constructor.assert_not_called()

    def test_normal_training_is_finite_and_updates_ema(self):
        seed_everything(self.training_config.seed)
        model, optimizer = self._model_optimizer()
        before = model.vq.embedding.clone()
        summary, step = train_epoch(
            model,
            optimizer,
            self.data.train_examples,
            self.model_config,
            self.training_config,
            torch.device("cpu"),
            1,
            0,
        )
        self.assertEqual(step, 1)
        self.assertTrue(
            all(math.isfinite(value) for value in summary["metrics"].values())
        )
        self.assertFalse(torch.equal(before, model.vq.embedding))

    def test_nonfinite_gradient_introduced_by_clipping_is_rejected(self):
        model, optimizer = self._model_optimizer()

        def corrupt_gradient(parameters, maximum):
            del maximum
            for parameter in parameters:
                if parameter.grad is not None:
                    parameter.grad.reshape(-1)[0] = float("nan")
                    break
            return torch.tensor(1.0)

        with mock.patch(
            "prototype.flat_baseline.training."
            "torch.nn.utils.clip_grad_norm_",
            side_effect=corrupt_gradient,
        ):
            with self.assertRaises(TrainingError) as captured:
                train_epoch(
                    model,
                    optimizer,
                    self.data.train_examples,
                    self.model_config,
                    self.training_config,
                    torch.device("cpu"),
                    1,
                    0,
                )
        self.assertEqual(
            captured.exception.code, "nonfinite_postclip_gradient"
        )

    def test_nonfinite_parameter_after_step_is_rejected(self):
        model, optimizer = self._model_optimizer()
        original_step = optimizer.step

        def corrupt_parameter():
            result = original_step()
            with torch.no_grad():
                next(model.parameters()).reshape(-1)[0] = float("inf")
            return result

        with mock.patch.object(
            optimizer, "step", side_effect=corrupt_parameter
        ):
            with self.assertRaises(TrainingError) as captured:
                train_epoch(
                    model,
                    optimizer,
                    self.data.train_examples,
                    self.model_config,
                    self.training_config,
                    torch.device("cpu"),
                    1,
                    0,
                )
        self.assertEqual(captured.exception.code, "nonfinite_model_state")

    def test_nonfinite_optimizer_state_after_step_is_rejected(self):
        model, optimizer = self._model_optimizer()
        original_step = optimizer.step

        def corrupt_optimizer():
            result = original_step()
            state = next(iter(optimizer.state.values()))
            state["exp_avg"].reshape(-1)[0] = float("inf")
            return result

        with mock.patch.object(
            optimizer, "step", side_effect=corrupt_optimizer
        ):
            with self.assertRaises(TrainingError) as captured:
                train_epoch(
                    model,
                    optimizer,
                    self.data.train_examples,
                    self.model_config,
                    self.training_config,
                    torch.device("cpu"),
                    1,
                    0,
                )
        self.assertEqual(
            captured.exception.code, "nonfinite_optimizer_state"
        )

    def test_epoch_codebook_statistics_aggregate_disjoint_assignments(self):
        value = torch.tensor(1.0)
        losses = _synthetic_loss(value)
        accumulator = _MetricAccumulator()
        accumulator.add(
            losses,
            SimpleNamespace(
                assignment_counts=torch.tensor([2, 0, 0, 0])
            ),
            1,
        )
        accumulator.add(
            losses,
            SimpleNamespace(
                assignment_counts=torch.tensor([0, 2, 0, 0])
            ),
            1,
        )
        codebook = accumulator.summary()["codebook"]
        self.assertEqual(codebook["active_code_count"], 2)
        self.assertEqual(codebook["codebook_utilization"], 0.5)
        self.assertAlmostEqual(codebook["codebook_perplexity"], 2.0)
        zero_accumulator = _MetricAccumulator()
        zero_accumulator.add(
            losses,
            SimpleNamespace(
                assignment_counts=torch.tensor([0, 0, 0, 0])
            ),
            1,
        )
        zero_codebook = zero_accumulator.summary()["codebook"]
        self.assertEqual(zero_codebook["active_code_count"], 0)
        self.assertEqual(zero_codebook["codebook_utilization"], 0.0)
        self.assertEqual(zero_codebook["codebook_perplexity"], 0.0)
        with self.assertRaises(TrainingError) as captured:
            accumulator.add(
                losses,
                SimpleNamespace(assignment_counts=torch.tensor([1, 0])),
                1,
            )
        self.assertEqual(
            captured.exception.code, "inconsistent_codebook_width"
        )
        invalid_counts = (
            torch.tensor([1.0, 0.0]),
            torch.tensor([True, False]),
            torch.tensor([1.0 + 0.0j, 0.0 + 0.0j]),
            torch.tensor([-1, 0], dtype=torch.long),
            torch.tensor([[1, 0]], dtype=torch.long),
            torch.empty(0, dtype=torch.long),
            torch.tensor([1, 0], dtype=torch.int32),
        )
        for counts in invalid_counts:
            with self.subTest(dtype=counts.dtype, shape=tuple(counts.shape)):
                current = _MetricAccumulator()
                with self.assertRaises(TrainingError) as invalid:
                    current.add(
                        losses,
                        SimpleNamespace(assignment_counts=counts),
                        1,
                    )
                self.assertEqual(
                    invalid.exception.code, "invalid_assignment_counts"
                )

    def test_validation_does_not_update_ema_and_restores_training_mode(self):
        seed_everything(self.training_config.seed)
        model, optimizer = self._model_optimizer()
        train_epoch(
            model,
            optimizer,
            self.data.train_examples,
            self.model_config,
            self.training_config,
            torch.device("cpu"),
            1,
            0,
        )
        self.assertTrue(optimizer.state)
        model.train()
        before = {
            name: value.clone()
            for name, value in model.state_dict().items()
        }
        optimizer_before = copy.deepcopy(optimizer.state_dict())
        evaluate_epoch(
            model,
            self.data.validation_examples,
            self.model_config,
            self.training_config,
            torch.device("cpu"),
        )
        self.assertTrue(model.training)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(
                value, before[name], rtol=0.0, atol=0.0
            )
        _assert_nested_tensors_equal(
            self, optimizer.state_dict(), optimizer_before
        )

    def test_per_example_losses_are_batch_and_order_invariant(self):
        seed_everything(41)
        model, _ = self._model_optimizer()
        examples = (
            self.data.train_examples + self.data.validation_examples
        )
        summaries = []
        for batch_size in (1, 2, len(examples)):
            config = replace(self.training_config, batch_size=batch_size)
            summaries.append(
                evaluate_epoch(
                    model,
                    examples,
                    self.model_config,
                    config,
                    torch.device("cpu"),
                )
            )
        reversed_summary = evaluate_epoch(
            model,
            tuple(reversed(examples)),
            self.model_config,
            replace(self.training_config, batch_size=2),
            torch.device("cpu"),
        )
        for name in summaries[0]["metrics"]:
            for summary in summaries[1:] + [reversed_summary]:
                self.assertAlmostEqual(
                    summaries[0]["metrics"][name],
                    summary["metrics"][name],
                    places=6,
                )

    def test_scalar_losses_are_means_of_exposed_example_losses(self):
        seed_everything(43)
        model, _ = self._model_optimizer()
        examples = (
            self.data.train_examples[:1]
            + self.data.validation_examples[:1]
        )
        batch = collate_flat(examples)
        inputs = batch.to_torch(torch)
        target = batch.target.to_torch(torch)
        model.eval()
        with torch.no_grad():
            output = model(target=target, **inputs)
            losses = flat_mixed_vq_loss(
                output, target, self.model_config
            )
        self.assertEqual(
            tuple(target["operation_mask"].sum(dim=1).tolist()), (1, 2)
        )
        for name, scalar in losses.as_dict().items():
            self.assertEqual(losses.per_example[name].shape, (2,))
            torch.testing.assert_close(
                scalar,
                losses.per_example[name].mean(),
                rtol=0.0,
                atol=1e-7,
            )

    def test_zero_eligible_losses_are_finite(self):
        logits = torch.zeros(1, 1, 2)
        output = SimpleNamespace(
            node_type_logits=logits,
            categorical_logits=tuple(logits for _ in range(9)),
            geometry=torch.zeros(1, 1, 39),
            edge_presence_logits=torch.zeros(1, 1, 1),
            edge_type_logits=torch.zeros(1, 1, 1, 2),
            operation_pointer_logits=torch.zeros(1, 0, 1),
            vq_per_example_loss=torch.zeros(1),
        )
        target = {
            "node_mask": torch.zeros(1, 1, dtype=torch.bool),
            "node_type_ids": torch.zeros(1, 1, dtype=torch.long),
            "categorical_attributes": torch.zeros(
                1, 1, 9, dtype=torch.long
            ),
            "geometry": torch.zeros(1, 1, 39),
            "geometry_mask": torch.zeros(1, 1, 39, dtype=torch.bool),
            "edge_index": torch.empty(2, 0, dtype=torch.long),
            "edge_type_ids": torch.empty(0, dtype=torch.long),
            "edge_offsets": torch.tensor([0, 0], dtype=torch.long),
            "operation_sequence": torch.empty(1, 0, dtype=torch.long),
            "operation_mask": torch.empty(1, 0, dtype=torch.bool),
        }
        losses = flat_mixed_vq_loss(output, target, self.model_config)
        for values in losses.per_example.values():
            self.assertTrue(torch.isfinite(values).all())
            torch.testing.assert_close(
                values, torch.zeros_like(values), rtol=0.0, atol=0.0
            )

    def test_ema_active_one_family_overfit_decreases_validation_loss(self):
        seed_everything(31)
        model, optimizer = self._model_optimizer()
        example = self.data.train_examples[:1]
        before = evaluate_epoch(
            model,
            example,
            self.model_config,
            self.training_config,
            torch.device("cpu"),
        )["metrics"]["total"]
        step = 0
        for epoch in range(1, 31):
            _, step = train_epoch(
                model,
                optimizer,
                example,
                self.model_config,
                self.training_config,
                torch.device("cpu"),
                epoch,
                step,
            )
        after = evaluate_epoch(
            model,
            example,
            self.model_config,
            self.training_config,
            torch.device("cpu"),
        )["metrics"]["total"]
        self.assertTrue(model.vq.training)
        self.assertLess(after, before)

    def test_jsonl_named_metrics_are_canonical(self):
        path = Path(self.temporary.name) / "events.jsonl"
        logger = JsonlLogger(path)
        logger.write(
            {
                "event": "train_epoch",
                "metrics": _summary(1.25)["metrics"],
                "codebook": _summary(1.25)["codebook"],
            }
        )
        payload = path.read_text(encoding="utf-8")
        record = json.loads(payload)
        self.assertEqual(record["event"], "train_epoch")
        self.assertEqual(
            set(record["metrics"]),
            {
                "total",
                "node_type",
                "categorical_attributes",
                "geometry",
                "edge_presence",
                "edge_type",
                "operation_pointer",
                "vq_commitment",
            },
        )
        self.assertEqual(
            payload.rstrip("\n"),
            json.dumps(
                record,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ),
        )

    def test_checkpoint_round_trip_restores_optimizer_and_progress(self):
        seed_everything(self.training_config.seed)
        model, optimizer = self._model_optimizer()
        _, step = train_epoch(
            model,
            optimizer,
            self.data.train_examples,
            self.model_config,
            self.training_config,
            torch.device("cpu"),
            1,
            0,
        )
        payload = checkpoint_payload(
            model,
            optimizer,
            1,
            step,
            self.model_config,
            self.training_config,
            2.5,
            torch,
        )
        path = Path(self.temporary.name) / "roundtrip.pt"
        save_checkpoint(path, payload, torch)
        restored_model, restored_optimizer = self._model_optimizer()
        restored = load_checkpoint(
            path,
            restored_model,
            restored_optimizer,
            self.model_config,
            self.training_config,
            torch,
            torch.device("cpu"),
        )
        self.assertEqual(restored["epoch"], 1)
        self.assertEqual(restored["global_step"], step)
        self.assertTrue(restored_optimizer.state)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(
                restored_model.state_dict()[name], value, rtol=0.0, atol=0.0
            )

    def test_incompatible_resume_configuration_is_rejected(self):
        model, optimizer = self._model_optimizer()
        payload = checkpoint_payload(
            model,
            optimizer,
            1,
            0,
            self.model_config,
            self.training_config,
            1.0,
            torch,
        )
        path = Path(self.temporary.name) / "incompatible.pt"
        save_checkpoint(path, payload, torch)
        with self.assertRaises(CheckpointError) as captured:
            load_checkpoint(
                path,
                model,
                optimizer,
                self.model_config,
                replace(self.training_config, batch_size=99),
                torch,
                torch.device("cpu"),
            )
        self.assertEqual(
            captured.exception.code, "incompatible_training_config"
        )

    def test_older_normal_checkpoint_without_vq_init_remains_loadable(self):
        model, optimizer = self._model_optimizer()
        payload = checkpoint_payload(
            model,
            optimizer,
            1,
            0,
            self.model_config,
            self.training_config,
            1.0,
            torch,
        )
        payload["training_config"].pop("vq_init")
        path = Path(self.temporary.name) / "legacy-normal.pt"
        save_checkpoint(path, payload, torch)
        restored = validated_checkpoint(
            path,
            self.model_config,
            self.training_config,
            torch,
        )
        self.assertNotIn("vq_init", restored["training_config"])

    def test_checkpoint_progress_requires_nonnegative_actual_integers(self):
        invalid_values = (
            1.5,
            "1",
            True,
            -1,
            None,
            float("inf"),
            float("nan"),
        )
        model, optimizer = self._model_optimizer()
        for kind in ("last", "best"):
            for field in ("epoch", "global_step"):
                for case_index, value in enumerate(invalid_values):
                    with self.subTest(
                        kind=kind, field=field, value=repr(value)
                    ):
                        payload = checkpoint_payload(
                            model,
                            optimizer,
                            1,
                            1,
                            self.model_config,
                            self.training_config,
                            1.0,
                            torch,
                            checkpoint_kind=kind,
                        )
                        payload[field] = value
                        path = (
                            Path(self.temporary.name)
                            / "invalid_{}_{}_{}.pt".format(
                                kind, field, case_index
                            )
                        )
                        save_checkpoint(path, payload, torch)
                        with self.assertRaises(CheckpointError) as captured:
                            validated_checkpoint(
                                path,
                                self.model_config,
                                self.training_config,
                                torch,
                            )
                        self.assertEqual(
                            captured.exception.code,
                            "malformed_checkpoint",
                        )

    def test_resume_restores_epoch_global_step_and_optimizer_state(self):
        first = train_run(
            self.corpus,
            model_config=self.model_config,
            training_config=self.training_config,
        )
        resumed_config = replace(self.training_config, epochs=2)
        second = train_run(
            self.corpus,
            model_config=self.model_config,
            training_config=resumed_config,
            resume_checkpoint=first.last_checkpoint,
        )
        checkpoint = torch.load(
            second.last_checkpoint, map_location=torch.device("cpu")
        )
        self.assertEqual(second.completed_epoch, 2)
        self.assertGreater(second.global_step, first.global_step)
        self.assertEqual(checkpoint["global_step"], second.global_step)
        self.assertTrue(checkpoint["optimizer_state"]["state"])
        records = [
            json.loads(line)
            for line in (
                Path(self.training_config.output_dir) / "metrics.jsonl"
            ).read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual(
            sum(item["event"] == "run_metadata" for item in records), 2
        )

    def test_uninterrupted_and_resumed_training_match_with_dropout(self):
        model_config = replace(self.model_config, dropout=0.2)
        full_config = replace(
            self.training_config,
            epochs=2,
            output_dir=str(Path(self.temporary.name) / "full"),
        )
        partial_config = replace(
            full_config,
            epochs=1,
            output_dir=str(Path(self.temporary.name) / "resumed"),
        )
        epoch_two_orders = []

        def recording_loader(
            examples,
            batch_size,
            workers,
            seed,
            shuffle,
            torch_module,
        ):
            loader = real_make_data_loader(
                examples,
                batch_size,
                workers,
                seed,
                shuffle,
                torch_module,
            )
            if not shuffle or seed != full_config.seed + 2:
                return loader

            def batches():
                family_order = []
                for batch in loader:
                    family_order.extend(batch.family_ids)
                    yield batch
                epoch_two_orders.append(tuple(family_order))

            return batches()

        with mock.patch(
            "prototype.flat_baseline.training.make_data_loader",
            side_effect=recording_loader,
        ):
            full = train_run(
                self.corpus,
                model_config=model_config,
                training_config=full_config,
            )
            partial = train_run(
                self.corpus,
                model_config=model_config,
                training_config=partial_config,
            )
            resumed = train_run(
                self.corpus,
                model_config=model_config,
                training_config=replace(partial_config, epochs=2),
                resume_checkpoint=partial.last_checkpoint,
            )
        full_checkpoint = torch.load(full.last_checkpoint, map_location="cpu")
        resumed_checkpoint = torch.load(
            resumed.last_checkpoint, map_location="cpu"
        )
        self.assertEqual(full.completed_epoch, resumed.completed_epoch)
        self.assertEqual(full.global_step, resumed.global_step)
        full_training = full_checkpoint["training_config"]
        resumed_training = resumed_checkpoint["training_config"]
        full_comparable = dict(full_checkpoint)
        resumed_comparable = dict(resumed_checkpoint)
        full_comparable.pop("training_config")
        resumed_comparable.pop("training_config")
        _assert_nested_tensors_equal(
            self, full_comparable, resumed_comparable
        )
        self.assertEqual(
            TrainingConfig(**full_training).resume_signature(),
            TrainingConfig(**resumed_training).resume_signature(),
        )
        self.assertEqual(len(epoch_two_orders), 2)
        self.assertEqual(epoch_two_orders[0], epoch_two_orders[1])
        full_records = [
            item
            for item in _read_jsonl(full.metrics_path)
            if item["event"] in ("train_epoch", "validation_epoch")
        ]
        resumed_records = [
            item
            for item in _read_jsonl(resumed.metrics_path)
            if item["event"] in ("train_epoch", "validation_epoch")
        ]
        self.assertEqual(full_records, resumed_records)

    def test_resume_to_new_directory_preserves_prior_best_checkpoint(self):
        old_config = replace(
            self.training_config,
            output_dir=str(Path(self.temporary.name) / "old_run"),
        )
        with mock.patch(
            "prototype.flat_baseline.training.train_epoch",
            return_value=(_summary(0.1), 1),
        ), mock.patch(
            "prototype.flat_baseline.training.evaluate_epoch",
            return_value=_summary(2.0),
        ):
            first = train_run(
                self.corpus,
                model_config=self.model_config,
                training_config=old_config,
            )
        new_config = replace(
            old_config,
            epochs=2,
            output_dir=str(Path(self.temporary.name) / "new_run"),
        )
        with mock.patch(
            "prototype.flat_baseline.training.train_epoch",
            return_value=(_summary(0.01), 2),
        ), mock.patch(
            "prototype.flat_baseline.training.evaluate_epoch",
            return_value=_summary(3.0),
        ):
            resumed = train_run(
                self.corpus,
                model_config=self.model_config,
                training_config=new_config,
                resume_checkpoint=first.last_checkpoint,
            )
        best_path = Path(resumed.best_checkpoint)
        self.assertTrue(best_path.is_file())
        best = torch.load(str(best_path), map_location="cpu")
        self.assertEqual(best["epoch"], 1)
        self.assertEqual(best["best_validation_metric"], 2.0)

    def test_relocation_requires_the_associated_best_checkpoint(self):
        old_config = replace(
            self.training_config,
            output_dir=str(Path(self.temporary.name) / "missing_old"),
        )
        with mock.patch(
            "prototype.flat_baseline.training.train_epoch",
            return_value=(_summary(0.1), 1),
        ), mock.patch(
            "prototype.flat_baseline.training.evaluate_epoch",
            return_value=_summary(2.0),
        ):
            first = train_run(
                self.corpus,
                model_config=self.model_config,
                training_config=old_config,
            )
        Path(first.best_checkpoint).unlink()
        new_config = replace(
            old_config,
            epochs=2,
            output_dir=str(Path(self.temporary.name) / "missing_new"),
        )
        with self.assertRaises(TrainingError) as captured:
            train_run(
                self.corpus,
                model_config=self.model_config,
                training_config=new_config,
                resume_checkpoint=first.last_checkpoint,
            )
        self.assertEqual(captured.exception.code, "missing_best_checkpoint")

    def test_incompatible_associated_best_checkpoint_is_rejected(self):
        old_config = replace(
            self.training_config,
            output_dir=str(Path(self.temporary.name) / "incompatible_old"),
        )
        with mock.patch(
            "prototype.flat_baseline.training.train_epoch",
            return_value=(_summary(0.1), 1),
        ), mock.patch(
            "prototype.flat_baseline.training.evaluate_epoch",
            return_value=_summary(2.0),
        ):
            first = train_run(
                self.corpus,
                model_config=self.model_config,
                training_config=old_config,
            )
        best = torch.load(first.best_checkpoint, map_location="cpu")
        best["model_config"] = replace(
            self.model_config, feedforward_dim=25
        ).to_dict()
        save_checkpoint(first.best_checkpoint, best, torch)
        new_config = replace(
            old_config,
            epochs=2,
            output_dir=str(Path(self.temporary.name) / "incompatible_new"),
        )
        with self.assertRaises(CheckpointError) as captured:
            train_run(
                self.corpus,
                model_config=self.model_config,
                training_config=new_config,
                resume_checkpoint=first.last_checkpoint,
            )
        self.assertEqual(
            captured.exception.code, "incompatible_model_config"
        )

    def test_associated_best_from_later_progress_is_rejected(self):
        for field in ("epoch", "global_step"):
            with self.subTest(field=field):
                old_config = replace(
                    self.training_config,
                    output_dir=str(
                        Path(self.temporary.name) / ("progress_" + field)
                    ),
                )
                first = train_run(
                    self.corpus,
                    model_config=self.model_config,
                    training_config=old_config,
                )
                best = torch.load(first.best_checkpoint, map_location="cpu")
                best[field] = best[field] + 1
                save_checkpoint(first.best_checkpoint, best, torch)
                new_config = replace(
                    old_config,
                    epochs=2,
                    output_dir=str(
                        Path(self.temporary.name)
                        / ("progress_new_" + field)
                    ),
                )
                with self.assertRaises(CheckpointError) as captured:
                    train_run(
                        self.corpus,
                        model_config=self.model_config,
                        training_config=new_config,
                        resume_checkpoint=first.last_checkpoint,
                    )
                self.assertEqual(
                    captured.exception.code,
                    "inconsistent_best_checkpoint",
                )

    def test_direct_best_resume_same_and_relocated_preserve_state(self):
        old_config = replace(
            self.training_config,
            output_dir=str(Path(self.temporary.name) / "direct_old"),
        )
        first = train_run(
            self.corpus,
            model_config=self.model_config,
            training_config=old_config,
        )
        prior_best = torch.load(first.best_checkpoint, map_location="cpu")
        relocated_config = replace(
            old_config,
            epochs=2,
            output_dir=str(Path(self.temporary.name) / "direct_new"),
        )
        with mock.patch(
            "prototype.flat_baseline.training.train_epoch",
            return_value=(_summary(0.1), prior_best["global_step"] + 1),
        ), mock.patch(
            "prototype.flat_baseline.training.evaluate_epoch",
            return_value=_summary(
                float(prior_best["best_validation_metric"]) + 1.0
            ),
        ):
            relocated = train_run(
                self.corpus,
                model_config=self.model_config,
                training_config=relocated_config,
                resume_checkpoint=first.best_checkpoint,
            )
        relocated_best = torch.load(
            relocated.best_checkpoint, map_location="cpu"
        )
        _assert_nested_tensors_equal(self, prior_best, relocated_best)
        with mock.patch(
            "prototype.flat_baseline.training.train_epoch",
            return_value=(_summary(0.1), prior_best["global_step"] + 1),
        ), mock.patch(
            "prototype.flat_baseline.training.evaluate_epoch",
            return_value=_summary(
                float(prior_best["best_validation_metric"]) + 1.0
            ),
        ):
            same = train_run(
                self.corpus,
                model_config=self.model_config,
                training_config=replace(old_config, epochs=2),
                resume_checkpoint=first.best_checkpoint,
            )
        self.assertTrue(Path(same.best_checkpoint).is_file())
        same_best = torch.load(same.best_checkpoint, map_location="cpu")
        _assert_nested_tensors_equal(self, prior_best, same_best)
        self.assertEqual(same_best["checkpoint_kind"], "best")

    def test_best_checkpoint_is_selected_only_from_validation_metric(self):
        config = replace(self.training_config, epochs=2)
        with mock.patch(
            "prototype.flat_baseline.training.train_epoch",
            side_effect=((_summary(0.01), 1), (_summary(0.001), 2)),
        ), mock.patch(
            "prototype.flat_baseline.training.evaluate_epoch",
            side_effect=(_summary(2.0), _summary(3.0)),
        ):
            result = train_run(
                self.corpus,
                model_config=self.model_config,
                training_config=config,
            )
        best = torch.load(result.best_checkpoint, map_location="cpu")
        self.assertEqual(best["epoch"], 1)
        self.assertEqual(best["best_validation_metric"], 2.0)

    def test_best_selection_is_invariant_to_validation_batch_size(self):
        seed_everything(self.training_config.seed)
        reference_model, _ = self._model_optimizer()
        epoch_states = {
            1: copy.deepcopy(reference_model.state_dict()),
        }
        with torch.no_grad():
            bias = reference_model.node_type_head.bias
            bias.add_(
                torch.arange(
                    bias.numel(), dtype=bias.dtype, device=bias.device
                )
                * 0.25
            )
        epoch_states[2] = copy.deepcopy(reference_model.state_dict())

        def install_epoch_state(
            model,
            optimizer,
            examples,
            model_config,
            training_config,
            device,
            epoch,
            global_step,
            torch_module=torch,
        ):
            del (
                optimizer,
                examples,
                model_config,
                training_config,
                device,
                torch_module,
            )
            model.load_state_dict(epoch_states[epoch])
            return _summary(float(epoch)), global_step + 1

        results = []
        for batch_size in (1, 2):
            config = replace(
                self.training_config,
                epochs=2,
                batch_size=batch_size,
                output_dir=str(
                    Path(self.temporary.name)
                    / "best_batch_{}".format(batch_size)
                ),
            )
            with mock.patch(
                "prototype.flat_baseline.training.train_epoch",
                side_effect=install_epoch_state,
            ):
                result = train_run(
                    self.corpus,
                    model_config=self.model_config,
                    training_config=config,
                )
            validation = tuple(
                item
                for item in _read_jsonl(result.metrics_path)
                if item["event"] == "validation_epoch"
            )
            results.append(
                (
                    torch.load(result.best_checkpoint, map_location="cpu"),
                    validation,
                )
            )
        for _, validation in results:
            self.assertEqual(len(validation), 2)
            self.assertNotAlmostEqual(
                validation[0]["metrics"]["total"],
                validation[1]["metrics"]["total"],
                places=6,
            )
        for epoch_index in range(2):
            self.assertAlmostEqual(
                results[0][1][epoch_index]["metrics"]["total"],
                results[1][1][epoch_index]["metrics"]["total"],
                places=6,
            )
        expected_epoch = min(
            results[0][1],
            key=lambda item: item["metrics"]["total"],
        )["epoch"]
        self.assertEqual(
            tuple(item[0]["epoch"] for item in results),
            (expected_epoch, expected_epoch),
        )

    def test_nonfinite_loss_aborts_with_structured_error(self):
        model, optimizer = self._model_optimizer()

        def nonfinite_loss(output, target, config):
            del target, config
            value = output.vq_loss * 0.0 + float("nan")
            return _synthetic_loss(
                value, len(self.data.train_examples)
            )

        with mock.patch(
            "prototype.flat_baseline.training.flat_mixed_vq_loss",
            side_effect=nonfinite_loss,
        ):
            with self.assertRaises(TrainingError) as captured:
                train_epoch(
                    model,
                    optimizer,
                    self.data.train_examples,
                    self.model_config,
                    self.training_config,
                    torch.device("cpu"),
                    1,
                    0,
                )
        self.assertEqual(captured.exception.code, "nonfinite_loss")

    def test_run_abort_emits_structured_failure_event(self):
        with mock.patch(
            "prototype.flat_baseline.training.train_epoch",
            side_effect=TrainingError("nonfinite_loss", "synthetic failure"),
        ):
            with self.assertRaises(TrainingError):
                train_run(
                    self.corpus,
                    model_config=self.model_config,
                    training_config=self.training_config,
                )
        records = [
            json.loads(line)
            for line in (
                Path(self.training_config.output_dir) / "metrics.jsonl"
            ).read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual(records[-1]["event"], "failure")
        self.assertEqual(records[-1]["failure_code"], "nonfinite_loss")
        self.assertFalse(
            any(
                item["event"] in ("train_epoch", "checkpoint")
                for item in records
            )
        )

    def test_poststep_failure_emits_no_success_or_checkpoint(self):
        output = Path(self.temporary.name) / "poststep_failure"
        config = replace(self.training_config, output_dir=str(output))
        with mock.patch(
            "prototype.flat_baseline.training._require_finite_model_state",
            side_effect=TrainingError(
                "nonfinite_model_state", "synthetic post-step failure"
            ),
        ):
            with self.assertRaises(TrainingError):
                train_run(
                    self.corpus,
                    model_config=self.model_config,
                    training_config=config,
                )
        records = _read_jsonl(output / "metrics.jsonl")
        self.assertEqual(records[-1]["failure_code"], "nonfinite_model_state")
        self.assertFalse(
            any(
                item["event"] in ("train_epoch", "checkpoint")
                for item in records
            )
        )
        self.assertFalse((output / "last.pt").exists())
        self.assertFalse((output / "best.pt").exists())

    def test_fresh_run_rejects_managed_output_artifacts(self):
        for name in ("metrics.jsonl", "last.pt", "best.pt"):
            with self.subTest(name=name):
                output = Path(self.temporary.name) / ("fresh_" + name)
                output.mkdir(parents=True)
                existing = output / name
                existing.write_bytes(b"old run\n")
                with self.assertRaises(TrainingError) as captured:
                    train_run(
                        self.corpus,
                        model_config=self.model_config,
                        training_config=replace(
                            self.training_config, output_dir=str(output)
                        ),
                    )
                self.assertEqual(
                    captured.exception.code, "managed_output_collision"
                )
                self.assertEqual(existing.read_bytes(), b"old run\n")

    def test_relocation_rejects_occupied_destination_without_writes(self):
        first = train_run(
            self.corpus,
            model_config=self.model_config,
            training_config=self.training_config,
        )
        for name in ("metrics.jsonl", "last.pt", "best.pt"):
            with self.subTest(name=name):
                output = Path(self.temporary.name) / ("occupied_" + name)
                output.mkdir()
                existing = output / name
                existing.write_bytes(b"unrelated")
                config = replace(
                    self.training_config,
                    epochs=2,
                    output_dir=str(output),
                )
                with self.assertRaises(TrainingError) as captured:
                    train_run(
                        self.corpus,
                        model_config=self.model_config,
                        training_config=config,
                        resume_checkpoint=first.last_checkpoint,
                    )
                self.assertEqual(
                    captured.exception.code, "managed_output_collision"
                )
                self.assertEqual(existing.read_bytes(), b"unrelated")
                self.assertEqual(tuple(output.iterdir()), (existing,))

    def test_resolved_symlink_checkpoint_controls_resume_directory(self):
        for kind in ("last", "best"):
            with self.subTest(kind=kind):
                run_a_dir = (
                    Path(self.temporary.name) / "symlink_a_{}".format(kind)
                )
                run_b_dir = (
                    Path(self.temporary.name) / "symlink_b_{}".format(kind)
                )
                run_a = train_run(
                    self.corpus,
                    model_config=self.model_config,
                    training_config=replace(
                        self.training_config, output_dir=str(run_a_dir)
                    ),
                )
                run_b = train_run(
                    self.corpus,
                    model_config=self.model_config,
                    training_config=replace(
                        self.training_config, output_dir=str(run_b_dir)
                    ),
                )
                source = Path(
                    run_b.last_checkpoint
                    if kind == "last"
                    else run_b.best_checkpoint
                )
                link = run_a_dir / "external_{}.pt".format(kind)
                self._symlink(source, link)
                managed_before = {
                    name: (run_a_dir / name).read_bytes()
                    for name in ("metrics.jsonl", "last.pt", "best.pt")
                }
                with self.assertRaises(TrainingError) as captured:
                    train_run(
                        self.corpus,
                        model_config=self.model_config,
                        training_config=replace(
                            self.training_config,
                            epochs=2,
                            output_dir=str(run_a_dir),
                        ),
                        resume_checkpoint=link,
                    )
                self.assertEqual(
                    captured.exception.code, "managed_output_collision"
                )
                for name, payload in managed_before.items():
                    self.assertEqual((run_a_dir / name).read_bytes(), payload)

    def test_external_symlink_to_run_is_same_directory_resume(self):
        for kind in ("last", "best"):
            with self.subTest(kind=kind):
                run_dir = (
                    Path(self.temporary.name) / "same_link_{}".format(kind)
                )
                first = train_run(
                    self.corpus,
                    model_config=self.model_config,
                    training_config=replace(
                        self.training_config, output_dir=str(run_dir)
                    ),
                )
                source = Path(
                    first.last_checkpoint
                    if kind == "last"
                    else first.best_checkpoint
                )
                link = (
                    Path(self.temporary.name) / "outside_{}.pt".format(kind)
                )
                self._symlink(source, link)
                resumed = train_run(
                    self.corpus,
                    model_config=self.model_config,
                    training_config=replace(
                        self.training_config,
                        epochs=2,
                        output_dir=str(run_dir),
                    ),
                    resume_checkpoint=link,
                )
                self.assertEqual(resumed.completed_epoch, 2)
                metadata = [
                    item
                    for item in _read_jsonl(resumed.metrics_path)
                    if item["event"] == "run_metadata"
                ][-1]
                self.assertEqual(
                    metadata["resume_checkpoint"], str(source.resolve())
                )

    def test_relative_and_equivalent_resume_paths_resolve_consistently(self):
        run_dir = Path(self.temporary.name) / "relative_run"
        first = train_run(
            self.corpus,
            model_config=self.model_config,
            training_config=replace(
                self.training_config, output_dir=str(run_dir)
            ),
        )
        previous = Path.cwd()
        try:
            os.chdir(self.temporary.name)
            resumed = train_run(
                self.corpus,
                model_config=self.model_config,
                training_config=replace(
                    self.training_config,
                    epochs=2,
                    output_dir="relative_run/../relative_run",
                ),
                resume_checkpoint="relative_run/./last.pt",
            )
            resumed_last = Path(resumed.last_checkpoint).resolve()
        finally:
            os.chdir(str(previous))
        self.assertEqual(resumed.completed_epoch, 2)
        self.assertEqual(
            resumed_last,
            Path(first.last_checkpoint).resolve(),
        )

    def test_train_run_jsonl_contains_documented_metadata(self):
        result = train_run(
            self.corpus,
            model_config=self.model_config,
            training_config=replace(
                self.training_config,
                output_dir=str(Path(self.temporary.name) / "metadata_run"),
            ),
        )
        records = _read_jsonl(result.metrics_path)
        metadata = records[0]
        self.assertEqual(metadata["event"], "run_metadata")
        for name in (
            "git_commit",
            "git_dirty",
            "git_status_porcelain",
            "source_tree_sha256",
            "train_family_ids",
            "validation_family_ids",
        ):
            self.assertIn(name, metadata)
        self.assertEqual(len(metadata["source_tree_sha256"]), 64)
        self.assertTrue(
            any(item["event"] == "train_epoch" for item in records)
        )
        self.assertTrue(
            any(item["event"] == "validation_epoch" for item in records)
        )
        self.assertTrue(any(item["event"] == "checkpoint" for item in records))


if __name__ == "__main__":
    unittest.main()
