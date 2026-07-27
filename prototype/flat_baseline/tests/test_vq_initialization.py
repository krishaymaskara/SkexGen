"""Focused contracts for deterministic train-only codebook initialization."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest import mock

try:
    import torch
except (ImportError, OSError):
    torch = None

from prototype.flat_baseline.training_config import TrainingConfig

if torch is not None:
    from prototype.flat_baseline.vq import EMAVectorQuantizer
    from prototype.flat_baseline.vq_initialization import (
        PSEUDO_COUNT_POLICY,
        VQInitializationError,
        _distinct_count,
        collect_train_prequant_vectors,
        deterministic_kmeans,
        initialize_quantizer_from_vectors,
        initialize_train_codebook,
    )


TORCH_REASON = "real PyTorch is required for VQ initialization tests"


@unittest.skipUnless(torch is not None, TORCH_REASON)
class VQInitializationTests(unittest.TestCase):
    def vectors(self, count=64, width=4):
        return torch.tensor([
            [
                float(index),
                float(index * index % 17),
                float(index % 7),
                float(index % 11),
            ][:width]
            for index in range(count)
        ], dtype=torch.float64)

    def test_default_initialization_mode_is_unchanged(self):
        self.assertEqual(TrainingConfig().vq_init, "normal")
        torch.manual_seed(2026)
        first = EMAVectorQuantizer(32, 4, 0.25, 0.99, 1e-5)
        torch.manual_seed(2026)
        second = EMAVectorQuantizer(32, 4, 0.25, 0.99, 1e-5)
        self.assertTrue(torch.equal(first.embedding, second.embedding))
        self.assertTrue(torch.equal(first.ema_weight, second.ema_weight))
        self.assertTrue(torch.equal(
            first.ema_cluster_size, second.ema_cluster_size
        ))

    def test_train_kmeans_is_deterministic_and_finite_for_32_codes(self):
        vectors = self.vectors()
        first = deterministic_kmeans(vectors, 32, 2026)
        second = deterministic_kmeans(vectors, 32, 2026)
        for left, right in zip(first, second):
            self.assertTrue(torch.equal(left, right))
        self.assertTrue(torch.isfinite(first[0]).all())
        self.assertEqual(first[0].shape, (32, 4))
        self.assertEqual(_distinct_count(first[0], 1e-6), 32)

    def test_codebook_and_ema_buffers_are_consistent(self):
        quantizer = EMAVectorQuantizer(8, 4, 0.25, 0.99, 1e-5)
        _, _, actual_counts, pseudo_counts, pseudo_sums = (
            initialize_quantizer_from_vectors(
            quantizer, self.vectors(32), 2026
            )
        )
        implied = (
            quantizer.ema_weight
            / quantizer.ema_cluster_size.unsqueeze(1)
        )
        torch.testing.assert_close(
            quantizer.embedding, implied, rtol=1e-5, atol=1e-6
        )
        self.assertTrue((quantizer.ema_cluster_size > 0).all())
        self.assertEqual(float(pseudo_counts.sum().item()), 8.0)
        torch.testing.assert_close(
            pseudo_counts / pseudo_counts.sum(),
            actual_counts / actual_counts.sum(),
            rtol=1e-12,
            atol=1e-12,
        )
        torch.testing.assert_close(
            pseudo_sums,
            pseudo_counts.unsqueeze(1) * quantizer.embedding.double(),
            rtol=1e-5,
            atol=1e-6,
        )

    def test_pseudo_counts_match_normal_effective_mass_under_exact_update(self):
        quantizer = EMAVectorQuantizer(8, 4, 0.25, 0.99, 1e-5)
        _, _, _, pseudo_counts, _ = initialize_quantizer_from_vectors(
            quantizer, self.vectors(32), 2026
        )
        normal_counts = torch.ones_like(pseudo_counts)
        batch_count = 12.0
        decay = quantizer.decay
        normal_fraction = (
            (1.0 - decay) * batch_count
            / (
                decay * float(normal_counts.sum().item())
                + (1.0 - decay) * batch_count
            )
        )
        treatment_fraction = (
            (1.0 - decay) * batch_count
            / (
                decay * float(pseudo_counts.sum().item())
                + (1.0 - decay) * batch_count
            )
        )
        self.assertEqual(normal_fraction, treatment_fraction)
        self.assertEqual(float(pseudo_counts.sum().item()), 8.0)

    def test_exact_quantizer_smoothing_equation_matches_real_forward(self):
        quantizer = EMAVectorQuantizer(4, 2, 0.25, 0.9, 1e-5)
        vectors = torch.tensor(
            [[-4.0, 0.0], [-3.0, 0.0], [1.0, 0.0], [2.0, 0.0],
             [3.0, 0.0], [4.0, 0.0], [8.0, 0.0], [9.0, 0.0]],
            dtype=torch.float64,
        )
        initialize_quantizer_from_vectors(quantizer, vectors, 2026)
        inputs = torch.tensor(
            [[[-3.5, 0.0], [1.5, 0.0], [8.5, 0.0]]],
            dtype=torch.float32,
        )
        before_counts = quantizer.ema_cluster_size.clone()
        before_weight = quantizer.ema_weight.clone()
        embedding = quantizer.embedding.clone()
        flat = inputs.reshape(-1, 2)
        assignments = (
            flat.pow(2).sum(1, keepdim=True)
            + embedding.pow(2).sum(1).unsqueeze(0)
            - 2.0 * torch.matmul(flat, embedding.t())
        ).argmin(1)
        batch_counts = torch.bincount(assignments, minlength=4)
        batch_sums = torch.zeros_like(before_weight)
        batch_sums.index_add_(0, assignments, flat)
        expected_counts = (
            quantizer.decay * before_counts
            + (1.0 - quantizer.decay) * batch_counts
        )
        expected_weight = (
            quantizer.decay * before_weight
            + (1.0 - quantizer.decay) * batch_sums
        )
        total = expected_counts.sum()
        smoothed = (
            (expected_counts + quantizer.epsilon)
            / (total + 4 * quantizer.epsilon)
            * total
        )
        expected_embedding = expected_weight / smoothed.unsqueeze(1)
        quantizer.train()
        quantizer(inputs)
        torch.testing.assert_close(
            quantizer.ema_cluster_size, expected_counts
        )
        torch.testing.assert_close(quantizer.ema_weight, expected_weight)
        torch.testing.assert_close(quantizer.embedding, expected_embedding)

    def test_duplicate_or_insufficient_vectors_fail_explicitly(self):
        cases = (
            (
                torch.zeros(3, 4, dtype=torch.float64),
                4,
                "insufficient_prequant_vectors",
            ),
            (
                torch.zeros(8, 4, dtype=torch.float64),
                4,
                "insufficient_distinct_prequant_vectors",
            ),
        )
        for vectors, count, code in cases:
            with self.subTest(code=code):
                with self.assertRaises(VQInitializationError) as captured:
                    deterministic_kmeans(vectors, count, 2026)
                self.assertEqual(captured.exception.code, code)

    def test_only_training_examples_enter_collection(self):
        quantizer = EMAVectorQuantizer(4, 4, 0.25, 0.99, 1e-5)
        model = SimpleNamespace(
            vq=quantizer,
            config=SimpleNamespace(latent_tokens=2),
        )
        training = ("train-a", "train-b")
        data = SimpleNamespace(
            train_examples=training,
            validation_examples=(object(),),
            train_family_ids=("train-a", "train-b"),
            validation_family_ids=("validation-a",),
        )
        config = replace(
            TrainingConfig(), seed=2026, vq_init="train-kmeans"
        )
        with mock.patch(
            "prototype.flat_baseline.vq_initialization."
            "collect_train_prequant_vectors",
            return_value=self.vectors(16),
        ) as collect:
            report = initialize_train_codebook(
                model, data, config, torch.device("cpu"), torch
            )
        self.assertIs(collect.call_args.args[1], training)
        self.assertEqual(report["partition"], "train")
        self.assertEqual(report["validation_family_count_used"], 0)
        self.assertEqual(report["test_family_count_used"], 0)
        self.assertEqual(report["pseudo_count_policy"], PSEUDO_COUNT_POLICY)
        self.assertEqual(report["effective_mass_ratio_to_normal"], 1.0)

    def test_initialization_reports_are_byte_identical(self):
        data = SimpleNamespace(
            train_examples=("train",),
            validation_examples=(),
            train_family_ids=("train",),
            validation_family_ids=(),
        )
        config = replace(
            TrainingConfig(), seed=2026, vq_init="train-kmeans"
        )
        reports = []
        with mock.patch(
            "prototype.flat_baseline.vq_initialization."
            "collect_train_prequant_vectors",
            return_value=self.vectors(32),
        ):
            for _ in range(2):
                model = SimpleNamespace(
                    vq=EMAVectorQuantizer(8, 4, 0.25, 0.99, 1e-5),
                    config=SimpleNamespace(latent_tokens=2),
                )
                reports.append(initialize_train_codebook(
                    model, data, config, torch.device("cpu"), torch
                ))
        self.assertEqual(reports[0], reports[1])

    def test_nonfinite_vectors_are_rejected(self):
        vectors = self.vectors(16)
        vectors[0, 0] = float("nan")
        with self.assertRaises(VQInitializationError) as captured:
            deterministic_kmeans(vectors, 4, 2026)
        self.assertEqual(captured.exception.code, "nonfinite_prequant")

    def test_collection_is_model_state_and_rng_neutral(self):
        class Batch:
            def to_torch(self, torch_module):
                return {
                    "categorical_ids": torch_module.ones(
                        2, 1, 1, dtype=torch_module.long
                    ),
                    "geometry": torch_module.zeros(2, 1, 1),
                    "geometry_mask": torch_module.zeros(
                        2, 1, 1, dtype=torch_module.bool
                    ),
                    "padding_mask": torch_module.ones(
                        2, 1, dtype=torch_module.bool
                    ),
                }

        class CollectionModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.dropout = torch.nn.Dropout(0.5)
                self.norm = torch.nn.BatchNorm1d(4)
                self.to_codebook = torch.nn.Linear(4, 3)
                self.vq = SimpleNamespace(embedding_dim=3)
                self.config = SimpleNamespace(latent_tokens=2)

            def encode_to_memory(
                self, categorical_ids, geometry, geometry_mask, padding_mask
            ):
                del geometry, geometry_mask, padding_mask
                values = categorical_ids[:, :1, :1].float().expand(
                    -1, 2, 4
                )
                values = self.dropout(values)
                values = self.norm(values.reshape(-1, 4)).reshape(2, 2, 4)
                return self.to_codebook(values)

        model = CollectionModel()
        model.train()
        state = {
            name: value.clone() for name, value in model.state_dict().items()
        }
        torch.manual_seed(991)
        rng = torch.get_rng_state().clone()
        with mock.patch(
            "prototype.flat_baseline.vq_initialization.make_data_loader",
            return_value=(Batch(),),
        ):
            vectors = collect_train_prequant_vectors(
                model,
                ("train-a", "train-b"),
                2,
                0,
                2026,
                torch.device("cpu"),
                torch,
            )
        self.assertEqual(vectors.shape, (4, 3))
        self.assertTrue(model.training)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, state[name]))
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))


if __name__ == "__main__":
    unittest.main()
