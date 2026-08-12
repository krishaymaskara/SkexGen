"""Real-PyTorch synthetic tests for accepted ADR-0011."""

from __future__ import annotations

import inspect
from pathlib import Path
import tempfile
import unittest

from prototype.graph_encoder.tests.fixtures import procedural_fixture


try:
    import torch
except ImportError:
    torch = None


TORCH_REASON = "representation-probe runtime tests require real PyTorch"


@unittest.skipUnless(torch is not None, TORCH_REASON)
class RepresentationProbeRuntimeTests(unittest.TestCase):
    def _examples(self):
        return tuple(
            procedural_fixture(template).physical
            for template in ("E", "R", "EE", "RE")
        )

    def test_target_free_extraction_records_exact_A_to_D_shapes(self):
        from prototype.graph_encoder.autonomous import autonomous_input_from_paired
        from prototype.graph_encoder.batching import build_paired_batch
        from prototype.graph_encoder.model import build_matched_ge1_models
        from prototype.graph_encoder.representation_probe import (
            extract_target_free_features,
        )

        examples = self._examples()
        paired = build_paired_batch(examples)
        for model in build_matched_ge1_models(seed=2026):
            model.requires_grad_(False)
            arm = model.config.encoder
            autonomous_input = autonomous_input_from_paired(paired, arm, torch)
            before = {
                name: value.detach().clone() for name, value in model.state_dict().items()
            }
            rows, autonomous = extract_target_free_features(
                model, (autonomous_input,), arm=arm
            )
            self.assertEqual(len(rows), 6)
            self.assertEqual(len(autonomous.conditions), 1)
            self.assertEqual(autonomous.conditions[0].condition, "P_true")
            for row in rows:
                self.assertEqual(len(row["C"]), 32)
                self.assertEqual(len(row["D"]), 36)
                self.assertGreater(row["A_normalized"], 0.0)
                self.assertLessEqual(row["A_normalized"], 1.0)
                self.assertIn(row["operation_type"], ("extrude", "revolve"))
            for name, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, before[name]), name)
            self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_target_argument_is_rejected_by_python_signature(self):
        from prototype.graph_encoder.representation_probe import (
            extract_target_free_features,
        )

        self.assertNotIn("target", inspect.signature(extract_target_free_features).parameters)
        with self.assertRaises(TypeError):
            extract_target_free_features(None, (), arm="flat", target=torch.zeros(1))

    def test_standardization_is_train_only_and_preserves_one_hot_columns(self):
        from prototype.graph_encoder.representation_probe import _standardize

        train = [[1.0] * 32 + [1.0, 0.0, 1.0, 0.0],
                 [3.0] * 32 + [0.0, 1.0, 0.0, 1.0]]
        test = [[101.0] * 32 + [1.0, 0.0, 0.0, 1.0]]
        train_x, test_x, evidence = _standardize(train, test, 32)
        self.assertTrue(torch.equal(train_x[:, 32:], torch.tensor(
            [[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]], dtype=torch.float64
        )))
        self.assertTrue(torch.equal(test_x[:, 32:], torch.tensor(
            [[1.0, 0.0, 0.0, 1.0]], dtype=torch.float64
        )))
        self.assertEqual(evidence["mean"], [2.0] * 32)
        self.assertEqual(test_x[0, 0].item(), 99.0)

    def test_linear_and_mlp_parameter_counts_are_derived(self):
        from prototype.graph_encoder.representation_probe import (
            _linear_fit_predict,
            _mlp_fit_predict,
        )

        x_c = torch.randn((10, 32), dtype=torch.float64)
        x_d = torch.randn((10, 36), dtype=torch.float64)
        labels = tuple(index % 5 for index in range(10))
        unused_predictions, linear_c = _linear_fit_predict(
            x_c, labels, x_c, 1.0, max_iter=10
        )
        unused_predictions, linear_d = _linear_fit_predict(
            x_d, labels, x_d, 1.0, max_iter=10
        )
        unused_predictions, mlp_c = _mlp_fit_predict(
            x_c, labels, x_c, 2026, updates=2
        )
        unused_predictions, mlp_d = _mlp_fit_predict(
            x_d, labels, x_d, 2026, updates=2
        )
        self.assertEqual(linear_c["parameter_count"], 165)
        self.assertEqual(linear_d["parameter_count"], 185)
        self.assertEqual(mlp_c["parameter_count"], 309)
        self.assertEqual(mlp_d["parameter_count"], 341)

    def test_linear_zero_initialization_is_deterministic(self):
        from prototype.graph_encoder.representation_probe import _linear_fit_predict

        features = torch.tensor(
            [[float(index), float(index % 3)] for index in range(15)],
            dtype=torch.float64,
        )
        labels = tuple(index % 5 for index in range(15))
        first, first_evidence = _linear_fit_predict(
            features, labels, features, 0.1, max_iter=20
        )
        second, second_evidence = _linear_fit_predict(
            features, labels, features, 0.1, max_iter=20
        )
        self.assertEqual(first, second)
        self.assertEqual(first_evidence["final_objective"], second_evidence["final_objective"])

    def test_mlp_seed_is_deterministic_and_updates_are_exact(self):
        from prototype.graph_encoder.representation_probe import _mlp_fit_predict

        features = torch.randn((15, 32), dtype=torch.float64)
        labels = tuple(index % 5 for index in range(15))
        first, first_evidence = _mlp_fit_predict(
            features, labels, features, 1234, updates=5
        )
        second, second_evidence = _mlp_fit_predict(
            features, labels, features, 1234, updates=5
        )
        self.assertEqual(first, second)
        self.assertEqual(first_evidence, second_evidence)
        self.assertEqual(first_evidence["updates"], 5)

    def test_feature_file_contains_no_target_or_label(self):
        from prototype.graph_encoder.representation_probe import (
            PROBE_ARMS,
            write_target_free_features,
        )

        rows = []
        for arm in PROBE_ARMS:
            for family_index in range(32):
                operation_count = 2 if family_index >= 16 else 1
                for operation_index in range(operation_count):
                    operation_type = (
                        "revolve" if family_index % 2 else "extrude"
                    )
                    # Force the exact aggregate 32 extrusion / 16 revolve mix.
                    flat_index = len([item for item in rows if item["arm"] == arm])
                    operation_type = "extrude" if flat_index < 32 else "revolve"
                    context = (
                        (1.0, 0.0) if operation_type == "extrude" else (0.0, 1.0)
                    ) + (
                        (1.0, 0.0) if operation_index == 0 else (0.0, 1.0)
                    )
                    rows.append({
                        "key": "{}:f{}:{}:{}".format(
                            arm, family_index, operation_index, operation_type
                        ),
                        "label_key": "f{}:{}:{}".format(
                            family_index, operation_index, operation_type
                        ),
                        "arm": arm,
                        "family_id": "f{}".format(family_index),
                        "operation_index": operation_index,
                        "operation_type": operation_type,
                        "A_normalized": 0.5,
                        "A_physical": 2.0 if operation_type == "extrude" else 180.0,
                        "B_raw_logit": 0.0,
                        "C": (0.0,) * 32,
                        "D": (0.0,) * 32 + context,
                    })
        with tempfile.TemporaryDirectory(prefix="probe-features-") as temporary:
            payload, manifest = write_target_free_features(Path(temporary), rows)
            observed = torch.load(
                str(Path(temporary) / "detached_features.pt"), map_location="cpu"
            )
        self.assertEqual(payload, observed)
        self.assertTrue(observed["target_free"])
        self.assertNotIn("labels", observed)
        self.assertNotIn("target", observed)
        self.assertFalse(manifest["target_or_label_content_present"])


if __name__ == "__main__":
    unittest.main()
