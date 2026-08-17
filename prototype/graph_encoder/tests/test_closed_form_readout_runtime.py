"""Real-PyTorch synthetic numerical tests for ADR-0012."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from prototype.graph_encoder.errors import GraphEncoderError


try:
    import torch
except ImportError:
    torch = None


TORCH_REASON = "closed-form readout runtime tests require real PyTorch"


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@unittest.skipUnless(torch is not None, TORCH_REASON)
class ClosedFormReadoutRuntimeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_exact_feature_construction_RE_slots_and_EE_repeated_prequant(self):
        from prototype.graph_encoder import closed_form_readout as readout

        features, labels = readout.synthetic_fixture()
        joined = readout.join_detached_rows(features, labels, "flat")
        extrude = readout.select_cohort(joined, "extrude", "full")
        matrices = {
            name: readout.construct_features(extrude, name, "extrude")["matrix"]
            for name in ("C", "P", "D-additive", "D-gated", "context-only")
        }
        self.assertEqual([matrices[name].size(1) for name in matrices], [32, 32, 34, 66, 2])
        ee = [index for index, row in enumerate(extrude) if row["operation_template"] == "EE"]
        first, second = ee[0], ee[1]
        self.assertTrue(torch.equal(matrices["P"][first], matrices["P"][second]))
        self.assertEqual(extrude[first]["family_id"], extrude[second]["family_id"])
        re_index = next(
            index for index, row in enumerate(extrude) if row["operation_template"] == "RE"
        )
        self.assertEqual(matrices["context-only"][re_index].tolist(), [0.0, 1.0])
        self.assertTrue(torch.equal(matrices["D-gated"][re_index, :32], torch.zeros(32, dtype=torch.float64)))
        self.assertTrue(torch.equal(matrices["D-gated"][re_index, 32:64], matrices["P"][re_index]))

    def test_train_only_center_scale_and_binary_not_scaled(self):
        from prototype.graph_encoder import closed_form_readout as readout

        train = torch.tensor([[0.0, 0.0], [2.0, 1.0]], dtype=torch.float64)
        test = torch.tensor([[100.0, 1.0]], dtype=torch.float64)
        result = readout.preprocess_fold(train, test, ("continuous", "binary"))
        self.assertEqual(result.means, (1.0, 0.5))
        self.assertEqual(result.scales, (1.0, 1.0))
        self.assertEqual(result.test.tolist(), [[99.0, 0.5]])

    def test_zero_variance_columns_are_dropped_and_reported(self):
        from prototype.graph_encoder import closed_form_readout as readout

        train = torch.tensor([[1.0, 0.0], [1.0, 1.0]], dtype=torch.float64)
        result = readout.preprocess_fold(train, train, ("continuous", "binary"))
        self.assertEqual(result.retained_columns, (1,))
        self.assertEqual(result.zero_variance_columns, (0,))

    def test_unpenalized_intercept_is_exact_with_no_retained_features(self):
        from prototype.graph_encoder import closed_form_readout as readout

        x = torch.ones((4, 2), dtype=torch.float64)
        targets = torch.nn.functional.one_hot(
            torch.tensor([2, 2, 2, 2]), num_classes=5
        ).to(torch.float64)
        cache = readout.RidgeSVDCache(x, x[:2], ("continuous", "binary"))
        scores = cache.scores(targets, 100.0)
        self.assertTrue(torch.equal(scores.argmax(dim=1), torch.tensor([2, 2])))
        self.assertTrue(torch.allclose(scores[:, 2], torch.ones(2, dtype=torch.float64)))

    def test_five_outputs_survive_a_missing_training_class(self):
        from prototype.graph_encoder import closed_form_readout as readout

        x = torch.arange(12, dtype=torch.float64).reshape(6, 2)
        labels = torch.tensor([0, 1, 2, 3, 0, 1])
        targets = torch.nn.functional.one_hot(labels, num_classes=5).to(torch.float64)
        cache = readout.RidgeSVDCache(x, x, ("continuous", "continuous"))
        scores = cache.scores(targets, 1.0)
        self.assertEqual(tuple(scores.shape), (6, 5))
        self.assertTrue(torch.equal(scores[:, 4], torch.zeros(6, dtype=torch.float64)))

    def test_lambda_and_argmax_ties_choose_largest_lambda_and_smallest_class(self):
        from prototype.graph_encoder import closed_form_readout as readout

        table = {
            "balanced_accuracy": torch.zeros((7, 3), dtype=torch.float64),
            "accuracy": torch.zeros((7, 3), dtype=torch.float64),
        }
        self.assertEqual(readout.select_lambdas_balanced_then_raw(table).tolist(), [6, 6, 6])
        tied = torch.zeros((2, 5), dtype=torch.float64)
        self.assertEqual(tied.argmax(dim=1).tolist(), [0, 0])

    def test_cached_SVD_and_uncached_reference_scores_and_predictions_agree(self):
        from prototype.graph_encoder import closed_form_readout as readout

        generator = torch.Generator().manual_seed(2026)
        train = torch.randn((9, 12), generator=generator, dtype=torch.float64)
        test = torch.randn((4, 12), generator=generator, dtype=torch.float64)
        labels = torch.tensor([0, 1, 2, 3, 4, 0, 2, 3, 1])
        targets = torch.nn.functional.one_hot(labels, num_classes=5).to(torch.float64)
        kinds = ("continuous",) * 10 + ("binary",) * 2
        train[:, -2:] = (train[:, -2:] > 0).to(torch.float64)
        test[:, -2:] = (test[:, -2:] > 0).to(torch.float64)
        cache = readout.RidgeSVDCache(train, test, kinds)
        for ridge_lambda in readout.RIDGE_LAMBDAS:
            cached = cache.scores(targets, ridge_lambda)
            reference = readout.uncached_reference_scores(
                train, test, kinds, targets, ridge_lambda
            )
            self.assertTrue(torch.allclose(cached, reference, atol=1e-9, rtol=1e-9))
            self.assertEqual(cached.argmax(dim=1).tolist(), reference.argmax(dim=1).tolist())

    def test_batched_and_individual_permutation_RHS_agree(self):
        from prototype.graph_encoder import closed_form_readout as readout

        x = torch.arange(48, dtype=torch.float64).reshape(8, 6)
        targets = torch.stack([
            torch.nn.functional.one_hot(
                torch.tensor(values), num_classes=5
            ).to(torch.float64)
            for values in ([0, 1, 2, 3, 4, 0, 1, 2], [2, 1, 0, 4, 3, 2, 1, 0])
        ])
        cache = readout.RidgeSVDCache(x, x[:3], ("continuous",) * 6)
        batched = cache.scores(targets, 0.1)
        individual = torch.stack([cache.scores(target, 0.1) for target in targets])
        self.assertTrue(torch.allclose(batched, individual, atol=1e-12, rtol=1e-12))

    def test_metric_record_reports_support_recall_counts_and_masking(self):
        from prototype.graph_encoder import closed_form_readout as readout

        record = readout.metric_record([0, 1, 2, 3, 4], [0, 0, 0, 0, 0])
        self.assertEqual(record["class_support"], [1, 1, 1, 1, 1])
        self.assertEqual(record["predicted_class_count"], [5, 0, 0, 0, 0])
        self.assertEqual(record["per_class_recall"], [1.0, 0.0, 0.0, 0.0, 0.0])
        self.assertTrue(record["class_masking_detected"])

    def test_nested_family_LOFO_has_no_family_overlap(self):
        from prototype.graph_encoder import closed_form_readout as readout

        features, labels = readout.synthetic_fixture()
        joined = readout.join_detached_rows(features, labels, "flat")
        rows = readout.select_cohort(joined, "extrude", "single_extrusion_E_RE")
        targets = readout.target_matrix(rows, 1)
        result = readout.fit_ridge_pipeline(
            rows, "context-only", targets, arm="flat", operation_type="extrude",
            cohort="single_extrusion_E_RE",
        )["record"]
        self.assertEqual(len(result["folds"]), 16)
        for fold in result["folds"]:
            self.assertNotIn(fold["held_out_family_id"], fold["train_family_ids"])
            self.assertIn(fold["selected_lambda"], readout.RIDGE_LAMBDAS)
            self.assertEqual(len(fold["inner_folds"]), 15)

    def _write_inputs(self, root):
        from prototype.graph_encoder import closed_form_readout as readout
        from prototype.graph_encoder import representation_probe as probe

        features, labels = readout.synthetic_fixture()
        feature_payload = {
            "schema_version": probe.PROBE_FEATURE_VERSION,
            "protocol_version": probe.PROBE_PROTOCOL_VERSION,
            "target_free": True,
            "feature_dimensions": dict(probe.FEATURE_DIMENSIONS),
            "rows": list(features),
        }
        feature_path = root / "detached_features.pt"
        torch.save(feature_payload, str(feature_path))
        feature_hash = _sha256(feature_path)
        manifest = {
            "schema_version": probe.PROBE_FEATURE_VERSION,
            "protocol_version": probe.PROBE_PROTOCOL_VERSION,
            "feature_file": "detached_features.pt",
            "feature_file_byte_size": feature_path.stat().st_size,
            "feature_file_sha256": feature_hash,
            "written_and_hashed_before_label_join": True,
            "target_or_label_content_present": False,
            "model_state_present": False,
            "optimizer_state_present": False,
            "row_count": 96,
            "feature_dimensions": dict(probe.FEATURE_DIMENSIONS),
        }
        (root / "feature_manifest.json").write_text(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")), encoding="utf-8"
        )
        label_payload = {
            "schema_version": probe.PROBE_LABEL_VERSION,
            "labels_joined_after_ge1_models_released": True,
            "grouping_unit": "physical_family",
            "class_grids": {name: list(values) for name, values in probe.OPERATION_GRIDS.items()},
            "rows": list(labels),
        }
        (root / "labels.json").write_text(
            json.dumps(label_payload, sort_keys=True, separators=(",", ":")), encoding="utf-8"
        )
        return {name: _sha256(root / name) for name in readout.READOUT_INPUT_HASHES}

    def test_synthetic_loader_enforces_hashes_target_isolation_join_order_and_immutability(self):
        from prototype.graph_encoder import closed_form_readout as readout

        with tempfile.TemporaryDirectory(prefix="readout-input-") as temporary:
            root = Path(temporary)
            hashes = self._write_inputs(root)
            before = dict(hashes)
            loaded = readout.load_readout_inputs(
                root, expected_hashes=hashes, synthetic_test=True
            )
            after = {name: _sha256(root / name) for name in hashes}
        self.assertEqual(before, after)
        self.assertEqual(len(loaded["feature_rows"]), 96)
        self.assertEqual(len(loaded["label_rows"]), 48)
        self.assertEqual(
            loaded["label_join_order"],
            "target_free_payload_validated_and_released_before_labels_parsed",
        )

    def test_hash_override_is_forbidden_outside_synthetic_tests(self):
        from prototype.graph_encoder import closed_form_readout as readout

        with self.assertRaises(GraphEncoderError) as context:
            readout.load_readout_inputs("unused", expected_hashes={}, synthetic_test=False)
        self.assertEqual(context.exception.code, "unsafe_hash_override")


if __name__ == "__main__":
    unittest.main()
