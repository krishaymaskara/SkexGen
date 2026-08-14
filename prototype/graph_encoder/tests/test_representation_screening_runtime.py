"""Real-PyTorch synthetic tests for artifact-only representation screening."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
from unittest import mock
import unittest

from prototype.graph_encoder.errors import GraphEncoderError


try:
    import torch
except ImportError:
    torch = None


TORCH_REASON = "representation-screening runtime tests require real PyTorch"


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


@unittest.skipUnless(torch is not None, TORCH_REASON)
class RepresentationScreeningRuntimeTests(unittest.TestCase):
    def _write_inputs(self, root):
        from prototype.graph_encoder import representation_probe as probe

        labels = []
        features = []
        family_index = 0
        operation_patterns = {
            "E": ((0, "extrude"),),
            "R": ((0, "revolve"),),
            "EE": ((0, "extrude"), (1, "extrude")),
            "RE": ((0, "revolve"), (1, "extrude")),
        }
        for template in ("E", "R", "EE", "RE"):
            for unused in range(8):
                family_id = "family-{:02d}".format(family_index)
                for operation_index, operation_type in operation_patterns[template]:
                    label_key = "{}:{}:{}".format(
                        family_id, operation_index, operation_type
                    )
                    class_index = (family_index + operation_index) % 5
                    labels.append({
                        "class_index": class_index,
                        "class_physical_value": probe.OPERATION_GRIDS[
                            operation_type
                        ][class_index],
                        "family_id": family_id,
                        "label_key": label_key,
                        "operation_index": operation_index,
                        "operation_template": template,
                        "operation_type": operation_type,
                    })
                    type_context = (
                        (1.0, 0.0)
                        if operation_type == "extrude" else (0.0, 1.0)
                    )
                    slot_context = (
                        (1.0, 0.0)
                        if operation_index == 0 else (0.0, 1.0)
                    )
                    for arm_index, arm in enumerate(probe.PROBE_ARMS):
                        continuous = tuple(
                            float(family_index + operation_index + arm_index + index) / 100.0
                            for index in range(32)
                        )
                        normalized = 0.5
                        physical = 2.0 if operation_type == "extrude" else 180.0
                        features.append({
                            "key": "{}:{}".format(arm, label_key),
                            "label_key": label_key,
                            "arm": arm,
                            "family_id": family_id,
                            "operation_index": operation_index,
                            "operation_type": operation_type,
                            "A_normalized": normalized,
                            "A_physical": physical,
                            "B_raw_logit": 0.0,
                            "C": continuous,
                            "D": continuous + type_context + slot_context,
                        })
                family_index += 1
        feature_payload = {
            "schema_version": probe.PROBE_FEATURE_VERSION,
            "protocol_version": probe.PROBE_PROTOCOL_VERSION,
            "target_free": True,
            "feature_dimensions": dict(probe.FEATURE_DIMENSIONS),
            "rows": features,
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
            json.dumps(manifest, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        label_payload = {
            "schema_version": probe.PROBE_LABEL_VERSION,
            "labels_joined_after_ge1_models_released": True,
            "grouping_unit": "physical_family",
            "class_grids": {
                name: list(values) for name, values in probe.OPERATION_GRIDS.items()
            },
            "rows": labels,
        }
        (root / "labels.json").write_text(
            json.dumps(label_payload, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        hashes = {
            name: _sha256(root / name)
            for name in (
                "detached_features.pt", "feature_manifest.json", "labels.json"
            )
        }
        return hashes

    def test_synthetic_hash_schema_and_alignment_validation(self):
        from prototype.graph_encoder import representation_screening as screen

        with tempfile.TemporaryDirectory(prefix="screen-input-") as temporary:
            root = Path(temporary)
            hashes = self._write_inputs(root)
            with mock.patch.dict(screen.SCREEN_INPUT_HASHES, hashes, clear=True):
                loaded = screen.load_screening_inputs(root)
        self.assertEqual(len(loaded["feature_rows"]), 96)
        self.assertEqual(len(loaded["label_rows"]), 48)
        self.assertEqual(loaded["input_hashes"], hashes)
        self.assertFalse(loaded["feature_manifest"]["model_state_present"])
        self.assertFalse(loaded["feature_manifest"]["optimizer_state_present"])

    def test_alignment_rejects_changed_operation_identity(self):
        from prototype.graph_encoder import representation_screening as screen

        features = ({
            "label_key": "family:0:extrude",
            "arm": "flat",
            "family_id": "family",
            "operation_index": 0,
            "operation_type": "extrude",
        }, {
            "label_key": "family:0:extrude",
            "arm": "typed_graph",
            "family_id": "family",
            "operation_index": 0,
            "operation_type": "revolve",
        })
        labels = ({
            "label_key": "family:0:extrude",
            "family_id": "family",
            "operation_index": 0,
            "operation_type": "extrude",
        },)
        with self.assertRaises(GraphEncoderError) as context:
            screen._validate_feature_label_alignment(features, labels)
        self.assertEqual(context.exception.code, "screening_input_alignment_failure")

    def test_loading_and_screening_construct_no_ge1_model(self):
        from prototype.graph_encoder import representation_screening as screen
        from prototype.graph_encoder.tests.test_representation_screening_contract import (
            _fake_pipeline,
        )

        with tempfile.TemporaryDirectory(prefix="screen-input-") as temporary:
            root = Path(temporary)
            hashes = self._write_inputs(root)
            with mock.patch.dict(screen.SCREEN_INPUT_HASHES, hashes, clear=True):
                loaded = screen.load_screening_inputs(root)
                with mock.patch.object(
                    screen, "fit_probe_pipeline", side_effect=_fake_pipeline
                ) as fitter:
                    results = screen.run_real_label_screen(
                        loaded["feature_rows"],
                        loaded["label_rows"],
                        loaded["input_hashes"],
                        execution_source={
                            "git_commit": "synthetic", "slurm_job_id": "123",
                        },
                        environment={
                            "python": "3.8.13",
                            "pytorch": "1.11.0",
                            "device": "cpu",
                            "cuda_available": False,
                            "cpu_threads": 1,
                            "host": "synthetic",
                            "slurm_job_id": "123",
                        },
                    )
        self.assertEqual(fitter.call_count, 16)
        self.assertEqual(results["pipeline_count"], 16)
        self.assertFalse(results["screening_summary"]["formal_representation_conclusion_available"])
        self.assertFalse(results["screening_summary"]["authorizes_model_or_decoder_repair"])


if __name__ == "__main__":
    unittest.main()
