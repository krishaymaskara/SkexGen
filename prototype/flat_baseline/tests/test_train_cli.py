"""Subprocess coverage for the public teacher-forced training CLI."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

try:
    import torch
except ImportError:
    torch = None

from prototype.controlled_data.builders import build_history
from prototype.controlled_data.identity import source_family_id
from prototype.model_data.tests.fixtures import source, write_physical_corpus
from prototype.representation.model import GeometryEncoding


TORCH_REASON = "the training CLI requires real PyTorch"


@unittest.skipUnless(torch is not None, TORCH_REASON)
class TrainingCliTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.corpus = self.root / "corpus"
        sources = (source("E"), source("R"))
        family_ids = tuple(
            source_family_id(
                build_history(item, GeometryEncoding.CONTINUOUS)
            )
            for item in sources
        )
        write_physical_corpus(
            self.corpus,
            sources,
            partitions={
                family_ids[0]: "train",
                family_ids[1]: "validation",
            },
        )
        self.model_config = self.root / "model.json"
        self.model_config.write_text(
            json.dumps(
                {
                    "model_dim": 8,
                    "num_heads": 2,
                    "feedforward_dim": 12,
                    "encoder_layers": 1,
                    "decoder_layers": 1,
                    "dropout": 0.0,
                    "max_nodes": 12,
                    "max_operations": 2,
                    "latent_tokens": 1,
                    "codebook_size": 4,
                    "codebook_dim": 4,
                    "edge_pair_dim": 8,
                }
            ),
            encoding="utf-8",
        )
        self.training_config = self.root / "training.json"
        self.training_config.write_text(
            json.dumps(
                {
                    "epochs": 1,
                    "batch_size": 1,
                    "device": "cpu",
                    "dataloader_workers": 0,
                }
            ),
            encoding="utf-8",
        )

    def _run(self, *extra, corpus=None, output=None, model=None, training=None):
        command = [
            sys.executable,
            "-m",
            "prototype.flat_baseline.train",
            "--corpus-dir",
            str(self.corpus if corpus is None else corpus),
            "--output-dir",
            str(self.root / "run" if output is None else output),
            "--model-config",
            str(self.model_config if model is None else model),
            "--training-config",
            str(self.training_config if training is None else training),
        ]
        command.extend(extra)
        environment = dict(os.environ)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        return subprocess.run(
            command,
            cwd=str(Path(__file__).resolve().parents[3]),
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )

    def _assert_error(self, result, code):
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertEqual(result.stdout, "")
        self.assertNotIn("Traceback", result.stderr)
        record = json.loads(result.stderr)
        self.assertEqual(record["error"]["code"], code)
        self.assertEqual(
            result.stderr,
            json.dumps(
                record,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n",
        )

    def test_configuration_failures_are_structured(self):
        cases = (
            ("malformed", "{", "malformed_model_config_json"),
            ("array", "[]", "non_object_model_config"),
            (
                "unknown_model",
                '{"unknown_model_field":1}',
                "unknown_model_config_field",
            ),
            ("invalid", '{"dropout":2.0}', "invalid_configuration"),
        )
        for name, payload, code in cases:
            with self.subTest(name=name):
                path = self.root / (name + ".json")
                path.write_text(payload, encoding="utf-8")
                result = self._run(model=path, output=self.root / name)
                self._assert_error(result, code)
                if name == "unknown_model":
                    self.assertIn("unknown_model_field", result.stderr)

        unknown_training = self.root / "unknown_training.json"
        unknown_training.write_text(
            '{"unknown_training_field":1}', encoding="utf-8"
        )
        result = self._run(
            training=unknown_training,
            output=self.root / "unknown_training",
        )
        self._assert_error(result, "unknown_training_config_field")
        self.assertIn("unknown_training_field", result.stderr)

    def test_missing_corpus_and_invalid_partitions_are_structured(self):
        result = self._run(
            corpus=self.root / "missing",
            output=self.root / "missing_run",
        )
        self._assert_error(result, "missing_corpus")
        for arguments, code in (
            (("--train-split", "test"), "invalid_training_partition"),
            (("--train-split", "unknown"), "invalid_training_partition"),
            (("--validation-split", "test"), "invalid_validation_partition"),
            (
                (
                    "--validation-split",
                    "secondary_systematic_validation",
                ),
                "invalid_validation_partition",
            ),
        ):
            with self.subTest(arguments=arguments):
                result = self._run(
                    *arguments,
                    output=self.root / ("partition_" + arguments[-1]),
                )
                self._assert_error(result, code)

        malformed = self.root / "malformed_corpus"
        malformed.mkdir()
        (malformed / "corpus_manifest.json").write_text(
            "{", encoding="utf-8"
        )
        result = self._run(
            corpus=malformed,
            output=self.root / "malformed_corpus_run",
        )
        self._assert_error(result, "json_loading")

    def test_managed_output_collision_is_structured_and_preserved(self):
        output = self.root / "occupied"
        output.mkdir()
        metrics = output / "metrics.jsonl"
        metrics.write_bytes(b"unrelated\n")
        result = self._run(output=output)
        self._assert_error(result, "managed_output_collision")
        self.assertEqual(metrics.read_bytes(), b"unrelated\n")

    def test_successful_minimal_invocation(self):
        output = self.root / "success"
        result = self._run(output=output)
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["last_checkpoint"], str(output / "last.pt"))
        self.assertTrue((output / "metrics.jsonl").is_file())
        self.assertTrue((output / "last.pt").is_file())
        self.assertTrue((output / "best.pt").is_file())


if __name__ == "__main__":
    unittest.main()
