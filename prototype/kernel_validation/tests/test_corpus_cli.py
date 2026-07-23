from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from prototype.kernel_validation.corpus import CorpusError, load_corpus
from prototype.kernel_validation.run import main

from prototype.kernel_validation.tests.fakes import FakeAdapter, make_sample


def _write_corpus(root: Path):
    samples = [make_sample("E", "continuous"), make_sample("E", "quantized")]
    (root / "samples").mkdir(parents=True)
    records = []
    for sample in samples:
        path = root / sample.relative_json_path
        path.write_text(sample.payload + "\n", encoding="utf-8")
        records.append(
            {
                "source_family_id": sample.source_family_id,
                "sample_id": sample.sample_id,
                "geometry_encoding": sample.geometry_encoding,
                "operation_template": sample.operation_template,
                "primitive_family": sample.primitive_family,
                "relative_json_path": sample.relative_json_path,
            }
        )
    manifest = {
        "generator_version": "test",
        "representation_schema_version": 1,
        "canonicalization_version": "test",
        "configuration_sha256": "abc",
        "generation_seed": 0,
        "total_source_family_count": 1,
        "total_sample_variant_count": 2,
        "samples": records,
    }
    (root / "corpus_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


class CorpusAndCliTests(unittest.TestCase):
    def test_loader_keeps_missing_sample_as_structured_input_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_corpus(root)
            first_path = next((root / "samples").iterdir())
            first_path.unlink()
            _, samples = load_corpus(root)
            self.assertEqual(sum(item.loading_error is not None for item in samples), 1)
            self.assertNotIn(str(root), next(item.loading_error for item in samples if item.loading_error))

    def test_loader_rejects_unsafe_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "corpus_manifest.json").write_text(
                json.dumps({"samples": [{
                    "source_family_id": "sf_x", "sample_id": "sv_x", "geometry_encoding": "continuous",
                    "operation_template": "E", "primitive_family": "circle", "relative_json_path": "../x"
                }]}), encoding="utf-8"
            )
            with self.assertRaises(CorpusError):
                load_corpus(root)

    def test_cli_success_and_existing_output_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            corpus = Path(directory) / "corpus"
            output = Path(directory) / "report"
            corpus.mkdir()
            _write_corpus(corpus)
            self.assertEqual(
                main(["--corpus-dir", str(corpus), "--output-dir", str(output)], adapter_factory=FakeAdapter),
                0,
            )
            self.assertTrue((output / "execution_report.json").exists())
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                code = main(["--corpus-dir", str(corpus), "--output-dir", str(output)], adapter_factory=FakeAdapter)
            self.assertEqual(code, 1)
            self.assertIn("output directory already exists", stderr.getvalue())

    def test_cli_missing_corpus_returns_one_without_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report"
            with contextlib.redirect_stderr(io.StringIO()):
                code = main(
                    ["--corpus-dir", str(Path(directory) / "missing"), "--output-dir", str(output)],
                    adapter_factory=FakeAdapter,
                )
            self.assertEqual(code, 1)
            self.assertFalse(output.exists())

    def test_no_display_imports_anywhere_in_package(self):
        root = Path(__file__).resolve().parents[1]
        for path in root.rglob("*.py"):
            self.assertNotIn("OCC" + ".Display", path.read_text(encoding="utf-8"), str(path))


if __name__ == "__main__":
    unittest.main()
