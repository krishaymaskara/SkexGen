"""Counterfactual endpoint and locality contract tests."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from prototype.model_data.counterfactual import load_counterfactual_examples
from prototype.model_data.errors import ModelDataError
from prototype.model_data.records import CounterfactualExample
from prototype.model_data.tests.fixtures import write_counterfactual_corpus


class CounterfactualTests(unittest.TestCase):
    def test_linkage_paths_and_evaluation_only_default(self):
        with tempfile.TemporaryDirectory() as temporary:
            family_id, base_id, edited_id = write_counterfactual_corpus(temporary)
            examples = load_counterfactual_examples(temporary)
        self.assertEqual(len(examples), 1)
        example = examples[0]
        self.assertEqual(example.edit_family_id, family_id)
        self.assertEqual(example.source.physical_family_id, base_id)
        self.assertEqual(example.target.physical_family_id, edited_id)
        self.assertEqual(example.partition, "test")
        self.assertTrue(example.evaluation_only)
        self.assertTrue(example.edited_attribute_paths)
        self.assertTrue(example.unaffected_attribute_paths)
        self.assertTrue(
            set(example.edited_attribute_paths).isdisjoint(
                example.unaffected_attribute_paths
            )
        )
        with self.assertRaises(TypeError):
            CounterfactualExample(
                example.source,
                example.target,
                example.edit_family_id,
                example.edit_sample_id,
                example.edit_type,
                example.edited_attribute_paths,
                example.unaffected_attribute_paths,
                example.split_name,
                example.partition,
                False,
            )

    def test_counterfactual_encoding_disagreement_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            write_counterfactual_corpus(temporary)
            manifest = json.loads(
                (Path(temporary) / "counterfactual_manifest.json").read_text()
            )
            quantized = next(
                item
                for item in manifest["samples"]
                if item["geometry_encoding"] == "quantized"
            )
            path = Path(temporary) / quantized["relative_json_path"]
            pair = json.loads(path.read_text())
            pair["locality"]["expected_unchanged_paths"] = []
            path.write_text(
                json.dumps(pair, sort_keys=True, separators=(",", ":")) + "\n"
            )
            with self.assertRaises(ModelDataError) as caught:
                load_counterfactual_examples(temporary)
        self.assertEqual(caught.exception.code, "counterfactual_physical_disagreement")

    def test_counterfactual_endpoint_leakage_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            write_counterfactual_corpus(temporary)
            split_path = Path(temporary) / "manifests" / "iid.json"
            split = json.loads(split_path.read_text())
            split["endpoints"][0]["partition"] = "validation"
            split_path.write_text(
                json.dumps(split, sort_keys=True, separators=(",", ":")) + "\n"
            )
            with self.assertRaises(ModelDataError) as caught:
                load_counterfactual_examples(temporary)
        self.assertEqual(caught.exception.code, "counterfactual_leakage")

    def test_stale_missing_duplicate_and_overlapping_paths_are_rejected(self):
        def remove_path(locality):
            locality["expected_unchanged_paths"].pop()

        def duplicate_path(locality):
            locality["expected_unchanged_paths"].append(
                dict(locality["expected_unchanged_paths"][0])
            )

        def overlap_path(locality):
            locality["expected_unchanged_paths"].append(
                dict(locality["allowed_changed_paths"][0])
            )

        def stale_path(locality):
            locality["expected_unchanged_paths"][0]["field_path"] = "stale.path"

        for mutation in (remove_path, duplicate_path, overlap_path, stale_path):
            with self.subTest(mutation=mutation.__name__):
                with tempfile.TemporaryDirectory() as temporary:
                    write_counterfactual_corpus(temporary)
                    manifest = json.loads(
                        (Path(temporary) / "counterfactual_manifest.json").read_text()
                    )
                    for record in manifest["samples"]:
                        path = Path(temporary) / record["relative_json_path"]
                        pair = json.loads(path.read_text())
                        mutation(pair["locality"])
                        path.write_text(
                            json.dumps(pair, sort_keys=True, separators=(",", ":"))
                            + "\n"
                        )
                    with self.assertRaises(ModelDataError) as caught:
                        load_counterfactual_examples(temporary)
                self.assertEqual(caught.exception.code, "counterfactual_locality")

    def test_malformed_path_is_structured_deserialization_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            write_counterfactual_corpus(temporary)
            manifest = json.loads(
                (Path(temporary) / "counterfactual_manifest.json").read_text()
            )
            for record in manifest["samples"]:
                path = Path(temporary) / record["relative_json_path"]
                pair = json.loads(path.read_text())
                del pair["locality"]["allowed_changed_paths"][0]["owner_id"]
                path.write_text(
                    json.dumps(pair, sort_keys=True, separators=(",", ":")) + "\n"
                )
            with self.assertRaises(ModelDataError) as caught:
                load_counterfactual_examples(temporary)
        self.assertEqual(caught.exception.code, "counterfactual_deserialization")

    def test_incomplete_edit_split_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            write_counterfactual_corpus(temporary)
            split_path = Path(temporary) / "manifests" / "iid.json"
            split = json.loads(split_path.read_text())
            split["samples"].pop()
            split_path.write_text(
                json.dumps(split, sort_keys=True, separators=(",", ":")) + "\n"
            )
            with self.assertRaises(ModelDataError) as caught:
                load_counterfactual_examples(temporary)
        self.assertEqual(caught.exception.code, "counterfactual_split_mismatch")


if __name__ == "__main__":
    unittest.main()
