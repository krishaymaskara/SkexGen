"""Physical-family loading and vocabulary contract tests."""

from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from prototype.model_data.errors import ModelDataError
from prototype.model_data.loader import load_physical_examples
from prototype.model_data.serialization import canonical_record_json
from prototype.model_data.tests.fixtures import source, write_physical_corpus
from prototype.model_data.vocab import ALL_VOCABULARIES
from prototype.representation.serialization import history_from_json, history_to_json


class PhysicalLoadingTests(unittest.TestCase):
    def test_variants_collapse_to_one_physical_example(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, samples = write_physical_corpus(temporary, (source("E"), source("RR")))
            examples = load_physical_examples(temporary)
        self.assertEqual(len(samples), 4)
        self.assertEqual(len(examples), 2)
        for example in examples:
            self.assertEqual(example.metadata.geometry_encodings, ("continuous", "quantized"))
            self.assertEqual(len(example.metadata.sample_ids), 2)
            self.assertEqual(len(set(example.metadata.sample_ids)), 2)

    def test_matching_manifest_ids_cannot_hide_decoded_physical_disagreement(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, samples = write_physical_corpus(temporary, (source("E"),))
            quantized = next(
                item for item in samples if item["geometry_encoding"] == "quantized"
            )
            path = Path(temporary) / quantized["relative_json_path"]
            history = history_from_json(path.read_text().rstrip("\n"))
            payload = json.loads(history_to_json(history))
            distance = next(
                item["geometry"]["distance"]
                for item in payload["geometry"]["node_geometry"]
                if item["geometry"]["geometry_type"] == "extrude"
            )
            distance["values"][0] += 1
            path.write_text(
                json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
            )
            with self.assertRaises(ModelDataError) as caught:
                load_physical_examples(temporary)
        self.assertEqual(caught.exception.code, "physical_disagreement")

    def test_encoding_specific_metadata_disagreement_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            write_physical_corpus(temporary, (source("E"),))
            path = Path(temporary) / "corpus_manifest.json"
            manifest = json.loads(path.read_text())
            manifest["samples"][0]["operation_template"] = "R"
            path.write_text(
                json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n"
            )
            with self.assertRaises(ModelDataError) as caught:
                load_physical_examples(temporary)
        self.assertEqual(caught.exception.code, "encoding_metadata_disagreement")

    def test_incomplete_family_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            write_physical_corpus(temporary, (source("E"),))
            manifest_path = Path(temporary) / "corpus_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["samples"] = manifest["samples"][:1]
            manifest["total_sample_variant_count"] = 1
            retained_id = manifest["samples"][0]["sample_id"]
            manifest["families"][0]["sample_ids"] = [retained_id]
            manifest_path.write_text(
                json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n"
            )
            split_path = Path(temporary) / "manifests" / "iid.json"
            split = json.loads(split_path.read_text())
            split["samples"] = [
                item for item in split["samples"] if item["sample_id"] == retained_id
            ]
            split_path.write_text(
                json.dumps(split, sort_keys=True, separators=(",", ":")) + "\n"
            )
            with self.assertRaises(ModelDataError) as caught:
                load_physical_examples(temporary)
        self.assertEqual(caught.exception.code, "variant_count")

    def test_malformed_manifest_is_rejected_with_structured_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            write_physical_corpus(temporary, (source("E"),))
            (Path(temporary) / "corpus_manifest.json").write_text("{bad json")
            with self.assertRaises(ModelDataError) as caught:
                load_physical_examples(temporary)
        self.assertEqual(caught.exception.code, "json_loading")

    def test_encoding_variants_cannot_cross_partitions(self):
        with tempfile.TemporaryDirectory() as temporary:
            write_physical_corpus(temporary, (source("E"),))
            split_path = Path(temporary) / "manifests" / "iid.json"
            split = json.loads(split_path.read_text())
            split["samples"][0]["partition"] = "test"
            split_path.write_text(
                json.dumps(split, sort_keys=True, separators=(",", ":")) + "\n"
            )
            with self.assertRaises(ModelDataError) as caught:
                load_physical_examples(temporary)
        self.assertEqual(caught.exception.code, "split_leakage")

    def test_duplicate_or_conflicting_family_split_records_are_rejected(self):
        for partition in ("train", "test"):
            with self.subTest(partition=partition):
                with tempfile.TemporaryDirectory() as temporary:
                    write_physical_corpus(temporary, (source("E"),))
                    split_path = Path(temporary) / "manifests" / "iid.json"
                    split = json.loads(split_path.read_text())
                    duplicate = dict(split["families"][0])
                    duplicate["partition"] = partition
                    split["families"].append(duplicate)
                    split_path.write_text(
                        json.dumps(split, sort_keys=True, separators=(",", ":"))
                        + "\n"
                    )
                    with self.assertRaises(ModelDataError) as caught:
                        load_physical_examples(temporary)
                self.assertEqual(caught.exception.code, "duplicate_split_family")

    def test_every_supported_split_inherits_family_assignment(self):
        with tempfile.TemporaryDirectory() as temporary:
            write_physical_corpus(temporary, (source("E"), source("RR")))
            iid_path = Path(temporary) / "manifests" / "iid.json"
            iid = json.loads(iid_path.read_text())
            for name in (
                "operation_template",
                "history_depth",
                "geometry_extrapolation",
            ):
                copy = json.loads(json.dumps(iid))
                copy["name"] = name
                (Path(temporary) / "manifests" / (name + ".json")).write_text(
                    json.dumps(copy, sort_keys=True, separators=(",", ":")) + "\n"
                )
            for name in (
                "iid",
                "operation_template",
                "history_depth",
                "geometry_extrapolation",
            ):
                examples = load_physical_examples(temporary, name)
                self.assertEqual({item.partition for item in examples}, {"train"})

    def test_unsafe_split_and_sample_paths_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            write_physical_corpus(temporary, (source("E"),))
            with self.assertRaises(ModelDataError) as caught:
                load_physical_examples(temporary, "../iid")
            self.assertEqual(caught.exception.code, "unsafe_split_name")
            manifest_path = Path(temporary) / "corpus_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["samples"][0]["relative_json_path"] = "../outside.json"
            manifest_path.write_text(
                json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n"
            )
            with self.assertRaises(ModelDataError) as caught:
                load_physical_examples(temporary)
        self.assertEqual(caught.exception.code, "unsafe_path")

    def test_repeated_construction_is_byte_identical(self):
        with tempfile.TemporaryDirectory() as temporary:
            write_physical_corpus(temporary, (source("ER"),))
            first = load_physical_examples(temporary)
            second = load_physical_examples(temporary)
        self.assertEqual(
            tuple(canonical_record_json(item) for item in first),
            tuple(canonical_record_json(item) for item in second),
        )

    def test_duplicate_continuous_and_quantized_variants_are_rejected(self):
        for duplicated_encoding in ("continuous", "quantized"):
            with self.subTest(duplicated_encoding=duplicated_encoding):
                with tempfile.TemporaryDirectory() as temporary:
                    _, samples = write_physical_corpus(temporary, (source("E"),))
                    duplicate = dict(
                        next(
                            item
                            for item in samples
                            if item["geometry_encoding"] == duplicated_encoding
                        )
                    )
                    manifest_path = Path(temporary) / "corpus_manifest.json"
                    manifest = json.loads(manifest_path.read_text())
                    manifest["samples"].append(duplicate)
                    manifest["total_sample_variant_count"] = 3
                    manifest["families"][0]["sample_ids"].append(
                        duplicate["sample_id"]
                    )
                    manifest_path.write_text(
                        json.dumps(manifest, sort_keys=True, separators=(",", ":"))
                        + "\n"
                    )
                    with self.assertRaises(ModelDataError) as caught:
                        load_physical_examples(temporary)
                self.assertEqual(caught.exception.code, "duplicate_sample")

    def test_more_than_two_variants_are_rejected_before_collapse(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, samples = write_physical_corpus(temporary, (source("E"),))
            original = samples[0]
            duplicate = dict(original)
            duplicate["sample_id"] = "sv_extra"
            duplicate["relative_json_path"] = "samples/sv_extra.json"
            shutil.copyfile(
                Path(temporary) / original["relative_json_path"],
                Path(temporary) / duplicate["relative_json_path"],
            )
            manifest_path = Path(temporary) / "corpus_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["samples"].append(duplicate)
            manifest["total_sample_variant_count"] = 3
            manifest["families"][0]["sample_ids"].append("sv_extra")
            manifest_path.write_text(
                json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n"
            )
            split_path = Path(temporary) / "manifests" / "iid.json"
            split = json.loads(split_path.read_text())
            split["samples"].append({**duplicate, "partition": "train"})
            split_path.write_text(
                json.dumps(split, sort_keys=True, separators=(",", ":")) + "\n"
            )
            with self.assertRaises(ModelDataError) as caught:
                load_physical_examples(temporary)
        self.assertEqual(caught.exception.code, "variant_count")

    def test_shuffled_manifests_do_not_change_examples(self):
        with tempfile.TemporaryDirectory() as temporary:
            write_physical_corpus(
                temporary, (source("E"), source("R"), source("ER"))
            )
            before = tuple(
                canonical_record_json(item)
                for item in load_physical_examples(temporary)
            )
            random.Random(7).shuffle(
                corpus := json.loads(
                    (Path(temporary) / "corpus_manifest.json").read_text()
                )["samples"]
            )
            manifest_path = Path(temporary) / "corpus_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["samples"] = corpus
            random.Random(8).shuffle(manifest["families"])
            manifest_path.write_text(
                json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n"
            )
            split_path = Path(temporary) / "manifests" / "iid.json"
            split = json.loads(split_path.read_text())
            random.Random(9).shuffle(split["families"])
            random.Random(10).shuffle(split["samples"])
            split_path.write_text(
                json.dumps(split, sort_keys=True, separators=(",", ":")) + "\n"
            )
            after = tuple(
                canonical_record_json(item)
                for item in load_physical_examples(temporary)
            )
        self.assertEqual(before, after)

    def test_hash_seed_does_not_change_canonical_records(self):
        with tempfile.TemporaryDirectory() as temporary:
            write_physical_corpus(temporary, (source("E"), source("RR")))
            script = (
                "from prototype.model_data.loader import load_physical_examples;"
                "from prototype.model_data.serialization import canonical_record_json;"
                "print('\\\\n'.join(canonical_record_json(x) for x in "
                f"load_physical_examples({temporary!r})))"
            )
            outputs = []
            for seed in ("1", "997"):
                environment = dict(os.environ)
                environment["PYTHONHASHSEED"] = seed
                completed = subprocess.run(
                    [sys.executable, "-c", script],
                    cwd=str(Path(__file__).resolve().parents[3]),
                    env=environment,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True,
                )
                outputs.append(completed.stdout)
        self.assertEqual(outputs[0], outputs[1])

    def test_incomplete_split_or_orphan_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            write_physical_corpus(temporary, (source("E"),))
            split_path = Path(temporary) / "manifests" / "iid.json"
            split = json.loads(split_path.read_text())
            split["samples"].pop()
            split_path.write_text(
                json.dumps(split, sort_keys=True, separators=(",", ":")) + "\n"
            )
            with self.assertRaises(ModelDataError) as caught:
                load_physical_examples(temporary)
            self.assertEqual(caught.exception.code, "incomplete_split")
        with tempfile.TemporaryDirectory() as temporary:
            write_physical_corpus(temporary, (source("E"),))
            (Path(temporary) / "samples" / "orphan.json").write_text("{}\n")
            with self.assertRaises(ModelDataError) as caught:
                load_physical_examples(temporary)
            self.assertEqual(caught.exception.code, "corpus_not_closed")

    def test_unknown_fields_counts_and_duplicate_family_records_are_rejected(self):
        mutations = (
            ("unknown_manifest_field", lambda value: value.update({"host": "machine"})),
            (
                "inconsistent_count",
                lambda value: value.update({"total_sample_variant_count": 999}),
            ),
            (
                "duplicate_family",
                lambda value: value["families"].append(dict(value["families"][0])),
            ),
        )
        for expected, mutation in mutations:
            with self.subTest(expected=expected):
                with tempfile.TemporaryDirectory() as temporary:
                    write_physical_corpus(temporary, (source("E"),))
                    path = Path(temporary) / "corpus_manifest.json"
                    value = json.loads(path.read_text())
                    mutation(value)
                    path.write_text(
                        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
                    )
                    with self.assertRaises(ModelDataError) as caught:
                        load_physical_examples(temporary)
                self.assertEqual(caught.exception.code, expected)

    def test_unknown_sample_field_and_missing_validity_field_are_rejected(self):
        mutations = (
            (
                "unknown_manifest_field",
                lambda record: record.update({"timestamp": "never"}),
            ),
            ("malformed_sample", lambda record: record.pop("schema_valid")),
        )
        for expected, mutation in mutations:
            with self.subTest(expected=expected):
                with tempfile.TemporaryDirectory() as temporary:
                    write_physical_corpus(temporary, (source("E"),))
                    path = Path(temporary) / "corpus_manifest.json"
                    manifest = json.loads(path.read_text())
                    mutation(manifest["samples"][0])
                    path.write_text(
                        json.dumps(manifest, sort_keys=True, separators=(",", ":"))
                        + "\n"
                    )
                    with self.assertRaises(ModelDataError) as caught:
                        load_physical_examples(temporary)
                self.assertEqual(caught.exception.code, expected)


class VocabularyTests(unittest.TestCase):
    def test_ids_are_sorted_and_traversal_independent(self):
        for vocabulary in ALL_VOCABULARIES:
            self.assertEqual(vocabulary.tokens[:2], ("<pad>", "<none>"))
            self.assertEqual(vocabulary.tokens[2:], tuple(sorted(vocabulary.tokens[2:])))
            forward = {token: vocabulary.id(token) for token in vocabulary.tokens}
            reverse = {
                token: vocabulary.id(token) for token in reversed(vocabulary.tokens)
            }
            self.assertEqual(forward, reverse)


if __name__ == "__main__":
    unittest.main()
