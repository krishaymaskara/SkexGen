"""Procedural contracts for the governed ADR-0014 narrow builder."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from prototype.graph_encoder.errors import GraphEncoderError
from prototype.graph_encoder.stage6_narrow_builder import (
    _authoritative_examples,
    _canonical,
    _loads,
    build_stage6_narrow_packages,
    verify_preparation_receipt,
)
from prototype.model_data.loader import load_partition_physical_examples
from prototype.model_data.tests.fixtures import source, write_physical_corpus


def _fixture(parent):
    root = Path(parent) / "source-corpus"
    sources = (source("E", extents=(1.0,)), source("R", parameters=(45.0,)))
    families, unused = write_physical_corpus(root, sources)
    family_ids = [row["source_family_id"] for row in families]
    corpus = json.loads((root / "corpus_manifest.json").read_text())
    split = json.loads((root / "manifests/iid.json").read_text())
    partition_by_family = {family_ids[0]: "train", family_ids[1]: "validation"}
    for row in split["families"] + split["samples"]:
        row["partition"] = partition_by_family[row["source_family_id"]]
    (root / "manifests/iid.json").write_text(_canonical(split) + "\n")
    authority = b"procedural-authority\n"
    (root / "manifests/operation_template.json").write_bytes(authority)
    digest = hashlib.sha256(authority).hexdigest()
    output = Path(parent) / "output"
    output.mkdir()
    return root, output, digest


def _examples(root, partition):
    physical = "train" if partition == "train" else "validation"
    examples = load_partition_physical_examples(root, "iid", physical)
    corpus = json.loads((Path(root) / "corpus_manifest.json").read_text())
    wanted = {
        sample for example in examples for sample in example.metadata.sample_ids
    }
    records = {
        row["sample_id"]: row for row in corpus["samples"]
        if row["sample_id"] in wanted
    }
    return examples, records


class NarrowBuilderContractTests(unittest.TestCase):
    def _patches(self, digest):
        policy = {
            "train": {
                "physical_partition": "train", "family_count": 1,
                "assignment_sha256": "1" * 64,
            },
            "development": {
                "physical_partition": "validation", "family_count": 1,
                "assignment_sha256": "2" * 64,
            },
        }
        return (
            mock.patch(
                "prototype.graph_encoder.stage6_narrow_builder.AUTHORITATIVE_FILE_SHA256",
                digest,
            ),
            mock.patch(
                "prototype.graph_encoder.stage6_narrow_loader.AUTHORITATIVE_FILE_SHA256",
                digest,
            ),
            mock.patch.dict(
                "prototype.graph_encoder.stage6_narrow_loader.PARTITIONS", policy,
                clear=True,
            ),
            mock.patch(
                "prototype.graph_encoder.stage6_narrow_builder._authoritative_examples",
                side_effect=lambda root, partition: _examples(root, partition),
            ),
        )

    def _build(self, root, output, digest, partition="both", job="123"):
        patches = self._patches(digest)
        with patches[0], patches[1], patches[2], patches[3]:
            return build_stage6_narrow_packages(
                corpus_root=root, output_parent=output, job_id=job,
                expected_source_manifest_sha256=digest, partition=partition,
            )

    def test_both_packages_are_canonical_loader_accepted_and_deterministic(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, output, digest = _fixture(temporary)
            receipt = self._build(root, output, digest)
            self.assertEqual(set(receipt["packages"]), {"train", "development"})
            self.assertEqual(receipt["packages"]["train"]["sample_count"], 2)
            self.assertFalse(receipt["source_absolute_payload_paths_recorded"])
            first = (output / "stage6-train/index.json").read_bytes()
            second_parent = Path(temporary) / "second"
            second_parent.mkdir()
            second = self._build(root, second_parent, digest)
            self.assertEqual(first, (second_parent / "stage6-train/index.json").read_bytes())
            self.assertEqual(receipt, second)

    def test_exact_pairing_and_payload_names_do_not_use_source_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, output, digest = _fixture(temporary)
            self._build(root, output, digest, partition="train")
            index = json.loads((output / "stage6-train/index.json").read_text())
            self.assertEqual(
                {row["geometry_encoding"] for row in index["samples"]},
                {"continuous", "quantized"},
            )
            self.assertTrue(all(path.startswith("families/") for path in index["payload_allowlist"]))
            self.assertFalse(any("samples/" in path for path in index["payload_allowlist"]))

    def test_wrong_manifest_hash_and_existing_destination_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, output, digest = _fixture(temporary)
            with self.assertRaises(GraphEncoderError):
                build_stage6_narrow_packages(
                    corpus_root=root, output_parent=output, job_id="123",
                    expected_source_manifest_sha256="0" * 64, partition="train",
                )
            self._build(root, output, digest, partition="train")
            with self.assertRaises(GraphEncoderError):
                self._build(root, output, digest, partition="train", job="124")

    def test_source_and_output_symlinks_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, output, digest = _fixture(temporary)
            source_link = Path(temporary) / "source-link"
            source_link.symlink_to(root, target_is_directory=True)
            patches = self._patches(digest)
            with patches[0], patches[1], patches[2], patches[3], self.assertRaises(Exception):
                build_stage6_narrow_packages(
                    corpus_root=source_link, output_parent=output, job_id="123",
                    expected_source_manifest_sha256=digest, partition="train",
                )
            output_link = Path(temporary) / "output-link"
            output_link.symlink_to(output, target_is_directory=True)
            with patches[0], patches[1], patches[2], patches[3], self.assertRaises(GraphEncoderError):
                build_stage6_narrow_packages(
                    corpus_root=root, output_parent=output_link, job_id="123",
                    expected_source_manifest_sha256=digest, partition="train",
                )

    def test_absolute_parent_and_source_payload_symlink_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, output, digest = _fixture(temporary)
            examples, records = _examples(root, "train")
            broken = copy.deepcopy(records)
            first = next(iter(broken.values()))
            first["relative_json_path"] = str((root / "samples/absolute.json").resolve())
            patches = self._patches(digest)
            with patches[0], patches[1], patches[2], mock.patch(
                "prototype.graph_encoder.stage6_narrow_builder._authoritative_examples",
                return_value=(examples, broken),
            ), self.assertRaises(GraphEncoderError):
                build_stage6_narrow_packages(
                    corpus_root=root, output_parent=output, job_id="123",
                    expected_source_manifest_sha256=digest, partition="train",
                )
            original = next(iter(records.values()))["relative_json_path"]
            target = root / original
            payload = target.read_bytes()
            target.unlink()
            replacement = root / "replacement.json"
            replacement.write_bytes(payload)
            target.symlink_to(replacement)
            with patches[0], patches[1], patches[2], mock.patch(
                "prototype.graph_encoder.stage6_narrow_builder._authoritative_examples",
                return_value=(examples, records),
            ), self.assertRaises(GraphEncoderError):
                build_stage6_narrow_packages(
                    corpus_root=root, output_parent=output, job_id="124",
                    expected_source_manifest_sha256=digest, partition="train",
                )

    def test_duplicate_missing_encoding_identity_and_metadata_disagreement_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, output, digest = _fixture(temporary)
            examples, records = _examples(root, "train")
            cases = []
            missing = dict(records)
            missing.pop(next(iter(missing)))
            cases.append(missing)
            identity = copy.deepcopy(records)
            next(iter(identity.values()))["sample_id"] = "wrong-id"
            cases.append(identity)
            metadata_example = copy.deepcopy(examples)
            from dataclasses import replace
            metadata_example = (replace(
                examples[0], metadata=replace(examples[0].metadata, primitive_family="wrong")
            ),)
            for index, value in enumerate(cases):
                out = Path(temporary) / ("out-" + str(index))
                out.mkdir()
                patches = self._patches(digest)
                with patches[0], patches[1], patches[2], mock.patch(
                    "prototype.graph_encoder.stage6_narrow_builder._authoritative_examples",
                    return_value=(examples, value),
                ), self.assertRaises(GraphEncoderError):
                    build_stage6_narrow_packages(
                        corpus_root=root, output_parent=out, job_id="12" + str(index),
                        expected_source_manifest_sha256=digest, partition="train",
                    )
            metadata_out = Path(temporary) / "metadata-out"
            metadata_out.mkdir()
            patches = self._patches(digest)
            with patches[0], patches[1], patches[2], mock.patch(
                "prototype.graph_encoder.stage6_narrow_builder._authoritative_examples",
                return_value=(metadata_example, records),
            ), self.assertRaises(GraphEncoderError):
                build_stage6_narrow_packages(
                    corpus_root=root, output_parent=metadata_out, job_id="129",
                    expected_source_manifest_sha256=digest, partition="train",
                )

    def test_protected_template_and_wrong_assignment_policy_fail(self):
        from dataclasses import replace
        with tempfile.TemporaryDirectory() as temporary:
            root, output, digest = _fixture(temporary)
            examples, records = _examples(root, "train")
            protected = (replace(
                examples[0], metadata=replace(examples[0].metadata, operation_template="RR")
            ),)
            patches = self._patches(digest)
            with patches[0], patches[1], patches[2], mock.patch(
                "prototype.graph_encoder.stage6_narrow_builder._authoritative_examples",
                return_value=(protected, records),
            ), self.assertRaises(GraphEncoderError):
                build_stage6_narrow_packages(
                    corpus_root=root, output_parent=output, job_id="123",
                    expected_source_manifest_sha256=digest, partition="train",
                )

    def test_two_package_failure_exposes_neither_final_and_retains_job_staging(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, output, digest = _fixture(temporary)
            from prototype.graph_encoder import stage6_narrow_builder as builder
            original = builder._package_index
            def fail_second(*args, **kwargs):
                if args[2] == "development":
                    raise GraphEncoderError("fixture_failure", "development")
                return original(*args, **kwargs)
            patches = self._patches(digest)
            with patches[0], patches[1], patches[2], patches[3], mock.patch.object(
                builder, "_package_index", side_effect=fail_second
            ), self.assertRaises(GraphEncoderError):
                builder.build_stage6_narrow_packages(
                    corpus_root=root, output_parent=output, job_id="987",
                    expected_source_manifest_sha256=digest, partition="both",
                )
            self.assertFalse((output / "stage6-train").exists())
            self.assertFalse((output / "stage6-development").exists())
            self.assertTrue((output / "stage6-train.incomplete-987").is_dir())

    def test_receipt_detects_payload_tampering_and_unexpected_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, output, digest = _fixture(temporary)
            self._build(root, output, digest)
            payload = next((output / "stage6-train/payloads").rglob("*.json"))
            payload.write_bytes(b"{}\n")
            patches = self._patches(digest)
            with patches[0], patches[1], patches[2], self.assertRaises(GraphEncoderError):
                verify_preparation_receipt(output)
        with tempfile.TemporaryDirectory() as temporary:
            root, output, digest = _fixture(temporary)
            self._build(root, output, digest)
            unexpected = output / "stage6-development/payloads/unexpected.json"
            unexpected.write_bytes(b"{}\n")
            patches = self._patches(digest)
            with patches[0], patches[1], patches[2], self.assertRaises(GraphEncoderError):
                verify_preparation_receipt(output)
        with tempfile.TemporaryDirectory() as temporary:
            root, output, digest = _fixture(temporary)
            self._build(root, output, digest)
            receipt_path = output / "stage6-preparation-receipt.json"
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt["source_access"]["rr_accessed"] = True
            receipt_path.write_text(_canonical(receipt) + "\n", encoding="utf-8")
            patches = self._patches(digest)
            with patches[0], patches[1], patches[2], self.assertRaises(GraphEncoderError):
                verify_preparation_receipt(output)

    def test_malformed_duplicate_nonfinite_json_and_bad_job_id_fail(self):
        for raw in ('{"a":1,"a":2}', '{"a":NaN}', '{'):
            with self.assertRaises(GraphEncoderError):
                _loads(raw, "fixture")
        with tempfile.TemporaryDirectory() as temporary:
            root, output, digest = _fixture(temporary)
            patches = self._patches(digest)
            with patches[0], patches[1], patches[2], patches[3], self.assertRaises(GraphEncoderError):
                build_stage6_narrow_packages(
                    corpus_root=root, output_parent=output, job_id="not-decimal",
                    expected_source_manifest_sha256=digest, partition="train",
                )


if __name__ == "__main__":
    unittest.main()
