"""Corpus-free contracts for the ADR-0014 narrow-index loader."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from prototype.graph_encoder.errors import GraphEncoderError
from prototype.graph_encoder.partitions import AUTHORITATIVE_FILE_SHA256
from prototype.graph_encoder.stage6_narrow_loader import (
    NARROW_INDEX_VERSION,
    PARTITIONS,
    _canonical,
    validate_stage6_narrow_index,
    load_stage6_train,
)
from prototype.graph_encoder.stage6_structure_only import PROTOCOL_VERSION


def _fixture(parent):
    root = Path(parent) / "payloads"
    root.mkdir()
    samples = []
    allowlist = []
    digest = hashlib.sha256(b"procedural\n").hexdigest()
    for family in ("family-a", "family-b"):
        for encoding in ("continuous", "quantized"):
            relative = family + "-" + encoding + ".json"
            (root / relative).write_bytes(b"procedural\n")
            allowlist.append(relative)
            samples.append({
                "source_family_id": family,
                "sample_id": family + "-" + encoding,
                "geometry_encoding": encoding,
                "relative_payload_path": relative,
                "payload_sha256": digest,
                "operation_template": "E",
                "primitive_family": "rectangle",
                "reference_plane": "XY",
                "history_depth": 1,
            })
    index = {
        "schema_version": NARROW_INDEX_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "partition_identity": "train",
        "parent_manifest_sha256": AUTHORITATIVE_FILE_SHA256,
        "assignment_identity": "operation_template:train",
        "assignment_sha256": "a" * 64,
        "expected_family_count": 2,
        "family_ids": ["family-a", "family-b"],
        "samples": sorted(samples, key=lambda row: (
            row["source_family_id"], row["geometry_encoding"],
            row["relative_payload_path"],
        )),
        "payload_allowlist": sorted(allowlist),
    }
    path = Path(parent) / "train-index.json"
    path.write_text(_canonical(index) + "\n", encoding="utf-8")
    return path, root, index


class NarrowLoaderContractTests(unittest.TestCase):
    def test_procedural_variants_reconstruct_one_physical_example(self):
        from prototype.controlled_data.builders import build_history
        from prototype.controlled_data.identity import sample_id, source_family_id
        from prototype.model_data.canonical import infer_metadata
        from prototype.model_data.tests.fixtures import source
        from prototype.representation.model import GeometryEncoding
        from prototype.representation.serialization import history_to_json

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "payloads"
            root.mkdir()
            histories = {
                encoding.value: build_history(source("E"), encoding)
                for encoding in (GeometryEncoding.CONTINUOUS, GeometryEncoding.QUANTIZED)
            }
            family_id = source_family_id(histories["continuous"])
            inferred = infer_metadata(histories["continuous"])
            samples = []
            allowlist = []
            for encoding in ("continuous", "quantized"):
                raw = history_to_json(histories[encoding])
                relative = encoding + ".json"
                (root / relative).write_text(raw, encoding="utf-8")
                allowlist.append(relative)
                samples.append({
                    "source_family_id": family_id,
                    "sample_id": sample_id(histories[encoding]),
                    "geometry_encoding": encoding,
                    "relative_payload_path": relative,
                    "payload_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
                    "operation_template": inferred[0],
                    "primitive_family": inferred[1],
                    "reference_plane": inferred[2],
                    "history_depth": 1,
                })
            document = {
                "schema_version": NARROW_INDEX_VERSION,
                "protocol_version": PROTOCOL_VERSION,
                "partition_identity": "train",
                "parent_manifest_sha256": AUTHORITATIVE_FILE_SHA256,
                "assignment_identity": "operation_template:train",
                "assignment_sha256": "a" * 64,
                "expected_family_count": 1,
                "family_ids": [family_id],
                "samples": sorted(samples, key=lambda row: (
                    row["source_family_id"], row["geometry_encoding"],
                    row["relative_payload_path"],
                )),
                "payload_allowlist": sorted(allowlist),
            }
            index = Path(temporary) / "index.json"
            index.write_text(_canonical(document) + "\n", encoding="utf-8")
            with mock.patch.dict(PARTITIONS, {"train": {
                "physical_partition": "train", "family_count": 1,
                "assignment_sha256": "a" * 64,
            }}):
                examples, evidence = load_stage6_train(index, root)
            self.assertEqual(len(examples), 1)
            self.assertEqual(examples[0].physical_family_id, family_id)
            self.assertEqual(evidence["verification_status"], "pass")

    def test_exact_sorted_allowlist_and_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            index, root, unused = _fixture(temporary)
            with mock.patch.dict(PARTITIONS, {"train": {
                "physical_partition": "train", "family_count": 2,
                "assignment_sha256": "a" * 64,
            }}):
                document, grouped, evidence = validate_stage6_narrow_index(
                    index, root, "train"
                )
            self.assertEqual(document["expected_family_count"], 2)
            self.assertEqual(set(grouped), {"family-a", "family-b"})
            self.assertEqual(evidence["verification_status"], "pass")
            self.assertEqual(evidence["expected_allowlist_sha256"],
                             evidence["observed_allowlist_sha256"])

    def test_unexpected_file_absolute_parent_and_duplicate_variant_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            index, root, document = _fixture(temporary)
            policy = {"train": {
                "physical_partition": "train", "family_count": 2,
                "assignment_sha256": "a" * 64,
            }}
            (root / "unexpected.json").write_text("{}\n", encoding="utf-8")
            with mock.patch.dict(PARTITIONS, policy), self.assertRaises(GraphEncoderError):
                validate_stage6_narrow_index(index, root, "train")
            (root / "unexpected.json").unlink()
            broken = copy.deepcopy(document)
            broken["samples"][1]["geometry_encoding"] = "continuous"
            index.write_text(_canonical(broken) + "\n", encoding="utf-8")
            with mock.patch.dict(PARTITIONS, policy), self.assertRaises(GraphEncoderError):
                validate_stage6_narrow_index(index, root, "train")

    def test_symlink_root_and_unresolved_hash_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            index, root, document = _fixture(temporary)
            policy = {"train": {
                "physical_partition": "train", "family_count": 2,
                "assignment_sha256": "a" * 64,
            }}
            link = Path(temporary) / "narrow-link"
            link.symlink_to(root, target_is_directory=True)
            with mock.patch.dict(PARTITIONS, policy), self.assertRaises(GraphEncoderError):
                validate_stage6_narrow_index(index, link, "train")
            document["samples"][0]["payload_sha256"] = "UNRESOLVED"
            index.write_text(_canonical(document) + "\n", encoding="utf-8")
            with mock.patch.dict(PARTITIONS, policy), self.assertRaises(GraphEncoderError):
                validate_stage6_narrow_index(index, root, "train")


if __name__ == "__main__":
    unittest.main()
