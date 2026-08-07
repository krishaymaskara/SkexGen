"""Manifest authority and protected-partition tests for GE1 C1."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from prototype.graph_encoder.errors import GraphEncoderError
from prototype.graph_encoder import partitions
from prototype.model_data.errors import ModelDataError


def _fixture_manifest():
    counts = dict(partitions.FAMILY_COUNTS)
    template_cycles = {
        "train": ("E", "R", "EE", "RE"),
        "validation": ("E", "R", "EE", "RE"),
        "test": ("ER",),
        "secondary_systematic_validation": ("RR",),
    }
    families = []
    samples = []
    ids_by_partition = {}
    for partition in sorted(counts):
        family_ids = []
        for index in range(counts[partition]):
            family_id = "{}_family_{:04d}".format(partition, index)
            template = template_cycles[partition][index % len(template_cycles[partition])]
            sample_ids = [family_id + "_continuous", family_id + "_quantized"]
            families.append({
                "source_family_id": family_id,
                "sample_ids": sample_ids,
                "operation_template": template,
                "partition": partition,
            })
            for sample_id in sample_ids:
                samples.append({
                    "source_family_id": family_id,
                    "sample_id": sample_id,
                    "operation_template": template,
                    "partition": partition,
                })
            family_ids.append(family_id)
        ids_by_partition[partition] = tuple(sorted(family_ids))
    manifest = {
        "name": partitions.SPLIT_NAME,
        "authoritative_assignment_unit": "source_family_id",
        "partition_counts": counts,
        "families": families,
        "samples": samples,
    }
    raw = (
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    assignment_hashes = tuple(
        (name, partitions._assignment_sha256(ids_by_partition[name]))
        for name in sorted(ids_by_partition)
    )
    authority = replace(
        partitions.FROZEN_MANIFEST_AUTHORITY,
        file_sha256=hashlib.sha256(raw).hexdigest(),
        assignment_sha256=assignment_hashes,
    )
    return manifest, raw, authority, ids_by_partition


def _write_manifest(root, raw):
    path = Path(root) / partitions.AUTHORITATIVE_RELATIVE_FILE
    path.parent.mkdir(parents=True)
    path.write_bytes(raw)


def _balanced_subset(manifest, size):
    per_template = 1 if size == 4 else 8
    by_template = {
        template: [
            item["source_family_id"]
            for item in manifest["families"]
            if item["partition"] == "train"
            and item["operation_template"] == template
        ][:per_template]
        for template in ("E", "R", "EE", "RE")
    }
    return tuple(sorted(
        family_id
        for values in by_template.values()
        for family_id in values
    ))


class ManifestVerificationTests(unittest.TestCase):
    def test_frozen_manifest_literals_are_exact(self):
        self.assertEqual(partitions.SPLIT_NAME, "operation_template")
        self.assertEqual(
            partitions.AUTHORITATIVE_RELATIVE_FILE,
            "manifests/operation_template.json",
        )
        self.assertEqual(
            partitions.AUTHORITATIVE_FILE_SHA256,
            "a9ac86a6dede054fbbba57e0906b210bab26036c3f5c150b332038f78d2dadb7",
        )
        self.assertEqual(dict(partitions.FAMILY_COUNTS)["train"], 407)
        self.assertEqual(sum(dict(partitions.FAMILY_COUNTS).values()), 680)
        self.assertEqual(sum(dict(partitions.SAMPLE_COUNTS).values()), 1360)
        self.assertEqual(
            dict(partitions.ASSIGNMENT_SHA256),
            {
                "train": "42d61d2224ae1a2110279f66d374516f8b5f220271407913147d8be390c66595",
                "validation": "9af48e34e0ae2e5fd266d2336b6adecdad543c75a2c470ea132b9c7a593afaf6",
                "test": "b18df0bdf9cc95575cc79f6cd2bdded5f9559e0964305a525969a2702de22663",
                "secondary_systematic_validation": (
                    "eb37468f4baf6540891add9293a66aee2077ecd7cb64d7cb486902a6eb58c494"
                ),
            },
        )

    def test_metadata_verification_is_sorted_unique_and_disjoint(self):
        unused_manifest, raw, authority, unused_ids = _fixture_manifest()
        verified = partitions._verify_manifest_bytes(raw, authority)
        train = verified.family_ids("train")
        development = verified.family_ids("validation")
        self.assertEqual(train, tuple(sorted(train)))
        self.assertEqual(len(train), len(set(train)))
        self.assertEqual(len(development), len(set(development)))
        self.assertTrue(set(train).isdisjoint(development))

    def test_wrong_file_hash_is_terminal(self):
        unused_manifest, raw, unused_authority, unused_ids = _fixture_manifest()
        with self.assertRaises(GraphEncoderError) as caught:
            partitions._verify_manifest_bytes(
                raw, partitions.FROZEN_MANIFEST_AUTHORITY
            )
        self.assertEqual(caught.exception.code, "manifest_authority_failure")
        self.assertEqual(
            caught.exception.detail, "authoritative manifest SHA-256 mismatch"
        )

    def test_count_template_and_assignment_hash_failures(self):
        manifest, raw, authority, unused_ids = _fixture_manifest()

        count_manifest = json.loads(raw.decode("utf-8"))
        count_manifest["samples"].pop()
        count_raw = (
            json.dumps(count_manifest, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        count_authority = replace(
            authority, file_sha256=hashlib.sha256(count_raw).hexdigest()
        )

        template_manifest = json.loads(raw.decode("utf-8"))
        train_family = next(
            item
            for item in template_manifest["families"]
            if item["partition"] == "train"
        )
        train_family["operation_template"] = "RR"
        template_raw = (
            json.dumps(template_manifest, sort_keys=True, separators=(",", ":"))
            + "\n"
        ).encode("utf-8")
        template_authority = replace(
            authority, file_sha256=hashlib.sha256(template_raw).hexdigest()
        )

        bad_hashes = list(authority.assignment_sha256)
        bad_hashes[0] = (bad_hashes[0][0], "0" * 64)
        assignment_authority = replace(
            authority, assignment_sha256=tuple(bad_hashes)
        )

        for current_raw, current_authority in (
            (count_raw, count_authority),
            (template_raw, template_authority),
            (raw, assignment_authority),
        ):
            with self.subTest(authority=current_authority):
                with self.assertRaises(GraphEncoderError) as caught:
                    partitions._verify_manifest_bytes(
                        current_raw, current_authority
                    )
                self.assertEqual(
                    caught.exception.code, "manifest_authority_failure"
                )


class PartitionAccessTests(unittest.TestCase):
    def test_safe_balanced_train_subsets_and_complete_development(self):
        manifest, raw, authority, unused_ids = _fixture_manifest()
        with tempfile.TemporaryDirectory() as temporary:
            _write_manifest(temporary, raw)
            for size in (4, 32):
                subset = _balanced_subset(manifest, size)
                with mock.patch.object(
                    partitions, "FROZEN_MANIFEST_AUTHORITY", authority
                ), mock.patch.object(
                    partitions,
                    "load_partition_physical_examples",
                    return_value=("loaded",),
                ) as loader:
                    self.assertEqual(
                        partitions.load_train(temporary, subset), ("loaded",)
                    )
                    loader.assert_called_once_with(
                        temporary,
                        "operation_template",
                        "train",
                        subset,
                    )

            with mock.patch.object(
                partitions, "FROZEN_MANIFEST_AUTHORITY", authority
            ), mock.patch.object(
                partitions,
                "load_partition_physical_examples",
                return_value=("development",),
            ) as loader:
                self.assertEqual(
                    partitions.load_development(temporary), ("development",)
                )
                loader.assert_called_once_with(
                    temporary,
                    "operation_template",
                    "validation",
                    None,
                )

    def test_invalid_train_subsets_are_rejected(self):
        manifest, raw, authority, ids = _fixture_manifest()
        valid = _balanced_subset(manifest, 4)
        invalid = (
            tuple(reversed(valid)),
            valid + (valid[-1],),
            ids["validation"][:4],
            valid[:3],
        )
        with tempfile.TemporaryDirectory() as temporary:
            _write_manifest(temporary, raw)
            with mock.patch.object(
                partitions, "FROZEN_MANIFEST_AUTHORITY", authority
            ), mock.patch.object(
                partitions, "load_partition_physical_examples"
            ) as loader:
                for subset in invalid:
                    with self.subTest(subset=subset):
                        with self.assertRaises(GraphEncoderError) as caught:
                            partitions.load_train(temporary, subset)
                        self.assertEqual(caught.exception.code, "invalid_train_subset")
                loader.assert_not_called()

    def test_protected_partitions_fail_before_verification_or_payload_loading(self):
        for partition in ("test", "secondary_systematic_validation", "iid"):
            with self.subTest(partition=partition), mock.patch.object(
                partitions, "_verify_authoritative_manifest"
            ) as verifier, mock.patch.object(
                partitions, "load_partition_physical_examples"
            ) as loader:
                with self.assertRaises(GraphEncoderError) as caught:
                    partitions._load_partition("unused", partition, None)
                self.assertEqual(
                    caught.exception.code, "protected_partition_access"
                )
                verifier.assert_not_called()
                loader.assert_not_called()

    def test_absent_manifest_is_terminal_not_skipped(self):
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
            partitions, "load_partition_physical_examples"
        ) as loader:
            with self.assertRaises(GraphEncoderError) as caught:
                partitions.load_train(temporary)
        self.assertEqual(caught.exception.code, "manifest_authority_failure")
        self.assertIn("absent", caught.exception.detail)
        loader.assert_not_called()

    def test_public_api_cannot_override_split_or_partition(self):
        with self.assertRaises(TypeError):
            partitions.load_train("unused", split_name="iid")
        with self.assertRaises(TypeError):
            partitions.load_development("unused", partition="test")

    def test_model_data_error_is_not_wrapped(self):
        unused_manifest, raw, authority, unused_ids = _fixture_manifest()
        original = ModelDataError("malformed_corpus", "fixture failure")
        with tempfile.TemporaryDirectory() as temporary:
            _write_manifest(temporary, raw)
            with mock.patch.object(
                partitions, "FROZEN_MANIFEST_AUTHORITY", authority
            ), mock.patch.object(
                partitions,
                "load_partition_physical_examples",
                side_effect=original,
            ):
                with self.assertRaises(ModelDataError) as caught:
                    partitions.load_development(temporary)
        self.assertIs(caught.exception, original)

    def test_graph_encoder_error_fields_are_stable(self):
        error = GraphEncoderError("authorization_failure", "stable detail")
        self.assertEqual(error.code, "authorization_failure")
        self.assertEqual(error.detail, "stable detail")
        self.assertEqual(str(error), "authorization_failure: stable detail")


if __name__ == "__main__":
    unittest.main()
