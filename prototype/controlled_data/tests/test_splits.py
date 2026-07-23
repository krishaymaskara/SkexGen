from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
import tempfile
import unittest

from prototype.controlled_data.config import GeneratorConfig
from prototype.controlled_data.dataset import generate_corpus
from prototype.controlled_data.splits import _ranked_partition, largest_remainder_counts


class SplitManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name) / "corpus"
        generate_corpus(cls.root, GeneratorConfig(68, seed=23))
        cls.corpus = json.loads((cls.root / "corpus_manifest.json").read_text())
        cls.manifests = {
            path.stem: json.loads(path.read_text())
            for path in (cls.root / "manifests").glob("*.json")
        }
        cls.family_metadata = {
            item["source_family_id"]: item for item in cls.corpus["families"]
        }

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_iid_counts_are_exact_largest_remainder_allocations(self):
        manifest = self.manifests["iid"]
        self.assertEqual(manifest["partition_counts"], {"test": 7, "train": 54, "validation": 7})
        self.assertEqual(largest_remainder_counts(7, (0.8, 0.1, 0.1)), (5, 1, 1))
        self.assertEqual(largest_remainder_counts(5, (0.8, 0.1, 0.1)), (4, 1, 0))
        expected = {
            60: (48, 6, 6),
            61: (49, 6, 6),
            62: (50, 6, 6),
            99: (79, 10, 10),
            101: (81, 10, 10),
        }
        for total, counts in expected.items():
            self.assertEqual(largest_remainder_counts(total, (0.8, 0.1, 0.1)), counts)
            assignment = _ranked_partition(
                [f"sf_{index:03d}" for index in range(total)],
                (0.8, 0.1, 0.1),
                ("train", "validation", "test"),
                "audit-iid",
                7,
            )
            observed = Counter(assignment.values())
            self.assertEqual(
                (observed["train"], observed["validation"], observed["test"]),
                counts,
            )

    def test_each_family_is_assigned_once_and_variants_inherit_partition(self):
        for manifest in self.manifests.values():
            families = manifest["families"]
            self.assertEqual(len(families), 68)
            assignments = {item["source_family_id"]: item["partition"] for item in families}
            self.assertEqual(len(assignments), 68)
            variants_by_family: dict[str, list[dict]] = {}
            for sample in manifest["samples"]:
                variants_by_family.setdefault(sample["source_family_id"], []).append(sample)
                self.assertEqual(sample["partition"], assignments[sample["source_family_id"]])
            for family_id, variants in variants_by_family.items():
                self.assertEqual(len(variants), 2)
                self.assertEqual(len({item["partition"] for item in variants}), 1)
                self.assertEqual(
                    {item["geometry_encoding"] for item in variants},
                    {"continuous", "quantized"},
                )

    def test_split_sample_sets_are_disjoint_by_family(self):
        for manifest in self.manifests.values():
            partitions: dict[str, set[str]] = {}
            for family in manifest["families"]:
                partitions.setdefault(family["partition"], set()).add(family["source_family_id"])
            values = list(partitions.values())
            for index, left in enumerate(values):
                for right in values[index + 1 :]:
                    self.assertTrue(left.isdisjoint(right))

    def test_operation_template_split_preserves_id_validation(self):
        manifest = self.manifests["operation_template"]
        for family in manifest["families"]:
            template = self.family_metadata[family["source_family_id"]]["operation_template"]
            if family["partition"] in {"train", "validation"}:
                self.assertIn(template, {"E", "R", "EE", "RE"})
            elif family["partition"] == "test":
                self.assertEqual(template, "ER")
            else:
                self.assertEqual(family["partition"], "secondary_systematic_validation")
                self.assertEqual(template, "RR")
        self.assertGreater(manifest["partition_counts"]["validation"], 0)
        self.assertEqual(manifest["definition"]["kind"], "held_out_operation_template")
        self.assertIn("secondary_systematic_validation", manifest["partition_counts"])
        self.assertNotEqual("secondary_systematic_validation", "validation")

    def test_history_depth_split_is_enforced_and_provisional(self):
        manifest = self.manifests["history_depth"]
        self.assertEqual(manifest["definition"]["benchmark_status"], "provisional")
        for family in manifest["families"]:
            depth = self.family_metadata[family["source_family_id"]]["history_depth"]
            self.assertEqual(depth, 2 if family["partition"] == "test" else 1)

    def test_geometry_target_range_does_not_leak(self):
        manifest = self.manifests["geometry_extrapolation"]
        for family in manifest["families"]:
            extent = self.family_metadata[family["source_family_id"]][
                "history_max_sketch_extent"
            ]
            if family["partition"] == "test":
                self.assertGreaterEqual(extent, 2.5)
            else:
                self.assertLessEqual(extent, 2.0)

    def test_all_held_out_manifests_report_nontarget_coverage(self):
        expected_fields = {
            "primitive_families",
            "reference_planes",
            "directions",
            "later_boolean_modes",
            "sketch_extent_bands",
            "extrusion_distances",
            "revolution_angles",
            "extent_order_relations",
            "direction_relations",
            "boolean_extent_relations",
            "boolean_direction_relations",
        }
        for name in ("operation_template", "history_depth", "geometry_extrapolation"):
            coverage = self.manifests[name]["coverage"]
            self.assertTrue(expected_fields.issubset(coverage["train"]))
            self.assertTrue(expected_fields.issubset(coverage["test"]))
            self.assertGreater(coverage["train"]["source_family_count"], 0)
            self.assertGreater(coverage["test"]["source_family_count"], 0)
