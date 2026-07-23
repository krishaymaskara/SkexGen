from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from prototype.controlled_data.splits import SPLIT_POLICY_VERSION

from prototype.counterfactual_edits.config import CounterfactualConfig
from prototype.counterfactual_edits.dataset import generate_corpus


class SplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name) / "corpus"
        generate_corpus(cls.root, CounterfactualConfig(68, seed=23))
        cls.manifests = {
            path.stem: json.loads(path.read_text())
            for path in (cls.root / "manifests").glob("*.json")
        }

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_families_samples_and_endpoints_inherit_one_partition(self):
        for name in (
            "iid",
            "operation_template",
            "history_depth",
            "geometry_extrapolation",
        ):
            manifest = self.manifests[name]
            self.assertEqual(manifest["split_policy_version"], SPLIT_POLICY_VERSION)
            assignments = {
                item["edit_family_id"]: item["partition"]
                for item in manifest["families"]
            }
            self.assertEqual(len(assignments), 68)
            for collection in ("samples", "endpoints"):
                for item in manifest[collection]:
                    self.assertEqual(
                        item["partition"], assignments[item["edit_family_id"]]
                    )

    def test_endpoint_exclusions_are_complete_sorted_and_physical(self):
        exclusions = self.manifests["endpoint_exclusions"]
        union = set()
        for partitions in exclusions["strategies"].values():
            for values in partitions.values():
                self.assertEqual(values, sorted(values))
                union.update(values)
        self.assertEqual(
            exclusions["all_evaluation_endpoint_source_family_ids"],
            sorted(union),
        )
        self.assertTrue(all(value.startswith("sf_") for value in union))

    def test_existing_held_out_definitions_are_reused(self):
        operation = self.manifests["operation_template"]["definition"]
        self.assertEqual(operation["train_validation_templates"], ["E", "R", "EE", "RE"])
        self.assertEqual(operation["test_templates"], ["ER"])
        self.assertEqual(
            operation["secondary_systematic_validation_templates"], ["RR"]
        )
