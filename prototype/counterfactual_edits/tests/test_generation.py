from __future__ import annotations

import getpass
import json
from pathlib import Path
import shutil
import socket
import tempfile
import unittest
from unittest.mock import patch

from prototype.kernel_validation.corpus import load_corpus

from prototype.counterfactual_edits import dataset as dataset_module
from prototype.counterfactual_edits.candidates import (
    CoverageSelectionError,
    candidate_counts,
    select_oriented_candidates,
)
from prototype.counterfactual_edits.config import CounterfactualConfig
from prototype.counterfactual_edits.dataset import generate_corpus


class GenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        root = Path(cls.temporary.name)
        cls.first = root / "first"
        cls.second = root / "second"
        cls.config = CounterfactualConfig(68, seed=17)
        cls.selected = select_oriented_candidates(cls.config)
        cls.counts = candidate_counts(cls.config)
        with patch(
            "prototype.counterfactual_edits.dataset.select_oriented_candidates",
            return_value=cls.selected,
        ), patch(
            "prototype.counterfactual_edits.dataset.candidate_counts",
            return_value=cls.counts,
        ):
            cls.manifest = generate_corpus(cls.first, cls.config)
            generate_corpus(cls.second, cls.config)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_exact_dataset_counts_and_layout(self):
        self.assertEqual(self.manifest["total_edit_family_count"], 68)
        self.assertEqual(self.manifest["total_edit_sample_count"], 136)
        self.assertEqual(self.manifest["total_physical_endpoint_count"], 136)
        self.assertEqual(self.manifest["total_endpoint_sample_count"], 272)
        self.assertEqual(len(list(self.first.rglob("*.json"))), 415)
        by_family = {}
        for sample in self.manifest["samples"]:
            by_family.setdefault(sample["edit_family_id"], []).append(sample)
        self.assertEqual(len(by_family), 68)
        for variants in by_family.values():
            self.assertEqual(len(variants), 2)
            self.assertEqual(
                {item["geometry_encoding"] for item in variants},
                {"continuous", "quantized"},
            )
            self.assertEqual(len({item["edit_family_id"] for item in variants}), 1)
            self.assertEqual(len({item["edit_sample_id"] for item in variants}), 2)

    def test_repeated_generation_is_byte_identical(self):
        snapshot = lambda root: {
            path.relative_to(root).as_posix(): path.read_bytes()
            for path in root.rglob("*")
            if path.is_file()
        }
        self.assertEqual(snapshot(self.first), snapshot(self.second))

    def test_endpoint_manifest_is_kernel_loader_compatible(self):
        metadata, samples = load_corpus(self.first)
        self.assertEqual(len(samples), 272)
        self.assertEqual(metadata["total_source_family_count"], 136)
        self.assertTrue(all(sample.loading_error is None for sample in samples))

    def test_manifests_are_environment_sanitized(self):
        forbidden = (
            str(self.first),
            getpass.getuser(),
            socket.gethostname(),
            "timestamp",
        )
        for path in (
            self.first / "counterfactual_manifest.json",
            self.first / "corpus_manifest.json",
            *(self.first / "manifests").glob("*.json"),
        ):
            payload = path.read_text()
            for value in forbidden:
                self.assertNotIn(value, payload)

    def test_undersized_request_has_structured_coverage_error(self):
        with self.assertRaises(CoverageSelectionError) as raised:
            from prototype.counterfactual_edits.candidates import select_oriented_candidates

            select_oriented_candidates(CounterfactualConfig(67))
        self.assertEqual(raised.exception.derived_minimum, 68)
        self.assertEqual(len(raised.exception.missing_primary_tokens), 68)

    def test_atomic_cleanup_after_write_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "failed"
            with patch(
                "prototype.counterfactual_edits.dataset._write_json",
                side_effect=OSError("manifest write failed"),
            ), patch(
                "prototype.counterfactual_edits.dataset.select_oriented_candidates",
                return_value=self.selected,
            ), patch(
                "prototype.counterfactual_edits.dataset.candidate_counts",
                return_value=self.counts,
            ):
                with self.assertRaises(OSError):
                    generate_corpus(output, self.config)
            self.assertFalse(output.exists())
            self.assertEqual(list(Path(temporary).iterdir()), [])

    def test_closed_tree_rejects_orphans_missing_files_and_corruption(self):
        corruptions = (
            ("orphan_history", self._orphan_history),
            ("orphan_pair", self._orphan_pair),
            ("missing_history", self._missing_history),
            ("missing_pair", self._missing_pair),
            ("pair_endpoint", self._corrupt_pair_endpoint),
            ("pair_identity", self._corrupt_pair_identity),
            ("locality", self._corrupt_pair_locality),
            ("duplicate_reference", self._duplicate_manifest_reference),
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, corrupt in corruptions:
                with self.subTest(case=name):
                    copy = root / name
                    shutil.copytree(self.first, copy)
                    corrupt(copy)
                    args = self._verification_inputs(copy)
                    with self.assertRaises(Exception):
                        dataset_module._verify_tree(copy, *args)

    def test_cleanup_after_every_publication_failure_stage(self):
        original_write_text = dataset_module._write_text

        def fail_for_directory(directory):
            def failing(path, payload):
                if path.parent.name == directory:
                    raise OSError(f"{directory} write failed")
                return original_write_text(path, payload)

            return failing

        stages = (
            (
                "history_write",
                patch(
                    "prototype.counterfactual_edits.dataset._write_text",
                    side_effect=fail_for_directory("histories"),
                ),
            ),
            (
                "pair_write",
                patch(
                    "prototype.counterfactual_edits.dataset._write_text",
                    side_effect=fail_for_directory("pairs"),
                ),
            ),
            (
                "manifest_write",
                patch(
                    "prototype.counterfactual_edits.dataset._write_json",
                    side_effect=OSError("manifest write failed"),
                ),
            ),
            (
                "final_verification",
                patch(
                    "prototype.counterfactual_edits.dataset._verify_tree",
                    side_effect=RuntimeError("verification failed"),
                ),
            ),
            (
                "rename",
                patch(
                    "prototype.counterfactual_edits.dataset.os.rename",
                    side_effect=OSError("rename failed"),
                ),
            ),
        )
        for name, failure_patch in stages:
            with self.subTest(stage=name), tempfile.TemporaryDirectory() as temporary:
                output = Path(temporary) / "failed"
                with patch(
                    "prototype.counterfactual_edits.dataset.select_oriented_candidates",
                    return_value=self.selected,
                ), patch(
                    "prototype.counterfactual_edits.dataset.candidate_counts",
                    return_value=self.counts,
                ), failure_patch:
                    with self.assertRaises(Exception):
                        generate_corpus(output, self.config)
                self.assertFalse(output.exists())
                self.assertEqual(list(Path(temporary).iterdir()), [])

    @staticmethod
    def _verification_inputs(root):
        counterfactual = json.loads(
            (root / "counterfactual_manifest.json").read_text()
        )
        corpus = json.loads((root / "corpus_manifest.json").read_text())
        splits = {
            name: json.loads((root / "manifests" / f"{name}.json").read_text())
            for name in (
                "iid",
                "operation_template",
                "history_depth",
                "geometry_extrapolation",
            )
        }
        exclusions = json.loads(
            (root / "manifests" / "endpoint_exclusions.json").read_text()
        )
        return counterfactual, corpus, splits, exclusions, CounterfactualConfig(
            68, seed=17
        )

    @staticmethod
    def _canonical_write(path, value):
        path.write_text(
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        )

    def _orphan_history(self, root):
        self._canonical_write(root / "histories" / "orphan.json", {})

    def _orphan_pair(self, root):
        self._canonical_write(root / "pairs" / "orphan.json", {})

    @staticmethod
    def _missing_history(root):
        next((root / "histories").glob("*.json")).unlink()

    @staticmethod
    def _missing_pair(root):
        next((root / "pairs").glob("*.json")).unlink()

    def _mutate_first_pair(self, root, mutate):
        path = next((root / "pairs").glob("*.json"))
        value = json.loads(path.read_text())
        mutate(value)
        self._canonical_write(path, value)

    def _corrupt_pair_endpoint(self, root):
        self._mutate_first_pair(
            root, lambda value: value["base"].update(sample_id="sv_wrong")
        )

    def _corrupt_pair_identity(self, root):
        self._mutate_first_pair(
            root, lambda value: value.update(edit_sample_id="ev_wrong")
        )

    def _corrupt_pair_locality(self, root):
        self._mutate_first_pair(
            root,
            lambda value: value["locality"]["expected_unchanged_paths"].pop(),
        )

    def _duplicate_manifest_reference(self, root):
        path = root / "corpus_manifest.json"
        value = json.loads(path.read_text())
        value["samples"][1]["relative_json_path"] = value["samples"][0][
            "relative_json_path"
        ]
        self._canonical_write(path, value)
