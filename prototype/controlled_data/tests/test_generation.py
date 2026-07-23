from __future__ import annotations

from dataclasses import replace
import json
import getpass
from pathlib import Path
import random
import socket
import tempfile
import unittest
from unittest.mock import patch

from prototype.representation.model import GeometryEncoding
from prototype.representation.model import ExtrudeGeometry, NumericValue
from prototype.representation.serialization import history_from_json, history_to_json
from prototype.representation.validation import validate_history

from prototype.controlled_data.builders import build_history
from prototype.controlled_data.config import (
    PHYSICAL_ID_DECIMAL_PLACES,
    ConfigurationError,
    GeneratorConfig,
    required_coverage_family_count,
)
from prototype.controlled_data.dataset import GenerationError, generate_corpus
from prototype.controlled_data.factors import (
    _coverage_anchor_indices,
    _mixed_radix_decode,
    _mixed_radix_encode,
    _permuted_index,
    factor_blocks,
    select_sources,
    total_candidate_count,
)
from prototype.controlled_data.identity import (
    canonical_physical_source_bytes,
    sample_id,
    source_family_id,
)


class GeneratedCorpusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        root = Path(cls.temporary.name)
        cls.first = root / "first"
        cls.second = root / "second"
        cls.config = GeneratorConfig(num_source_families=60, seed=17)
        generate_corpus(cls.first, cls.config)
        generate_corpus(cls.second, cls.config)
        cls.corpus = json.loads((cls.first / "corpus_manifest.json").read_text())

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_requested_family_count_produces_exactly_two_variants_each(self):
        self.assertEqual(self.corpus["total_source_family_count"], 60)
        self.assertEqual(self.corpus["total_sample_variant_count"], 120)
        self.assertEqual(len(self.corpus["families"]), 60)
        self.assertEqual(len(self.corpus["samples"]), 120)
        for family in self.corpus["families"]:
            self.assertEqual(len(family["sample_ids"]), 2)
            variants = [
                sample
                for sample in self.corpus["samples"]
                if sample["source_family_id"] == family["source_family_id"]
            ]
            self.assertEqual(
                {sample["geometry_encoding"] for sample in variants},
                {"continuous", "quantized"},
            )

    def test_repeated_runs_are_byte_identical(self):
        first = {
            path.relative_to(self.first).as_posix(): path.read_bytes()
            for path in self.first.rglob("*")
            if path.is_file()
        }
        second = {
            path.relative_to(self.second).as_posix(): path.read_bytes()
            for path in self.second.rglob("*")
            if path.is_file()
        }
        self.assertEqual(first, second)

    def test_all_histories_validate_and_round_trip_canonically(self):
        for record in self.corpus["samples"]:
            payload = (self.first / record["relative_json_path"]).read_text().rstrip("\n")
            history = history_from_json(payload)
            validate_history(history)
            self.assertEqual(history_to_json(history), payload)
            self.assertEqual(sample_id(history), record["sample_id"])
            self.assertEqual(source_family_id(history), record["source_family_id"])
            self.assertTrue(record["schema_valid"])
            self.assertTrue(record["serialization_round_trip_valid"])
            self.assertEqual(record["kernel_status"], "not_checked")

    def test_ids_and_sample_json_do_not_contain_selection_metadata(self):
        for record in self.corpus["samples"]:
            payload = (self.first / record["relative_json_path"]).read_text()
            self.assertNotIn("selection_order", payload)
            self.assertNotIn("generation_seed", payload)
        self.assertEqual(self.corpus["generation_seed"], 17)
        self.assertIsInstance(self.corpus["normalized_configuration"]["seed"], int)
        self.assertIsInstance(
            self.corpus["normalized_configuration"]["num_source_families"], int
        )

    def test_corpus_reproducibility_metadata_is_complete(self):
        expected = {
            "generator_version",
            "representation_schema_version",
            "canonicalization_version",
            "normalized_configuration",
            "configuration_sha256",
            "generation_seed",
            "total_source_family_count",
            "total_sample_variant_count",
        }
        self.assertTrue(expected.issubset(self.corpus))
        serialized = (self.first / "corpus_manifest.json").read_text()
        self.assertNotIn(str(self.first), serialized)
        self.assertNotIn("timestamp", serialized)

    def test_same_physical_history_has_same_family_id_across_encodings(self):
        source = select_sources(self.config)[0]
        continuous = build_history(source, GeometryEncoding.CONTINUOUS)
        quantized = build_history(source, GeometryEncoding.QUANTIZED)
        self.assertEqual(
            canonical_physical_source_bytes(continuous),
            canonical_physical_source_bytes(quantized),
        )
        self.assertEqual(source_family_id(continuous), source_family_id(quantized))
        self.assertNotEqual(sample_id(continuous), sample_id(quantized))

    def test_negative_zero_is_normalized_in_physical_identity(self):
        source = select_sources(self.config)[0]
        history = build_history(source, GeometryEncoding.CONTINUOUS)
        plane_record = next(item for item in history.geometry.node_geometry if item.node_id == "plane_1")
        changed_plane = replace(
            plane_record.geometry,
            origin=replace(plane_record.geometry.origin, values=(-0.0, 0.0, -0.0)),
        )
        changed_records = tuple(
            replace(item, geometry=changed_plane) if item.node_id == "plane_1" else item
            for item in history.geometry.node_geometry
        )
        changed = replace(history, geometry=replace(history.geometry, node_geometry=changed_records))
        self.assertEqual(source_family_id(history), source_family_id(changed))

    def test_affine_decode_noise_is_normalized_in_physical_identity(self):
        source = next(
            item
            for item in select_sources(self.config)
            if item.operation_template.value == "E"
        )
        continuous = build_history(source, GeometryEncoding.CONTINUOUS)
        quantized = build_history(source, GeometryEncoding.QUANTIZED)

        def replace_distance(history, value):
            records = tuple(
                replace(record, geometry=ExtrudeGeometry(value))
                if record.node_id == "extrude_1"
                else record
                for record in history.geometry.node_geometry
            )
            return replace(history, geometry=replace(history.geometry, node_geometry=records))

        continuous = replace_distance(continuous, NumericValue.continuous(0.3))
        quantized = replace_distance(
            quantized, NumericValue.quantized((3,), scale=0.1, offset=0.0)
        )
        self.assertEqual(source_family_id(continuous), source_family_id(quantized))

    def test_integer_and_float_physical_values_have_identical_ids(self):
        source = next(item for item in select_sources(self.config) if item.operation_template.value == "E")
        history = build_history(source, GeometryEncoding.CONTINUOUS)

        def replace_distance(value):
            records = tuple(
                replace(record, geometry=ExtrudeGeometry(NumericValue.continuous(value)))
                if record.node_id == "extrude_1"
                else record
                for record in history.geometry.node_geometry
            )
            return replace(history, geometry=replace(history.geometry, node_geometry=records))

        integer_history = replace_distance(1)
        float_history = replace_distance(1.0)
        self.assertEqual(source_family_id(integer_history), source_family_id(float_history))
        self.assertEqual(sample_id(integer_history), sample_id(float_history))

    def test_ids_are_independent_of_seed_selection_and_path_metadata(self):
        block_seed_one = factor_blocks(GeneratorConfig(60, seed=1))[0]
        block_seed_two = factor_blocks(GeneratorConfig(60, seed=999))[0]
        source_one = block_seed_one.decode(42)
        source_two = block_seed_two.decode(42)
        self.assertEqual(source_one, source_two)
        history_one = build_history(source_one, GeometryEncoding.CONTINUOUS)
        history_two = build_history(source_two, GeometryEncoding.CONTINUOUS)
        self.assertEqual(source_family_id(history_one), source_family_id(history_two))
        self.assertEqual(sample_id(history_one), sample_id(history_two))

        record = {
            "source_family_id": source_family_id(history_one),
            "sample_id": sample_id(history_one),
            "selection_order": 123,
            "partition": "test",
            "relative_json_path": "/different/location.json",
        }
        self.assertEqual(record["source_family_id"], source_family_id(history_one))
        self.assertEqual(record["sample_id"], sample_id(history_one))

    def test_no_final_directory_when_validation_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "failed"

            def invalid_builder(source, encoding):
                return replace(build_history(source, encoding), schema_version=2)

            with self.assertRaises(Exception):
                generate_corpus(output, self.config, history_builder=invalid_builder)
            self.assertFalse(output.exists())
            self.assertEqual(list(Path(temporary).iterdir()), [])

    def test_sample_id_collision_is_detected_and_cleaned_up(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "collision"
            with patch("prototype.controlled_data.dataset.sample_id", return_value="sv_collision"):
                with self.assertRaisesRegex(GenerationError, "duplicate sample ID"):
                    generate_corpus(output, self.config)
            self.assertFalse(output.exists())
            self.assertEqual(list(Path(temporary).iterdir()), [])

    def test_source_family_id_collision_is_detected_and_cleaned_up(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "collision"
            with patch(
                "prototype.controlled_data.dataset.source_family_id",
                return_value="sf_collision",
            ):
                with self.assertRaisesRegex(GenerationError, "duplicate source family ID"):
                    generate_corpus(output, self.config)
            self.assertFalse(output.exists())
            self.assertEqual(list(Path(temporary).iterdir()), [])

    def test_canonical_manifests_exclude_environment_and_path_metadata(self):
        manifest_paths = [self.first / "corpus_manifest.json"] + sorted(
            (self.first / "manifests").glob("*.json")
        )
        forbidden = (
            str(self.first),
            str(Path(tempfile.gettempdir()).resolve()),
            getpass.getuser(),
            socket.gethostname(),
            "timestamp",
        )
        for path in manifest_paths:
            payload = path.read_text()
            for value in forbidden:
                with self.subTest(path=path.name, forbidden=value):
                    self.assertNotIn(value, payload)

    def _assert_publication_failure_cleans_up(self, target: str, failure: Exception):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "failed"
            with patch(target, side_effect=failure):
                with self.assertRaises(type(failure)):
                    generate_corpus(output, self.config)
            self.assertFalse(output.exists())
            self.assertEqual(list(Path(temporary).iterdir()), [])

    def test_sample_write_failure_leaves_no_partial_dataset(self):
        self._assert_publication_failure_cleans_up(
            "prototype.controlled_data.dataset._write_text", OSError("sample write failed")
        )

    def test_manifest_write_failure_leaves_no_partial_dataset(self):
        self._assert_publication_failure_cleans_up(
            "prototype.controlled_data.dataset._write_json", OSError("manifest write failed")
        )

    def test_final_verification_failure_leaves_no_partial_dataset(self):
        self._assert_publication_failure_cleans_up(
            "prototype.controlled_data.dataset._verify_written_tree",
            GenerationError("verification failed"),
        )

    def test_atomic_rename_failure_leaves_no_partial_dataset(self):
        self._assert_publication_failure_cleans_up(
            "prototype.controlled_data.dataset.os.rename", OSError("rename failed")
        )

    def test_existing_output_directory_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "existing"
            output.mkdir()
            marker = output / "marker.txt"
            marker.write_text("preserve")
            with self.assertRaisesRegex(ConfigurationError, "already exists"):
                generate_corpus(output, self.config)
            self.assertEqual(marker.read_text(), "preserve")


class ConfigurationTests(unittest.TestCase):
    def test_invalid_configuration_is_rejected(self):
        cases = (
            GeneratorConfig(num_source_families=True),
            GeneratorConfig(num_source_families=0),
            GeneratorConfig(num_source_families=59),
            GeneratorConfig(num_source_families=60, sketch_extents=(1.0, float("inf"), 2.5)),
            GeneratorConfig(num_source_families=60, iid_ratios=(0.8, 0.3, -0.1)),
        )
        for config in cases:
            with self.subTest(config=config), self.assertRaises(ConfigurationError):
                config.validate()

    def test_default_candidate_space_size_is_exact(self):
        config = GeneratorConfig(60)
        blocks = factor_blocks(config)
        self.assertEqual([block.size for block in blocks[:4]], [270, 180, 270, 180])
        self.assertEqual([block.size for block in blocks[4:]], [16_200, 28_800] * 4)
        self.assertEqual(sum(block.size for block in blocks), 180_900)
        self.assertEqual(total_candidate_count(config), 180_900)

    def test_twelve_decimal_normalization_cannot_merge_accepted_grid_values(self):
        config = GeneratorConfig(60)
        for values in (
            config.sketch_extents,
            config.extrusion_distances,
            config.revolution_angles,
        ):
            normalized = [round(value, PHYSICAL_ID_DECIMAL_PLACES) for value in values]
            self.assertEqual(len(normalized), len(set(normalized)))
            self.assertGreater(min(abs(left - right) for left in values for right in values if left != right), 1e-12)
        with self.assertRaisesRegex(ConfigurationError, "physical-ID normalization"):
            GeneratorConfig(
                60,
                extrusion_distances=(1.0, 1.0 + 1e-13),
            ).validate()

    def test_selection_count_and_order_are_stable(self):
        config = GeneratorConfig(60, seed=9)
        self.assertEqual(select_sources(config), select_sources(config))
        self.assertEqual(len(select_sources(config)), 60)

    def test_affine_permutation_and_anchor_edge_cases(self):
        config = GeneratorConfig(60, seed=3)
        block = factor_blocks(config)[0]
        indices = [_permuted_index(block, index, config) for index in range(block.size)]
        self.assertEqual(set(indices), set(range(block.size)))
        anchors = _coverage_anchor_indices(block, config)
        self.assertEqual(len(anchors), len(set(anchors)))

        class UnitBlock:
            size = 1
            name = "unit"

        class EmptyBlock:
            size = 0
            name = "empty"

        self.assertEqual(_permuted_index(UnitBlock(), 0, config), 0)
        with self.assertRaises(ConfigurationError):
            _permuted_index(UnitBlock(), 1, config)
        with self.assertRaises(ConfigurationError):
            _permuted_index(EmptyBlock(), 0, config)

    def test_mixed_radix_round_trips_first_last_and_random_indices(self):
        generator = random.Random(42)
        for block in factor_blocks(GeneratorConfig(60)):
            indices = [0, block.size - 1]
            indices.extend(generator.randrange(block.size) for _ in range(10))
            for index in indices:
                digits = _mixed_radix_decode(index, block.radices)
                self.assertEqual(_mixed_radix_encode(digits, block.radices), index)
        self.assertEqual(_mixed_radix_decode(0, (1, 1, 1)), (0, 0, 0))
        self.assertEqual(_mixed_radix_encode((0, 0, 0), (1, 1, 1)), 0)
        with self.assertRaises(ConfigurationError):
            _mixed_radix_decode(0, (1, 0))

    def test_coverage_minimum_is_derived_from_unique_block_anchors(self):
        config = GeneratorConfig(60)
        blocks = factor_blocks(config)
        anchors = [_coverage_anchor_indices(block, config) for block in blocks]
        self.assertEqual(required_coverage_family_count(config), sum(map(len, anchors)))
        self.assertEqual(required_coverage_family_count(config), 60)
        self.assertTrue(all(len(items) == len(set(items)) for items in anchors))
        self.assertEqual(len(select_sources(config)), 60)
