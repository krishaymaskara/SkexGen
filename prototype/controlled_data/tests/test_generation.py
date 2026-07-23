from __future__ import annotations

from dataclasses import replace
from collections import Counter
import hashlib
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
    FEASIBILITY_POLICY_VERSION,
    PHYSICAL_ID_DECIMAL_PLACES,
    ConfigurationError,
    GeneratorConfig,
    required_coverage_family_count,
)
from prototype.controlled_data.dataset import GenerationError, generate_corpus
from prototype.controlled_data.factors import (
    _coverage_anchor_indices,
    _coverage_tokens,
    _mixed_radix_decode,
    _mixed_radix_encode,
    _permuted_index,
    factor_blocks,
    feasibility_status_counts,
    feasible_block_counts,
    select_sources,
    total_candidate_count,
    total_raw_candidate_count,
)
from prototype.controlled_data.factors import (
    ExtentBand,
    OperationTemplate,
    PhysicalSource,
    PrimitiveFamily,
    ReferencePlane,
)
from prototype.controlled_data.feasibility import (
    direction_relation,
    evaluate_feasibility,
    extent_order_relation,
)
from prototype.representation.model import BooleanMode, Direction
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
        cls.config = GeneratorConfig(num_source_families=68, seed=17)
        generate_corpus(cls.first, cls.config)
        generate_corpus(cls.second, cls.config)
        cls.corpus = json.loads((cls.first / "corpus_manifest.json").read_text())

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_requested_family_count_produces_exactly_two_variants_each(self):
        self.assertEqual(self.corpus["total_source_family_count"], 68)
        self.assertEqual(self.corpus["total_sample_variant_count"], 136)
        self.assertEqual(len(self.corpus["families"]), 68)
        self.assertEqual(len(self.corpus["samples"]), 136)
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
            "feasibility_policy_version",
            "representation_schema_version",
            "canonicalization_version",
            "normalized_configuration",
            "configuration_sha256",
            "generation_seed",
            "total_source_family_count",
            "total_sample_variant_count",
            "candidate_source_family_count",
            "raw_candidate_source_family_count",
        }
        self.assertTrue(expected.issubset(self.corpus))
        self.assertEqual(
            self.corpus["feasibility_policy_version"], FEASIBILITY_POLICY_VERSION
        )
        self.assertEqual(self.corpus["candidate_source_family_count"], 120_060)
        self.assertEqual(self.corpus["raw_candidate_source_family_count"], 180_900)
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
        block_seed_one = factor_blocks(GeneratorConfig(68, seed=1))[0]
        block_seed_two = factor_blocks(GeneratorConfig(68, seed=999))[0]
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
            GeneratorConfig(num_source_families=67),
            GeneratorConfig(num_source_families=68, sketch_extents=(1.0, float("inf"), 2.5)),
            GeneratorConfig(num_source_families=68, iid_ratios=(0.8, 0.3, -0.1)),
        )
        for config in cases:
            with self.subTest(config=config), self.assertRaises(ConfigurationError):
                config.validate()

    def test_default_candidate_space_size_is_exact(self):
        config = GeneratorConfig(68)
        blocks = factor_blocks(config)
        self.assertEqual([block.size for block in blocks[:4]], [270, 180, 270, 180])
        self.assertEqual([block.size for block in blocks[4:]], [16_200, 28_800] * 4)
        self.assertEqual(sum(block.size for block in blocks), 180_900)
        self.assertEqual(total_raw_candidate_count(config), 180_900)
        self.assertEqual(total_candidate_count(config), 120_060)

    def test_twelve_decimal_normalization_cannot_merge_accepted_grid_values(self):
        config = GeneratorConfig(68)
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
                68,
                extrusion_distances=(1.0, 1.0 + 1e-13),
            ).validate()

    def test_selection_count_and_order_are_stable(self):
        config = GeneratorConfig(68, seed=9)
        self.assertEqual(select_sources(config), select_sources(config))
        self.assertEqual(len(select_sources(config)), 68)

    def test_affine_permutation_and_anchor_edge_cases(self):
        config = GeneratorConfig(68, seed=3)
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
        for block in factor_blocks(GeneratorConfig(68)):
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
        config = GeneratorConfig(68)
        blocks = factor_blocks(config)
        anchors = [_coverage_anchor_indices(block, config) for block in blocks]
        self.assertEqual(required_coverage_family_count(config), sum(map(len, anchors)))
        self.assertEqual(required_coverage_family_count(config), 68)
        self.assertTrue(all(len(items) == len(set(items)) for items in anchors))
        self.assertEqual(len(select_sources(config)), 68)

    def test_exact_feasible_counts_are_derived_from_the_grid(self):
        config = GeneratorConfig(68)
        self.assertEqual(
            feasible_block_counts(config),
            {
                "E:in_range": 270,
                "E:extrapolation": 180,
                "R:in_range": 270,
                "R:extrapolation": 180,
                "EE:in_range": 8_910,
                "EE:extrapolation": 16_740,
                "ER:in_range": 12_096,
                "ER:extrapolation": 20_484,
                "RE:in_range": 12_096,
                "RE:extrapolation": 20_484,
                "RR:in_range": 9_774,
                "RR:extrapolation": 18_576,
            },
        )
        self.assertEqual(
            feasibility_status_counts(config),
            {
                "accepted": 120_060,
                "cut_complete_subtraction": 12_408,
                "cut_no_positive_volume_overlap": 35_100,
                "join_duplicate_geometry": 990,
                "join_tool_contained": 11_418,
                "mixed_containment_not_certified": 924,
            },
        )

    def test_feasible_grid_preserves_mode_extent_and_direction_diversity(self):
        modes = Counter()
        relations = Counter()
        directions = Counter()
        for block in factor_blocks(GeneratorConfig(68)):
            for index in range(block.size):
                item = block.decode(index)
                if not evaluate_feasibility(item).accepted:
                    continue
                mode = (
                    BooleanMode.NEW_BODY.value
                    if item.later_boolean_mode is None
                    else item.later_boolean_mode.value
                )
                modes[mode] += 1
                relations[(mode, extent_order_relation(item))] += 1
                directions[(mode, direction_relation(item))] += 1
        self.assertEqual(modes, {"join": 77_130, "cut": 42_030, "new_body": 900})
        self.assertEqual(
            relations,
            {
                ("join", "smaller"): 26_280,
                ("join", "equal"): 14_850,
                ("join", "larger"): 36_000,
                ("cut", "smaller"): 21_960,
                ("cut", "equal"): 7_830,
                ("cut", "larger"): 12_240,
                ("new_body", "single"): 900,
            },
        )
        self.assertEqual(
            directions,
            {
                ("join", "same"): 36_252,
                ("join", "opposite"): 40_878,
                ("cut", "same"): 18_252,
                ("cut", "opposite"): 23_778,
                ("new_body", "single"): 900,
            },
        )

    def test_coverage_anchors_are_accepted_unique_and_relationally_complete(self):
        config = GeneratorConfig(68)
        expected = {
            (mode, relation)
            for mode in ("join", "cut")
            for relation in ("smaller", "equal", "larger")
        }
        for block in factor_blocks(config):
            indices = _coverage_anchor_indices(block, config)
            self.assertEqual(len(indices), len(set(indices)))
            selected = [block.decode(index) for index in indices]
            expected_tokens = set()
            for index in range(block.size):
                item = block.decode(index)
                if evaluate_feasibility(item).accepted:
                    expected_tokens.update(_coverage_tokens(item))
            observed_tokens = set()
            for item in selected:
                observed_tokens.update(_coverage_tokens(item))
            self.assertEqual(observed_tokens, expected_tokens)
            if len(block.template.operations) == 2:
                self.assertEqual(
                    {
                        (item.later_boolean_mode.value, extent_order_relation(item))
                        for item in selected
                    },
                    expected,
                )

    def test_deterministic_rejection_never_selects_an_infeasible_candidate(self):
        first = select_sources(GeneratorConfig(100, seed=31))
        second = select_sources(GeneratorConfig(100, seed=31))
        self.assertEqual(first, second)
        self.assertTrue(all(evaluate_feasibility(item).accepted for item in first))

    def test_accepted_preexisting_history_keeps_identity_and_canonical_json(self):
        item = PhysicalSource(
            OperationTemplate.EE,
            PrimitiveFamily.RECTANGLE_LINES,
            ReferencePlane.XY,
            (1.0, 1.5),
            (Direction.POSITIVE, Direction.NEGATIVE),
            BooleanMode.JOIN,
            (1.0, 1.0),
            ExtentBand.IN_RANGE,
        )
        expected = {
            GeometryEncoding.CONTINUOUS: (
                "sf_d9e863f8c8686eaccf92b0d68db9790339c8a27f3b9c5c263cee5b71e1ee0f4e",
                "sv_66ea908d271ae821ff9b2582f82f0b53d53ee256aa79592f3c02d1b8c3735d90",
                "ed683bd50b71da9157a92514d4ba421818b28f09fe3cfc47aac67e43084257f4",
            ),
            GeometryEncoding.QUANTIZED: (
                "sf_d9e863f8c8686eaccf92b0d68db9790339c8a27f3b9c5c263cee5b71e1ee0f4e",
                "sv_3c3b3a889ffbf03fbafe26117eaeeae039aa8c009282c078261404331a83636c",
                "fac7f7358381639407c199724fd9600f421e4e991de711b6265261e8fec91e31",
            ),
        }
        for encoding, golden in expected.items():
            history = build_history(item, encoding)
            payload = history_to_json(history)
            self.assertEqual(source_family_id(history), golden[0])
            self.assertEqual(sample_id(history), golden[1])
            self.assertEqual(hashlib.sha256(payload.encode("utf-8")).hexdigest(), golden[2])
