"""Focused adversarial tests for the preregistered diagnostic replay."""

from __future__ import annotations

from dataclasses import fields
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from prototype.controlled_data.factors import PrimitiveFamily
from prototype.flat_baseline.artifact_manifest import (
    EVALUATION_ARTIFACTS,
    REPORT_NAME,
    WORKFLOW_EVIDENCE_ARTIFACTS,
)
from prototype.flat_baseline.constraint_manifold_replay import (
    ARMS,
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_EVALUATION_COMMIT,
    OracleCategories,
    PATHS,
    ReplayError,
    _autoregressive_sensitivity,
    _capsule_projection,
    _diagnostic_interpretation,
    _derived_geometry_mask,
    _json_safe,
    _projection_failure_csv,
    _publish_artifacts,
    _rectangle_projection,
    _scheduler_job_id,
    _validate_loaded_contract,
    make_replay_artifacts,
    run_replay,
    transform_prediction,
    validate_source_manifest,
    validate_validator_source,
)
from prototype.model_data.geometry import GEOMETRY_CHANNEL_SCALES
from prototype.flat_baseline.conversion import (
    validate_and_convert_raw_prediction,
)
from prototype.flat_baseline.evaluate_length_conditioned import (
    _raw_from_json,
    _raw_json_safe,
    _stage_two_prediction,
)
from prototype.flat_baseline.tests.test_conversion import _raw_from_target
from prototype.model_data.loader import load_physical_examples
from prototype.model_data.tests.fixtures import source, write_physical_corpus
from prototype.model_data.vocab import (
    NODE_TYPES,
    PRIMITIVE_TYPES,
)


def _namespace_json(value):
    if isinstance(value, SimpleNamespace):
        return {
            name: _namespace_json(item)
            for name, item in vars(value).items()
        }
    if isinstance(value, tuple):
        return [_namespace_json(item) for item in value]
    return value


class ConstraintManifoldReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus = tempfile.TemporaryDirectory()
        write_physical_corpus(
            cls.corpus.name,
            (
                source("E", PrimitiveFamily.RECTANGLE_LINES),
                source("R", PrimitiveFamily.CIRCLE),
                source("EE", PrimitiveFamily.CAPSULE_LINE_ARC),
            ),
        )
        cls.examples = {
            item.metadata.operation_template: item
            for item in load_physical_examples(cls.corpus.name)
        }

    @classmethod
    def tearDownClass(cls):
        cls.corpus.cleanup()

    def namespace_raw(self, template="E"):
        return _raw_from_json(_raw_json_safe(
            _raw_from_target(self.examples[template].target)
        ))

    def test_baseline_exactly_reconciles_and_outputs_are_deterministic(self):
        raw_records, examples, metadata = self.full_fixture()
        first = make_replay_artifacts(
            raw_records, examples, metadata, self.evidence_fixture()
        )
        second = make_replay_artifacts(
            raw_records, examples, metadata, self.evidence_fixture()
        )
        self.assertEqual(first, second)
        records = [
            json.loads(line)
            for line in first["replay_records.jsonl"].splitlines()
        ]
        self.assertEqual(len(records), 68 * len(PATHS) * len(ARMS))
        baseline = [item for item in records if item["arm"] == "baseline"]
        self.assertEqual(len(baseline), 136)
        self.assertTrue(all(not item["transformations"] for item in baseline))
        self.assertTrue(all(
            item["conversion"]["controlled_domain"]["valid"]
            for item in baseline
        ))
        projected = next(
            item for item in records
            if item["arm"] == "model_category_projection"
        )
        self.assertEqual(projected["profile_projection"]["attempt_count"], 1)
        self.assertEqual(projected["profile_projection"]["success_count"], 0)
        self.assertEqual(
            projected["profile_projection"]["on_manifold_no_op_count"], 1
        )
        aggregate = json.loads(first["aggregate_summary.json"])
        self.assertEqual(
            aggregate["arms"]["teacher_forced"]["semantic_constants"][
                "projection_residual_distributions"
            ],
            {},
        )
        run_metadata = json.loads(first["run_metadata.json"])
        self.assertEqual(
            run_metadata["source_bundle"]["slurm_job_id"], "3329040"
        )
        self.assertEqual(
            run_metadata["source_bundle"]["root_hidden_entries"], []
        )
        self.assertEqual(
            run_metadata["access_contract"]["torch_runtime_loaded"],
            "torch" in __import__("sys").modules,
        )
        self.assertEqual(
            run_metadata["validator_source"]["evaluation_commit"],
            EXPECTED_EVALUATION_COMMIT,
        )

        changed = json.loads(json.dumps(examples))
        changed[0]["stored_conversions"]["teacher_forced"][
            "controlled_domain"
        ]["valid"] = False
        with self.assertRaisesRegex(
            ReplayError, "baseline_conversion_mismatch"
        ):
            make_replay_artifacts(
                raw_records, changed, metadata, self.evidence_fixture()
            )

        with tempfile.TemporaryDirectory() as directory:
            first_output = Path(directory) / "first"
            second_output = Path(directory) / "second"
            first_publication = _publish_artifacts(first_output, first)
            second_publication = _publish_artifacts(second_output, second)
            self.assertIn(
                first_publication["backend"],
                ("renameat2", "renamex_np", "symlink"),
            )
            self.assertEqual(
                first_publication["backend"],
                second_publication["backend"],
            )
            self.assertEqual(
                (first_output / "sha256-manifest.txt").read_bytes(),
                (second_output / "sha256-manifest.txt").read_bytes(),
            )
            self.assertEqual(
                sorted(item.name for item in first_output.iterdir()),
                sorted(item.name for item in second_output.iterdir()),
            )

    def test_semantic_constants_change_no_categories_or_unrelated_fields(self):
        raw = self.namespace_raw("R")
        original = _namespace_json(raw)
        plane = next(
            item for item in raw.raw_nodes
            if NODE_TYPES.tokens[item.node_type_id] == "reference_plane"
        )
        plane_geometry = list(plane.normalized_geometry)
        plane_geometry[0:9] = [0.2] * 9
        plane.normalized_geometry = tuple(plane_geometry)
        axis = next(
            item for item in raw.raw_nodes
            if NODE_TYPES.tokens[item.node_type_id] == "axis"
        )
        axis_geometry = list(axis.normalized_geometry)
        axis_geometry[33:37] = (0.1, -0.2, 0.3, 0.4)
        axis.normalized_geometry = tuple(axis_geometry)
        before = _namespace_json(raw)
        transformed, changes, unavailable = transform_prediction(
            raw, "semantic_constants"
        )
        after = _namespace_json(transformed)
        self.assertFalse(unavailable)
        self.assertEqual(
            [item["kind"] for item in changes],
            ["reference_plane", "axis"],
        )
        self.assertEqual(
            [item["categorical_ids"] for item in before["raw_nodes"]],
            [item["categorical_ids"] for item in after["raw_nodes"]],
        )
        self.assertEqual(
            [item["derived_geometry_mask"] for item in before["raw_nodes"]],
            [item["derived_geometry_mask"] for item in after["raw_nodes"]],
        )
        for name in (
            "raw_edges",
            "raw_operation_pointers",
            "node_count",
            "predicted_operation_node_indices",
            "predicted_operation_count",
        ):
            self.assertEqual(before[name], after[name])
        for old_node, new_node in zip(before["raw_nodes"], after["raw_nodes"]):
            node_type = NODE_TYPES.tokens[old_node["node_type_id"]]
            changed_indices = (
                set(range(9)) if node_type == "reference_plane"
                else set(range(33, 37)) if node_type == "axis"
                else set()
            )
            for index, (old, new) in enumerate(zip(
                old_node["normalized_geometry"],
                new_node["normalized_geometry"],
            )):
                if index not in changed_indices:
                    self.assertEqual(old, new)
        self.assertNotEqual(original, before)

    def test_unsupported_model_profile_remains_unsupported(self):
        raw = self.namespace_raw("E")
        sketch = next(
            item for item in raw.raw_nodes
            if NODE_TYPES.tokens[item.node_type_id] == "sketch"
        )
        categories = list(sketch.categorical_ids)
        categories[4:8] = (
            PRIMITIVE_TYPES.id("circle"),
            PRIMITIVE_TYPES.id(None),
            PRIMITIVE_TYPES.id("line"),
            PRIMITIVE_TYPES.id("line"),
        )
        sketch.categorical_ids = tuple(categories)
        sketch.derived_geometry_mask = _derived_geometry_mask(
            "sketch", categories
        )
        original_categories = sketch.categorical_ids
        transformed, changes, unavailable = transform_prediction(
            raw, "model_category_projection"
        )
        transformed_sketch = next(
            item for item in transformed.raw_nodes
            if NODE_TYPES.tokens[item.node_type_id] == "sketch"
        )
        self.assertEqual(
            transformed_sketch.categorical_ids, original_categories
        )
        self.assertEqual(
            unavailable[0]["reason"],
            "unsupported_model_profile_pattern",
        )
        self.assertFalse(any(
            item["kind"].endswith("_profile") for item in changes
        ))
        result = validate_and_convert_raw_prediction(
            transformed, max_operations=2
        )
        self.assertIn(
            "unsupported_profile_pattern",
            result.controlled_domain.failure_codes,
        )

    def test_oracle_categories_cannot_enter_nonoracle_arms(self):
        class ForbiddenOracle:
            @property
            def reference_plane(self):
                raise AssertionError("oracle was inspected")

            @property
            def primitive_family(self):
                raise AssertionError("oracle was inspected")

        raw = self.namespace_raw("E")
        for arm in ARMS[:-1]:
            with self.subTest(arm=arm):
                with self.assertRaisesRegex(
                    ReplayError, "oracle_metadata_leakage"
                ):
                    transform_prediction(
                        raw, arm, oracle=ForbiddenOracle()
                    )
        self.assertEqual(
            [item.name for item in fields(OracleCategories)],
            ["reference_plane", "primitive_family"],
        )
        self.assertNotIn("geometry", OracleCategories.__dataclass_fields__)
        self.assertNotIn(
            "target",
            OracleCategories.__dataclass_fields__,
        )

    def test_projection_constructs_exact_frames_axes_and_closed_profiles(self):
        cases = (
            ("E", "rectangle_profile"),
            ("R", "circle_profile"),
            ("EE", "capsule_profile"),
        )
        for template, expected_kind in cases:
            with self.subTest(template=template):
                raw = self.namespace_raw(template)
                for node in raw.raw_nodes:
                    token = NODE_TYPES.tokens[node.node_type_id]
                    geometry = list(node.normalized_geometry)
                    if token == "reference_plane":
                        geometry[0:9] = [0.13] * 9
                    elif token == "axis":
                        geometry[33:37] = (0.1, -0.1, 0.2, 0.8)
                    elif token == "sketch" and expected_kind != "circle_profile":
                        for index, applicable in enumerate(
                            node.derived_geometry_mask
                        ):
                            if applicable:
                                geometry[index] += (
                                    ((index % 5) - 2) * 0.005
                                )
                    node.normalized_geometry = tuple(geometry)
                transformed, changes, unavailable = transform_prediction(
                    raw, "model_category_projection"
                )
                self.assertFalse(unavailable)
                self.assertIn(
                    expected_kind,
                    [item["kind"] for item in changes],
                )
                result = validate_and_convert_raw_prediction(
                    transformed, max_operations=2
                )
                self.assertTrue(
                    result.controlled_domain.valid,
                    result.controlled_domain.failure_codes,
                )
                plane = next(
                    item for item in transformed.raw_nodes
                    if NODE_TYPES.tokens[item.node_type_id]
                    == "reference_plane"
                )
                plane_id = plane.categorical_ids[3]
                self.assertIn(
                    plane_id,
                    tuple(range(2, 5)),
                )
                if template == "R":
                    axis = next(
                        item for item in transformed.raw_nodes
                        if NODE_TYPES.tokens[item.node_type_id] == "axis"
                    )
                    self.assertEqual(
                        axis.normalized_geometry[33:37],
                        (0.0, 0.0, 0.0, 1.0),
                    )

    def test_oracle_arm_uses_only_categories_and_can_repair_profile_choice(self):
        raw = self.namespace_raw("E")
        sketch = next(
            item for item in raw.raw_nodes
            if NODE_TYPES.tokens[item.node_type_id] == "sketch"
        )
        categories = list(sketch.categorical_ids)
        categories[4:8] = (
            PRIMITIVE_TYPES.id("circle"),
            PRIMITIVE_TYPES.id(None),
            PRIMITIVE_TYPES.id("line"),
            PRIMITIVE_TYPES.id("line"),
        )
        sketch.categorical_ids = tuple(categories)
        sketch.derived_geometry_mask = _derived_geometry_mask(
            "sketch", categories
        )
        transformed, changes, unavailable = transform_prediction(
            raw,
            "oracle_category_diagnostic",
            oracle=OracleCategories("XY", "rectangle_lines"),
        )
        self.assertFalse(unavailable)
        self.assertTrue(any(
            item["kind"] == "oracle_profile_category" for item in changes
        ))
        self.assertTrue(any(
            item["kind"] == "oracle_geometry_mask" for item in changes
        ))
        self.assertTrue(
            validate_and_convert_raw_prediction(
                transformed, max_operations=2
            ).controlled_domain.valid
        )

    def test_malformed_bundle_fails_before_scientific_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "bundle"
            self.write_bundle_fixture(root)
            missing = root / "evaluation" / sorted(EVALUATION_ARTIFACTS)[0]
            missing.unlink()
            with mock.patch(
                "prototype.flat_baseline.constraint_manifold_replay._load_bundle"
            ) as loader:
                with self.assertRaises(ReplayError):
                    run_replay(root, Path(directory) / "output")
                loader.assert_not_called()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "bundle"
            self.write_bundle_fixture(root)
            artifact = root / "workflow-evidence" / sorted(
                WORKFLOW_EVIDENCE_ARTIFACTS
            )[0]
            artifact.write_bytes(b"modified")
            with self.assertRaisesRegex(
                ReplayError, "source_hash_mismatch"
            ):
                validate_source_manifest(root)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "bundle"
            self.write_bundle_fixture(root)
            (root / "evaluation" / "extra.txt").write_text("extra")
            with self.assertRaisesRegex(
                ReplayError, "source_artifact_set_mismatch"
            ):
                validate_source_manifest(root)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "bundle"
            self.write_bundle_fixture(root)
            (root / ".download-audit").write_text("recorded but not hashed")
            bundle_evidence = validate_source_manifest(root)
            self.assertEqual(
                bundle_evidence.hidden_entries, (".download-audit",)
            )

    def test_collision_safe_output_never_replaces_existing_directory(self):
        raw_records, examples, metadata = self.full_fixture()
        artifacts = make_replay_artifacts(
            raw_records, examples, metadata, self.evidence_fixture()
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "diagnostic"
            output.mkdir()
            sentinel = output / "sentinel"
            sentinel.write_text("preserve")
            with self.assertRaisesRegex(ReplayError, "output_collision"):
                _publish_artifacts(output, artifacts)
            self.assertEqual(sentinel.read_text(), "preserve")

    def test_autoregressive_and_preregistered_interpretation_boundaries(self):
        def arm(valid, rescued=None):
            return {
                "controlled_domain_valid_count": valid,
                "invalid_to_valid_count": valid if rescued is None else rescued,
            }

        for teacher, predicted, expected_ratio, expected_label in (
            (0, 0, None, "not_material"),
            (10, 7, 0.7, "material"),
            (10, 8, 0.8, "not_material"),
        ):
            by_path = {
                "teacher_forced": {"projected": arm(teacher)},
                "predicted_history": {"projected": arm(predicted)},
            }
            result = _autoregressive_sensitivity(by_path, "projected")
            self.assertEqual(result["survival_ratio"], expected_ratio)
            self.assertEqual(result["label"], expected_label)

        by_path_arm = {
            path: {
                name: arm(0, 0)
                for name in ARMS
            }
            for path in PATHS
        }
        # Valid counts deliberately disagree with rescue counts.
        by_path_arm["teacher_forced"]["model_category_projection"] = arm(40, 2)
        by_path_arm["predicted_history"]["model_category_projection"] = arm(35, 3)
        by_path_arm["teacher_forced"][ARMS[3]] = arm(60, 20)
        by_path_arm["predicted_history"][ARMS[3]] = arm(30, 1)
        interpretation = _diagnostic_interpretation(by_path_arm)
        self.assertEqual(
            interpretation["constraint_parameterization_sensitivity"],
            "limited_or_path_specific",
        )
        self.assertEqual(
            interpretation["categorical_oracle_sensitivity"]["combined_label"],
            "strong",
        )
        self.assertEqual(
            interpretation["per_path"]["predicted_history"][
                "oracle_category_sensitivity"
            ],
            "regression",
        )

    def test_bundle_identity_scheduler_and_validator_are_verified(self):
        raw_records, examples, metadata = self.full_fixture()
        self.assertEqual(
            _validate_loaded_contract(
                raw_records,
                examples,
                metadata,
                scheduler_job_id="3329040",
            ),
            2,
        )
        mutations = (
            ("checkpoint_sha256", "0" * 64),
            ("repository_commit", "0" * 40),
            ("repaired_full_contract", False),
            ("evaluation_schema_version", 1),
        )
        for field, value in mutations:
            with self.subTest(field=field):
                changed = json.loads(json.dumps(metadata))
                changed[field] = value
                with self.assertRaisesRegex(
                    ReplayError, "source_bundle_identity"
                ):
                    _validate_loaded_contract(
                        raw_records,
                        examples,
                        changed,
                        scheduler_job_id="3329040",
                    )
        with self.assertRaisesRegex(ReplayError, "source_bundle_identity"):
            _validate_loaded_contract(
                raw_records,
                examples,
                metadata,
                scheduler_job_id="3329041",
            )
        with tempfile.TemporaryDirectory() as directory:
            scheduler = Path(directory) / "scheduler.txt"
            scheduler.write_text("slurm_job_id=3329040\nstate=FAILED\n")
            self.assertEqual(_scheduler_job_id(scheduler), "3329040")
            scheduler.write_text("slurm_job_id=3329040\nslurm_job_id=2\n")
            with self.assertRaisesRegex(
                ReplayError, "invalid_scheduler_evidence"
            ):
                _scheduler_job_id(scheduler)
        validator = validate_validator_source()
        self.assertEqual(
            validator.evaluation_commit, EXPECTED_EVALUATION_COMMIT
        )
        self.assertEqual(len(validator.per_file_sha256), 2)
        with mock.patch(
            "prototype.flat_baseline.constraint_manifold_replay._git_bytes",
            return_value=b"changed-validator",
        ):
            with self.assertRaisesRegex(
                ReplayError, "validator_source_mismatch"
            ):
                validate_validator_source()

    def test_real_bundle_two_family_golden_reconciliation(self):
        bundle = Path(
            "/Users/krishaymaskara/research/audited-runs/"
            "phase-b-full-validation/"
            "phase-b-repaired-validation-"
            "ce4abca8f450745a8832c0189aaae4a7d8263ec0-3329040"
        )
        if not bundle.is_dir():
            self.skipTest("immutable recovered bundle is not available locally")
        evidence = validate_source_manifest(bundle)
        self.assertEqual(
            evidence.manifest_sha256,
            "a13d5f35b345b6814912145a56e0cce0"
            "290ae677b030a1215b61e92558862541",
        )
        family_hashes = {
            "sf_0315dbbd0c69ebdc80097404b37b0e0484c6f4d14a79e1e1cf5aa4b788b4beb3": {
                "teacher_forced": "ba050866ac7b9d8c00654e2423dcf1f3566cc9f06b57770ba646df8082b6546f",
                "predicted_history": "bd20703f231f1555371acbf3179eab255b35f3c2d104261bc38c55384b8d2aa6",
            },
            "sf_0ee68f7559ece4d1c4d5bbc7e835964f03c9e79923098af599bc981b0389ef18": {
                "teacher_forced": "dbff238cbcc0057f19d38b1fdd856814fdb3c57a57e76a6ae04ddbaa8c9beffd",
                "predicted_history": "d67232aa784fcca0ab5d9bc18e75ecbf35978f8ce01ac6c9ac0ddd212395d713",
            },
        }
        raw = {
            item["family_id"]: item
            for item in map(
                json.loads,
                (bundle / "evaluation/raw_predictions.jsonl")
                .read_text()
                .splitlines(),
            )
        }
        examples = {
            item["family_id"]: item
            for item in map(
                json.loads,
                (bundle / "evaluation/examples.jsonl")
                .read_text()
                .splitlines(),
            )
        }
        for family_id, path_hashes in family_hashes.items():
            for path, expected_hash in path_hashes.items():
                stored = examples[family_id][path]["conversion"]
                stored_bytes = json.dumps(
                    stored, sort_keys=True, separators=(",", ":")
                ).encode()
                self.assertEqual(
                    hashlib.sha256(stored_bytes).hexdigest(), expected_hash
                )
                prediction = _stage_two_prediction(
                    _raw_from_json(raw[family_id][path]), path
                )
                actual = _json_safe(validate_and_convert_raw_prediction(
                    prediction, max_operations=2
                ))
                self.assertEqual(actual, stored)
                self.assertFalse(
                    actual["controlled_domain"]["valid"],
                    (family_id, path),
                )

    def test_projection_values_match_independent_closed_forms(self):
        cases = (
            (
                _rectangle_projection,
                (
                    ((9, 10), (-0.5, -0.25)),
                    ((11, 12), (0.5, -0.25)),
                    ((15, 16), (0.5, -0.25)),
                    ((17, 18), (0.5, 0.25)),
                    ((21, 22), (0.5, 0.25)),
                    ((23, 24), (-0.5, 0.25)),
                    ((27, 28), (-0.5, 0.25)),
                    ((29, 30), (-0.5, -0.25)),
                ),
                2.5,
            ),
            (
                _capsule_projection,
                (
                    ((9, 10), (0.25, -0.25)),
                    ((11, 12), (0.5, 0.0)),
                    ((13, 14), (0.25, 0.25)),
                    ((15, 16), (-0.25, 0.25)),
                    ((17, 18), (-0.5, 0.0)),
                    ((19, 20), (-0.25, -0.25)),
                    ((21, 22), (-0.25, -0.25)),
                    ((23, 24), (0.25, -0.25)),
                    ((27, 28), (0.25, 0.25)),
                    ((29, 30), (-0.25, 0.25)),
                ),
                1.5,
            ),
        )
        for function, points, denominator in cases:
            geometry = [0.0] * 39
            observations = []
            for ordinal, ((x_index, y_index), coefficient) in enumerate(points):
                physical_x = (
                    0.4 + coefficient[0] * 0.8 + 0.003 * ordinal
                )
                physical_y = (
                    -0.2 + coefficient[1] * 0.8
                    + 0.002 * (ordinal % 3)
                )
                geometry[x_index] = (
                    physical_x / GEOMETRY_CHANNEL_SCALES[x_index]
                )
                geometry[y_index] = (
                    physical_y / GEOMETRY_CHANNEL_SCALES[y_index]
                )
                observations.append((physical_x, physical_y))
            cx = sum(item[0] for item in observations) / len(observations)
            cy = sum(item[1] for item in observations) / len(observations)
            extent = sum(
                coefficient[0] * observed[0]
                + coefficient[1] * observed[1]
                for (_, coefficient), observed in zip(points, observations)
            ) / denominator
            channels, projected, _ = function(geometry)
            for index, value in zip(channels, projected):
                physical = value * GEOMETRY_CHANNEL_SCALES[index]
                occurrences = [
                    (coefficient, observed_indices)
                    for observed_indices, coefficient in points
                    if index in observed_indices
                ]
                coefficient, observed_indices = occurrences[0]
                expected = (
                    cx + coefficient[0] * extent
                    if index == observed_indices[0]
                    else cy + coefficient[1] * extent
                )
                self.assertAlmostEqual(physical, expected, places=12)

    def test_projection_failure_reasons_and_oracle_mask_discipline(self):
        raw = self.namespace_raw("E")
        sketch = next(
            node for node in raw.raw_nodes
            if NODE_TYPES.tokens[node.node_type_id] == "sketch"
        )
        mask = list(sketch.derived_geometry_mask)
        mask[37] = True
        sketch.derived_geometry_mask = tuple(mask)
        transformed, changes, unavailable = transform_prediction(
            raw,
            "oracle_category_diagnostic",
            oracle=OracleCategories("XY", "rectangle_lines"),
        )
        transformed_sketch = next(
            node for node in transformed.raw_nodes
            if NODE_TYPES.tokens[node.node_type_id] == "sketch"
        )
        self.assertTrue(transformed_sketch.derived_geometry_mask[37])
        self.assertTrue(all("is_no_op" in item for item in changes))
        self.assertFalse(unavailable)

        geometry = [0.0] * 39
        self.assertEqual(
            _rectangle_projection(geometry), "nonpositive_fitted_extent"
        )
        geometry[9] = float("nan")
        self.assertEqual(
            _rectangle_projection(geometry),
            "nonfinite_or_nonnumeric_observation",
        )
        self.assertEqual(
            _rectangle_projection([0.0] * 10), "malformed_geometry_width"
        )
        geometry = [0.0] * 39
        rectangle_points = (
            ((9, 10), (-0.5, -0.25)),
            ((11, 12), (0.5, -0.25)),
            ((15, 16), (0.5, -0.25)),
            ((17, 18), (0.5, 0.25)),
            ((21, 22), (0.5, 0.25)),
            ((23, 24), (-0.5, 0.25)),
            ((27, 28), (-0.5, 0.25)),
            ((29, 30), (-0.5, -0.25)),
        )
        for (x_index, y_index), coefficient in rectangle_points:
            geometry[x_index] = coefficient[0] * 9.0 / 4.0
            geometry[y_index] = coefficient[1] * 9.0 / 4.0
        self.assertEqual(
            _rectangle_projection(geometry), "projected_geometry_out_of_range"
        )

        circle = self.namespace_raw("R")
        circle_sketch = next(
            node for node in circle.raw_nodes
            if NODE_TYPES.tokens[node.node_type_id] == "sketch"
        )
        circle_geometry = list(circle_sketch.normalized_geometry)
        circle_geometry[11] = 0.0
        circle_sketch.normalized_geometry = tuple(circle_geometry)
        _, _, failures = transform_prediction(
            circle, "model_category_projection"
        )
        self.assertEqual(
            failures[0]["reason"],
            "nonpositive_radius_has_no_nearest_strict_projection",
        )
        rows = [
            dict(
                failures[0],
                family_id="sf_" + "0" * 64,
                path="teacher_forced",
                arm="model_category_projection",
            )
        ]
        csv_text = _projection_failure_csv(rows).decode()
        self.assertIn(
            "nonpositive_radius_has_no_nearest_strict_projection", csv_text
        )

        invalid_profile = self.namespace_raw("E")
        invalid_sketch = next(
            node for node in invalid_profile.raw_nodes
            if NODE_TYPES.tokens[node.node_type_id] == "sketch"
        )
        categories = list(invalid_sketch.categorical_ids)
        categories[4] = 999
        invalid_sketch.categorical_ids = tuple(categories)
        _, _, profile_failures = transform_prediction(
            invalid_profile, "model_category_projection"
        )
        self.assertEqual(
            profile_failures[0]["reason"],
            "invalid_decoded_profile_category",
        )

        invalid_plane = self.namespace_raw("E")
        plane = next(
            node for node in invalid_plane.raw_nodes
            if NODE_TYPES.tokens[node.node_type_id] == "reference_plane"
        )
        categories = list(plane.categorical_ids)
        categories[3] = 999
        plane.categorical_ids = tuple(categories)
        _, _, plane_failures = transform_prediction(
            invalid_plane, "semantic_constants"
        )
        self.assertEqual(
            plane_failures[0]["reason"],
            "invalid_decoded_plane_category",
        )

        actual_rectangle_failures = []
        rectangle_geometries = (
            ([0.0] * 39, "nonpositive_fitted_extent"),
            (
                [float("nan")] + [0.0] * 38,
                "nonfinite_or_nonnumeric_observation",
            ),
            ([0.0] * 10, "malformed_geometry_width"),
            (geometry, "projected_geometry_out_of_range"),
        )
        for candidate, expected_reason in rectangle_geometries:
            candidate_raw = self.namespace_raw("E")
            candidate_sketch = next(
                node for node in candidate_raw.raw_nodes
                if NODE_TYPES.tokens[node.node_type_id] == "sketch"
            )
            if expected_reason == "nonfinite_or_nonnumeric_observation":
                candidate[0], candidate[9] = candidate[9], candidate[0]
            candidate_sketch.normalized_geometry = tuple(candidate)
            _, _, candidate_failures = transform_prediction(
                candidate_raw, "model_category_projection"
            )
            self.assertEqual(
                candidate_failures[0]["reason"], expected_reason
            )
            actual_rectangle_failures.append(candidate_failures[0])
        all_rows = [
            dict(
                failure,
                family_id="sf_" + "{:064x}".format(index),
                path="teacher_forced",
                arm="model_category_projection",
            )
            for index, failure in enumerate(
                actual_rectangle_failures
                + failures + profile_failures + plane_failures
            )
        ]
        failure_csv = _projection_failure_csv(all_rows).decode()
        self.assertEqual(len(failure_csv.splitlines()), len(all_rows) + 1)
        for failure in all_rows:
            self.assertIn(failure["reason"], failure_csv)

    def test_malformed_and_nonfinite_geometry_have_registered_context(self):
        raw_records, examples, metadata = self.full_fixture()
        geometry = raw_records[0]["teacher_forced"]["raw_nodes"][0][
            "normalized_geometry"
        ]
        raw_records[0]["teacher_forced"]["raw_nodes"][0][
            "normalized_geometry"
        ] = geometry[:3]
        # Baseline conversion is checked first; transformation context is tested
        # directly because malformed records cannot reconcile to the frozen bundle.
        malformed = _raw_from_json(raw_records[0]["teacher_forced"])
        with self.assertRaisesRegex(
            ReplayError, "invalid_raw_node_width"
        ):
            transform_prediction(malformed, "semantic_constants")
        finite = self.namespace_raw("E")
        plane = next(
            node for node in finite.raw_nodes
            if NODE_TYPES.tokens[node.node_type_id] == "reference_plane"
        )
        values = list(plane.normalized_geometry)
        values[0] = float("nan")
        plane.normalized_geometry = tuple(values)
        with self.assertRaisesRegex(
            ReplayError, "nonfinite_raw_geometry"
        ):
            transform_prediction(finite, "semantic_constants")

    def test_post_publication_sync_failure_has_distinct_state(self):
        raw_records, examples, metadata = self.full_fixture()
        artifacts = make_replay_artifacts(
            raw_records, examples, metadata, self.evidence_fixture()
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "diagnostic"
            def publish(source, destination):
                source.rename(destination)
                return {"publication_backend": "test_atomic_rename"}

            with mock.patch(
                "prototype.flat_baseline.constraint_manifold_replay."
                "_atomic_no_replace",
                side_effect=publish,
            ), mock.patch(
                "prototype.flat_baseline.constraint_manifold_replay."
                "_fsync_directory",
                side_effect=(None, OSError("parent fsync failed")),
            ):
                with self.assertRaisesRegex(
                    ReplayError, "output_published_but_parent_unsynced"
                ):
                    _publish_artifacts(output, artifacts)
            self.assertTrue(output.exists())
            self.assertTrue(
                (output / "sha256-manifest.txt").is_file()
            )

    def full_fixture(self):
        base = _raw_from_target(self.examples["E"].target)
        raw_json = _raw_json_safe(base)
        teacher_json = json.loads(json.dumps(raw_json))
        teacher_json["prefix_feedback"] = "shifted_target_prefix"
        predicted_json = json.loads(json.dumps(raw_json))
        teacher_conversion = _json_safe(validate_and_convert_raw_prediction(
            _stage_two_prediction(
                _raw_from_json(teacher_json), "teacher_forced"
            ),
            max_operations=2,
        ))
        predicted_conversion = _json_safe(validate_and_convert_raw_prediction(
            _raw_from_json(predicted_json),
            max_operations=2,
        ))
        family_ids = [
            "sf_{:064x}".format(index) for index in range(68)
        ]
        raw_records = [
            {
                "family_id": family_id,
                "teacher_forced": teacher_json,
                "predicted_history": predicted_json,
            }
            for family_id in family_ids
        ]
        examples = [
            {
                "family_id": family_id,
                "operation_template": "E",
                "primitive_family": "rectangle_lines",
                "reference_plane": "XY",
                "stored_conversions": {
                    "teacher_forced": teacher_conversion,
                    "predicted_history": predicted_conversion,
                },
            }
            for family_id in family_ids
        ]
        metadata = {
            "authoritative_validation_family_ids": family_ids,
            "checkpoint_epoch": 44,
            "checkpoint_global_step": 748,
            "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
            "evaluation_schema_version": 2,
            "model_configuration": {"max_operations": 2},
            "partition": "validation",
            "payload_access": {
                "train_family_records_loaded": 0,
                "validation_family_records_loaded": 68,
                "test_family_records_loaded": 0,
            },
            "repaired_full_contract": True,
            "repository_commit": EXPECTED_EVALUATION_COMMIT,
            "selected_family_count": 68,
            "selected_family_ids": family_ids,
            "test_partition_evaluated": False,
        }
        return raw_records, examples, metadata

    def evidence_fixture(self):
        from prototype.flat_baseline.constraint_manifold_replay import (
            BundleEvidence,
        )
        return BundleEvidence(
            "/fixture/bundle",
            "a" * 64,
            {"evaluation/raw_predictions.jsonl": "b" * 64},
            {"evaluation/raw_predictions.jsonl": (1, 2, 3, 4)},
            (),
        )

    def write_bundle_fixture(self, root):
        root.mkdir()
        evaluation = root / "evaluation"
        evidence = root / "workflow-evidence"
        evaluation.mkdir()
        evidence.mkdir()
        paths = []
        for name in EVALUATION_ARTIFACTS:
            path = evaluation / name
            path.write_bytes(("evaluation:" + name).encode("utf-8"))
            paths.append(("evaluation/" + name, path))
        for name in WORKFLOW_EVIDENCE_ARTIFACTS:
            path = evidence / name
            path.write_bytes(("evidence:" + name).encode("utf-8"))
            paths.append(("workflow-evidence/" + name, path))
        report = root / REPORT_NAME
        report.write_bytes(b"report")
        paths.append((REPORT_NAME, report))
        manifest = root / "sha256-manifest.txt"
        manifest.write_text("".join(
            "{}  /archived/publication/{}\n".format(
                hashlib.sha256(path.read_bytes()).hexdigest(), relative
            )
            for relative, path in sorted(paths)
        ))


if __name__ == "__main__":
    unittest.main()
