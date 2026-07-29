"""Adversarial tests for the preregistered categorical-isolation replay."""

from __future__ import annotations

from dataclasses import fields
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import prototype.flat_baseline.categorical_isolation_replay as isolation
from prototype.controlled_data.factors import PrimitiveFamily
from prototype.flat_baseline.constraint_manifold_replay import (
    ReplayError,
    _json_safe,
    validate_source_manifest,
)
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
from prototype.model_data.vocab import NODE_TYPES


SOURCE_BUNDLE = Path(
    "/Users/krishaymaskara/research/audited-runs/"
    "phase-b-full-validation/"
    "phase-b-repaired-validation-"
    "ce4abca8f450745a8832c0189aaae4a7d8263ec0-3329040"
)
PARENT_REPLAY = Path(
    "/Users/krishaymaskara/research/audited-runs/"
    "phase-b-full-validation/"
    "constraint-manifold-replay-3329040-preregistered-v1"
)


def _plain(value):
    if hasattr(value, "__dict__"):
        return {
            name: _plain(item) for name, item in vars(value).items()
        }
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    return value


class CategoricalIsolationReplayTests(unittest.TestCase):
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

    def raw(self, template="EE"):
        return _raw_from_json(_raw_json_safe(
            _raw_from_target(self.examples[template].target)
        ))

    def test_plane_only_cannot_access_or_modify_profile_categories(self):
        raw = self.raw()
        before = _plain(raw)
        transformed, _, _ = isolation.transform_categorical_isolation(
            raw,
            isolation.PLANE_ONLY_ARM,
            plane_oracle=isolation.PlaneOracle("YZ"),
        )
        after = _plain(transformed)
        for old, new in zip(before["raw_nodes"], after["raw_nodes"]):
            if NODE_TYPES.tokens[old["node_type_id"]] == "sketch":
                self.assertEqual(
                    old["categorical_ids"][4:8],
                    new["categorical_ids"][4:8],
                )
                self.assertEqual(
                    old["derived_geometry_mask"][9:33],
                    new["derived_geometry_mask"][9:33],
                )
        self.assertEqual(
            [field.name for field in fields(isolation.PlaneOracle)],
            ["reference_plane"],
        )
        with self.assertRaisesRegex(
            ReplayError, "profile_oracle_capability"
        ):
            isolation.transform_categorical_isolation(
                raw,
                isolation.PLANE_ONLY_ARM,
                plane_oracle=isolation.PlaneOracle("YZ"),
                profile_oracle=isolation.ProfileOracle(
                    "capsule_line_arc"
                ),
            )

    def test_plane_only_rejects_non_noop_plane_mask_rederivation(self):
        raw = self.raw()
        plane = next(
            node for node in raw.raw_nodes
            if NODE_TYPES.tokens[node.node_type_id] == "reference_plane"
        )
        mask = list(plane.derived_geometry_mask)
        mask[0] = False
        plane.derived_geometry_mask = tuple(mask)
        with self.assertRaisesRegex(
            ReplayError, "plane_mask_rederivation_not_no_op"
        ):
            isolation.transform_categorical_isolation(
                raw,
                isolation.PLANE_ONLY_ARM,
                plane_oracle=isolation.PlaneOracle("YZ"),
            )

    def test_profile_only_cannot_access_or_modify_plane_categories(self):
        raw = self.raw()
        before = _plain(raw)
        transformed, _, _ = isolation.transform_categorical_isolation(
            raw,
            isolation.PROFILE_ONLY_ARM,
            profile_oracle=isolation.ProfileOracle("capsule_line_arc"),
        )
        after = _plain(transformed)
        for old, new in zip(before["raw_nodes"], after["raw_nodes"]):
            if NODE_TYPES.tokens[old["node_type_id"]] == "reference_plane":
                self.assertEqual(
                    old["categorical_ids"][3],
                    new["categorical_ids"][3],
                )
                self.assertEqual(
                    old["derived_geometry_mask"][0:9],
                    new["derived_geometry_mask"][0:9],
                )
        self.assertEqual(
            [field.name for field in fields(isolation.ProfileOracle)],
            ["primitive_family"],
        )
        with self.assertRaisesRegex(
            ReplayError, "plane_oracle_capability"
        ):
            isolation.transform_categorical_isolation(
                raw,
                isolation.PROFILE_ONLY_ARM,
                plane_oracle=isolation.PlaneOracle("XY"),
                profile_oracle=isolation.ProfileOracle(
                    "capsule_line_arc"
                ),
            )

    def test_isolated_arms_do_not_access_other_or_target_fields(self):
        class RestrictedExample(dict):
            def __getitem__(self, key):
                if key in (
                    "target",
                    "target_continuous_geometry",
                    "dimensions",
                    "edges",
                    "pointers",
                    "node_count",
                    "history",
                ):
                    raise AssertionError("forbidden target field accessed")
                if (
                    dict.__getitem__(self, "requested_arm")
                    == isolation.PLANE_ONLY_ARM
                    and key == "primitive_family"
                ):
                    raise AssertionError("profile oracle accessed")
                if (
                    dict.__getitem__(self, "requested_arm")
                    == isolation.PROFILE_ONLY_ARM
                    and key == "reference_plane"
                ):
                    raise AssertionError("plane oracle accessed")
                return dict.__getitem__(self, key)

        raw = self.raw()
        for arm in (
            isolation.PLANE_ONLY_ARM,
            isolation.PROFILE_ONLY_ARM,
        ):
            example = RestrictedExample(
                requested_arm=arm,
                reference_plane="XY",
                primitive_family="capsule_line_arc",
            )
            isolation._transform_for_example(raw, arm, example)
        self.assertEqual(
            set(isolation.PlaneOracle.__dataclass_fields__),
            {"reference_plane"},
        )
        self.assertEqual(
            set(isolation.ProfileOracle.__dataclass_fields__),
            {"primitive_family"},
        )

    def test_invalid_authoritative_labels_have_registered_failure(self):
        raw = self.raw()
        family_id = "sf_" + "f" * 64
        for arm, example in (
            (
                isolation.PLANE_ONLY_ARM,
                {
                    "family_id": family_id,
                    "reference_plane": "invalid-plane",
                },
            ),
            (
                isolation.PROFILE_ONLY_ARM,
                {
                    "family_id": family_id,
                    "primitive_family": "invalid-profile",
                },
            ),
        ):
            with self.subTest(arm=arm):
                with self.assertRaises(ReplayError) as raised:
                    isolation._transform_for_example(raw, arm, example)
                self.assertEqual(
                    raised.exception.code, "invalid_oracle_metadata"
                )
                self.assertEqual(raised.exception.detail, family_id)

    def test_every_arm_starts_from_unchanged_original(self):
        raw = self.raw()
        original = _plain(raw)
        outputs = {}
        for arm in isolation.ARMS:
            kwargs = {}
            if arm in (
                isolation.PLANE_ONLY_ARM,
                isolation.FULL_ORACLE_ARM,
            ):
                kwargs["plane_oracle"] = isolation.PlaneOracle("YZ")
            if arm in (
                isolation.PROFILE_ONLY_ARM,
                isolation.FULL_ORACLE_ARM,
            ):
                kwargs["profile_oracle"] = isolation.ProfileOracle(
                    "capsule_line_arc"
                )
            outputs[arm] = isolation.transform_categorical_isolation(
                raw, arm, **kwargs
            )[0]
            self.assertEqual(_plain(raw), original)
        first = outputs[isolation.PLANE_ONLY_ARM]
        first.raw_nodes = ()
        repeated = isolation.transform_categorical_isolation(
            raw,
            isolation.PROFILE_ONLY_ARM,
            profile_oracle=isolation.ProfileOracle("capsule_line_arc"),
        )[0]
        self.assertEqual(
            _plain(repeated),
            _plain(outputs[isolation.PROFILE_ONLY_ARM]),
        )

    def test_parent_equivalent_arms_reconcile_pinned_records(self):
        if not SOURCE_BUNDLE.is_dir() or not PARENT_REPLAY.is_dir():
            self.skipTest("immutable source and parent replay are unavailable")
        source_before = validate_source_manifest(SOURCE_BUNDLE)
        self.assertEqual(
            source_before.manifest_sha256,
            isolation.EXPECTED_SOURCE_MANIFEST_SHA256,
        )
        parent_before = isolation.validate_parent_replay_manifest(
            PARENT_REPLAY
        )
        parent_implementation = (
            isolation.validate_parent_implementation_source()
        )
        self.assertEqual(
            parent_implementation.commit,
            isolation.EXPECTED_PARENT_COMMIT,
        )
        loaded = isolation._load_bundle(SOURCE_BUNDLE)
        parent = isolation._load_parent_replay(PARENT_REPLAY)
        max_operations = isolation._validate_loaded_contract(
            loaded["raw_records"],
            loaded["example_records"],
            loaded["run_metadata"],
            scheduler_job_id=loaded["scheduler_job_id"],
        )
        example = loaded["example_records"][0]
        raw_record = loaded["raw_records"][0]
        parent_by_key = {
            (item["family_id"], item["path"], item["arm"]): item
            for item in parent["records"]
        }
        for path in isolation.PATHS:
            original = _raw_from_json(raw_record[path])
            baseline = validate_and_convert_raw_prediction(
                _stage_two_prediction(original, path),
                max_operations=max_operations,
            )
            for arm in (
                isolation.MODEL_MODEL_ARM,
                isolation.FULL_ORACLE_ARM,
            ):
                transformed, changes, unavailable = (
                    isolation._transform_for_example(
                        original, arm, example
                    )
                )
                result = validate_and_convert_raw_prediction(
                    _stage_two_prediction(transformed, path),
                    max_operations=max_operations,
                )
                parent_arm = isolation.PARENT_ARM_MAP[arm]
                legacy = isolation._legacy_record(
                    example,
                    path,
                    parent_arm,
                    baseline,
                    transformed,
                    changes,
                    unavailable,
                    result,
                )
                self.assertEqual(
                    legacy,
                    parent_by_key[
                        (example["family_id"], path, parent_arm)
                    ],
                )
        self.assertEqual(
            validate_source_manifest(SOURCE_BUNDLE), source_before
        )
        self.assertEqual(
            isolation.validate_parent_replay_manifest(PARENT_REPLAY),
            parent_before,
        )

    def test_factorial_arithmetic_and_transition_directions(self):
        effects = isolation._factorial_effects({
            isolation.MODEL_MODEL_ARM: 10,
            isolation.PLANE_ONLY_ARM: 14,
            isolation.PROFILE_ONLY_ARM: 17,
            isolation.FULL_ORACLE_ARM: 25,
        })
        self.assertEqual(effects["plane_only_gain"], 4)
        self.assertEqual(effects["profile_only_gain"], 7)
        self.assertEqual(effects["combined_gain"], 15)
        self.assertEqual(effects["interaction"], 4)
        records = [
            {
                "controlled_domain_valid": valid,
                "invalid_to_valid": transition == "rescue",
                "valid_to_invalid": transition == "regression",
            }
            for valid, transition in (
                (True, "rescue"),
                (False, "regression"),
                (True, "none"),
            )
        ]
        self.assertEqual(
            isolation._transition_counts(records),
            {
                "attempted_count": 3,
                "controlled_domain_valid_count": 2,
                "controlled_domain_valid_rate": 2 / 3,
                "invalid_to_valid_count": 1,
                "valid_to_invalid_count": 1,
            },
        )

    def test_five_capsule_regressions_are_a_pinned_golden_cohort(self):
        if not PARENT_REPLAY.is_dir():
            self.skipTest("immutable parent replay is unavailable")
        parent = isolation._load_parent_replay(PARENT_REPLAY)
        isolation._validate_parent_capsule_cohort(
            parent["projection_failures"]
        )
        records = []
        failures = []
        for family_id in isolation.CAPSULE_REGRESSION_IDS:
            for arm in isolation.ARMS:
                records.append({
                    "arm": arm,
                    "family_id": family_id,
                    "operation_template": "EE",
                    "path": "predicted_history",
                    "profile_family": "capsule_line_arc",
                })
                if arm in (
                    isolation.PROFILE_ONLY_ARM,
                    isolation.FULL_ORACLE_ARM,
                ):
                    failures.append({
                        "arm": arm,
                        "family_id": family_id,
                        "node_position": 4,
                        "path": "predicted_history",
                        "reason": "nonpositive_fitted_extent",
                    })
        report = isolation._capsule_regression_report(
            records, failures, parent["projection_failures"]
        )
        self.assertEqual(report["cohort_count"], 5)
        self.assertEqual(
            [item["family_id"] for item in report["rows"]],
            list(isolation.CAPSULE_REGRESSION_IDS),
        )
        self.assertTrue(all(
            item["first_appearance_arm"] == isolation.PROFILE_ONLY_ARM
            for item in report["rows"]
        ))
        with self.assertRaisesRegex(
            ReplayError, "capsule_regression_family_set"
        ):
            isolation._capsule_regression_report(
                records,
                failures[:-2],
                parent["projection_failures"],
            )
        extra = dict(
            failures[0],
            family_id="sf_" + "0" * 64,
        )
        with self.assertRaisesRegex(
            ReplayError, "capsule_regression_family_set"
        ):
            isolation._capsule_regression_report(
                records,
                failures + [extra],
                parent["projection_failures"],
            )

    def test_publication_is_deterministic_and_no_replace(self):
        artifacts = {
            name: (name + "\n").encode("utf-8")
            for name in isolation.OUTPUT_ARTIFACTS
        }
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first"
            second = Path(directory) / "second"
            isolation._publish_artifacts(first, artifacts)
            isolation._publish_artifacts(second, artifacts)
            self.assertEqual(
                (first / isolation.OUTPUT_MANIFEST).read_bytes(),
                (second / isolation.OUTPUT_MANIFEST).read_bytes(),
            )
            manifest = (
                first / isolation.OUTPUT_MANIFEST
            ).read_text().splitlines()
            self.assertEqual(len(manifest), len(isolation.OUTPUT_ARTIFACTS))
            self.assertEqual(
                [line.split("  ", 1)[1] for line in manifest],
                sorted(isolation.OUTPUT_ARTIFACTS),
            )
            for line in manifest:
                digest, name = line.split("  ", 1)
                self.assertEqual(
                    digest,
                    hashlib.sha256((first / name).read_bytes()).hexdigest(),
                )
            with self.assertRaisesRegex(ReplayError, "output_collision"):
                isolation._publish_artifacts(first, artifacts)
            missing = dict(artifacts)
            missing.pop(isolation.OUTPUT_ARTIFACTS[0])
            with self.assertRaisesRegex(
                ReplayError, "output_artifact_set"
            ):
                isolation._publish_artifacts(
                    Path(directory) / "missing", missing
                )
            extra = dict(artifacts, unexpected=b"unexpected\n")
            with self.assertRaisesRegex(
                ReplayError, "output_artifact_set"
            ):
                isolation._publish_artifacts(
                    Path(directory) / "extra", extra
                )
        with self.assertRaisesRegex(
            ReplayError, "output_overlaps_immutable_input"
        ):
            isolation._require_distinct_output(
                SOURCE_BUNDLE / "new-output",
                SOURCE_BUNDLE,
                PARENT_REPLAY,
            )
        with self.assertRaisesRegex(
            ReplayError, "output_overlaps_immutable_input"
        ):
            isolation._require_distinct_output(
                SOURCE_BUNDLE.parent,
                SOURCE_BUNDLE,
                PARENT_REPLAY,
            )
        with tempfile.TemporaryDirectory() as directory:
            alias = Path(directory) / "source-alias"
            alias.symlink_to(SOURCE_BUNDLE, target_is_directory=True)
            with self.assertRaisesRegex(
                ReplayError, "output_overlaps_immutable_input"
            ):
                isolation._require_distinct_output(
                    alias / "new-output",
                    SOURCE_BUNDLE,
                    PARENT_REPLAY,
                )

    def test_malformed_parent_fails_before_scientific_loading(self):
        if not SOURCE_BUNDLE.is_dir():
            self.skipTest("immutable source bundle is unavailable")
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory) / "parent"
            parent.mkdir()
            (parent / "sha256-manifest.txt").write_text("")
            with mock.patch.object(
                isolation, "_load_bundle"
            ) as source_loader, mock.patch.object(
                isolation, "_load_parent_replay"
            ) as parent_loader:
                with self.assertRaisesRegex(
                    ReplayError, "parent_artifact_set_mismatch"
                ):
                    isolation.run_categorical_isolation(
                        SOURCE_BUNDLE,
                        parent,
                        Path(directory) / "output",
                    )
                source_loader.assert_not_called()
                parent_loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
