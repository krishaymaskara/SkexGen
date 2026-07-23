from __future__ import annotations

from dataclasses import replace
import unittest

from prototype.controlled_data.builders import build_history
from prototype.controlled_data.factors import PrimitiveFamily
from prototype.representation.model import GeometryEncoding

from prototype.counterfactual_edits.candidates import iter_accepted_candidates
from prototype.counterfactual_edits.config import CounterfactualConfig, EditConfigurationError
from prototype.counterfactual_edits.locality import (
    build_locality_ground_truth,
    validate_locality,
)
from prototype.counterfactual_edits.model import AddressKind, EditType


class LocalityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.examples = {}
        for candidate in iter_accepted_candidates(CounterfactualConfig(68)):
            cls.examples.setdefault(candidate.edit_type, candidate)
            if len(cls.examples) == len(EditType):
                break

    def test_exact_allowed_path_categories(self):
        expected_kinds = {
            EditType.PROFILE_EXTENT: AddressKind.SKETCH_ELEMENT_GEOMETRY,
            EditType.EXTRUSION_DISTANCE: AddressKind.NODE_GEOMETRY,
            EditType.REVOLVE_ANGLE: AddressKind.NODE_GEOMETRY,
            EditType.OPERATION_DIRECTION: AddressKind.NODE_FIELD,
            EditType.BOOLEAN_MODE: AddressKind.NODE_FIELD,
        }
        for edit_type, candidate in self.examples.items():
            base = build_history(candidate.endpoint_a, GeometryEncoding.CONTINUOUS)
            edited = build_history(candidate.endpoint_b, GeometryEncoding.CONTINUOUS)
            locality = build_locality_ground_truth(
                base, edited, edit_type, candidate.target
            )
            self.assertTrue(locality.allowed_changed_paths)
            self.assertEqual(
                {item.kind for item in locality.allowed_changed_paths},
                {expected_kinds[edit_type]},
            )
            self.assertEqual(locality.structural_graph_edit_distance, 0)
            self.assertEqual(locality.semantic_parameter_edit_distance, 1)
            validate_locality(
                base, edited, locality, edit_type, candidate.target
            )

    def test_swapped_ground_truth_is_rejected(self):
        first = self.examples[EditType.EXTRUSION_DISTANCE]
        second = self.examples[EditType.PROFILE_EXTENT]
        base = build_history(first.endpoint_a, GeometryEncoding.CONTINUOUS)
        edited = build_history(first.endpoint_b, GeometryEncoding.CONTINUOUS)
        wrong = build_locality_ground_truth(
            build_history(second.endpoint_a, GeometryEncoding.CONTINUOUS),
            build_history(second.endpoint_b, GeometryEncoding.CONTINUOUS),
            second.edit_type,
            second.target,
        )
        with self.assertRaises(EditConfigurationError):
            validate_locality(
                base, edited, wrong, first.edit_type, first.target
            )

    def test_malformed_locality_metadata_is_rejected(self):
        candidate = self.examples[EditType.EXTRUSION_DISTANCE]
        base = build_history(candidate.endpoint_a, GeometryEncoding.CONTINUOUS)
        edited = build_history(candidate.endpoint_b, GeometryEncoding.CONTINUOUS)
        locality = build_locality_ground_truth(
            base, edited, candidate.edit_type, candidate.target
        )
        cases = {
            "omitted_unchanged": replace(
                locality,
                expected_unchanged_paths=locality.expected_unchanged_paths[1:],
            ),
            "extra_allowed": replace(
                locality,
                allowed_changed_paths=locality.allowed_changed_paths
                + (locality.expected_unchanged_paths[0],),
            ),
            "overlap": replace(
                locality,
                expected_unchanged_paths=locality.expected_unchanged_paths
                + (locality.allowed_changed_paths[0],),
            ),
            "wrong_downstream": replace(
                locality, causal_downstream_operation_ids=("not_an_operation",)
            ),
            "wrong_structural_distance": replace(
                locality, structural_graph_edit_distance=1
            ),
            "wrong_semantic_distance": replace(
                locality, semantic_parameter_edit_distance=2
            ),
            "duplicate_allowed": replace(
                locality,
                allowed_changed_paths=locality.allowed_changed_paths
                + (locality.allowed_changed_paths[0],),
            ),
            "duplicate_unchanged": replace(
                locality,
                expected_unchanged_paths=locality.expected_unchanged_paths
                + (locality.expected_unchanged_paths[0],),
            ),
        }
        nonexistent = replace(
            locality.allowed_changed_paths[0], owner_id="missing_owner"
        )
        cases["nonexistent"] = replace(
            locality,
            allowed_changed_paths=(nonexistent,),
        )
        for name, malformed in cases.items():
            with self.subTest(case=name):
                with self.assertRaises(EditConfigurationError):
                    validate_locality(
                        base,
                        edited,
                        malformed,
                        candidate.edit_type,
                        candidate.target,
                    )

    def test_profile_extent_paths_are_exact_for_every_primitive_family(self):
        examples = {}
        for candidate in iter_accepted_candidates(CounterfactualConfig(68)):
            if candidate.edit_type is EditType.PROFILE_EXTENT:
                examples.setdefault(candidate.endpoint_a.primitive_family, candidate)
            if set(examples) == set(PrimitiveFamily):
                break
        self.assertEqual(set(examples), set(PrimitiveFamily))
        for family, candidate in examples.items():
            base = build_history(candidate.endpoint_a, GeometryEncoding.CONTINUOUS)
            edited = build_history(candidate.endpoint_b, GeometryEncoding.CONTINUOUS)
            locality = build_locality_ground_truth(
                base, edited, candidate.edit_type, candidate.target
            )
            with self.subTest(family=family.value):
                self.assertTrue(locality.allowed_changed_paths)
                validate_locality(
                    base,
                    edited,
                    locality,
                    candidate.edit_type,
                    candidate.target,
                )
                malformed = replace(
                    locality,
                    allowed_changed_paths=locality.allowed_changed_paths[:-1],
                    expected_unchanged_paths=locality.expected_unchanged_paths
                    + (locality.allowed_changed_paths[-1],),
                )
                with self.assertRaises(EditConfigurationError):
                    validate_locality(
                        base,
                        edited,
                        malformed,
                        candidate.edit_type,
                        candidate.target,
                    )
