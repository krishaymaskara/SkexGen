from __future__ import annotations

from dataclasses import replace
import unittest

from prototype.counterfactual_edits.candidates import iter_accepted_candidates
from prototype.counterfactual_edits.config import CounterfactualConfig, EditConfigurationError
from prototype.counterfactual_edits.edits import (
    semantic_difference_count,
    validate_edit_endpoints,
)
from prototype.counterfactual_edits.model import EditType
from prototype.controlled_data.factors import (
    ExtentBand,
    OperationTemplate,
    PrimitiveFamily,
    ReferencePlane,
)
from prototype.representation.model import BooleanMode, Direction


class EditApplicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = CounterfactualConfig(68)
        cls.examples = {}
        for candidate in iter_accepted_candidates(cls.config):
            cls.examples.setdefault(candidate.edit_type, candidate)
            cls.depth_examples = getattr(cls, "depth_examples", {})
            cls.depth_examples.setdefault(candidate.endpoint_a.history_depth, candidate)
            if len(cls.examples) == len(EditType):
                if set(cls.depth_examples) == {1, 2}:
                    break

    def test_all_edit_types_change_exactly_one_factor_and_are_feasible(self):
        self.assertEqual(set(self.examples), set(EditType))
        for edit_type, candidate in self.examples.items():
            with self.subTest(edit_type=edit_type.value):
                self.assertEqual(
                    semantic_difference_count(
                        candidate.endpoint_a, candidate.endpoint_b
                    ),
                    1,
                )
                validate_edit_endpoints(
                    edit_type,
                    candidate.target,
                    candidate.endpoint_a,
                    candidate.endpoint_b,
                    self.config.factor_config,
                )

    def test_malformed_declared_edit_is_rejected(self):
        candidate = self.examples[EditType.EXTRUSION_DISTANCE]
        with self.assertRaises(EditConfigurationError):
            validate_edit_endpoints(
                candidate.edit_type,
                self.examples[EditType.PROFILE_EXTENT].target,
                candidate.endpoint_a,
                candidate.endpoint_b,
                self.config.factor_config,
            )

    def test_missing_and_extra_depth_dependent_entries_are_rejected(self):
        for depth, candidate in self.depth_examples.items():
            source = candidate.endpoint_a
            for field in ("sketch_extents", "directions", "operation_parameters"):
                values = getattr(source, field)
                replacements = (
                    values[:-1],
                    values + (values[-1],),
                )
                for malformed in replacements:
                    with self.subTest(depth=depth, field=field, length=len(malformed)):
                        with self.assertRaises(EditConfigurationError):
                            validate_edit_endpoints(
                                candidate.edit_type,
                                candidate.target,
                                replace(source, **{field: malformed}),
                                candidate.endpoint_b,
                                self.config.factor_config,
                            )

    def test_hidden_appended_value_cannot_accompany_intended_edit(self):
        candidate = self.depth_examples[1]
        malformed = replace(
            candidate.endpoint_b,
            directions=candidate.endpoint_b.directions
            + (candidate.endpoint_b.directions[-1],),
        )
        with self.assertRaises(EditConfigurationError):
            validate_edit_endpoints(
                candidate.edit_type,
                candidate.target,
                candidate.endpoint_a,
                malformed,
                self.config.factor_config,
            )

    def test_depth_specific_boolean_modes_are_enforced(self):
        depth_one = self.depth_examples[1]
        with self.assertRaises(EditConfigurationError):
            validate_edit_endpoints(
                depth_one.edit_type,
                depth_one.target,
                replace(
                    depth_one.endpoint_a,
                    later_boolean_mode=BooleanMode.JOIN,
                ),
                depth_one.endpoint_b,
                self.config.factor_config,
            )
        depth_two = self.depth_examples[2]
        with self.assertRaises(EditConfigurationError):
            validate_edit_endpoints(
                depth_two.edit_type,
                depth_two.target,
                replace(depth_two.endpoint_a, later_boolean_mode=None),
                depth_two.endpoint_b,
                self.config.factor_config,
            )

    def test_undeclared_factor_changes_are_rejected_exhaustively(self):
        candidate = next(
            item
            for item in iter_accepted_candidates(self.config)
            if item.endpoint_a.history_depth == 2
            and item.edit_type is EditType.PROFILE_EXTENT
        )
        base = candidate.endpoint_a
        edited = candidate.endpoint_b
        changes = {
            "operation_template": OperationTemplate.RR,
            "primitive_family": (
                PrimitiveFamily.CIRCLE
                if edited.primitive_family is not PrimitiveFamily.CIRCLE
                else PrimitiveFamily.RECTANGLE_LINES
            ),
            "reference_plane": (
                ReferencePlane.XZ
                if edited.reference_plane is not ReferencePlane.XZ
                else ReferencePlane.XY
            ),
            "configured_extent_band": (
                ExtentBand.EXTRAPOLATION
                if edited.configured_extent_band is ExtentBand.IN_RANGE
                else ExtentBand.IN_RANGE
            ),
            "later_boolean_mode": (
                BooleanMode.CUT
                if edited.later_boolean_mode is BooleanMode.JOIN
                else BooleanMode.JOIN
            ),
            "directions": (
                (
                    Direction.NEGATIVE
                    if edited.directions[0] is Direction.POSITIVE
                    else Direction.POSITIVE
                ),
                edited.directions[1],
            ),
            "operation_parameters": (
                edited.operation_parameters[0],
                (
                    self.config.factor_config.extrusion_distances[-1]
                    if edited.operation_parameters[1]
                    != self.config.factor_config.extrusion_distances[-1]
                    else self.config.factor_config.extrusion_distances[0]
                ),
            ),
        }
        for field, value in changes.items():
            with self.subTest(field=field):
                with self.assertRaises(EditConfigurationError):
                    validate_edit_endpoints(
                        candidate.edit_type,
                        candidate.target,
                        base,
                        replace(edited, **{field: value}),
                        self.config.factor_config,
                    )
