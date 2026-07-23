from __future__ import annotations

import unittest

from prototype.counterfactual_edits.candidates import select_oriented_candidates
from prototype.counterfactual_edits.config import CounterfactualConfig
from prototype.counterfactual_edits.identity import edit_family_id, edit_sample_id
from prototype.counterfactual_edits.model import EditTarget, EditType


class IdentityTests(unittest.TestCase):
    def test_direction_and_semantics_are_identity_inputs(self):
        target = EditTarget(0, "sketch_1", "sketch_extent")
        forward = edit_family_id(EditType.PROFILE_EXTENT, target, "sf_a", "sf_b")
        reverse = edit_family_id(EditType.PROFILE_EXTENT, target, "sf_b", "sf_a")
        other = edit_family_id(
            EditType.OPERATION_DIRECTION,
            EditTarget(0, "extrude_1", "direction"),
            "sf_a",
            "sf_b",
        )
        self.assertNotEqual(forward, reverse)
        self.assertNotEqual(forward, other)

    def test_encoding_specific_sample_identity(self):
        family = edit_family_id(
            EditType.PROFILE_EXTENT,
            EditTarget(0, "sketch_1", "sketch_extent"),
            "sf_a",
            "sf_b",
        )
        continuous = edit_sample_id(family, "continuous", "sv_a", "sv_b")
        quantized = edit_sample_id(family, "quantized", "sv_c", "sv_d")
        self.assertNotEqual(continuous, quantized)

    def test_selected_identities_do_not_depend_on_seed(self):
        left = select_oriented_candidates(CounterfactualConfig(68, seed=1))
        right = select_oriented_candidates(CounterfactualConfig(68, seed=999))
        summarize = lambda items: {
            (
                item.identified.lower_source_family_id,
                item.identified.higher_source_family_id,
                item.identified.candidate.edit_type.value,
                item.identified.candidate.target,
                item.base_source,
                item.edited_source,
            )
            for item in items
        }
        self.assertEqual(summarize(left), summarize(right))
