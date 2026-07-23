from __future__ import annotations

from dataclasses import replace
import json
import unittest

from prototype.controlled_data.builders import build_history
from prototype.representation.model import GeometryEncoding

from prototype.counterfactual_edits.candidates import iter_accepted_candidates
from prototype.counterfactual_edits.config import CounterfactualConfig
from prototype.counterfactual_edits.identity import edit_family_id, edit_sample_id
from prototype.counterfactual_edits.locality import build_locality_ground_truth
from prototype.counterfactual_edits.model import EditSample
from prototype.counterfactual_edits.serialization import (
    EditSerializationError,
    edit_sample_from_json,
    edit_sample_to_json,
)


class SerializationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        candidate = next(iter_accepted_candidates(CounterfactualConfig(68)))
        base = build_history(candidate.endpoint_a, GeometryEncoding.CONTINUOUS)
        edited = build_history(candidate.endpoint_b, GeometryEncoding.CONTINUOUS)
        locality = build_locality_ground_truth(
            base, edited, candidate.edit_type, candidate.target
        )
        family_id = edit_family_id(
            candidate.edit_type, candidate.target, "sf_base", "sf_edited"
        )
        cls.sample = EditSample(
            edit_sample_id(family_id, "continuous", "sv_base", "sv_edited"),
            family_id,
            "continuous",
            "sf_base",
            "sf_edited",
            "sv_base",
            "sv_edited",
            candidate.edit_type,
            candidate.target,
            candidate.endpoint_a.sketch_extents[0],
            candidate.endpoint_b.sketch_extents[0],
            locality,
        )

    def test_canonical_round_trip(self):
        payload = edit_sample_to_json(self.sample)
        restored = edit_sample_from_json(payload)
        self.assertEqual(edit_sample_to_json(restored), payload)

    def test_unknown_and_duplicate_fields_are_rejected(self):
        value = json.loads(edit_sample_to_json(self.sample))
        value["unknown"] = True
        with self.assertRaises(EditSerializationError):
            edit_sample_from_json(json.dumps(value))
        payload = edit_sample_to_json(self.sample)
        with self.assertRaises(EditSerializationError):
            edit_sample_from_json(payload[:-1] + ',"edit_sample_id":"duplicate"}')

    def test_boolean_is_rejected_as_operation_index(self):
        value = json.loads(edit_sample_to_json(self.sample))
        value["edit"]["operation_index"] = True
        with self.assertRaises(EditSerializationError):
            edit_sample_from_json(json.dumps(value))

    def test_unknown_geometry_encoding_is_rejected(self):
        value = json.loads(edit_sample_to_json(self.sample))
        value["geometry_encoding"] = "other"
        with self.assertRaises(EditSerializationError):
            edit_sample_from_json(json.dumps(value))
