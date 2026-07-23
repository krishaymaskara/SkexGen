from __future__ import annotations

from collections import Counter
import hashlib
import unittest

from prototype.counterfactual_edits.candidates import (
    _hash_orient,
    _orient_anchors,
    candidate_counts,
    declared_secondary_tokens,
    orientation_label,
    primary_token,
    select_oriented_candidates,
    undirected_key,
)
from prototype.counterfactual_edits.config import (
    CounterfactualConfig,
    primary_token_declarations,
    required_minimum,
)


class CandidateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = CounterfactualConfig(68, seed=11)
        cls.counts = candidate_counts(cls.config)
        cls.selected = select_oriented_candidates(cls.config)

    def test_exact_candidate_space_counts(self):
        self.assertEqual(self.counts["raw_directed"], 1_609_020)
        self.assertEqual(self.counts["raw_undirected"], 804_510)
        self.assertEqual(self.counts["accepted_undirected"], 448_920)
        self.assertEqual(
            self.counts["accepted_by_edit_type"],
            {
                "boolean_mode": 30_150,
                "extrusion_distance": 90_180,
                "operation_direction": 83_430,
                "profile_extent": 155_772,
                "revolve_angle": 89_388,
            },
        )

    def test_coverage_declarations_and_endpoint_disjoint_selection(self):
        self.assertEqual(required_minimum(), 68)
        self.assertEqual(len(self.counts["accepted_by_primary_token"]), 68)
        self.assertEqual(len(declared_secondary_tokens(self.config)), 100)
        self.assertEqual(len(self.selected), 68)
        endpoints = [
            value
            for item in self.selected
            for value in (
                item.identified.lower_source_family_id,
                item.identified.higher_source_family_id,
            )
        ]
        self.assertEqual(len(endpoints), 136)
        self.assertEqual(len(set(endpoints)), 136)
        self.assertEqual(len({primary_token(item.identified.candidate) for item in self.selected}), 68)

    def test_primary_cells_are_derived_and_match_enumeration(self):
        declared = set(primary_token_declarations())
        enumerated = {
            (
                parts[0],
                parts[1],
                int(parts[2]),
                parts[3],
            )
            for text in self.counts["accepted_by_primary_token"]
            for parts in (text.split("|"),)
        }
        self.assertEqual(declared, enumerated)
        self.assertEqual(len({item[:3] for item in declared}), 34)
        self.assertEqual(len(declared), 68)
        self.assertEqual(required_minimum(), len(declared))

    def test_orientation_quotas_are_exact(self):
        observed = Counter(
            (item.identified.candidate.edit_type.value, orientation_label(item))
            for item in self.selected
        )
        self.assertEqual(
            observed,
            {
                ("profile_extent", "increase"): 10,
                ("profile_extent", "decrease"): 10,
                ("extrusion_distance", "increase"): 5,
                ("extrusion_distance", "decrease"): 5,
                ("revolve_angle", "increase"): 5,
                ("revolve_angle", "decrease"): 5,
                ("operation_direction", "positive_to_negative"): 10,
                ("operation_direction", "negative_to_positive"): 10,
                ("boolean_mode", "join_to_cut"): 4,
                ("boolean_mode", "cut_to_join"): 4,
            },
        )

    def test_anchor_orientation_is_seed_and_input_order_independent(self):
        identified = tuple(item.identified for item in self.selected)
        forward = _orient_anchors(identified)
        reverse = _orient_anchors(tuple(reversed(identified)))
        descriptor = lambda items: sorted(
            (
                undirected_key(item.identified),
                orientation_label(item),
            )
            for item in items
        )
        self.assertEqual(descriptor(forward), descriptor(reverse))
        raw = tuple(item.candidate for item in identified)
        selected_forward = select_oriented_candidates(
            self.config, accepted_candidates=raw
        )
        selected_reverse = select_oriented_candidates(
            self.config, accepted_candidates=reversed(raw)
        )
        self.assertEqual(descriptor(selected_forward), descriptor(selected_reverse))

    def test_filler_hash_orientation_has_stable_golden(self):
        identified = self.selected[0].identified
        oriented = _hash_orient(identified)
        digest = hashlib.sha256(
            (
                oriented.identified.lower_source_family_id
                + "|"
                + oriented.identified.higher_source_family_id
                + "|"
                + orientation_label(oriented)
            ).encode("ascii")
        ).hexdigest()
        self.assertEqual(
            digest,
            "d5af32893a9179930a0acfbb358f0562896fc95d3d020587018598f2b5ff0007",
        )

    def test_undirected_keys_are_unique(self):
        keys = [undirected_key(item.identified) for item in self.selected]
        self.assertEqual(len(keys), len(set(keys)))
        first = self.selected[0].identified
        reversed_candidate = type(first.candidate)(
            first.candidate.edit_type,
            first.candidate.target,
            first.candidate.endpoint_b,
            first.candidate.endpoint_a,
        )
        from prototype.counterfactual_edits.candidates import identify_candidate

        self.assertEqual(
            undirected_key(first), undirected_key(identify_candidate(reversed_candidate))
        )

    def test_larger_selection_does_not_reuse_endpoints(self):
        selected = select_oriented_candidates(CounterfactualConfig(70, seed=37))
        endpoints = [
            value
            for item in selected
            for value in (
                item.identified.lower_source_family_id,
                item.identified.higher_source_family_id,
            )
        ]
        self.assertEqual(len(selected), 70)
        self.assertEqual(len(endpoints), len(set(endpoints)))
