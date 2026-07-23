from __future__ import annotations

from collections import Counter
import unittest

from prototype.representation.model import BooleanMode, Direction, GeometryEncoding

from prototype.controlled_data.builders import build_history
from prototype.controlled_data.config import GeneratorConfig
from prototype.controlled_data.factors import (
    ExtentBand,
    OperationTemplate,
    PhysicalSource,
    PrimitiveFamily,
    ReferencePlane,
    factor_blocks,
)
from prototype.controlled_data.feasibility import (
    FeasibilityStatus,
    evaluate_feasibility,
)
from prototype.controlled_data.identity import source_family_id


def source(
    template,
    primitive,
    extents,
    directions,
    mode,
    parameters,
    plane=ReferencePlane.XY,
    band=ExtentBand.IN_RANGE,
):
    return PhysicalSource(
        OperationTemplate(template),
        PrimitiveFamily(primitive),
        plane,
        tuple(float(item) for item in extents),
        tuple(Direction(item) for item in directions),
        None if mode is None else BooleanMode(mode),
        tuple(float(item) for item in parameters),
        band,
    )


class ObservedAdroitFailureTests(unittest.TestCase):
    def test_all_thirteen_observed_families_have_audited_rejections(self):
        cases = (
            ("ER", "capsule_line_arc", "XY", (2, 2), ("positive", "positive"), "cut", (3, 45), "in_range", "cut_no_positive_volume_overlap"),
            ("ER", "circle", "YZ", (1.5, 1.5), ("positive", "positive"), "cut", (0.5, 90), "in_range", "cut_no_positive_volume_overlap"),
            ("EE", "capsule_line_arc", "XZ", (2, 2), ("positive", "negative"), "cut", (1, 1), "in_range", "cut_no_positive_volume_overlap"),
            ("RR", "capsule_line_arc", "XY", (2, 2.5), ("negative", "positive"), "cut", (45, 45), "extrapolation", "cut_no_positive_volume_overlap"),
            ("RR", "capsule_line_arc", "YZ", (1.5, 2), ("negative", "negative"), "cut", (90, 180), "in_range", "cut_complete_subtraction"),
            ("RE", "capsule_line_arc", "XZ", (2, 1), ("positive", "negative"), "join", (360, 1.5), "in_range", "join_tool_contained"),
            ("RE", "circle", "XY", (3, 1.5), ("positive", "positive"), "cut", (90, 0.5), "extrapolation", "cut_no_positive_volume_overlap"),
            ("RE", "capsule_line_arc", "XZ", (3, 2), ("negative", "positive"), "join", (180, 1), "extrapolation", "join_tool_contained"),
            ("EE", "circle", "XY", (1, 3), ("negative", "positive"), "cut", (1.5, 3), "extrapolation", "cut_no_positive_volume_overlap"),
            ("RE", "capsule_line_arc", "XZ", (1.5, 1), ("negative", "negative"), "cut", (90, 3), "in_range", "cut_no_positive_volume_overlap"),
            ("EE", "circle", "XY", (1, 1.5), ("positive", "positive"), "cut", (2, 2), "in_range", "cut_complete_subtraction"),
            ("EE", "rectangle_lines", "YZ", (1, 1), ("negative", "negative"), "join", (1.5, 1.5), "in_range", "join_duplicate_geometry"),
            ("RR", "rectangle_lines", "XZ", (1.5, 2.5), ("negative", "negative"), "cut", (270, 270), "extrapolation", "cut_complete_subtraction"),
        )
        expected_ids = (
            "sf_102c420b79a43f25aff7bdc900d3f7fbf00803a47c524648796832117ff0e90e",
            "sf_5c68b1b7e28d8e52e6f7840ff922df2c97969eb62a2ff4d492741dc9968dd5ce",
            "sf_76c2d8b31f50c4c160abd4341d1ee9fe3e5a2e49fd34ff03858293ca8630f316",
            "sf_78a0df60a2e910e16cd7d871ef70a18adf94a9773f43ef26697832755cac1120",
            "sf_834d6991a86fcb2238ef80a9c3d58f149b6e861e79f60411a9d6bc6806f708b4",
            "sf_879edd04d832317b8f4c4e07c87b830f017875c008a5178b5516957fe1b6ccfc",
            "sf_acda903989dd3b4d9c540e0daac4322ebb75b728a7a4eeac22001552e13f81d0",
            "sf_afa376d7a13cdab4465706b8d4943a1bc7f00d0a824e1b24a14af8c2584bcfc5",
            "sf_d0b3292289b241b95b9d3e97a8d09695dfdb913cbd43781d62d87863ea4c4d7a",
            "sf_e6bfaaa1d29c0dd807580ef5aabffdce833d67f16e53fd5b900ccccd8d00b228",
            "sf_e75400117f3d57aa2195fe5dc04452c2d2167faf5be33ae0a78f3c1a8ab3827b",
            "sf_f2d66353426649e7632c797c3a5d53a8f991e8ce9f61edfbe3a9feaf6ada29b0",
            "sf_ff802589b8d44a387bd29413ea1ff869fa82496a8e06007052e6c10ac4a792e4",
        )
        observed = Counter()
        for case, expected_id in zip(cases, expected_ids):
            template, primitive, plane, extents, directions, mode, parameters, band, expected = case
            with self.subTest(template=template, primitive=primitive, expected=expected):
                item = source(
                    template,
                    primitive,
                    extents,
                    directions,
                    mode,
                    parameters,
                    ReferencePlane(plane),
                    ExtentBand(band),
                )
                status = evaluate_feasibility(item).status.value
                self.assertEqual(status, expected)
                self.assertEqual(
                    source_family_id(build_history(item, GeometryEncoding.CONTINUOUS)),
                    expected_id,
                )
                observed[status] += 1
        self.assertEqual(
            observed,
            {
                "cut_no_positive_volume_overlap": 7,
                "cut_complete_subtraction": 3,
                "join_tool_contained": 2,
                "join_duplicate_geometry": 1,
            },
        )


class AnalyticalBoundaryTests(unittest.TestCase):
    def assert_status(self, item, expected):
        self.assertEqual(evaluate_feasibility(item).status, expected)

    def test_ee_join_and_cut_direction_distance_extent_boundaries(self):
        self.assert_status(
            source("EE", "rectangle_lines", (1, 1), ("positive", "negative"), "join", (1, 1)),
            FeasibilityStatus.ACCEPTED,
        )
        self.assert_status(
            source("EE", "circle", (2, 1), ("positive", "positive"), "join", (2, 1)),
            FeasibilityStatus.JOIN_TOOL_CONTAINED,
        )
        self.assert_status(
            source("EE", "circle", (1, 1), ("positive", "positive"), "join", (2, 2)),
            FeasibilityStatus.JOIN_DUPLICATE_GEOMETRY,
        )
        self.assert_status(
            source("EE", "circle", (2, 1), ("positive", "positive"), "cut", (2, 1)),
            FeasibilityStatus.ACCEPTED,
        )
        self.assert_status(
            source("EE", "circle", (1, 2), ("positive", "positive"), "cut", (1, 1)),
            FeasibilityStatus.CUT_COMPLETE_SUBTRACTION,
        )
        self.assert_status(
            source("EE", "circle", (2, 1), ("positive", "negative"), "cut", (2, 1)),
            FeasibilityStatus.CUT_NO_POSITIVE_VOLUME_OVERLAP,
        )

    def test_rr_sector_overlap_full_rotation_and_exact_duplicates(self):
        self.assert_status(
            source("RR", "rectangle_lines", (1, 1), ("positive", "negative"), "cut", (180, 180)),
            FeasibilityStatus.CUT_NO_POSITIVE_VOLUME_OVERLAP,
        )
        self.assert_status(
            source("RR", "rectangle_lines", (1, 1), ("positive", "negative"), "cut", (270, 180)),
            FeasibilityStatus.ACCEPTED,
        )
        self.assert_status(
            source("RR", "rectangle_lines", (1, 1), ("positive", "negative"), "cut", (270, 360)),
            FeasibilityStatus.CUT_COMPLETE_SUBTRACTION,
        )
        self.assert_status(
            source("RR", "circle", (1, 1), ("positive", "negative"), "join", (360, 360)),
            FeasibilityStatus.JOIN_DUPLICATE_GEOMETRY,
        )
        self.assert_status(
            source("RR", "circle", (2, 1), ("positive", "negative"), "join", (360, 45)),
            FeasibilityStatus.JOIN_TOOL_CONTAINED,
        )

    def test_mixed_orientation_and_radial_noncontainment_witnesses(self):
        self.assert_status(
            source("ER", "circle", (1, 2), ("positive", "positive"), "cut", (1, 90)),
            FeasibilityStatus.CUT_NO_POSITIVE_VOLUME_OVERLAP,
        )
        self.assert_status(
            source("ER", "circle", (3, 1), ("positive", "negative"), "cut", (3, 180)),
            FeasibilityStatus.ACCEPTED,
        )
        self.assert_status(
            source("RE", "circle", (1, 3), ("positive", "negative"), "join", (180, 3)),
            FeasibilityStatus.ACCEPTED,
        )
        self.assert_status(
            source("RE", "circle", (2, 1), ("positive", "positive"), "cut", (360, 1)),
            FeasibilityStatus.ACCEPTED,
        )

    def test_profile_specific_containment_certificates_and_ambiguous_mixed_cases(self):
        rectangle = source(
            "ER", "rectangle_lines", (1, 2), ("positive", "positive"), "cut", (2, 360)
        )
        circle = source("ER", "circle", (1, 2), ("positive", "positive"), "cut", (2, 360))
        capsule = source(
            "ER", "capsule_line_arc", (1, 2), ("positive", "positive"), "cut", (2, 360)
        )
        self.assert_status(rectangle, FeasibilityStatus.CUT_COMPLETE_SUBTRACTION)
        self.assert_status(circle, FeasibilityStatus.MIXED_CONTAINMENT_NOT_CERTIFIED)
        self.assert_status(capsule, FeasibilityStatus.MIXED_CONTAINMENT_NOT_CERTIFIED)
        self.assert_status(
            source("RE", "circle", (2, 1), ("positive", "positive"), "join", (360, 2)),
            FeasibilityStatus.MIXED_CONTAINMENT_NOT_CERTIFIED,
        )

    def test_opposite_side_joins_with_positive_area_face_are_accepted(self):
        for template, parameters in (("EE", (1, 1)), ("ER", (1, 90))):
            with self.subTest(template=template):
                self.assert_status(
                    source(
                        template,
                        "capsule_line_arc",
                        (1, 1),
                        ("positive", "negative"),
                        "join",
                        parameters,
                    ),
                    FeasibilityStatus.ACCEPTED,
                )

    def test_join_disconnected_status_is_reserved_but_not_emitted_by_current_grid(self):
        self.assertIn(FeasibilityStatus.JOIN_DISCONNECTED, set(FeasibilityStatus))
        for block in factor_blocks(GeneratorConfig(68)):
            for index in range(block.size):
                self.assertIsNot(
                    evaluate_feasibility(block.decode(index)).status,
                    FeasibilityStatus.JOIN_DISCONNECTED,
                )
