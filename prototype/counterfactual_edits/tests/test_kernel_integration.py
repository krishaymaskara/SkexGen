from __future__ import annotations

import unittest

from prototype.counterfactual_edits.kernel_integration import (
    PairKernelAuditError,
    build_pair_execution_audit,
)


def result(source_id, sample_id, encoding, volume=1.0):
    return {
        "source_family_id": source_id,
        "sample_id": sample_id,
        "geometry_encoding": encoding,
        "kernel_status": "success",
        "final_metrics": {
            "volume": volume,
            "bounding_box": [0, 0, 0, 1, 1, 1],
            "solid_count": 1,
            "face_count": 6,
            "edge_count": 12,
            "vertex_count": 8,
        },
    }


class KernelIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.manifest = {
            "configuration_sha256": "config",
            "samples": [
                {
                    "edit_sample_id": "ev_c",
                    "edit_family_id": "ef_1",
                    "geometry_encoding": "continuous",
                    "base_source_family_id": "sf_a",
                    "edited_source_family_id": "sf_b",
                    "base_sample_id": "sv_ac",
                    "edited_sample_id": "sv_bc",
                },
                {
                    "edit_sample_id": "ev_q",
                    "edit_family_id": "ef_1",
                    "geometry_encoding": "quantized",
                    "base_source_family_id": "sf_a",
                    "edited_source_family_id": "sf_b",
                    "base_sample_id": "sv_aq",
                    "edited_sample_id": "sv_bq",
                },
            ],
        }
        self.report = {
            "report_version": 1,
            "samples": [
                result("sf_a", "sv_ac", "continuous"),
                result("sf_a", "sv_aq", "quantized"),
                result("sf_b", "sv_bc", "continuous", 2.0),
                result("sf_b", "sv_bq", "quantized", 2.0),
            ],
        }

    def test_successful_join_requires_both_endpoints_and_encoding_agreement(self):
        audit = build_pair_execution_audit(self.manifest, self.report)
        self.assertEqual(audit["successful_edit_samples"], 2)
        self.assertEqual(audit["encoding_agreeing_physical_endpoints"], 2)
        self.assertTrue(
            all(item["both_endpoints_successful"] for item in audit["edit_samples"])
        )

    def test_missing_endpoint_is_rejected(self):
        self.report["samples"].pop()
        with self.assertRaises(PairKernelAuditError):
            build_pair_execution_audit(self.manifest, self.report)

    def test_extra_and_duplicate_report_samples_are_rejected(self):
        extra = result("sf_extra", "sv_extra", "continuous")
        self.report["samples"].append(extra)
        with self.assertRaises(PairKernelAuditError):
            build_pair_execution_audit(self.manifest, self.report)
        self.report["samples"].pop()
        self.report["samples"].append(dict(self.report["samples"][0]))
        with self.assertRaises(PairKernelAuditError):
            build_pair_execution_audit(self.manifest, self.report)

    def test_each_endpoint_role_and_encoding_is_required(self):
        cases = {
            "base_continuous": "sv_ac",
            "base_quantized": "sv_aq",
            "edited_continuous": "sv_bc",
            "edited_quantized": "sv_bq",
        }
        for name, sample_id in cases.items():
            with self.subTest(case=name):
                report = {
                    **self.report,
                    "samples": [
                        item
                        for item in self.report["samples"]
                        if item["sample_id"] != sample_id
                    ],
                }
                with self.assertRaises(PairKernelAuditError):
                    build_pair_execution_audit(self.manifest, report)

    def test_duplicate_and_incomplete_edit_families_are_rejected(self):
        duplicate = dict(self.manifest["samples"][0])
        malformed = {
            **self.manifest,
            "samples": self.manifest["samples"] + [duplicate],
        }
        with self.assertRaises(PairKernelAuditError):
            build_pair_execution_audit(malformed, self.report)
        incomplete = {
            **self.manifest,
            "samples": self.manifest["samples"][:1],
        }
        with self.assertRaises(PairKernelAuditError):
            build_pair_execution_audit(incomplete, self.report)

    def test_wrong_result_source_and_encoding_metadata_are_rejected(self):
        for field, value in (
            ("source_family_id", "sf_wrong"),
            ("geometry_encoding", "quantized"),
        ):
            with self.subTest(field=field):
                report = {
                    **self.report,
                    "samples": [dict(item) for item in self.report["samples"]],
                }
                report["samples"][0][field] = value
                with self.assertRaises(PairKernelAuditError):
                    build_pair_execution_audit(self.manifest, report)

    def test_encoding_geometry_mismatch_is_surfaced(self):
        self.report["samples"][1]["final_metrics"]["volume"] = 1.5
        audit = build_pair_execution_audit(self.manifest, self.report)
        agreement = {
            item["source_family_id"]: item
            for item in audit["endpoint_encoding_agreement"]
        }
        self.assertFalse(agreement["sf_a"]["geometric_agreement"])
