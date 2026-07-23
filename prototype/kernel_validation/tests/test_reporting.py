from __future__ import annotations

from dataclasses import replace
import tempfile
from pathlib import Path
import unittest

from prototype.kernel_validation.executor import execute_sample
from prototype.kernel_validation.model import ShapeMetrics
from prototype.kernel_validation.reporting import aggregate_results, build_report, canonical_json, publish_report

from prototype.kernel_validation.tests.fakes import FakeAdapter, make_sample


class ReportingTests(unittest.TestCase):
    def _pair(self):
        continuous = execute_sample(make_sample("E", "continuous"), FakeAdapter())
        quantized = execute_sample(make_sample("E", "quantized"), FakeAdapter())
        self.assertEqual(continuous.source_family_id, quantized.source_family_id)
        return continuous, quantized

    def test_matching_pair_agrees(self):
        aggregate = aggregate_results(self._pair())
        paired = aggregate["paired"]
        self.assertEqual(paired["paired_status_agreement_count"], 1)
        self.assertEqual(paired["paired_geometric_agreement_count"], 1)

    def test_matching_status_but_volume_mismatch_is_reported(self):
        continuous, quantized = self._pair()
        changed = replace(quantized.final_metrics, volume=11.0)
        aggregate = aggregate_results((continuous, replace(quantized, final_metrics=changed)))
        self.assertEqual(
            aggregate["paired"]["geometric_disagreements"][0]["differing_metrics"], ["volume"]
        )

    def test_matching_status_but_bounding_box_mismatch_is_reported(self):
        continuous, quantized = self._pair()
        changed = replace(quantized.final_metrics, bounding_box=(0, 0, 0, 2, 1, 1))
        aggregate = aggregate_results((continuous, replace(quantized, final_metrics=changed)))
        self.assertIn("bounding_box", aggregate["paired"]["geometric_disagreements"][0]["differing_metrics"])

    def test_operation_progress_counts_unreached_operations(self):
        failed = execute_sample(make_sample("EE"), FakeAdapter(fail_method="extrude"))
        progress = aggregate_results((failed,))["operation_progress_by_type"]["extrude"]
        self.assertEqual(progress, {"scheduled": 2, "reached": 1, "kernel_completed": 0, "semantically_effective": 0, "failed": 1})

    def test_canonical_report_and_atomic_publication(self):
        results = self._pair()
        report = build_report(results, FakeAdapter().backend_versions(), {"configuration_sha256": "abc"})
        self.assertEqual(canonical_json(report), canonical_json(report))
        self.assertNotIn("timestamp", canonical_json(report))
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "report"
            publish_report(target, report)
            self.assertTrue((target / "execution_report.json").is_file())
            with self.assertRaises(Exception):
                publish_report(target, report)

    def test_failed_publication_leaves_no_final_or_temporary_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            target = parent / "report"
            with self.assertRaises(ValueError):
                publish_report(target, {"nonfinite": float("nan")})
            self.assertFalse(target.exists())
            self.assertEqual(list(parent.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
