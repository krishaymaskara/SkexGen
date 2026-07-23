from __future__ import annotations

from dataclasses import replace
import math
import unittest

from prototype.kernel_validation.executor import execute_sample

from prototype.kernel_validation.tests.fakes import FakeAdapter, make_sample


class ExecutorTests(unittest.TestCase):
    def test_successful_extrude(self):
        result = execute_sample(make_sample(), FakeAdapter())
        self.assertEqual(result.kernel_status, "success")
        self.assertTrue(result.final_shape_valid)
        self.assertEqual(result.final_metrics.volume, 10.0)
        self.assertTrue(result.operation_results[0].semantically_effective)

    def test_all_templates_encodings_primitives_and_planes_reach_adapter(self):
        for template in ("E", "R", "EE", "ER", "RE", "RR"):
            for encoding in ("continuous", "quantized"):
                for primitive, plane in (
                    ("rectangle_lines", "XY"),
                    ("circle", "XZ"),
                    ("capsule_line_arc", "YZ"),
                ):
                    with self.subTest(
                        template=template, encoding=encoding, primitive=primitive, plane=plane
                    ):
                        result = execute_sample(
                            make_sample(
                                template,
                                encoding,
                                primitive_family=primitive,
                                reference_plane=plane,
                            ),
                            FakeAdapter(),
                        )
                        self.assertEqual(result.kernel_status, "success", result.error_detail)

    def test_no_op_join_is_semantic_failure(self):
        result = execute_sample(make_sample("EE", later_mode="join"), FakeAdapter(no_effect_join=True))
        self.assertEqual(result.failure_category, "boolean_no_effect")
        self.assertEqual(result.failure_stage, "semantic_effect")
        self.assertEqual(result.failed_operation_index, 1)
        self.assertTrue(result.operation_results[1].kernel_completed)
        self.assertFalse(result.operation_results[1].semantically_effective)

    def test_no_op_cut_is_semantic_failure(self):
        result = execute_sample(make_sample("EE", later_mode="cut"), FakeAdapter(no_effect_cut=True))
        self.assertEqual(result.failure_category, "boolean_no_effect")
        self.assertEqual(result.failure_stage, "semantic_effect")

    def test_zero_and_nonfinite_volume_fail(self):
        for volume in (0.0, float("nan"), float("inf")):
            with self.subTest(volume=volume):
                result = execute_sample(make_sample(), FakeAdapter(feature_volumes=(volume,)))
                self.assertEqual(result.failure_category, "invalid_final_shape")
                self.assertEqual(result.failure_stage, "solid_validation")
                self.assertTrue(result.operation_results[0].kernel_completed)

    def test_scheduled_and_reached_distinguish_early_failure(self):
        result = execute_sample(
            make_sample("EE"),
            FakeAdapter(fail_method="extrude", fail_exception=RuntimeError("construction stopped")),
        )
        self.assertEqual(len(result.operation_results), 2)
        self.assertTrue(result.operation_results[0].scheduled)
        self.assertTrue(result.operation_results[0].reached)
        self.assertTrue(result.operation_results[0].failed)
        self.assertTrue(result.operation_results[1].scheduled)
        self.assertFalse(result.operation_results[1].reached)

    def test_partial_body_is_not_reported_as_failed_history_final_shape(self):
        class FailSecondExtrude(FakeAdapter):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def extrude(self, face, vector):
                self.calls += 1
                if self.calls == 2:
                    raise RuntimeError("second feature failed")
                return super().extrude(face, vector)

        result = execute_sample(make_sample("EE"), FailSecondExtrude())
        self.assertEqual(result.failed_operation_index, 1)
        self.assertIsNone(result.final_metrics)
        self.assertFalse(result.final_shape_valid)

    def test_authoritative_identity_mismatch_is_rejected(self):
        result = execute_sample(replace(make_sample(), source_family_id="sf_wrong"), FakeAdapter())
        self.assertEqual(result.failure_category, "schema_or_parse_failure")
        self.assertEqual(result.failure_stage, "corpus_loading")
        self.assertEqual(len(result.operation_results), 1)
        self.assertTrue(result.operation_results[0].scheduled)
        self.assertFalse(result.operation_results[0].reached)
        result = execute_sample(replace(make_sample(), sample_id="sv_wrong"), FakeAdapter())
        self.assertEqual(result.failure_category, "schema_or_parse_failure")
        result = execute_sample(replace(make_sample(), primitive_family="circle"), FakeAdapter())
        self.assertEqual(result.failure_stage, "corpus_loading")

    def test_failure_details_are_sanitized_and_deterministic(self):
        sample = make_sample()
        adapter = FakeAdapter(
            fail_method="make_wire",
            fail_exception=RuntimeError("bad\n path /private/tmp/random/run.step at 0xABC123"),
        )
        first = execute_sample(sample, adapter)
        second = execute_sample(sample, FakeAdapter(fail_method="make_wire", fail_exception=RuntimeError("bad\n path /private/tmp/random/run.step at 0x999999")))
        self.assertEqual(first.error_detail, second.error_detail)
        self.assertEqual(first.failure_stage, "wire_construction")
        self.assertNotIn("/private", first.error_detail)
        self.assertNotIn("0x", first.error_detail)
        self.assertNotIn("\n", first.error_detail)


if __name__ == "__main__":
    unittest.main()
