from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import unittest
from unittest.mock import patch

from prototype.controlled_data.config import ConfigurationError
from prototype.controlled_data.generate import build_parser, main


class CLITests(unittest.TestCase):
    def test_num_source_families_argument_is_required(self):
        args = build_parser().parse_args(
            ["--output-dir", "/tmp/example", "--seed", "4", "--num-source-families", "60"]
        )
        self.assertEqual(args.num_source_families, 60)
        self.assertEqual(args.seed, 4)

    def test_ambiguous_num_samples_argument_is_not_accepted(self):
        error = StringIO()
        with redirect_stderr(error), self.assertRaises(SystemExit) as context:
            build_parser().parse_args(
                [
                    "--output-dir",
                    "/tmp/example",
                    "--num-source-families",
                    "60",
                    "--num-samples",
                    "60",
                ]
            )
        self.assertEqual(context.exception.code, 2)
        self.assertIn("unrecognized arguments: --num-samples 60", error.getvalue())

    def test_runtime_configuration_error_has_exit_one_and_clear_message(self):
        error = StringIO()
        with patch(
            "prototype.controlled_data.generate.generate_corpus",
            side_effect=ConfigurationError("bad configuration"),
        ), redirect_stderr(error):
            code = main(
                ["--output-dir", "/tmp/example", "--num-source-families", "60"]
            )
        self.assertEqual(code, 1)
        self.assertEqual(
            error.getvalue(),
            "controlled-data generation failed: bad configuration\n",
        )

    def test_successful_main_has_exit_zero_and_reports_exact_counts(self):
        output = StringIO()
        manifest = {"total_source_family_count": 60, "total_sample_variant_count": 120}
        with patch(
            "prototype.controlled_data.generate.generate_corpus", return_value=manifest
        ), redirect_stdout(output):
            code = main(
                ["--output-dir", "/tmp/example", "--num-source-families", "60"]
            )
        self.assertEqual(code, 0)
        self.assertEqual(
            output.getvalue(),
            "generated 60 source families and 120 sample variants\n",
        )
