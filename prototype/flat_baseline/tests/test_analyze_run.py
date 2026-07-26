"""Focused tests for quiescent B0 run analysis."""

from __future__ import annotations

from dataclasses import replace
import csv
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from prototype.flat_baseline.config import FlatBaselineConfig
from prototype.flat_baseline.run_analysis import (
    RunAnalysisError,
    _degradation_warnings,
    _diagnostic_warnings,
    _dominance_warnings,
    _erratic_warnings,
    _nonimproving_warnings,
    analyze_run,
    write_analysis,
)
from prototype.flat_baseline.training_config import LOSS_METRICS, TrainingConfig


class _PickleTorch:
    calls = []

    @classmethod
    def load(cls, path, map_location=None):
        cls.calls.append((path, map_location))
        with Path(path).open("rb") as stream:
            return pickle.load(stream)


class AnalyzeRunTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        _PickleTorch.calls = []

    def test_valid_complete_run_and_atomic_outputs(self):
        run = self._complete_run()
        summary = analyze_run(run, torch_module=_PickleTorch)
        self.assertEqual(summary["analysis_schema_version"], 1)
        self.assertEqual(summary["run_status"], "complete")
        self.assertEqual(summary["best"]["epoch"], 2)
        self.assertEqual(summary["final"]["epoch"], 3)
        self.assertTrue(summary["checkpoints"]["best"]["logged"])
        self.assertTrue(summary["checkpoints"]["best"]["considered"])
        self.assertTrue(summary["checkpoints"]["last"]["logged"])
        self.assertTrue(summary["checkpoints"]["last"]["considered"])
        self.assertEqual(
            {call[1] for call in _PickleTorch.calls}, {"cpu"}
        )

        output = self.root / "analysis"
        published = write_analysis(
            run, output, torch_module=_PickleTorch
        )
        self.assertEqual(published, summary)
        self.assertEqual(
            json.loads((output / "summary.json").read_text()),
            summary,
        )
        rows = (output / "metrics.csv").read_text().splitlines()
        self.assertEqual(len(rows), 4)
        self.assertIn("gap_total", rows[0])
        with self.assertRaises(RunAnalysisError) as captured:
            write_analysis(run, output, torch_module=_PickleTorch)
        self.assertEqual(captured.exception.code, "analysis_output_exists")

        failed_output = self.root / "failed_analysis"
        with mock.patch(
            "prototype.flat_baseline.run_analysis._write_csv",
            side_effect=OSError("synthetic write failure"),
        ):
            with self.assertRaises(OSError):
                write_analysis(
                    run, failed_output, torch_module=_PickleTorch
                )
        self.assertFalse(failed_output.exists())
        self.assertFalse(
            any(
                item.name.startswith(".failed_analysis.tmp-")
                for item in self.root.iterdir()
            )
        )

    def test_valid_incomplete_prefix(self):
        run = self._new_run(planned_epochs=3)
        records = [self._metadata(run, planned_epochs=3)]
        records.append(self._epoch("train", 1, 1, 3.0))
        self._write_records(run, records)
        summary = analyze_run(run, torch_module=_PickleTorch)
        self.assertEqual(summary["run_status"], "incomplete")
        self.assertEqual(summary["latest_train_epoch"], 1)
        self.assertIsNone(summary["latest_validation_epoch"])

    def test_terminal_structured_failure(self):
        run = self._new_run(planned_epochs=3)
        records = [
            self._metadata(run, planned_epochs=3),
            self._epoch("train", 1, 1, 3.0),
            {
                "event": "failure",
                "failure_type": "TrainingError",
                "failure_code": "nonfinite_loss",
                "detail": "synthetic",
            },
        ]
        self._write_records(run, records)
        summary = analyze_run(run, torch_module=_PickleTorch)
        self.assertEqual(summary["run_status"], "incomplete")
        self.assertEqual(
            summary["terminal_failure"]["failure_code"], "nonfinite_loss"
        )

    def test_records_after_failure_are_rejected(self):
        run = self._new_run()
        records = [
            self._metadata(run),
            {
                "event": "failure",
                "failure_type": "TrainingError",
                "failure_code": "synthetic",
                "detail": "failed",
            },
            self._epoch("train", 1, 1, 1.0),
        ]
        self._write_records(run, records)
        self._assert_error(run, "nonterminal_failure")

    def test_malformed_nonfinite_and_noncanonical_jsonl(self):
        cases = (
            ("malformed", "{\n", "malformed_jsonl"),
            (
                "nonfinite",
                '{"event":"train_epoch","value":NaN}\n',
                "nonfinite_json_constant",
            ),
            (
                "noncanonical",
                '{"event": "run_metadata"}\n',
                "noncanonical_jsonl",
            ),
            ("blank", "\n", "blank_jsonl_line"),
        )
        for name, payload, code in cases:
            with self.subTest(name=name):
                run = self._new_run(name=name)
                (run / "metrics.jsonl").write_text(payload)
                self._assert_error(run, code)

    def test_duplicate_keys_nonfinite_constants_unknown_events_and_fields(self):
        payload_cases = (
            (
                "duplicate_key",
                '{"event":"run_metadata","event":"run_metadata"}\n',
                "duplicate_json_key",
            ),
            (
                "positive_infinity",
                '{"event":"train_epoch","value":Infinity}\n',
                "nonfinite_json_constant",
            ),
            (
                "negative_infinity",
                '{"event":"train_epoch","value":-Infinity}\n',
                "nonfinite_json_constant",
            ),
            (
                "unknown_event",
                '{"event":"future_event"}\n',
                "unknown_event",
            ),
        )
        for name, payload, code in payload_cases:
            with self.subTest(name=name):
                run = self._new_run(name=name)
                (run / "metrics.jsonl").write_text(payload)
                self._assert_error(run, code)

        run = self._new_run(name="unknown_field", planned_epochs=2)
        epoch = self._epoch("train", 1, 1, 2.0)
        epoch["future_field"] = "preserved warning"
        self._write_records(
            run, [self._metadata(run, planned_epochs=2), epoch]
        )
        summary = analyze_run(run, torch_module=_PickleTorch)
        warning = next(
            item
            for item in summary["warnings"]
            if item["code"] == "unknown_fields"
        )
        self.assertEqual(warning["evidence"]["fields"], ["future_field"])

    def test_duplicate_missing_and_out_of_order_epochs(self):
        cases = (
            ("duplicate", (1, 1), "nonincreasing_epoch"),
            ("missing", (1, 3), "missing_train_epoch"),
            ("out_of_order", (2, 1), "nonincreasing_epoch"),
        )
        for name, epochs, code in cases:
            with self.subTest(name=name):
                run = self._new_run(name=name, planned_epochs=3)
                records = [self._metadata(run, planned_epochs=3)]
                records.extend(
                    self._epoch("train", epoch, index + 1, 2.0)
                    for index, epoch in enumerate(epochs)
                )
                self._write_records(run, records)
                self._assert_error(run, code)

    def test_validation_without_train_is_rejected(self):
        run = self._new_run()
        self._write_records(
            run,
            [
                self._metadata(run),
                self._epoch("validation", 1, 1, 1.0),
            ],
        )
        self._assert_error(run, "validation_without_train")

    def test_missing_and_inconsistent_checkpoints(self):
        run = self._new_run(name="missing")
        records = [
            self._metadata(run),
            self._epoch("train", 1, 1, 2.0),
            self._epoch("validation", 1, 1, 2.0),
            self._checkpoint_event("best", 1, 1, 2.0),
        ]
        self._write_records(run, records)
        self._assert_error(run, "missing_logged_checkpoint")

        inconsistent = self._new_run(name="inconsistent")
        config = self._training_config(inconsistent)
        records = [
            self._metadata(inconsistent),
            self._epoch("train", 1, 1, 2.0),
            self._epoch("validation", 1, 1, 2.0),
            self._checkpoint_event("best", 1, 1, 2.0),
        ]
        self._checkpoint(
            inconsistent,
            "best",
            1,
            1,
            3.0,
            config,
        )
        self._write_records(inconsistent, records)
        self._assert_error(inconsistent, "checkpoint_event_mismatch")

    def test_strict_tie_preserves_earlier_best(self):
        run = self._new_run(planned_epochs=3)
        config = self._training_config(run, epochs=3)
        records = [self._metadata(run, planned_epochs=3)]
        for epoch, value in ((1, 3.0), (2, 2.0), (3, 2.0)):
            records.append(self._epoch("train", epoch, epoch, value + 1.0))
            records.append(self._epoch("validation", epoch, epoch, value))
            if epoch in (1, 2):
                records.append(
                    self._checkpoint_event("best", epoch, epoch, value)
                )
            records.append(
                self._checkpoint_event("last", epoch, epoch, min(3.0, value))
            )
        self._checkpoint(run, "best", 2, 2, 2.0, config)
        self._checkpoint(run, "last", 3, 3, 2.0, config)
        self._write_records(run, records)
        summary = analyze_run(run, torch_module=_PickleTorch)
        self.assertEqual(summary["best"]["epoch"], 2)

    def test_multiple_compatible_and_incompatible_metadata(self):
        run = self._new_run(planned_epochs=2)
        first = self._metadata(run, planned_epochs=1)
        second = self._metadata(run, planned_epochs=2, resumed=True)
        records = [
            first,
            self._epoch("train", 1, 1, 2.0),
            self._epoch("validation", 1, 1, 2.0),
            second,
            self._epoch("train", 2, 2, 1.0),
        ]
        self._write_records(run, records)
        summary = analyze_run(run, torch_module=_PickleTorch)
        self.assertEqual(summary["run_status"], "incomplete")

        second["model_config"] = {
            **second["model_config"],
            "model_dim": 64,
        }
        self._write_records(run, records)
        self._assert_error(run, "incompatible_metadata")

    def test_resume_metadata_types_and_state_are_validated(self):
        cases = (
            ("not_resumed_string", False, "last.pt"),
            ("resumed_null", True, None),
            ("resumed_boolean", True, True),
            ("resumed_integer", True, 7),
            ("resumed_empty", True, ""),
        )
        for name, resumed, checkpoint in cases:
            with self.subTest(name=name):
                run = self._new_run(name=name, planned_epochs=2)
                metadata = self._metadata(
                    run, planned_epochs=2, resumed=resumed
                )
                metadata["resume_checkpoint"] = checkpoint
                self._write_records(run, [metadata])
                self._assert_error(run, "invalid_metadata")

    def test_same_epoch_global_steps_must_match(self):
        validation_run = self._new_run(
            name="validation_step", planned_epochs=2
        )
        self._write_records(
            validation_run,
            [
                self._metadata(validation_run, planned_epochs=2),
                self._epoch("train", 1, 1, 2.0),
                self._epoch("validation", 1, 2, 2.0),
            ],
        )
        self._assert_error(
            validation_run, "inconsistent_epoch_global_step"
        )

        for kind in ("best", "last"):
            with self.subTest(kind=kind):
                run = self._new_run(
                    name="{}_event_step".format(kind), planned_epochs=2
                )
                records = [
                    self._metadata(run, planned_epochs=2),
                    self._epoch("train", 1, 1, 2.0),
                ]
                if kind == "best":
                    records.append(self._epoch("validation", 1, 1, 2.0))
                records.append(
                    self._checkpoint_event(kind, 1, 2, 2.0)
                )
                self._write_records(run, records)
                self._assert_error(
                    run, "inconsistent_epoch_global_step"
                )

        payload_run = self._new_run(
            name="payload_step", planned_epochs=2
        )
        config = self._training_config(payload_run, epochs=2)
        self._write_records(
            payload_run,
            [
                self._metadata(payload_run, planned_epochs=2),
                self._epoch("train", 1, 1, 2.0),
                self._checkpoint_event("last", 1, 1, None),
            ],
        )
        self._checkpoint(
            payload_run, "last", 1, 2, None, config
        )
        self._assert_error(payload_run, "checkpoint_event_mismatch")

    def test_activity_chronology_and_epoch_limits(self):
        activity_first = self._new_run(
            name="activity_first", planned_epochs=2
        )
        self._write_records(
            activity_first,
            [
                self._epoch("train", 1, 1, 2.0),
                self._metadata(activity_first, planned_epochs=2),
            ],
        )
        self._assert_error(activity_first, "activity_before_metadata")

        validation_first = self._new_run(
            name="validation_first", planned_epochs=2
        )
        self._write_records(
            validation_first,
            [
                self._metadata(validation_first, planned_epochs=2),
                self._epoch("validation", 1, 1, 2.0),
                self._epoch("train", 1, 1, 2.0),
            ],
        )
        self._assert_error(validation_first, "activity_before_train")

        checkpoint_first = self._new_run(
            name="checkpoint_first", planned_epochs=2
        )
        self._write_records(
            checkpoint_first,
            [
                self._metadata(checkpoint_first, planned_epochs=2),
                self._checkpoint_event("last", 1, 1, None),
                self._epoch("train", 1, 1, 2.0),
            ],
        )
        self._assert_error(checkpoint_first, "activity_before_train")

        beyond = self._new_run(name="beyond_plan", planned_epochs=2)
        self._write_records(
            beyond,
            [
                self._metadata(beyond, planned_epochs=2),
                self._epoch("train", 1, 1, 2.0),
                self._epoch("validation", 1, 1, 2.0),
                self._epoch("train", 2, 2, 2.0),
                self._epoch("validation", 2, 2, 2.0),
                self._epoch("train", 3, 3, 2.0),
            ],
        )
        self._assert_error(beyond, "epoch_exceeds_plan")

        increased = self._new_run(
            name="increased_plan", planned_epochs=2
        )
        self._write_records(
            increased,
            [
                self._metadata(increased, planned_epochs=1),
                self._epoch("train", 1, 1, 2.0),
                self._epoch("validation", 1, 1, 2.0),
                self._metadata(
                    increased, planned_epochs=2, resumed=True
                ),
                self._epoch("train", 2, 2, 2.0),
            ],
        )
        self.assertEqual(
            analyze_run(increased, torch_module=_PickleTorch)["run_status"],
            "incomplete",
        )

        decreased = self._new_run(
            name="decreased_plan", planned_epochs=2
        )
        self._write_records(
            decreased,
            [
                self._metadata(decreased, planned_epochs=2),
                self._metadata(
                    decreased, planned_epochs=1, resumed=True
                ),
            ],
        )
        self._assert_error(decreased, "incompatible_metadata")

    def test_checkpoint_paths_are_contained(self):
        external_root = self.root / "external"
        external_root.mkdir()

        logged = self._new_run(name="logged_external", planned_epochs=2)
        logged_config = self._training_config(logged, epochs=2)
        self._checkpoint(
            external_root, "best", 1, 1, 2.0, logged_config
        )
        (logged / "best.pt").symlink_to(external_root / "best.pt")
        self._write_records(
            logged,
            [
                self._metadata(logged, planned_epochs=2),
                self._epoch("train", 1, 1, 2.0),
                self._epoch("validation", 1, 1, 2.0),
                self._checkpoint_event("best", 1, 1, 2.0),
            ],
        )
        self._assert_error(logged, "unsafe_checkpoint_path")
        self.assertEqual(_PickleTorch.calls, [])

        _PickleTorch.calls = []
        automatic = self._new_run(
            name="automatic_external", planned_epochs=2
        )
        automatic_config = self._training_config(automatic, epochs=2)
        self._checkpoint(
            external_root, "last", 1, 1, None, automatic_config
        )
        (automatic / "last.pt").symlink_to(external_root / "last.pt")
        self._write_records(
            automatic,
            [
                self._metadata(automatic, planned_epochs=2),
                self._epoch("train", 1, 1, 2.0),
            ],
        )
        self._assert_error(automatic, "unsafe_checkpoint_path")
        self.assertEqual(_PickleTorch.calls, [])

        for name, relative in (
            ("absolute_path", str(external_root / "best.pt")),
            ("parent_path", "../best.pt"),
        ):
            with self.subTest(name=name):
                run = self._new_run(name=name, planned_epochs=2)
                event = self._checkpoint_event("best", 1, 1, 2.0)
                event["relative_path"] = relative
                self._write_records(
                    run,
                    [
                        self._metadata(run, planned_epochs=2),
                        self._epoch("train", 1, 1, 2.0),
                        self._epoch("validation", 1, 1, 2.0),
                        event,
                    ],
                )
                self._assert_error(run, "unsafe_checkpoint_path")

        contained = self._complete_run(name="contained_symlink")
        target = contained / "contained-best-target.pt"
        (contained / "best.pt").replace(target)
        (contained / "best.pt").symlink_to(target)
        summary = analyze_run(contained, torch_module=_PickleTorch)
        self.assertEqual(summary["run_status"], "complete")

    def test_unlogged_checkpoints_and_relocated_preserved_best(self):
        unlogged = self._new_run(name="unlogged", planned_epochs=1)
        config = self._training_config(unlogged)
        self._write_records(
            unlogged,
            [
                self._metadata(unlogged),
                self._epoch("train", 1, 1, 2.0),
                self._epoch("validation", 1, 1, 2.0),
            ],
        )
        self._checkpoint(unlogged, "best", 1, 1, 2.0, config)
        self._checkpoint(unlogged, "last", 1, 1, 2.0, config)
        summary = analyze_run(unlogged, torch_module=_PickleTorch)
        self.assertEqual(summary["run_status"], "incomplete")
        self.assertFalse(summary["checkpoints"]["best"]["considered"])
        self.assertFalse(summary["checkpoints"]["last"]["considered"])

        relocated = self._new_run(name="relocated", planned_epochs=3)
        relocated_config = self._training_config(relocated, epochs=3)
        records = [
            self._metadata(
                relocated, planned_epochs=3, resumed=True
            ),
            self._epoch("train", 3, 3, 2.0),
            self._checkpoint_event("last", 3, 3, 1.5),
        ]
        self._write_records(relocated, records)
        self._checkpoint(
            relocated, "best", 2, 2, 1.5, relocated_config
        )
        self._checkpoint(
            relocated, "last", 3, 3, 1.5, relocated_config
        )
        relocated_summary = analyze_run(
            relocated, torch_module=_PickleTorch
        )
        self.assertTrue(
            relocated_summary["checkpoints"]["best"][
                "preserved_relocated_best"
            ]
        )
        self.assertTrue(
            relocated_summary["checkpoints"]["best"]["considered"]
        )
        self.assertIsNone(
            relocated_summary["best"]["validation_metrics"]
        )
        self.assertIsNotNone(
            relocated_summary["best"]["limitation_reason"]
        )

        unrelated = self._new_run(name="unrelated", planned_epochs=2)
        self._write_records(
            unrelated,
            [
                self._metadata(unrelated, planned_epochs=2),
                self._epoch("train", 1, 1, 2.0),
            ],
        )
        (unrelated / "unrelated.pt").write_bytes(b"ignored")
        unrelated_summary = analyze_run(
            unrelated, torch_module=_PickleTorch
        )
        self.assertEqual(unrelated_summary["run_status"], "incomplete")

    def test_codebook_domains_boundaries_and_vq_statistics(self):
        valid = self._new_run(name="vq_valid", planned_epochs=3)
        records = [self._metadata(valid, planned_epochs=3)]
        for epoch, active, perplexity in (
            (1, 0, 0.0),
            (2, 1, 1.0),
            (3, 4, 4.0),
        ):
            records.append(
                self._epoch(
                    "train",
                    epoch,
                    epoch,
                    2.0,
                    active=active,
                    perplexity=perplexity,
                )
            )
            records.append(
                self._epoch(
                    "validation",
                    epoch,
                    epoch,
                    2.0,
                    active=active,
                    perplexity=perplexity,
                )
            )
        self._write_records(valid, records)
        summary = analyze_run(valid, torch_module=_PickleTorch)
        statistics = summary["vq"]["statistics"]["train"]
        self.assertEqual(
            statistics["active_code_count"],
            {"minimum": 0, "maximum": 4, "final": 4},
        )
        self.assertEqual(
            statistics["codebook_perplexity"],
            {"minimum": 0.0, "maximum": 4.0, "final": 4.0},
        )
        self.assertEqual(
            statistics["codebook_utilization"],
            {"minimum": 0.0, "maximum": 0.25, "final": 0.25},
        )
        self.assertEqual(
            summary["vq"]["statistics"]["validation"], statistics
        )

        full_active = self._new_run(
            name="full_active_tolerance", planned_epochs=2
        )
        self._write_records(
            full_active,
            [
                self._metadata(full_active, planned_epochs=2),
                self._epoch(
                    "train",
                    1,
                    1,
                    2.0,
                    active=16,
                    perplexity=16.000001,
                ),
            ],
        )
        full_active_summary = analyze_run(
            full_active, torch_module=_PickleTorch
        )
        self.assertEqual(
            full_active_summary["vq"]["train"][0][
                "codebook_perplexity"
            ],
            16.000001,
        )

        invalid_cases = (
            ("zero_with_perplexity", 0, 1.0),
            ("positive_with_zero", 1, 0.0),
            ("below_one", 2, 0.9),
            ("above_active", 2, 2.1),
            ("boolean_active", True, 1.0),
        )
        for name, active, perplexity in invalid_cases:
            with self.subTest(name=name):
                run = self._new_run(name=name, planned_epochs=2)
                self._write_records(
                    run,
                    [
                        self._metadata(run, planned_epochs=2),
                        self._epoch(
                            "train",
                            1,
                            1,
                            2.0,
                            active=active,
                            perplexity=perplexity,
                        ),
                    ],
                )
                expected = (
                    "invalid_integer"
                    if name == "boolean_active"
                    else "inconsistent_codebook"
                )
                self._assert_error(run, expected)

    def test_example_counts_learning_rate_boolean_fields_gaps_and_csv(self):
        mismatch_cases = (
            ("train_count", "train", "number_of_examples", 2,
             "inconsistent_example_count"),
            ("validation_count", "validation", "number_of_examples", 2,
             "inconsistent_example_count"),
            ("learning_rate", "train", "learning_rate", 0.1,
             "inconsistent_learning_rate"),
            ("boolean_epoch", "train", "epoch", True, "invalid_integer"),
            ("boolean_learning_rate", "train", "learning_rate", True,
             "nonfinite_or_invalid_number"),
        )
        for name, mode, field, value, code in mismatch_cases:
            with self.subTest(name=name):
                run = self._new_run(name=name, planned_epochs=2)
                records = [self._metadata(run, planned_epochs=2)]
                if mode == "validation":
                    records.append(self._epoch("train", 1, 1, 2.0))
                epoch = self._epoch(mode, 1, 1, 2.0)
                epoch[field] = value
                records.append(epoch)
                self._write_records(run, records)
                self._assert_error(run, code)

        complete = self._complete_run(name="csv_values")
        output = self.root / "csv_analysis"
        summary = write_analysis(
            complete, output, torch_module=_PickleTorch
        )
        self.assertEqual(summary["final"]["gaps"]["total"], -0.5)
        with (output / "metrics.csv").open(
            encoding="utf-8", newline=""
        ) as stream:
            reader = csv.DictReader(stream)
            rows = list(reader)
            expected_fields = [
                "epoch",
                "global_step",
                "learning_rate",
                "train_examples",
                "validation_examples",
            ]
            for prefix in ("train", "validation", "gap"):
                expected_fields.extend(
                    "{}_{}".format(prefix, name)
                    for name in LOSS_METRICS
                )
            for prefix in ("train", "validation"):
                expected_fields.extend(
                    "{}_{}".format(prefix, name)
                    for name in (
                        "active_code_count",
                        "codebook_utilization",
                        "codebook_perplexity",
                    )
                )
            self.assertEqual(reader.fieldnames, expected_fields)
        self.assertEqual([row["epoch"] for row in rows], ["1", "2", "3"])
        self.assertEqual(float(rows[-1]["train_total"]), 3.0)
        self.assertEqual(float(rows[-1]["validation_total"]), 2.5)
        self.assertEqual(float(rows[-1]["gap_total"]), -0.5)
        first_bytes = (output / "metrics.csv").read_bytes()
        self.assertTrue(first_bytes.endswith(b"\n"))
        self.assertNotIn(b"\r\n", first_bytes)
        second_output = self.root / "csv_analysis_second"
        write_analysis(
            complete, second_output, torch_module=_PickleTorch
        )
        self.assertEqual(
            first_bytes, (second_output / "metrics.csv").read_bytes()
        )

    def test_warning_boundaries_and_weighted_component_contribution(self):
        run = self._new_run(name="warnings", planned_epochs=14)
        model = replace(
            FlatBaselineConfig(),
            codebook_size=32,
            node_type_loss_weight=2.0,
            categorical_loss_weight=0.0,
            geometry_loss_weight=0.0,
            edge_presence_loss_weight=0.0,
            edge_type_loss_weight=0.0,
            operation_pointer_loss_weight=0.0,
            vq_loss_weight=0.0,
        )
        config = self._training_config(run, epochs=14)
        records = [
            self._metadata(
                run,
                planned_epochs=14,
                model_config=model,
            )
        ]
        best_value = None
        best_epoch = None
        for epoch in range(1, 15):
            train_total = 4.0 - 0.1 * epoch
            validation_total = 4.0 + 0.1 * epoch
            active = 4 if epoch >= 6 else 16
            perplexity = 1.4 if epoch >= 6 else 8.0
            records.append(
                self._epoch(
                    "train",
                    epoch,
                    epoch,
                    train_total,
                    active=active,
                    perplexity=perplexity,
                    model_config=model,
                    dominant_metric="node_type",
                )
            )
            records.append(
                self._epoch(
                    "validation",
                    epoch,
                    epoch,
                    validation_total,
                    active=active,
                    perplexity=perplexity,
                    model_config=model,
                    dominant_metric="node_type",
                )
            )
            if best_value is None or validation_total < best_value:
                best_value = validation_total
                best_epoch = epoch
                records.append(
                    self._checkpoint_event(
                        "best", epoch, epoch, best_value
                    )
                )
            records.append(
                self._checkpoint_event(
                    "last", epoch, epoch, best_value
                )
            )
        self._checkpoint(
            run, "best", best_epoch, best_epoch, best_value, config, model
        )
        self._checkpoint(
            run, "last", 14, 14, best_value, config, model
        )
        self._write_records(run, records)
        summary = analyze_run(run, torch_module=_PickleTorch)
        codes = {item["code"] for item in summary["warnings"]}
        self.assertIn("active_code_collapse", codes)
        self.assertIn("substantial_codebook_underuse", codes)
        self.assertIn("perplexity_near_one", codes)
        self.assertIn("sustained_validation_degradation", codes)
        self.assertIn("component_dominance", codes)
        dominance = next(
            item
            for item in summary["warnings"]
            if item["code"] == "component_dominance"
        )
        self.assertEqual(dominance["evidence"]["weight"], 2.0)
        self.assertEqual(
            dominance["evidence"]["contribution_ratios"],
            [1.0, 1.0, 1.0],
        )

    def test_independent_warning_boundaries(self):
        model = replace(FlatBaselineConfig(), codebook_size=32)
        best = {"epoch": None}

        def train_records(utilization, perplexity, start_epoch=1):
            return [
                {
                    "epoch": epoch,
                    "metrics": {
                        name: (1.0 if name == "total" else 1.0 / 7.0)
                        for name in LOSS_METRICS
                    },
                    "codebook": {
                        "active_code_count": int(round(32 * utilization)),
                        "codebook_utilization": utilization,
                        "codebook_perplexity": perplexity,
                    },
                }
                for epoch in range(start_epoch, start_epoch + 5)
            ]

        collapse = _diagnostic_warnings(
            train_records(0.125, 1.5, 6), [], model, best, "incomplete"
        )
        self.assertEqual(
            next(
                item for item in collapse
                if item["code"] == "active_code_collapse"
            )["affected_epochs"],
            [6, 7, 8, 9, 10],
        )
        above_collapse = _diagnostic_warnings(
            train_records(0.125001, 1.5001, 6),
            [],
            model,
            best,
            "incomplete",
        )
        self.assertNotIn(
            "active_code_collapse",
            {item["code"] for item in above_collapse},
        )

        at_half = _diagnostic_warnings(
            train_records(0.5, 2.0, 10), [], model, best, "incomplete"
        )
        self.assertNotIn(
            "substantial_codebook_underuse",
            {item["code"] for item in at_half},
        )
        below_half = _diagnostic_warnings(
            train_records(0.499, 2.0, 10), [], model, best, "incomplete"
        )
        underuse = next(
            item
            for item in below_half
            if item["code"] == "substantial_codebook_underuse"
        )
        self.assertEqual(
            underuse["code"], "substantial_codebook_underuse"
        )
        self.assertEqual(
            underuse["threshold"],
            "utilization < 0.50 for 5 consecutive train epochs at epoch 10 or later",
        )
        self.assertEqual(underuse["affected_epochs"], [10, 11, 12, 13, 14])
        self.assertEqual(
            underuse["evidence"],
            {"codebook_utilization": [0.499] * 5},
        )
        self.assertTrue(underuse["explanation"])

        at_perplexity = _diagnostic_warnings(
            train_records(0.5, 1.5), [], model, best, "incomplete"
        )
        perplexity_warning = next(
            item
            for item in at_perplexity
            if item["code"] == "perplexity_near_one"
        )
        self.assertEqual(
            perplexity_warning["code"], "perplexity_near_one"
        )
        self.assertEqual(
            perplexity_warning["threshold"],
            "perplexity <= 1.5 for 5 consecutive train epochs",
        )
        self.assertEqual(
            perplexity_warning["affected_epochs"], [1, 2, 3, 4, 5]
        )
        self.assertEqual(
            perplexity_warning["evidence"],
            {"codebook_perplexity": [1.5] * 5},
        )
        self.assertTrue(perplexity_warning["explanation"])
        above_perplexity = _diagnostic_warnings(
            train_records(0.5, 1.5001), [], model, best, "incomplete"
        )
        self.assertNotIn(
            "perplexity_near_one",
            {item["code"] for item in above_perplexity},
        )

    def test_loss_warning_boundaries_and_true_even_median(self):
        def metric_record(epoch, total, component):
            metrics = {
                name: 0.0 for name in LOSS_METRICS
            }
            metrics["total"] = total
            metrics["node_type"] = component
            return {
                "epoch": epoch,
                "metrics": metrics,
                "codebook": {
                    "active_code_count": 1,
                    "codebook_utilization": 1.0 / 16.0,
                    "codebook_perplexity": 1.0,
                },
            }

        model = FlatBaselineConfig()
        at_dominance = [
            metric_record(epoch, 1.0, 0.70) for epoch in range(1, 4)
        ]
        below_dominance = [
            metric_record(epoch, 1.0, 0.699999) for epoch in range(1, 4)
        ]
        zero_total = [
            metric_record(epoch, 0.0, 0.0) for epoch in range(1, 4)
        ]
        self.assertEqual(
            _dominance_warnings(at_dominance, [], model)[0]["code"],
            "component_dominance",
        )
        self.assertEqual(
            _dominance_warnings(below_dominance, [], model), []
        )
        self.assertEqual(_dominance_warnings(zero_total, [], model), [])

        paired_at = []
        paired_below = []
        for index in range(5):
            train = metric_record(index + 1, 2.0 - 0.1 * index, 0.0)
            validation_at = metric_record(
                index + 1, 1.0 + 0.0125 * index, 0.0
            )
            validation_below = metric_record(
                index + 1, 1.0 + 0.01249 * index, 0.0
            )
            paired_at.append((train, validation_at))
            paired_below.append((train, validation_below))
        degradation = _degradation_warnings(paired_at)[0]
        self.assertEqual(
            degradation["code"], "sustained_validation_degradation"
        )
        self.assertEqual(
            degradation["threshold"],
            "5 paired epochs and at least 5% validation increase",
        )
        self.assertEqual(degradation["affected_epochs"], [1, 2, 3, 4, 5])
        self.assertEqual(
            degradation["evidence"]["train_total"],
            [2.0, 1.9, 1.8, 1.7, 1.6],
        )
        self.assertEqual(
            degradation["evidence"]["validation_total"],
            [1.0, 1.0125, 1.025, 1.0375, 1.05],
        )
        self.assertAlmostEqual(
            degradation["evidence"]["relative_validation_increase"],
            0.05,
        )
        self.assertTrue(degradation["explanation"])
        self.assertEqual(_degradation_warnings(paired_below), [])

        nonimproving = [
            metric_record(index + 1, value, 0.0)
            for index, value in enumerate(
                [1.0] * 5 + [0.990001] * 5
            )
        ]
        exact_improvement = [
            metric_record(index + 1, value, 0.0)
            for index, value in enumerate([1.0] * 5 + [0.99] * 5)
        ]
        self.assertEqual(
            _nonimproving_warnings(nonimproving)[0]["code"],
            "non_improving_validation",
        )
        self.assertEqual(_nonimproving_warnings(exact_improvement), [])

        alternating = [
            metric_record(
                epoch,
                1.0 if epoch % 2 else 1.0101,
                0.0,
            )
            for epoch in range(1, 9)
        ]
        erratic = _erratic_warnings(alternating)
        self.assertEqual(erratic[0]["code"], "highly_erratic_validation")
        self.assertEqual(
            erratic[0]["evidence"]["minimum_delta_magnitude"],
            0.0100505,
        )

    def test_negative_total_is_rejected_before_dominance_analysis(self):
        run = self._new_run(name="negative_total", planned_epochs=2)
        epoch = self._epoch("train", 1, 1, 2.0)
        epoch["metrics"]["total"] = -1.0
        self._write_records(
            run, [self._metadata(run, planned_epochs=2), epoch]
        )
        self._assert_error(run, "invalid_number_range")

    def test_best_checkpoint_early_boundary(self):
        model = FlatBaselineConfig()
        train = [
            {
                "epoch": epoch,
                "metrics": {
                    name: (1.0 if name == "total" else 1.0 / 7.0)
                    for name in LOSS_METRICS
                },
                "codebook": {
                    "active_code_count": 16,
                    "codebook_utilization": 1.0,
                    "codebook_perplexity": 16.0,
                },
            }
            for epoch in range(1, 41)
        ]
        at_boundary = _diagnostic_warnings(
            train, [], model, {"epoch": 30}, "complete"
        )
        below_boundary = _diagnostic_warnings(
            train, [], model, {"epoch": 31}, "complete"
        )
        early = next(
            item
            for item in at_boundary
            if item["code"] == "best_checkpoint_early"
        )
        self.assertEqual(
            early["threshold"],
            "final_epoch - best_epoch >= 10",
        )
        self.assertEqual(early["affected_epochs"], [30, 40])
        self.assertEqual(
            early["evidence"],
            {
                "best_epoch": 30,
                "final_epoch": 40,
                "epoch_difference": 10,
            },
        )
        self.assertTrue(early["explanation"])
        self.assertNotIn(
            "best_checkpoint_early",
            {item["code"] for item in below_boundary},
        )

    def test_cli_exit_codes_atomic_output_and_no_traceback(self):
        run = self._new_run(name="cli", planned_epochs=2)
        self._write_records(
            run,
            [
                self._metadata(run, planned_epochs=2),
                self._epoch("train", 1, 1, 2.0),
            ],
        )
        output = self.root / "cli_analysis"
        result = self._run_cli(run, output)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, "")
        self.assertIn("B0 run analysis: incomplete", result.stdout)
        self.assertTrue((output / "summary.json").is_file())
        self.assertTrue((output / "metrics.csv").is_file())

        invalid = self._new_run(name="cli_invalid")
        (invalid / "metrics.jsonl").write_text("{\n")
        invalid_output = self.root / "invalid_analysis"
        result = self._run_cli(invalid, invalid_output)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stdout, "")
        self.assertNotIn("Traceback", result.stderr)
        error = json.loads(result.stderr)["error"]
        self.assertEqual(error["code"], "malformed_jsonl")
        self.assertEqual(error["run_status"], "invalid")
        self.assertFalse(invalid_output.exists())

    def _complete_run(self, name="run"):
        run = self._new_run(name=name, planned_epochs=3)
        config = self._training_config(run, epochs=3)
        records = [self._metadata(run, planned_epochs=3)]
        best_value = None
        best_epoch = None
        for epoch, validation_value in ((1, 3.0), (2, 2.0), (3, 2.5)):
            records.append(
                self._epoch("train", epoch, epoch, validation_value + 0.5)
            )
            records.append(
                self._epoch("validation", epoch, epoch, validation_value)
            )
            if best_value is None or validation_value < best_value:
                best_value = validation_value
                best_epoch = epoch
                records.append(
                    self._checkpoint_event(
                        "best", epoch, epoch, best_value
                    )
                )
            records.append(
                self._checkpoint_event("last", epoch, epoch, best_value)
            )
        self._checkpoint(
            run, "best", best_epoch, best_epoch, best_value, config
        )
        self._checkpoint(run, "last", 3, 3, best_value, config)
        self._write_records(run, records)
        return run

    def _new_run(self, name="run", planned_epochs=1):
        path = self.root / name
        path.mkdir()
        return path

    def _training_config(self, run, epochs=1):
        return TrainingConfig(
            seed=0,
            epochs=epochs,
            batch_size=2,
            learning_rate=1e-3,
            weight_decay=0.0,
            gradient_clip_norm=1.0,
            validation_interval=1,
            checkpoint_interval=1,
            dataloader_workers=0,
            device="cpu",
            output_dir=str(run),
            checkpoint_selection_metric="total",
        )

    def _metadata(
        self,
        run,
        planned_epochs=1,
        resumed=False,
        model_config=None,
    ):
        model = model_config or FlatBaselineConfig()
        training = self._training_config(run, epochs=planned_epochs)
        return {
            "event": "run_metadata",
            "model_config": model.to_dict(),
            "training_config": training.to_dict(),
            "git_commit": "a" * 40,
            "git_dirty": False,
            "git_status_porcelain": [],
            "source_tree_sha256": "b" * 64,
            "python_version": "3.8.13",
            "pytorch_version": "1.11.0",
            "resolved_device": "cpu",
            "split_manifest": "iid",
            "train_partition": "train",
            "validation_partition": "validation",
            "train_family_ids": ["train-family"],
            "validation_family_ids": ["validation-family"],
            "resumed": resumed,
            "resume_checkpoint": "last.pt" if resumed else None,
            "determinism_note": "synthetic",
        }

    def _epoch(
        self,
        mode,
        epoch,
        step,
        total,
        active=16,
        perplexity=8.0,
        model_config=None,
        dominant_metric=None,
    ):
        model = model_config or FlatBaselineConfig()
        components = {}
        if dominant_metric is None:
            weights = [
                float(getattr(model, field))
                for field in (
                    "node_type_loss_weight",
                    "categorical_loss_weight",
                    "geometry_loss_weight",
                    "edge_presence_loss_weight",
                    "edge_type_loss_weight",
                    "operation_pointer_loss_weight",
                    "vq_loss_weight",
                )
            ]
            value = total / sum(weights)
            components = {
                name: value
                for name in LOSS_METRICS
                if name != "total"
            }
        else:
            components = {
                name: 0.0 for name in LOSS_METRICS if name != "total"
            }
            weight_name = {
                "node_type": "node_type_loss_weight",
                "categorical_attributes": "categorical_loss_weight",
                "geometry": "geometry_loss_weight",
                "edge_presence": "edge_presence_loss_weight",
                "edge_type": "edge_type_loss_weight",
                "operation_pointer": "operation_pointer_loss_weight",
                "vq_commitment": "vq_loss_weight",
            }[dominant_metric]
            components[dominant_metric] = total / float(
                getattr(model, weight_name)
            )
        return {
            "event": mode + "_epoch",
            "epoch": epoch,
            "global_step": step,
            "learning_rate": 1e-3,
            "number_of_examples": 1,
            "metrics": {"total": total, **components},
            "codebook": {
                "active_code_count": active,
                "codebook_utilization": active / model.codebook_size,
                "codebook_perplexity": perplexity,
            },
        }

    def _checkpoint_event(self, kind, epoch, step, best):
        return {
            "event": "checkpoint",
            "checkpoint_kind": kind,
            "epoch": epoch,
            "global_step": step,
            "best_validation_metric": best,
            "relative_path": kind + ".pt",
        }

    def _checkpoint(
        self,
        run,
        kind,
        epoch,
        step,
        best,
        training_config,
        model_config=None,
    ):
        model = model_config or FlatBaselineConfig()
        payload = {
            "checkpoint_version": 1,
            "checkpoint_kind": kind,
            "model_state": {},
            "optimizer_state": {},
            "epoch": epoch,
            "global_step": step,
            "model_config": model.to_dict(),
            "training_config": training_config.to_dict(),
            "best_validation_metric": best,
            "rng_state": {},
            "data_state": {
                "split_manifest": "iid",
                "train_partition": "train",
                "validation_partition": "validation",
                "train_family_ids": ["train-family"],
                "validation_family_ids": ["validation-family"],
            },
        }
        with (run / (kind + ".pt")).open("wb") as stream:
            pickle.dump(payload, stream)

    def _write_records(self, run, records):
        payload = "".join(
            json.dumps(
                item,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
            for item in records
        )
        (run / "metrics.jsonl").write_text(payload, encoding="utf-8")

    def _assert_error(self, run, code):
        with self.assertRaises(RunAnalysisError) as captured:
            analyze_run(run, torch_module=_PickleTorch)
        self.assertEqual(captured.exception.code, code)

    def _run_cli(self, run, output):
        environment = dict(os.environ)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        return subprocess.run(
            [
                sys.executable,
                "-m",
                "prototype.flat_baseline.analyze_run",
                "--run-dir",
                str(run),
                "--output-dir",
                str(output),
            ],
            cwd=str(Path(__file__).resolve().parents[3]),
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )


if __name__ == "__main__":
    unittest.main()
