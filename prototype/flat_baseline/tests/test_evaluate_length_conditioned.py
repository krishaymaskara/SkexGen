"""Non-PyTorch tests for deterministic paired evaluation publication."""

from __future__ import annotations

import csv
import ctypes
from dataclasses import replace
import errno
import hashlib
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest import mock

from prototype.flat_baseline.evaluate_length_conditioned import (
    EVALUATION_SCHEMA_VERSION,
    EvaluationError,
    FAILURE_COLUMNS,
    LEGACY_METRICS_COLUMNS,
    LIMITATION,
    METRICS_COLUMNS,
    RAW_ARTIFACT,
    REPAIRED_CHECKPOINT_EPOCH,
    REPAIRED_CHECKPOINT_GLOBAL_STEP,
    REPAIRED_CHECKPOINT_SHA256,
    REPAIRED_CHECKPOINT_TRAINING_COMMIT,
    REPAIRED_EVALUATION_SCHEMA_VERSION,
    _atomic_no_replace,
    _configure_repaired_runtime,
    _failure_counters,
    _identifier_sha256,
    _json_document,
    _json_lines,
    _linux_rename_noreplace,
    _resolve_publication_directory,
    _resolve_publication_file,
    _run_metadata,
    _semantic_difference,
    _strict_load_model,
    _validate_authoritative_ids,
    checkpoint_sha256,
    completed_exit_code,
    corpus_identity,
    evaluate_records,
    load_and_validate_checkpoint,
    latent_usage,
    make_artifacts,
    parse_arguments,
    publish_artifacts,
    publish_json_report,
    run,
    select_examples,
    validate_artifact_directory,
)
from prototype.flat_baseline.tests.test_conversion import _raw_from_target
from prototype.flat_baseline.config import FlatBaselineConfig
from prototype.flat_baseline.training_config import TrainingConfig
from prototype.flat_baseline.verify_evaluation_artifacts import (
    _predicted_history_operation_coverage,
    _workflow_evidence_hashes,
    complete_report,
    compare_publications,
    inspect_publication,
    main as verifier_main,
    repaired_full_report,
)
from prototype.controlled_data.builders import build_history
from prototype.controlled_data.identity import source_family_id
from prototype.model_data.loader import load_physical_examples
from prototype.model_data.tests.fixtures import source, write_physical_corpus
from prototype.representation.model import GeometryEncoding


class FakeCFunction:
    def __init__(self, result=0, error_number=0):
        self.result = result
        self.error_number = error_number
        self.calls = []
        self.argtypes = None
        self.restype = None

    def __call__(self, *arguments):
        self.calls.append(arguments)
        ctypes.set_errno(self.error_number)
        return self.result


class FakeTensor:
    def __init__(self, value):
        self.value = value

    def contiguous(self):
        return self

    def to(self, device):
        return self


class FakeTorch:
    long = "long"
    float32 = "float32"
    bool = "bool"

    def __init__(self, checkpoint, events, mutate_on_load=False):
        self.checkpoint = checkpoint
        self.events = events
        self.mutate_on_load = mutate_on_load
        self.cuda = SimpleNamespace(is_available=lambda: False)

    def load(self, path, map_location):
        content = Path(path).read_bytes()
        self.events.append(("checkpoint_load", content, map_location))
        if self.mutate_on_load:
            Path(path).write_bytes(b"mutated-after-hash")
        return self.checkpoint

    def tensor(self, value, dtype):
        return FakeTensor(value)

    @staticmethod
    def device(request):
        return request


class FakeModel:
    def __init__(self, config, events):
        self.config = config
        self.events = events

    def to(self, device):
        self.events.append(("model_to", device))
        return self

    def load_state_dict(self, state, strict):
        self.events.append(("strict_load", state, strict))


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        corpus = Path(self.temporary.name) / "corpus"
        write_physical_corpus(
            corpus,
            (source("E"), source("R"), source("ER"), source("RR")),
        )
        self.examples = tuple(load_physical_examples(corpus))

    def tearDown(self):
        self.temporary.cleanup()

    def arguments(self, *extra):
        output = Path(self.temporary.name) / "output"
        values = (
            "--corpus-dir", "corpus",
            "--checkpoint", "best.pt",
            "--split-manifest", "iid",
            "--partition", "validation",
            "--output-dir", str(output),
            "--batch-size", "2",
            "--device", "cpu",
        )
        return parse_arguments(values + extra)

    def selected(self):
        partitions = tuple(
            replace(item, partition="validation") for item in self.examples
        )
        return select_examples(partitions, "validation")

    @staticmethod
    def decode(items):
        predictions = tuple(_raw_from_target(item.target) for item in items)
        return SimpleNamespace(
            latent_indices=tuple(item.latent_indices for item in predictions),
            teacher_forced=predictions,
            predicted_history=predictions,
        )

    def records(self, batch_size=2, decode=None):
        selected = self.selected()
        records, raw = evaluate_records(
            selected,
            batch_size=batch_size,
            max_operations=2,
            decode_batch=decode or self.decode,
        )
        return selected, records, raw

    def repeated_failure_artifacts(
        self, *, distinct_primary=False, include_edge=False
    ):
        selected = self.selected()[:1]
        raw = _raw_from_target(selected[0].target)
        nodes = list(raw.raw_nodes)
        for index in (0, 1):
            geometry = list(nodes[index].normalized_geometry)
            geometry[0] = float("nan")
            nodes[index] = replace(
                nodes[index], normalized_geometry=tuple(geometry)
            )
        raw = replace(raw, raw_nodes=tuple(nodes))
        if distinct_primary:
            raw = replace(
                raw,
                raw_operation_pointers=(
                    (object(),) + raw.raw_operation_pointers[1:]
                ),
            )
        if include_edge:
            raw = replace(
                raw,
                raw_edges=(
                    replace(raw.raw_edges[0], presence_logit=float("inf")),
                ) + raw.raw_edges[1:],
            )

        def decode(items):
            return SimpleNamespace(
                latent_indices=((0, 1),),
                teacher_forced=(raw,),
                predicted_history=(raw,),
            )

        records, raw_records = evaluate_records(
            selected,
            batch_size=1,
            max_operations=2,
            decode_batch=decode,
        )
        artifacts = make_artifacts(
            records,
            raw_records,
            self.metadata(selected),
            selected,
            False,
        )
        return selected, records, artifacts

    def run_fixture(
        self,
        name,
        *,
        transform=None,
        checkpoint_ids=None,
        family_limit=None,
        event_sink=None,
        corrupt_corpus=False,
        mutate_checkpoint_on_load=False,
    ):
        root = Path(self.temporary.name) / ("run-corpus-" + name)
        physical = (
            source("E"),
            source("R"),
            source("ER"),
            source("RR"),
            source("EE"),
        )
        identifiers = tuple(
            source_family_id(build_history(item, GeometryEncoding.CONTINUOUS))
            for item in physical
        )
        partitions = {
            identifier: (
                "test" if index == len(identifiers) - 1 else "validation"
            )
            for index, identifier in enumerate(identifiers)
        }
        write_physical_corpus(root, physical, partitions=partitions)
        if corrupt_corpus:
            manifest_path = root / "corpus_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["configuration_sha256"] = "0" * 64
            manifest_path.write_text(json.dumps(manifest))
        loaded = load_physical_examples(root)
        validation = tuple(sorted(
            item.physical_family_id
            for item in loaded
            if item.partition == "validation"
        ))
        selected = tuple(
            item for item in loaded if item.partition == "validation"
        )
        selected = tuple(sorted(
            selected, key=lambda item: item.physical_family_id
        ))
        if family_limit is not None:
            selected = selected[:family_limit]
        events = [] if event_sink is None else event_sink
        checkpoint_path = Path(self.temporary.name) / (name + ".pt")
        checkpoint_bytes = ("checkpoint-" + name).encode()
        checkpoint_path.write_bytes(checkpoint_bytes)
        payload = {
            "checkpoint_version": 1,
            "checkpoint_kind": "best",
            "model_state": {"weight": "opaque"},
            "optimizer_state": {},
            "epoch": 3,
            "global_step": 9,
            "model_config": FlatBaselineConfig().to_dict(),
            "training_config": TrainingConfig().to_dict(),
            "best_validation_metric": 0.5,
            "rng_state": {},
            "data_state": {
                "split_manifest": "iid",
                "validation_partition": "validation",
                "validation_family_ids": (
                    list(validation)
                    if checkpoint_ids is None
                    else list(checkpoint_ids)
                ),
            },
        }
        fake_torch = FakeTorch(
            payload, events, mutate_on_load=mutate_checkpoint_on_load
        )
        offset = [0]

        def paired_decoder(
            model,
            authoritative_batch,
            *,
            node_counts,
            node_count_source,
        ):
            inputs, target = authoritative_batch
            batch_count = len(node_counts)
            self.assertEqual(
                len(inputs["categorical_ids"].value), batch_count
            )
            self.assertEqual(len(target["node_mask"].value), batch_count)
            self.assertEqual(node_count_source, "target_canonical_metadata")
            current = selected[offset[0] : offset[0] + batch_count]
            self.assertEqual(
                tuple(len(item.target.node_type_ids) for item in current),
                tuple(node_counts),
            )
            predictions = []
            latent_rows = []
            for local_index, item in enumerate(current):
                raw = _raw_from_target(item.target)
                latent_rows.append(raw.latent_indices)
                if transform is not None:
                    raw = transform(raw, offset[0] + local_index)
                predictions.append(raw)
            offset[0] += batch_count
            events.append(("paired_decode", tuple(node_counts)))
            return SimpleNamespace(
                latent_indices=tuple(latent_rows),
                teacher_forced=tuple(predictions),
                predicted_history=tuple(predictions),
            )

        output = Path(self.temporary.name) / ("run-output-" + name)
        arguments = [
            "--corpus-dir", str(root),
            "--checkpoint", str(checkpoint_path),
            "--split-manifest", "iid",
            "--partition", "validation",
            "--output-dir", str(output),
            "--batch-size", "2",
            "--device", "cpu",
            "--write-raw-predictions",
        ]
        if family_limit is not None:
            arguments.extend(("--family-limit", str(family_limit)))
        parsed = parse_arguments(arguments)
        status = run(
            parsed,
            fake_torch,
            model_factory=lambda config: FakeModel(config, events),
            paired_decoder=paired_decoder,
            source_state_provider=lambda repository: {
                "git_commit": "c" * 40,
                "git_dirty": False,
                "source_tree_sha256": "d" * 64,
            },
        )
        return status, output, events, selected, checkpoint_bytes

    def metadata(self, selected, raw=False, repaired=True):
        identifiers = [item.physical_family_id for item in selected]
        metadata = {
            "evaluation_schema_version": (
                REPAIRED_EVALUATION_SCHEMA_VERSION
                if repaired else EVALUATION_SCHEMA_VERSION
            ),
            "representation_schema_version": 1,
            "partition": "validation",
            "repository_commit": "a" * 40,
            "checkpoint_sha256": "b" * 64,
            "checkpoint_data_state": {
                "validation_family_ids": identifiers,
            },
            "authoritative_validation_family_ids": identifiers,
            "authoritative_validation_family_ids_sha256": (
                _identifier_sha256(tuple(identifiers))
            ),
            "selected_family_ids": identifiers,
            "selected_family_count": len(selected),
            "family_limit": None,
            "model_configuration": {
                "codebook_size": 16,
                "max_operations": 2,
            },
            "raw_predictions_published": raw,
            "test_partition_evaluated": False,
            "limitation": LIMITATION,
        }
        if repaired:
            metadata["repaired_evaluation_contract"] = True
        return metadata

    @staticmethod
    def unsupported_rename_error(error_number=errno.EINVAL):
        return EvaluationError(
            "unsupported_no_replace",
            "injected unsupported renameat2",
            error_number=error_number,
            publication_backend="renameat2_syscall",
            platform_machine="x86_64",
        )

    def publish_with_symlink(self, output, artifacts):
        with mock.patch.object(sys, "platform", "linux"), mock.patch(
            "prototype.flat_baseline.evaluate_length_conditioned."
            "_linux_rename_noreplace",
            side_effect=self.unsupported_rename_error(),
        ):
            publish_artifacts(output, artifacts)

    @staticmethod
    def failure_rows(artifacts, path="teacher_forced"):
        rows = tuple(csv.DictReader(io.StringIO(
            artifacts["conversion_failures.csv"].decode()
        )))
        return tuple(item for item in rows if item["path"] == path)

    @staticmethod
    def aggregate_failure_count(artifacts, designation, code):
        summary = json.loads(artifacts["summary.json"])
        values = summary["teacher_forced"]["overall"]["validity"][
            designation + "_failure_counts"
        ]
        return next(item["count"] for item in values if item["code"] == code)

    def test_complete_cli_and_validation(self):
        parsed = self.arguments("--family-limit", "2", "--write-raw-predictions")
        self.assertEqual(parsed.partition, "validation")
        self.assertEqual(parsed.family_limit, 2)
        self.assertTrue(parsed.write_raw_predictions)
        invalid = (
            ("--partition", "other"),
            ("--batch-size", "0"),
            ("--family-limit", "-1"),
            ("--device", "mps"),
            ("--operation-count", "2"),
        )
        for option, value in invalid:
            base = list((
                "--corpus-dir", "c", "--checkpoint", "p",
                "--split-manifest", "iid", "--partition", "validation",
                "--output-dir", "o", "--batch-size", "2", "--device", "cpu",
            ))
            if option in base:
                base[base.index(option) + 1] = value
            else:
                base.extend((option, value))
            with self.subTest(option=option, value=value):
                with self.assertRaises(EvaluationError):
                    parse_arguments(tuple(base))
        with self.assertRaises(EvaluationError):
            parse_arguments(())

    def test_test_partition_requires_explicit_opt_in(self):
        values = list((
            "--corpus-dir", "c", "--checkpoint", "p",
            "--split-manifest", "iid", "--partition", "test",
            "--output-dir", "o", "--batch-size", "2", "--device", "cpu",
        ))
        with self.assertRaisesRegex(EvaluationError, "test_evaluation_forbidden"):
            parse_arguments(values)
        values.append("--allow-test-evaluation")
        self.assertEqual(parse_arguments(values).partition, "test")

    def test_repaired_smoke_cli_freezes_execution_configuration(self):
        parsed = self.arguments(
            "--family-limit", "6",
            "--batch-size", "3",
            "--write-raw-predictions",
            "--repaired-smoke-contract",
        )
        self.assertTrue(parsed.repaired_smoke_contract)
        invalid = (
            ("--family-limit", "5"),
            ("--batch-size", "2"),
            ("--device", "cuda"),
        )
        for replacement in invalid:
            with self.subTest(replacement=replacement):
                values = list((
                    "--corpus-dir", "corpus",
                    "--checkpoint", "best.pt",
                    "--split-manifest", "iid",
                    "--partition", "validation",
                    "--output-dir", "output",
                    "--family-limit", "6",
                    "--batch-size", "3",
                    "--device", "cpu",
                    "--write-raw-predictions",
                    "--repaired-smoke-contract",
                ))
                name, value = replacement
                values[values.index(name) + 1] = value
                with self.assertRaisesRegex(
                    EvaluationError, "repaired_smoke_configuration"
                ):
                    parse_arguments(values)

    def test_repaired_full_cli_requires_exact_validation_contract(self):
        parsed = self.arguments(
            "--batch-size", "32",
            "--write-raw-predictions",
            "--repaired-full-contract",
        )
        self.assertTrue(parsed.repaired_full_contract)
        self.assertIsNone(parsed.family_limit)
        invalid = (
            ("--family-limit", "68"),
            ("--batch-size", "31"),
            ("--device", "cuda"),
        )
        for option, value in invalid:
            with self.subTest(option=option, value=value):
                arguments = list((
                    "--corpus-dir", "corpus",
                    "--checkpoint", "best.pt",
                    "--split-manifest", "iid",
                    "--partition", "validation",
                    "--output-dir", "output",
                    "--batch-size", "32",
                    "--device", "cpu",
                    "--write-raw-predictions",
                    "--repaired-full-contract",
                ))
                if option == "--family-limit":
                    arguments.extend((option, value))
                else:
                    arguments[arguments.index(option) + 1] = value
                with self.assertRaisesRegex(
                    EvaluationError, "repaired_full_configuration"
                ):
                    parse_arguments(arguments)
        with self.assertRaisesRegex(
            EvaluationError, "repaired_contract_conflict"
        ):
            self.arguments(
                "--family-limit", "6",
                "--batch-size", "3",
                "--write-raw-predictions",
                "--repaired-smoke-contract",
                "--repaired-full-contract",
            )

    def test_repaired_runtime_sets_frozen_seed_and_one_torch_thread(self):
        events = []

        class Torch:
            threads = 4

            @classmethod
            def manual_seed(cls, seed):
                events.append(("seed", seed))

            @classmethod
            def set_num_threads(cls, count):
                events.append(("threads", count))
                cls.threads = count

            @classmethod
            def get_num_threads(cls):
                return cls.threads

        _configure_repaired_runtime(Torch, 2026)
        self.assertEqual(events, [("seed", 2026), ("threads", 1)])
        self.assertEqual(Torch.get_num_threads(), 1)

    def test_zero_inclusive_latent_usage_and_known_perplexity(self):
        records = (
            SimpleNamespace(latent_indices=(0, 0)),
            SimpleNamespace(latent_indices=(1, 1)),
        )
        usage = latent_usage(records, 32)
        self.assertEqual(len(usage["per_code_assignment_counts"]), 32)
        self.assertEqual(
            usage["per_code_assignment_counts"][:3], [2, 2, 0]
        )
        self.assertEqual(usage["per_code_assignment_counts"][2:], [0] * 30)
        self.assertEqual(usage["active_code_count"], 2)
        self.assertEqual(usage["codebook_utilization"], 2 / 32)
        self.assertAlmostEqual(usage["codebook_perplexity"], 2.0)
        self.assertTrue(usage["shared_by_decoding_paths"])

    def test_latent_usage_has_one_shared_headline_csv_row(self):
        selected, records, raw = self.records()
        artifacts = make_artifacts(
            records, raw, self.metadata(selected, True), selected, True
        )
        summary = json.loads(artifacts["summary.json"])
        self.assertEqual(
            summary["evaluation_schema_version"],
            REPAIRED_EVALUATION_SCHEMA_VERSION,
        )
        rows = tuple(csv.DictReader(io.StringIO(
            artifacts["metrics.csv"].decode()
        )))
        shared = tuple(row for row in rows if row["path"] == "shared_latent")
        self.assertEqual(len(shared), 1)
        self.assertEqual(
            int(shared[0]["active_code_count"]),
            summary["latent_usage"]["active_code_count"],
        )
        self.assertEqual(
            float(shared[0]["codebook_perplexity"]),
            summary["latent_usage"]["codebook_perplexity"],
        )
        self.assertEqual(
            len(summary["latent_usage"]["per_code_assignment_counts"]), 16
        )
        self.assertNotIn("latent_usage", summary["teacher_forced"])
        self.assertNotIn("latent_usage", summary["predicted_history"])

    def test_selection_is_authoritative_ordered_limited_and_nonempty(self):
        values = tuple(reversed(self.selected()))
        selected = select_examples(values, "validation", 2)
        self.assertEqual(
            tuple(item.physical_family_id for item in selected),
            tuple(sorted(item.physical_family_id for item in values))[:2],
        )
        duplicate = values + (values[0],)
        with self.assertRaisesRegex(EvaluationError, "duplicate_family_id"):
            select_examples(duplicate, "validation")
        with self.assertRaisesRegex(EvaluationError, "empty_selection"):
            select_examples(values, "train")

    def test_every_family_and_both_paths_are_evaluated(self):
        selected, records, _ = self.records()
        self.assertEqual(
            tuple(item.family_id for item in records),
            tuple(item.physical_family_id for item in selected),
        )
        for record in records:
            self.assertTrue(record.teacher_forced.metrics.raw_integrity.valid)
            self.assertTrue(record.predicted_history.metrics.raw_integrity.valid)

    def test_teacher_forced_provenance_is_adapted_without_changing_raw_evidence(self):
        selected = self.selected()

        def decode(items):
            predictions = tuple(
                replace(
                    _raw_from_target(item.target),
                    prefix_feedback="shifted_target_prefix",
                )
                for item in items
            )
            return SimpleNamespace(
                latent_indices=tuple(
                    item.latent_indices for item in predictions
                ),
                teacher_forced=predictions,
                predicted_history=tuple(
                    _raw_from_target(item.target) for item in items
                ),
            )

        records, raw_records = evaluate_records(
            selected,
            batch_size=2,
            max_operations=2,
            decode_batch=decode,
        )
        self.assertTrue(all(
            item.teacher_forced.conversion.raw_integrity.valid
            for item in records
        ))
        artifacts = make_artifacts(
            records,
            raw_records,
            self.metadata(selected, True),
            selected,
            True,
        )
        raw = json.loads(artifacts[RAW_ARTIFACT].splitlines()[0])
        self.assertEqual(
            raw["teacher_forced"]["prefix_feedback"],
            "shifted_target_prefix",
        )
        output = Path(self.temporary.name) / "teacher-provenance"
        publish_artifacts(
            output,
            artifacts,
            authoritative_examples=selected,
        )
        self.assertTrue(output.is_dir())

    def test_paired_and_raw_latent_indices_must_align(self):
        selected = self.selected()[:1]
        raw = _raw_from_target(selected[0].target)

        def decode(items):
            return SimpleNamespace(
                latent_indices=((15,),),
                teacher_forced=(raw,),
                predicted_history=(raw,),
            )

        records, raw_records = evaluate_records(
            selected,
            batch_size=1,
            max_operations=2,
            decode_batch=decode,
        )
        artifacts = make_artifacts(
            records,
            raw_records,
            self.metadata(selected, True),
            selected,
            True,
        )
        output = Path(self.temporary.name) / "latent-mismatch"
        with self.assertRaisesRegex(EvaluationError, "latent_index_mismatch"):
            publish_artifacts(
                output,
                artifacts,
                authoritative_examples=selected,
            )
        self.assertFalse(output.exists())

    def test_end_to_end_fake_run_publishes_complete_valid_result(self):
        status, output, events, selected, checkpoint_bytes = self.run_fixture(
            "valid", mutate_checkpoint_on_load=True
        )
        self.assertEqual(status, 0)
        self.assertTrue(output.is_dir())
        metadata = json.loads((output / "run_metadata.json").read_text())
        summary = json.loads((output / "summary.json").read_text())
        examples = tuple(
            json.loads(line)
            for line in (output / "examples.jsonl").read_text().splitlines()
        )
        self.assertEqual(
            metadata["selected_family_ids"],
            [item.physical_family_id for item in selected],
        )
        self.assertEqual(metadata["evaluation_schema_version"], 1)
        self.assertEqual(summary["evaluation_schema_version"], 1)
        self.assertNotIn("latent_usage", summary)
        self.assertNotIn("repaired_evaluation_contract", metadata)
        self.assertNotIn("repaired_smoke_contract", metadata)
        self.assertNotIn(
            "resolved_selected_family_ids_before_inference", metadata
        )
        self.assertNotIn("payload_access", metadata)
        self.assertEqual(
            (output / "metrics.csv").read_text().splitlines()[0].split(","),
            list(LEGACY_METRICS_COLUMNS),
        )
        self.assertEqual(len(examples), len(selected))
        self.assertFalse(metadata["test_partition_evaluated"])
        self.assertEqual(
            metadata["checkpoint_sha256"],
            hashlib.sha256(checkpoint_bytes).hexdigest(),
        )
        self.assertEqual(events[0], ("checkpoint_load", checkpoint_bytes, "cpu"))
        self.assertIn(("strict_load", {"weight": "opaque"}, True), events)
        self.assertEqual(
            sum(len(item[1]) for item in events if item[0] == "paired_decode"),
            len(selected),
        )

    def test_repaired_metadata_is_output_namespace_independent(self):
        selected, records, raw_records = self.records()
        base = {
            "corpus_dir": "corpus",
            "checkpoint": "/shared/repaired/best.pt",
            "split_manifest": "iid",
            "partition": "validation",
            "batch_size": 3,
            "device": "cpu",
            "family_limit": 6,
            "allow_test_evaluation": False,
            "write_raw_predictions": True,
            "repaired_smoke_contract": True,
        }
        checkpoint = {
            "checkpoint_kind": "best",
            "epoch": REPAIRED_CHECKPOINT_EPOCH,
            "global_step": REPAIRED_CHECKPOINT_GLOBAL_STEP,
            "data_state": {"validation_family_ids": [
                item.physical_family_id for item in selected
            ]},
        }
        source = lambda repository: {
            "git_commit": "c" * 40,
            "git_dirty": False,
            "source_tree_sha256": "d" * 64,
        }
        identifiers = tuple(
            item.physical_family_id for item in selected
        )
        artifacts = []
        with mock.patch(
            "prototype.flat_baseline.evaluate_length_conditioned._git_value",
            return_value="flat-mixed-baseline",
        ):
            for label in ("phase-b-smoke-a", "phase-b-smoke-b"):
                arguments = SimpleNamespace(
                    **dict(base, output_dir="/tmp/" + label)
                )
                metadata = _run_metadata(
                    arguments,
                    selected,
                    "b" * 64,
                    checkpoint,
                    FlatBaselineConfig(),
                    TrainingConfig(
                        output_dir="/tmp/training-" + label
                    ),
                    "cpu",
                    identifiers,
                    (),
                    {"train": 544, "validation": 68, "test": 68},
                    source,
                )
                artifacts.append(make_artifacts(
                    records, raw_records, metadata, selected, True
                ))
                self.assertNotIn(
                    "output_dir", metadata["training_configuration"]
                )
        self.assertEqual(artifacts[0], artifacts[1])
        hashes = [
            {
                name: hashlib.sha256(content).hexdigest()
                for name, content in bundle.items()
            }
            for bundle in artifacts
        ]
        self.assertEqual(hashes[0], hashes[1])
        for bundle in artifacts:
            combined = b"".join(bundle.values())
            self.assertNotIn(b"phase-b-smoke-a", combined)
            self.assertNotIn(b"phase-b-smoke-b", combined)
            self.assertNotIn(b"/tmp/", combined)

    def test_repaired_full_metadata_uses_final_access_and_branch_fallback(self):
        base = self.selected()[0]
        identifiers = tuple(
            "sf_{:064x}".format(index) for index in range(68)
        )
        selected = tuple(
            replace(
                base,
                physical_family_id=family_id,
                partition="validation",
            )
            for family_id in identifiers
        )
        records, raw_records = evaluate_records(
            selected,
            batch_size=32,
            max_operations=8,
            decode_batch=self.decode,
        )
        arguments = SimpleNamespace(
            checkpoint="/shared/repaired/best.pt",
            split_manifest="iid",
            partition="validation",
            family_limit=None,
            batch_size=32,
            device="cpu",
            write_raw_predictions=True,
            repaired_smoke_contract=False,
            repaired_full_contract=True,
        )
        checkpoint = {
            "checkpoint_kind": "best",
            "epoch": REPAIRED_CHECKPOINT_EPOCH,
            "global_step": REPAIRED_CHECKPOINT_GLOBAL_STEP,
            "data_state": {"validation_family_ids": list(identifiers)},
        }
        source = lambda repository: {
            "git_commit": "a" * 40,
            "git_dirty": False,
            "source_tree_sha256": "d" * 64,
        }
        preflight_access = {
            "train_family_records_loaded": 0,
            "validation_family_records_loaded": 0,
            "test_family_records_loaded": 0,
        }

        with mock.patch(
            "prototype.flat_baseline.evaluate_length_conditioned._git_value",
            return_value=None,
        ):
            failed_metadata = _run_metadata(
                arguments,
                selected,
                REPAIRED_CHECKPOINT_SHA256,
                checkpoint,
                FlatBaselineConfig(),
                TrainingConfig(seed=2026, vq_init="train-kmeans"),
                "cpu",
                identifiers,
                (),
                {"train": 544, "validation": 68, "test": 68},
                source,
            )
        self.assertEqual(
            failed_metadata["payload_access"],
            {
                "train_family_records_loaded": 0,
                "validation_family_records_loaded": 68,
                "test_family_records_loaded": 0,
                "loaded_family_ids": list(identifiers),
            },
        )
        failed_artifacts = make_artifacts(
            records, raw_records, failed_metadata, selected, True
        )
        with self.assertRaisesRegex(
            EvaluationError,
            "repository_branch expected='flat-mixed-baseline' actual=None",
        ):
            publish_artifacts(
                Path(self.temporary.name) / "full-missing-container-branch",
                failed_artifacts,
                expected_ids=identifiers,
                expected_partition="validation",
                authoritative_validation_ids=identifiers,
                authoritative_examples=selected,
                reviewed_commit="a" * 40,
                expected_checkpoint_sha256=REPAIRED_CHECKPOINT_SHA256,
            )

        probes = []

        def git_value(repository, *command):
            probes.append(command)
            if command == ("branch", "--show-current"):
                return None
            if command == ("symbolic-ref", "--short", "HEAD"):
                return "flat-mixed-baseline"
            self.fail("unexpected git probe {!r}".format(command))

        with mock.patch(
            "prototype.flat_baseline.evaluate_length_conditioned._git_value",
            side_effect=git_value,
        ):
            metadata = _run_metadata(
                arguments,
                selected,
                REPAIRED_CHECKPOINT_SHA256,
                checkpoint,
                FlatBaselineConfig(),
                TrainingConfig(seed=2026, vq_init="train-kmeans"),
                "cpu",
                identifiers,
                (),
                {"train": 544, "validation": 68, "test": 68},
                source,
            )
        self.assertEqual(
            probes,
            [
                ("branch", "--show-current"),
                ("symbolic-ref", "--short", "HEAD"),
            ],
        )
        self.assertEqual(
            preflight_access,
            {
                "train_family_records_loaded": 0,
                "validation_family_records_loaded": 0,
                "test_family_records_loaded": 0,
            },
        )
        self.assertEqual(
            metadata["payload_access"],
            {
                "train_family_records_loaded": 0,
                "validation_family_records_loaded": 68,
                "test_family_records_loaded": 0,
                "loaded_family_ids": list(identifiers),
            },
        )
        artifacts = make_artifacts(
            records, raw_records, metadata, selected, True
        )
        output = Path(self.temporary.name) / "full-final-access"
        publish_artifacts(
            output,
            artifacts,
            expected_ids=identifiers,
            expected_partition="validation",
            authoritative_validation_ids=identifiers,
            authoritative_examples=selected,
            reviewed_commit="a" * 40,
            expected_checkpoint_sha256=REPAIRED_CHECKPOINT_SHA256,
        )
        self.assertTrue(output.is_dir())

        damaged_metadata = dict(metadata)
        damaged_metadata["payload_access"] = dict(preflight_access)
        damaged_artifacts = make_artifacts(
            records, raw_records, damaged_metadata, selected, True
        )
        with self.assertRaisesRegex(
            EvaluationError,
            "payload_access expected=.*validation_family_records_loaded.*68",
        ):
            publish_artifacts(
                Path(self.temporary.name) / "full-preflight-access",
                damaged_artifacts,
                expected_ids=identifiers,
                expected_partition="validation",
                authoritative_validation_ids=identifiers,
                authoritative_examples=selected,
                reviewed_commit="a" * 40,
                expected_checkpoint_sha256=REPAIRED_CHECKPOINT_SHA256,
            )

    def test_repaired_resolved_selection_is_reconciled_everywhere(self):
        selected, records, raw_records = self.records()
        metadata = self.metadata(selected, raw=True)
        identifiers = [
            item.physical_family_id for item in selected
        ]
        metadata.update({
            "repaired_smoke_contract": True,
            "checkpoint_sha256": REPAIRED_CHECKPOINT_SHA256,
            "checkpoint_epoch": REPAIRED_CHECKPOINT_EPOCH,
            "checkpoint_global_step": REPAIRED_CHECKPOINT_GLOBAL_STEP,
            "checkpoint_training_source_commit": (
                REPAIRED_CHECKPOINT_TRAINING_COMMIT
            ),
            "authoritative_partition_family_counts": {
                "train": 544,
                "validation": 68,
                "test": 68,
            },
            "resolved_selected_family_ids_before_inference": identifiers,
            "family_limit": 6,
            "batch_size": 3,
            "requested_device": "cpu",
            "resolved_device": "cpu",
            "repository_branch": "flat-mixed-baseline",
            "source_dirty": False,
            "source_tree_sha256": "d" * 64,
            "payload_access": {
                "train_family_records_loaded": 0,
                "validation_family_records_loaded": len(identifiers),
                "test_family_records_loaded": 0,
                "loaded_family_ids": identifiers,
            },
        })
        artifacts = make_artifacts(
            records, raw_records, metadata, selected, True
        )
        damaged_metadata = json.loads(artifacts["run_metadata.json"])
        damaged_metadata[
            "resolved_selected_family_ids_before_inference"
        ] = list(reversed(identifiers))
        artifacts["run_metadata.json"] = _json_document(damaged_metadata)
        output = Path(self.temporary.name) / "bad-resolved-selection"
        with self.assertRaisesRegex(
            EvaluationError, "repaired_smoke_metadata"
        ):
            publish_artifacts(
                output,
                artifacts,
                expected_ids=identifiers,
                expected_partition="validation",
                authoritative_validation_ids=identifiers,
                authoritative_examples=selected,
                expected_checkpoint_sha256=REPAIRED_CHECKPOINT_SHA256,
            )
        self.assertFalse(output.exists())

    def test_repaired_publication_rejects_dirty_or_invalid_source(self):
        selected, records, raw_records = self.records()
        identifiers = [
            item.physical_family_id for item in selected
        ]
        metadata = self.metadata(selected, raw=True)
        metadata.update({
            "repaired_smoke_contract": True,
            "checkpoint_sha256": REPAIRED_CHECKPOINT_SHA256,
            "checkpoint_epoch": REPAIRED_CHECKPOINT_EPOCH,
            "checkpoint_global_step": REPAIRED_CHECKPOINT_GLOBAL_STEP,
            "checkpoint_training_source_commit": (
                REPAIRED_CHECKPOINT_TRAINING_COMMIT
            ),
            "authoritative_partition_family_counts": {
                "train": 544,
                "validation": 68,
                "test": 68,
            },
            "resolved_selected_family_ids_before_inference": identifiers,
            "family_limit": 6,
            "batch_size": 3,
            "requested_device": "cpu",
            "resolved_device": "cpu",
            "repository_branch": "flat-mixed-baseline",
            "source_dirty": False,
            "source_tree_sha256": "d" * 64,
            "payload_access": {
                "train_family_records_loaded": 0,
                "validation_family_records_loaded": len(identifiers),
                "test_family_records_loaded": 0,
                "loaded_family_ids": identifiers,
            },
        })
        for field, value in (
            ("source_dirty", True),
            ("resolved_device", "cuda"),
            ("source_tree_sha256", "invalid"),
        ):
            with self.subTest(field=field):
                damaged = dict(metadata, **{field: value})
                artifacts = make_artifacts(
                    records, raw_records, damaged, selected, True
                )
                output = (
                    Path(self.temporary.name)
                    / ("bad-repaired-source-" + field)
                )
                with self.assertRaisesRegex(
                    EvaluationError, "repaired_smoke_metadata"
                ):
                    publish_artifacts(
                        output,
                        artifacts,
                        expected_ids=identifiers,
                        expected_partition="validation",
                        authoritative_validation_ids=identifiers,
                        authoritative_examples=selected,
                        expected_checkpoint_sha256=(
                            REPAIRED_CHECKPOINT_SHA256
                        ),
                    )
                self.assertFalse(output.exists())

    def test_verifier_reloads_authority_and_hashes_checkpoint(self):
        _, output, _, selected, checkpoint_bytes = self.run_fixture(
            "verifier"
        )
        corpus = Path(self.temporary.name) / "run-corpus-verifier"
        checkpoint = Path(self.temporary.name) / "verifier.pt"
        stdout = io.StringIO()
        with mock.patch("sys.stdout", stdout):
            status = verifier_main((
                "--output", str(output),
                "--corpus-dir", str(corpus),
                "--checkpoint", str(checkpoint),
                "--reviewed-commit", "c" * 40,
                "--expected-authoritative-family-count", "5",
                "--expected-authoritative-validation-count", "4",
            ))
        self.assertEqual(status, 0)
        report = json.loads(stdout.getvalue())
        self.assertEqual(report["processed_family_count"], len(selected))
        self.assertEqual(
            json.loads((output / "run_metadata.json").read_text())[
                "checkpoint_sha256"
            ],
            hashlib.sha256(checkpoint_bytes).hexdigest(),
        )
        checkpoint.write_bytes(b"different-checkpoint")
        with mock.patch("sys.stderr", io.StringIO()):
            self.assertEqual(verifier_main((
                "--output", str(output),
                "--corpus-dir", str(corpus),
                "--checkpoint", str(checkpoint),
                "--reviewed-commit", "c" * 40,
            )), 1)

    def test_end_to_end_structural_invalidity_publishes_and_returns_two(self):
        status, output, _, selected, _ = self.run_fixture(
            "structural",
            transform=lambda raw, index: None if index == 0 else raw,
        )
        self.assertEqual(status, 2)
        self.assertEqual(
            len((output / "examples.jsonl").read_text().splitlines()),
            len(selected),
        )
        self.assertTrue((output / "summary.json").is_file())

    def test_end_to_end_controlled_invalidity_returns_zero(self):
        def controlled_invalid(raw, index):
            node = raw.raw_nodes[0]
            geometry = list(node.normalized_geometry)
            geometry[node.derived_geometry_mask.index(True)] = 2.0
            return replace(
                raw,
                raw_nodes=(
                    replace(node, normalized_geometry=tuple(geometry)),
                ) + raw.raw_nodes[1:],
            )

        status, output, _, _, _ = self.run_fixture(
            "controlled", transform=controlled_invalid
        )
        self.assertEqual(status, 0)
        first = json.loads(
            (output / "examples.jsonl").read_text().splitlines()[0]
        )
        self.assertTrue(
            first["teacher_forced"]["conversion"]["raw_integrity"]["valid"]
        )
        self.assertFalse(
            first["teacher_forced"]["conversion"]["controlled_domain"]["valid"]
        )

    def test_checkpoint_preflight_failure_never_constructs_or_decodes(self):
        events = []
        with self.assertRaisesRegex(EvaluationError, "checkpoint_family_mismatch"):
            self.run_fixture(
                "bad-checkpoint",
                checkpoint_ids=(),
                event_sink=events,
            )
        self.assertEqual(
            tuple(item[0] for item in events),
            ("checkpoint_load",),
        )
        self.assertFalse(
            (Path(self.temporary.name) / "run-output-bad-checkpoint").exists()
        )

    def test_corpus_preflight_failure_never_loads_checkpoint_or_decodes(self):
        events = []
        with self.assertRaisesRegex(EvaluationError, "invalid_corpus_identity"):
            self.run_fixture(
                "bad-corpus",
                corrupt_corpus=True,
                event_sink=events,
            )
        self.assertEqual(events, [])
        self.assertFalse(
            (Path(self.temporary.name) / "run-output-bad-corpus").exists()
        )

    def test_invalid_prediction_is_retained_and_later_family_continues(self):
        calls = [0]

        def decode(items):
            predictions = tuple(_raw_from_target(item.target) for item in items)
            calls[0] += len(items)
            return SimpleNamespace(
                latent_indices=tuple((1, 2) for _ in items),
                teacher_forced=(None,) + predictions[1:],
                predicted_history=predictions,
            )

        selected, records, _ = self.records(batch_size=4, decode=decode)
        self.assertEqual(calls[0], len(selected))
        self.assertEqual(len(records), len(selected))
        self.assertFalse(records[0].teacher_forced.conversion.raw_integrity.valid)
        self.assertTrue(records[-1].predicted_history.metrics.raw_integrity.valid)
        self.assertEqual(completed_exit_code(records), 2)

    def test_controlled_domain_only_invalidity_has_success_exit_code(self):
        selected = self.selected()

        def decode(items):
            predictions = []
            for item in items:
                raw = _raw_from_target(item.target)
                node = raw.raw_nodes[0]
                geometry = list(node.normalized_geometry)
                applicable = node.derived_geometry_mask.index(True)
                geometry[applicable] = 2.0
                predictions.append(replace(
                    raw,
                    raw_nodes=(
                        replace(node, normalized_geometry=tuple(geometry)),
                    ) + raw.raw_nodes[1:],
                ))
            return SimpleNamespace(
                latent_indices=tuple((0, 1) for _ in items),
                teacher_forced=tuple(predictions),
                predicted_history=tuple(predictions),
            )

        records, _ = evaluate_records(
            selected, batch_size=2, max_operations=2, decode_batch=decode
        )
        self.assertTrue(records[0].teacher_forced.conversion.raw_integrity.valid)
        self.assertFalse(records[0].teacher_forced.conversion.controlled_domain.valid)
        self.assertEqual(completed_exit_code(records), 0)

    def test_exact_artifacts_and_deterministic_replay_across_batch_sizes(self):
        selected, records_a, raw_a = self.records(batch_size=1)
        _, records_b, raw_b = self.records(batch_size=3)
        artifacts_a = make_artifacts(
            records_a, raw_a, self.metadata(selected, True), selected, True
        )
        artifacts_b = make_artifacts(
            records_b, raw_b, self.metadata(selected, True), selected, True
        )
        self.assertEqual(artifacts_a, artifacts_b)
        self.assertEqual(
            set(artifacts_a),
            {
                "run_metadata.json", "examples.jsonl", "summary.json",
                "metrics.csv", "conversion_failures.csv", RAW_ARTIFACT,
            },
        )
        for content in artifacts_a.values():
            self.assertTrue(content.endswith(b"\n"))
            self.assertFalse(content.endswith(b"\n\n"))
            self.assertNotIn(b"tmp-", content)
        output_a = Path(self.temporary.name) / "published-a"
        output_b = Path(self.temporary.name) / "published-b"
        publish_artifacts(output_a, artifacts_a)
        publish_artifacts(output_b, artifacts_b)
        validate_artifact_directory(output_a, True)
        identifiers = tuple(item.physical_family_id for item in selected)
        report = inspect_publication(
            output_a,
            authoritative_validation_ids=identifiers,
            authoritative_test_ids=(),
            reviewed_commit="a" * 40,
        )
        with self.assertRaisesRegex(
            EvaluationError, "repository_commit_mismatch"
        ):
            inspect_publication(
                output_a,
                authoritative_validation_ids=identifiers,
                authoritative_test_ids=(),
                reviewed_commit="e" * 40,
            )
        compare_publications(output_a, output_b)
        self.assertFalse(report["test_partition_evaluated"])
        self.assertEqual(report["processed_family_count"], len(selected))
        self.assertEqual(
            {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in output_a.iterdir()
            },
            {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in output_b.iterdir()
            },
        )
        machine = complete_report(
            full=report,
            smoke_a=report,
            smoke_b=inspect_publication(
                output_b,
                authoritative_validation_ids=identifiers,
                authoritative_test_ids=(),
                reviewed_commit="a" * 40,
            ),
            smoke_exit_codes=(0, 2),
            full_exit_code=0,
            job_id="123",
            repository_commit="a" * 40,
            regression_status="passed",
        )
        self.assertIn("python_version", machine)
        self.assertIn("pytorch_version", machine)
        self.assertIn("smoke_artifact_sha256", machine)
        self.assertIn("latent_index_histogram", machine)
        self.assertIn("observed_codebook_utilization_fraction", machine)
        self.assertFalse(machine["test_partition_evaluated"])

    def test_artifact_schema_order_counts_and_optional_raw(self):
        selected, records, raw = self.records()
        artifacts = make_artifacts(
            records, raw, self.metadata(selected), selected, False
        )
        self.assertNotIn(RAW_ARTIFACT, artifacts)
        lines = artifacts["examples.jsonl"].decode().splitlines()
        self.assertEqual(len(lines), len(selected))
        parsed = tuple(json.loads(line) for line in lines)
        self.assertEqual(
            tuple(item["family_id"] for item in parsed),
            tuple(item.physical_family_id for item in selected),
        )
        self.assertEqual(
            artifacts["metrics.csv"].decode().splitlines()[0].split(","),
            list(METRICS_COLUMNS),
        )
        self.assertTrue({
            "controlled_domain_valid_rate",
            "exact_node_type_sequence_rate",
            "exact_complete_ten_field_match_rate",
            "finite_geometry_mae",
            "finite_geometry_rmse",
            "exact_operation_type_sequence_rate",
            "pointer_overall_accuracy",
            "edge_micro_f1",
            "exact_typed_edge_set_match_rate",
            "active_code_count",
            "codebook_perplexity",
            "codebook_utilization",
        }.issubset(METRICS_COLUMNS))
        self.assertEqual(
            artifacts["conversion_failures.csv"].decode().splitlines()[0].split(","),
            list(FAILURE_COLUMNS),
        )
        summary = json.loads(artifacts["summary.json"])
        self.assertEqual(summary["attempted_example_count"], len(selected))

    def test_collision_parent_failure_and_no_partial_publication(self):
        selected, records, raw = self.records()
        artifacts = make_artifacts(
            records, raw, self.metadata(selected), selected, False
        )
        output = Path(self.temporary.name) / "collision"
        output.mkdir()
        with self.assertRaisesRegex(EvaluationError, "output_collision"):
            publish_artifacts(output, artifacts)
        file_output = Path(self.temporary.name) / "file-output"
        file_output.write_text("")
        with self.assertRaisesRegex(EvaluationError, "output_collision"):
            publish_artifacts(file_output, artifacts)
        missing_parent = Path(self.temporary.name) / "missing" / "output"
        with self.assertRaisesRegex(EvaluationError, "invalid_output_parent"):
            publish_artifacts(missing_parent, artifacts)
        self.assertFalse(missing_parent.exists())

        failed = Path(self.temporary.name) / "failed-write"
        with self.assertRaisesRegex(EvaluationError, "publication_failure"):
            publish_artifacts(failed, {"run_metadata.json": object()})
        self.assertFalse(failed.exists())
        self.assertFalse(tuple(Path(self.temporary.name).glob(".failed-write.tmp-*")))

        renamed = Path(self.temporary.name) / "failed-rename"
        with mock.patch(
            "prototype.flat_baseline.evaluate_length_conditioned._atomic_no_replace",
            side_effect=EvaluationError("publication_failure", "injected"),
        ):
            with self.assertRaisesRegex(EvaluationError, "publication_failure"):
                publish_artifacts(renamed, artifacts)
        self.assertFalse(renamed.exists())
        self.assertFalse(tuple(Path(self.temporary.name).glob(".failed-rename.tmp-*")))

    def test_linux_no_replace_uses_exported_libc_wrapper(self):
        wrapper = FakeCFunction()
        syscall = FakeCFunction()
        library = SimpleNamespace(renameat2=wrapper, syscall=syscall)
        with mock.patch(
            "prototype.flat_baseline.evaluate_length_conditioned.ctypes.CDLL",
            return_value=library,
        ):
            _linux_rename_noreplace(Path("source"), Path("destination"))
        self.assertEqual(len(wrapper.calls), 1)
        self.assertEqual(syscall.calls, [])
        arguments = wrapper.calls[0]
        self.assertEqual(arguments[0].value, -100)
        self.assertEqual(arguments[1].value, b"source")
        self.assertEqual(arguments[2].value, -100)
        self.assertEqual(arguments[3].value, b"destination")
        self.assertEqual(arguments[4].value, 1)

    def test_linux_no_replace_uses_direct_syscall_without_wrapper(self):
        syscall = FakeCFunction()
        library = SimpleNamespace(syscall=syscall)
        with mock.patch(
            "prototype.flat_baseline.evaluate_length_conditioned.ctypes.CDLL",
            return_value=library,
        ), mock.patch(
            "prototype.flat_baseline.evaluate_length_conditioned.platform.machine",
            return_value="x86_64",
        ):
            _linux_rename_noreplace(Path("source"), Path("destination"))
        self.assertEqual(len(syscall.calls), 1)
        arguments = syscall.calls[0]
        self.assertEqual(arguments[0].value, 316)
        self.assertEqual(arguments[1].value, -100)
        self.assertEqual(arguments[2].value, b"source")
        self.assertEqual(arguments[3].value, -100)
        self.assertEqual(arguments[4].value, b"destination")
        self.assertEqual(arguments[5].value, 1)

    def test_linux_unsupported_wrapper_tries_direct_syscall(self):
        wrapper = FakeCFunction(-1, errno.ENOTSUP)
        syscall = FakeCFunction()
        library = SimpleNamespace(renameat2=wrapper, syscall=syscall)
        with mock.patch(
            "prototype.flat_baseline.evaluate_length_conditioned.ctypes.CDLL",
            return_value=library,
        ), mock.patch(
            "prototype.flat_baseline.evaluate_length_conditioned.platform.machine",
            return_value="x86_64",
        ):
            result = _linux_rename_noreplace(
                Path("source"), Path("destination")
            )
        self.assertEqual(len(wrapper.calls), 1)
        self.assertEqual(len(syscall.calls), 1)
        self.assertEqual(result["publication_backend"], "renameat2")

    def test_linux_direct_syscall_collision_maps_to_output_collision(self):
        syscall = FakeCFunction(-1, errno.EEXIST)
        library = SimpleNamespace(syscall=syscall)
        with mock.patch(
            "prototype.flat_baseline.evaluate_length_conditioned.ctypes.CDLL",
            return_value=library,
        ), mock.patch(
            "prototype.flat_baseline.evaluate_length_conditioned.platform.machine",
            return_value="x86_64",
        ):
            with self.assertRaisesRegex(EvaluationError, "output_collision"):
                _linux_rename_noreplace(
                    Path("source"), Path("destination")
                )

    def test_linux_direct_syscall_enosys_is_unsupported(self):
        syscall = FakeCFunction(-1, errno.ENOSYS)
        library = SimpleNamespace(syscall=syscall)
        with mock.patch(
            "prototype.flat_baseline.evaluate_length_conditioned.ctypes.CDLL",
            return_value=library,
        ), mock.patch(
            "prototype.flat_baseline.evaluate_length_conditioned.platform.machine",
            return_value="x86_64",
        ):
            with self.assertRaisesRegex(
                EvaluationError, "unsupported_no_replace"
            ) as caught:
                _linux_rename_noreplace(
                    Path("source"), Path("destination")
                )
        self.assertEqual(caught.exception.error_number, errno.ENOSYS)
        self.assertIn("errno={}".format(errno.ENOSYS), caught.exception.detail)
        self.assertIn("errno_name=ENOSYS", caught.exception.detail)
        self.assertIn("backend=renameat2_syscall", caught.exception.detail)
        self.assertIn("platform_machine=x86_64", caught.exception.detail)

    def test_linux_direct_syscall_rejects_unknown_architecture(self):
        syscall = FakeCFunction()
        library = SimpleNamespace(syscall=syscall)
        with mock.patch(
            "prototype.flat_baseline.evaluate_length_conditioned.ctypes.CDLL",
            return_value=library,
        ), mock.patch(
            "prototype.flat_baseline.evaluate_length_conditioned.platform.machine",
            return_value="unknown-linux-architecture",
        ):
            with self.assertRaisesRegex(
                EvaluationError, "unsupported_no_replace"
            ):
                _linux_rename_noreplace(
                    Path("source"), Path("destination")
                )
        self.assertEqual(syscall.calls, [])

    @unittest.skipUnless(
        sys.platform.startswith("linux"),
        "actual Linux no-replace primitive requires Linux",
    )
    def test_actual_linux_no_replace_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source"
            destination_path = root / "destination"
            source_path.mkdir()
            (source_path / "evidence.txt").write_text("published")
            _atomic_no_replace(source_path, destination_path)
            self.assertFalse(source_path.exists())
            self.assertEqual(
                (destination_path / "evidence.txt").read_text(),
                "published",
            )

    def test_unsupported_renameat2_publishes_complete_sibling_symlink(self):
        selected, records, raw = self.records()
        artifacts = make_artifacts(
            records, raw, self.metadata(selected), selected, False
        )
        output = Path(self.temporary.name) / "symlink-publication"
        self.publish_with_symlink(output, artifacts)
        self.assertTrue(output.is_symlink())
        target = Path(os.readlink(str(output)))
        self.assertFalse(target.is_absolute())
        self.assertEqual(len(target.parts), 1)
        backing = output.parent / target
        self.assertTrue(backing.is_dir())
        self.assertEqual(
            {path.name for path in backing.iterdir()}, set(artifacts)
        )
        self.assertEqual(
            {path.name: path.read_bytes() for path in output.iterdir()},
            artifacts,
        )
        self.assertEqual(
            _resolve_publication_directory(output), backing.resolve()
        )
        identifiers = tuple(item.physical_family_id for item in selected)
        report = inspect_publication(
            output,
            authoritative_validation_ids=identifiers,
            authoritative_test_ids=(),
            reviewed_commit="a" * 40,
        )
        self.assertEqual(report["processed_family_count"], len(selected))

    def test_public_symlink_appears_only_after_backing_validation(self):
        selected, records, raw = self.records()
        artifacts = make_artifacts(
            records, raw, self.metadata(selected), selected, False
        )
        output = Path(self.temporary.name) / "validated-before-publication"
        real_validate = validate_artifact_directory
        observations = []

        def validating(directory, **arguments):
            observations.append((output.exists(), output.is_symlink()))
            return real_validate(directory, **arguments)

        with mock.patch.object(sys, "platform", "linux"), mock.patch(
            "prototype.flat_baseline.evaluate_length_conditioned."
            "_linux_rename_noreplace",
            side_effect=self.unsupported_rename_error(),
        ), mock.patch(
            "prototype.flat_baseline.evaluate_length_conditioned."
            "validate_artifact_directory",
            side_effect=validating,
        ):
            publish_artifacts(output, artifacts)
        self.assertEqual(observations, [(False, False)])
        self.assertTrue(output.is_symlink())

    def test_symlink_race_collisions_preserve_every_destination_type(self):
        selected, records, raw = self.records()
        artifacts = make_artifacts(
            records, raw, self.metadata(selected), selected, False
        )
        real_symlink = os.symlink
        for destination_type in ("file", "directory", "symlink"):
            with self.subTest(destination_type=destination_type):
                output = (
                    Path(self.temporary.name)
                    / ("symlink-race-" + destination_type)
                )

                def collide(target, link_name, target_is_directory=False):
                    if destination_type == "file":
                        output.write_bytes(b"untouched-file")
                    elif destination_type == "directory":
                        output.mkdir()
                        (output / "untouched").write_bytes(b"directory")
                    else:
                        real_symlink("untouched-target", str(output))
                    return real_symlink(
                        target,
                        link_name,
                        target_is_directory=target_is_directory,
                    )

                with mock.patch.object(sys, "platform", "linux"), mock.patch(
                    "prototype.flat_baseline.evaluate_length_conditioned."
                    "_linux_rename_noreplace",
                    side_effect=self.unsupported_rename_error(),
                ), mock.patch(
                    "prototype.flat_baseline.evaluate_length_conditioned."
                    "os.symlink",
                    side_effect=collide,
                ):
                    with self.assertRaisesRegex(
                        EvaluationError, "output_collision"
                    ):
                        publish_artifacts(output, artifacts)
                if destination_type == "file":
                    self.assertEqual(output.read_bytes(), b"untouched-file")
                elif destination_type == "directory":
                    self.assertEqual(
                        (output / "untouched").read_bytes(), b"directory"
                    )
                else:
                    self.assertEqual(
                        os.readlink(str(output)), "untouched-target"
                    )
                self.assertFalse(tuple(
                    Path(self.temporary.name).glob(
                        "." + output.name + ".tmp-*"
                    )
                ))

    def test_failed_symlink_publication_removes_unpublished_backing(self):
        selected, records, raw = self.records()
        artifacts = make_artifacts(
            records, raw, self.metadata(selected), selected, False
        )
        output = Path(self.temporary.name) / "symlink-permission-failure"
        with mock.patch.object(sys, "platform", "linux"), mock.patch(
            "prototype.flat_baseline.evaluate_length_conditioned."
            "_linux_rename_noreplace",
            side_effect=self.unsupported_rename_error(),
        ), mock.patch(
            "prototype.flat_baseline.evaluate_length_conditioned.os.symlink",
            side_effect=PermissionError(errno.EACCES, "injected"),
        ):
            with self.assertRaisesRegex(
                EvaluationError, "publication_failure"
            ):
                publish_artifacts(output, artifacts)
        self.assertFalse(output.exists())
        self.assertFalse(output.is_symlink())
        self.assertFalse(tuple(
            Path(self.temporary.name).glob(
                "." + output.name + ".tmp-*"
            )
        ))

    def test_unrelated_rename_failure_does_not_use_symlink_fallback(self):
        source_path = Path(self.temporary.name) / "unrelated-source"
        destination_path = Path(self.temporary.name) / "unrelated-destination"
        source_path.mkdir()
        failure = EvaluationError(
            "publication_failure",
            "injected unrelated rename failure",
        )
        with mock.patch.object(sys, "platform", "linux"), mock.patch(
            "prototype.flat_baseline.evaluate_length_conditioned."
            "_linux_rename_noreplace",
            side_effect=failure,
        ), mock.patch(
            "prototype.flat_baseline.evaluate_length_conditioned.os.symlink"
        ) as symlink:
            with self.assertRaisesRegex(
                EvaluationError, "publication_failure"
            ):
                _atomic_no_replace(source_path, destination_path)
        symlink.assert_not_called()
        self.assertTrue(source_path.is_dir())
        self.assertFalse(destination_path.exists())

    def test_publication_validator_rejects_unsafe_symlink_targets(self):
        selected, records, raw = self.records()
        artifacts = make_artifacts(
            records, raw, self.metadata(selected), selected, False
        )
        valid_public = Path(self.temporary.name) / "valid-public"
        self.publish_with_symlink(valid_public, artifacts)
        backing = _resolve_publication_directory(valid_public)
        valid_public.unlink()
        nested = Path(self.temporary.name) / "nested"
        nested.mkdir()
        outside_context = tempfile.TemporaryDirectory()
        self.addCleanup(outside_context.cleanup)
        outside = Path(outside_context.name)
        relay = Path(self.temporary.name) / "external-relay"
        os.symlink(str(outside), str(relay))
        not_directory = Path(self.temporary.name) / "not-directory"
        not_directory.write_text("file")
        cases = {
            "absolute": str(backing),
            "parent": "..",
            "nested": "nested/backing",
            "dangling": "missing-backing",
            "external": relay.name,
            "file": not_directory.name,
        }
        identifiers = tuple(item.physical_family_id for item in selected)
        for name, target in cases.items():
            with self.subTest(name=name):
                public = Path(self.temporary.name) / ("invalid-" + name)
                os.symlink(target, str(public))
                with self.assertRaisesRegex(
                    EvaluationError, "invalid_publication_symlink"
                ):
                    inspect_publication(
                        public,
                        authoritative_validation_ids=identifiers,
                        authoritative_test_ids=(),
                        reviewed_commit="a" * 40,
                    )

    def test_deterministic_replay_reads_through_public_symlinks(self):
        selected, records, raw = self.records()
        artifacts = make_artifacts(
            records, raw, self.metadata(selected), selected, False
        )
        output_a = Path(self.temporary.name) / "symlink-replay-a"
        output_b = Path(self.temporary.name) / "symlink-replay-b"
        self.publish_with_symlink(output_a, artifacts)
        self.publish_with_symlink(output_b, artifacts)
        self.assertTrue(output_a.is_symlink())
        self.assertTrue(output_b.is_symlink())
        self.assertTrue(compare_publications(output_a, output_b))

    def test_destination_created_at_atomic_rename_is_never_replaced(self):
        selected, records, raw = self.records()
        artifacts = make_artifacts(
            records, raw, self.metadata(selected), selected, False
        )
        output = Path(self.temporary.name) / "rename-race"
        real_rename = _atomic_no_replace

        def create_destination(source_path, destination_path):
            destination_path.mkdir()
            (destination_path / "survives.txt").write_text("untouched")
            return real_rename(source_path, destination_path)

        with mock.patch(
            "prototype.flat_baseline.evaluate_length_conditioned._atomic_no_replace",
            side_effect=create_destination,
        ):
            with self.assertRaisesRegex(EvaluationError, "output_collision"):
                publish_artifacts(output, artifacts)
        self.assertEqual(
            (output / "survives.txt").read_text(), "untouched"
        )
        self.assertFalse(tuple(
            Path(self.temporary.name).glob(".rename-race.tmp-*")
        ))

    def test_historical_output_directory_does_not_block_new_publication(self):
        selected, records, raw = self.records()
        artifacts = make_artifacts(
            records, raw, self.metadata(selected), selected, False
        )
        historical = Path(self.temporary.name) / "validation-3324856"
        historical.mkdir()
        (historical / "preserved.txt").write_text("historical")
        current = Path(self.temporary.name) / "phase-b-repaired-smoke-current"
        publish_artifacts(current, artifacts)
        self.assertTrue(current.exists())
        self.assertEqual(
            (historical / "preserved.txt").read_text(), "historical"
        )

    def test_repaired_smoke_slurm_workflow_is_separate_and_staged(self):
        script = (
            Path(__file__).resolve().parents[1]
            / "adroit"
            / "evaluate_repaired_smoke_cpu.slurm"
        ).read_text()
        stages = (
            "repository_preflight",
            "environment_preflight",
            "checkpoint_preflight",
            "partition_preflight",
            "regressions",
            "smoke_a",
            "smoke_b",
            "smoke_replay",
            "artifact_validation",
            "complete",
        )
        offsets = [script.index("stage " + stage) for stage in stages]
        self.assertEqual(offsets, sorted(offsets))
        self.assertIn("--repaired-smoke-contract", script)
        self.assertIn("train-kmeans-full-d549ebe", script)
        self.assertIn("EXPECTED_CHECKPOINT_SHA256=", script)
        self.assertIn("failure_stage=%s failure_line=%s exit_code=%s", script)
        self.assertIn("phase-b-repaired-smoke-${REVIEWED_COMMIT}", script)

    def test_repaired_full_slurm_workflow_is_validation_only_and_staged(self):
        script = (
            Path(__file__).resolve().parents[1]
            / "adroit"
            / "evaluate_repaired_validation_cpu.slurm"
        ).read_text()
        stages = (
            "repository_preflight",
            "environment_preflight",
            "checkpoint_preflight",
            "partition_preflight",
            "regressions",
            "execution_preflight",
            "full_evaluation",
            "post_evaluation_evidence",
            "artifact_validation",
            "artifact_manifest",
            "complete",
        )
        offsets = [script.index("stage " + stage) for stage in stages]
        self.assertEqual(offsets, sorted(offsets))
        self.assertIn("--repaired-full-contract", script)
        self.assertNotIn("--allow-test-evaluation", script)
        self.assertNotIn("--family-limit", script)
        self.assertNotIn("load_physical_examples", script)
        self.assertNotIn("OpenCascade", script)
        self.assertIn("--batch-size 32", script)
        self.assertIn("selected_family_count\": len(selected)", script)
        self.assertIn("test_family_records_loaded\": 0", script)
        self.assertIn("EXPECTED_CHECKPOINT_SHA256=", script)
        self.assertIn("gate-a-to-d-inputs.json", script)
        self.assertIn("sha256-manifest.txt", script)
        self.assertIn("--workflow-evidence-dir \"$EVIDENCE\"", script)
        self.assertGreaterEqual(
            script.count(
                'git status --porcelain=v1 --untracked-files=all'
            ),
            2,
        )
        self.assertIn("snapshot=post_evaluation", script)
        self.assertIn("torch.manual_seed(2026)", script)
        self.assertIn("torch.set_num_threads(1)", script)

    def test_repaired_full_report_preserves_inputs_without_gate_e_decision(self):
        validity = {
            "controlled_domain_valid_count": 20,
        }
        usage = {
            "active_code_count": 4,
            "codebook_perplexity": 2.5,
        }
        full = {
            "output_directory": "/new/full",
            "artifact_sha256": {"summary.json": "a" * 64},
            "processed_family_count": 68,
            "selected_validation_ids": [
                "sf_{:064x}".format(index) for index in range(68)
            ],
            "selected_validation_ids_sha256": "b" * 64,
            "teacher_forced_metrics": {"validity": validity},
            "predicted_history_metrics": {"validity": validity},
            "teacher_forced_validity": validity,
            "predicted_history_validity": validity,
            "metric_gaps": {},
            "latent_usage": usage,
            "gate_a_artifact_inputs": {
                "test_partition_evaluated": False,
            },
            "predicted_history_operation_coverage": {
                "extrude_qualifying_family_count": 3,
                "revolve_qualifying_family_count": 4,
                "overlap_qualifying_family_count": 1,
                "unique_qualifying_family_count": 6,
            },
        }
        report = repaired_full_report(
            full,
            full_exit_code=0,
            job_id="123",
            repository_commit="c" * 40,
            regression_status="passed",
            workflow_evidence_sha256={"environment.json": "d" * 64},
        )
        self.assertEqual(report["processed_family_count"], 68)
        self.assertEqual(
            report["gate_c_inputs"][
                "minimum_predicted_history_survival_count"
            ],
            10,
        )
        self.assertFalse(report["opencascade"]["gate_e_evaluated"])
        self.assertEqual(
            report["final_acceptance"]["status"], "not_determined"
        )
        self.assertEqual(
            report["workflow_evidence_sha256"]["environment.json"],
            "d" * 64,
        )

    def test_repaired_full_workflow_evidence_is_validated_and_hashed(self):
        root = Path(self.temporary.name) / "workflow-evidence"
        root.mkdir()
        identifiers = tuple(
            "sf_{:064x}".format(index) for index in range(68)
        )
        files = {
            "checkpoint.sha256": (
                REPAIRED_CHECKPOINT_SHA256 + "  /checkpoint/best.pt\n"
            ),
            "container.sha256": "a" * 64 + "  /container.sif\n",
            "corpus-manifests.sha256": (
                "b" * 64 + "  /corpus/corpus_manifest.json\n"
                + "c" * 64 + "  /corpus/manifests/iid.json\n"
            ),
            "environment.json": json.dumps({
                "cuda_available": False,
                "cuda_visible_devices": "",
                "mkl_num_threads": "1",
                "omp_num_threads": "1",
                "platform": "fixture",
                "python_version": "3.8.13",
                "pytorch_version": "1.11.0",
                "pythonhashseed": "0",
                "evaluation_seed": 2026,
                "torch_num_threads": 1,
            }, sort_keys=True, separators=(",", ":")) + "\n",
            "partition.json": json.dumps({
                "partition": "validation",
                "authoritative_partition_family_counts": {
                    "train": 544,
                    "validation": 68,
                    "test": 68,
                },
                "selected_family_ids": identifiers,
                "selected_family_ids_sha256": (
                    _identifier_sha256(identifiers)
                ),
                "selected_family_count": 68,
                "train_family_records_loaded": 0,
                "validation_family_records_loaded": 0,
                "test_family_records_loaded": 0,
                "test_partition_evaluated": False,
            }, sort_keys=True, separators=(",", ":")) + "\n",
            "regressions.txt": "status=passed\n",
            "repository.txt": (
                "flat-mixed-baseline\n" + "d" * 40 + "\n" + "d" * 40 + "\n"
            ),
            "scheduler.txt": (
                "slurm_job_id=3329040\n"
                "snapshot=post_evaluation\n"
                "full_evaluation_exit_code=0\n"
                "JobState=RUNNING RunTime=00:10:00 NodeList=node1\n"
            ),
        }
        for name, content in files.items():
            (root / name).write_text(content)
        hashes = _workflow_evidence_hashes(
            root,
            expected_ids=identifiers,
            expected_job_id="3329040",
            expected_full_exit_code=0,
        )
        self.assertEqual(set(hashes), set(files))
        with self.assertRaisesRegex(
            ValueError, "workflow scheduler evidence differs"
        ):
            _workflow_evidence_hashes(
                root,
                expected_ids=identifiers,
                expected_job_id="different-job",
                expected_full_exit_code=0,
            )
        (root / "environment.json").write_text(
            files["environment.json"].replace("3.8.13", "3.9.0")
        )
        with self.assertRaisesRegex(
            ValueError, "workflow environment evidence differs"
        ):
            _workflow_evidence_hashes(root, expected_ids=identifiers)

    def test_repaired_full_operation_coverage_uses_qualifying_family_sets(self):
        def item(family_id, operations, valid=True, exact=True):
            return {
                "family_id": family_id,
                "target_operation_sequence": operations,
                "predicted_history": {
                    "metrics": {
                        "controlled_domain": {"valid": valid},
                        "operations": {
                            "exact_operation_type_sequence": exact,
                        },
                    },
                },
            }

        coverage = _predicted_history_operation_coverage((
            item("extrude", ["extrude"]),
            item("revolve", ["revolve"]),
            item("mixed", ["extrude", "revolve"]),
            item("invalid", ["extrude"], valid=False),
            item("inexact", ["revolve"], exact=False),
        ))
        self.assertEqual(coverage, {
            "extrude_qualifying_family_count": 2,
            "revolve_qualifying_family_count": 2,
            "overlap_qualifying_family_count": 1,
            "unique_qualifying_family_count": 3,
        })

    def test_unsupported_no_replace_never_publishes(self):
        selected, records, raw = self.records()
        artifacts = make_artifacts(
            records, raw, self.metadata(selected), selected, False
        )
        output = Path(self.temporary.name) / "unsupported-rename"
        with mock.patch(
            "prototype.flat_baseline.evaluate_length_conditioned._atomic_no_replace",
            side_effect=EvaluationError(
                "unsupported_no_replace", "injected"
            ),
        ):
            with self.assertRaisesRegex(EvaluationError, "unsupported_no_replace"):
                publish_artifacts(output, artifacts)
        self.assertFalse(output.exists())

    def test_logically_inconsistent_parseable_artifacts_do_not_publish(self):
        selected, records, raw = self.records()
        artifacts = make_artifacts(
            records, raw, self.metadata(selected), selected, False
        )
        summary = json.loads(artifacts["summary.json"])
        summary["attempted_example_count"] += 1
        artifacts["summary.json"] = _json_document(summary)
        output = Path(self.temporary.name) / "bad-summary"
        with self.assertRaisesRegex(EvaluationError, "summary_count_mismatch"):
            publish_artifacts(output, artifacts)
        self.assertFalse(output.exists())

    def test_raw_examples_are_recomputed_against_authoritative_targets(self):
        selected, records, raw = self.records()
        artifacts = make_artifacts(
            records, raw, self.metadata(selected, True), selected, True
        )
        raw_values = tuple(
            json.loads(line)
            for line in artifacts[RAW_ARTIFACT].decode().splitlines()
        )
        first = raw_values[0]
        applicable = next(
            index
            for index, value in enumerate(
                first["teacher_forced"]["raw_nodes"][0][
                    "derived_geometry_mask"
                ]
            )
            if value
        )
        first["teacher_forced"]["raw_nodes"][0][
            "normalized_geometry"
        ][applicable] += 0.25
        artifacts[RAW_ARTIFACT] = _json_lines(raw_values)
        output = Path(self.temporary.name) / "raw-example-mismatch"
        with self.assertRaisesRegex(EvaluationError, "raw_example_mismatch"):
            publish_artifacts(
                output,
                artifacts,
                authoritative_examples=selected,
            )
        self.assertFalse(output.exists())

    def test_checkpoint_digest_is_independently_reconciled(self):
        selected, records, raw = self.records()
        artifacts = make_artifacts(
            records, raw, self.metadata(selected), selected, False
        )
        output = Path(self.temporary.name) / "wrong-checkpoint-digest"
        with self.assertRaisesRegex(
            EvaluationError, "checkpoint_digest_mismatch"
        ):
            publish_artifacts(
                output,
                artifacts,
                expected_checkpoint_sha256="c" * 64,
            )
        self.assertFalse(output.exists())

    def test_stored_conversion_and_metric_validity_must_match(self):
        selected, records, raw = self.records()
        artifacts = make_artifacts(
            records, raw, self.metadata(selected), selected, False
        )
        examples = tuple(
            json.loads(line)
            for line in artifacts["examples.jsonl"].decode().splitlines()
        )
        examples[0]["teacher_forced"]["metrics"]["raw_integrity"][
            "valid"
        ] = False
        artifacts["examples.jsonl"] = _json_lines(examples)
        output = Path(self.temporary.name) / "conversion-metric-mismatch"
        with self.assertRaisesRegex(
            EvaluationError, "conversion_metric_mismatch"
        ):
            publish_artifacts(output, artifacts)
        self.assertFalse(output.exists())

    def test_incorrect_metrics_csv_row_count_does_not_publish(self):
        selected, records, raw = self.records()
        artifacts = make_artifacts(
            records, raw, self.metadata(selected), selected, False
        )
        lines = artifacts["metrics.csv"].decode().splitlines()
        artifacts["metrics.csv"] = (
            "\n".join(lines[:-1]) + "\n"
        ).encode()
        output = Path(self.temporary.name) / "bad-metric-rows"
        with self.assertRaisesRegex(EvaluationError, "metrics_row_mismatch"):
            publish_artifacts(output, artifacts)
        self.assertFalse(output.exists())

    def test_complete_failure_fields_are_reconciled(self):
        selected = self.selected()

        def damaged(items):
            predictions = tuple(_raw_from_target(item.target) for item in items)
            return SimpleNamespace(
                latent_indices=tuple((0, 1) for _ in items),
                teacher_forced=(None,) + predictions[1:],
                predicted_history=predictions,
            )

        records, raw = evaluate_records(
            selected,
            batch_size=len(selected),
            max_operations=2,
            decode_batch=damaged,
        )
        base = make_artifacts(
            records, raw, self.metadata(selected), selected, False
        )
        for field in ("layer", "stage", "location", "detail"):
            artifacts = dict(base)
            rows = artifacts["conversion_failures.csv"].decode().splitlines()
            header = rows[0].split(",")
            values = rows[1].split(",")
            values[header.index(field)] = "wrong"
            rows[1] = ",".join(values)
            artifacts["conversion_failures.csv"] = (
                "\n".join(rows) + "\n"
            ).encode()
            output = Path(self.temporary.name) / ("bad-failure-" + field)
            with self.subTest(field=field):
                with self.assertRaisesRegex(
                    EvaluationError, "conversion_failure_mismatch"
                ):
                    publish_artifacts(output, artifacts)
                self.assertFalse(output.exists())

    def test_repeated_secondary_code_keeps_rows_but_counts_example_once(self):
        selected, _, artifacts = self.repeated_failure_artifacts(
            distinct_primary=True
        )
        rows = tuple(
            item for item in self.failure_rows(artifacts)
            if item["code"] == "nonfinite_geometry"
        )
        self.assertEqual(len(rows), 2)
        self.assertEqual(
            {item["designation"] for item in rows}, {"secondary"}
        )
        self.assertEqual(len({item["location"] for item in rows}), 2)
        self.assertEqual(
            self.aggregate_failure_count(
                artifacts, "any", "nonfinite_geometry"
            ),
            1,
        )
        output = Path(self.temporary.name) / "repeated-secondary"
        publish_artifacts(
            output,
            artifacts,
            expected_ids=(selected[0].physical_family_id,),
            expected_partition="validation",
            authoritative_validation_ids=(
                selected[0].physical_family_id,
            ),
        )
        self.assertTrue(output.is_dir())

    def test_primary_and_secondary_same_code_count_example_once(self):
        _, _, artifacts = self.repeated_failure_artifacts()
        rows = tuple(
            item for item in self.failure_rows(artifacts)
            if item["code"] == "nonfinite_geometry"
        )
        self.assertEqual(len(rows), 2)
        self.assertEqual(
            tuple(item["designation"] for item in rows),
            ("primary", "secondary"),
        )
        self.assertEqual(
            self.aggregate_failure_count(
                artifacts, "primary", "nonfinite_geometry"
            ),
            1,
        )
        self.assertEqual(
            self.aggregate_failure_count(
                artifacts, "any", "nonfinite_geometry"
            ),
            1,
        )

    def test_two_failure_codes_each_contribute_once_per_example(self):
        _, records, artifacts = self.repeated_failure_artifacts(
            include_edge=True
        )
        conversion = json.loads(artifacts["examples.jsonl"])[
            "teacher_forced"
        ]["conversion"]
        primary, any_failure = _failure_counters(({
            "conversion": conversion,
        },))
        self.assertEqual(primary["nonfinite_geometry"], 1)
        self.assertEqual(any_failure["nonfinite_geometry"], 1)
        self.assertEqual(any_failure["nonfinite_edge_logit"], 1)
        self.assertEqual(sum(any_failure.values()), 2)
        self.assertEqual(len(records), 1)

    def test_prepublication_accepts_stage_two_per_example_failure_counts(self):
        selected, _, artifacts = self.repeated_failure_artifacts(
            distinct_primary=True
        )
        output = Path(self.temporary.name) / "stage-two-counting"
        publish_artifacts(
            output,
            artifacts,
            expected_ids=(selected[0].physical_family_id,),
            expected_partition="validation",
            authoritative_validation_ids=(
                selected[0].physical_family_id,
            ),
        )
        self.assertTrue(
            validate_artifact_directory(
                output,
                expected_ids=(selected[0].physical_family_id,),
                expected_partition="validation",
                authoritative_validation_ids=(
                    selected[0].physical_family_id,
                ),
            )
        )

    def test_prepublication_rejects_failure_occurrence_counting(self):
        selected, _, artifacts = self.repeated_failure_artifacts(
            distinct_primary=True
        )
        summary = json.loads(artifacts["summary.json"])
        values = summary["teacher_forced"]["overall"]["validity"][
            "any_failure_counts"
        ]
        target = next(
            item for item in values
            if item["code"] == "nonfinite_geometry"
        )
        self.assertEqual(target["count"], 1)
        target["count"] = 2
        artifacts["summary.json"] = _json_document(summary)
        output = Path(self.temporary.name) / "occurrence-counting"
        with self.assertRaisesRegex(
            EvaluationError, "summary_failure_mismatch"
        ):
            publish_artifacts(
                output,
                artifacts,
                expected_ids=(selected[0].physical_family_id,),
                expected_partition="validation",
                authoritative_validation_ids=(
                    selected[0].physical_family_id,
                ),
            )
        self.assertFalse(output.exists())

    def test_wrong_authoritative_68_ids_are_rejected(self):
        authoritative = tuple(
            "sf_{:064x}".format(index) for index in range(68)
        )
        metadata = {
            "partition": "validation",
            "family_limit": None,
            "authoritative_validation_family_ids": list(authoritative),
            "authoritative_validation_family_ids_sha256": (
                _identifier_sha256(authoritative)
            ),
            "checkpoint_data_state": {
                "validation_family_ids": list(authoritative),
            },
        }
        wrong = authoritative[:-1] + ("sf_" + "f" * 64,)
        with self.assertRaisesRegex(
            EvaluationError, "authoritative_validation_mismatch"
        ):
            _validate_authoritative_ids(metadata, authoritative, wrong)

    def test_validation_selection_containing_test_id_is_rejected(self):
        selected, records, raw = self.records()
        artifacts = make_artifacts(
            records, raw, self.metadata(selected), selected, False
        )
        output = Path(self.temporary.name) / "test-family"
        identifiers = tuple(item.physical_family_id for item in selected)
        with self.assertRaisesRegex(EvaluationError, "test_family_selected"):
            publish_artifacts(
                output,
                artifacts,
                expected_ids=identifiers,
                expected_partition="validation",
                authoritative_validation_ids=identifiers,
                authoritative_test_ids=(identifiers[0],),
            )
        self.assertFalse(output.exists())

    def test_nonfinite_raw_is_tagged_and_scientific_json_remains_strict(self):
        selected = self.selected()
        raw = _raw_from_target(selected[0].target)
        node = replace(
            raw.raw_nodes[0],
            normalized_geometry=(float("nan"),) + raw.raw_nodes[0].normalized_geometry[1:],
        )
        damaged = replace(raw, raw_nodes=(node,) + raw.raw_nodes[1:])

        def decode(items):
            predictions = tuple(
                damaged if index == 0 else _raw_from_target(item.target)
                for index, item in enumerate(items)
            )
            return SimpleNamespace(
                latent_indices=tuple((0, 1) for _ in items),
                teacher_forced=predictions,
                predicted_history=predictions,
            )

        records, raws = evaluate_records(
            selected, batch_size=len(selected), max_operations=2, decode_batch=decode
        )
        artifacts = make_artifacts(
            records, raws, self.metadata(selected, True), selected, True
        )
        self.assertIn(b'"nonfinite_float":"nan"', artifacts[RAW_ARTIFACT])
        self.assertNotIn(b":NaN", artifacts[RAW_ARTIFACT])
        self.assertEqual(len(records), len(selected))

    def test_semantic_gap_matches_failure_records_by_code(self):
        predicted = {
            "failure_counts": [
                {"code": "beta", "count": 4},
                {"code": "alpha", "count": 2},
            ]
        }
        teacher = {
            "failure_counts": [
                {"code": "gamma", "count": 1},
                {"code": "alpha", "count": 1},
            ]
        }
        gap = _semantic_difference(predicted, teacher)["failure_counts"]
        self.assertEqual(
            [item["code"] for item in gap],
            ["alpha", "beta", "gamma"],
        )
        self.assertEqual(gap[0]["count"], 1)
        self.assertIsNone(gap[1]["teacher_forced"])
        self.assertIsNone(gap[2]["predicted_history"])

    def test_machine_report_publication_collision_write_and_rename_failures(self):
        report = {"python_version": "actual", "test_partition_evaluated": False}
        collision = Path(self.temporary.name) / "report-collision.json"
        collision.write_text("existing")
        with self.assertRaisesRegex(EvaluationError, "output_collision"):
            publish_json_report(collision, report)
        self.assertEqual(collision.read_text(), "existing")

        write_failure = Path(self.temporary.name) / "report-write.json"
        with mock.patch(
            "prototype.flat_baseline.evaluate_length_conditioned._json_document",
            side_effect=TypeError("injected"),
        ):
            with self.assertRaisesRegex(
                EvaluationError, "report_publication_failure"
            ):
                publish_json_report(write_failure, report)
        self.assertFalse(write_failure.exists())

        rename_failure = Path(self.temporary.name) / "report-rename.json"
        with mock.patch(
            "prototype.flat_baseline.evaluate_length_conditioned._atomic_no_replace",
            side_effect=EvaluationError("publication_failure", "injected"),
        ):
            with self.assertRaisesRegex(EvaluationError, "publication_failure"):
                publish_json_report(rename_failure, report)
        self.assertFalse(rename_failure.exists())

    def test_machine_report_symlink_publication_and_collision_cleanup(self):
        report = {"python_version": "actual", "test_partition_evaluated": False}
        output = Path(self.temporary.name) / "symlink-report.json"
        with mock.patch.object(sys, "platform", "linux"), mock.patch(
            "prototype.flat_baseline.evaluate_length_conditioned."
            "_linux_rename_noreplace",
            side_effect=self.unsupported_rename_error(errno.ENOTSUP),
        ):
            publish_json_report(output, report)
        self.assertTrue(output.is_symlink())
        target = Path(os.readlink(str(output)))
        self.assertEqual(len(target.parts), 1)
        backing = _resolve_publication_file(output)
        self.assertEqual(backing.parent, output.parent.resolve())
        self.assertEqual(json.loads(output.read_text()), report)
        self.assertTrue(backing.is_file())

        collision = Path(self.temporary.name) / "report-race.json"
        real_symlink = os.symlink

        def collide(target_name, link_name, target_is_directory=False):
            collision.write_bytes(b"untouched-report")
            return real_symlink(
                target_name,
                link_name,
                target_is_directory=target_is_directory,
            )

        with mock.patch.object(sys, "platform", "linux"), mock.patch(
            "prototype.flat_baseline.evaluate_length_conditioned."
            "_linux_rename_noreplace",
            side_effect=self.unsupported_rename_error(),
        ), mock.patch(
            "prototype.flat_baseline.evaluate_length_conditioned.os.symlink",
            side_effect=collide,
        ):
            with self.assertRaisesRegex(EvaluationError, "output_collision"):
                publish_json_report(collision, report)
        self.assertEqual(collision.read_bytes(), b"untouched-report")
        self.assertFalse(tuple(
            Path(self.temporary.name).glob(
                "." + collision.name + ".tmp-*"
            )
        ))

        failed = Path(self.temporary.name) / "report-symlink-failure.json"
        with mock.patch.object(sys, "platform", "linux"), mock.patch(
            "prototype.flat_baseline.evaluate_length_conditioned."
            "_linux_rename_noreplace",
            side_effect=self.unsupported_rename_error(),
        ), mock.patch(
            "prototype.flat_baseline.evaluate_length_conditioned.os.symlink",
            side_effect=PermissionError(errno.EACCES, "injected"),
        ):
            with self.assertRaisesRegex(
                EvaluationError, "publication_failure"
            ):
                publish_json_report(failed, report)
        self.assertFalse(failed.exists())
        self.assertFalse(tuple(
            Path(self.temporary.name).glob("." + failed.name + ".tmp-*")
        ))

    def test_checkpoint_hash_reads_exact_bytes(self):
        path = Path(self.temporary.name) / "checkpoint.pt"
        path.write_bytes(b"checkpoint-before-load")
        self.assertEqual(
            checkpoint_sha256(path),
            hashlib.sha256(b"checkpoint-before-load").hexdigest(),
        )

    def test_untouched_authoritative_fixture_corpus_identity_passes(self):
        root = Path(self.temporary.name) / "corpus"
        identity = corpus_identity(root, "iid")
        self.assertIsNone(identity["corpus_configuration_sha256"])
        self.assertEqual(
            identity["corpus_identity_validation"],
            "unavailable_in_fixture_manifest",
        )
        self.assertEqual(len(identity["corpus_manifest_sha256"]), 64)
        self.assertEqual(len(identity["split_manifest_sha256"]), 64)

    def test_frozen_pilot_manifest_configuration_hash_contract(self):
        root = Path(self.temporary.name) / "representative"
        (root / "manifests").mkdir(parents=True)
        declared = (
            "4118c170a14b527ccc0eccb21e9806c6470f35a44ab79a39da58e02133598289"
        )
        corpus = {
            "normalized_configuration": {
                "num_source_families": 680,
                "seed": 2026,
            },
            "configuration_sha256": declared,
            "representation_schema_version": 1,
        }
        split = {"corpus_configuration_sha256": declared}
        (root / "corpus_manifest.json").write_text(json.dumps(corpus))
        (root / "manifests" / "iid.json").write_text(json.dumps(split))
        identity = corpus_identity(root, "iid")
        self.assertEqual(identity["corpus_configuration_sha256"], declared)
        self.assertEqual(identity["corpus_identity_validation"], "validated")

    def test_model_state_loading_is_strict_and_rejects_incompatibility(self):
        class Model:
            def __init__(self, failure=False):
                self.failure = failure
                self.calls = []

            def load_state_dict(self, state, strict):
                self.calls.append((state, strict))
                if self.failure:
                    raise RuntimeError("missing or unexpected keys")

        model = Model()
        _strict_load_model(model, {"weight": 1})
        self.assertEqual(model.calls, [({"weight": 1}, True)])
        incompatible = Model(failure=True)
        with self.assertRaisesRegex(EvaluationError, "incompatible_model_state"):
            _strict_load_model(incompatible, {"unexpected": 1})
        self.assertEqual(incompatible.calls, [({"unexpected": 1}, True)])

    def test_checkpoint_structure_configuration_and_data_state_are_validated(self):
        selected = self.selected()
        path = Path(self.temporary.name) / "best.pt"
        path.write_bytes(b"checkpoint-payload")
        family_ids = [item.physical_family_id for item in selected]
        payload = {
            "checkpoint_version": 1,
            "checkpoint_kind": "best",
            "model_state": {"weight": "opaque"},
            "optimizer_state": {},
            "epoch": 7,
            "global_step": 11,
            "model_config": FlatBaselineConfig().to_dict(),
            "training_config": TrainingConfig().to_dict(),
            "best_validation_metric": 1.0,
            "rng_state": {},
            "data_state": {
                "split_manifest": "iid",
                "validation_partition": "validation",
                "validation_family_ids": family_ids,
            },
        }
        torch_module = SimpleNamespace(
            load=lambda loaded_path, map_location: payload
        )
        digest, loaded, model_config, training_config = (
            load_and_validate_checkpoint(
                path,
                torch_module,
                tuple(item.physical_family_id for item in selected),
                "iid",
            )
        )
        self.assertEqual(digest, hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertIs(loaded, payload)
        self.assertEqual(model_config, FlatBaselineConfig())
        self.assertEqual(training_config, TrainingConfig())

        malformed = dict(payload)
        malformed["model_config"] = {"unknown": 1}
        torch_module.load = lambda loaded_path, map_location: malformed
        with self.assertRaisesRegex(EvaluationError, "invalid_checkpoint"):
            load_and_validate_checkpoint(
                path,
                torch_module,
                tuple(item.physical_family_id for item in selected),
                "iid",
            )
        inconsistent = dict(payload)
        inconsistent["data_state"] = dict(
            payload["data_state"], validation_family_ids=[]
        )
        torch_module.load = lambda loaded_path, map_location: inconsistent
        with self.assertRaisesRegex(EvaluationError, "checkpoint_family_mismatch"):
            load_and_validate_checkpoint(
                path,
                torch_module,
                tuple(item.physical_family_id for item in selected),
                "iid",
            )

    def test_checkpoint_hash_precedes_and_can_prevent_deserialization(self):
        selected = self.selected()
        path = Path(self.temporary.name) / "hash-before-load.pt"
        path.write_bytes(b"actual-checkpoint-bytes")
        events = []

        class Torch:
            @staticmethod
            def load(loaded_path, map_location):
                events.append(("load", loaded_path, map_location))
                raise AssertionError("deserialization must not occur")

        with self.assertRaisesRegex(
            EvaluationError, "checkpoint_digest_mismatch"
        ):
            load_and_validate_checkpoint(
                path,
                Torch(),
                tuple(item.physical_family_id for item in selected),
                "iid",
                expected_sha256="0" * 64,
            )
        self.assertEqual(events, [])

    def test_missing_checkpoint_prevents_deserialization(self):
        selected = self.selected()
        events = []
        torch_module = SimpleNamespace(
            load=lambda *args, **kwargs: events.append((args, kwargs))
        )
        with self.assertRaisesRegex(
            EvaluationError, "checkpoint_unavailable"
        ):
            load_and_validate_checkpoint(
                Path(self.temporary.name) / "absent.pt",
                torch_module,
                tuple(item.physical_family_id for item in selected),
                "iid",
                expected_sha256="0" * 64,
            )
        self.assertEqual(events, [])

    def test_repaired_checkpoint_epoch_step_and_kmeans_provenance(self):
        selected = self.selected()
        path = Path(self.temporary.name) / "repaired-contract.pt"
        path.write_bytes(b"repaired-contract-fixture")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        validation_ids = [
            item.physical_family_id for item in selected
        ]
        train_ids = ("train-a", "train-b")
        model = FlatBaselineConfig(codebook_size=32)
        training = TrainingConfig(seed=2026, vq_init="train-kmeans")
        payload = {
            "checkpoint_version": 1,
            "checkpoint_kind": "best",
            "model_state": {"weight": "opaque"},
            "optimizer_state": {},
            "epoch": REPAIRED_CHECKPOINT_EPOCH,
            "global_step": REPAIRED_CHECKPOINT_GLOBAL_STEP,
            "model_config": model.to_dict(),
            "training_config": training.to_dict(),
            "best_validation_metric": 1.0,
            "rng_state": {},
            "data_state": {
                "split_manifest": "iid",
                "train_partition": "train",
                "validation_partition": "validation",
                "train_family_ids": list(train_ids),
                "validation_family_ids": validation_ids,
                "initialization_mode": "train-kmeans",
                "initialization_algorithm": "seeded-kmeans++",
                "initialization_seed": 2026,
                "initialization_pseudo_count_policy": "matched",
                "initialization_report_sha256": "a" * 64,
            },
        }
        torch_module = SimpleNamespace(
            load=lambda loaded_path, map_location: payload
        )
        loaded = load_and_validate_checkpoint(
            path,
            torch_module,
            tuple(validation_ids),
            "iid",
            expected_sha256=digest,
            expected_checkpoint_kind="best",
            expected_epoch=REPAIRED_CHECKPOINT_EPOCH,
            expected_global_step=REPAIRED_CHECKPOINT_GLOBAL_STEP,
            authoritative_train_ids=train_ids,
            require_train_kmeans=True,
            expected_codebook_size=32,
        )
        self.assertIs(loaded[1], payload)

        for field, value, expected_error in (
            ("checkpoint_kind", "last", "checkpoint_kind_mismatch"),
            ("epoch", 43, "checkpoint_epoch_mismatch"),
            ("global_step", 747, "checkpoint_global_step_mismatch"),
        ):
            with self.subTest(field=field):
                damaged = dict(payload)
                damaged[field] = value
                torch_module.load = (
                    lambda loaded_path, map_location, item=damaged: item
                )
                with self.assertRaisesRegex(EvaluationError, expected_error):
                    load_and_validate_checkpoint(
                        path,
                        torch_module,
                        tuple(validation_ids),
                        "iid",
                        expected_sha256=digest,
                        expected_checkpoint_kind="best",
                        expected_epoch=REPAIRED_CHECKPOINT_EPOCH,
                        expected_global_step=(
                            REPAIRED_CHECKPOINT_GLOBAL_STEP
                        ),
                        authoritative_train_ids=train_ids,
                        require_train_kmeans=True,
                        expected_codebook_size=32,
                    )

        damaged = dict(payload)
        damaged["data_state"] = dict(
            payload["data_state"], initialization_mode="normal"
        )
        torch_module.load = lambda loaded_path, map_location: damaged
        with self.assertRaisesRegex(
            EvaluationError, "checkpoint_initialization_provenance"
        ):
            load_and_validate_checkpoint(
                path,
                torch_module,
                tuple(validation_ids),
                "iid",
                expected_sha256=digest,
                expected_checkpoint_kind="best",
                expected_epoch=REPAIRED_CHECKPOINT_EPOCH,
                expected_global_step=REPAIRED_CHECKPOINT_GLOBAL_STEP,
                authoritative_train_ids=train_ids,
                require_train_kmeans=True,
                expected_codebook_size=32,
            )


if __name__ == "__main__":
    unittest.main()
