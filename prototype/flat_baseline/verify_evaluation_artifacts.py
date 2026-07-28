"""Strict post-publication checks and the Adroit machine report."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import sys

from prototype.model_data.loader import (
    load_physical_examples,
    load_partition_physical_examples,
    partition_family_ids,
)

from .evaluate_length_conditioned import (
    EvaluationError,
    REPAIRED_CHECKPOINT_SHA256,
    REPAIRED_FULL_FAMILY_COUNT,
    REPAIRED_PARTITION_COUNTS,
    REPAIRED_SMOKE_FAMILY_COUNT,
    _identifier_sha256,
    _resolve_publication_directory,
    checkpoint_sha256,
    publish_json_report,
    validate_artifact_directory,
)


def authoritative_corpus(
    corpus_dir,
    split_manifest="iid",
):
    examples = tuple(load_physical_examples(corpus_dir, split_manifest))
    validation = tuple(sorted(
        item.physical_family_id
        for item in examples
        if item.partition == "validation"
    ))
    test = tuple(sorted(
        item.physical_family_id
        for item in examples
        if item.partition == "test"
    ))
    if (
        len(validation) != len(set(validation))
        or len(test) != len(set(test))
    ):
        raise ValueError(
            "authoritative partition contains duplicate family IDs"
        )
    return examples, validation, test


def repaired_authoritative_corpus(
    corpus_dir, split_manifest="iid", family_limit=None
):
    authority = partition_family_ids(corpus_dir, split_manifest)
    validation = authority["validation"]
    test = authority["test"]
    selected = (
        validation if family_limit is None else validation[:family_limit]
    )
    examples = load_partition_physical_examples(
        corpus_dir,
        split_manifest,
        "validation",
        selected,
    )
    return examples, validation, test, authority


def authoritative_partition_ids(corpus_dir, split_manifest="iid"):
    _, validation, test = authoritative_corpus(corpus_dir, split_manifest)
    return validation, test


def inspect_publication(
    path,
    *,
    authoritative_validation_ids,
    authoritative_test_ids,
    reviewed_commit,
    family_limit=None,
    authoritative_examples=None,
    expected_checkpoint_sha256=None,
):
    expected = (
        tuple(authoritative_validation_ids)
        if family_limit is None
        else tuple(authoritative_validation_ids)[:family_limit]
    )
    validated = validate_artifact_directory(
        path,
        expected_ids=expected,
        expected_partition="validation",
        authoritative_validation_ids=authoritative_validation_ids,
        authoritative_test_ids=authoritative_test_ids,
        authoritative_examples=(
            None
            if authoritative_examples is None
            else tuple(
                item
                for item in authoritative_examples
                if item.physical_family_id in set(expected)
            )
        ),
        reviewed_commit=reviewed_commit,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
    )
    metadata = validated["metadata"]
    summary = validated["summary"]
    examples = validated["examples"]
    if metadata.get("repaired_evaluation_contract") is True:
        usage = summary["latent_usage"]
        counts = usage["per_code_assignment_counts"]
        histogram = {
            index: count for index, count in enumerate(counts)
        }
        active_count = usage["active_code_count"]
        utilization = usage["codebook_utilization"]
        perplexity = usage["codebook_perplexity"]
    else:
        histogram = Counter(
            value for item in examples for value in item["latent_indices"]
        )
        codebook_size = metadata["model_configuration"]["codebook_size"]
        active_count = len(histogram)
        utilization = active_count / codebook_size
        perplexity = None
    result = {
        "output_directory": str(Path(path)),
        "processed_family_count": len(expected),
        "selected_validation_ids": list(expected),
        "selected_validation_ids_sha256": _identifier_sha256(expected),
        "artifact_sha256": _artifact_hashes(path),
        "teacher_forced_validity": _validity_headline(
            summary["teacher_forced"]["overall"]["validity"]
        ),
        "predicted_history_validity": _validity_headline(
            summary["predicted_history"]["overall"]["validity"]
        ),
        "metric_gaps": summary["predicted_minus_teacher_forced"]["overall"],
        "dominant_failure_counts": _dominant_failures(validated["failures"]),
        "latent_index_histogram": {
            str(key): histogram[key] for key in sorted(histogram)
        },
        "active_observed_code_count": active_count,
        "observed_codebook_utilization_fraction": utilization,
        "test_partition_evaluated": metadata["test_partition_evaluated"],
    }
    if perplexity is not None:
        result["observed_codebook_perplexity"] = perplexity
        result["latent_usage"] = usage
    if metadata.get("repaired_full_contract") is True:
        result["teacher_forced_metrics"] = summary[
            "teacher_forced"
        ]["overall"]
        result["predicted_history_metrics"] = summary[
            "predicted_history"
        ]["overall"]
        result["predicted_history_operation_coverage"] = (
            _predicted_history_operation_coverage(examples)
        )
        result["gate_a_artifact_inputs"] = {
            "checkpoint_sha256": metadata["checkpoint_sha256"],
            "checkpoint_epoch": metadata["checkpoint_epoch"],
            "checkpoint_global_step": metadata["checkpoint_global_step"],
            "checkpoint_training_source_commit": metadata[
                "checkpoint_training_source_commit"
            ],
            "repository_commit": metadata["repository_commit"],
            "source_tree_sha256": metadata["source_tree_sha256"],
            "source_dirty": metadata["source_dirty"],
            "evaluation_seed": metadata["evaluation_seed"],
            "torch_num_threads": metadata["torch_num_threads"],
            "partition": metadata["partition"],
            "authoritative_partition_family_counts": metadata[
                "authoritative_partition_family_counts"
            ],
            "authoritative_validation_family_ids_sha256": metadata[
                "authoritative_validation_family_ids_sha256"
            ],
            "corpus_configuration_sha256": metadata[
                "corpus_configuration_sha256"
            ],
            "corpus_manifest_sha256": metadata[
                "corpus_manifest_sha256"
            ],
            "split_manifest_sha256": metadata[
                "split_manifest_sha256"
            ],
            "payload_access": metadata["payload_access"],
            "raw_predictions_published": metadata[
                "raw_predictions_published"
            ],
            "test_partition_evaluated": metadata[
                "test_partition_evaluated"
            ],
            "shared_reconstructed_latent_memory": usage[
                "shared_by_decoding_paths"
            ],
        }
    return result


def compare_publications(left, right):
    left_hashes = _artifact_hashes(left)
    right_hashes = _artifact_hashes(right)
    if left_hashes != right_hashes:
        raise ValueError("smoke publications are not byte-identical")
    return True


def complete_report(
    *,
    full,
    smoke_a,
    smoke_b,
    smoke_exit_codes,
    full_exit_code,
    job_id,
    repository_commit,
    regression_status,
):
    if any(code not in (0, 2) for code in (*smoke_exit_codes, full_exit_code)):
        raise ValueError("evaluation exit code is not a completed result")
    compare_publications(smoke_a["output_directory"], smoke_b["output_directory"])
    report = {
        "job_id": str(job_id),
        "repository_commit": repository_commit,
        "python_version": platform.python_version(),
        "pytorch_version": _pytorch_version(),
        "regression_status": regression_status,
        "smoke_exit_codes": list(smoke_exit_codes),
        "smoke_artifact_sha256": {
            "first": smoke_a["artifact_sha256"],
            "second": smoke_b["artifact_sha256"],
        },
        "byte_identical_smoke_replay": True,
        "full_evaluation_exit_code": full_exit_code,
        "full_output_directory": full["output_directory"],
        "full_artifact_sha256": full["artifact_sha256"],
        "processed_family_count": full["processed_family_count"],
        "teacher_forced_validity": full["teacher_forced_validity"],
        "predicted_history_validity": full["predicted_history_validity"],
        "metric_gaps": full["metric_gaps"],
        "dominant_failure_counts": full["dominant_failure_counts"],
        "latent_index_histogram": full["latent_index_histogram"],
        "active_observed_code_count": full["active_observed_code_count"],
        "observed_codebook_utilization_fraction": (
            full["observed_codebook_utilization_fraction"]
        ),
        "test_partition_evaluated": full["test_partition_evaluated"],
        "selected_validation_ids": full["selected_validation_ids"],
        "selected_validation_ids_sha256": (
            full["selected_validation_ids_sha256"]
        ),
    }
    if "latent_usage" in full:
        report["observed_codebook_perplexity"] = full[
            "observed_codebook_perplexity"
        ]
        report["latent_usage"] = full["latent_usage"]
    return report


def repaired_full_report(
    full,
    *,
    full_exit_code,
    job_id,
    repository_commit,
    regression_status,
    workflow_evidence_sha256,
):
    """Build the frozen, artifact-derived Gate A-D input report."""

    if (
        full_exit_code not in (0, 2)
        or job_id is None
        or regression_status is None
        or not workflow_evidence_sha256
    ):
        raise ValueError("full report completion evidence is required")
    usage = full["latent_usage"]
    teacher_valid = full["teacher_forced_validity"][
        "controlled_domain_valid_count"
    ]
    predicted_valid = full["predicted_history_validity"][
        "controlled_domain_valid_count"
    ]
    coverage = full["predicted_history_operation_coverage"]
    return {
        "report_kind": "repaired_phase_b_full_validation",
        "job_id": str(job_id),
        "repository_commit": repository_commit,
        "python_version": platform.python_version(),
        "pytorch_version": _pytorch_version(),
        "regression_status": regression_status,
        "full_evaluation_exit_code": full_exit_code,
        "evaluation_output_directory": full["output_directory"],
        "artifact_sha256": full["artifact_sha256"],
        "workflow_evidence_sha256": workflow_evidence_sha256,
        "processed_family_count": full["processed_family_count"],
        "selected_validation_ids": full["selected_validation_ids"],
        "selected_validation_ids_sha256": (
            full["selected_validation_ids_sha256"]
        ),
        "teacher_forced_metrics": full["teacher_forced_metrics"],
        "predicted_history_metrics": full["predicted_history_metrics"],
        "metric_gaps": full["metric_gaps"],
        "latent_usage": usage,
        "gate_a_artifact_inputs": full["gate_a_artifact_inputs"],
        "gate_b_inputs": {
            "active_code_count": usage["active_code_count"],
            "minimum_active_code_count": 2,
            "codebook_perplexity": usage["codebook_perplexity"],
            "minimum_codebook_perplexity": 2.0,
        },
        "gate_c_inputs": {
            "teacher_forced_controlled_domain_valid_count": teacher_valid,
            "predicted_history_controlled_domain_valid_count": predicted_valid,
            "minimum_each": 17,
            "minimum_predicted_history_survival_count": (
                (teacher_valid + 1) // 2
            ),
            "collapsed_predicted_history_valid_count": 0,
            "collapsed_comparison_job": "3324856",
        },
        "gate_d_inputs": {
            **coverage,
            "minimum_extrude_qualifying_family_count": 3,
            "minimum_revolve_qualifying_family_count": 3,
        },
        "opencascade": {
            "status": "not_run",
            "gate_e_evaluated": False,
        },
        "final_acceptance": {
            "status": "not_determined",
            "reason": "Gate E requires a separate frozen-artifact job",
        },
    }


def main(argv=None):
    parser = _parser()
    arguments = parser.parse_args(argv)
    try:
        if (
            arguments.repaired_smoke_contract
            and arguments.repaired_full_contract
        ):
            raise ValueError("choose exactly one repaired contract")
        repaired_contract = (
            arguments.repaired_smoke_contract
            or arguments.repaired_full_contract
        )
        if repaired_contract:
            examples, validation_ids, test_ids, authority = (
                repaired_authoritative_corpus(
                    arguments.corpus_dir,
                    arguments.split_manifest,
                    arguments.family_limit,
                )
            )
        else:
            examples, validation_ids, test_ids = authoritative_corpus(
                arguments.corpus_dir, arguments.split_manifest
            )
            authority = None
        partition_counts = (
            {
                name: len(authority[name])
                for name in ("train", "validation", "test")
            }
            if authority is not None else None
        )
        if (
            arguments.expected_authoritative_family_count is not None
            and (
                sum(partition_counts.values())
                if partition_counts is not None else len(examples)
            )
            != arguments.expected_authoritative_family_count
        ):
            raise ValueError("authoritative physical-family count differs")
        if (
            arguments.expected_authoritative_validation_count is not None
            and len(validation_ids)
            != arguments.expected_authoritative_validation_count
        ):
            raise ValueError("authoritative validation count differs")
        if repaired_contract and partition_counts != REPAIRED_PARTITION_COUNTS:
            raise ValueError("repaired partition authority differs")
        if (
            arguments.repaired_smoke_contract
            and arguments.family_limit != REPAIRED_SMOKE_FAMILY_COUNT
        ):
            raise ValueError("repaired smoke selection differs")
        if (
            arguments.repaired_full_contract
            and (
                arguments.family_limit is not None
                or len(validation_ids) != REPAIRED_FULL_FAMILY_COUNT
            )
        ):
            raise ValueError("repaired full selection differs")
        checkpoint_digest = checkpoint_sha256(arguments.checkpoint)
        if (
            repaired_contract
            and checkpoint_digest != REPAIRED_CHECKPOINT_SHA256
        ):
            raise ValueError("repaired checkpoint SHA-256 differs")
        primary = inspect_publication(
            arguments.output,
            authoritative_validation_ids=validation_ids,
            authoritative_test_ids=test_ids,
            reviewed_commit=arguments.reviewed_commit,
            family_limit=arguments.family_limit,
            authoritative_examples=examples,
            expected_checkpoint_sha256=checkpoint_digest,
        )
        if arguments.compare_output:
            inspect_publication(
                arguments.compare_output,
                authoritative_validation_ids=validation_ids,
                authoritative_test_ids=test_ids,
                reviewed_commit=arguments.reviewed_commit,
                family_limit=arguments.family_limit,
                authoritative_examples=examples,
                expected_checkpoint_sha256=checkpoint_digest,
            )
            compare_publications(arguments.output, arguments.compare_output)
            primary["byte_identical_replay"] = True
        if arguments.report_output:
            if arguments.repaired_full_contract:
                workflow_evidence_sha256 = _workflow_evidence_hashes(
                    arguments.workflow_evidence_dir,
                    expected_ids=tuple(primary["selected_validation_ids"]),
                )
                report = repaired_full_report(
                    primary,
                    full_exit_code=arguments.full_exit_code,
                    job_id=arguments.job_id,
                    repository_commit=arguments.reviewed_commit,
                    regression_status=arguments.regression_status,
                    workflow_evidence_sha256=workflow_evidence_sha256,
                )
            else:
                report = _report_from_arguments(
                    arguments, primary, examples, validation_ids, test_ids
                )
            publish_json_report(arguments.report_output, report)
            output = report
        else:
            output = primary
    except (EvaluationError, OSError, KeyError, TypeError, ValueError) as exc:
        sys.stderr.write("artifact_validation_failure: {}\n".format(exc))
        return 1
    sys.stdout.write(json.dumps(
        output, sort_keys=True, separators=(",", ":"), allow_nan=False
    ) + "\n")
    return 0


def _report_from_arguments(
    arguments, full, examples, validation_ids, test_ids
):
    required = (
        arguments.smoke_output_a,
        arguments.smoke_output_b,
        arguments.smoke_exit_code_a,
        arguments.smoke_exit_code_b,
        arguments.full_exit_code,
        arguments.job_id,
        arguments.regression_status,
    )
    if any(value is None for value in required):
        raise ValueError("complete report arguments are required")
    smoke_a = inspect_publication(
        arguments.smoke_output_a,
        authoritative_validation_ids=validation_ids,
        authoritative_test_ids=test_ids,
        reviewed_commit=arguments.reviewed_commit,
        family_limit=arguments.smoke_family_limit,
        authoritative_examples=examples,
        expected_checkpoint_sha256=checkpoint_sha256(arguments.checkpoint),
    )
    smoke_b = inspect_publication(
        arguments.smoke_output_b,
        authoritative_validation_ids=validation_ids,
        authoritative_test_ids=test_ids,
        reviewed_commit=arguments.reviewed_commit,
        family_limit=arguments.smoke_family_limit,
        authoritative_examples=examples,
        expected_checkpoint_sha256=checkpoint_sha256(arguments.checkpoint),
    )
    return complete_report(
        full=full,
        smoke_a=smoke_a,
        smoke_b=smoke_b,
        smoke_exit_codes=(
            arguments.smoke_exit_code_a,
            arguments.smoke_exit_code_b,
        ),
        full_exit_code=arguments.full_exit_code,
        job_id=arguments.job_id,
        repository_commit=arguments.reviewed_commit,
        regression_status=arguments.regression_status,
    )


def _parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--corpus-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split-manifest", default="iid")
    parser.add_argument("--reviewed-commit", required=True)
    parser.add_argument("--expected-authoritative-family-count", type=int)
    parser.add_argument("--expected-authoritative-validation-count", type=int)
    parser.add_argument("--family-limit", type=int)
    parser.add_argument("--compare-output")
    parser.add_argument("--report-output")
    parser.add_argument("--smoke-output-a")
    parser.add_argument("--smoke-output-b")
    parser.add_argument("--smoke-family-limit", type=int, default=6)
    parser.add_argument("--smoke-exit-code-a", type=int)
    parser.add_argument("--smoke-exit-code-b", type=int)
    parser.add_argument("--full-exit-code", type=int)
    parser.add_argument("--job-id")
    parser.add_argument("--regression-status")
    parser.add_argument("--workflow-evidence-dir")
    parser.add_argument("--repaired-smoke-contract", action="store_true")
    parser.add_argument("--repaired-full-contract", action="store_true")
    return parser


def _validity_headline(validity):
    fields = (
        "attempted_example_count",
        "raw_completion_count",
        "raw_integrity_valid_count",
        "reconstruction_target_valid_count",
        "controlled_domain_valid_count",
    )
    return {field: validity[field] for field in fields}


def _artifact_hashes(path):
    root = _resolve_publication_directory(path)
    return {
        item.name: hashlib.sha256(item.read_bytes()).hexdigest()
        for item in sorted(root.iterdir(), key=lambda item: item.name)
    }


def _workflow_evidence_hashes(path, *, expected_ids=None):
    if path is None:
        raise ValueError("workflow evidence directory is required")
    root = Path(path)
    required = {
        "checkpoint.sha256",
        "container.sha256",
        "corpus-manifests.sha256",
        "environment.json",
        "partition.json",
        "regressions.txt",
        "repository.txt",
        "scheduler.txt",
    }
    if (
        not root.is_dir()
        or root.is_symlink()
        or {item.name for item in root.iterdir()} != required
        or any(not item.is_file() or item.is_symlink() for item in root.iterdir())
    ):
        raise ValueError("workflow evidence set differs")
    environment = json.loads(
        (root / "environment.json").read_text(encoding="utf-8")
    )
    if (
        not str(environment.get("python_version", "")).startswith("3.8.")
        or str(environment.get("pytorch_version", "")).split("+")[0]
        != "1.11.0"
        or environment.get("cuda_available") is not False
        or environment.get("cuda_visible_devices") != ""
        or environment.get("omp_num_threads") != "1"
        or environment.get("mkl_num_threads") != "1"
        or environment.get("pythonhashseed") != "0"
        or environment.get("evaluation_seed") != 2026
        or environment.get("torch_num_threads") != 1
    ):
        raise ValueError("workflow environment evidence differs")
    partition = json.loads(
        (root / "partition.json").read_text(encoding="utf-8")
    )
    selected_ids = partition.get("selected_family_ids")
    if (
        partition.get("partition") != "validation"
        or partition.get("authoritative_partition_family_counts")
        != REPAIRED_PARTITION_COUNTS
        or partition.get("selected_family_count")
        != REPAIRED_FULL_FAMILY_COUNT
        or not isinstance(selected_ids, list)
        or len(selected_ids) != REPAIRED_FULL_FAMILY_COUNT
        or len(selected_ids) != len(set(selected_ids))
        or selected_ids != sorted(selected_ids)
        or (
            expected_ids is not None
            and tuple(selected_ids) != tuple(expected_ids)
        )
        or partition.get("train_family_records_loaded") != 0
        or partition.get("test_family_records_loaded") != 0
        or partition.get("test_partition_evaluated") is not False
        or partition.get("selected_family_ids_sha256")
        != _identifier_sha256(tuple(selected_ids))
    ):
        raise ValueError("workflow partition evidence differs")
    if (root / "regressions.txt").read_text(encoding="utf-8") != (
        "status=passed\n"
    ):
        raise ValueError("workflow regression evidence differs")
    checkpoint_line = (root / "checkpoint.sha256").read_text(
        encoding="utf-8"
    )
    if not checkpoint_line.startswith(REPAIRED_CHECKPOINT_SHA256 + "  "):
        raise ValueError("workflow checkpoint evidence differs")
    scheduler = (root / "scheduler.txt").read_text(encoding="utf-8")
    if (
        "snapshot=post_evaluation\n" not in scheduler
        or "full_evaluation_exit_code=" not in scheduler
        or "JobState=" not in scheduler
        or "RunTime=" not in scheduler
        or "NodeList=" not in scheduler
    ):
        raise ValueError("workflow scheduler evidence differs")
    return {
        item.name: hashlib.sha256(item.read_bytes()).hexdigest()
        for item in sorted(root.iterdir(), key=lambda item: item.name)
    }


def _dominant_failures(rows):
    counts = Counter(row["code"] for row in rows)
    return [
        {"code": code, "count": count}
        for code, count in sorted(
            counts.items(), key=lambda item: (-item[1], item[0])
        )[:10]
    ]


def _predicted_history_operation_coverage(examples):
    extrude = set()
    revolve = set()
    for item in examples:
        metrics = item["predicted_history"]["metrics"]
        if (
            not metrics["controlled_domain"]["valid"]
            or not metrics["operations"]["exact_operation_type_sequence"]
        ):
            continue
        operations = {
            value.lower() for value in item["target_operation_sequence"]
        }
        if "extrude" in operations:
            extrude.add(item["family_id"])
        if "revolve" in operations:
            revolve.add(item["family_id"])
    return {
        "extrude_qualifying_family_count": len(extrude),
        "revolve_qualifying_family_count": len(revolve),
        "overlap_qualifying_family_count": len(extrude & revolve),
        "unique_qualifying_family_count": len(extrude | revolve),
    }


def _pytorch_version():
    try:
        import torch
    except ImportError:
        return {"status": "unavailable"}
    return str(torch.__version__)


if __name__ == "__main__":
    sys.exit(main())
