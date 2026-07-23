"""Deterministic execution aggregation, paired comparison, and publication."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

from .config import (
    PAIRED_BOUNDING_BOX_ABSOLUTE_TOLERANCE,
    PAIRED_BOUNDING_BOX_RELATIVE_TOLERANCE,
    PAIRED_VOLUME_ABSOLUTE_TOLERANCE,
    PAIRED_VOLUME_RELATIVE_TOLERANCE,
    REPORT_VERSION,
)
from .model import FailureCategory, SampleExecutionResult


class ReportError(RuntimeError):
    """An execution report could not be built or atomically published."""


def aggregate_results(results: tuple[SampleExecutionResult, ...]) -> dict[str, Any]:
    ordered = tuple(sorted(results, key=lambda item: item.sample_id))
    total = len(ordered)
    successful = sum(item.kernel_status == "success" for item in ordered)
    failures = {category.value: 0 for category in FailureCategory}
    for item in ordered:
        if item.failure_category is not None:
            failures[item.failure_category] = failures.get(item.failure_category, 0) + 1
    return {
        "total_attempted": total,
        "total_successful": successful,
        "overall_success_rate": _rate(successful, total),
        "failures_by_category": failures,
        "by_operation_template": _group_success(ordered, "operation_template"),
        "by_boolean_mode": _operation_success(ordered, "boolean_mode"),
        "by_operation_type": _operation_success(ordered, "operation_type"),
        "by_primitive_family": _group_success(ordered, "primitive_family"),
        "by_geometry_encoding": _group_success(ordered, "geometry_encoding"),
        "operation_progress_by_type": _operation_progress(ordered, "operation_type"),
        "operation_progress_by_boolean_mode": _operation_progress(ordered, "boolean_mode"),
        "paired": _paired(ordered),
    }


def build_report(
    results: tuple[SampleExecutionResult, ...],
    backend_versions: dict[str, str | None],
    corpus_metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "report_version": REPORT_VERSION,
        "backend_versions": {
            "pythonocc_version": backend_versions.get("pythonocc_version"),
            "opencascade_version": backend_versions.get("opencascade_version"),
        },
        "corpus": corpus_metadata,
        "samples": [item.to_dict() for item in sorted(results, key=lambda item: item.sample_id)],
        "aggregate": aggregate_results(results),
    }


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def publish_report(output_dir: str | Path, report: dict[str, Any]) -> None:
    final = Path(output_dir)
    if final.exists():
        raise ReportError(f"output directory already exists: {final.name}")
    if not final.parent.exists():
        raise ReportError("output parent directory does not exist")
    temporary = Path(tempfile.mkdtemp(prefix=f".{final.name}.tmp-", dir=final.parent))
    try:
        target = temporary / "execution_report.json"
        payload = canonical_json(report) + "\n"
        with target.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if target.read_text(encoding="utf-8") != payload:
            raise ReportError("written execution report failed byte verification")
        if json.loads(payload) != report:
            raise ReportError("written execution report failed semantic verification")
        os.rename(temporary, final)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _group_success(results, attribute):
    groups: dict[str, list[SampleExecutionResult]] = {}
    for item in results:
        groups.setdefault(getattr(item, attribute), []).append(item)
    return {
        key: {
            "attempted": len(items),
            "successful": sum(item.kernel_status == "success" for item in items),
            "success_rate": _rate(sum(item.kernel_status == "success" for item in items), len(items)),
        }
        for key, items in sorted(groups.items())
    }


def _operation_success(results, attribute):
    groups: dict[str, list[bool]] = {}
    for result in results:
        for operation in result.operation_results:
            groups.setdefault(getattr(operation, attribute), []).append(operation.semantically_effective)
    return {
        key: {
            "attempted": len(values),
            "successful": sum(values),
            "success_rate": _rate(sum(values), len(values)),
        }
        for key, values in sorted(groups.items())
    }


def _operation_progress(results, attribute):
    groups: dict[str, list[Any]] = {}
    for result in results:
        for operation in result.operation_results:
            groups.setdefault(getattr(operation, attribute), []).append(operation)
    return {
        key: {
            "scheduled": sum(item.scheduled for item in items),
            "reached": sum(item.reached for item in items),
            "kernel_completed": sum(item.kernel_completed for item in items),
            "semantically_effective": sum(item.semantically_effective for item in items),
            "failed": sum(item.failed for item in items),
        }
        for key, items in sorted(groups.items())
    }


def _paired(results):
    families: dict[str, dict[str, SampleExecutionResult]] = {}
    for item in results:
        variants = families.setdefault(item.source_family_id, {})
        if item.geometry_encoding in variants:
            raise ReportError(f"duplicate {item.geometry_encoding} variant in source family")
        variants[item.geometry_encoding] = item
    status_agreement = 0
    compared = 0
    geometric_agreement = 0
    both_success = 0
    both_failed = 0
    mismatches = []
    complete_pairs = 0
    continuous_only_success = 0
    quantized_only_success = 0
    for family_id, variants in sorted(families.items()):
        continuous = variants.get("continuous")
        quantized = variants.get("quantized")
        if continuous is None or quantized is None:
            continue
        complete_pairs += 1
        c_success = continuous.kernel_status == "success"
        q_success = quantized.kernel_status == "success"
        if c_success == q_success:
            status_agreement += 1
        if c_success and q_success:
            both_success += 1
            compared += 1
            differences = _metric_differences(continuous, quantized)
            if differences:
                mismatches.append({"source_family_id": family_id, "differing_metrics": differences})
            else:
                geometric_agreement += 1
        elif not c_success and not q_success:
            both_failed += 1
        elif c_success:
            continuous_only_success += 1
        else:
            quantized_only_success += 1
    return {
        "total_source_families": len(families),
        "complete_pairs": complete_pairs,
        "paired_status_agreement_count": status_agreement,
        "paired_status_agreement_rate": _rate(status_agreement, complete_pairs),
        "both_successful": both_success,
        "both_failed": both_failed,
        "continuous_only_successful": continuous_only_success,
        "quantized_only_successful": quantized_only_success,
        "successful_pairs_geometrically_compared": compared,
        "paired_geometric_agreement_count": geometric_agreement,
        "paired_geometric_agreement_rate": _rate(geometric_agreement, compared),
        "geometric_disagreements": mismatches,
    }


def _metric_differences(left, right):
    if left.final_metrics is None or right.final_metrics is None:
        return ["missing_metrics"]
    a, b = left.final_metrics, right.final_metrics
    differences = []
    if a.volume is None or b.volume is None or not _close(
        a.volume, b.volume, PAIRED_VOLUME_ABSOLUTE_TOLERANCE, PAIRED_VOLUME_RELATIVE_TOLERANCE
    ):
        differences.append("volume")
    if a.bounding_box is None or b.bounding_box is None or any(
        not _close(x, y, PAIRED_BOUNDING_BOX_ABSOLUTE_TOLERANCE, PAIRED_BOUNDING_BOX_RELATIVE_TOLERANCE)
        for x, y in zip(a.bounding_box or (), b.bounding_box or ())
    ) or (a.bounding_box is not None and b.bounding_box is not None and len(a.bounding_box) != len(b.bounding_box)):
        differences.append("bounding_box")
    for name in ("solid_count", "face_count", "edge_count", "vertex_count"):
        if getattr(a, name) != getattr(b, name):
            differences.append(name)
    return differences


def _close(left, right, absolute, relative):
    return math.isfinite(left) and math.isfinite(right) and math.isclose(
        left, right, rel_tol=relative, abs_tol=absolute
    )


def _rate(numerator, denominator):
    return numerator / denominator if denominator else None
