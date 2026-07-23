"""Join endpoint kernel results into a deterministic edit-pair execution audit."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

from prototype.kernel_validation.config import (
    PAIRED_BOUNDING_BOX_ABSOLUTE_TOLERANCE,
    PAIRED_BOUNDING_BOX_RELATIVE_TOLERANCE,
    PAIRED_VOLUME_ABSOLUTE_TOLERANCE,
    PAIRED_VOLUME_RELATIVE_TOLERANCE,
)


class PairKernelAuditError(ValueError):
    pass


def build_pair_execution_audit(
    counterfactual_manifest: dict[str, Any],
    endpoint_report: dict[str, Any],
) -> dict[str, Any]:
    results = endpoint_report.get("samples")
    samples = counterfactual_manifest.get("samples")
    if not isinstance(results, list) or not isinstance(samples, list):
        raise PairKernelAuditError("both inputs must contain sample lists")

    expected_by_sample, expected_by_source, _ = _expected_endpoints(
        samples
    )
    by_sample = {}
    for result in results:
        if not isinstance(result, dict):
            raise PairKernelAuditError("endpoint report samples must be objects")
        sample_id = result.get("sample_id")
        if not isinstance(sample_id, str) or sample_id in by_sample:
            raise PairKernelAuditError("endpoint report has invalid or duplicate sample IDs")
        by_sample[sample_id] = result
    if set(by_sample) != set(expected_by_sample):
        raise PairKernelAuditError(
            "kernel report sample IDs differ from the authoritative endpoint set"
        )
    for sample_id, expected in expected_by_sample.items():
        result = by_sample[sample_id]
        if (
            result.get("sample_id") != sample_id
            or result.get("source_family_id") != expected["source_family_id"]
            or result.get("geometry_encoding") != expected["geometry_encoding"]
        ):
            raise PairKernelAuditError(
                f"kernel endpoint metadata disagrees for {sample_id}"
            )

    endpoint_agreement = {}
    by_source = {}
    for source_id, variants in sorted(expected_by_source.items()):
        continuous = by_sample[variants["continuous"]]
        quantized = by_sample[variants["quantized"]]
        by_source[source_id] = {
            "continuous": continuous,
            "quantized": quantized,
        }
        status_agreement = (
            continuous.get("kernel_status") == quantized.get("kernel_status")
        )
        geometric_agreement = bool(
            status_agreement
            and continuous.get("kernel_status") == "success"
            and _metrics_agree(
                continuous.get("final_metrics"),
                quantized.get("final_metrics"),
            )
        )
        endpoint_agreement[source_id] = {
            "complete_encoding_pair": True,
            "status_agreement": status_agreement,
            "geometric_agreement": geometric_agreement,
        }

    pair_results = []
    for sample in sorted(samples, key=lambda item: item["edit_sample_id"]):
        base = by_sample.get(sample["base_sample_id"])
        edited = by_sample.get(sample["edited_sample_id"])
        base_success = base.get("kernel_status") == "success"
        edited_success = edited.get("kernel_status") == "success"
        pair_results.append(
            {
                "edit_sample_id": sample["edit_sample_id"],
                "edit_family_id": sample["edit_family_id"],
                "geometry_encoding": sample["geometry_encoding"],
                "base_sample_id": sample["base_sample_id"],
                "edited_sample_id": sample["edited_sample_id"],
                "base_kernel_status": base.get("kernel_status"),
                "edited_kernel_status": edited.get("kernel_status"),
                "both_endpoints_successful": base_success and edited_success,
                "base_encoding_pair_agreement": endpoint_agreement[
                    sample["base_source_family_id"]
                ],
                "edited_encoding_pair_agreement": endpoint_agreement[
                    sample["edited_source_family_id"]
                ],
            }
        )
    successful = sum(item["both_endpoints_successful"] for item in pair_results)
    endpoint_pairs_agree = sum(
        item["status_agreement"] and (
            item["geometric_agreement"]
            or by_source[source_id]["continuous"].get("kernel_status") != "success"
        )
        for source_id, item in endpoint_agreement.items()
    )
    return {
        "pair_execution_audit_version": 1,
        "counterfactual_configuration_sha256": counterfactual_manifest.get(
            "configuration_sha256"
        ),
        "endpoint_report_version": endpoint_report.get("report_version"),
        "total_edit_samples": len(pair_results),
        "successful_edit_samples": successful,
        "success_rate": successful / len(pair_results) if pair_results else None,
        "total_physical_endpoints": len(endpoint_agreement),
        "encoding_agreeing_physical_endpoints": endpoint_pairs_agree,
        "endpoint_encoding_agreement": [
            {"source_family_id": source_id, **value}
            for source_id, value in sorted(endpoint_agreement.items())
        ],
        "edit_samples": pair_results,
    }


def _expected_endpoints(samples):
    by_family = {}
    edit_sample_ids = set()
    endpoint_samples = {}
    by_source = {}
    for sample in samples:
        if not isinstance(sample, dict):
            raise PairKernelAuditError("counterfactual samples must be objects")
        required = (
            "edit_sample_id",
            "edit_family_id",
            "geometry_encoding",
            "base_source_family_id",
            "edited_source_family_id",
            "base_sample_id",
            "edited_sample_id",
        )
        if any(not isinstance(sample.get(key), str) or not sample[key] for key in required):
            raise PairKernelAuditError("counterfactual sample metadata is incomplete")
        edit_sample_id = sample["edit_sample_id"]
        if edit_sample_id in edit_sample_ids:
            raise PairKernelAuditError("duplicate edit_sample_id")
        edit_sample_ids.add(edit_sample_id)
        encoding = sample["geometry_encoding"]
        if encoding not in ("continuous", "quantized"):
            raise PairKernelAuditError("unknown counterfactual geometry encoding")
        variants = by_family.setdefault(sample["edit_family_id"], {})
        if encoding in variants:
            raise PairKernelAuditError("duplicate edit-family encoding variant")
        variants[encoding] = sample

        for role in ("base", "edited"):
            source_id = sample[f"{role}_source_family_id"]
            sample_id = sample[f"{role}_sample_id"]
            expected = {
                "source_family_id": source_id,
                "geometry_encoding": encoding,
            }
            previous = endpoint_samples.setdefault(sample_id, expected)
            if previous != expected:
                raise PairKernelAuditError(
                    "endpoint sample ID maps to conflicting metadata"
                )
            source_variants = by_source.setdefault(source_id, {})
            previous_sample_id = source_variants.setdefault(encoding, sample_id)
            if previous_sample_id != sample_id:
                raise PairKernelAuditError(
                    "physical endpoint has conflicting encoding sample IDs"
                )

    for family_id, variants in by_family.items():
        if set(variants) != {"continuous", "quantized"}:
            raise PairKernelAuditError(
                f"edit family {family_id} lacks one encoding variant"
            )
        continuous = variants["continuous"]
        quantized = variants["quantized"]
        for field in (
            "base_source_family_id",
            "edited_source_family_id",
        ):
            if continuous[field] != quantized[field]:
                raise PairKernelAuditError(
                    f"edit family {family_id} has inconsistent endpoint ordering"
                )
    for source_id, variants in by_source.items():
        if set(variants) != {"continuous", "quantized"}:
            raise PairKernelAuditError(
                f"physical endpoint {source_id} lacks one encoding variant"
            )
    if len(endpoint_samples) != 4 * len(by_family):
        raise PairKernelAuditError("endpoint sample IDs are not globally unique")
    if len(by_source) != 2 * len(by_family):
        raise PairKernelAuditError("physical endpoints are not globally disjoint")
    return endpoint_samples, by_source, by_family


def publish_pair_execution_audit(output_dir: str | Path, audit: dict[str, Any]) -> None:
    final = Path(output_dir)
    if final.exists():
        raise PairKernelAuditError("output directory already exists")
    if not final.parent.exists():
        raise PairKernelAuditError("output parent does not exist")
    temporary = Path(tempfile.mkdtemp(prefix=f".{final.name}.tmp-", dir=final.parent))
    try:
        path = temporary / "pair_execution_audit.json"
        payload = json.dumps(
            audit,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ) + "\n"
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if json.loads(path.read_text(encoding="utf-8")) != audit:
            raise PairKernelAuditError("written pair audit failed verification")
        os.rename(temporary, final)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _metrics_agree(left, right) -> bool:
    if not isinstance(left, dict) or not isinstance(right, dict):
        return False
    for field in ("solid_count", "face_count", "edge_count", "vertex_count"):
        if left.get(field) != right.get(field):
            return False
    if not _close(
        left.get("volume"),
        right.get("volume"),
        PAIRED_VOLUME_ABSOLUTE_TOLERANCE,
        PAIRED_VOLUME_RELATIVE_TOLERANCE,
    ):
        return False
    a = left.get("bounding_box")
    b = right.get("bounding_box")
    if not isinstance(a, list) or not isinstance(b, list) or len(a) != len(b):
        return False
    return all(
        _close(
            x,
            y,
            PAIRED_BOUNDING_BOX_ABSOLUTE_TOLERANCE,
            PAIRED_BOUNDING_BOX_RELATIVE_TOLERANCE,
        )
        for x, y in zip(a, b)
    )


def _close(left, right, absolute, relative) -> bool:
    return (
        isinstance(left, (int, float))
        and not isinstance(left, bool)
        and isinstance(right, (int, float))
        and not isinstance(right, bool)
        and math.isfinite(float(left))
        and math.isfinite(float(right))
        and math.isclose(float(left), float(right), abs_tol=absolute, rel_tol=relative)
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counterfactual-corpus", required=True)
    parser.add_argument("--endpoint-report", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    try:
        corpus_root = Path(args.counterfactual_corpus)
        manifest = json.loads(
            (corpus_root / "counterfactual_manifest.json").read_text(encoding="utf-8")
        )
        report = json.loads(Path(args.endpoint_report).read_text(encoding="utf-8"))
        audit = build_pair_execution_audit(manifest, report)
        publish_pair_execution_audit(args.output_dir, audit)
    except Exception as exc:
        print(f"pair kernel audit failed: {type(exc).__name__}: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
