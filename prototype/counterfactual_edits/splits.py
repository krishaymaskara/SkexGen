"""Authoritative edit-family splits aligned with controlled-data policies."""

from __future__ import annotations

from collections import Counter
import hashlib
from typing import Any

from prototype.controlled_data.splits import (
    OPERATION_SECONDARY_VALIDATION_TEMPLATES,
    OPERATION_TEST_TEMPLATES,
    OPERATION_TRAIN_VALIDATION_TEMPLATES,
    PARTITION_ORDER,
    SPLIT_POLICY_VERSION,
    geometry_extent_partition_class,
    history_depth_partition_class,
    largest_remainder_counts,
    operation_template_partition_class,
)

from .config import CounterfactualConfig, EditConfigurationError


def build_split_manifests(
    families: list[dict[str, Any]], config: CounterfactualConfig
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    ids = [item["edit_family_id"] for item in families]
    if len(ids) != len(set(ids)):
        raise EditConfigurationError("edit_family_id values must be unique")
    iid = _ranked(
        ids, config.iid_ratios, PARTITION_ORDER, "counterfactual-iid-v1", config.seed
    )
    operation = _factor_assignment(
        families,
        config,
        lambda item: operation_template_partition_class(item["operation_template"]),
        "counterfactual-operation-template-v1",
    )
    depth = _factor_assignment(
        families,
        config,
        lambda item: history_depth_partition_class(item["history_depth"]),
        "counterfactual-history-depth-v1",
    )
    geometry = _factor_assignment(
        families,
        config,
        lambda item: geometry_extent_partition_class(
            item["history_max_sketch_extent"], config.factor_config
        ),
        "counterfactual-geometry-v1",
    )
    definitions = {
        "iid": {
            "kind": "iid",
            "ratios": list(config.iid_ratios),
            "exact_partition_sizes": True,
        },
        "operation_template": {
            "kind": "held_out_operation_template",
            "train_validation_templates": list(OPERATION_TRAIN_VALIDATION_TEMPLATES),
            "test_templates": list(OPERATION_TEST_TEMPLATES),
            "secondary_systematic_validation_templates": list(
                OPERATION_SECONDARY_VALIDATION_TEMPLATES
            ),
        },
        "history_depth": {
            "kind": "held_out_history_depth",
            "benchmark_status": "provisional",
            "train_validation_depths": [1],
            "test_depths": [2],
        },
        "geometry_extrapolation": {
            "kind": "held_out_geometry_range",
            "target_statistic": "history_max_sketch_extent",
            "train_validation_max": config.factor_config.in_range_extent_max,
            "test_min": config.factor_config.extrapolation_extent_min,
        },
    }
    assignments = {
        "iid": iid,
        "operation_template": operation,
        "history_depth": depth,
        "geometry_extrapolation": geometry,
    }
    manifests = {
        name: _manifest(name, families, assignment, definitions[name])
        for name, assignment in assignments.items()
    }
    exclusions = _endpoint_exclusions(families, assignments)
    _verify_assignments(families, manifests)
    return manifests, exclusions


def _factor_assignment(
    families,
    config,
    classify,
    label,
):
    eligible = [
        item["edit_family_id"]
        for item in families
        if classify(item) == "train_validation"
    ]
    assignment = _ranked(
        eligible,
        (1.0 - config.validation_ratio, config.validation_ratio),
        ("train", "validation"),
        label,
        config.seed,
    )
    for item in families:
        family_id = item["edit_family_id"]
        partition_class = classify(item)
        if partition_class == "test":
            assignment[family_id] = "test"
        elif partition_class == "secondary_systematic_validation":
            assignment[family_id] = "secondary_systematic_validation"
    return assignment


def _ranked(ids, ratios, partitions, label, seed):
    ranked = sorted(
        ids,
        key=lambda item: (
            hashlib.sha256(
                f"{label}\0{seed}\0{item}".encode("utf-8")
            ).digest(),
            item,
        ),
    )
    counts = largest_remainder_counts(len(ranked), ratios)
    result = {}
    cursor = 0
    for partition, count in zip(partitions, counts):
        for family_id in ranked[cursor : cursor + count]:
            result[family_id] = partition
        cursor += count
    return result


def _manifest(name, families, assignment, definition):
    family_records = []
    sample_records = []
    endpoint_records = []
    for family in sorted(families, key=lambda item: item["edit_family_id"]):
        partition = assignment[family["edit_family_id"]]
        family_record = {
            key: value for key, value in family.items() if key != "samples"
        }
        family_record["partition"] = partition
        family_records.append(family_record)
        for sample in family["samples"]:
            sample_records.append({**sample, "partition": partition})
        for role in ("base", "edited"):
            endpoint_records.append(
                {
                    "edit_family_id": family["edit_family_id"],
                    "role": role,
                    "source_family_id": family[f"{role}_source_family_id"],
                    "partition": partition,
                }
            )
    return {
        "manifest_version": 1,
        "split_policy_version": SPLIT_POLICY_VERSION,
        "name": name,
        "definition": definition,
        "authoritative_assignment_unit": "edit_family_id",
        "partition_counts": dict(sorted(Counter(assignment.values()).items())),
        "families": family_records,
        "samples": sorted(sample_records, key=lambda item: item["edit_sample_id"]),
        "endpoints": sorted(
            endpoint_records,
            key=lambda item: (
                item["source_family_id"],
                item["edit_family_id"],
                item["role"],
            ),
        ),
    }


def _endpoint_exclusions(families, assignments):
    lookup = {item["edit_family_id"]: item for item in families}
    strategies = {}
    union = set()
    evaluation_partitions = (
        "validation",
        "test",
        "secondary_systematic_validation",
    )
    for name, assignment in sorted(assignments.items()):
        records = {}
        for partition in evaluation_partitions:
            endpoint_ids = sorted(
                {
                    lookup[family_id][f"{role}_source_family_id"]
                    for family_id, assigned in assignment.items()
                    if assigned == partition
                    for role in ("base", "edited")
                }
            )
            if endpoint_ids:
                records[partition] = endpoint_ids
                union.update(endpoint_ids)
        strategies[name] = records
    return {
        "manifest_version": 1,
        "split_policy_version": SPLIT_POLICY_VERSION,
        "strategies": strategies,
        "all_evaluation_endpoint_source_family_ids": sorted(union),
    }


def _verify_assignments(families, manifests):
    expected = {item["edit_family_id"] for item in families}
    endpoints = {
        item[f"{role}_source_family_id"]
        for item in families
        for role in ("base", "edited")
    }
    if len(endpoints) != 2 * len(families):
        raise EditConfigurationError("endpoint-disjoint split input is violated")
    for name, manifest in manifests.items():
        authoritative = {
            item["edit_family_id"]: item["partition"]
            for item in manifest["families"]
        }
        if set(authoritative) != expected:
            raise EditConfigurationError(f"{name} does not assign every edit family")
        for sample in manifest["samples"]:
            if sample["partition"] != authoritative[sample["edit_family_id"]]:
                raise EditConfigurationError(f"{name} sample assignment disagrees")
        for endpoint in manifest["endpoints"]:
            if endpoint["partition"] != authoritative[endpoint["edit_family_id"]]:
                raise EditConfigurationError(f"{name} endpoint assignment disagrees")
