"""Authoritative source-family split assignment and coverage auditing."""

from __future__ import annotations

import hashlib
import math
from typing import Any

from .config import ConfigurationError, GeneratorConfig


PARTITION_ORDER = ("train", "validation", "test")
SPLIT_POLICY_VERSION = "controlled-data-splits-v1"
OPERATION_TRAIN_VALIDATION_TEMPLATES = ("E", "R", "EE", "RE")
OPERATION_TEST_TEMPLATES = ("ER",)
OPERATION_SECONDARY_VALIDATION_TEMPLATES = ("RR",)


def operation_template_partition_class(template: str) -> str:
    """Return the non-IID partition class frozen by the controlled benchmark."""

    if template in OPERATION_TRAIN_VALIDATION_TEMPLATES:
        return "train_validation"
    if template in OPERATION_TEST_TEMPLATES:
        return "test"
    if template in OPERATION_SECONDARY_VALIDATION_TEMPLATES:
        return "secondary_systematic_validation"
    raise ConfigurationError(f"unknown operation template {template!r}")


def history_depth_partition_class(depth: int) -> str:
    if isinstance(depth, bool) or not isinstance(depth, int):
        raise ConfigurationError("history depth must be an integer")
    if depth == 1:
        return "train_validation"
    if depth == 2:
        return "test"
    raise ConfigurationError(f"unsupported history depth {depth!r}")


def geometry_extent_partition_class(
    maximum_extent: float, config: GeneratorConfig
) -> str:
    if maximum_extent <= config.in_range_extent_max:
        return "train_validation"
    if maximum_extent >= config.extrapolation_extent_min:
        return "test"
    raise ConfigurationError("history extent lies in the excluded geometry gap")


def build_split_manifests(
    families: list[dict[str, Any]], config: GeneratorConfig
) -> dict[str, dict[str, Any]]:
    """Build all manifests from one immutable family/sample corpus."""

    family_ids = [item["source_family_id"] for item in families]
    if len(family_ids) != len(set(family_ids)):
        raise ConfigurationError("source_family_id values must be unique before splitting")

    iid_assignment = _ranked_partition(
        family_ids,
        config.iid_ratios,
        PARTITION_ORDER,
        "iid",
        config.seed,
    )
    operation_assignment = _operation_template_assignment(families, config)
    depth_assignment = _depth_assignment(families, config)
    geometry_assignment = _geometry_assignment(families, config)

    manifests = {
        "iid": _manifest(
            "iid",
            families,
            iid_assignment,
            {"kind": "iid", "ratios": list(config.iid_ratios), "exact_partition_sizes": True},
            _coverage_report(families, iid_assignment),
        ),
        "operation_template": _manifest(
            "operation_template",
            families,
            operation_assignment,
            {
                "kind": "held_out_operation_template",
                "train_validation_templates": ["E", "R", "EE", "RE"],
                "test_templates": ["ER"],
                "secondary_systematic_validation_templates": ["RR"],
            },
            _coverage_report(families, operation_assignment),
        ),
        "history_depth": _manifest(
            "history_depth",
            families,
            depth_assignment,
            {
                "kind": "held_out_history_depth",
                "benchmark_status": "provisional",
                "train_validation_depths": [1],
                "test_depths": [2],
            },
            _coverage_report(families, depth_assignment),
        ),
        "geometry_extrapolation": _manifest(
            "geometry_extrapolation",
            families,
            geometry_assignment,
            {
                "kind": "held_out_geometry_range",
                "target_statistic": "history_max_sketch_extent",
                "train_validation_max": config.in_range_extent_max,
                "test_min": config.extrapolation_extent_min,
            },
            _coverage_report(families, geometry_assignment),
        ),
    }
    _assert_all_family_assignments(manifests, families)
    _assert_held_out_rules(families, manifests, config)
    return manifests


def largest_remainder_counts(total: int, ratios: tuple[float, ...]) -> tuple[int, ...]:
    """Allocate an exact integer total with deterministic index-order ties."""

    if isinstance(total, bool) or not isinstance(total, int) or total < 0:
        raise ConfigurationError("partition total must be a nonnegative integer")
    if not ratios or any(not math.isfinite(item) or item < 0 for item in ratios):
        raise ConfigurationError("partition ratios must be finite and nonnegative")
    if not math.isclose(sum(ratios), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ConfigurationError("partition ratios must sum to 1")
    quotas = [total * item for item in ratios]
    counts = [math.floor(item) for item in quotas]
    remaining = total - sum(counts)
    order = sorted(range(len(ratios)), key=lambda i: (-(quotas[i] - counts[i]), i))
    for index in order[:remaining]:
        counts[index] += 1
    return tuple(counts)


def _ranked_partition(
    family_ids: list[str],
    ratios: tuple[float, ...],
    partitions: tuple[str, ...],
    label: str,
    seed: int,
) -> dict[str, str]:
    if len(ratios) != len(partitions):
        raise ConfigurationError("partition names and ratios must have equal length")
    ranked = sorted(
        family_ids,
        key=lambda family_id: (_rank_digest(label, seed, family_id), family_id),
    )
    counts = largest_remainder_counts(len(ranked), ratios)
    assignment: dict[str, str] = {}
    cursor = 0
    for partition, count in zip(partitions, counts):
        for family_id in ranked[cursor : cursor + count]:
            assignment[family_id] = partition
        cursor += count
    return assignment


def _operation_template_assignment(
    families: list[dict[str, Any]], config: GeneratorConfig
) -> dict[str, str]:
    eligible = [
        item["source_family_id"]
        for item in families
        if operation_template_partition_class(item["operation_template"])
        == "train_validation"
    ]
    assignment = _ranked_partition(
        eligible,
        (1.0 - config.validation_ratio, config.validation_ratio),
        ("train", "validation"),
        "operation-template-in-distribution",
        config.seed,
    )
    for item in families:
        family_id = item["source_family_id"]
        partition_class = operation_template_partition_class(item["operation_template"])
        if partition_class == "test":
            assignment[family_id] = "test"
        elif partition_class == "secondary_systematic_validation":
            assignment[family_id] = "secondary_systematic_validation"
    return assignment


def _depth_assignment(
    families: list[dict[str, Any]], config: GeneratorConfig
) -> dict[str, str]:
    depth_one = [
        item["source_family_id"]
        for item in families
        if history_depth_partition_class(item["history_depth"]) == "train_validation"
    ]
    assignment = _ranked_partition(
        depth_one,
        (1.0 - config.validation_ratio, config.validation_ratio),
        ("train", "validation"),
        "history-depth-in-distribution",
        config.seed,
    )
    assignment.update(
        {
            item["source_family_id"]: "test"
            for item in families
            if history_depth_partition_class(item["history_depth"]) == "test"
        }
    )
    return assignment


def _geometry_assignment(
    families: list[dict[str, Any]], config: GeneratorConfig
) -> dict[str, str]:
    in_range = [
        item["source_family_id"]
        for item in families
        if geometry_extent_partition_class(item["history_max_sketch_extent"], config)
        == "train_validation"
    ]
    assignment = _ranked_partition(
        in_range,
        (1.0 - config.validation_ratio, config.validation_ratio),
        ("train", "validation"),
        "geometry-in-distribution",
        config.seed,
    )
    assignment.update(
        {
            item["source_family_id"]: "test"
            for item in families
            if geometry_extent_partition_class(item["history_max_sketch_extent"], config)
            == "test"
        }
    )
    return assignment


def _manifest(
    name: str,
    families: list[dict[str, Any]],
    assignment: dict[str, str],
    definition: dict[str, Any],
    coverage: dict[str, Any],
) -> dict[str, Any]:
    family_records = []
    sample_records = []
    for family in sorted(families, key=lambda item: item["source_family_id"]):
        family_id = family["source_family_id"]
        partition = assignment[family_id]
        common = {
            "source_family_id": family_id,
            "partition": partition,
            "schema_valid": True,
            "serialization_round_trip_valid": True,
            "kernel_status": "not_checked",
        }
        family_metadata = {
            key: value
            for key, value in family.items()
            if key not in {"samples", "selection_order"}
        }
        family_records.append({**family_metadata, **common})
        for sample in family["samples"]:
            sample_records.append({**sample, "partition": partition})
    return {
        "manifest_version": 1,
        "name": name,
        "definition": definition,
        "authoritative_assignment_unit": "source_family_id",
        "partition_counts": _counts(assignment.values()),
        "coverage": coverage,
        "families": family_records,
        "samples": sorted(sample_records, key=lambda item: item["sample_id"]),
    }


def _coverage_report(
    families: list[dict[str, Any]], assignment: dict[str, str]
) -> dict[str, Any]:
    by_partition: dict[str, list[dict[str, Any]]] = {}
    for family in families:
        by_partition.setdefault(assignment[family["source_family_id"]], []).append(family)
    return {
        partition: {
            "source_family_count": len(items),
            "primitive_families": _unique(items, "primitive_family"),
            "reference_planes": _unique(items, "reference_plane"),
            "directions": _flattened_unique(items, "operation_directions"),
            "later_boolean_modes": sorted(
                {
                    mode
                    for item in items
                    for mode in item["boolean_modes"][1:]
                }
            ),
            "sketch_extent_bands": _unique(items, "sketch_extent_band"),
            "extrusion_distances": _flattened_unique(items, "extrusion_distances"),
            "revolution_angles": _flattened_unique(items, "revolution_angles"),
            "extent_order_relations": _unique(items, "extent_order_relation"),
            "direction_relations": _unique(items, "direction_relation"),
            "boolean_extent_relations": sorted(
                {
                    f"{item['boolean_modes'][1]}:{item['extent_order_relation']}"
                    for item in items
                    if len(item["boolean_modes"]) == 2
                }
            ),
            "boolean_direction_relations": sorted(
                {
                    f"{item['boolean_modes'][1]}:{item['direction_relation']}"
                    for item in items
                    if len(item["boolean_modes"]) == 2
                }
            ),
        }
        for partition, items in sorted(by_partition.items())
    }


def _assert_all_family_assignments(
    manifests: dict[str, dict[str, Any]], families: list[dict[str, Any]]
) -> None:
    expected = {item["source_family_id"] for item in families}
    for name, manifest in manifests.items():
        records = manifest["families"]
        assigned = [item["source_family_id"] for item in records]
        if len(assigned) != len(set(assigned)) or set(assigned) != expected:
            raise ConfigurationError(f"{name} does not assign every source family exactly once")
        authoritative = {item["source_family_id"]: item["partition"] for item in records}
        for sample in manifest["samples"]:
            if sample["partition"] != authoritative[sample["source_family_id"]]:
                raise ConfigurationError(f"{name} has a sample/family partition disagreement")


def _assert_held_out_rules(
    families: list[dict[str, Any]], manifests: dict[str, dict[str, Any]], config: GeneratorConfig
) -> None:
    lookup = {item["source_family_id"]: item for item in families}
    assignments = {
        name: {item["source_family_id"]: item["partition"] for item in manifest["families"]}
        for name, manifest in manifests.items()
    }
    operation = assignments["operation_template"]
    for family_id, partition in operation.items():
        template = lookup[family_id]["operation_template"]
        allowed = {
            "train": {"E", "R", "EE", "RE"},
            "validation": {"E", "R", "EE", "RE"},
            "test": {"ER"},
            "secondary_systematic_validation": {"RR"},
        }[partition]
        if template not in allowed:
            raise ConfigurationError("operation-template split violates its held-out template")

    depth = assignments["history_depth"]
    for family_id, partition in depth.items():
        expected_depth = 2 if partition == "test" else 1
        if lookup[family_id]["history_depth"] != expected_depth:
            raise ConfigurationError("history-depth split violates its held-out depth")

    geometry = assignments["geometry_extrapolation"]
    for family_id, partition in geometry.items():
        extent = lookup[family_id]["history_max_sketch_extent"]
        if partition == "test" and extent < config.extrapolation_extent_min:
            raise ConfigurationError("geometry test family is not in the extrapolation range")
        if partition != "test" and extent > config.in_range_extent_max:
            raise ConfigurationError("held-out geometry value leaked into train/validation")

    if config.require_full_coverage:
        operation_expected = {
            "primitive_families": {"rectangle_lines", "circle", "capsule_line_arc"},
            "reference_planes": {"XY", "XZ", "YZ"},
            "directions": {"positive", "negative"},
            "later_boolean_modes": {"join", "cut"},
            "sketch_extent_bands": {"in_range", "extrapolation"},
            "extrusion_distances": set(config.extrusion_distances),
            "revolution_angles": set(config.revolution_angles),
        }
        _require_partition_coverage(
            manifests["operation_template"], ("train", "test"), operation_expected
        )
        depth_expected = dict(operation_expected)
        depth_expected.pop("later_boolean_modes")
        _require_partition_coverage(manifests["history_depth"], ("train", "test"), depth_expected)
        geometry_expected = dict(operation_expected)
        geometry_expected.pop("sketch_extent_bands")
        _require_partition_coverage(
            manifests["geometry_extrapolation"], ("train", "test"), geometry_expected
        )


def _require_partition_coverage(
    manifest: dict[str, Any], partitions: tuple[str, ...], expected: dict[str, set[Any]]
) -> None:
    coverage = manifest["coverage"]
    for partition in partitions:
        if partition not in coverage:
            raise ConfigurationError(f"{manifest['name']} has no {partition} partition")
        for factor, required in expected.items():
            observed = set(coverage[partition][factor])
            if not required.issubset(observed):
                missing = sorted(required - observed)
                raise ConfigurationError(
                    f"{manifest['name']} {partition} lacks required {factor}: {missing!r}; "
                    "increase num_source_families or revise the factor grid"
                )


def _rank_digest(label: str, seed: int, family_id: str) -> str:
    return hashlib.sha256(
        f"controlled-data-split-v1\0{label}\0{seed}\0{family_id}".encode("utf-8")
    ).hexdigest()


def _counts(values) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        result[value] = result.get(value, 0) + 1
    return {key: result[key] for key in sorted(result)}


def _unique(items: list[dict[str, Any]], field: str) -> list[Any]:
    return sorted({item[field] for item in items})


def _flattened_unique(items: list[dict[str, Any]], field: str) -> list[Any]:
    return sorted({value for item in items for value in item[field]})
