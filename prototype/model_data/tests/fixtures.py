"""Small deterministic on-disk corpora for model-data tests."""

from __future__ import annotations

import json
from pathlib import Path

from prototype.controlled_data.builders import build_history
from prototype.controlled_data.factors import (
    ExtentBand,
    OperationTemplate,
    PhysicalSource,
    PrimitiveFamily,
    ReferencePlane,
)
from prototype.controlled_data.identity import sample_id, source_family_id
from prototype.counterfactual_edits.edits import before_after, target_for
from prototype.counterfactual_edits.identity import edit_family_id, edit_sample_id
from prototype.counterfactual_edits.locality import build_locality_ground_truth
from prototype.counterfactual_edits.model import EditSample, EditType
from prototype.counterfactual_edits.serialization import edit_sample_to_json
from prototype.representation.model import BooleanMode, Direction, GeometryEncoding
from prototype.representation.serialization import history_to_json


def source(
    template,
    primitive=PrimitiveFamily.RECTANGLE_LINES,
    plane=ReferencePlane.XY,
    extents=None,
    directions=None,
    mode=BooleanMode.JOIN,
    parameters=None,
):
    operation_template = OperationTemplate(template)
    depth = len(operation_template.operations)
    return PhysicalSource(
        operation_template,
        primitive,
        plane,
        tuple(extents or ((1.0,) * depth)),
        tuple(directions or ((Direction.POSITIVE,) * depth)),
        None if depth == 1 else mode,
        tuple(
            parameters
            or tuple(
                1.0 if item == "extrude" else 90.0
                for item in operation_template.operations
            )
        ),
        ExtentBand.IN_RANGE,
    )


def write_physical_corpus(root, sources, partitions=None, directory="samples"):
    root = Path(root)
    (root / directory).mkdir(parents=True)
    (root / "manifests").mkdir()
    samples = []
    families = []
    partitions = partitions or {}
    assignments = {}
    for physical in sources:
        variants = []
        family_id = None
        for encoding in (GeometryEncoding.CONTINUOUS, GeometryEncoding.QUANTIZED):
            history = build_history(physical, encoding)
            current_family = source_family_id(history)
            family_id = current_family if family_id is None else family_id
            assert current_family == family_id
            current_sample = sample_id(history)
            relative = f"{directory}/{current_sample}.json"
            (root / relative).write_text(history_to_json(history) + "\n")
            record = {
                "source_family_id": family_id,
                "sample_id": current_sample,
                "geometry_encoding": encoding.value,
                "operation_template": physical.operation_template.value,
                "primitive_family": physical.primitive_family.value,
                "relative_json_path": relative,
                "schema_valid": True,
                "serialization_round_trip_valid": True,
                "kernel_status": "not_checked",
            }
            variants.append(record)
            samples.append(record)
        partition = partitions.get(family_id, "train")
        assignments[family_id] = partition
        families.append(
            {
                "source_family_id": family_id,
                "sample_ids": sorted(item["sample_id"] for item in variants),
                "schema_valid": True,
                "serialization_round_trip_valid": True,
                "kernel_status": "not_checked",
            }
        )
    _write(
        root / "corpus_manifest.json",
        {
            "total_source_family_count": len(families),
            "total_sample_variant_count": len(samples),
            "families": families,
            "samples": sorted(samples, key=lambda item: item["sample_id"]),
        },
    )
    split_samples = [
        {**item, "partition": assignments[item["source_family_id"]]}
        for item in samples
    ]
    split_families = [
        {
            **item,
            "partition": assignments[item["source_family_id"]],
        }
        for item in families
    ]
    _write(
        root / "manifests" / "iid.json",
        {
            "manifest_version": 1,
            "name": "iid",
            "authoritative_assignment_unit": "source_family_id",
            "families": split_families,
            "samples": split_samples,
        },
    )
    return families, samples


def write_counterfactual_corpus(root):
    base = source("E", extents=(1.0,))
    edited = source("E", extents=(1.5,))
    write_physical_corpus(root, (base, edited), directory="histories")
    root = Path(root)
    edit_type = EditType.PROFILE_EXTENT
    target = target_for(edit_type, base, 0)
    base_continuous = build_history(base, GeometryEncoding.CONTINUOUS)
    edited_continuous = build_history(edited, GeometryEncoding.CONTINUOUS)
    base_family = source_family_id(base_continuous)
    edited_family = source_family_id(edited_continuous)
    family_id = edit_family_id(edit_type, target, base_family, edited_family)
    locality = build_locality_ground_truth(
        base_continuous, edited_continuous, edit_type, target
    )
    before, after = before_after(edit_type, base, edited, 0)
    pair_records = []
    endpoint_records = []
    for encoding in (GeometryEncoding.CONTINUOUS, GeometryEncoding.QUANTIZED):
        base_history = build_history(base, encoding)
        edited_history = build_history(edited, encoding)
        base_sample = sample_id(base_history)
        edited_sample = sample_id(edited_history)
        current_edit_id = edit_sample_id(
            family_id, encoding.value, base_sample, edited_sample
        )
        pair = EditSample(
            current_edit_id,
            family_id,
            encoding.value,
            base_family,
            edited_family,
            base_sample,
            edited_sample,
            edit_type,
            target,
            before,
            after,
            locality,
        )
        relative = f"pairs/{current_edit_id}.json"
        (root / "pairs").mkdir(exist_ok=True)
        (root / relative).write_text(edit_sample_to_json(pair) + "\n")
        pair_records.append(
            {
                "edit_sample_id": current_edit_id,
                "edit_family_id": family_id,
                "geometry_encoding": encoding.value,
                "base_source_family_id": base_family,
                "edited_source_family_id": edited_family,
                "base_sample_id": base_sample,
                "edited_sample_id": edited_sample,
                "relative_json_path": relative,
                "schema_valid": True,
                "serialization_round_trip_valid": True,
                "kernel_status": "not_checked",
            }
        )
    for role, source_id in (("base", base_family), ("edited", edited_family)):
        endpoint_records.append(
            {
                "edit_family_id": family_id,
                "role": role,
                "source_family_id": source_id,
                "partition": "test",
            }
        )
    _write(
        root / "counterfactual_manifest.json",
        {
            "families": [{"edit_family_id": family_id}],
            "samples": pair_records,
        },
    )
    _write(
        root / "manifests" / "iid.json",
        {
            "manifest_version": 1,
            "name": "iid",
            "authoritative_assignment_unit": "edit_family_id",
            "families": [{"edit_family_id": family_id, "partition": "test"}],
            "samples": [
                {**item, "partition": "test"} for item in pair_records
            ],
            "endpoints": endpoint_records,
        },
    )
    return family_id, base_family, edited_family


def _write(path, value):
    path.write_text(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )
