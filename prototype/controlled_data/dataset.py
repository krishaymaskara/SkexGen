"""Validated corpus assembly, deterministic manifests, and atomic publication."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Callable

from prototype.representation.model import CADHistory, GeometryEncoding
from prototype.representation.serialization import history_from_json, history_to_json
from prototype.representation.validation import validate_history

from .builders import build_history
from .config import (
    CANONICALIZATION_VERSION,
    GENERATOR_VERSION,
    REPRESENTATION_SCHEMA_VERSION,
    ConfigurationError,
    GeneratorConfig,
)
from .factors import PhysicalSource, select_sources, total_candidate_count
from .identity import canonical_physical_source_bytes, sample_id, source_family_id
from .splits import build_split_manifests


class GenerationError(RuntimeError):
    """Corpus generation failed before atomic publication."""


def generate_corpus(
    output_dir: str | os.PathLike[str],
    config: GeneratorConfig,
    *,
    history_builder: Callable[[PhysicalSource, GeometryEncoding], CADHistory] = build_history,
) -> dict[str, Any]:
    """Generate exactly two variants per family and publish them atomically."""

    config.validate()
    final_path = Path(output_dir)
    if final_path.exists():
        raise ConfigurationError(f"output directory already exists: {final_path}")
    parent = final_path.parent
    if not parent.exists():
        raise ConfigurationError(f"output parent does not exist: {parent}")
    temporary = Path(tempfile.mkdtemp(prefix=f".{final_path.name}.tmp-", dir=parent))
    try:
        samples_dir = temporary / "samples"
        manifests_dir = temporary / "manifests"
        samples_dir.mkdir()
        manifests_dir.mkdir()

        sources = select_sources(config)
        families: list[dict[str, Any]] = []
        sample_records: list[dict[str, Any]] = []
        seen_sample_ids: set[str] = set()
        seen_family_ids: set[str] = set()

        for selection_index, source in enumerate(sources):
            variants = []
            family_id: str | None = None
            physical_bytes: bytes | None = None
            for encoding in (GeometryEncoding.CONTINUOUS, GeometryEncoding.QUANTIZED):
                history = history_builder(source, encoding)
                validate_history(history)
                payload = history_to_json(history)
                restored = history_from_json(payload)
                validate_history(restored)
                if history_to_json(restored) != payload:
                    raise GenerationError("canonical serialization round trip changed bytes")
                current_physical = canonical_physical_source_bytes(history)
                current_family_id = source_family_id(history)
                if family_id is None:
                    family_id = current_family_id
                    physical_bytes = current_physical
                elif family_id != current_family_id or physical_bytes != current_physical:
                    raise GenerationError("continuous and quantized variants disagree physically")
                current_sample_id = sample_id(history)
                if current_sample_id in seen_sample_ids:
                    raise GenerationError(f"duplicate sample ID {current_sample_id}")
                seen_sample_ids.add(current_sample_id)
                relative_path = f"samples/{current_sample_id}.json"
                _write_text(temporary / relative_path, payload + "\n")
                record = {
                    "source_family_id": current_family_id,
                    "sample_id": current_sample_id,
                    **source.physical_metadata(),
                    "geometry_encoding": encoding.value,
                    "schema_valid": True,
                    "serialization_round_trip_valid": True,
                    "kernel_status": "not_checked",
                    "relative_json_path": relative_path,
                }
                variants.append(record)
                sample_records.append(record)
            if family_id is None:
                raise GenerationError("source family produced no representation variants")
            if family_id in seen_family_ids:
                raise GenerationError(f"duplicate source family ID {family_id}")
            seen_family_ids.add(family_id)
            families.append(
                {
                    "source_family_id": family_id,
                    "selection_order": selection_index,
                    **source.physical_metadata(),
                    "sample_ids": sorted(item["sample_id"] for item in variants),
                    "samples": sorted(variants, key=lambda item: item["sample_id"]),
                    "schema_valid": True,
                    "serialization_round_trip_valid": True,
                    "kernel_status": "not_checked",
                }
            )

        if len(families) != config.num_source_families:
            raise GenerationError("source-family count does not match the request")
        if len(sample_records) != 2 * config.num_source_families:
            raise GenerationError("sample-variant count is not exactly twice the family count")

        split_manifests = build_split_manifests(families, config)
        for manifest in split_manifests.values():
            manifest["corpus_configuration_sha256"] = config.sha256()
        corpus_manifest = {
            "generator_version": GENERATOR_VERSION,
            "representation_schema_version": REPRESENTATION_SCHEMA_VERSION,
            "canonicalization_version": CANONICALIZATION_VERSION,
            "normalized_configuration": config.normalized(),
            "configuration_sha256": config.sha256(),
            "generation_seed": config.seed,
            "candidate_source_family_count": total_candidate_count(config),
            "total_source_family_count": len(families),
            "total_sample_variant_count": len(sample_records),
            "families": sorted(
                (
                    {key: value for key, value in family.items() if key != "samples"}
                    for family in families
                ),
                key=lambda item: item["source_family_id"],
            ),
            "samples": sorted(sample_records, key=lambda item: item["sample_id"]),
        }
        _write_json(temporary / "corpus_manifest.json", corpus_manifest)
        for name, manifest in split_manifests.items():
            _write_json(manifests_dir / f"{name}.json", manifest)
        _verify_written_tree(temporary, corpus_manifest, split_manifests)
        os.rename(temporary, final_path)
        return corpus_manifest
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _write_json(path: Path, value: Any) -> None:
    _write_text(
        path,
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
    )


def _write_text(path: Path, value: str) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(value)


def _verify_written_tree(
    root: Path,
    corpus_manifest: dict[str, Any],
    split_manifests: dict[str, dict[str, Any]],
) -> None:
    with (root / "corpus_manifest.json").open(encoding="utf-8") as handle:
        if json.load(handle) != corpus_manifest:
            raise GenerationError("written corpus manifest failed verification")
    for sample in corpus_manifest["samples"]:
        payload = (root / sample["relative_json_path"]).read_text(encoding="utf-8")
        canonical = payload[:-1] if payload.endswith("\n") else payload
        restored = history_from_json(canonical)
        if history_to_json(restored) != canonical or sample_id(restored) != sample["sample_id"]:
            raise GenerationError(f"written sample failed verification: {sample['sample_id']}")
        if source_family_id(restored) != sample["source_family_id"]:
            raise GenerationError(f"written source identity failed verification: {sample['sample_id']}")
    for name, expected in split_manifests.items():
        with (root / "manifests" / f"{name}.json").open(encoding="utf-8") as handle:
            if json.load(handle) != expected:
                raise GenerationError(f"written {name} manifest failed verification")
