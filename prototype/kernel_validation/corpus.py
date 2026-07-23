"""Read an immutable controlled-data corpus without importing a CAD kernel."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .model import SampleInput


class CorpusError(ValueError):
    """The corpus manifest cannot be used safely."""


def load_corpus(corpus_dir: str | Path) -> tuple[dict[str, Any], tuple[SampleInput, ...]]:
    root = Path(corpus_dir)
    manifest_path = root / "corpus_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise CorpusError(f"cannot read corpus_manifest.json: {type(exc).__name__}") from exc
    if not isinstance(manifest, dict) or not isinstance(manifest.get("samples"), list):
        raise CorpusError("corpus manifest must contain a samples list")

    results = []
    seen_ids: set[str] = set()
    for raw in sorted(manifest["samples"], key=lambda item: str(item.get("sample_id", ""))):
        if not isinstance(raw, dict):
            raise CorpusError("each corpus sample record must be an object")
        required = (
            "source_family_id",
            "sample_id",
            "geometry_encoding",
            "operation_template",
            "primitive_family",
            "relative_json_path",
        )
        if any(not isinstance(raw.get(key), str) or not raw[key] for key in required):
            raise CorpusError("sample record has missing or invalid string fields")
        if raw["sample_id"] in seen_ids:
            raise CorpusError(f"duplicate sample record {raw['sample_id']}")
        seen_ids.add(raw["sample_id"])
        relative = Path(raw["relative_json_path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise CorpusError(f"unsafe relative_json_path for {raw['sample_id']}")
        payload: str | None = None
        loading_error: str | None = None
        try:
            payload = (root / relative).read_text(encoding="utf-8")
            payload = payload[:-1] if payload.endswith("\n") else payload
        except Exception as exc:
            loading_error = f"cannot read sample file: {type(exc).__name__}"
        results.append(
            SampleInput(
                source_family_id=raw["source_family_id"],
                sample_id=raw["sample_id"],
                geometry_encoding=raw["geometry_encoding"],
                operation_template=raw["operation_template"],
                primitive_family=raw["primitive_family"],
                relative_json_path=raw["relative_json_path"],
                payload=payload,
                loading_error=loading_error,
            )
        )
    metadata = {
        key: manifest.get(key)
        for key in (
            "generator_version",
            "representation_schema_version",
            "canonicalization_version",
            "configuration_sha256",
            "generation_seed",
            "total_source_family_count",
            "total_sample_variant_count",
        )
    }
    return metadata, tuple(results)
