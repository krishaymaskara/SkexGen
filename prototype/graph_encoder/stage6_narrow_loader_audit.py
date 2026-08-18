"""Static audit for the separately named ADR-0014 narrow loader."""

from __future__ import annotations

import ast
from pathlib import Path


AUDIT_VERSION = "GE1-STAGE6-NARROW-LOADER-AUDIT-v1"


def audit(repository_root="."):
    path = Path(repository_root) / "prototype/graph_encoder/stage6_narrow_loader.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path), feature_version=(3, 8))
    imports = {
        node.module for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    if "prototype.model_data.loader" in imports:
        raise AssertionError("narrow loader must not call the full-corpus loader")
    for required in (
        "load_stage6_train", "load_stage6_development",
        "canonical_physical_source_bytes", "history_from_json", "history_to_json",
        "source_family_id", "sample_id", "reconstruction_target",
        "unexpected_file_count", "payload_allowlist",
    ):
        if required not in source:
            raise AssertionError("missing narrow-loader control: " + required)
    return {
        "version": AUDIT_VERSION,
        "full_corpus_loader_imported": False,
        "partition_scoped_entry_points": 2,
        "identity_and_canonicalization_checks": True,
        "exact_allowlist_check": True,
    }


def main(argv=None):
    import argparse
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", default=".")
    args = parser.parse_args(argv)
    print(json.dumps(audit(args.repository_root), sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
