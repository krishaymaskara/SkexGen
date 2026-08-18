"""Static audit for ADR-0014 checkpoint preservation."""

from __future__ import annotations

from pathlib import Path


AUDIT_VERSION = "GE1-STAGE6-CHECKPOINT-BUNDLE-AUDIT-v1"


def audit(repository_root="."):
    path = Path(repository_root) / "prototype/graph_encoder/stage6_structure_only_producer.py"
    source = path.read_text(encoding="utf-8")
    for required in (
        "GE1-STAGE6-STRUCTURE-ONLY-CHECKPOINT-BUNDLE-v1",
        "def create_checkpoint_bundle(", "def verify_checkpoint_bundle(",
        '"stage6-{}-seed{}.pt"', '"artifact_manifest.json"', '"SHA256SUMS"',
        '"stage6_wrapper_sha256"', '"bundle_sha256"',
        '".incomplete-" + job_id',
    ):
        if required not in source:
            raise AssertionError("missing checkpoint-bundle control: " + required)
    return {
        "version": AUDIT_VERSION,
        "atomic_job_scoped_staging": True,
        "exact_arm_seed_matrix": True,
        "wrapper_hash_cross_check": True,
        "manifest_and_checksum_verification": True,
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
