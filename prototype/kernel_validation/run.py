"""Command-line entry point for deterministic kernel execution validation."""

from __future__ import annotations

import argparse
import sys

from .corpus import load_corpus
from .executor import execute_sample
from .reporting import build_report, publish_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: list[str] | None = None, *, adapter_factory=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if adapter_factory is None:
            from .occ_adapter import OpenCascadeAdapter

            adapter_factory = OpenCascadeAdapter
        adapter = adapter_factory()
        metadata, samples = load_corpus(args.corpus_dir)
        results = tuple(execute_sample(sample, adapter) for sample in samples)
        report = build_report(results, adapter.backend_versions(), metadata)
        publish_report(args.output_dir, report)
    except Exception as exc:
        print(f"kernel validation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"kernel validation wrote {len(results)} sample results")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
