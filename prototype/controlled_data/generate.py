"""Command-line entry point for deterministic controlled corpus generation."""

from __future__ import annotations

import argparse
import sys

from .config import ConfigurationError, GeneratorConfig
from .dataset import GenerationError, generate_corpus


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-source-families", type=int, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = generate_corpus(
            args.output_dir,
            GeneratorConfig(num_source_families=args.num_source_families, seed=args.seed),
        )
    except (ConfigurationError, GenerationError, OSError, ValueError) as exc:
        print(f"controlled-data generation failed: {exc}", file=sys.stderr)
        return 1
    print(
        f"generated {manifest['total_source_family_count']} source families and "
        f"{manifest['total_sample_variant_count']} sample variants"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
