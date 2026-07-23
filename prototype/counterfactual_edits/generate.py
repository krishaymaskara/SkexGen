"""Generate a deterministic counterfactual CAD edit-pair corpus."""

from __future__ import annotations

import argparse
import sys

from .config import CounterfactualConfig
from .dataset import generate_corpus


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-edit-families", type=int, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = generate_corpus(
            args.output_dir,
            CounterfactualConfig(args.num_edit_families, seed=args.seed),
        )
    except Exception as exc:
        print(
            f"counterfactual generation failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1
    print(
        "generated "
        f"{manifest['total_edit_family_count']} edit families and "
        f"{manifest['total_edit_sample_count']} edit samples"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
