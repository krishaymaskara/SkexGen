"""Run the one frozen corrected graph V1 C1 train/IID-validation pilot."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import sys

from .config import GraphV1Config
from .pilot import run_graph_v1_pilot
from .pilot_config import GraphPilotConfig, validate_partition_authorization


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--reviewed-commit", required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        pilot_config = replace(GraphPilotConfig(), output_dir=args.output_dir)
        validate_partition_authorization(pilot_config)
        result = run_graph_v1_pilot(
            args.corpus_dir,
            pilot_config,
            GraphV1Config(),
            repository_root=args.repository_root,
            reviewed_commit=args.reviewed_commit,
        )
    except Exception as exc:
        print(json.dumps({
            "event": "terminal_failure",
            "error_type": type(exc).__name__,
            "detail": str(exc),
            "systematic_partition_accessed": False,
            "test_partition_accessed": False,
        }, sort_keys=True, separators=(",", ":")), file=sys.stderr)
        return 2
    print(json.dumps({
        "event": "terminal_success",
        "output_dir": result.output_dir,
        "selected_checkpoint": result.selected_checkpoint,
        "final_checkpoint": result.final_checkpoint,
        "global_step": result.global_step,
        "examples_processed": result.examples_processed,
        "acceptance": result.terminal_summary["acceptance"],
        "systematic_partition_accessed": False,
        "test_partition_accessed": False,
    }, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
