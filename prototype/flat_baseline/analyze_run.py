"""Analyze a quiescent completed, failed, or canceled B0 run directory."""

from __future__ import annotations

import argparse
import json
import sys

from .run_analysis import RunAnalysisError, terminal_summary, write_analysis


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Validate and summarize a quiescent B0 run after its Slurm job has "
            "completed, failed, or been canceled. This is not a live monitor."
        )
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        summary = write_analysis(args.run_dir, args.output_dir)
    except RunAnalysisError as exc:
        _write_error(exc)
        return 2
    except Exception as exc:
        _write_error(
            RunAnalysisError(
                "unexpected_analysis_error",
                "{}: {}".format(type(exc).__name__, str(exc)),
            )
        )
        return 2
    print(terminal_summary(summary))
    return 0


def _write_error(exc):
    print(
        json.dumps(
            {
                "error": {
                    "code": exc.code,
                    "detail": exc.detail,
                    "run_status": "invalid",
                    "type": type(exc).__name__,
                }
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ),
        file=sys.stderr,
    )


if __name__ == "__main__":
    raise SystemExit(main())
