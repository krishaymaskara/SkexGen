"""Run the bounded V6 train-and-ordinary-IID-validation pilot."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

from .constrained_v6_config import ConstrainedProfileV6Config
from .constrained_v6_pilot import run_constrained_v6_pilot
from .constrained_v6_pilot_config import (
    ConstrainedV6PilotConfig,
    validate_pilot_partition_authorization,
)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        pilot_config = replace(
            ConstrainedV6PilotConfig(),
            output_dir=args.output_dir,
            device=args.device,
        )
        model_config = ConstrainedProfileV6Config()
        validate_pilot_partition_authorization(pilot_config)
        model_config.validate()
        if not Path(args.corpus_dir).is_dir():
            raise ValueError("corpus directory does not exist")
        result = run_constrained_v6_pilot(
            args.corpus_dir, pilot_config, model_config
        )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "event": "terminal_failure",
                    "error_code": getattr(
                        exc, "code", "unexpected_v6_pilot_error"
                    ),
                    "error_type": type(exc).__name__,
                    "detail": str(exc),
                    "training_partition": "train",
                    "validation_partition": "iid_validation",
                    "systematic_partition_accessed": False,
                    "test_partition_accessed": False,
                },
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ),
            file=sys.stderr,
        )
        return 2
    print(
        json.dumps(
            {
                "event": "terminal_success",
                "output_dir": result.output_dir,
                "metrics_path": result.metrics_path,
                "final_checkpoint": result.final_checkpoint,
                "selected_checkpoint": result.selected_checkpoint,
                "completed_epochs": result.completed_epochs,
                "global_step": result.global_step,
                "examples_processed": result.examples_processed,
                "acceptance": result.terminal_summary["acceptance"],
                "training_partition": "train",
                "validation_partition": "iid_validation",
                "systematic_partition_accessed": False,
                "test_partition_accessed": False,
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
