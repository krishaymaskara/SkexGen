"""Run the bounded train-only constrained V2 tiny-overfit protocol."""

from __future__ import annotations

import argparse
from dataclasses import fields, replace
import json
from pathlib import Path
import sys

from .constrained_v2_config import ConstrainedProfileV2Config
from .constrained_v2_training import run_v2_tiny_overfit
from .constrained_v2_training_config import ConstrainedV2TrainingConfig


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-config")
    parser.add_argument("--training-config")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--maximum-steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--logging-cadence", type=int)
    parser.add_argument("--checkpoint-cadence", type=int)
    parser.add_argument("--tiny-subset-size", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        model_config = _load(
            args.model_config, ConstrainedProfileV2Config
        )
        training_config = _load(
            args.training_config, ConstrainedV2TrainingConfig
        )
        overrides = {
            "output_dir": args.output_dir,
            "seed": args.seed,
            "maximum_steps": args.maximum_steps,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "logging_cadence": args.logging_cadence,
            "checkpoint_cadence": args.checkpoint_cadence,
            "tiny_subset_size": args.tiny_subset_size,
            "device": args.device,
        }
        training_config = replace(
            training_config,
            **{
                name: value
                for name, value in overrides.items()
                if value is not None
            }
        )
        model_config.validate()
        training_config.validate()
        if not Path(args.corpus_dir).is_dir():
            raise ValueError("corpus directory does not exist")
        result = run_v2_tiny_overfit(
            args.corpus_dir, model_config, training_config
        )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "event": "terminal_failure",
                    "error_code": getattr(
                        exc, "code", "unexpected_v2_training_error"
                    ),
                    "error_type": type(exc).__name__,
                    "detail": str(exc),
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
                "global_step": result.global_step,
                "selected_example_ids": list(
                    result.selected_example_ids
                ),
                "success_criteria": result.success_criteria,
                "training_partition": "train",
                "validation_partition": "none",
                "test_partition_accessed": False,
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )
    return 0


def _load(path, configuration_type):
    if path is None:
        return configuration_type()
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("configuration JSON must contain one object")
    known = {item.name for item in fields(configuration_type)}
    unknown = set(payload) - known
    if unknown:
        raise ValueError(
            "unknown configuration fields: {}".format(sorted(unknown))
        )
    if "profile_family_order" in payload:
        payload["profile_family_order"] = tuple(
            payload["profile_family_order"]
        )
    return configuration_type(**payload)


if __name__ == "__main__":
    raise SystemExit(main())
