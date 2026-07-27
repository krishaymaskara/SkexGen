"""Command-line entry point for teacher-forced B0-FLAT-MIXED-VQ training."""

from __future__ import annotations

import argparse
from dataclasses import fields, replace
import json
from pathlib import Path
import sys

from .config import FlatBaselineConfig
from .data import validate_model_selection_partitions
from .training import train_run
from .training_config import TrainingConfig


class TrainingCliError(ValueError):
    """A user-supplied CLI input cannot produce a training run."""

    def __init__(self, code, detail):
        self.code = code
        self.detail = detail
        super().__init__("{}: {}".format(code, detail))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", required=True)
    parser.add_argument("--split-name", default="iid")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--validation-split", default="validation")
    parser.add_argument("--output-dir")
    parser.add_argument("--resume-checkpoint")
    parser.add_argument("--model-config")
    parser.add_argument("--training-config")
    parser.add_argument("--overfit-families", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--gradient-clip-norm", type=float)
    parser.add_argument("--validation-interval", type=int)
    parser.add_argument("--checkpoint-interval", type=int)
    parser.add_argument("--dataloader-workers", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--checkpoint-selection-metric")
    parser.add_argument(
        "--vq-init", choices=("normal", "train-kmeans")
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        model_config = _load_config(
            args.model_config, FlatBaselineConfig, "model"
        )
        training_config = _load_config(
            args.training_config, TrainingConfig, "training"
        )
        overrides = {
            "output_dir": args.output_dir,
            "seed": args.seed,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "gradient_clip_norm": args.gradient_clip_norm,
            "validation_interval": args.validation_interval,
            "checkpoint_interval": args.checkpoint_interval,
            "dataloader_workers": args.dataloader_workers,
            "device": args.device,
            "checkpoint_selection_metric": args.checkpoint_selection_metric,
            "vq_init": args.vq_init,
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
        validate_model_selection_partitions(
            args.train_split, args.validation_split
        )
        if not Path(args.corpus_dir).is_dir():
            raise TrainingCliError(
                "missing_corpus",
                "corpus directory does not exist",
            )
        result = train_run(
            args.corpus_dir,
            args.split_name,
            args.train_split,
            args.validation_split,
            model_config,
            training_config,
            args.resume_checkpoint,
            args.overfit_families,
        )
    except Exception as exc:
        _write_cli_error(exc)
        return 2
    print(
        json.dumps(
            {
                "metrics_path": result.metrics_path,
                "last_checkpoint": result.last_checkpoint,
                "best_checkpoint": result.best_checkpoint,
            },
            sort_keys=True,
            indent=2,
        )
    )
    return 0


def _load_config(path, configuration_type, label):
    if path is None:
        return configuration_type()
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise TrainingCliError(
            "missing_{}_config".format(label),
            "{} configuration file cannot be read".format(label),
        ) from exc
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise TrainingCliError(
            "malformed_{}_config_json".format(label),
            "{} configuration is not valid JSON".format(label),
        ) from exc
    if not isinstance(payload, dict):
        raise TrainingCliError(
            "non_object_{}_config".format(label),
            "{} configuration JSON must contain one object".format(label),
        )
    known = {item.name for item in fields(configuration_type)}
    unknown = sorted(set(payload) - known)
    if unknown:
        raise TrainingCliError(
            "unknown_{}_config_field".format(label),
            "unknown {} configuration field: {}".format(
                label, unknown[0]
            ),
        )
    return configuration_type(**payload)


def _write_cli_error(exc):
    code = getattr(exc, "code", None)
    if code is None:
        if type(exc).__name__.endswith("ConfigurationError"):
            code = "invalid_configuration"
        elif isinstance(exc, FileNotFoundError):
            code = "missing_file"
        else:
            code = "unexpected_training_error"
    record = {
        "error": {
            "code": code,
            "detail": str(exc),
            "type": type(exc).__name__,
        }
    }
    print(
        json.dumps(
            record,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ),
        file=sys.stderr,
    )


if __name__ == "__main__":
    raise SystemExit(main())
