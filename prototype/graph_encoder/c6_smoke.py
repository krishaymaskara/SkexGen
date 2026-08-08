"""Authorized two-epoch, train-only C6 engineering smoke entry point."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
import time

import torch

from .autonomous import autonomous_input_from_paired, run_autonomous_evaluation
from .batching import build_paired_batch
from .config import GE1TrainingConfig
from .metrics import (
    complete_metrics_record,
    parameter_count_record,
    score_condition,
)
from .model import build_ge1_model, build_matched_ge1_models
from .partitions import load_train
from .training import load_training_checkpoint, run_ge1_training


C6_SMOKE_VERSION = "GE1-C6-TRAIN-ONLY-SMOKE-v2"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--arm", choices=("flat", "typed_graph"), default="flat")
    parser.add_argument("--seed", type=int, choices=(2026, 2027, 2028), default=2026)
    arguments = parser.parse_args(argv)
    record = run_smoke(
        corpus_dir=arguments.corpus_dir,
        output_dir=arguments.output_dir,
        repository_root=arguments.repository_root,
        expected_commit=arguments.expected_commit,
        arm=arguments.arm,
        seed=arguments.seed,
    )
    print(json.dumps({
        "event": "c6_train_only_smoke_success",
        "schema_version": record["schema_version"],
        "run_identity": record["run_identity"],
        "arm": record["arm"],
        "seed": record["seed"],
        "epoch": record["epoch"],
        "strict_checkpoint_reload": record["strict_checkpoint_reload"],
        "operation_template_train_payload_accessed": True,
        "development_accessed": False,
        "test_er_accessed": False,
        "systematic_rr_accessed": False,
        "iid_accessed": False,
        "history_depth_accessed": False,
        "geometry_extrapolation_accessed": False,
        "scientific_training_performed": False,
        "engineering_smoke_performed": True,
        "c7_or_later_performed": False,
    }, sort_keys=True, separators=(",", ":")), flush=True)
    return 0


def run_smoke(
    *, corpus_dir, output_dir, repository_root, expected_commit, arm, seed
):
    """Run exactly two epochs over operation_template.train through C1."""

    started = time.perf_counter()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    examples = tuple(load_train(corpus_dir))
    if len(examples) != 407:
        raise RuntimeError("authoritative train smoke requires exactly 407 families")
    flat_model, graph_model = build_matched_ge1_models(seed=seed)
    parameter_counts = parameter_count_record(flat_model, graph_model)
    model = flat_model if arm == "flat" else graph_model
    configuration = GE1TrainingConfig()
    checkpoints = output / "checkpoints"
    training = run_ge1_training(
        model,
        examples,
        training_config=configuration,
        checkpoint_directory=checkpoints,
        repository_root=repository_root,
        expected_commit=expected_commit,
        final_epoch=2,
        engineering_smoke=True,
    )
    if len(training.checkpoint_paths) != 2:
        raise AssertionError("C6 smoke must produce epoch-1 and epoch-2 checkpoints")

    reloaded = build_ge1_model(model.config)
    optimizer = torch.optim.AdamW(
        reloaded.parameters(),
        lr=configuration.learning_rate,
        weight_decay=configuration.weight_decay,
    )
    from .provenance import training_partition_identity
    partition_identity = training_partition_identity(
        tuple(item.physical_family_id for item in examples)
    )
    resumed = load_training_checkpoint(
        training.checkpoint_paths[-1],
        model=reloaded,
        optimizer=optimizer,
        training_config=configuration,
        partition_identity=partition_identity,
        expected_code_revision=expected_commit,
        restore_rng=True,
    )
    if resumed.completed_epoch != 2:
        raise AssertionError("strict smoke reload did not restore epoch 2")
    _assert_model_states_equal(model, reloaded)

    evaluation_batches = []
    ordered = tuple(sorted(examples, key=lambda item: item.physical_family_id))
    for start in range(0, len(ordered), configuration.batch_size):
        paired = build_paired_batch(ordered[start:start + configuration.batch_size])
        evaluation_batches.append(autonomous_input_from_paired(paired, arm))
    autonomous = run_autonomous_evaluation(
        reloaded, tuple(evaluation_batches), seed=seed
    )
    targets = {item.physical_family_id: item.target for item in ordered}
    condition_metrics = tuple(
        score_condition(condition, targets)
        for condition in autonomous.conditions
    )
    record = complete_metrics_record(
        arm=arm,
        seed=seed,
        epoch=2,
        checkpoint_identity=training.checkpoint_paths[-1],
        training_result=training,
        autonomous_result=autonomous,
        condition_metrics=condition_metrics,
        parameter_counts=parameter_counts,
        strict_checkpoint_reload=True,
        total_run_seconds=time.perf_counter() - started,
        templates_by_family={
            item.physical_family_id: item.metadata.operation_template
            for item in ordered
        },
    )
    record["smoke_version"] = C6_SMOKE_VERSION
    record["authorized_partition"] = "operation_template.train"
    record["physical_family_count"] = len(examples)
    record["operation_template_train_payload_accessed"] = True
    _atomic_json(output / "c6_metrics.json", record)
    return record


def _assert_model_states_equal(expected, observed):
    expected_state = expected.state_dict()
    observed_state = observed.state_dict()
    if list(expected_state) != list(observed_state):
        raise AssertionError("strict reload model keys differ")
    for name in expected_state:
        if not torch.equal(expected_state[name], observed_state[name]):
            raise AssertionError("strict reload tensor differs: {}".format(name))


def _atomic_json(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix="." + path.name + ".tmp-", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(record, stream, sort_keys=True, separators=(",", ":"))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, str(path))
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


if __name__ == "__main__":
    raise SystemExit(main())
