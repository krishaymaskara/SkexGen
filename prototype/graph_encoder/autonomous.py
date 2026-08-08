"""Target-free autonomous GE1 inference and frozen memory interventions."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import random
import time

try:
    import torch
except ImportError:  # Pure derangement and contract tests need no PyTorch.
    torch = None

from .errors import GraphEncoderError


AUTONOMOUS_EVALUATION_VERSION = "GE1-C6-AUTONOMOUS-v1"
MEMORY_INTERVENTION_VERSION = "GE1-C6-MEMORY-INTERVENTION-v1"
P_TRUE = "P_true"
P_SHUFFLE = "P_shuffle"
P_MEAN = "P_mean"
MEMORY_CONDITIONS = (P_TRUE, P_SHUFFLE, P_MEAN)


@dataclass(frozen=True)
class AutonomousInputBatch:
    """One target-free, deterministically ordered evaluation batch."""

    family_ids: tuple
    encoder_input: dict
    node_counts: tuple
    batch_identity: str


@dataclass(frozen=True)
class EncodedAutonomousRow:
    """One example's memory plus decoder-side bookkeeping, never a target."""

    family_id: str
    memory: object
    node_count: int
    batch_identity: str


@dataclass(frozen=True)
class AutonomousPrediction:
    family_id: str
    condition: str
    memory_source_family_id: object
    batch_identity: str
    raw_prediction: object
    constrained_prediction: object
    converted_prediction: object


@dataclass(frozen=True)
class AutonomousConditionResult:
    version: str
    condition: str
    available: bool
    unavailable_reason: object
    predictions: tuple
    memory_assignments: tuple
    batch_membership: tuple
    memory_altered: bool
    alteration_possible: bool
    singleton_mean_batches: tuple
    intervention_seconds: float
    elapsed_seconds: float


@dataclass(frozen=True)
class AutonomousEvaluationResult:
    version: str
    family_order: tuple
    encoding_seconds: float
    conditions: tuple


def autonomous_input_from_paired(paired, arm, torch_module=None):
    """Remove the target boundary and expose only the selected arm input."""

    runtime = torch if torch_module is None else torch_module
    if runtime is None:
        raise RuntimeError("autonomous inference requires PyTorch")
    family_ids = tuple(paired.family_ids)
    if not family_ids:
        raise GraphEncoderError("empty_autonomous_batch", "batch is empty")
    if arm == "flat":
        encoder_input = paired.flat_input.to_torch(runtime)
        node_counts = tuple(sum(row) for row in paired.flat_input.padding_mask)
    elif arm == "typed_graph":
        encoder_input = paired.graph_input.to_torch(runtime)
        bookkeeping = paired.graph_bookkeeping.to_torch(runtime)
        encoder_input["graph_offsets"] = bookkeeping["graph_offsets"]
        offsets = paired.graph_bookkeeping.graph_offsets
        node_counts = tuple(
            offsets[index + 1] - offsets[index]
            for index in range(len(offsets) - 1)
        )
    else:
        raise GraphEncoderError("invalid_configuration", "unknown encoder arm")
    identity = hashlib.sha256(
        ("\n".join(family_ids) + "\n").encode("utf-8")
    ).hexdigest()
    return AutonomousInputBatch(
        family_ids, encoder_input, node_counts, identity
    )


def deterministic_derangement(family_ids, seed):
    """Return a reproducible complete-set derangement over sorted families."""

    ordered = tuple(sorted(family_ids))
    if len(ordered) != len(set(ordered)):
        raise GraphEncoderError("duplicate_family_id", "family IDs must be unique")
    if len(ordered) < 2:
        raise GraphEncoderError(
            "shuffle_unavailable", "P_shuffle requires at least two examples"
        )
    digest = hashlib.sha256(
        "{}:{}".format(int(seed), "|".join(ordered)).encode("utf-8")
    ).digest()
    generator = random.Random(int.from_bytes(digest[:8], "big"))
    shift = generator.randrange(1, len(ordered))
    donors = ordered[shift:] + ordered[:shift]
    mapping = tuple(zip(ordered, donors))
    if any(recipient == donor for recipient, donor in mapping):
        raise AssertionError("cyclic derangement unexpectedly has a fixed point")
    return mapping


def run_autonomous_evaluation(model, input_batches, *, seed):
    """Encode once and decode P_true/P_shuffle/P_mean through one helper."""

    if torch is None:
        raise RuntimeError("autonomous inference requires PyTorch")
    batches = tuple(input_batches)
    if not batches:
        raise GraphEncoderError("empty_autonomous_set", "input batches are empty")
    all_ids = tuple(item for batch in batches for item in batch.family_ids)
    if all_ids != tuple(sorted(all_ids)) or len(all_ids) != len(set(all_ids)):
        raise GraphEncoderError(
            "invalid_autonomous_order",
            "evaluation batches must form one unique family-sorted sequence",
        )
    if any(not isinstance(batch, AutonomousInputBatch) for batch in batches):
        raise TypeError("every batch must be AutonomousInputBatch")

    was_training = model.training
    model.eval()
    encoding_start = time.perf_counter()
    encoded_rows = []
    try:
        with torch.no_grad():
            for batch in batches:
                encoded = model.encode(batch.encoder_input)
                if encoded.memory.size(0) != len(batch.family_ids):
                    raise GraphEncoderError(
                        "memory_alignment_failure", "memory rows do not align"
                    )
                for index, family_id in enumerate(batch.family_ids):
                    encoded_rows.append(EncodedAutonomousRow(
                        family_id,
                        encoded.memory[index:index + 1].detach().clone(),
                        int(batch.node_counts[index]),
                        batch.batch_identity,
                    ))
    finally:
        model.train(was_training)
    encoding_seconds = time.perf_counter() - encoding_start

    conditions = (
        _true_condition(model, tuple(encoded_rows)),
        _shuffle_condition(model, tuple(encoded_rows), seed),
        _mean_condition(model, tuple(encoded_rows)),
    )
    return AutonomousEvaluationResult(
        AUTONOMOUS_EVALUATION_VERSION,
        all_ids,
        encoding_seconds,
        conditions,
    )


def _true_condition(model, rows):
    supplied = tuple((row, row.family_id, row.memory) for row in rows)
    return _decode_condition(model, P_TRUE, supplied, False, False, (), 0.0)


def _shuffle_condition(model, rows, seed):
    intervention_start = time.perf_counter()
    if len(rows) < 2:
        return AutonomousConditionResult(
            AUTONOMOUS_EVALUATION_VERSION,
            P_SHUFFLE,
            False,
            "fewer_than_two_scored_examples",
            (),
            (),
            tuple((row.batch_identity, (row.family_id,)) for row in rows),
            False,
            False,
            (),
            time.perf_counter() - intervention_start,
            0.0,
        )
    by_id = {row.family_id: row for row in rows}
    mapping = deterministic_derangement(tuple(by_id), seed)
    supplied = tuple(
        (by_id[recipient], donor, by_id[donor].memory)
        for recipient, donor in mapping
    )
    changed = any(
        not torch.equal(row.memory, memory)
        for row, unused_donor, memory in supplied
    )
    distinct = any(
        not torch.equal(left.memory, right.memory)
        for index, left in enumerate(rows)
        for right in rows[index + 1:]
    )
    if distinct and not changed:
        raise GraphEncoderError(
            "memory_intervention_failure", "P_shuffle did not alter memory"
        )
    intervention_seconds = time.perf_counter() - intervention_start
    return _decode_condition(
        model, P_SHUFFLE, supplied, changed, distinct, (), intervention_seconds
    )


def _mean_condition(model, rows):
    intervention_start = time.perf_counter()
    by_batch = {}
    for row in rows:
        by_batch.setdefault(row.batch_identity, []).append(row)
    supplied = []
    changed = False
    possible = False
    singleton_batches = []
    for batch_identity in sorted(by_batch):
        members = tuple(by_batch[batch_identity])
        mean = torch.cat(tuple(item.memory for item in members), dim=0).mean(
            dim=0, keepdim=True
        )
        if len(members) == 1:
            singleton_batches.append(batch_identity)
        distinct = any(
            not torch.equal(members[0].memory, item.memory)
            for item in members[1:]
        )
        possible = possible or distinct
        for item in members:
            supplied.append((item, "batch_mean:" + batch_identity, mean.clone()))
            changed = changed or not torch.equal(item.memory, mean)
    supplied.sort(key=lambda item: item[0].family_id)
    if possible and not changed:
        raise GraphEncoderError(
            "memory_intervention_failure", "P_mean did not alter memory"
        )
    intervention_seconds = time.perf_counter() - intervention_start
    return _decode_condition(
        model, P_MEAN, tuple(supplied), changed, possible,
        tuple(singleton_batches), intervention_seconds
    )


def _decode_condition(
    model, condition, supplied, changed, possible, singleton_batches,
    intervention_seconds
):
    """The sole post-memory path; it contains no encoder-arm branch."""

    start = time.perf_counter()
    predictions = []
    assignments = []
    membership = {}
    for row, source, memory in supplied:
        output = model.decoder(
            memory,
            node_counts=torch.tensor(
                (row.node_count,), dtype=torch.long, device=memory.device
            ),
            node_count_source="target_free_input_node_count",
        )
        predictions.append(AutonomousPrediction(
            row.family_id,
            condition,
            source,
            row.batch_identity,
            output.raw_prediction[0],
            output.constrained_prediction[0],
            output.converted_prediction[0],
        ))
        assignments.append((row.family_id, source))
        membership.setdefault(row.batch_identity, []).append(row.family_id)
    return AutonomousConditionResult(
        AUTONOMOUS_EVALUATION_VERSION,
        condition,
        True,
        None,
        tuple(predictions),
        tuple(assignments),
        tuple(
            (identity, tuple(membership[identity]))
            for identity in sorted(membership)
        ),
        bool(changed),
        bool(possible),
        singleton_batches,
        float(intervention_seconds),
        time.perf_counter() - start,
    )
