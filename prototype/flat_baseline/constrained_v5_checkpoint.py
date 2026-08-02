"""Single strict checkpoint schema for all constrained V5 workflows."""

from __future__ import annotations

from collections.abc import Mapping
import math

from prototype.model_data.vocab import NODE_TYPES
from prototype.node_grammar import (
    V5_NODE_GRAMMAR,
    node_grammar_metadata,
)


V5_MODEL_METADATA_FIELDS = (
    "checkpoint_version",
    "model_name",
    "model_config_version",
    "decoder_contract_version",
    "learned_geometry_channel_indices",
    "canonical_plane_contract_id",
    "base_geometry_contract_id",
    "categorical_selection_contract_id",
    "categorical_selection_contract",
    "node_grammar_contract_id",
    "node_grammar_contract",
    "node_vocabulary",
    "valid_requested_node_counts",
    "operation_limit",
    "completion_algorithm_id",
    "model_config",
)
V5_TRAINING_STATE_FIELDS = (
    "model_state",
    "optimizer_state",
    "vq_state",
    "epoch",
    "global_step",
    "rng_state",
    "source_provenance",
)
V5_TINY_METADATA_FIELDS = (
    "training_config",
    "profile_family_order",
    "extent_bounds",
    "loss_weights",
    "selected_tiny_overfit_ids",
    "data_state",
)
V5_PILOT_METADATA_FIELDS = (
    "checkpoint_kind",
    "pilot_identity",
    "pilot_config",
    "optimization_config",
    "examples_processed",
    "configured_maximum_steps",
    "completed_epochs",
    "training_partition_state",
    "validation_partition_state",
    "training_sampler_state",
    "validation_cadence",
    "validation_summaries",
    "systematic_partition_accessed",
    "test_partition_accessed",
)
V5_TINY_CHECKPOINT_FIELDS = frozenset(
    V5_MODEL_METADATA_FIELDS + V5_TRAINING_STATE_FIELDS + V5_TINY_METADATA_FIELDS
)
V5_PILOT_CHECKPOINT_FIELDS = frozenset(
    V5_MODEL_METADATA_FIELDS + V5_TRAINING_STATE_FIELDS + V5_PILOT_METADATA_FIELDS
)


class V5CheckpointValidationError(ValueError):
    """A V5 payload violates the frozen checkpoint contract."""

    def __init__(self, detail):
        self.code = "malformed_checkpoint"
        self.detail = detail
        super().__init__("malformed_checkpoint: {}".format(detail))


def required_top_level_fields(checkpoint_kind):
    if checkpoint_kind == "tiny_overfit":
        return V5_TINY_CHECKPOINT_FIELDS
    if checkpoint_kind == "pilot_fixed_epoch":
        return V5_PILOT_CHECKPOINT_FIELDS
    raise V5CheckpointValidationError(
        "checkpoint kind {!r} is unsupported".format(checkpoint_kind)
    )


def validate_v5_checkpoint_field_set(checkpoint, checkpoint_kind):
    """Validate the exact variant field set without interpreting tensor state."""

    if not isinstance(checkpoint, Mapping):
        _invalid("checkpoint payload must be a mapping")
    expected_fields = required_top_level_fields(checkpoint_kind)
    actual_fields = set(checkpoint)
    missing = sorted(expected_fields - actual_fields)
    unexpected = sorted(actual_fields - expected_fields, key=repr)
    if missing:
        _invalid("missing field{} {}".format(
            "s" if len(missing) != 1 else "", ", ".join(missing)
        ))
    if unexpected:
        _invalid("unexpected field{} {}".format(
            "s" if len(unexpected) != 1 else "",
            ", ".join(repr(name) for name in unexpected),
        ))
    return checkpoint


def required_model_metadata_fields():
    return V5_MODEL_METADATA_FIELDS


def required_training_state_fields():
    return V5_TRAINING_STATE_FIELDS


def required_pilot_metadata_fields():
    return V5_PILOT_METADATA_FIELDS


def canonical_grammar_metadata():
    """Return deterministic, JSON-compatible V5 grammar provenance."""

    return node_grammar_metadata(V5_NODE_GRAMMAR)


def v5_model_metadata(model_config):
    """Construct the common model and grammar portion of every V5 payload."""

    return {
        "checkpoint_version": 5,
        "model_name": model_config.model_name,
        "model_config_version": model_config.model_config_version,
        "decoder_contract_version": model_config.decoder_contract_version,
        "learned_geometry_channel_indices": list(
            model_config.learned_geometry_channel_indices
        ),
        "canonical_plane_contract_id": model_config.canonical_plane_contract_id,
        "base_geometry_contract_id": model_config.base_geometry_contract_id,
        "categorical_selection_contract_id": (
            model_config.categorical_selection_contract_id
        ),
        "categorical_selection_contract": (
            model_config.categorical_selection_contract
        ),
        "node_grammar_contract_id": model_config.node_grammar_contract_id,
        "node_grammar_contract": canonical_grammar_metadata(),
        "node_vocabulary": list(NODE_TYPES.tokens),
        "valid_requested_node_counts": list(
            model_config.valid_requested_node_counts
        ),
        "operation_limit": model_config.max_operations,
        "completion_algorithm_id": model_config.completion_algorithm_id,
        "model_config": model_config.to_dict(),
    }


def validate_v5_checkpoint_payload(
    checkpoint,
    checkpoint_kind,
    model_config,
    *,
    torch_module,
):
    """Validate common V5 schema, metadata, state types, and provenance."""

    validate_v5_checkpoint_field_set(checkpoint, checkpoint_kind)

    for name, expected in v5_model_metadata(model_config).items():
        if checkpoint[name] != expected:
            if name == "node_grammar_contract":
                _invalid(_grammar_mismatch(checkpoint[name], expected))
            _invalid("field {} differs from the frozen V5 contract".format(name))

    for name in ("model_state", "optimizer_state", "vq_state"):
        if not isinstance(checkpoint[name], Mapping):
            _invalid("checkpoint field {} has invalid type".format(name))
    for name, value in checkpoint["model_state"].items():
        if not isinstance(name, str) or not torch_module.is_tensor(value):
            _invalid("model_state must map string names to tensors")
    _nonnegative_integer(checkpoint["epoch"], "epoch")
    _nonnegative_integer(checkpoint["global_step"], "global_step")
    _validate_source_provenance(checkpoint["source_provenance"])
    _validate_rng_state(checkpoint["rng_state"], torch_module)
    _validate_tensor_tree(checkpoint["model_state"], "model_state", torch_module)
    _validate_tensor_tree(
        checkpoint["optimizer_state"], "optimizer_state", torch_module
    )
    _validate_vq_state(checkpoint, torch_module)
    if checkpoint_kind == "tiny_overfit":
        for name in V5_TINY_METADATA_FIELDS:
            _validate_json_tree(checkpoint[name], name)
    return checkpoint


def validate_v5_pilot_checkpoint_state(
    checkpoint,
    pilot_config,
    train_example_count,
):
    """Validate fixed-budget pilot progress and protected-access facts."""

    epoch = checkpoint["epoch"]
    if not 1 <= epoch <= pilot_config.epochs:
        _invalid("checkpoint field epoch is outside the pilot budget")
    steps_per_epoch = int(math.ceil(
        train_example_count / float(pilot_config.batch_size)
    ))
    expected_values = (
        ("completed_epochs", epoch),
        ("global_step", epoch * steps_per_epoch),
        ("examples_processed", epoch * train_example_count),
        (
            "configured_maximum_steps",
            pilot_config.epochs * steps_per_epoch,
        ),
        ("validation_cadence", pilot_config.validation_interval),
    )
    for name, expected in expected_values:
        value = checkpoint[name]
        _nonnegative_integer(value, name)
        if value != expected:
            _invalid("checkpoint field {} disagrees with pilot progress".format(name))
    expected_sampler = {
        "base_seed": pilot_config.seed,
        "completed_epoch": epoch,
        "next_epoch_seed": pilot_config.seed + epoch + 1,
    }
    if checkpoint["training_sampler_state"] != expected_sampler:
        _invalid("training_sampler_state mismatch")
    summaries = checkpoint["validation_summaries"]
    if not isinstance(summaries, Mapping) or set(summaries) != {
        "teacher_forced", "autonomous"
    }:
        _invalid("validation_summaries field set mismatch")
    for name in ("systematic_partition_accessed", "test_partition_accessed"):
        if type(checkpoint[name]) is not bool:
            _invalid("checkpoint field {} must be Boolean".format(name))
        if checkpoint[name] is not False:
            _invalid("checkpoint field {} must be false".format(name))
    for name in V5_PILOT_METADATA_FIELDS:
        if name not in ("systematic_partition_accessed", "test_partition_accessed"):
            _validate_json_tree(checkpoint[name], name)
    return checkpoint


def _grammar_mismatch(actual, expected):
    if not isinstance(actual, Mapping):
        return "grammar metadata has invalid type"
    actual_fields = set(actual)
    expected_fields = set(expected)
    missing = sorted(expected_fields - actual_fields)
    extra = sorted(actual_fields - expected_fields, key=repr)
    if missing:
        return "grammar metadata missing field{} {}".format(
            "s" if len(missing) != 1 else "", ", ".join(missing)
        )
    if extra:
        return "grammar metadata has unexpected field{} {}".format(
            "s" if len(extra) != 1 else "",
            ", ".join(repr(name) for name in extra),
        )
    labels = {
        "contract_id": "grammar contract ID",
        "node_vocabulary": "grammar vocabulary order",
        "start_node": "grammar start state",
        "transitions": "grammar transition graph",
        "terminal_nodes": "grammar terminal states",
        "valid_requested_node_counts": "grammar valid lengths",
        "operation_limit": "grammar operation limit",
        "completion_algorithm_id": "grammar completion algorithm ID",
        "canonical_templates": "grammar canonical templates",
    }
    for name in expected:
        if actual[name] != expected[name]:
            return "{} mismatch".format(labels.get(name, "grammar {}".format(name)))
    return "grammar metadata differs"


def _validate_source_provenance(value):
    if not isinstance(value, Mapping):
        _invalid("source_provenance has invalid type")
    fields = {
        "git_commit", "git_dirty", "git_status_porcelain", "source_tree_sha256"
    }
    if set(value) != fields:
        _invalid("source_provenance field set mismatch")
    commit = value["git_commit"]
    if not _is_lower_hex(commit, 40):
        _invalid("source_provenance.git_commit must be a 40-character hex SHA")
    if type(value["git_dirty"]) is not bool:
        _invalid("source_provenance.git_dirty must be Boolean")
    status = value["git_status_porcelain"]
    if (
        not isinstance(status, list)
        or any(not isinstance(item, str) or not item for item in status)
        or status != sorted(status)
    ):
        _invalid("source_provenance.git_status_porcelain must be a sorted string list")
    digest = value["source_tree_sha256"]
    if not _is_lower_hex(digest, 64):
        _invalid("source_provenance.source_tree_sha256 must be a 64-character hex digest")


def _validate_rng_state(value, torch_module):
    if not isinstance(value, Mapping) or set(value) != {
        "python", "torch_cpu", "torch_cuda"
    }:
        _invalid("rng_state field set mismatch")
    python_state = value["python"]
    if (
        not isinstance(python_state, tuple)
        or len(python_state) != 3
        or python_state[0] != 3
        or not isinstance(python_state[1], tuple)
        or len(python_state[1]) != 625
        or any(isinstance(item, bool) or not isinstance(item, int)
               for item in python_state[1])
        or (
            python_state[2] is not None
            and (
                isinstance(python_state[2], bool)
                or not isinstance(python_state[2], (int, float))
                or not math.isfinite(float(python_state[2]))
            )
        )
    ):
        _invalid("rng_state.python has invalid type or shape")
    cpu_state = value["torch_cpu"]
    if (
        not torch_module.is_tensor(cpu_state)
        or cpu_state.dtype != torch_module.uint8
        or cpu_state.ndim != 1
    ):
        _invalid("rng_state.torch_cpu must be a one-dimensional uint8 tensor")
    cuda_state = value["torch_cuda"]
    if cuda_state is not None and (
        not isinstance(cuda_state, list)
        or any(
            not torch_module.is_tensor(item)
            or item.dtype != torch_module.uint8
            or item.ndim != 1
            for item in cuda_state
        )
    ):
        _invalid("rng_state.torch_cuda must be null or a tensor list")


def _validate_tensor_tree(value, path, torch_module):
    if torch_module.is_tensor(value):
        if value.dtype.is_floating_point and not bool(
            torch_module.isfinite(value).all().item()
        ):
            _invalid("checkpoint field {} is nonfinite".format(path))
        return
    if isinstance(value, Mapping):
        for name, item in value.items():
            _validate_tensor_tree(item, "{}.{}".format(path, name), torch_module)
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_tensor_tree(item, "{}[{}]".format(path, index), torch_module)
        return
    if value is None or type(value) in (bool, int, str):
        return
    if isinstance(value, float) and math.isfinite(value):
        return
    _invalid("checkpoint field {} has invalid or nonfinite value".format(path))


def _validate_vq_state(checkpoint, torch_module):
    vq_state = checkpoint["vq_state"]
    if any(
        not isinstance(name, str) or not torch_module.is_tensor(value)
        for name, value in vq_state.items()
    ):
        _invalid("vq_state must map string names to tensors")
    expected_names = {
        name[3:] for name in checkpoint["model_state"] if name.startswith("vq.")
    }
    if set(vq_state) != expected_names:
        _invalid("vq_state field set differs from model_state")
    for name, value in vq_state.items():
        model_value = checkpoint["model_state"].get("vq." + name)
        if (
            not torch_module.is_tensor(model_value)
            or not torch_module.equal(value, model_value)
        ):
            _invalid("vq_state.{} differs from model_state".format(name))


def _validate_json_tree(value, path):
    if value is None or type(value) in (bool, int, str):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            _invalid("checkpoint field {} is nonfinite".format(path))
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_json_tree(item, "{}[{}]".format(path, index))
        return
    if isinstance(value, Mapping):
        for name, item in value.items():
            if not isinstance(name, str):
                _invalid("checkpoint field {} has a non-string JSON key".format(path))
            _validate_json_tree(item, "{}.{}".format(path, name))
        return
    _invalid("checkpoint field {} is not JSON-compatible".format(path))


def _is_lower_hex(value, length):
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _nonnegative_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        _invalid("checkpoint field {} must be a nonnegative integer".format(name))


def _invalid(detail):
    raise V5CheckpointValidationError(detail)
