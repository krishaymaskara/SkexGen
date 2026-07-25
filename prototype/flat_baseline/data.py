"""Physical-family split selection and authoritative flat batching."""

from __future__ import annotations

from dataclasses import dataclass
import random

from prototype.model_data.adapters import adapt_flat_mixed
from prototype.model_data.batching import collate_flat
from prototype.model_data.loader import load_physical_examples


class TrainingDataError(ValueError):
    """A requested physical-family partition cannot be used for training."""

    def __init__(self, code, detail):
        self.code = code
        self.detail = detail
        super().__init__("{}: {}".format(code, detail))


@dataclass(frozen=True)
class TrainingData:
    train_examples: tuple
    validation_examples: tuple
    train_family_ids: tuple
    validation_family_ids: tuple


def validate_model_selection_partitions(
    train_partition, validation_partition
):
    """Freeze the only partitions authorized for optimization/model selection."""

    if train_partition != "train":
        raise TrainingDataError(
            "invalid_training_partition",
            "official training requires partition 'train'",
        )
    if validation_partition != "validation":
        raise TrainingDataError(
            "invalid_validation_partition",
            "checkpoint selection requires partition 'validation'",
        )


def load_training_data(
    corpus_dir,
    split_name,
    train_partition="train",
    validation_partition="validation",
    family_limit=None,
):
    """Load, partition, adapt, and stably limit physical examples."""

    validate_model_selection_partitions(
        train_partition, validation_partition
    )
    physical = load_physical_examples(corpus_dir, split_name)
    train = tuple(
        adapt_flat_mixed(item)
        for item in physical
        if item.partition == train_partition
    )
    validation = tuple(
        adapt_flat_mixed(item)
        for item in physical
        if item.partition == validation_partition
    )
    train = _limit(train, family_limit)
    validation = _limit(validation, family_limit)
    if not train:
        raise TrainingDataError(
            "empty_training_split",
            "partition {!r} contains no selected physical families".format(
                train_partition
            ),
        )
    if not validation:
        raise TrainingDataError(
            "empty_validation_split",
            "partition {!r} contains no selected physical families".format(
                validation_partition
            ),
        )
    train_ids = tuple(item.physical_family_id for item in train)
    validation_ids = tuple(item.physical_family_id for item in validation)
    overlap = set(train_ids) & set(validation_ids)
    if overlap:
        raise TrainingDataError(
            "family_split_leakage",
            "physical families occur in both requested partitions",
        )
    return TrainingData(train, validation, train_ids, validation_ids)


def make_data_loader(
    examples,
    batch_size,
    workers,
    seed,
    shuffle,
    torch_module,
):
    """Use PyTorch's loader with the frozen collate_flat contract."""

    if not examples:
        raise TrainingDataError("empty_loader", "cannot batch an empty split")
    generator = torch_module.Generator()
    generator.manual_seed(seed)
    return torch_module.utils.data.DataLoader(
        tuple(examples),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        collate_fn=collate_flat,
        generator=generator,
        worker_init_fn=_seed_worker,
        drop_last=False,
    )


def _limit(examples, family_limit):
    ordered = tuple(sorted(examples, key=lambda item: item.physical_family_id))
    if family_limit is None:
        return ordered
    if (
        isinstance(family_limit, bool)
        or not isinstance(family_limit, int)
        or family_limit <= 0
    ):
        raise TrainingDataError(
            "invalid_family_limit",
            "overfit family limit must be a positive integer",
        )
    return ordered[:family_limit]


def _seed_worker(worker_id):
    # DataLoader assigns each worker a deterministic torch seed. Mirror it into
    # Python's RNG without importing NumPy, which this pipeline does not use.
    del worker_id
    import torch

    random.seed(torch.initial_seed() % (2**32))
