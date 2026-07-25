"""Training configuration and lightweight data-contract tests."""

from __future__ import annotations

from dataclasses import replace
import json
import tempfile
import unittest

from prototype.flat_baseline.data import (
    TrainingDataError,
    load_training_data,
    validate_model_selection_partitions,
)
from prototype.flat_baseline.training_config import (
    TrainingConfig,
    TrainingConfigurationError,
)
from prototype.controlled_data.builders import build_history
from prototype.controlled_data.identity import source_family_id
from prototype.model_data.batching import collate_flat
from prototype.model_data.tests.fixtures import source, write_physical_corpus
from prototype.representation.model import GeometryEncoding


class TrainingConfigurationTests(unittest.TestCase):
    def test_validation_and_deterministic_serialization(self):
        config = TrainingConfig()
        config.validate()
        self.assertEqual(config.to_json(), TrainingConfig().to_json())
        self.assertEqual(
            json.loads(config.to_json()),
            config.to_dict(),
        )

    def test_invalid_values_are_rejected(self):
        invalid = (
            TrainingConfig(seed=-1),
            TrainingConfig(epochs=0),
            TrainingConfig(batch_size=True),
            TrainingConfig(learning_rate=0.0),
            TrainingConfig(weight_decay=float("nan")),
            TrainingConfig(gradient_clip_norm=-1.0),
            TrainingConfig(validation_interval=0),
            TrainingConfig(dataloader_workers=-1),
            TrainingConfig(device="mps"),
            TrainingConfig(output_dir=""),
            TrainingConfig(checkpoint_selection_metric="test_loss"),
        )
        for config in invalid:
            with self.subTest(config=config):
                with self.assertRaises(TrainingConfigurationError):
                    config.validate()

    def test_resume_signature_allows_only_run_extension_and_relocation(self):
        original = TrainingConfig()
        allowed = replace(
            original, epochs=20, output_dir="/another/run", device="cuda"
        )
        self.assertEqual(original.resume_signature(), allowed.resume_signature())
        changed = replace(original, learning_rate=0.25)
        self.assertNotEqual(
            original.resume_signature(), changed.resume_signature()
        )


class TrainingDataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        sources = (source("E"), source("R"), source("ER"))
        family_ids = tuple(
            source_family_id(build_history(item, GeometryEncoding.CONTINUOUS))
            for item in sources
        )
        partitions = {
            family_ids[0]: "train",
            family_ids[1]: "train",
            family_ids[2]: "validation",
        }
        write_physical_corpus(
            self.temporary.name, sources, partitions=partitions
        )

    def test_split_loading_preserves_families_and_mixed_lengths(self):
        data = load_training_data(self.temporary.name, "iid")
        self.assertEqual(len(data.train_examples), 2)
        self.assertEqual(len(data.validation_examples), 1)
        self.assertFalse(
            set(data.train_family_ids) & set(data.validation_family_ids)
        )
        self.assertGreater(
            len({len(item.categorical_ids) for item in data.train_examples}),
            1,
        )
        batch = collate_flat(data.train_examples)
        self.assertEqual(batch.family_ids, data.train_family_ids)
        self.assertEqual(len(batch.padding_mask), 2)
        self.assertNotEqual(
            sum(batch.padding_mask[0]), sum(batch.padding_mask[1])
        )

    def test_model_selection_partitions_are_frozen(self):
        invalid = (
            ("test", "validation", "invalid_training_partition"),
            ("unknown", "validation", "invalid_training_partition"),
            ("train", "test", "invalid_validation_partition"),
            (
                "train",
                "secondary_systematic_validation",
                "invalid_validation_partition",
            ),
            ("train", "unknown", "invalid_validation_partition"),
        )
        for train_partition, validation_partition, code in invalid:
            with self.subTest(
                train=train_partition, validation=validation_partition
            ):
                with self.assertRaises(TrainingDataError) as captured:
                    validate_model_selection_partitions(
                        train_partition, validation_partition
                    )
                self.assertEqual(captured.exception.code, code)

    def test_valid_model_selection_partitions_still_load(self):
        validate_model_selection_partitions("train", "validation")
        data = load_training_data(
            self.temporary.name,
            "iid",
            train_partition="train",
            validation_partition="validation",
        )
        self.assertTrue(data.train_examples)
        self.assertTrue(data.validation_examples)

    def test_overfit_limit_is_stable_and_does_not_reassign_partitions(self):
        first = load_training_data(
            self.temporary.name, "iid", family_limit=1
        )
        second = load_training_data(
            self.temporary.name, "iid", family_limit=1
        )
        self.assertEqual(first.train_family_ids, second.train_family_ids)
        self.assertEqual(
            first.validation_family_ids, second.validation_family_ids
        )
        self.assertEqual(len(first.train_examples), 1)
        self.assertEqual(len(first.validation_examples), 1)


if __name__ == "__main__":
    unittest.main()
