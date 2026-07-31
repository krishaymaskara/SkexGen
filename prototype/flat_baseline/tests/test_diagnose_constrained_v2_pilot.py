"""Frozen-checkpoint Stage 2G diagnostic tests."""

from __future__ import annotations

from dataclasses import replace
import ast
import math
from pathlib import Path
import tempfile
from unittest import mock
import unittest

try:
    import torch
except ImportError:
    torch = None

from prototype.controlled_data.builders import build_history
from prototype.controlled_data.factors import PrimitiveFamily
from prototype.controlled_data.identity import source_family_id
from prototype.model_data.tests.fixtures import source, write_physical_corpus
from prototype.representation.model import GeometryEncoding

from prototype.flat_baseline.constrained_v2_pilot_diagnostic_config import (
    ConstrainedV2PilotDiagnosticConfig,
    ConstrainedV2PilotDiagnosticError,
    DIAGNOSTIC_IDENTITY,
    EXPECTED_PILOT_SOURCE_COMMIT,
    validate_diagnostic_partition_authorization,
)

if torch is not None:
    from prototype.constrained_profile_decoder import profile_targets_for_loss
    from prototype.flat_baseline.checkpointing import capture_rng_state
    from prototype.flat_baseline.constrained_v2 import (
        REMAINING_CATEGORICAL_TARGET_INDICES,
        ConstrainedProfileV2Model,
    )
    from prototype.flat_baseline.constrained_v2_autonomous import (
        greedy_decode_v2,
    )
    from prototype.flat_baseline.constrained_v2_config import (
        ConstrainedProfileV2Config,
    )
    from prototype.flat_baseline.constrained_v2_pilot import (
        PILOT_CHECKPOINT_FIELDS,
        load_pilot_data,
    )
    from prototype.flat_baseline.constrained_v2_pilot_config import (
        ConstrainedV2PilotConfig,
    )
    from prototype.flat_baseline.diagnose_constrained_v2_pilot import (
        HYBRID_ARM_CONTRACTS,
        _aggregate_first_failures,
        _require_finite_json,
        _tree_equal,
        _vq_partition_summary,
        audit_frozen_checkpoint_payload,
        build_hybrid_arms,
        classify_diagnostic,
        classify_vq_collapse,
        family_prediction_summary,
        run_frozen_pilot_diagnostic,
        vq_assignment_summary,
    )
    from prototype.model_data.batching import collate_flat
    from prototype.model_data.vocab import NODE_TYPES


TORCH_REASON = "real PyTorch execution is deferred to the Adroit environment"


def _family_id(physical):
    return source_family_id(
        build_history(physical, GeometryEncoding.CONTINUOUS)
    )


def _sources():
    return (
        source("E", PrimitiveFamily.CIRCLE, extents=(1.0,)),
        source("R", PrimitiveFamily.RECTANGLE_LINES, extents=(2.0,)),
        source("ER", PrimitiveFamily.CAPSULE_LINE_ARC, extents=(1.0, 2.0)),
        source("R", PrimitiveFamily.CIRCLE, extents=(3.0,)),
        source("E", PrimitiveFamily.RECTANGLE_LINES, extents=(1.0,)),
        source("RR", PrimitiveFamily.CAPSULE_LINE_ARC, extents=(2.0, 3.0)),
    )


class DiagnosticConfigurationTests(unittest.TestCase):
    def test_canonical_identity_serialization_and_authorization(self):
        config = ConstrainedV2PilotDiagnosticConfig()
        config.validate()
        validate_diagnostic_partition_authorization(config)
        self.assertEqual(config.diagnostic_identity, DIAGNOSTIC_IDENTITY)
        self.assertEqual(
            config.expected_pilot_source_commit,
            EXPECTED_PILOT_SOURCE_COMMIT,
        )
        self.assertEqual(config.expected_global_step, 136)
        self.assertEqual(config.expected_epoch, 2)
        self.assertEqual(config.expected_validation_count, 68)
        self.assertEqual(
            config.to_json(), ConstrainedV2PilotDiagnosticConfig().to_json()
        )

    def test_rejects_protected_or_malformed_authorization(self):
        invalid = (
            {"training_partition": "test"},
            {"validation_partition": "test"},
            {"validation_partition": "systematic"},
            {"validation_partition_identity": "systematic_validation"},
            {"systematic_partition_accessed": True},
            {"test_partition_accessed": True},
            {"systematic_partition_accessed": 0},
            {"test_partition_accessed": None},
            {"expected_global_step": 135},
            {"expected_epoch": 1},
            {"batch_size": 4},
        )
        for changes in invalid:
            with self.subTest(changes=changes):
                with self.assertRaises(
                    ConstrainedV2PilotDiagnosticError
                ):
                    validate_diagnostic_partition_authorization(
                        replace(
                            ConstrainedV2PilotDiagnosticConfig(), **changes
                        )
                    )

    def test_python_38_grammar_pytorch_111_and_additive_scope(self):
        root = Path(__file__).resolve().parents[1]
        for name in (
            "constrained_v2_pilot_diagnostic_config.py",
            "diagnose_constrained_v2_pilot.py",
        ):
            path = root / name
            text = path.read_text(encoding="utf-8")
            ast.parse(text, filename=str(path), feature_version=(3, 8))
            for token in (
                "torch.compile", "torch.asarray", "torch.func",
                "torch.vmap", "Tensor.scatter_reduce",
                "optimizer.step", ".backward(", "model.train(",
            ):
                self.assertNotIn(token, text)
        for name in (
            "constrained_v2.py", "constrained_v2_losses.py",
            "constrained_v2_autonomous.py", "constrained_v2_pilot.py",
        ):
            self.assertNotIn(
                "diagnose_constrained_v2_pilot",
                (root / name).read_text(encoding="utf-8"),
            )


@unittest.skipUnless(torch is not None, TORCH_REASON)
class DiagnosticTensorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.corpus = root / "corpus"
        physical = _sources()
        validation_ids = {_family_id(item) for item in physical[3:]}
        partitions = {
            _family_id(item): (
                "validation" if _family_id(item) in validation_ids else "train"
            )
            for item in physical
        }
        write_physical_corpus(
            self.corpus, physical, partitions=partitions
        )
        self.data = load_pilot_data(
            self.corpus,
            ConstrainedV2PilotConfig(require_clean_source=False),
        )
        self.model_config = ConstrainedProfileV2Config(
            model_dim=32,
            num_heads=4,
            feedforward_dim=64,
            encoder_layers=1,
            decoder_layers=1,
            dropout=0.0,
            max_nodes=10,
            max_operations=2,
            latent_tokens=2,
            codebook_size=4,
            codebook_dim=8,
            edge_pair_dim=32,
        )
        torch.manual_seed(23)
        self.model = ConstrainedProfileV2Model(self.model_config)
        self.model.eval()

    def test_strict_checkpoint_identity_and_partition_fields(self):
        checkpoint = {name: None for name in PILOT_CHECKPOINT_FIELDS}
        checkpoint.update({
            "checkpoint_version": 2,
            "checkpoint_kind": "pilot_fixed_epoch",
            "model_name": "B0-FLAT-CONSTRAINED-PROFILE-v2",
            "model_config_version": 2,
            "decoder_contract_version": 2,
            "pilot_identity": "B0-FLAT-CONSTRAINED-PROFILE-v2-iid-pilot-v1",
            "epoch": 2,
            "global_step": 136,
            "completed_epochs": 2,
            "systematic_partition_accessed": False,
            "test_partition_accessed": False,
            "source_provenance": {
                "git_commit": EXPECTED_PILOT_SOURCE_COMMIT,
            },
            "training_partition_state": {
                "partition": "train",
                "partition_identity": "train",
                "family_count": 544,
            },
            "validation_partition_state": {
                "partition": "validation",
                "partition_identity": "iid_validation",
                "family_count": 68,
            },
        })
        config = ConstrainedV2PilotDiagnosticConfig()
        audit = audit_frozen_checkpoint_payload(checkpoint, config)
        self.assertEqual(audit["global_step"], 136)
        self.assertFalse(audit["systematic_partition_accessed"])
        self.assertFalse(audit["test_partition_accessed"])
        for name, value in (
            ("checkpoint_version", 1),
            ("model_config_version", 1),
            ("decoder_contract_version", 1),
            ("pilot_identity", "wrong-pilot"),
            ("global_step", 135),
            ("systematic_partition_accessed", True),
            ("test_partition_accessed", None),
        ):
            with self.subTest(name=name, value=value):
                malformed = dict(checkpoint)
                malformed[name] = value
                with self.assertRaises(
                    ConstrainedV2PilotDiagnosticError
                ):
                    audit_frozen_checkpoint_payload(malformed, config)

    def test_authorization_precedes_checkpoint_or_payload_access(self):
        invalid = replace(
            ConstrainedV2PilotDiagnosticConfig(),
            validation_partition="test",
            output_dir=str(Path(self.temporary.name) / "diagnostic"),
            require_clean_source=False,
        )
        with mock.patch(
            "prototype.flat_baseline.diagnose_constrained_v2_pilot."
            "torch.load"
        ) as checkpoint_load, mock.patch(
            "prototype.flat_baseline.diagnose_constrained_v2_pilot."
            "load_pilot_data"
        ) as data_load, mock.patch(
            "prototype.flat_baseline.diagnose_constrained_v2_pilot."
            "ConstrainedProfileV2Model"
        ) as model:
            with self.assertRaises(
                ConstrainedV2PilotDiagnosticError
            ):
                run_frozen_pilot_diagnostic(
                    self.corpus, "unused.pt", invalid
                )
        checkpoint_load.assert_not_called()
        data_load.assert_not_called()
        model.assert_not_called()

    def test_deterministic_assignments_perplexity_and_dominant_fraction(self):
        vectors = torch.tensor([[-1.0], [-1.0], [1.0], [1.0]])
        codebook = torch.tensor([[-1.0], [1.0]])
        ema = torch.tensor([2.0, 2.0])
        first = vq_assignment_summary(vectors, codebook, ema)
        second = vq_assignment_summary(vectors, codebook, ema)
        self.assertEqual(first, second)
        self.assertEqual(first["assignment_count_per_code"], [2, 2])
        self.assertEqual(first["active_code_ids"], [0, 1])
        self.assertAlmostEqual(first["perplexity"], 2.0)
        self.assertAlmostEqual(first["dominant_code_fraction"], 0.5)
        self.assertEqual(first["distance_to_assigned_code"]["mean"], 0.0)

    def test_encoder_and_codebook_collapse_classification(self):
        base = {
            "aggregate_prequantized_variance": 1.0,
            "inter_vector_distance": {"mean": 1.0},
            "codebook_norm_distribution": {"std": 1.0},
            "ema_cluster_sizes": [1.0, 1.0],
            "dead_code_count": 0,
        }
        encoder = dict(base)
        encoder["aggregate_prequantized_variance"] = 0.0
        encoder["inter_vector_distance"] = {"mean": 0.0}
        self.assertEqual(classify_vq_collapse(encoder), "encoder_collapse")
        codebook = dict(base)
        codebook["codebook_norm_distribution"] = {"std": 0.0}
        self.assertEqual(
            classify_vq_collapse(codebook), "codebook_or_ema_collapse"
        )
        both = dict(encoder)
        both["codebook_norm_distribution"] = {"std": 0.0}
        self.assertEqual(classify_vq_collapse(both), "both")
        self.assertEqual(classify_vq_collapse(base), "neither")

    def test_family_confusion_entropy_margin_and_non_sketch(self):
        logits = torch.tensor([
            [3.0, 1.0, 0.0],
            [0.0, 3.0, 1.0],
            [0.0, 1.0, 3.0],
        ])
        summary = family_prediction_summary(
            logits, torch.tensor([0, 1, 2]), torch.tensor([0, 1, 3])
        )
        self.assertEqual(summary["confusion_matrix"][0][0], 1)
        self.assertEqual(summary["confusion_matrix"][1][1], 1)
        self.assertEqual(summary["confusion_matrix"][2][3], 1)
        self.assertEqual(
            summary["predicted_class_histogram"]["non_sketch"], 1
        )
        self.assertAlmostEqual(summary["accuracy"], 2.0 / 3.0)
        self.assertGreater(summary["average_logit_margin"], 0.0)
        self.assertGreater(summary["average_entropy"], 0.0)
        self.assertTrue(math.isfinite(summary["family_loss"]))

    def test_hybrid_arm_boundaries_and_autonomous_target_independence(self):
        partition = self.data.validation
        batch = collate_flat(partition.flat_examples)
        inputs = batch.to_torch(torch)
        target = batch.target.to_torch(torch)
        profiles = profile_targets_for_loss(batch.target, inputs["geometry"])
        with torch.no_grad():
            output = self.model(
                target=target, profile_targets=profiles, **inputs
            )
            autonomous = greedy_decode_v2(
                self.model,
                inputs,
                node_counts=target["node_mask"].sum(dim=1),
                node_count_source="authorized_validation_length",
            )
            arms = build_hybrid_arms(
                output,
                target,
                profiles,
                partition.physical_examples,
                autonomous,
            )
        self.assertEqual(set(arms), set("ABCDEF"))
        self.assertIs(arms["A"][0], autonomous[0])
        self.assertFalse(HYBRID_ARM_CONTRACTS["A"]["oracle"])
        self.assertTrue(all(
            HYBRID_ARM_CONTRACTS[name]["oracle"]
            for name in "CDEF"
        ))
        predicted_retained = torch.stack(tuple(
            logits.argmax(-1) for logits in output.categorical_logits
        ), dim=-1)
        physical = partition.physical_examples[0]
        for position in range(len(physical.nodes)):
            self.assertEqual(
                arms["C"][0].raw_nodes[position].node_type_id,
                target["node_type_ids"][0, position].item(),
            )
            self.assertEqual(
                tuple(
                    arms["C"][0].raw_nodes[position].categorical_ids[index]
                    for index in (0, 1, 2, 3, 8)
                ),
                tuple(predicted_retained[0, position].tolist()),
            )
            self.assertEqual(
                tuple(
                    arms["D"][0].raw_nodes[position].categorical_ids[index]
                    for index in (0, 1, 2, 3, 8)
                ),
                tuple(target["categorical_attributes"][
                    0, position, REMAINING_CATEGORICAL_TARGET_INDICES
                ].tolist()),
            )
        self.assertEqual(
            arms["E"][0].predicted_operation_count,
            len(physical.operation_sequence),
        )
        self.assertEqual(arms["F"][0].raw_edges, arms["A"][0].raw_edges)
        self.assertEqual(
            arms["F"][0].raw_operation_pointers,
            arms["A"][0].raw_operation_pointers,
        )
        mutated_family = profiles.family_ids.clone()
        mutated_family[profiles.sketch_mask] = (
            mutated_family[profiles.sketch_mask] + 1
        ) % 3
        mutated_profiles = replace(profiles, family_ids=mutated_family)
        mutated = build_hybrid_arms(
            output,
            target,
            mutated_profiles,
            partition.physical_examples,
            autonomous,
        )
        self.assertIs(mutated["A"][0], autonomous[0])
        self.assertEqual(mutated["A"], arms["A"])

    def test_first_failure_aggregation_and_classification_rules(self):
        records = (
            {
                "earliest_failing_node_position": 1,
                "authoritative_node_type": "sketch",
                "predicted_node_type": "axis",
                "applicability_field_mismatches": ["reference_plane"],
                "reference_plane_failure_subtype": None,
                "operation_count": 1,
                "failure_code": "invalid_node_grammar",
            },
            {
                "earliest_failing_node_position": 2,
                "authoritative_node_type": "axis",
                "predicted_node_type": "sketch",
                "applicability_field_mismatches": [],
                "reference_plane_failure_subtype": "basis_or_origin",
                "operation_count": 2,
                "failure_code": "invalid_reference_plane_geometry",
            },
        )
        aggregate = _aggregate_first_failures(records)
        self.assertEqual(
            aggregate["earliest_failing_node_position_histogram"],
            {"1": 1, "2": 1},
        )
        self.assertEqual(
            aggregate["applicability_field_mismatch_counts"],
            {"reference_plane": 1},
        )
        arm = {
            name: {
                "validity": 0.0,
                "failure_reason_histogram": {},
                "total_count": 10,
            }
            for name in "ABCDEF"
        }
        arm["B"]["validity"] = 0.2
        arm["E"]["validity"] = 0.8
        arm["F"]["validity"] = 0.2
        vq = {
            name: {
                "active_code_count": 1,
                "dominant_code_fraction": 1.0,
                "collapse_classification": "encoder_collapse",
            }
            for name in ("train", "iid_validation")
        }
        classifications = classify_diagnostic(arm, vq)
        self.assertIn(
            "discrete_grammar_is_primary_bottleneck", classifications
        )
        self.assertIn(
            "autoregressive_exposure_is_primary_bottleneck", classifications
        )
        self.assertIn(
            "family_classification_is_primary_bottleneck", classifications
        )
        self.assertIn("vq_collapse_is_likely_contributor", classifications)

    def test_read_only_vq_model_rng_and_finite_json(self):
        state_before = {
            name: value.clone() for name, value in self.model.state_dict().items()
        }
        rng_before = capture_rng_state(torch)
        first = _vq_partition_summary(
            self.model, self.data.validation, 2, torch.device("cpu")
        )
        second = _vq_partition_summary(
            self.model, self.data.validation, 2, torch.device("cpu")
        )
        self.assertEqual(first, second)
        self.assertTrue(_tree_equal(state_before, self.model.state_dict()))
        self.assertTrue(_tree_equal(rng_before, capture_rng_state(torch)))
        for value in (math.nan, math.inf, -math.inf):
            with self.assertRaises(ConstrainedV2PilotDiagnosticError):
                _require_finite_json({"value": value})


if __name__ == "__main__":
    unittest.main()
