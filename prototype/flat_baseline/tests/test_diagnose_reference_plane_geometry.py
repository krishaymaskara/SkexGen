"""Focused Stage 2H reference-plane diagnostic tests."""

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
from prototype.controlled_data.factors import PrimitiveFamily, ReferencePlane
from prototype.controlled_data.identity import source_family_id
from prototype.model_data.geometry import (
    GEOMETRY_CHANNELS,
    GEOMETRY_CHANNEL_SCALES,
    GEOMETRY_WIDTH,
)
from prototype.model_data.tests.fixtures import source, write_physical_corpus
from prototype.representation.model import GeometryEncoding

from prototype.flat_baseline.constrained_v2_reference_plane_diagnostic_config import (
    DIAGNOSTIC_IDENTITY,
    EXPECTED_CHECKPOINT_SHA256,
    ReferencePlaneDiagnosticConfig,
    ReferencePlaneDiagnosticError,
    validate_reference_plane_partition_authorization,
)

if torch is not None:
    from prototype.constrained_profile_decoder import profile_targets_for_loss
    from prototype.flat_baseline.checkpointing import capture_rng_state
    from prototype.flat_baseline.constrained_v2 import (
        NON_PROFILE_GEOMETRY_INDICES,
        REMAINING_CATEGORICAL_TARGET_INDICES,
        ConstrainedProfileV2Model,
        scatter_non_profile_geometry,
        select_non_profile_geometry,
    )
    from prototype.flat_baseline.constrained_v2_autonomous import (
        greedy_decode_v2,
        validate_and_convert_v2_autonomous_prediction,
    )
    from prototype.flat_baseline.constrained_v2_config import ConstrainedProfileV2Config
    from prototype.flat_baseline.constrained_v2_conversion import (
        construct_v2_predicted_node_tensors,
    )
    from prototype.flat_baseline.constrained_v2_pilot import load_pilot_data
    from prototype.flat_baseline.constrained_v2_pilot_config import ConstrainedV2PilotConfig
    from prototype.flat_baseline.diagnose_constrained_v2_pilot import (
        _tree_equal,
        build_hybrid_arms,
    )
    from prototype.flat_baseline.diagnose_reference_plane_geometry import (
        ARM_CONTRACTS,
        ATOMIC_CONSTRAINT_ORDER,
        CANONICAL_FRAMES,
        OUTPUT_ARM_ORDER,
        REFERENCE_PLANE_CHANNELS,
        _partition_diagnostic,
        _require_finite_json,
        build_reference_plane_arms,
        canonical_reference_plane,
        canonicalize_autonomous_reference_planes,
        classify_reference_plane_diagnostic,
        evaluate_reference_plane,
        project_reference_plane,
        project_reference_plane_basis,
        project_reference_plane_origin,
        reference_plane_contract,
        run_frozen_reference_plane_diagnostic,
    )
    from prototype.flat_baseline.autonomous import derive_geometry_mask
    from prototype.model_data.batching import collate_flat
    from prototype.model_data.vocab import NODE_TYPES, REFERENCE_PLANES


TORCH_REASON = "real PyTorch execution is deferred to the Adroit environment"


def _family_id(physical):
    return source_family_id(
        build_history(physical, GeometryEncoding.CONTINUOUS)
    )


def _sources():
    return (
        source("E", PrimitiveFamily.CIRCLE, plane=ReferencePlane.XY, extents=(1.0,)),
        source("R", PrimitiveFamily.RECTANGLE_LINES, plane=ReferencePlane.XZ, extents=(2.0,)),
        source("ER", PrimitiveFamily.CAPSULE_LINE_ARC, plane=ReferencePlane.YZ, extents=(1.0, 2.0)),
        source("R", PrimitiveFamily.CIRCLE, plane=ReferencePlane.XY, extents=(3.0,)),
        source("E", PrimitiveFamily.RECTANGLE_LINES, plane=ReferencePlane.XZ, extents=(1.0,)),
        source("RR", PrimitiveFamily.CAPSULE_LINE_ARC, plane=ReferencePlane.YZ, extents=(2.0, 3.0)),
    )


class ReferencePlaneStaticTests(unittest.TestCase):
    def test_exact_channel_map_and_frozen_authorization(self):
        self.assertEqual(tuple(GEOMETRY_CHANNELS[:9]), (
            "plane_origin_x", "plane_origin_y", "plane_origin_z",
            "plane_x_axis_x", "plane_x_axis_y", "plane_x_axis_z",
            "plane_y_axis_x", "plane_y_axis_y", "plane_y_axis_z",
        ))
        self.assertEqual(tuple(GEOMETRY_CHANNEL_SCALES[:9]), (4.0, 4.0, 4.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0))
        config = ReferencePlaneDiagnosticConfig()
        config.validate()
        validate_reference_plane_partition_authorization(config)
        self.assertEqual(config.diagnostic_identity, DIAGNOSTIC_IDENTITY)
        self.assertEqual(config.expected_checkpoint_sha256, EXPECTED_CHECKPOINT_SHA256)

    def test_rejects_protected_or_malformed_authorization(self):
        changes = (
            {"training_partition": "test"},
            {"validation_partition": "systematic"},
            {"validation_partition_identity": "test"},
            {"systematic_partition_accessed": True},
            {"test_partition_accessed": True},
            {"systematic_partition_accessed": 0},
            {"test_partition_accessed": None},
            {"expected_checkpoint_sha256": "0" * 64},
            {"batch_size": 4},
        )
        for item in changes:
            with self.subTest(item=item), self.assertRaises(ReferencePlaneDiagnosticError):
                validate_reference_plane_partition_authorization(
                    replace(ReferencePlaneDiagnosticConfig(), **item)
                )

    def test_python_38_pytorch_111_and_additive_read_only_scope(self):
        root = Path(__file__).resolve().parents[1]
        for name in (
            "constrained_v2_reference_plane_diagnostic_config.py",
            "diagnose_reference_plane_geometry.py",
        ):
            path = root / name
            text = path.read_text(encoding="utf-8")
            ast.parse(text, filename=str(path), feature_version=(3, 8))
            for token in (
                "torch.compile", "torch.asarray", "torch.func", "torch.vmap",
                "optimizer.step", ".backward(", "model.train(",
                "torch.optim", "run_constrained_v2_pilot",
            ):
                self.assertNotIn(token, text)
        for name in (
            "constrained_v2.py", "constrained_v2_conversion.py",
            "constrained_v2_autonomous.py", "constrained_v2_losses.py",
        ):
            self.assertNotIn(
                "diagnose_reference_plane_geometry",
                (root / name).read_text(encoding="utf-8"),
            )


@unittest.skipUnless(torch is not None, TORCH_REASON)
class ReferencePlaneTensorTests(unittest.TestCase):
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
        write_physical_corpus(self.corpus, physical, partitions=partitions)
        self.data = load_pilot_data(
            self.corpus, ConstrainedV2PilotConfig(require_clean_source=False)
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
        torch.manual_seed(41)
        self.model = ConstrainedProfileV2Model(self.model_config)
        self.model.eval()

    def _batch_output(self):
        partition = self.data.validation
        batch = collate_flat(partition.flat_examples)
        inputs = batch.to_torch(torch)
        target = batch.target.to_torch(torch)
        profiles = profile_targets_for_loss(batch.target, inputs["geometry"])
        with torch.no_grad():
            output = self.model(
                target=target, profile_targets=profiles, **inputs
            )
        return partition, batch, inputs, target, profiles, output

    def test_contract_cross_checks_builder_serializer_loader_mask_and_frames(self):
        contract = reference_plane_contract()
        self.assertEqual(tuple(item["channel"] for item in contract["channels"]), REFERENCE_PLANE_CHANNELS)
        self.assertEqual(contract["serialization_order"], ["origin", "x_axis", "y_axis"])
        self.assertEqual(contract["canonical_frames"]["XZ"]["y_axis"], [0.0, 0.0, 1.0])
        self.assertIn("strict_converter", contract["independent_implementations_checked"])

    def test_every_atomic_constraint_and_controlled_frame_distinction(self):
        mask = (True,) * 9 + (False,) * (GEOMETRY_WIDTH - 9)
        canonical = (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
        valid = evaluate_reference_plane(canonical, mask, "XY")
        self.assertTrue(valid["controlled_contract_pass"])
        self.assertEqual(tuple(valid["constraints"]), ATOMIC_CONSTRAINT_ORDER)
        mutations = {
            "applicability_mask": (canonical, (False,) * GEOMETRY_WIDTH),
            "finiteness": (canonical[:3] + (math.nan,) + canonical[4:], mask),
            "normalized_bounds": (canonical[:3] + (2.0,) + canonical[4:], mask),
            "origin_zero": ((0.1,) + canonical[1:], mask),
            "x_axis_nonzero": (canonical[:3] + (0.0, 0.0, 0.0) + canonical[6:], mask),
            "y_axis_nonzero": (canonical[:6] + (0.0, 0.0, 0.0), mask),
            "x_axis_unit": (canonical[:3] + (0.5, 0.0, 0.0) + canonical[6:], mask),
            "y_axis_unit": (canonical[:6] + (0.0, 0.5, 0.0), mask),
            "axes_orthogonal": (canonical[:6] + (1.0, 0.0, 0.0), mask),
            "categorical_x_axis": ((0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0), mask),
            "categorical_y_axis": ((0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0), mask),
            "canonical_frame": ((0.1,) + canonical[1:], mask),
        }
        for name, (values, candidate_mask) in mutations.items():
            with self.subTest(name=name):
                result = evaluate_reference_plane(values, candidate_mask, "XY")
                self.assertFalse(result["constraints"][name]["passed"])
        rotated = evaluate_reference_plane(
            (0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0, 0.0),
            mask,
            "XY",
        )
        self.assertTrue(rotated["mathematical_form_pass"])
        self.assertFalse(rotated["controlled_contract_pass"])
        self.assertGreater(rotated["metrics"]["derived_basis_determinant"], 0.0)

    def test_projection_is_deterministic_target_free_and_handles_degeneracy(self):
        predicted = (0.2, -0.3, 0.4, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        origin_only = project_reference_plane_origin(predicted)
        basis_only = project_reference_plane_basis(predicted)
        combined = project_reference_plane(predicted)
        self.assertEqual(origin_only[:3], (0.0, 0.0, 0.0))
        self.assertEqual(origin_only[3:9], predicted[3:9])
        self.assertEqual(basis_only[:3], predicted[:3])
        self.assertEqual(combined[:3], origin_only[:3])
        self.assertEqual(combined[3:9], basis_only[3:9])
        self.assertEqual(combined, project_reference_plane(predicted))
        mask = (True,) * 9 + (False,) * 30
        for plane in CANONICAL_FRAMES:
            result = evaluate_reference_plane(combined, mask, plane)
            self.assertTrue(result["mathematical_form_pass"])
        noncanonical = project_reference_plane_basis(
            (0.0, 0.0, 0.0, -0.9, 0.1, 0.0, 0.2, -0.8, 0.3)
        )
        evaluation = evaluate_reference_plane(noncanonical, mask, "XY")
        self.assertTrue(evaluation["mathematical_form_pass"])
        self.assertFalse(evaluation["controlled_contract_pass"])
        with self.assertRaises(ValueError):
            project_reference_plane((math.inf,) + predicted[1:])
        with self.assertRaises(ValueError):
            project_reference_plane((2.0,) + predicted[1:])

    def test_sentinel_routing_detects_permutation_shift_mask_and_profile_source(self):
        node_types = torch.tensor([[NODE_TYPES.id("reference_plane")]])
        retained = torch.tensor([[[1, 1, 1, REFERENCE_PLANES.id("XY"), 1]]])
        family_logits = torch.tensor([[[9.0, 8.0, 7.0]]])
        raw_parameters = torch.tensor([[[0.91, 0.92, 0.93]]])
        sentinels = torch.tensor([[[-0.90 + index * 0.10 for index in range(15)]]])
        records = construct_v2_predicted_node_tensors(
            node_types, retained, family_logits, raw_parameters, sentinels
        )
        expected = sentinels[0, 0, :9]
        torch.testing.assert_close(records.geometry[0, 0, :9], expected, rtol=0, atol=0)
        self.assertTrue(records.geometry_mask[0, 0, :9].all())
        self.assertFalse(records.geometry_mask[0, 0, 9:].any())
        scattered = scatter_non_profile_geometry(sentinels)
        torch.testing.assert_close(
            scattered[..., NON_PROFILE_GEOMETRY_INDICES], sentinels,
            rtol=0, atol=0,
        )
        torch.testing.assert_close(
            select_non_profile_geometry(scattered), sentinels, rtol=0, atol=0
        )
        self.assertFalse(torch.equal(records.geometry[0, 0, :3], raw_parameters[0, 0]))

    def test_h0_exactly_reproduces_stage2g_e_and_oracle_boundaries(self):
        partition, unused_batch, inputs, target, profiles, output = self._batch_output()
        with torch.no_grad():
            autonomous = greedy_decode_v2(
                self.model, inputs,
                node_counts=target["node_mask"].sum(dim=1),
                node_count_source="authorized_validation_length",
            )
            stage2g = build_hybrid_arms(
                output, target, profiles, partition.physical_examples, autonomous
            )
            arms, h6_audits = build_reference_plane_arms(
                output, target, profiles, partition.physical_examples,
                autonomous,
            )
        self.assertEqual(arms["H0"], stage2g["E"])
        self.assertEqual(len(h6_audits), len(partition.physical_examples))
        self.assertEqual(tuple(OUTPUT_ARM_ORDER), (
            "H0", "H1a", "H1b", "H1c", "H2", "H3a", "H3b",
            "H3c", "H4", "H5", "H6",
        ))
        self.assertFalse(ARM_CONTRACTS["H4"]["enabled"])
        for example_index, example in enumerate(partition.physical_examples):
            h0 = arms["H0"][example_index]
            h1a = arms["H1a"][example_index]
            h1b = arms["H1b"][example_index]
            h1c = arms["H1c"][example_index]
            h2 = arms["H2"][example_index]
            h3a = arms["H3a"][example_index]
            h3b = arms["H3b"][example_index]
            h3c = arms["H3c"][example_index]
            h5 = arms["H5"][example_index]
            h6 = arms["H6"][example_index]
            plane = next(index for index, node in enumerate(example.nodes) if node.node_type == "reference_plane")
            predicted = h0.raw_nodes[plane].normalized_geometry
            authoritative = example.target.geometry[plane]
            self.assertEqual(h1a.raw_nodes[plane].normalized_geometry[:3], authoritative[:3])
            self.assertEqual(h1a.raw_nodes[plane].normalized_geometry[3:], predicted[3:])
            self.assertEqual(h1b.raw_nodes[plane].normalized_geometry[:3], predicted[:3])
            self.assertEqual(h1b.raw_nodes[plane].normalized_geometry[3:9], authoritative[3:9])
            self.assertEqual(h1b.raw_nodes[plane].normalized_geometry[9:], predicted[9:])
            self.assertEqual(h1c.raw_nodes[plane].normalized_geometry[:9], authoritative[:9])
            self.assertEqual(h1c.raw_nodes[plane].normalized_geometry[9:], predicted[9:])
            self.assertEqual(authoritative[:9], canonical_reference_plane(example.nodes[plane].reference_plane))
            self.assertEqual(h3a.raw_nodes[plane].normalized_geometry[:9], project_reference_plane_origin(predicted[:9]))
            self.assertEqual(h3b.raw_nodes[plane].normalized_geometry[:9], project_reference_plane_basis(predicted[:9]))
            self.assertEqual(h3c.raw_nodes[plane].normalized_geometry[:9], project_reference_plane(predicted[:9]))
            for position in range(len(example.nodes)):
                for channel in NON_PROFILE_GEOMETRY_INDICES:
                    self.assertEqual(h2.raw_nodes[position].normalized_geometry[channel], example.target.geometry[position][channel])
                for channel in range(9, 33):
                    self.assertEqual(h2.raw_nodes[position].normalized_geometry[channel], h0.raw_nodes[position].normalized_geometry[channel])
                for channel in NON_PROFILE_GEOMETRY_INDICES:
                    self.assertEqual(h5.raw_nodes[position].normalized_geometry[channel], h0.raw_nodes[position].normalized_geometry[channel])
            self.assertEqual(h6.raw_edges, autonomous[example_index].raw_edges)
            self.assertEqual(
                h6.raw_operation_pointers,
                autonomous[example_index].raw_operation_pointers,
            )
            for before, after in zip(
                autonomous[example_index].raw_nodes, h6.raw_nodes
            ):
                self.assertEqual(before.node_type_id, after.node_type_id)
                self.assertEqual(before.categorical_ids, after.categorical_ids)
                self.assertEqual(
                    before.normalized_geometry[9:],
                    after.normalized_geometry[9:],
                )

    def test_exact_lookup_and_h6_invalid_categories_never_use_targets(self):
        self.assertEqual(
            canonical_reference_plane("XY"),
            (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        )
        self.assertEqual(
            canonical_reference_plane("XZ"),
            (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        )
        self.assertEqual(
            canonical_reference_plane("YZ"),
            (0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        )
        partition, unused_batch, inputs, target, unused_profiles, unused_output = self._batch_output()
        with torch.no_grad():
            autonomous = greedy_decode_v2(
                self.model, inputs,
                node_counts=target["node_mask"].sum(dim=1),
                node_count_source="authorized_validation_length",
            )[0]
        base = autonomous.raw_nodes[0]
        categories = list(base.categorical_ids)
        categories[3] = REFERENCE_PLANES.id("XZ")
        geometry = tuple(-0.37 for unused in range(GEOMETRY_WIDTH))
        valid_node = replace(
            base,
            node_type_id=NODE_TYPES.id("reference_plane"),
            categorical_ids=tuple(categories),
            normalized_geometry=geometry,
            derived_geometry_mask=derive_geometry_mask(
                NODE_TYPES.id("reference_plane"), tuple(categories)
            ),
        )
        tail = tuple(
            replace(
                node,
                node_type_id=NODE_TYPES.id("profile"),
                derived_geometry_mask=derive_geometry_mask(
                    NODE_TYPES.id("profile"), node.categorical_ids
                ),
            )
            if node.node_type_id == NODE_TYPES.id("reference_plane") else node
            for node in autonomous.raw_nodes[1:]
        )
        valid_prediction = replace(
            autonomous, raw_nodes=(valid_node,) + tail
        )
        canonicalized, audit = canonicalize_autonomous_reference_planes(
            valid_prediction
        )
        self.assertEqual(
            canonicalized.raw_nodes[0].normalized_geometry[:9],
            canonical_reference_plane("XZ"),
        )
        self.assertEqual(
            canonicalized.raw_nodes[0].normalized_geometry[9:], geometry[9:]
        )
        self.assertEqual(audit["exact_canonicalization_count"], 1)
        self.assertFalse(audit["target_geometry_used"])
        self.assertFalse(audit["authoritative_category_used"])

        for invalid_id in (REFERENCE_PLANES.id(None), len(REFERENCE_PLANES.tokens) + 4):
            with self.subTest(invalid_id=invalid_id):
                invalid_categories = list(categories)
                invalid_categories[3] = invalid_id
                invalid_node = replace(
                    valid_node, categorical_ids=tuple(invalid_categories)
                )
                invalid_prediction = replace(
                    valid_prediction,
                    raw_nodes=(invalid_node,) + valid_prediction.raw_nodes[1:],
                )
                unchanged, invalid_audit = (
                    canonicalize_autonomous_reference_planes(
                        invalid_prediction
                    )
                )
                self.assertEqual(unchanged.raw_nodes[0], invalid_node)
                self.assertEqual(
                    invalid_audit["missing_or_invalid_predicted_category_count"],
                    1,
                )
                self.assertEqual(
                    invalid_audit["exact_canonicalization_count"], 0
                )
                self.assertFalse(invalid_audit["target_fallback_used"])
                conversion = validate_and_convert_v2_autonomous_prediction(
                    unchanged,
                    max_operations=self.model_config.max_operations,
                )
                self.assertFalse(conversion.controlled_domain.valid)

    def test_h3_does_not_use_target_geometry(self):
        partition, unused_batch, unused_inputs, target, profiles, output = self._batch_output()
        original, unused_audits = build_reference_plane_arms(
            output, target, profiles, partition.physical_examples
        )
        changed_examples = []
        for example in partition.physical_examples:
            plane = next(
                index for index, node in enumerate(example.nodes)
                if node.node_type == "reference_plane"
            )
            old_name = example.nodes[plane].reference_plane
            new_name = {"XY": "XZ", "XZ": "YZ", "YZ": "XY"}[old_name]
            geometry = list(example.target.geometry)
            geometry[plane] = (
                canonical_reference_plane(new_name) + geometry[plane][9:]
            )
            nodes = list(example.nodes)
            nodes[plane] = replace(nodes[plane], reference_plane=new_name)
            changed_examples.append(replace(
                example,
                nodes=tuple(nodes),
                target=replace(example.target, geometry=tuple(geometry)),
            ))
        changed, unused_changed_audits = build_reference_plane_arms(
            output, target, profiles, tuple(changed_examples)
        )
        for name in ("H3a", "H3b", "H3c"):
            self.assertEqual(original[name], changed[name])
        self.assertEqual(original["H1a"], changed["H1a"])
        for name in ("H1b", "H1c"):
            self.assertNotEqual(original[name], changed[name])

    def test_partition_authorization_precedes_checkpoint_and_payload_access(self):
        config = replace(
            ReferencePlaneDiagnosticConfig(),
            validation_partition="test",
            output_dir=str(Path(self.temporary.name) / "diagnostic"),
            require_clean_source=False,
        )
        with mock.patch(
            "prototype.flat_baseline.diagnose_reference_plane_geometry.torch.load"
        ) as checkpoint_load, mock.patch(
            "prototype.flat_baseline.diagnose_reference_plane_geometry.load_pilot_data"
        ) as data_load, mock.patch(
            "prototype.flat_baseline.diagnose_reference_plane_geometry.ConstrainedProfileV2Model"
        ) as model:
            with self.assertRaises(ReferencePlaneDiagnosticError):
                run_frozen_reference_plane_diagnostic(
                    self.corpus, "unused.pt", config
                )
        checkpoint_load.assert_not_called()
        data_load.assert_not_called()
        model.assert_not_called()

    def test_read_only_partition_diagnostic_and_finite_json(self):
        state_before = {name: value.clone() for name, value in self.model.state_dict().items()}
        rng_before = capture_rng_state(torch)
        result = _partition_diagnostic(
            self.model, self.data.validation, self.model_config, 2,
            torch.device("cpu"), include_arms=True,
        )
        self.assertEqual(result["partition_summary"]["example_count"], 3)
        self.assertTrue(_tree_equal(state_before, self.model.state_dict()))
        self.assertTrue(_tree_equal(rng_before, capture_rng_state(torch)))
        _require_finite_json(result)
        for value in (math.nan, math.inf, -math.inf):
            with self.assertRaises(Exception):
                _require_finite_json({"value": value})

    def test_classification_rules_are_predeclared_and_h5_is_not_a_repair(self):
        summaries = {
            name: {"validity": 0.0, "controlled_plane_pass_rate": 0.0}
            for name in ("H1a", "H1b", "H1c", "H3c")
        }
        summaries["H0"] = {"validity": 0.0}
        summaries["H2"] = {"validity": 1.0}
        summaries["H5"] = {"validity": 0.0}
        summaries["H6"] = {
            "validity": 0.2,
            "validity_improvement_over_arm_a": 0.2,
        }
        summaries["H1a"]["controlled_plane_pass_rate"] = 0.1
        summaries["H1b"]["controlled_plane_pass_rate"] = 0.1
        summaries["H1c"]["controlled_plane_pass_rate"] = 0.9
        summaries["H3c"]["controlled_plane_pass_rate"] = 0.2
        classifications = classify_reference_plane_diagnostic(summaries)
        self.assertIn(
            "reference_plane_origin_and_basis_both_contribute",
            classifications,
        )
        self.assertIn(
            "exact_categorical_canonicalization_is_required",
            classifications,
        )
        self.assertIn(
            "autonomous_categorical_canonicalization_is_viable",
            classifications,
        )
        summaries["H5"]["validity"] = 1.0
        self.assertEqual(
            classifications,
            classify_reference_plane_diagnostic(summaries),
        )
        summaries["H1a"]["controlled_plane_pass_rate"] = 0.8
        summaries["H1b"]["controlled_plane_pass_rate"] = 0.1
        summaries["H1c"]["controlled_plane_pass_rate"] = 0.8
        summaries["H3c"]["controlled_plane_pass_rate"] = 0.8
        origin = classify_reference_plane_diagnostic(summaries)
        self.assertIn("reference_plane_origin_is_primary", origin)
        self.assertIn("general_orthonormal_projection_is_sufficient", origin)
        summaries["H1a"]["controlled_plane_pass_rate"] = 0.1
        summaries["H1b"]["controlled_plane_pass_rate"] = 0.8
        basis = classify_reference_plane_diagnostic(summaries)
        self.assertIn("reference_plane_basis_is_primary", basis)
        for name in ("H1a", "H1b", "H1c", "H3c"):
            summaries[name]["validity"] = 0.0
            summaries[name]["controlled_plane_pass_rate"] = 0.0
        summaries["H6"]["validity_improvement_over_arm_a"] = 0.0
        routing = classify_reference_plane_diagnostic(summaries)
        self.assertIn(
            "reference_plane_channel_routing_or_contract_mismatch", routing
        )


if __name__ == "__main__":
    unittest.main()
