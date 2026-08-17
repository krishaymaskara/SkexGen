"""Corpus-free end-to-end and structural integration tests for ADR-0013."""

from __future__ import annotations

import ast
from dataclasses import fields, replace
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from prototype.graph_encoder import grid_magnitude as gm
from prototype.graph_encoder.batching import build_paired_batch
from prototype.graph_encoder.grid_magnitude_audit import (
    audit_grid_magnitude_sources,
)
from prototype.graph_encoder.tests.fixtures import procedural_fixture


try:
    import torch
except ImportError:  # pragma: no cover - local contract runtime may omit it
    torch = None


PACKAGE = Path(__file__).parents[1]
RUNNER = PACKAGE / "adroit" / "ge1_grid_magnitude_validation_cpu.slurm"
TORCH_REASON = "grid integration requires the authoritative PyTorch runtime"


class GridIntegrationStructuralTests(unittest.TestCase):
    """Locally runnable checks shared verbatim with the Adroit runner."""

    def test_runner_and_production_sources_pass_shared_structural_audit(self):
        record = audit_grid_magnitude_sources(PACKAGE, RUNNER)
        self.assertEqual(
            record["version"], "GE1-GRID-MAGNITUDE-SOURCE-AUDIT-v1"
        )
        self.assertEqual(record["repository_read_only_bind_count"], 1)
        self.assertFalse(record["submission_command_invoked"])
        self.assertFalse(record["scientific_data_path_declared"])
        self.assertFalse(record["scientific_command_invoked"])

    def test_real_training_call_passes_teacher_ordinal_logits(self):
        source = (PACKAGE / "training.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        calls = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "common_ge1_loss"
        ]
        self.assertEqual(len(calls), 1)
        keyword = next(
            item for item in calls[0].keywords
            if item.arg == "grid_magnitude_logits"
        )
        self.assertIsInstance(keyword.value, ast.Attribute)
        self.assertEqual(keyword.value.attr, "grid_magnitude_logits")
        self.assertIsInstance(keyword.value.value, ast.Name)
        self.assertEqual(keyword.value.value.id, "teacher")

    def test_finite_training_audit_receives_the_full_teacher_wrapper(self):
        source = (PACKAGE / "training.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        calls = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_assert_finite_tree"
        ]
        self.assertEqual(len(calls), 1)
        self.assertIsInstance(calls[0].args[0], ast.Name)
        self.assertEqual(calls[0].args[0].id, "teacher")

    def test_frozen_focused_test_counts_remain_exact(self):
        loader = unittest.defaultTestLoader
        pure = loader.loadTestsFromName(
            "prototype.graph_encoder.tests.test_grid_magnitude_contract"
        )
        runtime = loader.loadTestsFromName(
            "prototype.graph_encoder.tests.test_grid_magnitude_runtime"
        )
        self.assertEqual(pure.countTestCases(), 28)
        self.assertEqual(runtime.countTestCases(), 14)

    def test_grid_decode_interfaces_are_target_free(self):
        source = (PACKAGE / "shared_decoder.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        names = {
            "grid_magnitude_logits", "parameterize_remaining_geometry",
            "decode_prefix", "forward",
        }
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name in names:
                arguments = {
                    item.arg for item in node.args.args + node.args.kwonlyargs
                }
                with self.subTest(interface=node.name):
                    self.assertFalse(arguments & {"target", "label", "labels"})

    def test_metric_mask_routes_magnitude_without_erasing_axis(self):
        from prototype.graph_encoder.metrics import (
            _geometry_errors,
            _grid_geometry_metric_applicability,
        )

        # Revolve carries the four learned axis channels as well as magnitude;
        # extrusion legitimately has zero axis applicability.
        target = procedural_fixture("R").target
        nodes = tuple(
            SimpleNamespace(normalized_geometry=row)
            for row in target.geometry
        )
        prediction = SimpleNamespace(
            node_prediction=SimpleNamespace(raw_nodes=nodes)
        )
        historical = _geometry_errors(prediction, target)
        grid = _geometry_errors(
            prediction, target, exclude_grid_magnitudes=True
        )
        applicability = _grid_geometry_metric_applicability(target)
        self.assertEqual(
            historical["axis"]["applicable_channel_denominator"],
            grid["axis"]["applicable_channel_denominator"],
        )
        self.assertGreater(
            grid["axis"]["applicable_channel_denominator"], 0
        )
        self.assertGreater(
            historical["operation_parameter"][
                "applicable_channel_denominator"
            ],
            0,
        )
        self.assertEqual(
            grid["operation_parameter"]["applicable_channel_denominator"], 0
        )
        self.assertGreater(
            grid["operation_parameter"]["target_channel_denominator"], 0
        )
        self.assertEqual(
            grid["operation_parameter"]["physical_mae"]["reason"],
            "routed_to_versioned_grid_ordinal_metric",
        )
        self.assertFalse(
            grid["operation_parameter"][
                "supported_grid_magnitude_target_treated_as_absent"
            ]
        )
        self.assertGreater(applicability["magnitude_target_channel_count"], 0)
        self.assertEqual(
            applicability["magnitude_target_channel_count"],
            applicability["magnitude_ordinal_metric_channel_count"],
        )
        self.assertFalse(
            applicability["supported_grid_magnitude_target_treated_as_absent"]
        )


@unittest.skipIf(torch is None, TORCH_REASON)
class GridIntegrationTensorTests(unittest.TestCase):
    """Generated-tensor checks only; these are not scientific execution."""

    def setUp(self):
        torch.manual_seed(2026)
        examples = tuple(
            procedural_fixture(name).physical
            for name in ("E", "R", "EE", "RE")
        )
        self.paired = build_paired_batch(examples)
        self.target = self.paired.target.to_torch(torch)
        flat = self.paired.flat_input.to_torch(torch)
        from prototype.constrained_profile_decoder import (
            profile_targets_for_loss,
        )

        self.profiles = profile_targets_for_loss(
            self.paired.target, flat["geometry"]
        )

    def _models(self, parameterization=gm.GRID_MAGNITUDE_PARAMETERIZATION):
        from prototype.graph_encoder.model import build_matched_ge1_models

        return build_matched_ge1_models(
            seed=2026,
            operation_magnitude_parameterization=parameterization,
        )

    def _input(self, arm):
        from prototype.graph_encoder.autonomous import (
            autonomous_input_from_paired,
        )

        return autonomous_input_from_paired(self.paired, arm, torch)

    def _teacher(self, model):
        autonomous = self._input(model.config.encoder)
        return model.teacher_forced(
            autonomous.encoder_input, self.target, self.profiles
        )

    def test_both_arms_complete_teacher_forward_and_structured_loss(self):
        from prototype.graph_encoder.losses import GE1GridLoss, common_ge1_loss

        for model in self._models():
            with self.subTest(arm=model.config.encoder):
                teacher = self._teacher(model)
                self.assertIsNotNone(teacher.grid_magnitude_logits)
                self.assertEqual(
                    tuple(teacher.grid_magnitude_logits.shape),
                    (
                        len(self.paired.family_ids),
                        self.target["node_type_ids"].size(1),
                        len(gm.OPERATION_TYPES),
                        gm.ORDINAL_CUT_COUNT,
                    ),
                )
                loss = common_ge1_loss(
                    teacher.decoder_output,
                    self.target,
                    self.profiles,
                    model.config,
                    grid_magnitude_logits=teacher.grid_magnitude_logits,
                )
                self.assertIsInstance(loss, GE1GridLoss)
                added = sum(loss.grid_magnitude.weighted.values())
                torch.testing.assert_close(
                    loss.total,
                    loss.base_loss.total + added,
                    rtol=0.0,
                    atol=0.0,
                )
                self.assertGreater(
                    sum(loss.grid_magnitude.active_operation_counts.values()), 0
                )

    def test_grid_logits_exist_only_for_grid_identity(self):
        from prototype.graph_encoder.decoder_contract import (
            LEGACY_OPERATION_MAGNITUDE_PARAMETERIZATION,
            POSITIVE_OPERATION_MAGNITUDE_PARAMETERIZATION,
        )

        grid = self._teacher(self._models()[0])
        self.assertIsNotNone(grid.grid_magnitude_logits)
        for identity in (
            LEGACY_OPERATION_MAGNITUDE_PARAMETERIZATION,
            POSITIVE_OPERATION_MAGNITUDE_PARAMETERIZATION,
        ):
            model = self._models(identity)[0]
            with self.subTest(identity=identity):
                self.assertIsNone(self._teacher(model).grid_magnitude_logits)

    def test_ordinal_terms_send_finite_nonzero_gradients_to_head_and_trunk(self):
        from prototype.graph_encoder.grid_magnitude_metrics import (
            engineering_gradient_norms,
        )
        from prototype.graph_encoder.losses import common_ge1_loss

        model = self._models()[0]
        teacher = self._teacher(model)
        loss = common_ge1_loss(
            teacher.decoder_output,
            self.target,
            self.profiles,
            model.config,
            grid_magnitude_logits=teacher.grid_magnitude_logits,
        )
        model.zero_grad(set_to_none=True)
        ordinal_total = sum(loss.grid_magnitude.weighted.values())
        ordinal_total.backward()
        record = engineering_gradient_norms(model)
        for name in ("grid_head", "shared_decoder_trunk"):
            with self.subTest(group=name):
                self.assertTrue(record["groups"][name]["finite"])
                self.assertTrue(record["groups"][name]["nonzero"])
                self.assertGreater(record["groups"][name]["l2_norm"], 0.0)
        self.assertTrue(record["engineering_only"])
        self.assertFalse(record["scientific_training"])

    def test_discrete_decode_has_no_grid_head_gradient_path(self):
        model = self._models()[0]
        teacher = self._teacher(model)
        model.zero_grad(set_to_none=True)
        magnitudes = teacher.decoder_output.remaining_geometry[..., 4:6]
        decoded = model.decoder.grid_magnitude_head.normalized_values(
            teacher.grid_magnitude_logits
        )
        self.assertFalse(decoded.requires_grad)
        if magnitudes.requires_grad:
            gradients = torch.autograd.grad(
                magnitudes.sum(),
                tuple(model.decoder.grid_magnitude_head.parameters()),
                allow_unused=True,
            )
        else:
            gradients = tuple(
                None for unused in model.decoder.grid_magnitude_head.parameters()
            )
        self.assertTrue(all(
            value is None or not bool(value.abs().sum().item())
            for value in gradients
        ))

    def test_historical_identities_keep_exact_graph_v1_loss(self):
        from prototype.graph_baseline.config import GraphV1Config
        from prototype.graph_baseline.losses import graph_v1_loss
        from prototype.graph_encoder.decoder_contract import (
            LEGACY_OPERATION_MAGNITUDE_PARAMETERIZATION,
            POSITIVE_OPERATION_MAGNITUDE_PARAMETERIZATION,
        )
        from prototype.graph_encoder.losses import common_ge1_loss

        for identity in (
            LEGACY_OPERATION_MAGNITUDE_PARAMETERIZATION,
            POSITIVE_OPERATION_MAGNITUDE_PARAMETERIZATION,
        ):
            model = self._models(identity)[0]
            teacher = self._teacher(model)
            observed = common_ge1_loss(
                teacher.decoder_output,
                self.target,
                self.profiles,
                model.config,
            )
            expected = graph_v1_loss(
                teacher.decoder_output,
                self.target,
                self.profiles,
                GraphV1Config(),
            )
            with self.subTest(identity=identity):
                self.assertEqual(
                    tuple(item.name for item in fields(observed)),
                    tuple(item.name for item in fields(expected)),
                )
                for name in observed.per_example:
                    torch.testing.assert_close(
                        observed.per_example[name],
                        expected.per_example[name],
                        rtol=0.0,
                        atol=0.0,
                    )
                torch.testing.assert_close(
                    observed.total, expected.total, rtol=0.0, atol=0.0
                )

    def test_teacher_and_target_free_prefix_decode_share_exact_grid_mapping(self):
        model = self._models()[0].eval()
        teacher = self._teacher(model)
        teacher_values = model.decoder.grid_magnitude_head.normalized_values(
            teacher.grid_magnitude_logits
        )
        torch.testing.assert_close(
            teacher.decoder_output.remaining_geometry[..., 4:6],
            teacher_values,
            rtol=0.0,
            atol=0.0,
        )

        autonomous = self._input(model.config.encoder)
        memory = model.encode(autonomous.encoder_input).memory
        batch = memory.size(0)
        prefix = model.decoder.decode_prefix(
            memory,
            torch.empty(batch, 0, 10, dtype=torch.long),
            torch.empty(batch, 0, 39, dtype=memory.dtype),
            torch.empty(batch, 0, 39, dtype=torch.bool),
        )
        autonomous_logits = model.decoder.grid_magnitude_logits(
            prefix.decoded_states
        )
        autonomous_values = (
            model.decoder.grid_magnitude_head.normalized_values(
                autonomous_logits
            )
        )
        torch.testing.assert_close(
            prefix.remaining_geometry[..., 4:6],
            autonomous_values,
            rtol=0.0,
            atol=0.0,
        )
        parameters = inspect.signature(model.decoder.decode_prefix).parameters
        self.assertFalse(set(parameters) & {"target", "label", "labels"})

    def test_every_class_is_reachable_and_decodes_to_exact_grid_members(self):
        head = self._models()[0].decoder.grid_magnitude_head
        for position, operation_type in enumerate(gm.OPERATION_TYPES):
            with torch.no_grad():
                head.projections[position].weight.zero_()
                head.projections[position].weight[0, 0] = 1.0
                head.first_bias.zero_()
                head.bias_gaps.fill_(0.5413)
            seen = set()
            values = set()
            for scalar in torch.linspace(-6.0, 6.0, 200):
                states = torch.zeros(1, 1, head.model_dim)
                states[0, 0, 0] = scalar
                logits = head(states)
                class_index = int(head.class_indices(logits)[0, 0, position])
                seen.add(class_index)
                values.add(float(head.normalized_values(logits)[0, 0, position]))
            with self.subTest(operation_type=operation_type):
                self.assertEqual(seen, set(range(gm.GRID_CLASS_COUNT)))
                self.assertEqual(values, set(gm.NORMALIZED_GRIDS[operation_type]))

    def test_grid_checkpoint_v2_round_trip_and_cross_identity_preflight(self):
        from prototype.graph_encoder.checkpoint import (
            GE1CheckpointError,
            load_ge1_checkpoint,
            save_ge1_checkpoint,
        )
        from prototype.graph_encoder.decoder_contract import (
            GRID_CHECKPOINT_SCHEMA,
            POSITIVE_OPERATION_MAGNITUDE_PARAMETERIZATION,
        )
        from prototype.graph_encoder.model import GE1Model

        model = self._models()[0]
        revision = "a" * 40
        with tempfile.TemporaryDirectory(prefix="grid-checkpoint-test-") as root:
            path = Path(root) / "grid-v2.pt"
            save_ge1_checkpoint(path, model, code_revision=revision)
            loaded, payload = load_ge1_checkpoint(
                path,
                expected_code_revision=revision,
                expected_operation_magnitude_parameterization=(
                    gm.GRID_MAGNITUDE_PARAMETERIZATION
                ),
            )
            self.assertEqual(payload["checkpoint_schema"], GRID_CHECKPOINT_SCHEMA)
            for name, value in model.state_dict().items():
                torch.testing.assert_close(
                    value, loaded.state_dict()[name], rtol=0.0, atol=0.0
                )
            with patch.object(
                GE1Model, "load_state_dict", autospec=True
            ) as incompatible_load:
                with self.assertRaises(GE1CheckpointError):
                    load_ge1_checkpoint(
                        path,
                        expected_code_revision=revision,
                        expected_operation_magnitude_parameterization=(
                            POSITIVE_OPERATION_MAGNITUDE_PARAMETERIZATION
                        ),
                    )
                incompatible_load.assert_not_called()

    def test_grid_scalar_loss_excludes_only_magnitudes_and_preserves_axis(self):
        from prototype.graph_encoder.losses import common_ge1_loss

        model = self._models()[0]
        teacher = self._teacher(model)
        original_mask = self.target["geometry_mask"].clone()

        baseline = common_ge1_loss(
            teacher.decoder_output,
            self.target,
            self.profiles,
            model.config,
            grid_magnitude_logits=teacher.grid_magnitude_logits,
        )
        magnitude_values = teacher.decoder_output.remaining_geometry.clone()
        magnitude_values[..., 4:6] += 0.25
        changed_magnitude = replace(
            teacher.decoder_output, remaining_geometry=magnitude_values
        )
        magnitude_loss = common_ge1_loss(
            changed_magnitude,
            self.target,
            self.profiles,
            model.config,
            grid_magnitude_logits=teacher.grid_magnitude_logits,
        )
        torch.testing.assert_close(
            baseline.base_loss.remaining_geometry,
            magnitude_loss.base_loss.remaining_geometry,
            rtol=0.0,
            atol=0.0,
        )

        axis_values = teacher.decoder_output.remaining_geometry.clone()
        active_axis = teacher.decoder_output.remaining_geometry_mask[..., :4]
        first = active_axis.nonzero(as_tuple=False)[0]
        axis_values[first[0], first[1], first[2]] += 0.25
        changed_axis = replace(
            teacher.decoder_output, remaining_geometry=axis_values
        )
        axis_loss = common_ge1_loss(
            changed_axis,
            self.target,
            self.profiles,
            model.config,
            grid_magnitude_logits=teacher.grid_magnitude_logits,
        )
        self.assertFalse(torch.equal(
            baseline.base_loss.remaining_geometry,
            axis_loss.base_loss.remaining_geometry,
        ))
        self.assertTrue(torch.equal(
            self.target["geometry_mask"], original_mask
        ))
        applicability = baseline.grid_magnitude.applicability
        self.assertEqual(
            applicability["axis_target_channel_count"],
            applicability["axis_scalar_loss_channel_count"],
        )
        self.assertEqual(
            applicability["magnitude_ordinal_metric_channel_count"],
            applicability["magnitude_target_channel_count"],
        )

    def test_structured_diagnostics_are_complete_and_json_serializable(self):
        from prototype.graph_encoder.grid_magnitude_metrics import (
            engineering_grid_magnitude_diagnostic,
        )
        from prototype.graph_encoder.losses import common_ge1_loss

        model = self._models()[0]
        teacher = self._teacher(model)
        loss = common_ge1_loss(
            teacher.decoder_output,
            self.target,
            self.profiles,
            model.config,
            grid_magnitude_logits=teacher.grid_magnitude_logits,
        )
        model.zero_grad(set_to_none=True)
        loss.total.backward()
        values = loss.grid_magnitude.decoded_normalized_values
        record = engineering_grid_magnitude_diagnostic(
            loss.grid_magnitude,
            teacher_values=values,
            autonomous_values={name: item.clone() for name, item in values.items()},
            active_masks=loss.grid_magnitude.active_masks,
            model=model,
        )
        json.dumps(record, sort_keys=True)
        self.assertEqual(
            record["schema_version"],
            "GE1-GRID-MAGNITUDE-DIAGNOSTICS-v1",
        )
        self.assertTrue(
            record["teacher_forced_autonomous_grid_value_agreement"][
                "exact_agreement"
            ]
        )
        for operation_type in gm.OPERATION_TYPES:
            item = record["operation_types"][operation_type]
            self.assertEqual(len(item["confusion_matrix"]), 5)
            self.assertEqual(len(item["true_class_counts"]), 5)
            self.assertEqual(len(item["predicted_class_counts"]), 5)
            self.assertEqual(len(item["per_class_recall"]), 5)
            self.assertIn("raw_loss", item)
            self.assertIn("weighted_loss", item)
            for operation in item["operation_records"]:
                self.assertEqual(len(operation["ordinal_cut_probabilities"]), 4)
                self.assertIn("decision_margin", operation)
                self.assertIn("target_class", operation)
                self.assertIn("decoded_physical_value", operation)

    def test_magnitude_head_perturbation_cannot_change_structural_logits(self):
        model = self._models()[0].eval()
        teacher = self._teacher(model)
        states = teacher.decoder_output.decoded_states
        node_before = model.decoder.node_type_head(states).detach().clone()
        categorical_before = tuple(
            head(states).detach().clone()
            for head in model.decoder.remaining_categorical_heads
        )
        with torch.no_grad():
            for parameter in model.decoder.grid_magnitude_head.parameters():
                parameter.add_(torch.randn_like(parameter) * 9.0)
        torch.testing.assert_close(
            node_before,
            model.decoder.node_type_head(states),
            rtol=0.0,
            atol=0.0,
        )
        for expected, head in zip(
            categorical_before, model.decoder.remaining_categorical_heads
        ):
            torch.testing.assert_close(
                expected, head(states), rtol=0.0, atol=0.0
            )

    def test_engineering_only_synthetic_optimizer_step_is_finite(self):
        """One generated-tensor optimizer step; not scientific training."""

        from prototype.graph_encoder.losses import common_ge1_loss

        model = self._models()[0]
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        teacher = self._teacher(model)
        loss = common_ge1_loss(
            teacher.decoder_output,
            self.target,
            self.profiles,
            model.config,
            grid_magnitude_logits=teacher.grid_magnitude_logits,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.total.backward()
        optimizer.step()
        self.assertTrue(all(
            bool(parameter.isfinite().all().item())
            for parameter in model.parameters()
        ))


if __name__ == "__main__":
    unittest.main()
