"""Native-PyTorch forward, masking, VQ, loss, and gradient tests."""

from __future__ import annotations

from dataclasses import replace
import math
import tempfile
import unittest

try:
    import torch
except ImportError:
    torch = None

from prototype.model_data.adapters import adapt_flat_mixed
from prototype.model_data.batching import collate_flat
from prototype.model_data.loader import load_physical_examples
from prototype.model_data.tests.fixtures import source, write_physical_corpus

if torch is not None:
    from prototype.flat_baseline.config import FlatBaselineConfig
    from prototype.flat_baseline.losses import (
        dense_edge_targets,
        flat_mixed_vq_loss,
    )
    from prototype.flat_baseline.model import FlatMixedVQModel
    from prototype.model_data.geometry import GEOMETRY_WIDTH
    from prototype.model_data.vocab import EDGE_TYPES, NODE_TYPES


TORCH_REASON = "real PyTorch execution is deferred to the Adroit environment"


def _batch(templates=("E", "R", "ER")):
    temporary = tempfile.TemporaryDirectory()
    write_physical_corpus(
        temporary.name, tuple(source(template) for template in templates)
    )
    examples = load_physical_examples(temporary.name)
    batch = collate_flat(tuple(adapt_flat_mixed(item) for item in examples))
    inputs = batch.to_torch(torch)
    target = batch.target.to_torch(torch)
    return temporary, batch, inputs, target


@unittest.skipUnless(torch is not None, TORCH_REASON)
class ForwardTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(5)
        self.temporary, self.batch, self.inputs, self.target = _batch()
        self.addCleanup(self.temporary.cleanup)
        self.config = FlatBaselineConfig()
        self.model = FlatMixedVQModel(self.config)

    def _forward(self):
        return self.model(target=self.target, **self.inputs)

    def test_cpu_forward_shapes_dtypes_and_ranges(self):
        self.model.eval()
        with torch.no_grad():
            output = self._forward()
        batch_size, node_count = self.inputs["padding_mask"].shape
        self.assertEqual(
            output.node_type_logits.shape,
            (batch_size, node_count, len(NODE_TYPES.tokens)),
        )
        self.assertEqual(len(output.categorical_logits), 9)
        expected_widths = tuple(
            len(embedding.weight)
            for embedding in self.model.field_embeddings[1:]
        )
        self.assertEqual(
            tuple(item.shape for item in output.categorical_logits),
            tuple(
                (batch_size, node_count, width)
                for width in expected_widths
            ),
        )
        self.assertEqual(
            output.geometry.shape, (batch_size, node_count, GEOMETRY_WIDTH)
        )
        self.assertEqual(
            output.edge_presence_logits.shape,
            (batch_size, node_count, node_count),
        )
        self.assertEqual(
            output.edge_type_logits.shape,
            (batch_size, node_count, node_count, len(EDGE_TYPES.tokens)),
        )
        self.assertEqual(
            output.operation_pointer_logits.shape,
            (batch_size, self.config.max_operations, node_count),
        )
        self.assertEqual(
            output.quantized_memory.shape,
            (batch_size, self.config.latent_tokens, self.config.model_dim),
        )
        self.assertEqual(
            output.code_indices.shape,
            (batch_size, self.config.latent_tokens),
        )
        self.assertEqual(
            output.assignment_counts.shape, (self.config.codebook_size,)
        )
        for scalar in (
            output.vq_loss,
            output.active_code_count,
            output.codebook_utilization,
            output.codebook_perplexity,
        ):
            self.assertEqual(scalar.shape, ())
        self.assertEqual(output.node_type_logits.dtype, torch.float32)
        self.assertEqual(output.geometry.dtype, torch.float32)
        self.assertEqual(output.code_indices.dtype, torch.long)
        self.assertEqual(output.assignment_counts.dtype, torch.long)
        self.assertEqual(output.node_type_logits.device.type, "cpu")
        self.assertGreaterEqual(float(output.geometry.min()), -1.0)
        self.assertLessEqual(float(output.geometry.max()), 1.0)

    def test_true_means_real_padding_convention_is_respected(self):
        self.model.eval()
        with torch.no_grad():
            original = self._forward().quantized_memory
            changed = {key: value.clone() for key, value in self.inputs.items()}
            padded = ~changed["padding_mask"]
            changed["categorical_ids"][padded] = 1
            changed["geometry"][padded] = 0.875
            changed["geometry_mask"][padded] = True
            altered = self.model(target=self.target, **changed).quantized_memory
        torch.testing.assert_close(original, altered, rtol=0.0, atol=0.0)

    def test_vq_evaluation_is_deterministic_and_buffers_are_not_parameters(self):
        self.model.eval()
        parameter_ids = tuple(id(item) for item in self.model.parameters())
        with torch.no_grad():
            first = self._forward()
            second = self._forward()
        torch.testing.assert_close(
            first.quantized_memory, second.quantized_memory, rtol=0.0, atol=0.0
        )
        self.assertTrue(torch.equal(first.code_indices, second.code_indices))
        self.assertEqual(
            parameter_ids, tuple(id(item) for item in self.model.parameters())
        )
        parameter_names = {name for name, _ in self.model.named_parameters()}
        buffer_names = {name for name, _ in self.model.named_buffers()}
        for name in ("vq.embedding", "vq.ema_cluster_size", "vq.ema_weight"):
            self.assertIn(name, buffer_names)
            self.assertNotIn(name, parameter_names)

    def test_operation_pointer_masks_padded_node_positions(self):
        self.model.eval()
        with torch.no_grad():
            output = self._forward()
        invalid = ~self.target["node_mask"].unsqueeze(1).expand_as(
            output.operation_pointer_logits
        )
        if invalid.any():
            expected = torch.finfo(output.operation_pointer_logits.dtype).min
            self.assertTrue(
                torch.all(output.operation_pointer_logits[invalid] == expected)
            )


@unittest.skipUnless(torch is not None, TORCH_REASON)
class LossTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        self.temporary, self.batch, self.inputs, self.target = _batch()
        self.addCleanup(self.temporary.cleanup)
        self.config = FlatBaselineConfig()
        self.model = FlatMixedVQModel(self.config)

    def test_geometry_loss_ignores_inapplicable_and_padded_channels(self):
        self.model.eval()
        output = self.model(target=self.target, **self.inputs)
        baseline = flat_mixed_vq_loss(output, self.target, self.config)
        ignored = ~(
            self.target["geometry_mask"]
            & self.target["node_mask"].unsqueeze(-1)
        )
        changed_geometry = output.geometry.clone()
        changed_geometry[ignored] = 1000.0
        changed = replace(output, geometry=changed_geometry)
        altered = flat_mixed_vq_loss(changed, self.target, self.config)
        torch.testing.assert_close(
            baseline.geometry, altered.geometry, rtol=0.0, atol=0.0
        )

    def test_global_edge_offsets_reconstruct_each_graph_exactly(self):
        node_count = self.target["node_mask"].size(1)
        presence, edge_types, valid = dense_edge_targets(
            self.target, node_count
        )
        self.assertEqual(
            int(presence.sum().item()), self.target["edge_type_ids"].numel()
        )
        self.assertTrue(torch.all(valid[presence]))
        node_offset = 0
        for batch_index in range(len(self.batch.family_ids)):
            start = int(self.target["edge_offsets"][batch_index])
            stop = int(self.target["edge_offsets"][batch_index + 1])
            local_sources = (
                self.target["edge_index"][0, start:stop] - node_offset
            )
            local_targets = (
                self.target["edge_index"][1, start:stop] - node_offset
            )
            expected_types = self.target["edge_type_ids"][start:stop]
            self.assertTrue(
                torch.all(
                    presence[batch_index, local_sources, local_targets]
                )
            )
            torch.testing.assert_close(
                edge_types[batch_index, local_sources, local_targets],
                expected_types,
            )
            node_offset += int(
                self.target["node_mask"][batch_index].sum().item()
            )

    def test_operation_loss_ignores_invalid_slots(self):
        self.model.eval()
        output = self.model(target=self.target, **self.inputs)
        baseline = flat_mixed_vq_loss(output, self.target, self.config)
        changed = dict(self.target)
        changed["operation_sequence"] = self.target["operation_sequence"].clone()
        changed["operation_sequence"][~self.target["operation_mask"]] = 0
        altered = flat_mixed_vq_loss(output, changed, self.config)
        torch.testing.assert_close(
            baseline.operation_pointer,
            altered.operation_pointer,
            rtol=0.0,
            atol=0.0,
        )

    def test_backward_has_finite_gradients(self):
        self.model.train()
        output = self.model(target=self.target, **self.inputs)
        losses = flat_mixed_vq_loss(output, self.target, self.config)
        losses.total.backward()
        gradients = [
            parameter.grad
            for parameter in self.model.parameters()
            if parameter.grad is not None
        ]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(item).all() for item in gradients))
        self.assertTrue(math.isfinite(float(losses.total.detach())))

    def test_tiny_one_example_optimization_decreases_total_loss(self):
        self.temporary.cleanup()
        self.temporary, self.batch, self.inputs, self.target = _batch(("E",))
        self.addCleanup(self.temporary.cleanup)
        config = FlatBaselineConfig(
            model_dim=16,
            num_heads=2,
            feedforward_dim=24,
            codebook_dim=8,
            codebook_size=8,
            edge_pair_dim=12,
            latent_tokens=1,
        )
        torch.manual_seed(23)
        model = FlatMixedVQModel(config)
        model.train()
        model.vq.eval()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        initial = None
        final = None
        for _ in range(20):
            optimizer.zero_grad()
            output = model(target=self.target, **self.inputs)
            loss = flat_mixed_vq_loss(output, self.target, config).total
            if initial is None:
                initial = float(loss.detach())
            loss.backward()
            optimizer.step()
            final = float(loss.detach())
        self.assertLess(final, initial)


if __name__ == "__main__":
    unittest.main()
