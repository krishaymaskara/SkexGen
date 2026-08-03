"""Focused tensor, model, loss, conversion, and pilot-smoke tests."""

from __future__ import annotations

from dataclasses import replace
import inspect
from types import SimpleNamespace
import unittest

try:
    import torch
except ImportError:
    torch = None

from prototype.controlled_data.builders import build_history
from prototype.controlled_data.factors import PrimitiveFamily
from prototype.model_data.adapters import adapt_flat_mixed
from prototype.model_data.batching import collate_flat
from prototype.model_data.canonical import canonical_nodes_and_edges, reconstruction_target
from prototype.model_data.tests.fixtures import source
from prototype.representation.model import GeometryEncoding

from prototype.graph_baseline.graph_contract import (
    GRAPH_EDGE_CLASS_ORDER,
    GraphContractError,
    graph_from_reconstruction_target,
)


TORCH_REASON = "real PyTorch execution is deferred to the authoritative environment"


def _batch(*templates):
    examples = []
    for index, template in enumerate(templates):
        history = build_history(source(
            template,
            tuple(PrimitiveFamily)[index % len(tuple(PrimitiveFamily))],
            extents=(1.0,) * len(template),
        ), GeometryEncoding.CONTINUOUS)
        nodes, edges = canonical_nodes_and_edges(history)
        target = reconstruction_target(nodes, edges, history.structure.operation_sequence)
        examples.append(adapt_flat_mixed(SimpleNamespace(
            physical_family_id="graph-{}-{}".format(index, template),
            nodes=nodes, target=target,
        )))
    return collate_flat(tuple(examples))


@unittest.skipIf(torch is None, TORCH_REASON)
class GraphV1TensorTests(unittest.TestCase):
    def test_tensorization_mixed_lengths_masks_order_dtype_and_malformed(self):
        from prototype.graph_baseline.graph_tensors import graph_targets_from_reconstruction_batch
        batch = _batch("E", "R", "RR")
        target = batch.target.to_torch(torch)
        graph = graph_targets_from_reconstruction_batch(target)
        self.assertEqual(tuple(graph.edge_class_ids.shape), (3, 9, 9))
        self.assertEqual(graph.edge_class_ids.dtype, torch.long)
        self.assertEqual(graph.active_pair_mask.dtype, torch.bool)
        self.assertEqual(graph.node_counts.tolist(), [4, 5, 9])
        self.assertFalse(torch.diagonal(graph.active_pair_mask, dim1=1, dim2=2).any())
        self.assertFalse(graph.active_pair_mask[0, 4:].any())
        self.assertTrue(graph.edge_class_ids.is_contiguous())
        malformed = dict(target)
        malformed["edge_offsets"] = target["edge_offsets"][:-1]
        with self.assertRaises(ValueError):
            graph_targets_from_reconstruction_batch(malformed)

    def test_job_3338945_pair_orientation_masking_and_reconstruction(self):
        from prototype.flat_baseline.tests.test_constrained_v6 import V6GrammarTensorTests
        from prototype.graph_baseline.conversion import graph_prediction_from_evidence
        from prototype.graph_baseline.model import GraphV1Model
        from prototype.graph_baseline.config import GraphV1Config
        from prototype.graph_baseline.graph_tensors import (
            graph_type_allowed_mask,
            mask_graph_edge_logits,
        )
        from prototype.model_data.vocab import NODE_TYPES
        model = GraphV1Model(GraphV1Config())
        states = torch.randn(2, 5, model.config.model_dim)
        node_ids = torch.tensor([
            [5, 7, 4, 3, 0], [5, 7, 4, 2, 6]
        ], dtype=torch.long)
        active = torch.tensor([
            [True, True, True, True, False], [True] * 5
        ])
        memory = torch.randn(2, model.config.latent_tokens, model.config.model_dim)
        captured_features = []
        hook = model.graph_edge_decoder.register_forward_pre_hook(
            lambda unused_module, values: captured_features.append(values[0].detach().clone())
        )
        logits = model.decode_graph_edges(states, node_ids, memory, active)
        hook.remove()
        self.assertEqual(tuple(logits.shape), (2, 5, 5, len(GRAPH_EDGE_CLASS_ORDER)))
        width = model.config.model_dim
        torch.testing.assert_close(
            captured_features[0][0, 1, 0, :width], states[0, 1], rtol=0, atol=0
        )
        torch.testing.assert_close(
            captured_features[0][0, 1, 0, width:2 * width], states[0, 0],
            rtol=0, atol=0,
        )
        self.assertTrue(torch.isfinite(logits).all())
        self.assertFalse(torch.equal(logits[:, 1, 0], logits[:, 0, 1]))
        masked = mask_graph_edge_logits(logits, node_ids, active)
        self.assertFalse(masked.active_pair_mask[0, 4].any())
        self.assertEqual(masked.raw_class_ids.dtype, torch.long)
        self.assertEqual(masked.masked_class_ids.device, logits.device)
        self.assertTrue(masked.raw_class_ids.is_contiguous())
        self.assertTrue(masked.masked_class_ids.is_contiguous())
        self.assertTrue(masked.correction_mask.is_contiguous())
        self.assertNotIn("target", inspect.signature(model.decode_graph_edges).parameters)

        # Independently indexed [source, destination] evidence: profile (2) to
        # revolve (4) cannot be defined_in, while revolve to profile can use it.
        authoritative_ids = torch.tensor([[5, 7, 4, 2, 6]], dtype=torch.long)
        forced = torch.full((1, 5, 5, len(GRAPH_EDGE_CLASS_ORDER)), -20.0)
        forced[..., 0] = 0.0
        forced[0, 2, 4, 1] = 10.0
        forced[0, 4, 2, 1] = 10.0
        forced[0, 4, 2, 5] = 9.0
        oriented = mask_graph_edge_logits(
            forced, authoritative_ids, torch.ones(1, 5, dtype=torch.bool)
        )
        self.assertEqual(oriented.raw_class_ids[0, 2, 4].item(), 1)
        self.assertEqual(oriented.masked_class_ids[0, 2, 4].item(), 0)
        self.assertTrue(oriented.correction_mask[0, 2, 4].item())
        self.assertEqual(
            oriented.allowed_class_mask[0, 2, 4].tolist(),
            [True, False, False, False, False, False],
        )
        self.assertEqual(oriented.raw_class_ids[0, 4, 2].item(), 1)
        self.assertEqual(oriented.masked_class_ids[0, 4, 2].item(), 5)
        self.assertEqual(
            oriented.allowed_class_mask[0, 4, 2].tolist(),
            [True, False, False, False, False, True],
        )
        expected_classes = {
            ("profile", "sketch"): {0, 1},
            ("axis", "sketch"): {0, 1},
            ("sketch", "reference_plane"): {0, 3},
            ("extrude", "profile"): {0, 5},
            ("revolve", "profile"): {0, 5},
            ("revolve", "axis"): {0, 4},
            ("extrude", "extrude"): {0, 2},
            ("extrude", "revolve"): {0, 2},
            ("revolve", "extrude"): {0, 2},
            ("revolve", "revolve"): {0, 2},
        }
        semantic_types = (
            "axis", "extrude", "profile", "reference_plane", "revolve", "sketch"
        )
        for source_type in semantic_types:
            for destination_type in semantic_types:
                pair_types = torch.tensor([[
                    NODE_TYPES.id(source_type), NODE_TYPES.id(destination_type)
                ]], dtype=torch.long)
                allowed, unused_active = graph_type_allowed_mask(
                    pair_types, torch.ones(1, 2, dtype=torch.bool)
                )
                actual = {
                    class_id for class_id, value
                    in enumerate(allowed[0, 0, 1].tolist()) if value
                }
                self.assertEqual(
                    actual,
                    expected_classes.get((source_type, destination_type), {0}),
                )
        node_prediction = V6GrammarTensorTests()._authoritative_v6_prediction("R")
        prediction = graph_prediction_from_evidence(
            node_prediction,
            oriented.raw_class_ids[0].tolist(),
            oriented.masked_class_ids[0].tolist(),
            oriented.correction_mask[0].tolist(),
        )
        self.assertNotIn(
            (2, 4, 1),
            {(edge.source, edge.destination, edge.edge_type_id)
             for edge in prediction.graph.directed_typed_edges},
        )
        self.assertIn(
            (4, 2, 5),
            {(edge.source, edge.destination, edge.edge_type_id)
             for edge in prediction.graph.directed_typed_edges},
        )
        with self.assertRaisesRegex(GraphContractError, "graph_edge_type_mismatch"):
            graph_prediction_from_evidence(
                node_prediction,
                oriented.raw_class_ids[0].tolist(),
                [
                    [1 if (source, destination) == (2, 4) else 0
                     for destination in range(5)]
                    for source in range(5)
                ],
                [[False] * 5 for unused in range(5)],
            )

        # No flattening is used in production; unique legal class IDs survive
        # the exact [source, destination] matrix position at every valid length.
        for count in (4, 5, 7, 8, 9):
            pair_ids = torch.arange(count * count).reshape(count, count)
            for source in range(count):
                for destination in range(count):
                    self.assertEqual(
                        pair_ids.reshape(-1)[source * count + destination].item(),
                        pair_ids[source, destination].item(),
                    )
        torch.manual_seed(902)
        first = GraphV1Model(GraphV1Config())
        torch.manual_seed(902)
        second = GraphV1Model(GraphV1Config())
        self.assertTrue(all(
            torch.equal(first.state_dict()[name], second.state_dict()[name])
            for name in first.state_dict()
        ))

    def test_v6_node_path_capacity_optimizer_and_gradient_equivalence(self):
        from prototype.constrained_profile_decoder import profile_targets_for_loss
        from prototype.flat_baseline.constrained_v6 import ConstrainedProfileV6Model
        from prototype.flat_baseline.constrained_v6_config import ConstrainedProfileV6Config
        from prototype.graph_baseline.config import GraphV1Config
        from prototype.graph_baseline.conversion import graph_v1_teacher_forced_predictions
        from prototype.graph_baseline.losses import graph_v1_loss
        from prototype.graph_baseline.model import GraphV1Model, graph_decoder_parameter_count
        from prototype.graph_baseline.training import build_graph_optimizer
        from prototype.graph_baseline.training_config import GraphTrainingConfig
        batch = _batch("ER")
        inputs = batch.to_torch(torch)
        target = batch.target.to_torch(torch)
        profiles = profile_targets_for_loss(batch.target, inputs["geometry"])
        torch.manual_seed(41)
        flat = ConstrainedProfileV6Model(ConstrainedProfileV6Config())
        torch.manual_seed(41)
        graph = GraphV1Model(GraphV1Config())
        flat.eval(); graph.eval()
        flat_output = flat(target=target, profile_targets=profiles, **inputs)
        graph_output = graph(target=target, profile_targets=profiles, **inputs)
        for name in (
            "decoded_states", "node_type_logits", "profile_family_logits",
            "raw_profile_parameters", "remaining_geometry", "quantized_memory",
        ):
            torch.testing.assert_close(
                getattr(flat_output, name), getattr(graph_output, name), rtol=0, atol=0
            )
        graph_predictions = graph_v1_teacher_forced_predictions(
            graph_output,
            node_mask=target["node_mask"],
            node_count_source="job_3338945_regression",
        )
        for row, prediction in enumerate(graph_predictions):
            count = prediction.node_prediction.node_count
            self.assertEqual(
                graph_output.graph_constrained_node_type_ids[row, :count].tolist(),
                [node.node_type_id for node in prediction.node_prediction.raw_nodes],
            )
        self.assertEqual(
            sum(parameter.numel() for parameter in flat.parameters())
            - sum(parameter.numel() for parameter in graph.parameters()), 10
        )
        self.assertEqual(graph_decoder_parameter_count(graph), 3486)
        state_names = set(graph.state_dict())
        for forbidden in (
            "edge_source.weight", "edge_target.weight", "edge_presence_head.weight",
            "edge_type_head.weight", "operation_queries", "operation_keys.weight",
        ):
            self.assertNotIn(forbidden, state_names)
        optimizer = build_graph_optimizer(graph, GraphTrainingConfig())
        optimized = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
        self.assertTrue(all(id(parameter) in optimized for parameter in graph.graph_edge_decoder.parameters()))
        loss = graph_v1_loss(graph_output, target, profiles, graph.config)
        loss.total.backward()
        self.assertTrue(all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in graph.parameters() if parameter.requires_grad
        ))

    def test_graph_loss_manual_all_none_positive_mixed_and_padding(self):
        from prototype.graph_baseline.losses import _per_example_graph_cross_entropy
        logits = torch.tensor([[[[2.0, 0.0], [0.0, 2.0]], [[1.0, 0.0], [0.0, 1.0]]]])
        targets = torch.tensor([[[0, 1], [0, 0]]])
        mask = torch.tensor([[[False, True], [True, False]]])
        actual = _per_example_graph_cross_entropy(logits, targets, mask)
        manual = torch.nn.functional.cross_entropy(
            torch.stack((logits[0, 0, 1], logits[0, 1, 0])),
            torch.tensor([1, 0]),
        )
        torch.testing.assert_close(actual[0], manual, rtol=0, atol=0)
        zero = _per_example_graph_cross_entropy(logits, targets, torch.zeros_like(mask))
        self.assertEqual(zero.item(), 0.0)

    def test_strict_graph_conversion_canonical_and_malformed_edges(self):
        from prototype.flat_baseline.tests.test_constrained_v6 import V6GrammarTensorTests
        from prototype.graph_baseline.conversion import (
            graph_prediction_from_evidence, validate_and_convert_graph_prediction,
        )
        helper = V6GrammarTensorTests()
        for template in ("E", "R", "EE", "ER", "RE", "RR"):
            node_prediction = helper._authoritative_v6_prediction(template)
            history = build_history(source(template), GeometryEncoding.CONTINUOUS)
            nodes, edges = canonical_nodes_and_edges(history)
            target = reconstruction_target(nodes, edges, history.structure.operation_sequence)
            graph = graph_from_reconstruction_target(target)
            count = graph.node_count
            classes = [[0] * count for _ in range(count)]
            for edge in graph.directed_typed_edges:
                classes[edge.source][edge.destination] = edge.edge_type_id
            prediction = graph_prediction_from_evidence(
                node_prediction, classes, classes, [[False] * count for _ in range(count)]
            )
            result = validate_and_convert_graph_prediction(prediction)
            self.assertTrue(result.controlled_domain.valid)
            missing = [row[:] for row in classes]
            edge = graph.directed_typed_edges[0]
            missing[edge.source][edge.destination] = 0
            invalid = graph_prediction_from_evidence(
                node_prediction, missing, missing, [[False] * count for _ in range(count)]
            )
            self.assertFalse(validate_and_convert_graph_prediction(invalid).controlled_domain.valid)
            malformed_graph = replace(
                prediction.graph,
                directed_typed_edges=(graph.directed_typed_edges[0],) * 2,
            )
            with self.assertRaises(GraphContractError):
                validate_and_convert_graph_prediction(
                    replace(prediction, graph=malformed_graph)
                )

    def test_autonomous_is_target_free_and_target_mutations_cannot_change_output(self):
        from prototype.graph_baseline.autonomous import greedy_decode_graph_v1
        from prototype.graph_baseline.config import GraphV1Config
        from prototype.graph_baseline.model import GraphV1Model
        batch = _batch("R")
        inputs = batch.to_torch(torch)
        model = GraphV1Model(GraphV1Config()).eval()
        self.assertNotIn("target", inspect.signature(greedy_decode_graph_v1).parameters)
        first = greedy_decode_graph_v1(
            model, inputs, node_counts=torch.tensor([5]), node_count_source="test"
        )
        target_shadow = batch.target.to_torch(torch)
        target_shadow["edge_type_ids"].zero_()
        target_shadow["operation_sequence"].zero_()
        target_shadow["node_type_ids"].zero_()
        second = greedy_decode_graph_v1(
            model, inputs, node_counts=torch.tensor([5]), node_count_source="test"
        )
        self.assertEqual(first, second)
