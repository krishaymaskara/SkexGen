"""Focused tensor, model, loss, conversion, and pilot-smoke tests."""

from __future__ import annotations

from dataclasses import replace
import inspect
import json
from pathlib import Path
import tempfile
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
        self.assertIsNot(
            prediction.raw_graph_edge_predictions,
            prediction.masked_graph_edge_predictions,
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

    def test_job_3339188_complete_authoritative_node_path_and_capacity(self):
        from prototype.constrained_profile_decoder import profile_targets_for_loss
        from prototype.flat_baseline.constrained_v6 import ConstrainedProfileV6Model
        from prototype.flat_baseline.constrained_v6_autonomous import greedy_decode_v6
        from prototype.flat_baseline.constrained_v6_config import ConstrainedProfileV6Config
        from prototype.graph_baseline.autonomous import greedy_decode_graph_v1
        from prototype.graph_baseline.config import GraphV1Config
        from prototype.graph_baseline.conversion import graph_v1_teacher_forced_predictions
        from prototype.graph_baseline.losses import graph_v1_loss
        from prototype.graph_baseline.model import (
            GraphV1Model,
            graph_decoder_parameter_count,
            select_complete_graph_node_sequence,
        )
        from prototype.graph_baseline.training import build_graph_optimizer
        from prototype.graph_baseline.training_config import GraphTrainingConfig
        from prototype.node_grammar import (
            NodeGrammarError,
            validate_complete_node_sequence,
        )
        templates = ("E", "R", "EE", "ER", "RE", "RR")
        self.assertEqual(
            tuple(inspect.signature(select_complete_graph_node_sequence).parameters),
            ("logits", "node_mask"),
        )
        batch = _batch(*templates)
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
        self.assertEqual(
            tuple(graph_output.authoritative_graph_node_type_ids.shape),
            tuple(target["node_mask"].shape),
        )
        self.assertEqual(
            tuple(graph_output.graph_legal_node_type_mask.shape),
            tuple(target["node_mask"].shape) + (8,),
        )
        self.assertEqual(
            graph_output.authoritative_graph_node_type_ids.dtype, torch.long
        )
        self.assertEqual(
            graph_output.graph_node_type_correction_mask.dtype, torch.bool
        )
        self.assertEqual(
            graph_output.authoritative_graph_node_type_ids.device,
            graph_output.node_type_logits.device,
        )
        self.assertTrue(
            graph_output.authoritative_graph_node_type_ids.is_contiguous()
        )
        self.assertTrue(graph_output.graph_legal_node_type_mask.is_contiguous())
        self.assertTrue(torch.equal(
            graph_output.graph_node_type_correction_mask,
            target["node_mask"] & (
                graph_output.graph_raw_node_type_argmax_ids
                != graph_output.authoritative_graph_node_type_ids
            ),
        ))
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
        expected_by_count = {
            4: {(5, 7, 4, 3)},
            5: {(5, 7, 4, 2, 6)},
            7: {(5, 7, 4, 3, 7, 4, 3)},
            8: {
                (5, 7, 4, 3, 7, 4, 2, 6),
                (5, 7, 4, 2, 6, 7, 4, 3),
            },
            9: {(5, 7, 4, 2, 6, 7, 4, 2, 6)},
        }
        for row, prediction in enumerate(graph_predictions):
            count = prediction.node_prediction.node_count
            authoritative = tuple(
                graph_output.authoritative_graph_node_type_ids[
                    row, :count
                ].tolist()
            )
            self.assertEqual(
                list(authoritative),
                [node.node_type_id for node in prediction.node_prediction.raw_nodes],
            )
            self.assertIn(authoritative, expected_by_count[count])
            self.assertNotIn(0, authoritative)
            self.assertNotIn(1, authoritative)
            self.assertEqual(prediction.graph.node_type_ids, authoritative)
        counts = target["node_mask"].long().sum(dim=1)
        autonomous_graph = greedy_decode_graph_v1(
            graph, inputs, node_counts=counts,
            node_count_source="job_3339188_regression",
        )
        autonomous_v6 = greedy_decode_v6(
            graph, inputs, node_counts=counts,
            node_count_source="job_3339188_regression",
        )
        for graph_prediction, v6_prediction in zip(
            autonomous_graph, autonomous_v6
        ):
            expected = tuple(node.node_type_id for node in v6_prediction.raw_nodes)
            self.assertEqual(graph_prediction.graph.node_type_ids, expected)
            self.assertEqual(
                tuple(node.node_type_id for node in graph_prediction.node_prediction.raw_nodes),
                expected,
            )
        with self.assertRaises(NodeGrammarError) as caught:
            validate_complete_node_sequence(
                (5, 7, 4, 3, 6, 7, 4, 3), 8
            )
        self.assertEqual(caught.exception.code, "invalid_v5_generated_prefix")
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

    def test_production_training_step_smoke(self):
        from prototype.constrained_profile_decoder import profile_targets_for_loss
        from prototype.flat_baseline.checkpointing import save_checkpoint
        from prototype.graph_baseline.checkpoint import validate_graph_checkpoint
        from prototype.graph_baseline.config import GraphV1Config
        from prototype.graph_baseline.model import GraphV1Model
        from prototype.graph_baseline.pilot import graph_checkpoint_payload
        from prototype.graph_baseline.pilot_config import GraphPilotConfig
        from prototype.graph_baseline.provenance import (
            collect_graph_source_provenance,
            validate_graph_source_provenance,
        )
        from prototype.graph_baseline.tests.test_graph_provenance import (
            initialize_temporary_graph_repository,
        )
        from prototype.graph_baseline.training import (
            build_graph_optimizer,
            graph_training_step,
        )
        from prototype.graph_baseline.training_config import GraphTrainingConfig
        batch = _batch("E", "R")
        inputs = batch.to_torch(torch)
        target = batch.target.to_torch(torch)
        profiles = profile_targets_for_loss(batch.target, inputs["geometry"])
        model_config = GraphV1Config()
        training_config = GraphTrainingConfig()
        pilot_config = GraphPilotConfig(require_clean_source=False)
        source_temporary = tempfile.TemporaryDirectory()
        self.addCleanup(source_temporary.cleanup)
        repository = Path(source_temporary.name) / "source-repository"
        reviewed_commit = initialize_temporary_graph_repository(repository)
        provenance = collect_graph_source_provenance(repository)
        validate_graph_source_provenance(
            provenance, expected_commit=reviewed_commit
        )
        model = GraphV1Model(model_config)
        optimizer = build_graph_optimizer(model, training_config)
        parameters_before = {
            name: value.detach().clone()
            for name, value in model.named_parameters()
        }
        vq_before = model.vq.embedding.detach().clone()
        unused_output, losses = graph_training_step(
            model, optimizer, inputs, target, profiles,
            model_config, training_config,
        )
        del unused_output
        self.assertTrue(torch.isfinite(losses.total))
        self.assertTrue(all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in model.parameters() if parameter.requires_grad
        ))
        self.assertTrue(any(
            not torch.equal(parameters_before[name], parameter.detach())
            for name, parameter in model.named_parameters()
        ))
        self.assertFalse(torch.equal(vq_before, model.vq.embedding))
        self.assertTrue(optimizer.state)
        partition = SimpleNamespace(metadata=lambda: {
            "identity": "temporary_authorized_ordinary_partition"
        })
        data = SimpleNamespace(train=partition, validation=partition)
        with tempfile.TemporaryDirectory() as temporary:
            payload = graph_checkpoint_payload(
                model, optimizer, model_config, pilot_config, training_config,
                data, 1, 1, 2, 1,
                {"teacher_forced": {}, "autonomous": {}}, provenance,
                repository_root=repository,
            )
            unexpected_source = repository / "unexpected-source.txt"
            unexpected_source.write_text("must block checkpointing\n")
            with self.assertRaisesRegex(ValueError, "dirty_source_tree"):
                graph_checkpoint_payload(
                    model, optimizer, model_config, pilot_config,
                    training_config, data, 1, 1, 2, 1,
                    {"teacher_forced": {}, "autonomous": {}}, provenance,
                    repository_root=repository,
                )
            unexpected_source.unlink()
            path = Path(temporary) / "one-step.pt"
            save_checkpoint(path, payload, torch)
            loaded = torch.load(str(path), map_location="cpu")
            reloaded = GraphV1Model(model_config)
            reloaded_optimizer = build_graph_optimizer(
                reloaded, training_config
            )
            validate_graph_checkpoint(
                loaded, reloaded, model_config, pilot_config,
                training_config, torch,
                expected_source_provenance=provenance,
            )
            self.assertEqual(loaded["source_provenance"], provenance)
            reloaded.load_state_dict(loaded["model_state"], strict=True)
            reloaded_optimizer.load_state_dict(loaded["optimizer_state"])
            self.assertEqual(
                set(reloaded_optimizer.state_dict()["state"]),
                set(optimizer.state_dict()["state"]),
            )
        production_training_step_smoke = True
        scientific_training_run = False
        self.assertTrue(production_training_step_smoke)
        self.assertFalse(scientific_training_run)

    def test_tiny_two_epoch_control_flow_dry_run(self):
        from prototype.constrained_profile_decoder import profile_targets_for_loss
        from prototype.flat_baseline.checkpointing import save_checkpoint
        from prototype.flat_baseline.run_logging import JsonlLogger
        from prototype.graph_baseline.autonomous import greedy_decode_graph_v1
        from prototype.graph_baseline.checkpoint import validate_graph_checkpoint
        from prototype.graph_baseline.config import GraphV1Config
        from prototype.graph_baseline.conversion import (
            graph_v1_teacher_forced_predictions,
            validate_and_convert_graph_prediction,
        )
        from prototype.graph_baseline.losses import graph_v1_loss
        from prototype.graph_baseline.model import GraphV1Model
        from prototype.graph_baseline.pilot import (
            _require_acceptance,
            _require_finite_json,
            graph_checkpoint_payload,
        )
        from prototype.graph_baseline.pilot_config import GraphPilotConfig
        from prototype.graph_baseline.provenance import (
            collect_graph_source_provenance,
            validate_graph_source_provenance,
        )
        from prototype.graph_baseline.tests.test_graph_provenance import (
            initialize_temporary_graph_repository,
        )
        from prototype.graph_baseline.training import (
            build_graph_optimizer,
            graph_training_step,
        )
        from prototype.graph_baseline.training_config import GraphTrainingConfig
        batch = _batch("E", "R")
        inputs = batch.to_torch(torch)
        target = batch.target.to_torch(torch)
        profiles = profile_targets_for_loss(batch.target, inputs["geometry"])
        model_config = GraphV1Config()
        training_config = GraphTrainingConfig()
        pilot_config = GraphPilotConfig(require_clean_source=False)
        model = GraphV1Model(model_config)
        optimizer = build_graph_optimizer(model, training_config)
        partition = SimpleNamespace(metadata=lambda: {
            "identity": "tiny_authorized_ordinary_partition"
        })
        data = SimpleNamespace(train=partition, validation=partition)
        validations = {}
        with tempfile.TemporaryDirectory() as temporary:
            repository = Path(temporary) / "source-repository"
            reviewed_commit = initialize_temporary_graph_repository(repository)
            provenance = collect_graph_source_provenance(repository)
            validate_graph_source_provenance(
                provenance, expected_commit=reviewed_commit
            )
            logger = JsonlLogger(Path(temporary) / "metrics.jsonl")
            checkpoints = {}
            for epoch in (1, 2):
                graph_training_step(
                    model, optimizer, inputs, target, profiles,
                    model_config, training_config,
                )
                model.eval()
                with torch.no_grad():
                    output = model(
                        target=target, profile_targets=profiles, **inputs
                    )
                    losses = graph_v1_loss(
                        output, target, profiles, model_config
                    )
                    predictions = graph_v1_teacher_forced_predictions(
                        output, node_mask=target["node_mask"],
                        node_count_source="tiny_two_epoch_dry_run",
                    )
                    outcomes = tuple(
                        validate_and_convert_graph_prediction(item)
                        for item in predictions
                    )
                    autonomous_predictions = greedy_decode_graph_v1(
                        model,
                        inputs,
                        node_counts=target["node_mask"].long().sum(dim=1),
                        node_count_source="tiny_two_epoch_dry_run",
                    )
                    autonomous_outcomes = tuple(
                        validate_and_convert_graph_prediction(item)
                        for item in autonomous_predictions
                    )
                validations[epoch] = {
                    "teacher_forced": {
                        "total_loss": float(losses.total.item()),
                        "classified_count": len(outcomes),
                    },
                    "autonomous": {
                        "classified_count": len(autonomous_outcomes),
                    },
                }
                payload = graph_checkpoint_payload(
                    model, optimizer, model_config, pilot_config,
                    training_config, data, epoch, epoch, 2 * epoch, 2,
                    validations[epoch],
                    provenance, repository_root=repository,
                )
                path = Path(temporary) / "epoch-{:04d}.pt".format(epoch)
                save_checkpoint(path, payload, torch)
                checkpoints[epoch] = path
                logger.write({
                    "event": "epoch_validation", "epoch": epoch,
                    "validation": validations[epoch],
                })
            selected = min(
                validations,
                key=lambda item: validations[item]["teacher_forced"]["total_loss"],
            )
            for role, epoch in (("selected", selected), ("final", 2)):
                loaded = torch.load(str(checkpoints[epoch]), map_location="cpu")
                reloaded = GraphV1Model(model_config)
                reloaded_optimizer = build_graph_optimizer(
                    reloaded, training_config
                )
                validate_graph_checkpoint(
                    loaded, reloaded, model_config, pilot_config,
                    training_config, torch,
                    expected_source_provenance=provenance,
                )
                reloaded.load_state_dict(loaded["model_state"], strict=True)
                reloaded_optimizer.load_state_dict(loaded["optimizer_state"])
                self.assertEqual(loaded["epoch"], epoch, role)
            acceptance = _require_acceptance(
                {"tiny_two_epoch_control_flow_completed": True},
                {"systematic_partition_accessed": False,
                 "test_partition_accessed": False},
            )
            terminal = {
                "event": "terminal_success", "selected_epoch": selected,
                "completed_epochs": 2, "global_step": 2,
                "acceptance": acceptance,
                "systematic_partition_accessed": False,
                "test_partition_accessed": False,
            }
            _require_finite_json(terminal)
            logger.write(terminal)
            rows = [
                json.loads(line)
                for line in (Path(temporary) / "metrics.jsonl").read_text().splitlines()
            ]
            self.assertEqual([row["event"] for row in rows], [
                "epoch_validation", "epoch_validation", "terminal_success"
            ])
