"""C4 continuous flat and position-free relational encoder tests."""

from __future__ import annotations

import ast
from collections import deque
import inspect
from pathlib import Path
import unittest
from unittest.mock import patch

from prototype.graph_encoder.batching import build_paired_batch, permute_graph
from prototype.graph_encoder.tests.fixtures import procedural_fixture
from prototype.model_data.vocab import (
    BOOLEAN_MODES,
    DIRECTIONS,
    EDGE_TYPES,
    LOOP_ROLES,
    NODE_TYPES,
    OPERATION_TYPES,
    PRIMITIVE_TYPES,
    REFERENCE_PLANES,
)


try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from prototype.flat_baseline.config import FlatBaselineConfig
    from prototype.flat_baseline.model import FlatMixedVQModel
    from prototype.flat_baseline.vq import EMAVectorQuantizer
    from prototype.graph_encoder.encoders import (
        CAPACITY_TOLERANCE_PERCENT,
        FLAT_CAPACITY_MATCHED_FEEDFORWARD_WIDTH,
        GRAPH_FEEDFORWARD_WIDTH,
        EncodedMemory,
        FlatProgramEncoder,
        TypedGraphProgramEncoder,
        capacity_difference_percent,
        default_encoder_config,
        encoder_parameter_report,
    )
    from prototype.graph_encoder.relational import (
        DIRECTED_RELATION_CHANNELS,
        SEMANTIC_EDGE_TYPE_IDS,
        BasisRelationalLayer,
    )


TORCH_REASON = "C4 real-tensor tests require the authoritative PyTorch environment"
TEMPLATES = ("E", "R", "EE", "ER", "RE", "RR")
FLOAT32_ATOL = 1e-6
FLOAT32_RTOL = 1e-5
FLOAT64_ATOL = 1e-12
FLOAT64_RTOL = 1e-12
SENSITIVITY_MINIMUM = 1e-6


def _diameter(graph):
    node_count = len(graph.node_type_ids)
    neighbors = [set() for unused in range(node_count)]
    for source, destination in zip(graph.edge_index[0], graph.edge_index[1]):
        neighbors[source].add(destination)
        neighbors[destination].add(source)
    result = 0
    for start in range(node_count):
        distances = [-1] * node_count
        distances[start] = 0
        queue = deque((start,))
        while queue:
            current = queue.popleft()
            for neighbor in neighbors[current]:
                if distances[neighbor] < 0:
                    distances[neighbor] = distances[current] + 1
                    queue.append(neighbor)
        if min(distances) < 0:
            raise AssertionError("procedural fixture must be connected")
        result = max(result, max(distances))
    return result


class StaticContractTests(unittest.TestCase):
    def test_frozen_parameter_arithmetic_matches_documented_totals(self):
        vocabulary_total = sum(
            (
                len(NODE_TYPES.tokens),
                len(OPERATION_TYPES.tokens),
                len(BOOLEAN_MODES.tokens),
                len(DIRECTIONS.tokens),
                len(REFERENCE_PLANES.tokens),
                4 * len(PRIMITIVE_TYPES.tokens),
                len(LOOP_ROLES.tokens),
            )
        )
        model_dim = 32
        node_initialization = (
            vocabulary_total * model_dim
            + 2 * (39 * model_dim + model_dim)
            + 2 * model_dim
        )
        attention = 3 * model_dim * model_dim + 3 * model_dim
        attention += model_dim * model_dim + model_dim
        bottleneck = model_dim * 16 + 16 + 16 * model_dim + model_dim
        relation_layer = (
            2 * model_dim * model_dim
            + 10 * 2
            + model_dim * model_dim
            + model_dim
            + 2 * model_dim
        )
        flat_feedforward = (
            model_dim * 192 + 192 + 192 * model_dim + model_dim
        )
        flat_transformer = attention + flat_feedforward + 4 * model_dim
        flat_transformer += 2 * model_dim
        flat_total = (
            node_initialization
            + 16 * model_dim
            + 2 * model_dim
            + flat_transformer
            + bottleneck
        )
        graph_feedforward = (
            model_dim * 64 + 64 + 64 * model_dim + model_dim
        )
        graph_pooling = (
            2 * model_dim
            + attention
            + 2 * model_dim
            + graph_feedforward
            + 2 * model_dim
        )
        graph_total = (
            node_initialization
            + 3 * relation_layer
            + graph_pooling
            + bottleneck
        )
        self.assertEqual(node_initialization, 4224)
        self.assertEqual(relation_layer, 3188)
        self.assertEqual(graph_pooling, 8608)
        self.assertEqual(bottleneck, 1072)
        self.assertEqual(flat_total, 22800)
        self.assertEqual(graph_total, 23468)
        self.assertLessEqual(
            abs(graph_total - flat_total) * 100.0 / flat_total,
            5.0,
        )

    def test_three_layers_cover_every_procedural_template_diameter(self):
        diameters = {
            template: _diameter(procedural_fixture(template).graph)
            for template in TEMPLATES
        }
        self.assertEqual(
            diameters,
            {"E": 3, "R": 3, "EE": 3, "ER": 3, "RE": 3, "RR": 3},
        )
        self.assertEqual(max(diameters.values()), 3)

    def test_production_sources_have_no_loader_manifest_or_corpus_import(self):
        package = Path(__file__).parents[1]
        paths = (package / "encoders.py", package / "relational.py")
        forbidden_import_fragments = (
            "model_data.loader",
            "graph_encoder.partitions",
            "load_train",
            "load_development",
            "manifest",
            "corpus",
        )
        for path in paths:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source)
            imports = "\n".join(
                ast.dump(node)
                for node in ast.walk(tree)
                if isinstance(node, (ast.Import, ast.ImportFrom))
            )
            for fragment in forbidden_import_fragments:
                self.assertNotIn(fragment, imports)

    def test_graph_source_has_no_forbidden_semantic_input_names(self):
        path = Path(__file__).parents[1] / "encoders.py"
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        graph_class = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == "TypedGraphProgramEncoder"
        )
        argument_names = {
            argument.arg
            for node in graph_class.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            for argument in node.args.args
        }
        forbidden = {
            "target",
            "reconstruction_target",
            "operation_sequence",
            "chronological_positions",
            "family_ids",
            "template_labels",
            "split",
            "partition",
            "edge_offsets",
            "node_graph_ids",
            "local_node_indices",
            "padding_positions",
        }
        self.assertTrue(argument_names.isdisjoint(forbidden))
        self.assertNotIn("operation_sequence", source)

    def test_relational_source_is_sparse_and_has_no_active_channel_normalization(self):
        path = Path(__file__).parents[1] / "relational.py"
        source = path.read_text(encoding="utf-8")
        self.assertIn("index_add_", source)
        self.assertIn("clamp_min(1.0)", source)
        self.assertNotIn("one_hot", source)
        self.assertNotIn("adjacency", source)
        self.assertNotIn("active_channel_count", source)


@unittest.skipUnless(torch is not None, TORCH_REASON)
class C4TensorTestCase(unittest.TestCase):
    def _paired(self, templates=TEMPLATES):
        examples = tuple(procedural_fixture(name).physical for name in templates)
        return build_paired_batch(examples)

    def _flat_tensors(self, paired):
        return paired.flat_input.to_torch(torch)

    def _graph_tensors(self, paired, dtype=None):
        if dtype is None:
            dtype = torch.float32
        result = paired.graph_input.to_torch(torch)
        result["geometry"] = result["geometry"].to(dtype=dtype)
        result["graph_offsets"] = paired.graph_bookkeeping.to_torch(torch)[
            "graph_offsets"
        ]
        return result

    def _one_graph_tensors(self, graph, dtype=None):
        if dtype is None:
            dtype = torch.float32
        return {
            "node_type_ids": torch.tensor(graph.node_type_ids, dtype=torch.long),
            "categorical_attributes": torch.tensor(
                graph.categorical_attributes, dtype=torch.long
            ),
            "geometry": torch.tensor(graph.geometry, dtype=dtype),
            "geometry_mask": torch.tensor(graph.geometry_mask, dtype=torch.bool),
            "edge_index": torch.tensor(graph.edge_index, dtype=torch.long),
            "edge_type_ids": torch.tensor(graph.edge_type_ids, dtype=torch.long),
            "graph_offsets": torch.tensor(
                (0, len(graph.node_type_ids)), dtype=torch.long
            ),
        }


class EncoderShapeAndGradientTests(C4TensorTestCase):
    def test_common_output_shapes_dtypes_contiguity_and_finiteness(self):
        paired = self._paired()
        flat = FlatProgramEncoder().eval()
        graph = TypedGraphProgramEncoder().eval()
        with torch.no_grad():
            flat_result = flat(**self._flat_tensors(paired))
            graph_result = graph(**self._graph_tensors(paired))
        for result in (flat_result, graph_result):
            self.assertIsInstance(result, EncodedMemory)
            self.assertEqual(result.memory.shape, (6, 2, 32))
            self.assertEqual(result.prequant.shape, (6, 2, 16))
            self.assertEqual(result.memory.dtype, torch.float32)
            self.assertEqual(result.prequant.dtype, torch.float32)
            self.assertTrue(result.memory.is_contiguous())
            self.assertTrue(torch.isfinite(result.memory).all())
            self.assertTrue(torch.isfinite(result.prequant).all())

    def test_finite_gradients_reach_every_trainable_component(self):
        paired = self._paired()
        cases = (
            (FlatProgramEncoder(), self._flat_tensors(paired)),
            (TypedGraphProgramEncoder(), self._graph_tensors(paired)),
        )
        for model, inputs in cases:
            with self.subTest(model=type(model).__name__):
                model.train()
                result = model(**inputs)
                weights = torch.linspace(
                    0.5,
                    1.5,
                    result.memory.numel(),
                    dtype=result.memory.dtype,
                ).reshape_as(result.memory)
                loss = (result.memory * weights).sum() + 0.1 * result.prequant.square().sum()
                loss.backward()
                gradients = tuple(
                    parameter.grad
                    for parameter in model.parameters()
                    if parameter.requires_grad
                )
                self.assertTrue(gradients)
                self.assertTrue(all(item is not None for item in gradients))
                self.assertTrue(all(torch.isfinite(item).all() for item in gradients))

    def test_same_seed_is_deterministic_and_different_seed_changes_state(self):
        for model_type in (FlatProgramEncoder, TypedGraphProgramEncoder):
            with self.subTest(model=model_type.__name__):
                torch.manual_seed(71)
                first = model_type()
                torch.manual_seed(71)
                second = model_type()
                torch.manual_seed(72)
                third = model_type()
                first_state = first.state_dict()
                second_state = second.state_dict()
                third_state = third.state_dict()
                self.assertEqual(tuple(first_state), tuple(second_state))
                self.assertTrue(
                    all(torch.equal(first_state[name], second_state[name]) for name in first_state)
                )
                self.assertTrue(
                    any(not torch.equal(first_state[name], third_state[name]) for name in first_state)
                )


class FlatContinuousParityTests(C4TensorTestCase):
    def _baseline(self):
        config = default_encoder_config("flat")
        baseline_config = FlatBaselineConfig(
            model_dim=config.model_dim,
            num_heads=config.attention_heads,
            feedforward_dim=config.encoder_feedforward_width,
            encoder_layers=config.flat_encoder_layers,
            decoder_layers=config.decoder_layers,
            dropout=config.dropout,
            max_nodes=config.max_nodes,
            max_operations=config.max_operations,
            latent_tokens=config.latent_tokens,
            codebook_dim=config.bottleneck_dim,
        )
        return FlatMixedVQModel(baseline_config)

    def _continuous_reference(self, baseline, inputs):
        categorical_ids = inputs["categorical_ids"]
        geometry = inputs["geometry"]
        geometry_mask = inputs["geometry_mask"]
        padding_mask = inputs["padding_mask"]
        batch_size, node_count = categorical_ids.shape[:2]
        positions = torch.arange(node_count).unsqueeze(0)
        content = baseline._record_content(
            categorical_ids, geometry, geometry_mask
        )
        nodes = baseline.input_norm(
            content + baseline.node_position_embedding(positions)
        )
        queries = baseline.latent_queries.unsqueeze(0).expand(batch_size, -1, -1)
        encoder_input = torch.cat((queries, nodes), dim=1)
        query_mask = torch.zeros(
            batch_size,
            baseline.config.latent_tokens,
            dtype=torch.bool,
        )
        encoded = baseline.encoder(
            encoder_input,
            src_key_padding_mask=torch.cat((query_mask, ~padding_mask), dim=1),
        )
        prequant = baseline.to_codebook(
            encoded[:, : baseline.config.latent_tokens]
        )
        return prequant, baseline.from_codebook(prequant)

    def test_wrapper_is_exactly_equal_to_inherited_continuous_path(self):
        torch.manual_seed(83)
        baseline = self._baseline().eval()
        wrapper = FlatProgramEncoder.from_inherited(baseline).eval()
        inputs = self._flat_tensors(self._paired(("E", "R", "EE")))
        with torch.no_grad():
            expected_prequant, expected_memory = self._continuous_reference(
                baseline, inputs
            )
            observed = wrapper(**inputs)
        torch.testing.assert_close(
            observed.prequant, expected_prequant, rtol=0.0, atol=0.0
        )
        torch.testing.assert_close(
            observed.memory, expected_memory, rtol=0.0, atol=0.0
        )

    def test_neither_arm_invokes_nearest_code_assignment(self):
        paired = self._paired(("E", "R"))
        with patch.object(
            EMAVectorQuantizer,
            "forward",
            side_effect=AssertionError("nearest-code assignment is forbidden"),
        ) as quantizer:
            FlatProgramEncoder()(**self._flat_tensors(paired))
            TypedGraphProgramEncoder()(**self._graph_tensors(paired))
        quantizer.assert_not_called()

    def test_diagnostics_do_not_update_vq_or_ema_state(self):
        torch.manual_seed(89)
        baseline = self._baseline()
        before = {
            name: value.detach().clone()
            for name, value in baseline.vq.named_buffers()
        }
        wrapper = FlatProgramEncoder.from_inherited(baseline)
        result = wrapper(**self._flat_tensors(self._paired(("E", "R"))))
        self.assertTrue(torch.isfinite(result.prequant).all())
        after = dict(baseline.vq.named_buffers())
        self.assertEqual(set(before), set(after))
        self.assertTrue(
            all(torch.equal(before[name], after[name]) for name in before)
        )
        self.assertFalse(hasattr(wrapper, "vq"))


class RelationalAggregationTests(C4TensorTestCase):
    def test_exact_ten_channels_two_bases_and_per_channel_mean_sum(self):
        layer = BasisRelationalLayer(model_dim=2, basis_count=2, dropout=0.0)
        with torch.no_grad():
            layer.relation_bases.zero_()
            layer.relation_bases[0].copy_(torch.eye(2))
            layer.relation_mixing.zero_()
            layer.relation_mixing[:, 0] = 1.0
        self.assertEqual(DIRECTED_RELATION_CHANNELS, 10)
        self.assertEqual(layer.relation_transforms().shape, (10, 2, 2))
        nodes = torch.tensor(((1.0, 2.0), (3.0, 4.0), (5.0, 6.0)))
        edge_index = torch.tensor(((0, 1), (2, 2)), dtype=torch.long)
        edge_types = torch.tensor(
            (SEMANTIC_EDGE_TYPE_IDS[0], SEMANTIC_EDGE_TYPE_IDS[1]),
            dtype=torch.long,
        )
        aggregated = layer.aggregate_messages(nodes, edge_index, edge_types)
        torch.testing.assert_close(
            aggregated[2], nodes[0] + nodes[1], rtol=0.0, atol=0.0
        )
        # A second edge in the same channel is averaged, while the other
        # active channel is still summed rather than averaged away.
        same_channel_types = torch.tensor(
            (SEMANTIC_EDGE_TYPE_IDS[0], SEMANTIC_EDGE_TYPE_IDS[0]),
            dtype=torch.long,
        )
        same_channel = layer.aggregate_messages(
            nodes, edge_index, same_channel_types
        )
        torch.testing.assert_close(
            same_channel[2], (nodes[0] + nodes[1]) / 2.0, rtol=0.0, atol=0.0
        )

    def test_zero_edges_isolated_nodes_and_empty_channels_stay_finite(self):
        model = TypedGraphProgramEncoder().eval()
        inputs = self._graph_tensors(self._paired(("E",)))
        inputs["edge_index"] = torch.empty((2, 0), dtype=torch.long)
        inputs["edge_type_ids"] = torch.empty((0,), dtype=torch.long)
        with torch.no_grad():
            nodes = model.encode_nodes(
                inputs["node_type_ids"],
                inputs["categorical_attributes"],
                inputs["geometry"],
                inputs["geometry_mask"],
                inputs["edge_index"],
                inputs["edge_type_ids"],
            )
            result = model(**inputs)
        self.assertTrue(torch.isfinite(nodes).all())
        self.assertTrue(torch.isfinite(result.memory).all())


class GraphIsolationAndSensitivityTests(C4TensorTestCase):
    def test_graph_b_is_identical_alone_batched_and_after_graph_a_changes(self):
        fixtures = tuple(procedural_fixture(name) for name in ("E", "RR"))
        batch = build_paired_batch(tuple(item.physical for item in fixtures))
        graph_b_id = fixtures[1].physical.physical_family_id
        graph_b_index = batch.family_ids.index(graph_b_id)
        alone = build_paired_batch((fixtures[1].physical,))
        model = TypedGraphProgramEncoder().eval()
        batch_inputs = self._graph_tensors(batch)
        changed_inputs = {
            name: value.clone() for name, value in batch_inputs.items()
        }
        other_index = 1 - graph_b_index
        offsets = batch.graph_bookkeeping.graph_offsets
        start, stop = offsets[other_index : other_index + 2]
        changed_inputs["geometry"][start:stop] += 0.75
        with torch.no_grad():
            alone_memory = model(**self._graph_tensors(alone)).memory[0]
            batch_memory = model(**batch_inputs).memory[graph_b_index]
            changed_memory = model(**changed_inputs).memory[graph_b_index]
        torch.testing.assert_close(
            batch_memory, alone_memory, rtol=FLOAT32_RTOL, atol=FLOAT32_ATOL
        )
        torch.testing.assert_close(
            changed_memory, alone_memory, rtol=FLOAT32_RTOL, atol=FLOAT32_ATOL
        )

    def test_cross_graph_edge_is_rejected_before_aggregation_or_attention(self):
        inputs = self._graph_tensors(self._paired(("E", "R")))
        offsets = inputs["graph_offsets"]
        inputs["edge_index"] = inputs["edge_index"].clone()
        inputs["edge_index"][1, 0] = offsets[1]
        with self.assertRaisesRegex(ValueError, "crosses|boundary"):
            TypedGraphProgramEncoder()(**inputs)

    def _sensitivity_inputs(self):
        inputs = self._graph_tensors(self._paired(("R",)))
        inputs["geometry"] = inputs["geometry"].clone()
        inputs["geometry"][0, 0] = -0.75
        inputs["geometry"][1, 0] = 0.625
        inputs["geometry_mask"] = inputs["geometry_mask"].clone()
        inputs["geometry_mask"][:2, 0] = True
        inputs["edge_index"] = torch.tensor(((0,), (1,)), dtype=torch.long)
        inputs["edge_type_ids"] = torch.tensor(
            (EDGE_TYPES.id("uses_profile"),), dtype=torch.long
        )
        return inputs

    def test_direction_sensitivity_uses_asymmetric_content(self):
        torch.manual_seed(101)
        model = TypedGraphProgramEncoder().eval()
        original = self._sensitivity_inputs()
        reversed_edge = {
            name: value.clone() for name, value in original.items()
        }
        reversed_edge["edge_index"] = original["edge_index"].flip(0)
        with torch.no_grad():
            first = model(**original).memory
            second = model(**reversed_edge).memory
        self.assertGreater(
            float((first - second).abs().max()), SENSITIVITY_MINIMUM
        )

    def test_edge_type_sensitivity_holds_endpoints_and_direction_fixed(self):
        torch.manual_seed(103)
        model = TypedGraphProgramEncoder().eval()
        uses_profile = self._sensitivity_inputs()
        uses_axis = {
            name: value.clone() for name, value in uses_profile.items()
        }
        uses_axis["edge_type_ids"] = torch.tensor(
            (EDGE_TYPES.id("uses_axis"),), dtype=torch.long
        )
        self.assertTrue(torch.equal(uses_profile["edge_index"], uses_axis["edge_index"]))
        with torch.no_grad():
            first = model(**uses_profile).memory
            second = model(**uses_axis).memory
        self.assertGreater(
            float((first - second).abs().max()), SENSITIVITY_MINIMUM
        )

    def test_direction_and_type_sensitivity_are_separate_perturbations(self):
        inputs = self._sensitivity_inputs()
        reversed_edge = inputs["edge_index"].flip(0)
        changed_type = torch.tensor(
            (EDGE_TYPES.id("uses_axis"),), dtype=torch.long
        )
        self.assertFalse(torch.equal(inputs["edge_index"], reversed_edge))
        self.assertTrue(torch.equal(inputs["edge_type_ids"], inputs["edge_type_ids"]))
        self.assertTrue(torch.equal(inputs["edge_index"], inputs["edge_index"]))
        self.assertFalse(torch.equal(inputs["edge_type_ids"], changed_type))


class PermutationTests(C4TensorTestCase):
    def _assert_permutation_properties(self, dtype, atol, rtol):
        fixture = procedural_fixture("RR")
        permutation = tuple(reversed(range(len(fixture.graph.node_type_ids))))
        permuted = permute_graph(fixture.graph, permutation)
        original_inputs = self._one_graph_tensors(fixture.graph, dtype=dtype)
        permuted_inputs = self._one_graph_tensors(permuted, dtype=dtype)
        torch.manual_seed(107)
        model = TypedGraphProgramEncoder().to(dtype=dtype)
        model.eval()
        self.assertFalse(model.training)
        with torch.no_grad():
            original_nodes = model.encode_nodes(
                original_inputs["node_type_ids"],
                original_inputs["categorical_attributes"],
                original_inputs["geometry"],
                original_inputs["geometry_mask"],
                original_inputs["edge_index"],
                original_inputs["edge_type_ids"],
            )
            permuted_nodes = model.encode_nodes(
                permuted_inputs["node_type_ids"],
                permuted_inputs["categorical_attributes"],
                permuted_inputs["geometry"],
                permuted_inputs["geometry_mask"],
                permuted_inputs["edge_index"],
                permuted_inputs["edge_type_ids"],
            )
            original_memory = model(**original_inputs).memory
            permuted_memory = model(**permuted_inputs).memory
        inverse = [None] * len(permutation)
        for new, old in enumerate(permutation):
            inverse[old] = new
        torch.testing.assert_close(
            permuted_nodes[inverse], original_nodes, rtol=rtol, atol=atol
        )
        torch.testing.assert_close(
            permuted_memory, original_memory, rtol=rtol, atol=atol
        )

    def test_node_equivariance_and_memory_invariance_float32(self):
        self._assert_permutation_properties(
            torch.float32, FLOAT32_ATOL, FLOAT32_RTOL
        )

    def test_node_equivariance_and_memory_invariance_float64(self):
        self._assert_permutation_properties(
            torch.float64, FLOAT64_ATOL, FLOAT64_RTOL
        )


class InterfaceAndCapacityTests(C4TensorTestCase):
    def test_signatures_expose_no_target_or_operation_sequence(self):
        for function in (
            FlatProgramEncoder.forward,
            TypedGraphProgramEncoder.forward,
            TypedGraphProgramEncoder.initialize_nodes,
            TypedGraphProgramEncoder.encode_nodes,
        ):
            parameters = set(inspect.signature(function).parameters)
            self.assertNotIn("target", parameters)
            self.assertNotIn("operation_sequence", parameters)

    def test_graph_registers_no_position_embedding_parameter(self):
        names = tuple(name for name, unused in TypedGraphProgramEncoder().named_parameters())
        self.assertFalse(any("position" in name for name in names))
        self.assertFalse(any("offset" in name for name in names))
        flat_names = tuple(name for name, unused in FlatProgramEncoder().named_parameters())
        self.assertIn("node_position_embedding.weight", flat_names)

    def test_bookkeeping_cannot_enter_semantic_node_initialization(self):
        parameters = tuple(
            inspect.signature(
                TypedGraphProgramEncoder.initialize_nodes
            ).parameters
        )
        self.assertEqual(
            parameters,
            (
                "self",
                "node_type_ids",
                "categorical_attributes",
                "geometry",
                "geometry_mask",
            ),
        )
        forward = set(inspect.signature(TypedGraphProgramEncoder.forward).parameters)
        self.assertEqual(
            forward,
            {
                "self",
                "node_type_ids",
                "categorical_attributes",
                "geometry",
                "geometry_mask",
                "edge_index",
                "edge_type_ids",
                "graph_offsets",
            },
        )

    def test_flat_and_graph_arms_have_disjoint_parameter_objects(self):
        flat = FlatProgramEncoder()
        graph = TypedGraphProgramEncoder()
        flat_ids = {id(parameter) for parameter in flat.parameters()}
        graph_ids = {id(parameter) for parameter in graph.parameters()}
        self.assertTrue(flat_ids)
        self.assertTrue(graph_ids)
        self.assertTrue(flat_ids.isdisjoint(graph_ids))

    def test_exact_component_parameter_counts_and_capacity_gate(self):
        flat = FlatProgramEncoder()
        graph = TypedGraphProgramEncoder()
        self.assertEqual(
            encoder_parameter_report(flat),
            {
                "node_content_initialization": 4224,
                "chronological_position_embedding": 512,
                "latent_queries": 64,
                "transformer_encoder": 16928,
                "bottleneck_projection": 1072,
                "complete_flat_encoder": 22800,
            },
        )
        self.assertEqual(
            encoder_parameter_report(graph),
            {
                "node_initialization": 4224,
                "relation_bases": 6144,
                "relation_direction_mixing": 60,
                "graph_pooling": 8608,
                "bottleneck_projection": 1072,
                "complete_graph_encoder": 23468,
                "relational_layer_0": 3188,
                "relational_layer_1": 3188,
                "relational_layer_2": 3188,
            },
        )
        difference = capacity_difference_percent(flat, graph)
        self.assertAlmostEqual(difference, 2.9298245614035086)
        self.assertLessEqual(difference, CAPACITY_TOLERANCE_PERCENT)
        self.assertEqual(FLAT_CAPACITY_MATCHED_FEEDFORWARD_WIDTH, 192)
        self.assertEqual(GRAPH_FEEDFORWARD_WIDTH, 64)

    def test_real_cpu_tensor_smoke(self):
        paired = self._paired(("E", "R"))
        with torch.no_grad():
            flat = FlatProgramEncoder().eval()(**self._flat_tensors(paired))
            graph = TypedGraphProgramEncoder().eval()(**self._graph_tensors(paired))
        self.assertEqual(flat.memory.device.type, "cpu")
        self.assertEqual(graph.memory.device.type, "cpu")
        self.assertTrue(torch.isfinite(flat.memory).all())
        self.assertTrue(torch.isfinite(graph.memory).all())


if __name__ == "__main__":
    unittest.main()
