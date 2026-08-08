"""Real-PyTorch C6 training, checkpoint, intervention, and metric tests."""

from __future__ import annotations

from dataclasses import fields, is_dataclass, replace
from pathlib import Path
import subprocess
import tempfile
import unittest

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "real PyTorch is required")
class C6RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from prototype.graph_encoder.tests.fixtures import procedural_fixture

        cls.root = Path(__file__).resolve().parents[3]
        cls.commit = subprocess.run(
            ("git", "rev-parse", "HEAD"), cwd=str(cls.root),
            stdout=subprocess.PIPE, universal_newlines=True, check=True
        ).stdout.strip()
        status = subprocess.run(
            ("git", "status", "--porcelain=v1", "--untracked-files=all"),
            cwd=str(cls.root), stdout=subprocess.PIPE,
            universal_newlines=True, check=True
        ).stdout.strip()
        if status:
            raise unittest.SkipTest("C6 provenance runtime requires a clean tree")
        cls.examples = tuple(
            replace(
                procedural_fixture(template).physical,
                split_name="operation_template",
                partition="train",
            )
            for template in ("E", "R", "EE", "RE")
        )

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="ge1-c6-runtime-")

    def tearDown(self):
        self.temporary.cleanup()

    def _train(self, arm, final_epoch, subdirectory, resume=None):
        from prototype.graph_encoder.config import (
            GE1TrainingConfig, frozen_encoder_config,
        )
        from prototype.graph_encoder.model import build_ge1_model
        from prototype.graph_encoder.training import run_ge1_training

        model = build_ge1_model(frozen_encoder_config(arm, 2026))
        result = run_ge1_training(
            model,
            self.examples,
            training_config=GE1TrainingConfig(),
            checkpoint_directory=Path(self.temporary.name) / subdirectory,
            repository_root=self.root,
            expected_commit=self.commit,
            final_epoch=final_epoch,
            engineering_smoke=True,
            resume_checkpoint=resume,
        )
        return model, result

    def test_two_epoch_training_both_arms_has_identical_schedule(self):
        flat_model, flat = self._train("flat", 2, "flat")
        graph_model, graph = self._train("typed_graph", 2, "graph")
        self.assertEqual(flat.optimizer_steps, graph.optimizer_steps)
        self.assertEqual(
            flat.training_example_presentations,
            graph.training_example_presentations,
        )
        self.assertEqual(
            tuple(item.family_order for item in flat.epoch_records),
            tuple(item.family_order for item in graph.epoch_records),
        )
        self.assertEqual(
            tuple(item.batch_boundaries for item in flat.epoch_records),
            tuple(item.batch_boundaries for item in graph.epoch_records),
        )
        self.assertEqual(len(flat.checkpoint_paths), 2)
        self.assertEqual(len(graph.checkpoint_paths), 2)
        self.assertIsNone(flat.selected_experimental_checkpoint)
        self.assertTrue(all(
            torch.isfinite(parameter).all().item()
            for model in (flat_model, graph_model)
            for parameter in model.parameters()
        ))

    def test_uninterrupted_equals_epoch_one_resume(self):
        from prototype.graph_encoder.autonomous import (
            autonomous_input_from_paired, run_autonomous_evaluation,
        )
        from prototype.graph_encoder.batching import build_paired_batch
        from prototype.graph_encoder.metrics import score_condition

        continuous_model, continuous = self._train("flat", 2, "continuous")
        unused_epoch_one_model, epoch_one = self._train("flat", 1, "epoch-one")
        del unused_epoch_one_model
        resumed_model, resumed = self._train(
            "flat", 2, "resumed", epoch_one.checkpoint_paths[-1]
        )
        for name, value in continuous_model.state_dict().items():
            self.assertTrue(torch.equal(value, resumed_model.state_dict()[name]), name)
        continuous_checkpoint = torch.load(
            continuous.checkpoint_paths[-1], map_location="cpu"
        )
        resumed_checkpoint = torch.load(
            resumed.checkpoint_paths[-1], map_location="cpu"
        )

        def equal_tree(left, right):
            if torch.is_tensor(left) or torch.is_tensor(right):
                return (
                    torch.is_tensor(left)
                    and torch.is_tensor(right)
                    and torch.equal(left, right)
                )
            if is_dataclass(left) or is_dataclass(right):
                return (
                    type(left) is type(right)
                    and is_dataclass(left)
                    and all(
                        equal_tree(
                            getattr(left, field.name),
                            getattr(right, field.name),
                        )
                        for field in fields(left)
                    )
                )
            if isinstance(left, dict):
                return list(left) == list(right) and all(
                    equal_tree(left[key], right[key]) for key in left
                )
            if isinstance(left, (tuple, list)):
                return len(left) == len(right) and all(
                    equal_tree(a, b) for a, b in zip(left, right)
                )
            return left == right

        self.assertTrue(equal_tree(
            continuous_checkpoint["optimizer_state"],
            resumed_checkpoint["optimizer_state"],
        ))
        self.assertEqual(continuous.optimizer_steps, resumed.optimizer_steps)
        self.assertEqual(
            continuous.training_example_presentations,
            resumed.training_example_presentations,
        )
        self.assertEqual(
            continuous.epoch_records[1].family_order,
            resumed.epoch_records[1].family_order,
        )
        self.assertEqual(
            continuous.epoch_records[1].mean_loss,
            resumed.epoch_records[1].mean_loss,
        )
        autonomous_input = autonomous_input_from_paired(
            build_paired_batch(self.examples), "flat"
        )
        continuous_predictions = run_autonomous_evaluation(
            continuous_model, (autonomous_input,), seed=2026
        )
        resumed_predictions = run_autonomous_evaluation(
            resumed_model, (autonomous_input,), seed=2026
        )
        for expected, observed in zip(
            continuous_predictions.conditions, resumed_predictions.conditions
        ):
            expected_without_timing = replace(
                expected, intervention_seconds=0.0, elapsed_seconds=0.0
            )
            observed_without_timing = replace(
                observed, intervention_seconds=0.0, elapsed_seconds=0.0
            )
            self.assertTrue(equal_tree(
                expected_without_timing, observed_without_timing
            ))

            targets = {
                item.physical_family_id: item.target for item in self.examples
            }
            expected_metrics = score_condition(expected, targets)
            observed_metrics = score_condition(observed, targets)
            expected_metrics.pop("metric_computation_seconds")
            observed_metrics.pop("metric_computation_seconds")
            self.assertTrue(equal_tree(expected_metrics, observed_metrics))

    def test_strict_reload_restores_model_optimizer_counters_and_rng(self):
        from prototype.graph_encoder.config import GE1TrainingConfig, frozen_encoder_config
        from prototype.graph_encoder.model import build_ge1_model
        from prototype.graph_encoder.provenance import training_partition_identity
        from prototype.graph_encoder.training import load_training_checkpoint

        trained, result = self._train("typed_graph", 1, "reload")
        fresh = build_ge1_model(frozen_encoder_config("typed_graph", 2026))
        optimizer = torch.optim.AdamW(
            fresh.parameters(), lr=1e-3, weight_decay=0.0
        )
        partition = training_partition_identity(
            tuple(item.physical_family_id for item in self.examples)
        )
        resumed = load_training_checkpoint(
            result.checkpoint_paths[-1], model=fresh, optimizer=optimizer,
            training_config=GE1TrainingConfig(), partition_identity=partition,
            expected_code_revision=self.commit, restore_rng=True,
        )
        self.assertEqual(resumed.completed_epoch, 1)
        self.assertEqual(resumed.optimizer_step_count, 1)
        self.assertEqual(resumed.training_example_presentations, 4)
        for name, value in trained.state_dict().items():
            self.assertTrue(torch.equal(value, fresh.state_dict()[name]), name)
        self.assertTrue(optimizer.state)

    def test_parameter_counts_and_receptive_field(self):
        from prototype.graph_encoder.metrics import (
            parameter_count_record, receptive_field_record,
        )
        from prototype.graph_encoder.model import build_matched_ge1_models

        flat, graph = build_matched_ge1_models(2026)
        report = parameter_count_record(flat, graph)
        self.assertEqual(report["flat_total_encoder"], 22800)
        self.assertEqual(report["graph_total_encoder"], 23468)
        self.assertEqual(report["encoder_difference"], 668)
        self.assertTrue(report["shared_decoder_by_top_level_component"])
        self.assertTrue(receptive_field_record()["maximum_distance_covered"])

    def test_nonfinite_loss_component_and_gradient_are_terminal_structured(self):
        from types import SimpleNamespace
        from prototype.graph_encoder.training import (
            C6TrainingError, _assert_finite_gradients, _assert_finite_loss,
        )

        loss = SimpleNamespace(
            as_dict=lambda: {"total": torch.tensor(float("nan"))},
            per_example={"total": torch.tensor([float("nan")])},
        )
        with self.assertRaises(C6TrainingError) as caught:
            _assert_finite_loss(loss, "run", "flat", 2026, 1, 0, ("family",))
        self.assertEqual(caught.exception.code, "nonfinite_loss")
        self.assertFalse(caught.exception.failure_record["target_payload_serialized"])

        model = torch.nn.Linear(1, 1)
        model.weight.grad = torch.tensor([[float("inf")]])
        with self.assertRaises(C6TrainingError) as caught:
            _assert_finite_gradients(
                model, "run", "flat", 2026, 1, 0, ("family",), "pre_clip_gradient"
            )
        self.assertEqual(caught.exception.code, "nonfinite_gradient")

    def test_interventions_change_supplied_memory_and_keep_identity(self):
        from prototype.graph_encoder.autonomous import (
            autonomous_input_from_paired, run_autonomous_evaluation,
        )
        from prototype.graph_encoder.batching import build_paired_batch
        from prototype.graph_encoder.model import build_ge1_model
        from prototype.graph_encoder.config import frozen_encoder_config

        model = build_ge1_model(frozen_encoder_config("flat", 2026))
        paired = build_paired_batch(self.examples)
        inputs = autonomous_input_from_paired(paired, "flat")
        supplied_memories = []
        hook = model.decoder.register_forward_pre_hook(
            lambda unused_module, values: supplied_memories.append(
                values[0].detach().clone()
            )
        )
        result = run_autonomous_evaluation(model, (inputs,), seed=2026)
        hook.remove()
        by_condition = {item.condition: item for item in result.conditions}
        true = by_condition["P_true"]
        shuffled = by_condition["P_shuffle"]
        mean = by_condition["P_mean"]
        self.assertTrue(shuffled.available)
        self.assertTrue(all(left != right for left, right in shuffled.memory_assignments))
        self.assertTrue(shuffled.memory_altered)
        self.assertTrue(mean.memory_altered)
        self.assertEqual(
            tuple(item.family_id for item in true.predictions),
            tuple(item.family_id for item in shuffled.predictions),
        )
        for condition in result.conditions:
            for prediction in condition.predictions:
                self.assertIsNotNone(prediction.raw_prediction)
                self.assertIsNotNone(prediction.constrained_prediction)
                self.assertIsNotNone(prediction.converted_prediction)
        count = len(result.family_order)
        self.assertEqual(len(supplied_memories), count * 3)
        true_memory = {
            family: supplied_memories[index]
            for index, family in enumerate(result.family_order)
        }
        for index, (recipient, donor) in enumerate(shuffled.memory_assignments):
            self.assertTrue(torch.equal(
                supplied_memories[count + index], true_memory[donor]
            ))
            self.assertNotEqual(recipient, donor)
        expected_mean = torch.cat(
            tuple(true_memory[family] for family in result.family_order), dim=0
        ).mean(dim=0, keepdim=True)
        for memory in supplied_memories[2 * count:]:
            self.assertTrue(torch.equal(memory, expected_mean))

        singleton_input = autonomous_input_from_paired(
            build_paired_batch(self.examples[:1]), "flat"
        )
        singleton_supplied = []
        hook = model.decoder.register_forward_pre_hook(
            lambda unused_module, values: singleton_supplied.append(
                values[0].detach().clone()
            )
        )
        singleton = run_autonomous_evaluation(
            model, (singleton_input,), seed=2026
        )
        hook.remove()
        singleton_conditions = {item.condition: item for item in singleton.conditions}
        self.assertFalse(singleton_conditions["P_shuffle"].available)
        self.assertEqual(
            singleton_conditions["P_shuffle"].unavailable_reason,
            "fewer_than_two_scored_examples",
        )
        self.assertEqual(len(singleton_conditions["P_mean"].singleton_mean_batches), 1)
        self.assertTrue(torch.equal(singleton_supplied[0], singleton_supplied[1]))

    def test_complete_metric_path_all_conditions(self):
        from prototype.graph_encoder.autonomous import (
            autonomous_input_from_paired, run_autonomous_evaluation,
        )
        from prototype.graph_encoder.batching import build_paired_batch
        from prototype.graph_encoder.config import frozen_encoder_config
        from prototype.graph_encoder.metrics import score_condition
        from prototype.graph_encoder.model import build_ge1_model

        model = build_ge1_model(frozen_encoder_config("typed_graph", 2026))
        paired = build_paired_batch(self.examples)
        autonomous = run_autonomous_evaluation(
            model,
            (autonomous_input_from_paired(paired, "typed_graph"),),
            seed=2026,
        )
        targets = {item.physical_family_id: item.target for item in self.examples}
        metrics = tuple(score_condition(item, targets) for item in autonomous.conditions)
        self.assertEqual({item["condition"] for item in metrics}, {
            "P_true", "P_shuffle", "P_mean"
        })
        for record in metrics:
            self.assertTrue(record["available"])
            self.assertEqual(record["physical_family_denominator"], 4)
            self.assertIn("primary", record["aggregates"])
            self.assertIn("geometry_error_by_channel_family", record["aggregates"])

    def test_production_prefix_exact_cases_permutation_and_missing_dependency(self):
        from prototype.controlled_data.builders import build_history
        from prototype.flat_baseline.autonomous import RawDecodedNode
        from prototype.flat_baseline.tests.test_constrained_v6 import (
            V6GrammarTensorTests,
        )
        from prototype.graph_baseline.conversion import graph_prediction_from_evidence
        from prototype.graph_baseline.graph_contract import (
            graph_edge_class_id,
            graph_from_reconstruction_target,
        )
        from prototype.graph_encoder.metrics import score_prediction_prefix
        from prototype.model_data.canonical import (
            canonical_nodes_and_edges, reconstruction_target,
        )
        from prototype.model_data.tests.fixtures import source
        from prototype.representation.model import GeometryEncoding

        def perfect(template):
            node = V6GrammarTensorTests()._authoritative_v6_prediction(template)
            history = build_history(source(template), GeometryEncoding.CONTINUOUS)
            nodes, edges = canonical_nodes_and_edges(history)
            target = reconstruction_target(
                nodes, edges, history.structure.operation_sequence
            )
            graph = graph_from_reconstruction_target(target)
            classes = [[0] * graph.node_count for unused in range(graph.node_count)]
            for edge in graph.directed_typed_edges:
                classes[edge.source][edge.destination] = edge.edge_type_id
            prediction = graph_prediction_from_evidence(
                node, classes, classes,
                [[False] * graph.node_count for unused in range(graph.node_count)],
            )
            return prediction, target

        single, single_target = perfect("E")
        double, double_target = perfect("RR")
        self.assertEqual(
            score_prediction_prefix(single, len(single_target.operation_sequence)).normalized_longest_executable_operation_prefix,
            1.0,
        )
        score = score_prediction_prefix(double, len(double_target.operation_sequence))
        self.assertEqual(score.normalized_longest_executable_operation_prefix, 1.0)
        self.assertEqual(score.unnormalized_longest_executable_prefix, 2)

        order = tuple(reversed(range(double.graph.node_count)))
        old_to_new = {old: new for new, old in enumerate(order)}
        node = double.node_prediction
        aligned = (
            "profile_family_logits", "predicted_profile_family_ids",
            "raw_profile_parameters", "constrained_profile_parameters",
            "raw_categorical_argmax_ids", "node_conditioned_categorical_ids",
            "categorical_correction_mask", "raw_node_type_argmax_ids",
            "grammar_constrained_node_type_ids", "node_type_correction_mask",
            "legal_node_type_masks", "grammar_state_evidence",
            "raw_axis_geometry", "constrained_axis_geometry",
            "axis_geometry_correction_mask",
        )
        replacements = {
            name: tuple(getattr(node, name)[old] for old in order)
            for name in aligned
        }
        replacements["raw_nodes"] = tuple(
            replace(node.raw_nodes[old], position=new)
            for new, old in enumerate(order)
        )
        replacements["same_history_raw_shadow_nodes"] = tuple(
            replace(node.same_history_raw_shadow_nodes[old], position=new)
            for new, old in enumerate(order)
        )
        permuted_node = replace(node, **replacements)
        raw = tuple(
            tuple(double.raw_graph_edge_predictions[source][destination] for destination in order)
            for source in order
        )
        masked = tuple(
            tuple(double.masked_graph_edge_predictions[source][destination] for destination in order)
            for source in order
        )
        correction = tuple(
            tuple(double.graph_edge_correction_mask[source][destination] for destination in order)
            for source in order
        )
        permuted = graph_prediction_from_evidence(
            permuted_node, raw, masked, correction
        )
        permuted_score = score_prediction_prefix(
            permuted, len(double_target.operation_sequence)
        )
        self.assertEqual(permuted_score.normalized_longest_executable_operation_prefix, 1.0)

        depends = graph_edge_class_id("depends_on")
        missing = [list(row) for row in double.masked_graph_edge_predictions]
        for source_index in range(len(missing)):
            for destination_index in range(len(missing)):
                if missing[source_index][destination_index] == depends:
                    missing[source_index][destination_index] = 0
        ambiguous = graph_prediction_from_evidence(
            double.node_prediction,
            tuple(tuple(row) for row in missing),
            tuple(tuple(row) for row in missing),
            tuple(tuple(False for unused in row) for row in missing),
        )
        ambiguous_score = score_prediction_prefix(
            ambiguous, len(double_target.operation_sequence)
        )
        self.assertEqual(ambiguous_score.normalized_longest_executable_operation_prefix, 0.0)
        self.assertEqual(ambiguous_score.first_failed_stage, "canonicalization")

    def test_provenance_failure_prevents_checkpoint_publication(self):
        from prototype.graph_encoder.config import GE1TrainingConfig, frozen_encoder_config
        from prototype.graph_encoder.errors import GraphEncoderError
        from prototype.graph_encoder.model import build_ge1_model
        from prototype.graph_encoder.provenance import training_partition_identity
        from prototype.graph_encoder.training import (
            plateau_state, save_training_checkpoint,
        )

        model = build_ge1_model(frozen_encoder_config("flat", 2026))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        destination = Path(self.temporary.name) / "must-not-exist.pt"

        def fail(*unused_arguments, **unused_keywords):
            raise GraphEncoderError("dirty_source_tree", "test failure")

        with self.assertRaises(GraphEncoderError):
            save_training_checkpoint(
                destination, model=model, optimizer=optimizer,
                training_config=GE1TrainingConfig(), completed_epoch=1,
                optimizer_step_count=1, training_example_presentations=4,
                plateau=plateau_state((1.0,)), epoch_records=(),
                family_order_history=(), data_order_random_state=None,
                provenance_context=object(), provenance_verifier=fail,
                partition_identity=training_partition_identity(
                    tuple(item.physical_family_id for item in self.examples)
                ),
                expected_code_revision=self.commit,
            )
        self.assertFalse(destination.exists())


if __name__ == "__main__":
    unittest.main()
