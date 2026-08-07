"""Static C5 shared-decoder, position, inference, and checkpoint contracts."""

from __future__ import annotations

import ast
from pathlib import Path
import unittest

from prototype.graph_encoder.config import (
    FROZEN_FLAT_FEEDFORWARD_WIDTH,
    FROZEN_GRAPH_FEEDFORWARD_WIDTH,
    FROZEN_RELATION_BASIS_COUNT,
    frozen_encoder_config,
)
from prototype.graph_encoder.decoder_contract import (
    BOOKKEEPING_ONLY_VALUES,
    EXCLUDED_POSITION_SIGNALS,
    OUTPUT_POSITION_SIGNALS,
    output_position_contract_metadata,
)


PACKAGE = Path(__file__).parents[1]
PRODUCTION = (
    PACKAGE / "shared_decoder.py",
    PACKAGE / "model.py",
    PACKAGE / "losses.py",
    PACKAGE / "checkpoint.py",
    PACKAGE / "decoder_contract.py",
)


class C5StaticContractTests(unittest.TestCase):
    def test_frozen_encoder_capacity_authority_is_unchanged(self):
        self.assertEqual(FROZEN_FLAT_FEEDFORWARD_WIDTH, 192)
        self.assertEqual(FROZEN_GRAPH_FEEDFORWARD_WIDTH, 64)
        self.assertEqual(FROZEN_RELATION_BASIS_COUNT, 2)
        self.assertEqual(
            frozen_encoder_config("flat").encoder_feedforward_width, 192
        )
        self.assertEqual(
            frozen_encoder_config("typed_graph").encoder_feedforward_width, 64
        )
        with self.assertRaisesRegex(ValueError, "unauthorized_configuration"):
            frozen_encoder_config("flat").__class__(
                encoder="flat", encoder_feedforward_width=64
            ).validate()

    def test_output_position_inventory_is_complete_and_frozen(self):
        names = tuple(item.name for item in OUTPUT_POSITION_SIGNALS)
        self.assertEqual(names, (
            "decoder_output_step_absolute_embedding",
            "pair_source_absolute_embedding",
            "pair_destination_absolute_embedding",
            "pair_signed_relative_serialized_position",
        ))
        self.assertTrue(all(item.neural_influence for item in OUTPUT_POSITION_SIGNALS))
        self.assertTrue(all(item.applies_identically_to_both_arms for item in OUTPUT_POSITION_SIGNALS))
        self.assertTrue(all(not item.enters_encoder for item in OUTPUT_POSITION_SIGNALS))
        self.assertIn("graph_v1_c1_position_class_projection", EXCLUDED_POSITION_SIGNALS)
        self.assertEqual(
            output_position_contract_metadata()["semantic_signals"],
            [vars(item) for item in OUTPUT_POSITION_SIGNALS],
        )
        shared_source = (PACKAGE / "shared_decoder.py").read_text(
            encoding="utf-8"
        )
        retained_source = (
            PACKAGE.parent / "flat_baseline" / "model.py"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "shifted + self.decoder_position_embedding(positions)",
            retained_source,
        )
        for fragment in (
            "source_position = position_values.unsqueeze(2)",
            "destination_position = position_values.unsqueeze(1)",
            "positions.view(1, count, 1) - positions.view(1, 1, count)",
            "source_position,\n            destination_position,",
        ):
            self.assertIn(fragment, shared_source)

    def test_bookkeeping_contract_distinguishes_permitted_routing(self):
        values = {item["name"]: item for item in BOOKKEEPING_ONLY_VALUES}
        self.assertEqual(set(values), {
            "pair_source_and_destination_indices",
            "output_node_mask",
            "graph_offsets",
            "edge_offsets_and_node_graph_ids",
        })
        self.assertTrue(all(not item["neural_feature"] for item in values.values()))

    def test_inference_signatures_are_target_free(self):
        tree = ast.parse((PACKAGE / "shared_decoder.py").read_text())
        decoder = next(
            item for item in tree.body
            if isinstance(item, ast.ClassDef) and item.name == "SharedGE1Decoder"
        )
        forward = next(
            item for item in decoder.body
            if isinstance(item, ast.FunctionDef) and item.name == "forward"
        )
        arguments = {
            item.arg for item in forward.args.args + forward.args.kwonlyargs
        }
        self.assertEqual(
            arguments, {"self", "memory", "node_counts", "node_count_source"}
        )
        self.assertTrue(arguments.isdisjoint({
            "target", "operation_sequence", "family_id", "template", "partition"
        }))

    def test_model_forward_has_no_downstream_arm_branch(self):
        tree = ast.parse((PACKAGE / "model.py").read_text())
        model = next(
            item for item in tree.body
            if isinstance(item, ast.ClassDef) and item.name == "GE1Model"
        )
        forward = next(
            item for item in model.body
            if isinstance(item, ast.FunctionDef) and item.name == "forward"
        )
        tree_dump = ast.dump(forward)
        string_literals = {
            item.value for item in ast.walk(forward)
            if isinstance(item, ast.Constant) and isinstance(item.value, str)
        }
        attributes = {
            item.attr for item in ast.walk(forward)
            if isinstance(item, ast.Attribute)
        }
        self.assertNotIn("encoder", attributes)
        self.assertTrue(string_literals.isdisjoint({"typed_graph", "flat"}))
        self.assertIn("memory", attributes)
        self.assertNotIn("prequant", attributes)
        self.assertNotIn("config", tree_dump)

    def test_c5_production_has_no_data_loader_or_protected_access_import(self):
        forbidden = (
            "model_data.loader",
            "graph_encoder.partitions",
            "load_train",
            "load_development",
            "operation_template.json",
            "history_depth",
            "geometry_extrapolation",
        )
        for path in PRODUCTION:
            source = path.read_text(encoding="utf-8")
            imports = "\n".join(
                ast.dump(item)
                for item in ast.walk(ast.parse(source))
                if isinstance(item, (ast.Import, ast.ImportFrom))
            )
            for fragment in forbidden:
                self.assertNotIn(fragment, imports, str(path))

    def test_checkpoint_source_requires_strict_reload(self):
        source = (PACKAGE / "checkpoint.py").read_text(encoding="utf-8")
        self.assertIn("load_state_dict(state, strict=True)", source)
        self.assertNotIn("strict=False", source)

    def test_python_38_grammar_for_every_c5_source(self):
        for path in PRODUCTION:
            ast.parse(
                path.read_text(encoding="utf-8"),
                filename=str(path),
                feature_version=(3, 8),
            )


if __name__ == "__main__":
    unittest.main()
