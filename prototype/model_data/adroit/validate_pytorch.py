#!/usr/bin/env python3
"""CPU-only integration validation for prototype.model_data with real PyTorch."""

from __future__ import annotations

import compileall
import os
import platform
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


TEST_DIRECTORIES = (
    "prototype/representation/tests",
    "prototype/controlled_data/tests",
    "prototype/kernel_validation/tests",
    "prototype/counterfactual_edits/tests",
    "prototype/model_data/tests",
)

COMPILE_DIRECTORIES = (
    "prototype/representation",
    "prototype/controlled_data",
    "prototype/kernel_validation",
    "prototype/counterfactual_edits",
    "prototype/model_data",
)


def run_unit_tests():
    combined = unittest.TestSuite()

    for relative in TEST_DIRECTORIES:
        loader = unittest.TestLoader()
        discovered = loader.discover(
            start_dir=str(ROOT / relative),
            pattern="test_*.py",
            top_level_dir=str(ROOT),
        )
        combined.addTests(discovered)

    count = combined.countTestCases()
    print("Discovered tests:", count, flush=True)
    if count != 193:
        raise AssertionError("expected 193 tests, discovered %d" % count)

    result = unittest.TextTestRunner(verbosity=2).run(combined)
    if not result.wasSuccessful():
        raise SystemExit("unit-test validation failed")


def run_compileall():
    for relative in COMPILE_DIRECTORIES:
        path = ROOT / relative
        print("Compiling:", path, flush=True)
        if not compileall.compile_dir(str(path), quiet=1):
            raise AssertionError("compileall failed for %s" % path)


def assert_tensor(tensor, dtype, shape):
    import torch

    if not isinstance(tensor, torch.Tensor):
        raise AssertionError("expected torch.Tensor, received %r" % type(tensor))
    if tensor.dtype != dtype:
        raise AssertionError(
            "expected dtype %s, received %s" % (dtype, tensor.dtype)
        )
    if tuple(tensor.shape) != tuple(shape):
        raise AssertionError(
            "expected shape %r, received %r" % (tuple(shape), tuple(tensor.shape))
        )
    if not tensor.is_contiguous():
        raise AssertionError("tensor is not contiguous")
    if tensor.device.type != "cpu":
        raise AssertionError("validation tensor is not CPU-resident")


def assert_tensor_dict_equal(first, second):
    import torch

    if set(first) != set(second):
        raise AssertionError("tensor dictionaries have different keys")

    for name in first:
        if not torch.equal(first[name], second[name]):
            raise AssertionError("nondeterministic tensor field: %s" % name)


def validate_individual_records(flat, graph):
    import torch

    node_count = len(flat.categorical_ids)
    edge_count = len(graph.edge_type_ids)
    operation_count = len(graph.operation_sequence)

    flat_first = flat.to_torch()
    flat_second = flat.to_torch()
    assert_tensor_dict_equal(flat_first, flat_second)

    assert_tensor(
        flat_first["categorical_ids"], torch.long, (node_count, 10)
    )
    assert_tensor(flat_first["geometry"], torch.float32, (node_count, 39))
    assert_tensor(flat_first["geometry_mask"], torch.bool, (node_count, 39))

    graph_first = graph.to_torch()
    graph_second = graph.to_torch()
    assert_tensor_dict_equal(graph_first, graph_second)

    assert_tensor(graph_first["node_type_ids"], torch.long, (node_count,))
    assert_tensor(graph_first["edge_index"], torch.long, (2, edge_count))
    assert_tensor(graph_first["edge_type_ids"], torch.long, (edge_count,))
    assert_tensor(
        graph_first["categorical_attributes"],
        torch.long,
        (node_count, 9),
    )
    assert_tensor(graph_first["geometry"], torch.float32, (node_count, 39))
    assert_tensor(
        graph_first["geometry_mask"], torch.bool, (node_count, 39)
    )
    assert_tensor(
        graph_first["operation_sequence"],
        torch.long,
        (operation_count,),
    )

    target_first = graph.target.to_torch()
    target_second = graph.target.to_torch()
    assert_tensor_dict_equal(target_first, target_second)

    assert_tensor(target_first["node_type_ids"], torch.long, (node_count,))
    assert_tensor(
        target_first["categorical_attributes"],
        torch.long,
        (node_count, 9),
    )
    assert_tensor(target_first["edge_index"], torch.long, (2, edge_count))
    assert_tensor(target_first["edge_type_ids"], torch.long, (edge_count,))
    assert_tensor(
        target_first["boolean_mode_targets"], torch.long, (node_count,)
    )
    assert_tensor(
        target_first["operation_sequence"],
        torch.long,
        (operation_count,),
    )
    assert_tensor(
        target_first["geometry"], torch.float32, (node_count, 39)
    )
    assert_tensor(
        target_first["geometry_mask"], torch.bool, (node_count, 39)
    )


def validate_batches(flat_items, graph_items):
    import torch

    from prototype.model_data.batching import collate_flat, collate_graph

    flat_batch = collate_flat(tuple(reversed(flat_items)))
    graph_batch = collate_graph(tuple(reversed(graph_items)))

    # Collation must be order-independent and deterministic.
    if flat_batch != collate_flat(flat_items):
        raise AssertionError("flat collation is order-dependent")
    if graph_batch != collate_graph(graph_items):
        raise AssertionError("graph collation is order-dependent")

    batch_size = len(flat_items)
    maximum_nodes = max(len(item.categorical_ids) for item in flat_items)
    maximum_operations = max(
        len(item.operation_sequence) for item in graph_items
    )
    total_nodes = sum(len(item.node_type_ids) for item in graph_items)
    total_edges = sum(len(item.edge_type_ids) for item in graph_items)

    flat_tensors = flat_batch.to_torch()
    assert_tensor_dict_equal(flat_tensors, flat_batch.to_torch())

    assert_tensor(
        flat_tensors["categorical_ids"],
        torch.long,
        (batch_size, maximum_nodes, 10),
    )
    assert_tensor(
        flat_tensors["geometry"],
        torch.float32,
        (batch_size, maximum_nodes, 39),
    )
    assert_tensor(
        flat_tensors["geometry_mask"],
        torch.bool,
        (batch_size, maximum_nodes, 39),
    )
    assert_tensor(
        flat_tensors["padding_mask"],
        torch.bool,
        (batch_size, maximum_nodes),
    )

    flat_by_id = {
        item.physical_family_id: item for item in flat_items
    }
    for row, family_id in enumerate(flat_batch.family_ids):
        local = flat_by_id[family_id]
        local_nodes = len(local.categorical_ids)

        if not bool(flat_tensors["padding_mask"][row, :local_nodes].all()):
            raise AssertionError("valid flat nodes are masked out")
        if bool(flat_tensors["padding_mask"][row, local_nodes:].any()):
            raise AssertionError("flat padding is marked valid")

        expected_mask = torch.tensor(
            local.geometry_mask, dtype=torch.bool
        ).contiguous()
        if not torch.equal(
            flat_tensors["geometry_mask"][row, :local_nodes],
            expected_mask,
        ):
            raise AssertionError("flat geometry-mask alignment failed")

    graph_tensors = graph_batch.to_torch()
    assert_tensor_dict_equal(graph_tensors, graph_batch.to_torch())

    assert_tensor(
        graph_tensors["node_type_ids"], torch.long, (total_nodes,)
    )
    assert_tensor(
        graph_tensors["edge_index"], torch.long, (2, total_edges)
    )
    assert_tensor(
        graph_tensors["edge_type_ids"], torch.long, (total_edges,)
    )
    assert_tensor(
        graph_tensors["categorical_attributes"],
        torch.long,
        (total_nodes, 9),
    )
    assert_tensor(
        graph_tensors["geometry"], torch.float32, (total_nodes, 39)
    )
    assert_tensor(
        graph_tensors["geometry_mask"], torch.bool, (total_nodes, 39)
    )
    assert_tensor(
        graph_tensors["graph_offsets"], torch.long, (batch_size + 1,)
    )
    assert_tensor(
        graph_tensors["edge_offsets"], torch.long, (batch_size + 1,)
    )
    assert_tensor(
        graph_tensors["node_graph_ids"], torch.long, (total_nodes,)
    )
    assert_tensor(
        graph_tensors["node_mask"],
        torch.bool,
        (batch_size, maximum_nodes),
    )

    graph_by_id = {
        item.physical_family_id: item for item in graph_items
    }
    for graph_index, family_id in enumerate(graph_batch.family_ids):
        local = graph_by_id[family_id]
        node_offset = graph_batch.graph_offsets[graph_index]
        node_stop = graph_batch.graph_offsets[graph_index + 1]
        edge_start = graph_batch.edge_offsets[graph_index]
        edge_stop = graph_batch.edge_offsets[graph_index + 1]

        expected_edges = (
            torch.tensor(local.edge_index, dtype=torch.long)
            + node_offset
        ).contiguous()
        actual_edges = graph_tensors["edge_index"][:, edge_start:edge_stop]

        if not torch.equal(actual_edges, expected_edges):
            raise AssertionError(
                "graph edge indices were not offset exactly once"
            )

        expected_mask = torch.tensor(
            local.geometry_mask, dtype=torch.bool
        ).contiguous()
        actual_mask = graph_tensors["geometry_mask"][
            node_offset:node_stop
        ]
        if not torch.equal(actual_mask, expected_mask):
            raise AssertionError("graph geometry-mask alignment failed")

        expected_graph_ids = torch.full(
            (node_stop - node_offset,),
            graph_index,
            dtype=torch.long,
        )
        if not torch.equal(
            graph_tensors["node_graph_ids"][node_offset:node_stop],
            expected_graph_ids,
        ):
            raise AssertionError("node-to-graph assignments are incorrect")

    flat_target = flat_batch.target.to_torch()
    graph_target = graph_batch.target.to_torch()

    for target in (flat_target, graph_target):
        target_edges = target["edge_type_ids"].numel()

        assert_tensor(
            target["node_type_ids"],
            torch.long,
            (batch_size, maximum_nodes),
        )
        assert_tensor(
            target["categorical_attributes"],
            torch.long,
            (batch_size, maximum_nodes, 9),
        )
        assert_tensor(
            target["boolean_mode_targets"],
            torch.long,
            (batch_size, maximum_nodes),
        )
        assert_tensor(
            target["geometry"],
            torch.float32,
            (batch_size, maximum_nodes, 39),
        )
        assert_tensor(
            target["geometry_mask"],
            torch.bool,
            (batch_size, maximum_nodes, 39),
        )
        assert_tensor(
            target["node_mask"],
            torch.bool,
            (batch_size, maximum_nodes),
        )
        assert_tensor(
            target["operation_sequence"],
            torch.long,
            (batch_size, maximum_operations),
        )
        assert_tensor(
            target["operation_mask"],
            torch.bool,
            (batch_size, maximum_operations),
        )
        assert_tensor(
            target["edge_index"], torch.long, (2, target_edges)
        )
        assert_tensor(
            target["edge_type_ids"], torch.long, (target_edges,)
        )
        assert_tensor(
            target["edge_offsets"], torch.long, (batch_size + 1,)
        )

        padded_operations = ~target["operation_mask"]
        if bool(padded_operations.any()):
            if not bool(
                (target["operation_sequence"][padded_operations] == -1).all()
            ):
                raise AssertionError(
                    "padded operation indices do not use -1"
                )

    if not torch.equal(
        flat_target["node_mask"], flat_tensors["padding_mask"]
    ):
        raise AssertionError("flat input and target node masks disagree")


def validate_real_pytorch():
    import torch

    from prototype.controlled_data.factors import (
        PrimitiveFamily,
        ReferencePlane,
    )
    from prototype.model_data.adapters import (
        adapt_flat_mixed,
        adapt_typed_graph,
    )
    from prototype.model_data.counterfactual import (
        load_counterfactual_examples,
    )
    from prototype.model_data.loader import load_physical_examples
    from prototype.model_data.tests.fixtures import (
        source,
        write_counterfactual_corpus,
        write_physical_corpus,
    )

    torch.set_num_threads(1)

    print("Python executable:", sys.executable, flush=True)
    print("Python version:", platform.python_version(), flush=True)
    print("PyTorch version:", torch.__version__, flush=True)
    print("PyTorch CUDA build:", torch.version.cuda, flush=True)
    print("cuDNN version:", torch.backends.cudnn.version(), flush=True)
    print("CUDA visible devices:", os.environ.get("CUDA_VISIBLE_DEVICES"), flush=True)
    print("CUDA available:", torch.cuda.is_available(), flush=True)

    if sys.version_info[:2] != (3, 8):
        raise AssertionError("expected Python 3.8")
    if not torch.__version__.startswith("1.11"):
        raise AssertionError("expected PyTorch 1.11")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise AssertionError("CPU-only validation did not hide CUDA devices")

    with tempfile.TemporaryDirectory(prefix="model-data-physical-") as temporary:
        write_physical_corpus(
            temporary,
            (
                source(
                    "E",
                    PrimitiveFamily.RECTANGLE_LINES,
                    ReferencePlane.XY,
                ),
                source(
                    "R",
                    PrimitiveFamily.CIRCLE,
                    ReferencePlane.XZ,
                ),
                source(
                    "ER",
                    PrimitiveFamily.CAPSULE_LINE_ARC,
                    ReferencePlane.YZ,
                ),
            ),
        )

        physical_first = load_physical_examples(temporary)
        physical_second = load_physical_examples(temporary)

        if physical_first != physical_second:
            raise AssertionError("physical loading is nondeterministic")

        flat_items = tuple(
            adapt_flat_mixed(item) for item in physical_first
        )
        graph_items = tuple(
            adapt_typed_graph(item) for item in physical_first
        )

        if flat_items != tuple(
            adapt_flat_mixed(item) for item in physical_second
        ):
            raise AssertionError("flat adaptation is nondeterministic")
        if graph_items != tuple(
            adapt_typed_graph(item) for item in physical_second
        ):
            raise AssertionError("graph adaptation is nondeterministic")

        for flat, graph in zip(flat_items, graph_items):
            if flat.physical_family_id != graph.physical_family_id:
                raise AssertionError("flat and graph family alignment failed")
            if flat.target is not graph.target:
                raise AssertionError(
                    "flat and graph records do not share one target"
                )
            validate_individual_records(flat, graph)

        validate_batches(flat_items, graph_items)

    with tempfile.TemporaryDirectory(
        prefix="model-data-counterfactual-"
    ) as temporary:
        write_counterfactual_corpus(temporary)
        examples = load_counterfactual_examples(temporary)

        if len(examples) != 1:
            raise AssertionError("expected one counterfactual example")

        example = examples[0]
        if example.evaluation_only is not True:
            raise AssertionError("counterfactual record is not evaluation-only")
        if not example.edited_attribute_paths:
            raise AssertionError("counterfactual edited paths are empty")
        if not example.unaffected_attribute_paths:
            raise AssertionError("counterfactual unaffected paths are empty")
        if not set(example.edited_attribute_paths).isdisjoint(
            example.unaffected_attribute_paths
        ):
            raise AssertionError("counterfactual locality paths overlap")

        # Counterfactual records retain physical endpoints. Exercise both
        # adapters and reconstruction targets for source and target.
        for endpoint in (example.source, example.target):
            flat = adapt_flat_mixed(endpoint)
            graph = adapt_typed_graph(endpoint)
            validate_individual_records(flat, graph)

    print("Real PyTorch model-data validation passed.", flush=True)


def main():
    print("Repository:", ROOT, flush=True)
    print("Commit validation begins.", flush=True)

    run_unit_tests()
    run_compileall()
    validate_real_pytorch()

    print("ALL ADROIT MODEL-DATA CHECKS PASSED", flush=True)


if __name__ == "__main__":
    main()
