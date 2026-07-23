"""Deterministic model-ready CAD data interfaces."""

from .adapters import adapt_flat_mixed, adapt_typed_graph
from .batching import collate_flat, collate_graph
from .counterfactual import load_counterfactual_examples
from .errors import ModelDataError
from .loader import load_physical_examples

__all__ = (
    "ModelDataError",
    "adapt_flat_mixed",
    "adapt_typed_graph",
    "collate_flat",
    "collate_graph",
    "load_counterfactual_examples",
    "load_physical_examples",
)
