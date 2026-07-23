"""Deterministic controlled-data generation for the CAD representation prototype."""

from .config import ConfigurationError, GeneratorConfig
from .dataset import GenerationError, generate_corpus

__all__ = ["ConfigurationError", "GenerationError", "GeneratorConfig", "generate_corpus"]
