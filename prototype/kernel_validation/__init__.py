"""CAD-kernel execution validation for controlled representation histories."""

from .executor import execute_sample
from .reporting import aggregate_results

__all__ = ["aggregate_results", "execute_sample"]
