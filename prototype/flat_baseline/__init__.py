"""Flat mixed vector-quantized CAD reconstruction baseline."""

from __future__ import annotations

__all__ = (
    "FlatBaselineConfig",
    "TrainingConfig",
    "FlatMixedVQModel",
    "flat_mixed_vq_loss",
)


def __getattr__(name):
    # Keep configuration and static compatibility checks usable on machines
    # without PyTorch while preserving the package-level public API.
    if name == "FlatBaselineConfig":
        from .config import FlatBaselineConfig

        return FlatBaselineConfig
    if name == "TrainingConfig":
        from .training_config import TrainingConfig

        return TrainingConfig
    if name == "FlatMixedVQModel":
        from .model import FlatMixedVQModel

        return FlatMixedVQModel
    if name == "flat_mixed_vq_loss":
        from .losses import flat_mixed_vq_loss

        return flat_mixed_vq_loss
    raise AttributeError(name)
