"""Frozen optimizer settings shared with the V6 two-epoch pilot."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class GraphTrainingConfig:
    seed: int = 2026
    batch_size: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    gradient_clip_norm: float = 1.0
    device: str = "cpu"
    epochs: int = 2
    graph_edge_loss_weight: float = 1.0

    def validate(self):
        if (
            self.seed != 2026 or self.batch_size != 8 or self.epochs != 2
            or self.learning_rate != 1e-3 or self.weight_decay != 0.0
            or self.gradient_clip_norm != 1.0 or self.device != "cpu"
            or self.graph_edge_loss_weight != 1.0
        ):
            raise ValueError("graph training protocol differs from frozen V6")

    def to_dict(self):
        self.validate()
        return asdict(self)
