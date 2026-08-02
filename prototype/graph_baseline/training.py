"""Minimal graph-v1 optimization and strict checkpoint I/O."""

from __future__ import annotations

import torch

from .losses import graph_v1_loss
from .model import GraphV1Model


def build_graph_optimizer(model, config):
    if not isinstance(model, GraphV1Model):
        raise TypeError("model must be GraphV1Model")
    config.validate()
    return torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )


def graph_training_step(
    model, optimizer, inputs, target, profile_targets, model_config, training_config
):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    output = model(target=target, profile_targets=profile_targets, **inputs)
    losses = graph_v1_loss(output, target, profile_targets, model_config)
    losses.total.backward()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and (
            parameter.grad is None or not torch.isfinite(parameter.grad).all()
        ):
            raise ValueError("missing or nonfinite graph gradient for {}".format(name))
    torch.nn.utils.clip_grad_norm_(
        model.parameters(), training_config.gradient_clip_norm
    )
    optimizer.step()
    for value in model.state_dict().values():
        if torch.is_tensor(value) and not torch.isfinite(value).all():
            raise ValueError("nonfinite graph model state")
    return output, losses
