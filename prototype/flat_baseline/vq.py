"""Dense-one-hot-free exponential-moving-average vector quantization."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class VQOutput:
    quantized: torch.Tensor
    loss: torch.Tensor
    per_example_loss: torch.Tensor
    indices: torch.Tensor
    assignment_counts: torch.Tensor
    active_code_count: torch.Tensor
    utilization: torch.Tensor
    perplexity: torch.Tensor


class EMAVectorQuantizer(nn.Module):
    """One mixed codebook updated through registered EMA buffers."""

    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        commitment_cost,
        decay,
        epsilon,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon

        initial = torch.empty(num_embeddings, embedding_dim)
        nn.init.normal_(initial)
        self.register_buffer("embedding", initial)
        self.register_buffer("ema_cluster_size", torch.ones(num_embeddings))
        self.register_buffer("ema_weight", initial.clone())

    def forward(self, inputs):
        if inputs.size(-1) != self.embedding_dim:
            raise ValueError("VQ input dimension disagrees with the codebook")
        flat = inputs.reshape(-1, self.embedding_dim)
        distances = (
            flat.pow(2).sum(dim=1, keepdim=True)
            + self.embedding.pow(2).sum(dim=1).unsqueeze(0)
            - 2.0 * torch.matmul(flat, self.embedding.t())
        )
        flat_indices = distances.argmin(dim=1)
        quantized = F.embedding(flat_indices, self.embedding).view_as(inputs)
        counts = torch.bincount(
            flat_indices, minlength=self.num_embeddings
        )

        if self.training:
            with torch.no_grad():
                sums = torch.zeros_like(self.ema_weight)
                sums.index_add_(0, flat_indices, flat)
                self.ema_cluster_size.mul_(self.decay).add_(
                    counts.to(self.ema_cluster_size.dtype),
                    alpha=1.0 - self.decay,
                )
                self.ema_weight.mul_(self.decay).add_(
                    sums, alpha=1.0 - self.decay
                )
                total = self.ema_cluster_size.sum()
                smoothed = (
                    (self.ema_cluster_size + self.epsilon)
                    / (total + self.num_embeddings * self.epsilon)
                    * total
                )
                self.embedding.copy_(
                    self.ema_weight / smoothed.unsqueeze(1).clamp_min(self.epsilon)
                )

        per_example_loss = self.commitment_cost * (
            inputs - quantized.detach()
        ).pow(2).reshape(inputs.size(0), -1).mean(dim=1)
        loss = per_example_loss.mean()
        straight_through = inputs + (quantized - inputs).detach()
        probabilities = counts.to(inputs.dtype) / max(flat_indices.numel(), 1)
        nonzero = probabilities > 0
        perplexity = torch.exp(
            -(probabilities[nonzero] * torch.log(probabilities[nonzero])).sum()
        )
        active = nonzero.sum()
        return VQOutput(
            straight_through.contiguous(),
            loss,
            per_example_loss,
            flat_indices.view(inputs.shape[:-1]),
            counts,
            active,
            active.to(inputs.dtype) / float(self.num_embeddings),
            perplexity,
        )
