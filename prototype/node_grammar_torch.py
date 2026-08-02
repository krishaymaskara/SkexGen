"""PyTorch selection for the frozen V5 prefix-conditioned node grammar."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from prototype.model_data.vocab import NODE_TYPES
from prototype.node_grammar import (
    NodeGrammarError,
    V5_NODE_GRAMMAR,
    grammar_state_evidence,
    legal_next_node_ids,
    validate_node_grammar_contract,
)


@dataclass(frozen=True)
class NodeGrammarSelection:
    raw_node_type_argmax_ids: torch.Tensor
    grammar_constrained_node_type_ids: torch.Tensor
    node_type_correction_mask: torch.Tensor
    legal_node_type_mask: torch.Tensor
    grammar_state_evidence: tuple


def select_prefix_conditioned_node_types(
    raw_node_type_logits,
    generated_prefix_node_ids,
    requested_node_counts,
    *,
    current_positions,
    active_mask=None,
    contract=V5_NODE_GRAMMAR,
):
    """Mask illegal classes and deterministically select one legal next node."""

    validate_node_grammar_contract(contract)
    if not torch.is_tensor(raw_node_type_logits):
        raise NodeGrammarError("invalid_v5_node_logits", "logits must be a tensor")
    if (
        not raw_node_type_logits.dtype.is_floating_point
        or raw_node_type_logits.dim() != 2
        or raw_node_type_logits.size(0) == 0
        or raw_node_type_logits.size(1) != len(NODE_TYPES.tokens)
        or not torch.isfinite(raw_node_type_logits).all()
    ):
        raise NodeGrammarError(
            "invalid_v5_node_logits", "logits must be finite floating [B, C]"
        )
    batch_size = raw_node_type_logits.size(0)
    if (
        not torch.is_tensor(requested_node_counts)
        or requested_node_counts.dtype != torch.long
        or tuple(requested_node_counts.shape) != (batch_size,)
        or requested_node_counts.device != raw_node_type_logits.device
    ):
        raise NodeGrammarError(
            "invalid_v5_requested_node_count",
            "requested counts must be torch.long [B] on the logits device",
        )
    if not isinstance(generated_prefix_node_ids, (tuple, list)) or len(
        generated_prefix_node_ids
    ) != batch_size:
        raise NodeGrammarError(
            "invalid_v5_generated_prefix", "one prefix is required per batch row"
        )
    if (
        not torch.is_tensor(current_positions)
        or current_positions.dtype != torch.long
        or tuple(current_positions.shape) != (batch_size,)
        or current_positions.device != raw_node_type_logits.device
    ):
        raise NodeGrammarError(
            "invalid_v5_node_selection",
            "current positions must be torch.long [B] on the logits device",
        )
    if active_mask is None:
        active_mask = torch.ones(
            batch_size, dtype=torch.bool, device=raw_node_type_logits.device
        )
    if (
        not torch.is_tensor(active_mask)
        or active_mask.dtype != torch.bool
        or tuple(active_mask.shape) != (batch_size,)
        or active_mask.device != raw_node_type_logits.device
    ):
        raise NodeGrammarError(
            "invalid_v5_node_selection", "active mask must be Boolean [B]"
        )

    raw = raw_node_type_logits.argmax(dim=-1)
    selected = torch.full_like(raw, NODE_TYPES.id(None))
    legal_mask = torch.zeros_like(raw_node_type_logits, dtype=torch.bool)
    evidence = []
    for index in range(batch_size):
        if not bool(active_mask[index].item()):
            evidence.append({"state": "inactive"})
            continue
        count = int(requested_node_counts[index].item())
        prefix = tuple(generated_prefix_node_ids[index])
        if int(current_positions[index].item()) != len(prefix):
            raise NodeGrammarError(
                "invalid_v5_generated_prefix",
                "current position disagrees with constrained prefix length",
            )
        legal = legal_next_node_ids(prefix, count, contract)
        legal_mask[index, list(legal)] = True
        constrained_logits = raw_node_type_logits[index].masked_fill(
            ~legal_mask[index], float("-inf")
        )
        selected[index] = constrained_logits.argmax(dim=-1)
        if int(selected[index].item()) not in legal:
            raise NodeGrammarError(
                "invalid_v5_node_selection", "selected node is not legal"
            )
        evidence.append(grammar_state_evidence(prefix, count, legal))
    correction = active_mask & (raw != selected)
    return NodeGrammarSelection(
        raw.contiguous(),
        selected.contiguous(),
        correction.contiguous(),
        legal_mask.contiguous(),
        tuple(evidence),
    )
