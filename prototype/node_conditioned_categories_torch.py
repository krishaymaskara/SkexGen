"""PyTorch 1.11 node-conditioned categorical selection for V4."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from prototype.model_data.vocab import NODE_TYPES
from prototype.node_conditioned_categories import (
    NodeConditionedCategoricalError,
    V4_RETAINED_CATEGORICAL_FIELDS,
)


@dataclass(frozen=True)
class TensorNodeConditionedCategoricalSelection:
    raw_categorical_argmax_ids: torch.Tensor
    node_conditioned_categorical_ids: torch.Tensor
    correction_mask: torch.Tensor
    applicability_mask: torch.Tensor


def select_node_conditioned_categorical_ids(
    categorical_logits,
    predicted_node_type_ids,
    padding_mask=None,
    contract=V4_RETAINED_CATEGORICAL_FIELDS,
):
    """Select valid retained categories using predicted node types only."""

    leading_shape = _validate_inputs(
        categorical_logits, predicted_node_type_ids, padding_mask, contract
    )
    if padding_mask is None:
        padding_mask = torch.ones(
            leading_shape,
            dtype=torch.bool,
            device=predicted_node_type_ids.device,
        )
    raw = torch.stack(
        tuple(logits.argmax(dim=-1) for logits in categorical_logits), dim=-1
    ).detach()
    selected = torch.empty_like(raw)
    applicable = torch.zeros_like(raw, dtype=torch.bool)
    pad_node_id = NODE_TYPES.pad_id
    none_node_id = NODE_TYPES.id(None)
    invalid_node = padding_mask & (
        (predicted_node_type_ids < 0)
        | (predicted_node_type_ids >= len(NODE_TYPES.tokens))
        | (predicted_node_type_ids == none_node_id)
    )
    if invalid_node.any():
        raise NodeConditionedCategoricalError(
            "invalid_predicted_node_type",
            repr(predicted_node_type_ids[invalid_node].detach().cpu().tolist()),
        )
    predicted_padding = (~padding_mask) | (predicted_node_type_ids == pad_node_id)
    for field, logits in zip(contract, categorical_logits):
        result = torch.full_like(predicted_node_type_ids, field.sentinel_id)
        for node_type in field.applicable_node_types:
            node_id = NODE_TYPES.id(node_type)
            positions = padding_mask & (predicted_node_type_ids == node_id)
            if not positions.any():
                continue
            valid_ids = field.valid_ids(node_type)
            if not valid_ids:
                raise NodeConditionedCategoricalError(
                    "no_valid_category_for_predicted_node_type",
                    "{} has no valid IDs for {}".format(field.name, node_type),
                )
            valid_index = torch.tensor(
                valid_ids, dtype=torch.long, device=logits.device
            )
            local = logits.index_select(-1, valid_index).argmax(dim=-1)
            chosen = valid_index[local]
            result = torch.where(positions, chosen, result)
            applicable[..., field.output_position] = (
                applicable[..., field.output_position] | positions
            )
        result = torch.where(
            predicted_padding,
            torch.full_like(result, field.sentinel_id),
            result,
        )
        selected[..., field.output_position] = result
    correction = selected != raw
    _validate_post_selection(selected, predicted_node_type_ids, padding_mask)
    return TensorNodeConditionedCategoricalSelection(
        raw.contiguous(),
        selected.detach().contiguous(),
        correction.detach().contiguous(),
        applicable.detach().contiguous(),
    )


def _validate_inputs(logits_tuple, node_type_ids, padding_mask, contract):
    if isinstance(contract, tuple):
        for field in contract:
            for node_type in getattr(field, "applicable_node_types", ()):
                if not field.valid_ids(node_type):
                    raise NodeConditionedCategoricalError(
                        "no_valid_category_for_predicted_node_type",
                        "{} has no valid IDs for {}".format(
                            getattr(field, "name", "unknown"), node_type
                        ),
                    )
    if contract != V4_RETAINED_CATEGORICAL_FIELDS:
        raise NodeConditionedCategoricalError(
            "invalid_v4_categorical_logits", "categorical contract layout changed"
        )
    if not isinstance(logits_tuple, tuple) or len(logits_tuple) != len(
        V4_RETAINED_CATEGORICAL_FIELDS
    ):
        raise NodeConditionedCategoricalError(
            "invalid_v4_categorical_logits", "exactly five logits are required"
        )
    if not torch.is_tensor(node_type_ids) or node_type_ids.dtype != torch.long:
        raise NodeConditionedCategoricalError(
            "invalid_predicted_node_type", "node IDs must use torch.long"
        )
    leading_shape = tuple(node_type_ids.shape)
    if not leading_shape or node_type_ids.numel() == 0:
        raise NodeConditionedCategoricalError(
            "invalid_predicted_node_type", "node IDs must be nonempty"
        )
    for field, logits in zip(V4_RETAINED_CATEGORICAL_FIELDS, logits_tuple):
        if (
            not torch.is_tensor(logits)
            or not logits.dtype.is_floating_point
            or tuple(logits.shape) != leading_shape + (len(field.class_order),)
            or logits.device != node_type_ids.device
            or not torch.isfinite(logits).all()
        ):
            raise NodeConditionedCategoricalError(
                "invalid_v4_categorical_logits", field.name
            )
    if padding_mask is not None and (
        not torch.is_tensor(padding_mask)
        or padding_mask.dtype != torch.bool
        or tuple(padding_mask.shape) != leading_shape
        or padding_mask.device != node_type_ids.device
    ):
        raise NodeConditionedCategoricalError(
            "invalid_node_conditioned_categorical_selection",
            "padding mask is misaligned",
        )
    return leading_shape


def _validate_post_selection(selected, node_type_ids, padding_mask):
    flat_nodes = node_type_ids.reshape(-1).detach().cpu().tolist()
    flat_selected = selected.reshape(-1, selected.size(-1)).detach().cpu().tolist()
    flat_padding = padding_mask.reshape(-1).detach().cpu().tolist()
    for node_id, row, real in zip(flat_nodes, flat_selected, flat_padding):
        if not real:
            expected = [field.sentinel_id for field in V4_RETAINED_CATEGORICAL_FIELDS]
            if row != expected:
                raise NodeConditionedCategoricalError(
                    "invalid_node_conditioned_categorical_selection",
                    "padding row differs from sentinels",
                )
            continue
        from prototype.node_conditioned_categories import (
            validate_node_conditioned_categorical_row,
        )
        validate_node_conditioned_categorical_row(node_id, row)
