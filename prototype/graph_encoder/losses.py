"""One common per-example normalized loss for both GE1 encoder arms."""

from __future__ import annotations

from dataclasses import dataclass

from prototype.graph_baseline.config import GraphV1Config
from prototype.graph_baseline.losses import GraphV1Loss, graph_v1_loss

from .errors import GraphEncoderError
from .grid_magnitude import (
    GRID_MAGNITUDE_LOSS_VERSION,
    GRID_MAGNITUDE_PARAMETERIZATION,
    GRID_MAGNITUDE_REDUCTION,
    GRID_MAGNITUDE_DIAGNOSTICS_VERSION,
    GRID_MAGNITUDE_APPLICABILITY_VERSION,
    GRID_CLASS_COUNT,
    GRID_SOFTMAX_MAGNITUDE_LOSS_VERSION,
    GRID_SOFTMAX_MAGNITUDE_REDUCTION,
    GRID_SOFTMAX_MAGNITUDE_PARAMETERIZATION,
    NORMALIZED_GRIDS,
    PHYSICAL_GRIDS,
    OPERATION_NODE_TYPE_IDS,
    OPERATION_TYPES,
    ORDINAL_CUT_COUNT,
    SERIALIZED_CHANNELS,
    class_index_from_normalized_target,
)

from .config import GE1Config
from .decoder_contract import (
    COMMON_LOSS_VERSION,
    uses_grid_magnitude,
    uses_grid_softmax_magnitude,
)


GE1Loss = GraphV1Loss
LOSS_CONTRACT_VERSION = COMMON_LOSS_VERSION


@dataclass(frozen=True)
class GridMagnitudeLossTerms:
    """Structured GE1-owned ordinal loss and diagnostic tensors."""

    version: str
    reduction: dict
    raw: dict
    weighted: dict
    weights: dict
    active_operation_counts: dict
    batch_size: int
    raw_per_example: dict
    weighted_per_example: dict
    active_masks: dict
    cut_probabilities: dict
    predicted_classes: dict
    target_classes: dict
    decoded_normalized_values: dict
    decoded_physical_values: dict
    decision_margins: dict
    applicability: dict
    parameterization: str = GRID_MAGNITUDE_PARAMETERIZATION

    @property
    def uses_softmax_identity(self):
        return self.parameterization == GRID_SOFTMAX_MAGNITUDE_PARAMETERIZATION

    def diagnostic_record(self):
        """Return a detached, JSON-compatible engineering diagnostic."""

        by_type = {}
        for operation_type in OPERATION_TYPES:
            active = self.active_masks[operation_type]
            predicted = self.predicted_classes[operation_type][active]
            target = self.target_classes[operation_type][active]
            probabilities = self.cut_probabilities[operation_type][active]
            normalized = self.decoded_normalized_values[operation_type][active]
            physical = self.decoded_physical_values[operation_type][active]
            margins = self.decision_margins[operation_type][active]
            predicted_values = [
                int(value) for value in predicted.detach().cpu().tolist()
            ]
            target_values = [
                int(value) for value in target.detach().cpu().tolist()
            ]
            confusion = [
                [0] * GRID_CLASS_COUNT for _ in range(GRID_CLASS_COUNT)
            ]
            for expected, observed in zip(target_values, predicted_values):
                confusion[expected][observed] += 1
            true_counts = [sum(row) for row in confusion]
            predicted_counts = [
                sum(confusion[row][column] for row in range(5))
                for column in range(GRID_CLASS_COUNT)
            ]
            recall = [
                (
                    confusion[index][index] / float(true_counts[index])
                    if true_counts[index]
                    else None
                )
                for index in range(GRID_CLASS_COUNT)
            ]
            extreme = [
                recall[index] for index in (0, 4) if recall[index] is not None
            ]
            positions = active.nonzero(as_tuple=False).detach().cpu().tolist()
            probability_rows = probabilities.detach().cpu().tolist()
            normalized_rows = normalized.detach().cpu().tolist()
            physical_rows = physical.detach().cpu().tolist()
            margin_rows = margins.detach().cpu().tolist()
            records = []
            for index, (batch_index, node_index) in enumerate(positions):
                records.append({
                    "batch_index": int(batch_index),
                    "node_index": int(node_index),
                    "target_class": target_values[index],
                    "predicted_class": predicted_values[index],
                    "ordinal_cut_probabilities": [
                        float(value) for value in probability_rows[index]
                    ],
                    "decision_margin": float(margin_rows[index]),
                    "decoded_normalized_value": float(normalized_rows[index]),
                    "decoded_physical_value": float(physical_rows[index]),
                })
            by_type[operation_type] = {
                "active_operation_count": len(target_values),
                "raw_loss": float(
                    self.raw[operation_type].detach().cpu().item()
                ),
                "weighted_loss": float(
                    self.weighted[operation_type].detach().cpu().item()
                ),
                "weight": self.weights[operation_type],
                "confusion_matrix": confusion,
                "true_class_counts": true_counts,
                "predicted_class_counts": predicted_counts,
                "per_class_recall": recall,
                "extreme_class_recall": {
                    "class_0": recall[0],
                    "class_4": recall[4],
                    "mean_over_supported_extremes": (
                        sum(extreme) / float(len(extreme)) if extreme else None
                    ),
                },
                "zero_support_flags": [value == 0 for value in true_counts],
                "class_masking_flags": [
                    true_counts[index] > 0 and predicted_counts[index] == 0
                    for index in range(GRID_CLASS_COUNT)
                ],
                "operation_records": records,
            }
        record = {
            "schema_version": GRID_MAGNITUDE_DIAGNOSTICS_VERSION,
            "loss_version": self.version,
            "batch_size": self.batch_size,
            "active_operation_counts": self.active_operation_counts,
            "reduction": self.reduction,
            "applicability": self.applicability,
            "operation_types": by_type,
            "teacher_forced_autonomous_grid_value_agreement": {
                "available": False,
                "reason": "autonomous_values_not_supplied_to_loss_path",
            },
            "gradient_norms": {
                "available": False,
                "reason": "gradients_not_yet_computed",
            },
        }
        if self.uses_softmax_identity:
            # Added only for the new identity, so the ADR-0013 record stays
            # byte-identical.  The probability and margin fields above carry
            # identity-dependent content, so a softmax record says so.
            record["parameterization"] = self.parameterization
            record["decode_rule"] = "argmax_over_class_logits"
        return record


@dataclass(frozen=True)
class GE1GridLoss:
    """GE1-owned extension that leaves the frozen GraphV1Loss untouched."""

    base_loss: GraphV1Loss
    grid_magnitude: GridMagnitudeLossTerms
    total: object
    per_example: dict

    def __getattr__(self, name):
        return getattr(self.base_loss, name)

    def as_dict(self):
        values = dict(self.base_loss.as_dict())
        values["total"] = self.total
        for operation_type in OPERATION_TYPES:
            values["grid_magnitude_{}_raw".format(operation_type)] = (
                self.grid_magnitude.raw[operation_type]
            )
            values["grid_magnitude_{}_weighted".format(operation_type)] = (
                self.grid_magnitude.weighted[operation_type]
            )
        return values


def common_ge1_loss(
    output, target, profile_targets, config, grid_magnitude_logits=None
):
    """Use the retained Graph V1 per-example component and total assembly.

    Graph V1 already normalizes each component independently per example and
    then takes the batch mean.  C5 therefore introduces no legacy aggregation
    difference.  The continuous GE1 decoder supplies an exactly zero VQ term.
    """

    if not isinstance(config, GE1Config):
        raise TypeError("config must be GE1Config")
    config.validate()
    inherited = GraphV1Config(
        model_dim=config.model_dim,
        num_heads=config.attention_heads,
        decoder_layers=config.decoder_layers,
        max_nodes=config.max_nodes,
        max_operations=config.max_operations,
        latent_tokens=config.latent_tokens,
        codebook_dim=config.bottleneck_dim,
        dropout=config.dropout,
        node_type_loss_weight=config.node_type_loss_weight,
        categorical_loss_weight=config.categorical_loss_weight,
        geometry_loss_weight=config.geometry_loss_weight,
        profile_family_loss_weight=config.profile_family_loss_weight,
        profile_parameter_loss_weight=config.profile_parameter_loss_weight,
        graph_edge_loss_weight=config.graph_edge_loss_weight,
    )
    inherited.validate()
    grid_identity = uses_grid_magnitude(
        config.operation_magnitude_parameterization
    )
    if grid_identity and grid_magnitude_logits is None:
        raise GraphEncoderError(
            "missing_grid_magnitude_logits",
            "grid loss requires teacher-forced grid magnitude logits",
        )
    if not grid_identity and grid_magnitude_logits is not None:
        raise GraphEncoderError(
            "unexpected_grid_magnitude_logits",
            "historical magnitude identities must not receive grid logits",
        )
    scalar_target = (
        _grid_scalar_loss_target(target) if grid_identity else target
    )
    base = graph_v1_loss(output, scalar_target, profile_targets, inherited)
    if not grid_identity:
        return base
    return _with_grid_magnitude(
        base, grid_magnitude_logits, target, config
    )


def grid_magnitude_terms(grid_magnitude_logits, target, config):
    """Return raw and weighted extrusion/revolve grid magnitude losses.

    Reduction, frozen prospectively:

    1. one per-operation classification loss, which is the only step that
       differs between the two grid identities:

       - `...-GRID-ORDINAL-v1`: mean binary cross-entropy over four ordinal
         cuts (ADR-0013);
       - `...-GRID-SOFTMAX-v1`: cross-entropy over five independent class
         logits (ADR-0015);

    2. **sum** those per-operation losses within each example, per type;
    3. mean the per-example sums across the batch.

    Because step 2 sums rather than averages, the total is
    ``(1/B) * sum over all active operations``, so every active operation has
    coefficient ``1/B`` regardless of whether its example is an E, EE, R, or
    RE template.  A mean at step 2 would give an EE operation half the
    coefficient of an E operation, reproducing the dilution this term exists
    to remove.

    Axis and every other geometry channel are excluded by construction: this
    term reads only the grid head and the operation node types.
    """

    import torch

    softmax_identity = uses_grid_softmax_magnitude(
        config.operation_magnitude_parameterization
    )
    node_type_ids = target["node_type_ids"]
    node_mask = target["node_mask"]
    geometry = target["geometry"]
    geometry_mask = target["geometry_mask"]
    batch_size, node_count = node_type_ids.shape
    logit_width = GRID_CLASS_COUNT if softmax_identity else ORDINAL_CUT_COUNT
    expected_shape = (
        batch_size, node_count, len(OPERATION_TYPES), logit_width
    )
    if tuple(grid_magnitude_logits.shape) != expected_shape:
        raise GraphEncoderError(
            "invalid_grid_magnitude_logits",
            "grid magnitude logits must have shape {}".format(expected_shape),
        )
    raw = {}
    counts = {}
    raw_per_example = {}
    active_masks = {}
    cut_probabilities = {}
    predicted_classes = {}
    target_classes = {}
    decoded_normalized_values = {}
    decoded_physical_values = {}
    decision_margins = {}
    for position, operation_type in enumerate(OPERATION_TYPES):
        channel = SERIALIZED_CHANNELS[operation_type]
        type_id = OPERATION_NODE_TYPE_IDS[position]
        active = node_mask & (node_type_ids == type_id)
        if bool(active.any().item()):
            _assert_active_targets_are_on_grid(
                geometry, geometry_mask, active, channel, operation_type
            )
        logits = grid_magnitude_logits[..., position, :]
        if softmax_identity:
            labels, indices = _class_index_label_tensor(
                geometry, active, channel, operation_type, grid_magnitude_logits
            )
            # ``cross_entropy`` accepts only ``(rows, classes)``; the reshape
            # is pure bookkeeping and the result is restored to the node grid
            # before any reduction, so steps 2 and 3 are unchanged.
            flat = torch.nn.functional.cross_entropy(
                logits.reshape(-1, GRID_CLASS_COUNT),
                labels.reshape(-1),
                reduction="none",
            )
            unmasked = flat.reshape(labels.shape)
        else:
            labels, indices = _cumulative_label_tensor(
                geometry, active, channel, operation_type, grid_magnitude_logits
            )
            unmasked = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, labels, reduction="none"
            ).mean(dim=-1)
        per_operation = unmasked * active.to(unmasked.dtype)
        per_example_sum = per_operation.sum(dim=-1)
        raw[operation_type] = per_example_sum.mean()
        raw_per_example[operation_type] = per_example_sum
        counts[operation_type] = int(active.sum().item())
        active_masks[operation_type] = active
        if softmax_identity:
            # Same field, identity-appropriate content: five softmax class
            # probabilities rather than four sigmoid cut probabilities.
            cut_probabilities[operation_type] = torch.softmax(logits, dim=-1)
            predicted = logits.argmax(dim=-1)
        else:
            cut_probabilities[operation_type] = torch.sigmoid(logits)
            predicted = (logits > 0).sum(dim=-1)
        predicted_classes[operation_type] = predicted
        target_classes[operation_type] = indices
        normalized_table = torch.tensor(
            NORMALIZED_GRIDS[operation_type],
            dtype=logits.dtype,
            device=logits.device,
        )
        physical_table = torch.tensor(
            PHYSICAL_GRIDS[operation_type],
            dtype=logits.dtype,
            device=logits.device,
        )
        decoded_normalized_values[operation_type] = normalized_table[predicted]
        decoded_physical_values[operation_type] = physical_table[predicted]
        if softmax_identity:
            # Confidence of the decoded class over its nearest rival, which is
            # the softmax analogue of the ordinal head's smallest cut margin.
            top_two = torch.topk(logits, 2, dim=-1).values
            decision_margins[operation_type] = top_two[..., 0] - top_two[..., 1]
        else:
            decision_margins[operation_type] = logits.abs().min(dim=-1).values
    weights = {
        "extrude": float(config.grid_magnitude_extrude_loss_weight),
        "revolve": float(config.grid_magnitude_revolve_loss_weight),
    }
    weighted = {
        name: raw[name] * weights[name] for name in OPERATION_TYPES
    }
    weighted_per_example = {
        name: raw_per_example[name] * weights[name]
        for name in OPERATION_TYPES
    }
    return GridMagnitudeLossTerms(
        (
            GRID_SOFTMAX_MAGNITUDE_LOSS_VERSION
            if softmax_identity
            else GRID_MAGNITUDE_LOSS_VERSION
        ),
        (
            GRID_SOFTMAX_MAGNITUDE_REDUCTION
            if softmax_identity
            else GRID_MAGNITUDE_REDUCTION
        ).to_dict(),
        raw,
        weighted,
        weights,
        counts,
        batch_size,
        raw_per_example,
        weighted_per_example,
        active_masks,
        cut_probabilities,
        predicted_classes,
        target_classes,
        decoded_normalized_values,
        decoded_physical_values,
        decision_margins,
        _grid_applicability_record(target),
        config.operation_magnitude_parameterization,
    )


def _assert_active_targets_are_on_grid(
    geometry, geometry_mask, active, channel, operation_type
):
    """Reject masked-off, nonfinite, or off-grid active magnitude targets."""

    import torch

    if not bool(geometry_mask[..., channel][active].all().item()):
        raise GraphEncoderError(
            "inactive_magnitude_target",
            "an active {} node has an inactive magnitude channel".format(
                operation_type
            ),
        )
    values = geometry[..., channel][active]
    if not bool(torch.isfinite(values).all().item()):
        raise GraphEncoderError(
            "invalid_grid_target",
            "active {} magnitude targets must be finite".format(operation_type),
        )
    for value in values.detach().reshape(-1).tolist():
        class_index_from_normalized_target(operation_type, float(value))


def _cumulative_label_tensor(
    geometry, active, channel, operation_type, reference
):
    """Build detached CORAL targets; no gradient reaches label construction."""

    import torch

    grid = torch.tensor(
        NORMALIZED_GRIDS[operation_type],
        dtype=reference.dtype,
        device=reference.device,
    )
    values = geometry[..., channel].detach().to(reference.dtype)
    distance = (values.unsqueeze(-1) - grid).abs()
    indices = distance.argmin(dim=-1)
    cuts = torch.arange(
        ORDINAL_CUT_COUNT, device=reference.device
    ).reshape(*([1] * indices.dim()), ORDINAL_CUT_COUNT)
    labels = (indices.unsqueeze(-1) > cuts).to(reference.dtype)
    labels = (labels * active.unsqueeze(-1).to(reference.dtype)).detach()
    inactive = torch.full_like(indices, -1)
    indices = torch.where(active, indices, inactive).detach()
    return labels, indices


def _class_index_label_tensor(
    geometry, active, channel, operation_type, reference
):
    """Build detached integer class targets; no gradient reaches labels.

    Two tensors are returned for the same reason the ordinal builder returns
    two.  ``labels`` is a gather-safe `long` tensor whose inactive positions
    are set to class 0; those positions are zeroed by the active mask after the
    loss, so their value is never observed.  ``indices`` keeps ``-1`` at
    inactive positions and is the diagnostic view.
    """

    import torch

    grid = torch.tensor(
        NORMALIZED_GRIDS[operation_type],
        dtype=reference.dtype,
        device=reference.device,
    )
    values = geometry[..., channel].detach().to(reference.dtype)
    distance = (values.unsqueeze(-1) - grid).abs()
    indices = distance.argmin(dim=-1).to(torch.long)
    labels = torch.where(active, indices, torch.zeros_like(indices)).detach()
    inactive = torch.full_like(indices, -1)
    indices = torch.where(active, indices, inactive).detach()
    return labels, indices


def _grid_scalar_loss_target(target):
    """Return a local target view that suppresses only scalar magnitudes.

    The original target and its geometry mask remain unchanged for ordinal and
    metric applicability.  Only the Graph-V1 scalar-loss call receives this
    cloned mask, so axis channels 33..36 retain their historical semantics.
    """

    values = dict(target)
    geometry_mask = target["geometry_mask"].clone()
    for channel in SERIALIZED_CHANNELS.values():
        geometry_mask[..., channel] = False
    values["geometry_mask"] = geometry_mask
    return values


def _grid_applicability_record(target):
    node_mask = target["node_mask"]
    geometry_mask = target["geometry_mask"] & node_mask.unsqueeze(-1)
    axis_count = int(geometry_mask[..., 33:37].sum().item())
    magnitude_count = int(geometry_mask[..., 37:39].sum().item())
    return {
        "version": GRID_MAGNITUDE_APPLICABILITY_VERSION,
        "axis_target_channel_count": axis_count,
        "axis_scalar_loss_channel_count": axis_count,
        "magnitude_target_channel_count": magnitude_count,
        "magnitude_scalar_loss_channel_count": 0,
        "magnitude_ordinal_metric_channel_count": magnitude_count,
        "target_geometry_mask_mutated": False,
        "axis_semantics": "historical_supervision_and_metrics_retained",
        "magnitude_semantics": "ordinal_loss_and_grid_metrics",
    }


def _with_grid_magnitude(base, grid_magnitude_logits, target, config):
    """Add the two magnitude terms to the inherited total additively."""

    terms = grid_magnitude_terms(grid_magnitude_logits, target, config)
    added = terms.weighted["extrude"] + terms.weighted["revolve"]
    per_example = dict(base.per_example)
    per_example["total"] = (
        base.per_example["total"]
        + terms.weighted_per_example["extrude"]
        + terms.weighted_per_example["revolve"]
    )
    for operation_type in OPERATION_TYPES:
        per_example["grid_magnitude_{}_raw".format(operation_type)] = (
            terms.raw_per_example[operation_type]
        )
        per_example["grid_magnitude_{}_weighted".format(operation_type)] = (
            terms.weighted_per_example[operation_type]
        )
    return GE1GridLoss(base, terms, base.total + added, per_example)
