"""Grid-anchored operation-magnitude contracts and heads.

The controlled corpus emits operation magnitudes on two frozen five-value
grids.  The historical scalar parameterizations regress a continuous value and,
under a squared-error-shaped objective, settle on the conditional mean between
grid points.  This module replaces that readout with a classifier whose decoded
value is exactly a frozen grid member.

Two such classifier identities live here side by side.

``GE1-OPERATION-MAGNITUDE-GRID-ORDINAL-v1`` (ADR-0013) is a rank-consistent
CORAL head: one shared scalar projection per grid compared against four
strictly decreasing thresholds.  Corpus-free trajectory job ``3353008`` found
that its classes 2-4 never decoded above class 1, because a single shared
scalar cannot move class evidence independently and the upper cuts are jointly
unreachable when the projection's dynamic range is small relative to the bias
gaps.  That identity is retained here unchanged: it is immutable historical
evidence and remains runnable.

``GE1-OPERATION-MAGNITUDE-GRID-SOFTMAX-v1`` (ADR-0015) is the repair: one
``Linear(model_dim, 5)`` per grid, independent per-class logits, argmax decode,
and cross-entropy.  Every class owns a distinct weight row and bias, so class
evidence moves independently by construction.

Everything above the head builders is deliberately free of PyTorch so the grid,
label, validation, and decode contracts can be tested without a runtime.  Both
heads are imported lazily by callers that already require torch.
"""

from __future__ import annotations

from dataclasses import dataclass

from prototype.model_data.geometry import GEOMETRY_CHANNEL_SCALES
from prototype.model_data.vocab import NODE_TYPES

from .errors import GraphEncoderError


GRID_MAGNITUDE_PARAMETERIZATION = "GE1-OPERATION-MAGNITUDE-GRID-ORDINAL-v1"
GRID_MAGNITUDE_CONTRACT_VERSION = "GE1-GRID-MAGNITUDE-CONTRACT-v1"
GRID_MAGNITUDE_LOSS_VERSION = "GE1-GRID-MAGNITUDE-LOSS-v1"
# ADR-0015 replaces the shared-scalar CORAL readout with independent per-class
# logits.  It is a separate identity: the ordinal literals above and every
# ordinal code path below stay exactly as ADR-0013 froze them.
GRID_SOFTMAX_MAGNITUDE_PARAMETERIZATION = (
    "GE1-OPERATION-MAGNITUDE-GRID-SOFTMAX-v1"
)
GRID_SOFTMAX_MAGNITUDE_CONTRACT_VERSION = (
    "GE1-GRID-SOFTMAX-MAGNITUDE-CONTRACT-v1"
)
GRID_SOFTMAX_MAGNITUDE_LOSS_VERSION = "GE1-GRID-SOFTMAX-MAGNITUDE-LOSS-v1"
# Appended, never reordered, so no existing index or prefix moves.
GRID_MAGNITUDE_PARAMETERIZATIONS = (
    GRID_MAGNITUDE_PARAMETERIZATION,
    GRID_SOFTMAX_MAGNITUDE_PARAMETERIZATION,
)
GRID_MAGNITUDE_DIAGNOSTICS_VERSION = "GE1-GRID-MAGNITUDE-DIAGNOSTICS-v1"
GRID_MAGNITUDE_APPLICABILITY_VERSION = "GE1-GRID-MAGNITUDE-APPLICABILITY-v1"
GRID_MAGNITUDE_GRADIENT_VERSION = "GE1-GRID-MAGNITUDE-GRADIENTS-v1"

# Frozen physical grids.  These mirror `operation_fidelity` rather than
# redefining them; the equality is asserted at import time below so the two
# contracts can never drift apart silently.
EXTRUSION_PHYSICAL_GRID = (0.5, 1.0, 1.5, 2.0, 3.0)
REVOLVE_PHYSICAL_GRID = (45.0, 90.0, 180.0, 270.0, 360.0)

OPERATION_TYPES = ("extrude", "revolve")
GRID_CLASS_COUNT = 5
ORDINAL_CUT_COUNT = GRID_CLASS_COUNT - 1

# Serialized geometry channels and their scale-only physical conversion.
EXTRUSION_SERIALIZED_CHANNEL = 37
REVOLVE_SERIALIZED_CHANNEL = 38
OPERATION_SERIALIZED_CHANNELS = (
    EXTRUSION_SERIALIZED_CHANNEL,
    REVOLVE_SERIALIZED_CHANNEL,
)
# Compact indices inside the six-channel remaining-geometry vector (33..38).
EXTRUSION_COMPACT_CHANNEL = 4
REVOLVE_COMPACT_CHANNEL = 5
OPERATION_COMPACT_CHANNELS = (
    EXTRUSION_COMPACT_CHANNEL,
    REVOLVE_COMPACT_CHANNEL,
)

# Exact-match tolerance for deriving a class label from a normalized target.
# Frozen prospectively; it is a data-integrity tolerance, not a tuned value.
GRID_LABEL_TOLERANCE = 1e-6


def _normalized_grid(physical_grid, channel):
    scale = float(GEOMETRY_CHANNEL_SCALES[channel])
    return tuple(float(value) / scale for value in physical_grid)


EXTRUSION_NORMALIZED_GRID = _normalized_grid(
    EXTRUSION_PHYSICAL_GRID, EXTRUSION_SERIALIZED_CHANNEL
)
REVOLVE_NORMALIZED_GRID = _normalized_grid(
    REVOLVE_PHYSICAL_GRID, REVOLVE_SERIALIZED_CHANNEL
)

PHYSICAL_GRIDS = {
    "extrude": EXTRUSION_PHYSICAL_GRID,
    "revolve": REVOLVE_PHYSICAL_GRID,
}
NORMALIZED_GRIDS = {
    "extrude": EXTRUSION_NORMALIZED_GRID,
    "revolve": REVOLVE_NORMALIZED_GRID,
}
SERIALIZED_CHANNELS = {
    "extrude": EXTRUSION_SERIALIZED_CHANNEL,
    "revolve": REVOLVE_SERIALIZED_CHANNEL,
}
COMPACT_CHANNELS = {
    "extrude": EXTRUSION_COMPACT_CHANNEL,
    "revolve": REVOLVE_COMPACT_CHANNEL,
}

_NODE_TYPE_IDS = {
    "extrude": NODE_TYPES.id("extrude"),
    "revolve": NODE_TYPES.id("revolve"),
}
OPERATION_NODE_TYPE_IDS = tuple(_NODE_TYPE_IDS[name] for name in OPERATION_TYPES)


def _assert_grid_agreement():
    """Fail at import if the fidelity gate and this contract ever diverge."""

    from .operation_fidelity import EXTRUSION_GRID, REVOLVE_GRID

    if tuple(float(value) for value in EXTRUSION_GRID) != EXTRUSION_PHYSICAL_GRID:
        raise GraphEncoderError(
            "grid_contract_divergence",
            "extrusion grid differs from the frozen fidelity grid",
        )
    if tuple(float(value) for value in REVOLVE_GRID) != REVOLVE_PHYSICAL_GRID:
        raise GraphEncoderError(
            "grid_contract_divergence",
            "revolve grid differs from the frozen fidelity grid",
        )


_assert_grid_agreement()


def grid_contract_metadata(parameterization=None):
    """Return the deterministic JSON-compatible frozen grid contract.

    Called with no argument or with the ordinal identity this returns the
    ADR-0013 record byte-for-byte, including ``ordinal_cut_count``.  The
    softmax identity replaces the decode and rank-consistency strings and drops
    the cut count, which has no meaning for independent class logits.
    """

    identity = (
        GRID_MAGNITUDE_PARAMETERIZATION
        if parameterization is None
        else parameterization
    )
    if identity not in GRID_MAGNITUDE_PARAMETERIZATIONS:
        raise GraphEncoderError(
            "invalid_grid_parameterization",
            "unknown grid operation-magnitude parameterization",
        )
    softmax = identity == GRID_SOFTMAX_MAGNITUDE_PARAMETERIZATION
    record = {
        "version": (
            GRID_SOFTMAX_MAGNITUDE_CONTRACT_VERSION
            if softmax
            else GRID_MAGNITUDE_CONTRACT_VERSION
        ),
        "parameterization": identity,
        "class_count": GRID_CLASS_COUNT,
        "ordinal_cut_count": ORDINAL_CUT_COUNT,
        "label_tolerance": GRID_LABEL_TOLERANCE,
        "residual_implemented": False,
        "class_balancing_applied": False,
        "operation_types": list(OPERATION_TYPES),
        "grids": {
            name: {
                "physical": list(PHYSICAL_GRIDS[name]),
                "normalized": list(NORMALIZED_GRIDS[name]),
                "serialized_channel": SERIALIZED_CHANNELS[name],
                "compact_channel": COMPACT_CHANNELS[name],
                "normalization_scale": float(
                    GEOMETRY_CHANNEL_SCALES[SERIALIZED_CHANNELS[name]]
                ),
                "node_type_id": _NODE_TYPE_IDS[name],
            }
            for name in OPERATION_TYPES
        },
        "rank_consistency": "structural_shared_weight_with_decreasing_biases",
        "decode_rule": "count_of_cumulative_logits_greater_than_zero",
        "target_labels_used_only_inside_loss": True,
    }
    if softmax:
        del record["ordinal_cut_count"]
        record["rank_consistency"] = "none_independent_class_logits"
        record["decode_rule"] = "argmax_over_class_logits"
    return record


def _validate_operation_type(operation_type):
    if operation_type not in OPERATION_TYPES:
        raise GraphEncoderError(
            "invalid_grid_operation_type",
            "operation type must be extrude or revolve",
        )
    return operation_type


def operation_type_for_node_type_id(node_type_id):
    """Map an autonomously generated node type onto its grid, or None."""

    for name in OPERATION_TYPES:
        if node_type_id == _NODE_TYPE_IDS[name]:
            return name
    return None


def normalized_grid_value(operation_type, class_index):
    """Return the exact frozen normalized value for one class index."""

    _validate_operation_type(operation_type)
    _validate_class_index(class_index)
    return NORMALIZED_GRIDS[operation_type][class_index]


def physical_grid_value(operation_type, class_index):
    """Return the exact frozen physical value for one class index."""

    _validate_operation_type(operation_type)
    _validate_class_index(class_index)
    return PHYSICAL_GRIDS[operation_type][class_index]


def _validate_class_index(class_index):
    if (
        isinstance(class_index, bool)
        or not isinstance(class_index, int)
        or not 0 <= class_index < GRID_CLASS_COUNT
    ):
        raise GraphEncoderError(
            "invalid_grid_class_index",
            "class index must be an integer in [0, 5)",
        )
    return class_index


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and (
        value == value and value not in (float("inf"), float("-inf"))
    )


def class_index_from_normalized_target(operation_type, value):
    """Map one normalized target onto its frozen class index.

    Off-grid, nonfinite, malformed, or ambiguous values raise.  Nothing is
    snapped: an unexpected value indicates a data or masking defect and must
    surface rather than be silently absorbed.
    """

    _validate_operation_type(operation_type)
    if not _finite(value):
        raise GraphEncoderError(
            "invalid_grid_target", "normalized magnitude target must be finite"
        )
    grid = NORMALIZED_GRIDS[operation_type]
    matches = [
        index
        for index, candidate in enumerate(grid)
        if abs(float(value) - candidate) <= GRID_LABEL_TOLERANCE
    ]
    if len(matches) != 1:
        raise GraphEncoderError(
            "off_grid_magnitude_target",
            "normalized target {!r} is not exactly one {} grid member".format(
                value, operation_type
            ),
        )
    return matches[0]


def cumulative_labels(class_index):
    """Return the four CORAL cumulative targets for one class index.

    Cut ``k`` answers "is the class index greater than ``k``", so the label
    vector is a non-increasing prefix of ones whose sum is the class index.
    """

    _validate_class_index(class_index)
    return tuple(
        1.0 if class_index > cut else 0.0 for cut in range(ORDINAL_CUT_COUNT)
    )


def class_index_from_cumulative_flags(flags):
    """Decode a class index from four ordered cumulative decisions."""

    values = tuple(flags)
    if len(values) != ORDINAL_CUT_COUNT or any(
        not isinstance(item, bool) for item in values
    ):
        raise GraphEncoderError(
            "invalid_ordinal_decision",
            "ordinal decode requires four Boolean cumulative decisions",
        )
    # Structural rank consistency makes the satisfied set a prefix; assert it
    # so a violation is a loud contract failure rather than a silent miscount.
    seen_false = False
    for item in values:
        if seen_false and item:
            raise GraphEncoderError(
                "rank_inconsistent_ordinal_decision",
                "cumulative decisions must be a non-increasing prefix",
            )
        if not item:
            seen_false = True
    return sum(1 for item in values if item)


@dataclass(frozen=True)
class GridMagnitudeReduction:
    """The frozen per-operation-equal reduction, recorded for the artifact."""

    version: str = GRID_MAGNITUDE_LOSS_VERSION
    per_operation: str = "mean_binary_cross_entropy_over_four_ordinal_cuts"
    within_example: str = "sum_over_active_operations_of_that_type"
    across_batch: str = "mean_over_examples"
    per_operation_coefficient: str = "1/batch_size_for_every_active_operation"
    equal_contribution_proof: str = (
        "L = (1/B) * sum_e sum_{i in e} L_i = (1/B) * sum_i L_i, so "
        "dL/dL_i = 1/B for every active operation independently of whether its "
        "example is an E, EE, R, or RE template"
    )
    axis_channels_included: bool = False
    class_balancing_applied: bool = False

    def to_dict(self):
        return {
            "version": self.version,
            "per_operation": self.per_operation,
            "within_example": self.within_example,
            "across_batch": self.across_batch,
            "per_operation_coefficient": self.per_operation_coefficient,
            "equal_contribution_proof": self.equal_contribution_proof,
            "axis_channels_included": self.axis_channels_included,
            "class_balancing_applied": self.class_balancing_applied,
        }


GRID_MAGNITUDE_REDUCTION = GridMagnitudeReduction()
# ADR-0015 changes only step 1 of the reduction.  Steps 2-4, the equal
# per-operation contribution proof, the axis exclusion, and the absence of
# class balancing are identical to ADR-0013 by construction: the softmax record
# overrides exactly two fields.
GRID_SOFTMAX_MAGNITUDE_REDUCTION = GridMagnitudeReduction(
    version=GRID_SOFTMAX_MAGNITUDE_LOSS_VERSION,
    per_operation="cross_entropy_over_five_independent_class_logits",
)


def _require_torch():
    try:
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover - exercised on Adroit only
        raise RuntimeError(
            "grid-magnitude head construction requires PyTorch"
        ) from exc
    return torch, nn


def build_grid_magnitude_head(model_dim):
    """Construct the rank-consistent ordinal head (requires PyTorch)."""

    torch, nn = _require_torch()

    class GridMagnitudeHead(nn.Module):
        """Two independent CORAL groups with structural rank consistency.

        Each grid owns one shared projection and four biases that are strictly
        decreasing by construction, so the cumulative logits are non-increasing
        for every input and rank consistency cannot be violated numerically.
        """

        contract_version = GRID_MAGNITUDE_CONTRACT_VERSION
        parameterization = GRID_MAGNITUDE_PARAMETERIZATION

        def __init__(self, width):
            super().__init__()
            if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
                raise ValueError("model_dim must be a positive integer")
            self.model_dim = int(width)
            self.operation_types = OPERATION_TYPES
            # One projection per grid; deliberately not shared between grids so
            # extrusion and revolve keep disjoint parameter groups.
            self.projections = nn.ModuleList(
                nn.Linear(self.model_dim, 1, bias=False)
                for _ in OPERATION_TYPES
            )
            self.first_bias = nn.Parameter(torch.zeros(len(OPERATION_TYPES)))
            self.bias_gaps = nn.Parameter(
                torch.zeros(len(OPERATION_TYPES), ORDINAL_CUT_COUNT - 1)
            )
            # Keep ``nn.Linear``'s seeded nonzero initialization.  A zero
            # projection would give the ordinal loss zero derivative with
            # respect to decoded states on the first real training step and
            # would therefore violate the required head-to-trunk connection.

        def ordered_biases(self):
            """Return strictly decreasing biases ``[types, cuts]``."""

            gaps = nn.functional.softplus(self.bias_gaps)
            cumulative = torch.cumsum(gaps, dim=-1)
            first = self.first_bias.unsqueeze(-1)
            return torch.cat((first, first - cumulative), dim=-1)

        def forward(self, decoded_states):
            """Return cumulative logits ``[..., types, cuts]``."""

            if decoded_states.size(-1) != self.model_dim:
                raise ValueError("decoded state width differs from model_dim")
            projected = torch.cat(
                [projection(decoded_states) for projection in self.projections],
                dim=-1,
            ).unsqueeze(-1)
            return projected + self.ordered_biases()

        def cumulative_probabilities(self, decoded_states):
            return torch.sigmoid(self.forward(decoded_states))

        def class_indices(self, cumulative_logits):
            """Decode class indices by counting positive cumulative logits."""

            return (cumulative_logits > 0).sum(dim=-1)

        def normalized_values(self, cumulative_logits):
            """Return exact frozen normalized grid values ``[..., types]``.

            The decode is discrete, so no gradient flows through the geometry
            path; the head learns only through the ordinal loss.
            """

            indices = self.class_indices(cumulative_logits)
            table = torch.tensor(
                [NORMALIZED_GRIDS[name] for name in OPERATION_TYPES],
                dtype=cumulative_logits.dtype,
                device=cumulative_logits.device,
            )
            gathered = []
            for position in range(len(OPERATION_TYPES)):
                row = table[position]
                gathered.append(row[indices[..., position]])
            return torch.stack(gathered, dim=-1)

    return GridMagnitudeHead(model_dim)


def build_grid_softmax_magnitude_head(model_dim):
    """Construct the independent five-way class head (requires PyTorch).

    This is the ADR-0015 repair for the failure trajectory job ``3353008``
    measured on the ordinal head above.  There is no shared scalar, no
    cumulative cut, no bias ordering, and no structural rank consistency: each
    of the five classes owns a distinct weight row and bias, so class evidence
    moves independently and no class is reachable only by first crossing
    another class's threshold.
    """

    torch, nn = _require_torch()

    class GridSoftmaxMagnitudeHead(nn.Module):
        """Two independent five-way classifiers, one per frozen grid."""

        contract_version = GRID_SOFTMAX_MAGNITUDE_CONTRACT_VERSION
        parameterization = GRID_SOFTMAX_MAGNITUDE_PARAMETERIZATION

        def __init__(self, width):
            super().__init__()
            if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
                raise ValueError("model_dim must be a positive integer")
            self.model_dim = int(width)
            self.operation_types = OPERATION_TYPES
            # One classifier per grid; deliberately not shared between grids so
            # extrusion and revolve keep disjoint parameter groups, matching the
            # ordinal head's isolation.
            self.projections = nn.ModuleList(
                nn.Linear(self.model_dim, GRID_CLASS_COUNT)
                for _ in OPERATION_TYPES
            )
            # Keep ``nn.Linear``'s seeded nonzero initialization.  A zero
            # projection would give the classification loss zero derivative with
            # respect to decoded states on the first real training step and
            # would therefore violate the required head-to-trunk connection.

        def forward(self, decoded_states):
            """Return independent class logits ``[..., types, classes]``."""

            if decoded_states.size(-1) != self.model_dim:
                raise ValueError("decoded state width differs from model_dim")
            return torch.stack(
                [projection(decoded_states) for projection in self.projections],
                dim=-2,
            )

        def class_probabilities(self, class_logits):
            return torch.softmax(class_logits, dim=-1)

        def class_indices(self, class_logits):
            """Decode class indices as the argmax over independent logits."""

            return class_logits.argmax(dim=-1)

        def normalized_values(self, class_logits):
            """Return exact frozen normalized grid values ``[..., types]``.

            The decode is discrete, so no gradient flows through the geometry
            path; the head learns only through the classification loss.
            """

            indices = self.class_indices(class_logits)
            table = torch.tensor(
                [NORMALIZED_GRIDS[name] for name in OPERATION_TYPES],
                dtype=class_logits.dtype,
                device=class_logits.device,
            )
            gathered = []
            for position in range(len(OPERATION_TYPES)):
                row = table[position]
                gathered.append(row[indices[..., position]])
            return torch.stack(gathered, dim=-1)

    return GridSoftmaxMagnitudeHead(model_dim)
