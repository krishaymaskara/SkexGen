"""Leakage-safe teacher-forced prediction conversion for the V4 flat model."""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from prototype.model_data.geometry import GEOMETRY_WIDTH
from prototype.model_data.vocab import (
    BOOLEAN_MODES,
    DIRECTIONS,
    EDGE_TYPES,
    LOOP_ROLES,
    NODE_TYPES,
    OPERATION_TYPES,
    REFERENCE_PLANES,
)
from prototype.profile_geometry import (
    NO_PROFILE_FAMILY_ID,
    PROFILE_FAMILIES,
)
from prototype.profile_geometry_torch import (
    canonicalize_profile_tensors,
    constrain_profile_parameters,
)
from prototype.reference_plane_geometry_torch import (
    canonicalize_reference_plane_tensors,
)
from prototype.node_conditioned_categories import (
    V4_CATEGORICAL_CONTRACT_ID,
    V4_RETAINED_CATEGORICAL_FIELDS,
    validate_node_conditioned_categorical_row,
)
from prototype.node_conditioned_categories_torch import (
    select_node_conditioned_categorical_ids,
)

from .autonomous import (
    RAW_PREFIX_FEEDBACK,
    REQUESTED_LENGTH_TERMINATION,
    TEACHER_FORCED_PREFIX_FEEDBACK,
    RawDecodedEdge,
    RawDecodedNode,
    RawDecodedPointer,
    RawDecodedPrediction,
    derive_geometry_mask,
)
from .constrained_v4 import (
    REMAINING_GEOMETRY_WIDTH,
    ConstrainedProfileV4Output,
    scatter_remaining_geometry,
)
from .conversion import (
    ConversionResult,
    validate_and_convert_raw_prediction,
)


_RETAINED_ATTRIBUTE_VOCABULARIES = (
    OPERATION_TYPES,
    BOOLEAN_MODES,
    DIRECTIONS,
    REFERENCE_PLANES,
    LOOP_ROLES,
)


class V4PredictionContractError(ValueError):
    """V4 prediction evidence is missing or internally inconsistent."""

    def __init__(self, detail):
        self.code = "invalid_v4_autonomous_evaluation_result"
        self.detail = detail
        super().__init__("{}: {}".format(self.code, detail))


@dataclass(frozen=True)
class V4PredictedProfileTensors:
    """Predicted-family-routed compact and canonical profile tensors."""

    family_ids: torch.Tensor
    constrained_parameters: torch.Tensor
    primitive_type_ids: torch.Tensor
    geometry: torch.Tensor
    geometry_mask: torch.Tensor


@dataclass(frozen=True)
class V4PredictedNodeTensors:
    """Complete canonical node records built only from V4 predictions."""

    node_type_ids: torch.Tensor
    raw_categorical_argmax_ids: torch.Tensor
    node_conditioned_categorical_ids: torch.Tensor
    categorical_correction_mask: torch.Tensor
    categorical_ids: torch.Tensor
    family_ids: torch.Tensor
    constrained_parameters: torch.Tensor
    geometry: torch.Tensor
    geometry_mask: torch.Tensor


@dataclass(frozen=True)
class V4TeacherForcedRawPrediction(RawDecodedPrediction):
    """Existing raw record plus V4-specific profile prediction evidence."""

    profile_family_logits: tuple
    predicted_profile_family_ids: tuple
    raw_profile_parameters: tuple
    constrained_profile_parameters: tuple
    raw_categorical_argmax_ids: tuple
    node_conditioned_categorical_ids: tuple
    categorical_correction_mask: tuple
    categorical_contract_id: str


def canonicalize_v4_predicted_profiles(
    family_ids: torch.Tensor,
    raw_parameters: torch.Tensor,
    sketch_mask: torch.Tensor,
) -> V4PredictedProfileTensors:
    """Constrain and canonicalize using predicted, never target, families."""

    constrained = constrain_profile_parameters(
        raw_parameters, family_ids, sketch_mask
    )
    canonical = canonicalize_profile_tensors(
        family_ids, constrained, sketch_mask
    )
    return V4PredictedProfileTensors(
        family_ids,
        constrained,
        canonical.primitive_type_ids,
        canonical.geometry,
        canonical.geometry_mask,
    )


def construct_v4_predicted_node_tensors(
    node_type_ids: torch.Tensor,
    categorical_logits: tuple,
    profile_family_logits: torch.Tensor,
    raw_profile_parameters: torch.Tensor,
    remaining_geometry: torch.Tensor,
    node_mask: torch.Tensor = None,
) -> V4PredictedNodeTensors:
    """Construct complete feedback records from current-node predictions."""

    leading_shape = _validate_predicted_node_inputs(
        node_type_ids,
        profile_family_logits,
        raw_profile_parameters,
        remaining_geometry,
    )
    if node_mask is None:
        node_mask = torch.ones_like(node_type_ids, dtype=torch.bool)
    elif (
        not torch.is_tensor(node_mask)
        or node_mask.dtype != torch.bool
        or tuple(node_mask.shape) != leading_shape
        or node_mask.device != node_type_ids.device
    ):
        raise ValueError("node_mask must align with predicted nodes")
    selection = select_node_conditioned_categorical_ids(
        categorical_logits, node_type_ids, node_mask
    )
    retained_categorical_ids = selection.node_conditioned_categorical_ids
    sketch_mask = node_mask & (node_type_ids == NODE_TYPES.id("sketch"))
    predicted_families = profile_family_logits.argmax(dim=-1)
    family_ids = torch.where(
        sketch_mask,
        predicted_families,
        torch.full_like(node_type_ids, NO_PROFILE_FAMILY_ID),
    )
    profiles = canonicalize_v4_predicted_profiles(
        family_ids, raw_profile_parameters, sketch_mask
    )
    categorical_ids = torch.cat(
        (
            retained_categorical_ids[..., :4],
            profiles.primitive_type_ids,
            retained_categorical_ids[..., 4:],
        ),
        dim=-1,
    ).contiguous()
    planes = canonicalize_reference_plane_tensors(
        node_type_ids,
        retained_categorical_ids[..., 3],
        remaining_geometry,
        node_mask,
    )
    mask_rows = tuple(
        derive_geometry_mask(int(node_type), tuple(attributes))
        for node_type, attributes in zip(
            node_type_ids.reshape(-1).detach().cpu().tolist(),
            categorical_ids.reshape(-1, 9).detach().cpu().tolist(),
        )
    )
    geometry_mask = torch.tensor(
        mask_rows,
        dtype=torch.bool,
        device=node_type_ids.device,
    ).reshape(leading_shape + (GEOMETRY_WIDTH,)).contiguous()
    geometry_mask = geometry_mask & node_mask.unsqueeze(-1)
    if not torch.equal(
        geometry_mask[..., 9:33],
        profiles.geometry_mask[..., 9:33],
    ):
        raise AssertionError(
            "canonical profile mask disagrees with primitive slots"
        )
    if not torch.equal(
        geometry_mask[..., :9], planes.geometry_mask[..., :9]
    ):
        raise AssertionError(
            "canonical plane mask disagrees with predicted node/category"
        )
    scattered = scatter_remaining_geometry(remaining_geometry)
    combined = planes.geometry + profiles.geometry + scattered
    geometry = torch.where(
        geometry_mask, combined, torch.zeros_like(combined)
    ).contiguous()
    return V4PredictedNodeTensors(
        node_type_ids,
        selection.raw_categorical_argmax_ids,
        retained_categorical_ids,
        selection.correction_mask,
        categorical_ids,
        family_ids,
        profiles.constrained_parameters,
        geometry,
        geometry_mask,
    )


def v4_teacher_forced_predictions(
    output: ConstrainedProfileV4Output,
    *,
    node_mask: torch.Tensor,
    node_count_source: str,
) -> tuple:
    """Convert aligned V4 outputs into raw teacher-forced predictions.

    ``node_mask`` supplies only real sequence lengths and padding. The API does
    not accept current-node target categories, profile families, geometry, or
    geometry masks. Every current-node content field comes from ``output``.
    """

    _validate_conversion_inputs(output, node_mask, node_count_source)
    with torch.no_grad():
        node_type_ids = output.node_type_logits.argmax(dim=-1)
        records = construct_v4_predicted_node_tensors(
            node_type_ids,
            output.categorical_logits,
            output.profile_family_logits,
            output.raw_profile_parameters,
            output.remaining_geometry,
            node_mask,
        )

        return tuple(
            _prediction_row(
                output,
                records,
                node_mask,
                batch_index,
                node_count_source,
            )
            for batch_index in range(node_mask.size(0))
        )


def validate_and_convert_v4_teacher_forced_prediction(
    prediction: V4TeacherForcedRawPrediction,
    *,
    max_operations: int,
) -> ConversionResult:
    """Run the existing strict converter through a provenance-only view."""

    if not isinstance(prediction, V4TeacherForcedRawPrediction):
        raise TypeError(
            "prediction must be a V4TeacherForcedRawPrediction"
        )
    if prediction.prefix_feedback != TEACHER_FORCED_PREFIX_FEEDBACK:
        raise ValueError(
            "V4 teacher-forced prediction has invalid prefix provenance"
        )
    _validate_v4_raw_contract(prediction)
    compatible = replace(
        prediction, prefix_feedback=RAW_PREFIX_FEEDBACK
    )
    return validate_and_convert_raw_prediction(
        compatible, max_operations=max_operations
    )


def raw_argmax_prediction_for_reporting(prediction):
    """Return a read-only raw-argmax arm; never use it as model feedback."""

    if not isinstance(prediction, V4TeacherForcedRawPrediction):
        raise TypeError("prediction must carry V4 categorical evidence")
    if len(prediction.raw_categorical_argmax_ids) != len(prediction.raw_nodes):
        raise ValueError("raw categorical evidence is misaligned")
    nodes = []
    for node, retained in zip(
        prediction.raw_nodes, prediction.raw_categorical_argmax_ids
    ):
        if len(retained) != 5:
            raise ValueError("raw categorical evidence width differs")
        categories = list(node.categorical_ids)
        for target, value in zip((0, 1, 2, 3, 8), retained):
            categories[target] = value
        nodes.append(replace(node, categorical_ids=tuple(categories)))
    return replace(
        prediction,
        raw_nodes=tuple(nodes),
        prefix_feedback=RAW_PREFIX_FEEDBACK,
    )


def _prediction_row(
    output,
    records,
    node_mask,
    batch_index,
    node_count_source,
):
    node_count = int(node_mask[batch_index].sum().item())
    raw_nodes = []
    for position in range(node_count):
        node_type_id = int(
            records.node_type_ids[batch_index, position].item()
        )
        categorical_ids = (
            tuple(
                int(value)
                for value in records.categorical_ids[
                    batch_index, position
                ].detach().cpu().tolist()
            )
        )
        geometry_mask = tuple(
            bool(value)
            for value in records.geometry_mask[
                batch_index, position
            ].detach().cpu().tolist()
        )
        raw_nodes.append(
            RawDecodedNode(
                position,
                node_type_id,
                categorical_ids,
                tuple(
                    float(value)
                    for value in records.geometry[
                        batch_index, position
                    ].detach().cpu().tolist()
                ),
                geometry_mask,
            )
        )

    raw_edges = []
    for source_index in range(node_count):
        for target_index in range(node_count):
            presence_logit = float(
                output.edge_presence_logits[
                    batch_index, source_index, target_index
                ].item()
            )
            edge_type_id = int(
                output.edge_type_logits[
                    batch_index, source_index, target_index
                ].argmax(dim=-1).item()
            )
            raw_edges.append(
                RawDecodedEdge(
                    source_index,
                    target_index,
                    presence_logit,
                    presence_logit >= 0.0,
                    edge_type_id,
                )
            )

    operation_ids = {
        NODE_TYPES.id("extrude"),
        NODE_TYPES.id("revolve"),
    }
    operation_nodes = tuple(
        node.position
        for node in raw_nodes
        if node.node_type_id in operation_ids
    )
    max_operations = output.operation_pointer_logits.size(1)
    pointers = []
    for query_index in range(
        min(len(operation_nodes), max_operations)
    ):
        logits = output.operation_pointer_logits[
            batch_index, query_index, :node_count
        ]
        selected = int(logits.argmax(dim=-1).item())
        pointers.append(
            RawDecodedPointer(
                query_index,
                selected,
                float(logits[selected].item()),
            )
        )
    return V4TeacherForcedRawPrediction(
        tuple(
            int(value)
            for value in output.code_indices[
                batch_index
            ].detach().cpu().tolist()
        ),
        node_count,
        node_count_source,
        REQUESTED_LENGTH_TERMINATION,
        False,
        TEACHER_FORCED_PREFIX_FEEDBACK,
        tuple(raw_nodes),
        tuple(raw_edges),
        operation_nodes,
        len(operation_nodes),
        len(operation_nodes) > max_operations,
        tuple(pointers),
        tuple(
            tuple(float(value) for value in row)
            for row in output.profile_family_logits[
                batch_index, :node_count
            ].detach().cpu().tolist()
        ),
        tuple(
            int(value)
            for value in records.family_ids[
                batch_index, :node_count
            ].detach().cpu().tolist()
        ),
        tuple(
            tuple(float(value) for value in row)
            for row in output.raw_profile_parameters[
                batch_index, :node_count
            ].detach().cpu().tolist()
        ),
        tuple(
            tuple(float(value) for value in row)
            for row in records.constrained_parameters[
                batch_index, :node_count
            ].detach().cpu().tolist()
        ),
        tuple(
            tuple(int(value) for value in row)
            for row in records.raw_categorical_argmax_ids[
                batch_index, :node_count
            ].detach().cpu().tolist()
        ),
        tuple(
            tuple(int(value) for value in row)
            for row in records.node_conditioned_categorical_ids[
                batch_index, :node_count
            ].detach().cpu().tolist()
        ),
        tuple(
            tuple(bool(value) for value in row)
            for row in records.categorical_correction_mask[
                batch_index, :node_count
            ].detach().cpu().tolist()
        ),
        V4_CATEGORICAL_CONTRACT_ID,
    )


def _validate_v4_raw_contract(prediction):
    if not hasattr(prediction, "categorical_contract_id"):
        raise V4PredictionContractError("categorical contract metadata is missing")
    if prediction.categorical_contract_id != V4_CATEGORICAL_CONTRACT_ID:
        raise V4PredictionContractError("categorical contract metadata is wrong")
    if (
        not hasattr(prediction, "node_count")
        or isinstance(prediction.node_count, bool)
        or not isinstance(prediction.node_count, int)
        or prediction.node_count <= 0
    ):
        raise V4PredictionContractError("node_count must be a positive integer")
    count = prediction.node_count
    evidence = (
        prediction.raw_categorical_argmax_ids,
        prediction.node_conditioned_categorical_ids,
        prediction.categorical_correction_mask,
    )
    if not hasattr(prediction, "raw_nodes") or len(prediction.raw_nodes) != count:
        raise V4PredictionContractError("generated nodes disagree with node_count")
    if any(len(value) != count for value in evidence):
        raise V4PredictionContractError("categorical evidence is misaligned")
    for index, node in enumerate(prediction.raw_nodes):
        retained = (
            node.categorical_ids[0], node.categorical_ids[1],
            node.categorical_ids[2], node.categorical_ids[3],
            node.categorical_ids[8],
        )
        constrained = prediction.node_conditioned_categorical_ids[index]
        if retained != constrained:
            raise V4PredictionContractError(
                "raw node does not contain constrained categories"
            )
        validate_node_conditioned_categorical_row(node.node_type_id, retained)
        raw = prediction.raw_categorical_argmax_ids[index]
        corrections = prediction.categorical_correction_mask[index]
        if len(raw) != 5 or len(constrained) != 5 or len(corrections) != 5:
            raise V4PredictionContractError("categorical evidence width differs")
        if tuple(a != b for a, b in zip(raw, constrained)) != tuple(corrections):
            raise V4PredictionContractError(
                "categorical correction evidence differs"
            )


def _validate_predicted_node_inputs(
    node_type_ids,
    profile_family_logits,
    raw_profile_parameters,
    remaining_geometry,
):
    if not torch.is_tensor(node_type_ids) or node_type_ids.dtype != torch.long:
        raise TypeError("node_type_ids must be a torch.long tensor")
    if node_type_ids.dim() < 1 or node_type_ids.numel() == 0:
        raise ValueError("node_type_ids must have nonempty leading dimensions")
    leading_shape = tuple(node_type_ids.shape)
    floats = (
        (
            "profile_family_logits",
            profile_family_logits,
            leading_shape + (len(PROFILE_FAMILIES),),
        ),
        (
            "raw_profile_parameters",
            raw_profile_parameters,
            leading_shape + (3,),
        ),
        (
            "remaining_geometry",
            remaining_geometry,
            leading_shape + (REMAINING_GEOMETRY_WIDTH,),
        ),
    )
    reference_dtype = (
        raw_profile_parameters.dtype
        if torch.is_tensor(raw_profile_parameters)
        else None
    )
    for name, value, expected_shape in floats:
        if not torch.is_tensor(value) or not value.dtype.is_floating_point:
            raise TypeError("{} must be floating point".format(name))
        if tuple(value.shape) != expected_shape:
            raise ValueError("{} is misaligned".format(name))
        if value.dtype != reference_dtype:
            raise TypeError("predicted floating tensors must share dtype")
        if value.device != node_type_ids.device:
            raise ValueError("predicted node tensors must share device")
        if not torch.isfinite(value).all():
            raise ValueError("{} must be finite".format(name))
    return leading_shape


def _validate_conversion_inputs(output, node_mask, node_count_source):
    if not isinstance(output, ConstrainedProfileV4Output):
        raise TypeError("output must be ConstrainedProfileV4Output")
    if not torch.is_tensor(node_mask) or node_mask.dtype != torch.bool:
        raise TypeError("node_mask must be a torch.bool tensor")
    if node_mask.dim() != 2 or node_mask.size(0) == 0:
        raise ValueError("node_mask must have shape [B, N]")
    if not isinstance(node_count_source, str) or not node_count_source:
        raise ValueError("node_count_source must be a nonempty string")

    batch_size, node_count = node_mask.shape
    expected_nodes = (batch_size, node_count)
    float_fields = (
        (
            "node_type_logits",
            output.node_type_logits,
            expected_nodes + (len(NODE_TYPES.tokens),),
        ),
        (
            "profile_family_logits",
            output.profile_family_logits,
            expected_nodes + (len(PROFILE_FAMILIES),),
        ),
        (
            "raw_profile_parameters",
            output.raw_profile_parameters,
            expected_nodes + (3,),
        ),
        (
            "remaining_geometry",
            output.remaining_geometry,
            expected_nodes + (REMAINING_GEOMETRY_WIDTH,),
        ),
    )
    for name, value, expected_shape in float_fields:
        if (
            not torch.is_tensor(value)
            or not value.dtype.is_floating_point
        ):
            raise TypeError("{} must be floating point".format(name))
        if tuple(value.shape) != expected_shape:
            raise ValueError("{} is misaligned".format(name))
        if value.device != node_mask.device:
            raise ValueError("{} must share node_mask device".format(name))
        if not torch.isfinite(value).all():
            raise ValueError("{} must be finite".format(name))

    if (
        not isinstance(output.categorical_logits, tuple)
        or len(output.categorical_logits)
        != len(_RETAINED_ATTRIBUTE_VOCABULARIES)
    ):
        raise ValueError("V4 must expose five categorical logits")
    for logits, vocabulary in zip(
        output.categorical_logits,
        _RETAINED_ATTRIBUTE_VOCABULARIES,
    ):
        if (
            not torch.is_tensor(logits)
            or not logits.dtype.is_floating_point
        ):
            raise TypeError("categorical logits must be floating point")
        if logits.shape != expected_nodes + (len(vocabulary.tokens),):
            raise ValueError("categorical logits are misaligned")
        if logits.device != node_mask.device:
            raise ValueError("categorical logits must share node_mask device")
        if not torch.isfinite(logits).all():
            raise ValueError("categorical logits must be finite")

    pointer_logits = output.operation_pointer_logits
    if (
        not torch.is_tensor(pointer_logits)
        or not pointer_logits.dtype.is_floating_point
        or pointer_logits.dim() != 3
    ):
        raise TypeError(
            "operation_pointer_logits must be a rank-three floating tensor"
        )
    relation_fields = (
        (
            "edge_presence_logits",
            output.edge_presence_logits,
            expected_nodes + (node_count,),
        ),
        (
            "edge_type_logits",
            output.edge_type_logits,
            expected_nodes
            + (node_count, len(EDGE_TYPES.tokens)),
        ),
        (
            "operation_pointer_logits",
            pointer_logits,
            (
                batch_size,
                pointer_logits.size(1),
                node_count,
            ),
        ),
    )
    for name, value, expected in relation_fields:
        if (
            not torch.is_tensor(value)
            or not value.dtype.is_floating_point
        ):
            raise TypeError("{} must be floating point".format(name))
        if value.shape != expected:
            raise ValueError("{} is misaligned".format(name))
        if value.device != node_mask.device:
            raise ValueError("{} must share node_mask device".format(name))
        if not torch.isfinite(value).all():
            raise ValueError("{} must be finite".format(name))
    if pointer_logits.size(1) == 0:
        raise ValueError("operation pointer logits require at least one query")
    if (
        not torch.is_tensor(output.code_indices)
        or output.code_indices.dtype != torch.long
        or output.code_indices.dim() != 2
        or output.code_indices.size(0) != batch_size
        or output.code_indices.device != node_mask.device
    ):
        raise ValueError("code_indices must have shape [B, latent_tokens]")

    for row in node_mask:
        count = int(row.sum().item())
        if (
            count <= 0
            or not bool(row[:count].all().item())
            or bool(row[count:].any().item())
        ):
            raise ValueError(
                "node_mask rows must be positive contiguous prefixes"
            )
