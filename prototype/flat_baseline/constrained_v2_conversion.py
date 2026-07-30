"""Leakage-safe teacher-forced prediction conversion for the V2 flat model."""

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
from .constrained_v2 import (
    NON_PROFILE_GEOMETRY_WIDTH,
    ConstrainedProfileV2Output,
    scatter_non_profile_geometry,
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


@dataclass(frozen=True)
class V2PredictedProfileTensors:
    """Predicted-family-routed compact and canonical profile tensors."""

    family_ids: torch.Tensor
    constrained_parameters: torch.Tensor
    primitive_type_ids: torch.Tensor
    geometry: torch.Tensor
    geometry_mask: torch.Tensor


@dataclass(frozen=True)
class V2PredictedNodeTensors:
    """Complete canonical node records built only from V2 predictions."""

    node_type_ids: torch.Tensor
    categorical_ids: torch.Tensor
    family_ids: torch.Tensor
    constrained_parameters: torch.Tensor
    geometry: torch.Tensor
    geometry_mask: torch.Tensor


@dataclass(frozen=True)
class V2TeacherForcedRawPrediction(RawDecodedPrediction):
    """Existing raw record plus V2-specific profile prediction evidence."""

    profile_family_logits: tuple
    predicted_profile_family_ids: tuple
    raw_profile_parameters: tuple
    constrained_profile_parameters: tuple


def canonicalize_v2_predicted_profiles(
    family_ids: torch.Tensor,
    raw_parameters: torch.Tensor,
    sketch_mask: torch.Tensor,
) -> V2PredictedProfileTensors:
    """Constrain and canonicalize using predicted, never target, families."""

    constrained = constrain_profile_parameters(
        raw_parameters, family_ids, sketch_mask
    )
    canonical = canonicalize_profile_tensors(
        family_ids, constrained, sketch_mask
    )
    return V2PredictedProfileTensors(
        family_ids,
        constrained,
        canonical.primitive_type_ids,
        canonical.geometry,
        canonical.geometry_mask,
    )


def construct_v2_predicted_node_tensors(
    node_type_ids: torch.Tensor,
    retained_categorical_ids: torch.Tensor,
    profile_family_logits: torch.Tensor,
    raw_profile_parameters: torch.Tensor,
    non_profile_geometry: torch.Tensor,
) -> V2PredictedNodeTensors:
    """Construct complete feedback records from current-node predictions."""

    leading_shape = _validate_predicted_node_inputs(
        node_type_ids,
        retained_categorical_ids,
        profile_family_logits,
        raw_profile_parameters,
        non_profile_geometry,
    )
    sketch_mask = node_type_ids == NODE_TYPES.id("sketch")
    predicted_families = profile_family_logits.argmax(dim=-1)
    family_ids = torch.where(
        sketch_mask,
        predicted_families,
        torch.full_like(node_type_ids, NO_PROFILE_FAMILY_ID),
    )
    profiles = canonicalize_v2_predicted_profiles(
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
    if not torch.equal(
        geometry_mask[..., 9:33],
        profiles.geometry_mask[..., 9:33],
    ):
        raise AssertionError(
            "canonical profile mask disagrees with primitive slots"
        )
    scattered = scatter_non_profile_geometry(non_profile_geometry)
    combined = scattered + profiles.geometry
    geometry = torch.where(
        geometry_mask, combined, torch.zeros_like(combined)
    ).contiguous()
    return V2PredictedNodeTensors(
        node_type_ids,
        categorical_ids,
        family_ids,
        profiles.constrained_parameters,
        geometry,
        geometry_mask,
    )


def v2_teacher_forced_predictions(
    output: ConstrainedProfileV2Output,
    *,
    node_mask: torch.Tensor,
    node_count_source: str,
) -> tuple:
    """Convert aligned V2 outputs into raw teacher-forced predictions.

    ``node_mask`` supplies only real sequence lengths and padding. The API does
    not accept current-node target categories, profile families, geometry, or
    geometry masks. Every current-node content field comes from ``output``.
    """

    _validate_conversion_inputs(output, node_mask, node_count_source)
    with torch.no_grad():
        node_type_ids = output.node_type_logits.argmax(dim=-1)
        retained_ids = torch.stack(
            tuple(logits.argmax(dim=-1) for logits in output.categorical_logits),
            dim=-1,
        )
        records = construct_v2_predicted_node_tensors(
            node_type_ids,
            retained_ids,
            output.profile_family_logits,
            output.raw_profile_parameters,
            output.non_profile_geometry,
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


def validate_and_convert_v2_teacher_forced_prediction(
    prediction: V2TeacherForcedRawPrediction,
    *,
    max_operations: int,
) -> ConversionResult:
    """Run the existing strict converter through a provenance-only view."""

    if not isinstance(prediction, V2TeacherForcedRawPrediction):
        raise TypeError(
            "prediction must be a V2TeacherForcedRawPrediction"
        )
    if prediction.prefix_feedback != TEACHER_FORCED_PREFIX_FEEDBACK:
        raise ValueError(
            "V2 teacher-forced prediction has invalid prefix provenance"
        )
    compatible = replace(
        prediction, prefix_feedback=RAW_PREFIX_FEEDBACK
    )
    return validate_and_convert_raw_prediction(
        compatible, max_operations=max_operations
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
    return V2TeacherForcedRawPrediction(
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
    )


def _validate_predicted_node_inputs(
    node_type_ids,
    retained_categorical_ids,
    profile_family_logits,
    raw_profile_parameters,
    non_profile_geometry,
):
    if not torch.is_tensor(node_type_ids) or node_type_ids.dtype != torch.long:
        raise TypeError("node_type_ids must be a torch.long tensor")
    if node_type_ids.dim() < 1 or node_type_ids.numel() == 0:
        raise ValueError("node_type_ids must have nonempty leading dimensions")
    leading_shape = tuple(node_type_ids.shape)
    if (
        not torch.is_tensor(retained_categorical_ids)
        or retained_categorical_ids.dtype != torch.long
        or tuple(retained_categorical_ids.shape) != leading_shape + (5,)
    ):
        raise ValueError(
            "retained_categorical_ids must have shape [..., 5]"
        )
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
            "non_profile_geometry",
            non_profile_geometry,
            leading_shape + (NON_PROFILE_GEOMETRY_WIDTH,),
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
    if retained_categorical_ids.device != node_type_ids.device:
        raise ValueError("predicted node tensors must share device")
    id_fields = ((node_type_ids, NODE_TYPES),) + tuple(
        (retained_categorical_ids[..., index], vocabulary)
        for index, vocabulary in enumerate(
            _RETAINED_ATTRIBUTE_VOCABULARIES
        )
    )
    for values, vocabulary in id_fields:
        if (
            int(values.min().item()) < 0
            or int(values.max().item()) >= len(vocabulary.tokens)
        ):
            raise ValueError(
                "{} contains an invalid ID".format(vocabulary.name)
            )
    return leading_shape


def _validate_conversion_inputs(output, node_mask, node_count_source):
    if not isinstance(output, ConstrainedProfileV2Output):
        raise TypeError("output must be ConstrainedProfileV2Output")
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
            "non_profile_geometry",
            output.non_profile_geometry,
            expected_nodes + (NON_PROFILE_GEOMETRY_WIDTH,),
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
        raise ValueError("V2 must expose five categorical logits")
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
