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
    PRIMITIVE_TYPES,
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
    NON_PROFILE_GEOMETRY_INDICES,
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
        predicted_sketch_mask = (
            node_mask
            & (node_type_ids == NODE_TYPES.id("sketch"))
        )
        predicted_family_ids = torch.where(
            predicted_sketch_mask,
            output.profile_family_logits.argmax(dim=-1),
            torch.full_like(node_type_ids, NO_PROFILE_FAMILY_ID),
        )
        profiles = canonicalize_v2_predicted_profiles(
            predicted_family_ids,
            output.raw_profile_parameters,
            predicted_sketch_mask,
        )
        non_profile = scatter_non_profile_geometry(
            output.non_profile_geometry
        )
        geometry = non_profile + profiles.geometry

        return tuple(
            _prediction_row(
                output,
                node_type_ids,
                retained_ids,
                profiles,
                geometry,
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
    node_type_ids,
    retained_ids,
    profiles,
    geometry,
    node_mask,
    batch_index,
    node_count_source,
):
    node_count = int(node_mask[batch_index].sum().item())
    raw_nodes = []
    for position in range(node_count):
        node_type_id = int(
            node_type_ids[batch_index, position].item()
        )
        retained = tuple(
            int(value)
            for value in retained_ids[
                batch_index, position
            ].detach().cpu().tolist()
        )
        primitives = tuple(
            int(value)
            for value in profiles.primitive_type_ids[
                batch_index, position
            ].detach().cpu().tolist()
        )
        categorical_ids = (
            retained[0],
            retained[1],
            retained[2],
            retained[3],
            *primitives,
            retained[4],
        )
        applicability = derive_geometry_mask(
            node_type_id, categorical_ids
        )
        profile_mask = tuple(
            bool(value)
            for value in profiles.geometry_mask[
                batch_index, position
            ].detach().cpu().tolist()
        )
        if any(
            applicability[channel] != profile_mask[channel]
            for channel in range(9, 33)
        ):
            raise AssertionError(
                "canonical profile mask disagrees with primitive slots"
            )
        merged_mask = list(profile_mask)
        for channel in NON_PROFILE_GEOMETRY_INDICES:
            merged_mask[channel] = applicability[channel]
        raw_nodes.append(
            RawDecodedNode(
                position,
                node_type_id,
                categorical_ids,
                tuple(
                    float(value)
                    for value in geometry[
                        batch_index, position
                    ].detach().cpu().tolist()
                ),
                tuple(merged_mask),
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
            for value in profiles.family_ids[
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
            for row in profiles.constrained_parameters[
                batch_index, :node_count
            ].detach().cpu().tolist()
        ),
    )


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
        ("node_type_logits", output.node_type_logits, 3),
        ("profile_family_logits", output.profile_family_logits, 3),
        ("raw_profile_parameters", output.raw_profile_parameters, 3),
        (
            "non_profile_geometry",
            output.non_profile_geometry,
            NON_PROFILE_GEOMETRY_WIDTH,
        ),
    )
    for name, value, rank in float_fields:
        if (
            not torch.is_tensor(value)
            or not value.dtype.is_floating_point
        ):
            raise TypeError("{} must be floating point".format(name))
        if value.dim() != rank or value.shape[:2] != expected_nodes:
            raise ValueError("{} is misaligned".format(name))
        if value.device != node_mask.device:
            raise ValueError("{} must share node_mask device".format(name))
        if not torch.isfinite(value).all():
            raise ValueError("{} must be finite".format(name))
    if output.node_type_logits.size(-1) != len(NODE_TYPES.tokens):
        raise ValueError("node_type_logits has the wrong class width")
    if output.profile_family_logits.size(-1) != len(PROFILE_FAMILIES):
        raise ValueError("profile_family_logits must have width three")
    if output.raw_profile_parameters.size(-1) != 3:
        raise ValueError("raw_profile_parameters must have width three")

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
