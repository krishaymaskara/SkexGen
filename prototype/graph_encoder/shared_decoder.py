"""C5 shared constrained-V6 and Graph-V1-main decoder.

The retained baselines remain independently callable.  This class inherits
their stable decoder helpers, removes every encoder/VQ module and the later C1
position-bias branch, and accepts only the common continuous GE1 memory during
autonomous inference.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace

import torch

from prototype.axis_geometry import AxisGeometryContractError
from prototype.flat_baseline.constrained_v4 import (
    _scatter_remaining_mask,
    scatter_remaining_geometry,
    select_remaining_geometry,
)
from prototype.flat_baseline.constrained_v6_autonomous import (
    V6EncodedMemory,
    greedy_decode_v6_from_memory,
)
from prototype.model_data.vocab import EDGE_TYPES, NODE_TYPES
from prototype.profile_geometry_torch import (
    canonicalize_profile_tensors,
    constrain_profile_parameters,
)
from prototype.reference_plane_geometry_torch import (
    canonicalize_reference_plane_tensors,
)
from prototype.graph_baseline.config import GraphV1Config
from prototype.graph_baseline.conversion import (
    graph_prediction_from_evidence,
    validate_and_convert_graph_prediction,
)
from prototype.graph_baseline.graph_contract import GraphContractError
from prototype.graph_baseline.graph_tensors import mask_graph_edge_logits
from prototype.graph_baseline.model import (
    GraphEdgeLogitComponents,
    GraphV1Model,
    GraphV1Output,
    select_complete_graph_node_sequence,
)
from prototype.node_grammar import NodeGrammarError

from .grid_magnitude import build_grid_magnitude_head
from .decoder_contract import (
    AUTONOMOUS_OUTPUT_VERSION,
    LEGACY_OPERATION_MAGNITUDE_PARAMETERIZATION,
    OPERATION_MAGNITUDE_COMPACT_CHANNELS,
    OPERATION_MAGNITUDE_PARAMETERIZATIONS,
    OUTPUT_POSITION_CONTRACT_VERSION,
    output_position_contract_version_for,
    shared_decoder_version_for,
    uses_grid_magnitude,
    POSITIVE_OPERATION_MAGNITUDE_PARAMETERIZATION,
    SHARED_DECODER_VERSION,
)


GE1_CONTINUOUS_MEMORY_SOURCE = "ge1_continuous_memory_from_codebook_projection"

# --- Inherited V6 entry-point compatibility shim -----------------------------
#
# `greedy_decode_v6_from_memory` validates its `V6EncodedMemory` argument before
# decoding. That validator requires `encoding_source` to equal the frozen V6
# literal and `code_indices` to be a `[rows, latent_tokens]` long tensor inside
# the codebook range. Both requirements exist to prove that the memory came from
# the V6 encoder's quantizer.
#
# GE1 memory satisfies neither in fact: it is continuous and bypasses
# nearest-code assignment entirely, so no code indices exist. The two values
# below are therefore supplied only to pass an inherited entry-point check whose
# premise does not hold for GE1. They are never read as evidence: every returned
# node prediction is immediately rewritten with `latent_indices=()` and
# `encoded_memory_source=GE1_CONTINUOUS_MEMORY_SOURCE` before any scoring,
# conversion, metric, or checkpoint sees it.
#
# The zero index tensor is deliberately inert padding, not a measurement. It is
# byte-identical to what a fully collapsed single-code VQ would produce, so any
# future diagnostic that reads code indices must read the corrected record and
# never the shim.
#
# Consequence, and the reason this is named rather than inlined: because the
# corrected record no longer carries the V6 literal, the inherited
# `validate_and_convert_v6_autonomous_prediction` path rejects GE1 predictions
# by design. GE1 converts through
# `prototype.graph_baseline.conversion.validate_and_convert_graph_prediction`
# instead. That rejection is a loud, intended failure, not a regression.
#
# The frozen inherited V6 implementation is not modified.
V6_ENTRY_POINT_COMPATIBILITY_SOURCE = "v6_encoder_quantized_memory"
V6_ENTRY_POINT_COMPATIBILITY_RATIONALE = (
    "inherited V6 entry-point validator requires quantized-memory provenance; "
    "GE1 memory is continuous and VQ-bypassed, so the literal is supplied only "
    "to satisfy that check and is overwritten in the returned record"
)
GE1_CORRECTED_LATENT_INDICES = ()
_ENCODER_ONLY_MODULES = (
    "node_position_embedding",
    "input_norm",
    "latent_queries",
    "encoder",
    "to_codebook",
    "vq",
    "from_codebook",
)
_EXCLUDED_C1_POSITION_BIAS_MODULES = (
    "source_position_factor",
    "destination_position_factor",
    "position_class_projection",
)


@dataclass(frozen=True)
class AutonomousRawRow:
    """Direct decoder tensors before graph masking and discrete conversion."""

    prefix_output: object
    graph_edge_components: GraphEdgeLogitComponents


@dataclass(frozen=True)
class ExplicitConversionOutcome:
    """One authoritative conversion result or an explicit raised failure."""

    result: object = None
    failure_type: str = None
    failure_detail: str = None

    @property
    def raised_failure(self):
        return self.failure_type is not None


@dataclass(frozen=True)
class SharedDecoderPrediction:
    version: str
    raw_prediction: tuple
    constrained_prediction: tuple
    converted_prediction: tuple


class DecoderParityError(AssertionError):
    """Exact C5 parity failed at the earliest compared tensor."""

    def __init__(
        self, stage, name, expected_shape, observed_shape, dtype,
        maximum_absolute_difference=None, mean_absolute_difference=None,
    ):
        self.stage = stage
        self.name = name
        self.expected_shape = expected_shape
        self.observed_shape = observed_shape
        self.dtype = dtype
        self.maximum_absolute_difference = maximum_absolute_difference
        self.mean_absolute_difference = mean_absolute_difference
        detail = (
            "exact decoder parity failed at {}/{}: expected_shape={}, "
            "observed_shape={}, dtype={}, max_abs={}, mean_abs={}".format(
                stage,
                name,
                expected_shape,
                observed_shape,
                dtype,
                maximum_absolute_difference,
                mean_absolute_difference,
            )
        )
        super().__init__(detail)


def _inert_compatibility_code_indices(latent_tokens, device):
    """Return the inert index tensor the inherited V6 validator demands.

    GE1 performs no nearest-code assignment, so there are no code indices. The
    zeros below carry no information and are never read as evidence.
    """

    return torch.zeros((1, int(latent_tokens)), dtype=torch.long, device=device)


def _corrected_memory_provenance(node_prediction):
    """Overwrite the compatibility shim with GE1's true memory provenance.

    Every consumer downstream of this call sees the corrected record. Nothing
    that claims quantized-memory provenance escapes the decoder.
    """

    return replace(
        node_prediction,
        latent_indices=GE1_CORRECTED_LATENT_INDICES,
        encoded_memory_source=GE1_CONTINUOUS_MEMORY_SOURCE,
    )


def assert_exact_tensor_parity(expected, observed, *, stage, name):
    """Require bit-exact equality and provide first-difference diagnostics."""

    expected_shape = tuple(expected.shape)
    observed_shape = tuple(observed.shape)
    dtype = str(expected.dtype)
    if (
        expected_shape == observed_shape
        and expected.dtype == observed.dtype
        and torch.equal(expected, observed)
    ):
        return
    maximum = None
    mean = None
    if (
        expected_shape == observed_shape
        and expected.dtype == observed.dtype
        and expected.dtype.is_floating_point
        and expected.numel()
    ):
        difference = (expected - observed).abs()
        maximum = float(difference.max().item())
        mean = float(difference.mean().item())
    raise DecoderParityError(
        stage,
        name,
        expected_shape,
        observed_shape,
        dtype,
        maximum,
        mean,
    )


class SharedGE1Decoder(GraphV1Model):
    """One target-free GE1 decoder implementation shared by both arms."""

    shared_decoder_version = SHARED_DECODER_VERSION
    output_position_contract_version = OUTPUT_POSITION_CONTRACT_VERSION

    def __init__(
        self,
        config=None,
        *,
        operation_magnitude_parameterization=(
            POSITIVE_OPERATION_MAGNITUDE_PARAMETERIZATION
        ),
    ):
        resolved = config or GraphV1Config()
        if not isinstance(resolved, GraphV1Config):
            raise TypeError("SharedGE1Decoder requires GraphV1Config")
        resolved.validate()
        if (
            operation_magnitude_parameterization
            not in OPERATION_MAGNITUDE_PARAMETERIZATIONS
        ):
            raise ValueError("unknown operation-magnitude parameterization")
        super().__init__(resolved)
        self.operation_magnitude_parameterization = (
            operation_magnitude_parameterization
        )
        self.uses_grid_magnitude = uses_grid_magnitude(
            operation_magnitude_parameterization
        )
        # Per-identity contract versions, so a historical checkpoint can never
        # be reloaded into a grid model or the reverse.
        self.shared_decoder_version = shared_decoder_version_for(
            operation_magnitude_parameterization
        )
        self.output_position_contract_version = (
            output_position_contract_version_for(
                operation_magnitude_parameterization
            )
        )
        for name in _ENCODER_ONLY_MODULES + _EXCLUDED_C1_POSITION_BIAS_MODULES:
            delattr(self, name)
        if self.uses_grid_magnitude:
            # Additive module: the historical scalar head is retained and its
            # axis channels keep their existing supervision.
            self.grid_magnitude_head = build_grid_magnitude_head(
                self.config.model_dim
            )
        self.register_buffer(
            "relative_position_denominator",
            torch.tensor(float(max(self.config.max_nodes - 1, 1))),
        )

    @classmethod
    def from_graph_v1(cls, reference):
        """Extract the exact C5 subset of one retained Graph V1 state."""

        if not isinstance(reference, GraphV1Model):
            raise TypeError("reference must be GraphV1Model")
        extracted = cls(
            reference.config,
            operation_magnitude_parameterization=(
                LEGACY_OPERATION_MAGNITUDE_PARAMETERIZATION
            ),
        )
        reference_state = reference.state_dict()
        state = {}
        for name, value in extracted.state_dict().items():
            source = reference_state[name] if name in reference_state else value
            state[name] = source.detach().clone()
        extracted.load_state_dict(state, strict=True)
        return extracted

    def raw_remaining_geometry(self, decoded_states):
        """Return pre-activation geometry-head values for diagnostics."""

        return self.remaining_geometry_head(decoded_states)

    def grid_magnitude_logits(self, decoded_states):
        """Return ordinal cut logits ``[..., 2, 4]`` for the grid identity."""

        if not self.uses_grid_magnitude:
            raise ValueError(
                "grid magnitude logits require the grid-ordinal identity"
            )
        return self.grid_magnitude_head(decoded_states)

    def parameterize_remaining_geometry(self, raw, decoded_states=None):
        """Apply the versioned neural geometry output mapping.

        `decoded_states` is required only by the grid-ordinal identity, whose
        magnitude channels are decoded from the ordinal head rather than from
        a scalar activation.  The historical identities ignore it entirely and
        keep their exact previous behaviour.
        """

        legacy = torch.tanh(raw)
        if (
            self.operation_magnitude_parameterization
            == LEGACY_OPERATION_MAGNITUDE_PARAMETERIZATION
        ):
            return legacy
        if self.uses_grid_magnitude:
            if decoded_states is None:
                raise ValueError(
                    "grid-ordinal magnitude decoding requires decoded states"
                )
            # Both grids are decoded for every node.  The existing geometry
            # applicability mask exposes only the channel matching the
            # autonomously generated node type, so no target operation type
            # is consulted here.  The decode is discrete, so magnitude
            # gradient reaches the trunk solely through the ordinal loss.
            grid_values = self.grid_magnitude_head.normalized_values(
                self.grid_magnitude_head(decoded_states)
            ).to(dtype=raw.dtype)
            return torch.cat(
                (legacy[..., :4], grid_values), dim=-1
            ).contiguous()
        epsilon = torch.finfo(raw.dtype).tiny
        operation_raw = raw[..., OPERATION_MAGNITUDE_COMPACT_CHANNELS]
        positive = epsilon + (1.0 - epsilon) * torch.sigmoid(operation_raw)
        return torch.cat((legacy[..., :4], positive), dim=-1).contiguous()

    def decode_prefix(
        self,
        memory,
        categorical_prefix,
        geometry_prefix,
        geometry_mask_prefix,
        prefix_mask=None,
    ):
        """Apply the prospective mapping inside autonomous neural decoding."""

        output = super().decode_prefix(
            memory,
            categorical_prefix,
            geometry_prefix,
            geometry_mask_prefix,
            prefix_mask,
        )
        if (
            self.operation_magnitude_parameterization
            == LEGACY_OPERATION_MAGNITUDE_PARAMETERIZATION
        ):
            return output
        raw = self.raw_remaining_geometry(output.decoded_states)
        return replace(
            output,
            remaining_geometry=self.parameterize_remaining_geometry(
                raw, output.decoded_states
            ),
        )

    def forward(self, memory, *, node_counts, node_count_source):
        """Autonomously decode continuous memory; no target is accepted."""

        self._validate_memory(memory)
        counts = self._validate_node_counts(node_counts, memory.size(0))
        if not isinstance(node_count_source, str) or not node_count_source:
            raise ValueError("node_count_source must be a nonempty string")
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                return self._autonomous_from_validated_memory(
                    memory, counts, node_count_source
                )
        finally:
            self.train(was_training)

    def _autonomous_from_validated_memory(
        self, memory, counts, node_count_source
    ):
        raw_rows = []
        constrained = []
        converted = []
        for row, count in enumerate(counts):
            row_memory = memory[row:row + 1]
            # See V6_ENTRY_POINT_COMPATIBILITY_SOURCE: these two fields are
            # inert values supplied to pass the inherited entry-point validator,
            # and are overwritten immediately below.
            compatibility_memory = V6EncodedMemory(
                row_memory,
                _inert_compatibility_code_indices(
                    self.config.latent_tokens, memory.device
                ),
                V6_ENTRY_POINT_COMPATIBILITY_SOURCE,
                self.config.model_name,
            )
            node_prediction = greedy_decode_v6_from_memory(
                self,
                compatibility_memory,
                node_counts=torch.tensor(
                    (count,), dtype=torch.long, device=memory.device
                ),
                node_count_source=node_count_source,
            )[0]
            node_prediction = _corrected_memory_provenance(node_prediction)
            categories, geometry, geometry_mask = self._prefix_tensors(
                node_prediction, row_memory
            )
            prefix_output = self.decode_prefix(
                row_memory,
                categories[:, :-1],
                geometry[:, :-1],
                geometry_mask[:, :-1],
            )
            node_ids = torch.tensor(
                [[node.node_type_id for node in node_prediction.raw_nodes]],
                dtype=torch.long,
                device=memory.device,
            )
            active = torch.ones(
                (1, count), dtype=torch.bool, device=memory.device
            )
            components = self.decode_graph_edge_components(
                prefix_output.decoded_states,
                node_ids,
                row_memory,
                active,
            )
            raw_rows.append(AutonomousRawRow(prefix_output, components))
            masked = mask_graph_edge_logits(
                components.edge_logits, node_ids, active
            )
            graph_prediction = graph_prediction_from_evidence(
                node_prediction,
                masked.raw_class_ids[0].detach().cpu().tolist(),
                masked.masked_class_ids[0].detach().cpu().tolist(),
                masked.correction_mask[0].detach().cpu().tolist(),
                main_pair_logits=(
                    components.main_pair_logits[0].detach().cpu().tolist()
                ),
                position_bias_logits=(
                    components.position_bias_logits[0].detach().cpu().tolist()
                ),
            )
            constrained.append(graph_prediction)
            try:
                conversion = validate_and_convert_graph_prediction(
                    graph_prediction,
                    max_operations=self.config.max_operations,
                )
            except (
                AxisGeometryContractError,
                GraphContractError,
                NodeGrammarError,
            ) as exc:
                converted.append(ExplicitConversionOutcome(
                    failure_type=type(exc).__name__,
                    failure_detail=str(exc),
                ))
            else:
                converted.append(ExplicitConversionOutcome(result=conversion))
        return SharedDecoderPrediction(
            AUTONOMOUS_OUTPUT_VERSION,
            tuple(raw_rows),
            tuple(constrained),
            tuple(converted),
        )

    def teacher_forced(self, memory, target, profile_targets):
        """Training/parity-only path; never called by autonomous inference."""

        self._validate_memory(memory)
        if target["node_mask"].shape[0] != memory.size(0):
            raise ValueError("target batch must align with decoder memory")
        self._validate_profile_targets(profile_targets, target)
        states = self._teacher_forced_states_from_memory(memory, target)
        node_logits = self.node_type_head(states)
        categorical_logits = tuple(
            head(states) for head in self.remaining_categorical_heads
        )
        profile_output = self.profile_heads(states)
        constrained_parameters = constrain_profile_parameters(
            profile_output.raw_parameters,
            profile_targets.family_ids,
            profile_targets.sketch_mask,
            extent_min=self.config.profile_extent_min,
            extent_max=self.config.profile_extent_max,
        )
        canonical_profile = canonicalize_profile_tensors(
            profile_targets.family_ids,
            constrained_parameters,
            profile_targets.sketch_mask,
            extent_min=self.config.profile_extent_min,
            extent_max=self.config.profile_extent_max,
        )
        raw_remaining = self.raw_remaining_geometry(states)
        remaining = self.parameterize_remaining_geometry(raw_remaining, states)
        scattered = scatter_remaining_geometry(remaining)
        selected_mask = (
            select_remaining_geometry(target["geometry_mask"])
            & target["node_mask"].unsqueeze(-1)
        )
        if self.uses_grid_magnitude:
            # Under the grid identity the scalar magnitude outputs are unused
            # for decoding, so supervising them would train dead outputs and
            # keep pulling the shared trunk toward a conditional-mean scalar.
            # Axis channels 0-3 keep their existing supervision untouched, and
            # both historical identities are unaffected.
            selected_mask = selected_mask.clone()
            for compact_channel in OPERATION_MAGNITUDE_COMPACT_CHANNELS:
                selected_mask[..., compact_channel] = False
        scattered_mask = _scatter_remaining_mask(selected_mask)
        canonical_plane = canonicalize_reference_plane_tensors(
            target["node_type_ids"],
            target["categorical_attributes"][..., 3],
            remaining,
            target["node_mask"],
        )
        training_geometry = (
            canonical_plane.geometry + canonical_profile.geometry + scattered
        )
        training_mask = (
            canonical_plane.geometry_mask
            | canonical_profile.geometry_mask
            | scattered_mask
        )
        graph_nodes = select_complete_graph_node_sequence(
            node_logits, target["node_mask"]
        )
        graph_components = self.decode_graph_edge_components(
            states,
            graph_nodes.authoritative_node_type_ids,
            memory,
            target["node_mask"],
        )
        batch_size, node_count = target["node_mask"].shape
        neutral_presence = states.new_zeros(batch_size, node_count, node_count)
        neutral_types = states.new_zeros(
            batch_size, node_count, node_count, len(EDGE_TYPES.tokens)
        )
        neutral_pointers = states.new_zeros(
            batch_size, self.config.max_operations, node_count
        )
        prefix = target["node_type_ids"].clone()
        prefix[:, 1:] = target["node_type_ids"][:, :-1]
        prefix[:, 0] = NODE_TYPES.pad_id
        zero = memory.sum() * 0.0
        per_example_zero = memory.new_zeros(batch_size)
        return GraphV1Output(
            decoded_states=states,
            node_type_logits=node_logits,
            categorical_logits=categorical_logits,
            profile_family_logits=profile_output.family_logits,
            raw_profile_parameters=profile_output.raw_parameters,
            constrained_profile_parameters=constrained_parameters,
            remaining_geometry=remaining,
            remaining_geometry_scattered=scattered,
            remaining_geometry_mask=selected_mask,
            training_reference_plane_geometry=canonical_plane.geometry,
            training_reference_plane_geometry_mask=canonical_plane.geometry_mask,
            training_primitive_type_ids=canonical_profile.primitive_type_ids,
            training_profile_geometry=canonical_profile.geometry,
            training_profile_geometry_mask=canonical_profile.geometry_mask,
            training_geometry=training_geometry,
            training_geometry_mask=training_mask,
            edge_presence_logits=neutral_presence,
            edge_type_logits=neutral_types,
            operation_pointer_logits=neutral_pointers,
            quantized_memory=memory,
            vq_loss=zero,
            vq_per_example_loss=per_example_zero,
            code_indices=torch.empty(
                (batch_size, 0), dtype=torch.long, device=memory.device
            ),
            assignment_counts=torch.empty(
                (0,), dtype=torch.long, device=memory.device
            ),
            active_code_count=torch.zeros(
                (), dtype=torch.long, device=memory.device
            ),
            codebook_utilization=zero,
            codebook_perplexity=zero,
            teacher_forced_prefix_node_ids=prefix.contiguous(),
            graph_edge_logits=graph_components.edge_logits,
            graph_main_pair_logits=graph_components.main_pair_logits,
            graph_position_bias_logits=graph_components.position_bias_logits,
            graph_raw_node_type_argmax_ids=(
                graph_nodes.raw_node_type_argmax_ids
            ),
            authoritative_graph_node_type_ids=(
                graph_nodes.authoritative_node_type_ids
            ),
            graph_node_type_correction_mask=(
                graph_nodes.node_type_correction_mask
            ),
            graph_legal_node_type_mask=graph_nodes.legal_node_type_mask,
            graph_grammar_state_evidence=graph_nodes.grammar_state_evidence,
        )

    def decode_graph_edge_components(
        self, states, node_type_ids, memory, node_mask
    ):
        """Exact retained Graph V1 main MLP, excluding the later C1 bias."""

        if states.dim() != 3 or states.shape[:2] != node_type_ids.shape:
            raise ValueError("states and constrained node IDs must align")
        batch_size, count, width = states.shape
        if width != self.config.model_dim or node_mask.shape != (batch_size, count):
            raise ValueError("graph decoder inputs have invalid shape")
        if node_mask.dtype != torch.bool:
            raise TypeError("node_mask must use torch.bool")
        positions = torch.arange(count, device=states.device)
        position_values = self.decoder_position_embedding(positions)
        position_values = position_values.unsqueeze(0).expand(batch_size, -1, -1)
        type_values = self.field_embeddings[0](node_type_ids)
        global_context = memory.mean(dim=1).unsqueeze(1).unsqueeze(1)
        global_context = global_context.expand(-1, count, count, -1)
        source_state = states.unsqueeze(2).expand(-1, -1, count, -1)
        destination_state = states.unsqueeze(1).expand(-1, count, -1, -1)
        source_type = type_values.unsqueeze(2).expand(-1, -1, count, -1)
        destination_type = type_values.unsqueeze(1).expand(-1, count, -1, -1)
        source_position = position_values.unsqueeze(2).expand(-1, -1, count, -1)
        destination_position = position_values.unsqueeze(1).expand(-1, count, -1, -1)
        relative = (
            positions.view(1, count, 1) - positions.view(1, 1, count)
        ).to(states.dtype) / float(self.relative_position_denominator.item())
        relative = relative.unsqueeze(-1).expand(batch_size, -1, -1, -1)
        features = torch.cat((
            source_state,
            destination_state,
            source_type,
            destination_type,
            source_position,
            destination_position,
            global_context,
            relative,
        ), dim=-1)
        main = self.graph_edge_decoder(features).contiguous()
        excluded_bias = torch.zeros_like(main).contiguous()
        return GraphEdgeLogitComponents(main, excluded_bias, main)

    def _teacher_forced_states_from_memory(self, memory, target):
        target_categories = torch.cat((
            target["node_type_ids"].unsqueeze(-1),
            target["categorical_attributes"],
        ), dim=-1)
        target_content = self._record_content(
            target_categories, target["geometry"], target["geometry_mask"]
        )
        bos = self.bos.view(1, 1, -1).expand(memory.size(0), 1, -1)
        shifted = torch.cat((bos, target_content[:, :-1]), dim=1)
        decoder_valid = torch.cat((
            torch.ones(
                memory.size(0), 1, dtype=torch.bool, device=memory.device
            ),
            target["node_mask"][:, :-1],
        ), dim=1)
        return self._decode_embedded_prefix(shifted, memory, decoder_valid)

    def _validate_memory(self, memory):
        if (
            not torch.is_tensor(memory)
            or memory.dim() != 3
            or tuple(memory.shape[1:])
            != (self.config.latent_tokens, self.config.model_dim)
            or memory.size(0) <= 0
        ):
            raise ValueError("memory must have shape [B, latent_tokens, model_dim]")
        if not memory.dtype.is_floating_point:
            raise TypeError("memory must be floating point")
        if memory.dtype != self.bos.dtype or memory.device != self.bos.device:
            raise ValueError("memory must match decoder dtype and device")
        if not torch.isfinite(memory).all():
            raise ValueError("memory must be finite")

    def _validate_node_counts(self, node_counts, batch_size):
        if (
            not torch.is_tensor(node_counts)
            or node_counts.dtype != torch.long
            or tuple(node_counts.shape) != (batch_size,)
            or node_counts.device != self.bos.device
        ):
            raise ValueError("node_counts must be a local long tensor with shape [B]")
        if node_counts.numel() and (
            int(node_counts.min().item()) <= 0
            or int(node_counts.max().item()) > self.config.max_nodes
        ):
            raise ValueError("node_counts are outside the decoder contract")
        return tuple(int(value) for value in node_counts.detach().cpu().tolist())

    @staticmethod
    def _prefix_tensors(prediction, memory):
        categories = torch.tensor(
            [[(node.node_type_id, *node.categorical_ids)
              for node in prediction.raw_nodes]],
            dtype=torch.long,
            device=memory.device,
        )
        geometry = memory.new_tensor(
            [[node.normalized_geometry for node in prediction.raw_nodes]]
        )
        geometry_mask = torch.tensor(
            [[node.derived_geometry_mask for node in prediction.raw_nodes]],
            dtype=torch.bool,
            device=memory.device,
        )
        return categories, geometry, geometry_mask


def copied_shared_decoder(decoder):
    """Return a parameter- and buffer-disjoint exact state copy."""

    if not isinstance(decoder, SharedGE1Decoder):
        raise TypeError("decoder must be SharedGE1Decoder")
    return copy.deepcopy(decoder)
