"""Real-PyTorch tests for the ADR-0015 softmax magnitude classification head.

These require a PyTorch runtime.  They are corpus-free: every fixture is
synthetic and no corpus, manifest, checkpoint, or preserved payload is opened.
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

try:
    import torch
except ImportError:  # pragma: no cover - exercised only without PyTorch
    torch = None

from prototype.graph_encoder import grid_magnitude as gm
from prototype.graph_encoder.config import (
    grid_frozen_encoder_config,
    grid_softmax_frozen_encoder_config,
)
from prototype.graph_encoder.decoder_contract import (
    GRID_CHECKPOINT_SCHEMA,
    GRID_ORDINAL_OPERATION_MAGNITUDE_PARAMETERIZATION,
    GRID_OUTPUT_POSITION_CONTRACT_VERSION,
    GRID_SOFTMAX_CHECKPOINT_SCHEMA,
    GRID_SOFTMAX_OPERATION_MAGNITUDE_PARAMETERIZATION,
    GRID_SOFTMAX_SHARED_DECODER_VERSION,
    POSITIVE_OPERATION_MAGNITUDE_PARAMETERIZATION,
)


PACKAGE = Path(__file__).parents[1]
REASON = "softmax magnitude runtime tests require the PyTorch runtime"
SOFTMAX = GRID_SOFTMAX_OPERATION_MAGNITUDE_PARAMETERIZATION
ORDINAL = GRID_ORDINAL_OPERATION_MAGNITUDE_PARAMETERIZATION

# Mirrors the corpus-free trajectory diagnostic (job 3353008) exactly, so the
# reachability regression below reproduces its measurement rather than
# inventing a new one.
DIAGNOSTIC_SEED = 2026
DIAGNOSTIC_WIDTH = 32
DIAGNOSTIC_BATCH = 8
DIAGNOSTIC_NODES = 1
DIAGNOSTIC_UPDATES = 200
DIAGNOSTIC_LEARNING_RATE = 0.001
DIAGNOSTIC_CLIP_NORM = 1.0
GEOMETRY_WIDTH = 39


@unittest.skipIf(torch is None, REASON)
class GridSoftmaxHeadTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(DIAGNOSTIC_SEED)
        self.width = DIAGNOSTIC_WIDTH
        self.head = gm.build_grid_softmax_magnitude_head(self.width)

    def test_output_shape_and_dtype(self):
        states = torch.randn(3, 7, self.width, dtype=torch.float32)
        logits = self.head(states)
        self.assertEqual(
            tuple(logits.shape),
            (3, 7, len(gm.OPERATION_TYPES), gm.GRID_CLASS_COUNT),
        )
        self.assertTrue(torch.isfinite(logits).all())

    def test_head_identity_and_parameter_shapes(self):
        self.assertEqual(self.head.parameterization, SOFTMAX)
        self.assertEqual(self.head.contract_version,
                         gm.GRID_SOFTMAX_MAGNITUDE_CONTRACT_VERSION)
        shapes = {
            name: tuple(parameter.shape)
            for name, parameter in self.head.named_parameters()
        }
        self.assertEqual(shapes, {
            "projections.0.weight": (gm.GRID_CLASS_COUNT, self.width),
            "projections.0.bias": (gm.GRID_CLASS_COUNT,),
            "projections.1.weight": (gm.GRID_CLASS_COUNT, self.width),
            "projections.1.bias": (gm.GRID_CLASS_COUNT,),
        })

    def test_no_shared_scalar_or_ordered_bias_structure_remains(self):
        """The removed structure is precisely what made classes 2-4 unreachable."""

        names = dict(self.head.named_parameters())
        self.assertNotIn("first_bias", names)
        self.assertNotIn("bias_gaps", names)
        self.assertFalse(hasattr(self.head, "ordered_biases"))
        self.assertFalse(hasattr(self.head, "cumulative_probabilities"))

    def test_class_evidence_moves_independently(self):
        """Perturbing one class row must not move any other class logit."""

        states = torch.randn(4, 3, self.width)
        for position in range(len(gm.OPERATION_TYPES)):
            for target_class in range(gm.GRID_CLASS_COUNT):
                with torch.no_grad():
                    before = self.head(states)[..., position, :].clone()
                    self.head.projections[position].bias[target_class] += 2.5
                    after = self.head(states)[..., position, :]
                    self.head.projections[position].bias[target_class] -= 2.5
                delta = (after - before).abs()
                with self.subTest(position=position, cls=target_class):
                    self.assertGreater(float(delta[..., target_class].min()), 2.4)
                    others = [
                        index for index in range(gm.GRID_CLASS_COUNT)
                        if index != target_class
                    ]
                    self.assertEqual(float(delta[..., others].max()), 0.0)

    def test_logits_are_not_forced_to_be_rank_consistent(self):
        """The ordinal head cannot express this; the softmax head must."""

        with torch.no_grad():
            for parameter in self.head.parameters():
                parameter.normal_(0.0, 3.0)
        logits = self.head(torch.randn(64, 5, self.width) * 5.0)
        differences = logits[..., 1:] - logits[..., :-1]
        self.assertTrue(bool((differences > 0).any().item()))
        self.assertTrue(bool((differences < 0).any().item()))

    def test_decode_is_argmax_over_class_logits(self):
        logits = torch.randn(6, 4, len(gm.OPERATION_TYPES), gm.GRID_CLASS_COUNT)
        torch.testing.assert_close(
            self.head.class_indices(logits),
            logits.argmax(dim=-1),
            rtol=0.0,
            atol=0.0,
        )

    def test_class_probabilities_are_a_normalized_simplex(self):
        probabilities = self.head.class_probabilities(
            self.head(torch.randn(5, 3, self.width))
        )
        self.assertEqual(probabilities.size(-1), gm.GRID_CLASS_COUNT)
        self.assertTrue(bool((probabilities > 0).all().item()))
        torch.testing.assert_close(
            probabilities.sum(dim=-1),
            torch.ones(probabilities.shape[:-1]),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_decoded_values_are_exact_frozen_grid_members(self):
        values = self.head.normalized_values(
            self.head(torch.randn(16, 4, self.width))
        )
        for position, name in enumerate(gm.OPERATION_TYPES):
            allowed = torch.tensor(gm.NORMALIZED_GRIDS[name])
            for item in values[..., position].reshape(-1):
                self.assertTrue(
                    bool((allowed == item).any().item()),
                    "decoded value {} is not a frozen grid member".format(item),
                )

    def test_every_class_is_directly_addressable_for_both_grids(self):
        for position in range(len(gm.OPERATION_TYPES)):
            with torch.no_grad():
                self.head.projections[position].weight.zero_()
                self.head.projections[position].bias.zero_()
            for target_class in range(gm.GRID_CLASS_COUNT):
                with torch.no_grad():
                    self.head.projections[position].bias.zero_()
                    self.head.projections[position].bias[target_class] = 1.0
                logits = self.head(torch.zeros(1, 1, self.width))
                decoded = int(self.head.class_indices(logits)[0, 0, position])
                with self.subTest(position=position, cls=target_class):
                    self.assertEqual(decoded, target_class)

    def test_finite_nonzero_gradients_through_the_head(self):
        states = torch.randn(4, 3, self.width, requires_grad=True)
        logits = self.head(states)
        labels = torch.zeros(
            logits.shape[:-1], dtype=torch.long
        ).reshape(-1)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, gm.GRID_CLASS_COUNT), labels
        )
        loss.backward()
        for name, parameter in self.head.named_parameters():
            with self.subTest(parameter=name):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertGreater(float(parameter.grad.abs().sum()), 0.0)
        self.assertIsNotNone(states.grad)
        self.assertTrue(bool((states.grad.abs().sum() > 0).item()))

    def test_decode_carries_no_gradient(self):
        states = torch.randn(2, 2, self.width, requires_grad=True)
        values = self.head.normalized_values(self.head(states))
        self.assertFalse(values.requires_grad)

    def test_state_width_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            self.head(torch.randn(2, 2, self.width + 1))

    def test_invalid_model_dim_is_rejected(self):
        for width in (0, -1, True, 3.5):
            with self.subTest(width=width):
                with self.assertRaises(ValueError):
                    gm.build_grid_softmax_magnitude_head(width)


def _generated_states():
    """The trajectory diagnostic's fixed, non-trainable decoder states."""

    states = torch.zeros(DIAGNOSTIC_BATCH, DIAGNOSTIC_NODES, DIAGNOSTIC_WIDTH)
    states[..., 0] = 1.0
    states.requires_grad_(False)
    return states


def _generated_target(operation_type, target_class):
    """One synthetic teacher target holding a single frozen grid member."""

    position = gm.OPERATION_TYPES.index(operation_type)
    channel = gm.SERIALIZED_CHANNELS[operation_type]
    geometry = torch.zeros(DIAGNOSTIC_BATCH, DIAGNOSTIC_NODES, GEOMETRY_WIDTH)
    geometry[..., channel] = gm.NORMALIZED_GRIDS[operation_type][target_class]
    geometry_mask = torch.zeros(
        DIAGNOSTIC_BATCH, DIAGNOSTIC_NODES, GEOMETRY_WIDTH, dtype=torch.bool
    )
    geometry_mask[..., channel] = True
    return {
        "node_type_ids": torch.full(
            (DIAGNOSTIC_BATCH, DIAGNOSTIC_NODES),
            gm.OPERATION_NODE_TYPE_IDS[position],
            dtype=torch.long,
        ),
        "node_mask": torch.ones(
            DIAGNOSTIC_BATCH, DIAGNOSTIC_NODES, dtype=torch.bool
        ),
        "geometry": geometry,
        "geometry_mask": geometry_mask,
    }


def _decoded_class_after_training(builder, config, operation_type, target_class):
    """Train one freshly seeded head alone and return its decoded class."""

    from prototype.graph_encoder.losses import grid_magnitude_terms

    torch.manual_seed(DIAGNOSTIC_SEED)
    head = builder(DIAGNOSTIC_WIDTH)
    states = _generated_states()
    target = _generated_target(operation_type, target_class)
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=DIAGNOSTIC_LEARNING_RATE, weight_decay=0.0
    )
    for _ in range(DIAGNOSTIC_UPDATES):
        optimizer.zero_grad(set_to_none=True)
        terms = grid_magnitude_terms(head(states), target, config)
        terms.weighted[operation_type].backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), DIAGNOSTIC_CLIP_NORM)
        optimizer.step()
    position = gm.OPERATION_TYPES.index(operation_type)
    with torch.no_grad():
        return int(head.class_indices(head(states))[0, 0, position])


@unittest.skipIf(torch is None, REASON)
class GridSoftmaxReachabilityTests(unittest.TestCase):
    """The regression this whole change exists to fix.

    Corpus-free trajectory diagnostic job `3353008` trained the unchanged real
    ordinal head and loss on fixed generated decoder states and found that
    classes 2, 3, and 4 never decoded above class 1 within 200 AdamW updates.
    These two tests reproduce that measurement with the same seed, states,
    optimizer, budget, and real loss path, and assert the repair.
    """

    def test_every_class_becomes_reachable_within_the_diagnostic_budget(self):
        config = grid_softmax_frozen_encoder_config("flat")
        for operation_type in gm.OPERATION_TYPES:
            decoded = tuple(
                _decoded_class_after_training(
                    gm.build_grid_softmax_magnitude_head,
                    config,
                    operation_type,
                    target_class,
                )
                for target_class in range(gm.GRID_CLASS_COUNT)
            )
            with self.subTest(operation_type=operation_type):
                self.assertEqual(decoded, tuple(range(gm.GRID_CLASS_COUNT)))
                self.assertEqual(set(decoded), set(range(gm.GRID_CLASS_COUNT)))

    def test_the_ordinal_head_still_fails_the_same_protocol(self):
        """Pins job `3353008`'s finding so the diagnosis cannot silently rot."""

        config = grid_frozen_encoder_config("flat")
        for operation_type in gm.OPERATION_TYPES:
            decoded = tuple(
                _decoded_class_after_training(
                    gm.build_grid_magnitude_head,
                    config,
                    operation_type,
                    target_class,
                )
                for target_class in range(gm.GRID_CLASS_COUNT)
            )
            with self.subTest(operation_type=operation_type):
                self.assertLessEqual(max(decoded), 1)
                for target_class in (2, 3, 4):
                    self.assertNotIn(target_class, decoded)


@unittest.skipIf(torch is None, REASON)
class GridSoftmaxLossTests(unittest.TestCase):
    def setUp(self):
        self.config = grid_softmax_frozen_encoder_config("flat")
        torch.manual_seed(DIAGNOSTIC_SEED)
        self.head = gm.build_grid_softmax_magnitude_head(DIAGNOSTIC_WIDTH)
        self.states = _generated_states()

    def _terms(self, operation_type, target_class):
        from prototype.graph_encoder.losses import grid_magnitude_terms

        return grid_magnitude_terms(
            self.head(self.states),
            _generated_target(operation_type, target_class),
            self.config,
        )

    def test_wrong_width_logits_are_rejected(self):
        from prototype.graph_encoder.errors import GraphEncoderError
        from prototype.graph_encoder.losses import grid_magnitude_terms

        ordinal_shaped = torch.zeros(
            DIAGNOSTIC_BATCH,
            DIAGNOSTIC_NODES,
            len(gm.OPERATION_TYPES),
            gm.ORDINAL_CUT_COUNT,
        )
        with self.assertRaises(GraphEncoderError):
            grid_magnitude_terms(
                ordinal_shaped, _generated_target("extrude", 0), self.config
            )

    def test_softmax_shaped_logits_are_rejected_by_the_ordinal_identity(self):
        from prototype.graph_encoder.errors import GraphEncoderError
        from prototype.graph_encoder.losses import grid_magnitude_terms

        with self.assertRaises(GraphEncoderError):
            grid_magnitude_terms(
                self.head(self.states),
                _generated_target("extrude", 0),
                grid_frozen_encoder_config("flat"),
            )

    def test_raw_loss_equals_the_manual_cross_entropy_reduction(self):
        """Steps 2-4 of the ADR-0013 reduction must be byte-identical."""

        terms = self._terms("extrude", 3)
        logits = self.head(self.states)[..., 0, :]
        labels = torch.full(
            (DIAGNOSTIC_BATCH, DIAGNOSTIC_NODES), 3, dtype=torch.long
        )
        per_operation = torch.nn.functional.cross_entropy(
            logits.reshape(-1, gm.GRID_CLASS_COUNT),
            labels.reshape(-1),
            reduction="none",
        ).reshape(labels.shape)
        expected = per_operation.sum(dim=-1).mean()
        torch.testing.assert_close(
            terms.raw["extrude"], expected, rtol=1e-6, atol=1e-6
        )

    def test_inactive_operations_contribute_exactly_zero(self):
        terms = self._terms("extrude", 2)
        self.assertEqual(terms.active_operation_counts["revolve"], 0)
        self.assertEqual(float(terms.raw["revolve"].detach()), 0.0)
        self.assertEqual(
            terms.active_operation_counts["extrude"],
            DIAGNOSTIC_BATCH * DIAGNOSTIC_NODES,
        )

    def test_versions_and_reduction_record_are_the_softmax_ones(self):
        terms = self._terms("revolve", 4)
        self.assertEqual(terms.version, gm.GRID_SOFTMAX_MAGNITUDE_LOSS_VERSION)
        self.assertEqual(terms.parameterization, SOFTMAX)
        self.assertTrue(terms.uses_softmax_identity)
        self.assertEqual(
            terms.reduction["per_operation"],
            "cross_entropy_over_five_independent_class_logits",
        )

    def test_diagnostics_carry_five_class_probabilities_and_a_top_two_margin(self):
        terms = self._terms("extrude", 1)
        record = terms.diagnostic_record()
        json.dumps(record, sort_keys=True, allow_nan=False)
        self.assertEqual(record["parameterization"], SOFTMAX)
        self.assertEqual(record["decode_rule"], "argmax_over_class_logits")
        rows = record["operation_types"]["extrude"]["operation_records"]
        self.assertEqual(len(rows), DIAGNOSTIC_BATCH * DIAGNOSTIC_NODES)
        for row in rows:
            probabilities = row["ordinal_cut_probabilities"]
            self.assertEqual(len(probabilities), gm.GRID_CLASS_COUNT)
            self.assertAlmostEqual(sum(probabilities), 1.0, places=5)
            self.assertGreaterEqual(row["decision_margin"], 0.0)
            self.assertEqual(row["target_class"], 1)

    def test_diagnostic_record_keeps_every_ordinal_key(self):
        softmax = self._terms("extrude", 1).diagnostic_record()
        torch.manual_seed(DIAGNOSTIC_SEED)
        ordinal_head = gm.build_grid_magnitude_head(DIAGNOSTIC_WIDTH)
        from prototype.graph_encoder.losses import grid_magnitude_terms

        ordinal = grid_magnitude_terms(
            ordinal_head(self.states),
            _generated_target("extrude", 1),
            grid_frozen_encoder_config("flat"),
        ).diagnostic_record()
        self.assertTrue(set(ordinal) <= set(softmax))
        self.assertEqual(set(softmax) - set(ordinal),
                         {"parameterization", "decode_rule"})
        by_type = softmax["operation_types"]["extrude"]
        self.assertEqual(
            set(by_type), set(ordinal["operation_types"]["extrude"])
        )


@unittest.skipIf(torch is None, REASON)
class GridSoftmaxDecoderTests(unittest.TestCase):
    def _decoder(self, parameterization):
        from prototype.graph_encoder.shared_decoder import SharedGE1Decoder

        torch.manual_seed(DIAGNOSTIC_SEED)
        return SharedGE1Decoder(
            operation_magnitude_parameterization=parameterization
        )

    def test_softmax_identity_builds_the_softmax_head_and_v3_versions(self):
        decoder = self._decoder(SOFTMAX)
        self.assertTrue(decoder.uses_grid_magnitude)
        self.assertTrue(decoder.uses_grid_softmax_magnitude)
        self.assertEqual(decoder.grid_magnitude_head.parameterization, SOFTMAX)
        self.assertEqual(decoder.shared_decoder_version,
                         GRID_SOFTMAX_SHARED_DECODER_VERSION)
        self.assertEqual(decoder.output_position_contract_version,
                         GRID_OUTPUT_POSITION_CONTRACT_VERSION)

    def test_ordinal_identity_still_builds_the_ordinal_head(self):
        decoder = self._decoder(ORDINAL)
        self.assertTrue(decoder.uses_grid_magnitude)
        self.assertFalse(decoder.uses_grid_softmax_magnitude)
        self.assertEqual(decoder.grid_magnitude_head.parameterization, ORDINAL)

    def test_logits_have_the_five_wide_class_dimension(self):
        decoder = self._decoder(SOFTMAX)
        states = torch.randn(2, 6, decoder.config.model_dim)
        logits = decoder.grid_magnitude_logits(states)
        self.assertEqual(
            tuple(logits.shape),
            (2, 6, len(gm.OPERATION_TYPES), gm.GRID_CLASS_COUNT),
        )

    def test_historical_identity_still_refuses_grid_logits(self):
        decoder = self._decoder(POSITIVE_OPERATION_MAGNITUDE_PARAMETERIZATION)
        self.assertFalse(hasattr(decoder, "grid_magnitude_head"))
        with self.assertRaises(ValueError):
            decoder.grid_magnitude_logits(
                torch.randn(1, 1, decoder.config.model_dim)
            )

    def test_magnitude_channels_decode_to_grid_members_and_axes_stay_tanh(self):
        decoder = self._decoder(SOFTMAX)
        states = torch.randn(3, 5, decoder.config.model_dim)
        raw = decoder.raw_remaining_geometry(states)
        remaining = decoder.parameterize_remaining_geometry(raw, states)
        torch.testing.assert_close(
            remaining[..., :4], torch.tanh(raw)[..., :4], rtol=0.0, atol=0.0
        )
        for position, name in enumerate(gm.OPERATION_TYPES):
            allowed = torch.tensor(gm.NORMALIZED_GRIDS[name])
            for item in remaining[..., 4 + position].reshape(-1):
                self.assertTrue(bool((allowed == item).any().item()))

    def test_decoded_states_remain_required(self):
        decoder = self._decoder(SOFTMAX)
        raw = decoder.raw_remaining_geometry(
            torch.randn(2, 2, decoder.config.model_dim)
        )
        with self.assertRaises(ValueError):
            decoder.parameterize_remaining_geometry(raw)


@unittest.skipIf(torch is None, REASON)
class GridSoftmaxCapacityTests(unittest.TestCase):
    def _model_heads(self, parameterization):
        from prototype.graph_encoder.model import build_matched_ge1_models

        return build_matched_ge1_models(
            seed=DIAGNOSTIC_SEED,
            operation_magnitude_parameterization=parameterization,
        )

    @staticmethod
    def _trainable(model):
        return sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )

    @staticmethod
    def _head_parameters(model):
        return sum(
            parameter.numel()
            for name, parameter in model.named_parameters()
            if name.startswith("decoder.grid_magnitude_head.")
        )

    def test_recorded_head_parameter_counts(self):
        ordinal = self._model_heads(ORDINAL)[0]
        softmax = self._model_heads(SOFTMAX)[0]
        width = ordinal.config.model_dim
        self.assertEqual(self._head_parameters(ordinal), 2 * width + 8)
        self.assertEqual(
            self._head_parameters(softmax),
            2 * (gm.GRID_CLASS_COUNT * width + gm.GRID_CLASS_COUNT),
        )
        self.assertEqual(self._head_parameters(ordinal), 72)
        self.assertEqual(self._head_parameters(softmax), 330)

    def test_both_arms_grow_by_the_identical_shared_decoder_amount(self):
        ordinal = {
            model.config.encoder: self._trainable(model)
            for model in self._model_heads(ORDINAL)
        }
        softmax = {
            model.config.encoder: self._trainable(model)
            for model in self._model_heads(SOFTMAX)
        }
        deltas = {
            arm: softmax[arm] - ordinal[arm] for arm in ordinal
        }
        self.assertEqual(set(deltas.values()), {330 - 72})

    def test_capacity_parity_gate_still_passes(self):
        from prototype.graph_encoder.stage6_structure_only_producer import (
            capacity_gate,
        )

        for parameterization in (ORDINAL, SOFTMAX):
            counts = {
                model.config.encoder: self._trainable(model)
                for model in self._model_heads(parameterization)
            }
            gate = capacity_gate(counts["flat"], counts["typed_graph"])
            with self.subTest(parameterization=parameterization):
                self.assertTrue(gate["pass"])


@unittest.skipIf(torch is None, REASON)
class GridSoftmaxCheckpointTests(unittest.TestCase):
    def _model(self, parameterization):
        from prototype.graph_encoder.model import build_matched_ge1_models

        return build_matched_ge1_models(
            seed=DIAGNOSTIC_SEED,
            operation_magnitude_parameterization=parameterization,
        )[0]

    def test_state_dict_keys_and_shapes_are_incompatible_with_the_ordinal_head(self):
        prefix = "decoder.grid_magnitude_head."
        ordinal = {
            name[len(prefix):]: tuple(value.shape)
            for name, value in self._model(ORDINAL).state_dict().items()
            if name.startswith(prefix)
        }
        softmax = {
            name[len(prefix):]: tuple(value.shape)
            for name, value in self._model(SOFTMAX).state_dict().items()
            if name.startswith(prefix)
        }
        self.assertEqual(set(ordinal) - set(softmax),
                         {"first_bias", "bias_gaps"})
        self.assertEqual(set(softmax) - set(ordinal),
                         {"projections.0.bias", "projections.1.bias"})
        self.assertEqual(ordinal["projections.0.weight"], (1, 32))
        self.assertEqual(softmax["projections.0.weight"],
                         (gm.GRID_CLASS_COUNT, 32))

    def test_v3_round_trip_and_cross_identity_refusal(self):
        from prototype.graph_encoder.checkpoint import (
            GE1CheckpointError,
            load_ge1_checkpoint,
            save_ge1_checkpoint,
        )

        model = self._model(SOFTMAX)
        revision = "b" * 40
        with tempfile.TemporaryDirectory(prefix="grid-softmax-ckpt-") as root:
            path = Path(root) / "grid-v3.pt"
            save_ge1_checkpoint(path, model, code_revision=revision)
            loaded, payload = load_ge1_checkpoint(
                path,
                expected_code_revision=revision,
                expected_operation_magnitude_parameterization=SOFTMAX,
            )
            self.assertEqual(payload["checkpoint_schema"],
                             GRID_SOFTMAX_CHECKPOINT_SCHEMA)
            self.assertNotEqual(payload["checkpoint_schema"],
                                GRID_CHECKPOINT_SCHEMA)
            for name, value in model.state_dict().items():
                torch.testing.assert_close(
                    value, loaded.state_dict()[name], rtol=0.0, atol=0.0
                )
            for identity in (ORDINAL,
                             POSITIVE_OPERATION_MAGNITUDE_PARAMETERIZATION):
                with self.subTest(identity=identity):
                    with self.assertRaises(GE1CheckpointError):
                        load_ge1_checkpoint(
                            path,
                            expected_code_revision=revision,
                            expected_operation_magnitude_parameterization=(
                                identity
                            ),
                        )


@unittest.skipIf(torch is None, REASON)
class GridSoftmaxEndToEndTests(unittest.TestCase):
    """Both arms, the real teacher-forced path, and the assembled total."""

    def setUp(self):
        from prototype.constrained_profile_decoder import (
            profile_targets_for_loss,
        )
        from prototype.graph_encoder.batching import build_paired_batch
        from prototype.graph_encoder.tests.fixtures import procedural_fixture

        torch.manual_seed(DIAGNOSTIC_SEED)
        examples = tuple(
            procedural_fixture(name).physical
            for name in ("E", "R", "EE", "RE")
        )
        self.paired = build_paired_batch(examples)
        self.target = self.paired.target.to_torch(torch)
        flat = self.paired.flat_input.to_torch(torch)
        self.profiles = profile_targets_for_loss(
            self.paired.target, flat["geometry"]
        )

    def _teacher(self, model):
        from prototype.graph_encoder.autonomous import (
            autonomous_input_from_paired,
        )

        autonomous = autonomous_input_from_paired(
            self.paired, model.config.encoder, torch
        )
        return model.teacher_forced(
            autonomous.encoder_input, self.target, self.profiles
        )

    def test_both_arms_produce_five_wide_logits_and_an_additive_total(self):
        from prototype.graph_encoder.losses import GE1GridLoss, common_ge1_loss
        from prototype.graph_encoder.model import build_matched_ge1_models

        models = build_matched_ge1_models(
            seed=DIAGNOSTIC_SEED,
            operation_magnitude_parameterization=SOFTMAX,
        )
        for model in models:
            with self.subTest(arm=model.config.encoder):
                teacher = self._teacher(model)
                self.assertIsNotNone(teacher.grid_magnitude_logits)
                self.assertEqual(
                    tuple(teacher.grid_magnitude_logits.shape),
                    (
                        len(self.paired.family_ids),
                        self.target["node_type_ids"].size(1),
                        len(gm.OPERATION_TYPES),
                        gm.GRID_CLASS_COUNT,
                    ),
                )
                loss = common_ge1_loss(
                    teacher.decoder_output,
                    self.target,
                    self.profiles,
                    model.config,
                    grid_magnitude_logits=teacher.grid_magnitude_logits,
                )
                self.assertIsInstance(loss, GE1GridLoss)
                torch.testing.assert_close(
                    loss.total,
                    loss.base_loss.total
                    + sum(loss.grid_magnitude.weighted.values()),
                    rtol=0.0,
                    atol=0.0,
                )
                self.assertGreater(
                    sum(loss.grid_magnitude.active_operation_counts.values()), 0
                )

    def test_the_loss_reaches_both_the_head_and_the_shared_decoder_trunk(self):
        from prototype.graph_encoder.grid_magnitude_metrics import (
            engineering_gradient_norms,
        )
        from prototype.graph_encoder.losses import common_ge1_loss
        from prototype.graph_encoder.model import build_matched_ge1_models

        model = build_matched_ge1_models(
            seed=DIAGNOSTIC_SEED,
            operation_magnitude_parameterization=SOFTMAX,
        )[0]
        teacher = self._teacher(model)
        loss = common_ge1_loss(
            teacher.decoder_output,
            self.target,
            self.profiles,
            model.config,
            grid_magnitude_logits=teacher.grid_magnitude_logits,
        )
        model.zero_grad(set_to_none=True)
        sum(loss.grid_magnitude.weighted.values()).backward()
        norms = engineering_gradient_norms(model)
        for group in ("grid_head", "shared_decoder_trunk"):
            with self.subTest(group=group):
                self.assertTrue(norms["groups"][group]["finite"])
                self.assertTrue(norms["groups"][group]["nonzero"])

    def test_teacher_and_target_free_decode_agree_exactly(self):
        from prototype.graph_encoder.model import build_matched_ge1_models

        model = build_matched_ge1_models(
            seed=DIAGNOSTIC_SEED,
            operation_magnitude_parameterization=SOFTMAX,
        )[0]
        teacher = self._teacher(model)
        decoder = model.decoder
        states = teacher.decoder_output.decoded_states
        head = decoder.grid_magnitude_head
        from_logits = head.normalized_values(teacher.grid_magnitude_logits)
        recomputed = head.normalized_values(head(states))
        torch.testing.assert_close(
            from_logits, recomputed, rtol=0.0, atol=0.0
        )


if __name__ == "__main__":
    unittest.main()
