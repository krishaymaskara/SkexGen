# GE1 Grid Softmax Magnitude Implementation Contract

## Status and authority

| Item | Frozen value |
|---|---|
| Status | Accepted; implementation and corpus-free synthetic validation only |
| Parameterization | `GE1-OPERATION-MAGNITUDE-GRID-SOFTMAX-v1` |
| Contract version | `GE1-GRID-SOFTMAX-MAGNITUDE-CONTRACT-v1` |
| Shared decoder version | `GE1-SHARED-TYPED-EDGE-DECODER-V3` |
| Output-position version | `GE1-DECODER-OUTPUT-POSITIONS-v2` (reused) |
| Checkpoint schema | `GE1-CHECKPOINT-v3` |
| Authority | accepted [ADR-0015](../decisions/ADR-0015-ge1-grid-softmax-operation-magnitude-classification.md), Krishay Maskara, August 19, 2026 |
| Fidelity gate | unchanged |
| Scientific execution / Stage 6 / C8 | unauthorized / unauthorized / not begun |
| Opt-in | yes; no existing configuration or default changes |

This contract describes checked-in behaviour. It does not establish that any
model was trained, that any accessibility claim holds, or that the change meets
the gate.

## Frozen grids

Identical to [ADR-0013](../decisions/ADR-0013-ge1-grid-anchored-ordinal-operation-magnitude-repair.md)
and reused, not redefined.

| Operation | Physical grid | Normalized grid | Channel | Scale |
|---|---|---|---:|---:|
| extrude | `0.5, 1.0, 1.5, 2.0, 3.0` | `0.125, 0.25, 0.375, 0.5, 0.75` | 37 | 4.0 |
| revolve | `45, 90, 180, 270, 360` | `0.125, 0.25, 0.5, 0.75, 1.0` | 38 | 360.0 |

`grid_magnitude` still asserts exact equality with `operation_fidelity` at
import time, so the readout contract and the gate cannot drift apart silently.

## Head

`build_grid_softmax_magnitude_head(model_dim)` constructs
`GridSoftmaxMagnitudeHead`: one `nn.Linear(model_dim, 5)` **with bias** per
operation type, held in an `nn.ModuleList` in `OPERATION_TYPES` order.

```text
forward(decoded_states) -> [..., 2, 5]   independent class logits
class_indices(logits)   -> [..., 2]      logits.argmax(dim=-1)
class_probabilities     -> [..., 2, 5]   softmax(logits, dim=-1)
normalized_values       -> [..., 2]      exact frozen grid member per index
```

There is no shared scalar, no cumulative cut, no bias ordering, and no
structural rank consistency. Each of the five classes owns a distinct weight
row and bias, so class evidence moves independently.

`nn.Linear`'s seeded nonzero initialization is retained: the first backward
pass must have a nonzero derivative with respect to the shared decoder state,
so the head is connected to the trunk from the first update.

Decoding takes the argmax and emits exactly the frozen grid member at that
index. The decode is discrete, so no gradient flows through the geometry path;
the head learns only through the classification loss. Both grids are decoded
for every node, and the existing geometry applicability mask exposes only the
channel matching the autonomously generated node type.

There is no residual in v1.

### Contract metadata differences

`grid_contract_metadata(parameterization)` returns, for the softmax identity:

| Key | Value |
|---|---|
| `version` | `GE1-GRID-SOFTMAX-MAGNITUDE-CONTRACT-v1` |
| `parameterization` | `GE1-OPERATION-MAGNITUDE-GRID-SOFTMAX-v1` |
| `decode_rule` | `argmax_over_class_logits` |
| `rank_consistency` | `none_independent_class_logits` |
| `ordinal_cut_count` | absent |
| `class_balancing_applied` | `False` |

Called with no argument, or with the ordinal identity, it returns the ADR-0013
record byte-for-byte, including `ordinal_cut_count` and the ordinal
`decode_rule` and `rank_consistency` strings.

## Loss

```text
per_operation  = cross_entropy(class_logits[..., 5], class_index)
within_example = sum over that example's active operations of one type
across_batch   = mean over examples
total          = existing_total + 1.0 * L_extrude + 1.0 * L_revolve
```

`L = (1/B) * sum_e sum_{i in e} L_i = (1/B) * sum_i L_i`, so every active
operation has coefficient `1/B` independently of template. Steps 2, 3, and 4
are byte-identical to the ordinal contract; only step 1 changes.

Labels are integer class indices of dtype `long`, built by
`_class_index_label_tensor` inside `losses.py` only, detached, with inactive
positions set to the ignore index so they contribute nothing. Active magnitude
targets are validated before use by the unchanged
`_assert_active_targets_are_on_grid`: an active operation node whose magnitude
channel is masked off, nonfinite, or off-grid raises a typed error. Nothing is
snapped.

The ordinal path keeps `_cumulative_label_tensor` and
`binary_cross_entropy_with_logits` exactly as before. `grid_magnitude_terms`
selects on `config.operation_magnitude_parameterization`.

No class balancing and no loss reweighting are applied. DeepCAD's `beta = 2`
parameter weighting is recorded in ADR-0015 as a candidate follow-up and is
deliberately not applied here, because it would be a second variable.

## Diagnostics

`GridMagnitudeLossTerms` keeps every field and every returned key, so
downstream metrics and artifacts do not shift shape. Two fields are
reinterpreted per identity and the record states which interpretation applies.

| Field / key | Ordinal | Softmax |
|---|---|---|
| `cut_probabilities` / `ordinal_cut_probabilities` | 4 sigmoid cut probabilities | 5 softmax class probabilities |
| `decision_margins` / `decision_margin` | `min` absolute cut logit | top-1 minus top-2 class logit |

Two keys are **added** to the softmax diagnostic record, never removed or
reordered: `parameterization` and `decode_rule`. They are added only under the
softmax identity, so the ADR-0013 record stays byte-identical. Confusion
matrices, per-class recall,
extreme-class recall, support flags, masking flags, grid values, counts, and
the reduction record are unchanged.

`grid_magnitude_metrics.py` needs no change: it scores decoded grid classes
and never inspects logit shape.

## Identity and compatibility

| Parameterization | Decoder version | Output positions | Checkpoint schema |
|---|---|---|---|
| `...-TANH-LEGACY-v1` | `...-DECODER-V1` | `...-POSITIONS-v1` | `GE1-CHECKPOINT-v1` |
| `...-POSITIVE-v1` | `...-DECODER-V1` | `...-POSITIONS-v1` | `GE1-CHECKPOINT-v1` |
| `...-GRID-ORDINAL-v1` | `...-DECODER-V2` | `...-POSITIONS-v2` | `GE1-CHECKPOINT-v2` |
| `...-GRID-SOFTMAX-v1` | `...-DECODER-V3` | `...-POSITIONS-v2` | `GE1-CHECKPOINT-v3` |

`uses_grid_magnitude` is membership over
`GRID_OPERATION_MAGNITUDE_PARAMETERIZATIONS`, so both grid identities share the
grid loss routing, the grid-only configuration fields, and the exclusion of
compact channels 4 and 5 from the scalar remaining-geometry mask.
`checkpoint_schema_for` and `shared_decoder_version_for` branch on the exact
identity instead, so the softmax identity can never be handed the ordinal
schema or decoder version. `output_position_contract_version_for` reuses the
grid v2 value, because output positions genuinely do not change.

`OPERATION_MAGNITUDE_PARAMETERIZATIONS` appends the softmax identity, so no
existing tuple index or prefix moves.

### Checkpoint incompatibility

The softmax identity is checkpoint incompatible with every other identity.

| State-dict key | Ordinal | Softmax |
|---|---|---|
| `grid_magnitude_head.projections.N.weight` | `[1, model_dim]` | `[5, model_dim]` |
| `grid_magnitude_head.projections.N.bias` | absent | `[5]` |
| `grid_magnitude_head.first_bias` | `[2]` | absent |
| `grid_magnitude_head.bias_gaps` | `[2, 3]` | absent |

Strict loading would fail on shape. The governed schema comparison refuses
first with a typed error, following the tanh/positive precedent. No warm start,
partial load, or key remapping is provided in either direction.

## Configuration

`grid_softmax_frozen_encoder_config(encoder, seed)` mirrors
`grid_frozen_encoder_config` and is the only convenience constructor for the
new identity. Every default is unchanged: `GE1Config` still defaults to
`GE1-OPERATION-MAGNITUDE-POSITIVE-v1`, and `SharedGE1Decoder` still defaults to
the same. The identity is reachable only by passing it explicitly.

Grid-only serialized fields (`grid_magnitude_extrude_loss_weight`,
`grid_magnitude_revolve_loss_weight`) appear for both grid identities and stay
omitted for the two historical ones, so historical serialization remains
byte-identical.

## Capacity

`model_dim = 32`, summed over both operation types.

| Identity | Head parameters | Formula |
|---|---:|---|
| grid-ordinal | 72 | `2 * 32 + 8` |
| grid-softmax | 330 | `2 * (5 * 32 + 5)` |

The decoder is shared, so both arms change by the identical `+258` relative to
the ordinal identity. Measured whole-model trainable counts at seed `2026`:

| Identity | Flat | Typed graph | Difference | Relative | Gate |
|---|---:|---:|---:|---:|---|
| `...-POSITIVE-v1` | 45,104 | 45,772 | 668 | 0.014810 | pass |
| `...-GRID-ORDINAL-v1` | 45,176 | 45,844 | 668 | 0.014787 | pass |
| `...-GRID-SOFTMAX-v1` | 45,434 | 46,102 | 668 | 0.014703 | pass |

The capacity-parity gate compares the flat and typed-graph arms against each
other with a 5% inclusive tolerance. The absolute arm difference stays at 668
and the denominator grows, so the relative difference falls and the gate cannot
become harder. `GridSoftmaxCapacityTests` asserts the head counts, the equal
`+258` per-arm delta, and that `capacity_gate` still passes for both grid
identities.

## Isolation

`ORDINAL_CUT_COUNT` and `GRID_CLASS_COUNT` remain confined to
`grid_magnitude.py` and `losses.py`. No Stage 6, producer, checkpoint-identity,
metric, or audit module references either, so no logit-shape assumption
propagates outward. This is asserted structurally by
`GridSoftmaxIsolationTests`.

`grid_magnitude_sufficiency.py` continues to require the ordinal
parameterization and raises `legacy_checkpoint_ineligible` for anything else,
including the softmax identity. `GE1-C7-GRID-MAGNITUDE-SUFFICIENCY-v1` is an
ADR-0013 protocol and is not extended here.

## Tests

| Suite | Count | Identity under test |
|---|---:|---|
| `test_grid_magnitude_contract` | 28 | ordinal, pinned |
| `test_grid_magnitude_runtime` | 14 | ordinal, pinned |
| `test_grid_magnitude_integration` | 18 | ordinal, pinned |
| `test_grid_ordinal_trajectory_*` | pinned | ordinal, historical evidence |
| `test_grid_ordinal_horizon_*` | pinned | ordinal, historical evidence |
| `test_grid_softmax_magnitude_contract` | 27 | softmax |
| `test_grid_softmax_magnitude_runtime` | 36 | softmax |

New coverage lives in new files. The three ADR-0013 focused suites keep their
pinned counts of 28, 14, and 18, because
`prototype/graph_encoder/grid_ordinal_horizon_audit.py` and three committed
exact-commit Slurm runners assert those literals as preflight contracts for
already-completed jobs. Mutating them would invalidate historical runners.

### The regression that this change exists to fix

`GridSoftmaxReachabilityTests.test_every_class_becomes_reachable_within_budget`
mirrors trajectory diagnostic job `3353008`. On a synthetic fixture with all
five classes represented for both operation types, it trains the head alone for
a bounded number of AdamW updates and asserts that **every one of the five
class indices is decoded at least once**. The same test parameterized on the
ordinal head fails: classes 2, 3, and 4 are never produced, reproducing job
`3353008`'s finding in-suite.

## Runner

```text
prototype/graph_encoder/adroit/ge1_grid_softmax_magnitude_validation_cpu.slurm
```

A corpus-free, checkpoint-free, training-free one-hour-class CPU validation
runner. It takes one exact `EXPECTED_COMMIT` at submission, binds the
repository read-only exactly once, and never submits itself. It uses the
six-ID CUDA skip allowlist validator
(`grid_magnitude_audit.validate_cpu_softmax_complete_discovery`) rather than a
literal zero-skip gate, following the pattern commit `9c18b43` established
after job `3354930` failed on the stale gate.

Beyond the mechanical template it additionally audits the identity resolvers,
`ORDINAL_CUT_COUNT` confinement, and ADR-0013 runner immutability by diffing
the three ordinal runners against the parent commit.

## Non-authorization

Implementation, synthetic validation, documentation, and exactly one
corpus-free engineering-validation run authorized by
[ADR-0015](../decisions/ADR-0015-ge1-grid-softmax-operation-magnitude-classification.md).
No scientific run, no Stage 6, no C8, no development, RR, ER, or protected
access, no corpus, manifest, payload, checkpoint, model artifact, or CAD
kernel. No test outcome here authorizes any of those.
