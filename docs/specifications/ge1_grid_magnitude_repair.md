# GE1 Grid-Anchored Ordinal Magnitude Implementation Contract

## Status and authority

| Item | Frozen value |
|---|---|
| Status | Implemented; corpus-free validation only; no scientific execution authorized |
| Protocol | `GE1-C7-GRID-MAGNITUDE-SUFFICIENCY-v1` |
| Parameterization | `GE1-OPERATION-MAGNITUDE-GRID-ORDINAL-v1` |
| Checkpoint schema | `GE1-CHECKPOINT-v2` |
| Authority | proposed [ADR-0013](../decisions/ADR-0013-ge1-grid-anchored-ordinal-operation-magnitude-repair.md) |
| Fidelity gate | unchanged |
| Stage 6 / C8 / repair execution | unauthorized / not begun / unauthorized |

This contract describes checked-in behaviour. It does not establish that any
model was trained, that any accessibility claim holds, or that the repair meets
the gate.

## Frozen grids

| Operation | Physical grid | Normalized grid | Channel | Scale |
|---|---|---|---:|---:|
| extrude | `0.5, 1.0, 1.5, 2.0, 3.0` | `0.125, 0.25, 0.375, 0.5, 0.75` | 37 | 4.0 |
| revolve | `45, 90, 180, 270, 360` | `0.125, 0.25, 0.5, 0.75, 1.0` | 38 | 360.0 |

`grid_magnitude` mirrors the physical grids and asserts exact equality with
`operation_fidelity` at import time, so the readout contract and the gate
cannot drift apart silently.

## Head

Two independent groups, one per grid, each a single shared projection over the
32-dimensional per-operation decoder state plus four biases built as
`b_1 = first_bias` and `b_k = b_{k-1} - softplus(gap_{k-1})`. Biases are
strictly decreasing, so cumulative logits are non-increasing for every input:
rank consistency is structural, not asserted. Projection weights retain
PyTorch's seeded nonzero `Linear` initialization, while gaps initialize to
zero. The nonzero projection is required so the first ordinal backward pass
has a nonzero derivative with respect to the shared decoder state; the common
model seed still makes both arms' copied decoder initialization identical.

Decoding counts positive cumulative logits, yielding an index in `[0, 4]`, and
emits exactly the frozen grid member at that index. The decode is discrete, so
no gradient flows through the geometry path; the head learns only through the
ordinal loss. Both grids are decoded for every node and the existing geometry
applicability mask exposes only the channel matching the autonomously generated
node type.

There is no residual in v1.

## Loss

```text
per_operation  = mean binary cross-entropy over four ordinal cuts
within_example = sum over that example's active operations of one type
across_batch   = mean over examples
total          = existing_total + 1.0 * L_extrude + 1.0 * L_revolve
```

`L = (1/B) * sum_e sum_{i in e} L_i = (1/B) * sum_i L_i`, so every active
operation has coefficient `1/B` independently of template. Raw and weighted
extrusion and revolve values, the active operation counts, the weights, and the
reduction record are reported separately.

Active magnitude targets are validated before use: an active operation node
whose magnitude channel is masked off, nonfinite, or off-grid raises a typed
error. Nothing is snapped. Labels are built inside the loss module only, are
detached, and never enter an autonomous interface.

## Identity and compatibility

| Parameterization | Decoder version | Output positions | Checkpoint schema |
|---|---|---|---|
| `...-TANH-LEGACY-v1` | `...-DECODER-V1` | `...-POSITIONS-v1` | `GE1-CHECKPOINT-v1` |
| `...-POSITIVE-v1` | `...-DECODER-V1` | `...-POSITIONS-v1` | `GE1-CHECKPOINT-v1` |
| `...-GRID-ORDINAL-v1` | `...-DECODER-V2` | `...-POSITIONS-v2` | `GE1-CHECKPOINT-v2` |

Historical configurations serialize byte-identically: the two grid-only weight
fields are omitted from `to_dict()` for non-grid identities. Cross-identity
checkpoint loading fails on the schema comparison before state-tensor
compatibility checks or application to a model.
No warm start and no partial state loading is provided.

Under the grid identity only, compact channels 4 and 5 are excluded from the
remaining-geometry loss mask, because the scalar magnitude outputs are unused
for decoding. Axis channels 0-3 are unaffected.

The real training loop passes `teacher.grid_magnitude_logits` to
`common_ge1_loss`. The logits are taken from the decoder states returned by the
single teacher-forced decoder call, so no second shared-decoder forward is
introduced. Grid identity without logits and a historical identity with logits
both raise typed errors. Finite-output validation traverses the entire
GE1-owned teacher result, including logits and prequant state.

`GE1GridLoss` is a GE1-owned wrapper. It retains the unmodified
`GraphV1Loss` as `base_loss`, adds the weighted ordinal terms to `total` and
`per_example`, and exposes `GridMagnitudeLossTerms`. Historical identities
return the original `GraphV1Loss` object shape and serialization.

## Diagnostics and autonomous metrics

The following additive schemas do not modify frozen Graph V1 metrics:

| Record | Version |
|---|---|
| teacher-forced loss diagnostics | `GE1-GRID-MAGNITUDE-DIAGNOSTICS-v1` |
| target/loss/metric applicability | `GE1-GRID-MAGNITUDE-APPLICABILITY-v1` |
| autonomous class metrics | `GE1-GRID-MAGNITUDE-AUTONOMOUS-METRICS-v1` |
| teacher/autonomous exact-value agreement | `GE1-GRID-MAGNITUDE-VALUE-AGREEMENT-v1` |
| synthetic gradient norms | `GE1-GRID-MAGNITUDE-GRADIENTS-v1` |

Where applicable they report raw and weighted losses, batch size, active
operation counts, four cut probabilities, target and decoded classes, exact
normalized and physical grid values, minimum absolute-logit decision margin,
per-type confusion matrices, true/predicted counts, per-class and extreme-class
recall, zero support and masking, exact teacher/autonomous grid agreement, and
head/shared-trunk gradient norms. Gradient records are explicitly marked
engineering-only and never constitute scientific training evidence.

Autonomous class scoring introduces the target only after complete generation.
Generated node type selects the applicable grid; target or label is absent from
the head, prefix decode, model forward, and autonomous evaluation interfaces.
Cut probabilities and margins are explicitly unavailable when only a serialized
autonomous prediction remains, rather than being fabricated.

## Axis and magnitude metric routing

The grid decoder's training output suppresses compact magnitude channels 4 and
5. `common_ge1_loss` separately clones the authoritative serialized target
mask and clears only channels 37 and 38 for the frozen scalar-loss call. It
does not mutate the target supplied to ordinal loss or later metrics.

Consequently:

- serialized axis channels 33-36 (compact 0-3) keep the historical scalar
  supervision and physical-MAE applicability count;
- serialized magnitudes 37-38 have scalar-loss and scalar-MAE counts of zero
  under the grid identity, but retain their true target applicability count and
  an equal ordinal-metric count;
- the grid-only scalar operation-parameter MAE is explicitly undefined with
  reason `routed_to_versioned_grid_ordinal_metric`, while separately retaining
  the nonzero target-channel denominator; the adjacent applicability and
  ordinal records carry the supported targets rather than calling them absent;
- historical calls omit the optional grid parameterization and therefore
  retain the exact previous metric record and mask behavior.

## Structural preservation

Node-type, categorical, graph-edge, dependency, profile, axis, Boolean-mode,
direction, grammar-mask, strict-conversion, analytic-validity, and fidelity-gate
behaviour are unchanged. The magnitude head is additive and reads the decoder
state; it cannot write to structural logits, which a runtime test asserts by
perturbing only the head and requiring bit-identical node-type logits.

## Validation boundary

Corpus-free only. Pure contract tests run anywhere; real-PyTorch tests require
the authoritative runtime and are expected to run on the engineering-validation
runner. No corpus, manifest, preserved payload, scientific checkpoint, model
artifact, protected partition, scientific training, scientific inference, or
Slurm submission is part of this task. The generated-tensor optimizer-step test
is explicitly engineering-only.

The focused module counts remain exactly 28 PyTorch-independent contract tests
and 14 real-PyTorch runtime tests. A separate integration module adds six
locally runnable structural tests plus real-PyTorch generated-tensor coverage
for both arms, the connected loss, first-step head/trunk gradients, historical
identity parity, target-free teacher/autonomous mapping, class reachability,
v2 round-trip and pre-load cross-identity rejection, axis/magnitude routing,
structural isolation, diagnostics, and one engineering-only optimizer step.
The checkpoint test creates and removes only its own temporary synthetic state;
it does not open an external, preserved, repaired, or scientific checkpoint.

The Adroit validation runner freezes the current regression discovery
contracts: model data 86/0 skips, flat baseline 549/the four established skips,
graph baseline 17/0, representation 40/0, and controlled data 51/0. It also
requires zero skips in the complete graph-encoder suite under Python 3.8.13 and
PyTorch 1.11.0. Its only container bind is the detached clean exact-commit
repository checkout, read-only. No scientific path is declared or mounted.

Passing these checks would establish engineering integrity only. It would not
establish accessibility, fidelity-gate success, scientific repair success, or
authorization for scientific execution, Stage 6, or C8.
