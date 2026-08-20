# ADR-0015: GE1 Grid Softmax Operation-Magnitude Classification

- Status: `accepted`
- Proposal date: `2026-08-19`
- Acceptance date: `2026-08-19`
- Owner: project research team
- Designated GE1 reviewer: Krishay Maskara
- Adds to: [ADR-0013](ADR-0013-ge1-grid-anchored-ordinal-operation-magnitude-repair.md)
- Supersedes: no frozen scientific choice, gate, partition, result, or prior
  identity
- Superseded by: none

## Context

Three successive protocols have failed on operation-magnitude fidelity.

1. C7-v2 job `3344981` failed with `invalid_operation_parameter`.
2. Repaired-sufficiency job `3345280` at commit
   `705eb820f7a15d64fe650df7350b1553b2bc8172` decoded flat 18/32 extrusions
   and 9/16 revolves, typed graph 17/32 and 4/16.
3. ADR-0013 grid-ordinal job `3352404` at commit
   `de80d93208a1a98196a38d2af5cd47afd525d14f` decoded every extrusion
   magnitude correctly, decoded both 90-degree revolve targets as 45 degrees,
   and failed the strict revolve-fidelity gate.

Corpus-free, checkpoint-free trajectory diagnostic job `3353008` at commit
`c9b519201bca80b4adfb03d5b7ad56d51ffe43b1` then exercised the unchanged real
ordinal head and loss on fixed generated decoder states. With zero raw bias
gaps, class 1 finished as class 0 in both operations and **classes 2, 3, and 4
never decoded above class 1 at any recorded step within 200 AdamW updates**.
The diagnostic-only `0.5` gap intervention corrected class 1 but not class 3.

### Diagnosis

The failure is a property of the ordinal parameterization, not of the grids,
the loss reduction, the tolerance, the corpus, or the optimizer budget.

`build_grid_magnitude_head` gives each operation type

- one `nn.Linear(model_dim, 1, bias=False)` producing a single scalar, and
- `first_bias` of shape `[2]` with `bias_gaps` of shape `[2, 3]`, combined by
  `ordered_biases()` into a strictly decreasing `[2, 4]` bias vector,

so `forward` returns cumulative logits `s + b_k` of shape `[..., 2, 4]` and the
decode is `(logits > 0).sum(dim=-1)`.

Every class decision therefore depends on **one shared scalar** compared
against **four global thresholds**. Three consequences follow.

- Class evidence cannot move independently. Raising the score for class 3
  necessarily raises it for classes 1 and 2, because they are the same number.
- Rank consistency is structurally enforced, so the head cannot represent
  non-monotone class evidence even transiently during optimization.
- The upper classes are jointly unreachable whenever the projection's dynamic
  range is small relative to the accumulated bias gaps. Reaching class 4
  requires `s` to exceed all four thresholds at once, and gradient pressure
  from the lower cuts opposes the growth in `s` that the upper cuts need.

This is ordinal regression, not classification. Job `3353008`'s observation
that classes 2–4 are simply never produced is the expected signature of that
structure, not evidence of an implementation defect.

### Prior art

DeepCAD (arXiv 2105.09492) quantizes continuous CAD command parameters and
predicts them with independent per-value classification. Its ablation reports
median Chamfer distance `0.752` for the quantized-classification formulation
against `2.142` for direct regression. The frozen five-value grids in this
project are already the quantization; only the readout differs.

## Decision

Adopt a prospective, **opt-in**, additively versioned five-way softmax
magnitude classification identity.

```text
GE1-OPERATION-MAGNITUDE-GRID-SOFTMAX-v1
GE1-GRID-SOFTMAX-MAGNITUDE-CONTRACT-v1
GE1-SHARED-TYPED-EDGE-DECODER-V3
GE1-CHECKPOINT-v3
```

The designated reviewer accepts the following as frozen v1 decisions.

### Frozen head

Two independent groups, one per grid. Each group is a single
`nn.Linear(model_dim, 5)` **with bias** over the existing per-operation decoder
state. `forward` returns independent class logits of shape `[..., 2, 5]`.

There is no shared scalar, no cut structure, no bias ordering, and no
structural rank consistency. Each class owns its own weight row and its own
bias, so class evidence moves independently by construction. This is the exact
property whose absence job `3353008` measured.

`nn.Linear`'s seeded nonzero initialization is retained for the same reason as
in ADR-0013: the first backward pass must have a nonzero derivative with
respect to the shared decoder state.

### Frozen decode

`class_indices(logits) = logits.argmax(dim=-1)`, and the decoded value is
exactly the frozen grid member at that index. The decode is discrete, so no
gradient flows through the geometry path; the head learns only through the
magnitude loss. Both grids are decoded for every node, and the existing
geometry applicability mask exposes only the channel matching the autonomously
generated node type, so no target operation type is consulted during
generation.

There is no residual in v1, for ADR-0013's reason: a correct class already
yields exactly zero error, and a residual would conceal a wrong class.

### Frozen loss and reduction

`torch.nn.functional.cross_entropy` over the five class logits against the
integer class index, with the surrounding reduction **unchanged from
ADR-0013**:

```text
per_operation  = cross_entropy(class_logits, class_index)
within_example = sum over that example's active operations of one type
across_batch   = mean over examples
total          = existing_total + 1.0 * L_extrude + 1.0 * L_revolve
```

Step 2 sums rather than averages, so `L = (1/B) * sum_i L_i` and every active
operation has coefficient `1/B` regardless of E, EE, R, or RE template. Weights
stay frozen at `1.0` for both operation types.

Only the per-operation term in step 1 changes: mean binary cross-entropy over
four cumulative cuts becomes cross-entropy over five independent class logits.
Steps 2, 3, and 4, the active-operation masking, the on-grid target assertion,
and the exclusion of axis and every other geometry channel are byte-identical
to ADR-0013.

### What is deliberately held constant

Isolating one variable is the point of this record.

- **Input geometry stays continuous** through `geometry_projection`.
  Quantizing the encoder input is a separate experiment and would confound
  this one.
- **Grid values are unchanged**: `(0.5, 1.0, 1.5, 2.0, 3.0)` and
  `(45.0, 90.0, 180.0, 270.0, 360.0)`.
- **No loss reweighting.** DeepCAD applies `beta = 2` to parameter losses.
  That is a second variable. It is recorded here as a candidate follow-up and
  is **not** applied.
- **No class balancing.**
- **No changes** to node, graph, edge, profile, or categorical heads; to
  either encoder; to the training loop; to Stage 6; or to the grammar.

### ADR-0013 remains immutable

`GE1-OPERATION-MAGNITUDE-GRID-ORDINAL-v1` keeps its exact head, loss, decode,
`GE1-SHARED-TYPED-EDGE-DECODER-V2`, `GE1-DECODER-OUTPUT-POSITIONS-v2`, and
`GE1-CHECKPOINT-v2`. `cumulative_labels`, `class_index_from_cumulative_flags`,
`ORDINAL_CUT_COUNT`, `build_grid_magnitude_head`, and
`_cumulative_label_tensor` are retained unchanged and remain the sole path for
that identity. `GE1-C7-GRID-MAGNITUDE-SUFFICIENCY-v1` continues to require the
ordinal parameterization and rejects any other.

Job `3352404`'s scientific failure, jobs `3352969`, `3353008`, `3351837`,
`3354961`, `3355134`, and `3355342`, and every prior result stand exactly as
recorded. Nothing here reinterprets them.

### Version-string rationale

`checkpoint_schema_for`, `shared_decoder_version_for`, and
`output_position_contract_version_for` all branch on `uses_grid_magnitude`.
Widening that predicate alone to cover both grid identities would hand the
softmax identity the ordinal identity's `GE1-CHECKPOINT-v2` schema, which would
let a governed schema check pass on a structurally incompatible state dict.
Therefore:

| Parameterization | Decoder version | Output positions | Checkpoint schema |
|---|---|---|---|
| `...-TANH-LEGACY-v1` | `...-DECODER-V1` | `...-POSITIONS-v1` | `GE1-CHECKPOINT-v1` |
| `...-POSITIVE-v1` | `...-DECODER-V1` | `...-POSITIONS-v1` | `GE1-CHECKPOINT-v1` |
| `...-GRID-ORDINAL-v1` | `...-DECODER-V2` | `...-POSITIONS-v2` | `GE1-CHECKPOINT-v2` |
| `...-GRID-SOFTMAX-v1` | `...-DECODER-V3` | `...-POSITIONS-v2` | `GE1-CHECKPOINT-v3` |

The output-position contract is **reused** at v2 because the decoder output
positions genuinely do not change between the two grid identities; only the
magnitude readout does. Reuse there is correct rather than lax.

### Checkpoint incompatibility

`GE1-OPERATION-MAGNITUDE-GRID-SOFTMAX-v1` is explicitly **checkpoint
incompatible** with `GE1-OPERATION-MAGNITUDE-GRID-ORDINAL-v1` and with both
historical scalar identities. State-dict keys and shapes differ:
`grid_magnitude_head.projections.*.weight` changes from `[1, model_dim]` to
`[5, model_dim]`, `grid_magnitude_head.projections.*.bias` is new, and
`grid_magnitude_head.first_bias` and `grid_magnitude_head.bias_gaps` do not
exist. Strict loading would fail on tensor shape; the schema comparison
refuses first, with a typed error, following the tanh/positive precedent. No
warm start, partial load, or key remapping is provided in either direction.

## Preservation guarantees

- The operation-geometry-fidelity gate is untouched: `< 0.25`, `< 22.5`,
  `strictly_less_than_unrounded`, every operation and representation variant.
- The frozen grids, `GRID_LABEL_TOLERANCE`, and
  `class_index_from_normalized_target` are reused unchanged; the softmax loss
  needs exactly the class index that function already produces.
- Existing configurations and every default are unchanged. The new identity is
  reachable only through the explicit `grid_softmax_frozen_encoder_config`
  constructor or an explicit parameterization argument.
- Historical `tanh` and positive-sigmoid configurations continue to serialize
  byte-for-byte, omitting the two grid-only weight fields.
- Frozen `prototype/graph_baseline` and `prototype/flat_baseline` source is
  unmodified; `GraphV1Output` is never extended and the class logits travel in
  the existing GE1-owned wrapper.
- The three focused ADR-0013 suites keep their pinned counts of 28, 14, and 18
  tests, so the committed exact-commit runners for jobs `3351837`, `3352969`,
  `3353008`, and the horizon diagnostic retain valid preflight contracts. New
  coverage is added in separate files rather than by mutating a pinned count.
- `test_grid_ordinal_horizon_*` and `test_grid_ordinal_trajectory_*` remain
  pinned to the ordinal identity. They are historical evidence about the old
  head and must keep testing the old head.

## Capacity

Per model, summed over both operation types:

| Identity | Parameters | Formula |
|---|---:|---|
| grid-ordinal | 72 | `2 * model_dim + 8` |
| grid-softmax | 330 | `2 * (5 * model_dim + 5)` |

with `model_dim = 32`. The shared decoder is common to both arms, so the flat
and typed-graph arms grow by the identical `+258`.

Measured whole-model trainable counts at seed `2026`:

| Identity | Flat | Typed graph | Relative difference | Gate |
|---|---:|---:|---:|---|
| `...-POSITIVE-v1` | 45,104 | 45,772 | 0.014810 | pass |
| `...-GRID-ORDINAL-v1` | 45,176 | 45,844 | 0.014787 | pass |
| `...-GRID-SOFTMAX-v1` | 45,434 | 46,102 | 0.014703 | pass |

The capacity-parity gate compares the two arms against each other with a 5%
inclusive tolerance. An identical additive change to a shared module leaves the
absolute arm difference at 668 and increases the denominator, so the relative
difference falls from `0.014787` to `0.014703`: the gate becomes marginally
easier, never harder.

## Authorization boundary

This record authorizes **implementation, corpus-free synthetic validation, and
documentation only**.

It does **not** authorize, and nothing in it may be read as authorizing:

- any scientific run, any Adroit submission, or any runner execution;
- Stage 6, Stage 7, C8, or any finalization;
- development, RR, ER, IID, history-depth, geometry-extrapolation,
  other-corpus, or protected-partition access;
- any corpus, manifest, payload, preserved feature payload, external or
  scientific checkpoint, existing model or repaired artifact, or CAD kernel;
- warm start, checkpoint reuse, or any repair beyond the head and loss change
  specified above;
- any change to the ordinal identity, to any frozen gate, or to any recorded
  result.

No test outcome, diagnostic, or engineering result produced under this record
automatically authorizes a scientific job, another repair, protected access,
Stage 6, or C8. A separate prospective authorization is required before any
execution.

## Corpus-free engineering-validation authorization

On August 20, 2026, designated reviewer Krishay Maskara prospectively
authorized **exactly one** corpus-free, checkpoint-free, training-free Adroit
CPU execution of

```text
prototype/graph_encoder/adroit/ge1_grid_softmax_magnitude_validation_cpu.slurm
```

at one exact implementation commit supplied as `EXPECTED_COMMIT` at submission
time. This authorization is recorded **before** submission, so the procedural
deviation retained against job `3351837` is not repeated.

The run may execute only: the environment assertion, the 63 focused softmax
tests, complete graph-encoder discovery under the six-ID CUDA skip allowlist,
the five sibling regression suites, documentation validation, `compileall`, the
Python 3.8 grammar gate, the identity and cut-count confinement audits, the
structural source audit, and the ADR-0013 runner-immutability diff.

It may **not** open any corpus, manifest, payload, preserved feature payload,
external or scientific checkpoint, model or repaired artifact, CAD kernel, or
protected partition, and performs no scientific or engineering training, no
inference, and no checkpoint write. The runner never submits itself.

Its outcome is engineering evidence about checked-in source on the production
Python 3.8.13 / PyTorch 1.11.0 stack. It is **not** an artifact-validity gate
and cannot establish any scientific claim. Success, failure, timeout, or
infrastructure error does not authorize a retry, a scientific job, another
repair, protected access, Stage 6, or C8.

### Environment-dependent expectations recorded prospectively

The runner asserts exact suite counts. Two of them are host-dependent and are
frozen here so a mismatch is a real regression rather than a surprise.

- `prototype/flat_baseline/tests`: 549 tests with **exactly 4 skips** on Adroit.
  The four are `test_categorical_isolation_replay` (3) and
  `test_constraint_manifold_replay` (1), all gated on the macOS-only path
  `/Users/krishaymaskara/research/audited-runs/...`, which cannot exist on a
  Linux compute node. The macOS-only `actual Linux no-replace primitive`
  skip does **not** occur on Adroit, because that primitive is available there.
- complete graph-encoder discovery: **exactly the six** `Stage6CudaRuntimeTests`
  CUDA-unavailable skips in `CPU_DISCOVERY_CUDA_SKIP_IDS`. `test_c6_runtime`
  does not skip, because the runner enforces a clean tree before discovery.

## Expectation setting

The gate requires 48/48 operations and 32/32 families. The measured defect is
that classes 2–4 are unreachable, so this change is expected to make every
class reachable; that is exactly what the added regression test asserts.
Reachability is a necessary condition for the gate, **not** a sufficient one.
A materially improved but still failing scientific result would be a valid
outcome and must not be described as a partial pass. Nothing here predicts or
claims a scientific outcome.
