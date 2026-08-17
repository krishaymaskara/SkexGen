# ADR-0013: GE1 Grid-Anchored Ordinal Operation-Magnitude Repair

- Status: `proposed`
- Proposal date: `2026-08-17`
- Owner: project research team
- Designated GE1 reviewer: Krishay Maskara
- Adds to: [ADR-0009](ADR-0009-ge1-positive-operation-magnitude-repair.md),
  [ADR-0010](ADR-0010-ge1-repaired-train-sufficiency-protocol.md),
  [ADR-0012](ADR-0012-ge1-closed-form-representation-readout.md)
- Supersedes: no frozen scientific choice, gate, partition, or prior identity
- Superseded by: none

## Context

Repaired-sufficiency job `3345280` at commit
`705eb820f7a15d64fe650df7350b1553b2bc8172` completed as an immutable valid
train-only scientific failure. Both arms passed exact node sequence, exact
graph, applicable dependencies, strict conversion, analytic validity, and the
memory-use gates, and both failed operation-geometry fidelity: flat 18/32
extrusions and 9/16 revolves; typed graph 17/32 and 4/16.

A read-only forensic reconstruction of that artifact established three facts.

1. **Variance compression, not bias.** Signed error reverses sign across the
   target range with near-zero net bias; prediction standard deviation is
   56-76% of target standard deviation; per-target pass rate is monotone in
   distance from the class-conditional centre. Under `beta = 1.0` every
   observed error lies in the Smooth-L1 quadratic region, so the objective is
   effectively squared error whose optimum under residual uncertainty is the
   conditional mean — a value between grid points, which a +/-0.0625
   normalized window rejects.
2. **Channel dilution.** Reconstructing the final `remaining_geometry` value
   as `0.5 * e^2` matches the observed loss only when the four axis channels
   are included in the per-example mean (observed / expected 1.15 flat, 1.11
   typed graph) and not when they are excluded (0.51, 0.42).
3. **No saturation and no implementation defect.** Recovered raw logits lie in
   `[-2.15, +1.73]`; the compact-to-serialized mapping, mask symmetry,
   normalization, and train/eval parameterization identity are all correct.

The remaining question is whether magnitude-class information is accessible in
the decoder state. Adroit job `3351524` is recorded as the completed
closed-form readout and, as reported by the designated reviewer, **did not
establish controlled magnitude-class accessibility from the tested
decoder-state features**. That evidence is reviewer-supplied; its artifact,
hashes, and per-feature outcomes are not yet cited in this repository and must
be attached before any scientific execution under this record.

## Decision

Adopt a prospective grid-anchored ordinal magnitude parameterization.

```text
GE1-OPERATION-MAGNITUDE-GRID-ORDINAL-v1
GE1-C7-GRID-MAGNITUDE-SUFFICIENCY-v1
GE1-CHECKPOINT-v2
```

### Why a null decoder-state readout does not refute this design

A readout measures a **frozen** decoder state produced by the historical
regression objective. The grid head is not a readout: it is trained jointly and
its ordinal gradient flows into the shared decoder trunk, so it *changes* the
state it reads. A null readout therefore constrains the *current* state, not
the achievable one.

This is the load-bearing assumption of the record and it is stated as an
assumption, not a finding. It implies two non-negotiable implementation
properties, both tested:

- the ordinal loss must reach the shared trunk (no detached readout); and
- class-only decoding must remain the sole decode path in v1, so a wrong class
  is a visible hard miss rather than a concealed near-miss.

If a later diagnostic shows class information is absent at the decoder state
*and* present at the prequant bottleneck, an auxiliary prequant-level term
becomes the indicated next step. If it is absent at both, this record stops and
the question returns to the reviewer rather than escalating to an encoder
change.

### Frozen head

Two independent rank-consistent ordinal groups, one per grid. Each group owns a
single shared projection over the existing 32-dimensional per-operation decoder
state plus four biases that are **strictly decreasing by construction**
(`b_1`, then `b_k = b_{k-1} - softplus(delta_{k-1})`). Cumulative logits are
therefore non-increasing for every input and rank consistency cannot be
violated numerically rather than merely by convention. The class index is the
count of positive cumulative logits, and the decoded value is exactly the
frozen grid member at that index.

Both grids are decoded for every node; the existing geometry applicability mask
exposes only the channel matching the autonomously generated node type, so no
target operation type is consulted during generation. No residual is
implemented in v1: a correct class already yields exactly zero error, and a
residual would conceal a wrong class.

### Frozen loss and reduction

A GE1-level term computed only from the ordinal head and operation node types.
Axis and every other geometry channel are excluded by construction.

1. mean binary cross-entropy over the four ordinal cuts of one operation;
2. **sum** those per-operation losses within each example, per operation type;
3. mean the per-example sums across the batch;
4. add separately reported extrusion and revolve terms, weights frozen at
   `1.0`.

Step 2 sums rather than averages. The total is `(1/B) * sum over all active
operations`, so `dL/dL_i = 1/B` for every active operation regardless of E, EE,
R, or RE template. An earlier draft of this plan specified a within-example
mean; that was wrong and would have given an `EE` operation half the
coefficient of an `E` operation, reproducing the dilution the term exists to
remove. No class balancing is applied in v1.

### Consequence for the historical scalar supervision

Under the grid identity the scalar magnitude outputs are unused for decoding.
Compact channels 4 and 5 are therefore excluded from the existing
remaining-geometry mask **for this identity only**, so the trunk is not
simultaneously pulled toward a conditional-mean scalar. Axis channels 0-3 keep
their existing supervision, and both historical identities are byte-identical.

## Preservation guarantees

- The operation-geometry-fidelity gate is untouched: `< 0.25`, `< 22.5`,
  `strictly_less_than_unrounded`, every operation and representation variant.
- `GE1-OPERATION-MAGNITUDE-TANH-LEGACY-v1` and
  `GE1-OPERATION-MAGNITUDE-POSITIVE-v1` retain their exact numerical behaviour,
  their `GE1-CHECKPOINT-v1` schema, and byte-identical configuration
  serialization.
- Frozen `prototype/graph_baseline` and `prototype/flat_baseline` source is
  unmodified; `GraphV1Output` is never extended and the new logits travel in a
  GE1-owned wrapper.
- Jobs `3344981`, `3345013`, `3345280`, `3346513`, `3350017`, and `3351524`
  are neither modified nor reinterpreted.

## Authorization boundary

This record authorizes implementation, corpus-free synthetic testing, and an
unsubmitted exact-commit Adroit engineering-validation runner only. It does not
authorize scientific execution, Slurm submission, corpus or manifest access,
external or scientific checkpoint loading, scientific training, Stage 6, C8,
CAD-kernel use, or any accessibility claim. A temporary generated-state
round-trip is permitted only as the checkpoint-identity unit test and is
removed by that test. No result produced under this record may authorize a
later stage.

## Implementation completion record

The corpus-free implementation is complete in the working tree, but the ADR
remains proposed and no engineering or scientific run is claimed here.

- The real common training loop passes the ordinal logits already computed by
  `GE1Model.teacher_forced` into `common_ge1_loss`; no second decoder forward is
  performed. Finite-output traversal covers the complete GE1 teacher wrapper,
  including the logits. Seeded nonzero projection initialization gives the
  ordinal term an immediate gradient path into the shared decoder trunk.
- `GE1GridLoss` preserves the frozen `GraphV1Loss` object as `base_loss` and
  adds structured raw, weighted, per-example, count, probability, class,
  grid-value, margin, confusion, support, recall, masking, agreement, and
  synthetic-gradient evidence under explicit GE1 schema versions. Historical
  identities still return the original `GraphV1Loss` and its original record.
- Only the local target-mask view passed to the scalar Graph V1 loss clears
  serialized magnitude channels 37 and 38. The authoritative target mask is
  not mutated. Axis channels 33-36 retain their historical scalar supervision
  and physical-MAE metric; supported magnitudes are counted and scored in the
  ordinal diagnostic rather than being reported as absent.
- A separate synthetic integration suite exercises both encoder arms, loss and
  gradient flow, target-free decode parity, all grid classes, exact values,
  v2 checkpoint isolation, mask routing, structural isolation, diagnostics,
  and an explicitly engineering-only optimizer step. The pre-existing focused
  modules remain exactly 28 pure tests and 14 real-PyTorch tests.
- The source audit used by both the integration tests and the runner parses
  Python imports and signatures with AST and inspects executable shell after
  removing comments and heredoc bodies. Documentation words cannot trigger it,
  while a real submission command, scientific path, extra bind, target-bearing
  grid interface, label derivation outside the loss, or scientific entry point
  remains a failure.
- The strengthened one-hour CPU runner uses a detached clean exact-commit
  standalone checkout, the repository as its sole read-only bind, all five
  established regression suites with exact count/skip contracts, documentation
  and syntax validation, committed-whitespace and preservation checks, and
  terminal telemetry. Its log parent must exist before submission; the runner
  neither creates that parent nor submits itself.

These are prerequisites for later engineering validation. They do not show
decoder-state accessibility, repair success, 48/48 fidelity, or scientific
authorization.

## Open decisions for the reviewer

1. Attach the job `3351524` artifact and per-feature outcomes before execution.
2. Confirm ordinal rather than plain five-way softmax for v1.
3. Confirm the deferral of the residual to a possible v2.
4. Confirm equal `1.0` weights and the train-only tuning rule.
5. Confirm the grid-identity exclusion of compact channels 4 and 5 from the
   scalar remaining-geometry mask.

## Expectation setting

The gate requires 48/48 operations and 32/32 families. Current per-operation
accuracy is 56% (flat) and 44% (typed graph). This repair changes the
estimator's optimum and is expected to improve those materially; it is **not**
expected on present evidence to reach 48/48 in one step, and a materially
improved but still failing result is a valid outcome that must not be described
as a partial pass.
