# ADR-0017: GE1 Exploratory Stage 6 at the Autonomous-Stop Identity

- Status: `accepted`
- Proposal date: `2026-08-20`
- Acceptance date: `2026-08-20`
- Owner: project research team
- Designated GE1 reviewer: Krishay Maskara
- Adds to: [ADR-0014](ADR-0014-ge1-stage6-structure-only-comparison.md)
  and [ADR-0016](ADR-0016-ge1-autonomous-stop-and-unconstrained-node-decoding.md)
- Superseded by: none

## Context

Stage 6 producer job `3354961` completed all six epoch-200 train runs but
stopped at the train-side reliability boundary before development. Subsequent
jobs `3355134` and `3355342` showed that target node count and exact-length
grammar masking allowed substantial structure reconstruction without encoder
memory. ADR-0016 removed both channels under
`GE1-PAD-TERMINATED-UNCONSTRAINED-NODES-v1`, and engineering-validation job
`3355790` passed at exact commit
`288327e42b8575f774b9df668df288d3fa71ae33`.

ADR-0016 made retraining conditional on a memory-to-count linear probe over
checkpoints that had never been trained to encode stopping. Job `3355776`
validly failed the all-checkpoint rule: typed-graph seed 2028 reached balanced
accuracy `0.751` from memory and `0.726` from prequant against the frozen
`0.90` floor. Those values and that failed result remain immutable.

The probe is a pessimistic proxy for a new objective rather than a validity
condition on the trained autonomous-stop system. Treating it as an absolute
precondition prevents the only direct test of whether learned stopping works.
This record therefore authorizes one explicitly exploratory lifecycle whose
comparison number is reportable but whose interpretation is necessarily
inconclusive.

## Decision

### Exploratory continuation is opt-in

The producer retains every train-side gate and remains blocking by default.
An explicit `exploratory_development_access=true` mode records the complete
train-side gate result and continues to development when optimization or
structural-memory gates fail. It does not change thresholds, scores, or the
default exit-42 behavior.

Both the producer artifact and final artifact record the exploratory marker in
their resolved configuration and manifest. The finalizer forces
`interpretation_category=inconclusive` and
`stage6_result_eligible=false` whenever the marker is true, even if every
ordinary validity gate happens to pass. The paired development effect and all
component evidence remain reported as exploratory measurements.

### The probe becomes informative

Job `3355776` is reclassified from a blocking resource prerequisite to an
informative diagnostic. Its result is not weakened, rerun, or relabelled as a
pass. The `0.751/0.726` typed-graph seed-2028 failure remains evidence that
stopping may be difficult and justifies the mandatory smoke. It no longer
prevents this one exploratory execution.

### Identity and isolation

The run explicitly selects
`GE1-PAD-TERMINATED-UNCONSTRAINED-NODES-v1`. Node-generation identity is a
validated CLI input recorded in resolved configuration, timing evidence, and
checkpoint identities. No implicit producer default may select it.

The magnitude head remains
`GE1-OPERATION-MAGNITUDE-GRID-ORDINAL-v1`. Geometry and magnitude are
report-only under the frozen Stage 6 score, and changing to ADR-0015 softmax in
the same run would confound the comparison. This additive combination uses the
autonomous-stop decoder/output/checkpoint contracts V4/v3/v4; it does not
alter either the ordinal identity or the softmax identity.

### Mandatory exact-commit timing

Timing-v2 must be rerun at the final implementation commit with the ordinal
magnitude and autonomous-stop node-generation identities. Generation now
continues until `<pad>` or the 16-node cap, so earlier requested-count timing
cannot be reused. The frozen timing selector alone chooses CPU versus `cuda:0`
and the three-seed versus two-seed fallback. The producer must honor that
choice without override.

If separately authorized after review of the validation, timing, and smoke
evidence, the real producer allocation will have four hours available. This
changes only the resource envelope; the epoch-200 scientific budget remains
fixed.

### Mandatory two-epoch lifecycle smoke

Before the real producer, run the full lifecycle with exactly two training
epochs through timing validation, model construction, training, strict
checkpoint recovery, autonomous train scoring, recorded train-gate failure,
exploratory continuation, development loading/scoring, checkpoint bundling,
producer verification, and finalization.

`smoke_protocol=true` and `protocol_final_epoch=2` appear in producer and
final artifacts. A smoke may be finalized only as a verified lifecycle-smoke
artifact with `stage6_result_eligible=false` and
`interpretation_category=inconclusive`; it can never be accepted as a Stage 6
result. Default and real execution retain epoch 200. No resume implementation
is added.

## Preserved rules

- The structural-memory thresholds, `0.10` development threshold,
  interpretation rule, seed-selection algorithm, real epoch-200 budget,
  family counts, batches, optimizer, clipping, and capacity bound are
  unchanged.
- Train-side gate failures remain explicit evidence. Exploratory mode changes
  only whether development is opened after recording them.
- RR, ER, IID, Stage 7, C8, CAD-kernel execution, and every other protected
  partition remain closed.
- `prototype/flat_baseline`, ADR-0013, ADR-0015, and their pinned runners are
  immutable.
- Outcome never controls artifact integrity, and exploratory status never
  upgrades a failed validity gate.

## Authorized sequence

This record authorizes, in order:

1. implementation and corpus-free exact-commit CPU validation;
2. one exact-commit train-only timing-v2 GPU allocation;
3. one two-epoch smoke using the authorized narrow train and development
   packages, followed by producer audit and smoke-only finalization;
4. stop and report the validation, timing selection, and finalized smoke to the
   designated reviewer; the epoch-200 command may be prepared but submission
   requires a separate reviewer decision; and
5. experiment records for each completed or failed step.

The designated reviewer explicitly withheld submission authority for the
epoch-200 producer on `2026-08-20`. This operational hold supersedes the
earlier prospective sequence: completing the smoke does not itself authorize
the real job, and implementation agents must not submit it.

The smoke and real run may access only the already governed 407-family train
and 45-family development packages. Neither run authorizes RR, ER, IID,
history-depth, geometry-extrapolation, CAD execution, Stage 7, C8, repair,
threshold changes, retry, or a confirmatory claim. Failure or timeout does not
authorize an additional scientific attempt.

## Consequences

The run can produce the first development comparison number at the learned-
stop identity, but it cannot resolve the Stage 6 hypothesis confirmatorily.
The explicit schema markers prevent later consumers from mistaking either the
smoke or the exploratory real artifact for an ordinary Stage 6 result.
