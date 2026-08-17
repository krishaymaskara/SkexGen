# ADR-0012: GE1 Closed-Form Representation Readout

- Status: `accepted`
- Proposal date: `2026-08-16`
- Decision date: `2026-08-16`
- Owner: project research team
- Designated GE1 reviewer: Krishay Maskara
- Adds to: [ADR-0011](ADR-0011-ge1-repaired-representation-probe.md)
- Preserves as accepted and incomplete: ADR-0011 protocol
  `GE1-C7-REPAIRED-REPRESENTATION-PROBE-v1`
- Preserves as separate and incomplete: representation screen
  `GE1-C7-REPAIRED-REPRESENTATION-SCREEN-v1`
- Superseded by: none

## Context

ADR-0011 remains accepted and incomplete. Its authoritative job `3346513`
timed out after preserving the complete detached target-free feature and label
inputs, but before producing probe results. A narrower 16-pipeline iterative
screen was then prepared. Its job `3350017`, at exact commit
`8a1ce734abfdbdb7468d6900a0f34a4b914608b5`, also timed out.

The job-`3350017` transcript records state `TIMEOUT`, elapsed `01:00:15`, batch
elapsed `01:00:16`, batch state `CANCELLED`, batch exit `0:15`, and MaxRSS
`622608K`. Its focused 12/12 tests and 361/361 graph-encoder tests passed. The
model-data, flat-baseline, graph-baseline, representation, and controlled-data
regression suites completed with 86, 549, 17, 40, and 51 tests respectively;
the flat suite had its four expected skips. Documentation, compilation, source
audit, and import/export audit passed. The incomplete artifact directory
existed but was empty, so no finalized screen result or scientific conclusion
exists.

The inputs had been staged and hash-verified before submission, and the empty
staging directory proves that the module began after preflights. The module
creates staging before loading inputs, however, and Slurm killed the job
without terminal telemetry. Preserved-input content access is therefore
`indeterminate_not_terminally_certified`, not false. Its runner mounts and
source boundary nevertheless establish that no corpus, checkpoint, GE1 model,
training, repair, protected partition, Stage 6, or C8 work was available.
Job `3350017` is not an ADR-0011 result.

The iterative optimizers and repeated nested refits were inappropriate for a
one-hour allocation. The same observational question can be asked with a
fully specified closed-form estimator whose feature preprocessing and matrix
decompositions are independent of labels.

## Decision

The project adopts this additive artifact-only diagnostic:

```text
GE1-C7-CLOSED-FORM-READOUT-v1
GE1-C7-CLOSED-FORM-READOUT-ARTIFACT-v1
GE1-C7-CLOSED-FORM-READOUT-RESULTS-v1
```

The complete scientific and artifact contract is frozen in the
[closed-form readout specification](../specifications/ge1_closed_form_representation_readout.md).
This ADR changes the diagnostic estimator, not a GE1 model or the scientific
status of earlier work.

### Why closed-form ridge

Five-output ridge has a deterministic global solution, retains all five output
classes even when an inner fold lacks one, and supports an unpenalized
intercept exactly. It removes iterative convergence and initialization as
runtime variables while retaining nested physical-family LOFO selection.
The objective is

```text
(1 / (2n)) * ||Y - XW - b||_F^2 + (lambda / 2) * ||W||_F^2
```

with float64 CPU arithmetic and the seven-value grid from `1e-4` through
`100`. Each train-fold feature transform is fitted without test-fold access.

The thin SVD of a transformed training matrix depends only on X. Reusing it
across every lambda and across true and permuted one-hot right-hand sides is
the same algebraic ridge solution, not approximate fitting, omitted
preprocessing, or reuse of a label-dependent choice. Nested lambda selection
is rerun for every permuted target.

### Why slot-gated prequant

The family-level prequant P is identical for both operations in an EE family.
Appending a slot bit to one linear model cannot apply different P weights to
slots 0 and 1. Extrusion therefore uses primary D-gated features
`[P*slot_0, P*slot_1, slot_0, slot_1]`, allowing separate linear mappings.
D-additive `[P, slot_0, slot_1]` is a secondary structural control, and
slot-only context is a negative/context control. D-gated pays an explicit
power cost: each 32-column block has about half the extrusion support, and an
outer fold has roughly 30 operations against 66 nominal columns.

Every revolve occurs at slot 0. Context-only is constant, while D-additive and
D-gated reduce to P after zero-variance removal. Those three revolve variants
are recorded as `degenerate_by_construction` and are neither fit nor counted as
tests. P is the primary revolve upstream feature.

### Primary and secondary hypotheses

There are exactly 16 primary hypotheses, with raw and balanced accuracy kept
as separate metric families. Per arm, extrusion tests C accessibility,
D-gated accessibility, C minus grouped A/B, and D-gated minus C. Per arm,
revolve tests C accessibility, P accessibility, C minus grouped A/B, and P
minus C. The global full cohort is primary.

Extrusion P, D-additive, context-only, resubstitution, masking diagnostics, and
the E+RE one-extrusion-per-family sensitivity cohort are secondary. Revolve
has no redundant sensitivity cohort because its full 16-family cohort is
already one-to-one.

### Permutations and multiplicity

Family-label blocks are permuted within operation template and operation type.
The ordered EE two-extrusion block is indivisible. Seeds contain only protocol
identity, fixed seed 2026, operation type, and permutation index, so every arm
and feature uses the identical labels. Extrusion and revolve permutations are
paired by index. Every difference is recomputed between synchronized permuted
pipelines.

Within raw and balanced accuracy separately, a single centered maximum
statistic spans all 16 primary hypotheses. For each hypothesis the null is
centered by its own mean; each permutation takes the maximum across all 16;
the adjusted plus-one empirical p-value compares the observed centered value
with that global maximum. Step-down correction and kernel ridge are excluded
from v1.

### Runtime decision

After repository preflights but before any preserved-input mount, a synthetic
preflight exercises the real cohort/fold cardinalities, every feature
dimension, nested lambda selection, batched permutation targets, metrics, and
maxT bookkeeping. Projection includes already elapsed preflight time, three
minutes for finalization/verification, and a 1.25 safety factor.

The only permitted ladder is: choose B=999 when the projected complete job is
at most 45 minutes; otherwise choose B=499 when that projection is at most 45
minutes; otherwise abort before preserved-input access. No other B is valid.

## Interpretation limits

Controlled accessibility requires balanced LOFO accuracy at least 0.40, raw
accuracy above its marginal permutation 95th percentile, and both raw and
balanced global maxT-adjusted p-values at most 0.05. A C-minus-A/B or
upstream-minus-C claim additionally needs raw and balanced margins of at least
0.10 and adjusted significance in both metrics, with the prerequisite
accessibility pattern frozen by the specification.

A positive C-minus-A/B result licenses only the statement that magnitude-class
information reaches the operation decoder state but is not exploited by the
existing scalar path. A positive upstream-minus-C pattern identifies the
transformation or conditioning between prequant and C as a candidate failure
region; it does not prove destruction because accessibility may have become
nonlinear. D-gated success after D-additive failure identifies an additive
readout mismatch, not a decoder defect. Arm asymmetry is arm-specific
accessibility, not encoder superiority.

Ridge masking is reported whenever a supported class receives no predictions.
A negative result with masking cannot prove lack of linear accessibility. If
all primary tests are null, the only licensed conclusion is that no controlled
held-out-family accessibility evidence was found under these readouts.
Resubstitution is non-generalizing fit or memorization only.

This observational diagnostic cannot identify an exact code defect, prove
information absence or capacity exhaustion, prove that `latent_tokens=2` or
`bottleneck_dim=16` binds, predict a categorical head's 48/48 gate, establish
encoder superiority, or authorize an intervention.

## Access and authorization boundary

Authoritative execution may read only the three immutable detached-feature
files and their exact hashes named in the specification. The runner mounts no
corpus, GE1 manifest, checkpoint, repaired artifact, model artifact, or
protected partition. It constructs no GE1 model and performs no GE1 inference,
training, backward pass, optimizer step, checkpoint write, or repair.

This acceptance authorizes implementation, synthetic corpus-free validation,
an unsubmitted one-hour CPU runner, and local exact-source preparation. It does
not authorize local or Slurm scientific execution, submission, pushing,
repair, a categorical head, loss/model/latent changes, Stage 6, C8, or
protected access. A later exact execution still requires separate authority.

Krishay Maskara, acting as designated GE1 reviewer, accepted this additive ADR
and its linked specification on August 16, 2026 under those limits.

## Alternatives considered

### Retry either iterative readout

Rejected. Two one-hour timeouts show that iterative nested fitting is not a
reliable one-hour workflow.

### Add LDA

Rejected. With n smaller than d and potentially absent classes in inner folds,
LDA needs a separate shrinkage, prior, and missing-class contract.

### Add outcome-triggered kernel ridge

Rejected. Inspecting linear outcomes before creating a nonlinear family would
make the test family outcome-dependent. Kernel work requires an additive
preregistration.

### Treat D-additive as the only prequant readout

Rejected. It cannot apply slot-specific mappings to the repeated EE prequant.

## Consequences

- ADR-0011 remains accepted and incomplete.
- The representation screen remains separate and incomplete.
- Jobs `3346513` and `3350017` retain their timeout status and no scientific
  interpretation.
- Complete negative results finalize with exit code zero.
- Exactly five final artifact files are created atomically; incomplete staging
  is never authoritative.
- Any model change or nonlinear follow-up requires another decision.

## Related records

- [ADR-0011](ADR-0011-ge1-repaired-representation-probe.md)
- [Closed-form readout specification](../specifications/ge1_closed_form_representation_readout.md)
- [Representation-screen specification](../specifications/ge1_repaired_representation_screen.md)
- [Current project status](../status.md)
