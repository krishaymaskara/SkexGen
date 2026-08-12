# ADR-0011: GE1 Repaired-Checkpoint Representation Probe

- Status: `accepted`
- Proposal date: `2026-08-12`
- Decision date: `2026-08-12`
- Owner: project research team
- Designated GE1 reviewer: Krishay Maskara
- Adds to: [ADR-0010](ADR-0010-ge1-repaired-train-sufficiency-protocol.md)
- Preserves without reinterpretation: repaired-sufficiency commit
  `705eb820f7a15d64fe650df7350b1553b2bc8172`, Adroit job `3345280`
- Superseded by: none

## Context

The immutable repaired-sufficiency execution completed as a valid train-only
scientific failure. Both encoder arms passed exact node, graph, dependency,
strict-conversion, analytic-validity, and memory-use gates on the 32-family
scaled train cohort. The positive operation-magnitude parameterization removed
the earlier nonpositive operation predictions, but both arms failed the new
operation-geometry fidelity gate.

The remaining evidence does not distinguish two explanations:

1. magnitude-class information is present in the encoder memory or shared
   decoder state but is discarded by the existing scalar geometry head and
   regression objective; or
2. the finalized checkpoints no longer expose readily accessible information
   that separates the five extrusion and five revolve magnitudes.

ADR-0010 states that a failed repaired run cannot silently authorize another
intervention. This diagnostic introduces target-labeled post-hoc optimization
and a new scientific interpretation, even though it does not update GE1.
Consequently it requires a separately accepted additive record before model
implementation, checkpoint loading, or corpus access.

## Decision

The project adopts the separately versioned, prospective diagnostic:

```text
GE1-C7-REPAIRED-REPRESENTATION-PROBE-v1
```

The probe is read-only with respect to both finalized GE1 models. It may use
only the immutable scaled epoch-200 flat and position-free typed-graph
checkpoints from job `3345280`, autonomous `P_true` inference, the frozen
32-family operation-template train cohort, and detached features extracted
before magnitude labels enter the diagnostic.

The complete measurement, split, fitting, control, artifact, and
interpretation rules are specified in the
[representation-probe contract](../specifications/ge1_repaired_representation_probe.md).

### Frozen feature levels

For every autonomously emitted operation, each arm records:

- the existing parameterized operation-magnitude scalar;
- its raw pre-parameterization geometry-head logit;
- the per-operation shared-decoder hidden state immediately before the
  geometry head; and
- the two-token continuous encoder memory, flattened in token-major order and
  combined only with autonomously predicted operation type and canonical
  operation slot.

Target geometry, target magnitude, target operation type, target masks, and
target tensors are forbidden from model inference and feature extraction.
Targets may be joined to an already-written, detached, checksum-frozen feature
table only for scalar scoring and disposable probe fitting.

### Disposable probes

Extrusion and revolve are analyzed separately. Mandatory diagnostics are a
regularized multinomial linear probe on decoder state and a separately fitted
regularized multinomial linear probe on encoder memory plus predicted
operation type/slot. A fixed-capacity one-hidden-layer nonlinear probe is a
secondary accessibility diagnostic. Probe optimizers may contain only probe
parameters and may consume only detached saved features.

Every probe reports full-cohort resubstitution and leave-one-physical-family-
out results. Continuous and quantized variants cannot cross a split; the
protocol instead selects exactly one canonical continuous record per physical
family. Family is the grouping and statistical unit throughout.

### Interpretation boundaries

The following rules are prospective:

- decoder-state success beyond the existing scalar-only ceiling supports
  accessible information being discarded by the scalar head/loss;
- encoder-memory success without decoder-state success supports loss in
  decoder conditioning rather than a direct latent-width conclusion;
- linear failure with controlled nonlinear success supports nonlinear but not
  linear accessibility;
- failure of every controlled probe means only that these regression-trained
  checkpoints do not retain readily accessible class information; and
- strong resubstitution with weak held-out-family performance is memorization
  or accessibility evidence, not generalizable class separation.

The scalar optimal-threshold ceiling applies only to monotonic post-hoc
remapping of the existing one-dimensional scalar. It is not an upper bound on
a classifier reading decoder state or encoder memory.

No outcome may be described as proving that `latent_tokens=2` or
`bottleneck_dim=16` is a binding capacity limit. The continuous `2 x 16`
interface is not an eight-bit channel, and the protocol makes no independent-
bit requirement per program.

## Authorization boundary

This ADR authorizes only:

- diagnostic implementation and synthetic corpus-free tests;
- preparation of an unsubmitted exact-commit Adroit CPU runner;
- creation and local verification of a complete-history Git bundle; and
- a later, separately authorized Adroit execution using the operation-template
  manifest and frozen 32-family scaled train cohort.

It does not itself authorize corpus or manifest access, checkpoint loading for
scientific diagnosis, local diagnostic execution, Slurm submission, pushing,
Stage 6, C8, protected-partition access, CAD-kernel use, GE1 training, GE1
backward passes, checkpoint writes, or any model/decoder/loss/latent repair.

Krishay Maskara, acting as the designated GE1 reviewer, accepted this ADR and
the linked representation-probe contract on August 12, 2026. Implementation,
synthetic corpus-free testing, an unsubmitted runner, and local bundle
preparation may proceed. Scientific probe execution and every other excluded
action remain separately unauthorized.

## Alternatives considered

### Implement another magnitude head immediately

Rejected. The finalized evidence does not yet establish whether class
information reaches the shared decoder state.

### Treat scalar recalibration as an architectural upper bound

Rejected. A threshold remap sees only one scalar and cannot bound information
in a higher-dimensional upstream representation.

### Use resubstitution accuracy alone

Rejected. The cohort is small relative to the feature dimensions, so
resubstitution can demonstrate accessibility or memorization but not
held-out-family separation.

### Open development or RR to increase probe sample size

Rejected. The diagnostic is train-only, and all protected partitions retain
their existing access boundaries.

## Consequences

- The immutable job `3345280` result remains unchanged.
- The next question can be answered without updating a GE1 parameter.
- Grouped held-out-family evaluation and label-permutation controls constrain
  overinterpretation of high-dimensional probes.
- A successful probe could motivate a later proposal, but cannot itself
  authorize a categorical head or any other repair.
- A failed probe cannot establish architectural capacity exhaustion.

## Validation and evidence

Before any authoritative diagnostic execution, implementation must pass
synthetic target-isolation, detached-feature, grouped-split, scalar-analysis,
probe, permutation-control, artifact, checkpoint-immutability, Python 3.8,
PyTorch 1.11, protected-access, and clean-tree tests. The Adroit runner must
rerun repository regressions before any manifest, checkpoint, or payload
access.

## Review conditions

The designated-reviewer acceptance covers the protocol identity, feature
definitions, canonical-family rule, fitting and regularization procedures,
label-permutation control, geometry-loss audit, access boundary, and
non-authorization language. Any later change to those choices requires a new
additive or superseding record rather than a silent edit.

## Related records

- [ADR-0010](ADR-0010-ge1-repaired-train-sufficiency-protocol.md)
- [Repaired-sufficiency execution contract](../specifications/ge1_repaired_sufficiency_execution_contract.md)
- [Representation-probe contract](../specifications/ge1_repaired_representation_probe.md)
- [Current project status](../status.md)
