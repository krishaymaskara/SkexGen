# ADR-0016: GE1 Autonomous Stop and Unconstrained Node Decoding

- Status: `accepted`
- Proposal date: `2026-08-20`
- Acceptance date: `2026-08-20`
- Owner: project research team
- Designated GE1 reviewer: Krishay Maskara
- Adds to: [ADR-0014](ADR-0014-ge1-stage6-structure-only-comparison.md) and
  [ADR-0015](ADR-0015-ge1-grid-softmax-operation-magnitude-classification.md)
- Superseded by: none

## Context

GE1 autonomous decoding currently receives the target node count and applies
`V5_NODE_GRAMMAR` as an exact-length prefix mask. In the train/development
manifest, E, R, EE, and RE have unique lengths, so this output-side count
reveals the template. At most one node choice remains free. Train-only
zero-memory job `3355342` confirmed the leak: flat retained pooled structural
score `0.838`, typed graph retained `0.586`, and both arms had `P_true = 1.0`.

This is also unsafe for the once-only Stage 7 RR evaluation. RR is the only
length-nine completion, so exact-length grammar masking could emit it without
learning it from the encoder.

ADR-0014 said grammar validity “is never asserted by construction.” That was
inaccurate for the implemented masked decoder. Exact-length
`legal_next_node_ids` made a complete grammar-valid node sequence true by
construction. This record corrects that description prospectively; the
historical decision and results remain unchanged.

## Decision

Adopt an opt-in node-generation identity, independent of the magnitude
parameterization:

```text
GE1-PAD-TERMINATED-UNCONSTRAINED-NODES-v1
GE1-AUTONOMOUS-STOP-NODE-GENERATION-CONTRACT-v1
GE1-SHARED-TYPED-EDGE-DECODER-V4
GE1-DECODER-OUTPUT-POSITIONS-v3
GE1-CHECKPOINT-v4
```

Historical configurations retain
`GE1-REQUESTED-COUNT-GRAMMAR-MASKED-v1`, their existing decoder and checkpoint
versions, and their serialized shape. The new identity is used with ADR-0015's
grid-softmax magnitude head in the future retrain, but the two identity axes
remain explicit rather than conflated.

### Stop symbol and inference

`<pad>` (node type id `0`) is the learned terminator. No vocabulary entry or
head dimension is added. Autonomous inference accepts only encoder memory for
the new identity; it takes a plain argmax over all eight node-type logits,
without a transition or exact-length grammar mask. `<pad>` ends generation and
is excluded from `raw_nodes`, graph-node ownership, and ordered-pair edge
enumeration.

Generation attempts at most `max_nodes = 16` output positions. Reaching the
cap without `<pad>` is a recorded failed outcome, not an exception. The cap is
deliberately above the grammar maximum so over-generation remains observable.
The new path preserves raw, selected, and correction telemetry; selected is
identical to raw and every node-type correction flag is false.

The GE1 implementation is derived from
`prototype/flat_baseline/constrained_v6_autonomous.py` at commit
`12521f64c73bf5f818156fe62722e64a0537d59c`. The frozen original remains
byte-identical and all legacy identities continue to call it. GE1 conversion
continues through the graph conversion boundary. The frozen V6 conversion
module assumes a requested count only in its selection helper; the new
GE1-owned generator can reuse its record and axis helpers without copying the
whole module.

### One-terminator supervision

Teacher-forced targets for the new identity supervise exactly one additional
position. For a real sequence of length `N`, position `N` has node type
`<pad>` and `node_mask=True`; later padded positions remain masked out. All
non-node fields at the terminator are neutral and have no applicable geometry.
There is no new loss term: the existing node-type cross-entropy reads the
extended `node_mask`.

The terminator is not a graph node. Edge targets, edge logits, edge masking,
and pair enumeration operate over the `N` real-node positions only. Historical
targets and losses are unchanged unless the new identity is explicitly
selected.

### Version and checkpoint separation

The terminator introduces a genuine output position, so the output-position
contract advances to v3. The new generation semantics also receive decoder v4
and checkpoint schema v4. Strict loading rejects every prior identity before
state loading. Historical magnitude and generation combinations continue to
resolve to their exact prior versions.

### Stage 6 train-side reliability correction

Keep the following gates unchanged:

- train `P_true > 0`;
- the structural memory gate;
- 200 finite losses and gradients per run;
- plateau by epoch 200; and
- the fixed epoch-200 checkpoint.

Remove the per-seed cross-arm train-ceiling predicate
`abs(train_score_flat - train_score_typed_graph) <= 0.05`. Do not replace it.
It cannot distinguish failed optimization from the encoder effect under test,
is symmetric against either arm being better, and can veto a meaningful arm
difference before development opens. Per-seed train scores, shortfalls, and
their absolute cross-arm difference remain reported as nonbinding diagnostics.
This correction is frozen before any retrain result exists.

## Prerequisite memory-to-count diagnostic

Before any retrain, run one read-only train-only linear probe over the six
existing epoch-200 job-`3354961` checkpoints authenticated by job `3355134`.
For both `encoded.memory [B,2,32]` and `encoded.prequant [B,2,16]`, predict the
four observed train node-count classes with multinomial logistic regression,
using a deterministic within-train split, a majority-class baseline, and a
shuffled-label control.

The probe may read only the authorized narrow train package and the six
immutable checkpoint wrappers. It performs no model training, checkpoint
mutation, development access, protected access, or CAD execution. Its result
is a prerequisite diagnostic, not a scientific comparison and not a gate
whose outcome changes artifact validity. Failure to recover count means the
retrain remains unauthorized and is itself recorded as a bottleneck finding.

## Preservation guarantees

- `prototype/flat_baseline` is immutable and legacy autonomous output is
  byte-identical.
- `V5_NODE_GRAMMAR` transitions, terminal set, exact-length callers, encoders,
  magnitude heads, `max_nodes`, and the three pinned ADR-0013 runners are
  unchanged.
- Grammar validity becomes an observed score criterion that can fail under
  the new identity.
- Autoregressive prefix feedback remains. Removing that residual bypass would
  be a separate controlled change.
- RR and ER remain closed; the Stage 0 once-only RR authorization is untouched.

## Authorization boundary

This record authorizes documentation, implementation, synthetic tests, the
single read-only memory-to-count prerequisite job above, and one exact-commit
Adroit CPU engineering-validation run of the non-submitting stop-symbol
validation runner.

It does **not** authorize a retrain, Stage 6 producer or finalizer execution,
development, RR, ER, IID, any protected partition, Stage 7, C8, a CAD kernel,
or a scientific claim. No probe or engineering-validation outcome
automatically authorizes those actions.
