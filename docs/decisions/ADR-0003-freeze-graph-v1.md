# ADR-0003: Freeze Graph V1 After C1

- Status: `accepted`
- Decision date: `2026-08-03`
- Owners/reviewers: project research team
- Supersedes: none
- Superseded by: [ADR-0004](ADR-0004-ge1-single-manifest-encoder-comparison.md) for the immediate next-phase-selection gate only; the Graph V1 freeze remains accepted

## Context

ADR-0002 authorized one constrained flat-versus-graph comparison. The frozen
Flat V6 condition produced 0/68 complete IID-validation validity, with every
history advancing to and failing at structural edges. Initial Graph V1
improved complete validity to 22/68 by solving all single-operation programs,
but solved none of the 46 two-operation programs.

The protocol allowed one evidence-backed correction. C1 added a
capacity-neutral directed serialized-position branch. It improved
positive-edge precision, F1, and accuracy, but exact validity remained 22/68.
Its largest tradeoff was a fall in `depends_on` recall from 37/46 to 11/46.

The IID validation partition has now informed multiple architectural steps.
Further Graph V1 tuning would exceed the correction budget and make the
comparison increasingly validation-adaptive.

## Decision

Freeze Graph V1 after job `3341974`:

```text
graph scientific corrections used: 1
graph scientific correction limit: 1
additional Graph V1 corrections authorized: 0
additional Graph V1 pilot reruns authorized: 0
```

The frozen identities are:

```text
Flat V6: ac6ef718ae9bab7fa5a80d9f48d0976adf5cafad, job 3338639
Initial Graph V1: 089b9f3d0e5a61fb19ef3fa05e993fc4eceffdcb, job 3341942
Graph V1 C1: 0cd09ed34d4c4dd0d43e1456b7a06eb362ee7962, job 3341974
```

Any further neural work begins as a separately named experimental phase with
a new protocol, hypothesis, endpoint, and correction budget. It is not called
Graph V1 C2.

Systematic and held-out test partitions remain untouched. First systematic
access requires a reviewed protocol and frozen selection rule. Held-out test
access requires a later explicit decision after model selection is complete.

## Alternatives considered

### Add another pairwise correction

Rejected. The one-correction budget is exhausted, and C1 showed that local
edge improvements can redistribute errors without improving complete
validity.

### Extend training duration

Rejected within Graph V1. The two-epoch exposure was frozen for the matched
comparison, and current evidence does not isolate insufficient optimization
as the cause.

### Use the position-only prior as production repair

Rejected. The prior is scoring-only and may exploit a serialization shortcut;
inserting it would not test learned graph generation.

### Access the systematic partition immediately

Rejected until the next model and selection rule are frozen. Graph V1 has no
valid two-operation IID examples, so systematic access would not answer the
most immediate structural question.

## Consequences

### Positive

- Preserves the interpretability of the completed comparison.
- Prevents unlimited IID-validation-driven graph-head tuning.
- Makes the next scientific question explicit: global operation grouping or
  the causal role of latent collapse.
- Preserves systematic and held-out test data for a preregistered next phase.

### Tradeoffs and limitations

- Graph V1 remains unable to generate a valid two-operation program.
- The decision does not choose the next architecture.
- The causal contribution of VQ collapse remains unresolved.
- ADR-0002's systematic and localized-edit deliverables remain incomplete.

## Validation and evidence

The decision is supported by:

- [Frozen Flat V6](../experiments/b0_constrained_flat_v6.md);
- [Initial Graph V1](../experiments/b0_graph_v1_initial.md);
- [Graph V1 C1](../experiments/b0_graph_v1_c1.md); and
- [Constrained flat/graph milestone](../milestones/constrained_flat_graph_comparison.md).

## Implementation contract

Existing frozen tags, checkpoints, metrics, and run records remain immutable.
No code change is required to enforce the scientific freeze. A new runner may
reuse frozen components only if its new experiment identity, changed
components, capacity controls, partition policy, and checkpoint compatibility
are explicit.

## Review conditions

Revisit only through a superseding ADR after a separately named next-phase
protocol has been reviewed. Do not edit this ADR to retroactively authorize
another Graph V1 correction or protected-partition access.
