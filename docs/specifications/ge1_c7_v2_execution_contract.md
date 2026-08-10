# GE1 C7-v2 Prospective Execution Contract

Status: accepted prospective protocol under
[ADR-0008](../decisions/ADR-0008-ge1-c7-v2-200-epoch-protocol.md); implementation
and scientific execution have not yet occurred.

Reviewer: Krishay Maskara, August 10, 2026.

## Historical boundary

Formal C7-v1 job `3344505` remains a failed epoch-50 execution under
`GE1-C7-SUFFICIENCY-v1`. Optimization-diagnostic job `3344907` remains a
separate completed train-only diagnostic. C7-v2 does not mutate either record,
schema, checkpoint, or conclusion.

The diagnostic showed both unchanged arms first reaching autonomous exact
sufficiency at epoch 200. C7-v2 therefore tests a prospectively fixed
200-epoch budget while deferring, not deleting, the previously triggered
decoder-repair path.

## Identities

```text
GE1-C7-SUFFICIENCY-v2
GE1-C7-METRICS-v2
GE1-C7-ARTIFACT-v2
stage6_authorized_by_c7_v2
```

C7-v1 and diagnostic identities remain unchanged.

## Frozen arithmetic and lifecycle

| Cohort | Epochs | Updates/epoch | Updates/arm | Presentations/arm | Selected checkpoint |
|---|---:|---:|---:|---:|---|
| Tiny 4-family | 200 | 1 | 200 | 800 | fixed epoch 200 |
| Scaled 32-family | 200 | 4 | 800 | 6,400 | fixed epoch 200 |
| Eventual Stage 6 407-family | 200 | approximately 51 | approximately 10,200 | exactly 81,400 | fixed epoch 200 |

There is no early stopping, best-checkpoint selection, outcome-dependent
extension, warm start, resume, or diagnostic-checkpoint reuse.

Tiny and scaled each receive a separately constructed fresh matched seed-2026
flat/typed-graph pair. Shared components are byte-identical within a pair and
mutable state is disjoint. The scaled pair cannot inherit tiny state.

## Gate sequence

1. Verify exact clean source, standalone checkout, CPython 3.8.13, PyTorch
   1.11, CPU-only execution, and Slurm identity.
2. Verify the authoritative operation-template manifest using metadata only.
3. Reproduce the exact frozen C7-v1 four-family cohort and hash.
4. Open only those four train payloads.
5. Train both fresh tiny arms for exactly 200 epochs.
6. Write and strictly reload epoch-200 checkpoints into fresh models.
7. Score only fully autonomous `P_true` exactness.
8. Open the remaining scaled train payloads only if both tiny arms pass.
9. Reproduce the unchanged 32-family cohort and hash.
10. Construct a new fresh matched pair and train each scaled arm for exactly
    200 epochs, or 800 optimizer updates.
11. Write and strictly reload fixed epoch-200 checkpoints.
12. Apply unchanged scaled exact and autonomous memory gates.
13. Publish a new immutable v2 artifact.

## Scientific criteria

Every family must have exact unrounded `1.0` for exact node sequence, exact
typed graph, strict conversion, and complete executable validity. `EE` and
`RE` additionally require exact `depends_on`.

Memory use requires finite `P_true > 0`, `P_shuffle / P_true <= 0.80`, and
`P_mean / P_true <= 0.80` for each arm. No teacher-forced value can satisfy a
gate.

## Decisions and non-authorization

Tiny exact failure makes all scaled work `not_run`, blocks Stage 6, and returns
the project to the deferred repair path. Scaled exact failure does the same,
although already-authorized memory diagnostics may complete. Memory-only
failure is inconclusive, does not trigger repair, and blocks Stage 6.

Only complete passage sets `stage6_authorized_by_c7_v2=true`. A scientific
failure finalizes and exits zero. Infrastructure failure does not finalize,
exits nonzero, and authorizes nothing.

C7-v2 implementation and validation do not begin Stage 6, C8, or repair.
Development, RR, ER, IID, history-depth, and geometry-extrapolation remain
closed.

## Artifact and runner

The v2 artifact contains resolved configuration, v2 metrics, complete tiny
and any authorized scaled checkpoints, artifact manifest, and sorted
`SHA256SUMS` with final LF. It records exact commit, clean source digest,
Slurm identity, model lifecycle, fixed epoch, update/presentation arithmetic,
strict reload, every gate result, and access declarations.

The CPU runner must complete all tests and repository gates before manifest
or payload access, reject linked worktrees by requiring an ordinary `.git`
directory, preserve stdout/stderr, independently verify checkpoint hashes and
Slurm provenance, and emit explicit terminal success/failure events.

## Related records

- [ADR-0008](../decisions/ADR-0008-ge1-c7-v2-200-epoch-protocol.md)
- [C7-v1 record](../experiments/ge1_c7_pilot.md)
- [Optimization diagnostic record](../experiments/ge1_optimization_diagnostic.md)
- [Implementation plan](graph_encoder_implementation_plan.md)
- [Stage 0 preregistration](ge1_stage0_preregistration.md)
