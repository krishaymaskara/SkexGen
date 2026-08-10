# ADR-0008: GE1 C7-v2 Prospective 200-Epoch Protocol

- Status: `accepted`
- Proposal date: `2026-08-10`
- Decision date: `2026-08-10`
- Owner: project research team
- Designated GE1 reviewer: Krishay Maskara
- Adds to: [ADR-0004](ADR-0004-ge1-single-manifest-encoder-comparison.md)
- Prospectively supersedes: the C7 and eventual Stage 6 50-epoch budget in
  [ADR-0006](ADR-0006-ge1-c7-sufficiency-execution-contract.md) and
  `GE1-STAGE0-PREREG-v1`
- Preserves without reinterpretation: formal C7-v1 and
  [ADR-0007](ADR-0007-ge1-c7-optimization-sufficiency-diagnostic.md)
- Superseded by: none

## Context

Formal C7-v1 Adroit job `3344505` remains an immutable scientific failure at
its preregistered epoch-50 checkpoint. Both arms failed the four-family exact
gate, scaled work remained `not_run`, the preauthorized decoder-repair path
was triggered, and Stage 6 was not authorized.

The separate post-C7 optimization diagnostic completed as Adroit job
`3344907` at exact commit
`fbc6073f63da9f0e10b5db8c0c0d4786a48ce0c0`. With unchanged architectures,
decoder, losses, optimizer, seed, cohort, and autonomous criteria, both arms
first became exact-sufficient at the predetermined epoch-200 milestone and
remained exact at epoch 500. Its preregistered interpretation was
`undertraining_supported_both_arms`.

That result does not rewrite C7-v1 or itself authorize downstream work. It
does supply prospective evidence that the original 50-epoch budget was too
short for the tiny cohort. A separately versioned formal C7-v2 execution can
test the longer budget without changing historical artifacts.

## Decision

### Historical boundary

The project adopts C7-v2 prospectively. This decision:

- does not alter, rerun, relabel, overwrite, or reinterpret C7-v1 job
  `3344505`;
- does not alter or replace optimization-diagnostic job `3344907`;
- does not mutate any C7-v1 or diagnostic identity, schema, constant, test,
  artifact, or checkpoint;
- does not begin Stage 6, C8, or decoder-repair implementation; and
- keeps the C7-v1 repair trigger recorded while deferring that path during
  the governed C7-v2 test.

### Versioned identities

C7-v2 uses new identities:

```text
GE1-C7-SUFFICIENCY-v2
GE1-C7-METRICS-v2
GE1-C7-ARTIFACT-v2
```

Its Stage 6 decision field is:

```text
stage6_authorized_by_c7_v2
```

C7-v2 must not emit an ambiguous unversioned Stage 6 authorization or reuse a
C7-v1 identity under the new budget.

### Unchanged scientific system

Except for the training budget and corresponding fixed checkpoint, C7-v2
keeps unchanged:

- flat and position-free typed-graph encoders;
- shared decoder and model capacities;
- loss definitions and weights;
- AdamW, learning rate `1e-3`, weight decay `0`, and gradient clipping `1.0`;
- deterministic tiny and scaled cohorts, hashes, and seed 2026;
- fresh matched initialization and disjoint mutable model state;
- autonomous generation and scoring;
- exactness criteria and dependency requirement;
- memory conditions and `0.80` thresholds;
- gate dependencies, scientific-failure semantics, and infrastructure-failure
  semantics; and
- the operation-template-only access policy.

The hierarchical operation-group decoder repair is deferred, not erased. No
repair code is implemented or called by C7-v2.

### Prospective fixed budget

Epoch is the family-exposure coordinate.

| Cohort | Families | Epochs | Updates/epoch | Optimizer updates/arm | Presentations/arm |
|---|---:|---:|---:|---:|---:|
| Tiny | 4 | 200 | 1 | 200 | 800 |
| Scaled | 32 | 200 | 4 | 800 | 6,400 |
| Eventual Stage 6, if authorized | 407 | 200 | approximately 51 | approximately 10,200 | exactly 81,400 |

For every C7-v2 arm:

```text
checkpoint selection = fixed epoch 200
early stopping = false
best-checkpoint selection = false
outcome-dependent extension = false
warm start = false
resume from C7-v1 = false
resume from optimization diagnostic = false
reuse of diagnostic checkpoints = false
```

The same fixed 200-epoch budget applies to eventual Stage 6 only if a
finalized C7-v2 artifact first authorizes Stage 6.

### Cohorts and access

C7-v2 must reproduce the exact metadata-only C7-v1 cohorts:

- tiny: one `E`, `R`, `EE`, and `RE` family, selected-set SHA-256
  `c07e90675099cb2ea58575fe91683dc80b70cf3fa85fa6fa8de0c44dbe9783d6`;
- scaled: eight families from each of `E`, `R`, `EE`, and `RE`, using the
  unchanged C7-v1 selector and frozen hash.

Only the four tiny train payloads may be opened initially. Both tiny arms must
pass before any additional scaled payload is opened. Operation-template
development, RR systematic, ER test, IID, history-depth, and
geometry-extrapolation remain closed.

### Model lifecycle

C7-v2 constructs a new fresh matched seed-2026 pair for the tiny cohort. If
both tiny arms pass, it constructs a second, separate fresh matched seed-2026
pair for the scaled cohort. Scaled models cannot reuse tiny model objects,
parameters, buffers, optimizers, or checkpoints. Neither pair can resume from
C7-v1 or the optimization diagnostic.

Within each pair, shared components begin byte-identically and all mutable
parameters and buffers are disjoint.

### Exact and memory gates

At fixed epoch 200, every physical family must have exact unrounded `1.0` for:

```text
exact_node_sequence
exact_graph
strict_conversion
complete_executable_validity
```

`EE` and `RE` additionally require exact `depends_on`.

Scaled memory thresholds remain:

```text
finite P_true > 0
P_shuffle / P_true <= 0.80
P_mean / P_true <= 0.80
```

Only fully autonomous outputs count. Epoch-200 checkpoints are written before
evaluation and strictly reloaded into fresh models. Earlier checkpoints,
losses, plateaus, and teacher-forced outputs cannot select or satisfy a gate.

### Decision behavior

- If either tiny arm fails exact sufficiency, every scaled result is
  `not_run`, C7-v2 fails, Stage 6 remains blocked, and the preauthorized
  decoder-repair path becomes the next permitted implementation path.
- If scaled exact sufficiency fails, C7-v2 fails, Stage 6 remains blocked, and
  the decoder-repair path becomes next. Existing diagnostic memory
  interventions may still run after an already-trained scaled model under the
  unchanged gate contract.
- A memory-only failure sets `comparison_inconclusive=true`, does not trigger
  repair, and blocks Stage 6.
- Only complete passage of every tiny, scaled-exact, and scaled-memory gate
  for both arms sets `stage6_authorized_by_c7_v2=true`.
- Scientific gate failure finalizes a valid artifact and exits zero.
- Infrastructure, provenance, reload, access, malformed-metric, or artifact
  integrity failure exits nonzero and authorizes nothing.

Stage 6 remains blocked until a finalized C7-v2 artifact explicitly records
`stage6_authorized_by_c7_v2=true`.

### Artifact and runtime

C7-v2 publishes a new immutable external artifact under
`GE1-C7-ARTIFACT-v2`. It requires an exact clean standalone Git checkout,
CPython 3.8.13, PyTorch 1.11, CPU-only execution, and a nonempty decimal Slurm
job identity. Commit, source cleanliness, Slurm identity, checkpoint
provenance, access declarations, and hashes are verified before a terminal
success marker is emitted.

The CLI exposes only governed paths and exact commit. Implementation tests use
synthetic metadata, procedural model fixtures, and temporary output paths.
They do not open the corpus.

## Alternatives considered

### Rewrite C7-v1 as a 200-epoch run

Rejected. Its epoch-50 result and identities are immutable history.

### Reuse diagnostic epoch-200 checkpoints

Rejected. C7-v2 is a new formal clean-start execution, not post-hoc checkpoint
selection from an observed diagnostic.

### Implement decoder repair immediately

Deferred. The unchanged decoder demonstrated tiny-set exact sufficiency at
the prospective budget. Repair remains available if C7-v2 exact sufficiency
fails.

### Select the first successful checkpoint

Rejected. C7-v2 always trains to and selects fixed epoch 200.

## Consequences

- C7-v1 and the optimization diagnostic remain independently interpretable.
- C7-v2 tests the evidence-based longer budget with unchanged models and
  gates.
- The eventual Stage 6 budget is prospectively 200 epochs if C7-v2 passes.
- Development and every protected partition remain closed.
- No C7-v2 implementation or validation result by itself begins Stage 6 or
  C8.

## Validation and evidence

Implementation must prove C7-v1 remains at epoch 50, the diagnostic remains
unchanged, C7-v2 uses epoch 200, tiny/scaled arithmetic is exact, model pairs
are separately fresh, earlier or foreign checkpoints cannot satisfy C7-v2,
gate dependencies and thresholds are unchanged, protected loaders are
unreachable, and v2 artifacts enforce source, Slurm, access, checkpoint, and
checksum provenance.

The accepted decision authorizes implementation and later exact-commit Adroit
execution of C7-v2 under this contract. It does not authorize local corpus
execution, Stage 6, C8, repair implementation, or protected access.

## Related records

- [ADR-0004](ADR-0004-ge1-single-manifest-encoder-comparison.md)
- [ADR-0006](ADR-0006-ge1-c7-sufficiency-execution-contract.md)
- [ADR-0007](ADR-0007-ge1-c7-optimization-sufficiency-diagnostic.md)
- [Formal C7-v1 record](../experiments/ge1_c7_pilot.md)
- [Optimization diagnostic record](../experiments/ge1_optimization_diagnostic.md)
- [C7-v2 execution contract](../specifications/ge1_c7_v2_execution_contract.md)
- [Stage 0 preregistration](../specifications/ge1_stage0_preregistration.md)
