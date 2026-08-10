# GE1 Repaired Train-Sufficiency Execution Contract

## Status and identity

| Item | Frozen value |
|---|---|
| Status | Accepted prospective protocol; implementation pending |
| Governing decision | [ADR-0010](../decisions/ADR-0010-ge1-repaired-train-sufficiency-protocol.md) |
| Protocol | `GE1-C7-REPAIRED-SUFFICIENCY-v1` |
| Operation magnitude | `GE1-OPERATION-MAGNITUDE-POSITIVE-v1` |
| Historical C7-v2 | Immutable job `3344981` at `e325d5ad97957c08da4a19b4261560e8a4a472a4` |
| Engineering prerequisite | Passed job `3345044` at `12167ce7d0dc025c4b297b7ccb8a3011580bf0fc` |
| Scientific execution | Not authorized |
| Stage 6 / C8 | Not authorized / not begun |

This contract is separately versioned and prospective. It must not be called
C7-v2 or used to reinterpret the immutable C7-v2 failure. Job `3345044`
validated implementation behavior only; no repaired model has been trained.

## Frozen model and lifecycle

Both flat and typed-graph arms use the repaired positive-magnitude identity.
Legacy configurations and checkpoints are ineligible. The encoder
architectures, shared decoder except for the already accepted magnitude
mapping, parameter counts, optimizer, loss and weights, normalization,
direction/Boolean outputs, masks, seed, cohorts, and budgets do not change.
The typed graph encoder remains free of chronological or absolute-position
features, and inherited Flat V6 remains frozen.

| Cohort | Families | Batch | Epochs | Steps/epoch | Optimizer steps | Presentations | Selected checkpoint |
|---|---:|---:|---:|---:|---:|---:|---|
| Tiny | 4 | 8 | 200 | 1 | 200 | 800 | Fixed epoch 200 |
| Scaled | 32 | 8 | 200 | 4 | 800 | 6,400 | Fixed epoch 200 |

Tiny and scaled use separate fresh matched pairs. Each selected checkpoint is
strictly loaded into a fresh model before autonomous evaluation. Warm starts,
early stopping, best-loss selection, checkpoint reuse, and outcome-dependent
extensions are invalid.

## Operation-geometry fidelity contract

Only autonomous `P_true` predictions from the strict epoch-200 reload are
eligible. The frozen grids, scales, and strict thresholds are:

| Operation | Serialized channel | Physical targets | Scale | Strict maximum absolute error |
|---|---:|---|---:|---:|
| Extrusion | 37 | `0.5, 1.0, 1.5, 2.0, 3.0` | `4.0` | `< 0.25` controlled length units |
| Revolve | 38 | `45, 90, 180, 270, 360` | `360.0` | `< 22.5` degrees |

The threshold is half the minimum valid-grid spacing. Comparisons use
unrounded values. Equality fails.

For every sample record and target operation in canonical order, record:

- physical family and representation identity;
- operation index and target/predicted operation type;
- expected and observed active channel;
- predicted and target normalized values;
- predicted and target physical values;
- signed and absolute physical error;
- applicable threshold and strict comparison result; and
- a stable failure reason and first-failure stage when applicable.

Nonfinite, missing, masked, wrong-channel, wrong-operation, and undefined
values fail. A family passes only when every operation in every representation
variant passes. Its summary records the maximum absolute error. No mean may
satisfy the gate. One family failure fails the arm.

Existing geometry-channel-family physical errors remain report-only. No
success threshold is added for reference-plane, profile-primitive, axis, or
other non-operation geometry.

## Gate set and dependency

For each arm, the separately recorded gates are:

1. `tiny_exact_sufficiency`;
2. `tiny_operation_geometry_fidelity`;
3. `scaled_exact_sufficiency`;
4. `scaled_operation_geometry_fidelity`; and
5. `scaled_memory_use`.

Tiny exact sufficiency requires every family to have exact node sequence,
exact graph, strict conversion, analytic complete validity, and applicable
EE/RE `depends_on` exactness. Tiny fidelity must also pass for every family in
both arms before any scaled payload access.

Scaled exact and fidelity use the same criteria over 32 families. The memory
gate preserves the C7-v2 `P_true > 0`, `P_shuffle/P_true <= 0.80`, and
`P_mean/P_true <= 0.80` comparisons and donor-reporting contract.

If either tiny arm fails exactness or fidelity, all six scaled arm-gates are
`not_run`, scaled payload access remains false, and Stage 6 remains blocked.
Any scaled exact/fidelity failure is a completed scientific failure. A
memory-only failure is inconclusive. Overall pass requires all ten arm-gates
to pass.

Successful infrastructure execution finalizes an artifact and exits zero even
when a scientific gate fails. No failure automatically authorizes a repair.

## Target isolation

Targets may be paired with already-generated autonomous outputs only inside
metrics and gate computation. They are prohibited from:

- flat or typed-graph encoder inputs;
- latent memory or decoder inputs;
- autonomous generation calls;
- optimizer, loss-control, checkpoint-selection, or early-exit decisions; and
- public model/autonomous interfaces.

Source and runtime tests must prove that target geometry is not accepted or
forwarded by those interfaces.

## Artifact and provenance

The prospective artifact must include:

- resolved configuration and frozen arithmetic;
- metrics JSONL with exact, fidelity, and memory gate events;
- per-operation evidence and conservative per-family summaries;
- selected checkpoint paths, identities, epochs, and SHA-256 hashes;
- the repaired parameterization identity in configuration, training,
  checkpoints, provenance, metrics, gates, manifest, and terminal record;
- artifact manifest and `SHA256SUMS` with complete regular-file coverage;
- explicit manifest/payload/protected-partition access declarations; and
- a terminal record distinguishing infrastructure completion, scientific
  pass/failure, scaled authorization/access, Stage 6 authorization,
  CAD-kernel availability, and protected access.

Metrics may not contain raw corpus payloads or model state. Checkpoints remain
separately governed artifact files.

## Interpretation boundary

Passing demonstrates train-cohort exact structural sufficiency, controlled
operation-magnitude fidelity, and autonomous memory use for both repaired
arms under the frozen budget. It does not establish exact continuous geometry,
development or systematic generalization, comparative encoder superiority,
or CAD-kernel executability. CAD-kernel integration and any strong executable-
CAD claim remain separately governed.

## Current authorization

ADR-0010 authorizes code, corpus-free tests, an unsubmitted scientific runner,
and engineering validation preparation. It does not authorize scientific
training, local corpus/manifest access, Slurm submission, Stage 6, C8,
protected access, CAD-kernel integration, or pushing.
