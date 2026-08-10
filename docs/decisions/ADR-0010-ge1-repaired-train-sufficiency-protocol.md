# ADR-0010: GE1 Repaired Train-Sufficiency Protocol

- Status: `accepted`
- Proposal date: `2026-08-10`
- Decision date: `2026-08-10`
- Owner: project research team
- Designated GE1 reviewer: Krishay Maskara
- Adds to: [ADR-0009](ADR-0009-ge1-positive-operation-magnitude-repair.md)
- Preserves without reinterpretation: C7-v2 commit
  `e325d5ad97957c08da4a19b4261560e8a4a472a4`, Adroit job `3344981`
- Engineering prerequisite: positive-magnitude validation commit
  `12167ce7d0dc025c4b297b7ccb8a3011580bf0fc`, Adroit job `3345044`
- Superseded by: none

## Context

The immutable C7-v2 run was a completed train-only scientific failure. Its
operation-parameter diagnostic isolated five negative extrusion or revolve
magnitudes. ADR-0009 authorized a positive, bounded neural parameterization,
and job `3345044` established that implementation's engineering correctness
without opening a corpus or training a repaired model.

Engineering correctness is not target fidelity. A positive prediction may
still be far from the controlled target, and analytic controlled-domain
validity does not establish CAD-kernel execution. A new scientific run
therefore needs a prospective operation-geometry fidelity gate rather than
retroactively treating positivity as success.

The repository audit confirmed the following frozen evidence:

- extrusion targets are `(0.5, 1.0, 1.5, 2.0, 3.0)`;
- revolve targets are `(45.0, 90.0, 180.0, 270.0, 360.0)`;
- serialized channel 37 uses the length scale `4.0`;
- serialized channel 38 uses the angle scale `360.0`; and
- the accepted tiny/scaled epoch-200 arithmetic is respectively 200/800
  optimizer steps and 800/6,400 training-example presentations.

Half the minimum target-grid spacing is consequently `0.25` controlled length
units for extrusion and `22.5` degrees for revolve.

## Decision

### Separately versioned prospective protocol

The accepted protocol identity is:

```text
GE1-C7-REPAIRED-SUFFICIENCY-v1
```

It is not C7-v2. It does not change the C7-v2 code, schemas, artifact, result,
or interpretation. No repaired model has yet been trained.

Both arms must use `GE1-OPERATION-MAGNITUDE-POSITIVE-v1`. The legacy
`GE1-OPERATION-MAGNITUDE-TANH-LEGACY-v1` identity and all legacy checkpoints
are ineligible. The flat and position-free typed-graph encoders, shared
decoder capacity, losses and weights, optimizer, normalization, categorical
direction and Boolean outputs, masks, cohorts, seed, and budgets remain
unchanged. The inherited Flat V6 implementation remains frozen.

### Model lifecycle and training budget

The tiny phase uses four physical operation-template train families, batch
size 8, seed 2026, exactly 200 epochs, one optimizer step per epoch, exactly
200 optimizer steps, and exactly 800 family presentations per arm. Each arm
selects only its fixed epoch-200 checkpoint and strictly reloads it into a
fresh model before autonomous evaluation.

The scaled phase uses 32 physical train families, batch size 8, seed 2026,
exactly 200 epochs, four optimizer steps per epoch, exactly 800 optimizer
steps, and exactly 6,400 family presentations per arm. It uses a fresh matched
pair independent of the tiny models, fixed epoch-200 selection, and strict
fresh-model reload.

Warm starts, checkpoint reuse, best-loss selection, early stopping,
outcome-dependent extension, and post-outcome changes are forbidden.

### Operation-geometry fidelity

The hard gate is named `operation_geometry_fidelity`. It uses autonomous
`P_true` output from the strict reloaded epoch-200 checkpoint only. For every
sample record and every target operation in canonical order, it requires the
predicted operation type and active channel to match the target, reads channel
37 for extrusion or channel 38 for revolve, and records normalized and
physical predicted and target values, signed and absolute physical error,
threshold, and the unrounded decision.

Extrusion passes only when absolute physical error is strictly less than
`0.25`. Revolve passes only when absolute physical error is strictly less than
`22.5`. Equality fails. Nonfinite, missing, masked, wrong-channel,
wrong-operation, or otherwise undefined values fail.

Aggregation is conservative. Each representation variant and operation is
evaluated first. A physical family passes only when every variant and every
operation passes. The family summary records the maximum absolute error; no
average can hide an outlier. One failed operation fails its family, and one
failed family fails its arm.

The existing physical geometry-error metrics remain reportable for all
geometry-channel families. This ADR assigns no new success threshold to any
non-operation geometry. Passing the new gate does not establish exact
continuous geometry or CAD-kernel executability.

### Gate dependency and decision

Each tiny arm must pass, for every physical family:

- exact node sequence;
- exact typed graph;
- strict conversion;
- analytic complete validity;
- applicable `depends_on` exactness for EE and RE; and
- operation-geometry fidelity.

The scaled payload may be opened only after both tiny arms pass every exact
and fidelity criterion. Otherwise every scaled exact, fidelity, memory, and
access record is `not_run`.

Each scaled arm must pass the unchanged exact-sufficiency gate, the new
operation-geometry fidelity gate for every family, and the unchanged
autonomous memory-use gate. Overall repaired sufficiency passes only if both
arms pass all required scaled gates.

A tiny failure or scaled exact/fidelity failure is a completed repaired
scientific failure and blocks Stage 6. A memory-only failure is inconclusive
and blocks Stage 6. A successfully executed scientific failure must finalize
a valid artifact and exit zero. The result cannot authorize another repair;
any intervention requires a new diagnosis and governance decision.

Even a complete pass does not establish OpenCascade validity. CAD-kernel
integration remains a separately governed future gate before strong
“executable CAD” claims.

### Target isolation

Target geometry may enter only post-generation metric and gate computation.
It may not enter either encoder, latent memory, decoder inputs, autonomous
generation, checkpoint selection, training control, or any model interface.
Implementation and source audits must enforce this boundary.

### Versioned provenance and artifacts

The implementation must use new protocol, metrics, gate, checkpoint-role, and
artifact identities. Resolved configuration, training configuration,
inference and recovery checkpoints, provenance, metrics, gate records,
artifact manifest, and terminal record must carry
`GE1-OPERATION-MAGNITUDE-POSITIVE-v1`.

The final artifact must contain resolved configuration, metrics JSONL,
exact/fidelity/memory gate records, selected checkpoint identities and hashes,
per-operation and per-family fidelity evidence, an artifact manifest,
`SHA256SUMS`, explicit access declarations, and an unambiguous terminal
infrastructure/scientific decision. Raw corpus/model payloads and model state
remain outside metrics except for the separately governed selected
checkpoints.

## Authorization boundary

This accepted ADR authorizes implementation, corpus-free tests, preparation of
an unsubmitted scientific Slurm runner, and exact-commit engineering
validation only. It does not authorize scientific training, local corpus or
manifest access, Slurm submission, Stage 6, C8, protected-partition access,
CAD-kernel integration, or pushing a branch.

The first Adroit execution must be implementation validation only. Scientific
submission commands may not be prepared until that result is independently
audited and recorded.

## Alternatives considered

### Treat positivity as scientific sufficiency

Rejected. Positivity removes the diagnosed invalid-domain failure class but
does not measure distance from the target grid.

### Require zero numerical error

Rejected. The controlled grids support a prospective nearest-grid fidelity
criterion without pretending neural floating-point output must be bit-exact.

### Average errors across variants, operations, or families

Rejected. Averaging could conceal a failing representation or operation and
would weaken the family-level exact-sufficiency premise.

### Add CAD-kernel execution now

Rejected for this protocol. Kernel integration changes the evaluation system
and remains separately governed.

## Consequences

- The repaired scientific question becomes prospectively testable without
  changing C7-v2 history.
- Positive but inaccurate operation magnitudes cannot pass.
- Scaled access remains dependent on both tiny arms.
- Target tensors remain post-generation evaluator inputs only.
- Stage 6 remains blocked until an authoritative repaired artifact passes all
  required exact, fidelity, and memory gates.

## Related records

- [ADR-0008](ADR-0008-ge1-c7-v2-200-epoch-protocol.md)
- [ADR-0009](ADR-0009-ge1-positive-operation-magnitude-repair.md)
- [Repaired sufficiency execution contract](../specifications/ge1_repaired_sufficiency_execution_contract.md)
- [GE1 implementation plan](../specifications/graph_encoder_implementation_plan.md)
- [Operation-magnitude engineering validation](../experiments/ge1_operation_magnitude_repair_cpu_validation.md)
