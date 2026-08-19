# GE1 Stage 6 train-gate postmortem diagnostic

| Item | Frozen value |
|---|---|
| ADR | [ADR-0014](../decisions/ADR-0014-ge1-stage6-structure-only-comparison.md) |
| Diagnostic | `GE1-STAGE6-TRAIN-GATE-POSTMORTEM-v1` |
| Artifact | `GE1-STAGE6-TRAIN-GATE-POSTMORTEM-ARTIFACT-v1` |
| Failed producer commit | `5f4542f86756dae44435af27a6e072db6f27a8ef` |
| Failed producer job | `3354961` |
| Cohort | authorized narrow train only |
| Execution | read-only CPU inference; no training |
| Completed execution | job `3355134` at commit `ddf9617fb124fbc18f38302943f0c982019bfee9` |
| Current authority | completed train-only diagnostic; no downstream authority |

## Question and frozen reuse

The diagnostic resolves only which existing train-side predicate caused
`stage6_train_reliability_failure` in producer job `3354961`: per-arm/per-seed
optimization reliability, the paired per-seed train-ceiling comparison, or
the train structural-memory gate. It uses unchanged Stage 6 autonomous
evaluation, structural scoring, `optimization_reliability`, and
`structure_memory_gate_for_cohort`. It does not alter a gate, threshold,
model, decoder, loss, configuration, training path, producer, finalizer, or
existing artifact.

The six inputs are the job-scoped epoch-200 wrappers named
`stage6-{flat,typed_graph}-seed{2026,2027,2028}.pt`. Every wrapper requires a
separately supplied reviewed SHA-256 and exact wrapper, inner-checkpoint,
commit, source-tree, arm, seed, epoch, configuration, partition, arithmetic,
parameter-count, runtime, device, RNG, and provenance validation. The
diagnostic loads the already-authorized narrow `operation_template.train`
package through the unchanged strict narrow loader. It neither discovers nor
opens the retained generic checkpoint directory.

## Reported predicates

For each arm and seed, the summary records finite loss and gradient status,
first plateau epoch, plateau by epoch 200, fixed epoch-200 checkpoint status,
true-memory train structural score and ceiling shortfall, paired flat-versus-
typed-graph shortfall difference against the inclusive `0.05` gate,
`P_true`, `P_shuffle`, `P_mean`, both ratios against the inclusive `0.80`
gates, and every failed predicate. It separately identifies failed
optimization runs, failed ceiling seeds, failed memory arms, and the minimal
combined reason for the producer stop.

Train-only measurements do not license an encoder-superiority claim. The
scientific pass/failure outcome never controls diagnostic artifact validity.

## Atomic artifact

Successful diagnostic completion publishes exactly six canonical files:

1. `resolved_config.json`;
2. `train_family_records.jsonl`;
3. `training_histories.jsonl`;
4. `summary.json`;
5. `artifact_manifest.json`;
6. `SHA256SUMS`.

Finalization recomputes all scores and gates, verifies exact arm/seed/
condition/family coverage, verifies checkpoint and access evidence, and
requires complete manifest and checksum coverage. A result that reproduces a
failed scientific gate still finalizes as a complete valid diagnostic.

## Access and runner boundary

The CPU runner has exactly four container binds: repository read-only,
retained job work read-only, narrow train input read-only, and diagnostic
output parent read-write. It has no development, RR, ER/test, IID,
history-depth, geometry-extrapolation, corpus, manifest, external model,
repaired artifact, or protected-partition bind. The diagnostic performs
train-only autonomous inference. It performs no training, backward pass,
optimizer step, checkpoint write, modification, or scientific repair.

The resolved artifact also freezes the supplied execution context: Slurm job
`3354961` ended `FAILED` with exit `42:0`, elapsed `02:07:20`, MaxRSS
`855460K`, one allocated CPU, 49/49 focused tests, 620/620 complete tests with
only the six expected CUDA skips, the train-reliability failure code, six
retained wrappers, and no development access. Those values identify the
postmortem target; runtime validity still comes from strict input and artifact
verification rather than from a scientific-outcome expectation.

The runner requires Python 3.8.13, PyTorch 1.11.0, CPU-only execution, and one
CPU thread. It invokes the diagnostic exactly once, verifies its checksums and
artifact, and emits terminal success, failure, or timeout telemetry. The
runner contains no submission command.

Implementation and preparation alone did not authorize transfer,
train-package or checkpoint access, execution, or submission. Those acts
required separate review of the exact diagnostic commit, runner hash, six
wrapper hashes, safe paths, resource request, and a single submission.

The later authorized execution completed validly. Its authenticated
`P_true`, `P_shuffle`, and `P_mean` values, exact producer failure, artifact
checksums, and access record are in the [job-3355134 execution
record](../experiments/ge1_stage6_train_gate_postmortem_3355134.md). Completion
does not authorize another producer, development access, repair, finalization,
Stage 7, C8, or protected access.
