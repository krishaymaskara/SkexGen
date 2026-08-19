# GE1 Stage 6 train-only zero-memory diagnostic

| Item | Frozen value |
|---|---|
| ADR | [ADR-0014](../decisions/ADR-0014-ge1-stage6-structure-only-comparison.md) |
| Diagnostic | `GE1-STAGE6-ZERO-MEMORY-DIAGNOSTIC-v1` |
| Artifact | `GE1-STAGE6-ZERO-MEMORY-ARTIFACT-v1` |
| Record | `GE1-STAGE6-ZERO-MEMORY-RECORD-v1` |
| Producer job | `3354961` |
| Producer commit | `5f4542f86756dae44435af27a6e072db6f27a8ef` |
| Prior postmortem job | `3355134` |
| Prior postmortem commit | `ddf9617fb124fbc18f38302943f0c982019bfee9` |
| Implementation commit | `2eaabf2e5a88fc62f559c914123694539a93e188` |
| Completed diagnostic job | `3355342` |
| Cohort | authorized narrow train only |
| Execution | read-only CPU inference; no training |
| Current authority | completed evidence only; no further execution or access authority |

## Question and frozen intervention

This diagnostic asks only whether structural train performance remains when
the complete encoder-memory value tensor is replaced with exact zeros. It
compares that result with the authenticated true-memory records from
postmortem job `3355134`. It does not rerun true-memory inference, open
development, compare encoder superiority, or change ADR-0014.

Immediately before the established decoder path, `torch.zeros_like` replaces
the memory values. Shape, dtype, device, batch and recipient dimensions,
recipient node counts, masks, positions, constraints, ordering, decoder state,
and every other decoder input remain unchanged. There is no CPU round trip.
No global mean, cross-batch, cross-example, random, noise, retraining, repair,
or additional intervention is part of this version.

The diagnostic strictly recovers the same six CPU epoch-200 checkpoint
wrappers for flat and typed-graph seeds 2026/2027/2028. Their reviewed wrapper
SHA-256 values are immutable module and runner constants. The unchanged narrow
train loader supplies exactly 407 families. Checkpoint recovery and all
identity, provenance, source, configuration, partition, runtime, device, RNG,
and arithmetic checks are inherited unchanged from the train-gate postmortem.

## Prior true-memory authority

Before any train loader or checkpoint access, the diagnostic authenticates
the exact job-3355134 postmortem directory name, all five payload hashes, the
hash of `SHA256SUMS`, its complete file set, and the postmortem verifier result.
Only its stored `P_true` family records are used. Their structural scores are
derived with the unchanged scorer; no model inference or baseline redefinition
occurs. The exact prior record digest accompanies every zero-memory record.

## Records and descriptive calculations

There is exactly one `P_zero` record for each arm, seed, and train family:
`2 * 3 * 407 = 2,442`. Each record includes support, the authenticated prior
true-memory score, the zero-memory score, ratio and difference, full preserved
prediction/structural evidence, the recomputed score record, and intervention
evidence proving zero values and preserved tensor/device bookkeeping.

Primary descriptive aggregation uses the unchanged ADR-0014 train-memory
weighting: pool the 407 equally weighted family scores across the three equally
sized seeds separately by arm. Per-seed and E/R/EE/RE results are also
reported. The secondary family-balanced result is the equal E/R/EE/RE template
macro mean. It is descriptive and is not a validity gate.

This diagnostic creates no threshold. A complete scientific pass, failure,
large drop, small drop, or intermediate result cannot control artifact
validity. High zero-memory performance leaves decoder-side node counts,
positions, masks, constraints, or other unchanged decoder inputs plausible.
A substantial drop relative to the already observed mean-memory result would
make information in that mean vector a candidate explanation. Neither outcome
proves leakage, an exact defect, information absence, capacity exhaustion,
encoder superiority, repair, Stage 6 success, or development generalization.

## Atomic artifact and verifier

Successful completion atomically publishes exactly five files:

1. `resolved_config.json`;
2. `zero_memory_records.jsonl`;
3. `summary.json`;
4. `artifact_manifest.json`;
5. `SHA256SUMS`.

The verifier authenticates the prior postmortem again and requires the exact
source, job, protocol, checkpoint hashes, train-input evidence, record count,
unique arm/seed/family matrix, finite arithmetic, family/template alignment,
recomputed scores and summaries, manifest coverage, checksums, and absence of
unexpected files. Successful verification returns
`verification_status="pass"`. Scientific outcome never controls that status.

## Execution record

Reviewer-authorized job `3355342` completed at the exact implementation commit
with exit `0:0` in `00:04:09`; batch MaxRSS was `786736K`. All 14 focused
tests passed with zero skips. The complete five-file artifact and its local
checksums validate, and the job records an exact verifier pass. The separate
job-3355134 artifact required to rerun that verifier was not part of the
downloaded audit evidence, so its embedded hashes and verification result are
execution evidence rather than a second independent local authentication.

Across 1,221 records per arm, `P_true` was `1.0`. Flat recorded
`P_zero=0.8378378378378378` and authenticated
`P_mean=0.7903357903357904`; typed graph recorded
`P_zero=0.5855855855855856` and authenticated
`P_mean=0.8361998361998362`. The exact-zero output therefore remains
structurally informative, consistent with unchanged decoder-side/scalar-path
inputs, while the true-to-zero drops also show memory-value contribution.
The typed-graph zero-versus-mean contrast is consistent with useful
information in its batch-mean vector. None of these descriptive differences
has a diagnostic threshold, identifies a causal input or defect, establishes
encoder superiority, or supplies a development conclusion. Full audit details
are in the [job-3355342 execution
record](../experiments/ge1_stage6_zero_memory_3355342.md).

## Access and runner boundary

The separate CPU runner requests one CPU, one thread, 2 GB, and 30 minutes. It
binds only the repository read-only, each of the six exact checkpoint files
read-only, the narrow train package read-only, the exact postmortem artifact
read-only, and the new artifact parent read-write. It does not broadly bind
retained work and has no development, corpus, broad manifest, external or
unrelated checkpoint, model, repaired-artifact, RR, ER/test, IID,
history-depth, geometry-extrapolation, or protected-partition bind.

Focused tests must run with zero skips. The runner invokes the diagnostic once,
verifies the artifact and checksums, emits terminal success/failure/timeout
telemetry, and contains no submission command.

Completion of job `3355342` consumes only its separately granted diagnostic
authority. It does not authorize another transfer, input access, inference,
or submission. Development, protected data, Stage 6 continuation, C8, repair,
and threshold changes remain closed.
