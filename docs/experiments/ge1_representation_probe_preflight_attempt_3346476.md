# GE1 Representation-Probe Preflight Attempt 3346476

## Record status

| Item | Value |
|---|---|
| Experiment ID | `ge1-representation-probe-preflight-3346476` |
| Record status | `reported-only` |
| Scientific role | Authoritative-environment preflight attempt; no scientific probe execution |
| Outcome | Focused runtime-test implementation failure before source artifact, checkpoint, manifest, or corpus access |
| Source commit | `4baf596812471ca76a15b09df89ce38d0a4bf30c` |
| Slurm job | `3346476` |
| Scheduler state | `FAILED`, exit `1:0`, elapsed `00:00:13` |
| Record date | August 12, 2026 |

## Question and preflight boundary

The accepted representation-probe runner requires every environment, test,
regression, documentation, source, and access preflight to pass before it may
bind the immutable repaired-sufficiency artifact or open the authorized
operation-template train cohort. Job `3346476` stopped in the focused
real-PyTorch suite and never crossed that boundary.

## Results and findings

The authoritative transcript reports that the exact-commit, clean standalone
checkout, CPython `3.8.13`, PyTorch `1.11.0`, CPU-only, and one-thread gates
passed. All 17 pure representation-probe contract tests passed. Six of seven
real-PyTorch runtime tests passed; the failure was
`test_target_free_extraction_records_exact_A_to_D_shapes`. The complete
graph-encoder suite and every later regression gate were not reached.

## Access and execution declarations

All of the following remained false:

```text
source_repaired_artifact_accessed
source_checkpoint_accessed
operation_template_manifest_accessed
operation_template_train_payload_accessed
scaled_train_payload_accessed
development_accessed
systematic_rr_accessed
test_er_accessed
iid_accessed
history_depth_accessed
geometry_extrapolation_accessed
ge1_training_performed
backward_pass_performed
ge1_optimizer_constructed
checkpoint_written
repair_implemented_or_invoked
stage6_performed
c8_or_later_performed
scientific_probe_execution_performed
```

No finalized or partial scientific artifact was produced.

## Failure diagnosis and correction

The extractor conflated two distinct frozen encoder outputs:

- `encoded.memory` is `from_codebook(prequant)`, has shape `[B, 2, 32]`, and
  is the shared decoder input.
- `encoded.prequant` is the continuous `to_codebook` bottleneck, has shape
  `[B, 2, 16]`, and is the representation frozen for feature D.

Feature D requires the prequant row flattened in latent-token-major order to
32 values, followed by the autonomous predicted-operation-type and
canonical-slot one-hots for 36 values total. The failed implementation instead
required decoder-facing memory to have shape `[B, 2, 16]`, so the runtime test
correctly stopped before decoder inference or any scientific access.

The narrow correction validates both tensors separately, continues to pass
only `encoded.memory` to the decoder, and constructs only D's first 32 values
from detached `encoded.prequant`. Features A-C, the frozen D dimension, probe
statistics, artifacts, access rules, source identities, models, and accepted
scientific question do not change.

## Decision and interpretation

Job `3346476` is strictly a preflight implementation failure. It provides no
scientific result about representation accessibility, either encoder arm,
operation magnitudes, or the finalized repaired-sufficiency result. It does
not authorize corpus or checkpoint access, probe execution, repair, Stage 6,
or C8. A replacement exact-commit job must rerun every preflight from the
beginning; no output from this attempt is reusable as scientific evidence.

## Evidence limitation

This record is based on the designated reviewer's authoritative transcript.
Separate raw stdout, stderr, scheduler files, and hashes were not supplied, so
the evidence state is `reported-only` rather than `verified-local`.

## Artifacts and integrity

No finalized or partial scientific artifact, resolved configuration, feature
file, labels, metrics, artifact manifest, or `SHA256SUMS` publication exists
from job `3346476`. The transcript establishes the preflight boundary and
reported failure, but raw-file integrity cannot be independently rechecked
without the separate logs and hashes.

## Related records

- [ADR-0011](../decisions/ADR-0011-ge1-repaired-representation-probe.md)
- [Representation-probe contract](../specifications/ge1_repaired_representation_probe.md)
- [Repaired-sufficiency implementation validation](ge1_repaired_sufficiency_cpu_validation.md)
- [Current project status](../status.md)
