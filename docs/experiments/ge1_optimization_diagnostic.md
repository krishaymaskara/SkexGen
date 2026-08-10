# GE1 Post-C7 Optimization-Sufficiency Diagnostic

## Record status

| Item | Value |
|---|---|
| Experiment ID | `ge1-c7-optimization-diagnostic-3344907` |
| Record status | `verified-local` |
| Scientific role | Separate train-only diagnosis of formal C7-v1 optimization sufficiency |
| Scientific outcome | `undertraining_supported_both_arms` |
| Source commit | `fbc6073f63da9f0e10b5db8c0c0d4786a48ce0c0` |
| Slurm job | `3344907` |
| Compute host | `adroit-h11n2` |
| Scheduler | `COMPLETED`, exit `0:0`, elapsed `00:07:46`, MaxRSS `608184K` |
| Working tree | Exact detached commit and clean before and after execution |
| Execution date | August 9, 2026 |
| Audit date | August 10, 2026 |

## Question and predetermined decision rule

Formal C7-v1 job `3344505` failed autonomous exact sufficiency at epoch 50
while both training losses were still improving. The separate diagnostic
asked whether the unchanged flat and typed-graph arms could become exactly
sufficient on the same four train families under a predetermined 500-update
fresh-start trajectory.

Only autonomous `P_true` at updates 50, 100, 200, and 500 could establish
sufficiency. Every family required exact node sequence, exact typed graph,
strict conversion, and complete executable validity; `EE` and `RE` also
required exact `depends_on`. If both arms passed at any predetermined
milestone, the frozen interpretation was
`undertraining_supported_both_arms`.

The diagnostic could not alter C7-v1, authorize Stage 6, invoke decoder
repair, begin C8, or open a protected partition.

## Inputs and partition authority

| Item | Verified value |
|---|---|
| Manifest | `operation_template` |
| Manifest SHA-256 | `a9ac86a6dede054fbbba57e0906b210bab26036c3f5c150b332038f78d2dadb7` |
| Authorized partition | train |
| Selected templates | `E`, `R`, `EE`, `RE` |
| Selected-set SHA-256 | `c07e90675099cb2ea58575fe91683dc80b70cf3fa85fa6fa8de0c44dbe9783d6` |
| Seed | 2026 |
| Manifest / selected train payload opened | `true` / `true` |
| Development, RR, ER, IID, history-depth, geometry-extrapolation | all `false` |
| C8 or later performed | `false` |

Selection was metadata-only and reproduced the exact formal C7-v1 four-family
cohort before any payload access.

## Environment

| Item | Verified value |
|---|---|
| Host | `adroit-h11n2` |
| Python | CPython 3.8.13 |
| PyTorch | 1.11.0, CUDA build 11.3 |
| Device | CPU, CUDA unavailable, one PyTorch thread |
| Slurm identity | `3344907` across runtime, events, checkpoints, and verification |
| Source-tree SHA-256 | `b3daa9a58da19b3cf94e4f68592608959a4c68c223ece7aa14d0405ff90c4489` |

## Execution

Both arms started from one fresh matched seed-2026 initialization with
byte-identical shared components and disjoint mutable state. Neither resumed
from C7-v1. Each trained continuously for 500 optimizer updates over four
families, or 2,000 family presentations, with no early stop or
outcome-dependent extension.

The run retained one checkpoint every 25 updates, producing 20 checkpoints
per arm. Updates 50, 100, 200, and 500 were written before evaluation and
strictly reloaded into fresh models. Eight autonomous milestone measurements
were recorded.

## Results and findings

### Validation and integrity

| Gate | Result |
|---|---|
| Focused diagnostic suite | 20/20, zero skips |
| Complete graph-encoder suite | 238/238, zero skips |
| Model-data / flat / graph / representation / controlled regressions | 86 / 549 / 17 / 40 / 51 run; only four pre-existing flat skips |
| Documentation | 77 Markdown files passed |
| Compilation, Python 3.8 grammar, import/export, source-access audit | passed |
| Exact commit and clean tree | passed before and after execution |
| Checkpoints | 40 verified and loaded for Slurm provenance |
| Milestones | eight verified |
| Artifact | canonical, finite, immutable, and checksum-verified |

### Autonomous trajectory

| Arm | Update | Train loss | Exact nodes | Exact graph | Complete validity | Exact `depends_on` | Exact-sufficient |
|---|---:|---:|---:|---:|---:|---:|---:|
| Flat | 50 | 1.43294 | 4/4 | 0/4 | 0/4 | 0/2 | no |
| Flat | 100 | 0.31856 | 4/4 | 2/4 | 2/4 | 2/2 | no |
| Flat | 200 | 0.08863 | 4/4 | 4/4 | 4/4 | 2/2 | yes |
| Flat | 500 | 0.01483 | 4/4 | 4/4 | 4/4 | 2/2 | yes |
| Typed graph | 50 | 1.50635 | 3/4 | 0/4 | 0/4 | 0/2 | no |
| Typed graph | 100 | 0.32961 | 4/4 | 2/4 | 2/4 | 2/2 | no |
| Typed graph | 200 | 0.08798 | 4/4 | 4/4 | 4/4 | 2/2 | yes |
| Typed graph | 500 | 0.01514 | 4/4 | 4/4 | 4/4 | 2/2 | yes |

Strict conversion was 4/4 for both arms at every milestone. Both arms first
became exact-sufficient at update 200 and remained exact at update 500.

At update 500 the memory interventions were:

```text
flat:        P_true=1.00, P_shuffle=0.00, P_mean=0.00
typed_graph: P_true=1.00, P_shuffle=0.00, P_mean=0.50
```

Only `P_true` exactness governed the diagnostic interpretation.

## Decision and interpretation

The preregistered category is correctly:

```text
undertraining_supported_both_arms
```

The same unchanged arms and decoder that failed at update 50 fit all four
training families autonomously by update 200. This supports an inadequate
formal C7-v1 optimization budget for both arms on the tiny cohort rather than
an inability of the unchanged architecture to represent those four targets.

The record also states:

```text
original_c7_result_changed=false
stage6_authorized=false
decoder_repair_invoked=false
c8_or_later_performed=false
```

Formal C7-v1 remains failed. The result does not establish scaled or
generalization performance and does not automatically authorize any later
stage. Accepted
[ADR-0008](../decisions/ADR-0008-ge1-c7-v2-200-epoch-protocol.md) uses this
evidence prospectively for a separately versioned fixed-epoch-200 C7-v2
protocol while deferring, not erasing, the repair path.

## Limitations

- Only four selected train families and one seed were measured.
- The diagnostic did not run the 32-family scaled gate or any protected
  evaluation.
- Exactness at update 200 does not prove sufficiency on 32 or 407 families.
- The complete checkpoint binaries remain external and were not copied into
  this repository; the remote runner checksum-verified and loaded all 40.
- The container path and package versions were verified, but the container
  image itself was not cryptographically hashed in this record.

## Artifacts and integrity

| Artifact | SHA-256 |
|---|---|
| `artifact_manifest.json` | `1b30f7fb1bd16b914526e286aecd45dac3a04cd882297696dcbd86c810cd5eec` |
| `metrics.jsonl` | `aaf3114ed6e712696eb50e061da83e79f0399555746e1383a30a040d445a80c3` |
| `resolved_config.json` | `5a412f61a1b7d56ff70ffd5e2a4c419bbaa11fd113ae27fd3e2addb8b687831a` |
| `SHA256SUMS` | `06293a644f4ff41e3c2f3675f519478a9724339956e79c957cdf045e0b0f6b05` |
| stdout | `91cb69933b2a822e64255633535ddf3c425902d46ae519d23f949fce8b87430f` |
| stderr | `729355b77e4283f88de8344264670ecf32db851103f4490b9b529533d9bf753a` |

The manifest contains 42 entries: 40 checkpoints plus metrics and resolved
configuration. `SHA256SUMS` contains 43 ordered paths by adding the manifest;
the finalized directory contains 44 regular files by adding `SHA256SUMS`.

## Related records

- [Formal C7-v1 record](ge1_c7_pilot.md)
- [Preflight attempt 3344896](ge1_optimization_diagnostic_preflight_attempt_3344896.md)
- [ADR-0007](../decisions/ADR-0007-ge1-c7-optimization-sufficiency-diagnostic.md)
- [ADR-0008](../decisions/ADR-0008-ge1-c7-v2-200-epoch-protocol.md)
- [Diagnostic contract](../specifications/ge1_c7_optimization_sufficiency_diagnostic.md)
- [C7-v2 contract](../specifications/ge1_c7_v2_execution_contract.md)
