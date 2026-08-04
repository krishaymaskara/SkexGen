# B0 Graph V1 Engineering Readiness and C1 Validation

## Record status

| Item | Value |
|---|---|
| Experiment ID | `b0-graph-v1-engineering-readiness` |
| Record status | `verified` |
| Evidence state | `verified-local` |
| Scientific role | Pre-pilot engineering failures, provenance hardening, and authoritative C1 regression validation |
| Branch | `graph-profile-decoder` |
| Source range | `6484579` through `089b9f3`; C1 validation at `0cd09ed` |
| Working tree | Clean for accepted source commits |
| Slurm/local run IDs | `3338803`, `3338863`, `3338945`, `3339188`, `3339644`, `3339745`, `3339787`, `3340201`, `3341940`, `3341971`, `3341972` |
| Execution dates | August 2-3, 2026 |

This record preserves the engineering chain that made the two scientific
Graph V1 pilots trustworthy. Scheduler accounting and stdout/stderr for the
entire chain were recovered locally on August 3, 2026, together with the
incomplete job `3339787` namespace and the accepted scientific run
directories. Scientific outcomes remain recorded separately.

## Question and predetermined decision rule

The readiness work asked whether the graph pilot path could be trusted to:

- construct one internally consistent generated node sequence;
- tensorize and decode directed typed edges with the intended orientation;
- enforce legal masks without target-derived repair;
- preserve corpus closure and protected-partition rules;
- authorize exact clean source state before training and checkpointing;
- save and strictly reload compatible checkpoints;
- reproduce validation after reload; and
- retain the frozen Flat V4-V6 regression contracts.

A readiness failure did not count as a scientific model result. The
scientific pilot could begin only after the production-shaped path and focused
tests passed at a reviewed clean commit.

## Inputs and partition authority

| Item | Value |
|---|---|
| Production corpus authority | 544 train / 68 IID validation / 68 held-out IID test |
| Readiness fixtures | Bounded synthetic and production-shaped fixtures |
| Systematic partition accessed | `false` |
| Held-out test partition accessed | `false` |
| Scientific checkpoint selection performed | `false` for failed readiness jobs |

Job `3339787` processed the 544 training examples in epoch 1 but failed before
validation and checkpoint completion. Its state was not resumed or reused.

## Configuration

The readiness path exercised the same Graph V1 contract later used by the
scientific pilots: ordered source/destination pairs, six directed edge
classes, constrained V6 node rollout, independent strict conversion, fixed
two-epoch pilot control flow, and strict provenance.

The C1 validation retained all scientific settings and changed only the
directed-position branch under the one-correction protocol.

## Environment

The authoritative C1 validation ran on Adroit with:

```text
Python: 3.8.13
PyTorch: 1.11.0
CUDA build: 11.3
execution: CPU
```

The recovery contains per-job scheduler records and logs. Separate resolved
environment files were not present for the earlier readiness failures.

## Execution

### Initial Graph V1 readiness sequence

| Job | Observed failure | Repair and resulting commit |
|---|---|---|
| `3338803` | Duplicate controlled-data fixture family | Corrected fixture construction in `407b5db` |
| `3338863` | Missing `torch` import in graph prediction conversion | Added import in `74e7a43` |
| `3338945` | Teacher graph mask used node identity inconsistent with reconstructed nodes | Aligned teacher node-type masking in `daaa86a` |
| `3339188` | Individually legal current nodes did not form one complete grammar path | Added `authoritative_graph_node_type_ids` and raw shadow evidence in `dd241b1` |
| `3339644` | Focused test referenced graph decoder without import | Corrected import in `0eeda79` |
| `3339745` | Production-shaped Graph V1 readiness completed successfully | Authorized the first pilot submission |
| `3339787` | Source-provenance rejection after all 68 epoch-1 steps | Added explicit-root branch/commit/status/digest provenance in `119d2be` |
| `3340201` | Test fixture placed nested `.git` inside corpus closure | Separated corpus, source-repository, and output roots in `089b9f3` |
| `3341940` | Corrected provenance/readiness workflow completed successfully | Authorized scientific job `3341942` at `089b9f3` |

The accepted provenance module resolves branch identity with
`git symbolic-ref --quiet --short HEAD`, collects porcelain-v1 status and a
deterministic source digest from an explicit repository root, authorizes
source state before training, and rechecks it before checkpoint save and
strict reload.

### C1 authoritative validation

Before Adroit validation, the local Mac environment used Python `3.14.5` and
had no compatible PyTorch or Slurm installation. Dependency-free and syntax
coverage produced:

| Local suite | Discovered | Passed | Dependency skips |
|---|---:|---:|---:|
| Graph V1 | 17 | 6 | 11 |
| Flat V6 | 27 | 10 | 17 |
| Flat V5 | 21 | 9 | 12 |
| Flat V4 | 27 | 9 | 18 |
| Full flat suite | 549 | 299 | 250 |
| Model-data | 86 | 61 | 25 |

Representation tests passed 40/40, controlled-data tests passed 51/51, and
Python 3.8 grammar parsing passed for 232 files. Compilation, Slurm Bash
syntax, and `git diff --check` also passed. These local results were not used
as a substitute for the zero-skip authoritative Torch validation.

Job `3341971` failed a static validation-harness assertion that searched
production source for the current C1 commit even though the model metadata
correctly recorded the parent graph commit. It did not change production code
or run a scientific pilot.

Corrected job `3341972` completed in 5 minutes 57 seconds.

## Verified results

Job `3341972` passed:

| Validation layer | Result |
|---|---|
| Focused Graph V1/C1 | 17 run, 0 failures, 0 errors, 0 skips |
| Focused Flat V6 | 27 / 27 |
| Focused Flat V5 | 21 / 21 |
| Focused Flat V4 | 27 / 27 |
| Full flat suite | 549 run, 0 failures, 0 errors, 4 authorized skips |
| Representation suite | Passed |
| Controlled-data suite | Passed |
| Model-data suite | Passed |
| Python 3.8 grammar | Passed |
| Targeted compilation and full `compileall` | Passed |
| Slurm Bash syntax | Passed |
| `git diff --check` | Passed |

Focused C1 coverage included directed-position sensitivity, source/destination
asymmetry, target-free autonomous decoding, addition before masking, unchanged
mask semantics, finite gradient flow through all new parameters, parameter
counts, intentional incompatibility with initial Graph V1 checkpoints, strict
C1 checkpoint reload, two-epoch control flow, and a production training/save
smoke.

The four full-flat skips were the established immutable-external-bundle tests:

1. `test_five_capsule_regressions_are_a_pinned_golden_cohort`;
2. `test_malformed_parent_fails_before_scientific_loading`;
3. `test_parent_equivalent_arms_reconcile_pinned_records`;
4. `test_real_bundle_two_family_golden_reconciliation`.

## Decision

The initial path was accepted for a scientific pilot at commit `089b9f3...`.
The C1 implementation was accepted for its final scientific pilot after job
`3341972`. None of the failed readiness or validation-harness jobs was treated
as a checkpoint or scientific result.

## Interpretation

The sequence establishes engineering readiness and intervention isolation. It
does not itself establish graph-model quality. Its scientific value is that
the later 22/68 results can be interpreted as model outcomes rather than
known loader, mask, node-sequence, provenance, checkpoint, or fixture defects.

## Limitations and claims not supported

- The original scheduler records and stdout/stderr are recovered, but separate
  resolved environment files are absent for jobs before `3341972`.
- Full output directories were not created or preserved for every short
  readiness failure; their logs and repository regression fixtures are the
  primary evidence for those failures.
- Passing tests does not establish scientific validity or generalization.
- No systematic or test-partition result follows from readiness validation.
- Job `3339787` is not a one-epoch scientific pilot and must not be resumed.

## Artifacts and integrity

| Artifact | Location | SHA-256 | Availability |
|---|---|---|---|
| Complete constrained-decoder recovery archive | `/Users/krishaymaskara/research/audited-runs/skexgen-constrained-decoder-primary-evidence-20260803.tar.gz` | `5bd60fd884b23d607f452d82e8da1e71c97a119e9d8ad92ef0f3d91dbc4f2dcc` | Present; internal checksums verified |
| Readiness scheduler evidence | Archive jobs listed above | Covered by internal `SHA256SUMS` | Present and verified |
| Incomplete initial Graph namespace | Archive job `3339787` | Covered by internal `SHA256SUMS` | Present and verified |
| C1 validation output | Archive job `3341972` | Covered by internal `SHA256SUMS` | Present and verified |
| Accepted implementations and regression tests | Git commits listed above | Git object identity | Present |
| Scientific result bundles | Separate initial/C1 local archives | Recorded in linked experiment records | Present |

## Reproduction and validation

The outer archive hash matches on Adroit and the Mac, and all internal file
checksums pass. The scheduler states and elapsed times above were checked
against `sacct.txt`; this recovered the historical jobs and did not rerun them.

## Related records

- Result enabled by initial readiness: [Initial Graph V1](b0_graph_v1_initial.md)
- Result enabled by C1 validation: [Graph V1 C1](b0_graph_v1_c1.md)
- Flat comparison: [Frozen Flat V6](b0_constrained_flat_v6.md)
- Freeze decision: [ADR-0003](../decisions/ADR-0003-freeze-graph-v1.md)
- Package contract: [Graph-baseline README](../../prototype/graph_baseline/README.md)
