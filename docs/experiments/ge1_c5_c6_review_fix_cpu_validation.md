# GE1 C5/C6 Review-Fix Authoritative CPU Validation

## Record status

| Item | Value |
|---|---|
| Experiment ID | `ge1-c5-c6-review-fix-cpu-3344431` |
| Record status | `verified` |
| Scientific role | Combined C5/C6 review-fix infrastructure and train-only engineering revalidation |
| Branch | `graph-v1-experiment-record` |
| Source commit | `d29dc8299d32186907eb16d05c6102fcececf32e` |
| Source parent | `dae3a346f07bc1ede048437b26bac608a30a814f` |
| Slurm job | `3344431` |
| Compute host | `adroit-h11n2` |
| Working tree | Clean before and after validation |
| Execution date | August 8, 2026 |

## Question and predetermined decision rule

The validation asked whether exact C5/C6 review-fix commit
`d29dc8299d32186907eb16d05c6102fcececf32e` satisfies the combined
shared-decoder, training, recovery, autonomous-evaluation, metrics-v2, reporting,
provenance, and train-only engineering gate under Python 3.8 and PyTorch 1.11
on CPU.

The predetermined gate required:

- a clean checkout of the exact commit before and after validation;
- Python 3.8, PyTorch 1.11, and CPU-only execution;
- all 75 focused C5/C6 tests to run and pass with zero skips, including the
  new shared-decoder provenance integration test and prefix-envelope test;
- all 163 tests in the complete graph-encoder suite to run and pass with zero
  skips;
- successful model-data, flat-baseline, graph-baseline, representation, and
  controlled-data regression suites, with only the four established
  immutable-external-bundle skips permitted in the flat-baseline suite;
- successful documentation validation, graph-encoder compilation, Python 3.8
  grammar parsing, and target-leakage/common-decoder source audit;
- one complete two-epoch, 407-family `operation_template.train` engineering
  smoke for each encoder arm;
- strict epoch-2 checkpoint reload and one complete
  `GE1-C6-METRICS-v2` / `GE1-PRIMARY-REPORT-v1` record per arm;
- all five intervention fields to be present as finite values or
  reason-bearing structured nulls, `P_shuffle` donor agreement to be
  available, and `P_true` and `P_mean` donor agreement to be structurally
  unavailable; and
- zero development, RR systematic, ER test, IID, history-depth,
  geometry-extrapolation, scientific-training, C7, or later-stage access.

Any source or environment mismatch, dirty tree, focused or complete-suite
skip, unpermitted regression skip, test failure or error, incomplete smoke,
artifact-contract failure, protected access, or missing terminal-success
event would fail the gate.

## Inputs and partition authority

| Item | Verified value |
|---|---|
| Authorized manifest | `operation_template` |
| Authorized payload | Complete 407-family `operation_template.train` partition |
| Engineering arms | `flat`, `typed_graph` |
| Seed | 2026 |
| Engineering epochs | 2 per arm |
| Manifest opened | `true` |
| Train payload opened | `true` |
| Development opened | `false` |
| RR systematic opened | `false` |
| ER test opened | `false` |
| IID opened | `false` |
| History-depth opened | `false` |
| Geometry-extrapolation opened | `false` |
| Scientific training performed | `false` |
| C7 or later performed | `false` |

Both smokes used the guarded C1 train loader. The run did not authorize or
open development or any protected partition. The two-epoch executions are
engineering smokes, not selected scientific runs.

## Configuration under review

The validated commit contains the five accepted C5/C6 review fixes governed
where necessary by accepted
[ADR-0005](../decisions/ADR-0005-ge1-primary-reporting-and-metrics-v2.md):

1. an explicit inherited-V6 entry-point compatibility shim, corrected GE1
   memory provenance, and a runtime rejection check for the inherited
   autonomous converter path;
2. required finite-or-reasoned reporting for `P_true`, `P_shuffle`, `P_mean`,
   `R_shuffle`, and `R_mean`, with metrics schema v2 and primary-report
   contract v1;
3. field-by-field prefix validation with semantic fields preserved and
   converter bookkeeping recomputed under the shortened count;
4. donor-template agreement for `P_shuffle` against the
   random-distinct-family-donor baseline, with structured unavailability for
   non-donor conditions; and
5. an explicit authorized-seed validator in the canonical shared decoder.

These fixes do not change the protected-partition policy, scientific training
budget, comparison endpoint, or frozen memory-intervention thresholds.

## Environment

| Item | Verified value |
|---|---|
| Cluster / host | Princeton Adroit / `adroit-h11n2` |
| Slurm job | `3344431` |
| Scheduler state / exit | `COMPLETED` / `0:0` |
| Elapsed / peak RSS | 8 minutes 37 seconds / 609,048 KiB |
| Python | CPython 3.8.13 |
| PyTorch | 1.11.0, CUDA build 11.3 |
| CUDA available | `false` |
| Execution device | CPU |
| PyTorch CPU threads | 1 |
| Container | `/scratch/network/km6349/skexgen.sif` |
| Container content hash | Not preserved |

## Execution

The committed runner
`prototype/graph_encoder/adroit/c6_cpu_validation.slurm` checked the exact
commit and clean source state, entered the fixed container, verified the
runtime, and executed every gate in fail-fast order. It opened the authorized
train partition only after the focused, complete, regression, documentation,
compilation, grammar, and source-audit gates passed. The terminal-success
event was emitted only after both arm smokes and the repeated exact-commit and
clean-tree checks passed.

## Verified results

| Gate | Result |
|---|---|
| Focused C5/C6 suite | 75/75 passed; 0 failures, 0 errors, 0 skips |
| Shared-decoder provenance integration | Passed against the real inherited converter path |
| Prefix-envelope integration | Passed with exact shortened-count bookkeeping recomputation |
| Complete graph-encoder suite | 163/163 passed; 0 failures, 0 errors, 0 skips |
| Model-data regressions | 86/86 passed |
| Flat-baseline regressions | 549 run; 545 passed and exactly 4 authorized skips |
| Graph-baseline regressions | 17/17 passed |
| Representation regressions | 40/40 passed |
| Controlled-data regressions | 51/51 passed |
| Documentation validation | Passed for 71 Markdown files |
| Graph-encoder compilation | Passed |
| Python 3.8 grammar | Passed for all 22 enumerated C5/C6 source/test files |
| Target-leakage/common-decoder audit | Passed |
| Flat train-only smoke | Passed at epoch 2; strict checkpoint reload; complete v2 metrics record |
| Typed-graph train-only smoke | Passed at epoch 2; strict checkpoint reload; complete v2 metrics record |
| Metrics/reporting contract checks | 2/2 passed; one per arm |
| Final exact-commit and clean-tree gate | Passed |
| Terminal success marker | Present |

Both artifacts identified `GE1-C6-METRICS-v2`,
`GE1-C6-TRAIN-ONLY-SMOKE-v2`, and `GE1-PRIMARY-REPORT-v1`. Each contained all
five required intervention fields. Each reported `P_shuffle` donor agreement
as available over 407 assignments against the
`random_distinct_family_donor` baseline, and `P_true` and `P_mean` donor
agreement as structurally unavailable. Both smoke events declared train-only
payload access and retained every development/protected access flag as false.

The terse flat-suite progress stream contains four and only four skip
positions. Mapping its 549 characters to the repository's 549-test unittest
discovery order resolves them to the established immutable-external-bundle
tests:

1. `test_five_capsule_regressions_are_a_pinned_golden_cohort`;
2. `test_malformed_parent_fails_before_scientific_loading`;
3. `test_parent_equivalent_arms_reconcile_pinned_records`; and
4. `test_real_bundle_two_family_golden_reconciliation`.

No other suite emitted a skip summary.

## Decision

The combined C5/C6 review fixes pass their authoritative Python 3.8/PyTorch
1.11 CPU revalidation gate at exact commit
`d29dc8299d32186907eb16d05c6102fcececf32e`. The prior C5 and C6 validation
records remain valid historical evidence for their validated commits; job
`3344431` establishes authoritative runtime coverage for the subsequent
review-fix source and v2 artifact contracts. C7 and every later stage remain
unstarted.

## Interpretation

This result establishes that the corrected shared-decoder provenance path,
prefix envelope, seed validation, donor reporting, structured primary fields,
and v2 metrics artifact contract execute together in the authoritative
runtime. It also revalidates the full C5/C6 machinery and both authorized
train-only arm smokes while preserving every staged-access boundary.

## Limitations and claims not supported

This is an infrastructure and train-only engineering validation, not a
scientific encoder comparison. It does not establish decoder sufficiency,
trained memory sensitivity, an arm treatment effect, development performance,
RR systematic generalization, ER test performance, editing behavior, or any
C7 or later outcome. Only seed 2026 and two engineering epochs were used; the
frozen 50-epoch, multi-seed experiment was not run. The four permitted
flat-baseline skips concern unavailable immutable external replay bundles and
are outside both zero-skip graph-encoder gates.

The two generated `c6_metrics.json` artifacts were validated by the runner
and summarized in stdout, but independent copies were not supplied for this
local audit. This record validates their required identities and fields, not
unreported metric values. The container path and runtime versions were
recorded, but the container image was not cryptographically hashed.

The terminal transcript preserves an initial incomplete-bundle transfer and a
later post-run inspection command entered with a placeholder job identifier.
Both commands failed visibly and were corrected before the source checkout or
artifact-integrity evidence used for this decision. Neither was executed by
job `3344431` or bypassed a runner gate.

## Artifacts and integrity

The locally supplied raw artifacts were read directly and hashed during the
final audit:

| Artifact | Size | SHA-256 |
|---|---:|---|
| `c6-cpu-3344431.out` | 6,319 bytes | `cdbe714c4123b003bba7d3a2ec9b5f2ecea63ca0a98eba4fcbcd027dc46d25cb` |
| `c6-cpu-3344431.err` | 11,622 bytes | `f25cd214ab3a94d741352d7f5f21d86d48558a5eec52df8eb12e716c1eb70bad` |
| Complete terminal transcript | 32,983 bytes | `9dc4bf4b2cca747b7872726581c4e7ab447f97b6fc410c4b931e6045f355750a` |

The stdout and stderr hashes exactly match the values computed on Adroit
before download. The source was transported in a verified complete-history
Git bundle with SHA-256
`4694159701ce17d67f73abcd68b23ad7c4a4967e66818e42bebfbcc2a036b407`.

Local evidence paths at audit time were:

```text
/Users/krishaymaskara/Downloads/c5-c6-review-validation-3344431/c6-cpu-3344431.out
/Users/krishaymaskara/Downloads/c5-c6-review-validation-3344431/c6-cpu-3344431.err
/Users/krishaymaskara/.codex/attachments/1771bbd9-260d-4e20-80fe-8a6d44ce506f/pasted-text.txt
```

The scheduler logs and generated run namespace remain at:

```text
/scratch/network/km6349/c6_validation_runs/c6-cpu-3344431.out
/scratch/network/km6349/c6_validation_runs/c6-cpu-3344431.err
/scratch/network/km6349/c6_validation_runs/c6-3344431
```

The per-arm metrics records remain under the run namespace at
`flat/c6_metrics.json` and `typed_graph/c6_metrics.json`. Their independent
local sizes and hashes were not supplied and are not claimed here.

## Related records

- [ADR-0005](../decisions/ADR-0005-ge1-primary-reporting-and-metrics-v2.md)
- [C5 shared-decoder contract](../specifications/ge1_shared_decoder_contract.md)
- [C6 measurement contract](../specifications/ge1_c6_measurement_contract.md)
- [Authoritative C5 validation](ge1_c5_cpu_validation.md)
- [Authoritative C6 validation](ge1_c6_cpu_validation.md)
- [GE1 implementation plan](../specifications/graph_encoder_implementation_plan.md)
- [Current project status](../status.md)
