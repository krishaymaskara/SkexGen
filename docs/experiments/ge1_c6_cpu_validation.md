# GE1 C6 Authoritative CPU Validation

## Record status

| Item | Value |
|---|---|
| Experiment ID | `ge1-c6-cpu-3344367` |
| Record status | `verified` |
| Scientific role | C6 infrastructure, recovery, metrics, and train-only engineering validation |
| Branch | `graph-v1-experiment-record` |
| Source commit | `c88e967b96dd201fedcd495fa8d5ddaddd02caf3` |
| Source parent | `68c66ea4b4c1e1d01b9b9dc061ace1149bca7c5a` |
| Slurm job | `3344367` |
| Compute host | `adroit-h11n2` |
| Working tree | Clean before and after validation |
| Execution date | August 8, 2026 |

## Question and predetermined decision rule

The validation asked whether corrected exact C6 commit
`c88e967b96dd201fedcd495fa8d5ddaddd02caf3` satisfies the complete governed
training, recovery, autonomous evaluation, metrics, intervention, provenance,
and train-only engineering gate under Python 3.8 and PyTorch 1.11 on CPU.

The predetermined gate required:

- a clean checkout of the exact commit before and after validation;
- Python 3.8, PyTorch 1.11, and CPU-only execution;
- all 37 focused C6 tests to run and pass with zero skips;
- all 147 tests in the complete graph-encoder suite to run and pass with zero
  skips;
- successful model-data, flat-baseline, graph-baseline, representation, and
  controlled-data regression suites;
- successful documentation validation, graph-encoder compilation, Python 3.8
  grammar parsing, and the target-leakage/common-decoder source audit;
- one complete two-epoch, 407-family `operation_template.train` engineering
  smoke for each encoder arm;
- strict epoch-2 checkpoint reload and one complete three-condition metrics
  record per arm; and
- zero development, RR systematic, ER test, IID, history-depth,
  geometry-extrapolation, scientific-training, C7, or later-stage access.

Any failure, error, required-suite skip, environment or source-identity
mismatch, protected access, incomplete smoke, missing metrics record, or
missing terminal-success event would fail the gate.

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

Both smokes used the C1 guarded loader, so opening the train payload required
the frozen manifest identity and complete train assignment to pass. The run
did not authorize or expose any development or protected partition. The
two-epoch executions are infrastructure smokes, not selected scientific runs.

## Configuration under review

The validated commit contains the common deterministic C6 training loop,
finite-loss and finite-gradient enforcement, global-norm clipping, atomic
optimizer/RNG recovery checkpoints, provenance revalidation, target-free
three-condition autonomous inference, executable-prefix scoring,
physical-family macro metrics, memory interventions, capacity and receptive-
field reports, timing, and peak-memory measurement.

It also contains the corrections identified by failed jobs `3344337` and
`3344363`: a device-local `torch.long` autonomous node-count tensor, a missing
test import, and a test-only tensor-aware comparison of uninterrupted versus
resumed autonomous predictions and timing-neutral metrics. The final
correction changed no production model or protocol behavior.

## Environment

| Item | Verified value |
|---|---|
| Cluster / host | Princeton Adroit / `adroit-h11n2` |
| Slurm job | `3344367` |
| Scheduler state / exit | `COMPLETED` / `0:0` |
| Elapsed / peak RSS | 8 minutes 39 seconds / 602,800 KiB |
| Python | CPython 3.8.13 |
| PyTorch | 1.11.0, CUDA build 11.3 |
| CUDA available | `false` |
| Execution device | CPU |
| PyTorch CPU threads | 1 |
| Container | `/scratch/network/km6349/skexgen.sif` |
| Container content hash | Not preserved |

## Execution

The preserved repository runner
`prototype/graph_encoder/adroit/c6_cpu_validation.slurm` checked the exact
commit and clean source state, entered the fixed container, verified the
runtime, and executed every gate in fail-fast order. It opened the authorized
train partition only after all fixture, regression, documentation,
compilation, grammar, and source-audit gates passed. The terminal-success event
was emitted only after both arm smokes and the final exact-commit and clean-
tree checks passed.

## Verified results

| Gate | Result |
|---|---|
| Focused C6 suite | 37/37 passed; 0 failures, 0 errors, 0 skips |
| Corrected uninterrupted/resume test | Passed, including tensor-aware autonomous outputs and timing-neutral metrics |
| Complete graph-encoder suite | 147/147 passed; 0 failures, 0 errors, 0 skips |
| Model-data regressions | 86/86 passed |
| Flat-baseline regressions | 549 run; 545 passed and 4 pre-existing tests skipped |
| Graph-baseline regressions | 17/17 passed |
| Representation regressions | 40/40 passed |
| Controlled-data regressions | 51/51 passed |
| Documentation validation | Passed for 69 Markdown files |
| Graph-encoder compilation | Passed |
| Python 3.8 grammar | Passed for all 20 enumerated C6 source/test files |
| Target-leakage/common-decoder audit | Passed |
| Flat train-only smoke | Passed at epoch 2; strict checkpoint reload; complete metrics schema |
| Typed-graph train-only smoke | Passed at epoch 2; strict checkpoint reload; complete metrics schema |
| Final exact-commit and clean-tree gate | Passed |
| Terminal success marker | Present |

Both smokes emitted `GE1-C6-METRICS-v1`, declared
`operation_template_train_payload_accessed=true`, and retained every protected
access flag as false. The final marker records both arms, two complete metrics
records, engineering-smoke status true, and scientific-training status false.

## Decision

C6 passes its authoritative Python 3.8/PyTorch 1.11 CPU validation gate at
exact commit `c88e967b96dd201fedcd495fa8d5ddaddd02caf3`. C6 is complete. This
decision supersedes only the pending status created by failed attempts
`3344337` and `3344363`; both failed-attempt records remain part of the audit
history. C7 and every later stage remain unstarted.

## Interpretation

The result establishes that the governed C6 machinery executes in the
authoritative runtime, resumes deterministically through the tested two-epoch
boundary, produces target-free autonomous outputs for all three memory
conditions, computes the complete metrics path, strictly reloads each smoke
checkpoint, and treats both encoder arms through the same training and
post-memory decoder contracts. It also validates guarded access to the full
authorized train assignment while preserving every protected-data boundary.

## Limitations and claims not supported

This is an infrastructure and train-only engineering validation, not a
scientific comparison. It does not establish decoder sufficiency, trained
memory sensitivity, an arm treatment effect, development performance, RR
systematic generalization, ER test performance, editing behavior, or any C7
or later-stage outcome. Only seed 2026 and two engineering epochs were used;
the frozen 50-epoch, multi-seed experiment was not run. The four permitted
flat-baseline regression skips were outside the two zero-skip C6 gates; the
submitted terse regression output reports their count but not their individual
names. The container path and runtime versions were recorded, but the
container image was not cryptographically hashed.

The two generated `c6_metrics.json` artifacts were verified by the runner and
summarized by its success events, but copies were not supplied for this local
audit. This record therefore validates successful complete-record production,
not any unreported metric values inside those records.

## Artifacts and integrity

The submitted terminal transcript is retained as the conversation attachment
`49a0c3bd-7a93-4fef-9030-65b6684b07f8/pasted-text.txt`:

| Size | SHA-256 |
|---:|---|
| 11,846 bytes | `1202f290716379d9e1cae90328f25292bd042e47884491e1cbc67a64fdcec640` |

The scheduler logs and generated run namespace remain at:

```text
/scratch/network/km6349/c6_validation_runs/c6-cpu-3344367.out
/scratch/network/km6349/c6_validation_runs/c6-cpu-3344367.err
/scratch/network/km6349/c6_validation_runs/c6-3344367
```

The per-arm metrics records are expected under the run namespace at
`flat/c6_metrics.json` and `typed_graph/c6_metrics.json`. Their independent
local copies, sizes, and hashes were not supplied and are not claimed by this
record.

## Related records

- [First C6 validation attempt](ge1_c6_cpu_validation_attempt_3344337.md)
- [Second C6 validation attempt](ge1_c6_cpu_validation_attempt_3344363.md)
- [C6 measurement contract](../specifications/ge1_c6_measurement_contract.md)
- [GE1 implementation plan](../specifications/graph_encoder_implementation_plan.md)
- [Current project status](../status.md)
- [Authoritative C5 validation](ge1_c5_cpu_validation.md)
