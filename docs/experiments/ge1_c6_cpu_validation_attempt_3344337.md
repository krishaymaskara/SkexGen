# GE1 C6 Authoritative CPU Validation Attempt 3344337

## Record status

| Item | Value |
|---|---|
| Experiment ID | `ge1-c6-cpu-3344337` |
| Record status | `verified failed validation attempt` |
| Scientific role | Infrastructure, compatibility, and train-only smoke gate for C6 |
| Branch | `graph-v1-experiment-record` |
| Source commit | `ca6dd069ae700cff993c354c2116de11924930f8` |
| Source parent | `1e482a793908537ef90ba0f977df394b5e1adfee` |
| Slurm job | `3344337` |
| Compute host | `adroit-h11n2` |
| Execution date | August 7, 2026 |
| Decision | Failed during the focused suite; C6 remains unvalidated |

## Question and predetermined decision rule

The attempt asked whether exact C6 commit `ca6dd069ae700cff993c354c2116de11924930f8`
passed the frozen implementation, compatibility, regression, and two-arm
train-only smoke gate under the authoritative CPU environment. The
predetermined rule required every ordered gate and the terminal success marker;
any test failure, error, skip, identity mismatch, protected access, or missing
complete metrics record rejected the attempt.

## Gate and observed environment

The runner required a clean exact checkout, Python 3.8, PyTorch 1.11, CPU
execution, all 37 focused C6 tests and the complete graph-encoder suite with
zero skips, every listed regression and repository check, then two complete
407-family `operation_template.train` smoke records—one per arm. Any failure
before the terminal success marker rejected the run.

| Item | Observed value |
|---|---|
| Exact checkout | `ca6dd069ae700cff993c354c2116de11924930f8` |
| Source clean at start | `true` |
| Python | CPython 3.8.13 |
| PyTorch | 1.11.0, CUDA build 11.3 |
| CUDA available | `false` |
| Device | CPU |
| CPU threads | 1 |
| Container | `/scratch/network/km6349/skexgen.sif` |
| Scheduler state | `FAILED`, exit `1:0` |
| Elapsed / peak RSS | 15 seconds / 481,544 KiB |

The exact-commit and clean-tree checks passed before testing. Fail-fast
execution stopped after the focused suite, before the later clean-tree check.

## Focused-suite result

All 37 focused tests ran with zero skips and zero assertion failures. Thirty-
three passed and four ended in errors:

1. `test_complete_metric_path_all_conditions`,
   `test_interventions_change_supplied_memory_and_keep_identity`, and
   `test_uninterrupted_equals_epoch_one_resume` all reached the same production
   defect. `autonomous._decode_condition` supplied `node_counts` as a Python
   tuple, while the frozen C5 decoder contract requires a device-local
   `torch.long` tensor with shape `[B]`. The decoder correctly rejected the
   malformed boundary value before node rollout.
2. `test_provenance_failure_prevents_checkpoint_publication` raised
   `NameError` because its test-local failure callback referenced
   `GraphEncoderError` without importing it. This is a test defect; the
   production provenance path was not reached by that assertion.

The node-count defect is a C6 autonomous-adapter implementation error, not a
change to the C5 decoder contract. The correction constructs a one-row local
long tensor on `memory.device`. It does not change node counts, decoder
semantics, targets, interventions, or any frozen scientific choice. The second
correction adds only the missing test import.

## Access and scope audit

The runner failed in the focused procedural/in-memory suite before invoking
`c6_smoke.py`. Therefore:

```text
authoritative operation_template manifest accessed: false
authoritative operation_template train payload accessed: false
development accessed: false
RR systematic accessed: false
ER test accessed: false
IID accessed: false
history-depth accessed: false
geometry-extrapolation accessed: false
scientific training performed: false
two-epoch engineering smoke performed: false
C7 or later work performed: false
```

The runner's generic terminal-failure marker conservatively emitted
`not_confirmed_on_failure` for the manifest and train-payload fields. The
ordered stdout/stderr establishes the stronger result above: the process
stopped at the focused suite, and every focused runtime fixture is procedural.

## Gates not reached

The run did not reach:

- the complete graph-encoder suite;
- model-data, flat-baseline, graph-baseline, representation, or controlled-data
  regressions;
- documentation, compilation, Python 3.8 grammar, target-leakage, final Git,
  or source-integrity gates;
- either 407-family train-only smoke;
- epoch checkpoint publication or complete C6 metrics records.

Job `3344337` cannot be promoted to a C6 pass. The corrected exact commit must
repeat the complete runner from the beginning with zero focused or complete-
suite skips.

## Decision

Job `3344337` failed the predetermined C6 gate. It is retained as a failed
infrastructure-validation attempt and supplies no scientific result. C6
remains authoritatively unvalidated pending a complete corrected rerun.

## Interpretation

The evidence localizes the failure to one malformed C6 adapter value and one
missing test import. It supports the narrow corrections recorded above. It
does not support weakening the decoder contract, bypassing the test, or
changing any training, evaluation, intervention, or data-access decision.

## Limitations and claims not supported

The run stopped after 15 seconds and never reached the complete suite,
regressions, corpus smoke, checkpoints, or metrics artifacts. It does not
validate autonomous metrics, deterministic recovery, either two-epoch arm,
comparative performance, development performance, systematic generalization,
held-out testing, C7, or any later stage.

## Artifacts and integrity

The submitted terminal transcript is retained as the conversation attachment
`ce9aa631-3eb4-4ab2-953b-efd91572ba78/pasted-text.txt`:

| Size | SHA-256 |
|---:|---|
| 28,317 bytes | `fd4cc75bd61c2fbb061b5e199c0c37e19e1c4fb2a38d69b4a305673fe5598de7` |

The scheduler logs remain at:

```text
/scratch/network/km6349/c6_validation_runs/c6-cpu-3344337.out
/scratch/network/km6349/c6_validation_runs/c6-cpu-3344337.err
```

Their independent local file hashes were not supplied and are not claimed by
this record.

## Related records

- [C6 measurement contract](../specifications/ge1_c6_measurement_contract.md)
- [GE1 implementation plan](../specifications/graph_encoder_implementation_plan.md)
- [Current project status](../status.md)
- [Authoritative C5 validation](ge1_c5_cpu_validation.md)
