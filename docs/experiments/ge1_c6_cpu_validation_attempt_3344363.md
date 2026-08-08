# GE1 C6 Authoritative CPU Validation Attempt 3344363

## Record status

| Item | Value |
|---|---|
| Experiment ID | `ge1-c6-cpu-3344363` |
| Record status | `verified failed validation attempt` |
| Scientific role | Infrastructure, compatibility, and train-only smoke gate for C6 |
| Branch | `graph-v1-experiment-record` |
| Source commit | `68c66ea4b4c1e1d01b9b9dc061ace1149bca7c5a` |
| Source parent | `ca6dd069ae700cff993c354c2116de11924930f8` |
| Slurm job | `3344363` |
| Compute host | `adroit-h11n2` |
| Execution date | August 7, 2026 |
| Decision | Failed during the final focused test; C6 remains unvalidated |

## Question and predetermined decision rule

The attempt asked whether corrected exact C6 commit
`68c66ea4b4c1e1d01b9b9dc061ace1149bca7c5a` passed the frozen
implementation, compatibility, regression, and two-arm train-only smoke gate
under the authoritative CPU environment. The predetermined rule required every
ordered gate and the terminal success marker. Any test failure, error, skip,
identity mismatch, protected access, or incomplete metrics record rejected the
attempt.

## Gate and observed environment

The runner required a clean exact checkout, Python 3.8, PyTorch 1.11, CPU
execution, all 37 focused C6 tests and the complete graph-encoder suite with
zero skips, every listed regression and repository check, then two complete
407-family `operation_template.train` smokes. Any failure before the terminal
success marker rejected the run.

| Item | Observed value |
|---|---|
| Exact checkout | `68c66ea4b4c1e1d01b9b9dc061ace1149bca7c5a` |
| Source clean at start | `true` |
| Python | CPython 3.8.13 |
| PyTorch | 1.11.0, CUDA build 11.3 |
| CUDA available | `false` |
| Device | CPU |
| CPU threads | 1 |
| Container | `/scratch/network/km6349/skexgen.sif` |
| Scheduler state | `FAILED`, exit `1:0` |
| Elapsed / peak RSS | 17 seconds / 479,192 KiB |

The exact-commit and clean-tree checks passed before testing. Fail-fast
execution stopped after the focused suite, before the later clean-tree check.

## Focused-suite result

All 37 focused tests ran with zero skips and zero assertion failures. Thirty-
six passed and one ended in an error:

```text
test_uninterrupted_equals_epoch_one_resume
RuntimeError: Boolean value of Tensor with more than one value is ambiguous
```

The deterministic model-state, optimizer-state, counter, data-order, and loss
assertions preceding the error all passed. The test then used
generic `unittest` tuple/dataclass equality on autonomous prediction records
that contain multi-element PyTorch tensors. Dataclass equality delegated to
tensor `==`, whose result cannot be converted to one Boolean. This is a test
comparison defect, not evidence of unequal resumed predictions or a production
model failure.

The narrow correction recursively compares tensors with `torch.equal`,
recursively compares their containing dataclasses, normalizes only recorded
wall-clock durations, and compares the complete autonomous metric records with
only `metric_computation_seconds` removed. It changes no model, optimizer,
decoder, autonomous path, training rule, metric definition, data access, or
frozen scientific choice.

Job `3344363` also confirms that both corrections prompted by job `3344337`
worked: all three autonomous node-count boundary tests and the provenance
failure-publication test passed.

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

The generic terminal-failure marker conservatively emitted
`not_confirmed_on_failure` for the manifest and train-payload fields. Ordered
stdout/stderr establishes the stronger result above: the runner stopped at the
focused suite, and every focused runtime fixture is procedural.

## Gates not reached

The run did not reach:

- the complete graph-encoder suite;
- model-data, flat-baseline, graph-baseline, representation, or controlled-data
  regressions;
- documentation, compilation, Python 3.8 grammar, target-leakage, final Git,
  or source-integrity gates;
- either 407-family train-only smoke;
- 407-family smoke checkpoint publication or complete C6 metrics records.

Job `3344363` cannot be promoted to a C6 pass. The corrected exact commit must
repeat the complete runner from the beginning with zero focused or complete-
suite skips.

## Decision

Job `3344363` failed the predetermined C6 gate. It is retained as a failed
infrastructure-validation attempt and supplies no scientific result. C6
remains authoritatively unvalidated pending a complete corrected rerun.

## Interpretation

The evidence localizes the only remaining observed failure to a generic test
comparison that is invalid for tensor-bearing dataclasses. It supports a
test-only tensor-aware correction and a stronger timing-neutral autonomous
metrics equality assertion. It does not support changing production code or
weakening the recovery contract.

## Limitations and claims not supported

The run stopped after 17 seconds and never reached the complete suite,
regressions, corpus smoke, checkpoints, or metrics artifacts. It does not
validate the full C6 gate, either 407-family smoke, comparative performance,
development performance, systematic generalization, held-out testing, C7, or
any later stage.

## Artifacts and integrity

The submitted terminal transcript is retained as the conversation attachment
`614103df-6208-4e85-a6a0-dffe8255bc27/pasted-text.txt`:

| Size | SHA-256 |
|---:|---|
| 11,851 bytes | `4d4f80a1bd7681c24d07bded35f63db6915c947cc8cbd5738783f71754b59b50` |

The scheduler logs remain at:

```text
/scratch/network/km6349/c6_validation_runs/c6-cpu-3344363.out
/scratch/network/km6349/c6_validation_runs/c6-cpu-3344363.err
```

Their independent local file hashes were not supplied and are not claimed by
this record.

## Related records

- [First C6 validation attempt](ge1_c6_cpu_validation_attempt_3344337.md)
- [C6 measurement contract](../specifications/ge1_c6_measurement_contract.md)
- [GE1 implementation plan](../specifications/graph_encoder_implementation_plan.md)
- [Current project status](../status.md)
- [Authoritative C5 validation](ge1_c5_cpu_validation.md)
