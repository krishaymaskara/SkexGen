# GE1 C7-v2 Preflight Attempt 3344975

## Record status

| Item | Value |
|---|---|
| Experiment ID | `ge1-c7-v2-preflight-3344975` |
| Record status | `reported-only` |
| Scientific role | Authoritative-environment preflight attempt; no C7-v2 scientific execution |
| Outcome | Regression-runner implementation failure before manifest access |
| Source commit | `99a4587d73e82a5df5f12a54130b105ae264b419` |
| Slurm job | `3344975` |
| Host | `adroit-h11n2` |
| Scheduler state | `FAILED`, exit `1:0`, elapsed `00:00:37`, MaxRSS `488800K` |
| Record date | August 10, 2026 |

## Question and preflight decision rule

The C7-v2 runner requires every environment, test, regression, documentation,
source, and artifact preflight to pass before opening the authoritative
operation-template manifest. Job `3344975` stopped inside the sequential
regression block. It did not call `prototype.graph_encoder.c7_v2`, select a
cohort, open a manifest, load a CAD-history payload, train a model, evaluate a
checkpoint, or create a scientific artifact.

## Results and findings

The supplied terminal transcript reports:

- exact commit, clean source, and standalone-checkout gates passed;
- CPython `3.8.13`, PyTorch `1.11.0`, and CPU-only execution passed;
- focused C7-v2 suite: 28/28, zero failures, errors, or skips;
- complete graph-encoder suite: 266/266, zero failures, errors, or skips;
- model-data regression: 86/86, zero failures, errors, or skips;
- the next regression discovery failed before `flat_baseline` ran.

The JSON terminal-failure marker records all of the following as `false`:

```text
operation_template_manifest_accessed
operation_template_train_payload_accessed
tiny_train_payload_accessed
scaled_train_payload_accessed
development_accessed
systematic_rr_accessed
test_er_accessed
iid_accessed
history_depth_accessed
geometry_extrapolation_accessed
c7_v2_execution_completed
stage6_performed
c8_or_later_performed
decoder_repair_implemented_or_invoked
```

## Failure diagnosis

The regression block repeatedly invoked
`unittest.defaultTestLoader.discover(path)`. Under Python 3.8, this singleton
loader retained the first discovery's `_top_level_dir`. The next discovery
therefore raised:

```text
AssertionError: Path must be within the project
```

This is a test-runner implementation defect. It is not a model, training,
metric, target-alignment, data-access, checkpoint, or scientific gate result.
The narrow correction constructs a fresh `unittest.TestLoader` inside each
suite iteration and calls `discover(path, top_level_dir=".")`. Expected skip
rules remain zero for model-data, graph-baseline, representation, and
controlled-data, and exactly four established skips for flat-baseline on
Adroit.

## Mock-output clarification

The `/external/c7-v2` lines in stdout came from the pure CLI exit-behavior unit
test, which mocks `run_c7_v2`. They are test output, not filesystem evidence
and not scientific artifacts. No finalized or partial C7-v2 artifact exists
from job `3344975`.

## Decision and interpretation

Job `3344975` is recorded only as a preflight/test-runner implementation
failure with zero manifest and payload access. It does not pass or fail C7-v2,
does not change C7-v1 or the optimization diagnostic, and authorizes no Stage
6, C8, or decoder-repair work. A replacement job must start from the corrected
exact commit and rerun every preflight from the beginning. No output from job
`3344975` is reusable as scientific evidence.

## Limitations

The supplied terminal transcript includes scheduler status and the complete
displayed stdout/stderr. Separate downloaded raw log files and their hashes
were not supplied, so this record is `reported-only` rather than
`verified-local`.

## Artifacts and integrity

No scientific artifact, partial scientific artifact, checkpoint, resolved
configuration, metrics file, artifact manifest, or `SHA256SUMS` publication
was produced. The transcript was inspected, but separate stdout/stderr files
and their cryptographic hashes were not available for this record.

## Related records

- [ADR-0008](../decisions/ADR-0008-ge1-c7-v2-200-epoch-protocol.md)
- [C7-v2 execution contract](../specifications/ge1_c7_v2_execution_contract.md)
- [C7-v1 scientific result](ge1_c7_pilot.md)
- [Optimization diagnostic](ge1_optimization_diagnostic.md)
- [Current project status](../status.md)
