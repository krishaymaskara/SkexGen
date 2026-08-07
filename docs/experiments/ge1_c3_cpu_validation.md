# GE1 C3 Authoritative CPU Validation

## Record status

| Item | Value |
|---|---|
| Experiment ID | `ge1-c3-cpu-3344235` |
| Record status | `verified` |
| Scientific role | Infrastructure and compatibility validation |
| Branch | `graph-v1-experiment-record` |
| Source commit | `a85ad3a23a6587cbedad8a6693b6117c1edfacc7` |
| Working tree | Clean at validation start and end |
| Slurm/local run ID | `3344235` |
| Execution date | August 7, 2026 |

## Question and predetermined decision rule

The job asked whether the complete C3 target-separated paired-batching
implementation at the exact C3 commit works in the authoritative Adroit
Python 3.8 and PyTorch 1.11 CPU environment.

The gate required all of the following:

- an exact, clean checkout of the C3 commit before and after validation;
- CPython 3.8.13 and PyTorch 1.11.0 with CPU execution;
- every discovered `prototype.graph_encoder` test to pass;
- the real-PyTorch C3 batching smoke to run exactly once, pass, and report
  zero skips;
- the selected loader-free model-data regression suite to pass;
- `prototype/graph_encoder` to compile successfully; and
- a terminal success marker with zero corpus-manifest access, CAD-history
  payload access, systematic/test access, and training.

Any command failure, environment mismatch, dirty source tree, skipped
real-PyTorch smoke, or missing terminal success marker rejected the gate.

## Inputs and partition authority

| Item | Value |
|---|---|
| Dataset/corpus | Deterministic in-repository unit-test fixtures only |
| Corpus/configuration hash | Not applicable |
| Split manifest | Synthetic manifest fixtures only; authoritative manifest not opened |
| Train families used | None |
| Validation families used | None |
| Test families used | None |
| Test partition evaluated | `false` |

The stdout start and terminal records both declare
`authoritative_corpus_manifest_accessed=false`,
`authoritative_cad_history_payload_accessed=false`,
`systematic_partition_accessed=false`, `test_partition_accessed=false`, and
`training_performed=false`. The validation exercised contracts over fixtures;
it did not load any controlled-corpus payload or perform a scientific split
evaluation.

## Configuration

The preserved validation runner is
[`prototype/graph_encoder/adroit/c3_cpu_validation_a85ad3a.slurm`](../../prototype/graph_encoder/adroit/c3_cpu_validation_a85ad3a.slurm),
with SHA-256
`1447b7b5d189089c6c67355f32b20258b6b988c680f526ae3d2876b08280a9d8`.
It requested one node, one task, one CPU, 8 GiB of memory, and 30 minutes. It
fixed one OpenMP and one MKL thread, disabled visible CUDA devices, redirected
Python bytecode outside the checkout, and failed on any unsuccessful command.

The exact required source commit was
`a85ad3a23a6587cbedad8a6693b6117c1edfacc7`. The runner checked both the
commit and the complete Git porcelain status before validation and repeated
both checks immediately before emitting terminal success.

## Environment

| Item | Verified value |
|---|---|
| Cluster / host | Princeton Adroit / `adroit-h11n2` |
| Python | CPython 3.8.13 |
| PyTorch | 1.11.0 |
| PyTorch CUDA build | 11.3 |
| CUDA available | `false` |
| Execution device | CPU |
| PyTorch CPU threads | 1 |
| Container | `/scratch/network/km6349/skexgen.sif` |
| Container content hash | Not preserved |

The presence of CUDA 11.3 in the PyTorch build identifies the installed wheel;
the environment record independently confirms that CUDA was unavailable and
the validation device was CPU.

## Execution

The runner executed these substantive checks inside the container:

```text
python3.8 -B -m unittest discover prototype/graph_encoder/tests
python3.8 -B <targeted real-PyTorch smoke harness>
python3.8 -B -m unittest \
  prototype.model_data.tests.test_compatibility \
  prototype.model_data.tests.test_constrained_profile_decoder \
  prototype.model_data.tests.test_profile_geometry \
  prototype.model_data.tests.test_profile_geometry_torch
python3.8 -B -m compileall prototype/graph_encoder
```

Slurm wrote stdout and stderr under
`/scratch/network/km6349/c3_validation_runs/`. The locally inspected copies
are the two files listed under artifacts below.

## Verified results

- All 59 discovered graph-encoder tests passed in 1.486 seconds.
- The targeted real-PyTorch smoke ran once and passed with zero failures,
  errors, or skips.
- All 41 selected model-data regression tests passed in 0.175 seconds.
- `compileall` completed and emitted `graph_encoder_compileall=PASS`.
- Stdout identifies the exact C3 commit and records `source_clean=true`.
- The final event is `c3_cpu_validation_terminal_success` for job `3344235`
  and the exact C3 commit.
- No terminal-failure marker, traceback, unittest failure, or warning appears
  in either supplied log.
- Both the start and terminal records declare zero authoritative-manifest,
  CAD-history-payload, systematic-partition, test-partition, and training
  access.

## Decision

The authoritative C3 CPU compatibility gate **passed**. C3 is accepted as
working under Python 3.8.13 and PyTorch 1.11.0 on Adroit CPU at exact commit
`a85ad3a23a6587cbedad8a6693b6117c1edfacc7`.

This decision completes C3 validation only. It does not begin or validate C4.

## Interpretation

The result supports the narrow engineering claim that C3's paired flat/graph
batching, target separation, bookkeeping validation, permutation diagnostic,
and real-tensor boundary execute correctly in the frozen production Python
and PyTorch environment. The selected model-data regressions also show no
detected compatibility regression in the exercised shared geometry and
profile paths.

## Limitations and claims not supported

- The supplied artifacts do not include `sacct` output, so scheduler state,
  exit code, elapsed time, and allocation metadata were not independently
  verified. Completion is supported by the user report and the terminal
  success record in stdout.
- The logs do not print the submitted runner's SHA-256. Their ordered markers,
  fixed literals, and observed commands are consistent with the preserved
  runner, but byte identity between that file and the submitted script cannot
  be proved from the logs alone.
- The two logs are present locally but have not been placed in a frozen archive
  outside the Downloads directory.
- The container content hash was not recorded.
- This fixture-only run does not verify the authoritative physical manifest,
  load a train or development payload, measure training behavior, establish
  model correctness, or support any reconstruction or generalization claim.
- No relational encoder, shared model, training loop, checkpoint, or C4
  capability was exercised.

## Artifacts and integrity

| Artifact | Location | Size | SHA-256 | Availability |
|---|---|---:|---|---|
| Slurm stdout | `/Users/krishaymaskara/Downloads/c3-validation-3344235/c3-cpu-3344235.out` | 2,277 bytes | `1b176614816202db006de9ea9a1a1704acedef9697dd37c1d7485d8f9c3a745d` | present |
| Slurm stderr | `/Users/krishaymaskara/Downloads/c3-validation-3344235/c3-cpu-3344235.err` | 524 bytes | `0c812e273cf3cb13d72d0c38d788cb0111c14e64ead20c70ff41129d809b48fc` | present |
| Preserved Slurm runner | `prototype/graph_encoder/adroit/c3_cpu_validation_a85ad3a.slurm` | 5,361 bytes | `1447b7b5d189089c6c67355f32b20258b6b988c680f526ae3d2876b08280a9d8` | present |

The stdout and stderr hashes above were recomputed locally during the final
audit. Raw scheduler logs are intentionally not committed; this record keeps
their locations, sizes, and integrity hashes.

## Reproduction and validation

The final audit inspected both complete logs, recomputed all three artifact
hashes, checked the logs for failure/skip/traceback markers, and compared the
observed event sequence with the preserved runner. Repository documentation
validation was rerun after recording the result.

The integrity check is:

```text
shasum -a 256 \
  /Users/krishaymaskara/Downloads/c3-validation-3344235/c3-cpu-3344235.out \
  /Users/krishaymaskara/Downloads/c3-validation-3344235/c3-cpu-3344235.err \
  prototype/graph_encoder/adroit/c3_cpu_validation_a85ad3a.slurm
```

## Related records

- Prerequisite protocol: [GE1 Stage 0 preregistration](../specifications/ge1_stage0_preregistration.md)
- Implementation contract: [Typed graph encoder implementation plan](../specifications/graph_encoder_implementation_plan.md)
- Package contract: [Graph encoder README](../../prototype/graph_encoder/README.md)
- Current outcome: [Project status](../status.md)
