# GE1 C4 Authoritative CPU Validation

## Record status

| Item | Value |
|---|---|
| Experiment ID | `ge1-c4-cpu-adroit-h11n3-20260807` |
| Record status | `verified` |
| Scientific role | Infrastructure and compatibility validation |
| Branch | `graph-v1-experiment-record` |
| Source commit | `e66cd089462c2e075b9ff742e157c21e4d9a2a6e` |
| Working tree | Clean before and after validation |
| Slurm/local run ID | Interactive `srun`; job ID not captured |
| Execution date | August 7, 2026 |

## Question and predetermined decision rule

The validation asked whether the complete C4 standalone encoder
implementation works on an Adroit CPU at the exact C4 commit, including every
runtime-dependent test that was unavailable in the local no-PyTorch
environment.

The gate required:

- a clean detached checkout of exact commit
  `e66cd089462c2e075b9ff742e157c21e4d9a2a6e` before and after testing;
- all 26 targeted C4 tests to pass with zero skips;
- all 85 tests in the complete `prototype.graph_encoder` suite to pass;
- real-tensor execution on an Adroit CPU through the established Python 3.8
  container environment; and
- no training, decoder work, next-stage work, or protected corpus access.

Any failed test, skipped targeted test, commit mismatch, dirty checkout, or
missing terminal success marker rejected the gate.

## Inputs and partition authority

| Item | Value |
|---|---|
| Dataset/corpus | Procedural and temporary in-repository test fixtures only |
| Corpus/configuration hash | Not applicable |
| Split manifest | Synthetic temporary fixture manifests only |
| Train families used | None |
| Validation families used | None |
| Test families used | None |
| Test partition evaluated | `false` |

The exact tested C4 suite constructs E/R/EE/ER/RE/RR examples procedurally in
memory. The complete graph-encoder suite also exercises the C1 access boundary
with temporary synthetic fixtures. Neither command receives an authoritative
corpus path, partition name, or CAD-history payload path. Production import
guards at the tested commit keep unrestricted model-data loading confined to
the pre-existing audited C1 partition boundary; the C4 encoder modules import
no loader or manifest module.

The transcript does not emit explicit protected-access Boolean markers.
Zero authoritative manifest and payload access is therefore established from
the exact command and test-source audit, not from a runtime access counter.

## Configuration

The validation used the committed C4 defaults:

- flat feed-forward width 192;
- graph feed-forward width 64;
- three relational layers;
- two learned bases per layer;
- ten directed semantic message channels;
- two 32-dimensional latent queries;
- continuous 32-to-16-to-32 bottleneck projection; and
- frozen 5% trainable-parameter tolerance.

The tested assertions include 22,800 trainable flat-encoder parameters,
23,468 graph-encoder parameters, and a 2.9298245614% difference relative to
the flat control.

## Environment

| Item | Verified value |
|---|---|
| Cluster / host | Princeton Adroit / `adroit-h11n3` |
| Scheduler allocation | Interactive `srun`, one node, one task, one CPU, 8 GiB, 30 minutes |
| Python executable | `/root/miniconda3/bin/python3.8` |
| Exact Python patch version | Not printed in transcript |
| PyTorch | Importable and sufficient for all real-tensor tests; exact version not printed |
| CUDA visibility | Disabled with `CUDA_VISIBLE_DEVICES=""` |
| Execution device | CPU allocation and CPU test assertions passed |
| CPU threads | `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1` |
| Container | `/scratch/network/km6349/skexgen.sif` |
| Container content hash | Not preserved |

The container path and Python executable match the established C3 validation
environment, which directly reported Python 3.8.13 and PyTorch 1.11.0. Because
neither the C4 transcript nor either record contains a container content hash,
that version continuity is strong contextual evidence rather than an
independent C4 version measurement.

## Execution

The repository was transferred as a Git bundle, verified by SHA-256 on
Adroit, cloned to a new directory, and checked out in detached-HEAD state at
the exact C4 commit. The compute-node commands were:

```text
/root/miniconda3/bin/python3.8 \
  -m unittest prototype.graph_encoder.tests.test_encoders

/root/miniconda3/bin/python3.8 \
  -m unittest discover prototype/graph_encoder/tests
```

Both commands ran through Apptainer with a clean environment, CUDA hidden,
one OpenMP thread, one MKL thread, and the clean checkout bind-mounted at its
same absolute path.

## Verified results

- The targeted C4 suite passed 26/26 tests in 0.873 seconds with zero skips.
- The complete graph-encoder suite passed 85/85 tests in 2.371 seconds with
  zero skips.
- The targeted pass directly covers inherited flat continuous-path parity,
  no nearest-code assignment, unchanged VQ/EMA buffers, finite gradients,
  deterministic initialization, graph isolation, cross-graph rejection,
  direction sensitivity, edge-type sensitivity, zero-edge and isolated-node
  behavior, per-channel degree normalization, no active-channel
  renormalization, float32/float64 permutation properties, signature and
  position guards, parameter-object disjointness, exact parameter counts, the
  5% capacity gate, diameter coverage, and the CPU real-tensor smoke.
- Exact-commit and complete clean-worktree checks passed before and after both
  suites.
- The terminal marker `C4_ADROIT_VALIDATION_SUCCESS` was emitted.
- No test failure, skip, traceback, or runtime warning appears in either test
  result.

## Decision

The authoritative C4 CPU validation gate **passed**. C4 is accepted as a
validated standalone encoder implementation at exact commit
`e66cd089462c2e075b9ff742e157c21e4d9a2a6e`.

This decision completes C4 only. It does not begin C5, decoder integration,
training, evaluation, checkpointing, or protected-partition access.

## Interpretation

The result supports the engineering claim that the flat continuous wrapper
and position-free typed graph encoder satisfy their frozen interface,
capacity, isolation, sensitivity, gradient, and permutation contracts in the
authoritative CPU environment. It also confirms that the previously local-only
parameter-count and continuous-bypass assertions execute under real PyTorch.

It is not a scientific comparison between trained encoder arms. No decoder or
training behavior was evaluated.

## Limitations and claims not supported

- The interactive `srun` job ID, `sacct` state, exit code, elapsed allocation
  time, and scheduler stdout/stderr paths were not captured.
- The transcript did not print the Python patch version, PyTorch version, or
  CPU/device status directly. The executable name, CPU allocation, disabled
  CUDA visibility, passing CPU assertions, and established container provide
  supporting context but do not replace direct version output.
- The container content hash was not recorded.
- Protected-access booleans were not printed at runtime; the zero-access claim
  rests on the exact commands and inspected exact-commit test/import paths.
- The initial macOS `git bundle verify` command failed because it was executed
  outside a repository. This did not affect the transferred bytes or checkout:
  the remote checksum passed, the clone succeeded, and the retained bundle
  subsequently passed independent verification from the repository.
- The terminal transcript is locally preserved, but no separate immutable
  scheduler-log archive exists because validation ran interactively.
- C4 does not establish decoder compatibility, optimization, reconstruction,
  executable CAD validity, systematic generalization, or any trained-model
  result.

## Artifacts and integrity

| Artifact | Location | Size | SHA-256 | Availability |
|---|---|---:|---|---|
| Complete terminal transcript | `/Users/krishaymaskara/.codex/attachments/2b00e293-9022-4bbf-8a54-bf37f36b661e/pasted-text.txt` | 5,242 bytes | `2fdec7f801f63679de630780079452d76154592dd0d82deaa57701c5a3a5d574` | present |
| Complete-history Git bundle | `/Users/krishaymaskara/Downloads/c4-adroit-transfer/SkexGen-c4-e66cd0.bundle` | 77,868,831 bytes | `de265e1b122213eef8c24134861f878c530e7d2be6221e7b606868bf6863dd7e` | present |
| Bundle checksum file | `/Users/krishaymaskara/Downloads/c4-adroit-transfer/SkexGen-c4-e66cd0.bundle.sha256` | 91 bytes | `2b8a695cf6a72be1442fa995a200ae447795b2a46551ef4f986907be0cd36c8a` | present |

The retained bundle independently verifies as a complete-history bundle with
sole advertised `HEAD`
`e66cd089462c2e075b9ff742e157c21e4d9a2a6e`. Its recomputed local SHA-256
matches the checksum file and the remote `sha256sum -c` result in the
transcript.

## Reproduction and validation

The final audit read the complete transcript, recomputed all retained artifact
hashes, verified the Git bundle from inside the repository, confirmed its sole
advertised head, searched the transcript for failure/skip/traceback markers,
and reran repository documentation validation after recording the result.

Bundle verification is reproducible with:

```text
shasum -a 256 \
  /Users/krishaymaskara/Downloads/c4-adroit-transfer/SkexGen-c4-e66cd0.bundle

git bundle verify \
  /Users/krishaymaskara/Downloads/c4-adroit-transfer/SkexGen-c4-e66cd0.bundle
```

## Related records

- Prerequisite: [C3 authoritative CPU validation](ge1_c3_cpu_validation.md)
- Protocol: [GE1 Stage 0 preregistration](../specifications/ge1_stage0_preregistration.md)
- Implementation contract: [Typed graph encoder implementation plan](../specifications/graph_encoder_implementation_plan.md)
- Package contract: [Graph encoder README](../../prototype/graph_encoder/README.md)
- Current outcome: [Project status](../status.md)
