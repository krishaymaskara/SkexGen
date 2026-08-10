# GE1 Repaired-Sufficiency Authoritative CPU Validation

## Record status

| Item | Value |
|---|---|
| Experiment ID | `ge1-repaired-sufficiency-cpu-3345161` |
| Record status | `verified` |
| Scientific role | Corpus-free implementation and engineering validation only |
| Branch | `graph-v1-experiment-record` |
| Source commit | `6b62cab90ea8f693cde41c2abf0a261024057d45` |
| Governing decision | Accepted [ADR-0010](../decisions/ADR-0010-ge1-repaired-train-sufficiency-protocol.md) |
| Protocol | `GE1-C7-REPAIRED-SUFFICIENCY-v1` |
| Operation magnitude | `GE1-OPERATION-MAGNITUDE-POSITIVE-v1` |
| Slurm job | `3345161` |
| Compute host | `adroit-h11n2` |
| Working tree | Clean before and after validation; standalone detached checkout |
| Execution date | August 10, 2026 |

## Question and predetermined gate

The validation asked whether the prospective repaired train-sufficiency
implementation executes its corpus-free contract under the authoritative
Python 3.8/PyTorch 1.11 CPU environment without changing C7-v2, opening any
corpus, or beginning repaired scientific training.

The gate required:

- exact commit and clean standalone-checkout checks before and after testing;
- CPython 3.8.13, PyTorch 1.11.0, CPU-only execution, and one thread;
- all 32 focused tests, including all eight real-PyTorch tests, to pass with
  zero skips;
- the complete graph-encoder and repository regression suites to pass,
  permitting only the four established flat-baseline external-bundle skips;
- documentation, compilation, Python 3.8 grammar, import/export,
  target-leakage, protected-access, C7-v2 immutability, inherited-V6,
  Bash-syntax, whitespace, and clean-tree checks to pass; and
- no manifest, corpus payload, scientific training, scientific checkpoint,
  CAD kernel, Stage 6, or C8 activity.

The designated reviewer clarified that bounded backward passes and optimizer
steps over procedural fixtures are required engineering-validation activity.
They are not corpus training, engineering experimentation, or scientific
training.

## Evidence integrity and scheduler result

All supplied raw files were rehashed locally. The log checksum file covers
stdout, stderr, and `sacct` exactly; it does not cover itself. Its own digest
was independently recorded in the terminal transcript. No scientific artifact
manifest or artifact `SHA256SUMS` was produced.

| Evidence file | Bytes | SHA-256 | Checksum scope |
|---|---:|---|---|
| `repaired-validation-cpu-3345161.out` | 7,417 | `7d82c861f199fb17af469a12c04301b6388ef034f324802e7c58002b77ddf0ef` | Listed by log checksum file |
| `repaired-validation-cpu-3345161.err` | 7,472 | `32f626ee5194b73398c0aa959a0270ec7d6528835fd926090b03e6a64402524d` | Listed by log checksum file |
| `repaired-validation-3345161.sacct.txt` | 335 | `d4856e50b522ebe30668a0172d8d922c2f784b1092781610d683b94d12c82f96` | Listed by log checksum file |
| `repaired-validation-3345161.log-SHA256SUMS` | 308 | `846f2c7fc91ae1a78b2564fb2db59510a45482103c5db2ee82e8a74d2c9b70bf` | Terminal-transcript local hash only |

Scheduler and runner evidence agree on:

| Item | Verified value |
|---|---|
| Job / host | `3345161` / `adroit-h11n2` |
| Scheduler state / exit | `COMPLETED` / `0:0` |
| Elapsed / batch MaxRSS | `00:06:08` / `615848K` |
| Container | `/scratch/network/km6349/skexgen.sif` |
| Python | CPython 3.8.13 |
| PyTorch | 1.11.0, CUDA build 11.3 |
| CUDA available | `false` |
| Device / CPU threads | CPU / 1 |

The runner start record names the exact commit, standalone clean checkout,
job, host, container, protocol, repair identity, and negative access states.
The final gate rechecks the exact commit and clean tree before emitting the
terminal-success record.

## Test and repository results

| Gate | Verified result |
|---|---|
| Focused repaired-protocol suite | 32/32 passed; 0 failures, 0 errors, 0 skips |
| Complete graph-encoder suite | 324/324 passed; 0 failures, 0 errors, 0 skips |
| Model-data regressions | 86/86 passed; 0 skips |
| Flat-baseline regressions | 549 run; 545 passed; exactly 4 established skips |
| Graph-baseline regressions | 17/17 passed; 0 skips |
| Representation regressions | 40/40 passed; 0 skips |
| Controlled-data regressions | 51/51 passed; 0 skips |
| Documentation validation | Passed for 86 Markdown files |
| Graph-encoder compileall | Passed |
| Python 3.8 grammar | Passed for 45 enumerated source/test files |
| Import/access/target/history audit | Passed; 103 exports |
| Frozen C7-v2 source/tests | Three pinned SHA-256 values matched |
| Frozen inherited V6 audit | No `prototype/flat_baseline` change from the accepted parent |
| Scientific and validation-runner Bash syntax | Passed |
| Whitespace, exact-commit, final clean tree | Passed |

The regression runner constructed a fresh `unittest.TestLoader` with explicit
repository top level for each suite. Its exact-method guard confirms that all
four skips were the established flat-baseline external-bundle cases; every
other suite had zero skips. The `/external/c7-v2` stdout record and the C7-v2
failure record in the complete-suite stream are mocked CLI output from unit
tests, not real artifacts or data access.

## Focused protocol coverage

The 24 pure tests passed the frozen arithmetic, operation-fidelity thresholds,
unrounded boundary comparisons, failure handling, conservative aggregation,
tiny-to-scaled dependencies, memory-only decision, repaired-checkpoint
eligibility, target isolation, C7-v2 immutability, and separately versioned
artifact-verification contracts.

All eight real-PyTorch tests executed rather than skipping. They covered:

- synthetic training-path traversal by both repaired arms;
- matched shared-decoder initialization;
- strict repaired inference and recovery reload;
- rejected legacy/repaired checkpoint interchange;
- autonomous output entering fidelity computation only after generation;
- rejection of a target tensor by the autonomous interface;
- finite operation-magnitude gradients; and
- positive, finite, bounded repaired transformation over extreme logits.

The synthetic traversal performed bounded procedural-fixture backward passes
and optimizer steps. The truthful execution distinctions are:

```text
synthetic_unit_test_backward_passes_executed=true
synthetic_unit_test_optimizer_steps_executed=true
scientific_training_performed=false
corpus_training_performed=false
```

The test did not execute the frozen 200-epoch tiny or scaled scientific
arithmetic. It did not select or publish a scientific checkpoint.

## Protocol and checkpoint identity boundary

The new protocol remains prospective and separately named
`GE1-C7-REPAIRED-SUFFICIENCY-v1`. The implementation preserves the three
pinned C7-v2 source/test hashes, so it does not modify or reinterpret job
`3344981`, its decisions, schemas, or checkpoint identities.

The scientific configuration freezes fresh matched seed-2026 pairs, fixed
epoch-200 selection, strict recovery and inference reload, and no warm start,
checkpoint reuse, best-loss selection, early stopping, or outcome-dependent
extension. Its arithmetic is 200 steps and 800 presentations per tiny arm and
800 steps and 6,400 presentations per scaled arm.

The model-semantic compatibility identity
`GE1-OPERATION-MAGNITUDE-POSITIVE-v1` is embedded in and enforced by generic
inference and recovery checkpoint metadata and provenance. The broader
experimental identity `GE1-C7-REPAIRED-SUFFICIENCY-v1` belongs to the
surrounding resolved configuration, training/run configuration, metrics,
exact/fidelity/memory gates, artifact manifest, and terminal record. It is not
duplicated as a literal field in the generic checkpoint schema. Checkpoint and
configuration hashes, source commit, arm, seed, epoch, checkpoint role, and
the enclosing scientific artifact unambiguously bind each future checkpoint
to the repaired protocol.

## Fidelity, dependency, and target-isolation coverage

The verified implementation applies the hard gate only to autonomous
`P_true` operation magnitudes from strict epoch-200 reloads. Extrusion channel
37 uses the unchanged 4.0 scale and requires unrounded absolute physical error
`< 0.25`; revolve channel 38 uses the unchanged 360.0 scale and requires
`< 22.5` degrees. Equality fails. Missing, nonfinite, masked, wrong-channel,
wrong-operation, undefined, or positive-but-inaccurate values fail.

Every operation and loader-verified representation variant is scored before
aggregation. Every record must pass, the family summary uses the maximum
absolute error, and no averaging can hide an outlier. Existing non-operation
geometry errors remain report-only; the gate is not described as exact
continuous geometry or CAD-kernel validity.

Both tiny arms must pass exact node sequence, graph, strict conversion,
analytic complete validity, applicable `depends_on`, and operation fidelity
before scaled access. A tiny failure makes every scaled gate and payload
access `not_run`. Overall success requires both scaled arms to pass exactness,
fidelity, and unchanged memory use. Exact/fidelity failure is scientific
failure; memory-only failure is inconclusive; neither authorizes Stage 6 or a
new repair.

Static and runtime tests establish that target geometry enters only after
autonomous generation, inside metrics and gates. No target tensor is accepted
by either model or autonomous interface or used for memory, decoder input,
optimization control, or checkpoint selection.

## Access, artifacts, and authorization

The runner bound only the standalone repository read-only. It received no
manifest or corpus argument. Start and terminal records both declare no
access to:

- the operation-template manifest or train payload;
- tiny or scaled train payloads;
- development, RR, ER, IID, history-depth, or geometry-extrapolation;
- any other corpus or CAD-history payload.

No corpus or scientific training occurred. No scientific checkpoint,
resolved scientific configuration, metrics artifact, artifact manifest, or
scientific `SHA256SUMS` was produced. Temporary checkpoint files exercised by
unit tests were ephemeral procedural fixtures. No CAD kernel was used, no
repaired scientific result exists, and Stage 6 and C8 remain unauthorized and
unperformed.

## Decision

The authoritative implementation-validation gate **passed** for exact commit
`6b62cab90ea8f693cde41c2abf0a261024057d45` as Adroit job `3345161`.

This is implementation and engineering validation only. It establishes that
the prospective repaired protocol is eligible for a separate scientific
execution authorization. It is not a repaired scientific result, a C7-v2
rerun, evidence that geometry was learned, evidence of encoder superiority,
CAD-executor validation, or Stage 6 authorization. Scientific training and
submission remain unauthorized until separately directed.

## Limitations

- No manifest, corpus family, learned checkpoint, autonomous trained-model
  output, or CAD solid was evaluated.
- The protocol and artifact behavior were tested synthetically; no scientific
  artifact was finalized by this job.
- The container path and runtime were recorded, but the container image was
  not cryptographically hashed.

## Related records

- [ADR-0010](../decisions/ADR-0010-ge1-repaired-train-sufficiency-protocol.md)
- [Repaired sufficiency execution contract](../specifications/ge1_repaired_sufficiency_execution_contract.md)
- [Operation-magnitude repair validation](ge1_operation_magnitude_repair_cpu_validation.md)
- [GE1 implementation plan](../specifications/graph_encoder_implementation_plan.md)
- [Current project status](../status.md)
- [Graph encoder README](../../prototype/graph_encoder/README.md)
