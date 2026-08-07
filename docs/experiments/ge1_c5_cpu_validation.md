# GE1 C5 Authoritative CPU Validation

## Record status

| Item | Value |
|---|---|
| Experiment ID | `ge1-c5-cpu-3344290` |
| Record status | `verified` |
| Scientific role | Infrastructure, parity, and compatibility validation of C5 |
| Branch | `graph-v1-experiment-record` |
| Source commit | `996016df44b7f9a6cd5c092a3e3b7a87d9964f9d` |
| Source parent | `ffa6091cfecb37aa648ed3590f591f247a776754` |
| Slurm job | `3344290` |
| Compute host | `adroit-h11n2` |
| Working tree | Clean before and after validation |
| Execution date | August 7, 2026 |

## Question and predetermined decision rule

The validation asked whether the corrected exact C5 commit satisfies the
complete shared-decoder acceptance gate under Python 3.8 and PyTorch 1.11 on
CPU. The gate required:

- a clean checkout of exact commit
  `996016df44b7f9a6cd5c092a3e3b7a87d9964f9d` before and after testing;
- Python 3.8 and PyTorch 1.11 with CPU-only execution;
- all 22 focused C5 tests to run and pass with zero skips;
- all 110 tests in the complete graph-encoder suite to run and pass with zero
  skips;
- successful model-data, flat-baseline, graph-baseline, representation, and
  controlled-data regression suites;
- successful documentation validation, graph-encoder compilation, and Python
  3.8 grammar parsing; and
- no corpus manifest, CAD-history payload, RR systematic, ER test, scientific
  training, C7, or C8 access or execution.

Any failure, error, required-suite skip, count mismatch, environment mismatch,
commit mismatch, dirty tree, or missing terminal-success event would fail the
gate.

## Inputs and partition authority

| Item | Value |
|---|---|
| Dataset/corpus | Procedural model fixtures and synthetic manifest fixtures only |
| Authoritative manifest opened | `false` |
| Authoritative CAD-history payload opened | `false` |
| Systematic RR partition opened | `false` |
| Held-out ER test partition opened | `false` |
| Scientific training performed | `false` |
| C7 performed | `false` |
| C8 performed | `false` |

The focused C5 tests use constructed model fixtures. Existing regression tests
may execute optimizer smoke steps, but the runner records that no scientific
training occurred. Nothing in this validation authorizes later-stage data
access.

## Configuration under review

The validated commit contains the C5 shared decoder, common-memory model
wrapper, common per-example loss, autonomous raw/constrained/converted result
contract, frozen four-signal decoder output-position inventory, and strict
`GE1-CHECKPOINT-v1` API. It also contains the three test-only corrections
identified by failed job `3344278`:

- normalization of retained legacy VQ provenance before complete constrained
  graph-record comparison;
- use of the authoritative `primary_failure` and `secondary_failures`
  conversion-result fields; and
- discovery of padded rows and positions from `node_mask` after canonical
  batch sorting.

No production decoder, encoder, model, loss, checkpoint, canonicalization, or
data-loading implementation changed in that correction.

## Environment

| Item | Verified value |
|---|---|
| Cluster / host | Princeton Adroit / `adroit-h11n2` |
| Slurm job | `3344290` |
| Scheduler state / exit | `COMPLETED` / `0:0` |
| Elapsed / peak RSS | 5 minutes 55 seconds / 607,780 KiB |
| Python | CPython 3.8.13 |
| PyTorch | 1.11.0, CUDA build 11.3 |
| CUDA available | `false` |
| Execution device | CPU |
| PyTorch CPU threads | 1 |
| Container | `/scratch/network/km6349/skexgen.sif` |
| Container content hash | Not preserved |

The scheduler state, elapsed time, and peak memory are from the supplied
`sacct` summary. Environment and device values are emitted by the runner.

## Execution

The preserved repository runner
`prototype/graph_encoder/adroit/c5_cpu_validation.slurm` checked the exact
commit and clean source state, entered the fixed container, verified the
runtime, and ran each gate in fail-fast order. The terminal-success event was
emitted only after the final exact-commit and clean-tree assertions.

## Verified results

| Gate | Result |
|---|---|
| Focused C5 suite | 22/22 passed; 0 failures, 0 errors, 0 skips |
| Complete graph-encoder suite | 110/110 passed; 0 failures, 0 errors, 0 skips |
| Model-data regressions | 86/86 passed |
| Flat-baseline regressions | 549 run; 545 passed and 4 pre-existing tests skipped |
| Graph-baseline regressions | 17/17 passed |
| Representation regressions | 40/40 passed |
| Controlled-data regressions | 51/51 passed |
| Documentation validation | Passed for 65 Markdown files |
| Graph-encoder compilation | Passed |
| Python 3.8 grammar | Passed for all 7 enumerated C5 source/test files |
| Final exact-commit and clean-tree gate | Passed |
| Terminal success marker | Present |

The two previously failing real-PyTorch assertions for complete graph-contract
parity and autonomous result fields passed. The corrected padding/masking test
also passed. Exact constrained V6 node/geometry parity, initial Graph V1 main
pair-logit parity, common-loss parity, finite gradients, strict checkpoint
round-trip, matched decoder state, disjoint mutable objects, construction-order
independence, target-free inference signatures, and protected-import guards all
passed in the focused suite.

## Decision

C5 passes its authoritative Python 3.8/PyTorch 1.11 CPU validation gate at
exact commit `996016df44b7f9a6cd5c092a3e3b7a87d9964f9d`. C5 is complete. This
decision supersedes only the pending status created by failed attempt
`3344278`; it does not erase that historical attempt.

## Interpretation

The result establishes that both C4 encoder arms can use the frozen shared C5
decoder contract in the authoritative runtime, that the retained decoder
components meet their specified exact-parity boundaries on the procedural
fixtures, and that the common loss, autonomous prediction levels, gradients,
initialization isolation, and strict checkpoint behavior operate as specified.
It also confirms that the three corrections after job `3344278` repaired test
assertions without requiring a production-model change.

## Limitations and claims not supported

This is an infrastructure and parity validation, not a trained-model result.
It does not establish decoder sufficiency on the authorized train set, memory
sensitivity after training, comparative development performance, RR systematic
generalization, ER test performance, editing behavior, VQ behavior, or any C6,
C7, or C8 outcome. The container path and runtime versions were recorded, but
the container image itself was not cryptographically hashed. The four
flat-baseline regression skips were permitted outside the two zero-skip C5
gates and are not evidence about C5.

## Artifacts and integrity

| Artifact | Size | SHA-256 |
|---|---:|---|
| `c5-cpu-3344290.out` | 3,662 bytes | `5e726008e1306af4b8c4d968d2e8f938f7b5f2585c88bce4cfc0affc45455621` |
| `c5-cpu-3344290.err` | 4,664 bytes | `7cd72c066e3fd6c2a2d02b10238c098b5169be8b34d77876b54dd5b2cc50b930` |
| Incremental correction bundle | 7,363 bytes | `dffcbf7a1310c648028aec67458cbd70a7c92733f23727c9126c9132e63ffabd` |

The stdout and stderr are retained under
`/Users/krishaymaskara/Downloads/c5-validation-3344290/`. The incremental
bundle advertised exact `HEAD` `996016df44b7f9a6cd5c092a3e3b7a87d9964f9d`
and required the already transferred C5 parent
`ffa6091cfecb37aa648ed3590f591f247a776754`.

## Reproduction and validation

The final audit read complete stdout and stderr, recomputed both local artifact
hashes and sizes, checked every structured runner event, reconciled the
human-readable test output with the structured counts, and reran repository
documentation validation after recording the result.

## Related records

- Predecessor: [Failed C5 validation attempt 3344278](ge1_c5_cpu_validation_attempt_3344278.md)
- Contract: [C5 shared decoder contract](../specifications/ge1_shared_decoder_contract.md)
- Protocol: [GE1 implementation plan](../specifications/graph_encoder_implementation_plan.md)
- Package: [Graph encoder README](../../prototype/graph_encoder/README.md)
- Current outcome: [Project status](../status.md)
