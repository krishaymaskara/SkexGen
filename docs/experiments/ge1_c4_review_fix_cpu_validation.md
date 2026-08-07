# GE1 C4 Review-Fix Authoritative CPU Validation

## Record status

| Item | Value |
|---|---|
| Experiment ID | `ge1-c4-review-fix-cpu-3344265` |
| Record status | `verified` |
| Scientific role | Infrastructure and compatibility validation of the accepted C4 review fixes |
| Branch | `graph-v1-experiment-record` |
| Source commit | `3a41abc81f68ef6d6450465e05b3544f07a08b83` |
| Source parent | `91417c2dc4d265ed705d75b24f75aff7e7ca09ea` |
| Slurm job | `3344265` |
| Compute host | `adroit-h11n3` |
| Working tree | Clean before and after validation |
| Execution date | August 7, 2026 |

## Question and predetermined decision rule

The validation asked whether the C4 review-fix implementation works in the
authoritative Python 3.8/PyTorch 1.11 CPU environment. The reviewed change
freezes the completed capacity match in configuration and initializes every
genuinely shared encoder component from one arm-independent source.

The gate required:

- a clean checkout of exact commit
  `3a41abc81f68ef6d6450465e05b3544f07a08b83` before and after testing;
- Python 3.8.13 and PyTorch 1.11.0 with CUDA unavailable;
- both new shared-initialization tests to run and pass with zero skips;
- all 28 C4 encoder tests to run and pass with zero skips;
- all 88 tests in the complete graph-encoder suite to run and pass with zero
  skips;
- successful compilation of `prototype/graph_encoder`; and
- no training, C5 work, authoritative manifest access, CAD-history payload
  access, systematic-partition access, or test-partition access.

The runner treated a wrong count, skip, failure, error, environment mismatch,
commit mismatch, dirty source tree, or missing terminal-success event as a
failed validation.

## Inputs and partition authority

| Item | Value |
|---|---|
| Dataset/corpus | Procedural and temporary in-repository test fixtures only |
| Authoritative manifest opened | `false` |
| Authoritative CAD-history payload opened | `false` |
| Systematic RR partition opened | `false` |
| Held-out ER test partition opened | `false` |
| Training performed | `false` |
| C5 work performed | `false` |

The encoder tests construct the six controlled E/R/EE/ER/RE/RR templates in
memory. Partition-boundary tests use temporary synthetic fixture manifests and
verify that protected assignments fail before payload loading. The Slurm
runner supplied no corpus, authoritative-manifest, or CAD-history path.

## Configuration under review

The validated review-fix commit freezes:

- two relation bases for both encoder arms;
- flat encoder feed-forward width 192;
- typed-graph feed-forward width 64;
- 22,800 trainable flat-encoder parameters;
- 23,468 trainable graph-encoder parameters;
- a 2.9298245614% graph excess relative to the flat arm, within the frozen 5%
  capacity tolerance; and
- identical initial values, but disjoint parameter objects, across the shared
  `field_embeddings`, `geometry_projection`, `geometry_mask_projection`,
  `input_norm`, `latent_queries`, `to_codebook`, and `from_codebook`
  components.

The flat Transformer and typed-graph relational/pooling stacks remain
arm-specific. C5 decoder construction is absent.

## Environment

| Item | Verified value |
|---|---|
| Cluster / host | Princeton Adroit / `adroit-h11n3` |
| Slurm job | `3344265` |
| Python | CPython 3.8.13 |
| PyTorch | 1.11.0, CUDA build 11.3 |
| CUDA available | `false` |
| Execution device | CPU |
| PyTorch CPU threads | 1 |
| Container | `/scratch/network/km6349/skexgen.sif` |
| Container content hash | Not preserved |

## Execution

The source was transferred as a complete-history Git bundle and checked out
at the exact detached commit. The Slurm runner used `set -Eeuo pipefail`, a
terminal-failure trap, a clean Apptainer environment, hidden CUDA visibility,
one OpenMP thread, one MKL thread, and an external bytecode/temp directory so
the checkout stayed clean.

It ran three separately counted unittest gates:

1. the two newly added shared-initialization tests;
2. `prototype.graph_encoder.tests.test_encoders`; and
3. discovery under `prototype/graph_encoder/tests`.

It then compiled `prototype/graph_encoder` and repeated the exact-commit and
clean-worktree checks.

## Verified results

- Both shared-initialization tests passed: 2/2, zero failures, zero errors, and
  zero skips.
- The C4 encoder suite passed: 28/28, zero failures, zero errors, and zero
  skips.
- The complete graph-encoder suite passed: 88/88, zero failures, zero errors,
  and zero skips.
- `python3.8 -B -m compileall prototype/graph_encoder` passed.
- The environment assertions reported CPython 3.8.13, PyTorch 1.11.0, CPU
  execution, one thread, and unavailable CUDA.
- Exact-commit and clean-worktree checks passed before and after testing.
- The terminal event was `c4_review_fix_terminal_success` for commit
  `3a41abc81f68ef6d6450465e05b3544f07a08b83` and job `3344265`.
- Runtime markers explicitly report `false` for authoritative manifest,
  CAD-history payload, systematic partition, test partition, training, and C5
  access/work.

The verbose unittest output names both new tests and every C4 and complete-suite
test as `ok`. There is no failed test, traceback, terminal-failure event, or
skip in either log.

## Decision

The authoritative C4 review-fix CPU validation gate **passed**. Exact commit
`3a41abc81f68ef6d6450465e05b3544f07a08b83` is accepted as the validated C4
standalone-encoder implementation.

This supersedes the pending-revalidation status created by the review fix. It
does not supersede the historical result at the original C4 implementation
commit, and it does not begin or validate C5.

## Interpretation

The result confirms that the completed capacity choices are enforced in the
runtime configuration and that the genuinely shared components have identical
initial values across arms without sharing live parameter objects. It also
reconfirms all earlier C4 parity, gradient, isolation, sensitivity, capacity,
and permutation properties after the review changes.

This is an implementation and compatibility result, not a trained-model or
scientific encoder comparison.

## Limitations and claims not supported

- No `sacct` transcript was supplied, so scheduler state, scheduler exit code,
  elapsed allocation time, and maximum memory are not independently recorded.
  The runner's terminal-success event and complete stdout/stderr establish the
  process-level pass.
- The container content hash was not recorded.
- The stderr artifact is nonempty because Python's `unittest.TextTestRunner`
  writes verbose successful test output to stderr by default. It contains only
  named `ok` results and three final `OK` summaries, not runtime errors.
- The logs are preserved locally but are not packaged in a separate immutable
  archive.
- The result does not establish decoder integration, optimization,
  reconstruction quality, executable CAD validity, checkpoint behavior,
  systematic generalization, or any trained-model endpoint.

## Artifacts and integrity

| Artifact | Location | Size | SHA-256 | Availability |
|---|---|---:|---|---|
| Slurm stdout | `/Users/krishaymaskara/Downloads/c4-review-fix-validation-3344265/c4-review-fix-3344265.out` | 2,932 bytes | `6c7668e352a754a12a9a1c07daf99efc82ebecc7e78c24feaea7b1f86e04acc0` | present |
| Slurm stderr | `/Users/krishaymaskara/Downloads/c4-review-fix-validation-3344265/c4-review-fix-3344265.err` | 13,841 bytes | `6c25339b2b2e039907aa39cde56c145f508bd5e74422f74f1da283a47acf2732` | present |
| Complete-history Git bundle | `/Users/krishaymaskara/Downloads/c4-review-fix-3a41abc-transfer/SkexGen-c4-review-fix-3a41abc.bundle` | 77,882,327 bytes | `349e843238842590d66aa66aa5fcc3407f77aaa6aee4d3d32bb0a950a1cd395f` | present |
| Bundle checksum file | `/Users/krishaymaskara/Downloads/c4-review-fix-3a41abc-transfer/SkexGen-c4-review-fix-3a41abc.bundle.sha256` | 103 bytes | `d0f8c19d6bf9be5eb88cf18261f65d38eb247d936d46f9f755abfff2cb9b1306` | present |
| Submitted Slurm runner | `/private/tmp/c4_review_fix_validation_3a41abc.slurm` | 6,632 bytes | `2b392336bea1265cfd12c39008084c0ce6759894d1eaea0a5da747c35047447b` | present locally; not repository-tracked |

The retained bundle independently verifies as a complete-history bundle with
sole advertised `HEAD`
`3a41abc81f68ef6d6450465e05b3544f07a08b83`. Its recomputed SHA-256 matches
the checksum file.

## Reproduction and validation

The final audit read complete stdout and stderr, recomputed all retained
artifact hashes, verified the Git bundle from inside the repository, confirmed
its sole advertised head, checked every structured gate record, and reran
repository documentation validation after recording the result.

## Related records

- Predecessor: [Original C4 authoritative CPU validation](ge1_c4_cpu_validation.md)
- Protocol: [GE1 Stage 0 preregistration](../specifications/ge1_stage0_preregistration.md)
- Implementation contract: [Typed graph encoder implementation plan](../specifications/graph_encoder_implementation_plan.md)
- Package contract: [Graph encoder README](../../prototype/graph_encoder/README.md)
- Current outcome: [Project status](../status.md)
