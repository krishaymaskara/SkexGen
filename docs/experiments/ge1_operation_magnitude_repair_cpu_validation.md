# GE1 Operation-Magnitude Repair Authoritative CPU Validation

## Record status

| Item | Value |
|---|---|
| Experiment ID | `ge1-operation-magnitude-repair-cpu-3345044` |
| Record status | `verified` |
| Scientific role | Engineering validation of the prospective operation-magnitude repair only |
| Branch | `graph-v1-experiment-record` |
| Source commit | `12167ce7d0dc025c4b297b7ccb8a3011580bf0fc` |
| Repair implementation commit | `b43fa26e87cd058cc415a5718d52a1cbf3b60186` |
| Governing decision | Accepted [ADR-0009](../decisions/ADR-0009-ge1-positive-operation-magnitude-repair.md) |
| Slurm job | `3345044` |
| Compute host | `adroit-h11n2` |
| Working tree | Clean before and after validation; standalone detached checkout |
| Execution date | August 10, 2026 |

## Question and predetermined gate

The validation asked whether the prospective
`GE1-OPERATION-MAGNITUDE-POSITIVE-v1` decoder contract executes correctly in
the authoritative Python 3.8/PyTorch 1.11 CPU environment while preserving
the explicit historical
`GE1-OPERATION-MAGNITUDE-TANH-LEGACY-v1` behavior and every protected-access
boundary.

The gate required:

- the exact clean standalone checkout at commit
  `12167ce7d0dc025c4b297b7ccb8a3011580bf0fc` before and after testing;
- CPython 3.8.13, PyTorch 1.11.0, CPU-only execution, and one PyTorch thread;
- all 13 focused repair tests, including all eight real-PyTorch tests, to pass
  with zero skips;
- the complete graph-encoder suite to pass with zero failures, errors, or
  skips;
- every repository regression suite to pass, permitting only the four
  established flat-baseline external-bundle skips;
- documentation, compilation, Python 3.8 grammar, import/export,
  target-leakage, protected-access, frozen-V6, Bash-syntax, and clean-tree
  gates to pass; and
- no manifest, payload, corpus, training, checkpoint writing, CAD kernel,
  Stage 6, or C8 activity.

Any mismatch, test failure/error/unauthorized skip, dirty tree, protected
access, or absent terminal-success marker would fail validation.

## Evidence integrity and scheduler result

The three supplied raw files were rehashed locally. Their digests exactly
match `operation-magnitude-3345044.log-SHA256SUMS`:

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `operation-magnitude-cpu-3345044.out` | 6,897 | `6b757faae35d897d918abbe478ce2c6d4b4b5ba39206d486c0f68136bc19a4a8` |
| `operation-magnitude-cpu-3345044.err` | 4,766 | `2b3a77c8a415d1f2fd3f40574f72a135ec3aabac6543a28f5f1c243451f2f498` |
| `operation-magnitude-3345044.sacct.txt` | 335 | `2c4144c0df288a55a0c86640291b7fede1b92877a075dac9f363b9377afceb71` |
| `operation-magnitude-3345044.log-SHA256SUMS` | 308 | `635a8fb946d5f7cccafa9f914a4bb6afde33f6664724b758dd483c4ec26c6263` |

Scheduler and runner evidence agree on:

| Item | Verified value |
|---|---|
| Job / host | `3345044` / `adroit-h11n2` |
| Scheduler state / exit | `COMPLETED` / `0:0` |
| Elapsed / batch MaxRSS | `00:06:07` / `619956K` |
| Python | CPython 3.8.13 |
| PyTorch | 1.11.0, CUDA build 11.3 |
| CUDA available | `false` |
| Device / CPU threads | CPU / 1 |
| Container | `/scratch/network/km6349/skexgen.sif` |

The runner start event records the exact commit, clean source, standalone
checkout, identities, job, host, and access declarations. The terminal event
repeats the same commit, job, host, environment, identities, and negative
access/work declarations.

## Test and repository results

| Gate | Verified result |
|---|---|
| Focused operation-magnitude suite | 13/13 passed; 0 failures, 0 errors, 0 skips |
| Complete graph-encoder suite | 292/292 passed; 0 failures, 0 errors, 0 skips |
| Model-data regressions | 86/86 passed; 0 skips |
| Flat-baseline regressions | 549 run; 545 passed; exactly 4 established skips |
| Graph-baseline regressions | 17/17 passed; 0 skips |
| Representation regressions | 40/40 passed; 0 skips |
| Controlled-data regressions | 51/51 passed; 0 skips |
| Documentation validation | Passed for 83 Markdown files |
| Graph-encoder compileall | Passed |
| Python 3.8 grammar | Passed for all 41 enumerated source/test files |
| Import/export and protected-access/target audit | Passed; 90 exports |
| Frozen inherited V6 audit | Passed; no `prototype/flat_baseline` diff from the accepted ADR parent |
| Slurm runner Bash syntax | Passed |
| Final exact-commit and clean-tree checks | Passed |

The four flat-baseline skips are the previously established immutable external
replay-bundle cases recorded by the C5/C6 review-fix validation. Every other
suite reported zero skips. The `/external/c7-v2` terminal lines in stdout are
mocked CLI output from graph-encoder unit tests, not real artifacts or data
access.

## Repair behavior exercised

The verbose focused transcript names all five pure and eight real-PyTorch
tests as `ok`. Together with the complete-suite contract tests, they establish
the following engineering coverage:

- controlled extrusion and revolve targets are positive and bounded by one,
  including an exact normalized `1.0` 360-degree revolve target;
- the legacy configuration returns exact `tanh(raw)` values and exposes the
  unchanged raw head output;
- extreme negative, zero, positive, and maximum finite logits produce finite
  repaired operation magnitudes strictly above zero and no greater than one;
- only compact channels 4 and 5, serialized as extrusion distance 37 and
  revolve angle 38, use the repaired mapping;
- non-operation geometry, geometry masks, training masks, direction logits,
  Boolean logits, and all other categorical logits remain exact;
- all five synthetic diagnostic-shaped negative cases pass strict conversion
  and analytic controlled-domain validation without
  `invalid_operation_parameter`;
- legacy/repaired inference checkpoint interchange fails with stable
  `invalid_ge1_checkpoint`, and recovery checkpoint interchange fails with
  stable `invalid_training_checkpoint`;
- configuration, inference checkpoint fields, recovery checkpoint model
  configuration and provenance, C6 provenance, metrics, and reporting records
  carry the operation-magnitude identity;
- flat and typed-graph arms retain value-identical, independently mutable
  shared-decoder initialization under the repaired identity; and
- the typed graph encoder gains no chronological or absolute-position input.

The conversion test explicitly checks that no CAD-kernel result is present.
This validation establishes the neural and analytic contract only.

## Access and execution audit

The runner bound only the standalone repository, read-only. It supplied no
corpus or manifest path. Its source audit rejected model-data loader imports
from the repair-path modules and rechecked the target-free autonomous public
interface. Start and terminal records both declare `false` for:

- the operation-template manifest and train payload;
- development, RR systematic, ER test, IID, history-depth, and
  geometry-extrapolation partitions;
- every other corpus;
- scientific or engineering training;
- checkpoint writing and CAD-kernel use; and
- Stage 6 and C8-or-later work.

No optimizer step, backward pass, repaired scientific run, or model checkpoint
was produced. Temporary checkpoint objects used by strict unit tests are
ephemeral test fixtures, not trained or published model checkpoints.

## Decision

The authoritative engineering-validation gate **passed** for exact commit
`12167ce7d0dc025c4b297b7ccb8a3011580bf0fc`.

This accepts the prospective operation-magnitude implementation as validated
under Python 3.8.13 and PyTorch 1.11.0 on Adroit CPU. It does not reinterpret
or rerun C7-v2, authorize Stage 6, freeze or authorize a repaired scientific
protocol, establish CAD-executor validity, or provide evidence of comparative
encoder performance. Protected partitions remain closed.

## Limitations

- No corpus example, learned checkpoint, optimizer trajectory, or autonomous
  trained-model output was evaluated.
- Positive bounded magnitudes prevent the diagnosed analytic negativity class
  but do not establish target accuracy, solid construction, or CAD-kernel
  validity.
- The container path and runtime versions were recorded, but the container
  image was not cryptographically hashed.
- The terminal transcript contains an initial evidence-retrieval attempt with
  the literal placeholder `<job-id>`. It failed visibly after the completed
  job and was corrected to `3345044`; it did not affect submission, execution,
  or the verified downloaded files.

## Related records

- [ADR-0009](../decisions/ADR-0009-ge1-positive-operation-magnitude-repair.md)
- [GE1 implementation plan](../specifications/graph_encoder_implementation_plan.md)
- [Shared decoder contract](../specifications/ge1_shared_decoder_contract.md)
- [C7-v2 operation-parameter diagnostic](../specifications/ge1_c7_v2_operation_parameter_diagnostic.md)
- [Current project status](../status.md)
- [Graph encoder README](../../prototype/graph_encoder/README.md)
