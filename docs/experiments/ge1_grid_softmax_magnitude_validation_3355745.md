# GE1 Grid-Softmax Magnitude Authoritative CPU Validation

## Record status

| Item | Value |
|---|---|
| Experiment ID | `ge1-grid-softmax-magnitude-validation-3355745` |
| Record status | `verified` |
| Scientific role | Engineering validation of the prospective grid-softmax magnitude implementation only |
| Branch | `graph-v1-experiment-record` |
| Source commit | `12521f64c73bf5f818156fe62722e64a0537d59c` |
| Governing decision | Accepted [ADR-0015](../decisions/ADR-0015-ge1-grid-softmax-operation-magnitude-classification.md) |
| Contract | [Grid-softmax magnitude specification](../specifications/ge1_grid_softmax_magnitude.md) |
| Slurm job | `3355745` |
| Compute host | `adroit-h11n2` |
| Runner | `prototype/graph_encoder/adroit/ge1_grid_softmax_magnitude_validation_cpu.slurm` |
| Runner SHA-256 | `c77395c481ff598b2eac0c0bc9e6d1c8e8b281847c4ba4f9832a621623521973` |
| Working tree | Clean before and after validation; standalone detached checkout |
| Execution date | August 20, 2026 |

## Question and predetermined gate

The validation asked whether the prospective
`GE1-OPERATION-MAGNITUDE-GRID-SOFTMAX-v1` decoder contract executes correctly
in the authoritative Python 3.8.13 / PyTorch 1.11.0 CPU environment while
leaving `GE1-OPERATION-MAGNITUDE-GRID-ORDINAL-v1` and every protected-access
boundary unchanged.

The gate required, as frozen before the run:

- the exact clean standalone detached checkout at commit
  `12521f64c73bf5f818156fe62722e64a0537d59c` before and after testing;
- CPython 3.8.13, PyTorch 1.11.0, CPU-only execution, one PyTorch thread;
- exactly 27 focused contract tests and exactly 36 focused runtime tests —
  63 total — passing with zero skips;
- complete graph-encoder discovery passing with zero failures and zero errors,
  permitting **exactly** the six `Stage6CudaRuntimeTests` CUDA-unavailable
  skips in `CPU_DISCOVERY_CUDA_SKIP_IDS` and rejecting every missing,
  duplicate, or additional skip;
- the five sibling regression suites at exactly 86 / 549 / 17 / 40 / 51 tests,
  permitting only the four established flat-baseline skips;
- distinct checkpoint schemas and distinct shared-decoder versions for the two
  grid identities, with a deliberately shared output-position version;
- `ORDINAL_CUT_COUNT` confined to `grid_magnitude.py` and `losses.py`;
- the three ADR-0013 ordinal runners unchanged against parent commit
  `9c18b438e571dc49d4e51c90eda8e0a5108b0996`;
- documentation, compileall, Python 3.8 grammar, source-audit, Bash-syntax,
  whitespace, and clean-tree gates to pass; and
- no manifest, payload, corpus, training, checkpoint writing, CAD kernel,
  Stage 6, or C8 activity.

Any mismatch, failure, error, unauthorized skip, dirty tree, protected access,
or absent terminal-success marker would fail validation.

## Evidence integrity and scheduler result

The four supplied raw files were rehashed locally. The first three digests
exactly match `grid-softmax-3355745.log-SHA256SUMS`, which was generated on
Adroit before transfer; `shasum -a 256 -c` reported `OK` for all three. The
manifest cannot contain its own digest, so it is hashed separately.

| Artifact | Bytes | SHA-256 | Availability |
|---|---:|---|---|
| `grid-softmax-cpu-3355745.out` | 14,147 | `4fc8f87ff9f5af61efdac8c65edd68fff3e40d00e0341984e90a8874dca804ff` | `present` |
| `grid-softmax-cpu-3355745.err` | 13,022 | `42d8d33d09aea1ff8e80c9010674cac105cf9d12350a86356955af6c503862f5` | `present` |
| `grid-softmax-3355745.sacct.txt` | 335 | `a2f36ccc6504217ed1cca944ff5fc83ed3df9f6d558a700bbde61151ff80780f` | `present` |
| `grid-softmax-3355745.log-SHA256SUMS` | 287 | `a2894fdd63df2d6ed094f6b0a6364d1a779b25a0e023399856e6f755d32c481d` | `present` |

Local copies are retained at
`~/Downloads/ge1-grid-softmax-3355745/`. Scratch paths on Adroit are not
durable evidence on their own; these checksums are.

Scheduler result:

| Item | Value |
|---|---|
| State / exit code | `COMPLETED` / `0:0` |
| Elapsed | `00:09:20` |
| Batch step MaxRSS | `748828K` |
| Allocated CPUs | `1` |

Runner and scheduler evidence agree on commit `12521f6…`, job `3355745`, host
`adroit-h11n2`, Python `3.8.13`, PyTorch `1.11.0`, and `cpu` execution. The
terminal-success marker
`grid_softmax_magnitude_validation_terminal_success` is present and no
`grid_softmax_magnitude_validation_terminal_failure` marker appears in either
stream.

## Test and repository results

| Suite | Declared | Run | Failures | Errors | Skips |
|---|---:|---:|---:|---:|---:|
| `test_grid_softmax_magnitude_contract` | 27 | 27 | 0 | 0 | 0 |
| `test_grid_softmax_magnitude_runtime` | 36 | 36 | 0 | 0 | 0 |
| Complete graph-encoder discovery | 709 | 709 | 0 | 0 | 6 |
| `model_data` | 86 | 86 | 0 | 0 | 0 |
| `flat_baseline` | 549 | 549 | 0 | 0 | 4 |
| `graph_baseline` | 17 | 17 | 0 | 0 | 0 |
| `representation` | 40 | 40 | 0 | 0 | 0 |
| `controlled_data` | 51 | 51 | 0 | 0 | 0 |

The six complete-discovery skips matched `CPU_DISCOVERY_CUDA_SKIP_IDS`
exactly; they are the `Stage6CudaRuntimeTests` methods, which skip on any host
without CUDA. The four `flat_baseline` skips are the established
external-bundle skips.

Documentation validation passed for 108 Markdown files, `compileall` passed
over the whole `prototype/graph_encoder` package, and Python 3.8 grammar
parsing covered 103 Python files. The recorded runtime was CPython `3.8.13`
with PyTorch `1.11.0`, one thread, `cuda_available=false`. The PyTorch build
reports CUDA `11.3`; the host simply exposes no device, which is what makes
the six `Stage6CudaRuntimeTests` methods skip rather than fail.

**Environment-dependent count note.** A pre-submission local run under
PyTorch 2.13 reported 699 discovered tests with three
`UnpicklingError: Weights only load failed` errors, caused by torch ≥ 2.6
changing the `torch.load(weights_only=…)` default. Under the authoritative
1.11.0 environment the same discovery reported 709 tests with zero errors.
The most likely explanation is that a test module failed to import under
2.13, so `unittest` substituted a single placeholder test for that module's
methods. The authoritative count is 709; the local 2.13 figure is not
evidence about the production environment.

## Identity and contract separation exercised

The runner's identity audit confirmed in the production environment:

| Item | Softmax identity | Ordinal identity |
|---|---|---|
| Parameterization | `GE1-OPERATION-MAGNITUDE-GRID-SOFTMAX-v1` | `GE1-OPERATION-MAGNITUDE-GRID-ORDINAL-v1` |
| Checkpoint schema | `GE1-CHECKPOINT-v3` | `GE1-CHECKPOINT-v2` |
| Shared-decoder version | `GE1-SHARED-TYPED-EDGE-DECODER-V3` | `GE1-SHARED-TYPED-EDGE-DECODER-V2` |
| Output-position contract | `GE1-DECODER-OUTPUT-POSITIONS-v2` | `GE1-DECODER-OUTPUT-POSITIONS-v2` |

Both identities route through `uses_grid_magnitude`; the ordinal identity
retains tuple index 2 in `OPERATION_MAGNITUDE_PARAMETERIZATIONS`. Distinct
checkpoint schemas prevent a governed schema check from passing on a
structurally incompatible state dict, while the shared output-position version
records that output channels 37 and 38 are genuinely unchanged.

`ORDINAL_CUT_COUNT` was confirmed confined to `grid_magnitude.py` and
`losses.py`. The three ADR-0013 ordinal runners were byte-identical to parent
commit `9c18b43…`, so their exact-commit preflight contracts remain valid.

The focused suites assert each of these as executable contracts rather than
prose. Named tests that passed include
`test_softmax_gets_its_own_decoder_version_and_checkpoint_schema`
("a shared schema would pass a governed check on an incompatible state"),
`test_new_identity_is_appended_without_moving_any_existing_index`,
`test_every_default_is_unchanged_so_the_identity_is_opt_in`,
`test_state_dict_keys_and_shapes_are_incompatible_with_the_ordinal_head`,
`test_the_cut_count_never_leaves_its_two_owning_modules`,
`test_pinned_ordinal_suite_counts_are_unchanged`
("three committed exact-commit runners assert these literals"), and
`test_raw_loss_equals_the_manual_cross_entropy_reduction`
("steps 2–4 of the ADR-0013 reduction must be byte-identical").

Three tests target the diagnosed mechanism directly.
`test_class_evidence_moves_independently` requires that perturbing one class
row leaves every other class logit unmoved — the property a single shared
scalar structurally cannot provide. The two reachability tests then pin the
diagnosis in both directions: every grid class becomes reachable for the
softmax head within the diagnostic budget, and the ordinal head still fails
the same protocol. Job `3353008`'s finding is therefore preserved as an
executable assertion that cannot silently rot.

## Access and execution audit

The `GE1-GRID-MAGNITUDE-SOURCE-AUDIT-v1` structural audit passed with
`production_file_count=5`, `repository_read_only_bind_count=1`,
`submission_command_invoked=false`, `scientific_command_invoked=false`, and
`scientific_data_path_declared=false`.

Terminal telemetry recorded `false` for every protected-access and execution
field: operation-template manifest, operation-template train payload,
development, systematic RR, test ER, IID, history-depth,
geometry-extrapolation, other corpus, scientific training, engineering
training, checkpoint written, CAD kernel, Stage 6, and C8 or later.

The repository was bound read-only and no output namespace other than the
Slurm log parent was written. Bounded optimizer steps inside the reachability
tests operate on procedurally generated states and ephemeral test fixtures;
no trained or published model checkpoint was produced.

Two C7-v2 JSON telemetry lines appear inside the passing suites and are
expected test output, not run events. `c7_v2_terminal_execution_completed`
in stdout carries `"artifact_path":"/external/c7-v2"` and
`structured_reason=autonomous_exact_sufficiency_failure`;
`c7_v2_infrastructure_failure` in stderr carries `detail=preflight`. Both are
emitted by tests that deliberately exercise C7-v2 result paths against
procedural fixtures. **The `/external/c7-v2` string is a synthetic fixture
path, not a real external artifact, and no such path was opened.** Every
protected-access field in both lines reads `false`, and the runner's own
terminal telemetry independently records the same.

## Decision

The authoritative engineering-validation gate **passed** for exact commit
`12521f64c73bf5f818156fe62722e64a0537d59c`.

This accepts the prospective grid-softmax magnitude implementation as
validated under Python 3.8.13 and PyTorch 1.11.0 on Adroit CPU. It does not
reinterpret or rerun ADR-0013's scientific job `3352404`, authorize a
scientific execution of the softmax identity, authorize Stage 6, establish
CAD-executor validity, or provide evidence of comparative encoder
performance. Protected partitions remain closed.

## Limitations

- No corpus example, manifest, learned checkpoint, optimizer trajectory, or
  autonomous trained-model output was evaluated.
- Class reachability under the fixed-generated-state protocol is a **necessary
  condition** for the operation-fidelity gate, not a sufficient one. It is
  engineering evidence about the head and loss on generated states, not
  evidence about scientific checkpoints or corpus magnitude fidelity, and it
  does not predict the outcome of any future scientific run.
- The 709-test discovery total and the 86 / 549 / 17 / 40 / 51 sibling counts
  are pinned in the runner. They will require deliberate updating whenever
  tests are added, in the same way the earlier hard-pinned counts went stale.
- The container path and runtime versions were recorded, but the container
  image was not cryptographically hashed.
- An initial `scp` invocation used a single quoted multi-file remote argument,
  which the SFTP-backed transfer treated as one filename and rejected. It was
  reissued with explicit sources; no evidence was altered and the manifest
  digests matched on the successful transfer.
- Two operator shell sessions contained unsubstituted placeholders
  (`<paste-commit-from-step-1>` and `PUT_JOB_ID_HERE`). Both failed or printed
  visibly and were corrected before submission and monitoring respectively.
  An initial sync attempt cloned from an Adroit repository that lacked the
  branch and failed visibly; the checkout was then created by pushing the
  branch directly over SSH. None of these affected the validated checkout,
  the submitted job, or the recorded results.

## Related records

- [ADR-0015](../decisions/ADR-0015-ge1-grid-softmax-operation-magnitude-classification.md)
- [Grid-softmax magnitude specification](../specifications/ge1_grid_softmax_magnitude.md)
- [ADR-0013](../decisions/ADR-0013-ge1-grid-anchored-ordinal-operation-magnitude-repair.md)
- [Grid magnitude repair contract](../specifications/ge1_grid_magnitude_repair.md)
- [Operation-magnitude repair CPU validation](ge1_operation_magnitude_repair_cpu_validation.md)
- [Current project status](../status.md)
- [Graph encoder README](../../prototype/graph_encoder/README.md)
