# GE1 C7 Train-Only Sufficiency Pilot

## Record status

| Item | Value |
|---|---|
| Experiment ID | `ge1-c7-pilot-3344505` |
| Record status | `verified-local` |
| Scientific role | Formal C7 train-only decoder-sufficiency and memory-use gate |
| Scientific outcome | Completed negative result: autonomous exact sufficiency failed |
| Branch | `graph-v1-experiment-record` |
| Source commit | `4bfde4c726a585433ea4bb60e6ce9d245ae7c87d` |
| Slurm job | `3344505` |
| Compute host | `adroit-h11n2` |
| Working tree | Clean before and after execution |
| Execution date | August 8, 2026 |

## Question and predetermined decision rule

The formal C7 pilot asked whether each continuous-memory arm could exactly
reconstruct the same deterministic four-family train-only cohort after the
frozen 50-epoch budget, before any 32-family sufficiency or memory-use gate
was attempted.

The predetermined dependency order was:

1. train a fresh matched flat/typed-graph pair on one authorized train family
   from each of `E`, `R`, `EE`, and `RE`;
2. strictly reload each arm's epoch-50 checkpoint and require exact nodes,
   exact typed graph, successful conversion, complete validity, and exact
   `depends_on` for two-operation families on every family;
3. run an arm's scaled 32-family gate only if that arm passed its tiny gate;
4. run an arm's scaled memory interventions only if that arm reached the
   scaled gate; and
5. authorize Stage 6 only if every exact and memory gate passed for both arms.

A completed exact-sufficiency failure was a scientific negative result with
process exit zero. Infrastructure, provenance, access, reload, or artifact
integrity failures were nonzero errors. Exact tiny failure was preauthorized
to activate the decoder-repair trigger but did not implement or invoke repair.

## Inputs and partition authority

| Item | Verified value |
|---|---|
| Authorized manifest | `operation_template` |
| Manifest SHA-256 | `a9ac86a6dede054fbbba57e0906b210bab26036c3f5c150b332038f78d2dadb7` |
| Authorized payload | Four selected `operation_template.train` families only |
| Templates | `E`, `R`, `EE`, `RE` |
| Tiny-family set SHA-256 | `c07e90675099cb2ea58575fe91683dc80b70cf3fa85fa6fa8de0c44dbe9783d6` |
| Seed | 2026 |
| Train manifest / payload opened | `true` / `true` |
| Development opened | `false` |
| RR systematic opened | `false` |
| ER test opened | `false` |
| IID opened | `false` |
| History-depth opened | `false` |
| Geometry-extrapolation opened | `false` |
| C8 or later performed | `false` |

The four selected family identifiers were:

| Template | `source_family_id` |
|---|---|
| `E` | `sf_d11886750826237040bec2a8580aa378e367b3e7ba51dd972c57b476408c470c` |
| `R` | `sf_5bbeb6143e53c2d259f24a60b576708dd8267b0ad49ed62c31c966e36e503ac4` |
| `EE` | `sf_0626a6cb7e6b08cd7cae372a0d61d028cdc57c0d5bfd708a46397024e37d72f1` |
| `RE` | `sf_fe8e4fd3246a0c94e25c661b0e879fa9099d098c8f558424898e72f3822746ff` |

## Environment

| Item | Verified value |
|---|---|
| Cluster / host | Princeton Adroit / `adroit-h11n2` |
| Slurm job | `3344505` |
| Scheduler state / exit | `COMPLETED` / `0:0` |
| Elapsed | 1 minute 8 seconds |
| Python | CPython 3.8.13 |
| PyTorch | 1.11.0, CUDA build 11.3 |
| CUDA available | `false` |
| Execution device | CPU |
| PyTorch CPU threads | 1 |
| Container | `/scratch/network/km6349/skexgen.sif` |
| Governed source digest | `a1bfc1c3aecb38901b24283b1b5410b07946a45ca8191ea072758a9bffe9ccce` |

## Execution

The committed C7 runner first checked the detached exact commit and clean
source tree, then completed all pre-corpus validation gates. It opened only
the authoritative manifest and four selected train payloads. For each arm it
built a fresh seed-2026 model from the matched initialization, trained for 50
epochs, wrote the epoch-50 checkpoint before evaluation, and strictly loaded
that checkpoint into a fresh model for autonomous scoring.

Each arm performed one optimizer step per epoch over four families, for 50
optimizer steps and 200 family presentations. This matches the frozen tiny
arithmetic. The scaled budget of 200 steps and 1,600 presentations per arm
was not executed because neither arm passed the prerequisite tiny gate.

## Results and findings

### Validation and artifact gates

| Gate | Result |
|---|---|
| Focused C7 suite | 55/55 passed; 0 failures, errors, or skips |
| Complete graph-encoder suite | 218/218 passed; 0 failures, errors, or skips |
| Model-data regressions | 86/86 passed |
| Representation regressions | 40/40 passed |
| Documentation validation | Passed for 73 Markdown files |
| Compilation / Python 3.8 grammar | Passed |
| Source-leakage and common-decoder audit | Passed |
| Exact-commit and clean-tree gates | Passed before and after execution |
| Artifact verification | Passed; 104 regular files verified |
| Terminal completion marker | Present |

### Scientific gates

| Gate | Flat | Typed graph |
|---|---:|---:|
| Exact nodes | 4/4 | 3/4 |
| Exact typed graph | 0/4 | 0/4 |
| Conversion | 4/4 | 4/4 |
| Complete validity | 0/4 | 0/4 |
| Exact `depends_on` on `EE`/`RE` | 0/2 | 0/2 |
| Tiny exact-sufficiency gate | Failed | Failed |
| Scaled exact gate | `not_run` | `not_run` |
| Scaled shuffled-memory gate | `not_run` | `not_run` |
| Scaled mean-memory gate | `not_run` | `not_run` |

Both training losses declined throughout the 50 updates: approximately
7.57 to 1.43 for the flat arm and 7.55 to 1.51 for the typed-graph arm. Neither
trajectory met the frozen plateau definition, and each still improved by
about 16.5% over its last five epochs. The autonomous outputs converted, but
the predicted graphs omitted required structural relationships. Consequently
all four programs in each arm were invalid, including missing
`depends_on` on both two-operation families. The typed-graph arm also missed
one node sequence.

The four scaled records were correctly emitted as `not_run` rather than
failed measurements: exact and both memory gates for each arm were blocked by
that arm's failed tiny prerequisite.

## Decision and interpretation

The formal C7 execution completed successfully as a process and failed its
scientific sufficiency gate. It reported:

```text
overall_gate_pass=false
stage6_authorized_by_c7=false
preauthorized_decoder_repair_triggered=true
comparison_inconclusive=false
```

This is not evidence of an infrastructure failure, target/data mismatch, or a
gate-contract error. Exact-commit, clean-tree, runtime, strict reload,
arithmetic, test, access, and artifact checks passed. Target losses decreased
substantially, conversions succeeded, and the failure was concentrated in
autonomous structural prediction and downstream validity.

The evidence does not distinguish a hard 50-update optimization shortfall
from a decoder whose autonomous structural path cannot fit even the tiny
cohort under the existing recipe. The still-decreasing losses and lack of a
plateau make optimization sufficiency unresolved. The narrow next permitted
question is therefore a train-only optimization-sufficiency diagnostic; this
record itself authorizes neither that diagnostic nor a decoder implementation.

The preauthorized repair trigger was activated correctly because both arms
failed autonomous exact sufficiency. It permits the already specified repair
procedure to be considered under its governance, but does not by itself
authorize implementation, invocation, protected-data access, Stage 6, or C8.

## Limitations

- Only four selected train families, one seed, and 50 updates per arm were
  evaluated.
- No scaled, memory-intervention, development, RR, ER, or other-manifest
  result exists from this job.
- The run cannot determine whether longer optimization under the identical
  recipe would cross the exact gate.
- The complete checkpoints are externally retained but were not copied into
  this repository.
- The container path and package versions are recorded, but the container
  image was not cryptographically hashed.

## Provenance discrepancy

The log start/terminal events, scheduler evidence, and immutable artifact path
all identify Slurm job `3344505`. However,
`resolved_config.json` records `runtime.slurm_job_id=null`, and both selected
checkpoint provenance records likewise contain a null Slurm job identifier.
This violates the intended structured artifact-identity contract. It did not
change model initialization, training, checkpoint contents, outputs, metrics,
gate dependency logic, or partition access, so it does not alter the
scientific negative result. A later runner must forward and validate the
Slurm job identifier explicitly rather than relying on ambient variables
through an Apptainer clean environment.

## Artifacts and integrity

The supplied local bundle was read directly. Its internal checksum list and
artifact manifest verified successfully. The raw files are external evidence
and are not copied into the repository.

| Artifact | Size | SHA-256 |
|---|---:|---|
| `SHA256SUMS` | 12,456 bytes | `2feddb0c276ece23a48d1411658ba051007708bfabf3dcd534b4391f45e3b394` |
| `artifact_manifest.json` | 16,509 bytes | `1e1c528e6c6c0dbdd78ff23eae1f3427c6dc774bd26147737471b8384fa85138` |
| `c7-cpu-3344505.err` | 7,591 bytes | `fe2a0e4d5e14ebb555a40bd6c5800a9d1962db069d355dedfa620b1340eee38b` |
| `c7-cpu-3344505.out` | 11,138 bytes | `fd589171d9815ae9eeeef01e0fcaa388a23b7a8719f84dd2e93d944b640f84bc` |
| `metrics.jsonl` | 274,029 bytes | `e015e8d60e03ffcc62cf0030dacb7fcbef90c14452a8f509babb2da88429aeb5` |
| `resolved_config.json` | 22,457 bytes | `ccd75ad93d707b44935abcd3fec10c64f597ededd241c3911fd11f44dc4648cb` |
| Terminal transcript | 10,298 bytes | `28054ba1f5462e267847dce1f8db47fc0fadf80105f9ea32c77cf094663b5da8` |

The immutable external namespace is:

```text
/scratch/network/km6349/ge1_c7_runs/ge1-c7-4bfde4c726a585433ea4bb60e6ce9d245ae7c87d-3344505
```

The artifact manifest enumerates 102 entries: 100 epoch checkpoints plus
`metrics.jsonl` and `resolved_config.json`. `SHA256SUMS` adds the manifest,
and the published directory therefore contains 104 regular files.

## Related records

- [ADR-0006](../decisions/ADR-0006-ge1-c7-sufficiency-execution-contract.md)
- [Stage 0 preregistration](../specifications/ge1_stage0_preregistration.md)
- [C6 measurement contract](../specifications/ge1_c6_measurement_contract.md)
- [C5/C6 review-fix validation](ge1_c5_c6_review_fix_cpu_validation.md)
- [GE1 implementation plan](../specifications/graph_encoder_implementation_plan.md)
- [Current project status](../status.md)
