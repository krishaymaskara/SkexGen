# GE1 C7 Optimization-Sufficiency Diagnostic Contract

Status: accepted additive post-C7 contract; implementation exists; no
diagnostic run or result exists yet.

Authority: accepted
[ADR-0007](../decisions/ADR-0007-ge1-c7-optimization-sufficiency-diagnostic.md),
reviewed by Krishay Maskara on August 9, 2026.

## Scientific boundary

Formal C7 job `3344505` remains a completed scientific failure under its
frozen 50-epoch contract. This diagnostic does not rerun, overwrite, relabel,
or reinterpret that result. It asks only whether the two arms reach autonomous
exact sufficiency on the same four train families under a predetermined
500-update clean-start trajectory.

The diagnostic does not change ADR-0004, ADR-0005, ADR-0006, or
`GE1-STAGE0-PREREG-v1`; authorize Stage 6; invoke decoder repair; implement
C8; or alter any protected-access rule.

Its identities are:

```text
GE1-C7-OPTIMIZATION-SUFFICIENCY-DIAGNOSTIC-v1
GE1-C7-OPTIMIZATION-TRAJECTORY-v1
GE1-C7-OPTIMIZATION-ARTIFACT-v1
```

## Frozen inputs and lifecycle

The existing metadata-only C7 selector must reproduce the same exact `E`,
`R`, `EE`, and `RE` train family IDs and selected-set SHA-256
`c07e90675099cb2ea58575fe91683dc80b70cf3fa85fa6fa8de0c44dbe9783d6`.
Only those four payloads may then be opened. Every other train family,
development, RR, ER, IID, history-depth, and geometry-extrapolation remains
closed.

Both arms use seed 2026 and one fresh matched construction. Shared components
begin byte-identically; all mutable parameters and buffers are disjoint. No
formal C7 checkpoint, cross-arm warm start, or full-comparison checkpoint role
is permitted.

## Training and measurement

The unchanged recipe is AdamW, learning rate `1e-3`, weight decay `0`, global
gradient clipping `1.0`, and batch size 8. Four effective examples are
presented per optimizer update.

| Quantity | Frozen value |
|---|---:|
| Maximum optimizer updates per arm | 500 |
| Family presentations per arm | 2,000 |
| Autonomous milestones | 50, 100, 200, 500 |
| Recovery interval | Every 25 updates |
| Retained checkpoints per arm | 20 |

Both arms always train continuously through update 500. No loss, plateau,
autonomous output, or intermediate exact result can stop, extend, or select
training. A milestone checkpoint is written before measurement and strictly
reloaded into a newly constructed model. The frozen plateau formula remains
diagnostic only.

At every milestone, target-free autonomous `P_true`, `P_shuffle`, and
`P_mean` generation precedes scoring. Each event records all C6 per-family
structural, conversion, validity, prefix, dependency, attachment, failure,
and geometry metrics, all five primary reporting values, donor agreement, and
memory-alteration evidence.

Only autonomous `P_true` can establish exact sufficiency. Every family must
have exact unrounded `1.0` for node sequence, graph, strict conversion, and
complete executable validity; `EE` and `RE` additionally require exact
`depends_on`.

## Interpretation and non-authorization

After update 500 the artifact selects exactly one preregistered category:

- `undertraining_supported_both_arms`;
- `undertraining_or_optimization_difference_supported_one_arm`;
- `still_improving_at_update_500`;
- `plateaued_without_exact_sufficiency`; or
- `infrastructure_failure`.

A completed negative diagnostic is a valid zero-exit result. Infrastructure
failure does not finalize an artifact and exits nonzero. Every completed
record states:

```text
original_c7_result_changed=false
stage6_authorized=false
decoder_repair_invoked=false
c8_or_later_performed=false
```

The diagnostic never emits the formal C7 fields `overall_gate_pass`,
`stage6_authorized_by_c7`, or `preauthorized_decoder_repair_triggered`.

## Slurm and artifact contract

The runner explicitly forwards `SLURM_JOB_ID` through `apptainer --cleanenv`.
The value must be a nonempty decimal string and agree in resolved
configuration, run metadata, every checkpoint provenance payload and event,
the terminal event, and final verification.

Publication targets a new external non-symlink directory, stages beside it,
and atomically renames only after validation. The artifact contains
`resolved_config.json`, `metrics.jsonl`, 40 checkpoints,
`artifact_manifest.json`, and `SHA256SUMS`. JSON is canonical and finite;
paths are safe and relative; sizes and SHA-256 hashes are complete; checksums
are sorted with final LF; and mutation is detected.

## Implementation and runner

| Role | Location |
|---|---|
| Diagnostic runtime | `prototype/graph_encoder/optimization_diagnostic.py` |
| Additive sparse-checkpoint hook | `prototype/graph_encoder/training.py` |
| Pure/static/artifact tests | `prototype/graph_encoder/tests/test_optimization_diagnostic_contract.py` |
| Real-PyTorch tests | `prototype/graph_encoder/tests/test_optimization_diagnostic_runtime.py` |
| Adroit CPU runner | `prototype/graph_encoder/adroit/ge1_optimization_diagnostic_cpu.slurm` |

The runner must be submitted only from a clean exact checkout of the
implementation commit. Its only diagnostic CLI arguments are corpus path,
new output path, repository root, and expected commit. Before corpus access it
runs the focused suite with zero skips, the complete graph-encoder suite with
zero skips, relevant regressions, documentation validation, compileall,
Python 3.8 grammar, import/export, access/source audit, Bash syntax, diff, exact
commit, and clean-tree gates.

Example submission shape (the final handoff supplies the exact commit):

```bash
export EXPECTED_COMMIT="$(git rev-parse HEAD)"
sbatch --export=ALL,EXPECTED_COMMIT="$EXPECTED_COMMIT" \
  prototype/graph_encoder/adroit/ge1_optimization_diagnostic_cpu.slurm
```

Implementation completion is not a diagnostic result. No local corpus access,
diagnostic execution, Slurm submission, protected access, Stage 6
authorization, repair invocation, or C8 work is part of this contract record.
