# GE1 grid-ordinal trajectory diagnostic

| Item | Frozen value |
|---|---|
| Reviewer decision | Krishay Maskara, August 17, 2026 |
| Diagnostic | `GE1-GRID-ORDINAL-TRAJECTORY-DIAGNOSTIC-v1` |
| Artifact | `GE1-GRID-ORDINAL-TRAJECTORY-ARTIFACT-v1` |
| Nature | corpus-free, checkpoint-free, engineering-only |
| Authorization | exactly one separately submitted exact-commit Adroit job |
| Scientific execution / repair | false / unauthorized |

## Question and interpretation boundary

The diagnostic asks whether the repository's actual rank-consistent ordinal
head, actual `grid_magnitude_terms` loss, and AdamW optimizer exhibit the
proposed class-1/class-3 trajectory over 200 updates on generated fixed decoder
states. It does not reproduce or inspect job `3352404`; establish full-model,
encoder, corpus, or scientific behavior; identify an exact defect; or authorize
an initialization change, scientific rerun, repair, Stage 6, or C8.

The observed trajectory never controls artifact validity. A result that
supports, mixes, or falsifies the hypothesis finalizes successfully if its
execution and five-file artifact are complete and valid.

## Reused implementation and frozen conditions

The diagnostic imports and calls, without reimplementation:

- `prototype.graph_encoder.grid_magnitude.build_grid_magnitude_head`;
- `prototype.graph_encoder.losses.grid_magnitude_terms`;
- the current frozen grids, node-type IDs, channels, and grid-v2 configuration;
- PyTorch AdamW and gradient clipping.

Every condition independently resets seed 2026 and constructs a fresh width-32
head. The generated decoder state is `[8,1,32]`, has first component `1.0` and
all other components `0.0`, and has neither gradient nor optimizer state. Only
head parameters are trainable. AdamW uses learning rate `0.001`, weight decay
`0.0`, and gradient clipping norm `1.0` for exactly 200 updates.

The complete 20-condition matrix is both operation types times all five target
classes times raw-gap initializations `0.0` and `0.5`. Classes 1 and 3 under
both initializations are primary; classes 0, 2, and 4 are controls. The `0.5`
gap is applied only to the new in-memory head and does not modify production
initialization.

Targets use the real operation node-type IDs and masks, extrusion channel 37,
revolve channel 38, and the frozen normalized grids:

- extrusion: `[0.125, 0.25, 0.375, 0.5, 0.75]`;
- revolve: `[0.125, 0.25, 0.5, 0.75, 1.0]`.

## Trajectory and artifact

Each condition records step 0 and every post-update state through step 200,
for exactly 201 records per condition and 4,020 canonical JSONL records total.
Records include loss, projection score, first bias, three raw gaps, four
ordered biases/logits/probabilities, decoded class and values, minimum margin,
clipped parameter-group gradient norms, and correctness. Condition summaries
include initial/final class, first correct step, persistence, transitions,
final tensors/margins, all-steps trap descriptions, the step-200 hypothesis
evaluation, and the diagnostic-gap intervention predicate.

Atomic finalization produces exactly:

1. `resolved_config.json`;
2. `trajectory.jsonl`;
3. `summary.json`;
4. `artifact_manifest.json`;
5. `SHA256SUMS`.

The artifact records exact commit, job and host, Python/PyTorch/CPU identity,
source cleanliness and detached state, arithmetic and condition matrix, and
false access/authorization fields. It contains no checkpoint.

## Runner and access boundary

`prototype/graph_encoder/adroit/ge1_grid_ordinal_trajectory_cpu.slurm` requires
a clean standalone detached exact-commit checkout, Python 3.8.13, PyTorch
1.11.0, CPU only, one task/thread, 2 GB, and 15 minutes. Its only binds are the
repository read-only and a new artifact parent read-write. No corpus,
manifest, checkpoint, model/repaired artifact, CAD-kernel, or protected path is
declared. It runs focused and existing grid tests, complete graph-encoder
discovery, documentation/compile/AST/syntax/source/preservation checks, the
diagnostic exactly once, and artifact verification. It never submits itself or
invokes a scientific entry point.

The authorized remote parents are:

```text
/scratch/network/km6349/ge1_grid_ordinal_trajectory_runs
/scratch/network/km6349/ge1_grid_ordinal_trajectory_artifacts
```

No outcome automatically authorizes another job, retry, production change,
repair, protected access, Stage 6, or C8.
