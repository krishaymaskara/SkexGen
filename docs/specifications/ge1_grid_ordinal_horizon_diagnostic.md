# GE1 grid-ordinal extended-horizon diagnostic

| Item | Frozen value |
|---|---|
| Reviewer decision | Krishay Maskara, August 18, 2026 |
| Diagnostic | `GE1-GRID-ORDINAL-HORIZON-DIAGNOSTIC-v1` |
| Artifact | `GE1-GRID-ORDINAL-HORIZON-ARTIFACT-v1` |
| Nature | corpus-free, checkpoint-free, engineering-only |
| Status | implemented for preparation; untransferred and unsubmitted |

## Question and boundary

The diagnostic asks whether classes 2–4 remain below their target only because
200 updates are too short, or whether their upper ordinal cuts remain stuck
through 2,000 updates. It uses the actual unchanged repository head and loss
on generated fixed states. It cannot establish behavior of job `3352404`'s
checkpoints, a full decoder or encoder, a corpus, or scientific training. It
does not identify an exact defect or authorize a source change.

The observed outcome never controls artifact validity. Completion requires
only the frozen execution and a complete valid artifact.

## Frozen design

Each of the six conditions independently resets seed 2026 and constructs a
fresh width-32 real grid-magnitude head. Conditions are extrude and revolve
crossed with target classes 2, 3, and 4. Raw gaps use the unchanged `0.0`
source initialization. No head or optimizer state crosses conditions.

The fixed generated decoder state has shape `[8,1,32]`, first component `1.0`,
remaining components `0.0`, and no gradient or optimizer state. Only head
parameters are trainable. The real node IDs, channels 37/38, geometry and node
masks, normalized grids, and grid-v2 configuration construct every target.

AdamW uses learning rate `0.001`, weight decay `0.0`, and global gradient clip
norm `1.0`. Every condition executes exactly 2,000 optimizer updates and
records initialization plus every post-update state, producing 2,001 records
per condition and 12,006 records total.

## Records and summaries

Each record contains loss, logits, sigmoid probabilities, decoded class and
grid values, projection score, first bias, raw gaps, ordered biases, correctness,
and minimum margin. Projection, first-bias, and raw-gap gradients are recorded
individually before and after clipping, along with both global norms and a
clipping flag. AdamW step, `exp_avg`, and `exp_avg_sq` are recorded for the
active projection, first bias, and raw gaps.

Condition summaries include initial/final class, transitions, first correct
step, the first positive step for each cut, whether every upper cut crossed
zero, values at steps 200/500/1,000/2,000, and logit/loss changes over
200–500, 500–1,000, and 1,000–2,000.

## Artifact and runner

Atomic finalization produces exactly:

1. `resolved_config.json`;
2. `trajectory.jsonl`;
3. `summary.json`;
4. `artifact_manifest.json`;
5. `SHA256SUMS`.

`prototype/graph_encoder/adroit/ge1_grid_ordinal_horizon_cpu.slurm` requires a
clean standalone detached exact-commit checkout, Python 3.8.13, PyTorch 1.11.0,
one CPU/thread, 2 GB, and at most 15 minutes. Its only binds are the repository
read-only and the artifact parent read-write. It runs the diagnostic exactly
once after zero-skip validation and never submits itself.

The prepared remote parents are:

```text
/scratch/network/km6349/ge1_grid_ordinal_horizon_runs
/scratch/network/km6349/ge1_grid_ordinal_horizon_artifacts
```

No corpus, manifest, payload, checkpoint, model/repaired artifact, CAD kernel,
protected partition, scientific execution, repair, Stage 6, or C8 is permitted.
No outcome authorizes another diagnostic or job.
