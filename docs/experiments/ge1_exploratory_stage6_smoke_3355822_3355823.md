# GE1 Exploratory Stage 6 Two-Epoch Lifecycle Smoke, Jobs 3355822 and 3355823

## Disposition

The mandatory two-epoch full-lifecycle smoke **completed and finalized
cleanly** at exact commit
`42d5c489c7236031ff18dfd13ca88af428ca7510`. The producer and final artifacts
are explicitly exploratory, smoke-only, result-ineligible, and inconclusive.

| Item | Producer | Finalizer |
|---|---|---|
| Job | `3355822` | `3355823` |
| State / exit | `COMPLETED` / `0:0` | `COMPLETED` / `0:0` |
| Elapsed / batch MaxRSS | `00:11:53` / `1020096K` | `00:03:48` / `1074928K` |
| Device | CPU selected only by timing | CPU, tensor-independent |
| Runner SHA-256 | `7e91006b1dd8991ecf87595fd5a9ca1622ab2240b7c780c2efeb95c5a161ce5f` | `1fefc973b1a016387a3ee5d65247e6b2c7c0761e16b026313b3bd9a180de2fbc` |
| Focused tests | 50/50, zero skips | 27/27, zero skips |
| Complete discovery | 743/743, exact six CUDA skips | 743/743, exact six CUDA skips |

The producer used ordinal magnitude, autonomous stopping, seeds
`2026/2027/2028`, `exploratory_development_access=true`,
`smoke_protocol=true`, and `protocol_final_epoch=2`. All six arms trained to
epoch 2, strict recovery succeeded, train and development autonomous scoring
completed, six wrappers were bundled, 8,136 family-condition records were
verified, and the finalizer consumed the producer artifact without corpus or
checkpoint access.

The two-epoch optimization gate was expectedly false: all losses and gradients
were finite, but a two-epoch smoke cannot satisfy the epoch-200 plateau or
fixed-checkpoint predicates. The train structural-memory gate passed, the
exploratory continuation was recorded, and development scoring proceeded.
Final ordinary validity was false; the immutable artifact marker independently
forces result ineligibility and an inconclusive interpretation.

The smoke-only paired development effect was `0.04074074074074074`, below the
`0.10` threshold. This number is lifecycle evidence only, not a Stage 6 result
or a prediction of the held epoch-200 run.

## Finalized artifacts

Producer artifact:

`/scratch/network/km6349/ge1_stage6_smoke-42d5c48/ge1-stage6-structure-only-producer-42d5c489c7236031ff18dfd13ca88af428ca7510-3355822`

Checkpoint bundle:

`/scratch/network/km6349/ge1_stage6_smoke-42d5c48/ge1-stage6-structure-only-checkpoints-42d5c489c7236031ff18dfd13ca88af428ca7510-3355822`

Final smoke artifact:

`/scratch/network/km6349/ge1_stage6_smoke_final-42d5c48/ge1-stage6-structure-only-42d5c489c7236031ff18dfd13ca88af428ca7510-3355823`

| Artifact | `SHA256SUMS` SHA-256 |
|---|---|
| Producer | `093c1b971807938c16182716fde1de9b398c5b1734259edbbd748fee14f3eee0` |
| Checkpoint bundle | `11a5c87e7c79e235cbdcc3ca336bb087bf17e633b90e27052d608e6ea4bb6100` |
| Final smoke artifact | `a880e8895b916d33a3bfd891244c845bd4442c0e865e61063880e2feb5c673c6` |

The final artifact is 52MB and contains exactly five verified files:

| File | Bytes | SHA-256 |
|---|---:|---|
| `artifact_manifest.json` | 538 | `6e930745c9324d7ec4636c3d58c8195d5ff86a13e065295352479f9592d6bf3e` |
| `resolved_config.json` | 872,365 | `75ecf92d1ffabea985cef4f0188eabed2c21085b311cd2594bc16b472fe8bed3` |
| `family_metrics.jsonl` | 52,957,089 | `757b9c56c5823fd73637068f8d9e3e69d33ae3afb36aa5eb0087e364e4eddc3a` |
| `summary.json` | 35,455 | `f7820439a57ae812809b2428921a5f24e4b7e3d5ea864db27cb592e6153d3d3d` |
| `SHA256SUMS` | 342 | `a880e8895b916d33a3bfd891244c845bd4442c0e865e61063880e2feb5c673c6` |

All manifest and checksum validations passed in both jobs. RR, ER, IID,
history-depth, geometry-extrapolation, CAD, Stage 7, and C8 remained closed.

## Prepared epoch-200 command — not submitted

The timing-selected future command is preserved below for reviewer inspection.
It intentionally omits `SMOKE_EPOCHS`, uses the CPU runner and all three seeds,
and requests four hours. Before any separately authorized submission, the
verified timing JSON must be copied byte-for-byte into the new `RUN_PARENT`.

```bash
# NOT SUBMITTED — EXPLICIT DESIGNATED-REVIEWER HOLD
sbatch --job-name=ge1-stage6-exploratory \
  --partition=all --nodes=1 --ntasks=1 --cpus-per-task=1 \
  --mem=24G --time=04:00:00 \
  --output=/scratch/network/km6349/ge1_stage6_producer_runs/exploratory-42d5c48-%j.out \
  --error=/scratch/network/km6349/ge1_stage6_producer_runs/exploratory-42d5c48-%j.err \
  --export=ALL,EXPECTED_COMMIT=42d5c489c7236031ff18dfd13ca88af428ca7510,EXPECTED_RUNNER_SHA256=7e91006b1dd8991ecf87595fd5a9ca1622ab2240b7c780c2efeb95c5a161ce5f,REPOSITORY=/scratch/network/km6349/SkexGen-exploratory-stage6,TRAIN_INPUT_ROOT=/scratch/network/km6349/ge1_stage6_narrow_inputs-3a674705/stage6-train,DEVELOPMENT_INPUT_ROOT=/scratch/network/km6349/ge1_stage6_narrow_inputs-3a674705/stage6-development,RUN_PARENT=/scratch/network/km6349/ge1_stage6_exploratory_real-42d5c48,TIMING_EVIDENCE=/scratch/network/km6349/ge1_stage6_exploratory_real-42d5c48/timing.json,NODE_GENERATION_IDENTITY=GE1-PAD-TERMINATED-UNCONSTRAINED-NODES-v1,EXPLORATORY_DEVELOPMENT_ACCESS=true \
  prototype/graph_encoder/adroit/ge1_stage6_structure_only_producer.slurm
```

No epoch-200 producer job was submitted. Completing this smoke does not lift
the reviewer hold.
