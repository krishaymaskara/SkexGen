# B0 Phase B Paired Validation Evaluation

## Record status

| Item | Value |
|---|---|
| Experiment ID | `b0-phase-b-validation-3324856` |
| Record status | `verified` |
| Scientific role | Paired teacher-forced/predicted-history evaluation |
| Source commit | `42183a47ec8fa70e6109fbea33ecda9e5e84d071` |
| Working tree | Clean |
| Slurm job | `3324856` |
| Execution date | July 27, 2026 |
| Scheduler result | `COMPLETED`, exit `0:0`, 1:51 on `adroit-h11n3` |

## Question and evaluation contract

The evaluation measured the original selected checkpoint on all 68 IID
validation families under two decoding paths:

- teacher-forced prefix feedback;
- predicted-history raw-argmax prefix feedback with derived geometry masks.

Both paths used the same checkpoint and selected validation set. The target
canonical node count was supplied, so this run was length-conditioned rather
than autonomous template classification.

## Inputs and partition authority

| Item | Verified value |
|---|---|
| Checkpoint | Original B0 `best.pt`, epoch 50, step 850 |
| Checkpoint SHA-256 | `2732899de6d60d407d26a7cf176e7a7f4534580aa99ca973df6e9b740330222b` |
| Partition | IID validation |
| Validation families | 68 |
| Validation-ID SHA-256 | `31b6607ca360bf7f3ed2334ca798f3aa07dd70214192a13324b71439d56c0681` |
| Test partition evaluated | `false` |
| Corpus configuration SHA-256 | `57f11d3024b97c80eb6052d3517a097a88dc136d0befa103f2dc47fb62516d32` |
| Split-manifest SHA-256 | `8d6caf0d5584c818e90c9e5d877ab9dd00519cb343f7fabe90a6459802914ec3` |

The authoritative loader succeeded. The run used CPU, Python 3.8.13, and
PyTorch 1.11.0. Source-tree SHA-256:
`81e33360cbb7d3aca31b12da835b9a934135170d75ba86775f2b3453d277f862`.

## Validation of the evaluator

- Two smoke evaluations exited zero.
- All six smoke artifacts were byte-identical between repetitions.
- The full evaluation exited zero.
- Regression suites passed: 55, 14, 29, 49, 43, and 225 tests.
- All published full artifacts match the hashes in the final report.

## Verified reconstruction results

Both paths produced complete, raw-integrity-valid, reconstruction-target
records for all 68 families, but neither produced a controlled-domain-valid
history.

| Metric | Teacher-forced | Predicted history |
|---|---:|---:|
| Raw completion rate | 1.0000 | 1.0000 |
| Raw integrity validity | 1.0000 | 1.0000 |
| Reconstruction-target validity | 1.0000 | 1.0000 |
| Controlled-domain validity | 0.0000 | 0.0000 |
| Node-type token accuracy | 0.8705 | 0.6093 |
| Exact node-type sequence | 0.3382 | 0.3382 |
| Exact complete ten-field match | 0.0000 | 0.0000 |
| Geometry MAE | 3.3377 | 3.4623 |
| Geometry RMSE | 20.4508 | 23.7956 |
| Exact operation-type sequence | 0.3382 | 0.3382 |
| Pointer accuracy | 1.0000 | 0.3772 |
| Edge micro-F1 | 0.8651 | 0.5616 |
| Exact typed-edge set | 0.0000 | 0.0000 |

Primary controlled-domain failures included invalid node grammar and
unsupported profile patterns. The checkpoint assigned all 136 latent tokens
to code 17: one active code out of 32, utilization 0.03125.

## Decision and interpretation

The evaluator passed its engineering and reproducibility gate. The evaluated
checkpoint failed the reconstruction-quality gate. Predicted-history feedback
substantially degraded node, pointer, and edge metrics, but teacher forcing
did not produce any controlled-domain-valid reconstruction either. The
one-code latent state is therefore a more fundamental failure than prefix
feedback alone.

## Limitations and unsupported claims

- Target node count reveals substantial template information.
- This run did not execute predicted histories through OpenCascade.
- It evaluated the original collapsed checkpoint, not the repaired epoch-44
  checkpoint.
- The held-out test partition remained untouched.

## Artifacts and integrity

The frozen archive contains both smoke runs, the full evaluation, raw
predictions, per-example records, metrics, conversion failures, metadata,
logs, and the final report.

| Artifact | SHA-256 |
|---|---|
| Full `summary.json` | `c01888c198df92ab050a80af3bc1166316ac81966479e99bd1211e90eba38b08` |
| Full `metrics.csv` | `8394be84fabe85707b49add7f725d05d69f61fb59c64b823841cda4f81da7dd9` |
| Full `raw_predictions.jsonl` | `614da3e879142b1d257ebfe807279a23d4d06b818c58e89f297c5a6d658890f4` |
| Frozen archive | `9eaeeb91728d0e2c8e8a7de77a0ac837e92a9c3fe7c704c6011ee26fbbae0498` |

## Related records

- Model run: [Original B0 collapse](b0_original_training_collapse.md)
- Compatibility prerequisite: [Phase A](b0_phase_a_contract_validation.md)
- Resulting diagnosis: [VQ-collapse diagnosis](b0_vq_collapse_diagnosis.md)
