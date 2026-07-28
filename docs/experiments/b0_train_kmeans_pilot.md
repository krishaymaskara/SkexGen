# B0 Train-K-Means Pilot

## Record status

| Item | Value |
|---|---|
| Experiment ID | `b0-train-kmeans-pilot-3326278` |
| Record status | `verified` |
| Scientific role | Bounded anti-collapse intervention |
| Source commit | `856d6a62187f3d7f4b5cc7a9b4c9efc655b74ce1` |
| Working tree | Clean |
| Slurm job | `3326278` |
| Execution date | July 27, 2026 |
| Scheduler result | `COMPLETED`, exit `0:0`, 1:49 on `adroit-h11n2` |

## Question and predetermined gate

The pilot asked whether deterministic train-only k-means initialization could
prevent immediate one-code assignment collapse without simultaneously
changing the model, optimizer, losses, codebook size, or training data.

The epoch-2 and epoch-5 gates required at least two active codes and
perplexity at least 2.0. Passing all pilot gates permitted promotion to the
full 50-epoch continuation.

## Treatment isolation

The treatment collected 1,088 distinct prequant vectors—two latent positions
for each of 544 training families—and ran seeded k-means++ with eight fixed
Lloyd iterations for 32 centers.

| Initialization property | Verified value |
|---|---|
| Partition | Train only |
| Seed | 2026 |
| Validation families used | 0 |
| Test families used | 0 |
| Finite initialization | `true` |
| EMA pseudo-count total | 32.0 |
| Normal initialization count total | 32.0 |
| Effective mass ratio | 1.0 |
| Center SHA-256 | `fbe0b7fa3d54c608a1bdc963f88a98a3641e5c74a52fddcfeec8e43ea3de855d` |

Matching the pseudo-count total to the normal codebook's initial mass avoids
confounding data-informed centers with increased EMA inertia.

## Configuration and environment

The model and optimizer configuration matched the original run. The only
scientific treatment was `vq_init=train-kmeans`; execution moved to CPU.
Training used Python 3.8.13, PyTorch 1.11.0, seed 2026, batch size 32,
learning rate 0.001, and five epochs.

The baseline checkpoint SHA-256 was
`2732899de6d60d407d26a7cf176e7a7f4534580aa99ca973df6e9b740330222b`.
Source-tree SHA-256:
`19f85f3606397538f5e1e53f6ac5d66c8fd29c5ee8010e314da9da812b63a41c`.

## Verified results

| Epoch | Train active / perplexity | Validation active / perplexity |
|---:|---:|---:|
| 1 | 27 / 2.0872 | 3 / 2.1734 |
| 2 | 12 / 3.4962 | 5 / 2.9720 |
| 3 | 8 / 3.5813 | 4 / 3.6062 |
| 4 | 4 / 3.6572 | 4 / 3.3877 |
| 5 | 5 / 3.3982 | 4 / 3.6104 |

The run completed epoch 5 with decision `PASS_FOR_FULL_RETRAIN` and reason
`all_pilot_gates_passed`. The test partition was not evaluated.

## Verification

The workflow passed 19 focused tests, 283 broader flat-baseline tests, and 43
model-data tests. The initialization smoke and artifact validator both
passed.

## Interpretation and limitations

The treatment prevented the immediate total one-code collapse under the
predetermined five-epoch gate. The falling count of touched codes does not
show renewed total collapse: perplexity increased and remained above 2.0,
indicating several effective assignment regions rather than one.

This result justified continuing the same treatment. It did not establish
that four or five effective codes were enough for accurate or executable CAD
reconstruction.

## Artifacts and integrity

| Artifact | SHA-256 |
|---|---|
| `best.pt` | `3fa6bbe3437b03fd529795f1812c6151dfa373a894c2a1c11d1667d1feb2b672` |
| `last.pt` / full-run resume checkpoint | `2d5cb89eac7e0bf0d66315b6e9be8cc17eb5170ab9ade1919f8563dc9cce7c08` |
| Epoch metrics | `73061c4b6667d6236dcd68fdaf56332a7c71d40d0b0ff4d2ad4f39f111353978` |
| Frozen archive | `9eaeeb91728d0e2c8e8a7de77a0ac837e92a9c3fe7c704c6011ee26fbbae0498` |

## Related records

- Motivation: [VQ-collapse diagnosis](b0_vq_collapse_diagnosis.md)
- Continuation: [Full train-k-means retraining](b0_train_kmeans_full_retrain.md)
