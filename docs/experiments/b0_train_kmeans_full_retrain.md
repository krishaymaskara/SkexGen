# B0 Full Train-K-Means Retraining

## Record status

| Item | Value |
|---|---|
| Experiment ID | `b0-train-kmeans-full-3326757` |
| Record status | `verified` |
| Scientific role | Full epoch-boundary continuation of accepted pilot |
| Source commit | `d549ebe4478e166fda8d54f2b173572a1ab52e7a` |
| Working tree | Clean |
| Slurm job | `3326757` |
| Execution date | July 27, 2026 |
| Scheduler result | `COMPLETED`, exit `0:0`, 5:34 on `adroit-h11n2` |

## Question and predetermined gate

The run asked whether the accepted train-k-means treatment remained above the
collapse thresholds through the original authoritative 50-epoch budget.

Two consecutive completed training epochs with fewer than two active codes or
perplexity below 2.0 would stop the run. `PASS_FOR_EVALUATION` additionally
required all 50 epochs, finite values, valid selected-checkpoint provenance,
selected validation active-code count at least two, selected validation
perplexity at least 2.0, and no test evaluation.

## Resume and treatment authority

Training resumed exactly at the pilot's epoch-5 boundary using checkpoint
SHA-256
`2d5cb89eac7e0bf0d66315b6e9be8cc17eb5170ab9ade1919f8563dc9cce7c08`.
It restored the model and VQ EMA state, optimizer, Python and Torch RNG,
completed epoch/global step, and treatment provenance. K-means was not rerun.
There was no learning-rate scheduler.

The authoritative epoch budget was reconciled as 50 from the original
checkpoint and run metadata. Source-tree SHA-256:
`bbe447b8d192227d3f26d43e024fba9ffae4d4b13d724ee54328b71b146da8ef`.

## Inputs and partitions

| Item | Verified value |
|---|---|
| Train / validation / test families | 544 / 68 / 68 |
| K-means initialization partition | Train |
| Validation families used for initialization | 0 |
| Test families used for initialization | 0 |
| Test partition evaluated | `false` |

Training used CPU, Python 3.8.13, PyTorch 1.11.0, batch size 32, learning rate
0.001, weight decay 0.0001, and seed 2026.

## Verified results

The run completed epoch 50 with zero collapse-threshold violations and a
maximum violation streak of zero.

| Epoch range | Train usage | Validation usage |
|---|---|---|
| Pilot epoch 5 | 5 codes, perplexity 3.3982 | 4 codes, perplexity 3.6104 |
| Epoch 46 | 7 codes, perplexity 3.5175 | 6 codes, perplexity 3.2638 |
| Epoch 47 | 7 codes, perplexity 3.5276 | 6 codes, perplexity 3.2638 |
| Epoch 48 | 7 codes, perplexity 3.5453 | 6 codes, perplexity 3.2638 |
| Epoch 49 | 7 codes, perplexity 3.5261 | 6 codes, perplexity 3.2638 |
| Epoch 50 | 7 codes, perplexity 3.8550 | 6 codes, perplexity 3.2638 |

The predetermined validation-loss criterion selected:

| Selected-checkpoint item | Verified value |
|---|---|
| Checkpoint | `best.pt` |
| Epoch / global step | 44 / 748 |
| SHA-256 | `282988af00a2dc9a53e14ceb270537d35f85d5339dc989634f88af5f77931267` |
| Validation active codes | 5 |
| Validation perplexity | 3.1204100899 |
| Provenance valid | `true` |
| Required values finite | `true` |

The decision was `PASS_FOR_EVALUATION`, with reason
`full_training_and_selected_validation_gates_passed`.

## Verification

The workflow passed 33 focused full-retraining tests, 300 broader
flat-baseline tests, and 43 model-data tests. Pilot provenance, epoch-budget
reconciliation, and the full artifact validator passed.

## Interpretation and limitations

Train-only k-means initialization prevented a recurrence of total assignment
collapse for this seed and corpus. The model exhibits stable codebook
under-utilization rather than one-code collapse.

`PASS_FOR_EVALUATION` is not final scientific acceptance. The selected
epoch-44 checkpoint still requires the same validation-only paired
reconstruction protocol used on the original checkpoint. Code semantics,
OpenCascade execution, and localized editing remain untested.

## Artifacts and integrity

| Artifact | SHA-256 |
|---|---|
| `best.pt` | `282988af00a2dc9a53e14ceb270537d35f85d5339dc989634f88af5f77931267` |
| `last.pt` | `9e2f416e5262990dc6c86095f9cb94b8e88b5a65c828392cde24858dd9ac6ec7` |
| Epoch metrics | `e81b329eb1311646f7c65e9c8e6c43bba0086b7014a2970381a7d909f29ce3ee` |
| Decision report | `88df383675d4a0196e270c1afaa300a445acbcf9502c692cd2dc21bf142577a8` |
| Frozen archive | `9eaeeb91728d0e2c8e8a7de77a0ac837e92a9c3fe7c704c6011ee26fbbae0498` |

The frozen archive resides at
`/Users/krishaymaskara/research/audited-runs/skexgen-flat-evidence-20260728.tar.gz`
and passed its complete internal checksum manifest.

## Related records

- Intervention pilot: [Train-k-means pilot](b0_train_kmeans_pilot.md)
- Original failure: [Original B0 collapse](b0_original_training_collapse.md)
- Next gate: repeat [Phase B validation evaluation](b0_phase_b_validation_evaluation.md)
  with this selected checkpoint under a new immutable run.
