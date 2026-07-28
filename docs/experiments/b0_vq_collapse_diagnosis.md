# B0 VQ-Collapse Diagnosis

## Record status

| Item | Value |
|---|---|
| Experiment ID | `b0-vq-diagnosis-3325296` |
| Record status | `verified` |
| Scientific role | Root-cause diagnosis of the original one-code collapse |
| Source commit | `55b2841d17304abfff793de6c2cece13205bce3c` |
| Working tree | Clean |
| Slurm job | `3325296` |
| Execution date | July 27, 2026 |
| Scheduler result | `COMPLETED`, exit `0:0`, 2:03 on `adroit-h11n3` |

## Question

The diagnosis distinguished among encoder collapse, numerically collapsed
codebook vectors, gradient-flow failure, decoder latent indifference,
loss/mask imbalance, and nearest-code assignment collapse. It was explicitly
a diagnostic intervention, not a benchmark evaluation.

## Inputs and partition authority

| Item | Verified value |
|---|---|
| Original `best.pt` SHA-256 | `2732899de6d60d407d26a7cf176e7a7f4534580aa99ca973df6e9b740330222b` |
| Original `last.pt` SHA-256 | `aa99885f4120d634ef5990eee11fa4dfe1ab40704fd74ed1c0a527b8d7a145e9` |
| Corpus | 544 train / 68 validation / 68 test |
| Checkpoint partition IDs | Matched the authoritative corpus |
| Diagnostic families | Two selected physical families |
| Test partition evaluated | `false` |

The run used source-tree SHA-256
`57b0f200ddd2c02564b12f8fbcd3d31f186c90930136cd534fe840f76211c9e9`.

## Verification

The workflow ran 259 general tests, 43 model-data tests, and 34 focused
diagnosis tests. Both diagnostic smoke runs were byte-identical, the full
diagnosis passed its artifact verifier, and the final report marked
`full_diagnosis_verified=true`.

## Root-cause findings and interpretation

The directly supported conclusions are:

- encoder outputs before quantization were varied rather than constant;
- codebook embeddings remained distinguishable;
- finite nonzero gradients reached the encoder;
- the decoder changed materially when latent codes changed;
- the EMA codebook is a buffer, so optimizer gradient to it is not applicable;
- varied prequant vectors nevertheless fell in code 17's nearest-neighbor
  region;
- collapse was visible by epoch 1 validation and persisted from epoch 2;
- categorical-head imbalance and sparse geometry applicability were
  contributing evidence, not proof of a single exclusive cause.

The report classifies the supported mechanisms as:

1. nearest-code assignment collapse;
2. loss/mask imbalance;
3. training-time collapse.

The proximate failure is nearest-code assignment collapse, not constant
encoder output or numerically identical codebook entries.

## Resulting retraining gate

The diagnosis prescribed a bounded train-only, seed-controlled intervention.
The resulting minimum gates were:

- at least two active codes;
- perplexity at least 2.0;
- at least two distinct prequant vectors;
- material decoder sensitivity;
- finite nonzero encoder gradients;
- no test-partition evaluation.

The diagnosis itself did not implement or validate a repair.

## Limitations

- Only two families were selected for expensive diagnostic probes.
- Loss imbalance is contributory evidence; the workflow does not isolate it
  as a causal treatment.
- The diagnosis does not establish reconstruction quality or latent semantics.

## Artifacts and integrity

The frozen archive retains two smoke directories, the full diagnosis, final
report, logs, per-file artifact manifest, and scheduler evidence.

| Artifact | SHA-256 |
|---|---|
| Full root-cause report | `d69b199c33c1aa698ab663eda10088580380ca4b5f077d658d81d81a539a5c33` |
| Full run metadata | `aafee186cb31573ca977d3d6d54f909d4314be05386069ef8f49521e8210a073` |
| Full latent-usage table | `2d9c54efc0d8b6c46b238505c36fedbba63981a96f3bd4bfdf0d714b25d44183` |
| Frozen archive | `9eaeeb91728d0e2c8e8a7de77a0ac837e92a9c3fe7c704c6011ee26fbbae0498` |

## Related records

- Diagnosed run: [Original B0 collapse](b0_original_training_collapse.md)
- Preceding evaluation: [Phase B validation](b0_phase_b_validation_evaluation.md)
- Resulting intervention: [Train-k-means pilot](b0_train_kmeans_pilot.md)
