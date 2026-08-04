# B0 Initial Graph V1 Pilot

## Record status

| Item | Value |
|---|---|
| Experiment ID | `b0-graph-v1-initial-3341942` |
| Record status | `verified` |
| Evidence state | `verified-local` |
| Scientific role | Initial capacity-matched graph-native structural pilot |
| Branch | `graph-profile-decoder` |
| Source commit | `089b9f3d0e5a61fb19ef3fa05e993fc4eceffdcb` |
| Working tree | `clean` |
| Slurm/local run ID | `3341942` |
| Execution date | August 3, 2026 |
| Frozen tag | `graph-v1-initial-3341942` |

## Question and predetermined decision rule

The pilot tested whether replacing Flat V6's relation and pointer heads with
one explicit directed typed-edge decoder improved autonomous controlled-CAD
validity while preserving the V6 encoder, VQ bottleneck, node decoder,
constrained rollout, profile/plane/axis construction, data, training budget,
conversion, and validity rules.

The protocol allowed one initial pilot and at most one evidence-backed
scientific correction. Primary outcomes were exact graph match, strict graph
validity, and complete CAD validity. Edge accuracy alone was not sufficient.

## Inputs and partition authority

| Item | Value |
|---|---|
| Corpus | Authoritative 680-family controlled corpus |
| Train families used | 544 |
| IID validation families used | 68 |
| Systematic families used | 0 |
| Test families used | 0 |
| Test partition evaluated | `false` |

The IID validation partition was authorized for the pilot. Systematic and
held-out test partitions were rejected by the runner and metadata contracts.

## Configuration

```text
seed: 2026
epochs: 2
batch size: 8
steps: 136
training-example presentations: 1,088
device: CPU
edge vocabulary: <none>, defined_in, depends_on, placed_on, uses_axis,
                 uses_profile
pair decoder: 225 -> 15 -> 6 with GELU
graph decoder parameters: 3,486
total model parameters: 32,856
Flat V6 total parameters: 32,866
difference from Flat V6: -10
```

Only active, non-self, node-type-compatible edge classes were permitted. The
mask never inserted a canonical edge or repaired topology. Loss was
unweighted categorical cross-entropy over all six classes, averaged per
example over active ordered pairs and then over the batch.

## Environment

The authoritative run used the project's Adroit Python 3.8/PyTorch 1.11 CPU
environment. The preserved local archive includes scheduler stdout and
stderr, but not a separate resolved environment file.

## Execution

The committed runner is:

```text
prototype/graph_baseline/run_graph_v1_pilot.py
```

The external immutable output namespace was:

```text
/scratch/network/km6349/constrained_graph_v1_runs/
iid-pilot-089b9f3d0e5a61fb19ef3fa05e993fc4eceffdcb-3341942
```

The job completed two epochs, selected epoch 2, strictly reloaded selected and
final checkpoints, reproduced validation after reload, and passed all 16
focused graph tests then present.

Several earlier readiness jobs failed for engineering reasons and were not
scientific results. Their failure/repair sequence is preserved in the
[Graph V1 engineering-readiness record](b0_graph_v1_engineering_readiness.md).
Job `3339787` was not resumed.

## Verified results

### Program level

| Metric | Result |
|---|---:|
| Complete autonomous validity | 22 / 68 |
| Exact graph match | 22 / 68 |
| Conversion success | 68 / 68 |
| Single-operation validity | 22 / 22 |
| Two-operation validity | 0 / 46 |

| Family | Valid |
|---|---:|
| E | 12 / 12 |
| R | 10 / 10 |
| EE | 0 / 11 |
| ER | 0 / 9 |
| RE | 0 / 10 |
| RR | 0 / 16 |

All 46 two-operation programs failed with `invalid_boolean_sequence`.

### Edge level

| Metric | Result |
|---|---:|
| Positive-edge precision | 65.10% |
| Positive-edge recall | 70.59% |
| Positive-edge F1 | 67.73% |
| Edge-class accuracy | 88.35% |
| True positives | 360 |
| False positives | 193 |
| False negatives | 150 |

There were 553 predicted positive edges and 510 target positive edges. The
autonomous correction count was zero, raw and masked results were identical,
and the structural-violation histogram was empty.

### Scoring-only baselines and representation

| Decoder or prior | Exact graphs |
|---|---:|
| Initial Graph V1 | 22 / 68 |
| Node-type-pair prior | 22 / 68 |
| Edge-type prior | 22 / 68 |
| Position-only prior | 59 / 68 |

The VQ bottleneck used one active code with perplexity `1.0`.

Checkpoint SHA-256 values recorded by the run were:

```text
epoch 1: 037133729f4b27a7dc838218291f32a17333c1b945b9481e18aad93afcf67f37
epoch 2: 2fe26a4d88153a49bb70946c8b91b8c51fe37aca608c2f6c67ef11a7c6936d80
```

## Decision

The graph formulation materially improved complete validity over Flat V6 and
solved every single-operation family, but it solved no two-operation family.
The 59/68 position-only prior supported the one authorized correction: a
direct learned signal for distinguishing repeated instances by directed
serialized position.

## Interpretation

The model learned legal type-level relationships but did not reliably attach
repeated semantic roles to their operation-specific instances. The result
supports explicit graphs over the frozen flat structural heads for simple
programs, not globally consistent multi-operation graph generation.

## Limitations and claims not supported

- One seed and a two-epoch CPU pilot are insufficient for broad claims.
- The model is serialization-aware, not order-independent.
- Systematic and held-out test performance are unknown.
- The collapsed VQ representation may limit program-specific information,
  but its causal effect was not isolated.
- The scoring-only position prior is not a production decoder.
- The earlier lightweight result archive omits checkpoints, but the complete
  constrained-decoder recovery archive includes both original checkpoints.
- No separate resolved environment file or original per-run manifest was
  present; integrity is supplied by the recovery archive's checksum list.

## Artifacts and integrity

| Artifact | Location | SHA-256 | Availability |
|---|---|---|---|
| Local metrics/log archive | `/Users/krishaymaskara/Downloads/graph-v1-3341942-results.tar.gz` | `b7487bc32c8e9bdc5f436fddafd902f66bc5907364192784edfe7f132112cb79` | Present and verified |
| Complete constrained-decoder recovery archive | `/Users/krishaymaskara/research/audited-runs/skexgen-constrained-decoder-primary-evidence-20260803.tar.gz` | `5bd60fd884b23d607f452d82e8da1e71c97a119e9d8ad92ef0f3d91dbc4f2dcc` | Present; internal checksums verified |
| Metrics | Archive member `metrics.jsonl` | Covered by archive | Present and inspected |
| Slurm stdout/stderr | Archive members | Covered by archive | Present and inspected |
| Epoch checkpoints | Complete recovery archive, run `3341942` | Recorded hashes above | Present and verified |

## Reproduction and validation

The lightweight archive SHA-256 was checked against
`/Users/krishaymaskara/Downloads/graph-v1-final-bundle-checksums.txt`.
`metrics.jsonl` was parsed and its epoch-2 and terminal-success records were
checked against the program, edge, partition, and checkpoint claims above.
The scientific run was not rerun.

## Related records

- Baseline: [Frozen Flat V6](b0_constrained_flat_v6.md)
- Engineering prerequisite: [Graph V1 readiness](b0_graph_v1_engineering_readiness.md)
- Correction: [Graph V1 C1](b0_graph_v1_c1.md)
- Milestone: [Constrained flat/graph comparison](../milestones/constrained_flat_graph_comparison.md)
- Package contract: [Graph-baseline README](../../prototype/graph_baseline/README.md)
