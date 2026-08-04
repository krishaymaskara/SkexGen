# Frozen Flat V6 and Graph V1 Experiment Record

## 1. Research question

This controlled experiment tested whether replacing the flat decoder's structural relation and pointer heads with an explicit directed typed-edge decoder could improve autonomous CAD program validity while preserving the frozen V6 node and geometry path.

Serialized position and constrained node types reveal substantial topology in this controlled dataset. The results therefore are not evidence of order-independent graph generalization.

## 2. Fixed protocol

| Setting | Value |
| --- | ---: |
| Seed | 2026 |
| Training families | 544 |
| IID validation families | 68 |
| Epochs | 2 |
| Batch size | 8 |
| Steps | 136 |
| Training-example presentations | 1088 |
| Device | CPU |
| Systematic partition accessed | false |
| Held-out test partition accessed | false |

## 3. Frozen models

### Flat V6

- Commit: `ac6ef718ae9bab7fa5a80d9f48d0976adf5cafad`
- Pilot job: `3338639`
- Architecture: frozen V6 flat constrained decoder with structural relation and pointer heads
- Total parameters: 32,866
- Complete validity: 0/68
- Conversion success: 68/68
- Final failure: `unexpected_edge` for 68/68

### Initial Graph V1

- Commit: `089b9f3d0e5a61fb19ef3fa05e993fc4eceffdcb`
- Pilot job: `3341942`
- Architecture: shared pair decoder `225 → 15 → 6`
- Graph edge decoder parameters: 3,486
- Total parameters: 32,856
- Difference from Flat V6: 10
- Complete validity: 22/68
- Exact graph match: 22/68
- Two-operation validity: 0/46

### Graph V1 C1

- Commit: `0cd09ed34d4c4dd0d43e1456b7a06eb362ee7962`
- Pilot job: `3341974`
- Architecture: main pair MLP `225 → 14 → 6` plus a directed rank-6 ordered-position bias
- Graph edge decoder parameters: 3,482
- Total parameters: 32,852
- Difference from Flat V6: 14
- Complete validity: 22/68
- Exact graph match: 22/68
- Two-operation validity: 0/46

## 4. Final comparison

| Metric                    | Flat V6 | Initial Graph V1 | Graph V1 C1 |
| ------------------------- | ------: | ---------------: | ----------: |
| Complete validity         |    0/68 |            22/68 |       22/68 |
| Exact graph match         |     N/A |            22/68 |       22/68 |
| Single-operation validity |    0/22 |            22/22 |       22/22 |
| Two-operation validity    |    0/46 |             0/46 |        0/46 |
| Positive-edge precision   |     N/A |           65.10% |      73.08% |
| Positive-edge recall      |     N/A |           70.59% |      70.78% |
| Positive-edge F1          |     N/A |           67.73% |      71.91% |
| Edge-class accuracy       |     N/A |           88.35% |      90.37% |

Family validity was identical for Initial Graph V1 and Graph V1 C1:

| Family | Validity |
| --- | ---: |
| E | 12/12 |
| R | 10/10 |
| EE | 0/11 |
| ER | 0/9 |
| RE | 0/10 |
| RR | 0/16 |

For both graph models, all 46 two-operation programs failed with `invalid_boolean_sequence`.

## 5. C1 edge-level effect

| Edge outcome | Initial Graph V1 | Graph V1 C1 |
| --- | ---: | ---: |
| True positives | 360 | 361 |
| False positives | 193 | 133 |
| False negatives | 150 | 149 |

C1 removed 60 false-positive edges and improved positive-edge F1 by approximately 4.18 percentage points, but produced no additional exact graphs.

The principal edge-type tradeoff was:

| Edge type | Initial Graph V1 F1 | Graph V1 C1 F1 |
| --- | ---: | ---: |
| `defined_in` | 64.40% | 72.22% |
| `uses_axis` | 52.17% | 59.02% |
| `uses_profile` | 48.94% | 62.14% |
| `depends_on` | 80.43% | 38.60% |

For `depends_on`, initial recall was 37/46 = 80.43%, whereas C1 recall was 11/46 = 23.91%. C1 improved local attachment precision but omitted most operation-to-operation dependencies required by multi-operation programs.

## 6. Baseline interpretation

| Decoder or baseline | Exact graph matches |
| --- | ---: |
| Node-type-pair prior | 22/68 |
| Edge-type prior | 22/68 |
| Position-only prior | 59/68 |
| Initial Graph V1 | 22/68 |
| Graph V1 C1 | 22/68 |

The learned graph models remained tied with the node-type relational baselines and did not capture the positional topology exploited by the position-only prior.

## 7. VQ result

Both graph pilots reported:

```text
active code count: 1
perplexity: 1.0
```

This is inherited latent collapse. C1 was not intended to correct VQ behavior.

## 8. Supported conclusion

> Explicit graph prediction eliminated the flat decoder’s universal structural failure and perfectly recovered single-operation programs. However, the shared pairwise decoder did not learn globally consistent operation grouping in repeated-node programs. A capacity-neutral directed position bias improved local edge precision and F1, but did not improve exact graph validity because gains in attachment edges were offset by a collapse in operation-dependency recall.

## 9. What is not supported

These experiments do not establish:

- order-independent graph generalization;
- performance on the systematic partition;
- performance on the held-out test partition;
- performance beyond two operations;
- that more epochs would resolve the failure;
- that the position-only prior is a suitable deployable decoder; or
- that the VQ representation carries meaningful program diversity.

## 10. Freeze declaration

```text
graph scientific corrections used: 1
graph scientific correction limit: 1
additional graph corrections authorized: 0
additional graph pilot reruns authorized: 0
```

The graph experiment is frozen after job `3341974`.
