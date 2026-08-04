# B0 Graph V1 Directed-Position Correction C1

## Record status

| Item | Value |
|---|---|
| Experiment ID | `b0-graph-v1-c1-3341974` |
| Record status | `verified` |
| Evidence state | `verified-local` |
| Scientific role | Single authorized graph correction and final Graph V1 pilot |
| Branch | `graph-profile-decoder` |
| Source commit | `0cd09ed34d4c4dd0d43e1456b7a06eb362ee7962` |
| Parent commit | `089b9f3d0e5a61fb19ef3fa05e993fc4eceffdcb` |
| Working tree | `clean` |
| Slurm/local run ID | Validation `3341972`; pilot `3341974` |
| Execution date | August 3, 2026 |
| Frozen tag | `graph-v1-c1-final-3341974` |

## Question and predetermined decision rule

The initial pilot tied node-type relational priors at 22/68 exact graphs while
a scoring-only position prior reached 59/68. C1 tested one hypothesis:

> A direct directed serialized-position bias can distinguish repeated node
> instances more effectively than the initial shared pair features.

This was the only authorized scientific correction. The training protocol,
targets, losses, data, validity rules, node/geometry path, and legal-class
mask remained frozen. Program-level exact validity was the primary endpoint;
edge metrics were diagnostic.

## Inputs and partition authority

| Item | Value |
|---|---|
| Corpus | Authoritative 680-family controlled corpus |
| Train families used | 544 |
| IID validation families used | 68 |
| Systematic families used | 0 |
| Test families used | 0 |
| Test partition evaluated | `false` |

## Configuration

C1 summed a directed position branch with the main pair logits before the
unchanged legal mask:

```text
main pair MLP: 225 -> 14 -> 6
source-position embedding: 16 x 6
destination-position embedding: 16 x 6
elementwise source/destination interaction
bias-free class projection: 6 -> 6
position-branch parameters: 228
graph decoder parameters: 3,482
total model parameters: 32,852
difference from Flat V6: -14, approximately -0.043%
```

Training remained seed 2026, two epochs, batch size 8, 136 steps, 1,088
training-example presentations, and CPU execution.

The implementation changed 13 files, all under `prototype/graph_baseline`.
It did not modify Flat V6 production code, targets, graph loss, optimizer,
datasets, conversion-validity rules, or protected partitions.

## Environment

Authoritative validation job `3341972` used Python `3.8.13`, PyTorch `1.11.0`,
CUDA build `11.3`, and CPU execution on Adroit. The scientific pilot used the
same committed CPU workflow.

## Execution

Job `3341971` failed only because the validation harness searched production
source for the current C1 commit instead of the intentionally recorded parent
commit. It did not run the scientific pilot and did not change production
code.

Corrected validation job `3341972` passed 17/17 focused graph tests, the V4-V6
focused suites, the 549-test flat suite with four authorized external-bundle
skips, representation, controlled-data, and model-data suites, Python 3.8
grammar, compilation, `compileall`, Slurm syntax, and `git diff --check`.

Job `3341974` completed two epochs, selected epoch 2, strictly reloaded
selected and final checkpoints, reproduced results, and reached terminal
success.

## Verified results

### Program level

| Metric | Initial Graph V1 | C1 | Change |
|---|---:|---:|---:|
| Complete validity | 22 / 68 | 22 / 68 | 0 |
| Exact graph match | 22 / 68 | 22 / 68 | 0 |
| Single-operation validity | 22 / 22 | 22 / 22 | 0 |
| Two-operation validity | 0 / 46 | 0 / 46 | 0 |

All two-operation programs again failed with `invalid_boolean_sequence`.

### Edge level

| Metric | Initial | C1 | Change |
|---|---:|---:|---:|
| Positive-edge precision | 65.10% | 73.08% | +7.98 pp |
| Positive-edge recall | 70.59% | 70.78% | +0.20 pp |
| Positive-edge F1 | 67.73% | 71.91% | +4.18 pp |
| Edge-class accuracy | 88.35% | 90.37% | +2.02 pp |
| True positives | 360 | 361 | +1 |
| False positives | 193 | 133 | -60 |
| False negatives | 150 | 149 | -1 |

| Edge type | Initial F1 | C1 F1 |
|---|---:|---:|
| `defined_in` | 64.40% | 72.22% |
| `placed_on` | 92.11% | 92.11% |
| `uses_axis` | 52.17% | 59.02% |
| `uses_profile` | 48.94% | 62.14% |
| `depends_on` | 80.43% | 38.60% |

`depends_on` recall fell from `37/46 = 80.43%` to `11/46 = 23.91%`.
C1's dependency predictions had 100% precision but omitted most required
operation dependencies.

The epoch-2 position/main mean-absolute-logit ratio was approximately 13.58%
over all active pairs, 16.25% for single-operation examples, and 13.20% for
two-operation examples. The two-operation ratio fell from approximately
19.25% at epoch 1.

The position-only prior remained 59/68 exact graphs; C1 remained 22/68. VQ
usage remained one active code with perplexity `1.0`.

Checkpoint SHA-256 values recorded by the run were:

```text
epoch 1: 1183317cb75fe4e2913f516ef3891e0f4f8b8cd067e493283caeb5ac7696a350
epoch 2: 3afe12e3c761b7ecb81ae0eeac04d39ea445d4568553331607be4c489c8a96d9
```

## Decision

C1 materially improved local edge classification but produced no
program-level improvement. The correction budget was exhausted, and Graph V1
was frozen after job `3341974` with no further corrections or reruns
authorized.

## Interpretation

The additive position branch redistributed errors. It improved local
attachment precision but reduced the globally essential operation-dependency
recall. Independent pair classification remained insufficient for globally
consistent multi-operation grouping.

## Limitations and claims not supported

- This does not establish order-independent graph learning.
- Systematic and held-out test partitions remain untouched.
- It does not establish behavior beyond two operations or across seeds.
- More epochs were not tested and cannot be inferred to solve the failure.
- VQ collapse may matter but was not isolated.
- Improved edge F1 did not imply improved program validity.
- The earlier lightweight result archive omits checkpoints, but the complete
  constrained-decoder recovery archive includes both original checkpoints.
- No separate pilot environment file or original per-run manifest was
  present; integrity is supplied by the recovery archive's checksum list.

## Artifacts and integrity

| Artifact | Location | SHA-256 | Availability |
|---|---|---|---|
| Local metrics/log archive | `/Users/krishaymaskara/Downloads/graph-v1-c1-3341974-results.tar.gz` | `aa3c4cdabb66a8160d358662d23be57183db06b8341b7613232e6cdf6d4a8178` | Present and verified |
| Complete constrained-decoder recovery archive | `/Users/krishaymaskara/research/audited-runs/skexgen-constrained-decoder-primary-evidence-20260803.tar.gz` | `5bd60fd884b23d607f452d82e8da1e71c97a119e9d8ad92ef0f3d91dbc4f2dcc` | Present; internal checksums verified |
| Metrics | Archive member `metrics.jsonl` | Covered by archive | Present and inspected |
| Slurm stdout/stderr | Archive members | Covered by archive | Present and inspected |
| Epoch checkpoints | Complete recovery archive, run `3341974` | Recorded hashes above | Present and verified |

## Reproduction and validation

The lightweight archive hash matches
`/Users/krishaymaskara/Downloads/graph-v1-final-bundle-checksums.txt`.
`metrics.jsonl` was parsed and checked against the metrics, partition flags,
checkpoint identities, and terminal-success claims above. The run itself was
not repeated.

## Related records

- Parent: [Initial Graph V1](b0_graph_v1_initial.md)
- Engineering validation: [Graph V1 readiness](b0_graph_v1_engineering_readiness.md)
- Flat comparison: [Frozen Flat V6](b0_constrained_flat_v6.md)
- Milestone: [Constrained flat/graph comparison](../milestones/constrained_flat_graph_comparison.md)
- Handoff questions: [Graph V1 handoff](../reports/graph_v1_handoff_2026-08-03.md)
- Package contract: [Graph-baseline README](../../prototype/graph_baseline/README.md)
