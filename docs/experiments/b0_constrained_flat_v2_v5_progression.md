# B0 Constrained Flat Decoder V2-V5 Progression

## Record status

| Item | Value |
|---|---|
| Experiment ID | `b0-constrained-flat-v2-v5-progression` |
| Record status | `verified` |
| Evidence state | `verified-local` |
| Scientific role | Bounded staged decoder repair and failure localization |
| Branch | `constrained-profile-decoder` |
| Source commits | V2 series `c63d206`-`9e66231`; V3 `0127690`; V4 series `a49941f`-`98aa965`; V5 series `e6dc32e`-`d1eae8f` |
| Working tree | Clean commits; per-run provenance retained externally |
| Slurm/local run ID | V2 `3333190`; V3 `3334371`; V4 `3337600`; V5 `3338093` |
| Execution dates | July 30-August 2, 2026 |

This record consolidates the bounded V2-V5 engineering progression. On
August 3, 2026, the original Adroit run directories, checkpoints, metrics,
diagnostics, scheduler logs, and scheduler accounting were recovered into one
checksum-verified local archive. The accepted pilots and the failed runs that
motivated V4/V5 repairs are therefore distinguished from primary evidence
rather than from run names or later regression tests alone.

## Question and predetermined decision rule

The staged question was whether a flat chronological decoder could advance
from the epoch-44 profile-category/geometry failure to structurally evaluable
autonomous programs by correcting one immediate frozen blocker at a time.

Each version retained the previous version's unaffected architecture and
protocol. A completed pilot was an engineering gate, not a requirement for
nonzero complete validity. The next version was motivated by the first frozen
failure category of the preceding version:

| Version | Isolated intervention | Resulting next blocker |
|---|---|---|
| V2 | Category-conditioned compact profile parameters and deterministic geometry | Controlled reference-plane geometry |
| V3 | Canonical plane from predicted plane category | Categorical sentinel/applicability |
| V4 | Node-conditioned categorical selection | Node grammar |
| V5 | Prefix-conditioned exact-length node grammar | Axis geometry and then structural edges |

No version used systematic or held-out test data.

## Inputs and partition authority

| Item | Value |
|---|---|
| Corpus | Authoritative 680-family controlled corpus |
| Training families | 544 |
| IID validation families | 68 |
| Systematic families used | 0 |
| Held-out test families used | 0 |
| Test partition evaluated | `false` |

The two-epoch pilots used the ordinary IID validation partition for bounded
engineering iteration. It is therefore not an untouched final benchmark.

## Configuration

The shared pilot protocol was:

```text
seed: 2026
epochs: 2
batch size: 8
optimizer steps: 136
training-example presentations: 1,088
device: CPU
fresh normal VQ initialization
```

V2 also included a train-only tiny-overfit workflow and two frozen,
read-only diagnostic workflows. The V2 pilot used by those diagnostics was
commit `9e662310b41b3750a49547b95fa6af7a9235a8b4`, job `3333190`.

## Environment

The authoritative pilots and diagnostics ran through committed Adroit CPU
Slurm wrappers under the project's Python 3.8/PyTorch 1.11 environment. The
recovered scheduler accounting identifies the nodes, start/end times, elapsed
times, states, and exit codes. A separate resolved environment file was not
present in the recovered namespaces.

## Execution

Committed entry points and wrappers are preserved under:

```text
prototype/flat_baseline/run_constrained_v2_pilot.py
prototype/flat_baseline/run_constrained_v3_pilot.py
prototype/flat_baseline/run_constrained_v4_pilot.py
prototype/flat_baseline/run_constrained_v5_pilot.py
prototype/flat_baseline/adroit/constrained_v*_pilot_cpu.slurm
```

The runners use new versioned output roots and prohibit systematic and test
access. Frozen V2 diagnostics were read-only and published separately from
the pilot.

Known job references retained by current regression tests are:

| Job | Durable role in the current repository |
|---|---|
| `3333190` | Accepted V2 two-epoch IID pilot consumed by the frozen V2 diagnostics |
| `3334371` | Accepted V3 two-epoch IID pilot |
| `3334551` | Failed pre-V4 categorical/node-selection regression source; no training step completed |
| `3336430` | Failed V4 autonomous-node-count regression source after epoch 1 |
| `3337600` | Accepted corrected V4 two-epoch IID pilot |
| `3337640` | Failed V5 checkpoint-contract regression source after two training epochs |
| `3338093` | Accepted corrected V5 two-epoch IID pilot |

The failed jobs are engineering evidence, not scientific pilot outcomes. The
accepted V2-V5 jobs all ended `COMPLETED 0:0`, recorded terminal success,
selected epoch 2, and reported both systematic- and test-partition access as
`false`.

## Verified results

### V2 and reference-plane diagnosis

V2 implemented the neural profile-family head, compact family-conditioned
parameters, deterministic profile reconstruction, training, conversion,
autonomous decoding, and strict checkpoint contracts.

The train-only tiny-overfit work found that a 100-step snapshot could occur
during a temporary fresh-normal VQ/EMA assignment transition, while the
unchanged objective recovered by step 200. No VQ freeze or warmup was added.

The frozen plane diagnostic established that controlled reference planes
have no independent continuous freedom after the predicted `XY`, `XZ`, or
`YZ` category is fixed. This motivated V3's deterministic construction.

### V3

The checked-in experiment contract records:

```text
complete autonomous validity: 0 / 68
canonical generated planes and profiles: 68 / 68
conversion success: 12 / 68
first failures:
  categorical_pad_sentinel: 56
  node_category_applicability: 12
```

### V4

V4 selected categorical values under predicted-node-type applicability while
retaining raw argmax evidence. Its frozen result was:

```text
constrained conversion success: 68 / 68
raw categorical-arm conversion success: 22 / 68
complete validity: 0 / 68
first failures:
  invalid_node_grammar: 45
  unexpected_edge: 23
```

### V5

V5 selected nodes under the exact-length controlled prefix grammar using only
generated prefixes and authorized node count. Its corrected frozen result was:

```text
constrained grammar failures: 0
constrained conversion success: 68 / 68
complete validity: 0 / 68
first failures:
  invalid_axis_geometry: 45 revolve-containing examples
  unexpected_edge: 23 extrude-only examples
```

## Decision

The staged repair was accepted as an evidence-guided engineering progression.
V5 established working profile, plane, categorical, and node-grammar paths
and isolated canonical controlled axis construction as the final planned
flat non-relational intervention.

## Interpretation

The sequence supports deterministic construction where the controlled domain
has no genuine continuous or categorical freedom. It also shows why complete
validity must be evaluated stage by stage: each successful correction exposed
a later blocker rather than proving the model globally successful.

## Limitations and claims not supported

- The recovery archive has an internal per-file checksum list but no original
  per-run manifest or separate resolved environment file.
- Readiness-job outputs are preserved as scheduler logs; only the selected
  run namespaces listed below were copied in full.
- The results do not establish systematic or test generalization.
- The sequence used IID validation for model iteration.
- V2-V5 did not produce a completely valid autonomous program.
- No causal claim about VQ collapse follows from these staged corrections.

## Artifacts and integrity

| Artifact | Location | SHA-256 | Availability |
|---|---|---|---|
| Complete constrained-decoder recovery archive | `/Users/krishaymaskara/research/audited-runs/skexgen-constrained-decoder-primary-evidence-20260803.tar.gz` | `5bd60fd884b23d607f452d82e8da1e71c97a119e9d8ad92ef0f3d91dbc4f2dcc` | Present; internal checksums verified |
| V2 accepted checkpoint, epoch 2 | Archive run `3333190` | `516c480ad7477bd93d14ece7247057a831760a7399552845def2ac2846eb85b7` | Present and verified |
| V3 accepted checkpoint, epoch 2 | Archive run `3334371` | `de3f289fc8999d0c07da996c015ecb4061412b9f9cb80cbf0f770404d3f6f365` | Present and verified |
| V4 accepted checkpoint, epoch 2 | Archive run `3337600` | `88633c8c2bebdaf3d50c147aac2b0680e8db58905caded4117aba1fe25cd961b` | Present and verified |
| V5 accepted checkpoint, epoch 2 | Archive run `3338093` | `02d3daf74735e44e997084339e2f942eb1d14d21cc28b230f559b8f26de82be4` | Present and verified |
| V2 diagnostics | Archive jobs `3331643`, `3333229`, and `3333388` | Covered by internal `SHA256SUMS` | Present and verified |
| Failed regression namespaces | Archive jobs `3334551`, `3336430`, and `3337640` | Covered by internal `SHA256SUMS` | Present and verified |
| Implementations and tests | Repository commits listed above | Git object identity | Present |

## Reproduction and validation

The outer archive hash matches on Adroit and the Mac, and every archived file
passes the archive's internal `SHA256SUMS`. The accepted JSONL files parse,
end in terminal success, and independently confirm the metrics and protected-
partition claims above. This was recovery of original evidence, not a rerun.

## Related records

- Motivation: [Categorical-isolation replay](b0_phase_b_categorical_isolation_replay.md)
- Scope decision: [ADR-0002](../decisions/ADR-0002-three-week-flat-versus-graph-scope.md)
- Successor: [Frozen Flat V6](b0_constrained_flat_v6.md)
- Package contract: [Flat-baseline README](../../prototype/flat_baseline/README.md)
