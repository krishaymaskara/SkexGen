# B0 Frozen Constrained Flat V6

## Record status

| Item | Value |
|---|---|
| Experiment ID | `b0-constrained-flat-v6-3338639` |
| Record status | `verified` |
| Evidence state | `verified-local` |
| Scientific role | Frozen flat structural baseline |
| Branch | `constrained-profile-decoder` |
| Source commit | `ac6ef718ae9bab7fa5a80d9f48d0976adf5cafad` |
| Working tree | `clean` |
| Slurm/local run ID | `3338639` |
| Execution date | August 2, 2026 |
| Frozen tag | `flat-v6-final-3338639` |

## Question and predetermined decision rule

V6 asked whether canonical controlled-axis construction would remove V5's
universal revolve-containing axis failure while preserving the already frozen
profile, plane, categorical, and node-grammar paths.

The pilot required exact completion of the fixed two-epoch protocol and valid
canonical axes wherever applicable. It did not require nonzero complete CAD
validity. After the pilot, the flat decoder was frozen and remaining
structural failures were to be carried into the graph comparison. No Flat V7
was authorized.

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

```text
seed: 2026
epochs: 2
batch size: 8
optimization steps: 136
training-example presentations: 1,088
device: CPU
total parameters: 32,866
```

V6 retained V5's architecture, state-dictionary shapes, logits, losses,
VQ/EMA behavior, prefix grammar, category/profile/plane construction,
relation heads, pointer heads, and edge decoder. It changed only generated
axis-node geometry to the controlled sketch-local axis `(0,0)` with direction
`(0,1)`.

## Environment

The authoritative run used the committed Adroit CPU Slurm workflow. Scheduler
accounting and stdout/stderr are locally recovered; a separate resolved
environment file was not present in the run namespace.

## Execution

The committed runner and wrapper are:

```text
prototype/flat_baseline/run_constrained_v6_pilot.py
prototype/flat_baseline/adroit/constrained_v6_pilot_cpu.slurm
```

The immutable external output is associated with job `3338639` and frozen tag
`flat-v6-final-3338639`.

Job `3338085` was a failed V5 checkpoint-fix validation and is retained in the
V6-focused regression test name as historical provenance; it was not the
successful V6 readiness job. V6 validation job `3338478` failed before the
test-fixture repair. Corrected production validation job `3338602` completed
successfully, followed by accepted pilot `3338639`.

## Verified results

```text
complete autonomous validity: 0 / 68
graph conversion success: 68 / 68
node grammar completion: 68 / 68
canonical planes and profiles: 68 / 68
canonical axes where applicable: passed
first failing stage: controlled_edges
first failure category: unexpected_edge
examples failing there: 68 / 68
```

V6 therefore removed the deterministic axis blocker without producing a
valid complete program. Every validation family advanced to the unchanged
flat structural relation path and failed there.

## Decision

V6 passed its engineering acceptance boundary and was frozen as the flat
comparison condition. Its scientific outcome was negative: complete validity
remained zero. Structural prediction became the isolated comparison target
for Graph V1.

## Interpretation

The result establishes a clean flat structural baseline: non-relational
controlled construction succeeds, conversion succeeds, and the first failure
is uniformly the edge stage. It does not establish that the entire neural
model is useful or that deterministic constraints alone solve CAD generation.

## Limitations and claims not supported

- The recovery archive has an internal checksum list but no original per-run
  manifest or separate resolved environment file.
- The result uses one seed, two epochs, and the IID validation partition.
- VQ behavior remains collapsed in the later matched graph pilots and was not
  an intervention in this comparison.
- Systematic and held-out test performance are unknown.
- No complete autonomous V6 program was valid.

## Artifacts and integrity

| Artifact | Location | SHA-256 | Availability |
|---|---|---|---|
| Frozen source | Git commit and annotated tag | Git object identity | Present |
| Complete constrained-decoder recovery archive | `/Users/krishaymaskara/research/audited-runs/skexgen-constrained-decoder-primary-evidence-20260803.tar.gz` | `5bd60fd884b23d607f452d82e8da1e71c97a119e9d8ad92ef0f3d91dbc4f2dcc` | Present; internal checksums verified |
| Epoch-2 checkpoint | Archive run `3338639` | `aa9cd5158eea3e003a02a8482e4946e025d5d91284c3d14f10a86be3495fd35b` | Present and verified |
| Metrics and scheduler evidence | Archive jobs `3338478`, `3338602`, and `3338639` | Covered by internal `SHA256SUMS` | Present and verified |

## Reproduction and validation

The archive hash matches on Adroit and the Mac, and every internal checksum
passes. Job `3338639` metrics parse, end in terminal success, select the
verified epoch-2 checkpoint, and report systematic/test access as `false`.
The original job was recovered, not rerun.

## Related records

- Predecessor: [V2-V5 constrained progression](b0_constrained_flat_v2_v5_progression.md)
- Comparison: [Initial Graph V1](b0_graph_v1_initial.md)
- Final comparison milestone: [Constrained flat/graph comparison](../milestones/constrained_flat_graph_comparison.md)
- Package contract: [Flat-baseline README](../../prototype/flat_baseline/README.md)
