# Current Project Status

This page is the authoritative concise statement of the project's present
state. Detailed implementation contracts live in package READMEs, completed
run evidence lives under `docs/experiments`, and future protocols belong in
specifications or new decision records.

## Repository state

| Item | Current value |
|---|---|
| Active branch | `graph-v1-experiment-record` |
| Current reviewed revision | This documentation update; its literal hash is intentionally not embedded |
| Frozen Flat V6 commit | `ac6ef718ae9bab7fa5a80d9f48d0976adf5cafad` |
| Frozen initial Graph V1 commit | `089b9f3d0e5a61fb19ef3fa05e993fc4eceffdcb` |
| Frozen final Graph V1 C1 commit | `0cd09ed34d4c4dd0d43e1456b7a06eb362ee7962` |
| Status recorded | August 7, 2026 |
| Controlled domain | Single-body sketch, extrude, and revolve histories with one or two operations |
| Authoritative corpus | 680 physical families: 544 train, 68 IID validation, 68 held-out IID test |
| Authorized GE1 manifest | Operation-template only: 407 train, 45 development, 114 RR systematic, 114 ER test |
| GE1 protocol record | `GE1-STAGE0-PREREG-v1`; C0 frozen; Stage 1 paired data path complete; neural model not begun |
| GE1 reviewer | Krishay Maskara |
| Systematic partition accessed | `false` |
| Held-out test partition accessed | `false` |

The frozen tags are `flat-v6-final-3338639`,
`graph-v1-initial-3341942`, and `graph-v1-c1-final-3341974`.

## Research objective

The bounded flat-versus-graph experiment asked whether explicit directed
typed-edge prediction improves autonomous CAD validity over flat relation and
pointer heads when both conditions share the same constrained node and
geometry path.

That comparison is complete on IID validation. Accepted
[ADR-0004](decisions/ADR-0004-ge1-single-manifest-encoder-comparison.md)
authorizes the next separately named question:

> Under one shared continuous-memory decoder and the operation-template-only
> protocol, does a position-free typed graph encoder outperform the flat
> chronological encoder on executable-prefix reconstruction and RR
> compositional generalization?

Graph V1 itself is frozen under
[ADR-0003](decisions/ADR-0003-freeze-graph-v1.md). GE1 does not unfreeze it or
authorize Graph V1 C2.

## Completed implementation milestones

- Controlled representation, deterministic data generation, family-level
  splits, Boolean filtering, kernel execution, counterfactual pairs, and
  shared model-data contracts.
- Original `B0-FLAT-MIXED-VQ` implementation, paired reconstruction
  evaluation, VQ-collapse diagnosis, train-only k-means repair, and frozen
  failed epoch-44 evaluation.
- Preregistered constraint-manifold and categorical-isolation diagnostics.
- Shared category-conditioned compact profile parameters and deterministic
  circle, rectangle, and capsule construction.
- Canonical controlled reference planes, node-conditioned categorical
  selection, exact-length controlled node grammar, and canonical revolve
  axis construction.
- Frozen Flat V6 structural baseline.
- Capacity-matched directed typed-edge Graph V1 decoder and strict inverse
  conversion.
- Production-shaped graph readiness, clean-source provenance, strict
  checkpoint reload, and authoritative C1 regression validation.
- One evidence-backed directed-position correction, C1.
- Program-level, family-level, edge-level, and scoring-prior comparisons.
- Frozen commits and tags plus locally verified Graph V1 metrics/log archives.
- GE1 C1 configuration and protected loading boundary, C2 position-free graph
  canonicalization, and C3 target-separated paired flat/graph batching.

## Frozen comparison

All three conditions used seed 2026, 544 training families, 68 IID validation
families, two epochs, batch size eight, 136 optimizer steps, 1,088
training-example presentations, and CPU execution.

| Model | Job | Parameters | Complete validity | Single-operation | Two-operation |
|---|---:|---:|---:|---:|---:|
| Flat V6 | `3338639` | 32,866 | 0 / 68 | 0 / 22 | 0 / 46 |
| Initial Graph V1 | `3341942` | 32,856 | 22 / 68 | 22 / 22 | 0 / 46 |
| Graph V1 C1 | `3341974` | 32,852 | 22 / 68 | 22 / 22 | 0 / 46 |

Flat V6 converted all 68 histories and produced canonical constrained nodes,
profiles, planes, and axes, but every history first failed with
`unexpected_edge` at the structural edge stage.

Initial Graph V1 predicted one of six directed edge classes for every legal
ordered pair. It converted all 68 histories, exactly solved all 12 `E` and 10
`R` families, and failed all 46 `EE`, `ER`, `RE`, and `RR` families with
`invalid_boolean_sequence`.

C1 added a capacity-neutral directed serialized-position branch. It improved
positive-edge precision from 65.10% to 73.08%, F1 from 67.73% to 71.91%, and
edge-class accuracy from 88.35% to 90.37%, while reducing false positives from
193 to 133. Exact validity did not change. `depends_on` recall fell from
37/46 to 11/46, offsetting improvements in local attachment relations.

The node-type-pair and edge-type scoring priors, initial Graph V1, and C1 all
reached 22/68 exact graphs. A position-only scoring prior reached 59/68. The
controlled serialization therefore contains strong topology information that
neither learned pairwise decoder recovered.

Both graph pilots used one active VQ code with perplexity `1.0`.

Detailed evidence is in the [Flat V6](experiments/b0_constrained_flat_v6.md),
[Graph V1 readiness](experiments/b0_graph_v1_engineering_readiness.md),
[initial Graph V1](experiments/b0_graph_v1_initial.md), and
[Graph V1 C1](experiments/b0_graph_v1_c1.md) records.

## Current evidence-supported conclusion

Explicit graph prediction materially improves over the frozen flat structural
heads for single-operation controlled programs. Independent ordered-pair
classification is insufficient for globally consistent two-operation
grouping. C1 demonstrates that better local edge metrics do not necessarily
produce a valid CAD program when a globally essential relation such as
`depends_on` is omitted.

The evidence does not isolate whether VQ collapse causally limits
multi-operation grouping. The collapsed graph models still solve every
single-operation example, so latent collapse and structural grouping remain
separate candidate questions.

## Protected data status

```text
systematic partition accessed: false
held-out test partition accessed: false
```

The IID validation partition has been used for repeated scientific iteration
and is not an untouched final benchmark. GE1 may not open any IID,
history-depth, or geometry-extrapolation partition. Under the accepted
operation-template policy, RR remains closed until the one-time systematic
stage and ER remains closed throughout GE1.

## What has not been established

- Any valid two-operation autonomous program.
- Reliable operation-to-operation dependency prediction.
- Globally consistent repeated-instance grouping.
- Order-independent or permutation-robust graph learning.
- Systematic-generalization or held-out test performance.
- Localized learned-edit results.
- Diverse or semantically meaningful VQ usage.
- Multi-seed stability or behavior beyond two operations.
- OpenCascade execution of useful learned multi-operation predictions.
- Broad or real-CAD generality.

## Immediate scientific gate

ADR-0004 is accepted by designated reviewer Krishay Maskara, with fixed
epoch-50 checkpoint selection. The acceptance did not remove ADR-0004's
preauthorized two-seed timing fallback. The
[Stage 0 preregistration](specifications/ge1_stage0_preregistration.md) freezes
the complete protocol, records the verified manifest and assignment hashes,
and records zero protected payload access. C0 is complete. The Stage 1 paired
data path is implemented and procedurally validated for all six controlled
templates; neural encoder and model work has not begun. The immediate
engineering gate is the position-free relational encoder unit, with
bookkeeping restricted to addressing, membership, scatter/gather, masking,
and graph-local pooling.

No further Graph V1 correction or rerun is authorized. RR access remains
blocked until the accepted one-time systematic stage, and ER remains closed
throughout GE1.

## Documentation and artifact maintenance

The documentation system now contains separate records for the constrained
flat progression, frozen Flat V6, initial Graph V1, and C1. The combined graph
summary is retained only as a dated report.

The complete constrained-decoder recovery archive contains the original Flat
V2-V6 and Graph V1/C1 metrics, epoch checkpoints, scheduler accounting, and
stdout/stderr, with a transfer-verified outer hash and verified internal
checksums. The earlier lightweight Graph archives remain convenient subsets.
Separate resolved environment files and original per-run manifests were not
present in the recovered namespaces; those are now the remaining evidence
gaps and do not change any frozen result.

Run `python3 tools/check_documentation.py` before committing documentation
changes.
