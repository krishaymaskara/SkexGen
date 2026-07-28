# Current Project Status

This page is the authoritative, concise statement of the project's present
state. Detailed implementation contracts live in the prototype package
READMEs, completed-run evidence lives under `docs/experiments`, and proposed
future work lives in the research plan and experimental specification.

## Repository state

| Item | Current value |
|---|---|
| Active branch | `flat-mixed-baseline` |
| Current reviewed repository revision | This documentation commit; its hash is intentionally not embedded here |
| Phase B evaluator implementation commit | `182ff31a121627f7803b29de1859fae9759a1849` |
| Checkpoint training source commit | `d549ebe4478e166fda8d54f2b173572a1ab52e7a` |
| Authoritative checkpoint | Epoch 44, global step 748 |
| Checkpoint SHA-256 | `282988af00a2dc9a53e14ceb270537d35f85d5339dc989634f88af5f77931267` |
| Status recorded | July 28, 2026 |
| Controlled domain | Single-body sketch, extrude, and revolve histories |
| Authoritative corpus | 680 physical families: 544 train, 68 validation, 68 test |

The current reviewed repository revision is the documentation commit that
contains this page. Its literal hash is not embedded because doing so would
change the commit and create a self-referential hash problem. The evaluator
implementation is fixed at `182ff31...`. The separate `d549ebe...` commit is
provenance for the repaired training run and checkpoint; it is not described
as the current repository HEAD.

## Research objective

The project studies whether typed CAD dependency graphs and different
structure/geometry latent representations improve systematic generalization
and localized editing over a flat, mixed representation. Revolve is a
controlled experimental setting rather than the claimed contribution.

The intended principal comparison remains:

1. a flat/mixed VQ baseline;
2. a typed graph with discrete structure and discrete,
   structure-conditioned geometry;
3. a typed graph with discrete structure and continuous,
   structure-conditioned geometry.

## Completed implementation milestones

- Mapped and smoke-tested the inherited SkexGen pipeline.
- Implemented a versioned typed CAD representation.
- Implemented deterministic controlled extrude/revolve generation and
  systematic split manifests.
- Implemented analytical Boolean-feasibility filtering and independent
  OpenCascade execution validation.
- Implemented deterministic counterfactual edit pairs and locality metadata.
- Implemented a shared flat/graph model-data contract.
- Implemented the `B0-FLAT-MIXED-VQ` model and reproducible training
  infrastructure.
- Implemented oracle-node-count-conditioned autoregressive decoding.
- Implemented paired teacher-forced/predicted-history validation evaluation
  with shared reconstructed latent memory.
- Implemented VQ-collapse diagnosis.
- Implemented deterministic train-only k-means codebook initialization,
  a bounded pilot, and full epoch-boundary retraining.

These completion statements describe checked-in capabilities. Individual
scientific run results require separate experiment records.

## Current B0 evidence

The original meaningful B0 run completed operationally but collapsed to one
active VQ code. Subsequent diagnosis identified nearest-code assignment
collapse as the proximate failure: encoder outputs and codebook vectors
remained varied, gradients reached the encoder, and the decoder responded to
latent changes.

A bounded intervention used deterministic, training-partition-only k-means
initialization with EMA pseudo-count mass matched to normal initialization.
The accepted pilot proceeded to a 50-epoch continuation. The archived and
verified full run is:

| Item | Verified value |
|---|---|
| Slurm job | `3326757` |
| Training source commit | `d549ebe4478e166fda8d54f2b173572a1ab52e7a` |
| Training decision | `PASS_FOR_EVALUATION` |
| Completed epoch | 50 |
| Selected checkpoint | `best.pt`, epoch 44, global step 748 |
| Selected checkpoint SHA-256 | `282988af00a2dc9a53e14ceb270537d35f85d5339dc989634f88af5f77931267` |
| Selected validation active codes | 5 |
| Selected validation perplexity | 3.1204 |
| Test partition evaluated | `false` |

`PASS_FOR_EVALUATION` is a training and code-usage gate, not final scientific
acceptance. The repaired model still exhibits stable codebook
under-utilization. The values above are backed by the durable
[full-retraining record](experiments/b0_train_kmeans_full_retrain.md).

The [experiment evidence inventory](experiments/README.md) records which
primary bundles have been inspected. The counterfactual OpenCascade and
real-PyTorch model-data archives are locally checksum-verified. The original
B0, Phase A, Phase B, VQ-diagnosis, train-k-means pilot, and full-retraining
artifacts are also locally archived, internally checksum-verified, and
covered by individual experiment records. Accepted infrastructure stages are
summarized in the
[controlled-CAD foundation](milestones/controlled_cad_foundation.md) and
[flat-baseline](milestones/flat_baseline.md) milestone pages.

## What has not been established

- No graph-discrete or graph-hybrid neural model has been implemented.
- No capacity-matched graph-versus-flat comparison has been run.
- The repaired epoch-44 B0 checkpoint has not yet completed its final
  validation-only reconstruction evaluation.
- Executable OpenCascade reconstruction of repaired-model predictions has not
  been established.
- Latent-code semantics and localized learned edits have not been evaluated.
- Systematic-generalization model results and multi-seed comparisons do not
  yet exist.
- The held-out IID test partition has not been evaluated.

## Immediate scientific gate

The next scientific action is validation-only evaluation of the selected
epoch-44 checkpoint using the checked-in Phase B evaluator. That evaluation
should measure teacher-forced and predicted-history reconstruction, the three
validity layers, structural and geometric accuracy, operation/pointer/edge
metrics, and code usage. Test data must remain untouched until the validation
protocol and checkpoint are accepted.

OpenCascade execution, code semantics, and localized edit interventions may
then be added under a frozen protocol. Graph-model implementation should
begin only after the flat baseline has a credible, shared evaluation path.

## Documentation maintenance

There is no remaining record backfill for a completed run whose primary
bundle has been recovered locally. The next validation-only evaluation of the
repaired epoch-44 checkpoint must create a new immutable experiment record
rather than editing the original-checkpoint Phase B result.

Historical evidence marked `documented-external` in the inventory remains
eligible for integrity upgrade if its primary artifacts are recovered.
Documentation changes are checked with
`python3 tools/check_documentation.py`. This page continues to distinguish
checked-in implementation, verified run results, and conclusions that remain
unestablished.
