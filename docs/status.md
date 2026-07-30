# Current Project Status

This page is the authoritative, concise statement of the project's present
state. Detailed implementation contracts live in the prototype package
READMEs, completed-run evidence lives under `docs/experiments`, and proposed
future work lives in the research plan and experimental specification.

## Repository state

| Item | Current value |
|---|---|
| Active branch | `constrained-profile-decoder` |
| Current reviewed repository revision | This documentation commit; its hash is intentionally not embedded here |
| Phase B evaluator implementation commit | `182ff31a121627f7803b29de1859fae9759a1849` |
| Checkpoint training source commit | `d549ebe4478e166fda8d54f2b173572a1ab52e7a` |
| Authoritative checkpoint | Epoch 44, global step 748 |
| Checkpoint SHA-256 | `282988af00a2dc9a53e14ceb270537d35f85d5339dc989634f88af5f77931267` |
| Status recorded | July 30, 2026 |
| Controlled domain | Single-body sketch, extrude, and revolve histories |
| Authoritative corpus | 680 physical families: 544 train, 68 validation, 68 test |

The current reviewed repository revision is the documentation commit that
contains this page. Its literal hash is not embedded because doing so would
change the commit and create a self-referential hash problem. The evaluator
implementation is fixed at `182ff31...`. The separate `d549ebe...` commit is
provenance for the repaired training run and checkpoint; it is not described
as the current repository HEAD.

## Research objective

The active core question is:

> Does a typed dependency-graph representation improve systematic
> generalization and localized editing compared with a flat chronological
> representation when both use the same category-conditioned constrained
> continuous-geometry decoder?

The required principal comparison is one repaired flat model versus one
minimal typed dependency-graph model using the same decoder. Capacity and
training opportunity must be controlled closely enough for the representation
comparison to remain interpretable. Ordinary validation, one
systematic-generalization split, and one localized numerical-edit evaluation
form the required evaluation set.

This three-week scope supersedes the older active three-model roadmap for
future work; the broader six-to-eight-week plan remains labeled as historical
context in [the research plan](research_plan.md). Fully discrete graph
geometry and the complete discrete-versus-continuous geometry comparison are
deferred.

## Active implementation boundary

| Item | Current state |
|---|---|
| Epoch-44 flat checkpoint | Frozen as the reproducible failed unconstrained baseline; no results or artifacts are replaced |
| Deterministic profile-geometry contract | Implemented additively in commit `f1a3cb7d3744869126b0589563c866f6b28f3ca9` |
| Neural category-conditioned constrained decoder | Not yet implemented |
| Successor repaired-flat checkpoint | Not yet trained |
| Typed dependency-graph model | Not yet implemented |
| Fully discrete graph geometry | Deferred |

The deterministic contract maps an explicit profile family plus compact
continuous parameters to valid controlled-profile geometry and supports exact
extraction for canonical targets. It does not predict profile family or
continuous parameters, does not integrate those predictions into the neural
decoder, and does not constitute a trained neural result.

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
- Implemented an additive deterministic profile-family/continuous-parameter
  geometry construction and extraction contract.

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

The repaired checkpoint has also passed a deterministic six-family Phase B
smoke in Slurm job `3327631`. Both complete publications were byte-identical.
The authoritative loader selected the same six validation IDs before
inference, loaded six validation records and zero train/test records, and
reported `test_partition_evaluated: false`. This is an engineering and
determinism result, not the final validation result; see the
[immutable smoke record](experiments/b0_phase_b_repaired_smoke.md).

A dedicated full-validation workflow extends the repaired-smoke checkpoint,
partition-isolation, paired-decoding, schema-2, and artifact-validation
contract to exactly all 68 validation families in a new namespace. Slurm job
`3327735` passed its repository, environment, checkpoint, partition, and
regression preflights, then rejected the unpublished artifact bundle because
container-side branch discovery returned no branch name. After that repair,
job `3329040` completed inference and independent artifact validation, then
failed while creating the hash manifest because shell `find` did not traverse
the stable symlink-backed evaluation publication. Reviewed manifest-only
recovery independently revalidated the namespace and atomically replaced only
the incomplete manifest with a verified 15-entry stable-path manifest. The
original scheduler state remains `FAILED 1:0`; see the
[immutable full-validation record](experiments/b0_phase_b_repaired_full_validation.md).

The recovered full-validation bundle is now frozen and manifest-verified
locally. Two preregistered read-only diagnostics were completed over it. The
categorical-isolation replay used 544 rows in a 2-by-2
reference-plane/profile-category design. Reference-plane correction produced
zero controlled-validity gain on either path. Authoritative profile category
plus category-conditioned profile projection produced gains of +32 / 68
teacher-forced and +34 / 68 predicted-history, with zero factorial
interaction. Its model-plane/oracle-profile arm reached 68 / 68
teacher-forced and 63 / 68 predicted-history. The remaining five
predicted-history failures were second-operation `EE` capsules with
`nonpositive_fitted_extent`. These are diagnostic counterfactuals and do not
change the frozen evaluation gates; see the
[immutable categorical-isolation record](experiments/b0_phase_b_categorical_isolation_replay.md).

## What has not been established

- The neural category-conditioned constrained profile decoder has not been
  implemented; only its deterministic profile-geometry contract exists.
- No successor repaired-flat checkpoint has been trained.
- No typed dependency-graph neural model has been implemented.
- No capacity-matched graph-versus-flat comparison has been run.
- The repaired epoch-44 B0 checkpoint produced a recovered, fully validated
  68-family validation result, but neither decoding path produced any
  controlled-domain-valid history; frozen Gates C and D did not pass.
- Executable OpenCascade reconstruction of repaired-model predictions has not
  been established.
- Latent-code semantics and localized learned edits have not been evaluated.
- Systematic-generalization model results and multi-seed comparisons do not
  yet exist.
- The held-out IID test partition has not been evaluated.

## Immediate scientific direction

The recovered full validation used five active codes with perplexity
`3.120410089936484`, so its frozen Gate B inputs pass. Its teacher-forced and
predicted-history controlled-domain-valid counts are both 0/68, and its Gate D
extrude and revolve counts are both zero. Gates C and D therefore do not pass.
Gate E/OpenCascade was not run, and the frozen report continues to record
formal final acceptance as undetermined.

The validation-only diagnosis attributes the observed controlled-validity
failure to profile-family categorical errors together with unconstrained
profile geometry, while finding no controlled-validity gain from
reference-plane category correction. The deterministic geometry contract for
the bounded intervention is now implemented additively. The next neural step
is to integrate explicit profile-family prediction and compact continuous
profile parameters so the predicted family selects the applicable
parameterization and the continuous output satisfies that manifold. The five
second-operation capsule failures remain mandatory adversarial cases.

The repaired flat model is timeboxed. Once the shared decoder interface is
frozen, implementation of the minimal typed graph model begins rather than
waiting for broad flat-model optimization. The required principal experiment
then compares repaired flat versus typed graph under that same decoder,
capacity-conscious controls, ordinary validation, one systematic split, and
one localized numerical-edit test. Fully discrete graph geometry is deferred.
The held-out test partition must remain untouched.

## Documentation maintenance

There is no remaining record backfill for a completed run whose primary
bundle has been recovered locally. The repaired-checkpoint deterministic
smoke, recovered full validation, and categorical-isolation diagnostic have
separate immutable records. The full-validation source bundle and both
diagnostic namespaces are locally manifest-verified.

Historical evidence marked `documented-external` in the inventory remains
eligible for integrity upgrade if its primary artifacts are recovered.
Documentation changes are checked with
`python3 tools/check_documentation.py`. This page continues to distinguish
checked-in implementation, verified run results, and conclusions that remain
unestablished.
