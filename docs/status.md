# Current Project Status

This page is the authoritative concise statement of the project's present
state. Detailed implementation contracts live in package READMEs, completed
run evidence lives under `docs/experiments`, and future protocols belong in
specifications or new decision records.

## Repository state

| Item | Current value |
|---|---|
| Active branch | `graph-v1-experiment-record` |
| Current reviewed revision | Repaired-sufficiency job `3345280` remains an immutable valid train-only scientific failure; ADR-0011 job `3346513` and representation-screen job `3350017` timed out without finalized scientific results; accepted additive ADR-0012 implements a closed-form artifact-only readout and unsubmitted one-hour runner; no readout execution, accessibility claim, repair, Stage 6, or C8 is authorized |
| Frozen Flat V6 commit | `ac6ef718ae9bab7fa5a80d9f48d0976adf5cafad` |
| Frozen initial Graph V1 commit | `089b9f3d0e5a61fb19ef3fa05e993fc4eceffdcb` |
| Frozen final Graph V1 C1 commit | `0cd09ed34d4c4dd0d43e1456b7a06eb362ee7962` |
| Status recorded | August 16, 2026 |
| Controlled domain | Single-body sketch, extrude, and revolve histories with one or two operations |
| Authoritative corpus | 680 physical families: 544 train, 68 IID validation, 68 held-out IID test |
| Authorized GE1 manifest | Operation-template only: 407 train, 45 development, 114 RR systematic, 114 ER test |
| GE1 protocol record | `GE1-STAGE0-PREREG-v1` plus accepted ADR-0008, ADR-0009, and ADR-0010; C0-C6 complete; formal C7-v1, C7-v2, and repaired sufficiency all failed scientifically; job `3345280` preserved exact structure/validity/memory but failed scaled operation fidelity in both arms; Stage 6 remains unauthorized; C8 and later work not begun |
| GE1 C3 authoritative validation | Passed on Adroit CPU as job `3344235` at exact commit `a85ad3a23a6587cbedad8a6693b6117c1edfacc7` |
| GE1 C4 validation | Passed on Adroit CPU at exact commit `e66cd089462c2e075b9ff742e157c21e4d9a2a6e`: 26/26 targeted and 85/85 complete-suite tests |
| GE1 C4 review-fix revalidation | Passed as Adroit job `3344265` at exact commit `3a41abc81f68ef6d6450465e05b3544f07a08b83`: 2/2 new, 28/28 C4 encoder, and 88/88 complete-suite tests, all with zero skips |
| GE1 C5 authoritative validation | Passed as Adroit job `3344290` at exact commit `996016df44b7f9a6cd5c092a3e3b7a87d9964f9d`: 22/22 focused and 110/110 complete-suite tests, both with zero skips |
| GE1 C6 validation attempts | Jobs `3344337` and `3344363` failed before corpus access. The corrected second attempt at exact commit `68c66ea4b4c1e1d01b9b9dc061ace1149bca7c5a` passed 36/37 focused tests with zero skips; its sole error was a test-only tensor/dataclass equality assertion |
| GE1 C6 authoritative validation | Passed as Adroit job `3344367` at exact commit `c88e967b96dd201fedcd495fa8d5ddaddd02caf3`: 37/37 focused and 147/147 complete-suite tests with zero skips, followed by both 407-family train-only arm smokes |
| GE1 C5/C6 review-fix revalidation | Passed as Adroit job `3344431` at exact commit `d29dc8299d32186907eb16d05c6102fcececf32e`: 75/75 focused and 163/163 complete graph-encoder tests with zero skips, every regression and repository gate, and both v2 407-family train-only arm smokes |
| GE1 C7 formal pilot | Adroit job `3344505` at exact commit `4bfde4c726a585433ea4bb60e6ce9d245ae7c87d` completed with valid artifacts and failed both tiny autonomous exact gates; all scaled gates were `not_run`, repair was triggered procedurally, and Stage 6 was not authorized |
| GE1 post-C7 optimization diagnostic | Job `3344907` at exact commit `fbc6073f63da9f0e10b5db8c0c0d4786a48ce0c0` completed; both unchanged arms first passed autonomous exact sufficiency at update 200, yielding `undertraining_supported_both_arms`; Stage 6 was not authorized and protected partitions remained closed |
| GE1 C7-v2 execution | Job `3344981` at exact commit `e325d5ad97957c08da4a19b4261560e8a4a472a4` completed as a valid scientific failure: both tiny gates passed; scaled memory gates passed; flat failed 2/32 and typed graph failed 3/32 only on analytic complete validity; Stage 6 remained unauthorized and the repair path became procedurally next |
| GE1 C7-v2 preflight attempt | Job `3344975` at exact commit `99a4587d73e82a5df5f12a54130b105ae264b419` failed in the sequential regression loader before manifest or payload access; no scientific execution or artifact occurred |
| GE1 operation-parameter diagnostic | Adroit job `3345013` at exact diagnostic commit `802ae1d1e9deb3c7a6c428d320e4276a8b5e7e57` completed read-only: all five failures were active negative extrusion/revolve magnitudes; no training, repair, checkpoint write, protected-partition access, or C7-v2 mutation occurred |
| GE1 prospective magnitude repair | Accepted ADR-0009 superseded only the earlier hierarchical repair authorization; `GE1-OPERATION-MAGNITUDE-POSITIVE-v1` passed engineering validation and was exercised by finalized repaired-sufficiency job `3345280`, while Stage 6 remains unauthorized |
| GE1 magnitude-repair validation | Passed as Adroit CPU job `3345044` at exact commit `12167ce7d0dc025c4b297b7ccb8a3011580bf0fc`: 13/13 focused and 292/292 complete graph-encoder tests with zero skips, all regressions and repository gates, and zero corpus/training/kernel access |
| GE1 repaired sufficiency protocol | Accepted [ADR-0010](decisions/ADR-0010-ge1-repaired-train-sufficiency-protocol.md) froze `GE1-C7-REPAIRED-SUFFICIENCY-v1`; authoritative job `3345280` completed with both tiny arms passing, both scaled exact and memory gates passing, both scaled operation-fidelity gates failing, and Stage 6 unauthorized |
| GE1 repaired sufficiency implementation | Passed authoritative Adroit CPU validation as job `3345161` at exact commit `6b62cab90ea8f693cde41c2abf0a261024057d45`: 32/32 focused and 324/324 complete graph-encoder tests with zero skips, every regression/repository gate, and zero manifest/corpus/scientific access; subsequent scientific job `3345280` is recorded separately above |
| GE1 repaired representation probe | Accepted [ADR-0011](decisions/ADR-0011-ge1-repaired-representation-probe.md) is implemented as `GE1-C7-REPAIRED-REPRESENTATION-PROBE-v1` with pure and real-PyTorch synthetic tests plus an unsubmitted exact-commit CPU runner; local validation is corpus-free and real-PyTorch cases remain for authoritative Adroit preflight; checkpoint loading, corpus access, execution, repair, Stage 6, C8, pushing, and protected access remain unauthorized |
| GE1 representation-probe preflight attempt | Adroit job `3346476` at exact commit `4baf596812471ca76a15b09df89ce38d0a4bf30c` failed after 17/17 pure and 6/7 real-PyTorch focused tests because the extractor conflated the `[B,2,32]` decoder memory with the frozen `[B,2,16]` prequant probe bottleneck; no source artifact, checkpoint, manifest, payload, scientific probe, training, or later-stage work was accessed or performed, and no artifact exists |
| GE1 representation-probe timeout | Job `3346513` at exact commit `3ea43650bb793d8d55b3d61a1edb70ec7a920089` timed out during its large workload; its three verified preserved files contain the complete 96-row target-free feature and 48-row label stage, but no finalized ADR-0011 result, significance analysis, or authority |
| GE1 preserved-feature representation screen | `GE1-C7-REPAIRED-REPRESENTATION-SCREEN-v1` is implemented as a separate 16-pipeline non-permuted real-label screen with a one-hour artifact-only CPU runner; implementation validation and bundle preparation do not authorize submission, formal accessibility claims, repair, Stage 6, C8, corpus/checkpoint access, or relabeling job `3346513` |
| GE1 representation-screen timeout | Job `3350017` at exact commit `8a1ce734abfdbdb7468d6900a0f34a4b914608b5` timed out: state `TIMEOUT`, elapsed `01:00:15`, batch elapsed `01:00:16`, batch state `CANCELLED`, batch exit `0:15`, MaxRSS `622608K`; 12/12 focused and 361/361 graph-encoder tests plus the 86/549/17/40/51 regression suites, documentation, compilation, source audit, and import/export audit passed; its incomplete directory was empty and no finalized result or conclusion exists; preserved-input content access is `indeterminate_not_terminally_certified` because staging precedes loading and no terminal telemetry survived |
| GE1 closed-form v1 infrastructure failure | Adroit job `3351501` at exact commit `912e078f216f82ad37c00198adc20dd93eb647c6` failed rather than timed out: state `FAILED`, exit `1:0`, elapsed `00:06:28`, MaxRSS `611492K`; 34/34 focused and 395/395 graph-encoder tests, the 86/549/17/40/51 regression suites, documentation, compilation, and source audits passed; B=999 projected at `671.2964434385067` seconds; all three frozen inputs were hash-verified and loaded before independently fitted nonlinear-coordinate A/B midpoints disagreed; logs, timing record, and the incomplete directory remain audit evidence; no finalized result or scientific interpretation exists, and no corpus/checkpoint/model/inference/training/repair/protected partition/Stage 6/C8 access or execution occurred |
| GE1 closed-form representation readout | Accepted [ADR-0012](decisions/ADR-0012-ge1-closed-form-representation-readout.md) now freezes `GE1-C7-CLOSED-FORM-READOUT-v2`: its common A/B baseline uses one finite-precision weak order, tie collapse, strict-inversion rejection, and ordered fold anchors rather than numeric coordinate midpoints; separate A/B analyses remain descriptive; the 24 ridge pipelines, explicit revolve degeneracies/non-applicabilities, exactly 16 primary synchronized-permutation hypotheses, metric-specific centered maxT, and 999/499/abort one-hour CPU gate are unchanged; ADR-0011 and the screen remain incomplete, and no v2 authoritative job has been submitted |
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
- Authoritative C3 compatibility validation under Python 3.8.13 and PyTorch
  1.11.0 on Adroit CPU, including an unskipped real-PyTorch tensor smoke.
- C4 continuous flat wrapper and position-free typed graph encoder with ten
  directed relation channels, three two-basis layers, graph-local pooling,
  explicit VQ bypass, and a common encoded-memory contract.
- C4 arithmetic capacity match: 22,800 flat versus 23,468 graph encoder
  parameters, a 2.93% difference relative to the flat control.
- C5 shared decoder extraction, thin common-memory model wrapper, common
  per-example loss, frozen output-position inventory, autonomous
  raw/constrained/converted result contract, and strict model checkpoint.
- C6 common deterministic training loop, atomic optimizer/RNG recovery,
  provenance revalidation, target-free three-condition autonomous inference,
  executable-prefix and family-macro metrics, parameter/receptive-field
  reports, and timing/peak-memory measurement, authoritatively validated under
  Python 3.8.13 and PyTorch 1.11.0 on Adroit CPU.

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
templates. Its C3 production-environment compatibility gate passed on Adroit
CPU as job `3344235`; the detailed evidence and provenance limitations are in
the [C3 validation record](experiments/ge1_c3_cpu_validation.md). C4 now
implements the two standalone continuous encoders with bookkeeping restricted
to addressing, membership, scatter/gather, masking, and graph-local pooling.
Its authoritative Adroit CPU validation passed all targeted and complete-suite
tests; the exact-commit evidence and environment limitations are recorded in
the [C4 validation record](experiments/ge1_c4_cpu_validation.md).

A subsequent C4 implementation review produced four accepted fixes: the Stage 3
parity reference now names the width-192 capacity-matched flat implementation
rather than original Flat V6; relation-basis count and per-arm feed-forward
width are frozen consistently in configuration and documentation; genuinely
shared components are now initialized identically across arms from one
arm-independent source; and the capacity rationale records the corrected
inherited-width arithmetic. Those changes altered encoder construction, so
they received a separate authoritative Python 3.8.13/PyTorch 1.11.0 Adroit CPU
validation as job `3344265`. Both new shared-initialization tests, all 28 C4
encoder tests, and all 88 graph-encoder tests passed with zero skips; the
exact-commit evidence is in the [C4 review-fix validation
record](experiments/ge1_c4_review_fix_cpu_validation.md).

C5 now connects either C4 encoder to the same decoder class and forward
implementation. Its canonical decoder is constructed once and copied into
independent arm models with matched initial state and disjoint mutable objects.
The retained constrained V6 node/geometry path and initial Graph V1 main
typed-edge MLP define the component-parity boundary; the later C1 additive
position-bias branch is excluded. The exact four-signal output-position
inventory, permitted bookkeeping routes, target-free autonomous result,
common loss, and strict checkpoint are frozen in the
[C5 shared-decoder contract](specifications/ge1_shared_decoder_contract.md).

The first authoritative C5 attempt, Adroit job `3344278` at exact commit
`ffa6091cfecb37aa648ed3590f591f247a776754`, ran all 22 focused tests with zero
skips and zero errors: 19 passed and three assertions failed. Audit traced the
failures to test-contract mistakes involving legacy VQ provenance, outdated
conversion-result field names, and a batch-row-order assumption. No production
decoder or model source changed. The [failed-attempt
record](experiments/ge1_c5_cpu_validation_attempt_3344278.md) preserves the
exact evidence and protected-access audit.

Corrected exact commit `996016df44b7f9a6cd5c092a3e3b7a87d9964f9d`
passed authoritative Adroit job `3344290` under Python 3.8.13 and PyTorch
1.11.0 on CPU. All 22 focused C5 tests and all 110 complete graph-encoder tests
passed with zero skips. Every listed regression, documentation, compilation,
grammar, exact-commit, and clean-tree gate also passed. The [C5 validation
record](experiments/ge1_c5_cpu_validation.md) retains the exact evidence and
limitations. C5 is complete. Initial C6 Adroit job `3344337` stopped in the
focused suite: three tests
exposed the same tuple-versus-local-long-tensor autonomous node-count adapter
defect, and one test lacked its `GraphEncoderError` import. The focused
correction changed only those two points. Corrected exact-commit job `3344363`
then passed 36/37 focused tests with zero skips and confirmed both prior fixes.
Its sole error was a generic test comparison that asked Python to reduce a
multi-element tensor equality to one Boolean. After the test-only tensor-aware,
timing-neutral correction, exact-commit job `3344367` passed all 37 focused C6
tests and all 147 graph-encoder tests with zero skips, every regression and
repository gate, and both two-epoch 407-family train-only arm smokes with
strict reload and complete metrics records. The [C6 validation
record](experiments/ge1_c6_cpu_validation.md) retains the evidence and
limitations. C6 is complete. Development, RR, ER, C8, and later work have not
begun.

A subsequent C5/C6 implementation review produced five accepted fixes. The
reporting and artifact-version decisions are governed by accepted
[ADR-0005](decisions/ADR-0005-ge1-primary-reporting-and-metrics-v2.md), which
leaves the frozen `GE1-STAGE0-PREREG-v1` record unchanged. The inherited V6
entry-point compatibility shim is now named, documented, and
tested, with its corrected GE1 provenance pinned and the deliberate closure of
the inherited V6 autonomous conversion path recorded; decoder behavior is
unchanged and the frozen inherited V6 implementation is untouched. Every
primary result must now publish `P_true`, `P_shuffle`, `P_mean`, `R_shuffle`,
and `R_mean`, enforced as finite values or reason-bearing structured nulls by
`validate_primary_report`, because the shared
decoder's four output-side position signals and the frozen 59/68 position-only
prior make a primary number uninterpretable without memory evidence. New
artifacts use `GE1-C6-METRICS-v2` and the train-only smoke identity v2. The
prefix validation envelope is documented and tested field by field, separating
preserved semantic content from exactly recomputed converter bookkeeping, with
no scoring change. Donor/recipient template agreement is available for
`P_shuffle` against the random-distinct-donor baseline; `P_true` and `P_mean`
carry structured unavailable records while the frozen cyclic derangement is
retained unchanged.
The canonical decoder's seed check is an explicit authorized-seed validator
rather than a discarded configuration call. Those changes altered
`shared_decoder.py`, `metrics.py`, `model.py`, `config.py`, `c6_smoke.py`, and
the combined review-validation runner. Exact-commit Adroit CPU job `3344431`
passed all 75 focused C5/C6 tests and all 163 graph-encoder tests with zero
skips, every regression and repository gate, and both 407-family train-only
arm smokes. Both artifacts used `GE1-C6-METRICS-v2` and
`GE1-PRIMARY-REPORT-v1`; all five intervention fields and the required
donor-agreement availability states were present. The [combined review-fix
validation record](experiments/ge1_c5_c6_review_fix_cpu_validation.md)
retains the exact evidence and limitations. Current C5/C6 source is therefore
authoritatively covered.

Accepted
[ADR-0006](decisions/ADR-0006-ge1-c7-sufficiency-execution-contract.md)
now freezes the C7 execution details without changing Stage 0, ADR-0004, or
ADR-0005. The checked-in C7 implementation deterministically selects nested
4/32-family `E/R/EE/RE` cohorts from authoritative manifest metadata, builds
fresh matched seed-2026 model pairs per subset, trains only under the frozen
50-epoch recipe, requires strict epoch-50 autonomous exactness, applies the
scaled 0.80 memory-use gates, and publishes a self-verifying atomic artifact.
Its focused tests use only synthetic metadata, procedural fixtures, and
temporary artifacts. The formal train-only pilot then completed as Adroit job
`3344505` at exact commit
`4bfde4c726a585433ea4bb60e6ce9d245ae7c87d`. Both arms failed the
four-family autonomous exact gate; all four scaled gates were correctly
`not_run`, `overall_gate_pass=false`, and
`stage6_authorized_by_c7=false`. The still-decreasing tiny training losses
leave optimization sufficiency unresolved. The
[C7 record](experiments/ge1_c7_pilot.md) retains the raw-evidence hashes,
scientific interpretation, and the non-scientific Slurm-provenance
discrepancy. Development, RR, ER, IID, history-depth, and
geometry-extrapolation data remain unopened by C7. The hierarchical repair
was triggered procedurally but is not implemented or invoked, and C8 has not
begun.

Accepted
[ADR-0007](decisions/ADR-0007-ge1-c7-optimization-sufficiency-diagnostic.md)
added a narrow post-C7 optimization-sufficiency diagnostic because both tiny
losses were still improving substantially at update 50. After two preflight
failures before corpus access, corrected job `3344907` completed at exact
commit `fbc6073f63da9f0e10b5db8c0c0d4786a48ce0c0`. Both unchanged fresh
matched arms first became autonomously exact-sufficient at update 200 and
remained exact at update 500, so the frozen interpretation is
`undertraining_supported_both_arms`. The diagnostic did not change C7-v1,
authorize Stage 6, invoke repair, begin C8, or open a protected partition. The
[diagnostic record](experiments/ge1_optimization_diagnostic.md) retains the
audited hashes, trajectories, provenance, and limitations.

Accepted
[ADR-0008](decisions/ADR-0008-ge1-c7-v2-200-epoch-protocol.md) prospectively
establishes separately versioned C7-v2 identities and a fixed 200-epoch
budget for tiny and scaled gates. The same 200 epochs become the eventual
Stage 6 budget only if a finalized C7-v2 artifact explicitly records
`stage6_authorized_by_c7_v2=true`. C7-v1 and the diagnostic remain immutable.
The hierarchical repair path remains triggered in C7-v1 history and was
deferred while C7-v2 tested the evidence-based budget. The first authoritative
preflight, job `3344975`, passed the
environment, focused 28/28, complete graph-encoder 266/266, and model-data
86/86 gates, then stopped before flat-baseline discovery because Python 3.8's
singleton `defaultTestLoader` retained a prior discovery root. The runner now
constructs a fresh loader with explicit repository top level per regression
suite. The attempt opened no manifest or payload and produced no artifact;
see its [preflight record](experiments/ge1_c7_v2_preflight_attempt_3344975.md).

Corrected job `3344981` then completed C7-v2 at exact commit
`e325d5ad97957c08da4a19b4261560e8a4a472a4`. Both tiny arms passed before
scaled access. On the 32-family scaled cohort, both memory gates passed and
all node, graph, applicable `depends_on`, and strict-conversion criteria were
exact. Flat failed complete analytic validity on two families and typed graph
failed it on three, all with `invalid_operation_parameter`. The finalized
result is an overall C7-v2 failure, records
`stage6_authorized_by_c7_v2=false`, and makes the preauthorized repair path
procedurally next without implementing it.

Before any repair, Krishay Maskara authorized a narrower read-only
[operation-parameter diagnostic](specifications/ge1_c7_v2_operation_parameter_diagnostic.md).
It reuses the immutable scaled epoch-200 checkpoints and train cohort,
requires exact `P_true` reproduction, and records scalar and analytic traces
for both arms on the five failures. Strict conversion and analytic
controlled-domain validity are separate; CAD-kernel executor results are
structurally unavailable, and legal-but-geometrically-incompatible outcomes
are unassessable. Adroit job `3345013` completed that read-only diagnostic at
exact commit `802ae1d1e9deb3c7a6c428d320e4276a8b5e7e57`. It reproduced the
immutable C7-v2 evidence and isolated a negative active operation magnitude
in every failed case; masks, channels, units, normalization, direction,
Boolean mode, targets, and alignment remained consistent.

Accepted
[ADR-0009](decisions/ADR-0009-ge1-positive-operation-magnitude-repair.md)
therefore supersedes only the previously authorized hierarchical structural
repair. The prospective decoder mode
`GE1-OPERATION-MAGNITUDE-POSITIVE-v1` maps only extrusion-distance and
revolve-angle logits into `(0, 1]`; every other decoder output remains on the
legacy path. Historical `tanh` behavior remains explicitly versioned and
checkpoint-incompatible with the repaired configuration. This is an
implementation-validation step, not C7-v2, Stage 6, C8, or a repaired
scientific run. RR remains blocked until the accepted one-time systematic
stage, and ER remains closed throughout GE1.

Exact-commit Adroit CPU job `3345044` authoritatively validated that
prospective implementation under Python 3.8.13 and PyTorch 1.11.0. All 13
focused repair tests and all 292 graph-encoder tests passed with zero skips;
every regression and repository gate passed with only the four established
flat-baseline skips. The runner opened no manifest or payload, performed no
training, wrote no checkpoint, used no CAD kernel, and did not begin Stage 6
or C8. The [validation record](experiments/ge1_operation_magnitude_repair_cpu_validation.md)
preserves the checksums, coverage, decision, and limitations. This is
engineering validation only and does not authorize a repaired scientific
protocol.

Exact-commit Adroit CPU job `3345161` then authoritatively validated the
separately versioned repaired-sufficiency implementation at
`6b62cab90ea8f693cde41c2abf0a261024057d45`. All 32 focused tests, including
all eight real-PyTorch tests, and all 324 graph-encoder tests passed with zero
skips. Every regression and repository gate passed with only the four
established flat-baseline skips. Procedural fixtures intentionally exercised
bounded backward passes and optimizer steps; no manifest, corpus, scientific
training, scientific checkpoint or artifact, CAD kernel, Stage 6, or C8 was
opened or performed. The [validation record](experiments/ge1_repaired_sufficiency_cpu_validation.md)
preserves the evidence and reviewer clarification. This completes
implementation validation and makes the protocol eligible for a separate
execution authorization; it does not itself authorize submission or training.

The separately authorized repaired-sufficiency execution then completed on
Adroit as job `3345280` from exact clean commit
`705eb820f7a15d64fe650df7350b1553b2bc8172`. Both tiny arms passed exactness
and operation fidelity before scaled access. On the 32-family scaled train
cohort, both arms passed exact node, graph, applicable dependency, strict-
conversion, analytic-validity, and memory-use gates. Flat passed 18/32
extrusions and 9/16 revolves under the strict operation-fidelity thresholds;
typed graph passed 17/32 and 4/16. The completed artifact therefore records a
valid scientific failure, `comparison_inconclusive=false`, and
`stage6_authorized_by_repaired_sufficiency=false`. Development, RR, ER, IID,
history-depth, geometry-extrapolation, and other corpora remained unopened;
no CAD kernel, Stage 6, C8, or additional repair was used.

Accepted [ADR-0011](decisions/ADR-0011-ge1-repaired-representation-probe.md)
and its [representation-probe contract](specifications/ge1_repaired_representation_probe.md)
ask whether magnitude-class information remains accessible in the immutable
decoder states or continuous encoder bottlenecks. Krishay Maskara accepted
both records on August 12, 2026. Feature D uses the detached `[B,2,16]`
`encoded.prequant` bottleneck; the separate `[B,2,32]` `encoded.memory`
continues unchanged into the shared decoder. The first Adroit preflight, job
`3346476`, exposed and stopped on the earlier shape conflation before any
artifact or data access. The additive implementation, pure synthetic contract
suite, real-PyTorch synthetic suite, and unsubmitted CPU runner remain the
only authorized scope. No local corpus, source checkpoint, or repaired
artifact was opened. Bundle preparation is authorized; checkpoint loading,
corpus access, probe execution, Slurm submission, and any repair remain
blocked.

The subsequent ADR-0011 job `3346513` preserved the complete target-free
feature and label stage but timed out before its permutation-controlled
analysis finalized. It remains incomplete. The separate
[representation-screening contract](specifications/ge1_repaired_representation_screen.md)
uses only those three hashes and the existing 16 non-permuted real-label
pipelines. Its descriptive 0.40/0.40 LOFO flag is not significance evidence
and cannot authorize a repair or later stage.

That screen was attempted as Adroit job `3350017` at commit
`8a1ce734abfdbdb7468d6900a0f34a4b914608b5` and timed out after `01:00:15`
(`01:00:16` batch elapsed, batch `CANCELLED`, exit `0:15`, MaxRSS `622608K`).
All preflights completed, but the incomplete artifact directory was empty and
no scientific result exists. Because the module creates staging before input
loading and Slurm produced no terminal telemetry, preserved-input content
access is `indeterminate_not_terminally_certified`. Mount restrictions still
establish that no corpus, checkpoint, model, protected partition, training,
repair, Stage 6, or C8 path was available.

Accepted [ADR-0012](decisions/ADR-0012-ge1-closed-form-representation-readout.md)
and its [closed-form contract](specifications/ge1_closed_form_representation_readout.md)
replace neither incomplete protocol. They add a five-output closed-form ridge
readout over only the three preserved hashes, with slot-gated extrusion
prequant, structurally reduced revolve features, synchronized family-block
permutations, 16-primary-hypothesis centered maxT, and a pre-input synthetic
999/499/abort timing gate. Implementation, synthetic validation, and an
unsubmitted runner are authorized; submission and scientific execution are
not.

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
