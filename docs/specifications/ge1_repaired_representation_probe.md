# GE1 Repaired-Checkpoint Representation-Probe Contract

## Status and authority

| Item | Proposed frozen value |
|---|---|
| Status | Accepted for implementation, synthetic testing, an unsubmitted runner, and bundle preparation; scientific execution separately unauthorized |
| Governing decision | Accepted [ADR-0011](../decisions/ADR-0011-ge1-repaired-representation-probe.md) |
| Protocol | `GE1-C7-REPAIRED-REPRESENTATION-PROBE-v1` |
| Source result | Immutable repaired-sufficiency commit `705eb820f7a15d64fe650df7350b1553b2bc8172`, job `3345280` |
| Source protocol | `GE1-C7-REPAIRED-SUFFICIENCY-v1` |
| Execution target | Adroit, Python 3.8.13, PyTorch 1.11.0, CPU only, one thread |
| Designated reviewer | Krishay Maskara; accepted August 12, 2026 |
| Stage 6 / C8 | Unauthorized / not begun |

This contract is additive and prospective. It must not alter, relabel, rerun,
or replace job `3345280`; change a GE1 model, loss, decoder, gate, cohort, or
checkpoint; or imply authorization for later scientific stages.

## Diagnostic question

Do the finalized epoch-200 models retain information that distinguishes the
five controlled extrusion magnitudes and five controlled revolve magnitudes
before the existing scalar geometry output, or has readily accessible class
information already been lost upstream?

The probe separates scalar-head calibration from decoder-state and
encoder-memory accessibility. It does not test systematic generalization,
architectural capacity exhaustion, or CAD-kernel executability.

## Immutable source evidence

Before feature extraction, the implementation must verify the complete source
artifact and these direct SHA-256 identities:

| Source file | SHA-256 |
|---|---|
| `artifact_manifest.json` | `63b495d245277bb6e3a181e122722def1cfd42ca6b05fe73cbaab2f5c90663e2` |
| `metrics.jsonl` | `39bd071bc682034d582a7cfffa6a225dddb3e2cf1902ddf68890545df21b148f` |
| `resolved_config.json` | `c54102bb4c801c8955b9495581732188a88317fd676457961d361e7c31afcd4f` |
| `SHA256SUMS` | `04c97c3d795c8dd757ab5ee1bcff5421abe67d83d9dd674f9155f27de55edbe8` |
| Scaled flat recovery checkpoint | `b61666a307c56a7b59a91671b2e0b06914bb1e76d4a0b48a60a0a501eb3cdb31` |
| Scaled flat inference checkpoint | `f7c37677b8e707bfc35f1dcfd61ad13ec118b77528053a2553af39f6b2520b28` |
| Scaled typed-graph recovery checkpoint | `fffdb0ead00c05aa6d99f3ee0e471cccec63040e659130ae03f74bbc0dd73ec9` |
| Scaled typed-graph inference checkpoint | `8b3733df88a8bd73851a2e7979a503e7c18ef75acbbd4b626f524657aba58cb5` |

The source artifact path is expected to be:

```text
/scratch/network/km6349/ge1_repaired_sufficiency_artifacts/ge1-repaired-705eb820f7a15d64fe650df7350b1553b2bc8172-3345280
```

Each source file and model-state hash must match both before and after the
diagnostic. The source artifact is mounted or read in place; nothing in it may
be modified.

## Cohort and statistical unit

Only the authoritative `operation_template` manifest and the exact scaled
train cohort already frozen in the source artifact are eligible:

- 32 physical families;
- eight each of `E`, `R`, `EE`, and `RE`;
- scaled-family hash
  `0888d517561ca24c99065465f8a2378e0e8e457f4bd20024e7f983956456f9ce`;
- 48 ordered operations: 32 extrusions and 16 revolves; and
- seed 2026.

The diagnostic must reproduce the source cohort identities and order rather
than select a new cohort. It chooses exactly the `continuous` representation
record as the one canonical record for each physical family and rejects a
missing or duplicate canonical record. It verifies the source loader's
continuous/quantized physical-target equivalence but does not use the second
variant as another statistical observation.

Physical family is the grouping and statistical unit. Every operation from a
held-out family remains held out together. Operation index is preserved, so
the two operations in `EE` and `RE` histories cannot be conflated.

## Execution and target-isolation order

The only permitted order is:

1. require the exact clean diagnostic commit, standalone checkout, Slurm job
   identity, and frozen CPU runtime;
2. rerun all preflight suites before any manifest, checkpoint, or payload
   access;
3. verify the immutable source artifact and checkpoint hashes;
4. verify the operation-template manifest using metadata only;
5. reconstruct and load only the frozen 32-family scaled train cohort;
6. strictly reload each scaled epoch-200 recovery and inference checkpoint
   into a fresh evaluation model;
7. run deterministic autonomous `P_true` inference under `torch.no_grad()`;
8. reproduce the source `P_true` scalar, node, graph, dependency,
   strict-conversion, analytic-validity, and operation-fidelity results before
   accepting any extracted feature;
9. extract and detach every feature listed below, write a target-free feature
   file, hash it, and verify unchanged GE1 model state;
10. destroy or release all GE1 model objects;
11. only then join target magnitude labels to family/operation keys in a
    separate label file; and
12. fit disposable probes using only the detached feature and label files.

Model inference and feature extraction must not accept target geometry,
target magnitude, target operation type, target masks, target labels, or a
target tensor. The autonomously predicted operation type is eligible because
the source result established it as exact, but the diagnostic must verify that
exactness again before label joining.

No GE1 parameter may have `requires_grad` used for this diagnostic. There is
no GE1 backward pass, optimizer, training call, checkpoint write, gradient
buffer, parameter update, or model-state mutation. Probe optimizers must be
constructed only after GE1 models are released and may contain only disposable
probe parameters.

## Feature definitions

Features are keyed by arm, physical family, canonical operation index, and
autonomously predicted operation type.

### A. Existing scalar

Record the active normalized operation magnitude after
`GE1-OPERATION-MAGNITUDE-POSITIVE-v1` and its scale-only physical value:
channel 37 times 4 for extrusion or channel 38 times 360 for revolve.

### B. Raw scalar logit

Record the corresponding active output of `remaining_geometry_head` before
the positive parameterization. The active predicted type and mask select the
channel; target type and target mask may not do so.

### C. Per-operation decoder state

Record the 32-dimensional shared-decoder hidden state for the autonomously
emitted operation node immediately before `remaining_geometry_head`. Do not
record a target-position state, a teacher-forced state, or a state selected by
target operation type.

### D. Continuous encoder bottleneck with autonomous query context

Flatten `encoded.prequant`, the continuous encoder bottleneck immediately
after `to_codebook` and before `from_codebook`, in latent-token-major order.
Its frozen shape is `[2, 16]`, so flattening yields 32 values. Concatenate a
two-element one-hot autonomously predicted operation type and a two-element
one-hot canonical operation slot (`0` or `1`), yielding 36 values. The
separate decoder-facing `encoded.memory` is `from_codebook(prequant)`, has
shape `[2, 32]`, and remains the only tensor passed into the shared decoder;
it is not flattened into D. Local tensor indices and graph offsets remain
bookkeeping; no chronological or absolute-position input is added to the
typed-graph encoder.

The two-by-sixteen prequant bottleneck is a continuous 32-value
representation. It must not be described as an eight-bit channel, and the
diagnostic assigns no independent-bit requirement to any program.

## Scalar analyses

Extrusion and revolve are analyzed separately. For both A and B, report
per-class scalar ranges and rank order. Because the positive mapping is
monotonic, the optimal monotonic remapping ceiling for A and B must agree;
disagreement is an implementation failure.

For feature A, report:

1. **Existing fidelity-gate accuracy:** fraction of operations satisfying the
   source unrounded physical-error rule (`< 0.25` extrusion, `< 22.5` revolve)
   and fraction of families whose every operation passes.
2. **Nearest-grid accuracy:** map the physical scalar to the closest of the
   five eligible physical targets. Exact distance ties choose the smaller
   magnitude. Report a five-by-five confusion matrix.
3. **Best monotonic-threshold recalibration:** choose four nondecreasing
   thresholds that maximize complete-cohort five-class accuracy while
   preserving class order. Candidate cuts are midpoints between distinct
   observed scalar values, with `-infinity` and `+infinity` boundaries. Dynamic
   programming must consider empty intervals. Ties select the
   lexicographically smallest threshold tuple. Report thresholds, confusion,
   accuracy, balanced accuracy, and per-class recall.
4. **Grouped threshold generalization:** repeat threshold fitting inside each
   leave-one-family-out training fold and apply only to that held-out family.

The complete-cohort optimal-threshold result is called the **scalar-only
resubstitution ceiling**. It is an upper bound only for monotonic post-hoc
remapping of A/B on these observations. It is not an upper bound for C or D.

## Probe splits and preprocessing

Fit extrusion and revolve models separately with the fixed five-class label
order from their controlled physical grids.

- Full-cohort resubstitution fits on every eligible operation and scores the
  same operations.
- Leave-one-physical-family-out (LOFO) holds every operation from one eligible
  family out together, fits only on other families, and predicts the held-out
  operations. All held-out predictions are concatenated once.
- Continuous-valued columns are standardized using mean and population
  standard deviation from the applicable training fold only. A zero-variance
  column maps to zero. One-hot type/slot columns are not standardized.
- No target-derived feature selection, dimensionality reduction, checkpoint
  selection, operation deletion, or outcome-dependent rerun is permitted.

## Mandatory linear probes

Fit two separate five-class multinomial softmax regressions per arm and
operation type:

- C-linear: decoder state only; and
- D-linear: continuous prequant bottleneck plus autonomous type/slot only.

The objective is mean cross-entropy plus
`0.5 * lambda * ||W||_2^2`; the intercept is unpenalized. Fit in float64 on
CPU from zero-initialized weights with deterministic full-batch PyTorch
L-BFGS, `max_iter=500`, `history_size=100`, `tolerance_grad=1e-9`,
`tolerance_change=1e-12`, and `line_search_fn="strong_wolfe"`. Nonfinite loss,
failure to terminate, or run-to-run disagreement is an implementation
failure, not a scientific result.

The regularization grid is:

```text
1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100
```

For each outer LOFO fold, choose lambda using inner LOFO over the remaining
physical families. Select maximum inner balanced accuracy, then maximum raw
accuracy, then the largest lambda. For the full-cohort resubstitution model,
choose lambda by LOFO over the complete cohort using the same rule, then refit
on all operations. The output space always contains all five classes even
when an inner fold temporarily lacks one class.

Expected learned parameter counts, excluding no state, are 165 for C-linear
(`5 * 32 + 5`) and 185 for D-linear (`5 * 36 + 5`). The implementation must
derive and report the actual counts rather than trust these expectations.

## Secondary nonlinear probes

Run C-MLP and D-MLP as fixed secondary diagnostics. Each has one tanh hidden
layer of width 8, a five-class output, no dropout or normalization, and an L2
penalty of `1e-2` on weight matrices only. Fit detached standardized features
with full-batch Adam, learning rate `1e-2`, exactly 2,000 updates, no early
stopping, no checkpoint selection, and deterministic seed 2026. Each outer
fold is freshly initialized from a seed derived solely from protocol seed,
arm, operation type, feature level, and held-out family ID.

Expected parameter counts are 309 for C-MLP and 341 for D-MLP. Because these
counts exceed the operation counts, resubstitution performance cannot by
itself support generalizable separation.

## Label-permutation controls

Every C/D linear and MLP pipeline must run 100 deterministic label
permutations in addition to the true labels. Permute family label blocks
within operation template and operation type, preserving per-template class
support, family grouping, and the ordered two-operation block for `EE`.
Each permutation reruns the complete preprocessing, regularization selection
where applicable, fitting, and LOFO scoring.

For raw and balanced LOFO accuracy, report null mean, standard deviation,
95th percentile using the nearest-rank definition, maximum, and empirical
one-sided p-value `(1 + count(null >= observed)) / 101`. The permutation seed
stream is SHA-256-derived from the protocol identity, seed 2026, arm, operation
type, feature level, probe kind, and permutation index.

## Required metrics

For every arm, operation type, feature level, and probe kind, report:

- selected regularization and deterministic seed information;
- derived parameter count;
- resubstitution and LOFO operation accuracy;
- five-by-five confusion matrices;
- per-class support and recall;
- balanced accuracy;
- scalar-only ceiling and difference from it where applicable; and
- label-permutation summaries.

Combine the independently predicted extrusion and revolve outputs to report:

- exact 48-operation accuracy; and
- physical-family all-operations-correct accuracy over all 32 families,

for both resubstitution and concatenated LOFO predictions. Families with two
operations pass only when both are correct.

## Prospective interpretation rules

A probe has **held-out-family accessibility evidence** only when its LOFO
balanced accuracy is at least `0.40`, its empirical permutation p-value is at
most `0.05`, and its LOFO raw accuracy exceeds the corresponding permutation
95th percentile. Failure of any condition is reported as no controlled
held-out-family accessibility evidence.

The decoder-state probe **materially exceeds the scalar ceiling** only when it
has held-out-family accessibility evidence and its LOFO raw and balanced
accuracies each exceed the corresponding scalar grouped-threshold result by
at least `0.10`. The complete-cohort scalar-only resubstitution ceiling remains
reported alongside this rule but does not cap the higher-dimensional probe.

Apply these conclusions separately for extrusion and revolve:

- C succeeds materially beyond scalar: accessible decoder-state information
  is being discarded by the existing scalar head/loss; a later categorical
  magnitude-head proposal becomes scientifically motivated, not authorized.
- D has held-out-family accessibility evidence while C does not: information
  is lost in decoder conditioning; investigate operation-specific
  conditioning before latent width.
- Linear lacks evidence while the matched MLP has it: information is present
  but not linearly accessible; a small nonlinear operation-specific head may
  be sufficient.
- Every controlled probe lacks evidence: these regression-trained checkpoints
  do not retain readily accessible magnitude-class information. This does not
  prove capacity exhaustion.
- Resubstitution is strong but LOFO lacks evidence: report memorization or
  within-cohort accessibility only, not generalizable separation.

No finding permits an inference about development, RR, ER, systematic
generalization, CAD execution, or comparative encoder superiority.

## Read-only geometry-loss audit

The artifact must include a source-derived analytic audit, not a new loss
experiment.

`_per_example_smooth_l1` calls PyTorch `smooth_l1_loss` without overriding
`beta`, so under PyTorch 1.11 the beta is exactly `1.0`. With float32
`epsilon = torch.finfo(torch.float32).tiny`, the repaired normalized operation
prediction is bounded from `epsilon` through `1.0`. The normalized target
grids are:

- extrusion: `0.125, 0.25, 0.375, 0.5, 0.75`;
- revolve: `0.125, 0.25, 0.5, 0.75, 1.0`.

Therefore the possible signed normalized-error envelopes are
`[epsilon - 0.75, 0.875]` for extrusion and
`[epsilon - 1.0, 0.875]` for revolve. Errors remain in the Smooth-L1
quadratic region or exactly at its beta boundary; they do not enter the
strictly linear-above-beta region.

The loss first takes a masked mean per physical example, then a batch mean.
With geometry-loss weight `1.0`, the exact operation-channel coefficients
inside each example's remaining-geometry component are:

| Template | Active remaining channels | Per-operation coefficient | Total operation coefficient |
|---|---:|---|---:|
| E | 1 | extrusion `1` | `1` |
| R | 5 | revolve `1/5` | `1/5` |
| EE | 2 | each extrusion `1/2` | `1` |
| RE | 6 | revolve `1/6`; extrusion `1/6` | `1/3` |

The other four channels in R/RE are axis geometry. Because the scaled cohort
has eight families per template, the cohort-average coefficient on all
operation-channel errors is `19/30`; extrusion contributes `13/24`, revolve
`11/120`, and axis `11/30`. Viewed per operation rather than per family, the
mean coefficients are `13/24` over 32 extrusions, `11/60` over 16 revolves,
and `19/45` over all 48 operations.

The audit must distinguish this per-template masked-mean dilution from the
equal-template cohort average. Neither weighting choice may be called a defect
without quantitative probe or intervention evidence.

## Artifact and provenance contract

The proposed diagnostic artifact contains only:

- `resolved_config.json`;
- `detached_features.pt`, containing A-D plus autonomous family/operation keys
  but no targets or labels;
- `feature_manifest.json`, written and hashed before label joining;
- `labels.json`, containing only family/operation keys and controlled
  magnitude-class labels;
- `scalar_analysis.json`;
- `probe_results.json`;
- `metrics.jsonl`;
- `artifact_manifest.json`; and
- `SHA256SUMS`.

It contains no CAD history, raw corpus payload, target tensor, reconstructed
history, model state, optimizer state, source checkpoint, gradient, or CAD
kernel output. Every regular file is inventoried by path, byte size, and
SHA-256. Resolved configuration and terminal metrics explicitly declare all
access, immutable source identities, feature dimensions, split membership,
probe parameters, non-authorization state, and source-model hashes before and
after extraction.

## Access and non-authorization boundary

Only `operation_template.train` metadata and the frozen 32-family scaled train
payload are eligible. Development, RR systematic, ER test, IID,
history-depth, geometry-extrapolation, and every other corpus remain closed.
There is no CAD kernel.

The diagnostic does not authorize or perform scientific-model training,
Stage 6, C8, another repair, categorical-head implementation, loss
reweighting, FiLM conditioning, latent-token or bottleneck changes, checkpoint
writing, or protected access. A completed negative diagnostic is a valid
result and exits zero after artifact finalization. Infrastructure or source-
reproduction failures exit nonzero and have no scientific interpretation.

ADR-0011 received explicit designated-reviewer acceptance on August 12, 2026.
Implementation and synthetic corpus-free validation may proceed. Scientific
execution, checkpoint loading for diagnosis, corpus access, and Slurm
submission remain separately unauthorized even after implementation
validation.
