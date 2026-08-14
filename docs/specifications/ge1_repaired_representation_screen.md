# GE1 Repaired Representation-Screening Contract

## Status and authority

| Item | Frozen value |
|---|---|
| Status | Authorized for implementation, synthetic validation, an unsubmitted runner, and exact-commit bundle preparation |
| Protocol | `GE1-C7-REPAIRED-REPRESENTATION-SCREEN-v1` |
| Preserved feature source | Exact commit `3ea43650bb793d8d55b3d61a1edb70ec7a920089`, timed-out job `3346513` |
| Runtime target | Adroit, Python 3.8.13, PyTorch 1.11.0, CPU only, one thread, one hour |
| ADR-0011 | Remains accepted and incomplete; this screen neither completes nor replaces it |
| Stage 6 / C8 / repair | Unauthorized / not begun / unauthorized |

This separately identified preliminary screen uses already detached features.
It does not rerun inference, feature extraction, source-result reproduction,
scalar A/B analysis, GE1 training, or label permutations. A positive flag is
descriptive prioritization only. A negative screen is valid and must still
finalize successfully.

## Diagnostic question

> Do C or D appear to contain five-class magnitude information that
> generalizes across held-out physical families under the exact real-label
> probe models?

The screen cannot establish statistical significance, formal representation
accessibility, encoder superiority, or authority for a model change.

## Immutable inputs

The runtime may read only these three preserved job-`3346513` files:

| File | SHA-256 |
|---|---|
| `detached_features.pt` | `87dab4841f64f0c1b27db6676852da7cd980ba1bde61e81809fde3af745cc01c` |
| `feature_manifest.json` | `29220bda0677636627069ce6565bf367065f1ceaa3c2d86d3174058838598450` |
| `labels.json` | `70f82727ebe35613bcbae1267924d9a327d9744a757b6d076c8028e261cfe71f` |

Before constructing a disposable probe, verify all hashes and schemas; 96
target-free feature rows; 48 unique labels; 32 physical families; one row per
arm per label; 32 extrusion and 16 revolve operations; A/B/C/D dimensions
1/1/32/36; finite values; family, operation-index, and operation-type
alignment; D's predicted-type and canonical-slot suffix; and absence of model
or optimizer state. Any difference fails before probe construction.

Job `3346513` remains a timeout with no finalized probe result. Its preserved
files establish only that the feature/label stage completed.

## Frozen pipelines

Run production `fit_probe_pipeline` over the Cartesian product of:

- arms `flat` and `typed_graph`;
- operation types `extrude` and `revolve`;
- features C and D; and
- multinomial-linear and width-eight-tanh-MLP probes.

This gives exactly 16 pipelines. Supply no test limits or reduced settings.
Preserve physical-family LOFO, nested family-LOFO linear regularization,
train-fold-only standardization, the accepted seven-value regularization
grid, float64 zero-initialized L-BFGS linear probe, deterministic 2,000-update
MLP, seeds, tolerances, and parameter counts 165/185 and 309/341.

`run_permutation_controls` must not be imported or called. No scalar analysis,
combined accessibility decision, or formal ADR-0011 interpretation is part of
the screen.

## Screening flag

Set `candidate_for_full_controls=true` only when:

```text
family-LOFO accuracy >= 0.40
and
family-LOFO balanced accuracy >= 0.40
```

Permutation significance must be recorded as unavailable. The flag does not
support accessibility, comparison, Stage 6, C8, a categorical head, loss
reweighting, calibration, or repair.

## Artifact contract

A completed screen writes exactly:

```text
resolved_config.json
screening_results.json
metrics.jsonl
artifact_manifest.json
SHA256SUMS
```

Results include both source identities, input hashes, execution commit and
Slurm job, environment, class support, all 16 production pipeline records,
linear regularization selections, MLP seeds, fit/fold evidence,
resubstitution and LOFO metrics, flags, and explicit non-authorization.
Missing pipelines prevent finalization. A negative complete screen exits zero.

## Access and runtime boundary

The runner uses one CPU thread, no GPU, and one hour. Preflights run before
the preserved directory is mounted. Scientific execution mounts only the
exact clean source checkout, preserved input read-only, and fresh output
parent. No corpus, manifest, repaired source artifact, checkpoint, GE1 model,
encoder, or decoder is mounted or loaded.

Disposable probe optimization is permitted. GE1 backward passes, optimizers,
training, checkpoint writes, model changes, protected access, permutations,
Stage 6, C8, and repair remain forbidden. A timeout is infrastructure-only;
partial pipelines receive no scientific interpretation.

## Validation and execution boundary

Synthetic tests prove input validation, 16-pipeline construction,
production-fitter reuse without reduced settings, no permutation or loader
call, flag thresholds, artifact completeness, negative completion, and
non-authorization. The exact-commit runner reruns every repository preflight
before reading a preserved input.

This implementation task authorizes runner and bundle preparation only. It
does not submit the runner or execute the screen locally.
