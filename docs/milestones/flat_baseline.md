# Milestone: Flat Mixed/VQ Baseline

## Status and scope

Accepted as a training-and-evaluation-readiness milestone on July 28, 2026.

This stage delivered the `B0-FLAT-MIXED-VQ` implementation, reproducible
training infrastructure, length-conditioned decoding, paired validation
evaluation, VQ-collapse diagnosis, deterministic train-only k-means
initialization, and a full 50-epoch continuation.

Acceptance is deliberately narrow: the implementation is operational and the
selected epoch-44 checkpoint passed the predetermined minimum code-usage gate.
The checkpoint has not yet passed final reconstruction, executability,
semantic, or localized-edit evaluation.

## Implementation landmarks

| Capability | Commit |
|---|---|
| Flat mixed/VQ core | `73349b4` |
| Training pipeline and 680-family corpus runner | `19a1859` |
| Post-run analysis | `ba611fb` |
| Length-conditioned autoregressive decoding | `aa19f05` |
| Paired Phase B evaluator series | `ea16b53` through `42183a4` |
| VQ-collapse diagnosis series | `680ae18` through `55b2841` |
| Train-k-means pilot | `68c78f3`, finalized at `856d6a6` |
| Full epoch-boundary continuation | `d549ebe` |

The current package contract is maintained in the
[flat-baseline README](../../prototype/flat_baseline/README.md).

## Evidence chain

| Stage | Evidence and decision |
|---|---|
| Infrastructure | [CPU/CUDA validation](../experiments/b0_training_infrastructure_validation.md) established bounded runner behavior. |
| Corpus | [680-family pilot corpus](../experiments/b0_pilot_corpus_680.md) established 544 train, 68 validation, and 68 untouched test families. |
| Original training | [Original B0 run](../experiments/b0_original_training_collapse.md) completed but collapsed to one active VQ code. |
| Compatibility | [Phase A](../experiments/b0_phase_a_contract_validation.md) preserved the state and checkpoint contract. |
| Original-checkpoint evaluation | [Phase B](../experiments/b0_phase_b_validation_evaluation.md) validated the evaluator and quantified the collapsed checkpoint's weak predicted-history behavior. |
| Diagnosis | [VQ-collapse diagnosis](../experiments/b0_vq_collapse_diagnosis.md) identified nearest-code assignment collapse as the proximate failure. |
| Bounded intervention | [Train-k-means pilot](../experiments/b0_train_kmeans_pilot.md) passed its fixed five-epoch continuation gate. |
| Full continuation | [Full retraining](../experiments/b0_train_kmeans_full_retrain.md) completed 50 epochs and selected epoch 44 with five validation-active codes and perplexity 3.1204. |

The six run bundles from original training through full retraining are frozen
in a single locally verified archive whose checksum is recorded in the
[evidence inventory](../experiments/README.md).

## Supported conclusions

- The original training configuration suffered complete assignment collapse;
  low loss alone was not evidence of a useful discrete representation.
- The diagnosis isolated nearest-code assignment collapse while showing that
  encoder outputs, codebook vectors, gradient flow, and decoder latent
  sensitivity were not themselves uniformly degenerate.
- Train-only k-means initialization prevented recurrence of total assignment
  collapse for this corpus and seed through the 50-epoch budget.
- The selected epoch-44 checkpoint passed the predetermined
  `PASS_FOR_EVALUATION` gate without using the test partition.

## Limitations and claims not yet supported

- Five active validation codes out of 32 is continued under-utilization, not
  evidence of a semantically rich codebook.
- The repaired checkpoint has not yet been run through the frozen Phase B
  validation-only reconstruction protocol.
- OpenCascade executability, code semantics, and localized learned edits have
  not been established.
- There is no multi-seed result and no graph-model comparison.
- The held-out IID test partition remains untouched.
- `PASS_FOR_EVALUATION` is not scientific acceptance.

## Next gate

Run a new immutable validation-only Phase B evaluation of the selected
epoch-44 checkpoint, preserve its scheduler and artifact evidence, and create
a separate experiment record. Only after accepting that validation protocol
should the project add OpenCascade execution and semantics/edit analyses or
begin capacity-matched graph comparisons.

