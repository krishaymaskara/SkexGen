# GE1 C6 Measurement Contract

Status: additive frozen implementation contract; C6 code is implemented
locally and awaits the authoritative Python 3.8/PyTorch 1.11 train-only smoke.
Exact-commit jobs `3344337` and `3344363` failed in the procedural focused
suite before corpus access. The first required one autonomous-adapter repair
and one test import; the second passed those corrected paths and exposed only
an invalid generic test comparison on tensor-bearing dataclasses. The pending
test-only tensor-aware comparison and complete rerun do not modify this
contract.

This record resolves measurement details that were intentionally deferred by
the accepted Stage 0 preregistration. It does not change the architecture,
partitions, endpoints, training budget, or decision rules.

## Training and checkpoint selection

Both arms use 50 epochs, physical-family batch size 8, AdamW with learning
rate `1e-3` and weight decay `0`, global-norm clipping at `1.0`, and seeds
2026, 2027, and 2028. A checkpoint opportunity occurs after every completed
epoch. The only selected experimental checkpoint is fixed epoch 50. Epochs
1–49 are recovery, provenance, debugging, or predetermined diagnostic
artifacts; neither development results nor training loss can promote one.

`training_example_presentations` is the cumulative number of actual physical
examples in successfully completed optimizer batches. It includes examples
again in later epochs and excludes evaluation, padded rows, interventions,
and a failed step that did not complete its intended training presentation.
For 407 families this is 20,350 at epoch 50; there are 51 steps per epoch and
2,550 steps per run.

The plateau diagnostic uses epoch-mean, per-example-normalized training loss.
At epoch `e >= 10`, `recent_best` is the minimum over `e-4..e`,
`previous_best` is the minimum over `e-9..e-5`, and relative improvement is
`(previous_best - recent_best) / max(abs(previous_best), 1e-12)`. The first
value strictly below `0.01` is recorded. It cannot stop training, change the
learning rate, modify the budget, trigger evaluation, or select a checkpoint.

## Provenance and recovery

Every checkpoint records the exact commit and branch/detached state, porcelain
status, governed prototype source digest, frozen configuration and partition
digests, Python/PyTorch/device/host/job identity, arm, seed, schema, model and
optimizer state, counters, loss and plateau histories, data-order history, and
Python, NumPy, PyTorch CPU, optional CUDA, and sampler RNG states. The source,
configuration, partition, model, and checkpoint identities are rechecked
immediately before an atomic checkpoint replacement. A mismatch is terminal.
Generated outputs must be external or Git-ignored and are excluded from the
governed source digest.

## Target-free autonomous boundary

Only the encoder-specific C3 input adapter and selected C4 encoder precede the
continuous-memory boundary. One common helper supplies that memory, the
example's target-free input node count, and a fresh autonomous start state to
the C5 decoder. No target tensor, target chronology, target prefix, teacher
forcing, repair, or ground-truth substitution is accepted. Raw, constrained,
and explicit converted outcomes remain separately observable. Targets enter
only after generation inside metric scoring.

## Primary executable-prefix endpoint

The generated typed graph is ordered only by C2 relationship-based
canonicalization. Ambiguous or failed canonicalization scores zero; decoder
row order, target order, and template knowledge are forbidden fallbacks. For
each `k` from zero through target operation count `O`, evaluation retains the
first `k` complete canonical generated operation groups and removes only later
groups and nodes exclusively owned by them. It adds, repairs, or alters
nothing. Each positive prefix must pass strict reconstruction conversion and
controlled analytic validity. The primary score is the largest passing `k`,
capped at `O`, divided by `O`. Every attempted `k`, conversion result,
analytic result, failure code, and failure stage is retained.

## Aggregation and secondary measurements

There is one top-level observation per physical family. If duplicate sample
records exist, arithmetic means are taken within family before equal-weight
macro averaging across families unless a metric explicitly names another
denominator. Exact denominators accompany every aggregate. Undefined
precision, recall, attachment, geometry, intervention, or availability values
are structured nulls with reasons, never favorable zeros.

Secondary fields include complete executable and single-solid validity, exact
canonical graph and node sequence, `depends_on` precision/recall/exactness,
attachment accuracy, strict conversion, first-failure code and stage,
unnormalized prefix, and physical-unit geometry MAE. Geometry families are
reference plane channels 0–8, profile primitives 9–32, axis 33–36, and
operation parameter 37–38, using the frozen channel scales. Geometry error is
diagnostic and has no newly selected success tolerance.

First-failure stages are: encoder/input preparation, node rollout, geometry
prediction/reconstruction, edge prediction, canonicalization, prefix
construction, strict conversion, analytic validity, and no failure. C2's
reachable versus defensive-only codes remain distinct.

## Memory interventions

`P_true` supplies each example's own memory. `P_shuffle` applies one
seed-pinned cyclic derangement to the complete family-sorted scored set, so
every recipient receives another family's memory and no last-batch singleton
can create a fixed point. `P_mean` computes the mean independently at every
latent-token/feature position within each deterministic evaluation batch and
broadcasts it inside that batch. A singleton mean batch is explicitly equal
to true memory by construction. Only memory changes: identities, target-free
bookkeeping, autonomous prefix state, inputs, and targets do not.

All three conditions use the same fresh autonomous decoder and complete metric
path. `R_shuffle` and `R_mean` divide the respective aggregate family-macro
primary mean by `P_true`. When `P_true` is exactly zero, the ratio is null with
reason `zero_true_denominator`; no epsilon is added.

## Architectural and resource records

Actual instantiated counts must reproduce 22,800 flat encoder parameters and
23,468 graph encoder parameters: a difference of 668, or
2.9298245614% relative to flat. Decoder counts are recorded by top-level
component and must match across arms. The receptive-field record states only
that three relational layers cover the verified maximum undirected controlled
template diameter of three; it does not claim that useful information is
learned.

Wall time uses `time.perf_counter` and separates preparation, per-epoch
training, checkpoint/provenance, autonomous evaluation, metric computation,
interventions, and total process time. CPU memory uses
`resource.getrusage(RUSAGE_SELF).ru_maxrss`, converting macOS bytes or Linux
KiB to bytes. It is a non-resettable current-process cumulative high-water
mark, excludes children, and is not labeled as an isolated phase peak.

## C6 completion boundary

The authoritative completion gate is a two-epoch engineering smoke over only
the 407 `operation_template.train` families through the C1 loader, followed by
strict epoch-2 reload and the complete three-condition metrics record.
Development, ER, RR, IID, history-depth, and geometry-extrapolation data remain
closed. The smoke is not a scientific result and cannot tune any frozen
choice. C7 and later work remain unauthorized until this gate is reviewed.
