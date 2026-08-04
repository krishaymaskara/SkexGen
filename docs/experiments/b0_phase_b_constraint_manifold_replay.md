# B0 Phase B Constraint-Manifold Replay

## Record status

| Item | Value |
|---|---|
| Experiment ID | `b0-phase-b-constraint-manifold-3329040-v1` |
| Record status | `verified` |
| Evidence state | `verified-local` |
| Scientific role | Preregistered validation-only failure diagnosis |
| Branch | `flat-mixed-baseline` |
| Source commit | `e0829773589a07ab471bc3932bf91578107b5467` |
| Working tree | `clean` |
| Slurm/local run ID | Local replay over job `3329040` |
| Execution date | July 29, 2026 |

This immutable diagnostic does not amend the frozen Gate C and Gate D
failures, the unevaluated Gate E state, or formal acceptance recorded by the
repaired-checkpoint evaluation.

## Question and predetermined decision rule

The replay asked whether the repaired checkpoint's zero controlled validity
was sensitive to semantic constants, constrained continuous
parameterization, categorical oracle information, or autoregressive history.
The four preregistered arms were:

1. unmodified baseline;
2. semantic constants;
3. model categories plus constrained geometry;
4. authoritative categories plus constrained geometry.

The protocol fixed the arm order, interpretation thresholds, arithmetic,
immutable source, and output namespace before execution. It could diagnose
published predictions but could not change an evaluation gate.

## Inputs and partition authority

| Item | Value |
|---|---|
| Source evaluation | Repaired Phase B full validation from job `3329040` |
| Source manifest SHA-256 | `a13d5f35b345b6814912145a56e0cce0290ae677b030a1215b61e92558862541` |
| Checkpoint | Epoch 44, global step 748 |
| Checkpoint SHA-256 | `282988af00a2dc9a53e14ceb270537d35f85d5339dc989634f88af5f77931267` |
| Validation families used | 68 |
| Train families used | 0 |
| Test families used | 0 |
| Test partition evaluated | `false` |

The replay consumed only already published validation predictions. It did not
load the model or checkpoint, access corpus payloads or target continuous
geometry, run inference or training, or execute OpenCascade.

## Configuration

The replay created one record for every family, decoding path, and arm:

```text
68 families x 2 paths x 4 arms = 544 records
```

The validator was the frozen Phase B controlled-domain conversion path. The
authoritative-category arm was explicitly an oracle diagnostic, not an
autonomous model condition.

## Environment

The replay ran locally as a read-only artifact transformation. Run metadata
records no Torch runtime, training, Slurm execution, OpenCascade execution,
checkpoint loading, corpus access, or test-payload access.

## Execution

The committed runner is:

```text
prototype/flat_baseline/constraint_manifold_replay.py
```

It validated exact source provenance and published collision-safely to:

```text
/Users/krishaymaskara/research/audited-runs/phase-b-full-validation/
constraint-manifold-replay-3329040-preregistered-v1
```

## Verified results

| Arm | Teacher-forced valid | Predicted-history valid |
|---|---:|---:|
| Baseline | 0 / 68 | 0 / 68 |
| Semantic constants | 0 / 68 | 0 / 68 |
| Model categories plus constrained geometry | 36 / 68 | 29 / 68 |
| Authoritative categories plus constrained geometry | 68 / 68 | 63 / 68 |

The preregistered classifications were:

| Diagnostic | Result |
|---|---|
| Constraint-parameterization sensitivity | `strong` |
| Categorical-oracle sensitivity | `strong` |
| Semantic-constant sensitivity | `none` |
| Autoregressive sensitivity | `not_material` |

The predicted-history survival ratios were `29/36 = 80.56%` for model
categories and `63/68 = 92.65%` for authoritative categories.

The model-category predicted-history arm was valid for 29 of 32
extrude-first families and zero of 36 revolve-first families. The
authoritative-category arm recovered all 36 revolve-first families.

Five authoritative-category predicted-history failures remained. All were
second-sketch `EE` capsules with `nonpositive_fitted_extent`, showing that a
correct profile category can remain incompatible with independently
predicted raw continuous geometry.

## Decision

The replay passed its diagnostic execution and integrity requirements. It
supported a narrower categorical-isolation replay to separate reference-plane
and profile-family effects. Frozen Gates C and D remained failed, and Gate E
remained unevaluated.

## Interpretation

The result supports category-conditioned constrained output construction as
the next bounded intervention. It also shows that the complete
revolve-first failure was categorical rather than a general inability to
construct revolve geometry.

The oracle arm is not a checkpoint score. It demonstrates sensitivity to
information and constraints unavailable to the autonomous model.

## Limitations and claims not supported

- This is one checkpoint, one 68-family IID validation split, and one seed.
- The replay does not establish test or systematic generalization.
- It cannot separate every categorical field because the oracle arm changes
  multiple categories together.
- It does not establish OpenCascade executability.
- Diagnostic validity does not repair the original autonomous predictions.

## Artifacts and integrity

| Artifact | Location | SHA-256 | Availability |
|---|---|---|---|
| Replay manifest | Local immutable namespace | `6ff46389fb3947b3448e5e52860833d0592e83f70849536ddfd4aeb32f73b4b2` | Present and verified |
| Aggregate summary | Manifest-listed artifact | See manifest | Present and verified |
| Replay records | `replay_records.jsonl`, 544 rows | See manifest | Present and verified |
| Transition and failure tables | Manifest-listed CSV files | See manifest | Present and verified |
| Run metadata | `run_metadata.json` | See manifest | Present and verified |

Every manifest-listed artifact passed SHA-256 verification during the local
documentation audit.

## Reproduction and validation

The namespace can be revalidated with:

```bash
cd /Users/krishaymaskara/research/audited-runs/phase-b-full-validation/constraint-manifold-replay-3329040-preregistered-v1
shasum -a 256 -c sha256-manifest.txt
```

This validation was performed successfully. The scientific replay itself was
not rerun.

## Related records

- Source result: [Repaired-checkpoint full validation](b0_phase_b_repaired_full_validation.md)
- Protocol: [Constraint-manifold replay specification](../specifications/flat_baseline_phase_b_constraint_manifold_replay.md)
- Follow-up: [Categorical-isolation replay](b0_phase_b_categorical_isolation_replay.md)
- Checkpoint source: [Full train-k-means retraining](b0_train_kmeans_full_retrain.md)
