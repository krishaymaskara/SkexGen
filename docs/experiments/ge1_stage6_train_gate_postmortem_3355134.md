# GE1 Stage 6 Train-Gate Postmortem, Job 3355134

## Audit disposition

Job `3355134` is a valid completed read-only, train-only postmortem. The local
six-file artifact passes its SHA-256 manifest and the committed full artifact
verifier. The scientific result reproduces a failed producer predicate, but
that outcome does not invalidate the complete postmortem artifact.

| Item | Verified value |
|---|---|
| Job | `3355134` |
| Commit | `ddf9617fb124fbc18f38302943f0c982019bfee9` |
| Diagnostic | `GE1-STAGE6-TRAIN-GATE-POSTMORTEM-v1` |
| Artifact | `GE1-STAGE6-TRAIN-GATE-POSTMORTEM-ARTIFACT-v1` |
| Protocol | `GE1-STAGE6-STRUCTURE-ONLY-COMPARISON-v1` |
| Producer target | job `3354961`, commit `5f4542f86756dae44435af27a6e072db6f27a8ef` |
| Slurm state / exit | `COMPLETED` / `0:0` |
| Elapsed / batch MaxRSS | `00:07:21` / `910824K` |
| Runtime | host `adroit-h11n2`; Python `3.8.13`; PyTorch `1.11.0`; CPU only; one thread |
| Focused tests | 11 declared, 11 run, 0 skipped; all passed |
| Terminal telemetry | one completion, artifact-verification pass, and terminal success; no failure or timeout |

Documentation validation covered 102 Markdown files, Python 3.8 AST parsing
covered 97 Python files, and the postmortem structural audit recorded one
diagnostic invocation, one train-loader call, and zero training, backward,
optimizer-step, development-loader, protected-bind, or submission calls.

## Artifact and input authentication

The artifact contains exactly six regular files. `SHA256SUMS` has SHA-256
`927b6407542e8019c05ade07ce5c987462ab730bb7399e394e45c130c61e73f3`.

| File | SHA-256 |
|---|---|
| `artifact_manifest.json` | `1287a0d4397b6322291cece39e3c37449d4449713edc01ea29d3752f7e6a8a04` |
| `resolved_config.json` | `24d0bfbf5ab2fe11283d29434e1d4713dd26fd83e4dc9669d3f75a5c34e0c669` |
| `summary.json` | `0719022fc691d6c2e31ac2373735957246d298ba54bd269a023367416963f7cd` |
| `train_family_records.jsonl` | `9ccbf1b11d320b2f245669be44be284c0e35d73806ce9c6f4dac22bc866bb33e` |
| `training_histories.jsonl` | `96a6d63f61f9684bc0542c5b2913ba5ab353b0608f727a57f4a406f8f724fb89` |

The verifier confirmed six training histories, 7,326 train-family records,
six strict checkpoint recoveries, the exact arm/seed/condition/family matrix,
producer source commit and digest, narrow train identity, and
`verification_status="pass"`. Every wrapper was read-only and remained
unmodified. The authenticated wrapper SHA-256 values were:

| Arm | Seed | Wrapper SHA-256 |
|---|---:|---|
| flat | 2026 | `21c1573602704a86aa7d981ce71be0d1d8697c5ca96545e16268bfcf0ba30b2e` |
| typed graph | 2026 | `b1f24705ab71b319e5f4963ebd8d9cd84a10b5c104d0f80a1df03c5fca379bcc` |
| flat | 2027 | `6e0c6c499baa273caabad895768841099c00d46ac3abc2e9e8ff88cdba970a01` |
| typed graph | 2027 | `0208e4f0064b255396e03f5f348134fb4d0819fecc66b225b88898de36978425` |
| flat | 2028 | `e06ef970718933e2a3b56d054247bb8d5aa940cc6680c4bc1da81ab00cfeaad4` |
| typed graph | 2028 | `44b3eebd60a2f03545ba6c44998bb5bb4e05b5d0613a4da1fad2e6bb6626f327` |

## Recomputed train results

Scores were recomputed from all 7,326 supplied family records with the
unchanged Stage 6 scorer. Each condition has 407 families per arm and seed.
Because `P_true=1.0` throughout, each ratio equals its corresponding observed
score. The gate is inclusive: a ratio passes when it is at most `0.80`.

| Arm | Seed | Support per condition | P_true | P_shuffle | P_mean | Shuffle ratio / gate | Mean ratio / gate | Failed per-seed memory predicates |
|---|---:|---:|---:|---:|---:|---|---|---|
| flat | 2026 | 407 | 1.000000 | 0.683047 | 0.705160 | 0.683047 / pass | 0.705160 / pass | none |
| typed graph | 2026 | 407 | 1.000000 | 0.847666 | 0.808354 | 0.847666 / fail | 0.808354 / fail | `shuffle_over_true`, `mean_over_true` |
| flat | 2027 | 407 | 1.000000 | 0.776413 | 0.855037 | 0.776413 / pass | 0.855037 / fail | `mean_over_true` |
| typed graph | 2027 | 407 | 1.000000 | 0.658477 | 0.847666 | 0.658477 / pass | 0.847666 / fail | `mean_over_true` |
| flat | 2028 | 407 | 1.000000 | 0.606880 | 0.810811 | 0.606880 / pass | 0.810811 / fail | `mean_over_true` |
| typed graph | 2028 | 407 | 1.000000 | 0.710074 | 0.852580 | 0.710074 / pass | 0.852580 / fail | `mean_over_true` |

Pooled arm values use 1,221 records per condition:

| Arm | Support per condition | P_true | P_shuffle | P_mean | Shuffle ratio / gate | Mean ratio / gate | Arm gate |
|---|---:|---:|---:|---:|---|---|---|
| flat | 1,221 | 1.000000 | 0.688780 | 0.790336 | 0.688780 / pass | 0.790336 / pass | pass |
| typed graph | 1,221 | 1.000000 | 0.738739 | 0.836200 | 0.738739 / pass | 0.836200 / fail | fail |

All six optimization runs had finite losses and gradients, plateaued by epoch
200, and supplied the fixed epoch-200 checkpoint. First plateau epochs were
140/90 for flat/typed graph seed 2026, 112/106 for seed 2027, and 111/121 for
seed 2028. Every `P_true` score was 1.0, every train-ceiling shortfall was
zero, and every paired flat-versus-graph absolute shortfall difference was
zero, passing the inclusive `0.05` gate.

The exact producer-level failed predicate was only
`train_memory.typed_graph.mean_over_true`. Thus the minimal combined reason
job `3354961` stopped was `train_memory_use_gate`; optimization reliability
and train-ceiling comparison both passed. The per-seed failures above are
diagnostic decompositions and do not replace the pooled producer predicate.

## Safety and interpretation boundary

The postmortem performed train-only autonomous inference using the authorized
narrow train package and six retained wrappers. It performed no training,
backward pass, optimizer step, checkpoint mutation, repair, or development
access. It opened no RR, ER/test, IID, history-depth, geometry-extrapolation,
corpus, broad manifest, unrelated checkpoint, model artifact, repaired
artifact, or protected partition.

These train-only values diagnose why the producer stopped. They do not prove
encoder superiority, identify an exact defect, or authorize development
access, repair, another producer, finalization, Stage 7, C8, RR, ER/test, or
protected access. A failed scientific predicate is compatible with—and here
is contained in—a complete valid diagnostic artifact.
