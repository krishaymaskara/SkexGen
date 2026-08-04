# B0 Phase B Repaired-Checkpoint Full Validation

## Record status

| Item | Value |
|---|---|
| Experiment ID | `b0-phase-b-repaired-full-validation-3329040` |
| Record status | `recovered` and locally verified |
| Evidence state | `verified-local` |
| Scientific role | Frozen 68-family repaired-checkpoint validation |
| Evaluation commit | `ce4abca8f450745a8832c0189aaae4a7d8263ec0` |
| Recovery implementation | `b00ac0d1a645ae9559e269ed93f5ee7940a0512c` |
| Evaluation working tree | Clean |
| Slurm job | `3329040` |
| Original scheduler result | `FAILED`, exit `1:0` |
| Execution date | July 28, 2026 |

The scheduler result is intentionally retained as `FAILED 1:0`. Inference and
the complete artifact validator succeeded. The job failed afterward, during
manifest construction, because shell `find` did not traverse the stable
symlink-backed evaluation publication. Recovery did not rerun inference or
rewrite scheduler history.

## Question and frozen decision rule

The evaluation asked whether the repaired epoch-44 checkpoint met the frozen
Phase B validation requirements on exactly all 68 authoritative validation
families while preserving zero test access. Gates A-D used the thresholds in
the frozen Phase B specification. Gate E required a separate OpenCascade job,
so the frozen report retained formal final acceptance as undetermined when
Gate E was not run. Manifest-only recovery could authenticate an already
validated result but could not change any metric, threshold, or gate outcome.

## Evaluation contract

The run evaluated the selected repaired checkpoint on exactly all 68
authoritative IID validation families. It preserved paired teacher-forced and
predicted-history outputs from shared encoded, VQ-indexed, reconstructed
latent memory. Target canonical node count was the only target-derived input
to predicted-history generation.

| Item | Verified value |
|---|---|
| Checkpoint | Repaired B0 `best.pt`, epoch 44, global step 748 |
| Checkpoint SHA-256 | `282988af00a2dc9a53e14ceb270537d35f85d5339dc989634f88af5f77931267` |
| Checkpoint training source | `d549ebe4478e166fda8d54f2b173572a1ab52e7a` |
| Partition | IID validation |
| Authoritative partition counts | 544 train / 68 validation / 68 test |
| Validation families loaded/evaluated | 68 / 68 |
| Train families loaded/evaluated | 0 / 0 |
| Test families loaded/evaluated | 0 / 0 |
| Test partition evaluated | `false` |
| OpenCascade executed | `false` |

The manifest-only partition preflight recorded zero payloads loaded before
inference. Final evaluation metadata separately recorded exactly 68
validation-family payloads loaded and zero train/test payloads loaded. No
test target was constructed and no test inference occurred.

## Execution and recovery history

Job `3329040` passed repository, environment, checkpoint, partition, and
regression preflights. It then:

1. completed full inference for all 68 validation families;
2. published all six evaluation artifacts through the atomic symlink
   publication backend;
3. ran the complete artifact validator successfully;
4. published the frozen Gate A-D input report;
5. failed during the subsequent manifest stage.

The failed manifest contained exactly nine entries: the eight
workflow-evidence files and `gate-a-to-d-inputs.json`. It omitted all six
evaluation artifacts because `find "$EVALUATION" ... -type f` did not descend
through the command-line `evaluation` symlink.

The reviewed recovery revalidated the complete evaluation, checkpoint,
partition, payload-access metadata, workflow evidence, report, provenance,
and exact artifact sets before mutation. It then atomically replaced only the
incomplete manifest. The recovered manifest contains exactly 15 verified
SHA-256 entries:

- six stable `evaluation/...` artifact paths;
- eight stable `workflow-evidence/...` paths;
- `gate-a-to-d-inputs.json`.

Hidden symlink-backing names, duplicate files, and the manifest itself are
excluded. The recovery log records that inference and artifact validation
succeeded and that the original job remains `FAILED` because its only failure
occurred during post-validation manifest construction.

## Frozen scientific results

| Gate input | Verified result | Frozen requirement | Outcome |
|---|---:|---:|---|
| Active latent codes | 5 | At least 2 | Gate B input passes |
| Codebook perplexity | `3.120410089936484` | At least 2.0 | Gate B input passes |
| Teacher-forced controlled-domain-valid families | 0 / 68 | At least 17 | Gate C does not pass |
| Predicted-history controlled-domain-valid families | 0 / 68 | Gate C survival rule | Gate C does not pass |
| Qualifying predicted-history extrude families | 0 | At least 3 | Gate D does not pass |
| Qualifying predicted-history revolve families | 0 | At least 3 | Gate D does not pass |
| Gate E/OpenCascade | Not run | Separate frozen-artifact job | Unevaluated |

The repaired checkpoint used more than one code and therefore avoided the
original checkpoint's one-code collapse under the frozen Gate B inputs.
However, neither decoding path produced a controlled-domain-valid history in
any validation family. Consequently, the frozen Gate C reconstruction and
survival conditions did not pass. With no qualifying controlled-domain-valid
extrude or revolve family, Gate D also did not pass.

The frozen report deliberately records final acceptance as
`not_determined`, because Gate E is a separate unevaluated job. This record
does not change that report or the frozen acceptance rules. It records both
facts without conflation: formal final acceptance remains undetermined, and
the observed Gate C and Gate D inputs do not meet their frozen thresholds.

## Artifacts and integrity

The original recovered external namespace is:

```text
/scratch/network/km6349/flat_baseline_evaluations/
  phase-b-repaired-validation-ce4abca8f450745a8832c0189aaae4a7d8263ec0-3329040/
```

Its stable contract contains:

```text
evaluation/
  conversion_failures.csv
  examples.jsonl
  metrics.csv
  raw_predictions.jsonl
  run_metadata.json
  summary.json
workflow-evidence/
  checkpoint.sha256
  container.sha256
  corpus-manifests.sha256
  environment.json
  partition.json
  regressions.txt
  repository.txt
  scheduler.txt
gate-a-to-d-inputs.json
sha256-manifest.txt
```

The recovery transcript is retained externally as
`/scratch/network/km6349/phase-b-repaired-validation-3329040-recovery.log`.
The verified 15-entry manifest is authoritative for individual artifact
hashes. The flattened namespace was subsequently downloaded to:

```text
/Users/krishaymaskara/research/audited-runs/phase-b-full-validation/
phase-b-repaired-validation-ce4abca8f450745a8832c0189aaae4a7d8263ec0-3329040
```

Its 15-entry manifest has SHA-256
`a13d5f35b345b6814912145a56e0cce0290ae677b030a1215b61e92558862541`
and was checked locally. The recovery transcript itself remains external;
that does not change the locally verified scientific namespace or result.

## Interpretation and unsupported claims

- The full repaired-checkpoint validation inference and scientific artifact
  validation completed despite the scheduler's post-validation failure.
- The manifest-only recovery is not a replay and does not create a second
  model result.
- The repaired latent distribution is noncollapsed by Gate B's frozen
  thresholds, but reconstruction quality remains inadequate under Gates C
  and D.
- No held-out test family was loaded or evaluated.
- No OpenCascade execution occurred, so Gate E and executable-CAD validity
  remain unevaluated.
- This single repaired seed does not establish systematic generalization,
  multi-seed stability, or any graph-versus-flat conclusion.

## Related records

- Checkpoint source: [Full train-k-means retraining](b0_train_kmeans_full_retrain.md)
- Deterministic prerequisite: [Repaired-checkpoint Phase B smoke](b0_phase_b_repaired_smoke.md)
- Original collapsed-checkpoint comparison: [Original Phase B evaluation](b0_phase_b_validation_evaluation.md)
- Frozen protocol: [Phase B evaluation specification](../specifications/flat_baseline_phase_b_evaluation.md)
