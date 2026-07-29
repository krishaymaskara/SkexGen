# Flat-Baseline Phase B Validation Evaluation

## 1. Status, scope, and authority

This specification freezes the validation-only evaluation protocol for the
repaired `B0-FLAT-MIXED-VQ` checkpoint before any repaired-checkpoint
reconstruction result is examined.

The protocol is **FROZEN; DETERMINISTIC SMOKE VERIFIED; FULL-VALIDATION
WORKFLOW PREPARED AND AWAITING REVIEW**. The acceptance thresholds and
decision mapping below are predetermined. The implemented full-workflow
changes must be reviewed before the full evaluation is run. Closing a listed
gap without changing the scientific rule does not unfreeze the protocol. Any
threshold, denominator, partition, checkpoint, or decoding change after
repaired results are visible requires a new decision record and must not be
presented as predetermined.

This is a specification, not evidence that an evaluation ran. A completed run
must receive a new immutable record under `docs/experiments/`. The existing
flat-baseline milestone must not be marked complete from this specification.

Repository code is authoritative for current implementation behavior.
Completed-run and scientific claims below come only from verified experiment
records. The documentation authority order is:

1. `docs/status.md`;
2. `docs/experiments/`;
3. package READMEs;
4. specifications;
5. the research plan.

Status labels used in the capability audits are:

| Label | Meaning |
|---|---|
| `IMPLEMENTED_AND_TESTED` | Checked-in behavior exists and focused tests cover it. |
| `IMPLEMENTED_BUT_UNTESTED` | Checked-in behavior exists without focused coverage. |
| `PARTIALLY_IMPLEMENTED` | Some required data or behavior exists, but the frozen contract is not met end to end. |
| `MISSING` | The required behavior or output does not exist. |
| `AMBIGUOUS` | Code or evidence does not determine one stable interpretation. |

## 2. Audit context and conflicts

The repaired training run and the evaluation implementation have different
provenance and both must be recorded:

| Item | Frozen value |
|---|---|
| Repaired training job | `3326757` |
| Training decision | `PASS_FOR_EVALUATION` |
| Training source commit | `d549ebe4478e166fda8d54f2b173572a1ab52e7a` |
| Selected checkpoint | `best.pt`, epoch 44, global step 748 |
| Checkpoint SHA-256 | `282988af00a2dc9a53e14ceb270537d35f85d5339dc989634f88af5f77931267` |
| Corpus | `/scratch/network/km6349/controlled_corpora/b0-pilot-680-seed2026` |
| Partitions | 544 train / 68 validation / 68 test physical families |
| Collapsed comparison job | `3324856` |
| Audit branch | `flat-mixed-baseline` |
| Audit working commit | `1121b02dec0cc214ad269c6fb89f5e4398ff5bb1` |
| Current evaluator implementation commit | `42183a47ec8fa70e6109fbea33ecda9e5e84d071` |

The following provenance distinctions and remaining gaps must remain visible:

- The current reviewed repository commit is `1121b02...`, the Phase B
  evaluator implementation was last changed at `42183a...`, and the repaired
  checkpoint was trained from `d549ebe...`. These are distinct provenance
  facts; the training source is not the current repository HEAD.
- `prototype/flat_baseline/adroit/evaluate_validation_cpu.slurm` remains the
  historical collapsed-checkpoint workflow. The separate repaired smoke
  workflow is `evaluate_repaired_smoke_cpu.slurm`.
- The repaired smoke contract now uses manifest-only partition authority and
  partition-scoped loading. It resolves six validation IDs before inference,
  opens only those payloads, and records zero train/test family payload
  access.
- The repaired smoke contract now compares the predetermined checkpoint
  SHA-256 before deserialization and then validates epoch, global step, model
  configuration, train-k-means provenance, and corpus partitions.
- The evaluator produces symbolic reconstruction targets, not stable-ID
  `CADHistory` objects. The existing OpenCascade runner accepts canonical
  controlled-corpus histories, not Phase B reconstruction records.
- Repaired evaluation schema version 2 includes exact ten-field and exact
  typed-edge-set rates in `metrics.csv`, plus one shared latent-usage row.
  Historical non-repaired evaluation remains schema version 1 with its
  original artifact schemas, corpus-wide loader behavior, and no mandatory
  latent-usage or repaired-checkpoint fields.

No conflict above is silently resolved by this specification.

## 3. Frozen inputs and prohibitions

### 3.1 Required inputs

The evaluation must:

- use only the selected epoch-44, global-step-748 `best.pt`;
- compare its SHA-256 byte for byte with
  `282988af00a2dc9a53e14ceb270537d35f85d5339dc989634f88af5f77931267`
  before deserialization or model construction;
- record training source commit `d549ebe...`, job `3326757`, and the separate
  reviewed evaluation source commit;
- use the authoritative `iid` validation partition and process all 68
  validation physical families exactly once in lexicographic family-ID order;
- record the ordered validation-ID list and its SHA-256;
- use CPU, Python 3.8.x, and PyTorch 1.11.0 with CUDA unavailable;
- set deterministic seeds and single-threaded Torch, OpenMP, and MKL behavior;
- encode each example once, select VQ indices once, and reconstruct latent
  memory once;
- decode `teacher_forced` and `predicted_history` from that same memory;
- supply only target canonical node count as target-length information;
- label results oracle-node-count-conditioned and report the template-leakage
  limitation;
- publish into new, collision-safe, no-replace namespaces; and
- preserve training, pilot, diagnosis, and collapsed-evaluation artifacts
  byte for byte.

Teacher forcing means shifted target-prefix feedback. Predicted-history
decoding means raw argmax categorical feedback plus deterministically derived
geometry-applicability masks. Neither path may receive target operation
count, target operation types, target pointers, target edges, or target
geometry.

### 3.2 Partition prohibition

The evaluator, verifier, and model process must not open or deserialize any
test-family example. Reading corpus-level and split-manifest metadata that
establishes the declared partition sizes is allowed only if it does not load
test-family payloads. Test inference and test target construction are
forbidden. The evaluation record must report `test_family_records_loaded=0`
and `test_partition_evaluated=false`.

### 3.3 Result-blinding prohibition

Before this protocol and its required-before-smoke implementation patch are
reviewed, nobody executing this workflow may:

- run the repaired checkpoint on any family;
- inspect repaired teacher-forced or predicted-history reconstructions;
- execute repaired outputs through OpenCascade; or
- choose thresholds using repaired-checkpoint behavior.

## 4. Existing evaluator capability audit

All reconstruction metrics below are independently aggregated for
`teacher_forced` and `predicted_history`. Latent usage is shared, because both
paths use the same reconstructed latent memory; duplicating it by path would
misrepresent two identical observations as independent measurements.

| Required output | Status | Current behavior and limitation |
|---|---|---|
| Raw completion rate | `IMPLEMENTED_AND_TESTED` | Per-example boolean, counts, and rates are in JSON and headline CSV. |
| Raw integrity validity | `IMPLEMENTED_AND_TESTED` | Prerequisite-gated, with stable failures. |
| Reconstruction-target validity | `IMPLEMENTED_AND_TESTED` | Kept separate from controlled-domain semantics. |
| Controlled-domain validity | `IMPLEMENTED_AND_TESTED` | Restricted grammar, category, geometry, operation, edge, and cross-record checks. |
| Node-type token accuracy | `IMPLEMENTED_AND_TESTED` | Sufficient statistics and aggregate accuracy are stored. |
| Exact node-type sequence accuracy | `IMPLEMENTED_AND_TESTED` | Per-example exact flag and aggregate rate are stored. |
| Exact complete ten-field match | `IMPLEMENTED_AND_TESTED` | Stored in examples, summary, and headline CSV. |
| Geometry MAE | `IMPLEMENTED_AND_TESTED` | Finite applicable physical-channel error; nonfinite and unavailable counts remain visible. |
| Geometry RMSE | `IMPLEMENTED_AND_TESTED` | Uses aggregated squared-error sufficient statistics. |
| Operation-type sequence accuracy | `IMPLEMENTED_AND_TESTED` | Exact target-versus-predicted operation-type sequence rate. |
| Pointer accuracy | `IMPLEMENTED_AND_TESTED` | Overall and conditional forms exist; the frozen headline uses overall accuracy. |
| Edge micro-F1 | `IMPLEMENTED_AND_TESTED` | Micro precision, recall, and F1 use stored TP/FP/FN counts. |
| Exact typed-edge-set accuracy | `IMPLEMENTED_AND_TESTED` | Stored in examples, summary, and headline CSV. |
| Active-code count | `IMPLEMENTED_AND_TESTED` | The post-publication verifier derives it from stored latent indices. |
| Codebook perplexity | `IMPLEMENTED_AND_TESTED` | Computed once from the shared empirical validation assignment distribution. |
| Code utilization | `IMPLEMENTED_AND_TESTED` | The verifier reports observed active count divided by configured codebook size. |
| Per-code assignment counts | `IMPLEMENTED_AND_TESTED` | The core summary and verifier publish all configured entries, including zeros. |
| Failure categories by target template | `IMPLEMENTED_AND_TESTED` | Each template stratum contains primary and any-failure counts. |
| Failure categories by primitive/profile family | `IMPLEMENTED_AND_TESTED` | Circle, rectangle-line, and capsule-line/arc target strata contain failure counts. |
| Per-example failure-analysis records | `IMPLEMENTED_AND_TESTED` | Family metadata, latent indices, both paths, validity, failures, and sufficient statistics are stored. They are not executable CAD histories. |

`examples.jsonl` plus mandatory `raw_predictions.jsonl` and authoritative
validation targets are sufficient to recompute the paired reconstruction
aggregates exactly. The verifier already recomputes raw conversion and metric
records, rebuilds both aggregate trees, reconciles the teacher/predicted
difference, checks CSV rows, and checks failure rows. Exact recomputation is
not guaranteed when optional raw predictions are omitted; therefore raw
prediction publication is mandatory under this protocol.

The current strata describe failures of predictions grouped by authoritative
target template and primitive family. They do not classify a malformed
prediction as having a reliable predicted template or primitive family.

## 5. OpenCascade capability audit

The existing kernel package deserializes canonical controlled
`CADHistory` JSON and supports line, arc, and circle profiles, extrude,
revolve, `NEW_BODY`, `JOIN`, and `CUT`. It checks every feature and Boolean
result. Its fake-adapter tests do not establish real OpenCascade success.

The status below is relative to the required end-to-end input: frozen Phase B
reconstruction records.

| Execution requirement | Status | Current behavior and limitation |
|---|---|---|
| Accept Phase B reconstruction records | `MISSING` | No prediction-to-stable-identifier `CADHistory` adapter exists. |
| Raw record validity | `PARTIALLY_IMPLEMENTED` | Phase B computes and tests it, but the kernel runner cannot ingest or preserve it. |
| Controlled-domain validity | `PARTIALLY_IMPLEMENTED` | Phase B computes and tests it, but no handoff schema joins it to execution. |
| Conversion success | `MISSING` | There is no attempted/succeeded conversion record for Phase B predictions. |
| OpenCascade execution success | `PARTIALLY_IMPLEMENTED` | Canonical histories execute and report success, but Phase B predictions cannot reach the runner. |
| Non-null shape | `PARTIALLY_IMPLEMENTED` | Enforced and represented in shape metrics for canonical inputs, not reported as a separate Phase B rate. |
| Valid shape | `PARTIALLY_IMPLEMENTED` | Enforced for features and Boolean results; no Phase B aggregate. |
| Single-solid result | `PARTIALLY_IMPLEMENTED` | Exactly one solid is enforced; no Phase B aggregate. |
| Positive finite volume | `IMPLEMENTED_AND_TESTED` | Enforced above `1e-9` in the canonical executor, including focused failure tests; Phase B wiring is still absent. |
| Ineffective `JOIN` | `PARTIALLY_IMPLEMENTED` | Tested and reported as `boolean_no_effect`; JOIN and CUT are distinguishable from operation metadata but not separate failure categories. |
| Ineffective `CUT` | `PARTIALLY_IMPLEMENTED` | Same shared failure category and limitation as JOIN. |
| Extrude support | `IMPLEMENTED_AND_TESTED` | Canonical extrude construction and all controlled templates reach the adapter. |
| Revolve support | `IMPLEMENTED_AND_TESTED` | Canonical revolve construction and all controlled templates reach the adapter. |

The follow-up must preserve distinct facts rather than collapse them into one
`kernel_status`: input raw validity, input controlled-domain validity,
conversion attempted/succeeded, kernel call completed, final non-null,
kernel-valid, exactly-one-solid, positive-volume, and semantic Boolean
effectiveness.

## 6. Authoritative collapsed-checkpoint comparison

The verified experiment record for job `3324856` and its locally
checksum-verified frozen archive are authoritative. No retrieval command is
needed.

| Metric | Teacher-forced | Predicted history |
|---|---:|---:|
| Raw completion rate | 1.0000 | 1.0000 |
| Raw integrity validity | 1.0000 | 1.0000 |
| Reconstruction-target validity | 1.0000 | 1.0000 |
| Controlled-domain validity | 0.0000 | 0.0000 |
| Node-type token accuracy | 0.8705 | 0.6093 |
| Exact node-type sequence | 0.3382 | 0.3382 |
| Exact complete ten-field match | 0.0000 | 0.0000 |
| Geometry MAE | 3.3377 | 3.4623 |
| Geometry RMSE | 20.4508 | 23.7956 |
| Exact operation-type sequence | 0.3382 | 0.3382 |
| Pointer accuracy | 1.0000 | 0.3772 |
| Edge micro-F1 | 0.8651 | 0.5616 |
| Exact typed-edge set | 0.0000 | 0.0000 |

All 136 validation latent assignments used code 17: active-code count 1,
perplexity 1.0, and utilization 0.03125. Both paths attempted all 68
validation families. Neither produced a controlled-domain-valid history.
The run did not execute predictions through OpenCascade, so collapsed
execution metrics do not exist.

These values establish a zero baseline for predicted-history
controlled-domain validity. They do not establish an execution threshold or
provide a repaired-checkpoint result.

## 7. Predetermined Week 3 decision rule

All five gates are conjunctive. Latent usage alone can never accept B0.

### 7.1 Gate A: provenance and protocol

Gate A passes only if:

- the checkpoint hash equals the frozen repaired hash before load;
- checkpoint epoch, step, training job, training source commit, model
  configuration, and partition data state reconcile;
- the reviewed evaluation commit is clean and recorded;
- exactly the authoritative 68 validation IDs are processed once;
- no test-family record is loaded and no test model execution occurs;
- CPU, Python 3.8.x, PyTorch 1.11.0, deterministic settings, and shared-memory
  paired decoding are recorded;
- the fixed smoke replay is byte-identical in two new namespaces;
- the full artifact set, no-replace publication, report, hashes, and scheduler
  evidence validate;
- raw predictions are published; and
- every aggregate and failure table agrees exactly with recomputation from
  per-example/raw records and authoritative validation targets.

Any Gate A failure produces `REJECT_PROTOCOL_OR_PROVENANCE`; scientific
metrics from that run are not used for acceptance.

### 7.2 Gate B: latent usage

Over all latent assignments for the 68 validation families:

- active-code count must be at least 2; and
- empirical codebook perplexity must be at least 2.0.

The artifact must also publish the count for every configured code, including
zeros, total assignments, utilization, and dead-code rate. Failure produces
`REJECT_RECONSTRUCTION_QUALITY`.

### 7.3 Gate C: reconstruction improvement and autoregressive survival

Let `V_TF` and `V_PH` be controlled-domain-valid counts out of 68 for
teacher-forced and predicted-history decoding.

Reconstruction quality requires:

- `V_TF >= 17`; and
- `V_PH >= 17`, equivalently predicted-history controlled-domain validity
  at least 0.25 and an absolute improvement of at least 0.25 over job
  `3324856`.

Seventeen families is the smallest predetermined threshold interpreted as a
material validation-set capability rather than isolated successes. It is
strictly stronger than merely beating the collapsed value of zero while
remaining feasible for the small, bounded controlled domain.

Autoregressive survival additionally requires:

```text
V_PH >= ceil(0.5 * V_TF)
```

Thus at least half of teacher-forced controlled-domain-valid
reconstructions must survive predicted-prefix feedback. If teacher-forced
quality passes but either predicted-history condition fails, the decision is
`REJECT_AUTOREGRESSIVE_GENERALIZATION`. If teacher-forced quality itself
fails, the decision is `REJECT_RECONSTRUCTION_QUALITY`.

All other paired metrics remain mandatory diagnostic outputs, but none may
substitute for controlled-domain validity in this gate.

### 7.4 Gate D: representation coverage

Coverage is evaluated over distinct validation physical-family IDs and
against target operation-family membership. A qualifying reconstruction
must:

- be predicted-history controlled-domain valid;
- come from a target whose operation sequence contains the named family; and
- reproduce the target operation-type sequence exactly.

Let `Q_E` be the set of distinct qualifying validation family IDs whose
target operation sequence contains extrude, and let `Q_R` be the equivalent
set for revolve. A family is counted at most once within each category. A
qualifying mixed history may belong once to both `Q_E` and `Q_R`.

The report must publish:

```text
extrude_qualifying_family_count = |Q_E|
revolve_qualifying_family_count = |Q_R|
overlap_qualifying_family_count = |Q_E intersection Q_R|
unique_qualifying_family_count = |Q_E union Q_R|
```

Gate D requires `|Q_E| >= 3` and `|Q_R| >= 3`. Three avoids treating one
accidental valid history as credible coverage. A zero count for either
operation category is unconditionally disqualifying. Failure produces
`REJECT_RECONSTRUCTION_QUALITY`, unless Gate C already identifies
autoregressive failure.

### 7.5 Gate E: executable CAD

OpenCascade execution is required for full Week 3 acceptance. It runs in a
separate immutable job consuming the frozen Phase B artifacts and performing
no model inference.

Define:

- `C` as the number of controlled-domain-valid predicted-history
  reconstructions;
- `X` as the number of those reconstructions successfully converted to
  executable canonical `CADHistory`; and
- `E` as the number of successfully converted histories whose complete
  OpenCascade execution produces a non-null, kernel-valid, exactly-one-solid
  result with positive finite volume and semantically effective JOIN/CUT
  operations.

Every member of `C` must receive exactly one conversion attempt. Therefore
`0 <= E <= X <= C <= 68`.

Report separately:

```text
end_to_end_execution_rate = E / 68
domain_conditional_execution_rate = E / C
conversion_attempt_execution_rate = E / C
conversion_success_rate = X / C
```

Undefined zero-denominator rates remain `null`, never zero. Gate E requires:

- `X == C`;
- `E >= 17`;
- `E >= ceil(0.80 * C)`;
- at least one successful extrude reconstruction; and
- at least one successful revolve reconstruction.

When `C == 17`, all 17 must execute successfully: `E` cannot exceed `C`, and
the independent `E >= 17` floor therefore forces `E == 17`.

The 0.25 end-to-end threshold matches the predetermined material
reconstruction floor. The 0.80 conditional threshold requires the symbolic
validity layer to predict kernel behavior reliably while allowing a bounded
number of genuine kernel/geometry failures. This choice uses the research
plan's executable-history requirement, the deliberately small controlled
domain, the collapsed run's zero valid reconstructions, and the existing
executor's strict intermediate/final checks. It does not use repaired
results. These rules remain independent of all repaired-checkpoint results.

Any Gate E failure after A-D pass produces `REJECT_KERNEL_EXECUTION`.

### 7.6 Final label precedence

Apply the first matching row:

| Condition | Final label |
|---|---|
| Gate A fails | `REJECT_PROTOCOL_OR_PROVENANCE` |
| Teacher-forced Gate C quality fails, Gate B fails, or Gate D coverage fails without an autoregressive-only explanation | `REJECT_RECONSTRUCTION_QUALITY` |
| Teacher-forced quality passes but predicted-history Gate C or predicted-only coverage fails | `REJECT_AUTOREGRESSIVE_GENERALIZATION` |
| Gates A-D pass and Gate E fails | `REJECT_KERNEL_EXECUTION` |
| Gates A-E all pass | `ACCEPT_B0_FOR_WEEK4` |

Acceptance authorizes the repaired checkpoint as the Week 4 B0 reference. It
does not authorize test evaluation or establish any graph-versus-flat claim.

## 8. Deterministic smoke design

The smoke uses the lexicographically first six authoritative validation IDs.
Their explicit ordered list and list SHA-256 must be printed during preflight
and stored in the report. It runs both decoding paths with batch size 3.

Two independent, collision-safe namespaces use identical inputs and settings.
Every canonical artifact, including raw predictions, must be byte-identical
by filename and SHA-256. Both publications must pass the full artifact
verifier, aggregate recomputation, checkpoint literal-hash check, validation
selection check, and no-test-load assertion.

The smoke is an engineering determinism gate, not scientific evidence and not
an opportunity to revise thresholds. It performs no full OpenCascade audit.
After the Phase B-to-`CADHistory` adapter exists, a separate one-or-two-record
conversion/execution fixture smoke may be used, without inspecting repaired
model output, before the follow-up job.

## 9. Full validation-run design

After smoke acceptance, one new Slurm job:

1. verifies branch, clean reviewed evaluation commit, container, CPU-only
   environment, Python/PyTorch versions, corpus identity, validation count,
   literal checkpoint hash, and empty destination namespaces;
2. runs focused and full regression suites;
3. evaluates all 68 validation families in authoritative order with both
   paths and shared reconstructed memory;
4. publishes required artifacts with no replacement;
5. recomputes all metrics from raw and per-example evidence;
6. publishes complete latent usage, all reconstruction strata, and the
   frozen Gate A-D decision inputs;
7. records exit codes, scheduler job/state/node/elapsed data, source
   provenance, and environment; and
8. emits a collision-safe final report containing every artifact SHA-256.

The job must not update any earlier namespace. A completed result receives a
new experiment record even when rejected.

The prepared implementation uses
`prototype/flat_baseline/adroit/evaluate_repaired_validation_cpu.slurm` and
the evaluator/verifier flag `--repaired-full-contract`. This flag is distinct
from the repaired-smoke and historical evaluator modes. It requires all 68
validation families, no family limit, batch size 32, raw predictions, CPU,
and no test authorization. Slurm job `3327735` exercised the workflow but
failed closed during prepublication metadata validation. Job `3329040`
subsequently completed inference and artifact validation but failed during
manifest construction because `find` omitted a symlink-backed evaluation
directory. Neither failed job is accepted full-run evidence. The reviewed
recovery path must independently revalidate job `3329040` and may replace only
its incomplete nine-entry manifest with the exact 15-entry stable-path
manifest.

## 10. OpenCascade follow-up design

Use a separate immutable job. This isolates conversion from inference,
preserves the exact paired model outputs, allows PythonOCC dependencies to
fail independently, and permits replay without another checkpoint load.

The job must:

- consume the verified frozen Phase B publication by path plus artifact
  hashes;
- reject any publication that fails Phase B verification;
- perform no model or checkpoint load;
- attempt conversion for every predicted-history controlled-domain-valid
  record and no invalid record;
- synthesize deterministic stable identifiers without changing predicted
  structure or geometry;
- preserve the original family ID, target metadata used only for strata,
  prediction path, and Phase B artifact identity;
- execute every converted history through the existing strict executor;
- publish per-record conversion and execution states plus all denominators
  in Gate E;
- separate ineffective JOIN and CUT counts despite the executor's shared
  `boolean_no_effect` category;
- report extrude/revolve/template/primitive strata; and
- use a new no-replace namespace, canonical report, hashes, scheduler
  evidence, and deterministic replay or a bounded fixture replay.

## 11. Smallest implementation patch plan

### Implemented before smoke; review still required

- Partition-scoped validation loading in the evaluator and verifier avoids
  train/test payloads and records zero test-family records loaded.
- The repaired contract compares the frozen literal SHA-256 before checkpoint
  deserialization and validates the remaining checkpoint contract afterward.
- A separate repaired-evaluation Slurm workflow targets the job `3326757`
  epoch-44 checkpoint without overwriting the historical workflow or outputs.
- The core summary and verifier publish empirical perplexity, a full
  length-32 assignment-count vector, total assignments, utilization, and
  dead-code metrics.
- The headline CSV includes exact ten-field and exact typed-edge-set rates
  and one shared latent-usage row.
- Focused tests cover wrong/missing checkpoints, scoped payload access,
  zero-count codes, known perplexity, shared memory, CSV schema, aggregate
  recomputation, deterministic replay, and collision-safe publication.

### Implemented for full evaluation; review still required

- The workflow requires a clean reviewed commit equal to the branch and
  remote-tracking HEAD and uses a new commit-and-job-qualified full namespace.
- Raw prediction publication is mandatory under the repaired-full contract.
- The evaluation artifacts and Gate A-D report preserve source, corpus,
  checkpoint, partition-load, environment, scheduler, regression, hash, and
  manifest evidence in the new namespace; Slurm stdout/stderr paths are
  recorded and scheduler-managed.
- The machine-readable report contains the frozen paired metrics, shared
  latent usage, representation-coverage counts, and Gate A-D inputs while
  recording Gate E as not run and final acceptance as undetermined.

### Required before OpenCascade execution

- Implement and test a deterministic Phase B reconstruction-to-`CADHistory`
  adapter with stable IDs and no semantic repair.
- Add a runner that consumes and verifies frozen Phase B artifacts rather
  than a controlled-corpus directory.
- Preserve separate conversion, kernel, shape, solid, volume, and Boolean
  effect fields and the frozen denominators.
- Add extrude and revolve fixtures, failure fixtures for every execution
  state, ineffective JOIN/CUT separation, no-replace publication, and report
  verification.
- Validate the real PythonOCC environment in the immutable follow-up job.

### Optional analysis

- Failure galleries and representative records;
- confidence intervals clearly labeled as descriptive;
- per-template, per-primitive, and history-depth comparisons beyond the
  frozen gates; and
- latent-code semantic analysis after the Week 3 decision.

Optional analysis cannot change the frozen decision.

## 12. Claims this evaluation cannot establish

Even an `ACCEPT_B0_FOR_WEEK4` result cannot establish:

- autonomous node-count or EOS prediction;
- held-out IID test performance;
- systematic-combination, length, or parameter extrapolation;
- broad real-world CAD generality;
- graph superiority or a structure/geometry factorization advantage;
- latent-code semantics, disentanglement, or localized learned editing;
- multi-seed reliability or statistical significance; or
- that revolve support is itself a research contribution.

The evaluation is validation-only, oracle-node-count-conditioned, limited to
one repaired seed and the controlled single-body extrude/revolve domain.
