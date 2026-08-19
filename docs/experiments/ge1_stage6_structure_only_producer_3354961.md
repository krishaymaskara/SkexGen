# GE1 Stage 6 Structure-Only Producer, Job 3354961

## Disposition

Job `3354961` completed all authorized train-side computation but stopped at
the prospective train-reliability gate. It is a valid record of a scientific
gate failure, not a completed producer artifact and not an infrastructure
success that authorizes finalization. Development remained closed.

| Item | Verified value |
|---|---|
| Job | `3354961` |
| Commit | `5f4542f86756dae44435af27a6e072db6f27a8ef` |
| Producer / protocol | `GE1-STAGE6-STRUCTURE-ONLY-PRODUCER-v1` / `GE1-STAGE6-STRUCTURE-ONLY-COMPARISON-v1` |
| Runner | `prototype/graph_encoder/adroit/ge1_stage6_structure_only_producer.slurm` |
| Committed runner SHA-256 | `ff7f67f13b46fd5dc3af8c23fdb558a60d1849b613ecd7dec77015f68a9743a6` |
| Slurm state / exit | `FAILED` / `42:0` |
| Elapsed / batch MaxRSS | `02:07:20` / `855460K` |
| Allocation | one CPU; CPU selected by frozen timing-v2 evidence |
| Runtime | Python `3.8.13`, PyTorch `1.11.0`, one thread |
| Focused tests | 49 declared, 49 run, 0 skipped; all passed |
| Complete discovery | 620 declared and run; 0 failures, 0 errors, exactly 6 CUDA-only skips |
| Terminal scientific event | `stage6_train_reliability_failure` |

The generic runner error trap also emitted
`stage6_structure_only_producer_infrastructure_failure` after the producer
returned status 42. That event describes Slurm/nonzero-exit handling, not the
root cause: the producer's preceding structured event and the authenticated
postmortem identify the prospective scientific train gate as the cause. There
was no timeout, allocation loss, test failure, or dependency failure.

## Completed train-side work and stop point

All six independently seeded arm conditions reached epoch 200:

| Arm | Seeds | Epochs per run | Optimizer steps per run | Train families |
|---|---|---:|---:|---:|
| flat | 2026, 2027, 2028 | 200 | 10,200 | 407 |
| typed graph | 2026, 2027, 2028 | 200 | 10,200 | 407 |

The authenticated job-3355134 histories contain 200 finite loss and gradient
records for every run, and the six retained job-scoped wrappers all identify
epoch 200. The producer then performed train-only autonomous evaluation,
computed optimization reliability, paired train-ceiling checks, and the
train memory-use gate. It raised `stage6_train_reliability_failure` because a
train-side predicate failed.

The exception occurs before `load_stage6_development`, checkpoint-bundle
creation, producer-artifact creation, and work-root cleanup in the committed
source. Consequently:

- development was not loaded or evaluated;
- no producer artifact or finalized Stage 6 comparison was produced;
- the six `stage6-{arm}-seed{seed}.pt` wrappers remained in the job-scoped
  incomplete work root for separately authorized read-only diagnosis;
- RR, ER/test, IID, history-depth, geometry-extrapolation, Stage 7, C8, and
  every protected partition remained closed.

The wrapper identities and hashes were later strictly authenticated by the
[job-3355134 postmortem](ge1_stage6_train_gate_postmortem_3355134.md). The
postmortem established that optimization reliability and every paired
train-ceiling comparison passed; the minimal failed component was the
aggregate typed-graph mean-memory predicate.

## Interpretation and authority

Training completion, scientific gate failure, and scheduler status are three
different facts. Training completed for all six conditions. The prospective
scientific gate failed before development. Slurm therefore recorded the
intentional nonzero producer exit as `FAILED 42:0`, despite no infrastructure
defect being identified.

This job supplies no development comparison, encoder-superiority conclusion,
final artifact, repair authority, retry authority, or permission to run the
CPU finalizer. Another producer, development access, repair, finalization,
RR, Stage 7, C8, ER/test, or protected access requires a new explicit reviewer
decision.
