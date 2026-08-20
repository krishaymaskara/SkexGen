# GE1 Exploratory Stage 6 Authoritative CPU Validation, Job 3355816

## Disposition

The corpus-free implementation validation **passed** at exact commit
`42d5c489c7236031ff18dfd13ca88af428ca7510` under Python 3.8.13 and
PyTorch 1.11.0 on one Adroit CPU thread.

| Item | Verified value |
|---|---|
| Governing decision | [ADR-0017](../decisions/ADR-0017-ge1-exploratory-stage6-autonomous-stop-execution.md) |
| Job / state / exit | `3355816` / `COMPLETED` / `0:0` |
| Elapsed / batch MaxRSS | `00:10:16` / `762976K` |
| Host | `adroit-h11n3` |
| Runner | `prototype/graph_encoder/adroit/ge1_exploratory_producer_validation_cpu.slurm` |
| Runner SHA-256 | `bf2a9962693aa2011f70fcdb10dd162c49a6f1e64a2d64db71e0e6a2642e3276` |
| Source | clean standalone detached checkout at `42d5c489c7236031ff18dfd13ca88af428ca7510` |
| Magnitude / node identity | `GE1-OPERATION-MAGNITUDE-GRID-ORDINAL-v1` / `GE1-PAD-TERMINATED-UNCONSTRAINED-NODES-v1` |
| Checkpoint identity | `GE1-CHECKPOINT-v4` |

## Validation results

| Suite | Declared | Run | Failures | Errors | Skips |
|---|---:|---:|---:|---:|---:|
| Six focused exploratory/autonomous-stop/producer/timing modules | 51 | 51 | 0 | 0 | 0 |
| Complete graph-encoder discovery | 743 | 743 | 0 | 0 | 6 |
| `model_data` | 86 | 86 | 0 | 0 | 0 |
| `flat_baseline` | 549 | 549 | 0 | 0 | 4 |
| `graph_baseline` | 17 | 17 | 0 | 0 | 0 |
| `representation` | 40 | 40 | 0 | 0 | 0 |
| `controlled_data` | 51 | 51 | 0 | 0 | 0 |

The six complete-discovery skips were exactly the allowlisted CUDA-only
`Stage6CudaRuntimeTests`; no other skip was accepted. The four flat-baseline
skips were the established external-bundle skips. Documentation validation
passed for 114 Markdown files, compileall passed, and Python 3.8 AST parsing
covered 111 graph-encoder modules/tests. Both producer runners, timing,
finalization, checkpoint-bundle, and narrow-loader source audits passed. The
tree was clean and detached at both ends, and the frozen flat/ADR-0013 paths
were unchanged from the implementation parent.

The identity audit independently confirmed that the ordinary producer default
remains legacy, blocking, and result-eligible, while the explicit exploratory
smoke is epoch 2, autonomous-stop, checkpoint-v4, and result-ineligible.

## Evidence

| Artifact | SHA-256 |
|---|---|
| `/scratch/network/km6349/ge1_exploratory_validation_runs/exploratory-cpu-3355816.out` | `f81a7393768ab5b7ac64f384b6028416f44730fbb21c160e8b6cab35c6755e38` |
| `/scratch/network/km6349/ge1_exploratory_validation_runs/exploratory-cpu-3355816.err` | `b7072fdbff72ed5700caa50cc83af6ad2328dbe90f69f042c27860acf5540741` |

Terminal telemetry records no manifest, corpus, training, checkpoint write,
development, protected partition, CAD kernel, Stage 6, or C8 activity.

## Superseded attempt

Job `3355814` at commit `488a89669fe7d45c43fef7e86e900db91fd71e5b`
passed all 51 focused tests, all 743 complete tests, and every sibling suite,
then failed in the runner-only identity audit because it imported constants
from a nonexistent module. It opened no corpus and performed no training or
checkpoint work. Commit `42d5c48…` corrected only that audit import and
telemetry; job `3355816` is the authoritative result.

## Decision boundary

This accepts the implementation for the separately authorized timing and
two-epoch smoke. It does not authorize an epoch-200 producer; that submission
remains explicitly held for a separate designated-reviewer decision.
