# GE1 Stage 6 structure-only comparison contract

| Item | Frozen value |
|---|---|
| ADR | [ADR-0014](../decisions/ADR-0014-ge1-stage6-structure-only-comparison.md) |
| Protocol | `GE1-STAGE6-STRUCTURE-ONLY-COMPARISON-v1` |
| Score | `GE1-STAGE6-STRUCTURAL-PREFIX-v1` |
| Aggregation | `GE1-STAGE6-PAIRED-SEED-AGGREGATION-v1` |
| Memory gate | `GE1-STAGE6-STRUCTURAL-MEMORY-v1` |
| Artifact | `GE1-STAGE6-STRUCTURE-ONLY-ARTIFACT-v1` |
| Producer | `GE1-STAGE6-STRUCTURE-ONLY-PRODUCER-v1` |
| Execution record | `GE1-STAGE6-STRUCTURE-ONLY-EXECUTION-RECORD-v1` |
| Producer artifact | `GE1-STAGE6-STRUCTURE-ONLY-PRODUCER-ARTIFACT-v1` |
| Checkpoint | `GE1-STAGE6-STRUCTURE-ONLY-CHECKPOINT-v1` |
| Checkpoint bundle | `GE1-STAGE6-STRUCTURE-ONLY-CHECKPOINT-BUNDLE-v1` |
| Narrow index | `GE1-STAGE6-NARROW-INDEX-v1` |
| Current authority | implementation/preparation only |

## Question and isolation

This protocol asks only whether the accepted position-free typed-graph encoder
improves autonomous structural reconstruction and E/R/EE/RE compositional
generalization relative to the matched flat chronological encoder. It does not
replace or reinterpret earlier geometry, conversion, sufficiency, or
representation results.

The implementation module is pure scoring/finalization code. It imports no
partition, corpus, training, model, autonomous-generation, checkpoint, or
conversion loader. Synthetic fixtures are the only implementation inputs.

## Frozen execution record

A future derived record must identify all flat/typed-graph runs for seeds
2026/2027/2028, or the prospectively justified 2026/2027 fallback. Every run
must record fresh initialization, 407 train families, 200 epochs, batch 8,
AdamW `0.001/0.0`, clipping `1.0`, fixed epoch 200, parameter counts, loss and
plateau diagnostics, and absence of early stopping, warm start, or development
selection.

Timing evidence independently records raw and 20%-contingency projections for
both three and two seeds plus available wall time. The full set has priority;
fallback is permitted only when full does not fit and two does. If neither
fits, execution is resource-infeasible before scientific input access.

For every retained seed, arm, memory condition, cohort, and physical family,
the record contains preserved autonomous structural evidence. Counts are 407
families for train and 45 for development. Targets join only after generation.

## Structural-prefix computation

For each `k=1..O`, a prefix row contains only Boolean evidence for grammar
completion, exact operation types, exact chronological order, canonical graph,
applicable `depends_on`, and local ownership/attachments. The selected `k` is
the longest uninterrupted leading prefix for which every field passes. The
score is `k/O`.

Grammar is computed with `V5_NODE_GRAMMAR` from actual node IDs and requested
length. Full credit additionally requires equal predicted and target operation
counts; under-, exact-, and over-generation are explicit evidence. Diagnostic
failure order is grammar, type, chronology, `depends_on`,
ownership/attachments, then remaining canonical graph. Relation edges remain
subsets of the complete graph.

The scorer contains no conversion, analytic-validity, geometry, magnitude, or
CAD-kernel call. Those outcomes remain secondary fields and cannot change the
primary score.

## Aggregation and gates

The primary development effect pairs graph-minus-flat within family and seed,
means families within seed, then means retained seeds. The inclusive threshold
is `0.10`. The two-seed fallback additionally requires both seed effects to be
nonnegative.

The structural memory gate applies independently to train/development and
flat/typed-graph. It requires positive true-memory performance and inclusive
shuffled/true and mean/true ratios no greater than `0.80`.

Shuffle is a deterministic seed-and-batch-identity derangement within each
batch. Donor and recipient differ, donors cannot cross batches, singleton
batches fail closed, and mean memory remains batch-local.

Capacity parity, reliable optimization, provenance, input integrity, artifact
integrity, and memory use are validity gates. Outcome never controls whether a
complete artifact is valid.

Interpretation is exactly: `graph_supported` for an overall effect at least
`0.10` with validity and fallback conditions; `flat_supported` for an overall
effect at most `-0.10` with validity; `mixed` only when the overall effect lies
strictly between those bounds and a seed/template effect reaches absolute
`0.10`; `null` when validity passes and none reaches a threshold; and
`inconclusive` whenever a validity gate fails.

## Governed producer and strict recovery

The additive producer validates prospective timing and exact narrow input
allowlists before its only two scientific loaders. It trains independently
initialized matched arms for every retained seed using the frozen arithmetic,
writes only epoch 200, and recovers each checkpoint into a fresh model through
the established strict C6 loader. The Stage 6 wrapper binds protocol, commit,
source digest, arm, seed, epoch, model/grid configuration, partition identity
and hashes, arithmetic, parameter count, checkpoint schema, and hashes.

Every run must have 200 finite loss and gradient entries, reach the established
five-epoch moving-best 1% plateau after epoch 10 and by epoch 200, and use only
the fixed epoch-200 checkpoint. Within seed, graph/flat structural train-ceiling
shortfalls may differ by at most `0.05`. Geometry and complete geometric
validity cannot establish optimization reliability. Trainable parameter-count
difference remains inclusively bounded by `5%`.

Only the separately audited Stage 6 narrow loader is permitted. Train and
development each require an independently produced and reviewed narrow index
binding the authoritative manifest hash, assignment hash, exact sorted family
IDs, two variants per family, metadata, payload-only allowlist, and every
payload hash. Absolute/parent paths, symlinks, broad roots, unexpected files,
incomplete variants, protected partitions, and artifacts are rejected. No
real index or hash is established in implementation-only work.

Lifecycle order is timing, source, train index/load, matched construction, all
training, all strict recovery, train interventions and reliability, then and
only then development index/load and interventions. Development cannot affect
timing, seeds, training, checkpoint selection, or recovery.

## Artifacts

The producer first atomically creates six files: `resolved_config.json`,
`training_histories.jsonl`, `checkpoint_identities.jsonl`,
`family_records.jsonl`, `artifact_manifest.json`, and `SHA256SUMS`. Its verifier
requires exact canonical contents, complete arm/seed/condition/cohort/family
matrices, strict checkpoint provenance, access declarations, and finalizer
compatibility. Outcome is never an artifact-validity gate.

The separately atomic checkpoint bundle retains each exact wrapper, resolved
source/input/protocol identity, manifest, and checksums. Producer records bind
its digest and identities; finalization consumes only the producer artifact.
Source, input, producer, and checkpoint evidence reaches the final artifact.
Job-specific `.incomplete-<job_id>` staging is retained on failure and cannot
be consumed as final evidence or block a different job. Failure or timeout
does not authorize retry.

Atomic finalization creates:

1. `resolved_config.json`;
2. `family_metrics.jsonl`;
3. `summary.json`;
4. `artifact_manifest.json`;
5. `SHA256SUMS`.

The verifier requires the exact file set and identities, canonical JSON,
complete checksum coverage, internal score recomputation, paired aggregation,
memory ratios, validity, and interpretation.

## Prepared runners

`prototype/graph_encoder/adroit/ge1_stage6_structure_only_producer.slurm` has
no guessed allocation directives. A future authorized `sbatch` must supply the
reviewed allocation derived from prospective evidence. It verifies timing and
input declarations before accessing narrow roots, has exactly four binds, and
invokes the producer once. `ge1_stage6_structure_only_cpu.slurm` is the
separate finalizer and consumes the producer artifact read-only.

The runner exposes no corpus, manifest, checkpoint, model artifact, RR, ER,
IID, history-depth, or geometry-extrapolation path. It cannot generate the
derived scientific record and is not authorized for transfer or submission.

## Remaining prerequisites

Before execution, the reviewer must separately approve:

1. exact train and development allowlists/hashes and safe read-only binds;
2. prospective timing and full-three-seed versus fallback decision;
3. reviewed final CPU/GPU, memory, and wall-time allocation;
4. exact future commit and runner hashes;
5. scientific training/inference, development access, transfer, and one Slurm
   submission.

RR, Stage 7, ER, and all other protected access require later authorization.
