# ADR-0014: GE1 Stage 6 structure-only graph-versus-flat comparison

- Status: Accepted prospectively for implementation and preparation
- Date: August 18, 2026
- Decision maker: Krishay Maskara, designated GE1 reviewer
- Protocol: `GE1-STAGE6-STRUCTURE-ONLY-COMPARISON-v1`

## Context

Earlier GE1 sufficiency and operation-magnitude work coupled structural
reconstruction to geometry-dependent conversion and validity gates. Those
results remain immutable. They do not answer the narrower question of whether
the position-free typed-graph encoder improves autonomous structural
reconstruction and compositional generalization relative to the matched flat
chronological encoder.

This ADR creates a new prospective endpoint. It does not reinterpret C7,
repaired sufficiency, representation, grid-magnitude, conversion, analytic
validity, or CAD-kernel evidence.

## Decision

Adopt `GE1-STAGE6-STRUCTURE-ONLY-COMPARISON-v1` as a separately versioned,
prospective Stage 6 comparison. Implementation, procedural/synthetic testing,
artifact preparation, and a non-submitting runner are authorized. Scientific
training, inference, corpus/manifest/checkpoint access, development access,
transfer, and submission are not authorized by this decision or its
implementation.

## Frozen execution design

An eventual separately authorized execution uses the accepted matched flat
chronological and position-free typed-graph encoders, shared decoder,
grid-ordinal geometry implementation, loss assembly, optimizer, and capacity
controls without scientific-design changes. Execution device is selected
prospectively as exactly `cpu` or `cuda:0` by timing-v2; CPU remains the
default for every caller outside this separately versioned Stage 6 path.

- operation-template train: exactly 407 families;
- operation-template development: exactly 45 E/R/EE/RE families;
- retained seeds: 2026, 2027, and 2028;
- fresh matched initialization for every arm/seed pair;
- 200 epochs, batch size 8;
- AdamW, learning rate `0.001`, weight decay `0.0`;
- global gradient clipping norm `1.0`;
- fixed epoch-200 checkpoint;
- no early stopping, warm start, development tuning, or checkpoint selection.

A resource fallback may retain only seeds 2026 and 2027. It must be decided
from prospective timing before results exist and recorded with
`observed_results_used=false`. Timing records both three- and two-seed
projections with the frozen 20% contingency. Three seeds are used when they
fit; two are used only when three do not fit and two do; neither fitting aborts.
Both fallback seed effects must be nonnegative.

## Narrow preparation and hardware timing

`GE1-STAGE6-NARROW-BUILDER-v1` prepares only the authoritative 407-family
train and 45-family development assignments. It runs the established strict
corpus and operation-template validators, copies only canonical continuous and
quantized E/R/EE/RE payloads into isolated roots, and publishes canonical
`GE1-STAGE6-NARROW-INDEX-v1` indexes only after the independent narrow loader
accepts them. A job-scoped preparation receipt binds both indexes, allowlists,
payload-digest aggregates, assignment hashes, and access declarations without
recording source absolute payload paths. A two-package request stages and
validates both sides before governed publication; no receipt means an
interrupted pair is nonauthoritative.

The current producer requires
`GE1-STAGE6-STRUCTURE-ONLY-TIMING-v2`; timing-v1 is historical and cannot
authorize it. One future separately authorized GPU-allocation timing job
benchmarks both `cpu` and `cuda:0` using one fresh untimed five-epoch warm-up
and three independently fresh five-epoch measurements per arm/device. It uses
the slowest seconds-per-epoch observation, projects the unchanged 200-epoch
two-arm budget, and applies the exact 20% contingency. Development and all
scientific outcomes remain unavailable to timing.

If either device fits all three seeds, retain all three and select the feasible
device with the smallest contingent three-seed projection. Only when neither
device fits three may timing select seeds 2026/2027 using the smallest feasible
two-seed projection. Exact timing ties prefer CPU. If neither fits two, abort
as resource-infeasible. Queue time, loss direction, predictions, accuracy, and
scientific results cannot affect selection.

CUDA is explicit and cannot fall back to CPU. It requires one visible GPU,
PyTorch 1.11.0 with CUDA, deterministic algorithms, seeded Python/NumPy/CPU and
CUDA RNGs, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, TF32 disabled, and recorded CUDA,
cuDNN, GPU-name, compute-capability, memory, host, and thread identities. The
producer must match the selected timing hardware identity. Checkpoints bind
the selected device and runtime, preserve CUDA RNG state on CUDA, recover on
the selected device through an explicit governed map location, and remain
portable to CPU only for a future separately authorized consumer.

## Primary structural endpoint

For a target history containing `O` operations, score the largest `k/O` for
which every one of the first `k` autonomous predicted operation groups passes:

1. grammar-valid complete retained node sequence;
2. exact operation types and chronological order;
3. exact canonical graph;
4. exact applicable `depends_on` relationships;
5. exact local sketch/profile/axis/reference ownership and attachment.

Grammar is computed from the actual retained node IDs and actual requested
node count under `V5_NODE_GRAMMAR`; it is never asserted by construction. Full
credit additionally requires the predicted operation-group count to equal
`O`. Under- and over-generation are recorded explicitly, while a correct
uninterrupted leading prefix below `O` retains partial credit.

Diagnostic first-failure order is grammar, operation types, chronology,
`depends_on`, ownership/attachments, then remaining canonical graph. The
relation-specific edge sets are subsets of the complete canonical graph but
remain separate for earlier diagnostic attribution.

Evaluation joins targets only after target-free autonomous generation.
Predictions may not be repaired or populated from targets. The primary score
is computed from preserved raw/constrained prediction evidence before strict
conversion or analytic validity.

Extrusion/revolve magnitude, remaining continuous geometry, conversion,
analytic validity, and CAD execution are excluded from primary pass/fail.
Their metrics remain report-only.

## Paired comparison

Within each retained seed, pair graph and flat scores by physical family and
compute the family-mean graph-minus-flat effect. Average those seed-level
effects; never pool seed replicas as independent families. The primary
development threshold is an improvement of at least `0.10`. Report every seed
and E/R/EE/RE template separately.

## Structural memory-use gate

For train and development independently, and for each arm independently, use
the structure-only primary score under true, shuffled, and batch-mean memory:

```text
P_true > 0
P_shuffle / P_true <= 0.80
P_mean / P_true <= 0.80
```

Geometry or conversion failure cannot zero or suppress these scores. A failed
memory gate makes the comparison inconclusive.

`P_shuffle` is a deterministic derangement within each stable inference batch,
seeded by experiment seed and batch identity. Every donor differs from its
recipient and belongs to the same batch; singleton batches fail closed.
`P_mean` is likewise batch-local.

## Secondary evidence

Report exact node and operation-type sequences; exact graph; `depends_on`
precision, recall, and exactness; reference/attachment correctness; first
structural failure; E/R/EE/RE results; train structural ceilings; optimization
and plateau diagnostics; capacity parity; true/shuffled/mean sensitivity; and
report-only geometry, magnitude, conversion, and analytic-validity metrics.

## Interpretation

- **Graph supported:** mean improvement is at least `0.10`, fallback seed
  conditions if applicable pass, and capacity, optimization, provenance,
  artifact, scoring, and both-arm memory gates pass.
- **Flat supported:** mean graph-minus-flat effect is at most `-0.10` and all
  validity gates pass.
- **Mixed:** the overall effect is strictly between `-0.10` and `0.10`,
  validity passes, and a preregistered seed or E/R/EE/RE template effect has
  absolute magnitude at least `0.10`.
- **Null:** validity passes and no overall or seed/template threshold is met.
- **Inconclusive:** training reliability, capacity, memory, provenance,
  artifact integrity, or structural scoring validity fails.

A positive outcome licenses only a statement about structural reconstruction
and generalization in this controlled comparison. It is not evidence of
geometry improvement, executable CAD, complete reconstruction, general
encoder superiority, or Stage 7 confirmation.

## Producer, artifacts, and prepared runners

`GE1-STAGE6-STRUCTURE-ONLY-PRODUCER-v1` is the separately versioned governed
producer. It accepts only exact declared read-only train/development roots and
hash allowlists, validates prospective timing before either loader, creates
fresh matched arms for each retained seed, trains through epoch 200, strictly
reloads `GE1-STAGE6-STRUCTURE-ONLY-CHECKPOINT-v1` into a fresh model, then
generates target-free `P_true`, `P_shuffle`, and `P_mean` records. Targets are
joined only after generation. Capacity and optimization gates are recorded,
but no scientific outcome controls artifact validity.

`GE1-STAGE6-STRUCTURE-ONLY-PRODUCER-ARTIFACT-v1` is an atomic six-file
execution-record artifact containing resolved governance/provenance, training
histories, checkpoint identities, family records, a manifest, and checksums.
The finalizer verifies and consumes this artifact without opening a corpus or
checkpoint.

The only scientific inputs are separately created, independently reviewed,
hash-approved `GE1-STAGE6-NARROW-INDEX-v1` train and development indexes plus
payload-only roots. Each binds the authoritative manifest hash, assignment
hash, family/variant matrix, metadata, sorted allowlist, and payload digests.
The separate loader reconstructs canonical histories and identities without
invoking or relaxing the complete-corpus loader. Index creation, review,
transfer, and access authorization remain separate work.

`GE1-STAGE6-STRUCTURE-ONLY-CHECKPOINT-BUNDLE-v1` atomically preserves the six
exact epoch-200 wrappers, or four only after a valid timing fallback, with
source/input identities, manifest, and checksums. The producer artifact binds
its digest and identities; the finalizer never reads checkpoint bytes.

All staging is job-scoped as `.incomplete-<job_id>`. Failed evidence cannot be
consumed as final, collide with another job, or authorize a retry. Development
remains unopened until all training, strict recovery, train interventions, and
train-side optimization, ceiling, and memory checks succeed.

`GE1-STAGE6-STRUCTURE-ONLY-ARTIFACT-v1` atomically finalizes exactly five
files: resolved configuration, family metrics JSONL, summary, manifest, and
`SHA256SUMS`. Verification recomputes every structural prefix, paired primary
effect, memory gate, validity gate, and interpretation.

The preparation runner binds only repository/corpus/output. The GPU hardware
timing runner uses `--nv`, exposes no development path, and binds only
repository/train/output. The existing CPU producer and separate GPU producer
each bind repository/train/development/output, require timing-v2 to select
their exact device, and reject the other selection. The GPU producer uses
`--nv`, requires the reviewed GPU identity, and reports peak device memory.
The finalizer remains CPU-only, tensor-independent, and checkpoint-blind. No
runner contains `sbatch`; timing and producer allocation flags remain future
reviewer inputs.

Implementation does not create a real package, run real timing, authorize data
access, or authorize execution. The remaining operational order is exactly:

```text
package creation -> hash review -> timing -> allocation review ->
producer -> producer audit -> finalizer
```

Exact train/development allowlists and SHA-256 values, their safe Adroit bind
layout, timing evidence, final allocation, source and runner hashes at a future
commit, transfer, data access, and submission remain unresolved reviewer
inputs. Their placeholders are deliberately ineligible for execution.

## Post-acceptance execution status

Later reviewer acts separately authorized narrow-package preparation,
timing-v2, and one CPU producer attempt. Producer job `3354961` trained all
six arm/seed conditions through epoch 200 and retained six strict checkpoint
wrappers, but stopped at `stage6_train_reliability_failure` before development
access or finalization. It therefore produced no Stage 6 comparison result.
Train-gate postmortem job `3355134` subsequently finalized a read-only,
train-only diagnostic artifact over those wrappers.

Read-only train-only zero-memory job `3355342`, at exact commit
`2eaabf2e5a88fc62f559c914123694539a93e188`, completed validly with the
evidence-bundle qualification recorded in its [execution
record](../experiments/ge1_stage6_zero_memory_3355342.md). Exact-zero memory
retained pooled structural scores of `0.8378378378378378` for flat and
`0.5855855855855856` for typed graph, versus authenticated true-memory scores
of `1.0` for both and authenticated mean-memory scores of
`0.7903357903357904` and `0.8361998361998362`. This is descriptive evidence
that unchanged decoder-side/scalar-path inputs remain informative and that
encoder-memory values also contribute. It creates no threshold and neither
passes nor repairs the frozen Stage 6 memory gate.

These later acts do not amend this ADR's endpoint, gates, or interpretation.
Development remains unopened; no producer artifact or finalized Stage 6
comparison exists. Another producer, repair, development access, finalizer,
RR, Stage 7, C8, or protected access requires a new explicit reviewer
decision.

## Stage 7 boundary

RR and Stage 7 remain closed. A one-time RR evaluation requires frozen Stage 6
code/metrics, finalized development results, fixed epoch-200 checkpoints, a
prospectively recorded RR endpoint/stopping rule, and separate reviewer
authorization. ER and every other protected partition remain closed.

## Consequences

The additive implementation provides scoring, governed production, strict
recovery, producer-artifact verification, and finalization without changing
any production head, initialization, loss, model, decoder, training semantics,
geometry behavior, or gate. Explicit resolution and review of every input,
timing, resource, transfer, and execution prerequisite remains necessary.
