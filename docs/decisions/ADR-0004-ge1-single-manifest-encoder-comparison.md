# ADR-0004: GE1 Single-Manifest Encoder Comparison

- Status: `accepted`
- Proposal date: `2026-08-07`
- Decision date: `2026-08-07`
- Owner: project research team
- Designated GE1 reviewer: Krishay Maskara
- Supersedes: the next-phase-selection gate in [ADR-0003](ADR-0003-freeze-graph-v1.md) only
- Superseded by: [ADR-0009](ADR-0009-ge1-positive-operation-magnitude-repair.md)
  for the decoder-repair choice only; every other decision remains in force

## Context

[ADR-0002](ADR-0002-three-week-flat-versus-graph-scope.md) authorizes one
shared-decoder comparison between flat chronological and typed graph
representations. [ADR-0003](ADR-0003-freeze-graph-v1.md) freezes Graph V1
after C1 and requires any later neural work to begin as a separately named
phase. ADR-0003 also identifies global operation grouping and VQ collapse as
the immediate unresolved hypotheses. GE1 instead asks whether the input
encoder representation affects reconstruction and compositional
generalization after decoder sufficiency is established.

A durable decision is required before implementation because the four corpus
manifests are independent assignments over the same 680 physical families.
The IID train partition contains RR and ER families; it cannot be used to
train a model later evaluated on RR under the operation-template manifest.
Doing so would turn a claimed systematic evaluation into a partially seen
training evaluation.

Deterministic regeneration from the checked-in seed-2026 corpus contracts
produced these exact operation-template assignments:

| Partition | Templates | Families |
|---|---|---:|
| `train` | E, R, EE, RE | 407 |
| `validation` | E, R, EE, RE | 45 |
| `secondary_systematic_validation` | RR | 114 |
| `test` | ER | 114 |

The same audit found 89 RR and 89 ER families in IID train. A metadata-only
audit of the authoritative physical operation-template manifest subsequently
verified its counts and hashes without opening referenced sample payloads.
The frozen Stage 0 values and access audit are in the
[GE1 preregistration record](../specifications/ge1_stage0_preregistration.md).

Graph V1 and C1 produced no valid two-operation IID-validation programs.
Complete validity alone is therefore likely to be floor-bound on RR. A graded
program-level endpoint and a frozen tie-break are needed before protected
systematic access.

The accepted implementation protocol is the
[typed graph encoder implementation plan](../specifications/graph_encoder_implementation_plan.md).

## Decision

This decision is binding for GE1 implementation and evaluation.

### Experimental identity and scope

Authorize a separately named phase:

```text
GE1-SHARED-DECODER-ENCODER-COMPARISON
```

GE1 compares exactly two conditions:

1. flat chronological Transformer encoder;
2. position-free typed dependency-graph encoder.

Both conditions train the same downstream decoder implementation and use the
same continuous latent-memory interface, targets, losses, optimizer policy,
conversion, metrics, and partition rules. “Shared decoder” means matched code
and architecture trained from scratch in each arm, not shared learned weights.

Discrete VQ, a position-aware graph arm, and learned editing are outside core
GE1. They may be proposed only after core GE1 is complete and frozen, under
new protocols and data-access decisions.

This ADR does not unfreeze Graph V1, authorize Graph V1 C2, or alter any
completed checkpoint or experiment record.

### Single-manifest data authority

GE1 uses only `manifests/operation_template.json`:

- train: 407 E/R/EE/RE families;
- development: 45 E/R/EE/RE families;
- one-time confirmatory systematic evaluation: 114 RR families;
- closed test: 114 ER families.

No partition from the IID, history-depth, or geometry-extrapolation manifest
may be loaded for training, initialization, sufficiency, timing, checkpoint
selection, development analysis, or evaluation. ER remains closed throughout
GE1. RR remains closed until the one-time systematic stage.

The train-only tiny sufficiency set contains one family from each accessible
template E/R/EE/RE. The scaled sufficiency set contains 32 train families,
selected deterministically as eight from each accessible template. No RR or
ER physical payload may enter either set. Procedurally constructed unit
fixtures may cover all six templates without opening protected corpus
payloads.

Under this partition, revolve is observed alone and as the first operation in
RE, but never as the second operation. RR is therefore a compositional test of
a known component in a withheld sequence position.

### Architecture and capacity controls

The typed graph encoder receives no chronological or absolute node-position
embedding and does not consume `operation_sequence`. Node indices,
`graph_offsets`, edge endpoints, masks, and graph IDs remain bookkeeping only.
The graph encoder uses three typed bidirectional relational layers, sufficient
for the maximum undirected diameter of the controlled graphs, followed by
graph-local latent-query pooling.

Total trainable parameter counts between arms must differ by no more than 5%.
Only relation-basis count or encoder feed-forward width may be adjusted to
meet that tolerance. Parameter matching does not equate receptive fields and
the final report must state that limitation.

The shared decoder must first pass exact train-only sufficiency gates. If the
initial independent pairwise decoder fails, exactly one preauthorized common
repair is permitted before comparative development results: a hierarchical
operation-group decoder with one-to-one local-reference masks and an explicit
linear operation-dependency chain. No other decoder repair is authorized.

### Training and checkpoint decision

The planned comparison uses seeds 2026, 2027, and 2028. Each arm and seed uses:

- 50 epochs;
- batch size 8;
- AdamW, learning rate `1e-3`, zero weight decay;
- gradient clipping at `1.0`;
- 51 steps per epoch;
- 2,550 optimizer steps and 20,350 example presentations per run.

GE1 therefore plans 15,300 optimizer steps across six core runs.

Train-loss plateau means less than 1% relative improvement in the five-epoch
moving best after epoch 10. If either arm has not plateaued by epoch 50, the
comparison is optimization-inconclusive. It is also
optimization-inconclusive if the arms' normalized executable-prefix
train-ceiling shortfalls differ by more than 0.05 or their complete-validity
train-ceiling shortfalls differ by more than 0.10.

The selected checkpoint is fixed at epoch 50 for every arm and seed. Training
loss and plateau metrics remain required diagnostics but cannot select a
checkpoint. Development data may not select checkpoints or hyperparameters.

Before freezing the full run budget, time one train-only epoch for each arm on
the intended hardware and project wall time, peak memory, artifact storage,
and a 20% contingency. If resources are insufficient, first reduce from three
seeds to seeds 2026 and 2027. Under this fallback, both seed-level treatment
effects must be nonnegative. Do not shorten the 50-epoch budget below the
plateau requirement. If two seeds per arm are infeasible, GE1 remains blocked.

### Memory-use validity gate

On the 32-family train sufficiency set and the development set, run autonomous
decoding under true, batch-shuffled, and batch-mean memory. If `P` is mean
normalized executable-prefix score, require in both arms:

```text
P_shuffle / P_true <= 0.80
P_mean / P_true <= 0.80
P_true > 0
```

Teacher-forced degradation cannot satisfy this gate. Failure makes the encoder
comparison inconclusive and does not authorize a post-hoc conditioning repair.

### Powered development endpoint

The powered primary analysis is autonomous normalized longest executable
operation prefix on the 45-family operation-template development partition.
For a history with `O` operations, the score is the greatest strictly
convertible and analytically valid leading operation count divided by `O`.

The primary threshold is treatment minus control of at least 0.10. Compute a
paired family mean within each retained seed, then average seed-level means.
Do not pool seed outputs as independent families.

### Confirmatory RR endpoint

After code, checkpoints, and analysis rules are frozen, evaluate each selected
checkpoint once on the 114 RR families. The confirmatory threshold requires:

- retained-seed-mean treatment minus control of at least 0.10;
- at least 25% of paired RR families advancing by one complete operation on
  average across seeds;
- no more than 5% regressing by one operation on average across seeds.

Complete valid-history rate remains a required secondary endpoint.

### RR tie-break

If the RR normalized-prefix seed means are exactly tied, compare normalized
stage-of-first-failure progress using this fixed order:

1. grammar-valid complete node sequence;
2. graph canonicalization;
3. valid local profile/sketch/axis/plane references;
4. valid operation dependency chain;
5. strict conversion;
6. analytically valid first operation;
7. analytically valid second operation and complete history.

Each family receives the highest completed stage divided by seven. Use the
same paired-family-then-seed-mean calculation. The tie-break may support a
mixed-result diagnosis but cannot turn failed primary or confirmatory
thresholds into support for the graph encoder. If it also ties, report a null
tie without adding another metric.

### Decision rules

- **Graph encoder supported:** both powered development and confirmatory RR
  thresholds pass, without material loss of geometry accuracy or conversion
  reliability and without violating capacity, optimization, or memory-use
  gates.
- **Mixed:** one endpoint, the tie-break, a subgroup, or a dependency metric
  improves, but both main thresholds do not pass.
- **Null:** both arms satisfy validity gates but do not show a qualifying
  treatment advantage.
- **Inconclusive:** governance, access, decoder sufficiency, capacity,
  optimization, memory-use, provenance, or artifact validation fails.

Pairwise edge accuracy and edge F1 are diagnostics and cannot override the
program-level endpoints.

## Alternatives considered

### Train on IID and evaluate RR from operation-template

Rejected because manifest assignments overlap. In the frozen deterministic
assignment, IID train contains 89 RR families and 89 ER families.

### Use operation-template only for evaluation

Rejected because checkpoint selection, sufficiency, initialization, or timing
on another manifest can still load protected RR or ER families. The manifest
authority must cover the whole phase.

### Complete validity as the only primary endpoint

Rejected because every frozen learned arm has zero valid two-operation
programs. Normalized executable prefix supplies resolution above that floor;
complete validity remains required secondary evidence.

### Include discrete VQ, a position-aware arm, or editing in core GE1

Rejected because each changes the scientific question and compute burden.
They remain optional post-core proposals only.

### Reuse the frozen two-epoch budget

Rejected because 136 optimizer steps measure early optimization rather than a
stable representation comparison. The longer budget is paired with explicit
plateau and compute-feasibility gates.

## Consequences

### Positive

- RR and ER are protected from training and checkpoint leakage.
- The phase asks one encoder question with one continuous bottleneck.
- The primary metric has resolution when complete validity remains at zero.
- Compute, capacity, checkpointing, correction, and tie-break decisions are
  frozen before comparative results.
- Negative, mixed, null, and inconclusive outcomes have explicit meanings.

### Tradeoffs and limitations

- The train set shrinks from 544 IID families to 407 operation-template
  families.
- RR is deliberately difficult because revolve is never second during
  training.
- Development contains only 45 families.
- A fixed epoch-50 checkpoint may not maximize executable development
  performance, but it removes checkpoint-selection adaptivity and preserves
  development as an analysis rather than a selection set.
- Three relational layers and a Transformer control do not have matched
  receptive-field structure even when parameters are matched.
- The graded prefix endpoint does not replace full executable-history
  validity.

## Validation and evidence

This proposal is grounded in:

- the independent assignment construction in
  `prototype/controlled_data/splits.py`;
- the 680-family corpus configuration in
  [the corpus record](../experiments/b0_pilot_corpus_680.md);
- the frozen Flat V6, initial Graph V1, and C1 records;
- the accepted schema and scope decisions in ADR-0001 through ADR-0003; and
- deterministic local regeneration of the seed-2026 assignment counts.

C0 verified the physical `operation_template.json` counts and hashes without
opening RR or ER payloads. Protected history payload access remains governed
by the frozen staged-access policy.

## Implementation contract

Acceptance requires the repository to:

- mark ADR-0003 as superseded by ADR-0004 only for its immediate next-phase
  selection gate; preserve its Graph V1 freeze and evidence decisions;
- update `docs/status.md` to name GE1 as authorized but not implemented;
- freeze the full Stage 0 protocol values and hashes in a GE1 configuration or
  preregistration record;
- do not modify `prototype.graph_baseline.GraphV1Model`;
- create new GE1 code only under a separately named package and checkpoint
  identity;
- preserve zero RR, ER, and non-operation-manifest payload access until the
  authorized stage.

## Review conditions

ADR-0004 was accepted by Krishay Maskara, the designated GE1 reviewer, on
August 7, 2026, subject to fixed epoch-50 checkpoint selection; that condition
is incorporated above. The acceptance changed only the checkpoint rule, so
the preauthorized two-seed timing fallback remains in force. C0 is complete;
Stage 1 has not begun. RR and ER payload access remains unauthorized until the
accepted protocol's respective access stage.

A later change to manifest authority, model arms, decoder repair, training
budget, seeds, fixed epoch-50 selection, endpoint thresholds, tie-break, or
protected-data access requires a new superseding ADR rather than an edit to
this accepted record.
