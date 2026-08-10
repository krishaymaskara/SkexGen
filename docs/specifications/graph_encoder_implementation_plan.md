# Typed Graph Encoder Implementation Plan

*Accepted protocol for the next separately named SkexGen phase*

| Item | Decision |
|---|---|
| Status | Accepted protocol; C0-C6 complete; formal C7-v1 remains a scientific gate failure; optimization diagnostic job `3344907` supports undertraining in both arms; C7-v2 job `3344981` completed as a scaled exact-sufficiency failure; diagnostic job `3345013` isolated negative operation magnitudes; accepted ADR-0009 authorizes only prospective positive-magnitude implementation validation; Stage 6 remains unauthorized; C8 and later work not begun |
| Active scope source | [ADR-0002](../decisions/ADR-0002-three-week-flat-versus-graph-scope.md), July 30, 2026 |
| Historical motivation | Mentor-revised six-to-eight-week plan, July 16, 2026 |
| Authorization | Satisfied by accepted [ADR-0004](../decisions/ADR-0004-ge1-single-manifest-encoder-comparison.md) |
| Repository state used | Checked-in controlled corpus, model-data contract, Flat V6, and frozen Graph V1/C1 evidence |
| Experimental variable | Flat chronological Transformer encoder versus typed dependency-graph encoder |
| Shared downstream path | Latent interface, constrained node/geometry decoder, typed-edge decoder, conversion, losses, and evaluation |
| Controlled domain | One- and two-operation sketch/extrude/revolve histories |
| New phase name | `GE1-SHARED-DECODER-ENCODER-COMPARISON` |

## Purpose and relation to the research plan

The original research question is how CAD feature structure and geometry
should be represented to support systematic generalization and localized
editing. Hypothesis H1 predicts that a typed dependency graph will preserve
profile, axis, and feature relationships more reliably than a flat feature
sequence on unseen combinations and longer histories.

The completed Flat V6 versus Graph V1 comparison did not test H1. Both models
used the flat chronological encoder; they differed in their final structural
relation decoder. That decoder-side result is useful evidence, but it is not
an encoder comparison.

GE1 develops the encoder comparison required by the active three-week scope:

```text
flat chronological records -> flat Transformer encoder -----+
                                                            |
typed nodes and edges -----> typed graph encoder ------------+-> same latent interface
                                                                  -> same constrained node/geometry decoder
                                                                  -> same typed-edge decoder
                                                                  -> same strict converter
```

The primary GE1 question is:

> When the downstream decoder and training protocol are held constant, does
> encoding explicit typed CAD dependencies improve autonomous executable
> reconstruction and systematic generalization over encoding only the flat
> chronology?

This comparison changes both the information presented to the encoder and the
computation used to process it. It therefore tests **flat chronological versus
explicit typed-graph representation**, not a pure GNN-versus-Transformer
architecture effect. A pure architecture ablation would require giving both
encoders the same edge information and is outside the core comparison.

GE1 uses continuous program memory. Discrete VQ, a position-aware graph arm,
and learned editing are outside the core experiment.

## Governance and authorization

This specification cannot supersede an accepted decision record. ADR-0002
authorizes the shared-decoder flat-versus-graph scope. Accepted ADR-0004
supersedes ADR-0003 only for its immediate next-phase-selection gate,
preserves the Graph V1 freeze, and authorizes GE1 as a distinct encoder
question using decoder sufficiency and memory use as validity prerequisites.

ADR-0004:

- authorize GE1 as the next separately named phase;
- explain why decoder sufficiency is a prerequisite rather than a new
  decoder-side treatment;
- designate continuous-memory GE1 as the sole core encoder comparison;
- preauthorize the single decoder repair described in this plan, as
  prospectively superseded by accepted ADR-0009 after the C7-v2 diagnostic;
- freeze the systematic manifest, endpoint, access rule, and correction
  budget.

C0 is complete under the frozen
[Stage 0 preregistration](ge1_stage0_preregistration.md). The Stage 1 paired
data path is implemented, and its C3 authoritative Python 3.8/PyTorch 1.11 CPU
validation passed as Adroit job `3344235`. C4 implements the standalone flat
and typed-graph encoders. Its original authoritative CPU validation passed,
and the subsequent capacity-freezing and shared-initialization review fixes
were authoritatively revalidated on `adroit-h11n3` as job `3344265`: both new
tests, all 28 C4 encoder tests, and all 88 graph-encoder tests passed with zero
skips. C5 now implements the shared decoder, common loss, strict checkpoint,
and additive frozen output-position contract. Initial Adroit job `3344278` ran
all 22 focused tests with zero skips and zero errors; 19 passed and three
test-contract assertions failed. Those failures required test-only corrections
and no production model change. Corrected exact commit
`996016df44b7f9a6cd5c092a3e3b7a87d9964f9d` then passed authoritative Adroit
job `3344290`: all 22 focused C5 tests and all 110 graph-encoder tests passed
with zero skips, followed by every regression and repository gate. C5 is
complete. C6 implements the governed training, recovery, autonomous metrics,
and intervention machinery. Initial C6 Adroit job `3344337` failed the focused
suite because
the autonomous adapter passed a tuple instead of the C5-required local long
node-count tensor and one test omitted an error-class import. The narrow C6
correction passed those paths in exact-commit job `3344363`; 36/37 focused
tests passed with zero skips. Its sole error was a test-only generic equality
check on tensor-bearing dataclasses. After the tensor-aware, timing-neutral
test correction, exact-commit job `3344367` passed all 37 focused tests and all
147 graph-encoder tests with zero skips, every later runner gate, and both
authorized 407-family train-only arm smokes. The [C6 validation
record](../experiments/ge1_c6_cpu_validation.md) retains the complete evidence
and limitations. The later C5/C6 review fixes were then revalidated together
at exact commit `d29dc8299d32186907eb16d05c6102fcececf32e` by Adroit CPU job
`3344431`: 75/75 focused and 163/163 complete graph-encoder tests passed with
zero skips, followed by every regression and repository gate and both v2
407-family train-only arm smokes. The [combined validation
record](../experiments/ge1_c5_c6_review_fix_cpu_validation.md) retains the
evidence and limitations. C6 is complete. The subsequent C7 execution details
are frozen by accepted
[ADR-0006](../decisions/ADR-0006-ge1-c7-sufficiency-execution-contract.md).
C7 implementation now exists, and its formal train-only pilot completed as
Adroit job `3344505` at exact commit
`4bfde4c726a585433ea4bb60e6ce9d245ae7c87d`. Both arms failed the tiny
autonomous exact gate, all scaled gates remained `not_run`, and Stage 6 was
not authorized. The [C7 experiment record](../experiments/ge1_c7_pilot.md)
retains the complete evidence and limitations. RR and ER payload access
remains restricted by the accepted staged-access rules.

## Evidence that constrains the design

Only completed implementation and experiment evidence is used here. Older
repository suggestions about possible future paths do not govern this plan.

- The model-data package already exposes paired views of each physical family:
  flat records without edges and typed graph records with `edge_index` and
  `edge_type_ids`.
- Graph batches already provide concatenated nodes, graph offsets, edge
  offsets, graph IDs, and a dense node mask. No dataset-schema change is
  required.
- The schema-v1 representation stores `operation_sequence` as its authoritative
  execution-order record. In the controlled six-family subset, however, every
  valid later operation has exactly one `depends_on` edge to the immediately
  preceding operation. The one- or two-operation chain, local profile/sketch
  group, optional revolve axis, and shared plane are therefore recoverable
  from typed relationships without using tensor position.
- The current graph conversion does **not** perform that recovery. It scans
  node types in tensor order to reconstruct operation pointers, and the
  current model-data canonicalizer begins from `operation_sequence`. GE1 must
  add and test a graph-derived canonicalization step before claiming
  permutation invariance.
- Flat V6 and Graph V1 share the constrained node and geometry path. Graph V1
  supplies a typed-edge decoder and strict inverse conversion that can become
  the common GE1 downstream path.
- Frozen Graph V1/C1 solved all single-operation validation families but no
  two-operation family. Complete validity alone is therefore floor-bound on
  the systematic families most relevant to graph structure. GE1 needs a
  graded program-level primary endpoint while retaining complete validity as
  a required secondary endpoint.
- The existing serialization contains a strong position signal: a scoring-only
  position prior achieved 59/68 exact graphs. The shared decoder must be tested
  for dependence on encoder memory so that it cannot solve the task while
  ignoring either encoder.
- The graph pilots used one active VQ code with perplexity 1.0. Core GE1
  therefore bypasses quantization; discrete representation is not a core
  variable.
- The four split manifests are independent assignments over the same physical
  families. In the frozen 680-family corpus, the IID train partition contains
  89 RR and 89 ER families. It therefore cannot be combined with an
  operation-template systematic evaluation. GE1 uses the
  `operation_template` manifest exclusively for all loading, training,
  development, sufficiency, checkpoint handling, and evaluation.

## Single-manifest data authority

The expected frozen `operation_template` assignment is:

| GE1 role | Manifest partition | Templates | Families |
|---|---|---|---:|
| Train | `train` | E, R, EE, RE | 407 |
| In-distribution development | `validation` | E, R, EE, RE | 45 |
| Confirmatory systematic | `secondary_systematic_validation` | RR | 114 |
| Closed test | `test` | ER | 114 |

Stage 0 verified these counts and manifest hashes against the physical corpus
using metadata only. No IID, history-depth, or
geometry-extrapolation manifest partition may be loaded, including for
checkpoint selection, initialization, sufficiency checks, or timing.

Under this train pool, revolve appears alone and as the first operation in RE,
but never as the second operation. RR is therefore a hard compositional test
of a component observed individually and in another sequence position. The
45-family development comparison is the powered primary analysis. RR is a
separate one-time confirmatory endpoint and is never used to tune or select a
checkpoint.

## What is held constant

The two conditions must use the same:

- physical training families and family order;
- reconstruction targets;
- latent token count, latent width, and bottleneck mode;
- constrained node grammar and generated node path;
- category-conditioned compact profile heads;
- canonical profile, reference-plane, and revolve-axis construction;
- remaining categorical and continuous-geometry heads;
- typed-edge graph decoder architecture and legal-class mask;
- strict conversion and executability checks;
- objective definitions and weights;
- optimizer family, learning-rate schedule, batch size, number of example
  presentations, checkpoint rule, and random seeds;
- autonomous evaluation code and partition-access policy.

The flat control retains chronological order as part of its flat
representation. The primary typed graph encoder does not receive absolute or
chronological node-position embeddings, and it does not consume
`operation_sequence`. This asymmetry is intentional: the experiment compares
a chronological representation with an explicit typed-dependency
representation.

“Shared decoder” means one implementation and matched architecture trained
from scratch in each condition. It does **not** mean copying trained decoder
weights from one condition into the other.

## What changes

### Control: flat encoder

The control receives the existing `flat_mixed` input:

- padded chronological nodes `[B, N, 10]`;
- geometry and geometry mask `[B, N, 39]`;
- valid-node mask `[B, N]`;
- no dependency edges.

It retains the current input embeddings, serialized position embeddings,
latent queries, Transformer encoder, and existing `encode_to_memory(...)`
interface. The work is to wrap that existing interface in the GE1 model while
preserving its numerical behavior; V6 already reuses it.

### Treatment: typed graph encoder

The treatment receives the existing `typed_graph` input:

- concatenated node types `[sum(N)]`;
- nine categorical attributes `[sum(N), 9]`;
- geometry and geometry mask `[sum(N), 39]`;
- directed typed edges `[2, sum(E)]` plus edge types `[sum(E)]`;
- graph offsets and node-to-graph IDs used only for batching and graph
  membership.

It must not read chronological or absolute node positions,
`operation_sequence`, reconstruction targets, family labels, graph templates,
or systematic-partition metadata.

### Indices are bookkeeping, not semantic inputs

Local node indices remain necessary to address edge endpoints, gather and
scatter messages, construct padding masks, identify graph membership, and
align decoded outputs. `graph_offsets`, `edge_offsets`, and `node_graph_ids`
likewise remain necessary batching metadata.

These integers must never be embedded, projected, concatenated to node
features, or otherwise supplied as semantic values to the primary typed graph
encoder. Relabeling nodes by an arbitrary permutation, while applying the same
permutation to attributes, geometry, masks, and edge endpoints, must leave the
graph-level memory unchanged up to floating-point tolerance.

## Proposed typed graph encoder

### Node initialization

Reuse the flat encoder's field embeddings and geometry projections so that
both conditions begin from the same node content:

```text
h_i^0 = LayerNorm(
    sum(categorical field embeddings)
    + geometry projection(masked geometry)
    + geometry-mask projection
)
```

There is no position embedding in the primary typed graph encoder. Nodes are
distinguished through their attributes, geometry, typed neighborhoods, and
graph-level context rather than their row in a tensor.

### Typed message passing

Use a small relational message-passing network implemented in ordinary
PyTorch 1.11. Do not add PyTorch Geometric.

The authoritative edge direction is dependent object to referenced object.
For every edge `(source, destination, relation)`, create two message channels:

- reference-to-dependent: `destination -> source`, so an operation can receive
  information from its profile, axis, sketch, or prior operation;
- dependent-to-reference: `source -> destination`, so context can also flow in
  the reverse direction.

Use three residual relational layers, matching the maximum undirected
dependency diameter of the controlled graphs. Each layer performs degree-normalized
aggregation, adds a learned self projection, applies GELU, dropout, residual
addition, and LayerNorm. Relation transformations use two learned bases per
layer with relation-specific mixing coefficients. Three hops allow the
`depends_on` asymmetry of a two-operation history to reach the second
operation's profile and sketch and allow information to traverse the complete
controlled graph. Basis decomposition keeps the encoder close to the flat
encoder's parameter budget.

For node `i` at layer `l`:

```text
m_i = W_self h_i
      + sum over typed incoming channels r of mean(W_r h_j)
h_i_next = LayerNorm(h_i + dropout(GELU(m_i)))
```

Empty-edge graphs and isolated nodes remain valid because the self path is
always present.

### Graph-to-latent pooling

After message passing, use the same number and width of learned latent queries
as the flat encoder. A multi-head cross-attention block lets those queries
attend only to nodes in their graph. Apply a residual feed-forward block and
LayerNorm, then project to the common bottleneck dimension.

The encoder must output exactly:

```text
memory: [B, latent_tokens, model_dim]
```

Nothing downstream may branch on the encoder type.

## Deterministic graph canonicalization

The primary graph path must canonicalize a valid predicted or input graph
without consulting its incoming tensor order. For the controlled schema:

1. Validate node types, typed-edge compatibility, required cardinalities, and
   the absence of duplicate/self edges.
2. Extract operation nodes. The first operation has no outgoing `depends_on`
   edge; each later operation has exactly one such edge to its immediate
   predecessor. Reject branching, cycles, disconnected operations, or more
   than one valid chain under the GE1 scope.
3. Recover operation order by traversing that dependency chain from first to
   last.
4. For each operation, follow its unique `uses_profile` edge and the profile's
   unique `defined_in` edge to recover the local sketch. For a revolve, also
   follow its unique `uses_axis` edge and require that axis's `defined_in`
   target to be the same sketch.
5. Follow each sketch's unique `placed_on` edge. The controlled corpus has one
   shared reference plane, which is emitted first.
6. Emit each operation group in the existing canonical local order: sketch,
   profile, optional axis, operation.
7. Construct the old-index-to-canonical-index map; reorder node attributes,
   geometry, and masks; relabel all edge endpoints; sort edges by canonical
   source, edge class, and canonical destination; and derive
   `operation_sequence` from the recovered chain.
8. Reject any unassigned or multiply assigned node in the controlled scope.

This is a new GE1 adapter/conversion requirement. It does not retroactively
change schema v1's serialized authority or the frozen Graph V1 implementation.
If any otherwise valid future graph admits more than one execution order—for
example, independent operations without a body-dependency relation—order must
be represented explicitly by a semantic edge such as `depends_on` or a future
`precedes` relation. Tensor position may not silently resolve the ambiguity.

## Core continuous-memory bottleneck

Bypass nearest-code assignment while preserving latent token count and
dimensions:

```text
memory = from_codebook(to_codebook(encoded_queries))
```

This is the sole core H1 encoder experiment under the active shared-decoder
scope. It does not support a discrete-code claim. Its purpose is to establish
that:

- both encoders can drive the shared decoder;
- the decoder uses program memory;
- the typed graph path is wired correctly;
- failures are not being caused by a quantizer or collapsed codebook.

## Shared decoder selection and sufficiency gate

Create a new shared decoder module from the checked-in constrained V6
node/geometry path and Graph V1 typed-edge path. Frozen Graph V1 remains
unchanged as historical evidence.

Before the encoder comparison, the shared architecture must overfit a tiny
train-only set containing the four accessible operation templates (`E`, `R`,
`EE`, `RE`). The gate is:

- 100% exact node sequence;
- 100% exact typed graph;
- 100% strict conversion;
- 100% complete analytic validity;
- correct `depends_on` edge in every two-operation example.

This gate tests architectural sufficiency, not generalization. If it fails,
stop the comparison. Accepted
[ADR-0009](../decisions/ADR-0009-ge1-positive-operation-magnitude-repair.md)
prospectively supersedes the earlier hierarchical operation-group repair
authorization because C7-v2 was exact on node sequences, typed graphs,
attachments, dependencies, and strict conversion. Read-only diagnostic job
`3345013` instead isolated five negative active operation magnitudes. The only
authorized repair is therefore the shared, versioned positive mapping for
extrusion-distance and revolve-angle outputs. It is identical for both arms,
preserves every other decoder output and training choice, and must pass
implementation validation before any separately governed scientific
protocol. No repair is selected from development or protected data.

After the four-example smoke, run a scaled train-only sufficiency gate on 32
families selected deterministically as eight families from each of
`E/R/EE/RE`. Require the same exact-node, exact-graph, conversion, and
complete-validity criteria. Passing only the four-example gate is insufficient
because four independent memory-to-program mappings can be memorized without
establishing that the decoder handles repeated operation groups.

Because serialized output positions are highly predictive, run quantitative
memory-use checks on the 32-family sufficiency set and operation-template
development data using fully autonomous decoding:

1. shuffle memory across examples while retaining decoder prefixes;
2. replace memory with its batch mean.

Let `P_true`, `P_shuffle`, and `P_mean` be the mean normalized executable-prefix
scores under true, shuffled, and mean memory. Require
`P_shuffle / P_true <= 0.80` and `P_mean / P_true <= 0.80` in each arm, with
`P_true > 0`. Also report geometry error under all three conditions. A
teacher-forced version may be reported as a diagnostic but cannot satisfy the
gate. If the autonomous ratios fail, the experiment is inconclusive; do not
choose a new conditioning repair after observing comparative validation.

## Package and file plan

Do not edit the frozen `prototype.graph_baseline.GraphV1Model` identity. Add a
new package, `prototype/graph_encoder`, with the following responsibilities:

| Proposed file | Responsibility |
|---|---|
| `config.py` | GE1 model identity, encoder choice, bottleneck mode, dimensions, relation bases, and validation |
| `batching.py` | Paired flat/graph batches, index/offset bookkeeping, permutation utilities, and target-alignment checks |
| `canonicalization.py` | Position-independent dependency-chain recovery, canonical node order, edge relabeling, and operation-sequence derivation |
| `relational.py` | Dense-one-hot-free typed bidirectional message passing using `index_add_` |
| `encoders.py` | `FlatProgramEncoder`, `TypedGraphProgramEncoder`, and common encoded-memory result |
| `shared_decoder.py` | One constrained node/geometry plus typed-edge decoder implementation |
| `model.py` | Thin GE1 model selecting an encoder and calling the identical decoder |
| `losses.py` | Common loss assembly and per-example normalization |
| `training.py` | Paired seed/budget training, gradient checks, and checkpoints |
| `autonomous.py` | Encoder-specific input preparation followed by one common autonomous decoder/converter |
| `metrics.py` | Executable-prefix, failure-stage, exact-graph, validity, and memory-use diagnostics |
| `pilot.py` | C7 train-only sufficiency/memory gates and immutable artifacts; later development evaluation remains outside C7 |
| `tests/` | Unit, parity, integration, overfit, provenance, and access-control tests |

Existing code should be reused through imports where its identity is stable:

- `prototype.model_data` for both adapters and collators;
- `prototype.flat_baseline` embeddings, latent projections, constrained geometry, and grammar
  utilities;
- `prototype.graph_baseline.graph_contract` and `graph_tensors` for typed-edge
  vocabulary, targets, and legal masks;
- `prototype.graph_baseline.conversion` for strict graph conversion;
- `prototype.kernel_validation` for final executable-solid checks.

If reuse would require changing a frozen class, extract a behavior-preserving
shared helper and prove parity with the frozen implementation before GE1
training.

## Implementation sequence and acceptance tests

### Stage 0 — Freeze the protocol

1. Record accepted ADR-0004 and verify that the implementation configuration
   matches it exactly. The frozen values are recorded in the
   [GE1 Stage 0 preregistration](ge1_stage0_preregistration.md).
2. Use the model-arm identities and checkpoint schema/version assigned in the
   Stage 0 preregistration.
3. Record the primary endpoint, seeds, training budget, capacity tolerance,
   checkpoint-selection rule, correction budget, and partition-access rule.
4. Record which shared typed-edge decoder is used and exactly which output-side
   position signals it receives. Output alignment positions are distinct from
   semantic inputs to the graph encoder. The choice must be made before
   comparative results are inspected.
5. Freeze `operation_template` as the only GE1 manifest. Verify exactly 407
   E/R/EE/RE train families, 45 E/R/EE/RE development families, 114 RR
   systematic families, and 114 ER test families.
6. Hash all four operation-template assignments. Authorize only train and
   development payload loading before Stage 7; Stage 7 adds one-time RR
   access. ER and every partition of the IID, history-depth, and
   geometry-extrapolation manifests remain unauthorized throughout GE1.
   Verify zero payload access for all five protected sources at Stage 0.
7. Freeze the powered development endpoint, RR confirmatory endpoint,
   seed-combination rule, failure-stage tie-break, and effect thresholds.

**Exit:** the protocol is reviewable and no scientific choice depends on GE1
validation outcomes.

### Stage 1 — Paired data path

1. Build paired flat and typed-graph views from the same sorted physical
   examples.
2. Retain local indices and offsets only as edge-addressing and batching
   metadata; assert that they never enter graph node features.
3. Implement position-independent graph canonicalization and its
   old-index-to-canonical-index map.
4. Verify the graph and flat views have byte-equivalent targets after
   canonicalization.
5. Verify edge offsets, graph IDs, masks, and device moves for mixed lengths
   and empty edge sets.

**Tests:** train-accessible `E/R/EE/RE` batching, reversed input order, padding,
concatenated-index boundaries, no target leakage, and CPU real-tensor smoke.
Using procedurally constructed unit fixtures rather than protected corpus
payloads, test every `E/R/EE/ER/RE/RR` family: randomly permute nodes, relabel
every edge endpoint and node-aligned field consistently, and require
deterministic canonicalization to
recover the exact original executable node sequence, typed edges, geometry
alignment, and `operation_sequence`. Run multiple permutations, including
swaps between repeated same-type nodes in `EE` and `RR`.

**Exit:** both encoders receive different views of exactly the same families
and targets, and all six controlled families recover identically after
arbitrary consistent node permutation.

### Stage 2 — Relational encoder unit

**Implementation status:** C4 completes this stage with the frozen two bases,
ten directed channels, three layers, continuous memory contract, graph-local
pooling, and a capacity-matched flat wrapper. After the C4 review fixes,
authoritative Adroit CPU job `3344265` passed both new shared-initialization
tests, all 28 C4 encoder tests, and all 88 graph-encoder tests with zero skips.
The [review-fix validation
record](../experiments/ge1_c4_review_fix_cpu_validation.md) retains the
exact-commit and artifact audit. Stage 3/C5 passed its separate authoritative
runtime validation as Adroit job `3344290`.

1. Implement relation-basis mixing and both edge orientations.
2. Implement degree normalization with `index_add_`.
3. Add three residual layers and graph-local latent-query pooling.
4. Return common memory plus prequant diagnostics.

**Tests:** shape/dtype/contiguity, batch isolation, directed-edge sensitivity,
edge-type sensitivity, zero-edge behavior, finite gradients, deterministic
initialization, no target or `operation_sequence` argument in the encoder
signature, no position-embedding parameter in the primary graph encoder, and
graph-memory invariance after consistently permuting nodes and edges.

**Exit:** changing an edge direction or edge type changes treatment memory;
changing one graph never changes another graph's memory; changing only local
node numbering does not change treatment memory.

### Stage 3 — Shared decoder refactor

**Implementation status:** C5 implements this stage. The additive
[shared-decoder contract](ge1_shared_decoder_contract.md) freezes the parity
boundary and exact output-position inventory. Authoritative Adroit job
`3344278` reached all 22 focused tests with zero skips and zero errors, then
failed three test-contract assertions. The assertions were corrected without
changing production source. Corrected exact commit
`996016df44b7f9a6cd5c092a3e3b7a87d9964f9d` passed authoritative Adroit job
`3344290`, including 22/22 focused and 110/110 complete graph-encoder tests
with zero skips. The [validation
record](../experiments/ge1_c5_cpu_validation.md) retains the exact evidence.

1. Wrap the common latent-memory-to-history path.
2. Prove decoder-component parity on fixed common memory. The constrained V6
   decoder is authoritative for node and geometry calculations, and the
   initial Graph V1 main pair MLP is authoritative for typed-edge calculations.
   The later C1 additive directed-position-bias branch is excluded by the
   frozen Stage 0 choice. Complete GE1 flat-model equality to original Flat V6
   is neither expected nor required: the GE1 flat encoder has width 192,
   different initialization, and a continuous bottleneck, while original Flat
   V6 used width 64 and VQ. Encoder-caused differences are outside decoder
   parity.
3. Route both encoders through the same decoder class, forward implementation,
   output schema, and loss. Build one canonical initialization, then copy it
   into independent arm-specific decoder instances with disjoint parameters
   and buffers.
4. Keep raw, constrained, and converted predictions separately observable.
5. Freeze the four shared semantic output-position signals and distinguish
   them from permitted bookkeeping and prohibited graph-encoder chronology.
6. Define strict `GE1-CHECKPOINT-v1` reconstruction and state coverage.

**Tests:** exact intermediate node/geometry and main-pair parity; constrained
graph and conversion parity; first-difference diagnostics; state-dict coverage;
same decoder implementation and matched initial values; disjoint parameters
and buffers; arm-order-independent construction; common-memory-only routing;
per-example loss normalization; strict checkpoint reload; autonomous
raw/constrained/converted output; position-inventory routing; and absence of
target tensors in inference signatures.

**Exit:** swapping encoder choice changes only encoder modules and input
adapter; the decoder path is byte-identical code.

### Stage 4 — Capacity and optimization controls

1. Count encoder, decoder, bottleneck, and total parameters separately.
2. Tune only relation-basis count or feed-forward width to keep total trainable
   parameters within 5% between conditions.
3. Record that parameter matching does not match receptive field: the flat
   Transformer has global self-attention in one layer, whereas the primary
   graph encoder obtains a three-hop receptive field through three relational
   layers.
4. Plan three fixed seeds: `2026`, `2027`, and `2028`.
5. Under prospective ADR-0008, train every arm for 200 epochs, batch size
   eight, AdamW with learning rate
   `1e-3`, zero weight decay, and gradient clipping at `1.0`, unless the
   train-only engineering smoke rejects a setting before any comparative
   validation. No arm may receive extra presentations after comparative
   development evaluation begins. With 407 families this is 51 steps per
   epoch, approximately 10,200 optimizer steps, and exactly 81,400 example
   presentations per arm and seed, replacing the
   frozen two-epoch/136-step budget that measured only early optimization.
   Planned GE1 therefore contains approximately 61,200 optimizer steps across
   six runs.
6. Define train-loss plateau as less than 1% relative improvement in the
   five-epoch moving best after epoch 10. Report the plateau epoch, final train
   loss, train exact graph, train normalized executable prefix, and train
   complete validity for every arm and seed. If either arm has not plateaued
   by epoch 200, if the arms' normalized-prefix train-ceiling shortfalls differ
   by more than `0.05`, or if their complete-validity train-ceiling shortfalls
   differ by more than `0.10`, classify
   the encoder comparison as optimization-inconclusive rather than extending
   only one arm.
7. Select the fixed epoch-200 checkpoint for every arm and seed. Training loss,
   plateau epoch, and train ceilings remain required diagnostics but cannot
   select a checkpoint. Development data may not select checkpoints or
   hyperparameters.
8. Before freezing the full budget, time one train-only epoch for each arm on
   the intended hardware and record projected GE1 wall time, peak memory,
   artifact storage, and a 20% contingency. The timing run may not load
   development or protected payloads.
9. If projected resources are unavailable, reduce scope in this fixed order:
   reduce GE1 from three seeds to seeds `2026` and `2027`. Under the two-seed
   fallback, compute every seed mean over those two seeds and additionally
   require both seed-level treatment effects to be nonnegative. Do not reduce
   the 200-epoch budget. If two 200-epoch seeds
   per GE1 arm are infeasible, block the experiment rather than measuring
   early optimization again.
10. Report wall time and peak memory; parameter matching does not imply compute
   matching.

**Exit:** the capacity table and training budgets are frozen before the first
comparative pilot.

**C6 implementation status:** the measurement machinery for items 1, 3, and
8 is checked in under the additive
[C6 measurement contract](ge1_c6_measurement_contract.md). It also implements
the already frozen common loop, epoch checkpoints, diagnostic plateau formula,
autonomous train metrics, and recovery/provenance mechanics needed to test the
stage safely. This is not a claim that the full six-run experiment, Stage 5,
development evaluation, or protected evaluation has begun. C6 passed its
reviewed two-epoch 407-family train-only Adroit smoke for both arms as job
`3344367`.
Post-validation C5/C6 review fixes and accepted
[ADR-0005](../decisions/ADR-0005-ge1-primary-reporting-and-metrics-v2.md)
advance new metrics artifacts to v2. Job `3344367` remains valid historical
C6 evidence; combined exact-commit job `3344431` passed the required C5/C6
review-fix revalidation, including both v2 train-only artifacts. Formal C7
subsequently completed as a train-only scientific gate failure.

### Stage 5 — Train-only sufficiency and memory-use gates

1. Overfit one train family from each accessible template (`E/R/EE/RE`) in
   continuous mode.
2. Require the exact-node, exact-graph, conversion, and validity gates above.
3. Repeat the gate on the deterministic 32-family train-only set.
4. Run autonomous shuffled-memory and mean-memory interventions and apply the
   frozen 0.80 ratios.
5. Permit only the prospective positive operation-magnitude repair accepted in
   ADR-0009, applied identically to both conditions. The historical `tanh`
   mode remains explicit and checkpoint-incompatible. Implementation
   validation does not authorize scientific retraining or development access.

**Exit:** the decoder can represent the task and demonstrably uses memory.

**C7 implementation status:** accepted
[ADR-0006](../decisions/ADR-0006-ge1-c7-sufficiency-execution-contract.md)
freezes the metadata-only cohort ranking, seed-2026 fresh matched model
lifecycle, epoch-50 autonomous gates, scientific-failure semantics, and
immutable artifact format. `partitions.py`, `pilot.py`, the C7 CPU runner, and
focused C7 tests implement that contract. Implementation tests use only
synthetic metadata, procedural CAD fixtures, and temporary artifacts. Formal
Adroit job `3344505` completed the tiny gate at the frozen epoch-50
checkpoint: flat reached 4/4 exact nodes and typed graph 3/4, but both reached
0/4 exact graphs, 0/4 complete validity, and 0/2 exact `depends_on` on the
two-operation families. Both tiny gates therefore failed, all scaled and
memory gates were correctly `not_run`, and
`stage6_authorized_by_c7=false`. Development, RR, ER, and every unauthorized
manifest remain unopened by C7. C8 has not begun.

**Post-C7 diagnostic status:** accepted
[ADR-0007](../decisions/ADR-0007-ge1-c7-optimization-sufficiency-diagnostic.md)
governs an additive optimization-sufficiency trajectory prompted by the
non-plateaued update-50 losses. It reuses exactly the formal C7 four-family
train cohort and seed 2026, starts both arms freshly from the matched
initialization, trains uninterrupted through 500 optimizer updates, and
measures autonomous outputs only at updates 50, 100, 200, and 500. Corrected
Adroit job `3344907` at commit
`fbc6073f63da9f0e10b5db8c0c0d4786a48ce0c0` completed successfully. Both
unchanged arms first became autonomously exact-sufficient at update 200 and
remained exact at update 500, yielding
`undertraining_supported_both_arms`. It did not change formal C7-v1, invoke
repair, authorize Stage 6, open protected data, or begin C8. See the
[diagnostic contract](ge1_c7_optimization_sufficiency_diagnostic.md) and
[result record](../experiments/ge1_optimization_diagnostic.md).

**C7-v2 status:** accepted
[ADR-0008](../decisions/ADR-0008-ge1-c7-v2-200-epoch-protocol.md) freezes a
separate `GE1-C7-SUFFICIENCY-v2` execution at epoch 200. Tiny arithmetic is
200 updates and 800 presentations per arm; scaled arithmetic is 800 updates
and 6,400 presentations per arm. The unchanged C7-v1 cohorts, criteria,
memory thresholds, seed, models, decoder, losses, access rules, and failure
rules remain in force. Tiny and scaled use separately fresh matched pairs.
Job `3344981` at commit
`e325d5ad97957c08da4a19b4261560e8a4a472a4` passed both tiny gates, then
failed scaled exact sufficiency only because two flat and three typed-graph
families had analytic `invalid_operation_parameter` failures. Both scaled
memory gates passed; all 64 arm-family node/graph/strict-conversion criteria
and all applicable `depends_on` criteria were exact. The result blocks Stage
6 and makes the decoder-repair path procedurally next without implementing
repair. See the [C7-v2 contract](ge1_c7_v2_execution_contract.md).

**Pre-repair diagnostic status:** Krishay Maskara authorized a separately
versioned, train-only read-only diagnostic of those five failures. It loads
the immutable scaled epoch-200 checkpoints, reproduces original `P_true`
metrics before interpretation, and records both-arm scalar, mask, category,
conversion, and analytic-validity traces. Because no CAD kernel is part of
the evaluation contract, executor outcomes are structurally unavailable and
legal-but-geometrically-incompatible values are unassessable. The
[diagnostic contract](ge1_c7_v2_operation_parameter_diagnostic.md) was
executed read-only as Adroit job `3345013` at exact commit
`802ae1d1e9deb3c7a6c428d320e4276a8b5e7e57`. All five failures were negative
active extrusion-distance or revolve-angle magnitudes, while masks, channel
selection, scaling, units, direction, Boolean mode, targets, and alignment
were consistent. No training, source-checkpoint mutation, protected access,
Stage 6, or C8 occurred.

**Prospective repair status:** accepted
[ADR-0009](../decisions/ADR-0009-ge1-positive-operation-magnitude-repair.md)
replaces only the earlier hierarchical repair authorization. It versions the
historical behavior as `GE1-OPERATION-MAGNITUDE-TANH-LEGACY-v1` and the
prospective behavior as `GE1-OPERATION-MAGNITUDE-POSITIVE-v1`. Only compact
operation channels 4 and 5, serialized as extrusion distance 37 and revolve
angle 38, use `finfo(dtype).tiny + (1 - finfo(dtype).tiny) * sigmoid(raw)`.
All other geometry channels retain exact `tanh`. Configuration, checkpoints,
provenance, and reporting distinguish the modes and reject incompatible
loads. This implementation work does not authorize a repaired C7 run or
Stage 6.

### Stage 6 — Core continuous encoder pilot

1. After explicit C7-v2 authorization, train both conditions from fresh
   initialization for 200 epochs under the
   frozen retained seed set: normally 2026, 2027, and 2028, or 2026 and 2027
   if the preauthorized timing fallback was invoked and recorded.
2. Select the fixed epoch-200 checkpoint for every arm and retained seed.
3. Compare the graded normalized executable-prefix endpoint overall and by
   `E/R/EE/RE` on the 45-family operation-template development partition.
4. Report complete validity, exact graph, `depends_on` recall, node/geometry
   metrics, conversion failures, train-set ceilings, and autonomous memory
   sensitivity as required secondary evidence.

**Exit:** both paths train reliably and the powered development result is
reported. RR confirmation remains unknown until Stage 7.

### Stage 7 — One-time systematic evaluation

Open the predetermined systematic partition only after:

- code and configs are frozen and committed;
- both conditions pass autonomous inference, strict conversion, and artifact
  validation;
- the fixed epoch-200 rule has identified one checkpoint per arm and retained
  seed without development or systematic data;
- the exact systematic endpoint and stopping rule are recorded;
- the `operation_template` manifest and RR
  `secondary_systematic_validation` partition hashes match Stage 0.

Run each frozen GE1 checkpoint pair once on RR. Do not tune on the result.
The operation-template ER test and every IID, history-depth, and
geometry-extrapolation partition remain closed throughout GE1.

**Exit:** the project has the first defensible answer to H1 under this
controlled domain.

## Primary and secondary endpoints

### Powered primary development endpoint

Autonomous **normalized longest executable operation prefix per physical
family** on the frozen 45-family operation-template development partition.

For a history containing `O` operations, deterministically canonicalize the
generated graph, then evaluate prefixes containing the first `k` complete
operation groups for every `k` from zero through `O`. The family score is
`max(k) / O` for prefixes that strictly convert and pass analytic validity.
Thus a two-operation history scores `0`, `0.5`, or `1.0`. Prefix construction
may remove only later complete operation groups and their exclusively owned
sketch/profile/axis nodes; it may not repair predictions, substitute target
fields, or alter the retained prefix.

The powered primary threshold is a treatment-minus-control improvement of at
least `0.10`, computed by first taking the paired family mean within each
retained seed and then averaging the seed-level means. The planned analysis
uses three seeds. If the preauthorized timing fallback is invoked, it uses
seeds 2026 and 2027 and requires both seed-level effects to be nonnegative.
Report every seed separately; do not pool seed outputs as independent
families.

### Confirmatory RR systematic endpoint

Apply the same normalized-prefix metric once to the 114 RR families. The
confirmatory threshold is a retained-seed-mean treatment-minus-control
improvement of at least `0.10`, an average of at least 25% of paired RR
families advancing by one complete operation, and an average of no more than
5% regressing by one operation. Fractions are computed within seed and then
averaged across seeds.
These conditions are arithmetically consistent on the RR score grid of
`0/0.5/1.0`: 25% advancing and 5% regressing yields a net mean change of
`0.10` when every change is one operation.

### Pre-registered RR tie-break

If the RR normalized-prefix seed means are exactly tied, compare normalized
stage-of-first-failure progress using this fixed ordered pipeline:

1. grammar-valid complete node sequence;
2. graph canonicalization succeeds;
3. local profile/sketch/axis/plane references satisfy cardinality;
4. the operation dependency chain is valid;
5. strict history conversion succeeds;
6. the first operation passes analytic validity;
7. the second operation and complete history pass analytic validity.

Each family receives the highest completed stage divided by seven. Compare
arms using the same paired-family-then-seed-mean calculation. A tie-break
improvement describes later failure and supports a mixed-result diagnosis; it
cannot convert a failed primary or confirmatory threshold into “graph encoder
supported.” If this score is also tied, report a resolved null tie rather than
introducing another metric.

### Required secondary endpoints

- complete valid CAD history rate;
- exact typed-graph match;
- exact node sequence;
- `depends_on` precision/recall and exactness;
- profile, axis, and reference attachment accuracy;
- strict conversion rate and first-failure histogram;
- normalized stage of first failure and unnormalized longest executable prefix;
- valid single solid rate;
- applicable geometry error by channel family;
- parameter counts, example presentations, wall time, and seeds;
- memory-shuffle degradation;

Pairwise edge accuracy and edge F1 are diagnostics only. They cannot override
the graded program-level primary endpoint or complete-validity secondary
evidence.

## Decision rules

- **Graph encoder supported:** treatment meets the powered development
  threshold and the confirmatory RR threshold with no material loss of
  geometry accuracy or conversion reliability, and the improvement is not
  explained by capacity or training budget.
- **Mixed result:** treatment improves development, the RR tie-break, particular
  RR subgroups, or dependency relations but does not satisfy both endpoint
  thresholds. Report the boundary without broadening the claim.
- **Null result:** both encoders perform similarly after passing sufficiency
  and memory-use gates. Conclude that explicit graph encoding did not help in
  this controlled setting and budget.
- **Inconclusive:** decoder ignores memory, capacity control fails, either arm does not
  reach a comparable train plateau, the
  shared decoder cannot pass the 32-family sufficiency gate, or governance and
  partition rules are not satisfied. Do not interpret this as evidence
  against graph representations.

## Scope exclusions

Core GE1 does not add discrete VQ, a position-aware graph arm, learned editing,
fillet, language conditioning, real-CAD data, more than two operations,
multiple graph architectures, a learned code prior, or a new geometry
representation. It does not modify frozen Flat V6 or Graph V1 evidence. These
exclusions keep the result tied to one encoder question.

## Optional post-core extensions

None of the following belongs to GE1's implementation sequence, compute
budget, success criteria, or protected-data authorization. An extension may
begin only after Stages 0–7 are complete, the development and RR results are
frozen in a durable report, and the core GE1 code and checkpoints are
immutable. Each extension requires its own reviewed protocol, experiment
identity, correction budget, compute estimate, checkpoint rule, and data-access
decision. Its results cannot retroactively alter GE1 model selection or its
pre-registered interpretation.

### Optional discrete-VQ experiment

A later experiment may replace continuous memory with a discrete bottleneck.
It must first perform a train-only information-capacity audit, resize the
bottleneck if needed, initialize from train-only prequant vectors, measure
joint-code rather than only marginal utilization, and define achievable
entropy and autonomous memory-sensitivity gates. It receives no automatic RR
or ER access from GE1 and is a separate bottleneck experiment, not another
stage of the encoder comparison.

### Optional position-aware graph ablation

A later ablation may add chronological-position embeddings to the typed graph
encoder while retaining the frozen downstream path. It must be labeled as a
position-aware ablation and compared under a new matched protocol. It cannot
replace or revise the position-free primary graph condition.

### Optional learned-editing experiment

A later experiment may define an explicit requested-edit interface and use the
existing counterfactual pairs to measure target-change success, unaffected
node/edge preservation, unrelated-geometry preservation, strict execution
validity, and locality. Reconstruction or latent-code usage alone is not
evidence of learned editing.

## Recommended first implementation slice

The first code milestone should include only:

1. paired flat/graph batching;
2. deterministic graph canonicalization and the six-family permutation suite;
3. the position-free three-layer basis-decomposed relational encoder;
4. graph-local latent-query pooling;
5. continuous bottleneck mode;
6. shared-decoder parity tests;
7. the four-template and 32-family overfit gates plus autonomous quantitative
   memory interventions.

Do not begin a full comparative training run until that slice passes. It is
the shortest route to learning whether the proposed graph encoder is wired
correctly and whether the existing downstream path can expose an encoder
difference.
