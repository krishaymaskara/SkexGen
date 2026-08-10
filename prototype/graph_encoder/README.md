# GE1 Graph Encoder Package

## Implemented scope

This package implements the C1 boundary, C2 position-free graph canonicalizer,
C3 paired flat/graph batching, C4 continuous flat and position-free typed
graph encoders, C5 shared decoder integration, C6 training/measurement
machinery, and the C7 train-only sufficiency/memory gate implementation for
`GE1-SHARED-DECODER-ENCODER-COMPARISON`. It also implements the separately
identified post-C7 optimization-sufficiency diagnostic, C7-v2 and its
read-only operation-parameter diagnostic, and the prospective versioned
positive operation-magnitude repair for engineering validation only.

C1 provides:

- immutable, deterministic model and training configuration;
- exact frozen identities and encoder-to-arm derivation;
- operation-template-only manifest authority verification;
- train and complete-development payload access wrappers;
- unconditional protection of RR, ER, and every non-operation manifest; and
- stable GE1 boundary errors.

C2 provides structural validation and deterministic canonicalization of one
controlled typed graph at a time.

C3 provides deterministic paired-family sorting, target-separated flat and
graph encoder inputs, graph bookkeeping, inherited flat/graph collation
validation, and an explicit deterministic diagnostic permutation utility.

C4 provides:

- a three-layer, two-basis relational encoder with ten directed semantic
  channels;
- graph-local latent-query cross-attention pooling;
- a capacity-matched wrapper around the inherited flat Transformer encoder;
- a common continuous-memory result that bypasses nearest-code assignment;
- node-level permutation-equivariance diagnostics and graph-memory
  permutation-invariance tests; and
- exact component and arm-level parameter accounting.

C5 provides:

- one shared constrained-V6 node/geometry plus initial-Graph-V1-main decoder;
- independent arm decoder objects copied from one canonical initialization;
- explicit raw, constrained, and converted autonomous prediction levels;
- one per-example normalized typed-graph loss;
- a frozen decoder output-position inventory; and
- strict `GE1-CHECKPOINT-v1` save and reload.

C6 provides the common training loop, full recovery checkpoints, provenance,
target-free autonomous interventions, executable-prefix and secondary metrics,
family-macro aggregation, and resource measurements.

C7 provides:

- authoritative-metadata-only nested 4/32-family `E/R/EE/RE` selection;
- fresh matched seed-2026 model pairs for tiny and scaled gates;
- strict epoch-50 autonomous per-family exactness and scaled memory-use gates;
- explicit pass/fail/not-run dependency and Stage 6/repair decisions;
- a C7-native metrics envelope without mutating the C6 smoke identity;
- atomic self-verifying artifacts with complete checkpoints; and
- a fail-fast Python 3.8.13/PyTorch 1.11 CPU Slurm runner.

The formal C7 pilot completed as Adroit job `3344505`. Both arms failed the
four-family autonomous exact gate, all scaled gates remained `not_run`, and
Stage 6 was not authorized. The package does not implement C8 decoder repair,
protected evaluation access, or a later scientific result.

The additive post-C7 diagnostic uses the same four tiny train families and a
fresh seed-2026 matched pair, trains both arms continuously through 500
optimizer updates, strictly reloads measurements at updates 50/100/200/500,
and retains recovery checkpoints every 25 updates. Adroit job `3344907`
completed at exact commit `fbc6073f63da9f0e10b5db8c0c0d4786a48ce0c0`;
both unchanged arms first became autonomously exact-sufficient at update 200,
yielding `undertraining_supported_both_arms`. It did not change formal C7-v1,
authorize Stage 6, invoke repair, or begin C8.

Accepted ADR-0008 defined the separate fixed-epoch-200 C7-v2, completed as
job `3344981`. Read-only diagnostic job `3345013` then isolated negative
active operation magnitudes in all five failures. Accepted ADR-0009
supersedes only the earlier hierarchical repair authorization and permits the
prospective positive operation-magnitude implementation described below. No
repaired scientific run is authorized, and Stage 6 remains blocked.
The frozen
`prototype.flat_baseline` and
`prototype.graph_baseline` packages are reused by import only and remain
unchanged.

## Public API

```python
from prototype.graph_encoder import (
    C7SufficiencySelection,
    CanonicalizedGraph,
    C6TrainingError,
    FlatEncoderInput,
    GE1Config,
    GE1Model,
    GE1TrainingConfig,
    GraphBookkeeping,
    GraphCanonicalizationInput,
    GraphEncoderError,
    GraphNodeContent,
    GraphSemanticInput,
    MEMORY_CONDITIONS,
    METRICS_SCHEMA_VERSION,
    PRIMARY_REPORTING_CONTRACT_VERSION,
    PairedBatch,
    SharedGE1Decoder,
    authorize_c6_provenance,
    autonomous_input_from_paired,
    build_paired_batch,
    build_ge1_model,
    build_matched_ge1_models,
    canonicalize_graph,
    capacity_difference_percent,
    c7_family_rank,
    common_ge1_loss,
    default_encoder_config,
    encoder_parameter_report,
    frozen_encoder_config,
    frozen_feedforward_width,
    intervention_ratios,
    load_training_checkpoint,
    load_development,
    load_ge1_checkpoint,
    load_train,
    permute_graph,
    plateau_state,
    parameter_count_record,
    peak_memory_record,
    prefix_score_from_outcomes,
    receptive_field_record,
    run_autonomous_evaluation,
    run_ge1_training,
    save_ge1_checkpoint,
    save_training_checkpoint,
    score_condition,
    score_prediction_prefix,
    select_c7_sufficiency_subsets,
    selected_family_ids_sha256,
    training_contract_metadata,
    training_partition_identity,
    verify_c6_provenance,
)
```

When PyTorch is installed, the package additionally exports
`EncodedMemory`, both encoders, `SharedGE1Decoder`, `GE1Model`, matched-model
constructors, the common loss, parity diagnostics, and strict checkpoint
helpers. The tuple-only C1-C3 API plus the pure C6 configuration, plateau,
derangement, prefix, aggregation, and provenance contracts, including
`frozen_encoder_config` and `frozen_feedforward_width`, remains importable
without PyTorch.

`load_train(corpus_dir)` loads the complete authoritative train assignment.
Its optional `family_ids` argument accepts only a sorted, unique, balanced
four-family or 32-family subset: respectively one or eight train families from
each of E, R, EE, and RE. `load_development(corpus_dir)` always loads the
complete 45-family development assignment. Neither function accepts a split
or partition argument.

The package deliberately does not re-export unrestricted model-data loaders.
There is no API for RR, ER, IID, history-depth, geometry-extrapolation, or
counterfactual payload access.

### Future protected access is additive, never a relaxation

`load_train` and `load_development` refuse every protected partition
unconditionally and carry no authorization argument, override, or split
parameter. A separately audited one-time RR systematic evaluation must add a
**new, explicitly named, separately audited entry point**. Relaxing,
parameterizing, or adding a
flag to the existing C1 loader is prohibited, so that protected access stays
visible in the call graph, is reviewable in isolation, and cannot be reached
from the train/development path by configuration alone.

An AST guard (`tests/test_import_guard.py`) enforces that `partitions.py` is
the only module importing an unrestricted model-data loader. It scans every
module recursively, excluding only `tests`, `__pycache__`, `adroit`, and
`generated`. It inspects static import statements, so it is an **accident
guard, not an airtight security boundary**: dynamic lookup such as
`import prototype.model_data` followed by attribute access would evade it. Its
purpose is to make an inadvertent widening of data access fail loudly in
review.

## C2 single-graph canonicalization

`canonicalize_graph(graph)` accepts exactly one frozen
`GraphCanonicalizationInput`. Its six fields are the integer-ID and
node-aligned tuple fields produced by the existing typed-graph adapter:

- `node_type_ids`, using `prototype.model_data.vocab.NODE_TYPES`;
- `edge_index` and `edge_type_ids`, using
  `prototype.model_data.vocab.EDGE_TYPES`;
- nine integer `categorical_attributes` per node;
- 39 finite numeric `geometry` channels per node; and
- 39 Boolean `geometry_mask` channels per node.

The input has no operation sequence, reconstruction target, family/template,
split, partition, node-ID string, position vector, padding, batch dimension,
or graph-offset field. Incoming local node indices address tuple rows and edge
endpoints only; they carry no semantic chronology.

The frozen `CanonicalizedGraph` result contains:

- `canonical_to_old` and `old_to_canonical` inverse index maps;
- canonical node types, categorical attributes, geometry, and geometry masks;
- relabeled canonical `edge_index` and `edge_type_ids`; and
- the position-free derived `operation_sequence`.

### Position-free recovery algorithm

The C2 controlled scope requires exactly one shared reference plane and one or
two operations. The canonicalizer:

1. validates all aligned tuple shapes, integer vocabulary IDs, endpoints,
   edge compatibility, and graph-local structural constraints;
2. recovers the unique linear operation order from dependent-to-reference
   `depends_on` relations;
3. recovers each operation's sketch and profile through `uses_profile` and
   `defined_in`;
4. additionally recovers a revolve axis through `uses_axis` and `defined_in`,
   requiring the profile and axis to use the same sketch;
5. verifies that every recovered sketch is `placed_on` the one shared plane;
6. rejects multiply assigned or unassigned nodes;
7. emits the plane followed by each operation group in semantic execution
   order, with group order `sketch, profile, optional axis, operation`; and
8. relabels edges and sorts them by canonical source index, integer
   `EDGE_TYPES` ID, and canonical destination index.

No operation chronology is resolved using incoming row order, a node-ID
spelling, a family label, or a runtime edge-name comparison. Edge sorting uses
the model-data integer vocabulary, not `GRAPH_EDGE_CLASS_ORDER`.

## C2 stable failure taxonomy

Every C2 rejection raises `GraphEncoderError` with one of these stable `code`
values:

- `malformed_or_misaligned_node_fields`
- `invalid_node_type_id`
- `invalid_edge_type_id`
- `malformed_edge_index`
- `out_of_range_edge_endpoint`
- `incompatible_typed_edge`
- `duplicate_edge`
- `self_edge`
- `unsupported_plane_count_for_controlled_scope`
- `unsupported_operation_count_for_controlled_scope`
- `operation_chain_branching`
- `operation_chain_cycle`
- `disconnected_operation_chain`
- `ambiguous_operation_chain`
- `missing_or_multiple_operation_profile`
- `invalid_profile_target_type`
- `missing_or_multiple_profile_defined_in`
- `invalid_profile_sketch_target_type`
- `missing_forbidden_or_multiple_operation_axis`
- `invalid_axis_target_type`
- `missing_or_multiple_axis_defined_in`
- `invalid_axis_sketch_target_type`
- `axis_profile_sketch_mismatch`
- `missing_or_multiple_sketch_placed_on`
- `placement_on_non_plane_node`
- `placement_on_wrong_shared_plane`
- `multiply_assigned_node`
- `unassigned_node`

`GraphEncoderError.detail` remains concise diagnostic context; callers should
branch on `code`, not exact detail text.

### Reachable versus defensive-only codes

Controlled-scope rejection runs before any relational recovery, so an
unsupported plane or operation count can never be masked as a downstream chain
or placement defect. A consequence is that three codes are **not reachable**
through `canonicalize_graph`, because every graph that could produce them is
rejected earlier by the scope gate:

- `operation_chain_branching` — requires at least three operations;
- `disconnected_operation_chain` — requires at least three operations;
- `placement_on_wrong_shared_plane` — requires at least two planes.

These are exported as `DEFENSIVE_ONLY_FAILURE_CODES`. They are retained as
guards inside the recovery helpers, remain covered by tests that call those
helpers directly, and would become reachable again if a later GE1 scope admits
more operations or planes. `REACHABLE_FAILURE_CODES` is the complement.

A first-failure histogram built from `FAILURE_CODES` must expect the three
defensive buckets to stay empty; building it from `REACHABLE_FAILURE_CODES`
avoids three permanently-zero categories.

## Procedural parity contract

C2 tests construct E, R, EE, ER, RE, and RR fixtures entirely in memory. For
each template they use `model_data.tests.fixtures.source`, build continuous
and quantized histories, obtain authoritative nodes, edges, and reconstruction
targets from `model_data.canonical`, create the physical-example metadata and
identities with existing helpers, then call `adapt_typed_graph`. Only the six
narrow graph-content fields are passed to C2; the adapter target and operation
sequence are discarded.

Expected canonical node order, edge order, and operation sequence are never
handwritten. Parity is checked against the authoritative procedural target for
the unpermuted graph and for consistently relabeled permutations of every
node-aligned field and edge endpoint. The tests cover all 24 E permutations,
all 120 R permutations, and 200 deterministic distributed lexicographic ranks
for each of EE, ER, RE, and RR, plus named structural adversaries.

The parity guarantee is intentionally limited to fully assigned, single-plane
controlled graphs. Two behaviors deliberately differ from the general
`prototype.model_data.canonical` helper:

- multiple reference planes are rejected instead of being ordered by node ID;
- leftover or unassigned nodes are rejected instead of being appended by node
  ID.

If a later GE1 scope admits multiple planes, their order must be represented
or recovered semantically. C2 must not gain a node-ID tie-break.

The procedural ER and RR fixtures do not access protected corpus families.
ER stays closed throughout core GE1, and RR stays closed until a separately
audited one-time RR systematic evaluation.

## C3 paired flat/graph batching

`build_paired_batch(examples)` accepts an iterable of already-authorized
`PhysicalExample` objects. C3 does not load or choose a partition. It rejects
an empty iterable, non-physical records, invalid or duplicate family IDs, then
sorts internally by `physical_family_id`. Forward, reversed, and arbitrary
input order therefore produce the same immutable `PairedBatch`.

For every sorted family, the builder:

1. calls the inherited `adapt_flat_mixed` and `adapt_typed_graph` adapters;
2. verifies family identity and complete per-example target equivalence using
   UTF-8 bytes from `canonical_record_json`;
3. constructs a narrow target-free `GraphCanonicalizationInput` and runs C2;
4. requires every C2 semantic field and derived operation sequence to match
   the authoritative `PhysicalExample.target` exactly;
5. constructs the canonical typed view using C2 semantics and the already
   verified authoritative target;
6. calls the inherited `collate_flat` and `collate_graph` implementations;
7. verifies identical collated family order and byte-equivalent complete
   `ReconstructionBatch` targets; and
8. returns one shared target separately from both encoder inputs.

No comparison uses `repr`, pickle, object identity, approximate geometry, or a
topology-only subset.

### Paired-batch records and layouts

```text
PairedBatch
  family_ids                 alignment/provenance only
  flat_input                 FlatEncoderInput
  graph_input                GraphSemanticInput
    node_content             GraphNodeContent
    edge_index
    edge_type_ids
  graph_bookkeeping          GraphBookkeeping
  target                     one shared ReconstructionBatch
```

`FlatEncoderInput` contains padded `categorical_ids`, `geometry`,
`geometry_mask`, and `padding_mask`. `GraphSemanticInput` contains concatenated
node content plus canonical typed edges. `GraphBookkeeping` contains only
`graph_offsets`, `edge_offsets`, `node_graph_ids`, and the dense `node_mask`.
Family IDs remain on the outer record and never enter node content.

`GraphNodeContent` is the future semantic node-feature boundary. Its
constructor accepts only node type IDs, nine categorical attributes, geometry,
and geometry masks. It cannot receive offsets, graph IDs, masks used for graph
membership, padding positions, or family metadata. C3 establishes this API
separation; it does not claim how the unimplemented C4 encoder will use
bookkeeping.

The builder validates flat padding against each true node count. It validates
that graph and edge offsets start at zero, remain monotonic, end at the exact
concatenated totals, and match every source family. Node graph IDs and dense
node masks must reproduce those intervals exactly. Every edge is checked
inside the node interval belonging to its edge-offset segment, so cross-family
edges are terminal errors.

Encoder inputs and bookkeeping have separate `to_torch()` methods. There is no
combined `PairedBatch.to_torch()` method that could merge a target into an
encoder dictionary. All methods use ordinary contiguous tensors compatible
with the inherited PyTorch 1.11 boundary.

### Explicit permutation diagnostics

`permute_graph(graph, permutation)` accepts one narrow
`GraphCanonicalizationInput` and an explicit bijection with the frozen
convention:

```text
permutation[new_index] = old_index
```

It reorders every node-aligned semantic tuple, constructs the inverse
old-to-new map, relabels both edge rows, and preserves edge-type alignment.
Boolean entries, duplicates, wrong lengths, negative indices, and out-of-range
indices are rejected. The function neither generates randomness nor accepts,
copies, or inspects a target or operation sequence. The primary paired builder
never calls it. Training-time permutation augmentation is not part of core
GE1 and would require a separate frozen protocol decision.

### Empty-edge boundary

The inherited graph collator and C3's low-level validation/separation boundary
correctly represent an empty edge set as `edge_index = ((), ())`, empty edge
types, and stable zero edge offsets. This is an engineering property of the
packing boundary, not a valid complete controlled CAD program. The production
paired path always runs C2 first, so removing the required semantic relations
remains terminal.

## C3 stable failure taxonomy

Every C3 production rejection uses one of these `GraphEncoderError.code`
values:

- `empty_paired_batch`
- `invalid_physical_example`
- `duplicate_family_id`
- `family_alignment_mismatch`
- `canonical_target_mismatch`
- `flat_graph_target_mismatch`
- `collated_target_mismatch`
- `invalid_permutation`
- `invalid_flat_padding`
- `invalid_graph_offsets`
- `invalid_edge_offsets`
- `invalid_node_graph_ids`
- `cross_graph_edge`
- `invalid_node_mask`

Every code has a focused negative test. Target and bookkeeping leakage are
prevented structurally by distinct frozen record types and are tested through
dataclass fields, signatures, and separate tensor dictionaries rather than by
adding unreachable runtime error codes.

## C4 common continuous encoder contract

Both encoder arms return the same frozen `EncodedMemory` record:

```text
prequant: [batch_size, 2, 16]
memory:   [batch_size, 2, 32]
```

`prequant` is exactly the continuous output of the inherited
`to_codebook` projection. It is exposed for diagnostics only. `memory` is
exactly `from_codebook(prequant)`, is contiguous, and is the only tensor
passed to the C5 shared decoder. Neither arm owns or calls an EMA
quantizer, performs nearest-code assignment, or mutates codebook state.

`FlatProgramEncoder` deep-copies only the inherited flat encoder's content
embeddings, chronological position embedding, Transformer, latent queries,
and `to_codebook`/`from_codebook` projections. It does not retain the inherited
decoder or VQ module. Its parity test compares against the corresponding
continuous inherited path with identical state and inputs.

`TypedGraphProgramEncoder.forward(...)` accepts only:

```text
node_type_ids
categorical_attributes
geometry
geometry_mask
edge_index
edge_type_ids
graph_offsets
```

The first six values are C3 graph semantics. `graph_offsets` is the minimum
separate bookkeeping needed to validate graph boundaries and slice graph-local
pooling. It is never embedded, projected, concatenated to node content, or
used as a learned feature. The encoder accepts no target, operation sequence,
family/template/split label, node graph ID, edge offset, padding position, or
local index feature, and it registers no position-embedding parameter.

### Relational layers and pooling

Each of the three layers retains the stored dependent-to-reference edge
orientation and constructs its reverse message direction internally. Five
semantic edge types in two directions produce ten channels. Every layer owns
two `[32, 32]` learned bases and one separate pair of mixing coefficients for
each channel. The resulting ten transforms therefore lie in the span of the
two bases.

For every receiver and directed channel, messages are summed with
`index_add_`, divided by that channel's receiver degree clamped to at least
one, and then summed across channels. There is no second division by the
number of active channels. Empty channels contribute exactly zero. A learned
self projection, GELU, dropout, residual connection, and LayerNorm keep
isolated nodes and completely empty edge sets finite.

After three layers, each graph is pooled independently by two learned
32-dimensional queries using four-head cross-attention. Each query attends
only to the node interval identified by that graph's offsets. Cross-graph
edges are rejected before message aggregation, and separate attention calls
prevent cross-graph pooling.

### Permutation properties and controlled diameter

The diagnostic node interface is permutation equivariant: after a consistent
node permutation and endpoint relabeling, applying the inverse permutation to
node outputs recovers the originals. The pooled memory is permutation
invariant. Tests freeze float32 tolerances at `atol=1e-6`, `rtol=1e-5` and
float64 tolerances at approximately `1e-12` for both values, always in
evaluation mode.

Procedural in-memory fixtures verify these maximum undirected diameters:

| Template | E | R | EE | ER | RE | RR | Maximum |
|---|---:|---:|---:|---:|---:|---:|---:|
| Diameter | 3 | 3 | 3 | 3 | 3 | 3 | 3 |

No corpus manifest or history payload is used for this verification. The
frozen three relational layers therefore cover the complete controlled
template diameter.

### C4 parameter counts and capacity gate

The flat feed-forward width is frozen at 192 and the graph pooling
feed-forward width at 64. Increasing only the flat width from the
inherited 64 is the preregistered arithmetic-only capacity adjustment; it was
chosen before observing encoder behavior or training results. The inherited
implementation instantiated at width 192 remains the flat parity reference;
original Flat V6 at width 64 is not, and numerical equality with it is neither
expected nor required.

#### Why the flat arm was widened

The preregistration allows exactly two capacity adjustments: relation-basis
count and encoder feed-forward width. At the inherited width, and at every
value of the basis knob, the graph arm cannot be brought within the 5% gate by
shrinking the treatment:

| Configuration | Flat encoder | Graph encoder | Graph excess over flat |
|---|---:|---:|---:|
| Inherited width 64, two bases | 14,480 | 23,468 | 62.07% |
| Inherited width 64, one basis | 14,480 | 20,366 | 40.65% |
| **Selected: flat width 192, two bases** | **22,800** | **23,468** | **2.93%** |

One relation basis is the smallest the decomposition admits, and it still
leaves the graph arm 40.65% larger. Widening the flat feed-forward layer was
therefore the only adjustment within the authorized knobs that could satisfy
the gate; the basis count stayed at two because reducing it neither closes the
gap nor serves any other purpose. All counts include the final
`nn.TransformerEncoder` `LayerNorm`, which is easy to omit when recomputing
these by hand.

This is recorded as arithmetic, not as an expected advantage for either arm.
A larger control has more parameters, but parameter count is not monotonically
related to capability, and architecture and optimization effects are not
additive. The direction of the adjustment must not be read as making a
treatment win harder or easier.

| Component | Trainable parameters |
|---|---:|
| Graph node initialization | 4,224 |
| Graph relational layer 1 | 3,188 |
| Graph relational layer 2 | 3,188 |
| Graph relational layer 3 | 3,188 |
| Relation bases, nested in the three layers | 6,144 |
| Relation/direction mixing, nested in the three layers | 60 |
| Graph-local pooling, including latent queries | 8,608 |
| Graph bottleneck projection | 1,072 |
| **Complete graph encoder** | **23,468** |
| **Complete flat encoder wrapper** | **22,800** |

The graph arm has 668 more parameters, or `2.9298245614%` relative to the flat
control. This passes the frozen 5% rule. The nested basis and mixing rows are
reported for auditability and must not be added again to the relational-layer
totals. Parameter matching does not match receptive fields: the flat
Transformer has global self-attention, while the graph arm has a three-hop
typed receptive field.

#### Frozen capacity contract

Both former capacity-adjustment fields are now frozen at their selected values
in `config.py`, so configuration and the encoder modules tell one story:
`relation_basis_count` is 2 in every arm, and `encoder_feedforward_width` is
192 for `flat` and 64 for `typed_graph`. `GE1Config.validate()` rejects any
other value with `unauthorized_configuration`.

Because the frozen width differs per arm, `GE1Config(encoder)` alone is **not**
a complete configuration — its default width is the graph value. Use
`frozen_encoder_config(encoder, seed)` (re-exported by `encoders.py` as
`default_encoder_config`) as the constructor. `CAPACITY_ADJUSTMENT_FIELDS` is
retained to name what could have been adjusted, not what still can be.

### Shared-component initialization across arms

The two arms build differently sized Transformers, and `FlatMixedVQModel`
constructs its bottleneck projections *after* its encoder. Left alone, the arms
would therefore draw different values for `to_codebook` and `from_codebook` at
the same seed, even though those projections are a shared latent-interface
component.

Both arms instead copy every genuinely shared component from one
arm-independent source, `shared_initialization_source(config)`, which is
constructed **first**, before any arm-specific module, at a fixed feed-forward
width. The random stream reaching the shared components is therefore identical
in both arms, and no constructor reseeds globally. The shared set is:

```text
field_embeddings, geometry_projection, geometry_mask_projection,
input_norm, latent_queries, to_codebook, from_codebook
```

The Transformer encoder of the flat arm and the relational core and pooling
stack of the graph arm are arm-specific and are not shared. Values are shared;
`Parameter` objects are not, so training one arm cannot affect the other. The
Transformer this shared source builds is discarded.

`FlatProgramEncoder.from_inherited(model)` is unaffected: an explicitly
supplied model remains authoritative for every component, which is what makes
exact parity against it testable.

C5 applies the same rule to the decoder. It constructs one canonical decoder
before either arm and deep-copies its exact state into two independent decoder
objects. Initial names, shapes, dtypes, parameter values, and buffer values are
identical; live `Parameter` and buffer objects are disjoint. Encoder
construction order therefore cannot change decoder initialization.

The exact module counts, parameter-object disjointness, direction and edge-type
sensitivity, batch isolation, continuous flat parity, quantizer non-invocation,
finite gradients, shared-component initialization equality, and float32/float64
permutation properties are covered by `tests/test_encoders.py`. After the C4
review fixes, authoritative Adroit CPU job `3344265` passed both new
shared-initialization tests, all 28 C4 encoder tests, and all 88 graph-encoder
tests with zero skips; see the [review-fix validation
record](../../docs/experiments/ge1_c4_review_fix_cpu_validation.md).

## C5 shared decoder

`SharedGE1Decoder` consumes only C4's contiguous continuous
`memory [B, 2, 32]`. Diagnostic `prequant [B, 2, 16]` never crosses the decoder
boundary. `GE1Model` selects and calls the configured encoder first, then calls
the same decoder `forward` implementation. The downstream call receives no
arm identity and contains no encoder-dependent branch.

The decoder inherits stable constrained V6 helpers for causal prefix decoding,
node grammar, five categorical heads, compact profiles, canonical profile and
reference-plane geometry, remaining normalized geometry, and revolve axes. It
inherits the initial Graph V1 main pair-MLP formula and pair ordering for typed
edges. The later C1 additive directed-position-bias modules are absent, and
the main pair logits are the authoritative six-class edge logits.

Decoder-component parity uses supplied common memory and bypasses both legacy
and GE1 encoders. Exact retained constrained V6 intermediates and exact Graph
V1 main pair logits are compared before constrained records and conversion.
This does not claim complete GE1 flat-model equality to original width-64,
discrete-VQ Flat V6; the current flat encoder is intentionally width 192 and
continuous.

### Autonomous result contract

`SharedGE1Decoder.forward(memory, *, node_counts, node_count_source)` is
target-free. It rolls out only predicted constrained prefix records and returns
an immutable `SharedDecoderPrediction` with:

- `raw_prediction`: direct prefix-decoder tensors and unmasked main pair logits;
- `constrained_prediction`: frozen grammar/geometry/edge constraints and
  validated canonical graph records; and
- `converted_prediction`: authoritative Graph V1 conversion results or
  explicit structured raised failures.

The three records are separately allocated and are not overwritten during
later processing. Continuous predictions publish no fictitious VQ codes.

### Common loss

`common_ge1_loss` is the only GE1 loss assembly for either arm. It preserves
Graph V1's component definitions, weights, masks, valid-element behavior, and
per-example normalization. Each component is normalized within each example,
then example totals are averaged across the batch. Graph V1 already used this
policy, so C5 introduces no intentional legacy aggregation difference. The
continuous VQ commitment component is exactly zero.

### Prospective positive operation magnitudes

Accepted
[ADR-0009](../../docs/decisions/ADR-0009-ge1-positive-operation-magnitude-repair.md)
adds one prospective, shared decoder mode after read-only Adroit diagnostic job
`3345013` isolated negative active operation magnitudes in all five C7-v2
failures. Historical C7-v2 behavior remains explicit as
`GE1-OPERATION-MAGNITUDE-TANH-LEGACY-v1`; the new mode is
`GE1-OPERATION-MAGNITUDE-POSITIVE-v1`.

Only compact geometry channels 4 and 5, serialized as extrusion distance 37
and revolve angle 38, change. The prospective neural decoder applies
`finfo(dtype).tiny + (1 - finfo(dtype).tiny) * sigmoid(raw)`, producing a
finite normalized value in `(0, 1]` for finite logits. A normalized value of
exactly `1.0` is required by the controlled 360-degree revolve target. Compact
channels 0--3 retain exact `tanh`, and all masks, channel selection,
normalization scales, direction, Boolean mode, loss definitions, and weights
are unchanged. Raw head outputs remain available through
`raw_remaining_geometry`.

The selected identity is recorded in model configuration, inference and
recovery checkpoint provenance, C6 metrics records, and downstream artifacts.
Legacy and repaired checkpoints cannot be loaded interchangeably. Both arms
receive value-identical, independently mutable copies of the same selected
decoder mode. This guarantees the analytic positive-magnitude domain only; it
does not claim CAD-kernel execution validity.

### Output-side position and bookkeeping

The decoder receives exactly four semantic serialization-position signals,
identically for both arms:

1. learned absolute causal output-step embeddings added before the Transformer
   decoder;
2. learned source absolute output-position embeddings concatenated to ordered
   pair features;
3. learned destination absolute output-position embeddings concatenated to
   ordered pair features; and
4. constructed signed relative source-minus-destination position concatenated
   to ordered pair features.

Local pair indices address output rows and select or derive only these frozen
signals. Output node masks control grammar activity, padding, and legal pairs.
`graph_offsets` remains restricted to graph-local encoder validation, slicing,
and pooling; edge offsets and graph IDs remain batching/alignment metadata.
None is passed as an additional decoder feature. No output position enters
either encoder, and the typed graph encoder still has no chronological or
absolute position embedding.

The complete field-level inventory and parity boundary are frozen in the
[C5 shared-decoder contract](../../docs/specifications/ge1_shared_decoder_contract.md).

Because the decoder is strongly position-aware while the typed graph encoder is
position-free, a primary number cannot be interpreted on its own. The frozen
Graph V1 position-only scoring prior reached 59/68 exact graphs, so a decoder
carrying four position signals may recover much of this controlled topology
regardless of what either encoder supplies. The memory-intervention values are
therefore a reporting requirement; see "Primary reporting requirement" below.

### Inherited V6 entry-point compatibility shim

`greedy_decode_v6_from_memory` validates its `V6EncodedMemory` argument before
decoding. That inherited validator requires `encoding_source` to equal the
frozen literal `v6_encoder_quantized_memory` and `code_indices` to be a
`[rows, latent_tokens]` long tensor inside the codebook range. Both exist to
prove the memory came from the V6 encoder's quantizer.

GE1 memory satisfies neither in fact. It is continuous and bypasses
nearest-code assignment, so no code indices exist. `SharedGE1Decoder` therefore
supplies two inert values, named rather than inlined, purely to pass an
inherited entry-point check whose premise does not hold for GE1:

| Supplied value | Why it is false for GE1 | Where it is corrected |
|---|---|---|
| `V6_ENTRY_POINT_COMPATIBILITY_SOURCE` | memory is continuous, not quantized | `encoded_memory_source` becomes `GE1_CONTINUOUS_MEMORY_SOURCE` |
| zero `code_indices` from `_inert_compatibility_code_indices` | no nearest-code assignment occurred | `latent_indices` becomes `()` |

`_corrected_memory_provenance` rewrites both fields immediately, before any
scoring, conversion, metric, or checkpoint observes the record. Nothing that
claims quantized-memory provenance escapes the decoder.

Two consequences are deliberate and must not be treated as regressions:

- The zero index tensor is byte-identical to what a fully collapsed
  single-code VQ produces. Any future codebook diagnostic must read the
  corrected record, never the shim.
- Because the corrected record no longer carries the V6 literal, the inherited
  `validate_and_convert_v6_autonomous_prediction` path **rejects GE1
  predictions by design**. GE1 converts through
  `prototype.graph_baseline.conversion.validate_and_convert_graph_prediction`.
  That rejection is a loud, intended failure.

The frozen inherited V6 implementation is not modified. `tests/test_c5_contract.py`
pins the shim to the inherited literal without importing PyTorch, and
`tests/test_c5_shared_decoder.py` asserts that returned records carry GE1
provenance and empty latent indices, then passes an actual corrected prediction
to the inherited V6 converter and requires the stable
`invalid_v6_node_selection` provenance rejection.

### Strict checkpoint API

`save_ge1_checkpoint` and `load_ge1_checkpoint` implement
`GE1-CHECKPOINT-v1`. The payload records encoder identity and version, frozen
feed-forward width and relation-basis count, shared-decoder and output-position
versions, operation-magnitude parameterization, continuous-bottleneck status,
full configuration plus SHA-256,
architectural sizes, parameter and buffer inventories, decoder state-key
namespace, source commit, authoritative operation-template manifest hash, and
complete model state.

Reload reconstructs the selected frozen arm and uses `strict=True`. Missing,
unexpected, mislabeled, or incompatible metadata and state keys are terminal;
there is no permissive fallback.

Initial authoritative Adroit job `3344278` ran all 22 focused C5 tests with
zero skips and zero errors: 19 passed and three test assertions failed. Audit
found legacy-VQ-provenance, conversion-field-name, and batch-row-order defects
in the assertions, not production decoder behavior. Corrected exact commit
`996016df44b7f9a6cd5c092a3e3b7a87d9964f9d` passed the complete real-tensor
Python 3.8/PyTorch 1.11 CPU gate as Adroit job `3344290`: all 22 focused C5
and 110 complete graph-encoder tests passed with zero skips, followed by every
regression and repository check. See the
[authoritative validation
record](../../docs/experiments/ge1_c5_cpu_validation.md).

## Frozen identities and training policy

| Role | Literal |
|---|---|
| Model family | `GE1-MODEL-v1` |
| Flat arm | `GE1-FLAT-CHRONOLOGICAL-CONTINUOUS-V1` |
| Typed-graph arm | `GE1-TYPED-GRAPH-POSITION-FREE-CONTINUOUS-V1` |
| Shared decoder | `GE1-SHARED-TYPED-EDGE-DECODER-V1` |
| Checkpoint schema | `GE1-CHECKPOINT-v1` |
| Protocol | `GE1-STAGE0-PREREG-v1` |
| Experiment | `GE1-SHARED-DECODER-ENCODER-COMPARISON` |

`GE1Config.encoder` accepts only `flat` or `typed_graph` and derives the arm
identity. Only the continuous bottleneck is accepted. Numeric architecture
defaults come from the inherited flat/Graph V1 configuration contracts or the
accepted GE1 plan. Relation-basis count and encoder feed-forward width are the
only capacity fields that may later be adjusted.

`GE1TrainingConfig` fixes the 50-epoch, batch-eight AdamW policy, epoch-50
checkpoint selection, plateau definition, train-ceiling tolerances, planned
three seeds, and the exact preauthorized two-seed timing fallback.

## C6 training and autonomous measurement

C6 adds `training.py`, `autonomous.py`, `metrics.py`, and the train-only
`c6_smoke.py` runner. One common training loop uses paired C3 batches, the
selected permitted input, the C5 model/loss, finite-value enforcement, global
norm clipping at 1.0, and atomic recovery checkpoints after every completed
epoch. For a fixed seed, family order, batch boundaries, optimizer steps,
example presentations, and checkpoint opportunities are identical by arm.
The experimental checkpoint remains fixed epoch 50; earlier checkpoints
cannot be selected by training loss, development behavior, or plateau.

`training_example_presentations` is cumulative actual physical examples in
successful optimizer batches. The plateau diagnostic compares two adjacent
five-epoch moving-best windows starting at epoch 10, with denominator epsilon
`1e-12` and a strict less-than-1% threshold. It records only and never changes
training.

`run_autonomous_evaluation` accepts target-free input batches, encodes once,
and uses one post-memory decoder helper for `P_true`, the complete-set
seed-pinned `P_shuffle` derangement, and deterministic batch-local `P_mean`.
Targets enter only in `score_condition`, after generation. Raw, constrained,
and explicit conversion outcomes remain separate.

The primary scorer uses C2 to recover generated graph order. Ambiguity scores
zero without row-order or target fallback. It strictly converts unchanged
complete-operation prefixes and requires analytic validity. Metrics aggregate
within physical family before an equal-weight family macro mean and use
structured nulls for undefined denominators. Exact graph/node, dependency,
attachment, conversion, failure-stage, geometry, parameter, receptive-field,
timing, and cumulative-process peak-memory records are included.

### Prefix validation envelope

Prefix construction retains the plane plus the first `k` complete canonical
operation groups and discards later groups with their exclusively owned
sketch, profile, and axis nodes. It never repairs, substitutes, or re-runs a
prediction. The envelope is exact:

| Field | Treatment |
|---|---|
| `node_type_id` | preserved per retained node |
| `categorical_ids` | preserved per retained node |
| `normalized_geometry` | preserved per retained node |
| `derived_geometry_mask` | preserved per retained node |
| directed typed edges | filtered to both-endpoints-retained; classes preserved |
| node `position` | renumbered to the prefix row |
| `legal_node_type_masks` | re-derived for the shorter requested length |
| `grammar_state_evidence` | re-derived for the shorter requested length |
| `RawDecodedEdge` presence scores | re-derived as inert `±1.0` markers |

Only converter bookkeeping is re-derived, and only because it is a function of
the requested sequence length: `legal_next_node_ids` receives the shortened
total count, and the strict converter validates that bookkeeping against the
length it is given. Recomputed masks can happen to equal the corresponding
full-rollout masks, as they do for every retained position in the tested RR
prefix; equality or difference is not the test of recomputation. The runtime
test instead requires exact masks and `grammar_state_evidence` recomputed under
the shortened count, and requires every mask to admit the emitted node. No
semantic prediction content changes, so scoring behavior is unaffected.
`tests/test_c6_runtime.py` pins this table field by field.

### Primary reporting requirement

Every primary result must publish the five memory-intervention values beside
it: `P_true`, `P_shuffle`, `P_mean`, `R_shuffle`, and `R_mean`. This is a
reporting requirement, not an optional diagnostic, because the shared decoder
receives four output-side position signals and the frozen position-only prior
reached 59/68 exact graphs; a primary number without memory evidence cannot
distinguish encoder contribution from decoder position exploitation.

`validate_primary_report` enforces the five keys and `complete_metrics_record`
calls it before returning, so a metrics artifact cannot be emitted without
them. Every value must be finite numeric evidence or a reason-bearing
structured null of the form `{"value": null, "reason": "..."}`; bare `None`
is rejected. This additive reporting obligation is accepted in
[ADR-0005](../../docs/decisions/ADR-0005-ge1-primary-reporting-and-metrics-v2.md),
not inserted into the frozen Stage 0 record.

New artifacts use `GE1-C6-METRICS-v2`, carry
`primary_reporting_contract=GE1-PRIMARY-REPORT-v1`, and are emitted by
`GE1-C6-TRAIN-ONLY-SMOKE-v2`. Historical v1 artifacts retain their original
identity.

### Donor/recipient template agreement

The frozen `P_shuffle` intervention is a deterministic cyclic derangement over
sorted family IDs, so donors sit at a fixed offset rather than being drawn
uniformly. If sorted IDs correlate with operation template, donors may
systematically share the recipient's template, which makes `P_shuffle` easier
and pushes `R_shuffle` upward. That direction is conservative for the frozen
ratio gate, but a borderline value cannot be interpreted without knowing the
rate.

`donor_template_agreement` therefore reports the assignment count, the
same-template donor count, observed agreement, the chance baseline, and the
excess over chance for `P_shuffle`. The baseline is conditioned on the frozen
no-self rule: for recipient template counts `n_t` and `N` assignments it is
`sum_t n_t(n_t-1) / (N(N-1))`. `P_true` uses recipient memory and `P_mean`
uses synthetic `batch_mean:<identity>` memory, so both emit structured
unavailable donor-agreement records rather than pretending those sources are
families. This diagnostic does not change the derangement, ratios, or gates.

The full additive contract is
[GE1 C6 measurement](../../docs/specifications/ge1_c6_measurement_contract.md).
The checked-in C6 implementation passed its two-epoch 407-family
`operation_template.train` runtime smoke under Python 3.8 and PyTorch 1.11 on
Adroit as job `3344367`. That smoke opened no development or protected
partition and is not a scientific result.

Initial Adroit job `3344337` at exact commit `ca6dd069ae700cff993c354c2116de11924930f8`
ran all 37 focused C6 tests with zero skips: 33 passed and four errored. Three
errors were the same C6 boundary defect—`node_counts` was a tuple instead of
the C5-required device-local long tensor—and the fourth was a missing
`GraphEncoderError` test import. The focused correction changes only those
points. Fail-fast execution preceded the complete suite and both corpus
smokes, so the manifest and every payload remained unopened. See the
[failed-attempt record](../../docs/experiments/ge1_c6_cpu_validation_attempt_3344337.md).

Corrected exact-commit job `3344363` confirmed both fixes and passed 36/37
focused tests with zero skips. Its sole error was in the resume test itself:
generic dataclass equality cannot reduce a multi-element PyTorch tensor to one
Boolean. The pending test-only correction compares tensor-bearing predictions
recursively and compares complete metric records after removing timing fields.
The job again stopped before the complete suite and corpus smoke, so no
manifest or payload was opened. See the [second failed-attempt
record](../../docs/experiments/ge1_c6_cpu_validation_attempt_3344363.md).

Final exact-commit job `3344367` passed all 37 focused C6 tests and all 147
graph-encoder tests with zero skips, every regression and repository gate, and
both complete 407-family train-only arm smokes with strict epoch-2 reload and
complete three-condition metrics records. C6 is runtime-complete. See the
[authoritative validation
record](../../docs/experiments/ge1_c6_cpu_validation.md).

The C5/C6 review fixes documented above were implemented after job `3344367`.
They leave that historical result intact. Clean exact-commit Adroit CPU job
`3344431` passed all 75 focused C5/C6 tests and all 163 graph-encoder tests
with zero skips, every regression and repository gate, and both v2 407-family
train-only arm smokes. Current C5/C6 source is therefore authoritatively
covered; see the [combined validation
record](../../docs/experiments/ge1_c5_c6_review_fix_cpu_validation.md). C7
implementation was added afterward under accepted ADR-0006. Its formal pilot
completed separately as job `3344505`; see the
[C7 record](../../docs/experiments/ge1_c7_pilot.md).

## C7 train-only sufficiency and memory-use gates

`select_c7_sufficiency_subsets(corpus_dir)` verifies the existing manifest
authority and ranks only train-assignment metadata using the frozen
`GE1-C7-SUFFICIENCY-v1` hash material. It returns lexicographically sorted,
nested four- and 32-family selections with exactly one or eight families from
each of `E`, `R`, `EE`, and `RE`. The selector never opens a CAD-history
payload.

`pilot.py` builds a fresh matched seed-2026 arm pair separately for each
subset, runs the frozen 50-epoch recipe, strictly reloads epoch 50, and scores
only fully autonomous outputs. Tiny exact failure skips that arm's scaled
work; scaled exact failure still permits the governed memory interventions to
run. Completed scientific failures finalize normally, while environment,
provenance, access, malformed-metric, strict-reload, and integrity failures
remain nonzero infrastructure failures.

Scaled memory use requires finite unrounded `P_true > 0`,
`P_shuffle / P_true <= 0.80`, and `P_mean / P_true <= 0.80`. Overall passage
requires both exact gates and the memory gate for both arms. Only that outcome
sets `stage6_authorized_by_c7=true`; exact failure can record the already
preauthorized repair trigger, but C7 neither implements nor invokes repair.

The C7 CLI has only the corpus directory, new external output directory,
repository root, and expected commit as arguments. It stages a self-verifying
artifact beside the requested final path and atomically publishes
`resolved_config.json`, `metrics.jsonl`, complete checkpoints, an artifact
manifest, and `SHA256SUMS`. The CPU Slurm runner validates the exact clean
source and complete test/regression suite before any corpus access. The
formal runner completed as job `3344505`. Both tiny exact gates failed, all
scaled gates remained `not_run`, and Stage 6 remains unauthorized. The
artifact passed integrity checks, with the documented limitation that its
structured runtime and checkpoint provenance omitted the Slurm job ID even
though the scheduler log and artifact path identify job `3344505`.

## Post-C7 optimization-sufficiency diagnostic

`optimization_diagnostic.py` implements the accepted
`GE1-C7-OPTIMIZATION-SUFFICIENCY-DIAGNOSTIC-v1` contract. It reuses the C7
metadata selector and rejects any tiny cohort other than the exact four formal
C7 families and hash. Only those four `operation_template.train` payloads can
then be loaded.

The diagnostic constructs one fresh matched seed-2026 pair, trains each arm
for all 500 optimizer updates and 2,000 family presentations, retains updates
25/50/.../500, and measures strictly reloaded updates 50, 100, 200, and 500.
Autonomous `P_true` exactness alone determines sufficiency. Every milestone
also carries all three memory conditions, the five primary reporting values,
donor agreement, memory-alteration evidence, and the complete per-family C6
metric envelope.

The additive `run_ge1_training` arguments used here are opt-in. Existing C6
and formal C7 callers omit them and retain their every-epoch checkpoint and
50-epoch validation behavior. Diagnostic checkpoints are recovery/trajectory
artifacts only and cannot resume formal C7 or warm-start the full comparison.

The runner explicitly passes decimal `SLURM_JOB_ID` through Apptainer's clean
environment and verifies that the resolved configuration, run events, every
checkpoint payload, and terminal event agree. It publishes the
`GE1-C7-OPTIMIZATION-ARTIFACT-v1` bundle atomically only after complete
integrity validation. A completed negative trajectory exits successfully;
infrastructure failure leaves an incomplete staging directory and exits
nonzero.

The narrow CLI is:

```bash
python3 -m prototype.graph_encoder.optimization_diagnostic \
  --corpus-dir <authoritative-corpus> \
  --output-dir <new-external-output> \
  --repository-root <clean-exact-checkout> \
  --expected-commit <exact-commit>
```

The first authoritative-environment attempt, Slurm job `3344896` at commit
`0837717a90eebcfd6aaa0c0ca9cd47f18ac52f88`, stopped in the focused preflight
suite before manifest or payload access. An explicit `None` supplied to the
Slurm-ID validator was incorrectly treated as an omitted argument and fell
back to the ambient job ID. The implementation now uses a private sentinel:
omitting the argument retains environment-based provenance, while explicitly
supplying `None` is rejected. After a separate linked-worktree preflight
failure, standalone-checkout job `3344907` completed successfully. Both arms
first passed the exact gate at update 200 and remained exact at update 500.
Stage 6 remains unauthorized, repair was not invoked, C8 has not begun, and no
protected partition was opened. The audited
[result record](../../docs/experiments/ge1_optimization_diagnostic.md) and
frozen contract and runner requirements are in the
[diagnostic specification](../../docs/specifications/ge1_c7_optimization_sufficiency_diagnostic.md).

## Prospective C7-v2 protocol

`c7_v2.py` implements the separately identified `GE1-C7-SUFFICIENCY-v2`,
`GE1-C7-METRICS-v2`, and `GE1-C7-ARTIFACT-v2` protocol without mutating
`pilot.py` or `optimization_diagnostic.py`. It reuses the exact immutable
four- and 32-family C7-v1 cohorts, but constructs fresh matched seed-2026
models and trains to a fixed epoch 200. The shared training loop exposes this
through an opt-in `fixed_protocol_final_epoch`; existing callers omit it and
retain their original behavior.

Each arm retains only its selected epoch-200 checkpoint. The runner writes
that checkpoint before autonomous evaluation, strictly reloads it into a
fresh model, and records its commit, Slurm identity, and SHA-256. Neither an
earlier checkpoint, C7-v1 checkpoint, nor optimization-diagnostic checkpoint
can satisfy or initialize C7-v2. There is no early stopping,
best-checkpoint selection, outcome-dependent extension, resume, or warm
start.

The scaled payload loader is reachable only after both tiny arm gates pass.
If either tiny arm fails, every scaled gate is recorded `not_run` and scaled
payload access remains false. Exact failure makes the already authorized
decoder-repair path the next permitted work; memory-only failure is
inconclusive. Only six passing arm gates may set
`stage6_authorized_by_c7_v2=true`. Scientific failure still finalizes and
exits zero, whereas infrastructure or artifact failure exits nonzero.

`adroit/ge1_c7_v2_cpu.slurm` is the standalone-checkout CPU runner. It runs
the focused C7-v2 tests with zero PyTorch skips, the full graph-encoder suite
with zero skips, all regressions and repository checks, and only then permits
metadata or train-payload access. It independently checks every selected
checkpoint and artifact hash after execution.

The first authoritative preflight, Slurm job `3344975` at commit
`99a4587d73e82a5df5f12a54130b105ae264b419`, passed the focused 28/28,
complete graph-encoder 266/266, and model-data 86/86 gates with zero skips. It
then stopped before flat-baseline discovery because Python 3.8's singleton
`unittest.defaultTestLoader` retained the first regression discovery root.
The corrected loop constructs a fresh `unittest.TestLoader` and supplies
`top_level_dir="."` for every suite. Job `3344975` opened no manifest or
payload, performed no training, and produced no artifact; `/external/c7-v2`
stdout lines were emitted by a mocked CLI unit test. The complete
[preflight record](../../docs/experiments/ge1_c7_v2_preflight_attempt_3344975.md)
is non-scientific.

Corrected job `3344981` at exact commit
`e325d5ad97957c08da4a19b4261560e8a4a472a4` completed C7-v2. Both tiny gates
and both scaled memory gates passed. Each arm was exact on all scaled node
sequences, typed graphs, strict conversions, and applicable `depends_on`
relationships, but flat failed analytic complete validity on two families and
typed graph failed it on three. Every failure was
`invalid_operation_parameter`. The finalized result is a valid C7-v2 failure,
does not authorize Stage 6, and makes the deferred repair path procedurally
next without implementing it.

## C7-v2 operation-parameter diagnostic

`operation_parameter_diagnostic.py` implements the separately versioned
read-only follow-up authorized before repair. It verifies job `3344981`'s
artifact and exact scaled checkpoint hashes, reconstructs only the frozen
32-family train cohort, strictly loads each checkpoint into a fresh model,
and reproduces stable autonomous `P_true` metrics before extracting any
diagnostic trace. Both arms are evaluated for all five failed families.

The record separates strict conversion from analytic controlled-domain
validity. It reports the active operation channel's pre-`tanh` head output,
normalized and physical predictions and targets, errors, masks, categories,
profile scale, prefix, and analytic failures. No CAD kernel is used, so
executor results are structurally unavailable and a legal-but-geometrically-
incompatible parameter is unassessable. Raw CAD histories, graphs, model
state, optimizer state, and checkpoints are forbidden from the five-file
diagnostic artifact.

`adroit/ge1_operation_parameter_diagnostic_cpu.slurm` ran authoritatively as
job `3345013` at exact commit
`802ae1d1e9deb3c7a6c428d320e4276a8b5e7e57`. It ran all preflight suites
before data access, bound the source C7-v2 artifact and corpus read-only,
verified source hashes before and after inference, emitted explicit access
declarations, and created no checkpoints. The full contract is in the
[diagnostic specification](../../docs/specifications/ge1_c7_v2_operation_parameter_diagnostic.md).
The completed result identified negative active operation magnitudes in all
five failed arms without authorizing repair training, Stage 6, C8, or
protected-partition access. The prospective repair implementation above is a
separately versioned engineering-validation step.

## Manifest authority and access order

Before calling any physical-example payload loader, C1 reads only:

```text
manifests/operation_template.json
```

This hash check is also what binds the C0 audit, which ran against a local copy
because the remote connection failed, to the authoritative Adroit file. Until
the first successful Stage 1 read, the recorded counts and assignment hashes
are properties of the local copy alone.

It requires the exact frozen file SHA-256
`a9ac86a6dede054fbbba57e0906b210bab26036c3f5c150b332038f78d2dadb7`,
the 407/45/114/114 family counts, the 814/90/228/228 sample counts, permitted
templates in every partition, and all four assignment hashes. Missing,
unreadable, modified, or inconsistent authority fails before the inherited
payload loader is invoked. Verification cannot be disabled when the Adroit
path is unavailable.

After verification, C1 delegates only the authorized partition and optional
train subset to `prototype.model_data.loader.load_partition_physical_examples`.
Malformed or inconsistent corpus data continues to raise the original
`ModelDataError`; it is not hidden inside a GE1 error.

## C1 boundary error taxonomy

The non-C2 package boundary uses:

- `invalid_configuration` for malformed values;
- `identity_mismatch` for altered frozen schema identities;
- `unauthorized_configuration` for scientifically unauthorized settings;
- `invalid_train_subset` for unsafe sufficiency selections;
- `manifest_authority_failure` for absent or inconsistent authority; and
- `protected_partition_access` for any non-train/development request.

## Explicit limitations

- C6 supplies the common training/recovery and train-only autonomous
  measurement machinery and passed its authoritative runtime smoke. It does
  not supply development, systematic, or test access.
- ER and all IID, history-depth, and geometry-extrapolation partitions remain
  closed throughout core GE1.
- C3 passed its authoritative Python 3.8.13/PyTorch 1.11.0 CPU compatibility
  check on Adroit as job `3344235`; see the
  [validation record](../../docs/experiments/ge1_c3_cpu_validation.md).
- C4 and its capacity-freezing/shared-initialization review fixes passed
  authoritative real-PyTorch CPU validation. Every later implementation
  addition still requires its own production-environment validation; C4's
  result does not validate code that does not yet exist.
- C5 passed authoritative Python 3.8.13/PyTorch 1.11.0 CPU runtime validation
  as Adroit job `3344290` at exact commit
  `996016df44b7f9a6cd5c092a3e3b7a87d9964f9d`.
- C6 passed authoritative runtime validation as Adroit job `3344367`; the
  post-validation C5/C6 review-fix source and v2 artifacts passed combined
  exact-commit revalidation as Adroit job `3344431`. Formal C7 job `3344505`
  completed as a valid scientific failure of both tiny exact gates. C8 and
  every later GE1 chunk have not begun.
