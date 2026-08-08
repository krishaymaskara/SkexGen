# GE1 Graph Encoder Package

## Implemented scope

This package implements the C1 boundary, C2 position-free graph canonicalizer,
C3 paired flat/graph batching, C4 continuous flat and position-free typed
graph encoders, C5 shared decoder integration, and C6 training/measurement
machinery for
`GE1-SHARED-DECODER-ENCODER-COMPARISON`.

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
family-macro aggregation, and resource measurements. It does not implement
C7 sufficiency gates, C8 decoder repair, protected evaluation access, or a
scientific result. The frozen
`prototype.flat_baseline` and
`prototype.graph_baseline` packages are reused by import only and remain
unchanged.

## Public API

```python
from prototype.graph_encoder import (
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
    PairedBatch,
    SharedGE1Decoder,
    authorize_c6_provenance,
    autonomous_input_from_paired,
    build_paired_batch,
    build_ge1_model,
    build_matched_ge1_models,
    canonicalize_graph,
    capacity_difference_percent,
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

### Strict checkpoint API

`save_ge1_checkpoint` and `load_ge1_checkpoint` implement
`GE1-CHECKPOINT-v1`. The payload records encoder identity and version, frozen
feed-forward width and relation-basis count, shared-decoder and output-position
versions, continuous-bottleneck status, full configuration plus SHA-256,
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

The full additive contract is
[GE1 C6 measurement](../../docs/specifications/ge1_c6_measurement_contract.md).
The checked-in C6 implementation is not accepted as runtime-complete until its
two-epoch 407-family `operation_template.train` smoke passes under Python 3.8
and PyTorch 1.11 on Adroit. That smoke opens no development or protected
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
  measurement machinery. It does not supply development, systematic, or test
  access, and its authoritative runtime smoke remains pending.
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
- C6 is implemented locally and awaits authoritative runtime validation. C7,
  C8, and every later GE1 chunk have not begun.
