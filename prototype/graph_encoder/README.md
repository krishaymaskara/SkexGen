# GE1 Graph Encoder Package

## Implemented scope

This package implements the C1 boundary and the C2 position-free graph
canonicalizer for `GE1-SHARED-DECODER-ENCODER-COMPARISON`.

C1 provides:

- immutable, deterministic model and training configuration;
- exact frozen identities and encoder-to-arm derivation;
- operation-template-only manifest authority verification;
- train and complete-development payload access wrappers;
- unconditional protection of RR, ER, and every non-operation manifest; and
- stable GE1 boundary errors.

C2 provides structural validation and deterministic canonicalization of one
controlled typed graph at a time. It does not implement batching, a neural
encoder, a shared decoder, losses, checkpointing, training, evaluation, or a
scientific result. The frozen `prototype.flat_baseline` and
`prototype.graph_baseline` packages are reused by import only and remain
unchanged.

## Public API

```python
from prototype.graph_encoder import (
    CanonicalizedGraph,
    GE1Config,
    GE1TrainingConfig,
    GraphCanonicalizationInput,
    GraphEncoderError,
    canonicalize_graph,
    load_development,
    load_train,
)
```

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
parameter. The Stage 7 one-time RR evaluation must add a **new, explicitly
named, separately audited entry point**. Relaxing, parameterizing, or adding a
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

- C2 canonicalizes one graph only; C3 batching, padding, and offset work has
  not begun.
- No relational or flat neural encoder, model, or decoder class exists in this
  package yet.
- No loss, training loop, checkpoint, evaluation, or systematic-access
  capability exists.
- ER and all IID, history-depth, and geometry-extrapolation partitions remain
  closed throughout core GE1.
- Actual Python 3.8 execution on Adroit remains the authoritative compatibility
  check; local grammar parsing and compilation are preliminary checks only.
