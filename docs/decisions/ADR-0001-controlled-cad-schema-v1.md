# ADR-0001: Controlled CAD schema version 1

- Status: `accepted`
- Decision date: `2026-07-22`
- Documentation date: `2026-07-28`
- Owners/reviewers: project implementation team
- Supersedes: none
- Superseded by: none

## Context

The research plan proposed representing CAD histories as typed dependency
graphs, but its diagrams and the candidate experimental specification did not
yet define one consistent edge vocabulary or direction. In particular:

- the research-plan examples point from a sketch or axis toward the operation
  that consumes it;
- the candidate specification proposes consumer-to-resource direction but
  names containment and chronology edges `CREATED_BY` and `PRECEDES`;
- the implemented representation uses consumer-to-resource direction with
  `DEFINED_IN` and `DEPENDS_ON`.

A deterministic schema was required before controlled-data generation,
counterfactual pairing, model-data adaptation, and evaluation could share
identical structural meaning.

This ADR records the contract implemented by schema version 1. It freezes the
current controlled benchmark representation; it does not require every future
CAD schema or neural graph architecture to retain the same restrictions.

Related documents:

- [Research plan](../research_plan.md)
- [Experimental specification](../specifications/experimental_spec.md)
- [Current project status](../status.md)
- [Representation package contract](../../prototype/representation/README.md)

## Decision

### Version and modeling unit

The serialized representation declares `schema_version: 1`. One record
describes one ordered, single-body CAD feature history that contains at least
one solid-producing operation.

The history consists of:

```text
CADHistory
├── StructureGraph
│   ├── typed nodes
│   ├── typed directed edges
│   └── operation_sequence
└── GeometryStore
    ├── node geometry
    └── sketch-element geometry
```

`operation_sequence` is the authoritative execution-order record. Every
extrude or revolve node appears in it exactly once.

### Node vocabulary

Schema version 1 contains these graph-node types:

| Node type | Meaning |
|---|---|
| `REFERENCE_PLANE` | A local orthonormal frame supporting sketches |
| `SKETCH` | A collection of primitives and loop topology |
| `PROFILE` | A selected outer loop and optional ordered inner loops |
| `AXIS` | A construction axis defined in a sketch |
| `EXTRUDE` | A one-sided extrusion consuming one profile |
| `REVOLVE` | A revolution consuming one profile and one axis |

Lines, arcs, circles, and loops are typed subrecords inside a `SKETCH`; they
are not graph nodes in version 1. Stable primitive and loop IDs permit
element-local geometry and counterfactual-preservation addresses.

### Edge convention and vocabulary

All dependency and reference edges use the
**consumer-or-contained-entity to resource** convention. An edge source is
the entity whose interpretation requires the target.

| Edge type | Allowed source | Allowed target | Required cardinality |
|---|---|---|---:|
| `PLACED_ON` | `SKETCH` | `REFERENCE_PLANE` | Exactly one per sketch |
| `DEFINED_IN` | `PROFILE`, `AXIS` | `SKETCH` | Exactly one per profile or axis |
| `USES_PROFILE` | `EXTRUDE`, `REVOLVE` | `PROFILE` | Exactly one per operation |
| `USES_AXIS` | `REVOLVE` | `AXIS` | Exactly one per revolve; forbidden on extrude |
| `DEPENDS_ON` | `EXTRUDE`, `REVOLVE` | Earlier `EXTRUDE` or `REVOLVE` | None for the first operation; exactly the immediately preceding operation thereafter |

For example:

```text
SKETCH_1  --PLACED_ON----> REFERENCE_PLANE_1
PROFILE_1 --DEFINED_IN---> SKETCH_1
AXIS_1    --DEFINED_IN---> SKETCH_1
REVOLVE_1 --USES_PROFILE-> PROFILE_1
REVOLVE_1 --USES_AXIS----> AXIS_1
REVOLVE_2 --DEPENDS_ON---> REVOLVE_1
```

`DEPENDS_ON` is a semantic body dependency, not a forward chronology edge. It
therefore points backward from the consuming operation to the earlier
operation whose body state it consumes. In version 1 the current-body chain is
linear, so each later operation depends on the immediately preceding entry in
`operation_sequence`.

The schema names are `DEFINED_IN` and `DEPENDS_ON`; `CREATED_BY` and
`PRECEDES` are not version 1 edge types.

### Boolean and body semantics

- The first operation must use `NEW_BODY`.
- Every later operation must use `JOIN` or `CUT`.
- The final history represents one body.
- Multiple independent body identities and cross-body references are outside
  version 1.

The representation validator establishes structural and geometric validity.
The controlled-data generator adds a separate analytical
Boolean-feasibility policy, and OpenCascade validation remains an independent
execution check.

### Structure and geometry

Categorical structure and numerical geometry are stored separately while
remaining jointly constrained.

Structural information includes:

- node and primitive types;
- loop membership and ordered primitive references;
- profile loop references;
- Boolean mode and operation direction;
- typed dependency edges;
- operation order.

Geometric information includes:

- reference-plane origin and axes;
- line, arc, and circle parameters;
- axis point and direction;
- extrusion distance;
- revolve angle.

Geometry may be stored continuously or with affine quantization. Every
geometric rule is checked on decoded physical values. The two encodings are
representation variants of the same physical history, not separate physical
examples.

### Version 1 geometric conventions

- Numeric values must be finite.
- Reference-plane axes must be orthonormal within the schema tolerance.
- Axis directions must be unit vectors within the schema tolerance.
- Extrusion distance is a positive magnitude; direction is categorical.
- Extrusions are one-sided.
- Revolve angles are measured in degrees and lie in `(0, 360]`.
- A revolve's profile and axis must be defined in the same sketch.
- Loops support line/arc chains or one circle; mixed circle chains and
  ambiguous unsupported cases are rejected.

### Controlled-generator restrictions

The controlled generator deliberately uses a narrower subset of the general
version 1 representation:

- operation templates `E`, `R`, `EE`, `ER`, `RE`, and `RR`;
- one shared reference plane per generated history;
- a fresh sketch and profile for each operation;
- a same-sketch axis for every revolve;
- rectangle, circle, or capsule outer profiles;
- at most two operations;
- the first operation as `NEW_BODY` and the optional second as `JOIN` or
  `CUT`.

These are benchmark-generation choices, not universal CAD rules and not
requirements that every future version 1 producer must adopt unless enforced
by the representation validator itself.

### Determinism and identity

Canonical JSON sorts unordered structural collections by stable IDs while
preserving semantically ordered fields such as operation order, loop
traversal, inner-loop order, and primitive traversal. Unknown fields,
duplicate object keys, invalid enum values, and nonfinite JSON constants are
rejected.

Physical-family identity is computed from canonical structure and decoded
physical geometry. It excludes representation encoding, generation seed,
selection order, split, and file path. A separate sample identity includes
the exact continuous or quantized representation variant.

## Alternatives considered

### Resource-to-consumer edge direction

Edges such as `SKETCH → EXTRUDE` can resemble a forward data-flow diagram.
They were not selected because references are more naturally queried as
outgoing requirements of the consuming operation, and containment references
already use the contained entity as the source.

### `CREATED_BY` and `PRECEDES`

`CREATED_BY` suggests a feature-creation event that the version 1 prototype
does not explicitly execute. `DEFINED_IN` records reference scope without
that implication. `PRECEDES` duplicates forward chronology already preserved
by `operation_sequence`; `DEPENDS_ON` instead records the body state consumed
by a later operation.

### Sketch primitives and loops as graph nodes

First-class primitive nodes would enable finer graph message passing but
would substantially increase graph size and change node-level metrics. The
version 1 controlled experiment keeps them as stable, addressable sketch
subrecords. A future graph-model ADR may revisit this without changing the
meaning of existing schema version 1 files.

### Reference-plane fields embedded directly in sketches

Embedding the frame on each sketch would reduce node count. A
`REFERENCE_PLANE` node was selected so placement is an explicit typed
dependency and a plane can be shared deterministically.

### Forward chronology edges as execution authority

A `PRECEDES` chain could encode ordering, but ordered execution is simpler and
unambiguous as `operation_sequence`. `DEPENDS_ON` remains semantically
distinct from chronology.

## Consequences

### Positive

- Every package uses one edge direction and vocabulary.
- Reference cardinalities and body dependencies are mechanically validated.
- Continuous and quantized records can collapse safely to one physical
  learning example.
- Flat and graph adapters share identical reconstruction targets.
- Counterfactual edits can identify intended and preserved fields with stable
  structural addresses.
- Operation order and dependency meaning cannot be silently conflated.

### Tradeoffs and limitations

- The schema is not a general feature-history or multi-body CAD schema.
- Same-sketch revolve references exclude valid real-world constructions.
- Linear body dependency cannot express branching or merging histories.
- Sketch primitives are not available as graph-message-passing nodes without
  a later representation or adapter decision.
- Reference-plane nodes and explicit reference edges increase graph size.
- Schema validity does not imply OpenCascade executability or meaningful
  Boolean effect.

### Documentation impact

The research-plan and experimental-specification diagrams must be reconciled
to this vocabulary when they describe implemented version 1 behavior.
Proposed future alternatives may remain, but they must be labeled as such and
must not be presented as the current schema.

## Validation and evidence

The initial typed representation was committed as:

```text
1a13b7362e6e8aff4c2e48c58acc889f0a3535d1
Add typed CAD representation prototype
```

The deterministic controlled generator was committed as:

```text
7f216c0c214282084760fb340f468c0ee526deec
Add controlled CAD history generator
```

The checked-in validator enforces node/edge type compatibility, exact
reference cardinalities, operation-sequence completeness, backward
dependency order, the linear single-body chain, Boolean-position rules,
geometric bounds, and same-sketch revolve compatibility. The representation
package has 40 local standard-library tests. Independent controlled-data,
model-data, counterfactual, and kernel-validation packages build on the same
schema rather than redefining it.

Completed OpenCascade audits support the controlled generator and execution
pipeline, but kernel results are experiment evidence rather than part of the
schema definition.

## Implementation contract

Primary implementation locations:

- `prototype/representation/model.py` — enums and immutable records;
- `prototype/representation/validation.py` — structural and geometric
  invariants;
- `prototype/representation/serialization.py` — strict canonical JSON;
- `prototype/controlled_data/builders.py` — controlled v1 graph construction;
- `prototype/controlled_data/identity.py` — physical and representation
  identities;
- `prototype/model_data` — deterministic flat and graph adapters.

Any incompatible vocabulary, direction, cardinality, or serialization change
requires a new schema version and a migration decision. Adding a neural graph
adapter that preserves the version 1 meaning does not by itself require a new
schema version.

## Review conditions

Reconsider or supersede this ADR when the project needs:

- histories longer than the linear current-body contract can express;
- branching, merging, or multiple bodies;
- references to generated faces or evolving B-rep entities;
- cross-sketch revolve axes or profiles;
- primitive- or loop-level graph nodes;
- constraints, fillets, chamfers, sweeps, lofts, or assemblies;
- a different edge convention for the principal graph-model comparison;
- evidence that the current schema prevents a fair or meaningful experiment.

A later graph-model design may add model-side relations or tokenization, but
it must state whether those are derived views of schema version 1 or require a
superseding representation version.
