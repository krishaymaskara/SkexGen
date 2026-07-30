# Experimental Specification: Structured Extrude-and-Revolve CAD

## 0. Status and interpretation rules

This document translates [the research plan](../research_plan.md) into an
implementation and
evaluation contract. Some sections remain proposed future protocol, while the
controlled schema and completed package contracts are now implemented.
[ADR-0001](../decisions/ADR-0001-controlled-cad-schema-v1.md) is
authoritative for schema version 1 whenever historical proposal language in
this document differs from the checked-in representation.
[ADR-0002](../decisions/ADR-0002-three-week-flat-versus-graph-scope.md) is
authoritative for the active model comparison, shared decoder condition,
minimum evaluation set, and expansion rule. The broader model matrix and
phase language retained below record the earlier candidate protocol; any
fully discrete geometry path, complete discrete-versus-continuous comparison,
additional graph architecture, broad split suite, or extensive multi-seed
requirement is deferred unless the minimal ADR-0002 comparison is complete.

Every normative statement has one of these labels:

- **[ACTIVE PLAN REQUIREMENT]** — required by the approved July 30
  three-week scope.
- **[ACTIVE IMPLEMENTATION BOUNDARY]** — the current distinction between
  checked-in deterministic capability and unimplemented neural work.
- **[PLAN REQUIREMENT]** — stated directly in the research plan.
- **[PROPOSED V0]** — a concrete initial choice needed to make the experiment implementable; it is not yet established.
- **[IMPLEMENTED V1]** — frozen for controlled schema version 1 and enforced
  by checked-in code; it may remain a revisitable choice for a later schema.
- **[MENTOR DECISION]** — a choice that materially affects the scientific comparison and should be confirmed before the dataset or model interface is frozen.
- **[LATER IMPLEMENTATION DECISION]** — an engineering or reporting choice that must eventually be frozen, but does not need mentor approval in the next discussion.
- **[HISTORICAL ...]** — retained from the broader July 16 protocol and
  deferred wherever it conflicts with ADR-0002.

If an unlabeled explanation conflicts with a labeled statement, the labeled statement controls. Parameter ranges, token IDs, dataset sizes, split ratios, loss weights, codebook sizes, and neural-network dimensions remain unspecified unless explicitly marked as proposed.

## 1. Experimental objective and controlled scope

**[ACTIVE PLAN REQUIREMENT] Research question:** Does a typed
dependency-graph representation improve systematic generalization and
localized editing compared with a flat chronological representation when both
use the same category-conditioned constrained continuous-geometry decoder?

**[PLAN REQUIREMENT] Controlled domain:**

- single-solid CAD histories;
- sketches containing lines, arcs, circles, loops, profiles, and construction axes;
- extrusion and revolution operations;
- join and cut Boolean modes where the data and CAD kernel support them reliably;
- bounded history length and normalized geometric dimensions.

**[ACTIVE PLAN REQUIREMENT] Primary scientific comparison:**

1. one repaired flat chronological model; and
2. one minimal typed dependency-graph model.

Both conditions use one frozen category-conditioned constrained
continuous-geometry decoder, with capacity and training opportunity controlled
closely enough to isolate the input representation. The required evaluation is
ordinary validation, one predetermined systematic-generalization split, and
one localized numerical-edit test.

**[ACTIVE IMPLEMENTATION BOUNDARY]:** The deterministic profile-geometry
contract is implemented. Explicit neural profile-family prediction, compact
continuous parameter prediction, decoder integration, a successor flat
checkpoint, and the graph model are not implemented or trained. The epoch-44
checkpoint remains the frozen failed unconstrained baseline.

**[HISTORICAL PROPOSED V0] Earlier phased causal controls:** the superseded
candidate protocol organized `B0-FLAT-MIXED-VQ`,
`F0-FLAT-FACTORED-DISCRETE`, and `A-GRAPH-DISCRETE` as Phase 1, followed by
`F1-FLAT-FACTORED-HYBRID` and `B-GRAPH-HYBRID` for a Phase 2 geometry
ablation. This remains useful future-design context but is not the active
three-week experiment.

**[PROPOSED V0] Unit of modeling:** one complete, ordered CAD feature history that deterministically reconstructs one final solid. Intermediate states must also execute successfully so that an invalid early feature is not hidden by a later feature.

**[PROPOSED V0] Out of scope for the first frozen experiment:** fillets, chamfers, sweeps, lofts, assemblies, multiple final bodies, free-form splines, and constraint solving beyond what is necessary to execute the recorded sketch geometry.

## 2. Minimal CAD vocabulary

### 2.1 Graph-level node vocabulary

| Node type | Status | Meaning |
|---|---|---|
| `REFERENCE_PLANE` | **[IMPLEMENTED V1]** | A local 2D coordinate frame on which a sketch is placed. |
| `SKETCH` | **[PLAN REQUIREMENT + IMPLEMENTED V1]** | A collection of 2D primitives and loop topology in a reference-plane frame. |
| `PROFILE` | **[PLAN REQUIREMENT + IMPLEMENTED V1]** | One selectable closed sketch region, represented by an outer loop and optional inner loops. |
| `AXIS` | **[PLAN REQUIREMENT + IMPLEMENTED V1]** | A construction axis that can be referenced by a revolution. |
| `EXTRUDE` | **[PLAN REQUIREMENT + IMPLEMENTED V1]** | An extrusion feature consuming one profile. |
| `REVOLVE` | **[PLAN REQUIREMENT + IMPLEMENTED V1]** | A revolution feature consuming one profile and one axis. |

**[IMPLEMENTED V1] Minimality choice:** lines, arcs, circles, and loops are typed subrecords inside a `SKETCH`, not graph nodes. This keeps graph nodes at the feature/reference level while retaining sketch topology in the structural data.

**[FUTURE SCHEMA/MODEL DECISION]:** Reconsider whether sketch primitives and
loops should become first-class graph nodes before freezing the graph neural
model. Version 1 does not make them nodes; a future model-side derived graph
or schema version may revisit this because it changes graph size, message
passing, tokenization, matching metrics, and the meaning of “node accuracy.”

**[IMPLEMENTED V1]:** `REFERENCE_PLANE` is a node and every sketch has one
outgoing `PLACED_ON` edge. A later schema may reconsider this representation,
but version 1 producers and consumers must retain it.

### 2.2 Sketch primitive vocabulary

| Primitive/subrecord | Structural fields | Geometric fields |
|---|---|---|
| `LINE` | primitive ID, type, owning sketch, loop membership, position in loop | 2D start and end points |
| `ARC` | primitive ID, type, owning sketch, loop membership, position in loop, traversal direction | 2D start, midpoint, and end points |
| `CIRCLE` | primitive ID, type, owning sketch, loop membership | 2D center and radius |
| `LOOP` | loop ID, owning sketch, ordered primitive references, outer/inner role | no independent geometry; geometry is induced by its primitives |

All coordinates above are in the owning sketch's local 2D frame.

**[IMPLEMENTED V1] Arc parameterization:** start/mid/end points. It is easy
to serialize and determines the circular arc except in degenerate cases.

**[FUTURE SCHEMA DECISION]:** Version 1 reconstructs evaluated geometry, not
analytic sketch constraints or dimensions. Supporting constraint-level CAD
requires a later schema decision.

### 2.3 Operation vocabulary

| Operation | Required references | Required operation fields |
|---|---|---|
| `EXTRUDE` | one `PROFILE` | distance, direction, Boolean mode |
| `REVOLVE` | one `PROFILE`, one `AXIS` | angle, direction, Boolean mode |

**[PLAN REQUIREMENT] Boolean vocabulary:** `JOIN` and `CUT` where data quality permits.

**[IMPLEMENTED V1] Bootstrap mode:** the first operation must use `NEW_BODY`.
Every later operation must use `JOIN` or `CUT`. Join-on-empty is not a version
1 convention.

## 3. Typed graph definition

Let a history be a typed attributed directed graph

\[
G=(V,E,\tau_V,\tau_E,S,X),
\]

where `V` is the node set, `E` is the directed edge set, `tau_V` and `tau_E` are node and edge types, `S` contains structural/categorical fields, and `X` contains numerical geometry.

### 3.1 Implemented edge-direction convention

**[IMPLEMENTED V1] Convention:** dependency and reference edges point from
the consuming, dependent, or contained entity to the resource it needs.
Chronology is stored separately in `operation_sequence`.

Examples:

```text
SKETCH  --PLACED_ON----> REFERENCE_PLANE
PROFILE --DEFINED_IN---> SKETCH
AXIS    --DEFINED_IN---> SKETCH
EXTRUDE --USES_PROFILE-> PROFILE
REVOLVE --USES_PROFILE-> PROFILE
REVOLVE --USES_AXIS----> AXIS
REVOLVE_2 --DEPENDS_ON-> REVOLVE_1
```

The earlier research-plan example used the opposite direction. ADR-0001
resolves that inconsistency for version 1. Direction and edge meaning must
remain identical for every model, metric, and dataset split consuming this
schema.

### 3.2 Edge types and cardinalities

| Edge type | Allowed source | Allowed target | Cardinality and meaning |
|---|---|---|---|
| `PLACED_ON` | `SKETCH` | `REFERENCE_PLANE` | Exactly one per sketch. |
| `DEFINED_IN` | `PROFILE`, `AXIS` | `SKETCH` | Exactly one per profile or sketch-owned axis. |
| `USES_PROFILE` | `EXTRUDE`, `REVOLVE` | `PROFILE` | Exactly one per operation in the minimal scope. |
| `USES_AXIS` | `REVOLVE` | `AXIS` | Exactly one per revolve; forbidden on extrude. |
| `DEPENDS_ON` | `EXTRUDE`, `REVOLVE` | Earlier `EXTRUDE` or `REVOLVE` | None for the first operation; exactly the immediately preceding operation thereafter. |

**[IMPLEMENTED V1] Chronology encoding:** `operation_sequence` contains every
operation exactly once and is the authoritative execution order.

**[IMPLEMENTED V1] Feature/body dependency:** a single current solid is
updated in `operation_sequence` order. Every operation after the first has
one `DEPENDS_ON` edge to the immediately preceding operation. The edge points
backward to the body state consumed; no `BODY_STATE` or `MODIFIES` node/edge
is included.

**[FUTURE SCHEMA DECISION]:** Branching histories, multiple bodies, or
reference-dependent features may require explicit body-state nodes or richer
dependency edges. That question does not change the frozen version 1
contract.

## 4. Node schemas and dependencies

### 4.1 `REFERENCE_PLANE`

- **Required structural fields [PROPOSED V0]:** `node_id`, `node_type`.
- **Required geometric fields [PROPOSED V0]:** 3D `origin`, orthonormal 3D `x_axis`, `y_axis`, and derived or stored `normal`.
- **Required dependencies [PROPOSED V0]:** none.
- **Validity [PROPOSED V0]:** finite coordinates; unit, mutually orthogonal axes within tolerance; right-handed frame; unique ID.

### 4.2 `SKETCH`

- **Required structural fields [PLAN REQUIREMENT + PROPOSED V0]:** `node_id`, `node_type`, primitive subrecords, loop subrecords, primitive count, loop topology.
- **Required geometric fields [PLAN REQUIREMENT]:** local 2D primitive coordinates and dimensions.
- **Required dependencies [PROPOSED V0]:** exactly one `PLACED_ON` reference-plane edge.
- **Validity [PROPOSED V0]:** unique primitive and loop IDs; every referenced primitive exists; loop traversal is continuous and closed within tolerance; nonzero-length lines; valid non-collinear arcs; positive-radius circles; finite normalized values.

### 4.3 `PROFILE`

- **Required structural fields [PLAN REQUIREMENT + PROPOSED V0]:** `node_id`, `node_type`, one outer-loop reference, zero or more inner-loop references.
- **Required geometric fields [PROPOSED V0]:** none stored independently; geometry is induced by referenced loops.
- **Required dependencies [IMPLEMENTED V1]:** exactly one `DEFINED_IN` edge to the owning `SKETCH`.
- **Validity [PROPOSED V0]:** all loop references exist in that sketch; outer loop is closed and non-self-intersecting; inner loops are closed, lie inside the outer loop, and do not cross one another; enclosed area exceeds tolerance.

### 4.4 `AXIS`

- **Required structural fields [PLAN REQUIREMENT + PROPOSED V0]:** `node_id`, `node_type`, axis source/category if the dataset distinguishes construction-line versus reference-axis origins.
- **Required geometric fields [PLAN REQUIREMENT + PROPOSED V0]:** a local 2D point and normalized 2D direction in the owning sketch frame. The world-space line is derived from the sketch frame.
- **Required dependencies [IMPLEMENTED V1]:** exactly one `DEFINED_IN` edge to a `SKETCH`.
- **Validity [PROPOSED V0]:** finite point; nonzero direction normalized within tolerance; owning sketch and plane exist.

### 4.5 `EXTRUDE`

- **Required structural fields [PLAN REQUIREMENT]:** `node_id`, `node_type`, direction category, Boolean mode, profile identity.
- **Required geometric fields [PLAN REQUIREMENT]:** extrusion distance.
- **Required dependencies [IMPLEMENTED V1]:** exactly one `USES_PROFILE` edge; no `USES_AXIS` edge; and, unless first, one outgoing `DEPENDS_ON` edge to the immediately preceding operation.
- **Validity [PROPOSED V0]:** profile exists and is valid; distance is finite and has magnitude above tolerance; direction and signed-distance convention are not contradictory; Boolean mode is allowed at that history position; execution produces the required single valid solid.

### 4.6 `REVOLVE`

- **Required structural fields [PLAN REQUIREMENT]:** `node_id`, `node_type`, direction category, Boolean mode, profile identity, axis identity.
- **Required geometric fields [PLAN REQUIREMENT]:** revolution angle; axis position/orientation is carried by the referenced `AXIS` geometry.
- **Required dependencies [IMPLEMENTED V1]:** exactly one `USES_PROFILE` edge and one `USES_AXIS` edge; and, unless first, one outgoing `DEPENDS_ON` edge to the immediately preceding operation.
- **Validity [PROPOSED V0]:** profile and axis exist; both can be transformed into a common frame; angle is finite and nonzero; sweep does not create a kernel-invalid result; Boolean mode is allowed; execution produces the required single valid solid.

**[IMPLEMENTED V1]:** Extrusion distance is a positive magnitude and
direction is a separate categorical field. Revolve angle is measured in
degrees in `(0, 360]`, including a full revolution; direction remains
categorical.

## 5. Structural, geometric, and joint fields

| Field | Classification | Rationale/status |
|---|---|---|
| Node and primitive type | Structural | **[PLAN REQUIREMENT]** Categorical identity. |
| Operation order and dependency type | Structural | **[PLAN REQUIREMENT]** Determines construction organization. |
| Profile and axis references | Structural | **[PLAN REQUIREMENT]** Identity/reference, not numerical realization. |
| Boolean mode | Structural | **[PLAN REQUIREMENT]** Categorical operation semantics. |
| Primitive count and loop membership/order | Structural | **[PLAN REQUIREMENT]** Sketch topology. |
| Outer/inner loop role | Structural | **[PROPOSED V0]** Needed to reconstruct regions with holes. |
| Plane identity | Structural | **[PROPOSED V0]** Reference identity. |
| Sketch 2D coordinates and radii | Geometric | **[PLAN REQUIREMENT]** Numerical realization. |
| Plane origin and orientation | Geometric | **[PROPOSED V0]** Numerical placement. |
| Axis point and direction | Geometric conditioned on structure | **[PLAN REQUIREMENT]** Geometry attached to an identified axis. |
| Extrusion distance | Geometric conditioned on operation/profile | **[PLAN REQUIREMENT]** Numerical operation parameter. |
| Revolve angle | Geometric conditioned on operation/profile/axis | **[PLAN REQUIREMENT]** Numerical operation parameter. |
| Profile validity/compatibility | Joint structural-geometric constraint | **[PLAN REQUIREMENT]** Topology and coordinates jointly determine validity. |
| Boolean executability | Joint structural-geometric constraint | **[PROPOSED V0]** Mode, order, and resulting solid geometry all matter. |

The split is a modeling factorization, not a claim of statistical independence.

## 6. Global validity contract

A sample is valid only if all applicable levels below pass.

### 6.1 Schema and reference validity

**[PROPOSED V0]**

- every ID is unique in its namespace;
- all edge endpoints and subrecord references exist and have allowed types;
- every required edge has the stated cardinality;
- forbidden edge/node combinations are absent;
- all required fields are present, finite, and within the frozen normalized domain;
- `operation_sequence` contains every operation exactly once;
- each `DEPENDS_ON` edge points to an earlier operation, and version 1 forms
  one linear current-body dependency chain;
- every resource is defined no later than its first use in the deterministic serialization.

### 6.2 Sketch and profile validity

**[PROPOSED V0]**

- primitives are nondegenerate;
- referenced loops close within a documented tolerance;
- profiles have positive area;
- profile boundaries do not self-intersect;
- hole loops lie within their outer loop and do not intersect other boundaries;
- the CAD kernel accepts the reconstructed wires/faces.

### 6.3 Operation and Boolean validity

**[PLAN REQUIREMENT]** Every decoded history must be tested for sequence validity and final-solid validity.

**[PROPOSED V0]**

- each operation has exactly the parameters valid for its type;
- referenced profiles/axes are geometrically compatible;
- the initial operation creates a body under the frozen first-feature convention;
- each later join/cut is applicable to the current body;
- each intermediate result and the final result are one nonempty valid solid;
- reconstruction is replayed by the chosen CAD kernel, not accepted solely by token/schema checks.

### 6.4 Determinism and round trip

**[PROPOSED V0]** A valid source history must satisfy:

```text
source history
  -> canonical graph
  -> deterministic flat serialization
  -> parsed graph/history
  -> executable CAD sequence
```

The graph before and after serialization must be equal after canonical ID renumbering and numeric tolerance. Kernel output equivalence should be checked with a frozen geometric tolerance and, if available, volume and shape-distance checks.

**[IMPLEMENTED V1]:** The controlled execution audit uses OpenCascade through
the PythonOCC boundary, deterministic canonical JSON, and versioned package
tolerances. A future publication protocol must still freeze which equivalence
and kernel metrics are primary across model comparisons.

## 7. Equivalent flat/mixed representation

The flat baseline must contain the same information as the graph representation. It may change factorization and inductive bias, but not silently omit references or geometry.

### 7.1 Canonical ordering

**[IMPLEMENTED V1]** The model-data boundary uses this deterministic node
order:

1. reference planes;
2. one group for each entry in `operation_sequence`, containing its sketch,
   profile, optional revolve axis, and operation;
3. any remaining nodes by stable ID.

Primitive slots are sorted by stable primitive ID. Input and target edges use
canonical node positions. Semantic ordering remains explicit for operations,
loop traversal, inner-loop references, and other ordered fields. Ordering does
not depend on dictionary insertion, file traversal, manifest traversal, or
hash iteration.

### 7.2 Implemented flat tensor contract

**[IMPLEMENTED V1]** For `N` canonical nodes, the flat input contains:

```text
categorical_ids [N, 10]
geometry       [N, 39]
geometry_mask  [N, 39]
```

The ten categorical fields are node type, operation type, Boolean mode,
direction, reference plane, four primitive slots, and loop role. Geometry
uses fixed physical normalizers and an applicability mask; zero remains a
valid applicable value.

The shared reconstruction target additionally contains typed directed edges,
edge types, Boolean targets, and operation-sequence node positions. The flat
encoder deliberately receives no dependency edges, but the decoder predicts
the shared structural target. The typed-graph adapter receives those edges
explicitly. Both views therefore share one physical example and target while
differing in input inductive bias.

The earlier logical text grammar was a design sketch and is not an
implemented token format.

### 7.3 Mixed latent baseline

**[PLAN REQUIREMENT]** The primary flat baseline uses one mixed representation over the serialized history without explicit typed-graph factorization.

**[IMPLEMENTED V1]** `B0-FLAT-MIXED-VQ` consumes the flat categorical,
geometry, applicability, and node-mask tensors, without explicit dependency
edges, and produces one mixed VQ latent memory. Its decoder predicts node
types, categorical attributes, normalized geometry, typed edge pairs, and
operation pointers against the shared reconstruction target.

**[IMPLEMENTED V1 + PROPOSED COMPARISON] Phase status:**
`B0-FLAT-MIXED-VQ` is implemented and remains the required mixed baseline in
Phase 1. Its final scientific acceptance is not complete. `B1-FLAT-MIXED-CONT`
is optional and unimplemented.

## 8. Exact model comparisons

### 8.1 Active three-week comparison

| Condition | Input representation | Shared geometry condition | Required status |
|---|---|---|---|
| Repaired flat | Canonical chronological serialization without dependency edges | Category-conditioned constrained continuous-geometry decoder | Required |
| Minimal typed graph | Typed dependency graph under schema version 1 | The same category-conditioned constrained continuous-geometry decoder | Required |

The shared decoder must explicitly predict profile family and compact
continuous profile parameters, then use the family to select the deterministic
profile construction. Its interface, targets, losses, and evaluation treatment
must be frozen before the graph condition begins.

### 8.2 Historical candidate model matrix

The matrix below is preserved from the broader protocol. Its fully discrete
models, full geometry ablation, and optional baselines are deferred under
ADR-0002.

| ID | Input representation | Structure latent | Geometry latent | Decoder target | Required status |
|---|---|---|---|---|---|
| `B0-FLAT-MIXED-VQ` | Canonical flat serialization | Not separated | One mixed discrete VQ latent | Canonical flat stream | **[PLAN REQUIREMENT + PROPOSED V0] Phase 1 required** |
| `F0-FLAT-FACTORED-DISCRETE` | Canonical flat serialization, fields separated by type | Discrete VQ | Discrete VQ conditioned on flat structural latent | Canonical flat stream | **[PROPOSED V0] Phase 1 required** |
| `A-GRAPH-DISCRETE` | Typed graph plus geometry | Discrete VQ | Discrete VQ conditioned on graph/structure | Executable graph/history | **[PLAN REQUIREMENT + PROPOSED V0] Phase 1 required** |
| `F1-FLAT-FACTORED-HYBRID` | Canonical flat serialization, fields separated by type | Discrete VQ | Continuous latent conditioned on flat structural latent | Canonical flat stream | **[PROPOSED V0] Phase 2 required for the full geometry-ablation claim** |
| `B-GRAPH-HYBRID` | Typed graph plus geometry | Discrete VQ | Continuous latent conditioned on structural latent | Executable graph/history | **[PLAN REQUIREMENT + PROPOSED V0] Phase 2 required for the full geometry-ablation claim** |
| `B1-FLAT-MIXED-CONT` | Canonical flat serialization | Not separated | One mixed continuous latent | Canonical flat stream | **[PLAN REQUIREMENT if time permits] Optional** |

### 8.3 Controlled variables

**[PLAN REQUIREMENT]** Baselines must be capacity matched.

**[PROPOSED V0]** All required comparisons use:

- identical train/validation/test manifests;
- identical normalization and canonicalization;
- the same executable target semantics;
- approximately matched encoder/decoder parameter counts, decoder depth/width, latent information budget, optimizer family, update count, batch exposure, early-stopping rule, and hyperparameter-search budget;
- the same number of random seeds, with per-seed and aggregate results;
- teacher-forced reconstruction results and fully autoregressive/fully predicted results reported separately.

Exact architecture dimensions, VQ codebook counts/sizes, optimizer settings, and tolerances are **[LATER IMPLEMENTATION DECISION]** items after the data audit and compute budget are known.

### 8.4 Historical claims supported by the broader comparisons

**[PROPOSED V0]** Compare models in matched pairs:

- In **Phase 1**, `B0-FLAT-MIXED-VQ` versus `F0-FLAT-FACTORED-DISCRETE` tests factorization: the effect of separating structure and geometry while keeping the representation flat.
- In **Phase 1**, `F0-FLAT-FACTORED-DISCRETE` versus `A-GRAPH-DISCRETE` tests typed graph structure with discrete geometry.
- In **Phase 2**, `F1-FLAT-FACTORED-HYBRID` versus `B-GRAPH-HYBRID` tests typed graph structure in the hybrid setting.
- The cross-phase comparison `A-GRAPH-DISCRETE` versus `B-GRAPH-HYBRID`, interpreted together with `F0-FLAT-FACTORED-DISCRETE` versus `F1-FLAT-FACTORED-HYBRID`, tests discrete versus continuous conditioned geometry without confusing that change with graph structure.

If Phase 1 omits `F0`, all graph-specific claims must be withdrawn. If Phase 2 is not completed, the paper may report only the Phase 1 factorization and graph results and must not claim a complete discrete-versus-continuous geometry ablation.

### 8.5 Graph encoder and decoder ordering

**[PLAN REQUIREMENT]** Candidate structural encoders are a typed GNN, graph Transformer, or typed-edge/sequence Transformer. Geometry must be conditioned on structure.

**[PROPOSED V0]** Decode in two stages:

1. predict node types, edge types, operation order, references, loop topology, and Boolean modes;
2. predict only geometry allowed by that decoded structure, using operation/type masks.

During early training, geometry may be conditioned on ground-truth structure. Final evaluation must also condition on predicted structure.

**[LATER IMPLEMENTATION DECISION]:** Choose the graph encoder and decide whether latents are global per history, local per feature, or hierarchical, after graph granularity is confirmed.

## 9. Training protocol and order

**[ACTIVE PLAN REQUIREMENT] Training order:**

1. integrate explicit profile-family prediction and compact continuous profile
   parameters with the deterministic geometry contract;
2. freeze the shared neural decoder interface, targets, losses, and validity
   checks;
3. timebox the repaired flat model and produce one reproducible successor
   checkpoint;
4. begin the minimal typed graph model as soon as the shared decoder interface
   is frozen;
5. evaluate both models on ordinary validation, one predetermined systematic
   split, and one localized numerical-edit test; and
6. expand experiments only after that comparison and its concise report or
   presentation are complete.

**[HISTORICAL PLAN REQUIREMENT + PROPOSED V0] Earlier training order:**

1. freeze data schema, converter, deterministic serialization, split manifests, and executor;
2. begin **Phase 1** by training `B0-FLAT-MIXED-VQ` and verifying decreasing loss plus executable reconstructions;
3. train `F0-FLAT-FACTORED-DISCRETE` and verify the matched flat factorization comparison;
4. train `A-GRAPH-DISCRETE` only after `F0` is reproducible, then freeze and evaluate the minimum rigorous Phase 1 experiment;
5. begin **Phase 2** only if pursuing the full geometry-ablation claim: train `F1-FLAT-FACTORED-HYBRID` as the matched flat-hybrid control;
6. train `B-GRAPH-HYBRID` only after Phase 1 and `F1` are reproducible, then run the paired discrete-versus-continuous comparisons;
7. add optional `B1-FLAT-MIXED-CONT`, locality losses, compatibility losses, or other ablations only after the models required for the intended claim are reproducible.

**[PLAN REQUIREMENT] Loss groups:**

- structural reconstruction: node/primitive types, edges, operation order, references, loop topology, Boolean modes;
- geometric reconstruction: coordinates, radii/dimensions, plane and axis geometry, extrusion distances, revolve angles;
- VQ codebook and commitment losses for every discrete latent group;
- optional locality-consistency loss;
- optional structure/geometry compatibility loss.

**[PROPOSED V0] Reporting requirement:** record each unweighted loss component, its weight, aggregate loss, gradient/optimization failures, and execution validity by epoch. Do not use test-set execution to select checkpoints.

**[LATER IMPLEMENTATION DECISION]:** Freeze loss forms and weights, teacher-forcing schedule, optimizer, learning-rate schedule, batch size, stopping rule, seed count, and hyperparameter budget after small-scale feasibility runs and before full comparisons.

## 10. Dataset units, splits, and leakage controls

### 10.1 Dataset records

**[PROPOSED V0]** Store or manifest the following for every example:

- immutable source/example ID and source family;
- canonical graph and canonical flat serialization;
- executable source sequence;
- normalization metadata;
- operation count and operation-pattern label;
- structural statistics and parameter ranges;
- executor outcome and structured failure category;
- counterfactual-family ID and edit metadata when applicable.

### 10.2 Ordinary reconstruction split

**[PLAN REQUIREMENT]** Include an ordinary random split.

**[IMPLEMENTED V1]** Split by physical source family, never by continuous or
quantized serialization. The controlled IID manifest uses deterministic
family-level 80/10/10 allocation. Counterfactual edit families and endpoints
inherit one evaluation partition and remain evaluation-only.

Future real-data or publication-scale corpora require a separately frozen
split decision; they do not silently inherit the controlled-corpus counts.

### 10.3 Unseen-combination split

**[PLAN REQUIREMENT]** Withhold selected dependency compositions while ensuring that every individual operation and component occurs in training.

**[IMPLEMENTED V1]** The controlled `operation_template` split withholds `ER`
as the primary test template and retains `RR` as secondary systematic
validation. This is a bounded template-combination test, not broad
dependency-composition generalization.

**[PROPOSED V0]** Assign a canonical operation/dependency signature to every history, for example operation types plus profile/axis reuse and Boolean modes. Select one or more complete signatures—such as a particular extrude-then-revolve dependency pattern—for test only. Verify and report that all atomic node types, edge types, modes, and parameter bins in test also occur in train.

**[LATER IMPLEMENTATION DECISION]:** After the mentor identifies the most important generalization benchmark, choose its exact withheld signatures from the audited frequency table; do not choose them after observing model results.

### 10.4 Length-generalization split

**[PLAN REQUIREMENT]** Train primarily on shorter histories and test longer ones.

**[IMPLEMENTED V1]** The current `history_depth` split trains on depth-one
histories and evaluates depth-two histories. It is explicitly provisional and
does not satisfy the research plan's later longer-history objective.

**[PROPOSED V0]** Candidate split: train/validate on one-to-three operation histories and test on four-operation histories, matching the example in the plan. If data are too sparse, freeze another nonoverlapping length boundary before training.

**[LATER IMPLEMENTATION DECISION]:** Confirm what “feature count” counts—only extrude/revolve operations or all graph nodes—and freeze the length boundary after the data audit.

### 10.5 Parameter interpolation and extrapolation splits

**[PLAN REQUIREMENT]** Evaluate extrusion distance, revolve angle, sketch dimensions, and placements both in range and out of range.

**[IMPLEMENTED V1]** The current `geometry_extrapolation` manifest uses a
fixed sketch-extent statistic with in-range `E <= 2.0` and extrapolation
`E >= 2.5`. The generator also records fixed distance and angle grids, but the
current split does not yet constitute the complete per-parameter
interpolation/extrapolation matrix proposed below.

**[PROPOSED V0]** For each target parameter family:

- `INTERPOLATION`: hold out interior bins while retaining lower and higher bins in training;
- `EXTRAPOLATION-LOW`: test below the frozen training interval;
- `EXTRAPOLATION-HIGH`: test above the frozen training interval;
- keep structure distributions as similar as practical and report remaining imbalance.

Thresholds are computed from a pre-model data audit and then frozen. Normalization must not leak held-out extrema into training unless a fixed physical normalization is chosen in advance.

**[LATER IMPLEMENTATION DECISION]:** Confirm physical ranges, binning, whether low extrapolation is meaningful for each positive parameter, and the normalization policy.

### 10.6 Counterfactual-pair split

**[IMPLEMENTED V1]** Every selected edit family, its continuous/quantized
sample variants, and both physical endpoints inherit one partition. Endpoint
exclusion manifests prevent evaluation endpoints from entering later
training corpora. Version 1 contains deterministic one-factor pairs; learned
transfer-partner groups remain future work.

**[IMPLEMENTED V1 + MENTOR DECISION]:** The controlled benchmark corpus is
procedural. Mentor confirmation is still required on whether the principal
study also needs a real-data validation subset and what role that subset
plays in the final claim.

## 11. Localized editing tasks

Each edit evaluation receives a valid source history, a target entity/field, and a requested replacement or delta. It returns a decoded history. The evaluator compares it with an explicit intended target and an explicit preservation mask.

### 11.1 Editing intervention protocol

The following protocol is **[PROPOSED V0]** and must be confirmed as the immediate **[MENTOR DECISION]** on editing mechanism.

#### Edit request

Every model receives the same explicit request

```text
r = {
  target_record_id,
  target_field,
  edit_operator,     # SET or ADD
  edit_value,
  preservation_mask
}
```

`target_record_id` is the canonical operation/node index. `target_field` identifies a legal field such as `EXTRUDE.distance` or `REVOLVE.angle`. `edit_value` is represented in the same normalized units for every model. The preservation mask identifies all fields that must remain equal to the source.

This primary benchmark evaluates **instruction-conditioned localized executable editing**: the model is explicitly told both what to change and what to preserve. It must not be described as evidence that the representation automatically discovers which fields should remain unchanged.

#### Primary edit path

1. Encode the complete source history into the model's ordinary latent representation.
2. Embed `r` and supply it to a learned edit-conditioned decoder, or to a small edit-conditioning module immediately before that decoder.
3. Decode a complete edited history in one forward pass.
4. Execute the decoded history and score the target and preservation masks.

The primary protocol does **not** replace a latent variable with a code taken from another sample, and it does **not** optimize a latent at evaluation time. Those are separate optional interventions and must not be mixed into the primary comparison.

For numerical edits, the requested target value is explicitly supplied; the decoder is not asked to infer which change the evaluator intended. During paired edit training, the input is `(source_history, r)` and the target is the validated edited history. Reconstruction training remains a separate objective where `r` is a no-op request.

#### Structure and reference handling

For `EDIT-EXTRUDE-DISTANCE`, `EDIT-REVOLVE-ANGLE`, and numerical edit transfer, the primary **controlled-structure** evaluation supplies the preservation mask and clamps the source node types, topology, operation order, profile references, axis references, and Boolean modes. The model predicts edited geometry only. This isolates numerical editing quality from structural reconstruction errors.

A second **end-to-end** evaluation supplies the same request but does not clamp structure or references; the model must reproduce them. Results from controlled-structure and end-to-end editing are reported separately. Ground-truth structure must never be described as predicted.

For extrude-to-revolve substitution, the request explicitly supplies the replacement operation type and, under the controlled version, the valid profile and axis references. The end-to-end version predicts those references subject to type masks.

#### Equal request interface for flat baselines

The canonical graph ID and canonical flat-record index are deterministically aligned. A flat model receives the identical semantic request `r`; `target_record_id` addresses the corresponding serialized operation record and `target_field` addresses its field slot. It receives no graph adjacency. Graph models may use the target node and its typed neighborhood; flat models use record position and serialized references.

All models receive the same target value, operation (`SET` or `ADD`), preservation mask, and controlled-versus-end-to-end condition. Thus the comparison changes representation, not the information contained in the edit instruction.

#### Secondary inferred-preservation benchmark

**[LATER IMPLEMENTATION DECISION] Proposed later extension:** evaluate the same editing tasks with a reduced request containing only:

```text
r_inferred = {
  target_record_id,
  target_field,
  edit_operator,     # SET or ADD
  edit_value
}
```

The preservation mask is withheld from the model. The model must infer which fields should remain unchanged from the source graph or serialized source structure. The evaluator still retains the ground-truth preservation region for scoring. Both controlled-structure and end-to-end conditions remain applicable: controlled-structure still clamps the same structural and reference fields but does not supply a preservation mask, while end-to-end predicts structure and references. Results must be reported separately from the primary preservation-mask benchmark.

This secondary benchmark tests inferred edit locality. It is not part of the minimum Phase 1 experiment and is not required for either phase's principal claims.

#### Optional interventions, reported separately

- **Latent replacement:** replace a feature-local code only if the final architecture exposes aligned feature-local latents and a valid donor-selection rule is frozen.
- **Latent optimization:** optimize a latent against the target request while regularizing preservation, with identical step and compute budgets for all compatible models.

Neither optional method is part of the primary localized-editing result unless the mentor changes this protocol before implementation.

### 11.2 `EDIT-EXTRUDE-DISTANCE`

- **[PLAN REQUIREMENT] Target:** one selected `EXTRUDE.distance`.
- **[PROPOSED V0] Intervention:** set an absolute distance or apply a signed delta drawn from frozen interpolation/extrapolation ranges.
- **[PLAN REQUIREMENT] Preserve:** every other feature and parameter.
- **[PROPOSED V0] Exact invariants:** node/edge set, profile reference, operation order, direction, Boolean mode, all sketch/profile/axis geometry, and all nontarget operation parameters.

### 11.3 `EDIT-REVOLVE-ANGLE`

- **[PLAN REQUIREMENT] Target:** one selected `REVOLVE.angle`.
- **[PROPOSED V0] Intervention:** set an absolute angle or apply a signed delta.
- **[PLAN REQUIREMENT] Preserve:** profile, axis, earlier features, Boolean mode, and unrelated geometry.
- **[PROPOSED V0] Exact invariants:** node/edge set, all references and order, direction, Boolean mode, profile geometry, axis geometry, and all nontarget operation parameters.

### 11.4 `TRANSFER-NUMERICAL-EDIT`

- **[PLAN REQUIREMENT] Target:** transfer a numerical edit from one history to a structurally compatible history.
- **[PROPOSED V0] Compatibility:** same target operation type and parameter semantics, with valid target references and matching local structural neighborhood.
- **[PROPOSED V0] Intervention:** transfer a normalized delta, not necessarily the absolute source value, so differently scaled compatible models receive an analogous edit.
- **[PROPOSED V0] Preserve:** all target-history fields outside the selected target parameter.

**[LATER IMPLEMENTATION DECISION]:** Under the primary explicit-request protocol, freeze whether transfer uses an absolute value, raw delta, or normalized delta, and define structural compatibility exactly. Learned code replacement remains a separate optional intervention.

### 11.5 `SUBSTITUTE-EXTRUDE-WITH-REVOLVE`

- **[PLAN REQUIREMENT] Scope:** only restricted cases with a valid profile and axis.
- **[PROPOSED V0] Target:** replace one `EXTRUDE` node with one `REVOLVE`, retain the profile reference, attach a pre-existing compatible axis, and provide a valid angle/direction/mode.
- **[PROPOSED V0] Preserve:** all nodes and edges outside the substituted operation's local dependency neighborhood, plus all unaffected geometry.

**[LATER IMPLEMENTATION DECISION]:** Decide whether this categorical substitution is a required primary task or a secondary stress test. It is not directly comparable to a one-number edit and requires a policy for selecting the axis and angle.

### 11.6 Edit-pair construction

**[PROPOSED V0]** Whenever possible, generate paired ground truth by replaying the source history with exactly one controlled field changed and retaining the pair only if the edited history passes all validity checks. Record why rejected pairs fail.

## 12. Metrics

Metrics must be reported on the ordinary test set and separately on every systematic split. Invalid decodes remain in denominators unless a metric explicitly conditions on execution.

### 12.1 Structural reconstruction

**[PLAN REQUIREMENT]** Report node-type, edge-type, operation-sequence, profile-reference, and axis-reference accuracy.

**[PROPOSED V0] Definitions:**

- canonical exact graph match after deterministic ID renumbering;
- node/primitive-type accuracy and macro-F1 over aligned canonical records;
- typed-edge precision, recall, and F1;
- exact operation-sequence accuracy and per-position accuracy;
- exact profile-reference and axis-reference accuracy;
- Boolean-mode accuracy;
- exact loop-topology match.

Malformed or unalignable predictions score zero for affected exact-match metrics and enter the parse/reference failure counts.

### 12.2 Geometric reconstruction

**[PLAN REQUIREMENT]** Report errors for coordinates, distances, positions, orientations, and angles.

**[PROPOSED V0] Definitions:**

- normalized MAE and RMSE for scalar/coordinate fields;
- Euclidean point error in both normalized and physical units;
- radius and extrusion-distance absolute/relative error;
- orientation error as the smaller angular difference between equivalent normalized directions under the frozen direction convention;
- revolve-angle circular or bounded absolute error, according to the frozen angle convention;
- report metrics both over all decodes (invalid/missing fields penalized) and over structurally correct decodes.

### 12.3 Executability

**[PLAN REQUIREMENT] Metrics:** valid-sequence rate and valid-solid rate.

**[PROPOSED V0] Definitions:**

\[
\text{valid-sequence rate}=\frac{N_{\text{schema+reference+ordered history valid}}}{N_{\text{all decoded histories}}}
\]

\[
\text{valid-solid rate}=\frac{N_{\text{kernel executes to one nonempty valid solid}}}{N_{\text{all decoded histories}}}
\]

Also report conditional kernel success among valid sequences, and success after every intermediate operation.

### 12.4 Structured execution failures

**[PLAN REQUIREMENT]** Record kernel failures rather than hiding them.

**[PROPOSED V0] Mutually exclusive first-failure categories:**

1. parse/schema failure;
2. missing or type-invalid reference;
3. sketch/profile construction failure;
4. operation-parameter or compatibility failure;
5. Boolean failure;
6. invalid/empty/multiple-body final geometry;
7. CAD-kernel exception or timeout;
8. unknown failure.

### 12.5 Codebook behavior

**[PLAN REQUIREMENT]** For every discrete codebook report utilization, perplexity, and dead-code rate.

**[PROPOSED V0] Definitions over a frozen evaluation corpus:**

- utilization = number of selected entries divided by codebook size;
- perplexity = `exp(-sum_i p_i log p_i)` from empirical assignment frequencies;
- dead-code rate = fraction of entries with zero assignments, plus a separately reported near-dead threshold chosen before evaluation.

Report these separately for structure, geometry, and the mixed baseline; raw values are not directly comparable without codebook sizes and assignment counts.

### 12.6 Edit success and preservation

Let `T` be target fields, `U_s` unaffected structural fields, and `U_g` unaffected geometric fields. Let `d` be a normalized per-field distance and `epsilon_f` a frozen success tolerance.

**[PROPOSED V0] Target success:**

\[
\text{target-success}=\frac{1}{|T|}\sum_{f\in T}\mathbf{1}[d(\hat f,f^*)\le\epsilon_f]
\]

where `f*` is the intended edited value.

**[PROPOSED V0] Structural preservation:** exact-match fraction over `U_s`, plus an all-unaffected-structure-exact indicator.

**[PROPOSED V0] Geometric preservation:**

\[
1-\frac{1}{|U_g|}\sum_{f\in U_g}\min(1,d(\hat f,f_{source}))
\]

and raw unaffected-field MAE, reported together.

**[PLAN REQUIREMENT] Edit execution validity:** valid-sequence and valid-solid rates after editing.

**[PLAN REQUIREMENT] Locality concept:** targeted change divided by unintended change outside the edit region.

**[PROPOSED V0] Raw locality ratio:**

\[
L_{raw}=\frac{\Delta_T}{\Delta_U+\epsilon},
\]

where `Delta_T` is achieved normalized change on target fields and `Delta_U` is total normalized change outside the target mask. Because this is unbounded and can reward an incorrect oversized target edit, report it only alongside target error and preservation.

**[PROPOSED V0] Bounded companion locality:**

\[
L_{bounded}=\text{target-success}\cdot\frac{\Delta_T}{\Delta_T+\Delta_U+\epsilon}.
\]

**[LATER IMPLEMENTATION DECISION]:** Confirm field weights, normalizers, success tolerances, treatment of symmetric/equivalent geometry, and the primary locality formula before evaluation code is frozen.

### 12.7 Generalization reporting

**[PLAN REQUIREMENT]** Repeat reconstruction, executability, and edit metrics for unseen combinations, longer histories, interpolation, and extrapolation.

**[PROPOSED V0]** Report absolute performance and the gap from the matched in-distribution set. Include confidence intervals across seeds and, where applicable, bootstrap confidence intervals across histories or counterfactual families.

## 13. Evaluation matrix

| Evaluation | Phase 1: `B0`, `F0`, `A` | Phase 2: `F1`, `B` | Optional |
|---|---:|---:|---:|
| Ordinary reconstruction | Required for all three | Required for both for full geometry-ablation claim | `B1` if trained |
| Ordinary executability | Required for all three | Required for both for full geometry-ablation claim | `B1` if trained |
| Unseen combinations | Required for all three | Required for both for full geometry-ablation claim | `B1` if trained |
| Length generalization | Required for all three | Required for both for full geometry-ablation claim | `B1` if trained |
| Parameter interpolation | Required for all three | Required for both for full geometry-ablation claim | `B1` if trained |
| Parameter extrapolation | Required for all three | Required for both for full geometry-ablation claim | `B1` if trained |
| Primary preservation-mask extrude-distance edit | Required for all three | Required for both for full geometry-ablation claim | `B1` if trained |
| Primary preservation-mask revolve-angle edit | Required for all three | Required for both for full geometry-ablation claim | `B1` if trained |
| Primary preservation-mask numerical edit transfer | Required for all three | Required for both for full geometry-ablation claim | `B1` if trained |
| Inferred-preservation editing | Later extension; not Phase 1 minimum | Later extension; not required for geometry claim | If pursued |
| Extrude-to-revolve substitution | Secondary pending confirmation | Not required for geometry claim | If pursued |
| Codebook diagnostics | Required for every applicable codebook | Required for every applicable codebook for full geometry-ablation claim | Where applicable |

**[PLAN REQUIREMENT] Minimum convincing outcome:** the proposed models may be merely comparable on ordinary reconstruction, but should show a material and repeatable advantage in systematic generalization or localized editing. **[PROPOSED V0]** Phase 1 must support its factorization and graph comparisons independently. If Phase 2 is completed, the discrete-versus-continuous geometry comparison should expose a consistent tradeoff rather than being interpreted only through one aggregate score.

## 14. Freeze gates and reproducibility artifacts

### Gate 1: data feasibility

**[PLAN REQUIREMENT]** By the end of the initial data/representation phase, profile references, axes, operation order, and executable extrude/revolve histories must be recoverable. If not, simplify the schema or use procedural data before model development.

### Gate 2: round-trip execution

**[PLAN REQUIREMENT]** Before training, deterministic serialization and reconstruction must execute with an audited success rate, and failures must be categorized.

### Gate 3: Phase 1 minimum rigorous experiment

**[PLAN REQUIREMENT + PROPOSED V0]** Phase 1 is complete only when `B0-FLAT-MIXED-VQ`, `F0-FLAT-FACTORED-DISCRETE`, and `A-GRAPH-DISCRETE` are reproducible and evaluated on identical frozen manifests. Graph-specific claims require both Phase 1 pairwise comparisons.

### Gate 4: Phase 2 full geometry ablation

**[PLAN REQUIREMENT + PROPOSED V0]** Proceed to `B-GRAPH-HYBRID` only after Phase 1 and the matched `F1-FLAT-FACTORED-HYBRID` control are reproducible. A full discrete-versus-continuous conditioned-geometry claim requires both Phase 2 models and the paired cross-phase comparisons.

### Gate 5: meaningful benchmark

**[PLAN REQUIREMENT]** The project must produce meaningful generalization and locality results before adding operations or secondary architectural complexity.

**[PROPOSED V0] Frozen artifacts:**

- schema version and field dictionary;
- canonicalization/serialization version;
- executor/kernel version and tolerance file;
- dataset record manifest and audit report;
- split manifests, including counterfactual-family grouping;
- model configs and parameter/latent-capacity table;
- training seeds and checkpoint-selection rule;
- evaluation config, edit masks, metric tolerances, and failure taxonomy.

## 15. Resolved v1 decisions and remaining mentor questions

ADR-0001 and the checked-in version 1 packages resolve these earlier
questions for the current controlled benchmark:

1. primitives and loops are stable sketch subrecords, not graph nodes;
2. `REFERENCE_PLANE` is a node and `PLACED_ON` remains an edge;
3. references use consumer-to-resource direction and `DEFINED_IN`;
4. `operation_sequence` is execution authority, later operations point
   backward with `DEPENDS_ON`, and the first operation uses `NEW_BODY`;
5. the controlled corpus is procedural and independent execution audits use
   OpenCascade through PythonOCC.

These choices may be revisited only as explicit future model-view or schema
decisions; they are not unresolved properties of existing version 1 data.

The following scientific questions still require mentor confirmation before
their corresponding comparison or evaluation is frozen:

1. **Graph-model granularity:** consume version 1 nodes directly or derive a
   finer model-side graph with primitive/loop nodes.
2. **Real-data role:** whether a real extrude/revolve subset is required for
   validation beyond the procedural controlled benchmark.
3. **Required baseline set and phases:** approve `B0`/`F0`/`A` as the Phase 1
   minimum and `F1`/`B` only when pursuing the full geometry-ablation claim.
4. **Editing mechanism:** approve the explicit-request, one-forward-pass
   preservation-mask protocol as instruction-conditioned localized
   executable editing, with controlled-structure and end-to-end conditions;
   latent replacement, latent optimization, and inferred-preservation
   editing remain later alternatives.
5. **Most important generalization benchmark:** unseen compositions, history
   length, parameter extrapolation, or localized edit transfer. The chosen
   priority determines dataset and split design.

### 15.1 Deferred implementation checklist

The following still require documented choices before their relevant artifact is frozen, but they do not need mentor approval in the next discussion:

- graph encoder family and global versus feature-local latents;
- sketch and arc parameterization, constraints, holes, and geometry conventions;
- join/cut inclusion thresholds and exact split proportions/boundaries;
- edit-transfer delta convention and substitution-task details;
- model dimensions, codebook counts/sizes, and capacity-matching calculation;
- loss weights, optimizer, schedules, batch size, stopping, seeds, and search budget;
- kernel timeouts, numeric tolerances, equivalence tests, and round-trip acceptance threshold;
- metric weights, matching rules, success tolerances, locality formula, and confidence intervals.

## 16. Claims and interpretation constraints

**[PLAN REQUIREMENT]**

- Revolve support alone is not the novelty.
- Structure and geometry are not assumed statistically independent.
- The models should not be described as perfectly disentangled; incompatible edits and cross-factor leakage must be measured.
- Results from this curated scope do not establish broad CAD generality.
- Additional operation types should not be added until the core benchmark and matched comparisons are complete.

**[PROPOSED V0] Phase-specific claim limits:**

- **Phase 1 required:** `B0` versus `F0` supports a factorization claim, and `F0` versus `A` supports a typed-graph claim in the fully discrete setting.
- **Phase 2 required for the full geometry-ablation claim:** `F1` versus `B` tests the graph effect in the hybrid setting; `A` versus `B`, interpreted together with `F0` versus `F1`, supports the discrete-versus-continuous conditioned-geometry comparison.
- **Optional:** `B1` may strengthen interpretation of continuous mixed representations but is not required for either phase.
- The primary editing benchmark supports a claim about **instruction-conditioned localized executable editing**. Because it supplies a preservation mask, it does not show that a representation automatically infers which fields should remain unchanged.
- Inferred preservation may be claimed only if the separate later benchmark without a preservation mask is implemented and evaluated.

Until the mentor decisions above are resolved and the corresponding artifacts are frozen, all `PROPOSED V0` details are implementation candidates rather than established project facts.
