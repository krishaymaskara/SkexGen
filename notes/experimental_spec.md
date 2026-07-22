# Experimental Specification: Structured Extrude-and-Revolve CAD

## 0. Status and interpretation rules

This document translates `docs/research_plan.md` into a candidate implementation and evaluation contract. It does **not** record completed implementation work or settled architectural decisions.

This specification supersedes `notes/graph_schema.md`, which may be removed separately; it is not removed by this revision because only this file is in scope.

Every normative statement has one of these labels:

- **[PLAN REQUIREMENT]** — stated directly in the research plan.
- **[PROPOSED V0]** — a concrete initial choice needed to make the experiment implementable; it is not yet established.
- **[MENTOR DECISION]** — a choice that materially affects the scientific comparison and should be confirmed before the dataset or model interface is frozen.
- **[LATER IMPLEMENTATION DECISION]** — an engineering or reporting choice that must eventually be frozen, but does not need mentor approval in the next discussion.

If an unlabeled explanation conflicts with a labeled statement, the labeled statement controls. Parameter ranges, token IDs, dataset sizes, split ratios, loss weights, codebook sizes, and neural-network dimensions remain unspecified unless explicitly marked as proposed.

## 1. Experimental objective and controlled scope

**[PLAN REQUIREMENT] Research question:** How should CAD feature structure and geometry be represented to support systematic generalization and localized counterfactual editing across extrude-and-revolve feature histories?

**[PLAN REQUIREMENT] Controlled domain:**

- single-solid CAD histories;
- sketches containing lines, arcs, circles, loops, profiles, and construction axes;
- extrusion and revolution operations;
- join and cut Boolean modes where the data and CAD kernel support them reliably;
- bounded history length and normalized geometric dimensions.

**[PLAN REQUIREMENT] Primary scientific comparison:**

1. a capacity-matched flat/mixed representation;
2. a typed graph with discrete structure and discrete, structure-conditioned geometry;
3. a typed graph with discrete structure and continuous, structure-conditioned geometry.

**[PROPOSED V0] Phased causal controls:** organize implementation into the following evidence levels:

- **Phase 1 — minimum rigorous experiment:** `B0-FLAT-MIXED-VQ`, `F0-FLAT-FACTORED-DISCRETE`, and `A-GRAPH-DISCRETE`. These three models are sufficient to test whether separating structure and geometry helps over a mixed representation and whether typed graph structure helps over a flat factored representation in the fully discrete setting.
- **Phase 2 — full geometry-representation ablation:** `F1-FLAT-FACTORED-HYBRID` and `B-GRAPH-HYBRID`. These are required only for a full discrete-versus-continuous conditioned-geometry claim.
- **Optional:** `B1-FLAT-MIXED-CONT` remains an additional mixed-continuous control if time permits.

Without `F0`, Phase 1 tests a combined graph-and-factorization framework and cannot attribute an improvement specifically to graph structure. Without both Phase 2 models, the study must not make the full discrete-versus-continuous conditioned-geometry claim.

**[PROPOSED V0] Unit of modeling:** one complete, ordered CAD feature history that deterministically reconstructs one final solid. Intermediate states must also execute successfully so that an invalid early feature is not hidden by a later feature.

**[PROPOSED V0] Out of scope for the first frozen experiment:** fillets, chamfers, sweeps, lofts, assemblies, multiple final bodies, free-form splines, and constraint solving beyond what is necessary to execute the recorded sketch geometry.

## 2. Minimal CAD vocabulary

### 2.1 Graph-level node vocabulary

| Node type | Status | Meaning |
|---|---|---|
| `REFERENCE_PLANE` | **[PROPOSED V0]** | A local 2D coordinate frame on which a sketch is placed. The plan names `PLACED_ON` but does not name its target node; this node makes that dependency explicit. |
| `SKETCH` | **[PLAN REQUIREMENT]** | A collection of 2D primitives and loop topology in a reference-plane frame. |
| `PROFILE` | **[PLAN REQUIREMENT]** | One selectable closed sketch region, represented by an outer loop and optional inner loops. |
| `AXIS` | **[PLAN REQUIREMENT]** | A construction axis that can be referenced by a revolution. |
| `EXTRUDE` | **[PLAN REQUIREMENT]** | An extrusion feature consuming one profile. |
| `REVOLVE` | **[PLAN REQUIREMENT]** | A revolution feature consuming one profile and one axis. |

**[PROPOSED V0] Minimality choice:** lines, arcs, circles, and loops are typed subrecords inside a `SKETCH`, not graph nodes. This keeps graph nodes at the feature/reference level while retaining sketch topology in the structural data.

**[MENTOR DECISION]:** Confirm whether sketch primitives and loops should instead be first-class graph nodes. This changes graph size, message passing, tokenization, matching metrics, and the meaning of “node accuracy.”

**[MENTOR DECISION]:** Confirm whether `REFERENCE_PLANE` should be a node. If not, its frame must be embedded as fields of `SKETCH`, and `PLACED_ON` should be removed from the minimal graph vocabulary.

### 2.2 Sketch primitive vocabulary

| Primitive/subrecord | Structural fields | Geometric fields |
|---|---|---|
| `LINE` | primitive ID, type, owning sketch, loop membership, position in loop | 2D start and end points |
| `ARC` | primitive ID, type, owning sketch, loop membership, position in loop, traversal direction | 2D start, midpoint, and end points |
| `CIRCLE` | primitive ID, type, owning sketch, loop membership | 2D center and radius |
| `LOOP` | loop ID, owning sketch, ordered primitive references, outer/inner role | no independent geometry; geometry is induced by its primitives |

All coordinates above are in the owning sketch's local 2D frame.

**[PROPOSED V0] Arc parameterization:** start/mid/end points. It is easy to serialize and determines the circular arc except in degenerate cases.

**[LATER IMPLEMENTATION DECISION]:** Confirm the arc parameterization and whether analytic constraints/dimensions are reconstructed or only evaluated geometry is reconstructed.

### 2.3 Operation vocabulary

| Operation | Required references | Required operation fields |
|---|---|---|
| `EXTRUDE` | one `PROFILE` | distance, direction, Boolean mode |
| `REVOLVE` | one `PROFILE`, one `AXIS` | angle, direction, Boolean mode |

**[PLAN REQUIREMENT] Boolean vocabulary:** `JOIN` and `CUT` where data quality permits.

**[PROPOSED V0] Bootstrap mode:** permit a first operation to use `NEW_BODY` during data conversion even though the research plan only explicitly names join/cut. Alternatively, interpret the first `JOIN` against an empty body as body creation.

**[MENTOR DECISION]:** Decide the first-feature Boolean semantics before freezing the schema.

## 3. Typed graph definition

Let a history be a typed attributed directed graph

\[
G=(V,E,\tau_V,\tau_E,S,X),
\]

where `V` is the node set, `E` is the directed edge set, `tau_V` and `tau_E` are node and edge types, `S` contains structural/categorical fields, and `X` contains numerical geometry.

### 3.1 Proposed edge-direction convention

**[PROPOSED V0] Convention:** dependency edges point from the consuming/dependent node to the resource it needs. Chronology edges point forward in execution order.

Examples:

```text
SKETCH  --PLACED_ON----> REFERENCE_PLANE
PROFILE --CREATED_BY---> SKETCH
AXIS    --CREATED_BY---> SKETCH
EXTRUDE --USES_PROFILE-> PROFILE
REVOLVE --USES_PROFILE-> PROFILE
REVOLVE --USES_AXIS----> AXIS
EXTRUDE --PRECEDES-----> REVOLVE
```

**[MENTOR DECISION]:** The research plan's example draws `Sketch --USES_PROFILE--> Extrude` and `Axis --USES_AXIS--> Revolve`, which conflicts with the consumer-to-resource convention and with the named `PROFILE` node. Confirm one canonical direction before implementing graph extraction. Direction must be identical for every model, metric, and dataset split.

### 3.2 Edge types and cardinalities

| Edge type | Allowed source | Allowed target | Cardinality and meaning |
|---|---|---|---|
| `PLACED_ON` | `SKETCH` | `REFERENCE_PLANE` | Exactly one per sketch under the proposed plane-node design. |
| `CREATED_BY` | `PROFILE`, `AXIS` | `SKETCH` | Exactly one per profile or sketch-owned axis. |
| `USES_PROFILE` | `EXTRUDE`, `REVOLVE` | `PROFILE` | Exactly one per operation in the minimal scope. |
| `USES_AXIS` | `REVOLVE` | `AXIS` | Exactly one per revolve; forbidden on extrude. |
| `PRECEDES` | `EXTRUDE`, `REVOLVE` | `EXTRUDE`, `REVOLVE` | Connects consecutive operation nodes in execution order. |

**[PROPOSED V0] Chronology encoding:** include `PRECEDES` only between consecutive operations, not all transitively ordered pairs. The complete operation order is recovered by walking the chain.

**[PROPOSED V0] Feature/body dependency:** a single current solid is updated in `PRECEDES` order. No `BODY_STATE` or `MODIFIES` edge is included initially.

**[MENTOR DECISION]:** Decide whether chronological order is sufficient for Boolean/body dependency or whether explicit body-state nodes or `MODIFIES` edges are required.

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
- **Required dependencies [PROPOSED V0]:** exactly one `CREATED_BY` edge to the owning `SKETCH`.
- **Validity [PROPOSED V0]:** all loop references exist in that sketch; outer loop is closed and non-self-intersecting; inner loops are closed, lie inside the outer loop, and do not cross one another; enclosed area exceeds tolerance.

### 4.4 `AXIS`

- **Required structural fields [PLAN REQUIREMENT + PROPOSED V0]:** `node_id`, `node_type`, axis source/category if the dataset distinguishes construction-line versus reference-axis origins.
- **Required geometric fields [PLAN REQUIREMENT + PROPOSED V0]:** a local 2D point and normalized 2D direction in the owning sketch frame. The world-space line is derived from the sketch frame.
- **Required dependencies [PROPOSED V0]:** exactly one `CREATED_BY` edge to a `SKETCH`.
- **Validity [PROPOSED V0]:** finite point; nonzero direction normalized within tolerance; owning sketch and plane exist.

### 4.5 `EXTRUDE`

- **Required structural fields [PLAN REQUIREMENT]:** `node_id`, `node_type`, direction category, Boolean mode, profile identity.
- **Required geometric fields [PLAN REQUIREMENT]:** extrusion distance.
- **Required dependencies [PROPOSED V0]:** exactly one `USES_PROFILE` edge; zero or one incoming and zero or one outgoing `PRECEDES` edge, according to position in the operation chain.
- **Validity [PROPOSED V0]:** profile exists and is valid; distance is finite and has magnitude above tolerance; direction and signed-distance convention are not contradictory; Boolean mode is allowed at that history position; execution produces the required single valid solid.

### 4.6 `REVOLVE`

- **Required structural fields [PLAN REQUIREMENT]:** `node_id`, `node_type`, direction category, Boolean mode, profile identity, axis identity.
- **Required geometric fields [PLAN REQUIREMENT]:** revolution angle; axis position/orientation is carried by the referenced `AXIS` geometry.
- **Required dependencies [PROPOSED V0]:** exactly one `USES_PROFILE` edge and one `USES_AXIS` edge; zero or one incoming and zero or one outgoing `PRECEDES` edge.
- **Validity [PROPOSED V0]:** profile and axis exist; both can be transformed into a common frame; angle is finite and nonzero; sweep does not create a kernel-invalid result; Boolean mode is allowed; execution produces the required single valid solid.

**[LATER IMPLEMENTATION DECISION]:** Set legal numeric ranges and conventions for extrusion distance and revolve angle, including whether signed values encode direction and whether full `2π` revolutions are included.

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
- the `PRECEDES` subgraph is one acyclic chain over all operation nodes;
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

**[MENTOR DECISION]:** Confirm the CAD kernel, canonicalization policy, and numeric/geometric tolerances.

## 7. Equivalent flat/mixed representation

The flat baseline must contain the same information as the graph representation. It may change factorization and inductive bias, but not silently omit references or geometry.

### 7.1 Canonical ordering

**[PROPOSED V0]** Serialize in this deterministic order:

1. reference planes, ordered by canonical ID;
2. sketches, profiles, and axes, ordered by first use and then canonical ID;
3. operations in `PRECEDES` order;
4. primitives within a loop in traversal order; loops with the outer loop first, then inner loops by canonical geometric ordering.

IDs are renumbered locally from zero after canonicalization. References are encoded as earlier-record indices, not arbitrary source-system IDs.

### 7.2 Logical grammar

**[PROPOSED V0]** The logical record stream is:

```text
HISTORY_BEGIN
PLANE       <id> <origin> <frame>
SKETCH      <id> <plane_ref> <primitive_count> <loop_count>
LINE        <id> <loop_ref> <order> <start_xy> <end_xy>
ARC         <id> <loop_ref> <order> <start_xy> <mid_xy> <end_xy>
CIRCLE      <id> <loop_ref> <center_xy> <radius>
LOOP        <id> <role> <ordered_primitive_refs>
PROFILE     <id> <sketch_ref> <outer_loop_ref> <inner_loop_refs>
AXIS        <id> <sketch_ref> <point_xy> <direction_xy>
EXTRUDE     <id> <profile_ref> <direction> <boolean_mode> <distance>
REVOLVE     <id> <profile_ref> <axis_ref> <direction> <boolean_mode> <angle>
HISTORY_END
```

This is a semantic grammar, not a frozen token-number assignment. Missing record types are omitted per sample; operation-specific masks determine which fields are legal.

### 7.3 Mixed latent baseline

**[PLAN REQUIREMENT]** The primary flat baseline uses one mixed representation over the serialized history without explicit typed-graph factorization.

**[PROPOSED V0]** Its encoder consumes the structural tokens, references, and geometric values in the canonical stream and produces one mixed VQ latent. Its decoder predicts the same stream. The total latent slots, embedding width, approximate code capacity, decoder size, training examples, and optimization budget must be matched as closely as possible to the graph models and reported explicitly.

**[PROPOSED V0] Phase status:** `B0-FLAT-MIXED-VQ` is the required mixed baseline in Phase 1. `B1-FLAT-MIXED-CONT` is optional and is not required for either phase's principal claim.

## 8. Exact model comparisons

| ID | Input representation | Structure latent | Geometry latent | Decoder target | Required status |
|---|---|---|---|---|---|
| `B0-FLAT-MIXED-VQ` | Canonical flat serialization | Not separated | One mixed discrete VQ latent | Canonical flat stream | **[PLAN REQUIREMENT + PROPOSED V0] Phase 1 required** |
| `F0-FLAT-FACTORED-DISCRETE` | Canonical flat serialization, fields separated by type | Discrete VQ | Discrete VQ conditioned on flat structural latent | Canonical flat stream | **[PROPOSED V0] Phase 1 required** |
| `A-GRAPH-DISCRETE` | Typed graph plus geometry | Discrete VQ | Discrete VQ conditioned on graph/structure | Executable graph/history | **[PLAN REQUIREMENT + PROPOSED V0] Phase 1 required** |
| `F1-FLAT-FACTORED-HYBRID` | Canonical flat serialization, fields separated by type | Discrete VQ | Continuous latent conditioned on flat structural latent | Canonical flat stream | **[PROPOSED V0] Phase 2 required for the full geometry-ablation claim** |
| `B-GRAPH-HYBRID` | Typed graph plus geometry | Discrete VQ | Continuous latent conditioned on structural latent | Executable graph/history | **[PLAN REQUIREMENT + PROPOSED V0] Phase 2 required for the full geometry-ablation claim** |
| `B1-FLAT-MIXED-CONT` | Canonical flat serialization | Not separated | One mixed continuous latent | Canonical flat stream | **[PLAN REQUIREMENT if time permits] Optional** |

### 8.1 Controlled variables

**[PLAN REQUIREMENT]** Baselines must be capacity matched.

**[PROPOSED V0]** All required comparisons use:

- identical train/validation/test manifests;
- identical normalization and canonicalization;
- the same executable target semantics;
- approximately matched encoder/decoder parameter counts, decoder depth/width, latent information budget, optimizer family, update count, batch exposure, early-stopping rule, and hyperparameter-search budget;
- the same number of random seeds, with per-seed and aggregate results;
- teacher-forced reconstruction results and fully autoregressive/fully predicted results reported separately.

Exact architecture dimensions, VQ codebook counts/sizes, optimizer settings, and tolerances are **[LATER IMPLEMENTATION DECISION]** items after the data audit and compute budget are known.

### 8.2 Claims supported by the required comparisons

**[PROPOSED V0]** Compare models in matched pairs:

- In **Phase 1**, `B0-FLAT-MIXED-VQ` versus `F0-FLAT-FACTORED-DISCRETE` tests factorization: the effect of separating structure and geometry while keeping the representation flat.
- In **Phase 1**, `F0-FLAT-FACTORED-DISCRETE` versus `A-GRAPH-DISCRETE` tests typed graph structure with discrete geometry.
- In **Phase 2**, `F1-FLAT-FACTORED-HYBRID` versus `B-GRAPH-HYBRID` tests typed graph structure in the hybrid setting.
- The cross-phase comparison `A-GRAPH-DISCRETE` versus `B-GRAPH-HYBRID`, interpreted together with `F0-FLAT-FACTORED-DISCRETE` versus `F1-FLAT-FACTORED-HYBRID`, tests discrete versus continuous conditioned geometry without confusing that change with graph structure.

If Phase 1 omits `F0`, all graph-specific claims must be withdrawn. If Phase 2 is not completed, the paper may report only the Phase 1 factorization and graph results and must not claim a complete discrete-versus-continuous geometry ablation.

### 8.3 Graph encoder and decoder ordering

**[PLAN REQUIREMENT]** Candidate structural encoders are a typed GNN, graph Transformer, or typed-edge/sequence Transformer. Geometry must be conditioned on structure.

**[PROPOSED V0]** Decode in two stages:

1. predict node types, edge types, operation order, references, loop topology, and Boolean modes;
2. predict only geometry allowed by that decoded structure, using operation/type masks.

During early training, geometry may be conditioned on ground-truth structure. Final evaluation must also condition on predicted structure.

**[LATER IMPLEMENTATION DECISION]:** Choose the graph encoder and decide whether latents are global per history, local per feature, or hierarchical, after graph granularity is confirmed.

## 9. Training protocol and order

**[PLAN REQUIREMENT + PROPOSED V0] Training order:**

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

**[PROPOSED V0]** Split by source history or procedural template family, not by individual serialization. All augmented variants and counterfactual relatives of one source remain in one partition.

**[LATER IMPLEMENTATION DECISION]:** Set train/validation/test proportions after the data source and usable sample count are known.

### 10.3 Unseen-combination split

**[PLAN REQUIREMENT]** Withhold selected dependency compositions while ensuring that every individual operation and component occurs in training.

**[PROPOSED V0]** Assign a canonical operation/dependency signature to every history, for example operation types plus profile/axis reuse and Boolean modes. Select one or more complete signatures—such as a particular extrude-then-revolve dependency pattern—for test only. Verify and report that all atomic node types, edge types, modes, and parameter bins in test also occur in train.

**[LATER IMPLEMENTATION DECISION]:** After the mentor identifies the most important generalization benchmark, choose its exact withheld signatures from the audited frequency table; do not choose them after observing model results.

### 10.4 Length-generalization split

**[PLAN REQUIREMENT]** Train primarily on shorter histories and test longer ones.

**[PROPOSED V0]** Candidate split: train/validate on one-to-three operation histories and test on four-operation histories, matching the example in the plan. If data are too sparse, freeze another nonoverlapping length boundary before training.

**[LATER IMPLEMENTATION DECISION]:** Confirm what “feature count” counts—only extrude/revolve operations or all graph nodes—and freeze the length boundary after the data audit.

### 10.5 Parameter interpolation and extrapolation splits

**[PLAN REQUIREMENT]** Evaluate extrusion distance, revolve angle, sketch dimensions, and placements both in range and out of range.

**[PROPOSED V0]** For each target parameter family:

- `INTERPOLATION`: hold out interior bins while retaining lower and higher bins in training;
- `EXTRAPOLATION-LOW`: test below the frozen training interval;
- `EXTRAPOLATION-HIGH`: test above the frozen training interval;
- keep structure distributions as similar as practical and report remaining imbalance.

Thresholds are computed from a pre-model data audit and then frozen. Normalization must not leak held-out extrema into training unless a fixed physical normalization is chosen in advance.

**[LATER IMPLEMENTATION DECISION]:** Confirm physical ranges, binning, whether low extrapolation is meaningful for each positive parameter, and the normalization policy.

### 10.6 Counterfactual-pair split

**[PROPOSED V0]** A source, all of its one-edit variants, and any transferred-edit partner group must stay in one partition. If procedural generators are used, generator template and random seed families must not cross partitions in a way that duplicates geometry.

**[MENTOR DECISION]:** Confirm whether the main dataset is real, procedural, or a procedural training set plus real-data validation.

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

## 15. Immediate decisions for the next mentor discussion

Only the following scope-defining decisions are prioritized for immediate mentor confirmation:

1. **Graph granularity:** primitives/loops as sketch subrecords versus first-class graph nodes.
2. **Reference-plane representation:** add `REFERENCE_PLANE` nodes versus store the frame on `SKETCH`; consequently, whether `PLACED_ON` remains an edge.
3. **Edge direction:** canonical direction of `USES_PROFILE`, `USES_AXIS`, and `CREATED_BY`, given the inconsistent example in the plan.
4. **Body and first-operation semantics:** chronological `PRECEDES` chain versus explicit body dependency, and `NEW_BODY` versus join-on-empty or another bootstrap convention.
5. **Data source and CAD kernel:** real, procedural, or mixed data; the kernel used for conversion, round-trip execution, and validity.
6. **Required baseline set and phases:** approve `B0`/`F0`/`A` as the Phase 1 minimum and `F1`/`B` only when pursuing the full geometry-ablation claim.
7. **Editing mechanism:** approve the explicit-request, one-forward-pass preservation-mask protocol as instruction-conditioned localized executable editing, with controlled-structure and end-to-end conditions; latent replacement, latent optimization, and inferred-preservation editing remain later alternatives.
8. **Most important generalization benchmark:** unseen compositions, history length, parameter extrapolation, or localized edit transfer. The chosen priority determines dataset and split design.

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
