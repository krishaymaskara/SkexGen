# Approved Three-Week Research Plan

*Structured CAD Representations for Systematic Generalization and Localized Editing*

| Prepared for | Project context |
|---|---|
| Krishay Maskara | Allen-Blanchette Group, Princeton MAE |
| Active time horizon | Three weeks |
| Controlled operation setting | Sketch + extrude + revolve |
| Primary target | One fair, interpretable flat-versus-graph result and a concise technical report or presentation |
| Scope revision | Approved July 30, 2026 |

*Original six-to-eight-week plan revised July 16, 2026; active scope revised
July 30, 2026*

## Document role

This document records both the approved active scope and the broader original
roadmap as historical planning context. It governs future scope, not past
experimental history. Completed results and run records remain authoritative
in [the current project status](status.md) and `docs/experiments/`.

The July 30 revision resolves a conflict with the older roadmap: the original
plan made a fully-discrete-versus-continuous graph comparison central and
treated a same-representation flat-versus-graph ablation as stretch work. With
approximately three weeks remaining, the approved active plan instead makes
one same-decoder flat-versus-graph comparison central and defers the complete
discrete-versus-continuous comparison. The older roadmap is retained below
rather than rewritten as if it never governed planning.

## Active three-week scope

### Core research question

**Does a typed dependency-graph representation improve systematic
generalization and localized editing compared with a flat chronological
representation when both use the same category-conditioned constrained
continuous-geometry decoder?**

The controlled sketch/extrude/revolve domain isolates the effect of explicit
typed dependencies. The principal independent variable is flat chronological
versus typed dependency-graph representation. Both models must use the same
profile-family-aware continuous-geometry decoder interface so that geometry
parameterization is not a confound.

### Required core deliverables

- A shared category-conditioned constrained profile decoder.
- One repaired flat baseline using that decoder.
- One minimal typed dependency-graph model using the same decoder.
- One capacity-conscious flat-versus-graph comparison.
- Ordinary validation, one systematic-generalization split, and one localized
  numerical-edit evaluation.
- A concise technical report or presentation explaining the result and
  limitations.

The existing epoch-44 flat checkpoint remains the frozen, reproducible failed
unconstrained baseline. The checked-in deterministic profile-geometry
contract is a prerequisite for the shared neural decoder, not evidence that
the neural constrained decoder or a successor flat checkpoint is complete.
Current implementation and experiment state is maintained in
[the status page](status.md).

### Three-week schedule

#### Week 1 — Freeze the shared decoder and repair the flat model

- Integrate explicit profile-family prediction and compact continuous profile
  parameters with the deterministic profile-geometry contract.
- Freeze the shared category-conditioned constrained decoder interface,
  targets, losses, and validity checks.
- Timebox training and validation of one repaired flat chronological model.
- Preserve the epoch-44 checkpoint as the failed unconstrained reference
  rather than overwriting its evidence.

**Gate:** the shared neural decoder is implemented and the repaired flat model
has one reproducible checkpoint suitable for the planned evaluations.

#### Week 2 — Implement the minimal typed graph model

- Begin graph implementation as soon as the shared decoder interface is
  frozen.
- Implement one minimal typed dependency-graph encoder using the existing
  controlled schema and the same decoder.
- Keep model capacity and training opportunity sufficiently comparable to make
  the representation contrast interpretable.
- Avoid adding alternate graph architectures or a fully discrete geometry
  path.

**Gate:** one graph checkpoint and one repaired-flat checkpoint can be
evaluated through the same decoder and evaluation interfaces.

#### Week 3 — Run the minimal comparison and report it

- Evaluate both models on ordinary validation.
- Evaluate both models on one predetermined systematic-generalization split.
- Evaluate both models on one localized numerical-edit test with target-change
  and preservation measures.
- Report capacity, validity, failures, limitations, and any inconclusive or
  negative result.
- Produce a concise technical report or presentation.

**Success criterion:** one fair, interpretable repaired-flat-versus-typed-graph
comparison using the shared decoder on ordinary validation, one systematic
split, and one localized-edit test.

### Stretch and future work

The following work is outside the required three-week core:

- A graph model with fully discrete geometry.
- The full discrete-versus-continuous geometry comparison.
- Multiple graph architectures.
- Broad evaluation across every systematic split.
- Extensive multi-seed experiments.
- Real-CAD validation.
- Additional operations such as fillet.
- Language conditioning.
- Publication preparation.

Experiments may expand only after the minimal comparison is complete. Any
stretch result must be labeled as such and must not delay the required
comparison or report.

### Active claims and limitations

- The project may test whether typed dependency structure helps under a shared
  decoder; it may not attribute differences to geometry representation because
  that condition is held fixed.
- A result on the controlled corpus does not establish broad real-CAD
  generality.
- One systematic split, one localized-edit test, and limited seeds constrain
  the breadth and statistical strength of conclusions.
- The deterministic profile-geometry contract is implemented; the neural
  constrained decoder, successor flat checkpoint, and graph model are not yet
  implemented or trained.

## Historical six-to-eight-week planning context

Everything from this heading through the original execution pointer records
the broader July 16 plan. It is retained to explain earlier architectural
choices and completed work, but it is not the active schedule or deliverable
set. Where it conflicts with the active three-week scope above, the active
scope governs future work.

## 1. Revised Project Thesis

**Central research question:** How should CAD feature structure and geometry be represented to support systematic generalization and localized counterfactual editing across extrude-and-revolve feature histories?

**Working hypothesis:** A typed dependency-graph representation with discrete structural codes and structure-conditioned geometry will support more reliable compositional generalization and localized edits than flat or fully mixed representations. A hybrid model with discrete structure and continuous geometry may outperform a fully discrete model on parameter extrapolation and fine-grained editing.

## 2. What Changed from the Previous Plan

- Revolve remains the controlled extension, but adding an operation is no longer the main contribution.
- The CAD plan is represented as a typed dependency graph rather than only as a flat sequence.
- Geometry is modeled conditionally on feature structure instead of being assumed fully independent.
- The central architecture comparison becomes fully discrete geometry versus discrete structure with continuous geometry.
- The main evaluation shifts from generic code swapping to a counterfactual benchmark measuring unseen combinations, length extrapolation, parameter extrapolation, and localized edit transfer.
- Supporting additional operations such as fillet is deferred unless the representation and benchmark are already complete.

## 3. Executive Summary

This project studies the internal representation of editable CAD histories. The experimental domain is deliberately restricted to sketches, extrusions, and revolutions so that the research can isolate representation and generalization rather than become an operation-engineering project. Each CAD model is represented as a typed feature-dependency graph. Structural information includes feature types, typed dependencies, profile and axis references, Boolean relationships, and sketch topology. Geometric information includes coordinates, dimensions, extrusion distances, axis placement, and revolve angles.

Two principal models will be compared. The first uses discrete vector-quantized representations for both structure and geometry. The second uses discrete structural codes but retains geometry in a continuous latent space conditioned on the structure. A capacity-matched flat or mixed representation will serve as the primary baseline. The strongest contribution would be a benchmark showing which representation best supports systematic generalization and localized counterfactual editing while preserving executable CAD histories.

A successful six-week result is a functioning extrude-and-revolve dataset and execution pipeline, matched representation baselines, and quantitative results on reconstruction, executability, unseen combinations, parameter extrapolation, and edit locality. Weeks seven and eight are reserved for stronger ablations, real-data validation, or paper preparation—not automatically for adding more operations.

## 4. Research Objectives and Hypotheses

### 4.1 Primary objective

Determine how feature structure and geometry should be represented so that a CAD model can generalize compositionally and accept localized edits without unintentionally changing unrelated parts of the construction history.

### 4.2 Testable hypotheses

- **H1 - Graph structure:** A typed dependency graph will preserve profile, axis, and feature relationships more reliably than a flat feature sequence on unseen combinations and longer histories.
- **H2 - Conditional geometry:** Modeling geometry as conditional on feature structure will improve validity and reduce incompatible parameter predictions.
- **H3 - Hybrid representation:** Discrete structure with continuous geometry will outperform fully discrete geometry on parameter extrapolation and small localized numerical edits.
- **H4 - Fully discrete representation:** Fully discrete structure and geometry may provide stronger categorical recombination and code-level editability, but may lose numerical precision or extrapolation ability.
- **H5 - Counterfactual locality:** A structured representation will perform requested edits while preserving unaffected feature nodes, dependency edges, and geometric parameters more accurately than mixed baselines.

### 4.3 Claims to avoid

- Do not present revolve support itself as the novelty.
- Do not claim that structure and geometry are statistically independent; the architecture explicitly treats geometry as structure-conditioned.
- Do not claim perfect disentanglement; report cross-factor leakage and incompatible edits directly.
- Do not claim broad CAD generality from a curated extrude-and-revolve subset.
- Do not add a new operation unless the core benchmark and matched model comparison are already complete.

## 5. Core Project Scope

### 5.1 Supported feature vocabulary

- Sketch entities: line, arc, circle, loop, profile, and construction axis.
- Feature operations: extrude and revolve, including join and cut where data quality permits.
- Revolve attributes: profile reference, axis reference, angle, direction, and Boolean mode.
- Extrude attributes: profile reference, distance, direction, and Boolean mode.
- Single-solid histories with bounded feature count and normalized dimensions.

### 5.2 Core deliverables

- A typed dependency-graph schema for extrude-and-revolve histories.
- A versioned dataset with ordinary and systematic-generalization splits.
- A flat or mixed representation baseline.
- A fully discrete graph-structured model.
- A discrete-structure/continuous-geometry model.
- A counterfactual evaluation suite for localized editing and generalization.
- An executable-history checker and structured failure analysis.

### 5.3 Stretch work

- Validate the conclusions on a small real-CAD subset if the core experiments use procedural data.
- Add a graph-versus-flat ablation with the same latent representation.
- Introduce feature-local latent variables instead of only model-level latents.
- Add a restricted fillet experiment only if the paper results are already complete and at least two weeks remain.

## 6. Data Representation

### 6.1 Typed dependency graph

Represent each model as a graph $G = (V, E)$. Nodes encode typed CAD entities or features. Edges encode typed dependencies rather than only chronological order.

| Graph element | Examples |
|---|---|
| Node types | `SKETCH`, `PROFILE`, `AXIS`, `EXTRUDE`, `REVOLVE` |
| Dependency edges | `USES_PROFILE`, `USES_AXIS`, `DEPENDS_ON`, `DEFINED_IN`, `PLACED_ON` |
| Structural node attributes | Feature type, Boolean mode, primitive type, loop membership |
| Geometric node attributes | Coordinates, dimensions, distance, angle, position, orientation |

```text
Sketch_1  --PLACED_ON----> Reference_Plane_1
Profile_1 --DEFINED_IN---> Sketch_1
Extrude_1 --USES_PROFILE-> Profile_1
Sketch_2  --PLACED_ON----> Reference_Plane_1
Profile_2 --DEFINED_IN---> Sketch_2
Axis_2    --DEFINED_IN---> Sketch_2
Revolve_1 --USES_PROFILE-> Profile_2
Revolve_1 --USES_AXIS----> Axis_2
Revolve_1 --DEPENDS_ON---> Extrude_1
```

Chronological order can still be retained as a typed edge or positional feature, but it should not be the only representation of dependency.
The implemented controlled schema records chronology in an authoritative
`operation_sequence` and uses backward `DEPENDS_ON` edges for body-state
dependencies. See
[ADR-0001](decisions/ADR-0001-controlled-cad-schema-v1.md).

### 6.2 Structural versus geometric information

| Attribute | Primary representation |
|---|---|
| Operation type | Structure |
| Feature order and dependency type | Structure |
| Profile or axis identity | Structure |
| Boolean join/cut mode | Structure |
| Sketch primitive count and loop topology | Structure |
| Sketch coordinates and dimensions | Geometry |
| Extrusion distance | Geometry |
| Revolve angle | Geometry |
| Axis position and orientation | Geometry conditioned on structural axis identity |
| Profile shape compatibility | Joint structural-geometric constraint |

Ambiguous attributes should be documented explicitly. The goal is not to force complete independence, but to separate reusable structural relationships from their numerical realization while allowing geometry to depend on structure.

## 7. Proposed Model Family

### 7.1 Shared structural encoder

Encode the typed dependency graph using a graph neural network, graph Transformer, or typed-edge Transformer. The encoder outputs a structural representation that is quantized into discrete codes:

```text
z_structure = Q_structure(E_graph(G_structure))
```

The structural codes should summarize feature types, typed dependencies, sketch topology, references, and Boolean relationships.

### 7.2 Model A: fully discrete structure and geometry

```text
z_structure = Q_structure(E_structure(G))
z_geometry  = Q_geometry(E_geometry(X | G))
history_hat = D(z_structure, z_geometry)
```

This is the closest extension of SkexGen. Both structural and geometric information are compressed into learned codebooks. It may support categorical recombination and discrete editing, but quantization may reduce numerical precision and extrapolation.

### 7.3 Model B: discrete structure and continuous conditional geometry

```text
z_structure = Q_structure(E_structure(G))
z_geometry  = E_geometry(X | z_structure)
history_hat = D(z_structure, z_geometry)
```

The structure remains discrete, while geometry is encoded continuously and conditioned on the structural representation. This model is expected to better support small parameter changes, interpolation, and extrapolation beyond training ranges.

### 7.4 Primary baseline

Use a capacity-matched flat or mixed representation that receives the serialized feature history without explicit typed graph factorization. The strongest baseline is one mixed VQ representation with the same approximate latent capacity and decoder size. If time permits, add a continuous mixed latent baseline.

### 7.5 Decoder and executability

The decoder reconstructs an executable feature history. It predicts structure first or conditions each geometric prediction on the decoded structure. Operation-specific masks prevent impossible parameter types. Ground-truth structure can be used during early training before evaluating fully predicted histories.

## 8. Training Objectives

- Structural reconstruction loss: node types, edge types, feature order, references, and Boolean modes.
- Geometry reconstruction loss: coordinates, extrusion distances, axis geometry, and revolve angles.
- VQ codebook and commitment losses for every discrete latent group.
- Execution-validity evaluation after decoding; kernel failures are recorded even if they are not directly differentiable.
- Optional locality consistency loss: a counterfactual edit should change targeted attributes while preserving designated unaffected attributes.
- Optional compatibility loss: penalize geometry that is invalid for the decoded feature structure.

Teacher forcing should be used for initial reconstruction training. More complex objectives should be introduced only after the matched baseline and both principal model variants train reliably.

## 9. Counterfactual and Systematic-Generalization Benchmark

### 9.1 Unseen operation combinations

Withhold selected combinations while ensuring that each individual operation and component appears during training. Example: train on isolated extrude and revolve patterns but test on a specific extrude-then-revolve dependency pattern not observed during training.

### 9.2 Length generalization

Train primarily on short histories and evaluate on longer histories. The exact split should reflect data availability, for example training on one to three features and testing on four-feature histories.

### 9.3 Parameter interpolation and extrapolation

Create in-range and out-of-range tests for extrusion distances, revolve angles, dimensions, and placements. Continuous geometry is expected to have an advantage on small edits and parameter extrapolation; this must be measured rather than assumed.

### 9.4 Localized counterfactual edits

- Change one extrusion distance while preserving all other features and parameters.
- Change one revolve angle while preserving the profile, axis, earlier features, and Boolean mode.
- Transfer a numerical edit from one model to a structurally compatible model.
- Substitute extrude with revolve only in restricted cases where a valid profile and axis are available.

### 9.5 Counterfactual metrics

| Metric | What it measures |
|---|---|
| Edit success | Whether the requested target attribute changed correctly |
| Structural preservation | Whether unaffected feature nodes and typed edges remain unchanged |
| Geometric preservation | Whether unrelated dimensions and placements remain stable |
| Execution validity | Whether the edited history executes and produces a valid solid |
| Locality score | Targeted change divided by unintended change outside the edit region |
| Generalization accuracy | Performance on withheld combinations, lengths, or parameter ranges |

## 10. Main Experimental Comparisons

| Model | Structure | Geometry | Purpose |
|---|---|---|---|
| Baseline | Flat or mixed | Mixed discrete or continuous | Tests whether explicit graph factorization is necessary |
| Model A | Typed graph, discrete | Discrete and structure-conditioned | Tests the full SkexGen-style discrete extension |
| Model B | Typed graph, discrete | Continuous and structure-conditioned | Tests whether geometry should remain continuous |

### 10.1 Required metrics

- Node-type, edge-type, operation-sequence, profile-reference, and axis-reference accuracy.
- Parameter error for coordinates, distances, positions, orientations, and angles.
- Valid-sequence and valid-solid execution rates.
- Codebook utilization, perplexity, and dead-code rate for discrete models.
- Counterfactual edit success and preservation metrics.
- Performance on unseen combinations, longer histories, and parameter extrapolation.

### 10.2 Minimum convincing result

The proposed models do not need to dominate ordinary reconstruction. A convincing result would show comparable reconstruction and validity but substantially better systematic generalization or localized editing. The discrete-continuous comparison should reveal a consistent tradeoff: for example, discrete geometry may improve categorical recombination while continuous geometry improves precision and extrapolation.

## 11. Original Eight-Week Execution Plan

This schedule preserves the intended dependency order and acceptance logic.
Calendar-relative wording below is historical planning context; it must not be
used to infer current progress.

### Week 1 - Baseline, literature, and representation specification

- Run or inspect the existing SkexGen pipeline and trace topology, geometry, and extrusion representations.
- Confirm Linux/NVIDIA, storage, CAD-kernel, and dataset access.
- Define the first typed dependency-graph schema for sketch, profile, axis, extrude, and revolve nodes.
- Create an attribute table assigning every field to structure, geometry, or a joint constraint.
- Identify the source of extrude-and-revolve histories or commit to procedural generation.
- Write the benchmark specification before model development: unseen combinations, length split, parameter ranges, and edit pairs.

**Checkpoint:** Deliverable: repository map, data feasibility memo, graph schema v1, benchmark specification, and go/no-go decision.

### Week 2 - Dataset, graph extraction, and execution pipeline

- Build or adapt the parser that converts each history into the typed graph and geometric attribute tensors.
- Implement deterministic serialization so the graph can be decoded back into an executable sequence.
- Generate or extract counterfactual pairs with one controlled change.
- Create ordinary random splits plus systematic-generalization splits.
- Implement a kernel-based validity checker and structured failure categories.
- Audit distributions of feature counts, dependency types, angles, distances, and parameter ranges.

**Checkpoint:** Deliverable: frozen dataset v1, graph examples, counterfactual pair set, split manifest, and execution report.

### Week 3 - Flat or mixed baseline

- Implement the capacity-matched flat/mixed baseline first.
- Train on a small subset, confirm loss decreases, and verify executable reconstructions.
- Establish reproducible configuration files, checkpointing, logging, and evaluation scripts.
- Measure initial reconstruction, parameter error, validity, and systematic split performance.
- Correct data or tokenization problems before adding the graph models.

**Checkpoint:** Decision gate: the baseline reconstructs both extrude and revolve examples and the evaluation pipeline is automatic.

### Week 4 - Typed graph plus fully discrete geometry

- Implement the typed graph structural encoder and structural VQ codebook.
- Implement the discrete geometry encoder conditioned on structural features.
- Use a shared decoder or matched decoder architecture to reconstruct executable histories.
- Monitor codebook collapse, utilization, and leakage.
- Run the first graph-versus-flat and discrete-model comparison.

**Checkpoint:** Deliverable: Model A checkpoint, reconstruction table, validity table, and codebook diagnostics.

### Week 5 - Typed graph plus continuous conditional geometry

- Replace the geometry codebook with a continuous structure-conditioned geometry latent while preserving the structural encoder and decoder capacity as closely as possible.
- Train and tune only the minimum changes needed for a fair comparison.
- Evaluate reconstruction precision, interpolation, parameter extrapolation, and small localized edits.
- Compare failure types between fully discrete and hybrid models.

**Checkpoint:** Deliverable: Model B checkpoint and the central discrete-versus-continuous geometry comparison.

### Week 6 - Counterfactual and systematic-generalization experiments

- Run all frozen benchmark splits: unseen combinations, length generalization, interpolation, and extrapolation.
- Run localized edit tests and calculate target-change and preservation metrics.
- Repeat the most important experiments across multiple seeds if compute permits.
- Conduct leakage or probe analysis to measure what each representation contains.
- Create failure galleries and identify whether errors arise from structure, geometry, reference handling, or kernel execution.

**Checkpoint:** Deliverable: complete evidence needed to answer the research question and a draft result table set.

### Week 7 - Strengthening the contribution

- Select one improvement based on Week 6 evidence rather than beginning several new tasks.
- Preferred options: graph-versus-flat ablation, real-data validation, feature-local latent variables, stronger counterfactual pairs, or a locality-consistency objective.
- Add restricted fillet support only if the full core manuscript and results are already stable.
- Run capacity and codebook-size controls needed to rule out trivial explanations.

**Checkpoint:** Deliverable: one high-value extension or ablation that strengthens the central claim.

### Week 8 - Paper, reproducibility, and release preparation

- Write the method, dataset, benchmark, results, limitations, and future-work sections.
- Create the architecture figure, graph schema figure, counterfactual examples, and failure-analysis figure.
- Freeze experiment configurations and clean the repository.
- Prepare a six-to-eight-page workshop manuscript and an expanded arXiv draft if results justify it.
- Obtain mentor and professor approval before public release or submission.

**Checkpoint:** Deliverable: reproducible code package, final tables and figures, and a complete manuscript draft.

## 12. Compressed Six-Week Version

| Week | Priority |
|---|---|
| 1 | Baseline reproduction, graph schema, dataset decision, benchmark specification |
| 2 | Dataset, graph conversion, counterfactual pairs, splits, and validity checker |
| 3 | Flat/mixed baseline and reproducible experiment pipeline |
| 4 | Fully discrete graph model |
| 5 | Hybrid discrete-structure/continuous-geometry model |
| 6 | Systematic-generalization and localized-edit evaluation; paper-quality analysis |

In the six-week version, do not add fillet, language conditioning, or a code prior. Weeks seven and eight are valuable primarily for stronger evidence and writing.

## 13. Decision Gates

### End of Week 1

- Proceed only if a credible data source and executable graph conversion path exist.
- Reduce the graph schema if references or placements cannot be recovered reliably.

### End of Week 2

- Proceed only if ground-truth histories round-trip through the representation and execute successfully.
- Use procedural data for the controlled benchmark if real histories are too inconsistent.

### End of Week 4

- Proceed to the hybrid model only after the baseline and fully discrete model train reproducibly.
- If the graph model fails, determine whether the failure is graph encoding, decoding, or data alignment before adding objectives.

### End of Week 6

- A positive result requires a meaningful advantage on generalization or edit locality—not merely attractive reconstructions.
- A mixed result should motivate one precise Week 7 ablation or objective.
- A negative result can still support a rigorous paper if the benchmark clearly shows where discrete factorization fails.

## 14. Primary Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Insufficient revolve histories | Procedurally generate controlled histories; validate later on a real subset. |
| Graph schema becomes too complex | Restrict node and edge types; preserve only dependencies required for extrude and revolve. |
| Structure and geometry remain entangled | Treat geometry as conditional; report leakage; test hybrid continuous geometry. |
| Continuous model has unfairly higher effective capacity | Match encoder/decoder size and latent dimension; report parameter counts. |
| Counterfactual edits are ambiguous | Use automatically generated pairs with exactly one known change and explicit preservation targets. |
| Kernel failures obscure model quality | Classify parse, reference, sketch, Boolean, and geometry failures separately. |
| Too many experiments for six weeks | Prioritize the three-model comparison and four benchmark categories; defer new operations. |

## 15. Publication Positioning

The strongest paper claim is not that SkexGen can be extended to revolve. It is that CAD structure and geometry require different representational treatment, and that the choice between discrete and continuous geometry materially affects systematic generalization and localized editing.

- **Potential workshop contribution:** a controlled empirical study comparing graph-structured fully discrete and hybrid discrete-continuous CAD representations.
- **Potential benchmark contribution:** a counterfactual suite for unseen operation combinations, history length, parameter extrapolation, and localized edit transfer.
- **Potential full-conference extension:** richer reference-dependent operations, dynamic B-rep grounding, larger real-world histories, and language-conditioned counterfactual editing.

A strong summer result can naturally expand into a larger paper if the benchmark demonstrates a clear representation tradeoff and the graph-structured model yields more local, valid, and systematic edits. The next-stage work should deepen the representation and evaluation before simply expanding the operation vocabulary.

## 16. Questions to Resolve with the Mentor

- Should the typed graph include sketch primitives as nodes, or should each sketch remain a structured sub-sequence attached to a feature node?
- Which graph encoder is most compatible with the group codebase: typed GNN, graph Transformer, or sequence Transformer with explicit relation tokens?
- Should the primary contribution emphasize the representation comparison, the counterfactual benchmark, or both?
- Can the group provide real extrude-and-revolve histories, or should the controlled benchmark begin procedurally?
- Which generalization test is most important for the lab: unseen combinations, length extrapolation, parameter extrapolation, or localized edit transfer?
- What minimum baseline set would the professor consider sufficient for a workshop paper?
- Is feature-local geometry preferred over one global geometry latent for the first implementation?

## 17. Scientific Claim Checklist

This checklist records prerequisites for future claims, not implementation
progress. Current completion state belongs in `docs/status.md`.

### Before claiming systematic generalization

- [ ] Every withheld component appears individually during training.
- [ ] Ordinary in-distribution and systematic splits are both reported.
- [ ] Execution validity is reported alongside token or graph accuracy.
- [ ] Multiple seeds or confidence intervals are used for the main comparison when feasible.

### Before claiming localized editing

- [ ] Each test pair specifies the exact target edit and unaffected attributes.
- [ ] Target-change and preservation metrics are reported separately.
- [ ] Hand-selected visual examples are supplemented with aggregate results.
- [ ] Invalid and non-local edits are included in the failure analysis.

## 18. Current execution pointer

Do not derive the immediate next action from this original roadmap. Use
[the current project status](status.md), which is updated whenever the
scientific gate or authoritative checkpoint changes.
