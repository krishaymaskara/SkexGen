# ADR-0002: Three-week flat-versus-graph research scope

- Status: `accepted`
- Decision date: `2026-07-30`
- Owners/reviewers: project research team
- Supersedes: the active scope and schedule in the July 16 six-to-eight-week research plan; that plan remains historical context
- Superseded by: none

## Context

Approximately three weeks remain in the project. The original model family is
too broad for that interval: it called for a flat baseline, a typed graph with
fully discrete geometry, a typed graph with continuous conditional geometry,
multiple systematic evaluations, stronger ablations, and publication-oriented
work.

Repository diagnostics also revealed a prerequisite that the original roadmap
did not isolate. The frozen epoch-44 flat checkpoint predicts incompatible
profile-family categories and unconstrained profile geometry. Validation-only
categorical isolation showed that authoritative profile family plus
category-conditioned projection improved controlled validity, whereas
reference-plane correction did not. The checked-in deterministic
profile-geometry contract now defines valid controlled-profile construction,
but no neural constrained decoder or successor checkpoint exists.

The approved scope conflicts with the older active roadmap in two places: the
principal comparison changes from fully-discrete-versus-continuous graph
geometry to flat-versus-graph representation, and the same-decoder comparison
changes from stretch work to the core experiment. Completed scientific
results, failures, checkpoint identities, partition-use statements, and
experiment records are unaffected. Related documents are:

- [Active research plan](../research_plan.md)
- [Current project status](../status.md)
- [Categorical-isolation replay](../experiments/b0_phase_b_categorical_isolation_replay.md)
- [Controlled CAD schema](ADR-0001-controlled-cad-schema-v1.md)

## Decision

Prioritize one fair, interpretable comparison between:

1. one repaired flat chronological model; and
2. one minimal typed dependency-graph model.

Both models must use the same category-conditioned constrained
continuous-geometry decoder. The decoder must explicitly predict profile
family and compact continuous profile parameters, with the family selecting
the applicable deterministic geometry construction. Repairing the
profile-family/geometry inconsistency revealed by the baseline diagnostics is
a prerequisite to both model conditions.

The repaired flat effort is timeboxed. Graph implementation begins once the
shared decoder interface is frozen. Capacity, training opportunity, and
evaluation treatment must be controlled sufficiently to keep the
representation comparison interpretable.

The minimum success criterion is completion of one comparison on:

- ordinary validation;
- one predetermined systematic-generalization split; and
- one localized numerical-edit test with target-change and preservation
  measures.

Experiments may expand only after this minimal comparison and its concise
technical report or presentation are complete.

## Alternatives considered

### Retain the original three-model comparison

This would preserve the planned fully-discrete-versus-continuous geometry
study, but it would spread the remaining time across two graph geometry
variants before establishing the more basic representation comparison. It is
not selected for the three-week core.

### Repair and optimize only the flat model

This would address the diagnosed decoder inconsistency with lower
implementation risk, but it would not answer whether typed dependency graphs
improve generalization or edit locality. The flat repair remains necessary,
but it is a shared prerequisite rather than the final research objective.

### Compare flat and graph models with different decoders

This could reuse model-specific implementations, but decoder differences would
confound the effect attributed to representation. It is rejected for the
principal comparison.

## Consequences

### Positive

- The project isolates one principal independent variable: flat chronology
  versus typed dependency structure.
- A shared constrained decoder addresses the observed baseline inconsistency
  without favoring either representation.
- The reduced evaluation set is feasible within the remaining interval and
  still covers ordinary performance, systematic generalization, and edit
  locality.
- Negative or inconclusive results remain interpretable if capacity and
  training controls are reported.

### Tradeoffs and limitations

- A graph model with fully discrete geometry is deferred.
- The complete discrete-versus-continuous geometry comparison is deferred.
- Multiple graph architectures, broad coverage of every systematic split,
  extensive multi-seed experiments, real-CAD validation, fillet and other
  operations, language conditioning, and publication preparation are outside
  the required scope.
- Limited splits and seeds constrain statistical breadth.
- Results in the controlled sketch/extrude/revolve corpus cannot establish
  broad real-CAD generality.
- Holding the decoder fixed supports a representation claim, not a claim
  comparing continuous and discrete geometry.

## Validation and evidence

The scope decision is based on the approved three-week plan and the repository
evidence chain through the immutable categorical-isolation replay. That replay
supports the profile-family/geometry repair prerequisite; it does not prove
that the neural constrained decoder or either successor model works.

Completion requires durable experiment records for the successor checkpoints
and evaluations. The held-out test partition remains untouched unless a later
reviewed protocol explicitly authorizes its first use.

## Implementation contract

This ADR changes planning and future experimental priority only. It does not
modify the controlled schema, source code, datasets, model architecture,
completed artifacts, or immutable experiment records.

Future neural work must:

- reuse one frozen category-conditioned constrained continuous-geometry
  decoder interface in both model conditions;
- retain the epoch-44 checkpoint as the failed unconstrained baseline;
- identify the flat and graph capacity and training controls;
- keep fully discrete graph geometry outside the required comparison; and
- record all new runs separately from the existing evidence chain.

## Review conditions

Reconsider or expand this scope only after the minimal flat-versus-graph
comparison and concise report are complete, or if the shared neural decoder
cannot be made reproducible within its timebox. Any change to the principal
comparison, decoder condition, evaluation set, or held-out test policy should
receive a new superseding ADR rather than silently rewriting this record.
