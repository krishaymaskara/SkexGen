# Architecture and Research Decisions

This directory contains durable architecture decision records (ADRs) for
choices that materially affect schemas, model comparisons, evaluation, data
authority, or scientific interpretation.

Use the [decision record template](../templates/decision-record.md). Each ADR
should describe its context, the accepted decision, alternatives considered,
consequences, evidence, and supersession status. ADRs record why a choice was
made; package READMEs remain authoritative for the current implemented API.

Use monotonically increasing names such as:

```text
ADR-0001-controlled-cad-schema-v1.md
ADR-0002-physical-family-learning-unit.md
```

An accepted ADR is immutable except for correcting factual errors or marking
it superseded by a later ADR.

## Decision index

- [ADR-0001: Controlled CAD schema version
  1](ADR-0001-controlled-cad-schema-v1.md) — freezes the implemented node
  vocabulary, consumer-to-resource edge convention, `DEFINED_IN` and
  `DEPENDS_ON` semantics, operation sequence, single-body Boolean rules, and
  controlled-generator restrictions.
- [ADR-0002: Three-week flat-versus-graph research
  scope](ADR-0002-three-week-flat-versus-graph-scope.md) — narrows the active
  project to one repaired-flat-versus-typed-graph comparison under a shared
  category-conditioned constrained continuous-geometry decoder.
