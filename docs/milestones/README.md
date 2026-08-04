# Milestone Summaries

Milestone pages summarize what a project stage delivered and link to the
experiment records that support it. They answer “what is complete?” without
duplicating raw run details.

## Accepted milestones

- [Controlled CAD foundation](controlled_cad_foundation.md) — versioned
  representation, deterministic controlled data, kernel validation,
  counterfactual evaluation data, and the shared model-data interface.
- [Flat mixed/VQ baseline](flat_baseline.md) — baseline implementation,
  original collapse, diagnosis, bounded repair, and the current
  evaluation-readiness boundary.
- [Constrained Flat V6 and Graph V1
  comparison](constrained_flat_graph_comparison.md) — shared constrained
  decoder, frozen flat structural baseline, capacity-matched graph pilots,
  the single correction, and the single- versus two-operation boundary.

## Maintenance rule

A milestone page should identify:

- its scope and acceptance criteria;
- implementation commits;
- completed experiment records;
- conclusions supported by those records;
- limitations and deferred work;
- the next gate.

Experiment evidence belongs under `docs/experiments`, current package
behavior belongs in `prototype/*/README.md`, and present overall state belongs
in `docs/status.md`.

Create or update a milestone page only when its acceptance boundary changes.
New runs normally add experiment records; they change a milestone only when
they satisfy, invalidate, or materially refine that milestone's gate.
