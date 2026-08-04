# Structured CAD Representations Research Extension

This repository contains an active research extension of the original
SkexGen codebase. The project studies how CAD feature structure and geometry
should be represented to support systematic generalization and localized
editing in a controlled sketch, extrude, and revolve domain.

The implemented work includes:

- a typed, versioned CAD dependency representation;
- deterministic controlled data and systematic split generation;
- analytical Boolean-feasibility filtering and OpenCascade validation;
- a counterfactual edit benchmark;
- shared flat and graph model-data interfaces;
- a flat/mixed VQ baseline with paired validation evaluation;
- VQ-collapse diagnosis and train-only k-means initialization;
- a shared category-conditioned constrained profile/plane/axis decoder;
- a frozen constrained Flat V6 structural baseline;
- an effectively capacity-matched graph-native typed-edge baseline;
- one frozen evidence-backed graph correction and complete IID-validation
  comparison.

Revolve support itself is not the proposed contribution. The completed
bounded comparison asks whether explicit typed-edge prediction improves over
flat relation and pointer heads under the same constrained node/geometry path.
Graph V1 improves complete IID-validation validity from 0/68 to 22/68 by
solving every single-operation program, but neither the initial nor corrected
pairwise graph decoder solves a two-operation program. Systematic and held-out
test partitions remain untouched. See the current status for the precise
result and next gate.

## Documentation

All active project documentation lives under `docs/`:

- [Current project status](docs/status.md)
- [Documentation index and authority](docs/README.md)
- [Experiment evidence inventory](docs/experiments/README.md)
- [Milestone summaries](docs/milestones/README.md)
- [Research plan](docs/research_plan.md)
- [Experimental specification](docs/specifications/experimental_spec.md)
- [Inherited-system references](docs/reference/README.md)

Package-level implementation contracts live in `prototype/*/README.md`.
The original upstream README is preserved only as an
[inherited-system reference](docs/reference/upstream_skexgen_readme.md).
