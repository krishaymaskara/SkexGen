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
- VQ-collapse diagnosis and train-only k-means initialization.

Revolve support itself is not the proposed contribution. The intended
scientific comparison is between a flat/mixed baseline, a typed graph with
fully discrete structure-conditioned geometry, and a typed graph with
discrete structure plus continuous conditioned geometry. The graph models
and final comparison are not yet complete.

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
