# Specifications

Specifications define proposed or frozen research and evaluation contracts.
They do not establish that an implementation exists or an experiment
succeeded.

- [Structured extrude-and-revolve experimental
  specification](experimental_spec.md)
- [Flat-baseline Phase B validation evaluation
  specification](flat_baseline_phase_b_evaluation.md)
- [Typed graph encoder implementation
  plan](graph_encoder_implementation_plan.md)
- [GE1 frozen Stage 0 configuration and
  preregistration](ge1_stage0_preregistration.md)
- [GE1 C5 shared decoder and output-position
  contract](ge1_shared_decoder_contract.md)
- [GE1 C6 training, autonomous measurement, and intervention
  contract](ge1_c6_measurement_contract.md)
- [GE1 C7 optimization-sufficiency diagnostic
  contract](ge1_c7_optimization_sufficiency_diagnostic.md)
- [GE1 C7-v2 prospective execution
  contract](ge1_c7_v2_execution_contract.md)

Each specification labels requirements as implemented, proposed, frozen, or
awaiting a decision. When its current-state language differs from the
[status page](../status.md), the status page controls.

## Active diagnostic preregistrations

- [Phase B constraint-manifold replay](flat_baseline_phase_b_constraint_manifold_replay.md):
  a read-only, four-arm diagnostic over the immutable recovered job `3329040`
  bundle. It cannot alter the frozen Gate C or Gate D failures.
- [Phase B categorical-isolation replay](flat_baseline_phase_b_categorical_isolation_replay.md):
  a preregistered 2-by-2 plane-category/profile-category diagnostic over the
  immutable recovered bundle and completed constraint-manifold replay.
