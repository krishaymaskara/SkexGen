# ADR-0007: GE1 C7 Optimization-Sufficiency Diagnostic

- Status: `accepted`
- Proposal date: `2026-08-09`
- Decision date: `2026-08-09`
- Owner: project research team
- Designated GE1 reviewer: Krishay Maskara
- Adds to: [ADR-0006](ADR-0006-ge1-c7-sufficiency-execution-contract.md)
- Supersedes: none
- Superseded by: none

## Context

Formal C7 Adroit job `3344505` completed successfully under its frozen
50-epoch contract and failed both four-family autonomous exact-sufficiency
gates. On that cohort, 50 epochs were only 50 optimizer updates. Both arms'
losses decreased through update 50, neither satisfied the frozen plateau
diagnostic, and the final moving-window improvement remained approximately
16.5% against the 1% plateau threshold. Missing required graph edges were the
predominant autonomous failure.

The formal C7 result is a valid scientific failure. It does not, however,
distinguish a decoder that cannot express the four targets from one that had
not received sufficient optimization. A fixed-update, train-only trajectory
can answer that narrower question without reopening or replacing C7.

## Decision

### Scope and identities

The project will implement one additive, post-C7 optimization-sufficiency
diagnostic with these frozen identities:

```text
GE1-C7-OPTIMIZATION-SUFFICIENCY-DIAGNOSTIC-v1
GE1-C7-OPTIMIZATION-TRAJECTORY-v1
GE1-C7-OPTIMIZATION-ARTIFACT-v1
```

This diagnostic:

- does not change ADR-0004, ADR-0005, ADR-0006, or
  `GE1-STAGE0-PREREG-v1`;
- does not overwrite, relabel, rerun, reinterpret, or change the formal C7
  result;
- does not authorize Stage 6;
- does not implement or invoke C8 or the preauthorized decoder repair;
- does not alter any protected-access rule; and
- is a post-C7 diagnostic, not a replacement C7 execution.

Changing the family set, seed, initialization, optimizer, maximum update
count, measurement milestones, autonomous semantics, or interpretation rules
requires another ADR.

### Data boundary

Only the same four `operation_template.train` families selected by formal C7
are eligible:

```text
E  sf_d11886750826237040bec2a8580aa378e367b3e7ba51dd972c57b476408c470c
R  sf_5bbeb6143e53c2d259f24a60b576708dd8267b0ad49ed62c31c966e36e503ac4
EE sf_0626a6cb7e6b08cd7cae372a0d61d028cdc57c0d5bfd708a46397024e37d72f1
RE sf_fe8e4fd3246a0c94e25c661b0e879fa9099d098c8f558424898e72f3822746ff
```

The lexicographically sorted selected-set SHA-256 is:

```text
c07e90675099cb2ea58575fe91683dc80b70cf3fa85fa6fa8de0c44dbe9783d6
```

The existing metadata-only C7 selector must reproduce exactly these IDs and
this digest before payload access. No new salt or payload-aware selection is
permitted. The only permitted payload access is these four train families.
Operation-template development, RR systematic, ER test, IID, history-depth,
geometry-extrapolation, and every other train family remain forbidden.

Implementation and tests use only synthetic metadata, procedural fixtures,
and temporary directories. They do not open an authoritative manifest or CAD
history.

### Model lifecycle

Each arm uses seed 2026 and a fresh initialization. The flat and typed-graph
models are constructed as a matched pair, with byte-identical shared-component
initialization and disjoint mutable parameters and buffers. The diagnostic
cannot resume from formal C7 checkpoints, warm-start either arm, or publish
its checkpoints as full-comparison checkpoints.

### Fixed-update training

The existing recipe remains unchanged:

| Item | Frozen value |
|---|---:|
| Optimizer | AdamW |
| Learning rate | `1e-3` |
| Weight decay | `0` |
| Global gradient clipping | `1.0` |
| Batch size | 8 |
| Effective examples per update | 4 |
| Seed | 2026 |
| Maximum optimizer updates | 500 |
| Family presentations per arm | 2,000 |

Optimizer update count is authoritative. The current one-batch cohort makes
one update equal one epoch, but epoch is only an accompanying diagnostic
coordinate. Each arm trains continuously through all 500 updates. No loss,
plateau observation, autonomous output, or intermediate exact result may stop
or extend training or select a checkpoint.

Predetermined autonomous milestones are exactly updates 50, 100, 200, and
500. A milestone checkpoint is written before evaluation and strictly loaded
into a newly constructed model. Scientific checkpoints are retained at all
four milestones, recovery checkpoints every 25 updates, and update 500.
Duplicate recovery/scientific milestones produce one file, for 20 retained
checkpoints per arm. This additive schedule does not change C6 or formal C7
checkpoint behavior.

The frozen plateau calculation remains diagnostic only. Deterministic family
ordering, batching, common loss, and loss-component semantics are unchanged.

### Autonomous measurements

At every milestone, the existing target-free autonomous path runs before
target scoring. Per arm and physical family, the trajectory records:

```text
exact_node_sequence
exact_graph
strict_conversion
complete_executable_validity
depends_on_exactness for EE and RE
normalized executable prefix
first failure code
stage of first failure
attachment accuracy
dependency precision and recall
geometry diagnostics
```

It also records `P_true`, `P_shuffle`, `P_mean`, all five primary reporting
fields, donor-agreement availability, and memory-alteration evidence. Only
autonomous `P_true` exactness may establish sufficiency; teacher-forced values
and losses remain diagnostics.

An arm is exact-sufficient at a milestone only when every one of the four
families has exact unrounded `1.0` for node sequence, graph, strict conversion,
and complete executable validity, and both `EE` and `RE` have
`depends_on_exactness == 1.0`.

### Final interpretation

After all 500 updates, the artifact must emit:

```text
flat_ever_exact_at_predetermined_milestone
typed_graph_ever_exact_at_predetermined_milestone
flat_first_exact_update
typed_graph_first_exact_update
flat_exact_at_update_500
typed_graph_exact_at_update_500
flat_plateau_observed
typed_graph_plateau_observed
optimization_budget_explanation
original_c7_result_changed=false
stage6_authorized=false
decoder_repair_invoked=false
c8_or_later_performed=false
```

The interpretation category is exactly one of:

- `undertraining_supported_both_arms`: both arms become exact-sufficient at a
  predetermined milestone;
- `undertraining_or_optimization_difference_supported_one_arm`: exactly one
  arm becomes exact-sufficient;
- `still_improving_at_update_500`: neither arm passes and at least one arm has
  not plateaued;
- `plateaued_without_exact_sufficiency`: neither arm passes and both arms
  satisfy the frozen plateau diagnostic; or
- `infrastructure_failure`: execution, provenance, strict reload, access,
  metrics, or artifact integrity fails.

The diagnostic must not emit `overall_gate_pass`, reuse the formal C7 decision
identity, activate the C7 repair trigger, or authorize Stage 6. A completed
negative diagnostic finalizes and exits zero. Infrastructure failures do not
finalize and exit nonzero.

### Slurm provenance

The runner explicitly passes `SLURM_JOB_ID` through the Apptainer clean
environment. It must be a nonempty decimal string and must agree across
`resolved_config.json`, run metadata, every checkpoint provenance record, the
terminal event, and artifact verification. Missing, null, malformed, or
inconsistent identity is an infrastructure failure.

### Artifact and CLI

The CLI exposes only `--corpus-dir`, `--output-dir`, `--repository-root`, and
`--expected-commit`. All other choices are governed constants.

Publication uses a new external, non-symlink output path and an incomplete
sibling directory followed by atomic rename. The final artifact contains at
least:

```text
resolved_config.json
metrics.jsonl
checkpoints/
artifact_manifest.json
SHA256SUMS
```

JSON is canonical and finite. Manifest paths are safe and relative. Every
file has a recorded byte size and SHA-256. Checksums are sorted with final LF,
verified against mutation, and followed by a terminal diagnostic-completion
event. The artifact records source identity, Slurm identity, data boundary,
fresh initialization, training arithmetic, milestones, checkpoint schedule,
loss trajectories, strict reload evidence, autonomous metrics, interpretation,
and complete access declarations.

## Alternatives considered

### Rerun formal C7 for more epochs

Rejected. That would change a frozen result after observing it. The new
identity preserves C7 and asks a distinct optimization question.

### Continue from epoch-50 checkpoints

Rejected. Resume would measure another 450 updates from an observed run, not
a predetermined clean-start trajectory.

### Stop at the first exact milestone

Rejected. It would make training duration outcome-dependent and erase whether
exactness persists at update 500.

### Write every update checkpoint

Rejected. The fixed 25-update recovery schedule provides bounded recovery and
all scientific milestones without 1,000 checkpoint files.

### Use training loss or teacher forcing as sufficiency

Rejected. The scientific problem is autonomous executable reconstruction.
Losses and teacher-forced quantities cannot replace that endpoint.

## Consequences

- Formal C7 job `3344505` remains an unchanged scientific failure.
- The new run can distinguish milestone exactness from persistent failure
  under a ten-times-longer update budget without opening protected data.
- Both arms receive the same matched clean-start treatment and fixed schedule.
- Slurm identity becomes an enforced structured provenance field.
- No diagnostic outcome automatically authorizes Stage 6 or invokes repair.
- C8 remains unstarted.

## Validation and evidence

Implementation validation must cover frozen cohort identity, clean matched
initialization, 500-update arithmetic, exact checkpoint schedule, no early
stop, milestone strict reload, autonomous exactness, all five interpretations,
artifact integrity, structured Slurm identity, access ordering, source
cleanliness, and noninterference with C6/formal C7 behavior.

Local tests may use only synthetic or procedural inputs. A later exact-commit
Adroit run is required for a scientific result; implementation completion
alone establishes no diagnostic outcome.

## Related records

- [ADR-0004](ADR-0004-ge1-single-manifest-encoder-comparison.md)
- [ADR-0005](ADR-0005-ge1-primary-reporting-and-metrics-v2.md)
- [ADR-0006](ADR-0006-ge1-c7-sufficiency-execution-contract.md)
- [Formal C7 experiment record](../experiments/ge1_c7_pilot.md)
- [GE1 Stage 0 preregistration](../specifications/ge1_stage0_preregistration.md)
- [GE1 implementation plan](../specifications/graph_encoder_implementation_plan.md)
