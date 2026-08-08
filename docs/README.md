# Documentation Index

This repository separates present project state, scientific evidence,
implemented contracts, intended research, and historical reasoning. The
separation prevents a plan from being mistaken for completed work and
prevents a checked-in capability from being mistaken for a successful
scientific result.

Git history is the code change log. It is not a substitute for experiment
provenance or a current project-status record.

- [Project documentation structure and workflow](documentation_structure.md)

## Documentation authority

When documents disagree, use this order:

1. [Current project status](status.md) — what is accepted, incomplete, and
   next now.
2. `docs/experiments` — evidence from completed runs, including provenance,
   results, decisions, and limitations.
3. `prototype/*/README.md` — current checked-in package and API contracts.
4. `docs/specifications` — frozen or proposed evaluation design, according
   to each specification's labels.
5. The research plan — motivation, hypotheses, scope, and roadmap.
6. `docs/reference` — explicitly historical inherited-system material.

Current repository code overrides stale implementation details in prose, but
code alone does not establish that an external experiment ran or succeeded.

## Current state

- [Current project status](status.md)

Update this page whenever a milestone is accepted, a scientific gate changes,
the authoritative checkpoint or corpus changes, test data is first used, or
the immediate next action changes.

## Normative research documents

- [Research plan](research_plan.md) — project thesis, hypotheses, scope,
  milestones, and claims to avoid.
- [Specifications](specifications/README.md)
- [GE1 C5 shared decoder and output-position
  contract](specifications/ge1_shared_decoder_contract.md) — additive frozen
  record for decoder parity, position routing, autonomous output, loss, and
  strict checkpoints.
- [GE1 C6 measurement
  contract](specifications/ge1_c6_measurement_contract.md) — additive frozen
  record for training, recovery, target-free evaluation, executable-prefix
  scoring, family-macro aggregation, interventions, and resource reporting.
- [ADR-0006: GE1 C7 sufficiency execution
  contract](decisions/ADR-0006-ge1-c7-sufficiency-execution-contract.md) —
  accepted additive record for metadata-only train cohorts, fresh matched
  gate models, epoch-50 autonomous decisions, and immutable C7 artifacts.
- [Experimental specification](specifications/experimental_spec.md) —
  detailed structured extrude-and-revolve CAD experiment and evaluation
  contract.

## Experiment evidence

- [Experiment evidence inventory](experiments/README.md) — availability,
  integrity, job/source mapping, and retrieval gaps for all known runs.
- [Counterfactual OpenCascade
  audit](experiments/counterfactual_opencascade_audit.md) — 272 endpoint
  executions, 136 edit-pair results, and continuous/quantized agreement.
- [Real-PyTorch model-data
  validation](experiments/model_data_pytorch_validation.md) — the shared
  interface under Python 3.8 and PyTorch 1.11 on Adroit.
- [B0 training-infrastructure
  validation](experiments/b0_training_infrastructure_validation.md) — CPU
  validation and CUDA engineering smoke evidence.
- [B0 680-family pilot
  corpus](experiments/b0_pilot_corpus_680.md) — generation configuration,
  split counts, integrity hash, and limitations.
- [Original B0 training and
  collapse](experiments/b0_original_training_collapse.md) — operationally
  successful training that collapsed to one code.
- [Phase A contract
  validation](experiments/b0_phase_a_contract_validation.md) — state-dict and
  checkpoint compatibility evidence.
- [Phase B validation
  evaluation](experiments/b0_phase_b_validation_evaluation.md) — paired
  teacher-forced/predicted-history results for the original checkpoint.
- [VQ-collapse
  diagnosis](experiments/b0_vq_collapse_diagnosis.md) — root-cause evidence
  and the bounded repair gate.
- [Train-k-means
  pilot](experiments/b0_train_kmeans_pilot.md) — train-only initialization
  treatment and five-epoch decision.
- [Full train-k-means
  retraining](experiments/b0_train_kmeans_full_retrain.md) — 50-epoch
  continuation and selected epoch-44 checkpoint.
- [Repaired-checkpoint Phase B deterministic
  smoke](experiments/b0_phase_b_repaired_smoke.md) — six-family
  validation-only payload and byte-identical repeat-publication evidence.
- [Repaired-checkpoint Phase B full
  validation](experiments/b0_phase_b_repaired_full_validation.md) — recovered
  68-family result, manifest-only finalization, and frozen Gate B-D inputs.
- [Phase B categorical-isolation
  replay](experiments/b0_phase_b_categorical_isolation_replay.md) —
  preregistered plane/profile factorial diagnosis over the immutable repaired
  validation bundle and constraint-manifold replay.
- [Phase B constraint-manifold
  replay](experiments/b0_phase_b_constraint_manifold_replay.md) — four-arm
  diagnostic separating semantic constants, constrained geometry,
  categorical oracle sensitivity, and autoregressive sensitivity.
- [Constrained Flat V2-V5
  progression](experiments/b0_constrained_flat_v2_v5_progression.md) — staged
  profile, plane, categorical, and node-grammar repair with external-evidence
  limitations.
- [Frozen constrained Flat
  V6](experiments/b0_constrained_flat_v6.md) — capacity reference and uniform
  structural-edge failure on all 68 IID-validation families.
- [Graph V1 engineering readiness and C1
  validation](experiments/b0_graph_v1_engineering_readiness.md) — all
  pre-pilot failure/repair jobs, provenance hardening, and authoritative
  regression validation.
- [Initial Graph V1](experiments/b0_graph_v1_initial.md) — explicit typed-edge
  decoder, 22/68 complete validity, and the single- versus two-operation
  boundary.
- [Graph V1 directed-position
  C1](experiments/b0_graph_v1_c1.md) — the single authorized correction,
  improved edge metrics, unchanged complete validity, and freeze decision.
- [GE1 C3 authoritative CPU
  validation](experiments/ge1_c3_cpu_validation.md) — exact-commit Python
  3.8/PyTorch 1.11 fixture validation of target-separated paired batching on
  Adroit CPU, with zero corpus or protected-partition access.
- [GE1 C4 authoritative CPU
  validation](experiments/ge1_c4_cpu_validation.md) — exact-commit validation
  of both standalone encoders, including continuous parity, capacity,
  sensitivity, isolation, gradients, and permutation contracts.
- [GE1 C4 review-fix authoritative CPU
  validation](experiments/ge1_c4_review_fix_cpu_validation.md) — exact-commit
  revalidation of the frozen capacity contract and arm-independent shared
  initialization, including all C4 and graph-encoder tests with zero skips.
- [GE1 C5 authoritative CPU validation attempt
  3344278](experiments/ge1_c5_cpu_validation_attempt_3344278.md) — failed
  focused-suite attempt traced to three test-contract assertions and retained
  as historical evidence.
- [GE1 C5 authoritative CPU
  validation](experiments/ge1_c5_cpu_validation.md) — corrected exact-commit
  validation of shared-decoder parity, common loss, autonomous output,
  gradients, initialization isolation, and strict checkpoints.
- [GE1 C6 authoritative CPU validation attempt
  3344337](experiments/ge1_c6_cpu_validation_attempt_3344337.md) — failed
  focused-suite attempt identifying one autonomous node-count adapter defect
  and one missing test import; no corpus smoke or payload access occurred.
- [GE1 C6 authoritative CPU validation attempt
  3344363](experiments/ge1_c6_cpu_validation_attempt_3344363.md) — corrected
  exact-commit rerun in which 36/37 focused tests passed; the sole error was a
  test-only generic equality check on tensor-bearing dataclasses, before any
  corpus smoke or payload access.
- [GE1 C6 authoritative CPU
  validation](experiments/ge1_c6_cpu_validation.md) — corrected exact-commit
  validation of governed training, deterministic recovery, autonomous
  metrics, memory interventions, and both authorized train-only arm smokes.
- [GE1 C5/C6 review-fix authoritative CPU
  validation](experiments/ge1_c5_c6_review_fix_cpu_validation.md) — combined
  exact-commit revalidation of shared-decoder provenance, prefix handling,
  structured primary reporting, donor agreement, metrics v2, and both
  authorized train-only arm smokes.
- [Historical 60-family OpenCascade
  validation](experiments/kernel_validation_60.md) — real-kernel results that
  motivated analytical Boolean-feasibility filtering.
- [Pretrained SkexGen smoke
  test](experiments/pretrained_skexgen_smoke_test.md) — reduced pretrained
  inference and OBJ-export validation on Adroit.

Experiment records should use the
[experiment-record template](templates/experiment-record.md). A run is not
durably documented until the record distinguishes its question,
predetermined gate, inputs and partitions, configuration, environment,
verified results, interpretation, unsupported claims, and artifact integrity.

The experiment inventory records any remaining evidence-recovery gaps. All
completed runs with locally recovered evidence bundles currently have
individual records.

## Prototype and package contracts

- [Typed CAD representation](../prototype/representation/README.md)
- [Controlled synthetic data](../prototype/controlled_data/README.md)
- [CAD-kernel validation](../prototype/kernel_validation/README.md)
- [Counterfactual edit benchmark](../prototype/counterfactual_edits/README.md)
- [Model-ready data interface](../prototype/model_data/README.md)
- [B0 flat mixed/VQ baseline](../prototype/flat_baseline/README.md)
- [Graph-native constrained baseline](../prototype/graph_baseline/README.md)

These READMEs describe what the checked-in implementations support, how to use
them, and which cases remain outside their current scope. They are not
substitutes for completed-run evidence.

## Decisions and milestones

- [Architecture and research decisions](decisions/README.md)
- [Milestone summaries](milestones/README.md)
- [Controlled CAD foundation](milestones/controlled_cad_foundation.md)
- [Flat mixed/VQ baseline](milestones/flat_baseline.md)
- [Constrained Flat V6 and Graph V1
  comparison](milestones/constrained_flat_graph_comparison.md)
- [Decision-record template](templates/decision-record.md)

Decision records explain choices with lasting effects on schemas, model
comparisons, data authority, or interpretation. Milestone pages summarize a
stage and link to its supporting experiment records; they should not duplicate
raw metrics or logs.

## Reports

- [Project reports](reports/README.md)
- [Mentor progress report — July 28,
  2026](reports/mentor_progress_report_2026-07-28.md)
- [Flat V6 and Graph V1 progress report — August 3,
  2026](reports/graph_v1_progress_report_2026-08-03.md)
- [Graph V1 research handoff — August 3,
  2026](reports/graph_v1_handoff_2026-08-03.md)

Reports are dated audience-facing snapshots derived from the authoritative
status and evidence records. They are not maintained as alternate status
pages.

## Inherited-system reference

- [Reference index](reference/README.md)
- [Inherited SkexGen repository map](reference/inherited_skexgen_repository.md)
- [Original upstream README](reference/upstream_skexgen_readme.md)

Reference material explains the inherited SkexGen codebase but does not
describe current project status. Dated execution evidence belongs in
`docs/experiments`; duplicated weekly progress notes are not maintained.

## Artifact policy

Raw datasets, generated corpora, checkpoints, scheduler logs, metrics files,
CAD solids, caches, containers, `__pycache__` directories, and `.pyc` files
are not normally committed. Experiment records retain compact provenance,
verified aggregate results, integrity values when available, and external
artifact locations when they are stable and known.

An external scratch path is not durable by itself. Important evidence should
also have a content hash and, where practical, a frozen archive outside
ephemeral scratch storage. Missing or unrecoverable provenance must be stated
explicitly rather than reconstructed from memory.

## Completion rule

A scientific or infrastructure milestone is complete only when:

1. its implementation and configuration are identifiable;
2. required validation has run in the authoritative environment;
3. artifacts and partition use have been checked;
4. a durable experiment record states the result and limitations;
5. the current status and relevant package contract are updated.

Historical failed runs remain part of the evidence chain when they motivated
a diagnosis or intervention. They should be documented rather than rewritten
as successful runs.

All project Markdown belongs in `docs/`, the repository-root README, or a
package-level `prototype/*/README.md`. The former `notes/` documentation
layout is retired.

Run `python3 tools/check_documentation.py` before committing documentation
changes. The checker validates local Markdown links, evidence-index coverage,
milestone-index coverage, verified-local record coverage, and the required
sections of structured experiment records.
