# Documentation Index

Repository documentation is organized by authority and purpose. Research
plans and specifications define intended experiments; prototype READMEs
define implemented contracts and limitations; `docs/experiments` records
completed runs and evidence; and `notes` contains progress reports and
detailed analysis. Git history is the chronological change log.

## Normative research documents

- [Research plan](research_plan.md) — project thesis, hypotheses, scope,
  milestones, and claims to avoid.
- [Experimental specification](../notes/experimental_spec.md) — detailed
  structured extrude-and-revolve CAD experiment and evaluation contract.

## Experiment evidence

- [B0 training-infrastructure
  validation](experiments/b0_training_infrastructure_validation.md) — CPU
  validation and CUDA engineering smoke evidence.
- [B0 680-family pilot
  corpus](experiments/b0_pilot_corpus_680.md) — generation configuration,
  split counts, integrity hash, and limitations.
- [Historical 60-family OpenCascade
  validation](experiments/kernel_validation_60.md) — real-kernel results that
  motivated analytical Boolean-feasibility filtering.
- [Pretrained SkexGen smoke
  test](../notes/adroit_pretrained_smoke_test.md) — reduced pretrained
  inference and OBJ-export validation on Adroit.

## Prototype and package contracts

- [Typed CAD representation](../prototype/representation/README.md)
- [Controlled synthetic data](../prototype/controlled_data/README.md)
- [CAD-kernel validation](../prototype/kernel_validation/README.md)
- [Counterfactual edit benchmark](../prototype/counterfactual_edits/README.md)
- [Model-ready data interface](../prototype/model_data/README.md)
- [B0 flat mixed/VQ baseline](../prototype/flat_baseline/README.md)

These READMEs describe what the checked-in implementations support, how to use
them, and which cases remain outside their current scope. They are not
substitutes for completed-run evidence.

## Weekly progress and design notes

- [Week 3 progress report](../notes/week3.md)
- [Experimental specification](../notes/experimental_spec.md)

The weekly report is intentionally concise and links to durable evidence.
Detailed design reasoning remains in the experimental specification and
package contracts rather than being duplicated in progress notes.

## Historical repository and environment analysis

- [SkexGen repository map](../notes/repository_map.md)
- [Adroit pretrained smoke test](../notes/adroit_pretrained_smoke_test.md)

These records explain the inherited SkexGen codebase and the environment used
to establish baseline operability.

## Artifact policy

Raw datasets, generated corpora, checkpoints, scheduler logs, metrics files,
CAD solids, caches, containers, `__pycache__` directories, and `.pyc` files
are not normally committed. Experiment records retain compact provenance,
verified aggregate results, integrity values when available, and external
artifact locations when they are stable and known.
