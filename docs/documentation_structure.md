# Project Documentation Structure

## Purpose

This document explains how project documentation is organized and maintained.
The goal is to keep current status, experimental evidence, implementation
contracts, future plans, and historical reference material separate. This
prevents planned work from being mistaken for completed work and prevents a
successful code change from being mistaken for a successful scientific
experiment.

The main entry point is `docs/README.md`. The concise source of current
project state is `docs/status.md`.

## Directory structure

```text
README.md
docs/
  README.md
  documentation_structure.md
  status.md
  research_plan.md
  decisions/
  experiments/
  milestones/
  specifications/
  reports/
  reference/
  templates/
prototype/
  representation/README.md
  controlled_data/README.md
  kernel_validation/README.md
  counterfactual_edits/README.md
  model_data/README.md
  flat_baseline/README.md
  graph_baseline/README.md
  graph_encoder/README.md
tools/
  check_documentation.py
```

The repository-root `README.md` introduces the project and points readers to
the documentation system. It should remain short. Detailed project
documentation belongs under `docs/`, while implementation-specific contracts
belong in the relevant `prototype/*/README.md`.

## Authority order

When two documents appear to disagree, use the following order:

1. `docs/status.md`
2. `docs/experiments/`
3. `prototype/*/README.md`
4. `docs/specifications/`
5. `docs/research_plan.md`
6. `docs/reference/`

Repository code overrides stale descriptions of implementation behavior.
However, code alone does not prove that an external run occurred or that a
scientific gate passed. Those claims require experiment evidence.

## Current project status

File:

```text
docs/status.md
```

This is the authoritative answer to the question, “Where is the project
now?” It records:

- the active branch and accepted repository revision;
- the authoritative corpus and checkpoint;
- completed implementation milestones;
- the current evidence-supported conclusion;
- important results that have not yet been established;
- the immediate scientific gate;
- test-partition status.

Update this file when the immediate next action changes, a new checkpoint or
corpus becomes authoritative, a milestone is accepted or rejected, the test
partition is first used, or an important limitation changes.

Do not place detailed run logs or complete metric tables here. Link to the
relevant experiment record instead.

## Experiment evidence

Directory:

```text
docs/experiments/
```

The `docs/experiments/README.md` file is the evidence inventory. It maps each
known run to its source revision, job identifier, partition use, artifact
location, evidence state, and durable record.

Each meaningful experiment receives its own Markdown file. An experiment
record should contain:

- a stable experiment identifier;
- source and working-tree provenance;
- the question and predetermined decision rule;
- corpus and partition authority;
- configuration and environment;
- verified results;
- the resulting decision;
- careful interpretation;
- limitations and unsupported claims;
- artifact locations and checksums;
- reproduction or validation information;
- links to related records.

Use `docs/templates/experiment-record.md` when creating a record.

Experiment records are immutable evidence. If an experiment is rerun, create
a new record rather than rewriting the old result. The new record can
supersede, reproduce, diagnose, or extend the earlier one.

Failed runs should remain documented when they affected a scientific or
engineering decision. For example, the original one-code B0 collapse is part
of the evidence chain leading to diagnosis and train-only k-means
initialization.

Raw datasets, checkpoints, logs, generated solids, and large reports normally
remain outside Git. The experiment record should preserve stable artifact
locations, checksums, scheduler results, and enough provenance to interpret
the external bundle.

## Package and implementation contracts

Locations:

```text
prototype/representation/README.md
prototype/controlled_data/README.md
prototype/kernel_validation/README.md
prototype/counterfactual_edits/README.md
prototype/model_data/README.md
prototype/flat_baseline/README.md
prototype/graph_baseline/README.md
prototype/graph_encoder/README.md
```

These files describe the behavior of checked-in code. They contain API
contracts, schemas, tensor shapes, configuration fields, commands,
determinism rules, supported cases, and implementation limits.

Update the relevant package README when code changes its public behavior.
Do not use package documentation to claim that a cluster experiment
succeeded. Link to an experiment record for that evidence.

## Milestones

Directory:

```text
docs/milestones/
```

Milestone pages summarize what a project stage delivered. They identify:

- the milestone's scope and acceptance boundary;
- important implementation commits;
- experiment records supporting acceptance;
- conclusions supported by the evidence;
- limitations and deferred work;
- the next gate.

Milestones should summarize rather than duplicate detailed metrics. Update a
milestone only when its acceptance boundary changes. A normal new experiment
does not require a milestone update unless it satisfies, invalidates, or
materially changes that milestone.

## Decisions

Directory:

```text
docs/decisions/
```

Decision records explain choices with lasting effects on the project. Examples
include:

- freezing or revising a representation schema;
- changing dependency-edge semantics;
- changing partition authority;
- changing the principal model comparison;
- changing an evaluation or checkpoint-selection rule;
- authorizing first use of the test partition.

Use `docs/templates/decision-record.md`. A decision record explains what was
chosen, why it was chosen, which alternatives were considered, and what
consequences follow. It is not an experiment report.

## Specifications

Directory:

```text
docs/specifications/
```

Specifications define the research and evaluation contracts. They contain
implemented requirements, proposed protocols, and unresolved decisions.
Every requirement should be labeled clearly enough that a reader can tell
whether it is implemented, proposed, or awaiting approval.

Update a specification before a protocol-changing experiment when possible.
If a protocol must change after results have been examined, preserve the
original experiment record and document the new decision explicitly.

Specifications do not prove implementation or experimental success. Current
implementation behavior belongs in package READMEs, and completed-run evidence
belongs in experiment records.

## Research plan

File:

```text
docs/research_plan.md
```

The research plan records the project thesis, hypotheses, intended
comparisons, original staged roadmap, risks, and scientific claim
requirements.

It is not the current progress tracker. Update it only when the research
question, project scope, main comparison, or overall scientific strategy
changes. Use `docs/status.md` for current progress and immediate next steps.

## Reports

Directory:

```text
docs/reports/
```

Reports are dated, audience-specific summaries such as mentor updates. They
are derived from the status page, milestones, experiment records, and package
contracts.

Reports are snapshots. Do not continually edit an old report to represent the
current project. Create a newly dated report when another external update is
needed. Reports do not override the current status or evidence records.

## Inherited-system reference

Directory:

```text
docs/reference/
```

This directory contains frozen historical information about the inherited
SkexGen system, including the upstream README and the repository map created
before the research extension.

Reference documents are normally not updated. If the current implementation
changes, update the corresponding package README rather than the inherited
reference. Reference material has the lowest documentation authority.

## Templates

Directory:

```text
docs/templates/
```

Templates define the minimum information expected in new experiment and
decision records. Update a template only when the documentation standard
itself changes. Changing a template does not require rewriting historical
records unless missing information materially affects their interpretation.

## Routine workflow for a new experiment

Before the run:

1. Identify the exact scientific or engineering question.
2. Write the acceptance, rejection, stopping, and selection rules.
3. Freeze the corpus, split, configuration, and checkpoint inputs.
4. Identify the source commit and confirm working-tree provenance.
5. Confirm whether validation or test data will be used.

During and immediately after the run:

1. Preserve scheduler accounting, stdout, and stderr.
2. Preserve the resolved configuration and environment information.
3. Preserve metrics, reports, manifests, and important checkpoints.
4. Create an artifact manifest and checksums.
5. Record whether the run completed and why it passed or failed.

Documentation updates:

1. Create a new record under `docs/experiments/`.
2. Add the record to `docs/experiments/README.md`.
3. Add it to `docs/README.md`.
4. Update `docs/status.md` if the current conclusion or next gate changed.
5. Update a milestone only if its acceptance boundary changed.
6. Update a package README only if checked-in behavior changed.
7. Create a decision record if a lasting rule or protocol changed.

## Routine workflow for an implementation change

When code behavior changes:

1. Update the relevant `prototype/*/README.md`.
2. Add or update tests for the implementation contract.
3. Update a specification if the scientific protocol changed.
4. Create a decision record if the change alters a lasting architectural or
   interpretive choice.
5. Do not update an experiment record unless documenting a new run.
6. Update current status only if the project gate or accepted capability
   changed.

## Routine workflow for a completed milestone

When a stage becomes complete:

1. Confirm that its implementation is identifiable.
2. Confirm that the required validation ran in the authoritative environment.
3. Confirm that partition use and artifacts were checked.
4. Confirm that every supporting run has an experiment record.
5. Update or create the milestone summary.
6. Update `docs/status.md`.
7. Link the milestone from `docs/milestones/README.md` and `docs/README.md`.

## Documentation validation

Run the following command before committing documentation changes:

```bash
python3 tools/check_documentation.py
```

The checker validates:

- local Markdown links;
- experiment-index coverage;
- milestone-index coverage;
- reference, report, and specification index coverage;
- durable records for locally verified evidence;
- required semantic sections in structured experiment records;
- the unified documentation layout, including rejection of loose pages at the
  root of `docs/` outside the four designated root documents.

The retired `notes/` layout must not be recreated. New project Markdown
belongs in an indexed `docs/` collection, one of `docs/README.md`,
`docs/documentation_structure.md`, `docs/research_plan.md`, or
`docs/status.md`, the root README, or an appropriate package README.

## Practical update summary

The files updated most often during normal research are:

```text
docs/status.md
docs/experiments/README.md
docs/experiments/<new-experiment-record>.md
prototype/<affected-package>/README.md
```

Files updated occasionally are:

```text
docs/milestones/<milestone>.md
docs/decisions/<decision>.md
docs/specifications/<specification>.md
docs/reports/<dated-report>.md
```

Files that are normally stable are:

```text
docs/research_plan.md
docs/reference/*
docs/templates/*
README.md
```

Following these boundaries keeps the project readable: status says what is
true now, experiment records show what happened, package READMEs explain what
the code does, specifications state the protocol, decisions explain lasting
choices, milestones summarize accepted stages, and reports communicate dated
progress to external readers.
