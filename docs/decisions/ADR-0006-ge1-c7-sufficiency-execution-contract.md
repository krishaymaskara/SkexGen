# ADR-0006: GE1 C7 Sufficiency Execution Contract

- Status: `accepted`
- Proposal date: `2026-08-08`
- Decision date: `2026-08-08`
- Owner: project research team
- Designated GE1 reviewer: Krishay Maskara
- Adds to: [ADR-0004](ADR-0004-ge1-single-manifest-encoder-comparison.md)
  and [ADR-0005](ADR-0005-ge1-primary-reporting-and-metrics-v2.md)
- Supersedes: no frozen scientific choice in ADR-0004, ADR-0005, or
  `GE1-STAGE0-PREREG-v1`
- Superseded by: [ADR-0009](ADR-0009-ge1-positive-operation-magnitude-repair.md)
  for the decoder-repair choice only; the completed C7-v1 contract and result
  remain immutable

## Context

The accepted GE1 plan requires train-only decoder-sufficiency and autonomous
memory-use gates before development evaluation or the full encoder comparison.
Stage 5 freezes the accessible operation templates, exact-sufficiency
criteria, 32-family scaled set, intervention thresholds, and one possible
later hierarchical decoder repair. It does not fully specify deterministic
family selection, run lifecycle, scientific-failure handling, or immutable
artifact publication.

Those execution choices must be frozen before C7 implementation and before
any CAD-history payload is opened. Otherwise family selection, checkpoint
selection, rerun behavior, or artifact finalization could become
outcome-dependent. This record is an additive implementation and execution
contract. It changes no partition, model, threshold, training budget,
checkpoint rule, protected-access rule, or Stage 0 endpoint.

## Decision

### Scope and identities

C7 implements only the train-only Stage 5 sufficiency and memory-use gates.
Its identities are:

```text
GE1-C7-PILOT-v1
GE1-C7-SUFFICIENCY-GATES-v1
GE1-C7-ARTIFACT-MANIFEST-v1
```

C7 does not relabel or mutate `GE1-C6-METRICS-v2`. It uses a separately
versioned result envelope while reusing the C6 pure scoring, intervention,
donor-agreement, capacity, receptive-field, timing, and memory helpers.

C7 may open only `operation_template.train` families from templates `E`, `R`,
`EE`, and `RE`. Development, RR systematic, ER test, IID, history-depth, and
geometry-extrapolation partitions remain closed. The C7 production path must
not import or call `load_development`.

### Metadata-only deterministic family selection

Selection uses only the SHA-pinned authoritative `operation_template.json`
assignment metadata. For every train candidate in each accessible template,
the rank is:

```python
material = (
    "GE1-C7-SUFFICIENCY-v1\n"
    + operation_template
    + "\n"
    + source_family_id
    + "\n"
).encode("utf-8")
rank = hashlib.sha256(material).hexdigest()
```

Candidates are sorted by `(rank, source_family_id)` independently within
`E`, `R`, `EE`, and `RE`. The tiny set contains the first-ranked family per
template. The scaled set contains the first eight per template. Each returned
set is lexicographically sorted before `load_train`, and the tiny set must be
nested in the scaled set.

Each selected-set digest is SHA-256 over its lexicographically sorted family
IDs encoded as UTF-8, one per line, with a real final LF. Selection must verify
the existing manifest authority, template composition, 4/32 unique family
counts, 1/8 per-template counts, nestedness, and zero payload access.

### Seed, model lifecycle, and training

C7 uses only matched gate seed `2026`. Seeds 2027 and 2028 remain reserved for
the later full comparison and cannot be introduced after observing a C7
result.

Tiny and scaled gates each construct a fresh matched model pair. Shared
components begin byte-identically between arms while every mutable parameter
and buffer object remains disjoint. Scaled training cannot continue from a
tiny checkpoint, arm order cannot change initialization, and no C7 checkpoint
can warm-start or be represented as a full-comparison checkpoint.

Every arm/subset run uses the frozen continuous-memory recipe: 50 epochs,
batch size 8, AdamW, learning rate `1e-3`, weight decay `0`, and global
gradient clipping at `1.0`. The common C6 training loop is used. Expected
arithmetic is:

| Subset | Families | Presentations | Optimizer steps |
|---|---:|---:|---:|
| tiny | 4 | 200 | 50 |
| scaled | 32 | 1,600 | 200 |

Only a strict reload of epoch 50 may satisfy a gate. Earlier checkpoints and
training losses are diagnostics or recovery evidence only.

### Autonomous exact-sufficiency gates

Only fully autonomous `P_true` outputs from the strictly reloaded epoch-50
checkpoint can satisfy exact sufficiency. Every selected physical family must
have exact unrounded values of `1.0` for:

```text
exact_node_sequence
exact_graph
strict_conversion
complete_executable_validity
```

Every two-operation `EE` and `RE` family must additionally have
`depends_on_exactness == 1.0`. Teacher-forced output can be diagnostic only.
Controlled analytic validity remains authoritative; C7 adds no geometry
tolerance and requires no OpenCascade execution.

Each arm receives structured `tiny_exact_sufficiency`,
`scaled_exact_sufficiency`, and `scaled_memory_use` results. Status is exactly
`pass`, `fail`, or `not_run`; `not_run` never passes. Results include arm,
seed, subset identity and count, epoch/checkpoint identity, every criterion
and denominator, failing families and fields, reason, and access declarations.

Both tiny-arm gates run. A scientifically failed or incomplete tiny gate
marks that arm's scaled gates `not_run`, permits the other arm to continue
when execution remains healthy, and fails overall C7. If scaled exact
sufficiency fails after training, the scaled memory interventions still run
as diagnostics. Infrastructure errors are not scientific gate failures.

### Autonomous memory-use gate

The memory gate runs only on the 32-family scaled set. Evaluation order is
lexicographic, batch size is 8, and therefore batch membership is four fixed
full batches. The frozen C6 meanings of `P_true`, full-set seed-2026 cyclic
`P_shuffle`, and batch-local latent-position `P_mean` are unchanged. No
target, ground-truth prefix, repair, or teacher forcing enters generation.

The endpoint is the existing physical-family macro normalized executable-
prefix score. Each arm passes only if unrounded finite values satisfy:

```text
P_true > 0
P_shuffle / P_true <= 0.80
P_mean / P_true <= 0.80
```

There is no hidden epsilon. Zero denominator, unavailable, bare-null,
malformed, missing, or nonfinite values fail. The artifact reports all five
required primary fields, three-condition geometry diagnostics, frozen donor-
agreement semantics, memory-change/intervention evidence, and deterministic
batch membership.

### Overall decision and repair boundary

Overall C7 passes only if all six arm-level results pass. The final decision
records:

```text
overall_gate_pass
stage6_authorized_by_c7
preauthorized_decoder_repair_triggered
comparison_inconclusive
```

`stage6_authorized_by_c7` equals true only for an overall pass. Any autonomous
tiny or scaled exact-sufficiency failure in either arm triggers only the repair
already authorized by ADR-0004. Memory-only failure sets
`comparison_inconclusive=true` and does not trigger repair. Infrastructure
failure never authorizes repair. C7 cannot implement or invoke the repair,
invent another repair, begin C8, or authorize Stage 6 from Slurm completion
alone.

A completed scientific failure is a valid result: after successful immutable
artifact finalization, the CLI and Slurm job complete successfully while
reporting `overall_gate_pass=false`. Exceptions, incomplete execution,
provenance or environment mismatches, unauthorized access, malformed metrics,
strict-reload failure, and artifact-integrity failure are infrastructure
failures and exit nonzero.

### Immutable publication and CLI

C7 v1 has no automatic resume. Every run targets a new external non-symlink
directory outside the repository and corpus. Work occurs in a same-parent
`.incomplete-<job-id>` staging directory, and atomic rename occurs only after
all planned work and integrity validation completes. A valid scientific gate
failure is finalized; an interrupted directory remains visibly incomplete.

The final artifact contains at least:

```text
resolved_config.json
metrics.jsonl
checkpoints/
artifact_manifest.json
SHA256SUMS
```

Ordinary files are written atomically. `artifact_manifest.json` lists every
ordinary artifact except itself and `SHA256SUMS` by safe relative path, byte
size, and SHA-256. `SHA256SUMS` then covers every regular finalized file except
itself, including the manifest, sorted lexicographically with a final LF. All
sizes and hashes are independently verified before the final directory
rename. Finalized artifacts are immutable by contract.

The narrow CLI requires only `--corpus-dir`, `--output-dir`,
`--repository-root`, and `--expected-commit`. Partition, template, subset,
salt, seed, training, checkpoint, threshold, autonomous semantics, and decoder
choices are governed constants rather than CLI options.

### Authoritative runner

The C7 Slurm runner is CPU-only under Python 3.8.13 and PyTorch 1.11.0 with
one CPU thread. It requires the exact supplied commit and a clean detached or
branch checkout, but no pushed branch or origin identity. Before corpus
access, it runs focused C7 tests with zero PyTorch skips, the complete graph-
encoder suite with zero PyTorch skips, relevant regressions, documentation,
compilation, Python 3.8 grammar, source audits, and Bash syntax validation.

After the pilot it independently verifies the required files, manifest,
checksums, final C7 event, exact commit, and clean source. It prints the
overall, Stage 6 authorization, and repair-trigger values and emits an
execution-completed terminal marker for either scientific pass or scientific
failure. Its failure trap uses `not_confirmed_on_failure` for train access when
an infrastructure failure makes access state unknowable, while every
protected-access flag remains false by construction.

## Alternatives considered

### Select subsets from loaded CAD histories

Rejected. Payload-aware selection could leak geometry or target difficulty
into the gate cohort. Metadata-only ranking is reproducible before payload
access.

### Continue scaled training from tiny checkpoints

Rejected. Warm-starting would entangle two gates and make scaled behavior
dependent on the tiny cohort.

### Treat early-epoch success as sufficient

Rejected. Fixed epoch-50 selection is already frozen. Earlier performance
cannot select or satisfy a C7 checkpoint.

### Stop interventions after scaled exact failure

Rejected. A completed scaled model still supplies governed diagnostic memory-
use evidence, while exact failure independently keeps C7 failed.

### Exit nonzero for a scientific gate failure

Rejected. Scheduler failure would conflate a valid negative result with broken
execution and encourage accidental reruns. Artifact fields, not Slurm state,
govern scientific authorization.

### Reuse and mutate the C6 metrics record

Rejected. The C6 record truthfully identifies an engineering smoke with no C7
activity. Mutating those fields after construction would erase provenance.

## Consequences

- C7 cohorts and hashes are reproducible from authoritative metadata alone.
- Both arms receive matched, fresh, order-independent treatment at each gate.
- Scientific gate failure is preserved without authorizing downstream work or
  masquerading as an infrastructure error.
- The artifact is self-checking and explicitly incomplete until atomic
  finalization.
- C7 implementation and validation can use procedural fixtures without
  opening the formal corpus.
- Stage 6 remains unauthorized until a finalized formal artifact explicitly
  reports `stage6_authorized_by_c7=true`.
- C8 and the hierarchical repair remain unimplemented and uninvoked.

## Validation and evidence

Implementation validation must cover metadata ranking and selected-set hashes,
protected-access exclusion, fresh matched model construction, training
arithmetic, epoch-50 strict reload, per-family exactness, dependency criteria,
gate dependency and repair flags, exact memory thresholds, donor semantics,
artifact finalization and mutation detection, structured terminal events, and
the Slurm contract.

All implementation-time runtime tests use procedural or temporary fixtures.
The formal train-only corpus pilot is a separate later action. Passing local
or Adroit implementation tests does not make C7 complete and produces no C7
scientific pass/fail result.

## Implementation contract

| Item | Frozen value or location |
|---|---|
| Pilot identity | `GE1-C7-PILOT-v1` |
| Gate identity | `GE1-C7-SUFFICIENCY-GATES-v1` |
| Artifact manifest | `GE1-C7-ARTIFACT-MANIFEST-v1` |
| Pilot | `prototype/graph_encoder/pilot.py` |
| Metadata selector | `prototype/graph_encoder/partitions.py` |
| Authoritative runner | `prototype/graph_encoder/adroit/ge1_pilot_cpu.slurm` |
| Tests | `prototype/graph_encoder/tests/test_c7_contract.py` and `test_c7_runtime.py` |

The package README is authoritative for the implemented field-level API. The
accepted Stage 0 preregistration and ADR-0004/0005 remain unchanged.

## Review conditions

Changing the selection salt or bytes, subset sizes, template set, seed,
training recipe, checkpoint, criteria, threshold semantics, failure behavior,
artifact schema, access policy, or repair boundary requires a later ADR.
Implementation review may correct a defect without changing this contract,
but the formal C7 pilot must run only from a reviewed, clean exact commit.
