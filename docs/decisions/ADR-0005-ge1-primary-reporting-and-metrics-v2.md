# ADR-0005: GE1 Primary Reporting and Metrics V2 Addendum

- Status: `accepted`
- Proposal date: `2026-08-08`
- Decision date: `2026-08-08`
- Owner: project research team
- Designated GE1 reviewer: Krishay Maskara
- Adds to: [ADR-0004](ADR-0004-ge1-single-manifest-encoder-comparison.md)
- Supersedes: no frozen scientific choice in ADR-0004 or `GE1-STAGE0-PREREG-v1`
- Superseded by: none

## Context

The frozen Stage 0 record states that changing a frozen choice requires a
superseding ADR. After C6 passed authoritative job `3344367`, implementation
review identified an unrecorded reporting obligation: every reported primary
result needs the five frozen memory-intervention values beside it. The shared
decoder retains four output-side serialized-position signals, and the frozen
position-only scoring prior recovered 59/68 exact graphs. A primary endpoint
without memory-use evidence is therefore incomplete evidence about encoder
contribution.

The same review added donor-template diagnostics and found three details that
must be explicit before emitting another artifact:

1. undefined intervention values need reason-bearing structured nulls rather
   than bare `null` values;
2. only `P_shuffle` has a donor family—`P_true` is self-memory and `P_mean`
   uses synthetic `batch_mean:<identity>` memory; and
3. a random donor comparison must respect the no-self-donor constraint.

These additions change the metrics artifact shape and the required
`complete_metrics_record` input. Reusing `GE1-C6-METRICS-v1` would make the
validated C6 artifacts and new artifacts share an identity despite different
contracts. The designated reviewer required an accepted addendum instead of
editing the frozen preregistration silently.

## Decision

### Primary reporting completeness

Every artifact that reports a GE1 primary result must include all five frozen
memory-intervention values:

```text
P_true
P_shuffle
P_mean
R_shuffle
R_mean
```

Each value must be either a finite number or a structured undefined value of
the form `{"value": null, "reason": "<nonempty reason>"}`. A missing key,
bare `null`, empty reason, Boolean, or nonfinite number makes the primary
report incomplete and prevents publication.

This is `GE1-PRIMARY-REPORT-v1`. It is a reporting obligation only. It changes
no decoder input or output position signal, intervention, 0.80 memory-use
threshold, endpoint, checkpoint rule, partition policy, or decision threshold.

### Donor-template diagnostic

Donor-template agreement is available only for `P_shuffle`, where every
recipient has one distinct physical-family donor. `P_true` and `P_mean` must
emit structured unavailable donor-agreement records; a synthetic
`batch_mean:<identity>` must never be resolved through the family-template
map.

For recipient template counts `n_t` and `N` assignments, the chance baseline
is a uniformly random distinct-family donor:

```text
sum_t n_t (n_t - 1) / (N (N - 1))
```

The report names this baseline, preserves observed agreement and excess over
chance, and does not alter the frozen cyclic derangement.

### Artifact and smoke identities

New complete metrics artifacts use `GE1-C6-METRICS-v2` and carry
`primary_reporting_contract=GE1-PRIMARY-REPORT-v1`. The train-only runner that
emits them uses `GE1-C6-TRAIN-ONLY-SMOKE-v2`.

Historical v1 artifacts, including authoritative job `3344367`, remain valid
evidence for the C6 gate they ran. They are not rewritten or relabeled as v2.
The current C5/C6 source requires a new clean exact-commit validation before
C7 because the earlier jobs do not cover the review fixes or v2 artifact.

## Alternatives considered

### Insert the obligation directly into `GE1-STAGE0-PREREG-v1`

Rejected. That record is frozen and explicitly requires a later ADR for a
frozen-choice change. Silent insertion would erase the chronology of the
decision even though the thresholds themselves remain unchanged.

### Keep `GE1-C6-METRICS-v1`

Rejected. The donor block, primary-report contract, and required template map
make the new artifact structurally and semantically distinct from validated
v1 artifacts.

### Keep the independent-draw template baseline

Rejected. `P_shuffle` forbids self-donors, so `sum_t (n_t/N)^2` answers a
different question and yields 1/2 rather than the correct 1/3 for a balanced
2/2 example.

### Treat `P_mean` identifiers as donor families

Rejected. Batch means are synthetic memory sources, not physical families,
and cannot have operation-template metadata.

## Consequences

- A primary result cannot be published without an explicit memory-use record.
- `P_shuffle` donor-template agreement is interpretable against the correct
  no-self baseline.
- `P_true` and `P_mean` donor agreement remain explicit but unavailable rather
  than misleading or error-prone.
- `complete_metrics_record` requires `templates_by_family`; callers written
  for v1 must migrate explicitly.
- Existing v1 artifacts remain historically accurate and distinguishable.
- Current C5/C6 source is not authoritatively validated until the new exact-
  commit Adroit gate passes.

## Validation and evidence

Required validation includes:

- pure tests for all five required intervention values and structured nulls;
- a real `batch_mean:<identity>` donor-unavailability test;
- exact no-self chance-baseline tests;
- a complete v2 metrics-record test covering all three memory conditions;
- the C5 compatibility-shim provenance integration test;
- the shortened-count prefix bookkeeping integration test;
- every new C5/C6 real-PyTorch test with zero skips on Adroit;
- the complete graph-encoder suite and both 407-family train-only smokes; and
- artifact inspection confirming the v2 schema, five intervention values,
  available `P_shuffle` donor agreement, and structured-unavailable `P_true`
  and `P_mean` donor agreement.

## Implementation contract

Affected identities and modules:

| Item | Frozen value or location |
|---|---|
| Primary reporting | `GE1-PRIMARY-REPORT-v1` |
| Complete metrics schema | `GE1-C6-METRICS-v2` |
| Train-only smoke | `GE1-C6-TRAIN-ONLY-SMOKE-v2` |
| Enforcement and donor diagnostic | `prototype/graph_encoder/metrics.py` |
| Authorized caller | `prototype/graph_encoder/c6_smoke.py` |
| Tests | `prototype/graph_encoder/tests/test_c6_contract.py` and `test_c6_runtime.py` |

The package README is authoritative for the current field-level API. The
accepted Stage 0 preregistration remains unchanged.

## Review conditions

Any later change to the five required values, structured-null form, donor
semantics, chance baseline, intervention algorithm, metric schema, memory-use
threshold, or primary endpoint requires another ADR. Adding a new caller of
`complete_metrics_record` requires a test proving that it supplies the complete
scored-family template map without opening unauthorized partitions.
