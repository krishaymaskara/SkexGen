# GE1 Closed-Form Representation-Readout Contract

## Status and authority

| Item | Frozen value |
|---|---|
| Status | Accepted v2 contract correction, synthetic validation, runner, and exact-source preparation |
| Protocol | `GE1-C7-CLOSED-FORM-READOUT-v2` |
| Artifact | `GE1-C7-CLOSED-FORM-READOUT-ARTIFACT-v2` |
| Results | `GE1-C7-CLOSED-FORM-READOUT-RESULTS-v2` |
| Source | Three detached files preserved by timed-out job `3346513` |
| Runtime | Adroit CPU, Python 3.8.13, PyTorch 1.11.0, one thread, one hour |
| ADR-0011 / screen | Accepted and incomplete / separate and incomplete |
| Scientific execution | V1 job `3351501` failed without a result; v2 is not yet submitted |

This is an observational accessibility diagnostic. Every result must carry
the complete limitations and non-authorization block from
[ADR-0012](../decisions/ADR-0012-ge1-closed-form-representation-readout.md).

V1 is immutable failed audit history. Job `3351501` at commit
`912e078f216f82ad37c00198adc20dd93eb647c6` failed with exit `1:0` after
`00:06:28` and MaxRSS `611492K`, after hash verification and loading of all
three frozen inputs. Its B=999 projection was `671.2964434385067` seconds.
No finalized scientific result exists. Preserve its stdout/stderr, timing
record, and incomplete directory without deletion or mutation. V2 is a
required identity bump because the repaired common scalar baseline affects
primary difference hypotheses and the protocol-derived permutation seeds.

## Immutable inputs and access order

Only these regular, non-symlink files may be read:

| File | SHA-256 |
|---|---|
| `detached_features.pt` | `87dab4841f64f0c1b27db6676852da7cd980ba1bde61e81809fde3af745cc01c` |
| `feature_manifest.json` | `29220bda0677636627069ce6565bf367065f1ceaa3c2d86d3174058838598450` |
| `labels.json` | `70f82727ebe35613bcbae1267924d9a327d9744a757b6d076c8028e261cfe71f` |

Verify all three hashes, the target-free manifest, the 96-row feature payload,
finite A/B/C/D values and dimensions, and the absence of target/model/optimizer
state. Release the feature payload object before parsing labels. Then verify 48
labels, physical grids, family/template structure, one feature row per arm and
label, and identity alignment before joining labels in memory.

Preflights and synthetic timing run with no input bind. The analysis bind adds
only this read-only directory to the clean source and output parent. No corpus,
manifest, repaired result, checkpoint, model artifact, or protected partition
may be mounted.

## Cohorts

There are eight E, eight R, eight EE, and eight RE families. E has extrusion
slot 0; R has revolve slot 0; EE has extrusion slots 0 and 1; RE has revolve
slot 0 and extrusion slot 1. Thus extrusion has 32 operations in 24 families,
revolve has 16 in 16 families, only EE duplicates one type, and every revolve
is slot 0.

The complete full cohort is primary. The secondary extrusion sensitivity
selects E+RE: 16 families with exactly one extrusion each. Revolve sensitivity
is a structured `not_applicable_full_cohort_already_one_to_one` record.

## Scalar baseline

Run the existing A-physical and B-raw-logit scalar definitions, physical
fidelity scoring, monotonic resubstitution ceilings, and coordinate-specific
physical-family LOFO grouped thresholds. Report their raw/balanced accuracy,
confusion, fixed-five-class support/recall, and predictions descriptively.
Never choose between their coordinate-specific results for a primary test.

Before using labels, construct one common finite-precision weak order. Reject
any pair whose strict order in A is reversed in B. Collapse exact ties in
either coordinate transitively; sigmoid saturation therefore removes a
distinction from the common path even if B retains it. The resulting ordered
blocks contain exactly the distinctions preserved by both coordinates.

For each LOFO training fold, candidate class boundaries are negative infinity,
positive infinity, or an upper common-order anchor. Numeric A/B midpoints are
forbidden. A held-out block between adjacent training anchors remains on the
smaller-class side until the upper anchor, prospectively freezing the
same-gap ambiguity. Fit four nondecreasing boundaries by the existing exact
five-class dynamic-programming objective and lexicographically earliest
boundary tie break.

The label-free common order is frozen once per arm/type cohort. Refit the four
supervised boundaries for the observed target and every synchronized
permutation. Emit identical A and B audit views from this single fit. This
result is the only `A/B grouped scalar` baseline used in C-minus-A/B primary
differences; the separate A and B fits remain descriptive.

## Features and execution matrix

All values come only from detached rows.

| Feature | Operation | Definition | Nominal dim. | Role |
|---|---|---|---:|---|
| C | both | operation decoder state | 32 | primary |
| P | both | first 32 D dimensions; no context | 32 | primary for revolve, secondary for extrusion |
| D-additive | extrusion | `[P, slot_0, slot_1]` | 34 | secondary structural control |
| D-gated | extrusion | `[P*slot_0, P*slot_1, slot_0, slot_1]` | 66 | primary upstream extrusion readout |
| context-only | extrusion | `[slot_0, slot_1]` | 2 | secondary negative/context control |

No constant type one-hot is added because types are analyzed separately.
D-additive failure cannot establish absence in P. D-gated permits distinct
slot mappings, at the cost of about half-cohort support per 32-column block
and roughly 30 outer-training operations against 66 nominal columns.

For revolve, context-only is constant and D-additive/D-gated equal P after
training-fold zero-variance removal. Record all three as
`degenerate_by_construction`; do not fit or count them.

The executed matrix contains 24 pipelines: five full extrusion and five
E+RE-sensitivity extrusion features per arm, plus C and P full revolve per
arm. Six revolve feature degeneracies and two revolve sensitivity
non-applicabilities are explicit structured records.

## Estimator and preprocessing

For n observations, five-column one-hot Y, and unpenalized b, minimize:

```text
(1 / (2n)) * ||Y - XW - b||_F^2 + (lambda / 2) * ||W||_F^2
```

Use CPU float64 and retain all five outputs even if a training fold lacks a
class. Predict with argmax; native first-maximum semantics freezes the
smallest-class-index tie break.

Within every training fold:

1. compute population variance and drop exactly zero-variance columns;
2. center all retained columns;
3. divide continuous columns by their population standard deviation;
4. do not variance-scale binary slots; and
5. apply those retained columns, means, and scales to the test fold.

Record retained indices/dimension, dropped columns, column kinds, means, and
scales per outer fold. No full-cohort or test statistic may enter a fold.

After centering X and Y, solve with the thin SVD and ridge term `n*lambda`.
One label-independent SVD serves all seven lambdas and all batched true and
permuted right-hand sides. An uncached primal solve exists only as a synthetic
test oracle. Scores and predictions must agree within a strict float64
tolerance.

## Splits and selection

The outer split holds out one physical family. For each outer training set,
inner physical-family LOFO evaluates lambdas:

```text
1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100
```

Choose maximum balanced accuracy, then maximum raw accuracy, then the largest
lambda. Apply the selected lambda to that outer fold. Run the same nested rule
on the complete cohort for a clearly labeled non-generalizing resubstitution
record.

All fit and aggregate metrics include a 5x5 confusion matrix, true support,
predicted count, recall, and `class_masking_detected` when any supported class
has zero predictions. Masked negative ridge results carry an explicit limit.

## Permutations and timing ladder

For each operation type and permutation index, derive one seed from protocol,
2026, operation type, and index only. Within each template, shuffle complete
family label blocks. Preserve class support and the ordered EE pair. Apply the
same map to both arms and every feature; restrict the full extrusion map for
the E+RE sensitivity. Pair extrusion/revolve permutation indices. Refit the
common ordered boundaries and nested lambda selection for every permuted
target. Only label-free common ranks, X preprocessing, and SVDs are reusable.

Synthetic timing uses actual cohort/fold counts, dimensions 2/32/34/66,
batched targets, all executed/structured combinations, metrics, and maxT. Its
projection includes elapsed repository-preflight time, a 1.25 analysis safety
factor, and 180 seconds for finalization/verification. Select only B=999 when
projected at most 2700 seconds; otherwise B=499 under the same limit; otherwise
abort before input access. Empirical p-values use
`(1 + count(null >= observed)) / (B + 1)`.

## Primary family and maxT

Exactly 16 hypotheses are primary. For each arm:

- extrusion: C access, D-gated access, C minus A/B, D-gated minus C;
- revolve: C access, P access, C minus A/B, P minus C.

Raw and balanced accuracy form separate multiplicity families. For hypothesis
j, subtract its null mean from observed and null values. At permutation b,
take the maximum centered value across all 16. The plus-one adjusted p-value
compares the observed centered statistic with these maxima. Report observed,
null count/mean/population SD/nearest-rank p95/maximum, raw p, centered value,
adjusted p, and maxT family size. Each difference uses the two pipelines from
the same permutation, never an observed fixed baseline.

Secondary pipelines receive marginal permutation summaries and an explicit
`secondary_not_in_primary_family` maxT status. Step-down and kernel ridge are
excluded.

## Prospective interpretations

Accessibility requires LOFO balanced accuracy at least 0.40, raw accuracy
strictly above marginal null p95, and raw and balanced maxT-adjusted p at most
0.05.

C materially exceeds A/B only if C passes accessibility, both C-minus-A/B
metrics are at least 0.10, and both differences have adjusted p at most 0.05.
The licensed statement is: “Magnitude-class information reaches the operation
decoder state but is not exploited by the existing scalar output path.”

A conditioning candidate requires D-gated (extrusion) or P (revolve) access,
no C access, both upstream-minus-C metrics at least 0.10, and both adjusted p
at most 0.05. The licensed statement identifies the transformation or
conditioning as a candidate region, not proven information destruction.

D-additive failure with D-gated success means the additive family readout was
structurally mismatched to slots. Context-only remains descriptive; comparable
performance prevents representation-only attribution. Arm differences are
arm-specific accessibility only. If every primary is null, report only: “No
controlled held-out-family accessibility evidence was found under the tested
readouts.” Strong resubstitution with null LOFO is within-cohort fit or
memorization.

## Results, telemetry, and artifact

Each executed pipeline reports identity/dimension, fold preprocessing,
lambdas, family identities, observed LOFO/resubstitution predictions and full
metrics, masking, permutation summaries, timing, and factorization counts.
Top-level results report scalar analyses, 16 corrected hypotheses, combined
descriptive operation and family summaries, structured nonexecuted records,
interpretations, limitations, access, and authority.

Flush JSON telemetry for job/environment/repository preflights, timing start
and completion, B selection/gate, input verification, each pipeline,
permutation-target completion, validation, and terminal status. Once analysis
starts, a killed job reports input access as
`indeterminate_not_terminally_certified` unless terminal telemetry certifies
otherwise.

Write in a unique `.incomplete-<job>` directory and fsync atomic files. A
complete artifact contains exactly:

```text
resolved_config.json
readout_results.json
metrics.jsonl
artifact_manifest.json
SHA256SUMS
```

Validate pipeline coverage, schemas, byte sizes, hashes, checksum coverage,
terminal telemetry, runtime, and authority before atomically renaming staging.
Incomplete paths are non-authoritative. A complete negative result exits zero.

## Non-authorization

No result identifies an exact bug; proves information absent, capacity
exhausted, or latent dimensions binding; predicts 48/48; establishes encoder
superiority; or authorizes a categorical head, other repair, Stage 6, C8, or
protected access. Scientific submission/execution requires separate authority.
