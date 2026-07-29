# Phase B Constraint-Manifold Replay Preregistration

## 1. Status, authority, and scientific boundary

This document preregisters a read-only diagnostic replay of the immutable
recovered repaired-checkpoint Phase B validation bundle from Slurm job
`3329040`. It is a post-result diagnostic protocol: the frozen Phase B
results were already inspected before this protocol was written.

The original scientific results remain authoritative and immutable:

- teacher-forced Gate C did not pass with 0 / 68 controlled-domain-valid
  families;
- predicted-history Gate C did not pass with 0 / 68 controlled-domain-valid
  families;
- Gate D did not pass with zero qualifying extrude families and zero
  qualifying revolve families;
- Gate E and OpenCascade remain unevaluated; and
- formal final acceptance remains `not_determined` under the frozen report.

No replay result may overwrite, reinterpret, repair, supersede, or be pooled
with those results. Replay outputs are diagnostic counterfactuals, not model
evaluation outputs and not inputs to the frozen Gate A-E decision mapping.

The immutable input namespace is:

```text
/Users/krishaymaskara/research/audited-runs/phase-b-full-validation/
  phase-b-repaired-validation-ce4abca8f450745a8832c0189aaae4a7d8263ec0-3329040
```

The frozen Phase B specification and immutable experiment record remain the
authority for the original run:

- [Phase B evaluation specification](flat_baseline_phase_b_evaluation.md)
- [Recovered full-validation record](../experiments/b0_phase_b_repaired_full_validation.md)

## 2. Input and execution contract

The replay must:

- require the flattened local audited copy at the preregistered bundle path;
  symlink-backed cluster publication directories must be flattened during the
  separately audited download/freeze step;
- verify the recovered bundle's exact 15-entry SHA-256 manifest before
  parsing any scientific artifact;
- verify checkpoint SHA-256
  `282988af00a2dc9a53e14ceb270537d35f85d5339dc989634f88af5f77931267`,
  evaluation commit `ce4abca8f450745a8832c0189aaae4a7d8263ec0`,
  repaired-full contract, schema version 2, and scheduler job `3329040`;
- verify that the conversion and historical-evaluator source bytes equal
  those at the evaluation commit and record their SHA-256 values;
- consume exactly 68 family records in authoritative lexicographic order and
  both stored paths, `teacher_forced` and `predicted_history`;
- reconcile the baseline conversion result byte-for-byte at the JSON-value
  level with the stored conversion result for every family and path;
- call the existing `validate_and_convert_raw_prediction` implementation with
  its unchanged tolerance, ordering, and failure definitions;
- create exactly one replay row for every family, path, and arm, for 544 rows;
- write only to a new, caller-selected, nonexistent output directory through
  collision-safe no-replace publication; and
- publish exactly the six diagnostic artifacts named in Section 7.

The replay must not load a checkpoint, corpus payload, target reconstruction
record, target continuous geometry, model, Torch runtime, OpenCascade, or
test-family payload. It must not run inference, training, Slurm, or kernel
execution.

Only these immutable bundle data may influence transformations:

- raw decoded predictions;
- the model configuration's frozen `max_operations`;
- for Arm 4 only, authoritative `reference_plane` and `primitive_family`
  labels from the per-family metadata.

Authoritative operation-template and profile-family labels may be used after
conversion solely to form the required descriptive strata. They must not be
passed into the transformation functions for Arms 1-3.

## 3. Frozen arms

Arms are applied independently to the original raw prediction. No arm may use
the output of a preceding arm as its input.

### Arm 1: `baseline`

Use the original prediction unchanged. The compatibility-only
teacher-forced provenance adapter used by the frozen evaluator remains in
force before validation. Every conversion field must equal the immutable
stored result.

### Arm 2: `semantic_constants`

Preserve every decoded categorical ID and every unrelated field.

- For a reference-plane node with a real model-decoded plane category,
  replace normalized geometry channels 0-8 with the exact canonical origin
  and frame implied by that decoded category.
- For every axis node, replace normalized geometry channels 33-36 with
  `(0, 0, 0, 1)`.
- Do not alter sketch geometry, operation parameters, masks, categories,
  edges, pointers, node counts, node types, operation metadata, or retained
  logits.

An invalid or sentinel decoded plane category is not repaired.

### Arm 3: `model_category_projection`

Apply Arm 2, then inspect only the model-decoded primitive categories.
Unsupported patterns remain unchanged and retain
`unsupported_profile_pattern`.

Supported profiles use the Euclidean least-squares projection in physical
primitive-coordinate space:

- **Circle:** preserve the decoded center and positive radius exactly. A
  nonpositive radius has no attained nearest point on the strict positive
  radius manifold and is therefore left unchanged with a recorded
  projection failure.
- **Rectangle:** fit `(center_x, center_y, extent)` to all decoded applicable
  endpoints. The projected corners are
  `(cx-E/2, cy-E/4)`, `(cx+E/2, cy-E/4)`,
  `(cx+E/2, cy+E/4)`, and `(cx-E/2, cy+E/4)`. The four line slots connect
  consecutive corners and close the loop.
- **Capsule:** fit `(center_x, center_y, extent)` to all decoded applicable
  line and arc points. Radius is `E/4`; left and right centers are
  `cx-E/4` and `cx+E/4`. Canonical sorted slots are right arc, left arc,
  lower line, and upper line. Arc midpoint and shared endpoint construction
  exactly close the loop.

Rectangle and capsule projections are unavailable when the least-squares
system is singular, the fitted extent is nonpositive, an input is nonfinite,
or any projected normalized channel would leave `[-1, 1]`. Unavailable
projections leave the original sketch unchanged.

The rectangle and capsule design matrices are frozen and data-independent;
their normal matrices are nonsingular by construction. The singular-system
branch remains a defensive contract check. A short geometry vector is
reported separately as `malformed_geometry_width`, not as a nonfinite input.

### Arm 4: `oracle_category_diagnostic`

This arm is explicitly oracle-assisted and is not a model evaluation.

- Replace the predicted reference-plane category with the authoritative
  family plane category.
- Replace every sketch's four primitive categories with the authoritative
  family profile pattern.
- Re-derive geometry applicability masks only where those oracle category
  substitutions require it.
- Construct the exact plane and axis constants.
- Apply the same nearest-profile projection as Arm 3, using raw model
  continuous outputs as the fit observations.

This arm must not substitute target dimensions, centers, extents, continuous
geometry, operation parameters, operation categories, boolean modes,
directions, edges, pointers, node counts, node types, operation sequences, or
histories.

## 4. Transformation evidence

Every attempted field transformation is recorded with:

- family, path, arm, node position, transformation kind, and field name;
- exact channel or categorical indices;
- original and projected values;
- physical L2, RMS, and maximum-absolute residuals where numeric;
- projection method; and
- an explicit reason when projection is unavailable.

Each transformation record includes `is_no_op`. No-op records preserve the
audit trail but are excluded from residual distributions so exact constants
and identity circle projections cannot dilute displacement evidence.

Unsupported model-category patterns in Arm 3 must produce projection-failure
records and remain unsupported. Arm 2 never attempts profile projection.

## 5. Frozen diagnostic metrics

For each path and arm, report:

- attempted count and raw-completion count/rate;
- raw-integrity-valid count/rate;
- reconstruction-target-valid count/rate;
- controlled-domain-valid count/rate;
- primary-failure counts and any-failure counts in frozen failure order;
- invalid-to-valid family transitions;
- transitions grouped by the exact ordered baseline failure-code
  combination;
- family-level and sketch-level supported-profile eligibility;
- residual count, minimum, 25th percentile, median, 75th percentile, 90th
  percentile, 95th percentile, maximum, mean, RMS, and maximum-absolute
  residual, grouped by transformation kind;
- the same validity and transition counts by authoritative operation
  template and authoritative profile family.

Teacher-forced and predicted-history results remain separate.

## 6. Preregistered interpretation rules

These criteria classify diagnostic evidence only:

1. **Replay validity:** any manifest failure, malformed artifact, baseline
   mismatch, row-order mismatch, or output-contract mismatch invalidates the
   entire replay. Partial results are not interpretable.
2. **Semantic-constant sensitivity:** Arm 2 rescuing 17 or more families on a
   path is `strong`; 1-16 is `limited`; zero is `none`.
3. **Constraint-parameterization sensitivity:** Arm 3 rescuing at least
   17 / 68 families on both paths is `strong`; rescuing 1-16 on either path
   or at least 17 on only one path is `limited_or_path_specific`; rescuing
   none is `none`.
4. **Categorical/oracle sensitivity:** Arm 4 adding at least 17 valid
   families over Arm 3 on either path is `strong`; adding 1-16 is `limited`;
   adding none is `none`; a negative difference is `regression`. The combined
   label uses the maximum additional valid-family count over the two paths.
5. **Autoregressive sensitivity:** for any projected arm with at least one
   teacher-forced valid family, predicted-history survival below 0.80 of the
   teacher-forced valid count is `material`; otherwise it is `not_material`
   for that arm. This is diagnostic nomenclature, not the frozen Gate C rule.
6. Residual magnitudes are descriptive. No post-run residual cutoff may be
   introduced to relabel a projection or a family.

The threshold of 17 is the already frozen 25% material-reconstruction floor;
using it here provides a familiar effect-size boundary but does not rerun or
amend Gate C.

## 7. Output contract

A successful replay publishes exactly:

```text
replay_records.jsonl
family_transitions.csv
aggregate_summary.json
projection_failures.csv
run_metadata.json
sha256-manifest.txt
```

The manifest hashes the other five files by stable relative name and excludes
itself. JSON is canonical compact JSON with sorted keys. JSONL and CSV order
are fixed by family ID, path order, arm order, and registered failure or
transformation order. The output namespace is never replaced.

`run_metadata.json` must record the immutable source-manifest digest, all
verified source-entry hashes, tool and preregistration hashes, repository
state, the frozen Gate C/D boundary, explicit leakage prohibitions, and the
fact that no inference, training, OpenCascade, Slurm, test payload, or target
continuous geometry was accessed. It also records dot-prefixed root entry
names and the verified validator-source hashes. The actual no-replace
publication backend is known only after the already-hashed namespace is
published and is therefore reported by the command on standard output; it
cannot truthfully be embedded into that namespace without a second
publication transaction.

## 8. Decision after replay

The first completed replay receives a new immutable experiment record. If
Arm 3 shows strong constraint-parameterization sensitivity, the next
implementation milestone is a separately reviewed constrained or factorized
decoder experiment. If Arm 4 shows strong additional categorical sensitivity,
that milestone must also address categorical or latent representation. If
Arm 4 remains below 17 valid families on either path, broader model-output
failures must be diagnosed before training or kernel execution.

Under every outcome, the original Gate C and Gate D failures remain frozen.
