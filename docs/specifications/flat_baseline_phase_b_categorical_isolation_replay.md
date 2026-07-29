# Phase B Categorical-Isolation Replay Preregistration

## 1. Status and immutable scientific boundary

This document freezes, before implementation or execution, a read-only
2-by-2 factorial diagnostic over the completed Phase B constraint-manifold
replay. The factors are use of the model-decoded versus authoritative
reference-plane category and use of the model-decoded versus authoritative
profile-family category.

The diagnostic cannot alter or reinterpret any prior result. The original
teacher-forced and predicted-history Gate C results remain frozen failures,
Gate D remains a frozen failure, and Gate E/OpenCascade remain unevaluated.
Every output is diagnostic only and is not a model-evaluation result.

The two immutable inputs are:

```text
/Users/krishaymaskara/research/audited-runs/phase-b-full-validation/
  phase-b-repaired-validation-ce4abca8f450745a8832c0189aaae4a7d8263ec0-3329040

/Users/krishaymaskara/research/audited-runs/phase-b-full-validation/
  constraint-manifold-replay-3329040-preregistered-v1
```

The source evaluation bundle must retain its verified 15-entry manifest with
SHA-256
`a13d5f35b345b6814912145a56e0cce0290ae677b030a1215b61e92558862541`.
The completed replay must retain its exact five-entry manifest with SHA-256:

```text
6ff46389fb3947b3448e5e52860833d0592e83f70849536ddfd4aeb32f73b4b2
```

Neither input namespace may be modified, overwritten, republished, or used as
an output location.

## 2. Allowed information and provenance

Before parsing scientific records, the implementation must independently
validate both immutable manifests and their exact artifact sets. It must then:

- require the recovered evaluation identity frozen by the parent replay:
  checkpoint epoch 44, global step 748, checkpoint SHA-256
  `282988af00a2dc9a53e14ceb270537d35f85d5339dc989634f88af5f77931267`,
  evaluation commit `ce4abca8f450745a8832c0189aaae4a7d8263ec0`,
  schema 2, repaired-full contract, and Slurm job `3329040`;
- require the completed replay's recorded repository commit
  `e0829773589a07ab471bc3932bf91578107b5467`;
- consume exactly 68 lexicographically ordered validation families and both
  paths, for 136 original raw predictions;
- verify that the parent replay contains exactly one record per family, path,
  and parent arm, in its frozen order;
- use the unchanged controlled-domain validator and projection mathematics;
  and
- revalidate both input manifests after all artifacts have been constructed
  but before publication.

Only the original raw prediction, frozen `max_operations`, and the following
two authoritative categorical labels may influence a transformation:

- `reference_plane`; and
- `primitive_family`.

The authoritative operation template and profile family may additionally be
used after conversion to form registered strata. No arm may access target
continuous geometry, dimensions, centers, extents, operation parameters,
edges, pointers, node counts, node types, operation histories, target
histories, corpus payloads, checkpoint payloads, model state, or test-family
payloads.

Plane-only transformation code receives a one-field plane capability.
Profile-only transformation code receives a one-field profile capability.
The two capabilities are distinct frozen types so that the unused
authoritative category is structurally inaccessible.

## 3. Frozen factorial arms

Every arm starts independently from a deep copy of the same original raw
prediction. No arm may receive or observe another arm's output.

### `model_plane_model_profile`

- Preserve the model-decoded reference-plane category.
- Preserve the model-decoded primitive categories.
- Construct the exact plane frame implied by the model category and the exact
  controlled-domain axis.
- Project supported model-decoded circle, rectangle, or capsule geometry
  using raw model geometry as observations.
- Leave unsupported patterns unsupported.

This arm must reconcile exactly, after arm-name normalization, with the
completed replay's `model_category_projection` record.

### `oracle_plane_model_profile`

- Replace only the reference-plane category with authoritative
  `reference_plane`.
- Re-derive only reference-plane applicability channels and construct its
  exact canonical frame.
- Preserve all model-decoded profile categories and profile applicability
  channels.
- Project only profiles supported by those model-decoded categories.
- Construct the exact controlled-domain axis.

This arm must not receive or inspect `primitive_family`.

### `model_plane_oracle_profile`

- Preserve the model-decoded reference-plane category and construct the exact
  frame implied by it.
- Replace only the four primitive categories with the authoritative
  `primitive_family` pattern.
- Re-derive only sketch profile-applicability channels 9 through 32.
- Generate the exact supported-profile manifold using raw model continuous
  geometry as fitting observations.
- Construct the exact controlled-domain axis.

This arm must not receive or inspect authoritative `reference_plane`.

### `oracle_plane_oracle_profile`

- Replace the reference-plane category with authoritative
  `reference_plane`.
- Replace the primitive categories with authoritative `primitive_family`.
- Re-derive only the applicability channels required by those substitutions.
- Construct exact plane and axis constants and project the authoritative
  supported profile using raw model geometry as observations.

This arm must reconcile exactly, after arm-name normalization, with the
completed replay's `oracle_category_diagnostic` record.

Projection unavailability retains the parent replay's exact mathematical
meaning and reason strings. No tolerance, geometry scale, primitive ordering,
or validity failure definition may change.

## 4. Row and metric contract

The replay publishes exactly one row per family, path, and arm: 544 rows in
lexicographic family order, then `teacher_forced`, `predicted_history`, then
the arm order in Section 3.

For each path and arm, report:

- attempted and controlled-domain-valid counts and valid rate;
- invalid-to-valid transitions relative to
  `model_plane_model_profile`;
- valid-to-invalid regressions relative to
  `model_plane_model_profile`;
- exact ordered controlled-domain failure-code combinations;
- projection-unavailability reason counts;
- the same attempted, valid, invalid-to-valid, and valid-to-invalid counts by
  authoritative operation template;
- the same counts by authoritative profile family; and
- the same counts split by the first authoritative operation, exactly `E` or
  `R`.

Teacher-forced and predicted-history metrics remain separate.

For each path, compute the controlled-domain-valid-count factorial effects:

```text
plane_only_gain = oracle_plane_model_profile - model_plane_model_profile
profile_only_gain = model_plane_oracle_profile - model_plane_model_profile
combined_gain = oracle_plane_oracle_profile - model_plane_model_profile
interaction = oracle_plane_oracle_profile
              - oracle_plane_model_profile
              - model_plane_oracle_profile
              + model_plane_model_profile
```

The five already observed predicted-history `EE` capsule failures with
`nonpositive_fitted_extent` in the parent full-oracle arm are a pinned audit
cohort:

```text
sf_7b44d531d7f3646bd54685ebfa22a0cf2aca0115aa4eb5ac1cebae7f9375375e
sf_85bfcef3e95d63fcc36a4cdeae8497a2909cabe3c58eed3687a9efd917b3aeaf
sf_96d5102c2a138cd471e8699841125d3da5953645df906f86c3e23ccb1e97983a
sf_a4071c6e4d9144b58494e0e9d7a6322421789d448827bbef61532416671576de
sf_c29d2f2ca87aeffe549395ce758d0d77b8435a69d5c3cdfb1b7e5d9c4a8a1caf
```

For each cohort member, record every arm in which
`nonpositive_fitted_extent` occurs and the first such arm in registered arm
order. The parent replay must independently confirm all five IDs, path,
operation template, profile family, node position, and reason before the new
analysis begins.

## 5. Interpretation rules

These rules describe diagnostic counts only:

1. A positive plane-only gain is evidence of reference-plane categorical
   sensitivity conditional on model profile categories; zero is no observed
   gain; a negative value is a regression.
2. A positive profile-only gain is evidence of profile-family categorical
   sensitivity conditional on model plane categories; zero is no observed
   gain; a negative value is a regression.
3. A positive combined gain is evidence that the joint authoritative
   categories improve diagnostic validity; it does not measure model
   performance.
4. Positive interaction denotes super-additive joint validity, negative
   interaction denotes sub-additivity or antagonism, and zero denotes
   additivity on the count scale.
5. The two isolated arms are parallel counterfactuals, not a temporal
   sequence. “First appearance” for the capsule cohort means first in the
   frozen reporting order only.
6. No count, gain, interaction, or isolated replay outcome can amend Gate C,
   Gate D, final acceptance, or Gate E.

No post-run threshold or alternative arm ordering may be introduced.

## 6. Artifact and publication contract

The preregistered output namespace is:

```text
/Users/krishaymaskara/research/audited-runs/phase-b-full-validation/
  categorical-isolation-replay-3329040-preregistered-v1
```

It must not exist before execution and must be published exactly once. A
successful new collision-safe namespace contains exactly:

```text
factorial_records.jsonl
family_transitions.csv
aggregate_summary.json
projection_failures.csv
capsule_regressions.json
run_metadata.json
sha256-manifest.txt
```

The manifest hashes the other six artifacts by stable relative name and
excludes itself. JSON uses canonical compact sorted-key serialization. JSONL
and CSV order follow the frozen family, path, arm, failure, and reason orders.
Publication is fsynced, atomic, no-replace, and reports its actual backend
after success.

`run_metadata.json` records both immutable input paths, manifest digests and
entry hashes, source evaluation identity, parent replay identity, validator
source hashes, tool and preregistration hashes, current repository state,
explicit oracle capabilities per arm, and the frozen Gate C/D/E boundary.

Any manifest mismatch, source mutation, parent-arm reconciliation mismatch,
cohort mismatch, row/order mismatch, arithmetic mismatch, output collision,
or artifact-set mismatch invalidates the entire diagnostic. Partial outputs
are not interpretable and must not be published.
