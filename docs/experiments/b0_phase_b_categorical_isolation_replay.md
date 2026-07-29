# B0 Phase B Categorical-Isolation Replay

## Record status

| Item | Value |
|---|---|
| Experiment ID | `b0-phase-b-categorical-isolation-3329040-v1` |
| Record status | `verified` |
| Evidence state | `verified-local` |
| Scientific role | Preregistered validation-only categorical diagnosis |
| Implementation commit | `60325f9a3872bd82c783c687b68a7023bccc2d1b` |
| Implementation working tree | Clean |
| Parent replay implementation | `e0829773589a07ab471bc3932bf91578107b5467` |
| Evaluation commit | `ce4abca8f450745a8832c0189aaae4a7d8263ec0` |
| Execution mode | Local, read-only replay over immutable artifacts |
| Execution date | July 29, 2026 |

This is an immutable diagnostic record. It does not amend the original Phase B
evaluation, its frozen Gate C and Gate D failures, its undetermined formal
acceptance, or the unevaluated Gate E state.

## Objective and frozen scientific question

The preregistered question was whether the controlled-domain-validity gain in
the completed constraint-manifold replay came from correcting the
reference-plane category, correcting the profile-family category, or their
interaction. A 2-by-2 factorial replay independently combined model versus
authoritative reference-plane categories with model versus authoritative
profile-family categories.

All four arms started from the original raw prediction. Oracle access was
limited to `reference_plane` and `primitive_family`. The replay could not use
target continuous geometry, dimensions, operation parameters, edges,
pointers, node counts, histories, corpus payloads, model state, or test
payloads.

The interpretation rules, arm order, arithmetic, immutable inputs, and output
namespace were frozen before execution in the
[categorical-isolation preregistration](../specifications/flat_baseline_phase_b_categorical_isolation_replay.md).

## Inputs and provenance

| Item | Verified value |
|---|---|
| Immutable source bundle | `/Users/krishaymaskara/research/audited-runs/phase-b-full-validation/phase-b-repaired-validation-ce4abca8f450745a8832c0189aaae4a7d8263ec0-3329040` |
| Source manifest SHA-256 | `a13d5f35b345b6814912145a56e0cce0290ae677b030a1215b61e92558862541` |
| Source checkpoint | Epoch 44, global step 748 |
| Checkpoint SHA-256 | `282988af00a2dc9a53e14ceb270537d35f85d5339dc989634f88af5f77931267` |
| Immutable parent replay | `/Users/krishaymaskara/research/audited-runs/phase-b-full-validation/constraint-manifold-replay-3329040-preregistered-v1` |
| Parent replay manifest SHA-256 | `6ff46389fb3947b3448e5e52860833d0592e83f70849536ddfd4aeb32f73b4b2` |
| Parent replay implementation SHA-256 | `82a77271d70fc123bf1b24c5d3a1818d386ad11ba830baab590bfd2f4bf33a2e` |
| Categorical-isolation implementation | `60325f9a3872bd82c783c687b68a7023bccc2d1b` |
| Tool SHA-256 | `43f06ba580640f3867649d8ef2ce43cc0691f29e8e5299e5019fc7ce328679ca` |
| Preregistration SHA-256 | `da33e8c7ad728058c53c0dcb5f4cb7122202470a189669e458d69f7b4b73e491` |
| Validator-source aggregate SHA-256 | `7f1bd7ba5b9880a4b86148a4332f00f4c7675d2030d84745653bfb5c8835a470` |
| Immutable output namespace | `/Users/krishaymaskara/research/audited-runs/phase-b-full-validation/categorical-isolation-replay-3329040-preregistered-v1` |
| Output manifest SHA-256 | `e9c353a98fd37f5a491c34970d1859c73d61ebb7341d1ac14ca020a6f51aea63` |

The run metadata records a clean implementation checkout at the stated
commit. It also records no checkpoint/model loading, corpus access, target
continuous-geometry access, test-family access, inference, training, Slurm,
Torch runtime, OpenCascade, or kernel execution.

The source bundle contains exactly the 68 authoritative validation families.
The original repaired-checkpoint evaluation loaded zero train families and
zero test families and did not evaluate the test partition. This diagnostic
consumed only the already published source predictions and parent replay.

## Execution and artifact integrity

The committed runner validated the exact source and parent manifests before
scientific parsing, reconciled its model/model and full-oracle arms with the
corresponding immutable parent records, constructed all four arms
independently, revalidated both inputs, and published to the preregistered
collision-safe namespace.

Independent local inspection verified:

- exactly six entries in `sha256-manifest.txt`;
- the exact required six-file manifest set with no missing or duplicate path;
- all six SHA-256 values against the local artifact bytes;
- manifest SHA-256
  `e9c353a98fd37f5a491c34970d1859c73d61ebb7341d1ac14ca020a6f51aea63`;
- exactly 544 lines in `factorial_records.jsonl`;
- all 544 JSONL rows parsed successfully; and
- metadata implementation commit
  `60325f9a3872bd82c783c687b68a7023bccc2d1b` with
  `repository_dirty: false`.

## Verified arm results

### Teacher-forced

| Reference-plane category | Profile-family category | Controlled-domain valid |
|---|---|---:|
| Model | Model | 36 / 68 |
| Authoritative | Model | 36 / 68 |
| Model | Authoritative | 68 / 68 |
| Authoritative | Authoritative | 68 / 68 |

### Predicted-history

| Reference-plane category | Profile-family category | Controlled-domain valid |
|---|---|---:|
| Model | Model | 29 / 68 |
| Authoritative | Model | 29 / 68 |
| Model | Authoritative | 63 / 68 |
| Authoritative | Authoritative | 63 / 68 |

Relative to model/model, the predicted-history authoritative-profile arm
rescued 39 previously invalid families and regressed five previously valid
families, for a net gain of 34. The teacher-forced authoritative-profile arm
rescued 32 families with no regressions.

## Factorial effects

| Effect on controlled-domain-valid count | Teacher-forced | Predicted-history |
|---|---:|---:|
| Plane-only gain | 0 | 0 |
| Profile-only gain | +32 | +34 |
| Combined gain | +32 | +34 |
| Plane/profile interaction | 0 | 0 |

Authoritative reference-plane categories produced no validity gain with model
profiles or authoritative profiles. The full-oracle result exactly equaled
the corresponding model-plane/oracle-profile result on both paths, yielding
zero count-scale interaction.

## First-operation result

The first-operation stratum exposes where the predicted-history gain and
regressions occurred:

| Arm | E-first valid | R-first valid |
|---|---:|---:|
| Model plane / model profile | 29 / 32 | 0 / 36 |
| Model plane / authoritative profile | 27 / 32 | 36 / 36 |

The authoritative-profile arm repaired all 36 R-first families. Within the
32 E-first families it changed 29 / 32 valid to 27 / 32 valid: three
previously invalid families were rescued and five previously valid families
regressed.

## Five predicted-history capsule regressions

The five regressions were all `EE`, authoritative
`capsule_line_arc` families. Each first acquired
`nonpositive_fitted_extent` at node position 4 in
`model_plane_oracle_profile`; the same failure remained in
`oracle_plane_oracle_profile`. Neither model-profile arm produced that
failure.

| Family ID | First failing isolated arm |
|---|---|
| `sf_7b44d531d7f3646bd54685ebfa22a0cf2aca0115aa4eb5ac1cebae7f9375375e` | `model_plane_oracle_profile` |
| `sf_85bfcef3e95d63fcc36a4cdeae8497a2909cabe3c58eed3687a9efd917b3aeaf` | `model_plane_oracle_profile` |
| `sf_96d5102c2a138cd471e8699841125d3da5953645df906f86c3e23ccb1e97983a` | `model_plane_oracle_profile` |
| `sf_a4071c6e4d9144b58494e0e9d7a6322421789d448827bbef61532416671576de` | `model_plane_oracle_profile` |
| `sf_c29d2f2ca87aeffe549395ce758d0d77b8435a69d5c3cdfb1b7e5d9c4a8a1caf` | `model_plane_oracle_profile` |

The fitted extent is constrained to be strictly positive. For these five
second-operation capsule sketches, the raw predicted-history geometry had no
attained nearest point under that constraint, so projection remained
unavailable rather than silently changing the failure definition.

## Decision and scientific interpretation

Within this preregistered diagnostic, reference-plane categorical correction
has no controlled-validity effect. The entire net gain comes from the
authoritative-profile arms, which jointly substitute the profile-family
category and project the raw continuous coordinates onto the corresponding
controlled profile manifold.

The evidence therefore attributes the observed controlled-validity failure
to profile-family categorical errors together with unconstrained profile
geometry, not to reference-plane category selection. Because the
authoritative-profile intervention couples category substitution with
category-specific geometric projection, it does not estimate a
category-only effect separate from geometry. Instead, it shows that profile
categories and continuous profile geometry must be generated jointly under a
shared constraint.

The next bounded intervention is a category-conditioned constrained profile
decoder. Its category decision must determine the applicable primitive
geometry parameterization, and its continuous output must live on or project
through the selected profile manifold. The five second-operation capsule
failures must remain an explicit adversarial case.

## Limitations and claims not supported

- Authoritative categories are oracle diagnostics, not valid autonomous model
  outputs and not a checkpoint-evaluation score.
- No target continuous geometry, target dimensions, operation parameters,
  edges, pointers, node counts, histories, corpus payloads, or test payloads
  were accessed.
- The authoritative-profile arm changes both category selection and
  category-conditioned geometric projection, so it cannot separate those two
  contributions further.
- The result covers one repaired checkpoint, one validation split, and 68
  validation families; it does not establish multi-seed or held-out-test
  generalization.
- Frozen Gate C and Gate D remain failed. Diagnostic counterfactual validity
  cannot retroactively repair either gate.
- Gate E/OpenCascade remains unevaluated.
- This replay does not constitute checkpoint acceptance or final Phase B
  acceptance.

## Artifacts and integrity

All paths below are relative to the immutable categorical-isolation output
namespace.

| Artifact | SHA-256 | Availability |
|---|---|---|
| `aggregate_summary.json` | `2f8a0832405d00101771f3aed54b8fa0c6f98aa4019d64fb045118a207423a56` | Present, verified locally |
| `capsule_regressions.json` | `19dd91892e9af4cb24c17d7431c595ec816d9d07d6219560d432409b09453bb8` | Present, verified locally |
| `factorial_records.jsonl` | `f77286aeead80c29d59f3567391d3fd7b21eec12b207fa30ad381604683fe102` | Present, verified locally |
| `family_transitions.csv` | `e2f7958050bfc03a010820bff9d61c41643c1ea1e5fafc56abee4e48f8278cc7` | Present, verified locally |
| `projection_failures.csv` | `52a38f8edbed7d148ad0689bb21e5a2584bbf0ea408eb10e74c8db63e76456e9` | Present, verified locally |
| `run_metadata.json` | `8f06eb21a5c6f6648d688cae43a3f9fb9ec2efb86fed80186faa05d6efa817b4` | Present, verified locally |
| `sha256-manifest.txt` | `e9c353a98fd37f5a491c34970d1859c73d61ebb7341d1ac14ca020a6f51aea63` | Present, verified locally |

## Related records

- Frozen original result:
  [Repaired-checkpoint full validation](b0_phase_b_repaired_full_validation.md)
- Parent protocol:
  [Constraint-manifold replay preregistration](../specifications/flat_baseline_phase_b_constraint_manifold_replay.md)
- This diagnostic's protocol:
  [Categorical-isolation preregistration](../specifications/flat_baseline_phase_b_categorical_isolation_replay.md)
- Checkpoint source:
  [Full train-k-means retraining](b0_train_kmeans_full_retrain.md)
