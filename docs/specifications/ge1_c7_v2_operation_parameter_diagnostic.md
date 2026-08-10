# GE1 C7-v2 Operation-Parameter Diagnostic Contract

## Status and authority

| Field | Frozen value |
|---|---|
| Contract | `GE1-C7-V2-OPERATION-PARAMETER-DIAGNOSTIC-v1` |
| Status | Completed read-only on Adroit as job `3345013`; source C7-v2 result unchanged |
| Reviewer authorization | Krishay Maskara, August 10, 2026 |
| Source result | C7-v2 commit `e325d5ad97957c08da4a19b4261560e8a4a472a4`, job `3344981` |
| Source artifact | `/scratch/network/km6349/ge1_c7_v2_runs/ge1-c7-v2-e325d5ad97957c08da4a19b4261560e8a4a472a4-3344981` |
| Execution environment | Adroit, Python 3.8.13, PyTorch 1.11.0, CPU only |

This is a separately versioned, read-only diagnostic of the finalized C7-v2
scientific failure. It does not alter, relabel, rerun, or replace that result.
It does not authorize Stage 6, C8, decoder repair, geometry repair, or access
to another corpus partition.

## Question

C7-v2 produced exact node sequences and typed graphs, exact applicable
`depends_on` relationships, strict conversion, and passing memory gates for
both scaled arms. Five physical families nevertheless failed complete
executable validity with `invalid_operation_parameter`. The diagnostic asks
which observable scalar, normalization, mask, category, or conversion
mechanism produced each failure, while keeping encoder-memory versus shared
geometry-head causation unresolved unless the recorded evidence distinguishes
them.

The frozen failures are:

| Failing arm | Family | Template |
|---|---|---|
| Flat | `sf_140d5984ae7b954a9b0232bc3f1d1e0bf24b73971e54ccce6dddfd253ff522e6` | RE |
| Flat | `sf_93d4027a05325354f073f6ebe3375e1d2fe52a1fe1fc6c914fdcaef98e7d169c` | RE |
| Typed graph | `sf_69cc286ec6c7d7993635edb269176860e99dbaf9877d467a6e0aa31afdd44378` | EE |
| Typed graph | `sf_caab9609073e3323d6d748a4d4a05174bd7978bb938f86e69d28441dcc6dc24c` | E |
| Typed graph | `sf_fe25554783fbed5e372dbd12b4db1ee6a590f9eb90ef5f674314b161ad4303c3` | EE |

Each family is evaluated under both arms; the arm that passed in C7-v2 is a
within-family control.

## Corrected validity semantics

The reviewer-authorized correction separates four questions that the
repository can and cannot answer:

1. **Strict conversion** is the existing strict converter's
   `reconstruction_target.valid` result. It records whether the predicted
   graph and parameter structure can be reconstructed under that contract.
2. **Analytic controlled-domain validity** is the existing analytic
   validator's `controlled_domain.valid` result. For the active operation
   parameter, `invalid_operation_parameter` means the denormalized
   `extrude_distance` or `revolve_angle` is less than or equal to zero.
3. **Executor validity is structurally unavailable.** The controlled GE1
   evaluation path uses no CAD kernel, so every executor field must be
   `{available: false, result: null,
   reason: no_cad_kernel_in_evaluation_contract}`. No analytic result may be
   relabeled as a kernel execution.
4. **Legal-but-geometrically-incompatible parameters are unassessable.** A
   positive analytic parameter is not evidence that a CAD kernel would create
   a valid single solid. Without such a kernel, this classification remains
   null with the same explicit no-kernel reason.

This correction changes only the diagnostic's reporting contract. It does
not change C7-v2 metrics, gates, thresholds, models, decoder, conversion
logic, or the meaning of the finalized job `3344981` artifact.

## Frozen source identities

Before inference, the diagnostic verifies the complete source artifact, then
requires these direct identities:

| Source file | SHA-256 |
|---|---|
| `metrics.jsonl` | `fe151f45df045dd18182053a42e304600b7c272f9d072339707fdc00dc24e2fd` |
| `artifact_manifest.json` | `9c526c89dda8fb72bfafc0c922c73fae6461b0f5728eaaeaeea31d0335994d37` |
| Flat scaled epoch-200 checkpoint | `ff628c7e1c5391869237658a00941efbe2481cb6c272d7c9e56affad2a99e992` |
| Typed-graph scaled epoch-200 checkpoint | `7535c8bd4e01ac586beff431a228bc2c244b7ff5d97faf745b5d2125d35e0cd2` |

Each checkpoint must also identify source commit `e325d5ad...`, Slurm job
`3344981`, the correct arm, seed 2026, CPU runtime, fixed epoch 200, 800
optimizer steps, 6,400 family presentations, and the exact scaled-partition
identity. State is loaded strictly into a fresh inference model. Stored
optimizer evidence is validated only as an immutable checkpoint field; no
optimizer is constructed or restored.

## Execution and reproduction order

The only permitted execution order is:

1. require the exact clean diagnostic commit and the authoritative CPU
   runtime;
2. verify the immutable C7-v2 artifact and source checkpoint hashes;
3. verify the authoritative `operation_template` manifest using metadata;
4. reconstruct the exact frozen 32-family scaled train cohort;
5. load only those 32 train payloads, preserving C7-v2 family order and
   batch size eight;
6. load each source checkpoint into a fresh model and run deterministic
   autonomous `P_true` inference;
7. reproduce the original stable per-family validity, prefix, graph,
   dependency, geometry-error, and failure results exactly, excluding only
   wall-clock measurement fields;
8. reject the diagnosis unless both complete reproductions pass; and
9. extract minimal traces for both arms on the five frozen families.

There is no optimizer, backward pass, training call, checkpoint write,
teacher-forced decision path, model-parameter update, adaptive extension, or
repair call. Model-state hashes before and after inference must match, and
the four immutable source-file hashes must match before and after the job.

## Minimal diagnostic record

For every operation in each of the five families and for both arms, the
diagnosis records only:

- family, template, arm, operation index and type;
- the active geometry channel and mask evidence;
- accessible pre-`tanh` head output and post-`tanh` normalized prediction;
- the exact scale-only denormalization and physical prediction;
- normalized and physical target scalars and signed/absolute error;
- analytic domain, boundary margin, and scalar classification;
- predicted and target direction and Boolean mode;
- the absence of a separate extent category in this schema;
- the associated profile's physical extent as limited scale context;
- strict-conversion and analytic controlled-domain validity;
- primary and secondary analytic failures, executable prefix, failed
  operation, and first failure stage/code;
- the other arm's aligned scalar control; and
- the structurally unavailable executor and geometric-compatibility fields.

Complete CAD histories, reconstructed history JSON, graph node/edge payloads,
sample payloads, model state, optimizer state, and checkpoints are forbidden
from the diagnostic artifact. The artifact contains exactly
`diagnosis.json`, `resolved_config.json`, `metrics.jsonl`,
`artifact_manifest.json`, and `SHA256SUMS`.

## Access and authorization boundary

The implementation has one payload entry point: `load_train` with the exact
32-family scaled cohort. It may access the authoritative
`operation_template` manifest and those train payloads only. Development, RR,
ER, IID, history-depth, geometry-extrapolation, and every other corpus remain
unreachable and must be declared unopened.

The diagnostic is evidence collection, not a repair experiment. A completed
artifact must record that the C7-v2 result is unchanged, Stage 6 remains
unauthorized, C8 has not begun, and neither decoder nor geometry repair was
implemented or invoked. After the authoritative result is returned, work
stops for scientific review.

## Validation and delivery state

The focused pure suite freezes checkpoint identities, analytic boundaries,
no-kernel unavailability, both-arm coverage, exact metric reproduction, raw
payload rejection, protected-access reachability, runner scope, and Python
3.8 grammar. The prepared Slurm runner reruns the complete graph-encoder and
repository regression suites before any manifest or payload access, verifies
the source artifact read-only before and after inference, and emits distinct
preflight, incomplete-diagnostic, postflight, and terminal-success markers.

Adroit job `3345013` completed at exact diagnostic commit
`802ae1d1e9deb3c7a6c428d320e4276a8b5e7e57` under Python 3.8.13, PyTorch
1.11.0, and CPU execution. The artifact and semantic verification passed,
both arms were traced for all five families, and the immutable C7-v2 evidence
was reproduced before diagnosis. Every failed arm had a negative active
extrusion-distance or revolve-angle magnitude; the other arm's same-family
control was positive and analytically legal. Masks, channels, scale-only
normalization and denormalization, units, direction and Boolean categories,
targets, and alignment were consistent.

The completed diagnostic performed no training, backward pass, optimizer
step, checkpoint write, decoder repair, Stage 6, or C8 work. Only the
authorized operation-template train/scaled cohort was accessed; all protected
partitions and other corpora remained unopened. Executor results remain
structurally unavailable and geometric compatibility remains unassessable.
The result supports the separately accepted prospective decision in
[ADR-0009](../decisions/ADR-0009-ge1-positive-operation-magnitude-repair.md);
it does not itself authorize a repaired scientific run.

## Related records

- [C7-v2 execution contract](ge1_c7_v2_execution_contract.md)
- [ADR-0008](../decisions/ADR-0008-ge1-c7-v2-200-epoch-protocol.md)
- [GE1 implementation plan](graph_encoder_implementation_plan.md)
- [Current project status](../status.md)
