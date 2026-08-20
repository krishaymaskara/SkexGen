# GE1 Memory-to-Count Prerequisite Probe

## Record status

| Item | Value |
|---|---|
| Experiment ID | `ge1-memory-count-probe-3355776` |
| Record status | `verified` |
| Scientific role | Read-only train-only prerequisite diagnostic |
| Source commit | `25d9510d2b7b502b69b7387f1f0937509d3ef9ba` |
| Governing decision | [ADR-0016](../decisions/ADR-0016-ge1-autonomous-stop-and-unconstrained-node-decoding.md) |
| Contract | [Autonomous stop node-generation contract](../specifications/ge1_autonomous_stop_node_generation.md) |
| Slurm job | `3355776` |
| Compute host | `adroit-h11n2` |
| Runtime | Python 3.8.13, PyTorch 1.11.0, CPU, one thread |
| Runner | `prototype/graph_encoder/adroit/ge1_memory_count_probe_cpu.slurm` |
| Outcome | Valid artifact; prerequisite rule did not pass |

## Question and predetermined gate

Before implementing a retrain that must learn when to emit `<pad>`, the probe
asked whether the four train node-count classes `(4, 5, 7, 8)` were linearly
recoverable from `encoded.memory [B,2,32]` and
`encoded.prequant [B,2,16]` in all six retained epoch-200 checkpoints.

The exact-commit implementation froze a SHA-256-ordered, class-stratified
within-train split; float64 multinomial logistic regression with L2 `1e-3`;
majority and shuffled-training-label controls; and an all-checkpoint rule. A
checkpoint/feature pair passed only at balanced accuracy at least `0.90` and a
margin of at least `0.10` over each control. The prerequisite passed only if
all twelve pairs passed. The diagnostic outcome did not control artifact
validity.

## Inputs, access, and integrity

The runner exposed read-only bindings for the authorized narrow train package,
the six exact job-`3354961` checkpoint wrappers, the authenticated job-`3355134`
postmortem artifact, and the exact detached repository. It exposed no
development, RR, ER, IID, history-depth, geometry-extrapolation, other corpus,
or CAD-kernel resource. GE1 parameters had gradients disabled before feature
extraction. Only disposable linear-probe parameters were optimized.

The artifact contains 2,442 feature rows: 407 train families × two arms ×
three seeds. Its internal manifest and `SHA256SUMS` verified successfully.

| Artifact | SHA-256 |
|---|---|
| `artifact_manifest.json` | `1379253060de52e512aadd9c1afb8a63f97104a89915250f6a156921bb28e763` |
| `features.jsonl` | `d67845cab87fc15b6b1958ad418bea01397ca6a49bb462af450fa3951e77222f` |
| `probe_results.json` | `84b84775669a458eab8a0ee05f6764e9a899da3176aebc02ab15e24c8ecbc00e` |
| `resolved_config.json` | `1c512cf84fb51d418ca8b7b4fbeadc32fd9146a6102f74c13e91c7bdf18581ac` |
| `SHA256SUMS` | `424d74bb3f7c5ab23b913fcc420c3eb58b3fa139a14936d05ae8a508e03b336b` |
| stdout | `e0b17cee40261fb2fe7a08106b97ac6de46aa4eb27cea93425f32b8d1a90ab33` |
| stderr | `0dfbec1b7b6afa7b4c2e68c40c590d0f6ba561679cca0d88df2a334ee7323810` |

Scheduler evidence was `COMPLETED / 0:0`, elapsed `00:01:31`, one allocated
CPU, and batch MaxRSS `691152K`. All six focused tests passed with zero skips;
documentation, compileall, Python 3.8 AST, source/access audit, Bash syntax,
whitespace, exact-commit, and clean-tree checks passed.

## Results and findings

Balanced accuracy by checkpoint:

| Arm | Seed | Memory | Prequant | Both pass? |
|---|---:|---:|---:|---|
| flat | 2026 | 1.000 | 1.000 | yes |
| flat | 2027 | 1.000 | 1.000 | yes |
| flat | 2028 | 0.926 | 0.951 | yes |
| typed graph | 2026 | 0.975 | 0.988 | yes |
| typed graph | 2027 | 0.988 | 0.988 | yes |
| typed graph | 2028 | 0.751 | 0.726 | **no** |

The majority baseline was `0.25` everywhere. Shuffled-label controls ranged
from `0.194` to `0.318`. Mean balanced accuracy was:

| Arm | Memory | Prequant | All seeds recoverable? |
|---|---:|---:|---|
| flat | 0.975 | 0.984 | yes / yes |
| typed graph | 0.905 | 0.900 | no / no |

The artifact is valid, but the all-checkpoint prerequisite is **false** because
typed-graph seed 2028 missed the frozen `0.90` accuracy floor in both feature
spaces. The result is not a near-pass or an authorization to weaken the rule.

## Decision and interpretation

Do not authorize the stop-symbol retrain. Count information is strong and far
above both controls in every checkpoint, but it is not uniformly linearly
recoverable under the prospective rule. In particular, the typed-graph
seed-2028 bottleneck is the failure the prerequisite was designed to expose.

ADR-0016 separately authorizes source implementation and bounded engineering
validation, so those may proceed. They do not create retrain or Stage 6
authority.

## Limitations

- This is linear accessibility on one deterministic within-train split, not a
  proof that a nonlinear decoder can or cannot learn stopping.
- Node count is correlated with template in the current train partition; the
  diagnostic measures recoverability, not a causal representation.
- The six checkpoints predate the autonomous-stop objective and were not
  trained to encode count explicitly.
- No development or protected example was evaluated.

## Artifacts and integrity

The finalized artifact is retained on Adroit at
`/scratch/network/km6349/ge1_memory_count_probe_runs/ge1-memory-count-probe-25d9510d2b7b502b69b7387f1f0937509d3ef9ba-3355776`.
Artifact verification reported `verification_status=pass` and
`outcome_controls_artifact_validity=false`.

## Related records

- [ADR-0016](../decisions/ADR-0016-ge1-autonomous-stop-and-unconstrained-node-decoding.md)
- [GE1 autonomous stop node-generation contract](../specifications/ge1_autonomous_stop_node_generation.md)
- [Zero-memory job 3355342](ge1_stage6_zero_memory_3355342.md)
- [Train-gate postmortem job 3355134](ge1_stage6_train_gate_postmortem_3355134.md)
