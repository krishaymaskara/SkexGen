# B0 Phase A State and Checkpoint Contract Validation

## Record status

| Item | Value |
|---|---|
| Experiment ID | `b0-phase-a-contract-3323988` |
| Record status | `verified` |
| Scientific role | Decoder/API and backward-compatibility validation |
| Phase A commit | `08985a39dc6e7cdb5f2af96e5f24fd9139df0f10` |
| Comparison base | `ba611fbe628ac76a7a1449a30dd388c52b72ef11` |
| Slurm job | `3323988` |
| Execution date | July 26, 2026 |
| Scheduler result | `COMPLETED`, exit `0:0`, 0:12 on `adroit-h11n2` |

## Question and decision rule

Phase A added deterministic length-conditioned autoregressive decoding. The
validation asked whether that change preserved the existing model state
contract and could strictly load the original epoch-50 checkpoint. Acceptance
required identical ordered state entries and strict checkpoint loading.

## Verified results

| Check | Result |
|---|---|
| `prototype` import | Pass |
| Model import | Pass |
| Base state entries | 126 |
| Phase A state entries | 126 |
| Ordered state-dict contract | Pass |
| Original checkpoint epoch/global step | 50 / 850 |
| Strict checkpoint load | Pass |
| Base state-contract SHA-256 | `e3e70b1db4d0ab146a1b6473fc3296c5a6fe1e49535e4d3aeda18e718bc3b92a` |
| Phase A state-contract SHA-256 | `e3e70b1db4d0ab146a1b6473fc3296c5a6fe1e49535e4d3aeda18e718bc3b92a` |

The stderr file is empty. Stdout records Python 3.8.13 and PyTorch 1.11.0.
The archive contains both state-contract files and the exact dump helper.

## Decision and interpretation

Phase A passed its compatibility gate. This supports decoder mechanics,
state/API compatibility, strict loading of the original checkpoint, and
deterministic length-conditioned decoding.

It does not rehabilitate the checkpoint's scientific quality. The loaded
checkpoint is the one-code-collapsed model documented separately.

## Provenance limitations

The archive identifies the Phase A and base commits through the run namespace
and retained workflow output. It does not contain a separate source-tree
status or source-tree hash for this short continuation job, so this record
does not add an independently verified clean-tree claim.

No dataset partition was evaluated by this state-contract comparison.

## Artifacts and integrity

| Artifact | Availability |
|---|---|
| State dumps and helper | Present in frozen multi-run archive |
| Stdout/stderr | Present |
| Frozen archive SHA-256 | `9eaeeb91728d0e2c8e8a7de77a0ac837e92a9c3fe7c704c6011ee26fbbae0498` |

Archive:
`/Users/krishaymaskara/research/audited-runs/skexgen-flat-evidence-20260728.tar.gz`.
All internal archive checks passed.

## Related records

- Checkpoint source: [Original B0 collapse](b0_original_training_collapse.md)
- Downstream evaluation: [Phase B validation evaluation](b0_phase_b_validation_evaluation.md)

