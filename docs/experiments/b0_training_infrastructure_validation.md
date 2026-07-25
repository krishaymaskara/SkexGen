# B0-FLAT-MIXED-VQ Training-Infrastructure Validation

## Purpose

This record documents the completed CPU validation and CUDA engineering smoke
test for the teacher-forced `B0-FLAT-MIXED-VQ` training infrastructure.

| Item | Verified value |
|---|---|
| Branch | `flat-mixed-baseline` |
| Validation working tree | Dirty |
| Validation base commit | `73349b475069fd68acd7940f8fde90dadb37e7eb` |
| Subsequent canonical commit | `19a18591e1799e51447af4f31fbb0510761af444` |
| Canonical commit description | `19a1859 Add flat baseline training pipeline` |

The validation jobs ran from a dirty `flat-mixed-baseline` working tree based
on commit `73349b4`. The intended production-source changes validated by those
jobs were subsequently committed as `19a1859`. These records therefore
document dirty-tree validation preceding the canonical commit, rather than a
clean-checkout validation of `19a1859`.

## CPU validation

Slurm job `3322236` completed with exit code `0:0` in 27 seconds on a CPU
compute node.

| Environment item | Verified value |
|---|---|
| Python | 3.8.13 |
| PyTorch | 1.11.0 |
| PyTorch CUDA build | 11.3 |
| Compute device | CPU |

The run verified:

- all 63 flat-baseline tests passed, with zero skips;
- all 39 model-data tests passed;
- `compileall` completed successfully.

The final job marker was:

```text
ALL FINAL B0 CPU VALIDATION CHECKS PASSED
```

Raw scheduler output remains outside the repository:

```text
/scratch/network/km6349/flat_baseline_runs/b0-final-3322236.out
/scratch/network/km6349/flat_baseline_runs/b0-final-3322236.err
```

## CUDA smoke validation

Slurm job `3322251` completed with exit code `0:0` in 36 seconds.

| Environment item | Verified value |
|---|---|
| GPU | NVIDIA A100 80 GB PCIe |
| Python | 3.8.13 |
| PyTorch | 1.11.0 |
| PyTorch CUDA build | 11.3 |
| CUDA available | `true` |

The engineering-smoke configuration used:

- one selected physical family for training;
- one selected physical family for validation;
- a tiny smoke-test model;
- batch size 1;
- learning rate 0.001;
- an initial run covering epochs 1–2;
- a relocated resume through epoch 3.

The run verified actual CUDA forward and backward optimization, EMA-codebook
operation, creation of `metrics.jsonl`, `last.pt`, and `best.pt`, checkpoint
relocation, restoration of optimizer/model/RNG state, resumed CUDA training,
and checkpoint loading with `map_location=cpu`.

The observed validation total losses were:

| Epoch | Validation total loss |
|---:|---:|
| 1 | 9.053128242492676 |
| 2 | 8.846267700195312 |
| 3 | 8.638562202453613 |

The final job marker was:

```text
ALL FINAL B0 CUDA SMOKE CHECKS PASSED
```

Raw scheduler output and run directories remain outside the repository:

```text
/scratch/network/km6349/flat_baseline_runs/b0-cuda-3322251.out
/scratch/network/km6349/flat_baseline_runs/b0-cuda-3322251.err
/scratch/network/km6349/flat_baseline_runs/b0-cuda-3322251-initial
/scratch/network/km6349/flat_baseline_runs/b0-cuda-3322251-resume
```

## Interpretation and limitations

The CUDA run was an engineering smoke test, not a scientific result. Its one
training family and one validation family are sufficient to exercise the
training and resume infrastructure, but they do not establish:

- generalization to unseen physical histories;
- autonomous CAD reconstruction;
- executable CAD validity;
- comparative performance against typed-graph representations.

The decreasing losses above are observations from this bounded smoke run, not
evidence for a research claim. The complete Slurm wrapper commands and exact
tiny-model dimensions are not recorded here because they cannot be
reconstructed accurately from the repository documentation alone.

Raw logs, checkpoints, metrics files, container files, `__pycache__`
directories, and `.pyc` files are intentionally not copied into Git.
