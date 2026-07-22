# Adroit Pretrained SkexGen Smoke Test

## 1. Purpose

This note records a successful pretrained SkexGen inference smoke test run on Princeton Adroit on July 22, 2026. The goal was to verify that the original pretrained model could load its published checkpoints, sample candidate CAD histories, retain valid results, and export OBJ files in the Adroit environment.

This was a reduced smoke test, not a full reproduction of the paper's 20,000-sample evaluation. It verifies basic end-to-end operability of the pretrained sampling path under the settings below; it does not establish paper-level generation metrics, validity rates, diversity, or reproducibility at the original evaluation scale.

## 2. Environment

| Item | Verified value |
|---|---|
| Date | July 22, 2026 |
| Cluster | Princeton Adroit |
| Repository branch | `baseline-reproduction` |
| Repository path | `/scratch/network/km6349/SkexGen` |
| Container | `/scratch/network/km6349/skexgen.sif` |
| Python | 3.8.13 |
| PyTorch | 1.11.0 |
| PyTorch CUDA build | 11.3 |
| GPU | NVIDIA A100-PCIE-40GB MIG 3g.20gb |

No tracked SkexGen source files were modified for this test.

## 3. Checkpoint layout

The pretrained checkpoints were available relative to the repository root at:

```text
proj_log/exp_sketch/cmdenc_epoch_300.pt
proj_log/exp_sketch/paramenc_epoch_300.pt
proj_log/exp_sketch/sketchdec_epoch_300.pt
proj_log/exp_extrude/extenc_epoch_200.pt
proj_log/exp_extrude/extdec_epoch_200.pt
proj_log/exp_code/code_epoch_800.pt
```

## 4. Slurm configuration

The successful Slurm job was job `3319488`. It ran on an NVIDIA A100-PCIE-40GB MIG 3g.20gb allocation and completed in 14 seconds.

The sampler was reduced to the following smoke-test settings in the temporary job wrapper:

```text
BS = 16
NUM_SAMPLE = 1
NUM_TRHEADS = 1
```

These values intentionally exercise the inference and export path at small scale. They are not the repository's full evaluation settings.

## 5. Compatibility workaround

The PythonOCC GUI import was stubbed only in the temporary Slurm wrapper to avoid a GLIBC/libGLX compatibility conflict in the containerized, headless cluster environment. This workaround affected GUI import setup only; it did not modify tracked SkexGen source files or replace the CAD construction and OBJ export path being tested.

## 6. Verified result

Job `3319488` completed successfully with the following observed result:

- 16 candidates were sampled.
- 15 CAD results were retained.
- OBJ export completed successfully.
- Results were written to `proj_log/smoke_sample`.
- Total runtime was 14 seconds.

This verifies that, with the listed container, checkpoints, software versions, GPU allocation, and temporary GUI-import workaround, the pretrained SkexGen sampling pipeline can execute end to end on Adroit.

## 7. Remaining limitations

- The run used `NUM_SAMPLE = 1` with a batch size of 16; it did not run the original 20,000-sample evaluation.
- The observed 15-of-16 retention result is a smoke-test observation, not a statistically meaningful validity estimate.
- Paper-reproduction metrics, including large-sample validity, uniqueness, novelty, diversity, and distributional comparisons, were not computed.
- Runtime and resource behavior at the original batch and sample counts remain unverified.
- Reproducibility across random seeds, repeated jobs, other Adroit GPU types, or a non-container environment remains unverified.
- The GLIBC/libGLX conflict still requires the temporary PythonOCC GUI-import stub in this container environment unless the underlying library compatibility is resolved.
