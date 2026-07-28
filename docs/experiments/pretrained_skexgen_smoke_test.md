# Pretrained SkexGen Adroit Smoke Test

## Record status

| Item | Value |
|---|---|
| Experiment ID | `pretrained-skexgen-smoke-3319488` |
| Record status | `verified` |
| Scientific role | Inherited-pipeline environment smoke test |
| Branch | `baseline-reproduction` |
| Source commit | Not preserved |
| Working tree | No tracked source changes; temporary wrapper used |
| Slurm job | `3319488` |
| Execution date | July 22, 2026 |
| Evidence availability | `documented-external` |

## Question and decision rule

The smoke test asked whether the inherited pretrained SkexGen checkpoints
could load, sample candidate CAD histories, retain valid results, and export
OBJ files end to end on Princeton Adroit.

Success required the reduced sampler and OBJ-export path to complete without
modifying tracked SkexGen source. This was an operability gate, not an attempt
to reproduce the paper's 20,000-sample evaluation.

## Inputs and partitions

The run used the published pretrained topology, geometry, extrusion, and code
checkpoints from `proj_log/exp_sketch`, `proj_log/exp_extrude`, and
`proj_log/exp_code`. It generated new samples and did not train or evaluate
against the controlled project corpus. No current-project test partition was
used.

The recorded checkpoint layout was:

```text
proj_log/exp_sketch/cmdenc_epoch_300.pt
proj_log/exp_sketch/paramenc_epoch_300.pt
proj_log/exp_sketch/sketchdec_epoch_300.pt
proj_log/exp_extrude/extenc_epoch_200.pt
proj_log/exp_extrude/extdec_epoch_200.pt
proj_log/exp_code/code_epoch_800.pt
```

## Configuration and environment

| Item | Recorded value |
|---|---|
| Cluster | Princeton Adroit |
| Repository | `/scratch/network/km6349/SkexGen` |
| Container | `/scratch/network/km6349/skexgen.sif` |
| Python | 3.8.13 |
| PyTorch | 1.11.0 |
| PyTorch CUDA build | 11.3 |
| GPU | NVIDIA A100-PCIE-40GB MIG 3g.20gb |
| Batch size | 16 |
| Sample batches | 1 |
| Worker threads | 1 |

The temporary wrapper expressed the reduced sampler settings as:

```text
BS = 16
NUM_SAMPLE = 1
NUM_TRHEADS = 1
```

The temporary Slurm wrapper stubbed the PythonOCC GUI import to avoid a
GLIBC/libGLX conflict in the headless container. It did not replace the CAD
construction or OBJ-export path.

## Verified results

- Slurm job `3319488` completed in 14 seconds.
- The sampler produced 16 candidates.
- Fifteen CAD results were retained.
- OBJ export completed.
- Results were written to `proj_log/smoke_sample`.

These are the observations preserved in the contemporaneous project note.
The scheduler log and output directory were not recovered during the later
documentation audit.

## Decision and interpretation

The inherited pretrained sampling path passed the bounded environment smoke
gate. This established enough upstream operability to map the original
pipeline and proceed with the controlled research extension.

The result is not a paper reproduction or a statistically meaningful
generation benchmark.

## Limitations and unsupported claims

- The run used only one batch of 16 rather than 20,000 samples.
- The 15-of-16 retention observation is not a validity-rate estimate.
- Validity, uniqueness, novelty, diversity, and distributional metrics were
  not computed.
- Multi-seed and cross-environment reproducibility were not tested.
- Runtime and resource behavior at full scale remain unknown.
- The GUI-import workaround remains environment-specific.
- The exact source commit and primary run artifacts were not preserved.

## Artifacts and integrity

| Artifact | Location | SHA-256 | Availability |
|---|---|---|---|
| Scheduler stdout/stderr | Original Adroit job `3319488` | Not preserved | External/missing |
| Generated samples | `proj_log/smoke_sample` in the run checkout | Not preserved | External/missing |
| Pretrained checkpoints | Original `proj_log` paths | Not preserved | External/missing |

This record preserves the complete unique content of the former standalone
smoke-test note. No primary bundle was available for checksum verification.

## Reproduction and validation

The inherited commands and checkpoint layout are retained in the
[upstream README](../reference/upstream_skexgen_readme.md), while the inspected
pipeline is documented in the
[inherited repository map](../reference/inherited_skexgen_repository.md).

The smoke run was not reproduced during documentation consolidation.

## Related records

- Historical reference: [Inherited SkexGen repository map](../reference/inherited_skexgen_repository.md)
- Upstream instructions: [Original SkexGen README](../reference/upstream_skexgen_readme.md)
- Successor milestone: [Controlled CAD foundation](../milestones/controlled_cad_foundation.md)
