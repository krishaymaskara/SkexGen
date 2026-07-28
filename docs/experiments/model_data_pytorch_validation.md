# Real-PyTorch Model-Data Validation

## Record status

| Item | Value |
|---|---|
| Experiment ID | `model-data-pytorch-3322011` |
| Record status | `verified` |
| Scientific role | Infrastructure and compatibility validation |
| Branch | `model-baselines` |
| Validated source | `91d733d56764beac8be06aa542466fe1b7643876` |
| Validation scripts | Later committed as `038e4977be4eb3f11d4d3528da82af29f10e4116` |
| Working tree | Dirty: two validation files were staged |
| Slurm job | `3322011`, `COMPLETED 0:0` |
| Execution date | July 23, 2026 |

## Question and predetermined decision rule

The job asked whether the shared model-data interface worked with real
PyTorch tensors in the authoritative Python 3.8 environment, rather than only
with the tensor-compatible adapter used by local tests.

The gate required the discovered repository tests, source compilation,
individual and batched tensor conversion, dtype/shape/contiguity checks,
offset checks, empty-edge handling, and reconstruction-target checks to pass.
The validation script ended unsuccessfully on any failed check.

## Inputs and partition authority

| Item | Value |
|---|---|
| Dataset/corpus | Deterministic bounded test fixtures |
| Train / validation / test families used | None |
| Learned model trained or evaluated | No |
| Test partition evaluated | `false` |

This was contract validation over fixtures, not a corpus-scale training or
scientific evaluation run. Counterfactual loading checks retained
edit-family partition authority, but no benchmark partition metrics were
computed.

## Configuration and environment

| Item | Verified value |
|---|---|
| Host | `adroit-h11n2` |
| Python | 3.8.13 |
| PyTorch | 1.11.0 |
| PyTorch CUDA build | 11.3 |
| cuDNN | 8200 |
| Execution device | CPU only |
| CPU threads | One OMP and one MKL thread |
| Container | `/scratch/network/km6349/skexgen.sif`; content hash not preserved |

The Slurm script requested one node, one task, one CPU, 8 GiB of memory, and
30 minutes. It cleared `CUDA_VISIBLE_DEVICES` and invoked Python 3.8 through
Apptainer with a clean environment.

## Execution

The preserved runner executed:

```text
/root/miniconda3/bin/python3.8 \
  prototype/model_data/adroit/validate_pytorch.py
```

The job ran from `/scratch/network/km6349/SkexGen`, beginning at
`2026-07-23T22:27:25-04:00` and finishing at
`2026-07-23T22:39:18-04:00`.

## Verified results

- 193 tests were discovered and all 193 passed in 704.924 seconds.
- The representation, controlled-data, kernel-validation,
  counterfactual-edits, and model-data packages compiled.
- Real PyTorch individual and batched conversion passed.
- Tensor dtypes, shapes, contiguity, graph offsets, empty-edge behavior, and
  reconstruction targets passed.
- The validation script printed `ALL ADROIT MODEL-DATA CHECKS PASSED`.
- The scheduler result was `COMPLETED 0:0`.

## Decision and interpretation

The real-PyTorch compatibility gate passed. The result supports the claim
that the shared model-data contract constructs the intended PyTorch 1.11
tensors under Python 3.8 on Adroit.

This is infrastructure evidence only. It does not establish model
convergence, reconstruction quality, capacity matching, or systematic
generalization.

## Limitations and unsupported claims

- Job `3322011` did not run from a clean checkout of commit `038e497`.
  Instead, it ran at `91d733d` with the two archived validation files staged;
  those exact files were committed afterward as `038e497`.
- The bundle does not include a complete working-tree status or staged-diff
  hash beyond the archived validation files and its provenance note.
- Execution was CPU-only, so it does not validate CUDA model-data behavior.
- The container content hash and complete scheduler accounting are not
  preserved.
- Fixtures do not substitute for corpus-scale model training or evaluation.

## Artifacts and integrity

| Artifact | Location | SHA-256 | Availability |
|---|---|---|---|
| Frozen validation archive | `/Users/krishaymaskara/research/audited-runs/model-data-pytorch-038e497-job3322011.tar.gz` | `43e5ca38eaf1f687fd4ab6b2da8d08dc872119e7243712dc68418aa15f8c8012` | Present locally |
| Slurm runner | Archive member `model_data_pytorch_cpu.slurm` | Not recorded separately | Present in archive |
| Python validator | Archive member `validate_pytorch.py` | Not recorded separately | Present in archive |
| Stdout / stderr | Archive members for job `3322011` | Not recorded separately | Present in archive |

The 8,931-byte gzip/tar archive extracted successfully. Its validation
summary, runner, validator, stdout, and stderr were inspected during the July
28 documentation audit.

## Reproduction and validation

The committed validation infrastructure at `038e497` contains the archived
runner and validator. Faithful reproduction requires the source and
validation-script provenance described above and a compatible Adroit
container.

The archive was inspected and the output claims were cross-checked against
the preserved scripts and stderr test summary. The job itself was not rerun
during documentation backfill.

## Related records

- Package contract: [Model-ready data interface](../../prototype/model_data/README.md)
- Milestone: [Controlled CAD foundation](../milestones/controlled_cad_foundation.md)
- Downstream validation: [B0 training infrastructure](b0_training_infrastructure_validation.md)

