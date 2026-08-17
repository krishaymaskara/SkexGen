# GE1 Grid-Magnitude Engineering Validation, Job 3351837

## Review disposition

Designated GE1 reviewer Krishay Maskara reviewed this execution on August 17,
2026. The job is accepted as technically valid engineering evidence, with a
recorded procedural deviation: it was submitted before the then-proposed
ADR-0013 explicitly authorized any Slurm submission. It is not a scientific
result and is not retroactively relabelled as prospectively authorized.

| Item | Verified value |
|---|---|
| Job | `3351837` |
| Exact commit | `bdb7148dc4ebe751bda1a66cd167fcf035495762` |
| Protocol | `GE1-C7-GRID-MAGNITUDE-SUFFICIENCY-v1` |
| Parameterization | `GE1-OPERATION-MAGNITUDE-GRID-ORDINAL-v1` |
| Checkpoint schema | `GE1-CHECKPOINT-v2` |
| Host | `adroit-h11n2` |
| State / exit | `COMPLETED` / `0:0` |
| Elapsed / batch MaxRSS | `00:06:19` / `615440K` |
| Runtime | CPython `3.8.13`, PyTorch `1.11.0`, CPU only, one thread |

The transcript and terminal record identify a clean, standalone, detached
checkout at the exact commit. The repository was the sole read-only container
bind.

## Validation evidence

| Suite or check | Result |
|---|---:|
| Pure grid contract | 28 run, 0 skipped |
| Real-PyTorch grid runtime | 14 run, 0 skipped |
| Synthetic grid integration | 18 run, 0 skipped |
| Complete graph encoder | 460 run, 0 skipped |
| Model-data regression | 86 run, 0 skipped |
| Flat-baseline regression | 549 run, 4 frozen permitted skips |
| Graph-baseline regression | 17 run, 0 skipped |
| Representation regression | 40 run, 0 skipped |
| Controlled-data regression | 51 run, 0 skipped |

Documentation validation, externalized bytecode compilation, Python 3.8 AST
parsing over 60 files, Bash syntax, the shared structural source audit, frozen
identity checks, source-preservation checks, and committed-whitespace checks
all passed.

The supplied checksum manifest was independently matched to all three files:

| Evidence file | SHA-256 |
|---|---|
| `grid-validation-3351837.out` | `5bb2be5462879ab79f6514e35dd3d7e3a6f10b534666320d5ac20d3f58d3fb70` |
| `grid-validation-3351837.err` | `497c366ed5441be1306364fb4ab12f5fbe9f751c954a81c18716cac7ce077971` |
| `grid-validation-3351837.sacct.txt` | `306a0ba578995ca8451399dbaa378eaf7bf113eb9380b6a2dc8de39043ca0807` |

## Boundary

The job used procedural fixtures, generated tensors, an engineering-only
optimizer step, and a temporary synthetic checkpoint round-trip. It did not
access a corpus, manifest, preserved payload, external or scientific
checkpoint, model or repaired artifact, or protected partition. It performed
no scientific training, scientific inference, CAD-kernel work, Stage 6, or C8.

The strongest licensed conclusion is that the corpus-free ADR-0013
implementation at the recorded commit passed its frozen engineering contract.
It does not establish decoder-state accessibility, operation fidelity, repair
success, or 48/48 scientific performance.
