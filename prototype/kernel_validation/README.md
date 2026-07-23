# CAD-kernel execution validation

This package checks whether schema-valid samples from `prototype/controlled_data`
also execute as meaningful, single-body histories in OpenCascade. It never changes
the input corpus. The only output is a deterministic `execution_report.json`.

## Architecture

- `corpus.py` reads the controlled corpus and preserves per-sample loading failures.
- `executor.py` deserializes with the representation package, verifies identities
  with `controlled_data.identity`, resolves world geometry, and executes operations.
- `adapter.py` is the PythonOCC-free protocol used by orchestration and unit tests.
- `occ_adapter.py` is the only PythonOCC import boundary. It does not import GUI code.
- `reporting.py` performs deterministic aggregation, paired comparisons, and atomic
  report publication.
- `run.py` provides the CLI. `tests/` uses a fake adapter and requires no CAD kernel.

The plane normal is `x_axis cross y_axis`. Extrusion uses that normal, negated for
the negative direction. Revolve angles are converted from degrees to radians and
negated for the negative direction. Profiles with holes and nonlinear or
multi-body histories remain unsupported in version 1. No healing, fuzzy Boolean
tolerance, kernel timeout, solid export, or input filtering is performed.

## Execution contract

Each outer loop is converted to a wire of line, three-point arc, or circle edges.
The wire must be closed and valid before it becomes a planar face. An extrusion
prisms the face along the signed plane normal. A revolution rotates it around its
world-space sketch axis. The first operation establishes `NEW_BODY`; later `JOIN`
and `CUT` features are fused with or subtracted from the current body.

Every standalone feature and Boolean result must be non-null, kernel-valid, contain
exactly one solid, and have finite volume greater than `1e-9`. Disjoint joins are
therefore rejected. A JOIN must increase volume, and a CUT must decrease volume,
beyond `max(1e-9, 1e-9 * max(abs(before), abs(after)))`. A kernel-valid no-op is
reported as `boolean_no_effect` and is not a successful executable history.

Failure details are whitespace-normalized, path-redacted, and truncated to 240
characters. `failed_operation_index` is zero-based. Failure categories and stages
are stable strings defined in `model.py`.

## Deterministic report

The report contains sorted sample records, separate `pythonocc_version` and
`opencascade_version` fields, aggregate success rates, and operation progress
(`scheduled`, `reached`, `kernel_completed`, `semantically_effective`, `failed`)
by operation type and Boolean mode. It contains neither timestamps nor absolute
paths.

Successful continuous/quantized pairs are compared using final volume, axis-aligned
bounding box, and exact solid/face/edge/vertex counts. Volume uses absolute and
relative tolerances of `1e-9`; bounding-box coordinates use absolute `1e-8` and
relative `1e-9`. Status agreement and geometric agreement are reported separately.

## Local tests

```bash
python3 -m unittest discover -s prototype/kernel_validation/tests -v
```

These tests inject a fake adapter and do not establish real OpenCascade validity.

## Adroit integration

Prepare a controlled corpus containing 60 source families (120 variants), then run
the integration as a CPU Slurm job. The command inside the job is:

```bash
apptainer exec \
  --cleanenv \
  --bind /scratch/network/km6349:/scratch/network/km6349 \
  /scratch/network/km6349/skexgen.sif \
  /root/miniconda3/bin/python3.8 \
  -m prototype.kernel_validation.run \
  --corpus-dir <corpus> \
  --output-dir <report>
```

`adroit/kernel_validation_cpu.slurm` is a CPU-only template; replace its corpus and
report placeholders before submission. Generated reports and CAD solids must not
be committed.

The three prototype packages retain postponed modern annotation syntax but avoid
runtime type expressions and standard-library calls introduced after Python 3.8.
They are intended to run with the container's Python 3.8.13 interpreter. The local
fake-adapter tests and the container-side Python 3.8 test suite remain prerequisites
for the real integration run.
