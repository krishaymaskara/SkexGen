# Historical OpenCascade Kernel Validation: 60 Physical Families

## Purpose and status

This record documents the historical real-OpenCascade integration run that
tested whether schema-valid controlled CAD histories also executed as valid,
semantically effective single solids. It preserves evidence from the earlier
60-family generator configuration; it is not the current corpus-generation
procedure.

The preserved artifact bundle identifies the run as Slurm job `3319654` and
contains its canonical `corpus_manifest.json` and
`execution_report.json`. The source commit, calendar date, original cluster
paths, exact submission command, corpus-tree hash, report-file hash, and an
archived source patch were not preserved in Git and cannot be stated from the
available evidence.

## Historical corpus

The preserved corpus manifest records:

| Item | Verified value |
|---|---|
| Physical source families | 60 |
| Continuous/quantized representation variants | 120 |
| Generation seed | 2026 |
| Generator version | `1.0` |
| Representation schema version | 1 |
| Canonicalization version | `cad-history-json-v1` |
| Configuration SHA-256 | `7f4aca5e5d7dc35c2fc3dc9782d36cfbb22cca2b6c652c8a1a61f68e562b2dfc` |
| Historical raw candidate count | 180,900 |

Each physical family contributed exactly two representation variants: one
continuous and one quantized. A representation variant was one kernel
execution attempt; family-level results pair the two attempts that encode the
same decoded physical history.

## Verified execution result

The execution report records PythonOCC version `7.5.1`; the underlying
OpenCascade version was unavailable in the report.

| Outcome | Physical families | Representation variants |
|---|---:|---:|
| Successful | 47 | 94 |
| `boolean_no_effect` | 10 | 20 |
| Zero-solid `invalid_final_shape` | 3 | 6 |
| Total | 60 | 120 |

Continuous and quantized variants had perfect status agreement: all 60
physical-family pairs agreed on success or failure. All 47 successful pairs
also passed the report's paired geometric-agreement comparison.

The report recorded no schema or parse failures. Schema validity and kernel
validity are nevertheless different contracts: 13 schema-valid physical
families failed the stronger OpenCascade execution contract. Ten contained a
Boolean operation that did not change volume meaningfully, while three
produced a zero-solid final result.

## Consequence for generator development

The historical failures exposed analytically infeasible or ineffective
Boolean factor combinations in the earlier generator space. They motivated
the later pure-Python Boolean-feasibility classifier and coverage work in
`prototype.controlled_data`. OpenCascade results are not used as an
environment-dependent corpus-selection filter.

The current full-coverage generator procedure requires 68 physical source
families and produces 136 continuous/quantized variants. That current
procedure is documented in the
[kernel-validation package README](../../prototype/kernel_validation/README.md);
the historical 60-family result remains evidence about the earlier generator
behavior, not a recommendation to request 60 families now.

Raw corpora, execution reports, kernel solids, scheduler logs, caches, and
container files remain outside Git.
