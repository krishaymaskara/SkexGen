# Counterfactual OpenCascade Audit

## Record status

| Item | Value |
|---|---|
| Experiment ID | `counterfactual-opencascade-audit-3321612` |
| Record status | `verified` |
| Scientific role | Evaluation-only kernel and edit-pair validation |
| Branch | `counterfactual-edit-pairs` |
| Source commit | `3ae4f6cf96b22c6ef33f592dc4e6e47020e059c6` |
| Working tree | Not preserved in the bundle |
| Primary Slurm job | `3321612` |
| Independent replay | `3321613` |
| Execution date | July 23, 2026 |

## Question and predetermined decision rule

The audit asked whether every selected counterfactual endpoint could be
executed as a valid single solid in OpenCascade, whether every edit pair had
two successful physical endpoints, and whether continuous and quantized
encodings agreed at the physical-endpoint level.

The run passed only if all encoded endpoints executed successfully, every
edit sample succeeded, and the two encodings agreed for every physical
endpoint. The replay was additionally expected to reproduce the primary
corpus, endpoint report, and pair audit byte for byte.

## Inputs and partition authority

| Item | Verified value |
|---|---|
| Configuration SHA-256 | `cf9a16d315e84a074dd4c671198620853bab61f1733b35dba8fd097a015e2f79` |
| Generation seed | 17 |
| Edit families / samples | 68 / 136 |
| Physical / encoded endpoints | 136 / 272 |
| Assignment unit | `edit_family_id` |
| Learned model evaluated | No |
| Test partition evaluated | Not applicable |

The counterfactual manifest identifies schema version 1, generator version
1.0, `counterfactual-json-v1` edit canonicalization, and
`cad-history-json-v1` representation canonicalization. Both endpoints of an
edit family inherit the same partition, and the endpoint-exclusion manifest
keeps evaluation endpoints available for exclusion from future training
corpora.

## Configuration and environment

The package used analytical Boolean-feasibility filtering before pair
selection; OpenCascade results were not used to choose examples. The selected
families span the fixed version-1 edit and coverage contract described in the
[counterfactual package README](../../prototype/counterfactual_edits/README.md).

The execution report identifies PythonOCC 7.5.1. The underlying OpenCascade
version, Python version, container hash, hardware, scheduler allocation, and
clean/dirty working-tree status were not preserved in the archive.

## Execution

Job `3321612` generated the corpus, validated all 272 encoded endpoint
histories, and joined the endpoint report into a pair-execution audit. Job
`3321613` repeated the same workflow. Both stdout logs identify the same full
source commit, and both stderr files are empty.

The primary output namespace was:

```text
/scratch/network/km6349/counterfactual_runs/job-3321612/
  corpus/
  endpoint-report/
  pair-audit/
```

## Verified results

| Measure | Result |
|---|---:|
| Encoded endpoint execution success | 272 / 272 |
| Successful edit samples | 136 / 136 |
| Continuous/quantized physical-endpoint agreement | 136 / 136 |
| Pair-audit success rate | 1.0 |
| Endpoint final shape | Valid single solid for every result |

The archive directly contains all 272 endpoint results, the pair-execution
audit, the corpus and split manifests, the primary logs, and the replay logs.
The separate Python 3.8 validation job is reported as 154/154 passing in the
bundled validation note.

## Decision and interpretation

The audit passed its kernel-execution and encoding-agreement gate. It supports
using this fixed corpus as an evaluation-only counterfactual benchmark and
shows that the selected endpoints are executable under the preserved
PythonOCC environment.

This is benchmark-infrastructure evidence. It is not learned-model evidence
and does not establish edit locality, reconstruction quality, or
generalization for any neural representation.

## Limitations and unsupported claims

- The replay output trees are not both retained, so the bundled statement
  that `diff -qr` found them byte-identical cannot be rerun from this archive
  alone.
- The primary bundle does not preserve scheduler accounting, the exact
  container hash, hardware, or working-tree status.
- The OpenCascade version field is null even though PythonOCC 7.5.1 is
  identified.
- The audit covers the controlled version-1 domain only.
- No trainable model or held-out learned-model partition was evaluated.

## Artifacts and integrity

| Artifact | Location | SHA-256 | Availability |
|---|---|---|---|
| Frozen audit archive | `/Users/krishaymaskara/research/audited-runs/counterfactual-audit-3ae4f6c-job3321612.tar.gz` | `527d14811fa4238ee84d4f318f37d956dbc4bf31ec7051ad90644167ceca9fa0` | Present locally |
| Primary corpus and reports | Archive namespace `job-3321612/` | Individual hashes not preserved | Present in archive |
| Replay corpus and reports | Original job namespace `job-3321613/` | Not preserved | Missing from archive |

The 482,536-byte gzip/tar archive extracted successfully, and its JSON
manifests and aggregate reports parsed during the July 28 documentation
audit.

## Reproduction and validation

The package README provides the generator and report-join commands. Exact
reproduction additionally requires the preserved configuration, seed 17,
source commit, and a PythonOCC 7.5.1-compatible environment.

The primary artifact was inspected and its aggregate counts were recomputed
from the contained reports. Reproduction was not rerun during documentation
backfill. Byte identity between the original jobs is a bundled reported
result, not an independently repeated check.

## Related records

- Package contract: [Counterfactual CAD edit pairs](../../prototype/counterfactual_edits/README.md)
- Kernel prerequisite: [Historical 60-family kernel validation](kernel_validation_60.md)
- Milestone: [Controlled CAD foundation](../milestones/controlled_cad_foundation.md)

