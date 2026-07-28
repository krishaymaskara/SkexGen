# Experiment Record: <Short descriptive name>

## Record status

| Item | Value |
|---|---|
| Experiment ID | `<stable-id>` |
| Record status | `draft`, `verified`, or `superseded` |
| Scientific role | `<infrastructure validation, pilot, ablation, diagnosis, evaluation>` |
| Branch | `<branch>` |
| Source commit | `<full 40-character commit>` |
| Working tree | `clean` or `dirty` |
| Slurm/local run ID | `<job ID or local identifier>` |
| Execution date | `<YYYY-MM-DD>` |

If the working tree was dirty, record the exact source-tree hash, status,
patch hash, or explicitly state which provenance was not preserved. Never
describe a dirty-tree run as validation of a later clean commit.

## Question and predetermined decision rule

State the narrow question this experiment answers. Record acceptance,
rejection, stopping, and checkpoint-selection rules as they existed before
examining the result.

## Inputs and partition authority

| Item | Value |
|---|---|
| Dataset/corpus | `<logical name and path>` |
| Corpus/configuration hash | `<hash or unavailable>` |
| Split manifest | `<name and version>` |
| Train families used | `<count or none>` |
| Validation families used | `<count or none>` |
| Test families used | `<count or none>` |
| Test partition evaluated | `true` or `false` |

Explain the learning/evaluation unit and how duplicate encodings or related
counterfactual endpoints are prevented from crossing partitions.

## Configuration

Record the complete configuration or link to an immutable configuration
artifact. Include seeds, model dimensions, optimization settings, evaluation
conditions, and any intentional deviation from the preceding baseline.

## Environment

Record the Python, PyTorch, CUDA, PythonOCC/OpenCascade, hardware, container,
and scheduler information relevant to interpreting or reproducing the run.
Use `not applicable`, `unavailable`, or `not preserved` rather than guessing.

## Execution

Provide the exact invocation or identify the committed runner that contains
it. Record the output namespace and collision/resume behavior.

## Verified results

Report primary metrics, diagnostics, completion state, and failure categories.
Separate directly verified artifact values from values copied from an
external report or historical summary.

## Decision

State the outcome using the predetermined rule. Distinguish engineering
success, benchmark validity, training-gate success, and scientific evidence.

## Interpretation

Explain what the result supports and why. Avoid causal or generalization
claims that the design does not isolate.

## Limitations and claims not supported

List important missing controls, unavailable evidence, restricted scope,
test-set restrictions, and conclusions that must not be drawn.

## Artifacts and integrity

| Artifact | Location | SHA-256 | Availability |
|---|---|---|---|
| `<name>` | `<path>` | `<hash>` | `present`, `external`, or `missing` |

Include the artifact-manifest hash when one exists. Scratch paths alone are
not durable evidence; preserve a checksum and archival location.

## Reproduction and validation

List commands needed to validate the preserved artifacts or reproduce the
experiment. State whether reproduction was actually performed and whether
outputs were byte-identical, numerically equivalent, or only semantically
consistent.

## Related records

- Prerequisite: `<link>`
- Supersedes or diagnoses: `<link>`
- Resulting decision/milestone: `<link>`
