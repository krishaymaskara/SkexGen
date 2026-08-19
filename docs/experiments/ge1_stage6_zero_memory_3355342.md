# GE1 Stage 6 Zero-Memory Diagnostic, Job 3355342

## Audit disposition

Job `3355342` is valid with qualifications as a completed train-only,
read-only diagnostic at exact commit
`2eaabf2e5a88fc62f559c914123694539a93e188`. The supplied local evidence
contains stdout, stderr, `sacct`, and the complete five-file
`GE1-STAGE6-ZERO-MEMORY-ARTIFACT-v1` artifact. Local checksum, manifest,
record, score, coverage, and aggregation checks passed.

The qualification is evidence-bundle scope. The committed verifier requires
the separate job-3355134 postmortem artifact, which was not included in this
download and was not opened during this audit. The job stdout records that the
exact verifier ran with that input and returned `verification_status="pass"`;
the embedded prior identities and hashes are internally consistent but cannot
be independently reauthenticated from the downloaded five files alone.
The local Python 3.14 analyzer differs from the stored Python 3.8 summary by
one ULP only for the typed-graph equal-template macro (`0.5833333333333333`
versus `0.5833333333333334`) because the Python versions accumulate `sum`
differently. Python 3.8-style sequential accumulation reproduces the stored
value; no displayed six-decimal result or interpretation changes.

| Item | Verified value |
|---|---|
| Job | `3355342` |
| Diagnostic commit | `2eaabf2e5a88fc62f559c914123694539a93e188` |
| Diagnostic / artifact | `GE1-STAGE6-ZERO-MEMORY-DIAGNOSTIC-v1` / `GE1-STAGE6-ZERO-MEMORY-ARTIFACT-v1` |
| Protocol | `GE1-STAGE6-STRUCTURE-ONLY-COMPARISON-v1` |
| State / exit | `COMPLETED` / `0:0` |
| Elapsed / batch MaxRSS | `00:04:09` / `786736K` |
| Runtime | Python `3.8.13`, PyTorch `1.11.0`, CPU only, one thread |
| Focused tests | 14 declared, 14 run, 0 skipped; all passed |
| Records | 2 arms x 3 seeds x 407 train families = 2,442 |
| Artifact `SHA256SUMS` SHA-256 | `a3ed6bd494b10d2065b0a47434efd125e4fe01780add260a75b1e726b979d984` |

The transcript and artifact source record establish a clean, standalone,
detached checkout at the exact commit. Documentation validation covered 103
Markdown files and Python 3.8 AST parsing covered 101 Python files. Bytecode
compilation, Bash syntax, committed-whitespace validation, and the
zero-memory structural/access audit were silent `set -e` preflights before
the diagnostic; terminal success establishes that they passed. The access
audit recorded one diagnostic invocation, ten exact binds, six checkpoint-
file binds, one train-loader call, and zero training, backward, optimizer,
development-loader, protected-bind, or submission calls. Stderr contains only
the expected unittest progress stream ending in `OK`.

## Descriptive result

`P_mean` is the authenticated job-3355134 value; it was not recomputed in this
job. The corresponding authenticated `P_shuffle` evidence and complete
three-condition tables are in the [job-3355134 postmortem
record](ge1_stage6_train_gate_postmortem_3355134.md). All other displayed
values were recomputed from the 2,442 supplied records. This diagnostic
creates no threshold or new gate.

| Arm | Seed | Support | P_true | P_zero | P_mean | P_zero / P_true | P_mean / P_true | P_zero - P_true |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| flat | 2026 | 407 | 1.000000 | 0.756757 | 0.705160 | 0.756757 | 0.705160 | -0.243243 |
| typed graph | 2026 | 407 | 1.000000 | 0.756757 | 0.808354 | 0.756757 | 0.808354 | -0.243243 |
| flat | 2027 | 407 | 1.000000 | 1.000000 | 0.855037 | 1.000000 | 0.855037 | 0.000000 |
| typed graph | 2027 | 407 | 1.000000 | 0.501229 | 0.847666 | 0.501229 | 0.847666 | -0.498771 |
| flat | 2028 | 407 | 1.000000 | 0.756757 | 0.810811 | 0.756757 | 0.810811 | -0.243243 |
| typed graph | 2028 | 407 | 1.000000 | 0.498771 | 0.852580 | 0.498771 | 0.852580 | -0.501229 |
| flat aggregate | all | 1,221 | 1.000000 | 0.837838 | 0.790336 | 0.837838 | 0.790336 | -0.162162 |
| typed-graph aggregate | all | 1,221 | 1.000000 | 0.585586 | 0.836200 | 0.585586 | 0.836200 | -0.414414 |

Per-seed equal-template macro `P_zero` values were `0.75`, `1.00`, and
`0.75` for flat and `0.75`, `0.50`, and `0.50` for typed graph. Aggregate
equal-template macros were `0.833333` and `0.583333`, respectively. These are
descriptive secondary summaries, with four templates and no validity gate.
Individual record scores ranged from zero to one in every arm/seed except flat
seed 2027, where all 407 scores were one.

By template across all three seeds, flat `P_zero` was `1.0` for E, R, and EE
and `0.333333` for RE. Typed-graph `P_zero` was `0.666667` for E, R, and EE
and `0.333333` for RE. Template supports were 306 E, 315 R, 303 EE, and 297
RE records per arm.

## Licensed interpretation and boundary

Exact-zero encoder memory still left structurally informative train output:
the pooled score was `0.837838` for flat and `0.585586` for typed graph. This
is consistent with information supplied by unchanged decoder-side state,
including node counts, positions, masks, constraints, and scalar paths. It
does not identify which such input is causal.

True-to-zero drops of `0.162162` and `0.414414` show that train performance is
not wholly independent of encoder memory, while the nonzero residuals show it
is not wholly determined by the memory values either. Relative to authenticated
`P_mean`, flat `P_zero` was higher by `0.047502`, while typed-graph `P_zero`
was lower by `0.250614`. No prospective rule makes those differences
significant or material. The typed-graph contrast is consistent with the
batch-mean vector retaining useful information beyond the exact-zero decoder
baseline; the high flat zero baseline helps explain part of the earlier weak
train-memory sensitivity. It does not fully explain the prior gate, replace
its shuffled-memory evidence, or change its result.

This train-only diagnostic supports no encoder-superiority, development,
systematic, held-out, defect-localization, information-absence, or capacity-
exhaustion claim. It creates no threshold, changes no ADR-0014 gate, and does
not authorize repair, another training or inference job, Stage 6 production,
C8, RR, ER/test, development, or any protected access.
