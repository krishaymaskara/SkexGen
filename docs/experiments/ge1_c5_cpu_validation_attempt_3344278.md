# GE1 C5 Authoritative CPU Validation Attempt 3344278

## Record status

| Item | Value |
|---|---|
| Experiment ID | `ge1-c5-cpu-3344278` |
| Record status | `verified failed validation attempt` |
| Scientific role | Infrastructure and compatibility validation of C5 |
| Branch | `graph-v1-experiment-record` |
| Source commit | `ffa6091cfecb37aa648ed3590f591f247a776754` |
| Source parent | `a952d2f5775f39f67a8d33d06c3530c62a9d11a4` |
| Slurm job | `3344278` |
| Compute host | `adroit-h11n2` |
| Execution date | August 7, 2026 |
| Decision | Failed; C5 remains authoritatively unvalidated |

## Question and predetermined decision rule

The validation asked whether exact C5 commit
`ffa6091cfecb37aa648ed3590f591f247a776754` satisfies the complete shared
decoder acceptance gate under Python 3.8 and PyTorch 1.11 on CPU. Acceptance
required every focused and complete-suite test to pass with the frozen counts
and zero skips, followed by every regression, documentation, compilation,
grammar, and source-integrity check named by the runner. Any failure, error,
skip, count mismatch, environment mismatch, dirty tree, or missing terminal
success marker meant that C5 remained unvalidated.

## Gate and environment

The runner required a clean checkout of the exact C5 commit, Python 3.8,
PyTorch 1.11, CPU execution, 22/22 focused C5 tests with zero skips, 110/110
complete graph-encoder tests with zero skips, all listed regression suites,
documentation validation, compilation, Python 3.8 grammar parsing, and a clean
unchanged checkout after testing.

The verified environment was:

| Item | Observed value |
|---|---|
| Python | CPython 3.8.13 |
| PyTorch | 1.11.0, CUDA build 11.3 |
| CUDA available | `false` |
| Device | CPU |
| CPU threads | 1 |
| Container | `/scratch/network/km6349/skexgen.sif` |
| Scheduler state | `FAILED`, exit `1:0` |
| Elapsed / peak RSS | 14 seconds / 469,696 KiB |

The checkout and bundle both resolved to
`ffa6091cfecb37aa648ed3590f591f247a776754`. The source tree was clean at
runner start. The fail-fast runner exited after the focused suite, so it did
not reach its final clean-tree check.

## Focused-suite result

The focused suite ran all 22 tests with zero skips and zero errors. Nineteen
passed and three assertions failed:

1. `test_exact_constrained_graph_contract_and_conversion_for_main_path`
   compared complete prediction envelopes. The only reported difference was
   `latent_indices`: retained Graph V1 carried the fake compatibility VQ tuple
   `(0, 0)`, while continuous GE1 correctly carried `()`. Latent indices are
   encoder/VQ provenance, not a decoder or constrained-graph output. Exact
   node/geometry tensors and main pair logits had already passed.
2. `test_autonomous_levels_are_separate_target_free_and_deterministic` checked
   nonexistent `ConversionResult.first_failure` and
   `ConversionResult.additional_failures` attributes. The authoritative record
   names are `primary_failure` and `secondary_failures`. Before reaching this
   assertion, repeated autonomous constrained and converted predictions were
   equal, result levels were distinct, and no raised conversion failure was
   observed.
3. `test_padding_and_masked_pairs_contribute_zero` assumed the E fixture was
   batch row zero. C3 intentionally sorts paired families by physical family
   ID, so row zero was the five-node R fixture in this run. The test therefore
   changed an active node and correctly changed its loss. The independent
   common-loss parity test passed.

These are test-contract defects. They do not identify a production decoder,
loss, conversion, checkpoint, initialization, or routing difference.

The 19 passing focused tests include exact constrained V6 node/geometry
intermediate parity, exact initial Graph V1 main-pair-logit parity, exclusion
of the C1 bias, common Graph V1 per-example loss parity, finite forward loss
and gradients, strict checkpoint round-trip and rejection behavior, matched
decoder state, disjoint parameter and buffer objects, mutation independence,
arm-order-independent decoder initialization, common-memory-only routing,
target-free signatures, frozen capacity, output-position inventory, Python
3.8 grammar, and protected-import guards.

## Correction and rerun requirement

The focused C5 correction:

- normalizes retained reference `code_indices` to the shared continuous-memory
  empty tensor before comparing complete constrained graph records;
- checks the actual authoritative conversion-result field names; and
- derives the padded row and position from `target["node_mask"]` after C3
  sorting instead of assuming input order.

It changes no production source, decoder formula, loss formula, checkpoint
schema, model configuration, position signal, partition policy, or protected
access rule. C5 must be rerun at the exact focused correction commit. This
failed attempt cannot be promoted to an authoritative C5 pass.

## Decision

Job `3344278` failed the predetermined gate. The focused suite did not pass,
and fail-fast execution did not reach the complete graph-encoder suite,
regressions, documentation check, compilation, grammar check, or final
clean-tree assertion. C5 is therefore not authoritatively validated.

## Interpretation

The evidence supports a narrow implementation conclusion: nineteen focused
C5 contracts passed, and the three reported differences are defects in the
test assertions rather than evidence of a production-code mismatch. That
diagnosis justifies a test-only correction and exact-commit rerun. It does not
convert this failed job into proof of C5 parity or runtime acceptance.

## Limitations and claims not supported

The run stopped at the first failed suite, so it provides no new evidence for
the later gates in the runner. It does not validate the corrected tests, the
complete graph-encoder suite, training, comparative encoder performance,
systematic generalization, held-out test behavior, editing, VQ, or any later
GE1 stage. Artifact hashes establish integrity of the retained local copies,
not independent cluster provenance or container identity.

## Access and scope audit

Runtime markers report:

```text
corpus manifest accessed: false
corpus CAD-history payload accessed: false
systematic partition accessed: false
test partition accessed: false
scientific training performed: false
C7 performed: false
C8 performed: false
```

The focused suite used procedural model fixtures and static source checks.
Because the fail-fast gate stopped immediately, later regression optimizer
smokes did not run.

## Artifacts

| Artifact | Size | SHA-256 |
|---|---:|---|
| `c5-cpu-3344278.out` | 922 bytes | `dfae383e24125bd0dc92d6c04775444f47aad1a67654328d4762d8ddb8cb944e` |
| `c5-cpu-3344278.err` | 6,246 bytes | `dfa931889e619855b14fb7a05ffe07f79b04a92935ba3d9575e0ef4c665ae0d1` |
| Complete terminal transcript | 6,793 bytes | `0c20bda855b2f18c280f56739e3f5831c24cd6d59a2338d1d86ed28a2fad76ee` |
| `SkexGen-c5-ffa6091.bundle` | 77,915,331 bytes | `be8c297fa2616ea6968e2b84b1fda4a9be957fa7e8d4c732e0341373083e0ec3` |

The stdout and stderr are retained under
`/Users/krishaymaskara/Downloads/c5-validation-3344278/`. The complete bundle
is retained under `/Users/krishaymaskara/Downloads/c5-ffa6091-transfer/`.

## Related records

- [C5 shared decoder contract](../specifications/ge1_shared_decoder_contract.md)
- [GE1 implementation plan](../specifications/graph_encoder_implementation_plan.md)
- [Current project status](../status.md)
- [C4 review-fix validation](ge1_c4_review_fix_cpu_validation.md)
