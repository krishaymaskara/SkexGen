# Milestone: Controlled CAD Foundation

## Status and scope

Accepted as an infrastructure milestone on July 28, 2026.

This stage established the controlled data and interface foundation needed for
flat-versus-graph representation experiments:

- a versioned typed CAD-history schema;
- deterministic extrude/revolve history generation;
- family-level systematic split authority;
- analytical Boolean-feasibility filtering;
- independent OpenCascade execution validation;
- deterministic one-factor counterfactual edit pairs;
- a shared flat/graph model-data interface.

Acceptance means these contracts are checked in, their bounded validation
evidence is documented, and downstream models can consume the same data
authority. It does not mean a learned representation has passed a scientific
benchmark.

## Implementation landmarks

| Capability | Commit |
|---|---|
| Typed CAD representation | `1a13b73` |
| Controlled CAD generator | `7f216c0` |
| CAD-kernel execution validation | `fc1a840` |
| Analytical Boolean feasibility | `0f0a80d` |
| Counterfactual edit benchmark | `3ae4f6c` |
| Shared model-data interface | `91d733d` |
| Adroit PyTorch validation infrastructure | `038e497` |

The normative schema and dependency directions are frozen in
[ADR-0001](../decisions/ADR-0001-controlled-cad-schema-v1.md). Current
behavior belongs in the package READMEs linked from the
[documentation index](../README.md).

## Acceptance evidence

| Evidence | What it establishes |
|---|---|
| [Historical 60-family kernel validation](../experiments/kernel_validation_60.md) | The early generator executed under a real CAD kernel and exposed Boolean failures that motivated analytical filtering. |
| [Counterfactual OpenCascade audit](../experiments/counterfactual_opencascade_audit.md) | All 272 encoded endpoints executed; all 136 edit samples succeeded; continuous and quantized physical endpoints agreed. |
| [Real-PyTorch model-data validation](../experiments/model_data_pytorch_validation.md) | The shared interface passed 193 tests and constructed real PyTorch 1.11 tensors under Python 3.8 on Adroit. |
| [680-family B0 corpus](../experiments/b0_pilot_corpus_680.md) | A deterministic 544/68/68 family-level train/validation/test corpus was generated for the first baseline. |

The counterfactual and model-data primary bundles are locally
checksum-verified. The historical kernel record and corpus record remain
documented external evidence rather than locally complete artifact bundles.

## Supported conclusions

- The controlled version-1 schema and generator provide a deterministic,
  family-grouped basis for the planned comparisons.
- The selected counterfactual corpus is executable in the preserved
  PythonOCC environment and suitable as an evaluation-only benchmark.
- The shared interface can provide flat and graph views without changing
  split authority, and its tensor conversion works in the target PyTorch
  environment.
- The 680-family B0 corpus preserves a held-out 68-family test partition that
  has not yet been used.

## Limitations and deferred work

- The domain is intentionally restricted to single-body sketch, extrude, and
  revolve histories.
- Historical 60-family kernel artifacts and the complete 680-family corpus
  bundle were not recovered into the local audited-run store.
- The counterfactual audit does not evaluate learned edits or causal locality
  in a neural representation.
- At this milestone's acceptance date, no graph model had been implemented;
  later graph work is tracked in the
  [constrained flat/graph milestone](constrained_flat_graph_comparison.md).
- OpenCascade execution for learned model predictions remains a future
  evaluation gate.

## Next gate

The next milestone was a credible flat mixed/VQ baseline using this shared
data authority. Its implementation, collapse diagnosis, repair, and current
acceptance boundary are summarized in the
[flat-baseline milestone](flat_baseline.md).
