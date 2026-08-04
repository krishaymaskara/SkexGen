# Graph V1 Research Handoff Questions — August 3, 2026

This dated handoff accompanies the
[Graph V1 progress report](graph_v1_progress_report_2026-08-03.md). It records
open questions rather than current status or an authorized experiment
protocol.

The questions below are prompts for the next research discussion. They intentionally do not prescribe answers or authorize additional graph corrections, pilot reruns, or partition access.

## 1. Scientific interpretation

- Why did the pairwise graph decoder exactly solve single-operation programs but no two-operation programs?
- Why did C1 improve local edge metrics while reducing `depends_on` recall?
- Which conclusions are supported by the unchanged exact-match result despite improved edge-level precision and F1?
- Is the position-only prior revealing dataset leakage, useful serialization structure, or both?

## 2. Architectural implications

- Does independent typed-edge classification provide enough context to enforce globally consistent CAD operation dependencies?
- Which architectural assumptions should be reconsidered in light of the gap between local edge metrics and complete-program validity?
- What should remain frozen to preserve the validity of the comparison?

## 3. Whether to redesign the representation

- Does the current serialized node representation expose topology in a way that obscures whether a model has learned graph structure?
- Would a representation designed around permutation or order perturbations better distinguish relational learning from serialization learning?
- Which representation changes could be evaluated without conflating them with decoder-capacity changes?

## 4. Whether to study operation grouping explicitly

- Does multi-operation CAD require a hierarchical operation-group representation rather than independent pair classification?
- What supervision or diagnostic would distinguish operation grouping failure from individual edge classification failure?
- How should Boolean-sequence dependencies be represented if local pair decisions are insufficient?

## 5. Whether VQ collapse should become the next isolated experiment

- Should the next isolated experiment target latent collapse rather than graph decoding?
- What evidence would establish that VQ collapse is causally limiting program-level generalization?
- Which decoder and evaluation components would need to remain frozen during an isolated VQ study?

## 6. How to present the result in a research report

- How should the single-operation success and two-operation failure be presented without overstating graph generalization?
- How should the edge-level improvement and unchanged exact validity be reported together?
- Which tables or failure examples best communicate the `depends_on` tradeoff?

## 7. What evidence would be required before accessing systematic data

- What result would justify systematic-partition evaluation?
- Which hypothesis, endpoint, and decision rule would need to be preregistered before access?
- Which model, checkpoint, metrics, and analysis code would need to be frozen first?

## 8. What evidence would be required before accessing held-out test data

- What independent evidence would justify held-out test-partition access?
- Which systematic-partition result and precommitted acceptance rule would be required first?
- What safeguards would prevent test results from motivating further model selection or graph corrections?
