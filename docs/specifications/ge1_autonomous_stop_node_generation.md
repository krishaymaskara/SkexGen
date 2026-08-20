# GE1 Autonomous Stop Node-Generation Contract

## Status and authority

| Item | Frozen value |
|---|---|
| Status | Accepted; implementation and bounded validation only |
| Legacy identity | `GE1-REQUESTED-COUNT-GRAMMAR-MASKED-v1` |
| New identity | `GE1-PAD-TERMINATED-UNCONSTRAINED-NODES-v1` |
| Contract | `GE1-AUTONOMOUS-STOP-NODE-GENERATION-CONTRACT-v1` |
| Terminator | `<pad>` / node type id `0` |
| Shared decoder | `GE1-SHARED-TYPED-EDGE-DECODER-V4` |
| Output positions | `GE1-DECODER-OUTPUT-POSITIONS-v3` |
| Checkpoint schema | `GE1-CHECKPOINT-v4` |
| Generation cap | `max_nodes = 16` attempted output positions |
| Authority | accepted [ADR-0016](../decisions/ADR-0016-ge1-autonomous-stop-and-unconstrained-node-decoding.md) |
| Scientific run / Stage 6 / protected access | unauthorized / unauthorized / unauthorized |

## Training target contract

For each target containing `N` real graph nodes, the new identity exposes a
node-type target of width at least `N + 1`:

```text
node_type_ids[0:N] = real node types
node_type_ids[N]   = NODE_TYPES.pad_id
node_mask[0:N+1]   = True
node_mask[N+1:]    = False
```

Exactly one `<pad>` is active. Categorical, Boolean, geometry, and geometry-mask
values at that position are neutral. Operation targets and graph edges are not
extended. The inherited node-type cross-entropy is the sole stopping loss.

Encoder inputs and graph bookkeeping retain the real-node count. The
terminator is decoder-output supervision only and never enters either encoder.

## Autonomous decode contract

For the new identity:

1. validate memory `[B,2,32]`;
2. start from BOS with no supplied node count;
3. take `argmax` over all eight node-type logits at each position;
4. if the result is `<pad>`, record learned termination and stop;
5. otherwise construct the predicted record and feed it back autoregressively;
6. stop after at most 16 attempted positions;
7. if no `<pad>` was observed, record `generation_cap_reached=True` and score
   the outcome as failed without raising.

No `legal_next_node_ids` result masks logits. Raw and selected node IDs are
identical, correction masks are false, and the legal-mask telemetry records
that no class correction was applied. Grammar validity is evaluated only by
the scorer/conversion boundary.

The terminating `<pad>` occupies output position `N` but is stripped before
`raw_nodes`. Pair logits and graph edges are computed on decoded states
`0:N`, with an all-true real-node mask. No pair touches position `N`.

## Legacy preservation

The legacy identity continues to require `node_counts` and a nonempty
`node_count_source`, calls frozen `greedy_decode_v6_from_memory`, applies the
exact-length V5 grammar mask, and returns the same prediction bytes for equal
weights and input memory. `prototype/flat_baseline` remains unchanged.

`legal_next_node_ids(prefix, requested_node_count)` retains exact-length
semantics. Its additive transition-only mode accepts no requested length and
is not used to mask the new identity; it exists for validation and diagnostics
without changing historical callers.

## Stage 6 scoring and optimization

`score_structural_prefix` continues to score structural evidence without
conversion or geometry. Its existing `predicted_operation_group_count`,
`under_generation`, `exact_generation`, and `over_generation` fields now
receive genuinely autonomous lengths. Grammar-invalid node sequences must set
`grammar_valid_complete=False` and cannot receive full credit.

`optimization_reliability` keeps arm-internal finite/gradient/plateau/fixed-
checkpoint checks. Train score and cross-arm difference records remain
diagnostic, but no cross-arm ceiling threshold contributes to `pass`.

## Validation requirements

- new-identity generation can emit a grammar-invalid real-node sequence and
  the structural grammar criterion fails;
- learned under-generation and over-generation populate the existing score
  fields correctly;
- cap reached is recorded, returned, and scored as failure;
- exactly one terminator position contributes to node-type loss;
- the terminator never enters edge pairs;
- node raw/selected IDs agree and correction is identically false;
- every legacy identity still uses counts and exact-length masking with
  byte-identical autonomous output;
- the frozen source file matches commit `12521f6`;
- focused, complete, sibling, documentation, compilation, Python 3.8 grammar,
  and source-audit gates pass in the authoritative CPU environment.

## Non-authorization

The checked-in implementation and validation evidence do not authorize model
training, Stage 6, development, RR, ER, Stage 7, C8, protected access, or a
scientific interpretation. Autoregressive conditioning is an acknowledged
residual bypass channel and is outside this contract.
