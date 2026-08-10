# ADR-0009: GE1 Positive Operation-Magnitude Repair

- Status: `accepted`
- Proposal date: `2026-08-10`
- Decision date: `2026-08-10`
- Owner: project research team
- Designated GE1 reviewer: Krishay Maskara
- Adds to: [ADR-0008](ADR-0008-ge1-c7-v2-200-epoch-protocol.md)
- Prospectively supersedes: only the hierarchical operation-group repair
  authorization in [ADR-0004](ADR-0004-ge1-single-manifest-encoder-comparison.md),
  [ADR-0006](ADR-0006-ge1-c7-sufficiency-execution-contract.md),
  [ADR-0008](ADR-0008-ge1-c7-v2-200-epoch-protocol.md), and
  `GE1-STAGE0-PREREG-v1`
- Preserves without reinterpretation: C7-v2 commit
  `e325d5ad97957c08da4a19b4261560e8a4a472a4`, Adroit job `3344981`, and
  operation-parameter diagnostic commit
  `802ae1d1e9deb3c7a6c428d320e4276a8b5e7e57`, Adroit job `3345013`
- Superseded by: none

## Context

C7-v2 completed as an immutable train-only scientific failure. Both encoder
arms reconstructed all 32 scaled node sequences and typed graphs exactly,
recovered every applicable `depends_on` relation, passed strict conversion,
and passed the memory-use gates. Flat failed analytic complete validity on two
families and typed graph failed on three.

The separately authorized, read-only operation-parameter diagnostic reproduced
the original deterministic batches and stable metrics exactly before tracing
the five failures. Every failure was an active negative extrusion-distance or
revolve-angle magnitude. Geometry masks, selected channels, scale-only
normalization and denormalization, units, direction categories, Boolean modes,
targets, and alignment were consistent. The other encoder arm produced an
analytically legal positive value for every same-family control.

The earlier hierarchical operation-group repair targets node grouping and
typed structural relations. Those outputs were exact in C7-v2, so that repair
is not targeted to the observed failure mechanism.

## Target and decoder contract audit

The controlled representation defines extrusion distance as a positive
magnitude with direction represented categorically. Revolve angle lies in
`(0, 360]` degrees. Model-data channel 37 divides extrusion distance by the
frozen length scale `4.0`; channel 38 divides revolve angle by the frozen angle
scale `360.0`.

The controlled generator uses extrusion distances
`(0.5, 1.0, 1.5, 2.0, 3.0)` and revolve angles
`(45.0, 90.0, 180.0, 270.0, 360.0)`. Therefore every active operation-
magnitude target is finite, strictly positive, and no greater than `1.0`
after normalization. Exact normalized `1.0` targets do occur: a 360-degree
revolve maps to `360 / 360 = 1.0`. The frozen controlled extrusion set does
not reach `1.0`; its maximum is `3 / 4 = 0.75`.

The existing GE1 decoder applies symmetric `tanh` to all six learned remaining-
geometry channels, including compact channels 4 and 5 that scatter to
serialized channels 37 and 38. That mapping permits negative operation
magnitudes even though their semantic target domain is `(0, 1]`.

## Decision

### Prospective repair identity

The repaired parameterization identity is:

```text
GE1-OPERATION-MAGNITUDE-POSITIVE-v1
```

The historical behavior remains explicitly identifiable as:

```text
GE1-OPERATION-MAGNITUDE-TANH-LEGACY-v1
```

The repaired identity is prospective. It must not be described as C7-v2,
must not alter the finalized job `3344981` result, and must not alter the
read-only evidence from job `3345013`.

### Neural parameterization

Only the active extrusion-distance and revolve-angle outputs change. For a
finite raw scalar `x` and floating-point dtype `d`, the repaired mapping is:

```text
epsilon(d) + (1 - epsilon(d)) * sigmoid(x)
```

where `epsilon(d) = torch.finfo(d).tiny`, the smallest positive normal value
for that dtype. This guarantees a finite normalized result in `(0, 1]`, even
when a very negative finite input causes `sigmoid(x)` to round to zero. Exact
`1.0` remains representable when a very large finite input causes the sigmoid
to round to one, matching the observed 360-degree target endpoint.

All other learned remaining-geometry channels retain the exact legacy `tanh`
mapping. The repair occurs in the neural decoder before loss calculation and
autonomous reconstruction. It is not converter-side correction, evaluator-
side clamping, absolute value, ReLU, post-hoc replacement, or unbounded
softplus.

The same implementation and constants apply to both encoder arms. Direction
and Boolean prediction remain unchanged and separate from magnitude. Raw
pre-activation values remain available for diagnostic reporting.

### Preserved system

The decision changes none of the following:

- flat or position-free typed-graph encoders;
- node, profile, typed-edge, or other geometry decoding;
- geometry masks or channel selection;
- normalization and denormalization scales;
- direction or Boolean prediction;
- loss definitions, loss weights, optimizer, data, seeds, cohorts, budgets,
  gates, or protected-access rules;
- the frozen inherited V6 implementation; or
- the absence of chronological-position input to the typed graph encoder.

### Configuration and checkpoint boundary

Configuration, inference checkpoints, recovery checkpoints, provenance,
artifacts, and reporting records must name the operation-magnitude
parameterization identity. Repaired and legacy configurations are distinct.
Strict load must reject a checkpoint whose recorded parameterization differs
from the receiving model, using a stable checkpoint-identity error. There is
no implicit legacy fallback.

The legacy configuration exists only to reproduce and test the historical
`tanh` behavior. New repair validation uses the positive identity. No C7-v2
checkpoint is relabeled or converted.

### Authorization boundary

This ADR authorizes implementation and exact-commit engineering validation of
the prospective operation-magnitude repair only. It does not authorize:

- scientific training or a repaired C7 execution;
- Stage 6 or C8;
- any corpus, manifest, or payload access during implementation validation;
- development, RR, ER, IID, history-depth, or geometry-extrapolation access;
- CAD-kernel integration; or
- pushing a branch.

CAD-kernel integration remains a separately governed future change. Analytic
positivity is not executor validity.

## Alternatives considered

### Hierarchical operation-group repair

Superseded for the present repair path. C7-v2 already achieved exact nodes,
graphs, dependencies, and strict conversion, while all five observed failures
were continuous negative magnitudes.

### Absolute value, ReLU, or post-hoc clamping

Rejected. These mappings either create non-differentiable/boundary behavior or
hide invalid neural outputs outside the decoder and loss path.

### Unbounded softplus

Rejected. It violates the frozen normalized upper bound unless followed by a
second contract-changing mapping.

### Change loss weights, direction, units, or normalization

Rejected by the available evidence. The diagnostic found those paths
consistent and did not isolate a loss-weighting defect.

## Consequences

- The observed negative-magnitude failure class becomes impossible under the
  repaired neural output contract.
- Positive analytic magnitude does not prove numeric target accuracy or CAD-
  kernel executability.
- Historical and repaired checkpoints cannot be silently mixed.
- A separately frozen prospective scientific protocol is still required after
  authoritative implementation validation.
- Stage 6 remains blocked and every protected partition remains closed.

## Required implementation evidence

Implementation tests must prove the legacy path is exact, the repaired path is
finite and in `(0, 1]` across extreme finite logits, only the two operation
channels change, masks/categories remain invariant, the five synthetic
failure-shaped cases avoid `invalid_operation_parameter`, identities propagate
through configuration/checkpoint/reporting records, incompatible loads fail,
both arms retain matched shared-decoder initialization, and no chronological
position or corpus access is introduced.

## Related records

- [ADR-0004](ADR-0004-ge1-single-manifest-encoder-comparison.md)
- [ADR-0006](ADR-0006-ge1-c7-sufficiency-execution-contract.md)
- [ADR-0008](ADR-0008-ge1-c7-v2-200-epoch-protocol.md)
- [C7-v2 execution contract](../specifications/ge1_c7_v2_execution_contract.md)
- [Operation-parameter diagnostic](../specifications/ge1_c7_v2_operation_parameter_diagnostic.md)
- [GE1 implementation plan](../specifications/graph_encoder_implementation_plan.md)
