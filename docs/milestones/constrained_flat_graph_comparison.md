# Milestone: Constrained Flat and Graph V1 Comparison

## Status and scope

Accepted as a completed bounded IID-validation comparison on August 3, 2026.

This milestone delivered a shared constrained node/geometry path, a frozen
Flat V6 structural baseline, an effectively capacity-matched graph-native
edge decoder, one preregistered evidence-backed graph correction, strict
program- and edge-level evaluation, and frozen reproducibility identities.

Acceptance is narrow. The milestone establishes an IID-validation structural
comparison and a useful failure boundary. It does not satisfy ADR-0002's
remaining systematic-generalization or localized-edit evaluation goals and
does not authorize held-out test access.

## Implementation landmarks

| Capability | Commit |
|---|---|
| Deterministic profile geometry | `f1a3cb7` |
| Torch profile geometry | `e82e891` |
| Neural constrained-profile heads and losses | `3f565df` |
| Constrained Flat V2 | `c63d206` |
| Canonical reference planes | `0127690` |
| Node-conditioned categorical selection | `a49941f` |
| Prefix-conditioned node grammar | `e6dc32e` |
| Canonical controlled revolve axis | `322add1` |
| Frozen Flat V6 | `ac6ef71` |
| Initial graph-native decoder | `6484579` |
| Frozen initial Graph V1 | `089b9f3` |
| Directed-position correction C1 | `0cd09ed` |

Current implementation contracts live in the
[flat](../../prototype/flat_baseline/README.md) and
[graph](../../prototype/graph_baseline/README.md) package READMEs.

## Evidence chain

| Stage | Evidence and decision |
|---|---|
| Failure motivation | [Categorical-isolation replay](../experiments/b0_phase_b_categorical_isolation_replay.md) localized profile-family/category-inconsistent geometry. |
| Bounded flat repair | [V2-V5 progression](../experiments/b0_constrained_flat_v2_v5_progression.md) removed profile, plane, category, and node-grammar blockers. |
| Frozen flat condition | [Flat V6](../experiments/b0_constrained_flat_v6.md) reached 68/68 conversion but 0/68 validity, with every first failure at structural edges. |
| Engineering readiness | [Graph V1 readiness and C1 validation](../experiments/b0_graph_v1_engineering_readiness.md) preserved all failed pre-pilot jobs, provenance repairs, and the authoritative 17-test C1 validation. |
| Initial graph condition | [Initial Graph V1](../experiments/b0_graph_v1_initial.md) reached 22/68 validity, solving all 22 single-operation and no two-operation programs. |
| Single correction | [Graph V1 C1](../experiments/b0_graph_v1_c1.md) improved edge F1 but left exact validity unchanged at 22/68. |

## Supported conclusions

- Category-conditioned deterministic construction can eliminate invalid
  profile, plane, categorical-applicability, node-grammar, and controlled-axis
  failure classes in this bounded domain.
- Explicit typed-edge prediction materially improves complete validity over
  the frozen flat relation/pointer heads on IID validation.
- Initial Graph V1 and C1 both solve every single-operation program and no
  two-operation program.
- The independent pairwise decoder learns legal local relationships but not
  globally consistent repeated-instance grouping.
- C1 improved positive-edge precision and F1 but reduced `depends_on` recall,
  demonstrating that edge-level improvement need not improve CAD validity.
- A position-only scoring prior exposes substantial serialization structure
  that the learned decoders did not recover.

## Limitations and claims not yet supported

- Neither systematic nor held-out test performance is known.
- The models are serialization-aware; order-independent graph learning is not
  established.
- VQ collapse recurred in both graph pilots.
- There is no multi-seed or beyond-two-operation result.
- Local graph archives preserve metrics and scheduler logs but not checkpoint
  binaries.
- The constrained-flat V2-V6 artifact chain is not fully recovered locally.

## Next gate

Graph V1 is frozen. A separately named protocol must choose one isolated next
question: explicit multi-operation grouping/dependency decoding or the causal
role of latent collapse. It must freeze a program-level acceptance rule,
correction budget, capacity/training controls, and the criterion for first
systematic-partition access. Held-out test data remains untouched.
