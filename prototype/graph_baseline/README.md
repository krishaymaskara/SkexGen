# Graph-native controlled CAD baseline V1

The flat V6 result is frozen at commit
`ac6ef718ae9bab7fa5a80d9f48d0976adf5cafad`, pilot job `3338639`.
It generated grammar-valid nodes, canonical profiles, planes, and axes for all
68 IID-validation records, but every record first failed on `unexpected_edge`.
Graph V1 therefore changes only structural prediction.

The graph representation is `B0-CONTROLLED-CAD-TYPED-GRAPH-v1`. Nodes retain
the canonical chronological order used by model-data. Edges are directed from
the dependent/owned object to its referenced object. The ordered categorical
edge vocabulary is `<none>`, `defined_in`, `depends_on`, `placed_on`,
`uses_axis`, and `uses_profile`. Self-edges, duplicates, inactive-node edges,
and multiple edge types on one ordered pair are forbidden.

With zero-based chronological node indices, the six canonical templates are:

| Template | Nodes | Directed typed edges |
| --- | --- | --- |
| E | `plane, sketch, profile, extrude` | `1-placed_on->0`, `2-defined_in->1`, `3-uses_profile->2` |
| R | `plane, sketch, profile, axis, revolve` | E's ownership edges with `4-uses_profile->2`, plus `3-defined_in->1`, `4-uses_axis->3` |
| EE | `plane, sketch, profile, extrude, sketch, profile, extrude` | each sketch/profile/operation has its E edge; `6-depends_on->3` |
| ER | `plane, sketch, profile, extrude, sketch, profile, axis, revolve` | first E edges; second R edges; `7-depends_on->3` |
| RE | `plane, sketch, profile, axis, revolve, sketch, profile, extrude` | first R edges; second E edges; `7-depends_on->4` |
| RR | `plane, sketch, profile, axis, revolve, sketch, profile, axis, revolve` | both R edge groups; `8-depends_on->4` |

Here every sketch is `placed_on` node 0, every profile is `defined_in` its
sketch, and every operation `uses_profile` its local profile. The table's
shorthand expands those repeated edges exactly; it is documentation only and
is never used as an inference template.

In the existing representation these relationships are already serialized in
`edge_index`/`edge_type_ids`. `operation_sequence` separately records execution
order. The flat neural path duplicates that structure as an all-pairs
presence/type relation matrix plus operation-pointer logits: operation pointers
select the chronological operation nodes, while `depends_on` edges encode the
inter-operation dependency. Graph V1 predicts the typed edges once. Its inverse
keeps the generated node records, translates predicted graph class IDs back to
the authoritative model-data edge IDs, derives chronological operation indices
and pointers from generated operation nodes, and lets the independent strict
flat converter verify that those derived fields and predicted edges agree.
Canonical node order is therefore meaningful execution serialization and also
exposes enough position information for the documented scoring-only
position-template baseline to recover all controlled edges.

The V6 trainable flat structural modules—source/target relation projections,
edge-presence and edge-type heads, operation queries, and operation keys—have
3,496 parameters. The single corrected Graph V1 identity uses a 225→14→6 GELU
main pair MLP with 3,254 parameters plus a directed rank-6 ordered-position
bias with 228 parameters, for a 3,482-parameter graph decoder. Its 32,852 total
parameters differ from frozen V6's 32,866 by 14 (about 0.043%). Every
graph-decoder parameter participates in prediction.

Pair features, in order, are source and destination decoded states, their
constrained node-type embeddings, source and destination position embeddings,
mean quantized program memory, and signed relative serialized position. The
only production mask removes inactive/self pairs and edge types incompatible
with the generated source/destination node types. It never inserts a required
edge or selects a nearest template. Raw classes, masked classes, and every mask
correction remain reported separately.

Targets are used only by the per-example mean categorical cross-entropy and
scoring. Autonomous decoding accepts source records and authorized node counts
only. It generates the exact V6 constrained node path, predicts graph edges,
derives legacy operation records from the predicted graph, and submits the
result to an independent strict converter without repair.

The pilot is frozen to seed 2026, CPU, 544 train and 68 ordinary-IID validation
examples, two epochs, batch size eight, 136 optimizer steps, and 1,088 example
presentations. Primary outcomes are exact graph match, strict graph validity,
and complete CAD validity; pair accuracy alone is not evidence of improvement.
The documented node-type-pair and position-only rules are scoring-only
ablations and cannot affect production predictions. Because the explicit type
system admits at most one present edge class for any source/destination node
type pair, the edge-type-prior arm and node-type-pair-only arm coincide; both
names are still reported so that this fact is visible rather than omitted.

The graph experiment is frozen after the single evidence-backed correction
documented below. No further graph-model correction is permitted.

## Scientific correction C1

Initial Graph V1 pilot job `3341942` completed the frozen two-epoch protocol.
It produced 22/68 exact graphs and complete valid CAD programs: every
single-operation E and R example was valid, while all 46 two-operation examples
failed with `invalid_boolean_sequence`. Its 553 predicted positive edges versus
510 targets, 68/68 conversion success, empty structural-violation histogram,
and zero mask corrections ruled out all-none collapse and illegal-class
masking. The node-type-pair prior tied Graph V1 at 22/68, while the scoring-only
position prior reached 59/68. This supports one specific hypothesis: repeated
semantic instances need a more direct chronological alignment signal.

Correction C1 adds only
`ordered_position_bias(source_position, destination_position)` to the main pair
logits before the unchanged legal-class mask. Distinct learned 16×6 source and
destination factors are multiplied elementwise and projected by a bias-free
6×6 class projection. The branch consumes serialized source and destination
indices only; it does not consume targets, templates, canonical edges, family
labels, or validation statistics. Normal deterministic PyTorch initialization
under seed 2026 is retained. This remains a serialization-position-aware,
not order-independent, graph decoder and tests whether explicit directed order
improves repeated-instance alignment in this controlled dataset.

The corrected identity is
`B0-GRAPH-NATIVE-EDGE-DECODER-V1-POSITION-BIAS-C1`. Correction metadata records
index/limit 1/1, parent commit
`089b9f3d0e5a61fb19ef3fa05e993fc4eceffdcb`, parent pilot job `3341942`, and
hypothesis
`explicit-directed-ordered-position-bias-for-repeated-instance-alignment`.
Initial Graph V1 checkpoints are intentionally incompatible. Job `3341942`
artifacts are evidence only and are never loaded or resumed.

## Pre-pilot node-sequence audit

Authoritative job `3339188` exposed a teacher-forced container defect before
the first scientific pilot: independently selected current nodes, each legal
against a shifted target prefix, did not necessarily concatenate into one
complete controlled program. Graph V1 now has one explicit production field,
`authoritative_graph_node_type_ids`. It is a complete grammar-constrained
rollout selected from raw node logits using only authorized sequence lengths.
It supplies graph type features, legal edge masks, graph records, V6 contract
validation, strict conversion, and metrics. `graph_raw_node_type_argmax_ids`
remains shadow evidence and never enters conversion.

Teacher-forced graph prediction uses decoded teacher-forced neural states but
rolls node types forward through the V5 grammar without reading target node
IDs. Autonomous prediction continues to use the exact constrained V6 rollout
stored in its generated node records. Both paths therefore have one internally
consistent sequence; target edges remain confined to loss and scoring.

Sequence-like evidence has the following fixed roles:

| Field | Meaning | May enter conversion |
| --- | --- | --- |
| `node_type_logits.argmax` | unconstrained raw class evidence | no |
| `graph_raw_node_type_argmax_ids` | aligned raw graph-node shadow | no |
| `authoritative_graph_node_type_ids` | complete teacher graph rollout | yes |
| `teacher_forced_prefix_node_ids` | inherited shifted training prefixes | no |
| `grammar_constrained_node_type_ids` in the graph prediction | serialized copy of the authoritative rollout | yes |
| autonomous `raw_nodes` | exact constrained V6 rollout records | yes |

The production-shaped readiness path covers authorization, ordinary train/IID
loading, graph tensorization, model and node construction, teacher-forced and
autonomous graph prediction, strict conversion, metrics, checkpoint save and
reload, finite JSON, and terminal acceptance construction. The first
engineering attempt failed before validation and checkpoint completion; the
later initial scientific pilot `3341942` completed and supplied the evidence
for the now-consumed C1 correction.

## Pilot source provenance

Job `3339787` was an engineering failure, not a scientific result. It completed
68 epoch-1 training steps and processed 544 examples, but did not complete
epoch validation or produce a successful checkpoint. Its state must not be
resumed or reused; the corrected pilot starts from fresh deterministic
initialization in a new immutable output directory.

Graph V1 source authorization now collects branch, commit, porcelain-v1 status,
and the deterministic source digest from an explicit repository root. Branch
identity uses `git symbolic-ref --quiet --short HEAD`; cleanliness uses
`git status --porcelain=v1 --untracked-files=all`. The reviewed commit and
expected `graph-profile-decoder` branch are checked before training. Immediately
before every checkpoint and strict selected/final reload, the repository is
recollected and required to be clean and exactly equivalent to the authorized
branch, commit, and digest. The checkpoint stores that verified-equivalent
record. Detached HEAD, command failure, field/type errors, dirtiness, and
identity mismatch are terminal structured provenance errors.
