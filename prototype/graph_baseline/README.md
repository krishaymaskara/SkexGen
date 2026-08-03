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
3,496 parameters. Graph V1 replaces them with a shared 225→15→6 GELU pair MLP
with 3,486 parameters. Total capacity differs by ten parameters, below 1%.
Every graph-decoder parameter participates in prediction.

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

The graph experiment will not be repeatedly revised until it produces valid CAD.
After the initial pilot, at most one evidence-backed graph correction is permitted.

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
scientific graph pilot attempt failed before validation and checkpoint
completion, so it produced no scientific result and the one permitted
evidence-backed scientific graph correction remains unused.

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
