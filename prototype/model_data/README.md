# Model-ready CAD data contract

`prototype.model_data` is the deterministic boundary between the version 1
CAD corpora and future baseline models. It contains no neural network,
training, loss, or experiment code.

## Learning unit and loading

`load_physical_examples(corpus_dir, split_name)` reads the existing corpus and
split manifests. Exactly one continuous and one quantized sample must exist
for every `source_family_id`. Both histories are strictly deserialized,
canonicalized, and converted to the authoritative decoded physical
descriptor from `prototype.controlled_data.identity`. A disagreement is an
error. The two encodings then collapse to one immutable `PhysicalExample`;
they are never two training examples.

Collapse is permitted only after checking canonical JSON, schema validation,
decoded physical bytes, authoritative source/sample IDs, declared encoding,
and agreement of all physical manifest metadata. There must be exactly two
records and exactly the set `{continuous, quantized}`. Neither encoding,
manifest traversal order, sample path, generation seed, nor selection order
chooses the target: the verified continuous form is the fixed canonical
reconstruction convention.

`physical_family_id` is the existing `source_family_id`. Partition authority
also remains at that family ID. Sample-level partition fields are checked only
as inherited assertions. Canonical records contain no machine path,
timestamp, host, or environment metadata.

Each physical example contains family/sample metadata, split and partition,
the ordered operation IDs, canonical typed nodes and edges, normalized
geometry and applicability masks, and a common reconstruction target.
Reference planes precede chronological feature groups. Each feature group is
ordered sketch, profile, optional revolve axis, then operation. Edges are
sorted by canonical node position, edge vocabulary ID, and target position.
Remaining nodes, if any, are sorted by ID. Families and batches are sorted by
physical-family ID. Primitive slots are sorted by stable primitive ID. These
rules do not use dictionary insertion order, file-system traversal, hashes,
or manifest ordering.

The retained canonical reconstruction JSON is the continuous serialization
of the already-verified physical history. It is a target artifact, not an
additional input example.

## Fixed vocabularies and geometry

Every vocabulary has `<pad>` ID 0 and `<none>` ID 1. All remaining enum values
are lexicographically sorted at module construction, so IDs cannot depend on
corpus traversal. Separate vocabularies cover node, edge, operation,
primitive, Boolean mode, direction, reference plane, and loop role.

Every node has these exact 39 continuous channels and a parallel Boolean
applicability mask:

| Index | Channel |
| ---: | --- |
| 0–2 | `plane_origin_x`, `plane_origin_y`, `plane_origin_z` |
| 3–5 | `plane_x_axis_x`, `plane_x_axis_y`, `plane_x_axis_z` |
| 6–8 | `plane_y_axis_x`, `plane_y_axis_y`, `plane_y_axis_z` |
| 9–14 | `primitive_0_x0`, `primitive_0_y0`, `primitive_0_x1`, `primitive_0_y1`, `primitive_0_x2`, `primitive_0_y2` |
| 15–20 | `primitive_1_x0`, `primitive_1_y0`, `primitive_1_x1`, `primitive_1_y1`, `primitive_1_x2`, `primitive_1_y2` |
| 21–26 | `primitive_2_x0`, `primitive_2_y0`, `primitive_2_x1`, `primitive_2_y1`, `primitive_2_x2`, `primitive_2_y2` |
| 27–32 | `primitive_3_x0`, `primitive_3_y0`, `primitive_3_x1`, `primitive_3_y1`, `primitive_3_x2`, `primitive_3_y2` |
| 33–34 | `axis_point_x`, `axis_point_y` |
| 35–36 | `axis_direction_x`, `axis_direction_y` |
| 37 | `extrude_distance` |
| 38 | `revolve_angle` |

Line slots use `(start_x, start_y, end_x, end_y)`; arc slots use start,
midpoint, and end; circle slots use `(center_x, center_y, radius)`. Unused
values are zero and masked false. Lengths are divided by the fixed version 1
scale 4.0, angles by 360.0, and unit direction/frame components are unchanged.
All geometry is decoded before normalization. The current controlled grid has
sketch extents at most 3.0, extrusion distances at most 3.0, angles at most
360 degrees, and constructed coordinates with absolute value at most 3.5, so
every applicable normalized value lies in `[-1, 1]`.

The accepted contract is inclusive `[-1, 1]`. A finite value outside it is a
structured `geometry_out_of_range` error; values are never clipped. Negative
coordinates and unit-direction components are retained. Operation direction
is a separate categorical field. Numeric zero is legitimate and is
distinguished from non-applicability by a true mask entry. Nonfinite values
are rejected.

## Input adapters

For a history with `N` canonical nodes, `E` edges, and `O` operations:

- `flat_mixed.categorical_ids`: `[N, 10]`, comprising node type plus nine
  node attributes.
- `flat_mixed.geometry`, `geometry_mask`: `[N, 39]`.
- `typed_graph.node_type_ids`: `[N]`.
- `typed_graph.edge_index`: `[2, E]`; `edge_type_ids`: `[E]`.
- `typed_graph.categorical_attributes`: `[N, 9]`.
- `typed_graph.geometry`, `geometry_mask`: `[N, 39]`.
- `typed_graph.operation_sequence`: `[O]`, containing canonical node indices.

The 10 flat categorical columns, in order, are:

1. `node_type`
2. `operation_type`
3. `boolean_mode`
4. `direction`
5. `reference_plane`
6. `primitive_0`
7. `primitive_1`
8. `primitive_2`
9. `primitive_3`
10. `loop_role`

The nine graph categorical-attribute columns are flat columns 2–10 in exactly
that order. Graph node type is held separately in `node_type_ids`.

The flat adapter deliberately exposes no dependency edges. Both future graph
architectures consume the same `typed_graph` record; latent discreteness is
not a dataset property.

The shared reconstruction target has node type `[N]`, categorical attributes
`[N, 9]`, Boolean targets `[N]`, dependencies `[2, E]` plus edge types `[E]`,
operation sequence `[O]`, geometry `[N, 39]`, and geometry masks `[N, 39]`.

## Batching and tensors

Collation sorts examples by physical-family ID. Flat batches pad to
`categorical_ids [B,N_max,10]`, geometry and geometry mask
`[B,N_max,39]`, and `padding_mask [B,N_max]`. Graph batches concatenate nodes
and edges into node types `[sum(N)]`, attributes `[sum(N),9]`, geometry and
geometry mask `[sum(N),39]`, edge index `[2,sum(E)]`, and edge types
`[sum(E)]`. They provide
`graph_offsets [B+1]`, `edge_offsets [B+1]`, `node_graph_ids [sum(N)]`, and
`node_mask [B, N_max]`. Reconstruction batches pad node and operation targets,
use `-1` only for masked operation indices, and concatenate target edges with
offsets. Input edge indices are offset exactly once. Empty edge sets are
supported; an empty batch is a structured `empty_batch` error.

Core arrays are immutable standard-library tuples so corpus inspection does
not require PyTorch. Calling `to_torch()` lazily imports PyTorch and returns
contiguous ordinary `torch.Tensor` objects. All categorical IDs, edge/node
indices, offsets, graph IDs, and operation indices use `torch.long`; geometry
uses `torch.float32`; masks use `torch.bool`. The implementation uses
`torch.tensor(..., dtype=...).contiguous()`, which is compatible with PyTorch
1.11. No PyTorch Geometric or newer tensor API is used. Local tests can inject
a tensor-compatible adapter when PyTorch is unavailable.

Real tensor construction was validated on Adroit in CPU-only Slurm job
`3322011` under Python 3.8.13 and PyTorch 1.11.0. The run executed 193 tests,
performed deterministic individual and batched tensor conversion, checked
dtypes, shapes, contiguity, graph offsets, empty-edge behavior, reconstruction
targets, compilation, and reported `193/193` passing. The validated
model-data source was commit `91d733d`; the exact staged validation scripts
used by the job were subsequently committed as `038e497`. The dirty-tree
provenance, results, limitations, and frozen archive checksum are preserved in
the [real-PyTorch validation
record](../../docs/experiments/model_data_pytorch_validation.md).

## Counterfactual evaluation

`load_counterfactual_examples` validates both encoding-specific pair records,
recomputes the authoritative edit-family and edit-sample IDs, links their
physical source and target examples, and checks endpoint/sample membership.
Edit-family partition assignment is authoritative and both endpoints must
inherit it. The result preserves allowed-changed and expected-unchanged
attribute paths. The upstream authoritative locality validator checks that
these paths are valid, disjoint, duplicate-free, exhaustive, and exactly
equal to the actual decoded source/target difference. Missing, malformed, or
stale paths are errors. `evaluation_only` is fixed to true and is not a
constructor argument, preventing accidental opt-out.

Corpus manifests, split manifests, family/sample counts, sample references,
safe relative paths, and authoritative assignments are checked before use.
Every encoding variant must be assigned exactly once. Referenced sample files
must close the corpus file set; unreferenced/orphan files and unknown
manifest/sample fields are rejected. Canonical model records include no
timestamps, absolute paths, hostnames, usernames, environment versions,
generation seeds, or selection order.

All violations raise `ModelDataError` with a stable `code`, concise `detail`,
and optional family ID. The package and tests parse under Python 3.8 grammar;
postponed annotations permit modern annotation spelling without changing
runtime behavior.
