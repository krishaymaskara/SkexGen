# Controlled Synthetic CAD Data

This CPU-only, standard-library prototype generates small deterministic corpora
using `prototype.representation`. It does not duplicate that package's schema,
validator, or canonical JSON serializer, and it does not execute a CAD kernel.

## Scope

Version 1 supports the operation templates `E`, `R`, `EE`, `ER`, `RE`, and
`RR`. Each operation has a fresh sketch and profile; each revolve's axis is
defined in the same sketch as its profile; and every later operation depends
on the immediately preceding operation. The first operation uses `NEW_BODY`,
and a second operation uses `JOIN` or `CUT`.

Generated histories satisfy two distinct contracts. The representation
validator establishes schema and serialization validity. Before construction,
the generator also applies the versioned pure-Python analytical feasibility
policy to exclude Boolean assignments known to be disconnected, ineffective,
or destructive of the complete body. This executable-corpus contract is an
analytical promise about the frozen builders, not a claim that a kernel has
run: every generated record still has `kernel_status: "not_checked"` until an
independent OpenCascade audit evaluates it.

## Identities and immutable corpus

`source_family_id` hashes canonical structure and decoded physical geometry.
It excludes encoding, seed, selection order, split, serialization format, and
path. Continuous and quantized variants therefore share one source family.
Physical identity normalizes negative zero and rounds decoded values to twelve
decimal places, below the representation validator's geometric tolerances, to
remove harmless affine-decoding noise. Accepted default length grids are
separated by at least `0.125` and angle grids by at least `1` degree, so this
normalization is eleven or more orders of magnitude finer than distinct
configured values. Configuration validation rejects grids whose values would
collide after normalization or cannot be represented by the frozen quantizers.

`sample_id` hashes the complete canonical representation variant, including
its continuous or quantized numeric records. A request for `N` source families
always creates exactly `2N` sample files.

Sample JSON is generated once under `samples/`. Independent source-family
assignments are stored under:

```text
manifests/iid.json
manifests/operation_template.json
manifests/history_depth.json
manifests/geometry_extrapolation.json
```

Every variant inherits its partition from the manifest's one authoritative
assignment for its `source_family_id`.

## Geometry and extrapolation

The sketch statistic is maximum two-dimensional bounding-box extent:

- rectangle: width `E`, height `E / 2`;
- circle: diameter `E`;
- capsule: total width `E`, total height `E / 2`.

For a history, the statistic is the maximum `E` over all sketches. The frozen
in-range band is `E <= 2.0`; the extrapolation band is `E >= 2.5`. Boundaries
come from configuration, never from generated observations.

The `ER` benchmark is a held-out operation-template split, not broad
dependency-composition generalization. Its validation set is drawn from the
same `E`, `R`, `EE`, and `RE` template pool as training. `RR` is a separate
secondary systematic-validation partition.

The depth-one versus depth-two split is a toy, provisional version 1 benchmark.
It does not replace the later longer-history experiment in the research plan.

## Analytical Boolean feasibility

`JOIN` and `CUT` use different geometric contracts. A `JOIN` may be valid
without volumetric overlap when the two features share a positive-area face;
it must remain one connected solid and add volume. A `CUT` must have
positive-volume overlap and leave a positive-volume residual. The policy uses
exact extrusion intervals and revolve sectors for `EE` and `RR`. For `ER` and
`RE`, it combines exact orientation/sector tests with primitive-specific
radial containment certificates and explicit non-containment witnesses.

The classifier records structured reasons including duplicate or contained
joins, boundary-only cuts, complete subtraction, and disconnected joins.
Mixed cases that overlap but cannot be safely proved contained or
non-contained are classified as `mixed_containment_not_certified` and are not
generated. The status remains explicit rather than being folded into a generic
rejection. No OpenCascade result affects selection, IDs, or corpus ordering;
OpenCascade remains an independent post-generation check for implementation
mistakes and kernel robustness.

## Bounded deterministic selection

The complete default raw grid contains 180,900 physical assignments. The
analytical policy accepts 120,060 source families, corresponding to 240,120
continuous/quantized variants. Generation enumerates raw mixed-radix indices,
filters before constructing `CADHistory` objects or identities, and uses a
deterministic affine permutation to fill a bounded request. It never
materializes the Cartesian collection of complete histories.

Accepted coverage anchors explicitly cover primitive family, reference plane,
direction values, numerical parameter grids, and—within every depth-two
template/band block—all six combinations of `JOIN`/`CUT` with smaller, equal,
and larger second-sketch extents. Same/opposite direction relations remain
represented wherever feasible.

With the default grids, full coverage requires 68 source families. This is
derived as five anchors for each of the four depth-one template/band blocks
and six relational anchors for each of the eight depth-two template/band
blocks: `4 * 5 + 8 * 6 = 68`. Custom grids recalculate the marginal maximum,
while the six declared Boolean/extent tokens remain mandatory. Smaller
requests are rejected rather than silently weakening held-out-factor
coverage.

## CLI

From the repository root:

```bash
python3 -m prototype.controlled_data.generate \
  --output-dir /tmp/skexgen-controlled \
  --seed 0 \
  --num-source-families 100
```

Generation occurs in a temporary sibling directory. Only after samples,
identities, manifests, split constraints, round trips, and written files pass
verification is that directory atomically renamed to the requested path.

## Related evidence

The [B0 680-family pilot-corpus
record](../../docs/experiments/b0_pilot_corpus_680.md) documents one completed
bounded generation run without duplicating its evidence here.
