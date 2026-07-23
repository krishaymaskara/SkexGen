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

Generated histories are schema-valid and serialization-round-trip-valid.
Their `kernel_status` is `not_checked`: in particular, a `JOIN` or `CUT`
record must not be described as an executable solid until a CAD kernel has
evaluated it.

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

## Bounded deterministic selection

The complete default grid contains 180,900 physical families and 361,800
encoding variants. Generation uses mixed-radix block indexing, five coverage
anchors per `(operation template, extent band)` block, and a deterministic
affine permutation to fill the requested bounded subset. It never constructs
the complete Cartesian collection of `CADHistory` objects.

With the default grids, full split coverage requires 60 source families. This
is calculated as five diagonal anchors in each of the twelve
`(operation template, extent band)` blocks: five is the largest mandatory
factor-domain size, six templates are emitted, and each has two extent bands.
Custom grids recalculate the minimum. Smaller requests are rejected rather
than silently weakening held-out-factor coverage.

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
