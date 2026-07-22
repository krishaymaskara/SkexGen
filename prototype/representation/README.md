# Typed CAD Representation Prototype (Schema Version 1)

This directory is a CPU-only, Python-standard-library prototype. It does not
import or modify the SkexGen data, model, training, or checkpoint pipeline.

## Scope

Version 1 represents single-body histories containing reference planes,
sketches, profiles, axes, extrudes, and revolves. The first solid-producing
operation uses `NEW_BODY`; every later operation uses `JOIN` or `CUT`. Later
`NEW_BODY` operations, multiple body identities, and cross-body references are
outside this prototype. This is a deliberate experimental restriction, not a
general rule of CAD systems.

`operation_sequence` is the only execution-order record. A `DEPENDS_ON` edge
has separate semantic meaning: its source operation consumes the result of its
target operation. Edges use the consumer-to-resource convention. Because
version 1 keeps one current solid, each operation after the first depends on
the immediately preceding operation.

Other edge meanings are:

- `PLACED_ON`: sketch to reference plane.
- `DEFINED_IN`: profile or axis to the sketch that declares it.
- `USES_PROFILE`: extrude or revolve to its profile.
- `USES_AXIS`: revolve to its axis.

`DEFINED_IN` describes containment/reference scope without implying that this
prototype executes a feature-creation process. Version 1 requires a revolve's
profile and axis to be defined in the same sketch. That is also a simplifying
prototype restriction rather than a general CAD requirement.

## Structure and geometry

Discrete structure and numerical geometry are separate. Sketch primitives and
loops are subrecords with stable IDs. Their IDs are unique within a sketch,
and a primitive and loop in the same sketch may not share an ID. Geometry is
addressed either by `node_id` or by `(sketch_id, element_id)`, which permits
future element-local edit and preservation measurements.

Loops have no independent numerical payload in version 1: their geometry is
induced by their ordered primitive references. Collection fields use tuples so
the frozen dataclasses do not contain mutable list state.

Profiles explicitly name one outer loop and an ordered list of inner loops.
All referenced loops must be declared by the profile's `DEFINED_IN` sketch.

Geometry may be continuous or affine-quantized. Quantized physical values are
decoded as `code * scale + offset`; every geometric rule is evaluated on the
decoded values. Codes are strict integers, scale is finite and positive, and
offset is finite.

Frozen version 1 conventions:

- extrusion distance is a positive magnitude;
- extrusion direction is a separate discrete attribute;
- extrusions are one-sided;
- revolve angles are degrees in `(0, 360]`;
- all numeric values are finite;
- plane axes are orthonormal within `FRAME_ORTHONORMAL_TOLERANCE = 1e-6`;
- axis directions are unit vectors within `AXIS_DIRECTION_TOLERANCE = 1e-6`;
- loop endpoints meet within `LOOP_CONTINUITY_TOLERANCE = 1e-6`.

Loop validation supports line/arc chains and a single circle. It rejects mixed
circle chains and ambiguous combinations instead of attempting full planar
computational geometry or CAD-kernel execution.

## Deterministic JSON

Canonical JSON includes `schema_version: 1`. Serialization sorts nodes, edges,
primitive declarations, loop declarations, and geometry records by stable IDs;
semantic list order (operation sequence, loop traversal, and ordered inner
loops) is preserved. Parsing checks exact fields, types, and enums and rejects
unknown fields. Serializing a parsed canonical document produces byte-identical
JSON. Histories with the same IDs and semantic ordering serialize identically
regardless of their in-memory tuple insertion order.

Duplicate JSON object fields and non-finite JSON constants are rejected.
Continuous numbers and quantization metadata are emitted as canonical floats,
while quantized codes remain integers.

## Running tests

From the repository root:

```bash
python3 -m unittest discover -s prototype/representation/tests -v
```
