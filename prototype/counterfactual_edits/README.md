# Counterfactual CAD Edit Pairs

This standard-library package builds deterministic, endpoint-disjoint edit
pairs on top of `prototype.representation` and
`prototype.controlled_data`. It does not modify the SkexGen model, dataset,
training, sampling, or pretrained-model pipeline.

## Version 1 scope

Version 1 supports one-factor edits to profile extent, extrusion distance,
revolve angle, operation direction, and a later operation's JOIN/CUT mode.
Numeric edits connect adjacent frozen-grid values. Profile-extent endpoints
remain in the same in-range or extrapolation band. Operation-type and
reference-plane edits are deferred because they change graph topology or
world geometry beyond the narrow v1 locality contract.

Both endpoints independently pass the controlled-data analytical feasibility
policy. OpenCascade is never used to select pairs.

## Identities and locality

Each endpoint retains the existing `source_family_id` and `sample_id`.
`edit_family_id` is directed and encoding-independent; `edit_sample_id`
additionally commits to one continuous or quantized endpoint pair. Seed,
selection order, paths, splits, and kernel results do not enter identities.

Every pair records typed changed and unchanged representation addresses,
unchanged edge triples, unchanged operation order, and causally downstream
operations. Structural graph-edit distance is zero and semantic parameter
edit distance is one. Downstream executed geometry may legitimately change
even when its symbolic parameters remain unchanged.

## Candidate and coverage contract

The default factor grid contains 804,510 raw undirected edit candidates and
448,920 candidates whose endpoints are both analytically feasible. Selection
uses 68 primary tokens: 34 edit-type/template/position cells in each of two
extent bands. The deterministic greedy matching also covers 100 declared
secondary tokens and reserves exactly 136 distinct physical endpoints.

Coverage anchors use exact balanced orientation quotas. Fillers use the
versioned SHA-256 orientation policy. No endpoint may occur in more than one
selected edit family.

## Dataset

Generation publishes atomically:

```text
counterfactual-corpus/
  counterfactual_manifest.json
  corpus_manifest.json
  histories/<sample_id>.json
  pairs/<edit_sample_id>.json
  manifests/
    iid.json
    operation_template.json
    history_depth.json
    geometry_extrapolation.json
    endpoint_exclusions.json
```

The endpoint `corpus_manifest.json` is directly consumable by
`prototype.kernel_validation`. Pair files reference endpoint IDs rather than
embedding histories. The endpoint-exclusion manifest lists every physical
validation, test, and secondary-systematic-validation endpoint for later
training-corpus filtering.

```bash
python3 -m prototype.counterfactual_edits.generate \
  --output-dir /tmp/counterfactual-smoke \
  --seed 0 \
  --num-edit-families 68
```

After running endpoint kernel validation, join its report without importing
PythonOCC:

```bash
python3 -m prototype.counterfactual_edits.kernel_integration \
  --counterfactual-corpus /tmp/counterfactual-smoke \
  --endpoint-report /tmp/endpoint-report/execution_report.json \
  --output-dir /tmp/pair-audit
```

The implementation avoids runtime Python 3.9/3.10 constructs and is intended
to run under the existing Python 3.8.13 Adroit container.

## Validation evidence

The [counterfactual OpenCascade audit
record](../../docs/experiments/counterfactual_opencascade_audit.md) documents
the preserved jobs and archive. Under PythonOCC 7.5.1, all 272 encoded
endpoint histories executed successfully, all 136 edit samples succeeded,
and continuous and quantized executions agreed for all 136 physical
endpoints. This establishes bounded benchmark executability, not learned
model edit performance.

## Known implementation limits

Generation currently enumerates the full candidate space once for selection
and again for recorded count metadata. Publication rejects an output directory
that exists when generation starts, but it does not attempt to resolve the
narrow race in which another process creates that directory immediately before
the atomic rename. Directory-fsync crash durability is outside the prototype
scope.
