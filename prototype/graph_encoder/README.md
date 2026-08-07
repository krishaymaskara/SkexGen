# GE1 Graph Encoder Package

## Implemented scope

This package implements **C1 only** for
`GE1-SHARED-DECODER-ENCODER-COMPARISON`:

- immutable, deterministic model and training configuration;
- exact frozen identities and encoder-to-arm derivation;
- operation-template-only manifest authority verification;
- train and complete-development payload access wrappers;
- unconditional protection of RR, ER, and every non-operation manifest; and
- stable GE1 boundary errors.

It does not implement an encoder, neural model, shared decoder, checkpoint,
training loop, evaluation, or scientific result. The frozen
`prototype.flat_baseline` and `prototype.graph_baseline` packages are reused by
import only and remain unchanged.

## Public API

```python
from prototype.graph_encoder import (
    GE1Config,
    GE1TrainingConfig,
    GraphEncoderError,
    load_development,
    load_train,
)
```

`load_train(corpus_dir)` loads the complete authoritative train assignment.
Its optional `family_ids` argument accepts only a sorted, unique, balanced
four-family or 32-family subset: respectively one or eight train families from
each of E, R, EE, and RE. `load_development(corpus_dir)` always loads the
complete 45-family development assignment. Neither function accepts a split
or partition argument.

The package deliberately does not re-export unrestricted model-data loaders.
There is no C1 API for RR, ER, IID, history-depth, geometry-extrapolation, or
counterfactual payload access.

## Frozen identities

| Role | Literal |
|---|---|
| Model family | `GE1-MODEL-v1` |
| Flat arm | `GE1-FLAT-CHRONOLOGICAL-CONTINUOUS-V1` |
| Typed-graph arm | `GE1-TYPED-GRAPH-POSITION-FREE-CONTINUOUS-V1` |
| Shared decoder | `GE1-SHARED-TYPED-EDGE-DECODER-V1` |
| Checkpoint schema | `GE1-CHECKPOINT-v1` |
| Protocol | `GE1-STAGE0-PREREG-v1` |
| Experiment | `GE1-SHARED-DECODER-ENCODER-COMPARISON` |

`GE1Config.encoder` accepts only `flat` or `typed_graph` and derives the arm
identity. Only the continuous bottleneck is accepted. Numeric architecture
defaults come from the inherited flat/Graph V1 configuration contracts or the
accepted GE1 plan. Relation-basis count and encoder feed-forward width are the
only capacity fields that may later be adjusted.

`GE1TrainingConfig` fixes the 50-epoch, batch-eight AdamW policy, epoch-50
checkpoint selection, plateau definition, train-ceiling tolerances, planned
three seeds, and the exact preauthorized two-seed timing fallback.

## Manifest authority and access order

Before calling any physical-example payload loader, C1 reads only:

```text
manifests/operation_template.json
```

It requires the exact frozen file SHA-256
`a9ac86a6dede054fbbba57e0906b210bab26036c3f5c150b332038f78d2dadb7`,
the 407/45/114/114 family counts, the 814/90/228/228 sample counts, permitted
templates in every partition, and all four assignment hashes. Missing,
unreadable, modified, or inconsistent authority fails before the inherited
payload loader is invoked. Verification cannot be disabled when the Adroit
path is unavailable.

After verification, C1 delegates only the authorized partition and optional
train subset to `prototype.model_data.loader.load_partition_physical_examples`.
Malformed or inconsistent corpus data continues to raise the original
`ModelDataError`; it is not hidden inside a GE1 error.

## Error taxonomy

`GraphEncoderError` always exposes stable `code` and `detail` fields. C1 uses:

- `invalid_configuration` for malformed values;
- `identity_mismatch` for altered frozen schema identities;
- `unauthorized_configuration` for scientifically unauthorized settings;
- `invalid_train_subset` for unsafe sufficiency selections;
- `manifest_authority_failure` for absent or inconsistent authority; and
- `protected_partition_access` for any non-train/development request.

## Explicit limitations

- C1 provides configuration and access control, not graph canonicalization or
  paired flat/graph adapters.
- No model or decoder class exists in this package yet.
- No checkpoint or training artifact exists.
- RR remains closed until a separately audited Stage 7 capability is added.
- ER and all IID, history-depth, and geometry-extrapolation partitions remain
  closed throughout core GE1.
- Actual Python 3.8 execution on Adroit remains the authoritative compatibility
  check; local grammar parsing and compilation are preliminary checks only.
