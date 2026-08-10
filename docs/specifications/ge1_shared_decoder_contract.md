# GE1 C5 Shared Decoder and Output-Position Contract

| Field | Frozen value |
|---|---|
| Status | Implemented and authoritatively validated in C5 at exact commit `996016df44b7f9a6cd5c092a3e3b7a87d9964f9d` |
| Scope | Stage 3 shared-decoder extraction and parity proof only |
| Model identity | `GE1-MODEL-v1` |
| Decoder identity | `GE1-SHARED-TYPED-EDGE-DECODER-V1` |
| Checkpoint identity | `GE1-CHECKPOINT-v1` |
| Output-position identity | `GE1-DECODER-OUTPUT-POSITIONS-v1` |
| Common loss identity | `GE1-COMMON-GRAPH-LOSS-v1` |
| Prospective operation-magnitude identity | `GE1-OPERATION-MAGNITUDE-POSITIVE-v1` |
| Historical operation-magnitude identity | `GE1-OPERATION-MAGNITUDE-TANH-LEGACY-v1` |
| Prospective repair validation | Passed on Adroit CPU as job `3345044` at exact commit `12167ce7d0dc025c4b297b7ccb8a3011580bf0fc` |
| Bottleneck | Continuous; decoder input is `memory [B, 2, 32]` |

This additive record discharges frozen Stage 0 item 4 without rewriting the
historical preregistration. It records the decoder and output-side position
signals before any GE1 comparative result is inspected.

## Parity authorities and boundary

The shared decoder has two retained authorities:

- constrained V6 is authoritative for causal node rollout, prefix grammar,
  node masks, five categorical heads, profile-family and compact-profile
  prediction, canonical profile reconstruction, reference-plane construction,
  remaining normalized geometry, and revolve-axis construction;
- the initial Graph V1 main ordered-pair MLP is authoritative for typed-edge
  features, class logits, pair masking, graph records, and strict graph-to-CAD
  conversion.

The Graph V1 C1 additive directed-position-bias branch is excluded. C5 uses
the initial main pair logits directly as its six-class typed-edge logits. It
does not instantiate `source_position_factor`, `destination_position_factor`,
or `position_class_projection`.

Parity is established with fixed memory, masks, state, dtype, device,
evaluation mode, and output positions. The constrained V6 node/geometry
intermediates and initial Graph V1 main-pair logits must be exactly equal, not
approximately equal. Constrained graph records and authoritative conversion
results must then be equal for the same constrained graph. A parity failure
reports the first tensor name, stage, shape, dtype, maximum absolute
difference, and mean absolute difference.

Complete GE1 flat-model equality to original Flat V6 is not part of this
boundary. The current flat encoder has a frozen width-192 feed-forward layer,
a different construction sequence, and a continuous bottleneck; original Flat
V6 used width 64 and discrete VQ. Encoder differences must not be described as
decoder-parity failures.

## Shared implementation and independent instances

Both arms instantiate `SharedGE1Decoder` and call the same `forward`
implementation. A canonical decoder is constructed once from an
arm-independent local random stream, then deep-copied into the separately
trained arm models. The copies have:

- identical parameter and buffer names, shapes, dtypes, and initial values;
- distinct `Parameter` and buffer object identities;
- independent mutable state; and
- a decoder namespace independent of arm construction order.

The complete arm models do not share live encoder or decoder parameters. The
thin wrapper chooses and calls the encoder before the common-memory boundary.
Only `EncodedMemory.memory`, never diagnostic `prequant`, is passed to the
decoder. No code after that boundary receives or branches on encoder identity.

## Prediction levels

Autonomous `SharedGE1Decoder.forward` accepts only continuous memory, requested
node counts, and a nonempty node-count provenance label. It does not accept a
target, operation sequence, target nodes, geometry, edges, reference pointers,
family/template/split identity, or an encoder-arm label.

It returns one immutable `SharedDecoderPrediction` with three distinct tuples:

1. `raw_prediction` contains direct causal-prefix decoder tensors and the
   unmasked main pairwise typed-edge logits;
2. `constrained_prediction` contains grammar-constrained nodes, geometry,
   masked typed relationships, and a validated canonical graph record;
3. `converted_prediction` contains the authoritative Graph V1 conversion
   result or an explicit structured raised-failure record.

Building a later level does not overwrite either earlier level. Autonomous
prefix feedback uses only predicted constrained records. The compatibility
placeholder needed by the retained V6 helper is removed from published
prediction provenance: continuous GE1 predictions expose an empty latent-index
tuple and `ge1_continuous_memory_from_codebook_projection` as their memory
source.

## Frozen output-side position inventory

These are semantic serialized-output positions. Both arms receive them
identically after the common memory boundary, and none enters either encoder.

| Exact signal | Source and shape | Construction and decoder entry |
|---|---|---|
| `decoder_output_step_absolute_embedding` | `decoder_position_embedding(arange(output_length))`, `[1,L,32]`, floating point | Learned absolute output-step embedding; added to BOS/shifted prefix content before `decoder_input_norm` and the causal Transformer decoder |
| `pair_source_absolute_embedding` | same embedding table at each source index, `[B,N,N,32]`, floating point | Learned absolute source serialization position; concatenated into the main pair-MLP input |
| `pair_destination_absolute_embedding` | same table at each destination index, `[B,N,N,32]`, floating point | Learned absolute destination serialization position; concatenated into the main pair-MLP input |
| `pair_signed_relative_serialized_position` | `(source-destination)/(max_nodes-1)`, `[B,N,N,1]`, floating point | Constructed signed relative position; concatenated into the main pair-MLP input |

The main pair feature order remains exactly:

```text
source decoded state
destination decoded state
source constrained-node-type embedding
destination constrained-node-type embedding
source absolute output-position embedding
destination absolute output-position embedding
global mean of common continuous memory
signed relative serialized output position
```

The first six blocks have width 32, global memory has width 32, and the final
relative value has width one, for the retained `7 * 32 + 1` input width. The
MLP remains `Linear(225,14) -> GELU -> Linear(14,6)`.

No new neural position signal is introduced. In particular, the excluded C1
rank-six source/destination factor branch is not part of C5.

## Bookkeeping versus semantic position

Local pair indices address output rows and form ordered source/destination
pairs. They also select or derive the four frozen semantic decoder-position
signals above; this is their only permitted neural influence. The dense output
node mask controls requested-length grammar activity, padding exclusion, and
active non-self pair masking, but is not embedded or projected.

`graph_offsets` remains an encoder-side bookkeeping tensor used only for
graph-boundary validation, graph-local slicing, and pooling. `edge_offsets`,
`node_graph_ids`, and dense batching masks remain C3 validation and alignment
metadata and do not cross the decoder interface. No bookkeeping value is
declared harmless merely because it is stored separately: signatures, source
routing, and tests enforce these uses.

Chronological or absolute node-position embeddings remain prohibited in the
typed graph encoder. Output serialization positions are allowed because they
belong to the one decoder shared after both representations have already been
encoded.

## Common loss

Both arms call `common_ge1_loss`, which delegates to the retained Graph V1
per-example loss assembly. That authority already normalizes each active
component within each example and then averages the example totals across the
batch. C5 therefore has no intentional legacy aggregation difference.

Separately observable components are node type, categorical attributes,
reference-plane category, remaining geometry, profile family, profile
parameters, typed graph edge, none class, positive edge/edge type, and VQ
commitment. Component masks, ignore behavior, weights, and stabilizers are
unchanged. The continuous bottleneck supplies an exactly zero per-example VQ
term.

## Prospective operation-magnitude parameterization

Accepted
[ADR-0009](../decisions/ADR-0009-ge1-positive-operation-magnitude-repair.md)
adds a versioned prospective neural-output mode without changing the frozen C5
parity result. Historical C7-v2 behavior remains explicit as
`GE1-OPERATION-MAGNITUDE-TANH-LEGACY-v1`, under which all six compact
remaining-geometry channels use exact `tanh(raw)`.

The prospective `GE1-OPERATION-MAGNITUDE-POSITIVE-v1` mode changes only
compact channels 4 and 5, which scatter to serialized extrusion-distance
channel 37 and revolve-angle channel 38. For those two channels it applies:

```text
torch.finfo(raw.dtype).tiny
    + (1 - torch.finfo(raw.dtype).tiny) * sigmoid(raw)
```

The result is finite and in `(0, 1]` for finite logits, including sigmoid
underflow at a very negative logit. Exact `1.0` remains representable and is
required by the controlled 360-degree revolve target. Compact channels 0--3
retain exact `tanh`; masks, channel selection, direction, Boolean mode,
normalization, denormalization, loss assembly, and weights are unchanged. The
mapping runs inside the neural decoder before both teacher-forced loss and
autonomous reconstruction. The raw head pre-activation remains separately
callable for diagnostics.

The mode is a required model-configuration and checkpoint-provenance field.
Strict inference and recovery checkpoint loading rejects a mode mismatch with
the existing stable invalid-checkpoint error. Both encoder arms receive
disjoint, value-identical copies of the same selected decoder mode. This
analytic parameter-domain guarantee makes no CAD-kernel validity claim.

Authoritative Adroit CPU job `3345044` exercised both identities under Python
3.8.13 and PyTorch 1.11.0. All 13 focused repair tests and all 292 complete
graph-encoder tests passed with zero skips, including exact legacy parity,
extreme-logit bounds, channel isolation, unchanged categorical/mask outputs,
five synthetic conversion cases, both strict checkpoint boundaries, identity
propagation, and matched arm initialization. The runner opened no corpus,
performed no training, and used no CAD kernel. The
[validation record](../experiments/ge1_operation_magnitude_repair_cpu_validation.md)
is engineering evidence only; it does not authorize or report a repaired
scientific run.

## Strict checkpoint contract

`GE1-CHECKPOINT-v1` records the arm and encoder identities, frozen feed-forward
width, relation-basis count, shared-decoder and output-position versions,
operation-magnitude parameterization, continuous-bottleneck status, full
frozen configuration and its SHA-256,
architectural sizes, parameter and buffer inventories, exact decoder key
namespace, source commit, authoritative operation-template manifest hash, and
complete model state.

Reload reconstructs the frozen selected arm, rejects missing, unexpected,
mislabeled, or incompatible metadata and state keys, and calls
`load_state_dict(..., strict=True)`. There is no `strict=False` fallback.

## Scope boundary

C5 does not implement or run training, comparative development evaluation,
the C7 train-only sufficiency or memory-use gates, any C8 decoder repair, VQ,
editing, or a position-aware graph encoder. It does not authorize or access RR,
ER, IID, history-depth, geometry-extrapolation, a corpus manifest, or a CAD
history payload. Authoritative C5 runtime acceptance required the new
real-PyTorch tests to pass under Python 3.8 and PyTorch 1.11 on CPU; that gate
passed as Adroit job `3344290`.

Initial Adroit job `3344278` at commit
`ffa6091cfecb37aa648ed3590f591f247a776754` ran all 22 focused tests with zero
skips and zero errors; 19 passed and three test-contract assertions failed.
The [attempt record](../experiments/ge1_c5_cpu_validation_attempt_3344278.md)
shows that the corrections concern only legacy VQ provenance normalization,
authoritative conversion-result field names, and discovery of padded positions
after canonical batch sorting. Production decoder and model behavior is
unchanged. The failed attempt is not C5 runtime acceptance.

Corrected exact commit `996016df44b7f9a6cd5c092a3e3b7a87d9964f9d`
subsequently passed all 22 focused C5 tests and all 110 complete graph-encoder
tests with zero skips under Python 3.8.13/PyTorch 1.11.0 on Adroit CPU. Every
regression and repository gate also passed. The [authoritative validation
record](../experiments/ge1_c5_cpu_validation.md) preserves the evidence and
limitations.
