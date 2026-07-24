# B0-FLAT-MIXED-VQ baseline core

`prototype.flat_baseline` is the first neural-model plumbing milestone for the
structured extrude-and-revolve experiments. It implements one flat,
mixed discrete/continuous encoder, one mixed EMA vector-quantization
bottleneck, a teacher-forced decoder, and reconstruction losses. It contains
no training CLI, graph encoder, factored stream, hybrid latent, code prior, or
counterfactual training logic.

The implementation consumes `FlatBatch.to_torch()` and the accompanying
`ReconstructionBatch.to_torch()` directly. It does not define another data or
target format.

## Inputs and encoder

For batch size `B` and padded node count `N`:

- `categorical_ids`: `torch.long [B, N, 10]`
- `geometry`: `torch.float32 [B, N, 39]`
- `geometry_mask`: `torch.bool [B, N, 39]`
- `padding_mask`: `torch.bool [B, N]`, where **true means a real node**

The ten categorical columns use the frozen `prototype.model_data`
vocabularies in this order: node type, operation type, Boolean mode,
direction, reference plane, four primitive slots, and loop role. Each field
has its own embedding table. Masked geometry values and the geometry
applicability mask are projected separately. Their projections, the ten
embeddings, and a learned chronological node-position embedding are summed
and layer-normalized into one mixed node stream.

Learned latent query tokens are prepended to that stream. A native,
pre-layer-normalized `torch.nn.TransformerEncoder` summarizes the node
sequence. Dependency edges and edge types are deliberately absent from this
encoder.

## Mixed EMA VQ bottleneck

The latent-query states are projected into one codebook and assigned by
nearest squared Euclidean distance. Assignment uses integer indices,
`torch.bincount`, and `index_add_`; it does not materialize a dense one-hot
matrix. Code vectors, cluster sizes, and EMA sums are registered buffers.
They are updated in place under `torch.no_grad()` only in training mode;
`nn.Parameter` objects are never replaced during a forward pass.

The straight-through quantized memory has shape `[B, L, D]`, where `L` is
`latent_tokens` and `D` is `model_dim`. The model also returns code indices
`torch.long [B, L]`, commitment loss, assignment counts, active-code count,
utilization, and perplexity. Evaluation mode does not update the codebook.

## Decoder and outputs

The decoder receives a learned BOS vector followed by shifted ground-truth
node records. A causal, pre-layer-normalized
`torch.nn.TransformerDecoder` cross-attends to the quantized memory. It
returns:

- decoded states: `float [B, N, D]`
- node-type logits: `float [B, N, V_node]`
- nine categorical-logit tensors: `float [B, N, V_field]`
- normalized geometry: `float [B, N, 39]`, bounded to `[-1, 1]` by `tanh`
- ordered-pair edge-presence logits: `float [B, N, N]`
- ordered-pair edge-type logits: `float [B, N, N, V_edge]`
- operation pointer logits: `float [B, O_max, N]`

Pairwise heads operate only on decoded states. Operation queries point to
canonical node positions; padded node positions receive the minimum finite
logit. No predicted or target edge is fed back into the flat encoder.

## Loss contract

`flat_mixed_vq_loss(output, target, config)` consumes the shared collated
reconstruction target and returns a `FlatMixedVQLoss` with the weighted total
and these independently normalized components:

1. node-type cross entropy over real nodes;
2. the arithmetic mean of nine separately evaluated categorical
   cross-entropies over real nodes;
3. Smooth L1 over applicable geometry channels on real nodes;
4. balanced edge-presence BCE: the mean positive-edge loss and mean
   non-edge loss are averaged, so the number of non-edges cannot dominate;
5. edge-type cross entropy on true edges only;
6. operation-pointer cross entropy on valid operation slots only;
7. VQ commitment loss.

Self-pairs and pairs containing padded nodes are excluded from the edge
losses. Target edges are converted from the globally offset,
concatenated `ReconstructionBatch` representation into per-example ordered
pair matrices. Every loss weight is explicit in `FlatBaselineConfig`.

## Configuration and compatibility

`FlatBaselineConfig` is a frozen dataclass. Every architecture dimension,
dropout/EMA value, and loss weight is explicit, validated, and serializable
through deterministic `to_dict()` and `to_json()` methods. Its defaults are
small plumbing defaults, not research-scale hyperparameters.

The package parses with Python 3.8 grammar and targets native PyTorch 1.11.
It uses ordinary PyTorch tensors, works on the input tensors' CPU or CUDA
device, contains no hard-coded `.cuda()` call, and has no PyTorch Geometric
dependency. PyTorch is imported lazily at the package boundary, so the
configuration and source-compatibility checks remain usable in environments
without PyTorch.

Run the focused tests with:

```bash
python3 -m unittest discover -s prototype/flat_baseline/tests -v
```

The tensor forward, backward, masking, VQ, and tiny-optimization tests require
real PyTorch. They are skipped with an explicit reason when PyTorch is absent;
the intended compatibility environment is Python 3.8 with PyTorch 1.11 on
Adroit.
