# B0-FLAT-MIXED-VQ baseline core

`prototype.flat_baseline` is the first neural-model plumbing milestone for the
structured extrude-and-revolve experiments. It implements one flat,
mixed discrete/continuous encoder, one mixed EMA vector-quantization
bottleneck, a teacher-forced decoder, reconstruction losses, and a
teacher-forced training CLI. It contains no graph encoder, factored stream,
hybrid latent, code prior, or counterfactual training logic.

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

The tensor forward, backward, masking, VQ, training, checkpoint, resume, and
tiny-optimization tests require real PyTorch. They are skipped with an
explicit reason when PyTorch is absent; the intended compatibility
environment is Python 3.8 with PyTorch 1.11 on Adroit.

## Teacher-forced training milestone

The package now includes reproducible single-process training of the existing
model. The learning unit remains one decoded physical family from
`prototype.model_data`; continuous and quantized serialization variants never
become duplicate examples. The only data path is:

```text
load_physical_examples -> adapt_flat_mixed -> collate_flat
```

`TrainingConfig` is a frozen configuration separate from
`FlatBaselineConfig`. Its small defaults are plumbing defaults, not final
research hyperparameters. It records the seed, epoch and batch counts,
optimizer values, clipping threshold, validation/checkpoint intervals,
worker count, device selection, output directory, and validation-loss metric
used for checkpoint selection.

A minimal CPU command is:

```bash
python3 -m prototype.flat_baseline.train \
  --corpus-dir /path/to/controlled-corpus \
  --split-name iid \
  --train-split train \
  --validation-split validation \
  --output-dir /path/to/run \
  --device cpu
```

Model and training configurations can be supplied as JSON objects:

```bash
python3 -m prototype.flat_baseline.train \
  --corpus-dir /path/to/controlled-corpus \
  --split-name operation_template \
  --train-split train \
  --validation-split validation \
  --model-config /path/to/model.json \
  --training-config /path/to/training.json \
  --output-dir /path/to/run
```

Command-line training options override fields loaded from the training JSON.
The official training path accepts exactly the `train` partition for
optimization and exactly the ordinary `validation` partition for checkpoint
selection. Test and secondary systematic-validation partitions are rejected;
they belong to a separate evaluation path.
`--overfit-families N` deterministically caps each requested partition to its
first `N` physical-family IDs without reassigning families or combining
partitions. `--resume-checkpoint /path/to/last.pt` resumes at the next epoch.
The requested epoch count, output directory, and device may change on resume;
all model settings and optimization-relevant training settings must match.

### Artifacts and checkpoint semantics

The output directory contains:

- `metrics.jsonl`: canonical JSON Lines for run metadata, training epochs,
  validation epochs, checkpoint events, and structured failures;
- `last.pt`: the latest interval or final epoch checkpoint;
- `best.pt`: the checkpoint with the minimum configured validation-loss
  component.

Run metadata records complete configurations, Python and PyTorch versions,
resolved device, and the exact physical-family IDs used in the requested
training and validation partitions. Source provenance has four fields:
`git_commit`, `git_dirty`, sorted `git_status_porcelain`, and
`source_tree_sha256`. The digest hashes sorted relative paths and raw contents
of every Python source under `prototype/`; it identifies the code actually
available to the run even when the working tree is dirty. A commit identifies
repository history, while the source digest identifies the exact source tree.
Resumed-run metadata records the fully resolved checkpoint path.

Every loss component is normalized independently for each physical example.
The scalar optimization loss and epoch metrics are arithmetic means of those
per-example values, so mixed history lengths and batch boundaries do not
change example weighting. Inapplicable or empty component selections
contribute a finite zero for that example. Every named loss, learning rate,
global step, example count, codebook utilization, perplexity, and active-code
count is retained.

Codebook diagnostics are epoch-global statistics. Assignment counts are
summed across every batch first; active-code count, utilization, and
perplexity are then computed once from that aggregate distribution. They are
not averages of per-batch utilization or perplexity.

Checkpoints contain model and optimizer states, completed epoch, global step,
both configurations, best validation metric, and Python/PyTorch CPU and
available CUDA RNG states. They also retain the split manifest, partition
names, and exact physical-family IDs, so resume cannot silently switch
training data. Files are written to a flushed sibling temporary file and
atomically replaced. `best.pt` is updated only from validation data; test
partitions are rejected by both the direct API and CLI. Associated best
checkpoints must not come from a later epoch or global step than the resumed
checkpoint. Epoch and global-step fields must be nonnegative actual integers.

A fresh, non-resume run rejects an output directory that already contains
`metrics.jsonl`, `last.pt`, or `best.pt`; it will not append to or overwrite a
managed artifact from an earlier run. Resume may append only when the output
directory is the resume checkpoint's own run directory. Relocation requires
an empty destination with no managed artifacts; an occupied unrelated
destination is rejected without modifying its files. For a valid relocation,
the associated prior `best.pt` is validated and atomically copied into the
new directory. It remains the reported best checkpoint unless a later
validation loss improves the restored best metric. Resume checkpoint targets
are resolved before comparing directories or locating sibling `best.pt`, so
equivalent paths and symbolic links cannot bypass relocation protection.

Training keeps EMA codebook updates enabled, applies configurable gradient
clipping, and checks gradients both before and after clipping. After every
optimizer step it rejects non-finite floating/complex model parameters,
buffers, or optimizer-state tensors before logging a successful epoch or
saving a checkpoint. Validation uses `model.eval()` and `torch.no_grad()`,
does not update model or EMA state, and restores training mode afterward.
Python and PyTorch are seeded, DataLoader shuffling is seeded per epoch, and
evaluation order is fixed. This supports useful reproducibility but does not
claim bitwise determinism for every CUDA kernel.

Expected configuration, path, partition, and output-collision failures are
reported on standard error as one canonical JSON object with a stable error
code and exit status 2; setup failures leave standard output empty. Once a
legitimate run owns its logger, runtime
failures are also appended as structured `failure` events. Errors detected
before ownership is established never append to an unrelated `metrics.jsonl`.

This milestone remains teacher-forced reconstruction plumbing. It does not
implement autoregressive decoding, CAD-kernel execution of predictions,
counterfactual training, a learned code prior, distributed training, or final
research-scale hyperparameters.
