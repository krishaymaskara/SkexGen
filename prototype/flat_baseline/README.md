# B0-FLAT-MIXED-VQ baseline core

`prototype.flat_baseline` is the first neural-model milestone for the
structured extrude-and-revolve experiments. It implements one flat,
mixed discrete/continuous encoder, one mixed EMA vector-quantization
bottleneck, teacher-forced training, length-conditioned autoregressive
decoding, paired validation evaluation, VQ-collapse diagnosis, and
deterministic train-only k-means codebook initialization and retraining. It
contains no graph encoder, factored stream, hybrid latent, code prior,
counterfactual training, or learned edit mechanism.

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

### Post-run analysis

After a Slurm job has completed, failed, or been canceled, analyze its
quiescent run directory on a CPU with:

```bash
python3 -m prototype.flat_baseline.analyze_run \
  --run-dir /path/to/run \
  --output-dir /path/to/new-analysis-directory
```

This is not a live-training monitor. The analyzer strictly validates the
canonical JSONL history and CPU-loaded trusted project checkpoints, supports
ordinary same-directory resumes, and atomically writes schema-versioned
`summary.json` plus `metrics.csv`. Complete and valid incomplete runs return
exit code 0; structurally invalid artifacts return one canonical error on
standard error and exit code 2. Diagnostic warnings are labeled heuristics and
do not change structural status or make scientific claims about autonomous
decoding, CAD-kernel execution, or the Week 3 gate. Raw per-code assignment
identities were not persisted and therefore cannot be reconstructed.

Checkpoint evidence is confined to the resolved run directory before loading.
A symbolic link is accepted only when it resolves to a regular file directly
inside that same resolved directory; links escaping the run are rejected.
Merely placing unlogged `best.pt` or `last.pt` files in a run cannot make it
complete. The final `last.pt` and an ordinary `best.pt` require matching log
events. The sole exception is an explicitly resumed relocated run whose
compatible preserved `best.pt` predates the metric history available in the
new directory. Such unavailable earlier epoch metrics remain `null` with a
structured limitation reason.

The training-infrastructure milestone described above remains
teacher-forced reconstruction plumbing. Later sections describe
autoregressive evaluation and VQ diagnostics, but the package still does not
implement CAD-kernel execution of predictions, counterfactual training, a
learned code prior, distributed training, or final research-scale
hyperparameters.

## Phase A length-conditioned autoregressive decoding

The package includes a raw, greedy
`decode_length_conditioned()` interface for inspecting decoder behavior
without shifted ground-truth node records. It starts from the model's
existing learned BOS vector, feeds back each complete raw argmax node record,
and predicts edges and operation pointers from one final aligned pass over
the generated prefix.

This is **oracle-node-count-conditioned**, not fully autonomous termination.
The caller supplies each target canonical node count because the model has no
trained EOS token or node-count head. For corpus evaluation,
`node_count_source` must be recorded as `target_canonical_metadata`. Node
count substantially leaks operation-template information in this version.
The operation count is not supplied: it is derived from the generated raw
extrude/revolve node types, and only the corresponding bounded pointer-query
outputs are retained.

The Phase A result contains raw categorical argmax selections, normalized
continuous geometry, deterministically derived geometry-applicability masks,
all directed edge predictions, and raw operation pointers. It does not add
CAD identifiers, sanitize malformed output, convert predictions into the
controlled representation, or claim representation validity or OpenCascade
executability. No unconditional latent prior exists; latent indices must
come from an encoded example or an explicitly supplied sequence.

Phase A was added without changing the trained state contract. Adroit
validation compared ordered state-dict keys, shapes, and dtypes against base
commit `ba611fb`, then strictly loaded the existing epoch-50 checkpoint.

## Phase B paired validation evaluation

`evaluate_length_conditioned.py` evaluates two decoder paths from identical
reconstructed VQ memory:

```text
encode once
→ select VQ indices once
→ reconstruct memory once
├── teacher-forced decoding
└── predicted-history decoding
```

This isolates ground-truth versus generated prefix history from
straight-through numerical differences or a second encoder/VQ pass. Both
paths remain conditioned on the authoritative target node count. The
evaluator records that limitation because node count partially reveals the
operation template.

Raw predictions pass through prerequisite-gated conversion with three
separate validity layers:

1. `raw_integrity_valid` — raw decoder output is complete, typed, finite, and
   internally well formed;
2. `reconstruction_target_valid` — output can form the structural flat target
   independently of controlled-domain semantics;
3. `controlled_domain_valid` — output obeys the restricted version 1 CAD
   grammar and cross-record constraints.

Each example has one deterministic primary failure and only independently
meaningful secondary failures. Downstream checks are suppressed when their
prerequisites fail, preventing one malformed node from creating a cascade of
uninterpretable errors. Nonfinite predictions remain in attempted-example
denominators and are reported separately from finite-subset geometry error.

The evaluator reports:

- raw completion and all three validity rates;
- node-type token and exact-sequence accuracy;
- finite-subset geometry MAE and RMSE;
- exact operation-type sequence rate;
- operation-pointer accuracy;
- edge micro precision, recall, and F1;
- per-template and other deterministic strata.

Historical evaluator invocations retain evaluation schema version 1, the
original CSV and artifact schemas, corpus-wide loading and validation, and
optional raw prediction publication. They do not require latent-usage
publication or any repaired-checkpoint metadata.

Repaired evaluation schema version 2 publishes one shared latent-usage record
derived from the reconstructed memory used by both decoding paths. It
contains the complete zero-inclusive codebook histogram, active-code count,
utilization, perplexity, dead-code count, and dead-code rate. Latent
assignments are counted once rather than once per decoder path. The headline
CSV includes exact ten-field and exact typed-edge-set rates and one
`shared_latent` row for active-code count, perplexity, and utilization.

It publishes canonical:

```text
conversion_failures.csv
examples.jsonl
metrics.csv
run_metadata.json
summary.json
```

Optional `raw_predictions.jsonl` output is explicit. Publication uses a
collision-safe no-replace contract and an artifact verifier. Test evaluation
requires an explicit `--allow-test-evaluation` flag and is recorded in run
metadata; validation workflows do not enable it.

Schema 2 and its expanded CSV fields are enabled only by the repaired
evaluation contract. The repaired deterministic-smoke contract uses
manifest-only partition
authority followed by partition-scoped payload loading. It resolves the
lexicographically first six validation IDs before inference, opens only
their twelve continuous/quantized payloads, records zero train/test family
payload access, and rejects an ID assigned to another partition before
payload access. The `--repaired-smoke-contract` flag additionally requires
validation, six families, batch size 3, CPU, raw predictions, authoritative
544/68/68 counts, the repaired epoch-44/global-step-748 checkpoint, its
frozen SHA-256, 32-code model configuration, and train-k-means provenance.
The file hash is compared before `torch.load`.

The separate `--repaired-full-contract` flag preserves those repaired
checkpoint, partition-scoped loading, paired-decoding, schema-2, and raw
publication rules while requiring no family limit, batch size 32, and exactly
all 68 authoritative validation families. It cannot be combined with the
smoke contract or test authorization. Its post-publication report contains
the frozen Phase B metrics, shared latent usage, and machine-readable Gate
A-D inputs, but explicitly leaves Gate E and final acceptance undetermined.

A validation invocation is:

```bash
python3 -m prototype.flat_baseline.evaluate_length_conditioned \
  --corpus-dir /path/to/controlled-corpus \
  --checkpoint /path/to/best.pt \
  --split-manifest iid \
  --partition validation \
  --output-dir /path/to/new-evaluation \
  --batch-size 16 \
  --device cpu \
  --write-raw-predictions
```

The separate Adroit repaired-smoke workflow is:

```text
prototype/flat_baseline/adroit/evaluate_repaired_smoke_cpu.slurm
```

It retains two collision-safe publications and verifies every declared
deterministic artifact byte for byte. The earlier
`evaluate_validation_cpu.slurm` remains the historical collapsed-checkpoint
workflow and is not the repaired submission target.

After review, the full repaired-validation workflow is:

```text
prototype/flat_baseline/adroit/evaluate_repaired_validation_cpu.slurm
```

It uses a new commit-and-job-qualified namespace, processes all 68 validation
families, runs the complete artifact verifier, and preserves environment,
scheduler, repository, checkpoint, corpus-manifest, regression, artifact-hash,
and scheduler-log-path evidence. It performs neither test-family loading nor
OpenCascade execution.

This phase reconstructs authoritative flat targets and measures symbolic
predictions. It does not yet synthesize stable CAD identifiers into a full
`CADHistory` or execute predicted programs in OpenCascade.

## VQ-collapse diagnosis

`diagnose_vq_collapse.py` is a deterministic train/validation-only diagnostic,
not a trainer. It reconciles authoritative partitions and inspects:

- prequant vector diversity and assignment regions;
- numerical codebook diversity and EMA state;
- empirical code usage;
- gradient flow to the encoder and pre-VQ projection;
- decoder sensitivity to latent changes;
- masking and loss balance;
- available checkpoint history and collapse timing.

Its root-cause vocabulary distinguishes encoder-output collapse,
nearest-code assignment collapse, codebook-embedding collapse, decoder latent
insensitivity, loss/mask imbalance, gradient-flow failure, training-time
collapse, and unresolved combinations. Required artifacts are published
atomically with SHA-256 integrity metadata. Verification can compare two
bounded smoke outputs for byte-identical replay and validate a full diagnostic
report without touching the test partition.

The command family is:

```bash
python3 -m prototype.flat_baseline.diagnose_vq_collapse run \
  --corpus-dir /path/to/controlled-corpus \
  --training-run-dir /path/to/training-run \
  --output-dir /path/to/new-diagnosis \
  --reviewed-commit <full-commit>

python3 -m prototype.flat_baseline.diagnose_vq_collapse verify \
  --output /path/to/diagnosis \
  --corpus-dir /path/to/controlled-corpus \
  --training-run-dir /path/to/training-run \
  --reviewed-commit <full-commit>
```

## Train-only k-means initialization

`TrainingConfig.vq_init` accepts `normal` and `train-kmeans`; normal
initialization remains the default. The k-means treatment:

1. constructs the ordinarily seeded untrained model;
2. collects each learned latent-query prequant vector from authoritative
   training families only under `eval()` and `no_grad()`;
3. runs seeded k-means++ and eight fixed Lloyd iterations;
4. initializes all embedding vectors;
5. initializes EMA counts with positive cluster-proportional pseudo-counts
   whose total equals the codebook size;
6. initializes EMA sums consistently from pseudo-counts and centers;
7. creates the optimizer only after initialization.

Matching total pseudo-count mass to normal initialization avoids changing EMA
inertia as a hidden second intervention. Initialization reports include
partition counts, vector and center diagnostics, EMA equations, hashes, and
explicit zero validation/test use.

## Bounded pilot and full retraining

`vq_pilot.py` implements a fixed five-epoch treatment pilot with early
step-level and epoch-global code-usage diagnostics. The pilot fails if epoch
2 training has fewer than two active codes or perplexity below 2.0. It returns
`PASS_FOR_FULL_RETRAIN` only when epoch-5 train and validation each have at
least two active codes and perplexity at least 2.0, all required values are
finite, provenance is valid, and no test examples were used.

`vq_full_retrain.py` promotes an accepted pilot by exact epoch-boundary resume.
It restores model, VQ/EMA, optimizer, RNG, epoch, global-step, best-metric, and
initialization provenance; it does not rerun k-means. The authoritative epoch
budget is reconciled against the original baseline checkpoint and log.

During full retraining, either fewer than two active training codes or
perplexity below 2.0 is a threshold violation. Two consecutive completed
training violations stop the run. `PASS_FOR_EVALUATION` requires:

- completion of the authoritative epoch budget;
- a provenance-valid selected best checkpoint;
- finite selected-checkpoint values;
- selected validation usage of at least two active codes and perplexity at
  least 2.0;
- confirmation that the test partition was not evaluated.

`PASS_FOR_EVALUATION` is deliberately not final model acceptance. It says the
training and minimum assignment-usage gates passed; reconstruction,
executability, semantics, localized edits, and comparative scientific value
must be established separately. Current verified project state and
limitations are tracked in
[the project status](../../docs/status.md).

## Related evidence

The [B0 training-infrastructure validation
record](../../docs/experiments/b0_training_infrastructure_validation.md)
documents the completed CPU validation and CUDA engineering smoke test.
Subsequent verified records cover:

- [the original one-code collapse](../../docs/experiments/b0_original_training_collapse.md);
- [Phase A compatibility](../../docs/experiments/b0_phase_a_contract_validation.md);
- [Phase B paired validation](../../docs/experiments/b0_phase_b_validation_evaluation.md);
- [the VQ-collapse diagnosis](../../docs/experiments/b0_vq_collapse_diagnosis.md);
- [the train-k-means pilot](../../docs/experiments/b0_train_kmeans_pilot.md);
- [the full train-k-means retraining](../../docs/experiments/b0_train_kmeans_full_retrain.md).
