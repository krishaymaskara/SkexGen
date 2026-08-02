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

## Constraint-manifold replay diagnostic

`constraint_manifold_replay` is a standalone, read-only analysis of an
already published repaired Phase B bundle. It verifies the source bundle's
15-entry manifest before scientific parsing, reconciles the unchanged
baseline against every stored conversion result, and applies the four arms
frozen in the
[replay preregistration](../../docs/specifications/flat_baseline_phase_b_constraint_manifold_replay.md).
It does not import Torch and never invokes corpus-payload, checkpoint,
model-execution, or target-continuous-geometry code paths. The input must be
the flattened local audited copy; a symlink-backed cluster publication is
flattened as part of the separately verified download/freeze process.

The caller supplies a nonexistent output namespace:

```bash
python3 -m prototype.flat_baseline.constraint_manifold_replay \
  --bundle /path/to/recovered-phase-b-bundle \
  --output /path/to/new-diagnostic-namespace
```

Publication is atomic and no-replace. A successful namespace contains exactly
`replay_records.jsonl`, `family_transitions.csv`, `aggregate_summary.json`,
`projection_failures.csv`, `run_metadata.json`, and
`sha256-manifest.txt` through its stable published path. On Linux the stable
path may be a relative symlink to a hidden sibling backing directory when
`renameat2(RENAME_NOREPLACE)` is unavailable; the command prints the actual
publication backend after success. These artifacts are diagnostic only: they
cannot change the frozen Gate C or Gate D failures or assign a Gate E result.

## Categorical-isolation replay diagnostic

`categorical_isolation_replay` is the separately preregistered 2-by-2
reference-plane/profile-category diagnostic over the immutable recovered
bundle and completed constraint-manifold replay. It verifies both namespaces
before scientific parsing, requires unchanged parent projection semantics,
and reconciles its model/model and full-oracle arms exactly with the parent
records. Plane-only and profile-only transformations receive distinct
one-field oracle capabilities.

The caller supplies both immutable inputs and a new nonexistent output:

```bash
python3 -m prototype.flat_baseline.categorical_isolation_replay \
  --source-bundle /path/to/recovered-phase-b-bundle \
  --parent-replay /path/to/completed-constraint-manifold-replay \
  --output /path/to/new-categorical-isolation-namespace
```

The exact arms, factorial arithmetic, five-family capsule cohort, seven-file
output contract, and non-retroactivity boundary are frozen in the
[categorical-isolation preregistration](../../docs/specifications/flat_baseline_phase_b_categorical_isolation_replay.md).
This diagnostic does not rerun or reinterpret the completed replay and cannot
alter Gate C, Gate D, or the unevaluated Gate E state.

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
OpenCascade execution. Its final manifest is an exact 15-entry stable-path
contract: six evaluation artifacts, eight workflow-evidence artifacts, and
the Gate A-D input report. Symlink backing names and the manifest itself are
excluded.

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

## Constrained-profile V2 tiny training

The Stage 2E smoke workflow is additive and train-only:

```bash
python -m prototype.flat_baseline.train_constrained_v2 \
  --corpus-dir /path/to/controlled-corpus \
  --output-dir /new/path/v2-tiny-overfit
```

It deterministically selects 6–12 training-partition examples, uses fresh
normal VQ initialization, performs at most 500 teacher-forced optimizer steps,
and writes a strict version-2 checkpoint. It does not load a validation or test
payload, perform validation-based checkpoint selection, initialize from a V1
checkpoint, or constitute a full pilot or scientific result.

The bounded smoke evaluates deterministic snapshots at steps 100 and 200 for
diagnostics and makes its success decision only from the initial snapshot and
the completed step-500 snapshot. A synthetic train-fixture diagnostic found
that 100 steps can be insufficient during a temporary fresh-normal VQ/EMA
code-assignment transition, while the unchanged objective recovered by step
200. No VQ freeze or warmup policy was adopted.

The authoritative CPU wrapper is
`adroit/constrained_v2_tiny_overfit_cpu.slurm`; set `REVIEWED_COMMIT` to the
reviewed clean commit before submission.

## Constrained-profile V2 ordinary-validation pilot

Stage 2F adds an engineering pilot that starts a fresh V2 model with normal
VQ initialization, trains on the complete authorized training partition for
exactly two shuffled deterministic epochs (batch size 8), and evaluates the
complete ordinary IID validation partition after each epoch:

```bash
python -m prototype.flat_baseline.run_constrained_v2_pilot \
  --corpus-dir /path/to/controlled-corpus \
  --output-dir /new/immutable/path/v2-iid-pilot \
  --device cpu
```

The two-epoch budget is 136 optimizer steps and 1,088 example presentations
for the authoritative 544-family training partition. It is the smaller
meaningful budget considered for the CPU pilot: every training family is seen
twice, without extending the run in response to validation results.

Teacher-forced validation reports every V2 loss component, applicable profile
counts, family confusion and per-class accuracy, family-routed compact
parameter errors, and VQ usage. Length-conditioned autonomous validation
reports validity by profile, node count, operation family, and operation
count; family confusion; finite/canonical/conversion rates; and registered
conversion failure codes. The final checkpoint is always saved. The
lowest ordinary-validation total-loss checkpoint is identified only as
predeclared diagnostic metadata, then strictly reloaded to reproduce both
validation summaries without VQ EMA mutation.

This workflow is ordinary-validation-only. It rejects systematic and held-out
test access, never initializes from V1 or a prior V2 checkpoint, performs no
hyperparameter search or early stopping, and is not a final training run or a
V1-versus-V2 scientific comparison. The authoritative CPU wrapper is
`adroit/constrained_v2_pilot_cpu.slurm`; it must be submitted only after
setting `REVIEWED_COMMIT` to the reviewed clean Stage 2F commit. No pilot
success is claimed until that Slurm job completes.

## Frozen V2 IID-pilot diagnostic

Stage 2G is a read-only diagnostic for the frozen Stage 2F epoch-2 checkpoint.
It evaluates only the complete training and ordinary IID-validation
partitions, never trains, and verifies that model state, VQ state, RNG state,
and checkpoint bytes remain unchanged:

```bash
python -m prototype.flat_baseline.diagnose_constrained_v2_pilot \
  --corpus-dir /path/to/controlled-corpus \
  --checkpoint /path/to/frozen/epoch-0002.pt \
  --epoch1-checkpoint /path/to/frozen/epoch-0001.pt \
  --output-dir /new/immutable/path/v2-iid-diagnostic \
  --device cpu
```

The diagnostic reports encoder/prequantized and codebook/EMA collapse
evidence, teacher-forced and autonomous family-head behavior, strict failure
localization, and six frozen replay arms. Arm A is fully autonomous; Arm B
uses teacher-forced history with predicted current outputs; Arms C-E apply
increasingly complete, explicitly labeled oracle discrete interventions; and
Arm F changes only the profile family and its deterministic canonical
consequences within generated history. Metrics from oracle arms are never
reported as autonomous performance.

The diagnostic emits evidence-based bottleneck classifications under
predeclared thresholds. It neither prescribes nor applies a production fix.
The authoritative wrapper is
`adroit/constrained_v2_pilot_diagnostic_cpu.slurm`; it uses a new diagnostic
directory and never writes into the Stage 2F pilot run.

## Frozen reference-plane geometry diagnostic

Stage 2H is a second read-only diagnostic over the same frozen epoch-2 V2
checkpoint. It traces reference-plane geometry through the schema,
serialization, normalization, V2 15-channel non-profile output, applicability
mask, and strict controlled-domain validator. Plane channels 0-2 are an origin
normalized by the length scale 4.0; channels 3-5 and 6-8 are unscaled,
dimensionless x- and y-axis vectors. The general schema requires orthonormal
axes within `1e-6`, while the controlled contract additionally requires zero
origin and the exact categorical `XY`, `XZ`, or `YZ` frame within `1e-6`.

```bash
python -m prototype.flat_baseline.diagnose_reference_plane_geometry \
  --corpus-dir /path/to/controlled-corpus \
  --checkpoint /path/to/frozen/epoch-0002.pt \
  --output-dir /new/immutable/path/v2-reference-plane-diagnostic \
  --device cpu
```

H0 exactly reproduces Stage 2G Arm E. H1a replaces only the authoritative
origin, H1b only the authoritative basis, and H1c the complete authoritative
plane. H2 replaces all non-profile continuous geometry. Target-free H3a zeros
only the origin, H3b Gram-Schmidt-projects only the predicted basis, and H3c
combines both operations. These H3 geometry interventions read no target
geometry but remain oracle diagnostic arms because they start from H0's
authoritative discrete structure. H5 replaces only compact profile parameters. H6
starts from fully autonomous Stage 2G Arm A and constructs an exact controlled
frame solely from each generated node's predicted plane category; missing or
invalid categories are never replaced from targets. H4 is explicitly
inapplicable because no independent bounded plane scalar remains after the
categorical orientation is fixed. The workflow does not train, create an
optimizer, mutate VQ state, or access systematic or held-out test partitions.
Its authoritative CPU wrapper is
`adroit/constrained_v2_reference_plane_diagnostic_cpu.slurm`.

## Constrained-profile V3 canonical-plane decoder

V3 is an additive production contract motivated by the frozen Stage 2H
finding that controlled reference-plane channels have no independent
continuous freedom. V1 and the immutable V2 baseline remain available under
their original entry points, output contracts, checkpoints, pilots, and
diagnostics. V3 has the distinct identity
`B0-FLAT-CONSTRAINED-PROFILE-CANONICAL-PLANE-v3` and uses checkpoint,
configuration, and decoder-contract version 3.

The serialized geometry remains width 39. Channels 0-8 are now constructed
deterministically from the predicted `XY`, `XZ`, or `YZ` category; channels
9-32 retain the existing category-conditioned compact-profile reconstruction;
and channels 33-38 retain the unchanged learned axis/operation geometry. Thus
the V3 continuous head has width 6, reduced from V2's width 15, in the exact
serialized order `(33, 34, 35, 36, 37, 38)`. A predicted reference-plane node
with a sentinel, missing, malformed, or out-of-range plane category fails with
`invalid_predicted_reference_plane_category`; it is never repaired from an
authoritative category or a default frame.

V3 removes channels 0-8 from continuous regression while preserving the
reference-plane categorical objective. Teacher-forced reporting constructs
the current plane from the predicted current category, and autonomous
feedback uses generated categories and accepts no targets. Axis channels
33-38 are intentionally unchanged and axis redesign is outside this revision.

The fresh V3 IID engineering pilot uses seed 2026, the 544-example train
partition, the 68-example ordinary IID-validation partition, two epochs,
batch size 8, 136 optimizer steps, and CPU execution. It starts from a fresh
V3 initialization, writes to a new immutable V3 run directory, and uses only
positive engineering acceptance predicates. Systematic and held-out test
partitions are prohibited. This section describes the prepared protocol, not
a scientific-success result; no such claim is made before its authoritative
Slurm run completes.

## Constrained-profile V4 node-conditioned categories

The immutable V3 engineering pilot completed its two-epoch protocol but
produced `0/68` autonomous valid reconstructions. All 68 generated planes and
profiles were canonical; the first failures were instead 56
`categorical_pad_sentinel` and 12 `node_category_applicability` failures. V4
therefore leaves V3 geometry, model capacity, logits, losses, encoder, VQ,
relations, and continuous heads unchanged and introduces only an additive
prediction-selection contract.

For each predicted node, V4 preserves the raw argmax for the retained fields
`operation_type`, `boolean_mode`, `direction`, `reference_plane`, and
`loop_role`. It then derives applicability solely from the predicted node
type. Applicable fields select the highest-logit valid real class after
excluding `<pad>`, `<none>`, reserved classes, and node-incompatible classes;
inapplicable fields deterministically receive `<none>`. Predicted padding rows
receive `<none>` in every retained field. The constrained IDs alone enter the
serialized record, canonical-plane construction, strict conversion, and
autonomous feedback. Raw IDs remain read-only reporting evidence, accompanied
by an exact correction mask.

This is not an oracle repair: V4 never reads an authoritative current node
type, category, applicability mask, or geometry during selection. A
structurally admissible but wrong category remains a categorical prediction
error. Strict V4 conversion independently verifies the node-conditioned
contract and never repairs externally supplied records.

The immutable V4 pilot completed that protocol. Constrained conversion
succeeded for `68/68` IID-validation examples, versus `22/68` for the raw
categorical arm, while complete CAD validity remained `0/68`. Sentinel and
categorical-applicability failures were eliminated. The remaining first
failures were 45 `invalid_node_grammar` and 23 `unexpected_edge` results.

## Constrained-profile V5 prefix node grammar

V5 is the additive production response to the frozen V4 node-grammar result.
It retains the V4 architecture, parameter count, raw logits, all losses,
categorical selector, geometry construction, VQ behavior, relation heads, and
edge decoding. Its only scientific change is the deterministic selection of
predicted node IDs under the frozen controlled prefix grammar:

```text
E:  reference_plane, sketch, profile, extrude
R:  reference_plane, sketch, profile, axis, revolve
EE: reference_plane, sketch, profile, extrude, sketch, profile, extrude
ER: reference_plane, sketch, profile, extrude, sketch, profile, axis, revolve
RE: reference_plane, sketch, profile, axis, revolve, sketch, profile, extrude
RR: reference_plane, sketch, profile, axis, revolve, sketch, profile, axis, revolve
```

The legal requested lengths are exactly 4, 5, 7, 8, and 9. At each active
position V5 exhaustively enumerates the tiny immutable grammar and masks any
candidate that cannot complete to the authorized exact length. This uses only
the constrained prefix and authorized length; it never reads the current or
future target, target operation family, categories, or geometry. The raw node
argmax remains same-history shadow evidence. Only the constrained node ID
controls V4 categorical applicability, plane/profile/axis construction,
operation counting, pointers, and autonomous feedback. The raw shadow is not
an independent rollout and never changes future logits.

The fresh V5 pilot retains seed 2026, 544 training examples, 68 ordinary IID
validation examples, two epochs, batch size 8, 136 optimizer steps, 1,088
example presentations, normal VQ behavior, and CPU execution. It uses a new
immutable V5 output root. Systematic and held-out test access is prohibited.
The edge decoder is deliberately unchanged, so V5 may still have zero complete
CAD validity; pilot completion is an engineering gate, not a scientific-success
claim.

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
