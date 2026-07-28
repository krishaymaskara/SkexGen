# Project Progress Report

Prepared for mentor review  
July 28, 2026

## Project objective

This project investigates how CAD feature structure and geometry should be
represented to support systematic generalization and localized editing. The
experimental domain is deliberately restricted to single-body construction
histories containing sketches, extrusions, and revolutions. Revolve support is
being used to create a controlled setting with meaningful feature references
and dependencies; adding revolve is not itself the intended research
contribution.

The main research comparison is between a flat mixed representation, a typed
dependency-graph representation with discrete structure and discrete
structure-conditioned geometry, and a typed graph with discrete structure and
continuous structure-conditioned geometry. The central questions are whether
explicit feature dependencies improve compositional generalization and edit
locality, and whether continuous conditioned geometry improves numerical
precision and extrapolation relative to fully discrete geometry.

At the current stage, the controlled representation, data, validation, and
counterfactual infrastructure are implemented. The first flat mixed/VQ
baseline is also implemented and trained. The graph models and the final
comparative experiments have not yet been implemented.

## Progress on the research foundation

I first mapped the inherited SkexGen system and established that its
pretrained sampling path could run in the available Princeton Adroit
environment. A reduced pretrained smoke test, Slurm job 3319488, loaded the
published checkpoints, sampled 16 candidates, retained 15 CAD results, and
completed OBJ export in 14 seconds. This was an environment and pipeline
operability check, not a reproduction of the paper's full evaluation.

I then implemented a versioned representation for controlled CAD histories.
The representation includes reference planes, sketches, profiles, revolve
axes, extrude and revolve operations, Boolean modes, chronological operation
order, and typed dependency/reference edges. Structure and numerical geometry
are separated explicitly. Continuous and quantized encodings share a common
physical-family identity, which prevents two encodings of the same history
from being treated as independent training examples or crossing data
partitions.

On top of this representation, I implemented a deterministic controlled-data
generator for the operation templates E, R, EE, ER, RE, and RR. It produces
family-level IID and systematic split manifests. The current systematic split
infrastructure includes operation-template, history-depth, and geometry-range
conditions. Partition authority is always the physical source family rather
than an individual serialized sample.

An early OpenCascade audit of 60 physical families was important in refining
the generator. Of those families, 47 executed successfully, 10 produced
Boolean operations with no meaningful effect, and 3 produced an invalid
zero-solid result. Continuous and quantized variants agreed perfectly on
success or failure. These failures demonstrated that schema validity alone
was insufficient and motivated a deterministic analytical
Boolean-feasibility policy. OpenCascade results are used as independent
validation rather than as an environment-dependent selection filter.

I also implemented an independent CAD-kernel execution package with structured
failure categories and deterministic reporting. It validates that each
operation and final result is non-null, kernel-valid, single-solid, and has
positive finite volume. JOIN and CUT operations must have a measurable
semantic effect.

The counterfactual benchmark is now implemented for one-factor changes to
profile extent, extrusion distance, revolve angle, operation direction, and a
later operation's JOIN/CUT mode. Each pair records the intended changed
attribute, expected unchanged attributes, unchanged edges and operation
order, and causally downstream operations. Selection is deterministic and
endpoint-disjoint.

The main counterfactual OpenCascade audit ran as job 3321612 with an
independent replay in job 3321613. It produced 68 edit families, 136
encoding-specific edit samples, 136 physical endpoints, and 272 encoded
endpoint histories. All 272 encoded endpoints executed successfully, all 136
edit samples passed the pair audit, and continuous and quantized executions
agreed for all 136 physical endpoints. The preserved note reports that the
primary and replay output trees were byte-identical. This establishes the
executability of the fixed counterfactual benchmark, but it does not yet
measure learned edit performance.

Finally, I implemented a shared model-data boundary that exposes aligned flat
and graph views of the same physical histories and reconstruction targets.
The interface enforces family-level split authority, exact agreement between
continuous and quantized variants, deterministic vocabularies and ordering,
geometry masks, typed edges, batching offsets, and explicit reconstruction
targets. Real-PyTorch validation in job 3322011 passed 193 tests under Python
3.8.13 and PyTorch 1.11.0 on Adroit. This confirms that the interface works
with the target environment rather than only with local tensor-compatible
test adapters.

These results collectively complete the controlled CAD foundation needed for
the planned representation comparisons.

## Flat mixed/VQ baseline

The first neural baseline, B0-FLAT-MIXED-VQ, is implemented. It uses a flat
chronological node stream with mixed categorical and continuous geometry
inputs, a Transformer encoder, learned latent queries, one 32-entry EMA VQ
codebook, and a shared reconstruction decoder. Dependency edges are
intentionally omitted from the B0 encoder so that later graph models can test
the value of explicit typed relationships.

The training infrastructure includes deterministic configuration,
checkpointing, relocation-safe resume, provenance checks, JSONL metrics,
validation, and artifact verification. CPU job 3322236 passed all 63
flat-baseline and 39 model-data tests available at that stage. CUDA smoke job
3322251 exercised forward and backward optimization, EMA VQ updates,
checkpoint creation, checkpoint relocation, restoration of model, optimizer,
and random-number-generator state, and resumed training on an NVIDIA A100
GPU.

These infrastructure jobs ran from a dirty working tree based on commit
73349b4. The intended production-source changes were subsequently committed
as 19a1859. I have retained that distinction in the experiment record rather
than presenting the jobs as clean-checkout validation of the later commit.

Job 3322880 generated the current authoritative pilot corpus. It contains 680
physical families and 1,360 continuous/quantized serialized variants. The IID
split contains 544 training families, 68 validation families, and 68 held-out
test families. The test partition has not been used for training, checkpoint
selection, diagnosis, or evaluation.

The first meaningful B0 training run, job 3323110, completed all 50 epochs and
850 optimizer steps. The minimum validation-loss checkpoint was epoch 50,
with validation total loss 0.4498. However, the VQ representation collapsed
to one active code out of 32, with perplexity 1.0 and utilization 0.03125.
Validation had already collapsed to one code in epoch 1. The run therefore
passed operationally but failed the scientific representation gate. This was
an important negative result: low teacher-forced loss did not imply that the
model had learned a meaningful discrete latent representation.

I next added oracle-node-count-conditioned autoregressive decoding while
preserving the trained state contract. Phase A compatibility job 3323988
confirmed that the base and extended models had the same 126 ordered state
entries and identical state-contract hashes, and that the original epoch-50
checkpoint loaded strictly.

The Phase B evaluator measures two reconstruction paths from the same
reconstructed latent memory: teacher-forced prefix feedback and
predicted-history feedback. It separately reports raw completion, raw
integrity, reconstruction-target validity, controlled-domain validity,
structural accuracy, geometry error, operation and pointer accuracy, typed
edge metrics, and code usage. Repeated smoke evaluations were byte-identical,
and the full original-checkpoint evaluation completed in job 3324856 on all 68
validation families.

For the collapsed checkpoint, both paths produced complete and internally
well-formed raw records, but neither produced a controlled-domain-valid
history. Teacher-forced versus predicted-history node-type accuracy was
0.8705 versus 0.6093; pointer accuracy was 1.0000 versus 0.3772; edge
micro-F1 was 0.8651 versus 0.5616; and exact complete ten-field match was zero
for both paths. Geometry MAE was 3.3377 versus 3.4623. All 136 validation
latent assignments selected code 17. These results showed that
predicted-history feedback caused additional degradation, but that the
one-code latent state was a more fundamental problem because teacher forcing
also produced no controlled-domain-valid reconstruction.

## Collapse diagnosis and intervention

I implemented a focused VQ-collapse diagnosis and ran it in job 3325296. The
diagnosis distinguished possible encoder collapse, codebook collapse,
gradient failure, decoder insensitivity, loss/mask imbalance, and
nearest-code assignment collapse.

The encoder's prequantization outputs were varied rather than constant, the
codebook vectors remained distinct, finite nonzero gradients reached the
encoder, and the decoder responded materially to changes in latent codes.
Nevertheless, the varied encoder outputs all fell within the nearest-neighbor
region of code 17. The proximate failure was therefore nearest-code assignment
collapse during training. Categorical-loss imbalance and sparse geometry
applicability were identified as plausible contributing factors, but the
diagnostic design did not isolate them causally.

Based on this diagnosis, I introduced a bounded intervention: deterministic
k-means++ initialization using prequantization vectors from the training
partition only. All 32 codebook entries are initialized, and the EMA
pseudo-count mass is matched to the normal initialization so that the
intervention does not silently change EMA inertia. Validation and test
families are excluded from initialization.

The five-epoch pilot, job 3326278, passed its predetermined
PASS_FOR_FULL_RETRAIN gate. Training used 27 active codes at epoch 1 and 5 at
epoch 5; validation used 3 active codes at epoch 1 and 4 at epoch 5.
Validation perplexity remained above 2.0 and finished at 3.6104. The decrease
in the number of touched codes showed under-utilization, but not a recurrence
of complete one-code collapse.

The accepted pilot was then continued by exact epoch-boundary resume through
the original 50-epoch budget in job 3326757. The continuation restored the
model, VQ and EMA state, optimizer, random-number-generator state, epoch,
global step, best-metric state, and initialization provenance without rerunning
k-means. The run completed with no collapse-threshold violations. At epoch 50,
training used 7 codes with perplexity 3.8550 and validation used 6 codes with
perplexity 3.2638.

The predetermined validation-loss rule selected the epoch-44 checkpoint at
global step 748. That checkpoint used 5 validation codes with perplexity
3.1204 and passed the PASS_FOR_EVALUATION training and code-usage gate. This
shows that train-only k-means initialization prevented total assignment
collapse for this corpus and seed. It does not yet establish that the
remaining code usage is sufficient for accurate reconstruction, executable
CAD, interpretable semantics, or localized editing. Five active validation
codes out of 32 remains substantial under-utilization.

## Progress relative to the research plan

In terms of the original eight-week roadmap, the core Week 1 work is largely
complete. The inherited pipeline has been mapped and smoke-tested, the target
environment has been validated, the typed schema has been frozen, and the
controlled benchmark and representation specification have been written.

The planned Week 2 data and execution stage is also substantially complete.
The project now has deterministic controlled generation, family-level IID
and systematic splits, graph and flat serialization, independent OpenCascade
validation, structured kernel failure categories, and a fixed counterfactual
pair set. The 680-family corpus is suitable for the current bounded baseline
work, although it should not yet be interpreted as a final publication-scale
dataset.

The Week 3 flat-baseline stage is implemented and has progressed through
training, failure analysis, a bounded repair, and full retraining. The
engineering infrastructure is reproducible, and the selected repaired
checkpoint has passed the minimum training gate. This stage is not
scientifically complete because the repaired checkpoint has not yet undergone
the frozen paired reconstruction evaluation.

Some later-plan infrastructure has been completed early, particularly the
counterfactual benchmark, systematic split manifests, validity reporting, and
shared flat/graph data contract. However, the discrete graph model, the hybrid
graph model, capacity-matched graph-versus-flat comparisons,
systematic-generalization model results, learned counterfactual editing,
multi-seed analysis, and final manuscript experiments have not begun.

The project is therefore at the boundary between establishing a credible flat
baseline and beginning the graph-model comparison. It would be premature to
start the graph models before confirming that the repaired flat checkpoint
can produce sufficiently credible validation reconstructions through the
shared evaluator.

## Current limitations

The controlled domain remains intentionally narrow: single-body histories
with sketches, extrusions, revolutions, and JOIN/CUT operations. This is
appropriate for isolating representation effects but limits direct claims
about general CAD systems.

The repaired result currently comes from one seed and one bounded corpus.
Codebook usage remains low, and the cause of the remaining under-utilization
has not been isolated. No repaired-checkpoint reconstruction or OpenCascade
execution result exists yet.

The current autoregressive evaluation is conditioned on the target node
count, which reveals template information and is not fully autonomous
generation. This condition is shared and explicit, but conclusions must be
limited accordingly.

No graph neural baseline exists yet, so the principal research hypothesis has
not been tested. There are also no multi-seed comparisons, confidence
intervals, learned edit-locality results, or final systematic-generalization
results. The held-out IID test partition remains untouched and should stay
untouched until the validation protocol and model-selection process are
frozen.

## Immediate next steps

The immediate next step is to run a new validation-only Phase B evaluation of
the selected epoch-44 checkpoint. This should use the already checked-in
paired evaluator without modifying the original-checkpoint record. The run
should preserve the source commit, checkpoint hash, scheduler accounting,
stdout and stderr, complete metrics and per-example artifacts, artifact
manifest, and frozen archive checksum. It should continue to exclude the test
partition.

The repaired-checkpoint report should compare teacher-forced and
predicted-history results against the original collapsed checkpoint. The
primary questions are whether controlled-domain validity rises above zero,
whether structural and geometric reconstruction improve materially,
whether predicted-history degradation is reduced, and whether the repaired
code usage remains above the predetermined collapse threshold during
evaluation. The decision rule for accepting the checkpoint as the flat
baseline should be written before examining the new results.

If the repaired checkpoint produces credible validation reconstructions, the
next evaluation layer should execute reconstructed predictions through
OpenCascade under a frozen protocol. This would distinguish schema-valid,
controlled-domain-valid, and kernel-executable histories. After that, the
project can evaluate latent-code usage and semantics and apply the fixed
counterfactual edits to measure target-change success and preservation of
unaffected attributes.

Once the flat baseline and evaluation path are accepted, the next
implementation priority is the typed graph model with discrete structure and
discrete structure-conditioned geometry. It should use the same physical
families, split authority, reconstruction targets, decoder conditions, and
approximately matched capacity as the flat baseline. The hybrid
discrete-structure/continuous-geometry model should follow only after the
fully discrete graph comparison is reproducible. The final experimental stage
will require systematic split evaluation, localized-edit metrics, failure
analysis, and multiple seeds or uncertainty estimates for the main
comparisons.

The held-out test set should be evaluated only after the validation protocol,
model variants, checkpoint-selection rules, and primary comparisons have been
frozen.

## Reproducibility and documentation

Completed runs are now separated into immutable experiment records rather
than being inferred from progress notes or Git history. The counterfactual
and real-PyTorch model-data bundles are locally checksum-verified. The
original B0 training, Phase A, Phase B, VQ diagnosis, k-means pilot, and full
retraining outputs are preserved in a frozen multi-run archive whose
Adroit-to-macOS SHA-256 matched and whose 91-file internal checksum manifest
passed completely. The repository also has current status, decision,
milestone, specification, package-contract, and evidence-index documents,
plus an automated documentation-integrity checker.

## Points for mentor feedback

The most useful immediate feedback would be on the acceptance boundary for
the repaired flat baseline: specifically, what minimum improvement in
controlled-domain validity and predicted-history reconstruction should be
required before beginning graph-model implementation. Guidance would also be
useful on whether five to six active validation codes out of 32 should be
accepted provisionally if reconstruction is credible, or whether additional
anti-collapse work should precede the graph comparison.

Before implementing the graph models, I would also like to confirm the minimum
model set needed for a convincing comparison and the preferred
capacity-matching rule. In particular, the plan should clarify whether the
first comparison should include only B0 and the fully discrete graph model or
also require a flat factored discrete control before proceeding to the hybrid
graph model.

## Summary

The project has moved from an inherited flat CAD-generation codebase to a
controlled and well-validated research platform for studying structure and
geometry representations. The representation, deterministic data generator,
kernel validation, counterfactual benchmark, shared model-data interface, and
flat mixed/VQ baseline are implemented. The first baseline exposed complete
VQ assignment collapse; that failure was measured, diagnosed, and addressed
with a narrowly controlled train-only k-means intervention. Full retraining
then passed the predetermined minimum code-usage gate and selected an
epoch-44 checkpoint without using the test partition.

The most important unresolved question is now whether that repaired checkpoint
produces credible validation reconstructions. Answering that question is the
immediate gate before OpenCascade prediction evaluation, representation
semantics and edit-locality analysis, and the planned graph-versus-flat model
comparison.
