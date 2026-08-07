# Experiment Evidence Inventory

This page is the index of experiment records and the working inventory of
external evidence. It tracks whether a claim has a durable repository record,
whether its primary artifacts were actually inspected, and what remains to be
retrieved. An implementation commit or a conversation report is not treated
as proof that an external run completed.

## Evidence states

| State | Meaning |
|---|---|
| `verified-local` | A primary artifact bundle is present on this workstation; its archive, checksum, and relevant contents were inspected. |
| `located-external` | A live cluster listing or scheduler record identifies the artifact, but it has not yet been archived and inspected locally. |
| `documented-external` | A repository experiment record exists, but the raw external artifacts were not available during this audit. |
| `reported-only` | Results are present in the supplied project conversation record, but no primary artifact bundle or complete experiment record is available locally. |
| `implementation-only` | The runner and tests are committed, but no completed external run was identified. |
| `missing` | Required evidence or provenance is not known. |

These labels describe evidence availability, not whether an experiment's
scientific result passed.

## Inventory snapshot

This inventory was updated on August 7, 2026 from branch
`graph-v1-experiment-record`. The documentation commit's literal hash is not
embedded because it would be self-referential. The audit used:

- the current repository and its complete local Git history;
- the supplied cross-chat project conversation record;
- `/Users/krishaymaskara/research/audited-runs`;
- a read-only search under `/Users/krishaymaskara/research` and
  `/Users/krishaymaskara/Downloads`.

A read-only SSH connection from Codex reached `adroit.princeton.edu`, but this
session did not have an accepted public key or interactive authentication.
The user subsequently supplied a live Adroit listing and `sacct` transcript.
The corresponding flat-baseline artifacts were then archived, transferred,
checksummed, and inspected. The live transcript is retained as context; the
frozen archive described below is now the primary evidence.

| Evidence item | Implementation/source commit | Job | Primary location | Partition status | Evidence state | Durable record |
|---|---|---:|---|---|---|---|
| Pretrained SkexGen Adroit smoke | Not preserved; branch `baseline-reproduction` | `3319488` | Original logs and samples not recovered | Inherited-model generation smoke; no controlled split evaluation | `documented-external` | [Record](pretrained_skexgen_smoke_test.md) |
| Historical 60-family kernel validation | Not preserved | `3319654` | Preserved bundle referenced by the existing record; current local path not found | Not a learned-model split evaluation | `documented-external` | [Record](kernel_validation_60.md) |
| Counterfactual OpenCascade audit | `3ae4f6cf96b22c6ef33f592dc4e6e47020e059c6` | `3321612`; replay `3321613` | Local archive listed below | Evaluation-only counterfactual corpus; no learned model evaluated | `verified-local` | [Record](counterfactual_opencascade_audit.md) |
| Real-PyTorch model-data validation | Validated source `91d733d56764beac8be06aa542466fe1b7643876`; scripts later committed as `038e4977be4eb3f11d4d3528da82af29f10e4116` | `3322011` | Local archive listed below | No trained-model partition evaluation | `verified-local` | [Record](model_data_pytorch_validation.md) |
| B0 CPU/CUDA infrastructure validation | Dirty tree based on `73349b475069fd68acd7940f8fde90dadb37e7eb`; intended code later committed as `19a18591e1799e51447af4f31fbb0510761af444` | `3322236`, `3322251` | Adroit paths in record | Bounded fixtures only; not a scientific split result | `documented-external` | [Record](b0_training_infrastructure_validation.md) |
| 680-family controlled corpus | `19a18591e1799e51447af4f31fbb0510761af444` | `3322880` | `/scratch/network/km6349/controlled_corpora/b0-pilot-680-seed2026` | 544 train / 68 validation / 68 test; no test evaluation | `documented-external` | [Record](b0_pilot_corpus_680.md) |
| Original meaningful B0 training | `81bf70d05aa81c1eb5143c1745c532adb0783bad` | `3323110` | Frozen archive below | 544 train / 68 validation / 0 test evaluated | `verified-local` | [Record](b0_original_training_collapse.md) |
| Phase A state/checkpoint contract | `08985a39dc6e7cdb5f2af96e5f24fd9139df0f10`; comparison base `ba611fbe628ac76a7a1449a30dd388c52b72ef11` | `3323988` | Frozen archive below | No dataset evaluation | `verified-local` | [Record](b0_phase_a_contract_validation.md) |
| Phase B paired evaluator | `42183a47ec8fa70e6109fbea33ecda9e5e84d071` | `3324856` | Frozen archive below | 68 validation / 0 test; original collapsed checkpoint | `verified-local` | [Record](b0_phase_b_validation_evaluation.md) |
| VQ-collapse diagnosis | `55b2841d17304abfff793de6c2cece13205bce3c` | `3325296` | Frozen archive below | Two diagnostic families; test false | `verified-local` | [Record](b0_vq_collapse_diagnosis.md) |
| Train-k-means pilot | `856d6a62187f3d7f4b5cc7a9b4c9efc655b74ce1` | `3326278` | Frozen archive below | 544 train / 68 validation; test false | `verified-local` | [Record](b0_train_kmeans_pilot.md) |
| Full train-k-means retraining | `d549ebe4478e166fda8d54f2b173572a1ab52e7a` | `3326757` | Frozen archive below | 544 train / 68 validation; test false | `verified-local` | [Record](b0_train_kmeans_full_retrain.md) |
| Repaired-checkpoint Phase B smoke | `ea4f1e152195c209042c8f713c62be6d7393b3bb` | `3327631` | Two local publications listed below | 6 validation / 0 train / 0 test evaluated | `verified-local` | [Record](b0_phase_b_repaired_smoke.md) |
| Repaired-checkpoint Phase B full validation | Evaluation `ce4abca8f450745a8832c0189aaae4a7d8263ec0`; recovery implementation `b00ac0d1a645ae9559e269ed93f5ee7940a0512c` | `3329040`, recovered after post-validation failure | Local immutable bundle listed below | 68 validation / 0 train / 0 test evaluated; test false | `verified-local` | [Record](b0_phase_b_repaired_full_validation.md) |
| Phase B constraint-manifold replay | `e0829773589a07ab471bc3932bf91578107b5467` | Local diagnostic over job `3329040` | Local immutable replay listed below | 68 validation families; diagnostic only; zero test payload access | `verified-local` | [Record](b0_phase_b_constraint_manifold_replay.md) |
| Phase B categorical-isolation replay | `60325f9a3872bd82c783c687b68a7023bccc2d1b` | Local diagnostic over job `3329040` | Local immutable replay listed below | 68 validation families, 544 factorial rows; diagnostic only; zero test payload access | `verified-local` | [Record](b0_phase_b_categorical_isolation_replay.md) |
| Constrained Flat V2-V5 progression | V2 `9e66231`; V3 `0127690`; V4 `98aa965`; V5 `d1eae8f` | accepted pilots `3333190`, `3334371`, `3337600`, `3338093`; failed regressions `3334551`, `3336430`, `3337640` | Complete recovery archive listed below | 544 train / 68 IID validation; systematic/test false | `verified-local` | [Record](b0_constrained_flat_v2_v5_progression.md) |
| Frozen constrained Flat V6 | `ac6ef718ae9bab7fa5a80d9f48d0976adf5cafad` | failed validation `3338478`; passed validation `3338602`; pilot `3338639` | Complete recovery archive; frozen Git tag local | 544 train / 68 IID validation; systematic/test false | `verified-local` | [Record](b0_constrained_flat_v6.md) |
| Graph V1 engineering readiness and C1 validation | Initial series `6484579`-`089b9f3`; C1 `0cd09ed` | failures `3338803`, `3338863`, `3338945`, `3339188`, `3339644`, `3339787`, `3340201`, `3341971`; passes `3339745`, `3341940`, `3341972` | Scheduler evidence and incomplete pilot namespace in complete recovery archive | Bounded fixtures plus one incomplete train epoch; systematic/test false | `verified-local` | [Record](b0_graph_v1_engineering_readiness.md) |
| Initial Graph V1 | `089b9f3d0e5a61fb19ef3fa05e993fc4eceffdcb` | `3341942` | Complete recovery archive plus lightweight metrics/log archive | 544 train / 68 IID validation; systematic/test false | `verified-local` | [Record](b0_graph_v1_initial.md) |
| Graph V1 directed-position C1 | `0cd09ed34d4c4dd0d43e1456b7a06eb362ee7962` | validation `3341972`; pilot `3341974` | Complete recovery archive plus lightweight metrics/log archive | 544 train / 68 IID validation; systematic/test false | `verified-local` | [Record](b0_graph_v1_c1.md) |
| GE1 C3 authoritative CPU compatibility validation | `a85ad3a23a6587cbedad8a6693b6117c1edfacc7` | `3344235` | Local stdout/stderr and preserved runner listed in record | Fixture-only validation; authoritative manifest and CAD-history payloads not opened; systematic/test false | `verified-local` | [Record](ge1_c3_cpu_validation.md) |
| GE1 C4 authoritative CPU validation | `e66cd089462c2e075b9ff742e157c21e4d9a2a6e` | Interactive `srun`; ID not captured | Local terminal transcript and complete-history Git bundle listed in record | Procedural and temporary fixture validation only; authoritative corpus payloads not opened; systematic/test false | `verified-local` | [Record](ge1_c4_cpu_validation.md) |
| GE1 C4 review-fix authoritative CPU validation | `3a41abc81f68ef6d6450465e05b3544f07a08b83` | `3344265` | Local stdout/stderr and complete-history Git bundle listed in record | Fixture-only validation; authoritative manifest and CAD-history payloads not opened; systematic/test false | `verified-local` | [Record](ge1_c4_review_fix_cpu_validation.md) |

### Current-through-August-3 coverage

The rows above cover the complete material sequence in the August 3
consolidated project record:

- the failed epoch-44 evaluation and its two diagnostic replays;
- the shared constrained-profile foundation and Flat V2-V5 staged repair;
- V6 readiness and the frozen Flat V6 pilot;
- every named Graph V1 readiness/provenance failure and repair;
- the initial Graph V1 scientific pilot;
- failed and corrected C1 authoritative validation; and
- the final C1 pilot, freeze decision, protected-partition state, and local
  metrics/log archives.

Implementation-only commits that did not create an external run remain in Git
history and package contracts rather than receiving artificial inventory rows.
Intermediate job identifiers preserved only as focused-test regression names
are labeled as such and are not promoted to accepted scientific results.

## Locally verified bundles

### Counterfactual OpenCascade audit

| Item | Verified value |
|---|---|
| Archive | `/Users/krishaymaskara/research/audited-runs/counterfactual-audit-3ae4f6c-job3321612.tar.gz` |
| Size | 482,536 bytes |
| SHA-256 | `527d14811fa4238ee84d4f318f37d956dbc4bf31ec7051ad90644167ceca9fa0` |
| Source commit | `3ae4f6cf96b22c6ef33f592dc4e6e47020e059c6` |
| Primary job | `3321612` |
| Independent replay | `3321613` |
| Python 3.8 test job | `3321607`, 154/154 reported in the bundled validation note |

The gzip/tar archive extracted successfully. Its canonical manifests and
reports parsed successfully. The primary and replay stderr files are empty,
and their stdout files identify the same source commit.

The bundle directly verifies:

- 68 edit families and 136 edit samples;
- 136 physical endpoints represented by 272 continuous/quantized histories;
- 272/272 successful encoded endpoint executions;
- 136/136 successful edit samples;
- 136/136 continuous/quantized endpoint agreements;
- PythonOCC `7.5.1`;
- generation seed 17;
- configuration SHA-256
  `cf9a16d315e84a074dd4c671198620853bab61f1733b35dba8fd097a015e2f79`.

The bundled validation note reports that job `3321612` and replay job
`3321613` were compared with `diff -qr` and were byte-identical. That claim is
supported by the note and both bundled logs; only the primary generated tree
is retained in this archive, so the comparison cannot be independently
repeated from this archive alone.

### Real-PyTorch model-data validation

| Item | Verified value |
|---|---|
| Archive | `/Users/krishaymaskara/research/audited-runs/model-data-pytorch-038e497-job3322011.tar.gz` |
| Size | 8,931 bytes |
| SHA-256 | `43e5ca38eaf1f687fd4ab6b2da8d08dc872119e7243712dc68418aa15f8c8012` |
| Validated source | `91d733d56764beac8be06aa542466fe1b7643876` |
| Validation scripts | Later committed exactly as `038e4977be4eb3f11d4d3528da82af29f10e4116` |
| Job | `3322011`, `COMPLETED 0:0` |

The gzip/tar archive extracted successfully. It contains the Slurm script,
validation script, stdout, stderr, and a compact validation summary. The
stderr transcript ends with 193 tests passing in 704.924 seconds. Stdout
records:

- Python 3.8.13;
- PyTorch 1.11.0 with CUDA build 11.3;
- CPU-only execution;
- real model-data tensor validation passed;
- execution from `/scratch/network/km6349/SkexGen`;
- start `2026-07-23T22:27:25-04:00` and finish
  `2026-07-23T22:39:18-04:00`.

The repository revision was `91d733d`, while the two validation files were
staged, so the job's working tree was not clean. The archive states that those
exact staged contents were subsequently committed as `038e497`; both commits
exist in the current Git history. This is valid evidence for the tested source
plus the archived validation scripts, not a claim that job `3322011` ran from
a clean checkout of `038e497`.

## Frozen flat-baseline evidence archive

The following namespaces were copied with publication symlinks dereferenced:

| Job/evidence | Expected run or report | Expected scheduler logs |
|---|---|---|
| `3323110` original B0 | `/scratch/network/km6349/flat_baseline_runs/b0-pilot-680-k32-lr1e3-seed2026` | Not recovered from repository |
| `3323988` Phase A contract | `/scratch/network/km6349/flat_baseline_runs/validation/phase-a-08985a3-contract-3323988` | `/scratch/network/km6349/flat_baseline_runs/phase-a-contract-3323988.{out,err}` |
| `3324856` Phase B evaluator | `/scratch/network/km6349/flat_baseline_evaluations/validation-42183a47ec8fa70e6109fbea33ecda9e5e84d071-3324856`, two smoke runs, and `validation-report-3324856.json` | `/scratch/network/km6349/phase-b-evaluation-3324856.{out,err}` |
| `3325296` diagnosis | `/scratch/network/km6349/flat_baseline_vq_diagnostics/vq-diagnosis-55b2841d17304abfff793de6c2cece13205bce3c-3325296` and `vq-diagnosis-report-3325296.json` | `/scratch/network/km6349/phase-b-vq-diagnosis-3325296.{out,err}` |
| `3326278` pilot | `/scratch/network/km6349/flat_baseline_runs/train-kmeans-pilot-856d6a62187f3d7f4b5cc7a9b4c9efc655b74ce1-3326278` | `/scratch/network/km6349/flat-vq-kmeans-pilot-3326278.{out,err}` |
| `3326757` full retrain | `/scratch/network/km6349/flat_baseline_runs/train-kmeans-full-d549ebe4478e166fda8d54f2b173572a1ab52e7a-3326757` | `/scratch/network/km6349/flat-vq-kmeans-full-3326757.{out,err}` |

The archived scheduler transcript reports all six jobs as `COMPLETED`
with exit code `0:0`. It records:

| Job | Elapsed | Node |
|---:|---:|---|
| `3323110` | 3:46 | `adroit-h11g3` |
| `3323988` | 0:12 | `adroit-h11n2` |
| `3324856` | 1:51 | `adroit-h11n3` |
| `3325296` | 2:03 | `adroit-h11n3` |
| `3326278` | 1:49 | `adroit-h11n2` |
| `3326757` | 5:34 | `adroit-h11n2` |

Archive:
`/Users/krishaymaskara/research/audited-runs/skexgen-flat-evidence-20260728.tar.gz`

SHA-256:
`9eaeeb91728d0e2c8e8a7de77a0ac837e92a9c3fe7c704c6011ee26fbbae0498`

The hash matched on Adroit and macOS. The archive extracted without unsafe
paths, and all 91 files covered by its internal `SHA256SUMS` passed.

Future recovered jobs should retain the same minimum evidence set:

1. `sacct` state, exit code, elapsed time, and node;
2. stdout and stderr;
3. resolved public paths and symlink targets;
4. run configuration and source/working-tree provenance;
5. partition provenance and explicit test-use flag;
6. decision or final report;
7. metrics or diagnostic summaries;
8. artifact manifest and verification result;
9. SHA-256 for the frozen archive and important checkpoints.

Checkpoint and corpus archives should remain outside Git. Their experiment
records should retain stable archival paths and hashes.

## Locally verified repaired Phase B smoke

Job `3327631` published two artifact directories under
`/Users/krishaymaskara/research/audited-runs/phase-b-smoke/`, suffixed `-a`
and `-b`. Each contains `conversion_failures.csv`, `examples.jsonl`,
`metrics.csv`, `raw_predictions.jsonl`, `run_metadata.json`, and
`summary.json`.

All six direct file comparisons passed. The corresponding SHA-256 values
matched, and each ordered six-file hash manifest had SHA-256
`726d457f28549089f7c991c572ed6f90bc1c234a91910f26fe46ae7afcd3c88e`.
The metadata verifies a clean `ea4f1e1...` source, the selected epoch-44
checkpoint, six validation records loaded, and zero train/test records
loaded. The deterministic smoke passed; it did not evaluate the remaining
62 validation families or establish final validation quality. See the
[immutable smoke record](b0_phase_b_repaired_smoke.md).

## Locally verified repaired Phase B validation and diagnostic replays

Job `3329040` completed inference and complete artifact validation for all 68
validation families, then exited `FAILED 1:0` during post-validation manifest
construction. A reviewed manifest-only recovery independently revalidated
the existing namespace and replaced only its incomplete nine-entry manifest
with a verified 15-entry stable-path manifest. The recovered report records
five active codes and perplexity `3.120410089936484`, but zero
controlled-domain-valid families on both decoding paths and zero qualifying
extrude/revolve families. Gate E was not run and formal final acceptance
remains undetermined. See the
[immutable full-validation record](b0_phase_b_repaired_full_validation.md).

The flattened recovered namespace is now present locally at:

```text
/Users/krishaymaskara/research/audited-runs/phase-b-full-validation/
  phase-b-repaired-validation-ce4abca8f450745a8832c0189aaae4a7d8263ec0-3329040
```

Its verified 15-entry manifest has SHA-256
`a13d5f35b345b6814912145a56e0cce0290ae677b030a1215b61e92558862541`.
This upgrades the primary bundle to `verified-local`; the original scheduler
state remains `FAILED 1:0`.

Two immutable read-only diagnostic namespaces are also locally verified:

| Diagnostic | Implementation | Manifest SHA-256 |
|---|---|---|
| Constraint-manifold replay | `e0829773589a07ab471bc3932bf91578107b5467` | `6ff46389fb3947b3448e5e52860833d0592e83f70849536ddfd4aeb32f73b4b2` |
| Categorical-isolation replay | `60325f9a3872bd82c783c687b68a7023bccc2d1b` | `e9c353a98fd37f5a491c34970d1859c73d61ebb7341d1ac14ca020a6f51aea63` |

The categorical-isolation manifest covers six verified artifacts, including
exactly 544 factorial JSONL rows. It isolates zero plane-only gain, profile
gains of +32 teacher-forced and +34 predicted-history, and zero interaction.
See the
[immutable categorical-isolation record](b0_phase_b_categorical_isolation_replay.md).

## Locally verified constrained-decoder primary evidence

The original Flat V2-V6 and Graph V1/C1 namespaces were recovered from
Adroit into:

```text
/Users/krishaymaskara/research/audited-runs/
  skexgen-constrained-decoder-primary-evidence-20260803.tar.gz
```

Its SHA-256 is
`5bd60fd884b23d607f452d82e8da1e71c97a119e9d8ad92ef0f3d91dbc4f2dcc`.
The hash matched before and after transfer. Every file passes the archive's
internal `SHA256SUMS`.

The archive contains scheduler accounting and stdout/stderr for the complete
Flat V1-V6/Graph V1 engineering sequence selected for recovery; the original
V2 diagnostics; accepted V2-V6 pilot metrics and checkpoints; failed V4/V5
regression namespaces; the incomplete Graph job `3339787`; both complete
Graph scientific namespaces; and frozen-tag provenance. It does not contain
separate resolved environment files or original per-run manifests.

Accepted epoch-2 checkpoint hashes are:

| Run | Job | SHA-256 |
|---|---:|---|
| Flat V2 | `3333190` | `516c480ad7477bd93d14ece7247057a831760a7399552845def2ac2846eb85b7` |
| Flat V3 | `3334371` | `de3f289fc8999d0c07da996c015ecb4061412b9f9cb80cbf0f770404d3f6f365` |
| Flat V4 | `3337600` | `88633c8c2bebdaf3d50c147aac2b0680e8db58905caded4117aba1fe25cd961b` |
| Flat V5 | `3338093` | `02d3daf74735e44e997084339e2f942eb1d14d21cc28b230f559b8f26de82be4` |
| Flat V6 | `3338639` | `aa9cd5158eea3e003a02a8482e4946e025d5d91284c3d14f10a86be3495fd35b` |
| Initial Graph V1 | `3341942` | `2fe26a4d88153a49bb70946c8b91b8c51fe37aca608c2f6c67ef11a7c6936d80` |
| Graph V1 C1 | `3341974` | `3afe12e3c761b7ecb81ae0eeac04d39ea445d4568553331607be4c489c8a96d9` |

## Lightweight Graph V1 metrics and scheduler bundles

| Run | Archive | SHA-256 | Inspected contents |
|---|---|---|---|
| Initial Graph V1, job `3341942` | `/Users/krishaymaskara/Downloads/graph-v1-3341942-results.tar.gz` | `b7487bc32c8e9bdc5f436fddafd902f66bc5907364192784edfe7f132112cb79` | `metrics.jsonl`, stdout, stderr |
| Graph V1 C1, job `3341974` | `/Users/krishaymaskara/Downloads/graph-v1-c1-3341974-results.tar.gz` | `aa3c4cdabb66a8160d358662d23be57183db06b8341b7613232e6cdf6d4a8178` | `metrics.jsonl`, stdout, stderr |

Both archive hashes match
`/Users/krishaymaskara/Downloads/graph-v1-final-bundle-checksums.txt`.
The JSONL files parse and independently verify terminal success, epoch-2
selection, zero systematic/test access, 22/68 exact and complete validity,
the single- versus two-operation split, edge metrics, prior comparisons, and
checkpoint hash metadata.

These remain convenient partial result bundles. The complete recovery archive
above now supplies the original checkpoint binaries and an internal per-file
checksum list. Separate resolved environment files and original per-run
manifests remain unavailable.

## Record backlog

Every completed run with recovered local evidence now has a durable
experiment record. The repaired-checkpoint full-validation bundle and both
diagnostic replays are locally manifest-verified. The Flat V2-V6 and Graph
V1/C1 metrics, checkpoints, and scheduler evidence are locally checksum-
verified in the complete constrained-decoder archive. Remaining reproducibility
gaps are limited to original per-run manifests and separate resolved
environment files that were not present in those namespaces.
