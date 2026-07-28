# B0 Phase B Repaired-Checkpoint Deterministic Smoke

## Record status

| Item | Value |
|---|---|
| Experiment ID | `b0-phase-b-repaired-smoke-3327631` |
| Record status | `verified` |
| Scientific role | Deterministic evaluator smoke |
| Source commit | `ea4f1e152195c209042c8f713c62be6d7393b3bb` |
| Working tree | Clean |
| Slurm job | `3327631` |
| Execution date | July 28, 2026 |
| Scheduler result | Not preserved in the downloaded artifact set |

## Question and decision rule

This bounded run asked whether the repaired Phase B evaluator could load the
selected repaired checkpoint, resolve a fixed validation-only subset before
inference, publish the complete artifact set twice, and reproduce every
published byte. The deterministic smoke passed if both publications
completed with the repaired smoke/evaluation contracts enabled and all six
corresponding artifact files were byte-identical.

This was an engineering and determinism gate only. It was not the
predetermined 68-family validation evaluation and could not establish final
reconstruction quality.

## Inputs and partition authority

| Item | Verified value |
|---|---|
| Checkpoint | Repaired B0 `best.pt`, epoch 44, global step 748 |
| Checkpoint SHA-256 | `282988af00a2dc9a53e14ceb270537d35f85d5339dc989634f88af5f77931267` |
| Checkpoint training source | `d549ebe4478e166fda8d54f2b173572a1ab52e7a` |
| Corpus configuration SHA-256 | `57f11d3024b97c80eb6052d3517a097a88dc136d0befa103f2dc47fb62516d32` |
| Corpus manifest SHA-256 | `3be4bb2d03e0500ae6e5b5a878cfaafcc8c3a74180a5b9bc2cfc55071b21e2ff` |
| Split manifest / SHA-256 | `iid` / `8d6caf0d5584c818e90c9e5d877ab9dd00519cb343f7fabe90a6459802914ec3` |
| Authoritative partition counts | 544 train / 68 validation / 68 test |
| Authoritative validation-ID SHA-256 | `31b6607ca360bf7f3ed2334ca798f3aa07dd70214192a13324b71439d56c0681` |
| Train families loaded/evaluated | 0 / 0 |
| Validation families loaded/evaluated | 6 / 6 |
| Test families loaded/evaluated | 0 / 0 |
| Test partition evaluated | `false` |

The authoritative loader succeeded and validated the corpus identity. The
six selected IDs were resolved before inference, match the loaded IDs, and
are the first six IDs in the authoritative ordered validation list:

1. `sf_0315dbbd0c69ebdc80097404b37b0e0484c6f4d14a79e1e1cf5aa4b788b4beb3`
2. `sf_0c1844c00b6fafb0a1493eeff3fb48bde8bbba151590173b520175aeea159f36`
3. `sf_0ee68f7559ece4d1c4d5bbc7e835964f03c9e79923098af599bc981b0389ef18`
4. `sf_1472029af69ec20a0428f622c718c99938aa4f6a2e3a0414d1d6bc06c304ad8c`
5. `sf_1981f3acc573dad74ef3518491051e030f20f9eb2150b5cba48c00578109ff03`
6. `sf_22c08141bc66bc0afa3a80baae4ce818391ac7233888bb20254b6d1df58acdf9`

No test families were evaluated.

## Configuration and environment

Both decoding paths used the same reconstructed latent memory. The
teacher-forced path used shifted-target-prefix feedback; predicted history
used raw argmax feedback with a derived geometry mask. Target canonical node
count was supplied. The family limit was 6, evaluation batch size was 3, and
the requested and resolved device were CPU.

The metadata records evaluation schema version 2, representation schema
version 1, source-tree SHA-256
`17c8ddbd4ace120927f9f0039c3af7cccd562d140efd4334ebe46fc3344b8ba7`,
and `source_dirty: false`. Python, PyTorch, CUDA, node, elapsed time, and
scheduler state were not preserved in the downloaded artifact set.

## Verified results

Both publications contain six raw-prediction records, six per-example
records, metrics, controlled-domain failures, metadata, and a summary.

| Metric | Teacher-forced | Predicted history |
|---|---:|---:|
| Attempted families | 6 | 6 |
| Raw completion rate | 1.0000 | 1.0000 |
| Raw integrity validity | 1.0000 | 1.0000 |
| Reconstruction-target validity | 1.0000 | 1.0000 |
| Controlled-domain validity | 0.0000 | 0.0000 |
| Node-type token accuracy | 1.0000 | 1.0000 |
| Exact node-type sequence | 1.0000 | 1.0000 |
| Exact complete ten-field match | 0.0000 | 0.0000 |
| Geometry MAE | 2.7998 | 2.7766 |
| Geometry RMSE | 18.2511 | 18.0970 |
| Exact operation-type sequence | 1.0000 | 1.0000 |
| Pointer accuracy | 1.0000 | 1.0000 |
| Edge micro-F1 | 1.0000 | 1.0000 |
| Exact typed-edge set | 1.0000 | 1.0000 |

The shared latent assignments used 4 of 32 codes, with perplexity 2.9304
and utilization 0.125 across 12 assignments. Controlled-domain failures
included unsupported profile patterns, invalid reference-plane geometry,
invalid axis geometry, and invalid primitive geometry.

## Decision and interpretation

The deterministic smoke passed. Direct comparison of the downloaded files
verified that both publications were byte-identical: all six corresponding
files have the same SHA-256 in publication `a` and publication `b`.

This result supports clean-checkout provenance, correct validation-only
payload access, repaired-contract activation, complete publication, and
deterministic repetition for this fixed six-family smoke. The metric values
are smoke diagnostics, not an acceptance or rejection decision for the
repaired checkpoint.

## Limitations and claims not supported

- Only 6 of the 68 validation families were evaluated.
- No train or test families were evaluated.
- The smoke subset is the first six authoritative validation IDs, not a
  full validation evaluation or a separately randomized sample.
- Target canonical node count was supplied and partially reveals the target
  template; these are not autonomous template-classification results.
- No predicted history was executed through OpenCascade.
- The downloaded set contains no scheduler logs, environment transcript, or
  frozen archive; job completion state and software versions cannot be
  independently verified from these files.
- Zero controlled-domain validity in six smoke families must not be
  generalized to all 68 validation families.
- The full 68-family validation evaluation remains the next gate.

## Artifacts and integrity

Both publication directories are under:
`/Users/krishaymaskara/research/audited-runs/phase-b-smoke/`.

| Artifact | SHA-256 in both publications |
|---|---|
| `conversion_failures.csv` | `5bdb22de3c1d6f09135fa37cc822950b7026968d0807c17e91bdc2e508d45d5b` |
| `examples.jsonl` | `67cefe08c0bfba608f433e5017f9c00dac19d252d2c66892c9050d9573fdfda1` |
| `metrics.csv` | `3060b7faa4c89790b9a4cb10da8e009dd6aab003df3c39aeb26606416e937e08` |
| `raw_predictions.jsonl` | `b955d2ddf10a9ce74a90e25b84c0432be945e5d9034b276e51b3989b95a8c7d9` |
| `run_metadata.json` | `7779169d8a6b2abf006c4e2e7bab0644c9797afaaf1c92d33c7d34b702dc7088` |
| `summary.json` | `c5283c307a49f1612d4166226040a4a76a91f9f13a344f1abc6ab9ba9d339a59` |
| Ordered six-file SHA-256 manifest | `726d457f28549089f7c991c572ed6f90bc1c234a91910f26fe46ae7afcd3c88e` |

The ordered manifest hash was computed from the six filename-and-SHA-256
lines in the table's order. It matched for both directories. Direct `cmp`
checks also passed for all six corresponding files.

## Reproduction and validation

No inference was rerun. Validation was limited to parsing the preserved JSON,
JSONL, and CSV files; checking partition-access metadata and selected IDs;
hashing every artifact; and directly comparing the two publications. The
publications were byte-identical, not merely numerically equivalent.

## Related records

- Checkpoint source: [Full train-k-means retraining](b0_train_kmeans_full_retrain.md)
- Evaluator predecessor: [Original-checkpoint Phase B evaluation](b0_phase_b_validation_evaluation.md)

