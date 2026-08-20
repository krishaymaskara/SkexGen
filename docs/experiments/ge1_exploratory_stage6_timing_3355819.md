# GE1 Exploratory Stage 6 Hardware Re-Timing, Job 3355819

## Disposition

Exact-commit timing-v2 **passed** and retained the prior resource choice:
`cpu` with all three seeds `2026`, `2027`, and `2028`. Device selection and
seed selection therefore did not change from the earlier CPU/three-seed plan.

| Item | Verified value |
|---|---|
| Source commit | `42d5c489c7236031ff18dfd13ca88af428ca7510` |
| Source-tree SHA-256 | `5a58941742cbcb0b51a044d27f2418f19fcfcccc4ba5d750c921d0d410246fbd` |
| Job / state / exit | `3355819` / `COMPLETED` / `0:0` |
| Elapsed / batch MaxRSS | `00:09:21` / `6206380K` |
| Allocation | one A100 80GB GPU allocation, one CPU thread, 16GB RAM |
| Runner SHA-256 | `d2f8e8a2954a82fdad2852e35b057d2a66d7559c27532d88957956b73cb84e47` |
| Magnitude / node identity | `GE1-OPERATION-MAGNITUDE-GRID-ORDINAL-v1` / `GE1-PAD-TERMINATED-UNCONSTRAINED-NODES-v1` |
| Selection | CPU, three seeds, no fallback |
| Selection reason | `fastest feasible three-seed contingent projection` |

## Candidate measurements and projection

Both candidates used a 14,400-second prospective envelope. Each conservative
rate is the maximum of three fresh five-epoch measurements after a separate
fresh warm-up.

| Candidate | Flat sec/epoch | Typed-graph sec/epoch | Three-seed projection + 20% | Feasible |
|---|---:|---:|---:|---|
| CPU | `4.565040245838463` | `5.052053121756762` | `6924.3072246685615` s | `true` |
| `cuda:0` | `7.797342937812209` | `9.225434354227037` | `12256.399650268255` s | `true` |

CPU was the faster feasible three-seed candidate. The timing job accessed only
the governed narrow train package; development and scientific outcomes were
not available and did not influence selection.

## Artifact

Finalized timing artifact:

`/scratch/network/km6349/ge1_stage6_timing-42d5c48/ge1-stage6-hardware-timing-42d5c489c7236031ff18dfd13ca88af428ca7510-3355819`

| File | SHA-256 |
|---|---|
| `SHA256SUMS` | `b07c1c75f009d85f4c55d3f9b78f0c95fa9b113853ea7b6b6cc7e3665c7dba52` |
| `artifact_manifest.json` | `3f10ba3ed6830dc4792bb0d566cfad3dcbd12fb02693e6feed66fa7d9b3738fb` |
| `timing.json` | `aef5d84f448080e4b421acdbdb960462fd1b617ff5c133e0de3b4e92ace66ee0` |
| stdout | `f35ac2e58b381227341712f309c9cc17980176a0a0a1ef154a3a32f7b2d0cf7b` |
| stderr | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |

The artifact and its checksums passed the committed verifier.

## Superseded attempt

Job `3355818` failed in one second before train access because its submission
omitted the explicit `REPOSITORY` export and the runner's historical default
checkout did not match the expected commit. Job `3355819` supplied the clean
exploratory checkout explicitly; no source or protocol change separated the
attempts.
