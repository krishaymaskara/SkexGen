# B0 Pilot Controlled Corpus: 680 Physical Families

## Purpose and provenance

This record documents the first bounded pilot corpus generated for the
`B0-FLAT-MIXED-VQ` milestone.

| Item | Verified value |
|---|---|
| Branch | `flat-mixed-baseline` |
| Generated from commit | `19a18591e1799e51447af4f31fbb0510761af444` |
| Commit description | `19a1859 Add flat baseline training pipeline` |
| Slurm job | `3322880` |
| Job result | `COMPLETED`, exit code `0:0` |
| Elapsed time | 24 seconds |

## Corpus configuration

The corpus was generated at:

```text
/scratch/network/km6349/controlled_corpora/b0-pilot-680-seed2026
```

| Configuration item | Verified value |
|---|---:|
| Generation seed | 2026 |
| Physical source families | 680 |
| Continuous/quantized sample variants | 1,360 |
| IID training families | 544 |
| IID validation families | 68 |
| IID test families | 68 |
| Reported size | 17 MB |
| File count | 1,365 |

Each physical family has one continuous and one quantized representation
variant. Split membership remains authoritative at the physical-family level.

## Integrity result

The verified corpus-tree SHA-256 was:

```text
f71425c589f9520cef0c671bf114258c2a4d0801e3e6ffc90c5bbe255f2e0671
```

The final job marker was:

```text
ALL B0 PILOT CORPUS GENERATION CHECKS PASSED
```

## Reproducible generator invocation

The generator CLI and all arguments needed to request this corpus can be
reconstructed from the repository:

```bash
python3 -m prototype.controlled_data.generate \
  --output-dir /scratch/network/km6349/controlled_corpora/b0-pilot-680-seed2026 \
  --seed 2026 \
  --num-source-families 680
```

This command must be run from the repository revision recorded above to target
the documented generator implementation. The exact Slurm submission wrapper
is not included because it is not available in the repository documentation.

## Use and limitations

- This is a bounded first pilot corpus and is not necessarily publication
  scale.
- IID test families must not be used for training or checkpoint selection.
- Counterfactual evaluation endpoints were not included.
- The corpus record does not claim model generalization or model quality.

Raw scheduler logs remain outside the repository:

```text
/scratch/network/km6349/controlled_corpus_runs/b0-corpus680-3322880.out
/scratch/network/km6349/controlled_corpus_runs/b0-corpus680-3322880.err
```

The generated corpus, raw logs, caches, and other runtime artifacts are
intentionally not copied into Git.
