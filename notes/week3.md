# Week 3 Progress Report

## B0-FLAT-MIXED-VQ milestone

The Adroit CPU and CUDA validation jobs ran from a dirty
`flat-mixed-baseline` working tree based on commit
`73349b475069fd68acd7940f8fde90dadb37e7eb`. The intended production-source
changes validated by those jobs were subsequently committed as
`19a18591e1799e51447af4f31fbb0510761af444`
(`19a1859 Add flat baseline training pipeline`). These jobs therefore record
dirty-tree validation preceding the canonical commit, not clean-checkout
validation of `19a1859`.

CPU job `3322236` passed all 63 flat-baseline and 39 model-data tests. CUDA job
`3322251` verified forward/backward training, EMA VQ, checkpointing,
relocation, and resume on an NVIDIA A100 80 GB PCIe GPU.

After synchronization from commit `19a1859`, job `3322880` generated a
deterministic pilot corpus containing 680 physical families, 1,360
continuous/quantized variants, and exact IID family counts of 544 training,
68 validation, and 68 test.

Durable validation details and limitations are recorded in:

- [B0 training-infrastructure validation](../docs/experiments/b0_training_infrastructure_validation.md)
- [B0 680-family pilot corpus](../docs/experiments/b0_pilot_corpus_680.md)
