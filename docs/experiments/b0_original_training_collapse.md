# Original B0 Training and One-Code Collapse

## Record status

| Item | Value |
|---|---|
| Experiment ID | `b0-original-3323110` |
| Record status | `verified` |
| Scientific role | First meaningful flat/mixed VQ training run |
| Branch | `flat-mixed-baseline` |
| Source commit | `81bf70d05aa81c1eb5143c1745c532adb0783bad` |
| Working tree | Clean |
| Slurm job | `3323110` |
| Execution date | July 26, 2026 |
| Scheduler result | `COMPLETED`, exit `0:0`, 3:46 on `adroit-h11g3` |

## Question and decision rule

The run asked whether the implemented `B0-FLAT-MIXED-VQ` model could train on
the 680-family corpus and learn a nondegenerate discrete representation.
Infrastructure completion and decreasing loss were necessary but not
sufficient: active-code count and perplexity had to remain meaningfully above
the one-code state.

## Inputs and partitions

| Item | Verified value |
|---|---|
| Corpus | `/scratch/network/km6349/controlled_corpora/b0-pilot-680-seed2026` |
| Corpus configuration SHA-256 | `57f11d3024b97c80eb6052d3517a097a88dc136d0befa103f2dc47fb62516d32` |
| Corpus manifest SHA-256 | `3be4bb2d03e0500ae6e5b5a878cfaafcc8c3a74180a5b9bc2cfc55071b21e2ff` |
| Split | IID |
| Train families | 544 |
| Validation families | 68 |
| Test families evaluated | 0 |

The run metadata contains the complete train and validation family-ID lists
and contains no test-family list.

## Configuration and environment

The model had 200,705 trainable parameters, model width 64, two encoder and
two decoder layers, two latent tokens, 32 codebook entries of width 32, and a
maximum of nine nodes and two operations. Training used batch size 32,
learning rate 0.001, weight decay 0.0001, gradient clipping at 1.0, seed 2026,
and 50 epochs. Validation and checkpointing occurred every epoch, with
validation total loss as the checkpoint-selection metric.

The run used Python 3.8.13, PyTorch 1.11.0, and a Tesla V100-PCIE-32GB.
Recorded source-tree SHA-256:
`459a9152e5459e6398861e5bb106e59789b14dfbbd9b8fabfecf43f905f44a7d`.

## Verified results

The job completed all 50 epochs and 850 optimizer steps. The selected
`best.pt` was epoch 50 because it had the minimum validation total loss:

| Metric | Epoch 50 validation |
|---|---:|
| Total loss | 0.4497575151 |
| Active codes | 1 of 32 |
| Codebook perplexity | 1.0 |
| Codebook utilization | 0.03125 |

Training briefly touched five codes in epoch 1, with perplexity 1.1234.
Validation had already collapsed to one code in epoch 1; training and
validation then remained at one code.

## Decision and interpretation

The run passed operational execution but failed the scientific
representation gate through total one-code assignment collapse. Low
teacher-forced validation loss did not demonstrate a useful latent
representation.

The later Phase B evaluation confirmed that every one of the 136 validation
latent assignments selected code 17 and that neither teacher-forced nor
predicted-history reconstruction produced a controlled-domain-valid example.

## Limitations and unsupported claims

- This run does not establish autonomous reconstruction, executable CAD
  validity, systematic generalization, or localized editing.
- No held-out test family was evaluated.
- The training artifacts do not by themselves identify why collapse occurred;
  that question is addressed by the separate diagnosis record.

## Artifacts and integrity

| Artifact | SHA-256 |
|---|---|
| `best.pt` | `2732899de6d60d407d26a7cf176e7a7f4534580aa99ca973df6e9b740330222b` |
| `last.pt` | `aa99885f4120d634ef5990eee11fa4dfe1ab40704fd74ed1c0a527b8d7a145e9` |
| Frozen multi-run archive | `9eaeeb91728d0e2c8e8a7de77a0ac837e92a9c3fe7c704c6011ee26fbbae0498` |

The frozen archive is stored at
`/Users/krishaymaskara/research/audited-runs/skexgen-flat-evidence-20260728.tar.gz`.
It transferred with the same SHA-256 on Adroit and macOS, extracted
successfully, and passed all 91 checks in its internal `SHA256SUMS`.

## Related records

- Corpus: [B0 680-family pilot corpus](b0_pilot_corpus_680.md)
- Compatibility: [Phase A contract validation](b0_phase_a_contract_validation.md)
- Evaluation: [Phase B validation evaluation](b0_phase_b_validation_evaluation.md)
- Diagnosis: [VQ-collapse diagnosis](b0_vq_collapse_diagnosis.md)

