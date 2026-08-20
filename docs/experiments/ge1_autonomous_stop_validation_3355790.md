# GE1 Autonomous-Stop Authoritative CPU Validation

## Record status

| Item | Value |
|---|---|
| Experiment ID | `ge1-autonomous-stop-validation-3355790` |
| Record status | `verified` |
| Scientific role | Engineering validation of the prospective autonomous-stop implementation only |
| Branch | `graph-v1-experiment-record` |
| Source commit | `288327e42b8575f774b9df668df288d3fa71ae33` |
| Governing decision | Accepted [ADR-0016](../decisions/ADR-0016-ge1-autonomous-stop-and-unconstrained-node-decoding.md) |
| Contract | [Autonomous-stop node-generation specification](../specifications/ge1_autonomous_stop_node_generation.md) |
| Slurm job | `3355790` |
| Compute host | `adroit-h11n2` |
| Runtime | CPython 3.8.13, PyTorch 1.11.0, CPU, one thread |
| Runner | `prototype/graph_encoder/adroit/ge1_stop_symbol_validation_cpu.slurm` |
| Runner SHA-256 | `b577214eaaaee29b97612f43a29ad86b9836a6db195a46254311ea9cd922ee8c` |
| Working tree | Clean before and after validation; standalone detached checkout |
| Execution date | August 20, 2026 |
| Outcome | Authoritative engineering-validation gate passed |

## Question and predetermined gate

The validation asked whether the opt-in
`GE1-PAD-TERMINATED-UNCONSTRAINED-NODES-v1` implementation executes in the
authoritative environment while preserving every historical identity and the
frozen Flat V6 source. The gate required:

- the exact clean detached commit before and after execution;
- CPython 3.8.13, PyTorch 1.11.0, CPU-only execution, and one thread;
- 19 focused tests passing with zero skips;
- complete graph-encoder discovery passing with exactly the six allowlisted
  CUDA-unavailable skips and no other skip, failure, or error;
- sibling suites at exactly 86 / 549 / 17 / 40 / 51 tests, with only the four
  established flat-baseline skips;
- separate V4 shared-decoder, v3 output-position, and v4 checkpoint contracts;
- documentation, compileall, Python 3.8 AST, source, confinement, frozen-runner,
  frozen-flat, Bash-syntax, whitespace, and clean-tree checks; and
- zero corpus, manifest, checkpoint, training, CAD-kernel, Stage 6, C8, or
  protected-partition access.

Any mismatch or absent terminal-success marker failed the job.

## Evidence integrity and scheduler result

The two raw logs were copied from Adroit and independently rehashed after the
job. Their local digests matched the values computed on Adroit.

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `stop-symbol-cpu-3355790.out` | 14,420 | `12060a14a6fa4c4ccf82214696fdad988a8479bb674d3bdfe27562ea426a12bc` |
| `stop-symbol-cpu-3355790.err` | 6,014 | `f93a37c4d235a0aa1220ae1bd080e9e6e9fc08c45f3ad000c0867978eb7f0318` |

Scheduler evidence was `COMPLETED / 0:0`, elapsed `00:08:10`, one allocated
CPU, and batch MaxRSS `738252K`. The terminal-success record agrees on exact
commit `288327e...`, job `3355790`, host `adroit-h11n2`, and the runtime. No
terminal-failure marker appears in either stream.

## Test and repository results

| Suite | Declared | Run | Failures | Errors | Skips |
|---|---:|---:|---:|---:|---:|
| `test_autonomous_stop_contract` | 11 | 11 | 0 | 0 | 0 |
| `test_autonomous_stop_runtime` | 8 | 8 | 0 | 0 | 0 |
| Complete graph-encoder discovery | 734 | 734 | 0 | 0 | 6 |
| `model_data` | 86 | 86 | 0 | 0 | 0 |
| `flat_baseline` | 549 | 549 | 0 | 0 | 4 |
| `graph_baseline` | 17 | 17 | 0 | 0 | 0 |
| `representation` | 40 | 40 | 0 | 0 | 0 |
| `controlled_data` | 51 | 51 | 0 | 0 | 0 |

The six discovery skips matched `CPU_DISCOVERY_CUDA_SKIP_IDS` exactly. The
four flat-baseline skips were the established external-bundle skips.
Documentation validation passed, `compileall` passed, and the Python 3.8 AST
gate parsed 110 files. The source audit passed, `ORDINAL_CUT_COUNT` remained
confined to `grid_magnitude.py` and `losses.py`, all three pinned ADR-0013
runners were unchanged, and the final tree remained clean. The frozen
`constrained_v6_autonomous.py` SHA-256 remained
`6391352ae11a7ee2a7dc3623e57bc14a135cdfac5b3453e379d13d696a7d5fb2`.

Two C7-v2 JSON records in the test streams are deliberate procedural-fixture
outputs, not scientific run events. Their protected-access fields are false,
and the runner's own terminal record independently reports every access and
execution field as false.

## Contracts exercised

The focused tests establish that the new identity uses plain argmax over all
eight node classes, stops on and strips one learned `<pad>`, records failure
without raising when the 16-node cap is reached, and allows invalid grammar to
reach the scorer. They also establish exactly one active terminator target,
exclusion of that position from graph pairs and graph loss, count-free
autonomous dispatch, legacy count/mask behavior, and cross-identity checkpoint
rejection.

The identity audit reported:

| Contract | Value |
|---|---|
| Node generation | `GE1-PAD-TERMINATED-UNCONSTRAINED-NODES-v1` |
| Magnitude | `GE1-OPERATION-MAGNITUDE-GRID-SOFTMAX-v1` |
| Shared decoder | `GE1-SHARED-TYPED-EDGE-DECODER-V4` |
| Output positions | `GE1-DECODER-OUTPUT-POSITIONS-v3` |
| Checkpoint schema | `GE1-CHECKPOINT-v4` |

The Stage 6 producer's cross-arm train-score difference is now diagnostic and
nonbinding; its per-run optimization checks and structural memory gates remain
binding and unchanged.

## Decision and limits

The authoritative engineering-validation gate **passed** for exact commit
`288327e42b8575f774b9df668df288d3fa71ae33`.

This validates the implementation under the production CPU runtime. It does
not validate a trained autonomous-stop checkpoint, establish encoder
performance, or authorize retraining, Stage 6, development, RR, ER, IID, CAD
execution, or any scientific run. The preceding [memory-to-count prerequisite
probe](ge1_memory_count_probe_3355776.md) remains false because typed-graph
seed 2028 missed the frozen all-checkpoint floor; therefore the retrain remains
blocked.

## Limitations

- No learned autonomous-stop checkpoint or corpus example was evaluated.
- Procedural runtime fixtures establish contract behavior, not termination or
  grammar-validity rates after training.
- CPU validation cannot execute the six CUDA-only Stage 6 runtime methods;
  their exact skip identities were checked instead.
- The locally copied logs live in temporary storage; their recorded digests,
  terminal telemetry, and Adroit scratch copies are the retained evidence.

## Related records

- [ADR-0016](../decisions/ADR-0016-ge1-autonomous-stop-and-unconstrained-node-decoding.md)
- [Autonomous-stop specification](../specifications/ge1_autonomous_stop_node_generation.md)
- [Memory-to-count prerequisite job 3355776](ge1_memory_count_probe_3355776.md)
- [Zero-memory job 3355342](ge1_stage6_zero_memory_3355342.md)
