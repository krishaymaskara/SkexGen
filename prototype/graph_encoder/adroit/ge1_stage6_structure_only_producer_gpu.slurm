#!/bin/bash
# GPU producer runner. Future reviewed submission supplies allocation flags.
#SBATCH --signal=B:TERM@60

set -Eeuo pipefail

infrastructure_failure() {
  status="${1:-$?}"
  trap - ERR
  peak="unavailable"
  if command -v nvidia-smi >/dev/null 2>&1; then
    peak="$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null | head -1 || true)"
  fi
  printf '{"event":"stage6_structure_only_gpu_infrastructure_failure","commit":"%s","job_id":"%s","exit_status":%s,"gpu_used_memory_mib":"%s"}\n' \
    "${EXPECTED_COMMIT:-unresolved}" "${SLURM_JOB_ID:-unresolved}" "$status" "$peak" >&2
  exit "$status"
}

terminal_timeout() {
  trap - ERR
  printf '{"event":"stage6_structure_only_gpu_terminal_timeout","commit":"%s","job_id":"%s","exit_status":124}\n' \
    "${EXPECTED_COMMIT:-unresolved}" "${SLURM_JOB_ID:-unresolved}" >&2
  exit 124
}

trap infrastructure_failure ERR
trap terminal_timeout TERM

: "${EXPECTED_COMMIT:?exact reviewed commit required}"
: "${EXPECTED_RUNNER_SHA256:?exact reviewed runner SHA-256 required}"
: "${SLURM_JOB_ID:?must run under a separately authorized GPU allocation}"
: "${TRAIN_INPUT_ROOT:?narrow authorized train input root required}"
: "${DEVELOPMENT_INPUT_ROOT:?narrow authorized development input root required}"
: "${RUN_PARENT:?governed output parent required}"
: "${TIMING_EVIDENCE:?verified timing-v2 JSON required}"
: "${CUDA_VISIBLE_DEVICES:?Slurm must expose exactly one GPU}"

REPOSITORY="${REPOSITORY:-/scratch/network/km6349/SkexGen-stage6-structure-only}"
CONTAINER="${CONTAINER:-/scratch/network/km6349/skexgen.sif}"
PYTHON="${PYTHON:-/root/miniconda3/bin/python3.8}"
RUNNER="$REPOSITORY/prototype/graph_encoder/adroit/ge1_stage6_structure_only_producer_gpu.slurm"
OUTPUT_DIR="$RUN_PARENT/ge1-stage6-structure-only-producer-${EXPECTED_COMMIT}-${SLURM_JOB_ID}"
CHECKPOINT_BUNDLE_DIR="$RUN_PARENT/ge1-stage6-structure-only-checkpoints-${EXPECTED_COMMIT}-${SLURM_JOB_ID}"
TRAIN_INDEX="$TRAIN_INPUT_ROOT/index.json"
TRAIN_ROOT="$TRAIN_INPUT_ROOT/payloads"
DEVELOPMENT_INDEX="$DEVELOPMENT_INPUT_ROOT/index.json"
DEVELOPMENT_ROOT="$DEVELOPMENT_INPUT_ROOT/payloads"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPYCACHEPREFIX="/tmp/ge1-stage6-producer-gpu-pyc-${SLURM_JOB_ID}"
export TMPDIR="/tmp/ge1-stage6-producer-gpu-${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"

container_python() {
  apptainer exec --nv --cleanenv \
    --env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
    --env CUBLAS_WORKSPACE_CONFIG="$CUBLAS_WORKSPACE_CONFIG" \
    --env OMP_NUM_THREADS=1 \
    --env MKL_NUM_THREADS=1 \
    --env PYTHONPYCACHEPREFIX="$PYTHONPYCACHEPREFIX" \
    --env TMPDIR="$TMPDIR" \
    --bind "$REPOSITORY:$REPOSITORY:ro" \
    --bind "$TRAIN_INPUT_ROOT:$TRAIN_INPUT_ROOT:ro" \
    --bind "$DEVELOPMENT_INPUT_ROOT:$DEVELOPMENT_INPUT_ROOT:ro" \
    --bind "$RUN_PARENT:$RUN_PARENT:rw" \
    "$CONTAINER" "$PYTHON" "$@"
}

test "${CUDA_VISIBLE_DEVICES#*,}" = "$CUDA_VISIBLE_DEVICES"
test -d "$REPOSITORY/.git"
test ! -f "$REPOSITORY/.git"
test -f "$TIMING_EVIDENCE"
test -f "$TRAIN_INDEX"
test -d "$TRAIN_ROOT"
test -f "$DEVELOPMENT_INDEX"
test -d "$DEVELOPMENT_ROOT"
test -d "$RUN_PARENT"
test -f "$CONTAINER"
test ! -e "$OUTPUT_DIR"
test ! -e "$CHECKPOINT_BUNDLE_DIR"
test "$(sha256sum "$RUNNER" | cut -d ' ' -f 1)" = "$EXPECTED_RUNNER_SHA256"
cd "$REPOSITORY"
test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git symbolic-ref -q --short HEAD || true)"
test -z "$(git status --porcelain=v1 --untracked-files=all)"
test "$(git rev-parse --show-toplevel)" = "$REPOSITORY"
test -z "$(git rev-parse --show-superproject-working-tree)"

container_python -B - "$TIMING_EVIDENCE" <<'PY'
import json
import os
import sys
from pathlib import Path
import torch
from prototype.graph_encoder.stage6_timing import validate_timing_evidence
assert sys.version_info[:3] == (3, 8, 13)
assert torch.__version__.split("+")[0] == "1.11.0"
assert torch.version.cuda is not None
assert torch.cuda.is_available() and torch.cuda.device_count() == 1
assert torch.get_num_threads() == 1
assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
record = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
validate_timing_evidence(record, required_device="cuda:0")
PY
container_python -B -m prototype.graph_encoder.stage6_structure_only_producer_audit "$RUNNER"
bash -n "$RUNNER"

set +e
container_python -B -m prototype.graph_encoder.stage6_structure_only_producer \
  --train-root "$TRAIN_ROOT" \
  --train-index "$TRAIN_INDEX" \
  --development-root "$DEVELOPMENT_ROOT" \
  --development-index "$DEVELOPMENT_INDEX" \
  --timing-evidence "$TIMING_EVIDENCE" \
  --output-dir "$OUTPUT_DIR" \
  --checkpoint-bundle-dir "$CHECKPOINT_BUNDLE_DIR" \
  --repository-root "$REPOSITORY" \
  --expected-commit "$EXPECTED_COMMIT" \
  --job-id "$SLURM_JOB_ID" \
  --allow-fallback \
  --require-selected-device cuda:0
producer_status=$?
set -e
if [[ "$producer_status" -eq 42 ]]; then
  printf '{"event":"stage6_structure_only_gpu_scientific_failure","commit":"%s","job_id":"%s","exit_status":42}\n' \
    "$EXPECTED_COMMIT" "$SLURM_JOB_ID" >&2
  exit 42
fi
if [[ "$producer_status" -ne 0 ]]; then
  infrastructure_failure "$producer_status"
fi

(
  cd "$OUTPUT_DIR"
  sha256sum -c SHA256SUMS
)
(
  cd "$CHECKPOINT_BUNDLE_DIR"
  sha256sum -c SHA256SUMS
)
container_python -B - "$OUTPUT_DIR" "$CHECKPOINT_BUNDLE_DIR" <<'PY'
import json
import sys
from prototype.graph_encoder.stage6_structure_only_producer import (
    load_execution_record, verify_checkpoint_bundle, verify_producer_artifact,
)
verification = verify_producer_artifact(sys.argv[1])
payload = load_execution_record(sys.argv[1])
assert payload["execution_evidence"]["selected_execution_device"] == "cuda:0"
assert payload["execution_evidence"]["cuda_peak_memory_bytes"] > 0
bundle = verify_checkpoint_bundle(
    sys.argv[2], retained_seeds=tuple(payload["retained_seeds"])
)
assert bundle["bundle_sha256"] == verification["checkpoint_bundle_sha256"]
print(json.dumps({
    "event":"stage6_structure_only_gpu_producer_verified",
    "cuda_peak_memory_bytes":payload["execution_evidence"]["cuda_peak_memory_bytes"],
    **verification,
}, sort_keys=True, separators=(",", ":")), flush=True)
PY

test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git status --porcelain=v1 --untracked-files=all)"
printf '{"event":"stage6_structure_only_gpu_terminal_success","commit":"%s","job_id":"%s","artifact":"%s","checkpoint_bundle":"%s"}\n' \
  "$EXPECTED_COMMIT" "$SLURM_JOB_ID" "$OUTPUT_DIR" "$CHECKPOINT_BUNDLE_DIR"
