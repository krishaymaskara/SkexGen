#!/bin/bash
# Prospective GPU-allocation timing runner. Allocation flags are intentionally
# absent; the future reviewed command supplies partition, GPU, memory, and time.
#SBATCH --signal=B:TERM@60

set -Eeuo pipefail

terminal_failure() {
  status=$?
  trap - ERR
  printf '{"event":"stage6_hardware_timing_terminal_failure","commit":"%s","job_id":"%s","exit_status":%s}\n' \
    "${EXPECTED_COMMIT:-unresolved}" "${SLURM_JOB_ID:-unresolved}" "$status" >&2
  exit "$status"
}

terminal_timeout() {
  trap - ERR
  printf '{"event":"stage6_hardware_timing_terminal_timeout","commit":"%s","job_id":"%s","exit_status":124}\n' \
    "${EXPECTED_COMMIT:-unresolved}" "${SLURM_JOB_ID:-unresolved}" >&2
  exit 124
}

trap terminal_failure ERR
trap terminal_timeout TERM

: "${EXPECTED_COMMIT:?exact reviewed commit required}"
: "${EXPECTED_RUNNER_SHA256:?exact reviewed runner SHA-256 required}"
: "${SLURM_JOB_ID:?must execute under a separately authorized GPU allocation}"
: "${TRAIN_INPUT_ROOT:?approved narrow train package required}"
: "${TIMING_OUTPUT_PARENT:?timing output parent required}"
: "${AVAILABLE_WALL_SECONDS_CPU:?reviewed CPU candidate wall time required}"
: "${AVAILABLE_WALL_SECONDS_CUDA:?reviewed CUDA candidate wall time required}"
: "${CUDA_VISIBLE_DEVICES:?Slurm must expose exactly one GPU}"
: "${NODE_GENERATION_IDENTITY:?explicit node-generation identity required}"

REPOSITORY="${REPOSITORY:-/scratch/network/km6349/SkexGen-stage6-structure-only}"
CONTAINER="${CONTAINER:-/scratch/network/km6349/skexgen.sif}"
PYTHON="${PYTHON:-/root/miniconda3/bin/python3.8}"
RUNNER="$REPOSITORY/prototype/graph_encoder/adroit/ge1_stage6_hardware_timing_gpu.slurm"
TRAIN_INDEX="$TRAIN_INPUT_ROOT/index.json"
TRAIN_ROOT="$TRAIN_INPUT_ROOT/payloads"
OUTPUT_DIR="$TIMING_OUTPUT_PARENT/ge1-stage6-hardware-timing-${EXPECTED_COMMIT}-${SLURM_JOB_ID}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPYCACHEPREFIX="/tmp/ge1-stage6-timing-pyc-${SLURM_JOB_ID}"
export TMPDIR="/tmp/ge1-stage6-timing-${SLURM_JOB_ID}"
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
    --bind "$TIMING_OUTPUT_PARENT:$TIMING_OUTPUT_PARENT:rw" \
    "$CONTAINER" "$PYTHON" "$@"
}

test "${CUDA_VISIBLE_DEVICES#*,}" = "$CUDA_VISIBLE_DEVICES"
test -d "$REPOSITORY/.git"
test ! -f "$REPOSITORY/.git"
test -f "$TRAIN_INDEX"
test -d "$TRAIN_ROOT"
test ! -L "$TRAIN_INPUT_ROOT"
test -d "$TIMING_OUTPUT_PARENT"
test ! -e "$OUTPUT_DIR"
test -f "$CONTAINER"
test "$(sha256sum "$RUNNER" | cut -d ' ' -f 1)" = "$EXPECTED_RUNNER_SHA256"
cd "$REPOSITORY"
test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git symbolic-ref -q --short HEAD || true)"
test -z "$(git status --porcelain=v1 --untracked-files=all)"
test "$(git rev-parse --show-toplevel)" = "$REPOSITORY"
test -z "$(git rev-parse --show-superproject-working-tree)"

container_python -B - <<'PY'
import os
import sys
import torch
assert sys.version_info[:3] == (3, 8, 13)
assert torch.__version__.split("+")[0] == "1.11.0"
assert torch.version.cuda is not None
assert torch.cuda.is_available()
assert torch.cuda.device_count() == 1
assert torch.get_num_threads() == 1
assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
PY
container_python -B -m prototype.graph_encoder.stage6_timing_audit "$RUNNER"
bash -n "$RUNNER"

container_python -B -m prototype.graph_encoder.stage6_timing \
  --train-index "$TRAIN_INDEX" \
  --train-root "$TRAIN_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --repository-root "$REPOSITORY" \
  --expected-commit "$EXPECTED_COMMIT" \
  --job-id "$SLURM_JOB_ID" \
  --available-wall-seconds-cpu "$AVAILABLE_WALL_SECONDS_CPU" \
  --available-wall-seconds-cuda "$AVAILABLE_WALL_SECONDS_CUDA" \
  --node-generation-identity "$NODE_GENERATION_IDENTITY"

(
  cd "$OUTPUT_DIR"
  sha256sum -c SHA256SUMS
)
container_python -B - "$OUTPUT_DIR" <<'PY'
import json
import sys
from prototype.graph_encoder.stage6_timing import verify_timing_artifact
print(json.dumps({
    "event":"stage6_hardware_timing_verified",
    **verify_timing_artifact(sys.argv[1]),
}, sort_keys=True, separators=(",", ":")), flush=True)
PY

test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git status --porcelain=v1 --untracked-files=all)"
printf '{"event":"stage6_hardware_timing_terminal_success","commit":"%s","job_id":"%s","artifact":"%s"}\n' \
  "$EXPECTED_COMMIT" "$SLURM_JOB_ID" "$OUTPUT_DIR"
