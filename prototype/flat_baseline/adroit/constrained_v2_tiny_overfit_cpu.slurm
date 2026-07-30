#!/bin/bash
#SBATCH --job-name=v2-tiny-overfit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/network/km6349/v2-tiny-overfit-%j.out
#SBATCH --error=/scratch/network/km6349/v2-tiny-overfit-%j.err

set -Eeuo pipefail

terminal_failure() {
  status=$?
  trap - ERR
  printf '{"event":"slurm_terminal_failure","commit":"%s","job_id":"%s","exit_status":%s,"test_partition_accessed":false}\n' \
    "${REVIEWED_COMMIT:-unresolved}" "${SLURM_JOB_ID:-unresolved}" "$status" >&2
  exit "$status"
}
trap terminal_failure ERR

REPOSITORY="/scratch/network/km6349/SkexGen"
CORPUS="/scratch/network/km6349/controlled_corpora/b0-pilot-680-seed2026"
RUN_ROOT="/scratch/network/km6349/constrained_v2_runs"
CONTAINER="/scratch/network/km6349/skexgen.sif"
PYTHON="/root/miniconda3/bin/python3.8"
: "${REVIEWED_COMMIT:?set REVIEWED_COMMIT to the reviewed Stage 2E commit}"
: "${SLURM_JOB_ID:?this workflow must run under Slurm}"

OUTPUT="${RUN_ROOT}/tiny-overfit-${REVIEWED_COMMIT}-${SLURM_JOB_ID}"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPYCACHEPREFIX="/tmp/v2-tiny-overfit-${SLURM_JOB_ID}"

container_python() {
  apptainer exec \
    --cleanenv \
    --env CUDA_VISIBLE_DEVICES="" \
    --env OMP_NUM_THREADS=1 \
    --env MKL_NUM_THREADS=1 \
    --env PYTHONPYCACHEPREFIX="$PYTHONPYCACHEPREFIX" \
    --bind /scratch/network/km6349:/scratch/network/km6349 \
    "$CONTAINER" "$PYTHON" "$@"
}

cd "$REPOSITORY"
BRANCH="$(git branch --show-current)"
COMMIT="$(git rev-parse HEAD)"
test "$BRANCH" = "constrained-profile-decoder"
test "$COMMIT" = "$REVIEWED_COMMIT"
test "$(git rev-parse origin/constrained-profile-decoder)" = "$REVIEWED_COMMIT"
test -z "$(git status --porcelain=v1 --untracked-files=all)"
test -d "$CORPUS"
test ! -e "$OUTPUT"
test ! -L "$OUTPUT"
printf '{"event":"slurm_start","branch":"%s","commit":"%s","job_id":"%s","container":"%s","training_partition":"train","validation_partition":"none","test_partition_accessed":false}\n' \
  "$BRANCH" "$COMMIT" "$SLURM_JOB_ID" "$CONTAINER"

container_python - <<'PY'
import sys
import torch
assert sys.version_info[:2] == (3, 8)
assert torch.__version__.split("+")[0] == "1.11.0"
assert not torch.cuda.is_available()
print("python_version={}".format(sys.version.split()[0]))
print("pytorch_version={}".format(torch.__version__))
print("pytorch_cuda_build={}".format(torch.version.cuda))
print("execution_device=cpu")
PY

container_python -B -m unittest \
  prototype.flat_baseline.tests.test_constrained_v2_training

container_python -B -m prototype.flat_baseline.train_constrained_v2 \
  --corpus-dir "$CORPUS" \
  --output-dir "$OUTPUT" \
  --device cpu \
  --seed 2026 \
  --maximum-steps 500 \
  --batch-size 8 \
  --tiny-subset-size 8

test -f "$OUTPUT/final.pt"
test -f "$OUTPUT/metrics.jsonl"
printf '{"event":"slurm_terminal_success","commit":"%s","job_id":"%s","training_partition":"train","validation_partition":"none","test_partition_accessed":false}\n' \
  "$REVIEWED_COMMIT" "$SLURM_JOB_ID"
