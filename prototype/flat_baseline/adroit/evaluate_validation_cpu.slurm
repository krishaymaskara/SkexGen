#!/bin/bash
#SBATCH --job-name=flat-eval-validation
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/network/km6349/phase-b-evaluation-%j.out
#SBATCH --error=/scratch/network/km6349/phase-b-evaluation-%j.err

set -euo pipefail

REPOSITORY="/scratch/network/km6349/SkexGen"
CORPUS="/scratch/network/km6349/controlled_corpora/b0-pilot-680-seed2026"
CHECKPOINT="/scratch/network/km6349/flat_baseline_runs/b0-pilot-680-k32-lr1e3-seed2026/best.pt"
CONTAINER="/scratch/network/km6349/skexgen.sif"
PYTHON="/root/miniconda3/bin/python3.8"
RUN_ROOT="/scratch/network/km6349/flat_baseline_evaluations"
: "${REVIEWED_COMMIT:?set REVIEWED_COMMIT to the accepted Stage 4-6 commit}"
: "${SLURM_JOB_ID:?this workflow must run under Slurm}"

cd "$REPOSITORY"
test "$(git branch --show-current)" = "flat-mixed-baseline"
test "$(git rev-parse HEAD)" = "$REVIEWED_COMMIT"
test "$(git rev-parse origin/flat-mixed-baseline)" = "$REVIEWED_COMMIT"
test -z "$(git status --porcelain=v1 --untracked-files=all)"
test -f "$CHECKPOINT"
test -d "$CORPUS"
mkdir -p "$RUN_ROOT"
test -d "$RUN_ROOT"

SMOKE_A="${RUN_ROOT}/validation-smoke-${SLURM_JOB_ID}-a"
SMOKE_B="${RUN_ROOT}/validation-smoke-${SLURM_JOB_ID}-b"
FULL="${RUN_ROOT}/validation-${REVIEWED_COMMIT}-${SLURM_JOB_ID}"
REPORT="${RUN_ROOT}/validation-report-${SLURM_JOB_ID}.json"
test ! -e "$SMOKE_A"
test ! -e "$SMOKE_B"
test ! -e "$FULL"
test ! -e "$REPORT"

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPYCACHEPREFIX="/tmp/flat-eval-pycache-${SLURM_JOB_ID}"

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

container_python -c \
  'import sys,torch; assert sys.version_info[:2] == (3,8); assert torch.__version__.split("+")[0] == "1.11.0"; assert not torch.cuda.is_available()'

container_python -B -m unittest \
  prototype.flat_baseline.tests.test_evaluate_length_conditioned
container_python -B -m unittest \
  prototype.flat_baseline.tests.test_shared_memory_decoding
container_python -B -m unittest \
  prototype.flat_baseline.tests.test_autonomous \
  prototype.flat_baseline.tests.test_model
container_python -B -m unittest \
  prototype.flat_baseline.tests.test_conversion \
  prototype.flat_baseline.tests.test_metrics
container_python -B -m unittest discover \
  -s prototype/model_data/tests -t .
container_python -B -m unittest discover \
  -s prototype/flat_baseline/tests -t .

run_evaluation() {
  local destination="$1"
  local limit="$2"
  local batch_size="$3"
  local arguments=(
    -B -m prototype.flat_baseline.evaluate_length_conditioned
    --corpus-dir "$CORPUS"
    --checkpoint "$CHECKPOINT"
    --split-manifest iid
    --partition validation
    --output-dir "$destination"
    --batch-size "$batch_size"
    --device cpu
    --write-raw-predictions
  )
  if test -n "$limit"; then
    arguments+=(--family-limit "$limit")
  fi
  set +e
  container_python "${arguments[@]}"
  local status=$?
  set -e
  if test "$status" -ne 0 && test "$status" -ne 2; then
    return "$status"
  fi
  test -d "$destination"
  printf '%s\n' "$status"
}

SMOKE_A_STATUS="$(run_evaluation "$SMOKE_A" 6 3)"
container_python -B -m prototype.flat_baseline.verify_evaluation_artifacts \
  --output "$SMOKE_A" \
  --corpus-dir "$CORPUS" \
  --checkpoint "$CHECKPOINT" \
  --reviewed-commit "$REVIEWED_COMMIT" \
  --expected-authoritative-family-count 680 \
  --expected-authoritative-validation-count 68 \
  --family-limit 6
test -z "$(find "$RUN_ROOT" -maxdepth 1 -type d -name '.validation-smoke-*.tmp-*' -print -quit)"

SMOKE_B_STATUS="$(run_evaluation "$SMOKE_B" 6 3)"
container_python -B -m prototype.flat_baseline.verify_evaluation_artifacts \
  --output "$SMOKE_A" \
  --compare-output "$SMOKE_B" \
  --corpus-dir "$CORPUS" \
  --checkpoint "$CHECKPOINT" \
  --reviewed-commit "$REVIEWED_COMMIT" \
  --expected-authoritative-family-count 680 \
  --expected-authoritative-validation-count 68 \
  --family-limit 6
test -z "$(find "$RUN_ROOT" -maxdepth 1 -type d -name '.validation-smoke-*.tmp-*' -print -quit)"

FULL_STATUS="$(run_evaluation "$FULL" "" 32)"
container_python -B -m prototype.flat_baseline.verify_evaluation_artifacts \
  --output "$FULL" \
  --corpus-dir "$CORPUS" \
  --checkpoint "$CHECKPOINT" \
  --reviewed-commit "$REVIEWED_COMMIT" \
  --expected-authoritative-family-count 680 \
  --expected-authoritative-validation-count 68 \
  --report-output "$REPORT" \
  --smoke-output-a "$SMOKE_A" \
  --smoke-output-b "$SMOKE_B" \
  --smoke-exit-code-a "$SMOKE_A_STATUS" \
  --smoke-exit-code-b "$SMOKE_B_STATUS" \
  --full-exit-code "$FULL_STATUS" \
  --job-id "$SLURM_JOB_ID" \
  --regression-status passed
test -z "$(find "$RUN_ROOT" -maxdepth 1 -type d -name '.validation-*.tmp-*' -print -quit)"
test -z "$(find "$RUN_ROOT" -maxdepth 1 -type f -name '.validation-report-*.tmp-*' -print -quit)"
test -f "$REPORT"
