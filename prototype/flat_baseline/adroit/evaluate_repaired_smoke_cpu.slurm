#!/bin/bash
#SBATCH --job-name=flat-repaired-smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/network/km6349/phase-b-repaired-smoke-%j.out
#SBATCH --error=/scratch/network/km6349/phase-b-repaired-smoke-%j.err

set -Eeuo pipefail

REPOSITORY="/scratch/network/km6349/SkexGen"
CORPUS="/scratch/network/km6349/controlled_corpora/b0-pilot-680-seed2026"
CHECKPOINT="/scratch/network/km6349/flat_baseline_runs/train-kmeans-full-d549ebe4478e166fda8d54f2b173572a1ab52e7a-3326757/best.pt"
EXPECTED_CHECKPOINT_SHA256="282988af00a2dc9a53e14ceb270537d35f85d5339dc989634f88af5f77931267"
CONTAINER="/scratch/network/km6349/skexgen.sif"
PYTHON="/root/miniconda3/bin/python3.8"
RUN_ROOT="/scratch/network/km6349/flat_baseline_evaluations"
: "${REVIEWED_COMMIT:?set REVIEWED_COMMIT to the reviewed evaluator commit}"
: "${SLURM_JOB_ID:?this workflow must run under Slurm}"

CURRENT_STAGE="initialization"
failure_report() {
  local status=$?
  local line="${BASH_LINENO[0]:-${LINENO}}"
  printf 'failure_stage=%s failure_line=%s exit_code=%s\n' \
    "$CURRENT_STAGE" "$line" "$status" >&2
  exit "$status"
}
trap failure_report ERR

stage() {
  CURRENT_STAGE="$1"
  printf 'stage=%s\n' "$CURRENT_STAGE"
}

container_python() {
  apptainer exec \
    --cleanenv \
    --env CUDA_VISIBLE_DEVICES="" \
    --env OMP_NUM_THREADS=1 \
    --env MKL_NUM_THREADS=1 \
    --env PYTHONHASHSEED=0 \
    --env PYTHONPYCACHEPREFIX="$PYTHONPYCACHEPREFIX" \
    --bind /scratch/network/km6349:/scratch/network/km6349 \
    "$CONTAINER" "$PYTHON" "$@"
}

stage repository_preflight
cd "$REPOSITORY"
test "$(git branch --show-current)" = "flat-mixed-baseline"
test "$(git rev-parse HEAD)" = "$REVIEWED_COMMIT"
test "$(git rev-parse origin/flat-mixed-baseline)" = "$REVIEWED_COMMIT"
test -z "$(git status --porcelain=v1 --untracked-files=all)"
test -d "$CORPUS"
test -f "$CHECKPOINT"
mkdir -p "$RUN_ROOT"
test -d "$RUN_ROOT"

SMOKE_A="${RUN_ROOT}/phase-b-repaired-smoke-${REVIEWED_COMMIT}-${SLURM_JOB_ID}-a"
SMOKE_B="${RUN_ROOT}/phase-b-repaired-smoke-${REVIEWED_COMMIT}-${SLURM_JOB_ID}-b"
test ! -e "$SMOKE_A"
test ! -e "$SMOKE_B"

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONHASHSEED=0
export PYTHONPYCACHEPREFIX="/tmp/flat-repaired-smoke-pycache-${SLURM_JOB_ID}"

stage environment_preflight
container_python -c \
  'import sys,torch; assert sys.version_info[:2] == (3,8); assert torch.__version__.split("+")[0] == "1.11.0"; assert not torch.cuda.is_available()'

stage checkpoint_preflight
ACTUAL_CHECKPOINT_SHA256="$(sha256sum "$CHECKPOINT" | awk '{print $1}')"
test "$ACTUAL_CHECKPOINT_SHA256" = "$EXPECTED_CHECKPOINT_SHA256"
printf 'checkpoint_sha256=%s checkpoint_epoch=44 checkpoint_global_step=748\n' \
  "$ACTUAL_CHECKPOINT_SHA256"

stage partition_preflight
container_python -B - "$CORPUS" <<'PY'
import json
import sys

from prototype.model_data.loader import partition_family_ids

authority = partition_family_ids(sys.argv[1], "iid")
counts = {
    name: len(authority[name])
    for name in ("train", "validation", "test")
}
if counts != {"train": 544, "validation": 68, "test": 68}:
    raise RuntimeError("authoritative partition counts differ")
selected = authority["validation"][:6]
if len(selected) != 6:
    raise RuntimeError("six validation smoke IDs are required")
print(json.dumps({
    "partition": "validation",
    "authoritative_partition_family_counts": counts,
    "selected_family_ids": list(selected),
    "selected_family_count": len(selected),
    "train_family_records_loaded": 0,
    "validation_family_records_loaded": 0,
    "test_family_records_loaded": 0,
}, sort_keys=True, separators=(",", ":")))
PY

stage regressions
container_python -B -m unittest \
  prototype.flat_baseline.tests.test_evaluate_length_conditioned
container_python -B -m unittest \
  prototype.model_data.tests.test_loading
container_python -B -m unittest discover \
  -s prototype/model_data/tests -t .
container_python -B -m unittest discover \
  -s prototype/flat_baseline/tests -t .

run_smoke() {
  local destination="$1"
  set +e
  container_python -B -m \
    prototype.flat_baseline.evaluate_length_conditioned \
    --corpus-dir "$CORPUS" \
    --checkpoint "$CHECKPOINT" \
    --split-manifest iid \
    --partition validation \
    --output-dir "$destination" \
    --family-limit 6 \
    --batch-size 3 \
    --device cpu \
    --write-raw-predictions \
    --repaired-smoke-contract
  local status=$?
  set -e
  if test "$status" -ne 0 && test "$status" -ne 2; then
    return "$status"
  fi
  test -e "$destination"
  printf '%s\n' "$status"
}

stage smoke_a
SMOKE_A_STATUS="$(run_smoke "$SMOKE_A")"

stage smoke_b
SMOKE_B_STATUS="$(run_smoke "$SMOKE_B")"

stage smoke_replay
container_python -B -m \
  prototype.flat_baseline.verify_evaluation_artifacts \
  --output "$SMOKE_A" \
  --compare-output "$SMOKE_B" \
  --corpus-dir "$CORPUS" \
  --checkpoint "$CHECKPOINT" \
  --reviewed-commit "$REVIEWED_COMMIT" \
  --expected-authoritative-family-count 680 \
  --expected-authoritative-validation-count 68 \
  --family-limit 6 \
  --repaired-smoke-contract

stage artifact_validation
for output in "$SMOKE_A" "$SMOKE_B"; do
  container_python -B -m \
    prototype.flat_baseline.verify_evaluation_artifacts \
    --output "$output" \
    --corpus-dir "$CORPUS" \
    --checkpoint "$CHECKPOINT" \
    --reviewed-commit "$REVIEWED_COMMIT" \
    --expected-authoritative-family-count 680 \
    --expected-authoritative-validation-count 68 \
    --family-limit 6 \
    --repaired-smoke-contract
done
test "$SMOKE_A_STATUS" = "$SMOKE_B_STATUS"

stage complete
printf 'smoke_a=%s\nsmoke_b=%s\nsmoke_exit_code=%s\n' \
  "$SMOKE_A" "$SMOKE_B" "$SMOKE_A_STATUS"
