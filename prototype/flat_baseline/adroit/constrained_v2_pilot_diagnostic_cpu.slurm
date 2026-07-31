#!/bin/bash
#SBATCH --job-name=v2-iid-diagnostic
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/network/km6349/v2-iid-diagnostic-%j.out
#SBATCH --error=/scratch/network/km6349/v2-iid-diagnostic-%j.err

set -Eeuo pipefail

terminal_failure() {
  status=$?
  trap - ERR
  printf '{"event":"slurm_terminal_failure","commit":"%s","job_id":"%s","exit_status":%s,"training_occurred":false,"systematic_partition_accessed":false,"test_partition_accessed":false}\n' \
    "${REVIEWED_COMMIT:-unresolved}" "${SLURM_JOB_ID:-unresolved}" "$status" >&2
  exit "$status"
}
trap terminal_failure ERR

REPOSITORY="/scratch/network/km6349/SkexGen"
CORPUS="/scratch/network/km6349/controlled_corpora/b0-pilot-680-seed2026"
PILOT_RUN="/scratch/network/km6349/constrained_v2_runs/iid-pilot-9e662310b41b3750a49547b95fa6af7a9235a8b4-3333190"
CHECKPOINT="${PILOT_RUN}/epoch-0002.pt"
EPOCH1_CHECKPOINT="${PILOT_RUN}/epoch-0001.pt"
RUN_ROOT="/scratch/network/km6349/constrained_v2_diagnostics"
CONTAINER="/scratch/network/km6349/skexgen.sif"
PYTHON="/root/miniconda3/bin/python3.8"
: "${REVIEWED_COMMIT:?set REVIEWED_COMMIT to the reviewed Stage 2G commit}"
: "${SLURM_JOB_ID:?this workflow must run under Slurm}"

OUTPUT="${RUN_ROOT}/iid-diagnostic-${REVIEWED_COMMIT}-${SLURM_JOB_ID}"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPYCACHEPREFIX="/tmp/v2-iid-diagnostic-${SLURM_JOB_ID}"

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
HOST="$(hostname)"
test "$BRANCH" = "constrained-profile-decoder"
test "$COMMIT" = "$REVIEWED_COMMIT"
test "$(git rev-parse origin/constrained-profile-decoder)" = "$REVIEWED_COMMIT"
test -z "$(git status --porcelain=v1 --untracked-files=all)"
test -d "$CORPUS"
test -f "$CHECKPOINT"
test -f "$EPOCH1_CHECKPOINT"
test ! -e "$OUTPUT"
test ! -L "$OUTPUT"
CHECKPOINT_HASH_BEFORE="$(sha256sum "$CHECKPOINT" | awk '{print $1}')"
printf '{"event":"slurm_start","branch":"%s","commit":"%s","job_id":"%s","host":"%s","container":"%s","checkpoint":"%s","checkpoint_sha256":"%s","training_partition":"train","validation_partition":"iid_validation","training_occurred":false,"systematic_partition_accessed":false,"test_partition_accessed":false}\n' \
  "$BRANCH" "$COMMIT" "$SLURM_JOB_ID" "$HOST" "$CONTAINER" "$CHECKPOINT" "$CHECKPOINT_HASH_BEFORE"

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
  prototype.flat_baseline.tests.test_diagnose_constrained_v2_pilot

container_python -B -m prototype.flat_baseline.diagnose_constrained_v2_pilot \
  --corpus-dir "$CORPUS" \
  --checkpoint "$CHECKPOINT" \
  --epoch1-checkpoint "$EPOCH1_CHECKPOINT" \
  --output-dir "$OUTPUT" \
  --device cpu

test -f "$OUTPUT/metrics.jsonl"
test -f "$OUTPUT/first_failures.jsonl"
CHECKPOINT_HASH_AFTER="$(sha256sum "$CHECKPOINT" | awk '{print $1}')"
test "$CHECKPOINT_HASH_BEFORE" = "$CHECKPOINT_HASH_AFTER"
container_python - "$OUTPUT/metrics.jsonl" <<'PY'
import json
import sys
records = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8")]
assert records[0]["event"] == "diagnostic_metadata"
assert records[-1]["event"] == "terminal_success"
assert records[-1]["validation_family_count"] == 68
assert records[-1]["systematic_partition_accessed"] is False
assert records[-1]["test_partition_accessed"] is False
print("diagnostic_metrics_contract=ok")
PY
printf '{"event":"slurm_terminal_success","commit":"%s","job_id":"%s","checkpoint_sha256":"%s","training_occurred":false,"systematic_partition_accessed":false,"test_partition_accessed":false}\n' \
  "$REVIEWED_COMMIT" "$SLURM_JOB_ID" "$CHECKPOINT_HASH_AFTER"
