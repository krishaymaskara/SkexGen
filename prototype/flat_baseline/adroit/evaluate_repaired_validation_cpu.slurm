#!/bin/bash
#SBATCH --job-name=flat-repaired-validation
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/network/km6349/phase-b-repaired-validation-%j.out
#SBATCH --error=/scratch/network/km6349/phase-b-repaired-validation-%j.err

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
test -f "$CONTAINER"
mkdir -p "$RUN_ROOT"
test -d "$RUN_ROOT"

NAMESPACE="${RUN_ROOT}/phase-b-repaired-validation-${REVIEWED_COMMIT}-${SLURM_JOB_ID}"
EVALUATION="${NAMESPACE}/evaluation"
REPORT="${NAMESPACE}/gate-a-to-d-inputs.json"
EVIDENCE="${NAMESPACE}/workflow-evidence"
HASH_MANIFEST="${NAMESPACE}/sha256-manifest.txt"
test ! -e "$NAMESPACE"
mkdir "$NAMESPACE"
mkdir "$EVIDENCE"

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONHASHSEED=0
export PYTHONPYCACHEPREFIX="/tmp/flat-repaired-validation-pycache-${SLURM_JOB_ID}"

stage environment_preflight
container_python -B -c \
  'import sys,torch; assert sys.version_info[:2] == (3,8); assert torch.__version__.split("+")[0] == "1.11.0"; assert not torch.cuda.is_available()'
container_python -B - "$EVIDENCE/environment.json" <<'PY'
import json
import os
import platform
import sys

import torch

torch.manual_seed(2026)
torch.set_num_threads(1)
payload = {
    "cuda_available": torch.cuda.is_available(),
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
    "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
    "platform": platform.platform(),
    "python_version": platform.python_version(),
    "pytorch_version": str(torch.__version__),
    "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
    "evaluation_seed": 2026,
    "torch_num_threads": torch.get_num_threads(),
}
with open(sys.argv[1], "w", encoding="utf-8") as stream:
    json.dump(payload, stream, sort_keys=True, separators=(",", ":"))
    stream.write("\n")
PY
{
  git status --porcelain=v1 --untracked-files=all
  git branch --show-current
  git rev-parse HEAD
  git rev-parse origin/flat-mixed-baseline
  git show -s --format=fuller HEAD
} > "$EVIDENCE/repository.txt"
{
  printf 'slurm_job_id=%s\n' "$SLURM_JOB_ID"
  printf 'slurm_job_name=%s\n' "${SLURM_JOB_NAME:-}"
  printf 'slurm_job_node_list=%s\n' "${SLURM_JOB_NODELIST:-}"
  printf 'slurm_submit_dir=%s\n' "${SLURM_SUBMIT_DIR:-}"
  printf 'stdout=/scratch/network/km6349/phase-b-repaired-validation-%s.out\n' \
    "$SLURM_JOB_ID"
  printf 'stderr=/scratch/network/km6349/phase-b-repaired-validation-%s.err\n' \
    "$SLURM_JOB_ID"
  scontrol show job "$SLURM_JOB_ID"
} > "$EVIDENCE/scheduler.txt"
sha256sum "$CONTAINER" > "$EVIDENCE/container.sha256"

stage checkpoint_preflight
ACTUAL_CHECKPOINT_SHA256="$(sha256sum "$CHECKPOINT" | awk '{print $1}')"
test "$ACTUAL_CHECKPOINT_SHA256" = "$EXPECTED_CHECKPOINT_SHA256"
printf '%s  %s\n' "$ACTUAL_CHECKPOINT_SHA256" "$CHECKPOINT" \
  > "$EVIDENCE/checkpoint.sha256"
printf 'checkpoint_sha256=%s checkpoint_epoch=44 checkpoint_global_step=748\n' \
  "$ACTUAL_CHECKPOINT_SHA256"

stage partition_preflight
container_python -B - "$CORPUS" "$EVIDENCE/partition.json" <<'PY'
import hashlib
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
selected = authority["validation"]
if len(selected) != 68:
    raise RuntimeError("all 68 validation IDs are required")
digest = hashlib.sha256(
    ("\n".join(selected) + "\n").encode("utf-8")
).hexdigest()
payload = {
    "partition": "validation",
    "authoritative_partition_family_counts": counts,
    "selected_family_ids": list(selected),
    "selected_family_ids_sha256": digest,
    "selected_family_count": len(selected),
    "train_family_records_loaded": 0,
    "validation_family_records_loaded": 0,
    "test_family_records_loaded": 0,
    "test_partition_evaluated": False,
}
with open(sys.argv[2], "w", encoding="utf-8") as stream:
    json.dump(payload, stream, sort_keys=True, separators=(",", ":"))
    stream.write("\n")
print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
PY
sha256sum \
  "$CORPUS/corpus_manifest.json" \
  "$CORPUS/manifests/iid.json" \
  > "$EVIDENCE/corpus-manifests.sha256"

stage regressions
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
container_python -B tools/check_documentation.py
git diff --check
bash -n prototype/flat_baseline/adroit/evaluate_repaired_validation_cpu.slurm
container_python -B - "$REPOSITORY" <<'PY'
import ast
from pathlib import Path
import sys

for path in Path(sys.argv[1]).rglob("*.py"):
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
PY
printf 'status=passed\n' > "$EVIDENCE/regressions.txt"

stage execution_preflight
test "$(git branch --show-current)" = "flat-mixed-baseline"
test "$(git rev-parse HEAD)" = "$REVIEWED_COMMIT"
test "$(git rev-parse origin/flat-mixed-baseline)" = "$REVIEWED_COMMIT"
test -z "$(git status --porcelain=v1 --untracked-files=all)"
container_python -B -c \
  'import sys,torch; assert sys.version_info[:2] == (3,8); assert torch.__version__.split("+")[0] == "1.11.0"; assert not torch.cuda.is_available()'

stage full_evaluation
set +e
container_python -B -m \
  prototype.flat_baseline.evaluate_length_conditioned \
  --corpus-dir "$CORPUS" \
  --checkpoint "$CHECKPOINT" \
  --split-manifest iid \
  --partition validation \
  --output-dir "$EVALUATION" \
  --batch-size 32 \
  --device cpu \
  --write-raw-predictions \
  --repaired-full-contract
FULL_STATUS=$?
set -e
if test "$FULL_STATUS" -ne 0 && test "$FULL_STATUS" -ne 2; then
  exit "$FULL_STATUS"
fi
test -e "$EVALUATION"

stage post_evaluation_evidence
{
  printf 'snapshot=post_evaluation\n'
  printf 'full_evaluation_exit_code=%s\n' "$FULL_STATUS"
  scontrol show job "$SLURM_JOB_ID"
} >> "$EVIDENCE/scheduler.txt"

stage artifact_validation
container_python -B -m \
  prototype.flat_baseline.verify_evaluation_artifacts \
  --output "$EVALUATION" \
  --corpus-dir "$CORPUS" \
  --checkpoint "$CHECKPOINT" \
  --reviewed-commit "$REVIEWED_COMMIT" \
  --expected-authoritative-family-count 680 \
  --expected-authoritative-validation-count 68 \
  --report-output "$REPORT" \
  --full-exit-code "$FULL_STATUS" \
  --job-id "$SLURM_JOB_ID" \
  --regression-status passed \
  --workflow-evidence-dir "$EVIDENCE" \
  --repaired-full-contract
test -f "$REPORT"

stage artifact_manifest
find "$EVALUATION" "$EVIDENCE" -type f -print0 \
  | sort -z \
  | xargs -0 sha256sum > "$HASH_MANIFEST"
sha256sum "$REPORT" >> "$HASH_MANIFEST"
test "$(wc -l < "$HASH_MANIFEST")" -ge 13

stage complete
printf 'namespace=%s\nevaluation=%s\nreport=%s\nhash_manifest=%s\n' \
  "$NAMESPACE" "$EVALUATION" "$REPORT" "$HASH_MANIFEST"
printf 'full_exit_code=%s\n' "$FULL_STATUS"
