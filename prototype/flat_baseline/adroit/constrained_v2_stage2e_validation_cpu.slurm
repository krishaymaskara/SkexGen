#!/bin/bash
#SBATCH --job-name=v2-stage2e-validate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/network/km6349/v2-stage2e-validate-%j.out
#SBATCH --error=/scratch/network/km6349/v2-stage2e-validate-%j.err

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
CONTAINER="/scratch/network/km6349/skexgen.sif"
PYTHON="/root/miniconda3/bin/python3.8"
: "${REVIEWED_COMMIT:?set REVIEWED_COMMIT to the reviewed Stage 2E correction}"
: "${SLURM_JOB_ID:?this workflow must run under Slurm}"

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPYCACHEPREFIX="/tmp/v2-stage2e-validate-${SLURM_JOB_ID}"

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
printf '{"event":"slurm_start","branch":"%s","commit":"%s","job_id":"%s","container":"%s","training_partition":"synthetic_train_fixture","validation_partition":"none","test_partition_accessed":false}\n' \
  "$BRANCH" "$COMMIT" "$SLURM_JOB_ID" "$CONTAINER"

container_python - <<'PY'
import ast
import json
from pathlib import Path
import sys
import torch
import unittest

assert sys.version_info[:3] == (3, 8, 13)
assert torch.__version__.split("+")[0] == "1.11.0"
assert not torch.cuda.is_available()
for name in (
    "prototype/flat_baseline/constrained_v2_training_config.py",
    "prototype/flat_baseline/constrained_v2_training.py",
    "prototype/flat_baseline/tests/test_constrained_v2_training.py",
):
    path = Path(name)
    compile(path.read_text(encoding="utf-8"), str(path), "exec")
    ast.parse(
        path.read_text(encoding="utf-8"),
        filename=str(path),
        feature_version=(3, 8),
    )
suite = unittest.defaultTestLoader.loadTestsFromName(
    "prototype.flat_baseline.tests.test_constrained_v2_training"
)
result = unittest.TextTestRunner(verbosity=2).run(suite)
record = {
    "event": "focused_stage2e_validation",
    "python_version": sys.version.split()[0],
    "pytorch_version": torch.__version__,
    "execution_device": "cpu",
    "tests_run": result.testsRun,
    "failures": len(result.failures),
    "errors": len(result.errors),
    "skipped": len(result.skipped),
    "production_steps_under_test": 200,
    "vq_policy": "unchanged_production_ema",
    "test_partition_accessed": False,
}
print(json.dumps(record, sort_keys=True, separators=(",", ":")))
if not result.wasSuccessful() or result.skipped:
    raise SystemExit(1)
PY

printf '{"event":"slurm_terminal_success","commit":"%s","job_id":"%s","production_steps_under_test":200,"skipped":0,"test_partition_accessed":false}\n' \
  "$REVIEWED_COMMIT" "$SLURM_JOB_ID"
