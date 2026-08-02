#!/bin/bash
#SBATCH --job-name=v6-iid-pilot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/network/km6349/v6-iid-pilot-%j.out
#SBATCH --error=/scratch/network/km6349/v6-iid-pilot-%j.err

set -Eeuo pipefail

terminal_failure() {
  status=$?
  trap - ERR
  printf '{"event":"slurm_terminal_failure","commit":"%s","job_id":"%s","exit_status":%s,"training_partition":"train","validation_partition":"iid_validation","systematic_partition_accessed":false,"test_partition_accessed":false}\n' \
    "${REVIEWED_COMMIT:-unresolved}" "${SLURM_JOB_ID:-unresolved}" "$status" >&2
  exit "$status"
}
trap terminal_failure ERR

REPOSITORY="/scratch/network/km6349/SkexGen"
CORPUS="/scratch/network/km6349/controlled_corpora/b0-pilot-680-seed2026"
RUN_ROOT="/scratch/network/km6349/constrained_v6_runs"
CONTAINER="/scratch/network/km6349/skexgen.sif"
PYTHON="/root/miniconda3/bin/python3.8"
: "${REVIEWED_COMMIT:?set REVIEWED_COMMIT to the reviewed V6 constrained-axis commit}"
: "${SLURM_JOB_ID:?this workflow must run under Slurm}"

OUTPUT="${RUN_ROOT}/iid-pilot-${REVIEWED_COMMIT}-${SLURM_JOB_ID}"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPYCACHEPREFIX="/tmp/v6-iid-pilot-${SLURM_JOB_ID}"

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
test ! -e "$OUTPUT"
test ! -L "$OUTPUT"
printf '{"event":"slurm_start","branch":"%s","commit":"%s","job_id":"%s","host":"%s","container":"%s","training_partition":"train","validation_partition":"iid_validation","systematic_partition_accessed":false,"test_partition_accessed":false}\n' \
  "$BRANCH" "$COMMIT" "$SLURM_JOB_ID" "$HOST" "$CONTAINER"

container_python - <<'PY'
import sys
import torch
assert sys.version_info[:3] == (3, 8, 13)
assert torch.__version__.split("+")[0] == "1.11.0"
assert torch.version.cuda == "11.3"
assert not torch.cuda.is_available()
print("python_version={}".format(sys.version.split()[0]))
print("pytorch_version={}".format(torch.__version__))
print("pytorch_cuda_build={}".format(torch.version.cuda))
print("execution_device=cpu")
PY

container_python -B - <<'PY'
import sys
import unittest
suite = unittest.defaultTestLoader.loadTestsFromNames([
    "prototype.flat_baseline.tests.test_constrained_v6",
    "prototype.flat_baseline.tests.test_constrained_v6_pilot",
])
result = unittest.TextTestRunner(verbosity=2).run(suite)
print("focused_v6_tests_run={}".format(result.testsRun))
print("focused_v6_tests_skipped={}".format(len(result.skipped)))
if not result.wasSuccessful() or result.skipped:
    sys.exit(1)
PY

container_python -B -m prototype.flat_baseline.run_constrained_v6_pilot \
  --corpus-dir "$CORPUS" \
  --output-dir "$OUTPUT" \
  --device cpu

test -f "$OUTPUT/epoch-0002.pt"
test -f "$OUTPUT/metrics.jsonl"
sha256sum "$OUTPUT"/epoch-*.pt
container_python - "$OUTPUT/metrics.jsonl" <<'PY'
import json
import sys
records = [
    json.loads(line)
    for line in open(sys.argv[1], encoding="utf-8")
    if line.strip()
]
metadata = records[0]
terminal = records[-1]
assert metadata["event"] == "run_metadata"
assert metadata["training_partition"]["family_count"] == 544
assert metadata["validation_partition"]["family_count"] == 68
assert metadata["configured_maximum_steps"] == 136
assert terminal["event"] == "terminal_success"
assert terminal["completed_epochs"] == 2
assert terminal["global_step"] == 136
assert terminal["examples_processed"] == 1088
assert terminal["acceptance"]["constrained_node_grammar_contract_satisfied"] is True
assert terminal["acceptance"]["constrained_axis_geometry_contract_satisfied"] is True
assert terminal["systematic_partition_accessed"] is False
assert terminal["test_partition_accessed"] is False
print("pilot_metrics_contract=ok")
PY
printf '{"event":"slurm_terminal_success","commit":"%s","job_id":"%s","training_partition":"train","validation_partition":"iid_validation","systematic_partition_accessed":false,"test_partition_accessed":false}\n' \
  "$REVIEWED_COMMIT" "$SLURM_JOB_ID"
