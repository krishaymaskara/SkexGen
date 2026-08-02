#!/bin/bash
#SBATCH --job-name=graph-v1-iid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/network/km6349/graph-v1-iid-%j.out
#SBATCH --error=/scratch/network/km6349/graph-v1-iid-%j.err

set -Eeuo pipefail

terminal_failure() {
  status=$?
  trap - ERR
  printf '{"event":"slurm_terminal_failure","commit":"%s","job_id":"%s","exit_status":%s,"systematic_partition_accessed":false,"test_partition_accessed":false}\n' \
    "${REVIEWED_COMMIT:-unresolved}" "${SLURM_JOB_ID:-unresolved}" "$status" >&2
  exit "$status"
}
trap terminal_failure ERR

REPOSITORY="/scratch/network/km6349/SkexGen"
CORPUS="/scratch/network/km6349/controlled_corpora/b0-pilot-680-seed2026"
RUN_ROOT="/scratch/network/km6349/constrained_graph_v1_runs"
CONTAINER="/scratch/network/km6349/skexgen.sif"
PYTHON="/root/miniconda3/bin/python3.8"
: "${REVIEWED_COMMIT:?set REVIEWED_COMMIT to the reviewed graph V1 commit}"
: "${SLURM_JOB_ID:?this workflow must run under Slurm}"
OUTPUT="${RUN_ROOT}/iid-pilot-${REVIEWED_COMMIT}-${SLURM_JOB_ID}"

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPYCACHEPREFIX="/tmp/graph-v1-${SLURM_JOB_ID}"

container_python() {
  apptainer exec --cleanenv \
    --env CUDA_VISIBLE_DEVICES="" \
    --env OMP_NUM_THREADS=1 \
    --env MKL_NUM_THREADS=1 \
    --env PYTHONPYCACHEPREFIX="$PYTHONPYCACHEPREFIX" \
    --bind /scratch/network/km6349:/scratch/network/km6349 \
    "$CONTAINER" "$PYTHON" "$@"
}

cd "$REPOSITORY"
test "$(git branch --show-current)" = "graph-profile-decoder"
test "$(git rev-parse HEAD)" = "$REVIEWED_COMMIT"
test "$(git rev-parse origin/graph-profile-decoder)" = "$REVIEWED_COMMIT"
test -z "$(git status --porcelain=v1 --untracked-files=all)"
test -d "$CORPUS"
test ! -e "$OUTPUT"
test ! -L "$OUTPUT"

container_python - <<'PY'
import sys
import torch
assert sys.version_info[:3] == (3, 8, 13), "Python 3.8.13 required"
assert torch.__version__.split("+")[0] == "1.11.0"
assert torch.version.cuda == "11.3"
assert not torch.cuda.is_available()
print("python_version={}".format(sys.version.split()[0]))
print("pytorch_version={}".format(torch.__version__))
print("cuda_build={}".format(torch.version.cuda))
print("execution_device=cpu")
PY

container_python -B - <<'PY'
import sys
import unittest
suite = unittest.defaultTestLoader.discover(
    "prototype/graph_baseline/tests", pattern="test_*.py"
)
result = unittest.TextTestRunner(verbosity=2).run(suite)
print("focused_graph_tests_run={}".format(result.testsRun))
print("focused_graph_tests_skipped={}".format(len(result.skipped)))
if not result.wasSuccessful() or result.skipped:
    sys.exit(1)
PY

container_python -B -m prototype.graph_baseline.run_graph_v1_pilot \
  --corpus-dir "$CORPUS" \
  --output-dir "$OUTPUT"

test -f "$OUTPUT/epoch-0001.pt"
test -f "$OUTPUT/epoch-0002.pt"
test -f "$OUTPUT/metrics.jsonl"
sha256sum "$OUTPUT"/epoch-*.pt
container_python - "$OUTPUT/metrics.jsonl" <<'PY'
import json
import sys
rows = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8") if line.strip()]
assert rows[0]["event"] == "run_metadata"
terminal = rows[-1]
assert terminal["event"] == "terminal_success"
assert terminal["global_step"] == 136
assert terminal["examples_processed"] == 1088
for name in (
    "graph_representation_contract_satisfied",
    "graph_predictions_finite",
    "graph_converter_outcomes_classified",
):
    assert terminal["acceptance"][name] is True
assert terminal["systematic_partition_accessed"] is False
assert terminal["test_partition_accessed"] is False
print("graph_pilot_metrics_contract=ok")
PY

printf '{"event":"slurm_terminal_success","commit":"%s","job_id":"%s","systematic_partition_accessed":false,"test_partition_accessed":false}\n' \
  "$REVIEWED_COMMIT" "$SLURM_JOB_ID"
