#!/bin/bash
#SBATCH --job-name=ge1-c3-cpu-validate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/network/km6349/c3_validation_runs/c3-cpu-%j.out
#SBATCH --error=/scratch/network/km6349/c3_validation_runs/c3-cpu-%j.err

set -Eeuo pipefail

terminal_failure() {
  status=$?
  trap - ERR
  printf '{"event":"c3_cpu_validation_terminal_failure","commit":"%s","job_id":"%s","exit_status":%s,"authoritative_corpus_manifest_accessed":false,"authoritative_cad_history_payload_accessed":false,"synthetic_manifest_fixture_tests":true,"systematic_partition_accessed":false,"test_partition_accessed":false,"training_performed":false}\n' \
    "${EXPECTED_COMMIT:-unresolved}" "${SLURM_JOB_ID:-unresolved}" "$status" >&2
  exit "$status"
}
trap terminal_failure ERR

EXPECTED_COMMIT="a85ad3a23a6587cbedad8a6693b6117c1edfacc7"
REPOSITORY="/scratch/network/km6349/SkexGen-c3-a85ad3a"
CONTAINER="/scratch/network/km6349/skexgen.sif"
PYTHON="/root/miniconda3/bin/python3.8"
: "${SLURM_JOB_ID:?this workflow must run under Slurm}"

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPYCACHEPREFIX="/tmp/ge1-c3-cpu-${SLURM_JOB_ID}"
export TMPDIR="/tmp/ge1-c3-cpu-tmp-${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"

container_python() {
  apptainer exec \
    --cleanenv \
    --env CUDA_VISIBLE_DEVICES="" \
    --env OMP_NUM_THREADS=1 \
    --env MKL_NUM_THREADS=1 \
    --env PYTHONPYCACHEPREFIX="$PYTHONPYCACHEPREFIX" \
    --env TMPDIR="$TMPDIR" \
    --bind "$REPOSITORY:$REPOSITORY" \
    "$CONTAINER" "$PYTHON" "$@"
}

test -d "$REPOSITORY/.git"
test -f "$CONTAINER"
cd "$REPOSITORY"

CHECKED_OUT_COMMIT="$(git rev-parse HEAD)"
SOURCE_STATUS="$(git status --porcelain=v1 --untracked-files=all)"
test "$CHECKED_OUT_COMMIT" = "$EXPECTED_COMMIT"
test -z "$SOURCE_STATUS"

printf '{"event":"c3_cpu_validation_start","commit":"%s","job_id":"%s","host":"%s","container":"%s","source_clean":true,"execution_device":"cpu","authoritative_corpus_manifest_accessed":false,"authoritative_cad_history_payload_accessed":false,"synthetic_manifest_fixture_tests":true,"systematic_partition_accessed":false,"test_partition_accessed":false,"training_performed":false}\n' \
  "$CHECKED_OUT_COMMIT" "$SLURM_JOB_ID" "$(hostname)" "$CONTAINER"
printf 'checked_out_commit=%s\n' "$CHECKED_OUT_COMMIT"

container_python -B - <<'PY'
import json
import platform
import sys

import torch

assert sys.version_info[:3] == (3, 8, 13), "Python 3.8.13 required"
assert torch.__version__.split("+")[0] == "1.11.0", "PyTorch 1.11.0 required"
assert not torch.cuda.is_available(), "CUDA must be unavailable for CPU validation"
device = torch.device("cpu")
assert device.type == "cpu"
record = {
    "event": "c3_cpu_validation_environment",
    "python_version": sys.version.split()[0],
    "python_implementation": platform.python_implementation(),
    "pytorch_version": torch.__version__,
    "pytorch_cuda_build": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "device": device.type,
    "cpu_threads": torch.get_num_threads(),
}
print(json.dumps(record, sort_keys=True, separators=(",", ":")), flush=True)
PY

printf 'c3_graph_encoder_suite=START\n'
container_python -B -m unittest discover prototype/graph_encoder/tests
printf 'c3_graph_encoder_suite=PASS\n'

# unittest normally treats a skipped test as a successful command. Inspect the
# real-PyTorch C3 smoke result directly and require exactly one unskipped pass.
container_python -B - <<'PY'
import json
import sys
import unittest

name = (
    "prototype.graph_encoder.tests.test_batching."
    "BoundaryAndTensorTests."
    "test_cpu_real_tensor_smoke_keeps_boundaries_separate"
)
suite = unittest.defaultTestLoader.loadTestsFromName(name)
result = unittest.TextTestRunner(verbosity=2).run(suite)
record = {
    "event": "c3_real_pytorch_smoke",
    "tests_run": result.testsRun,
    "failures": len(result.failures),
    "errors": len(result.errors),
    "skipped": len(result.skipped),
}
print(json.dumps(record, sort_keys=True, separators=(",", ":")), flush=True)
if (
    result.testsRun != 1
    or not result.wasSuccessful()
    or result.skipped
):
    raise SystemExit("C3 real-PyTorch smoke did not execute as one unskipped pass")
PY

printf 'model_data_regression_suite=START\n'
container_python -B -m unittest \
  prototype.model_data.tests.test_compatibility \
  prototype.model_data.tests.test_constrained_profile_decoder \
  prototype.model_data.tests.test_profile_geometry \
  prototype.model_data.tests.test_profile_geometry_torch
printf 'model_data_regression_suite=PASS\n'

printf 'graph_encoder_compileall=START\n'
container_python -B -m compileall prototype/graph_encoder
printf 'graph_encoder_compileall=PASS\n'

test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git status --porcelain=v1 --untracked-files=all)"

printf '{"event":"c3_cpu_validation_terminal_success","commit":"%s","job_id":"%s","python":"3.8.13","pytorch":"1.11.0","execution_device":"cpu","real_pytorch_smoke_skipped":false,"authoritative_corpus_manifest_accessed":false,"authoritative_cad_history_payload_accessed":false,"synthetic_manifest_fixture_tests":true,"systematic_partition_accessed":false,"test_partition_accessed":false,"training_performed":false}\n' \
  "$EXPECTED_COMMIT" "$SLURM_JOB_ID"
