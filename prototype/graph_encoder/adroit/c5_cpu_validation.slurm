#!/bin/bash
#SBATCH --job-name=ge1-c5-cpu-validate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/network/km6349/c5_validation_runs/c5-cpu-%j.out
#SBATCH --error=/scratch/network/km6349/c5_validation_runs/c5-cpu-%j.err

set -Eeuo pipefail

terminal_failure() {
  status=$?
  trap - ERR
  printf '{"event":"c5_cpu_validation_terminal_failure","commit":"%s","job_id":"%s","exit_status":%s,"corpus_manifest_accessed":false,"corpus_cad_history_payload_accessed":false,"synthetic_manifest_fixture_tests":true,"procedural_model_fixtures":true,"systematic_partition_accessed":false,"test_partition_accessed":false,"scientific_training_performed":false,"regression_optimizer_smokes_may_execute":true,"c7_performed":false,"c8_performed":false}\n' \
    "${EXPECTED_COMMIT:-unresolved}" "${SLURM_JOB_ID:-unresolved}" "$status" >&2
  exit "$status"
}
trap terminal_failure ERR

: "${EXPECTED_COMMIT:?submit with --export=ALL,EXPECTED_COMMIT=<exact-C5-commit>}"
REPOSITORY="${REPOSITORY:-/scratch/network/km6349/SkexGen-c5}"
CONTAINER="/scratch/network/km6349/skexgen.sif"
PYTHON="/root/miniconda3/bin/python3.8"
: "${SLURM_JOB_ID:?this workflow must run under Slurm}"

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPYCACHEPREFIX="/tmp/ge1-c5-cpu-${SLURM_JOB_ID}"
export TMPDIR="/tmp/ge1-c5-cpu-tmp-${SLURM_JOB_ID}"
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

printf '{"event":"c5_cpu_validation_start","commit":"%s","job_id":"%s","host":"%s","container":"%s","source_clean":true,"execution_device":"cpu","corpus_manifest_accessed":false,"corpus_cad_history_payload_accessed":false,"synthetic_manifest_fixture_tests":true,"procedural_model_fixtures":true,"systematic_partition_accessed":false,"test_partition_accessed":false,"scientific_training_performed":false,"regression_optimizer_smokes_may_execute":true,"c7_performed":false,"c8_performed":false}\n' \
  "$CHECKED_OUT_COMMIT" "$SLURM_JOB_ID" "$(hostname)" "$CONTAINER"
printf 'checked_out_commit=%s\n' "$CHECKED_OUT_COMMIT"

container_python -B - <<'PY'
import json
import platform
import sys

import torch

assert sys.version_info[:2] == (3, 8), "Python 3.8 required"
assert torch.__version__.split("+")[0] == "1.11.0", "PyTorch 1.11.0 required"
assert not torch.cuda.is_available(), "CUDA must be unavailable for CPU validation"
record = {
    "event": "c5_cpu_validation_environment",
    "python_version": sys.version.split()[0],
    "python_implementation": platform.python_implementation(),
    "pytorch_version": torch.__version__,
    "pytorch_cuda_build": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "device": torch.device("cpu").type,
    "cpu_threads": torch.get_num_threads(),
}
print(json.dumps(record, sort_keys=True, separators=(",", ":")), flush=True)
PY

container_python -B - <<'PY'
import json
import unittest

names = (
    "prototype.graph_encoder.tests.test_c5_contract",
    "prototype.graph_encoder.tests.test_c5_shared_decoder",
)
suite = unittest.TestSuite(
    unittest.defaultTestLoader.loadTestsFromName(name) for name in names
)
result = unittest.TextTestRunner(verbosity=2).run(suite)
record = {
    "event": "c5_focused_suite",
    "tests_run": result.testsRun,
    "failures": len(result.failures),
    "errors": len(result.errors),
    "skipped": len(result.skipped),
}
print(json.dumps(record, sort_keys=True, separators=(",", ":")), flush=True)
if result.testsRun != 22 or not result.wasSuccessful() or result.skipped:
    raise SystemExit("focused C5 suite must pass 22/22 with zero skips")
PY

container_python -B - <<'PY'
import json
import unittest

suite = unittest.defaultTestLoader.discover("prototype/graph_encoder/tests")
result = unittest.TextTestRunner(verbosity=1).run(suite)
record = {
    "event": "c5_complete_graph_encoder_suite",
    "tests_run": result.testsRun,
    "failures": len(result.failures),
    "errors": len(result.errors),
    "skipped": len(result.skipped),
}
print(json.dumps(record, sort_keys=True, separators=(",", ":")), flush=True)
if result.testsRun != 110 or not result.wasSuccessful() or result.skipped:
    raise SystemExit("complete graph-encoder suite must pass 110/110 with zero skips")
PY

printf 'model_data_regression_suite=START\n'
container_python -B -m unittest discover prototype/model_data/tests
printf 'model_data_regression_suite=PASS\n'

printf 'flat_baseline_regression_suite=START\n'
container_python -B -m unittest discover prototype/flat_baseline/tests
printf 'flat_baseline_regression_suite=PASS\n'

printf 'graph_baseline_regression_suite=START\n'
container_python -B -m unittest discover prototype/graph_baseline/tests
printf 'graph_baseline_regression_suite=PASS\n'

printf 'representation_regression_suite=START\n'
container_python -B -m unittest discover prototype/representation/tests
printf 'representation_regression_suite=PASS\n'

printf 'controlled_data_regression_suite=START\n'
container_python -B -m unittest discover prototype/controlled_data/tests
printf 'controlled_data_regression_suite=PASS\n'

printf 'documentation_validation=START\n'
container_python -B tools/check_documentation.py
printf 'documentation_validation=PASS\n'

printf 'graph_encoder_compileall=START\n'
container_python -B -m compileall prototype/graph_encoder
printf 'graph_encoder_compileall=PASS\n'

container_python -B - <<'PY'
import ast
from pathlib import Path

paths = (
    Path("prototype/graph_encoder/checkpoint.py"),
    Path("prototype/graph_encoder/decoder_contract.py"),
    Path("prototype/graph_encoder/losses.py"),
    Path("prototype/graph_encoder/model.py"),
    Path("prototype/graph_encoder/shared_decoder.py"),
    Path("prototype/graph_encoder/tests/test_c5_contract.py"),
    Path("prototype/graph_encoder/tests/test_c5_shared_decoder.py"),
)
for path in paths:
    ast.parse(
        path.read_text(encoding="utf-8"),
        filename=str(path),
        feature_version=(3, 8),
    )
print("C5_PYTHON38_GRAMMAR_OK {}".format(len(paths)), flush=True)
PY

test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git status --porcelain=v1 --untracked-files=all)"

printf '{"event":"c5_cpu_validation_terminal_success","commit":"%s","job_id":"%s","python":"3.8","pytorch":"1.11.0","execution_device":"cpu","focused_c5_tests":22,"focused_c5_skips":0,"complete_graph_encoder_tests":110,"complete_graph_encoder_skips":0,"corpus_manifest_accessed":false,"corpus_cad_history_payload_accessed":false,"synthetic_manifest_fixture_tests":true,"procedural_model_fixtures":true,"systematic_partition_accessed":false,"test_partition_accessed":false,"scientific_training_performed":false,"regression_optimizer_smokes_may_execute":true,"c7_performed":false,"c8_performed":false}\n' \
  "$EXPECTED_COMMIT" "$SLURM_JOB_ID"
