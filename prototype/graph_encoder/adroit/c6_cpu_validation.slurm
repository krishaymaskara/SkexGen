#!/bin/bash
#SBATCH --job-name=ge1-c6-cpu-validate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/network/km6349/c6_validation_runs/c6-cpu-%j.out
#SBATCH --error=/scratch/network/km6349/c6_validation_runs/c6-cpu-%j.err

set -Eeuo pipefail

terminal_failure() {
  status=$?
  trap - ERR
  printf '{"event":"c6_cpu_validation_terminal_failure","commit":"%s","job_id":"%s","exit_status":%s,"operation_template_manifest_accessed":"not_confirmed_on_failure","operation_template_train_payload_accessed":"not_confirmed_on_failure","development_accessed":false,"systematic_rr_accessed":false,"test_er_accessed":false,"iid_accessed":false,"history_depth_accessed":false,"geometry_extrapolation_accessed":false,"scientific_training_performed":false,"engineering_smoke_completed":false,"c7_or_later_performed":false}\n' \
    "${EXPECTED_COMMIT:-unresolved}" "${SLURM_JOB_ID:-unresolved}" "$status" >&2
  exit "$status"
}
trap terminal_failure ERR

: "${EXPECTED_COMMIT:?submit with --export=ALL,EXPECTED_COMMIT=<exact-C6-commit>}"
REPOSITORY="${REPOSITORY:-/scratch/network/km6349/SkexGen-c6}"
CORPUS_DIR="${CORPUS_DIR:-/scratch/network/km6349/controlled_corpora/b0-pilot-680-seed2026}"
RUN_ROOT="${RUN_ROOT:-/scratch/network/km6349/c6_validation_runs/c6-${SLURM_JOB_ID}}"
CONTAINER="/scratch/network/km6349/skexgen.sif"
PYTHON="/root/miniconda3/bin/python3.8"
: "${SLURM_JOB_ID:?this workflow must run under Slurm}"

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPYCACHEPREFIX="/tmp/ge1-c6-cpu-${SLURM_JOB_ID}"
export TMPDIR="/tmp/ge1-c6-cpu-tmp-${SLURM_JOB_ID}"
mkdir -p "$TMPDIR" "$RUN_ROOT"

container_python() {
  apptainer exec \
    --cleanenv \
    --env CUDA_VISIBLE_DEVICES="" \
    --env OMP_NUM_THREADS=1 \
    --env MKL_NUM_THREADS=1 \
    --env PYTHONPYCACHEPREFIX="$PYTHONPYCACHEPREFIX" \
    --env TMPDIR="$TMPDIR" \
    --bind "$REPOSITORY:$REPOSITORY" \
    --bind "$CORPUS_DIR:$CORPUS_DIR" \
    --bind "$RUN_ROOT:$RUN_ROOT" \
    "$CONTAINER" "$PYTHON" "$@"
}

test -d "$REPOSITORY/.git"
test -d "$CORPUS_DIR"
test -f "$CONTAINER"
cd "$REPOSITORY"

CHECKED_OUT_COMMIT="$(git rev-parse HEAD)"
SOURCE_STATUS="$(git status --porcelain=v1 --untracked-files=all)"
test "$CHECKED_OUT_COMMIT" = "$EXPECTED_COMMIT"
test -z "$SOURCE_STATUS"

printf '{"event":"c6_cpu_validation_start","commit":"%s","job_id":"%s","host":"%s","container":"%s","source_clean":true,"execution_device":"cpu","authorized_partition":"operation_template.train","development_accessed":false,"systematic_rr_accessed":false,"test_er_accessed":false,"iid_accessed":false,"history_depth_accessed":false,"geometry_extrapolation_accessed":false,"scientific_training_performed":false,"c7_or_later_performed":false}\n' \
  "$CHECKED_OUT_COMMIT" "$SLURM_JOB_ID" "$(hostname)" "$CONTAINER"

container_python -B - <<'PY'
import json
import platform
import sys
import torch

assert sys.version_info[:2] == (3, 8), "Python 3.8 required"
assert torch.__version__.split("+")[0] == "1.11.0", "PyTorch 1.11.0 required"
assert not torch.cuda.is_available(), "CUDA must be unavailable"
print(json.dumps({
    "event": "c6_cpu_validation_environment",
    "python_version": sys.version.split()[0],
    "python_implementation": platform.python_implementation(),
    "pytorch_version": torch.__version__,
    "pytorch_cuda_build": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "device": torch.device("cpu").type,
    "cpu_threads": torch.get_num_threads(),
}, sort_keys=True, separators=(",", ":")), flush=True)
PY

container_python -B - <<'PY'
import json
import unittest

names = (
    "prototype.graph_encoder.tests.test_c6_contract",
    "prototype.graph_encoder.tests.test_c6_runtime",
)
suite = unittest.TestSuite(
    unittest.defaultTestLoader.loadTestsFromName(name) for name in names
)
result = unittest.TextTestRunner(verbosity=2).run(suite)
print(json.dumps({
    "event": "c6_focused_suite",
    "tests_run": result.testsRun,
    "failures": len(result.failures),
    "errors": len(result.errors),
    "skipped": len(result.skipped),
}, sort_keys=True, separators=(",", ":")), flush=True)
if result.testsRun < 33 or not result.wasSuccessful() or result.skipped:
    raise SystemExit("focused C6 suite must pass with zero skips")
PY

container_python -B - <<'PY'
import json
import unittest

suite = unittest.defaultTestLoader.discover("prototype/graph_encoder/tests")
result = unittest.TextTestRunner(verbosity=1).run(suite)
print(json.dumps({
    "event": "c6_complete_graph_encoder_suite",
    "tests_run": result.testsRun,
    "failures": len(result.failures),
    "errors": len(result.errors),
    "skipped": len(result.skipped),
}, sort_keys=True, separators=(",", ":")), flush=True)
if result.testsRun < 143 or not result.wasSuccessful() or result.skipped:
    raise SystemExit("complete graph_encoder suite must pass with zero skips")
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

paths = tuple(sorted(Path("prototype/graph_encoder").glob("*.py"))) + tuple(
    sorted(Path("prototype/graph_encoder/tests").glob("test_c6_*.py"))
)
for path in paths:
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path), feature_version=(3, 8))
print("C6_PYTHON38_GRAMMAR_OK {}".format(len(paths)), flush=True)
PY

container_python -B - <<'PY'
import ast
from pathlib import Path

source = Path("prototype/graph_encoder/autonomous.py").read_text(encoding="utf-8")
tree = ast.parse(source)
for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name in (
        "autonomous_input_from_paired", "run_autonomous_evaluation",
        "_decode_condition",
    ):
        arguments = [item.arg for item in node.args.args + node.args.kwonlyargs]
        assert "target" not in arguments
        assert "operation_sequence" not in arguments
helper = next(
    node for node in tree.body
    if isinstance(node, ast.FunctionDef) and node.name == "_decode_condition"
)
helper_source = ast.get_source_segment(source, helper)
assert helper_source.count("model.decoder(") == 1
assert "encoder ==" not in helper_source
assert "teacher_forced" not in helper_source
print("C6_TARGET_LEAKAGE_AND_COMMON_DECODER_AUDIT=PASS", flush=True)
PY

for arm in flat typed_graph; do
  printf 'c6_two_epoch_train_only_smoke_%s=START\n' "$arm"
  container_python -B -m prototype.graph_encoder.c6_smoke \
    --corpus-dir "$CORPUS_DIR" \
    --output-dir "$RUN_ROOT/$arm" \
    --repository-root "$REPOSITORY" \
    --expected-commit "$EXPECTED_COMMIT" \
    --arm "$arm" \
    --seed 2026
  test -s "$RUN_ROOT/$arm/c6_metrics.json"
  printf 'c6_two_epoch_train_only_smoke_%s=PASS\n' "$arm"
done

git diff --check
test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git status --porcelain=v1 --untracked-files=all)"

printf '{"event":"c6_cpu_validation_terminal_success","commit":"%s","job_id":"%s","python":"3.8","pytorch":"1.11.0","execution_device":"cpu","focused_c6_minimum_tests":33,"complete_graph_encoder_minimum_tests":143,"all_pytorch_related_skips":0,"two_epoch_train_only_smoke_arms":["flat","typed_graph"],"complete_metrics_records":2,"operation_template_manifest_accessed":true,"operation_template_train_payload_accessed":true,"development_accessed":false,"systematic_rr_accessed":false,"test_er_accessed":false,"iid_accessed":false,"history_depth_accessed":false,"geometry_extrapolation_accessed":false,"scientific_training_performed":false,"engineering_smoke_performed":true,"c7_or_later_performed":false,"run_root":"%s"}\n' \
  "$EXPECTED_COMMIT" "$SLURM_JOB_ID" "$RUN_ROOT"
