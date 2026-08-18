#!/bin/bash
#SBATCH --job-name=ge1-stage6-structure-only
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --signal=B:TERM@60
#SBATCH --output=/scratch/network/km6349/ge1_stage6_structure_only_runs/stage6-structure-%j.out
#SBATCH --error=/scratch/network/km6349/ge1_stage6_structure_only_runs/stage6-structure-%j.err

set -Eeuo pipefail

terminal_failure() {
  status=$?
  trap - ERR
  printf '{"event":"stage6_structure_only_terminal_failure","commit":"%s","job_id":"%s","exit_status":%s,"scientific_execution_authorized_by_implementation":false,"rr_accessed":false,"er_accessed":false,"iid_accessed":false,"history_depth_accessed":false,"geometry_extrapolation_accessed":false,"cad_kernel_executed":false,"stage7_authorized":false}\n' \
    "${EXPECTED_COMMIT:-unresolved}" "${SLURM_JOB_ID:-unresolved}" "$status" >&2
  exit "$status"
}

terminal_timeout() {
  trap - ERR
  printf '{"event":"stage6_structure_only_terminal_timeout","commit":"%s","job_id":"%s","exit_status":124,"scientific_execution_authorized_by_implementation":false,"rr_accessed":false,"er_accessed":false,"iid_accessed":false,"history_depth_accessed":false,"geometry_extrapolation_accessed":false,"cad_kernel_executed":false,"stage7_authorized":false}\n' \
    "${EXPECTED_COMMIT:-unresolved}" "${SLURM_JOB_ID:-unresolved}" >&2
  exit 124
}

trap terminal_failure ERR
trap terminal_timeout TERM

: "${EXPECTED_COMMIT:?future execution requires a separately authorized exact commit}"
: "${SLURM_JOB_ID:?this runner must execute under Slurm}"
: "${STAGE6_PRODUCER_ARTIFACT:?bind a verified producer artifact directory}"

REPOSITORY="${REPOSITORY:-/scratch/network/km6349/SkexGen-stage6-structure-only}"
RUN_PARENT="${RUN_PARENT:-/scratch/network/km6349/ge1_stage6_structure_only_artifacts}"
LOG_PARENT="/scratch/network/km6349/ge1_stage6_structure_only_runs"
CONTAINER="/scratch/network/km6349/skexgen.sif"
PYTHON="/root/miniconda3/bin/python3.8"
OUTPUT_DIR="$RUN_PARENT/ge1-stage6-structure-only-${EXPECTED_COMMIT}-${SLURM_JOB_ID}"

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPYCACHEPREFIX="/tmp/ge1-stage6-structure-only-${SLURM_JOB_ID}"
export TMPDIR="/tmp/ge1-stage6-structure-only-tmp-${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"

container_python() {
  apptainer exec \
    --cleanenv \
    --env SLURM_JOB_ID="$SLURM_JOB_ID" \
    --env CUDA_VISIBLE_DEVICES="" \
    --env OMP_NUM_THREADS=1 \
    --env MKL_NUM_THREADS=1 \
    --env PYTHONPYCACHEPREFIX="$PYTHONPYCACHEPREFIX" \
    --env TMPDIR="$TMPDIR" \
    --bind "$REPOSITORY:$REPOSITORY:ro" \
    --bind "$STAGE6_PRODUCER_ARTIFACT:$STAGE6_PRODUCER_ARTIFACT:ro" \
    --bind "$RUN_PARENT:$RUN_PARENT:rw" \
    "$CONTAINER" "$PYTHON" "$@"
}

test -d "$REPOSITORY/.git"
test ! -f "$REPOSITORY/.git"
test -d "$STAGE6_PRODUCER_ARTIFACT"
test -f "$STAGE6_PRODUCER_ARTIFACT/artifact_manifest.json"
test -f "$STAGE6_PRODUCER_ARTIFACT/SHA256SUMS"
test -d "$RUN_PARENT"
test -d "$LOG_PARENT"
test -f "$CONTAINER"
test ! -e "$OUTPUT_DIR"
test ! -L "$OUTPUT_DIR"
cd "$REPOSITORY"

test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git symbolic-ref -q --short HEAD || true)"
test -z "$(git status --porcelain=v1 --untracked-files=all)"
test "$(git rev-parse --show-toplevel)" = "$REPOSITORY"
test -z "$(git rev-parse --show-superproject-working-tree)"

printf '{"event":"stage6_structure_only_start","protocol":"GE1-STAGE6-STRUCTURE-ONLY-COMPARISON-v1","commit":"%s","job_id":"%s","host":"%s","source_clean":true,"standalone_checkout":true,"detached_head":true,"execution_device":"cpu","cpu_threads":1,"memory_gb":12,"wall_hours":2,"scientific_execution_authorized_by_implementation":false,"rr_accessed":false,"er_accessed":false,"iid_accessed":false,"history_depth_accessed":false,"geometry_extrapolation_accessed":false,"cad_kernel_executed":false,"stage7_authorized":false}\n' \
  "$EXPECTED_COMMIT" "$SLURM_JOB_ID" "$(hostname)"

container_python -B - <<'PY'
import json
import platform
import sys
import torch
assert sys.version_info[:3] == (3, 8, 13)
assert torch.__version__.split("+")[0] == "1.11.0"
assert not torch.cuda.is_available()
assert torch.get_num_threads() == 1
print(json.dumps({
    "event": "stage6_structure_only_environment",
    "python": platform.python_version(),
    "pytorch": torch.__version__,
    "cuda_available": False,
    "cpu_threads": torch.get_num_threads(),
}, sort_keys=True, separators=(",", ":")), flush=True)
PY

container_python -B - <<'PY'
import json
import unittest
names = (
    "prototype.graph_encoder.tests.test_stage6_structure_only_contract",
    "prototype.graph_encoder.tests.test_stage6_structure_only_runtime",
)
suite = unittest.TestSuite(
    unittest.defaultTestLoader.loadTestsFromName(name) for name in names
)
declared = suite.countTestCases()
result = unittest.TextTestRunner(verbosity=2).run(suite)
print(json.dumps({
    "event": "stage6_structure_only_focused_tests",
    "declared_tests": declared,
    "tests_run": result.testsRun,
    "failures": len(result.failures),
    "errors": len(result.errors),
    "skipped": len(result.skipped),
}, sort_keys=True, separators=(",", ":")), flush=True)
if result.testsRun != declared or not result.wasSuccessful() or result.skipped:
    raise SystemExit("all focused Stage 6 tests must pass with zero skips")
PY

container_python -B - <<'PY'
import unittest
suite = unittest.TestLoader().discover("prototype/graph_encoder/tests", top_level_dir=".")
result = unittest.TextTestRunner(verbosity=1).run(suite)
if not result.wasSuccessful() or result.skipped:
    raise SystemExit("complete graph-encoder discovery must pass with zero skips")
PY

container_python -B tools/check_documentation.py
container_python -B -m compileall -q prototype/graph_encoder
container_python -B - <<'PY'
import ast
from pathlib import Path
paths = tuple(sorted(Path("prototype/graph_encoder").glob("*.py"))) + tuple(
    sorted(Path("prototype/graph_encoder/tests").glob("test_*.py"))
)
for path in paths:
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path), feature_version=(3, 8))
print("GE1_STAGE6_STRUCTURE_ONLY_PYTHON38_AST=PASS {}".format(len(paths)), flush=True)
PY
container_python -B -m prototype.graph_encoder.stage6_structure_only_audit \
  prototype/graph_encoder/adroit/ge1_stage6_structure_only_cpu.slurm
bash -n prototype/graph_encoder/adroit/ge1_stage6_structure_only_cpu.slurm
git show --check "$EXPECTED_COMMIT" --

container_python -B -m prototype.graph_encoder.stage6_structure_only \
  --producer-artifact "$STAGE6_PRODUCER_ARTIFACT" \
  --output-dir "$OUTPUT_DIR" \
  --expected-commit "$EXPECTED_COMMIT" \
  --job-id "$SLURM_JOB_ID"

test "$(find "$OUTPUT_DIR" -type f | wc -l | tr -d ' ')" = 5
(
  cd "$OUTPUT_DIR"
  sha256sum -c SHA256SUMS
)
container_python -B - "$OUTPUT_DIR" "$EXPECTED_COMMIT" "$SLURM_JOB_ID" <<'PY'
import json
import sys
from prototype.graph_encoder.stage6_structure_only import verify_artifact
print(json.dumps(verify_artifact(
    sys.argv[1], expected_commit=sys.argv[2], expected_job_id=sys.argv[3]
), sort_keys=True, separators=(",", ":")), flush=True)
PY

test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git symbolic-ref -q --short HEAD || true)"
test -z "$(git status --porcelain=v1 --untracked-files=all)"

printf '{"event":"stage6_structure_only_terminal_success","protocol":"GE1-STAGE6-STRUCTURE-ONLY-COMPARISON-v1","commit":"%s","job_id":"%s","artifact":"%s","scientific_execution_authorized_by_implementation":false,"rr_accessed":false,"er_accessed":false,"iid_accessed":false,"history_depth_accessed":false,"geometry_extrapolation_accessed":false,"cad_kernel_executed":false,"stage7_authorized":false}\n' \
  "$EXPECTED_COMMIT" "$SLURM_JOB_ID" "$OUTPUT_DIR"
