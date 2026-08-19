#!/bin/bash
# Read-only ADR-0014 zero-memory diagnostic for producer job 3354961.
# No submission command is present; submission remains a separate reviewer act.
#SBATCH --job-name=ge1-s6-zero-memory
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/network/km6349/ge1_stage6_zero_memory_runs/zero-memory-%j.out
#SBATCH --error=/scratch/network/km6349/ge1_stage6_zero_memory_runs/zero-memory-%j.err
#SBATCH --signal=B:TERM@60

set -Eeuo pipefail

infrastructure_failure() {
  status="${1:-$?}"
  trap - ERR
  printf '{"event":"stage6_zero_memory_terminal_failure","commit":"%s","job_id":"%s","producer_job_id":"3354961","prior_postmortem_job_id":"3355134","exit_status":%s,"training_performed":false,"development_accessed":false}\n' \
    "${EXPECTED_COMMIT:-unresolved}" "${SLURM_JOB_ID:-unresolved}" "$status" >&2
  exit "$status"
}

terminal_timeout() {
  trap - ERR
  printf '{"event":"stage6_zero_memory_terminal_timeout","commit":"%s","job_id":"%s","producer_job_id":"3354961","prior_postmortem_job_id":"3355134","exit_status":124,"training_performed":false,"development_accessed":false}\n' \
    "${EXPECTED_COMMIT:-unresolved}" "${SLURM_JOB_ID:-unresolved}" >&2
  exit 124
}

trap infrastructure_failure ERR
trap terminal_timeout TERM

: "${EXPECTED_COMMIT:?exact reviewed diagnostic commit required}"
: "${EXPECTED_RUNNER_SHA256:?exact reviewed runner SHA-256 required}"
: "${SLURM_JOB_ID:?must run under a separately authorized Slurm allocation}"
: "${RETAINED_WORK_ROOT:?job 3354961 retained work root required}"
: "${TRAIN_INPUT_ROOT:?authorized narrow train package required}"
: "${POSTMORTEM_ARTIFACT:?exact job-3355134 artifact required}"
: "${RUN_PARENT:?diagnostic output parent required}"

PRODUCER_COMMIT="5f4542f86756dae44435af27a6e072db6f27a8ef"
PRODUCER_JOB_ID="3354961"
PRIOR_COMMIT="ddf9617fb124fbc18f38302943f0c982019bfee9"
PRIOR_JOB_ID="3355134"
PRIOR_SHA256SUMS_SHA256="927b6407542e8019c05ade07ce5c987462ab730bb7399e394e45c130c61e73f3"
EXPECTED_RETAINED_NAME="ge1-stage6-structure-only-producer-${PRODUCER_COMMIT}-${PRODUCER_JOB_ID}.work.incomplete-${PRODUCER_JOB_ID}"
EXPECTED_PRIOR_NAME="ge1-stage6-train-gate-postmortem-${PRIOR_COMMIT}-${PRIOR_JOB_ID}"
REPOSITORY="${REPOSITORY:-/scratch/network/km6349/SkexGen-stage6-structure-only}"
CONTAINER="${CONTAINER:-/scratch/network/km6349/skexgen.sif}"
PYTHON="${PYTHON:-/root/miniconda3/bin/python3.8}"
RUNNER="$REPOSITORY/prototype/graph_encoder/adroit/ge1_stage6_zero_memory_cpu.slurm"
TRAIN_INDEX="$TRAIN_INPUT_ROOT/index.json"
TRAIN_ROOT="$TRAIN_INPUT_ROOT/payloads"
OUTPUT_DIR="$RUN_PARENT/ge1-stage6-zero-memory-${EXPECTED_COMMIT}-${SLURM_JOB_ID}"

FLAT_2026="$RETAINED_WORK_ROOT/stage6-flat-seed2026.pt"
TYPED_GRAPH_2026="$RETAINED_WORK_ROOT/stage6-typed_graph-seed2026.pt"
FLAT_2027="$RETAINED_WORK_ROOT/stage6-flat-seed2027.pt"
TYPED_GRAPH_2027="$RETAINED_WORK_ROOT/stage6-typed_graph-seed2027.pt"
FLAT_2028="$RETAINED_WORK_ROOT/stage6-flat-seed2028.pt"
TYPED_GRAPH_2028="$RETAINED_WORK_ROOT/stage6-typed_graph-seed2028.pt"
FLAT_2026_SHA256="21c1573602704a86aa7d981ce71be0d1d8697c5ca96545e16268bfcf0ba30b2e"
TYPED_GRAPH_2026_SHA256="b1f24705ab71b319e5f4963ebd8d9cd84a10b5c104d0f80a1df03c5fca379bcc"
FLAT_2027_SHA256="6e0c6c499baa273caabad895768841099c00d46ac3abc2e9e8ff88cdba970a01"
TYPED_GRAPH_2027_SHA256="0208e4f0064b255396e03f5f348134fb4d0819fecc66b225b88898de36978425"
FLAT_2028_SHA256="e06ef970718933e2a3b56d054247bb8d5aa940cc6680c4bc1da81ab00cfeaad4"
TYPED_GRAPH_2028_SHA256="44b3eebd60a2f03545ba6c44998bb5bb4e05b5d0613a4da1fad2e6bb6626f327"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPYCACHEPREFIX="/tmp/ge1-stage6-zero-memory-pyc-${SLURM_JOB_ID}"
export TMPDIR="/tmp/ge1-stage6-zero-memory-${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"

container_python() {
  apptainer exec --cleanenv \
    --env OMP_NUM_THREADS=1 \
    --env MKL_NUM_THREADS=1 \
    --env PYTHONPYCACHEPREFIX="$PYTHONPYCACHEPREFIX" \
    --env TMPDIR="$TMPDIR" \
    --bind "$REPOSITORY:$REPOSITORY:ro" \
    --bind "$FLAT_2026:$FLAT_2026:ro" \
    --bind "$TYPED_GRAPH_2026:$TYPED_GRAPH_2026:ro" \
    --bind "$FLAT_2027:$FLAT_2027:ro" \
    --bind "$TYPED_GRAPH_2027:$TYPED_GRAPH_2027:ro" \
    --bind "$FLAT_2028:$FLAT_2028:ro" \
    --bind "$TYPED_GRAPH_2028:$TYPED_GRAPH_2028:ro" \
    --bind "$TRAIN_INPUT_ROOT:$TRAIN_INPUT_ROOT:ro" \
    --bind "$POSTMORTEM_ARTIFACT:$POSTMORTEM_ARTIFACT:ro" \
    --bind "$RUN_PARENT:$RUN_PARENT:rw" \
    "$CONTAINER" "$PYTHON" "$@"
}

test -d "$REPOSITORY/.git"
test ! -f "$REPOSITORY/.git"
test "$(basename "$RETAINED_WORK_ROOT")" = "$EXPECTED_RETAINED_NAME"
test "$(basename "$POSTMORTEM_ARTIFACT")" = "$EXPECTED_PRIOR_NAME"
test -d "$TRAIN_INPUT_ROOT"
test -f "$TRAIN_INDEX"
test -d "$TRAIN_ROOT"
test -d "$POSTMORTEM_ARTIFACT"
test -d "$RUN_PARENT"
test -f "$CONTAINER"
test ! -e "$OUTPUT_DIR"
test "$(sha256sum "$RUNNER" | cut -d ' ' -f 1)" = "$EXPECTED_RUNNER_SHA256"

cd "$REPOSITORY"
test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git symbolic-ref -q --short HEAD || true)"
test -z "$(git status --porcelain=v1 --untracked-files=all)"
test "$(git rev-parse --show-toplevel)" = "$REPOSITORY"
test -z "$(git rev-parse --show-superproject-working-tree)"
git cat-file -e "$EXPECTED_COMMIT^{commit}"
git cat-file -e "$PRODUCER_COMMIT^{commit}"
git cat-file -e "$PRIOR_COMMIT^{commit}"

test "$(sha256sum "$FLAT_2026" | cut -d ' ' -f 1)" = "$FLAT_2026_SHA256"
test "$(sha256sum "$TYPED_GRAPH_2026" | cut -d ' ' -f 1)" = "$TYPED_GRAPH_2026_SHA256"
test "$(sha256sum "$FLAT_2027" | cut -d ' ' -f 1)" = "$FLAT_2027_SHA256"
test "$(sha256sum "$TYPED_GRAPH_2027" | cut -d ' ' -f 1)" = "$TYPED_GRAPH_2027_SHA256"
test "$(sha256sum "$FLAT_2028" | cut -d ' ' -f 1)" = "$FLAT_2028_SHA256"
test "$(sha256sum "$TYPED_GRAPH_2028" | cut -d ' ' -f 1)" = "$TYPED_GRAPH_2028_SHA256"
test "$(sha256sum "$POSTMORTEM_ARTIFACT/SHA256SUMS" | cut -d ' ' -f 1)" = "$PRIOR_SHA256SUMS_SHA256"

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
    "event": "stage6_zero_memory_environment",
    "python": platform.python_version(),
    "pytorch": torch.__version__,
    "device": "cpu",
    "threads": 1,
}, sort_keys=True, separators=(",", ":")), flush=True)
PY

container_python -B - <<'PY'
import json
import unittest
names = (
    "prototype.graph_encoder.tests.test_stage6_zero_memory_contract",
    "prototype.graph_encoder.tests.test_stage6_zero_memory_runtime",
)
suite = unittest.TestSuite(
    unittest.defaultTestLoader.loadTestsFromName(name) for name in names
)
declared = suite.countTestCases()
result = unittest.TextTestRunner(verbosity=2).run(suite)
print(json.dumps({
    "event": "stage6_zero_memory_focused_tests",
    "declared": declared,
    "run": result.testsRun,
    "skipped": len(result.skipped),
}, sort_keys=True, separators=(",", ":")), flush=True)
if result.testsRun != declared or not result.wasSuccessful() or result.skipped:
    raise SystemExit("all zero-memory focused tests must pass with zero skips")
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
print("GE1_STAGE6_ZERO_MEMORY_PYTHON38_AST=PASS {}".format(len(paths)), flush=True)
PY
container_python -B -m prototype.graph_encoder.stage6_zero_memory_audit "$RUNNER"
bash -n "$RUNNER"
git show --check "$EXPECTED_COMMIT" --

container_python -B -m prototype.graph_encoder.stage6_zero_memory_diagnostic \
  --checkpoint "stage6-flat-seed2026.pt=$FLAT_2026" \
  --checkpoint "stage6-typed_graph-seed2026.pt=$TYPED_GRAPH_2026" \
  --checkpoint "stage6-flat-seed2027.pt=$FLAT_2027" \
  --checkpoint "stage6-typed_graph-seed2027.pt=$TYPED_GRAPH_2027" \
  --checkpoint "stage6-flat-seed2028.pt=$FLAT_2028" \
  --checkpoint "stage6-typed_graph-seed2028.pt=$TYPED_GRAPH_2028" \
  --train-index "$TRAIN_INDEX" \
  --train-root "$TRAIN_ROOT" \
  --prior-postmortem-artifact "$POSTMORTEM_ARTIFACT" \
  --output-dir "$OUTPUT_DIR" \
  --repository-root "$REPOSITORY" \
  --expected-commit "$EXPECTED_COMMIT" \
  --producer-job-id "$PRODUCER_JOB_ID" \
  --prior-postmortem-job-id "$PRIOR_JOB_ID" \
  --slurm-job-id "$SLURM_JOB_ID"

(
  cd "$OUTPUT_DIR"
  sha256sum -c SHA256SUMS
)
container_python -B - "$OUTPUT_DIR" "$EXPECTED_COMMIT" "$SLURM_JOB_ID" "$POSTMORTEM_ARTIFACT" <<'PY'
import json
import sys
from prototype.graph_encoder.stage6_zero_memory_diagnostic import verify_artifact
result = verify_artifact(
    sys.argv[1],
    expected_commit=sys.argv[2],
    expected_slurm_job_id=sys.argv[3],
    prior_postmortem_artifact=sys.argv[4],
)
assert result["verification_status"] == "pass"
print(json.dumps({
    "event": "stage6_zero_memory_artifact_verified",
    **result,
}, sort_keys=True, separators=(",", ":")), flush=True)
PY

test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git status --porcelain=v1 --untracked-files=all)"
printf '{"event":"stage6_zero_memory_terminal_success","commit":"%s","job_id":"%s","producer_job_id":"3354961","prior_postmortem_job_id":"3355134","artifact":"%s","training_performed":false,"development_accessed":false}\n' \
  "$EXPECTED_COMMIT" "$SLURM_JOB_ID" "$OUTPUT_DIR"
