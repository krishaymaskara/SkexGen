#!/bin/bash
# Read-only ADR-0014 train-gate postmortem for producer job 3354961.
# No submission command is present; submission remains a separate reviewer act.
#SBATCH --job-name=ge1-s6-postmortem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/network/km6349/ge1_stage6_train_gate_postmortem_runs/postmortem-%j.out
#SBATCH --error=/scratch/network/km6349/ge1_stage6_train_gate_postmortem_runs/postmortem-%j.err
#SBATCH --signal=B:TERM@60

set -Eeuo pipefail

infrastructure_failure() {
  status="${1:-$?}"
  trap - ERR
  printf '{"event":"stage6_train_gate_postmortem_terminal_failure","commit":"%s","job_id":"%s","producer_job_id":"3354961","exit_status":%s,"training_performed":false,"development_accessed":false}\n' \
    "${EXPECTED_COMMIT:-unresolved}" "${SLURM_JOB_ID:-unresolved}" "$status" >&2
  exit "$status"
}

terminal_timeout() {
  trap - ERR
  printf '{"event":"stage6_train_gate_postmortem_terminal_timeout","commit":"%s","job_id":"%s","producer_job_id":"3354961","exit_status":124,"training_performed":false,"development_accessed":false}\n' \
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
: "${RUN_PARENT:?diagnostic output parent required}"
: "${EXPECTED_FLAT_2026_SHA256:?reviewed wrapper SHA-256 required}"
: "${EXPECTED_TYPED_GRAPH_2026_SHA256:?reviewed wrapper SHA-256 required}"
: "${EXPECTED_FLAT_2027_SHA256:?reviewed wrapper SHA-256 required}"
: "${EXPECTED_TYPED_GRAPH_2027_SHA256:?reviewed wrapper SHA-256 required}"
: "${EXPECTED_FLAT_2028_SHA256:?reviewed wrapper SHA-256 required}"
: "${EXPECTED_TYPED_GRAPH_2028_SHA256:?reviewed wrapper SHA-256 required}"

PRODUCER_COMMIT="5f4542f86756dae44435af27a6e072db6f27a8ef"
PRODUCER_JOB_ID="3354961"
REPOSITORY="${REPOSITORY:-/scratch/network/km6349/SkexGen-stage6-structure-only}"
CONTAINER="${CONTAINER:-/scratch/network/km6349/skexgen.sif}"
PYTHON="${PYTHON:-/root/miniconda3/bin/python3.8}"
RUNNER="$REPOSITORY/prototype/graph_encoder/adroit/ge1_stage6_train_gate_postmortem_cpu.slurm"
TRAIN_INDEX="$TRAIN_INPUT_ROOT/index.json"
TRAIN_ROOT="$TRAIN_INPUT_ROOT/payloads"
OUTPUT_DIR="$RUN_PARENT/ge1-stage6-train-gate-postmortem-${EXPECTED_COMMIT}-${SLURM_JOB_ID}"
EXPECTED_RETAINED_NAME="ge1-stage6-structure-only-producer-${PRODUCER_COMMIT}-${PRODUCER_JOB_ID}.work.incomplete-${PRODUCER_JOB_ID}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPYCACHEPREFIX="/tmp/ge1-stage6-postmortem-pyc-${SLURM_JOB_ID}"
export TMPDIR="/tmp/ge1-stage6-postmortem-${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"

container_python() {
  apptainer exec --cleanenv \
    --env OMP_NUM_THREADS=1 \
    --env MKL_NUM_THREADS=1 \
    --env PYTHONPYCACHEPREFIX="$PYTHONPYCACHEPREFIX" \
    --env TMPDIR="$TMPDIR" \
    --bind "$REPOSITORY:$REPOSITORY:ro" \
    --bind "$RETAINED_WORK_ROOT:$RETAINED_WORK_ROOT:ro" \
    --bind "$TRAIN_INPUT_ROOT:$TRAIN_INPUT_ROOT:ro" \
    --bind "$RUN_PARENT:$RUN_PARENT:rw" \
    "$CONTAINER" "$PYTHON" "$@"
}

test -d "$REPOSITORY/.git"
test ! -f "$REPOSITORY/.git"
test -d "$RETAINED_WORK_ROOT"
test "$(basename "$RETAINED_WORK_ROOT")" = "$EXPECTED_RETAINED_NAME"
test -d "$TRAIN_INPUT_ROOT"
test -f "$TRAIN_INDEX"
test -d "$TRAIN_ROOT"
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

test "$(sha256sum "$RETAINED_WORK_ROOT/stage6-flat-seed2026.pt" | cut -d ' ' -f 1)" = "$EXPECTED_FLAT_2026_SHA256"
test "$(sha256sum "$RETAINED_WORK_ROOT/stage6-typed_graph-seed2026.pt" | cut -d ' ' -f 1)" = "$EXPECTED_TYPED_GRAPH_2026_SHA256"
test "$(sha256sum "$RETAINED_WORK_ROOT/stage6-flat-seed2027.pt" | cut -d ' ' -f 1)" = "$EXPECTED_FLAT_2027_SHA256"
test "$(sha256sum "$RETAINED_WORK_ROOT/stage6-typed_graph-seed2027.pt" | cut -d ' ' -f 1)" = "$EXPECTED_TYPED_GRAPH_2027_SHA256"
test "$(sha256sum "$RETAINED_WORK_ROOT/stage6-flat-seed2028.pt" | cut -d ' ' -f 1)" = "$EXPECTED_FLAT_2028_SHA256"
test "$(sha256sum "$RETAINED_WORK_ROOT/stage6-typed_graph-seed2028.pt" | cut -d ' ' -f 1)" = "$EXPECTED_TYPED_GRAPH_2028_SHA256"

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
    "event": "stage6_train_gate_postmortem_environment",
    "python": platform.python_version(),
    "pytorch": torch.__version__,
    "device": "cpu",
    "threads": 1,
}, sort_keys=True, separators=(",", ":")), flush=True)
PY

container_python -B - <<'PY'
import json
import unittest
name = "prototype.graph_encoder.tests.test_stage6_train_gate_postmortem_contract"
suite = unittest.defaultTestLoader.loadTestsFromName(name)
declared = suite.countTestCases()
result = unittest.TextTestRunner(verbosity=2).run(suite)
print(json.dumps({
    "event": "stage6_train_gate_postmortem_focused_tests",
    "declared": declared,
    "run": result.testsRun,
    "skipped": len(result.skipped),
}, sort_keys=True, separators=(",", ":")), flush=True)
if result.testsRun != declared or not result.wasSuccessful() or result.skipped:
    raise SystemExit("all focused postmortem tests must pass with zero skips")
PY

container_python -B tools/check_documentation.py
container_python -B -m compileall -q prototype/graph_encoder
container_python -B - <<'PY'
import ast
from pathlib import Path
paths = tuple(sorted(Path("prototype/graph_encoder").glob("*.py"))) + tuple(
    sorted(Path("prototype/graph_encoder/tests").glob("test_*.py")))
for path in paths:
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path), feature_version=(3, 8))
print("GE1_STAGE6_POSTMORTEM_PYTHON38_AST=PASS {}".format(len(paths)), flush=True)
PY
container_python -B -m prototype.graph_encoder.stage6_train_gate_postmortem_audit "$RUNNER"
bash -n "$RUNNER"
git show --check "$EXPECTED_COMMIT" --

container_python -B -m prototype.graph_encoder.stage6_train_gate_postmortem \
  --retained-work-root "$RETAINED_WORK_ROOT" \
  --train-index "$TRAIN_INDEX" \
  --train-root "$TRAIN_ROOT" \
  --checkpoint-sha256 "stage6-flat-seed2026.pt=$EXPECTED_FLAT_2026_SHA256" \
  --checkpoint-sha256 "stage6-typed_graph-seed2026.pt=$EXPECTED_TYPED_GRAPH_2026_SHA256" \
  --checkpoint-sha256 "stage6-flat-seed2027.pt=$EXPECTED_FLAT_2027_SHA256" \
  --checkpoint-sha256 "stage6-typed_graph-seed2027.pt=$EXPECTED_TYPED_GRAPH_2027_SHA256" \
  --checkpoint-sha256 "stage6-flat-seed2028.pt=$EXPECTED_FLAT_2028_SHA256" \
  --checkpoint-sha256 "stage6-typed_graph-seed2028.pt=$EXPECTED_TYPED_GRAPH_2028_SHA256" \
  --output-dir "$OUTPUT_DIR" \
  --repository-root "$REPOSITORY" \
  --expected-commit "$EXPECTED_COMMIT" \
  --producer-job-id "$PRODUCER_JOB_ID" \
  --slurm-job-id "$SLURM_JOB_ID"

(
  cd "$OUTPUT_DIR"
  sha256sum -c SHA256SUMS
)
container_python -B - "$OUTPUT_DIR" "$EXPECTED_COMMIT" "$SLURM_JOB_ID" <<'PY'
import json
import sys
from prototype.graph_encoder.stage6_train_gate_postmortem import verify_artifact
result = verify_artifact(
    sys.argv[1], expected_commit=sys.argv[2], expected_slurm_job_id=sys.argv[3]
)
assert result["verification_status"] == "pass"
print(json.dumps({
    "event": "stage6_train_gate_postmortem_artifact_verified",
    **result,
}, sort_keys=True, separators=(",", ":")), flush=True)
PY

test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git status --porcelain=v1 --untracked-files=all)"
printf '{"event":"stage6_train_gate_postmortem_terminal_success","commit":"%s","job_id":"%s","producer_job_id":"3354961","artifact":"%s","training_performed":false,"development_accessed":false}\n' \
  "$EXPECTED_COMMIT" "$SLURM_JOB_ID" "$OUTPUT_DIR"
