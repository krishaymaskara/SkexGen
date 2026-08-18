#!/bin/bash
#SBATCH --job-name=ge1-grid-ordinal-trajectory
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:15:00
#SBATCH --signal=B:TERM@60
#SBATCH --output=/scratch/network/km6349/ge1_grid_ordinal_trajectory_runs/grid-ordinal-trajectory-%j.out
#SBATCH --error=/scratch/network/km6349/ge1_grid_ordinal_trajectory_runs/grid-ordinal-trajectory-%j.err

set -Eeuo pipefail

DIAGNOSTIC_COMPLETED=false

grid_ordinal_trajectory_terminal_failure() {
  status=$?
  trap - ERR
  printf '{"event":"grid_ordinal_trajectory_terminal_failure","commit":"%s","job_id":"%s","exit_status":%s,"diagnostic_completed":false,"engineering_only":true,"scientific_execution":false,"scientific_training":false,"scientific_inference":false,"corpus_accessed":false,"manifest_accessed":false,"train_payload_accessed":false,"preserved_feature_payload_accessed":false,"checkpoint_accessed":false,"model_artifact_accessed":false,"repaired_artifact_accessed":false,"cad_kernel_accessed":false,"protected_partition_accessed":false,"scientific_job_authorized":false,"repair_authorized":false,"another_job_authorized":false,"stage6_authorized":false,"c8_authorized":false}\n' \
    "${EXPECTED_COMMIT:-unresolved}" "${SLURM_JOB_ID:-unresolved}" "$status" >&2
  exit "$status"
}

grid_ordinal_trajectory_terminal_timeout() {
  trap - ERR TERM
  printf '{"event":"grid_ordinal_trajectory_terminal_timeout","commit":"%s","job_id":"%s","exit_status":124,"diagnostic_completed":false,"engineering_only":true,"scientific_execution":false,"scientific_training":false,"scientific_inference":false,"corpus_accessed":false,"manifest_accessed":false,"train_payload_accessed":false,"preserved_feature_payload_accessed":false,"checkpoint_accessed":false,"model_artifact_accessed":false,"repaired_artifact_accessed":false,"cad_kernel_accessed":false,"protected_partition_accessed":false,"scientific_job_authorized":false,"repair_authorized":false,"another_job_authorized":false,"stage6_authorized":false,"c8_authorized":false}\n' \
    "${EXPECTED_COMMIT:-unresolved}" "${SLURM_JOB_ID:-unresolved}" >&2
  exit 124
}
trap grid_ordinal_trajectory_terminal_failure ERR
trap grid_ordinal_trajectory_terminal_timeout TERM

: "${EXPECTED_COMMIT:?engineering diagnostic requires an explicitly authorized exact commit}"
: "${SLURM_JOB_ID:?this workflow must run under Slurm}"
REPOSITORY="${REPOSITORY:-/scratch/network/km6349/SkexGen-grid-magnitude}"
RUN_PARENT="${RUN_PARENT:-/scratch/network/km6349/ge1_grid_ordinal_trajectory_artifacts}"
LOG_PARENT="/scratch/network/km6349/ge1_grid_ordinal_trajectory_runs"
OUTPUT_DIR="${RUN_PARENT}/ge1-grid-ordinal-trajectory-${EXPECTED_COMMIT}-${SLURM_JOB_ID}"
CONTAINER="/scratch/network/km6349/skexgen.sif"
PYTHON="/root/miniconda3/bin/python3.8"
AUTHORIZATION_PARENT_COMMIT="de80d93208a1a98196a38d2af5cd47afd525d14f"

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPYCACHEPREFIX="/tmp/ge1-grid-ordinal-trajectory-${SLURM_JOB_ID}"
export TMPDIR="/tmp/ge1-grid-ordinal-trajectory-tmp-${SLURM_JOB_ID}"
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
    --bind "$RUN_PARENT:$RUN_PARENT:rw" \
    "$CONTAINER" "$PYTHON" "$@"
}

test -d "$REPOSITORY/.git"
test ! -f "$REPOSITORY/.git"
test -d "$RUN_PARENT"
test -d "$LOG_PARENT"
test -f "$CONTAINER"
test ! -e "$OUTPUT_DIR"
test ! -L "$OUTPUT_DIR"
cd "$REPOSITORY"

CHECKED_OUT_COMMIT="$(git rev-parse HEAD)"
SOURCE_STATUS="$(git status --porcelain=v1 --untracked-files=all)"
BRANCH_NAME="$(git symbolic-ref -q --short HEAD || true)"
test "$CHECKED_OUT_COMMIT" = "$EXPECTED_COMMIT"
test -z "$SOURCE_STATUS"
test -z "$BRANCH_NAME"
test "$(git rev-parse --show-toplevel)" = "$REPOSITORY"
test -z "$(git rev-parse --show-superproject-working-tree)"
git merge-base --is-ancestor "$AUTHORIZATION_PARENT_COMMIT" "$EXPECTED_COMMIT"

printf '{"event":"grid_ordinal_trajectory_start","commit":"%s","job_id":"%s","host":"%s","source_clean":true,"standalone_checkout":true,"detached_head":true,"exact_commit":true,"execution_device":"cpu","cpu_task_count":1,"cpu_thread_count":1,"memory_gb":2,"wall_minutes":15,"engineering_only":true,"scientific_execution":false,"scientific_training":false,"scientific_inference":false,"corpus_accessed":false,"manifest_accessed":false,"train_payload_accessed":false,"preserved_feature_payload_accessed":false,"checkpoint_accessed":false,"model_artifact_accessed":false,"repaired_artifact_accessed":false,"cad_kernel_accessed":false,"protected_partition_accessed":false,"scientific_job_authorized":false,"repair_authorized":false,"another_job_authorized":false,"stage6_authorized":false,"c8_authorized":false}\n' \
  "$CHECKED_OUT_COMMIT" "$SLURM_JOB_ID" "$(hostname)"

container_python -B - <<'PY'
import json
import os
import platform
import sys
import torch

assert sys.version_info[:3] == (3, 8, 13)
assert torch.__version__.split("+")[0] == "1.11.0"
assert not torch.cuda.is_available()
assert torch.get_num_threads() == 1
assert os.environ["SLURM_JOB_ID"].isdecimal()
print(json.dumps({
    "event": "grid_ordinal_trajectory_environment",
    "python_version": sys.version.split()[0],
    "python_implementation": platform.python_implementation(),
    "pytorch_version": torch.__version__,
    "pytorch_cuda_build": torch.version.cuda,
    "cuda_available": False,
    "execution_device": "cpu",
    "cpu_threads": torch.get_num_threads(),
}, sort_keys=True, separators=(",", ":")), flush=True)
PY

container_python -B - <<'PY'
import json
import unittest

for name, expected in (
    ("prototype.graph_encoder.tests.test_grid_ordinal_trajectory_contract", 17),
    ("prototype.graph_encoder.tests.test_grid_ordinal_trajectory_runtime", 8),
):
    suite = unittest.defaultTestLoader.loadTestsFromName(name)
    declared = suite.countTestCases()
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    print(json.dumps({
        "event": "grid_ordinal_trajectory_focused_suite",
        "suite": name,
        "declared_tests": declared,
        "tests_run": result.testsRun,
        "expected_tests": expected,
        "skipped": len(result.skipped),
    }, sort_keys=True, separators=(",", ":")), flush=True)
    if declared != expected or result.testsRun != expected or not result.wasSuccessful() or result.skipped:
        raise SystemExit("focused trajectory suite differs")
PY

container_python -B - <<'PY'
import json
import unittest

for name, expected in (
    ("prototype.graph_encoder.tests.test_grid_magnitude_contract", 28),
    ("prototype.graph_encoder.tests.test_grid_magnitude_runtime", 14),
    ("prototype.graph_encoder.tests.test_grid_magnitude_integration", 18),
    ("prototype.graph_encoder.tests.test_grid_magnitude_sufficiency_contract", 31),
):
    suite = unittest.defaultTestLoader.loadTestsFromName(name)
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    if suite.countTestCases() != expected or result.testsRun != expected or not result.wasSuccessful() or result.skipped:
        raise SystemExit("existing grid suite differs: {}".format(name))
    print(json.dumps({
        "event": "grid_ordinal_trajectory_existing_grid_suite",
        "suite": name, "tests_run": result.testsRun, "skipped": 0,
    }, sort_keys=True, separators=(",", ":")), flush=True)
PY

container_python -B - <<'PY'
import json
import unittest

result = unittest.TextTestRunner(verbosity=1).run(
    unittest.TestLoader().discover("prototype/graph_encoder/tests", top_level_dir=".")
)
print(json.dumps({
    "event": "grid_ordinal_trajectory_complete_graph_encoder_suite",
    "tests_run": result.testsRun,
    "failures": len(result.failures),
    "errors": len(result.errors),
    "skipped": len(result.skipped),
}, sort_keys=True, separators=(",", ":")), flush=True)
if result.testsRun != 516 or not result.wasSuccessful() or result.skipped:
    raise SystemExit("all 516 graph-encoder tests must pass with zero skips")
PY

container_python -B tools/check_documentation.py
container_python -B -m compileall -q prototype/graph_encoder
container_python -B - <<'PY'
import ast
from pathlib import Path

root = Path("prototype/graph_encoder")
paths = tuple(sorted(root.glob("*.py"))) + tuple(sorted((root / "tests").glob("test_*.py")))
for path in paths:
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path), feature_version=(3, 8))
print("GRID_ORDINAL_TRAJECTORY_PYTHON38_AST=PASS {}".format(len(paths)), flush=True)
PY
container_python -B -m prototype.graph_encoder.grid_ordinal_trajectory_audit \
  prototype/graph_encoder/adroit/ge1_grid_ordinal_trajectory_cpu.slurm
container_python -B - "$AUTHORIZATION_PARENT_COMMIT" "$EXPECTED_COMMIT" <<'PY'
from pathlib import Path
import subprocess
import sys

parent, commit = sys.argv[1:]
allowed = {
    "prototype/graph_encoder/grid_ordinal_trajectory.py",
    "prototype/graph_encoder/grid_ordinal_trajectory_audit.py",
    "prototype/graph_encoder/adroit/ge1_grid_ordinal_trajectory_cpu.slurm",
    "prototype/graph_encoder/tests/test_grid_ordinal_trajectory_contract.py",
    "prototype/graph_encoder/tests/test_grid_ordinal_trajectory_runtime.py",
}
output = subprocess.check_output(
    ("git", "diff", "--name-only", parent, commit, "--", "prototype"),
    universal_newlines=True,
)
observed = {line for line in output.splitlines() if line}
assert observed == allowed, (observed, allowed)
print("GRID_ORDINAL_TRAJECTORY_PRODUCTION_PRESERVATION=PASS", flush=True)
PY
bash -n prototype/graph_encoder/adroit/ge1_grid_ordinal_trajectory_cpu.slurm
git show --check "$EXPECTED_COMMIT" --
test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git symbolic-ref -q --short HEAD || true)"
test -z "$(git status --porcelain=v1 --untracked-files=all)"

container_python -B -m prototype.graph_encoder.grid_ordinal_trajectory \
  --output-dir "$OUTPUT_DIR" \
  --repository-root "$REPOSITORY" \
  --expected-commit "$EXPECTED_COMMIT"

for name in resolved_config.json trajectory.jsonl summary.json artifact_manifest.json SHA256SUMS
do
  test -s "$OUTPUT_DIR/$name"
done
test "$(find "$OUTPUT_DIR" -type f | wc -l | tr -d ' ')" = 5
(
  cd "$OUTPUT_DIR"
  sha256sum -c SHA256SUMS
)

container_python -B - "$OUTPUT_DIR" "$EXPECTED_COMMIT" "$SLURM_JOB_ID" <<'PY'
import json
import sys
from prototype.graph_encoder.grid_ordinal_trajectory import verify_artifact

root, commit, job_id = sys.argv[1:]
verified = verify_artifact(
    root, expected_commit=commit, expected_slurm_job_id=job_id
)
print(json.dumps({
    "event": "grid_ordinal_trajectory_artifact_verified",
    **verified,
}, sort_keys=True, separators=(",", ":")), flush=True)
PY

test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git status --porcelain=v1 --untracked-files=all)"
DIAGNOSTIC_COMPLETED=true

container_python -B - "$OUTPUT_DIR" "$EXPECTED_COMMIT" "$SLURM_JOB_ID" <<'PY'
import json
from pathlib import Path
import sys

root, commit, job_id = sys.argv[1:]
summary = json.loads((Path(root) / "summary.json").read_text(encoding="utf-8"))
print(json.dumps({
    "event": "grid_ordinal_trajectory_terminal_success",
    "commit": commit,
    "job_id": job_id,
    "artifact_path": root,
    "interpretation_category": summary["hypothesis_evaluation"]["interpretation_category"],
    "diagnostic_completed": True,
    "engineering_only": True,
    "scientific_execution": False,
    "scientific_training": False,
    "scientific_inference": False,
    "corpus_accessed": False,
    "manifest_accessed": False,
    "train_payload_accessed": False,
    "preserved_feature_payload_accessed": False,
    "checkpoint_accessed": False,
    "model_artifact_accessed": False,
    "repaired_artifact_accessed": False,
    "cad_kernel_accessed": False,
    "protected_partition_accessed": False,
    "scientific_job_authorized": False,
    "repair_authorized": False,
    "another_job_authorized": False,
    "stage6_authorized": False,
    "c8_authorized": False,
}, sort_keys=True, separators=(",", ":")), flush=True)
PY
