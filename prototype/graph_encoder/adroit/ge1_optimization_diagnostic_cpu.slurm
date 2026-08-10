#!/bin/bash
#SBATCH --job-name=ge1-c7-opt-diagnostic
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/network/km6349/ge1_optimization_diagnostic_runs/optimization-cpu-%j.out
#SBATCH --error=/scratch/network/km6349/ge1_optimization_diagnostic_runs/optimization-cpu-%j.err

set -Eeuo pipefail

MANIFEST_ACCESSED=false
TRAIN_PAYLOAD_ACCESSED=false
DIAGNOSTIC_COMPLETED=false

terminal_failure() {
  status=$?
  trap - ERR
  printf '{"event":"optimization_diagnostic_terminal_failure","commit":"%s","job_id":"%s","exit_status":%s,"interpretation_category":"infrastructure_failure","operation_template_manifest_accessed":%s,"operation_template_train_payload_accessed":%s,"development_accessed":false,"systematic_rr_accessed":false,"test_er_accessed":false,"iid_accessed":false,"history_depth_accessed":false,"geometry_extrapolation_accessed":false,"optimization_diagnostic_completed":false,"original_c7_result_changed":false,"stage6_authorized":false,"decoder_repair_invoked":false,"c8_or_later_performed":false}\n' \
    "${EXPECTED_COMMIT:-unresolved}" "${SLURM_JOB_ID:-unresolved}" "$status" \
    "$MANIFEST_ACCESSED" "$TRAIN_PAYLOAD_ACCESSED" >&2
  exit "$status"
}
trap terminal_failure ERR

: "${EXPECTED_COMMIT:?submit with --export=ALL,EXPECTED_COMMIT=<exact-diagnostic-commit>}"
: "${SLURM_JOB_ID:?this workflow must run under Slurm}"
[[ "$SLURM_JOB_ID" =~ ^[0-9]+$ ]]
REPOSITORY="${REPOSITORY:-/scratch/network/km6349/SkexGen-optimization-diagnostic}"
CORPUS_DIR="${CORPUS_DIR:-/scratch/network/km6349/controlled_corpora/b0-pilot-680-seed2026}"
RUN_PARENT="${RUN_PARENT:-/scratch/network/km6349/ge1_optimization_diagnostic_runs}"
OUTPUT_DIR="${RUN_PARENT}/ge1-c7-optimization-${EXPECTED_COMMIT}-${SLURM_JOB_ID}"
CONTAINER="/scratch/network/km6349/skexgen.sif"
PYTHON="/root/miniconda3/bin/python3.8"

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPYCACHEPREFIX="/tmp/ge1-optimization-${SLURM_JOB_ID}"
export TMPDIR="/tmp/ge1-optimization-tmp-${SLURM_JOB_ID}"
mkdir -p "$TMPDIR" "$RUN_PARENT"

container_python() {
  apptainer exec \
    --cleanenv \
    --env SLURM_JOB_ID="$SLURM_JOB_ID" \
    --env CUDA_VISIBLE_DEVICES="" \
    --env OMP_NUM_THREADS=1 \
    --env MKL_NUM_THREADS=1 \
    --env PYTHONPYCACHEPREFIX="$PYTHONPYCACHEPREFIX" \
    --env TMPDIR="$TMPDIR" \
    --bind "$REPOSITORY:$REPOSITORY" \
    --bind "$CORPUS_DIR:$CORPUS_DIR" \
    --bind "$RUN_PARENT:$RUN_PARENT" \
    "$CONTAINER" "$PYTHON" "$@"
}

test -d "$REPOSITORY/.git"
test -d "$CORPUS_DIR"
test -f "$CONTAINER"
test ! -e "$OUTPUT_DIR"
test ! -L "$OUTPUT_DIR"
cd "$REPOSITORY"

CHECKED_OUT_COMMIT="$(git rev-parse HEAD)"
SOURCE_STATUS="$(git status --porcelain=v1 --untracked-files=all)"
test "$CHECKED_OUT_COMMIT" = "$EXPECTED_COMMIT"
test -z "$SOURCE_STATUS"

printf '{"event":"optimization_diagnostic_validation_start","commit":"%s","job_id":"%s","host":"%s","source_clean":true,"execution_device":"cpu","operation_template_manifest_accessed":false,"operation_template_train_payload_accessed":false,"development_accessed":false,"systematic_rr_accessed":false,"test_er_accessed":false,"iid_accessed":false,"history_depth_accessed":false,"geometry_extrapolation_accessed":false,"optimization_diagnostic_completed":false,"original_c7_result_changed":false,"stage6_authorized":false,"decoder_repair_invoked":false,"c8_or_later_performed":false}\n' \
  "$CHECKED_OUT_COMMIT" "$SLURM_JOB_ID" "$(hostname)"

container_python -B - <<'PY'
import json
import os
import platform
import sys
import torch

assert sys.version_info[:3] == (3, 8, 13), "Python 3.8.13 required"
assert torch.__version__.split("+")[0] == "1.11.0", "PyTorch 1.11.0 required"
assert not torch.cuda.is_available(), "CUDA must be unavailable"
assert torch.get_num_threads() == 1, "one PyTorch CPU thread required"
job_id = os.environ.get("SLURM_JOB_ID")
assert job_id and job_id.isdecimal(), "SLURM_JOB_ID must cross cleanenv"
print(json.dumps({
    "event": "optimization_diagnostic_environment",
    "python_version": sys.version.split()[0],
    "python_implementation": platform.python_implementation(),
    "pytorch_version": torch.__version__,
    "pytorch_cuda_build": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "execution_device": "cpu",
    "cpu_threads": torch.get_num_threads(),
    "slurm_job_id": job_id,
}, sort_keys=True, separators=(",", ":")), flush=True)
PY

container_python -B - <<'PY'
import json
import unittest

names = (
    "prototype.graph_encoder.tests.test_optimization_diagnostic_contract",
    "prototype.graph_encoder.tests.test_optimization_diagnostic_runtime",
)
suite = unittest.TestSuite(
    unittest.defaultTestLoader.loadTestsFromName(name) for name in names
)
result = unittest.TextTestRunner(verbosity=2).run(suite)
print(json.dumps({
    "event": "optimization_diagnostic_focused_suite",
    "tests_run": result.testsRun,
    "failures": len(result.failures),
    "errors": len(result.errors),
    "skipped": len(result.skipped),
}, sort_keys=True, separators=(",", ":")), flush=True)
if result.testsRun != 20 or not result.wasSuccessful() or result.skipped:
    raise SystemExit("all 20 diagnostic tests must pass with zero skips")
PY

container_python -B - <<'PY'
import json
import unittest

suite = unittest.defaultTestLoader.discover("prototype/graph_encoder/tests")
result = unittest.TextTestRunner(verbosity=1).run(suite)
print(json.dumps({
    "event": "optimization_diagnostic_complete_graph_encoder_suite",
    "tests_run": result.testsRun,
    "failures": len(result.failures),
    "errors": len(result.errors),
    "skipped": len(result.skipped),
}, sort_keys=True, separators=(",", ":")), flush=True)
if result.testsRun != 238 or not result.wasSuccessful() or result.skipped:
    raise SystemExit("all 238 graph_encoder tests must pass with zero skips")
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
    sorted(Path("prototype/graph_encoder/tests").glob("*.py"))
)
for path in paths:
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path), feature_version=(3, 8))
print("OPTIMIZATION_DIAGNOSTIC_PYTHON38_GRAMMAR_OK {}".format(len(paths)), flush=True)
PY

container_python -B - <<'PY'
import ast
from pathlib import Path

path = Path("prototype/graph_encoder/optimization_diagnostic.py")
source = path.read_text(encoding="utf-8")
tree = ast.parse(source)
imported = {
    alias.name
    for node in ast.walk(tree)
    if isinstance(node, ast.ImportFrom)
    for alias in node.names
}
names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
assert "load_development" not in imported | names
assert "secondary_systematic_validation" not in source
assert "load_partition_physical_examples" not in source
assert "teacher_forced(" not in source
assert "run_pilot(" not in source
calls = {
    node.func.id
    for node in ast.walk(tree)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
}
assert "load_development" not in calls
assert "run_pilot" not in calls
print("OPTIMIZATION_DIAGNOSTIC_ACCESS_SOURCE_AUDIT=PASS", flush=True)
PY

container_python -B - <<'PY'
import prototype.graph_encoder as package
assert package.__all__
print("GRAPH_ENCODER_IMPORT_EXPORT_OK {}".format(len(package.__all__)), flush=True)
PY

bash -n prototype/graph_encoder/adroit/ge1_optimization_diagnostic_cpu.slurm
git diff --check
test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git status --porcelain=v1 --untracked-files=all)"

MANIFEST_ACCESSED='"not_confirmed_on_failure"'
TRAIN_PAYLOAD_ACCESSED='"not_confirmed_on_failure"'
container_python -B -m prototype.graph_encoder.optimization_diagnostic \
  --corpus-dir "$CORPUS_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --repository-root "$REPOSITORY" \
  --expected-commit "$EXPECTED_COMMIT"
MANIFEST_ACCESSED=true
TRAIN_PAYLOAD_ACCESSED=true

test -s "$OUTPUT_DIR/resolved_config.json"
test -s "$OUTPUT_DIR/metrics.jsonl"
test -d "$OUTPUT_DIR/checkpoints"
test -s "$OUTPUT_DIR/artifact_manifest.json"
test -s "$OUTPUT_DIR/SHA256SUMS"

(
  cd "$OUTPUT_DIR"
  sha256sum -c SHA256SUMS
)

container_python -B - "$OUTPUT_DIR" "$SLURM_JOB_ID" <<'PY'
import json
from pathlib import Path
import sys
import torch

from prototype.graph_encoder.optimization_diagnostic import (
    CHECKPOINT_UPDATES,
    FROZEN_FAMILY_IDS,
    FROZEN_FAMILY_IDS_SHA256,
    verify_diagnostic_artifact,
)

root = Path(sys.argv[1])
job_id = sys.argv[2]
verified = verify_diagnostic_artifact(root, expected_slurm_job_id=job_id)
resolved = json.loads((root / "resolved_config.json").read_text(encoding="utf-8"))
events = [
    json.loads(line)
    for line in (root / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    if line
]
terminal = events[-1]
assert resolved["runtime"]["slurm_job_id"] == job_id
assert tuple(resolved["selection"]["family_ids"]) == FROZEN_FAMILY_IDS
assert resolved["selection"]["family_ids_sha256"] == FROZEN_FAMILY_IDS_SHA256
assert resolved["training_arithmetic"]["maximum_optimizer_updates"] == 500
assert resolved["training_arithmetic"]["training_example_presentations"] == 2000
assert tuple(resolved["training_arithmetic"]["scientific_measurement_milestones"]) == (50, 100, 200, 500)
checkpoint_paths = sorted(root.glob("checkpoints/*/*.pt"))
assert len(checkpoint_paths) == 2 * len(CHECKPOINT_UPDATES)
for path in checkpoint_paths:
    payload = torch.load(str(path), map_location="cpu")
    assert payload["provenance"]["slurm_job_id"] == job_id
assert terminal["event"] == "optimization_diagnostic_completed"
assert terminal["slurm_job_id"] == job_id
assert terminal["original_c7_result_changed"] is False
assert terminal["stage6_authorized"] is False
assert terminal["decoder_repair_invoked"] is False
assert terminal["c8_or_later_performed"] is False
for name in (
    "development_accessed", "systematic_rr_accessed", "test_er_accessed",
    "iid_accessed", "history_depth_accessed", "geometry_extrapolation_accessed",
):
    assert terminal[name] is False
milestones = [item for item in events if item.get("event") == "diagnostic_milestone_measurement"]
assert len(milestones) == 8
for item in milestones:
    assert set(item["primary_reporting"]) == {"P_true", "P_shuffle", "P_mean", "R_shuffle", "R_mean"}
    donor = {row["condition"]: row for row in item["donor_template_agreement"]}
    assert donor["P_shuffle"]["available"] is True
    assert donor["P_true"]["available"] is False
    assert donor["P_mean"]["available"] is False
print(json.dumps({
    "event": "optimization_diagnostic_artifact_verified",
    "slurm_job_id": job_id,
    "interpretation_category": verified["interpretation_category"],
    "artifact_manifest_sha256": verified["artifact_manifest_sha256"],
    "sha256sums_sha256": verified["sha256sums_sha256"],
    "regular_file_count": verified["regular_file_count"],
    "milestone_measurements": len(milestones),
    "checkpoints": len(checkpoint_paths),
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
events = [
    json.loads(line)
    for line in (Path(root) / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    if line
]
terminal = events[-1]
print(json.dumps({
    "event": "optimization_diagnostic_terminal_success",
    "commit": commit,
    "job_id": job_id,
    "interpretation_category": terminal["interpretation_category"],
    "focused_diagnostic_tests": 20,
    "complete_graph_encoder_tests": 238,
    "pytorch_related_skips": 0,
    "operation_template_manifest_accessed": True,
    "operation_template_train_payload_accessed": True,
    "development_accessed": False,
    "systematic_rr_accessed": False,
    "test_er_accessed": False,
    "iid_accessed": False,
    "history_depth_accessed": False,
    "geometry_extrapolation_accessed": False,
    "optimization_diagnostic_completed": True,
    "original_c7_result_changed": False,
    "stage6_authorized": False,
    "decoder_repair_invoked": False,
    "c8_or_later_performed": False,
    "artifact_path": root,
}, sort_keys=True, separators=(",", ":")), flush=True)
PY
