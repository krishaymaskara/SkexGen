#!/bin/bash
#SBATCH --job-name=ge1-c7-train-gates
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/network/km6349/c7_validation_runs/c7-cpu-%j.out
#SBATCH --error=/scratch/network/km6349/c7_validation_runs/c7-cpu-%j.err

set -Eeuo pipefail

MANIFEST_ACCESSED=false
TRAIN_PAYLOAD_ACCESSED=false
C7_GATE_COMPLETED=false

terminal_failure() {
  status=$?
  trap - ERR
  printf '{"event":"c7_pilot_terminal_failure","commit":"%s","job_id":"%s","exit_status":%s,"operation_template_manifest_accessed":%s,"operation_template_train_payload_accessed":%s,"development_accessed":false,"systematic_rr_accessed":false,"test_er_accessed":false,"iid_accessed":false,"history_depth_accessed":false,"geometry_extrapolation_accessed":false,"c7_gate_completed":false,"c8_or_later_performed":false}\n' \
    "${EXPECTED_COMMIT:-unresolved}" "${SLURM_JOB_ID:-unresolved}" "$status" \
    "$MANIFEST_ACCESSED" "$TRAIN_PAYLOAD_ACCESSED" >&2
  exit "$status"
}
trap terminal_failure ERR

: "${EXPECTED_COMMIT:?submit with --export=ALL,EXPECTED_COMMIT=<exact-C7-commit>}"
: "${SLURM_JOB_ID:?this workflow must run under Slurm}"
REPOSITORY="${REPOSITORY:-/scratch/network/km6349/SkexGen-c7}"
CORPUS_DIR="${CORPUS_DIR:-/scratch/network/km6349/controlled_corpora/b0-pilot-680-seed2026}"
RUN_PARENT="${RUN_PARENT:-/scratch/network/km6349/ge1_c7_runs}"
OUTPUT_DIR="${RUN_PARENT}/ge1-c7-${EXPECTED_COMMIT}-${SLURM_JOB_ID}"
CONTAINER="/scratch/network/km6349/skexgen.sif"
PYTHON="/root/miniconda3/bin/python3.8"

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPYCACHEPREFIX="/tmp/ge1-c7-cpu-${SLURM_JOB_ID}"
export TMPDIR="/tmp/ge1-c7-cpu-tmp-${SLURM_JOB_ID}"
mkdir -p "$TMPDIR" "$RUN_PARENT"

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

printf '{"event":"c7_pilot_validation_start","commit":"%s","job_id":"%s","host":"%s","source_clean":true,"execution_device":"cpu","operation_template_manifest_accessed":false,"operation_template_train_payload_accessed":false,"development_accessed":false,"systematic_rr_accessed":false,"test_er_accessed":false,"iid_accessed":false,"history_depth_accessed":false,"geometry_extrapolation_accessed":false,"c7_gate_completed":false,"c8_or_later_performed":false}\n' \
  "$CHECKED_OUT_COMMIT" "$SLURM_JOB_ID" "$(hostname)"

container_python -B - <<'PY'
import json
import platform
import sys
import torch

assert sys.version_info[:3] == (3, 8, 13), "Python 3.8.13 required"
assert torch.__version__.split("+")[0] == "1.11.0", "PyTorch 1.11.0 required"
assert not torch.cuda.is_available(), "CUDA must be unavailable"
assert torch.get_num_threads() == 1, "one PyTorch CPU thread required"
print(json.dumps({
    "event": "c7_pilot_environment",
    "python_version": sys.version.split()[0],
    "python_implementation": platform.python_implementation(),
    "pytorch_version": torch.__version__,
    "pytorch_cuda_build": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "execution_device": "cpu",
    "cpu_threads": torch.get_num_threads(),
}, sort_keys=True, separators=(",", ":")), flush=True)
PY

container_python -B - <<'PY'
import json
import unittest

names = (
    "prototype.graph_encoder.tests.test_c7_contract",
    "prototype.graph_encoder.tests.test_c7_runtime",
)
suite = unittest.TestSuite(
    unittest.defaultTestLoader.loadTestsFromName(name) for name in names
)
result = unittest.TextTestRunner(verbosity=2).run(suite)
print(json.dumps({
    "event": "c7_focused_suite",
    "tests_run": result.testsRun,
    "failures": len(result.failures),
    "errors": len(result.errors),
    "skipped": len(result.skipped),
}, sort_keys=True, separators=(",", ":")), flush=True)
if result.testsRun < 55 or not result.wasSuccessful() or result.skipped:
    raise SystemExit("focused C7 suite must pass with zero skips")
PY

container_python -B - <<'PY'
import json
import unittest

suite = unittest.defaultTestLoader.discover("prototype/graph_encoder/tests")
result = unittest.TextTestRunner(verbosity=1).run(suite)
print(json.dumps({
    "event": "c7_complete_graph_encoder_suite",
    "tests_run": result.testsRun,
    "failures": len(result.failures),
    "errors": len(result.errors),
    "skipped": len(result.skipped),
}, sort_keys=True, separators=(",", ":")), flush=True)
if result.testsRun < 218 or not result.wasSuccessful() or result.skipped:
    raise SystemExit("complete graph_encoder suite must pass with zero skips")
PY

printf 'model_data_regression_suite=START\n'
container_python -B -m unittest discover prototype/model_data/tests
printf 'model_data_regression_suite=PASS\n'
printf 'representation_regression_suite=START\n'
container_python -B -m unittest discover prototype/representation/tests
printf 'representation_regression_suite=PASS\n'

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
    sorted(Path("prototype/graph_encoder/tests").glob("test_c7_*.py"))
)
for path in paths:
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path), feature_version=(3, 8))
print("C7_PYTHON38_GRAMMAR_OK {}".format(len(paths)), flush=True)
PY

container_python -B - <<'PY'
import ast
from pathlib import Path

pilot_path = Path("prototype/graph_encoder/pilot.py")
source = pilot_path.read_text(encoding="utf-8")
tree = ast.parse(source)
for node in ast.walk(tree):
    if isinstance(node, ast.ImportFrom):
        assert not any(item.name == "load_development" for item in node.names)
    if isinstance(node, ast.Name):
        assert node.id != "load_development"
assert "secondary_systematic_validation" not in source
assert "load_partition_physical_examples" not in source
assert "teacher_forced(" not in source
assert "hierarchical" not in source.lower()

autonomous = Path("prototype/graph_encoder/autonomous.py").read_text(encoding="utf-8")
autonomous_tree = ast.parse(autonomous)
helper = next(
    node for node in autonomous_tree.body
    if isinstance(node, ast.FunctionDef) and node.name == "_decode_condition"
)
helper_source = ast.get_source_segment(autonomous, helper)
assert helper_source.count("model.decoder(") == 1
assert "encoder ==" not in helper_source
assert "teacher_forced" not in helper_source
print("C7_TARGET_LEAKAGE_COMMON_DECODER_SOURCE_AUDIT=PASS", flush=True)
PY

bash -n prototype/graph_encoder/adroit/ge1_pilot_cpu.slurm
git diff --check
test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git status --porcelain=v1 --untracked-files=all)"

MANIFEST_ACCESSED='"not_confirmed_on_failure"'
TRAIN_PAYLOAD_ACCESSED='"not_confirmed_on_failure"'
container_python -B -m prototype.graph_encoder.pilot \
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

container_python -B - "$OUTPUT_DIR" <<'PY'
import json
import sys
from pathlib import Path

from prototype.graph_encoder.pilot import verify_c7_artifact

root = Path(sys.argv[1])
verified = verify_c7_artifact(root)
rows = [
    json.loads(line)
    for line in (root / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    if line
]
terminal = rows[-1]
assert terminal["event"] == "c7_pilot_execution_completed"
assert terminal["c7_execution_completed"] is True
assert terminal["operation_template_manifest_accessed"] is True
assert terminal["operation_template_train_payload_accessed"] is True
for name in (
    "development_accessed", "systematic_rr_accessed", "test_er_accessed",
    "iid_accessed", "history_depth_accessed",
    "geometry_extrapolation_accessed", "c8_or_later_performed",
):
    assert terminal[name] is False
print(json.dumps({
    "event": "c7_artifact_verified",
    "overall_gate_pass": terminal["overall_gate_pass"],
    "stage6_authorized_by_c7": terminal["stage6_authorized_by_c7"],
    "preauthorized_decoder_repair_triggered": terminal[
        "preauthorized_decoder_repair_triggered"
    ],
    "artifact_manifest_sha256": verified["artifact_manifest_sha256"],
    "sha256sums_sha256": verified["sha256sums_sha256"],
    "regular_file_count": verified["regular_file_count"],
}, sort_keys=True, separators=(",", ":")), flush=True)
PY

test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git status --porcelain=v1 --untracked-files=all)"
C7_GATE_COMPLETED=true

container_python -B - "$OUTPUT_DIR" "$EXPECTED_COMMIT" "$SLURM_JOB_ID" <<'PY'
import json
import sys
from pathlib import Path

root, commit, job_id = sys.argv[1:]
rows = [
    json.loads(line)
    for line in (Path(root) / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    if line
]
terminal = rows[-1]
print(json.dumps({
    "event": "c7_pilot_terminal_execution_completed",
    "commit": commit,
    "job_id": job_id,
    "overall_gate_pass": terminal["overall_gate_pass"],
    "stage6_authorized_by_c7": terminal["stage6_authorized_by_c7"],
    "preauthorized_decoder_repair_triggered": terminal[
        "preauthorized_decoder_repair_triggered"
    ],
    "comparison_inconclusive": terminal["comparison_inconclusive"],
    "operation_template_manifest_accessed": True,
    "operation_template_train_payload_accessed": True,
    "development_accessed": False,
    "systematic_rr_accessed": False,
    "test_er_accessed": False,
    "iid_accessed": False,
    "history_depth_accessed": False,
    "geometry_extrapolation_accessed": False,
    "c7_gate_completed": True,
    "c8_or_later_performed": False,
    "artifact_path": root,
}, sort_keys=True, separators=(",", ":")), flush=True)
PY
