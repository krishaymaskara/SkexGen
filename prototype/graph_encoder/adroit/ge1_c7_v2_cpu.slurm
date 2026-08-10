#!/bin/bash
#SBATCH --job-name=ge1-c7-v2-gates
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/network/km6349/c7_v2_validation_runs/c7-v2-cpu-%j.out
#SBATCH --error=/scratch/network/km6349/c7_v2_validation_runs/c7-v2-cpu-%j.err

set -Eeuo pipefail

MANIFEST_ACCESSED=false
TRAIN_PAYLOAD_ACCESSED=false
TINY_PAYLOAD_ACCESSED=false
SCALED_PAYLOAD_ACCESSED=false
C7_V2_COMPLETED=false

terminal_failure() {
  status=$?
  trap - ERR
  printf '{"event":"c7_v2_terminal_failure","commit":"%s","job_id":"%s","exit_status":%s,"operation_template_manifest_accessed":%s,"operation_template_train_payload_accessed":%s,"tiny_train_payload_accessed":%s,"scaled_train_payload_accessed":%s,"development_accessed":false,"systematic_rr_accessed":false,"test_er_accessed":false,"iid_accessed":false,"history_depth_accessed":false,"geometry_extrapolation_accessed":false,"c7_v2_execution_completed":false,"stage6_performed":false,"c8_or_later_performed":false,"decoder_repair_implemented_or_invoked":false}\n' \
    "${EXPECTED_COMMIT:-unresolved}" "${SLURM_JOB_ID:-unresolved}" "$status" \
    "$MANIFEST_ACCESSED" "$TRAIN_PAYLOAD_ACCESSED" \
    "$TINY_PAYLOAD_ACCESSED" "$SCALED_PAYLOAD_ACCESSED" >&2
  exit "$status"
}
trap terminal_failure ERR

: "${EXPECTED_COMMIT:?submit with --export=ALL,EXPECTED_COMMIT=<exact-C7-v2-commit>}"
: "${SLURM_JOB_ID:?this workflow must run under Slurm}"
REPOSITORY="${REPOSITORY:-/scratch/network/km6349/SkexGen-c7-v2}"
CORPUS_DIR="${CORPUS_DIR:-/scratch/network/km6349/controlled_corpora/b0-pilot-680-seed2026}"
RUN_PARENT="${RUN_PARENT:-/scratch/network/km6349/ge1_c7_v2_runs}"
OUTPUT_DIR="${RUN_PARENT}/ge1-c7-v2-${EXPECTED_COMMIT}-${SLURM_JOB_ID}"
CONTAINER="/scratch/network/km6349/skexgen.sif"
PYTHON="/root/miniconda3/bin/python3.8"

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPYCACHEPREFIX="/tmp/ge1-c7-v2-cpu-${SLURM_JOB_ID}"
export TMPDIR="/tmp/ge1-c7-v2-cpu-tmp-${SLURM_JOB_ID}"
mkdir -p "$TMPDIR" "$RUN_PARENT" /scratch/network/km6349/c7_v2_validation_runs

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
test ! -f "$REPOSITORY/.git"
test -d "$CORPUS_DIR"
test -f "$CONTAINER"
test ! -e "$OUTPUT_DIR"
test ! -L "$OUTPUT_DIR"
cd "$REPOSITORY"

CHECKED_OUT_COMMIT="$(git rev-parse HEAD)"
SOURCE_STATUS="$(git status --porcelain=v1 --untracked-files=all)"
test "$CHECKED_OUT_COMMIT" = "$EXPECTED_COMMIT"
test -z "$SOURCE_STATUS"

printf '{"event":"c7_v2_validation_start","commit":"%s","job_id":"%s","host":"%s","source_clean":true,"standalone_checkout":true,"execution_device":"cpu","operation_template_manifest_accessed":false,"operation_template_train_payload_accessed":false,"tiny_train_payload_accessed":false,"scaled_train_payload_accessed":false,"development_accessed":false,"systematic_rr_accessed":false,"test_er_accessed":false,"iid_accessed":false,"history_depth_accessed":false,"geometry_extrapolation_accessed":false,"c7_v2_execution_completed":false,"stage6_performed":false,"c8_or_later_performed":false,"decoder_repair_implemented_or_invoked":false}\n' \
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
    "event": "c7_v2_environment",
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
    "prototype.graph_encoder.tests.test_c7_v2_contract",
    "prototype.graph_encoder.tests.test_c7_v2_runtime",
)
suite = unittest.TestSuite(
    unittest.defaultTestLoader.loadTestsFromName(name) for name in names
)
result = unittest.TextTestRunner(verbosity=2).run(suite)
print(json.dumps({
    "event": "c7_v2_focused_suite",
    "tests_run": result.testsRun,
    "failures": len(result.failures),
    "errors": len(result.errors),
    "skipped": len(result.skipped),
}, sort_keys=True, separators=(",", ":")), flush=True)
if result.testsRun < 25 or not result.wasSuccessful() or result.skipped:
    raise SystemExit("focused C7-v2 suite must pass with zero skips")
PY

container_python -B - <<'PY'
import json
import unittest

suite = unittest.defaultTestLoader.discover("prototype/graph_encoder/tests")
result = unittest.TextTestRunner(verbosity=1).run(suite)
print(json.dumps({
    "event": "c7_v2_complete_graph_encoder_suite",
    "tests_run": result.testsRun,
    "failures": len(result.failures),
    "errors": len(result.errors),
    "skipped": len(result.skipped),
}, sort_keys=True, separators=(",", ":")), flush=True)
if not result.wasSuccessful() or result.skipped:
    raise SystemExit("complete graph_encoder suite must pass with zero skips")
PY

container_python -B - <<'PY'
import json
import unittest

suites = (
    ("model_data", "prototype/model_data/tests", 0),
    ("flat_baseline", "prototype/flat_baseline/tests", 4),
    ("graph_baseline", "prototype/graph_baseline/tests", 0),
    ("representation", "prototype/representation/tests", 0),
    ("controlled_data", "prototype/controlled_data/tests", 0),
)
for name, path, expected_skips in suites:
    loader = unittest.TestLoader()
    suite = loader.discover(path, top_level_dir=".")
    result = unittest.TextTestRunner(verbosity=1).run(
        suite
    )
    print(json.dumps({
        "event": "c7_v2_regression_suite",
        "suite": name,
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
        "expected_skips": expected_skips,
    }, sort_keys=True, separators=(",", ":")), flush=True)
    if not result.wasSuccessful() or len(result.skipped) != expected_skips:
        raise SystemExit("{} regression suite differs".format(name))
PY

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
    sorted(Path("prototype/graph_encoder/tests").glob("test_c7_v2*.py"))
)
for path in paths:
    ast.parse(
        path.read_text(encoding="utf-8"), filename=str(path),
        feature_version=(3, 8),
    )
print("C7_V2_PYTHON38_GRAMMAR_OK {}".format(len(paths)), flush=True)
PY

container_python -B - <<'PY'
import ast
from pathlib import Path

path = Path("prototype/graph_encoder/c7_v2.py")
source = path.read_text(encoding="utf-8")
tree = ast.parse(source)
imports = {
    alias.name for node in ast.walk(tree)
    if isinstance(node, ast.ImportFrom) for alias in node.names
}
assert "load_development" not in imports
assert "load_partition_physical_examples" not in imports
assert "secondary_systematic_validation" not in source
assert "teacher_forced(" not in source
called = set()
for node in ast.walk(tree):
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name):
            called.add(node.func.id.lower())
        elif isinstance(node.func, ast.Attribute):
            called.add(node.func.attr.lower())
assert not {
    name for name in called
    if name.startswith("c8") or name.startswith("stage6")
    or "hierarchical" in name or name == "repair_decoder"
}
assert "C7_V2_CHECKPOINT_EPOCH = 200" in source
assert "fixed_protocol_final_epoch=C7_V2_CHECKPOINT_EPOCH" in source
assert "resume_checkpoint=" not in source
print("C7_V2_SOURCE_ACCESS_AUDIT=PASS", flush=True)
PY

container_python -B - <<'PY'
import prototype.graph_encoder as package
from prototype.graph_encoder import c7_v2

assert package.__all__
assert c7_v2.C7_V2_CHECKPOINT_EPOCH == 200
print("GRAPH_ENCODER_IMPORT_EXPORT_OK {}".format(len(package.__all__)), flush=True)
PY

bash -n prototype/graph_encoder/adroit/ge1_c7_v2_cpu.slurm
git diff --check
test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git status --porcelain=v1 --untracked-files=all)"

MANIFEST_ACCESSED='"not_confirmed_on_failure"'
TRAIN_PAYLOAD_ACCESSED='"not_confirmed_on_failure"'
TINY_PAYLOAD_ACCESSED='"not_confirmed_on_failure"'
SCALED_PAYLOAD_ACCESSED='"not_confirmed_on_failure"'
container_python -B -m prototype.graph_encoder.c7_v2 \
  --corpus-dir "$CORPUS_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --repository-root "$REPOSITORY" \
  --expected-commit "$EXPECTED_COMMIT"

test -s "$OUTPUT_DIR/resolved_config.json"
test -s "$OUTPUT_DIR/metrics.jsonl"
test -d "$OUTPUT_DIR/checkpoints"
test -s "$OUTPUT_DIR/artifact_manifest.json"
test -s "$OUTPUT_DIR/SHA256SUMS"

(
  cd "$OUTPUT_DIR"
  sha256sum -c SHA256SUMS
)

container_python -B - "$OUTPUT_DIR" "$EXPECTED_COMMIT" "$SLURM_JOB_ID" <<'PY'
import json
from pathlib import Path
import sys
import torch

from prototype.graph_encoder.c7_v2 import verify_c7_v2_artifact

root = Path(sys.argv[1])
commit = sys.argv[2]
job_id = sys.argv[3]
verified = verify_c7_v2_artifact(
    root, expected_commit=commit, expected_slurm_job_id=job_id
)
resolved = json.loads((root / "resolved_config.json").read_text(encoding="utf-8"))
events = [
    json.loads(line)
    for line in (root / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    if line
]
terminal = events[-1]
assert resolved["c7_v2_configuration"]["epochs"] == 200
assert resolved["training_arithmetic"]["tiny"]["optimizer_steps"] == 200
assert resolved["training_arithmetic"]["tiny"]["training_example_presentations"] == 800
assert resolved["training_arithmetic"]["scaled"]["optimizer_steps"] == 800
assert resolved["training_arithmetic"]["scaled"]["training_example_presentations"] == 6400
assert resolved["training_arithmetic"]["eventual_stage6_if_authorized"]["training_example_presentations"] == 81400
assert resolved["historical_records"]["formal_c7_v1"]["immutable_result_changed"] is False
assert resolved["historical_records"]["optimization_diagnostic"]["result_changed"] is False
assert terminal["event"] == "c7_v2_execution_completed"
assert terminal["c7_v2_execution_completed"] is True
assert terminal["operation_template_manifest_accessed"] is True
assert terminal["operation_template_train_payload_accessed"] is True
assert terminal["tiny_train_payload_accessed"] is True
for name in (
    "development_accessed", "systematic_rr_accessed", "test_er_accessed",
    "iid_accessed", "history_depth_accessed", "geometry_extrapolation_accessed",
    "stage6_performed", "c8_or_later_performed",
    "decoder_repair_implemented_or_invoked",
):
    assert terminal[name] is False
checkpoint_paths = sorted(root.glob("checkpoints/*/*/*.pt"))
assert len(checkpoint_paths) in (2, 4)
for path in checkpoint_paths:
    payload = torch.load(str(path), map_location="cpu")
    assert payload["completed_epoch"] == 200
    assert payload["selected_checkpoint_epoch"] == 200
    assert payload["selected_experimental_checkpoint"] is True
    assert payload["provenance"]["git_commit"] == commit
    assert payload["provenance"]["slurm_job_id"] == job_id
print(json.dumps({
    "event": "c7_v2_artifact_verified",
    "commit": commit,
    "job_id": job_id,
    "c7_v2_overall_gate_pass": terminal["c7_v2_overall_gate_pass"],
    "stage6_authorized_by_c7_v2": terminal["stage6_authorized_by_c7_v2"],
    "preauthorized_decoder_repair_path_next": terminal[
        "preauthorized_decoder_repair_path_next"
    ],
    "comparison_inconclusive": terminal["comparison_inconclusive"],
    "checkpoint_count": len(checkpoint_paths),
    "artifact_manifest_sha256": verified["artifact_manifest_sha256"],
    "sha256sums_sha256": verified["sha256sums_sha256"],
    "regular_file_count": verified["regular_file_count"],
}, sort_keys=True, separators=(",", ":")), flush=True)
PY

test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git status --porcelain=v1 --untracked-files=all)"
MANIFEST_ACCESSED=true
TRAIN_PAYLOAD_ACCESSED=true
TINY_PAYLOAD_ACCESSED=true
C7_V2_COMPLETED=true

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
    "event": "c7_v2_terminal_success",
    "commit": commit,
    "job_id": job_id,
    "c7_v2_overall_gate_pass": terminal["c7_v2_overall_gate_pass"],
    "stage6_authorized_by_c7_v2": terminal["stage6_authorized_by_c7_v2"],
    "preauthorized_decoder_repair_path_next": terminal[
        "preauthorized_decoder_repair_path_next"
    ],
    "comparison_inconclusive": terminal["comparison_inconclusive"],
    "operation_template_manifest_accessed": True,
    "operation_template_train_payload_accessed": True,
    "tiny_train_payload_accessed": True,
    "scaled_train_payload_accessed": terminal["scaled_train_payload_accessed"],
    "development_accessed": False,
    "systematic_rr_accessed": False,
    "test_er_accessed": False,
    "iid_accessed": False,
    "history_depth_accessed": False,
    "geometry_extrapolation_accessed": False,
    "c7_v2_execution_completed": True,
    "stage6_performed": False,
    "c8_or_later_performed": False,
    "decoder_repair_implemented_or_invoked": False,
    "artifact_path": root,
}, sort_keys=True, separators=(",", ":")), flush=True)
PY
