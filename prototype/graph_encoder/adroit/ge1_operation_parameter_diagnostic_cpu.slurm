#!/bin/bash
#SBATCH --job-name=ge1-op-param-diagnostic
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/network/km6349/ge1_operation_parameter_diagnostic_validation_runs/operation-parameter-cpu-%j.out
#SBATCH --error=/scratch/network/km6349/ge1_operation_parameter_diagnostic_validation_runs/operation-parameter-cpu-%j.err

set -Eeuo pipefail

PHASE=preflight
MANIFEST_ACCESSED=false
TRAIN_PAYLOAD_ACCESSED=false
SCALED_PAYLOAD_ACCESSED=false
DIAGNOSTIC_COMPLETED=false

terminal_failure() {
  status=$?
  trap - ERR
  case "$PHASE" in
    preflight) classification=infrastructure_or_preflight_failure ;;
    diagnostic) classification=incomplete_diagnostic_failure ;;
    postflight) classification=artifact_or_postflight_verification_failure ;;
    *) classification=unclassified_runner_failure ;;
  esac
  printf '{"event":"operation_parameter_diagnostic_terminal_failure","failure_classification":"%s","phase":"%s","commit":"%s","job_id":"%s","exit_status":%s,"operation_template_manifest_accessed":%s,"operation_template_train_payload_accessed":%s,"scaled_train_payload_accessed":%s,"development_accessed":false,"systematic_rr_accessed":false,"test_er_accessed":false,"iid_accessed":false,"history_depth_accessed":false,"geometry_extrapolation_accessed":false,"other_corpus_accessed":false,"diagnostic_completed":false,"training_performed":false,"backward_pass_performed":false,"optimizer_step_performed":false,"checkpoint_written":false,"source_checkpoints_modified":false,"stage6_performed":false,"c8_or_later_performed":false,"decoder_repair_implemented_or_invoked":false,"geometry_repair_implemented_or_invoked":false}\n' \
    "$classification" "$PHASE" "${EXPECTED_COMMIT:-unresolved}" \
    "${SLURM_JOB_ID:-unresolved}" "$status" "$MANIFEST_ACCESSED" \
    "$TRAIN_PAYLOAD_ACCESSED" "$SCALED_PAYLOAD_ACCESSED" >&2
  exit "$status"
}
trap terminal_failure ERR

: "${EXPECTED_COMMIT:?submit with --export=ALL,EXPECTED_COMMIT=<exact-diagnostic-commit>}"
: "${SLURM_JOB_ID:?this workflow must run under Slurm}"
REPOSITORY="${REPOSITORY:-/scratch/network/km6349/SkexGen-operation-parameter-diagnostic}"
CORPUS_DIR="${CORPUS_DIR:-/scratch/network/km6349/controlled_corpora/b0-pilot-680-seed2026}"
SOURCE_C7_V2_ARTIFACT="/scratch/network/km6349/ge1_c7_v2_runs/ge1-c7-v2-e325d5ad97957c08da4a19b4261560e8a4a472a4-3344981"
RUN_PARENT="${RUN_PARENT:-/scratch/network/km6349/ge1_operation_parameter_diagnostic_runs}"
LOG_PARENT="/scratch/network/km6349/ge1_operation_parameter_diagnostic_validation_runs"
OUTPUT_DIR="${RUN_PARENT}/ge1-operation-parameter-${EXPECTED_COMMIT}-${SLURM_JOB_ID}"
CONTAINER="/scratch/network/km6349/skexgen.sif"
PYTHON="/root/miniconda3/bin/python3.8"

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPYCACHEPREFIX="/tmp/ge1-operation-parameter-${SLURM_JOB_ID}"
export TMPDIR="/tmp/ge1-operation-parameter-tmp-${SLURM_JOB_ID}"
mkdir -p "$TMPDIR" "$RUN_PARENT" "$LOG_PARENT"

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
    --bind "$CORPUS_DIR:$CORPUS_DIR:ro" \
    --bind "$SOURCE_C7_V2_ARTIFACT:$SOURCE_C7_V2_ARTIFACT:ro" \
    --bind "$RUN_PARENT:$RUN_PARENT" \
    "$CONTAINER" "$PYTHON" "$@"
}

test -d "$REPOSITORY/.git"
test ! -f "$REPOSITORY/.git"
test -d "$CORPUS_DIR"
test -d "$SOURCE_C7_V2_ARTIFACT"
test -f "$CONTAINER"
test ! -e "$OUTPUT_DIR"
test ! -L "$OUTPUT_DIR"
cd "$REPOSITORY"

CHECKED_OUT_COMMIT="$(git rev-parse HEAD)"
SOURCE_STATUS="$(git status --porcelain=v1 --untracked-files=all)"
test "$CHECKED_OUT_COMMIT" = "$EXPECTED_COMMIT"
test -z "$SOURCE_STATUS"

printf '{"event":"operation_parameter_diagnostic_validation_start","commit":"%s","source_c7_v2_commit":"e325d5ad97957c08da4a19b4261560e8a4a472a4","source_c7_v2_job_id":"3344981","job_id":"%s","host":"%s","source_clean":true,"standalone_checkout":true,"execution_device":"cpu","operation_template_manifest_accessed":false,"operation_template_train_payload_accessed":false,"scaled_train_payload_accessed":false,"development_accessed":false,"systematic_rr_accessed":false,"test_er_accessed":false,"iid_accessed":false,"history_depth_accessed":false,"geometry_extrapolation_accessed":false,"other_corpus_accessed":false,"diagnostic_completed":false,"training_performed":false,"checkpoint_written":false,"stage6_performed":false,"c8_or_later_performed":false,"decoder_repair_implemented_or_invoked":false,"geometry_repair_implemented_or_invoked":false}\n' \
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
    "event": "operation_parameter_diagnostic_environment",
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

suite = unittest.TestLoader().loadTestsFromName(
    "prototype.graph_encoder.tests."
    "test_operation_parameter_diagnostic_contract"
)
result = unittest.TextTestRunner(verbosity=2).run(suite)
print(json.dumps({
    "event": "operation_parameter_diagnostic_focused_suite",
    "tests_run": result.testsRun,
    "failures": len(result.failures),
    "errors": len(result.errors),
    "skipped": len(result.skipped),
}, sort_keys=True, separators=(",", ":")), flush=True)
if result.testsRun < 9 or not result.wasSuccessful() or result.skipped:
    raise SystemExit("focused diagnostic suite must pass with zero skips")
PY

container_python -B - <<'PY'
import json
import unittest

suite = unittest.TestLoader().discover(
    "prototype/graph_encoder/tests", top_level_dir="."
)
result = unittest.TextTestRunner(verbosity=1).run(suite)
print(json.dumps({
    "event": "operation_parameter_diagnostic_complete_graph_encoder_suite",
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
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    print(json.dumps({
        "event": "operation_parameter_diagnostic_regression_suite",
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
    sorted(Path("prototype/graph_encoder/tests").glob("test_*.py"))
)
for path in paths:
    ast.parse(
        path.read_text(encoding="utf-8"), filename=str(path),
        feature_version=(3, 8),
    )
print("OPERATION_PARAMETER_DIAGNOSTIC_PYTHON38_GRAMMAR_OK {}".format(
    len(paths)
), flush=True)
PY

container_python -B - <<'PY'
import ast
from pathlib import Path

path = Path("prototype/graph_encoder/operation_parameter_diagnostic.py")
source = path.read_text(encoding="utf-8")
tree = ast.parse(source)
imports = {
    alias.name for node in ast.walk(tree)
    if isinstance(node, ast.ImportFrom) for alias in node.names
}
calls = set()
for node in ast.walk(tree):
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name):
            calls.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            calls.add(node.func.attr)
assert "load_train" in imports
assert "load_development" not in imports
assert "load_partition_physical_examples" not in imports
assert "secondary_systematic_validation" not in source
assert not {
    "backward", "run_ge1_training", "save_training_checkpoint", "step",
    "train", "repair_decoder",
} & calls
assert source.count("load_train(") == 1
assert "FAILED_FAMILY_IDS" in source
assert "SOURCE_CHECKPOINTS" in source
print("OPERATION_PARAMETER_DIAGNOSTIC_SOURCE_ACCESS_AUDIT=PASS", flush=True)
PY

container_python -B - <<'PY'
import prototype.graph_encoder as package
from prototype.graph_encoder import operation_parameter_diagnostic as diagnostic

assert package.__all__
assert diagnostic.SOURCE_C7_V2_JOB_ID == "3344981"
assert len(diagnostic.FAILED_FAMILY_IDS) == 5
print("GRAPH_ENCODER_IMPORT_EXPORT_OK {}".format(len(package.__all__)), flush=True)
PY

bash -n prototype/graph_encoder/adroit/ge1_operation_parameter_diagnostic_cpu.slurm
git diff --check
test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git status --porcelain=v1 --untracked-files=all)"

SOURCE_HASHES_BEFORE="$TMPDIR/source-c7-v2-files.before.sha256"
SOURCE_HASHES_AFTER="$TMPDIR/source-c7-v2-files.after.sha256"
sha256sum \
  "$SOURCE_C7_V2_ARTIFACT/metrics.jsonl" \
  "$SOURCE_C7_V2_ARTIFACT/artifact_manifest.json" \
  "$SOURCE_C7_V2_ARTIFACT/checkpoints/scaled/flat/flat-seed2026-epoch0200.pt" \
  "$SOURCE_C7_V2_ARTIFACT/checkpoints/scaled/typed_graph/typed_graph-seed2026-epoch0200.pt" \
  > "$SOURCE_HASHES_BEFORE"

PHASE=diagnostic
MANIFEST_ACCESSED='"not_confirmed_on_failure"'
TRAIN_PAYLOAD_ACCESSED='"not_confirmed_on_failure"'
SCALED_PAYLOAD_ACCESSED='"not_confirmed_on_failure"'
container_python -B -m prototype.graph_encoder.operation_parameter_diagnostic run \
  --corpus-dir "$CORPUS_DIR" \
  --source-c7-v2-artifact "$SOURCE_C7_V2_ARTIFACT" \
  --output-dir "$OUTPUT_DIR" \
  --repository-root "$REPOSITORY" \
  --expected-commit "$EXPECTED_COMMIT"

PHASE=postflight
test -s "$OUTPUT_DIR/diagnosis.json"
test -s "$OUTPUT_DIR/resolved_config.json"
test -s "$OUTPUT_DIR/metrics.jsonl"
test -s "$OUTPUT_DIR/artifact_manifest.json"
test -s "$OUTPUT_DIR/SHA256SUMS"
test "$(find "$OUTPUT_DIR" -type f | wc -l | tr -d ' ')" = "5"
test -z "$(find "$OUTPUT_DIR" -name '*.pt' -print)"

(
  cd "$OUTPUT_DIR"
  sha256sum -c SHA256SUMS
)

container_python -B -m prototype.graph_encoder.operation_parameter_diagnostic verify \
  --artifact-dir "$OUTPUT_DIR" \
  --expected-commit "$EXPECTED_COMMIT" \
  --expected-slurm-job-id "$SLURM_JOB_ID"

sha256sum \
  "$SOURCE_C7_V2_ARTIFACT/metrics.jsonl" \
  "$SOURCE_C7_V2_ARTIFACT/artifact_manifest.json" \
  "$SOURCE_C7_V2_ARTIFACT/checkpoints/scaled/flat/flat-seed2026-epoch0200.pt" \
  "$SOURCE_C7_V2_ARTIFACT/checkpoints/scaled/typed_graph/typed_graph-seed2026-epoch0200.pt" \
  > "$SOURCE_HASHES_AFTER"
cmp "$SOURCE_HASHES_BEFORE" "$SOURCE_HASHES_AFTER"

container_python -B - "$OUTPUT_DIR" <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
diagnosis = json.loads((root / "diagnosis.json").read_text(encoding="utf-8"))
resolved = json.loads(
    (root / "resolved_config.json").read_text(encoding="utf-8")
)
assert len(diagnosis["family_diagnostics"]) == 5
for family in diagnosis["family_diagnostics"]:
    assert sorted(item["arm"] for item in family["arm_results"]) == [
        "flat", "typed_graph",
    ]
    for arm in family["arm_results"]:
        assert arm["executor_result"] == {
            "available": False,
            "result": None,
            "reason": "no_cad_kernel_in_evaluation_contract",
        }
        assert arm["legal_parameter_geometric_incompatibility"] == {
            "assessable": False,
            "classification": None,
            "reason": "no_cad_kernel_in_evaluation_contract",
        }
assert all(
    item["status"] == "exact_stable_reproduction_passed"
    for item in diagnosis["reproduction"]
)
assert resolved["access"]["diagnostic_completed"] is True
print(json.dumps({
    "event": "operation_parameter_diagnostic_semantic_audit_passed",
    "family_count": 5,
    "both_arms_per_family": True,
    "strict_and_analytic_validity_recorded": True,
    "executor_results_structurally_unavailable": True,
    "geometric_incompatibility_unassessable": True,
    "source_c7_v2_immutable": True,
}, sort_keys=True, separators=(",", ":")), flush=True)
PY

test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git status --porcelain=v1 --untracked-files=all)"
MANIFEST_ACCESSED=true
TRAIN_PAYLOAD_ACCESSED=true
SCALED_PAYLOAD_ACCESSED=true
DIAGNOSTIC_COMPLETED=true

printf '{"event":"operation_parameter_diagnostic_terminal_success","commit":"%s","source_c7_v2_commit":"e325d5ad97957c08da4a19b4261560e8a4a472a4","source_c7_v2_job_id":"3344981","job_id":"%s","host":"%s","diagnostic_completed":true,"source_c7_v2_result_changed":false,"source_checkpoints_modified":false,"operation_template_manifest_accessed":true,"operation_template_train_payload_accessed":true,"scaled_train_payload_accessed":true,"development_accessed":false,"systematic_rr_accessed":false,"test_er_accessed":false,"iid_accessed":false,"history_depth_accessed":false,"geometry_extrapolation_accessed":false,"other_corpus_accessed":false,"training_performed":false,"backward_pass_performed":false,"optimizer_step_performed":false,"checkpoint_written":false,"stage6_performed":false,"c8_or_later_performed":false,"decoder_repair_implemented_or_invoked":false,"geometry_repair_implemented_or_invoked":false,"artifact_path":"%s"}\n' \
  "$EXPECTED_COMMIT" "$SLURM_JOB_ID" "$(hostname)" "$OUTPUT_DIR"
