#!/bin/bash
#SBATCH --job-name=ge1-representation-screen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/network/km6349/ge1_representation_screening_runs/screen-cpu-%j.out
#SBATCH --error=/scratch/network/km6349/ge1_representation_screening_runs/screen-cpu-%j.err

set -Eeuo pipefail

INPUT_ACCESSED=false

terminal_failure() {
  status=$?
  trap - ERR
  printf '{"event":"representation_screening_terminal_failure","protocol_version":"GE1-C7-REPAIRED-REPRESENTATION-SCREEN-v1","commit":"%s","job_id":"%s","exit_status":%s,"preserved_feature_label_artifact_accessed":%s,"corpus_accessed":false,"operation_template_manifest_accessed":false,"source_repaired_artifact_accessed":false,"source_repaired_checkpoint_accessed":false,"development_accessed":false,"systematic_rr_accessed":false,"test_er_accessed":false,"iid_accessed":false,"history_depth_accessed":false,"geometry_extrapolation_accessed":false,"other_corpus_accessed":false,"ge1_model_constructed_or_loaded":false,"ge1_encoder_or_decoder_constructed_or_loaded":false,"ge1_training_performed":false,"ge1_backward_pass_performed":false,"ge1_optimizer_constructed":false,"ge1_checkpoint_written":false,"label_permutations_performed":false,"model_or_decoder_repair_implemented_or_invoked":false,"stage6_performed":false,"c8_or_later_performed":false,"representation_screening_completed":false,"formal_adr0011_representation_probe_completed":false}\n' \
    "${EXPECTED_COMMIT:-unresolved}" "${SLURM_JOB_ID:-unresolved}" "$status" \
    "$INPUT_ACCESSED" >&2
  exit "$status"
}
trap terminal_failure ERR

: "${EXPECTED_COMMIT:?screening requires an explicitly authorized exact commit}"
: "${SLURM_JOB_ID:?this workflow must run under Slurm}"
REPOSITORY="${REPOSITORY:-/scratch/network/km6349/SkexGen-representation-screening}"
INPUT_DIR="${INPUT_DIR:-/scratch/network/km6349/ge1_representation_probe_artifacts/ge1-probe-3ea43650bb793d8d55b3d61a1edb70ec7a920089-3346513.incomplete-3346513}"
RUN_PARENT="${RUN_PARENT:-/scratch/network/km6349/ge1_representation_screening_artifacts}"
OUTPUT_DIR="${RUN_PARENT}/ge1-screen-${EXPECTED_COMMIT}-${SLURM_JOB_ID}"
LOG_PARENT="/scratch/network/km6349/ge1_representation_screening_runs"
CONTAINER="/scratch/network/km6349/skexgen.sif"
PYTHON="/root/miniconda3/bin/python3.8"
SOURCE_FEATURE_COMMIT="3ea43650bb793d8d55b3d61a1edb70ec7a920089"

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPYCACHEPREFIX="/tmp/ge1-screen-${SLURM_JOB_ID}"
export TMPDIR="/tmp/ge1-screen-tmp-${SLURM_JOB_ID}"
mkdir -p "$TMPDIR" "$RUN_PARENT" "$LOG_PARENT"

preflight_python() {
  apptainer exec \
    --cleanenv \
    --env SLURM_JOB_ID="$SLURM_JOB_ID" \
    --env CUDA_VISIBLE_DEVICES="" \
    --env OMP_NUM_THREADS=1 \
    --env MKL_NUM_THREADS=1 \
    --env PYTHONPYCACHEPREFIX="$PYTHONPYCACHEPREFIX" \
    --env TMPDIR="$TMPDIR" \
    --bind "$REPOSITORY:$REPOSITORY:ro" \
    --bind "$RUN_PARENT:$RUN_PARENT" \
    "$CONTAINER" "$PYTHON" "$@"
}

screening_python() {
  apptainer exec \
    --cleanenv \
    --env SLURM_JOB_ID="$SLURM_JOB_ID" \
    --env CUDA_VISIBLE_DEVICES="" \
    --env OMP_NUM_THREADS=1 \
    --env MKL_NUM_THREADS=1 \
    --env PYTHONPYCACHEPREFIX="$PYTHONPYCACHEPREFIX" \
    --env TMPDIR="$TMPDIR" \
    --bind "$REPOSITORY:$REPOSITORY:ro" \
    --bind "$INPUT_DIR:$INPUT_DIR:ro" \
    --bind "$RUN_PARENT:$RUN_PARENT" \
    "$CONTAINER" "$PYTHON" "$@"
}

test -d "$REPOSITORY/.git"
test ! -f "$REPOSITORY/.git"
test -f "$CONTAINER"
test ! -e "$OUTPUT_DIR"
test ! -L "$OUTPUT_DIR"
cd "$REPOSITORY"

CHECKED_OUT_COMMIT="$(git rev-parse HEAD)"
SOURCE_STATUS="$(git status --porcelain=v1 --untracked-files=all)"
test "$CHECKED_OUT_COMMIT" = "$EXPECTED_COMMIT"
test -z "$SOURCE_STATUS"
git cat-file -e "${SOURCE_FEATURE_COMMIT}^{commit}"

printf '{"event":"representation_screening_start","protocol_version":"GE1-C7-REPAIRED-REPRESENTATION-SCREEN-v1","commit":"%s","job_id":"%s","host":"%s","source_clean":true,"standalone_checkout":true,"execution_device":"cpu","source_feature_commit":"3ea43650bb793d8d55b3d61a1edb70ec7a920089","source_feature_job_id":"3346513","preserved_feature_label_artifact_accessed":false,"corpus_accessed":false,"operation_template_manifest_accessed":false,"source_repaired_artifact_accessed":false,"source_repaired_checkpoint_accessed":false,"development_accessed":false,"systematic_rr_accessed":false,"test_er_accessed":false,"iid_accessed":false,"history_depth_accessed":false,"geometry_extrapolation_accessed":false,"other_corpus_accessed":false,"ge1_model_constructed_or_loaded":false,"ge1_encoder_or_decoder_constructed_or_loaded":false,"ge1_training_performed":false,"ge1_backward_pass_performed":false,"ge1_optimizer_constructed":false,"ge1_checkpoint_written":false,"label_permutations_performed":false,"model_or_decoder_repair_implemented_or_invoked":false,"stage6_performed":false,"c8_or_later_performed":false}\n' \
  "$CHECKED_OUT_COMMIT" "$SLURM_JOB_ID" "$(hostname)"

preflight_python -B - <<'PY'
import json
import platform
import sys
import torch

assert sys.version_info[:3] == (3, 8, 13)
assert torch.__version__.split("+")[0] == "1.11.0"
assert not torch.cuda.is_available()
assert torch.get_num_threads() == 1
print(json.dumps({
    "event": "representation_screening_environment",
    "python_version": sys.version.split()[0],
    "python_implementation": platform.python_implementation(),
    "pytorch_version": torch.__version__,
    "pytorch_cuda_build": torch.version.cuda,
    "cuda_available": False,
    "execution_device": "cpu",
    "cpu_threads": torch.get_num_threads(),
}, sort_keys=True, separators=(",", ":")), flush=True)
PY

# All preflights run without mounting the preserved input directory.
preflight_python -B - <<'PY'
import json
import unittest

names = (
    "prototype.graph_encoder.tests.test_representation_screening_contract",
    "prototype.graph_encoder.tests.test_representation_screening_runtime",
)
suite = unittest.TestSuite(
    unittest.TestLoader().loadTestsFromName(name) for name in names
)
result = unittest.TextTestRunner(verbosity=2).run(suite)
if result.testsRun < 10 or not result.wasSuccessful() or result.skipped:
    raise SystemExit("focused representation-screening tests differ")
print(json.dumps({
    "event": "representation_screening_focused_preflight",
    "tests_run": result.testsRun,
    "failures": len(result.failures),
    "errors": len(result.errors),
    "skips": len(result.skipped),
}, sort_keys=True, separators=(",", ":")), flush=True)
PY
preflight_python -B - <<'PY'
import json
import unittest

result = unittest.TextTestRunner(verbosity=1).run(
    unittest.TestLoader().discover(
        "prototype/graph_encoder/tests", top_level_dir="."
    )
)
if not result.wasSuccessful() or result.skipped:
    raise SystemExit("complete graph_encoder suite differs")
print(json.dumps({
    "event": "representation_screening_complete_graph_encoder_preflight",
    "tests_run": result.testsRun,
    "failures": len(result.failures),
    "errors": len(result.errors),
    "skips": len(result.skipped),
}, sort_keys=True, separators=(",", ":")), flush=True)
PY
preflight_python -B - <<'PY'
import json
import unittest

flat_expected_methods = {
    "test_five_capsule_regressions_are_a_pinned_golden_cohort",
    "test_malformed_parent_fails_before_scientific_loading",
    "test_parent_equivalent_arms_reconcile_pinned_records",
    "test_real_bundle_two_family_golden_reconciliation",
}
records = []
for name, path, expected_skips in (
    ("model_data", "prototype/model_data/tests", 0),
    ("flat_baseline", "prototype/flat_baseline/tests", 4),
    ("graph_baseline", "prototype/graph_baseline/tests", 0),
    ("representation", "prototype/representation/tests", 0),
    ("controlled_data", "prototype/controlled_data/tests", 0),
):
    loader = unittest.TestLoader()
    result = unittest.TextTestRunner(verbosity=1).run(
        loader.discover(path, top_level_dir=".")
    )
    observed = {item._testMethodName for item, unused_reason in result.skipped}
    expected = flat_expected_methods if name == "flat_baseline" else set()
    if (
        not result.wasSuccessful()
        or len(result.skipped) != expected_skips
        or observed != expected
    ):
        raise SystemExit("regression suite differs: {}".format(path))
    records.append({"name": name, "tests_run": result.testsRun,
                    "skips": len(result.skipped)})
print(json.dumps({
    "event": "representation_screening_regression_preflight", "suites": records,
}, sort_keys=True, separators=(",", ":")), flush=True)
PY
preflight_python -B tools/check_documentation.py
preflight_python -B -m compileall prototype/graph_encoder
preflight_python -B - <<'PY'
import ast
from pathlib import Path

root = Path("prototype/graph_encoder")
paths = tuple(sorted(root.glob("*.py"))) + tuple(
    sorted((root / "tests").glob("test_*.py"))
)
for path in paths:
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path),
              feature_version=(3, 8))
screen_path = root / "representation_screening.py"
tree = ast.parse(screen_path.read_text(encoding="utf-8"))
imports = {
    alias.name for node in ast.walk(tree)
    if isinstance(node, ast.ImportFrom) for alias in node.names
}
calls = {
    node.func.id for node in ast.walk(tree)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
}
for forbidden in (
    "run_permutation_controls", "run_representation_probe",
    "extract_target_free_features", "load_train",
    "select_c7_sufficiency_subsets", "load_ge1_checkpoint",
    "build_ge1_model", "build_matched_ge1_models",
):
    assert forbidden not in imports
    assert forbidden not in calls
assert "fit_probe_pipeline" in imports
assert "fit_probe_pipeline" in calls
print("REPRESENTATION_SCREENING_SOURCE_AUDIT=PASS", flush=True)
PY
preflight_python -B - <<'PY'
import prototype.graph_encoder as package
import prototype.graph_encoder.representation_screening as screen

assert screen.SCREEN_PROTOCOL_VERSION == (
    "GE1-C7-REPAIRED-REPRESENTATION-SCREEN-v1"
)
assert "SCREEN_PROTOCOL_VERSION" in package.__all__
assert "verify_screening_artifact" in package.__all__
print("REPRESENTATION_SCREENING_IMPORT_EXPORT_AUDIT=PASS", flush=True)
PY
git diff --quiet "$SOURCE_FEATURE_COMMIT" "$EXPECTED_COMMIT" -- \
  prototype/graph_encoder/representation_probe.py \
  prototype/graph_encoder/repaired_sufficiency.py \
  prototype/graph_encoder/c7_v2.py \
  prototype/graph_encoder/operation_parameter_diagnostic.py \
  prototype/graph_encoder/shared_decoder.py \
  prototype/graph_encoder/losses.py \
  prototype/graph_encoder/encoders.py \
  prototype/graph_encoder/model.py
bash -n prototype/graph_encoder/adroit/ge1_representation_screening_cpu.slurm
git diff --check
test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git status --porcelain=v1 --untracked-files=all)"

test -d "$INPUT_DIR"
test ! -L "$INPUT_DIR"
for name in detached_features.pt feature_manifest.json labels.json
do
  test -f "$INPUT_DIR/$name"
  test ! -L "$INPUT_DIR/$name"
done

INPUT_ACCESSED='"not_confirmed_on_failure"'
screening_python -B -m prototype.graph_encoder.representation_screening \
  --input-dir "$INPUT_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --repository-root "$REPOSITORY" \
  --expected-commit "$EXPECTED_COMMIT"

for name in \
  resolved_config.json screening_results.json metrics.jsonl \
  artifact_manifest.json SHA256SUMS
do
  test -s "$OUTPUT_DIR/$name"
done
(
  cd "$OUTPUT_DIR"
  sha256sum -c SHA256SUMS
)

preflight_python -B - "$OUTPUT_DIR" "$EXPECTED_COMMIT" "$SLURM_JOB_ID" <<'PY'
import json
from pathlib import Path
import sys

from prototype.graph_encoder.representation_screening import (
    verify_screening_artifact,
)

root = Path(sys.argv[1])
verified = verify_screening_artifact(
    root,
    expected_commit=sys.argv[2],
    expected_slurm_job_id=sys.argv[3],
)
results = json.loads((root / "screening_results.json").read_text("utf-8"))
assert len(results["pipelines"]) == 16
assert results["screening_summary"]["permutation_significance_available"] is False
assert results["screening_summary"]["adr0011_representation_probe_complete"] is False
print(json.dumps({
    "event": "representation_screening_artifact_verified", **verified,
}, sort_keys=True, separators=(",", ":")), flush=True)
PY

INPUT_ACCESSED=true
test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git status --porcelain=v1 --untracked-files=all)"
printf '{"event":"representation_screening_terminal_success","protocol_version":"GE1-C7-REPAIRED-REPRESENTATION-SCREEN-v1","commit":"%s","job_id":"%s","host":"%s","preserved_feature_label_artifact_accessed":true,"corpus_accessed":false,"operation_template_manifest_accessed":false,"source_repaired_artifact_accessed":false,"source_repaired_checkpoint_accessed":false,"development_accessed":false,"systematic_rr_accessed":false,"test_er_accessed":false,"iid_accessed":false,"history_depth_accessed":false,"geometry_extrapolation_accessed":false,"other_corpus_accessed":false,"ge1_model_constructed_or_loaded":false,"ge1_encoder_or_decoder_constructed_or_loaded":false,"ge1_training_performed":false,"ge1_backward_pass_performed":false,"ge1_optimizer_constructed":false,"ge1_checkpoint_written":false,"disposable_probe_optimization_performed":true,"label_permutations_performed":false,"model_or_decoder_repair_implemented_or_invoked":false,"categorical_head_implemented":false,"geometry_loss_changed":false,"calibration_repair_implemented":false,"stage6_performed":false,"c8_or_later_performed":false,"representation_screening_completed":true,"formal_adr0011_representation_probe_completed":false}\n' \
  "$EXPECTED_COMMIT" "$SLURM_JOB_ID" "$(hostname)"
