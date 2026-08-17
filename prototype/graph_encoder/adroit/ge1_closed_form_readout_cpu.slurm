#!/bin/bash
#SBATCH --job-name=ge1-closed-form-readout
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/network/km6349/ge1_closed_form_readout_runs/readout-cpu-%j.out
#SBATCH --error=/scratch/network/km6349/ge1_closed_form_readout_runs/readout-cpu-%j.err

set -Eeuo pipefail

JOB_START_EPOCH="$(date +%s)"
INPUT_ACCESS_STATUS="not_accessed"

terminal_failure() {
  status=$?
  trap - ERR
  printf '{"event":"closed_form_readout_terminal_failure","protocol_version":"GE1-C7-CLOSED-FORM-READOUT-v2","commit":"%s","job_id":"%s","exit_status":%s,"preserved_input_content_access_status":"%s","corpus_accessed":false,"operation_template_manifest_accessed":false,"source_repaired_checkpoint_accessed":false,"ge1_model_constructed_or_loaded":false,"ge1_training_performed":false,"ge1_backward_pass_performed":false,"repair_performed":false,"protected_partition_accessed":false,"stage6_performed":false,"c8_or_later_performed":false}\n' \
    "${EXPECTED_COMMIT:-unresolved}" "${SLURM_JOB_ID:-unresolved}" "$status" \
    "$INPUT_ACCESS_STATUS" >&2
  exit "$status"
}
trap terminal_failure ERR

: "${EXPECTED_COMMIT:?closed-form readout requires an exact authorized commit}"
: "${SLURM_JOB_ID:?closed-form readout must run under Slurm}"
REPOSITORY="${REPOSITORY:-/scratch/network/km6349/SkexGen-closed-form-readout}"
INPUT_DIR="${INPUT_DIR:-/scratch/network/km6349/ge1_representation_probe_artifacts/ge1-probe-3ea43650bb793d8d55b3d61a1edb70ec7a920089-3346513.incomplete-3346513}"
RUN_PARENT="${RUN_PARENT:-/scratch/network/km6349/ge1_closed_form_readout_artifacts}"
LOG_PARENT="/scratch/network/km6349/ge1_closed_form_readout_runs"
CONTAINER="/scratch/network/km6349/skexgen.sif"
PYTHON="/root/miniconda3/bin/python3.8"
OUTPUT_DIR="${RUN_PARENT}/ge1-closed-form-${EXPECTED_COMMIT}-${SLURM_JOB_ID}"
TIMING_RECORD="${RUN_PARENT}/timing-${EXPECTED_COMMIT}-${SLURM_JOB_ID}.json"
SOURCE_FEATURE_COMMIT="3ea43650bb793d8d55b3d61a1edb70ec7a920089"

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONPYCACHEPREFIX="/tmp/ge1-closed-form-${SLURM_JOB_ID}"
export TMPDIR="/tmp/ge1-closed-form-tmp-${SLURM_JOB_ID}"
mkdir -p "$TMPDIR" "$RUN_PARENT" "$LOG_PARENT"

preflight_python() {
  apptainer exec \
    --cleanenv \
    --env SLURM_JOB_ID="$SLURM_JOB_ID" \
    --env CUDA_VISIBLE_DEVICES="" \
    --env OMP_NUM_THREADS=1 \
    --env MKL_NUM_THREADS=1 \
    --env OPENBLAS_NUM_THREADS=1 \
    --env PYTHONPYCACHEPREFIX="$PYTHONPYCACHEPREFIX" \
    --env TMPDIR="$TMPDIR" \
    --bind "$REPOSITORY:$REPOSITORY:ro" \
    --bind "$RUN_PARENT:$RUN_PARENT" \
    "$CONTAINER" "$PYTHON" "$@"
}

readout_python() {
  apptainer exec \
    --cleanenv \
    --env SLURM_JOB_ID="$SLURM_JOB_ID" \
    --env CUDA_VISIBLE_DEVICES="" \
    --env OMP_NUM_THREADS=1 \
    --env MKL_NUM_THREADS=1 \
    --env OPENBLAS_NUM_THREADS=1 \
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
test ! -e "$TIMING_RECORD"
cd "$REPOSITORY"

CHECKED_OUT_COMMIT="$(git rev-parse HEAD)"
SOURCE_STATUS="$(git status --porcelain=v1 --untracked-files=all)"
test "$CHECKED_OUT_COMMIT" = "$EXPECTED_COMMIT"
test -z "$SOURCE_STATUS"
git cat-file -e "${SOURCE_FEATURE_COMMIT}^{commit}"

printf '{"event":"closed_form_readout_job_start","protocol_version":"GE1-C7-CLOSED-FORM-READOUT-v2","commit":"%s","job_id":"%s","host":"%s","source_clean":true,"standalone_checkout":true,"execution_device":"cpu","preserved_input_content_access_status":"not_accessed","corpus_accessed":false,"checkpoint_accessed":false,"model_accessed":false,"protected_partition_accessed":false}\n' \
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
    "event": "closed_form_environment",
    "python_version": sys.version.split()[0],
    "python_implementation": platform.python_implementation(),
    "pytorch_version": torch.__version__,
    "pytorch_cuda_build": torch.version.cuda,
    "cuda_available": False,
    "execution_device": "cpu",
    "cpu_threads": torch.get_num_threads(),
}, sort_keys=True, separators=(",", ":")), flush=True)
PY

# Every repository preflight and the timing gate runs with no preserved-input bind.
preflight_python -B - <<'PY'
import json
import unittest

names = (
    "prototype.graph_encoder.tests.test_closed_form_readout_contract",
    "prototype.graph_encoder.tests.test_closed_form_readout_runtime",
)
suite = unittest.TestSuite(
    unittest.TestLoader().loadTestsFromName(name) for name in names
)
result = unittest.TextTestRunner(verbosity=2).run(suite)
if result.testsRun < 20 or not result.wasSuccessful() or result.skipped:
    raise SystemExit("focused closed-form tests differ")
print(json.dumps({
    "event": "closed_form_focused_preflight",
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
    unittest.TestLoader().discover("prototype/graph_encoder/tests", top_level_dir=".")
)
if not result.wasSuccessful() or result.skipped:
    raise SystemExit("complete graph_encoder suite differs")
print(json.dumps({
    "event": "closed_form_graph_encoder_preflight",
    "tests_run": result.testsRun,
    "failures": len(result.failures),
    "errors": len(result.errors),
    "skips": len(result.skipped),
}, sort_keys=True, separators=(",", ":")), flush=True)
PY

preflight_python -B - <<'PY'
import json
import unittest

flat_expected = {
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
    expected = flat_expected if name == "flat_baseline" else set()
    if not result.wasSuccessful() or len(result.skipped) != expected_skips or observed != expected:
        raise SystemExit("regression suite differs: {}".format(path))
    records.append({"name": name, "tests_run": result.testsRun,
                    "skips": len(result.skipped)})
print(json.dumps({
    "event": "closed_form_regression_preflight", "suites": records,
}, sort_keys=True, separators=(",", ":")), flush=True)
PY

preflight_python -B tools/check_documentation.py
printf '{"event":"closed_form_documentation_preflight","status":"pass"}\n'
preflight_python -B -m compileall prototype/graph_encoder
printf '{"event":"closed_form_compilation_preflight","status":"pass"}\n'

preflight_python -B - <<'PY'
import ast
from pathlib import Path

root = Path("prototype/graph_encoder")
for path in tuple(sorted(root.glob("*.py"))) + tuple(sorted((root / "tests").glob("test_*.py"))):
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path), feature_version=(3, 8))
source = root / "closed_form_readout.py"
tree = ast.parse(source.read_text(encoding="utf-8"))
imports = {
    alias.name for node in ast.walk(tree)
    if isinstance(node, ast.ImportFrom) for alias in node.names
}
calls = {
    node.func.id for node in ast.walk(tree)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
}
for forbidden in (
    "load_train", "load_development", "select_c7_sufficiency_subsets",
    "load_ge1_checkpoint", "build_ge1_model", "build_matched_ge1_models",
    "run_ge1_training", "extract_target_free_features",
):
    assert forbidden not in imports
    assert forbidden not in calls
print("CLOSED_FORM_SOURCE_ACCESS_AUDIT=PASS", flush=True)
PY

preflight_python -B - <<'PY'
import prototype.graph_encoder as package
import prototype.graph_encoder.closed_form_readout as readout

assert readout.READOUT_PROTOCOL_VERSION == "GE1-C7-CLOSED-FORM-READOUT-v2"
assert "READOUT_PROTOCOL_VERSION" in package.__all__
assert "verify_readout_artifact" in package.__all__
print("CLOSED_FORM_IMPORT_EXPORT_AUDIT=PASS", flush=True)
PY

git diff --quiet "$SOURCE_FEATURE_COMMIT" "$EXPECTED_COMMIT" -- \
  prototype/graph_encoder/representation_probe.py \
  prototype/graph_encoder/repaired_sufficiency.py \
  prototype/graph_encoder/shared_decoder.py \
  prototype/graph_encoder/losses.py \
  prototype/graph_encoder/encoders.py \
  prototype/graph_encoder/model.py
bash -n prototype/graph_encoder/adroit/ge1_closed_form_readout_cpu.slurm
git diff --check
test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git status --porcelain=v1 --untracked-files=all)"
printf '{"event":"closed_form_source_preflight","status":"pass"}\n'

ELAPSED_PREFLIGHT_SECONDS="$(( $(date +%s) - JOB_START_EPOCH ))"
preflight_python -B -m prototype.graph_encoder.closed_form_readout synthetic-timing \
  --elapsed-repository-preflight-seconds "$ELAPSED_PREFLIGHT_SECONDS" \
  --output "$TIMING_RECORD"
test -s "$TIMING_RECORD"
preflight_python -B - "$TIMING_RECORD" <<'PY'
import json
from pathlib import Path
import sys

record = json.loads(Path(sys.argv[1]).read_text("utf-8"))
assert record["status"] == "pass"
assert record["selected_permutation_count"] in (999, 499)
assert record["projected_complete_seconds"][str(record["selected_permutation_count"])] <= 2700
print(json.dumps({
    "event": "closed_form_permutation_count_selected",
    "selected_permutation_count": record["selected_permutation_count"],
    "reason": record["reason"],
    "projected_complete_seconds": record["projected_complete_seconds"],
}, sort_keys=True, separators=(",", ":")), flush=True)
PY

# The preserved directory becomes available only after all preflights and timing pass.
test -d "$INPUT_DIR"
test ! -L "$INPUT_DIR"
for name in detached_features.pt feature_manifest.json labels.json
do
  test -f "$INPUT_DIR/$name"
  test ! -L "$INPUT_DIR/$name"
done

INPUT_ACCESS_STATUS="indeterminate_not_terminally_certified"
readout_python -B -m prototype.graph_encoder.closed_form_readout run \
  --input-dir "$INPUT_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --repository-root "$REPOSITORY" \
  --expected-commit "$EXPECTED_COMMIT" \
  --timing-record "$TIMING_RECORD"
INPUT_ACCESS_STATUS="verified_and_loaded"

for name in resolved_config.json readout_results.json metrics.jsonl artifact_manifest.json SHA256SUMS
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

from prototype.graph_encoder.closed_form_readout import verify_readout_artifact

verified = verify_readout_artifact(
    Path(sys.argv[1]), expected_commit=sys.argv[2], expected_slurm_job_id=sys.argv[3]
)
print(json.dumps({
    "event": "closed_form_artifact_validation", **verified,
}, sort_keys=True, separators=(",", ":")), flush=True)
PY

test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git status --porcelain=v1 --untracked-files=all)"
printf '{"event":"closed_form_readout_terminal_success","protocol_version":"GE1-C7-CLOSED-FORM-READOUT-v2","commit":"%s","job_id":"%s","preserved_input_content_access_status":"verified_and_loaded","corpus_accessed":false,"checkpoint_accessed":false,"model_accessed":false,"training_performed":false,"backward_pass_performed":false,"repair_performed":false,"protected_partition_accessed":false,"stage6_performed":false,"c8_or_later_performed":false}\n' \
  "$EXPECTED_COMMIT" "$SLURM_JOB_ID"
