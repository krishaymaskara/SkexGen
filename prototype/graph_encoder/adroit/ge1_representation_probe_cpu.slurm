#!/bin/bash
#SBATCH --job-name=ge1-representation-probe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/network/km6349/ge1_representation_probe_runs/probe-cpu-%j.out
#SBATCH --error=/scratch/network/km6349/ge1_representation_probe_runs/probe-cpu-%j.err

set -Eeuo pipefail

SOURCE_ARTIFACT_ACCESSED=false
SOURCE_CHECKPOINT_ACCESSED=false
MANIFEST_ACCESSED=false
TRAIN_PAYLOAD_ACCESSED=false
SCALED_PAYLOAD_ACCESSED=false

terminal_failure() {
  status=$?
  trap - ERR
  printf '{"event":"representation_probe_terminal_failure","commit":"%s","job_id":"%s","exit_status":%s,"source_repaired_artifact_accessed":%s,"source_repaired_checkpoint_accessed":%s,"operation_template_manifest_accessed":%s,"operation_template_train_payload_accessed":%s,"scaled_train_payload_accessed":%s,"development_accessed":false,"systematic_rr_accessed":false,"test_er_accessed":false,"iid_accessed":false,"history_depth_accessed":false,"geometry_extrapolation_accessed":false,"other_corpus_accessed":false,"ge1_training_performed":false,"ge1_backward_pass_performed":false,"ge1_optimizer_constructed":false,"ge1_checkpoint_written":false,"model_or_decoder_repair_implemented_or_invoked":false,"stage6_performed":false,"c8_or_later_performed":false,"representation_probe_completed":false}\n' \
    "${EXPECTED_COMMIT:-unresolved}" "${SLURM_JOB_ID:-unresolved}" "$status" \
    "$SOURCE_ARTIFACT_ACCESSED" "$SOURCE_CHECKPOINT_ACCESSED" \
    "$MANIFEST_ACCESSED" "$TRAIN_PAYLOAD_ACCESSED" "$SCALED_PAYLOAD_ACCESSED" >&2
  exit "$status"
}
trap terminal_failure ERR

: "${EXPECTED_COMMIT:?probe execution requires an explicitly authorized exact commit}"
: "${SLURM_JOB_ID:?this workflow must run under Slurm}"
REPOSITORY="${REPOSITORY:-/scratch/network/km6349/SkexGen-representation-probe}"
CORPUS_DIR="${CORPUS_DIR:-/scratch/network/km6349/controlled_corpora/b0-pilot-680-seed2026}"
SOURCE_ARTIFACT="${SOURCE_ARTIFACT:-/scratch/network/km6349/ge1_repaired_sufficiency_artifacts/ge1-repaired-705eb820f7a15d64fe650df7350b1553b2bc8172-3345280}"
RUN_PARENT="${RUN_PARENT:-/scratch/network/km6349/ge1_representation_probe_artifacts}"
OUTPUT_DIR="${RUN_PARENT}/ge1-probe-${EXPECTED_COMMIT}-${SLURM_JOB_ID}"
LOG_PARENT="/scratch/network/km6349/ge1_representation_probe_runs"
CONTAINER="/scratch/network/km6349/skexgen.sif"
PYTHON="/root/miniconda3/bin/python3.8"
GOVERNANCE_COMMIT="b5be3dddf25f5e320e61262af030228dec331b91"

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPYCACHEPREFIX="/tmp/ge1-probe-${SLURM_JOB_ID}"
export TMPDIR="/tmp/ge1-probe-tmp-${SLURM_JOB_ID}"
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
    --bind "$REPOSITORY:$REPOSITORY:ro" \
    --bind "$RUN_PARENT:$RUN_PARENT" \
    "$CONTAINER" "$PYTHON" "$@"
}

scientific_python() {
  apptainer exec \
    --cleanenv \
    --env SLURM_JOB_ID="$SLURM_JOB_ID" \
    --env CUDA_VISIBLE_DEVICES="" \
    --env OMP_NUM_THREADS=1 \
    --env MKL_NUM_THREADS=1 \
    --env PYTHONPYCACHEPREFIX="$PYTHONPYCACHEPREFIX" \
    --env TMPDIR="$TMPDIR" \
    --bind "$REPOSITORY:$REPOSITORY:ro" \
    --bind "$CORPUS_DIR:$CORPUS_DIR:ro" \
    --bind "$SOURCE_ARTIFACT:$SOURCE_ARTIFACT:ro" \
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
git cat-file -e "${GOVERNANCE_COMMIT}^{commit}"

printf '{"event":"representation_probe_start","protocol_version":"GE1-C7-REPAIRED-REPRESENTATION-PROBE-v1","commit":"%s","job_id":"%s","host":"%s","source_clean":true,"standalone_checkout":true,"execution_device":"cpu","source_repaired_commit":"705eb820f7a15d64fe650df7350b1553b2bc8172","source_repaired_job_id":"3345280","source_repaired_result_changed":false,"source_repaired_artifact_accessed":false,"source_repaired_checkpoint_accessed":false,"operation_template_manifest_accessed":false,"operation_template_train_payload_accessed":false,"scaled_train_payload_accessed":false,"development_accessed":false,"systematic_rr_accessed":false,"test_er_accessed":false,"iid_accessed":false,"history_depth_accessed":false,"geometry_extrapolation_accessed":false,"other_corpus_accessed":false,"ge1_training_performed":false,"ge1_backward_pass_performed":false,"ge1_optimizer_constructed":false,"ge1_checkpoint_written":false,"model_or_decoder_repair_implemented_or_invoked":false,"stage6_performed":false,"c8_or_later_performed":false}\n' \
  "$CHECKED_OUT_COMMIT" "$SLURM_JOB_ID" "$(hostname)"

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
    "event": "representation_probe_environment",
    "python_version": sys.version.split()[0],
    "python_implementation": platform.python_implementation(),
    "pytorch_version": torch.__version__,
    "pytorch_cuda_build": torch.version.cuda,
    "cuda_available": False,
    "execution_device": "cpu",
    "cpu_threads": torch.get_num_threads(),
}, sort_keys=True, separators=(",", ":")), flush=True)
PY

# Every preflight completes before source artifact, manifest, checkpoint, or
# corpus access. Submission of this prepared runner remains separately
# unauthorized by ADR-0011 acceptance alone.
container_python -B - <<'PY'
import json
import unittest

names = (
    "prototype.graph_encoder.tests.test_representation_probe_contract",
    "prototype.graph_encoder.tests.test_representation_probe_runtime",
)
suite = unittest.TestSuite(
    unittest.TestLoader().loadTestsFromName(name) for name in names
)
result = unittest.TextTestRunner(verbosity=2).run(suite)
if result.testsRun < 20 or not result.wasSuccessful() or result.skipped:
    raise SystemExit("focused representation-probe tests differ")
print(json.dumps({
    "event": "representation_probe_focused_preflight",
    "tests_run": result.testsRun,
    "failures": len(result.failures),
    "errors": len(result.errors),
    "skips": len(result.skipped),
}, sort_keys=True, separators=(",", ":")), flush=True)
PY
container_python -B - <<'PY'
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
    "event": "representation_probe_complete_graph_encoder_preflight",
    "tests_run": result.testsRun,
    "failures": len(result.failures),
    "errors": len(result.errors),
    "skips": len(result.skipped),
}, sort_keys=True, separators=(",", ":")), flush=True)
PY
container_python -B - <<'PY'
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
    "event": "representation_probe_regression_preflight", "suites": records,
}, sort_keys=True, separators=(",", ":")), flush=True)
PY
container_python -B tools/check_documentation.py
container_python -B -m compileall prototype/graph_encoder
container_python -B - <<'PY'
import ast
from pathlib import Path

root = Path("prototype/graph_encoder")
paths = tuple(sorted(root.glob("*.py"))) + tuple(
    sorted((root / "tests").glob("test_*.py"))
)
for path in paths:
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path),
              feature_version=(3, 8))
source = (root / "representation_probe.py").read_text(encoding="utf-8")
tree = ast.parse(source)
imports = {
    alias.name for node in ast.walk(tree)
    if isinstance(node, ast.ImportFrom) for alias in node.names
}
assert "load_development" not in imports
assert "load_partition_physical_examples" not in imports
assert "run_ge1_training" not in imports
assert "save_ge1_checkpoint" not in imports
assert "save_training_checkpoint" not in imports
assert "secondary_systematic_validation" not in source
extractor = next(
    node for node in tree.body
    if isinstance(node, ast.FunctionDef)
    and node.name == "extract_target_free_features"
)
arguments = [item.arg for item in extractor.args.args + extractor.args.kwonlyargs]
assert "target" not in arguments and "targets" not in arguments
print("REPRESENTATION_PROBE_SOURCE_AUDIT=PASS", flush=True)
PY
container_python -B - <<'PY'
import prototype.graph_encoder as package
import prototype.graph_encoder.representation_probe as probe

assert probe.PROBE_PROTOCOL_VERSION == (
    "GE1-C7-REPAIRED-REPRESENTATION-PROBE-v1"
)
assert "PROBE_FEATURE_VERSION" in package.__all__
print("REPRESENTATION_PROBE_IMPORT_EXPORT_AUDIT=PASS", flush=True)
PY
git diff --quiet "$GOVERNANCE_COMMIT" "$EXPECTED_COMMIT" -- \
  prototype/graph_encoder/repaired_sufficiency.py \
  prototype/graph_encoder/c7_v2.py \
  prototype/graph_encoder/operation_parameter_diagnostic.py \
  prototype/graph_encoder/shared_decoder.py \
  prototype/graph_encoder/losses.py \
  prototype/graph_encoder/encoders.py
bash -n prototype/graph_encoder/adroit/ge1_representation_probe_cpu.slurm
git diff --check
test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git status --porcelain=v1 --untracked-files=all)"

test -d "$CORPUS_DIR"
test ! -L "$CORPUS_DIR"
test -d "$SOURCE_ARTIFACT"
test ! -L "$SOURCE_ARTIFACT"
SOURCE_ARTIFACT_ACCESSED='"not_confirmed_on_failure"'
SOURCE_CHECKPOINT_ACCESSED='"not_confirmed_on_failure"'
MANIFEST_ACCESSED='"not_confirmed_on_failure"'
TRAIN_PAYLOAD_ACCESSED='"not_confirmed_on_failure"'
SCALED_PAYLOAD_ACCESSED='"not_confirmed_on_failure"'
scientific_python -B -m prototype.graph_encoder.representation_probe \
  --corpus-dir "$CORPUS_DIR" \
  --source-artifact "$SOURCE_ARTIFACT" \
  --output-dir "$OUTPUT_DIR" \
  --repository-root "$REPOSITORY" \
  --expected-commit "$EXPECTED_COMMIT"

for name in \
  resolved_config.json detached_features.pt feature_manifest.json labels.json \
  scalar_analysis.json probe_results.json metrics.jsonl artifact_manifest.json \
  SHA256SUMS
do
  test -s "$OUTPUT_DIR/$name"
done
(
  cd "$OUTPUT_DIR"
  sha256sum -c SHA256SUMS
)

container_python -B - "$OUTPUT_DIR" "$EXPECTED_COMMIT" "$SLURM_JOB_ID" <<'PY'
import json
from pathlib import Path
import sys

from prototype.graph_encoder.representation_probe import verify_probe_artifact

root = Path(sys.argv[1])
commit = sys.argv[2]
job_id = sys.argv[3]
verified = verify_probe_artifact(
    root, expected_commit=commit, expected_slurm_job_id=job_id
)
events = [
    json.loads(line)
    for line in (root / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    if line
]
terminal = events[-1]
assert terminal["event"] == "representation_probe_completed"
assert terminal["source_repaired_result_changed"] is False
assert terminal["feature_row_count"] == 96
assert terminal["label_row_count"] == 48
assert terminal["pipeline_count"] == 16
for name in (
    "development_accessed", "systematic_rr_accessed", "test_er_accessed",
    "iid_accessed", "history_depth_accessed", "geometry_extrapolation_accessed",
    "other_corpus_accessed", "ge1_training_performed",
    "ge1_backward_pass_performed", "ge1_optimizer_constructed",
    "ge1_checkpoint_written", "model_or_decoder_repair_implemented_or_invoked",
    "stage6_performed", "c8_or_later_performed",
):
    assert terminal[name] is False
print(json.dumps({
    "event": "representation_probe_artifact_verified",
    "commit": commit,
    "job_id": job_id,
    **verified,
}, sort_keys=True, separators=(",", ":")), flush=True)
PY

SOURCE_ARTIFACT_ACCESSED=true
SOURCE_CHECKPOINT_ACCESSED=true
MANIFEST_ACCESSED=true
TRAIN_PAYLOAD_ACCESSED=true
SCALED_PAYLOAD_ACCESSED=true
test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git status --porcelain=v1 --untracked-files=all)"
printf '{"event":"representation_probe_terminal_success","protocol_version":"GE1-C7-REPAIRED-REPRESENTATION-PROBE-v1","commit":"%s","job_id":"%s","host":"%s","source_repaired_result_changed":false,"source_repaired_artifact_accessed":true,"source_repaired_checkpoint_accessed":true,"operation_template_manifest_accessed":true,"operation_template_train_payload_accessed":true,"scaled_train_payload_accessed":true,"development_accessed":false,"systematic_rr_accessed":false,"test_er_accessed":false,"iid_accessed":false,"history_depth_accessed":false,"geometry_extrapolation_accessed":false,"other_corpus_accessed":false,"ge1_training_performed":false,"ge1_backward_pass_performed":false,"ge1_optimizer_constructed":false,"ge1_checkpoint_written":false,"model_or_decoder_repair_implemented_or_invoked":false,"cad_kernel_available":false,"cad_kernel_used":false,"stage6_performed":false,"c8_or_later_performed":false,"representation_probe_completed":true}\n' \
  "$EXPECTED_COMMIT" "$SLURM_JOB_ID" "$(hostname)"
