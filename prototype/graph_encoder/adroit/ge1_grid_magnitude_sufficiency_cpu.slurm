#!/bin/bash
#SBATCH --job-name=ge1-grid-magnitude-sufficiency
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=24:00:00
#SBATCH --signal=B:TERM@60
#SBATCH --output=/scratch/network/km6349/ge1_grid_magnitude_sufficiency_runs/grid-magnitude-cpu-%j.out
#SBATCH --error=/scratch/network/km6349/ge1_grid_magnitude_sufficiency_runs/grid-magnitude-cpu-%j.err

set -Eeuo pipefail

MANIFEST_ACCESSED=false
TRAIN_PAYLOAD_ACCESSED=false
TINY_PAYLOAD_ACCESSED=false
SCALED_PAYLOAD_ACCESSED=false

terminal_failure() {
  status=$?
  trap - ERR
  printf '{"event":"grid_magnitude_sufficiency_terminal_failure","commit":"%s","job_id":"%s","exit_status":%s,"operation_template_manifest_accessed":%s,"operation_template_train_payload_accessed":%s,"tiny_train_payload_accessed":%s,"scaled_train_payload_accessed":%s,"development_accessed":false,"systematic_rr_accessed":false,"test_er_accessed":false,"iid_accessed":false,"history_depth_accessed":false,"geometry_extrapolation_accessed":false,"other_corpus_accessed":false,"infrastructure_completed":false,"grid_magnitude_sufficiency_execution_completed":false,"another_scientific_job_authorized":false,"additional_repair_authorized":false,"stage6_performed":false,"c8_or_later_performed":false,"cad_kernel_available":false,"cad_kernel_used":false,"additional_repair_implemented_or_invoked":false}\n' \
    "${EXPECTED_COMMIT:-unresolved}" "${SLURM_JOB_ID:-unresolved}" "$status" \
    "$MANIFEST_ACCESSED" "$TRAIN_PAYLOAD_ACCESSED" \
    "$TINY_PAYLOAD_ACCESSED" "$SCALED_PAYLOAD_ACCESSED" >&2
  exit "$status"
}

terminal_timeout() {
  trap - ERR TERM
  printf '{"event":"grid_magnitude_sufficiency_terminal_timeout","commit":"%s","job_id":"%s","exit_status":124,"operation_template_manifest_accessed":%s,"operation_template_train_payload_accessed":%s,"tiny_train_payload_accessed":%s,"scaled_train_payload_accessed":%s,"development_accessed":false,"systematic_rr_accessed":false,"test_er_accessed":false,"iid_accessed":false,"history_depth_accessed":false,"geometry_extrapolation_accessed":false,"other_corpus_accessed":false,"infrastructure_completed":false,"grid_magnitude_sufficiency_execution_completed":false,"another_scientific_job_authorized":false,"additional_repair_authorized":false,"stage6_performed":false,"c8_or_later_performed":false,"cad_kernel_available":false,"cad_kernel_used":false}\n' \
    "${EXPECTED_COMMIT:-unresolved}" "${SLURM_JOB_ID:-unresolved}" \
    "$MANIFEST_ACCESSED" "$TRAIN_PAYLOAD_ACCESSED" \
    "$TINY_PAYLOAD_ACCESSED" "$SCALED_PAYLOAD_ACCESSED" >&2
  exit 124
}
trap terminal_failure ERR
trap terminal_timeout TERM

: "${EXPECTED_COMMIT:?scientific execution requires an explicitly authorized exact commit}"
: "${SLURM_JOB_ID:?this workflow must run under Slurm}"
REPOSITORY="${REPOSITORY:-/scratch/network/km6349/SkexGen-grid-magnitude}"
CORPUS_DIR="${CORPUS_DIR:-/scratch/network/km6349/controlled_corpora/b0-pilot-680-seed2026}"
RUN_PARENT="${RUN_PARENT:-/scratch/network/km6349/ge1_grid_magnitude_sufficiency_artifacts}"
OUTPUT_DIR="${RUN_PARENT}/ge1-grid-magnitude-${EXPECTED_COMMIT}-${SLURM_JOB_ID}"
LOG_PARENT="/scratch/network/km6349/ge1_grid_magnitude_sufficiency_runs"
CONTAINER="/scratch/network/km6349/skexgen.sif"
PYTHON="/root/miniconda3/bin/python3.8"
AUTHORIZATION_PARENT_COMMIT="bdb7148dc4ebe751bda1a66cd167fcf035495762"

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPYCACHEPREFIX="/tmp/ge1-grid-magnitude-${SLURM_JOB_ID}"
export TMPDIR="/tmp/ge1-grid-magnitude-tmp-${SLURM_JOB_ID}"
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
    --bind "$CORPUS_DIR:$CORPUS_DIR:ro" \
    --bind "$RUN_PARENT:$RUN_PARENT:rw" \
    "$CONTAINER" "$PYTHON" "$@"
}

test -d "$REPOSITORY/.git"
test ! -f "$REPOSITORY/.git"
test -d "$CORPUS_DIR"
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

printf '{"event":"grid_magnitude_sufficiency_start","commit":"%s","job_id":"%s","host":"%s","source_clean":true,"standalone_checkout":true,"detached_head":true,"exact_commit":true,"execution_device":"cpu","protocol_version":"GE1-C7-GRID-MAGNITUDE-SUFFICIENCY-v1","parameterization":"GE1-OPERATION-MAGNITUDE-GRID-ORDINAL-v1","checkpoint_schema":"GE1-CHECKPOINT-v2","operation_template_manifest_accessed":false,"operation_template_train_payload_accessed":false,"tiny_train_payload_accessed":false,"scaled_train_payload_accessed":false,"development_accessed":false,"systematic_rr_accessed":false,"test_er_accessed":false,"iid_accessed":false,"history_depth_accessed":false,"geometry_extrapolation_accessed":false,"other_corpus_accessed":false,"another_scientific_job_authorized":false,"additional_repair_authorized":false,"stage6_performed":false,"c8_or_later_performed":false,"cad_kernel_available":false,"cad_kernel_used":false,"additional_repair_implemented_or_invoked":false}\n' \
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
    "event": "grid_magnitude_sufficiency_environment",
    "python_version": sys.version.split()[0],
    "python_implementation": platform.python_implementation(),
    "pytorch_version": torch.__version__,
    "pytorch_cuda_build": torch.version.cuda,
    "cuda_available": False,
    "execution_device": "cpu",
    "cpu_threads": torch.get_num_threads(),
}, sort_keys=True, separators=(",", ":")), flush=True)
PY

# Every preflight is rerun from the exact separately authorized scientific
# commit before the one train-only execution begins.
container_python -B - <<'PY'
import json
import unittest

suites = (
    (
        "grid_pure_contract",
        "prototype.graph_encoder.tests.test_grid_magnitude_contract",
        28,
    ),
    (
        "grid_real_pytorch_runtime",
        "prototype.graph_encoder.tests.test_grid_magnitude_runtime",
        14,
    ),
    (
        "grid_synthetic_integration",
        "prototype.graph_encoder.tests.test_grid_magnitude_integration",
        None,
    ),
    (
        "scientific_protocol_contract",
        "prototype.graph_encoder.tests.test_grid_magnitude_sufficiency_contract",
        31,
    ),
)
for name, module, expected in suites:
    suite = unittest.defaultTestLoader.loadTestsFromName(module)
    declared = suite.countTestCases()
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    print(json.dumps({
        "event": "grid_magnitude_sufficiency_focused_suite",
        "suite": name,
        "tests_run": result.testsRun,
        "declared_tests": declared,
        "expected_tests": expected,
        "skipped": len(result.skipped),
    }, sort_keys=True, separators=(",", ":")), flush=True)
    if (
        (expected is not None and declared != expected)
        or result.testsRun != declared
        or not result.wasSuccessful()
        or result.skipped
    ):
        raise SystemExit("focused grid-magnitude protocol suite differs")
PY
container_python -B - <<'PY'
import unittest

result = unittest.TextTestRunner(verbosity=1).run(
    unittest.TestLoader().discover(
        "prototype/graph_encoder/tests", top_level_dir="."
    )
)
if not result.wasSuccessful() or result.skipped:
    raise SystemExit("complete graph_encoder suite differs")
PY
container_python -B - <<'PY'
import unittest

flat_expected_methods = {
    "test_five_capsule_regressions_are_a_pinned_golden_cohort",
    "test_malformed_parent_fails_before_scientific_loading",
    "test_parent_equivalent_arms_reconcile_pinned_records",
    "test_real_bundle_two_family_golden_reconciliation",
}
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
    observed_skip_methods = {
        item._testMethodName for item, unused_reason in result.skipped
    }
    expected_methods = flat_expected_methods if name == "flat_baseline" else set()
    if (
        not result.wasSuccessful()
        or len(result.skipped) != expected_skips
        or observed_skip_methods != expected_methods
    ):
        raise SystemExit("regression suite differs: {}".format(path))
PY
container_python -B tools/check_documentation.py
container_python -B -m compileall -q prototype/graph_encoder
container_python -B - <<'PY'
import ast
import hashlib
from pathlib import Path

root = Path("prototype/graph_encoder")
paths = tuple(sorted(root.glob("*.py"))) + tuple(
    sorted((root / "tests").glob("test_*.py"))
)
for path in paths:
    ast.parse(
        path.read_text(encoding="utf-8"), filename=str(path),
        feature_version=(3, 8),
    )
source = (root / "grid_magnitude_sufficiency.py").read_text(encoding="utf-8")
tree = ast.parse(source)
imports = {
    alias.name for node in ast.walk(tree)
    if isinstance(node, ast.ImportFrom) for alias in node.names
}
assert "load_development" not in imports
assert "load_partition_physical_examples" not in imports
assert "secondary_systematic_validation" not in source
assert "build_legacy_matched_ge1_models" not in source
frozen = {
    "c7_v2.py": "67f83efe3ef58478ad3c00b846d43f8928b7c642e42ee8fcbc19cce0f63e01aa",
    "tests/test_c7_v2_contract.py": "c8e607095b9d2444f9bb1438f6784f089bcce6abdafdcd307fb4643050d96330",
    "tests/test_c7_v2_runtime.py": "78721bdbaac7c2a04223e079151de610a90718dcfcbb1525a4cc1b66c800380c",
}
for relative, expected in frozen.items():
    assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == expected
print("GRID_MAGNITUDE_SCIENTIFIC_SOURCE_AUDIT=PASS", flush=True)
PY
container_python -B -m prototype.graph_encoder.grid_magnitude_audit \
  prototype/graph_encoder/adroit/ge1_grid_magnitude_validation_cpu.slurm
container_python -B -m prototype.graph_encoder.grid_magnitude_sufficiency_audit \
  prototype/graph_encoder/adroit/ge1_grid_magnitude_sufficiency_cpu.slurm
git diff --quiet "$AUTHORIZATION_PARENT_COMMIT" "$EXPECTED_COMMIT" -- \
  prototype/graph_encoder/operation_fidelity.py
git diff --quiet "$AUTHORIZATION_PARENT_COMMIT" "$EXPECTED_COMMIT" -- \
  prototype/graph_baseline
git diff --quiet "$AUTHORIZATION_PARENT_COMMIT" "$EXPECTED_COMMIT" -- \
  prototype/flat_baseline
bash -n prototype/graph_encoder/adroit/ge1_grid_magnitude_sufficiency_cpu.slurm
bash -n prototype/graph_encoder/adroit/ge1_grid_magnitude_validation_cpu.slurm
git show --check "$EXPECTED_COMMIT" --
test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git symbolic-ref -q --short HEAD || true)"
test -z "$(git status --porcelain=v1 --untracked-files=all)"

MANIFEST_ACCESSED='"not_confirmed_on_failure"'
TRAIN_PAYLOAD_ACCESSED='"not_confirmed_on_failure"'
TINY_PAYLOAD_ACCESSED='"not_confirmed_on_failure"'
SCALED_PAYLOAD_ACCESSED='"not_confirmed_on_failure"'
container_python -B -m prototype.graph_encoder.grid_magnitude_sufficiency \
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

from prototype.graph_encoder.decoder_contract import (
    GRID_ORDINAL_OPERATION_MAGNITUDE_PARAMETERIZATION,
)
from prototype.graph_encoder.grid_magnitude_sufficiency import (
    verify_grid_magnitude_artifact,
)

root = Path(sys.argv[1])
commit = sys.argv[2]
job_id = sys.argv[3]
verified = verify_grid_magnitude_artifact(
    root, expected_commit=commit, expected_slurm_job_id=job_id
)
resolved = json.loads((root / "resolved_config.json").read_text(encoding="utf-8"))
events = [
    json.loads(line)
    for line in (root / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    if line
]
terminal = events[-1]
assert resolved["protocol_version"] == "GE1-C7-GRID-MAGNITUDE-SUFFICIENCY-v1"
assert resolved["operation_magnitude_parameterization"] == (
    GRID_ORDINAL_OPERATION_MAGNITUDE_PARAMETERIZATION
)
assert resolved["training_arithmetic"]["tiny"]["optimizer_steps"] == 200
assert resolved["training_arithmetic"]["tiny"]["training_example_presentations"] == 800
assert resolved["training_arithmetic"]["scaled"]["optimizer_steps"] == 800
assert resolved["training_arithmetic"]["scaled"]["training_example_presentations"] == 6400
assert terminal["event"] == "grid_magnitude_sufficiency_execution_completed"
assert terminal["infrastructure_completed"] is True
assert terminal["cad_kernel_available"] is False
assert terminal["stage6_authorized_by_grid_magnitude_sufficiency"] is False
assert terminal["another_scientific_job_authorized"] is False
assert terminal["additional_repair_authorized"] is False
for name in (
    "development_accessed", "systematic_rr_accessed", "test_er_accessed",
    "iid_accessed", "history_depth_accessed", "geometry_extrapolation_accessed",
    "other_corpus_accessed", "stage6_performed", "c8_or_later_performed",
):
    assert terminal[name] is False
for path in sorted(root.glob("checkpoints/*/*/*.pt")):
    payload = torch.load(str(path), map_location="cpu")
    assert payload["checkpoint_schema"] == "GE1-CHECKPOINT-v2"
    if "training_state_schema" in payload:
        assert payload["completed_epoch"] == 200
        assert payload["selected_checkpoint_epoch"] == 200
        assert payload["selected_experimental_checkpoint"] is True
        assert payload["model_config"]["operation_magnitude_parameterization"] == (
            GRID_ORDINAL_OPERATION_MAGNITUDE_PARAMETERIZATION
        )
        assert payload["model_config"]["checkpoint_schema"] == (
            "GE1-CHECKPOINT-v2"
        )
        assert payload["provenance"]["operation_magnitude_parameterization"] == (
            GRID_ORDINAL_OPERATION_MAGNITUDE_PARAMETERIZATION
        )
    else:
        assert payload["operation_magnitude_parameterization"] == (
            GRID_ORDINAL_OPERATION_MAGNITUDE_PARAMETERIZATION
        )
print(json.dumps({
    "event": "grid_magnitude_sufficiency_artifact_verified",
    "commit": commit,
    "job_id": job_id,
    "checkpoint_count": verified["checkpoint_count"],
    "grid_magnitude_scientific_gate_pass": terminal["grid_magnitude_scientific_gate_pass"],
    "grid_magnitude_scientific_failure": terminal["grid_magnitude_scientific_failure"],
    "comparison_inconclusive": terminal["comparison_inconclusive"],
    "stage6_authorized_by_grid_magnitude_sufficiency": terminal[
        "stage6_authorized_by_grid_magnitude_sufficiency"
    ],
    "scaled_train_payload_accessed": terminal["scaled_train_payload_accessed"],
    "cad_kernel_available": False,
}, sort_keys=True, separators=(",", ":")), flush=True)
PY

MANIFEST_ACCESSED=true
TRAIN_PAYLOAD_ACCESSED=true
TINY_PAYLOAD_ACCESSED=true
SCALED_PAYLOAD_ACCESSED="$(container_python -B - "$OUTPUT_DIR" <<'PY'
import json
from pathlib import Path
import sys
events = [json.loads(line) for line in (Path(sys.argv[1]) / "metrics.jsonl").read_text().splitlines() if line]
print("true" if events[-1]["scaled_train_payload_accessed"] else "false")
PY
)"

test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git status --porcelain=v1 --untracked-files=all)"
printf '{"event":"grid_magnitude_sufficiency_terminal_success","commit":"%s","job_id":"%s","host":"%s","protocol_version":"GE1-C7-GRID-MAGNITUDE-SUFFICIENCY-v1","infrastructure_completed":true,"scientific_decision_finalized":true,"operation_template_manifest_accessed":true,"operation_template_train_payload_accessed":true,"tiny_train_payload_accessed":true,"scaled_train_payload_accessed":%s,"development_accessed":false,"systematic_rr_accessed":false,"test_er_accessed":false,"iid_accessed":false,"history_depth_accessed":false,"geometry_extrapolation_accessed":false,"other_corpus_accessed":false,"cad_kernel_available":false,"cad_kernel_used":false,"another_scientific_job_authorized":false,"additional_repair_authorized":false,"stage6_performed":false,"stage6_authorized":false,"c8_or_later_performed":false,"c8_authorized":false,"additional_repair_implemented_or_invoked":false}\n' \
  "$EXPECTED_COMMIT" "$SLURM_JOB_ID" "$(hostname)" "$SCALED_PAYLOAD_ACCESSED"
