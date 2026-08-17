#!/bin/bash
#SBATCH --job-name=ge1-grid-magnitude-validation
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/network/km6349/ge1_grid_magnitude_runs/grid-validation-%j.out
#SBATCH --error=/scratch/network/km6349/ge1_grid_magnitude_runs/grid-validation-%j.err

# Corpus-free ADR-0013 engineering validation only.  The log parent must be
# created before submission because Slurm opens stdout/stderr before this
# script starts.  This script never submits itself or runs scientific work.

set -Eeuo pipefail

terminal_failure() {
  status=$?
  trap - ERR
  printf '{"event":"ge1_grid_magnitude_validation_terminal_failure","commit":"%s","job_id":"%s","exit_status":%s,"corpus_accessed":false,"manifest_accessed":false,"preserved_payload_accessed":false,"external_checkpoint_accessed":false,"scientific_checkpoint_accessed":false,"synthetic_temporary_checkpoint_may_have_been_created":true,"model_artifact_accessed":false,"repaired_artifact_accessed":false,"protected_partition_accessed":false,"scientific_training_performed":false,"scientific_inference_performed":false,"cad_kernel_used":false,"stage6_performed":false,"c8_or_later_performed":false,"scientific_artifact_produced":false}\n' \
    "${EXPECTED_COMMIT:-unresolved}" "${SLURM_JOB_ID:-unresolved}" "$status" >&2
  exit "$status"
}
trap terminal_failure ERR

: "${EXPECTED_COMMIT:?supply one exact implementation commit at submission}"
: "${SLURM_JOB_ID:?this engineering validation must run under Slurm}"
REPOSITORY="/scratch/network/km6349/SkexGen-grid-magnitude"
LOG_PARENT="/scratch/network/km6349/ge1_grid_magnitude_runs"
CONTAINER="/scratch/network/km6349/skexgen.sif"
PYTHON="/root/miniconda3/bin/python3.8"

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPYCACHEPREFIX="/tmp/ge1-grid-magnitude-pycache-${SLURM_JOB_ID}"
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
    "$CONTAINER" "$PYTHON" "$@"
}

test "$EXPECTED_COMMIT" = "$(printf '%s' "$EXPECTED_COMMIT" | tr -cd '0-9a-f')"
test "${#EXPECTED_COMMIT}" -eq 40
test -d "$LOG_PARENT"
test -d "$REPOSITORY/.git"
test ! -f "$REPOSITORY/.git"
test -f "$CONTAINER"
cd "$REPOSITORY"
git cat-file -e "${EXPECTED_COMMIT}^{commit}"
test "$(git rev-parse --show-toplevel)" = "$REPOSITORY"
test -z "$(git rev-parse --show-superproject-working-tree)"

CHECKED_OUT_COMMIT="$(git rev-parse HEAD)"
SOURCE_STATUS="$(git status --porcelain=v1 --untracked-files=all)"
BRANCH_NAME="$(git symbolic-ref -q --short HEAD || true)"
test "$CHECKED_OUT_COMMIT" = "$EXPECTED_COMMIT"
test -z "$BRANCH_NAME"
test -z "$SOURCE_STATUS"

printf '{"event":"ge1_grid_magnitude_validation_start","commit":"%s","job_id":"%s","host":"%s","container":"%s","checkout":"%s","source_clean":true,"standalone_checkout":true,"detached_head":true,"exact_commit":true,"execution_device":"cpu","cpu_task_count":1,"cpu_thread_count":1,"memory_gb":12,"engineering_validation_only":true,"scientific_execution_authorized":false}\n' \
  "$CHECKED_OUT_COMMIT" "$SLURM_JOB_ID" "$(hostname)" "$CONTAINER" "$REPOSITORY"

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
    "event": "ge1_grid_magnitude_validation_environment",
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

suites = (
    (
        "pure_contract",
        "prototype.graph_encoder.tests.test_grid_magnitude_contract",
        28,
    ),
    (
        "real_pytorch_runtime",
        "prototype.graph_encoder.tests.test_grid_magnitude_runtime",
        14,
    ),
    (
        "synthetic_integration",
        "prototype.graph_encoder.tests.test_grid_magnitude_integration",
        None,
    ),
)
for name, module, expected_count in suites:
    suite = unittest.defaultTestLoader.loadTestsFromName(module)
    declared_count = suite.countTestCases()
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    print(json.dumps({
        "event": "ge1_grid_magnitude_focused_suite",
        "suite": name,
        "declared_tests": declared_count,
        "tests_run": result.testsRun,
        "expected_tests": expected_count,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
    }, sort_keys=True, separators=(",", ":")), flush=True)
    if (
        (expected_count is not None and declared_count != expected_count)
        or result.testsRun != declared_count
        or not result.wasSuccessful()
        or result.skipped
    ):
        raise SystemExit("{} grid suite differs".format(name))
PY

container_python -B - <<'PY'
import json
import unittest

suite = unittest.defaultTestLoader.discover(
    "prototype/graph_encoder/tests", top_level_dir="."
)
declared_count = suite.countTestCases()
result = unittest.TextTestRunner(verbosity=1).run(suite)
print(json.dumps({
    "event": "ge1_grid_magnitude_complete_graph_encoder_suite",
    "declared_tests": declared_count,
    "tests_run": result.testsRun,
    "failures": len(result.failures),
    "errors": len(result.errors),
    "skipped": len(result.skipped),
}, sort_keys=True, separators=(",", ":")), flush=True)
if result.testsRun != declared_count or not result.wasSuccessful() or result.skipped:
    raise SystemExit("complete graph_encoder suite must pass with zero skips")
PY

container_python -B - <<'PY'
import json
import unittest

suites = (
    ("model_data", "prototype/model_data/tests", 86, 0),
    ("flat_baseline", "prototype/flat_baseline/tests", 549, 4),
    ("graph_baseline", "prototype/graph_baseline/tests", 17, 0),
    ("representation", "prototype/representation/tests", 40, 0),
    ("controlled_data", "prototype/controlled_data/tests", 51, 0),
)
flat_expected_methods = {
    "test_five_capsule_regressions_are_a_pinned_golden_cohort",
    "test_malformed_parent_fails_before_scientific_loading",
    "test_parent_equivalent_arms_reconcile_pinned_records",
    "test_real_bundle_two_family_golden_reconciliation",
}
for name, path, expected_count, expected_skips in suites:
    suite = unittest.defaultTestLoader.discover(path, top_level_dir=".")
    declared_count = suite.countTestCases()
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    skip_methods = {
        item._testMethodName for item, unused_reason in result.skipped
    }
    expected_methods = flat_expected_methods if name == "flat_baseline" else set()
    print(json.dumps({
        "event": "ge1_grid_magnitude_regression_suite",
        "suite": name,
        "declared_tests": declared_count,
        "tests_run": result.testsRun,
        "expected_tests": expected_count,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
        "expected_skips": expected_skips,
        "skip_methods": sorted(skip_methods),
    }, sort_keys=True, separators=(",", ":")), flush=True)
    if (
        declared_count != expected_count
        or result.testsRun != expected_count
        or not result.wasSuccessful()
        or len(result.skipped) != expected_skips
        or skip_methods != expected_methods
    ):
        raise SystemExit("{} regression suite differs".format(name))
PY

printf '{"event":"ge1_grid_magnitude_validation_step","step":"documentation_validation","state":"start"}\n'
container_python -B tools/check_documentation.py
printf '{"event":"ge1_grid_magnitude_validation_step","step":"documentation_validation","state":"pass"}\n'

printf '{"event":"ge1_grid_magnitude_validation_step","step":"compileall","state":"start"}\n'
container_python -B -m compileall -q prototype/graph_encoder
printf '{"event":"ge1_grid_magnitude_validation_step","step":"compileall","state":"pass"}\n'

container_python -B - <<'PY'
import ast
import json
from pathlib import Path

paths = tuple(sorted(Path("prototype/graph_encoder").glob("*.py"))) + tuple(
    sorted(Path("prototype/graph_encoder/tests").glob("test_*.py"))
)
for path in paths:
    ast.parse(
        path.read_text(encoding="utf-8"),
        filename=str(path),
        feature_version=(3, 8),
    )
print(json.dumps({
    "event": "ge1_grid_magnitude_python38_ast",
    "state": "pass",
    "file_count": len(paths),
}, sort_keys=True, separators=(",", ":")), flush=True)
PY

container_python -B -m prototype.graph_encoder.grid_magnitude_audit \
  prototype/graph_encoder/adroit/ge1_grid_magnitude_validation_cpu.slurm

container_python -B - <<'PY'
from prototype.graph_encoder.decoder_contract import (
    LEGACY_CHECKPOINT_SCHEMA,
    LEGACY_OPERATION_MAGNITUDE_PARAMETERIZATION,
    OUTPUT_POSITION_CONTRACT_VERSION,
    POSITIVE_OPERATION_MAGNITUDE_PARAMETERIZATION,
    SHARED_DECODER_VERSION,
)
from prototype.graph_encoder.operation_fidelity import (
    EXTRUSION_ABSOLUTE_ERROR_MAX_EXCLUSIVE,
    REVOLVE_ABSOLUTE_ERROR_MAX_EXCLUSIVE,
    operation_fidelity_contract,
)

assert EXTRUSION_ABSOLUTE_ERROR_MAX_EXCLUSIVE == 0.25
assert REVOLVE_ABSOLUTE_ERROR_MAX_EXCLUSIVE == 22.5
assert operation_fidelity_contract()["comparison"] == (
    "strictly_less_than_unrounded"
)
assert LEGACY_OPERATION_MAGNITUDE_PARAMETERIZATION == (
    "GE1-OPERATION-MAGNITUDE-TANH-LEGACY-v1"
)
assert POSITIVE_OPERATION_MAGNITUDE_PARAMETERIZATION == (
    "GE1-OPERATION-MAGNITUDE-POSITIVE-v1"
)
assert LEGACY_CHECKPOINT_SCHEMA == "GE1-CHECKPOINT-v1"
assert SHARED_DECODER_VERSION == "GE1-SHARED-TYPED-EDGE-DECODER-V1"
assert OUTPUT_POSITION_CONTRACT_VERSION == "GE1-DECODER-OUTPUT-POSITIONS-v1"
print("GE1_GRID_MAGNITUDE_FROZEN_IDENTITY_AUDIT=PASS", flush=True)
PY

git diff --quiet "${EXPECTED_COMMIT}^" "$EXPECTED_COMMIT" -- \
  prototype/graph_encoder/operation_fidelity.py
git diff --quiet "${EXPECTED_COMMIT}^" "$EXPECTED_COMMIT" -- \
  prototype/graph_baseline
git diff --quiet "${EXPECTED_COMMIT}^" "$EXPECTED_COMMIT" -- \
  prototype/flat_baseline
git show --check "$EXPECTED_COMMIT" --
bash -n prototype/graph_encoder/adroit/ge1_grid_magnitude_validation_cpu.slurm

test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git symbolic-ref -q --short HEAD || true)"
test -z "$(git status --porcelain=v1 --untracked-files=all)"

printf '{"event":"ge1_grid_magnitude_validation_terminal_success","commit":"%s","job_id":"%s","host":"%s","protocol_version":"GE1-C7-GRID-MAGNITUDE-SUFFICIENCY-v1","parameterization":"GE1-OPERATION-MAGNITUDE-GRID-ORDINAL-v1","checkpoint_schema":"GE1-CHECKPOINT-v2","engineering_validation_completed":true,"scientific_result_produced":false,"corpus_accessed":false,"manifest_accessed":false,"preserved_payload_accessed":false,"external_checkpoint_accessed":false,"scientific_checkpoint_accessed":false,"synthetic_temporary_checkpoint_roundtrip_executed":true,"synthetic_temporary_checkpoint_retained":false,"model_artifact_accessed":false,"repaired_artifact_accessed":false,"protected_partition_accessed":false,"scientific_training_performed":false,"scientific_inference_performed":false,"synthetic_engineering_optimizer_step_executed":true,"cad_kernel_used":false,"stage6_performed":false,"stage6_authorized":false,"c8_or_later_performed":false,"c8_authorized":false,"scientific_execution_authorized":false,"scientific_artifact_produced":false}\n' \
  "$(git rev-parse HEAD)" "$SLURM_JOB_ID" "$(hostname)"
