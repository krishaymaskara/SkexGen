#!/bin/bash
#SBATCH --job-name=ge1-stop-symbol-validation
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/network/km6349/ge1_stop_symbol_validation_runs/stop-symbol-cpu-%j.out
#SBATCH --error=/scratch/network/km6349/ge1_stop_symbol_validation_runs/stop-symbol-cpu-%j.err

set -Eeuo pipefail

terminal_failure() {
  status=$?
  trap - ERR
  printf '{"event":"autonomous_stop_validation_terminal_failure","commit":"%s","job_id":"%s","exit_status":%s,"operation_template_manifest_accessed":false,"operation_template_train_payload_accessed":false,"development_accessed":false,"systematic_rr_accessed":false,"test_er_accessed":false,"iid_accessed":false,"history_depth_accessed":false,"geometry_extrapolation_accessed":false,"other_corpus_accessed":false,"scientific_training_performed":false,"engineering_training_performed":false,"checkpoint_written":false,"cad_kernel_used":false,"stage6_performed":false,"c8_or_later_performed":false,"slurm_submitted_by_implementation_agent":true}\n' \
    "${EXPECTED_COMMIT:-unresolved}" "${SLURM_JOB_ID:-unresolved}" "$status" >&2
  exit "$status"
}
trap terminal_failure ERR

: "${EXPECTED_COMMIT:?submit with --export=ALL,EXPECTED_COMMIT=<exact-stop-commit>}"
: "${SLURM_JOB_ID:?this validation must run under Slurm}"
REPOSITORY="${REPOSITORY:-/scratch/network/km6349/SkexGen-autonomous-stop}"
LOG_PARENT="/scratch/network/km6349/ge1_stop_symbol_validation_runs"
CONTAINER="/scratch/network/km6349/skexgen.sif"
PYTHON="/root/miniconda3/bin/python3.8"
ADR_PARENT_COMMIT="25d9510d2b7b502b69b7387f1f0937509d3ef9ba"
RUNNER="prototype/graph_encoder/adroit/ge1_stop_symbol_validation_cpu.slurm"

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPYCACHEPREFIX="/tmp/ge1-autonomous-stop-${SLURM_JOB_ID}"
export TMPDIR="/tmp/ge1-autonomous-stop-tmp-${SLURM_JOB_ID}"
mkdir -p "$TMPDIR" "$LOG_PARENT"

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

test -d "$REPOSITORY/.git"
test ! -f "$REPOSITORY/.git"
test -f "$CONTAINER"
cd "$REPOSITORY"

CHECKED_OUT_COMMIT="$(git rev-parse HEAD)"
SOURCE_STATUS="$(git status --porcelain=v1 --untracked-files=all)"
test "$CHECKED_OUT_COMMIT" = "$EXPECTED_COMMIT"
test -z "$SOURCE_STATUS"
git cat-file -e "${ADR_PARENT_COMMIT}^{commit}"

printf '{"event":"autonomous_stop_validation_start","commit":"%s","job_id":"%s","host":"%s","container":"%s","source_clean":true,"standalone_checkout":true,"execution_device":"cpu","node_generation_identity":"GE1-PAD-TERMINATED-UNCONSTRAINED-NODES-v1","magnitude_identity":"GE1-OPERATION-MAGNITUDE-GRID-SOFTMAX-v1","operation_template_manifest_accessed":false,"operation_template_train_payload_accessed":false,"development_accessed":false,"systematic_rr_accessed":false,"test_er_accessed":false,"iid_accessed":false,"history_depth_accessed":false,"geometry_extrapolation_accessed":false,"other_corpus_accessed":false,"scientific_training_performed":false,"engineering_training_performed":false,"checkpoint_written":false,"cad_kernel_used":false,"stage6_performed":false,"c8_or_later_performed":false}\n' \
  "$CHECKED_OUT_COMMIT" "$SLURM_JOB_ID" "$(hostname)" "$CONTAINER"

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
    "event": "autonomous_stop_validation_environment",
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

expected = {
    "prototype.graph_encoder.tests.test_autonomous_stop_contract": 11,
    "prototype.graph_encoder.tests.test_autonomous_stop_runtime": 8,
}
total_declared = 0
total_run = 0
for module, expected_count in expected.items():
    suite = unittest.defaultTestLoader.loadTestsFromName(module)
    declared_count = suite.countTestCases()
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    print(json.dumps({
        "event": "autonomous_stop_focused_suite",
        "module": module,
        "declared_tests": declared_count,
        "expected_tests": expected_count,
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
    }, sort_keys=True, separators=(",", ":")), flush=True)
    if (
        declared_count != expected_count
        or result.testsRun != declared_count
        or not result.wasSuccessful()
        or result.skipped
    ):
        raise SystemExit("focused suite {} differs".format(module))
    total_declared += declared_count
    total_run += result.testsRun
print(json.dumps({
    "event": "autonomous_stop_focused_total",
    "declared_tests": total_declared,
    "tests_run": total_run,
    "expected_tests": 19,
}, sort_keys=True, separators=(",", ":")), flush=True)
if total_declared != 19:
    raise SystemExit("focused autonomous-stop suite total must be exactly 19")
PY

# The six Stage6CudaRuntimeTests methods skip on any CPU host.  The allowlist
# validator accepts exactly those and rejects every missing, duplicate, or
# additional skip; a literal zero-skip gate would fail here for that reason.
container_python -B - <<'PY'
import json
import unittest

from prototype.graph_encoder.grid_magnitude_audit import (
    validate_cpu_softmax_complete_discovery,
)

suite = unittest.defaultTestLoader.discover(
    "prototype/graph_encoder/tests", top_level_dir="."
)
declared_count = suite.countTestCases()
result = unittest.TextTestRunner(verbosity=1).run(suite)
telemetry = validate_cpu_softmax_complete_discovery(declared_count, result)
print(json.dumps(telemetry, sort_keys=True, separators=(",", ":")), flush=True)
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
for name, path, expected_count, expected_skips in suites:
    suite = unittest.defaultTestLoader.discover(path, top_level_dir=".")
    declared_count = suite.countTestCases()
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    print(json.dumps({
        "event": "autonomous_stop_regression_suite",
        "suite": name,
        "declared_tests": declared_count,
        "expected_tests": expected_count,
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
        "expected_skips": expected_skips,
    }, sort_keys=True, separators=(",", ":")), flush=True)
    if (
        declared_count != expected_count
        or result.testsRun != declared_count
        or not result.wasSuccessful()
        or len(result.skipped) != expected_skips
    ):
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
print("AUTONOMOUS_STOP_PYTHON38_GRAMMAR_OK {}".format(len(paths)), flush=True)
PY

container_python -B - <<'PY'
import json

from prototype.graph_encoder.decoder_contract import (
    AUTONOMOUS_STOP_NODE_GENERATION_IDENTITY,
    GRID_SOFTMAX_OPERATION_MAGNITUDE_PARAMETERIZATION,
    LEGACY_NODE_GENERATION_IDENTITY,
    checkpoint_schema_for,
    shared_decoder_version_for,
    output_position_contract_version_for,
)

softmax = GRID_SOFTMAX_OPERATION_MAGNITUDE_PARAMETERIZATION
stop = AUTONOMOUS_STOP_NODE_GENERATION_IDENTITY
legacy = LEGACY_NODE_GENERATION_IDENTITY

# Learned stopping changes the decoder, checkpoint, and output-position
# contracts independently of the retained grid-softmax magnitude identity.
assert checkpoint_schema_for(softmax, stop) != checkpoint_schema_for(softmax, legacy)
assert shared_decoder_version_for(softmax, stop) != shared_decoder_version_for(softmax, legacy)
assert output_position_contract_version_for(softmax, stop) != (
    output_position_contract_version_for(softmax, legacy)
)
print(json.dumps({
    "event": "autonomous_stop_identity_audit",
    "node_generation_identity": stop,
    "magnitude_identity": softmax,
    "checkpoint_schema": checkpoint_schema_for(softmax, stop),
    "decoder_version": shared_decoder_version_for(softmax, stop),
    "output_position_version": output_position_contract_version_for(softmax, stop),
}, sort_keys=True, separators=(",", ":")), flush=True)
PY

container_python -B - <<'PY'
import json
from pathlib import Path
import re

root = Path("prototype")
offenders = sorted(
    str(path) for path in root.rglob("*.py")
    if "__pycache__" not in path.parts
    and "/tests/" not in path.as_posix()
    and re.search(r"\bORDINAL_CUT_COUNT\b", path.read_text(encoding="utf-8"))
)
expected = [
    "prototype/graph_encoder/grid_magnitude.py",
    "prototype/graph_encoder/losses.py",
]
print(json.dumps({
    "event": "autonomous_stop_cut_count_confinement",
    "modules": offenders,
    "expected": expected,
}, sort_keys=True, separators=(",", ":")), flush=True)
if offenders != expected:
    raise SystemExit("ORDINAL_CUT_COUNT escaped its two production modules")
PY

container_python -B -m prototype.graph_encoder.grid_magnitude_audit "$RUNNER"

# ADR-0013's ordinal path stays runnable and immutable across this change.
git diff --quiet "$ADR_PARENT_COMMIT" "$EXPECTED_COMMIT" -- \
  prototype/graph_encoder/adroit/ge1_grid_magnitude_validation_cpu.slurm
git diff --quiet "$ADR_PARENT_COMMIT" "$EXPECTED_COMMIT" -- \
  prototype/graph_encoder/adroit/ge1_grid_ordinal_horizon_cpu.slurm
git diff --quiet "$ADR_PARENT_COMMIT" "$EXPECTED_COMMIT" -- \
  prototype/graph_encoder/adroit/ge1_grid_ordinal_trajectory_cpu.slurm
git diff --quiet "$ADR_PARENT_COMMIT" "$EXPECTED_COMMIT" -- \
  prototype/flat_baseline
bash -n "$RUNNER"
git diff --check
test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git status --porcelain=v1 --untracked-files=all)"

printf '{"event":"autonomous_stop_validation_terminal_success","commit":"%s","job_id":"%s","host":"%s","python":"3.8.13","pytorch":"1.11.0","execution_device":"cpu","node_generation_identity":"GE1-PAD-TERMINATED-UNCONSTRAINED-NODES-v1","magnitude_identity":"GE1-OPERATION-MAGNITUDE-GRID-SOFTMAX-v1","focused_tests":19,"focused_skips":0,"complete_graph_encoder_cuda_skips":6,"flat_baseline_expected_skips":4,"operation_template_manifest_accessed":false,"operation_template_train_payload_accessed":false,"development_accessed":false,"systematic_rr_accessed":false,"test_er_accessed":false,"iid_accessed":false,"history_depth_accessed":false,"geometry_extrapolation_accessed":false,"other_corpus_accessed":false,"scientific_training_performed":false,"engineering_training_performed":false,"checkpoint_written":false,"cad_kernel_used":false,"stage6_performed":false,"c8_or_later_performed":false,"slurm_submitted_by_implementation_agent":true}\n' \
  "$EXPECTED_COMMIT" "$SLURM_JOB_ID" "$(hostname)"
