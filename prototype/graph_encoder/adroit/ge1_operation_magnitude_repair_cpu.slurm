#!/bin/bash
#SBATCH --job-name=ge1-op-magnitude-validation
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/network/km6349/ge1_operation_magnitude_validation_runs/operation-magnitude-cpu-%j.out
#SBATCH --error=/scratch/network/km6349/ge1_operation_magnitude_validation_runs/operation-magnitude-cpu-%j.err

set -Eeuo pipefail

terminal_failure() {
  status=$?
  trap - ERR
  printf '{"event":"operation_magnitude_repair_validation_terminal_failure","commit":"%s","job_id":"%s","exit_status":%s,"operation_template_manifest_accessed":false,"operation_template_train_payload_accessed":false,"development_accessed":false,"systematic_rr_accessed":false,"test_er_accessed":false,"iid_accessed":false,"history_depth_accessed":false,"geometry_extrapolation_accessed":false,"other_corpus_accessed":false,"scientific_training_performed":false,"engineering_training_performed":false,"checkpoint_written":false,"cad_kernel_used":false,"stage6_performed":false,"c8_or_later_performed":false,"slurm_submitted_by_implementation_agent":false}\n' \
    "${EXPECTED_COMMIT:-unresolved}" "${SLURM_JOB_ID:-unresolved}" "$status" >&2
  exit "$status"
}
trap terminal_failure ERR

: "${EXPECTED_COMMIT:?submit with --export=ALL,EXPECTED_COMMIT=<exact-repair-commit>}"
: "${SLURM_JOB_ID:?this validation must run under Slurm}"
REPOSITORY="${REPOSITORY:-/scratch/network/km6349/SkexGen-operation-magnitude-repair}"
LOG_PARENT="/scratch/network/km6349/ge1_operation_magnitude_validation_runs"
CONTAINER="/scratch/network/km6349/skexgen.sif"
PYTHON="/root/miniconda3/bin/python3.8"
ADR_PARENT_COMMIT="110ae7ab4b3308cd5c76cda26d8d62a18eec1cee"

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPYCACHEPREFIX="/tmp/ge1-operation-magnitude-${SLURM_JOB_ID}"
export TMPDIR="/tmp/ge1-operation-magnitude-tmp-${SLURM_JOB_ID}"
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

printf '{"event":"operation_magnitude_repair_validation_start","commit":"%s","job_id":"%s","host":"%s","container":"%s","source_clean":true,"standalone_checkout":true,"execution_device":"cpu","repair_identity":"GE1-OPERATION-MAGNITUDE-POSITIVE-v1","legacy_identity":"GE1-OPERATION-MAGNITUDE-TANH-LEGACY-v1","operation_template_manifest_accessed":false,"operation_template_train_payload_accessed":false,"development_accessed":false,"systematic_rr_accessed":false,"test_er_accessed":false,"iid_accessed":false,"history_depth_accessed":false,"geometry_extrapolation_accessed":false,"other_corpus_accessed":false,"scientific_training_performed":false,"engineering_training_performed":false,"checkpoint_written":false,"cad_kernel_used":false,"stage6_performed":false,"c8_or_later_performed":false}\n' \
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
    "event": "operation_magnitude_repair_validation_environment",
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
    "prototype.graph_encoder.tests.test_operation_magnitude_repair_contract",
    "prototype.graph_encoder.tests.test_operation_magnitude_repair_runtime",
)
suite = unittest.TestSuite(
    unittest.TestLoader().loadTestsFromName(name) for name in names
)
result = unittest.TextTestRunner(verbosity=2).run(suite)
print(json.dumps({
    "event": "operation_magnitude_repair_focused_suite",
    "tests_run": result.testsRun,
    "failures": len(result.failures),
    "errors": len(result.errors),
    "skipped": len(result.skipped),
}, sort_keys=True, separators=(",", ":")), flush=True)
if result.testsRun < 13 or not result.wasSuccessful() or result.skipped:
    raise SystemExit("all focused repair tests must pass with zero skips")
PY

container_python -B - <<'PY'
import json
import unittest

suite = unittest.TestLoader().discover(
    "prototype/graph_encoder/tests", top_level_dir="."
)
result = unittest.TextTestRunner(verbosity=1).run(suite)
print(json.dumps({
    "event": "operation_magnitude_repair_complete_graph_encoder_suite",
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
        "event": "operation_magnitude_repair_regression_suite",
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
print("OPERATION_MAGNITUDE_REPAIR_PYTHON38_GRAMMAR_OK {}".format(
    len(paths)
), flush=True)
PY

container_python -B - <<'PY'
import ast
from pathlib import Path
import prototype.graph_encoder as package

assert "POSITIVE_OPERATION_MAGNITUDE_PARAMETERIZATION" in package.__all__
assert "LEGACY_OPERATION_MAGNITUDE_PARAMETERIZATION" in package.__all__
assert "build_legacy_matched_ge1_models" in package.__all__
assert package.POSITIVE_OPERATION_MAGNITUDE_PARAMETERIZATION == (
    "GE1-OPERATION-MAGNITUDE-POSITIVE-v1"
)

production = (
    "decoder_contract.py", "config.py", "shared_decoder.py", "model.py",
    "checkpoint.py", "provenance.py", "training.py", "metrics.py",
)
root = Path("prototype/graph_encoder")
for name in production:
    source = (root / name).read_text(encoding="utf-8")
    tree = ast.parse(source, filename=name)
    imported_modules = {
        node.module for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert "prototype.model_data.loader" not in imported_modules

source = (root / "autonomous.py").read_text(encoding="utf-8")
tree = ast.parse(source)
for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name in (
        "autonomous_input_from_paired", "run_autonomous_evaluation",
        "_decode_condition",
    ):
        arguments = [item.arg for item in node.args.args + node.args.kwonlyargs]
        assert "target" not in arguments
        assert "operation_sequence" not in arguments

decoder_source = (root / "shared_decoder.py").read_text(encoding="utf-8")
decoder_tree = ast.parse(decoder_source)
decoder_class = next(
    node for node in decoder_tree.body
    if isinstance(node, ast.ClassDef) and node.name == "SharedGE1Decoder"
)
parameterize = next(
    node for node in decoder_class.body
    if isinstance(node, ast.FunctionDef)
    and node.name == "parameterize_remaining_geometry"
)
parameterize_source = ast.get_source_segment(decoder_source, parameterize)
assert "encoder" not in parameterize_source
assert "torch.sigmoid" in parameterize_source
assert "torch.finfo" in parameterize_source
print("OPERATION_MAGNITUDE_IMPORT_ACCESS_TARGET_AUDIT=PASS {}".format(
    len(package.__all__)
), flush=True)
PY

git diff --quiet "$ADR_PARENT_COMMIT" "$EXPECTED_COMMIT" -- \
  prototype/flat_baseline
bash -n prototype/graph_encoder/adroit/ge1_operation_magnitude_repair_cpu.slurm
git diff --check
test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git status --porcelain=v1 --untracked-files=all)"

printf '{"event":"operation_magnitude_repair_validation_terminal_success","commit":"%s","job_id":"%s","host":"%s","python":"3.8.13","pytorch":"1.11.0","execution_device":"cpu","repair_identity":"GE1-OPERATION-MAGNITUDE-POSITIVE-v1","legacy_identity":"GE1-OPERATION-MAGNITUDE-TANH-LEGACY-v1","focused_repair_skips":0,"complete_graph_encoder_skips":0,"flat_baseline_expected_skips":4,"operation_template_manifest_accessed":false,"operation_template_train_payload_accessed":false,"development_accessed":false,"systematic_rr_accessed":false,"test_er_accessed":false,"iid_accessed":false,"history_depth_accessed":false,"geometry_extrapolation_accessed":false,"other_corpus_accessed":false,"scientific_training_performed":false,"engineering_training_performed":false,"checkpoint_written":false,"cad_kernel_used":false,"stage6_performed":false,"c8_or_later_performed":false,"slurm_submitted_by_implementation_agent":false}\n' \
  "$EXPECTED_COMMIT" "$SLURM_JOB_ID" "$(hostname)"
