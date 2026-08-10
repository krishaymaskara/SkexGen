#!/bin/bash
#SBATCH --job-name=ge1-repaired-validation
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/network/km6349/ge1_repaired_sufficiency_validation_runs/repaired-validation-cpu-%j.out
#SBATCH --error=/scratch/network/km6349/ge1_repaired_sufficiency_validation_runs/repaired-validation-cpu-%j.err

set -Eeuo pipefail

terminal_failure() {
  status=$?
  trap - ERR
  printf '{"event":"repaired_sufficiency_validation_terminal_failure","commit":"%s","job_id":"%s","exit_status":%s,"operation_template_manifest_accessed":false,"operation_template_train_payload_accessed":false,"tiny_train_payload_accessed":false,"scaled_train_payload_accessed":false,"development_accessed":false,"systematic_rr_accessed":false,"test_er_accessed":false,"iid_accessed":false,"history_depth_accessed":false,"geometry_extrapolation_accessed":false,"other_corpus_accessed":false,"scientific_training_performed":false,"corpus_training_performed":false,"synthetic_unit_test_optimizer_steps_may_execute":true,"scientific_checkpoint_published":false,"cad_kernel_used":false,"stage6_performed":false,"c8_or_later_performed":false,"slurm_submitted_by_implementation_agent":false}\n' \
    "${EXPECTED_COMMIT:-unresolved}" "${SLURM_JOB_ID:-unresolved}" "$status" >&2
  exit "$status"
}
trap terminal_failure ERR

: "${EXPECTED_COMMIT:?submit with --export=ALL,EXPECTED_COMMIT=<exact-implementation-commit>}"
: "${SLURM_JOB_ID:?this validation must run under Slurm}"
REPOSITORY="${REPOSITORY:-/scratch/network/km6349/SkexGen-repaired-sufficiency-validation}"
LOG_PARENT="/scratch/network/km6349/ge1_repaired_sufficiency_validation_runs"
CONTAINER="/scratch/network/km6349/skexgen.sif"
PYTHON="/root/miniconda3/bin/python3.8"
ADR_PARENT_COMMIT="110ae7ab4b3308cd5c76cda26d8d62a18eec1cee"

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPYCACHEPREFIX="/tmp/ge1-repaired-validation-${SLURM_JOB_ID}"
export TMPDIR="/tmp/ge1-repaired-validation-tmp-${SLURM_JOB_ID}"
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

printf '{"event":"repaired_sufficiency_validation_start","commit":"%s","job_id":"%s","host":"%s","container":"%s","source_clean":true,"standalone_checkout":true,"execution_device":"cpu","protocol_version":"GE1-C7-REPAIRED-SUFFICIENCY-v1","repair_identity":"GE1-OPERATION-MAGNITUDE-POSITIVE-v1","operation_template_manifest_accessed":false,"operation_template_train_payload_accessed":false,"tiny_train_payload_accessed":false,"scaled_train_payload_accessed":false,"development_accessed":false,"systematic_rr_accessed":false,"test_er_accessed":false,"iid_accessed":false,"history_depth_accessed":false,"geometry_extrapolation_accessed":false,"other_corpus_accessed":false,"scientific_training_performed":false,"corpus_training_performed":false,"synthetic_unit_test_optimizer_steps_may_execute":true,"scientific_checkpoint_published":false,"cad_kernel_used":false,"stage6_performed":false,"c8_or_later_performed":false}\n' \
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
    "event": "repaired_sufficiency_validation_environment",
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
    "prototype.graph_encoder.tests.test_repaired_sufficiency_contract",
    "prototype.graph_encoder.tests.test_repaired_sufficiency_runtime",
)
suite = unittest.TestSuite(
    unittest.TestLoader().loadTestsFromName(name) for name in names
)
result = unittest.TextTestRunner(verbosity=2).run(suite)
print(json.dumps({
    "event": "repaired_sufficiency_focused_suite",
    "tests_run": result.testsRun,
    "failures": len(result.failures),
    "errors": len(result.errors),
    "skipped": len(result.skipped),
}, sort_keys=True, separators=(",", ":")), flush=True)
if result.testsRun < 32 or not result.wasSuccessful() or result.skipped:
    raise SystemExit("all repaired-sufficiency tests must pass with zero skips")
PY

container_python -B - <<'PY'
import json
import unittest

suite = unittest.TestLoader().discover(
    "prototype/graph_encoder/tests", top_level_dir="."
)
result = unittest.TextTestRunner(verbosity=1).run(suite)
print(json.dumps({
    "event": "repaired_sufficiency_complete_graph_encoder_suite",
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
flat_expected_methods = {
    "test_five_capsule_regressions_are_a_pinned_golden_cohort",
    "test_malformed_parent_fails_before_scientific_loading",
    "test_parent_equivalent_arms_reconcile_pinned_records",
    "test_real_bundle_two_family_golden_reconciliation",
}
for name, path, expected_skips in suites:
    loader = unittest.TestLoader()
    suite = loader.discover(path, top_level_dir=".")
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    print(json.dumps({
        "event": "repaired_sufficiency_regression_suite",
        "suite": name,
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
        "expected_skips": expected_skips,
    }, sort_keys=True, separators=(",", ":")), flush=True)
    observed_skip_methods = {
        item._testMethodName for item, unused_reason in result.skipped
    }
    expected_methods = flat_expected_methods if name == "flat_baseline" else set()
    if (
        not result.wasSuccessful()
        or len(result.skipped) != expected_skips
        or observed_skip_methods != expected_methods
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
print("REPAIRED_SUFFICIENCY_PYTHON38_GRAMMAR_OK {}".format(len(paths)), flush=True)
PY

container_python -B - <<'PY'
import ast
import hashlib
from pathlib import Path
import prototype.graph_encoder as package
from prototype.graph_encoder import repaired_sufficiency as repaired

assert repaired.REPAIRED_PROTOCOL_VERSION == "GE1-C7-REPAIRED-SUFFICIENCY-v1"
assert repaired.RepairedSufficiencyConfiguration().to_dict()[
    "operation_magnitude_parameterization"
] == "GE1-OPERATION-MAGNITUDE-POSITIVE-v1"
for name in (
    "REPAIRED_PROTOCOL_VERSION", "operation_fidelity_record",
    "verify_repaired_artifact",
):
    assert name in package.__all__

root = Path("prototype/graph_encoder")
source = (root / "repaired_sufficiency.py").read_text(encoding="utf-8")
tree = ast.parse(source)
imports = {
    alias.name for node in ast.walk(tree)
    if isinstance(node, ast.ImportFrom) for alias in node.names
}
assert "load_development" not in imports
assert "load_partition_physical_examples" not in imports
assert "secondary_systematic_validation" not in source
assert "build_legacy_matched_ge1_models" not in source
assert "fixed_protocol_final_epoch=REPAIRED_CHECKPOINT_EPOCH" in source
assert "targets =" in source
assert source.index("run_autonomous_evaluation(") < source.index("targets =")

autonomous_source = (root / "autonomous.py").read_text(encoding="utf-8")
autonomous_tree = ast.parse(autonomous_source)
for node in autonomous_tree.body:
    if isinstance(node, ast.FunctionDef) and node.name in (
        "autonomous_input_from_paired", "run_autonomous_evaluation",
        "_decode_condition",
    ):
        arguments = [item.arg for item in node.args.args + node.args.kwonlyargs]
        assert "target" not in arguments
        assert "target_geometry" not in arguments

frozen = {
    "c7_v2.py": "67f83efe3ef58478ad3c00b846d43f8928b7c642e42ee8fcbc19cce0f63e01aa",
    "tests/test_c7_v2_contract.py": "c8e607095b9d2444f9bb1438f6784f089bcce6abdafdcd307fb4643050d96330",
    "tests/test_c7_v2_runtime.py": "78721bdbaac7c2a04223e079151de610a90718dcfcbb1525a4cc1b66c800380c",
}
for relative, expected in frozen.items():
    assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == expected
print("REPAIRED_SUFFICIENCY_IMPORT_ACCESS_TARGET_HISTORY_AUDIT=PASS {}".format(
    len(package.__all__)
), flush=True)
PY

git diff --quiet "$ADR_PARENT_COMMIT" "$EXPECTED_COMMIT" -- \
  prototype/flat_baseline
bash -n prototype/graph_encoder/adroit/ge1_repaired_sufficiency_validation_cpu.slurm
bash -n prototype/graph_encoder/adroit/ge1_repaired_sufficiency_cpu.slurm
git diff --check
test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git status --porcelain=v1 --untracked-files=all)"

printf '{"event":"repaired_sufficiency_validation_terminal_success","commit":"%s","job_id":"%s","host":"%s","python":"3.8.13","pytorch":"1.11.0","execution_device":"cpu","protocol_version":"GE1-C7-REPAIRED-SUFFICIENCY-v1","repair_identity":"GE1-OPERATION-MAGNITUDE-POSITIVE-v1","focused_repaired_tests":32,"focused_repaired_skips":0,"complete_graph_encoder_skips":0,"flat_baseline_expected_skips":4,"operation_template_manifest_accessed":false,"operation_template_train_payload_accessed":false,"tiny_train_payload_accessed":false,"scaled_train_payload_accessed":false,"development_accessed":false,"systematic_rr_accessed":false,"test_er_accessed":false,"iid_accessed":false,"history_depth_accessed":false,"geometry_extrapolation_accessed":false,"other_corpus_accessed":false,"scientific_training_performed":false,"corpus_training_performed":false,"synthetic_unit_test_optimizer_steps_executed":true,"scientific_checkpoint_published":false,"cad_kernel_used":false,"stage6_performed":false,"c8_or_later_performed":false,"slurm_submitted_by_implementation_agent":false}\n' \
  "$EXPECTED_COMMIT" "$SLURM_JOB_ID" "$(hostname)"
