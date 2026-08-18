#!/bin/bash
# Prospective producer runner. Allocation flags are intentionally absent:
# reviewed timing evidence must determine them at the future sbatch boundary.
#SBATCH --signal=B:TERM@60

set -Eeuo pipefail

infrastructure_failure() {
  status="${1:-$?}"
  trap - ERR
  printf '{"event":"stage6_structure_only_producer_infrastructure_failure","commit":"%s","job_id":"%s","exit_status":%s}\n' \
    "${EXPECTED_COMMIT:-unresolved}" "${SLURM_JOB_ID:-unresolved}" "$status" >&2
  exit "$status"
}

terminal_timeout() {
  trap - ERR
  printf '{"event":"stage6_structure_only_producer_terminal_timeout","commit":"%s","job_id":"%s","exit_status":124}\n' \
    "${EXPECTED_COMMIT:-unresolved}" "${SLURM_JOB_ID:-unresolved}" >&2
  exit 124
}

trap infrastructure_failure ERR
trap terminal_timeout TERM

: "${EXPECTED_COMMIT:?exact reviewed commit required}"
: "${EXPECTED_RUNNER_SHA256:?exact reviewed runner SHA-256 required}"
: "${SLURM_JOB_ID:?must run under a separately authorized Slurm allocation}"
: "${TRAIN_INPUT_ROOT:?narrow authorized train input root required}"
: "${DEVELOPMENT_INPUT_ROOT:?narrow authorized development input root required}"
: "${RUN_PARENT:?governed output parent required}"
: "${TIMING_EVIDENCE:?prospective timing evidence in RUN_PARENT required}"

REPOSITORY="${REPOSITORY:-/scratch/network/km6349/SkexGen-stage6-structure-only}"
CONTAINER="${CONTAINER:-/scratch/network/km6349/skexgen.sif}"
PYTHON="${PYTHON:-/root/miniconda3/bin/python3.8}"
RUNNER="$REPOSITORY/prototype/graph_encoder/adroit/ge1_stage6_structure_only_producer.slurm"
OUTPUT_DIR="$RUN_PARENT/ge1-stage6-structure-only-producer-${EXPECTED_COMMIT}-${SLURM_JOB_ID}"
CHECKPOINT_BUNDLE_DIR="$RUN_PARENT/ge1-stage6-structure-only-checkpoints-${EXPECTED_COMMIT}-${SLURM_JOB_ID}"
TRAIN_INDEX="$TRAIN_INPUT_ROOT/index.json"
TRAIN_ROOT="$TRAIN_INPUT_ROOT/payloads"
DEVELOPMENT_INDEX="$DEVELOPMENT_INPUT_ROOT/index.json"
DEVELOPMENT_ROOT="$DEVELOPMENT_INPUT_ROOT/payloads"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPYCACHEPREFIX="/tmp/ge1-stage6-producer-pyc-${SLURM_JOB_ID}"
export TMPDIR="/tmp/ge1-stage6-producer-${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"

container_python() {
  apptainer exec --cleanenv \
    --env OMP_NUM_THREADS=1 \
    --env MKL_NUM_THREADS=1 \
    --env PYTHONPYCACHEPREFIX="$PYTHONPYCACHEPREFIX" \
    --env TMPDIR="$TMPDIR" \
    --bind "$REPOSITORY:$REPOSITORY:ro" \
    --bind "$TRAIN_INPUT_ROOT:$TRAIN_INPUT_ROOT:ro" \
    --bind "$DEVELOPMENT_INPUT_ROOT:$DEVELOPMENT_INPUT_ROOT:ro" \
    --bind "$RUN_PARENT:$RUN_PARENT:rw" \
    "$CONTAINER" "$PYTHON" "$@"
}

test -d "$REPOSITORY/.git"
test ! -f "$REPOSITORY/.git"
test -d "$RUN_PARENT"
test -f "$TIMING_EVIDENCE"
test -f "$CONTAINER"
test ! -e "$OUTPUT_DIR"
test ! -e "$CHECKPOINT_BUNDLE_DIR"
test "$(sha256sum "$RUNNER" | cut -d ' ' -f 1)" = "$EXPECTED_RUNNER_SHA256"
cd "$REPOSITORY"
test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git symbolic-ref -q --short HEAD || true)"
test -z "$(git status --porcelain=v1 --untracked-files=all)"
test "$(git rev-parse --show-toplevel)" = "$REPOSITORY"
test -z "$(git rev-parse --show-superproject-working-tree)"
git cat-file -e "$EXPECTED_COMMIT^{commit}"

# This pure validation is deliberately before either scientific root is read.
container_python -B - "$TIMING_EVIDENCE" <<'PY'
import json
import sys
from pathlib import Path
from prototype.graph_encoder.stage6_structure_only_producer import validate_timing_evidence
timing = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
validate_timing_evidence(timing, allow_fallback=True, required_device="cpu")
print('{"event":"stage6_structure_only_prospective_gates_passed"}', flush=True)
PY

test -f "$TRAIN_INDEX"
test -d "$TRAIN_ROOT"
test -f "$DEVELOPMENT_INDEX"
test -d "$DEVELOPMENT_ROOT"

container_python -B - <<'PY'
import platform
import sys
import torch
assert sys.version_info[:3] == (3, 8, 13)
assert torch.__version__.split("+")[0] == "1.11.0"
assert torch.get_num_threads() == 1
print('{"event":"stage6_structure_only_producer_environment","python":"%s","pytorch":"%s","threads":1}' % (
    platform.python_version(), torch.__version__), flush=True)
PY

container_python -B - <<'PY'
import json
import unittest
names = (
    "prototype.graph_encoder.tests.test_stage6_structure_only_contract",
    "prototype.graph_encoder.tests.test_stage6_structure_only_runtime",
    "prototype.graph_encoder.tests.test_stage6_structure_only_producer_contract",
    "prototype.graph_encoder.tests.test_stage6_structure_only_producer_runtime",
    "prototype.graph_encoder.tests.test_stage6_narrow_loader_contract",
)
suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(name) for name in names)
declared = suite.countTestCases()
result = unittest.TextTestRunner(verbosity=2).run(suite)
print(json.dumps({"event":"stage6_structure_only_producer_focused_tests",
                  "declared":declared,"run":result.testsRun,"skipped":len(result.skipped)},
                 sort_keys=True, separators=(",", ":")), flush=True)
if result.testsRun != declared or not result.wasSuccessful() or result.skipped:
    raise SystemExit("all focused producer tests must pass with zero skips")
PY

container_python -B - <<'PY'
import unittest
suite = unittest.TestLoader().discover("prototype/graph_encoder/tests", top_level_dir=".")
result = unittest.TextTestRunner(verbosity=1).run(suite)
if not result.wasSuccessful() or result.skipped:
    raise SystemExit("complete graph-encoder discovery must pass with zero skips")
PY
container_python -B tools/check_documentation.py
container_python -B -m compileall -q prototype/graph_encoder
container_python -B - <<'PY'
import ast
from pathlib import Path
paths = tuple(sorted(Path("prototype/graph_encoder").glob("*.py"))) + tuple(
    sorted(Path("prototype/graph_encoder/tests").glob("test_*.py")))
for path in paths:
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path), feature_version=(3, 8))
print("GE1_STAGE6_PRODUCER_PYTHON38_AST=PASS {}".format(len(paths)), flush=True)
PY
container_python -B -m prototype.graph_encoder.stage6_structure_only_producer_audit "$RUNNER"
container_python -B -m prototype.graph_encoder.stage6_narrow_loader_audit
container_python -B -m prototype.graph_encoder.stage6_checkpoint_bundle_audit
bash -n "$RUNNER"
git show --check "$EXPECTED_COMMIT" --

set +e
container_python -B -m prototype.graph_encoder.stage6_structure_only_producer \
  --train-root "$TRAIN_ROOT" \
  --train-index "$TRAIN_INDEX" \
  --development-root "$DEVELOPMENT_ROOT" \
  --development-index "$DEVELOPMENT_INDEX" \
  --timing-evidence "$TIMING_EVIDENCE" \
  --output-dir "$OUTPUT_DIR" \
  --checkpoint-bundle-dir "$CHECKPOINT_BUNDLE_DIR" \
  --repository-root "$REPOSITORY" \
  --expected-commit "$EXPECTED_COMMIT" \
  --job-id "$SLURM_JOB_ID" \
  --allow-fallback \
  --require-selected-device cpu
producer_status=$?
set -e
if [[ "$producer_status" -eq 42 ]]; then
  printf '{"event":"stage6_structure_only_producer_scientific_failure","commit":"%s","job_id":"%s","exit_status":42}\n' \
    "$EXPECTED_COMMIT" "$SLURM_JOB_ID" >&2
  exit 42
fi
if [[ "$producer_status" -ne 0 ]]; then
  infrastructure_failure "$producer_status"
fi

(
  cd "$OUTPUT_DIR"
  sha256sum -c SHA256SUMS
)
(
  cd "$CHECKPOINT_BUNDLE_DIR"
  sha256sum -c SHA256SUMS
)
container_python -B - "$OUTPUT_DIR" "$CHECKPOINT_BUNDLE_DIR" <<'PY'
import json
import sys
from prototype.graph_encoder.stage6_structure_only import summarize_execution
from prototype.graph_encoder.stage6_structure_only_producer import (
    load_execution_record, verify_checkpoint_bundle, verify_producer_artifact,
)
verification = verify_producer_artifact(sys.argv[1])
payload = load_execution_record(sys.argv[1])
bundle = verify_checkpoint_bundle(sys.argv[2], retained_seeds=tuple(payload["retained_seeds"]))
assert bundle["bundle_sha256"] == verification["checkpoint_bundle_sha256"]
unused, summary = summarize_execution(payload)
print(json.dumps({"event":"stage6_structure_only_producer_verified", **verification,
                  "interpretation_category":summary["interpretation_category"]},
                 sort_keys=True, separators=(",", ":")), flush=True)
if summary["interpretation_category"] == "inconclusive":
    print('{"event":"stage6_structure_only_producer_scientific_failure"}', flush=True)
PY

test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git status --porcelain=v1 --untracked-files=all)"
printf '{"event":"stage6_structure_only_producer_terminal_success","commit":"%s","job_id":"%s","artifact":"%s","checkpoint_bundle":"%s"}\n' \
  "$EXPECTED_COMMIT" "$SLURM_JOB_ID" "$OUTPUT_DIR" "$CHECKPOINT_BUNDLE_DIR"
