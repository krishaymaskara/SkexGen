#!/bin/bash
#SBATCH --job-name=flat-vq-diagnosis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/network/km6349/phase-b-vq-diagnosis-%j.out
#SBATCH --error=/scratch/network/km6349/phase-b-vq-diagnosis-%j.err

set -Eeuo pipefail

CURRENT_STAGE="startup"

report_failure() {
  local exit_code="$?"
  local failed_line="$1"
  trap - ERR
  printf 'failed_stage=%s\n' "$CURRENT_STAGE" >&2
  printf 'failed_line=%s\n' "$failed_line" >&2
  printf 'exit_code=%s\n' "$exit_code" >&2
  exit "$exit_code"
}

trap 'report_failure "$LINENO"' ERR

stage() {
  CURRENT_STAGE="$1"
  printf 'stage=%s\n' "$CURRENT_STAGE"
}

REPOSITORY="/scratch/network/km6349/SkexGen"
CORPUS="/scratch/network/km6349/controlled_corpora/b0-pilot-680-seed2026"
TRAINING_RUN="/scratch/network/km6349/flat_baseline_runs/b0-pilot-680-k32-lr1e3-seed2026"
CHECKPOINT="${TRAINING_RUN}/best.pt"
CONTAINER="/scratch/network/km6349/skexgen.sif"
PYTHON="/root/miniconda3/bin/python3.8"
RUN_ROOT="/scratch/network/km6349/flat_baseline_vq_diagnostics"
: "${REVIEWED_COMMIT:?set REVIEWED_COMMIT to the reviewed diagnosis commit}"
: "${SLURM_JOB_ID:?this workflow must run under Slurm}"

stage repository_preflight
cd "$REPOSITORY"
test "$(git branch --show-current)" = "flat-mixed-baseline"
test "$(git rev-parse HEAD)" = "$REVIEWED_COMMIT"
test "$(git rev-parse origin/flat-mixed-baseline)" = "$REVIEWED_COMMIT"
test -z "$(git status --porcelain=v1 --untracked-files=all)"
test -d "$CORPUS"
test -d "$TRAINING_RUN"
test -f "$CHECKPOINT"
mkdir -p "$RUN_ROOT"

SMOKE_A="${RUN_ROOT}/vq-smoke-${SLURM_JOB_ID}-a"
SMOKE_B="${RUN_ROOT}/vq-smoke-${SLURM_JOB_ID}-b"
FULL="${RUN_ROOT}/vq-diagnosis-${REVIEWED_COMMIT}-${SLURM_JOB_ID}"
REPORT="${RUN_ROOT}/vq-diagnosis-report-${SLURM_JOB_ID}.json"
test ! -e "$SMOKE_A"
test ! -L "$SMOKE_A"
test ! -e "$SMOKE_B"
test ! -L "$SMOKE_B"
test ! -e "$FULL"
test ! -L "$FULL"
test ! -e "$REPORT"
test ! -L "$REPORT"

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPYCACHEPREFIX="/tmp/flat-vq-diagnosis-pycache-${SLURM_JOB_ID}"

container_python() {
  apptainer exec \
    --cleanenv \
    --env CUDA_VISIBLE_DEVICES="" \
    --env OMP_NUM_THREADS=1 \
    --env MKL_NUM_THREADS=1 \
    --env PYTHONPYCACHEPREFIX="$PYTHONPYCACHEPREFIX" \
    --bind /scratch/network/km6349:/scratch/network/km6349 \
    "$CONTAINER" "$PYTHON" "$@"
}

container_python - "$CORPUS" "$CHECKPOINT" <<'PY'
import hashlib
from pathlib import Path
import sys
import torch

from prototype.model_data.loader import load_physical_examples

assert sys.version_info[:2] == (3, 8)
assert torch.__version__.split("+")[0] == "1.11.0"
assert not torch.cuda.is_available()
corpus = tuple(load_physical_examples(sys.argv[1], "iid"))
counts = {
    name: sum(item.partition == name for item in corpus)
    for name in ("train", "validation", "test")
}
assert len(corpus) == 680
assert counts == {"train": 544, "validation": 68, "test": 68}
digest = hashlib.sha256(Path(sys.argv[2]).read_bytes()).hexdigest()
assert len(digest) == 64
print("checkpoint_sha256={}".format(digest))
print("authoritative_partition_counts={}".format(counts))
PY

stage publication_preflight
container_python - "$RUN_ROOT" "$SLURM_JOB_ID" <<'PY'
from pathlib import Path
import os
import shutil
import sys

from prototype.flat_baseline.evaluate_length_conditioned import (
    EvaluationError,
    _atomic_no_replace,
    _fsync_directory,
)

root = Path(sys.argv[1])
preflight = root / (".vq-publication-preflight-" + sys.argv[2])
preflight.mkdir()
source = preflight / "backing"
public = preflight / "public"
collision_source = preflight / "collision-backing"
existing = preflight / "existing"
source.mkdir()
collision_source.mkdir()
existing.mkdir()
with (source / "complete").open("wb") as stream:
    stream.write(b"complete")
    stream.flush()
    os.fsync(stream.fileno())
_fsync_directory(source)
try:
    result = _atomic_no_replace(source, public)
    assert public.is_dir()
    assert (public / "complete").read_bytes() == b"complete"
    try:
        _atomic_no_replace(collision_source, existing)
    except EvaluationError as exc:
        assert exc.code == "output_collision"
    else:
        raise RuntimeError("publication collision unexpectedly succeeded")
    assert existing.is_dir()
    print("publication_backend={}".format(result["publication_backend"]))
    print("publication_preflight=PASS")
finally:
    shutil.rmtree(str(preflight))
PY

stage regressions
container_python -B -m unittest discover \
  -s prototype/flat_baseline/tests -t .
container_python -B -m unittest discover \
  -s prototype/model_data/tests -t .
container_python -B -m unittest \
  prototype.flat_baseline.tests.test_diagnose_vq_collapse

run_diagnosis() {
  local output="$1"
  shift
  container_python -B -m prototype.flat_baseline.diagnose_vq_collapse run \
    --corpus-dir "$CORPUS" \
    --training-run-dir "$TRAINING_RUN" \
    --output-dir "$output" \
    --split-manifest iid \
    --reviewed-commit "$REVIEWED_COMMIT" \
    --batch-size 32 \
    "$@"
}

stage smoke_a
run_diagnosis "$SMOKE_A" \
  --train-limit 4 \
  --validation-limit 4 \
  --sensitivity-family-limit 2
container_python -B -m prototype.flat_baseline.diagnose_vq_collapse verify \
  --output "$SMOKE_A" \
  --corpus-dir "$CORPUS" \
  --training-run-dir "$TRAINING_RUN" \
  --reviewed-commit "$REVIEWED_COMMIT"

stage smoke_b
run_diagnosis "$SMOKE_B" \
  --train-limit 4 \
  --validation-limit 4 \
  --sensitivity-family-limit 2

stage smoke_replay
container_python -B -m prototype.flat_baseline.diagnose_vq_collapse verify \
  --output "$SMOKE_A" \
  --compare-output "$SMOKE_B" \
  --corpus-dir "$CORPUS" \
  --training-run-dir "$TRAINING_RUN" \
  --reviewed-commit "$REVIEWED_COMMIT"

stage full_diagnosis
run_diagnosis "$FULL" --sensitivity-family-limit 4

stage full_verification
container_python -B -m prototype.flat_baseline.diagnose_vq_collapse verify \
  --output "$FULL" \
  --corpus-dir "$CORPUS" \
  --training-run-dir "$TRAINING_RUN" \
  --reviewed-commit "$REVIEWED_COMMIT"

stage final_report
container_python -B -m prototype.flat_baseline.diagnose_vq_collapse verify \
  --output "$FULL" \
  --corpus-dir "$CORPUS" \
  --training-run-dir "$TRAINING_RUN" \
  --reviewed-commit "$REVIEWED_COMMIT" \
  --smoke-output-a "$SMOKE_A" \
  --smoke-output-b "$SMOKE_B" \
  --report-output "$REPORT"

test -d "$SMOKE_A"
test -d "$SMOKE_B"
test -d "$FULL"
test -f "$REPORT"
test -z "$(find "$RUN_ROOT" -maxdepth 1 -name '.vq-*.tmp-*' -print -quit)"

container_python - "$REPORT" <<'PY'
import json
from pathlib import Path
import sys

report = json.loads(Path(sys.argv[1]).read_text())
assert report["byte_identical_smoke_replay"] is True
assert report["full_diagnosis_verified"] is True
assert report["test_partition_evaluated"] is False
PY

stage complete
