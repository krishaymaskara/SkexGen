#!/bin/bash
#SBATCH --job-name=ge1-stage6-narrow-builder
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --signal=B:TERM@60

set -Eeuo pipefail

terminal_failure() {
  status=$?
  trap - ERR
  printf '{"event":"stage6_narrow_builder_terminal_failure","commit":"%s","job_id":"%s","exit_status":%s}\n' \
    "${EXPECTED_COMMIT:-unresolved}" "${SLURM_JOB_ID:-unresolved}" "$status" >&2
  exit "$status"
}

terminal_timeout() {
  trap - ERR
  printf '{"event":"stage6_narrow_builder_terminal_timeout","commit":"%s","job_id":"%s","exit_status":124}\n' \
    "${EXPECTED_COMMIT:-unresolved}" "${SLURM_JOB_ID:-unresolved}" >&2
  exit 124
}

trap terminal_failure ERR
trap terminal_timeout TERM

: "${EXPECTED_COMMIT:?exact reviewed commit required}"
: "${EXPECTED_RUNNER_SHA256:?exact reviewed runner SHA-256 required}"
: "${SLURM_JOB_ID:?must execute under a separately authorized allocation}"
: "${CORPUS_ROOT:?exact authorized corpus root required}"
: "${OUTPUT_PARENT:?exact preparation output parent required}"

EXPECTED_MANIFEST_SHA256="a9ac86a6dede054fbbba57e0906b210bab26036c3f5c150b332038f78d2dadb7"
REPOSITORY="${REPOSITORY:-/scratch/network/km6349/SkexGen-stage6-structure-only}"
CONTAINER="${CONTAINER:-/scratch/network/km6349/skexgen.sif}"
PYTHON="${PYTHON:-/root/miniconda3/bin/python3.8}"
RUNNER="$REPOSITORY/prototype/graph_encoder/adroit/ge1_stage6_narrow_builder_cpu.slurm"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPYCACHEPREFIX="/tmp/ge1-stage6-builder-pyc-${SLURM_JOB_ID}"
export TMPDIR="/tmp/ge1-stage6-builder-${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"

container_python() {
  apptainer exec --cleanenv \
    --env OMP_NUM_THREADS=1 \
    --env MKL_NUM_THREADS=1 \
    --env PYTHONPYCACHEPREFIX="$PYTHONPYCACHEPREFIX" \
    --env TMPDIR="$TMPDIR" \
    --bind "$REPOSITORY:$REPOSITORY:ro" \
    --bind "$CORPUS_ROOT:$CORPUS_ROOT:ro" \
    --bind "$OUTPUT_PARENT:$OUTPUT_PARENT:rw" \
    "$CONTAINER" "$PYTHON" "$@"
}

test -d "$REPOSITORY/.git"
test ! -f "$REPOSITORY/.git"
test -d "$CORPUS_ROOT"
test ! -L "$CORPUS_ROOT"
test -d "$OUTPUT_PARENT"
test ! -L "$OUTPUT_PARENT"
test -f "$CONTAINER"
test "$(sha256sum "$RUNNER" | cut -d ' ' -f 1)" = "$EXPECTED_RUNNER_SHA256"
cd "$REPOSITORY"
test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git symbolic-ref -q --short HEAD || true)"
test -z "$(git status --porcelain=v1 --untracked-files=all)"
test "$(git rev-parse --show-toplevel)" = "$REPOSITORY"
test -z "$(git rev-parse --show-superproject-working-tree)"

container_python -B - <<'PY'
import sys
assert sys.version_info[:3] == (3, 8, 13)
PY
container_python -B -m prototype.graph_encoder.stage6_narrow_builder_audit "$RUNNER"
container_python -B -m prototype.graph_encoder.stage6_narrow_loader_audit
bash -n "$RUNNER"

container_python -B -m prototype.graph_encoder.stage6_narrow_builder \
  --corpus-root "$CORPUS_ROOT" \
  --output-parent "$OUTPUT_PARENT" \
  --job-id "$SLURM_JOB_ID" \
  --expected-source-manifest-sha256 "$EXPECTED_MANIFEST_SHA256" \
  --partition both

container_python -B - "$OUTPUT_PARENT" <<'PY'
import json
import sys
from pathlib import Path
from prototype.graph_encoder.stage6_narrow_builder import verify_preparation_receipt
from prototype.graph_encoder.stage6_narrow_loader import (
    load_stage6_development, load_stage6_train,
)
root = Path(sys.argv[1])
train, train_evidence = load_stage6_train(
    root / "stage6-train/index.json", root / "stage6-train/payloads"
)
development, development_evidence = load_stage6_development(
    root / "stage6-development/index.json",
    root / "stage6-development/payloads",
)
receipt = verify_preparation_receipt(root)
print(json.dumps({
    "event":"stage6_narrow_builder_verified",
    "train_family_count":len(train),
    "development_family_count":len(development),
    "train_index_sha256":train_evidence["index_identity_sha256"],
    "development_index_sha256":development_evidence["index_identity_sha256"],
    "receipt_status":receipt["verification_status"],
}, sort_keys=True, separators=(",", ":")), flush=True)
PY

test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git status --porcelain=v1 --untracked-files=all)"
printf '{"event":"stage6_narrow_builder_terminal_success","commit":"%s","job_id":"%s","receipt":"%s"}\n' \
  "$EXPECTED_COMMIT" "$SLURM_JOB_ID" "$OUTPUT_PARENT/stage6-preparation-receipt.json"
