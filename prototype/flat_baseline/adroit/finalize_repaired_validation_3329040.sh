#!/bin/bash

set -Eeuo pipefail

JOB_ID="3329040"
RUN_COMMIT="ce4abca8f450745a8832c0189aaae4a7d8263ec0"
REPOSITORY="/scratch/network/km6349/SkexGen"
CORPUS="/scratch/network/km6349/controlled_corpora/b0-pilot-680-seed2026"
CHECKPOINT="/scratch/network/km6349/flat_baseline_runs/train-kmeans-full-d549ebe4478e166fda8d54f2b173572a1ab52e7a-3326757/best.pt"
CONTAINER="/scratch/network/km6349/skexgen.sif"
PYTHON="/root/miniconda3/bin/python3.8"
NAMESPACE="/scratch/network/km6349/flat_baseline_evaluations/phase-b-repaired-validation-${RUN_COMMIT}-${JOB_ID}"
EVALUATION="${NAMESPACE}/evaluation"
EVIDENCE="${NAMESPACE}/workflow-evidence"
REPORT="${NAMESPACE}/gate-a-to-d-inputs.json"
HASH_MANIFEST="${NAMESPACE}/sha256-manifest.txt"
STDOUT="/scratch/network/km6349/phase-b-repaired-validation-${JOB_ID}.out"
STDERR="/scratch/network/km6349/phase-b-repaired-validation-${JOB_ID}.err"
: "${RECOVERY_COMMIT:?set RECOVERY_COMMIT to the reviewed recovery commit}"

CURRENT_STAGE="initialization"
failure_report() {
  local status=$?
  printf 'recovery_failure_stage=%s exit_code=%s\n' \
    "$CURRENT_STAGE" "$status" >&2
  exit "$status"
}
trap failure_report ERR

stage() {
  CURRENT_STAGE="$1"
  printf 'recovery_stage=%s\n' "$CURRENT_STAGE"
}

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONHASHSEED=0
export PYTHONPYCACHEPREFIX="/tmp/flat-repaired-validation-recovery-${JOB_ID}"

container_python() {
  apptainer exec \
    --cleanenv \
    --env CUDA_VISIBLE_DEVICES="" \
    --env OMP_NUM_THREADS=1 \
    --env MKL_NUM_THREADS=1 \
    --env PYTHONHASHSEED=0 \
    --env PYTHONPYCACHEPREFIX="$PYTHONPYCACHEPREFIX" \
    --bind /scratch/network/km6349:/scratch/network/km6349 \
    "$CONTAINER" "$PYTHON" "$@"
}

stage recovery_source_preflight
cd "$REPOSITORY"
test "$(git branch --show-current)" = "flat-mixed-baseline"
test "$(git rev-parse HEAD)" = "$RECOVERY_COMMIT"
test "$(git rev-parse origin/flat-mixed-baseline)" = "$RECOVERY_COMMIT"
test -z "$(git status --porcelain=v1 --untracked-files=all)"
test -d "$NAMESPACE"
test -e "$EVALUATION"
test -d "$EVIDENCE"
test -e "$REPORT"
test -f "$HASH_MANIFEST"
test -f "$STDOUT"
test -f "$STDERR"
test -f "$CHECKPOINT"
test -f "$CONTAINER"

stage original_failure_preflight
SACCT_RECORD="$(
  sacct -X -j "$JOB_ID" --noheader --parsable2 \
    --format=JobIDRaw,State,ExitCode
)"
printf '%s\n' "$SACCT_RECORD"
printf '%s\n' "$SACCT_RECORD" \
  | awk -F'|' -v job="$JOB_ID" \
      '$1 == job && $2 ~ /^FAILED/ && $3 != "0:0" { found=1 } END { exit !found }'
grep -F 'stage=artifact_validation' "$STDOUT"
grep -F 'stage=artifact_manifest' "$STDOUT"
if grep -F 'stage=complete' "$STDOUT"; then
  printf 'original job unexpectedly reached complete\n' >&2
  exit 1
fi
grep -F \
  'failure_stage=artifact_manifest failure_line=261 exit_code=1' \
  "$STDERR"
test "$(wc -l < "$HASH_MANIFEST")" -eq 9

stage independent_artifact_revalidation
container_python -B -m \
  prototype.flat_baseline.verify_evaluation_artifacts \
  --output "$EVALUATION" \
  --corpus-dir "$CORPUS" \
  --checkpoint "$CHECKPOINT" \
  --reviewed-commit "$RUN_COMMIT" \
  --expected-authoritative-family-count 680 \
  --expected-authoritative-validation-count 68 \
  --existing-report "$REPORT" \
  --job-id "$JOB_ID" \
  --regression-status passed \
  --workflow-evidence-dir "$EVIDENCE" \
  --repaired-full-contract

stage manifest_recovery
container_python -B -m prototype.flat_baseline.artifact_manifest \
  --namespace "$NAMESPACE" \
  --output "$HASH_MANIFEST" \
  --replace-incomplete
test "$(wc -l < "$HASH_MANIFEST")" -eq 15
sha256sum -c "$HASH_MANIFEST"

stage recovery_complete
printf '%s\n' \
  "recovery_record job_id=${JOB_ID} original_slurm_state=FAILED" \
  "original_failure_stage=artifact_manifest original_failure_line=261" \
  "inference_status=succeeded artifact_validation_status=revalidated" \
  "manifest_replacement=atomic manifest_entry_count=15" \
  "run_commit=${RUN_COMMIT} recovery_commit=${RECOVERY_COMMIT}"
