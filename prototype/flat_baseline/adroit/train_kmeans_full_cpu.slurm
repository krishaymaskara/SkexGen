#!/bin/bash
#SBATCH --job-name=flat-vq-kmeans-full
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/network/km6349/flat-vq-kmeans-full-%j.out
#SBATCH --error=/scratch/network/km6349/flat-vq-kmeans-full-%j.err

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
BASELINE_RUN="/scratch/network/km6349/flat_baseline_runs/b0-pilot-680-k32-lr1e3-seed2026"
PILOT_COMMIT="856d6a62187f3d7f4b5cc7a9b4c9efc655b74ce1"
PILOT_JOB="3326278"
RUN_ROOT="/scratch/network/km6349/flat_baseline_runs"
PILOT_RUN="${RUN_ROOT}/train-kmeans-pilot-${PILOT_COMMIT}-${PILOT_JOB}"
CONTAINER="/scratch/network/km6349/skexgen.sif"
PYTHON="/root/miniconda3/bin/python3.8"
: "${REVIEWED_COMMIT:?set REVIEWED_COMMIT to the reviewed promotion commit}"
: "${SLURM_JOB_ID:?this workflow must run under Slurm}"

OUTPUT="${RUN_ROOT}/train-kmeans-full-${REVIEWED_COMMIT}-${SLURM_JOB_ID}"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPYCACHEPREFIX="/tmp/flat-vq-kmeans-full-${SLURM_JOB_ID}"

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

stage repository_preflight
cd "$REPOSITORY"
test "$(git branch --show-current)" = "flat-mixed-baseline"
test "$(git rev-parse HEAD)" = "$REVIEWED_COMMIT"
test "$(git rev-parse origin/flat-mixed-baseline)" = "$REVIEWED_COMMIT"
test -z "$(git status --porcelain=v1 --untracked-files=all)"
test -d "$CORPUS"
test -f "${BASELINE_RUN}/best.pt"
test -f "${BASELINE_RUN}/metrics.jsonl"
test -d "$PILOT_RUN"
test -f "${PILOT_RUN}/last.pt"
test ! -e "$OUTPUT"
test ! -L "$OUTPUT"

stage environment_preflight
container_python - <<'PY'
import sys
import torch

assert sys.version_info[:2] == (3, 8)
assert torch.__version__.split("+")[0] == "1.11.0"
assert not torch.cuda.is_available()
print("python_version={}".format(sys.version.split()[0]))
print("pytorch_version={}".format(torch.__version__))
PY

stage partition_preflight
container_python - "$CORPUS" <<'PY'
import sys

from prototype.model_data.loader import load_physical_examples

examples = tuple(load_physical_examples(sys.argv[1], "iid"))
counts = {
    name: sum(item.partition == name for item in examples)
    for name in ("train", "validation", "test")
}
assert len(examples) == 680
assert counts == {"train": 544, "validation": 68, "test": 68}
print("authoritative_partition_counts={}".format(counts))
print("test_partition_evaluated=False")
PY

stage regressions
container_python -B -m unittest \
  prototype.flat_baseline.tests.test_vq_full_retrain \
  prototype.flat_baseline.tests.test_vq_pilot \
  prototype.flat_baseline.tests.test_training_config
container_python -B -m unittest discover \
  -s prototype/flat_baseline/tests -t .
container_python -B -m unittest discover \
  -s prototype/model_data/tests -t .

stage pilot_provenance
container_python -B - "$PILOT_RUN" <<'PY'
import json
from pathlib import Path
import sys
import torch

root = Path(sys.argv[1])
decision = json.loads((root / "pilot_decision.json").read_text())
checkpoint = torch.load(str(root / "last.pt"), map_location="cpu")
assert decision["decision"] == "PASS_FOR_FULL_RETRAIN"
assert decision["completed_epoch"] == 5
assert decision["test_partition_evaluated"] is False
assert checkpoint["epoch"] == 5
assert checkpoint["training_config"]["vq_init"] == "train-kmeans"
assert checkpoint["training_config"]["seed"] == 2026
assert checkpoint["training_config"]["learning_rate"] == 1e-3
assert checkpoint["training_config"]["device"] == "cpu"
assert checkpoint["model_config"]["codebook_size"] == 32
assert checkpoint["data_state"]["initialization_report_sha256"]
print("SUCCESSFUL_PILOT_PROVENANCE=PASS")
PY

stage epoch_budget_preflight
container_python -B - "$BASELINE_RUN" <<'PY'
import sys

from prototype.flat_baseline.vq_full_retrain import (
    authoritative_epoch_budget,
)

budget = authoritative_epoch_budget(sys.argv[1])
assert budget == 50
print("authoritative_full_epoch_budget={}".format(budget))
PY

stage full_training
container_python -B -m prototype.flat_baseline.vq_full_retrain \
  --corpus-dir "$CORPUS" \
  --baseline-run "$BASELINE_RUN" \
  --pilot-run "$PILOT_RUN" \
  --output-dir "$OUTPUT" \
  --reviewed-commit "$REVIEWED_COMMIT" \
  --vq-init train-kmeans \
  --device cpu \
  --expected-train-count 544 \
  --expected-validation-count 68 \
  --expected-test-count 68

stage artifact_validation
container_python -B - "$OUTPUT" "$BASELINE_RUN" <<'PY'
import hashlib
import json
from pathlib import Path
import sys
import torch

root = Path(sys.argv[1])
baseline = Path(sys.argv[2])
required = {
    "artifact_manifest.json",
    "best.pt",
    "epoch_metrics.jsonl",
    "initialization_report.json",
    "last.pt",
    "partition_provenance.json",
    "run_configuration.json",
    "training_decision.json",
}
assert required == {path.name for path in root.iterdir()}
manifest = json.loads((root / "artifact_manifest.json").read_text())
assert set(manifest["artifact_names"]) == required - {
    "artifact_manifest.json"
}
for name, expected in manifest["artifact_sha256"].items():
    observed = hashlib.sha256((root / name).read_bytes()).hexdigest()
    assert observed == expected
configuration = json.loads((root / "run_configuration.json").read_text())
decision = json.loads((root / "training_decision.json").read_text())
partitions = json.loads((root / "partition_provenance.json").read_text())
baseline_checkpoint = torch.load(str(baseline / "best.pt"), map_location="cpu")
last = torch.load(str(root / "last.pt"), map_location="cpu")
assert configuration["authoritative_epoch_budget"] == (
    baseline_checkpoint["training_config"]["epochs"]
)
assert configuration["training_config"]["device"] == "cpu"
assert configuration["training_config"]["vq_init"] == "train-kmeans"
assert configuration["resume_strategy"] == "exact_epoch_boundary_resume"
assert last["training_config"]["device"] == "cpu"
assert last["data_state"]["initialization_mode"] == "train-kmeans"
assert partitions["test_partition_evaluated"] is False
assert decision["test_partition_evaluated"] is False
assert decision["decision"] in {
    "PASS_FOR_EVALUATION",
    "FAIL_ASSIGNMENT_COLLAPSE",
    "FAIL_NUMERICAL",
    "FAIL_PROVENANCE",
    "FAIL_OTHER",
}
assert decision["code_usage_trend"] is not None
if decision["decision"] == "PASS_FOR_EVALUATION":
    assert decision["completed_epoch"] == 50
    selected = decision["selected_best_checkpoint"]
    assert selected["provenance_valid"] is True
    assert selected["required_values_finite"] is True
    assert selected["validation_usage"]["active_code_count"] >= 2
    assert selected["validation_usage"]["codebook_perplexity"] >= 2.0
print("FULL_RETRAIN_ARTIFACT_VALIDATION=PASS")
print("training_decision={}".format(decision["decision"]))
if decision["decision"] != "PASS_FOR_EVALUATION":
    raise SystemExit(3)
PY

stage complete
