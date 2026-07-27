#!/bin/bash
#SBATCH --job-name=flat-vq-kmeans-pilot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/network/km6349/flat-vq-kmeans-pilot-%j.out
#SBATCH --error=/scratch/network/km6349/flat-vq-kmeans-pilot-%j.err

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
BASELINE_CHECKPOINT="${BASELINE_RUN}/best.pt"
RUN_ROOT="/scratch/network/km6349/flat_baseline_runs"
CONTAINER="/scratch/network/km6349/skexgen.sif"
PYTHON="/root/miniconda3/bin/python3.8"
: "${REVIEWED_COMMIT:?set REVIEWED_COMMIT to the reviewed treatment commit}"
: "${SLURM_JOB_ID:?this workflow must run under Slurm}"

OUTPUT="${RUN_ROOT}/train-kmeans-pilot-${REVIEWED_COMMIT}-${SLURM_JOB_ID}"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPYCACHEPREFIX="/tmp/flat-vq-kmeans-pilot-${SLURM_JOB_ID}"

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
test -f "$BASELINE_CHECKPOINT"
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
  prototype.flat_baseline.tests.test_vq_initialization \
  prototype.flat_baseline.tests.test_vq_pilot
container_python -B -m unittest discover \
  -s prototype/flat_baseline/tests -t .
container_python -B -m unittest discover \
  -s prototype/model_data/tests -t .

stage codebook_initialization
container_python -B - "$CORPUS" "$BASELINE_CHECKPOINT" <<'PY'
from dataclasses import replace
import sys
import torch

from prototype.flat_baseline.config import FlatBaselineConfig
from prototype.flat_baseline.data import TrainingData, load_training_data
from prototype.flat_baseline.evaluate_length_conditioned import _json_document
from prototype.flat_baseline.model import FlatMixedVQModel
from prototype.flat_baseline.training import seed_everything
from prototype.flat_baseline.training_config import TrainingConfig
from prototype.flat_baseline.vq_initialization import initialize_train_codebook
from prototype.flat_baseline.vq_initialization import (
    initialization_report_sha256,
)

checkpoint = torch.load(sys.argv[2], map_location="cpu")
model_config = FlatBaselineConfig(**checkpoint["model_config"])
training_config = replace(
    TrainingConfig(**checkpoint["training_config"]),
    vq_init="train-kmeans",
    device="cpu",
)
assert training_config.seed == 2026
assert model_config.codebook_size == 32
data = load_training_data(sys.argv[1], "iid", "train", "validation")
subset = TrainingData(
    data.train_examples[:32],
    (),
    data.train_family_ids[:32],
    (),
)
reports = []
states = []
models = []
for _ in range(2):
    seed_everything(training_config.seed)
    model = FlatMixedVQModel(model_config)
    states.append({
        name: value.detach().clone()
        for name, value in model.state_dict().items()
        if not name.startswith("vq.")
    })
    reports.append(initialize_train_codebook(
        model, subset, training_config, torch.device("cpu"), torch
    ))
    models.append(model)
assert _json_document(reports[0]) == _json_document(reports[1])
assert reports[0]["report_sha256"] == initialization_report_sha256(
    reports[0]
)
assert reports[0]["validation_family_count_used"] == 0
assert reports[0]["test_family_count_used"] == 0
assert reports[0]["prequant_vector_count"] == (
    len(subset.train_examples) * model_config.latent_tokens
)
assert reports[0]["pseudo_count_total"] == model_config.codebook_size
assert reports[0]["effective_mass_ratio_to_normal"] == 1.0
for index, model in enumerate(models):
    for name, value in model.state_dict().items():
        if not name.startswith("vq."):
            assert torch.equal(value, states[index][name])
    assert torch.all(model.vq.ema_cluster_size > 0)
    assert torch.equal(
        model.vq.ema_weight,
        model.vq.ema_cluster_size.unsqueeze(1) * model.vq.embedding,
    )
for name, value in models[0].state_dict().items():
    assert torch.equal(value, models[1].state_dict()[name])
print("TRAIN_KMEANS_INITIALIZATION_SMOKE=PASS")
PY

stage pilot_training
container_python -B -m prototype.flat_baseline.vq_pilot \
  --corpus-dir "$CORPUS" \
  --baseline-checkpoint "$BASELINE_CHECKPOINT" \
  --output-dir "$OUTPUT" \
  --reviewed-commit "$REVIEWED_COMMIT" \
  --vq-init train-kmeans \
  --device cpu \
  --expected-train-count 544 \
  --expected-validation-count 68 \
  --expected-test-count 68

stage pilot_validation
container_python -B - "$OUTPUT" <<'PY'
import hashlib
import json
from pathlib import Path
import sys
import torch

root = Path(sys.argv[1])
decision = json.loads((root / "pilot_decision.json").read_text())
required = {"artifact_manifest.json", "pilot_decision.json"}
if decision["completed_epoch"] >= 1:
    required.update({
        "best.pt",
        "code_usage_steps.jsonl",
        "epoch_metrics.jsonl",
        "initialization_report.json",
        "last.pt",
        "partition_provenance.json",
        "run_configuration.json",
    })
assert required.issubset({path.name for path in root.iterdir()})
manifest = json.loads((root / "artifact_manifest.json").read_text())
assert set(manifest["artifact_names"]) == {
    path.name for path in root.iterdir()
    if path.name != "artifact_manifest.json"
}
for name, expected in manifest["artifact_sha256"].items():
    observed = hashlib.sha256((root / name).read_bytes()).hexdigest()
    assert observed == expected
if decision["completed_epoch"] >= 1:
    for name in ("best.pt", "last.pt"):
        checkpoint = torch.load(str(root / name), map_location="cpu")
        assert checkpoint["training_config"]["vq_init"] == "train-kmeans"
        assert checkpoint["training_config"]["seed"] == 2026
        assert checkpoint["training_config"]["device"] == "cpu"
        assert checkpoint["data_state"]["train_partition"] == "train"
        assert checkpoint["data_state"]["validation_partition"] == "validation"
        assert checkpoint["data_state"]["initialization_mode"] == "train-kmeans"
        assert checkpoint["data_state"]["initialization_seed"] == 2026
        assert checkpoint["data_state"]["initialization_algorithm"]
        assert checkpoint["data_state"]["initialization_pseudo_count_policy"]
        assert checkpoint["data_state"]["initialization_report_sha256"]
        assert checkpoint["data_state"]["reviewed_commit"]
    partitions = json.loads(
        (root / "partition_provenance.json").read_text()
    )
    configuration = json.loads(
        (root / "run_configuration.json").read_text()
    )
    assert configuration["training_config"]["device"] == "cpu"
    assert partitions["test_partition_evaluated"] is False
    assert partitions["test_family_count_used_for_initialization"] == 0
print("PILOT_ARTIFACT_VALIDATION=PASS")
PY

stage decision_report
container_python -B - "$OUTPUT" <<'PY'
import json
from pathlib import Path
import sys

decision = json.loads(
    (Path(sys.argv[1]) / "pilot_decision.json").read_text()
)
assert decision["decision"] in {
    "PASS_FOR_FULL_RETRAIN",
    "FAIL_ASSIGNMENT_COLLAPSE",
    "FAIL_NUMERICAL",
    "FAIL_PROVENANCE",
    "FAIL_OTHER",
}
assert decision["test_partition_evaluated"] is False
print("pilot_decision={}".format(decision["decision"]))
PY

stage complete
