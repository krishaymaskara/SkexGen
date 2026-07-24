#!/bin/bash
#SBATCH --job-name=model-data-pytorch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/network/km6349/model_data_runs/model-data-pytorch-%j.out
#SBATCH --error=/scratch/network/km6349/model_data_runs/model-data-pytorch-%j.err

set -euo pipefail

cd /scratch/network/km6349/SkexGen

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPYCACHEPREFIX="/tmp/model-data-pycache-${SLURM_JOB_ID}"

echo "Host: $(hostname)"
echo "Commit: $(git rev-parse --short HEAD)"
echo "Started: $(date -Is)"

apptainer exec \
  --cleanenv \
  --env CUDA_VISIBLE_DEVICES="" \
  --env OMP_NUM_THREADS=1 \
  --env MKL_NUM_THREADS=1 \
  --env PYTHONPYCACHEPREFIX="$PYTHONPYCACHEPREFIX" \
  --bind /scratch/network/km6349:/scratch/network/km6349 \
  /scratch/network/km6349/skexgen.sif \
  /root/miniconda3/bin/python3.8 \
  prototype/model_data/adroit/validate_pytorch.py

echo "Finished: $(date -Is)"
