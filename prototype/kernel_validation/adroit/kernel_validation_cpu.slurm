#!/bin/bash
#SBATCH --job-name=skexgen-kernel-validation
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=kernel-validation-%j.out
#SBATCH --error=kernel-validation-%j.err

set -eu

cd /scratch/network/km6349/SkexGen

apptainer exec \
  --cleanenv \
  --bind /scratch/network/km6349:/scratch/network/km6349 \
  /scratch/network/km6349/skexgen.sif \
  /root/miniconda3/bin/python3.8 \
  -m prototype.kernel_validation.run \
  --corpus-dir '<corpus>' \
  --output-dir '<report>'
