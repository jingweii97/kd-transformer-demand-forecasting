#!/bin/bash -l
#SBATCH --partition=cpu-epyc-genoa
#SBATCH --job-name=m5_prepare_dataset
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --qos=normal
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -e

module load miniconda/24.1.2 2>/dev/null || module load miniconda/3 2>/dev/null || echo "[Info] Using default system conda"
source $(conda info --base)/etc/profile.d/conda.sh 2>/dev/null || true
conda activate m5_env 2>/dev/null || true

mkdir -p logs
echo "Preprocessing dataset on CPU compute node..."
python scripts/prepare_dataset.py --env dicc "$@"
echo "Dataset preparation completed successfully."
