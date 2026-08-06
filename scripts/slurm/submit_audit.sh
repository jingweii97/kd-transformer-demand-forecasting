#!/bin/bash -l
#SBATCH --partition=gpu-a100
#SBATCH --job-name=m5_audit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --qos=normal
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -e

module load miniconda/24.1.2 2>/dev/null || module load miniconda/3 2>/dev/null || echo "[Info] Using default system conda"
source $(conda info --base)/etc/profile.d/conda.sh 2>/dev/null || true
conda activate m5_env 2>/dev/null || conda activate kd_env 2>/dev/null || true

mkdir -p logs
echo "Running Teacher-Student Comparability Audit"
python scripts/audit_comparability.py --env dicc --experiment full "$@"
