#!/bin/bash -l
#SBATCH --partition=gpu-a100
#SBATCH --job-name=eval_ckpt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --qos=long
#SBATCH --output=logs/slurm/eval_ckpt_%j.out
#SBATCH --error=logs/slurm/eval_ckpt_%j.err

set -e

# Execute HPC storage setup helper if on cluster
if [ -f "scripts/slurm/setup_hpc_storage.sh" ]; then
    bash scripts/slurm/setup_hpc_storage.sh || true
fi

module load miniconda/24.1.2 2>/dev/null || module load miniconda/3 2>/dev/null || echo "[Info] Using default system conda"
source $(conda info --base)/etc/profile.d/conda.sh 2>/dev/null || true
conda activate m5_env 2>/dev/null || true

mkdir -p logs/slurm

echo "STARTING VALIDATION CHECKPOINT EVALUATION..."

python scripts/evaluate_checkpoints.py --env dicc

echo "EVALUATION COMPLETE!"
