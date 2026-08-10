#!/bin/bash -l
# Thin launcher: all evaluation logic remains in the existing Python evaluator.
#SBATCH --partition=gpu-a100
#SBATCH --job-name=eval_tft64_wi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --qos=long
#SBATCH --output=logs/slurm/eval_tft64_wi_%j.out
#SBATCH --error=logs/slurm/eval_tft64_wi_%j.err

set -e
cd "$(dirname "$0")/../.."

if [ -f "scripts/slurm/setup_hpc_storage.sh" ]; then
    bash scripts/slurm/setup_hpc_storage.sh || true
fi

module load miniconda/24.1.2 2>/dev/null || module load miniconda/3 2>/dev/null || echo "[Info] Using default system conda"
source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate m5_env 2>/dev/null || true
mkdir -p logs/slurm

echo "UNIQUE RETAINED WRMSSE-INFORMED CHECKPOINTS AND HASHES"
python scripts/evaluate_tft64_wrmsse_informed.py --env dicc --print-plan
echo "EVALUATOR COMMAND: python scripts/evaluate_tft64_wrmsse_informed.py --env dicc"
python scripts/evaluate_tft64_wrmsse_informed.py --env dicc
