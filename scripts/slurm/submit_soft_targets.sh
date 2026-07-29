#!/bin/bash -l
#SBATCH --partition=gpu-a100
#SBATCH --job-name=m5_soft_targets
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --qos=long
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -e
EXP_NAME=${1:-"exp_full_phase1"}
TEACHER_CKPT=${2:-"outputs/teacher/$EXP_NAME/best_tft_teacher.ckpt"}
shift 2 2>/dev/null || shift 1 2>/dev/null || true

module load miniconda/24.1.2 2>/dev/null || module load miniconda/3 2>/dev/null || echo "[Info] Using default system conda"
source $(conda info --base)/etc/profile.d/conda.sh 2>/dev/null || true
conda activate m5_env 2>/dev/null || true

mkdir -p logs
echo "Generating Soft Targets for experiment: $EXP_NAME using checkpoint: $TEACHER_CKPT"
python scripts/generate_soft_targets.py --env dicc --experiment full --exp-name "$EXP_NAME" --checkpoint-path "$TEACHER_CKPT" "$@"
