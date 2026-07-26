#!/bin/bash -l
#SBATCH --partition=gpu-a100
#SBATCH --job-name=m5_hpc_pipeline
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --qos=long
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Exit on error
set -e

EXP_NAME=${1:-"exp_full_phase1"}

echo "=================================================="
echo "Starting M5 KD Pipeline on UM DICC HPC Cluster"
echo "Job ID       : $SLURM_JOB_ID"
echo "Node         : $SLURMD_NODENAME"
echo "Partition    : $SLURM_JOB_PARTITION"
echo "Experiment   : $EXP_NAME"
echo "Start Time   : $(date)"
echo "=================================================="

# 0. Execute HPC Storage Symlink Setup
if [ -f "scripts/slurm/setup_hpc_storage.sh" ]; then
    bash scripts/slurm/setup_hpc_storage.sh || true
fi

# 1. Load Miniconda Module and Activate Conda Environment
module load miniconda/24.1.2 2>/dev/null || module load miniconda/3 2>/dev/null || echo "[Info] Using default system conda"
source $(conda info --base)/etc/profile.d/conda.sh 2>/dev/null || true
conda activate m5_env 2>/dev/null || true

# Ensure logs directory exists
mkdir -p logs

# Step 1: Preprocess dataset
echo "=== Step 1: Preprocessing Dataset ==="
python scripts/prepare_dataset.py --env dicc

# Step 2: Train TFT Teacher Model
echo "=== Step 2: Training TFT Teacher Model ==="
python scripts/train_teacher.py --env dicc --experiment full --exp-name "$EXP_NAME"

# Step 3: Generate Teacher Soft Targets
echo "=== Step 3: Generating Soft Targets ==="
TEACHER_CKPT="outputs/teacher/$EXP_NAME/best_tft_teacher.ckpt"
python scripts/generate_soft_targets.py --env dicc --experiment full --exp-name "$EXP_NAME" --checkpoint-path "$TEACHER_CKPT"

# Step 4: Train Student Model (Without KD Baseline)
echo "=== Step 4: Training Student Model (No KD) ==="
python scripts/train_student.py --env dicc --experiment full --exp-name "$EXP_NAME" --no-kd

# Step 5: Train Student Model (With KD)
echo "=== Step 5: Training Student Model (With KD) ==="
python scripts/train_student.py --env dicc --experiment full --exp-name "$EXP_NAME" --kd

# Step 6: Evaluate All Models on ID & OOD Splits
echo "=== Step 6: Evaluating Models ==="
python scripts/evaluate_models.py --env dicc --experiment full --exp-name "$EXP_NAME"

echo "=================================================="
echo "Pipeline completed successfully at $(date)"
echo "=================================================="
