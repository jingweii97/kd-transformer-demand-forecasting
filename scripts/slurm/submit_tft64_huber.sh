#!/bin/bash -l
#SBATCH --partition=gpu-a100
#SBATCH --job-name=tft64_hub
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --qos=long
#SBATCH --output=logs/slurm/tft64_hub_%j.out
#SBATCH --error=logs/slurm/tft64_hub_%j.err

set -e
EXP_NAME="tft64_huber"

# Execute HPC storage setup helper if on cluster
if [ -f "scripts/slurm/setup_hpc_storage.sh" ]; then
    bash scripts/slurm/setup_hpc_storage.sh || true
fi

if [ -d "outputs/teacher/$EXP_NAME" ]; then
    echo "Error: Output directory outputs/teacher/$EXP_NAME already exists. Aborting to prevent overwrite."
    exit 1
fi

module load miniconda/24.1.2 2>/dev/null || module load miniconda/3 2>/dev/null || echo "[Info] Using default system conda"
source $(conda info --base)/etc/profile.d/conda.sh 2>/dev/null || true
conda activate m5_env 2>/dev/null || true

mkdir -p logs/slurm

echo "FRESH TFT-64 HUBER INITIALIZATION"
echo "RESUME CHECKPOINT: NONE"
echo "INITIAL EPOCH: 0"
echo "INITIAL GLOBAL STEP: 0"

# Train command
python scripts/train_teacher.py --env dicc --experiment tft64_huber --exp-name "$EXP_NAME" "$@"

# Run verification
python scripts/verify_tft_run.py \
    --exp-name "$EXP_NAME" \
    --hidden-size 64 \
    --hidden-continuous-size 8 \
    --lstm-layers 1 \
    --attention-heads 4
