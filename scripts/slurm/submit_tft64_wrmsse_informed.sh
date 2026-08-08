#!/bin/bash -l
# Final WRMSSE-informed TFT-64 teacher run. Non-loss settings match tft64_huber.
#SBATCH --partition=gpu-a100
#SBATCH --job-name=tft64_wi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --qos=long
#SBATCH --output=logs/slurm/tft64_wi_%j.out
#SBATCH --error=logs/slurm/tft64_wi_%j.err

set -e
EXP_NAME="tft64_wrmsse_informed"

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

echo "FRESH TFT-64 WRMSSE-INFORMED INITIALIZATION"
echo "RESUME CHECKPOINT: NONE"
echo "INITIAL EPOCH: 0"
echo "INITIAL GLOBAL STEP: 0"

python scripts/train_teacher.py --env dicc --experiment tft64_wrmsse_informed --exp-name "$EXP_NAME" "$@"

python scripts/verify_tft_run.py \
    --exp-name "$EXP_NAME" \
    --hidden-size 64 \
    --hidden-continuous-size 8 \
    --lstm-layers 1 \
    --attention-heads 4
