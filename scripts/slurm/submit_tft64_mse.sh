#!/bin/bash -l
# Plain-MSE TFT-64 control run. Non-loss settings match tft64_huber.
#SBATCH --partition=gpu-a100
#SBATCH --job-name=tft64_mse
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --qos=long
#SBATCH --output=logs/slurm/tft64_mse_%j.out
#SBATCH --error=logs/slurm/tft64_mse_%j.err

set -e
cd "$(dirname "$0")/../.."
EXP_NAME="tft64_mse"

if [ -f "scripts/slurm/setup_hpc_storage.sh" ]; then
    bash scripts/slurm/setup_hpc_storage.sh || true
fi

if [ -d "outputs/teacher/$EXP_NAME" ]; then
    echo "Error: Output directory outputs/teacher/$EXP_NAME already exists. Aborting to prevent overwrite."
    exit 1
fi

module load miniconda/24.1.2 2>/dev/null || module load miniconda/3 2>/dev/null || echo "[Info] Using default system conda"
source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate m5_env 2>/dev/null || true

mkdir -p logs/slurm

echo "FRESH TFT-64 MSE INITIALIZATION"
echo "experiment = tft64_mse"
echo "hidden_size = 64"
echo "hidden_continuous_size = 8"
echo "lstm_layers = 1"
echo "attention_head_size = 4"
echo "dropout = 0.1"
echo "loss = mse"
echo "output_size = 1"
echo "gradient_clip_val = 0.1"
echo "seed = 42"
echo "dataset weight = None"
echo "resume checkpoint = none"

python scripts/audit_tft64_mse.py --env dicc --experiment "$EXP_NAME"
python scripts/train_teacher.py --env dicc --experiment "$EXP_NAME" --exp-name "$EXP_NAME" "$@"
python scripts/verify_tft_run.py \
    --exp-name "$EXP_NAME" \
    --hidden-size 64 \
    --hidden-continuous-size 8 \
    --lstm-layers 1 \
    --attention-heads 4
python scripts/evaluate_checkpoints.py \
    --env dicc \
    --checkpoint-glob "outputs/teacher/$EXP_NAME/*.ckpt"
