#!/bin/bash -l
# Read-only verification of the already generated WI epoch-9 soft-target cache.
#SBATCH --partition=gpu-a100
#SBATCH --job-name=wi_e09_cache_audit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --qos=long
#SBATCH --output=logs/slurm/wi_e09_cache_audit_%j.out
#SBATCH --error=logs/slurm/wi_e09_cache_audit_%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_ROOT"
if [ ! -f "scripts/verify_soft_target_cache.py" ]; then
    echo "Error: submit from the repository root; resolved directory: $REPO_ROOT"
    exit 1
fi

EXP_NAME="wi_e09_43841d3d_verified"
TEACHER_CKPT="outputs/teacher/tft64_wrmsse_informed/tft64-wrmsse-informed-epoch=epoch=09-val_loss=val_loss=1.466434.ckpt"
EXPECTED_SHA256="43841d3db586b7acad384f5068a23a609773ec1ad55e13bf004dc364f2d9bdf2"

ACTUAL_SHA256="$(sha256sum "$TEACHER_CKPT" | awk '{print $1}')"
if [ "$ACTUAL_SHA256" != "$EXPECTED_SHA256" ]; then
    echo "Error: epoch-9 teacher checkpoint SHA-256 mismatch"
    exit 1
fi

module load miniconda/24.1.2 2>/dev/null || module load miniconda/3 2>/dev/null || echo "[Info] Using default system conda"
source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate m5_env

CMD=(python scripts/verify_soft_target_cache.py --env dicc --experiment full --exp-name "$EXP_NAME" --checkpoint-path "$TEACHER_CKPT" --samples-per-store 3)
echo "verified teacher SHA-256 = $ACTUAL_SHA256"
echo "verification command: ${CMD[*]}"
"${CMD[@]}"
