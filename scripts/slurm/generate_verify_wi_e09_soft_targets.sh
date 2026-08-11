#!/bin/bash -l
# Generate a new, versioned WI epoch-9 cache and verify it before any KD run.
#SBATCH --partition=gpu-a100
#SBATCH --job-name=wi_e09_cache_verify
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --qos=long
#SBATCH --output=logs/slurm/wi_e09_cache_verify_%j.out
#SBATCH --error=logs/slurm/wi_e09_cache_verify_%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_ROOT"
if [ ! -f "scripts/generate_soft_targets.py" ]; then
    echo "Error: submit from the repository root; resolved directory: $REPO_ROOT"
    exit 1
fi

EXP_NAME="wi_e09_43841d3d_verified"
TARGET_DIR="artifacts/soft_targets"
TEACHER_CKPT="outputs/teacher/tft64_wrmsse_informed/tft64-wrmsse-informed-epoch=epoch=09-val_loss=val_loss=1.466434.ckpt"
EXPECTED_SHA256="43841d3db586b7acad384f5068a23a609773ec1ad55e13bf004dc364f2d9bdf2"
STORES=(CA_1 CA_2 CA_3 CA_4 TX_1 TX_2 TX_3 WI_1 WI_2 WI_3)

if [ ! -f "$TEACHER_CKPT" ]; then
    echo "Error: teacher checkpoint is missing: $TEACHER_CKPT"
    exit 1
fi

ACTUAL_SHA256="$(sha256sum "$TEACHER_CKPT" | awk '{print $1}')"
if [ "$ACTUAL_SHA256" != "$EXPECTED_SHA256" ]; then
    echo "Error: epoch-9 checkpoint SHA-256 mismatch"
    echo "Expected: $EXPECTED_SHA256"
    echo "Actual:   $ACTUAL_SHA256"
    exit 1
fi

for STORE in "${STORES[@]}"; do
    if [ -e "${TARGET_DIR}/${EXP_NAME}_${STORE}.pt" ] || [ -e "${TARGET_DIR}/${EXP_NAME}_${STORE}.json" ]; then
        echo "Error: refusing to overwrite an existing verified-cache artifact for ${STORE}"
        exit 1
    fi
done

module load miniconda/24.1.2 2>/dev/null || module load miniconda/3 2>/dev/null || echo "[Info] Using default system conda"
source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate m5_env

GENERATE_CMD=(python scripts/generate_soft_targets.py --env dicc --experiment full --exp-name "$EXP_NAME" --checkpoint-path "$TEACHER_CKPT")
VERIFY_CMD=(python scripts/verify_soft_target_cache.py --env dicc --experiment full --exp-name "$EXP_NAME" --checkpoint-path "$TEACHER_CKPT" --samples-per-store 3)

echo "verified teacher SHA-256 = $ACTUAL_SHA256"
echo "generation command: ${GENERATE_CMD[*]}"
"${GENERATE_CMD[@]}"

echo "verification command: ${VERIFY_CMD[*]}"
"${VERIFY_CMD[@]}"

echo "WI EPOCH-9 SOFT TARGETS GENERATED AND VERIFIED"
