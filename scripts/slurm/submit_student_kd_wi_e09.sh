#!/bin/bash -l
# Provisional vanilla response-KD run using the retained WRMSSE-informed epoch-9 teacher.
# This launcher contains no training logic; scripts/train_student.py performs the training.
#SBATCH --partition=gpu-a100
#SBATCH --job-name=student_kd_wi_e09
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --qos=long
#SBATCH --output=logs/slurm/student_kd_wi_e09_%j.out
#SBATCH --error=logs/slurm/student_kd_wi_e09_%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_ROOT"
if [ ! -f "scripts/train_student.py" ]; then
    echo "Error: submit this job from the repository root; resolved directory: $REPO_ROOT"
    exit 1
fi

EXP_NAME="wi_e09_43841d3d"
TARGET_DIR="artifacts/soft_targets"
TEACHER_CKPT="outputs/teacher/tft64_wrmsse_informed/tft64-wrmsse-informed-epoch=epoch=09-val_loss=val_loss=1.466434.ckpt"
EXPECTED_SHA256="43841d3db586b7acad384f5068a23a609773ec1ad55e13bf004dc364f2d9bdf2"
OUTPUT_DIR="outputs/student/kd/${EXP_NAME}"
STORES=(CA_1 CA_2 CA_3 CA_4 TX_1 TX_2 TX_3 WI_1 WI_2 WI_3)

if [ -e "$OUTPUT_DIR" ]; then
    echo "Error: output directory already exists: $OUTPUT_DIR"
    echo "Refusing to overwrite or resume this provisional KD run."
    exit 1
fi

if [ ! -f "$TEACHER_CKPT" ]; then
    echo "Error: expected epoch-9 teacher checkpoint is missing: $TEACHER_CKPT"
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
    TARGET_FILE="${TARGET_DIR}/${EXP_NAME}_${STORE}.pt"
    PROVENANCE_FILE="${TARGET_DIR}/${EXP_NAME}_${STORE}.json"
    if [ ! -s "$TARGET_FILE" ] || [ ! -s "$PROVENANCE_FILE" ]; then
        echo "Error: missing or empty soft target/provenance file for ${STORE}"
        echo "Expected: $TARGET_FILE"
        echo "Expected: $PROVENANCE_FILE"
        exit 1
    fi
    if ! grep -Fq '"checkpoint_path": "/home/user/yeoh97/repo/' "$PROVENANCE_FILE" \
        || ! grep -Fq 'tft64-wrmsse-informed-epoch=epoch=09-val_loss=val_loss=1.466434.ckpt' "$PROVENANCE_FILE"; then
        echo "Error: provenance for ${STORE} does not identify the retained WRMSSE-informed epoch-9 checkpoint"
        exit 1
    fi
done

if [ -f "scripts/slurm/setup_hpc_storage.sh" ]; then
    bash scripts/slurm/setup_hpc_storage.sh || true
fi

module load miniconda/24.1.2 2>/dev/null || module load miniconda/3 2>/dev/null || echo "[Info] Using default system conda"
source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate m5_env 2>/dev/null || true

echo "PROVISIONAL VANILLA RESPONSE-KD RUN"
echo "teacher checkpoint = $TEACHER_CKPT"
echo "teacher SHA-256 = $ACTUAL_SHA256"
echo "soft target prefix = ${TARGET_DIR}/${EXP_NAME}_<STORE>.pt"
echo "student output = $OUTPUT_DIR"
echo "command: python scripts/train_student.py --env dicc --experiment full --exp-name $EXP_NAME --kd --soft-targets-path $TARGET_DIR --alpha 0.5"

python scripts/train_student.py \
    --env dicc \
    --experiment full \
    --exp-name "$EXP_NAME" \
    --kd \
    --soft-targets-path "$TARGET_DIR" \
    --alpha 0.5
