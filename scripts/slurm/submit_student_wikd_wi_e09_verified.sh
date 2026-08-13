#!/bin/bash -l
# Student-WIKD: WRMSSE-informed ground truth plus WI epoch-9 response KD.
#SBATCH --partition=gpu-a100
#SBATCH --job-name=student_wikd_wi9
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --qos=long
#SBATCH --output=logs/slurm/student_wikd_wi9_%j.out
#SBATCH --error=logs/slurm/student_wikd_wi9_%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_ROOT"
if [ ! -f "scripts/train_student.py" ]; then
    echo "Error: submit from the repository root; resolved directory: $REPO_ROOT"
    exit 1
fi

EXP_NAME="student_wikd_wi_e09_verified"
CACHE_EXP_NAME="wi_e09_43841d3d_verified"
TARGET_DIR="artifacts/soft_targets"
TEACHER_CKPT="outputs/teacher/tft64_wrmsse_informed/tft64-wrmsse-informed-epoch=epoch=09-val_loss=val_loss=1.466434.ckpt"
EXPECTED_SHA256="43841d3db586b7acad384f5068a23a609773ec1ad55e13bf004dc364f2d9bdf2"
OUTPUT_DIR="outputs/student/kd/${EXP_NAME}"
STORES=(CA_1 CA_2 CA_3 CA_4 TX_1 TX_2 TX_3 WI_1 WI_2 WI_3)

if [ -e "$OUTPUT_DIR" ]; then
    echo "Error: output directory already exists: $OUTPUT_DIR"
    echo "Refusing to overwrite or resume a retained-checkpoint experiment."
    exit 1
fi
if [ ! -s "$TEACHER_CKPT" ]; then
    echo "Error: teacher checkpoint is missing or empty: $TEACHER_CKPT"
    exit 1
fi

ACTUAL_SHA256="$(sha256sum "$TEACHER_CKPT" | awk '{print $1}')"
if [ "$ACTUAL_SHA256" != "$EXPECTED_SHA256" ]; then
    echo "Error: WI epoch-9 teacher checkpoint SHA-256 mismatch"
    echo "Expected: $EXPECTED_SHA256"
    echo "Actual:   $ACTUAL_SHA256"
    exit 1
fi

for STORE in "${STORES[@]}"; do
    TARGET_FILE="${TARGET_DIR}/${CACHE_EXP_NAME}_${STORE}.pt"
    PROVENANCE_FILE="${TARGET_DIR}/${CACHE_EXP_NAME}_${STORE}.json"
    if [ ! -s "$TARGET_FILE" ] || [ ! -s "$PROVENANCE_FILE" ]; then
        echo "Error: missing verified soft-target artifact for ${STORE}"
        exit 1
    fi
    if ! grep -Fq "\"checkpoint_sha256\": \"${EXPECTED_SHA256}\"" "$PROVENANCE_FILE"; then
        echo "Error: verified-cache provenance SHA mismatch for ${STORE}"
        exit 1
    fi
done

module load miniconda/24.1.2 2>/dev/null || module load miniconda/3 2>/dev/null || echo "[Info] Using default system conda"
source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate m5_env

CMD=(python scripts/train_student.py --env dicc --experiment student_wikd --exp-name "$EXP_NAME" --kd --soft-targets-path "$TARGET_DIR" --soft-targets-exp-name "$CACHE_EXP_NAME" --alpha 0.5 --supervised-loss wrmsse_informed)
echo "STUDENT-WIKD: alpha * WRMSSE-informed ground-truth loss + (1-alpha) * WRMSSE-informed teacher loss"
echo "alpha = 0.5; verified teacher SHA-256 = $ACTUAL_SHA256"
echo "checkpoint policy: top 5 by val_loss plus last.ckpt"
echo "command: ${CMD[*]}"
"${CMD[@]}"
