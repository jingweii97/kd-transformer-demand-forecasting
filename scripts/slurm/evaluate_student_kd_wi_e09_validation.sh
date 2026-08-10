#!/bin/bash -l
# Thin validation-only launcher for the provisional WRMSSE-informed epoch-9 KD student.
# It performs fresh inference only for the new KD student. The existing authoritative
# supervised-student, Huber, and WRMSSE-informed reference rows are reused afterwards.
#SBATCH --partition=gpu-a100
#SBATCH --job-name=eval_kd_wi_e09
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --qos=long
#SBATCH --output=logs/slurm/eval_kd_wi_e09_%j.out
#SBATCH --error=logs/slurm/eval_kd_wi_e09_%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_ROOT"
if [ ! -f "scripts/audit_comparability.py" ]; then
    echo "Error: submit this job from the repository root; resolved directory: $REPO_ROOT"
    exit 1
fi

KD_STUDENT="outputs/student/kd/wi_e09_43841d3d/best_student.ckpt"
OUT_DIR="outputs/student/kd/wi_e09_43841d3d/common_validation_evaluation"

if [ ! -s "$KD_STUDENT" ]; then
    echo "Error: required KD student checkpoint is missing or empty: $KD_STUDENT"
    exit 1
fi
if [ -e "$OUT_DIR" ]; then
    echo "Error: validation output directory already exists: $OUT_DIR"
    exit 1
fi

if [ -f "scripts/slurm/setup_hpc_storage.sh" ]; then
    bash scripts/slurm/setup_hpc_storage.sh || true
fi

module load miniconda/24.1.2 2>/dev/null || module load miniconda/3 2>/dev/null || echo "[Info] Using default system conda"
source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate m5_env 2>/dev/null || true

echo "VALIDATION-ONLY COMMON KD COMPARISON"
echo "Validation boundary: days 1526-1553"
echo "No ID/OOD/held-out test data will be used."
echo "Fresh inference: $KD_STUDENT"
echo "Archived reference rows: outputs/teacher/tft64_wrmsse_informed/common_validation_evaluation.csv"
echo "Command: python scripts/audit_comparability.py --env dicc --experiment full ..."

python scripts/audit_comparability.py \
    --env dicc \
    --experiment full \
    --output-dir "$OUT_DIR" \
    --model student "KD student (WI epoch 9, alpha 0.5)" "$KD_STUDENT"
