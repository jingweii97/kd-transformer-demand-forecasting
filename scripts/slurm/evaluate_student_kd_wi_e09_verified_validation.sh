#!/bin/bash -l
# Thin validation-only launcher for the corrected WI epoch-9 KD student.
# Metric and prediction logic remain exclusively in scripts/audit_comparability.py.
# best_student.ckpt and last.ckpt are retained aliases and must be byte-identical.
#SBATCH --partition=gpu-a100
#SBATCH --job-name=eval_kd_wi_e09_v
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --qos=long
#SBATCH --output=logs/slurm/eval_kd_wi_e09_verified_%j.out
#SBATCH --error=logs/slurm/eval_kd_wi_e09_verified_%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_ROOT"
if [ ! -f "scripts/audit_comparability.py" ]; then
    echo "Error: submit this job from the repository root; resolved directory: $REPO_ROOT"
    exit 1
fi

EXP_DIR="outputs/student/kd/wi_e09_43841d3d_verified"
BEST_CHECKPOINT="$EXP_DIR/best_student.ckpt"
LAST_CHECKPOINT="$EXP_DIR/last.ckpt"
OUT_DIR="$EXP_DIR/common_validation_evaluation"

for CHECKPOINT in "$BEST_CHECKPOINT" "$LAST_CHECKPOINT"; do
    if [ ! -s "$CHECKPOINT" ]; then
        echo "Error: retained KD checkpoint is missing or empty: $CHECKPOINT"
        exit 1
    fi
done
if [ -e "$OUT_DIR" ]; then
    echo "Error: validation output directory already exists: $OUT_DIR"
    exit 1
fi

BEST_SHA256="$(sha256sum "$BEST_CHECKPOINT" | awk '{print $1}')"
LAST_SHA256="$(sha256sum "$LAST_CHECKPOINT" | awk '{print $1}')"
if [ "$BEST_SHA256" != "$LAST_SHA256" ]; then
    echo "Error: retained KD checkpoint aliases have inconsistent SHA-256 values"
    echo "best_student.ckpt: $BEST_SHA256"
    echo "last.ckpt:         $LAST_SHA256"
    exit 1
fi

if [ -f "scripts/slurm/setup_hpc_storage.sh" ]; then
    bash scripts/slurm/setup_hpc_storage.sh || true
fi

module load miniconda/24.1.2 2>/dev/null || module load miniconda/3 2>/dev/null || echo "[Info] Using default system conda"
source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate m5_env 2>/dev/null || true

echo "VALIDATION-ONLY COMMON KD EVALUATION"
echo "Validation boundary: days 1526-1553"
echo "No ID/OOD/held-out test data will be used."
echo "Unique checkpoint selected: $BEST_CHECKPOINT"
echo "Duplicate retained alias: $LAST_CHECKPOINT"
echo "SHA-256: $BEST_SHA256"
echo "Command: python scripts/audit_comparability.py --env dicc --experiment full --output-dir $OUT_DIR --model student KD_student_WI_epoch_9_verified $BEST_CHECKPOINT"

python scripts/audit_comparability.py \
    --env dicc \
    --experiment full \
    --output-dir "$OUT_DIR" \
    --model student "KD student (WI epoch 9, verified cache)" "$BEST_CHECKPOINT"
