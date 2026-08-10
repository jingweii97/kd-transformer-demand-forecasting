#!/bin/bash -l
# Provisional held-out evaluation of the fixed WRMSSE-informed epoch-9 KD branch.
# All evaluation logic remains in scripts/evaluate_models.py.
#SBATCH --partition=gpu-a100
#SBATCH --job-name=eval_kd_wi_e09_test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --qos=long
#SBATCH --output=logs/slurm/eval_kd_wi_e09_test_%j.out
#SBATCH --error=logs/slurm/eval_kd_wi_e09_test_%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_ROOT"
if [ ! -f "scripts/evaluate_models.py" ]; then
    echo "Error: submit this job from the repository root; resolved directory: $REPO_ROOT"
    exit 1
fi

EXP_NAME="wi_e09_43841d3d_heldout"
BASELINE_STUDENT="outputs/student/no_kd/exp_full_phase1/best_student.ckpt"
WI_TEACHER="outputs/teacher/tft64_wrmsse_informed/tft64-wrmsse-informed-epoch=epoch=09-val_loss=val_loss=1.466434.ckpt"
KD_STUDENT="outputs/student/kd/wi_e09_43841d3d/best_student.ckpt"
OUTPUT_DIR="outputs/evaluation/${EXP_NAME}"
EXPECTED_TEACHER_SHA256="43841d3db586b7acad384f5068a23a609773ec1ad55e13bf004dc364f2d9bdf2"

for CHECKPOINT in "$BASELINE_STUDENT" "$WI_TEACHER" "$KD_STUDENT"; do
    if [ ! -s "$CHECKPOINT" ]; then
        echo "Error: required checkpoint is missing or empty: $CHECKPOINT"
        exit 1
    fi
done
if [ -e "$OUTPUT_DIR" ]; then
    echo "Error: held-out evaluation output already exists: $OUTPUT_DIR"
    exit 1
fi

ACTUAL_TEACHER_SHA256="$(sha256sum "$WI_TEACHER" | awk '{print $1}')"
if [ "$ACTUAL_TEACHER_SHA256" != "$EXPECTED_TEACHER_SHA256" ]; then
    echo "Error: WRMSSE-informed epoch-9 teacher SHA-256 mismatch"
    echo "Expected: $EXPECTED_TEACHER_SHA256"
    echo "Actual:   $ACTUAL_TEACHER_SHA256"
    exit 1
fi

if [ -f "scripts/slurm/setup_hpc_storage.sh" ]; then
    bash scripts/slurm/setup_hpc_storage.sh || true
fi

module load miniconda/24.1.2 2>/dev/null || module load miniconda/3 2>/dev/null || echo "[Info] Using default system conda"
source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate m5_env 2>/dev/null || true

echo "PROVISIONAL HELD-OUT EVALUATION: WI epoch-9 KD branch"
echo "teacher checkpoint = $WI_TEACHER"
echo "teacher SHA-256 = $ACTUAL_TEACHER_SHA256"
echo "baseline student = $BASELINE_STUDENT"
echo "KD student = $KD_STUDENT"
echo "output directory = $OUTPUT_DIR"
echo "command: python scripts/evaluate_models.py --env dicc --experiment full --exp-name $EXP_NAME ..."

python scripts/evaluate_models.py \
    --env dicc \
    --experiment full \
    --exp-name "$EXP_NAME" \
    --teacher-checkpoint "$WI_TEACHER" \
    --student-nokd-checkpoint "$BASELINE_STUDENT" \
    --student-kd-checkpoint "$KD_STUDENT"
