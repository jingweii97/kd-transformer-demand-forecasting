#!/bin/bash -l
# Held-out evaluation of a validation-selected WIS or WIKD checkpoint.
# Selection and metric logic remain in the existing Python evaluators.
#SBATCH --partition=gpu-a100
#SBATCH --job-name=eval_selected_student
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --qos=long
#SBATCH --output=logs/slurm/eval_selected_student_%j.out
#SBATCH --error=logs/slurm/eval_selected_student_%j.err

set -euo pipefail
if [ "$#" -ne 2 ]; then
    echo "Usage: sbatch $0 <student-label> <student-run-dir>"
    exit 2
fi

STUDENT_LABEL="$1"
RUN_DIR="$2"
REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_ROOT"
if [ ! -f "scripts/evaluate_models.py" ] || [ ! -f "scripts/resolve_selected_student_checkpoint.py" ]; then
    echo "Error: submit from repository root; resolved directory: $REPO_ROOT"
    exit 1
fi

MANIFEST="$RUN_DIR/common_validation_evaluation/selected_checkpoint.json"
BASELINE_STUDENT="outputs/student/no_kd/exp_full_phase1/best_student.ckpt"
WI_TEACHER="outputs/teacher/tft64_wrmsse_informed/tft64-wrmsse-informed-epoch=epoch=09-val_loss=val_loss=1.466434.ckpt"
EXPECTED_TEACHER_SHA256="43841d3db586b7acad384f5068a23a609773ec1ad55e13bf004dc364f2d9bdf2"
RUN_NAME="$(basename "$RUN_DIR")"
EXP_NAME="${RUN_NAME}_validation_selected_heldout"
OUTPUT_DIR="outputs/evaluation/${EXP_NAME}"

for CHECKPOINT in "$BASELINE_STUDENT" "$WI_TEACHER"; do
    if [ ! -s "$CHECKPOINT" ]; then
        echo "Error: required checkpoint is missing or empty: $CHECKPOINT"
        exit 1
    fi
done
if [ ! -s "$MANIFEST" ]; then
    echo "Error: successful validation selection manifest is required: $MANIFEST"
    exit 1
fi
if [ -e "$OUTPUT_DIR" ]; then
    echo "Error: held-out output already exists: $OUTPUT_DIR"
    exit 1
fi

ACTUAL_TEACHER_SHA256="$(sha256sum "$WI_TEACHER" | awk '{print $1}')"
if [ "$ACTUAL_TEACHER_SHA256" != "$EXPECTED_TEACHER_SHA256" ]; then
    echo "Error: WI epoch-9 teacher SHA-256 mismatch"
    exit 1
fi
mapfile -t SELECTED < <(python scripts/resolve_selected_student_checkpoint.py --manifest "$MANIFEST" --run-dir "$RUN_DIR")
SELECTED_CHECKPOINT="${SELECTED[0]}"
SELECTED_SHA256="${SELECTED[1]}"
SELECTED_WRMSSE="${SELECTED[2]}"

module load miniconda/24.1.2 2>/dev/null || module load miniconda/3 2>/dev/null || echo "[Info] Using default system conda"
source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate m5_env

CMD=(python scripts/evaluate_models.py --env dicc --experiment full --exp-name "$EXP_NAME" --teacher-checkpoint "$WI_TEACHER" --student-nokd-checkpoint "$BASELINE_STUDENT" --student-kd-checkpoint "$SELECTED_CHECKPOINT" --selected-student-label "$STUDENT_LABEL")
echo "HELD-OUT EVALUATION USING VALIDATION-SELECTED CHECKPOINT"
echo "student label = $STUDENT_LABEL"
echo "selected checkpoint = $SELECTED_CHECKPOINT"
echo "selected SHA-256 = $SELECTED_SHA256"
echo "selected validation WRMSSE = $SELECTED_WRMSSE"
echo "command: ${CMD[*]}"
"${CMD[@]}"
